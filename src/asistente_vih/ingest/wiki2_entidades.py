"""Memoria de entidades (diccionario canonico) del corpus 2Wiki.

Primera pieza del pipeline de evaluacion: construye, UNA vez y solo desde el
corpus (nunca desde el oro del benchmark), una entrada por entidad canonica con
sus alias, su descripcion y los pasajes donde aparece. Analogo al linker
terminologico del caso VIH, pero derivado del propio corpus (generaliza a
cualquier dominio; Wikidata queda como verificacion opcional).

Pasos:
  1. Ficha por MENCION (entidad x pasaje): nombre + descriptores (frases de
     las aserciones de ese pasaje en las que participa) -> embedding. La
     ficha-titulo (sujeto del articulo) HEREDA las fechas del sujeto si no
     tiene propias (evita fundir homonimos: 'Christopher Newton (criminal)').
  2. Candidatos a misma entidad: coseno entre fichas >= UMBRAL_CAND y nombres
     compatibles (token significativo comun o coseno de nombre alto) y sin
     contradiccion de fechas de nacimiento/muerte.
  3. Adjudicacion: nombre base identico -> fusion automatica, SALVO que una de
     las formas lleve desambiguador entre parentesis (-> juez); banda ambigua
     -> LLM decide viendo las dos fichas (lotes, cache SQLite). Titulo->sujeto:
     automatica solo cuando el titulo es mencion SINTETICA (no extraida) y el
     sujeto es unico; si el titulo ya es mencion propia o hay empate -> juez.
     Union-find -> grupos. Cada union queda REGISTRADA con su origen.
  4. Entrada canonica por grupo: nombre (preferencia: titulo de pasaje),
     alias, descripcion fusionada, pasajes; embedding de la ficha fusionada.

Salida (artifacts/wiki2/), con sufijo de version `ver` (p.ej. "_v4"; vacio =
la version evaluada en el documento, v3):
  entidades_<split><ver>.json           lista de entradas canonicas
  entidades_mapa_<split><ver>.json      "titulo||forma_plegada" -> id canonico
  entidades_uniones_<split><ver>.jsonl  procedencia: {a, b, origen} por union
  emb_fichas_<split><ver>.npz           ids canonicos + matriz (ficha fusionada)

    python -m asistente_vih.ingest.wiki2_entidades --split sondeo --ver _v4
    python -m asistente_vih.ingest.wiki2_entidades --split benchmark --legado --ver _legado --sin-embeddings
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict

import numpy as np
from openai import OpenAI

from ..config import ARTIFACTS_DIR

WIKI2_DIR = ARTIFACTS_DIR / "wiki2"
MODELO_EMB = "text-embedding-3-small"
MODELO_LLM = "gpt-4o-mini"
UMBRAL_CAND = 0.75      # coseno de ficha para ser candidato
UMBRAL_AUTO = 0.92      # coseno de ficha para fusion sin LLM
UMBRAL_NOMBRE = 0.80    # coseno de nombre que exime del token comun
VECINOS = 15            # vecinos por ficha considerados
LOTE_LLM = 8            # pares por llamada de adjudicacion
_GENERICOS = {"the", "and", "of", "de", "la", "el", "von", "van", "der", "di",
              "film", "song", "duke", "king", "prince", "princess", "count"}

# Origenes de union (procedencia, E2): quien decidio que dos fichas son la
# misma entidad.
ORIGENES = ("auto_nombre", "auto_casi", "titulo_sintetico",
            "juez_vecinos", "juez_titulo_mencion", "juez_titulo_empate")


def _plegar(s: str) -> str:
    return " ".join(s.lower().split())


def _tokens_sig(nombre: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", " ", nombre.lower()))
    return {t for t in toks if len(t) > 2 and t not in _GENERICOS}


def _desambiguador(forma: str) -> str:
    """Texto entre parentesis de un titulo/forma ('(criminal)', '(1958 film)')."""
    m = re.search(r"\(([^)]*)\)", forma)
    return _plegar(m.group(1)) if m else ""


def _anios(aserciones: list[dict], entidad_pleg: str, rel: str) -> set[str]:
    out = set()
    for a in aserciones:
        if a["relacion_base"] == rel and _plegar(a["sujeto"]) == entidad_pleg:
            m = re.search(r"\b(1\d{3}|20\d{2})\b", a["objeto"])
            if m:
                out.add(m.group(1))
    return out


def _embeber(client: OpenAI, textos: list[str]) -> np.ndarray:
    mat = np.zeros((len(textos), 1536), dtype=np.float32)
    for i in range(0, len(textos), 512):
        r = client.embeddings.create(input=textos[i:i + 512], model=MODELO_EMB)
        for j, d in enumerate(r.data):
            mat[i + j] = d.embedding
        if (i // 512) % 10 == 0:
            print(f"  emb {min(i + 512, len(textos))}/{len(textos)}")
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------ paso 1: fichas

def _mejor_sujeto(fichas: list[dict], idx: list[int], t: int) -> tuple[int | None, bool]:
    """Candidato a SUJETO del articulo para la ficha-titulo t: la mencion
    no-titulo del pasaje con mayor solape de tokens significativos (>=1),
    desempatando por la que es sujeto de mas aserciones (longitud de desc).
    Devuelve (mejor, empate): empate = otra mencion con el mismo solape."""
    toks_t = _tokens_sig(fichas[t]["forma"])
    puntuados = []
    for i in idx:
        if i == t or fichas[i]["es_titulo"]:
            continue
        sol = len(toks_t & _tokens_sig(fichas[i]["forma"]))
        if sol >= 1 and _fechas_compatibles(fichas[t], fichas[i]):
            puntuados.append(((sol, len(fichas[i]["desc"])), -i))   # -i: a igualdad, la primera
    if not puntuados:
        return None, False
    puntuados.sort(reverse=True)
    empate = len(puntuados) > 1 and puntuados[1][0][0] == puntuados[0][0][0]
    return -puntuados[0][1], empate


def fichas_por_mencion(docs: list[dict], heredar_fechas: bool = True) -> list[dict]:
    """Una ficha por (pasaje, entidad). Ademas, el TITULO del pasaje se registra
    como mencion propia (alias del sujeto del articulo): un articulo de
    enciclopedia nombra a su sujeto en el titulo, y los demas pasajes suelen
    referirse a el por esa forma ('Charles Band') aunque el texto use la forma
    legal ('Charles Robert Band'). Se une al sujeto en `unir_titulos`.

    heredar_fechas: la ficha-titulo sin fechas propias toma las del sujeto
    candidato unico del pasaje (v4). Con False se reproduce la v3."""
    fichas = []
    for d in docs:
        titulo = d["chunk_id"]
        base = re.sub(r"\(.*?\)", " ", titulo).strip()
        entidades = list(d["entidades"])
        formas_pleg = {_plegar(e) for e in entidades}
        sintetica = _plegar(base) not in formas_pleg and _plegar(titulo) not in formas_pleg
        if sintetica:
            entidades.append(base)                 # mencion sintetica = titulo
        inicio = len(fichas)
        for forma in entidades:
            pleg = _plegar(forma)
            frases = [a["descripcion_relacion"] for a in d["aserciones"]
                      if pleg in _plegar(a["descripcion_relacion"]) or
                      pleg in (_plegar(a["sujeto"]), _plegar(a["objeto"]))]
            desc = " ".join(dict.fromkeys(f for f in frases if f))[:400]
            es_titulo = _plegar(titulo) == pleg or _plegar(base) == pleg
            fichas.append({
                "clave": f"{titulo}||{pleg}", "forma": forma, "pleg": pleg,
                "pasaje": titulo, "es_titulo": es_titulo,
                "sintetica": bool(sintetica and es_titulo),
                "desamb": _desambiguador(titulo) if es_titulo else _desambiguador(forma),
                "desc": desc,
                "nac": _anios(d["aserciones"], pleg, "DATE_OF_BIRTH"),
                "mue": _anios(d["aserciones"], pleg, "DATE_OF_DEATH"),
                "fechas_heredadas": False,
            })
        if heredar_fechas:
            idx = list(range(inicio, len(fichas)))
            tit = [i for i in idx if fichas[i]["es_titulo"]]
            if tit and not fichas[tit[0]]["nac"] and not fichas[tit[0]]["mue"]:
                mejor, empate = _mejor_sujeto(fichas, idx, tit[0])
                if mejor is not None and not empate and \
                        (fichas[mejor]["nac"] or fichas[mejor]["mue"]):
                    fichas[tit[0]]["nac"] = set(fichas[mejor]["nac"])
                    fichas[tit[0]]["mue"] = set(fichas[mejor]["mue"])
                    fichas[tit[0]]["fechas_heredadas"] = True
    return fichas


def _texto_ficha(f: dict) -> str:
    return f"{f['forma']} — {f['desc']}" if f["desc"] else f["forma"]


def unir_titulos(fichas: list[dict]) -> list[tuple[int, int, str]]:
    """Propone unir la mencion-titulo de cada pasaje con su sujeto candidato.
    Devuelve (titulo, sujeto, caso):
      titulo_sintetico         titulo no extraido, sujeto unico -> union directa
      titulo_sintetico_empate  titulo no extraido, varios sujetos -> juez
      titulo_mencion           el titulo YA es mencion propia -> juez (la v3 lo
                               unia siempre y fundia parientes/secuelas)."""
    por_pasaje: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(fichas):
        por_pasaje[f["pasaje"]].append(i)
    props = []
    for pasaje, idx in por_pasaje.items():
        tit = [i for i in idx if fichas[i]["es_titulo"]]
        if not tit:
            continue
        t = tit[0]
        mejor, empate = _mejor_sujeto(fichas, idx, t)
        if mejor is None:
            continue
        if fichas[t]["sintetica"]:
            caso = "titulo_sintetico_empate" if empate else "titulo_sintetico"
        else:
            caso = "titulo_mencion"
        props.append((t, mejor, caso))
    return props


# ------------------------------------------------------------ paso 2-3: grupos

_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b|\b([ivx]+)\b(?=[ ,.)]|$)")


def _ordinales(nombre: str) -> set[str]:
    """Numerales de titulos nobiliarios/regnales ('4th', 'II'): si difieren,
    son personas distintas (padre/hijo, sucesores)."""
    return {a or b for a, b in _ORDINAL.findall(nombre.lower())}


def _nombre_base(forma: str) -> str:
    return _plegar(re.sub(r"\(.*?\)", " ", forma))


def _fechas_compatibles(a: dict, b: dict) -> bool:
    if a["nac"] and b["nac"] and a["nac"].isdisjoint(b["nac"]):
        return False
    if a["mue"] and b["mue"] and a["mue"].isdisjoint(b["mue"]):
        return False
    oa, ob = _ordinales(a["forma"]), _ordinales(b["forma"])
    return not (oa and ob and oa != ob)


def _clasificar_par(a: dict, b: dict, cos_ficha: float, cos_nombre: float,
                    legado: bool = False) -> str | None:
    """'auto_nombre' / 'auto_casi' (fusion directa), 'llm' (adjudicar) o None
    (descartar). Canal principal = NOMBRE; la ficha aporta contexto."""
    if not _fechas_compatibles(a, b):
        return None
    da, db = a.get("desamb", ""), b.get("desamb", "")
    desamb_distinto = (not legado) and bool(da or db) and da != db
    if _nombre_base(a["forma"]) == _nombre_base(b["forma"]):
        if desamb_distinto:
            return "llm"                                # homonimo con desambiguador
        return "auto_nombre"                            # nombre identico
    if cos_nombre >= 0.90 and cos_ficha >= 0.70:
        if desamb_distinto:
            return "llm"                                # 'The Girl of the Golden West (1922)' vs '(1923)'
        return "auto_casi"                              # casi identico + contexto afin
    if cos_nombre >= UMBRAL_NOMBRE:
        return "llm"                                    # variante de nombre
    if cos_ficha >= UMBRAL_CAND and len(_tokens_sig(a["forma"]) & _tokens_sig(b["forma"])) >= 1:
        return "llm"                                    # contexto afin + token comun
    return None


class _UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def _adjudicar(client: OpenAI, con: sqlite3.Connection,
               pares: list[tuple[int, int]], fichas: list[dict]) -> set[tuple[int, int]]:
    """LLM decide 'misma entidad' para la banda ambigua; cache por par."""
    si: set[tuple[int, int]] = set()
    pendientes = []
    for i, j in pares:
        k = f"{fichas[i]['clave']}##{fichas[j]['clave']}"
        fila = con.execute("SELECT r FROM adj WHERE k=?", (k,)).fetchone()
        if fila:
            if fila[0] == "1":
                si.add((i, j))
        else:
            pendientes.append((i, j, k))
    print(f"  adjudicacion LLM: {len(pares) - len(pendientes)} en cache, {len(pendientes)} nuevas")
    for b0 in range(0, len(pendientes), LOTE_LLM):
        lote = pendientes[b0:b0 + LOTE_LLM]
        lineas = "\n".join(
            f"{n}. A: \"{_texto_ficha(fichas[i])[:220]}\" (from article '{fichas[i]['pasaje']}')\n"
            f"   B: \"{_texto_ficha(fichas[j])[:220]}\" (from article '{fichas[j]['pasaje']}')"
            for n, (i, j, _) in enumerate(lote))
        try:
            r = client.chat.completions.create(
                model=MODELO_LLM, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content":
                           "For each pair, decide whether A and B refer to the SAME real-world "
                           "entity (same person/work/place), judging by names, dates, roles and "
                           "context. Be strict: relatives, successors or namesakes (father/son, "
                           "4th vs 5th Baron, Agnes vs Albert of the same house), remakes of a "
                           "film, or different people sharing a surname are NOT the same. "
                           "A short form and a full form of one name (\"H. P. Lovecraft\" / "
                           "\"Howard Phillips Lovecraft\") ARE the same. "
                           'Reply JSON: {"same": {"<n>": true/false, ...}}\n\n' + lineas}])
            res = json.loads(r.choices[0].message.content).get("same", {})
        except Exception:
            res = {}
        for n, (i, j, k) in enumerate(lote):
            v = bool(res.get(str(n), False))
            con.execute("INSERT OR REPLACE INTO adj VALUES (?,?)", (k, "1" if v else "0"))
            if v:
                si.add((i, j))
        con.commit()
        if (b0 // LOTE_LLM) % 50 == 0:
            print(f"  adjudicados {min(b0 + LOTE_LLM, len(pendientes))}/{len(pendientes)}")
    return si


def construir(split: str, ver: str = "", legado: bool = False,
              sin_embeddings: bool = False) -> None:
    """legado=True reproduce la logica de la v3 (sin herencia de fechas, sin
    regla de desambiguador, titulo->sujeto siempre automatico): sirve para
    escribir la PROCEDENCIA de las uniones de la version evaluada sin tocarla.
    sin_embeddings=True omite la ficha fusionada (npz) — para auditorias."""
    docs = [json.loads(l) for l in open(WIKI2_DIR / f"openie_{split}.jsonl", encoding="utf-8")]
    client = OpenAI()
    fichas = fichas_por_mencion(docs, heredar_fechas=not legado)
    n_her = sum(1 for f in fichas if f["fechas_heredadas"])
    print(f"{split}{ver}: {len(fichas)} menciones de entidad en {len(docs)} pasajes"
          f"{' (legado)' if legado else f'; fichas-titulo con fechas heredadas: {n_her}'}")

    # embeddings de fichas y de nombres (cache npz; independiente de la version)
    ruta_f = WIKI2_DIR / f"emb_menciones_{split}.npz"
    if ruta_f.exists() and len(np.load(ruta_f)["ids"]) == len(fichas):
        d = np.load(ruta_f); F, N = d["F"], d["N"]
    else:
        F = _embeber(client, [_texto_ficha(f) for f in fichas])
        N = _embeber(client, [f["forma"] for f in fichas])
        np.savez_compressed(ruta_f, ids=np.array([f["clave"] for f in fichas]), F=F, N=N)

    # candidatos por DOS canales: vecinos por nombre (principal) y por ficha
    uf = _UF(len(fichas))
    uniones: list[tuple[int, int, str]] = []      # procedencia
    ambiguos: dict[tuple[int, int], str] = {}     # par -> origen si el juez dice si
    for i0 in range(0, len(fichas), 2000):
        bloque_f = F[i0:i0 + 2000] @ F.T
        bloque_n = N[i0:i0 + 2000] @ N.T
        for bi in range(bloque_f.shape[0]):
            i = i0 + bi
            fila_f, fila_n = bloque_f[bi], bloque_n[bi]
            cand = set(np.argpartition(-fila_f, VECINOS + 1)[:VECINOS + 1].tolist())
            cand |= set(np.argpartition(-fila_n, VECINOS + 1)[:VECINOS + 1].tolist())
            for j in cand:
                if j <= i or fichas[i]["pasaje"] == fichas[j]["pasaje"]:
                    continue                        # misma pagina: otra entidad
                cf, cn = float(fila_f[j]), float(fila_n[j])
                if cn < UMBRAL_NOMBRE and cf < UMBRAL_CAND:
                    continue
                veredicto = _clasificar_par(fichas[i], fichas[j], cf, cn, legado=legado)
                if veredicto in ("auto_nombre", "auto_casi"):
                    uf.union(i, j); uniones.append((i, j, veredicto))
                elif veredicto == "llm":
                    ambiguos.setdefault((i, j), "juez_vecinos")
    n_auto = len(uniones)
    n_tit = 0
    for t, m, caso in unir_titulos(fichas):
        par = (min(t, m), max(t, m))
        if legado or caso == "titulo_sintetico":
            uf.union(t, m); uniones.append((par[0], par[1], caso)); n_tit += 1
        else:
            ambiguos.setdefault(par, "juez_" + caso)
    print(f"  fusiones automaticas: {n_auto} | titulo->sujeto directas: {n_tit} | "
          f"pares para el juez: {len(ambiguos)} "
          f"({dict(Counter(ambiguos.values()))})")

    WIKI2_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(WIKI2_DIR / "entidades_cache.sqlite", timeout=120)  # varios procesos
    con.execute("CREATE TABLE IF NOT EXISTS adj (k TEXT PRIMARY KEY, r TEXT)")
    aceptados = _adjudicar(client, con, sorted(ambiguos), fichas)
    for i, j in sorted(aceptados):
        uf.union(i, j); uniones.append((i, j, ambiguos[(i, j)]))
    print(f"  uniones por origen: {dict(Counter(o for _, _, o in uniones))}")

    # entradas canonicas
    grupos: dict[int, list[int]] = defaultdict(list)
    for i in range(len(fichas)):
        grupos[uf.find(i)].append(i)
    origen_grupo: dict[int, Counter] = defaultdict(Counter)
    for i, j, o in uniones:
        origen_grupo[uf.find(i)][o] += 1
    entradas, mapa, ids_usados = [], {}, set()
    for gid, miembros in grupos.items():
        fs = [fichas[i] for i in miembros]
        titulos = [f for f in fs if f["es_titulo"]]
        canon = (titulos[0]["forma"] if titulos else
                 max((f["forma"] for f in fs), key=lambda s: sum(1 for f in fs if f["forma"] == s)))
        alias = sorted({f["forma"] for f in fs} | ({f["pasaje"] for f in titulos}))
        desc = " ".join(dict.fromkeys(fr for f in fs for fr in f["desc"].split(". ") if fr))[:600]
        eid = f"can:{_plegar(canon)}"
        if eid in ids_usados:                           # colision de nombre: sufijo
            eid = f"{eid}#{len(entradas)}"
        ids_usados.add(eid)
        entradas.append({"id": eid, "nombre": canon, "alias": alias, "descripcion": desc,
                         "pasajes": sorted({f["pasaje"] for f in fs}),
                         "pasaje_propio": titulos[0]["pasaje"] if titulos else None,
                         "n_menciones": len(fs),
                         "origenes": dict(origen_grupo.get(gid, {}))})
        for f in fs:
            mapa[f["clave"]] = eid
    print(f"  entradas canonicas: {len(entradas)} (de {len(fichas)} menciones; "
          f"{sum(1 for e in entradas if e['n_menciones'] > 1)} con >1 mencion)")

    json.dump(entradas, open(WIKI2_DIR / f"entidades_{split}{ver}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(mapa, open(WIKI2_DIR / f"entidades_mapa_{split}{ver}.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    with open(WIKI2_DIR / f"entidades_uniones_{split}{ver}.jsonl", "w", encoding="utf-8") as f:
        for i, j, o in uniones:
            f.write(json.dumps({"a": fichas[i]["clave"], "b": fichas[j]["clave"], "origen": o},
                               ensure_ascii=False) + "\n")
    if not sin_embeddings:
        Fc = _embeber(client, [f"{e['nombre']} ({', '.join(e['alias'][:4])}) — {e['descripcion'][:300]}"
                               for e in entradas])
        np.savez_compressed(WIKI2_DIR / f"emb_fichas_{split}{ver}.npz",
                            ids=np.array([e["id"] for e in entradas]), mat=Fc)
    print(f"OK entidades_{split}{ver}.json / entidades_mapa_{split}{ver}.json / "
          f"entidades_uniones_{split}{ver}.jsonl"
          f"{'' if sin_embeddings else f' / emb_fichas_{split}{ver}.npz'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True,
                    help="benchmark, sondeo, hotpot_benchmark, hotpot_sondeo, musique_*")
    ap.add_argument("--ver", default="", help="sufijo de version de los artefactos (p.ej. _v4)")
    ap.add_argument("--legado", action="store_true", help="logica v3 (solo para procedencia)")
    ap.add_argument("--sin-embeddings", action="store_true")
    args = ap.parse_args()
    construir(args.split, ver=args.ver, legado=args.legado, sin_embeddings=args.sin_embeddings)


if __name__ == "__main__":
    main()
