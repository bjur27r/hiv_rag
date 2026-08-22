"""Memoria de entidades (diccionario canonico) del corpus 2Wiki.

Primera pieza del pipeline de evaluacion: construye, UNA vez y solo desde el
corpus (nunca desde el oro del benchmark), una entrada por entidad canonica con
sus alias, su descripcion y los pasajes donde aparece. Analogo al linker
terminologico del caso VIH, pero derivado del propio corpus (generaliza a
cualquier dominio; Wikidata queda como verificacion opcional).

Pasos:
  1. Ficha por MENCION (entidad x pasaje): nombre + descriptores (frases de
     las aserciones de ese pasaje en las que participa) -> embedding.
  2. Candidatos a misma entidad: coseno entre fichas >= UMBRAL_CAND y nombres
     compatibles (token significativo comun o coseno de nombre alto) y sin
     contradiccion de fechas de nacimiento/muerte.
  3. Adjudicacion: coseno de ficha >= UMBRAL_AUTO -> fusion automatica;
     banda [UMBRAL_CAND, UMBRAL_AUTO) -> LLM decide viendo las dos fichas
     (lotes, cache SQLite). Union-find -> grupos.
  4. Entrada canonica por grupo: nombre (preferencia: titulo de pasaje),
     alias, descripcion fusionada, pasajes; embedding de la ficha fusionada.

Salida (artifacts/wiki2/):
  entidades_<split>.json        lista de entradas canonicas
  entidades_mapa_<split>.json   "titulo||forma_plegada" -> id canonico
  emb_fichas_<split>.npz        ids canonicos + matriz (ficha fusionada)

    python -m asistente_vih.ingest.wiki2_entidades --split sondeo
    python -m asistente_vih.ingest.wiki2_entidades --split benchmark
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict

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


def _plegar(s: str) -> str:
    return " ".join(s.lower().split())


def _tokens_sig(nombre: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", " ", nombre.lower()))
    return {t for t in toks if len(t) > 2 and t not in _GENERICOS}


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

def fichas_por_mencion(docs: list[dict]) -> list[dict]:
    fichas = []
    for d in docs:
        titulo = d["chunk_id"]
        for forma in d["entidades"]:
            pleg = _plegar(forma)
            frases = [a["descripcion_relacion"] for a in d["aserciones"]
                      if pleg in _plegar(a["descripcion_relacion"]) or
                      pleg in (_plegar(a["sujeto"]), _plegar(a["objeto"]))]
            desc = " ".join(dict.fromkeys(f for f in frases if f))[:400]
            fichas.append({
                "clave": f"{titulo}||{pleg}", "forma": forma, "pleg": pleg,
                "pasaje": titulo, "es_titulo": _plegar(titulo) == pleg or
                _plegar(re.sub(r"\(.*?\)", "", titulo)).strip() == pleg,
                "desc": desc,
                "nac": _anios(d["aserciones"], pleg, "DATE_OF_BIRTH"),
                "mue": _anios(d["aserciones"], pleg, "DATE_OF_DEATH"),
            })
    return fichas


def _texto_ficha(f: dict) -> str:
    return f"{f['forma']} — {f['desc']}" if f["desc"] else f["forma"]


# ------------------------------------------------------------ paso 2-3: grupos

def _compatibles(a: dict, b: dict, cos_nombre: float) -> bool:
    if a["nac"] and b["nac"] and a["nac"].isdisjoint(b["nac"]):
        return False
    if a["mue"] and b["mue"] and a["mue"].isdisjoint(b["mue"]):
        return False
    if cos_nombre >= UMBRAL_NOMBRE:
        return True
    return bool(_tokens_sig(a["forma"]) & _tokens_sig(b["forma"]))


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
                           "context. Different people with similar names are NOT the same. "
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


def construir(split: str) -> None:
    docs = [json.loads(l) for l in open(WIKI2_DIR / f"openie_{split}.jsonl", encoding="utf-8")]
    client = OpenAI()
    fichas = fichas_por_mencion(docs)
    print(f"{split}: {len(fichas)} menciones de entidad en {len(docs)} pasajes")

    # embeddings de fichas y de nombres (cache npz)
    ruta_f = WIKI2_DIR / f"emb_menciones_{split}.npz"
    if ruta_f.exists() and len(np.load(ruta_f)["ids"]) == len(fichas):
        d = np.load(ruta_f); F, N = d["F"], d["N"]
    else:
        F = _embeber(client, [_texto_ficha(f) for f in fichas])
        N = _embeber(client, [f["forma"] for f in fichas])
        np.savez_compressed(ruta_f, ids=np.array([f["clave"] for f in fichas]), F=F, N=N)

    # candidatos: vecinos por coseno de ficha, filtrados por compatibilidad
    uf = _UF(len(fichas)); ambiguos = []
    auto = 0
    for i0 in range(0, len(fichas), 2000):
        bloque = F[i0:i0 + 2000] @ F.T
        for bi, fila in enumerate(bloque):
            i = i0 + bi
            cand = np.argpartition(-fila, VECINOS + 1)[:VECINOS + 1]
            for j in cand:
                j = int(j)
                if j <= i or fila[j] < UMBRAL_CAND:
                    continue
                if fichas[i]["pasaje"] == fichas[j]["pasaje"]:
                    continue                        # misma pagina: otra entidad
                cn = float(N[i] @ N[j])
                if not _compatibles(fichas[i], fichas[j], cn):
                    continue
                if fila[j] >= UMBRAL_AUTO:
                    uf.union(i, j); auto += 1
                else:
                    ambiguos.append((i, j))
    print(f"  fusiones automaticas: {auto} | pares ambiguos para el LLM: {len(ambiguos)}")

    WIKI2_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(WIKI2_DIR / "entidades_cache.sqlite")
    con.execute("CREATE TABLE IF NOT EXISTS adj (k TEXT PRIMARY KEY, r TEXT)")
    for i, j in _adjudicar(client, con, ambiguos, fichas):
        uf.union(i, j)

    # entradas canonicas
    grupos: dict[int, list[int]] = defaultdict(list)
    for i in range(len(fichas)):
        grupos[uf.find(i)].append(i)
    entradas, mapa = [], {}
    for gid, miembros in grupos.items():
        fs = [fichas[i] for i in miembros]
        titulos = [f for f in fs if f["es_titulo"]]
        canon = (titulos[0]["forma"] if titulos else
                 max((f["forma"] for f in fs), key=lambda s: sum(1 for f in fs if f["forma"] == s)))
        alias = sorted({f["forma"] for f in fs} | ({f["pasaje"] for f in titulos}))
        desc = " ".join(dict.fromkeys(fr for f in fs for fr in f["desc"].split(". ") if fr))[:600]
        eid = f"can:{_plegar(canon)}"
        if eid in mapa.values():                        # colision de nombre: sufijo
            eid = f"{eid}#{len(entradas)}"
        entradas.append({"id": eid, "nombre": canon, "alias": alias, "descripcion": desc,
                         "pasajes": sorted({f["pasaje"] for f in fs}),
                         "pasaje_propio": titulos[0]["pasaje"] if titulos else None,
                         "n_menciones": len(fs)})
        for f in fs:
            mapa[f["clave"]] = eid
    print(f"  entradas canonicas: {len(entradas)} (de {len(fichas)} menciones; "
          f"{sum(1 for e in entradas if e['n_menciones'] > 1)} con >1 mencion)")

    Fc = _embeber(client, [f"{e['nombre']} ({', '.join(e['alias'][:4])}) — {e['descripcion'][:300]}"
                           for e in entradas])
    np.savez_compressed(WIKI2_DIR / f"emb_fichas_{split}.npz",
                        ids=np.array([e["id"] for e in entradas]), mat=Fc)
    json.dump(entradas, open(WIKI2_DIR / f"entidades_{split}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(mapa, open(WIKI2_DIR / f"entidades_mapa_{split}.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"OK entidades_{split}.json / entidades_mapa_{split}.json / emb_fichas_{split}.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["sondeo", "benchmark"], required=True)
    construir(ap.parse_args().split)


if __name__ == "__main__":
    main()
