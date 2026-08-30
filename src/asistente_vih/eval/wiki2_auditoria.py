"""Auditoria E1 del diccionario canonico con ORO EXTERNO: los identificadores
Wikidata que el propio conjunto 2Wiki trae en cada pregunta (`evidences` +
`evidences_id`, y `entity_ids` en las de comparacion). Sin coste de API.

Unidad: par (forma, QID) del oro, resuelto a una entrada del diccionario por
alias exacto (plegado) y, si el alias es ambiguo, restringiendo a las entradas
que aparecen en los pasajes oro de esa pregunta; si aun no, por el mapa de
menciones (pasaje oro || forma).

Medidas:
  sobre-fusion   entradas con >= 2 QIDs distintos (parientes, secuelas, homonimos)
  infra-fusion   QIDs repartidos en >= 2 entradas (la misma entidad partida)
  precision / exhaustividad de fusion por PARES de unidades (misma entrada <->
  mismo QID), como en el reconocimiento de entidades (pairwise P/R/F1)
  atribucion     si hay fichero de procedencia (entidades_uniones_*), que
                 origen de union conecta las fichas de QIDs distintos en cada
                 entrada sobre-fundida.

El oro del benchmark se usa SOLO para medir; nunca para construir.

    python -m asistente_vih.eval.wiki2_auditoria --ver ""        # v3 (procedencia: _legado)
    python -m asistente_vih.eval.wiki2_auditoria --ver _v4
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque

from .wiki2 import WIKI2_DIR, cargar_split, titulos_oro
from ..retrieval.wiki2_catrag import _plegar_alias


def _plegar(s: str) -> str:
    return " ".join(s.lower().split())


def pares_oro(preguntas: list[dict]) -> list[dict]:
    """(forma, qid, id de pregunta, titulos oro) para cada entidad del oro."""
    out, vistos = [], set()
    for q in preguntas:
        oro = titulos_oro(q)
        pares = []
        for (s, _, o), (sid, _, oid) in zip(q.get("evidences", []), q.get("evidences_id", [])):
            pares += [(s, sid), (o, oid)]
        if not q.get("evidences_id") and q.get("entity_ids"):
            # comparacion: evidences_id vacio; entity_ids lista los sujetos en orden
            sujetos = list(dict.fromkeys(ev[0] for ev in q.get("evidences", [])))
            ids = q["entity_ids"].split("_")
            if len(sujetos) == len(ids):
                pares += list(zip(sujetos, ids))
        sujetos = {_plegar_alias(ev[0]) for ev in q.get("evidences", [])}
        titulos = {_plegar_alias(t) for t in oro}
        for forma, qid in pares:
            if not str(qid).startswith("Q") or not forma:
                continue
            k = (forma, qid, q["_id"])
            if k in vistos:
                continue
            vistos.add(k)
            pl = _plegar_alias(forma)
            out.append({"forma": forma, "qid": qid, "q": q["_id"], "oro": oro,
                        "cadena": pl in sujetos or pl in titulos})
    return out


def cargar_diccionario(split: str, ver: str) -> tuple[dict, dict, dict]:
    entradas = {e["id"]: e for e in json.loads(
        (WIKI2_DIR / f"entidades_{split}{ver}.json").read_text(encoding="utf-8"))}
    mapa = json.loads((WIKI2_DIR / f"entidades_mapa_{split}{ver}.json").read_text(encoding="utf-8"))
    alias: dict[str, set[str]] = defaultdict(set)
    for e in entradas.values():
        for al in e["alias"] + [e["nombre"]]:
            alias[_plegar_alias(al)].add(e["id"])
    return entradas, mapa, alias


def resolver(par: dict, entradas: dict, mapa: dict, alias: dict) -> tuple[str | None, str]:
    """-> (id de entrada o None, via): 'alias', 'alias+oro', 'mapa', 'ambiguo', 'sin'."""
    cands = set(alias.get(_plegar_alias(par["forma"]), ()))
    if len(cands) == 1:
        return next(iter(cands)), "alias"
    if len(cands) > 1:
        en_oro = {c for c in cands if set(entradas[c]["pasajes"]) & par["oro"]}
        if len(en_oro) == 1:
            return next(iter(en_oro)), "alias+oro"
        if len(en_oro) > 1:
            return None, "ambiguo"
    for t in par["oro"]:
        eid = mapa.get(f"{t}||{_plegar(par['forma'])}")
        if eid:
            return eid, "mapa"
    return None, "ambiguo" if cands else "sin"


def _pares_de_unidades(unidades: list[dict]) -> dict:
    """P/R/F1 por pares: unidad = (forma, qid) distinta resuelta a entrada.
    Solo cuentan pares de unidades con FORMA distinta: la misma forma con dos
    QIDs (Michael Curtiz Q1559143/Q51491) es ruido del oro u homonimia
    irresoluble por nombre, no una decision del diccionario."""
    por_entrada: dict[str, list[int]] = defaultdict(list)
    por_qid: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(unidades):
        por_entrada[u["eid"]].append(i); por_qid[u["qid"]].append(i)
    def pares(idx):
        return [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]
                if _plegar_alias(unidades[a]["forma"]) != _plegar_alias(unidades[b]["forma"])]
    misma_e = [p for v in por_entrada.values() for p in pares(v)]
    mismo_q = [p for v in por_qid.values() for p in pares(v)]
    correctos = sum(1 for a, b in misma_e if unidades[a]["qid"] == unidades[b]["qid"])
    p = correctos / len(misma_e) if misma_e else float("nan")
    r = correctos / len(mismo_q) if mismo_q else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"pares_misma_entrada": len(misma_e), "pares_mismo_qid": len(mismo_q),
            "pares_correctos": correctos, "precision": p, "exhaustividad": r, "f1": f1}


def _atribuir(entradas: dict, mapa: dict, uniones_ruta, sobre: list[dict]) -> dict:
    """Para cada entrada sobre-fundida, camino de uniones entre una ficha del
    QID A y una del QID B; cuenta los origenes que aparecen en esos caminos y
    el origen 'puente' (el menos frecuente del camino, que es el sospechoso)."""
    if not uniones_ruta.exists():
        return {}
    ady: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for l in open(uniones_ruta, encoding="utf-8"):
        u = json.loads(l)
        ady[u["a"]].append((u["b"], u["origen"])); ady[u["b"]].append((u["a"], u["origen"]))
    fichas_de: dict[str, list[str]] = defaultdict(list)
    for clave, eid in mapa.items():
        fichas_de[eid].append(clave)
    en_caminos, puentes, sin_camino = Counter(), Counter(), 0
    for s in sobre:
        claves = fichas_de[s["eid"]]
        formas = {u["qid"]: _plegar(u["forma"]) for u in s["unidades"]}
        qids = list(formas)
        for a_i in range(len(qids)):
            for b_i in range(a_i + 1, len(qids)):
                origen_set = {c for c in claves if c.split("||", 1)[1] == formas[qids[a_i]]}
                destino = {c for c in claves if c.split("||", 1)[1] == formas[qids[b_i]]}
                if not origen_set or not destino:
                    sin_camino += 1
                    continue
                prev: dict[str, tuple[str, str] | None] = {c: None for c in origen_set}
                cola = deque(origen_set); fin = None
                while cola and fin is None:
                    x = cola.popleft()
                    for y, o in ady.get(x, ()):
                        if y not in prev:
                            prev[y] = (x, o); cola.append(y)
                            if y in destino:
                                fin = y; break
                if fin is None:
                    sin_camino += 1
                    continue
                camino = []
                while prev[fin] is not None:
                    x, o = prev[fin]; camino.append(o); fin = x
                for o in set(camino):
                    en_caminos[o] += 1
                # el origen minoritario del camino es el sospechoso de puente
                puentes[min(set(camino), key=lambda o: (sum(1 for c in camino if c == o), o))] += 1
    return {"origenes_en_caminos": dict(en_caminos), "origen_puente": dict(puentes),
            "sin_camino": sin_camino}


def auditar(split: str = "benchmark", ver: str = "", procedencia: str | None = None,
            n_ejemplos: int = 12, solo_cadena: bool = True) -> dict:
    """solo_cadena: unidades = sujetos de evidencia y titulos oro (las entidades
    que el recuperador debe encadenar); fuera quedan objetos-literal (gentilicios,
    lugares) cuyo QID en el oro es inconsistente."""
    preguntas, _ = cargar_split(split)
    pares = pares_oro(preguntas)
    n_total = len(pares)
    if solo_cadena:
        pares = [p for p in pares if p["cadena"]]
    entradas, mapa, alias = cargar_diccionario(split, ver)
    vias = Counter(); unidades_por_clave: dict[tuple[str, str], dict] = {}
    for p in pares:
        eid, via = resolver(p, entradas, mapa, alias)
        vias[via] += 1
        if eid:
            unidades_por_clave.setdefault((p["forma"], p["qid"]), {"forma": p["forma"], "qid": p["qid"], "eid": eid})
    unidades = list(unidades_por_clave.values())
    por_entrada: dict[str, list[dict]] = defaultdict(list)
    por_qid: dict[str, set[str]] = defaultdict(set)
    for u in unidades:
        por_entrada[u["eid"]].append(u); por_qid[u["qid"]].add(u["eid"])
    def qids_formas_distintas(us):
        # QIDs con al menos una forma distinta de las de otro QID de la entrada
        por_q = defaultdict(set)
        for u in us:
            por_q[u["qid"]].add(_plegar_alias(u["forma"]))
        qs = list(por_q)
        return [q for q in qs if any(por_q[q] - por_q[o] for o in qs if o != q)]
    sobre, mismo_nombre = [], 0
    for e, us in por_entrada.items():
        if len({u["qid"] for u in us}) < 2:
            continue
        qs = qids_formas_distintas(us)
        if len(qs) > 1:
            sobre.append({"eid": e, "nombre": entradas[e]["nombre"],
                          "unidades": [u for u in us if u["qid"] in qs],
                          "origenes": entradas[e].get("origenes", {})})
        else:
            mismo_nombre += 1
    infra = [{"qid": q, "entradas": sorted(es),
              "formas": sorted({u["forma"] for u in unidades if u["qid"] == q})}
             for q, es in por_qid.items() if len(es) > 1]
    res = {
        "split": split, "ver": ver, "pares_oro": len(pares), "pares_oro_total": n_total,
        "solo_cadena": solo_cadena, "vias": dict(vias),
        "mismo_nombre_distinto_qid": mismo_nombre,
        "unidades_resueltas": len(unidades), "qids_distintos": len(por_qid),
        "entradas_tocadas": len(por_entrada),
        "sobre_fusion": {"n_entradas": len(sobre),
                         "n_qids_implicados": sum(len({u['qid'] for u in s['unidades']}) for s in sobre)},
        "infra_fusion": {"n_qids": len(infra)},
        "pares": _pares_de_unidades(unidades),
    }
    ruta_proc = WIKI2_DIR / f"entidades_uniones_{split}{procedencia if procedencia is not None else ver}.jsonl"
    res["atribucion"] = _atribuir(entradas, mapa, ruta_proc, sobre)
    res["ejemplos_sobre"] = [{"entrada": s["eid"], "qids": sorted({(u["qid"], u["forma"]) for u in s["unidades"]}),
                              "origenes": s["origenes"]} for s in sobre[:n_ejemplos]]
    res["ejemplos_infra"] = infra[:n_ejemplos]
    return res


def imprimir(res: dict) -> None:
    print(f"\n== auditoria QID: {res['split']}{res['ver'] or ' (v3)'} ==")
    print(f"pares oro (forma, QID): {res['pares_oro']} de {res['pares_oro_total']}"
          f"{' (solo entidades de la cadena)' if res['solo_cadena'] else ''}  vias: {res['vias']}")
    print(f"entradas con la MISMA forma bajo QIDs distintos (ruido del oro/homonimia, excluidas): "
          f"{res['mismo_nombre_distinto_qid']}")
    print(f"unidades resueltas: {res['unidades_resueltas']} | QIDs distintos: {res['qids_distintos']} "
          f"| entradas tocadas: {res['entradas_tocadas']}")
    p = res["pares"]
    print(f"SOBRE-fusion: {res['sobre_fusion']['n_entradas']} entradas con >=2 QIDs "
          f"({res['sobre_fusion']['n_qids_implicados']} QIDs)   "
          f"INFRA-fusion: {res['infra_fusion']['n_qids']} QIDs partidos en >=2 entradas")
    print(f"por pares: precision {100*p['precision']:.1f}%  exhaustividad {100*p['exhaustividad']:.1f}%  "
          f"F1 {100*p['f1']:.1f}%  (misma entrada {p['pares_misma_entrada']}, mismo QID "
          f"{p['pares_mismo_qid']}, correctos {p['pares_correctos']})")
    if res.get("atribucion"):
        a = res["atribucion"]
        print(f"atribucion de la sobre-fusion — origen puente: {a['origen_puente']} | "
              f"presentes en caminos: {a['origenes_en_caminos']} | sin camino: {a['sin_camino']}")
    print("ejemplos sobre-fusion:")
    for e in res["ejemplos_sobre"]:
        print(f"  {e['entrada']}: " + "; ".join(f"{q} «{f}»" for q, f in e["qids"]) +
              (f"   origenes={e['origenes']}" if e["origenes"] else ""))
    print("ejemplos infra-fusion:")
    for e in res["ejemplos_infra"]:
        print(f"  {e['qid']} {e['formas']} -> {e['entradas']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="benchmark")
    ap.add_argument("--ver", default="")
    ap.add_argument("--procedencia", default=None,
                    help="sufijo del fichero de uniones si difiere de --ver (v3: _legado)")
    ap.add_argument("--guardar", action="store_true")
    ap.add_argument("--todas", action="store_true", help="incluir objetos-literal (gentilicios...)")
    args = ap.parse_args()
    res = auditar(args.split, args.ver, args.procedencia, solo_cadena=not args.todas)
    imprimir(res)
    if args.guardar:
        ruta = WIKI2_DIR / f"auditoria_qid_{args.split}{args.ver or '_v3'}.json"
        ruta.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"guardado en {ruta}")


if __name__ == "__main__":
    main()
