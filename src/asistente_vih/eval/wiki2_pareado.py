"""Comparacion PAREADA de dos sistemas sobre el benchmark 2Wiki.

Lee artifacts/wiki2/resultados_<A>.jsonl y resultados_<B>.jsonl (misma lista
de preguntas) y, para cada metrica, calcula la diferencia media A-B con IC 95%
por bootstrap pareado (remuestreo de preguntas), el recuento de preguntas que
gana/pierde A, y el desglose por tipo. Un IC que no cruza 0 = diferencia
significativa (SIG).

    python -m asistente_vih.eval.wiki2_pareado catrag_router_v3dicc_benchmark hipporag2_benchmark
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .wiki2 import WIKI2_DIR, N_BOOTSTRAP, SEMILLA

METRICAS = ("recall@2", "recall@5", "full_chain@5", "full_chain@10", "full_chain@20")


def _cargar(nombre: str) -> dict[str, dict]:
    ruta = WIKI2_DIR / f"resultados_{nombre}.jsonl"
    return {f["_id"]: f for f in (json.loads(l) for l in open(ruta, encoding="utf-8"))}


def comparar(a: str, b: str, metricas=METRICAS) -> dict:
    A, B = _cargar(a), _cargar(b)
    ids = [i for i in A if i in B]
    rng = np.random.default_rng(SEMILLA)
    salida = {"A": a, "B": b, "n": len(ids), "metricas": {}}
    print(f"\n== {a}  vs  {b}  (n={len(ids)} preguntas pareadas) ==")
    for m in metricas:
        da = np.array([A[i][m] for i in ids]); db = np.array([B[i][m] for i in ids])
        dif = da - db
        boots = rng.choice(dif, size=(N_BOOTSTRAP, len(dif)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        gana, pierde = int((dif > 0).sum()), int((dif < 0).sum())
        salida["metricas"][m] = {"A": float(da.mean()), "B": float(db.mean()),
                                 "dif": float(dif.mean()), "ic": [float(lo), float(hi)],
                                 "sig": sig, "gana": gana, "pierde": pierde}
        print(f"  {m:14s} A {100*da.mean():5.1f}  B {100*db.mean():5.1f}  "
              f"dif {100*dif.mean():+5.1f} [{100*lo:+5.1f}, {100*hi:+5.1f}] {sig:3s}  "
              f"gana {gana:3d} / pierde {pierde:3d}")
    # por tipo, full_chain@5
    print("  por tipo (full_chain@5):")
    salida["por_tipo"] = {}
    for tipo in sorted({A[i]["type"] for i in ids}):
        sel = [i for i in ids if A[i]["type"] == tipo]
        dif = np.array([A[i]["full_chain@5"] - B[i]["full_chain@5"] for i in sel])
        boots = rng.choice(dif, size=(N_BOOTSTRAP, len(dif)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        salida["por_tipo"][tipo] = {"dif": float(dif.mean()), "ic": [float(lo), float(hi)], "sig": sig}
        print(f"    {tipo:18s} n={len(sel):3d}  dif {100*dif.mean():+5.1f} "
              f"[{100*lo:+5.1f}, {100*hi:+5.1f}] {sig}")
    with open(WIKI2_DIR / f"pareado_{a}__vs__{b}.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a"); ap.add_argument("b")
    args = ap.parse_args()
    comparar(args.a, args.b)


if __name__ == "__main__":
    main()
