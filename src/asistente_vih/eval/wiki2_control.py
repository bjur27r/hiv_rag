"""Analisis del CONTROL SIN GRAFO frente al Plan A v4 (mismo presupuesto de LLM).

Lee resultados_<v4>.jsonl y resultados_<control>.jsonl (mismas preguntas) y
descompone la diferencia de cadena completa a k=5:
  - pareado global y por tipo/estructura (bootstrap, como wiki2_pareado);
  - tabla de contingencia (ambos aciertan / solo v4 / solo control / ninguno);
  - en las preguntas que solo acierta la v4: donde deja el control el oro que
    le falta (puestos 6-10, 11-20, fuera del top-20) -> distingue fallo de
    ORDEN (el pasaje estaba entre sus candidatos) de fallo de ALCANCE (no lo
    encontro ni por coseno ni por titulo);
  - lo mismo al reves, para las que solo acierta el control.

    python -m asistente_vih.eval.wiki2_control plan_v4_4omini_benchmark denso_plan_4omini_benchmark
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from .wiki2 import WIKI2_DIR, N_BOOTSTRAP, SEMILLA, ESTRUCTURAS, cargar_split, titulos_oro


def _cargar(nombre: str) -> dict[str, dict]:
    return {f["_id"]: f for f in (json.loads(l) for l in open(WIKI2_DIR / f"resultados_{nombre}.jsonl", encoding="utf-8"))}


def _posicion(top: list[str], t: str) -> str:
    if t in top[:5]:
        return "top-5"
    if t in top[:10]:
        return "6-10"
    if t in top[:20]:
        return "11-20"
    return "fuera"


def analizar(a: str, b: str, split: str) -> dict:
    A, B = _cargar(a), _cargar(b)
    preg = {q["_id"]: q for q in cargar_split(split)[0]}
    ids = [i for i in A if i in B]
    rng = np.random.default_rng(SEMILLA)
    out = {"A": a, "B": b, "n": len(ids), "metricas": {}, "contingencia": {}, "solo_A": {}, "solo_B": {}}
    print(f"\n== {a}  (A)  vs  {b}  (B)  ·  {split}  ·  n={len(ids)} ==")
    for m in ("recall@2", "recall@5", "full_chain@2", "full_chain@5", "full_chain@10", "full_chain@20"):
        dif = np.array([A[i][m] - B[i][m] for i in ids])
        boots = rng.choice(dif, size=(N_BOOTSTRAP, len(dif)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5]); sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        out["metricas"][m] = {"A": float(np.mean([A[i][m] for i in ids])), "B": float(np.mean([B[i][m] for i in ids])),
                              "dif": float(dif.mean()), "ic": [float(lo), float(hi)], "sig": sig}
        print(f"  {m:14s} A {100*out['metricas'][m]['A']:5.1f}  B {100*out['metricas'][m]['B']:5.1f}  "
              f"dif {100*dif.mean():+5.1f} [{100*lo:+5.1f}, {100*hi:+5.1f}] {sig}")
    # por tipo y por estructura (FC@5)
    tipos = sorted({A[i]["type"] for i in ids})
    estructura = next((e for e in ESTRUCTURAS.values() if set(tipos) <= {t for ts in e.values() for t in ts}), {})
    grupos = [(t, [i for i in ids if A[i]["type"] == t]) for t in tipos]
    grupos += [(f"[{et}]", [i for i in ids if A[i]["type"] in ts]) for et, ts in estructura.items() if len(ts) > 1]
    print("  FC@5 por tipo / estructura:")
    out["por_tipo"] = {}
    for et, sel in grupos:
        dif = np.array([A[i]["full_chain@5"] - B[i]["full_chain@5"] for i in sel])
        boots = rng.choice(dif, size=(N_BOOTSTRAP, len(dif)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5]); sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        out["por_tipo"][et] = {"n": len(sel), "A": float(np.mean([A[i]["full_chain@5"] for i in sel])),
                               "B": float(np.mean([B[i]["full_chain@5"] for i in sel])), "dif": float(dif.mean()),
                               "ic": [float(lo), float(hi)], "sig": sig}
        print(f"    {et:22s} n={len(sel):4d}  A {100*out['por_tipo'][et]['A']:5.1f}  B {100*out['por_tipo'][et]['B']:5.1f}  "
              f"dif {100*dif.mean():+5.1f} [{100*lo:+5.1f}, {100*hi:+5.1f}] {sig}")
    # contingencia
    c = Counter((int(A[i]["full_chain@5"]), int(B[i]["full_chain@5"])) for i in ids)
    out["contingencia"] = {"ambos": c[(1, 1)], "solo_A": c[(1, 0)], "solo_B": c[(0, 1)], "ninguno": c[(0, 0)]}
    print(f"  contingencia FC@5: ambos {c[(1,1)]} · solo A {c[(1,0)]} · solo B {c[(0,1)]} · ninguno {c[(0,0)]}")
    # donde deja el oro perdido el sistema que falla, en las preguntas que el otro acierta
    for etiqueta, gana, pierde, clave in (("solo A acierta: donde deja B el oro", A, B, "solo_A"),
                                          ("solo B acierta: donde deja A el oro", B, A, "solo_B")):
        pos = Counter(); tipos_c = Counter()
        for i in ids:
            if gana[i]["full_chain@5"] == 1 and pierde[i]["full_chain@5"] == 0:
                tipos_c[A[i]["type"]] += 1
                for t in titulos_oro(preg[i]) - set(pierde[i]["top"][:5]):
                    pos[_posicion(pierde[i]["top"], t)] += 1
        out[clave] = {"por_tipo": dict(tipos_c), "posicion_oro_perdido": dict(pos)}
        print(f"  {etiqueta}: {dict(pos)}  · por tipo {dict(tipos_c)}")
    ruta = WIKI2_DIR / f"control_{a}__vs__{b}.json"
    ruta.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("A"); ap.add_argument("B")
    ap.add_argument("--split", default=None, help="por defecto se infiere del nombre de A")
    args = ap.parse_args()
    split = args.split or ("hotpot_benchmark" if "hotpot_benchmark" in args.A else "hotpot_sondeo" if "hotpot_sondeo" in args.A
                           else "benchmark" if "benchmark" in args.A else "sondeo")
    analizar(args.A, args.B, split)


if __name__ == "__main__":
    main()
