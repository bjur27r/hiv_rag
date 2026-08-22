"""Paso 3 - Calibracion del disparador HITL contra el banco.

El recall del HITL es una metrica de SEGURIDAD: un falso negativo (no interrumpir
cuando debia) es peor que un falso positivo. Aqui se corre el router (Haiku) sobre
el banco y se compara la prediccion de HITL (misma regla que el grafo: decision
seria, o consulta de paciente con dato faltante) contra la etiqueta 'requiere_hitl'.

Reporta matriz de confusion, precision/recall/F1 global y por nivel.

    python -m asistente_vih.eval.hitl_calibracion            # 505 (Haiku, ~barato)
    python -m asistente_vih.eval.hitl_calibracion --muestra 120
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from ..agent.graph import ROUTER_SCHEMA, ROUTER_SYS, _router_heuristico
from ..config import EVAL_DATASET_PATH
from ..llm.client import LLM, MODELO_ROUTER


def predice_hitl(cls: dict) -> bool:
    paciente = cls.get("tipo_consulta") == "paciente"
    return bool(cls.get("decision_seria") or (paciente and cls.get("datos_faltantes")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=int, help="evalua solo las primeras N preguntas")
    args = ap.parse_args()

    banco = [json.loads(l) for l in open(EVAL_DATASET_PATH, encoding="utf-8") if l.strip()]
    if args.muestra:
        banco = banco[:args.muestra]

    llm = LLM(MODELO_ROUTER)
    print(f"Router: {llm.provider}:{llm.modelo} | {'SIMULADO' if llm.simulado else 'REAL'} | {len(banco)} preguntas\n")

    tp = fp = fn = tn = 0
    por_nivel = defaultdict(lambda: [0, 0, 0, 0])  # nivel -> [tp,fp,fn,tn]
    fallos_recall = []  # casos HITL gold que el router no detecto (los importantes)

    errores = 0
    for i, e in enumerate(banco, 1):
        q = e["inputs"]["pregunta"]
        try:
            cls = llm.json(ROUTER_SYS, q, ROUTER_SCHEMA, max_tokens=800,
                           simulado_valor=_router_heuristico(q))
        except Exception:
            errores += 1
            continue
        pred = predice_hitl(cls)
        gold = bool(e["outputs"].get("requiere_hitl"))
        niv = e["metadata"].get("nivel")
        if pred and gold:
            tp += 1; por_nivel[niv][0] += 1
        elif pred and not gold:
            fp += 1; por_nivel[niv][1] += 1
        elif not pred and gold:
            fn += 1; por_nivel[niv][2] += 1
            fallos_recall.append((e["metadata"].get("id"), q[:70]))
        else:
            tn += 1; por_nivel[niv][3] += 1
        if i % 50 == 0:
            print(f"  ...{i}/{len(banco)}")

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f = prf(tp, fp, fn)
    print("\n=== Calibracion HITL ===")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}  (errores router={errores})")
    print(f"  Precision={p:.1%}  Recall={r:.1%}  F1={f:.1%}")
    print(f"  (Recall es seguridad: FN={fn} casos serios NO interrumpidos)")
    print("\n  Por nivel:")
    for niv in sorted(x for x in por_nivel if x is not None):
        a, b, c, d = por_nivel[niv]
        pp, rr, ff = prf(a, b, c)
        print(f"    Nivel {niv}: TP={a} FP={b} FN={c} TN={d} | P={pp:.0%} R={rr:.0%}")
    if fallos_recall:
        print(f"\n  Falsos negativos (HITL no disparado) — revisar el router:")
        for cid, q in fallos_recall[:15]:
            print(f"    [{cid}] {q}")


if __name__ == "__main__":
    main()
