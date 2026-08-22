"""Paso 2 - Evaluacion a nivel de RESPUESTA (no solo recuperacion).

Corre el pipeline real (recuperar -> generar citado -> verificar fidelidad) sobre
una muestra del banco y usa el VERIFICADOR como evaluador automatico: la tasa de
fidelidad y de abstencion por nivel son la senal de calidad de extremo a extremo.

Si LANGSMITH_TRACING=true, cada llamada queda trazada en LangSmith. La validacion
plena de correccion CLINICA exige el patron oro validado (paso 1) — esto mide
grounding/fidelidad, no correccion clinica absoluta.

  python -m asistente_vih.eval.answer_eval               # muestra estratificada (3/nivel)
  python -m asistente_vih.eval.answer_eval --por-nivel 8
  python -m asistente_vih.eval.answer_eval --full        # las 505 (coste alto, lento)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from ..config import EVAL_DATASET_PATH
from ..llm.client import LLM, MODELO_SINTESIS, MODELO_VERIFICADOR
from ..llm.generate import generar_respuesta
from ..llm.verify import verificar_fidelidad


def _muestra(banco, por_nivel):
    sel, cont = [], defaultdict(int)
    for e in banco:
        n = e["metadata"].get("nivel")
        if cont[n] < por_nivel:
            sel.append(e); cont[n] += 1
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-nivel", type=int, default=3, help="preguntas por nivel en la muestra")
    ap.add_argument("--full", action="store_true", help="evalua las 505 (coste alto)")
    args = ap.parse_args()

    banco = [json.loads(l) for l in open(EVAL_DATASET_PATH, encoding="utf-8") if l.strip()]
    casos = banco if args.full else _muestra(banco, args.por_nivel)

    from ..retrieval.dense import DenseRetriever
    from ..retrieval.decompose import DecomposingRetriever
    denso = DenseRetriever()
    decomp = DecomposingRetriever(denso)
    sint, verif = LLM(MODELO_SINTESIS), LLM(MODELO_VERIFICADOR)
    print(f"Pipeline: denso + {sint.provider}:{sint.modelo} (sintesis) + {verif.modelo} (verif)")
    print(f"Modo: {'SIMULADO' if sint.simulado else 'REAL'} | {len(casos)} preguntas\n")

    agg = defaultdict(lambda: {"n": 0, "fiel": 0, "abst": 0, "parcial": 0})
    for i, e in enumerate(casos, 1):
        q = e["inputs"]["pregunta"]
        m = e["metadata"]
        nivel = m.get("nivel")
        multi = (m.get("n_guias") or "1").isdigit() and int(m.get("n_guias")) >= 2
        retr = decomp if multi else denso
        a = agg[nivel]
        a["n"] += 1
        try:
            hits = retr.search(q, k=20 if multi else 8)
            gen = generar_respuesta(q, hits, llm=sint)
            v = verificar_fidelidad(q, gen["respuesta"], hits, llm=verif)
        except Exception as ex:
            a["error"] = a.get("error", 0) + 1
            print(f"  [{i}/{len(casos)}] N{nivel} {m.get('id')}: ERROR {type(ex).__name__}")
            continue
        a["fiel"] += int(bool(v.get("fiel")))
        a["abst"] += int(bool(gen.get("abstenida")))
        a["parcial"] += int(v.get("veredicto") == "parcial")
        print(f"  [{i}/{len(casos)}] N{nivel} {m.get('id')}: fiel={v.get('fiel')} "
              f"abst={gen.get('abstenida')} ver={v.get('veredicto')}")

    print("\n=== Calidad a nivel de respuesta (verificador como evaluador) ===")
    tot = defaultdict(int)
    for niv in sorted(x for x in agg if x is not None):
        a = agg[niv]
        for k in ("n", "fiel", "abst", "parcial"):
            tot[k] += a[k]
        print(f"  Nivel {niv} (n={a['n']}): fidelidad={a['fiel']/a['n']:.0%} "
              f"abstencion={a['abst']/a['n']:.0%} parcial={a['parcial']/a['n']:.0%}")
    if tot["n"]:
        print(f"  GLOBAL (n={tot['n']}): fidelidad={tot['fiel']/tot['n']:.0%} "
              f"abstencion={tot['abst']/tot['n']:.0%}")
    if not args.full:
        print(f"\n  Proyeccion: la muestra son {tot['n']} preguntas; el banco completo son 505 "
              f"(~{505/max(tot['n'],1):.0f}x el coste/tiempo). Usa --full para el run completo.")


if __name__ == "__main__":
    main()
