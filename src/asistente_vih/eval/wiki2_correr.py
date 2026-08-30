"""Lanzador de mediciones del Plan A (v4 generalizado) sobre el sondeo o el
banco, con las ablaciones por conmutador y la comparacion pareada.

    PYTHONPATH=src python3 -m asistente_vih.eval.wiki2_correr --split sondeo --modelo gpt-4o-mini
    PYTHONPATH=src python3 -m asistente_vih.eval.wiki2_correr --split sondeo --modelo gpt-4o-mini --dicc "" --nombre plan_v4_dicc3_4omini_sondeo
    PYTHONPATH=src python3 -m asistente_vih.eval.wiki2_correr --split benchmark --modelo deepseek-chat --pareado plan_deepseek_benchmark hipporag2_benchmark

El nombre por defecto codifica la configuracion: plan_v4[_<ablacion>]_<modelo>_<split>.
"""
from __future__ import annotations

import argparse
import re
import time

from .wiki2 import cargar_split, evaluar, imprimir
from ..retrieval.wiki2_plan import Wiki2PlanRAG


def _etiqueta_modelo(modelo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", modelo.split("/")[-1].lower().replace("gpt-", "")
                  .replace("deepseek-chat", "deepseek"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="sondeo",
                    help="benchmark, sondeo, hotpot_benchmark, hotpot_sondeo, musique_*")
    ap.add_argument("--modelo", default="gpt-4o-mini")
    ap.add_argument("--dicc", default="_v4", help='sufijo del diccionario ("" = v3)')
    ap.add_argument("--enlazador", default="l2", choices=["l2", "v3"])
    ap.add_argument("--sin-conjunto", action="store_true")
    ap.add_argument("--sin-salto", action="store_true")
    ap.add_argument("--conjunto-solo-no-comparacion", action="store_true",
                    help="como el Plan A del 25-08: sin conjunto en comparison")
    ap.add_argument("--max-rondas", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="solo las n primeras preguntas (humo)")
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--no-guardar", action="store_true")
    ap.add_argument("--pareado", nargs="*", default=None,
                    help="nombres de resultados contra los que comparar (wiki2_pareado)")
    args = ap.parse_args()

    r = Wiki2PlanRAG(split=args.split, modelo=args.modelo, dicc=args.dicc,
                     enlazador=args.enlazador, usar_conjunto=not args.sin_conjunto,
                     usar_salto=not args.sin_salto,
                     conjunto_en_comparacion=not args.conjunto_solo_no_comparacion,
                     max_rondas=args.max_rondas)
    preguntas, _ = cargar_split(args.split)
    if args.limit:
        preguntas = preguntas[:args.limit]
    if args.nombre is None:
        abl = ("_dicc3" if args.dicc == "" else "") + ("_enl3" if args.enlazador == "v3" else "") \
            + ("_sinconjunto" if args.sin_conjunto else "") + ("_sinsalto" if args.sin_salto else "") \
            + ("_conjcomp" if args.conjunto_solo_no_comparacion else "") \
            + (f"_r{args.max_rondas}" if args.max_rondas != 3 else "")
        args.nombre = f"plan_v4{abl}_{_etiqueta_modelo(args.modelo)}_{args.split}"
    print(f"config: split={args.split} modelo={args.modelo} dicc={args.dicc!r} enlazador={args.enlazador} "
          f"conjunto={not args.sin_conjunto} (en comparacion: {not args.conjunto_solo_no_comparacion}) "
          f"salto={not args.sin_salto} rondas<={args.max_rondas} n={len(preguntas)} -> {args.nombre}")
    t0 = time.time()
    agg = evaluar(r.buscar, preguntas, nombre=args.nombre,
                  guardar=not args.no_guardar and not args.limit)
    imprimir(agg)
    print(f"tiempo {time.time()-t0:.0f}s; llamadas LLM nuevas (no en cache): {r.n_llamadas}")
    if args.pareado:
        from .wiki2_pareado import comparar
        for otro in args.pareado:
            comparar(args.nombre, otro)


if __name__ == "__main__":
    main()
