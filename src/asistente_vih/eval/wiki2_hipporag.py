"""Baseline HippoRAG 2 sobre 2Wiki con su codigo vendorizado (Fase D).

Corre en el venv del proyecto (deps de HippoRAG instaladas ahi, no en el
sistema): indexa el corpus con su OpenIE (gpt-4o-mini, 2 llamadas/pasaje) y
recupera con su pipeline completo (query-to-fact + recognition memory + siembra
densa + PPR igraph). Mismos modelos que nuestro sistema: paridad garantizada.

Los artefactos de indexacion van a artifacts/wiki2/hipporag_<split>/ (openie
json reanudable + embedding stores parquet). Resultados via eval.wiki2.evaluar.

    venv/bin/python -m asistente_vih.eval.wiki2_hipporag --split benchmark --humo
    venv/bin/python -m asistente_vih.eval.wiki2_hipporag --split benchmark
"""
from __future__ import annotations

import argparse
import json
import sys

from ..config import ROOT
from .wiki2 import (WIKI2_DIR, cargar_benchmark, cargar_sondeo, evaluar,
                    imprimir, KS_DEFECTO)

sys.path.insert(0, str(ROOT / "HippoRAG" / "src"))

MODELO_LLM = "gpt-4o-mini"
MODELO_EMB = "text-embedding-3-small"


def correr(split: str, humo: bool = False) -> None:
    from hipporag import HippoRAG

    if split == "benchmark":
        preguntas, corpus = cargar_benchmark()
    else:
        preguntas, corpus = cargar_sondeo()
    if humo:
        # mini-prueba: pasajes oro de las 3 primeras preguntas + relleno
        oros = {t for q in preguntas[:3] for t, _ in q["supporting_facts"]}
        corpus = [c for c in corpus if c["title"] in oros] + corpus[:20]
        preguntas = preguntas[:3]

    docs = [f"{c['title']}\n{c['text']}" for c in corpus]
    save_dir = WIKI2_DIR / f"hipporag_{split}{'_humo' if humo else ''}"
    rag = HippoRAG(save_dir=str(save_dir), llm_model_name=MODELO_LLM,
                   embedding_model_name=MODELO_EMB)
    rag.index(docs)

    kmax = max(KS_DEFECTO)
    consultas = [q["question"] for q in preguntas]
    soluciones = rag.retrieve(queries=consultas, num_to_retrieve=kmax)
    por_query = {q: [d.split("\n", 1)[0] for d in s.docs]
                 for q, s in zip(consultas, soluciones)}

    def buscar(query: str, k: int) -> list[str]:
        return por_query.get(query, [])[:k]

    agg = evaluar(buscar, preguntas, nombre=f"hipporag2_{split}",
                  guardar=not humo)
    imprimir(agg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["benchmark", "sondeo"], default="benchmark")
    ap.add_argument("--humo", action="store_true", help="mini-prueba de fontaneria")
    args = ap.parse_args()
    correr(args.split, humo=args.humo)


if __name__ == "__main__":
    main()
