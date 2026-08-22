# -*- coding: utf-8 -*-
"""Demo end-to-end del pipeline RAG: recuperar -> generar citado -> verificar.

Funciona SIN clave (modo simulado): ejercita el flujo completo y la integracion
de modulos. Con ANTHROPIC_API_KEY (y VOYAGE_API_KEY para --hybrid) corre de verdad.

    python scripts/demo_rag.py "¿A partir de que FG puede usarse TAF?"
    python scripts/demo_rag.py --hybrid "rifampicina e inhibidores de integrasa"
"""
import argparse
import sys
from pathlib import Path

# La consola de Windows (cp1252) no imprime ciertos Unicode (>=, simbolos clinicos).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asistente_vih.llm.client import LLM, MODELO_SINTESIS, MODELO_VERIFICADOR
from asistente_vih.llm.generate import generar_respuesta
from asistente_vih.llm.verify import verificar_fidelidad
from asistente_vih.retrieval import BM25Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pregunta", nargs="?", default="¿A partir de que filtrado glomerular puede usarse TAF?")
    ap.add_argument("--hybrid", action="store_true", help="usa el hibrido BM25+denso (requiere VOYAGE_API_KEY)")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    print("Construyendo recuperador...")
    bm25 = BM25Retriever.from_jsonl()
    retr = bm25
    if args.hybrid:
        from asistente_vih.retrieval.dense import DenseRetriever
        from asistente_vih.retrieval.hybrid import HybridRetriever
        retr = HybridRetriever(bm25, DenseRetriever())

    sintetizador = LLM(MODELO_SINTESIS)
    verificador = LLM(MODELO_VERIFICADOR)
    modo = "SIMULADO (sin clave del proveedor)" if sintetizador.simulado else "REAL"
    print(f"Modo: {modo} | LLM={sintetizador.provider}:{sintetizador.modelo}\n")

    print("=" * 70); print("PREGUNTA"); print("=" * 70); print(args.pregunta)

    hits = retr.search(args.pregunta, k=args.k)
    print("\n" + "=" * 70); print(f"RECUPERACION (top {len(hits)})"); print("=" * 70)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.guia} | {h.seccion[:50]} | pag {h.pagina} | score {h.score}")

    print("\n" + "=" * 70); print("RESPUESTA CITADA"); print("=" * 70)
    gen = generar_respuesta(args.pregunta, hits, llm=sintetizador)
    print(gen["respuesta"])
    print(f"\n(abstenida={gen['abstenida']}, fragmentos={len(gen['fragmentos_usados'])})")

    print("\n" + "=" * 70); print("VERIFICACION DE FIDELIDAD"); print("=" * 70)
    v = verificar_fidelidad(args.pregunta, gen["respuesta"], hits, llm=verificador)
    print(f"veredicto = {v['veredicto']} | fiel = {v['fiel']}")
    if v["afirmaciones_sin_respaldo"]:
        print("sin respaldo:", v["afirmaciones_sin_respaldo"])
    print("explicacion:", v["explicacion"])


if __name__ == "__main__":
    main()
