"""Harness de recall@k de RECUPERACION sobre el banco de 505 preguntas.

Mide, sin LLM, si el recuperador trae el material correcto. Dos senales:

  - recall_guia@k  : el top-k cubre las guias esperadas (señal robusta).
                     Se reporta en version 'todas' (estricta: cubre TODAS las
                     guias implicadas) y 'alguna' (laxa: al menos una).
  - recall_secc@k  : entre las preguntas con anclaje de seccion NUMERICO, el
                     top-k trae un chunk de la guia esperada cuya seccion
                     coincide por prefijo (señal mas estricta).

AVISO: el patron oro del banco esta 'Pendiente' de validacion clinica y el
anclaje es a veces tematico (no numerico). Por eso esto es una señal de
recuperacion APROXIMADA, util para iterar el chunking y como linea base antes
de anadir denso + reranker; no es una metrica de correccion clinica.

    python -m asistente_vih.eval.recall
    python -m asistente_vih.eval.recall --k 1 5 10 20
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from ..config import EVAL_DATASET_PATH
from ..retrieval import BM25Retriever
from ..retrieval.analyzer import fold

# Etiqueta de guia (banco) -> codigo de guia (config), comparado por texto plegado.
_LABEL_A_CODIGO = {
    "tar adultos (2022)": "TAR_ADULTOS_2022",
    "vacunas/prevencion": "PREVENTIVA_VACUNAS",
    "tb-vih": "TB_VIH",
    "embarazo/transmision vertical": "EMBARAZO",
    "profilaxis posexposicion (pep)": "PEP",
    "metabolico/riesgo cv": "RIESGO_CV_METAB",
    "neurocognitivo (hand)": "NEUROCOGNITIVO_2013",
    "adherencia": "ADHERENCIA_2020",
}
_SECC_NUM = re.compile(r"§\s*(\d+(?:\.\d+)*)")
_CHUNK_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def guias_esperadas(label: str) -> set[str]:
    out = set()
    for parte in label.split("+"):
        cod = _LABEL_A_CODIGO.get(fold(parte).strip())
        if cod:
            out.add(cod)
    return out


def seccion_anclaje(anclaje: str) -> str | None:
    m = _SECC_NUM.search(anclaje or "")
    return m.group(1) if m else None


def seccion_chunk(seccion: str) -> str | None:
    m = _CHUNK_NUM.match(seccion or "")
    return m.group(1) if m else None


def cargar_banco() -> list[dict]:
    return [json.loads(l) for l in open(EVAL_DATASET_PATH, encoding="utf-8") if l.strip()]


def _pct(n, d):
    return f"{100*n/d:5.1f}%" if d else "  n/a"


def _muestra_estratificada(banco: list[dict], n: int) -> list[dict]:
    """Toma n preguntas repartidas por nivel, round-robin determinista."""
    from collections import deque
    por_nivel = defaultdict(deque)
    for ej in banco:
        por_nivel[ej["metadata"].get("nivel")].append(ej)
    niveles = sorted(por_nivel, key=lambda x: (x is None, x))
    out = []
    while len(out) < n and any(por_nivel[nv] for nv in niveles):
        for nv in niveles:
            if por_nivel[nv] and len(out) < n:
                out.append(por_nivel[nv].popleft())
    return out


def medir(retriever, banco: list[dict], ks: list[int],
          registro_path=None, etiqueta: str = "") -> dict:
    """Evalua un retriever y devuelve los acumuladores de recall.

    Con registro_path, vuelca ademas un JSONL de DIAGNOSTICO POR PREGUNTA:
    rango del primer acierto de guia y de seccion (para MRR y taxonomia de
    fallos), guias del top-20 y causa aproximada del fallo.
    """
    kmax = max(ks)
    freg = open(registro_path, "a", encoding="utf-8") if registro_path else None
    m = {
        "guia_alguna": {k: 0 for k in ks},
        "guia_todas": {k: 0 for k in ks},
        "secc_hit": {k: 0 for k in ks},
        "por_nivel": defaultdict(lambda: {k: 0 for k in ks}),
        "n_nivel": defaultdict(int),
        "n_con_guia": 0, "n_con_secc": 0,
    }
    for ej in banco:
        meta = ej["metadata"]
        nivel = meta.get("nivel")
        esperadas = guias_esperadas(meta.get("guias_implicadas", ""))
        secc = seccion_anclaje(ej["outputs"].get("anclaje", ""))
        m["n_nivel"][nivel] += 1
        if esperadas:
            m["n_con_guia"] += 1
        if secc and esperadas:
            m["n_con_secc"] += 1

        hits = retriever.search(ej["inputs"]["pregunta"], k=kmax)
        for k in ks:
            topk = hits[:k]
            guias_top = {h.guia for h in topk}
            if esperadas:
                if esperadas & guias_top:
                    m["guia_alguna"][k] += 1
                if esperadas <= guias_top:
                    m["guia_todas"][k] += 1
                    m["por_nivel"][nivel][k] += 1
            if secc and esperadas and any(
                    h.guia in esperadas and (seccion_chunk(h.seccion) or "").startswith(secc)
                    for h in topk):
                m["secc_hit"][k] += 1

        if freg is not None:
            rango_guia = rango_secc = None
            for i, h in enumerate(hits, start=1):
                if rango_guia is None and h.guia in esperadas:
                    rango_guia = i
                if (rango_secc is None and secc and h.guia in esperadas
                        and (seccion_chunk(h.seccion) or "").startswith(secc)):
                    rango_secc = i
            if not esperadas:
                causa = "sin_gold"
            elif rango_guia is None:
                causa = "guia_no_recuperada"
            elif secc and rango_secc is None:
                causa = "seccion_no_recuperada"
            elif rango_guia > 8:
                causa = "orden_tardio"   # esta en el pool pero fuera de lo que lee la sintesis
            else:
                causa = "ok"
            freg.write(json.dumps({
                "config": etiqueta, "id": ej.get("id"), "nivel": nivel,
                "pregunta": ej["inputs"]["pregunta"][:200],
                "guias_esperadas": sorted(esperadas), "seccion_esperada": secc,
                "rango_primer_acierto_guia": rango_guia,
                "rango_primer_acierto_seccion": rango_secc,
                "guias_top20": sorted({h.guia for h in hits[:20]}),
                "causa": causa}, ensure_ascii=False) + "\n")
    if freg is not None:
        freg.close()
    return m


def imprimir(nombre: str, m: dict, ks: list[int]) -> None:
    print(f"\n================  {nombre}  ================")
    print("recall_guia@k         alguna |   todas")
    for k in ks:
        print(f"  k={k:<3} {_pct(m['guia_alguna'][k], m['n_con_guia'])} | "
              f"{_pct(m['guia_todas'][k], m['n_con_guia'])}")
    print("recall_secc@k (anclaje numerico)")
    for k in ks:
        print(f"  k={k:<3} {_pct(m['secc_hit'][k], m['n_con_secc'])}")
    print("recall_guia@k 'todas' por nivel")
    print("  nivel | " + " | ".join(f"k={k}".rjust(7) for k in ks))
    for niv in sorted(x for x in m["n_nivel"] if x is not None):
        fila = " | ".join(_pct(m["por_nivel"][niv][k], m["n_nivel"][niv]).rjust(7) for k in ks)
        print(f"  {niv:>5} | {fila}   (n={m['n_nivel'][niv]})")


def main():
    ap = argparse.ArgumentParser(description="recall@k de recuperacion sobre el banco")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--rerank", action="store_true",
                    help="ademas de BM25, evalua BM25 + reordenador heuristico y compara")
    ap.add_argument("--cross-encoder", action="store_true",
                    help="usa el cross-encoder neuronal en lugar del heuristico (requiere sentence-transformers)")
    ap.add_argument("--dense", action="store_true", help="evalua el denso (embeddings)")
    ap.add_argument("--hybrid", action="store_true", help="evalua el hibrido BM25+denso (RRF)")
    ap.add_argument("--hybrid-denso", action="store_true", help="hibrido ponderado hacia el denso (1:3)")
    ap.add_argument("--rerank-dense", action="store_true", help="denso + reordenador heuristico")
    ap.add_argument("--decompose", action="store_true", help="descomposicion de consulta sobre el denso (usa LLM router)")
    ap.add_argument("--nivel", type=int, choices=[1, 2, 3, 4], help="evalua solo un nivel (subconjunto barato)")
    ap.add_argument("--no-bm25", action="store_true", help="omite la linea base BM25")
    ap.add_argument("--catrag", action="store_true",
                    help="evalua el recuperador de grafo CatRAG (usa LLM + embeddings por consulta)")
    ap.add_argument("--fusion-grafo", action="store_true",
                    help="evalua la fusion RRF denso+CatRAG (la rama 'paralela' del agente)")
    ap.add_argument("--rerank-fusion", action="store_true",
                    help="fusion denso+CatRAG + cross-encoder (la config de produccion del Sprint 1)")
    ap.add_argument("--por-pregunta", metavar="RUTA",
                    help="vuelca diagnostico por pregunta (JSONL) a la ruta dada")
    ap.add_argument("--hoprag", action="store_true",
                    help="traversal de aristas-pregunta SIN LLM (ablacion C2)")
    ap.add_argument("--hoprag-llm", action="store_true",
                    help="traversal con LLM eligiendo el salto (ablacion C3)")
    ap.add_argument("--limit", type=int,
                    help="muestra estratificada por nivel de N preguntas (determinista)")
    args = ap.parse_args()
    ks = sorted(set(args.k))

    print("Construyendo indice BM25...")
    bm25 = BM25Retriever.from_jsonl()
    print(f"  {len(bm25.docs)} chunks indexados.")
    banco = cargar_banco()
    if args.nivel:
        banco = [e for e in banco if e["metadata"].get("nivel") == args.nivel]
    if args.limit and args.limit < len(banco):
        banco = _muestra_estratificada(banco, args.limit)
    print(f"Preguntas: {len(banco)}" + (f" (solo Nivel {args.nivel})" if args.nivel else ""))

    if not args.no_bm25:
        imprimir("BM25 (linea base)", medir(bm25, banco, ks), ks)

    necesita_denso = args.dense or args.hybrid or args.hybrid_denso or args.rerank_dense or args.decompose
    if necesita_denso:
        from ..retrieval.dense import DenseRetriever
        denso = DenseRetriever()
        if args.dense:
            imprimir("Denso", medir(denso, banco, ks), ks)
        if args.hybrid:
            from ..retrieval.hybrid import HybridRetriever
            imprimir("Hibrido BM25+denso (RRF 1:1)", medir(HybridRetriever(bm25, denso), banco, ks), ks)
        if args.hybrid_denso:
            from ..retrieval.hybrid import HybridRetriever
            h = HybridRetriever(bm25, denso, peso_lexico=1.0, peso_denso=3.0)
            imprimir("Hibrido ponderado al denso (1:3)", medir(h, banco, ks), ks)
        if args.rerank_dense:
            from ..retrieval.pipeline import RerankingRetriever
            from ..retrieval.rerank import HeuristicReranker
            rr = RerankingRetriever(denso, HeuristicReranker(), top_n=50)
            imprimir("Denso + reordenador heuristico", medir(rr, banco, ks), ks)
        if args.decompose:
            from ..retrieval.decompose import DecomposingRetriever
            imprimir("Denso + descomposicion de consulta", medir(DecomposingRetriever(denso), banco, ks), ks)

    reg = args.por_pregunta

    if args.catrag or args.fusion_grafo or args.rerank_fusion:
        import openai
        from ..retrieval.catrag_retriever import CatRAGRetriever
        catrag = CatRAGRetriever(openai.OpenAI())
        if args.catrag:
            imprimir("CatRAG (grafo + PPR)",
                     medir(catrag, banco, ks, reg, "catrag"), ks)
        if args.fusion_grafo or args.rerank_fusion:
            from ..retrieval.dense import DenseRetriever
            from ..retrieval.hybrid import HybridRetriever
            h = HybridRetriever(DenseRetriever(), catrag, pool=30)
            if args.fusion_grafo:
                imprimir("Fusion denso+CatRAG (RRF 1:1)",
                         medir(h, banco, ks, reg, "fusion"), ks)
            if args.rerank_fusion:
                from ..retrieval.pipeline import RerankingRetriever
                from ..retrieval.rerank import CrossEncoderReranker, rerank_diverso

                class _Diverso:
                    def __init__(self, rr): self.rr = rr
                    def rerank(self, q, hits, k):
                        return rerank_diverso(self.rr, q, hits, k)

                rr = RerankingRetriever(h, _Diverso(CrossEncoderReranker()), top_n=30)
                imprimir("Fusion + cross-encoder diverso (Sprint 1)",
                         medir(rr, banco, ks, reg, "rerank_fusion_diverso"), ks)

    if args.hoprag or args.hoprag_llm:
        import openai
        from ..retrieval.hoprag import HopRAGTraversal
        cli = openai.OpenAI()
        if args.hoprag:
            imprimir("HopRAG traversal (sin LLM)",
                     medir(HopRAGTraversal(cli, usar_llm=False), banco, ks, reg, "hoprag_sim"), ks)
        if args.hoprag_llm:
            imprimir("HopRAG traversal (LLM)",
                     medir(HopRAGTraversal(cli, usar_llm=True), banco, ks, reg, "hoprag_llm"), ks)

    if args.rerank or args.cross_encoder:
        from ..retrieval.pipeline import RerankingRetriever
        if args.cross_encoder:
            from ..retrieval.rerank import CrossEncoderReranker
            rr, etq = CrossEncoderReranker(), "BM25 + cross-encoder"
        else:
            from ..retrieval.rerank import HeuristicReranker
            rr, etq = HeuristicReranker(), "BM25 + reordenador heuristico"
        imprimir(etq, medir(RerankingRetriever(bm25, rr, top_n=50), banco, ks), ks)


if __name__ == "__main__":
    main()
