"""Reordenadores (rerankers) detras de una interfaz comun.

  - HeuristicReranker: reordena el top-N usando las señales que YA extraemos
    (solape de farmacos/condiciones, intencion numerica + umbrales presentes,
    solape con el encabezado de seccion) combinadas con el score BM25 normalizado.
    Determinista, sin modelo, ejecutable ya. Es el reranker "barato".

  - CrossEncoderReranker: el reranker de manual (cross-encoder multilingue). Se
    enchufa detras de la MISMA interfaz; hace falta sentence-transformers y un
    modelo. Si no estan disponibles, falla con un mensaje claro (no a medias).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..ingest import metadata as md
from .analyzer import analizar
from .base import Hit

# Pesos del reordenador heuristico (ajustables / calibrables).
W_BASE = 1.0
W_ENTIDAD = 0.8
W_NUMERICO = 0.6
W_ENCABEZADO = 0.4


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        raise NotImplementedError


def _norm(valores: list[float]) -> list[float]:
    if not valores:
        return []
    lo, hi = min(valores), max(valores)
    if hi - lo < 1e-9:
        return [1.0 for _ in valores]
    return [(v - lo) / (hi - lo) for v in valores]


class HeuristicReranker(Reranker):
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        qf = set(md.extraer_farmacos(query))
        qc = set(md.extraer_condiciones(query))
        q_numerica = bool(md.extraer_umbrales(query)) or any(
            t in query.lower() for t in ("fg", "cd4", "carga viral", "copias", "filtrado", "aclaramiento"))
        q_enc = set(analizar(query))

        base = _norm([h.score for h in hits])
        nuevos = []
        for h, b in zip(hits, base):
            ent = len(qf & set(h.farmacos)) + len(qc & set(h.condiciones))
            num = 1.0 if (q_numerica and h.umbrales) else 0.0
            enc = len(q_enc & set(analizar(h.seccion)))
            s = (W_BASE * b + W_ENTIDAD * min(ent, 3) / 3
                 + W_NUMERICO * num + W_ENCABEZADO * min(enc, 3) / 3)
            nuevos.append((s, h))
        nuevos.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in nuevos[:k]]


def rerank_diverso(reranker: "Reranker", query: str, hits: list[Hit], k: int) -> list[Hit]:
    """Reordena con el reranker y luego GARANTIZA diversidad de guia en el
    top-k: primera pasada toma el mejor chunk de cada guia presente (en orden
    de score), el resto se rellena por score puro.

    Motivo (medido): el cross-encoder concentra chunks casi identicos de la
    misma guia arriba y expulsa la cobertura multi-guia del top-5, que es lo
    que la sintesis multi-aspecto necesita."""
    ordenados = reranker.rerank(query, hits, k=len(hits))
    sel, guias_vistas, en_sel = [], set(), set()
    for h in ordenados:
        if h.guia not in guias_vistas and len(sel) < k:
            sel.append(h)
            guias_vistas.add(h.guia)
            en_sel.add(h.chunk_id)
    for h in ordenados:
        if len(sel) >= k:
            break
        if h.chunk_id not in en_sel:
            sel.append(h)
            en_sel.add(h.chunk_id)
    return sel


class CrossEncoderReranker(Reranker):
    """Cross-encoder multilingue (p. ej. mmarco-mMiniLMv2). Requiere instalacion."""

    def __init__(self, modelo: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        try:
            from sentence_transformers import CrossEncoder  # noqa: lazy import
        except ImportError as e:
            raise RuntimeError(
                "CrossEncoderReranker requiere 'sentence-transformers'. Instala el extra "
                "o usa HeuristicReranker. La interfaz es la misma.") from e
        self._modelo = CrossEncoder(modelo)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        pares = [(query, h.texto) for h in hits]
        scores = self._modelo.predict(pares)
        orden = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        return [h for _, h in orden[:k]]
