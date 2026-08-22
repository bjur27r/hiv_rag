"""Recuperador con reordenado: recupera top-N con el base y reordena a top-k.

Mantiene la interfaz `Retriever`, asi que es intercambiable y se puede medir con
el mismo harness de recall que el BM25 pelado.
"""
from __future__ import annotations

from .base import Hit, Retriever
from .rerank import Reranker


class RerankingRetriever(Retriever):
    def __init__(self, base: Retriever, reranker: Reranker, top_n: int = 50):
        self.base = base
        self.reranker = reranker
        self.top_n = top_n

    def search(self, query: str, k: int = 10) -> list[Hit]:
        candidatos = self.base.search(query, k=max(self.top_n, k))
        return self.reranker.rerank(query, candidatos, k=k)
