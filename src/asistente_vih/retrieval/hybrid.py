"""Recuperador hibrido: fusion de BM25 (lexico) y denso (semantico) por RRF.

Reciprocal Rank Fusion combina los rankings sin necesidad de calibrar escalas
de score entre motores: cada documento suma 1/(c + rango) por cada lista en la
que aparece. Robusto y estandar. El denso aporta la semantica; BM25 conserva los
tokens exactos (farmacos, dosis, codigos) que el denso difumina.
"""
from __future__ import annotations

from dataclasses import replace

from .base import Hit, Retriever


def rrf(listas: list[list[Hit]], k: int, c: int = 60,
        pesos: list[float] | None = None) -> list[Hit]:
    pesos = pesos or [1.0] * len(listas)
    puntua: dict[str, float] = {}
    primero: dict[str, Hit] = {}
    for lista, w in zip(listas, pesos):
        for rango, h in enumerate(lista):
            puntua[h.chunk_id] = puntua.get(h.chunk_id, 0.0) + w / (c + rango)
            primero.setdefault(h.chunk_id, h)
    ordenados = sorted(puntua.items(), key=lambda x: x[1], reverse=True)[:k]
    # replace() conserva TODOS los campos del Hit (vigencia, recomendaciones,
    # aserciones, trazas) — reconstruirlo a mano los perdia silenciosamente.
    return [replace(primero[cid], score=round(sc, 5)) for cid, sc in ordenados]


class HybridRetriever(Retriever):
    def __init__(self, lexico: Retriever, denso: Retriever, pool: int = 50, c: int = 60,
                 peso_lexico: float = 1.0, peso_denso: float = 1.0):
        self.lexico = lexico
        self.denso = denso
        self.pool = pool
        self.c = c
        self.pesos = [peso_lexico, peso_denso]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        n = max(self.pool, k)
        return rrf([self.lexico.search(query, n), self.denso.search(query, n)],
                   k=k, c=self.c, pesos=self.pesos)
