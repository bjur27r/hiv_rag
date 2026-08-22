"""Capa de recuperacion (pluggable).

En Fase 1 se entrega un recuperador lexico BM25 con analizador en espanol y la
interfaz `Retriever`, para poder MEDIR recall de recuperacion sobre el banco
antes de decidir el motor de produccion. El recuperador denso y el hibrido se
anaden detras de la misma interfaz cuando haya embeddings disponibles.
"""
from .base import Hit, Retriever
from .bm25 import BM25Retriever

__all__ = ["Hit", "Retriever", "BM25Retriever"]
