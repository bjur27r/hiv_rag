"""Recuperador denso: coseno sobre los vectores Voyage cacheados.

A escala de ~2.700 chunks una busqueda exacta en numpy es instantanea y sin
servidor. En produccion este componente se sustituye por Chroma/Qdrant/OpenSearch
detras de la MISMA interfaz Retriever, sin tocar el resto del sistema.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import CHUNKS_PATH
from .base import Hit, Retriever
from .embeddings import cargar_indice, construir_indice, get_embedder
from .recs import recomendaciones_de


class DenseRetriever(Retriever):
    def __init__(self, embedder=None):
        self.embedder = embedder or get_embedder()
        construir_indice(self.embedder)  # idempotente
        self.ids, self.matriz = cargar_indice(self.embedder.modelo)
        self._meta = self._cargar_meta()

    def _cargar_meta(self) -> dict[str, dict]:
        meta = {}
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    meta[c["chunk_id"]] = c
        return meta

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.embedder.embed_query(query)          # ya normalizado
        scores = self.matriz @ q                       # coseno (vectores normalizados)
        idx = np.argsort(-scores)[:k]
        hits: list[Hit] = []
        for i in idx:
            c = self._meta[self.ids[i]]
            hits.append(Hit(chunk_id=c["chunk_id"], guia=c["guia"], seccion=c.get("seccion", ""),
                            pagina=c.get("pagina", 0), score=float(scores[i]), texto=c.get("texto", ""),
                            farmacos=c.get("farmacos", []), condiciones=c.get("condiciones", []),
                            umbrales=c.get("umbrales", []), ambito=c.get("ambito", []),
                            fecha_vigencia=c.get("fecha_vigencia", ""),
                            recomendaciones=recomendaciones_de(c["chunk_id"])))
        return hits
