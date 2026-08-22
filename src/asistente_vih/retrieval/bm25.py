"""Recuperador lexico BM25 en Python puro sobre los chunks de la ingesta.

Suficiente para medir recall a esta escala (~2.700 chunks) sin servidor ni
dependencias. Implementa la formula BM25 clasica (Robertson/Sparck-Jones) con
indice invertido. Cuando se anada el denso, este BM25 es la rama lexica del
hibrido (fusion por RRF).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ..config import CHUNKS_PATH
from .analyzer import analizar
from .base import Hit, Retriever
from .recs import recomendaciones_de


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []           # metadatos del chunk
        self.doc_tokens: list[Counter] = []  # tf por doc
        self.doc_len: list[int] = []
        self.df: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[int]] = defaultdict(list)  # term -> doc ids
        self.avgdl: float = 0.0

    # ------------------------------------------------------------------ build
    @classmethod
    def from_jsonl(cls, ruta: Path | None = None, **kw) -> "BM25Retriever":
        r = cls(**kw)
        ruta = ruta or CHUNKS_PATH
        with open(ruta, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                r._add(c)
        r._finalize()
        return r

    def _add(self, c: dict) -> None:
        toks = analizar(c.get("texto", ""))
        tf = Counter(toks)
        i = len(self.docs)
        self.docs.append(c)
        self.doc_tokens.append(tf)
        self.doc_len.append(len(toks))
        for term in tf:
            self.df[term] += 1
            self.postings[term].append(i)

    def _finalize(self) -> None:
        n = len(self.docs)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        for term, df in self.df.items():
            # idf BM25 con suavizado (siempre positivo).
            self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = analizar(query)
        scores: dict[int, float] = defaultdict(float)
        for term in q:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id in self.postings[term]:
                tf = self.doc_tokens[doc_id][term]
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom
        mejores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        out: list[Hit] = []
        for doc_id, sc in mejores:
            c = self.docs[doc_id]
            out.append(Hit(chunk_id=c["chunk_id"], guia=c["guia"], seccion=c.get("seccion", ""),
                           pagina=c.get("pagina", 0), score=round(sc, 4), texto=c.get("texto", ""),
                           farmacos=c.get("farmacos", []), condiciones=c.get("condiciones", []),
                           umbrales=c.get("umbrales", []), ambito=c.get("ambito", []),
                           fecha_vigencia=c.get("fecha_vigencia", ""),
                           recomendaciones=recomendaciones_de(c["chunk_id"])))
        return out
