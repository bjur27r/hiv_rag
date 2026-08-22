"""Carga de PDF con PyMuPDF preservando pagina y bloques de texto.

Devuelve, por pagina, la lista de bloques (con su bounding-box) para que el
chunking pueda razonar sobre la estructura (encabezados, parrafos, tablas)
en lugar de partir el texto a ciegas por numero de tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Bloque:
    pagina: int          # 1-indexada
    texto: str
    y0: float            # posicion vertical (para ordenar/detectar encabezados)
    size_max: float      # tamanio de fuente maximo del bloque (heuristica de titulo)
    tipo: str = "texto"  # "texto" | "tabla"


def _dentro(bbox, tablas_bbox) -> bool:
    """True si el bloque cae dentro de la bbox de alguna tabla (para no duplicarlo)."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for tx0, ty0, tx1, ty1 in tablas_bbox:
        if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
            return True
    return False


def cargar_pdf(path: Path) -> list[Bloque]:
    bloques: list[Bloque] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            # 1) Tablas estructuradas (dosis, interacciones): se extraen como markdown.
            tablas_bbox = []
            try:
                tabs = page.find_tables()
                for tab in tabs.tables:
                    md = tab.to_markdown().strip()
                    if md and len(md) > 30:
                        bloques.append(Bloque(pagina=i, texto=md, y0=tab.bbox[1],
                                              size_max=0.0, tipo="tabla"))
                        tablas_bbox.append(tab.bbox)
            except Exception:
                pass  # si find_tables falla, seguimos solo con texto

            # 2) Texto, excluyendo lo que ya esta dentro de una tabla.
            data = page.get_text("dict")
            for blk in data.get("blocks", []):
                if blk.get("type") != 0:  # 0 = texto
                    continue
                if _dentro(blk["bbox"], tablas_bbox):
                    continue
                partes: list[str] = []
                size_max = 0.0
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        partes.append(span.get("text", ""))
                        size_max = max(size_max, span.get("size", 0.0))
                texto = " ".join(" ".join(partes).split())
                if texto:
                    bloques.append(Bloque(pagina=i, texto=texto, y0=blk["bbox"][1],
                                          size_max=size_max))
    return bloques
