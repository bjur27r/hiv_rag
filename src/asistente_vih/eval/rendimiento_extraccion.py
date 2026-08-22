"""Rendimiento de extraccion por guia: ¿cuanto conocimiento normativo se pierde?

Compara los marcadores de grado de evidencia (A-I ... C-III) presentes en el
texto crudo del PDF con los que sobreviven en chunks.jsonl. Un rendimiento bajo
señala perdida silenciosa (tipicamente tablas mal extraidas). Es el mejor
predictor de donde el RAG no podra responder por mas que recupere bien.

    python -m asistente_vih.eval.rendimiento_extraccion
"""
from __future__ import annotations

import json
import re

from ..config import CHUNKS_PATH, DATA_DIR, GUIAS
from ..ingest.extraer_recomendaciones import _RX_GRADO


def _contar(texto: str) -> int:
    return len(_RX_GRADO.findall(texto))


def main():
    import pymupdf

    en_chunks: dict[str, int] = {g: 0 for g in GUIAS}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                if c["guia"] in en_chunks:
                    en_chunks[c["guia"]] += _contar(c.get("texto", ""))

    print(f"{'guia':<22} {'PDF':>6} {'chunks':>7} {'rendimiento':>12}")
    total_pdf = total_chunks = 0
    for codigo, meta in GUIAS.items():
        pdf = DATA_DIR / meta.archivo
        if not pdf.exists():
            print(f"{codigo:<22} {'(sin PDF en data/)':>26}")
            continue
        doc = pymupdf.open(pdf)
        n_pdf = sum(_contar(p.get_text()) for p in doc)
        doc.close()
        n_ch = en_chunks[codigo]
        total_pdf += n_pdf
        total_chunks += n_ch
        pct = f"{100 * n_ch / n_pdf:5.1f}%" if n_pdf else "  n/a"
        aviso = "  <-- PERDIDA" if n_pdf and n_ch / n_pdf < 0.8 else ""
        print(f"{codigo:<22} {n_pdf:>6} {n_ch:>7} {pct:>12}{aviso}")
    if total_pdf:
        print(f"{'TOTAL':<22} {total_pdf:>6} {total_chunks:>7} "
              f"{100 * total_chunks / total_pdf:11.1f}%")


if __name__ == "__main__":
    main()
