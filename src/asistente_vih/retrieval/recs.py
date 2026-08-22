"""Cargador cacheado de recomendaciones normativas por chunk (Fase 1).

Los retrievers lo usan para adjuntar a cada Hit las recomendaciones (polaridad
deontica + grado) extraidas de su chunk, de modo que la generacion reciba la
normatividad etiquetada y no solo texto plano.
"""
from __future__ import annotations

import json
from functools import lru_cache

from ..config import ARTIFACTS_DIR

RECS_PATH = ARTIFACTS_DIR / "recomendaciones.jsonl"


@lru_cache(maxsize=1)
def _por_chunk() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not RECS_PATH.exists():
        return out
    with open(RECS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["chunk_id"], []).append(r)
    return out


def recomendaciones_de(chunk_id: str) -> list[dict]:
    return _por_chunk().get(chunk_id, [])
