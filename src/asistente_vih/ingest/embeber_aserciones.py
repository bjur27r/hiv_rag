"""Embeddings de las aserciones del grafo (Fase 3b: pesos dinamicos).

Cada asercion se embebe (text-embedding-3-small, el mismo espacio que las
entidades) como "intervencion relacion resultado. descripcion". En consulta,
CatRAG modula el peso de las aristas que ENTRAN a cada asercion por su
similitud con la pregunta: el paso query-aware del diseno CatRAG.

Salida: artifacts/embeddings/aserciones_small.npz  (ids + matriz normalizada)

    python -m asistente_vih.ingest.embeber_aserciones
"""
from __future__ import annotations

import json

import numpy as np

from ..config import ARTIFACTS_DIR

TRIPLETS = ARTIFACTS_DIR / "triplets.jsonl"
OUT = ARTIFACTS_DIR / "embeddings" / "aserciones_small.npz"
MODELO = "text-embedding-3-small"
LOTE = 512


def _texto(a: dict) -> str:
    partes = [a.get("intervencion") or "", a.get("relacion_base") or "",
              a.get("resultado_esperado") or ""]
    desc = (a.get("descripcion_relacion") or "")[:300]
    return (" ".join(p for p in partes if p) + ". " + desc).strip()


def main():
    import openai
    from ..config import _cargar_dotenv
    _cargar_dotenv()
    client = openai.OpenAI()

    ids, textos = [], []
    with open(TRIPLETS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            for a in d.get("aserciones", []):
                aid = a.get("id_asercion")
                if aid:
                    ids.append(aid)
                    textos.append(_texto(a) or aid)

    print(f"Aserciones a embeber: {len(ids)}")
    vecs = []
    for i in range(0, len(textos), LOTE):
        r = client.embeddings.create(input=textos[i:i + LOTE], model=MODELO)
        vecs.extend(d.embedding for d in r.data)
        print(f"  {min(i + LOTE, len(textos))}/{len(textos)}")
    m = np.array(vecs, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=np.array(ids), mat=m)
    print(f"Guardado {OUT} ({m.shape})")


if __name__ == "__main__":
    main()
