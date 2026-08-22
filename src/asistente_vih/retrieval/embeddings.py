"""Embeddings densos — proveedor INTERCAMBIABLE (Anthropic no ofrece embeddings).

Soporta Voyage AI y OpenAI; se elige por entorno (EMBEDDINGS_PROVIDER) o por la
clave disponible. Toda la dependencia de proveedor se aisla aqui: cambiar de uno
a otro no toca ni el recuperador ni el resto del sistema.

Para este caso (RAG clinico en espanol) ambos son equivalentes en la practica;
ninguno tiene un modelo medico especifico, asi que se usa el general multilingue
de mayor calidad. Los vectores se calculan UNA vez y se cachean en disco
(artifacts/embeddings/) por nombre de modelo.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from ..config import ARTIFACTS_DIR, CHUNKS_PATH

EMB_DIR = ARTIFACTS_DIR / "embeddings"
_LOTE = 128  # limite de textos por llamada

# --- Cache en disco de embeddings de CONSULTA (acelera/abarata re-evaluaciones) ---
_QC: dict[str, dict[str, np.ndarray]] = {}
_QC_DIRTY: set[str] = set()


def _qc_path(modelo: str) -> Path:
    return EMB_DIR / f"qcache_{modelo.replace('/', '_')}.pkl"


def _qc(modelo: str) -> dict[str, np.ndarray]:
    if modelo not in _QC:
        p = _qc_path(modelo)
        _QC[modelo] = pickle.loads(p.read_bytes()) if p.exists() else {}
    return _QC[modelo]


def cached_query(modelo: str, texto: str, compute) -> np.ndarray:
    cache = _qc(modelo)
    h = hashlib.sha1(texto.encode("utf-8")).hexdigest()
    v = cache.get(h)
    if v is None:
        v = compute()
        cache[h] = v
        _QC_DIRTY.add(modelo)
    return v


@atexit.register
def _qc_flush() -> None:
    if not _QC_DIRTY:
        return
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    for m in _QC_DIRTY:
        _qc_path(m).write_bytes(pickle.dumps(_QC[m]))


def _normalizar(vecs: list[list[float]]) -> np.ndarray:
    m = np.asarray(vecs, dtype=np.float32)
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return m / norm  # coseno == producto escalar


class VoyageEmbedder:
    def __init__(self, modelo: str | None = None):
        self.modelo = modelo or os.getenv("VOYAGE_MODELO", "voyage-3.5")
        self._cliente = None

    def _c(self):
        if self._cliente is None:
            try:
                import voyageai
            except ImportError as e:
                raise RuntimeError("Falta 'voyageai'. pip install voyageai") from e
            if not os.getenv("VOYAGE_API_KEY"):
                raise RuntimeError("Falta VOYAGE_API_KEY en el entorno.")
            self._cliente = voyageai.Client()
        return self._cliente

    def _embed(self, textos: list[str], input_type: str) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(textos), _LOTE):
            res = self._c().embed(textos[i:i + _LOTE], model=self.modelo, input_type=input_type)
            vecs.extend(res.embeddings)
        return _normalizar(vecs)

    def embed_documents(self, textos: list[str]) -> np.ndarray:
        return self._embed(textos, "document")

    def embed_query(self, texto: str) -> np.ndarray:
        return cached_query(self.modelo, texto, lambda: self._embed([texto], "query")[0])


class OpenAIEmbedder:
    def __init__(self, modelo: str | None = None):
        self.modelo = modelo or os.getenv("OPENAI_EMBED_MODELO", "text-embedding-3-large")
        self._cliente = None

    def _c(self):
        if self._cliente is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError("Falta 'openai'. pip install openai") from e
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")
            self._cliente = OpenAI()
        return self._cliente

    def _embed(self, textos: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(textos), _LOTE):
            res = self._c().embeddings.create(model=self.modelo, input=textos[i:i + _LOTE])
            vecs.extend(d.embedding for d in res.data)
        return _normalizar(vecs)

    def embed_documents(self, textos: list[str]) -> np.ndarray:
        return self._embed(textos)

    def embed_query(self, texto: str) -> np.ndarray:
        return cached_query(self.modelo, texto, lambda: self._embed([texto])[0])


def get_embedder(modelo: str | None = None):
    """Factoria. Proveedor por defecto: OpenAI. Voyage queda como alternativa opcional.

    Se puede forzar con EMBEDDINGS_PROVIDER=voyage; si esta vacio, se usa OpenAI
    (salvo que solo haya VOYAGE_API_KEY y no OPENAI_API_KEY).
    """
    prov = os.getenv("EMBEDDINGS_PROVIDER", "").lower()
    if not prov:
        if not os.getenv("OPENAI_API_KEY") and os.getenv("VOYAGE_API_KEY"):
            prov = "voyage"
        else:
            prov = "openai"  # por defecto
    if prov == "openai":
        return OpenAIEmbedder(modelo)
    if prov == "voyage":
        return VoyageEmbedder(modelo)
    raise ValueError(f"EMBEDDINGS_PROVIDER desconocido: {prov!r} (usa 'openai' o 'voyage')")


def ruta_indice(modelo: str) -> Path:
    seguro = modelo.replace("/", "_")
    return EMB_DIR / f"{seguro}.npz"


def construir_indice(embedder=None, chunks_path: Path | None = None) -> Path:
    """Calcula y cachea los vectores de todos los chunks. Idempotente."""
    embedder = embedder or get_embedder()
    chunks_path = chunks_path or CHUNKS_PATH
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    destino = ruta_indice(embedder.modelo)
    if destino.exists():
        print(f"  Indice denso ya existe: {destino}")
        return destino

    ids, textos = [], []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                ids.append(c["chunk_id"])
                textos.append(c.get("texto", ""))
    print(f"  Embebiendo {len(textos)} chunks con Voyage ({embedder.modelo})...")
    matriz = embedder.embed_documents(textos)
    np.savez_compressed(destino, ids=np.array(ids, dtype=object), vectores=matriz)
    print(f"  Guardado: {destino}  ({matriz.shape})")
    return destino


def cargar_indice(modelo: str) -> tuple[list[str], np.ndarray]:
    datos = np.load(ruta_indice(modelo), allow_pickle=True)
    return list(datos["ids"]), datos["vectores"]


if __name__ == "__main__":  # python -m asistente_vih.retrieval.embeddings -> construye el indice denso
    construir_indice()
