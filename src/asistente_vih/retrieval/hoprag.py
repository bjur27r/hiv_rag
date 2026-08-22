"""Traversal Retrieve-Reason-Prune sobre las aristas-pregunta (Sprint 3, HopRAG).

Recupera por las ARISTAS (cada arista es una pregunta-puente indexable),
razona el recorrido (un LLM barato — o similitud pura en la variante sin LLM —
elige que pregunta seguir desde cada chunk) y poda por Helpfulness:
    H = 1/2 * (similitud chunk-consulta + frecuencia de visita normalizada)
La frecuencia de visita premia los chunks donde CONVERGEN varios caminos.

Es el modo ESCALADO: pensado para dispararse desde el evaluador de suficiencia
en consultas multi-guia, no para toda consulta.
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from ..config import ARTIFACTS_DIR, CHUNKS_PATH
from .base import Hit, Retriever
from .recs import recomendaciones_de

ARISTAS_PATH = ARTIFACTS_DIR / "aristas_pregunta.jsonl"
EMB_PATH = ARTIFACTS_DIR / "embeddings" / "preguntas_small.npz"


class HopRAGTraversal(Retriever):
    def __init__(self, openai_client, usar_llm: bool = True,
                 modelo_traversal: str = "gpt-4o-mini"):
        self.client = openai_client
        self.usar_llm = usar_llm
        self.modelo = modelo_traversal

        self.aristas = [json.loads(l) for l in open(ARISTAS_PATH, encoding="utf-8") if l.strip()]
        data = np.load(EMB_PATH, allow_pickle=False)
        M, n_out = data["mat"], int(data["n_out"][0])
        # Embedding de cada arista = su pregunta out-coming.
        self.E = M[[a["fila_emb"] for a in self.aristas]]
        # in-questions por chunk (para la similitud chunk-consulta del Prune).
        self.M_in = M[n_out:]
        self.in_rows_por_chunk: Dict[str, list[int]] = {}
        fila = 0
        for l in open(ARTIFACTS_DIR / "preguntas_chunk.jsonl", encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            if d.get("error"):
                continue
            for _ in d.get("in", []):
                self.in_rows_por_chunk.setdefault(d["chunk_id"], []).append(fila)
                fila += 1

        self.salientes: Dict[str, list[int]] = {}
        for i, a in enumerate(self.aristas):
            self.salientes.setdefault(a["src"], []).append(i)

        self.chunks_db = {}
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    self.chunks_db[c["chunk_id"]] = c
        print(f"HopRAG: {len(self.aristas)} aristas, {len(self.salientes)} chunks con salida.")

    def _embed(self, texto: str):
        r = self.client.embeddings.create(input=[texto], model="text-embedding-3-small")
        v = np.array(r.data[0].embedding, dtype=np.float32)
        return v / max(np.linalg.norm(v), 1e-9)

    def _elegir_arista_llm(self, query: str, idxs: list[int]) -> int | None:
        """El LLM elige la pregunta-puente mas util para responder query (o ninguna)."""
        opciones = "\n".join(f"[{n}] {self.aristas[i]['pregunta'][:150]}"
                             for n, i in enumerate(idxs))
        try:
            r = self.client.chat.completions.create(
                model=self.modelo, temperature=0.0, max_tokens=4,
                messages=[{"role": "user", "content":
                           f"Pregunta objetivo: {query}\n\nPreguntas-puente disponibles:\n{opciones}\n\n"
                           f"¿Cual ayuda mas a responder la pregunta objetivo? Responde SOLO el numero, "
                           f"o -1 si ninguna ayuda."}])
            n = int(r.choices[0].message.content.strip().split()[0])
            return idxs[n] if 0 <= n < len(idxs) else None
        except Exception:
            return None

    def search(self, query: str, k: int = 10, n_hop: int = 3,
               top_k_aristas: int = 15, presupuesto_llm: int = 20) -> List[Hit]:
        qv = self._embed(query)

        # RETRIEVE: top aristas por similitud consulta<->pregunta-puente.
        sims = self.E @ qv
        top = np.argsort(-sims)[:top_k_aristas]
        cola = []
        visitas: Dict[str, int] = {}
        for i in top:
            dst = self.aristas[i]["dst"]
            visitas[dst] = visitas.get(dst, 0) + 1
            if dst not in cola:
                cola.append(dst)

        # REASON: BFS; en cada chunk se elige la arista saliente mas util.
        llm_usadas = 0
        for _ in range(n_hop):
            siguiente = []
            for cid in cola:
                idxs = self.salientes.get(cid, [])
                if not idxs:
                    continue
                elegido = None
                if self.usar_llm and llm_usadas < presupuesto_llm:
                    elegido = self._elegir_arista_llm(query, idxs[:8])
                    llm_usadas += 1
                if elegido is None:
                    sub = max(idxs, key=lambda i: float(self.E[i] @ qv))
                    if float(self.E[sub] @ qv) >= 0.35:
                        elegido = sub
                if elegido is None:
                    continue
                dst = self.aristas[elegido]["dst"]
                nuevo = dst not in visitas
                visitas[dst] = visitas.get(dst, 0) + 1
                if nuevo:
                    siguiente.append(dst)
            cola = siguiente
            if not cola:
                break

        if not visitas:
            return []

        # PRUNE: Helpfulness = sim(chunk, q) via in-questions + visitas normalizadas.
        total_vis = sum(visitas.values())
        puntuados = []
        for cid, nvis in visitas.items():
            rows = self.in_rows_por_chunk.get(cid)
            sim = float(np.max(self.M_in[rows] @ qv)) if rows else 0.0
            h = 0.5 * (sim + nvis / total_vis)
            puntuados.append((h, cid))
        puntuados.sort(reverse=True)

        hits = []
        for h_score, cid in puntuados[:k]:
            c = self.chunks_db.get(cid, {})
            hits.append(Hit(chunk_id=cid, guia=c.get("guia", "?"), seccion=c.get("seccion", ""),
                            pagina=c.get("pagina", 0), score=round(h_score, 4),
                            texto=c.get("texto", ""), farmacos=c.get("farmacos", []),
                            condiciones=c.get("condiciones", []), umbrales=c.get("umbrales", []),
                            ambito=c.get("ambito", []), fecha_vigencia=c.get("fecha_vigencia", ""),
                            recomendaciones=recomendaciones_de(cid)))
        return hits
