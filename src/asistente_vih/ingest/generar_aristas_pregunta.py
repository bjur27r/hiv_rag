"""Aristas-pregunta estilo HopRAG (Sprint 3).

Por cada chunk genera con DeepSeek:
  - out-coming (4): preguntas que el chunk SUSCITA pero NO responde
    (los "cabos sueltos" que apuntan a otros chunks, tipicamente de OTRA guia)
  - in-coming (2): preguntas que el chunk SI responde

El casado out->in (Jaccard de keywords + coseno de embeddings) construye las
aristas logicas dirigidas chunk->chunk que el traversal Retrieve-Reason-Prune
recorre en modo escalado.

Con CHECKPOINT por linea (reanudable, a diferencia de extraccion_masiva).

Salidas: artifacts/preguntas_chunk.jsonl  (preguntas por chunk)
         artifacts/aristas_pregunta.jsonl (aristas casadas)   [--solo-casar]
         artifacts/embeddings/preguntas_small.npz

    python -m asistente_vih.ingest.generar_aristas_pregunta            # generar
    python -m asistente_vih.ingest.generar_aristas_pregunta --solo-casar
"""
from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np

from ..config import ARTIFACTS_DIR, CHUNKS_PATH, _cargar_dotenv
from ..retrieval.analyzer import analizar

PREGUNTAS_PATH = ARTIFACTS_DIR / "preguntas_chunk.jsonl"
ARISTAS_PATH = ARTIFACTS_DIR / "aristas_pregunta.jsonl"
EMB_PATH = ARTIFACTS_DIR / "embeddings" / "preguntas_small.npz"

MAX_CONC = 10
PROMPT = """Analiza este fragmento de una guia clinica GeSIDA de VIH y genera preguntas en español:

1. "out": exactamente 4 preguntas que el fragmento SUSCITA pero NO puede responder con su propio
   texto (cabos sueltos: interacciones que menciona sin detallar, poblaciones que remite a otra
   guia, condiciones que nombra sin desarrollar).
2. "in": exactamente 2 preguntas cuya respuesta SI esta contenida en el fragmento.

Responde SOLO JSON: {{"out": ["...","...","...","..."], "in": ["...","..."]}}

FRAGMENTO (guia {guia}, seccion {seccion}):
{texto}"""


async def _procesar(client, sem, c: dict) -> dict:
    async with sem:
        try:
            r = await client.chat.completions.create(
                model="deepseek-chat", temperature=0.3,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": PROMPT.format(
                    guia=c["guia"], seccion=c.get("seccion", "")[:60],
                    texto=c.get("texto", "")[:1600])}])
            d = json.loads(r.choices[0].message.content)
            return {"chunk_id": c["chunk_id"], "guia": c["guia"],
                    "out": [q for q in d.get("out", []) if isinstance(q, str)][:4],
                    "in": [q for q in d.get("in", []) if isinstance(q, str)][:2]}
        except Exception as e:
            return {"chunk_id": c["chunk_id"], "guia": c["guia"], "error": str(e)[:200]}


async def generar():
    import os
    from openai import AsyncOpenAI
    _cargar_dotenv()
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                         base_url="https://api.deepseek.com")

    hechos = set()
    if PREGUNTAS_PATH.exists():
        with open(PREGUNTAS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if not d.get("error"):
                        hechos.add(d["chunk_id"])
    chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8") if l.strip()]
    pendientes = [c for c in chunks if c["chunk_id"] not in hechos]
    print(f"Chunks: {len(chunks)} | ya hechos: {len(hechos)} | pendientes: {len(pendientes)}")

    sem = asyncio.Semaphore(MAX_CONC)
    LOTE = 100
    with open(PREGUNTAS_PATH, "a", encoding="utf-8") as out:
        for i in range(0, len(pendientes), LOTE):
            lote = pendientes[i:i + LOTE]
            res = await asyncio.gather(*[_procesar(client, sem, c) for c in lote])
            for r in res:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            errs = sum(1 for r in res if r.get("error"))
            print(f"  {min(i + LOTE, len(pendientes))}/{len(pendientes)} (errores lote: {errs})",
                  flush=True)


def casar():
    """Embebe todas las preguntas y casa out->in por similitud hibrida."""
    import openai
    _cargar_dotenv()
    client = openai.OpenAI()

    regs = [json.loads(l) for l in open(PREGUNTAS_PATH, encoding="utf-8")
            if l.strip() and "error" not in l[:120]]
    outs, ins = [], []   # (chunk_id, guia, pregunta, keywords)
    for r in regs:
        if r.get("error"):
            continue
        for q in r.get("out", []):
            outs.append((r["chunk_id"], r["guia"], q, set(analizar(q))))
        for q in r.get("in", []):
            ins.append((r["chunk_id"], r["guia"], q, set(analizar(q))))
    print(f"Preguntas: {len(outs)} out / {len(ins)} in")

    textos = [q for _, _, q, _ in outs] + [q for _, _, q, _ in ins]
    if EMB_PATH.exists():
        data = np.load(EMB_PATH, allow_pickle=False)
        M = data["mat"]
        assert M.shape[0] == len(textos), "embeddings desfasados; borra preguntas_small.npz"
    else:
        vecs = []
        for i in range(0, len(textos), 512):
            r = client.embeddings.create(input=textos[i:i + 512],
                                         model="text-embedding-3-small")
            vecs.extend(d.embedding for d in r.data)
            print(f"  emb {min(i + 512, len(textos))}/{len(textos)}", flush=True)
        M = np.array(vecs, dtype=np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-9)
        EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMB_PATH, mat=M, n_out=np.array([len(outs)]))

    V_out, V_in = M[:len(outs)], M[len(outs):]
    aristas = []
    for i, (cid, guia, q, kw) in enumerate(outs):
        cos = V_in @ V_out[i]
        top = np.argsort(-cos)[:8]
        mejor, mejor_sim = None, 0.0
        for j in top:
            cid_dst, guia_dst, q_dst, kw_dst = ins[j]
            if cid_dst == cid:
                continue
            jac = len(kw & kw_dst) / max(len(kw | kw_dst), 1)
            sim = 0.5 * (jac + float(cos[j]))
            if sim > mejor_sim:
                mejor, mejor_sim = j, sim
        if mejor is not None and mejor_sim >= 0.35:
            cid_dst, guia_dst, q_dst, _ = ins[mejor]
            aristas.append({"src": cid, "dst": cid_dst, "pregunta": q,
                            "pregunta_in": q_dst, "sim": round(mejor_sim, 3),
                            "inter_guia": guia != guia_dst,
                            "fila_emb": i})
    with open(ARISTAS_PATH, "w", encoding="utf-8") as f:
        for a in aristas:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    inter = sum(1 for a in aristas if a["inter_guia"])
    print(f"Aristas: {len(aristas)} ({inter} inter-guia, {100*inter/max(len(aristas),1):.0f}%)")
    print(f"Guardado en {ARISTAS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-casar", action="store_true")
    args = ap.parse_args()
    if args.solo_casar:
        casar()
    else:
        asyncio.run(generar())
