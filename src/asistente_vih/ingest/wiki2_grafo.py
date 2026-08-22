"""Materializa los grafos 2Wiki desde la extraccion OpenIE (Fase B).

De artifacts/wiki2/openie_<split>.jsonl produce, por split:

  grafo_<split>.graphml        REIFICADO: Asercion como nodo de primera clase.
  grafo_<split>_plano.graphml  APLANADO (ablacion): la misma extraccion con
                               cada asercion colapsada a arista Entidad->Entidad
                               (estilo HippoRAG 2). Misma extraccion, cero coste
                               de API extra.
  emb_aserciones_<split>.npz   embedding de descripcion_relacion (ids + mat).
  emb_entidades_<split>.npz    embedding del nombre de entidad (ids + mat).

Esquema de aristas del reificado — REUTILIZA los tipos del grafo VIH para que
CatRAGRetriever funcione sin cambios (el nombre clinico es solo etiqueta):
  Asercion -> Entidad sujeto   tipo_relacion=HAS_INTERVENTION  (se invierte en consulta)
  Asercion -> Entidad objeto   tipo_relacion=HAS_OUTCOME       (idem)
  Asercion -> Chunk            tipo_relacion=EN_CHUNK          (direccion ya correcta)
  Entidad  -> Chunk            tipo_relacion=MENCIONADO_EN     (freq anti-hub)
  Entidad <-> Entidad          tipo_relacion=SINONIMO_DE       (coseno >= UMBRAL)

Prefijos de nodo: 'chunk:<titulo>', 'ent:<forma plegada>', id_asercion tal cual.
El objeto se materializa como entidad solo si aparece en la lista de entidades
del pasaje (los literales — fechas, oficios — viajan en la asercion, no como nodo).

    python -m asistente_vih.ingest.wiki2_grafo --split sondeo
    python -m asistente_vih.ingest.wiki2_grafo --split benchmark
"""
from __future__ import annotations

import argparse
import json
import re

import networkx as nx
import numpy as np
from openai import OpenAI

from ..config import ARTIFACTS_DIR

WIKI2_DIR = ARTIFACTS_DIR / "wiki2"
MODELO_EMB = "text-embedding-3-small"
# 0.80 = default de HippoRAG 2 (synonymy_edge_sim_threshold; su topk es 2047).
# Con 0.90 salian ~360 pares y el PPR se cortaba en cuanto variaba el nombre
# ("Edward III" vs "Edward III of England"); paridad medida el 2026-08-21.
UMBRAL_SINONIMIA = 0.80
TOPK_SINONIMIA = 100          # tope de vecinos por entidad (anti-explosion)
LOTE_EMB = 512


def _plegar(s: str) -> str:
    return " ".join(s.lower().split())


def _embeber(client: OpenAI, textos: list[str], ruta, ids: list[str]) -> np.ndarray:
    """Embebe con cache en npz: si la ruta existe con los mismos ids, reusa."""
    if ruta.exists():
        d = np.load(ruta, allow_pickle=False)
        if list(d["ids"]) == ids:
            return d["mat"]
    mat = np.zeros((len(textos), 1536), dtype=np.float32)
    for i in range(0, len(textos), LOTE_EMB):
        r = client.embeddings.create(input=textos[i:i + LOTE_EMB], model=MODELO_EMB)
        for j, dat in enumerate(r.data):
            mat[i + j] = dat.embedding
        if (i // LOTE_EMB) % 5 == 0:
            print(f"  emb {min(i + LOTE_EMB, len(textos))}/{len(textos)}")
    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
    np.savez_compressed(ruta, ids=np.array(ids), mat=mat)
    return mat


def _sinonimias(ids: list[str], mat: np.ndarray) -> list[tuple[str, str]]:
    """Pares de entidades con coseno >= UMBRAL, a lo sumo TOPK vecinos por
    entidad (por bloques, sin O(n^2) en RAM)."""
    pares = []
    for i0 in range(0, len(ids), 2000):
        bloque = mat[i0:i0 + 2000] @ mat.T          # (b, n)
        for bi, fila in enumerate(bloque):
            i = i0 + bi
            cand = np.nonzero(fila >= UMBRAL_SINONIMIA)[0]
            if len(cand) > TOPK_SINONIMIA:
                cand = cand[np.argsort(-fila[cand])[:TOPK_SINONIMIA]]
            for j in cand:
                if j > i:
                    pares.append((ids[i], ids[int(j)]))
    return pares


def construir(split: str, canonico: bool = False) -> None:
    """canonico=True: los nodos-entidad son las entradas del diccionario
    (ingest.wiki2_entidades); cada mencion se resuelve a su id canonico y las
    variantes de nombre colapsan en UN nodo. Sin aristas de sinonimia (la
    fusion las sustituye). Salida grafo_<split>_canon.graphml."""
    docs = [json.loads(l) for l in
            open(WIKI2_DIR / f"openie_{split}.jsonl", encoding="utf-8")]
    client = OpenAI()
    mapa: dict[str, str] = {}
    if canonico:
        mapa = json.loads((WIKI2_DIR / f"entidades_mapa_{split}.json").read_text(encoding="utf-8"))
        entradas = {e["id"]: e for e in
                    json.loads((WIKI2_DIR / f"entidades_{split}.json").read_text(encoding="utf-8"))}

    def nodo_ent(titulo: str, forma: str) -> str:
        if canonico:
            return mapa.get(f"{titulo}||{_plegar(forma)}", f"ent:{_plegar(forma)}")
        return f"ent:{_plegar(forma)}"

    # --- nodos y datos base
    g = nx.MultiDiGraph()
    plano = nx.MultiDiGraph()
    ent_forma: dict[str, str] = {}      # id de nodo -> forma superficial
    aser_ids, aser_txt = [], []
    for d in docs:
        chunk = f"chunk:{d['chunk_id']}"
        for G in (g, plano):
            G.add_node(chunk, tipo="Chunk", titulo=d["chunk_id"])
        ents_chunk = set()
        formas_chunk: dict[str, str] = {}   # id nodo -> forma en ESTE pasaje
        entidades = list(d["entidades"])
        if canonico:                     # el titulo tambien es mencion (alias)
            base = re.sub(r"\(.*?\)", " ", d["chunk_id"]).strip()
            entidades.append(base)
        for e in entidades:
            eid = nodo_ent(d["chunk_id"], e)
            nombre = entradas[eid]["nombre"] if canonico and eid in entradas else e
            ent_forma.setdefault(eid, nombre)
            ents_chunk.add(eid)
            formas_chunk[eid] = e
        for eid in ents_chunk:
            for G in (g, plano):
                G.add_node(eid, tipo="Entidad", forma=ent_forma[eid])
                G.add_edge(eid, chunk, tipo_relacion="MENCIONADO_EN")
        for a in d["aserciones"]:
            aid = a["id_asercion"]
            suj = nodo_ent(d["chunk_id"], a["sujeto"])
            obj = nodo_ent(d["chunk_id"], a["objeto"])
            aser_ids.append(aid)
            aser_txt.append(a["descripcion_relacion"] or
                            f"{a['sujeto']} {a['relacion_base']} {a['objeto']}")
            # reificado: Asercion nodo + roles + procedencia
            g.add_node(aid, tipo="Asercion", relacion_base=a["relacion_base"],
                       descripcion=a["descripcion_relacion"])
            g.add_edge(aid, chunk, tipo_relacion="EN_CHUNK")
            if suj in ent_forma:
                g.add_edge(aid, suj, tipo_relacion="HAS_INTERVENTION")
            if obj in ent_forma and obj != suj:
                g.add_edge(aid, obj, tipo_relacion="HAS_OUTCOME")
            # Reificacion n-aria completa: participantes mencionados en la
            # frase que no son sujeto/objeto (p.ej. "song by A **and B**")
            # tambien reciben rol — sin esto, la entidad extra queda sin
            # arista y el PPR no puede alcanzarla desde el hecho.
            frase = _plegar(a.get("descripcion_relacion") or "")
            for eid in ents_chunk - {suj, obj}:
                if _plegar(formas_chunk.get(eid, ent_forma[eid])) in frase:
                    g.add_edge(aid, eid, tipo_relacion="HAS_OUTCOME")
            # plano: Entidad->Entidad si ambos son entidades (estilo HippoRAG 2)
            if suj in ent_forma and obj in ent_forma and obj != suj:
                plano.add_edge(suj, obj, tipo_relacion=a["relacion_base"] or "REL")

    # --- embeddings
    print(f"{split}: {len(docs)} chunks, {len(ent_forma)} entidades, "
          f"{len(aser_ids)} aserciones")
    emb_aser = _embeber(client, aser_txt, WIKI2_DIR / f"emb_aserciones_{split}.npz",
                        aser_ids)
    if canonico:
        # los embeddings de entidad canonica son las fichas del diccionario
        # (emb_fichas_<split>.npz, ya construidas); sin sinonimia vectorial.
        sufijo = "_canon"
    else:
        ent_ids = sorted(ent_forma)
        emb_ent = _embeber(client, [ent_forma[e] for e in ent_ids],
                           WIKI2_DIR / f"emb_entidades_{split}.npz", ent_ids)
        pares = _sinonimias(ent_ids, emb_ent)       # sinonimia vectorial
        for u, v in pares:
            for G in (g, plano):
                if G.has_node(u) and G.has_node(v):
                    G.add_edge(u, v, tipo_relacion="SINONIMO_DE")
        print(f"sinonimia (coseno>={UMBRAL_SINONIMIA}): {len(pares)} pares")
        sufijo = ""

    nx.write_graphml(g, WIKI2_DIR / f"grafo_{split}{sufijo}.graphml")
    nx.write_graphml(plano, WIKI2_DIR / f"grafo_{split}{sufijo}_plano.graphml")
    print(f"OK grafo_{split}{sufijo}.graphml: {g.number_of_nodes()} nodos / "
          f"{g.number_of_edges()} aristas | plano: {plano.number_of_nodes()} / "
          f"{plano.number_of_edges()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["sondeo", "benchmark"], required=True)
    ap.add_argument("--canonico", action="store_true",
                    help="nodos-entidad = diccionario canonico (wiki2_entidades)")
    args = ap.parse_args()
    construir(args.split, canonico=args.canonico)


if __name__ == "__main__":
    main()
