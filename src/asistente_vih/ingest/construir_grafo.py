"""Script para construir el Grafo de Conocimiento Reificado y el Índice FAISS.

Lee el diccionario de entidades y las aserciones n-dimensionales extraídas.
1. Calcula los embeddings de todas las entidades únicas (text-embedding-3-small).
2. Construye un índice FAISS para búsqueda vectorial ultra-rápida.
3. Construye un grafo dirigido (NetworkX) con nodos de entidad, nodos de chunk, 
   y nodos de aserción (reificados), conectados por aristas semánticas.
4. Exporta el grafo a GraphML.
"""

import json
import os
import math
import numpy as np
import faiss
import networkx as nx
from openai import OpenAI
from ..config import ARTIFACTS_DIR

DICCIONARIO_PATH = ARTIFACTS_DIR / "entity_dictionary.json"
RELACIONES_PATH = ARTIFACTS_DIR / "triplets.jsonl"
FAISS_OUT_PATH = ARTIFACTS_DIR / "entities.faiss"
FAISS_MAP_OUT_PATH = ARTIFACTS_DIR / "entities_faiss_map.json"
GRAPH_OUT_PATH = ARTIFACTS_DIR / "knowledge_graph.graphml"

def get_embeddings(client: OpenAI, textos: list[str], batch_size: int = 1000) -> np.ndarray:
    all_embeddings = []
    total_batches = math.ceil(len(textos) / batch_size)
    
    for i in range(0, len(textos), batch_size):
        batch = textos[i : i + batch_size]
        print(f"Calculando embeddings... Lote {i//batch_size + 1}/{total_batches}")
        
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)
        
    return np.array(all_embeddings, dtype=np.float32)

def construir_indice_faiss(diccionario: dict) -> None:
    """Embebe los nombres canónicos y escribe entities.faiss + su mapa."""
    nombres_canonicos = list(diccionario.keys())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Define OPENAI_API_KEY en el entorno para calcular los embeddings.")
        return
    client = OpenAI(api_key=api_key)

    if nombres_canonicos:
        vectores = get_embeddings(client, nombres_canonicos)
        dimension = vectores.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(vectores)

        faiss.write_index(index, str(FAISS_OUT_PATH))
        with open(FAISS_MAP_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"id_to_canon": nombres_canonicos}, f, ensure_ascii=False)

        print(f"Índice FAISS guardado en {FAISS_OUT_PATH}")
    else:
        print("Advertencia: No hay entidades para indexar.")


def construir_grafo_nx(diccionario: dict) -> None:
    """Construye y exporta el grafo reificado. No requiere claves de API."""
    nombres_canonicos = list(diccionario.keys())
    print("\nConstruyendo Grafo de Conocimiento Reificado (NetworkX)...")
    G = nx.MultiDiGraph()
    
    # Añadir Nodos de Entidad
    for canon in nombres_canonicos:
        G.add_node(canon, tipo="Entidad", categoria=diccionario[canon].get("tipo_entidad", "Desconocido"))
        
    nodos_asercion = 0
    aristas_reificadas = 0
    aristas_contexto = 0
    
    if not RELACIONES_PATH.exists():
        print(f"ERROR: No se encontró el archivo de relaciones en {RELACIONES_PATH}")
        return

    with open(RELACIONES_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            if not linea.strip(): continue
            datos_chunk = json.loads(linea)
            chunk_id = datos_chunk["chunk_id"]
            
            # Añadir Nodo de Chunk
            if not G.has_node(chunk_id):
                G.add_node(chunk_id, tipo="Chunk")
            
            entidades_en_este_chunk = set()
            
            for aser in datos_chunk.get("aserciones", []):
                id_asercion = aser.get("id_asercion", f"{chunk_id}_X")
                
                # Crear el Nodo de Aserción Central
                G.add_node(
                    id_asercion,
                    tipo="Asercion",
                    tipo_asercion=aser.get("tipo_asercion", "EvidenciaClinica"),
                    evidence_level=str(aser.get("evidence_level", "None")),
                    is_negated=str(aser.get("is_negated", False)),
                    relacion_base=aser.get("relacion_base", "OTHER"),
                    descripcion_relacion=aser.get("descripcion_relacion", "")
                )
                nodos_asercion += 1
                
                # Conectar la Aserción con su Chunk de origen
                G.add_edge(id_asercion, chunk_id, tipo_relacion="EXTRAIDO_DE")
                aristas_contexto += 1
                
                # Conectar Aserción con Intervención, Población y Resultado
                intervencion = (aser.get("intervencion") or "").strip().title()
                poblacion = (aser.get("poblacion_diana") or "").strip().title()
                resultado = (aser.get("resultado_esperado") or "").strip().title()
                
                for entidad, tipo_arista in [
                    (intervencion, "HAS_INTERVENTION"),
                    (poblacion, "TARGET_POPULATION"),
                    (resultado, "HAS_OUTCOME")
                ]:
                    if entidad and entidad in diccionario:
                        entidades_en_este_chunk.add(entidad)
                        G.add_edge(id_asercion, entidad, tipo_relacion=tipo_arista)
                        aristas_reificadas += 1

            # Añadir Aristas de Contexto (Entidad <-> Chunk)
            for entidad in entidades_en_este_chunk:
                G.add_edge(entidad, chunk_id, tipo_relacion="MENCIONADO_EN")
                G.add_edge(chunk_id, entidad, tipo_relacion="CONTIENE_MENCION")
                aristas_contexto += 2

    # Nodos de RECOMENDACION (Fase 1): unidad normativa con polaridad y grado.
    recs_path = ARTIFACTS_DIR / "recomendaciones.jsonl"
    nodos_rec = 0
    if recs_path.exists():
        with open(recs_path, "r", encoding="utf-8") as f:
            for linea in f:
                if not linea.strip():
                    continue
                r = json.loads(linea)
                chunk_id = r["chunk_id"]
                if not G.has_node(chunk_id):
                    G.add_node(chunk_id, tipo="Chunk")
                G.add_node(r["id"], tipo="Recomendacion", guia=r.get("guia", ""),
                           accion_deontica=r.get("accion_deontica", "sin_determinar"),
                           grado=r.get("grado") or "",
                           fecha_vigencia=r.get("fecha_vigencia", ""))
                G.add_edge(chunk_id, r["id"], tipo_relacion="CONTIENE_RECOMENDACION")
                nodos_rec += 1
                # Enlazar entidades mencionadas en el TEXTO de la recomendacion
                # (Fase 3b: la recomendacion recibe masa del PPR via entidades).
                texto_rec = r.get("texto", "").lower()
                enlazadas = 0
                for canon in nombres_canonicos:
                    if len(canon) >= 5 and enlazadas < 12 and canon.lower() in texto_rec:
                        G.add_edge(canon, r["id"], tipo_relacion="MENCIONADA_EN_REC")
                        enlazadas += 1
    print(f"- Nodos de Recomendacion: {nodos_rec}")

    # Capa CONCEPTUAL (Fase 3): conceptos SNOMED anclados + ancestros IS_A.
    # Nodos con prefijo "sct:" para no colisionar con nombres de entidad.
    anclajes_path = ARTIFACTS_DIR / "anclajes.jsonl"
    nodos_concepto = aristas_ancla = aristas_isa = 0
    if anclajes_path.exists():
        try:
            from ..terminologia.linker import get_linker
            lk = get_linker()
        except Exception as e:
            print(f"Capa conceptual omitida (terminologia no disponible): {e}")
            lk = None
        if lk is not None:
            sctids = set()
            with open(anclajes_path, "r", encoding="utf-8") as f:
                for linea in f:
                    if not linea.strip():
                        continue
                    r = json.loads(linea)
                    sctid = r.get("sctid")
                    if not sctid or not G.has_node(r["entidad"]):
                        continue
                    nodo = f"sct:{sctid}"
                    if not G.has_node(nodo):
                        G.add_node(nodo, tipo="Concepto", sctid=sctid,
                                   nombre=lk.nombre_sctid(sctid), nivel=0)
                        nodos_concepto += 1
                    G.add_edge(r["entidad"], nodo, tipo_relacion="ANCLADA_A",
                               metodo=r.get("metodo", ""), score=float(r.get("score", 0)))
                    aristas_ancla += 1
                    sctids.add(sctid)
            for sctid in sctids:
                for padre, d in lk.ancestros(sctid, max_saltos=2,
                                             max_descendientes_hub=300):
                    nodo_p = f"sct:{padre}"
                    if not G.has_node(nodo_p):
                        G.add_node(nodo_p, tipo="Concepto", sctid=padre,
                                   nombre=lk.nombre_sctid(padre), nivel=d)
                        nodos_concepto += 1
                # aristas IS_A solo entre conceptos presentes en el grafo
            presentes = {n for n, dt in G.nodes(data=True) if dt.get("tipo") == "Concepto"}
            for nodo in presentes:
                sct = G.nodes[nodo]["sctid"]
                for padre, d in lk.ancestros(sct, max_saltos=1):
                    nodo_p = f"sct:{padre}"
                    if nodo_p in presentes and not G.has_edge(nodo, nodo_p):
                        G.add_edge(nodo, nodo_p, tipo_relacion="IS_A")
                        aristas_isa += 1
    print(f"- Nodos de Concepto (capa SNOMED): {nodos_concepto}")
    print(f"- Aristas ANCLADA_A: {aristas_ancla} | IS_A: {aristas_isa}")

    # Aristas de SINONIMIA entre entidades (Fase 3c, como HippoRAG 2):
    # (a) vectorial: cos >= 0.8 sobre el FAISS de entidades ya construido;
    # (b) verificada: entidades ancladas al MISMO SCTID (estrella por grupo).
    sin_vec = sin_sct = 0
    if FAISS_OUT_PATH.exists() and FAISS_MAP_OUT_PATH.exists():
        index = faiss.read_index(str(FAISS_OUT_PATH))
        with open(FAISS_MAP_OUT_PATH, "r", encoding="utf-8") as f:
            id_to_canon = json.load(f)["id_to_canon"]
        vecs = index.reconstruct_n(0, index.ntotal)
        # vectores OpenAI ya normalizados: L2^2 = 2 - 2*cos  =>  cos>=0.8 <=> L2^2 <= 0.4
        D, I = index.search(vecs, 6)
        for i in range(index.ntotal):
            u = id_to_canon[i]
            if not G.has_node(u):
                continue
            for dist, j in zip(D[i], I[i]):
                if j <= i or dist > 0.4:
                    continue
                v = id_to_canon[j]
                if G.has_node(v) and not G.has_edge(u, v):
                    G.add_edge(u, v, tipo_relacion="SINONIMO_DE")
                    sin_vec += 1
    if anclajes_path.exists():
        por_sctid = {}
        with open(anclajes_path, "r", encoding="utf-8") as f:
            for linea in f:
                if linea.strip():
                    r = json.loads(linea)
                    if r.get("sctid") and G.has_node(r["entidad"]):
                        por_sctid.setdefault(r["sctid"], []).append(r["entidad"])
        for ents in por_sctid.values():
            for otro in ents[1:]:
                if not G.has_edge(ents[0], otro):
                    G.add_edge(ents[0], otro, tipo_relacion="SINONIMO_DE")
                    sin_sct += 1
    print(f"- Aristas SINONIMO_DE: vectorial={sin_vec} mismo-SCTID={sin_sct}")

    # Exportar el grafo
    nx.write_graphml(G, str(GRAPH_OUT_PATH))
    
    print("\n¡Grafo Reificado construido con éxito!")
    print(f"- Nodos totales: {G.number_of_nodes()} (incluye entidades, aserciones y chunks)")
    print(f"- Nodos de Aserción: {nodos_asercion}")
    print(f"- Aristas de reificación: {aristas_reificadas}")
    print(f"- Aristas de contexto: {aristas_contexto}")
    print(f"- Grafo guardado en: {GRAPH_OUT_PATH}")


def main(solo_grafo: bool = False):
    try:
        with open('.env') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v
    except Exception:
        pass

    print("Cargando Diccionario de Entidades...")
    with open(DICCIONARIO_PATH, "r", encoding="utf-8") as f:
        diccionario = json.load(f)
    print(f"Total entidades: {len(diccionario)}")

    if not solo_grafo:
        construir_indice_faiss(diccionario)
    construir_grafo_nx(diccionario)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Construye el grafo reificado y el indice FAISS de entidades")
    ap.add_argument("--solo-grafo", action="store_true",
                    help="reconstruye solo el GraphML (sin re-embeber entidades; no requiere API)")
    main(solo_grafo=ap.parse_args().solo_grafo)
