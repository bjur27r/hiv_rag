"""Módulo de Recuperación Basada en Grafos (CatRAG).

Implementa el Symbolic Anchoring (Entity Linking vía FAISS) y el 
algoritmo de Paseo Aleatorio (Personalized PageRank) sobre el Grafo de Conocimiento.
"""

import json
import faiss
import numpy as np
import networkx as nx
from openai import OpenAI
from typing import List, Dict, Any

from ..config import ARTIFACTS_DIR, CHUNKS_PATH
from .base import Hit, Retriever
from .recs import recomendaciones_de

class CatRAGRetriever(Retriever):
    """El balance grafo/denso se controla con tres diales (via entorno):
    CATRAG_PESO_PASAJE (0.05): masa de la siembra densa de pasajes.
    CATRAG_DAMPING (0.5): difusion del PPR (mas alto = mas exploracion de grafo).
    CATRAG_DENSE_TOPN (300): cuantos pasajes siembra el denso.
    Extremos medidos: (0.05, 0.5) preciso-temprano tipo denso; (0, 0.85)
    cobertura profunda multi-guia tipo grafo puro."""

    def __init__(self, openai_client: OpenAI):
        import os
        self.client = openai_client
        # Defaults = punto balanceado medido (2026-08-15): @1 46%, @20 91%,
        # nivel4@20 68%. Extremos: (0.05,0.5,300) ~denso; (0,0.85,-) ~grafo puro.
        self.peso_pasaje = float(os.getenv("CATRAG_PESO_PASAJE", "0.015"))
        self.damping = float(os.getenv("CATRAG_DAMPING", "0.7"))
        self.dense_top_n = int(os.getenv("CATRAG_DENSE_TOPN", "100"))
        self.faiss_path = ARTIFACTS_DIR / "entities.faiss"
        self.faiss_map_path = ARTIFACTS_DIR / "entities_faiss_map.json"
        self.graph_path = ARTIFACTS_DIR / "knowledge_graph.graphml"

        self.index = None
        self.id_to_canon = []
        self.graph = None
        self.search_graph = None
        self.chunks_db = {}

        self._load_data()

    def _load_data(self):
        """Carga el índice FAISS, el Grafo y los textos de los fragmentos en memoria."""
        print("Cargando motor CatRAG...")
        # 1. Cargar FAISS y mapeo
        if self.faiss_path.exists() and self.faiss_map_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            with open(self.faiss_map_path, "r", encoding="utf-8") as f:
                self.id_to_canon = json.load(f)["id_to_canon"]

        # 2. Cargar Grafo NetworkX
        if self.graph_path.exists():
            self.graph = nx.read_graphml(str(self.graph_path))
            self.search_graph = self._construir_grafo_busqueda()
            self._preparar_ppr()

        # 3. Cargar textos originales (indexados por su chunk_id canonico)
        if CHUNKS_PATH.exists():
            try:
                with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        self.chunks_db[data["chunk_id"]] = data
            except Exception as e:
                print(f"Error cargando chunks_db: {e}")
                
        # 4. Cargar aserciones
        triplets_path = ARTIFACTS_DIR / "triplets.jsonl"
        if triplets_path.exists():
            try:
                with open(triplets_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        cid = data.get("chunk_id")
                        if cid and cid in self.chunks_db:
                            self.chunks_db[cid]["aserciones"] = data.get("aserciones", [])
            except Exception as e:
                print(f"Error cargando aserciones: {e}")
                    
        print(f"CatRAG listo: {len(self.id_to_canon)} entidades indexadas, Grafo con {len(self.graph.nodes) if self.graph else 0} nodos.")

    def _construir_grafo_busqueda(self) -> nx.DiGraph:
        """Grafo dirigido para el PPR, construido UNA vez.

        Invierte las aristas de reificacion (HAS_INTERVENTION / TARGET_POPULATION /
        HAS_OUTCOME, que en el GraphML van Asercion -> Entidad) para que la energia
        fluya Entidad -> Asercion -> Chunk. El atributo de arista es 'tipo_relacion'
        (el que escribe ingest/construir_grafo.py).
        """
        _INVERTIR = ("HAS_INTERVENTION", "TARGET_POPULATION", "HAS_OUTCOME")
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nodes(data=True))
        for u, v, data in self.graph.edges(data=True):
            tipo_rel = data.get("tipo_relacion", "")
            if tipo_rel == "CONTIENE_RECOMENDACION":
                # Fase 3b: la Recomendacion entra al flujo INVERTIDA
                # (Recomendacion -> Chunk): recibe masa de las entidades que
                # menciona y la vierte en su chunk. No drena masa del chunk.
                g.add_edge(v, u, weight=1.0)
                continue
            if tipo_rel == "MENCIONADA_EN_REC":
                g.add_edge(u, v, weight=1.0)   # Entidad -> Recomendacion
                continue
            if tipo_rel == "SINONIMO_DE":
                g.add_edge(u, v, weight=1.0)   # sinonimia: no dirigida
                g.add_edge(v, u, weight=1.0)
                continue
            if tipo_rel == "ANCLADA_A":
                # Capa conceptual: bidireccional Entidad <-> Concepto para que
                # una semilla conceptual fluya hacia sus entidades textuales.
                g.add_edge(u, v, weight=1.0)
                g.add_edge(v, u, weight=1.0)
            elif tipo_rel == "IS_A":
                # Subsuncion con peso reducido (evita uniformar el PPR).
                g.add_edge(u, v, weight=0.5)
                g.add_edge(v, u, weight=0.5)
            elif tipo_rel in _INVERTIR:
                # tp=True marca aristas de poblacion (para el boost por intencion)
                g.add_edge(v, u, weight=1.0, tp=(tipo_rel == "TARGET_POPULATION"))
            else:
                g.add_edge(u, v, weight=1.0)   # Asercion -> Chunk, Entidad <-> Chunk
        return g

    def _preparar_ppr(self):
        """Precomputa las estructuras scipy del PPR dinamico (una vez).

        Guarda la matriz de adyacencia estatica y, si existen los embeddings
        de aserciones (ingest.embeber_aserciones), el mapeo nodo-embedding
        para modular por consulta las aristas que ENTRAN a cada asercion.
        """
        self._ppr = None
        self._dense = None  # DenseRetriever perezoso para la siembra de pasajes
        # Frecuencia de cada entidad (nº de chunks donde aparece): la semilla
        # se divide por ella (anti-hub, como HippoRAG 2). Independiente de scipy:
        # debe existir aunque el PPR dinamico no este disponible.
        self._freq_ent: Dict[str, int] = {}
        for u, _, d in self.graph.edges(data=True):
            if d.get("tipo_relacion") == "MENCIONADO_EN":
                self._freq_ent[u] = self._freq_ent.get(u, 0) + 1
        try:
            import scipy.sparse as sp
        except ImportError:
            return
        nodos = list(self.search_graph.nodes())
        idx = {n: i for i, n in enumerate(nodos)}
        filas, cols, datos = [], [], []
        filas_tp, cols_tp, datos_tp = [], [], []
        for u, v, d in self.search_graph.edges(data=True):
            filas.append(idx[u]); cols.append(idx[v]); datos.append(float(d.get("weight", 1.0)))
            if d.get("tp"):
                filas_tp.append(idx[u]); cols_tp.append(idx[v]); datos_tp.append(1.0)
        n = len(nodos)
        A = sp.csr_matrix((datos, (filas, cols)), shape=(n, n))
        A_tp = sp.csr_matrix((datos_tp, (filas_tp, cols_tp)), shape=(n, n))
        # Columnas de chunks con umbrales numericos (FG/CD4/carga viral...)
        umbral_cols = np.array([idx[cid] for cid, c in self.chunks_db.items()
                                if c.get("umbrales") and cid in idx], dtype=int)
        self._ppr = {"nodos": nodos, "idx": idx, "A": A, "A_tp": A_tp,
                     "umbral_cols": umbral_cols, "aser_pos": None, "aser_emb": None}

        emb_path = ARTIFACTS_DIR / "embeddings" / "aserciones_small.npz"
        if emb_path.exists():
            data = np.load(emb_path, allow_pickle=False)
            fila_de = {aid: i for i, aid in enumerate(data["ids"])}
            _REIF = ("HAS_INTERVENTION", "TARGET_POPULATION", "HAS_OUTCOME")
            pos_nodo, pos_emb, aser_ents = [], [], []
            for n in nodos:
                if self.graph.nodes[n].get("tipo") == "Asercion" and n in fila_de:
                    pos_nodo.append(idx[n]); pos_emb.append(fila_de[n])
                    aser_ents.append([v for _, v, d in self.graph.out_edges(n, data=True)
                                      if d.get("tipo_relacion", "") in _REIF])
            if pos_nodo:
                self._ppr["aser_pos"] = np.array(pos_nodo)
                self._ppr["aser_emb"] = data["mat"][np.array(pos_emb)]
                self._ppr["aser_ents"] = aser_ents
            print(f"PPR dinamico: {len(pos_nodo)} aserciones con embedding.")


    def _semillas_por_hechos(self, qvec, top_m: int = 15,
                             link_top_k: int = 5) -> Dict[str, float]:
        """Siembra PRINCIPAL (query-to-fact de HippoRAG 2): las aserciones mas
        afines a la consulta aportan sus entidades como semillas, con peso
        = score del hecho / frecuencia de la entidad (anti-hub)."""
        if qvec is None or self._ppr.get("aser_emb") is None:
            return {}
        sims = self._ppr["aser_emb"] @ qvec
        smin, smax = float(sims.min()), float(sims.max())
        norm = (sims - smin) / (smax - smin + 1e-9)
        orden = np.argsort(-sims)[:top_m]
        pesos, cuenta = {}, {}
        for j in orden:
            for ent in self._ppr["aser_ents"][j]:
                freq = self._freq_ent.get(ent, 1) or 1
                pesos[ent] = pesos.get(ent, 0.0) + float(norm[j]) / freq
                cuenta[ent] = cuenta.get(ent, 0) + 1
        for ent in pesos:
            pesos[ent] /= cuenta[ent]
        top = sorted(pesos.items(), key=lambda x: -x[1])[:link_top_k]
        return dict(top)

    def _semillas_densas(self, query: str, top_n: int = 300,
                         peso_pasaje: float = 0.05) -> Dict[str, float]:
        """Siembra los nodos-chunk con su score denso normalizado x 0.05:
        el ranking denso entra DENTRO del PPR (fusion interna, HippoRAG 2)."""
        if self._dense is None:
            try:
                from .dense import DenseRetriever
                self._dense = DenseRetriever()
            except Exception as e:
                print(f"Siembra densa no disponible: {e}")
                self._dense = False
        if not self._dense:
            return {}
        hits = self._dense.search(query, k=top_n)
        if not hits:
            return {}
        s = np.array([h.score for h in hits], dtype=np.float32)
        mm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return {h.chunk_id: peso_pasaje * float(m)
                for h, m in zip(hits, mm) if self.search_graph.has_node(h.chunk_id)}

    def _embed_query(self, query: str):
        try:
            r = self.client.embeddings.create(input=[query], model="text-embedding-3-small")
            v = np.array(r.data[0].embedding, dtype=np.float32)
            return v / max(np.linalg.norm(v), 1e-9)
        except Exception:
            return None

    def _ppr_dinamico(self, personalization: Dict[str, float],
                      qvec=None, damping: float = 0.5,
                      intencion: Dict | None = None) -> Dict[str, float]:
        """PPR por potencias sobre scipy, con las aristas hacia cada asercion
        escaladas por su similitud con la consulta (paso query-aware CatRAG)
        y sesgo por INTENCION del router (patron MAGMA): consultas de paciente
        amplifican TARGET_POPULATION; las numericas, los chunks con umbrales.
        damping=0.5 como HippoRAG 2 (menos difusion = menos deriva a hubs)."""
        import scipy.sparse as sp
        nodos, idx, A = self._ppr["nodos"], self._ppr["idx"], self._ppr["A"]
        n = len(nodos)
        v = np.zeros(n)
        for nodo, peso in personalization.items():
            if nodo in idx:
                v[idx[nodo]] = peso
        if v.sum() == 0:
            return {}
        v /= v.sum()

        intencion = intencion or {}
        if intencion.get("tipo_consulta") == "paciente":
            A = (A + 0.5 * self._ppr["A_tp"]).tocsr()   # TARGET_POPULATION x1.5

        mult = np.ones(n)
        if self._ppr["aser_pos"] is not None and qvec is not None:
            sims = self._ppr["aser_emb"] @ qvec
            mult[self._ppr["aser_pos"]] = 0.25 + 0.75 * np.clip(sims, 0.0, 1.0)
        if intencion.get("logica_numerica") and len(self._ppr["umbral_cols"]):
            mult[self._ppr["umbral_cols"]] *= 1.3
        if not np.all(mult == 1.0):
            A = A.multiply(mult[None, :]).tocsr()

        salidas = np.asarray(A.sum(axis=1)).ravel()
        P = sp.diags(np.divide(1.0, salidas, out=np.zeros_like(salidas),
                               where=salidas > 0)) @ A
        colgantes = salidas == 0
        p = v.copy()
        for _ in range(80):
            masa_colgante = p[colgantes].sum()
            p_nuevo = damping * (P.T @ p + masa_colgante * v) + (1 - damping) * v
            if np.abs(p_nuevo - p).sum() < 1e-9:
                p = p_nuevo
                break
            p = p_nuevo
        return {nodos[i]: float(p[i]) for i in np.nonzero(p)[0]}

    def _extract_query_entities(self, query: str) -> List[str]:
        """Usa el LLM para extraer y EXPANDIR (sinónimos) las entidades clave de la pregunta."""
        prompt = (
            f"Extrae los conceptos médicos clave de esta consulta y para cada uno genera 2 sinónimos o variantes comunes. "
            f"Devuelve una lista plana separada por comas (ej. Fallo renal, Insuficiencia renal, Daño renal, Tuberculosis, TBC). "
            f"Consulta: {query}"
        )
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            entities = res.choices[0].message.content.split(",")
            return [e.strip() for e in entities if e.strip()]
        except Exception as e:
            print(f"Error extrayendo entidades: {e}")
            return [query] # Fallback a embedder toda la pregunta

    def _symbolic_anchoring(self, entities: List[str], top_k: int = 2) -> List[str]:
        """Busca TODAS las entidades y sus sinónimos en FAISS (Anclaje Multi-Semilla)."""
        if not self.index or not entities: return []
        
        # Calcular embeddings de TODAS las entidades expandidas
        res = self.client.embeddings.create(
            input=entities,
            model="text-embedding-3-small"
        )
        vectors = np.array([d.embedding for d in res.data], dtype=np.float32)
        
        # Buscar en FAISS top_k para cada sinónimo
        distances, indices = self.index.search(vectors, top_k)
        
        seed_nodes = set()
        for i, entity_matches in enumerate(indices):
            for match_idx in entity_matches:
                if match_idx >= 0 and match_idx < len(self.id_to_canon):
                    seed_nodes.add(self.id_to_canon[match_idx])
                    
        return list(seed_nodes)

    def _explain_path(self, chunk_id: str, seed_nodes: set) -> List[str]:
        """Extrae la traza de razonamiento desde las semillas hasta este chunk."""
        paths = []
        if not self.graph: return paths
        undir_g = self.graph.to_undirected()
        
        chunk_neighbors = set(undir_g.neighbors(chunk_id))
        
        for seed in seed_nodes:
            if not undir_g.has_node(seed): continue
            
            # Mención directa (largo 1)
            if chunk_id in undir_g.neighbors(seed):
                paths.append(f"Semilla [{seed}] ➔ Mención directa ➔ {chunk_id}")
                
            # Camino de largo 2 (Semilla -> Aserción -> Chunk)
            seed_neighbors = set(undir_g.neighbors(seed))
            common_neighbors = chunk_neighbors.intersection(seed_neighbors)
            
            for mid in common_neighbors:
                if self.graph.nodes[mid].get("tipo") == "Asercion":
                    rel = self.graph.nodes[mid].get("relacion_base", "REL")
                    paths.append(f"Semilla [{seed}] ➔ Aserción [{rel}] ➔ {chunk_id}")
                    
        return list(set(paths))

    def _semillas_conceptuales(self, query_entities: List[str]) -> Dict[str, float]:
        """Semillas de la capa SNOMED: ancla los terminos de la consulta al
        SCTID y activa el concepto + ancestros presentes en el grafo con peso
        decaido 1/2^distancia (expansion por subsuncion del plan)."""
        try:
            from ..terminologia.linker import get_linker
            lk = get_linker()
        except Exception:
            return {}
        pesos: Dict[str, float] = {}
        for ent in query_entities:
            try:
                cand = lk.anclar(ent)
            except Exception:
                continue
            if not (cand and cand.sctid):
                continue
            nodo = f"sct:{cand.sctid}"
            if self.search_graph.has_node(nodo):
                pesos[nodo] = max(pesos.get(nodo, 0.0), 1.0)
            for sctid, d in lk.ancestros(cand.sctid, max_saltos=2,
                                         max_descendientes_hub=300):
                nodo_a = f"sct:{sctid}"
                if self.search_graph.has_node(nodo_a):
                    pesos[nodo_a] = max(pesos.get(nodo_a, 0.0), 0.5 ** d)
        return pesos

    def search(self, query: str, k: int = 10, modo: str | None = None,
               intencion: Dict | None = None) -> List[Hit]:
        """Flujo principal: Entity Linking -> PPR -> Ranking de Chunks.

        modo: 'balanceado' (default; mejor precision temprana) o 'grafo'
        (cobertura profunda multi-guia: sin siembra densa, damping alto y
        anclas NER fuertes — el perfil que dio 84% nivel-4 @20). El router
        del agente lo elige por consulta; CATRAG_MODO lo fuerza en evals.
        """
        if not self.graph:
            print("Grafo no inicializado.")
            return []
        import os
        modo = modo or os.getenv("CATRAG_MODO") or "balanceado"
        if modo == "grafo":
            peso_pasaje, dense_top_n, damping, eps = 0.0, 0, 0.85, 1.0
        else:
            peso_pasaje, dense_top_n, damping, eps = (
                self.peso_pasaje, self.dense_top_n, self.damping, 0.05)

        # 1. Extraer entidades de la pregunta (para las anclas debiles)
        query_entities = self._extract_query_entities(query)
        qvec = self._embed_query(query) if getattr(self, "_ppr", None) else None

        # 2. Personalizacion en 4 capas (jerarquia de confianza HippoRAG2/CatRAG):
        #    a) PRINCIPAL query-to-fact: entidades de las aserciones afines,
        #       peso = score_hecho / frecuencia (anti-hub)
        #    b) pasajes con score denso x 0.05 (fusion interna)
        #    c) anclas simbolicas NER: peso debil epsilon (regularizacion)
        #    d) conceptos SNOMED + ancestros: peso moderado con decaimiento
        personalization: Dict[str, float] = dict(self._semillas_por_hechos(qvec))
        if peso_pasaje > 0 and dense_top_n > 0:
            for cid, w in self._semillas_densas(query, top_n=dense_top_n,
                                                peso_pasaje=peso_pasaje).items():
                personalization[cid] = personalization.get(cid, 0.0) + w
        anclas = [s for s in self._symbolic_anchoring(query_entities, top_k=2)
                  if self.search_graph.has_node(s)]
        for s in anclas:
            personalization[s] = personalization.get(s, 0.0) + eps
        for nodo, peso in self._semillas_conceptuales(query_entities).items():
            personalization[nodo] = max(personalization.get(nodo, 0.0), 0.1 * peso)
        if not personalization:
            return []

        # 3. Context-Aware Traversal: PPR dinamico (aristas hacia aserciones
        # moduladas por similitud con la consulta) con fallback al estatico.
        if getattr(self, "_ppr", None):
            ppr_scores = self._ppr_dinamico(personalization, qvec, damping=damping,
                                            intencion=intencion)
        else:
            ppr_scores = nx.pagerank(self.search_graph, alpha=damping,
                                     personalization=personalization, weight="weight")

        # 4. Filtrar y ordenar resultados (Nos interesan los Chunks, no las entidades)
        chunk_scores = []
        for node, score in ppr_scores.items():
            if self.graph.nodes[node].get("tipo") == "Chunk":
                chunk_scores.append((node, score))

        # Ordenar de mayor a menor probabilidad
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:k]
        
        # 5. Formatear como objetos Hit
        hits = []
        for rank, (chunk_id, score) in enumerate(top_chunks):
            data = self.chunks_db.get(chunk_id, {})
            texto = data.get("texto", "Texto no encontrado.")
            trace = self._explain_path(chunk_id, set(anclas))
            hit = Hit(
                chunk_id=chunk_id,
                guia=data.get("guia", "Desconocida"),
                seccion=data.get("seccion", ""),
                pagina=data.get("pagina", 0),
                score=float(score),
                texto=texto,
                farmacos=data.get("farmacos", []),
                condiciones=data.get("condiciones", []),
                umbrales=data.get("umbrales", []),
                ambito=data.get("ambito", []),
                aserciones=data.get("aserciones", []),
                reasoning_trace=trace,
                fecha_vigencia=data.get("fecha_vigencia", ""),
                recomendaciones=recomendaciones_de(chunk_id)
            )
            hits.append(hit)
            
        return hits
