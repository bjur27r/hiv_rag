"""CatRAG portado al benchmark 2Wiki (Fase C): PPR sobre el grafo reificado.

Port autocontenido del nucleo de CatRAGRetriever (mismas mecanicas: siembra
query-to-fact anti-hub, siembra densa de pasajes, anclas simbolicas debiles,
PPR por potencias con modulacion de aserciones por consulta) sobre los
artefactos de artifacts/wiki2/. No toca el codigo de produccion VIH.

Variantes de modulacion (sondeo Fase C):
  lite           coseno de la pregunta ENTERA contra cada asercion (como produccion).
  descomposicion la pregunta se descompone por LLM en sub-consultas por salto;
                 mult = max coseno sobre las sub-consultas (1 llamada/query).
  juez           juez LLM sobre las aserciones top de la frontera de las
                 semillas (Etapa II; ~1 llamada/query).

Uso programatico (con eval.wiki2.evaluar):
    r = Wiki2CatRAG(split="sondeo", variante="lite")
    agg = evaluar(r.buscar, preguntas, nombre="catrag_lite_sondeo")
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
from openai import OpenAI

from ..config import ARTIFACTS_DIR

WIKI2_DIR = ARTIFACTS_DIR / "wiki2"
MODELO_EMB = "text-embedding-3-small"
MODELO_LLM = "gpt-4o-mini"
_INVERTIR = ("HAS_INTERVENTION", "HAS_OUTCOME")


def _cargar_corpus(split: str) -> list[dict]:
    from ..eval.wiki2 import ruta_corpus
    return json.loads(ruta_corpus(split).read_text(encoding="utf-8"))


_ROL = (r"(director|screenwriter|performer|composer|producer|editor|star|"
        r"cast member|author|writer)")

_PAL_PREG = {"who", "what", "which", "when", "where", "why", "how", "are", "do",
             "does", "is", "was", "were", "did", "has", "have", "both", "film",
             "films", "movie", "movies", "song", "songs", "album", "albums",
             "book", "books", "novel", "novels", "the", "a", "an"}
_CONECTORES = {"of", "the", "de", "von", "van", "and", "&", "la", "le", "du",
               "der", "di", "y", "for", "in", "on", "to", "del", "da", "el"}


def _plegar_alias(s: str) -> str:
    import re
    s = re.sub(r"\(.*?\)", " ", s.lower())
    return " ".join(re.findall(r"[a-z0-9]+", s))


def _plegar_con_desamb(s: str) -> str:
    """Plegado que CONSERVA el contenido del parentesis ('The Girl of the Golden
    West (1922 film)' -> 'the girl of the golden west 1922 film'): distingue
    homonimos que _plegar_alias confunde."""
    import re
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def spans_entidad(pregunta: str) -> list[str]:
    """Menciones candidatas: secuencias de palabras capitalizadas (con
    conectores en minuscula entre ellas) fuera de las palabras de pregunta.
    'Who is the mother of the director of film Polish-Russian War (Film)?'
    -> ['Polish-Russian War (Film)']."""
    import re
    spans, actual = [], []
    for tok in pregunta.strip().rstrip("?").split():
        limpio = tok.strip(",;:.\"'")
        base = re.sub(r"['’][sS]$", "", limpio).strip("()")
        if base and (base[0].isupper() or base[0].isdigit()) and base.lower() not in _PAL_PREG:
            actual.append(limpio)
        elif actual and base.lower() in _CONECTORES:
            actual.append(limpio)
        else:
            if actual:
                spans.append(actual)
            actual = []
    if actual:
        spans.append(actual)
    out = []
    for s in spans:
        while s and s[-1].lower() in _CONECTORES:
            s.pop()
        if s:
            # El posesivo se quita solo al FINAL del tramo ("Lothair II's mother"
            # -> "Lothair II"); dentro del tramo forma parte del nombre ("God's
            # Gift to Women"), y el plegado lo trata como el alias del diccionario.
            s[-1] = re.sub(r"['’][sS]$", "", s[-1])
            out.append(" ".join(s))
    return out


def clasificar_tipo(pregunta: str) -> str:
    """Router de intencion por reglas lexicas (96,7% en las 11.376 de
    calibracion; validado 2026-08-21). Coste cero por consulta. La confusion
    residual inference<->compositional es benigna: ambos rutan a 'juez'."""
    import re
    q = pregunta.lower()
    dos = bool(re.search(r"\b(or|both)\b", q))
    if dos and re.search(_ROL, q) and \
            re.search(r"\b(film|movie|book|novel|song|album)s?\b", q):
        return "bridge_comparison"
    if re.search(r"\b(are both|do both|were both|which .{0,60}(first|earlier|"
                 r"later|more|older|younger|longer)|same (country|nationality|"
                 r"year|place))\b", q) or " or " in q:
        return "comparison"
    if re.search(r"\b(grand(father|mother|parent|son|daughter)|in-law|great-|"
                 r"uncle|aunt|sibling|maternal|paternal)\b", q):
        return "inference"
    return "compositional"


class Wiki2CatRAG:
    """Diales por defecto = punto 'balanceado' de produccion (peso_pasaje 0.015,
    damping 0.7, dense_top_n 100, eps 0.05); `modo='grafo'` replica el perfil
    de cobertura profunda (sin siembra densa, damping 0.85, anclas fuertes)."""

    def __init__(self, split: str = "sondeo", variante: str = "lite",
                 modo: str = "balanceado", plano: bool = False,
                 canonico: bool = False, peso_enlace: float = 0.5,
                 dicc: str = "", enlazador: str = "v3"):
        """canonico=True: grafo sobre el diccionario de entidades
        (grafo_<split>_canon<dicc>) + linker de menciones de la pregunta + masa
        directa en el pasaje propio de cada entidad sembrada (peso_enlace).
        dicc: sufijo de version del diccionario ("" = v3, "_v4" = curado).
        enlazador: 'v3' (alias exacto -> coseno tramo-ficha >= 0,80) o 'l2'
        (alias exacto -> de nombre a nombre: Dice de trigramas >= 0,80 o
        coseno de nombre >= 0,85, ficha como desempate; colisiones de alias
        resueltas por ficha en vez de por orden de carga)."""
        self.split, self.variante, self.canonico = split, variante, canonico
        self.peso_enlace, self.dicc, self.enlazador = peso_enlace, dicc, enlazador
        sufijo = ("_canon" + dicc if canonico else "") + ("_plano" if plano else "")
        self.g = nx.read_graphml(WIKI2_DIR / f"grafo_{split}{sufijo}.graphml")
        self.client = OpenAI()
        self.entradas: dict[str, dict] = {}
        self.alias_idx: dict[str, str] = {}          # alias plegado -> primera entrada (v3)
        self.alias_multi: dict[str, list[str]] = {}  # alias plegado -> todas las entradas
        self.alias_desamb: dict[str, list[str]] = {} # alias con parentesis conservado -> entradas
        self.alias_lista = None                      # indice L2 (perezoso)
        if canonico:
            for e in json.loads((WIKI2_DIR / f"entidades_{split}{dicc}.json").read_text(encoding="utf-8")):
                self.entradas[e["id"]] = e
                for al in e["alias"] + [e["nombre"]]:
                    pl = _plegar_alias(al)
                    if not pl:
                        continue
                    self.alias_idx.setdefault(pl, e["id"])
                    lst = self.alias_multi.setdefault(pl, [])
                    if e["id"] not in lst:
                        lst.append(e["id"])
                    if "(" in al:                        # forma con desambiguador
                        lst2 = self.alias_desamb.setdefault(_plegar_con_desamb(al), [])
                        if e["id"] not in lst2:
                            lst2.append(e["id"])
        if modo == "grafo":
            self.peso_pasaje, self.dense_top_n, self.damping, self.eps = 0.0, 0, 0.85, 1.0
        elif modo == "paridad":
            # Defaults publicados de HippoRAG 2: passage_node_weight 0.05,
            # damping 0.5, y TODOS los pasajes sembrados (dense_top_n=0 = todos).
            # Adopcion pre-registrada (no busqueda en benchmark).
            self.peso_pasaje, self.dense_top_n, self.damping, self.eps = 0.05, 0, 0.5, 0.05
        else:
            self.peso_pasaje, self.dense_top_n, self.damping, self.eps = 0.015, 100, 0.7, 0.05
        self._preparar(split, plano)

    # ------------------------------------------------------------ preparacion

    def _preparar(self, split: str, plano: bool) -> None:
        import scipy.sparse as sp
        # grafo de busqueda dirigido (reificacion invertida, sinonimia bidireccional)
        gb = nx.DiGraph()
        gb.add_nodes_from(self.g.nodes(data=True))
        for u, v, d in self.g.edges(data=True):
            t = d.get("tipo_relacion", "")
            if t == "SINONIMO_DE":
                gb.add_edge(u, v, weight=1.0)
                gb.add_edge(v, u, weight=1.0)
            elif t in _INVERTIR:
                gb.add_edge(v, u, weight=1.0)      # Entidad -> Asercion
            else:
                gb.add_edge(u, v, weight=1.0)      # Asercion/Entidad -> Chunk, rel planas
        self.nodos = list(gb.nodes())
        self.idx = {n: i for i, n in enumerate(self.nodos)}
        filas, cols, datos = [], [], []
        for u, v, d in gb.edges(data=True):
            filas.append(self.idx[u]); cols.append(self.idx[v]); datos.append(d["weight"])
        n = len(self.nodos)
        self.A = sp.csr_matrix((datos, (filas, cols)), shape=(n, n))

        d = np.load(WIKI2_DIR / f"emb_aserciones_{split}.npz", allow_pickle=False)
        fila_de = {a: i for i, a in enumerate(d["ids"])}
        pos, emb = [], []
        for nd in self.nodos:
            if self.g.nodes[nd].get("tipo") == "Asercion" and nd in fila_de:
                pos.append(self.idx[nd]); emb.append(fila_de[nd])
        self.aser_pos = np.array(pos)
        self.aser_emb = d["mat"][np.array(emb)] if pos else None
        self.aser_ids = [self.nodos[i] for i in pos]
        self.aser_ents = [[v for _, v, dd in self.g.out_edges(self.nodos[i], data=True)
                           if dd.get("tipo_relacion") in _INVERTIR] for i in pos]

        ruta_ent = (WIKI2_DIR / f"emb_fichas_{split}{self.dicc}.npz" if self.canonico
                    else WIKI2_DIR / f"emb_entidades_{split}.npz")
        de = np.load(ruta_ent, allow_pickle=False)
        self.ent_ids = list(de["ids"]); self.ent_emb = de["mat"]
        self.ent_pos = {e: i for i, e in enumerate(self.ent_ids)}
        self.aser_idx = {a: i for i, a in enumerate(self.aser_ids)}

        self.freq_ent: dict[str, int] = {}
        for u, _, dd in self.g.edges(data=True):
            if dd.get("tipo_relacion") == "MENCIONADO_EN":
                self.freq_ent[u] = self.freq_ent.get(u, 0) + 1

        # embeddings de pasajes (cache npz) para la siembra densa
        corpus = _cargar_corpus(split)
        self.titulos = [c["title"] for c in corpus]
        ruta = WIKI2_DIR / f"emb_chunks_{split}.npz"
        if ruta.exists():
            dc = np.load(ruta, allow_pickle=False)
            self.chunk_emb = dc["mat"]
        else:
            textos = [c["title"] + ". " + c["text"] for c in corpus]
            mat = np.zeros((len(textos), 1536), dtype=np.float32)
            for i in range(0, len(textos), 512):
                r = self.client.embeddings.create(input=textos[i:i + 512],
                                                  model=MODELO_EMB)
                for j, dat in enumerate(r.data):
                    mat[i + j] = dat.embedding
            mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
            np.savez_compressed(ruta, ids=np.array(self.titulos), mat=mat)
            self.chunk_emb = mat

    # ------------------------------------------------------------ semillas

    def _variante_efectiva(self, query: str) -> str:
        """'router': comparison predicho -> descomposicion (su especialista,
        FC@5 96% en sondeo); resto -> juez (mejor global y unico que arma
        cadenas de 4). Los demas valores de variante se usan tal cual."""
        if self.variante != "router":
            return self.variante
        return "descomposicion" if clasificar_tipo(query) == "comparison" else "juez"

    def _qvecs(self, query: str, variante: str) -> list[np.ndarray]:
        """Vector(es) de consulta segun la variante."""
        entradas = [query]
        if variante == "descomposicion":
            r = self.client.chat.completions.create(
                model=MODELO_LLM, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content":
                           'Decompose this multi-hop question into its single-hop '
                           'sub-queries (2 to 4), each a short standalone phrase; use '
                           '[X] for the unknown bridge entity. JSON: {"subqueries": [...]}. '
                           f"Question: {query}"}])
            try:
                subs = json.loads(r.choices[0].message.content).get("subqueries", [])
                entradas += [s for s in subs if isinstance(s, str) and s.strip()]
            except json.JSONDecodeError:
                pass
        r = self.client.embeddings.create(input=entradas, model=MODELO_EMB)
        out = []
        for dat in r.data:
            v = np.array(dat.embedding, dtype=np.float32)
            out.append(v / max(np.linalg.norm(v), 1e-9))
        return out

    def _sims_aserciones(self, query: str, qvecs: list[np.ndarray],
                         variante: str) -> tuple[np.ndarray, list[int] | None]:
        """Afinidad por asercion (max coseno sobre los vectores de consulta) y,
        en la variante 'juez', la lista de indices APROBADOS por el filtro
        (recognition memory). aprobados=None cuando el filtro no corre."""
        sims = np.max(np.stack([self.aser_emb @ v for v in qvecs]), axis=0)
        aprobados = None
        if variante == "juez":
            extra = self._frontera(self._enlaces_actuales, sims) if self.canonico else None
            aprobados = self._filtrar_hechos(query, sims, extra=extra)
            if aprobados:
                mult = sims.copy()
                mult[aprobados] = 1.0        # el paseo prefiere lo aprobado
                sims = mult
        return sims, aprobados

    # Demostraciones al estilo del filtro de HippoRAG 2 (su Figura 4): ensenan
    # a CONSERVAR los saltos intermedios y TODOS los candidatos a entidad
    # puente, sin resolver la pregunta con conocimiento propio del modelo.
    _DEMOS_FILTRO = """Example 1
Question: When is the director of film The Ancestor's birthday?
Facts:
0: Jean-Jacques Annaud was born on 1 October 1943.
1: Tsui Hark was born on 15 February 1950.
2: The Ancestor was directed by Guido Brignone.
3: Benh Zeitlin was born on 14 October 1982.
Selected: [2]
(The director facts about OTHER films are useless; keep the bridge fact even though the birthday itself is not listed.)

Example 2
Question: Who is the child of the performer of song Always On My Mind?
Facts:
0: Always On My Mind was recorded by Brenda Lee.
1: Always On My Mind became a hit for Willie Nelson.
2: Always On My Mind was written by Wayne Carson.
3: Elvis Presley recorded Always On My Mind in 1972.
Selected: [0, 1, 3]
(Several artists could be "the performer": keep ALL performer candidates; never use outside knowledge to pick one. The writer is not a performer.)

Example 3
Question: Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?
Facts:
0: Imperial River is located in Florida, United States.
1: Amaradia flows through Ro ia de Amaradia.
2: Imperial River may refer to South America.
Selected: [0, 1]
"""

    # ------------------------------------------------------------ linker

    _UMBRAL_FICHA = 0.80      # v3: coseno tramo-ficha
    _UMBRAL_DICE = 0.80       # L2: Dice de trigramas sobre el alias plegado
    _UMBRAL_NOMBRE = 0.85     # L2: coseno tramo-nombre del alias

    def _embeber_textos(self, textos: list[str]) -> list[np.ndarray]:
        r = self.client.embeddings.create(input=textos, model=MODELO_EMB)
        out = []
        for dat in r.data:
            v = np.array(dat.embedding, dtype=np.float32)
            out.append(v / max(np.linalg.norm(v), 1e-9))
        return out

    @staticmethod
    def _trigramas(s: str) -> set[str]:
        s = f" {s} "
        return {s[i:i + 3] for i in range(len(s) - 2)}

    def _preparar_alias(self) -> None:
        """Indice del escalon 2 de nombre a nombre (L2): alias plegados,
        trigramas para Dice e incrustacion del NOMBRE solo
        (emb_alias_<split><dicc>.npz, cache; ~31 800 alias en el banco)."""
        if self.alias_lista is not None:
            return
        self.alias_lista = sorted(self.alias_multi)
        self.alias_tri = [self._trigramas(a) for a in self.alias_lista]
        self.alias_len = np.array([len(t) for t in self.alias_tri], dtype=np.float32)
        self.tri_idx: dict[str, list[int]] = {}
        for i, tri in enumerate(self.alias_tri):
            for t in tri:
                self.tri_idx.setdefault(t, []).append(i)
        ruta = WIKI2_DIR / f"emb_alias_{self.split}{self.dicc}.npz"
        if ruta.exists():
            d = np.load(ruta, allow_pickle=False)
            if len(d["ids"]) == len(self.alias_lista) and all(
                    a == b for a, b in zip(d["ids"], self.alias_lista)):
                self.alias_emb = d["mat"]
                return
        mat = np.zeros((len(self.alias_lista), 1536), dtype=np.float32)
        for i in range(0, len(self.alias_lista), 512):
            for j, v in enumerate(self._embeber_textos(self.alias_lista[i:i + 512])):
                mat[i + j] = v
        np.savez_compressed(ruta, ids=np.array(self.alias_lista), mat=mat)
        self.alias_emb = mat

    def _dice_candidatos(self, texto_pleg: str, n: int = 5) -> list[tuple[int, float]]:
        """Alias con mayor coeficiente de Dice de trigramas con el texto."""
        tri = self._trigramas(texto_pleg)
        cuenta: dict[int, int] = {}
        for t in tri:
            for i in self.tri_idx.get(t, ()):
                cuenta[i] = cuenta.get(i, 0) + 1
        if not cuenta:
            return []
        idx = np.fromiter(cuenta.keys(), dtype=np.int64)
        inter = np.fromiter(cuenta.values(), dtype=np.float32)
        dice = 2 * inter / (len(tri) + self.alias_len[idx])
        orden = np.argsort(-dice)[:n]
        return [(int(idx[o]), float(dice[o])) for o in orden]

    def _exactos(self, texto: str) -> list[str]:
        """Escalon 1: alias exacto tras plegado. Si el texto trae parentesis,
        primero con el desambiguador conservado (L2/E7: 'X (1922 film)' no es
        'X (1923 film)'); despues sin el (con y sin parentesis)."""
        if "(" in texto and self.enlazador != "v3":
            ids = [e for e in self.alias_desamb.get(_plegar_con_desamb(texto), ()) if e in self.idx]
            if ids:
                return ids
        formas = [texto] + ([texto.split("(")[0]] if "(" in texto else [])
        for forma in formas:
            pl = _plegar_alias(forma)
            if not pl:
                continue
            if self.enlazador == "v3":
                eid = self.alias_idx.get(pl)
                if eid and eid in self.idx:
                    return [eid]
            else:
                ids = [e for e in self.alias_multi.get(pl, ()) if e in self.idx]
                if ids:
                    return ids
        return []

    def _desempatar(self, cands: list[str], v: np.ndarray) -> str:
        """Entre entradas distintas con el mismo alias (o puntuacion), la de
        ficha (nombre, alias, descripcion) mas afin al texto."""
        con_pos = [(e, self.ent_pos[e]) for e in cands if e in self.ent_pos]
        if not con_pos:
            return cands[0]
        s = self.ent_emb[[p for _, p in con_pos]] @ v
        return con_pos[int(np.argmax(s))][0]

    def _escalon2_nombre(self, texto: str, v: np.ndarray) -> str | None:
        """L2: de nombre a nombre. Canal lexico: Dice de trigramas >= 0,80
        sobre el alias plegado (captura truncamientos y erratas y penaliza la
        palabra que falta: 'bad education movie' no es 'bad education'). Canal
        vectorial: coseno de nombre >= 0,85, solo si el lexico no admite a
        nadie (variantes sin superficie comun). Entre entradas a menos de 0,02
        de la mejor del canal que decide, desempate por ficha."""
        self._preparar_alias()
        pl = _plegar_alias(texto)
        if not pl:
            return None
        puntos: dict[str, float] = {}
        for i, dice in self._dice_candidatos(pl):
            if dice >= self._UMBRAL_DICE:
                for e in self.alias_multi[self.alias_lista[i]]:
                    puntos[e] = max(puntos.get(e, 0.0), dice)
        puntos = {e: p for e, p in puntos.items() if e in self.idx}
        if not puntos:
            sc = self.alias_emb @ v
            for i in np.argsort(-sc)[:5]:
                if sc[i] >= self._UMBRAL_NOMBRE:
                    for e in self.alias_multi[self.alias_lista[int(i)]]:
                        puntos[e] = max(puntos.get(e, 0.0), float(sc[i]))
            puntos = {e: p for e, p in puntos.items() if e in self.idx}
        if not puntos:
            return None
        mejor = max(puntos.values())
        empate = [e for e, p in puntos.items() if p >= mejor - 0.02]
        return empate[0] if len(empate) == 1 else self._desempatar(empate, v)

    def _enlazar_textos(self, textos: list[str]) -> list[str | None]:
        """Enlaza cada texto (tramo de la pregunta o forma canonica devuelta
        por el analista) a una entrada del diccionario: escalon 1 exacto y,
        si falla o hay colision, escalon 2 segun `enlazador`. Una sola llamada
        de incrustacion para todos los pendientes."""
        out: list[str | None] = [None] * len(textos)
        pendientes = []
        for k, texto in enumerate(textos):
            cands = self._exactos(texto)
            if len(cands) == 1:
                out[k] = cands[0]
            elif texto.strip():
                pendientes.append((k, cands))
        if not pendientes:
            return out
        # v3 incrusta el tramo tal cual (contra fichas); L2, plegado (contra nombres)
        entradas = [textos[k] if self.enlazador == "v3" else (_plegar_alias(textos[k]) or textos[k])
                    for k, _ in pendientes]
        for (k, cands), v in zip(pendientes, self._embeber_textos(entradas)):
            if len(cands) > 1:
                out[k] = self._desempatar(cands, v)
            elif self.enlazador == "l2":
                out[k] = self._escalon2_nombre(textos[k], v)
            else:
                sc = self.ent_emb @ v
                j = int(np.argmax(sc))
                if sc[j] >= self._UMBRAL_FICHA and self.ent_ids[j] in self.idx:
                    out[k] = self.ent_ids[j]
        return out

    def _enlazar_menciones(self, query: str) -> list[str]:
        """Primera etapa (solo canonico): menciones de la pregunta -> ids del
        diccionario (ver _enlazar_textos)."""
        if not self.canonico:
            return []
        import re
        spans = []
        for span in spans_entidad(query):
            # 'A and B' / 'A or B': si el conjunto no es alias, probar las partes
            if not self._exactos(span) and re.search(r"\s(and|or|&)\s", span):
                spans.extend(p.strip() for p in re.split(r"\s(?:and|or|&)\s", span) if p.strip())
            else:
                spans.append(span)
        return list(dict.fromkeys(e for e in self._enlazar_textos(spans) if e))

    def _frontera(self, enlazadas: list[str], sims: np.ndarray, por_ent: int = 15) -> list[int]:
        """Aserciones colgadas de las entidades enlazadas (frontera de las
        semillas, CatRAG), las mas afines a la consulta por coseno."""
        out = []
        for eid in enlazadas:
            asers = [self.aser_idx[u] for u, _, d in self.g.in_edges(eid, data=True)
                     if d.get("tipo_relacion") in _INVERTIR and u in self.aser_idx]
            asers.sort(key=lambda j: -sims[j])
            out.extend(asers[:por_ent])
        return out

    def _filtrar_hechos(self, query: str, sims: np.ndarray,
                        top_k: int = 40, max_sel: int = 4,
                        extra: list[int] | None = None) -> list[int]:
        """Filtro tipo recognition memory (HippoRAG 2): el LLM SELECCIONA hasta
        max_sel hechos del top-K por coseno (+ frontera `extra`) que sirvan
        para responder, incluyendo saltos intermedios y candidatos puente.
        Devuelve indices en el espacio de aserciones (vacio -> fallback)."""
        orden = list(np.argsort(-sims)[:top_k])
        if extra:
            vistos = set(orden)
            orden += [j for j in extra if j not in vistos][:20]
        orden = np.array(orden)
        lineas = "\n".join(f"{i}: {self.g.nodes[self.aser_ids[j]].get('descripcion', '')[:160]}"
                           for i, j in enumerate(orden))
        r = self.client.chat.completions.create(
            model=MODELO_LLM, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content":
                       "You filter facts for a question-answering system over a knowledge "
                       f"graph. Select up to {max_sel} facts from the numbered list that help "
                       "answer the question, INCLUDING intermediate-hop facts (e.g. the film's "
                       "director when the question asks about that director's family). If a "
                       "role in the question (performer, director, founder...) has several "
                       "candidate entities in the facts, keep them all — do NOT use your own "
                       "knowledge to choose. If no fact is relevant, return an empty list.\n\n"
                       f"{self._DEMOS_FILTRO}\n"
                       'Reply with JSON: {"selected": [<indices>]}\n\n'
                       f"Question: {query}\nFacts:\n{lineas}"}])
        try:
            sel = json.loads(r.choices[0].message.content).get("selected", [])
            return [int(orden[int(i)]) for i in sel
                    if str(i).isdigit() and int(i) < len(orden)][:max_sel]
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def _semillas(self, query: str, qvecs, sims,
                  aprobados: list[int] | None = None) -> dict[str, float]:
        pers: dict[str, float] = {}
        # a) semillas por hechos.
        if aprobados is not None:
            # COMPUERTA DURA (recognition memory, HippoRAG 2): solo siembran
            # las entidades de los hechos aprobados por el filtro; peso = media
            # del coseno de sus hechos aprobados. Lista vacia -> fallback: sin
            # semillas de frase, decide la siembra densa (ranking ~denso puro).
            pesos, cuenta = {}, {}
            for j in aprobados:
                for ent in self.aser_ents[j]:
                    pesos[ent] = pesos.get(ent, 0.0) + float(np.clip(sims[j], 0, 1))
                    cuenta[ent] = cuenta.get(ent, 0) + 1
            pers.update({e: pesos[e] / cuenta[e] for e in pesos})
        else:
            # compuerta blanda original (lite/descomposicion): top-15 por
            # coseno con anti-hub.
            norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
            pesos, cuenta = {}, {}
            for j in np.argsort(-sims)[:15]:
                for ent in self.aser_ents[j]:
                    f = self.freq_ent.get(ent, 1) or 1
                    pesos[ent] = pesos.get(ent, 0.0) + float(norm[j]) / f
                    cuenta[ent] = cuenta.get(ent, 0) + 1
            for ent in pesos:
                pesos[ent] /= cuenta[ent]
            pers.update(dict(sorted(pesos.items(), key=lambda x: -x[1])[:5]))
        # b) siembra densa de pasajes: TODOS los nodos-pasaje con su score
        # denso normalizado x peso (HippoRAG 2, apendice G.1: "all passage
        # nodes are taken as seed nodes"). dense_top_n > 0 restringe (modos
        # antiguos); paridad usa todos.
        if self.peso_pasaje > 0:
            s = self.chunk_emb @ qvecs[0]
            if 0 < self.dense_top_n < len(s):
                top = np.argsort(-s)[:self.dense_top_n]
            else:
                top = np.arange(len(s))
            mm = (s[top] - s[top].min()) / (s[top].max() - s[top].min() + 1e-9)
            for t, m in zip(top, mm):
                nodo = f"chunk:{self.titulos[int(t)]}"
                if nodo in self.idx:
                    pers[nodo] = pers.get(nodo, 0.0) + self.peso_pasaje * float(m)
        # c) anclas simbolicas debiles: top-2 entidades por coseno de cada qvec
        for v in qvecs:
            se = self.ent_emb @ v
            for j in np.argsort(-se)[:2]:
                nodo = self.ent_ids[int(j)]
                if nodo in self.idx:
                    pers[nodo] = pers.get(nodo, 0.0) + self.eps
        # d) diccionario: entidades enlazadas desde la pregunta (anclas fuertes)
        #    y masa DIRECTA en el pasaje propio de toda entidad sembrada
        #    (enlazada o aprobada por el filtro): el pasaje ES la entidad.
        if self.canonico:
            for eid in self._enlaces_actuales:
                pers[eid] = max(pers.get(eid, 0.0), self.peso_enlace)
            for eid, w in list(pers.items()):
                ent = self.entradas.get(eid)
                if ent and ent.get("pasaje_propio"):
                    nodo = f"chunk:{ent['pasaje_propio']}"
                    if nodo in self.idx:
                        pers[nodo] = pers.get(nodo, 0.0) + self.peso_enlace * w
        return pers

    # ------------------------------------------------------------ PPR

    def _ppr(self, pers: dict[str, float], sims: np.ndarray) -> np.ndarray:
        import scipy.sparse as sp
        n = len(self.nodos)
        v = np.zeros(n)
        for nodo, peso in pers.items():
            if nodo in self.idx:
                v[self.idx[nodo]] = peso
        if v.sum() == 0:
            return v
        v /= v.sum()
        mult = np.ones(n)
        if len(self.aser_pos):
            mult[self.aser_pos] = 0.25 + 0.75 * np.clip(sims, 0.0, 1.0)
        A = self.A.multiply(mult[None, :]).tocsr()
        salidas = np.asarray(A.sum(axis=1)).ravel()
        P = sp.diags(np.divide(1.0, salidas, out=np.zeros_like(salidas),
                               where=salidas > 0)) @ A
        colgantes = salidas == 0
        p = v.copy()
        for _ in range(80):
            p_nuevo = self.damping * (P.T @ p + p[colgantes].sum() * v) \
                + (1 - self.damping) * v
            if np.abs(p_nuevo - p).sum() < 1e-9:
                return p_nuevo
            p = p_nuevo
        return p

    def buscar(self, query: str, k: int = 20) -> list[str]:
        variante = self._variante_efectiva(query)
        self._enlaces_actuales = self._enlazar_menciones(query)
        qvecs = self._qvecs(query, variante)
        sims, aprobados = self._sims_aserciones(query, qvecs, variante)
        pers = self._semillas(query, qvecs, sims, aprobados)
        p = self._ppr(pers, sims)
        orden = np.argsort(-p)
        titulos = []
        for i in orden:
            if p[i] <= 0:
                break
            nodo = self.nodos[i]
            if nodo.startswith("chunk:"):
                titulos.append(nodo[len("chunk:"):])
                if len(titulos) >= k:
                    break
        return titulos
