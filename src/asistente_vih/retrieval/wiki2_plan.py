"""Plan A: analista de la pregunta (NER + plan + seleccion), conjunto final y
salto dirigido. Extiende la v3 (Wiki2CatRAG canonico) sin modificarla.

Tres llamadas LLM:
  1. ANALISTA (siempre): recibe pregunta + hechos candidatos (top-40 coseno
     union frontera) y devuelve menciones canonicas, el PLAN de la cadena
     (huecos), la seleccion de hechos que los rellenan (tope dinamico = n.o de
     huecos, max 8) y los huecos sin cubrir.
  2. CONJUNTO (salvo comparison): elige, del top-12 del PPR, los 5 pasajes que
     JUNTOS cubren el plan. Guarda conservadora: el top-2 del PPR se conserva.
  3. SALTO (solo si hay huecos sin cubrir, 1 ronda): cada hueco se embebe, se
     buscan sus hechos mas afines, un mini-filtro los adjudica y se re-siembra.

Proveedores por nombre de modelo: "gpt-*" -> OpenAI; "deepseek-*" -> DeepSeek;
"groq/<modelo>" -> Groq (API compatible OpenAI). Toda llamada se cachea en
SQLite por (modelo, hash del prompt): re-mediciones a coste cero.

    r = Wiki2PlanRAG(split="sondeo", modelo="gpt-4o-mini")
    agg = evaluar(r.buscar, preguntas, nombre="plan_gpt4omini_sondeo")
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import numpy as np
from openai import OpenAI

from .wiki2_catrag import (Wiki2CatRAG, WIKI2_DIR, MODELO_EMB, _plegar_alias,
                           clasificar_tipo, _cargar_corpus)

_PROMPT_ANALISTA = """You are the retrieval analyst of a multi-hop question answering system.

Given a question and a numbered list of candidate facts extracted from the corpus, reply with STRICT JSON:

{"mentions":  [{"text": "...", "canonical": "<the exact Wikipedia article title form>"}],
 "plan":      ["<slot 1, e.g. 'director of film A -> X'>", "<slot 2, e.g. 'date of death of X'>", ...],
 "selected":  [<indices of facts that fill a slot>],
 "unfilled":  ["<slot with NO selected fact, phrased with the entity name if known>"]}

Rules:
- The PLAN lists every fact needed to answer, as slots, INCLUDING intermediate hops.
- SELECT every candidate fact that fills a slot (up to 8). A fact naming the director/performer/
  father is essential even if it carries no date.
- If a role in the question (performer, director, founder...) has SEVERAL candidate entities in
  the facts, select them ALL. Never use your own knowledge to choose among them or to skip a hop.
- In the plan, write an entity name ONLY if a candidate fact states it; otherwise use [X].
  NEVER fill a slot from your own world knowledge.
- "unfilled" lists slots no candidate fact covers, phrased as a retrieval query using what IS
  known (e.g. "director of The Magician (1958 film)" or "date of death of Kurt Neumann").

Example
Question: Which film has the director died first, Film A or Film B?
Facts: 0: Film A is a 1960 crime film.  1: Film A was directed by M. Fric.  2: E. Carroll stars in Film B.  3: Film B was directed by K. Neumann.  4: M. Fric died in 1968.
{"mentions": [{"text": "Film A", "canonical": "Film A"}, {"text": "Film B", "canonical": "Film B"}],
 "plan": ["director of Film A -> M. Fric", "date of death of M. Fric", "director of Film B -> K. Neumann", "date of death of K. Neumann"],
 "selected": [1, 3, 4],
 "unfilled": ["date of death of K. Neumann"]}
"""

_PROMPT_CONJUNTO = """Pick the SET of 5 passages (by index) that TOGETHER cover every slot of the plan
(all hops of the chain). Prefer chain completeness over individual similarity.
Reply STRICT JSON: {"selected": [<5 indices>]}
"""

_PROMPT_SALTO = """Select the facts (by index) that fill this missing slot of the plan. Each fact shows
[the article it comes from]: a fact from the article about the slot's entity DOES refer to it even
if the fact uses a different or original-language title for the same work. If none fills the slot,
reply with an empty list. STRICT JSON: {"selected": [<indices>]}
"""


class Wiki2PlanRAG(Wiki2CatRAG):
    """Plan A sobre la v3. `modelo` gobierna las tres llamadas; los
    conmutadores permiten medir cada pieza por separado (ablacion)."""

    def __init__(self, split: str = "sondeo", modelo: str = "gpt-4o-mini",
                 usar_conjunto: bool = True, usar_salto: bool = True):
        super().__init__(split=split, variante="juez", modo="paridad", canonico=True)
        self.modelo = modelo
        self.usar_conjunto, self.usar_salto = usar_conjunto, usar_salto
        self.textos = {c["title"]: c["text"] for c in _cargar_corpus(split)}
        self._cache = sqlite3.connect(WIKI2_DIR / "plan_cache.sqlite")
        self._cache.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, r TEXT)")
        self._clientes: dict[str, tuple[OpenAI, str]] = {}

    # ------------------------------------------------------------ LLM

    def _cliente_para(self, modelo: str) -> tuple[OpenAI, str]:
        if modelo not in self._clientes:
            if modelo.startswith("deepseek"):
                cli = OpenAI(base_url="https://api.deepseek.com",
                             api_key=os.environ["DEEPSEEK_API_KEY"])
                self._clientes[modelo] = (cli, modelo)
            elif modelo.startswith("groq/"):
                cli = OpenAI(base_url="https://api.groq.com/openai/v1",
                             api_key=os.environ["GROQ_API_KEY"])
                self._clientes[modelo] = (cli, modelo.split("/", 1)[1])
            else:
                self._clientes[modelo] = (self.client, modelo)
        return self._clientes[modelo]

    def _llm_json(self, prompt: str) -> dict:
        k = hashlib.sha256((self.modelo + "\x00" + prompt).encode()).hexdigest()
        fila = self._cache.execute("SELECT r FROM llm WHERE k=?", (k,)).fetchone()
        if fila:
            crudo = fila[0]
        else:
            cli, nombre = self._cliente_para(self.modelo)
            try:
                r = cli.chat.completions.create(
                    model=nombre, temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}])
                crudo = r.choices[0].message.content
            except Exception:
                # algunos proveedores (Groq + modelos razonadores) fallan la
                # validacion json_object: reintento libre y extraccion de {...}
                r = cli.chat.completions.create(
                    model=nombre, temperature=0.0,
                    messages=[{"role": "user", "content":
                               prompt + "\nRespond ONLY with the JSON object."}])
                texto = r.choices[0].message.content or ""
                i, f = texto.find("{"), texto.rfind("}")
                crudo = texto[i:f + 1] if 0 <= i < f else "{}"
            self._cache.execute("INSERT OR REPLACE INTO llm VALUES (?,?)", (k, crudo))
            self._cache.commit()
        try:
            return json.loads(crudo)
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------ llamada 1

    def _analista(self, query: str, sims: np.ndarray,
                  enlaces: list[str]) -> tuple[list[int], list[str], list[str]]:
        """-> (aprobados, huecos, enlaces ampliados con las menciones)."""
        top40 = list(np.argsort(-sims)[:40])
        extra = self._frontera(enlaces, sims)
        orden = top40 + [j for j in extra if j not in set(top40)][:20]
        lineas = "\n".join(f"{i}: {self.g.nodes[self.aser_ids[j]].get('descripcion','')[:130]}"
                           for i, j in enumerate(orden))
        d = self._llm_json(f"{_PROMPT_ANALISTA}\nQuestion: {query}\nFacts:\n{lineas}")
        n_plan = len(d.get("plan", [])) or 4
        sel = [int(orden[int(i)]) for i in d.get("selected", [])
               if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(orden)]
        sel = sel[:min(8, max(2, n_plan))]
        nuevos = list(enlaces)
        for m in d.get("mentions", []):
            can = m.get("canonical") or m.get("text") or ""
            eid = self.alias_idx.get(_plegar_alias(can))
            if eid and eid in self.idx and eid not in nuevos:
                nuevos.append(eid)
        huecos = [h for h in d.get("unfilled", []) if isinstance(h, str) and h.strip()][:2]
        plan = [p for p in d.get("plan", []) if isinstance(p, str)]
        return sel, huecos, nuevos, plan

    # ------------------------------------------------------------ llamada 3

    def _salto(self, huecos: list[str], aprobados: list[int],
               enlaces: list[str]) -> list[int]:
        """Por cada hueco, candidatos por DOS vias: (1) estructural — las
        aserciones del pasaje propio de la entidad enlazada a la que se
        refiere el hueco (inmune a alias en otro idioma: 'The Magician' cuyo
        articulo habla de 'Ansiktet'); (2) coseno del texto del hueco."""
        r = self.client.embeddings.create(input=huecos, model=MODELO_EMB)
        nuevos: list[int] = []
        for hueco, dat in zip(huecos, r.data):
            v = np.array(dat.embedding, dtype=np.float32)
            v /= max(np.linalg.norm(v), 1e-9)
            hueco_pleg = _plegar_alias(hueco)
            estructural: list[int] = []
            for eid in enlaces:
                ent = self.entradas.get(eid, {})
                nombres = [ent.get("nombre", "")] + list(ent.get("alias", []))
                if ent.get("pasaje_propio") and any(
                        _plegar_alias(n) and _plegar_alias(n) in hueco_pleg for n in nombres):
                    pref = f"{ent['pasaje_propio']}::"
                    estructural += [self.aser_idx[a] for a in self.aser_ids
                                    if a.startswith(pref) and a in self.aser_idx][:12]
            cos_top = [int(j) for j in np.argsort(-(self.aser_emb @ v))[:8]]
            cand = [j for j in dict.fromkeys(estructural + cos_top)
                    if j not in set(aprobados)][:16]
            lineas = "\n".join(
                f"{i}: [{self.aser_ids[j].rsplit('::', 1)[0]}] "
                f"{self.g.nodes[self.aser_ids[j]].get('descripcion','')[:120]}"
                for i, j in enumerate(cand))
            d = self._llm_json(f"{_PROMPT_SALTO}\nMissing slot: {hueco}\nFacts:\n{lineas}")
            nuevos += [cand[int(i)] for i in d.get("selected", [])
                       if str(i).isdigit() and int(i) < len(cand)][:2]
        return nuevos

    # ------------------------------------------------------------ llamada 2

    def _conjunto(self, query: str, plan_txt: str, titulos: list[str]) -> list[str]:
        lineas = "\n".join(f"{i}: [{t}] {self.textos.get(t,'')[:140]}"
                           for i, t in enumerate(titulos[:12]))
        d = self._llm_json(f"{_PROMPT_CONJUNTO}\nQuestion: {query}\nPlan: {plan_txt}\n"
                           f"Passages:\n{lineas}")
        idx = [int(i) for i in d.get("selected", [])
               if str(i).isdigit() and int(i) < min(12, len(titulos))]
        elegidos = [titulos[i] for i in idx]
        # guarda conservadora: el top-2 del PPR nunca se pierde
        for t in titulos[:2]:
            if t not in elegidos:
                elegidos.insert(0, t)
        # completar a 5 en orden PPR y anexar el resto para R@10/R@20
        for t in titulos:
            if len(elegidos) >= 5:
                break
            if t not in elegidos:
                elegidos.append(t)
        return elegidos[:5] + [t for t in titulos if t not in elegidos[:5]]

    # ------------------------------------------------------------ flujo

    def buscar(self, query: str, k: int = 20) -> list[str]:
        enlaces0 = self._enlazar_menciones(query)
        qv = self._qvecs(query, "lite")[0]
        sims = self.aser_emb @ qv
        aprobados, huecos, enlaces, plan = self._analista(query, sims, enlaces0)
        self._ultimo_plan = plan
        self._enlaces_actuales = enlaces
        stilde = sims.copy()
        if aprobados:
            stilde[aprobados] = 1.0
        pers = self._semillas(query, [qv], stilde, aprobados)
        p = self._ppr(pers, stilde)
        titulos = [self.nodos[i][6:] for i in np.argsort(-p)
                   if self.nodos[i].startswith("chunk:")][:60]
        # llamada 3: salto dirigido si quedaron huecos
        if self.usar_salto and huecos:
            extra = self._salto(huecos, aprobados, enlaces)
            if extra:
                aprobados = aprobados + extra
                stilde[extra] = 1.0
                pers = self._semillas(query, [qv], stilde, aprobados)
                p = self._ppr(pers, stilde)
                titulos = [self.nodos[i][6:] for i in np.argsort(-p)
                           if self.nodos[i].startswith("chunk:")][:60]
        # llamada 2: conjunto final (no en comparison)
        if self.usar_conjunto and clasificar_tipo(query) != "comparison":
            plan_txt = "; ".join(plan) if plan else query
            titulos = self._conjunto(query, plan_txt, titulos)
        return titulos[:k]
