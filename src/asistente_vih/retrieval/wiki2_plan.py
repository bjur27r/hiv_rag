"""Plan A: analista de la pregunta (NER + plan + seleccion), conjunto final y
salto dirigido. Extiende la v3 (Wiki2CatRAG canonico) sin modificarla.

Version generalizada (2026-08-30; la del 25-08 esta en git, commit a72bde5):
  - una sola estrategia para todo tipo de pregunta (el conjunto final tambien
    en comparacion; el enrutador lexico de 2Wiki queda solo como etiqueta);
  - el analista devuelve, por hueco sin cubrir, la entidad ANCLA a la que se
    refiere: los candidatos estructurales del salto salen de su pasaje propio
    (antes: subcadena del nombre en el texto del hueco);
  - las formas canonicas del analista se enlazan con el escalon 2 de nombre a
    nombre (L2) cuando el recuperador base lo tiene activado;
  - rondas de salto: si tras un salto quedan huecos con comodin [X], el
    analista re-planifica con los hechos descubiertos a la vista y se salta de
    nuevo (a lo sumo `max_rondas`; en cadenas de dos saltos degenera en una);
  - demostraciones del analista derivadas del conjunto de CALIBRACION
    (Convoy Buddies / Tip Toes; Zoolander 2), no del banco.

Tres llamadas LLM:
  1. ANALISTA (siempre): recibe pregunta + hechos candidatos (top-40 coseno
     union frontera, mas los aprobados en rondas previas) y devuelve menciones
     canonicas, el PLAN de la cadena (huecos), la seleccion de hechos que los
     rellenan (tope dinamico = n.o de huecos, max 8) y los huecos sin cubrir
     con su ancla.
  2. SALTO (solo si hay huecos con ancla conocida): cada hueco se embebe, se
     buscan sus hechos mas afines entre los del pasaje propio del ancla y por
     coseno, un mini-filtro los adjudica y se re-siembra.
  3. CONJUNTO: elige, del top-12 del PPR, los 5 pasajes que JUNTOS cubren el
     plan. Guarda conservadora: el top-2 del PPR se conserva.

Proveedores por nombre de modelo: "gpt-*" -> OpenAI; "deepseek-*" -> DeepSeek;
"groq/<modelo>" -> Groq (API compatible OpenAI). Toda llamada se cachea en
SQLite por (modelo, hash del prompt): re-mediciones a coste cero.

    r = Wiki2PlanRAG.v4(split="sondeo", modelo="deepseek-chat")
    agg = evaluar(r.buscar, preguntas, nombre="plan_v4_deepseek_sondeo")
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import numpy as np
from openai import OpenAI

from .wiki2_catrag import (Wiki2CatRAG, WIKI2_DIR, _plegar_alias,
                           clasificar_tipo, _cargar_corpus)

_PROMPT_ANALISTA = """You are the retrieval analyst of a multi-hop question answering system.

Given a question and a numbered list of candidate facts extracted from the corpus, reply with STRICT JSON:

{"mentions": [{"text": "...", "canonical": "<the exact Wikipedia article title form>"}],
 "plan":     ["<slot 1, e.g. 'director of film A -> X'>", "<slot 2, e.g. 'date of death of X'>", ...],
 "selected": [<indices of facts that fill a slot>],
 "unfilled": [{"query": "<slot with NO selected fact, phrased as a retrieval query>",
               "anchor": "<the known entity the slot is about, exactly as named in a fact or in the question; null if the slot is about a still-unknown [X]>"}]}

Rules:
- The PLAN lists every fact needed to answer, as slots, INCLUDING intermediate hops.
- SELECT every candidate fact that fills a slot (up to 8). A fact naming the director/performer/
  father is essential even if it carries no date.
- If a role in the question (performer, director, founder...) has SEVERAL candidate entities in
  the facts, select them ALL. Never use your own knowledge to choose among them or to skip a hop.
- In the plan, write an entity name ONLY if a candidate fact states it; otherwise use [X].
  NEVER fill a slot from your own world knowledge, and never select a fact about an entity that
  no candidate fact connects to the question.
- "unfilled" lists slots no candidate fact covers. "anchor" is the known entity the slot asks
  about (the film, the person...); a slot about a still-unknown [X] has anchor null.

Example 1
Question: Which film has the director who died first, Convoy Buddies or Tip Toes?
Facts: 0: Convoy Buddies is a 1975 Italian comedy film.  1: Convoy Buddies was directed by Giuliano Carnimeo.  2: Tip Toes is a 1927 British silent comedy film.  3: Tip Toes was directed by Herbert Wilcox.  4: Herbert Wilcox died on 15 May 1977.  5: Dorothy Gish starred in Tip Toes.
{"mentions": [{"text": "Convoy Buddies", "canonical": "Convoy Buddies"}, {"text": "Tip Toes", "canonical": "Tip Toes"}],
 "plan": ["director of Convoy Buddies -> Giuliano Carnimeo", "date of death of Giuliano Carnimeo", "director of Tip Toes -> Herbert Wilcox", "date of death of Herbert Wilcox"],
 "selected": [1, 3, 4],
 "unfilled": [{"query": "date of death of Giuliano Carnimeo", "anchor": "Giuliano Carnimeo"}]}

Example 2
Question: Who is the mother of the director of film Zoolander 2?
Facts: 0: Zoolander 2 is a 2016 American comedy film.  1: Zoolander 2 stars Owen Wilson.  2: Ben Stiller is the son of Jerry Stiller and Anne Meara.  3: Zoolander is a 2001 film directed by Ben Stiller.
{"mentions": [{"text": "Zoolander 2", "canonical": "Zoolander 2"}],
 "plan": ["director of Zoolander 2 -> [X]", "mother of [X]"],
 "selected": [],
 "unfilled": [{"query": "director of Zoolander 2", "anchor": "Zoolander 2"}, {"query": "mother of [X]", "anchor": null}]}
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

MAX_HUECOS = 4          # huecos sin cubrir considerados por ronda
MAX_CAND_SALTO = 16     # candidatos por hueco (estructurales + coseno)


class Wiki2PlanRAG(Wiki2CatRAG):
    """Plan A sobre la v3/v4. `modelo` gobierna las tres llamadas; los
    conmutadores permiten medir cada pieza por separado (ablacion)."""

    def __init__(self, split: str = "sondeo", modelo: str = "gpt-4o-mini",
                 usar_conjunto: bool = True, usar_salto: bool = True,
                 dicc: str = "", enlazador: str = "v3",
                 conjunto_en_comparacion: bool = True, max_rondas: int = 3):
        super().__init__(split=split, variante="juez", modo="paridad", canonico=True,
                         dicc=dicc, enlazador=enlazador)
        self.modelo = modelo
        self.usar_conjunto, self.usar_salto = usar_conjunto, usar_salto
        self.conjunto_en_comparacion, self.max_rondas = conjunto_en_comparacion, max_rondas
        self.textos = {c["title"]: c["text"] for c in _cargar_corpus(split)}
        self._aser_por_pasaje: dict[str, list[int]] = {}
        for j, a in enumerate(self.aser_ids):
            self._aser_por_pasaje.setdefault(a.rsplit("::", 1)[0], []).append(j)
        self._cache = sqlite3.connect(WIKI2_DIR / "plan_cache.sqlite", timeout=120)
        self._cache.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, r TEXT)")
        self._clientes: dict[str, tuple[OpenAI, str]] = {}
        self.n_llamadas = 0

    @classmethod
    def v4(cls, split: str = "sondeo", modelo: str = "deepseek-chat", **kw) -> "Wiki2PlanRAG":
        """Configuracion generalizada: diccionario curado (_v4) + enlazador L2
        + conjunto en todos los tipos + rondas de salto."""
        kw.setdefault("dicc", "_v4"); kw.setdefault("enlazador", "l2")
        return cls(split=split, modelo=modelo, **kw)

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
            self.n_llamadas += 1
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

    def _analista(self, query: str, sims: np.ndarray, enlaces: list[str],
                  extra: list[int] = ()) -> tuple[list[int], list[dict], list[str], list[str]]:
        """-> (aprobados, huecos [{query, anchor}], enlaces ampliados con las
        menciones, plan). `extra`: hechos aprobados en rondas previas, que el
        analista debe ver para re-planificar con los puentes descubiertos."""
        top40 = list(np.argsort(-sims)[:40])
        vistos = set(top40)
        orden = top40 + [j for j in self._frontera(enlaces, sims) if j not in vistos][:20]
        vistos = set(orden)
        orden += [j for j in extra if j not in vistos]
        lineas = "\n".join(f"{i}: {self.g.nodes[self.aser_ids[j]].get('descripcion','')[:130]}"
                           for i, j in enumerate(orden))
        d = self._llm_json(f"{_PROMPT_ANALISTA}\nQuestion: {query}\nFacts:\n{lineas}")
        plan = [p for p in d.get("plan", []) if isinstance(p, str)]
        n_plan = len(plan) or 4
        sel = [int(orden[int(i)]) for i in d.get("selected", [])
               if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(orden)]
        sel = list(dict.fromkeys(sel))[:min(8, max(2, n_plan))]
        canon = [(m.get("canonical") or m.get("text") or "").strip()
                 for m in d.get("mentions", []) if isinstance(m, dict)]
        canon = [c for c in canon if c]
        nuevos = list(enlaces)
        for eid in (self._enlazar_textos(canon) if canon else []):
            if eid and eid not in nuevos:
                nuevos.append(eid)
        huecos = []
        for h in d.get("unfilled", [])[:MAX_HUECOS]:
            if isinstance(h, dict) and str(h.get("query", "")).strip():
                ancla = h.get("anchor")
                huecos.append({"query": str(h["query"]).strip(),
                               "anchor": str(ancla).strip() if isinstance(ancla, str) and ancla.strip() else None})
            elif isinstance(h, str) and h.strip():
                huecos.append({"query": h.strip(), "anchor": None})
        return sel, huecos, nuevos, plan

    # ------------------------------------------------------------ llamada 2

    def _salto(self, huecos: list[dict], aprobados: list[int],
               enlaces: list[str]) -> list[int]:
        """Por cada hueco, candidatos por DOS vias: (1) estructural — las
        aserciones del pasaje propio del ANCLA del hueco (enlazada por el
        diccionario; inmune a alias en otro idioma: 'The Magician' cuyo
        articulo habla de 'Ansiktet'); sin ancla enlazable, las entidades ya
        enlazadas cuyo nombre aparece en el hueco; (2) coseno del texto del
        hueco. Un mini-filtro adjudica (<= 2 hechos por hueco)."""
        if not huecos:
            return []
        vecs = self._embeber_textos([h["query"] for h in huecos])
        anclas = [h["anchor"] for h in huecos if h["anchor"]]
        eid_ancla = dict(zip(anclas, self._enlazar_textos(anclas))) if anclas else {}
        nuevos: list[int] = []
        for h, v in zip(huecos, vecs):
            eids = [eid_ancla[h["anchor"]]] if h["anchor"] and eid_ancla.get(h["anchor"]) else []
            if not eids:                                    # respaldo: nombre en el hueco
                hp = _plegar_alias(h["query"])
                for eid in enlaces:
                    ent = self.entradas.get(eid, {})
                    nombres = [ent.get("nombre", "")] + list(ent.get("alias", []))
                    if any(_plegar_alias(n) and _plegar_alias(n) in hp for n in nombres):
                        eids.append(eid)
            estructural: list[int] = []
            for eid in eids:
                pp = self.entradas.get(eid, {}).get("pasaje_propio")
                if pp:
                    estructural += self._aser_por_pasaje.get(pp, [])[:12]
            cos_top = [int(j) for j in np.argsort(-(self.aser_emb @ v))[:8]]
            cand = [j for j in dict.fromkeys(estructural + cos_top)
                    if j not in set(aprobados)][:MAX_CAND_SALTO]
            if not cand:
                continue
            lineas = "\n".join(
                f"{i}: [{self.aser_ids[j].rsplit('::', 1)[0]}] "
                f"{self.g.nodes[self.aser_ids[j]].get('descripcion','')[:120]}"
                for i, j in enumerate(cand))
            d = self._llm_json(f"{_PROMPT_SALTO}\nMissing slot: {h['query']}\nFacts:\n{lineas}")
            nuevos += [cand[int(i)] for i in d.get("selected", [])
                       if str(i).isdigit() and int(i) < len(cand)][:2]
        return list(dict.fromkeys(nuevos))

    # ------------------------------------------------------------ llamada 3

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

    def _paseo(self, query: str, qv: np.ndarray, sims: np.ndarray,
               aprobados: list[int]) -> list[str]:
        stilde = sims.copy()
        if aprobados:
            stilde[aprobados] = 1.0
        pers = self._semillas(query, [qv], stilde, aprobados)
        p = self._ppr(pers, stilde)
        return [self.nodos[i][6:] for i in np.argsort(-p)
                if self.nodos[i].startswith("chunk:")][:60]

    def buscar(self, query: str, k: int = 20) -> list[str]:
        enlaces0 = self._enlazar_menciones(query)
        qv = self._qvecs(query, "lite")[0]
        sims = self.aser_emb @ qv
        aprobados, huecos, enlaces, plan = self._analista(query, sims, enlaces0)
        self._ultimo_plan, self._enlaces_actuales = plan, enlaces
        titulos = self._paseo(query, qv, sims, aprobados)
        self._rondas = 0
        while self.usar_salto and huecos and self._rondas < self.max_rondas:
            self._rondas += 1
            buscables = [h for h in huecos if h["anchor"] or "[" not in h["query"]]
            if not buscables:
                break
            extra = [j for j in self._salto(buscables, aprobados, enlaces) if j not in aprobados]
            if not extra:
                break
            aprobados = aprobados + extra
            titulos = self._paseo(query, qv, sims, aprobados)
            pendientes = [h for h in huecos if h not in buscables]     # con comodin [X]
            if not pendientes or self._rondas >= self.max_rondas:
                break
            # re-planificar con los puentes descubiertos a la vista
            sel2, huecos, enlaces, plan2 = self._analista(query, sims, enlaces, extra=aprobados)
            nuevos = [j for j in sel2 if j not in aprobados]
            self._ultimo_plan, self._enlaces_actuales = plan2 or plan, enlaces
            plan = plan2 or plan
            if nuevos:
                aprobados = aprobados + nuevos
                titulos = self._paseo(query, qv, sims, aprobados)
        if self.usar_conjunto and (self.conjunto_en_comparacion or
                                   clasificar_tipo(query) != "comparison"):
            titulos = self._conjunto(query, "; ".join(plan) if plan else query, titulos)
        return titulos[:k]
