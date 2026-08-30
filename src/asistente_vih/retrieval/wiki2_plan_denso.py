"""CONTROL SIN GRAFO con el mismo presupuesto de LLM que el Plan A v4.

Pregunta que responde: ¿lo que gana el Plan A v4 lo pone el grafo (aserciones
reificadas + memoria canonica + paseo) o lo pone el planificador LLM? Este
recuperador conserva TODO lo que es LLM y quita TODO lo que es grafo:

  conserva  analista (plan + seleccion + huecos con ancla), salto dirigido por
            hueco con rondas de re-planificacion, seleccion final de conjunto,
            mismas consignas, mismas reglas, mismas demostraciones de
            calibracion, mismos topes y el mismo modelo -> ~3 llamadas/pregunta.
  quita     OpenIE, memoria canonica, grafo, PPR y frontera.
  sustituye hechos por PASAJES: el analista lee el top-20 denso (titulo +
            comienzo del texto, mismo presupuesto de caracteres que 40 hechos);
            el salto busca el hueco por coseno sobre pasajes (+ el pasaje cuyo
            TITULO es el ancla, variante 'titulos'); el orden final es
            "seleccionados primero, despues por coseno" en vez del paseo.

Variantes:
  titulos=True  (por defecto) indice de titulos plegados: las menciones de la
                pregunta y las formas canonicas del analista que coinciden con
                un titulo se fuerzan como candidatos/anclas. Es un mecanismo
                estandar sin grafo (equivale al "pasaje propio" sin diccionario).
  titulos=False denso puro: solo cosenos.

Indice necesario: emb_chunks_<split>.npz (ya existe para los 4 splits).

    r = PlanDenso(split="sondeo", modelo="gpt-4o-mini")
    agg = evaluar(r.buscar, preguntas, nombre="denso_plan_4omini_sondeo")
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3

import numpy as np
from openai import OpenAI

from .wiki2_catrag import (WIKI2_DIR, MODELO_EMB, _plegar_alias, _plegar_con_desamb,
                           _cargar_corpus, spans_entidad)
from .wiki2_plan import _PROMPT_CONJUNTO, MAX_HUECOS

TOP_ANALISTA = 20        # pasajes que lee el analista (20 x 260 car. ~ 40 hechos x 130)
CAR_ANALISTA = 260
TOP_SALTO = 8            # pasajes por coseno del hueco
CAR_SALTO = 320
MAX_CAND_SALTO = 12

_PROMPT_ANALISTA_P = """You are the retrieval analyst of a multi-hop question answering system.

Given a question and a numbered list of candidate passages (title in brackets, then the beginning of the text) retrieved from the corpus, reply with STRICT JSON:

{"mentions": [{"text": "...", "canonical": "<the exact Wikipedia article title form>"}],
 "plan":     ["<slot 1, e.g. 'director of film A -> X'>", "<slot 2, e.g. 'date of death of X'>", ...],
 "selected": [<indices of passages that fill a slot>],
 "unfilled": [{"query": "<slot with NO selected passage, phrased as a retrieval query>",
               "anchor": "<the known entity the slot is about, exactly as named in a passage or in the question; null if the slot is about a still-unknown [X]>"}]}

Rules:
- The PLAN lists every fact needed to answer, as slots, INCLUDING intermediate hops.
- SELECT every candidate passage that fills a slot (up to 8). A passage naming the director/performer/
  father is essential even if it carries no date.
- If a role in the question (performer, director, founder...) has SEVERAL candidate entities in
  the passages, select them ALL. Never use your own knowledge to choose among them or to skip a hop.
- In the plan, write an entity name ONLY if a candidate passage states it; otherwise use [X].
  NEVER fill a slot from your own world knowledge, and never select a passage about an entity that
  no candidate passage connects to the question.
- "unfilled" lists slots no candidate passage covers. "anchor" is the known entity the slot asks
  about (the film, the person...); a slot about a still-unknown [X] has anchor null.

Example 1
Question: Which film has the director who died first, Convoy Buddies or Tip Toes?
Passages: 0: [Convoy Buddies] Convoy Buddies is a 1975 Italian comedy film directed by Giuliano Carnimeo.  1: [Tip Toes] Tip Toes is a 1927 British silent comedy film directed by Herbert Wilcox and starring Dorothy Gish.  2: [Herbert Wilcox] Herbert Wilcox (19 April 1890 - 15 May 1977) was a British film producer and director.  3: [Dorothy Gish] Dorothy Gish was an American actress of the silent era.
{"mentions": [{"text": "Convoy Buddies", "canonical": "Convoy Buddies"}, {"text": "Tip Toes", "canonical": "Tip Toes"}],
 "plan": ["director of Convoy Buddies -> Giuliano Carnimeo", "date of death of Giuliano Carnimeo", "director of Tip Toes -> Herbert Wilcox", "date of death of Herbert Wilcox"],
 "selected": [0, 1, 2],
 "unfilled": [{"query": "date of death of Giuliano Carnimeo", "anchor": "Giuliano Carnimeo"}]}

Example 2
Question: Who is the mother of the director of film Zoolander 2?
Passages: 0: [Zoolander 2] Zoolander 2 is a 2016 American action comedy film. It stars Owen Wilson and Penelope Cruz.  1: [Ben Stiller] Benjamin Edward Meara Stiller is an American actor, the son of comedians Jerry Stiller and Anne Meara.  2: [Zoolander] Zoolander is a 2001 American comedy film directed by Ben Stiller.
{"mentions": [{"text": "Zoolander 2", "canonical": "Zoolander 2"}],
 "plan": ["director of Zoolander 2 -> [X]", "mother of [X]"],
 "selected": [],
 "unfilled": [{"query": "director of Zoolander 2", "anchor": "Zoolander 2"}, {"query": "mother of [X]", "anchor": null}]}
"""

_PROMPT_SALTO_P = """Select the passages (by index) that fill this missing slot of the plan. Each passage shows
[its title]: the passage about the slot's entity DOES refer to it even if it uses a different or
original-language title for the same work. If none fills the slot, reply with an empty list.
STRICT JSON: {"selected": [<indices>]}
"""


class PlanDenso:
    def __init__(self, split: str = "sondeo", modelo: str = "gpt-4o-mini", titulos: bool = True,
                 usar_conjunto: bool = True, usar_salto: bool = True, max_rondas: int = 3,
                 replan_sin_extra: bool = False, car_ancla_replan: int = 700):
        """replan_sin_extra (variante 'plus', post-hoc): si tras un salto quedan
        huecos con comodin [X], el analista re-planifica AUNQUE el salto no haya
        aportado pasajes nuevos, viendo los pasajes-ancla completos (hasta
        car_ancla_replan caracteres). Refuerza al control cuando el puente esta
        escrito en el pasaje-ancla pero el analista no lo extrajo del fragmento
        de 260 caracteres."""
        self.split, self.modelo, self.titulos = split, modelo, titulos
        self.replan_sin_extra, self.car_ancla_replan = replan_sin_extra, car_ancla_replan
        self.usar_conjunto, self.usar_salto, self.max_rondas = usar_conjunto, usar_salto, max_rondas
        self.client = OpenAI()
        corpus = _cargar_corpus(split)
        self.titulos_lista = [c["title"] for c in corpus]
        self.textos = {c["title"]: c["text"] for c in corpus}
        ruta = WIKI2_DIR / f"emb_chunks_{split}.npz"
        if ruta.exists():
            d = np.load(ruta, allow_pickle=False)
            assert list(d["ids"]) == self.titulos_lista, "emb_chunks no alineado con el corpus"
            self.chunk_emb = d["mat"]
        else:
            textos = [c["title"] + ". " + c["text"] for c in corpus]
            mat = np.zeros((len(textos), 1536), dtype=np.float32)
            for i in range(0, len(textos), 512):
                for j, v in enumerate(self._embeber(textos[i:i + 512])):
                    mat[i + j] = v
            np.savez_compressed(ruta, ids=np.array(self.titulos_lista), mat=mat)
            self.chunk_emb = mat
        # indice de titulos (variante 'titulos'): plegado con desambiguador y sin el
        self.idx_titulo: dict[str, list[int]] = {}
        self.idx_titulo_desamb: dict[str, list[int]] = {}
        for i, t in enumerate(self.titulos_lista):
            self.idx_titulo.setdefault(_plegar_alias(t), []).append(i)
            if "(" in t:
                self.idx_titulo_desamb.setdefault(_plegar_con_desamb(t), []).append(i)
        self._cache = sqlite3.connect(WIKI2_DIR / "plan_cache.sqlite", timeout=120)
        self._cache.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, r TEXT)")
        self._clientes: dict[str, tuple[OpenAI, str]] = {}
        self.n_llamadas = 0

    # ------------------------------------------------------------ utilidades

    def _embeber(self, textos: list[str]) -> list[np.ndarray]:
        r = self.client.embeddings.create(input=textos, model=MODELO_EMB)
        out = []
        for dat in r.data:
            v = np.array(dat.embedding, dtype=np.float32)
            out.append(v / max(np.linalg.norm(v), 1e-9))
        return out

    def _cliente_para(self, modelo: str) -> tuple[OpenAI, str]:
        if modelo not in self._clientes:
            if modelo.startswith("deepseek"):
                self._clientes[modelo] = (OpenAI(base_url="https://api.deepseek.com",
                                                api_key=os.environ["DEEPSEEK_API_KEY"]), modelo)
            elif modelo.startswith("groq/"):
                self._clientes[modelo] = (OpenAI(base_url="https://api.groq.com/openai/v1",
                                                api_key=os.environ["GROQ_API_KEY"]), modelo.split("/", 1)[1])
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
                r = cli.chat.completions.create(model=nombre, temperature=0.0,
                                                response_format={"type": "json_object"},
                                                messages=[{"role": "user", "content": prompt}])
                crudo = r.choices[0].message.content
            except Exception:
                r = cli.chat.completions.create(model=nombre, temperature=0.0,
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

    def _titulos_de(self, texto: str) -> list[int]:
        """Indice de titulos: primero con desambiguador conservado, luego plegado
        (con y sin parentesis). Solo en la variante 'titulos'."""
        if not self.titulos or not texto.strip():
            return []
        if "(" in texto:
            ids = self.idx_titulo_desamb.get(_plegar_con_desamb(texto))
            if ids:
                return ids
        for forma in ([texto] + ([texto.split("(")[0]] if "(" in texto else [])):
            ids = self.idx_titulo.get(_plegar_alias(forma))
            if ids:
                return ids[:3]
        return []

    def _enlazar_pregunta(self, query: str) -> list[int]:
        """Menciones de la pregunta -> pasajes cuyo titulo coincide (sin diccionario)."""
        out = []
        for span in spans_entidad(query):
            ids = self._titulos_de(span)
            if not ids and re.search(r"\s(and|or|&)\s", span):
                for p in re.split(r"\s(?:and|or|&)\s", span):
                    ids += self._titulos_de(p.strip())
            out += ids
        return list(dict.fromkeys(out))

    def _linea(self, i: int, car: int) -> str:
        t = self.titulos_lista[i]
        return f"[{t}] {self.textos.get(t, '')[:car]}"

    # ------------------------------------------------------------ llamadas

    def _analista(self, query: str, sims: np.ndarray, forzados: list[int],
                  extra: list[int] = (), car_ancla: int = CAR_ANALISTA) -> tuple[list[int], list[dict], list[int], list[str]]:
        top = [int(j) for j in np.argsort(-sims)[:TOP_ANALISTA]]
        orden = list(dict.fromkeys(forzados + top + list(extra)))
        anclados = set(forzados) | set(extra)
        lineas = "\n".join(f"{i}: {self._linea(j, car_ancla if j in anclados else CAR_ANALISTA)}"
                            for i, j in enumerate(orden))
        d = self._llm_json(f"{_PROMPT_ANALISTA_P}\nQuestion: {query}\nPassages:\n{lineas}")
        plan = [p for p in d.get("plan", []) if isinstance(p, str)]
        n_plan = len(plan) or 4
        sel = [orden[int(i)] for i in d.get("selected", [])
               if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(orden)]
        sel = list(dict.fromkeys(sel))[:min(8, max(2, n_plan))]
        anclas = list(forzados)
        for m in d.get("mentions", []):
            if isinstance(m, dict):
                for j in self._titulos_de((m.get("canonical") or m.get("text") or "").strip()):
                    if j not in anclas:
                        anclas.append(j)
        huecos = []
        for h in d.get("unfilled", [])[:MAX_HUECOS]:
            if isinstance(h, dict) and str(h.get("query", "")).strip():
                a = h.get("anchor")
                huecos.append({"query": str(h["query"]).strip(),
                               "anchor": str(a).strip() if isinstance(a, str) and a.strip() else None})
            elif isinstance(h, str) and h.strip():
                huecos.append({"query": h.strip(), "anchor": None})
        return sel, huecos, anclas, plan

    def _salto(self, huecos: list[dict], aprobados: list[int]) -> list[int]:
        if not huecos:
            return []
        vecs = self._embeber([h["query"] for h in huecos])
        nuevos: list[int] = []
        for h, v in zip(huecos, vecs):
            estructural = self._titulos_de(h["anchor"]) if h["anchor"] else []
            cos_top = [int(j) for j in np.argsort(-(self.chunk_emb @ v))[:TOP_SALTO]]
            cand = [j for j in dict.fromkeys(estructural + cos_top) if j not in set(aprobados)][:MAX_CAND_SALTO]
            if not cand:
                continue
            lineas = "\n".join(f"{i}: {self._linea(j, CAR_SALTO)}" for i, j in enumerate(cand))
            d = self._llm_json(f"{_PROMPT_SALTO_P}\nMissing slot: {h['query']}\nPassages:\n{lineas}")
            nuevos += [cand[int(i)] for i in d.get("selected", [])
                       if str(i).isdigit() and int(i) < len(cand)][:2]
        return list(dict.fromkeys(nuevos))

    def _conjunto(self, query: str, plan_txt: str, titulos: list[str]) -> list[str]:
        lineas = "\n".join(f"{i}: [{t}] {self.textos.get(t, '')[:140]}" for i, t in enumerate(titulos[:12]))
        d = self._llm_json(f"{_PROMPT_CONJUNTO}\nQuestion: {query}\nPlan: {plan_txt}\nPassages:\n{lineas}")
        idx = [int(i) for i in d.get("selected", []) if str(i).isdigit() and int(i) < min(12, len(titulos))]
        elegidos = [titulos[i] for i in idx]
        for t in titulos[:2]:
            if t not in elegidos:
                elegidos.insert(0, t)
        for t in titulos:
            if len(elegidos) >= 5:
                break
            if t not in elegidos:
                elegidos.append(t)
        return elegidos[:5] + [t for t in titulos if t not in elegidos[:5]]

    # ------------------------------------------------------------ flujo

    def _orden(self, sims: np.ndarray, aprobados: list[int], anclas: list[int]) -> list[str]:
        """Sustituto del paseo: aprobados por el analista/salto primero (en su
        orden), luego los pasajes-ancla, luego el resto por coseno."""
        cabeza = list(dict.fromkeys(aprobados + anclas))
        resto = [int(j) for j in np.argsort(-sims) if int(j) not in set(cabeza)]
        return [self.titulos_lista[j] for j in (cabeza + resto)[:60]]

    def buscar(self, query: str, k: int = 20) -> list[str]:
        qv = self._embeber([query])[0]
        sims = self.chunk_emb @ qv
        forzados = self._enlazar_pregunta(query)
        aprobados, huecos, anclas, plan = self._analista(query, sims, forzados)
        self._ultimo_plan = plan
        titulos = self._orden(sims, aprobados, anclas)
        self._rondas = 0
        while self.usar_salto and huecos and self._rondas < self.max_rondas:
            self._rondas += 1
            buscables = [h for h in huecos if h["anchor"] or "[" not in h["query"]]
            if not buscables:
                break
            extra = [j for j in self._salto(buscables, aprobados) if j not in aprobados]
            if extra:
                aprobados = aprobados + extra
                titulos = self._orden(sims, aprobados, anclas)
            elif not self.replan_sin_extra:
                break
            pendientes = [h for h in huecos if h not in buscables]
            if not pendientes or self._rondas >= self.max_rondas:
                break
            sel2, huecos, anclas, plan2 = self._analista(
                query, sims, anclas, extra=aprobados,
                car_ancla=self.car_ancla_replan if self.replan_sin_extra else CAR_ANALISTA)
            plan = plan2 or plan; self._ultimo_plan = plan
            nuevos = [j for j in sel2 if j not in aprobados]
            if nuevos:
                aprobados = aprobados + nuevos
                titulos = self._orden(sims, aprobados, anclas)
        if self.usar_conjunto:
            titulos = self._conjunto(query, "; ".join(plan) if plan else query, titulos)
        return titulos[:k]
