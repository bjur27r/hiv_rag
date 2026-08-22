"""StateGraph del asistente clinico agentico (LangGraph).

Razona la ESTRATEGIA (clasificar, descomponer, recuperar, verificar, cuando
detenerse) con grounding estricto: cada afirmacion citada, verificada, y pausa
de supervision medica (HITL) ante decision seria o dato faltante.

Corre offline (modo simulado, sin claves) con un recuperador BM25 de respaldo;
con claves usa denso (OpenAI) + Claude para router/sintesis/verificacion.
"""
from __future__ import annotations

import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from dataclasses import asdict

from ..llm.client import (LLM, MODELO_ROUTER, MODELO_SINTESIS, MODELO_VERIFICADOR, _DEFAULTS)
from ..llm.generate import generar_respuesta
from ..llm.verify import verificar_fidelidad
from ..memory import MemoryStore
from ..memory import pii
from ..retrieval.base import Hit
from ..retrieval.catrag_retriever import CatRAGRetriever


def _a_hits(dics: list[dict]) -> list[Hit]:
    """Reconstruye objetos Hit desde los dicts guardados en el estado."""
    return [Hit(**d) for d in dics]

MAX_REC = 1   # reintentos de recuperacion
MAX_GEN = 1   # reintentos de generacion tras verificacion fallida


# --------------------------------------------------------------------- estado
class Estado(TypedDict, total=False):
    pregunta: str
    user_id: str
    thread_id: str
    pregunta_seud: str
    riesgo_pii: str
    clasificacion: dict
    hits: list           # list[Hit]
    hits_vectorial: list # Adaptive RAG
    hits_grafo: list     # Adaptive RAG
    suficiencia: dict    # veredicto CRAG con accion correctiva
    ecl: dict            # analisis exhaustivo de clase (agente ECL)
    intentos_rec: int
    intentos_gen: int
    borrador: str
    veredicto: dict
    respuesta: str
    abstenida: bool
    hitl_pregunta: str
    confirmacion: str
    contexto_previo: str
    cfg_llm: dict          # seleccion de modelo por peticion: {provider, router, sintesis, verificador}
    ruta: Annotated[list, operator.add]


def _llm(state: "Estado", rol: str) -> LLM:
    """Construye el LLM del rol segun la seleccion de la peticion (o los defaults)."""
    c = state.get("cfg_llm") or {}
    prov = c.get("provider")
    modelo = c.get(rol)
    if prov and not modelo:
        modelo = _DEFAULTS.get(prov, {}).get(rol)
    if not modelo:
        modelo = {"router": MODELO_ROUTER, "sintesis": MODELO_SINTESIS,
                  "verificador": MODELO_VERIFICADOR}[rol]
    return LLM(modelo, provider=prov)


# ------------------------------------------------------- router (clasificador)
ROUTER_SYS = """Clasifica la consulta clinica sobre VIH. Distingue primero el tipo:
- "general": pregunta sobre lo que dicen las guias, SIN referirse a un paciente concreto.
- "paciente": el medico describe un paciente concreto (sus datos, su situacion).

Reglas ESTRICTAS para no interrumpir de mas (la interrupcion solo para lo realmente serio):
- decision_seria = true SOLO si concurren las DOS cosas: (a) se plantea una DECISION o CAMBIO
  de tratamiento concreto para un escenario (no una pregunta de "que dice/recomienda la guia"),
  Y (b) hay un factor de riesgo EXPLICITO que la hace insegura: antecedente de fracaso o
  resistencia, contraindicacion grave, interaccion mayor, o decision irreversible.
  Son decision_seria=FALSE (aunque mencionen farmacos o cambios en abstracto): "¿que pauta de
  inicio recomienda la guia?", "¿que pruebas hacer antes de X?", "¿se puede usar X con FG<Y?",
  "¿que recomienda la guia en [poblacion]?", preguntas de mecanismo o de una sola guia.
- En consultas "general", datos_faltantes DEBE ser [] (se responde lo que dice la guia).
- En consultas "paciente", datos_faltantes solo datos que CAMBIARIAN la recomendacion y no
  constan (FG, CD4, genotipo), nunca datos de contexto genericos.

Devuelve: tipo_consulta, nivel (1-4), guias_candidatas, multi_aspecto (cruza varias guias),
logica_numerica (umbrales FG/CD4/carga viral/PEP), datos_faltantes, decision_seria, estrategia_recuperacion."""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "tipo_consulta": {"type": "string", "enum": ["general", "paciente"]},
        "nivel": {"type": "integer"},
        "guias_candidatas": {"type": "array", "items": {"type": "string"}},
        "multi_aspecto": {"type": "boolean"},
        "logica_numerica": {"type": "boolean"},
        "datos_faltantes": {"type": "array", "items": {"type": "string"}},
        "decision_seria": {"type": "boolean"},
        "estrategia_recuperacion": {"type": "string", "enum": ["vectorial", "grafo", "paralela"]},
    },
    "required": ["tipo_consulta", "nivel", "guias_candidatas", "multi_aspecto",
                 "logica_numerica", "datos_faltantes", "decision_seria", "estrategia_recuperacion"],
    "additionalProperties": False,
}

_RX_SERIA = re.compile(r"\b(cambiar?|cambio|fracaso|simplificar|suspender|switch)\b", re.I)
_RX_NUM = re.compile(r"\b(fg|filtrado|aclaramiento|cd4|carga viral|copias|72\s*h)\b", re.I)


_RX_PACIENTE = re.compile(r"\b(mi |soy |tengo |mi paciente|un paciente|este paciente|llevo )\b", re.I)


def _router_heuristico(q: str) -> dict:
    multi = (" y " in q.lower()) or len(re.findall(r"\b(tb|tuberculosis|embarazo|rifampicina|vacuna|hepatitis)\b", q, re.I)) >= 1
    estrategia = "paralela" if multi else "vectorial"
    return {"tipo_consulta": "paciente" if _RX_PACIENTE.search(q) else "general",
            "nivel": 3 if multi else 1, "guias_candidatas": [], "multi_aspecto": multi,
            "logica_numerica": bool(_RX_NUM.search(q)), "datos_faltantes": [],
            "decision_seria": bool(_RX_SERIA.search(q)), "estrategia_recuperacion": estrategia}


# ------------------------------------------------------------------ retriever
_RETR = {}  # cache de proceso


def _embeddings_disponibles() -> bool:
    import os
    prov = os.getenv("EMBEDDINGS_PROVIDER", "").lower()
    if not prov:
        prov = "openai" if os.getenv("OPENAI_API_KEY") else ("voyage" if os.getenv("VOYAGE_API_KEY") else "openai")
    return bool(os.getenv("OPENAI_API_KEY" if prov == "openai" else "VOYAGE_API_KEY"))


def _retrievers():
    if not _RETR:
        usado = "bm25"
        if _embeddings_disponibles():
            try:
                from ..retrieval.dense import DenseRetriever
                from ..retrieval.decompose import DecomposingRetriever
                d = DenseRetriever()
                _RETR["base"], _RETR["decomp"] = d, DecomposingRetriever(d)
                usado = "denso"
            except Exception:
                pass
        if usado == "bm25":
            from ..retrieval import BM25Retriever
            b = BM25Retriever.from_jsonl()
            _RETR["base"], _RETR["decomp"] = b, b
            
        try:
            import openai
            _RETR["catrag"] = CatRAGRetriever(openai.OpenAI())
        except Exception as e:
            print(f"CatRAG no inicializado: {e}")
            _RETR["catrag"] = None
            
    return _RETR["base"], _RETR["decomp"], _RETR.get("catrag")


# ------------------------------------------------------------------- nodos
def n_intake(state: Estado) -> dict:
    r = pii.scrub(state["pregunta"])
    uid, tid = state.get("user_id", "anon"), state.get("thread_id", "consulta")
    contexto = ""
    try:
        mem = MemoryStore()
        # Contexto ANTES de registrar el turno actual (solo turnos previos del hilo).
        contexto = mem.construir_contexto_sistema(uid, tid, r.texto)
        mem.registrar_turno(tid, uid, "medico", state["pregunta"])  # seudonimiza dentro
    except Exception:
        pass

    pregunta_final = r.texto
    if contexto and "Contexto de esta conversacion" in contexto:
        try:
            llm_router = _llm(state, "router")
            prompt_rew = (f"Reescribe la ultima pregunta del usuario para que sea autocontenida "
                          f"y se entienda sin leer el historial. Incorpora sujetos, verbos o contextos "
                          f"omitidos. Responde SOLO con la pregunta reescrita.\n\n"
                          f"Historial:\n{contexto}\n\nPregunta actual: {pregunta_final}\n\nPregunta reescrita:")
            res = llm_router.texto("Eres un asistente util.", prompt_rew, max_tokens=150)
            if res and len(res) > 5:
                pregunta_final = res.strip().strip('"').strip("'")
        except Exception as e:
            print(f"Error reescribiendo query: {e}")

    return {"pregunta_seud": pregunta_final, "riesgo_pii": r.riesgo_reident,
            "contexto_previo": contexto, "intentos_rec": 0, "intentos_gen": 0, "ruta": ["intake"]}


def n_router(state: Estado) -> dict:
    q = state["pregunta_seud"]
    cls = _llm(state, "router").json(ROUTER_SYS, q, ROUTER_SCHEMA, max_tokens=500,
                                  simulado_valor=_router_heuristico(q))
    return {"clasificacion": cls, "ruta": ["router"]}


# Criba barata: solo se invoca el detector LLM del ECL si la pregunta "huele"
# a pregunta de CLASE (todos los miembros de una categoria farmacologica).
_RX_CLASE = re.compile(
    r"\b(que|qué|cuales|cuáles)\b.{0,60}\b(inhibidores?|farmacos?|fármacos?|"
    r"antirretrovirales|analogos?|análogos?|pautas|opciones|itian|itinn|insti)\b", re.I)


def n_ecl(state: Estado) -> dict:
    """Agente ECL (Sprint 3): en preguntas de clase, resuelve la subsuncion
    SNOMED y adjunta la tabla EXHAUSTIVA (incluidos los negativos) para la
    sintesis. No sustituye la recuperacion: la complementa."""
    q = state["pregunta_seud"]
    if not _RX_CLASE.search(q):
        return {"ruta": ["ecl(skip)"]}
    try:
        from ..terminologia.ecl import consultar
        r = consultar(q, llm=_llm(state, "router"))
    except Exception as e:
        print(f"ECL no disponible: {e}")
        return {"ruta": ["ecl(error)"]}
    if not r.get("aplica") or r.get("error") or not r.get("miembros") and not r.get("sin_recomendacion_que_cumpla"):
        return {"ruta": ["ecl(no_aplica)"]}
    return {"ecl": r, "ruta": [f"ecl({len(r.get('miembros', []))} miembros)"]}


def _bloque_ecl(r: dict) -> str:
    if not r:
        return ""
    lineas = [f"[Analisis exhaustivo de clase — terminologia SNOMED, verificado]",
              f"Clase: {r.get('clase')} | poblacion: {r.get('poblacion') or '-'} | "
              f"miembros de la clase presentes en el corpus: {r.get('n_miembros_en_corpus')}"]
    for m in r.get("miembros", []):
        for rec in m.get("recomendaciones", [])[:2]:
            lineas.append(f" - {m['entidad']} (SCTID {m['sctid']}): {rec['accion']}"
                          f"{', grado ' + rec['grado'] if rec.get('grado') else ''} "
                          f"(guia {rec['guia']}, pag {rec['pagina']}, vigencia {rec.get('vigencia')}): "
                          f"\"{rec['texto'][:180]}\"")
    sin = r.get("sin_recomendacion_que_cumpla") or []
    if sin:
        lineas.append(f" - SIN recomendacion que cumpla los filtros en el corpus: {', '.join(sin[:10])}")
    lineas.append("(Usa este analisis para asegurar EXHAUSTIVIDAD: menciona todos los miembros, "
                  "incluidos los que no tienen recomendacion. Cita cada recomendacion por su guia y pagina.)")
    return "\n".join(lineas)


def n_recuperar_vectorial(state: Estado) -> dict:
    base, decomp, _ = _retrievers()
    cls = state.get("clasificacion", {})
    retr = decomp if cls.get("multi_aspecto") else base
    k = 20 if cls.get("multi_aspecto") else 8
    k += 6 * state.get("intentos_rec", 0)
    hits = retr.search(state["pregunta_seud"], k=k)
    return {"hits_vectorial": [asdict(h) for h in hits], "ruta": ["recuperar_vec"]}


def n_recuperar_grafo(state: Estado) -> dict:
    _, _, catrag = _retrievers()
    if not catrag:
        return {"hits_grafo": [], "ruta": ["recuperar_grafo_skip"]}
    k = 10 + 5 * state.get("intentos_rec", 0)
    # El router elige el perfil del PPR: 'grafo' (cobertura profunda) para
    # consultas multi-guia o cuando pidio explicitamente la rama de grafo;
    # 'balanceado' (precision temprana) para el resto.
    cls = state.get("clasificacion", {})
    modo = "grafo" if (cls.get("multi_aspecto")
                       or cls.get("estrategia_recuperacion") == "grafo") else "balanceado"
    hits = catrag.search(state["pregunta_seud"], k=k, modo=modo,
                         intencion={"tipo_consulta": cls.get("tipo_consulta"),
                                    "logica_numerica": cls.get("logica_numerica")})
    return {"hits_grafo": [asdict(h) for h in hits],
            "ruta": [f"recuperar_grafo({modo})"]}


_RERANKER = None


def _get_reranker():
    """Cross-encoder de produccion (Sprint 1), singleton perezoso.

    Si sentence-transformers no esta instalado, el juez degrada al orden de
    fusion sin reordenar (nunca rompe el flujo)."""
    global _RERANKER
    if _RERANKER is None:
        try:
            from ..retrieval.rerank import CrossEncoderReranker
            _RERANKER = CrossEncoderReranker()
        except Exception as e:
            print(f"Reranker no disponible ({e}); juez sin reordenar.")
            _RERANKER = False
    return _RERANKER or None


def n_juez_recuperacion(state: Estado) -> dict:
    cls = state.get("clasificacion", {})
    estrategia = cls.get("estrategia_recuperacion", "vectorial")

    hits_vec = state.get("hits_vectorial", [])
    hits_graf = state.get("hits_grafo", [])

    if estrategia == "vectorial":
        pool, etiqueta = _a_hits(hits_vec), "juez_vec"
    elif estrategia == "grafo":
        pool, etiqueta = _a_hits(hits_graf), "juez_graf"
    else:
        # Estrategia paralela: fusion RRF (robusta a escalas de score distintas).
        from ..retrieval.hybrid import rrf
        pool, etiqueta = rrf([_a_hits(hits_graf), _a_hits(hits_vec)], k=30), "juez_paralelo"

    # Sprint 1: el cross-encoder reordena el pool; la sintesis lee los 8
    # primeros, asi que el orden fino ES la evidencia que ve el LLM.
    rr = _get_reranker()
    if rr is not None and pool:
        try:
            from ..retrieval.rerank import rerank_diverso
            pool = rerank_diverso(rr, state["pregunta_seud"], pool, k=15)
            etiqueta += "+rerank"
        except Exception as e:
            print(f"Rerank fallo ({e}); orden de fusion.")
    return {"hits": [asdict(h) for h in pool[:15]], "ruta": [etiqueta]}


SUF_SYS = """Eres el evaluador de SUFICIENCIA de un asistente clinico sobre guias GeSIDA.
Dado la pregunta, su clasificacion y el pool de fragmentos recuperados, decide si el pool
BASTA para responder con citas. Acciones:
- continuar: el pool cubre la pregunta (todas las guias implicadas si es multi-aspecto).
- ampliar: falta material del MISMO eje; reintentar con mas resultados.
- cambiar_modo: la pregunta cruza guias/aspectos que la estrategia actual no trae; pasar a paralela.
- abstener: el corpus no cubre la pregunta (no fuerces una respuesta).
Se conservador con 'abstener' (solo si es claro) y no pidas ampliar por perfeccionismo."""

SUF_SCHEMA = {
    "type": "object",
    "properties": {
        "suficiente": {"type": "boolean"},
        "accion": {"type": "string", "enum": ["continuar", "ampliar", "cambiar_modo", "abstener"]},
        "motivo": {"type": "string"},
    },
    "required": ["suficiente", "accion", "motivo"],
    "additionalProperties": False,
}


def n_suficiencia(state: Estado) -> dict:
    """Evaluador CRAG (Sprint 2): veredicto con ACCION correctiva, con la
    heuristica original como fallback y como valor en modo simulado."""
    hits = state.get("hits", [])
    cls = state.get("clasificacion", {})
    guias = sorted({h["guia"] for h in hits})
    suf_heur = bool(hits) and (not cls.get("multi_aspecto") or len(guias) >= 2)
    heur = {"suficiente": suf_heur,
            "accion": "continuar" if suf_heur else ("ampliar" if hits else "abstener"),
            "motivo": "heuristica"}
    resumen = "\n".join(
        f"[{i}] guia={h['guia']} secc={h.get('seccion','')[:50]} :: {h.get('texto','')[:140]}"
        for i, h in enumerate(hits[:10], 1)) or "(pool vacio)"
    try:
        v = _llm(state, "router").json(
            SUF_SYS,
            f"Pregunta: {state['pregunta_seud']}\n"
            f"Clasificacion: multi_aspecto={cls.get('multi_aspecto')}, "
            f"estrategia={cls.get('estrategia_recuperacion')}, "
            f"guias_candidatas={cls.get('guias_candidatas')}\n"
            f"Guias en el pool: {guias}\nPool:\n{resumen}",
            SUF_SCHEMA, max_tokens=300, simulado_valor=heur)
    except Exception:
        v = heur
    out = {"suficiencia": v, "ruta": [f"suficiencia({v.get('accion')})"]}
    if v.get("accion") in ("ampliar", "cambiar_modo"):
        out["intentos_rec"] = state.get("intentos_rec", 0) + 1
    if v.get("accion") == "cambiar_modo":
        out["clasificacion"] = {**cls, "estrategia_recuperacion": "paralela"}
    return out


def n_hitl(state: Estado) -> dict:
    cls = state.get("clasificacion", {})
    motivo = "decision seria" if cls.get("decision_seria") else "dato faltante"
    falta = cls.get("datos_faltantes") or ["confirmacion del medico"]
    # PAUSA el grafo y espera al medico; se reanuda con Command(resume=...)
    aporte = interrupt({"motivo": motivo, "falta": falta,
                        "pregunta_al_medico": f"Antes de responder ({motivo}), aporta/confirma: {falta}"})
    return {"confirmacion": str(aporte), "ruta": ["hitl"]}


def n_sintetizar(state: Estado) -> dict:
    extra = (f"\n[Contexto aportado por el medico: {state['confirmacion']}]"
             if state.get("confirmacion") else "")
    # Memoria conversacional: perfil de estilo + turnos previos relevantes (filtrado).
    gen = generar_respuesta(state["pregunta_seud"] + extra, _a_hits(state.get("hits", [])),
                            llm=_llm(state, "sintesis"), preferencias=state.get("contexto_previo", ""),
                            contexto_extra=_bloque_ecl(state.get("ecl") or {}))
    return {"borrador": gen["respuesta"], "abstenida": gen["abstenida"],
            "intentos_gen": state.get("intentos_gen", 0) + 1, "ruta": ["sintetizar"]}


def n_verificar(state: Estado) -> dict:
    hits = _a_hits(state.get("hits", []))
    v = verificar_fidelidad(state["pregunta_seud"], state["borrador"], hits,
                            llm=_llm(state, "verificador"),
                            contexto_extra=_bloque_ecl(state.get("ecl") or {}))
    # Panel de lentes (Sprint 2): polaridad deontica + umbrales numericos.
    from ..llm.lentes import aplicar_lentes
    v = aplicar_lentes(state["borrador"], hits, v)
    return {"veredicto": v, "ruta": ["verificar" + ("+lentes" if v.get("lentes") else "")]}


def n_recuperar_dirigido(state: Estado) -> dict:
    """Cierre del bucle (Sprint 2): las afirmaciones SIN RESPALDO del veredicto
    se convierten en consultas dirigidas; la evidencia nueva se fusiona con la
    existente y se reordena antes de regenerar. (Antes se regeneraba con la
    misma evidencia insuficiente.)"""
    claims = (state.get("veredicto") or {}).get("afirmaciones_sin_respaldo") or []
    hits = _a_hits(state.get("hits", []))
    if claims:
        base, _, _ = _retrievers()
        vistos = {h.chunk_id for h in hits}
        for claim in claims[:3]:
            for h in base.search(claim[:300], k=3):
                if h.chunk_id not in vistos:
                    vistos.add(h.chunk_id)
                    hits.append(h)
        rr = _get_reranker()
        if rr is not None:
            try:
                hits = rr.rerank(state["pregunta_seud"], hits, k=15)
            except Exception:
                pass
    return {"hits": [asdict(h) for h in hits[:15]],
            "ruta": [f"recuperar_dirigido({len(claims)} afirmaciones)"]}


def n_abstener(state: Estado) -> dict:
    return {"respuesta": "Las guias proporcionadas no permiten una respuesta fiable a esta consulta.",
            "abstenida": True, "ruta": ["abstener"]}


def n_finalizar(state: Estado) -> dict:
    return {"respuesta": state.get("borrador", ""), "ruta": ["finalizar"]}


def n_audit(state: Estado) -> dict:
    try:
        mem = MemoryStore()
        guias = sorted({h["guia"] for h in state.get("hits", [])})
        mem.registrar_turno(
            state.get("thread_id", "consulta"), state.get("user_id", "anon"), "asistente",
            state.get("respuesta", ""), guias_citadas=guias,
            nivel=state.get("clasificacion", {}).get("nivel"),
            hitl="hitl" in state.get("ruta", []), abstenido=state.get("abstenida", False),
            ruta="->".join(state.get("ruta", [])))
    except Exception:
        pass
    return {"ruta": ["audit"]}


# --------------------------------------------------------------- aristas cond.
def _tras_router(state: Estado) -> list[str]:
    est = state.get("clasificacion", {}).get("estrategia_recuperacion", "vectorial")
    if est == "vectorial": return ["recuperar_vectorial"]
    if est == "grafo": return ["recuperar_grafo"]
    return ["recuperar_vectorial", "recuperar_grafo"]

def _tras_suficiencia(state: Estado) -> list[str]:
    v = state.get("suficiencia", {})
    hits = state.get("hits", [])
    cls = state.get("clasificacion", {})
    accion = v.get("accion", "continuar")
    if accion in ("ampliar", "cambiar_modo") and state.get("intentos_rec", 0) <= MAX_REC:
        est = cls.get("estrategia_recuperacion", "vectorial")
        if est == "vectorial": return ["recuperar_vectorial"]
        if est == "grafo": return ["recuperar_grafo"]
        return ["recuperar_vectorial", "recuperar_grafo"]
    if not hits or accion == "abstener":
        return ["abstener"]
    paciente = cls.get("tipo_consulta") == "paciente"
    if cls.get("decision_seria") or (paciente and cls.get("datos_faltantes")):
        return ["hitl"]
    return ["sintetizar"]


def _tras_verificar(state: Estado) -> str:
    v = state.get("veredicto", {})
    if v.get("fiel"):
        return "finalizar"
    if state.get("intentos_gen", 0) <= MAX_GEN:
        # Con afirmaciones concretas sin respaldo: buscar ESO antes de regenerar.
        if v.get("afirmaciones_sin_respaldo"):
            return "recuperar_dirigido"
        return "sintetizar"
    return "abstener"


# --------------------------------------------------------------- construccion
def construir_grafo(checkpointer=None):
    g = StateGraph(Estado)
    for nombre, fn in [("intake", n_intake), ("router", n_router), ("ecl", n_ecl),
                       ("recuperar_vectorial", n_recuperar_vectorial),
                       ("recuperar_grafo", n_recuperar_grafo),
                       ("juez", n_juez_recuperacion),
                       ("suficiencia", n_suficiencia), ("hitl", n_hitl), ("sintetizar", n_sintetizar),
                       ("verificar", n_verificar), ("recuperar_dirigido", n_recuperar_dirigido),
                       ("abstener", n_abstener),
                       ("finalizar", n_finalizar), ("audit", n_audit)]:
        g.add_node(nombre, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "router")
    g.add_edge("router", "ecl")

    g.add_conditional_edges("ecl", _tras_router,
                            ["recuperar_vectorial", "recuperar_grafo"])
                            
    g.add_edge("recuperar_vectorial", "juez")
    g.add_edge("recuperar_grafo", "juez")
    
    g.add_edge("juez", "suficiencia")
    
    g.add_conditional_edges("suficiencia", _tras_suficiencia,
                            ["recuperar_vectorial", "recuperar_grafo", "abstener", "hitl", "sintetizar"])
    g.add_edge("hitl", "sintetizar")
    g.add_edge("sintetizar", "verificar")
    g.add_conditional_edges("verificar", _tras_verificar,
                            {"finalizar": "finalizar", "sintetizar": "sintetizar",
                             "recuperar_dirigido": "recuperar_dirigido", "abstener": "abstener"})
    g.add_edge("recuperar_dirigido", "sintetizar")
    g.add_edge("finalizar", "audit")
    g.add_edge("abstener", "audit")
    g.add_edge("audit", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)
