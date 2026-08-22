"""Puente entre la API y el grafo LangGraph: streaming de eventos para SSE.

Emite eventos de la TRAZA agentica (un 'paso' por nodo) y al final la respuesta
citada o una pausa HITL. La respuesta usa el grafo ya construido (denso + Claude/
OpenAI/Groq segun la seleccion). El checkpointer en memoria permite el HITL
(pausa/reanuda) dentro del proceso; en produccion multi-worker -> checkpointer Postgres.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _grafo():
    from asistente_vih.agent import construir_grafo
    return construir_grafo()


def _citas_de(estado_values: dict) -> list[dict]:
    hits = estado_values.get("hits", []) or []
    citas = []
    for i, h in enumerate(hits[:8], start=1):
        # El grafo guarda los hits como dicts (asdict); tolera tambien objetos Hit.
        d = h if isinstance(h, dict) else vars(h)
        citas.append({"n": i, "chunk_id": d.get("chunk_id", ""), "guia": d.get("guia", ""),
                      "seccion": d.get("seccion", ""), "pagina": d.get("pagina", ""),
                      "texto": (d.get("texto") or "")[:1200],
                      "vigencia": d.get("fecha_vigencia", ""),
                      "reasoning_trace": d.get("reasoning_trace") or []})
    return citas


def stream_eventos(pregunta: str, user_id: str, thread_id: str,
                   cfg_llm: dict | None = None, resume: str | None = None):
    """Generador de eventos (dicts) para SSE."""
    from langgraph.types import Command
    g = _grafo()
    config = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        entrada = Command(resume=resume)
    else:
        entrada = {"pregunta": pregunta, "user_id": user_id, "thread_id": thread_id,
                   "cfg_llm": cfg_llm or {}}

    try:
        for chunk in g.stream(entrada, config, stream_mode="updates"):
            for nodo in chunk:
                if nodo != "__interrupt__":
                    yield {"type": "paso", "nodo": nodo}
    except Exception as ex:
        yield {"type": "error", "mensaje": f"{type(ex).__name__}: {ex}"}
        return

    estado = g.get_state(config)
    intrs = [i for t in (estado.tasks or []) for i in (getattr(t, "interrupts", None) or [])]
    if intrs:  # el grafo se ha pausado (HITL)
        yield {"type": "hitl", "payload": intrs[0].value}
        return

    v = estado.values
    ver = v.get("veredicto") or {}
    ecl = v.get("ecl") or {}
    yield {"type": "respuesta",
           "texto": v.get("respuesta", ""),
           "abstenida": bool(v.get("abstenida")),
           "veredicto": ver.get("veredicto"),
           "citas": _citas_de(v),
           "ruta": v.get("ruta", []),
           # Detalle del proceso de razonamiento (para el desplegable del chat).
           "detalle": {
               "clasificacion": v.get("clasificacion") or {},
               "suficiencia": v.get("suficiencia") or {},
               "veredicto": ver,
               "ecl": ({"clase": ecl.get("clase"), "poblacion": ecl.get("poblacion"),
                        "n_miembros_en_corpus": ecl.get("n_miembros_en_corpus")}
                       if ecl else None),
           }}
    yield {"type": "fin"}


# --------------------------------------------- modo 'Razonar el caso' (bucle)
_PASO_DE_TRAZA = {"buscar": "recuperar", "preguntar": "hitl", "responder": "sintetizar"}


def stream_eventos_corazonamiento(pregunta: str, user_id: str, thread_id: str,
                                  cfg_llm: dict | None = None, resume: str | None = None):
    """Generador de eventos SSE para el modo de CO-RAZONAMIENTO.

    Camino separado del grafo: invoca el bucle de corazonamiento.py. Traduce su
    evento a la MISMA forma SSE que el grafo (paso / hitl / respuesta / fin), para
    que el frontend lo consuma sin cambios estructurales.
    """
    from asistente_vih.agent import corazonamiento as cr

    try:
        if resume is not None:
            ev = cr.responder_medico(thread_id, resume)
        else:
            ev = cr.iniciar(pregunta, user_id=user_id, thread_id=thread_id, cfg_llm=cfg_llm or {})
    except Exception as ex:
        yield {"type": "error", "mensaje": f"{type(ex).__name__}: {ex}"}
        return

    # Traza -> pasos en vivo (reusa las etiquetas del grafo cuando aplica).
    for entrada in ev.get("traza", []):
        verbo = entrada.split("(")[0].split(":")[0].strip()
        if verbo in _PASO_DE_TRAZA:
            yield {"type": "paso", "nodo": _PASO_DE_TRAZA[verbo]}

    tipo = ev.get("tipo")
    if tipo == "preguntar":
        yield {"type": "hitl", "payload": {
            "tipo": "corazonamiento",
            "motivo": "razonar el caso",
            "razonamiento": ev.get("razonamiento", ""),
            "criterio": ev.get("criterio", ""),
            "opciones": ev.get("opciones", []),
            "pregunta_al_medico": f"Segun {ev.get('criterio', 'la guia')}, ¿que aplica en tu caso?"}}
        return
    if tipo == "error":
        yield {"type": "error", "mensaje": ev.get("mensaje", "error")}
        return

    # respuesta | abstener
    yield {"type": "respuesta",
           "texto": ev.get("texto", ""),
           "abstenida": bool(ev.get("abstenida")) or tipo == "abstener",
           "veredicto": ev.get("veredicto"),
           "citas": ev.get("citas", []),
           "ruta": ev.get("traza", []),
           "detalle": {"veredicto": {"veredicto": ev.get("veredicto")}}}
    yield {"type": "fin"}
