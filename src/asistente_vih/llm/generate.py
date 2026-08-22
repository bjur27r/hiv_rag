"""Generacion de la respuesta CITADA a partir de los fragmentos recuperados.

El system prompt impone el grounding estricto: la respuesta se apoya SOLO en los
fragmentos, cada afirmacion lleva su cita, y si la guia no lo cubre se abstiene.
El modelo no aporta conocimiento propio (eso es trabajo del verificador comprobar).
"""
from __future__ import annotations

from ..retrieval.base import Hit
from .client import LLM, MODELO_SINTESIS

SYSTEM = """Eres un asistente clinico que responde SOLO a partir de las guias GeSIDA de VIH \
que se te proporcionan como fragmentos numerados. Reglas innegociables:
1. Usa UNICAMENTE la informacion de los fragmentos. No aportes conocimiento propio.
2. Cada afirmacion clinica debe ir seguida de su cita: [n] (guia, seccion, pagina) y, si \
consta, el grado de evidencia.
3. Si los fragmentos NO cubren la pregunta, abstente con claridad: "Las guias proporcionadas \
no recogen esta cuestion." No inventes.
4. Cuando proceda, indica que opciones quedan descartadas y por que.
5. Si el tema es objeto de debate o matiz en la guia, senalalo explicitamente.
6. Eres apoyo a la decision; la decision final es del medico. No prescribes.
7. Si el medico aporta datos de un paciente sin formular una pregunta explicita, asume que busca \
una evaluacion clinica de esos datos. Analiza el caso y ofrece las recomendaciones pertinentes \
segun las guias, sin quejarte de que falta una pregunta.
8. Cada fragmento indica la VIGENCIA de su guia. Si fragmentos de guias con vigencias distintas \
difieren sobre el mismo punto, prioriza el mas reciente y señala explicitamente que el otro puede \
estar obsoleto (distingue obsolescencia de contradiccion real).
9. Respeta la POLARIDAD de las [Recomendaciones normativas] etiquetadas: una recomendacion marcada \
como no_recomendado o generalmente_no_recomendado NUNCA puede presentarse como recomendada, ni al reves."""


def _contexto(hits: list[Hit], max_frag: int = 8) -> str:
    lineas = []
    for i, h in enumerate(hits[:max_frag], start=1):
        vig = f", vigencia={h.fecha_vigencia}" if getattr(h, "fecha_vigencia", "") else ""
        cab = f"[{i}] (guia={h.guia}, seccion={h.seccion[:60]}, pag={h.pagina}{vig})"

        texto_aserciones = ""
        if h.aserciones:
            texto_aserciones = "\n[Aserciones Clínicas Estructuradas]:\n"
            for a in h.aserciones:
                texto_aserciones += f" - {a.get('intervencion')} -> {a.get('relacion_base')} -> {a.get('resultado_esperado')} (Evidencia: {a.get('evidence_level')}). {a.get('descripcion_relacion')}\n"

        texto_recs = ""
        if getattr(h, "recomendaciones", None):
            texto_recs = "\n[Recomendaciones normativas]:\n"
            for r in h.recomendaciones:
                grado = f", grado={r['grado']}" if r.get("grado") else ""
                texto_recs += f" - (accion={r['accion_deontica']}{grado}) \"{r['texto']}\"\n"

        lineas.append(f"{cab}\n[Texto Original]:\n{h.texto}{texto_aserciones}{texto_recs}")
    return "\n\n".join(lineas)


def generar_respuesta(pregunta: str, hits: list[Hit],
                      llm: LLM | None = None, preferencias: str = "",
                      contexto_extra: str = "") -> dict:
    """Devuelve {respuesta, fragmentos_usados, abstenida(bool aproximado)}.

    contexto_extra: bloque adicional verificable (p. ej. el analisis exhaustivo
    de clase del agente ECL) que se inserta tras los fragmentos."""
    llm = llm or LLM(MODELO_SINTESIS)
    if not hits and not contexto_extra:
        return {"respuesta": "Las guias proporcionadas no recogen esta cuestion.",
                "fragmentos_usados": [], "abstenida": True}

    contexto = _contexto(hits)
    bloque_extra = f"\n\n{contexto_extra}" if contexto_extra else ""
    extra = f"\n\n{preferencias}" if preferencias else ""
    user = (f"Pregunta del medico:\n{pregunta}\n\n"
            f"Fragmentos de las guias:\n{contexto}{bloque_extra}{extra}\n\n"
            f"Responde siguiendo las reglas, con citas [n].")

    if llm.simulado:
        respuesta = (f"[SIMULADO] Segun la guia, respuesta a: {pregunta} "
                     f"[1] (guia={hits[0].guia}, seccion={hits[0].seccion[:40]}, pag={hits[0].pagina}).")
    else:
        respuesta = llm.texto(SYSTEM, user, max_tokens=4000)

    abstenida = "no recogen" in respuesta.lower() or "no cubre" in respuesta.lower()
    return {"respuesta": respuesta,
            "fragmentos_usados": [h.chunk_id for h in hits[:8]],
            "abstenida": abstenida}
