"""Verificador de fidelidad (LLM-as-judge): ultima barrera anti-alucinacion.

Comprueba, afirmacion a afirmacion, que la respuesta esta respaldada por los
fragmentos. Usa structured outputs para un veredicto fiable. Si la respuesta no
es fiel, el orquestador la regenera o la convierte en abstencion.

Se usa un modelo potente (Opus por defecto) porque el coste de dejar pasar una
afirmacion sin respaldo es alto.
"""
from __future__ import annotations

from ..retrieval.base import Hit
from .client import LLM, MODELO_VERIFICADOR
from .generate import _contexto

SYSTEM = """Eres un verificador de fidelidad clinica. Recibes una pregunta, una respuesta \
generada y los fragmentos de guia en los que deberia apoyarse. Tu tarea: determinar si CADA \
afirmacion clinica de la respuesta esta respaldada por los fragmentos. Eres estricto: ante la \
duda, marca la afirmacion como no respaldada. No evaluas si la respuesta es buena, solo si esta \
ANCLADA en los fragmentos. Devuelve el veredicto en el formato pedido."""

SCHEMA = {
    "type": "object",
    "properties": {
        "veredicto": {"type": "string", "enum": ["fiel", "parcial", "no_fiel"]},
        "fiel": {"type": "boolean"},
        "afirmaciones_sin_respaldo": {"type": "array", "items": {"type": "string"}},
        "explicacion": {"type": "string"},
    },
    "required": ["veredicto", "fiel", "afirmaciones_sin_respaldo", "explicacion"],
    "additionalProperties": False,
}


def verificar_fidelidad(pregunta: str, respuesta: str, hits: list[Hit],
                        llm: LLM | None = None, contexto_extra: str = "") -> dict:
    """contexto_extra: evidencia adicional que la sintesis TAMBIEN vio (p. ej.
    el analisis exhaustivo del agente ECL); sin esto, el verificador marcaria
    como no respaldado justo el contenido derivado de esa evidencia."""
    llm = llm or LLM(MODELO_VERIFICADOR)
    bloque = f"\n\n{contexto_extra}" if contexto_extra else ""
    user = (f"Pregunta:\n{pregunta}\n\nRespuesta generada:\n{respuesta}\n\n"
            f"Fragmentos de guia:\n{_contexto(hits)}{bloque}\n\n"
            f"Verifica el respaldo de cada afirmacion.")
    # max_tokens holgado: con adaptive thinking (Opus) el razonamiento consume
    # presupuesto antes del JSON; si es bajo, el JSON se trunca y no parsea.
    return llm.json(
        SYSTEM, user, SCHEMA, max_tokens=6000,
        simulado_valor={"veredicto": "fiel", "fiel": True,
                        "afirmaciones_sin_respaldo": [],
                        "explicacion": "[SIMULADO] verificacion no ejecutada (sin clave)."},
    )
