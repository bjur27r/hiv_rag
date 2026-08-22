"""Descomposicion de consulta: ataca el problema multi-guia (N4).

Una pregunta como "rifampicina + INI en un paciente con TB y CD4 bajos" exige
evidencia de varias guias a la vez. Un unico vector de consulta promedia todos
los aspectos y no recupera bien TODAS las guias. Aqui se parte la pregunta en
sub-preguntas atomicas (una por aspecto/guia), se recupera cada una por separado
y se fusionan los resultados (RRF). Asi cada guia relevante tiene su propia
oportunidad de aparecer en el top-k.

La particion la hace un LLM barato (router/Haiku) con salida estructurada. En
modo simulado (sin clave) no parte: devuelve la pregunta original.
"""
from __future__ import annotations

from ..llm.client import LLM, MODELO_ROUTER
from .base import Hit, Retriever
from .hybrid import rrf

SYSTEM = """Eres un descompositor de consultas clinicas sobre VIH. Divide la pregunta del \
medico en sub-preguntas ATOMICAS, una por cada aspecto o guia implicada (p. ej. interaccion \
farmacologica, ajuste de dosis por funcion renal, momento de inicio del tratamiento, \
contraindicacion, vacunas...). Si la pregunta ya es simple y de un solo aspecto, devuelve una \
sola sub-pregunta (la original). Devuelve sub-preguntas autonomas y buscables, no mas de las \
necesarias."""

SCHEMA = {
    "type": "object",
    "properties": {"subpreguntas": {"type": "array", "items": {"type": "string"}}},
    "required": ["subpreguntas"],
    "additionalProperties": False,
}


class DecomposingRetriever(Retriever):
    def __init__(self, base: Retriever, llm: LLM | None = None, max_sub: int = 4):
        self.base = base
        self.llm = llm or LLM(MODELO_ROUTER)
        self.max_sub = max_sub

    def subpreguntas(self, query: str) -> list[str]:
        if self.llm.simulado:
            return [query]
        out = self.llm.json(SYSTEM, query, SCHEMA, max_tokens=600,
                            simulado_valor={"subpreguntas": [query]})
        subs = [s.strip() for s in out.get("subpreguntas", []) if s.strip()]
        return subs[:self.max_sub] or [query]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        subs = self.subpreguntas(query)
        if len(subs) <= 1:
            return self.base.search(query, k)
        n = max(k, 20)
        listas = [self.base.search(s, n) for s in subs]
        listas.append(self.base.search(query, n))  # tambien la original
        return rrf(listas, k=k)
