"""Inferir-y-confirmar: propone actualizaciones del perfil, NUNCA las aplica.

Analiza los turnos episodicos del usuario y detecta patrones de FORMA
repetidos (p. ej. pedir respuestas mas breves varias veces). Devuelve
`Propuesta`s con su motivo y la evidencia (turn_ids). Aplicarlas requiere una
confirmacion explicita (store.confirmar_propuesta), igual que una buena higiene
de memoria: se infiere, se pregunta, y solo entonces se actua.

Las heuristicas son deliberadamente transparentes y conservadoras: ante la duda,
no proponen. Mejor no personalizar que personalizar mal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .episodic import Turn

UMBRAL_REPETICION = 3  # nº de senales para proponer un cambio


@dataclass
class Propuesta:
    clave: str
    valor: object
    motivo: str
    evidencia: list[int] = field(default_factory=list)  # turn_ids que la respaldan


# Senales lexicas (sobre turnos del medico). Cada detector es un (clave, valor, regex, motivo).
_DETECTORES = [
    ("verbosidad", "conciso",
     re.compile(r"\b(m[aá]s\s+(?:corto|breve|conciso)|res[uú]me|abrevia|demasiado\s+largo)\b", re.I),
     "el profesional pide respuestas mas breves de forma recurrente"),
    ("verbosidad", "detallado",
     re.compile(r"\b(m[aá]s\s+detalle|amplia|exp[ll]ica\s+mejor|m[aá]s\s+extenso)\b", re.I),
     "el profesional pide mas detalle de forma recurrente"),
    ("mostrar_grado_evidencia", True,
     re.compile(r"\b(grado|nivel)\s+de\s+evidencia\b", re.I),
     "el profesional pregunta con frecuencia por el grado de evidencia"),
    ("listar_opciones_descartadas", True,
     re.compile(r"\b(qu[eé]\s+se\s+descarta|opciones?\s+descartad|por\s+qu[eé]\s+no)\b", re.I),
     "el profesional pregunta con frecuencia por las opciones descartadas"),
]


def proponer(turnos: list[Turn], perfil_actual: dict[str, object] | None = None) -> list[Propuesta]:
    """Devuelve propuestas de preferencia no aplicadas. `perfil_actual`: clave->valor ya fijado."""
    perfil_actual = perfil_actual or {}
    propuestas: list[Propuesta] = []
    medico = [t for t in turnos if t.role == "medico"]

    for clave, valor, patron, motivo in _DETECTORES:
        evidencia = [t.turn_id for t in medico if patron.search(t.texto or "")]
        if len(evidencia) >= UMBRAL_REPETICION and perfil_actual.get(clave) != valor:
            propuestas.append(Propuesta(clave=clave, valor=valor, motivo=motivo,
                                        evidencia=evidencia))

    # Evita proponer 'conciso' y 'detallado' a la vez: gana el de mas evidencia.
    verbos = [p for p in propuestas if p.clave == "verbosidad"]
    if len(verbos) > 1:
        ganador = max(verbos, key=lambda p: len(p.evidencia))
        propuestas = [p for p in propuestas if p.clave != "verbosidad"] + [ganador]

    return propuestas
