"""Interfaz comun de recuperacion. Mantiene el motor intercambiable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Hit:
    chunk_id: str
    guia: str
    seccion: str
    pagina: int
    score: float
    texto: str
    # Metadatos del chunk (utiles para el reordenador y los filtros):
    farmacos: list = field(default_factory=list)
    condiciones: list = field(default_factory=list)
    umbrales: list = field(default_factory=list)
    ambito: list = field(default_factory=list)
    aserciones: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)
    # Guardarrail de vigencia + recomendaciones normativas del chunk (Fase 1):
    fecha_vigencia: str = ""
    recomendaciones: list = field(default_factory=list)


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Devuelve los k fragmentos mas relevantes para la consulta."""
        raise NotImplementedError
