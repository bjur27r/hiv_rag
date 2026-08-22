"""Esquema de un fragmento (Chunk) indexable.

Los metadatos no son decorativos: son los que habilitan el filtrado por vigencia
(guardarrail de seguridad), el filtrado por poblacion (Nivel 2 del banco) y el
anclaje de la cita (seccion/pagina) que exige cada respuesta.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    # --- Identidad y trazabilidad ---
    chunk_id: str            # estable: {guia}::{pagina}::{indice}
    guia: str                # codigo de la guia (config.GUIAS)
    titulo_guia: str
    version: str
    fecha_vigencia: str      # YYYY-MM-DD; filtro de vigencia obligatorio
    seccion: str             # encabezado de seccion mas cercano (anclaje de cita)
    pagina: int              # 1-indexada (anclaje de cita)

    # --- Contenido ---
    texto: str
    tipo: str = "texto"      # "texto" | "tabla" | "lista"

    # --- Filtros y anclaje de entidades ---
    ambito: list[str] = field(default_factory=list)        # poblacion/tema (filtro Nivel 2)
    farmacos: list[str] = field(default_factory=list)      # entidades para grafo/lexico
    condiciones: list[str] = field(default_factory=list)
    umbrales: list[str] = field(default_factory=list)      # p. ej. "FG>=30", "CD4<200"
    nivel_evidencia: str | None = None                     # GRADE A/B/C si se detecta

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Chunk":
        return Chunk(**json.loads(line))
