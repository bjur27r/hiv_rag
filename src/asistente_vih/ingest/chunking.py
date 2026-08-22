"""Chunking estructural: agrupa bloques por seccion y los corta por tamanio.

Estrategia:
  1. Detectar encabezados de seccion (numerados tipo "3.2 ..." o fuente mayor que
     el cuerpo) para mantener el anclaje seccion -> cita.
  2. Acumular texto dentro de una seccion hasta `objetivo` caracteres, sin partir
     a mitad de bloque (evita romper una tabla de dosis o un umbral).
  3. Solape (`solape`) entre fragmentos contiguos para no perder contexto en bordes.
"""
from __future__ import annotations

import re
import statistics

from ..config import GuiaMeta
from ..knowledge.schema import Chunk
from . import metadata
from .loaders import Bloque

_HEADING_NUM = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})[\.\)]?\s+\S")


def _es_encabezado(b: Bloque, size_cuerpo: float) -> bool:
    if _HEADING_NUM.match(b.texto) and len(b.texto) < 120:
        return True
    # Fuente notablemente mayor que el cuerpo y linea corta -> probable titulo.
    return b.size_max >= size_cuerpo * 1.15 and len(b.texto) < 120


def _es_tabla(texto: str) -> bool:
    # Heuristica simple: muchas cifras/separadores sugieren tabla de dosis/interacciones.
    digitos = sum(c.isdigit() for c in texto)
    return digitos >= 8 and ("|" in texto or texto.count("  ") >= 3 or digitos / max(len(texto), 1) > 0.12)


def fragmentar(
    bloques: list[Bloque],
    guia: GuiaMeta,
    objetivo: int = 1100,
    solape: int = 150,
) -> list[Chunk]:
    if not bloques:
        return []

    size_cuerpo = statistics.median(b.size_max for b in bloques) or 10.0
    chunks: list[Chunk] = []
    seccion_actual = "(inicio)"
    buffer = ""
    pagina_buffer = bloques[0].pagina
    idx = 0

    def _emit(texto: str, pagina: int, tipo: str) -> None:
        nonlocal idx
        texto = texto.strip()
        if not texto:
            return
        cid = f"{guia.codigo}::p{pagina}::{idx}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                guia=guia.codigo,
                titulo_guia=guia.titulo,
                version=guia.version,
                fecha_vigencia=guia.fecha_vigencia,
                seccion=seccion_actual,
                pagina=pagina,
                texto=texto,
                tipo=tipo,
                ambito=list(guia.ambito),
                farmacos=metadata.extraer_farmacos(texto),
                condiciones=metadata.extraer_condiciones(texto),
                umbrales=metadata.extraer_umbrales(texto),
                nivel_evidencia=metadata.extraer_nivel_evidencia(texto),
            )
        )
        idx += 1

    for b in bloques:
        if b.tipo == "tabla":  # tabla ya estructurada (find_tables) -> fragmento propio
            if buffer:
                _emit(buffer, pagina_buffer, "texto")
                buffer = ""
            _emit(b.texto, b.pagina, "tabla")
            pagina_buffer = b.pagina
            continue

        if _es_encabezado(b, size_cuerpo):
            if buffer:
                _emit(buffer, pagina_buffer, "texto")
                buffer = ""
            seccion_actual = b.texto[:120]
            pagina_buffer = b.pagina
            continue

        if _es_tabla(b.texto):
            # Las tablas se emiten como fragmento propio para no fragmentarlas.
            if buffer:
                _emit(buffer, pagina_buffer, "texto")
                buffer = ""
            _emit(b.texto, b.pagina, "tabla")
            pagina_buffer = b.pagina
            continue

        if not buffer:
            pagina_buffer = b.pagina
        buffer = f"{buffer} {b.texto}".strip()

        if len(buffer) >= objetivo:
            _emit(buffer, pagina_buffer, "texto")
            buffer = buffer[-solape:]  # arrastra solape al siguiente fragmento
            pagina_buffer = b.pagina

    if buffer:
        _emit(buffer, pagina_buffer, "texto")
    return chunks
