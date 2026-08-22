"""Limite de seudonimizacion: se aplica SIEMPRE antes de persistir un turno.

Detecta identificadores directos y los sustituye por marcadores neutros, y
estima un riesgo de reidentificacion por combinacion de cuasi-identificadores
(p. ej. localidad pequena + condicion poco frecuente + fecha).

AVISO: es un punto de partida conservador para identificadores directos. La
deteccion completa de nombres propios y la evaluacion formal de reidentificacion
requieren validacion del DPO y, probablemente, modelos especificos. No se debe
considerar suficiente para produccion sin esa validacion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- Identificadores directos (alta precision) ---
_PATRONES = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("TELEFONO", re.compile(r"\b(?:\+34[\s-]?)?(?:\d[\s-]?){9}\b")),
    ("DNI", re.compile(r"\b\d{8}[- ]?[A-Za-z]\b")),
    ("NIE", re.compile(r"\b[XYZxyz][- ]?\d{7}[- ]?[A-Za-z]\b")),
    ("NHC", re.compile(r"\b(?:NHC|n[ºo]?\.?\s*historia|historia\s*cl[ií]nica)\D{0,3}\d{4,}\b", re.IGNORECASE)),
    ("FECHA", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ("FECHA", re.compile(r"\b\d{1,2}\s+de\s+[a-záéíóú]+\s+de\s+\d{4}\b", re.IGNORECASE)),
    ("CENTRO", re.compile(r"\b(?:Hospital|Centro de Salud|Cl[ií]nica|Ambulatorio|Albergue|M[oó]dulo)\s+[A-ZÁÉÍÓÚ][\w.\-]*", re.IGNORECASE)),
]

# --- Senales de cuasi-identificadores (riesgo de combinacion) ---
_RIESGO_LOCALIDAD = re.compile(r"\b(?:pueblo|aldea|localidad|barrio)\b.{0,40}?\b\d{2,4}\s*habitantes\b", re.IGNORECASE)
_RIESGO_UNICO = re.compile(r"\b[uú]nic[oa]\s+(?:paciente|caso)\b", re.IGNORECASE)
_RIESGO_NOTORIEDAD = re.compile(r"\b(?:muy conocido|deportista profesional|accidente medi[aá]tico|liga\s+\w+)\b", re.IGNORECASE)
_RIESGO_ANIO = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class ResultadoPII:
    texto: str               # texto ya seudonimizado
    n_directos: int          # nº de identificadores directos sustituidos
    tipos: list[str]         # tipos detectados (EMAIL, DNI...)
    riesgo_reident: str      # "bajo" | "medio" | "alto"
    motivos_riesgo: list[str]


def _evaluar_riesgo(texto: str) -> tuple[str, list[str]]:
    motivos: list[str] = []
    if _RIESGO_LOCALIDAD.search(texto):
        motivos.append("localidad muy pequena")
    if _RIESGO_UNICO.search(texto):
        motivos.append("unicidad del paciente/caso")
    if _RIESGO_NOTORIEDAD.search(texto):
        motivos.append("notoriedad publica")
    n_anios = len(set(_RIESGO_ANIO.findall(texto)))
    if n_anios:
        motivos.append("fecha/anio concreto")
    # Combinacion de >=2 cuasi-identificadores -> riesgo alto.
    if len(motivos) >= 2:
        return "alto", motivos
    if motivos:
        return "medio", motivos
    return "bajo", motivos


def scrub(texto: str) -> ResultadoPII:
    """Seudonimiza identificadores directos y estima el riesgo de reidentificacion."""
    tipos: list[str] = []
    n = 0
    out = texto
    for etiqueta, patron in _PATRONES:
        def _sub(_m, _e=etiqueta):
            return f"[{_e}]"
        out, k = patron.subn(_sub, out)
        if k:
            tipos.append(etiqueta)
            n += k
    riesgo, motivos = _evaluar_riesgo(texto)
    return ResultadoPII(texto=out, n_directos=n, tipos=sorted(set(tipos)),
                        riesgo_reident=riesgo, motivos_riesgo=motivos)
