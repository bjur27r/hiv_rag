"""Extraccion ligera de entidades clinicas y metadatos de un fragmento.

Es deliberadamente conservadora y basada en lexico/regex: en Fase 1 sirve para
poblar los filtros (farmacos, umbrales, nivel de evidencia) y, mas adelante,
para sembrar los nodos del grafo de conocimiento. La normalizacion fina (mapear
sinonimos a un vocabulario canonico) se aborda en Fase 4.
"""
from __future__ import annotations

import re

# Vocabulario semilla de farmacos/clases ARV (ampliable). Se busca como token,
# por eso conviene mantener mayusculas tal y como aparecen en las guias.
FARMACOS = [
    "DTG", "BIC", "RAL", "EVG", "CAB", "dolutegravir", "bictegravir", "raltegravir",
    "cabotegravir", "elvitegravir",
    "TAF", "TDF", "FTC", "3TC", "ABC", "tenofovir", "emtricitabina", "lamivudina",
    "abacavir",
    "DRV", "ATV", "darunavir", "atazanavir", "RPV", "DOR", "EFV", "NVP",
    "rilpivirina", "doravirina", "efavirenz", "nevirapina",
    "rifampicina", "rifabutina", "cobicistat", "ritonavir",
]

CONDICIONES = [
    "insuficiencia renal", "tuberculosis", "embarazo", "hepatitis B", "VHB", "VHC",
    "fracaso virologico", "carga viral", "resistencia", "neurocognitivo", "HAND",
    "riesgo cardiovascular", "sindrome metabolico", "VIH-2", "infeccion aguda",
    "comorbilidad", "polifarmacia", "adherencia",
]

# Umbrales numericos: el motor los detecta como texto; el calculo lo hace luego
# una herramienta determinista (threshold_calc), nunca el LLM.
_UMBRAL_RE = re.compile(
    r"(?:FG|filtrado glomerular|aclaramiento|CD4|carga viral|copias|HLA-B\*?5701)"
    r"[^.\n]{0,40}?(?:[<>≥≤]=?\s*\d+|\d+\s*(?:mL/min|copias|c[eé]lulas))",
    re.IGNORECASE,
)

_EVIDENCIA_RE = re.compile(r"\b(A|B|C)-?(I|II|III)\b")  # GRADE estilo GeSIDA: A-I, B-II...


def extraer_farmacos(texto: str) -> list[str]:
    encontrados = {f for f in FARMACOS if re.search(rf"\b{re.escape(f)}\b", texto)}
    return sorted(encontrados)


def extraer_condiciones(texto: str) -> list[str]:
    low = texto.lower()
    return sorted({c for c in CONDICIONES if c.lower() in low})


def extraer_umbrales(texto: str) -> list[str]:
    return sorted({m.group(0).strip() for m in _UMBRAL_RE.finditer(texto)})


def extraer_nivel_evidencia(texto: str) -> str | None:
    m = _EVIDENCIA_RE.search(texto)
    return m.group(0) if m else None
