"""Configuracion central: rutas y registro de guias GeSIDA.

El REGISTRO DE GUIAS es la fuente de verdad de los metadatos de version/vigencia.
Es una propuesta de partida derivada del nombre de archivo y DEBE ser revisada por
el equipo clinico (la fecha de vigencia es un guardarrail de seguridad: una guia
caducada puede producir una recomendacion peligrosa).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Rutas del proyecto -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]


def _cargar_dotenv() -> None:
    """Carga ROOT/.env en os.environ (sin dependencias). No pisa lo ya definido."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for linea in env.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave, valor = clave.strip(), valor.strip().strip('"').strip("'")
        if clave and clave not in os.environ:
            os.environ[clave] = valor


_cargar_dotenv()
DATA_DIR = ROOT / "data"
TESTS_DIR = ROOT / "tests"
ARTIFACTS_DIR = ROOT / "artifacts"
MEMORY_DIR = ARTIFACTS_DIR / "memory"  # capa episodica + perfiles de usuario

# Retencion por defecto de los turnos episodicos (dato de salud regulado).
# Es un valor de partida; lo fija la politica de proteccion de datos / DPO.
RETENCION_EPISODICA_DIAS = 365

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
EVAL_DATASET_PATH = ARTIFACTS_DIR / "eval_dataset.jsonl"
QUESTION_BANK_XLSX = TESTS_DIR / "Banco_preguntas_RAG_GeSIDA.xlsx"


@dataclass(frozen=True)
class GuiaMeta:
    """Metadatos de una guia. `ambito` permite el filtrado por poblacion/tema."""
    codigo: str          # identificador corto y estable usado en metadatos y grafo
    titulo: str          # nombre legible
    version: str         # version o anio de la guia
    fecha_vigencia: str  # YYYY-MM-DD aproximada del documento vigente (a validar)
    ambito: tuple[str, ...]  # etiquetas de poblacion/tema (filtro de metadatos, Nivel 2)
    archivo: str         # nombre de archivo en data/


# Registro: archivo PDF -> metadatos. Alineado con la columna "Guias implicadas"
# del banco de preguntas (p. ej. "TAR adultos (2022)").
GUIAS: dict[str, GuiaMeta] = {
    "TAR_ADULTOS_2022": GuiaMeta(
        codigo="TAR_ADULTOS_2022",
        titulo="Tratamiento antirretroviral en adultos (TAR)",
        version="2022",
        fecha_vigencia="2022-01-01",
        ambito=("adultos", "tar", "inicio", "simplificacion", "fracaso"),
        archivo="GuiaGeSIDAPlanNacionalSobreElSidaRespectoAlTratamientoAntirretroviralEnAdultosInfectadosPorElVirusDeLaInmunodeficienciaHumanaActualizacionEnero2022.pdf",
    ),
    "ADHERENCIA_2020": GuiaMeta(
        codigo="ADHERENCIA_2020",
        titulo="Adherencia al tratamiento antirretroviral",
        version="2020",
        fecha_vigencia="2020-02-01",
        ambito=("adherencia", "adultos"),
        archivo="GUIA_GESIDA_febrero_2020_Adherencia.pdf",
    ),
    "PEP": GuiaMeta(
        codigo="PEP",
        titulo="Profilaxis postexposicion ocupacional y no ocupacional (VIH/VHB/VHC)",
        version="consenso",
        fecha_vigencia="2015-01-01",
        ambito=("pep", "profilaxis", "exposicion", "adultos", "ninos"),
        archivo="GESIDA-Documento-de-Consenso-sobre-la-Profilaxis-Posexposicion-Ocupacional-y-No-Ocupacional-al-VIH-VHB-y-VHC-en-Adultos-y-Ninos.pdf",
    ),
    "TB_VIH": GuiaMeta(
        codigo="TB_VIH",
        titulo="Tuberculosis en pacientes con VIH",
        version="consenso",
        fecha_vigencia="2015-01-01",
        ambito=("tuberculosis", "coinfeccion", "adultos"),
        archivo="gesida_TB_en_VIH.pdf",
    ),
    "EMBARAZO": GuiaMeta(
        codigo="EMBARAZO",
        titulo="VIH y embarazo",
        version="consenso",
        fecha_vigencia="2018-01-01",
        ambito=("embarazo", "mujer", "transmision-vertical"),
        archivo="gesida_VIH_embarazo.pdf",
    ),
    "NEUROCOGNITIVO_2013": GuiaMeta(
        codigo="NEUROCOGNITIVO_2013",
        titulo="Manejo clinico de las alteraciones neurocognitivas (HAND)",
        version="2013",
        fecha_vigencia="2013-01-01",
        ambito=("neurocognitivo", "hand", "snc", "adultos"),
        archivo="gesidadcyrc2013-ManejoclinicodelasalteracionesNC.pdf",
    ),
    "RIESGO_CV_METAB": GuiaMeta(
        codigo="RIESGO_CV_METAB",
        titulo="Alteraciones metabolicas y riesgo cardiovascular en personas con VIH",
        version="1.1",
        fecha_vigencia="2019-01-01",
        ambito=("cardiovascular", "metabolico", "comorbilidad", "adultos"),
        archivo="Documento-de-consenso-en-relacion-con-las-alteraciones-metabolicas-y-riesgo-cardiovascular-en-las-personas-con-VIH_V1-1.pdf",
    ),
    "PREVENTIVA_VACUNAS": GuiaMeta(
        codigo="PREVENTIVA_VACUNAS",
        titulo="Medicina preventiva, vacunas y salud publica en VIH",
        version="consenso",
        fecha_vigencia="2018-01-01",
        ambito=("vacunas", "prevencion", "salud-publica", "adultos"),
        archivo="GESIDA-Documento-de-Consenso-de-GeSIDA-y-la-Sociedad-Espanola-de-Medicina-Preventiva-Salud-Publica-y-Gestion-Sanitaria.pdf",
    ),
}


def guia_por_archivo(nombre: str) -> GuiaMeta | None:
    """Devuelve los metadatos de la guia a partir del nombre de archivo PDF."""
    for g in GUIAS.values():
        if g.archivo == nombre:
            return g
    return None
