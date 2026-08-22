"""Panel de verificacion con LENTES (Sprint 2): chequeos clinicos deterministas
que complementan al juez LLM de fidelidad.

  - Lente DEONTICA: ¿la respuesta presenta como recomendado un farmaco que las
    recomendaciones citadas marcan no_recomendado / generalmente_no_recomendado?
    (la inversion de polaridad es el fallo clinicamente letal)
  - Lente NUMERICA: ¿todo umbral numerico de la respuesta (FG/CD4/carga viral/
    dosis) aparece en algun fragmento citado?

Ambas son cribas de alta precision: solo señalan lo que pueden afirmar; el
matiz fino sigue siendo del verificador LLM. Sus hallazgos se anexan a
`afirmaciones_sin_respaldo`, con lo que alimentan la recuperacion dirigida.
"""
from __future__ import annotations

import re

from ..ingest import metadata as md
from ..retrieval.base import Hit

# Marco recomendador sin negacion cercana (ventana de la misma frase).
_RX_RECOMIENDA = re.compile(
    r"\b(se\s+recomienda|recomendad[oa]s?|de\s+elecci[oó]n|preferente|"
    r"pauta\s+recomendada|debe\s+(?:iniciarse|usarse|emplearse))\b", re.I)
_RX_NEGACION = re.compile(r"\b(no|nunca|evitar|desaconsej|contraindicad)\w*", re.I)

_RX_NUMERO_UNIDAD = re.compile(
    r"\b(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:ml/min|mg|copias|c[eé]l|cel/μl|/mm3|mm3|%|semanas?|horas?|h\b)",
    re.I)


def _frases(texto: str) -> list[str]:
    return re.split(r"(?<=[.;:])\s+", texto)


def lente_deontica(respuesta: str, hits: list[Hit]) -> list[str]:
    """Inversiones de polaridad detectadas (lista de afirmaciones problema)."""
    hallazgos = []
    negativas = []
    for h in hits[:8]:
        for r in getattr(h, "recomendaciones", []) or []:
            if r.get("accion_deontica") in ("no_recomendado", "generalmente_no_recomendado"):
                farmacos = md.extraer_farmacos(r.get("texto", ""))
                if farmacos:
                    negativas.append((set(farmacos), r))
    if not negativas:
        return hallazgos
    for frase in _frases(respuesta):
        if not _RX_RECOMIENDA.search(frase) or _RX_NEGACION.search(frase):
            continue
        f_farmacos = set(md.extraer_farmacos(frase))
        for neg_farmacos, r in negativas:
            comunes = f_farmacos & neg_farmacos
            if comunes:
                hallazgos.append(
                    f"POLARIDAD: la respuesta presenta como recomendado {sorted(comunes)} "
                    f"pero una recomendacion citada lo marca {r['accion_deontica']} "
                    f"(guia {r.get('guia')}, pag {r.get('pagina')})")
    return hallazgos


def lente_numerica(respuesta: str, hits: list[Hit]) -> list[str]:
    """Umbrales numericos de la respuesta sin respaldo en los fragmentos citados."""
    corpus = " ".join((h.texto or "") for h in hits[:8]).lower().replace(",", ".")
    hallazgos = []
    for m in _RX_NUMERO_UNIDAD.finditer(respuesta):
        num = m.group(1).replace(",", ".")
        if num not in corpus:
            hallazgos.append(
                f"NUMERICO: el valor '{m.group(0).strip()}' de la respuesta no aparece "
                f"en los fragmentos citados")
    return hallazgos


def aplicar_lentes(respuesta: str, hits: list[Hit], veredicto: dict) -> dict:
    """Fusiona los hallazgos de las lentes con el veredicto del juez LLM."""
    hallazgos = lente_deontica(respuesta, hits) + lente_numerica(respuesta, hits)
    if hallazgos:
        veredicto = dict(veredicto)
        veredicto["fiel"] = False
        if veredicto.get("veredicto") == "fiel":
            veredicto["veredicto"] = "parcial"
        veredicto["afirmaciones_sin_respaldo"] = (
            list(veredicto.get("afirmaciones_sin_respaldo") or []) + hallazgos)
        veredicto["lentes"] = hallazgos
    return veredicto
