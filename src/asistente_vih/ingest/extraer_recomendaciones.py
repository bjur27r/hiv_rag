"""Extraccion de RECOMENDACIONES como unidad de primera clase (Fase 1).

Recorre chunks.jsonl y extrae frases normativas: las que llevan grado de
evidencia (A-I ... C-III, con o sin guion: "AI", "B-II") y/o verbo deontico.
Cada recomendacion conserva su polaridad (accion deontica), el grado
normalizado, el literal y la trazabilidad completa (chunk, guia, seccion,
pagina, vigencia).

Taxonomia deontica: la del propio emisor (Tabla 4.2 de la guia PPE de GeSIDA):
  recomendado | considerar | generalmente_no_recomendado | no_recomendado
mas 'sin_determinar' para frases con grado pero sin verbo clasificable
(candidatas a slot-filling con LLM en una fase posterior).

Salida: artifacts/recomendaciones.jsonl

    python -m asistente_vih.ingest.extraer_recomendaciones
"""
from __future__ import annotations

import json
import re
from collections import Counter

from ..config import ARTIFACTS_DIR, CHUNKS_PATH, GUIAS

RECS_PATH = ARTIFACTS_DIR / "recomendaciones.jsonl"

# Grado de evidencia estilo GeSIDA (modificacion de criterios IDSA). Con o sin
# guion y con o sin parentesis: "(A-I)", "(AII)", "B-III". Exigir el contexto
# de parentesis o guion evita falsos positivos ("... a II" en prosa).
_RX_GRADO = re.compile(r"\(\s*([ABC])\s*[-–]?\s*(I{1,3})\s*\)|\b([ABC])\s*[-–]\s*(I{1,3})\b")

# Polaridad deontica. El ORDEN importa: lo negado se comprueba antes que lo
# afirmativo ("no debe" contiene "debe"; "debe evitarse" es negativo).
_DEONTICO = [
    ("generalmente_no_recomendado", re.compile(
        r"(?:generalmente|en general|de forma (?:general|rutinaria|sistematica))\s+no\s+se\s+recomienda"
        r"|no\s+se\s+recomienda\s+(?:de\s+forma\s+)?(?:general|rutinaria|sistematica)", re.I)),
    ("no_recomendado", re.compile(
        r"\bno\s+(?:se\s+)?(?:recomienda|recomiendan|aconseja|debe[n]?|deberia[n]?)\b"
        r"|\bdebe[n]?\s+evitarse\b|\bse\s+desaconseja\b|\bcontraindicad[oa]s?\b"
        r"|\bevitar(?:se)?\s+(?:el|la|los|las)\b|\bno\s+esta[n]?\s+recomendad[oa]s?\b", re.I)),
    ("considerar", re.compile(
        r"\b(?:puede[n]?\s+(?:considerarse|valorarse|utilizarse|emplearse|usarse)"
        r"|se\s+(?:puede[n]?\s+considerar|sugiere|podria)|considerar|valorar"
        r"|podria[n]?\s+(?:utilizarse|emplearse|usarse|considerarse)|es\s+razonable|opcional(?:mente)?)\b", re.I)),
    ("recomendado", re.compile(
        r"\b(?:se\s+recomienda[n]?|esta[n]?\s+recomendad[oa]s?|recomendamos"
        r"|se\s+debe[n]?|debe[n]?[ra]?[n]?\b|es\s+(?:necesario|preciso|obligad[oa])"
        r"|se\s+aconseja[n]?|hay\s+que)\b", re.I)),
]

# Corte de frases conservador (no parte siglas tipo "p. ej." perfecto, pero
# suficiente para frases normativas largas de guia).
_RX_FRASE = re.compile(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚÑ¿(])")


def _grado_normalizado(texto: str) -> tuple[str | None, str | None]:
    """Devuelve (grado 'A-I', literal encontrado) o (None, None)."""
    m = _RX_GRADO.search(texto)
    if not m:
        return None, None
    letra = m.group(1) or m.group(3)
    nivel = m.group(2) or m.group(4)
    return f"{letra.upper()}-{nivel.upper()}", m.group(0)


def _accion_deontica(texto: str) -> str:
    for accion, rx in _DEONTICO.items() if isinstance(_DEONTICO, dict) else _DEONTICO:
        if rx.search(texto):
            return accion
    return "sin_determinar"


def _frases(texto: str) -> list[str]:
    return [f.strip() for f in _RX_FRASE.split(texto) if len(f.strip()) >= 40]


def extraer() -> list[dict]:
    recs, vistas = [], set()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            guia = c["guia"]
            vigencia = c.get("fecha_vigencia") or (
                GUIAS[guia].fecha_vigencia if guia in GUIAS else "")
            n = 0
            for frase in _frases(c.get("texto", "")):
                grado, literal = _grado_normalizado(frase)
                accion = _accion_deontica(frase)
                # Es recomendacion si trae grado O verbo deontico claro.
                if grado is None and accion == "sin_determinar":
                    continue
                # Dedup por solape entre chunks contiguos (150 chars de solape).
                clave = (guia, re.sub(r"\s+", " ", frase.lower())[:120])
                if clave in vistas:
                    continue
                vistas.add(clave)
                recs.append({
                    "id": f"{c['chunk_id']}::rec{n}",
                    "chunk_id": c["chunk_id"],
                    "guia": guia,
                    "fecha_vigencia": vigencia,
                    "seccion": c.get("seccion", ""),
                    "pagina": c.get("pagina", 0),
                    "texto": frase,
                    "grado": grado,
                    "grado_literal": literal,
                    "accion_deontica": accion,
                })
                n += 1
    return recs


def main():
    recs = extraer()
    with open(RECS_PATH, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    por_guia = Counter(r["guia"] for r in recs)
    por_accion = Counter(r["accion_deontica"] for r in recs)
    con_grado = sum(1 for r in recs if r["grado"])
    print(f"Recomendaciones extraidas: {len(recs)}  (con grado de evidencia: {con_grado})")
    print("\nPor guia:")
    for g, n in por_guia.most_common():
        print(f"  {g:<22} {n}")
    print("\nPor accion deontica:")
    for a, n in por_accion.most_common():
        print(f"  {a:<30} {n}")
    print(f"\nGuardado en {RECS_PATH}")


if __name__ == "__main__":
    main()
