"""Analizador de texto en espanol para la recuperacion lexica.

Hace lo que en el documento llamamos "el analizador espanol bien puesto":
minusculas, plegado de acentos, eliminacion de palabras vacias y un stemming
ligero y conservador. Conservador a proposito: NO toca tokens cortos ni con
digitos (3TC, DTG, TAF, FG, CD4), porque ahi un stemming agresivo destruye la
senal lexica que mas importa en este dominio.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "y",
    "o", "a", "que", "con", "por", "para", "se", "su", "sus", "al", "es", "si",
    "como", "le", "lo", "mi", "tiene", "puede", "debe", "cual", "cuales", "tengo",
    "soy", "ante", "tras", "sobre", "entre", "cuando", "donde", "porque", "este",
    "esta", "estos", "estas", "ese", "esa", "hay", "ser", "son", "the", "of",
    "cuanto", "cuanta", "deben", "pueden", "qué", "cuál", "más", "muy", "ya",
}


def fold(texto: str) -> str:
    """Minusculas + sin acentos (pero conserva la enie como 'n' no, mejor 'ni')."""
    t = texto.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t


def _stem(tok: str) -> str:
    # Solo palabras 'largas' y alfabeticas: protege fármacos/abreviaturas/umbrales.
    if len(tok) <= 4 or any(c.isdigit() for c in tok):
        return tok
    for suf in ("ciones", "cion", "mente", "idades", "idad", "es", "as", "os", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def analizar(texto: str) -> list[str]:
    toks = _TOKEN.findall(fold(texto))
    return [_stem(t) for t in toks if t not in STOPWORDS and len(t) > 1]
