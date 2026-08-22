"""Registro de proveedores/modelos para el selector del frontend.

Claude y OpenAI como principales; Groq (Qwen3-32B, Llama 3...) como alternativas
SELECCIONABLES por el usuario (no sustitutos). `disponibles()` marca cuales tienen
clave configurada en el entorno.
"""
from __future__ import annotations

import os

PROVEEDORES = {
    "anthropic": {
        "label": "Claude (Anthropic)",
        "key_env": "ANTHROPIC_API_KEY",
        "modelos": [
            {"id": "claude-opus-4-8", "label": "Claude Opus 4.8"},
            {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6"},
            {"id": "claude-haiku-4-5", "label": "Claude Haiku 4.5"},
        ],
    },
    "openai": {
        "label": "OpenAI",
        "key_env": "OPENAI_API_KEY",
        "modelos": [
            {"id": "gpt-4o", "label": "GPT-4o"},
            {"id": "gpt-4o-mini", "label": "GPT-4o mini"},
        ],
    },
    "groq": {
        "label": "Groq (alternativas)",
        "key_env": "GROQ_API_KEY",
        "modelos": [
            {"id": "qwen/qwen3-32b", "label": "Qwen3 32B"},
            {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B"},
            {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B"},
        ],
    },
}


def disponibles() -> list[dict]:
    """Lista de proveedores con sus modelos, marcando si la clave esta configurada.

    Nota: los IDs de Groq deben verificarse contra la documentacion vigente de Groq
    (cambian con frecuencia); estan centralizados aqui para editarlos en un solo sitio.
    """
    out = []
    for prov, info in PROVEEDORES.items():
        out.append({
            "provider": prov,
            "label": info["label"],
            "disponible": bool(os.getenv(info["key_env"])),
            "modelos": info["modelos"],
        })
    return out
