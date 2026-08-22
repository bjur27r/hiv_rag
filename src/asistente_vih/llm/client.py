"""Cliente LLM con proveedor INTERCAMBIABLE (OpenAI o Anthropic).

Una sola interfaz (LLM.texto / LLM.json) con dos backends detras. Se elige por
LLM_PROVIDER (openai|anthropic); por defecto OpenAI. Los modelos por rol
(router/sintesis/verificador) tienen defaults por proveedor y son sobreescribibles
por entorno (MODELO_ROUTER / MODELO_SINTESIS / MODELO_VERIFICADOR).

Claves leidas del entorno (OPENAI_API_KEY / ANTHROPIC_API_KEY); nunca hardcodeadas.
Sin clave -> modo SIMULADO (respuestas deterministas) para ejercitar el flujo.
"""
from __future__ import annotations

import json
import os
from typing import Callable

from .. import config  # noqa: F401  (carga .env al importar)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Defaults de modelo por proveedor y rol.
_DEFAULTS = {
    "openai": {"router": "gpt-4o-mini", "sintesis": "gpt-4o", "verificador": "gpt-4o"},
    "anthropic": {"router": "claude-haiku-4-5", "sintesis": "claude-sonnet-4-6",
                  "verificador": "claude-opus-4-8"},
    "groq": {"router": "llama-3.1-8b-instant", "sintesis": "llama-3.3-70b-versatile",
             "verificador": "llama-3.3-70b-versatile"},
}

# Groq expone una API compatible con OpenAI.
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _pertenece(modelo: str, provider: str) -> bool:
    """Heuristica: el ID de modelo corresponde al proveedor activo."""
    m = modelo.lower()
    if provider == "anthropic":
        return m.startswith("claude")
    if provider == "openai":
        return m.startswith(("gpt", "o1", "o3", "o4", "chatgpt"))
    if provider == "groq":
        return m.startswith(("llama", "qwen", "mixtral", "gemma", "deepseek")) or "/" in m
    return True


def _modelo(rol: str) -> str:
    """Override por entorno SOLO si corresponde al proveedor; si no, default del proveedor.

    Asi, un MODELO_* heredado de otro proveedor (p. ej. claude-* con LLM_PROVIDER=openai)
    no rompe las llamadas: se ignora y se usa el default de OpenAI.
    """
    ov = os.getenv(f"MODELO_{rol.upper()}")
    if ov and _pertenece(ov, LLM_PROVIDER):
        return ov
    return _DEFAULTS[LLM_PROVIDER][rol]


MODELO_ROUTER = _modelo("router")
MODELO_SINTESIS = _modelo("sintesis")
MODELO_VERIFICADOR = _modelo("verificador")

_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}


def _traceable(nombre: str):
    """No-op salvo que LangSmith este activo (LANGSMITH_TRACING=true)."""
    if os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes"):
        try:
            from langsmith import traceable
            return traceable(name=nombre)
        except ImportError:
            pass
    def _id(fn: Callable):
        return fn
    return _id


class LLM:
    def __init__(self, modelo: str = MODELO_SINTESIS, simulado: bool | None = None,
                 provider: str | None = None):
        self.provider = (provider or LLM_PROVIDER).lower()
        self.modelo = modelo
        falta_clave = not os.getenv(_KEY_ENV[self.provider])
        self.simulado = falta_clave if simulado is None else simulado
        self._cliente = None

    def _c(self):
        if self._cliente is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._cliente = OpenAI()
            elif self.provider == "groq":
                from openai import OpenAI  # API compatible con OpenAI
                self._cliente = OpenAI(base_url=_GROQ_BASE_URL, api_key=os.getenv("GROQ_API_KEY"))
            else:
                import anthropic
                self._cliente = anthropic.Anthropic()
        return self._cliente

    def _ant_extra(self, effort: str, schema: dict | None = None) -> dict:
        """Params opcionales de Anthropic segun capacidades del modelo.

        Haiku 4.5 NO soporta adaptive thinking ni effort (400), pero si structured
        outputs. Opus/Sonnet 4.6+/Fable soportan todo.
        """
        es_haiku = "haiku" in self.modelo.lower()
        extra: dict = {}
        oc: dict = {}
        if not es_haiku:
            extra["thinking"] = {"type": "adaptive"}
            oc["effort"] = effort
        if schema is not None:
            oc["format"] = {"type": "json_schema", "schema": schema}
        if oc:
            extra["output_config"] = oc
        return extra

    # ------------------------------------------------------------- texto
    @_traceable("llm.texto")
    def texto(self, system: str, user: str, max_tokens: int = 4000, effort: str = "high") -> str:
        if self.simulado:
            return f"[SIMULADO/{self.provider}:{self.modelo}] {user[:120]}"
        if self.provider in ("openai", "groq"):
            r = self._c().chat.completions.create(
                model=self.modelo, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return (r.choices[0].message.content or "").strip()
        # anthropic: streaming + get_final_message (robusto ante salidas largas)
        with self._c().messages.stream(
            model=self.modelo, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            **self._ant_extra(effort)) as stream:
            msg = stream.get_final_message()
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    # -------------------------------------------------------------- json
    def _json_raw(self, system: str, user: str, schema: dict, max_tokens: int, effort: str) -> str:
        if self.provider in ("openai", "groq"):
            r = self._c().chat.completions.create(
                model=self.modelo, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "salida", "strict": True, "schema": schema}})
            return r.choices[0].message.content or ""
        resp = self._c().messages.create(
            model=self.modelo, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            **self._ant_extra(effort, schema))
        return "".join(b.text for b in resp.content if b.type == "text")

    @_traceable("llm.json")
    def json(self, system: str, user: str, schema: dict, max_tokens: int = 2000,
             effort: str = "high", simulado_valor: dict | None = None) -> dict:
        if self.simulado:
            return simulado_valor if simulado_valor is not None else {}
        # Reintento una vez: structured outputs casi siempre dan JSON valido, pero un
        # truncado/refusal puntual produce texto no parseable; reintentar lo resuelve.
        ultimo = ""
        for intento in range(2):
            ultimo = self._json_raw(system, user, schema, max_tokens, effort)
            try:
                return json.loads(ultimo)
            except json.JSONDecodeError:
                continue
        raise ValueError(f"JSON invalido del modelo tras reintento: {ultimo[:160]!r}")
