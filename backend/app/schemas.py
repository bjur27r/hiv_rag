"""Esquemas Pydantic de la API."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class Credenciales(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class CambioPassword(BaseModel):
    password_actual: str
    password_nueva: str = Field(min_length=6)


class UsuarioOut(BaseModel):
    id: int
    email: str


class ThreadOut(BaseModel):
    id: str
    title: str


class MensajeOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list | None = None
    model: str | None = None
    veredicto: str | None = None


class ThreadDetalle(BaseModel):
    id: str
    title: str
    messages: list[MensajeOut]


class Preguntar(BaseModel):
    pregunta: str
    provider: str | None = None     # "anthropic" | "openai" | "groq"
    modelo: str | None = None       # modelo de SINTESIS elegido por el usuario
    resume: str | None = None       # respuesta del medico para reanudar un HITL
    modo: str | None = None         # "directa" (grafo, por defecto) | "corazonamiento" (co-razonamiento)
