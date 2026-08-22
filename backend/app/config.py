"""Configuracion del backend (variables de entorno)."""
from __future__ import annotations

import os


class Settings:
    # Postgres en docker-compose; SQLite por defecto para desarrollo local.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./backend.db")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-cambiar-en-produccion")
    JWT_ALG: str = "HS256"
    JWT_EXPIRE_MIN: int = int(os.getenv("JWT_EXPIRE_MIN", "720"))
    COOKIE_NAME: str = "vih_token"
    COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000").split(",")


settings = Settings()
