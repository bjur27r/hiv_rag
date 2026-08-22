"""Dependencias: sesion de BD y usuario autenticado."""
from __future__ import annotations

from fastapi import Cookie, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal, User
from .security import decodificar_token


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _token_de(cookie: str | None, authorization: str | None) -> str | None:
    if cookie:
        return cookie
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:]
    return None


def get_current_user(
    db: Session = Depends(get_db),
    vih_token: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> User:
    token = _token_de(vih_token, authorization)
    payload = decodificar_token(token) if token else None
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No autenticado")
    user = db.get(User, int(payload["sub"]))
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Usuario no encontrado")
    return user
