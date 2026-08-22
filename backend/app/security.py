"""Hash de contrasenas (argon2) y tokens JWT."""
from __future__ import annotations

import datetime as dt

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings

_pwd = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(p: str) -> str:
    return _pwd.hash(p)


def verify_password(p: str, hashed: str) -> bool:
    return _pwd.verify(p, hashed)


def crear_token(user_id: int, email: str) -> str:
    ahora = dt.datetime.now(dt.timezone.utc)
    payload = {"sub": str(user_id), "email": email,
               "exp": ahora + dt.timedelta(minutes=settings.JWT_EXPIRE_MIN), "iat": ahora}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


def decodificar_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
    except JWTError:
        return None
