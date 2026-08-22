"""Autenticacion: registro, login, logout, cambio de contrasena."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import User
from .deps import get_current_user, get_db
from .schemas import CambioPassword, Credenciales, UsuarioOut
from .security import crear_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(settings.COOKIE_NAME, token, httponly=True, samesite="lax",
                    secure=settings.COOKIE_SECURE, max_age=settings.JWT_EXPIRE_MIN * 60)


@router.post("/register")
def register(cred: Credenciales, resp: Response, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.email == cred.email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "El email ya esta registrado")
    user = User(email=cred.email, hashed_password=hash_password(cred.password))
    db.add(user); db.commit(); db.refresh(user)
    token = crear_token(user.id, user.email)
    _set_cookie(resp, token)
    return {"token": token, "user": {"id": user.id, "email": user.email}}


@router.post("/login")
def login(cred: Credenciales, resp: Response, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == cred.email))
    if not user or not verify_password(cred.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciales invalidas")
    token = crear_token(user.id, user.email)
    _set_cookie(resp, token)
    return {"token": token, "user": {"id": user.id, "email": user.email}}


@router.post("/logout")
def logout(resp: Response):
    resp.delete_cookie(settings.COOKIE_NAME)
    return {"ok": True}


@router.post("/change-password")
def change_password(body: CambioPassword, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    if not verify_password(body.password_actual, user.hashed_password):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "La contrasena actual no es correcta")
    user.hashed_password = hash_password(body.password_nueva)
    db.commit()
    return {"ok": True}


@router.get("/me", response_model=UsuarioOut)
def me(user: User = Depends(get_current_user)):
    return UsuarioOut(id=user.id, email=user.email)
