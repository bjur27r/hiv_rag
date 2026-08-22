"""Endpoint del selector de modelos (proveedores + modelos disponibles)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from .db import User
from .deps import get_current_user

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def listar_modelos(_: User = Depends(get_current_user)):
    from asistente_vih.llm.modelos import disponibles
    return {"proveedores": disponibles()}
