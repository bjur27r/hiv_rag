"""Hilos de conversacion: CRUD + /ask con streaming SSE sobre el grafo."""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agent_runtime import stream_eventos, stream_eventos_corazonamiento
from .db import Message, SessionLocal, Thread, User
from .deps import get_current_user, get_db
from .schemas import MensajeOut, Preguntar, ThreadDetalle, ThreadOut

router = APIRouter(prefix="/threads", tags=["threads"])


def _propio(db: Session, thread_id: str, user: User) -> Thread:
    th = db.get(Thread, thread_id)
    if not th or th.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Hilo no encontrado")
    return th


@router.get("", response_model=list[ThreadOut])
def listar(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ths = db.scalars(select(Thread).where(Thread.user_id == user.id)
                     .order_by(Thread.created_at.desc())).all()
    return [ThreadOut(id=t.id, title=t.title) for t in ths]


@router.post("", response_model=ThreadOut)
def crear(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    th = Thread(id=uuid.uuid4().hex, user_id=user.id, title="Nueva consulta")
    db.add(th); db.commit()
    return ThreadOut(id=th.id, title=th.title)


@router.get("/{thread_id}", response_model=ThreadDetalle)
def detalle(thread_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    th = _propio(db, thread_id, user)
    msgs = [MensajeOut(id=m.id, role=m.role, content=m.content, citations=m.citations,
                       model=m.model, veredicto=m.veredicto) for m in th.messages]
    return ThreadDetalle(id=th.id, title=th.title, messages=msgs)


@router.patch("/{thread_id}", response_model=ThreadOut)
def renombrar(thread_id: str, title: str, db: Session = Depends(get_db),
              user: User = Depends(get_current_user)):
    th = _propio(db, thread_id, user)
    th.title = title[:200]; db.commit()
    return ThreadOut(id=th.id, title=th.title)


@router.delete("/{thread_id}")
def borrar(thread_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    th = _propio(db, thread_id, user)
    db.delete(th); db.commit()
    return {"ok": True}


@router.post("/{thread_id}/ask")
def preguntar(thread_id: str, body: Preguntar, db: Session = Depends(get_db),
              user: User = Depends(get_current_user)):
    th = _propio(db, thread_id, user)
    cfg = {"provider": body.provider, "sintesis": body.modelo} if body.provider else {}
    etiqueta_modelo = f"{body.provider}:{body.modelo}" if body.provider else None

    if body.resume is None:
        # Guarda el turno del medico y autotitula el hilo si es el primero.
        if not th.messages:
            th.title = body.pregunta[:60] + ("…" if len(body.pregunta) > 60 else "")
        db.add(Message(thread_id=thread_id, role="medico", content=body.pregunta))
        db.commit()

    def sse(ev: dict) -> str:
        return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    # Modo de reflexion: 'corazonamiento' (bucle co-razonamiento) solo si se pide
    # explicitamente; cualquier otro valor (incluido el 'directa' del frontend o
    # ausencia de modo) ejecuta el grafo LangGraph completo (ECL, CRAG, lentes).
    runtime = stream_eventos_corazonamiento if body.modo == "corazonamiento" else stream_eventos

    def generar():
        texto, citas, veredicto = "", [], None
        for ev in runtime(body.pregunta, str(user.id), thread_id, cfg, body.resume):
            if ev.get("type") == "respuesta":
                texto, citas, veredicto = ev["texto"], ev["citas"], ev["veredicto"]
            yield sse(ev)
        if texto:  # persiste la respuesta del asistente
            with SessionLocal() as s:
                s.add(Message(thread_id=thread_id, role="asistente", content=texto,
                              citations=citas, model=etiqueta_modelo, veredicto=veredicto))
                s.commit()

    return StreamingResponse(generar(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
