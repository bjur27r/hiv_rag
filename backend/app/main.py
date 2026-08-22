"""App FastAPI del asistente clinico VIH."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import init_db
from . import auth, models_api, threads, curation

app = FastAPI(title="Asistente Clinico VIH — API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {"ok": True}


app.include_router(auth.router)
app.include_router(threads.router)
app.include_router(models_api.router)
app.include_router(curation.router)
