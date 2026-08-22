# -*- coding: utf-8 -*-
"""Test local del backend (SQLite + modo simulado, sin gasto de API)."""
import os, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Forzar simulado y SQLite ANTES de importar la app.
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["DATABASE_URL"] = "sqlite:///./backend_test.db"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

Path(ROOT / "backend_test.db").unlink(missing_ok=True)

import json
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as c:  # el context manager dispara startup -> init_db
    print("health:", c.get("/health").json())

    r = c.post("/auth/register", json={"email": "medico@hospital.es", "password": "secreto123"})
    print("register:", r.status_code, r.json()["user"])

    r = c.post("/auth/login", json={"email": "medico@hospital.es", "password": "secreto123"})
    print("login:", r.status_code, "cookie set:", "vih_token" in c.cookies)

    print("models:", [(p["provider"], p["disponible"]) for p in c.get("/models").json()["proveedores"]])

    th = c.post("/threads").json()
    print("thread creado:", th)

    with c.stream("POST", f"/threads/{th['id']}/ask",
                  json={"pregunta": "¿A partir de que FG puede usarse TAF?"}) as s:
        tipos = []
        for line in s.iter_lines():
            if line and line.startswith("data: "):
                tipos.append(json.loads(line[6:])["type"])
        print("eventos SSE:", tipos)

    det = c.get(f"/threads/{th['id']}").json()
    print("hilo titulo:", det["title"])
    print("mensajes:", [(m["role"], m["content"][:45]) for m in det["messages"]])

    r = c.post("/auth/change-password", json={"password_actual": "secreto123", "password_nueva": "nuevo123"})
    print("change-password:", r.status_code, r.json())
    print("\nOK backend test")
