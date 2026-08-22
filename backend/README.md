# Backend — Asistente Clínico VIH (FastAPI)

Contenedor independiente del frontend. Envuelve el grafo agéntico (LangGraph) y
expone autenticación, hilos y respuesta con **streaming SSE** (traza agéntica +
respuesta citada). Selector de modelos **Claude / OpenAI / Groq**.

## Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| POST | `/auth/register` · `/auth/login` · `/auth/logout` · `/auth/change-password` | Auth (JWT en cookie httpOnly) |
| GET | `/auth/me` | Usuario actual |
| GET/POST | `/threads` | Listar / crear hilos |
| GET | `/threads/{id}` | Hilo + mensajes (recuperar antiguos) |
| PATCH/DELETE | `/threads/{id}` | Renombrar / borrar |
| POST | `/threads/{id}/ask` | Preguntar → **SSE**: eventos `paso` (traza) → `respuesta`/`hitl` → `fin` |
| GET | `/models` | Proveedores + modelos disponibles (selector) |

`/ask` acepta `{pregunta, provider, modelo, resume}`. `resume` reanuda una pausa HITL.

## Ejecución con Docker (frontend y backend en contenedores separados)
```bash
# desde la raíz del repo, con .env (OPENAI_API_KEY / ANTHROPIC_API_KEY / GROQ_API_KEY)
docker compose up --build        # levanta postgres + backend (frontend en Fase B)
# API en http://localhost:8000  ·  docs en /docs
```
El backend necesita el índice denso: el Dockerfile copia `artifacts/chunks.jsonl` y
`artifacts/embeddings/` (ejecuta antes `ingest` + `build_dense` si no existen).

## Desarrollo local (sin Docker, SQLite)
```bash
pip install -r backend/requirements.txt
export PYTHONPATH=src           # PowerShell: $env:PYTHONPATH="src"
uvicorn app.main:app --reload --app-dir backend
# requiere claves en el entorno para respuestas reales; sin ellas corre en modo simulado
```

## Notas
- Streaming: el grafo se emite por nodos (`paso`) y al final `respuesta` (con citas) o `hitl`.
- HITL: el checkpointer en memoria permite pausa/reanudación dentro del proceso; en
  producción multi-worker, usar el checkpointer **Postgres** de LangGraph.
- Persistencia: usuarios/hilos/mensajes en Postgres (SQLite en local).
