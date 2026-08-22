# Asistente Clínico Agéntico — Guías GeSIDA (VIH)

Asistente médico **riguroso, proactivo y agéntico** sobre las guías clínicas GeSIDA de VIH.
Razona la *estrategia* (qué recuperar, cómo combinar evidencia, qué calcular, cuándo
detenerse) pero **nunca el contenido clínico**: cada afirmación queda anclada a una cita
verificable y la respuesta se verifica antes de entregarse.

**Stack:** LangGraph (orquestación agéntica + HITL) · recuperación densa + descomposición ·
Claude / OpenAI / **Groq** (selector de modelos) · FastAPI · React · Postgres · Docker.

> Herramienta de **apoyo a la decisión** del médico especialista. No prescribe. Las
> respuestas citan guía, sección y página, y un verificador comprueba su fidelidad.

---

## Arquitectura (frontend y backend en contenedores separados)

```
┌──────────────┐   /api (proxy)   ┌───────────────────────────┐   ┌────────────┐
│  frontend    │ ───────────────▶ │  backend (FastAPI)        │──▶│  postgres  │
│ React + Nginx│ ◀── SSE stream── │  + motor RAG agéntico     │   │ usuarios/  │
│  :3000       │                  │  (LangGraph)  :8000       │   │ hilos      │
└──────────────┘                  └───────────────────────────┘   └────────────┘
```

El **motor** (`src/asistente_vih/`) hace: ingesta de guías → recuperación (densa +
descomposición multi‑guía) → router → síntesis citada → verificación de fidelidad →
HITL/abstención → auditoría, con memoria conversacional por hilo.

---

## Puesta en marcha desde cero

### 0. Requisitos
- Docker + Docker Compose.
- Para la ingesta/índice (paso 2): Python 3.10+ y las dependencias del motor.
- Claves de API (al menos una de LLM + una de embeddings):
  - LLM: `ANTHROPIC_API_KEY` y/o `OPENAI_API_KEY` y/o `GROQ_API_KEY`.
  - Embeddings (obligatorio para la recuperación densa): `OPENAI_API_KEY` (por defecto) o `VOYAGE_API_KEY`.

### 1. Configura las claves
```bash
cp .env.example .env      # edita .env y pon tus claves (NUNCA lo subas a git)
```

### 2. Genera la base de conocimiento (una vez, en el host)
```bash
pip install -r requirements.txt
export PYTHONPATH=src                       # PowerShell: $env:PYTHONPATH="src"

python -m asistente_vih.ingest.run_ingest   # data/*.pdf -> artifacts/chunks.jsonl (~2.700 chunks)
python -m asistente_vih.retrieval.embeddings # construye el índice denso (necesita OPENAI/VOYAGE key)
python -m asistente_vih.eval.build_dataset   # tests/*.xlsx -> artifacts/eval_dataset.jsonl (505 preguntas)
```
> El contenedor del backend copia `artifacts/chunks.jsonl` y `artifacts/embeddings/`, así que
> estos pasos deben ejecutarse **antes** de construir las imágenes.

### 3. Levanta el sistema
```bash
docker compose up --build
```
- App web: **http://localhost:3000**
- API + docs: **http://localhost:8000/docs**

Regístrate con un email/contraseña, abre una consulta y pregunta. Elige el modelo
(Claude / OpenAI / Groq) en el selector de la barra superior.

---

## Estructura del repositorio
```
src/asistente_vih/        # MOTOR (paquete Python)
  ingest/                 # carga PDF + chunking estructural + tablas + metadatos
  retrieval/              # bm25, denso (embeddings), híbrido (RRF), reranker, descomposición
  llm/                    # cliente multi-proveedor (Claude/OpenAI/Groq), generación, verificador
  memory/                 # memoria episódica + perfil + seudonimización (PII)
  agent/                  # grafo LangGraph (router→recuperar→sintetizar→verificar→HITL→audit)
  eval/                   # recall@k, calibración HITL, eval de respuesta, exports de validación
backend/                  # API FastAPI (contenedor propio) — auth, hilos, /ask SSE, /models
frontend/                 # SPA React + Vite + Nginx (contenedor propio)
docs/                     # documento de arquitectura (.docx)
data/                     # guías GeSIDA (PDF)  ·  tests/  banco de 505 preguntas (.xlsx)
docker-compose.yml        # postgres + backend + frontend
```

## Evaluación y utilidades (host, `PYTHONPATH=src`)
```bash
python -m asistente_vih.eval.recall --dense --hybrid     # recall@k de recuperación
python -m asistente_vih.eval.hitl_calibracion            # precisión/recall del disparador HITL
python -m asistente_vih.eval.answer_eval --por-nivel 3   # calidad de respuesta (muestra)
python -m asistente_vih.eval.full_eval                   # eval completa 505 (resumible) + xlsx revisión
python scripts/demo_graph.py --simulado                  # demo del grafo (sin claves)
```

## Modelos y seguridad
- **Selector de modelos**: Claude (Opus/Sonnet/Haiku), OpenAI (GPT‑4o/mini) y **Groq**
  (Qwen3‑32B, Llama 3) como alternativas seleccionables. Configurable por `.env`.
- **Privacidad**: la consulta pasa por un límite de **seudonimización** antes de procesarse;
  el registro de auditoría guarda siempre la versión seudonimizada.
- **Claves**: viven en `.env` (en `.gitignore`). Nunca se versionan ni se escriben en código.

## Notas para producción
- Sustituir el checkpointer en memoria por el **checkpointer Postgres** de LangGraph (HITL multi‑worker).
- Validación clínica del patrón oro (`artifacts/validacion_clinica.xlsx`) antes de usar la eval como gate.
- TLS, rotación de claves, política de retención de datos de salud (RGPD).
