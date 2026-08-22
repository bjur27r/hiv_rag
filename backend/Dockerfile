# Backend del asistente clinico (contenedor independiente del frontend).
# Contexto de build = raiz del repo (para acceder a src/ y artifacts/).
FROM python:3.12-slim

WORKDIR /app

# Dependencias
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Motor RAG/agentico + backend
COPY src/ ./src/
COPY backend/app/ ./app/
# Indice denso + chunks (necesarios para la recuperacion en runtime).
COPY artifacts/chunks.jsonl ./artifacts/chunks.jsonl
COPY artifacts/embeddings/ ./artifacts/embeddings/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
