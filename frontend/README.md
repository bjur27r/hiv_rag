# Frontend — Asistente Clínico VIH (React + Vite)

Contenedor independiente del backend. SPA en React/TypeScript; Nginx la sirve y
hace **proxy `/api` → backend** (mismo origen → sin CORS, cookie httpOnly funciona).

## Qué incluye
- **Login / registro** (auth sencilla), **logout** y **cambio de contraseña** (Ajustes).
- **Chat con streaming estilo agente de código**: muestra la **traza de pasos**
  (Clasificando → Recuperando guías → Verificando…) y luego la respuesta en markdown.
- **Hilos**: crear nuevos y **recuperar antiguos** (barra lateral).
- **Copiar** respuestas; **chips de cita → panel de trazabilidad** con el fragmento de guía
  (guía, sección, página).
- **Selector de modelos**: Claude / OpenAI / **Groq** (Qwen3‑32B, Llama 3…), desde `/models`.
- **HITL**: si el grafo pide confirmación (decisión seria / dato faltante), aparece un
  formulario para aportar el dato y **reanudar** la respuesta.
- Badge de **veredicto de fidelidad** y aviso legal.

## Desarrollo local
```bash
cd frontend
npm install
npm run dev        # http://localhost:3000 ; proxy /api -> http://localhost:8000 (backend local)
```

## Producción (Docker, contenedor propio)
Se levanta junto al backend desde la raíz:
```bash
docker compose up --build     # frontend en http://localhost:3000, backend en :8000
```
El `Dockerfile` compila con Node y sirve el build estático con Nginx; `nginx.conf`
enruta `/api` al servicio `backend` (con `proxy_buffering off` para el streaming SSE).
