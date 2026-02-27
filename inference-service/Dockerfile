# ---------------------------------------------------------------------------
# Dockerfile — Inference Service (Python/FastAPI)
# ---------------------------------------------------------------------------
#
# Estrategia de capas (orden importa para cache de Docker):
#   1. Base mínima           → cambia raramente
#   2. Dependencias sistema  → cambia raramente
#   3. requirements.txt      → cambia poco (solo al agregar deps)
#   4. pip install           → se cachea si requirements.txt no cambió
#   5. Código fuente         → cambia frecuentemente, va al final
#
# Multi-worker:
#   El número de workers se controla via la variable de entorno WORKERS.
#   Por defecto es 1. Para producción: WORKERS=4 (o 2*CPUs+1).
#
#   Con múltiples workers:
#   - Cada worker carga su propio modelo en memoria (~800MB por worker)
#   - El cache L1 (dict) es por-proceso → cada worker empieza vacío
#   - El cache L2 (Redis) es compartido → L1 se llena desde Redis en la
#     primera request por worker, sin necesidad de inferencia
#   - ChromaDB en modo persistent client es seguro para lectura concurrente
#
# Tamaño estimado de imagen: ~400MB (sin modelo)
# ---------------------------------------------------------------------------

FROM python:3.11-slim

LABEL maintainer="inference-engineering"
LABEL description="FastAPI inference service — multi-worker, RAG, Prometheus"
LABEL version="6.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    WORKERS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Shell form para poder leer la variable de entorno WORKERS en runtime.
# Con CMD en exec form ["uvicorn", ...] las env vars no se expanden.
#
# --workers $WORKERS          → número de procesos worker (default: 1)
# --timeout-keep-alive 65     → keepalive mayor que el timeout del LB
# --access-log                → log de cada request (en formato uvicorn)
CMD uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${WORKERS:-1} \
    --log-level warning \
    --timeout-keep-alive 65
