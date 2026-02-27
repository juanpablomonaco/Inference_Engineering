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
# Modelo ML:
#   El modelo all-MiniLM-L6-v2 se descarga en runtime via HuggingFace Hub.
#   Se persiste en un volumen Docker para no re-descargarse entre reinicios.
#   Ver docker-compose.yml para la configuración del volumen.
#
# Tamaño estimado de imagen: ~400MB (sin el modelo)
# Con modelo cacheado en volumen: ~490MB total
# ---------------------------------------------------------------------------

# [1] Base — python:3.11-slim es significativamente más pequeña que python:3.11
#     No incluye compiladores ni herramientas de desarrollo innecesarias
FROM python:3.11-slim

# [2] Metadata de la imagen
LABEL maintainer="inference-engineering"
LABEL description="FastAPI inference service with semantic search"
LABEL version="2.0.0"

# [3] Variables de entorno
#
# PYTHONUNBUFFERED=1    → Los logs aparecen en tiempo real (sin buffering)
#                         Crítico para ver los logs de startup en Docker
#
# PYTHONDONTWRITEBYTECODE=1 → No genera archivos .pyc
#                             Reduce tamaño de imagen y capas
#
# HF_HOME               → Directorio donde HuggingFace cachea modelos
#                          Debe coincidir con el volumen en docker-compose
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

# [4] Directorio de trabajo dentro del contenedor
WORKDIR /app

# [5] Dependencias del sistema (mínimas)
#     build-essential es necesario para compilar algunas dependencias C de numpy/torch
#     curl es útil para health checks manuales desde dentro del contenedor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# [6] Copiar SOLO requirements.txt primero
#     Esto permite que Docker cachee la capa de pip install
#     Si el código cambia pero requirements.txt no, pip no se re-ejecuta
COPY requirements.txt .

# [7] Instalar dependencias Python
#     --no-cache-dir evita guardar el cache de pip en la imagen
#     --upgrade pip asegura que usamos la versión más reciente
RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt

# [8] Copiar el código fuente (cambia frecuentemente, al final)
#     Al estar después del pip install, los cambios de código
#     no invalidan la capa de dependencias
COPY . .

# [9] Exponer el puerto del servicio
EXPOSE 8000

# [10] Health check integrado en Docker
#      Docker marcará el contenedor como unhealthy si /health retorna != 200
#      Interval: espera 30s entre checks (el startup del modelo tarda ~5-10s)
#      Start-period: 60s de gracia para el startup inicial
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# [11] Comando de inicio
#      --host 0.0.0.0  → acepta conexiones externas al contenedor
#      --port 8000     → puerto que coincide con EXPOSE
#      --workers 1     → un solo worker (el modelo es un singleton por proceso)
#      --log-level warning → uvicorn usa nuestro logger JSON, no el propio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "warning"]
