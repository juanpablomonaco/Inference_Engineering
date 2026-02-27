"""
main.py
-------
Entry point de la aplicación FastAPI.

Responsabilidades:
  1. Lifespan manager: orquesta el startup (carga modelo, precomputa corpus)
  2. Registra todos los routers (endpoints)
  3. Middleware de logging: mide latencia total de cada request
  4. Manejo global de errores

Principio: main.py NO contiene lógica de negocio.
Solo conecta las piezas. Toda la lógica vive en services/.

Startup order (bloqueante — FastAPI no acepta requests hasta completar):
  [1] load_model()         → embedding_model.py
  [2] precompute()         → search_service.py (también llena el cache)
  [3] set flags            → health_service.py
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse

import models.embedding_model as embedding_model
import services.search_service as search_service
import services.embedding_service as embedding_service
import services.vector_store as vector_store
from services.health_service import health_status
from services.metrics_store import metrics
from models.schemas import (
    EmbeddingRequest, EmbeddingResponse,
    SearchRequest, SearchResponse, SearchResult,
    IngestRequest, IngestResponse,
    HealthResponse, MetricsResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup y shutdown de la aplicación
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Context manager que maneja el ciclo de vida de la aplicación.

    Todo el código antes del `yield` se ejecuta al startup.
    Todo el código después del `yield` se ejecuta al shutdown.

    FastAPI no acepta requests hasta que el lifespan completa el startup.
    Esto garantiza que el modelo y el corpus estén listos antes de servir tráfico.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info("application_startup_started")

    startup_start = time.perf_counter()

    # [1] Cargar el modelo de embeddings
    try:
        embedding_model.load_model()
        health_status.set_model_loaded(True)
    except Exception as e:
        logger.error("model_loading_failed", extra={"error": str(e)})
        health_status.set_model_loaded(False)
        # En producción podrías hacer raise aquí para fallar el startup
        # y forzar un restart del contenedor.

    # [2] Inicializar ChromaDB vector store
    try:
        vector_store.init_store()
    except Exception as e:
        logger.error("vector_store_init_failed", extra={"error": str(e)})

    # [3] Indexar corpus seed en ChromaDB (idempotente via upsert)
    try:
        search_service.precompute()
        health_status.set_corpus_initialized(True)
        health_status.set_cache_ready(embedding_service.is_ready())
    except Exception as e:
        logger.error("corpus_precompute_failed", extra={"error": str(e)})
        health_status.set_corpus_initialized(False)
        health_status.set_cache_ready(False)

    startup_elapsed = (time.perf_counter() - startup_start) * 1000

    logger.info(
        "application_startup_completed",
        extra={
            "ready": health_status.is_ready(),
            "corpus_size": search_service.corpus_size(),
            "cache_size": embedding_service.cache_size(),
            "total_startup_ms": round(startup_elapsed, 2),
        },
    )

    yield  # ← La aplicación sirve requests a partir de aquí

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("application_shutdown")


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Inference Service",
    description=(
        "Servicio de inferencia educativo que demuestra embeddings, "
        "búsqueda semántica y observabilidad con FastAPI + SentenceTransformers."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware — latencia total por request
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_timing_middleware(request: Request, call_next) -> Response:
    """
    Mide la latencia total de cada request HTTP.

    Loguea path, método, status code y duración.
    Incrementa el contador global de requests.
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # No contar health checks en el total (evita ruido en métricas)
    if request.url.path not in ("/health", "/metrics"):
        metrics.record_request()

    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "total_ms": round(elapsed_ms, 2),
        },
    )

    return response


# ---------------------------------------------------------------------------
# POST /embedding
# ---------------------------------------------------------------------------

@app.post(
    "/embedding",
    response_model=EmbeddingResponse,
    summary="Genera el embedding de un texto",
    tags=["Inference"],
)
async def create_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Recibe un texto y retorna su vector de embedding (384 dimensiones).

    - Cache-first: si el texto ya fue procesado, retorna instantáneamente.
    - elapsed_ms refleja el tiempo real (0ms si fue cache hit).
    """
    if not embedding_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    vector, cache_hit, elapsed_ms = embedding_service.embed(request.text)

    return EmbeddingResponse(
        embedding=vector.tolist(),
        dimension=len(vector),
        cache_hit=cache_hit,
        elapsed_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Busca el texto más similar en el corpus",
    tags=["Inference"],
)
async def semantic_search(request: SearchRequest) -> SearchResponse:
    """
    Recibe una query y retorna los documentos más similares del corpus.

    - top_k: cuántos resultados retornar (default 1).
    - Solo computa el embedding de la query.
    - Los embeddings del corpus están en ChromaDB (HNSW index).
    """
    if not health_status.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Corpus may not be initialized.",
        )

    try:
        result = search_service.search(request.query, top_k=request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SearchResponse(
        result=result["result"],
        score=result["score"],
        results=[SearchResult(**r) for r in result["results"]],
        elapsed_ms=result["elapsed_ms"],
    )


# ---------------------------------------------------------------------------
# POST /ingest  (Fase 2)
# ---------------------------------------------------------------------------

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Indexa un documento en el vector store",
    tags=["Inference"],
    status_code=201,
)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Agrega o actualiza un documento en ChromaDB.

    - Genera el embedding del texto y lo almacena con upsert.
    - Idempotente: enviar el mismo id actualiza el documento existente.
    - El documento queda disponible para /search inmediatamente.
    """
    if not embedding_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        result = search_service.ingest(request.id, request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(**result)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado de salud del servicio",
    tags=["Observability"],
)
async def health_check() -> JSONResponse:
    """
    Retorna el estado de inicialización del sistema.

    - HTTP 200: todos los componentes están listos.
    - HTTP 503: algún componente falló en el startup.

    Usado por load balancers y orquestadores como Readiness Probe.
    """
    status_data = health_status.to_dict()
    http_status = 200 if health_status.is_ready() else 503

    return JSONResponse(content=status_data, status_code=http_status)


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Métricas de observabilidad en tiempo real",
    tags=["Observability"],
)
async def get_metrics() -> MetricsResponse:
    """
    Snapshot de métricas calculadas en tiempo real:
    - Requests totales
    - Latencias promedio (embedding, search)
    - Cache hit ratio

    En producción, este endpoint sería reemplazado por un scraper
    de Prometheus con métricas en formato /metrics de prometheus-client.
    """
    snapshot = metrics.get_snapshot()
    return MetricsResponse(**snapshot)
