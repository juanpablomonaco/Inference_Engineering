"""
schemas.py
----------
Contratos de datos de la API usando Pydantic v2.

Cada endpoint tiene su propio par Request/Response.
Esto es schema-first design: el contrato se define antes de la lógica.

Beneficio: FastAPI valida automáticamente los tipos de entrada
y genera documentación OpenAPI sin configuración adicional.
"""

from pydantic import BaseModel, Field
from typing import Any


# ---------------------------------------------------------------------------
# POST /embedding
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    """Cuerpo de la request para generar un embedding."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Texto a convertir en embedding.",
        examples=["Machine learning is a subset of artificial intelligence."],
    )


class EmbeddingResponse(BaseModel):
    """Respuesta del endpoint /embedding."""

    embedding: list[float] = Field(
        description="Vector de embedding como lista de floats (384 dimensiones)."
    )
    dimension: int = Field(description="Número de dimensiones del vector.")
    cache_hit: bool = Field(
        description="True si el embedding fue recuperado del cache en memoria."
    )
    elapsed_ms: float = Field(
        description="Tiempo total de procesamiento en milisegundos."
    )


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Cuerpo de la request para búsqueda semántica."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Pregunta o texto a buscar en el corpus.",
        examples=["What is backpropagation?"],
    )
    top_k: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Número de resultados a retornar (top-k más similares).",
    )


class SearchResult(BaseModel):
    """Un resultado individual de búsqueda semántica."""

    id: str = Field(description="ID del documento.")
    text: str = Field(description="Contenido del documento.")
    score: float = Field(description="Score de similitud coseno (0.0 a 1.0).")


class SearchResponse(BaseModel):
    """Respuesta del endpoint /search."""

    result: str = Field(description="Texto del documento más similar (top-1).")
    score: float = Field(description="Score de similitud coseno del top-1 (0.0 a 1.0).")
    results: list[SearchResult] = Field(
        description="Lista completa de resultados top-k."
    )
    elapsed_ms: float = Field(
        description="Tiempo total de procesamiento en milisegundos."
    )


# ---------------------------------------------------------------------------
# POST /ingest  (Fase 2)
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Cuerpo de la request para ingestar un documento."""

    id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identificador único del documento.",
        examples=["doc_custom_001"],
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Contenido textual del documento a indexar.",
        examples=["Gradient descent is an optimization algorithm..."],
    )


class IngestResponse(BaseModel):
    """Respuesta del endpoint /ingest."""

    id: str = Field(description="ID del documento indexado.")
    total_documents: int = Field(
        description="Total de documentos en el store tras el upsert."
    )
    elapsed_ms: float = Field(description="Tiempo de procesamiento en milisegundos.")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Estado del sistema. Retorna 503 si algún flag core es False."""

    status: str = Field(description="'ok' si el sistema core está listo.")
    model_loaded: bool = Field(description="True si el modelo de embeddings cargó.")
    corpus_initialized: bool = Field(
        description="True si el corpus fue indexado en ChromaDB."
    )
    cache_ready: bool = Field(description="True si el cache L1 fue inicializado.")
    redis_connected: bool = Field(
        description="True si Redis está disponible (opcional)."
    )
    ollama_ready: bool = Field(
        description="True si Ollama está disponible para /rag (opcional)."
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class MetricsResponse(BaseModel):
    """Snapshot de métricas de observabilidad en tiempo real."""

    total_requests: int = Field(description="Total de requests procesadas.")
    total_embedding_calls: int = Field(
        description="Total de llamadas al endpoint /embedding."
    )
    total_search_calls: int = Field(
        description="Total de llamadas al endpoint /search."
    )
    avg_embedding_ms: float = Field(
        description="Latencia promedio de generación de embedding (ms)."
    )
    avg_search_ms: float = Field(
        description="Latencia promedio de búsqueda semántica (ms)."
    )
    cache_hits: int = Field(
        description="Número de veces que el embedding fue encontrado en cache."
    )
    cache_misses: int = Field(
        description="Número de veces que el embedding fue computado desde cero."
    )
    cache_hit_ratio: float = Field(
        description="Ratio de efectividad del cache (0.0 a 1.0)."
    )


# ---------------------------------------------------------------------------
# POST /rag  (Fase 3)
# ---------------------------------------------------------------------------


class RagRequest(BaseModel):
    """Cuerpo de la request para RAG."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Pregunta para el pipeline RAG.",
        examples=["How does backpropagation work in neural networks?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Número de documentos de contexto a recuperar.",
    )


class RagResponse(BaseModel):
    """Respuesta del endpoint /rag."""

    answer: str = Field(description="Respuesta generada por el LLM con contexto RAG.")
    sources: list[SearchResult] = Field(description="Documentos usados como contexto.")
    model: str = Field(description="Modelo LLM usado para la generación.")
    backend: str = Field(description="Backend LLM usado: 'ollama' o 'vllm'.")
    retrieve_ms: float = Field(description="Tiempo de retrieval semántico (ms).")
    generate_ms: float = Field(description="Tiempo de generación LLM (ms).")
    total_ms: float = Field(description="Tiempo total del pipeline RAG (ms).")


# ---------------------------------------------------------------------------
# Modelo genérico para respuestas de error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Respuesta estándar para errores HTTP."""

    detail: str = Field(description="Descripción del error.")
    extra: dict[str, Any] | None = Field(
        default=None, description="Información adicional del error."
    )
