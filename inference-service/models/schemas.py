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
    dimension: int = Field(
        description="Número de dimensiones del vector."
    )
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


class SearchResponse(BaseModel):
    """Respuesta del endpoint /search."""

    result: str = Field(
        description="Texto del corpus más similar a la query."
    )
    score: float = Field(
        description="Score de similitud coseno entre query y resultado (0.0 a 1.0)."
    )
    elapsed_ms: float = Field(
        description="Tiempo total de procesamiento en milisegundos."
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Estado del sistema. Retorna 503 si algún flag es False."""

    status: str = Field(
        description="'ok' si el sistema está listo, 'unavailable' si no."
    )
    model_loaded: bool = Field(
        description="True si el modelo de embeddings fue cargado correctamente."
    )
    corpus_initialized: bool = Field(
        description="True si los embeddings del corpus fueron precomputados."
    )
    cache_ready: bool = Field(
        description="True si el cache de embeddings fue inicializado."
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class MetricsResponse(BaseModel):
    """Snapshot de métricas de observabilidad en tiempo real."""

    total_requests: int = Field(description="Total de requests procesadas.")
    total_embedding_calls: int = Field(description="Total de llamadas al endpoint /embedding.")
    total_search_calls: int = Field(description="Total de llamadas al endpoint /search.")
    avg_embedding_ms: float = Field(description="Latencia promedio de generación de embedding (ms).")
    avg_search_ms: float = Field(description="Latencia promedio de búsqueda semántica (ms).")
    cache_hits: int = Field(description="Número de veces que el embedding fue encontrado en cache.")
    cache_misses: int = Field(description="Número de veces que el embedding fue computado desde cero.")
    cache_hit_ratio: float = Field(description="Ratio de efectividad del cache (0.0 a 1.0).")


# ---------------------------------------------------------------------------
# Modelo genérico para respuestas de error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Respuesta estándar para errores HTTP."""

    detail: str = Field(description="Descripción del error.")
    extra: dict[str, Any] | None = Field(
        default=None, description="Información adicional del error."
    )
