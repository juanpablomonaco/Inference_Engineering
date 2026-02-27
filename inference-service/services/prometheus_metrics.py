"""
prometheus_metrics.py
---------------------
Métricas en formato Prometheus usando prometheus-client.

Fase 5: reemplaza el endpoint /metrics JSON con métricas reales
en formato Prometheus text exposition (scrapeables por Grafana/Prometheus).

Métricas expuestas:
  inference_requests_total         Counter  (por endpoint, método, status_code)
  inference_request_duration_seconds Histogram (latencia por endpoint)
  inference_embedding_duration_seconds Histogram (latencia de embedding)
  inference_search_duration_seconds   Histogram (latencia de search)
  inference_rag_duration_seconds      Histogram (latencia de RAG pipeline)
  inference_cache_hits_total          Counter  (por nivel: l1, l2)
  inference_cache_misses_total        Counter
  inference_corpus_size               Gauge    (documentos en ChromaDB)
  inference_model_loaded              Gauge    (0/1)

Por qué Histograms en lugar de Gauges para latencia:
  Los Histograms permiten calcular percentiles (P50, P95, P99) en Prometheus.
  Un Gauge solo da el valor actual — no el historial de distribución.
  P95 es la métrica más útil en producción: "el 95% de requests termina en X ms".

Buckets de latencia (segundos):
  Elegidos para cubrir el rango típico de un servicio de inferencia:
  0.01s (10ms) → cache hit
  0.1s  (100ms) → embedding
  0.5s  (500ms) → búsqueda con corpus grande
  2.0s  → generación LLM rápida
  10.0s → generación LLM lenta (CPU)

Endpoint:
  GET /metrics/prometheus → texto Prometheus (para scraping)
  GET /metrics           → JSON amigable (mantener para dev)
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Definición de métricas
# ---------------------------------------------------------------------------

# Namespace para todas las métricas del servicio
NS = "inference"

# Contador de requests por endpoint y status
REQUEST_COUNT = Counter(
    f"{NS}_requests_total",
    "Total number of HTTP requests",
    ["endpoint", "method", "status_code"],
)

# Histograma de latencia por endpoint (incluye P50, P95, P99)
REQUEST_LATENCY = Histogram(
    f"{NS}_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Histograma de latencia de embedding (solo cómputo, sin cache hits)
EMBEDDING_LATENCY = Histogram(
    f"{NS}_embedding_duration_seconds",
    "Embedding computation latency (cache misses only)",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5],
)

# Histograma de latencia de búsqueda semántica
SEARCH_LATENCY = Histogram(
    f"{NS}_search_duration_seconds",
    "Semantic search latency (retrieval only)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# Histograma de latencia del pipeline RAG completo
RAG_LATENCY = Histogram(
    f"{NS}_rag_duration_seconds",
    "Full RAG pipeline latency (retrieve + generate)",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# Contadores de cache
CACHE_HITS = Counter(
    f"{NS}_cache_hits_total",
    "Total embedding cache hits",
    ["level"],  # "l1" o "l2"
)

CACHE_MISSES = Counter(
    f"{NS}_cache_misses_total",
    "Total embedding cache misses (model inference required)",
)

# Gauges de estado del sistema
CORPUS_SIZE = Gauge(
    f"{NS}_corpus_size",
    "Number of documents in the vector store",
)

MODEL_LOADED = Gauge(
    f"{NS}_model_loaded",
    "1 if embedding model is loaded, 0 otherwise",
)

REDIS_CONNECTED = Gauge(
    f"{NS}_redis_connected",
    "1 if Redis is connected, 0 otherwise",
)

OLLAMA_READY = Gauge(
    f"{NS}_ollama_ready",
    "1 if Ollama LLM is available, 0 otherwise",
)


# ---------------------------------------------------------------------------
# Funciones helper para registrar métricas desde services/
# ---------------------------------------------------------------------------


def record_request(
    endpoint: str, method: str, status_code: int, duration_s: float
) -> None:
    """Registra una request HTTP completada."""
    REQUEST_COUNT.labels(
        endpoint=endpoint, method=method, status_code=str(status_code)
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_s)


def record_embedding(duration_s: float, cache_level: str | None) -> None:
    """
    Registra una operación de embedding.

    Args:
        duration_s: Duración en segundos.
        cache_level: "l1", "l2" si fue cache hit, None si fue miss.
    """
    if cache_level:
        CACHE_HITS.labels(level=cache_level).inc()
    else:
        CACHE_MISSES.inc()
        EMBEDDING_LATENCY.observe(duration_s)


def record_search(duration_s: float) -> None:
    """Registra una operación de búsqueda semántica."""
    SEARCH_LATENCY.observe(duration_s)


def record_rag(duration_s: float) -> None:
    """Registra una operación RAG completa."""
    RAG_LATENCY.observe(duration_s)


def update_system_gauges(
    corpus_size: int,
    model_loaded: bool,
    redis_connected: bool,
    ollama_ready: bool,
) -> None:
    """Actualiza los Gauges de estado del sistema."""
    CORPUS_SIZE.set(corpus_size)
    MODEL_LOADED.set(1.0 if model_loaded else 0.0)
    REDIS_CONNECTED.set(1.0 if redis_connected else 0.0)
    OLLAMA_READY.set(1.0 if ollama_ready else 0.0)


def get_prometheus_output() -> tuple[bytes, str]:
    """
    Genera el output en formato Prometheus text exposition.

    Returns:
        Tupla (bytes, content_type) lista para retornar como Response HTTP.
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
