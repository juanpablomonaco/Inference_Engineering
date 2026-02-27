"""
metrics_store.py
----------------
Store in-memory de métricas de observabilidad.

Actúa como un singleton global que acumula estadísticas durante
el ciclo de vida de la aplicación. Es consultado por GET /metrics.

En producción, este módulo sería reemplazado por prometheus-client
sin necesidad de cambiar los callers — solo la implementación interna.

Thread-safety: Para este proyecto educativo usamos operaciones atómicas
simples de Python (GIL protege int/float). En producción con múltiples
workers se usaría threading.Lock o métricas fuera del proceso.
"""

from dataclasses import dataclass


@dataclass
class MetricsStore:
    """
    Acumulador de métricas de la aplicación.

    Todos los campos son mutables y se actualizan durante el runtime.
    Los valores calculados (promedios, ratios) se computan en get_snapshot().
    """

    # Contadores de requests
    total_requests: int = 0
    total_embedding_calls: int = 0
    total_search_calls: int = 0

    # Acumuladores de latencia (en ms)
    embedding_time_sum_ms: float = 0.0
    search_time_sum_ms: float = 0.0

    # Cache stats
    cache_hits: int = 0
    cache_misses: int = 0

    def record_request(self) -> None:
        """Incrementa el contador total de requests."""
        self.total_requests += 1

    def record_embedding(self, elapsed_ms: float, *, cache_hit: bool) -> None:
        """
        Registra una operación de embedding.

        Args:
            elapsed_ms: Tiempo que tardó la operación (0.0 si fue cache hit).
            cache_hit: True si el embedding vino del cache.
        """
        self.total_embedding_calls += 1
        self.embedding_time_sum_ms += elapsed_ms

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_search(self, elapsed_ms: float) -> None:
        """
        Registra una operación de búsqueda semántica.

        Args:
            elapsed_ms: Tiempo total de la búsqueda.
        """
        self.total_search_calls += 1
        self.search_time_sum_ms += elapsed_ms

    def get_snapshot(self) -> dict:
        """
        Retorna un snapshot calculado de todas las métricas.

        Los promedios y ratios se calculan al momento de la consulta.
        Retorna 0.0 si no hay datos suficientes (evita división por cero).
        """
        avg_embedding_ms = (
            self.embedding_time_sum_ms / self.total_embedding_calls
            if self.total_embedding_calls > 0
            else 0.0
        )

        avg_search_ms = (
            self.search_time_sum_ms / self.total_search_calls
            if self.total_search_calls > 0
            else 0.0
        )

        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_ratio = (
            self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "total_embedding_calls": self.total_embedding_calls,
            "total_search_calls": self.total_search_calls,
            "avg_embedding_ms": round(avg_embedding_ms, 2),
            "avg_search_ms": round(avg_search_ms, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": round(cache_hit_ratio, 4),
        }


# ---------------------------------------------------------------------------
# Instancia global — usada por todos los servicios
# ---------------------------------------------------------------------------

# Esta instancia es importada directamente por los servicios:
#   from services.metrics_store import metrics
metrics = MetricsStore()
