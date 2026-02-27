"""
health_service.py
-----------------
Gestiona el estado de readiness del sistema.

Expone flags booleanos que indican si cada componente crítico
fue inicializado correctamente durante el startup.

Por qué esto importa en producción:
  Los load balancers y orquestadores (ECS, Railway, Fly.io) llaman
  a GET /health antes de enrutar tráfico. Si retorna 503, el pod
  no recibe requests hasta que esté listo. Esto es el patrón
  "Readiness Probe" de Kubernetes, implementado aquí de forma simple.

Uso:
    from services.health_service import health_status
    health_status.set_model_loaded(True)
    health_status.is_ready()  # → True/False
"""

from dataclasses import dataclass


@dataclass
class HealthStatus:
    """
    Estado de inicialización de los componentes del sistema.

    Cada flag corresponde a un paso del startup de la aplicación.
    El sistema solo se considera 'ready' cuando todos son True.
    """

    model_loaded: bool = False
    corpus_initialized: bool = False
    cache_ready: bool = False
    redis_connected: bool = False   # Fase 4: opcional, no bloquea readiness
    ollama_ready: bool = False      # Fase 3: opcional, solo /rag lo requiere

    def set_model_loaded(self, value: bool) -> None:
        self.model_loaded = value

    def set_corpus_initialized(self, value: bool) -> None:
        self.corpus_initialized = value

    def set_cache_ready(self, value: bool) -> None:
        self.cache_ready = value

    def set_redis_connected(self, value: bool) -> None:
        """Redis es opcional — no bloquea el readiness del servicio core."""
        self.redis_connected = value

    def set_ollama_ready(self, value: bool) -> None:
        """Ollama es opcional — solo /rag lo requiere."""
        self.ollama_ready = value

    def is_ready(self) -> bool:
        """
        Core readiness: modelo + corpus + cache in-memory.
        Redis y Ollama son opcionales — su ausencia no bloquea /search.
        """
        return self.model_loaded and self.corpus_initialized and self.cache_ready

    def to_dict(self) -> dict:
        """Serializa el estado actual como diccionario."""
        return {
            "status": "ok" if self.is_ready() else "unavailable",
            "model_loaded": self.model_loaded,
            "corpus_initialized": self.corpus_initialized,
            "cache_ready": self.cache_ready,
            "redis_connected": self.redis_connected,
            "ollama_ready": self.ollama_ready,
        }


# ---------------------------------------------------------------------------
# Instancia global — actualizada durante el lifespan de FastAPI
# ---------------------------------------------------------------------------

health_status = HealthStatus()
