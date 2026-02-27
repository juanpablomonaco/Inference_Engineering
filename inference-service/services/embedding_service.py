"""
embedding_service.py
--------------------
Orquesta la generación de embeddings con cache in-memory.

Responsabilidades:
  1. Mantener un cache dict[str, np.ndarray] para evitar recomputar
     embeddings de textos ya vistos.
  2. Ser el único punto de acceso al modelo — ningún otro módulo
     llama directamente a embedding_model.encode().
  3. Reportar métricas (cache hit/miss, latencia) al metrics_store.
  4. Precargarse con los embeddings del corpus en el startup.

Flujo por request:
  embed(text)
    → cache hit?  → retorna vector sin inferencia  (O(1))
    → cache miss? → modelo.encode() → guarda en cache → retorna vector
"""

import numpy as np
from utils.logger import get_logger
from utils.timer import Timer
from services.metrics_store import metrics
import models.embedding_model as embedding_model

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cache in-memory: text → embedding vector
# ---------------------------------------------------------------------------

_cache: dict[str, np.ndarray] = {}


def embed(text: str) -> tuple[np.ndarray, bool, float]:
    """
    Genera o recupera el embedding de un texto.

    Primero busca en el cache. Si no está, llama al modelo y
    guarda el resultado para futuras requests.

    Args:
        text: Texto a embeder.

    Returns:
        Tupla (vector, cache_hit, elapsed_ms):
          - vector: np.ndarray de shape (384,)
          - cache_hit: True si vino del cache
          - elapsed_ms: Tiempo de la operación (0.0 si fue cache hit)
    """
    # ── Cache lookup ──────────────────────────────────────────────────────
    if text in _cache:
        logger.debug(
            "embedding_cache_hit",
            extra={"text_preview": text[:60]},
        )
        metrics.record_embedding(0.0, cache_hit=True)
        return _cache[text], True, 0.0

    # ── Cache miss: inferencia real ───────────────────────────────────────
    with Timer() as t:
        vector = embedding_model.encode(text)

    _cache[text] = vector

    logger.info(
        "embedding_computed",
        extra={
            "cache_hit": False,
            "duration_ms": round(t.elapsed_ms, 2),
            "text_preview": text[:60],
        },
    )
    metrics.record_embedding(t.elapsed_ms, cache_hit=False)

    return vector, False, t.elapsed_ms


def preload_texts(texts: list[str]) -> None:
    """
    Precarga embeddings en el cache para una lista de textos.

    Llamado durante el startup para asegurar que el corpus
    ya esté en cache antes de aceptar requests.

    Args:
        texts: Lista de textos a precomputar.
    """
    logger.info(
        "cache_preload_started",
        extra={"total_texts": len(texts)},
    )

    with Timer() as t:
        for text in texts:
            if text not in _cache:
                vector = embedding_model.encode(text)
                _cache[text] = vector

    logger.info(
        "cache_preload_completed",
        extra={
            "total_texts": len(texts),
            "cache_size": len(_cache),
            "duration_ms": round(t.elapsed_ms, 2),
        },
    )


def cache_size() -> int:
    """Retorna el número de embeddings actualmente en cache."""
    return len(_cache)


def is_ready() -> bool:
    """Retorna True si el cache fue inicializado (tiene al menos un item)."""
    return len(_cache) > 0
