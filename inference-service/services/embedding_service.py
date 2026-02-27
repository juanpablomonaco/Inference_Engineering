"""
embedding_service.py
--------------------
Orquesta la generación de embeddings con cache de dos niveles.

Fase 4: cache L1 (in-memory dict) + cache L2 (Redis distribuido).

Jerarquía de cache:
  L1 In-Memory dict  → O(1), sub-microsegundo, por-proceso, no persiste
  L2 Redis           → ~0.5ms, compartido entre workers, persiste 24h
  L3 Model inference → ~50ms, cómputo real con SentenceTransformer

Flujo:
  embed(text)
    → L1 hit?  → retorna vector                    (sub-μs)
    → L2 hit?  → carga en L1, retorna vector        (~0.5ms)
    → L1+L2 miss → modelo.encode() → guarda L1+L2  (~50ms)

Por qué dos niveles:
  - L1 elimina overhead de red para textos frecuentes en el mismo worker
  - L2 comparte cache entre múltiples workers y sobrevive reinicios
  - Si Redis cae, L1 sigue funcionando (graceful degradation)

Separación de responsabilidades:
  - embedding_service.py: lógica de cache, decide qué nivel usar
  - redis_cache.py: solo habla con Redis (sin conocimiento de embedding)
  - embedding_model.py: solo hace inferencia (sin conocimiento de cache)
"""

import numpy as np
from utils.logger import get_logger
from utils.timer import Timer
from services.metrics_store import metrics
import services.redis_cache as redis_cache
import models.embedding_model as embedding_model

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# L1 Cache in-memory: text → embedding vector (por-proceso)
# ---------------------------------------------------------------------------

_l1_cache: dict[str, np.ndarray] = {}


def embed(text: str) -> tuple[np.ndarray, bool, float]:
    """
    Genera o recupera el embedding de un texto usando cache de dos niveles.

    Args:
        text: Texto a embeder.

    Returns:
        Tupla (vector, cache_hit, elapsed_ms):
          - vector: np.ndarray de shape (384,)
          - cache_hit: True si vino de algún nivel de cache
          - elapsed_ms: Tiempo (0.0 si L1 hit, ~0.5ms si L2 hit, ~50ms si miss)
    """
    # ── L1: In-memory cache (sub-microsegundo) ────────────────────────────
    if text in _l1_cache:
        logger.debug("embedding_l1_cache_hit", extra={"text_preview": text[:60]})
        metrics.record_embedding(0.0, cache_hit=True)
        return _l1_cache[text], True, 0.0

    # ── L2: Redis cache (~0.5ms) ──────────────────────────────────────────
    with Timer() as redis_timer:
        redis_vector = redis_cache.get(text)

    if redis_vector is not None:
        # Promover a L1 para futuras requests del mismo worker
        _l1_cache[text] = redis_vector
        logger.debug(
            "embedding_l2_cache_hit",
            extra={"text_preview": text[:60], "redis_ms": round(redis_timer.elapsed_ms, 2)},
        )
        metrics.record_embedding(redis_timer.elapsed_ms, cache_hit=True)
        return redis_vector, True, redis_timer.elapsed_ms

    # ── L3: Model inference (~50ms) ───────────────────────────────────────
    with Timer() as t:
        vector = embedding_model.encode(text)

    # Guardar en ambos niveles
    _l1_cache[text] = vector
    redis_cache.set(text, vector)

    logger.info(
        "embedding_computed",
        extra={
            "cache_hit": False,
            "duration_ms": round(t.elapsed_ms, 2),
            "redis_available": redis_cache.is_available(),
            "text_preview": text[:60],
        },
    )
    metrics.record_embedding(t.elapsed_ms, cache_hit=False)

    return vector, False, t.elapsed_ms


def preload_texts(texts: list[str]) -> None:
    """
    Precarga embeddings en ambos niveles de cache para una lista de textos.

    Args:
        texts: Lista de textos a precomputar.
    """
    logger.info("cache_preload_started", extra={"total_texts": len(texts)})

    with Timer() as t:
        for text in texts:
            if text not in _l1_cache:
                embed(text)  # llena L1 + L2 automáticamente

    logger.info(
        "cache_preload_completed",
        extra={
            "total_texts": len(texts),
            "l1_size": len(_l1_cache),
            "redis_available": redis_cache.is_available(),
            "duration_ms": round(t.elapsed_ms, 2),
        },
    )


def cache_size() -> int:
    """Retorna el número de embeddings en el cache L1 (in-memory)."""
    return len(_l1_cache)


def is_ready() -> bool:
    """Retorna True si el cache L1 tiene al menos un item."""
    return len(_l1_cache) > 0
