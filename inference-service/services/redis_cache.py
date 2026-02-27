"""
redis_cache.py
--------------
Cache distribuido de embeddings usando Redis.

Fase 4: reemplaza el dict{} in-memory de embedding_service.py
con Redis para soportar múltiples workers y persistencia entre reinicios.

Por qué Redis para cache de embeddings:
  - Compartido entre múltiples workers Uvicorn (el dict{} es por-proceso)
  - Persiste entre reinicios del servicio (los embeddings no se pierden)
  - TTL configurable: los embeddings viejos se expiran automáticamente
  - Sub-ms de latencia: Redis opera en memoria, comparable al dict Python

Serialización:
  Los np.ndarray se serializan a bytes con numpy.tobytes() antes de guardar.
  Al leer, se reconstruyen con np.frombuffer().
  Esto es más eficiente que JSON (no hay conversión float → string → float).

Fallback:
  Si Redis no está disponible, redis_cache degrada gracefully a cache miss.
  El embedding se computa normalmente — no hay error, solo menos performance.
  Esto es el patrón "circuit breaker lite" para dependencias opcionales.

Configuración:
  REDIS_URL: URL de conexión (default: redis://redis:6379)
  REDIS_EMBEDDING_TTL: Segundos antes de expirar (default: 86400 = 24h)
  REDIS_KEY_PREFIX: Prefijo de keys para namespacing (default: "emb:")
"""

import os
import hashlib
import numpy as np
import redis
from utils.logger import get_logger

logger = get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REDIS_EMBEDDING_TTL = int(os.getenv("REDIS_EMBEDDING_TTL", "86400"))  # 24h
REDIS_KEY_PREFIX = "emb:"

# ---------------------------------------------------------------------------
# Cliente Redis — lazy-initialized, con fallback
# ---------------------------------------------------------------------------

_redis_client: redis.Redis | None = None
_redis_available: bool = False


def init_redis() -> None:
    """
    Inicializa la conexión a Redis. Llamado en el startup de FastAPI.
    Si Redis no está disponible, loguea warning y continúa sin cache.
    """
    global _redis_client, _redis_available

    try:
        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=False,  # necesitamos bytes para los vectores
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=False,
        )
        # Ping para verificar conectividad
        _redis_client.ping()
        _redis_available = True
        logger.info("redis_connected", extra={"url": REDIS_URL})
    except Exception as e:
        _redis_available = False
        logger.warning(
            "redis_unavailable_using_memory_cache",
            extra={"url": REDIS_URL, "error": str(e)},
        )


def _make_key(text: str) -> str:
    """
    Genera una key Redis para un texto.

    Usa SHA256 del texto para:
      1. Evitar keys muy largas (Redis tiene límite de 512MB pero es ineficiente)
      2. Normalizar el namespace
      3. Evitar colisiones accidentales con caracteres especiales

    Args:
        text: Texto original.

    Returns:
        Key Redis con formato "emb:<sha256[:16]>"
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{REDIS_KEY_PREFIX}{text_hash}"


def get(text: str) -> np.ndarray | None:
    """
    Busca el embedding de un texto en Redis.

    Args:
        text: Texto cuyo embedding se busca.

    Returns:
        np.ndarray si está en cache, None si no.
    """
    if not _redis_available or _redis_client is None:
        return None

    try:
        key = _make_key(text)
        data = _redis_client.get(key)
        if data is None:
            return None
        # Reconstruir ndarray desde bytes
        vector = np.frombuffer(data, dtype=np.float32)
        return vector
    except Exception as e:
        logger.warning("redis_get_failed", extra={"error": str(e)})
        return None


def set(text: str, vector: np.ndarray) -> None:
    """
    Guarda el embedding de un texto en Redis con TTL.

    Args:
        text: Texto original (usado para generar la key).
        vector: Embedding np.ndarray a guardar.
    """
    if not _redis_available or _redis_client is None:
        return

    try:
        key = _make_key(text)
        # Serializar ndarray a bytes (float32 para reducir tamaño vs float64)
        data = vector.astype(np.float32).tobytes()
        _redis_client.setex(key, REDIS_EMBEDDING_TTL, data)
    except Exception as e:
        logger.warning("redis_set_failed", extra={"error": str(e)})


def is_available() -> bool:
    """Retorna True si Redis está conectado y operacional."""
    return _redis_available


def flush_embeddings() -> int:
    """
    Elimina todos los embeddings cacheados en Redis.
    Útil para forzar recomputación tras cambiar el modelo.

    Returns:
        Número de keys eliminadas.
    """
    if not _redis_available or _redis_client is None:
        return 0

    try:
        keys = _redis_client.keys(f"{REDIS_KEY_PREFIX}*")
        if keys:
            deleted = _redis_client.delete(*keys)
            logger.info("redis_cache_flushed", extra={"deleted_keys": deleted})
            return deleted
        return 0
    except Exception as e:
        logger.warning("redis_flush_failed", extra={"error": str(e)})
        return 0
