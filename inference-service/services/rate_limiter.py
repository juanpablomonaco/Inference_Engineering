"""
rate_limiter.py
---------------
Rate limiting por IP usando sliding window en Redis.

Algoritmo: Sliding Window Counter
  - Ventana de 60 segundos
  - Límite configurable por variable de entorno
  - Si Redis no está disponible: degrada gracefully (no limita)

Por qué Sliding Window vs Fixed Window:
  Fixed Window: fácil de implementar pero permite burst al borde de ventanas.
  Ejemplo: 100 req/min, con fixed window puedes hacer 100 en el segundo 59
           y 100 más en el segundo 61 → 200 requests en 2 segundos.
  Sliding Window: la ventana se mueve con cada request, previene el burst.

Implementación con Redis ZSET (sorted set):
  - Clave: "ratelimit:<ip>"
  - Cada request se agrega como score=timestamp en el ZSET
  - Al consultar, se eliminan los timestamps fuera de la ventana (ZREMRANGEBYSCORE)
  - ZCARD cuenta los requests dentro de la ventana
  - Pipeline atómico para evitar race conditions

Configuración:
  RATE_LIMIT_REQUESTS: máximo de requests por ventana (default: 60)
  RATE_LIMIT_WINDOW_SECONDS: tamaño de la ventana (default: 60)

Si Redis no está disponible: no se limita (fail-open).
  En producción crítica, fail-closed sería más seguro.
"""

import os
import time
from fastapi import Request, HTTPException, status
from utils.logger import get_logger
import services.redis_cache as redis_cache

logger = get_logger(__name__)

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_KEY_PREFIX = "ratelimit:"


def _get_client_ip(request: Request) -> str:
    """
    Extrae la IP del cliente de la request.

    Revisa X-Forwarded-For primero (para requests detrás de proxy/load balancer)
    antes de usar request.client.host.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency que verifica el rate limit para la IP del cliente.

    Si Redis no está disponible: no limita (fail-open).
    Si el límite se excede: retorna 429 Too Many Requests.

    Args:
        request: Request FastAPI (inyectada automáticamente).

    Raises:
        HTTPException 429: Cuando se excede el rate limit.
    """
    # Si Redis no está disponible, no aplicar rate limiting
    if not redis_cache.is_available() or redis_cache._redis_client is None:
        return

    client_ip = _get_client_ip(request)
    key = f"{RATE_LIMIT_KEY_PREFIX}{client_ip}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    try:
        pipe = redis_cache._redis_client.pipeline()
        # 1. Eliminar requests fuera de la ventana deslizante
        pipe.zremrangebyscore(key, 0, window_start)
        # 2. Contar requests en la ventana actual
        pipe.zcard(key)
        # 3. Agregar la request actual
        pipe.zadd(key, {str(now): now})
        # 4. Establecer TTL en la key para auto-cleanup
        pipe.expire(key, RATE_LIMIT_WINDOW * 2)
        results = pipe.execute()

        request_count = results[1]  # resultado del ZCARD

        if request_count >= RATE_LIMIT_REQUESTS:
            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "client_ip": client_ip,
                    "request_count": request_count,
                    "limit": RATE_LIMIT_REQUESTS,
                    "window_seconds": RATE_LIMIT_WINDOW,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s.",
                headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
            )

    except HTTPException:
        raise
    except Exception as e:
        # Si Redis falla durante la operación, no bloquear la request
        logger.warning("rate_limit_check_failed", extra={"error": str(e)})
