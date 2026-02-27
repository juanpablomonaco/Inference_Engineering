"""
auth.py
-------
Autenticación via API Keys usando FastAPI dependencies.

Patrón: Bearer Token en el header Authorization.
  Authorization: Bearer <api_key>

Las API keys se configuran como variable de entorno:
  API_KEYS=key1,key2,key3   (lista separada por comas)

Si API_KEYS no está seteada o está vacía, la autenticación está
DESACTIVADA — útil para desarrollo local sin auth overhead.

Por qué este diseño:
  - FastAPI dependencies son composables: se pueden añadir a endpoints
    individuales o a routers completos
  - La lista de keys en env var es suficiente para este scope educativo
  - Evolución natural: reemplazar el set de keys por una consulta a DB/Redis

Uso en endpoints:
    from services.auth import require_api_key
    from fastapi import Depends

    @app.post("/search")
    async def search(request: SearchRequest, _=Depends(require_api_key)):
        ...

Notas de seguridad:
  - Las keys deben tener al menos 32 caracteres aleatorios (uuid4 o similar)
  - En producción, rotar keys periódicamente
  - Loguear intentos fallidos (sin exponer la key inválida en logs)
"""

import os
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración de API Keys desde variables de entorno
# ---------------------------------------------------------------------------

_raw_keys = os.getenv("API_KEYS", "")
VALID_API_KEYS: set[str] = {k.strip() for k in _raw_keys.split(",") if k.strip()}

AUTH_ENABLED = len(VALID_API_KEYS) > 0

if AUTH_ENABLED:
    logger.info("auth_enabled", extra={"num_keys": len(VALID_API_KEYS)})
else:
    logger.info("auth_disabled_no_api_keys_configured")

# Esquema Bearer para extraer el token del header Authorization
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> str:
    """
    FastAPI dependency que valida la API key.

    Si AUTH_ENABLED=False → permite todas las requests (dev mode).
    Si AUTH_ENABLED=True  → valida el Bearer token contra VALID_API_KEYS.

    Args:
        credentials: Extraído automáticamente del header Authorization.

    Returns:
        La API key validada (útil para logging por key).

    Raises:
        HTTPException 401: Sin header Authorization.
        HTTPException 403: API key inválida.
    """
    if not AUTH_ENABLED:
        return "no-auth"

    if credentials is None:
        logger.warning("auth_missing_credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Use: Authorization: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = credentials.credentials

    if api_key not in VALID_API_KEYS:
        # Loguear el intento sin exponer la key inválida
        logger.warning(
            "auth_invalid_api_key",
            extra={"key_prefix": api_key[:4] + "..." if len(api_key) > 4 else "***"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    logger.debug("auth_success", extra={"key_prefix": api_key[:4] + "..."})
    return api_key
