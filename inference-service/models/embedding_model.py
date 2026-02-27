"""
embedding_model.py
------------------
Singleton responsable de cargar y exponer el modelo de embeddings.

Principio: el modelo se carga UNA sola vez al startup de la aplicación.
Nunca se recarga entre requests. Este módulo es stateless después del load.

Modelo elegido: all-MiniLM-L6-v2
  - 22M parámetros
  - 384 dimensiones de salida
  - Corre en CPU (~50ms por inferencia)
  - No requiere GPU ni API key
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton — una sola instancia compartida por toda la aplicación
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model() -> None:
    """
    Carga el modelo en memoria. Debe llamarse una sola vez durante el startup
    de FastAPI (lifespan). Bloquea hasta que el modelo esté listo.
    """
    global _model

    logger.info("model_loading_started", extra={"model": MODEL_NAME})

    from utils.timer import Timer
    with Timer() as t:
        _model = SentenceTransformer(MODEL_NAME)

    logger.info(
        "model_loading_completed",
        extra={"model": MODEL_NAME, "duration_ms": round(t.elapsed_ms, 2)},
    )


def get_model() -> SentenceTransformer:
    """
    Retorna la instancia del modelo. Lanza RuntimeError si load_model()
    no fue llamado previamente (protección de contrato).
    """
    if _model is None:
        raise RuntimeError(
            "Model has not been loaded. Call load_model() during application startup."
        )
    return _model


def is_loaded() -> bool:
    """Indica si el modelo está disponible para inferencia."""
    return _model is not None


def encode(text: str) -> np.ndarray:
    """
    Genera el embedding de un texto usando el modelo cargado.

    Args:
        text: Texto de entrada a codificar.

    Returns:
        np.ndarray de shape (384,) con el vector de embedding.
    """
    model = get_model()
    # convert_to_numpy=True asegura un np.ndarray consistente
    vector: np.ndarray = model.encode(text, convert_to_numpy=True)
    return vector
