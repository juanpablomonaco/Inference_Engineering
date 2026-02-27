"""
search_service.py
-----------------
Implementa la búsqueda semántica sobre el corpus precomputado.

Responsabilidades:
  1. Mantener el corpus precomputado: lista de {text, vector} en memoria.
  2. precompute(): llamado UNA vez en startup, nunca en requests.
  3. search(): llamado en cada request, usa solo el embedding de la query.

Flujo de search():
  1. embed(query)          → embedding_service (cache-first)
  2. cosine_similarity     → utils/similarity (función pura)
  3. argmax de scores      → texto más similar
  4. retornar resultado    → al router

Nota sobre O(n):
  Con 15 documentos, O(n) es trivial (~3ms).
  La evolución natural para grandes corpus es reemplazar el loop
  por FAISS o Qdrant sin cambiar la interfaz pública de este módulo.
"""

from dataclasses import dataclass
import numpy as np

from utils.logger import get_logger
from utils.timer import Timer
from utils.similarity import cosine_similarity
from services.metrics_store import metrics
import services.embedding_service as embedding_service
from data.documents import get_documents, Document

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Estructura del corpus precomputado
# ---------------------------------------------------------------------------

@dataclass
class IndexedDocument:
    """
    Documento con su embedding precomputado.
    Almacenado en memoria durante toda la vida de la aplicación.
    """
    doc: Document
    embedding: np.ndarray


# ---------------------------------------------------------------------------
# Store del corpus — poblado en startup, inmutable después
# ---------------------------------------------------------------------------

_corpus: list[IndexedDocument] = []


def precompute(documents: list[Document] | None = None) -> None:
    """
    Precomputa los embeddings de todos los documentos del corpus.

    IMPORTANTE: debe llamarse SOLO durante el startup de FastAPI.
    No llamar desde endpoints.

    Args:
        documents: Lista de documentos a indexar.
                   Si es None, usa get_documents() del corpus seed.
    """
    global _corpus

    if documents is None:
        documents = get_documents()

    logger.info(
        "corpus_precompute_started",
        extra={"total_documents": len(documents)},
    )

    with Timer() as t:
        indexed: list[IndexedDocument] = []
        for doc in documents:
            # embed() usa el cache internamente — también llena embedding_service._cache
            vector, cache_hit, _ = embedding_service.embed(doc.text)
            indexed.append(IndexedDocument(doc=doc, embedding=vector))

    _corpus = indexed

    logger.info(
        "corpus_precompute_completed",
        extra={
            "total_documents": len(_corpus),
            "duration_ms": round(t.elapsed_ms, 2),
        },
    )


def search(query: str) -> dict:
    """
    Busca el documento más similar a la query en el corpus precomputado.

    Flujo:
      1. Genera embedding de la query (cache-first via embedding_service)
      2. Calcula cosine similarity contra cada doc del corpus (ya embeddeado)
      3. Retorna el documento con mayor score

    Args:
        query: Texto de búsqueda.

    Returns:
        dict con: result (texto), score (float), elapsed_ms (float)

    Raises:
        RuntimeError: Si el corpus no fue precomputado (precompute() no llamado).
        ValueError: Si la query es vacía o produce un vector inválido.
    """
    if not _corpus:
        raise RuntimeError(
            "Corpus has not been initialized. "
            "Call precompute() during application startup."
        )

    with Timer() as total_timer:
        # ── Paso 1: embedding de la query ─────────────────────────────────
        # Este es el ÚNICO cómputo ML por request.
        # Los embeddings del corpus vienen del startup (ya precomputados).
        query_vector, cache_hit, embed_ms = embedding_service.embed(query)

        # ── Paso 2: calcular similitud contra todos los docs ──────────────
        scores: list[float] = []
        for indexed_doc in _corpus:
            score = cosine_similarity(query_vector, indexed_doc.embedding)
            scores.append(score)

        # ── Paso 3: encontrar el más similar ──────────────────────────────
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_doc = _corpus[best_idx]
        best_score = scores[best_idx]

    metrics.record_search(total_timer.elapsed_ms)

    logger.info(
        "search_completed",
        extra={
            "query_preview": query[:60],
            "result_id": best_doc.doc.id,
            "score": round(best_score, 4),
            "query_cache_hit": cache_hit,
            "corpus_size": len(_corpus),
            "duration_ms": round(total_timer.elapsed_ms, 2),
        },
    )

    return {
        "result": best_doc.doc.text,
        "score": round(best_score, 4),
        "elapsed_ms": round(total_timer.elapsed_ms, 2),
    }


def corpus_size() -> int:
    """Retorna el número de documentos en el corpus."""
    return len(_corpus)


def is_ready() -> bool:
    """Retorna True si el corpus fue precomputado."""
    return len(_corpus) > 0
