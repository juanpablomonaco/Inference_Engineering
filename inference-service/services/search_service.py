"""
search_service.py
-----------------
Búsqueda semántica sobre el vector store (ChromaDB en Fase 2).

Cambios respecto a Fase 1:
  - El corpus ya NO se mantiene en una lista in-memory
  - ChromaDB persiste los documentos en disco (/app/chroma_data)
  - La búsqueda usa HNSW index en lugar de O(n) brute-force
  - precompute() hace upsert en ChromaDB (idempotente en reinicios)
  - search() soporta top_k para retornar múltiples resultados

La interfaz pública (precompute, search, corpus_size, is_ready) no cambió
respecto a Fase 1 — los callers en main.py no necesitan modificaciones.
"""

from utils.logger import get_logger
from utils.timer import Timer
from services.metrics_store import metrics
import services.embedding_service as embedding_service
import services.vector_store as vector_store
from data.documents import get_documents, Document

logger = get_logger(__name__)


def precompute(documents: list[Document] | None = None) -> None:
    """
    Indexa los documentos del corpus en ChromaDB.

    Usa upsert → es idempotente: llamar múltiples veces no duplica docs.
    En Fase 1 esto reconstruía la lista en memoria en cada startup.
    En Fase 2, si el store ya tiene los docs, el upsert simplemente los actualiza.

    Args:
        documents: Documentos a indexar. None → usa get_documents() del seed.
    """
    if documents is None:
        documents = get_documents()

    logger.info(
        "corpus_index_started",
        extra={"total_documents": len(documents)},
    )

    with Timer() as t:
        ids: list[str] = []
        texts: list[str] = []
        embeddings: list[list[float]] = []

        for doc in documents:
            vector, _, _ = embedding_service.embed(doc.text)
            ids.append(doc.id)
            texts.append(doc.text)
            embeddings.append(vector.tolist())

        total = vector_store.add_documents(ids, texts, embeddings)

    logger.info(
        "corpus_index_completed",
        extra={
            "indexed": len(documents),
            "total_in_store": total,
            "duration_ms": round(t.elapsed_ms, 2),
        },
    )


def ingest(doc_id: str, text: str) -> dict:
    """
    Indexa un documento individual en tiempo de ejecución.

    Llamado por POST /ingest. Genera embedding y hace upsert en ChromaDB.

    Args:
        doc_id: ID único del documento.
        text: Contenido textual.

    Returns:
        dict con id, total_docs en store, elapsed_ms.
    """
    with Timer() as t:
        vector, cache_hit, _ = embedding_service.embed(text)
        total = vector_store.add_documents([doc_id], [text], [vector.tolist()])

    logger.info(
        "document_ingested",
        extra={
            "doc_id": doc_id,
            "total_in_store": total,
            "cache_hit": cache_hit,
            "duration_ms": round(t.elapsed_ms, 2),
        },
    )

    return {
        "id": doc_id,
        "total_documents": total,
        "elapsed_ms": round(t.elapsed_ms, 2),
    }


def search(query: str, top_k: int = 1) -> dict:
    """
    Busca los documentos más similares a la query.

    Fase 2: delega a ChromaDB (HNSW) en lugar de O(n) brute-force.
    Solo computa el embedding de la query — los del corpus están en ChromaDB.

    Args:
        query: Texto de búsqueda.
        top_k: Número de resultados a retornar.

    Returns:
        dict con: result (top-1 texto), score, results (lista top-k), elapsed_ms.

    Raises:
        RuntimeError: Si el vector store no fue inicializado.
        ValueError: Si el store está vacío.
    """
    if not vector_store.is_ready():
        raise RuntimeError("Vector store not initialized.")

    if vector_store.count() == 0:
        raise ValueError("No documents in the vector store. Use POST /ingest to add documents.")

    with Timer() as total_timer:
        # Único cómputo ML por request
        query_vector, cache_hit, _ = embedding_service.embed(query)

        # Búsqueda HNSW en ChromaDB
        results = vector_store.search(query_vector.tolist(), n_results=top_k)

    metrics.record_search(total_timer.elapsed_ms)

    top_result = results[0] if results else {"text": "", "score": 0.0, "id": ""}

    logger.info(
        "search_completed",
        extra={
            "query_preview": query[:60],
            "top_result_id": top_result.get("id"),
            "top_score": top_result.get("score"),
            "top_k": top_k,
            "query_cache_hit": cache_hit,
            "store_size": vector_store.count(),
            "duration_ms": round(total_timer.elapsed_ms, 2),
        },
    )

    return {
        "result": top_result.get("text", ""),
        "score": top_result.get("score", 0.0),
        "results": results,
        "elapsed_ms": round(total_timer.elapsed_ms, 2),
    }


def corpus_size() -> int:
    """Retorna el número de documentos en el store."""
    return vector_store.count()


def is_ready() -> bool:
    """Retorna True si el vector store está inicializado y tiene documentos."""
    return vector_store.is_ready() and vector_store.count() > 0
