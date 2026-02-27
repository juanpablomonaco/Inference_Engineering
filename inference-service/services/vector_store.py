"""
vector_store.py
---------------
Abstracción sobre ChromaDB para almacenamiento y búsqueda vectorial persistente.

Fase 2: reemplaza el in-memory list[IndexedDocument] de search_service.py
con un vector store real que persiste en disco.

Por qué ChromaDB:
  - Embebido (no necesita servidor separado): corre dentro del proceso Python
  - Persiste en disco: los documentos sobreviven reinicios
  - API simple: add(), query() sin configuración compleja
  - Perfecto para aprender vector stores antes de pasar a Qdrant/Pinecone

Arquitectura:
  - ChromaDB corre en modo "persistent client" → escribe en /app/chroma_data
  - La colección "documents" almacena: id, embedding, texto (como metadata)
  - En startup: se carga el corpus seed si la colección está vacía
  - En runtime: POST /ingest agrega documentos dinámicamente

Seam de evolución:
  Para migrar a Qdrant, solo cambia este archivo.
  La interfaz pública (add_documents, search, count) permanece igual.
"""

import chromadb
from chromadb.config import Settings

from utils.logger import get_logger
from utils.timer import Timer

logger = get_logger(__name__)

# Directorio donde ChromaDB persiste los datos
# En Docker: montado como volumen para sobrevivir reinicios
CHROMA_PATH = "/app/chroma_data"
COLLECTION_NAME = "documents"

# ---------------------------------------------------------------------------
# Instancia global del cliente ChromaDB
# ---------------------------------------------------------------------------

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def init_store() -> None:
    """
    Inicializa el cliente ChromaDB y obtiene (o crea) la colección.
    Llamado una sola vez en el startup de FastAPI.
    """
    global _client, _collection

    logger.info("vector_store_init_started", extra={"path": CHROMA_PATH})

    with Timer() as t:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            # Usamos embeddings propios (los de SentenceTransformer)
            # ChromaDB no genera embeddings — solo los almacena y busca
            metadata={"hnsw:space": "cosine"},
        )

    logger.info(
        "vector_store_init_completed",
        extra={
            "collection": COLLECTION_NAME,
            "existing_docs": _collection.count(),
            "duration_ms": round(t.elapsed_ms, 2),
        },
    )


def get_collection() -> chromadb.Collection:
    """Retorna la colección activa. Falla si init_store() no fue llamado."""
    if _collection is None:
        raise RuntimeError("Vector store not initialized. Call init_store() at startup.")
    return _collection


def add_documents(ids: list[str], texts: list[str], embeddings: list[list[float]]) -> int:
    """
    Agrega documentos al vector store.

    Usa upsert para idempotencia: si un doc con el mismo id ya existe,
    lo actualiza en lugar de duplicarlo. Esto permite re-indexar el corpus
    seed en cada startup sin crear duplicados.

    Args:
        ids: Identificadores únicos por documento.
        texts: Textos originales (almacenados como metadata).
        embeddings: Vectores de embedding ya computados.

    Returns:
        Número de documentos en la colección después del upsert.
    """
    collection = get_collection()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,  # ChromaDB llama "documents" al texto, no al objeto
    )

    total = collection.count()
    logger.info(
        "documents_upserted",
        extra={"added": len(ids), "total_in_store": total},
    )
    return total


def search(query_embedding: list[float], n_results: int = 1) -> list[dict]:
    """
    Busca los documentos más similares usando ChromaDB's HNSW index.

    A diferencia de la búsqueda O(n) de Fase 1, ChromaDB usa HNSW
    (Hierarchical Navigable Small World) — una estructura de grafos que
    permite búsqueda aproximada en O(log n). Fundamental para escalar.

    Args:
        query_embedding: Vector de la query (lista de floats).
        n_results: Cuántos resultados retornar (top-k).

    Returns:
        Lista de dicts con: text, score (distancia → convertida a similitud), id
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances"],
    )

    # ChromaDB retorna distancia coseno (0=idéntico, 2=opuesto)
    # Convertimos a similitud: score = 1 - distance
    output = []
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc_text, distance, doc_id in zip(docs, distances, ids):
        output.append({
            "id": doc_id,
            "text": doc_text,
            "score": round(1.0 - distance, 4),
        })

    return output


def count() -> int:
    """Retorna el número de documentos almacenados."""
    if _collection is None:
        return 0
    return _collection.count()


def is_ready() -> bool:
    """True si el store fue inicializado."""
    return _collection is not None
