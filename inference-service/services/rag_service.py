"""
rag_service.py
--------------
Implementa el pipeline RAG con backend de generación swappeable.

Fase 6: soporta dos backends via INFERENCE_BACKEND env var:
  - "ollama" (default): desarrollo local, CPU, modelos ~1-7B
  - "vllm":            producción, GPU, throughput alto (batching)

Ambos backends exponen la misma interfaz:
  client.health_check() → bool
  client.generate(system_prompt, user_prompt) → str
  client.pull_model() → None

Cambiar de backend no requiere modificar este archivo.

Flujo completo:
  POST /rag
    → search(query, top_k)      [ChromaDB HNSW]
    → build_prompt(query, docs)  [augmentation]
    → client.generate(...)       [Ollama o vLLM]
    → respuesta con fuentes

Configuración:
  INFERENCE_BACKEND : "ollama" | "vllm"
  OLLAMA_BASE_URL   : URL de Ollama (default: http://ollama:11434)
  VLLM_BASE_URL     : URL de vLLM   (default: http://vllm:8001)
  OLLAMA_MODEL      : modelo a usar en ambos backends
  RAG_TOP_K         : documentos de contexto a recuperar
"""

import os
from utils.logger import get_logger
from utils.timer import Timer
import services.search_service as search_service
from services.ollama_client import OllamaClient
from services.vllm_client import VLLMClient

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8001")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Client factory — lazy-initialized, backend seleccionado por env var
# ---------------------------------------------------------------------------

_client: OllamaClient | VLLMClient | None = None


def get_ollama_client() -> OllamaClient | VLLMClient:
    """
    Retorna el cliente LLM activo según INFERENCE_BACKEND.

    Llamado get_ollama_client() por compatibilidad con main.py,
    pero puede retornar un VLLMClient si INFERENCE_BACKEND=vllm.
    """
    global _client
    if _client is None:
        if INFERENCE_BACKEND == "vllm":
            _client = VLLMClient(base_url=VLLM_BASE_URL, model=OLLAMA_MODEL)
            logger.info(
                "llm_backend_initialized",
                extra={"backend": "vllm", "url": VLLM_BASE_URL},
            )
        else:
            _client = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
            logger.info(
                "llm_backend_initialized",
                extra={"backend": "ollama", "url": OLLAMA_BASE_URL},
            )
    return _client


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable AI assistant specialized in machine learning and artificial intelligence.
Answer questions accurately and concisely based on the provided context.
If the context doesn't fully answer the question, say so clearly but still provide what you know.
Always cite which part of the context supports your answer."""


def build_prompt(query: str, context_docs: list[dict]) -> str:
    """
    Construye el prompt RAG con los documentos recuperados como contexto.

    Args:
        query: Pregunta del usuario.
        context_docs: Lista de {text, score, id} del retriever.

    Returns:
        Prompt completo listo para enviar al LLM.
    """
    context_sections = []
    for i, doc in enumerate(context_docs, 1):
        context_sections.append(
            f"[Document {i}] (relevance: {doc['score']:.2f})\n{doc['text']}"
        )
    context_str = "\n\n".join(context_sections)

    return f"""Context information:
---
{context_str}
---

Question: {query}

Answer based on the context above:"""


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


def rag(query: str, top_k: int | None = None) -> dict:
    """
    Pipeline RAG completo: retrieve → augment → generate.

    Args:
        query: Pregunta del usuario.
        top_k: Documentos de contexto. None → usa RAG_TOP_K env var.

    Returns:
        dict con answer, sources, model, backend, retrieve_ms, generate_ms, total_ms.

    Raises:
        OllamaUnavailableError | VLLMUnavailableError: Si el LLM no está disponible.
        RuntimeError: Si el vector store no está inicializado.
    """
    k = top_k if top_k is not None else RAG_TOP_K

    with Timer() as total_timer:
        # ── Retrieve ─────────────────────────────────────────────────────
        with Timer() as retrieve_timer:
            search_result = search_service.search(query, top_k=k)
            context_docs = search_result["results"]

        logger.info(
            "rag_retrieve_completed",
            extra={
                "query_preview": query[:60],
                "docs_retrieved": len(context_docs),
                "top_score": context_docs[0]["score"] if context_docs else 0,
                "retrieve_ms": round(retrieve_timer.elapsed_ms, 2),
            },
        )

        # ── Augment ───────────────────────────────────────────────────────
        prompt = build_prompt(query, context_docs)

        # ── Generate ──────────────────────────────────────────────────────
        client = get_ollama_client()
        with Timer() as generate_timer:
            answer = client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )

        logger.info(
            "rag_generate_completed",
            extra={
                "backend": INFERENCE_BACKEND,
                "model": OLLAMA_MODEL,
                "answer_length": len(answer),
                "generate_ms": round(generate_timer.elapsed_ms, 2),
            },
        )

    logger.info(
        "rag_pipeline_completed",
        extra={
            "query_preview": query[:60],
            "backend": INFERENCE_BACKEND,
            "total_ms": round(total_timer.elapsed_ms, 2),
        },
    )

    return {
        "answer": answer,
        "sources": context_docs,
        "model": OLLAMA_MODEL,
        "backend": INFERENCE_BACKEND,
        "retrieve_ms": round(retrieve_timer.elapsed_ms, 2),
        "generate_ms": round(generate_timer.elapsed_ms, 2),
        "total_ms": round(total_timer.elapsed_ms, 2),
    }


def is_ollama_available() -> bool:
    """Verifica si el backend LLM configurado está disponible."""
    try:
        return get_ollama_client().health_check()
    except Exception:
        return False
