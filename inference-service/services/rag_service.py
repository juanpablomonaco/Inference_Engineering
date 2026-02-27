"""
rag_service.py
--------------
Implementa Retrieval-Augmented Generation (RAG) combinando:
  1. Retrieval: búsqueda semántica sobre ChromaDB (search_service)
  2. Augmentation: construcción del prompt con el contexto recuperado
  3. Generation: llamada a Ollama (LLM local) para generar la respuesta

Flujo completo:
  POST /rag  →  search(query, top_k=3)  →  build_prompt()  →  ollama.chat()  →  respuesta

Por qué Ollama:
  - Corre localmente como servicio Docker, sin costos de API
  - Compatible con OpenAI API format (fácil de swapear después)
  - Modelos livianos: llama3.2:1b (~1.3GB), tinyllama (~600MB)
  - Perfecto para aprender el patrón RAG sin dependencias externas

Configuración:
  OLLAMA_BASE_URL: URL del servicio Ollama (default: http://ollama:11434)
  OLLAMA_MODEL: Modelo a usar (default: llama3.2:1b)
  RAG_TOP_K: Número de documentos a recuperar como contexto (default: 3)

Separación de responsabilidades:
  - rag_service.py: orquesta retrieve → augment → generate
  - search_service.py: solo retrieval (sin conocimiento de LLM)
  - ollama_client.py: solo comunicación con Ollama (sin conocimiento de RAG)
"""

import os
from utils.logger import get_logger
from utils.timer import Timer
import services.search_service as search_service
from services.ollama_client import OllamaClient, OllamaUnavailableError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración via variables de entorno
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Cliente Ollama (lazy-initialized)
_ollama_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """Retorna el cliente Ollama, creándolo si no existe."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    return _ollama_client


# ---------------------------------------------------------------------------
# Construcción del prompt RAG
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable AI assistant specialized in machine learning and artificial intelligence.
Answer questions accurately and concisely based on the provided context.
If the context doesn't fully answer the question, say so clearly but still provide what you know.
Always cite which part of the context supports your answer."""

def build_prompt(query: str, context_docs: list[dict]) -> str:
    """
    Construye el prompt RAG concatenando la query con el contexto recuperado.

    Patrón estándar de RAG:
      SYSTEM: instrucciones del asistente
      CONTEXT: documentos recuperados (numerados, con score)
      QUESTION: la query del usuario

    Args:
        query: Pregunta del usuario.
        context_docs: Lista de dicts con {text, score, id} del retriever.

    Returns:
        Prompt completo para enviar al LLM.
    """
    context_sections = []
    for i, doc in enumerate(context_docs, 1):
        context_sections.append(
            f"[Document {i}] (relevance: {doc['score']:.2f})\n{doc['text']}"
        )

    context_str = "\n\n".join(context_sections)

    prompt = f"""Context information:
---
{context_str}
---

Question: {query}

Answer based on the context above:"""

    return prompt


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def rag(query: str, top_k: int | None = None) -> dict:
    """
    Pipeline RAG completo: retrieve → augment → generate.

    Args:
        query: Pregunta del usuario.
        top_k: Número de documentos de contexto. None → usa RAG_TOP_K env var.

    Returns:
        dict con:
          - answer: Respuesta generada por el LLM
          - sources: Lista de documentos usados como contexto
          - model: Modelo Ollama usado
          - retrieve_ms: Tiempo de retrieval
          - generate_ms: Tiempo de generación
          - total_ms: Tiempo total

    Raises:
        OllamaUnavailableError: Si Ollama no está disponible
        RuntimeError: Si el vector store no está inicializado
    """
    k = top_k if top_k is not None else RAG_TOP_K

    with Timer() as total_timer:

        # ── Paso 1: RETRIEVE ─────────────────────────────────────────────
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

        # ── Paso 2: AUGMENT — construir prompt con contexto ───────────────
        prompt = build_prompt(query, context_docs)

        # ── Paso 3: GENERATE — llamar a Ollama ───────────────────────────
        client = get_ollama_client()
        with Timer() as generate_timer:
            answer = client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )

        logger.info(
            "rag_generate_completed",
            extra={
                "model": OLLAMA_MODEL,
                "answer_length": len(answer),
                "generate_ms": round(generate_timer.elapsed_ms, 2),
            },
        )

    logger.info(
        "rag_pipeline_completed",
        extra={
            "query_preview": query[:60],
            "total_ms": round(total_timer.elapsed_ms, 2),
        },
    )

    return {
        "answer": answer,
        "sources": context_docs,
        "model": OLLAMA_MODEL,
        "retrieve_ms": round(retrieve_timer.elapsed_ms, 2),
        "generate_ms": round(generate_timer.elapsed_ms, 2),
        "total_ms": round(total_timer.elapsed_ms, 2),
    }


def is_ollama_available() -> bool:
    """Verifica si Ollama está disponible y el modelo está cargado."""
    try:
        client = get_ollama_client()
        return client.health_check()
    except Exception:
        return False
