"""
documents.py
------------
Corpus de documentos en memoria. Actúa como la "base de conocimiento"
del sistema de búsqueda semántica.

En producción, este módulo sería reemplazado por una conexión a
ChromaDB, Qdrant o cualquier vector store. La interfaz (get_documents())
permanecería igual — solo cambiaría la implementación interna.

Esta es la Fase 1 del camino hacia un sistema RAG completo.

Nota: Los documentos son sobre Machine Learning / AI intencionalmente,
para que la búsqueda semántica demuestre resultados coherentes.
"""

from dataclasses import dataclass


@dataclass
class Document:
    """
    Unidad básica del corpus.

    Attributes:
        id: Identificador único del documento.
        text: Contenido textual que se indexará.
    """
    id: str
    text: str


# ---------------------------------------------------------------------------
# Corpus seed — 15 documentos sobre ML/AI
# ---------------------------------------------------------------------------

_DOCUMENTS: list[Document] = [
    Document(
        id="doc_001",
        text=(
            "A neural network is a computational model inspired by the human brain, "
            "consisting of layers of interconnected nodes that process information "
            "using weighted connections."
        ),
    ),
    Document(
        id="doc_002",
        text=(
            "Backpropagation is an algorithm used to train neural networks by computing "
            "gradients of the loss function with respect to the weights, enabling the "
            "network to learn from errors."
        ),
    ),
    Document(
        id="doc_003",
        text=(
            "An embedding is a dense vector representation of data (text, images, etc.) "
            "in a continuous vector space, where semantically similar items are placed "
            "closer together."
        ),
    ),
    Document(
        id="doc_004",
        text=(
            "Cosine similarity measures the angle between two vectors in a high-dimensional "
            "space. A value close to 1.0 means the vectors point in the same direction, "
            "indicating high semantic similarity."
        ),
    ),
    Document(
        id="doc_005",
        text=(
            "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
            "language model. The retriever finds relevant documents from a knowledge base, "
            "and the generator produces an answer grounded in that context."
        ),
    ),
    Document(
        id="doc_006",
        text=(
            "A transformer is a neural network architecture based on self-attention "
            "mechanisms. It processes all tokens in parallel and has become the foundation "
            "for modern large language models like GPT and BERT."
        ),
    ),
    Document(
        id="doc_007",
        text=(
            "Inference in machine learning refers to the process of using a trained model "
            "to make predictions on new, unseen data. Optimizing inference latency is "
            "critical for production systems."
        ),
    ),
    Document(
        id="doc_008",
        text=(
            "A vector database is a specialized storage system optimized for similarity "
            "search over high-dimensional vectors. Examples include Pinecone, Qdrant, "
            "Weaviate, and ChromaDB."
        ),
    ),
    Document(
        id="doc_009",
        text=(
            "Semantic search finds documents based on meaning and context rather than "
            "exact keyword matches. It uses embeddings to represent queries and documents "
            "in the same vector space."
        ),
    ),
    Document(
        id="doc_010",
        text=(
            "Overfitting occurs when a model learns the training data too well, including "
            "its noise and outliers, resulting in poor generalization to new data. "
            "Regularization techniques help prevent it."
        ),
    ),
    Document(
        id="doc_011",
        text=(
            "A large language model (LLM) is a deep learning model trained on massive "
            "text datasets to understand and generate human language. GPT-4, Claude, "
            "and Llama are examples of LLMs."
        ),
    ),
    Document(
        id="doc_012",
        text=(
            "Microservices architecture decomposes an application into small, independent "
            "services that communicate over APIs. Each service owns its data and can be "
            "deployed independently."
        ),
    ),
    Document(
        id="doc_013",
        text=(
            "FastAPI is a modern Python web framework for building APIs. It uses type "
            "hints for automatic data validation via Pydantic and generates OpenAPI "
            "documentation automatically."
        ),
    ),
    Document(
        id="doc_014",
        text=(
            "Attention mechanism allows neural networks to focus on relevant parts of the "
            "input when producing each output token. The formula is: "
            "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V."
        ),
    ),
    Document(
        id="doc_015",
        text=(
            "Sentence transformers are models fine-tuned to produce semantically meaningful "
            "sentence embeddings. The all-MiniLM-L6-v2 model maps sentences to a "
            "384-dimensional vector space."
        ),
    ),
]


def get_documents() -> list[Document]:
    """
    Retorna el corpus completo de documentos.

    Esta función es la interfaz pública del módulo.
    En una evolución futura, aquí se conectaría a una DB real
    sin cambiar el contrato para los callers.

    Returns:
        Lista de Document con id y texto.
    """
    return _DOCUMENTS
