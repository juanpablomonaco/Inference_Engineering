"""
similarity.py
-------------
Función pura de similitud coseno entre dos vectores.

Por qué implementarla en lugar de usar sklearn:
  - Transparencia algorítmica: ves exactamente qué significa
    "similitud entre vectores"
  - Sin dependencias innecesarias
  - Testeable de forma aislada

Fórmula:
    cosine_similarity(A, B) = dot(A, B) / (norm(A) * norm(B))

El resultado está en el rango [-1.0, 1.0]:
  1.0  → vectores idénticos en dirección
  0.0  → vectores ortogonales (sin relación)
 -1.0  → vectores opuestos

Para embeddings de texto con all-MiniLM-L6-v2, los scores
típicos de textos relacionados están entre 0.7 y 0.99.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos vectores.

    Args:
        a: Vector numpy de shape (N,).
        b: Vector numpy de shape (N,).

    Returns:
        Float en [-1.0, 1.0]. Valores más cercanos a 1.0 indican
        mayor similitud semántica.

    Raises:
        ValueError: Si alguno de los vectores tiene norma cero (vector nulo).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            "Cannot compute cosine similarity with a zero-norm vector. "
            "This may indicate an empty or invalid input text."
        )

    # dot product dividido por el producto de las normas
    similarity: float = float(np.dot(a, b) / (norm_a * norm_b))
    return similarity


def top_k_indices(scores: list[float], k: int = 1) -> list[int]:
    """
    Retorna los índices de los k scores más altos, en orden descendente.

    Args:
        scores: Lista de scores de similitud.
        k: Número de resultados a retornar.

    Returns:
        Lista de índices ordenados por score descendente.
    """
    if not scores:
        return []

    k = min(k, len(scores))
    # argsort en orden descendente
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return sorted_indices[:k]
