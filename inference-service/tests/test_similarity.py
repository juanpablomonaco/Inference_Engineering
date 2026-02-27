"""
test_similarity.py
------------------
Tests unitarios para utils/similarity.py.

Estos tests no requieren modelo ML ni servicios externos —
solo numpy. Son los más rápidos del suite y corren en CI sin deps adicionales.
"""

import numpy as np
import pytest
from utils.similarity import cosine_similarity, top_k_indices


class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_returns_minus_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_returns_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_result_range(self):
        """Cosine similarity debe estar en [-1, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            a = rng.standard_normal(384)
            b = rng.standard_normal(384)
            score = cosine_similarity(a, b)
            assert -1.0 <= score <= 1.0

    def test_magnitude_invariant(self):
        """Cosine similarity es invariante a la magnitud del vector."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # mismo vector, doble magnitud
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_raises(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero-norm"):
            cosine_similarity(a, b)

    def test_high_dimensional_vector(self):
        """Funciona con vectores de 384 dims (dimensión del modelo)."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(384)
        b = rng.standard_normal(384)
        score = cosine_similarity(a, b)
        assert isinstance(score, float)


class TestTopKIndices:
    def test_top1_returns_highest(self):
        scores = [0.1, 0.9, 0.3, 0.7]
        result = top_k_indices(scores, k=1)
        assert result == [1]

    def test_top2_returns_sorted_descending(self):
        scores = [0.1, 0.9, 0.3, 0.7]
        result = top_k_indices(scores, k=2)
        assert result == [1, 3]

    def test_k_larger_than_list_clips(self):
        scores = [0.5, 0.3]
        result = top_k_indices(scores, k=10)
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        assert top_k_indices([], k=1) == []
