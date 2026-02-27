"""
test_schemas.py
---------------
Tests de los schemas Pydantic — validan contratos de la API.

No requieren modelo ML, solo pydantic.
"""

import pytest
from pydantic import ValidationError
from models.schemas import (
    EmbeddingRequest,
    SearchRequest,
    IngestRequest,
    RagRequest,
    MetricsResponse,
)


class TestEmbeddingRequest:
    def test_valid(self):
        req = EmbeddingRequest(text="hello world")
        assert req.text == "hello world"

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(text="")

    def test_text_too_long_raises(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(text="x" * 2049)

    def test_max_length_ok(self):
        req = EmbeddingRequest(text="x" * 2048)
        assert len(req.text) == 2048


class TestSearchRequest:
    def test_default_top_k(self):
        req = SearchRequest(query="test query")
        assert req.top_k == 1

    def test_custom_top_k(self):
        req = SearchRequest(query="test", top_k=5)
        assert req.top_k == 5

    def test_top_k_too_high_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=21)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)


class TestIngestRequest:
    def test_valid(self):
        req = IngestRequest(id="doc_001", text="Some document text")
        assert req.id == "doc_001"

    def test_empty_id_raises(self):
        with pytest.raises(ValidationError):
            IngestRequest(id="", text="Some text")

    def test_id_too_long_raises(self):
        with pytest.raises(ValidationError):
            IngestRequest(id="x" * 129, text="text")


class TestRagRequest:
    def test_default_top_k(self):
        req = RagRequest(query="what is ML?")
        assert req.top_k == 3

    def test_top_k_bounded(self):
        with pytest.raises(ValidationError):
            RagRequest(query="test", top_k=11)


class TestMetricsResponse:
    def test_valid_snapshot(self):
        m = MetricsResponse(
            total_requests=10,
            total_embedding_calls=8,
            total_search_calls=5,
            avg_embedding_ms=45.2,
            avg_search_ms=3.1,
            cache_hits=6,
            cache_misses=2,
            cache_hit_ratio=0.75,
        )
        assert m.cache_hit_ratio == 0.75
