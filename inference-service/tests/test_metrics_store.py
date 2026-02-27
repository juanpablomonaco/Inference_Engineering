"""
test_metrics_store.py
---------------------
Tests del MetricsStore in-memory.
"""

import pytest
from services.metrics_store import MetricsStore


class TestMetricsStore:
    def setup_method(self):
        """Crea una instancia fresca antes de cada test."""
        self.store = MetricsStore()

    def test_initial_state_zero(self):
        assert self.store.total_requests == 0
        assert self.store.cache_hits == 0
        assert self.store.cache_misses == 0

    def test_record_request_increments(self):
        self.store.record_request()
        self.store.record_request()
        assert self.store.total_requests == 2

    def test_record_embedding_cache_hit(self):
        self.store.record_embedding(0.0, cache_hit=True)
        assert self.store.cache_hits == 1
        assert self.store.cache_misses == 0

    def test_record_embedding_cache_miss(self):
        self.store.record_embedding(50.0, cache_hit=False)
        assert self.store.cache_misses == 1
        assert self.store.embedding_time_sum_ms == 50.0

    def test_avg_embedding_ms(self):
        self.store.record_embedding(40.0, cache_hit=False)
        self.store.record_embedding(60.0, cache_hit=False)
        snap = self.store.get_snapshot()
        assert snap["avg_embedding_ms"] == 50.0

    def test_avg_search_ms(self):
        self.store.record_search(10.0)
        self.store.record_search(20.0)
        snap = self.store.get_snapshot()
        assert snap["avg_search_ms"] == 15.0

    def test_cache_hit_ratio(self):
        self.store.record_embedding(0.0, cache_hit=True)
        self.store.record_embedding(0.0, cache_hit=True)
        self.store.record_embedding(50.0, cache_hit=False)
        snap = self.store.get_snapshot()
        assert snap["cache_hit_ratio"] == pytest.approx(2 / 3, abs=0.01)

    def test_no_division_by_zero_on_empty(self):
        snap = self.store.get_snapshot()
        assert snap["avg_embedding_ms"] == 0.0
        assert snap["avg_search_ms"] == 0.0
        assert snap["cache_hit_ratio"] == 0.0



