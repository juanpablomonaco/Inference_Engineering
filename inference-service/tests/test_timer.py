"""
test_timer.py
-------------
Tests del context manager Timer de utils/timer.py.
"""

import time
from utils.timer import Timer


class TestTimer:
    def test_measures_elapsed_ms(self):
        with Timer() as t:
            time.sleep(0.05)  # 50ms
        assert t.elapsed_ms >= 45.0  # margen para CI lento
        assert t.elapsed_ms < 500.0

    def test_fast_operation_near_zero(self):
        with Timer() as t:
            _ = 1 + 1
        assert t.elapsed_ms >= 0.0
        assert t.elapsed_ms < 10.0

    def test_elapsed_available_after_exit(self):
        t = Timer()
        with t:
            pass
        assert isinstance(t.elapsed_ms, float)
        assert t.elapsed_ms >= 0.0
