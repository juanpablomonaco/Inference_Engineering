"""
timer.py
--------
Context manager para medir latencia de bloques de código.

Diseñado como cross-cutting concern: cualquier capa (services, models)
puede medir su propia duración sin depender de frameworks externos.

Uso:
    from utils.timer import Timer

    with Timer() as t:
        result = some_expensive_operation()

    print(f"Took {t.elapsed_ms:.2f}ms")
"""

import time


class Timer:
    """
    Context manager que mide el tiempo de ejecución de un bloque.

    Attributes:
        elapsed_ms: Tiempo transcurrido en milisegundos.
                    Disponible después de salir del bloque `with`.
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        # time.perf_counter() es más preciso que time.time() para latencias
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        elapsed_seconds = time.perf_counter() - self._start
        self.elapsed_ms = elapsed_seconds * 1000
