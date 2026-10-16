"""Small wall-clock timing helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Stopwatch:
    def __init__(self) -> None:
        self.elapsed_ms = 0.0


@contextmanager
def stopwatch() -> Iterator[Stopwatch]:
    watch = Stopwatch()
    start = time.perf_counter()
    try:
        yield watch
    finally:
        watch.elapsed_ms = (time.perf_counter() - start) * 1000.0
