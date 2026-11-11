"""Bound active local searches and per-process request rate independently of model access."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic

from rag.config import ApiConfig

WINDOW_SECONDS = 60


class RequestLimitError(RuntimeError):
    """The local process has reached its configured rate or concurrency capacity."""


class SearchLimiter:
    """A single-process development limiter; hosted replicas require a shared limit."""

    def __init__(self, config: ApiConfig, clock: Callable[[], float] = monotonic) -> None:
        self._slots = threading.BoundedSemaphore(config.max_concurrent_searches)
        self._rate_limit = config.requests_per_minute
        self._clock = clock
        self._arrivals: deque[float] = deque()
        self._lock = threading.Lock()

    @contextmanager
    def admit(self) -> Iterator[None]:
        """Reject excess work immediately and release its slot when the request ends."""
        if not self._slots.acquire(blocking=False):
            raise RequestLimitError("search_busy")
        try:
            with self._lock:
                now = self._clock()
                while self._arrivals and self._arrivals[0] <= now - WINDOW_SECONDS:
                    self._arrivals.popleft()
                if len(self._arrivals) >= self._rate_limit:
                    raise RequestLimitError("rate_limited")
                self._arrivals.append(now)
            yield
        finally:
            self._slots.release()
