"""Ensure failed or rejected requests release capacity without bypassing the rate window."""

from __future__ import annotations

import pytest

from rag.api.limits import RequestLimitError, SearchLimiter
from rag.config import ApiConfig


def test_search_capacity_recovers_after_failure_and_rate_window() -> None:
    now = 100.0
    limiter = SearchLimiter(
        ApiConfig(max_concurrent_searches=1, requests_per_minute=2), lambda: now
    )
    with pytest.raises(ValueError, match="dependency failed"), limiter.admit():
        with pytest.raises(RequestLimitError, match="search_busy"), limiter.admit():
            pass
        raise ValueError("dependency failed")
    with limiter.admit():
        pass
    with pytest.raises(RequestLimitError, match="rate_limited"), limiter.admit():
        pass
    now += 61
    with limiter.admit():
        pass
