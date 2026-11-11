"""Protect parallel retrieval and the single normalized pipeline contract."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from rag.config import PipelineConfig
from rag.models import RetrievedHit, SearchScope
from rag.pipeline import retrieve


def test_hybrid_retrieval_runs_independent_paths_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2, timeout=2)

    def keyword(*args: object) -> list[RetrievedHit]:
        barrier.wait()
        return []

    def vector(*args: object) -> list[RetrievedHit]:
        barrier.wait()
        return []

    monkeypatch.setattr("rag.pipeline.bm25_search", keyword)
    monkeypatch.setattr("rag.pipeline.vector_search", vector)
    result = retrieve(Mock(), PipelineConfig(), "versions", SearchScope(), "hybrid", Mock())
    assert result.hits == []
    assert {"bm25", "embedding", "vector", "fusion", "retrieval_total"} <= result.timings_ms.keys()
