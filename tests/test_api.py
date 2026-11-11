"""Keep dependency failures classified and request IDs distinct without exposing details."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from opensearchpy import TransportError

from rag.api.app import create_app, get_search_client
from rag.config import PipelineConfig
from rag.generation import CitationValidationError
from rag.streaming import StreamEvent


def test_readiness_failure_redacts_dependency_details_and_keeps_request_ids_unique() -> None:
    application = create_app(PipelineConfig())
    search = Mock()
    search.cluster.health.side_effect = TransportError(503, "private dependency details")
    application.dependency_overrides[get_search_client] = lambda: search
    with TestClient(application) as client:
        first = client.get("/health/ready")
        second = client.get("/health/ready")
    assert first.status_code == second.status_code == 503
    assert first.json() == {"detail": {"code": "search_unavailable"}}
    assert "private dependency details" not in first.text
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


def test_stream_validation_failure_sends_terminal_error_without_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_stream(*args: object) -> Iterator[StreamEvent]:
        yield StreamEvent("token", {"text": "A provisional claim [invented]."})
        raise CitationValidationError("Private provider details")

    monkeypatch.setattr("rag.api.stream.stream_answer", invalid_stream)
    application = create_app(PipelineConfig())
    application.dependency_overrides[get_search_client] = lambda: Mock()
    with TestClient(application) as client:
        response = client.post("/search/stream", json={"query": "A question"})
    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: error" in response.text
    assert "invalid_citations" in response.text
    assert "event: done" not in response.text
    assert "Private provider details" not in response.text
