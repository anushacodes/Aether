"""Keep dependency failures classified and request IDs distinct without exposing details."""

from __future__ import annotations

from unittest.mock import Mock

from fastapi.testclient import TestClient
from opensearchpy import TransportError

from rag.api.app import create_app, get_search_client
from rag.config import PipelineConfig


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
