"""Administrator boundaries apply before state access; HTTP acceptance never runs a build."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from opensearchpy import NotFoundError
from pydantic import SecretStr

from rag.api.app import create_app, get_search_client
from rag.config import PipelineConfig


def test_administration_auth_idempotency_and_acceptance_boundary(tmp_path: Path) -> None:
    config = PipelineConfig()
    config.ingestion.admin_token = SecretStr("test-only-administrator-token-long-enough")
    config.publication.state_path = tmp_path / "state.sqlite3"
    search = Mock()
    search.indices.get_alias.side_effect = NotFoundError(404, "missing")
    application = create_app(config)
    application.dependency_overrides[get_search_client] = lambda: search
    with TestClient(application) as client:
        for method, path in [
            ("POST", "/ingestion"),
            ("GET", "/ingestion/catalog"),
            ("GET", "/ingestion/" + "1" * 32),
            ("POST", "/ingestion/" + "1" * 32 + "/cancel"),
            ("POST", "/ingestion/" + "1" * 32 + "/replay"),
        ]:
            assert client.request(method, path, json={"base_revision": None}).status_code == 401
        assert not config.publication.state_path.exists()
        headers = {
            "Authorization": "Bearer test-only-administrator-token-long-enough",
            "Idempotency-Key": "one-build",
        }
        accepted = client.post("/ingestion", json={"base_revision": None}, headers=headers)
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        repeated = client.post("/ingestion", json={"base_revision": None}, headers=headers)
        assert repeated.json()["job_id"] == job_id
        assert repeated.json()["status"] == "queued"
        conflict = client.post(
            "/ingestion", json={"base_revision": None, "catalog_id": "different"}, headers=headers
        )
        assert conflict.status_code == 409
        assert client.get(f"/ingestion/{job_id}", headers=headers).json()["status"] == "queued"
        assert client.get("/ingestion/" + "0" * 32, headers=headers).status_code == 404
        assert (
            client.post(f"/ingestion/{job_id}/cancel", headers=headers).json()["status"]
            == "cancelled"
        )
    search.bulk.assert_not_called()
    search.indices.update_aliases.assert_not_called()
