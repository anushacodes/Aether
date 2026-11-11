"""Administrator boundaries apply before state access; HTTP acceptance never runs a build."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from opensearchpy import NotFoundError
from pydantic import SecretStr

from rag.api.app import create_app, get_search_client
from rag.config import PipelineConfig
from rag.jobs import JobStore


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
            ("GET", "/ingestion"),
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


def test_job_pages_survive_new_submissions_and_protect_durable_history(tmp_path: Path) -> None:
    config = PipelineConfig()
    config.ingestion.admin_token = SecretStr("local-test-token")
    config.publication.state_path = tmp_path / "state.sqlite3"
    search = Mock()
    search.indices.get_alias.side_effect = NotFoundError(404, "missing")
    application = create_app(config)
    application.dependency_overrides[get_search_client] = lambda: search
    headers = {"Authorization": "Bearer local-test-token"}
    with TestClient(application) as client:
        assert (
            client.get("/ingestion", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )
        assert not config.publication.state_path.exists()
        accepted = []
        for key in ["first", "second", "third", "fourth"]:
            response = client.post(
                "/ingestion",
                json={"base_revision": None},
                headers=headers | {"Idempotency-Key": key},
            )
            assert response.status_code == 202
            accepted.append(response.json()["job_id"])
        store = JobStore(config.publication.state_path)
        store.claim()
        store.finish(accepted[0], "ready")
        store.claim()
        store.finish(accepted[1], "failed", error_code="search_request_failed", retryable=True)
        store.cancel(accepted[2])
        page_response = client.get("/ingestion?limit=2", headers=headers)
        assert page_response.headers["Cache-Control"] == "no-store"
        first = page_response.json()
        assert [row["job_id"] for row in first["jobs"]] == accepted[:1:-1]
        assert [row["status"] for row in first["jobs"]] == ["queued", "cancelled"]
        client.post(
            "/ingestion",
            json={"base_revision": None},
            headers=headers | {"Idempotency-Key": "newer-between-pages"},
        )
        older = client.get(
            f"/ingestion?limit=2&before={first['next_before']}", headers=headers
        ).json()
        assert [row["job_id"] for row in older["jobs"]] == accepted[1::-1]
        assert older["next_before"] is None
        assert older["jobs"][0]["attempts"][-1]["retryable"] is True
        assert older["jobs"][0]["attempts"][-1]["error_code"] == "search_request_failed"
        assert older["jobs"][1]["snapshot_id"] == accepted[0]
        assert not {"inputs", "idempotency_key", "admin_token"} & older["jobs"][0].keys()
        for query in ["limit=51", "limit=0", "before=0", "before=9223372036854775808"]:
            assert client.get(f"/ingestion?{query}", headers=headers).status_code == 422
        disabled = create_app(PipelineConfig())
        with TestClient(disabled) as disabled_client:
            assert disabled_client.get("/ingestion", headers=headers).status_code == 503
