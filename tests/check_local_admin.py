"""Verify protected administration and publication through real local search, without AWS.

Own only temporary state/indexes. A registered ready job is historical; refreshed catalog
and searches, rather than the job label alone, establish the current published corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from opensearchpy import TransportError
from pydantic import SecretStr

from rag.api.app import create_app, get_search_client
from rag.config import PipelineConfig, load_config
from rag.publication import active_index
from rag.search_store import create_search_client
from rag.worker import run_one


def check(config: PipelineConfig, directory: Path) -> dict[str, object]:
    """Drive the same approved mutations exposed by the administrator browser screen."""
    baseline_aliases = {config.publication.alias, "s3-rag-core"}
    config.publication.alias = f"s3-rag-admin-check-{uuid4().hex}"
    config.index.name = config.publication.alias
    config.publication.state_path = directory / "state.sqlite3"
    config.publication.snapshot_dir = directory / "snapshots"
    config.embedding.cache_path = directory / "embeddings.sqlite3"
    config.reranking.enabled = False
    token = "isolated-admin-check-not-a-real-secret"
    config.ingestion.admin_token = SecretStr(token)
    headers = {"Authorization": f"Bearer {token}"}
    created: set[str] = set()
    with create_search_client(config) as search:
        original = {alias: active_index(search, alias) for alias in baseline_aliases}
        application = create_app(config)
        application.dependency_overrides[get_search_client] = lambda: search
        try:
            with (
                TestClient(application) as api,
                patch("rag.worker.create_bedrock_client", side_effect=AssertionError("No AWS")),
            ):
                assert api.get("/ingestion").status_code == 401
                assert not config.publication.state_path.exists()
                catalog = api.get("/ingestion/catalog", headers=headers).json()
                assert catalog["active_revision"] is None

                def submit(key: str, body: dict[str, object]) -> str:
                    response = api.post(
                        "/ingestion", json=body, headers=headers | {"Idempotency-Key": key}
                    )
                    assert response.status_code == 202, response.text
                    job_id = response.json()["job_id"]
                    created.add(f"s3-rag-snapshot-{job_id}")
                    return job_id

                first = submit("first-publication", {"base_revision": None})
                assert submit("first-publication", {"base_revision": None}) == first
                assert run_one(search, config).status == "ready"
                catalog = api.get("/ingestion/catalog", headers=headers).json()
                first_revision = catalog["active_revision"]
                sections = catalog["active_snapshot"]["sections"]
                assert len(sections) == 3
                assert search.count(index=first_revision)["count"] == 3
                replacement = submit("approved-update", {"base_revision": first_revision})
                with patch.object(search, "bulk", side_effect=TransportError(503, "injected")):
                    assert run_one(search, config).status == "failed"
                assert active_index(search, config.publication.alias) == first_revision
                page = api.get("/ingestion", headers=headers)
                assert page.headers["Cache-Control"] == "no-store"
                failed = page.json()["jobs"][0]
                assert failed["job_id"] == replacement and failed["status"] == "failed"
                assert failed["attempts"][-1]["error_code"] == "search_request_failed"
                assert failed["attempts"][-1]["retryable"] is True
                assert (
                    api.post(f"/ingestion/{replacement}/replay", headers=headers).status_code == 202
                )
                assert run_one(search, config).status == "ready"
                catalog = api.get("/ingestion/catalog", headers=headers).json()
                assert catalog["active_revision"] != first_revision
                assert catalog["active_snapshot"]["document_count"] == 3
                queued = submit("cancel-queued", {"base_revision": catalog["active_revision"]})
                assert (
                    api.post(f"/ingestion/{queued}/cancel", headers=headers).json()["status"]
                    == "cancelled"
                )
                assert run_one(search, config) is None
                removed = sections[0]["section_id"]
                deletion = {
                    "operation": "delete",
                    "base_revision": catalog["active_revision"],
                    "remove_section_ids": [removed],
                }
                delete_id = submit("confirmed-delete", deletion)
                assert run_one(search, config).status == "ready"
                after = api.get("/ingestion/catalog", headers=headers).json()
                assert after["active_revision"] == f"s3-rag-snapshot-{delete_id}"
                assert removed not in {
                    section["section_id"] for section in after["active_snapshot"]["sections"]
                }
                assert (
                    search.count(
                        index=after["active_revision"],
                        body={"query": {"term": {"section_id": removed}}},
                    )["count"]
                    == 0
                )
                assert search.count(index=first_revision)["count"] == 3
                result = api.post("/retrieval", json={"query": "versioning delete", "mode": "bm25"})
                assert result.status_code == 200, result.text
                assert all(hit["chunk"]["section_id"] != removed for hit in result.json()["hits"])
                stale = api.post(
                    "/ingestion", json=deletion, headers=headers | {"Idempotency-Key": "stale"}
                )
                assert stale.status_code == 409
                assert stale.json()["detail"]["code"] == "stale_base_revision"
                statuses = {
                    row["job_id"]: row["status"]
                    for row in api.get("/ingestion", headers=headers).json()["jobs"]
                }
                assert statuses == {
                    first: "ready",
                    replacement: "ready",
                    queued: "cancelled",
                    delete_id: "ready",
                }
                assert original == {alias: active_index(search, alias) for alias in original}
                return {
                    "approved_rebuild_and_update": "passed",
                    "failed_attempt_replay": "passed",
                    "queued_cancellation": "passed",
                    "refreshed_catalog_and_search_after_deletion": "passed",
                    "stale_revision_rejected": "passed",
                    "historical_snapshot_preserved": "passed",
                    "durable_job_listing": "passed",
                    "baseline_aliases_unchanged": True,
                    "aws_calls": 0,
                }
        finally:
            for index in created:
                if search.indices.exists(index=index):
                    search.indices.delete(index=index)


def main() -> None:
    """Keep the real-service check separate from the offline correctness suite."""
    with TemporaryDirectory(prefix="s3-rag-admin-") as directory:
        print(json.dumps(check(load_config(Path("config.toml")), Path(directory)), indent=2))


if __name__ == "__main__":
    main()
