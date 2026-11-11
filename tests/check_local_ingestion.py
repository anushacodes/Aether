"""Exercise durable ingestion against local OpenSearch, with synthetic embeddings and no AWS.

Run after `corpus fetch` and `make search-up`. Temporary state, indexes, and aliases are
isolated and removed. The subprocess crash proves OS lock release and durable recovery;
these checks are intentionally separate from the offline unit suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from opensearchpy import TransportError
from pydantic import SecretStr

from rag.api.app import create_app, get_search_client
from rag.config import PipelineConfig, load_config
from rag.jobs import JobStore
from rag.publication import active_index
from rag.search_store import create_search_client
from rag.worker import WorkerBusyError, run_one

CHILD = """
import signal, sys
from pathlib import Path
from rag.jobs import JobStore
from rag.worker import worker_lock
with worker_lock(Path(sys.argv[1])):
    job = JobStore(Path(sys.argv[1])).claim()
    print(job.job_id, flush=True)
    signal.pause()
"""


def check(config: PipelineConfig, directory: Path) -> dict[str, object]:
    """Verify real publication/failure paths while the provider boundary is a strict stub."""
    config.publication.alias = f"s3-rag-ingestion-check-{uuid4().hex}"
    config.index.name = config.publication.alias
    config.publication.state_path = directory / "state.sqlite3"
    config.publication.snapshot_dir = directory / "snapshots"
    config.embedding.cache_path = directory / "embeddings.sqlite3"
    config.cloud.region = "us-east-1"
    config.cloud.development_model_budget_usd = 1  # Stub-only configuration; never an AWS call.
    config.models.generation_model_id = None
    token = "local-verification-token-not-a-real-secret"
    config.ingestion.admin_token = SecretStr(token)
    runtime = Mock()

    def synthetic_vector(**_: object) -> dict[str, object]:
        vector = [1.0] + [0.0] * (config.models.embedding_dimensions - 1)
        return {
            "body": BytesIO(json.dumps({"embedding": vector, "inputTextTokenCount": 1}).encode())
        }

    runtime.invoke_model.side_effect = synthetic_vector
    created: set[str] = set()
    with create_search_client(config) as search:
        application = create_app(config)
        application.dependency_overrides[get_search_client] = lambda: search
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with (
                TestClient(application) as api,
                patch("rag.worker.create_bedrock_client", return_value=runtime),
            ):
                assert api.post("/ingestion", json={"base_revision": None}).status_code == 401
                assert not config.publication.state_path.exists()
                base: str | None = None

                def submit(key: str, *, removed: list[str] | None = None) -> str:
                    body = {"base_revision": base, "with_embeddings": True}
                    if removed:
                        body |= {"operation": "delete", "remove_section_ids": removed}
                    response = api.post(
                        "/ingestion", json=body, headers=headers | {"Idempotency-Key": key}
                    )
                    assert response.status_code == 202, response.text
                    job_id = response.json()["job_id"]
                    created.add(f"s3-rag-snapshot-{job_id}")
                    return job_id

                first = submit("first-build")
                assert submit("first-build") == first
                assert runtime.invoke_model.call_count == 0
                child = subprocess.Popen(
                    [sys.executable, "-c", CHILD, str(config.publication.state_path)],
                    stdout=subprocess.PIPE,
                    text=True,
                )
                try:
                    assert child.stdout is not None and child.stdout.readline().strip() == first
                    try:
                        run_one(search, config)
                    except WorkerBusyError:
                        pass
                    else:
                        raise AssertionError("Second worker ran while first process was live")
                finally:
                    child.kill()
                    child.wait(timeout=5)
                recovered = run_one(search, config)
                assert recovered is not None and recovered.status == "ready"
                assert recovered.attempts[0].outcome == "interrupted"
                base = active_index(search, config.publication.alias)
                assert search.count(index=config.publication.alias)["count"] == 3
                assert runtime.invoke_model.call_count == 3
                catalog = api.get("/ingestion/catalog", headers=headers).json()
                assert catalog["active_revision"] == base
                assert len(catalog["active_snapshot"]["sections"]) == 3

                uncertain = submit("published-before-status")
                with patch("rag.worker.JobStore.finish", side_effect=KeyboardInterrupt):
                    try:
                        run_one(search, config)
                    except KeyboardInterrupt:
                        pass
                    else:
                        raise AssertionError("Post-publication interruption did not occur")
                assert (
                    api.get(f"/ingestion/{uncertain}", headers=headers).json()["stage"]
                    == "publishing"
                )
                assert (
                    api.post(f"/ingestion/{uncertain}/cancel", headers=headers).status_code == 409
                )
                reconciled = run_one(search, config)
                assert reconciled is not None and reconciled.job_id == uncertain
                assert (
                    reconciled.status == "ready" and reconciled.attempts[0].outcome == "interrupted"
                )
                assert runtime.invoke_model.call_count == 3
                base = active_index(search, config.publication.alias)

                cancelled = submit("cancel-queued")
                assert (
                    api.post(f"/ingestion/{cancelled}/cancel", headers=headers).json()["status"]
                    == "cancelled"
                )
                assert run_one(search, config) is None

                failed = submit("failed-build")
                with patch.object(
                    search, "bulk", side_effect=TransportError(503, "injected failure")
                ):
                    result = run_one(search, config)
                assert result is not None and result.status == "failed"
                assert result.attempts[-1].retryable
                assert active_index(search, config.publication.alias) == base
                assert api.post(f"/ingestion/{failed}/replay", headers=headers).status_code == 202
                replayed = run_one(search, config)
                assert replayed is not None and replayed.status == "ready"
                assert len(replayed.attempts) == 2
                assert runtime.invoke_model.call_count == 3
                base = active_index(search, config.publication.alias)

                accepted = submit("first-at-same-base")
                stale = submit("second-at-same-base")
                assert run_one(search, config).job_id == accepted
                stale_result = run_one(search, config)
                assert stale_result is not None and stale_result.job_id == stale
                assert stale_result.attempts[-1].error_code == "stale_base_revision"
                assert not stale_result.attempts[-1].retryable
                assert runtime.invoke_model.call_count == 3
                base = active_index(search, config.publication.alias)

                deletion = submit("delete-api-section", removed=["api-deleteobject-description"])
                assert run_one(search, config).job_id == deletion
                assert search.count(index=config.publication.alias)["count"] == 2
                assert runtime.invoke_model.call_count == 3
                assert (
                    len(
                        api.get("/ingestion/catalog", headers=headers).json()["active_snapshot"][
                            "sections"
                        ]
                    )
                    == 2
                )
                assert (
                    api.post(
                        "/retrieval",
                        json={"query": "versioning", "scope": {"permitted_guides": ["s3-api"]}},
                    ).json()["hits"]
                    == []
                )
                base = active_index(search, config.publication.alias)
                empty = submit(
                    "delete-remaining",
                    removed=["ug-how-versioning-works", "ug-delete-versioned-objects"],
                )
                assert run_one(search, config).job_id == empty
                assert search.count(index=config.publication.alias)["count"] == 0
                assert api.get("/health/ready").status_code == 200
                assert (
                    api.get("/ingestion/catalog", headers=headers).json()["active_snapshot"][
                        "sections"
                    ]
                    == []
                )
                answer = api.post("/search", json={"query": "versioning"})
                assert (
                    answer.status_code == 200 and answer.json()["status"] == "insufficient_evidence"
                )
                assert runtime.invoke_model.call_count == 3
                store = JobStore(config.publication.state_path)
                assert store.get(empty).status == "ready"
                return {
                    "passed": [
                        "administrator boundary",
                        "idempotent submission",
                        "single worker",
                        "process-death recovery",
                        "post-publication interruption reconciliation",
                        "queued cancellation",
                        "staging failure preserves active corpus",
                        "explicit replay",
                        "stale base rejection",
                        "cached vectors across rebuild/delete",
                        "deleted evidence disappears",
                        "delete-all publishes empty corpus",
                        "empty readiness and answer",
                    ],
                    "synthetic_embedding_requests": runtime.invoke_model.call_count,
                    "real_aws_requests": 0,
                }
        finally:
            for name in created:
                if search.indices.exists(index=name):
                    search.indices.delete(index=name)


def main() -> None:
    config = load_config(Path("config.toml"))
    if config.opensearch_url.host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("This verification only runs against local OpenSearch")
    with TemporaryDirectory(prefix="s3-rag-ingestion-") as temporary:
        print(json.dumps(check(config, Path(temporary)), indent=2))


if __name__ == "__main__":
    main()
