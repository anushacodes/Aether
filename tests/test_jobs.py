"""Durable intent and worker ownership survive retries without admitting a second mutation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.config import PipelineConfig
from rag.corpus import load_sources
from rag.jobs import IngestionRequest, JobCancelled, JobConflict, JobInputs, JobStore
from rag.models import SubsetManifest
from rag.publication import SnapshotRecord, SnapshotStore
from rag.worker import WorkerBusyError, run_one, worker_lock


def inputs(config: PipelineConfig) -> JobInputs:
    """Use versioned real manifest contracts without opening source text or calling providers."""
    return JobInputs(
        sources=load_sources(Path("data/manifests/sources.json")),
        subset=SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes()),
        target_alias=config.publication.alias,
        chunking=config.chunking,
        models=config.models,
        retrieval=config.retrieval,
        generation=config.generation,
    )


def test_interrupted_job_preserves_intent_and_history_across_store_restart(tmp_path: Path) -> None:
    config = PipelineConfig()
    path = tmp_path / "jobs.sqlite3"
    request = IngestionRequest(base_revision=None)
    original = JobStore(path).submit("one-mutation", request, inputs(config))
    store = JobStore(path)
    assert store.submit("one-mutation", request, inputs(config)).job_id == original.job_id
    with pytest.raises(JobConflict, match="queue_full"):
        store.submit("excess-pending-work", request, inputs(config), max_pending=1)
    with pytest.raises(JobConflict, match="idempotency"):
        store.submit(
            "one-mutation", request.model_copy(update={"with_embeddings": True}), inputs(config)
        )
    store.claim()
    recovered = JobStore(path).claim()
    assert recovered is not None and recovered.job_id == original.job_id
    assert [attempt.outcome for attempt in recovered.attempts] == ["interrupted", "running"]
    assert store.inputs(original.job_id) == inputs(config)
    store.finish(original.job_id, "failed", error_code="search_request_failed", retryable=True)
    assert store.replay(original.job_id, 3).status == "queued"
    replayed = store.claim()
    assert replayed is not None and len(replayed.attempts) == 3
    store.finish(original.job_id, "failed", error_code="search_request_failed", retryable=True)
    with pytest.raises(JobConflict, match="attempt_limit"):
        store.replay(original.job_id, 3)


def test_worker_lock_and_publication_fence_prevent_competing_work(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    config = PipelineConfig()
    store = JobStore(path)
    job = store.submit(
        "cancel-before-publish", IngestionRequest(base_revision=None), inputs(config)
    )
    with worker_lock(path), pytest.raises(WorkerBusyError):
        with worker_lock(path):
            pytest.fail("A second worker acquired the live worker lock")
    store.claim()
    store.cancel(job.job_id)
    with pytest.raises(JobCancelled):
        store.prepare_publish(job.job_id)
    store.finish(job.job_id, "cancelled")
    other = store.submit("already-publishing", IngestionRequest(base_revision=None), inputs(config))
    store.claim()
    store.prepare_publish(other.job_id)
    with pytest.raises(JobConflict, match="publication_started"):
        store.cancel(other.job_id)


def test_recovery_after_successful_publication_does_not_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig()
    config.publication.state_path = tmp_path / "state.sqlite3"
    store = JobStore(config.publication.state_path)
    job = store.submit(
        "published-before-crash", IngestionRequest(base_revision=None), inputs(config)
    )
    store.claim()
    store.prepare_publish(job.job_id)
    snapshot = SnapshotRecord(
        snapshot_id=job.job_id,
        index_name=f"s3-rag-snapshot-{job.job_id}",
        build_id="1" * 64,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="2" * 64,
        document_count=3,
        corpus_id="s3-pilot-v1",
        has_vectors=False,
    )
    SnapshotStore(config.publication.state_path).register(snapshot)
    with SnapshotStore(config.publication.state_path).transaction() as connection:
        connection.execute(
            "UPDATE snapshots SET published_at = 'confirmed' WHERE snapshot_id = ?", (job.job_id,)
        )
    client = Mock()
    client.indices.get_alias.return_value = {"newer-external-snapshot": {"aliases": {}}}
    build = Mock(side_effect=AssertionError("Recovery must not rebuild published work"))
    monkeypatch.setattr("rag.worker.build_snapshot", build)
    recovered = run_one(client, config)
    assert recovered is not None and recovered.status == "ready"
    assert recovered.snapshot_id == job.job_id
    assert recovered.attempts[0].outcome == "interrupted"
    build.assert_not_called()
