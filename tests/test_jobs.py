"""Durable intent and worker ownership survive retries without admitting a second mutation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import PipelineConfig
from rag.corpus import load_sources
from rag.jobs import IngestionRequest, JobConflict, JobInputs, JobStore
from rag.models import SubsetManifest


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
