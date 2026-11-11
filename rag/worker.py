"""Run one local corpus mutation at a time, reconciling interrupted publication safely."""

from __future__ import annotations

import fcntl
import logging
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from opensearchpy import OpenSearch, TransportError

from rag.aws_clients import create_bedrock_client
from rag.config import PipelineConfig
from rag.embeddings import Embedder
from rag.jobs import IngestionJob, JobCancelled, JobConflict, JobStore
from rag.publication import (
    SnapshotStore,
    StaleRevisionError,
    active_index,
    build_snapshot,
    publish_snapshot,
)

logger = logging.getLogger(__name__)


class WorkerBusyError(RuntimeError):
    """Another live local process owns ingestion execution."""


@contextmanager
def worker_lock(state_path: Path) -> Iterator[None]:
    """OS ownership survives slow model calls and is released automatically on process death."""
    path = state_path.resolve().with_name(state_path.name + ".worker.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkerBusyError("Another local ingestion worker is running") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _failure(error: Exception) -> tuple[str, bool]:
    if isinstance(error, StaleRevisionError):
        return "stale_base_revision", False
    if isinstance(error, TransportError):
        return "search_request_failed", error.status_code in {408, 429, 500, 502, 503, 504, "N/A"}
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code")
        return "model_request_failed", code in {
            "ThrottlingException",
            "ServiceUnavailableException",
            "ModelTimeoutException",
        }
    if isinstance(error, BotoCoreError):
        return "model_unavailable", True
    if isinstance(error, OSError):
        return "local_files_unavailable", True
    if isinstance(error, ValueError):
        return "invalid_build_inputs", False
    return "worker_error", False


def run_one(client: OpenSearch, config: PipelineConfig) -> IngestionJob | None:
    """Claim/recover one job; success is reconciled before considering another paid build."""
    with worker_lock(config.publication.state_path):
        store = JobStore(config.publication.state_path)
        job = store.claim()
        if job is None:
            return None
        try:
            inputs = store.inputs(job.job_id)
            if inputs.target_alias != config.publication.alias:
                raise JobConflict("Worker target differs from accepted job")
            settings = config.model_copy(deep=True)
            settings.chunking = inputs.chunking
            settings.models = inputs.models
            settings.retrieval = inputs.retrieval
            settings.generation = inputs.generation
            snapshots = SnapshotStore(config.publication.state_path)
            completed = snapshots.get(job.job_id)
            current = active_index(client, config.publication.alias)
            if completed is not None and (
                snapshots.was_published(job.job_id) or current == completed.index_name
            ):
                if current == completed.index_name:
                    publish_snapshot(client, settings, completed, job.request.base_revision)
                return store.finish(job.job_id, "ready")
            store.checkpoint(job.job_id)
            if len(job.attempts) > config.ingestion.max_attempts:
                raise JobConflict("Attempt limit reached after interruption")
            if current != job.request.base_revision:
                raise StaleRevisionError("Active corpus changed before execution")
            with ExitStack() as stack:
                embedder = None
                if job.request.with_embeddings:
                    runtime = stack.enter_context(closing(create_bedrock_client(settings)))
                    embedder = Embedder(settings, runtime)
                snapshot = build_snapshot(
                    client,
                    settings,
                    inputs.sources,
                    inputs.subset,
                    config.ingestion.raw_dir,
                    job.job_id,
                    embedder=embedder,
                    checkpoint=lambda: store.checkpoint(job.job_id),
                )
            store.prepare_publish(job.job_id)
            publish_snapshot(client, settings, snapshot, job.request.base_revision)
            return store.finish(job.job_id, "ready")
        except JobCancelled:
            return store.finish(job.job_id, "cancelled")
        except Exception as error:
            # This is the worker boundary: preserve a durable failure, never a secret payload.
            code, retryable = _failure(error)
            logger.error(
                "Ingestion failed", extra={"job_id": job.job_id, "error_type": type(error).__name__}
            )
            return store.finish(job.job_id, "failed", error_code=code, retryable=retryable)
