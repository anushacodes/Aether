"""Persist local ingestion intent, attempts, and guarded transitions before execution."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from rag.config import ChunkingConfig, GenerationConfig, ModelConfig, RetrievalConfig
from rag.models import Identifier, Record, SourceManifest, SubsetManifest


class JobConflict(ValueError):
    """A requested transition would change accepted intent or a committed publication."""


class JobCancelled(RuntimeError):
    """A worker reached a safe cancellation checkpoint before publication began."""


class IngestionRequest(Record):
    """Browser input identifies approved content, never arbitrary server files or URLs."""

    operation: Literal["rebuild", "delete"] = "rebuild"
    catalog_id: Identifier = "pilot"
    base_revision: str | None = Field(pattern=r"^s3-rag-snapshot-[0-9a-f]{32}$")
    with_embeddings: bool = False
    remove_section_ids: list[Identifier] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def valid_mutation(self) -> IngestionRequest:
        """Keep deletion explicit and prevent irrelevant options from changing intent."""
        if self.operation == "delete" and (not self.base_revision or not self.remove_section_ids):
            raise ValueError("Deletion requires a base revision and section IDs")
        if self.operation == "rebuild" and self.remove_section_ids:
            raise ValueError("Rebuild cannot include removed sections")
        if len(set(self.remove_section_ids)) != len(self.remove_section_ids):
            raise ValueError("Removed section IDs must be unique")
        return self


class JobInputs(Record):
    """Freeze accepted manifests and behavior settings without copying credentials."""

    sources: SourceManifest
    subset: SubsetManifest
    target_alias: str
    chunking: ChunkingConfig
    models: ModelConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig


class JobAttempt(Record):
    """Keep the outcome of each execution, including recovery after a worker crash."""

    started_at: datetime
    finished_at: datetime | None = None
    outcome: Literal["running", "ready", "failed", "cancelled", "interrupted"] = "running"
    error_code: str | None = None
    retryable: bool = False


class IngestionJob(Record):
    """The durable public job view; source text and admin credentials are excluded."""

    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    request: IngestionRequest
    status: Literal["queued", "running", "ready", "failed", "cancelled"] = "queued"
    stage: str = "queued"
    created_at: datetime
    updated_at: datetime
    cancel_requested: bool = False
    attempts: list[JobAttempt] = Field(default_factory=list)
    snapshot_id: str | None = None


def request_hash(request: IngestionRequest) -> str:
    """A repeated key must preserve the caller's accepted mutation, including its base."""
    return hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()


class JobStore:
    """Use short SQLite transactions; a separate process lock owns worker execution."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS ingestion_jobs ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE NOT NULL, "
                "idempotency_key TEXT UNIQUE NOT NULL, request_hash TEXT NOT NULL, "
                "status TEXT NOT NULL, record_json TEXT NOT NULL, inputs_json TEXT NOT NULL)"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize transitions without holding a database transaction during a build."""
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                yield connection

    def get(self, job_id: str) -> IngestionJob | None:
        """Read progress across process restarts without claiming worker ownership."""
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT record_json FROM ingestion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return IngestionJob.model_validate_json(row[0]) if row else None

    def existing(self, key: str, request: IngestionRequest) -> IngestionJob | None:
        """Return accepted intent even if its catalog has subsequently changed."""
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT request_hash, record_json FROM ingestion_jobs WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
        if row and row[0] != request_hash(request):
            raise JobConflict("idempotency_key_conflict")
        return IngestionJob.model_validate_json(row[1]) if row else None

    def submit(
        self, key: str, request: IngestionRequest, inputs: JobInputs, *, max_pending: int = 20
    ) -> IngestionJob:
        """Accept exactly one immutable payload for an idempotency key, including races."""
        now = datetime.now(UTC)
        job = IngestionJob(job_id=uuid4().hex, request=request, created_at=now, updated_at=now)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT request_hash, record_json FROM ingestion_jobs WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row:
                if row[0] != request_hash(request):
                    raise JobConflict("idempotency_key_conflict")
                return IngestionJob.model_validate_json(row[1])
            pending = connection.execute(
                "SELECT COUNT(*) FROM ingestion_jobs WHERE status IN ('queued', 'running')"
            ).fetchone()[0]
            if pending >= max_pending:
                raise JobConflict("ingestion_queue_full")
            connection.execute(
                "INSERT INTO ingestion_jobs "
                "(job_id, idempotency_key, request_hash, status, record_json, inputs_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    key,
                    request_hash(request),
                    job.status,
                    job.model_dump_json(),
                    inputs.model_dump_json(),
                ),
            )
        return job

    def inputs(self, job_id: str) -> JobInputs:
        """Load the captured input rather than re-reading a possibly changed catalog."""
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT inputs_json FROM ingestion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobInputs.model_validate_json(row[0])

    @staticmethod
    def _save(connection: sqlite3.Connection, job: IngestionJob) -> None:
        job.updated_at = datetime.now(UTC)
        connection.execute(
            "UPDATE ingestion_jobs SET status = ?, record_json = ? WHERE job_id = ?",
            (job.status, job.model_dump_json(), job.job_id),
        )

    @staticmethod
    def _require(connection: sqlite3.Connection, job_id: str) -> IngestionJob:
        row = connection.execute(
            "SELECT record_json FROM ingestion_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return IngestionJob.model_validate_json(row[0])

    def claim(self) -> IngestionJob | None:
        """Recover an interrupted job before new work; caller must hold the process lock."""
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT record_json FROM ingestion_jobs WHERE status IN ('running', 'queued') "
                "ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, sequence LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job = IngestionJob.model_validate_json(row[0])
            now = datetime.now(UTC)
            if job.status == "running":
                job.attempts[-1].outcome = "interrupted"
                job.attempts[-1].finished_at = now
                job.attempts[-1].error_code = "worker_interrupted"
                job.attempts[-1].retryable = True
            job.attempts.append(JobAttempt(started_at=now))
            job.status = "running"
            if job.stage != "publishing":
                job.stage = "building"
            self._save(connection, job)
            return job

    def checkpoint(self, job_id: str) -> None:
        """Cancellation is cooperative until the publication fence is entered."""
        job = self.get(job_id)
        if job is None or job.status != "running":
            raise JobConflict("job_not_running")
        if job.cancel_requested:
            raise JobCancelled("cancel_requested")

    def prepare_publish(self, job_id: str) -> None:
        """Atomically stop accepting cancellation before touching the active alias."""
        with self.transaction() as connection:
            job = self._require(connection, job_id)
            if job.status != "running":
                raise JobConflict("job_not_running")
            if job.cancel_requested:
                raise JobCancelled("cancel_requested")
            job.stage = "publishing"
            self._save(connection, job)

    def finish(
        self,
        job_id: str,
        status: Literal["ready", "failed", "cancelled"],
        *,
        error_code: str | None = None,
        retryable: bool = False,
    ) -> IngestionJob:
        """Close the current attempt without losing earlier execution history."""
        with self.transaction() as connection:
            job = self._require(connection, job_id)
            if job.status != "running":
                raise JobConflict("job_not_running")
            job.status = status
            job.stage = status
            attempt = job.attempts[-1]
            attempt.outcome = status
            attempt.finished_at = datetime.now(UTC)
            attempt.error_code = error_code
            attempt.retryable = retryable
            if status == "ready":
                job.snapshot_id = job.job_id
            self._save(connection, job)
            return job

    def cancel(self, job_id: str) -> IngestionJob:
        """Cancel queued work immediately; running work stops at its next safe boundary."""
        with self.transaction() as connection:
            job = self._require(connection, job_id)
            if job.status == "running" and job.stage == "publishing":
                raise JobConflict("publication_started")
            if job.status not in {"queued", "running"}:
                raise JobConflict("job_already_terminal")
            job.cancel_requested = True
            if job.status == "queued":
                job.status = job.stage = "cancelled"
            self._save(connection, job)
            return job

    def replay(self, job_id: str, max_attempts: int) -> IngestionJob:
        """Explicit replay preserves accepted inputs, identity, and attempt history."""
        with self.transaction() as connection:
            job = self._require(connection, job_id)
            if job.status != "failed":
                raise JobConflict("only_failed_jobs_can_be_replayed")
            if len(job.attempts) >= max_attempts:
                raise JobConflict("attempt_limit_reached")
            job.status = job.stage = "queued"
            job.cancel_requested = False
            self._save(connection, job)
            return job
