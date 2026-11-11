"""Own workflow call budgets and cooperative deadlines without abandoning in-flight SDK work."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic
from typing import Any

from rag.aws_clients import (
    BedrockRuntime,
    RerankRuntime,
    create_bedrock_client,
    create_reranking_client,
)
from rag.config import PipelineConfig


class WorkflowDeadlineError(TimeoutError):
    """The workflow deadline passed; no further retrieval or model work may be scheduled."""


class WorkflowBudgetError(RuntimeError):
    """The workflow exhausted its explicit logical model-call allowance."""


class WorkflowCancelled(RuntimeError):
    """A cancelled workflow reached a cooperative cleanup boundary."""


class BoundedRuntime:
    """Lazily own one single-attempt SDK client and gate every embedding/assessment/answer call.

    Deadlines are checked before and after blocking stages, not by detaching their threads.
    An in-flight SDK call can continue until its socket timeout. Its caller retains capacity
    and client ownership until the blocking operation actually exits.
    """

    def __init__(
        self,
        config: PipelineConfig,
        cancelled: threading.Event,
        runtime: BedrockRuntime | None = None,
        *,
        clock: Callable[[], float] | None = None,
        rerank_runtime: RerankRuntime | None = None,
    ) -> None:
        self.config = config
        self.cancelled = cancelled
        self.runtime = runtime
        self.owns_runtime = runtime is None
        self.rerank_runtime = rerank_runtime
        self.owns_rerank_runtime = rerank_runtime is None
        self.clock = clock or monotonic
        self.started = self.clock()
        self.deadline = self.started + config.workflow.timeout_seconds
        self.model_calls = 0
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    def is_set(self) -> bool:
        """Observe caller cancellation and internal cleanup without changing transport state."""
        return self.cancelled.is_set() or self._stopped.is_set()

    def stop(self) -> None:
        """Stop graph nodes while leaving the caller able to send a final result or error."""
        self._stopped.set()

    def checkpoint(self) -> None:
        """Stop follow-up work after cancellation or the total scheduling deadline."""
        if self.is_set():
            raise WorkflowCancelled("Workflow was cancelled")
        if self.clock() >= self.deadline:
            raise WorkflowDeadlineError("Workflow deadline exceeded")

    @contextmanager
    def _call_slot(self) -> Iterator[None]:
        """Reserve one shared call after successful client setup, before releasing the lock."""
        with self._lock:
            self.checkpoint()
            self.config.require_cloud_settings(require_generation=True)
            if self.model_calls >= self.config.workflow.max_model_calls:
                raise WorkflowBudgetError("Workflow model-call budget exhausted")
            yield
            self.checkpoint()
            self.model_calls += 1

    def _begin(self) -> BedrockRuntime:
        with self._call_slot():
            if self.runtime is None:
                self.runtime = create_bedrock_client(
                    self.config, require_generation=True, max_attempts=1
                )
        return self.runtime

    def rerank(self, **kwargs: Any) -> dict[str, Any]:
        """Count native reranking in the same five-call budget as embeddings and answers."""
        with self._call_slot():
            if self.rerank_runtime is None:
                self.rerank_runtime = create_reranking_client(self.config)
        response = self.rerank_runtime.rerank(**kwargs)
        self.checkpoint()
        return response

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        """Count embedding misses in the same allowance as assessment and generation."""
        response = self._begin().invoke_model(**kwargs)
        try:
            self.checkpoint()
        except (WorkflowCancelled, WorkflowDeadlineError):
            response["body"].close()
            raise
        return response

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        """Count one complete generation call and reject a response arriving after the deadline."""
        response = self._begin().converse(**kwargs)
        self.checkpoint()
        return response

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        """Count one streaming generation call; its consumer also checks each yielded event."""
        response = self._begin().converse_stream(**kwargs)
        try:
            self.checkpoint()
        except (WorkflowCancelled, WorkflowDeadlineError):
            response["stream"].close()
            raise
        return response

    def close(self) -> None:
        """Close only a client created here, after its owning graph has finished."""
        try:
            if self.owns_runtime and self.runtime is not None:
                self.runtime.close()
        finally:
            if self.owns_rerank_runtime and self.rerank_runtime is not None:
                self.rerank_runtime.close()
