"""Keep comparison capacity owned by blocking work through disconnects and deadlines."""

from __future__ import annotations

import asyncio
import threading
from time import monotonic

from fastapi import HTTPException, Request
from opensearchpy import OpenSearch

from rag.api.limits import SearchLimiter
from rag.api.observability import observation
from rag.comparison import compare_retrieval
from rag.config import PipelineConfig
from rag.models import ComparisonRequest, ComparisonResult
from rag.workflow_limits import WorkflowCancelled, WorkflowDeadlineError

POLL_SECONDS = 0.1


async def comparison_response(
    body: ComparisonRequest,
    request: Request,
    client: OpenSearch,
    config: PipelineConfig,
    limiter: SearchLimiter,
) -> ComparisonResult:
    """Cancel follow-up work cooperatively without freeing a still-running SDK worker's slot."""
    lease = limiter.acquire()
    cancelled = threading.Event()
    deadline = monotonic() + config.api.comparison_timeout_seconds
    current = observation(request)

    def checkpoint() -> None:
        if cancelled.is_set():
            raise WorkflowCancelled("Comparison caller disconnected")
        if monotonic() >= deadline:
            raise WorkflowDeadlineError("Comparison deadline exceeded")

    def produce() -> ComparisonResult:
        started = monotonic()
        outcome = "failed"
        try:
            result = compare_retrieval(client, config, body, checkpoint=checkpoint)
            if current is not None:
                current.record(result)
            outcome = "completed"
            return result
        except (WorkflowCancelled, WorkflowDeadlineError):
            outcome = "stopped"
            raise
        finally:
            lease.close()
            if current is not None:
                current.emit(
                    "search_worker",
                    lifecycle_outcome=outcome,
                    elapsed_ms=(monotonic() - started) * 1000,
                )

    try:
        future = asyncio.get_running_loop().run_in_executor(None, produce)
    except Exception:
        lease.close()
        raise

    def observe_completion(task: asyncio.Future[ComparisonResult]) -> None:
        if not task.cancelled():
            task.exception()

    future.add_done_callback(observe_completion)
    try:
        while not future.done():
            if await request.is_disconnected():
                raise HTTPException(499, detail={"code": "request_cancelled"})
            if monotonic() >= deadline:
                raise HTTPException(504, detail={"code": "request_timeout"})
            await asyncio.wait({future}, timeout=POLL_SECONDS)
        return future.result()
    finally:
        cancelled.set()
