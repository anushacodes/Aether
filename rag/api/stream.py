"""Bridge blocking provider streams to bounded, disconnect-aware server-sent events."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from concurrent.futures import CancelledError, TimeoutError
from time import monotonic

from fastapi import Request
from opensearchpy import OpenSearch
from starlette.responses import StreamingResponse

from rag.api.errors import dependency_failure
from rag.api.limits import SearchLease
from rag.api.observability import observation
from rag.config import PipelineConfig
from rag.models import SearchRequest
from rag.streaming import StreamEvent, stream_answer

QUEUE_CAPACITY = 8
POLL_SECONDS = 0.25
HEARTBEAT_SECONDS = 10


def encode_event(event: StreamEvent) -> str:
    """Encode one complete SSE frame without allowing source text to create event headers."""
    return f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


def streaming_response(
    body: SearchRequest,
    request: Request,
    client: OpenSearch,
    config: PipelineConfig,
    lease: SearchLease,
) -> StreamingResponse:
    """Hold capacity until the worker exits and stop future work when the browser leaves."""
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=QUEUE_CAPACITY)
    cancelled = threading.Event()
    deadline = monotonic() + config.api.stream_timeout_seconds
    request_id = request.state.request_id
    current = observation(request)

    def send(event: StreamEvent | None) -> bool:
        if cancelled.is_set() or loop.is_closed():
            return False
        pending_put = outbox.put(event)
        try:
            pending = asyncio.run_coroutine_threadsafe(pending_put, loop)
        except RuntimeError:
            pending_put.close()
            cancelled.set()
            return False
        while True:
            try:
                pending.result(timeout=POLL_SECONDS)
                return True
            except TimeoutError:
                if cancelled.is_set() or monotonic() >= deadline:
                    pending.cancel()
                    cancelled.set()
                    return False
            except CancelledError:
                return False

    def produce() -> None:
        started = monotonic()
        outcome = "stopped"
        error_code = None
        try:
            events = stream_answer(client, config, body, cancelled)
            try:
                for event in events:
                    if event.event == "done" and current is not None:
                        current.record_data(event.data)
                    if not send(event):
                        break
                    if event.event == "done":
                        outcome = "terminal_result_queued"
            finally:
                events.close()
        except Exception as error:
            # A worker failure must become a terminal event, never a false completed answer.
            failure = dependency_failure(error)
            error_code = failure[1] if failure is not None else "generation_failed"
            outcome = "failed"
            send(StreamEvent("error", {"code": error_code, "request_id": request_id}))
        finally:
            send(None)
            lease.close()
            if current is not None:
                current.emit(
                    "search_worker",
                    lifecycle_outcome=outcome,
                    error_code=error_code,
                    elapsed_ms=(monotonic() - started) * 1000,
                )

    loop.run_in_executor(None, produce)

    async def consume() -> AsyncIterator[str]:
        last_event = monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    return
                if monotonic() >= deadline:
                    yield encode_event(
                        StreamEvent("error", {"code": "request_timeout", "request_id": request_id})
                    )
                    return
                try:
                    event = await asyncio.wait_for(outbox.get(), timeout=POLL_SECONDS)
                except TimeoutError:
                    if monotonic() - last_event >= HEARTBEAT_SECONDS:
                        yield ": keepalive\n\n"
                        last_event = monotonic()
                    continue
                if event is None:
                    return
                yield encode_event(event)
                last_event = monotonic()
        finally:
            cancelled.set()

    return StreamingResponse(
        consume(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
