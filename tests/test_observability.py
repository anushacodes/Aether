"""Keep request/worker correlation truthful without logging private content or raw errors."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.responses import StreamingResponse

from rag.api.app import create_app, get_search_client
from rag.api.observability import LOGGER_NAME, EventFormatter, configure_event_logging
from rag.config import PipelineConfig
from rag.models import AnswerResult, ChunkRecord, RetrievalResult, RetrievedHit
from rag.streaming import StreamEvent


@pytest.fixture
def event_output() -> Iterator[io.StringIO]:
    """Inspect the actual isolated JSON formatter, including data omitted from ordinary extras."""
    configure_event_logging()
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(EventFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    try:
        yield output
    finally:
        logger.removeHandler(handler)
        handler.close()


def records(output: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_http_events_keep_snapshot_metrics_and_exclude_content_and_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch, event_output: io.StringIO
) -> None:
    secret = "private-credential-marker"
    question = "private-question-marker"
    config = PipelineConfig(models={"generation_model_id": secret})
    config.ingestion.admin_token = SecretStr(secret)
    chunk = ChunkRecord.model_validate_json(
        json.dumps(json.loads(Path("tests/fixtures/records.json").read_text())["chunk"])
    ).model_copy(update={"original_text": "private-passage-marker"})
    retrieved = RetrievalResult(
        hits=[RetrievedHit(chunk=chunk, method="bm25", rank=1, score=1)],
        mode="bm25",
        timings_ms={"bm25": 1, "retrieval_total": 2, "private-timing-marker": 3},
        snapshot_index="s3-rag-snapshot-" + "1" * 32,
    )
    provider = Mock(return_value=retrieved)
    monkeypatch.setattr("rag.api.app.retrieve", provider)
    app = create_app(config)
    app.dependency_overrides[get_search_client] = lambda: Mock()

    @app.get("/broken-response")
    def broken_response() -> StreamingResponse:
        def body() -> Iterator[bytes]:
            yield b"private-body-marker"
            raise RuntimeError("private-provider-marker")

        return StreamingResponse(body(), media_type="text/plain")

    with TestClient(app) as client:
        success = client.post(
            f"/retrieval?token={secret}",
            json={"query": question},
            headers={"Authorization": f"Bearer {secret}", "X-Request-ID": secret},
        )
        assert success.status_code == 200
        assert success.json()["snapshot_index"] == retrieved.snapshot_index
        assert success.headers["X-Request-ID"] != secret
        denied = client.get("/ingestion", headers={"Authorization": "Bearer wrong"})
        assert denied.status_code == 401 and denied.headers["Cache-Control"] == "no-store"
        assert client.get(f"/{secret}?question={question}").status_code == 404
        provider.side_effect = RuntimeError(f"private-provider-marker {secret} {question}")
        failed = client.post(
            "/retrieval", json={"query": question}, headers={"Origin": "http://localhost:3000"}
        )
        assert failed.status_code == 500
        assert failed.json() == {"detail": {"code": "internal_error"}}
        assert failed.headers["X-Request-ID"]
        assert failed.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert failed.headers["Access-Control-Expose-Headers"] == "X-Request-ID"
        interrupted = client.get("/broken-response")
        assert interrupted.status_code == 200 and interrupted.text == "private-body-marker"
    events = records(event_output)
    assert len(events) == 5
    assert events[0]["request_id"] == success.headers["X-Request-ID"]
    assert events[0]["result"]["snapshot_index"] == retrieved.snapshot_index
    assert events[0]["result"]["timings_ms"] == {"bm25": 1, "retrieval_total": 2}
    assert events[0]["result"]["model_cost_usd"] is None
    assert events[1]["error_code"] == "administrator_required"
    assert events[2]["route"] == "unmatched"
    assert events[3]["error_type"] == "RuntimeError"
    assert events[3]["lifecycle_outcome"] == "application_error"
    assert events[4]["lifecycle_outcome"] == "application_error"
    assert events[4]["status_code"] == 200
    assert len({event["request_id"] for event in events}) == 5
    assert len(events[0]["pipeline_id"]) == 64
    assert len(events[0]["code_id"]) == 64 and events[0]["schema_version"] == 1
    assert events[0]["prompt_version"]
    for marker in [
        secret,
        question,
        "private-passage-marker",
        "private-timing-marker",
        "private-provider-marker",
        "private-body-marker",
    ]:
        assert marker not in event_output.getvalue()


def test_concurrent_streams_keep_separate_http_and_worker_outcomes(
    monkeypatch: pytest.MonkeyPatch, event_output: io.StringIO
) -> None:
    both_entered, release = threading.Event(), threading.Event()
    barrier = threading.Barrier(2, action=both_entered.set, timeout=3)

    def stream(*args: object) -> Iterator[StreamEvent]:
        body = args[2]
        yield StreamEvent("token", {"text": body.query})
        barrier.wait()
        assert release.wait(timeout=3)
        if body.query == "private-failing-question":
            raise RuntimeError("private-provider-error")
        result = AnswerResult(
            answer="private-answer",
            status="insufficient_evidence",
            citation_ids=[],
            evidence=[],
            timings_ms={"bm25": 1},
            snapshot_index="s3-rag-snapshot-" + "2" * 32,
            usage_tokens={"generation_input": 7, "generation_output": 2},
        )
        yield StreamEvent("done", result.model_dump(mode="json"))

    monkeypatch.setattr("rag.api.stream.stream_answer", stream)
    app = create_app(PipelineConfig())
    app.dependency_overrides[get_search_client] = lambda: Mock()
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post, "/search/stream", json={"query": "private-failing-question"}
        )
        second = pool.submit(
            client.post, "/search/stream", json={"query": "private-success-question"}
        )
        try:
            assert both_entered.wait(timeout=1)
            health = client.get("/health/live")
            assert health.status_code == 200
        finally:
            release.set()
        failed, succeeded = first.result(timeout=5), second.result(timeout=5)
    assert "event: error" in failed.text and "event: done" not in failed.text
    assert "event: done" in succeeded.text
    entries = records(event_output)
    for response, outcome in [(failed, "failed"), (succeeded, "terminal_result_queued")]:
        related = [
            entry for entry in entries if entry["request_id"] == response.headers["X-Request-ID"]
        ]
        assert {entry["event"] for entry in related} == {"http_request", "search_worker"}
        worker = next(entry for entry in related if entry["event"] == "search_worker")
        assert worker["lifecycle_outcome"] == outcome
        if outcome == "failed":
            assert worker["error_code"] == "generation_failed" and worker["result"] is None
        else:
            assert worker["result"]["known_usage_tokens"] == {
                "generation_input": 7,
                "generation_output": 2,
            }
            assert worker["result"]["snapshot_index"].endswith("2" * 32)
    assert "private-" not in event_output.getvalue()


def test_disconnect_is_logged_before_the_blocked_worker_actually_exits(
    monkeypatch: pytest.MonkeyPatch, event_output: io.StringIO
) -> None:
    entered, release = threading.Event(), threading.Event()

    def blocked(*args: object) -> Iterator[StreamEvent]:
        yield StreamEvent("token", {"text": "private-provisional-output"})
        entered.set()
        assert release.wait(timeout=3)
        raise RuntimeError("private-late-provider-error")

    monkeypatch.setattr("rag.api.stream.stream_answer", blocked)
    app = create_app(PipelineConfig())
    app.dependency_overrides[get_search_client] = lambda: Mock()

    async def scenario() -> None:
        disconnected = asyncio.Event()
        delivered = []
        body_read = False

        async def receive() -> dict[str, object]:
            nonlocal body_read
            if not body_read:
                body_read = True
                return {"type": "http.request", "body": b'{"query":"private-question"}'}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            delivered.append(message)
            if message["type"] == "http.response.body" and b"event: token" in message.get(
                "body", b""
            ):
                assert await asyncio.to_thread(entered.wait, 1)
                disconnected.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "POST",
            "path": "/search/stream",
            "raw_path": b"/search/stream",
            "query_string": b"secret=private-query-string",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "headers": [(b"content-type", b"application/json")],
        }
        try:
            await asyncio.wait_for(app(scope, receive, send), timeout=2)
            assert entered.is_set()
            entries = records(event_output)
            assert len(entries) == 1 and entries[0]["event"] == "http_request"
            assert entries[0]["disconnected"] is True
            assert entries[0]["lifecycle_outcome"] == "disconnected"
        finally:
            release.set()
        for _ in range(100):
            if len(records(event_output)) == 2:
                break
            await asyncio.sleep(0.01)
        entries = records(event_output)
        assert len(entries) == 2
        assert entries[1]["event"] == "search_worker"
        assert entries[0]["request_id"] == entries[1]["request_id"]
        assert entries[1]["lifecycle_outcome"] == "failed"
        response_headers = dict(
            next(message for message in delivered if message["type"] == "http.response.start")[
                "headers"
            ]
        )
        assert response_headers[b"x-request-id"].decode() == entries[0]["request_id"]
        assert "private-" not in event_output.getvalue()

    asyncio.run(scenario())
