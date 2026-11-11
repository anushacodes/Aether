"""Correlate content-free HTTP/worker outcomes without exporting questions or evidence.

Application events use an allowlist, never exception messages, arbitrary extra fields,
headers, URLs, or response bodies. HTTP completion and blocking-worker exit are distinct.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import Request
from starlette.datastructures import MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag import __version__
from rag.artifacts import pipeline_fingerprint
from rag.config import PipelineConfig
from rag.generation import ASSESSMENT_PROMPT_VERSION, PROMPT_VERSION
from rag.models import (
    AnswerResult,
    ComparisonRequest,
    ComparisonResult,
    RetrievalResult,
    SearchRequest,
)

LOGGER_NAME = "rag.api.events"
TIMING_KEYS = {
    "bm25",
    "embedding",
    "vector",
    "fusion",
    "reranking",
    "retrieval_total",
    "context",
    "assessment",
    "generation",
    "total",
}
USAGE_KEYS = {
    "embedding_input",
    "assessment_input",
    "assessment_output",
    "generation_input",
    "generation_output",
}
IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$")
OUTCOMES = {"answered", "insufficient_evidence", "failed", "ok", "error"}
EVENT_FIELDS = {
    "route",
    "method",
    "status_code",
    "response_complete",
    "disconnected",
    "lifecycle_outcome",
    "elapsed_ms",
    "error_type",
    "error_code",
}


class EventFormatter(logging.Formatter):
    """Only render a constructed event; generic messages, exceptions and extras stay excluded."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "safe_event", {})
        return json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "service_version": __version__,
                "schema_version": 1,
            }
            | event,
            sort_keys=True,
            allow_nan=False,
        )


def configure_event_logging() -> None:
    """Own one isolated JSON handler without changing third-party or root logger settings."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(EventFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else None


@lru_cache(maxsize=1)
def _code_fingerprint() -> str:
    """Identify running package inputs even when changes have not been committed yet."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _numbers(value: object, keys: set[str]) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    return {
        key: number
        for key, number in value.items()
        if key in keys and type(number) in {int, float} and math.isfinite(number) and number >= 0
    }


def _result_summary(value: dict[str, object]) -> dict[str, object]:
    """Take only bounded metrics/identity; never emit response text or nested source objects."""
    evidence = value.get("evidence", value.get("hits", []))
    workflow = value.get("workflow")
    ranker = value.get("reranking")
    count = value.get("evidence_count", len(evidence) if isinstance(evidence, list) else None)
    summary: dict[str, object] = {
        "snapshot_index": _identifier(value.get("snapshot_index")),
        "outcome": value.get("status")
        if isinstance(value.get("status"), str) and value["status"] in OUTCOMES
        else "ok",
        "evidence_count": count if type(count) is int and count >= 0 else None,
        "timings_ms": _numbers(value.get("timings_ms"), TIMING_KEYS),
        "known_usage_tokens": _numbers(value.get("usage_tokens"), USAGE_KEYS),
        "model_cost_usd": None,
        "usage_status": "reported_partial" if value.get("usage_tokens") else "unavailable",
    }
    if isinstance(workflow, dict):
        calls = workflow.get("model_calls")
        summary["model_calls"] = calls if type(calls) is int and 0 <= calls <= 5 else None
    else:
        summary["model_calls"] = None
    if isinstance(ranker, dict):
        summary["reranking"] = {
            "status": ranker.get("status")
            if isinstance(ranker.get("status"), str)
            and ranker["status"] in {"applied", "fallback", "failed", "skipped"}
            else None,
            "attempted_calls": ranker.get("attempted_calls")
            if type(ranker.get("attempted_calls")) is int and ranker["attempted_calls"] in {0, 1}
            else None,
            "usage_status": ranker.get("usage_status")
            if ranker.get("usage_status") in {"not_reported", "not_called"}
            else "not_reported",
        }
    return summary


@dataclass
class RequestObservation:
    """One request's fixed identity and atomically replaced sanitized result summary."""

    request_id: str
    identity: dict[str, object]
    modes: list[str] = field(default_factory=list)
    workflow: bool = False
    result: dict[str, object] | None = None
    error_code: str | None = None

    def describe(self, body: SearchRequest | ComparisonRequest) -> None:
        self.modes = [body.mode] if isinstance(body, SearchRequest) else list(body.modes)
        self.workflow = body.workflow if isinstance(body, SearchRequest) else False

    def record(self, value: RetrievalResult | AnswerResult | ComparisonResult) -> None:
        if isinstance(value, ComparisonResult):
            arms = []
            for arm in value.arms:
                arms.append(
                    {
                        "mode": arm.mode,
                        "outcome": arm.status,
                        "error_code": arm.error_code,
                        "elapsed_ms": arm.elapsed_ms,
                        "model_calls": arm.model_calls,
                        "embedding_cache_hits": arm.embedding_cache_hits,
                        "known_embedding_input_tokens": arm.known_embedding_input_tokens,
                        "model_cost_usd": arm.model_cost_usd,
                    }
                )
            self.result = {
                "snapshot_index": _identifier(value.snapshot_index),
                "outcome": "ok"
                if all(arm["outcome"] == "ok" for arm in arms)
                else "partial_or_failed",
                "arms": arms,
            }
        else:
            self.record_data(
                {
                    "snapshot_index": value.snapshot_index,
                    "status": value.status if isinstance(value, AnswerResult) else "ok",
                    "evidence_count": len(value.evidence)
                    if isinstance(value, AnswerResult)
                    else len(value.hits),
                    "timings_ms": value.timings_ms,
                    "usage_tokens": value.usage_tokens if isinstance(value, AnswerResult) else {},
                    "workflow": {"model_calls": value.workflow.model_calls}
                    if isinstance(value, AnswerResult) and value.workflow
                    else None,
                    "reranking": {
                        "status": value.reranking.status,
                        "attempted_calls": value.reranking.attempted_calls,
                        "usage_status": value.reranking.usage_status,
                    }
                    if value.reranking
                    else None,
                }
            )

    def record_data(self, value: dict[str, object]) -> None:
        self.result = _result_summary(value)

    def emit(self, event: Literal["http_request", "search_worker"], **fields: object) -> None:
        # All caller fields below are controlled lifecycle metadata, not request/provider payloads.
        payload = (
            self.identity
            | {
                "event": event,
                "request_id": self.request_id,
                "modes": self.modes,
                "workflow": self.workflow,
                "result": self.result,
                "error_code": self.error_code,
            }
            | {key: value for key, value in fields.items() if key in EVENT_FIELDS}
        )
        logging.getLogger(LOGGER_NAME).info("application event", extra={"safe_event": payload})


def observation(request: Request) -> RequestObservation | None:
    """Direct helper callers outside the HTTP app may intentionally have no observer."""
    return getattr(getattr(request, "state", None), "observation", None)


class RequestLogMiddleware:
    """Observe the whole streamed response using per-call ASGI state and no body buffering."""

    def __init__(self, app: ASGIApp, config: PipelineConfig) -> None:
        self.app = app
        self.allowed_origins = tuple(config.api.allowed_origins)
        self.identity = {
            "code_id": _code_fingerprint(),
            "pipeline_id": pipeline_fingerprint(config),
            "prompt_version": PROMPT_VERSION,
            "assessment_prompt_version": ASSESSMENT_PROMPT_VERSION,
            "model_id_hashes": {
                name: hashlib.sha256(value.encode()).hexdigest() if value else None
                for name, value in {
                    "embedding": config.models.embedding_model_id,
                    "generation": config.models.generation_model_id,
                    "reranking": config.reranking.model_arn if config.reranking.enabled else None,
                }.items()
            },
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        current = RequestObservation(uuid4().hex, self.identity)
        state = scope.setdefault("state", {})
        state["request_id"] = current.request_id
        state["observation"] = current
        status = 500
        response_started = False
        completed = False
        disconnected = False
        outcome = "response_incomplete"
        error_type = None

        async def receive_message() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def send_message(message: Message) -> None:
            nonlocal status, completed, response_started
            if message["type"] == "http.response.start":
                status = message["status"]
                response_started = True
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = current.request_id
                if scope["path"] == "/ingestion" or scope["path"].startswith("/ingestion/"):
                    headers["Cache-Control"] = "no-store"
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            await self.app(scope, receive_message, send_message)
            outcome = (
                "disconnected" if disconnected else "response_complete" if completed else outcome
            )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as error:
            outcome = "application_error"
            error_type = type(error).__name__
            # Do not let an unexpected message/traceback expose provider or request content.
            # Once headers have started, close the incomplete body without inventing success.
            try:
                if not response_started:
                    response = JSONResponse({"detail": {"code": "internal_error"}}, status_code=500)
                    # This boundary is outside the normal CORS middleware. Apply the same
                    # origin policy so browser callers can read its safe error and request ID.
                    error_app = CORSMiddleware(
                        response,
                        allow_origins=self.allowed_origins,
                        expose_headers=["X-Request-ID"],
                    )
                    await error_app(scope, receive_message, send_message)
                elif not completed:
                    await send_message({"type": "http.response.body", "body": b""})
            except OSError:
                disconnected = True
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            current.emit(
                "http_request",
                route=route,
                method=scope["method"]
                if scope["method"] in {"GET", "POST", "HEAD", "OPTIONS"}
                else "OTHER",
                status_code=status,
                response_complete=completed,
                disconnected=disconnected,
                error_type=error_type,
                lifecycle_outcome=outcome,
                elapsed_ms=(perf_counter() - started) * 1000,
            )
