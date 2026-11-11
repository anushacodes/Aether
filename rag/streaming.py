"""Stream provisional model text, validating completion and citations before a final result."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import ExitStack, closing
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

from opensearchpy import OpenSearch

from rag.aws_clients import BedrockRuntime, RerankRuntime, create_bedrock_client
from rag.config import PipelineConfig
from rag.embeddings import Embedder
from rag.generation import (
    INSUFFICIENT_RESPONSE,
    AssembledContext,
    assemble_context,
    finalize_answer,
    generation_arguments,
    generation_usage,
)
from rag.models import AnswerResult, SearchRequest
from rag.pipeline import retrieve
from rag.workflow_limits import WorkflowCancelled

MAX_RESPONSE_CHARACTERS = 32_000


class GenerationStreamError(RuntimeError):
    """A provider stream ended early, exceeded bounds, or reported an embedded failure."""


class CancellationSignal(Protocol):
    """Streaming only needs to observe cancellation; its owner controls the signal."""

    def is_set(self) -> bool:
        """Report whether the owning request or workflow stopped."""
        ...


@dataclass(frozen=True)
class StreamEvent:
    """One transport-independent search event; tokens are provisional until done."""

    event: Literal["status", "sources", "token", "done", "error", "usage"]
    data: dict[str, object]


def stream_model_text(
    client: BedrockRuntime,
    config: PipelineConfig,
    question: str,
    context: AssembledContext,
    cancelled: CancellationSignal,
) -> Iterator[StreamEvent]:
    """Normalize SDK events and close the provider stream on error or cancellation."""
    if cancelled.is_set():
        return
    response = client.converse_stream(**generation_arguments(config, question, context))
    stop_reason = None
    usage_received = False
    characters = 0
    with closing(response["stream"]) as events:
        for event in events:
            if cancelled.is_set():
                return
            if any(key.endswith("Exception") for key in event):
                raise GenerationStreamError("Provider reported a streaming failure")
            text = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
            if text is not None:
                characters += len(text)
                if characters > MAX_RESPONSE_CHARACTERS:
                    raise GenerationStreamError("Generated response exceeds the character limit")
                yield StreamEvent("token", {"text": text})
            if "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")
            if "metadata" in event:
                usage = generation_usage(event["metadata"].get("usage"))
                usage_received = True
                yield StreamEvent("usage", dict(usage))
    if not cancelled.is_set() and stop_reason not in {"end_turn", "stop_sequence"}:
        raise GenerationStreamError("Provider stream ended without a complete answer")
    if not cancelled.is_set() and not usage_received:
        raise GenerationStreamError("Provider stream ended without token usage")


def stream_answer(
    client: OpenSearch,
    config: PipelineConfig,
    request: SearchRequest,
    cancelled: threading.Event,
    runtime: BedrockRuntime | None = None,
    *,
    rerank_runtime: RerankRuntime | None = None,
) -> Iterator[StreamEvent]:
    """Reuse normal retrieval/context/citation logic and stop scheduling work after disconnect."""
    if request.workflow:
        from rag.workflow import stream_with_workflow

        yield from stream_with_workflow(
            client, config, request, cancelled, runtime, rerank_runtime=rerank_runtime
        )
        return
    with ExitStack() as stack:
        if cancelled.is_set():
            return
        started = perf_counter()
        yield StreamEvent("status", {"stage": "retrieval"})
        embedder = None
        if request.mode != "bm25":
            if runtime is None:
                runtime = stack.enter_context(
                    closing(create_bedrock_client(config, require_generation=True))
                )
            embedder = Embedder(config, runtime)

        def checkpoint() -> None:
            if cancelled.is_set():
                raise WorkflowCancelled("Search was cancelled")

        try:
            retrieved = retrieve(
                client,
                config,
                request.query,
                request.scope,
                request.mode,
                embedder,
                checkpoint=checkpoint,
                **({"rerank_runtime": rerank_runtime} if rerank_runtime is not None else {}),
            )
        except WorkflowCancelled:
            return
        if cancelled.is_set():
            return
        context = assemble_context(retrieved.hits, config.retrieval)
        yield StreamEvent(
            "sources", {"evidence": [chunk.model_dump(mode="json") for chunk in context.evidence]}
        )
        if not context.evidence:
            result = AnswerResult(
                answer=INSUFFICIENT_RESPONSE,
                citation_ids=[],
                status="insufficient_evidence",
                evidence=[],
                timings_ms=retrieved.timings_ms,
                reranking=retrieved.reranking,
                snapshot_index=retrieved.snapshot_index,
            )
            yield StreamEvent("done", result.model_dump(mode="json"))
            return
        if cancelled.is_set():
            return
        if runtime is None:
            runtime = stack.enter_context(
                closing(create_bedrock_client(config, require_generation=True))
            )
        yield StreamEvent("status", {"stage": "generation"})
        generation_started = perf_counter()
        parts: list[str] = []
        usage: dict[str, int] = {}
        for event in stream_model_text(runtime, config, request.query, context, cancelled):
            if cancelled.is_set():
                return
            if event.event == "token":
                parts.append(str(event.data["text"]))
                yield event
            elif event.event == "usage":
                usage = {
                    name: int(value) for name, value in event.data.items() if isinstance(value, int)
                }
        if cancelled.is_set():
            return
        timings = retrieved.timings_ms | {
            "generation": (perf_counter() - generation_started) * 1000,
            "total": (perf_counter() - started) * 1000,
        }
        result = finalize_answer("".join(parts).strip(), context, timings)
        result.usage_tokens = usage
        result.reranking = retrieved.reranking
        result.snapshot_index = retrieved.snapshot_index
        yield StreamEvent("done", result.model_dump(mode="json"))
