"""Run two scoped retrieval modes on one snapshot, retaining partial dependency failures."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, closing
from time import perf_counter

from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, ReadTimeoutError
from opensearchpy import OpenSearch, TransportError

from rag.aws_clients import create_bedrock_client
from rag.config import ConfigurationError, PipelineConfig
from rag.embeddings import Embedder
from rag.indexing import resolve_read_config, verify_index
from rag.models import ComparisonArm, ComparisonRequest, ComparisonResult
from rag.pipeline import retrieve
from rag.reranking import RerankingError


class _ModeUnavailable(ValueError):
    """A known index limitation that should not schedule a paid embedding request."""


def _error_code(error: Exception) -> str:
    if isinstance(error, _ModeUnavailable):
        return str(error)
    if isinstance(error, ConfigurationError):
        return "model_not_configured"
    if isinstance(error, RerankingError):
        return "reranking_timeout" if error.report.reason == "timeout" else "reranking_failed"
    if isinstance(error, TransportError):
        return "search_unavailable"
    if isinstance(error, (ConnectTimeoutError, ReadTimeoutError)):
        return "model_timeout"
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        return (
            "model_throttled"
            if code in {"ThrottlingException", "TooManyRequestsException"}
            else "model_request_failed"
        )
    return (
        "model_unavailable" if isinstance(error, BotoCoreError) else "invalid_dependency_response"
    )


def compare_retrieval(
    client: OpenSearch,
    config: PipelineConfig,
    request: ComparisonRequest,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> ComparisonResult:
    """Pin once, execute in declared order, and report attempts/cache effects separately.

    This is a live inspection, not a controlled benchmark: the second mode can reuse
    the first mode's embedding cache, and the two measurements are sequential.
    """
    started = perf_counter()
    if checkpoint:
        checkpoint()
    pinned = resolve_read_config(client, config)
    arms = []
    for mode in request.modes:
        if checkpoint:
            checkpoint()
        arm_started = perf_counter()
        embedder = None
        result = None
        error_code = None
        reranking = None
        try:
            with ExitStack() as stack:
                if mode != "bm25":
                    # Normal publication guarantees full vector coverage. Refuse known text-only
                    # or mismatched snapshots before credential discovery or query embedding.
                    try:
                        verify_index(client, pinned)
                    except ValueError as error:
                        raise _ModeUnavailable("invalid_index_configuration") from error
                    total = client.count(index=pinned.index.name)["count"]
                    vectors = client.count(
                        index=pinned.index.name,
                        body={"query": {"exists": {"field": "embedding"}}},
                    )["count"]
                    if total == 0 or vectors != total:
                        raise _ModeUnavailable("vector_index_not_ready")
                    if checkpoint:
                        checkpoint()
                    runtime = stack.enter_context(
                        closing(create_bedrock_client(pinned, max_attempts=1))
                    )
                    embedder = Embedder(pinned, runtime)
                result = retrieve(
                    client,
                    pinned,
                    request.query,
                    request.scope,
                    mode,
                    embedder,
                    checkpoint=checkpoint,
                )
                reranking = result.reranking
        except (TransportError, BotoCoreError, ClientError, ValueError) as error:
            result = None
            if isinstance(error, RerankingError):
                reranking = error.report
            error_code = _error_code(error)
        # Cancellation/deadline errors are intentionally outside the per-mode fallback boundary.
        if checkpoint:
            checkpoint()
        calls = (embedder.request_count if embedder else 0) + (
            reranking.attempted_calls if reranking else 0
        )
        arms.append(
            ComparisonArm(
                mode=mode,
                status="error" if error_code else "ok",
                result=result,
                error_code=error_code,
                elapsed_ms=(perf_counter() - arm_started) * 1000,
                model_calls=calls,
                embedding_cache_hits=embedder.cache_hits if embedder else 0,
                known_embedding_input_tokens=embedder.input_tokens if embedder else 0,
                model_cost_usd=None if calls else 0,
                reranking=reranking,
            )
        )
    return ComparisonResult(
        query=request.query,
        scope=request.scope,
        snapshot_index=pinned.index.name,
        arms=arms,
        total_ms=(perf_counter() - started) * 1000,
        retrieval_settings=pinned.retrieval.model_dump(mode="json"),
        reranking_settings=pinned.reranking.model_dump(mode="json"),
    )
