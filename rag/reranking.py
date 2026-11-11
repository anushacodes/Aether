"""Optionally reorder a bounded candidate window without modifying evidence or scope."""

from __future__ import annotations

import math
from collections.abc import Callable
from time import monotonic

from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, ReadTimeoutError

from rag.aws_clients import RerankRuntime, create_reranking_client, require_reranking_settings
from rag.config import PipelineConfig
from rag.models import RerankingReport, RetrievedHit, SearchRequest


class RerankingError(ValueError):
    """A required rerank failed; the report retains attempted work without provider details."""

    def __init__(self, report: RerankingReport) -> None:
        self.report = report
        super().__init__(f"Reranking failed: {report.reason}")


def rerank_candidates(
    config: PipelineConfig,
    query: str,
    hits: list[RetrievedHit],
    *,
    runtime: RerankRuntime | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[list[RetrievedHit], RerankingReport | None]:
    """Return a full permutation or an explicit original-order fallback; never drop sources."""
    settings = config.reranking
    if not settings.enabled:
        return hits, None
    query = SearchRequest(query=query).query
    window = hits[: settings.max_candidates]
    if len({hit.chunk.chunk_id for hit in hits}) != len(hits):
        raise ValueError("Reranking requires unique original candidate IDs")
    characters = len(query) + sum(len(hit.chunk.original_text) for hit in window)
    report = RerankingReport(
        status="skipped",
        model_arn=settings.model_arn,
        input_count=len(hits),
        candidate_ids=[hit.chunk.chunk_id for hit in window],
        input_characters=characters,
        attempted_calls=0,
        usage_status="not_called",
        reason="not_enough_candidates",
    )
    if len(window) < 2:
        return hits, report
    require_reranking_settings(config)

    def failed(reason: str) -> tuple[list[RetrievedHit], RerankingReport]:
        report.reason = reason
        report.status = "fallback" if settings.failure_policy == "original_order" else "failed"
        if report.status == "failed":
            raise RerankingError(report)
        return hits, report

    if characters > settings.max_total_characters or any(
        len(hit.chunk.original_text) > settings.max_document_characters for hit in window
    ):
        return failed("input_limit")
    if checkpoint is not None:
        checkpoint()
    owned = runtime is None
    deadline = monotonic() + settings.timeout_seconds
    try:
        try:
            if runtime is None:
                runtime = create_reranking_client(config)
            if checkpoint is not None:
                checkpoint()
            if monotonic() >= deadline:
                return failed("timeout")
            report.attempted_calls = 1
            report.usage_status = "not_reported"
            response = runtime.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=[
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {
                            "type": "TEXT",
                            "textDocument": {"text": hit.chunk.original_text},
                        },
                    }
                    for hit in window
                ],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {"modelArn": settings.model_arn},
                        "numberOfResults": len(window),
                    },
                },
            )
        except (ConnectTimeoutError, ReadTimeoutError):
            return failed("timeout")
        except (BotoCoreError, ClientError):
            return failed("provider_error")
        # Workflow cancellation/deadline/budget exceptions must propagate, never become fallback.
        if checkpoint is not None:
            checkpoint()
        if monotonic() >= deadline:
            return failed("timeout")
        results = response.get("results")
        if (
            response.get("nextToken")
            or not isinstance(results, list)
            or len(results) != len(window)
        ):
            return failed("incomplete_response")
        scores: dict[int, float] = {}
        for row in results:
            if not isinstance(row, dict):
                return failed("invalid_response")
            index, score = row.get("index"), row.get("relevanceScore")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(window)
                or index in scores
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                return failed("invalid_response")
            scores[index] = float(score)
        order = sorted(scores, key=lambda i: (-scores[i], i))
        ranked = [window[i] for i in order] + hits[len(window) :]
        by_id = {window[i].chunk.chunk_id: score for i, score in scores.items()}
        ranked = [
            hit.model_copy(update={"rank": rank, "rerank_score": by_id.get(hit.chunk.chunk_id)})
            for rank, hit in enumerate(ranked, 1)
        ]
        report.status, report.reason = "applied", None
        return ranked, report
    finally:
        if owned and runtime is not None:
            runtime.close()
