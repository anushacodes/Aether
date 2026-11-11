"""Classify expected dependency failures consistently across JSON and SSE transports."""

from __future__ import annotations

from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, ReadTimeoutError
from opensearchpy import TransportError

from rag.api.limits import RequestLimitError
from rag.config import ConfigurationError
from rag.generation import CitationValidationError
from rag.reranking import RerankingError
from rag.streaming import GenerationStreamError
from rag.workflow_limits import WorkflowBudgetError, WorkflowDeadlineError


def dependency_failure(error: Exception) -> tuple[int, str] | None:
    """Return a stable public code, leaving unexpected bugs to the sanitized outer boundary."""
    if isinstance(error, (WorkflowDeadlineError, ConnectTimeoutError, ReadTimeoutError)):
        return 504, "request_timeout"
    if isinstance(error, WorkflowBudgetError):
        return 429, "workflow_budget_exhausted"
    if isinstance(error, RequestLimitError):
        return 429, str(error)
    if isinstance(error, ConfigurationError):
        return 503, "model_not_configured"
    if isinstance(error, RerankingError):
        return (
            (504, "reranking_timeout")
            if error.report.reason == "timeout"
            else (502, "reranking_failed")
        )
    if isinstance(error, CitationValidationError):
        return 502, "invalid_citations"
    if isinstance(error, GenerationStreamError):
        return 502, "incomplete_generation"
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        return (
            (429, "model_throttled")
            if code in {"ThrottlingException", "TooManyRequestsException"}
            else (502, "model_request_failed")
        )
    if isinstance(error, BotoCoreError):
        return 503, "model_unavailable"
    if isinstance(error, TransportError):
        return 503, "search_unavailable"
    if isinstance(error, ValueError):
        return 502, "invalid_dependency_response"
    return None
