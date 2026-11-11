"""Compose reusable retrieval stages with explicit clients and measured timings."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Literal

from opensearchpy import OpenSearch

from rag.aws_clients import BedrockRuntime, create_bedrock_client
from rag.config import PipelineConfig
from rag.embeddings import Embedder
from rag.fusion import reciprocal_rank_fusion
from rag.generation import INSUFFICIENT_RESPONSE, generate_answer
from rag.indexing import resolve_read_config
from rag.models import AnswerResult, RetrievalResult, RetrievedHit, SearchRequest, SearchScope
from rag.retrieval import bm25_search, vector_search


def _keyword_stage(
    client: OpenSearch,
    config: PipelineConfig,
    query: str,
    scope: SearchScope,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[list[RetrievedHit], dict[str, float]]:
    started = perf_counter()
    if checkpoint is None:
        hits = bm25_search(client, config, query, scope)
    else:
        hits = bm25_search(client, config, query, scope, checkpoint=checkpoint)
    return hits, {"bm25": (perf_counter() - started) * 1000}


def _vector_stage(
    client: OpenSearch,
    config: PipelineConfig,
    query: str,
    scope: SearchScope,
    embedder: Embedder,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[list[RetrievedHit], dict[str, float]]:
    started = perf_counter()
    if checkpoint is not None:
        checkpoint()
    vector = embedder.embed_text(query)
    embedded = perf_counter()
    if checkpoint is None:
        hits = vector_search(client, config, vector, scope)
    else:
        hits = vector_search(client, config, vector, scope, checkpoint=checkpoint)
    return hits, {
        "embedding": (embedded - started) * 1000,
        "vector": (perf_counter() - embedded) * 1000,
    }


def retrieve(
    client: OpenSearch,
    config: PipelineConfig,
    query: str,
    scope: SearchScope,
    mode: Literal["bm25", "vector", "hybrid"],
    embedder: Embedder | None = None,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> RetrievalResult:
    """Run the requested retrievers; hybrid starts independent work concurrently."""
    query = SearchRequest(query=query, mode=mode, scope=scope).query
    started = perf_counter()
    if checkpoint is not None:
        checkpoint()
    config = resolve_read_config(client, config)
    if mode == "bm25":
        hits, timings = _keyword_stage(client, config, query, scope, checkpoint)
    elif mode in {"vector", "hybrid"}:
        if embedder is None:
            raise ValueError("Vector and hybrid retrieval require a configured embedder")
        if mode == "vector":
            hits, timings = _vector_stage(client, config, query, scope, embedder, checkpoint)
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                keyword = executor.submit(_keyword_stage, client, config, query, scope, checkpoint)
                vector = executor.submit(
                    _vector_stage, client, config, query, scope, embedder, checkpoint
                )
                keyword_hits, keyword_timings = keyword.result()
                vector_hits, vector_timings = vector.result()
            fusion_started = perf_counter()
            hits = reciprocal_rank_fusion(
                [keyword_hits, vector_hits],
                constant=config.retrieval.rrf_constant,
                limit=config.retrieval.candidate_count,
            )
            timings = (
                keyword_timings
                | vector_timings
                | {"fusion": (perf_counter() - fusion_started) * 1000}
            )
    else:
        raise ValueError(f"Unknown retrieval mode: {mode!r}")
    timings["retrieval_total"] = (perf_counter() - started) * 1000
    return RetrievalResult(hits=hits, mode=mode, timings_ms=timings)


def answer_question(
    client: OpenSearch,
    config: PipelineConfig,
    request: SearchRequest,
    runtime: BedrockRuntime | None = None,
    embedder: Embedder | None = None,
) -> AnswerResult:
    """Use the shared retrieval path and create a generation client only when evidence exists."""
    if request.workflow:
        from rag.workflow import answer_with_workflow

        if embedder is not None:
            raise ValueError("Workflow owns its budgeted embedder; pass a runtime instead")
        return answer_with_workflow(client, config, request, runtime)
    owns_runtime = False
    try:
        if request.mode != "bm25" and embedder is None:
            if runtime is None:
                runtime = create_bedrock_client(config, require_generation=True)
                owns_runtime = True
            embedder = Embedder(config, runtime)
        retrieved = retrieve(client, config, request.query, request.scope, request.mode, embedder)
        if not retrieved.hits:
            return AnswerResult(
                answer=INSUFFICIENT_RESPONSE,
                citation_ids=[],
                status="insufficient_evidence",
                evidence=[],
                timings_ms=retrieved.timings_ms,
            )
        if runtime is None:
            runtime = create_bedrock_client(config, require_generation=True)
            owns_runtime = True
        return generate_answer(runtime, config, request.query, retrieved.hits, retrieved.timings_ms)
    finally:
        if owns_runtime and runtime is not None:
            runtime.close()
