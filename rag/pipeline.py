"""Compose reusable retrieval stages with explicit clients and measured timings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Literal

from opensearchpy import OpenSearch

from rag.config import PipelineConfig
from rag.embeddings import Embedder
from rag.fusion import reciprocal_rank_fusion
from rag.models import RetrievalResult, RetrievedHit, SearchScope
from rag.retrieval import bm25_search, vector_search


def _keyword_stage(
    client: OpenSearch, config: PipelineConfig, query: str, scope: SearchScope
) -> tuple[list[RetrievedHit], dict[str, float]]:
    started = perf_counter()
    hits = bm25_search(client, config, query, scope)
    return hits, {"bm25": (perf_counter() - started) * 1000}


def _vector_stage(
    client: OpenSearch, config: PipelineConfig, query: str, scope: SearchScope, embedder: Embedder
) -> tuple[list[RetrievedHit], dict[str, float]]:
    started = perf_counter()
    vector = embedder.embed_text(query)
    embedded = perf_counter()
    hits = vector_search(client, config, vector, scope)
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
) -> RetrievalResult:
    """Run the requested retrievers; hybrid starts independent work concurrently."""
    started = perf_counter()
    if mode == "bm25":
        hits, timings = _keyword_stage(client, config, query, scope)
    elif mode in {"vector", "hybrid"}:
        if embedder is None:
            raise ValueError("Vector and hybrid retrieval require a configured embedder")
        if mode == "vector":
            hits, timings = _vector_stage(client, config, query, scope, embedder)
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                keyword = executor.submit(_keyword_stage, client, config, query, scope)
                vector = executor.submit(_vector_stage, client, config, query, scope, embedder)
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
