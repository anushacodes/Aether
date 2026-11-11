"""Retrieve original evidence while enforcing source restrictions outside query text."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from opensearchpy import OpenSearch

from rag.config import PipelineConfig
from rag.embeddings import validate_embedding
from rag.indexing import verify_index
from rag.models import ChunkRecord, RetrievedHit, SearchScope

MAX_QUERY_CHARACTERS = 2000


def _validate_query(query: str) -> None:
    if not query.strip() or len(query) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"Query must contain 1–{MAX_QUERY_CHARACTERS} nonblank characters")


def scope_filters(scope: SearchScope) -> list[dict[str, object]]:
    """Apply the same authoritative scope to keyword and vector queries."""
    filters: list[dict[str, object]] = [
        {"terms": {"guide": scope.permitted_guides}},
        {"term": {"source_scope": scope.source_scope}},
    ]
    if scope.corpus_id is not None:
        filters.append({"term": {"corpus_id": scope.corpus_id}})
    return filters


def _normalize_hits(
    response: dict[str, Any], method: Literal["bm25", "vector"], scope: SearchScope
) -> list[RetrievedHit]:
    """Validate the SDK boundary and reject records whose provenance contradicts filters."""
    raw_hits = sorted(response["hits"]["hits"], key=lambda hit: (-hit["_score"], hit["_id"]))
    hits = []
    for rank, hit in enumerate(raw_hits, 1):
        chunk = ChunkRecord.model_validate(hit["_source"]["record"])
        if (
            chunk.chunk_id != hit["_id"]
            or chunk.provenance.guide not in scope.permitted_guides
            or chunk.source_scope != scope.source_scope
            or (scope.corpus_id is not None and chunk.corpus_id != scope.corpus_id)
        ):
            raise ValueError("Retrieved record contradicts its identity or authoritative scope")
        hits.append(RetrievedHit(chunk=chunk, method=method, rank=rank, score=hit["_score"]))
    return hits


def bm25_search(
    client: OpenSearch,
    config: PipelineConfig,
    query: str,
    scope: SearchScope,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> list[RetrievedHit]:
    """Rank matching original passages and headings using OpenSearch BM25."""
    _validate_query(query)
    if checkpoint is not None:
        checkpoint()
    response = client.search(
        index=config.index.name,
        body={
            "size": config.retrieval.candidate_count,
            "_source": ["record"],
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["original_text", "heading_text^1.5"],
                            }
                        }
                    ],
                    "filter": scope_filters(scope),
                }
            },
        },
    )
    return _normalize_hits(response, "bm25", scope)


def vector_search(
    client: OpenSearch,
    config: PipelineConfig,
    vector: list[float],
    scope: SearchScope,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> list[RetrievedHit]:
    """Use Lucene's filtered kNN query in the exact configured embedding space."""
    validated = validate_embedding(vector, config.models.embedding_dimensions)
    if checkpoint is not None:
        checkpoint()
    verify_index(client, config)
    if checkpoint is not None:
        checkpoint()
    response = client.search(
        index=config.index.name,
        body={
            "size": config.retrieval.candidate_count,
            "_source": ["record"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": validated,
                        "k": config.retrieval.candidate_count,
                        "filter": {"bool": {"filter": scope_filters(scope)}},
                    }
                }
            },
        },
    )
    return _normalize_hits(response, "vector", scope)
