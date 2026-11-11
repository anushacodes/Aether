"""Combine rankings without treating unrelated retriever scores as comparable."""

from __future__ import annotations

from rag.models import ChunkRecord, RetrievedHit


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedHit]], *, constant: int = 60, limit: int = 20
) -> list[RetrievedHit]:
    """Sum one-based reciprocal ranks by chunk ID and break ties by stable identity."""
    if constant <= 0 or limit <= 0:
        raise ValueError("RRF constant and result limit must be positive")
    scores: dict[str, float] = {}
    chunks: dict[str, ChunkRecord] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for hit in ranking:
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                raise ValueError(f"A ranking contains duplicate chunk ID {chunk_id}")
            seen.add(chunk_id)
            if chunk_id in chunks and chunks[chunk_id] != hit.chunk:
                raise ValueError(f"Rankings disagree about evidence for {chunk_id}")
            chunks[chunk_id] = hit.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (constant + hit.rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        RetrievedHit(chunk=chunks[chunk_id], method="hybrid", rank=rank, score=scores[chunk_id])
        for rank, chunk_id in enumerate(ordered, 1)
    ]


def diversify_evidence(hits: list[RetrievedHit], limit: int) -> list[RetrievedHit]:
    """Prefer distinct sections and exclude overlapping windows from the same evidence."""
    if limit <= 0:
        raise ValueError("Evidence limit must be positive")
    selected: list[RetrievedHit] = []
    deferred: list[RetrievedHit] = []
    sections: set[tuple[str, str, str]] = set()
    for hit in hits:
        chunk = hit.chunk
        section = (chunk.corpus_id, chunk.provenance.source_id, chunk.section_id)
        if section in sections:
            deferred.append(hit)
        else:
            selected.append(hit)
            sections.add(section)
        if len(selected) >= limit:
            return selected
    for hit in deferred:
        chunk = hit.chunk
        overlaps = any(
            other.chunk.corpus_id == chunk.corpus_id
            and other.chunk.provenance.source_id == chunk.provenance.source_id
            and other.chunk.section_id == chunk.section_id
            and (
                chunk.text_start is None
                or chunk.text_end is None
                or other.chunk.text_start is None
                or other.chunk.text_end is None
                or chunk.text_start < other.chunk.text_end
                and chunk.text_end > other.chunk.text_start
            )
            for other in selected
        )
        if not overlaps:
            selected.append(hit)
        if len(selected) >= limit:
            break
    return selected
