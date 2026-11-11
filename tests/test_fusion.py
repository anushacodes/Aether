"""Check ranking math and prevent repeated source windows from filling context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.fusion import diversify_evidence, reciprocal_rank_fusion
from rag.models import ChunkRecord, RetrievedHit


def _hit(identifier: str, rank: int, section: str = "one") -> RetrievedHit:
    record = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    chunk = ChunkRecord.model_validate(record).model_copy(
        update={"chunk_id": identifier, "section_id": section}
    )
    return RetrievedHit(chunk=chunk, method="bm25", rank=rank, score=100 / rank)


def test_rrf_matches_hand_calculation_and_stable_ties() -> None:
    left = [_hit("a", 1), _hit("shared", 2)]
    right = [_hit("b", 1), _hit("shared", 2)]
    merged = reciprocal_rank_fusion([left, right])
    assert [hit.chunk.chunk_id for hit in merged] == ["shared", "a", "b"]
    assert merged[0].score == pytest.approx(2 / 62)
    assert merged[1].score == pytest.approx(1 / 61)
    assert [hit.chunk.chunk_id for hit in reciprocal_rank_fusion([[], left])] == ["a", "shared"]


def test_context_prefers_other_sections_over_overlapping_windows() -> None:
    hits = [_hit("first", 1), _hit("overlap", 2), _hit("other", 3, "two")]
    assert [hit.chunk.chunk_id for hit in diversify_evidence(hits, 3)] == ["first", "other"]
