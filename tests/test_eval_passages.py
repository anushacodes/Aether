"""Ensure tiny overlaps and stale source labels cannot inflate retrieval quality."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag.chunking import chunk_section
from rag.config import ChunkingConfig
from rag.eval.dataset import EvaluationPassage
from rag.eval.passages import map_passages
from rag.models import SectionRecord


def test_only_complete_matching_source_spans_receive_passage_credit() -> None:
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    section = SectionRecord.model_validate(fixture)
    chunks = chunk_section(section, ChunkingConfig(size=20, overlap=5))
    first = chunks[0]
    passage = EvaluationPassage(
        passage_id="support",
        source_id=section.source_id,
        section_id=section.section_id,
        raw_section_sha256=section.raw_sha256,
        cleaning_version=section.cleaning_version,
        text_start=0,
        text_end=first.text_end,
        text_sha256=hashlib.sha256(first.original_text.encode("utf-8")).hexdigest(),
    )
    mapping = map_passages([passage], [section], chunks)
    assert mapping.complete[chunks[0].chunk_id] == {"support"}
    assert mapping.complete[chunks[1].chunk_id] == set()
    assert mapping.partial[chunks[1].chunk_id] == {"support"}
    stale = passage.model_copy(update={"raw_section_sha256": "0" * 64})
    with pytest.raises(ValueError, match="lineage/hash mismatch"):
        map_passages([stale], [section], chunks)
