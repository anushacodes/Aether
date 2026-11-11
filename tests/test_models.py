"""Verify serialized evidence rejects text/hash mismatches."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.models import ChunkRecord, SectionRecord


@pytest.fixture
def examples() -> dict[str, dict[str, object]]:
    return json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))


def test_section_rejects_same_length_text_tampering(examples: dict[str, dict[str, object]]) -> None:
    section = copy.deepcopy(examples["section"])
    original = SectionRecord.model_validate(section).raw_text
    section["raw_text"] = ("Y" if original[0] == "X" else "X") + original[1:]
    with pytest.raises(ValidationError, match="raw_sha256 must match raw_text"):
        SectionRecord.model_validate(section)


def test_chunk_rejects_incorrect_content_hash(examples: dict[str, dict[str, object]]) -> None:
    chunk = copy.deepcopy(examples["chunk"])
    chunk["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content_sha256 must match original_text"):
        ChunkRecord.model_validate(chunk)
