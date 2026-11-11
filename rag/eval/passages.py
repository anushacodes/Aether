"""Map source-passage judgments onto chunk boundaries without tiny-overlap credit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rag.eval.dataset import EvaluationPassage
from rag.models import ChunkRecord, SectionRecord


@dataclass(frozen=True)
class PassageMapping:
    """Separate fully contained supporting spans from partial overlap by chunk ID."""

    complete: dict[str, set[str]]
    partial: dict[str, set[str]]

    def covered_by(self, chunk_ids: list[str]) -> set[str]:
        """Return unique source passages completely present in the selected chunks."""
        return {passage for chunk_id in chunk_ids for passage in self.complete.get(chunk_id, set())}


def map_passages(
    passages: list[EvaluationPassage], sections: list[SectionRecord], chunks: list[ChunkRecord]
) -> PassageMapping:
    """Verify lineage, then credit only passages whose full span occurs in one chunk."""
    by_section = {(section.source_id, section.section_id): section for section in sections}
    for passage in passages:
        section = by_section.get((passage.source_id, passage.section_id))
        if section is None:
            raise ValueError(f"Unknown labeled section: {passage.section_id}")
        text = section.text[passage.text_start : passage.text_end]
        if (
            passage.raw_section_sha256 != section.raw_sha256
            or passage.cleaning_version != section.cleaning_version
            or passage.text_end > len(section.text)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != passage.text_sha256
        ):
            raise ValueError(f"Labeled passage lineage/hash mismatch: {passage.passage_id}")
    complete: dict[str, set[str]] = {}
    partial: dict[str, set[str]] = {}
    for chunk in chunks:
        section = by_section.get((chunk.provenance.source_id, chunk.section_id))
        if (
            section is None
            or chunk.text_start is None
            or chunk.text_end is None
            or chunk.provenance != section.provenance
            or chunk.section_raw_start != section.raw_start
            or chunk.section_raw_end != section.raw_end
            or chunk.original_text != section.text[chunk.text_start : chunk.text_end]
        ):
            raise ValueError(f"Chunk cannot be mapped to exact source text: {chunk.chunk_id}")
        complete[chunk.chunk_id] = set()
        partial[chunk.chunk_id] = set()
        for passage in passages:
            if (passage.source_id, passage.section_id) != (
                chunk.provenance.source_id,
                chunk.section_id,
            ):
                continue
            if chunk.text_start <= passage.text_start and chunk.text_end >= passage.text_end:
                complete[chunk.chunk_id].add(passage.passage_id)
            elif chunk.text_start < passage.text_end and chunk.text_end > passage.text_start:
                partial[chunk.chunk_id].add(passage.passage_id)
    return PassageMapping(complete, partial)
