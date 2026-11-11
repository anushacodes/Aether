"""Shared records. Raw offsets always address decoded, unmodified UTF-8 text."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Guide = Literal["s3-userguide", "s3-api"]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")]
Nonempty = Annotated[str, Field(min_length=1)]
Duration = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRecord(Record):
    source_id: Identifier
    guide: Guide
    revision: Revision
    original_filename: Literal["s3-userguide.txt", "s3-api.txt"]
    repository_url: HttpUrl
    sha256: Digest
    publisher_url: HttpUrl


class SourceManifest(Record):
    schema_version: Literal[1] = 1
    revision: Revision
    sources: list[SourceRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_sources(self) -> Self:
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("source_id values must be unique")
        if len({s.original_filename for s in self.sources}) != len(self.sources):
            raise ValueError("source filenames must be unique")
        if any(s.revision != self.revision for s in self.sources):
            raise ValueError("all sources must match the manifest revision")
        if any(s.original_filename != f"{s.guide}.txt" for s in self.sources):
            raise ValueError("source filename must match its guide")
        return self


class SectionSelection(Record):
    section_id: Identifier
    source_id: Identifier
    heading_path: list[Nonempty] = Field(min_length=1)
    raw_start: int = Field(ge=0)
    raw_end: int = Field(gt=0)
    raw_sha256: Digest
    source_scope: Literal["general-purpose"] = "general-purpose"
    selection_reason: Nonempty

    @model_validator(mode="after")
    def ordered_range(self) -> Self:
        if self.raw_end <= self.raw_start:
            raise ValueError("raw_end must be greater than raw_start")
        return self


class SubsetManifest(Record):
    schema_version: Literal[1] = 1
    corpus_id: Identifier
    source_revision: Revision
    offset_unit: Literal["unicode-codepoints-utf8-no-newline-normalization"]
    sections: list[SectionSelection] = Field(min_length=1)
    exclusions: list[Nonempty]

    @model_validator(mode="after")
    def unique_nonoverlapping_sections(self) -> Self:
        if len({s.section_id for s in self.sections}) != len(self.sections):
            raise ValueError("section_id values must be unique")
        ordered = sorted(self.sections, key=lambda s: (s.source_id, s.raw_start))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.source_id == current.source_id and current.raw_start < previous.raw_end:
                raise ValueError("selected raw ranges must not overlap within a source")
        return self


class SectionRecord(Record):
    section_id: Identifier
    source_id: Identifier
    corpus_id: Identifier
    heading_path: list[Nonempty] = Field(min_length=1)
    raw_start: int = Field(ge=0)
    raw_end: int = Field(gt=0)
    raw_sha256: Digest
    raw_text: Nonempty
    text: Nonempty
    source_scope: Literal["general-purpose"]
    cleaning_version: Nonempty
    provenance: SourceRecord

    @model_validator(mode="after")
    def consistent_lineage(self) -> Self:
        if self.raw_end - self.raw_start != len(self.raw_text):
            raise ValueError("raw range length must equal raw_text character count")
        if self.source_id != self.provenance.source_id:
            raise ValueError("source_id must match provenance")
        return self


class ChunkRecord(Record):
    chunk_id: Identifier
    section_id: Identifier
    corpus_id: Identifier
    original_text: Nonempty
    source_scope: Literal["general-purpose"]
    heading_path: list[Nonempty] = Field(min_length=1)
    section_raw_start: int = Field(ge=0)
    section_raw_end: int = Field(gt=0)
    chunker_version: Nonempty
    provenance: SourceRecord

    @model_validator(mode="after")
    def ordered_section_range(self) -> Self:
        if self.section_raw_end <= self.section_raw_start:
            raise ValueError("section raw range must be ordered")
        return self


class RetrievedHit(Record):
    chunk: ChunkRecord
    method: Literal["bm25", "vector", "hybrid"]
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class AnswerResult(Record):
    answer: str
    citation_ids: list[Identifier]
    status: Literal["answered", "insufficient_evidence", "failed"]
    evidence: list[ChunkRecord]
    timings_ms: dict[str, Duration]

    @model_validator(mode="after")
    def known_citations(self) -> Self:
        known = {chunk.chunk_id for chunk in self.evidence}
        if not set(self.citation_ids).issubset(known):
            raise ValueError("citation IDs must exist in the supplied evidence")
        return self
