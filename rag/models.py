"""Shared records. Raw offsets always address decoded, unmodified UTF-8 text."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

Guide = Literal["s3-userguide", "s3-api"]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")]
Nonempty = Annotated[str, Field(min_length=1)]
Duration = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Record(BaseModel):
    """Reject unrecognized fields at serialized-data boundaries."""

    model_config = ConfigDict(extra="forbid")


class SourceRecord(Record):
    """Identify the pinned source snapshot and its separate publisher reference."""

    source_id: Identifier
    guide: Guide
    revision: Revision
    original_filename: Literal["s3-userguide.txt", "s3-api.txt"]
    repository_url: HttpUrl
    sha256: Digest
    publisher_url: HttpUrl


class SourceManifest(Record):
    """List unique, compatible source snapshots for one repository revision."""

    schema_version: Literal[1] = 1
    revision: Revision
    sources: list[SourceRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_sources(self) -> Self:
        """Require one identity and filename per source at the declared revision."""
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
    """Select a source range using raw character offsets and a range hash."""

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
        """Reject empty or reversed raw ranges."""
        if self.raw_end <= self.raw_start:
            raise ValueError("raw_end must be greater than raw_start")
        return self


class SubsetManifest(Record):
    """Define a reproducible corpus subset without overlapping source selections."""

    schema_version: Literal[1] = 1
    corpus_id: Identifier
    source_revision: Revision
    offset_unit: Literal["unicode-codepoints-utf8-no-newline-normalization"]
    sections: list[SectionSelection]
    exclusions: list[Nonempty]

    @model_validator(mode="after")
    def unique_nonoverlapping_sections(self) -> Self:
        """Validate selection identities and reject ambiguous overlapping evidence."""
        if len({s.section_id for s in self.sections}) != len(self.sections):
            raise ValueError("section_id values must be unique")
        ordered = sorted(self.sections, key=lambda s: (s.source_id, s.raw_start))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.source_id == current.source_id and current.raw_start < previous.raw_end:
                raise ValueError("selected raw ranges must not overlap within a source")
        return self


class SectionRecord(Record):
    """Preserve selected source text beside its cleaned extraction result."""

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
        """Verify the selected raw text against its declared range, hash, and source."""
        if self.raw_end - self.raw_start != len(self.raw_text):
            raise ValueError("raw range length must equal raw_text character count")
        if self.source_id != self.provenance.source_id:
            raise ValueError("source_id must match provenance")
        if hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest() != self.raw_sha256:
            raise ValueError("raw_sha256 must match raw_text")
        return self


class ExampleContinuation(Record):
    """Identify the ordered fragments of one oversized source example."""

    block_id: Identifier
    part_index: int = Field(ge=1)
    part_count: int = Field(ge=2)

    @model_validator(mode="after")
    def ordered_part(self) -> Self:
        """Reject fragment positions outside the declared continuation count."""
        if self.part_index > self.part_count:
            raise ValueError("part_index must not exceed part_count")
        return self


class ChunkRecord(Record):
    """Carry exact evidence text, optional cleaned spans, and source provenance."""

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
    text_start: int | None = Field(default=None, ge=0)
    text_end: int | None = Field(default=None, gt=0)
    content_sha256: Digest | None = None
    estimated_tokens: int | None = Field(default=None, gt=0)
    continuations: list[ExampleContinuation] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_section_range(self) -> Self:
        """Check the raw parent range and any supplied cleaned-text span or content hash."""
        if self.section_raw_end <= self.section_raw_start:
            raise ValueError("section raw range must be ordered")
        if (self.text_start is None) != (self.text_end is None):
            raise ValueError("cleaned text offsets must be supplied together")
        if self.text_start is not None and self.text_end is not None:
            if self.text_end - self.text_start != len(self.original_text):
                raise ValueError(
                    "cleaned text range length must equal original_text character count"
                )
        if self.content_sha256 is not None:
            content_hash = hashlib.sha256(self.original_text.encode("utf-8")).hexdigest()
            if self.content_sha256 != content_hash:
                raise ValueError("content_sha256 must match original_text")
        return self


class RetrievedHit(Record):
    """Normalize an individual retriever's rank and method-specific score."""

    chunk: ChunkRecord
    method: Literal["bm25", "vector", "hybrid"]
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    rerank_score: float | None = Field(default=None, allow_inf_nan=False)


class RerankingReport(Record):
    """Preserve cost-relevant stage outcomes without treating relevance as confidence."""

    status: Literal["applied", "fallback", "skipped", "failed"]
    model_arn: str | None
    input_count: int = Field(ge=0)
    candidate_ids: list[Identifier]
    input_characters: int = Field(ge=0)
    attempted_calls: int = Field(ge=0, le=1)
    reason: str | None = None
    usage_status: Literal["not_reported", "not_called"]
    model_cost_usd: None = None


class RetrievalResult(Record):
    """Expose normalized candidates and measured stage timings to CLI, API, and evals."""

    hits: list[RetrievedHit]
    mode: Literal["bm25", "vector", "hybrid"]
    timings_ms: dict[str, Duration]
    reranking: RerankingReport | None = None
    snapshot_index: Identifier | None = None


class SearchScope(Record):
    """Authoritative restrictions passed separately from untrusted question text."""

    permitted_guides: list[Guide] = Field(
        default_factory=lambda: ["s3-userguide", "s3-api"], min_length=1
    )
    source_scope: Literal["general-purpose"] = "general-purpose"
    corpus_id: Identifier | None = None


class SearchRequest(Record):
    """Validate a bounded question and preserve source restrictions across every entry point."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    mode: Literal["bm25", "vector", "hybrid"] = "bm25"
    scope: SearchScope = Field(default_factory=SearchScope)
    workflow: bool = False


class ComparisonRequest(Record):
    """Compare exactly two supported modes with one unchanged question and scope."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    scope: SearchScope = Field(default_factory=SearchScope)
    modes: list[Literal["bm25", "vector", "hybrid"]] = Field(
        default_factory=lambda: ["bm25", "hybrid"], min_length=2, max_length=2
    )

    @model_validator(mode="after")
    def distinct_modes(self) -> Self:
        """Reject duplicate work before search admission or model setup."""
        if len(set(self.modes)) != 2:
            raise ValueError("Choose two distinct retrieval modes")
        return self


class ComparisonArm(Record):
    """Keep a failed mode alongside successful evidence and its actual attempted work."""

    mode: Literal["bm25", "vector", "hybrid"]
    status: Literal["ok", "error"]
    result: RetrievalResult | None = None
    error_code: str | None = None
    elapsed_ms: Duration
    model_calls: int = Field(ge=0)
    embedding_cache_hits: int = Field(ge=0)
    known_embedding_input_tokens: int = Field(ge=0)
    model_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    reranking: RerankingReport | None = None

    @model_validator(mode="after")
    def consistent_outcome(self) -> Self:
        """A failed mode cannot masquerade as successful retrieval or known paid cost."""
        if self.status == "ok":
            if self.result is None or self.result.mode != self.mode or self.error_code is not None:
                raise ValueError("Successful comparison requires matching retrieval results")
        elif self.result is not None or not self.error_code:
            raise ValueError("Failed comparison requires an error code without results")
        if self.model_cost_usd is not None and (self.model_calls or self.model_cost_usd != 0):
            raise ValueError("Only zero-call comparison cost is known without configured prices")
        return self


class ComparisonResult(Record):
    """Describe live observations without inferring relevance or a winning configuration."""

    query: Nonempty
    scope: SearchScope
    snapshot_index: Identifier
    arms: list[ComparisonArm] = Field(min_length=2, max_length=2)
    total_ms: Duration
    retrieval_settings: dict[str, object]
    reranking_settings: dict[str, object]
    execution: Literal["sequential_shared_cache"] = "sequential_shared_cache"


class WorkflowStep(Record):
    """A bounded assessment decision, not an assertion that every claim is supported."""

    retrieval_pass: int = Field(ge=1, le=2)
    query: Nonempty
    decision: Literal["answer", "retry", "insufficient"]
    missing_evidence: str
    context_chunk_ids: list[Identifier] = Field(default_factory=list)
    reranking: RerankingReport | None = None


class WorkflowReport(Record):
    """Expose the concrete snapshot, bounded work, and structured routing decisions."""

    snapshot_index: str
    retrieval_passes: int = Field(ge=0, le=2)
    model_calls: int = Field(ge=0, le=5)
    assessment_prompt_version: str
    steps: list[WorkflowStep]


class AnswerResult(Record):
    """Represent an answer outcome with inspectable evidence and bounded timings."""

    answer: str
    citation_ids: list[Identifier]
    status: Literal["answered", "insufficient_evidence", "failed"]
    evidence: list[ChunkRecord]
    timings_ms: dict[str, Duration]
    usage_tokens: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    workflow: WorkflowReport | None = None
    reranking: RerankingReport | None = None
    snapshot_index: Identifier | None = None

    @model_validator(mode="after")
    def known_citations(self) -> Self:
        """Validate citation membership without claiming to verify factual support."""
        known = {chunk.chunk_id for chunk in self.evidence}
        if not set(self.citation_ids).issubset(known):
            raise ValueError("citation IDs must exist in the supplied evidence")
        return self
