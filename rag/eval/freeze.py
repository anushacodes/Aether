"""Archive reviewed labels and pinned evidence without assigning or certifying them.

A manifest is written last. An interrupted archive without that manifest is invalid;
existing destinations are never overwritten. Review records preserve a human claim,
not independent proof of its truth or complete semantic duplicate detection.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from rag.corpus import CLEANING_VERSION, clean_section, extract_sections
from rag.eval.dataset import EvaluationDataset
from rag.eval.passages import map_passages
from rag.models import (
    Digest,
    Identifier,
    Nonempty,
    Record,
    Revision,
    SectionRecord,
    SourceManifest,
    SubsetManifest,
)

ARCHIVE_FILES = frozenset(
    {"dataset.json", "review.json", "sources.json", "subset.json", "sections.jsonl"}
)
Split = Literal["development", "held_out"]


class DatasetReview(Record):
    """Bind an explicitly supplied review/grouping claim to exact semantic labels."""

    schema_version: Literal[1] = 1
    dataset_sha256: Digest
    reviewer: Nonempty
    reviewed_at: datetime
    semantic_groups_reviewed: Literal[True]
    note: Nonempty

    @model_validator(mode="after")
    def explicit_review(self) -> Self:
        """Require a timezone and meaningful reviewer text for retained provenance."""
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("Review timestamp must include a timezone")
        if not self.reviewer.strip() or not self.note.strip():
            raise ValueError("Review record requires a reviewer and explanatory note")
        return self


class DatasetFreeze(Record):
    """Identify exact archive bytes and their reviewed split/group assignment."""

    schema_version: Literal[1] = 1
    created_at: datetime
    dataset_id: Identifier
    corpus_id: Identifier
    dataset_sha256: Digest
    source_revision: Revision
    file_sha256: dict[str, Digest]
    question_ids: dict[Split, list[Identifier]]
    categories: dict[Split, dict[str, int]]
    groups: dict[Identifier, Split]


def _validate_labels(
    dataset: EvaluationDataset, review: DatasetReview, sections: list[SectionRecord]
) -> None:
    """Check assignment, declared groups, original evidence, and authoritative scopes."""
    dataset.require_reviewed()
    if review.dataset_sha256 != dataset.fingerprint():
        raise ValueError("Review record does not match the exact dataset fingerprint")
    if any(not dataset.questions_for_split(split) for split in ("development", "held_out")):
        raise ValueError("A freeze requires nonempty development and held-out splits")
    if any(not question.group_id for question in dataset.questions):
        raise ValueError("Every frozen question requires a reviewed group ID")
    if any(section.corpus_id != dataset.corpus_id for section in sections):
        raise ValueError("Frozen sections and dataset corpus IDs differ")
    if len({(s.source_id, s.section_id) for s in sections}) != len(sections):
        raise ValueError("Frozen section identities must be unique")
    if any(
        section.cleaning_version != CLEANING_VERSION
        or section.text != clean_section(section.raw_text)
        for section in sections
    ):
        raise ValueError("Frozen cleaned sections differ from the recorded cleaning implementation")
    # Labels remain independent of any chosen chunk configuration.
    map_passages(dataset.passages, sections, [])
    by_section = {(section.source_id, section.section_id): section for section in sections}
    passages = {passage.passage_id: passage for passage in dataset.passages}
    for question in dataset.questions:
        referenced = set(question.reference_passage_ids) | set(question.relevance_grades)
        for group in question.required_evidence_groups:
            if not group:
                raise ValueError("Required evidence groups cannot be empty")
            referenced.update(group)
        for identifier in referenced:
            passage = passages[identifier]
            section = by_section[passage.source_id, passage.section_id]
            if (
                section.provenance.guide not in question.permitted_guides
                or section.source_scope != question.source_scope
            ):
                raise ValueError(
                    f"Frozen passage is outside question scope: {question.question_id}"
                )


def _manifest(
    dataset: EvaluationDataset,
    sources: SourceManifest,
    files: dict[str, bytes],
    created_at: datetime,
) -> DatasetFreeze:
    """Derive inventory from retained labels rather than trusting supplied counts."""
    splits: tuple[Split, ...] = ("development", "held_out")
    return DatasetFreeze(
        created_at=created_at,
        dataset_id=dataset.dataset_id,
        corpus_id=dataset.corpus_id,
        dataset_sha256=dataset.fingerprint(),
        source_revision=sources.revision,
        file_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
        question_ids={
            split: [question.question_id for question in dataset.questions_for_split(split)]
            for split in splits
        },
        categories={
            split: dict(Counter(q.category for q in dataset.questions_for_split(split)))
            for split in splits
        },
        groups={q.group_id: q.split for q in dataset.questions if q.group_id},
    )


def freeze_dataset(
    dataset_path: Path,
    review_path: Path,
    sources_path: Path,
    subset_path: Path,
    raw_dir: Path,
    output: Path,
) -> DatasetFreeze:
    """Verify pinned raw sources and preserve exact reviewed inputs with a final manifest."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Freeze output already exists: {output}")
    dataset_bytes = dataset_path.read_bytes()
    dataset = EvaluationDataset.model_validate_json(dataset_bytes)
    # Fail closed on the actual unreviewed bank before reading raw sources or writing output.
    dataset.require_reviewed()
    files = {
        "dataset.json": dataset_bytes,
        "review.json": review_path.read_bytes(),
        "sources.json": sources_path.read_bytes(),
        "subset.json": subset_path.read_bytes(),
    }
    review = DatasetReview.model_validate_json(files["review.json"])
    sources = SourceManifest.model_validate_json(files["sources.json"])
    subset = SubsetManifest.model_validate_json(files["subset.json"])
    if subset.corpus_id != dataset.corpus_id:
        raise ValueError("Frozen subset and dataset corpus IDs differ")
    sections = extract_sections(sources, subset, raw_dir)
    _validate_labels(dataset, review, sections)
    files["sections.jsonl"] = "".join(
        section.model_dump_json() + "\n" for section in sections
    ).encode()
    manifest = _manifest(dataset, sources, files, datetime.now(UTC))
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def verify_dataset_freeze(directory: Path) -> DatasetFreeze:
    """Reject altered bytes, identities, assignments, or incomplete archives offline."""
    manifest = DatasetFreeze.model_validate_json((directory / "manifest.json").read_bytes())
    if set(manifest.file_sha256) != ARCHIVE_FILES:
        raise ValueError("Frozen archive must identify exactly its five retained input files")
    if manifest.created_at.utcoffset() is None:
        raise ValueError("Freeze timestamp must include a timezone")
    files = {name: (directory / name).read_bytes() for name in sorted(ARCHIVE_FILES)}
    if any(
        hashlib.sha256(raw).hexdigest() != manifest.file_sha256[name] for name, raw in files.items()
    ):
        raise ValueError("Frozen archive file hash mismatch")
    dataset = EvaluationDataset.model_validate_json(files["dataset.json"])
    review = DatasetReview.model_validate_json(files["review.json"])
    sources = SourceManifest.model_validate_json(files["sources.json"])
    subset = SubsetManifest.model_validate_json(files["subset.json"])
    sections = [
        SectionRecord.model_validate_json(line)
        for line in files["sections.jsonl"].splitlines()
        if line.strip()
    ]
    if subset.source_revision != sources.revision or subset.corpus_id != dataset.corpus_id:
        raise ValueError("Frozen source/subset identity mismatch")
    selections = {(s.source_id, s.section_id): s for s in subset.sections}
    provenance = {source.source_id: source for source in sources.sources}
    if len(sections) != len(selections):
        raise ValueError("Frozen section inventory differs from its subset")
    for section in sections:
        selection = selections.get((section.source_id, section.section_id))
        if (
            selection is None
            or section.provenance != provenance.get(section.source_id)
            or any(
                getattr(section, field) != getattr(selection, field)
                for field in ("raw_start", "raw_end", "raw_sha256", "heading_path", "source_scope")
            )
        ):
            raise ValueError("Frozen section provenance differs from source/subset")
    _validate_labels(dataset, review, sections)
    if _manifest(dataset, sources, files, manifest.created_at) != manifest:
        raise ValueError("Frozen manifest inventory does not match retained labels")
    return manifest
