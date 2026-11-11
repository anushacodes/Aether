"""Validate independent evaluation labels without promoting drafts to benchmark truth."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from rag.models import Digest, Guide, Identifier, Record


class EvaluationPassage(Record):
    """A supporting span in cleaned source text, anchored to its original raw section."""

    passage_id: Identifier
    source_id: Identifier
    section_id: Identifier
    raw_section_sha256: Digest
    cleaning_version: str
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)
    text_sha256: Digest

    @model_validator(mode="after")
    def ordered_span(self) -> Self:
        """Reject empty or reversed labeled spans."""
        if self.text_end <= self.text_start:
            raise ValueError("Labeled passage end must follow its start")
        return self


class EvaluationQuestion(Record):
    """A question, authoritative scope, and separately reviewed source-passage labels."""

    question_id: Identifier
    query: str = Field(min_length=1, max_length=2000)
    category: Literal[
        "direct_fact",
        "troubleshooting",
        "multi_section",
        "comparison",
        "api_parameter",
        "unanswerable",
    ]
    permitted_guides: list[Guide] = Field(min_length=1)
    source_scope: Literal["general-purpose"]
    reference_answer: str
    reference_passage_ids: list[Identifier]
    expected_answerability: Literal["answerable", "unanswerable"]
    split: Literal["development", "held_out"]
    group_id: Identifier | None = None
    review_status: Literal["draft", "reviewed"] = "draft"
    unanswerable_reason: str | None = None
    relevance_grades: dict[Identifier, Literal[0, 1, 2]] = Field(default_factory=dict)
    required_evidence_groups: list[list[Identifier]] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_answerability(self) -> Self:
        """Keep answerable references and documented missing evidence distinct."""
        if self.expected_answerability == "answerable":
            if not self.reference_passage_ids or not self.reference_answer.strip():
                raise ValueError(
                    "Answerable questions require source passages and a reference answer"
                )
        elif self.reference_passage_ids or not self.unanswerable_reason:
            raise ValueError("Unanswerable questions require a reason and no supporting passages")
        return self


class EvaluationDataset(Record):
    """Version label provenance and prevent equivalent questions leaking across splits."""

    dataset_id: Identifier
    corpus_id: Identifier
    review_status: Literal["draft", "source_checked_by_assistant_pending_human_review", "reviewed"]
    note: str
    questions: list[EvaluationQuestion] = Field(min_length=1)
    passages: list[EvaluationPassage] = Field(default_factory=list)

    @model_validator(mode="after")
    def isolated_splits(self) -> Self:
        """Reject duplicate IDs and cross-split question/group reuse."""
        identifiers: set[str] = set()
        if len({passage.passage_id for passage in self.passages}) != len(self.passages):
            raise ValueError("Evaluation passage IDs must be unique")
        groups: dict[str, str] = {}
        texts: dict[str, str] = {}
        for question in self.questions:
            if question.question_id in identifiers:
                raise ValueError(f"Duplicate evaluation question ID: {question.question_id}")
            identifiers.add(question.question_id)
            normalized = " ".join(question.query.casefold().split())
            for key, values in ((normalized, texts), (question.group_id, groups)):
                if key is None:
                    continue
                if key in values and values[key] != question.split:
                    raise ValueError("Equivalent questions/evidence groups cannot cross splits")
                values[key] = question.split
        return self

    def require_reviewed(self) -> None:
        """Refuse benchmark scoring until the dataset and every question have human review."""
        if self.review_status != "reviewed" or any(
            question.review_status != "reviewed" for question in self.questions
        ):
            raise ValueError(
                "Evaluation labels require human review; drafts are not benchmark truth"
            )

        known_passages = {passage.passage_id for passage in self.passages}
        for question in self.questions:
            referenced = set(question.reference_passage_ids) | set(question.relevance_grades)
            referenced |= {
                identifier for group in question.required_evidence_groups for identifier in group
            }
            if not referenced.issubset(known_passages):
                raise ValueError("Reviewed labels must reference fixed source passages")

    def fingerprint(self) -> str:
        """Identify the exact labels, rubric fields, and split assignment used by a run."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


def load_dataset(path: Path, *, allow_drafts: bool = False) -> EvaluationDataset:
    """Read labels only as evaluation inputs; this path is never accepted for source ingestion."""
    dataset = EvaluationDataset.model_validate_json(path.read_bytes())
    if not allow_drafts:
        dataset.require_reviewed()
    return dataset
