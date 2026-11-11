"""Prepare offline judge samples from the exact evidence in explicitly saved answers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from rag.eval.dataset import EvaluationDataset
from rag.eval.passages import map_passages
from rag.models import (
    AnswerResult,
    ChunkRecord,
    Digest,
    Identifier,
    Record,
    SearchRequest,
    SectionRecord,
)


class AnswerBinding(Record):
    """Associate one saved answer artifact with a dataset question explicitly."""

    question_id: Identifier
    answer_path: Path


class JudgeSample(Record):
    """The four exact fields consumed by the selected Ragas collection metrics."""

    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


class PreparedAnswer(Record):
    """Preserve nonapplicable/failed generations without manufacturing judge scores."""

    question_id: Identifier
    answer_path: Path
    review_status: Literal["draft", "reviewed"]
    status: Literal[
        "ready",
        "unanswerable",
        "unexpected_answer",
        "insufficient_evidence",
        "generation_failed",
        "missing_context",
        "error",
    ]
    artifact_sha256: Digest | None = None
    pipeline_fingerprint: Digest | None = None
    context_passages: list[ChunkRecord] = Field(default_factory=list)
    sample: JudgeSample | None = None
    error: str | None = None


def prepare_answers(
    dataset: EvaluationDataset,
    bindings: list[AnswerBinding],
    sections: list[SectionRecord],
    *,
    max_samples: int,
) -> list[PreparedAnswer]:
    """Validate saved provenance/scope before any optional dependency or provider is used."""
    if not bindings or len(bindings) > max_samples:
        raise ValueError(f"Provide between one and {max_samples} saved answers; none are truncated")
    questions = {question.question_id: question for question in dataset.questions}
    identifiers = [binding.question_id for binding in bindings]
    if len(set(identifiers)) != len(identifiers) or not set(identifiers).issubset(questions):
        raise ValueError("Answer bindings must reference distinct known dataset questions")
    rows: list[PreparedAnswer] = []
    for binding in bindings:
        question = questions[binding.question_id]
        base = {
            "question_id": binding.question_id,
            "answer_path": binding.answer_path,
            "review_status": question.review_status,
        }
        artifact_hash = None
        try:
            raw = binding.answer_path.read_bytes()
            artifact_hash = hashlib.sha256(raw).hexdigest()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Saved answer must be a JSON object")
            request = SearchRequest.model_validate(value["request"])
            result = AnswerResult.model_validate(value["result"])
            if not isinstance(value.get("context_passages"), list):
                raise ValueError("Saved answer must include its exact selected context")
            context = [ChunkRecord.model_validate(chunk) for chunk in value["context_passages"]]
            if context != result.evidence:
                raise ValueError("Saved context and answer evidence differ")
            if (
                request.query != question.query.strip()
                or set(request.scope.permitted_guides) != set(question.permitted_guides)
                or request.scope.source_scope != question.source_scope
                or request.scope.corpus_id not in (None, dataset.corpus_id)
                or any(
                    chunk.corpus_id != dataset.corpus_id
                    or chunk.provenance.guide not in question.permitted_guides
                    or chunk.source_scope != question.source_scope
                    for chunk in context
                )
            ):
                raise ValueError(
                    "Saved answer question, corpus, or guide scope differs from labels"
                )
            map_passages(dataset.passages, sections, context)
            sample = None
            if result.status == "failed":
                status = "generation_failed"
            elif question.expected_answerability == "unanswerable":
                status = "unexpected_answer" if result.status == "answered" else "unanswerable"
            elif result.status != "answered":
                status = "insufficient_evidence"
            elif not context or not result.answer.strip():
                status = "missing_context"
            else:
                status = "ready"
                sample = JudgeSample(
                    user_input=request.query,
                    response=result.answer,
                    retrieved_contexts=[chunk.original_text for chunk in context],
                    reference=question.reference_answer,
                )
            rows.append(
                PreparedAnswer(
                    **base,
                    status=status,
                    artifact_sha256=artifact_hash,
                    pipeline_fingerprint=value["pipeline_fingerprint"],
                    context_passages=context,
                    sample=sample,
                )
            )
        except (OSError, KeyError, ValueError) as error:
            rows.append(
                PreparedAnswer(
                    **base, status="error", artifact_sha256=artifact_hash, error=str(error)
                )
            )
    return rows
