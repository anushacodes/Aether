"""Protect selected-evidence integrity and keep unsupported answers out of judge averages."""

from __future__ import annotations

import json
from pathlib import Path

from rag.artifacts import save_answer_run
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.eval.answers import AnswerBinding, prepare_answers
from rag.eval.dataset import load_dataset
from rag.models import AnswerResult, SearchRequest, SearchScope, SectionRecord


def test_preparation_uses_exact_selected_context_and_refuses_altered_artifacts(
    tmp_path: Path,
) -> None:
    config = PipelineConfig()
    section = SectionRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    )
    chunk = chunk_section(section, config.chunking)[0]
    dataset = load_dataset(Path("data/eval/pilot.json"), allow_drafts=True)
    question = dataset.questions[0]
    request = SearchRequest(
        query=question.query,
        scope=SearchScope(permitted_guides=question.permitted_guides, corpus_id=dataset.corpus_id),
    )
    result = AnswerResult(
        answer=f"A retained version. [{chunk.chunk_id}]",
        citation_ids=[chunk.chunk_id],
        status="answered",
        evidence=[chunk],
        timings_ms={},
    )
    output = tmp_path / "answer"
    save_answer_run(output, request, result, config)
    binding = AnswerBinding(question_id=question.question_id, answer_path=output / "answer.json")
    prepared = prepare_answers(dataset, [binding], [section], max_samples=5)[0]
    assert prepared.status == "ready" and prepared.review_status == "draft"
    assert prepared.sample.retrieved_contexts == [chunk.original_text]
    assert prepared.sample.response == result.answer
    assert prepared.sample.reference == question.reference_answer
    saved = json.loads(binding.answer_path.read_text(encoding="utf-8"))
    saved["context_passages"] = []
    binding.answer_path.write_text(json.dumps(saved), encoding="utf-8")
    altered = prepare_answers(dataset, [binding], [section], max_samples=5)[0]
    assert altered.status == "error" and "evidence differ" in altered.error
    assert altered.sample is None


def test_unanswerable_generation_and_abstention_remain_distinct_without_judge_samples(
    tmp_path: Path,
) -> None:
    config = PipelineConfig()
    section = SectionRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    )
    chunk = chunk_section(section, config.chunking)[0]
    dataset = load_dataset(Path("data/eval/pilot.json"), allow_drafts=True)
    question = dataset.questions[0].model_copy(
        update={
            "query": "Is my live bucket public?",
            "expected_answerability": "unanswerable",
            "reference_passage_ids": [],
            "unanswerable_reason": "No live account state",
        }
    )
    dataset.questions = [question]
    request = SearchRequest(
        query=question.query,
        scope=SearchScope(permitted_guides=question.permitted_guides),
    )
    bindings = []
    for name, result in (
        (
            "abstention",
            AnswerResult(
                answer="Insufficient evidence",
                citation_ids=[],
                status="insufficient_evidence",
                evidence=[],
                timings_ms={},
            ),
        ),
        (
            "unexpected",
            AnswerResult(
                answer=f"Invented live state [{chunk.chunk_id}]",
                citation_ids=[chunk.chunk_id],
                status="answered",
                evidence=[chunk],
                timings_ms={},
            ),
        ),
    ):
        output = tmp_path / name
        save_answer_run(output, request, result, config)
        bindings.append(
            AnswerBinding(question_id=question.question_id, answer_path=output / "answer.json")
        )
    abstention = prepare_answers(dataset, [bindings[0]], [section], max_samples=5)[0]
    unexpected = prepare_answers(dataset, [bindings[1]], [section], max_samples=5)[0]
    assert (abstention.status, unexpected.status) == ("unanswerable", "unexpected_answer")
    assert abstention.sample is unexpected.sample is None
