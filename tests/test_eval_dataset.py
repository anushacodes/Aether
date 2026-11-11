"""Keep unreviewed labels and leaked question groups out of release scoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.config import PipelineConfig
from rag.eval.answers import AnswerBinding, prepare_answers
from rag.eval.comparison import compare_retrieval_runs
from rag.eval.dataset import EvaluationDataset, load_dataset
from rag.eval.pooling import pool_candidates
from rag.eval.runs import run_retrieval_evaluation


def test_drafts_cannot_be_scored_and_equivalent_questions_cannot_cross_splits() -> None:
    path = Path("data/eval/pilot.json")
    with pytest.raises(ValueError, match="human review"):
        load_dataset(path)
    draft = load_dataset(path, allow_drafts=True)
    assert draft.review_status != "reviewed"
    values = draft.model_dump()
    duplicate = dict(values["questions"][0], question_id="leaked-copy", split="held_out")
    values["questions"].append(duplicate)
    with pytest.raises(ValueError, match="cannot cross splits"):
        EvaluationDataset.model_validate(values)


def test_candidate_review_does_not_assign_a_split_or_erase_prior_group_exposure(
    tmp_path: Path,
) -> None:
    values = load_dataset(Path("data/eval/pilot.json"), allow_drafts=True).model_dump()
    first = values["questions"][0]
    first["group_id"] = "exposed-family"
    candidate = dict(
        first,
        question_id="new-candidate",
        query="A different question in the same family",
        split="candidate",
    )
    values["questions"].append(candidate)
    bank = EvaluationDataset.model_validate(values)
    assert "new-candidate" not in {q.question_id for q in bank.questions_for_split("development")}
    assert bank.questions_for_split("held_out") == []
    with pytest.raises(ValueError, match="split assignment"):
        bank.questions_for_split("candidate")
    search = Mock()
    with pytest.raises(ValueError, match="split assignment"):
        run_retrieval_evaluation(
            search,
            PipelineConfig(),
            bank,
            [],
            [],
            "bm25",
            "candidate",
            tmp_path / "run",
            allow_drafts=True,
        )
    assert not search.mock_calls and not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="split assignment"):
        pool_candidates(bank, [tmp_path / "first"], "candidate")
    with pytest.raises(ValueError, match="split assignment"):
        compare_retrieval_runs(
            bank, [tmp_path / "first", tmp_path / "second"], "candidate", allow_drafts=True
        )
    with pytest.raises(ValueError, match="split assignment"):
        prepare_answers(
            bank,
            [AnswerBinding(question_id="new-candidate", answer_path=tmp_path / "missing.json")],
            [],
            max_samples=1,
        )
    # Even marking every reference reviewed does not freeze an unassigned question.
    reviewed = bank.model_dump()
    reviewed["review_status"] = "reviewed"
    for question in reviewed["questions"]:
        question["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="split assignment"):
        EvaluationDataset.model_validate(reviewed).require_reviewed()
    # A candidate between development and held-out entries cannot overwrite the guard.
    values["questions"].append(dict(candidate, question_id="leaked-held-out", split="held_out"))
    with pytest.raises(ValueError, match="cannot cross splits"):
        EvaluationDataset.model_validate(values)
