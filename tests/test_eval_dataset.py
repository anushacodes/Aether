"""Keep unreviewed labels and leaked question groups out of release scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.eval.dataset import EvaluationDataset, load_dataset


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
