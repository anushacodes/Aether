"""Check retrieval scores against manual math and preserve missing-label outcomes."""

from __future__ import annotations

import math

import pytest

from rag.eval.metrics import retrieval_metrics


def test_retrieval_metrics_match_a_hand_calculated_ranking() -> None:
    result = retrieval_metrics(
        ["irrelevant", "partial", "direct"],
        {"irrelevant": 0, "partial": 1, "direct": 2, "missing": 2},
    )
    assert result.hit_at_5 == 1
    assert result.recall_at_10 == pytest.approx(2 / 3)
    assert result.mrr_at_10 == pytest.approx(1 / 2)
    actual = 1 / math.log2(3) + 3 / math.log2(4)
    ideal = 3 + 3 / math.log2(3) + 1 / math.log2(4)
    assert result.ndcg_at_10 == pytest.approx(actual / ideal)


def test_unjudged_or_unanswerable_questions_do_not_become_fabricated_scores() -> None:
    unjudged = retrieval_metrics(["new-evidence"], {"known-evidence": 2})
    assert unjudged.status == "incomplete_judgments"
    assert unjudged.unjudged_ids == ("new-evidence",)
    assert unjudged.ndcg_at_10 is None
    unanswerable = retrieval_metrics([], {}, answerable=False)
    assert unanswerable.status == "inapplicable"
    assert unanswerable.recall_at_10 is None
