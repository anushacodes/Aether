"""Compute exact retrieval metrics against a fixed, graded source-passage judgment set."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrievalMetrics:
    """Distinguish measured scores from inapplicable or incompletely judged rankings."""

    status: Literal["scored", "inapplicable", "incomplete_judgments"]
    hit_at_5: float | None = None
    hit_at_10: float | None = None
    recall_at_10: float | None = None
    mrr_at_10: float | None = None
    ndcg_at_10: float | None = None
    unjudged_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize metric status alongside values so missing scores stay explicit."""
        return asdict(self)


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def retrieval_metrics(
    ranked_passage_ids: list[str], judgments: dict[str, int], *, answerable: bool = True
) -> RetrievalMetrics:
    """Score unique source passages, never inventing relevance for unjudged evidence.

    Recall's denominator is every positive passage in the supplied judgment set,
    independent of which retriever produced this ranking. Grades are 0, 1, or 2.
    """
    if any(grade not in {0, 1, 2} for grade in judgments.values()):
        raise ValueError("Passage relevance grades must be 0, 1, or 2")
    if len(set(ranked_passage_ids)) != len(ranked_passage_ids):
        raise ValueError("Ranked passage IDs must be deduplicated before scoring")
    if not answerable:
        return RetrievalMetrics(status="inapplicable")
    positives = {passage_id for passage_id, grade in judgments.items() if grade > 0}
    if not positives:
        raise ValueError("Answerable questions require positive reviewed relevance judgments")
    unjudged = tuple(
        passage_id for passage_id in ranked_passage_ids[:10] if passage_id not in judgments
    )
    if unjudged:
        return RetrievalMetrics(status="incomplete_judgments", unjudged_ids=unjudged)
    top_ten = ranked_passage_ids[:10]
    relevant_ranks = [rank for rank, passage_id in enumerate(top_ten, 1) if passage_id in positives]
    ideal_grades = sorted(judgments.values(), reverse=True)[:10]
    actual_grades = [judgments[passage_id] for passage_id in top_ten]
    return RetrievalMetrics(
        status="scored",
        hit_at_5=float(any(rank <= 5 for rank in relevant_ranks)),
        hit_at_10=float(bool(relevant_ranks)),
        recall_at_10=len(set(top_ten) & positives) / len(positives),
        mrr_at_10=1 / relevant_ranks[0] if relevant_ranks else 0,
        ndcg_at_10=_dcg(actual_grades) / _dcg(ideal_grades),
    )
