"""Capture every evaluation outcome before aggregating quality, failures, and timings."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from botocore.exceptions import BotoCoreError, ClientError
from opensearchpy import OpenSearch, TransportError

from rag.config import PipelineConfig
from rag.embeddings import Embedder
from rag.eval.dataset import EvaluationDataset, EvaluationQuestion
from rag.eval.metrics import retrieval_metrics
from rag.eval.passages import PassageMapping, map_passages
from rag.models import ChunkRecord, SearchScope, SectionRecord
from rag.pipeline import retrieve


class EvaluationRow(TypedDict):
    """The persisted per-question result, including explicit failed or unscored outcomes."""

    question_id: str
    query: str
    category: str
    mode: str
    status: str
    hits: NotRequired[list[dict[str, object]]]
    timings_ms: NotRequired[dict[str, float]]
    latency_ms: NotRequired[float]
    metrics: NotRequired[dict[str, object]]
    metric_status: NotRequired[str]
    covered_passages: NotRequired[list[str]]
    error_type: NotRequired[str]
    error: NotRequired[str]


def _score(
    question: EvaluationQuestion, ranked_ids: list[str], mapping: PassageMapping, reviewed: bool
) -> dict[str, object]:
    if not reviewed:
        return {
            "status": "draft_labels",
            "reason": "Human review is pending; no quality score computed",
        }
    if question.expected_answerability == "unanswerable":
        return retrieval_metrics([], {}, answerable=False).to_dict()
    grades = question.relevance_grades
    positives = {identifier for identifier, grade in grades.items() if grade > 0}
    if not positives:
        return {"status": "incomplete_judgments", "reason": "No graded positive passages"}
    judgments: dict[str, int] = {}
    for chunk_id, complete in mapping.complete.items():
        touched = complete | mapping.partial[chunk_id]
        judged = touched & grades.keys()
        if judged:
            judgments[chunk_id] = max(
                grades[passage] if passage in complete else 0 for passage in judged
            )
    if not any(judgments.values()):
        return {
            "status": "unmapped_reference",
            "reason": "No chunk fully covers a positive labeled span",
        }
    metrics = retrieval_metrics(ranked_ids, judgments).to_dict()
    if metrics["status"] == "scored":
        # Recall counts fixed source passages, not the number of overlapping chunk windows.
        metrics["recall_at_10"] = len(mapping.covered_by(ranked_ids[:10]) & positives) / len(
            positives
        )
    return metrics


def _summary(rows: list[EvaluationRow]) -> dict[str, object]:
    statuses = Counter(str(row["status"]) for row in rows)
    scored = [
        row["metrics"] for row in rows if row.get("metric_status") == "scored" and "metrics" in row
    ]
    metric_names = ("hit_at_5", "hit_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_10")
    means = {}
    for name in metric_names:
        values = [
            float(value) for metric in scored if isinstance(value := metric.get(name), (int, float))
        ]
        means[name] = statistics.mean(values) if values else None
    latencies = [row["latency_ms"] for row in rows if row["status"] == "ok" and "latency_ms" in row]
    return {
        "questions": len(rows),
        "outcomes": dict(statuses),
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
        "metrics": means,
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "categories": {
            category: dict(
                Counter(str(row["status"]) for row in rows if row["category"] == category)
            )
            for category in sorted({str(row["category"]) for row in rows})
        },
    }


def run_retrieval_evaluation(
    client: OpenSearch,
    config: PipelineConfig,
    dataset: EvaluationDataset,
    sections: list[SectionRecord],
    chunks: list[ChunkRecord],
    mode: Literal["bm25", "vector", "hybrid"],
    split: Literal["development", "held_out"],
    output: Path,
    *,
    embedder: Embedder | None = None,
    allow_drafts: bool = False,
) -> dict[str, object]:
    """Save per-question JSONL and a complete summary; drafts remain explicitly unscored."""
    if not allow_drafts or split == "held_out":
        dataset.require_reviewed()
    questions = [question for question in dataset.questions if question.split == split]
    if not questions:
        raise ValueError(f"Dataset contains no questions in split {split!r}")
    if any(chunk.corpus_id != dataset.corpus_id for chunk in chunks):
        raise ValueError("Evaluation dataset and corpus IDs differ")
    mapping = map_passages(dataset.passages, sections, chunks)
    reviewed = dataset.review_status == "reviewed" and all(
        question.review_status == "reviewed" for question in questions
    )
    identity = {
        "dataset_sha256": dataset.fingerprint(),
        "mode": mode,
        "split": split,
        "chunk_ids": sorted(chunk.chunk_id for chunk in chunks),
        "chunking": config.chunking.model_dump(mode="json"),
        "models": config.models.model_dump(mode="json"),
        "retrieval": config.retrieval.model_dump(mode="json"),
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    rows: list[EvaluationRow] = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for question in questions:
            row: EvaluationRow = {
                "question_id": question.question_id,
                "query": question.query,
                "category": question.category,
                "mode": mode,
                "status": "ok",
            }
            try:
                scope = SearchScope(
                    permitted_guides=question.permitted_guides,
                    source_scope=question.source_scope,
                    corpus_id=dataset.corpus_id,
                )
                result = retrieve(client, config, question.query, scope, mode, embedder)
                identifiers = [hit.chunk.chunk_id for hit in result.hits]
                if any(identifier not in mapping.complete for identifier in identifiers):
                    raise ValueError("Retrieved index differs from the recorded chunk snapshot")
                metrics = _score(question, identifiers, mapping, reviewed)
                row.update(
                    {
                        "hits": [hit.model_dump(mode="json") for hit in result.hits],
                        "timings_ms": result.timings_ms,
                        "latency_ms": result.timings_ms["retrieval_total"],
                        "metrics": metrics,
                        "metric_status": str(metrics["status"]),
                        "covered_passages": sorted(mapping.covered_by(identifiers[:10])),
                    }
                )
            except (TransportError, BotoCoreError, ClientError, ValueError) as error:
                row.update(
                    {
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "metric_status": "error",
                    }
                )
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
    summary = _summary(rows) | {
        "run_fingerprint": run_fingerprint,
        "inputs": identity,
        "finished_at": datetime.now(UTC).isoformat(),
        "reviewed_labels": reviewed,
        "cache_conditions": {"requests": embedder.request_count, "cache_hits": embedder.cache_hits}
        if embedder
        else {"model_requests": 0},
        "metric_units": {
            "hit_mrr_ndcg": "chunks mapped to fully covered graded source spans",
            "recall": "distinct positive source passages; full-span coverage required",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
