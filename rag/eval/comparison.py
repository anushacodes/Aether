"""Compare saved retrieval observations on aligned questions without running a model."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from rag.eval.dataset import EvaluationDataset, EvaluationQuestion
from rag.eval.pooling import pool_candidates
from rag.models import SearchScope

METRICS = ("hit_at_5", "hit_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_10")
CONFIGURATION_FIELDS = (
    "mode",
    "chunking",
    "models",
    "retrieval",
    "reranking",
    "enrichment_fingerprint",
)


def _number(value: object, field: str, *, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid saved {field}")
    return float(value)


def _latencies(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95_nearest_rank": ordered[math.ceil(0.95 * len(values)) - 1] if values else None,
    }


def _observation(
    question: EvaluationQuestion,
    row: dict[str, Any] | None,
    inputs: dict[str, Any],
    configuration_id: str | None,
    reviewed: bool,
) -> dict[str, Any]:
    """Validate the saved boundary, keeping absence and failed work explicit."""
    if row is None:
        return {"status": "missing", "metrics": None, "latency_ms": None}
    expected_scope = SearchScope(
        permitted_guides=question.permitted_guides,
        source_scope=question.source_scope,
        corpus_id=inputs["corpus_id"],
    )
    if row.get("scope") != expected_scope.model_dump(mode="json"):
        raise ValueError("Saved question scope differs from the dataset")
    if row.get("review_status") != question.review_status:
        raise ValueError("Saved question review status differs from the dataset")
    if row.get("expected_answerability") != question.expected_answerability:
        raise ValueError("Saved question answerability differs from the dataset")
    if (inputs["schema_version"] == 3 or row.get("configuration_id") is not None) and row.get(
        "configuration_id"
    ) != configuration_id:
        raise ValueError("Saved row and summary configuration IDs differ")
    if row["status"] == "error":
        return {
            "status": "error",
            "error_type": row.get("error_type"),
            "error": row.get("error"),
            "metrics": None,
            "latency_ms": None,
            "reranking": row.get("reranking"),
        }
    if row.get("index") != inputs.get("index") or not row.get("index"):
        raise ValueError("Successful saved rows must use the run's pinned index")
    latency = row.get("latency_ms")
    if latency is not None:
        latency = _number(latency, "latency")
    metrics = None
    metric_status = "draft_labels" if not reviewed else row.get("metric_status")
    if metric_status == "scored":
        saved = row.get("metrics")
        if not isinstance(saved, dict) or saved.get("status") != "scored":
            raise ValueError("Saved scored outcome has no matching metric record")
        if question.expected_answerability != "answerable":
            raise ValueError("Unanswerable questions cannot have scored retrieval metrics")
        metrics = {name: _number(saved.get(name), name, maximum=1) for name in METRICS}
    context = row.get("context")
    context_ids = context.get("chunk_ids", []) if isinstance(context, dict) else None
    hit_ids = [hit["chunk"]["chunk_id"] for hit in row.get("hits", [])]
    if len(set(hit_ids)) != len(hit_ids):
        raise ValueError("Saved retrieval contains duplicate candidate IDs")
    if context_ids is not None and (
        not isinstance(context_ids, list)
        or any(identifier not in hit_ids for identifier in context_ids)
        or len(set(context_ids)) != len(context_ids)
    ):
        raise ValueError("Saved context contains IDs outside its retrieved candidates")
    return {
        "status": "ok",
        "metric_status": metric_status,
        "metrics": metrics,
        "latency_ms": latency,
        "ranked_chunk_ids": hit_ids,
        "context": context,
        "covered_passages": row.get("covered_passages"),
        "reranking": row.get("reranking"),
    }


def _configuration_differences(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose differing settings without interpreting derived index/chunk IDs as tuning axes."""
    differences = []
    for name in CONFIGURATION_FIELDS:
        values = [run["inputs"].get(name, "not_recorded") for run in runs]
        if any(value != values[0] for value in values[1:]):
            differences.append({"field": name, "values_in_run_order": values})
    return differences


def compare_retrieval_runs(
    dataset: EvaluationDataset,
    runs: list[Path],
    split: Literal["development", "held_out"],
    *,
    allow_drafts: bool = False,
) -> dict[str, Any]:
    """Align every requested question; score deltas use the same fully scored sample set.

    Saved metric values are reported, not recomputed from source judgments. Artifact hashes
    establish which local files were read; they are not signatures or independent label review.
    """
    if len(runs) < 2:
        raise ValueError("Comparison requires at least two distinct saved runs")
    if not allow_drafts or split == "held_out":
        dataset.require_reviewed()
    # Reuse candidate-pool validation for exact labels/split, identities, corpus, and source scope.
    pool = pool_candidates(dataset, runs, split)
    reviewed = dataset.review_status == "reviewed" and all(
        question.review_status == "reviewed" for question in dataset.questions
    )
    questions = dataset.questions_for_split(split)
    observations: list[dict[str, dict[str, Any]]] = []
    manifests = []
    for provenance in pool["runs"]:
        directory = Path(provenance["directory"])
        summary_bytes = (directory / "summary.json").read_bytes()
        result_bytes = (directory / "results.jsonl").read_bytes()
        if (
            hashlib.sha256(summary_bytes).hexdigest() != provenance["summary_sha256"]
            or hashlib.sha256(result_bytes).hexdigest() != provenance["results_sha256"]
        ):
            raise ValueError("Saved run changed during comparison")
        summary = json.loads(summary_bytes)
        inputs = provenance["inputs"]
        if inputs.get("schema_version") not in (2, 3):
            raise ValueError("Comparison requires retrieval artifact schema 2 or 3")
        expected_fingerprint = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode("utf-8")
        ).hexdigest()
        expected_configuration = hashlib.sha256(
            json.dumps(
                {key: value for key, value in inputs.items() if key != "index"}, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        if summary.get("run_fingerprint") != expected_fingerprint:
            raise ValueError("Saved run fingerprint does not match its inputs")
        if summary.get("configuration_id") not in (None, expected_configuration):
            raise ValueError("Saved configuration ID does not match its inputs")
        if inputs["schema_version"] == 3 and (
            summary.get("configuration_id") is None
            or not isinstance(inputs.get("reranking"), dict)
            or "enrichment_fingerprint" not in inputs
        ):
            raise ValueError("Schema 3 run lacks complete retrieval identity")
        rows = {
            row["question_id"]: row
            for line in result_bytes.splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
        observations.append(
            {
                question.question_id: _observation(
                    question,
                    rows.get(question.question_id),
                    inputs | {"corpus_id": dataset.corpus_id},
                    summary.get("configuration_id"),
                    reviewed,
                )
                for question in questions
            }
        )
        conditions = summary.get("cache_conditions", {})
        if not isinstance(conditions, dict):
            raise ValueError("Invalid saved cache conditions")
        calls = conditions.get("model_requests")
        if calls is not None:
            _number(calls, "model request count")
            if not isinstance(calls, int):
                raise ValueError("Saved model request count must be an integer")
        manifests.append(
            provenance
            | {
                "label": directory.name,
                "configuration_id": summary.get("configuration_id"),
                "execution_conditions": summary.get("execution_conditions"),
                "cache_conditions": conditions,
                "recorded_model_requests": calls,
                "model_cost_usd": 0.0 if calls == 0 else None,
                "model_cost_status": "not_called" if calls == 0 else "unknown",
                "identity_limitations": ["schema_2_omits_reranking_and_metadata_identity"]
                if inputs["schema_version"] == 2
                else [],
            }
        )
    paired_scored = [
        question.question_id
        for question in questions
        if all(run[question.question_id]["metrics"] is not None for run in observations)
    ]
    paired_latency = [
        question.question_id
        for question in questions
        if all(run[question.question_id]["latency_ms"] is not None for run in observations)
    ]
    for manifest, run in zip(manifests, observations, strict=True):
        manifest.update(
            {
                "outcomes": dict(Counter(row["status"] for row in run.values())),
                "questions": len(questions),
                "scored_questions": sum(row["metrics"] is not None for row in run.values()),
                "metric_outcomes": dict(
                    Counter(row.get("metric_status", row["status"]) for row in run.values())
                ),
                "success_latency_ms": _latencies(
                    [row["latency_ms"] for row in run.values() if row["latency_ms"] is not None]
                ),
                "paired_latency_ms": _latencies(
                    [run[identifier]["latency_ms"] for identifier in paired_latency]
                ),
                "paired_metrics": {
                    name: statistics.mean(
                        run[identifier]["metrics"][name] for identifier in paired_scored
                    )
                    if paired_scored
                    else None
                    for name in METRICS
                },
            }
        )
    comparisons = []
    for number, run in enumerate(observations[1:], start=2):
        metrics = {}
        for name in METRICS:
            deltas = [
                run[identifier]["metrics"][name] - observations[0][identifier]["metrics"][name]
                for identifier in paired_scored
            ]
            metrics[name] = {
                "mean_delta": statistics.mean(deltas) if deltas else None,
                "higher": sum(value > 0 for value in deltas),
                "lower": sum(value < 0 for value in deltas),
                "equal": sum(value == 0 for value in deltas),
            }
        comparisons.append({"baseline_run": 1, "candidate_run": number, "metrics": metrics})
    return {
        "schema_version": 1,
        "kind": "saved_retrieval_comparison",
        "quality_status": "draft_labels"
        if not reviewed
        else ("paired_saved_metrics" if paired_scored else "no_common_scored_questions"),
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint(),
        "corpus_id": dataset.corpus_id,
        "split": split,
        "reviewed_labels": reviewed,
        "questions": len(questions),
        "paired_scored_question_ids": paired_scored,
        "paired_latency_question_ids": paired_latency,
        "configuration_differences": _configuration_differences(manifests),
        "runs": manifests,
        "comparisons": comparisons,
        "per_question": [
            {
                "question_id": question.question_id,
                "query": question.query,
                "expected_answerability": question.expected_answerability,
                "observations_in_run_order": [run[question.question_id] for run in observations],
            }
            for question in questions
        ],
        "notes": [
            "First run is the explicit baseline; no winner or release configuration is selected.",
            "Quality means/deltas use only questions scored in every run; "
            "all failures and missing rows remain counted.",
            "Metrics are saved observations, not independently recomputed or new label review.",
            "Latency covers recorded successful retrieval calls, "
            "not end-to-end answers or failed-call time.",
            "Cache, execution order, host load, and concurrency may differ; "
            "these descriptive samples are not a controlled latency benchmark.",
            "Model cost excludes hosting and offline indexing/enrichment; "
            "unknown usage/prices remain unknown.",
            "Source coverage fields and draft references are diagnostics, "
            "not independent relevance judgments.",
        ],
    }


def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Render the aligned denominators and gaps beside recorded latency and cost."""

    def cell(value: object) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Saved retrieval comparison",
        "",
        f"Quality status: {cell(report['quality_status'])}. Questions: {report['questions']}. "
        f"Common scored questions: {len(report['paired_scored_question_ids'])}.",
        "",
        "| Run | OK / error / missing | Scored | Paired MRR@10 | Paired nDCG@10 | "
        "Successful retrieval median ms (N) | Recorded model requests | Model cost USD |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        counts = run["outcomes"]
        latency = run["success_latency_ms"]
        lines.append(
            f"| {run['run_number']}: {cell(run['label'])} | "
            f"{counts.get('ok', 0)} / {counts.get('error', 0)} / {counts.get('missing', 0)} | "
            f"{run['scored_questions']} | {cell(run['paired_metrics']['mrr_at_10'])} | "
            f"{cell(run['paired_metrics']['ndcg_at_10'])} | "
            f"{cell(latency['median'])} ({latency['samples']}) | "
            f"{cell(run['recorded_model_requests'])} | {cell(run['model_cost_usd'])} |"
        )
    lines.extend(["", "Recorded configuration differences:", ""])
    for difference in report["configuration_differences"]:
        lines.append(
            f"- {cell(difference['field'])}: {cell(json.dumps(difference['values_in_run_order']))}"
        )
    if not report["configuration_differences"]:
        lines.append(
            "- None among recorded query/chunk/model settings; see JSON for artifact identities."
        )
    lines.extend(["", "Limits:", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    for run in report["runs"]:
        for limitation in run["identity_limitations"]:
            lines.append(f"- Run {run['run_number']}: {limitation}.")
    return "\n".join(lines) + "\n"
