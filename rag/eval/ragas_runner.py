"""Run optional offline Ragas metrics serially over reviewed, prepared answer artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from langsmith import tracing_context

from rag.aws_clients import BedrockRuntime
from rag.config import ConfigurationError, PipelineConfig
from rag.corpus import write_jsonl
from rag.eval.answers import PreparedAnswer
from rag.eval.dataset import EvaluationDataset

if TYPE_CHECKING:
    from ragas.metrics.collections.base import BaseMetric

    from rag.eval.ragas_judge import BedrockJudge


def _components(
    config: PipelineConfig,
    runtime: BedrockRuntime | None,
) -> tuple[BedrockJudge, list[tuple[str, BaseMetric, tuple[str, ...]]], str]:
    """Load optional dependencies only after disabling their external usage reporting."""
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    try:
        from ragas import __version__
        from ragas._analytics import do_not_track
        from ragas.metrics.collections import (
            ContextPrecisionWithReference,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        from rag.eval.ragas_judge import BedrockJudge
    except ImportError as error:
        raise ConfigurationError(
            "Optional evaluator dependencies unavailable; run uv sync --group eval: " + str(error)
        ) from error
    # The pinned library caches this environment flag, including earlier imports by a caller.
    do_not_track.cache_clear()
    judge = BedrockJudge(config, runtime)
    metrics = [
        ("faithfulness", Faithfulness(llm=judge), ("user_input", "response", "retrieved_contexts")),
        (
            "context_precision_with_reference",
            ContextPrecisionWithReference(llm=judge),
            ("user_input", "reference", "retrieved_contexts"),
        ),
        (
            "context_recall",
            ContextRecall(llm=judge),
            ("user_input", "retrieved_contexts", "reference"),
        ),
        (
            "factual_correctness",
            FactualCorrectness(
                llm=judge,
                mode="f1",
                beta=1.0,
                atomicity="low",
                coverage="low",
            ),
            ("response", "reference"),
        ),
    ]
    return judge, metrics, __version__


async def _score_answers(
    prepared: list[PreparedAnswer],
    judge: BedrockJudge,
    metrics: list[tuple[str, BaseMetric, tuple[str, ...]]],
    output: Path,
    run_fingerprint: str,
) -> list[dict[str, object]]:
    """Flush each outcome before aggregation, including failures and undefined values."""
    from ragas import SingleTurnSample

    rows: list[dict[str, object]] = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for answer in prepared:
            sample = (
                SingleTurnSample.model_validate(answer.sample.model_dump())
                if answer.sample
                else None
            )
            for name, metric, fields in metrics:
                judge.metric_name = name
                calls, hits = judge.model_calls, judge.cache_hits
                input_tokens, output_tokens = judge.known_input_tokens, judge.known_output_tokens
                started = monotonic()
                row: dict[str, object] = {
                    "question_id": answer.question_id,
                    "run_fingerprint": run_fingerprint,
                    "artifact_sha256": answer.artifact_sha256,
                    "metric": name,
                    "answer_outcome": answer.status,
                    "status": "inapplicable",
                    "value": None,
                }
                if answer.status == "error":
                    row.update(status="error", reason=answer.error)
                elif answer.status != "ready" or sample is None:
                    row["reason"] = answer.status
                else:
                    try:
                        values = sample.model_dump()
                        result = await metric.ascore(**{field: values[field] for field in fields})
                        value = float(result.value)
                        if not math.isfinite(value):
                            row.update(status="undefined", reason="Metric has no finite score")
                        elif not 0 <= value <= 1:
                            raise ValueError("Evaluator score is outside [0, 1]")
                        else:
                            row.update(status="scored", value=value)
                    except Exception as error:
                        # Offline per-metric boundary: one judge/parser/budget failure cannot
                        # hide the rest of the requested metrics. Interrupts still propagate.
                        row.update(
                            status="error", error_type=type(error).__name__, reason=str(error)
                        )
                row.update(
                    model_calls=judge.model_calls - calls,
                    cache_hits=judge.cache_hits - hits,
                    known_input_tokens=judge.known_input_tokens - input_tokens,
                    known_output_tokens=judge.known_output_tokens - output_tokens,
                    usage_complete=judge.usage_complete,
                    latency_ms=(monotonic() - started) * 1000,
                )
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
    return rows


def run_ragas_evaluation(
    config: PipelineConfig,
    dataset: EvaluationDataset,
    prepared: list[PreparedAnswer],
    output: Path,
    *,
    runtime: BedrockRuntime | None = None,
) -> dict[str, object]:
    """Require reviewed labels and explicit judge settings before any client can be created."""
    dataset.require_reviewed()
    if not config.evaluation.judge_model_id or not config.evaluation.judge_model_id.strip():
        raise ConfigurationError("Set RAG_EVALUATOR_MODEL_ID explicitly for offline judging")
    config.require_cloud_settings(require_generation=False)
    if not prepared or len(prepared) > config.evaluation.max_samples:
        raise ValueError("Prepared answer count exceeds the selected evaluation limit")
    known = {question.question_id for question in dataset.questions}
    if any(
        answer.question_id not in known or answer.review_status != "reviewed" for answer in prepared
    ):
        raise ValueError("Prepared answers must belong to reviewed dataset questions")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "prepared.jsonl", [answer.model_dump(mode="json") for answer in prepared])
    judge, metrics, version = _components(config, runtime)
    try:
        rubric = {
            "factual_correctness": {
                "mode": "f1",
                "beta": 1.0,
                "atomicity": "low",
                "coverage": "low",
            }
        }
        identity = {
            "schema_version": 1,
            "dataset_sha256": dataset.fingerprint(),
            "ragas_version": version,
            "aws_region": config.cloud.region,
            "judge_settings": config.evaluation.model_dump(mode="json", exclude={"cache_path"}),
            "metrics": [name for name, _, _ in metrics],
            "rubric": rubric,
            "runtime_injected": not judge.owns_runtime,
            "answers": [
                {
                    "question_id": answer.question_id,
                    "artifact_sha256": answer.artifact_sha256,
                    "pipeline_fingerprint": answer.pipeline_fingerprint,
                    "status": answer.status,
                }
                for answer in prepared
            ],
        }
        run_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        (output / "manifest.json").write_text(
            json.dumps(identity | {"run_fingerprint": run_fingerprint}, indent=2) + "\n",
            encoding="utf-8",
        )
        # Both Ragas telemetry and ambient LangSmith tracing are disabled for this local tool.
        with tracing_context(enabled=False):
            rows = asyncio.run(_score_answers(prepared, judge, metrics, output, run_fingerprint))
        summary = {
            "schema_version": 1,
            "run_fingerprint": run_fingerprint,
            "ragas_version": version,
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.fingerprint(),
            "questions": len(prepared),
            "question_outcomes": dict(Counter(answer.status for answer in prepared)),
            "metric_outcomes": dict(Counter(str(row["status"]) for row in rows)),
            "metrics": {},
            "judge_settings": config.evaluation.model_dump(mode="json", exclude={"cache_path"}),
            "concurrency": 1,
            "sdk_attempts_per_request": 1 if judge.owns_runtime else None,
            "runtime_injected": not judge.owns_runtime,
            "model_calls": judge.model_calls,
            "cache_hits": judge.cache_hits,
            "known_input_tokens": judge.known_input_tokens,
            "known_output_tokens": judge.known_output_tokens,
            "usage_complete": judge.usage_complete,
            "cost_usd": None,
            "cost_status": "unpriced; call limits do not enforce a dollar ceiling",
            "rubric": rubric,
        }
        for name, _, _ in metrics:
            metric_rows = [row for row in rows if row["metric"] == name]
            values = [float(row["value"]) for row in metric_rows if row["status"] == "scored"]
            summary["metrics"][name] = {
                "outcomes": dict(Counter(str(row["status"]) for row in metric_rows)),
                "scored": len(values),
                "mean": statistics.mean(values) if values else None,
            }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        judge.close()
