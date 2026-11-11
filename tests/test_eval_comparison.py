"""Protect aligned denominators, draft gates, and saved-run integrity in comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.cli import main
from rag.config import PipelineConfig
from rag.eval.comparison import compare_retrieval_runs, render_comparison_markdown
from rag.eval.dataset import EvaluationDataset, EvaluationPassage, EvaluationQuestion
from rag.eval.pooling import pool_candidates
from rag.models import ChunkRecord, RetrievedHit, SearchScope


def _dataset(*, reviewed: bool = True) -> EvaluationDataset:
    """Use synthetic labels only; never promote the checked-in S3 datasets."""
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))
    section = fixture["section"]
    return EvaluationDataset(
        dataset_id="synthetic-comparison",
        corpus_id=section["corpus_id"],
        review_status="reviewed" if reviewed else "draft",
        note="Synthetic saved-metric fixture; no real human review or quality claim.",
        passages=[
            EvaluationPassage(
                passage_id="synthetic-span",
                source_id=section["source_id"],
                section_id=section["section_id"],
                raw_section_sha256=section["raw_sha256"],
                cleaning_version=section["cleaning_version"],
                text_start=0,
                text_end=20,
                text_sha256=hashlib.sha256(section["text"][:20].encode()).hexdigest(),
            )
        ],
        questions=[
            EvaluationQuestion(
                question_id=f"synthetic-{number}",
                query=f"Synthetic comparison question {number}?",
                category="direct_fact",
                permitted_guides=["s3-userguide"],
                source_scope="general-purpose",
                reference_answer="Synthetic reference, not a measured S3 answer.",
                reference_passage_ids=["synthetic-span"],
                relevance_grades={"synthetic-span": 1},
                expected_answerability="answerable",
                review_status="reviewed" if reviewed else "draft",
                split="development",
            )
            for number in range(3)
        ],
    )


def _save_run(
    path: Path,
    dataset: EvaluationDataset,
    scores: list[float | str | None],
    *,
    schema: int = 3,
    model_requests: int = 0,
) -> None:
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))
    chunk = ChunkRecord.model_validate(fixture["chunk"])
    config = PipelineConfig()
    identity = {
        "schema_version": schema,
        "dataset_sha256": dataset.fingerprint(),
        "split": "development",
        "mode": "bm25",
        "chunk_ids": [chunk.chunk_id],
        "chunking": config.chunking.model_dump(mode="json"),
        "retrieval": config.retrieval.model_dump(mode="json"),
        "models": config.models.model_dump(mode="json"),
    }
    if schema == 3:
        identity |= {"reranking": config.reranking.model_dump(), "enrichment_fingerprint": None}
    configuration_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity["index"] = "synthetic-snapshot"
    summary = {
        "inputs": identity,
        "run_fingerprint": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "cache_conditions": {"model_requests": model_requests},
        # Deliberately incorrect summary aggregates: comparison must use aligned rows.
        "metrics": {"mrr_at_10": 999},
        "questions": 999,
        "reviewed_labels": True,
    }
    if schema == 3:
        summary["configuration_id"] = configuration_id
    rows = []
    for question, score in zip(dataset.questions, scores, strict=True):
        if score is None:
            continue
        row = {
            "question_id": question.question_id,
            "query": question.query,
            "mode": "bm25",
            "status": "error" if score == "error" else "ok",
            "review_status": question.review_status,
            "expected_answerability": question.expected_answerability,
            "scope": SearchScope(
                permitted_guides=question.permitted_guides,
                corpus_id=dataset.corpus_id,
            ).model_dump(mode="json"),
            "index": identity["index"],
        }
        if schema == 3:
            row["configuration_id"] = configuration_id
        if score == "error":
            row.update({"error_type": "ReadTimeoutError", "error": "Synthetic timeout"})
        else:
            row.update(
                {
                    "latency_ms": 10 * (len(rows) + 1),
                    "metric_status": "scored",
                    "metrics": {
                        "status": "scored",
                        **dict.fromkeys(
                            ("hit_at_5", "hit_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_10"),
                            score,
                        ),
                    },
                    "hits": [
                        RetrievedHit(chunk=chunk, method="bm25", rank=1, score=2).model_dump(
                            mode="json"
                        )
                    ],
                    "context": {"chunk_ids": [chunk.chunk_id], "text": "[]"},
                }
            )
        rows.append(row)
    path.mkdir()
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_comparison_uses_common_scored_questions_without_hiding_failed_or_missing_rows(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _save_run(baseline, dataset, [0, 1, 0.5], schema=2)
    _save_run(candidate, dataset, [1, "error", None], model_requests=1)
    report = compare_retrieval_runs(dataset, [baseline, candidate], "development")
    assert report["quality_status"] == "paired_saved_metrics"
    assert report["paired_scored_question_ids"] == ["synthetic-0"]
    assert report["paired_latency_question_ids"] == ["synthetic-0"]
    first, second = report["runs"]
    assert first["questions"] == second["questions"] == 3
    assert first["outcomes"] == {"ok": 3}
    assert second["outcomes"] == {"ok": 1, "error": 1, "missing": 1}
    assert [run["scored_questions"] for run in report["runs"]] == [3, 1]
    assert [run["paired_metrics"]["mrr_at_10"] for run in report["runs"]] == [0, 1]
    assert first["success_latency_ms"]["samples"] == 3
    assert first["paired_latency_ms"]["samples"] == 1
    assert first["model_cost_usd"] == 0 and second["model_cost_usd"] is None
    assert first["identity_limitations"] and not second["identity_limitations"]
    delta = report["comparisons"][0]["metrics"]["mrr_at_10"]
    assert delta == {"mean_delta": 1, "higher": 1, "lower": 0, "equal": 0}
    assert report["per_question"][1]["observations_in_run_order"][1]["status"] == "error"
    markdown = render_comparison_markdown(report)
    assert "1 / 1 / 1" in markdown and "unavailable" in markdown
    assert "no winner" in markdown and "not a controlled latency benchmark" in markdown


def test_drafts_cannot_be_scored_and_cli_needs_no_services_or_optional_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(reviewed=False)
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for directory in (baseline, candidate):
        _save_run(directory, dataset, [1, 1, 1])
    with pytest.raises(ValueError, match="human review"):
        compare_retrieval_runs(dataset, [baseline, candidate], "development")
    with pytest.raises(ValueError, match="human review"):
        compare_retrieval_runs(dataset, [baseline, candidate], "held_out", allow_drafts=True)
    report = compare_retrieval_runs(
        dataset, [baseline, candidate], "development", allow_drafts=True
    )
    assert report["quality_status"] == "draft_labels" and not report["reviewed_labels"]
    assert report["paired_scored_question_ids"] == []
    assert all(run["scored_questions"] == 0 for run in report["runs"])
    assert all(value is None for run in report["runs"] for value in run["paired_metrics"].values())
    search, bedrock = Mock(side_effect=AssertionError), Mock(side_effect=AssertionError)
    monkeypatch.setattr("rag.cli.create_search_client", search)
    monkeypatch.setattr("rag.cli.create_bedrock_client", bedrock)
    labels = tmp_path / "drafts.json"
    labels.write_text(dataset.model_dump_json(), encoding="utf-8")
    output = tmp_path / "comparison"
    args = [
        "eval",
        "compare",
        "--dataset",
        str(labels),
        "--run",
        str(baseline),
        "--run",
        str(candidate),
        "--allow-drafts",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    saved = (output / "comparison.json").read_bytes()
    assert json.loads(saved)["quality_status"] == "draft_labels"
    assert (output / "comparison.md").exists()
    assert main(args) == 1
    assert (output / "comparison.json").read_bytes() == saved
    search.assert_not_called()
    bedrock.assert_not_called()


def test_comparison_refuses_mixed_identity_scope_invalid_metrics_and_changed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset()
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for directory in (baseline, candidate):
        _save_run(directory, dataset, [1, 1, 1])
    summary_path, rows_path = candidate / "summary.json", candidate / "results.jsonl"
    summary_bytes, rows_bytes = summary_path.read_bytes(), rows_path.read_bytes()
    summary = json.loads(summary_bytes)
    summary["inputs"]["retrieval"]["candidate_count"] = 10
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        compare_retrieval_runs(dataset, [baseline, candidate], "development")
    summary_path.write_bytes(summary_bytes)
    for field, value in (
        ("scope", {}),
        ("latency_ms", float("nan")),
        ("metrics", {"status": "scored", "mrr_at_10": 2}),
        ("index", "different-snapshot"),
        ("configuration_id", None),
    ):
        rows = [json.loads(line) for line in rows_bytes.splitlines()]
        rows[0][field] = value
        rows_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        with pytest.raises(ValueError):
            compare_retrieval_runs(dataset, [baseline, candidate], "development")
    rows_path.write_bytes(rows_bytes)

    def changing_pool(*args: object, **kwargs: object) -> dict[str, object]:
        pool = pool_candidates(*args, **kwargs)
        rows_path.write_bytes(rows_bytes + b"\n")
        return pool

    monkeypatch.setattr("rag.eval.comparison.pool_candidates", changing_pool)
    with pytest.raises(ValueError, match="changed during comparison"):
        compare_retrieval_runs(dataset, [baseline, candidate], "development")
