"""Keep per-question failures and unreviewed outcomes visible in saved evaluation runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.eval.dataset import EvaluationPassage, load_dataset
from rag.eval.runs import run_retrieval_evaluation
from rag.models import RerankingReport, RetrievalResult, RetrievedHit, SectionRecord
from rag.reranking import RerankingError


def test_eval_persists_failures_and_never_scores_draft_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig()
    config.chunking.size = 150
    config.chunking.overlap = 20
    config.retrieval.context_passages = 1
    config.index.enrichment_fingerprint = "a" * 64
    config.reranking.enabled = True
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    section = SectionRecord.model_validate(fixture)
    chunks = chunk_section(section, config.chunking)
    dataset = load_dataset(Path("data/eval/pilot.json"), allow_drafts=True)
    dataset.questions = dataset.questions[:2]
    tail = section.text[-80:]
    dataset.passages = [
        EvaluationPassage(
            passage_id="test-source-tail",
            source_id=section.source_id,
            section_id=section.section_id,
            raw_section_sha256=section.raw_sha256,
            cleaning_version=section.cleaning_version,
            text_start=len(section.text) - len(tail),
            text_end=len(section.text),
            text_sha256=hashlib.sha256(tail.encode("utf-8")).hexdigest(),
        )
    ]
    dataset.questions[0].reference_passage_ids = ["test-source-tail"]
    dataset.questions[0].required_evidence_groups = [["test-source-tail"]]
    pinned = config.model_copy(deep=True)
    pinned.index.name = "snapshot-pinned"
    resolve = Mock(return_value=pinned)
    monkeypatch.setattr("rag.eval.runs.resolve_read_config", resolve)
    retrieval = Mock(
        side_effect=[
            RetrievalResult(
                hits=[
                    RetrievedHit(chunk=chunks[0], method="bm25", rank=1, score=2),
                    RetrievedHit(chunk=chunks[-1], method="bm25", rank=2, score=1),
                ],
                mode="bm25",
                timings_ms={"retrieval_total": 1},
                reranking=RerankingReport(
                    status="applied",
                    model_arn=None,
                    input_count=2,
                    candidate_ids=[chunks[0].chunk_id, chunks[-1].chunk_id],
                    input_characters=100,
                    attempted_calls=1,
                    usage_status="not_reported",
                ),
            ),
            RerankingError(
                RerankingReport(
                    status="failed",
                    model_arn=None,
                    input_count=2,
                    candidate_ids=[chunks[0].chunk_id, chunks[-1].chunk_id],
                    input_characters=100,
                    attempted_calls=1,
                    usage_status="not_reported",
                    reason="provider_error",
                )
            ),
        ]
    )
    monkeypatch.setattr("rag.eval.runs.retrieve", retrieval)
    output = tmp_path / "run"
    summary = run_retrieval_evaluation(
        Mock(), config, dataset, [section], chunks, "bm25", "development", output, allow_drafts=True
    )
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in rows] == ["ok", "error"]
    assert rows[0]["metric_status"] == "draft_labels"
    assert rows[0]["covered_passages"] == ["test-source-tail"]
    context = rows[0]["context"]
    assert context["chunk_ids"] == [chunks[0].chunk_id]
    assert json.loads(context["text"])[0]["text"] == chunks[0].original_text
    assert context["covered_passages"] == []
    assert context["missing_reference_passages"] == ["test-source-tail"]
    assert context["missing_evidence_groups"] == [["test-source-tail"]]
    assert "context" not in rows[1]
    assert all(row["index"] == "snapshot-pinned" for row in rows)
    resolve.assert_called_once()
    assert all(call.args[1].index.name == "snapshot-pinned" for call in retrieval.call_args_list)
    assert summary["inputs"]["index"] == "snapshot-pinned"
    assert all(row["configuration_id"] == summary["configuration_id"] for row in rows)
    assert summary["outcomes"] == {"ok": 1, "error": 1}
    assert summary["scored"] == 0
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["questions"] == 2
    assert summary["cache_conditions"]["model_requests"] == 2
    assert summary["cache_conditions"]["reranking_requests"] == 2
    assert rows[1]["reranking"]["status"] == "failed"
    assert summary["reranking"]["model_cost_usd"] is None
    assert summary["inputs"]["enrichment_fingerprint"] == "a" * 64
    retrieval.side_effect = None
    retrieval.return_value = RetrievalResult(
        hits=[], mode="bm25", timings_ms={"retrieval_total": 1}
    )
    config.index.enrichment_fingerprint = "b" * 64
    changed = run_retrieval_evaluation(
        Mock(),
        config,
        dataset,
        [section],
        chunks,
        "bm25",
        "development",
        tmp_path / "changed",
        allow_drafts=True,
    )
    assert changed["configuration_id"] != summary["configuration_id"]
    config.reranking.failure_policy = "original_order"
    fallback = run_retrieval_evaluation(
        Mock(),
        config,
        dataset,
        [section],
        chunks,
        "bm25",
        "development",
        tmp_path / "fallback",
        allow_drafts=True,
    )
    assert fallback["configuration_id"] != changed["configuration_id"]
