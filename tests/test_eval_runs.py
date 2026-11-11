"""Keep per-question failures and unreviewed outcomes visible in saved evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.eval.dataset import load_dataset
from rag.eval.runs import run_retrieval_evaluation
from rag.models import RetrievalResult, RetrievedHit, SectionRecord


def test_eval_persists_failures_and_never_scores_draft_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig()
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    section = SectionRecord.model_validate(fixture)
    chunks = chunk_section(section, config.chunking)
    dataset = load_dataset(Path("data/eval/pilot.json"), allow_drafts=True)
    dataset.questions = dataset.questions[:2]
    retrieval = Mock(
        side_effect=[
            RetrievalResult(
                hits=[RetrievedHit(chunk=chunks[0], method="bm25", rank=1, score=1)],
                mode="bm25",
                timings_ms={"retrieval_total": 1},
            ),
            ValueError("Search failed"),
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
    assert summary["outcomes"] == {"ok": 1, "error": 1}
    assert summary["scored"] == 0
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["questions"] == 2
