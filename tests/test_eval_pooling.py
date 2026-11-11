"""Keep alternative evidence and failed requests visible without mixing label revisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.eval.dataset import EvaluationDataset, load_dataset
from rag.eval.pooling import pool_candidates
from rag.models import ChunkRecord, RetrievedHit


def _save_run(
    path: Path,
    dataset: EvaluationDataset,
    mode: str,
    rows: list[dict[str, object]],
    chunks: list[ChunkRecord],
) -> None:
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {
                "run_fingerprint": "a" * 64,
                "inputs": {
                    "dataset_sha256": dataset.fingerprint(),
                    "split": "development",
                    "mode": mode,
                    "chunk_ids": [chunk.chunk_id for chunk in chunks],
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_pool_keeps_all_rankings_alternative_candidates_and_failed_or_missing_questions(
    tmp_path: Path,
) -> None:
    dataset = load_dataset(Path("data/eval/pilot.json"), allow_drafts=True)
    dataset.questions = dataset.questions[:2]
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))
    chunk = ChunkRecord.model_validate(fixture["chunk"])
    alternative = chunk.model_copy(update={"chunk_id": "alternative-boundary-config"})
    keyword, vector = tmp_path / "keyword", tmp_path / "vector"
    first, second = dataset.questions
    for directory, mode, chunks in (
        (keyword, "bm25", [chunk]),
        (vector, "vector", [alternative, chunk]),
    ):
        rows = [
            {
                "question_id": first.question_id,
                "query": first.query,
                "mode": mode,
                "status": "ok",
                "hits": [
                    RetrievedHit(chunk=item, method=mode, rank=rank, score=1).model_dump(
                        mode="json"
                    )
                    for rank, item in enumerate(chunks, start=1)
                ],
            }
        ]
        if mode == "bm25":
            rows.append(
                {
                    "question_id": second.question_id,
                    "query": second.query,
                    "mode": mode,
                    "status": "error",
                    "error_type": "ValueError",
                    "error": "Search failed",
                }
            )
        _save_run(directory, dataset, mode, rows, chunks)
    pool = pool_candidates(dataset, [keyword, vector], "development")
    assert pool["status"] == "candidate_pool_pending_review"
    questions = pool["questions"]
    candidates = {item["chunk"]["chunk_id"]: item for item in questions[0]["candidates"]}
    assert set(candidates) == {chunk.chunk_id, alternative.chunk_id}
    assert [item["position"] for item in candidates[chunk.chunk_id]["observations"]] == [1, 2]
    assert candidates[chunk.chunk_id]["chunk"]["original_text"] == chunk.original_text
    assert questions[1]["candidates"] == []
    assert [item["status"] for item in questions[1]["outcomes"]] == ["error", "missing"]
    assert questions[1]["outcomes"][0]["error"] == "Search failed"
    assert all(question["question"]["review_status"] == "draft" for question in questions)


def test_pool_refuses_different_labels_and_conflicting_evidence(tmp_path: Path) -> None:
    dataset = load_dataset(Path("data/eval/pilot.json"), allow_drafts=True)
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))
    chunk = ChunkRecord.model_validate(fixture["chunk"])
    question = dataset.questions[0]
    row = {
        "question_id": question.question_id,
        "query": question.query,
        "mode": "bm25",
        "status": "ok",
        "hits": [RetrievedHit(chunk=chunk, method="bm25", rank=1, score=1).model_dump(mode="json")],
    }
    original, changed = tmp_path / "original", tmp_path / "changed"
    _save_run(original, dataset, "bm25", [row], [chunk])
    other_labels = dataset.model_copy(update={"dataset_id": "different-label-revision"})
    _save_run(changed, other_labels, "bm25", [row], [chunk])
    with pytest.raises(ValueError, match="dataset fingerprint or split differs"):
        pool_candidates(dataset, [original, changed], "development")
    (changed / "summary.json").write_bytes((original / "summary.json").read_bytes())
    row["hits"][0]["chunk"]["heading_path"] = ["Changed evidence under the same ID"]
    (changed / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting records"):
        pool_candidates(dataset, [original, changed], "development")
