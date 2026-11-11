"""Pool saved retriever candidates for evidence review without assigning relevance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

from rag.eval.dataset import EvaluationDataset
from rag.models import RetrievedHit


class PooledCandidate(TypedDict):
    """One original chunk with every observed ranking, including repeat runs."""

    chunk: dict[str, object]
    observations: list[dict[str, object]]


class QuestionPool(TypedDict):
    """Keep failed/missing results alongside deduplicated candidate evidence."""

    question: dict[str, object]
    outcomes: list[dict[str, object]]
    candidates: list[PooledCandidate]


def pool_candidates(
    dataset: EvaluationDataset,
    runs: list[Path],
    split: Literal["development", "held_out"],
) -> dict[str, object]:
    """Combine compatible saved runs; pooled text is a review input, never ground truth."""
    if not runs or len({run.resolve() for run in runs}) != len(runs):
        raise ValueError("Candidate pooling requires distinct saved run directories")
    questions = dataset.questions_for_split(split)
    if not questions:
        raise ValueError(f"Dataset contains no questions in split {split!r}")
    pools: dict[str, QuestionPool] = {
        question.question_id: {
            "question": question.model_dump(mode="json"),
            "outcomes": [],
            "candidates": [],
        }
        for question in questions
    }
    indexed: dict[tuple[str, str], PooledCandidate] = {}
    provenance: list[dict[str, object]] = []
    fingerprint = dataset.fingerprint()
    for run_number, run in enumerate(runs, start=1):
        summary_bytes = (run / "summary.json").read_bytes()
        result_bytes = (run / "results.jsonl").read_bytes()
        summary = json.loads(summary_bytes)
        if not isinstance(summary, dict) or not isinstance(summary.get("inputs"), dict):
            raise ValueError(f"Run has no input manifest: {run}")
        inputs = summary["inputs"]
        if inputs.get("dataset_sha256") != fingerprint or inputs.get("split") != split:
            raise ValueError(f"Run dataset fingerprint or split differs: {run}")
        if inputs.get("mode") not in ("bm25", "vector", "hybrid"):
            raise ValueError(f"Run has an unknown retrieval mode: {run}")
        chunk_ids = inputs.get("chunk_ids")
        if not isinstance(chunk_ids, list) or not all(isinstance(i, str) for i in chunk_ids):
            raise ValueError(f"Run has no valid chunk inventory: {run}")
        recorded_chunks = set(chunk_ids)
        provenance.append(
            {
                "run_number": run_number,
                "directory": str(run),
                "run_fingerprint": summary.get("run_fingerprint"),
                "inputs": inputs,
                "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
            }
        )
        rows: dict[str, dict[str, object]] = {}
        for line in result_bytes.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get("question_id") if isinstance(row, dict) else None
            if not isinstance(identifier, str) or identifier not in pools:
                raise ValueError(f"Run contains an unknown question: {run}")
            if identifier in rows:
                raise ValueError(f"Run repeats question {identifier}: {run}")
            rows[identifier] = row
        for question in questions:
            pool = pools[question.question_id]
            row = rows.get(question.question_id)
            if row is None:
                pool["outcomes"].append({"run_number": run_number, "status": "missing"})
                continue
            if row.get("query") != question.query or row.get("mode") != inputs.get("mode"):
                raise ValueError(f"Run question or retrieval mode differs: {run}")
            if row.get("status") not in ("ok", "error"):
                raise ValueError(f"Run contains an unknown outcome: {run}")
            outcome: dict[str, object] = {"run_number": run_number, "status": row["status"]}
            if row["status"] == "error":
                outcome.update({"error_type": row.get("error_type"), "error": row.get("error")})
            pool["outcomes"].append(outcome)
            hits = row.get("hits", [])
            if not isinstance(hits, list):
                raise ValueError(f"Run has no valid hit list: {run}")
            for position, saved_hit in enumerate(hits, start=1):
                hit = RetrievedHit.model_validate(saved_hit)
                chunk = hit.chunk
                if (
                    chunk.chunk_id not in recorded_chunks
                    or chunk.corpus_id != dataset.corpus_id
                    or chunk.provenance.guide not in question.permitted_guides
                    or chunk.source_scope != question.source_scope
                ):
                    raise ValueError(f"Candidate is outside the recorded corpus/scope: {run}")
                key = (question.question_id, chunk.chunk_id)
                record = chunk.model_dump(mode="json")
                if key not in indexed:
                    indexed[key] = {"chunk": record, "observations": []}
                    pool["candidates"].append(indexed[key])
                candidate = indexed[key]
                if candidate["chunk"] != record:
                    raise ValueError(f"Conflicting records for chunk ID: {chunk.chunk_id}")
                candidate["observations"].append(
                    {
                        "run_number": run_number,
                        "position": position,
                        **hit.model_dump(mode="json", exclude={"chunk"}),
                    }
                )
    return {
        "schema_version": 1,
        "status": "candidate_pool_pending_review",
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": fingerprint,
        "corpus_id": dataset.corpus_id,
        "split": split,
        "dataset_review_status": dataset.review_status,
        "note": "Candidates are observations, not relevance judgments. Verify against pinned "
        "sources and label alternative evidence before scoring. Pool order is not a quality rank.",
        "runs": provenance,
        "passages": [passage.model_dump(mode="json") for passage in dataset.passages],
        "questions": list(pools.values()),
    }
