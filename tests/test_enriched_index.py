"""Reject altered or mixed metadata experiments before indexing or embedding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.config import PipelineConfig
from rag.enriched_index import build_enriched_index, load_enriched_inputs
from rag.enrichment import run_enrichment
from rag.models import ChunkRecord


def test_metadata_loader_checks_content_binding_beyond_saved_file_hashes(tmp_path: Path) -> None:
    chunk = ChunkRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    )
    config = PipelineConfig(
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
        enrichment={"model_id": "test-model", "cache_path": tmp_path / "cache.sqlite3"},
    )
    runtime = Mock()
    runtime.converse.return_value = {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "content": [
                    {
                        "text": json.dumps(
                            {
                                "summary": None,
                                "topic": "unknown",
                                "intent": "unknown",
                                "related_concepts": [],
                            }
                        )
                    }
                ]
            }
        },
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }
    output = tmp_path / "metadata"
    run_enrichment(config, [chunk], output, runtime=runtime)
    inputs = load_enriched_inputs(output)
    assert inputs.chunks == [chunk] and inputs.embedding_texts == [chunk.original_text]
    saved = (output / "results.jsonl").read_bytes()
    (output / "results.jsonl").write_bytes(saved + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_enriched_inputs(output)
    (output / "results.jsonl").write_bytes(saved)
    summaries = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    bad = json.loads(saved)
    bad["chunk_sha256"] = "0" * 64
    (output / "results.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    summaries["artifact_sha256"]["results.jsonl"] = hashlib.sha256(
        (output / "results.jsonl").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")
    client = Mock()
    embedder = Mock()
    with pytest.raises(ValueError, match="original chunk"):
        build_enriched_index(client, config, output, tmp_path / "index", embedder=embedder)
    assert not client.mock_calls and not embedder.mock_calls
    assert not (tmp_path / "index").exists()
