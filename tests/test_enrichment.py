"""Protect evidence integrity, bounded failure handling, and reusable metadata artifacts."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

from rag.cli import main
from rag.config import PipelineConfig
from rag.enrichment import EnrichmentRecord, embedding_input, run_enrichment
from rag.generation import assemble_context
from rag.models import ChunkRecord, RetrievedHit

METADATA = {
    "summary": "Versioning keeps multiple versions of an object.",
    "topic": "versioning",
    "intent": "concept",
    "related_concepts": ["object versions"],
    "example_phrasings": ["What does versioning retain?"],
}


def source_chunks(count: int = 1) -> list[ChunkRecord]:
    """Use the existing representative evidence record with distinct source identities."""
    value = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    return [ChunkRecord.model_validate(value | {"chunk_id": f"chunk-{i}"}) for i in range(count)]


def configured(tmp_path: Path, **settings: object) -> PipelineConfig:
    return PipelineConfig(
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
        enrichment={
            "model_id": "test-metadata-model",
            "cache_path": tmp_path / "cache.sqlite3",
            **settings,
        },
    )


def response(text: str) -> dict[str, object]:
    return {
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }


def outcomes(output: Path) -> list[EnrichmentRecord]:
    return [
        EnrichmentRecord.model_validate_json(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_native_repair_cache_reuse_and_prefix_leave_citation_evidence_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = configured(tmp_path)
    chunks = source_chunks()
    before = chunks[0].model_dump_json()
    with (
        closing(
            boto3.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )
        ) as runtime,
        Stubber(runtime) as stub,
    ):
        expected = {
            "modelId": "test-metadata-model",
            "system": ANY,
            "messages": ANY,
            "inferenceConfig": {"maxTokens": 768, "temperature": 0},
        }
        stub.add_response("converse", response('{"topic":"invented-label"}'), expected)
        stub.add_response("converse", response(json.dumps(METADATA)), expected)
        first = run_enrichment(config, chunks, tmp_path / "first", runtime=runtime)
        assert first["model_calls"] == 2 and first["known_input_tokens"] == 20
        assert first["outcomes"] == {"enriched": 1} and first["model_cost_usd"] is None
        stub.assert_no_pending_responses()
    factory = Mock(side_effect=AssertionError("Cache reuse must not discover credentials"))
    monkeypatch.setattr("rag.enrichment.create_bedrock_client", factory)
    second = run_enrichment(config, chunks, tmp_path / "cached")
    assert second["model_calls"] == 0 and second["outcomes"] == {"cache_hit": 1}
    factory.assert_not_called()
    result = outcomes(tmp_path / "first")[0]
    assert embedding_input(chunks[0], result).endswith(chunks[0].original_text)
    assert "Inferred retrieval metadata (not evidence)" in embedding_input(chunks[0], result)
    context = assemble_context(
        [RetrievedHit(chunk=chunks[0], method="bm25", rank=1, score=1)],
        config.retrieval,
    )
    assert json.loads(context.text)[0]["text"] == chunks[0].original_text
    assert chunks[0].model_dump_json() == before
    for name, digest in first["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path / "first" / name).read_bytes()).hexdigest() == digest
    assert first["review_status"] == "unreviewed"
    changed = chunks[0].model_copy(update={"heading_path": ["A different source heading"]})
    with pytest.raises(ValueError, match="original chunk"):
        embedding_input(changed, result)
    # Changed source input and prompt versions must both miss the old valid cache.
    new_runtime = Mock()
    new_runtime.converse.return_value = response(json.dumps(METADATA))
    run_enrichment(config, [changed], tmp_path / "new-heading", runtime=new_runtime)
    monkeypatch.setattr("rag.enrichment.PROMPT_VERSION", "changed-prompt")
    run_enrichment(config, chunks, tmp_path / "new-prompt", runtime=new_runtime)
    assert new_runtime.converse.call_count == 2
    new_runtime.close.assert_not_called()


def test_shared_call_limit_preserves_all_chunks_and_never_caches_failed_metadata(
    tmp_path: Path,
) -> None:
    config = configured(tmp_path, max_model_calls=2)
    chunks = source_chunks(3)
    runtime = Mock()
    runtime.converse.return_value = response('{"topic":"invented-label"}')
    summary = run_enrichment(config, chunks, tmp_path / "failed", runtime=runtime)
    results = outcomes(tmp_path / "failed")
    assert summary["model_calls"] == 2 and runtime.converse.call_count == 2
    assert summary["outcomes"] == {"content_only": 3}
    assert [r.reason for r in results] == ["invalid_metadata", "call_limit", "call_limit"]
    assert [embedding_input(c, r) for c, r in zip(chunks, results, strict=True)] == [
        c.original_text for c in chunks
    ]
    runtime.converse.return_value = response(json.dumps(METADATA))
    retry = run_enrichment(config, chunks[:1], tmp_path / "retry", runtime=runtime)
    assert retry["model_calls"] == 1 and retry["cache_hits"] == 0


def test_late_provider_and_usage_failures_have_honest_content_only_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr("rag.enrichment.monotonic", lambda: clock[0])
    config = configured(tmp_path, timeout_seconds=1)
    chunks = source_chunks(2)
    late = Mock()

    def complete_late(**_: object) -> dict[str, object]:
        clock[0] = 2.0
        return response(json.dumps(METADATA))

    late.converse.side_effect = complete_late
    summary = run_enrichment(config, chunks, tmp_path / "late", runtime=late)
    assert summary["model_calls"] == 1 and summary["known_input_tokens"] == 10
    assert summary["usage_complete"] is True
    assert [r.reason for r in outcomes(tmp_path / "late")] == ["deadline", "deadline"]
    for name, value, reason in (
        ("usage", response(json.dumps(METADATA)) | {"usage": {}}, "invalid_usage"),
        (
            "provider",
            ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "private"}}, "Converse"
            ),
            "provider_error",
        ),
    ):
        runtime = Mock()
        if isinstance(value, Exception):
            runtime.converse.side_effect = value
        else:
            runtime.converse.return_value = value
        current = run_enrichment(config, chunks[:1], tmp_path / name, runtime=runtime)
        assert current["model_calls"] == 1 and current["usage_complete"] is False
        assert outcomes(tmp_path / name)[0].reason == reason
        assert "private" not in (tmp_path / name / "results.jsonl").read_text(encoding="utf-8")


def test_cli_rejects_invalid_intent_before_client_creation_and_closes_owned_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunks = source_chunks(2)
    chunk_path = tmp_path / "chunks.jsonl"
    chunk_path.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[cloud]\nregion="us-east-1"\ndevelopment_model_budget_usd=1\n'
        '[enrichment]\nmodel_id="test-model"\nmax_chunks=1\n'
        f'cache_path="{tmp_path / "cache.sqlite3"}"\n',
        encoding="utf-8",
    )
    factory = Mock()
    factory.return_value.converse.return_value = response(json.dumps(METADATA))
    monkeypatch.setattr("rag.enrichment.create_bedrock_client", factory)
    args = ["--config", str(config_path), "enrich", "--chunks", str(chunk_path)]
    for name, selection in (("too-many", []), ("unknown", ["--chunk-id", "missing"])):
        assert main([*args, *selection, "--output", str(tmp_path / name)]) == 1
        assert not (tmp_path / name).exists()
    factory.assert_not_called()
    configured_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        configured_text.replace('model_id="test-model"', 'model_id=""'), encoding="utf-8"
    )
    assert (
        main([*args, "--chunk-id", chunks[0].chunk_id, "--output", str(tmp_path / "missing-model")])
        == 1
    )
    factory.assert_not_called()
    assert not (tmp_path / "missing-model").exists()
    config_path.write_text(configured_text, encoding="utf-8")
    assert main([*args, "--chunk-id", chunks[0].chunk_id, "--output", str(tmp_path / "cli")]) == 0
    assert factory.call_args.kwargs == {"max_attempts": 1}
    factory.return_value.close.assert_called_once()
    factory.reset_mock()
    assert main([*args, "--chunk-id", chunks[0].chunk_id, "--output", str(tmp_path / "cli")]) == 1
    factory.assert_not_called()
    assert "already exists" in capsys.readouterr().err
