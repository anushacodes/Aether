"""Keep six treatments controlled, bind metadata to originals, and guard model setup."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from opensearchpy import NotFoundError

from rag.chunking import chunk_section
from rag.cli import main
from rag.config import PipelineConfig
from rag.corpus import write_jsonl
from rag.embeddings import Embedder
from rag.enriched_index import build_enriched_index
from rag.enrichment import run_enrichment
from rag.eval.experiments import PRESETS, preflight_experiment, resolve_experiment
from rag.indexing import index_definition
from rag.models import ChunkRecord, SectionRecord


def enriched_binding(
    tmp_path: Path, *, descriptive: bool = True
) -> tuple[PipelineConfig, list[ChunkRecord], Path]:
    """Create genuine metadata/index record formats with local provider/search doubles."""
    chunk = ChunkRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    )
    config = PipelineConfig(
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
        models={"embedding_dimensions": 256},
        embedding={"cache_path": tmp_path / "vectors.sqlite3"},
        enrichment={"model_id": "test-model", "cache_path": tmp_path / "metadata.sqlite3"},
        reranking={"model_arn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"},
    )
    provider = Mock()
    provider.converse.return_value = {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "content": [
                    {
                        "text": json.dumps(
                            {
                                "summary": "Metadata treatment" if descriptive else None,
                                "topic": "versioning" if descriptive else "unknown",
                                "intent": "concept" if descriptive else "unknown",
                                "related_concepts": [],
                                "example_phrasings": [],
                            }
                        )
                    }
                ]
            }
        },
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }
    provider.invoke_model.return_value = {
        "body": io.BytesIO(
            json.dumps(
                {
                    "embedding": [1.0] * 256,
                    "inputTextTokenCount": 1,
                }
            ).encode()
        )
    }
    run = tmp_path / "metadata"
    run_enrichment(config, [chunk], run, runtime=provider)
    client = Mock()
    client.indices.exists.return_value = False
    client.indices.exists_alias.return_value = False
    client.bulk.return_value = {"items": [{"index": {"_id": chunk.chunk_id, "status": 201}}]}
    client.count.return_value = {"count": 1}
    output = tmp_path / "index"
    build_enriched_index(client, config, run, output, embedder=Embedder(config, provider))
    return config, [chunk], output / "index.json"


def test_six_presets_are_offline_and_change_only_declared_treatments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    search, model, labels = Mock(), Mock(), Mock()
    monkeypatch.setattr("rag.cli.create_search_client", search)
    monkeypatch.setattr("rag.cli.create_bedrock_client", model)
    monkeypatch.setattr("rag.cli.load_dataset", labels)
    assert main(["eval", "experiments"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert [preset["identifier"] for preset in manifest["experiments"]] == [
        f"B{i}" for i in range(6)
    ]
    assert not search.called and not model.called and not labels.called
    config, chunks, binding = enriched_binding(tmp_path)
    config.retrieval.metadata_boost = 3
    config.reranking.enabled = True
    config.reranking.failure_policy = "original_order"
    config.models.generation_model_id = "generation-changed-after-indexing"
    original = config.model_dump()
    for preset in PRESETS:
        selected = resolve_experiment(
            preset.identifier, config, chunks, binding if preset.metadata_vectors else None
        )
        expected = config.model_copy(deep=True)
        expected.retrieval.metadata_boost = 0
        expected.reranking.enabled = preset.reranking
        expected.reranking.failure_policy = "error"
        if preset.metadata_vectors:
            record = json.loads(binding.read_text(encoding="utf-8"))
            expected.index.name = record["index_name"]
            expected.index.enrichment_fingerprint = record["enrichment_fingerprint"]
            assert selected.provenance["prefixed_chunks"] == 1
        assert selected.config == expected and config.model_dump() == original
        assert selected.provenance["definition"]["mode"] == preset.mode


def test_enriched_treatments_reject_unbound_or_incompatible_records_before_services(
    tmp_path: Path,
) -> None:
    config, chunks, binding = enriched_binding(tmp_path)
    for identifier, supplied in [("B3", None), ("B0", binding)]:
        with pytest.raises(ValueError, match="Only B3/B5"):
            resolve_experiment(identifier, config, chunks, supplied)
    original = binding.read_bytes()
    record = json.loads(original)
    for changed in [
        {"has_vectors": False},
        {"enrichment_fingerprint": "0" * 64},
        {"embedding_preprocessing_version": "different"},
        {"models": record["models"] | {"embedding_dimensions": 512}},
    ]:
        binding.write_text(json.dumps(record | changed), encoding="utf-8")
        with pytest.raises(ValueError, match="shared model/chunk"):
            resolve_experiment("B3", config, chunks, binding)
    binding.write_bytes(original)
    altered = chunks[0].model_copy(update={"original_text": chunks[0].original_text + " changed"})
    with pytest.raises(ValueError, match="shared model/chunk"):
        resolve_experiment("B3", config, [altered], binding)
    config.index.enrichment_fingerprint = "a" * 64
    with pytest.raises(ValueError, match="original baseline"):
        resolve_experiment("B2", config, chunks)
    empty_config, empty_chunks, empty_binding = enriched_binding(
        tmp_path / "no-hints", descriptive=False
    )
    with pytest.raises(ValueError, match="changed prefix"):
        resolve_experiment("B3", empty_config, empty_chunks, empty_binding)


def test_cli_preflight_rejects_mutable_wrong_or_incomplete_indexes_before_model_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PipelineConfig()
    section = SectionRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    )
    chunks = chunk_section(section, config.chunking)
    sections_file, chunks_file = tmp_path / "sections.jsonl", tmp_path / "chunks.jsonl"
    write_jsonl(sections_file, [section.model_dump(mode="json")])
    write_jsonl(chunks_file, [chunk.model_dump(mode="json") for chunk in chunks])
    client = MagicMock()
    client.__enter__.return_value = client
    client.indices.get_alias.side_effect = NotFoundError(404, "no alias")
    client.indices.get_mapping.return_value = {
        config.index.name: {"mappings": index_definition(config)["mappings"]}
    }
    complete = {
        "hits": {
            "hits": [
                {"_id": chunk.chunk_id, "_source": {"record": chunk.model_dump(mode="json")}}
                for chunk in chunks
            ]
        }
    }
    model = Mock()
    monkeypatch.setattr("rag.cli.load_config", Mock(return_value=config))
    monkeypatch.setattr("rag.cli.create_search_client", Mock(return_value=client))
    monkeypatch.setattr("rag.cli.create_bedrock_client", model)
    for name, blocked, response, vectors in [
        ("writable", False, complete, len(chunks)),
        ("missing", True, {"hits": {"hits": []}}, len(chunks)),
        ("partial", True, complete | {"timed_out": True}, len(chunks)),
        ("vectors", True, complete, 0),
    ]:
        client.indices.get_settings.return_value = {
            config.index.name: {"settings": {"index": {"blocks": {"write": blocked}}}}
        }
        client.search.return_value = response
        client.count.return_value = {"count": vectors}
        output = tmp_path / name
        assert (
            main(
                [
                    "eval",
                    "retrieval",
                    "--experiment",
                    "B1",
                    "--allow-drafts",
                    "--sections",
                    str(sections_file),
                    "--chunks",
                    str(chunks_file),
                    "--output",
                    str(output),
                ]
            )
            == 1
        )
        assert not output.exists()
    model.assert_not_called()
    client.indices.get_settings.return_value = {
        config.index.name: {"settings": {"index": {"blocks": {"write": "true"}}}}
    }
    client.search.return_value = complete
    client.count.return_value = {"count": len(chunks)}
    selected = preflight_experiment(client, resolve_experiment("B1", config, chunks), chunks)
    assert selected.provenance["preflight_index"] == config.index.name
