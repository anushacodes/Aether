"""A stale or tampered staging build must never replace the readable corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.artifacts import build_manifest, pipeline_fingerprint
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.corpus import load_sources
from rag.indexing import index_definition, resolve_read_config
from rag.models import SectionRecord, SubsetManifest
from rag.publication import (
    SnapshotRecord,
    SnapshotStore,
    StaleRevisionError,
    build_snapshot,
    publish_snapshot,
)


def test_publication_rejects_stale_revision_and_changed_artifacts(tmp_path: Path) -> None:
    config = PipelineConfig()
    config.publication.state_path = tmp_path / "state.sqlite3"
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"build": "verified"}', encoding="utf-8")
    snapshot = SnapshotRecord(
        snapshot_id="1" * 32,
        index_name="s3-rag-snapshot-" + "1" * 32,
        build_id="2" * 64,
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        document_count=3,
        corpus_id="pilot-v1",
        has_vectors=False,
    )
    SnapshotStore(config.publication.state_path).register(snapshot)
    client = Mock()
    client.indices.get_mapping.return_value = {
        snapshot.index_name: {"mappings": index_definition(config)["mappings"]}
    }
    client.count.return_value = {"count": 3}
    client.indices.get_alias.return_value = {"newer-snapshot": {"aliases": {}}}
    with pytest.raises(StaleRevisionError):
        publish_snapshot(client, config, snapshot, base_revision="older-snapshot")
    client.indices.update_aliases.assert_not_called()
    manifest.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        publish_snapshot(client, config, snapshot, base_revision="newer-snapshot")
    client.indices.update_aliases.assert_not_called()


def test_request_pins_the_active_alias_without_mutating_shared_config() -> None:
    config = PipelineConfig()
    config.index.name = config.publication.alias
    client = Mock()
    client.indices.get_alias.return_value = {"s3-rag-snapshot-" + "1" * 32: {"aliases": {}}}
    pinned = resolve_read_config(client, config)
    assert pinned.index.name == "s3-rag-snapshot-" + "1" * 32
    assert config.index.name == config.publication.alias


def test_legacy_build_reuse_ignores_answer_settings_but_rejects_changed_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig()
    config.publication.state_path = tmp_path / "state.sqlite3"
    snapshot_id = "a" * 32
    config.index.name = f"s3-rag-snapshot-{snapshot_id}"
    section = SectionRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    )
    chunks = chunk_section(section, config.chunking)
    current = build_manifest(config, chunks, has_vectors=False, corpus_id=section.corpus_id)
    legacy = current | {"schema_version": 1, "pipeline_fingerprint": "old-query-settings"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    snapshot = SnapshotRecord(
        snapshot_id=snapshot_id,
        index_name=config.index.name,
        build_id=legacy["build_id"],
        manifest_path=path,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        document_count=len(chunks),
        corpus_id=section.corpus_id,
        has_vectors=False,
    )
    SnapshotStore(config.publication.state_path).register(snapshot)
    before = pipeline_fingerprint(config)
    config.workflow.max_model_calls = 1
    config.retrieval.context_passages = 1
    config.models.generation_model_id = "different-answer-model"
    assert pipeline_fingerprint(config) != before
    assert (
        build_manifest(config, chunks, has_vectors=False, corpus_id=section.corpus_id)["build_id"]
        == current["build_id"]
    )
    monkeypatch.setattr("rag.publication.extract_sections", lambda *args: [section])
    sources = load_sources(Path("data/manifests/sources.json"))
    subset = SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes())
    client = Mock()
    assert build_snapshot(client, config, sources, subset, tmp_path, snapshot_id) == snapshot
    client.bulk.assert_not_called()
    config.chunking.overlap = 2
    config.chunking.size = 10
    with pytest.raises(ValueError, match="different build inputs"):
        build_snapshot(client, config, sources, subset, tmp_path, snapshot_id)
