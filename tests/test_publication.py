"""A stale or tampered staging build must never replace the readable corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.config import PipelineConfig
from rag.indexing import index_definition, resolve_read_config
from rag.publication import SnapshotRecord, SnapshotStore, StaleRevisionError, publish_snapshot


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
