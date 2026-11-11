"""Restore retained snapshots without rebuilding or deleting indexes used by active readers."""

from __future__ import annotations

from datetime import datetime

from opensearchpy import OpenSearch

from rag.config import PipelineConfig
from rag.ingestion import snapshot_inputs
from rag.models import Record
from rag.publication import SnapshotRecord, SnapshotStore, active_index, publish_snapshot


class RetainedSnapshot(Record):
    """Publication history is distinct from the alias's current target."""

    snapshot: SnapshotRecord
    first_published_at: datetime
    active: bool


def retained_snapshots(client: OpenSearch, config: PipelineConfig) -> list[RetainedSnapshot]:
    """List previously published local builds; retention currently has no automatic pruning."""
    store = SnapshotStore(config.publication.state_path)
    current = active_index(client, config.publication.alias)
    with store.transaction() as connection:
        rows = connection.execute(
            "SELECT payload, published_at FROM snapshots WHERE published_at IS NOT NULL "
            "ORDER BY published_at DESC, snapshot_id"
        ).fetchall()
    records = []
    for payload, published_at in rows:
        snapshot = SnapshotRecord.model_validate_json(payload)
        records.append(
            RetainedSnapshot(
                snapshot=snapshot,
                first_published_at=datetime.fromisoformat(published_at),
                active=snapshot.index_name == current,
            )
        )
    return records


def rollback_snapshot(
    client: OpenSearch,
    config: PipelineConfig,
    snapshot_id: str,
    base_revision: str,
) -> SnapshotRecord:
    """Restore an explicitly selected published build under the normal stale-base fence.

    This restores the target's evidence, including passages deleted by later publications.
    It retains the replaced index so requests pinned there can finish safely.
    """
    store = SnapshotStore(config.publication.state_path)
    target = store.get(snapshot_id)
    if target is None or not store.was_published(snapshot_id):
        raise ValueError("Rollback requires a previously published snapshot")
    snapshot_inputs(target)
    settings = client.indices.get_settings(
        index=target.index_name, name="index.blocks.write", flat_settings=True
    )
    blocked = settings.get(target.index_name, {}).get("settings", {}).get("index.blocks.write")
    if blocked not in (True, "true"):
        raise ValueError("Rollback target must remain write-blocked")
    vector_count = client.count(
        index=target.index_name, body={"query": {"exists": {"field": "embedding"}}}
    )["count"]
    if vector_count != (target.document_count if target.has_vectors else 0):
        raise ValueError("Rollback target has incomplete or unexpected vector coverage")
    # Publication rechecks manifest identity, index compatibility/count, and the active base.
    publish_snapshot(client, config, target, base_revision)
    return target
