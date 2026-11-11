"""Build immutable local corpus snapshots and publish them through one atomic alias switch."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from opensearchpy import NotFoundError, OpenSearch
from pydantic import Field

from rag.artifacts import build_manifest, save_build_manifest
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.corpus import extract_sections, write_jsonl
from rag.embeddings import Embedder
from rag.indexing import bulk_index_chunks, ensure_index, verify_index
from rag.models import Digest, Record, SourceManifest, SubsetManifest

SQLITE_TIMEOUT_SECONDS = 30


class StaleRevisionError(ValueError):
    """A corpus changed after a mutation chose its base revision."""


class SnapshotRecord(Record):
    """A completed immutable build whose files and index can be verified before publication."""

    snapshot_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    index_name: str = Field(pattern=r"^s3-rag-snapshot-[0-9a-f]{32}$")
    build_id: Digest
    manifest_path: Path
    manifest_sha256: Digest
    document_count: int = Field(ge=0)
    corpus_id: str
    has_vectors: bool
    artifact_sha256: dict[str, Digest] = Field(default_factory=dict)


class SnapshotStore:
    """Register completed builds and serialize publishers across local worker processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS snapshots "
                "(snapshot_id TEXT PRIMARY KEY, payload TEXT NOT NULL, published_at TEXT)"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold a local writer lock until publication finishes or rolls back."""
        with closing(sqlite3.connect(self.path, timeout=SQLITE_TIMEOUT_SECONDS)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                yield connection

    def get(self, snapshot_id: str) -> SnapshotRecord | None:
        """Recover an already completed attempt without rebuilding or re-embedding it."""
        with closing(sqlite3.connect(self.path, timeout=SQLITE_TIMEOUT_SECONDS)) as connection:
            row = connection.execute(
                "SELECT payload FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        return SnapshotRecord.model_validate_json(row[0]) if row else None

    def register(self, record: SnapshotRecord) -> None:
        """Persist a ready build before an alias can expose it to search."""
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO snapshots(snapshot_id, payload) VALUES (?, ?)",
                (record.snapshot_id, record.model_dump_json()),
            )

    def for_index(self, index_name: str) -> SnapshotRecord | None:
        """Locate the persisted manifests behind an active concrete snapshot."""
        with closing(sqlite3.connect(self.path, timeout=SQLITE_TIMEOUT_SECONDS)) as connection:
            row = connection.execute(
                "SELECT payload FROM snapshots WHERE json_extract(payload, '$.index_name') = ?",
                (index_name,),
            ).fetchone()
        return SnapshotRecord.model_validate_json(row[0]) if row else None

    def was_published(self, snapshot_id: str) -> bool:
        """Reconcile a worker crash after publication but before its job was marked ready."""
        with closing(sqlite3.connect(self.path, timeout=SQLITE_TIMEOUT_SECONDS)) as connection:
            row = connection.execute(
                "SELECT published_at FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        return row is not None and row[0] is not None


def active_index(client: OpenSearch, alias: str) -> str | None:
    """Resolve exactly one published snapshot; an absent alias has no base revision."""
    try:
        aliases = client.indices.get_alias(name=alias)
    except NotFoundError:
        return None
    if len(aliases) != 1:
        raise ValueError("The active corpus alias must reference exactly one index")
    return next(iter(aliases))


def build_snapshot(
    client: OpenSearch,
    config: PipelineConfig,
    sources: SourceManifest,
    subset: SubsetManifest,
    raw_dir: Path,
    snapshot_id: str,
    *,
    embedder: Embedder | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> SnapshotRecord:
    """Verify sources, build separately, and register only complete immutable artifacts."""
    if not re.fullmatch(r"[0-9a-f]{32}", snapshot_id):
        raise ValueError("Snapshot IDs must be generated UUID hex values")
    if checkpoint is not None:
        checkpoint()
    sections = extract_sections(sources, subset, raw_dir)
    chunks = [chunk for section in sections for chunk in chunk_section(section, config.chunking)]
    staged = config.model_copy(deep=True)
    staged.index.name = f"s3-rag-snapshot-{snapshot_id}"
    manifest = build_manifest(
        staged, chunks, has_vectors=embedder is not None, corpus_id=subset.corpus_id
    )
    store = SnapshotStore(config.publication.state_path)
    existing = store.get(snapshot_id)
    if existing is not None:
        if existing.build_id != manifest["build_id"]:
            raise ValueError("A snapshot ID cannot be reused for different build inputs")
        return existing
    if client.indices.exists(index=staged.index.name):
        aliases = client.indices.get_alias(index=staged.index.name)
        if any(value.get("aliases") for value in aliases.values()):
            raise ValueError("Refusing to rebuild an index referenced by an alias")
        client.indices.delete(index=staged.index.name)
    if checkpoint is not None:
        checkpoint()
    vectors = embedder.embed_chunks(chunks, checkpoint=checkpoint) if embedder else None
    if checkpoint is not None:
        checkpoint()
    if chunks:
        count = bulk_index_chunks(client, staged, chunks, vectors)
    else:
        ensure_index(client, staged)
        count = client.count(index=staged.index.name)["count"]
        if count != 0:
            raise ValueError("An empty snapshot must have no indexed documents")
    vector_count = client.count(
        index=staged.index.name, body={"query": {"exists": {"field": "embedding"}}}
    )["count"]
    if vector_count != (count if vectors is not None else 0):
        raise ValueError("Staging index has incomplete or unexpected vector coverage")
    output = config.publication.snapshot_dir / snapshot_id
    output.mkdir(parents=True, exist_ok=True)
    save_build_manifest(output / "sources.json", sources.model_dump(mode="json"))
    save_build_manifest(output / "subset.json", subset.model_dump(mode="json"))
    write_jsonl(
        output / "sections.jsonl", (section.model_dump(mode="json") for section in sections)
    )
    write_jsonl(output / "chunks.jsonl", (chunk.model_dump(mode="json") for chunk in chunks))
    artifact_hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("sources.json", "subset.json", "sections.jsonl", "chunks.jsonl")
    }
    manifest["artifact_sha256"] = artifact_hashes
    save_build_manifest(output / "manifest.json", manifest)
    if checkpoint is not None:
        checkpoint()
    client.indices.put_settings(index=staged.index.name, body={"index.blocks.write": True})
    record = SnapshotRecord(
        snapshot_id=snapshot_id,
        index_name=staged.index.name,
        build_id=manifest["build_id"],
        manifest_path=output / "manifest.json",
        manifest_sha256=hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        document_count=count,
        corpus_id=subset.corpus_id,
        has_vectors=vectors is not None,
        artifact_sha256=artifact_hashes,
    )
    store.register(record)
    return record


def publish_snapshot(
    client: OpenSearch, config: PipelineConfig, record: SnapshotRecord, base_revision: str | None
) -> None:
    """Atomically replace a compatible alias only if the caller's base is still current."""
    store = SnapshotStore(config.publication.state_path)
    with store.transaction() as connection:
        registered = connection.execute(
            "SELECT payload FROM snapshots WHERE snapshot_id = ?", (record.snapshot_id,)
        ).fetchone()
        if registered is None or SnapshotRecord.model_validate_json(registered[0]) != record:
            raise ValueError("Snapshot must be registered before publication")
        if hashlib.sha256(record.manifest_path.read_bytes()).hexdigest() != record.manifest_sha256:
            raise ValueError("Snapshot manifest hash mismatch")
        staged = config.model_copy(deep=True)
        staged.index.name = record.index_name
        verify_index(client, staged)
        if client.count(index=record.index_name)["count"] != record.document_count:
            raise ValueError("Snapshot index count changed before publication")
        current = active_index(client, config.publication.alias)
        if current != record.index_name:
            if current != base_revision:
                raise StaleRevisionError(
                    "Active corpus changed; submit a mutation against the current revision"
                )
            actions: list[dict[str, object]] = []
            if current is not None:
                actions.append(
                    {
                        "remove": {
                            "index": current,
                            "alias": config.publication.alias,
                            "must_exist": True,
                        }
                    }
                )
            actions.append(
                {
                    "add": {
                        "index": record.index_name,
                        "alias": config.publication.alias,
                        "is_write_index": False,
                    }
                }
            )
            client.indices.update_aliases(body={"actions": actions})
            if active_index(client, config.publication.alias) != record.index_name:
                raise ValueError("Alias publication could not be confirmed")
        connection.execute(
            "UPDATE snapshots SET published_at = COALESCE(published_at, ?) WHERE snapshot_id = ?",
            (datetime.now(UTC).isoformat(), record.snapshot_id),
        )
