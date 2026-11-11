"""Resolve approved local mutations into immutable, replayable ingestion inputs."""

from __future__ import annotations

import hashlib
import re

from opensearchpy import OpenSearch

from rag.config import PipelineConfig
from rag.corpus import load_sources
from rag.jobs import IngestionJob, IngestionRequest, JobConflict, JobInputs, JobStore
from rag.models import SourceManifest, SubsetManifest
from rag.publication import SnapshotRecord, SnapshotStore, active_index


def snapshot_inputs(record: SnapshotRecord) -> tuple[SourceManifest, SubsetManifest]:
    """Verify persisted input artifacts before deriving a deletion from an old snapshot."""
    if hashlib.sha256(record.manifest_path.read_bytes()).hexdigest() != record.manifest_sha256:
        raise ValueError("Snapshot manifest hash mismatch")
    required = {"sources.json", "subset.json", "sections.jsonl", "chunks.jsonl"}
    if set(record.artifact_sha256) != required:
        raise JobConflict("legacy_snapshot_requires_rebuild")
    for name in required:
        content = (record.manifest_path.parent / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != record.artifact_sha256[name]:
            raise ValueError("Snapshot artifact hash mismatch")
    sources = SourceManifest.model_validate_json(
        (record.manifest_path.parent / "sources.json").read_bytes()
    )
    subset = SubsetManifest.model_validate_json(
        (record.manifest_path.parent / "subset.json").read_bytes()
    )
    return sources, subset


def ingestion_catalog(client: OpenSearch, config: PipelineConfig) -> dict[str, object]:
    """Show approved rebuilds and actual pinned sections available for deletion."""
    subsets = {
        identifier: SubsetManifest.model_validate_json(path.read_bytes())
        for identifier, path in config.ingestion.approved_subsets.items()
    }
    current = active_index(client, config.publication.alias)
    record = SnapshotStore(config.publication.state_path).for_index(current or "")
    active: dict[str, object] | None = None
    if record is not None:
        active = {
            "snapshot_id": record.snapshot_id,
            "corpus_id": record.corpus_id,
            "document_count": record.document_count,
            "has_vectors": record.has_vectors,
            "artifact_state": "legacy-rebuild-required",
            "sections": None,
        }
        if record.artifact_sha256:
            _, subset = snapshot_inputs(record)
            active["artifact_state"] = "verified"
            active["sections"] = [
                {"section_id": section.section_id, "heading_path": section.heading_path}
                for section in subset.sections
            ]
    return {
        "active_revision": current,
        "active_snapshot": active,
        "catalog": {
            identifier: {
                "corpus_id": subset.corpus_id,
                "sections": [
                    {"section_id": section.section_id, "heading_path": section.heading_path}
                    for section in subset.sections
                ],
            }
            for identifier, subset in subsets.items()
        },
    }


def submit_ingestion(
    client: OpenSearch,
    config: PipelineConfig,
    store: JobStore,
    key: str,
    request: IngestionRequest,
) -> IngestionJob:
    """Accept durable intent without downloading sources, embedding, or building an index."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", key):
        raise ValueError(
            "Idempotency keys must contain 1–128 letters, numbers, dots, underscores, or dashes"
        )
    existing = store.existing(key, request)
    if existing is not None:
        return existing
    if active_index(client, config.publication.alias) != request.base_revision:
        raise JobConflict("stale_base_revision")
    if request.operation == "rebuild":
        path = config.ingestion.approved_subsets.get(request.catalog_id)
        if path is None:
            raise ValueError("Unknown approved catalog ID")
        sources = load_sources(config.ingestion.sources_path)
        subset = SubsetManifest.model_validate_json(path.read_bytes())
    else:
        record = SnapshotStore(config.publication.state_path).for_index(request.base_revision or "")
        if record is None:
            raise JobConflict("base_snapshot_not_registered")
        if request.with_embeddings != record.has_vectors:
            raise JobConflict("deletion_must_preserve_vector_coverage")
        sources, subset = snapshot_inputs(record)
        removed = set(request.remove_section_ids)
        known = {section.section_id for section in subset.sections}
        if not removed <= known:
            raise ValueError("Deletion includes unknown section IDs")
        values = subset.model_dump()
        values["sections"] = [
            section for section in subset.sections if section.section_id not in removed
        ]
        # Derive identity from accepted intent; repeated preparation stays deterministic.
        suffix = hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()[:12]
        values["corpus_id"] = f"s3-delete-{suffix}"
        subset = SubsetManifest.model_validate(values)
    if request.with_embeddings:
        config.require_cloud_settings(require_generation=False)
    inputs = JobInputs(
        sources=sources,
        subset=subset,
        target_alias=config.publication.alias,
        chunking=config.chunking,
        models=config.models,
        retrieval=config.retrieval,
        generation=config.generation,
    )
    return store.submit(key, request, inputs, max_pending=config.ingestion.max_pending_jobs)
