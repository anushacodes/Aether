"""Resolve six controlled retrieval treatments without duplicating whole configurations.

Only mode, document embedding input and ranker enablement vary. Local artifact guards
and read-only search preflight run before any query-model client is constructed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from opensearchpy import OpenSearch
from pydantic import Field

from rag.aws_clients import require_reranking_settings
from rag.config import ModelConfig, PipelineConfig
from rag.enriched_index import load_enriched_inputs
from rag.indexing import MAX_CORPUS_CHUNKS, resolve_read_config, verify_index
from rag.models import ChunkRecord, Digest, Record

PRESET_VERSION = 1
RetrievalMode = Literal["bm25", "vector", "hybrid"]


@dataclass(frozen=True)
class ExperimentPreset:
    """Keep one treatment definition separate from run-specific models and index bindings."""

    identifier: str
    mode: RetrievalMode
    metadata_vectors: bool = False
    reranking: bool = False


PRESETS = (
    ExperimentPreset("B0", "bm25"),
    ExperimentPreset("B1", "vector"),
    ExperimentPreset("B2", "hybrid"),
    ExperimentPreset("B3", "hybrid", metadata_vectors=True),
    ExperimentPreset("B4", "hybrid", reranking=True),
    ExperimentPreset("B5", "hybrid", metadata_vectors=True, reranking=True),
)


class EnrichedIndexRecord(Record):
    """The existing index builder's complete successful record; this is not a signature."""

    schema_version: Literal[1]
    index_name: str = Field(pattern=r"^s3-rag-experiment-[0-9a-f]{32}$")
    enrichment_fingerprint: Digest
    source_run: Path
    source_artifact_sha256: dict[str, Digest]
    review_status: Literal["unreviewed"]
    has_vectors: bool = Field(strict=True)
    models: ModelConfig
    chunker_versions: list[str] = Field(min_length=1)
    embedding_preprocessing_version: str
    document_count: int = Field(gt=0, le=MAX_CORPUS_CHUNKS, strict=True)
    enriched_count: int = Field(ge=0, le=MAX_CORPUS_CHUNKS, strict=True)
    embedding_requests: int = Field(ge=0, strict=True)
    embedding_cache_hits: int = Field(ge=0, strict=True)
    embedding_input_tokens: int = Field(ge=0, strict=True)


@dataclass(frozen=True)
class ExperimentSelection:
    """A copied effective configuration and nonsecret annotation for its saved run."""

    config: PipelineConfig
    preset: ExperimentPreset
    provenance: dict[str, object]


def preset_manifest() -> dict[str, object]:
    """Describe treatments offline without loading labels, discovering credentials or querying."""
    return {
        "schema_version": PRESET_VERSION,
        "metadata_boost": 0,
        "reranking_failure_policy": "error",
        "experiments": [asdict(preset) for preset in PRESETS],
    }


def resolve_experiment(
    identifier: str,
    config: PipelineConfig,
    chunks: list[ChunkRecord],
    enriched_index: Path | None = None,
) -> ExperimentSelection:
    """Validate a treatment and exact metadata inputs locally; leave the base config untouched."""
    preset = next((preset for preset in PRESETS if preset.identifier == identifier), None)
    if preset is None:
        raise ValueError("Choose an experiment from B0–B5")
    if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Experiment requires nonempty unique original chunks")
    if config.index.enrichment_fingerprint is not None:
        raise ValueError("Experiment base configuration must target the original baseline")
    if preset.metadata_vectors != (enriched_index is not None):
        raise ValueError("Only B3/B5 require --enriched-index with a successful index.json")
    selected = config.model_copy(deep=True)
    selected.retrieval.metadata_boost = 0
    selected.reranking.enabled = preset.reranking
    selected.reranking.failure_policy = "error"
    provenance: dict[str, object] = {
        "schema_version": PRESET_VERSION,
        "definition": asdict(preset),
        "metadata_boost": 0,
        "reranking_failure_policy": "error",
    }
    if enriched_index is not None:
        raw_record = enriched_index.read_bytes()
        record = EnrichedIndexRecord.model_validate_json(raw_record)
        inputs = load_enriched_inputs(record.source_run)
        fingerprint = hashlib.sha256(
            json.dumps(inputs.artifact_sha256, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if (
            record.source_artifact_sha256 != inputs.artifact_sha256
            or record.enrichment_fingerprint != fingerprint
            or not record.has_vectors
            or record.document_count != len(chunks)
            or record.enriched_count != sum(row.metadata is not None for row in inputs.records)
            or record.chunker_versions != sorted({chunk.chunker_version for chunk in chunks})
            or record.embedding_preprocessing_version != config.chunking.preprocessing_version
            or record.models.model_dump(exclude={"generation_model_id"})
            != config.models.model_dump(exclude={"generation_model_id"})
            or {chunk.chunk_id: chunk for chunk in inputs.chunks}
            != {chunk.chunk_id: chunk for chunk in chunks}
        ):
            raise ValueError("Enriched index record differs from the shared model/chunk inputs")
        prefixed = sum(
            text != chunk.original_text
            for chunk, text in zip(inputs.chunks, inputs.embedding_texts, strict=True)
        )
        if not prefixed:
            raise ValueError("Metadata-vector treatment requires at least one changed prefix")
        if enriched_index.read_bytes() != raw_record or any(
            hashlib.sha256((record.source_run / name).read_bytes()).hexdigest() != digest
            for name, digest in inputs.artifact_sha256.items()
        ):
            raise ValueError("Experiment inputs changed during validation")
        selected.index.name = record.index_name
        selected.index.enrichment_fingerprint = fingerprint
        provenance |= {
            "enriched_index_record_sha256": hashlib.sha256(raw_record).hexdigest(),
            "source_artifact_sha256": inputs.artifact_sha256,
            "prefixed_chunks": prefixed,
            "document_count": record.document_count,
            "metadata_review_status": record.review_status,
        }
    if preset.reranking:
        require_reranking_settings(selected)
    return ExperimentSelection(selected, preset, provenance)


def preflight_experiment(
    client: OpenSearch, selection: ExperimentSelection, chunks: list[ChunkRecord]
) -> ExperimentSelection:
    """Pin and inspect an exact immutable corpus before query embeddings or ranker setup."""
    pinned = resolve_read_config(client, selection.config)
    verify_index(client, pinned, require_inferred_metadata=selection.preset.metadata_vectors)
    settings = client.indices.get_settings(index=pinned.index.name)
    blocked = settings[pinned.index.name]["settings"]["index"].get("blocks", {}).get("write")
    if blocked is not True and blocked != "true":
        raise ValueError("Experiment index must be write-blocked")
    response = client.search(
        index=pinned.index.name,
        body={
            "size": MAX_CORPUS_CHUNKS + 1,
            "track_total_hits": True,
            "query": {"match_all": {}},
            "_source": ["record"],
        },
    )
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise ValueError("Experiment corpus inspection was incomplete")
    stored: dict[str, ChunkRecord] = {}
    for hit in response["hits"]["hits"]:
        chunk = ChunkRecord.model_validate(hit["_source"]["record"])
        if chunk.chunk_id != hit["_id"] or chunk.chunk_id in stored:
            raise ValueError("Experiment index has inconsistent source identities")
        stored[chunk.chunk_id] = chunk
    if stored != {chunk.chunk_id: chunk for chunk in chunks}:
        raise ValueError("Experiment index must contain exactly the shared original chunks")
    if selection.preset.mode != "bm25":
        vector_count = client.count(
            index=pinned.index.name, body={"query": {"exists": {"field": "embedding"}}}
        )["count"]
        if vector_count != len(chunks):
            raise ValueError("Experiment requires complete document-vector coverage")
    return replace(
        selection,
        config=pinned,
        provenance=selection.provenance | {"preflight_index": pinned.index.name},
    )
