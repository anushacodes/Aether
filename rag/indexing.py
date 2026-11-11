"""Define compatible OpenSearch corpus indexes without mixing embedding spaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from opensearchpy import NotFoundError, OpenSearch

from rag.config import PipelineConfig
from rag.embeddings import validate_embedding
from rag.models import ChunkRecord

INDEX_SCHEMA_VERSION = 1
MAX_CORPUS_CHUNKS = 500


def _index_metadata(config: PipelineConfig) -> dict[str, str | int | bool]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model_id": config.models.embedding_model_id,
        "embedding_dimensions": config.models.embedding_dimensions,
        "normalize": config.models.normalize,
        "preprocessing_version": config.chunking.preprocessing_version,
    }


def index_definition(config: PipelineConfig) -> dict[str, object]:
    """Map searchable text and authoritative filters separately from full evidence records."""
    return {
        "settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "dynamic": "strict",
            "_meta": _index_metadata(config),
            "properties": {
                "original_text": {"type": "text"},
                "heading_text": {"type": "text"},
                "guide": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "section_id": {"type": "keyword"},
                "corpus_id": {"type": "keyword"},
                "source_scope": {"type": "keyword"},
                "chunker_version": {"type": "keyword"},
                "record": {"type": "object", "enabled": False},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": config.models.embedding_dimensions,
                    "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
                },
            },
        },
    }


def ensure_index(client: OpenSearch, config: PipelineConfig) -> bool:
    """Create the configured index, or reject incompatible existing embedding metadata."""
    name = config.index.name
    if client.indices.exists(index=name):
        verify_index(client, config)
        return False
    client.indices.create(index=name, body=index_definition(config))
    return True


def verify_index(client: OpenSearch, config: PipelineConfig) -> None:
    """Reject query/index embedding-space mismatches without creating or changing an index."""
    mapping = client.indices.get_mapping(index=config.index.name)
    # An alias resolves to its concrete index name in get_mapping responses.
    if len(mapping) != 1:
        raise ValueError("Search requires exactly one compatible corpus snapshot")
    actual = next(iter(mapping.values()))["mappings"].get("_meta")
    expected = _index_metadata(config)
    if not isinstance(actual, dict) or any(
        actual.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            f"Index {config.index.name!r} has incompatible model or preprocessing settings"
        )


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Read evidence records and reject duplicate identities before publishing any data."""
    chunks = [
        ChunkRecord.model_validate_json(line)
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]
    if not chunks:
        raise ValueError("Chunk input is empty")
    if len(chunks) > MAX_CORPUS_CHUNKS:
        raise ValueError(
            f"Corpus exceeds the {MAX_CORPUS_CHUNKS}-chunk limit; reduce coherent scope"
        )
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk input contains duplicate IDs")
    return chunks


def resolve_read_config(client: OpenSearch, config: PipelineConfig) -> PipelineConfig:
    """Pin a logical request to one concrete index before concurrent queries can cross updates."""
    try:
        aliases = client.indices.get_alias(name=config.index.name)
    except NotFoundError:
        return config
    if len(aliases) != 1:
        raise ValueError("Search requires one active corpus snapshot")
    resolved = config.model_copy(deep=True)
    resolved.index.name = next(iter(aliases))
    return resolved


def bulk_index_chunks(
    client: OpenSearch,
    config: PipelineConfig,
    chunks: list[ChunkRecord],
    embeddings: Mapping[str, list[float]] | None = None,
) -> int:
    """Upsert stable IDs, inspect every bulk item, and verify the full corpus count."""
    if client.indices.exists_alias(name=config.index.name):
        raise ValueError("Active aliases are immutable; build and publish a new snapshot")
    if not chunks or len(chunks) > MAX_CORPUS_CHUNKS:
        raise ValueError(f"Indexing requires between 1 and {MAX_CORPUS_CHUNKS} chunks")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Cannot index duplicate chunk IDs")
    operations: list[dict[str, object]] = []
    for chunk in chunks:
        document = {
            "original_text": chunk.original_text,
            "heading_text": " > ".join(chunk.heading_path),
            "guide": chunk.provenance.guide,
            "source_id": chunk.provenance.source_id,
            "section_id": chunk.section_id,
            "corpus_id": chunk.corpus_id,
            "source_scope": chunk.source_scope,
            "chunker_version": chunk.chunker_version,
            "record": chunk.model_dump(mode="json"),
        }
        if embeddings is not None:
            if chunk.chunk_id not in embeddings:
                raise ValueError(f"Missing embedding for {chunk.chunk_id}")
            document["embedding"] = validate_embedding(
                embeddings[chunk.chunk_id], config.models.embedding_dimensions
            )
        operations.extend(
            [{"index": {"_index": config.index.name, "_id": chunk.chunk_id}}, document]
        )
    ensure_index(client, config)
    response = client.bulk(body=operations, refresh="wait_for")
    failures = [
        {"chunk_id": item["index"].get("_id"), "status": item["index"].get("status")}
        for item in response["items"]
        if item["index"].get("error") or item["index"].get("status", 500) >= 300
    ]
    if failures or len(response["items"]) != len(chunks):
        raise ValueError(f"Bulk indexing failed or returned incomplete results: {failures[:10]}")
    count = client.count(index=config.index.name)["count"]
    if count != len(chunks):
        raise ValueError(f"Index count {count} differs from {len(chunks)} expected unique chunks")
    return count
