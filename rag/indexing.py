"""Define compatible OpenSearch corpus indexes without mixing embedding spaces."""

from __future__ import annotations

from opensearchpy import OpenSearch

from rag.config import PipelineConfig

INDEX_SCHEMA_VERSION = 1


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
        mapping = client.indices.get_mapping(index=name)
        if mapping[name]["mappings"].get("_meta") != _index_metadata(config):
            raise ValueError(f"Index {name!r} has incompatible model or preprocessing settings")
        return False
    client.indices.create(index=name, body=index_definition(config))
    return True
