"""Build isolated immutable experiment indexes from verified metadata-run artifacts.

No active alias is changed. Inferred text is searchable only through an explicit boost;
source records and citation context always retain their original content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from opensearchpy import OpenSearch

from rag.config import PipelineConfig
from rag.embeddings import Embedder
from rag.enrichment import (
    PREFIX_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    EnrichmentRecord,
    InferredMetadata,
    embedding_input,
)
from rag.indexing import bulk_index_chunks, load_chunks
from rag.models import ChunkRecord

ARTIFACT_FILES = {"manifest.json", "chunks.jsonl", "results.jsonl", "embedding-inputs.jsonl"}


@dataclass(frozen=True)
class EnrichedInputs:
    """An integrity-checked, unreviewed experiment input, not an authoritative source."""

    chunks: list[ChunkRecord]
    records: list[EnrichmentRecord]
    embedding_texts: list[str]
    artifact_sha256: dict[str, str]


def load_enriched_inputs(path: Path) -> EnrichedInputs:
    """Reject partial, mixed-version, altered, or mismatched runs before any model request."""
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Metadata summary must be an object")
    hashes = summary.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != ARTIFACT_FILES:
        raise ValueError("Metadata run must have complete fixed-name artifact hashes")
    for name in sorted(ARTIFACT_FILES):
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Metadata artifact hash mismatch: {name}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Metadata manifest must be an object")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("prefix_version") != PREFIX_VERSION
        or manifest.get("prompt_version") != PROMPT_VERSION
        or manifest.get("system_prompt") != SYSTEM_PROMPT
        or manifest.get("response_schema") != InferredMetadata.model_json_schema()
        or manifest.get("chunks_sha256") != hashes["chunks.jsonl"]
    ):
        raise ValueError("Unsupported or inconsistent metadata input manifest")
    chunks = load_chunks(path / "chunks.jsonl")
    records = [
        EnrichmentRecord.model_validate_json(line)
        for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    saved_inputs = [
        json.loads(line)
        for line in (path / "embedding-inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != len(chunks) or len(saved_inputs) != len(chunks):
        raise ValueError("Metadata run must cover every original chunk exactly once")
    texts = []
    for chunk, record, saved in zip(chunks, records, saved_inputs, strict=True):
        text = embedding_input(chunk, record)
        if saved != {"chunk_id": chunk.chunk_id, "text": text}:
            raise ValueError("Saved embedding input differs from verified source and metadata")
        texts.append(text)
    return EnrichedInputs(chunks, records, texts, hashes)


def build_enriched_index(
    client: OpenSearch,
    config: PipelineConfig,
    run: Path,
    output: Path,
    *,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    """Keep baseline aliases intact while indexing metadata separately from evidence records."""
    inputs = load_enriched_inputs(run)
    if embedder is not None and (
        embedder.config.models != config.models
        or embedder.config.chunking.preprocessing_version != config.chunking.preprocessing_version
    ):
        raise ValueError("Experiment embedder must use the configured model and preprocessing")
    if output.exists():
        raise FileExistsError(f"Experiment output already exists: {output}")
    staged = config.model_copy(deep=True)
    staged.index.name = f"s3-rag-experiment-{uuid4().hex}"
    staged.index.enrichment_fingerprint = hashlib.sha256(
        json.dumps(inputs.artifact_sha256, sort_keys=True).encode("utf-8")
    ).hexdigest()
    # A fresh generated name cannot target the baseline alias or a previous experiment.
    if client.indices.exists(index=staged.index.name):
        raise ValueError("Experiment index name already exists")
    output.mkdir(parents=True, exist_ok=False)
    intent = {
        "schema_version": 1,
        "index_name": staged.index.name,
        "enrichment_fingerprint": staged.index.enrichment_fingerprint,
        "source_run": str(run),
        "source_artifact_sha256": inputs.artifact_sha256,
        "review_status": "unreviewed",
        "has_vectors": embedder is not None,
        "models": config.models.model_dump(mode="json"),
        "chunker_versions": sorted({chunk.chunker_version for chunk in inputs.chunks}),
        "embedding_preprocessing_version": config.chunking.preprocessing_version,
    }
    (output / "intent.json").write_text(json.dumps(intent, indent=2) + "\n", encoding="utf-8")
    before = (
        (embedder.request_count, embedder.cache_hits, embedder.input_tokens)
        if embedder
        else (0, 0, 0)
    )
    try:
        vectors = None
        if embedder is not None:
            vectors = {
                chunk.chunk_id: embedder.embed_text(text)
                for chunk, text in zip(inputs.chunks, inputs.embedding_texts, strict=True)
            }
        inferred_text: dict[str, str] = {}
        for chunk, result in zip(inputs.chunks, inputs.records, strict=True):
            metadata = result.metadata
            inferred_text[chunk.chunk_id] = (
                "\n".join(
                    part
                    for part in (
                        metadata.summary or "",
                        metadata.topic if metadata.topic != "unknown" else "",
                        metadata.intent if metadata.intent != "unknown" else "",
                        *metadata.related_concepts,
                        *metadata.example_phrasings,
                    )
                    if part
                )
                if metadata
                else ""
            )
        count = bulk_index_chunks(
            client, staged, inputs.chunks, vectors, inferred_text=inferred_text
        )
        vector_count = client.count(
            index=staged.index.name, body={"query": {"exists": {"field": "embedding"}}}
        )["count"]
        if vector_count != (count if embedder is not None else 0):
            raise ValueError("Experiment index has incomplete vector coverage")
        client.indices.put_settings(index=staged.index.name, body={"index.blocks.write": True})
        result = intent | {
            "document_count": count,
            "enriched_count": sum(record.metadata is not None for record in inputs.records),
            "embedding_requests": embedder.request_count - before[0] if embedder else 0,
            "embedding_cache_hits": embedder.cache_hits - before[1] if embedder else 0,
            "embedding_input_tokens": embedder.input_tokens - before[2] if embedder else 0,
        }
        (output / "index.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    except Exception:
        # This command owns only its fresh staging name; failed builds never switch an alias.
        if client.indices.exists(index=staged.index.name):
            client.indices.delete(index=staged.index.name)
        raise
