"""Persist reproducible local build/answer records without recording credentials or API headers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from rag.config import PipelineConfig
from rag.corpus import write_jsonl
from rag.generation import GROUNDING_INSTRUCTIONS, PROMPT_VERSION
from rag.models import AnswerResult, ChunkRecord, SearchRequest


def pipeline_fingerprint(config: PipelineConfig) -> str:
    """Identify behavior settings while excluding credentials, budgets, and machine-local paths."""
    settings = {
        "chunking": config.chunking.model_dump(mode="json"),
        "retrieval": config.retrieval.model_dump(mode="json"),
        "models": config.models.model_dump(mode="json"),
        "generation": config.generation.model_dump(mode="json"),
        "prompt_version": PROMPT_VERSION,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def build_manifest(
    config: PipelineConfig, chunks: list[ChunkRecord], *, has_vectors: bool
) -> dict[str, object]:
    """Tie the stored chunk identities to their pinned source snapshots and embedding space."""
    sources = {
        chunk.provenance.source_id: chunk.provenance.model_dump(mode="json") for chunk in chunks
    }
    identity = {
        "schema_version": 1,
        "pipeline_fingerprint": pipeline_fingerprint(config),
        "corpus_ids": sorted({chunk.corpus_id for chunk in chunks}),
        "chunk_ids": sorted(chunk.chunk_id for chunk in chunks),
        "sources": [sources[source_id] for source_id in sorted(sources)],
        "chunking": config.chunking.model_dump(mode="json"),
        "embedding": config.models.model_dump(mode="json", exclude={"generation_model_id"}),
        "has_vectors": has_vectors,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return identity | {
        "build_id": build_id,
        "index": config.index.name,
        "created_at": datetime.now(UTC).isoformat(),
    }


def save_build_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Atomically publish one JSON document using the existing artifact writer."""
    write_jsonl(path, [manifest])


def save_answer_run(
    output: Path, request: SearchRequest, result: AnswerResult, config: PipelineConfig
) -> None:
    """Save an explicit run with selected context kept separate from retrieval candidates."""
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "request": request.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "pipeline_fingerprint": pipeline_fingerprint(config),
        "prompt_version": PROMPT_VERSION,
        "system_prompt": GROUNDING_INSTRUCTIONS,
        "model_called": bool(result.usage_tokens),
        "generation_model_id": config.models.generation_model_id,
        "generation_parameters": config.generation.model_dump(mode="json"),
        "context_passages": [chunk.model_dump(mode="json") for chunk in result.evidence],
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_jsonl(output / "answer.json", [record])
