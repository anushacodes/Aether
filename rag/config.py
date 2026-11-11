"""Resolve defaults < TOML < explicit RAG_ environment overrides. No SDK calls."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, HttpUrl, SecretStr, model_validator

from rag.models import Record

ENV_FIELDS = {
    "RAG_AWS_REGION": ("cloud", "region"),
    "RAG_AWS_PROFILE": ("cloud", "profile"),
    "RAG_ARTIFACT_BUCKET": ("cloud", "artifact_bucket"),
    "RAG_DEVELOPMENT_MODEL_BUDGET_USD": ("cloud", "development_model_budget_usd"),
    "RAG_EMBEDDING_MODEL_ID": ("models", "embedding_model_id"),
    "RAG_EMBEDDING_DIMENSIONS": ("models", "embedding_dimensions"),
    "RAG_GENERATION_MODEL_ID": ("models", "generation_model_id"),
    "RAG_EVALUATOR_MODEL_ID": ("evaluation", "judge_model_id"),
    "RAG_SEARCH_INDEX": ("index", "name"),
    "RAG_ADMIN_TOKEN": ("ingestion", "admin_token"),
    "RAG_ALLOWED_ORIGINS": ("api", "allowed_origins"),
}


class ConfigurationError(ValueError):
    """A required deployment or paid-call choice has not been supplied."""


class ChunkingConfig(Record):
    """Validated source-window settings shared by the CLI and chunker."""

    strategy: Literal["fixed", "heading-aware"] = "fixed"
    size: int = Field(default=500, gt=0)
    overlap: int = Field(default=75, ge=0)
    counting_method: Literal["whitespace-estimate"] = "whitespace-estimate"
    preprocessing_version: str = Field(default="raw-v1", min_length=1)

    @model_validator(mode="after")
    def valid_overlap(self) -> Self:
        """Reject settings that cannot advance through the source text."""
        if self.overlap >= self.size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class ModelConfig(Record):
    """Record compatible model identifiers and embedding dimensions."""

    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: Literal[256, 512, 1024] = 1024
    normalize: bool = True
    generation_model_id: str | None = None


class RetrievalConfig(Record):
    """Bound candidate selection and the planned generation context."""

    candidate_count: int = Field(default=20, ge=1, le=100)
    context_passages: int = Field(default=5, ge=1, le=20)
    context_budget: int = Field(default=4000, gt=0)
    rrf_constant: int = Field(default=60, gt=0)


class CloudConfig(Record):
    """Keep optional cloud choices separate from offline corpus configuration."""

    region: str | None = None
    profile: str | None = None
    artifact_bucket: str | None = None
    development_model_budget_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class EmbeddingConfig(Record):
    """Bound model work and keep the reusable vector cache on local disk."""

    cache_path: Path = Path("data/cache/embeddings.sqlite3")
    concurrency: int = Field(default=2, ge=1, le=8)
    max_requests_per_run: int = Field(default=500, ge=1, le=500)


class GenerationConfig(Record):
    """Bound answer length and keep reproducible generation settings explicit."""

    max_output_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)


class IndexConfig(Record):
    """Name the local corpus index independently of OpenSearch connectivity."""

    name: str = Field(default="s3-rag-active", pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=100)


class PublicationConfig(Record):
    """Keep immutable corpus builds and the local publisher lock on durable storage."""

    alias: str = Field(default="s3-rag-active", pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=100)
    state_path: Path = Path("data/state.sqlite3")
    snapshot_dir: Path = Path("data/snapshots")


class ApiConfig(Record):
    """Keep browser origins and local request limits explicit."""

    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    max_concurrent_searches: int = Field(default=4, ge=1, le=8)
    requests_per_minute: int = Field(default=30, ge=1, le=120)
    stream_timeout_seconds: int = Field(default=90, ge=5, le=180)


class IngestionConfig(Record):
    """Only server-approved manifests can enter the local durable worker."""

    admin_token: SecretStr | None = None
    sources_path: Path = Path("data/manifests/sources.json")
    approved_subsets: dict[str, Path] = Field(
        default_factory=lambda: {"pilot": Path("data/manifests/pilot.json")}
    )
    raw_dir: Path = Path("data/raw")
    max_attempts: int = Field(default=3, ge=1, le=5)
    max_pending_jobs: int = Field(default=20, ge=1, le=100)


class WorkflowConfig(Record):
    """Bound an opt-in graph independently of provider retry and output settings."""

    max_retrieval_passes: Literal[1, 2] = 2
    max_model_calls: int = Field(default=5, ge=1, le=5)
    timeout_seconds: float = Field(default=90, ge=1, le=180, allow_inf_nan=False)
    assessment_max_tokens: int = Field(default=256, ge=64, le=512)


class EvaluationConfig(Record):
    """Keep the offline judge explicit, serial, and separate from the answer model."""

    judge_model_id: str | None = None
    max_samples: int = Field(default=5, ge=1, le=120)
    max_model_calls: int = Field(default=50, ge=1, le=500)
    max_output_tokens: int = Field(default=2048, ge=1, le=4096)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    cache_path: Path = Path("data/cache/evaluator.sqlite3")


class PipelineConfig(Record):
    """The resolved configuration passed explicitly to each pipeline stage."""

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    publication: PublicationConfig = Field(default_factory=PublicationConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    opensearch_url: HttpUrl = HttpUrl("http://localhost:9200")

    def require_cloud_settings(
        self, *, require_bucket: bool = False, require_generation: bool = True
    ) -> None:
        """Reject missing cloud choices without contacting AWS or discovering credentials."""
        required = {
            "RAG_AWS_REGION": self.cloud.region,
            "RAG_DEVELOPMENT_MODEL_BUDGET_USD": self.cloud.development_model_budget_usd,
        }
        if require_generation:
            required["RAG_GENERATION_MODEL_ID"] = self.models.generation_model_id
        if require_bucket:
            required["RAG_ARTIFACT_BUCKET"] = self.cloud.artifact_bucket
        missing = [
            name
            for name, value in required.items()
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing:
            raise ConfigurationError("Cloud configuration missing: " + ", ".join(missing))


def load_config(
    path: Path | None = None, *, environ: Mapping[str, str] | None = None
) -> PipelineConfig:
    """Merge explicit overrides with optional TOML, then validate the resolved settings."""
    values = tomllib.loads(path.read_text(encoding="utf-8")) if path else {}
    env = os.environ if environ is None else environ
    for variable, (group, field) in ENV_FIELDS.items():
        value = env.get(variable, "").strip()
        if value:
            section = values.setdefault(group, {})
            if not isinstance(section, dict):
                raise ValueError(f"Configuration section {group!r} must be a TOML table")
            # Literal dimensions must reach Pydantic as integers, not arbitrary strings.
            if variable == "RAG_EMBEDDING_DIMENSIONS":
                section[field] = int(value)
            elif variable == "RAG_ALLOWED_ORIGINS":
                section[field] = json.loads(value)
            else:
                section[field] = value
    if env.get("RAG_OPENSEARCH_URL", "").strip():
        values["opensearch_url"] = env["RAG_OPENSEARCH_URL"].strip()
    return PipelineConfig.model_validate(values)
