"""Resolve defaults < TOML < explicit RAG_ environment overrides. No SDK calls."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, HttpUrl, model_validator

from rag.models import Record


class ChunkingConfig(Record):
    strategy: Literal["fixed", "heading-aware"] = "fixed"
    size: int = Field(default=500, gt=0)
    overlap: int = Field(default=75, ge=0)
    counting_method: Literal["whitespace-estimate"] = "whitespace-estimate"
    preprocessing_version: str = Field(default="raw-v1", min_length=1)

    @model_validator(mode="after")
    def valid_overlap(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class ModelConfig(Record):
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: Literal[256, 512, 1024] = 1024
    normalize: bool = True
    generation_model_id: str | None = None


class RetrievalConfig(Record):
    candidate_count: int = Field(default=20, ge=1, le=100)
    context_passages: int = Field(default=5, ge=1, le=20)
    context_budget: int = Field(default=4000, gt=0)
    rrf_constant: int = Field(default=60, gt=0)


class CloudConfig(Record):
    region: str | None = None
    profile: str | None = None
    artifact_bucket: str | None = None
    development_model_budget_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class PipelineConfig(Record):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    opensearch_url: HttpUrl = HttpUrl("http://localhost:9200")

    def require_cloud_settings(self, *, require_bucket: bool = False) -> None:
        required = {
            "RAG_AWS_REGION": self.cloud.region,
            "RAG_GENERATION_MODEL_ID": self.models.generation_model_id,
            "RAG_DEVELOPMENT_MODEL_BUDGET_USD": self.cloud.development_model_budget_usd,
        }
        if require_bucket:
            required["RAG_ARTIFACT_BUCKET"] = self.cloud.artifact_bucket
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Cloud configuration missing: " + ", ".join(missing))


ENV_FIELDS = {
    "RAG_AWS_REGION": ("cloud", "region"),
    "RAG_AWS_PROFILE": ("cloud", "profile"),
    "RAG_ARTIFACT_BUCKET": ("cloud", "artifact_bucket"),
    "RAG_DEVELOPMENT_MODEL_BUDGET_USD": ("cloud", "development_model_budget_usd"),
    "RAG_EMBEDDING_MODEL_ID": ("models", "embedding_model_id"),
    "RAG_EMBEDDING_DIMENSIONS": ("models", "embedding_dimensions"),
    "RAG_GENERATION_MODEL_ID": ("models", "generation_model_id"),
}


def load_config(
    path: Path | None = None, *, environ: Mapping[str, str] | None = None
) -> PipelineConfig:
    values = tomllib.loads(path.read_text(encoding="utf-8")) if path else {}
    env = os.environ if environ is None else environ
    for variable, (group, field) in ENV_FIELDS.items():
        value = env.get(variable, "").strip()
        if value:
            # Literal integer values are intentionally explicit rather than loosely coerced.
            if variable == "RAG_EMBEDDING_DIMENSIONS":
                value = int(value)
            values.setdefault(group, {})[field] = value
    if env.get("RAG_OPENSEARCH_URL", "").strip():
        values["opensearch_url"] = env["RAG_OPENSEARCH_URL"].strip()
    return PipelineConfig.model_validate(values)
