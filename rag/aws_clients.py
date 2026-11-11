"""Create bounded AWS clients through the standard credential chain; never store keys."""

from __future__ import annotations

import re
from typing import Any, Protocol, cast

import boto3
from botocore.config import Config

from rag.config import ConfigurationError, PipelineConfig

CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 30
MAX_REQUEST_ATTEMPTS = 3


class BedrockRuntime(Protocol):
    """The SDK's dynamic response boundary, shared by real clients and test doubles."""

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        """Return a model-specific streaming-body response."""
        ...

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        """Return a normalized generation response and its token usage."""
        ...

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        """Return the SDK event stream for incremental generation."""
        ...

    def close(self) -> None:
        """Release the SDK connection pool when its owning workflow finishes."""
        ...


class RerankRuntime(Protocol):
    """Native reranking uses a separate Bedrock agent-runtime service client."""

    def rerank(self, **kwargs: Any) -> dict[str, Any]:
        """Return source-array indices and relevance scores, without reported token usage."""
        ...

    def close(self) -> None:
        """Release an owned connection pool."""
        ...


class ArtifactS3(Protocol):
    """The native S3 boundary for deliberate artifact transfers."""

    def get_public_access_block(self, **kwargs: Any) -> dict[str, Any]:
        """Read expected-owner bucket public-access controls."""
        ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        """Conditionally store an encrypted, checksummed object."""
        ...

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        """Return the downloaded streaming body and version metadata."""
        ...

    def close(self) -> None:
        """Release the owned SDK pool."""
        ...


def create_artifact_client(config: PipelineConfig) -> ArtifactS3:
    """Use bounded single-attempt S3 requests without requiring a model-call allowance."""
    if not config.cloud.region or not config.cloud.region.strip():
        raise ConfigurationError("S3 artifact transfers require RAG_AWS_REGION")
    sdk_config = Config(
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        read_timeout=READ_TIMEOUT_SECONDS,
        retries={"total_max_attempts": 1, "mode": "standard"},
    )
    return cast(ArtifactS3, create_aws_session(config).client("s3", config=sdk_config))


def require_reranking_settings(config: PipelineConfig) -> None:
    """Reject missing/foreign-region model choices before discovering credentials."""
    config.require_cloud_settings(require_generation=False)
    arn = config.reranking.model_arn or ""
    match = re.fullmatch(
        r"arn:(aws|aws-us-gov|aws-cn):bedrock:([^:]+)::foundation-model/[^/\s]+", arn
    )
    if match is None or match[2] != config.cloud.region:
        raise ConfigurationError(
            "Reranking requires a foundation-model ARN in the configured region"
        )


def create_reranking_client(config: PipelineConfig) -> RerankRuntime:
    """Use one SDK attempt and bounded sockets for an explicitly configured ranker."""
    require_reranking_settings(config)
    timeout = config.reranking.timeout_seconds
    sdk_config = Config(
        connect_timeout=min(CONNECT_TIMEOUT_SECONDS, timeout),
        read_timeout=timeout,
        retries={"total_max_attempts": 1, "mode": "standard"},
    )
    return cast(
        RerankRuntime, create_aws_session(config).client("bedrock-agent-runtime", config=sdk_config)
    )


def create_aws_session(config: PipelineConfig) -> boto3.Session:
    """Use an explicit profile/region or the caller's ordinary AWS credential chain."""
    return boto3.Session(profile_name=config.cloud.profile, region_name=config.cloud.region)


def create_bedrock_client(
    config: PipelineConfig,
    *,
    require_generation: bool = False,
    max_attempts: int = MAX_REQUEST_ATTEMPTS,
) -> BedrockRuntime:
    """Require a deliberate allowance before discovering credentials or invoking a model."""
    config.require_cloud_settings(require_generation=require_generation)
    if not 1 <= max_attempts <= MAX_REQUEST_ATTEMPTS:
        raise ValueError("Model SDK attempts must be between one and three")
    session = create_aws_session(config)
    sdk_config = Config(
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        read_timeout=READ_TIMEOUT_SECONDS,
        retries={"total_max_attempts": max_attempts, "mode": "standard"},
    )
    return cast(BedrockRuntime, session.client("bedrock-runtime", config=sdk_config))
