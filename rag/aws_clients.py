"""Create bounded AWS clients through the standard credential chain; never store keys."""

from __future__ import annotations

from typing import Any, Protocol, cast

import boto3
from botocore.config import Config

from rag.config import PipelineConfig

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


def create_aws_session(config: PipelineConfig) -> boto3.Session:
    """Use an explicit profile/region or the caller's ordinary AWS credential chain."""
    return boto3.Session(profile_name=config.cloud.profile, region_name=config.cloud.region)


def create_bedrock_client(
    config: PipelineConfig, *, require_generation: bool = False
) -> BedrockRuntime:
    """Require a deliberate allowance before discovering credentials or invoking a model."""
    config.require_cloud_settings(require_generation=require_generation)
    session = create_aws_session(config)
    sdk_config = Config(
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        read_timeout=READ_TIMEOUT_SECONDS,
        retries={"total_max_attempts": MAX_REQUEST_ATTEMPTS, "mode": "standard"},
    )
    return cast(BedrockRuntime, session.client("bedrock-runtime", config=sdk_config))
