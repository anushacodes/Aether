"""Keep model-client construction behind the explicit cloud allowance."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from rag.aws_clients import create_bedrock_client
from rag.config import PipelineConfig


def test_missing_allowance_fails_before_credential_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = Mock()
    monkeypatch.setattr("rag.aws_clients.boto3.Session", session_factory)
    with pytest.raises(ValueError, match="RAG_DEVELOPMENT_MODEL_BUDGET_USD"):
        create_bedrock_client(PipelineConfig(cloud={"region": "us-east-1"}))
    session_factory.assert_not_called()

    configured = PipelineConfig(cloud={"region": "us-east-1", "development_model_budget_usd": 1})
    create_bedrock_client(configured, max_attempts=1)
    sdk_config = session_factory.return_value.client.call_args.kwargs["config"]
    assert sdk_config.retries["total_max_attempts"] == 1
    assert sdk_config.connect_timeout <= 5 and sdk_config.read_timeout <= 30
