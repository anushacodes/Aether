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
