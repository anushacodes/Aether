"""Protect source identity, bounded native calls, and explicit reranking failure behavior."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from rag.aws_clients import create_reranking_client
from rag.config import PipelineConfig
from rag.models import ChunkRecord, RetrievedHit
from rag.reranking import RerankingError, rerank_candidates
from rag.workflow_limits import WorkflowCancelled

ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"


def configured() -> PipelineConfig:
    return PipelineConfig(
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
        reranking={"enabled": True, "model_arn": ARN, "max_candidates": 2},
    )


def candidates() -> list[RetrievedHit]:
    source = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    return [
        RetrievedHit(
            chunk=ChunkRecord.model_validate(source | {"chunk_id": f"chunk-{i}"}),
            method="hybrid",
            rank=i + 1,
            score=10 - i,
        )
        for i in range(3)
    ]


def ranked() -> dict[str, object]:
    return {"results": [{"index": 1, "relevanceScore": 3.0}, {"index": 0, "relevanceScore": 2.0}]}


def test_native_reranking_preserves_original_scores_records_and_unranked_tail() -> None:
    config = configured()
    hits = candidates()
    before = [hit.model_dump_json() for hit in hits]
    with (
        closing(
            boto3.client(
                "bedrock-agent-runtime",
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )
        ) as client,
        Stubber(client) as stub,
    ):
        stub.add_response(
            "rerank",
            ranked(),
            {
                "queries": [{"type": "TEXT", "textQuery": {"text": "object versions"}}],
                "sources": [
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {
                            "type": "TEXT",
                            "textDocument": {"text": hit.chunk.original_text},
                        },
                    }
                    for hit in hits[:2]
                ],
                "rerankingConfiguration": {
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {"modelArn": ARN},
                        "numberOfResults": 2,
                    },
                },
            },
        )
        result, report = rerank_candidates(config, "object versions", hits, runtime=client)
        stub.assert_no_pending_responses()
    assert [hit.chunk.chunk_id for hit in result] == ["chunk-1", "chunk-0", "chunk-2"]
    assert [hit.score for hit in result] == [9, 10, 8]
    assert [hit.rerank_score for hit in result] == [3, 2, None]
    assert [hit.rank for hit in result] == [1, 2, 3]
    assert [hit.model_dump_json() for hit in hits] == before
    assert report.status == "applied" and report.attempted_calls == 1
    assert report.input_count == 3 and report.candidate_ids == ["chunk-0", "chunk-1"]
    assert report.usage_status == "not_reported" and report.model_cost_usd is None


def test_invalid_permutations_and_limits_preserve_explicit_original_order_fallback() -> None:
    config, hits = configured(), candidates()
    runtime = Mock()
    invalid = (
        {"results": [{"index": 0, "relevanceScore": 1}, {"index": 0, "relevanceScore": 0}]},
        {"results": [{"index": 9, "relevanceScore": 1}, {"index": 0, "relevanceScore": 0}]},
        {
            "results": [
                {"index": 1, "relevanceScore": float("nan")},
                {"index": 0, "relevanceScore": 0},
            ]
        },
        ranked() | {"nextToken": "must-not-trigger-another-call"},
        {"results": []},
    )
    for response in invalid:
        runtime.rerank.return_value = response
        with pytest.raises(RerankingError) as failure:
            rerank_candidates(config, "versions", hits, runtime=runtime)
        assert failure.value.report.status == "failed" and failure.value.report.attempted_calls == 1
    config.reranking.failure_policy = "original_order"
    runtime.rerank.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "sensitive provider details"}},
        "Rerank",
    )
    result, report = rerank_candidates(config, "versions", hits, runtime=runtime)
    assert result is hits and report.status == "fallback" and report.reason == "provider_error"
    assert "sensitive" not in report.model_dump_json()
    runtime.reset_mock()
    config.reranking.max_total_characters = 1
    result, report = rerank_candidates(config, "versions", hits, runtime=runtime)
    assert result is hits and report.reason == "input_limit" and report.attempted_calls == 0
    runtime.rerank.assert_not_called()
    result, report = rerank_candidates(PipelineConfig(), "versions", hits, runtime=runtime)
    assert result is hits and report is None


def test_late_results_and_cancellation_cannot_hide_behind_fallback_or_leak_owned_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, hits = configured(), candidates()
    config.reranking.timeout_seconds = 1
    config.reranking.failure_policy = "original_order"
    now = [0.0]
    monkeypatch.setattr("rag.reranking.monotonic", lambda: now[0])
    runtime = Mock()

    def late(**_: object) -> dict[str, object]:
        now[0] = 2
        return ranked()

    runtime.rerank.side_effect = late
    factory = Mock(return_value=runtime)
    monkeypatch.setattr("rag.reranking.create_reranking_client", factory)
    result, report = rerank_candidates(config, "versions", hits)
    assert result is hits and report.reason == "timeout" and report.attempted_calls == 1
    runtime.close.assert_called_once()
    runtime.reset_mock()
    now[0] = 0
    cancelled = [False]

    def respond_and_cancel(**_: object) -> dict[str, object]:
        cancelled[0] = True
        return ranked()

    def checkpoint() -> None:
        if cancelled[0]:
            raise WorkflowCancelled("cancelled")

    runtime.rerank.side_effect = respond_and_cancel
    with pytest.raises(WorkflowCancelled):
        rerank_candidates(config, "versions", hits, checkpoint=checkpoint)
    runtime.close.assert_called_once()
    assert runtime.rerank.call_count == 1


def test_reranker_configuration_fails_before_credentials_and_uses_single_sdk_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()
    monkeypatch.setattr("rag.aws_clients.boto3.Session", session)
    config = configured()
    config.reranking.model_arn = ARN.replace("us-east-1", "us-west-2")
    with pytest.raises(ValueError, match="configured region"):
        create_reranking_client(config)
    config.reranking.model_arn = ARN
    config.cloud.development_model_budget_usd = None
    with pytest.raises(ValueError, match="RAG_DEVELOPMENT_MODEL_BUDGET_USD"):
        create_reranking_client(config)
    session.assert_not_called()
    config.cloud.development_model_budget_usd = 1
    config.reranking.timeout_seconds = 3
    create_reranking_client(config)
    args, kwargs = session.return_value.client.call_args
    assert args == ("bedrock-agent-runtime",)
    assert kwargs["config"].retries["total_max_attempts"] == 1
    assert kwargs["config"].read_timeout == 3 and kwargs["config"].connect_timeout == 3
