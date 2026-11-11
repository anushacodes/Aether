"""Exercise paid-check admission, real SDK schemas, and failure/usage accounting offline."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber
from botocore.validate import validate_parameters

from rag.cli import main
from rag.config import PipelineConfig
from rag.models import ChunkRecord
from rag.provider_check import CheckPrices, run_provider_check


def inputs() -> tuple[PipelineConfig, ChunkRecord, CheckPrices]:
    config = PipelineConfig(
        models={"generation_model_id": "test-model", "embedding_dimensions": 256},
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
    )
    chunk = ChunkRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    )
    prices = CheckPrices(
        billing="on_demand_per_token",
        region="us-east-1",
        embedding_model_id=config.models.embedding_model_id,
        generation_model_id="test-model",
        retrieved_on=datetime.now(UTC).date(),
        source_url="https://aws.amazon.com/bedrock/pricing/",
        embedding_input_usd_per_million=Decimal("0.1"),
        generation_input_usd_per_million=Decimal("1"),
        generation_output_usd_per_million=Decimal("2"),
    )
    return config, chunk, prices


class ProviderEvents:
    """Only the event transport is simulated; validate its invocation against the SDK model."""

    def __init__(self, text: str, *, interrupted: bool = False) -> None:
        self.text = text
        self.interrupted = interrupted
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, object]]:
        yield {"contentBlockDelta": {"delta": {"text": self.text}}}
        if self.interrupted:
            yield {"internalServerException": {"message": "private-provider-marker"}}
            return
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 300, "outputTokens": 20}}}

    def close(self) -> None:
        self.closed = True


def embedding_response(*, invalid: bool = False) -> dict[str, object]:
    encoded = json.dumps(
        {"embedding": [1.0] * (1 if invalid else 256), "inputTextTokenCount": 5}
    ).encode()
    return {
        "body": StreamingBody(io.BytesIO(encoded), len(encoded)),
        "contentType": "application/json",
    }


def generation_response(text: str) -> dict[str, object]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 300, "outputTokens": 20, "totalTokens": 320},
        "metrics": {"latencyMs": 1},
    }


def test_offline_guards_and_cli_refuse_before_credentials_or_overwriting_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, chunk, prices = inputs()
    factory = Mock()
    monkeypatch.setattr("rag.provider_check.create_bedrock_client", factory)
    for changed, selected, rates, maximum, authorized in [
        (config, chunk, prices, Decimal(".01"), False),
        (PipelineConfig(), chunk, prices, Decimal(".01"), True),
        (config, chunk, prices.model_copy(update={"region": "us-west-2"}), Decimal(".01"), True),
        (
            config,
            chunk,
            prices.model_copy(update={"retrieved_on": prices.retrieved_on - timedelta(days=31)}),
            Decimal(".01"),
            True,
        ),
        (config, chunk, prices, Decimal("NaN"), True),
        (config, chunk, prices, Decimal("2"), True),
        (
            config,
            chunk.model_copy(update={"original_text": "x" * 4001}),
            prices,
            Decimal(".01"),
            True,
        ),
    ]:
        with pytest.raises(ValueError):
            run_provider_check(
                changed, selected, rates, maximum, tmp_path / "unused", allow_paid=authorized
            )
        assert not (tmp_path / "unused").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "report.json").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_provider_check(config, chunk, prices, Decimal(".01"), existing, allow_paid=True)
    assert (existing / "report.json").read_text(encoding="utf-8") == "preserve"
    assert (
        main(
            [
                "bedrock-check",
                "--prices",
                "missing.json",
                "--chunk-id",
                chunk.chunk_id,
                "--max-cost-usd",
                ".01",
                "--output",
                str(tmp_path / "cli"),
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit) as invalid:
        main(
            [
                "bedrock-check",
                "--prices",
                "missing.json",
                "--chunk-id",
                chunk.chunk_id,
                "--max-cost-usd",
                "not-money",
                "--output",
                str(tmp_path / "invalid"),
            ]
        )
    assert invalid.value.code == 2
    factory.assert_not_called()


def test_sdk_success_counts_exact_prompt_reserves_cost_and_retains_only_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, chunk, prices = inputs()
    text = f"Object versions are retained [{chunk.chunk_id}]."
    output = tmp_path / "run"
    events = ProviderEvents(text)
    with (
        closing(
            boto3.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )
        ) as sdk,
        Stubber(sdk) as stub,
    ):
        stub.add_response(
            "count_tokens", {"inputTokens": 300}, {"modelId": "test-model", "input": ANY}
        )
        stub.add_response(
            "invoke_model",
            embedding_response(),
            {
                "modelId": config.models.embedding_model_id,
                "contentType": "application/json",
                "accept": "application/json",
                "body": ANY,
            },
        )
        stub.add_response("converse", generation_response(text))
        counted_inputs = []

        def observe(model: object, params: dict[str, object], **kwargs: object) -> None:
            current = json.loads((output / "report.json").read_text(encoding="utf-8"))
            assert current["stages"][-1]["status"] == "running"
            if "input" in params:
                counted_inputs.append(params["input"]["converse"])
            elif "messages" in params:
                assert {key: params[key] for key in ("messages", "system")} == counted_inputs[0]
                assert params["inferenceConfig"]["maxTokens"] == 64

        sdk.meta.events.register("before-parameter-build.bedrock-runtime.*", observe)

        def streamed(**kwargs: object) -> dict[str, object]:
            validate_parameters(
                kwargs, sdk.meta.service_model.operation_model("ConverseStream").input_shape
            )
            observe(None, kwargs)
            return {"stream": events}

        monkeypatch.setattr(sdk, "converse_stream", streamed)
        closed = Mock(wraps=sdk.close)
        monkeypatch.setattr(sdk, "close", closed)
        factory = Mock(return_value=sdk)
        monkeypatch.setattr("rag.provider_check.create_bedrock_client", factory)
        config_file = tmp_path / "check.toml"
        config_file.write_text(
            '[cloud]\nregion="us-east-1"\ndevelopment_model_budget_usd=1\n'
            '[models]\ngeneration_model_id="test-model"\nembedding_dimensions=256\n',
            encoding="utf-8",
        )
        price_file = tmp_path / "prices.json"
        price_file.write_text(prices.model_dump_json(), encoding="utf-8")
        chunk_file = tmp_path / "chunks.jsonl"
        chunk_file.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
        assert (
            main(
                [
                    "--config",
                    str(config_file),
                    "bedrock-check",
                    "--allow-paid",
                    "--prices",
                    str(price_file),
                    "--chunks",
                    str(chunk_file),
                    "--chunk-id",
                    chunk.chunk_id,
                    "--max-cost-usd",
                    ".01",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        stub.assert_no_pending_responses()
        closed.assert_called_once()
        assert factory.call_args.kwargs == {"require_generation": True, "max_attempts": 1}
    assert report["status"] == "passed" and report["inference_attempts"] == 3
    assert report["stages"][1]["repeat_cache_hits"] == 1 and events.closed
    assert Decimal(report["estimated_upper_cost_usd"]) <= Decimal(".01")
    assert Decimal(report["total_cost_usd"]) == Decimal(".0006805")
    assert all(row["status"] == "passed" for row in report["stages"])
    assert config.generation.max_output_tokens == 512
    serialized = (output / "report.json").read_text(encoding="utf-8")
    assert chunk.original_text not in serialized and text not in serialized
    assert not (output / ".report.tmp").exists()


def test_failures_stop_later_inferences_and_keep_uncertain_billing_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, chunk, prices = inputs()
    text = f"Object versions are retained [{chunk.chunk_id}]."
    for failure, expected_attempts in [
        ("count", 0),
        ("budget", 0),
        ("embedding", 1),
        ("stream", 3),
    ]:
        events = ProviderEvents(text, interrupted=True)
        with (
            closing(
                boto3.client(
                    "bedrock-runtime",
                    region_name="us-east-1",
                    aws_access_key_id="testing",
                    aws_secret_access_key="testing",
                )
            ) as sdk,
            Stubber(sdk) as stub,
        ):
            if failure == "count":
                stub.add_client_error(
                    "count_tokens",
                    service_error_code="AccessDeniedException",
                    service_message="private-provider-marker",
                )
            else:
                stub.add_response("count_tokens", {"inputTokens": 300})
            if failure in {"embedding", "stream"}:
                stub.add_response(
                    "invoke_model", embedding_response(invalid=failure == "embedding")
                )
            if failure == "stream":
                stub.add_response("converse", generation_response(text))
            streaming = Mock(return_value={"stream": events})
            monkeypatch.setattr(sdk, "converse_stream", streaming)
            monkeypatch.setattr("rag.provider_check.create_bedrock_client", Mock(return_value=sdk))
            report = run_provider_check(
                config,
                chunk,
                prices,
                Decimal(".000001" if failure == "budget" else ".01"),
                tmp_path / failure,
                allow_paid=True,
            )
            stub.assert_no_pending_responses()
        assert report["status"] == "failed" and report["inference_attempts"] == expected_attempts
        assert report["total_cost_usd"] == ("0" if expected_attempts == 0 else None)
        assert "private-provider-marker" not in json.dumps(report)
        if failure == "stream":
            assert events.closed and report["stages"][-1]["status"] == "failed"
            assert Decimal(report["known_cost_usd"]) > 0
        else:
            streaming.assert_not_called()
