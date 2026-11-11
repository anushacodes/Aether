"""Run an explicitly authorized, priced three-inference Bedrock capability check.

The caller supplies current on-demand rates. CountTokens gates both generation calls;
Titan's full input ceiling reserves embedding cost conservatively. No production cache,
index, hosted resource, or evaluation label is changed.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import closing
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, cast

from pydantic import Field, HttpUrl

from rag.aws_clients import BedrockRuntime, create_bedrock_client
from rag.config import PipelineConfig
from rag.embeddings import Embedder
from rag.generation import (
    assemble_context,
    converse_text,
    finalize_answer,
    generation_arguments,
)
from rag.models import ChunkRecord, Record, RetrievedHit
from rag.streaming import stream_model_text

TITAN_INPUT_TOKEN_CEILING = 8192
MAX_OUTPUT_TOKENS = 64
MAX_CHUNK_CHARACTERS = 4000
MAX_GENERATION_INPUT_TOKENS = 100_000
MAX_PRICE_AGE_DAYS = 30
MILLION = Decimal(1_000_000)
QUESTION = "Summarize the supplied passage briefly."


class CheckPrices(Record):
    """Caller-attested rates for these exact models, region and on-demand token billing."""

    schema_version: Literal[1] = 1
    billing: Literal["on_demand_per_token"]
    region: str = Field(min_length=1)
    embedding_model_id: str = Field(min_length=1)
    generation_model_id: str = Field(min_length=1)
    retrieved_on: date
    source_url: HttpUrl
    embedding_input_usd_per_million: Decimal = Field(gt=0, max_digits=18, decimal_places=9)
    generation_input_usd_per_million: Decimal = Field(gt=0, max_digits=18, decimal_places=9)
    generation_output_usd_per_million: Decimal = Field(gt=0, max_digits=18, decimal_places=9)


class CheckRuntime(BedrockRuntime, Protocol):
    """Token counting is only needed by this deliberate preflight, not ordinary answering."""

    def count_tokens(self, **kwargs: Any) -> dict[str, Any]:
        """Count the exact Converse input before any inference is attempted."""
        ...


def _save_report(output: Path, report: dict[str, Any]) -> None:
    temporary = output / ".report.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


def _token_count(value: object, ceiling: int) -> int:
    if type(value) is not int or not 0 <= value <= ceiling:
        raise ValueError("Provider token count exceeds the check's accepted bound")
    return value


def run_provider_check(
    config: PipelineConfig,
    chunk: ChunkRecord,
    prices: CheckPrices,
    max_cost_usd: Decimal,
    output: Path,
    *,
    allow_paid: bool,
) -> dict[str, Any]:
    """Stop on the first failure; retain attempted calls and unknown charges in a new report."""
    # All local guards precede output reservation, SDK construction and credential discovery.
    if not allow_paid:
        raise ValueError("Provider check requires explicit --allow-paid")
    config.require_cloud_settings()
    if config.models.embedding_model_id != "amazon.titan-embed-text-v2:0":
        raise ValueError("Provider check supports only the existing Titan V2 embedding adapter")
    if not max_cost_usd.is_finite() or max_cost_usd <= 0:
        raise ValueError("Provider-check maximum cost must be a positive finite amount")
    if max_cost_usd > Decimal(str(config.cloud.development_model_budget_usd)):
        raise ValueError("Provider-check maximum exceeds the configured development allowance")
    if (
        prices.region != config.cloud.region
        or prices.embedding_model_id != config.models.embedding_model_id
        or prices.generation_model_id != config.models.generation_model_id
    ):
        raise ValueError("Price record must match the configured region and exact model IDs")
    if not 0 <= (datetime.now(UTC).date() - prices.retrieved_on).days <= MAX_PRICE_AGE_DAYS:
        raise ValueError("Price record must be from the last 30 days and cannot be future-dated")
    if not chunk.original_text.strip() or len(chunk.original_text) > MAX_CHUNK_CHARACTERS:
        raise ValueError("Select one nonblank original chunk of at most 4000 characters")
    settings = config.model_copy(deep=True)
    settings.generation.max_output_tokens = MAX_OUTPUT_TOKENS
    settings.generation.temperature = 0
    settings.embedding.max_requests_per_run = 1
    settings.embedding.cache_path = output / "embedding-cache.sqlite3"
    context = assemble_context(
        [RetrievedHit(chunk=chunk, method="bm25", rank=1, score=1)], settings.retrieval
    )
    arguments = generation_arguments(settings, QUESTION, context)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "region": config.cloud.region,
        "prices": prices.model_dump(mode="json"),
        "max_cost_usd": str(max_cost_usd),
        "estimated_upper_cost_usd": None,
        "known_cost_usd": "0",
        "total_cost_usd": None,
        "maximum_inference_attempts": 3,
        "sdk_attempts_per_operation": 1,
        "chunk_id": chunk.chunk_id,
        "chunk_sha256": hashlib.sha256(chunk.original_text.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "stages": [],
    }
    stage: dict[str, Any] | None = None
    stage_started = perf_counter()
    known_cost = Decimal(0)
    started = perf_counter()

    def begin(name: str, *, paid: bool) -> None:
        nonlocal stage, stage_started
        stage_started = perf_counter()
        stage = {"operation": name, "inference_attempted": paid, "status": "running"}
        report["stages"].append(stage)
        # Persist intent before invocation. A crash leaves uncertainty, never a false zero bill.
        _save_report(output, report)

    def complete(usage: dict[str, int], cost: Decimal) -> None:
        nonlocal known_cost
        assert stage is not None
        stage.update(
            status="passed",
            usage_tokens=usage,
            cost_usd=str(cost),
            elapsed_ms=(perf_counter() - stage_started) * 1000,
        )
        known_cost += cost
        report["known_cost_usd"] = str(known_cost)
        _save_report(output, report)

    try:
        _save_report(output, report)
        with closing(
            create_bedrock_client(settings, require_generation=True, max_attempts=1)
        ) as sdk:
            runtime = cast(CheckRuntime, sdk)
            begin("count_tokens", paid=False)
            counted = runtime.count_tokens(
                modelId=arguments["modelId"],
                input={"converse": {key: arguments[key] for key in ("messages", "system")}},
            )
            input_tokens = _token_count(counted.get("inputTokens"), MAX_GENERATION_INPUT_TOKENS)
            upper = (
                TITAN_INPUT_TOKEN_CEILING * prices.embedding_input_usd_per_million
                + 2 * input_tokens * prices.generation_input_usd_per_million
                + 2 * MAX_OUTPUT_TOKENS * prices.generation_output_usd_per_million
            ) / MILLION
            report["estimated_upper_cost_usd"] = str(upper)
            complete({"generation_input_per_call": input_tokens}, Decimal(0))
            if upper > max_cost_usd:
                stage = None
                report["error_code"] = "cost_limit_exceeded"
                raise ValueError("Conservative priced request bound exceeds the supplied maximum")

            begin("embedding", paid=True)
            embedder = Embedder(settings, runtime)
            vector = embedder.embed_text(chunk.original_text)
            used = _token_count(embedder.input_tokens, TITAN_INPUT_TOKEN_CEILING)
            complete(
                {"embedding_input": used}, used * prices.embedding_input_usd_per_million / MILLION
            )
            assert stage is not None
            stage["dimensions"] = len(vector)
            # Reuse the same adapter/cache, proving this repeat does not send another request.
            repeated = embedder.embed_text(chunk.original_text)
            if repeated != vector or embedder.request_count != 1 or embedder.cache_hits != 1:
                raise ValueError("Repeated embedding did not reuse the same validated cache entry")
            stage["repeat_cache_hits"] = embedder.cache_hits
            _save_report(output, report)

            for name in ("generation", "streaming"):
                begin(name, paid=True)
                if name == "generation":
                    generated = converse_text(runtime, settings, QUESTION, context)
                    text = generated.text
                    usage = {
                        "generation_input": generated.input_tokens,
                        "generation_output": generated.output_tokens,
                    }
                else:
                    parts: list[str] = []
                    usage = {}
                    for event in stream_model_text(
                        runtime, settings, QUESTION, context, threading.Event()
                    ):
                        if event.event == "token":
                            parts.append(str(event.data["text"]))
                        elif event.event == "usage":
                            usage = dict(event.data)
                    text = "".join(parts).strip()
                used_input = _token_count(usage.get("generation_input"), input_tokens)
                used_output = _token_count(usage.get("generation_output"), MAX_OUTPUT_TOKENS)
                cost = (
                    used_input * prices.generation_input_usd_per_million
                    + used_output * prices.generation_output_usd_per_million
                ) / MILLION
                complete(usage, cost)
                if not text:
                    raise ValueError("Provider returned no text")
                answer = finalize_answer(text, context, {})
                assert stage is not None
                stage["answer_status"] = answer.status
                stage["citation_count"] = len(answer.citation_ids)
                _save_report(output, report)
        report["status"] = "passed"
        report["total_cost_usd"] = str(known_cost)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error_type"] = type(error).__name__
        report.setdefault("error_code", "provider_check_failed")
        if stage is not None:
            stage["status"] = "failed"
            stage["elapsed_ms"] = (perf_counter() - stage_started) * 1000
        # Do not write raw provider messages, responses, question/passage text or credentials.
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        report["elapsed_ms"] = (perf_counter() - started) * 1000
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["inference_attempts"] = sum(item["inference_attempted"] for item in report["stages"])
        if report["inference_attempts"] == 0:
            report["total_cost_usd"] = "0"
        _save_report(output, report)
    return report
