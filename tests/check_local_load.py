"""Measure a bounded real-HTTP BM25 workload and overload on isolated local search.

Use only the previously exposed 20 development drafts. No model client may be
created. Client timings include HTTP and response validation, not executor queue
time; this is a local transport/admission check, not a hosted capacity benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import uvicorn

from rag.api.app import create_app, get_search_client
from rag.artifacts import pipeline_fingerprint
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.corpus import extract_sections, load_sources, write_jsonl
from rag.eval.dataset import EvaluationQuestion, load_dataset
from rag.indexing import bulk_index_chunks
from rag.models import RetrievalResult, SubsetManifest
from rag.publication import active_index
from rag.search_store import create_search_client

BASELINE_REQUESTS = 20
BASELINE_CLIENTS = 4
BURST_REQUESTS = 40
BURST_CLIENTS = 8
HTTP_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class Observation:
    """Keep failed requests in the denominator without persisting response/source bodies."""

    phase: str
    sequence: int
    question_id: str
    started_seconds: float
    finished_seconds: float
    queue_ms: float
    latency_ms: float
    http_status: int | None
    outcome: Literal["ok", "http_error", "transport_error", "invalid_response"]
    error_code: str | None
    request_id: str | None
    snapshot_index: str | None
    hits: int | None
    server_retrieval_ms: float | None


def _request(
    url: str,
    index: str,
    corpus_id: str,
    question: EvaluationQuestion,
    phase: str,
    sequence: int,
    enqueued: float,
    phase_started: float,
) -> Observation:
    """Make one HTTP attempt, including errors; do not retry or print private response bodies."""
    started = time.monotonic()
    status, request_id, snapshot, hits, server_ms = None, None, None, None, None
    outcome: Literal["ok", "http_error", "transport_error", "invalid_response"] = "ok"
    code = None
    request = Request(
        url + "/retrieval",
        data=json.dumps(
            {
                "query": question.query,
                "mode": "bm25",
                "scope": {
                    "permitted_guides": question.permitted_guides,
                    "source_scope": question.source_scope,
                    "corpus_id": corpus_id,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                status = response.status
                request_id = response.headers.get("X-Request-ID")
                raw = response.read()
        except HTTPError as error:
            with error:
                status = error.code
                request_id = error.headers.get("X-Request-ID")
                raw = error.read()
        if status == 200:
            result = RetrievalResult.model_validate_json(raw)
            snapshot = result.snapshot_index
            hits = len(result.hits)
            server_ms = result.timings_ms.get("retrieval_total")
            if (
                result.mode != "bm25"
                or snapshot != index
                or not request_id
                or result.reranking is not None
                or any(
                    hit.chunk.corpus_id != corpus_id
                    or hit.chunk.source_scope != question.source_scope
                    or hit.chunk.provenance.guide not in question.permitted_guides
                    for hit in result.hits
                )
            ):
                outcome, code = "invalid_response", "source_or_request_identity_mismatch"
        else:
            outcome = "http_error"
            body = json.loads(raw)
            detail = body.get("detail") if isinstance(body, dict) else None
            code = detail.get("code") if isinstance(detail, dict) else None
            if not isinstance(code, str) or not request_id:
                outcome, code = "invalid_response", "missing_error_identity"
    except (URLError, TimeoutError, OSError) as error:
        outcome, code = "transport_error", type(error).__name__
    except ValueError:
        outcome, code = "invalid_response", "invalid_response_schema"
    finished = time.monotonic()
    return Observation(
        phase,
        sequence,
        question.question_id,
        started - phase_started,
        finished - phase_started,
        (started - enqueued) * 1000,
        (finished - started) * 1000,
        status,
        outcome,
        code,
        request_id,
        snapshot,
        hits,
        server_ms,
    )


def _latencies(values: list[float]) -> dict[str, float] | None:
    """Use nearest-rank percentiles, including an explicit empty-sample outcome."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[math.ceil(len(values) * 0.5) - 1],
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
        "max": ordered[-1],
        "mean": sum(values) / len(values),
    }


def _workload(
    url: str,
    index: str,
    corpus_id: str,
    questions: list[EvaluationQuestion],
    phase: str,
    count: int,
    clients: int,
) -> tuple[list[Observation], dict[str, object]]:
    """Measure fixed offered work with bounded clients and an honest common duration."""
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=clients) as executor:
        futures = [
            executor.submit(
                _request,
                url,
                index,
                corpus_id,
                questions[number % len(questions)],
                phase,
                number,
                time.monotonic(),
                started,
            )
            for number in range(count)
        ]
        rows = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    successful = [row for row in rows if row.outcome == "ok"]
    active, peak = 0, 0
    for _, change in sorted(
        [(row.started_seconds, 1) for row in rows] + [(row.finished_seconds, -1) for row in rows]
    ):
        active += change
        peak = max(active, peak)
    return rows, {
        "phase": phase,
        "offered_requests": count,
        "completed_observations": len(rows),
        "client_concurrency_limit": clients,
        "observed_peak_client_requests": peak,
        "elapsed_seconds": elapsed,
        "completed_requests_per_second": len(rows) / elapsed,
        "successful_requests_per_second": len(successful) / elapsed,
        "successes": len(successful),
        "failures": len(rows) - len(successful),
        "http_status_counts": dict(Counter(str(row.http_status) for row in rows)),
        "outcomes": dict(Counter(row.outcome for row in rows)),
        "error_codes": dict(Counter(row.error_code for row in rows if row.error_code)),
        "latency_all_ms": _latencies([row.latency_ms for row in rows]),
        "latency_success_ms": _latencies([row.latency_ms for row in successful]),
        "client_queue_ms": _latencies([row.queue_ms for row in rows]),
        "server_retrieval_ms": _latencies(
            [row.server_retrieval_ms for row in successful if row.server_retrieval_ms is not None]
        ),
    }


def check(output: Path) -> dict[str, object]:
    """Run native HTTP/OpenSearch, retain every result and remove only the owned index."""
    output.mkdir(parents=True, exist_ok=False)
    config = PipelineConfig(chunking={"strategy": "heading-aware"})
    index = f"s3-rag-load-check-{uuid4().hex}"
    config.index.name = index
    dataset_path = Path("data/eval/core-development.json")
    dataset = load_dataset(dataset_path, allow_drafts=True)
    questions = dataset.questions_for_split("development")
    assert len(questions) == BASELINE_REQUESTS
    sources_path, subset_path = (
        Path("data/manifests/sources.json"),
        Path("data/manifests/core.json"),
    )
    subset = SubsetManifest.model_validate_json(subset_path.read_bytes())
    sections = extract_sections(load_sources(sources_path), subset, Path("data/raw"))
    chunks = [chunk for section in sections for chunk in chunk_section(section, config.chunking)]
    assert subset.corpus_id == dataset.corpus_id and not config.reranking.enabled
    provider = Mock(
        side_effect=AssertionError("This local workload must not create a model client")
    )
    with create_search_client(config) as search:
        aliases = {name: active_index(search, name) for name in ("s3-rag-active", "s3-rag-core")}
        try:
            bulk_index_chunks(search, config, chunks)
            search.indices.put_settings(index=index, body={"index.blocks.write": True})
            application = create_app(config)
            application.dependency_overrides[get_search_client] = lambda: search
            with (
                patch("rag.api.app.create_bedrock_client", provider),
                patch("rag.pipeline.create_bedrock_client", provider),
                patch("rag.aws_clients.boto3.Session", provider),
                socket.socket() as listener,
            ):
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                url = f"http://127.0.0.1:{listener.getsockname()[1]}"
                server = uvicorn.Server(
                    uvicorn.Config(
                        application,
                        access_log=False,
                        log_config=None,
                        timeout_graceful_shutdown=5,
                    )
                )
                thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
                thread.start()
                try:
                    deadline = time.monotonic() + 10
                    while not server.started and thread.is_alive() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    assert server.started, "isolated HTTP server did not start"
                    rows, baseline = _workload(
                        url,
                        index,
                        dataset.corpus_id,
                        questions,
                        "baseline",
                        BASELINE_REQUESTS,
                        BASELINE_CLIENTS,
                    )
                    burst_rows, burst = _workload(
                        url,
                        index,
                        dataset.corpus_id,
                        questions,
                        "overload",
                        BURST_REQUESTS,
                        BURST_CLIENTS,
                    )
                    rows += burst_rows
                finally:
                    server.should_exit = True
                    thread.join(timeout=15)
                    assert not thread.is_alive(), "isolated HTTP server did not stop"
            assert not provider.called
            unchanged = aliases == {name: active_index(search, name) for name in aliases}
            assert unchanged
            report = {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "scope": "local real HTTP and BM25; no model or hosted capacity measurement",
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.fingerprint(),
                "review_status": dataset.review_status,
                "quality_scores_computed": False,
                "pipeline_id": pipeline_fingerprint(config),
                "snapshot_index": index,
                "sections": len(sections),
                "chunks": len(chunks),
                "server_limits": config.api.model_dump(mode="json"),
                "phases": [baseline, burst],
                "timing_definition": "client HTTP plus response validation; queue time separate",
                "percentile_definition": "nearest rank; no discarded failures or retries",
                "cache_conditions": (
                    "fresh index; indexing warms host/storage; no cache reset; "
                    "same server for both phases; new HTTP connection per request; "
                    "reused search pool"
                ),
                "rate_window": "both phases share one 60-second in-process admission window",
                "environment": {
                    "python": platform.python_version(),
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "opensearch": search.info()["version"],
                },
                "input_file_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [dataset_path, sources_path, subset_path, Path(__file__)]
                },
                "source_module_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path("rag").rglob("*.py"))
                },
                "checks": {
                    "baseline_all_ok": baseline["successes"] == BASELINE_REQUESTS,
                    "concurrent_http_observed": baseline["observed_peak_client_requests"] > 1,
                    "overload_refused": any(row.http_status == 429 for row in burst_rows),
                    "only_expected_burst_outcomes": all(
                        row.outcome == "ok"
                        or (
                            row.outcome == "http_error"
                            and row.http_status == 429
                            and row.error_code in {"search_busy", "rate_limited"}
                        )
                        for row in burst_rows
                    ),
                    "unique_request_ids": len({row.request_id for row in rows}) == len(rows),
                },
                "baseline_aliases_unchanged": unchanged,
                "real_aws_requests": 0,
            }
            write_jsonl(output / "requests.jsonl", [asdict(row) for row in rows])
        finally:
            if search.indices.exists(index=index):
                search.indices.delete(index=index)
            assert not search.indices.exists(index=index), "owned workload index still exists"
    report["owned_index_removed"] = True
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Keep the native check separate from unit tests and retain inspectable measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/runs") / f"local-load-{uuid4().hex}"
    )
    args = parser.parse_args()
    report = check(args.output)
    print(json.dumps(report | {"output": str(args.output)}, indent=2))
    if not all(report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
