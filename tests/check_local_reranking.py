"""Check reranked local search, answers, streaming, and workflow limits without provider calls."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

from botocore.exceptions import ReadTimeoutError
from fastapi.testclient import TestClient

from rag.api.app import create_app, get_search_client
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.corpus import extract_sections, load_sources
from rag.generation import assemble_context
from rag.indexing import bulk_index_chunks
from rag.models import SearchRequest, SearchScope, SubsetManifest
from rag.pipeline import answer_question, retrieve
from rag.search_store import create_search_client
from rag.streaming import stream_answer
from rag.workflow_limits import WorkflowBudgetError


def response(text: str) -> dict[str, object]:
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }


class EventStream:
    """A finite SDK-stream double whose owner must close it."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, object]]:
        yield {"contentBlockDelta": {"delta": {"text": self.text}}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}

    def close(self) -> None:
        self.closed = True


def check() -> dict[str, object]:
    """Use isolated real indexes; all model boundaries are in-process doubles."""
    with TemporaryDirectory(prefix="aether-rerank-check-") as directory, ExitStack() as stack:
        root = Path(directory)
        config = PipelineConfig(
            models={"generation_model_id": "stub-answer-model"},
            cloud={"region": "us-east-1", "development_model_budget_usd": 1},
            publication={"state_path": root / "state.sqlite3", "snapshot_dir": root / "snapshots"},
            reranking={
                "enabled": True,
                "model_arn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0",
            },
        )
        token = uuid4().hex
        old, new, alias = (f"s3-rag-rerank-{name}-{token}" for name in ("old", "new", "alias"))
        config.index.name = old
        sources = load_sources(Path("data/manifests/sources.json"))
        subset = SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes())
        chunks = [
            chunk
            for section in extract_sections(sources, subset, Path("data/raw"))
            for chunk in chunk_section(section, config.chunking)
        ]
        search = stack.enter_context(create_search_client(config))
        owned = [old, new]
        service_clients: list[Mock] = []
        streams: list[EventStream] = []
        mode = ["reverse"]
        source_batches: list[list[str]] = []

        def rank(**kwargs: object) -> dict[str, object]:
            texts = [
                source["inlineDocumentSource"]["textDocument"]["text"]
                for source in kwargs["sources"]
            ]
            assert all(text in {c.original_text for c in chunks} for text in texts)
            source_batches.append(texts)
            if mode[0] == "switch":
                search.indices.update_aliases(
                    body={
                        "actions": [
                            {"remove": {"index": old, "alias": alias}},
                            {"add": {"index": new, "alias": alias}},
                        ]
                    }
                )
                mode[0] = "reverse"
            if mode[0] == "invalid":
                return {"results": []}
            if mode[0] == "timeout":
                raise ReadTimeoutError(endpoint_url="https://example.invalid")
            return {
                "results": [
                    {"index": i, "relevanceScore": float(i)} for i in reversed(range(len(texts)))
                ]
            }

        def rank_client(*_: object, **__: object) -> Mock:
            client = Mock()
            client.rerank.side_effect = rank
            service_clients.append(client)
            return client

        assessments = [0]

        def generate(**kwargs: object) -> dict[str, object]:
            task = json.loads(kwargs["messages"][0]["content"][0]["text"])
            if "output_schema" in task:
                assessments[0] += 1
                verdict = {"decision": "answer", "missing_evidence": "none", "additional_terms": []}
                if assessments[0] == 1:
                    verdict = {
                        "decision": "retry",
                        "missing_evidence": "missing_context",
                        "additional_terms": ["delete marker"],
                    }
                return response(json.dumps(verdict))
            identifier = task["reference_data"][0]["citation_id"]
            return response(f"Synthetic integration answer [{identifier}].")

        def model_client(*_: object, **__: object) -> Mock:
            client = Mock()
            client.converse.side_effect = generate

            def streamed(**kwargs: object) -> dict[str, object]:
                text = generate(**kwargs)["output"]["message"]["content"][0]["text"]
                events = EventStream(text)
                streams.append(events)
                return {"stream": events}

            client.converse_stream.side_effect = streamed
            service_clients.append(client)
            return client

        for target, factory in (
            ("rag.reranking.create_reranking_client", rank_client),
            ("rag.workflow_limits.create_reranking_client", rank_client),
            ("rag.pipeline.create_bedrock_client", model_client),
            ("rag.streaming.create_bedrock_client", model_client),
            ("rag.workflow_limits.create_bedrock_client", model_client),
        ):
            stack.enter_context(patch(target, side_effect=factory))
        try:
            bulk_index_chunks(search, config, chunks)
            config.index.name = new
            bulk_index_chunks(search, config, chunks[-1:])
            search.indices.update_aliases(
                body={"actions": [{"add": {"index": old, "alias": alias}}]}
            )
            config.index.name = alias
            original = config.model_copy(deep=True)
            original.reranking.enabled = False
            scope = SearchScope(corpus_id=subset.corpus_id)
            baseline = retrieve(search, original, "versioning", scope, "bm25")
            result = retrieve(search, config, "versioning", scope, "bm25")
            assert [hit.chunk.chunk_id for hit in result.hits] == [
                hit.chunk.chunk_id for hit in reversed(baseline.hits)
            ]
            assert result.reranking.status == "applied" and result.reranking.attempted_calls == 1
            assert "reranking" in result.timings_ms
            assert {
                c.model_dump_json()
                for c in assemble_context(result.hits, config.retrieval).evidence
            } == {c.model_dump_json() for c in chunks}
            request = SearchRequest(query="versioning", scope=scope)
            answer = answer_question(search, config, request)
            assert answer.status == "answered" and answer.reranking.status == "applied"
            events = list(stream_answer(search, config, request, threading.Event()))
            assert (
                events[-1].event == "done" and events[-1].data["reranking"]["status"] == "applied"
            )
            assert streams[-1].closed
            mode[0] = "invalid"
            fallback = config.model_copy(deep=True)
            fallback.reranking.failure_policy = "original_order"
            result = retrieve(search, fallback, "versioning", scope, "bm25")
            assert result.reranking.status == "fallback"
            assert [h.chunk.chunk_id for h in result.hits] == [
                h.chunk.chunk_id for h in baseline.hits
            ]
            app = create_app(config)
            app.dependency_overrides[get_search_client] = lambda: search
            with TestClient(app) as http:
                body = request.model_dump(mode="json")
                failed = http.post("/retrieval", json=body)
                assert (
                    failed.status_code == 502
                    and failed.json()["detail"]["code"] == "reranking_failed"
                )
                streamed = http.post("/search/stream", json=body)
                assert (
                    '"code": "reranking_failed"' in streamed.text
                    and "event: done" not in streamed.text
                )
                mode[0] = "timeout"
                timed_out = http.post("/retrieval", json=body)
                assert timed_out.status_code == 504
                assert timed_out.json()["detail"]["code"] == "reranking_timeout"
                timed_stream = http.post("/search/stream", json=body)
                assert '"code": "reranking_timeout"' in timed_stream.text
                assert "event: done" not in timed_stream.text
                mode[0] = "reverse"
                success = http.post("/retrieval", json=body)
                assert (
                    success.status_code == 200
                    and success.json()["reranking"]["status"] == "applied"
                )
                restricted = http.post(
                    "/retrieval",
                    json={"query": "versioning", "scope": {"permitted_guides": ["s3-api"]}},
                )
                assert restricted.status_code == 200
                assert all(
                    hit["chunk"]["provenance"]["guide"] == "s3-api"
                    for hit in restricted.json()["hits"]
                )
            mode[0] = "switch"
            assessments[0] = 0
            workflow = request.model_copy(update={"workflow": True})
            result = answer_question(search, config, workflow)
            assert result.workflow.model_calls == 5 and result.workflow.retrieval_passes == 2
            assert result.workflow.snapshot_index == old
            assert all(step.reranking.status == "applied" for step in result.workflow.steps)
            assert all(len(texts) == 3 for texts in source_batches[-2:])
            search.indices.update_aliases(
                body={
                    "actions": [
                        {"remove": {"index": new, "alias": alias}},
                        {"add": {"index": old, "alias": alias}},
                    ]
                }
            )
            limited = config.model_copy(deep=True)
            limited.workflow.max_model_calls = 1
            try:
                answer_question(search, limited, workflow)
            except WorkflowBudgetError:
                pass
            else:
                raise AssertionError("Rerank calls must consume the shared workflow allowance")
            assert all(client.close.call_count == 1 for client in service_clients)
        finally:
            for name in owned:
                if search.indices.exists(index=name):
                    search.indices.delete(index=name)
    return {
        "passed": [
            "real local reranked retrieval",
            "original evidence and source IDs",
            "normal and streamed answers",
            "explicit fallback",
            "HTTP/SSE failure contracts",
            "scope filters",
            "two-pass pinned workflow",
            "shared five-call limit",
            "owned-client and stream cleanup",
        ],
        "real_aws_requests": 0,
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
