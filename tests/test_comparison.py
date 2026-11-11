"""Protect live snapshot alignment, honest call accounting, and worker-owned capacity."""

from __future__ import annotations

import asyncio
import io
import json
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from fastapi import HTTPException
from fastapi.testclient import TestClient

from rag.api.app import create_app, get_search_client
from rag.api.comparison import comparison_response
from rag.api.limits import RequestLimitError, SearchLimiter
from rag.comparison import compare_retrieval
from rag.config import PipelineConfig
from rag.models import ChunkRecord, ComparisonRequest, RetrievalResult, RetrievedHit


def _result(mode: str) -> RetrievalResult:
    chunk = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    return RetrievalResult(
        mode=mode,
        hits=[RetrievedHit(chunk=ChunkRecord.model_validate(chunk), method=mode, rank=1, score=1)],
        timings_ms={"retrieval_total": 1},
    )


def test_live_api_keeps_keyword_sources_when_vectors_are_missing_without_model_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig(api={"requests_per_minute": 2})
    pinned = config.model_copy(deep=True)
    pinned.index.name = "pinned-before-alias-switch"
    resolve = Mock(return_value=pinned)
    retrieve = Mock(return_value=_result("bm25"))
    factory = Mock(side_effect=AssertionError("No model setup for a text-only index"))
    monkeypatch.setattr("rag.comparison.resolve_read_config", resolve)
    monkeypatch.setattr("rag.comparison.verify_index", Mock())
    monkeypatch.setattr("rag.comparison.retrieve", retrieve)
    monkeypatch.setattr("rag.comparison.create_bedrock_client", factory)
    search = Mock()
    search.count.side_effect = lambda **kwargs: {"count": 0 if "body" in kwargs else 3}
    app = create_app(config)
    app.dependency_overrides[get_search_client] = lambda: search
    request = {"query": "object versions", "modes": ["bm25", "hybrid"]}
    with TestClient(app) as client:
        invalid = client.post("/retrieval/compare", json=request | {"modes": ["bm25", "bm25"]})
        assert invalid.status_code == 422
        response = client.post("/retrieval/compare", json=request)
        assert response.status_code == 200
        result = response.json()
        assert result["snapshot_index"] == pinned.index.name
        assert result["execution"] == "sequential_shared_cache"
        assert [arm["status"] for arm in result["arms"]] == ["ok", "error"]
        assert result["arms"][0]["result"]["hits"][0]["chunk"]["original_text"]
        assert result["arms"][1]["error_code"] == "vector_index_not_ready"
        assert all(arm["model_calls"] == arm["model_cost_usd"] == 0 for arm in result["arms"])
        assert response.headers["X-Request-ID"]
        assert client.post("/retrieval/compare", json=request).status_code == 200
        assert client.post("/retrieval/compare", json=request).status_code == 429
    assert all(call.args[1].index.name == pinned.index.name for call in retrieve.call_args_list)
    factory.assert_not_called()
    assert resolve.call_count == 2


def test_live_modes_share_cache_but_keep_call_counts_failures_and_client_ownership_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig(
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
        models={"embedding_dimensions": 256},
        embedding={"cache_path": tmp_path / "embeddings.sqlite3"},
    )
    pinned = config.model_copy(deep=True)
    pinned.index.name = "original-snapshot"
    resolve = Mock(return_value=pinned)
    monkeypatch.setattr("rag.comparison.resolve_read_config", resolve)
    monkeypatch.setattr("rag.comparison.verify_index", Mock())
    search = Mock()
    search.count.return_value = {"count": 2}

    def runtime() -> Mock:
        client = Mock()
        payload = json.dumps({"embedding": [1.0] + [0.0] * 255, "inputTextTokenCount": 3}).encode()
        client.invoke_model.return_value = {
            "body": StreamingBody(io.BytesIO(payload), len(payload))
        }
        return client

    first, second = runtime(), runtime()
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr("rag.comparison.create_bedrock_client", factory)

    def retrieve(
        client: object,
        settings: PipelineConfig,
        query: str,
        scope: object,
        mode: str,
        embedder: object,
        **kwargs: object,
    ) -> RetrievalResult:
        assert settings.index.name == pinned.index.name
        embedder.embed_text(query)
        return _result(mode)

    monkeypatch.setattr("rag.comparison.retrieve", retrieve)
    request = ComparisonRequest(query="object versions", modes=["vector", "hybrid"])
    result = compare_retrieval(search, config, request)
    resolve.assert_called_once()
    assert [arm.model_calls for arm in result.arms] == [1, 0]
    assert [arm.embedding_cache_hits for arm in result.arms] == [0, 1]
    assert [arm.known_embedding_input_tokens for arm in result.arms] == [3, 0]
    assert [arm.model_cost_usd for arm in result.arms] == [None, 0]
    first.invoke_model.assert_called_once()
    second.invoke_model.assert_not_called()
    assert all(call.kwargs["max_attempts"] == 1 for call in factory.call_args_list)
    first.close.assert_called_once()
    second.close.assert_called_once()
    failed, working = runtime(), runtime()
    failed.invoke_model.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "sensitive account details"}},
        "InvokeModel",
    )
    factory.side_effect = [failed, working]
    result = compare_retrieval(
        search, config, request.model_copy(update={"query": "different input"})
    )
    assert [arm.status for arm in result.arms] == ["error", "ok"]
    assert result.arms[0].error_code == "model_throttled"
    assert [arm.model_calls for arm in result.arms] == [1, 1]
    assert all(arm.model_cost_usd is None for arm in result.arms)
    assert "sensitive" not in result.model_dump_json()
    failed.close.assert_called_once()
    working.close.assert_called_once()


def test_comparison_disconnect_and_deadline_keep_capacity_until_worker_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(disconnect: bool) -> None:
        config = PipelineConfig(api={"max_concurrent_searches": 1, "comparison_timeout_seconds": 5})
        limiter = SearchLimiter(config.api)
        started, release, disconnected = threading.Event(), threading.Event(), threading.Event()
        now = [0.0]
        monkeypatch.setattr("rag.api.comparison.monotonic", lambda: now[0])

        def blocked(*args: object, checkpoint: object) -> None:
            started.set()
            assert release.wait(timeout=3)
            checkpoint()

        monkeypatch.setattr("rag.api.comparison.compare_retrieval", blocked)

        class Connection:
            async def is_disconnected(self) -> bool:
                return disconnected.is_set()

        task = asyncio.create_task(
            comparison_response(
                ComparisonRequest(query="object versions"), Connection(), Mock(), config, limiter
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            if disconnect:
                disconnected.set()
            else:
                now[0] = 6
            with pytest.raises(HTTPException) as failure:
                await task
            assert failure.value.status_code == (499 if disconnect else 504)
            with pytest.raises(RequestLimitError, match="search_busy"):
                limiter.acquire()
        finally:
            release.set()
        for _ in range(100):
            await asyncio.sleep(0.01)
            try:
                lease = limiter.acquire()
            except RequestLimitError:
                continue
            lease.close()
            break
        else:
            pytest.fail("Comparison worker did not release capacity")

    asyncio.run(scenario(True))
    asyncio.run(scenario(False))
