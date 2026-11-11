"""Ensure incomplete or cancelled provider streams cannot produce a validated final answer."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from rag.config import PipelineConfig
from rag.generation import assemble_context
from rag.models import ChunkRecord, RetrievedHit
from rag.streaming import GenerationStreamError, stream_model_text


def test_incomplete_and_cancelled_streams_close_the_provider_connection() -> None:
    fixture = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    chunk = ChunkRecord.model_validate(fixture)
    config = PipelineConfig(
        models={"generation_model_id": "test-model"},
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
    )
    context = assemble_context(
        [RetrievedHit(chunk=chunk, method="bm25", rank=1, score=1)], config.retrieval
    )
    events = MagicMock()
    events.__iter__.side_effect = lambda: iter(
        [{"contentBlockDelta": {"delta": {"text": "partial"}}}]
    )
    runtime = Mock()
    runtime.converse_stream.return_value = {"stream": events}
    with pytest.raises(GenerationStreamError, match="complete answer"):
        list(stream_model_text(runtime, config, "A question", context, threading.Event()))
    events.close.assert_called_once()

    events.close.reset_mock()
    cancelled = threading.Event()
    stream = stream_model_text(runtime, config, "A question", context, cancelled)
    assert next(stream).event == "token"
    cancelled.set()
    assert list(stream) == []
    events.close.assert_called_once()
