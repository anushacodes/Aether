"""Reject partial OpenSearch bulk success before a build can be treated as complete."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.config import PipelineConfig
from rag.indexing import bulk_index_chunks, index_definition
from rag.models import ChunkRecord


def test_bulk_failure_is_explicit_even_when_other_items_succeed() -> None:
    config = PipelineConfig()
    record = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["chunk"]
    chunk = ChunkRecord.model_validate(record)
    other = chunk.model_copy(update={"chunk_id": "another-chunk"})
    client = Mock()
    client.indices.exists_alias.return_value = False
    client.indices.exists.return_value = True
    client.indices.get_mapping.return_value = {
        config.index.name: {"mappings": index_definition(config)["mappings"]}
    }
    client.bulk.return_value = {
        "items": [
            {"index": {"_id": chunk.chunk_id, "status": 201}},
            {"index": {"_id": other.chunk_id, "status": 429, "error": {"type": "rejected"}}},
        ]
    }
    with pytest.raises(ValueError, match="another-chunk"):
        bulk_index_chunks(client, config, [chunk, other])
    client.count.assert_not_called()
