"""Protect cache compatibility and avoid reusing invalid model output."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.config import ModelConfig, PipelineConfig
from rag.embeddings import Embedder, EmbeddingCache, embedding_cache_key


def test_embedding_cache_is_persistent_and_configuration_specific(tmp_path: Path) -> None:
    model = ModelConfig(embedding_dimensions=256)
    key = embedding_cache_key("exact input", model, "raw-v1")
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    vector = [0.1] * 256
    cache.put(key, vector)
    reopened = EmbeddingCache(cache.path)
    assert reopened.get(key, 256) == vector
    changed_keys = [
        embedding_cache_key("exact input ", model, "raw-v1"),
        embedding_cache_key("exact input", ModelConfig(embedding_dimensions=512), "raw-v1"),
        embedding_cache_key("exact input", model, "enriched-v1"),
        embedding_cache_key("exact input", ModelConfig(normalize=False), "raw-v1"),
    ]
    assert all(reopened.get(changed, 256) is None for changed in changed_keys)
    with pytest.raises(ValueError, match="exactly 512"):
        reopened.get(key, 512)


def test_invalid_vector_cannot_replace_valid_cached_output(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    cache.put("existing", [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        cache.put("existing", [float("nan"), 0.0])
    assert cache.get("existing", 2) == [1.0, 0.0]


def test_embedder_reuses_cache_and_never_sends_blank_or_incompatible_output(tmp_path: Path) -> None:
    config = PipelineConfig(
        models={"embedding_dimensions": 256},
        embedding={"cache_path": tmp_path / "embeddings.sqlite3"},
        cloud={"region": "us-east-1", "development_model_budget_usd": 1},
    )
    client = Mock()
    client.invoke_model.return_value = {
        "body": io.BytesIO(
            json.dumps({"embedding": [0.1] * 256, "inputTextTokenCount": 3}).encode()
        )
    }
    embedder = Embedder(config, client)
    first = embedder.embed_text("S3 object versions")
    assert embedder.embed_text("S3 object versions") == first
    assert embedder.request_count == 1
    assert embedder.cache_hits == 1
    assert embedder.input_tokens == 3
    with pytest.raises(ValueError, match="blank"):
        embedder.embed_text("  ")
    client.invoke_model.assert_called_once()

    client.invoke_model.return_value = {"body": io.BytesIO(b'{"embedding": [0.1]}')}
    with pytest.raises(ValueError, match="exactly 256"):
        embedder.embed_text("different input")
    key = embedding_cache_key("different input", config.models, "raw-v1")
    assert embedder.cache.get(key, 256) is None
