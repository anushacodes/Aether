"""Protect cache compatibility and avoid reusing invalid model output."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import ModelConfig
from rag.embeddings import EmbeddingCache, embedding_cache_key


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
