"""Validate embedding vectors and reuse them across processes without storing input text."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from rag.config import ModelConfig

SQLITE_TIMEOUT_SECONDS = 10


def validate_embedding(value: object, dimensions: int) -> list[float]:
    """Reject incompatible, nonfinite, or zero vectors before caching or cosine search."""
    if not isinstance(value, list) or len(value) != dimensions:
        raise ValueError(f"Embedding must contain exactly {dimensions} numbers")
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        for number in value
    ):
        raise ValueError("Embedding values must be finite numbers")
    vector = [float(number) for number in value]
    if not any(vector):
        raise ValueError("Embedding must have nonzero magnitude for cosine search")
    return vector


def embedding_cache_key(text: str, model: ModelConfig, preprocessing_version: str) -> str:
    """Fingerprint exact input and every setting that defines its embedding space."""
    identity = {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "model_id": model.embedding_model_id,
        "dimensions": model.embedding_dimensions,
        "normalize": model.normalize,
        "preprocessing_version": preprocessing_version,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EmbeddingCache:
    """A transactional local vector cache with a separate connection per operation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            closing(sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)) as connection,
            connection,
        ):
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
            )

    def get(self, cache_key: str, dimensions: int) -> list[float] | None:
        """Return a validated cached vector; corrupt entries fail without a paid retry."""
        with (
            closing(sqlite3.connect(self.path, timeout=SQLITE_TIMEOUT_SECONDS)) as connection,
            connection,
        ):
            row = connection.execute(
                "SELECT vector FROM embeddings WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return validate_embedding(json.loads(row[0]), dimensions)

    def put(self, cache_key: str, vector: list[float]) -> None:
        """Publish a complete vector atomically so interrupted writes cannot poison reuse."""
        validated = validate_embedding(vector, len(vector))
        with (
            closing(sqlite3.connect(self.path, timeout=SQLITE_TIMEOUT_SECONDS)) as connection,
            connection,
        ):
            connection.execute(
                "INSERT INTO embeddings(cache_key, vector) VALUES (?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET vector = excluded.vector",
                (cache_key, json.dumps(validated, allow_nan=False)),
            )
