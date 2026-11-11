"""Protect cleanup when an OpenSearch setup check fails after creating its index."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from rag.search_store import check_search_store


def test_failed_search_removes_only_its_temporary_index() -> None:
    client = Mock()
    client.cluster.health.return_value = {"status": "green", "timed_out": False}
    client.search.side_effect = RuntimeError("Search interrupted")

    with pytest.raises(RuntimeError, match="Search interrupted"):
        check_search_store(client)

    created_index = client.indices.create.call_args.kwargs["index"]
    assert created_index.startswith("s3-rag-check-")
    client.indices.delete.assert_called_once_with(index=created_index)
