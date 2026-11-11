"""Exercise live comparison against isolated OpenSearch indexes with native SDK stubs."""

from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from rag.api.app import create_app, get_search_client
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.corpus import extract_sections, load_sources
from rag.indexing import bulk_index_chunks
from rag.models import SubsetManifest
from rag.search_store import create_search_client


def check() -> dict[str, object]:
    """Preserve user aliases and delete only indexes owned by this check."""
    with TemporaryDirectory(prefix="aether-live-comparison-") as directory:
        config = PipelineConfig(
            models={"embedding_dimensions": 256},
            cloud={"region": "us-east-1", "development_model_budget_usd": 1},
            embedding={"cache_path": Path(directory) / "vectors.sqlite3"},
        )
        subset = SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes())
        sections = extract_sections(
            load_sources(Path("data/manifests/sources.json")), subset, Path("data/raw")
        )
        chunks = [
            chunk for section in sections for chunk in chunk_section(section, config.chunking)
        ]
        alias = f"s3-rag-compare-check-{uuid4().hex}"
        original, replacement = f"{alias}-original", f"{alias}-replacement"
        config.index.name = alias
        runtime = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        original_close = runtime.close
        runtime.close = Mock(wraps=original_close)
        vector = [1.0] + [0.0] * 255
        with create_search_client(config) as client:
            retained = {
                name: client.indices.get_alias(name=name)
                for name in ("s3-rag-active", "s3-rag-core")
                if client.indices.exists_alias(name=name)
            }
            try:
                for name, records, vectors in (
                    (original, chunks, {chunk.chunk_id: vector for chunk in chunks}),
                    (replacement, chunks[:1], None),
                ):
                    stage = config.model_copy(deep=True)
                    stage.index.name = name
                    bulk_index_chunks(client, stage, records, vectors)
                    client.indices.put_settings(index=name, body={"index.blocks.write": True})
                client.indices.put_alias(index=original, name=alias)
                searches = []

                class SwitchingClient:
                    """Switch publication during the first result, leaving both readers pinned."""

                    indices = client.indices

                    def count(self, **kwargs: object) -> dict[str, object]:
                        return client.count(**kwargs)

                    def search(self, **kwargs: object) -> dict[str, object]:
                        result = client.search(**kwargs)
                        searches.append(kwargs["index"])
                        if len(searches) == 1:
                            client.indices.update_aliases(
                                body={
                                    "actions": [
                                        {"remove": {"index": original, "alias": alias}},
                                        {"add": {"index": replacement, "alias": alias}},
                                    ]
                                }
                            )
                        return result

                app = create_app(config)
                app.dependency_overrides[get_search_client] = lambda: SwitchingClient()
                factory = Mock(return_value=runtime)
                query = "What happens when I delete a versioned object?"
                payload = json.dumps({"embedding": vector, "inputTextTokenCount": 12}).encode()
                with (
                    Stubber(runtime) as stub,
                    patch("rag.comparison.create_bedrock_client", factory),
                    TestClient(app) as http,
                ):
                    stub.add_response(
                        "invoke_model",
                        {
                            "body": StreamingBody(io.BytesIO(payload), len(payload)),
                            "contentType": "application/json",
                        },
                        {
                            "modelId": config.models.embedding_model_id,
                            "contentType": "application/json",
                            "accept": "application/json",
                            "body": json.dumps(
                                {"inputText": query, "dimensions": 256, "normalize": True}
                            ),
                        },
                    )
                    body = {
                        "query": query,
                        "modes": ["bm25", "vector"],
                        "scope": {
                            "permitted_guides": ["s3-userguide"],
                            "source_scope": "general-purpose",
                            "corpus_id": subset.corpus_id,
                        },
                    }
                    response = http.post("/retrieval/compare", json=body)
                    assert response.status_code == 200, response.text
                    first = response.json()
                    assert first["snapshot_index"] == original
                    assert [arm["status"] for arm in first["arms"]] == ["ok", "ok"]
                    assert searches == [original, original]
                    expected = {
                        chunk.chunk_id: chunk
                        for chunk in chunks
                        if chunk.provenance.guide == "s3-userguide"
                    }
                    for arm in first["arms"]:
                        hits = arm["result"]["hits"]
                        assert {hit["chunk"]["chunk_id"] for hit in hits} == set(expected)
                        assert all(
                            hit["chunk"]["original_text"]
                            == expected[hit["chunk"]["chunk_id"]].original_text
                            for hit in hits
                        )
                    assert [arm["model_calls"] for arm in first["arms"]] == [0, 1]
                    assert [arm["model_cost_usd"] for arm in first["arms"]] == [0, None]
                    assert first["arms"][1]["known_embedding_input_tokens"] == 12
                    stub.assert_no_pending_responses()
                    runtime.close.assert_called_once()
                    factory.assert_called_once()
                    factory.reset_mock()
                    second = http.post("/retrieval/compare", json=body).json()
                    assert second["snapshot_index"] == replacement
                    assert [arm["status"] for arm in second["arms"]] == ["ok", "error"]
                    assert second["arms"][1]["error_code"] == "vector_index_not_ready"
                    third = http.post(
                        "/retrieval/compare", json=body | {"modes": ["vector", "bm25"]}
                    ).json()
                    assert [arm["status"] for arm in third["arms"]] == ["error", "ok"]
                    factory.assert_not_called()
                    assert all(arm["model_calls"] == 0 for arm in second["arms"] + third["arms"])
                assert all(
                    client.indices.get_alias(name=name) == value for name, value in retained.items()
                )
            finally:
                if not runtime.close.called:
                    runtime.close()
                for name in (original, replacement):
                    if client.indices.exists(index=name):
                        client.indices.delete(index=name)
        return {
            "live_alias_switch": "pinned",
            "scope_and_original_sources": "passed",
            "native_embedding_stub_calls": 1,
            "text_only_preflight": "passed",
            "partial_failure_both_orders": "passed",
            "owned_client_cleanup": "passed",
            "provider_calls": 0,
        }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
