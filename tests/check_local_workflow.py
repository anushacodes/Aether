"""Verify real local hybrid workflow mechanics with SDK stubs, never AWS/model requests."""

from __future__ import annotations

import json
from contextlib import closing
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber
from fastapi.testclient import TestClient

from rag.api.app import create_app, get_search_client
from rag.chunking import chunk_section
from rag.config import PipelineConfig, load_config
from rag.corpus import extract_sections, load_sources
from rag.indexing import bulk_index_chunks
from rag.models import SearchRequest, SearchScope, SubsetManifest
from rag.pipeline import answer_question
from rag.search_store import create_search_client
from rag.workflow import expanded_query


def converse_response(text: str) -> dict[str, object]:
    """Use the actual SDK response shape, including usage and stop metadata."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }


def check(config: PipelineConfig, temporary: Path) -> dict[str, object]:
    config.embedding.cache_path = temporary / "embeddings.sqlite3"
    config.models.embedding_dimensions = 256
    config.models.generation_model_id = "test-model"
    config.cloud.region = "us-east-1"
    config.cloud.development_model_budget_usd = 1  # Stub-only configuration.
    token = uuid4().hex
    alias = f"s3-rag-workflow-check-{token}"
    old_index = f"s3-rag-workflow-old-{token}"
    new_index = f"s3-rag-workflow-new-{token}"
    sources = load_sources(config.ingestion.sources_path)
    subset = SubsetManifest.model_validate_json(
        config.ingestion.approved_subsets["pilot"].read_bytes()
    )
    sections = extract_sections(sources, subset, config.ingestion.raw_dir)
    chunks = [chunk for section in sections for chunk in chunk_section(section, config.chunking)]
    vector = [1.0] + [0.0] * 255
    original = "What happens when DeleteObject deletes an object in a versioned bucket?"
    refined = expanded_query(original, ["delete marker"])
    cited = next(chunk for chunk in chunks if chunk.provenance.guide == "s3-userguide")
    answer = f"Synthetic integration answer [{cited.chunk_id}]."
    first_assessment = json.dumps(
        {
            "decision": "retry",
            "missing_evidence": "missing_context",
            "additional_terms": ["delete marker"],
        }
    )
    final_assessment = json.dumps(
        {"decision": "answer", "missing_evidence": "none", "additional_terms": []}
    )
    with create_search_client(config) as search:
        try:
            config.index.name = old_index
            bulk_index_chunks(search, config, chunks, {chunk.chunk_id: vector for chunk in chunks})
            config.index.name = new_index
            other = [chunk for chunk in chunks if chunk.provenance.guide == "s3-api"]
            bulk_index_chunks(search, config, other, {chunk.chunk_id: vector for chunk in other})
            search.indices.update_aliases(
                body={"actions": [{"add": {"index": old_index, "alias": alias}}]}
            )
            config.index.name = alias
            runtime = boto3.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
                config=Config(retries={"total_max_attempts": 1}),
            )
            model_responses = 0

            def switch_after_first_assessment(**_: object) -> None:
                nonlocal model_responses
                model_responses += 1
                if model_responses == 1:
                    search.indices.update_aliases(
                        body={
                            "actions": [
                                {"remove": {"index": old_index, "alias": alias}},
                                {"add": {"index": new_index, "alias": alias}},
                            ]
                        }
                    )

            runtime.meta.events.register(
                "after-call.bedrock-runtime.Converse", switch_after_first_assessment
            )
            with closing(runtime), Stubber(runtime) as stubber:
                for query, assessment in [
                    (original, first_assessment),
                    (refined, final_assessment),
                ]:
                    encoded = json.dumps({"embedding": vector, "inputTextTokenCount": 1}).encode()
                    stubber.add_response(
                        "invoke_model",
                        {
                            "body": StreamingBody(BytesIO(encoded), len(encoded)),
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
                    stubber.add_response(
                        "converse",
                        converse_response(assessment),
                        {
                            "modelId": "test-model",
                            "system": ANY,
                            "messages": ANY,
                            "inferenceConfig": {
                                "maxTokens": config.workflow.assessment_max_tokens,
                                "temperature": 0,
                            },
                        },
                    )
                stubber.add_response(
                    "converse",
                    converse_response(answer),
                    {
                        "modelId": "test-model",
                        "system": ANY,
                        "messages": ANY,
                        "inferenceConfig": {
                            "maxTokens": config.generation.max_output_tokens,
                            "temperature": 0,
                        },
                    },
                )
                request = SearchRequest(
                    query=original,
                    mode="hybrid",
                    workflow=True,
                    scope=SearchScope(
                        permitted_guides=["s3-userguide"], corpus_id=subset.corpus_id
                    ),
                )
                result = answer_question(search, config, request, runtime)
                stubber.assert_no_pending_responses()
            assert result.status == "answered"
            assert result.workflow is not None
            assert result.workflow.model_calls == 5 and result.workflow.retrieval_passes == 2
            assert result.workflow.snapshot_index == old_index
            assert all(chunk.provenance.guide == "s3-userguide" for chunk in result.evidence)
            assert result.usage_tokens["embedding_input"] == 2
            assert search.indices.get_alias(name=alias).keys() == {new_index}
            assert result.workflow.steps[1].query.startswith(original)
            search.indices.update_aliases(
                body={
                    "actions": [
                        {"remove": {"index": new_index, "alias": alias}},
                        {"add": {"index": old_index, "alias": alias}},
                    ]
                }
            )
            streamed_runtime = Mock()
            streamed_runtime.converse.return_value = converse_response(final_assessment)
            provider = MagicMock()
            provider.__iter__.side_effect = lambda: iter(
                [
                    {"contentBlockDelta": {"delta": {"text": answer}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
                ]
            )
            streamed_runtime.converse_stream.return_value = {"stream": provider}
            application = create_app(config)
            application.dependency_overrides[get_search_client] = lambda: search
            with (
                TestClient(application) as api,
                patch("rag.workflow_limits.create_bedrock_client", return_value=streamed_runtime),
            ):
                streamed = api.post("/search/stream", json={"query": original, "workflow": True})
            assert "event: token" in streamed.text and "event: done" in streamed.text
            assert "event: error" not in streamed.text
            provider.close.assert_called_once()
            streamed_runtime.close.assert_called_once()
            return {
                "passed": [
                    "native filtered hybrid search",
                    "two query embeddings",
                    "structured retry and answer",
                    "one pinned snapshot across an alias switch",
                    "original scope and identifiers",
                    "five-call ceiling",
                    "real-search SSE completion and cleanup",
                ],
                "stubbed_graph_model_calls": 5,
                "real_aws_requests": 0,
            }
        finally:
            for index in (old_index, new_index):
                if search.indices.exists(index=index):
                    search.indices.delete(index=index)


def main() -> None:
    config = load_config(Path("config.toml"))
    if config.opensearch_url.host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Workflow verification requires local OpenSearch")
    with TemporaryDirectory(prefix="s3-rag-workflow-") as temporary:
        print(json.dumps(check(config, Path(temporary)), indent=2))


if __name__ == "__main__":
    main()
