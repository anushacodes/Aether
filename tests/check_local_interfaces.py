"""Compare CLI, JSON and SSE over real isolated search with native model schemas/stubs."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

import boto3
from botocore.stub import Stubber
from botocore.validate import validate_parameters
from fastapi.testclient import TestClient

from rag.api.app import create_app, get_search_client
from rag.artifacts import pipeline_fingerprint
from rag.chunking import chunk_section
from rag.cli import main
from rag.config import PipelineConfig
from rag.corpus import extract_sections, load_sources
from rag.indexing import bulk_index_chunks
from rag.models import SubsetManifest
from rag.search_store import create_search_client


class ProviderEvents:
    """The event transport is simulated; generation/citation handling remains real."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, object]]:
        yield {"contentBlockDelta": {"delta": {"text": self.text}}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}

    def close(self) -> None:
        self.closed = True


def _frames(text: str) -> list[tuple[str, dict[str, object]]]:
    frames = []
    for frame in text.split("\n\n"):
        if frame.startswith("event: "):
            event, data = frame.split("\ndata: ", 1)
            frames.append((event.removeprefix("event: "), json.loads(data)))
    return frames


def check() -> dict[str, object]:
    """Remove only owned data and prove interface semantics, not live-provider quality."""
    with TemporaryDirectory(prefix="aether-interface-parity-") as directory:
        temporary = Path(directory)
        index = f"s3-rag-interface-check-{uuid4().hex}"
        alias = f"{index}-active"
        config = PipelineConfig(
            index={"name": alias},
            models={"generation_model_id": "test-model"},
            cloud={"region": "us-east-1", "development_model_budget_usd": 1},
        )
        subset = SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes())
        sections = extract_sections(
            load_sources(Path("data/manifests/sources.json")), subset, Path("data/raw")
        )
        chunks = [
            chunk for section in sections for chunk in chunk_section(section, config.chunking)
        ]
        requests: list[dict[str, object]] = []
        streams: list[ProviderEvents] = []
        runtimes = []
        stubs = []
        answer_text = ""

        def runtime_factory(*args: object, **kwargs: object) -> object:
            runtime = boto3.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )
            stub = Stubber(runtime)
            stub.add_response(
                "converse",
                {
                    "output": {
                        "message": {"role": "assistant", "content": [{"text": answer_text}]}
                    },
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                    "metrics": {"latencyMs": 1},
                },
            )
            stub.activate()

            def observe(params: dict[str, object], **kwargs: object) -> None:
                requests.append(params)

            runtime.meta.events.register("before-parameter-build.bedrock-runtime.Converse", observe)

            def streamed(**params: object) -> dict[str, object]:
                validate_parameters(
                    params, runtime.meta.service_model.operation_model("ConverseStream").input_shape
                )
                requests.append(params)
                # Keep Stubber active throughout: an unexpected SDK operation must fail
                # locally rather than falling through to a provider endpoint.
                events = ProviderEvents(answer_text)
                streams.append(events)
                return {"stream": events}

            runtime.converse_stream = streamed
            runtime.close = Mock(wraps=runtime.close)
            runtimes.append(runtime)
            stubs.append(stub)
            return runtime

        with create_search_client(config) as search:
            retained = {
                name: search.indices.get_alias(name=name)
                for name in ("s3-rag-active", "s3-rag-core")
                if search.indices.exists_alias(name=name)
            }
            try:
                staging = config.model_copy(deep=True)
                staging.index.name = index
                assert bulk_index_chunks(search, staging, chunks) == len(chunks)
                search.indices.put_settings(index=index, body={"index.blocks.write": True})
                search.indices.put_alias(index=index, name=alias)
                app = create_app(config)
                app.dependency_overrides[get_search_client] = lambda: search
                with (
                    patch("rag.cli.load_config", return_value=config),
                    patch("rag.pipeline.create_bedrock_client", runtime_factory),
                    patch("rag.streaming.create_bedrock_client", runtime_factory),
                    TestClient(app) as http,
                ):
                    for case, guide in (
                        ("guide_answer", "s3-userguide"),
                        ("api_answer", "s3-api"),
                        ("empty_scope", "s3-api"),
                        ("invalid_citation", "s3-userguide"),
                    ):
                        cited = next(chunk for chunk in chunks if chunk.provenance.guide == guide)
                        answer_text = f"Synthetic parity response [{cited.chunk_id}]."
                        if case == "invalid_citation":
                            answer_text = "Synthetic unsupported citation [missing-parity-source]."
                        query = "What happens when I delete a versioned object?"
                        corpus = (
                            "s3-not-in-this-check" if case == "empty_scope" else subset.corpus_id
                        )
                        saved = temporary / case
                        command = [
                            "ask",
                            query,
                            "--guide",
                            guide,
                            "--corpus-id",
                            corpus,
                            "--save-run",
                            str(saved),
                        ]
                        output, errors = io.StringIO(), io.StringIO()
                        before = len(requests)
                        with redirect_stdout(output), redirect_stderr(errors):
                            exit_code = main(command)
                        body = {
                            "query": query,
                            "scope": {"permitted_guides": [guide], "corpus_id": corpus},
                        }
                        complete = http.post("/search", json=body)
                        streamed = http.post("/search/stream", json=body)
                        frames = _frames(streamed.text)
                        if case == "invalid_citation":
                            # Complete answers preserve a failed result/usage artifact, while
                            # streaming must reject provisional tokens with a terminal error.
                            assert exit_code == 1 and saved.exists() and not errors.getvalue()
                            assert complete.status_code == 200
                            cli_result = json.loads(output.getvalue())
                            assert cli_result["status"] == complete.json()["status"] == "failed"
                            assert {
                                key: value
                                for key, value in cli_result.items()
                                if key != "timings_ms"
                            } == {
                                key: value
                                for key, value in complete.json().items()
                                if key != "timings_ms"
                            }
                            assert cli_result["citation_ids"] == []
                            assert "outside its evidence" in cli_result["answer"]
                            saved_failure = json.loads(
                                (saved / "answer.json").read_text(encoding="utf-8")
                            )
                            assert saved_failure["result"] == cli_result
                            assert [name for name, _ in frames][-1] == "error"
                            assert frames[-1][1]["code"] == "invalid_citations"
                            assert not any(name == "done" for name, _ in frames)
                        else:
                            assert exit_code == 0 and complete.status_code == 200
                            terminal = [data for name, data in frames if name == "done"]
                            assert len(terminal) == 1 and not any(
                                name == "error" for name, _ in frames
                            )
                            results = [json.loads(output.getvalue()), complete.json(), terminal[0]]
                            # Independent wall-clock timings differ; compare all semantic fields.
                            normalized = [
                                {key: value for key, value in result.items() if key != "timings_ms"}
                                for result in results
                            ]
                            assert normalized[0] == normalized[1] == normalized[2]
                            assert all(result["snapshot_index"] == index for result in results)
                            artifact = json.loads(
                                (saved / "answer.json").read_text(encoding="utf-8")
                            )
                            assert artifact["result"] == results[0]
                            assert artifact["context_passages"] == results[0]["evidence"]
                            assert artifact["pipeline_fingerprint"] == pipeline_fingerprint(config)
                            assert all(
                                item["provenance"]["guide"] == guide
                                for item in artifact["context_passages"]
                            )
                            if case == "empty_scope":
                                assert results[0]["status"] == "insufficient_evidence"
                                assert not results[0]["evidence"] and len(requests) == before
                            else:
                                assert results[0]["status"] == "answered" and results[0]["evidence"]
                        if case != "empty_scope":
                            actual = requests[before:]
                            assert len(actual) == 3 and actual[0] == actual[1] == actual[2]
                            references = json.loads(actual[0]["messages"][0]["content"][0]["text"])[
                                "reference_data"
                            ]
                            expected = {
                                chunk.chunk_id: chunk.original_text
                                for chunk in chunks
                                if chunk.provenance.guide == guide
                            }
                            assert references and all(
                                item["text"] == expected[item["citation_id"]] for item in references
                            )
                assert len(runtimes) == 9 and all(
                    runtime.close.call_count == 1 for runtime in runtimes
                )
                assert len(streams) == 3 and all(stream.closed for stream in streams)
                assert all(
                    search.indices.get_alias(name=name) == value for name, value in retained.items()
                )
            finally:
                for stub in stubs:
                    stub.deactivate()
                for runtime in runtimes:
                    if not runtime.close.called:
                        runtime.close()
                if search.indices.exists(index=index):
                    search.indices.delete(index=index)
                assert not search.indices.exists_alias(name=alias)
        return {
            "interfaces": ["CLI", "JSON", "SSE"],
            "scenarios": ["guide answer", "API answer", "empty scoped corpus", "invalid citation"],
            "semantic_results_and_exact_prompts": "matched",
            "concrete_snapshot_and_original_context": "matched",
            "native_sdk_stub_converse_responses": 6,
            "simulated_schema_checked_streams": 3,
            "provider_calls": 0,
            "baseline_aliases": "preserved",
            "owned_index_cleanup": "passed",
        }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
