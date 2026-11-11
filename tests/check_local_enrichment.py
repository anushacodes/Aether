"""Check isolated metadata indexes against real local OpenSearch with zero provider calls."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

from rag.artifacts import pipeline_fingerprint
from rag.chunking import chunk_section
from rag.cli import main
from rag.config import PipelineConfig
from rag.corpus import extract_sections, load_sources
from rag.embeddings import Embedder
from rag.enriched_index import build_enriched_index, load_enriched_inputs
from rag.enrichment import run_enrichment
from rag.eval.experiments import PRESETS
from rag.generation import assemble_context
from rag.indexing import bulk_index_chunks, ensure_index
from rag.models import SearchScope, SubsetManifest
from rag.pipeline import retrieve
from rag.retrieval import bm25_search, vector_search
from rag.search_store import create_search_client


def check() -> dict[str, object]:
    """Prove field separation, bounded fallback, filtered retrieval, vectors, and cleanup."""
    with TemporaryDirectory(prefix="aether-enriched-check-") as directory:
        root = Path(directory)
        config = PipelineConfig(
            cloud={"region": "us-east-1", "development_model_budget_usd": 1},
            embedding={"cache_path": root / "vectors.sqlite3"},
            enrichment={"model_id": "stub-model", "cache_path": root / "metadata.sqlite3"},
        )
        sources = load_sources(Path("data/manifests/sources.json"))
        subset = SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes())
        sections = extract_sections(sources, subset, Path("data/raw"))
        chunks = [
            chunk for section in sections for chunk in chunk_section(section, config.chunking)
        ]
        assert len(chunks) == 3
        assert "inferredneedle" not in " ".join(c.original_text for c in chunks).lower()
        metadata = {
            "summary": "inferredneedle",
            "topic": "versioning",
            "intent": "concept",
            "related_concepts": [],
            "example_phrasings": [],
        }
        normal = {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": json.dumps(metadata)}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        runtime = Mock()
        runtime.converse.side_effect = [normal, normal | {"stopReason": "max_tokens"}, normal]
        run = root / "run"
        prepared = run_enrichment(config, chunks, run, runtime=runtime)
        assert prepared["outcomes"] == {"enriched": 2, "content_only": 1}
        inputs = load_enriched_inputs(run)
        assert inputs.embedding_texts[1] == chunks[1].original_text
        embedded_texts: list[str] = []

        def embed(**kwargs: object) -> dict[str, object]:
            embedded_texts.append(json.loads(kwargs["body"])["inputText"])
            vector = [1.0] + [0.0] * (config.models.embedding_dimensions - 1)
            return {
                "body": io.BytesIO(
                    json.dumps({"embedding": vector, "inputTextTokenCount": 2}).encode()
                )
            }

        runtime.invoke_model.side_effect = embed
        with create_search_client(config) as client:
            # Preserve both user-visible corpora; only generated experiment names are owned here.
            aliases = ("s3-rag-active", "s3-rag-core")
            before = {
                name: client.indices.get_alias(name=name)
                for name in aliases
                if client.indices.exists_alias(name=name)
            }
            owned: list[str] = []
            try:
                config_path = root / "experiment.toml"
                config_path.write_text(
                    f"opensearch_url={json.dumps(str(config.opensearch_url))}\n", encoding="utf-8"
                )
                captured = io.StringIO()
                with redirect_stdout(captured):
                    assert (
                        main(
                            [
                                "--config",
                                str(config_path),
                                "index",
                                "enriched",
                                "--run",
                                str(run),
                                "--output",
                                str(root / "plain"),
                            ]
                        )
                        == 0
                    )
                plain = json.loads(captured.getvalue())
                owned.append(plain["index_name"])
                assert plain["document_count"] == 3 and plain["enriched_count"] == 2
                read = config.model_copy(deep=True)
                read.index.name = plain["index_name"]
                read.index.enrichment_fingerprint = plain["enrichment_fingerprint"]
                assert pipeline_fingerprint(read) != pipeline_fingerprint(config)
                scope = SearchScope()
                assert not bm25_search(client, read, "inferredneedle", scope)
                read.retrieval.metadata_boost = 2
                boosted = retrieve(client, read, "inferredneedle", scope, "bm25").hits
                assert {hit.chunk.chunk_id for hit in boosted} == {
                    chunks[0].chunk_id,
                    chunks[2].chunk_id,
                }
                context = assemble_context(boosted, config.retrieval)
                assert "inferredneedle" not in context.text
                assert {c.model_dump_json() for c in context.evidence}.issubset(
                    {c.model_dump_json() for c in chunks}
                )
                api_hits = bm25_search(
                    client,
                    read,
                    "inferredneedle",
                    SearchScope(permitted_guides=["s3-api"]),
                )
                assert {hit.chunk.chunk_id for hit in api_hits} == {
                    chunks[i].chunk_id for i in (0, 2) if chunks[i].provenance.guide == "s3-api"
                }
                assert not bm25_search(
                    client, read, "inferredneedle", SearchScope(corpus_id="missing")
                )
                config_path.write_text(
                    f"opensearch_url={json.dumps(str(config.opensearch_url))}\n"
                    f"[index]\nname={json.dumps(read.index.name)}\n"
                    f"enrichment_fingerprint={json.dumps(read.index.enrichment_fingerprint)}\n"
                    "[retrieval]\nmetadata_boost=2\n",
                    encoding="utf-8",
                )
                captured = io.StringIO()
                with redirect_stdout(captured):
                    assert (
                        main(["--config", str(config_path), "search", "query", "inferredneedle"])
                        == 0
                    )
                assert len(json.loads(captured.getvalue())["hits"]) == 2
                # A fresh CI cluster has no published pilot; own this negative-control index.
                baseline_name = f"s3-rag-preset-baseline-{uuid4().hex}"
                owned.append(baseline_name)
                baseline = config.model_copy(deep=True)
                baseline.index.name = baseline_name
                ensure_index(client, baseline)
                baseline.retrieval.metadata_boost = 2
                try:
                    bm25_search(client, baseline, "inferredneedle", scope)
                except ValueError as error:
                    assert "separate enriched index" in str(error)
                else:
                    raise AssertionError(
                        "A baseline index must not silently accept metadata boosts"
                    )
                vectorized = build_enriched_index(
                    client,
                    config,
                    run,
                    root / "vectors",
                    embedder=Embedder(config, runtime),
                )
                owned.append(vectorized["index_name"])
                assert embedded_texts == inputs.embedding_texts
                read.index.name = vectorized["index_name"]
                read.index.enrichment_fingerprint = vectorized["enrichment_fingerprint"]
                hits = vector_search(client, read, [1.0] + [0.0] * 1023, scope)
                assert {hit.chunk.chunk_id for hit in hits} == {c.chunk_id for c in chunks}
                mismatched = read.model_copy(deep=True)
                mismatched.index.enrichment_fingerprint = None
                unused_embedder = Mock()
                try:
                    retrieve(client, mismatched, "inferredneedle", scope, "vector", unused_embedder)
                except ValueError as error:
                    assert "fingerprint" in str(error)
                else:
                    raise AssertionError(
                        "Enriched vectors require their explicit input fingerprint"
                    )
                unused_embedder.embed_text.assert_not_called()
                settings = client.indices.get_settings(index=read.index.name)
                assert settings[read.index.name]["settings"]["index"]["blocks"]["write"] == "true"

                # Exercise every declared treatment over identical real source records.
                base = config.model_copy(deep=True)
                base.index.name = baseline_name
                base.reranking.model_arn = (
                    "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"
                )
                vector = [1.0] + [0.0] * 1023
                bulk_index_chunks(client, base, chunks, {c.chunk_id: vector for c in chunks})
                client.indices.put_settings(index=baseline_name, body={"index.blocks.write": True})
                sections_file, chunks_file = root / "sections.jsonl", run / "chunks.jsonl"
                # chunks.jsonl already belongs to the verified metadata run; do not rewrite it.
                sections_file.write_text(
                    "".join(section.model_dump_json() + "\n" for section in sections),
                    encoding="utf-8",
                )
                ranker = Mock()
                ranker.rerank.side_effect = lambda **kwargs: {
                    "results": [
                        {"index": i, "relevanceScore": float(len(kwargs["sources"]) - i)}
                        for i in range(len(kwargs["sources"]))
                    ]
                }
                configuration_ids: set[str] = set()
                shared_inputs = None
                with (
                    patch("rag.cli.load_config", return_value=base),
                    patch("rag.cli.create_bedrock_client", return_value=runtime),
                    patch("rag.reranking.create_reranking_client", return_value=ranker),
                ):
                    for preset in PRESETS:
                        captured = io.StringIO()
                        output = root / preset.identifier
                        command = [
                            "eval",
                            "retrieval",
                            "--experiment",
                            preset.identifier,
                            "--allow-drafts",
                            "--sections",
                            str(sections_file),
                            "--chunks",
                            str(chunks_file),
                            "--output",
                            str(output),
                        ]
                        if preset.metadata_vectors:
                            command.extend(
                                ["--enriched-index", str(root / "vectors" / "index.json")]
                            )
                        with redirect_stdout(captured):
                            assert main(command) == 0
                        summary = json.loads(captured.getvalue())
                        configuration_ids.add(summary["configuration_id"])
                        common = {
                            key: summary["inputs"][key]
                            for key in (
                                "dataset_sha256",
                                "split",
                                "chunk_ids",
                                "chunking",
                                "models",
                                "retrieval",
                            )
                        }
                        if shared_inputs is None:
                            shared_inputs = common
                        assert common == shared_inputs
                        assert summary["scored"] == 0 and summary["questions"] == 3
                        assert (
                            summary["experiment"]["definition"]["identifier"] == preset.identifier
                        )
                        assert summary["inputs"]["mode"] == preset.mode
                        assert summary["inputs"]["reranking"]["enabled"] == preset.reranking
                        assert (
                            bool(summary["inputs"]["enrichment_fingerprint"])
                            == preset.metadata_vectors
                        )
                        assert summary["inputs"]["retrieval"]["metadata_boost"] == 0
                        intent = json.loads(
                            (output / "experiment.json").read_text(encoding="utf-8")
                        )
                        assert intent == summary["experiment"]
                        rows = [
                            json.loads(line)
                            for line in (output / "results.jsonl")
                            .read_text(encoding="utf-8")
                            .splitlines()
                        ]
                        assert all("inferredneedle" not in row["context"]["text"] for row in rows)
                assert ranker.rerank.called
                assert len(configuration_ids) == len(PRESETS)

                def fail_build(
                    client: object, staged: PipelineConfig, *_: object, **__: object
                ) -> int:
                    ensure_index(client, staged, inferred_metadata=True)
                    raise ValueError("injected staging failure")

                with patch("rag.enriched_index.bulk_index_chunks", side_effect=fail_build):
                    try:
                        build_enriched_index(client, config, run, root / "failed")
                    except ValueError as error:
                        assert "injected staging failure" in str(error)
                    else:
                        raise AssertionError("Injected failure must be reported")
                failed = json.loads((root / "failed" / "intent.json").read_text(encoding="utf-8"))
                assert not client.indices.exists(index=failed["index_name"])
                assert {name: client.indices.get_alias(name=name) for name in before} == before
            finally:
                for name in owned:
                    if client.indices.exists(index=name):
                        client.indices.delete(index=name)
    return {
        "passed": [
            "verified metadata inputs",
            "content-only fallback keeps every chunk",
            "metadata boost is explicit",
            "original-only context",
            "authoritative guide filters",
            "baseline metadata mismatch rejected",
            "prefixed vectors with stable source IDs",
            "write-blocked experiments",
            "B0–B5 controlled shared-input treatments",
            "failed staging cleanup",
            "baseline aliases preserved",
        ],
        "real_aws_requests": 0,
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
