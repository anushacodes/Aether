"""Adapt CLI input to the validated corpus pipeline and deliberate terminal output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack, closing
from pathlib import Path
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
from opensearchpy import TransportError

from rag import __version__
from rag.artifacts import build_manifest, save_answer_run, save_build_manifest
from rag.aws_clients import create_bedrock_client
from rag.chunking import chunk_inventory, chunk_section
from rag.config import ChunkingConfig, PipelineConfig, load_config
from rag.corpus import (
    download_sources,
    extract_sections,
    heading_inventory,
    load_sources,
    write_jsonl,
)
from rag.embeddings import Embedder, embedding_manifest
from rag.eval.dataset import load_dataset
from rag.eval.runs import run_retrieval_evaluation
from rag.indexing import bulk_index_chunks, ensure_index, load_chunks
from rag.ingestion import ingestion_catalog, submit_ingestion
from rag.jobs import IngestionRequest, JobStore
from rag.models import ChunkRecord, SearchRequest, SearchScope, SectionRecord, SubsetManifest
from rag.pipeline import answer_question, retrieve
from rag.publication import active_index, build_snapshot, publish_snapshot
from rag.search_store import check_search_store, create_search_client
from rag.worker import WorkerBusyError, run_one
from rag.workflow_limits import WorkflowBudgetError


def _add_question_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query")
    parser.add_argument("--guide", action="append", choices=("s3-userguide", "s3-api"))
    parser.add_argument("--corpus-id")
    parser.add_argument("--mode", choices=("bm25", "vector", "hybrid"), default="bm25")


def build_parser() -> argparse.ArgumentParser:
    """Expose implemented commands without requiring cloud services at startup."""
    parser = argparse.ArgumentParser(
        prog="s3-rag", description="Build an evidence-backed AWS S3 documentation corpus."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="Optional TOML pipeline configuration")
    commands = parser.add_subparsers(dest="command")
    worker = commands.add_parser("worker", help="Run the single local durable ingestion worker")
    worker.add_argument(
        "--once", action="store_true", help="Execute one queued/recoverable job and exit"
    )
    jobs = commands.add_parser("jobs", help="Submit and inspect local durable corpus mutations")
    job_operations = jobs.add_subparsers(dest="operation", required=True)
    job_operations.add_parser("catalog", help="Show the active revision and approved section IDs")
    submit = job_operations.add_parser("submit")
    submit.add_argument(
        "--key", required=True, help="Unique idempotency key; reuse only for the same intent"
    )
    submit.add_argument(
        "--base-revision",
        required=True,
        help="Concrete snapshot name, or 'none' for first publication",
    )
    submit.add_argument("--catalog-id", default="pilot")
    submit.add_argument("--delete-section", action="append", default=[])
    submit.add_argument("--with-embeddings", action="store_true")
    for operation in ("status", "cancel", "replay"):
        job_operations.add_parser(operation).add_argument("job_id")
    config = commands.add_parser("config", help="Validate configuration without making AWS calls")
    config.add_argument("--cloud", action="store_true", help="Require paid-call configuration")
    config.add_argument(
        "--require-bucket", action="store_true", help="Also require an artifact bucket"
    )
    search = commands.add_parser("search", help="Check the configured OpenSearch service")
    search_operations = search.add_subparsers(dest="operation", required=True)
    search_operations.add_parser(
        "check", help="Check health, index/search a temporary document, then remove it"
    )
    query = search_operations.add_parser("query", help="Retrieve original source passages")
    _add_question_arguments(query)
    ask = commands.add_parser("ask", help="Answer from retrieved passages with source citations")
    _add_question_arguments(ask)
    ask.add_argument(
        "--save-run", type=Path, help="Save question and exact answer context in a new directory"
    )
    ask.add_argument(
        "--workflow",
        action="store_true",
        help="Opt into bounded evidence assessment and one retrieval retry",
    )
    evaluation = commands.add_parser(
        "eval", help="Validate and evaluate independent quality labels"
    )
    eval_operations = evaluation.add_subparsers(dest="operation", required=True)
    validate = eval_operations.add_parser("validate", help="Check labels and prevent split leakage")
    validate.add_argument("--dataset", type=Path, default=Path("data/eval/pilot.json"))
    validate.add_argument(
        "--allow-drafts", action="store_true", help="Inspect drafts without certifying labels"
    )
    evaluate = eval_operations.add_parser(
        "retrieval", help="Capture rankings and reviewed-label metrics"
    )
    evaluate.add_argument("--dataset", type=Path, default=Path("data/eval/pilot.json"))
    evaluate.add_argument(
        "--sections", type=Path, default=Path("data/processed/pilot-sections.jsonl")
    )
    evaluate.add_argument("--chunks", type=Path, default=Path("data/processed/pilot-chunks.jsonl"))
    evaluate.add_argument("--mode", choices=("bm25", "vector", "hybrid"), default="bm25")
    evaluate.add_argument("--split", choices=("development", "held_out"), default="development")
    evaluate.add_argument(
        "--allow-drafts",
        action="store_true",
        help="Capture development rankings without quality claims",
    )
    evaluate.add_argument(
        "--output", type=Path, help="New run directory; existing runs are never overwritten"
    )
    embed = commands.add_parser(
        "embed", help="Embed chunk records through Bedrock with a disk cache"
    )
    embed.add_argument("--chunks", type=Path, default=Path("data/processed/pilot-chunks.jsonl"))
    embed.add_argument("--output", type=Path, default=Path("data/processed/pilot-embeddings.jsonl"))
    index = commands.add_parser("index", help="Create and populate the configured corpus index")
    index_operations = index.add_subparsers(dest="operation", required=True)
    index_operations.add_parser("create", help="Create the index or verify embedding compatibility")
    build = index_operations.add_parser("build", help="Index chunks with stable source identities")
    build.add_argument("--chunks", type=Path, default=Path("data/processed/pilot-chunks.jsonl"))
    build.add_argument(
        "--with-embeddings", action="store_true", help="Generate/reuse Bedrock vectors"
    )
    build.add_argument("--manifest", type=Path, default=Path("data/processed/pilot-build.json"))
    publish = index_operations.add_parser(
        "publish", help="Build a verified immutable snapshot and switch the active alias"
    )
    publish.add_argument("--sources", type=Path, default=Path("data/manifests/sources.json"))
    publish.add_argument("--subset", type=Path, default=Path("data/manifests/pilot.json"))
    publish.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    publish.add_argument("--with-embeddings", action="store_true")
    corpus = commands.add_parser(
        "corpus", help="Fetch, inspect, and extract pinned source snapshots"
    )
    operations = corpus.add_subparsers(dest="operation", required=True)
    for name, description in (
        ("fetch", "Download and SHA-256 verify the pinned source files"),
        ("inventory", "List heuristic heading candidates with raw character offsets"),
        ("extract", "Extract and verify the selected pilot ranges"),
        ("chunk", "Extract and chunk the selected ranges without model calls"),
    ):
        operation = operations.add_parser(name, help=description)
        operation.add_argument("--sources", type=Path, default=Path("data/manifests/sources.json"))
        operation.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
        if name == "inventory":
            operation.add_argument(
                "--output", type=Path, default=Path("data/processed/headings.jsonl")
            )
        if name in {"extract", "chunk"}:
            operation.add_argument("--subset", type=Path, default=Path("data/manifests/pilot.json"))
            operation.add_argument(
                "--output",
                type=Path,
                default=Path(
                    "data/processed/pilot-chunks.jsonl"
                    if name == "chunk"
                    else "data/processed/pilot-sections.jsonl"
                ),
            )
        if name == "chunk":
            operation.add_argument("--strategy", choices=("fixed", "heading-aware"))
            operation.add_argument("--size", type=int, help="Maximum whitespace-token estimate")
            operation.add_argument("--overlap", type=int, help="Desired maximum overlap estimate")
    return parser


def _run_chunking(
    args: argparse.Namespace, sections: list[SectionRecord], base_config: ChunkingConfig
) -> None:
    values = base_config.model_dump()
    for key in ("strategy", "size", "overlap"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    config = ChunkingConfig.model_validate(values)
    chunks: list[ChunkRecord] = []
    for section in sections:
        chunks.extend(chunk_section(section, config))
    write_jsonl(args.output, (chunk.model_dump(mode="json") for chunk in chunks))
    print(json.dumps(chunk_inventory(sections, chunks, config), indent=2))
    print(f"Wrote {len(chunks)} chunks to {args.output}")


def _run_corpus(args: argparse.Namespace, config: PipelineConfig) -> None:
    sources = load_sources(args.sources)
    if args.operation == "fetch":
        for path in download_sources(sources, args.raw_dir):
            print(f"Verified {path}")
        return
    if args.operation == "inventory":
        rows = heading_inventory(sources, args.raw_dir)
        write_jsonl(args.output, rows)
        print(f"Wrote {len(rows)} candidate headings to {args.output}; inspect before selecting.")
        return
    subset = SubsetManifest.model_validate_json(args.subset.read_bytes())
    sections = extract_sections(sources, subset, args.raw_dir)
    if args.operation == "chunk":
        _run_chunking(args, sections, config.chunking)
        return
    write_jsonl(args.output, (section.model_dump(mode="json") for section in sections))
    print(f"Wrote {len(sections)} verified sections to {args.output}")
    for section in sections:
        print(
            f"  {section.section_id}: {len(section.raw_text)} raw characters, "
            f"{len(section.text)} cleaned characters"
        )


def _run_jobs(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Local CLI access relies on filesystem access; HTTP requires administrator auth."""
    if args.operation == "catalog":
        with create_search_client(config) as client:
            print(json.dumps(ingestion_catalog(client, config), indent=2))
        return
    store = JobStore(config.publication.state_path)
    if args.operation == "submit":
        request = IngestionRequest(
            operation="delete" if args.delete_section else "rebuild",
            catalog_id=args.catalog_id,
            base_revision=None if args.base_revision == "none" else args.base_revision,
            with_embeddings=args.with_embeddings,
            remove_section_ids=args.delete_section,
        )
        with create_search_client(config) as client:
            result = submit_ingestion(client, config, store, args.key, request)
    elif args.operation == "cancel":
        result = store.cancel(args.job_id)
    elif args.operation == "replay":
        result = store.replay(args.job_id, config.ingestion.max_attempts)
    else:
        result = store.get(args.job_id)
        if result is None:
            raise ValueError("Job not found")
    print(result.model_dump_json(indent=2))


def _run_worker(args: argparse.Namespace, config: PipelineConfig) -> int:
    with create_search_client(config) as client:
        while True:
            result = run_one(client, config)
            if result is not None:
                print(result.model_dump_json(indent=2), flush=True)
            if args.once:
                return 1 if result is not None and result.status == "failed" else 0
            if result is None:
                time.sleep(2)


def _run_index(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Keep staging and publication separate from mutable development index commands."""
    if args.operation != "publish" and config.index.name == config.publication.alias:
        raise ValueError(
            "Use 'index publish' for the active alias; mutable builds need a separate index"
        )
    with ExitStack() as stack:
        client = stack.enter_context(create_search_client(config))
        if args.operation == "create":
            result = {"index": config.index.name, "created": ensure_index(client, config)}
        else:
            embedder = None
            if args.with_embeddings:
                runtime = stack.enter_context(closing(create_bedrock_client(config)))
                embedder = Embedder(config, runtime)
            if args.operation == "publish":
                base_revision = active_index(client, config.publication.alias)
                snapshot = build_snapshot(
                    client,
                    config,
                    load_sources(args.sources),
                    SubsetManifest.model_validate_json(args.subset.read_bytes()),
                    args.raw_dir,
                    uuid4().hex,
                    embedder=embedder,
                )
                publish_snapshot(client, config, snapshot, base_revision)
                result = snapshot.model_dump(mode="json") | {"alias": config.publication.alias}
            else:
                chunks = load_chunks(args.chunks)
                vectors = embedder.embed_chunks(chunks) if embedder else None
                count = bulk_index_chunks(client, config, chunks, vectors)
                manifest = build_manifest(config, chunks, has_vectors=vectors is not None)
                save_build_manifest(args.manifest, manifest)
                result = {
                    "index": config.index.name,
                    "indexed": count,
                    "vectors": vectors is not None,
                }
    print(json.dumps(result, indent=2))


def _run_evaluation(args: argparse.Namespace, config: PipelineConfig) -> int:
    dataset = load_dataset(args.dataset, allow_drafts=args.allow_drafts)
    if args.operation == "validate":
        print(
            json.dumps(
                {
                    "dataset_id": dataset.dataset_id,
                    "questions": len(dataset.questions),
                    "review_status": dataset.review_status,
                    "sha256": dataset.fingerprint(),
                },
                indent=2,
            )
        )
        return 0
    sections = [
        SectionRecord.model_validate_json(line)
        for line in args.sections.read_bytes().splitlines()
        if line.strip()
    ]
    chunks = load_chunks(args.chunks)
    output = args.output or Path("reports/runs") / f"retrieval-{uuid4().hex}"
    with ExitStack() as stack:
        client = stack.enter_context(create_search_client(config))
        embedder = None
        if args.mode != "bm25":
            runtime = stack.enter_context(closing(create_bedrock_client(config)))
            embedder = Embedder(config, runtime)
        summary = run_retrieval_evaluation(
            client,
            config,
            dataset,
            sections,
            chunks,
            args.mode,
            args.split,
            output,
            embedder=embedder,
            allow_drafts=args.allow_drafts,
        )
    print(json.dumps(summary | {"output": str(output)}, indent=2))
    outcomes = summary["outcomes"]
    return 1 if isinstance(outcomes, dict) and outcomes.get("error", 0) else 0


def main(argv: list[str] | None = None) -> int:
    """Run the selected command, returning useful errors for expected input/I/O failures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        config = load_config(args.config)
        if args.command == "jobs":
            _run_jobs(args, config)
        elif args.command == "worker":
            return _run_worker(args, config)
        elif args.command == "config":
            if args.cloud or args.require_bucket:
                config.require_cloud_settings(require_bucket=args.require_bucket)
            print("Configuration valid (no network requests made).")
        elif args.command == "search":
            with create_search_client(config) as client:
                if args.operation == "check":
                    result = check_search_store(client)
                else:
                    scope = SearchScope(
                        permitted_guides=args.guide or ["s3-userguide", "s3-api"],
                        corpus_id=args.corpus_id,
                    )
                    if args.mode == "bm25":
                        retrieved = retrieve(client, config, args.query, scope, args.mode)
                    else:
                        with closing(create_bedrock_client(config)) as runtime:
                            embedder = Embedder(config, runtime)
                            retrieved = retrieve(
                                client, config, args.query, scope, args.mode, embedder
                            )
                    result = retrieved.model_dump(mode="json")
            print(json.dumps(result, indent=2))
        elif args.command == "eval":
            return _run_evaluation(args, config)
        elif args.command == "ask":
            request = SearchRequest(
                query=args.query,
                mode=args.mode,
                workflow=args.workflow,
                scope=SearchScope(
                    permitted_guides=args.guide or ["s3-userguide", "s3-api"],
                    corpus_id=args.corpus_id,
                ),
            )
            with create_search_client(config) as client:
                answer = answer_question(client, config, request)
            if args.save_run is not None:
                save_answer_run(args.save_run, request, answer, config)
            print(answer.model_dump_json(indent=2))
            return 1 if answer.status == "failed" else 0
        elif args.command == "embed":
            chunks = load_chunks(args.chunks)
            with closing(create_bedrock_client(config)) as runtime:
                embedder = Embedder(config, runtime)
                embedder.embed_chunks(chunks)
            write_jsonl(args.output, embedding_manifest(chunks, config))
            print(
                json.dumps(
                    {
                        "chunks": len(chunks),
                        "requests": embedder.request_count,
                        "cache_hits": embedder.cache_hits,
                        "input_tokens": embedder.input_tokens,
                    },
                    indent=2,
                )
            )
        elif args.command == "index":
            _run_index(args, config)
        else:
            _run_corpus(args, config)
    except TransportError as error:
        print(f"Error: OpenSearch request failed: {error}", file=sys.stderr)
        return 1
    except (BotoCoreError, ClientError) as error:
        print(f"Error: AWS request failed: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, WorkerBusyError, WorkflowBudgetError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
