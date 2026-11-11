"""Adapt CLI input to the validated corpus pipeline and deliberate terminal output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from contextlib import ExitStack, closing
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
from opensearchpy import TransportError
from pydantic import TypeAdapter

from rag import __version__
from rag.artifact_store import (
    load_artifact_bundle,
    prepare_artifact_bundle,
    upload_artifact_bundle,
    validate_artifact_target,
)
from rag.artifacts import build_manifest, save_answer_run, save_build_manifest
from rag.aws_clients import create_artifact_client, create_bedrock_client
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
from rag.enriched_index import build_enriched_index, load_enriched_inputs
from rag.enrichment import run_enrichment
from rag.eval.answers import AnswerBinding, prepare_answers
from rag.eval.comparison import compare_retrieval_runs, render_comparison_markdown
from rag.eval.dataset import load_dataset
from rag.eval.experiments import PRESETS, preflight_experiment, preset_manifest, resolve_experiment
from rag.eval.freeze import freeze_dataset, verify_dataset_freeze
from rag.eval.pooling import pool_candidates
from rag.eval.runs import run_retrieval_evaluation
from rag.indexing import bulk_index_chunks, ensure_index, load_chunks
from rag.ingestion import ingestion_catalog, submit_ingestion
from rag.jobs import IngestionRequest, JobStore
from rag.models import ChunkRecord, SearchRequest, SearchScope, SectionRecord, SubsetManifest
from rag.pipeline import answer_question, retrieve
from rag.provider_check import CheckPrices, run_provider_check
from rag.publication import active_index, build_snapshot, publish_snapshot
from rag.search_store import check_search_store, create_search_client
from rag.snapshots import retained_snapshots, rollback_snapshot
from rag.worker import WorkerBusyError, run_one
from rag.workflow_limits import WorkflowBudgetError


def _add_question_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query")
    parser.add_argument("--guide", action="append", choices=("s3-userguide", "s3-api"))
    parser.add_argument("--corpus-id")
    parser.add_argument("--mode", choices=("bm25", "vector", "hybrid"), default="bm25")


def _cost_amount(value: str) -> Decimal:
    """Turn invalid currency input into an ordinary parser error before any side effects."""
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("Cost must be a positive finite USD amount") from error
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError("Cost must be a positive finite USD amount")
    return amount


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
    artifacts = commands.add_parser(
        "artifacts", help="Prepare and explicitly transfer corpus bundles"
    )
    artifact_operations = artifacts.add_subparsers(dest="operation", required=True)
    prepare = artifact_operations.add_parser(
        "prepare", help="Create a verified local text-only bundle"
    )
    prepare.add_argument("--sources", type=Path, default=Path("data/manifests/sources.json"))
    prepare.add_argument("--subset", type=Path, required=True)
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output", type=Path, required=True)
    upload = artifact_operations.add_parser(
        "upload", help="Upload to an existing private S3 bucket"
    )
    upload.add_argument("--bundle", type=Path, required=True)
    upload.add_argument("--expected-owner", required=True)
    upload.add_argument("--prefix", default="s3-rag")
    upload.add_argument("--allow-write", action="store_true")
    upload.add_argument("--output", type=Path, required=True, help="New local receipt directory")
    provider = commands.add_parser(
        "bedrock-check", help="Explicitly run a bounded, priced real-provider capability check"
    )
    provider.add_argument(
        "--allow-paid", action="store_true", help="Allow three inference attempts"
    )
    provider.add_argument(
        "--prices", type=Path, required=True, help="Current exact-model price JSON"
    )
    provider.add_argument("--max-cost-usd", type=_cost_amount, required=True)
    provider.add_argument("--chunks", type=Path, default=Path("data/processed/pilot-chunks.jsonl"))
    provider.add_argument(
        "--chunk-id", required=True, help="One original chunk of at most 4000 characters"
    )
    provider.add_argument(
        "--output", type=Path, required=True, help="New local report/cache directory"
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
    eval_operations.add_parser(
        "experiments", help="List the versioned B0–B5 treatments without calling services"
    )
    freeze = eval_operations.add_parser(
        "freeze", help="Archive reviewed split labels and verified pinned source provenance offline"
    )
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--review-record", type=Path, required=True)
    freeze.add_argument("--sources", type=Path, default=Path("data/manifests/sources.json"))
    freeze.add_argument("--subset", type=Path, required=True)
    freeze.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    freeze.add_argument("--output", type=Path, required=True)
    verify = eval_operations.add_parser(
        "verify-freeze", help="Verify retained frozen label bytes and identities without services"
    )
    verify.add_argument("--archive", type=Path, required=True)
    validate = eval_operations.add_parser("validate", help="Check labels and prevent split leakage")
    validate.add_argument("--dataset", type=Path, default=Path("data/eval/pilot.json"))
    validate.add_argument(
        "--allow-drafts", action="store_true", help="Inspect drafts without certifying labels"
    )
    pool = eval_operations.add_parser("pool", help="Pool saved candidates for evidence review")
    pool.add_argument("--dataset", type=Path, default=Path("data/eval/pilot.json"))
    pool.add_argument("--run", type=Path, action="append", required=True)
    pool.add_argument("--split", choices=("development", "held_out"), default="development")
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--allow-drafts", action="store_true")
    compare = eval_operations.add_parser(
        "compare", help="Compare aligned saved retrieval runs without calling services"
    )
    compare.add_argument("--dataset", type=Path, default=Path("data/eval/pilot.json"))
    compare.add_argument("--run", type=Path, action="append", required=True)
    compare.add_argument("--split", choices=("development", "held_out"), default="development")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--allow-drafts", action="store_true")
    for name, help_text in (
        ("prepare-answers", "Validate saved answer samples without calling a judge"),
        ("judge", "Run optional Ragas metrics on reviewed saved answers"),
    ):
        answer_eval = eval_operations.add_parser(name, help=help_text)
        answer_eval.add_argument("--dataset", type=Path, default=Path("data/eval/pilot.json"))
        answer_eval.add_argument("--answers", type=Path, required=True)
        answer_eval.add_argument(
            "--sections", type=Path, default=Path("data/processed/pilot-sections.jsonl")
        )
        answer_eval.add_argument("--output", type=Path, required=True)
        if name == "prepare-answers":
            answer_eval.add_argument("--allow-drafts", action="store_true")
        else:
            answer_eval.set_defaults(allow_drafts=False)
    evaluate = eval_operations.add_parser(
        "retrieval", help="Capture rankings and reviewed-label metrics"
    )
    evaluate.add_argument("--dataset", type=Path, default=Path("data/eval/pilot.json"))
    evaluate.add_argument(
        "--sections", type=Path, default=Path("data/processed/pilot-sections.jsonl")
    )
    evaluate.add_argument("--chunks", type=Path, default=Path("data/processed/pilot-chunks.jsonl"))
    evaluate.add_argument("--mode", choices=("bm25", "vector", "hybrid"))
    evaluate.add_argument("--experiment", choices=[preset.identifier for preset in PRESETS])
    evaluate.add_argument(
        "--enriched-index", type=Path, help="Successful index.json binding for B3/B5 only"
    )
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
    enrich = commands.add_parser("enrich", help="Prepare unreviewed metadata for explicit chunks")
    enrich.add_argument("--chunks", type=Path, default=Path("data/processed/pilot-chunks.jsonl"))
    enrich.add_argument(
        "--chunk-id", action="append", help="Select an exact ID; repeat to select more"
    )
    enrich.add_argument("--output", type=Path, required=True, help="New experiment directory")
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
    index_operations.add_parser("snapshots", help="List retained publication history")
    enriched = index_operations.add_parser("enriched", help="Build an isolated metadata experiment")
    enriched.add_argument("--run", type=Path, required=True, help="Complete metadata run directory")
    enriched.add_argument(
        "--output", type=Path, required=True, help="New experiment record directory"
    )
    enriched.add_argument("--with-embeddings", action="store_true")
    rollback = index_operations.add_parser(
        "rollback", help="Restore a retained published snapshot without rebuilding"
    )
    rollback.add_argument("snapshot_id")
    rollback.add_argument("--base-revision", required=True, help="Expected active concrete index")
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
    if (
        args.operation not in {"publish", "snapshots", "rollback", "enriched"}
        and config.index.name == config.publication.alias
    ):
        raise ValueError(
            "Use 'index publish' for the active alias; mutable builds need a separate index"
        )
    if args.operation == "enriched":
        # Validate files/intent before even discovering cloud credentials for an optional embedder.
        load_enriched_inputs(args.run)
        if args.output.exists():
            raise FileExistsError(f"Experiment output already exists: {args.output}")
    with ExitStack() as stack:
        client = stack.enter_context(create_search_client(config))
        if args.operation == "snapshots":
            result = {
                "alias": config.publication.alias,
                "snapshots": [
                    row.model_dump(mode="json") for row in retained_snapshots(client, config)
                ],
            }
        elif args.operation == "rollback":
            restored = rollback_snapshot(client, config, args.snapshot_id, args.base_revision)
            result = restored.model_dump(mode="json") | {"alias": config.publication.alias}
        elif args.operation == "create":
            result = {"index": config.index.name, "created": ensure_index(client, config)}
        else:
            embedder = None
            if args.with_embeddings:
                runtime = stack.enter_context(closing(create_bedrock_client(config)))
                embedder = Embedder(config, runtime)
            if args.operation == "enriched":
                result = build_enriched_index(
                    client, config, args.run, args.output, embedder=embedder
                )
            elif args.operation == "publish":
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
    if args.operation == "experiments":
        print(json.dumps(preset_manifest(), indent=2))
        return 0
    if args.operation in {"freeze", "verify-freeze"}:
        if args.operation == "freeze":
            manifest = freeze_dataset(
                args.dataset,
                args.review_record,
                args.sources,
                args.subset,
                args.raw_dir,
                args.output,
            )
        else:
            manifest = verify_dataset_freeze(args.archive)
        print(manifest.model_dump_json(indent=2))
        return 0
    dataset = load_dataset(args.dataset, allow_drafts=args.allow_drafts)
    if args.operation == "retrieval" and args.split == "held_out":
        dataset.require_reviewed()
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
    if args.operation == "pool":
        pool = pool_candidates(dataset, args.run, args.split)
        serialized = json.dumps(pool, indent=2, ensure_ascii=False, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
        print(json.dumps({"status": pool["status"], "output": str(args.output)}, indent=2))
        return 0
    if args.operation == "compare":
        report = compare_retrieval_runs(
            dataset, args.run, args.split, allow_drafts=args.allow_drafts
        )
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        markdown = render_comparison_markdown(report)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "comparison.json").write_text(serialized + "\n", encoding="utf-8")
        (args.output / "comparison.md").write_text(markdown, encoding="utf-8")
        print(json.dumps({"quality_status": report["quality_status"], "output": str(args.output)}))
        return 0
    sections = [
        SectionRecord.model_validate_json(line)
        for line in args.sections.read_bytes().splitlines()
        if line.strip()
    ]
    if args.operation in {"prepare-answers", "judge"}:
        bindings = TypeAdapter(list[AnswerBinding]).validate_json(args.answers.read_bytes())
        prepared = prepare_answers(
            dataset, bindings, sections, max_samples=config.evaluation.max_samples
        )
        if args.operation == "prepare-answers":
            args.output.mkdir(parents=True, exist_ok=False)
            write_jsonl(
                args.output / "prepared.jsonl", [row.model_dump(mode="json") for row in prepared]
            )
            summary = {
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.fingerprint(),
                "review_status": dataset.review_status,
                "questions": len(prepared),
                "outcomes": dict(Counter(row.status for row in prepared)),
                "model_calls": 0,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            failed = any(row.status == "error" for row in prepared)
        else:
            # Optional evaluation dependencies are not imported by application startup.
            from rag.eval.ragas_runner import run_ragas_evaluation

            summary = run_ragas_evaluation(config, dataset, prepared, args.output)
            failed = any(
                summary["metric_outcomes"].get(status, 0) for status in ("error", "undefined")
            )
        print(json.dumps(summary | {"output": str(args.output)}, indent=2))
        return 1 if failed else 0
    chunks = load_chunks(args.chunks)
    output = args.output or Path("reports/runs") / f"retrieval-{uuid4().hex}"
    selection = None
    if args.experiment is not None:
        if args.mode is not None:
            raise ValueError("Use --experiment or --mode, not both")
        selection = resolve_experiment(args.experiment, config, chunks, args.enriched_index)
        config = selection.config
        mode = selection.preset.mode
    else:
        if args.enriched_index is not None:
            raise ValueError("--enriched-index requires a B3/B5 experiment")
        mode = args.mode or "bm25"
    with ExitStack() as stack:
        client = stack.enter_context(create_search_client(config))
        if selection is not None:
            selection = preflight_experiment(client, selection, chunks)
            config = selection.config
        embedder = None
        if mode != "bm25":
            runtime = stack.enter_context(closing(create_bedrock_client(config)))
            embedder = Embedder(config, runtime)
        summary = run_retrieval_evaluation(
            client,
            config,
            dataset,
            sections,
            chunks,
            mode,
            args.split,
            output,
            embedder=embedder,
            allow_drafts=args.allow_drafts,
            experiment=selection.provenance if selection is not None else None,
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
        elif args.command == "artifacts":
            if args.operation == "prepare":
                manifest = prepare_artifact_bundle(
                    config, args.sources, args.subset, args.raw_dir, args.output
                )
                print(manifest.model_dump_json(indent=2))
            else:
                if not args.allow_write:
                    raise ValueError("S3 artifact upload requires explicit --allow-write")
                bucket = config.cloud.artifact_bucket
                validate_artifact_target(bucket, args.expected_owner, args.prefix)
                bundle = load_artifact_bundle(args.bundle)
                if args.output.exists() or args.output.is_symlink():
                    raise FileExistsError(f"Artifact receipt output already exists: {args.output}")
                with closing(create_artifact_client(config)) as client:
                    receipt = upload_artifact_bundle(
                        client,
                        bundle,
                        bucket,
                        args.expected_owner,
                        args.prefix,
                        args.output,
                        allow_write=True,
                    )
                print(json.dumps(receipt, indent=2))
        elif args.command == "bedrock-check":
            if not args.allow_paid:
                raise ValueError("Provider check requires explicit --allow-paid")
            prices = CheckPrices.model_validate_json(args.prices.read_text(encoding="utf-8"))
            chunks = [
                chunk for chunk in load_chunks(args.chunks) if chunk.chunk_id == args.chunk_id
            ]
            if len(chunks) != 1:
                raise ValueError("Select exactly one chunk ID present in the input")
            report = run_provider_check(
                config, chunks[0], prices, args.max_cost_usd, args.output, allow_paid=True
            )
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "passed" else 1
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
        elif args.command == "enrich":
            chunks = load_chunks(args.chunks)
            if args.chunk_id is not None:
                selected = set(args.chunk_id)
                if len(selected) != len(args.chunk_id) or selected - {c.chunk_id for c in chunks}:
                    raise ValueError("Select distinct chunk IDs present in the input")
                chunks = [chunk for chunk in chunks if chunk.chunk_id in selected]
            summary = run_enrichment(config, chunks, args.output)
            print(json.dumps(summary | {"output": str(args.output)}, indent=2))
            return 1 if summary["outcomes"].get("content_only", 0) else 0
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
