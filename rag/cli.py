"""Adapt CLI input to the validated corpus pipeline and deliberate terminal output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag import __version__
from rag.chunking import chunk_inventory, chunk_section
from rag.config import ChunkingConfig, PipelineConfig, load_config
from rag.corpus import (
    download_sources,
    extract_sections,
    heading_inventory,
    load_sources,
    write_jsonl,
)
from rag.models import ChunkRecord, SectionRecord, SubsetManifest


def build_parser() -> argparse.ArgumentParser:
    """Expose implemented commands without requiring cloud services at startup."""
    parser = argparse.ArgumentParser(
        prog="s3-rag", description="Build an evidence-backed AWS S3 documentation corpus."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="Optional TOML pipeline configuration")
    commands = parser.add_subparsers(dest="command")
    config = commands.add_parser("config", help="Validate configuration without making AWS calls")
    config.add_argument("--cloud", action="store_true", help="Require paid-call configuration")
    config.add_argument(
        "--require-bucket", action="store_true", help="Also require an artifact bucket"
    )
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


def main(argv: list[str] | None = None) -> int:
    """Run the selected command, returning useful errors for expected input/I/O failures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        config = load_config(args.config)
        if args.command == "config":
            if args.cloud or args.require_bucket:
                config.require_cloud_settings(require_bucket=args.require_bucket)
            print("Configuration valid (no network requests made).")
        else:
            _run_corpus(args, config)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
