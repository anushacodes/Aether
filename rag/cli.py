"""Only implemented local commands are exposed. Future search stages import nothing yet."""

import argparse
import sys
from pathlib import Path

from rag import __version__
from rag.config import load_config
from rag.corpus import (
    download_sources,
    extract_sections,
    heading_inventory,
    load_sources,
    write_jsonl,
)
from rag.models import SubsetManifest


def build_parser() -> argparse.ArgumentParser:
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
    ):
        operation = operations.add_parser(name, help=description)
        operation.add_argument("--sources", type=Path, default=Path("data/manifests/sources.json"))
        operation.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
        if name == "inventory":
            operation.add_argument(
                "--output", type=Path, default=Path("data/processed/headings.jsonl")
            )
        if name == "extract":
            operation.add_argument("--subset", type=Path, default=Path("data/manifests/pilot.json"))
            operation.add_argument(
                "--output", type=Path, default=Path("data/processed/pilot-sections.jsonl")
            )
    return parser


def main(argv: list[str] | None = None) -> int:
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
            return 0
        sources = load_sources(args.sources)
        if args.operation == "fetch":
            for path in download_sources(sources, args.raw_dir):
                print(f"Verified {path}")
        elif args.operation == "inventory":
            rows = heading_inventory(sources, args.raw_dir)
            write_jsonl(args.output, rows)
            print(
                f"Wrote {len(rows)} candidate headings to {args.output}; inspect before selecting."
            )
        elif args.operation == "extract":
            subset = SubsetManifest.model_validate_json(args.subset.read_bytes())
            sections = extract_sections(sources, subset, args.raw_dir)
            write_jsonl(args.output, [section.model_dump(mode="json") for section in sections])
            print(f"Wrote {len(sections)} verified sections to {args.output}")
            for section in sections:
                print(
                    f"  {section.section_id}: {len(section.raw_text)} raw characters, "
                    f"{len(section.text)} cleaned characters"
                )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
