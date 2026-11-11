"""Verify pinned source bytes and publish reproducible local corpus artifacts.

Raw offsets address unmodified UTF-8 text. Network and filesystem operations
stay here so chunking can remain independent of infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypedDict
from urllib.request import urlopen

from rag.models import SectionRecord, SourceManifest, SubsetManifest

SOURCE_REPOSITORY = (
    "https://raw.githubusercontent.com/PranavMishra17/"
    "Metadata-Enrichment-with-LLMs-for-RAGs-Internal-Knowledge-Retrieval"
)
MAX_SOURCE_BYTES = 32 * 1024 * 1024
DOWNLOAD_BLOCK_BYTES = 64 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30
MIN_HEADING_LENGTH = 3
MAX_HEADING_LENGTH = 100
CLEANING_VERSION = "pdf-headers-v1"
HEADERS = {
    "Amazon Simple Storage Service User Guide",
    "Amazon Simple Storage Service API Reference",
}
FOOTER = re.compile(r"^.{1,120} API Version 2006-03-01 [0-9]+\s*$")


class HeadingCandidate(TypedDict):
    """A heuristic heading with offsets into the unmodified source text."""

    source_id: str
    raw_start: int
    raw_end: int
    candidate_heading: str


def sha256(data: bytes) -> str:
    """Return the SHA-256 digest used to verify pinned source artifacts."""
    return hashlib.sha256(data).hexdigest()


def load_sources(path: Path) -> SourceManifest:
    """Validate a source manifest before accessing its files or URLs."""
    return SourceManifest.model_validate_json(path.read_bytes())


def download_sources(manifest: SourceManifest, raw_dir: Path) -> list[Path]:
    """Fetch only pinned, allowlisted filenames; verify before publishing each file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for source in manifest.sources:
        target = raw_dir / source.original_filename
        if target.exists():
            if sha256(target.read_bytes()) != source.sha256:
                raise ValueError(f"Hash mismatch for {target}; remove it before fetching again")
            paths.append(target)
            continue
        url = f"{SOURCE_REPOSITORY}/{source.revision}/input_files/{source.original_filename}"
        temporary_path: Path | None = None
        try:
            with (
                urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
                tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as output,
            ):
                temporary_path = Path(output.name)
                size = 0
                while block := response.read(DOWNLOAD_BLOCK_BYTES):
                    size += len(block)
                    if size > MAX_SOURCE_BYTES:
                        raise ValueError(f"Source exceeds {MAX_SOURCE_BYTES} bytes: {url}")
                    output.write(block)
            data = temporary_path.read_bytes()
            if sha256(data) != source.sha256:
                raise ValueError(f"Downloaded source hash mismatch: {source.original_filename}")
            data.decode("utf-8")  # Fail before publication if encoding is incompatible.
            temporary_path.replace(target)
            paths.append(target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return paths


def read_verified_sources(manifest: SourceManifest, raw_dir: Path) -> dict[str, str]:
    """Decode only hash-matched snapshots, preserving original newline sequences."""
    texts = {}
    for source in manifest.sources:
        path = raw_dir / source.original_filename
        data = path.read_bytes()
        if sha256(data) != source.sha256:
            raise ValueError(f"Hash mismatch for {path}; raw source must match the pinned manifest")
        # Do not use read_text(): universal newline conversion would alter raw offsets.
        texts[source.source_id] = data.decode("utf-8")
    return texts


def heading_inventory(manifest: SourceManifest, raw_dir: Path) -> list[HeadingCandidate]:
    """Heuristic candidates, not a parsed outline; selections require source inspection."""
    candidates: list[HeadingCandidate] = []
    for source_id, text in read_verified_sources(manifest, raw_dir).items():
        offset = 0
        for line in text.splitlines(keepends=True):
            label = line.rstrip("\r\n")
            if (
                MIN_HEADING_LENGTH <= len(label) <= MAX_HEADING_LENGTH
                and label[0].isupper()
                and label == label.strip()
                and not label.endswith((".", ":", ",", ";"))
                and "..." not in label
                and label not in HEADERS
                and not FOOTER.fullmatch(label)
            ):
                candidates.append(
                    {
                        "source_id": source_id,
                        "raw_start": offset,
                        "raw_end": offset + len(label),
                        "candidate_heading": label,
                    }
                )
            offset += len(line)
    return candidates


def clean_section(raw_text: str) -> str:
    """Remove only known whole-line PDF headers/footers; preserve other text verbatim."""
    return "".join(
        line
        for line in raw_text.splitlines(keepends=True)
        if line.rstrip("\r\n") not in HEADERS and not FOOTER.fullmatch(line.rstrip("\r\n"))
    )


def extract_sections(
    sources: SourceManifest, subset: SubsetManifest, raw_dir: Path
) -> list[SectionRecord]:
    """Validate all selected raw ranges before returning cleaned records with lineage."""
    if subset.source_revision != sources.revision:
        raise ValueError("Subset revision does not match source manifest")
    texts = read_verified_sources(sources, raw_dir)
    provenance = {source.source_id: source for source in sources.sources}
    records = []
    for selection in subset.sections:
        if selection.source_id not in texts:
            raise ValueError(f"Unknown source_id: {selection.source_id}")
        source_text = texts[selection.source_id]
        if selection.raw_end > len(source_text):
            raise ValueError(f"Raw range exceeds source length: {selection.section_id}")
        raw_text = source_text[selection.raw_start : selection.raw_end]
        if sha256(raw_text.encode("utf-8")) != selection.raw_sha256:
            raise ValueError(f"Section range/hash mismatch: {selection.section_id}")
        records.append(
            SectionRecord(
                **selection.model_dump(exclude={"selection_reason"}),
                corpus_id=subset.corpus_id,
                raw_text=raw_text,
                text=clean_section(raw_text),
                cleaning_version=CLEANING_VERSION,
                provenance=provenance[selection.source_id],
            )
        )
    return records


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Publish complete output atomically so a failed build cannot leave partial JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", newline="\n", delete=False
        ) as output:
            temporary_path = Path(output.name)
            for row in rows:
                output.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
