"""Verify lossless evidence windows, example boundaries, identities, and CLI builds."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Literal

import pytest

from rag.chunking import chunk_section
from rag.cli import main
from rag.config import ChunkingConfig
from rag.corpus import sha256
from rag.models import ChunkRecord, SectionRecord


def _make_section(text: str, *, section_id: str = "test-section") -> SectionRecord:
    return SectionRecord.model_validate(
        {
            "section_id": section_id,
            "source_id": "test-source",
            "corpus_id": "test-corpus",
            "heading_path": ["Synthetic source"],
            "raw_start": 0,
            "raw_end": len(text),
            "raw_sha256": sha256(text.encode()),
            "raw_text": text,
            "text": text,
            "source_scope": "general-purpose",
            "cleaning_version": "synthetic-v1",
            "provenance": {
                "source_id": "test-source",
                "guide": "s3-userguide",
                "revision": "a" * 40,
                "original_filename": "s3-userguide.txt",
                "repository_url": "https://example.com/source",
                "publisher_url": "https://example.com/publisher",
                "sha256": sha256(text.encode()),
            },
        }
    )


def _assert_coverage(text: str, chunks: list[ChunkRecord], size: int) -> None:
    covered_until = 0
    for chunk in chunks:
        assert chunk.text_start is not None
        assert chunk.text_end is not None
        assert chunk.estimated_tokens is not None
        assert chunk.text_start <= covered_until
        assert chunk.text_end > covered_until
        assert chunk.original_text == text[chunk.text_start : chunk.text_end]
        assert sha256(chunk.original_text.encode()) == chunk.content_sha256
        assert len(re.findall(r"\S+", chunk.original_text)) == chunk.estimated_tokens <= size
        covered_until = chunk.text_end
    assert covered_until == len(text)


@pytest.mark.parametrize("strategy", ["fixed", "heading-aware"])
def test_exact_coverage_with_crlf_unicode_and_overlap(
    strategy: Literal["fixed", "heading-aware"],
) -> None:
    text = "  # Heading\r\n\r\nCafé objects remain. Keep every version!\r\n" + "More words. " * 30
    settings = ChunkingConfig(strategy=strategy, size=12, overlap=3)
    chunks = chunk_section(_make_section(text), settings)
    assert len(chunks) > 1
    _assert_coverage(text, chunks, settings.size)
    assert chunk_section(_make_section(text), settings) == chunks


@pytest.mark.parametrize(
    ("strategy", "example"),
    [
        ("fixed", '```python\nprint("hello world")\nprint("keep both lines")\n```\n'),
        ("heading-aware", "~~~shell\naws s3 ls\n~~~\n"),
        ("fixed", '{\n  "object": "sample key",\n  "versionId": "abc"\n}\n'),
        (
            "heading-aware",
            '<VersioningConfiguration xmlns="example">\n<Status>Enabled</Status>\n'
            "</VersioningConfiguration>\n",
        ),
        ("heading-aware", '    print("hello world")\n    print("keep indentation")\n'),
    ],
)
def test_short_examples_stay_complete_in_every_intersection(
    strategy: Literal["fixed", "heading-aware"], example: str
) -> None:
    text = "Leading words before an example.\n\n" + example + "\nThen more context afterwards.\n"
    source = _make_section(text)
    settings = ChunkingConfig(strategy=strategy, size=14, overlap=8)
    chunks = chunk_section(source, settings)
    _assert_coverage(text, chunks, settings.size)
    start = text.index(example)
    end = start + len(example)
    # Ignore whitespace-only overlap at the start/end of an example.
    example_tokens = list(re.finditer(r"\S+", example))
    meaningful_start = start + example_tokens[0].start()
    meaningful_end = start + example_tokens[-1].end()
    intersections = [
        c for c in chunks if c.text_start < meaningful_end and c.text_end > meaningful_start
    ]
    assert intersections
    assert all(c.text_start <= start and c.text_end >= end for c in intersections)
    assert all(not c.continuations for c in chunks)


@pytest.mark.parametrize("strategy", ["fixed", "heading-aware"])
def test_long_example_has_explicit_continuation_metadata(
    strategy: Literal["fixed", "heading-aware"],
) -> None:
    text = "```text\n" + " ".join(f"word-{i}" for i in range(45)) + "\n```\n"
    chunks = chunk_section(
        _make_section(text), ChunkingConfig(strategy=strategy, size=12, overlap=3)
    )
    _assert_coverage(text, chunks, 12)
    parts = [c.continuations[0] for c in chunks]
    assert len({p.block_id for p in parts}) == 1
    assert [p.part_index for p in parts] == list(range(1, len(parts) + 1))
    assert all(p.part_count == len(parts) for p in parts)


def test_heading_aware_chunks_stop_before_a_new_heading() -> None:
    text = "# First\nSome useful text.\n\n# Second\nDifferent useful text.\n"
    chunks = chunk_section(
        _make_section(text), ChunkingConfig(strategy="heading-aware", size=50, overlap=0)
    )
    assert len(chunks) == 2
    assert chunks[0].original_text.startswith("# First")
    assert chunks[1].original_text.startswith("# Second")
    assert not any("# First" in c.original_text and "# Second" in c.original_text for c in chunks)


def test_chunk_identity_tracks_configuration_and_section_lineage() -> None:
    text = "One short paragraph with useful evidence."
    settings = ChunkingConfig(size=20, overlap=2)
    original = chunk_section(_make_section(text), settings)[0]
    different_section = chunk_section(_make_section(text, section_id="another-section"), settings)[
        0
    ]
    different_settings = chunk_section(_make_section(text), ChunkingConfig(size=21, overlap=2))[0]
    assert len({original.chunk_id, different_section.chunk_id, different_settings.chunk_id}) == 3
    assert original.section_raw_start == 0
    assert original.section_raw_end == len(text)


def test_small_windows_always_advance_with_large_overlap() -> None:
    randomizer = random.Random(17)
    for _ in range(75):
        text = "\n\n".join(
            randomizer.choice(["# Heading", '{"x": 1}', "a b c d e. F g h i.", "```\nx y z\n```"])
            for _ in range(8)
        )
        size = randomizer.randint(4, 18)
        for strategy in ("fixed", "heading-aware"):
            settings = ChunkingConfig(strategy=strategy, size=size, overlap=size - 1)
            chunks = chunk_section(_make_section(text), settings)
            _assert_coverage(text, chunks, size)


def test_chunk_cli_writes_valid_records_and_inventory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Use the committed illustrative section, never the downloaded corpus or AWS.
    example = json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_text = example["raw_text"]
    source = dict(example["provenance"], sha256=sha256(raw_text.encode()))
    (raw_dir / source["original_filename"]).write_bytes(raw_text.encode())
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps({"revision": source["revision"], "sources": [source]}), encoding="utf-8"
    )
    subset = tmp_path / "subset.json"
    subset.write_text(
        json.dumps(
            {
                "corpus_id": "cli-test",
                "source_revision": source["revision"],
                "offset_unit": "unicode-codepoints-utf8-no-newline-normalization",
                "sections": [
                    {
                        "section_id": "test",
                        "source_id": source["source_id"],
                        "heading_path": example["heading_path"],
                        "raw_start": 0,
                        "raw_end": len(raw_text),
                        "raw_sha256": sha256(raw_text.encode()),
                        "selection_reason": "Test fixture",
                    }
                ],
                "exclusions": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"
    args = [
        "corpus",
        "chunk",
        "--sources",
        str(sources),
        "--subset",
        str(subset),
        "--raw-dir",
        str(raw_dir),
        "--output",
        str(output),
        "--size",
        "100",
        "--overlap",
        "15",
    ]
    assert main(args) == 0
    first = output.read_bytes()
    assert len([ChunkRecord.model_validate_json(line) for line in first.splitlines()]) > 1
    assert '"oversized_chunks": 0' in capsys.readouterr().out
    assert main(args) == 0
    assert output.read_bytes() == first
    assert main(args + ["--size", "10", "--overlap", "10"]) == 1
    assert output.read_bytes() == first  # Invalid settings cannot clobber the last valid build.
