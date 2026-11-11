"""Verify source integrity and extraction without network access or cloud credentials."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from rag.corpus import (
    download_sources,
    extract_sections,
    sha256,
    write_jsonl,
)
from rag.models import SourceManifest, SubsetManifest

CorpusFixture = tuple[SourceManifest, SubsetManifest, Path, bytes]


@pytest.fixture
def corpus(tmp_path: Path) -> CorpusFixture:
    # CRLF and multibyte text expose accidental newline/byte offset conversion.
    raw = (
        "Contents\r\nExample heading\r\n\r\n"
        "Amazon Simple Storage Service User Guide\r\n"
        "Versioning API Version 2006-03-01 12\r\n"
        "Example heading\r\nWarning: Keep café versions.\r\n"
        '<VersioningConfiguration xmlns="example"><Status>Enabled</Status>\r\n'
        "</VersioningConfiguration>\r\n"
    ).encode()
    path = tmp_path / "s3-userguide.txt"
    path.write_bytes(raw)
    sources = SourceManifest.model_validate(
        {
            "revision": "a" * 40,
            "sources": [
                {
                    "source_id": "test-guide",
                    "guide": "s3-userguide",
                    "revision": "a" * 40,
                    "original_filename": path.name,
                    "repository_url": "https://example.com/pinned-source",
                    "publisher_url": "https://example.com/publisher",
                    "sha256": sha256(raw),
                }
            ],
        }
    )
    text = raw.decode()
    start = text.index("Amazon Simple Storage")
    subset = SubsetManifest.model_validate(
        {
            "corpus_id": "test-pilot",
            "source_revision": sources.revision,
            "offset_unit": "unicode-codepoints-utf8-no-newline-normalization",
            "sections": [
                {
                    "section_id": "example",
                    "source_id": "test-guide",
                    "heading_path": ["Example heading"],
                    "raw_start": start,
                    "raw_end": len(text),
                    "raw_sha256": sha256(text[start:].encode()),
                    "selection_reason": "Synthetic integrity test",
                }
            ],
            "exclusions": [],
        }
    )
    return sources, subset, tmp_path, raw


def test_extract_preserves_raw_offsets_and_examples(corpus: CorpusFixture) -> None:
    sources, subset, raw_dir, raw = corpus
    section = extract_sections(sources, subset, raw_dir)[0]
    assert section.raw_text == raw.decode()[section.raw_start : section.raw_end]
    assert "\r\n" in section.raw_text
    assert "café" in section.text
    assert '<VersioningConfiguration xmlns="example"><Status>Enabled</Status>\r\n' in section.text
    assert "Warning: Keep café versions.\r\n" in section.text
    assert "Amazon Simple Storage Service User Guide" not in section.text
    assert "API Version" not in section.text
    assert extract_sections(sources, subset, raw_dir)[0] == section


def test_corrupted_source_fails_before_extraction(corpus: CorpusFixture) -> None:
    sources, subset, raw_dir, raw = corpus
    (raw_dir / "s3-userguide.txt").write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        extract_sections(sources, subset, raw_dir)


def test_incorrect_selection_offsets_fail(corpus: CorpusFixture) -> None:
    sources, subset, raw_dir, _ = corpus
    values = subset.model_dump()
    values["sections"][0]["raw_start"] += 1
    with pytest.raises(ValueError, match="range/hash mismatch"):
        extract_sections(sources, SubsetManifest.model_validate(values), raw_dir)


def test_downloader_verifies_before_publishing(
    corpus: CorpusFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sources, _, _, raw = corpus
    output = tmp_path / "download"
    calls: list[str] = []

    def fetch(url: str, timeout: int) -> io.BytesIO:
        calls.append(url)
        assert timeout == 30
        return io.BytesIO(raw)

    monkeypatch.setattr("rag.corpus.urlopen", fetch)
    paths = download_sources(sources, output)
    assert paths[0].read_bytes() == raw
    assert f"/{sources.revision}/input_files/s3-userguide.txt" in calls[0]
    download_sources(sources, output)
    assert len(calls) == 1


def test_download_hash_failure_leaves_no_partial_file(
    corpus: CorpusFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sources, _, _, _ = corpus
    output = tmp_path / "bad-download"

    def fetch_corrupted_source(url: str, timeout: int) -> io.BytesIO:
        return io.BytesIO(b"wrong source")

    monkeypatch.setattr("rag.corpus.urlopen", fetch_corrupted_source)
    with pytest.raises(ValueError, match="hash mismatch"):
        download_sources(sources, output)
    assert list(output.iterdir()) == []


def test_failed_jsonl_write_preserves_existing_artifact(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    original = '{"previous": "valid artifact"}\n'
    output.write_text(original, encoding="utf-8")
    rows = iter([{"valid": "café"}, {"invalid": object()}])
    with pytest.raises(TypeError, match="not JSON serializable"):
        write_jsonl(output, rows)
    assert output.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [output]
