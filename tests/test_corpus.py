import io
import json
from pathlib import Path

import pytest

from rag.corpus import (
    clean_section,
    download_sources,
    extract_sections,
    heading_inventory,
    sha256,
)
from rag.models import SourceManifest, SubsetManifest


@pytest.fixture
def corpus(tmp_path):
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


def test_extract_preserves_raw_offsets_and_examples(corpus):
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


def test_corrupted_source_fails_before_extraction(corpus):
    sources, subset, raw_dir, raw = corpus
    (raw_dir / "s3-userguide.txt").write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        extract_sections(sources, subset, raw_dir)


@pytest.mark.parametrize("change", ["wrong_offset", "out_of_bounds", "wrong_revision"])
def test_invalid_subset_fails(corpus, change):
    sources, subset, raw_dir, _ = corpus
    values = subset.model_dump()
    if change == "wrong_offset":
        values["sections"][0]["raw_start"] += 1
        message = "range/hash mismatch"
    elif change == "out_of_bounds":
        values["sections"][0]["raw_end"] += 1
        message = "exceeds source length"
    else:
        values["source_revision"] = "b" * 40
        message = "revision"
    with pytest.raises(ValueError, match=message):
        extract_sections(sources, SubsetManifest.model_validate(values), raw_dir)


def test_inventory_offsets_address_original_text(corpus):
    sources, _, raw_dir, raw = corpus
    text = raw.decode()
    candidates = heading_inventory(sources, raw_dir)
    assert any(candidate["candidate_heading"] == "Example heading" for candidate in candidates)
    for candidate in candidates:
        assert text[candidate["raw_start"] : candidate["raw_end"]] == candidate["candidate_heading"]


def test_cleaner_does_not_remove_partial_header_matches():
    text = 'Warning: Amazon Simple Storage Service User Guide\n{"error": "NoSuchKey"}\n'
    assert clean_section(text) == text


def test_downloader_verifies_before_publishing(corpus, monkeypatch, tmp_path):
    sources, _, _, raw = corpus
    output = tmp_path / "download"
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        assert timeout == 30
        return io.BytesIO(raw)

    monkeypatch.setattr("rag.corpus.urlopen", fetch)
    paths = download_sources(sources, output)
    assert paths[0].read_bytes() == raw
    assert f"/{sources.revision}/input_files/s3-userguide.txt" in calls[0]
    download_sources(sources, output)
    assert len(calls) == 1


def test_download_hash_failure_leaves_no_partial_file(corpus, monkeypatch, tmp_path):
    sources, _, _, _ = corpus
    output = tmp_path / "bad-download"
    monkeypatch.setattr("rag.corpus.urlopen", lambda *a, **kw: io.BytesIO(b"wrong source"))
    with pytest.raises(ValueError, match="hash mismatch"):
        download_sources(sources, output)
    assert list(output.iterdir()) == []


def test_question_drafts_reference_selected_passages_only():
    subset = SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes())
    dataset = json.loads(Path("data/eval/pilot.json").read_text())
    sections = {section.section_id for section in subset.sections}
    assert dataset["corpus_id"] == subset.corpus_id
    assert len(dataset["questions"]) == 3
    for question in dataset["questions"]:
        assert set(question["reference_passage_ids"]).issubset(sections)
        assert question["split"] == "development"
