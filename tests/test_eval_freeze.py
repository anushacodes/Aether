"""Preserve reviewed split provenance and reject changed or prematurely frozen labels."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.cli import main
from rag.corpus import extract_sections
from rag.eval.dataset import EvaluationDataset, EvaluationPassage
from rag.eval.freeze import DatasetReview, freeze_dataset, verify_dataset_freeze
from rag.models import SourceManifest, SubsetManifest


def reviewed_inputs(tmp_path: Path) -> dict[str, Path]:
    """Create explicitly synthetic reviewed labels over a tiny hash-verified raw source."""
    raw = "S3 Versioning\nReplacing a key retains the earlier version as noncurrent.\n"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    revision = "a" * 40
    sources = SourceManifest.model_validate(
        {
            "revision": revision,
            "sources": [
                {
                    "source_id": "fixture-source",
                    "guide": "s3-userguide",
                    "revision": revision,
                    "original_filename": "s3-userguide.txt",
                    "repository_url": "https://example.com/pinned/s3-userguide.txt",
                    "publisher_url": "https://example.com/publisher",
                    "sha256": digest,
                }
            ],
        }
    )
    subset = SubsetManifest.model_validate(
        {
            "schema_version": 1,
            "corpus_id": "synthetic-freeze-fixture",
            "source_revision": revision,
            "offset_unit": "unicode-codepoints-utf8-no-newline-normalization",
            "sections": [
                {
                    "section_id": "versioning",
                    "source_id": "fixture-source",
                    "heading_path": ["Versioning"],
                    "raw_start": 0,
                    "raw_end": len(raw),
                    "raw_sha256": digest,
                    "selection_reason": "Synthetic test input; no real benchmark claim",
                }
            ],
            "exclusions": [],
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "s3-userguide.txt").write_text(raw, encoding="utf-8")
    section = extract_sections(sources, subset, raw_dir)[0]
    passage = EvaluationPassage(
        passage_id="retained-version",
        source_id=section.source_id,
        section_id=section.section_id,
        raw_section_sha256=section.raw_sha256,
        cleaning_version=section.cleaning_version,
        text_start=0,
        text_end=len(section.text),
        text_sha256=hashlib.sha256(section.text.encode()).hexdigest(),
    )
    dataset = EvaluationDataset.model_validate(
        {
            "dataset_id": "synthetic-reviewed-fixture-v1",
            "corpus_id": subset.corpus_id,
            "review_status": "reviewed",
            "note": "Synthetic fixture only; never promote actual project draft labels",
            "passages": [passage.model_dump()],
            "questions": [
                {
                    "question_id": "retention",
                    "query": "What happens to the previous version when a key is replaced?",
                    "category": "direct_fact",
                    "permitted_guides": ["s3-userguide"],
                    "source_scope": "general-purpose",
                    "reference_answer": "It is retained as noncurrent.",
                    "reference_passage_ids": [passage.passage_id],
                    "expected_answerability": "answerable",
                    "review_status": "reviewed",
                    "split": "development",
                    "group_id": "retention-family",
                },
                {
                    "question_id": "live-state",
                    "query": "Is my live bucket public?",
                    "category": "unanswerable",
                    "permitted_guides": ["s3-userguide"],
                    "source_scope": "general-purpose",
                    "reference_answer": "Cannot determine live state from documentation",
                    "reference_passage_ids": [],
                    "expected_answerability": "unanswerable",
                    "unanswerable_reason": "No live account state in the pinned source",
                    "review_status": "reviewed",
                    "split": "held_out",
                    "group_id": "live-state-family",
                },
            ],
        }
    )
    review = DatasetReview(
        dataset_sha256=dataset.fingerprint(),
        reviewer="synthetic fixture reviewer",
        reviewed_at=datetime.now(UTC),
        semantic_groups_reviewed=True,
        note="Test provenance only; not independent review of the actual core bank",
    )
    paths = {name: tmp_path / f"{name}.json" for name in ("dataset", "review", "sources", "subset")}
    for name, record in (
        ("dataset", dataset),
        ("review", review),
        ("sources", sources),
        ("subset", subset),
    ):
        paths[name].write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return paths | {"raw_dir": raw_dir, "output": tmp_path / "archive"}


def freeze_inputs(paths: dict[str, Path]) -> None:
    """Call the public freeze boundary with a fixture's files."""
    freeze_dataset(
        paths["dataset"],
        paths["review"],
        paths["sources"],
        paths["subset"],
        paths["raw_dir"],
        paths["output"],
    )


def test_freeze_cli_preserves_exact_inputs_and_verifies_offline_after_originals_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = reviewed_inputs(tmp_path)
    search, runtime = Mock(), Mock()
    monkeypatch.setattr("rag.cli.create_search_client", search)
    monkeypatch.setattr("rag.cli.create_bedrock_client", runtime)
    assert (
        main(
            [
                "eval",
                "freeze",
                "--dataset",
                str(paths["dataset"]),
                "--review-record",
                str(paths["review"]),
                "--sources",
                str(paths["sources"]),
                "--subset",
                str(paths["subset"]),
                "--raw-dir",
                str(paths["raw_dir"]),
                "--output",
                str(paths["output"]),
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["question_ids"] == {"development": ["retention"], "held_out": ["live-state"]}
    for name in ("dataset", "review", "sources", "subset"):
        assert (paths["output"] / f"{name}.json").read_bytes() == paths[name].read_bytes()
        paths[name].write_text("original input changed after archival", encoding="utf-8")
    (paths["raw_dir"] / "s3-userguide.txt").unlink()
    assert main(["eval", "verify-freeze", "--archive", str(paths["output"])]) == 0
    assert json.loads(capsys.readouterr().out) == manifest
    assert not search.called and not runtime.called
    saved = (paths["output"] / "manifest.json").read_bytes()
    # A repeated freeze cannot overwrite a valid archive.
    with pytest.raises(FileExistsError):
        freeze_dataset(
            paths["output"] / "dataset.json",
            paths["output"] / "review.json",
            paths["output"] / "sources.json",
            paths["output"] / "subset.json",
            tmp_path / "raw-restored",
            paths["output"],
        )
    assert (paths["output"] / "manifest.json").read_bytes() == saved


def test_freeze_refuses_unreviewed_unassigned_unscoped_and_unbound_inputs_before_writes(
    tmp_path: Path,
) -> None:
    paths = reviewed_inputs(tmp_path)
    original = json.loads(paths["dataset"].read_text(encoding="utf-8"))
    cases = (
        ("draft", "human review"),
        ("candidate", "split assignment"),
        ("group", "group ID"),
        ("one_split", "nonempty"),
        ("scope", "outside question scope"),
        ("lineage", "lineage/hash mismatch"),
    )
    for case, message in cases:
        value = json.loads(json.dumps(original))
        if case == "draft":
            value["questions"][0]["review_status"] = "draft"
        elif case == "candidate":
            value["questions"][1]["split"] = "candidate"
        elif case == "group":
            value["questions"][0]["group_id"] = None
        elif case == "one_split":
            value["questions"][1]["split"] = "development"
        elif case == "scope":
            value["questions"][0]["permitted_guides"] = ["s3-api"]
        else:
            value["passages"][0]["text_sha256"] = "0" * 64
        dataset = EvaluationDataset.model_validate(value)
        paths["dataset"].write_text(dataset.model_dump_json(), encoding="utf-8")
        review = json.loads(paths["review"].read_text(encoding="utf-8"))
        review["dataset_sha256"] = dataset.fingerprint()
        paths["review"].write_text(json.dumps(review), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            freeze_inputs(paths)
        assert not paths["output"].exists()
    paths["dataset"].write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="exact dataset fingerprint"):
        freeze_inputs(paths)
    assert not paths["output"].exists()


def test_verification_rejects_changed_files_inventory_and_missing_commit_manifest(
    tmp_path: Path,
) -> None:
    paths = reviewed_inputs(tmp_path)
    freeze_inputs(paths)
    archive = paths["output"]
    data = (archive / "dataset.json").read_bytes()
    (archive / "dataset.json").write_bytes(data + b"\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify_dataset_freeze(archive)
    (archive / "dataset.json").write_bytes(data)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    manifest["question_ids"]["held_out"] = []
    (archive / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory does not match"):
        verify_dataset_freeze(archive)
    manifest["file_sha256"]["../dataset.json"] = "0" * 64
    (archive / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly its five"):
        verify_dataset_freeze(archive)
    (archive / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_dataset_freeze(archive)
