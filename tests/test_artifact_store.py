"""Protect source-bound artifact transfers, conditional writes and full read-back integrity."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from botocore.stub import Stubber

from rag.artifact_store import (
    PUBLIC_ACCESS_FLAGS,
    load_artifact_bundle,
    prepare_artifact_bundle,
    upload_artifact_bundle,
)
from rag.cli import main
from rag.config import PipelineConfig
from rag.models import SectionRecord, SourceManifest, SubsetManifest

BUCKET, OWNER, PREFIX = "synthetic-artifact-bucket", "123456789012", "test-artifacts"


def source_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Use an existing passage fixture as a tiny complete synthetic pinned source."""
    section = SectionRecord.model_validate(
        json.loads(Path("tests/fixtures/records.json").read_text(encoding="utf-8"))["section"]
    )
    raw = section.raw_text.encode()
    digest = hashlib.sha256(raw).hexdigest()
    source = section.provenance.model_copy(update={"sha256": digest})
    sources = SourceManifest(revision=source.revision, sources=[source])
    subset = SubsetManifest.model_validate(
        {
            "schema_version": 1,
            "corpus_id": "synthetic-artifact-fixture",
            "source_revision": source.revision,
            "offset_unit": "unicode-codepoints-utf8-no-newline-normalization",
            "sections": [
                {
                    "section_id": section.section_id,
                    "source_id": source.source_id,
                    "heading_path": section.heading_path,
                    "raw_start": 0,
                    "raw_end": len(raw.decode()),
                    "raw_sha256": digest,
                    "selection_reason": "Small synthetic transfer fixture",
                }
            ],
            "exclusions": [],
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / source.original_filename).write_bytes(raw)
    sources_path, subset_path = tmp_path / "sources.json", tmp_path / "subset.json"
    sources_path.write_text(sources.model_dump_json(indent=2) + "\n", encoding="utf-8")
    subset_path.write_text(subset.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return sources_path, subset_path, raw_dir


def rehash_archive(directory: Path) -> None:
    """Forge internally consistent hashes to exercise source binding beyond checksums."""
    path = directory / "bundle.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    for file in value["files"]:
        raw = (directory / file["path"]).read_bytes()
        file.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    identity = {key: value[key] for key in ("schema_version", "corpus_id", "source_revision")}
    identity["files"] = sorted(value["files"], key=lambda file: file["path"])
    value["bundle_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(value), encoding="utf-8")


def test_local_cli_prepare_reproduces_sources_and_refuses_unbound_or_unapproved_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sources, subset, raw_dir = source_inputs(tmp_path)
    sdk = Mock()
    monkeypatch.setattr("rag.cli.create_artifact_client", sdk)
    archive = tmp_path / "bundle"
    assert (
        main(
            [
                "artifacts",
                "prepare",
                "--sources",
                str(sources),
                "--subset",
                str(subset),
                "--raw-dir",
                str(raw_dir),
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    bundle = load_artifact_bundle(archive)
    assert result["bundle_id"] == bundle.manifest.bundle_id
    assert bundle.files["sources.json"] == sources.read_bytes()
    assert bundle.files["subset.json"] == subset.read_bytes()
    assert json.loads(bundle.files["build.json"])["has_vectors"] is False
    with pytest.raises(FileExistsError):
        prepare_artifact_bundle(PipelineConfig(), sources, subset, raw_dir, archive)
    assert (
        main(
            [
                "artifacts",
                "upload",
                "--bundle",
                str(archive),
                "--expected-owner",
                OWNER,
                "--output",
                str(tmp_path / "not-created"),
            ]
        )
        == 1
    )
    assert "--allow-write" in capsys.readouterr().err
    assert not (tmp_path / "not-created").exists() and not sdk.called
    chunks = archive / "chunks.jsonl"
    original = chunks.read_bytes()
    chunks.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="size"):
        load_artifact_bundle(archive)
    # Hashing a changed normalized artifact cannot make it match pinned source extraction.
    records = [json.loads(line) for line in original.splitlines()]
    records[0]["original_text"] = "Invented unsupported source content."
    records[0]["content_sha256"] = hashlib.sha256(records[0]["original_text"].encode()).hexdigest()
    chunks.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    rehash_archive(archive)
    with pytest.raises(ValueError, match="normalized records differ"):
        load_artifact_bundle(archive)
    monkeypatch.setattr(
        "rag.cli.load_config",
        Mock(
            return_value=PipelineConfig(
                cloud={
                    "region": "us-east-1",
                    "artifact_bucket": BUCKET,
                }
            )
        ),
    )
    assert (
        main(
            [
                "artifacts",
                "upload",
                "--allow-write",
                "--bundle",
                str(archive),
                "--expected-owner",
                OWNER,
                "--output",
                str(tmp_path / "invalid-receipt"),
            ]
        )
        == 1
    )
    assert not sdk.called and not (tmp_path / "invalid-receipt").exists()
    chunks.write_bytes(original)
    (archive / "build.json").write_text("[]", encoding="utf-8")
    rehash_archive(archive)
    with pytest.raises(ValueError, match="build definition must be an object"):
        load_artifact_bundle(archive)


def test_native_s3_conditional_upload_verifies_versions_and_existing_objects_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sources, subset, raw_dir = source_inputs(tmp_path)
    archive = tmp_path / "bundle"
    prepare_artifact_bundle(PipelineConfig(), sources, subset, raw_dir, archive)
    bundle = load_artifact_bundle(archive)
    root = f"{PREFIX}/bundles/{bundle.manifest.bundle_id}"
    objects = sorted(bundle.files.items()) + [("bundle.json", bundle.manifest_bytes)]
    streams = []
    with (
        closing(
            boto3.client(
                "s3",
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )
        ) as client,
        Stubber(client) as stub,
    ):
        stub.add_response(
            "get_public_access_block",
            {
                "PublicAccessBlockConfiguration": dict.fromkeys(PUBLIC_ACCESS_FLAGS, True),
            },
            {"Bucket": BUCKET, "ExpectedBucketOwner": OWNER},
        )
        for number, (name, raw) in enumerate(objects):
            parameters = {
                "Bucket": BUCKET,
                "ExpectedBucketOwner": OWNER,
                "Key": f"{root}/{name}",
                "Body": raw,
                "ContentLength": len(raw),
                "ServerSideEncryption": "AES256",
                "IfNoneMatch": "*",
                "ChecksumSHA256": base64.b64encode(hashlib.sha256(raw).digest()).decode(),
            }
            get = {"Bucket": BUCKET, "ExpectedBucketOwner": OWNER, "Key": f"{root}/{name}"}
            if number == 0:
                stub.add_client_error(
                    "put_object",
                    service_error_code="PreconditionFailed",
                    http_status_code=412,
                    expected_params=parameters,
                )
            else:
                stub.add_response("put_object", {"VersionId": f"version-{number}"}, parameters)
                get["VersionId"] = f"version-{number}"
            stream = io.BytesIO(raw)
            streams.append(stream)
            stub.add_response(
                "get_object",
                {
                    "Body": StreamingBody(stream, len(raw)),
                    "ContentLength": len(raw),
                    "ETag": "not-a-content-hash",
                    "VersionId": f"version-{number}",
                },
                get,
            )
        settings = PipelineConfig(cloud={"region": "us-east-1", "artifact_bucket": BUCKET})
        factory = Mock(return_value=client)
        monkeypatch.setattr("rag.cli.load_config", Mock(return_value=settings))
        monkeypatch.setattr("rag.cli.create_artifact_client", factory)
        assert (
            main(
                [
                    "artifacts",
                    "upload",
                    "--allow-write",
                    "--bundle",
                    str(archive),
                    "--expected-owner",
                    OWNER,
                    "--prefix",
                    PREFIX,
                    "--output",
                    str(tmp_path / "receipt"),
                ]
            )
            == 0
        )
        receipt = json.loads(capsys.readouterr().out)
        factory.assert_called_once_with(settings)
        stub.assert_no_pending_responses()
    assert receipt["status"] == "verified"
    assert receipt["objects"][0]["write_status"] == "existing"
    assert receipt["objects"][-1]["key"] == f"{root}/bundle.json"
    assert all(row["download_verified"] for row in receipt["objects"])
    assert all(stream.closed for stream in streams)
    assert json.loads((tmp_path / "receipt/receipt.json").read_text()) == receipt


def test_public_bucket_corruption_and_conflicts_leave_failed_receipts_without_commit_marker(
    tmp_path: Path,
) -> None:
    sources, subset, raw_dir = source_inputs(tmp_path)
    archive = tmp_path / "bundle"
    prepare_artifact_bundle(PipelineConfig(), sources, subset, raw_dir, archive)
    bundle = load_artifact_bundle(archive)
    raw = bundle.files[sorted(bundle.files)[0]]
    for failure in ("public_bucket", "corrupt", "conflict"):
        with (
            closing(
                boto3.client(
                    "s3",
                    region_name="us-east-1",
                    aws_access_key_id="testing",
                    aws_secret_access_key="testing",
                )
            ) as client,
            Stubber(client) as stub,
        ):
            flags = dict.fromkeys(PUBLIC_ACCESS_FLAGS, True)
            if failure == "public_bucket":
                flags["IgnorePublicAcls"] = False
            stub.add_response(
                "get_public_access_block",
                {
                    "PublicAccessBlockConfiguration": flags,
                },
                {"Bucket": BUCKET, "ExpectedBucketOwner": OWNER},
            )
            stream = None
            if failure == "conflict":
                stub.add_client_error(
                    "put_object",
                    service_error_code="ConditionalRequestConflict",
                    http_status_code=409,
                )
            elif failure == "corrupt":
                stub.add_response("put_object", {})
                stream = io.BytesIO(b"x" * len(raw))
                stub.add_response(
                    "get_object",
                    {
                        "Body": StreamingBody(stream, len(raw)),
                        "ContentLength": len(raw),
                        "ETag": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
                    },
                )
            exception = ClientError if failure == "conflict" else ValueError
            with pytest.raises(
                exception,
                match={
                    "public_bucket": "all four",
                    "corrupt": "SHA-256",
                    "conflict": "ConditionalRequestConflict",
                }[failure],
            ):
                upload_artifact_bundle(
                    client,
                    bundle,
                    BUCKET,
                    OWNER,
                    PREFIX,
                    tmp_path / failure,
                    allow_write=True,
                )
            stub.assert_no_pending_responses()
            if stream is not None:
                assert stream.closed
        receipt = json.loads((tmp_path / failure / "receipt.json").read_text())
        assert receipt["status"] == "failed" and receipt["objects"] == []
        assert receipt["error_type"]
