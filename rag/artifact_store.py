"""Prepare source-bound local bundles and explicitly publish verified S3 copies.

Bundles contain only pinned sources and reproducible text-only build inputs,
never arbitrary directories, credentials, evaluation labels or model caches.
Conditional object writes are followed by full download/hash verification;
the bundle manifest is uploaded last as a completion marker.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from botocore.exceptions import ClientError
from pydantic import Field, model_validator

from rag.artifacts import build_manifest
from rag.aws_clients import ArtifactS3
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.corpus import extract_sections, write_jsonl
from rag.models import Digest, Identifier, Record, Revision, SourceManifest, SubsetManifest

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
READ_BLOCK_BYTES = 64 * 1024
FIXED_FILES = frozenset(
    {"sources.json", "subset.json", "sections.jsonl", "chunks.jsonl", "build.json"}
)
PUBLIC_ACCESS_FLAGS = (
    "BlockPublicAcls",
    "IgnorePublicAcls",
    "BlockPublicPolicy",
    "RestrictPublicBuckets",
)


class BundleFile(Record):
    """Bind one approved relative filename to its exact bytes."""

    path: str
    size_bytes: int = Field(ge=0, le=MAX_FILE_BYTES)
    sha256: Digest


def _identity(corpus_id: str, revision: str, files: list[BundleFile]) -> str:
    """Content-address the complete inventory, independent of local paths and S3 location."""
    value = {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "source_revision": revision,
        "files": [file.model_dump() for file in sorted(files, key=lambda item: item.path)],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ArtifactBundle(Record):
    """Describe a bounded corpus bundle without claiming publication or model quality."""

    schema_version: int = Field(default=1, ge=1, le=1)
    bundle_id: Digest
    corpus_id: Identifier
    source_revision: Revision
    created_at: datetime
    files: list[BundleFile] = Field(min_length=6, max_length=7)

    @model_validator(mode="after")
    def consistent_inventory(self) -> Self:
        """Reject ambiguous inventory, oversized inputs and a false content identity."""
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("Artifact filenames must be unique")
        if sum(file.size_bytes for file in self.files) > MAX_BUNDLE_BYTES:
            raise ValueError("Artifact bundle exceeds the 64 MiB limit")
        if self.created_at.utcoffset() is None:
            raise ValueError("Artifact creation time requires a timezone")
        if self.bundle_id != _identity(self.corpus_id, self.source_revision, self.files):
            raise ValueError("Artifact bundle identity does not match its file inventory")
        return self


@dataclass(frozen=True)
class PreparedBundle:
    """Retain validated immutable byte snapshots through an upload, avoiding path rereads."""

    manifest: ArtifactBundle
    manifest_bytes: bytes
    files: dict[str, bytes]


def _read_bounded(path: Path, maximum: int = MAX_FILE_BYTES) -> bytes:
    """Enforce the size limit while reading even if a file changes after stat."""
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f"Artifact file exceeds its size limit: {path.name}")
    return raw


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def _build_files(
    config: PipelineConfig, sources: SourceManifest, subset: SubsetManifest, raw_dir: Path
) -> dict[str, bytes]:
    """Reproduce normalized records and a source/model-bound text-only build definition."""
    sections = extract_sections(sources, subset, raw_dir)
    chunks = [chunk for section in sections for chunk in chunk_section(section, config.chunking)]
    return {
        "sections.jsonl": "".join(
            section.model_dump_json() + "\n" for section in sections
        ).encode(),
        "chunks.jsonl": "".join(chunk.model_dump_json() + "\n" for chunk in chunks).encode(),
        "build.json": _json_bytes(
            build_manifest(config, chunks, has_vectors=False, corpus_id=subset.corpus_id)
        ),
    }


def prepare_artifact_bundle(
    config: PipelineConfig, sources_path: Path, subset_path: Path, raw_dir: Path, output: Path
) -> ArtifactBundle:
    """Prepare only approved source/build inputs locally, without constructing an SDK client."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Artifact output already exists: {output}")
    sources_bytes, subset_bytes = _read_bounded(sources_path), _read_bounded(subset_path)
    sources = SourceManifest.model_validate_json(sources_bytes)
    subset = SubsetManifest.model_validate_json(subset_bytes)
    files = {"sources.json": sources_bytes, "subset.json": subset_bytes}
    for source in sources.sources:
        raw = _read_bounded(raw_dir / source.original_filename)
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise ValueError("Artifact raw source differs from its pinned manifest")
        files[f"raw/{source.original_filename}"] = raw
    files.update(_build_files(config, sources, subset, raw_dir))
    inventory = [
        BundleFile(path=name, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        for name, raw in sorted(files.items())
    ]
    manifest = ArtifactBundle(
        bundle_id=_identity(subset.corpus_id, sources.revision, inventory),
        corpus_id=subset.corpus_id,
        source_revision=sources.revision,
        created_at=datetime.now(UTC),
        files=inventory,
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    with (output / "bundle.json").open("x", encoding="utf-8") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_artifact_bundle(directory: Path) -> PreparedBundle:
    """Verify fixed filenames, byte hashes and reproducible source/build binding before AWS."""
    manifest_bytes = _read_bounded(directory / "bundle.json", maximum=64 * 1024)
    manifest = ArtifactBundle.model_validate_json(manifest_bytes)
    names = {file.path for file in manifest.files}
    allowed_raw = {"raw/s3-userguide.txt", "raw/s3-api.txt"}
    if not FIXED_FILES < names or not names <= FIXED_FILES | allowed_raw:
        raise ValueError("Artifact bundle contains unexpected or missing filenames")
    files = {}
    for file in manifest.files:
        raw = _read_bounded(directory / file.path, maximum=file.size_bytes)
        if len(raw) != file.size_bytes or hashlib.sha256(raw).hexdigest() != file.sha256:
            raise ValueError("Artifact bundle file size/hash mismatch")
        files[file.path] = raw
    sources = SourceManifest.model_validate_json(files["sources.json"])
    subset = SubsetManifest.model_validate_json(files["subset.json"])
    if names != FIXED_FILES | {f"raw/{source.original_filename}" for source in sources.sources}:
        raise ValueError("Artifact raw inventory does not match the source manifest")
    if subset.corpus_id != manifest.corpus_id or sources.revision != manifest.source_revision:
        raise ValueError("Artifact corpus/source identity mismatch")
    build = json.loads(files["build.json"])
    if not isinstance(build, dict):
        raise ValueError("Artifact build definition must be an object")
    config = PipelineConfig(
        chunking=build["chunking"], models=build["embedding"], index={"name": build["index"]}
    )
    expected = _build_files(config, sources, subset, directory / "raw")
    for name in ("sections.jsonl", "chunks.jsonl"):
        if files[name] != expected[name]:
            raise ValueError("Artifact normalized records differ from pinned source/configuration")
    expected_build = json.loads(expected["build.json"])
    if set(build) != set(expected_build) or (build | {"created_at": None}) != (
        expected_build | {"created_at": None}
    ):
        raise ValueError("Artifact build definition differs from normalized source records")
    if (
        not isinstance(build["created_at"], str)
        or datetime.fromisoformat(build["created_at"]).utcoffset() is None
    ):
        raise ValueError("Artifact build creation time requires a timezone-qualified timestamp")
    return PreparedBundle(manifest, manifest_bytes, files)


def validate_artifact_target(bucket: str | None, owner: str, prefix: str) -> None:
    """Restrict transfer to an explicit ordinary bucket/owner and normalized key prefix."""
    if not bucket or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("Configure an existing general-purpose RAG_ARTIFACT_BUCKET")
    if not re.fullmatch(r"[0-9]{12}", owner):
        raise ValueError("Expected bucket owner must be a 12-digit AWS account ID")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/._-]{0,199}", prefix) or any(
        part in {"", ".", ".."} for part in prefix.split("/")
    ):
        raise ValueError("Artifact prefix must contain normalized nonempty path components")


def _put_and_verify(
    client: ArtifactS3, bucket: str, owner: str, key: str, raw: bytes
) -> dict[str, object]:
    """Never overwrite an existing key; verify bytes after creation or a 412 response."""
    digest = hashlib.sha256(raw).hexdigest()
    version = None
    status = "created"
    try:
        response = client.put_object(
            Bucket=bucket,
            Key=key,
            ExpectedBucketOwner=owner,
            Body=raw,
            ContentLength=len(raw),
            ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
            ServerSideEncryption="AES256",
            IfNoneMatch="*",
        )
        version = response.get("VersionId")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "PreconditionFailed":
            raise
        status = "existing"
    parameters = {"Bucket": bucket, "Key": key, "ExpectedBucketOwner": owner}
    if version is not None:
        parameters["VersionId"] = version
    downloaded = client.get_object(**parameters)
    body = downloaded["Body"]
    try:
        if downloaded.get("ContentLength") != len(raw):
            raise ValueError("S3 read-back length differs from the local artifact")
        received, hasher = 0, hashlib.sha256()
        while block := body.read(READ_BLOCK_BYTES):
            received += len(block)
            if received > len(raw):
                raise ValueError("S3 read-back exceeded the expected artifact size")
            hasher.update(block)
        if received != len(raw) or hasher.hexdigest() != digest:
            raise ValueError("S3 read-back SHA-256 differs from the local artifact")
    finally:
        body.close()
    return {
        "key": key,
        "sha256": digest,
        "size_bytes": len(raw),
        "write_status": status,
        "version_id": downloaded.get("VersionId", version),
        "download_verified": True,
    }


def upload_artifact_bundle(
    client: ArtifactS3,
    bundle: PreparedBundle,
    bucket: str,
    owner: str,
    prefix: str,
    output: Path,
    *,
    allow_write: bool = False,
) -> dict[str, object]:
    """Publish a verified bundle with durable partial progress; never provision or delete.

    This is an exception boundary: retain a content-free failed receipt, then re-raise
    the original SDK/integrity error. Interrupted transfers can resume at the same keys.
    """
    if not allow_write:
        raise ValueError("S3 artifact upload requires explicit --allow-write")
    validate_artifact_target(bucket, owner, prefix)
    output.mkdir(parents=True, exist_ok=False)
    root = f"{prefix}/bundles/{bundle.manifest.bundle_id}"
    objects: list[dict[str, object]] = []
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "incomplete",
        "bucket": bucket,
        "expected_owner": owner,
        "bundle_id": bundle.manifest.bundle_id,
        "corpus_id": bundle.manifest.corpus_id,
        "manifest_key": f"{root}/bundle.json",
        "objects": objects,
        "created_at": datetime.now(UTC).isoformat(),
    }
    path = output / "receipt.json"
    write_jsonl(path, [receipt])
    try:
        access = client.get_public_access_block(Bucket=bucket, ExpectedBucketOwner=owner)
        flags = access.get("PublicAccessBlockConfiguration", {})
        if not all(flags.get(flag) is True for flag in PUBLIC_ACCESS_FLAGS):
            raise ValueError("Artifact bucket must enable all four public-access blocks")
        for name, raw in sorted(bundle.files.items()):
            objects.append(_put_and_verify(client, bucket, owner, f"{root}/{name}", raw))
            write_jsonl(path, [receipt])
        objects.append(
            _put_and_verify(client, bucket, owner, f"{root}/bundle.json", bundle.manifest_bytes)
        )
        receipt["status"] = "verified"
        write_jsonl(path, [receipt])
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        write_jsonl(path, [receipt])
        raise
    return receipt
