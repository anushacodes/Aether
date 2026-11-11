"""Verify explicit rollback against isolated real local indexes, without model calls."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from rag.cli import main
from rag.config import load_config
from rag.corpus import load_sources
from rag.models import SubsetManifest
from rag.publication import (
    StaleRevisionError,
    active_index,
    build_snapshot,
    publish_snapshot,
)
from rag.search_store import create_search_client
from rag.snapshots import retained_snapshots, rollback_snapshot


def check(directory: Path) -> dict[str, object]:
    """Keep the default pilot untouched while restoring previous and empty corpus states."""
    config = load_config(Path("config.toml"))
    config.publication.alias = f"s3-rag-rollback-check-{uuid4().hex}"
    config.index.name = config.publication.alias
    config.publication.state_path = directory / "state.sqlite3"
    config.publication.snapshot_dir = directory / "snapshots"
    sources = load_sources(Path("data/manifests/sources.json"))
    subset = SubsetManifest.model_validate_json(Path("data/manifests/pilot.json").read_bytes())
    created: list[str] = []
    with create_search_client(config) as search:
        try:
            builds = []
            for sections in (subset.sections, subset.sections[1:], []):
                identity = uuid4().hex
                created.append(f"s3-rag-snapshot-{identity}")
                selected = subset.model_copy(update={"sections": sections})
                builds.append(
                    build_snapshot(search, config, sources, selected, Path("data/raw"), identity)
                )
            original, replacement, empty = builds
            publish_snapshot(search, config, original, None)
            try:
                rollback_snapshot(search, config, replacement.snapshot_id, original.index_name)
                raise AssertionError("An unpublished build was accepted for rollback")
            except ValueError as error:
                assert "previously published" in str(error)
            publish_snapshot(search, config, replacement, original.index_name)
            restored_output = StringIO()
            with (
                patch("rag.cli.load_config", return_value=config),
                redirect_stdout(restored_output),
            ):
                assert (
                    main(
                        [
                            "index",
                            "rollback",
                            original.snapshot_id,
                            "--base-revision",
                            replacement.index_name,
                        ]
                    )
                    == 0
                )
            assert json.loads(restored_output.getvalue())["snapshot_id"] == original.snapshot_id
            assert active_index(search, config.publication.alias) == original.index_name
            assert search.count(index=config.publication.alias)["count"] == 3
            assert search.count(index=replacement.index_name)["count"] == 2
            # A repeat after a lost acknowledgement is safe even with the old base.
            rollback_snapshot(search, config, original.snapshot_id, replacement.index_name)
            try:
                rollback_snapshot(search, config, replacement.snapshot_id, empty.index_name)
                raise AssertionError("A stale rollback overwrote the active corpus")
            except StaleRevisionError:
                pass
            chunk_path = replacement.manifest_path.parent / "chunks.jsonl"
            saved = chunk_path.read_bytes()
            chunk_path.write_bytes(saved + b"\n")
            try:
                rollback_snapshot(search, config, replacement.snapshot_id, original.index_name)
                raise AssertionError("Changed rollback artifacts were accepted")
            except ValueError as error:
                assert "artifact hash mismatch" in str(error)
            finally:
                chunk_path.write_bytes(saved)
            search.indices.put_settings(
                index=replacement.index_name, body={"index.blocks.write": False}
            )
            try:
                rollback_snapshot(search, config, replacement.snapshot_id, original.index_name)
                raise AssertionError("A writable rollback target was accepted")
            except ValueError as error:
                assert "write-blocked" in str(error)
            finally:
                search.indices.put_settings(
                    index=replacement.index_name, body={"index.blocks.write": True}
                )
            publish_snapshot(search, config, empty, original.index_name)
            assert search.count(index=config.publication.alias)["count"] == 0
            rollback_snapshot(search, config, replacement.snapshot_id, empty.index_name)
            assert search.count(index=config.publication.alias)["count"] == 2
            assert search.count(index=empty.index_name)["count"] == 0
            listed = retained_snapshots(search, config)
            assert len(listed) == 3
            assert [row.snapshot.snapshot_id for row in listed if row.active] == [
                replacement.snapshot_id
            ]
            listing_output = StringIO()
            with patch("rag.cli.load_config", return_value=config), redirect_stdout(listing_output):
                assert main(["index", "snapshots"]) == 0
            assert len(json.loads(listing_output.getvalue())["snapshots"]) == 3
            return {
                "passed": [
                    "unpublished rollback rejected",
                    "atomic restore without rebuilding",
                    "repeated rollback is idempotent",
                    "stale rollback rejected",
                    "tampered artifacts rejected",
                    "writable target rejected",
                    "restore after delete-all",
                    "replaced snapshots remain readable",
                    "publication history distinguishes active target",
                    "CLI rollback and listing",
                ],
                "real_aws_requests": 0,
            }
        finally:
            for index in created:
                if search.indices.exists(index=index):
                    search.indices.delete(index=index)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="s3-rag-rollback-") as location:
        print(json.dumps(check(Path(location)), indent=2))
