"""Verify configuration precedence and useful failures at the CLI boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.cli import main
from rag.config import load_config


def test_environment_overrides_file_and_blank_values_do_not(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.toml"
    config_file.write_text(
        '[cloud]\nregion = "us-east-1"\n[models]\nembedding_dimensions = 512\n', encoding="utf-8"
    )
    config = load_config(
        config_file,
        environ={"RAG_AWS_REGION": "", "RAG_EMBEDDING_DIMENSIONS": "256"},
    )
    assert config.cloud.region == "us-east-1"
    assert config.models.embedding_dimensions == 256


def test_cloud_settings_require_explicit_positive_allowance() -> None:
    environment = {"RAG_AWS_REGION": "us-east-1", "RAG_GENERATION_MODEL_ID": "test-model"}
    with pytest.raises(ValueError, match="RAG_DEVELOPMENT_MODEL_BUDGET_USD"):
        load_config(environ=environment).require_cloud_settings()
    environment["RAG_DEVELOPMENT_MODEL_BUDGET_USD"] = "1.50"
    config = load_config(environ=environment)
    config.require_cloud_settings()
    with pytest.raises(ValueError, match="RAG_ARTIFACT_BUCKET"):
        config.require_cloud_settings(require_bucket=True)


def test_cli_reports_malformed_configuration_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text('models = "invalid"\n', encoding="utf-8")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "512")
    assert main(["--config", str(path), "config"]) == 1
    error = capsys.readouterr().err
    assert "must be a TOML table" in error
    assert "Traceback" not in error
