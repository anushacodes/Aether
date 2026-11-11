from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.cli import main
from rag.config import load_config


def test_environment_overrides_file_and_blank_values_do_not(tmp_path):
    config_file = tmp_path / "settings.toml"
    config_file.write_text('[cloud]\nregion = "us-east-1"\n[models]\nembedding_dimensions = 512\n')
    config = load_config(
        config_file,
        environ={"RAG_AWS_REGION": "", "RAG_EMBEDDING_DIMENSIONS": "256"},
    )
    assert config.cloud.region == "us-east-1"
    assert config.models.embedding_dimensions == 256


@pytest.mark.parametrize("dimensions", ["1536", "not-a-number"])
def test_invalid_embedding_dimensions_fail(dimensions):
    with pytest.raises(ValueError):
        load_config(environ={"RAG_EMBEDDING_DIMENSIONS": dimensions})


def test_local_config_needs_no_cloud_settings():
    config = load_config(Path("config.toml"), environ={})
    assert config.cloud.region is None
    with pytest.raises(ValueError, match="RAG_AWS_REGION"):
        config.require_cloud_settings()


def test_config_rejects_typo_and_invalid_overlap(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text("[chunking]\nsize = 100\noverlap = 100\n")
    with pytest.raises(ValidationError, match="overlap"):
        load_config(path, environ={})
    path.write_text("embedding_dimentions = 1024\n")
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_config(path, environ={})


def test_cloud_settings_require_explicit_positive_allowance():
    environment = {"RAG_AWS_REGION": "us-east-1", "RAG_GENERATION_MODEL_ID": "test-model"}
    with pytest.raises(ValueError, match="RAG_DEVELOPMENT_MODEL_BUDGET_USD"):
        load_config(environ=environment).require_cloud_settings()
    environment["RAG_DEVELOPMENT_MODEL_BUDGET_USD"] = "1.50"
    config = load_config(environ=environment)
    config.require_cloud_settings()
    with pytest.raises(ValueError, match="RAG_ARTIFACT_BUCKET"):
        config.require_cloud_settings(require_bucket=True)
    environment["RAG_DEVELOPMENT_MODEL_BUDGET_USD"] = "nan"
    with pytest.raises(ValidationError):
        load_config(environ=environment)


def test_cli_reports_missing_inputs_without_traceback(tmp_path, capsys):
    assert main(["corpus", "extract", "--sources", str(tmp_path / "missing.json")]) == 1
    error = capsys.readouterr().err
    assert "Error:" in error
    assert "Traceback" not in error
