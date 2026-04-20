"""Tests for src/utils/runtime.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from climatelens.utils import runtime


@pytest.fixture(autouse=True)
def clean_azureml_env(monkeypatch):
    """Ensure AzureML-detection doesn't leak between tests."""
    for name in runtime._AZUREML_SIGNALS:
        monkeypatch.delenv(name, raising=False)


def test_is_azureml_false_by_default(monkeypatch):
    assert runtime.is_azureml() is False


def test_is_azureml_detects_each_signal(monkeypatch):
    for name in runtime._AZUREML_SIGNALS:
        monkeypatch.setenv(name, "x")
        assert runtime.is_azureml() is True
        monkeypatch.delenv(name)


def test_find_dotenv_walks_up(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DATA_DIR=/x")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    result = runtime.find_dotenv(nested)

    assert result == tmp_path / ".env"


def test_find_dotenv_returns_none_when_missing(tmp_path):
    assert runtime.find_dotenv(tmp_path) is None


def test_load_runtime_uses_env_overrides(tmp_path, monkeypatch):
    data = tmp_path / "my_data"
    processed = tmp_path / "my_processed"
    output = tmp_path / "my_output_data"
    vis = tmp_path / "my_vis"
    models = tmp_path / "my_models"
    data.mkdir()

    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("PROCESSED_DATA_DIR", str(processed))
    monkeypatch.setenv("OUTPUT_DATA_DIR", str(output))
    monkeypatch.setenv("OUTPUT_VIS_DIR", str(vis))
    monkeypatch.setenv("MODELS_DIR", str(models))

    cfg = runtime.load_runtime(search_from=tmp_path)

    assert cfg.data_dir == data
    assert cfg.processed_data_dir == processed
    assert cfg.output_data_dir == output
    assert cfg.output_vis_dir == vis
    assert cfg.models_dir == models
    # Write-side dirs should be created eagerly.
    assert processed.exists()
    assert output.exists()
    assert vis.exists()
    assert models.exists()


def test_load_runtime_falls_back_to_repo_defaults(tmp_path, monkeypatch):
    # No env vars, no .env -> use repo-relative defaults but rooted at the
    # real repo (since runtime.py looks relative to itself).
    monkeypatch.chdir(tmp_path)
    cfg = runtime.load_runtime(search_from=tmp_path, ensure_dirs=False)

    assert cfg.data_dir.name == "data"
    assert cfg.processed_data_dir.name == "processed"
    assert cfg.in_azureml is False


def test_load_runtime_azureml_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("AZUREML_RUN_ID", "fake-run")
    monkeypatch.chdir(tmp_path)

    cfg = runtime.load_runtime(search_from=tmp_path, ensure_dirs=True)

    assert cfg.in_azureml is True
    assert cfg.data_dir == Path("./data")
    assert cfg.output_vis_dir == Path("./outputs/visualizations")
    assert (tmp_path / "outputs" / "visualizations").exists()
