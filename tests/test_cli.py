"""
Tests for the ``climatelens`` CLI (ticket #36).

The pipeline callables are always mocked -- these tests must never load a
model or touch the filesystem. ``climatelens.cli`` imports them lazily inside
``_load_pipelines``, so patching that one seam keeps the whole heavy ML stack
out of the test run (and out of CI, which does not install it).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from climatelens import cli as cli_module


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def pipelines(monkeypatch):
    """Patch out both pipeline stages; hand the mocks back to the test."""
    emotion = MagicMock(return_value=0, name="run_emotion_pipeline")
    visualization = MagicMock(return_value=None, name="run_visualization_pipeline")
    monkeypatch.setattr(cli_module, "_load_pipelines", lambda: (emotion, visualization))
    return emotion, visualization


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_group_help(runner):
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output


def test_run_help_lists_options(runner):
    result = runner.invoke(cli_module.cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--sample-size" in result.output
    assert "--seed" in result.output


def test_help_does_not_import_the_pipelines(runner, monkeypatch):
    """--help must not pull in torch/transformers."""
    boom = MagicMock(side_effect=AssertionError("pipelines imported for --help"))
    monkeypatch.setattr(cli_module, "_load_pipelines", boom)

    result = runner.invoke(cli_module.cli, ["run", "--help"])

    assert result.exit_code == 0
    boom.assert_not_called()


# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------


def test_success_exits_zero(runner, pipelines):
    emotion, visualization = pipelines

    result = runner.invoke(cli_module.cli, ["run"])

    assert result.exit_code == 0
    emotion.assert_called_once()
    visualization.assert_called_once()


def test_nonzero_status_from_emotion_stage_exits_nonzero(runner, pipelines):
    emotion, visualization = pipelines
    emotion.return_value = 1

    result = runner.invoke(cli_module.cli, ["run"])

    assert result.exit_code != 0
    # The second stage must not run once the first has failed.
    visualization.assert_not_called()


def test_exception_in_emotion_stage_is_not_swallowed(runner, pipelines):
    emotion, visualization = pipelines
    emotion.side_effect = RuntimeError("model load failed")

    result = runner.invoke(cli_module.cli, ["run"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    visualization.assert_not_called()


def test_exception_in_visualization_stage_is_not_swallowed(runner, pipelines):
    _, visualization = pipelines
    visualization.side_effect = RuntimeError("plotly blew up")

    result = runner.invoke(cli_module.cli, ["run"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


# ---------------------------------------------------------------------------
# option plumbing
# ---------------------------------------------------------------------------


def test_sample_size_and_seed_are_passed_through(runner, pipelines):
    emotion, visualization = pipelines

    result = runner.invoke(cli_module.cli, ["run", "--sample-size", "250", "--seed", "7"])

    assert result.exit_code == 0
    emotion.assert_called_once_with(sample_size=250, seed=7)
    visualization.assert_called_once_with(sample_size=250, seed=7)


def test_defaults_are_none_and_42(runner, pipelines):
    emotion, visualization = pipelines

    result = runner.invoke(cli_module.cli, ["run"])

    assert result.exit_code == 0
    emotion.assert_called_once_with(sample_size=None, seed=42)
    visualization.assert_called_once_with(sample_size=None, seed=42)


def test_sample_size_must_be_positive(runner, pipelines):
    emotion, _ = pipelines

    result = runner.invoke(cli_module.cli, ["run", "--sample-size", "0"])

    assert result.exit_code != 0
    emotion.assert_not_called()


def test_sample_size_must_be_an_integer(runner, pipelines):
    emotion, _ = pipelines

    result = runner.invoke(cli_module.cli, ["run", "--sample-size", "lots"])

    assert result.exit_code != 0
    emotion.assert_not_called()
