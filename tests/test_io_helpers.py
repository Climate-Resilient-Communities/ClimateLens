"""Tests for src/utils/io_helpers.py."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from climatelens.utils import io_helpers


def test_require_columns_ok():
    df = pd.DataFrame({"a": [1], "b": [2]})
    io_helpers.require_columns(df, ["a", "b"])


def test_require_columns_missing_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(io_helpers.SchemaError, match="missing required columns"):
        io_helpers.require_columns(df, ["a", "b"], context="fixture")


def test_pick_text_column_prefers_first_candidate():
    df = pd.DataFrame({"text": ["x"], "body": ["y"]})
    assert io_helpers.pick_text_column(df, ("body", "text")) == "body"


def test_pick_text_column_none_when_absent():
    df = pd.DataFrame({"other": ["x"]})
    assert io_helpers.pick_text_column(df) is None


def test_drop_missing_text_drops_nan_and_resets_index():
    df = pd.DataFrame({"text": ["a", None, "c"], "other": [1, 2, 3]})
    out = io_helpers.drop_missing_text(df, "text")
    assert list(out["text"]) == ["a", "c"]
    assert list(out.index) == [0, 1]


def test_drop_missing_text_raises_on_missing_column():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(io_helpers.SchemaError):
        io_helpers.drop_missing_text(df, "text")


def test_safe_write_csv_creates_parent(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    out = tmp_path / "sub" / "nested" / "file.csv"
    io_helpers.safe_write_csv(df, out)
    assert out.exists()
    roundtrip = pd.read_csv(out)
    assert roundtrip.equals(df)


def test_discover_csvs_sorted_and_unique(tmp_path):
    (tmp_path / "b.csv").write_text("")
    (tmp_path / "a.csv").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = io_helpers.discover_csvs(tmp_path)
    assert [p.name for p in result] == ["a.csv", "b.csv"]


# ---------------------------------------------------------------------------
# write helpers + dry_run
#
# The figure stand-ins expose the one method each helper calls, so these
# tests need neither matplotlib nor plotly (CI installs neither).
# ---------------------------------------------------------------------------


class _FakeMatplotlibFigure:
    def __init__(self):
        self.kwargs = None

    def savefig(self, path, **kwargs):
        self.kwargs = kwargs
        Path(path).write_bytes(b"png")


class _FakePlotlyFigure:
    def write_html(self, path):
        Path(path).write_text("<html></html>")


def _write_with(helper, path, *, dry_run):
    if helper is io_helpers.safe_write_csv:
        return helper(pd.DataFrame({"a": [1]}), path, dry_run=dry_run)
    if helper is io_helpers.safe_savefig:
        return helper(_FakeMatplotlibFigure(), path, dry_run=dry_run, dpi=10)
    if helper is io_helpers.safe_write_html:
        return helper(_FakePlotlyFigure(), path, dry_run=dry_run)
    return helper("hello", path, dry_run=dry_run)


WRITE_HELPERS = [
    io_helpers.safe_write_csv,
    io_helpers.safe_savefig,
    io_helpers.safe_write_html,
    io_helpers.safe_write_text,
]


@pytest.mark.parametrize("helper", WRITE_HELPERS, ids=lambda h: h.__name__)
def test_dry_run_writes_nothing(helper, tmp_path, caplog):
    out = tmp_path / "never" / "created" / "out.file"

    with caplog.at_level(logging.INFO, logger="climatelens.utils.io_helpers"):
        returned = _write_with(helper, out, dry_run=True)

    assert returned == out
    assert not out.exists()
    assert not out.parent.exists()
    assert not (tmp_path / "never").exists()
    assert [r.getMessage() for r in caplog.records] == [f"[dry-run] would write {out}"]


@pytest.mark.parametrize("helper", WRITE_HELPERS, ids=lambda h: h.__name__)
def test_write_helpers_create_parent_and_file(helper, tmp_path):
    out = tmp_path / "sub" / "nested" / "out.file"

    returned = _write_with(helper, out, dry_run=False)

    assert returned == out
    assert out.exists()


def test_safe_savefig_forwards_savefig_kwargs(tmp_path):
    fig = _FakeMatplotlibFigure()
    io_helpers.safe_savefig(fig, tmp_path / "f.png", dpi=150, bbox_inches="tight")
    assert fig.kwargs == {"dpi": 150, "bbox_inches": "tight"}


def test_safe_write_text_is_utf8(tmp_path):
    out = tmp_path / "r.html"
    io_helpers.safe_write_text("café 🌍", out)
    assert out.read_text(encoding="utf-8") == "café 🌍"
