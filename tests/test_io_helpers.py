"""Tests for src/utils/io_helpers.py."""

from __future__ import annotations

import pandas as pd
import pytest

from utils import io_helpers


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
