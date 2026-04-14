"""
Tests for the pure helpers in topic_modeling.py that don't require the
heavy ML stack (BERTopic, torch, sentence-transformers) to be installed.

The BERTopic import at module level would pull that whole stack in, so we
import the helpers via ``importlib`` only when those deps are available,
and otherwise skip the module.
"""

from __future__ import annotations

import importlib
from datetime import datetime

import pandas as pd
import pytest

bertopic = pytest.importorskip("bertopic", reason="heavy ML deps not installed")


@pytest.fixture(scope="module")
def tm():
    return importlib.import_module("topic_modeling")


def test_select_params_prefers_explicit_profile(tm):
    assert tm._select_params("whatever", profile="twitter") is tm.DATASET_PARAMS["twitter"]


def test_select_params_falls_back_to_default(tm):
    assert tm._select_params("novel_dataset") is tm.DATASET_PARAMS["default"]


def test_select_params_substring_fallback(tm):
    chosen = tm._select_params("my_twitter_v2")
    assert chosen is tm.DATASET_PARAMS["twitter"]


def test_calculate_optimal_bins_bounds(tm):
    # < 2 valid timestamps -> min_bins
    assert tm.calculate_optimal_bins([datetime(2020, 1, 1)], min_bins=7, max_bins=50) == 7

    # Long span -> clamped to max_bins
    stamps = [datetime(2010, 1, 1), datetime(2024, 12, 31)]
    assert tm.calculate_optimal_bins(stamps, min_bins=5, max_bins=12) == 12


def test_prepare_timestamps_reddit_unix(tm):
    df = pd.DataFrame({"created_utc": [1577836800, 1577923200, 1578009600]})
    result = tm.prepare_timestamps({"r": df}, "r")
    assert result is not None
    assert len(result) == 3


def test_prepare_timestamps_twitter_iso(tm):
    df = pd.DataFrame({"created_at": ["2020-01-01T00:00:00", "2020-01-02T00:00:00"]})
    result = tm.prepare_timestamps({"t": df}, "t")
    assert result is not None
    assert all(isinstance(x, pd.Timestamp) for x in result)


def test_prepare_timestamps_missing_column(tm):
    df = pd.DataFrame({"text": ["hi"]})
    assert tm.prepare_timestamps({"x": df}, "x") is None
