"""
Tests for the pure helpers in dynamic_topic_modeling.py.

That module imports BERTopic (and plotly) at module level, which pulls in the
heavy ML stack, so we import it via ``importlib`` only when those deps are
available, and otherwise skip the module.
"""

from __future__ import annotations

import importlib
from datetime import datetime

import pandas as pd
import pytest

bertopic = pytest.importorskip("bertopic", reason="heavy ML deps not installed")


@pytest.fixture(scope="module")
def dtm():
    return importlib.import_module("climatelens.nlp_pipeline.dynamic_topic_modeling")


def test_calculate_optimal_bins_bounds(dtm):
    # < 2 valid timestamps -> min_bins
    assert dtm.calculate_optimal_bins([datetime(2020, 1, 1)], min_bins=7, max_bins=50) == 7

    # Long span -> clamped to max_bins
    stamps = [datetime(2010, 1, 1), datetime(2024, 12, 31)]
    assert dtm.calculate_optimal_bins(stamps, min_bins=5, max_bins=12) == 12


def test_prepare_timestamps_reddit_unix(dtm):
    df = pd.DataFrame({"created_utc": [1577836800, 1577923200, 1578009600]})
    result = dtm.prepare_timestamps({"r": df}, "r")
    assert result is not None
    assert len(result) == 3


def test_prepare_timestamps_twitter_iso(dtm):
    df = pd.DataFrame({"created_at": ["2020-01-01T00:00:00", "2020-01-02T00:00:00"]})
    result = dtm.prepare_timestamps({"t": df}, "t")
    assert result is not None
    assert all(isinstance(x, pd.Timestamp) for x in result)


def test_prepare_timestamps_missing_column(dtm):
    df = pd.DataFrame({"text": ["hi"]})
    assert dtm.prepare_timestamps({"x": df}, "x") is None
