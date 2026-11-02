"""
Tests for ``io_helpers.drop_missing_then_sample``.

``--sample-size N`` must yield N usable rows. Dropping null text *after*
sampling (the old order in ``_process_one``) came out short, so the helper
drops first and samples second.
"""

from __future__ import annotations

import pandas as pd

from climatelens.utils.io_helpers import drop_missing_then_sample


def _frame_with_nulls(total: int = 100, every: int = 4) -> pd.DataFrame:
    """*total* rows; every *every*-th row has null text (25 of 100 by default)."""
    return pd.DataFrame(
        {
            "text": [None if i % every == 0 else f"post {i}" for i in range(total)],
            "id": range(total),
        }
    )


def test_returns_exactly_n_rows_when_null_text_present():
    df = _frame_with_nulls()
    assert df["text"].isna().sum() == 25

    out = drop_missing_then_sample(df, text_col="text", n=60, seed=42)

    assert len(out) == 60
    assert out["text"].notna().all()
    assert list(out.index) == list(range(60))


def test_n_larger_than_remaining_returns_all_remaining_rows():
    df = _frame_with_nulls()

    out = drop_missing_then_sample(df, text_col="text", n=90, seed=42)

    assert len(out) == 75
    assert out["text"].notna().all()


def test_same_seed_gives_same_rows():
    df = _frame_with_nulls()

    a = drop_missing_then_sample(df, text_col="text", n=20, seed=7)
    b = drop_missing_then_sample(df, text_col="text", n=20, seed=7)
    c = drop_missing_then_sample(df, text_col="text", n=20, seed=8)

    assert a["id"].tolist() == b["id"].tolist()
    assert a["id"].tolist() != c["id"].tolist()
