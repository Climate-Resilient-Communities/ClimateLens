"""
Pytest configuration for the ClimateLens test suite.

Exposes ``src/`` on ``sys.path`` so tests can import ``utils.*`` and the
pipeline modules directly, plus a handful of dataframe / path fixtures used
across multiple test files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Make the pipeline modules importable.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def src_dir() -> Path:
    return SRC_DIR


@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    """Directory holding the tiny CSVs committed under ``data/``."""
    return REPO_ROOT / "data"


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, sample_data_dir: Path) -> Path:
    """Copy the committed sample CSVs into a scratch directory."""
    dest = tmp_path / "data"
    dest.mkdir()
    for src in sample_data_dir.glob("*.csv"):
        (dest / src.name).write_bytes(src.read_bytes())
    return dest


@pytest.fixture()
def twitter_fixture_df() -> pd.DataFrame:
    """Minimal Twitter-shaped dataframe."""
    return pd.DataFrame(
        {
            "created_at": [
                "2020-01-01T00:00:00",
                "2020-01-02T00:00:00",
                "2020-01-03T00:00:00",
            ],
            "text": [
                "RT @someone Check https://t.co/abc climate climate climate change!",
                "@handle I love clean energy https://example.com",
                "Global warming is real, but hope remains",
            ],
        }
    )


@pytest.fixture()
def reddit_fixture_df() -> pd.DataFrame:
    """Minimal Reddit-shaped dataframe."""
    return pd.DataFrame(
        {
            "subreddit": ["anticonsumption"] * 3,
            "body": [
                "eco-anxiety is making me lose sleep",
                "we should reduce consumption and waste",
                "climate grief is real for younger folks",
            ],
            "created_utc": [1577836800, 1577923200, 1578009600],
        }
    )
