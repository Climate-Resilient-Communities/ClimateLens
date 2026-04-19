"""
Small I/O and dataframe helpers shared across pipeline stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .logging_config import get_logger

log = get_logger(__name__)


class SchemaError(ValueError):
    """Raised when a dataframe doesn't satisfy the caller's column contract."""


def require_columns(df: pd.DataFrame, required: Iterable[str], *, context: str = "") -> None:
    """
    Raise :class:`SchemaError` if any of *required* columns are missing.

    The point is to fail loudly at stage boundaries instead of producing a
    silently-empty output.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        prefix = f"{context}: " if context else ""
        raise SchemaError(
            f"{prefix}missing required columns {missing}; available: {list(df.columns)}"
        )


def pick_text_column(
    df: pd.DataFrame, candidates: Iterable[str] = ("body", "text")
) -> Optional[str]:
    """Return the first candidate column present in *df*, or ``None``."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def drop_missing_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Drop rows where *text_col* is NaN and reset the index.

    Keeping these in sync is critical for positional assignment downstream
    (e.g. ``df["topic"] = topics``), so we do it once at load time.
    """
    if text_col not in df.columns:
        raise SchemaError(f"text column {text_col!r} not found in dataframe")
    before = len(df)
    df = df[df[text_col].notna()].reset_index(drop=True)
    after = len(df)
    if after < before:
        log.info(
            "dropped %d rows with null %r (%d -> %d)",
            before - after,
            text_col,
            before,
            after,
        )
    return df


def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def discover_csvs(data_dir: Path, *, patterns: Iterable[str] = ("*.csv",)) -> List[Path]:
    """Return CSVs in *data_dir* matching any of the given glob patterns, sorted."""
    data_dir = Path(data_dir)
    seen: List[Path] = []
    for pattern in patterns:
        seen.extend(sorted(data_dir.glob(pattern)))
    # Preserve order while deduping.
    uniq: List[Path] = []
    seen_set = set()
    for p in seen:
        if p not in seen_set:
            uniq.append(p)
            seen_set.add(p)
    return uniq
