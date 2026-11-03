"""
Small I/O and dataframe helpers shared across pipeline stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional

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


def _skip_write(path: Path, dry_run: bool) -> bool:
    """Log and return ``True`` when *dry_run* says this write must not happen."""
    if dry_run:
        log.info("[dry-run] would write %s", path)
    return dry_run


def safe_write_csv(df: pd.DataFrame, path: Path, *, dry_run: bool = False) -> Path:
    """Write a CSV to *path*, creating parent directories as needed.

    With ``dry_run=True`` nothing is created -- neither the file nor its parent
    directory -- and a single ``[dry-run] would write <path>`` line is logged.
    The same contract holds for every ``safe_*`` writer below.
    """
    path = Path(path)
    if _skip_write(path, dry_run):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def safe_savefig(fig: Any, path: Path, *, dry_run: bool = False, **savefig_kwargs: Any) -> Path:
    """Save a matplotlib figure via ``fig.savefig(path, **savefig_kwargs)``."""
    path = Path(path)
    if _skip_write(path, dry_run):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **savefig_kwargs)
    return path


def safe_write_html(fig: Any, path: Path, *, dry_run: bool = False) -> Path:
    """Write a plotly figure via ``fig.write_html``."""
    path = Path(path)
    if _skip_write(path, dry_run):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    return path


def safe_write_text(text: str, path: Path, *, dry_run: bool = False) -> Path:
    """Write *text* to *path* as UTF-8."""
    path = Path(path)
    if _skip_write(path, dry_run):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def sample_dataframe(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    """Return a seeded random sample of *n* rows, re-indexed from zero.

    Sampling is random rather than a head slice: the Reddit export is ordered
    oldest-first, so ``df.head(n)`` would return only the earliest rows and
    silently break any time-based analysis downstream.

    The index is reset because later stages concatenate positionally against a
    fresh ``RangeIndex`` (see ``add_emotions_to_datasets``); a sparse index
    from ``sample`` would align as a union and produce NaN rows instead.

    ``n`` of ``None``, or one that is not smaller than the frame, returns *df*
    unchanged.
    """
    if n is None or n >= len(df):
        return df
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    log.info("sampled %d of %d rows (seed=%d)", len(sampled), len(df), seed)
    return sampled


def drop_missing_then_sample(
    df: pd.DataFrame, *, text_col: str, n: Optional[int], seed: int
) -> pd.DataFrame:
    """Drop rows with missing *text_col*, then take a seeded sample of *n*.

    Dropping first means ``n`` is the number of usable rows returned: exactly
    ``n``, or every remaining row if fewer than ``n`` survive the drop.
    Sampling first and dropping afterwards would come out short.
    """
    df = drop_missing_text(df, text_col)
    return sample_dataframe(df, n, seed)


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
