"""
Split a cleaned Twitter CSV into N equal-sized chunks.

Usage::

    python src/utils/twitter_chunks.py --input <cleaned.csv> \\
        --output-dir <dir> [--n-chunks 32]

This used to be an orphan notebook cell referencing a ``df_full`` that was
never defined at module scope - running it raised ``NameError``. It's now a
self-contained CLI so downstream callers and tests can exercise it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def split_dataframe(df: pd.DataFrame, n_chunks: int):
    """Yield successive roughly-equal chunks of *df*."""
    if n_chunks <= 0:
        raise ValueError("n_chunks must be >= 1")
    if len(df) == 0:
        return
    chunk_size = len(df) // n_chunks + (len(df) % n_chunks > 0)
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = df.iloc[start:end]
        if chunk.empty:
            return
        yield i, chunk


def write_chunks(
    input_path: Path,
    output_dir: Path,
    n_chunks: int = 32,
    prefix: str = "climate_twitter_clean",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    written: list[Path] = []
    for i, chunk in split_dataframe(df, n_chunks):
        out = output_dir / f"{prefix}_{i + 1}.csv"
        chunk.to_csv(out, index=False)
        written.append(out)
        print(f"Saved {len(chunk)} rows to {out}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-chunks", type=int, default=32)
    parser.add_argument("--prefix", default="climate_twitter_clean")
    args = parser.parse_args()

    write_chunks(args.input, args.output_dir, args.n_chunks, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
