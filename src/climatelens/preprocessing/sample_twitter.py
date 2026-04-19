"""
Create a fixed-size sample from a larger cleaned Twitter CSV.

Usage::

    python src/utils/twitter_sample.py --input <cleaned.csv> \\
        --output <sample.csv> [--n 100000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_sample(input_path: Path, output_path: Path, n: int = 100_000) -> Path:
    df = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(n).to_csv(output_path, index=False)
    print(f"Sample dataset created: {output_path} ({min(n, len(df))} rows)")
    return output_path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=100_000)
    args = parser.parse_args()
    make_sample(args.input, args.output, args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
