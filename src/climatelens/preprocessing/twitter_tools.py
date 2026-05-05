from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()
data_dir, twitter_raw_dir = os.getenv("DATA_DIR"), os.getenv("TWITTER_RAW_DIR")


def make_sample(input_path: Path, output_path: Path, n: int = 100_000) -> Path:
    df = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(n).to_csv(output_path, index=False)
    print(f"Sample dataset created: {output_path} ({min(n, len(df))} rows)")
    return output_path


"""
Split a cleaned Twitter CSV into N equal-sized chunks.

Usage::

    python src/utils/twitter_chunks.py --input <cleaned.csv> \\
        --output-dir <dir> [--n-chunks 32]

This used to be an orphan notebook cell referencing a ``df_full`` that was
never defined at module scope - running it raised ``NameError``. It's now a
self-contained CLI so downstream callers and tests can exercise it.
"""


def split_dataframe(df: DataFrame, n_chunks: int):
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


"""
Process raw Twitter JSONL data into a cleaned CSV format.
"""


def preview_jsonl(file_path: str, num_lines: int = 10) -> List[str]:
    preview_data: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in range(num_lines):
            try:
                preview_data.append(json.loads(f.readline()))
            except (json.JSONDecodeError, StopIteration):
                continue
    return preview_data


def convert_jsonl_to_csv(input_path: str, output_path: str, fieldnames: List[str]):
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    writer.writerow({field: data.get(field, "") for field in fieldnames})
                except json.JSONDecodeError:
                    continue


def process_twitter_data(
    input_path: str, output_path: str, desired_columns: Optional[List[str]] = None
) -> DataFrame:
    # Process Twitter JSONL data and save as CSV.

    if desired_columns is None:  # columns we want to keep for analysis
        desired_columns = ["created_at", "text"]

    # Preview and display column info
    preview_rows = preview_jsonl(input_path)
    df_preview = DataFrame(preview_rows)

    print("Preview columns:", df_preview.columns.tolist())
    print(f"\nSample of {desired_columns[0]} and {desired_columns[1]}:")

    # Convert full file
    convert_jsonl_to_csv(input_path, output_path, desired_columns)

    # Load and display final info
    df_full = pd.read_csv(output_path)
    print("\nTwitter dataframe information:")
    print(df_full.info())

    return df_full


def process_twitter_data_main() -> DataFrame:
    input_path = Path(twitter_raw_dir) / "climate.jsonl"  # raw NDJSON Twitter file
    output_path = Path(data_dir) / "twitter_climate_clean.csv"  # cleaned CSV file

    df = process_twitter_data(input_path, output_path)  # Process data

    if len(df) >= 5:
        print("\nRandom sample (5 rows):")
        print(df.sample(5))

    return df


def main() -> int:
    """CLI entry point - choose between sampling or chunking."""
    parser = argparse.ArgumentParser(description="Twitter data processing utilities")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Create a sample CSV")
    sample_parser.add_argument("--input", required=True, type=Path)
    sample_parser.add_argument("--output", required=True, type=Path)
    sample_parser.add_argument("--n", type=int, default=100_000)

    # Chunks command
    chunk_parser = subparsers.add_parser("chunks", help="Split into chunks")
    chunk_parser.add_argument("--input", required=True, type=Path)
    chunk_parser.add_argument("--output-dir", required=True, type=Path)
    chunk_parser.add_argument("--n-chunks", type=int, default=32)
    chunk_parser.add_argument("--prefix", default="climate_twitter_clean")

    args = parser.parse_args()

    if args.command == "sample":
        make_sample(args.input, args.output, args.n)
    elif args.command == "chunks":
        write_chunks(args.input, args.output_dir, args.n_chunks, args.prefix)

    return 0


if __name__ == "__main__":
    # Choose which main to run
    # For CLI tool: raise SystemExit(main())
    # For Twitter processing: df = process_twitter_data_main()
    df = process_twitter_data_main()
