"""
End-to-end smoke test for the preprocessing stage.

Runs ``data_preprocessing.run_pipeline`` against the small sample CSVs
committed in ``data/`` and asserts:

* The raw input files are **not modified** (key invariant -- the previous
  version of the code overwrote them in place).
* A preprocessed CSV is written for every registered dataset.
* Each output has the ``cleaned_text`` column and non-empty rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import data_preprocessing as dp
from utils import datasets as ds_mod


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preprocessing_end_to_end_does_not_mutate_raw(tmp_data_dir, tmp_path):
    processed_dir = tmp_path / "processed"

    # Snapshot raw-file hashes before we run anything.
    before = {p.name: _hash(p) for p in tmp_data_dir.glob("*.csv")}
    assert before, "fixture data should include at least one CSV"

    written = dp.run_pipeline(tmp_data_dir, processed_dir)

    # Raw files must be byte-identical after the run.
    after = {p.name: _hash(p) for p in tmp_data_dir.glob("*.csv")}
    assert before == after, "preprocessing must not mutate raw inputs"

    # Every resolvable dataset in the registry produced an output.
    specs = ds_mod.discover_datasets(tmp_data_dir)
    assert len(written) == len(specs)
    assert {Path(p).name for p in written} == {f"{s.name}.csv" for s in specs}


def test_preprocessing_outputs_have_cleaned_text(tmp_data_dir, tmp_path):
    processed_dir = tmp_path / "processed"
    written = dp.run_pipeline(tmp_data_dir, processed_dir)

    assert written, "expected at least one preprocessed file"
    for out_path in written:
        df = pd.read_csv(out_path)
        assert "cleaned_text" in df.columns
        assert len(df) > 0
        # Every row meets the minimum length policy.
        assert (df["cleaned_text"].str.split().str.len() >= dp.MIN_DOCUMENT_WORDS).all()
