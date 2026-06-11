"""Basic tests for data loading."""

import pandas as pd
import pytest

from src.climatelens.utils.process_datasets import process_datasets


# Test that a valid CSV file loads correctly
def test_loads_valid_csv_with_body_column(tmp_path):
    # Create a test file
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "body": ["Hello world", "Another post"],
        "created_utc": [123456, 789012]
    })
    df.to_csv(csv_file, index=False)

    # Load it
    dfs, docs_dict, _ = process_datasets(str(tmp_path))

    # Check it worked
    assert len(dfs) == 1
    assert len(docs_dict["test_data"]) == 2
    assert docs_dict["test_data"][0] == "Hello world"

# Test that files without body/text column are skipped
def test_skips_file_with_wrong_column(tmp_path, capsys):
    csv_file = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_column": ["data"]}).to_csv(csv_file, index=False)

    dfs, _, _ = process_datasets(str(tmp_path))

    assert len(dfs) == 0
    captured = capsys.readouterr()
    assert "No ('body', 'text') column found" in captured.out

# Test that multiple files are all loaded
def test_handles_multiple_files(tmp_path):
    for name in ["reddit", "twitter", "climate"]:
        file = tmp_path / f"{name}.csv"
        pd.DataFrame({"body": ["test"]}).to_csv(file, index=False)

    dfs, _, _ = process_datasets(str(tmp_path))

    assert set(dfs.keys()) == {"reddit", "twitter", "climate"}
