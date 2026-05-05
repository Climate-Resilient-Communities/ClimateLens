import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas import DataFrame


# Process CSV datasets from a given directory
def process_datasets(
    data_path: str, text_cols: Tuple[str, ...] = ("body", "text")
) -> Tuple[Dict[str, DataFrame], Dict[str, List[str]], Dict[str, Path]]:
    """
    Returns:
       Tuple of (dfs, docs_dict, datasets)
       - dfs: Dictionary mapping dataset names to DataFrames
       - docs_dict: Dictionary mapping dataset names to lists of document texts
       - datasets: Dictionary mapping dataset names to file paths
    """
    # maps dataset names to...
    datasets: Dict[str, Path] = {}  # file paths
    dfs: Dict[str, DataFrame] = {}  # DataFrames
    docs_dict: Dict[str, List[str]] = {}  # lists of document texts
    failed: List[str] = []

    data_path = Path(data_path)

    for file_path in data_path.glob("*.csv"):
        name = re.sub(r"^filtered_|_clean$", "", file_path.stem)
        datasets[name] = file_path

        try:
            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                print(f"{name} Parser error, switching to python interpreter")
                df = pd.read_csv(
                    file_path,
                    engine="python",
                    on_bad_lines="skip",  # maybe experiment with "warn"
                )

            text_col = next((c for c in text_cols if c in df.columns), None)

            if not text_col:
                print(f"Skipping {name}. No {text_cols} column found.")
                failed.append(name)
                continue

            dfs[name] = df
            docs_dict[name] = df[df[text_col].notna()][text_col].tolist()

            print(f"Loaded {name} ({len(dfs[name])} rows)")

        except Exception as e:
            print(f"Error loading {name}: {e}")
            failed.append(name)

    print(f"{len(dfs)}/{len(datasets)} datasets loaded successfully")
    if failed:
        print(f"Failed to load: {', '.join(failed)}")

    return dfs, docs_dict, datasets
