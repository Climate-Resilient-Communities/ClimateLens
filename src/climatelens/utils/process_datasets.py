import re
from pathlib import Path

import pandas as pd


def process_datasets(data_path, text_cols=("body", "text")):
    datasets, dfs, docs_dict, failed = {}, {}, {}, []
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
