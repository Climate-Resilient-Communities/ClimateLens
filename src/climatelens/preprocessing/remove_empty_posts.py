### finding empty rows/posts

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def load_environment():
    env_path = Path(__file__).resolve().parent / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        data_dir = os.getenv("DATA_DIR")
        print("env variables loaded")
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    return data_dir


data_dir = load_environment()
if not data_dir:
    raise EnvironmentError("DATA_DIR must be set in the .env file.")

data_dir = Path(data_dir)
if not data_dir.exists() or not data_dir.is_dir():
    raise NotADirectoryError(f"{data_dir} is not a valid directory")

for csv_path in data_dir.glob("*.csv"):
    print(f"\nProcessing: {csv_path.name}")

    df = pd.read_csv(csv_path)

    # Choose text column
    if "body" in df.columns:
        text_col = "body"
    elif "text" in df.columns:
        text_col = "text"
    else:
        print("Skipped (no 'body' or 'text' column)")
        continue

    mask = df[text_col].astype(str).str.strip().eq("")
    print("Empty/whitespace rows:", mask.sum())

    before = len(df)

    df = df[df[text_col].notna() & df[text_col].astype(str).str.strip().ne("")].reset_index(
        drop=True
    )

    after = len(df)
    print(f"Removed {before - after} rows")

    df.to_csv(csv_path, index=False)

print("\nDone processing all CSV files.")
