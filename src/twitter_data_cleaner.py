import csv
import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

def load_environment():
  try:
    import google.colab
    from google.colab import drive
    drive.mount("/content/drive")

    base_path = "..."
    env_path = Path(base_path) / ".env"
  except ImportError:
    env_path = Path(__file__).resolve().parent / ".env"

  if env_path.exists():
    load_dotenv(env_path)
    print("Loading environment variables")
    data_dir, twitter_raw_dir = os.getenv("DATA_DIR"), os.getenv("TWITTER_RAW_DIR")
  else:
    raise FileNotFoundError(f".env file not found at {env_path}")

  return data_dir, twitter_raw_dir

data_dir, twitter_raw_dir = load_environment()
if not data_dir or not twitter_raw_dir:
    raise EnvironmentError("DATA_DIR and TWITTER_RAW_DIR must be set in the .env file.")

input_path = Path(twitter_raw_dir) / "climate.jsonl" # raw NDJSON Twitter file
output_path = Path(data_dir) / "twitter_climate_clean.csv" # cleaned CSV path

# preview first 100 lines
preview_rows = []
with open(input_path, "r", encoding="utf-8") as f:
    for _ in range(100):
        try:
            preview_rows.append(json.loads(f.readline()))
        except Exception:
            continue

df = pd.DataFrame(preview_rows)
#df.head() # uncomment if working in notebook

print("Preview columns:", df.columns.tolist())

desired_columns = ["created_at", "text"]

df_clean = df[desired_columns].copy()
print(df_clean.sample(5)) #bc it's .py file
df_clean.info()

with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=desired_columns)
    writer.writeheader()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                writer.writerow({
                    "created_at": data.get("created_at", ""),
                    "text": data.get("text", "")
                })
            except Exception:
                continue

df_full = pd.read_csv(output_path)
print(f"Twitter dataframe information:/n{df_full.info()}")