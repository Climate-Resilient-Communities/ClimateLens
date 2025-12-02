import csv
import json
from pathlib import Path
import pandas as pd

input_path = Path("...") # raw NDJSON Twitter file
output_path = Path("...") # cleaned CSV path

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