import csv
import json
import os
from pathlib import Path

import pandas as pd

input_path = Path("...")
output_path = Path("...")

# Collect just the first 10 lines to preview
preview_rows = []
with open(input_path, 'r', encoding='utf-8') as f:
    for _ in range(100):
        try:
            preview_rows.append(json.loads(f.readline()))
        except:
            continue

df = pd.DataFrame(preview_rows)
#df.head() # uncomment if working in notebook

print("Preview columns:", df.columns.tolist())

# Keep only the columns we care about
desired_columns = ['created_at', 'text']
df_clean = df[desired_columns].copy()
df_clean.sample(5)

df_clean.info()

with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=desired_columns)
    writer.writeheader()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                writer.writerow({
                    'created_at': data.get('created_at', ''),
                    'text': data.get('text', '')
                })
            except Exception:
                continue  # skip malformed lines

df_full = pd.read_csv(output_path)
df_full.info()

df.dropna(inplace=True) #remove in memory
df.to_csv("climate_twitter_clean.csv", index=False) #remove in file