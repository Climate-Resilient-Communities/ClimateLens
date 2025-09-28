import os
import pandas as pd
import json
import csv
from pathlib import Path

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

### Split Cleaned CSV Into 32 Chunks

n_chunks = 32
chunk_size = len(df_full) // n_chunks + (len(df_full)%n_chunks > 0)

output_folder = "..."
os.makedirs(output_folder, exist_ok=True)

for i in range(n_chunks):
    start=i*chunk_size
    end=start+chunk_size
    chunk= df_full.iloc[start:end]

    if chunk.empty:
        break

    chunk_file = f"{output_folder}/climate_twitter_clean_{i+1}.csv"
    chunk.to_csv(chunk_file, index=False)
    print(f"Saved {len(chunk)} rows to {chunk_file}")

chunk1= Path("...")
cleaned_file_size_bytes = os.path.getsize(chunk1)
print(f"The size of '{chunk1}': {cleaned_file_size_bytes / (1024 * 1024):.2f} MB")

### Creating sample dataset

first_chunk_path = "..."
df_first_chunk = pd.read_csv(first_chunk_path)

df_sample = df_first_chunk.head(2736)

sample_path = "..."
df_sample.to_csv(sample_path, index=False)

print(f"Sample dataset created: {sample_path}")