import csv
import json
import os
from pathlib import Path
import pandas as pd

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