import csv
import json
import os
from pathlib import Path
import pandas as pd

first_chunk_path = "..."
df_first_chunk = pd.read_csv(first_chunk_path)

df_sample = df_first_chunk.head(2736)

sample_path = "..."
df_sample.to_csv(sample_path, index=False)

print(f"Sample dataset created: {sample_path}")