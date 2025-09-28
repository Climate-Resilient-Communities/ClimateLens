import os
import pandas as pd
import json
import csv
from pathlib import Path

search_terms = [
  "climate change", "global warming",
  "eco-anxiety", "climate anxiety", "eco-distress",
  "eco-depression", "climate depression", "climate distress",
  "climate worry", "climate fear", "climate doom",
  "eco-grief", "ecological grief", "climate grief", "solastalgia",
  "environmental melancholia",
  "eco-anger", "eco-frustration", "eco-guilt",
  "collective guilt", "powerlessness", "helplessness",
  "despair", "eco-paralysis", "ecophobia",
  "post-traumatic stress", "PTSD"
]

# === CONFIGURATION ===
input_file = Path("...")  # Change to another file when done
output_file = Path("...")

def contains_keywords(text, keywords):
    """Checks if any keyword appears in the given text."""
    if not text:
        return False
    text = text.lower()
    return any(term in text for term in keywords)

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        try:
            entry = json.loads(line)
            print(json.dumps(entry, indent=2))  # Pretty-print for inspection
        except json.JSONDecodeError:
            print("Skipping malformed line")

with open(output_file, "w", newline='', encoding="utf-8") as csv_out:
    writer = csv.DictWriter(csv_out, fieldnames=["subreddit", "body", "created_utc"])
    writer.writeheader()

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                body = entry.get("body", "")
                if contains_keywords(body, search_terms):
                    writer.writerow({
                        "subreddit": entry.get("subreddit"),
                        "body": body,
                        "created_utc": entry.get("created_utc")
                    })
            except json.JSONDecodeError:
                continue  # skip malformed lines

df = pd.read_csv("...")

total_rows = len(df)
print("Total rows:", total_rows)

df.head()

input_folder = Path("...")
output_folder = Path("...")
output_folder.mkdir(exist_ok=True)

# Iterating over all .jsonl files in input folder
for file in os.listdir(input_folder):
    if not file.endswith(".jsonl"):
        continue

    input_path = input_folder / file

    # Peek at first valid line
    with open(input_path, "r", encoding="utf-8") as f:
        first_valid_line = None
        for line in f:
            try:
                first_valid_line = json.loads(line)
                break
            except:
                continue

    if not first_valid_line:
        print(f"Skipping unreadable or empty file: {file}")
        continue

    is_comment = "body" in first_valid_line
    type_tag = "comments" if is_comment else "submissions"
    subreddit = first_valid_line.get("subreddit", "unknown").lower()

    name_prefix = f"filtered_{subreddit}_{type_tag}.csv"
    output_path = output_folder / name_prefix

    print(f"Processing {file} → {output_path.name}")

    match_count = 0
    with open(output_path, "w", newline='', encoding='utf-8') as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=["subreddit", "body", "created_utc"])
        writer.writeheader()

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    text = entry.get("body") if is_comment else entry.get("selftext") or entry.get("title")
                    if contains_keywords(text, search_terms):
                        writer.writerow({
                            "subreddit": entry.get("subreddit"),
                            "body": text,
                            "created_utc": entry.get("created_utc")
                        })
                        match_count += 1
                except json.JSONDecodeError:
                    continue

    if match_count == 0:
        print(f"No matches found in {file}")
    else:
        print(f"{match_count} matches written to {output_path.name}")