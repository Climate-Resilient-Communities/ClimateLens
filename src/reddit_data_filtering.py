import csv
import json
import os
from pathlib import Path
from dotenv import load_dotenv

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

def contains_keywords(text, keywords):
    """Checks if any keyword appears in the given text."""
    if not text:
        return False
    lower = text.lower()
    return any(term in lower for term in keywords)

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
    data_dir, reddit_raw_dir = os.getenv("DATA_DIR"), os.getenv("REDDIT_RAW_DIR")
  else:
    raise FileNotFoundError(f".env file not found at {env_path}")

  return data_dir, reddit_raw_dir

data_dir, reddit_raw_dir = load_environment()
if not data_dir or not reddit_raw_dir:
    raise EnvironmentError("DATA_DIR and CODE_DIR must be set in the .env file.")

### Batch process folder of JSONL files
input_folder = reddit_raw_dir
output_folder = data_dir
output_folder.mkdir(exist_ok=True)

# Iterating over all .jsonl files in input folder
# this for loop works, but can be improved for logic and readability
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

    # Determine file type
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