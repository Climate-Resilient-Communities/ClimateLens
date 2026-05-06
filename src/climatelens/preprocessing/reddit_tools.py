import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
data_dir: Optional[str] = os.getenv("DATA_DIR")
reddit_raw_dir: Optional[str] = os.getenv("REDDIT_RAW_DIR")

search_terms = [
    "climate change",
    "global warming",
    "eco-anxiety",
    "climate anxiety",
    "eco-distress",
    "eco-depression",
    "climate depression",
    "climate distress",
    "climate worry",
    "climate fear",
    "climate doom",
    "eco-grief",
    "ecological grief",
    "climate grief",
    "solastalgia",
    "environmental melancholia",
    "eco-anger",
    "eco-frustration",
    "eco-guilt",
    "collective guilt",
    "powerlessness",
    "helplessness",
    "despair",
    "eco-paralysis",
    "ecophobia",
    "post-traumatic stress",
    "PTSD",
]


def contains_keywords(text: Optional[str], keywords: List[str]) -> bool:
    """Checks if any keyword appears in the given text.

    Each keyword is tried as a case-insensitive regex pattern. If a keyword
    is not a valid regex, plain case-insensitive substring matching is used
    as a fallback, preserving backward compatibility with simple string terms.
    """
    if not text:
        return False
    lower: str = text.lower()
    for term in keywords:
        try:
            if re.search(term, text, re.IGNORECASE):
                return True
        except re.error:
            if term.lower() in lower:
                return True
    return False


def peek_first_valid_line(file_path: Path) -> Optional[Dict[str, Any]]:
    """Read first valid JSON line from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def determine_file_type(first_entry: Dict[str, Any]) -> tuple[bool, str]:
    """Determine if file contains comments or submissions."""
    is_comment: bool = "body" in first_entry
    type_tag: str = "comments" if is_comment else "submissions"
    return is_comment, type_tag


def extract_text_from_entry(entry: Dict[str, Any], is_comment: bool) -> Optional[str]:
    """Extract text content from a Reddit entry."""
    if is_comment:
        return entry.get("body")
    else:
        return entry.get("selftext") or entry.get("title")


def process_reddit_file(
    input_path: Path, output_path: Path, is_comment: bool, search_terms: List[str]
) -> int:
    """Process a single Reddit JSONL file and return match count."""
    match_count: int = 0

    with open(output_path, "w", newline="", encoding="utf-8") as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=["subreddit", "body", "created_utc"])
        writer.writeheader()

        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=input_path.name, unit=" lines", leave=False):
                try:
                    entry: Dict[str, Any] = json.loads(line)
                    text: Optional[str] = extract_text_from_entry(entry, is_comment)

                    if contains_keywords(text, search_terms):
                        writer.writerow(
                            {
                                "subreddit": entry.get("subreddit"),
                                "body": text,
                                "created_utc": entry.get("created_utc"),
                            }
                        )
                        match_count += 1
                except json.JSONDecodeError:
                    continue

    return match_count


def process_all_reddit_files(
    input_folder: Path, output_folder: Path, search_terms: List[str]
) -> None:
    """Batch process all JSONL files in a folder."""
    output_folder.mkdir(exist_ok=True)

    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".jsonl")]
    for file in tqdm(jsonl_files, desc="Processing Reddit files", unit="file"):
        input_path: Path = input_folder / file

        first_valid_line: Optional[Dict[str, Any]] = peek_first_valid_line(input_path)

        if not first_valid_line:
            print(f"Skipping unreadable or empty file: {file}")
            continue

        is_comment: bool
        type_tag: str
        is_comment, type_tag = determine_file_type(first_valid_line)

        subreddit: str = first_valid_line.get("subreddit", "unknown").lower()
        name_prefix: str = f"filtered_{subreddit}_{type_tag}.csv"
        output_path: Path = output_folder / name_prefix

        print(f"Processing {file} → {output_path.name}")

        match_count: int = process_reddit_file(input_path, output_path, is_comment, search_terms)

        if match_count == 0:
            print(f"No matches found in {file}")
        else:
            print(f"{match_count} matches written to {output_path.name}")


def main() -> None:
    """Main entry point for Reddit data processing."""
    if not data_dir or not reddit_raw_dir:
        print("Error: DATA_DIR and REDDIT_RAW_DIR must be set in .env file")
        return

    input_folder: Path = Path(reddit_raw_dir)
    output_folder: Path = Path(data_dir)

    if not input_folder.exists():
        print(f"Error: Input folder {input_folder} does not exist")
        return

    process_all_reddit_files(input_folder, output_folder, search_terms)


if __name__ == "__main__":
    main()
