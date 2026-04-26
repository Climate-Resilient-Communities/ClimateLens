"""
Data Preprocessing Pipeline for Climate NLP Project.

This module provides text cleaning and preprocessing for Twitter and Reddit
datasets. It reads from ``runtime.data_path`` (read-only) and writes cleaned
copies to ``runtime.processed_data_path`` - the raw inputs are never modified.

Usage::

    python src/data_preprocessing.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv("data_path")
if not data_path:
    raise EnvironmentError("data_path must be set in the .env file.")
if not data_path.exists() or not data_path.is_dir():
        raise NotADirectoryError(f"{data_path} is not a valid directory")
data_path = Path(data_path)

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Add src/ to sys.path so utils imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.datasets import DatasetSpec, discover_datasets  # noqa: E402
from utils.io_helpers import drop_missing_text, safe_write_csv  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402

# from climatelens.preprocessing import datasets
from utils.process_datasets import process_datasets
from utils.runtime import load_runtime  # noqa: E402

_, _, datasets = process_datasets()

log = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

SWEAR_VARIANTS = [
    "fuck",
    "fucking",
    "fucked",
    "fuckin",
    "fck",
    "f*ck",
    "f@ck",
    "shit",
    "shitty",
    "shitshow",
    "bullshit",
    "bs",
    "sh*t",
    "ass",
    "asshole",
    "a**",
    "arse",
    "bitch",
    "b*tch",
    "damn",
    "d*mn",
    "crap",
    "dick",
    "pussy",
    "piss",
    "prick",
    "whore",
    "slut",
    "cunt",
    "mf",
    "motherfucker",
]

ADDITIONAL_STOPWORDS = [
    "rt",
    "tweet",
    "repost",
    "replied",
    "comments",
    "comment",
    "upvote",
    "downvote",
    "subreddit",
    "thread",
    "user",
    "followers",
    "post",
    "share",
    "like",
    "reply",
    "hashtag",
    "hashtags",
    "link",
    "bio",
    "mention",
    "tagged",
    "followed",
    "following",
    "message",
    "profile",
    "climate",
    "change",
    "global",
    "warming",
    "yes",
    "great",
    "of",
    "love",
    "great",
    "thank",
    "you",
    "good",
    "like",
    "go",
    # Twitter-specific artifacts that slip through
    "https",
    "http",
    "co",
    "amp",
    "t",
    "www",
    "url",
    "pic",
    "twitter",
    "com",
]

# Words to preserve (negations, modals, interrogatives)
PRESERVE_WORDS = {
    "not",
    "no",
    "nor",
    "should",
    "could",
    "would",
    "must",
    "might",
    "may",
    "don't",
    "do",
    "does",
    "did",
    "why",
    "what",
    "how",
    "if",
    "that",
    "this",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "it",
}

MIN_DOCUMENT_WORDS = 3


# =============================================================================
# TEXT CLEANING
# =============================================================================


def build_custom_stopwords() -> set:
    """Combine NLTK stopwords with project-specific additions."""
    stop_words = set(stopwords.words("english"))
    combined = stop_words.union(SWEAR_VARIANTS).union(ADDITIONAL_STOPWORDS)
    return combined - PRESERVE_WORDS


def remove_consecutive_repeats(tokens: List[str]) -> List[str]:
    """Drop consecutive duplicate tokens (keeps the first occurrence)."""
    if not tokens:
        return tokens
    cleaned = [tokens[0]]
    for i in range(1, len(tokens)):
        if tokens[i] != tokens[i - 1]:
            cleaned.append(tokens[i])
    return cleaned


def highlight_issues(text: str):
    """Debug helper: return (repeated_words, slang_terms) found in *text*."""
    lowered = text.lower()
    repeated = re.findall(r"\b(\w+)\s+\1\b", lowered)
    slang = [word for word in SWEAR_VARIANTS if word in lowered]
    return repeated, slang


_URL_PATTERNS = [
    re.compile(r"http[s]?://\S+"),
    re.compile(r"t\.co/\S+"),
    re.compile(r"www\.\S+"),
]
_HANDLE_PATTERN = re.compile(r"@\w+")
_RT_PATTERN = re.compile(r"\bRT\b|\brt\b", flags=re.IGNORECASE)
_HTML_ENTITY_PATTERN = re.compile(r"&\w+;")
_URL_FRAGMENT_PATTERN = re.compile(r"\b(https?|co|www|amp|pic)\b", flags=re.IGNORECASE)


def preprocess_text(text: str, custom_stopwords: set) -> str:
    """
    Clean and preprocess a single text for downstream topic/emotion modeling.

    Applies Twitter-specific cleaning (URLs, handles, RT markers) followed by
    tokenization, stopword removal, and consecutive-duplicate removal.
    """
    for pattern in _URL_PATTERNS:
        text = pattern.sub("", text)
    text = _HANDLE_PATTERN.sub("", text)
    text = _RT_PATTERN.sub("", text)
    text = _HTML_ENTITY_PATTERN.sub("", text)
    text = _URL_FRAGMENT_PATTERN.sub("", text)

    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in custom_stopwords]
    tokens = remove_consecutive_repeats(tokens)
    return " ".join(tokens)

def remove_empty_posts():
    for csv_path in data_path.glob("*.csv"):
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

# =============================================================================
# PIPELINE
# =============================================================================


def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already present."""
    log.info("ensuring NLTK resources are available")
    for resource in ("stopwords", "punkt_tab", "punkt"):
        nltk.download(resource, quiet=True)


def preprocess_dataset(
    spec: DatasetSpec,
    custom_stopwords: set,
    processed_dir: Path,
    *,
    min_words: int = MIN_DOCUMENT_WORDS,
) -> Path:
    """
    Clean a single dataset and write the result to ``processed_dir``.

    Returns the path written. The original file in ``spec.path`` is never
    modified.
    """
    log.info("processing %s (%s)", spec.name, spec.path.name)
    df = pd.read_csv(spec.path)

    if spec.text_column not in df.columns:
        log.warning(
            "skipping %s: text column %r missing (have %s)",
            spec.name,
            spec.text_column,
            list(df.columns),
        )
        return spec.processed_path(processed_dir)

    df = drop_missing_text(df, spec.text_column)

    df[spec.cleaned_text_column] = (
        df[spec.text_column].astype(str).apply(lambda x: preprocess_text(x, custom_stopwords))
    )

    before = len(df)
    df = df[df[spec.cleaned_text_column].str.split().str.len() >= min_words]
    after = len(df)
    if after < before:
        log.info("dropped %d short docs (< %d words)", before - after, min_words)

    out = spec.processed_path(processed_dir)
    safe_write_csv(df, out)
    log.info("wrote %s (%d rows)", out, len(df))
    return out


def run_pipeline(
    data_path: Path,
    processed_dir: Path,
    *,
    registry_path: Optional[Path] = None,
    specs: Optional[Iterable[DatasetSpec]] = None,
) -> List[Path]:
    """
    End-to-end: discover datasets, clean them, and write results.

    ``specs`` lets tests inject a pre-built list; when ``None`` we read the
    YAML registry.
    """
    _ensure_nltk_resources()
    custom_stopwords = build_custom_stopwords()

    dataset_specs = (
        list(specs) if specs is not None else discover_datasets(data_path, registry_path)
    )
    if not dataset_specs:
        log.warning("no datasets matched the registry in %s", data_path)
        return []

    if not datasets:
        print(f"No datasets found in {data_path}")
        return

    print("\nCollected Datasets:")
    for key, value in datasets.items():
        print(f"  {key}: {value}")

    # Load datasets
    dfs = process_datasets(datasets)  # check if these are equivalent
    # dfs = loading_datasets(datasets)
    print(f"\n{len(dfs)} dataframes loaded successfully\n")

    # Process each dataset
    for name, df in dfs.items():
        print(f"Processing dataset: {name}")

        # Identify text column
        text_col = "body" if "body" in df.columns else "text"

        # Preview issues (optional debugging)
        # samples = []
        # for idx, row in df.head(10).iterrows():
        #     text = str(row[text_col])
        #     repeated, slang = highlight_issues(text)
        #     samples.append({
        #         "original_text": text,
        #         "repeated_words": repeated,
        #         "slang_terms": slang
        #     })
        # peek_df = pd.DataFrame(samples)
        # print(peek_df)

        # Apply preprocessing
        df["cleaned_text"] = (
            df[text_col].astype(str).apply(lambda x: preprocess_text(x, custom_stopwords))
        )

        # Filter out documents that are too short (< 3 words)
        df = df[df["cleaned_text"].str.split().str.len() >= 3]

        # Save cleaned dataset
        df.to_csv(datasets[name], index=False)
        print(f"{name} cleaning complete! ({len(df)} documents retained)\n")

    log.info("preprocessing %d datasets", len(dataset_specs))
    written: List[Path] = []
    for spec in dataset_specs:
        try:
            written.append(preprocess_dataset(spec, custom_stopwords, processed_dir))
        except Exception:
            log.exception("preprocessing failed for %s", spec.name)
    return written


def main() -> int:
    runtime = load_runtime()
    log.info("data preprocessing starting")
    log.info("paths:\n%s", runtime.describe())

    written = run_pipeline(runtime.data_path, runtime.processed_data_path)
    if not written:
        log.error("no datasets were processed")
        return 1
    log.info("preprocessing complete; wrote %d files", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
