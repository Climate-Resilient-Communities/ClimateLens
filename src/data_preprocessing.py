"""
Data Preprocessing Pipeline for Climate NLP Project

This module provides text cleaning and preprocessing for Twitter and Reddit datasets.
Run directly to process all datasets in the configured data directory.

Usage:
    python src/data_preprocessing.py
"""

import os
import re
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv


# =============================================================================
# CONSTANTS
# =============================================================================

SWEAR_VARIANTS = [
    'fuck', 'fucking', 'fucked', 'fuckin', 'fck', 'f*ck', 'f@ck',
    'shit', 'shitty', 'shitshow', 'bullshit', 'bs', 'sh*t',
    'ass', 'asshole', 'a**', 'arse',
    'bitch', 'b*tch',
    'damn', 'd*mn',
    'crap', 'dick', 'pussy', 'piss', 'prick',
    'whore', 'slut', 'cunt', 'mf', 'motherfucker',
]

ADDITIONAL_STOPWORDS = [
    'rt', 'tweet', 'repost', 'replied', 'comments', 'comment', 'upvote', 'downvote', 'subreddit',
    'thread', 'user', 'followers', 'post', 'share', 'like', 'reply', 'hashtag', 'hashtags', 'link',
    'bio', 'mention', 'tagged', 'followed', 'following', 'message', 'profile', 'climate', 'change',
    'global', 'warming', 'yes', 'great', 'of',
    'love', 'great', 'thank', 'you', 'good', 'like', 'go',
    # Twitter-specific artifacts that slip through
    'https', 'http', 'co', 'amp', 't', 'www', 'url', 'pic', 'twitter', 'com',
]

# Words to preserve (negations, modals, interrogatives)
PRESERVE_WORDS = {
    'not', 'no', 'nor', 'should', 'could', 'would', 'must', 'might', 'may',
    "don't", 'do', 'does', 'did', 'why', 'what', 'how', 'if', 'that', 'this',
    'i', 'you', 'we', 'they', 'he', 'she', 'it'
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def build_custom_stopwords():
    """
    Build the custom stopwords set by combining NLTK stopwords with our additions.

    Returns:
        set: Combined stopwords set with preserved words removed
    """
    stop_words = set(stopwords.words("english"))
    custom_stopwords = stop_words.union(SWEAR_VARIANTS).union(ADDITIONAL_STOPWORDS)
    custom_stopwords = custom_stopwords - PRESERVE_WORDS
    return custom_stopwords


def load_datasets(data_path, prefix, datasets):
    """
    Scan a directory for CSV files matching a prefix and add them to datasets dict.

    Args:
        data_path: Path to directory containing CSV files
        prefix: File prefix to filter (e.g., "filtered_" or "clean_")
        datasets: Dictionary to populate with {name: file_path}
    """
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)

        if os.path.isfile(file_path) and file.endswith('.csv'):
            file_name = file.replace(prefix, "").replace(".csv", "")
            datasets[file_name] = file_path


def loading_datasets(datasets):
    """
    Load CSV datasets into pandas DataFrames.

    Args:
        datasets: Dictionary of {name: file_path}

    Returns:
        dict: {name: DataFrame}
    """
    dfs = {}

    for name, path in datasets.items():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        # Identify text column
        if "body" in df.columns:
            text_col = "body"
        elif "text" in df.columns:
            text_col = "text"
        else:
            print(f"Skipping {name}. No 'body' or 'text' column.")
            continue

        print(f'Loaded {name}')
        dfs[name] = df

    return dfs


def remove_consecutive_repeats(tokens):
    """
    Remove consecutive duplicate tokens.

    Args:
        tokens: List of tokens

    Returns:
        List of tokens with consecutive duplicates removed
    """
    if not tokens:
        return tokens

    cleaned = [tokens[0]]
    for i in range(1, len(tokens)):
        if tokens[i] != tokens[i-1]:
            cleaned.append(tokens[i])
    return cleaned


def highlight_issues(text):
    """
    Identify repeated words and profanity in text (for debugging).

    Args:
        text: Input text string

    Returns:
        Tuple of (repeated_words, slang_terms)
    """
    lowered = text.lower()
    repeated = re.findall(r'\b(\w+)\s+\1\b', lowered)
    slang = [word for word in SWEAR_VARIANTS if word in lowered]
    return repeated, slang


def preprocess_text(text, custom_stopwords):
    """
    Clean and preprocess text for topic modeling.

    Applies Twitter-specific cleaning (URLs, handles, RT markers) followed by
    tokenization, stopword removal, and deduplication.

    Args:
        text: Raw text string
        custom_stopwords: Set of stopwords to remove

    Returns:
        Cleaned text string
    """
    # Apply regex cleaning BEFORE tokenization to remove Twitter artifacts

    # Remove URLs (http, https, and t.co shortened links)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r't\.co/\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove Twitter handles (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove retweet markers (RT, rt) at beginning of tweets
    text = re.sub(r'\bRT\b|\brt\b', '', text, flags=re.IGNORECASE)

    # Remove HTML entities like &amp;
    text = re.sub(r'&\w+;', '', text)

    # Remove standalone URL fragments (common leftovers: co, https, http)
    text = re.sub(r'\b(https?|co|www|amp|pic)\b', '', text, flags=re.IGNORECASE)

    # Tokenize and filter
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in custom_stopwords]
    tokens = remove_consecutive_repeats(tokens)

    return ' '.join(tokens)


def run_pipeline(data_path):
    """
    Main processing pipeline: load datasets, clean text, and save results.

    Args:
        data_path: Path to directory containing CSV files
    """
    # Download NLTK data if needed
    print("Ensuring NLTK data is available...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

    # Build stopwords set
    custom_stopwords = build_custom_stopwords()

    # Discover datasets
    datasets = {}
    load_datasets(data_path, "filtered_", datasets)
    load_datasets(data_path, "clean_", datasets)

    if not datasets:
        print(f"No datasets found in {data_path}")
        return

    print(f"\nCollected Datasets:")
    for key, value in datasets.items():
        print(f'  {key}: {value}')

    # Load datasets
    dfs = loading_datasets(datasets)
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
        df['cleaned_text'] = df[text_col].astype(str).apply(
            lambda x: preprocess_text(x, custom_stopwords)
        )

        # Filter out documents that are too short (< 3 words)
        df = df[df['cleaned_text'].str.split().str.len() >= 3]

        # Save cleaned dataset
        df.to_csv(datasets[name], index=False)
        print(f"  {name} cleaning complete! ({len(df)} documents retained)\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    data_path = os.getenv("DATA_DIR")

    if not data_path:
        print("ERROR: DATA_DIR not found in .env file")
        exit(1)

    if not os.path.exists(data_path):
        print(f"ERROR: Data directory does not exist: {data_path}")
        exit(1)

    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Data directory: {data_path}\n")

    run_pipeline(data_path)

    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)