# ClimateLens Preprocessing Pipeline

This document describes the complete preprocessing workflow applied to Reddit and Twitter datasets before analysis. The pipeline consists of extraction, cleaning, and text normalization.

The final output consists of standardized CSV datasets ready for NLP analysis.

# Reddit Processing
+ Load raw Reddit JSONL files containing comments or submissions.
+ Filter records using the climate-related keyword list.
+ Extract `subreddit`, `body`, and `created_utc`.
+ Remove missing values using `dropna(inplace=True)`.
+ Remove duplicate posts based on the body column.
+ Convert `created_utc` to `created_dt` for temporal analysis.
+ Ensure UTF-8 encoding.
+ Save standardized CSV files.


# Twitter Processing
+ Load the raw Twitter JSONL archive.
+ Extract `created_at` and `text`.
+ Remove missing values.
+ Remove duplicate tweets based on the `text` field.
+ Split the cleaned dataset into approximately 32 equally sized CSV files (optional).
+ Ensure UTF-8 encoding.
+ Save standardized CSV files.

# Text Normalization Pipeline
Each cleaned dataset receives an additional `cleaned_text` column used exclusively for topic modeling; the original text remains unchanged. The preprocessing code automatically detects the available text column (`body` for Reddit or `text` for Twitter), and datasets are collected into dictionaries keyed by dataset name.

# Custom Stopword Construction
The default NLTK English stopword list is extended with three groups:
- Profanities and toxic filler words
- Platform-specific boilerplate: rt, repost, upvote, tweet, hashtag.
- Generic high-frequency fillers: love, great, good, thank, like.

These additions reduce sentiment-heavy noise that contributes little to topic discovery.

# Preserved Words
Words normally removed by standard stopword lists are intentionally retained when they carry semantic meaning, including negations (not, no, nor, don't), modal verbs (should, could, would, might, must), and interrogatives (why, how, what, if).

# Tokenization Rules

Each document undergoes the following preprocessing steps.

1. Convert to lowercase.
2. Tokenization.
3. Remove punctuation.
4. Remove URLs.
5. Remove numbers.
6. Keep alphabetic tokens only.
7. Remove custom stopwords.
8. Remove consecutive duplicate words.

Example: `very very hot` becomes `very hot`.

# Outputs
The final standardized CSV datasets contain the original text (`body` for Reddit or `text` for Twitter) alongside `cleaned_text`, which is used for downstream NLP and topic modeling.