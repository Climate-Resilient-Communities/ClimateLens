# ClimateLens Data Schema

This document defines the structure of the ClimateLens datasets after extraction from Reddit and Twitter. It serves as the authoritative reference for all dataset fields used throughout the project.

---

# Sources

ClimateLens currently incorporates two public social media sources.

- Reddit
    - Comments and submissions collected from selected subreddits.
    - Filtered using a climate-related keyword list.
- Twitter
    - Tweets containing climate-related terms.
    - Extracted from publicly available Twitter archives.

All downstream preprocessing and modeling assumes these schemas.

---

# Reddit Schema

The Reddit datasets intentionally contain only a small subset of the original Pushshift metadata.

| Field | Type | Description | Example |
|--------|------|-------------|---------|
| subreddit | string | Community where the comment or submission was posted | Anticonsumption |
| body | string | Text content of the comment or submission | "From the website: Buy Nothing Day is just a few days away..." |
| created_utc | integer | Unix timestamp (UTC) | 1259252332 |

## Notes

### created_utc

The stored timestamp is a Unix UTC timestamp.

For temporal analyses (such as Dynamic Topic Modeling), this timestamp should later be converted into a datetime column: `created_dt`.

This conversion is not required during initial preprocessing, but is expected before any time-series analysis.

### Dataset Type

The `subreddit` field identifies only the subreddit.

Whether a record originated from:

- comments
- submissions

is determined by the dataset filename rather than an explicit column.

Example:

```
climate_comments.csv
climate_submissions.csv
```

---

# Twitter Schema

Although the original Twitter archive contains 36 variables, nearly all are unused.

Many metadata fields contain almost entirely missing values. For example:

- geo
- coordinates

contain only 47 non-null values within the first one million tweets, making geospatial analysis infeasible.

Only the following fields are retained.

| Field | Type | Description | Example |
|--------|------|-------------|---------|
| created_at | string (RFC 2822) | Tweet creation timestamp | Tue Sep 07 18:41:48 +0000 2021 |
| text | string | Raw tweet text | "And how does the change in climate..." |

---

# Derived Fields

The preprocessing pipeline creates additional fields that are not present in the raw data.

## created_dt

Generated from:

- Reddit: `created_utc`
- Twitter: `created_at`

Purpose:

- Monthly grouping
- Yearly grouping
- Dynamic Topic Modeling

## cleaned_text

Generated after text normalization.

Purpose:

- Topic Modeling
- Dynamic Topic Modeling
- Topic Merging

The original text fields (`body` and `text`) are always preserved. The emotion analysis uses the original text to account for the proper wording.