# ClimateLens Data Governance

This document defines versioning, storage, and data management practices for ClimateLens datasets.

# Data Lifecycle

The project follows the pipeline below:

```
Raw JSONL
      ↓
Keyword Filtering
      ↓
Schema Extraction
      ↓
Cleaning
      ↓
Text Normalization
      ↓
CSV Datasets
      ↓
Analysis & Modeling
```

Raw datasets are never modified.

All preprocessing steps are reproducible from the original JSONL files.

# Data Storage & Usage
Raw datasets are stored as JSONL files, while processed and derived datasets are stored as CSV files. Derived datasets may include generated fields such as `cleaned_text` and `created_dt`.

For topic modeling, `cleaned_text` is used as the primary input. Dynamic Topic Modeling additionally uses `created_dt` to group documents into time bins, typically by month. Emotion analysis uses Reddit `body` and Twitter `text`, with resulting emotion labels (disappoinment, neutral, excitement, etc.) stored in the derived datasets.

These datasets support topic discovery, topic evolution over time, cross-platform comparisons, and emotion trend analysis.

# Versioning

The current schema version is **v1.1**, released in **August 2026**.

# Update Policy

Any modification to the dataset structure should be documented.

Any change to the dataset structure or preprocessing process must be documented, including adding, removing, or renaming fields, changing preprocessing behavior, or modifying text-normalization rules. Each structural or processing change should increment the documented schema version and be reflected throughout the project documentation.