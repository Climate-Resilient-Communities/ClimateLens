# Emotion Visualization Pipeline

This script generates visualizations from Twitter and Reddit datasets for climate anxiety analysis.

## What It Generates

- **Word clouds (PNG):** Word clouds for each detected emotion and one combined word cloud per dataset.
- **Emotion time series (HTML):** Interactive charts showing emotion trends over time:
  - Percentage of posts for each emotion.
  - Stacked counts of posts for each emotion.
- **Summary report (HTML):** Overall post counts and emotion distributions for each dataset.

## Requirements

The pipeline expects CSV files for Twitter and Reddit. Required columns:

- **Twitter:** `created_at`, `text`
- **Reddit:** `body`, `created_utc`

If the CSVs already include an `emotion_label` column, those labels are reused. Otherwise, the script classifies text using the Hugging Face model `j-hartmann/emotion-english-distilroberta-base`.

## Running the Script

Run from the project environment:

```bash
python emotion_visualizations.py
```

## Input and Output Directories

The script uses the project's runtime configuration to determine paths. By default, the output directory is `./visualizations`, with files organized as follows:

```text
visualizations/
└── emotions/
    ├── wordclouds/
    │   ├── twitter_*_wordcloud.png
    │   └── reddit_*_wordcloud.png
    ├── timeseries/
    │   ├── twitter_emotion_timeseries.html
    │   ├── twitter_emotion_timeseries_stacked.html
    │   ├── reddit_emotion_timeseries.html
    │   └── reddit_emotion_timeseries_stacked.html
    └── emotion_analysis_summary.html
```

The input directory is selected by the project's runtime configuration. The pipeline prefers the emotion-classified output directory and falls back to the configured data directory if no CSV files are found.

## Emotions

The model identifies seven emotions:

`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, and `surprise`.

## Notes

- Twitter data is grouped by hour for time-series charts.
- Reddit data is grouped by month.
- The HTML time-series charts are interactive and can be opened in a web browser.
- If required CSV files or columns are missing, the pipeline reports an error and exits.