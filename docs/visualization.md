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

**Topics \> Barchart**

**What is the visualization showing (data, time range, key
variables)?**\
It's showing the ranking of topics based on how much people are talking
about it.

**Why is this visualization useful / what insight does it unlock?\**
It shows the number and frequency of keywords that appear under a
topic + it allows for the user to compare different keywords.

**When or by whom it would typically be used (e.g., leadership, ops
team, trend tracker)?\**
Not fully fleshed out yet, but mental health professionals, researchers.

**Comments on the current visualization:**

1.  Showing all the bar charts at once is causing information overload.

2.  X-axes have different ranges. So we cannot compare the bars on one
    bar chart with the other.

**Suggestions/Proposal:**

1.  Show one topic bar chart at once.

2.  Keep the X-axes range the same to make it comparable

**Visualization:**

![](media/image2.png){width="5.864583333333333in" height="6.40625in"}

**Questions**

- What unit is used for the number of keywords?

  - The unit apparently changes in each topic, it should be consistent.

- How about we show comparisons between two (or more) topics, to see how
  many times similar keywords appear in different topics?

- How are these being ranked?

  - To rank, would it make sense for the bars to be sorted in a
    descending order?

**Topics \> Hierarchy**

**What is the visualization showing (data, time range, key
variables)?**\
This chart shows the hierarchical structure of topics. [Broad categories
are broken down into specific sub-themes.]{.mark}

**Why is this visualization useful / what insight does it unlock?\**
To be discussed with Zainab.

**When or by whom it would typically be used (e.g., leadership, ops
team, trend tracker)?\**
Not fully fleshed out yet, but mental health professionals,
researchers.\`

**Comments on the current visualization:**

1.  The hierarchy is not clear as the nodes are concentrated and
    overlapping on the chart.

2.  Topic names are not clear.

**Suggestions/Proposal:**

We can show it in the form of a Tree map with levels. This visualization
will work only if we have some value attached to each topic.

**Visualization:** *This is an indicative visualization. I am working on
a chart for Climate Lens.*

*See a similar example:
[[https://www.highcharts.com/demo/highcharts/treemap-with-levels]{.underline}](https://www.highcharts.com/demo/highcharts/treemap-with-levels)*

![](media/image3.png){width="6.5in" height="3.4444444444444446in"}

- In this treemap, Topic 9 is on the top of the Treemap. Topic 11 and
  Topic 7 are children of Topic 9. Topic 1 is child of Topic 11. Topic 6
  and Topic 2 are children of Topic 7. Topic 12 and Topic 4 are children
  of Topic 6.

- The size shows how many times they were used and also defines the area
  on the map.

**Questions**

- the

**Sentiments \> Distribution**

**Comments on the current visualization:**

The pie chart is too big to be legible on big screens.

**Suggestions/Proposal:** Keep the size of the pie chart fixed (should
be legible on mobile phones). Then it would be eligible on all devices.

**Sentiments \> Probability**

**Comments on the current visualization:** Can we show the
interpretation from the charts? E.g. "[*people may be expressing worry
without being strongly negative*"]{.mark}


The pipeline's visualization stages convert model outputs into usable datasets and human-readable artifacts.

## `emotion_visualizations.py`

The visualization stage consumes emotion-classified CSV files from:

```text
OUTPUT_DATA_DIR
```

If the input already contains:

```text
emotion_label
```

the stage reuses the existing classifications instead of running the emotion model again.

This avoids unnecessary duplicate inference.

If emotion labels are not already present, the script falls back to:

```text
j-hartmann/emotion-english-distilroberta-base
```

and performs classification before generating the visualizations.

### Outputs

The visualization stage creates:

```text
OUTPUT_VIS_DIR/
└── emotions/
    ├── wordclouds/
    └── timeseries/
```

as well as a summary HTML report.

The visualizations provide a way to inspect the distribution and evolution of detected emotions.
