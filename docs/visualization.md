**Topics \> Intertopic Map**

**What is the visualization showing (data, time range, key
variables)?**\
Each topic has been represented by a bubble. The size of the bubbles
shows the intensity(how many times it has been used) of the topic. The
distance between two topics shows how closely the topics are related.

**Why is this visualization useful / what insight does it unlock?\**
It tells the users the topics that are primarily used and its intensity.
Also the similarity between those topics.

**When or by whom it would typically be used (e.g., leadership, ops
team, trend tracker)?\**
Not fully fleshed out yet, but mental health professionals, researchers.

**Comments on the current visualization:**

1.  The bubble chart works well for this purpose. It just needs some
    refinement.

2.  The bubbles don't have a threshold for very small sizes. As a
    result, the target size becomes difficult to tap.

3.  The hover window is not readable. Hove doesn't work on touch
    devices, so it's not recommended.

**Suggestions/Proposal:**

*See a similar example:
[[https://www.highcharts.com/demo/highcharts/bubble]{.underline}](https://www.highcharts.com/demo/highcharts/bubble)*

**Visualization:**

![](media/image1.png){width="6.5in" height="3.0555555555555554in"}

- The size of a topic bubble is based on how many times the keywords
  under that topic have been used

- On clicking one bubble, the right column with the topic name and list
  of keywords for that topic opens up.

- Hover is only active on Desktop

**Questions**

- The distance shows how similar or different the topics are; is that
  difference denoted by a number? Is there a scale on which this
  comparison lies?

- Topic 1: 20

  - Topic 2: 10; will topic 1 be double topic 2's size? *The size of the
    bubble is proportional to the size / number of times it appears*

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