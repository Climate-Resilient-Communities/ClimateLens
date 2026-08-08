# ClimateLens Dataset Catalog

This document describes every dataset included in ClimateLens.

---

# Reddit Dataset Organization

Reddit datasets are already partitioned by subreddit & content type (comments or submissions).

Total Reddit datasets:

```
32
```

(16 subreddits × 2 content types)

Largest dataset:

```
collapse_comments.csv
```

113,656 posts

Smallest dataset:

```
anxietydepression_submissions.csv
```

87 posts

# Twitter Dataset Organization

The original cleaned Twitter dataset is approximately `5 GB`

After preprocessing, it is split into approximately:

```
32 CSV files
```

Each chunk is roughly:

```
103 MB
```

This significantly improves processing speed and memory usage.

---

# Dataset Statistics

| File                        | Posts   | Size (MB) | Avg Post Characters | Avg Post Words | Max Post Characters |
|-----------------------------|--------|-----------|---------------------|----------------|---------------------|
| anticonsumption_comments     | 2714   | 3.25      | 743.56              | 122.97         | 9729                |
| anticonsumption_submissions  | 165    | 0.25      | 930.01              | 155.02         | 17943               |
| anxietydepression_comments   | 179    | 0.36      | 1321.99             | 208.05         | 9314                |
| anxietydepression_submissions| 87     | 0.28      | 2050.29             | 360.99         | 12123               |
| anxietyhelp_comments         | 93     | 0.16      | 1100.27             | 192.56         | 7212                |
| anxietyhelp_submissions      | 93     | 0.25      | 1704.91             | 307.03         | 15073               |
| anxiety_comments             | 2579   | 4.31      | 1050.55             | 188.46         | 9972                |
| anxiety_submissions          | 1502   | 4.75      | 1997.95             | 369.36         | 19841               |
| climatechange_comments       | 24882  | 35.65     | 911.88              | 146.28         | 10049               |
| climatechange_submissions    | 6579   | 4.01      | 378.46              | 61.64          | 33529               |
| climateoffensive_comments    | 5699   | 7.96      | 923.09              | 132.17         | 10067               |
| climateoffensive_submissions | 1478   | 1.82      | 796.87              | 119.56         | 15685               |
| climate_comments             | 25325  | 30.59     | 782.25              | 118.95         | 12850               |
| climate_submissions          | 20972  | 6.52      | 188.38              | 29.7           | 35624               |
| collapse_comments            | 112904 | 138.82    | 773.10              | 127.72         | 19700               |
| collapse_submissions         | 9281   | 15.72     | 1095.55             | 165.57         | 39838               |
| depression_comments          | 10373  | 16.08     | 967.83              | 177.02         | 9990                |
| depression_submissions       | 8215   | 29.76     | 2288.76             | 430.19         | 36068               |
| environment_comments         | 88156  | 91.82     | 658.69              | 105.24         | 11442               |
| environment_submissions      | 34189  | 10.87     | 190.00              | 29.64          | 33715               |
| getting_over_it_comments     | 211    | 0.52      | 1541.80             | 281.12         | 9426                |
| getting_over_it_submissions  | 115    | 0.48      | 2651.44             | 485.81         | 12447               |
| lostgeneration_comments      | 8497   | 9.91      | 723.86              | 123.14         | 9884                |
| lostgeneration_submissions   | 481    | 0.89      | 1159.42             | 197.18         | 37874               |
| mentalhealth_comments        | 2044   | 4.82      | 1550.14             | 239.26         | 10360               |
| mentalhealth_submissions     | 1616   | 6.89      | 2688.70             | 493.53         | 36807               |
| offmychest_comments          | 5320   | 7.70      | 905.71              | 164.60         | 9874                |
| offmychest_submissions       | 4354   | 21.79     | 3172.06             | 597.88         | 39445               |
| sustainability_comments      | 3045   | 4.58      | 955.35              | 149.48         | 9999                |
| sustainability_submissions   | 889    | 1.05      | 782.39              | 106.92         | 39581               |
| teenagers_comments           | 20642  | 15.47     | 467.08              | 80.62          | 10024               |
| teenagers_submissions        | 3026   | 9.04      | 1891.03             | 339.73         | 40000               |
| twitter_climate_clean        | 19803553| 4424.93   | 130.49              | 19.97          | 388                 |
| twitter_sample               | 100000 | 22.37     | 130.51              | 19.99          | 162                 |
| twitter_tiny_sample          | 2736   | 1.21      | 121.49              | 17.83          | 151                 |