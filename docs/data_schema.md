# **🌍 ClimateLens Data Schema**

This schema defines the structure of the **climate-related social media datasets** (Reddit \+ Twitter), serving as a single source of truth for analysis, reporting, and model development.

## **Sources**

* **Reddit**: Extracted comments and submissions containing climate-anxiety keywords.  
* **Twitter**: Extracted tweets containing climate-related terms.  
* **Scope**: Publicly available text data, preprocessed into structured CSVs containing climate-related terms.

## **Data Entities & Fields**

### **Reddit**

	There aren’t many variables in the reddit datasets to begin with, and we’re concerned with even less.

| Field | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| subreddit | string | Community where the comment/submission was posted | Anticonsumption |
| body | string | Text content of the comment or post | “From the website: Buy Nothing Day is just a few days away…” |
| created\_utc | integer | Unix timestamp (UTC) of creation | 1259252332 |

It should be noted that the UTC time in the data should be converted into datetime for grouping by month/year in dynamic topic modeling (add as an extra column), but the sample datasets don’t have that right away.  
	It should also be noted that the subreddit column doesn’t mention if the dataset is a comments or submission set. This is found within the naming of each dataset.

### **Twitter**

While this dataset has 36 columns, most serve no purpose or simply don’t have enough non-null values to have them be useful.   
For example, the two spatial variables: “geo”, “coordinates”, have only 47 non-null values within the first million entries. These were of interest at first so we may use geospatial analysis, but unfortunately there’s too little to consider it useful.  
Hence, we only find ourselves concerned with two variables:

| Field | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| created\_at | string (RFC 2822\) | Timestamp of tweet creation (RFC 2822\) | Tue Sep 07 18:41:48 \+0000 2021 |
| text | string | Raw tweet text | “And how does the change in climate…” |

## **Processing Rules**

### **Reddit**

1. Load raw JSONL files containing Reddit comments and submissions.  
2. Filter records by climate-anxiety keyword list.  
3. Extract subset of fields: subreddit, body, created\_utc.  
4. Drop missing values (dropna(inplace=True)).  
5. Deduplicate on body to remove repeated posts.  
6. Convert timestamps: add created\_dt from created\_utc.  
   1. This can be done later, not necessary now  
7. Ensure UTF-8 encoding for all text fields.  
8. Save cleaned outputs into standardized CSV files.

### **Twitter**

1. Load raw JSONL file of tweets.  
2. Extract subset of fields: created\_at, text  
3. Drop missing values (dropna(inplace=True)).  
4. Deduplicate on text to remove retweets/duplicates.  
5. Split large datasets into \~32 equal-sized CSV chunks.  
6. Ensure UTF-8 encoding for text fields.  
7. Save cleaned outputs into standardized CSV files.

### **About File Splitting**

* Reddit CSVs are already split by subreddit and source file.  
  * 32 total files (16 subs × 2 types: comments/submissions)  
  * Largest file is filtered\_collapse\_comments.csv with 113,656 entries  
  * Smallest is filtered\_anxietydepression\_submissions.csv with 87 entries  
* Twitter CSVs are split into 32 chunks for manageable size.  
  * Original size 5 GB, split into 32 chunks after cleaning (\~103 MB each)

## **Text Cleaning & Normalization Pipeline**

After filtering climate-related Reddit and Twitter data, we apply a structured text preprocessing workflow to prepare the datasets for downstream NLP tasks. Each dataset receives a new column “cleaned\_text” containing the normalized text, which will be used for topic modeling only.

### **Dataset Loading**

* Detects whether a dataset uses the body (Reddit) or text (Twitter) field.  
* Collects all documents into dictionaries keyed by dataset name.

### **Custom Stopword Construction**

We extended the default NLTK English stopwords with three custom layers:

* Swear variants to filter profanities and toxic filler.  
* Platform-specific filler (e.g., *rt, repost, upvote, tweet, hashtag*) to remove social media boilerplate.  
* Generic high-frequency fillers (e.g., *love, great, good, thank, like*) to reduce sentiment-heavy noise.

At the same time, we preserved key words crucial for semantic meaning and negation handling, such as:

* Negations (*not, no, nor, don’t*)  
* Modals (*should, could, would, might, must*)  
* Interrogatives (*why, how, what, if*)

### **Tokenization & Cleaning Rules**

For each text document:

1. Lowercase the text.  
2. Tokenize into words.  
3. Keep only alphabetic tokens (remove numbers, punctuation, URLs, etc.).  
4. Remove stopwords from the custom stopword list.  
5. Remove consecutive duplicate words (e.g., *“very very hot” → “very hot”*).

## **Usage in Analysis & Modeling**

* **Topic Modeling** (regular, dynamic, and merging)  
  * Use “cleaned\_text”  
  * Group by created\_dt (monthly bins) for dynamic topic modeling.  
* **Emotional Analysis**:  
  * Input: body/text.  
  * Output: emotion labels (negative, neutral, positive) stored in derived schema.  
* **Reporting**:  
  * Highlight emerging topics.  
  * Monitor topic variations over time.

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

## **Versioning & Governance**
* **Schema Version**: v1.1 (March 2026\)  
* **Update Policy**: Any new field addition must be documented and versioned.  
* **Data Storage**: CSV (primary), files are initially JSONL.  
* **Data Retention**: Raw JSONL preserved; cleaned CSVs replaceable.