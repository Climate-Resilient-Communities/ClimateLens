# NLP Pipeline

ClimateLens runs a multi-stage NLP pipeline that transforms raw social-media datasets into cleaned text, topic-modeling results, emotion classifications, and visualizations.

The pipeline is designed to support multiple datasets through the dataset registry in `src/config/datasets.yaml`. Dataset-specific configuration determines which topic-modeling and emotion-classification profiles are used.

## Overview

The NLP pipeline consists of four primary stages:

```text
 DATA_DIR (read-only)
      │
      ▼
 data_preprocessing.py
      │
      ▼
 PROCESSED_DATA_DIR
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
 topic_modeling.py      emotion_classification.py
      │                      │
      │                      ▼
      │                OUTPUT_DATA_DIR
      │                      │
      │                      ▼
      │              emotion_visualizations.py
      │                      │
      ▼                      ▼
 PROCESSED_DATA_DIR      OUTPUT_VIS_DIR
```

At a high level, the pipeline:

1. Loads raw CSV datasets from `DATA_DIR`.
2. Cleans and normalizes text.
3. Generates sentence embeddings for topic modeling.
4. Groups documents into semantic topics using BERTopic.
5. Classifies documents into emotion categories.
6. Generates topic and emotion visualizations.
7. Writes all derived artifacts to separate output directories without modifying the original raw data.

The pipeline is intentionally file-based: each stage reads from an input directory and writes its results to a separate output location or to a later-stage output directory.

# 1. Data Preprocessing

## `data_preprocessing.py`

### Purpose

`data_preprocessing.py` prepares raw social-media text for downstream NLP models. It converts raw CSV data into a standardized representation containing a `cleaned_text` column.

### Input

The script searches `DATA_DIR` for CSV files matching the `filename_patterns` registered for each dataset in:

```text
src/config/datasets.yaml
```

### Processing

The preprocessing stage performs Twitter-style text cleaning, including:

* URL removal
* `@handle` removal
* Retweet (`RT`) handling
* HTML entity cleanup
* Lowercasing
* Tokenization
* Stopword removal
* Removal of swear-word variants
* Preservation of negations
* Removal of documents containing fewer than `MIN_DOCUMENT_WORDS` tokens

The default minimum document length is:

```text
MIN_DOCUMENT_WORDS = 3
```

Preserving negations is important for downstream sentiment and emotion analysis because removing words such as `not` can change the semantic meaning of a sentence.

### Output

For each dataset, the script writes:

```text
PROCESSED_DATA_DIR/<dataset>.csv
```

with the cleaned text stored in:

```text
cleaned_text
```

The original data in `DATA_DIR` is never overwritten.

### Considerations

Preprocessing can significantly affect downstream topic and emotion quality. Aggressive cleaning may remove useful contextual information, while insufficient cleaning can cause models to cluster documents based on URLs, usernames, retweet markers, or other noise.

The first run may also download the required NLTK corpora.

# 2. Embeddings

## Sentence-transformer embeddings

Topic modeling uses sentence embeddings to convert text into numerical vectors representing semantic meaning.

The primary embedding model currently used by the pipeline is:

```text
sentence-transformers/all-MiniLM-L12-v2
```

This model provides a useful balance between semantic representation quality and computational cost.

### Embedding model comparison

Several sentence-transformer models were evaluated during development.

| [Model](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models)                               | Dimensions | Maximum input   | Benchmark* | Relative considerations                                           | Project observations                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------- | ---------: | :-------------- | ---------: | :---------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)                                         |        768 | 384 word pieces |      69.57 | High quality but computationally expensive                        | Produced useful Reddit clusters but required refinement                              |
| [`all-distilroberta-v1`](https://huggingface.co/sentence-transformers/all-distilroberta-v1)                                   |        768 | 128 word pieces |      68.73 | Higher-dimensional representation with greater computational cost | Reasonable for Twitter; less effective for Reddit                                    |
| [`all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)                                         |        384 | 256 word pieces |      68.70 | Good balance of quality, speed, and model size                    | Good general-purpose choice; performed better for Reddit than `all-distilroberta-v1` |
| [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)                                           |        384 | 256 word pieces |      68.06 | Faster and smaller than L12                                       | Useful option when inference speed is more important                                 |
| [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |        768 | 128 tokens      |      65.83 | Large model with substantially higher resource requirements       | Potentially useful for multilingual data but comparatively expensive                 |
| [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |        384 | 128 tokens      |      64.25 | Smaller multilingual alternative                                  | Performed well on Twitter; useful for multilingual content                           |
| [`paraphrase-MiniLM-L3-v2`](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)                             |        384 | 128 tokens      |      62.29 | Very small and fast                                               | Lower benchmark performance than the stronger MiniLM variants                        |
| [`distiluse-base-multilingual-cased-v1`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)   |        512 | 128 tokens      |      61.30 | Supports multiple languages but has a larger representation       | Twitter clusters were reasonable; Reddit results were poor                           |
| [`paraphrase-albert-small-v2`](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2)                       |          — | 100 tokens      |      64.46 | Small model with a short input limit                              | Lower-cost alternative for constrained workloads                                     |

* Benchmark values above are model benchmark scores from the development comparison and should not be interpreted as direct measurements of final ClimateLens topic-modeling accuracy. Different models and benchmarks may use different datasets and evaluation procedures.

### Why `all-MiniLM-L12-v2`?

`all-MiniLM-L12-v2` was selected as the primary embedding model because it provides a strong compromise between:

* Semantic quality
* Embedding dimensionality
* Inference speed
* Model size
* Suitability for social-media text

Its 384-dimensional representation is substantially smaller than the 768-dimensional representations produced by models such as `all-mpnet-base-v2`, reducing downstream computational and memory requirements. This alone saved time days of compute and we don't lose too much information.

The development experiments also suggested that it provided a better balance for Reddit data than `all-distilroberta-v1`, while remaining competitive for Twitter data.

### Performance considerations

Embedding generation is one of the most computationally significant stages of the pipeline because every document must be converted into a vector representation.

The main factors affecting runtime and memory usage are:

* Number of documents
* Document length
* Embedding dimensionality
* Model architecture
* CPU vs. GPU execution
* Batch size

Higher-dimensional models generally increase downstream storage and clustering costs as well as embedding computation.

For large datasets, smaller models such as the MiniLM family can provide substantial performance benefits while retaining useful semantic information.

### Strengths and limitations

**Strengths**

* Captures semantic similarity rather than relying solely on keyword overlap.
* Works well for short social-media documents.
* Enables clustering of semantically related documents even when they use different vocabulary.
* MiniLM models provide a relatively good quality-to-cost tradeoff.

**Limitations**

* Embeddings can lose information when documents exceed the model's maximum input length.
* Social-media slang, sarcasm, multilingual text, and domain-specific terminology can reduce embedding quality.
* Different embedding models can produce noticeably different topic clusters.
* Higher-dimensional models require more memory and computation.

# 3. Topic Modeling

## `topic_modeling.py`

### Purpose

`topic_modeling.py` identifies groups of semantically similar documents and assigns topic information to the processed datasets.

The implementation uses BERTopic together with sentence-transformer embeddings, dimensionality reduction, clustering, and topic representation techniques.

### Model architecture

The topic-modeling workflow is approximately:

```text
Cleaned documents
       │
       ▼
Sentence-transformer embeddings
       │
       ▼
UMAP dimensionality reduction
       │
       ▼
HDBSCAN clustering
       │
       ▼
BERTopic topic representation
       │
       ▼
Topic assignments + topic metadata
```

### Embeddings

The primary embedding model is:

```text
sentence-transformers/all-MiniLM-L12-v2
```

The resulting vectors are passed to the topic-modeling pipeline.

### UMAP

UMAP is used to reduce the dimensionality of the embeddings before clustering.

The pipeline uses a seeded UMAP configuration to improve reproducibility between runs.

Dimensionality reduction reduces the computational burden placed on the clustering stage while attempting to preserve meaningful semantic structure.

### HDBSCAN

HDBSCAN groups documents into dense regions of the reduced embedding space.

Unlike methods requiring a fixed number of clusters in advance, HDBSCAN can identify a variable number of clusters and can assign documents to noise/outlier groups.

This is useful for social-media datasets where the number of meaningful topics is not known beforehand.

### Topic representation

BERTopic uses topic representations to identify the terms that characterize each cluster.

The pipeline uses Maximal Marginal Relevance (MMR) to improve topic-word diversity and reduce redundant topic terms.

Optional Cohere integration can be used to generate more human-readable topic labels.

The optional language-model integration uses:

```text
Cohere command-r
```

and requires:

```text
COHERE_API_KEY
```

when enabled.

### Dataset-specific configuration

Topic-modeling hyperparameters are selected through `DATASET_PARAMS` according to each dataset's `topic_profile` in:

```text
src/config/datasets.yaml
```

This allows different datasets to use different topic-modeling configurations without changing the pipeline code.

### Outputs

The topic-modeling stage updates:

```text
PROCESSED_DATA_DIR/<dataset>.csv
```

with topic-related columns and saves the trained model to:

```text
MODELS_DIR/<dataset>.safetensors
```

It can also generate topic visualizations under:

```text
OUTPUT_VIS_DIR/
├── IDM/
├── hierarchies/
├── barcharts/
└── dtm/
```

### Performance

Topic modeling is computationally dependent on several stages:

1. Embedding generation
2. UMAP dimensionality reduction
3. HDBSCAN clustering
4. Topic representation generation
5. Optional language-model-based topic labeling

Embedding generation and dimensionality reduction can become significant bottlenecks as dataset size increases.

Clustering complexity also increases with the number of documents and the dimensionality of the data, although UMAP reduces the dimensionality before HDBSCAN is applied.

The optional Cohere labeling step introduces network/API latency and therefore has different performance characteristics from the local modeling stages.

### Strengths

* Does not require the number of topics to be specified in advance.
* Combines semantic embeddings with density-based clustering.
* Can identify outliers rather than forcing every document into a topic.
* Produces interpretable topic representations.
* Supports visual exploration of topic relationships.
* Can be configured differently for different datasets.

### Limitations

* Topic quality depends strongly on embedding quality and clustering parameters.
* Small or highly heterogeneous datasets may produce unstable or difficult-to-interpret topics.
* Social-media text can contain significant noise, slang, sarcasm, and multilingual content.
* Topic clusters may require manual interpretation or refinement.
* Increasing dataset size increases embedding, dimensionality-reduction, and clustering costs.
* Automatically generated topic labels should be treated as summaries rather than authoritative interpretations.

### Use cases

Topic modeling can be used to:

* Identify major themes in social-media discussions.
* Compare topics between datasets.
* Track changes in discussion themes.
* Identify emerging topics.
* Explore differences between communities or platforms.
* Provide structured input for downstream analysis.

# 3.1 Dynamic Topic Modeling

## `dynamic_topic_modeling.py`

### Purpose

`dynamic_topic_modeling.py` extends the static BERTopic analysis by examining how topics change over time.

Instead of producing only a single topic representation for the entire dataset, the script uses BERTopic's `topics_over_time` functionality to calculate topic representations across temporal intervals. This makes it possible to study changes in discussion themes, topic prevalence, and topic terminology over the lifetime of a dataset.

The script operates on an already-trained BERTopic model and therefore does not train a separate topic model from scratch.

### Input

The dynamic topic-modeling stage uses three primary inputs:

* The original dataset dataframes, used to obtain timestamps.
* The trained BERTopic models produced by the topic-modeling stage.
* The cleaned documents corresponding to each dataset.

The script supports several timestamp column conventions. It checks the following columns in order:

```text
created_utc
created_at
timestamp
date
datetime
```

Reddit datasets typically use `created_utc`, which contains Unix timestamps, while Twitter datasets commonly use `created_at`, which contains datetime strings.

Timestamps are parsed with pandas and invalid timestamps are discarded from the temporal analysis. If no suitable timestamp column exists, or if no valid timestamps can be parsed, dynamic topic modeling is skipped for that dataset.

### Temporal binning

The number of temporal bins can be supplied explicitly, but the current implementation automatically determines the number of bins when no value is provided.

The heuristic is approximately one temporal bin per 30 days, constrained by:

```text
Minimum bins: 10
Maximum bins: 50
```

Therefore, datasets with short time spans still receive at least 10 bins, while very long datasets are capped at 50 bins.

If the dataset covers less than one week, the script warns that dynamic topic modeling may not be particularly meaningful.

The binning strategy should therefore be interpreted as a practical heuristic rather than a statistically optimal temporal resolution.

### Processing

The main analysis is performed using:

```python
topic_model.topics_over_time(
    docs=docs,
    timestamps=timestamps,
    nr_bins=nr_bins,
    evolution_tuning=True,
    global_tuning=True,
)
```

`evolution_tuning=True` allows topic representations to be refined for individual time periods, while `global_tuning=True` uses the global topic representations as a reference.

The resulting `topics_over_time` dataframe contains the temporal topic information generated by BERTopic and can be used for further analysis.

The script then creates an interactive Plotly visualization using:

```python
topic_model.visualize_topics_over_time(...)
```

The visualization currently focuses on the top 10 topics.

### Outputs

For each successfully processed dataset, the script produces:

```text
<dtm_dir>/<dataset>_topics_over_time.csv
<dtm_dir>/<dataset>_topics_over_time.html
```

The CSV contains the temporal topic data for downstream analysis.

The HTML file contains an interactive Plotly visualization showing topic evolution over time.

### Performance

The script records and reports the wall-clock execution time for the dynamic topic-modeling operation.

However, the current implementation does **not** provide standardized benchmark results across datasets or hardware configurations.

Runtime depends on factors including:

* Number of documents.
* Number of temporal bins.
* Number of topics in the trained BERTopic model.
* Document length.
* Complexity of the underlying topic model.
* Available CPU/GPU resources.
* BERTopic and dependency versions.

Dynamic topic modeling also operates on top of an existing BERTopic model, so its cost should be considered separately from the initial embedding, dimensionality-reduction, and clustering stages.

The temporal output size is primarily influenced by the number of topics and temporal bins. Increasing the number of bins produces finer temporal resolution but also increases the amount of temporal topic information that must be calculated and stored.

The current implementation reports empirical runtime for individual executions, but these timings should not be interpreted as general performance benchmarks.

### Strengths

* Reuses the existing trained BERTopic model.
* Does not require training a separate topic model for every time period.
* Supports datasets with different timestamp formats.
* Automatically selects a temporal resolution when one is not specified.
* Provides both machine-readable CSV output and interactive visualizations.
* Enables analysis of changing discussion themes over time.

### Limitations and considerations

Dynamic topic modeling depends on the quality of the underlying static BERTopic model. Poor topic clusters can therefore produce difficult-to-interpret temporal trends.

Temporal results are also sensitive to the selected number of bins. Too few bins can hide meaningful changes, while too many bins can produce sparse or unstable temporal representations.

The timestamp quality of the source dataset is another important consideration. Missing or incorrectly parsed timestamps can remove documents from meaningful temporal analysis.

The script should therefore be viewed as an exploratory temporal-analysis tool rather than a definitive measurement of topic evolution.

### Use cases

Dynamic topic modeling can be used to:

* Track how climate-related discussion themes change over time.
* Identify emerging or declining topics.
* Examine changes in topic terminology.
* Compare periods of increased discussion activity.
* Investigate topic evolution around significant events.
* Compare temporal patterns between social-media datasets.
* Support longitudinal analysis of public discussion.

# 4. Emotion Model Comparison

The development comparison included several candidate emotion models.

| Model ([HF Docs](https://huggingface.co/docs/transformers/v4.57.0/en/main_classes/pipelines#transformers.TextClassificationPipeline))                                                                                                        | Emotion labels | Performance                   | Strength                                                | Limitation                                                | Project-relevant notes                                                                      |
| ------------------------------------------------------------------------------------------------------------ | -------------: | ----------------------------- | ------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [`bert-emotion`](https://huggingface.co/boltuix/bert-emotion)                                                |             13 | ~90–95% accuracy              | Lightweight; fast inference                             | Smaller emotion taxonomy                                  | BERT-mini/micro architecture (~6M parameters); designed for real-time/offline use           |
| [`roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions)                        |             28 | 47.4% accuracy; 45.0% F1      | Broad emotion coverage                                  | Poorer performance on rare emotions                       | Multi-label classification using GoEmotions; appropriate for social-media emotion detection |
| [`modernbert-base-go-emotions`](https://huggingface.co/cirimus/modernbert-base-go-emotions)                  |             28 | 46.5% F1 default; 54.1% tuned | Modern architecture; improved F1 after threshold tuning | Generalization beyond Reddit may be limited               | Trained on GoEmotions/Reddit; threshold tuning is important for evaluation                  |
| [`emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)                                                |             7 | 60% accuracy              | Lightweight; fast inference                             | Small emotion taxonomy                                  | ...           |
| [`bertweet-base-sentiment-analysis`](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) |              — | —                             | Strong Twitter specialization                           | Sentiment-focused rather than full emotion classification | Fine-tuned on English tweets; useful as a Twitter-specific comparison model                 |                                                                                                               | 

These figures should **not** be interpreted as a direct ranking because the models may have been evaluated using different datasets, metrics, thresholds, and evaluation procedures. Moreover, the evaluation setup/metric may not be directly comparable. The F1 scores are probably the more useful comparison for multi-label tasks.

For ClimateLens, model selection is based primarily on dataset suitability and the project's configuration rather than simply choosing the model with the highest reported benchmark number.

### Runtime considerations

Emotion classification requires transformer inference for each document.

Runtime is affected by:

* Number of documents
* Document length
* Model size
* Batch size
* Hardware
* Number of output labels

The development benchmarks showed meaningful differences between models, with the lightweight BERT-based model being more suitable for low-latency applications and larger RoBERTa/ModernBERT models requiring greater computational resources.

For large datasets, batching and GPU acceleration can substantially reduce total inference time.

# 5. Emotion Classification

## `emotion_classification.py`

### Purpose

`emotion_classification.py` assigns emotion labels to individual documents using a dataset-specific transformer model.

The model is selected using the `emotion_profile` defined in:

```text
src/config/datasets.yaml
```

### Model selection

The current profiles are:

| Profile   | Model                              | Labels | Primary use                                            |
| --------- | ---------------------------------- | -----: | ------------------------------------------------------ |
| `twitter` | `boltuix/bert-emotion`             |     13 | Twitter/social-media emotion classification            |
| `reddit`  | `SamLowe/roberta-base-go_emotions` |     28 | Reddit and broader social-media emotion classification |
| `default` | `SamLowe/roberta-base-go_emotions` |     28 | General-purpose fallback                               |

| Model ([HF Docs](https://huggingface.co/docs/transformers/v4.57.0/en/main_classes/pipelines#transformers.TextClassificationPipeline)) | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Time R \| T    |
| ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| [bertweet-base-sentiment-analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)                            | variant of RoBERTa used for sentiment analysis specifically fine-tuned on English tweets, trained on SemEval 2017 dataset (~40k tweets)                                                                                                                                                                                                                                                                                                                                                                                                   | 18:16 \| 09:17 |
| [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                                                   | Fine-tuned roberta-base model for multi-label emotion classification using the GoEmotions dataset (28 emotion labels). Outputs 28 probabilities per input. Trained for 3 epochs with a 2e-5 learning rate and 0.01 weight decay. Achieved 47.4% accuracy, 57.5% precision, 39.6% recall, and 45.0% F1 score. Performance varies by label, with lower scores for rare emotions like grief and relief. Data cleaning could improve results. Suitable for emotion detection in social media posts, reviews, and other multi-label NLP tasks. | 27:42 \| 08:59 |
| [modernbert-base-go-emotions](https://huggingface.co/cirimus/modernbert-base-go-emotions)                                             | Fine-tuned ModernBERT-base on the GoEmotions dataset (58,000 Reddit comments, 28 emotion labels) for multi-label classification. Trained for 3 epochs with a 2e-5 learning rate, batch size 16, weight decay 0.01, 500 warmup steps, using AdamW. Achieved 97.0% accuracy and 46.5% F1 at a 0.5 threshold; threshold tuning raises F1 to 54.1%. Performs well on common labels, less so on rare ones like grief. Best suited for social media sentiment and psychological analysis. Limited generalization beyond Reddit.                 | 20:10 \| 45:46 |
| [bert-emotion](https://huggingface.co/boltuix/bert-emotion)                                                                           | Lightweight emotion detection model based on BERT-mini/micro, fine-tuned for real-time, offline use on edge and IoT devices. Uses a 4-layer, 128-hidden size architecture with 4 attention heads, ~6 million parameters, and ~20MB quantized size. Supports 13 emotions with ~90–95% accuracy. Designed for low-latency inference in privacy-first applications like social media sentiment and mental health monitoring.                                                                                                                 | 

## `boltuix/bert-emotion`

This is a lightweight BERT-based emotion classifier designed for low-resource and low-latency use.

The model uses a compact architecture with approximately:

* 4 transformer layers
* 128 hidden dimensions
* 4 attention heads
* Approximately 6 million parameters
* Approximately 20 MB when quantized

It supports 13 emotion categories.

The model documentation reports approximately 90–95% accuracy, although this figure should be interpreted in the context of the model's own evaluation data rather than as an expected accuracy on ClimateLens datasets.

### Strengths

* Small model size.
* Low computational requirements.
* Suitable for relatively fast inference.
* Designed for social-media-style text.
* Appropriate when a smaller emotion taxonomy is acceptable.

### Limitations

* Provides fewer emotion categories than GoEmotions.
* Reported accuracy may not generalize directly to ClimateLens datasets.
* A smaller model can miss subtle distinctions between related emotions.
* Social-media sarcasm and implicit emotions remain difficult to classify.

## `SamLowe/roberta-base-go_emotions`

This model is based on RoBERTa and is fine-tuned on the GoEmotions dataset.

It supports 28 emotion labels and produces a probability for each label.

Reported development metrics include:

* Accuracy: 47.4%
* Precision: 57.5%
* Recall: 39.6%
* F1: 45.0%

Performance varies considerably between emotion categories. Common emotions generally perform better than rare categories such as grief and relief.

### Strengths

* Broad 28-label emotion taxonomy.
* Trained specifically for emotion classification.
* Suitable for social-media and conversational text.
* Provides more detailed emotion information than the 13-label model.

### Limitations

* Rare emotions are substantially more difficult to classify.
* Performance metrics from the GoEmotions evaluation should not be treated as ClimateLens-specific accuracy.
* Multi-label emotion classification is inherently more difficult than simple sentiment classification.
* Data cleaning and domain differences can affect results.
* Predictions may be less reliable for text that differs substantially from the training distribution.

## `cirimus/modernbert-base-go-emotions`

A ModernBERT-based GoEmotions model was also evaluated during development.

It was fine-tuned on the GoEmotions dataset using 28 emotion labels.

Reported results include:

* Accuracy: 97.0%
* F1: 46.5% at a 0.5 threshold
* F1: 54.1% after threshold tuning

The large difference between accuracy and F1 demonstrates why accuracy should not be used alone to evaluate multi-label emotion classification, particularly when classes are imbalanced.

### Strengths

* Modern transformer architecture.
* Strong performance on common labels.
* Threshold tuning can improve F1.
* Designed for social-media-style emotion classification.

### Limitations

* Performance is weaker for rare emotions such as grief.
* The underlying training data is heavily associated with Reddit.
* Generalization to other domains or platforms may be limited.
* Larger models can increase inference cost.
* It is not currently the default model used by the pipeline.

# 6. Data Flow

The complete pipeline can be summarized as follows:

```text
Raw CSV datasets
      │
      ▼
┌───────────────────────────┐
│ data_preprocessing.py     │
│                           │
│ Cleaning                  │
│ Tokenization              │
│ Stopword filtering        │
│ Document filtering        │
└─────────────┬─────────────┘
              │
              ▼
      Processed CSV files
              │
              ├──────────────────────┐
              │                      │
              ▼                      ▼
┌─────────────────────────┐  ┌──────────────────────────┐
│ topic_modeling.py       │  │ emotion_classification.py│
│                         │  │                          │
│ Sentence embeddings     │  │ Transformer inference    │
│ UMAP                    │  │ Dataset-specific model   │
│ HDBSCAN                 │  │                          │
│ BERTopic                │  │                          │
│ MMR                     │  │                          │
└────────────┬────────────┘  └─────────────┬────────────┘
             │                             │
             ▼                             ▼
       Topic-enriched CSV          Emotion-enriched CSV
             │                             │
             ▼                             ▼
       Topic models +             emotion_visualizations.py
       visualizations                     │
                                           ▼
                                   Emotion visualizations
                                   + summary HTML
```

The two major analytical branches are therefore:

* **Topic modeling:** identifies what people are discussing.
* **Emotion classification:** identifies how people are expressing emotion.

These outputs can be analyzed independently or combined for more detailed analysis of emotional responses within individual topics.

# 7. Performance and Scaling Considerations

## Main computational bottlenecks

The most computationally expensive operations are generally:

1. Sentence embedding generation.
2. UMAP dimensionality reduction.
3. HDBSCAN clustering.
4. Transformer-based emotion classification.
5. Optional external LLM-based topic labeling.

The exact runtime depends heavily on dataset size and hardware.

### Embedding performance

Embedding model choice has a direct effect on both runtime and memory consumption.

Larger models such as `all-mpnet-base-v2` provide higher-dimensional representations but require more computation and storage.

Smaller MiniLM models offer a better tradeoff when processing large quantities of social-media data.

### Topic-modeling performance

Topic modeling becomes increasingly expensive as the number of documents grows.

Memory requirements are influenced by:

* Number of documents
* Embedding dimensions
* Intermediate UMAP representations
* Clustering structures
* Stored topic-model artifacts

The choice of embedding model therefore affects not only embedding generation but also later stages of topic modeling.

### Emotion-classification performance

Emotion classification performs transformer inference over individual documents.

Smaller models are preferable when:

* Dataset size is very large.
* Low latency is important.
* Hardware resources are limited.

Larger models may be preferable when:

* More nuanced emotion distinctions are required.
* Additional compute is available.
* The model is better aligned with the target dataset.

### CPU vs. GPU

The transformer-based embedding and emotion-classification stages can benefit significantly from GPU acceleration.

CPU execution remains possible but may become a bottleneck for large datasets.

For reproducible comparisons, benchmark results should always specify:

* Dataset size
* Hardware
* Batch size
* Model version
* Number of documents
* Whether inference was performed on CPU or GPU

# 8. Model Strengths and Tradeoffs

There is no single model that is optimal for every ClimateLens dataset.

### Embeddings

**`all-MiniLM-L12-v2`**

Best general-purpose tradeoff currently used by the pipeline.

* 384 dimensions
* 256-token maximum input
* Good semantic quality
* Moderate computational cost

**`all-mpnet-base-v2`**

Better suited to workloads where semantic representation quality is prioritized over speed.

* 768 dimensions
* 384-word-piece maximum input
* Strong benchmark performance
* Significantly more computationally expensive

**`all-MiniLM-L6-v2`**

Useful when processing speed and resource usage are the priority.

* 384 dimensions
* 256-token maximum input
* Smaller/faster architecture
* Slightly lower benchmark performance than L12

### Emotion models

**`bert-emotion`**

Best suited to lightweight Twitter-oriented classification.

* Small
* Fast
* 13 emotion labels
* Lower resource requirements

**`roberta-base-go_emotions`**

Best suited when a broader emotion taxonomy is needed.

* 28 emotion labels
* Strong alignment with social-media emotion analysis
* More computationally expensive
* Rare emotions remain difficult

**`modernbert-base-go-emotions`**

Potential alternative for experiments requiring a more modern architecture.

* 28 labels
* Good performance on common emotions
* Threshold tuning can improve F1
* Greater domain/generalization considerations

# 9. Key Limitations

The pipeline's outputs should be treated as analytical signals rather than ground truth.

Important limitations include:

### Domain shift

Models trained on one type of social-media content may perform differently on another platform, dataset, or subject matter.

### Short and noisy text

Social-media posts can contain:

* Misspellings
* Slang
* Abbreviations
* Hashtags
* Sarcasm
* Code-switching
* Incomplete sentences

These characteristics can make both topic and emotion classification more difficult.

### Sarcasm and implicit emotion

Transformer models can struggle when the literal text differs from the author's intended meaning.

### Class imbalance

Emotion datasets contain substantially more examples of some emotions than others. Rare emotions therefore tend to have weaker classification performance.

### Topic interpretation

A topic cluster is a statistical grouping of documents, not necessarily a clearly defined real-world subject. Topic labels should therefore be reviewed before being used as definitive interpretations.

### Reproducibility

Although the pipeline uses seeded UMAP configuration, some components and model implementations may still produce variation between environments or library versions.

For reproducible experiments, model versions, package versions, parameters, hardware, and dataset versions should be recorded.

# 10. Use Cases

The NLP pipeline can support several types of analysis.

### Topic discovery

Identify the major themes present in a dataset without manually defining categories beforehand.

### Trend analysis

Track how topics or emotions change over time.

### Cross-platform comparison

Compare the themes and emotional patterns found in Twitter and Reddit datasets.

### Topic-emotion analysis

Combine topic assignments with emotion labels to answer questions such as:

* Which topics generate the strongest emotional responses?
* Which emotions are associated with specific topics?
* How does the emotional response to a topic change over time?

### Exploratory research

Use topic clusters, emotion distributions, and visualizations to identify patterns that can later be investigated manually or with more specialized statistical methods.

# 11. Outputs

Depending on the stages executed, the pipeline produces several types of artifacts.

### Processed datasets

```text
PROCESSED_DATA_DIR/<dataset>.csv
```

Contains cleaned text and, after topic modeling, topic-related columns.

### Topic models

```text
MODELS_DIR/<dataset>.safetensors
```

### Topic visualizations

```text
OUTPUT_VIS_DIR/
├── IDM/
├── hierarchies/
├── barcharts/
└── dtm/
```

### Emotion-classified datasets

```text
OUTPUT_DATA_DIR/<dataset>_with_emotions.csv
```

### Emotion visualizations

```text
OUTPUT_VIS_DIR/
└── emotions/
    ├── wordclouds/
    └── timeseries/
```

A summary HTML report is also generated by the emotion visualization stage.

# 12. Configuration and Dataset Management

Datasets are registered in:

```text
src/config/datasets.yaml
```

Each dataset specifies information such as:

* Canonical dataset name
* Raw filename patterns
* Relevant columns
* Topic-modeling profile
* Emotion-classification profile

To add a new dataset:

1. Add an entry to `src/config/datasets.yaml`.
2. Provide the canonical name.
3. Add filename patterns matching the raw CSV.
4. Configure the appropriate columns and model profiles.
5. Place the raw CSV in `DATA_DIR`.
6. Re-run the pipeline.

The pipeline automatically discovers the new dataset through the registry.

# 13. Running Locally

Copy the environment template:

```bash
cp .env.example .env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
make pipeline
```

Alternatively, the stages can be executed individually:

```bash
python src/data_preprocessing.py
python src/topic_modeling.py
python src/emotion_classification.py
python src/emotion_visualizations.py
```

Environment variables include:

```text
DATA_DIR
PROCESSED_DATA_DIR
MODELS_DIR
OUTPUT_DATA_DIR
OUTPUT_VIS_DIR
```

An optional:

```text
COHERE_API_KEY
```

can be supplied when Cohere-based topic labeling is enabled.

# 14. AzureML

The pipeline can also be executed through AzureML using:

```text
azureml/AML_job.py
```

which submits the pipeline to an AzureML cluster through:

```text
azureml/run_scripts.sh
```

AzureML execution is detected automatically when any of the following environment variables are present:

```text
AZUREML_RUN_ID
AZUREML_EXPERIMENT_ID
AZUREML_OUTPUT_DIR
```

When running in AzureML, the pipeline writes its outputs under:

```text
./outputs/...
```

rather than the default local directory structure.

# Summary

ClimateLens combines text preprocessing, sentence embeddings, BERTopic-based topic modeling, transformer-based emotion classification, and postprocessing/visualization into a single dataset-driven NLP workflow.

The main design tradeoffs are between **semantic quality, computational cost, dataset suitability, and interpretability**.

For topic modeling, `all-MiniLM-L12-v2` provides the current balance between embedding quality and computational efficiency. Larger models such as `all-mpnet-base-v2` can capture more complex semantic information but require substantially more resources.

For emotion classification, the pipeline selects models according to the characteristics of the dataset. `bert-emotion` provides a lightweight 13-label solution for Twitter-oriented data, while `roberta-base-go_emotions` provides a broader 28-label emotion taxonomy for Reddit and general use.

Ultimately, model benchmark scores should be considered alongside **runtime, memory requirements, dataset domain, class imbalance, cluster quality, and interpretability**. Reported model-card metrics are useful for comparison, but they should not be treated as guarantees of performance on ClimateLens datasets.