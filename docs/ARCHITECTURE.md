# ClimateLens Architecture

ClimateLens is an NLP and machine-learning system for exploring **climate-related discussion and emotional expression in social-media data**. It processes data from platforms such as Reddit and Twitter, cleans and normalizes the text, identifies semantic topics, classifies emotional expression, and produces visualizations that allow researchers and other users to explore the results.

This document provides a high-level overview of how the major components fit together. It intentionally does not reproduce the detailed implementation documentation for each component. Instead, it acts as a map to the project's more specific documentation.



## Architecture at a Glance

At a high level, ClimateLens can be viewed as the following pipeline:

```text
                         ┌─────────────────────┐
                         │     Data Sources    │
                         │                     │
                         │  Reddit + Twitter   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Extraction   │
                         │   & Filtering       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Preprocessing    │
                         │                     │
                         │ Cleaning            │
                         │ Tokenization        │
                         │ Stopword filtering  │
                         │ Document filtering  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Processed Data    │
                         │   cleaned_text      │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌──────────────────────┐
          │   Topic Modeling    │       │ Emotion Classification│
          │                     │       │                      │
          │ Sentence Embeddings │       │ Transformer Model    │
          │ UMAP                │       │ Dataset-specific     │
          │ HDBSCAN             │       │ emotion profile      │
          │ BERTopic            │       │                      │
          │ MMR                 │       └──────────┬───────────┘
          └──────────┬──────────┘                  │
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌──────────────────────┐
          │ Topic-enriched Data │       │ Emotion-enriched Data│
          │ + Topic Models      │       │                      │
          └──────────┬──────────┘       └──────────┬───────────┘
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌──────────────────────┐
          │ Topic Visualizations│       │ Emotion Visualizations│
          │                     │       │                      │
          │ Intertopic Map      │       │ Distributions        │
          │ Bar Charts          │       │ Probability           │
          │ Hierarchies         │       │ Word Clouds           │
          │ Topics Over Time    │       │ Time Series           │
          └─────────────────────┘       └──────────────────────┘
```

The system therefore has two primary analytical branches:

* **Topic modeling** answers *what people are discussing*.
* **Emotion classification** answers *how people are emotionally expressing themselves*.

These branches can be analyzed independently or combined to investigate relationships between topics and emotional responses.



# 1. Data Sources

ClimateLens currently works with social-media data from **Reddit and Twitter**.

### Reddit

Reddit comments and submissions are filtered using a climate-related keyword list. The standardized Reddit schema retains:

* `subreddit`
* `body`
* `created_utc`

### Twitter

Twitter datasets contain climate-related tweets extracted from publicly available Twitter archives. The standardized schema retains:

* `created_at`
* `text`

The original text and source metadata are preserved so that downstream analysis can distinguish between raw/source data and derived NLP fields.

For the complete dataset definitions, see **[Data Schema](data_schema.md)**.



# 2. Data Extraction and Filtering

Before the NLP pipeline begins, source-specific processing converts the raw Reddit and Twitter data into standardized CSV datasets.

The extraction stage performs tasks such as:

* Loading raw JSONL data.
* Filtering Reddit records using climate-related keywords.
* Extracting the fields required by ClimateLens.
* Removing missing records.
* Removing duplicate posts or tweets.
* Normalizing timestamps.
* Writing standardized UTF-8 CSV files.

The output of this stage becomes the input to the text preprocessing pipeline.

Detailed source-specific processing and filtering is documented in **[Preprocessing](preprocessing.md)**.

The resulting field definitions are documented in **[Data Schema](data_schema.md)**.



# 3. Preprocessing

The preprocessing stage prepares the social-media text for downstream NLP models.

The main implementation is:

```text
data_preprocessing.py
```

The raw/source text is retained while a separate:

```text
cleaned_text
```

field is generated for NLP analysis.

The text normalization process includes:

1. Lowercasing.
2. Tokenization.
3. Punctuation removal.
4. URL removal.
5. Number removal.
6. Alphabetic-token filtering.
7. Custom stopword removal.
8. Removal of consecutive duplicate words.
9. Filtering documents that are too short.

The preprocessing pipeline also deliberately preserves semantically important words such as negations and modal verbs.

This separation between **original text** and **cleaned text** is important. Topic modeling uses the normalized `cleaned_text`, while emotion analysis can use the original wording where appropriate.

For the complete preprocessing workflow, see **[Preprocessing](preprocessing.md)**.

The preprocessing stage and its interaction with the rest of the NLP pipeline are also described in **[NLP Pipeline](pipeline.md)**.



# 4. Dataset Configuration

ClimateLens uses a dataset registry to make the pipeline dataset-driven.

Dataset configuration is maintained in:

```text
src/config/datasets.yaml
```

The registry determines information such as:

* Dataset name.
* Raw filename patterns.
* Relevant source columns.
* Topic-modeling profile.
* Emotion-classification profile.

This allows different datasets to use different model configurations without requiring changes to the core pipeline code.

Conceptually:

```text
                 datasets.yaml
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    Topic profile             Emotion profile
          │                         │
          ▼                         ▼
    Topic modeling          Emotion classification
```

Adding a new dataset therefore primarily involves registering its data and configuration rather than creating an entirely new processing pipeline.



# 5. NLP and Modeling Pipeline

The core NLP functionality is implemented as a series of independent stages.

```text
Processed text
      │
      ├──────────────────────────────┐
      │                              │
      ▼                              ▼
 Embeddings                    Emotion Classification
      │                              │
      ▼                              ▼
 Topic Modeling                 Emotion Labels
      │
      ├───────────────┐
      ▼               ▼
Static Topics     Dynamic Topics
      │               │
      └───────┬───────┘
              ▼
       Visualizations
```

The complete implementation and execution flow is documented in **[NLP Pipeline](pipeline.md)**.



## 5.1 Embeddings

The topic-modeling branch first converts cleaned documents into numerical semantic representations.

The current default model is:

```text
sentence-transformers/all-MiniLM-L12-v2
```

Each document is represented as a 384-dimensional embedding.

The embeddings provide the semantic representation used by the subsequent dimensionality-reduction and clustering stages.

Implementation:

```text
embeddings.py
```

The embedding stage is computationally significant because every document must be processed by the transformer model.

For model comparisons, performance considerations, and embedding-specific details, see **[NLP Pipeline — Embeddings](pipeline.md)**.



# 6. Topic Modeling

ClimateLens currently uses **BERTopic**, rather than a traditional LDA-based topic model.

The topic-modeling architecture is:

```text
Cleaned documents
       │
       ▼
Sentence-transformer embeddings
       │
       ▼
       UMAP
       │
       ▼
     HDBSCAN
       │
       ▼
    BERTopic
       │
       ▼
Topic representations
       │
       ▼
Topic assignments + metadata
```

The major components are:

### Sentence embeddings

Provide semantic representations of documents.

### UMAP

Reduces the dimensionality of the embeddings while attempting to preserve meaningful semantic structure.

### HDBSCAN

Clusters documents into dense regions of semantic similarity and can identify outlier documents without requiring a fixed number of clusters.

### BERTopic

Uses the resulting clusters to construct interpretable topic representations.

### MMR

Maximal Marginal Relevance is used to improve topic-word diversity and reduce redundant topic terms.

### Optional topic labeling

Cohere can optionally be used to generate more human-readable topic labels.

Topic-modeling configuration is dataset-specific and is selected through the dataset registry.

The implementation is primarily contained in:

```text
topic_modeling.py
```

For the complete topic-modeling architecture, configuration, outputs, strengths, and limitations, see **[NLP Pipeline — Topic Modeling](pipeline.md)**.



# 7. Dynamic Topic Modeling

ClimateLens also supports analysis of how topics change over time.

The implementation is:

```text
dynamic_topic_modeling.py
```

Rather than training a new topic model for every time period, the system reuses an existing BERTopic model and applies BERTopic's temporal topic functionality.

```text
Trained BERTopic model
          │
          │
Dataset timestamps
          │
          ▼
   Temporal binning
          │
          ▼
 topics_over_time(...)
          │
          ├───────────────┐
          ▼               ▼
       CSV output    Interactive HTML
```

This makes it possible to investigate:

* Emerging topics.
* Declining topics.
* Changes in topic terminology.
* Changes in discussion intensity.
* Topic evolution around significant events.

The temporal analysis supports multiple source timestamp conventions, including Reddit's `created_utc` and Twitter's `created_at`.

For the full implementation and temporal-analysis details, see **[NLP Pipeline — Dynamic Topic Modeling](pipeline.md)**.



# 8. Emotion Classification

The second major analytical branch classifies the emotional expression contained in individual documents.

The implementation is:

```text
emotion_classification.py
```

Unlike topic modeling, which uses a common embedding/topic-model architecture, emotion classification uses a **dataset-specific transformer model**.

The model is selected using the `emotion_profile` in:

```text
src/config/datasets.yaml
```

Current profiles include:

| Profile   | Model                              | Labels |
|  | - | --: |
| `twitter` | `boltuix/bert-emotion`             |     13 |
| `reddit`  | `SamLowe/roberta-base-go_emotions` |     28 |
| `default` | `SamLowe/roberta-base-go_emotions` |     28 |

Conceptually:

```text
Original social-media text
          │
          ▼
Dataset-specific transformer
          │
          ▼
Emotion probabilities
          │
          ▼
Emotion labels
          │
          ▼
Emotion-enriched dataset
```

The project has evaluated multiple candidate emotion models during development, but these comparisons should not be interpreted as ClimateLens-specific benchmarks.

For the model details, tradeoffs, and candidate-model comparisons, see **[NLP Pipeline — Emotion Classification](pipeline.md)**.



# 9. Postprocessing and Outputs

The modeling stages produce derived datasets and model artifacts rather than modifying the original raw data.

Typical outputs include:

```text
PROCESSED_DATA_DIR/
└── <dataset>.csv
```

Topic models:

```text
MODELS_DIR/
└── <dataset>.safetensors
```

Emotion-enriched datasets:

```text
OUTPUT_DATA_DIR/
└── <dataset>_with_emotions.csv
```

Topic visualization artifacts:

```text
OUTPUT_VIS_DIR/
├── IDM/
├── hierarchies/
├── barcharts/
└── dtm/
```

Emotion visualization artifacts:

```text
OUTPUT_VIS_DIR/
└── emotions/
    ├── wordclouds/
    └── timeseries/
```

This file-based architecture allows individual stages to be rerun without modifying the original source datasets.



# 10. Visualization

Visualization is the presentation layer of ClimateLens.

The visualization components transform model outputs into interactive or human-readable artifacts that allow users to explore the results.

The visualization work currently covers two broad categories:

```text
                    Model Outputs
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Topic Outputs         Emotion Outputs
              │                     │
              ▼                     ▼
       Topic Visualizations   Emotion Visualizations
```

## Topic visualizations

The project includes visualizations such as:

* **Intertopic Map** — represents topics as bubbles, where size indicates topic intensity and spatial relationships represent similarity.
* **Topic Bar Charts** — show the relative frequency or importance of terms associated with topics.
* **Topic Hierarchy** — represents relationships between broad topics and more specific subtopics.
* **Topics Over Time** — shows how topic representations change over temporal intervals.

The visualization documentation also records current design considerations and proposed refinements.

See **[Visualization](visualization.md)** for the current visualization specifications and open design questions.

## Emotion visualizations

The emotion visualization stage is implemented through:

```text
emotion_visualizations.py
```

It consumes emotion-classified datasets and generates artifacts including:

* Emotion word clouds.
* Emotion time series.
* Emotion distributions.
* Emotion probability visualizations.
* A summary HTML report.

If emotion labels are already present in the input dataset, the visualization stage reuses them rather than running emotion inference again.

For the current implementation details, see **[NLP Pipeline — Visualization](pipeline.md)** and **[Visualization](visualization.md)**.



# 11. Data Flow

The complete ClimateLens data flow can be summarized as:

```text
┌───────────────────────┐
│ Raw Reddit / Twitter │
│        data           │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Extraction & Filtering│
│                       │
│ Keywords              │
│ Deduplication         │
│ Missing-value removal │
│ Schema normalization  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Standardized CSV      │
│ datasets              │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Text Preprocessing    │
│                       │
│ Cleaning              │
│ Tokenization          │
│ Stopwords             │
│ Filtering             │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ cleaned_text          │
└───────────┬───────────┘
            │
       ┌────┴────┐
       │         │
       ▼         ▼
┌───────────┐ ┌───────────────┐
│   Topic   │ │    Emotion    │
│  Modeling │ │ Classification│
└─────┬─────┘ └───────┬───────┘
      │               │
      ▼               ▼
┌───────────┐ ┌───────────────┐
│  Topics   │ │   Emotions    │
│ + Models  │ │ + Probabilities│
└─────┬─────┘ └───────┬───────┘
      │               │
      ▼               ▼
┌───────────┐ ┌───────────────┐
│   Topic   │ │    Emotion    │
│Visualization│ │ Visualization│
└───────────┘ └───────────────┘
```

Dynamic topic modeling branches from the trained topic model:

```text
                 Trained BERTopic Model
                          │
                          ▼
                  Dataset timestamps
                          │
                          ▼
                 Dynamic Topic Model
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
              CSV data       Interactive HTML
```



# 12. Evaluation

Evaluation is intentionally **not yet part of the active architecture**.

The project documentation currently describes model-specific development observations and external model-card metrics, but ClimateLens does not yet have a standardized evaluation pipeline or benchmarking suite.

Therefore, evaluation is not represented as a completed processing stage in the current architecture.

Once formal evaluation is implemented, it can be integrated into the architecture between modeling and visualization:

```text
                 Modeling
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      Evaluation         Visualization
```

The future evaluation layer can assess topics and emotion predictions using appropriate metrics and standardized ClimateLens datasets.

Until then, metrics reported in model documentation should not be interpreted as ClimateLens-specific performance measurements.



# 13. AzureML Integration

ClimateLens can be executed locally or remotely through **Azure Machine Learning (AzureML)**.

The AzureML integration provides a way to run the existing pipeline on remote compute without requiring a separate implementation of the NLP workflow.

The architecture is:

```text
                 Local Project
                      │
                      ▼
              AzureML Job Script
               AML_job.py
                      │
                      ▼
             AzureML Run Script
             run_scripts.sh
                      │
                      ▼
              AzureML Workspace
                      │
                      ▼
             Remote Compute
                      │
                      ▼
             ClimateLens Pipeline
                      │
                      ▼
                  Outputs
```

The main AzureML-related scripts are:

```text
azureml/AML_job.py
azureml/run_scripts.sh
```

The AzureML environment detects remote execution using environment variables such as:

```text
AZUREML_RUN_ID
AZUREML_EXPERIMENT_ID
AZUREML_OUTPUT_DIR
```

When executed through AzureML, generated artifacts are written under:

```text
./outputs/...
```

rather than the default local output locations.

AzureML therefore acts primarily as the **execution and compute layer**. The underlying ClimateLens preprocessing and modeling components remain the same.

For workspace configuration, dependency management, compute setup, and troubleshooting, see **[AzureML Configuration](azure_configuration.md)**.



# 14. Technology Stack

The major technologies used by ClimateLens include:

| Technology                    | Role                                                 |
| -- | - |
| **Python**                    | Core implementation language                         |
| **pandas**                    | Dataset loading, transformation, and manipulation    |
| **NLTK**                      | Tokenization and stopword resources                  |
| **Sentence Transformers**     | Semantic document embeddings                         |
| **Hugging Face Transformers** | Emotion classification models                        |
| **BERTopic**                  | Topic modeling                                       |
| **UMAP**                      | Embedding dimensionality reduction                   |
| **HDBSCAN**                   | Density-based document clustering                    |
| **Plotly**                    | Interactive visualizations                           |
| **Streamlit**                 | User-facing exploration/application layer            |
| **NumPy**                     | Numerical and embedding data structures              |
| **AzureML**                   | Remote execution and machine-learning infrastructure |
| **Cohere** *(optional)*       | LLM-assisted topic labeling                          |

The modeling stack can therefore be thought of as:

```text
Python
 │
 ├── pandas / NumPy
 │
 ├── NLTK
 │
 ├── Sentence Transformers
 │       │
 │       └── Embeddings
 │
 ├── BERTopic
 │    ├── UMAP
 │    ├── HDBSCAN
 │    └── MMR
 │
 ├── Hugging Face Transformers
 │       │
 │       └── Emotion Classification
 │
 ├── Plotly / Streamlit
 │       │
 │       └── Visualization & Exploration
 │
 └── AzureML
         │
         └── Remote Execution
```



# 15. Configuration and Directory Boundaries

ClimateLens separates source data, intermediate data, models, and visualization outputs.

The main logical boundaries are:

```text
DATA_DIR
    │
    │ raw/source data
    ▼
PROCESSED_DATA_DIR
    │
    │ cleaned + topic-enriched data
    ├───────────────┐
    │               │
    ▼               ▼
MODELS_DIR      OUTPUT_DATA_DIR
                    │
                    ▼
              OUTPUT_VIS_DIR
```

This separation prevents derived artifacts from overwriting source data and allows individual stages to be rerun independently.

The main environment variables controlling these locations are:

```text
DATA_DIR
PROCESSED_DATA_DIR
MODELS_DIR
OUTPUT_DATA_DIR
OUTPUT_VIS_DIR
```



# 16. Extensibility

ClimateLens is designed so that new datasets, preprocessing rules, models, and visualizations can be added without redesigning the entire system.

## Adding a new dataset

A new dataset can generally be introduced by:

1. Adding it to `src/config/datasets.yaml`.
2. Defining its filename patterns.
3. Identifying its relevant columns.
4. Selecting appropriate topic and emotion profiles.
5. Placing the source data in `DATA_DIR`.

The existing pipeline can then discover and process the dataset through the registry.

See **[NLP Pipeline — Configuration and Dataset Management](pipeline.md)** for the detailed process.

## Adding a preprocessing step

Preprocessing is centralized in the preprocessing stage, making it possible to introduce additional cleaning or normalization operations without changing the topic-modeling or emotion-classification implementations.

Any new preprocessing operation should preserve the distinction between source text and derived `cleaned_text` where possible.

See **[Preprocessing](preprocessing.md)** for the current processing rules.

## Adding a new embedding model

The embedding stage can be extended to support additional Sentence Transformer models.

A new model can be evaluated based on:

* Semantic quality.
* Embedding dimensionality.
* Inference speed.
* Memory requirements.
* Dataset suitability.

See **[NLP Pipeline — Embeddings](pipeline.md)** for the current model comparison.

## Adding a new topic-modeling configuration

Topic-modeling parameters are selected through dataset profiles, allowing different datasets to use different configurations.

The existing architecture can therefore accommodate additional topic-modeling configurations without duplicating the entire pipeline.

## Adding a new emotion model

Emotion models are selected through dataset-specific emotion profiles.

This allows additional Hugging Face classification models to be introduced while keeping the surrounding pipeline unchanged.

## Adding new visualizations

Visualization stages consume derived model outputs rather than raw data. New visualizations can therefore be added as additional consumers of:

* Topic assignments.
* Topic metadata.
* Dynamic topic outputs.
* Emotion labels.
* Emotion probabilities.
* Temporal information.

This separation allows the presentation layer to evolve independently from the underlying NLP models.



# 17. Key Documentation Map

`ARCHITECTURE.md` is intended to provide the starting point for understanding the system. Readers who want to go deeper should use the following documentation:

### Core pipeline

**[NLP Pipeline (`pipeline.md`)](pipeline.md)**

The most important technical document for understanding how ClimateLens processes data end-to-end. It covers:

* Data preprocessing.
* Embeddings.
* Topic modeling.
* Dynamic topic modeling.
* Emotion classification.
* Visualization outputs.
* Model comparisons.
* Performance considerations.
* Configuration.
* Local execution.
* AzureML execution.

### Data preparation

**[Preprocessing (`preprocessing.md`)](preprocessing.md)**

Provides the detailed source-data and text-normalization workflow for Reddit and Twitter, including:

* Extraction.
* Climate-related filtering.
* Deduplication.
* Text cleaning.
* Stopword construction.
* Tokenization.
* Derived fields.

### Dataset structure

**[Data Schema (`data_schema.md`)](data_schema.md)**

Defines the authoritative schema for:

* Reddit data.
* Twitter data.
* Timestamps.
* `cleaned_text`.
* Derived fields.

This is the best reference when working directly with ClimateLens datasets.

### Visualization

**[Visualization (`visualization.md`)](visualization.md)**

Documents the current visualization designs and proposed improvements, including:

* Intertopic maps.
* Topic bar charts.
* Topic hierarchies.
* Emotion distributions.
* Emotion probabilities.
* Visualization usability considerations.

This document is still evolving as the visualization layer is refined.

### AzureML infrastructure

**[AzureML Configuration (`azure_configuration.md`)](azure_configuration.md)**

Documents the AzureML environment and infrastructure, including:

* Workspace information.
* Compute setup.
* Dependency management.
* Storage considerations.
* Jupyter/kernel configuration.
* Common AzureML troubleshooting issues.

### Source code

The main implementation areas corresponding to this architecture are:

```text
src/
├── config/
│   └── datasets.yaml
│
├── data_preprocessing.py
├── embeddings.py
├── topic_modeling.py
├── dynamic_topic_modeling.py
├── emotion_classification.py
└── emotion_visualizations.py

azureml/
├── AML_job.py
└── run_scripts.sh
```



# 18. Putting It All Together

The ClimateLens architecture can ultimately be understood as a sequence of transformations:

```text
              SOCIAL-MEDIA DATA
               Reddit / Twitter
                      │
                      ▼
             DATA PREPARATION
        Extraction + filtering + schema
                      │
                      ▼
                PREPROCESSING
          Cleaning + normalization
                      │
                      ▼
                CLEANED TEXT
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     TOPIC ANALYSIS          EMOTION ANALYSIS
          │                       │
   ┌──────┴──────┐                │
   │             │                │
   ▼             ▼                ▼
Embeddings   Dynamic Topics   Transformer
   │             │                │
   ▼             │                ▼
UMAP/HDBSCAN     │          Emotion Labels
   │             │                │
   ▼             │                │
 BERTopic        │                │
   │             │                │
   └──────┬──────┘                │
          │                       │
          └───────────┬───────────┘
                      ▼
                 VISUALIZATION
                      │
                      ▼
             HUMAN INTERPRETATION
```

The architecture deliberately separates **data preparation**, **NLP modeling**, **visualization**, and **execution infrastructure**.

This separation makes it possible to change one part of the system—for example, introducing a new emotion model or preprocessing rule—without requiring the entire application to be redesigned.

The result is a dataset-driven architecture that can support exploratory research into **what people discuss about climate, how they emotionally respond, and how both topics and emotional expression change over time**.