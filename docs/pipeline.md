# Pipeline Reference

ClimateLens runs a 4-stage data pipeline. Each stage is a standalone script
under `src/` that reads one directory and writes to another - nothing is ever
written back over its input.

```
 DATA_DIR (read-only)
      │
      ▼
 data_preprocessing.py   ──►  PROCESSED_DATA_DIR
      │                                 │
      │                                 ▼
      │                       topic_modeling.py   (writes to PROCESSED_DATA_DIR)
      │                                 │
      ▼                                 ▼
 emotion_classification.py  ──►  OUTPUT_DATA_DIR
                                        │
                                        ▼
                          emotion_visualizations.py  ──►  OUTPUT_VIS_DIR
```

## Stage 1 - `data_preprocessing.py`

| | |
|---|---|
| **Input**  | `DATA_DIR/*.csv` matching any pattern in `src/config/datasets.yaml` |
| **Output** | `PROCESSED_DATA_DIR/<dataset>.csv` with an added `cleaned_text` column |
| **Env**    | `DATA_DIR`, `PROCESSED_DATA_DIR` |
| **Side effects** | Downloads NLTK corpora on first run |

Applies Twitter-style cleaning (URLs, @handles, `RT`, HTML entities),
lowercases, tokenizes, removes stopwords + swear variants while preserving
negations, and drops documents under `MIN_DOCUMENT_WORDS` (3) tokens.

## Stage 2 - `topic_modeling.py`

| | |
|---|---|
| **Input**  | `PROCESSED_DATA_DIR/*.csv` (falls back to `DATA_DIR` if empty) |
| **Output** | `PROCESSED_DATA_DIR/<dataset>.csv` (topic columns added) and `MODELS_DIR/<dataset>.safetensors` |
| **Env**    | `PROCESSED_DATA_DIR`, `MODELS_DIR`, `OUTPUT_VIS_DIR`, optional `COHERE_API_KEY` |
| **Visualizations** | `OUTPUT_VIS_DIR/{IDM,hierarchies,barcharts,dtm}/<dataset>*.html` |

Uses BERTopic with `sentence-transformers/all-MiniLM-L12-v2` embeddings, UMAP
(seeded) + HDBSCAN clustering, MaximalMarginalRelevance for representation,
and optionally Cohere `command-r` for topic labels.
Hyperparameters come from `DATASET_PARAMS` keyed on the dataset's
`topic_profile` in the YAML registry.

## Stage 3 - `emotion_classification.py`

| | |
|---|---|
| **Input**  | `PROCESSED_DATA_DIR/*.csv` (falls back to `DATA_DIR`) |
| **Output** | `OUTPUT_DATA_DIR/<dataset>_with_emotions.csv` |
| **Env**    | `PROCESSED_DATA_DIR`, `OUTPUT_DATA_DIR`, `OUTPUT_VIS_DIR` |

Picks the emotion model per dataset from `emotion_profile` in the registry:

- `twitter`  → `boltuix/bert-emotion` (13 labels)
- `reddit`  → `SamLowe/roberta-base-go_emotions` (28 labels)
- `default` → `SamLowe/roberta-base-go_emotions`

Writes a simple `<dataset>_emotion_counts.png` bar chart to `OUTPUT_VIS_DIR`.

## Stage 4 - `emotion_visualizations.py`

| | |
|---|---|
| **Input**  | `OUTPUT_DATA_DIR/*.csv` (emotion-classified; falls back to `DATA_DIR`) |
| **Output** | `OUTPUT_VIS_DIR/emotions/{wordclouds,timeseries}/` + summary HTML |
| **Env**    | `OUTPUT_DATA_DIR`, `OUTPUT_VIS_DIR` |

If `emotion_label` is already present on the input (i.e. stage 3 ran), this
stage skips re-classification and reuses those labels. Otherwise it loads
`j-hartmann/emotion-english-distilroberta-base` and classifies on the fly.

## Adding a new dataset

1. Add an entry to `src/config/datasets.yaml` with the canonical `name`,
   a `filename_patterns` list that matches the raw CSV, and the columns +
   profiles the pipeline should use.
2. Drop the raw CSV into `DATA_DIR`.
3. Re-run the pipeline. Stages discover the new dataset automatically.

## Running locally

```bash
cp .env.example .env          # edit paths if you want non-default locations
pip install -r requirements.txt
make pipeline                 # or: python src/data_preprocessing.py && ...
```

## Running in AzureML

`azureml/AML_job.py` submits the whole pipeline to an AzureML cluster via
`azureml/run_scripts.sh`. AzureML detection is automatic: when any of
`AZUREML_RUN_ID` / `AZUREML_EXPERIMENT_ID` / `AZUREML_OUTPUT_DIR` is set,
the pipeline writes to `./outputs/...` instead of the local default layout.
