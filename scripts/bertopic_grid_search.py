"""
BERTopic grid search with n-gram experimentation and dataset-size ablation.

Environment variables:
    DATA_DIR: directory containing input CSV files (default: ./data)
    CODE_DIR: project/output root (default: current working directory)

Each CSV must contain a text column named "body" or "text".
Results are saved as CSV files; fitted BERTopic objects are not serialized.
"""

import logging
import os
import re
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).expanduser()
CODE_DIR = Path(os.getenv("CODE_DIR", PROJECT_ROOT)).expanduser()
OUTPUT_DIR = CODE_DIR / "testing" / "topic_modeling"

# Keep the grid intentionally small initially. Expand only after a pilot run.
EMBEDDING_MODELS = {
    "minilm-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-L12": "sentence-transformers/all-MiniLM-L12-v2",
}

# One tuple per CountVectorizer instance; each tuple is run as a separate experiment.
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (2, 3)]

# Small/large ablation: the large sample is capped to control runtime.
# If a dataset contains fewer rows, all available rows are used.
DATASET_SIZE_LIMITS = {
    "small": 500,
    "large": 5000,
}

UMAP_GRID = {
    "n_neighbors": [3, 8],
    "n_components": [5, 8],
}
HDBSCAN_GRID = {
    "min_cluster_size": [5, 8],
    "min_samples": [5, 8],
}
BERTOPIC_GRID = {
    "min_topic_size": [5, 8],
    "mmr_value": [0.3],
}
FIXED_PARAMS = {
    "nr_topics": "auto",
    "top_n_words": 10,
}


def load_docs(data_dir, text_cols=("body", "text")):
    """Load CSV text columns, skipping files that have no supported text column."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    docs = {}
    for fp in sorted(data_dir.glob("*.csv")):
        name = re.sub(r"(_clean|_filtered)?\.csv$", "", fp.name)
        df = pd.read_csv(fp)
        col = next((candidate for candidate in text_cols if candidate in df.columns), None)
        if col is None:
            LOGGER.warning("Skipping %s: no text column among %s", fp.name, text_cols)
            continue

        texts = df[col].dropna().astype(str).str.strip()
        texts = texts[texts.ne("")]
        if texts.empty:
            LOGGER.warning("Skipping %s: text column has no non-empty rows", fp.name)
            continue

        docs[name] = texts.tolist()
        LOGGER.info("Loaded %s: %s documents", name, f"{len(docs[name]):,}")

    if not docs:
        raise ValueError(f"No usable CSV datasets found in {data_dir}")
    return docs


def sample_docs(docs, limit, random_state=42):
    """Return a deterministic random sample, preserving all rows below the limit."""
    if len(docs) <= limit:
        return list(docs)
    # Shuffle the full list before slicing. Because the seed is fixed, the
    # small sample is a subset of the large sample for the same dataset.
    shuffled = pd.Series(docs).sample(frac=1, random_state=random_state).tolist()
    return shuffled[:limit]


def compute_embeddings(docs, embedding_path, batch_size=32):
    """Compute document embeddings for one dataset/model combination."""
    from sentence_transformers import SentenceTransformer

    LOGGER.info("Computing embeddings with %s for %s documents", embedding_path, len(docs))
    model = SentenceTransformer(embedding_path)
    return model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def run_experiment(
    docs,
    embeddings,
    embedding_name,
    dataset_name,
    dataset_size_label,
    ngram_range,
    umap_params,
    hdbscan_params,
    bertopic_params,
):
    """Fit one BERTopic configuration and return scalar metrics plus the model."""
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        min_df=0.01,
        max_df=0.95,
    )
    umap_model = UMAP(
        metric="cosine",
        random_state=42,
        **umap_params,
    )
    hdbscan_model = HDBSCAN(
        metric="euclidean",
        prediction_data=True,
        **hdbscan_params,
    )
    representation_model = MaximalMarginalRelevance(diversity=bertopic_params["mmr_value"])

    bertopic_params_for_model = {
        key: value for key, value in bertopic_params.items() if key != "mmr_value"
    }
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        top_n_words=FIXED_PARAMS["top_n_words"],
        nr_topics=FIXED_PARAMS["nr_topics"],
        **bertopic_params_for_model,
    )

    start = time.time()
    topics, _ = topic_model.fit_transform(docs, embeddings)
    runtime = time.time() - start

    n_topics = len(set(topics) - {-1})
    outlier_fraction = topics.count(-1) / len(topics) if topics else 0.0

    # Include the range as two numeric columns for easier sorting/filtering.
    result = {
        "dataset": dataset_name,
        "dataset_size_label": dataset_size_label,
        "n_documents": len(docs),
        "embedding_model": embedding_name,
        "ngram_min": ngram_range[0],
        "ngram_max": ngram_range[1],
        "ngram_range": str(ngram_range),
        "n_topics": n_topics,
        "outlier_frac": round(outlier_fraction, 4),
        "runtime_sec": round(runtime, 2),
        "mmr_diversity": representation_model.diversity,
        **umap_params,
        **hdbscan_params,
        **bertopic_params,
    }
    return result, topic_model


def grid_search(docs_dict, output_dir=OUTPUT_DIR):
    """
    Run the grid for every dataset, sample-size condition, embedding model,
    n-gram range, and valid UMAP/HDBSCAN/BERTopic parameter combination.

    Saves:
      - bertopic_grid_results_<timestamp>.csv: all successful experiments
      - bertopic_ngram_ablation_<timestamp>.csv: same results grouped so that
        n-gram effects can be compared within dataset/size/embedding conditions
      - bertopic_grid_errors_<timestamp>.csv: failed configurations and errors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    fitted_models = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_name, all_docs in docs_dict.items():
        for size_label, limit in DATASET_SIZE_LIMITS.items():
            docs = sample_docs(all_docs, limit)
            LOGGER.info(
                "Dataset=%s | condition=%s | documents=%s",
                dataset_name,
                size_label,
                len(docs),
            )

            for embedding_name, embedding_path in EMBEDDING_MODELS.items():
                try:
                    embeddings = compute_embeddings(docs, embedding_path)
                except Exception as exc:
                    LOGGER.exception("Embedding failed for %s / %s", dataset_name, embedding_name)
                    errors.append(
                        {
                            "dataset": dataset_name,
                            "dataset_size_label": size_label,
                            "embedding_model": embedding_name,
                            "ngram_range": None,
                            "error": repr(exc),
                        }
                    )
                    continue

                for ngram_range in NGRAM_RANGES:
                    for umap_values in product(*UMAP_GRID.values()):
                        umap_params = dict(zip(UMAP_GRID.keys(), umap_values))
                        for hdb_values in product(*HDBSCAN_GRID.values()):
                            hdbscan_params = dict(zip(HDBSCAN_GRID.keys(), hdb_values))
                            for bertopic_values in product(*BERTOPIC_GRID.values()):
                                bertopic_params = dict(zip(BERTOPIC_GRID.keys(), bertopic_values))

                                # Avoid configurations where the topic minimum is smaller
                                # than the cluster minimum.
                                if (
                                    bertopic_params["min_topic_size"]
                                    < hdbscan_params["min_cluster_size"]
                                ):
                                    continue

                                try:
                                    LOGGER.info(
                                        "Running dataset=%s size=%s embedding=%s ngram=%s "
                                        "UMAP=%s HDBSCAN=%s BERTopic=%s",
                                        dataset_name,
                                        size_label,
                                        embedding_name,
                                        ngram_range,
                                        umap_params,
                                        hdbscan_params,
                                        bertopic_params,
                                    )
                                    result, model = run_experiment(
                                        docs=docs,
                                        embeddings=embeddings,
                                        embedding_name=embedding_name,
                                        dataset_name=dataset_name,
                                        dataset_size_label=size_label,
                                        ngram_range=ngram_range,
                                        umap_params=umap_params,
                                        hdbscan_params=hdbscan_params,
                                        bertopic_params=bertopic_params,
                                    )
                                    results.append(result)
                                    # Keep models in memory only; do not put Python objects in CSV.
                                    fitted_models[len(results) - 1] = model
                                except Exception as exc:
                                    LOGGER.exception(
                                        "Experiment failed: dataset=%s size=%s embedding=%s ngram=%s",
                                        dataset_name,
                                        size_label,
                                        embedding_name,
                                        ngram_range,
                                    )
                                    errors.append(
                                        {
                                            "dataset": dataset_name,
                                            "dataset_size_label": size_label,
                                            "embedding_model": embedding_name,
                                            "ngram_range": str(ngram_range),
                                            "umap_params": str(umap_params),
                                            "hdbscan_params": str(hdbscan_params),
                                            "bertopic_params": str(bertopic_params),
                                            "error": repr(exc),
                                        }
                                    )

    results_df = pd.DataFrame(results)
    results_path = output_dir / f"bertopic_grid_results_{timestamp}.csv"
    ablation_path = output_dir / f"bertopic_ngram_ablation_{timestamp}.csv"
    errors_path = output_dir / f"bertopic_grid_errors_{timestamp}.csv"

    results_df.to_csv(results_path, index=False)

    if not results_df.empty:
        # One row per n-gram configuration and condition. Means summarize the
        # remaining parameter combinations; std shows sensitivity to the grid.
        group_columns = [
            "dataset",
            "dataset_size_label",
            "n_documents",
            "embedding_model",
            "ngram_min",
            "ngram_max",
            "ngram_range",
        ]
        ablation_df = (
            results_df.groupby(group_columns, dropna=False)
            .agg(
                experiments=("runtime_sec", "count"),
                mean_topics=("n_topics", "mean"),
                mean_outlier_frac=("outlier_frac", "mean"),
                std_outlier_frac=("outlier_frac", "std"),
                mean_runtime_sec=("runtime_sec", "mean"),
            )
            .reset_index()
            .sort_values(["dataset", "dataset_size_label", "embedding_model", "mean_outlier_frac"])
        )
    else:
        ablation_df = pd.DataFrame()
    ablation_df.to_csv(ablation_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    LOGGER.info("Saved grid results: %s", results_path)
    LOGGER.info("Saved n-gram ablation: %s", ablation_path)
    LOGGER.info("Saved errors: %s", errors_path)

    if not results_df.empty:
        LOGGER.info(
            "Best candidates (lowest mean-independent outlier fraction, then most topics):\n%s",
            results_df.sort_values(["outlier_frac", "n_topics"], ascending=[True, False])
            .head(10)
            .to_string(index=False),
        )
    else:
        LOGGER.warning("No experiments completed successfully. Inspect %s", errors_path)

    return results_df, ablation_df, fitted_models, results_path, ablation_path, errors_path


if __name__ == "__main__":
    datasets = load_docs(DATA_DIR)
    grid_results, ngram_ablation, fitted_models, results_path, ablation_path, errors_path = (
        grid_search(datasets, OUTPUT_DIR)
    )
