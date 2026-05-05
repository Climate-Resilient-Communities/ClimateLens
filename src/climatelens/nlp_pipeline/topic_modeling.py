import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

warnings.filterwarnings("ignore")

# Dataset-specific configs
DATASET_PARAMS: Dict[str, Dict[str, Any]] = {
    "twitter": {
        "min_df": 0.05,
        "max_df": 0.90,
        "n_neighbors": 5,
        "min_cluster_size": 5,
        "min_samples": 5,
        "min_topic_size": 5,
        "nr_topics": "auto",
    },  # more aggressive clustering for shorter texts.
    "reddit": {
        "min_df": 0.05,
        "max_df": 0.90,
        "n_neighbors": 15,
        "min_cluster_size": 70,
        "min_samples": 10,
        "min_topic_size": 100,
        "nr_topics": "auto",
    },
    "reddit_small": {
        "min_df": 0.01,
        "max_df": 0.95,
        "n_neighbors": 3,
        "min_cluster_size": 3,
        "min_samples": 3,
        "min_topic_size": 15,
        "nr_topics": "auto",  # experiment with nr_topics=params.get("nr_topics", "auto") for testing
    },  # fallback for visualization error (less than 4 topics)
}


def create_submodels(
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[CountVectorizer, UMAP, HDBSCAN, MaximalMarginalRelevance]:
    params = params or {
        "min_df": 0.05,
        "max_df": 0.9,
        "n_neighbors": 6,
        "min_cluster_size": 7,
        "min_topic_size": 7,
    }

    # Twitter-specific stopwords to filter artifacts that slip through preprocessing
    twitter_stopwords = [
        "https",
        "http",
        "co",
        "rt",
        "amp",
        "t",
        "www",
        "url",
        "pic",
        "twitter",
        "com",
        "ru",
        "tldrs",
    ]  # do we really need this? we can take these out much earlier

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=params["min_df"],
        max_df=params["max_df"],
        stop_words=twitter_stopwords,  # Fallback to catch any Twitter artifacts
    )  # look into removing the stopwords

    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=5,
        metric="cosine",
        low_memory=False,
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean",
        prediction_data=True,  # need to check this param
    )

    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    representation_model = mmr_model

    return vectorizer_model, umap_model, hdbscan_model, representation_model


def bert_model(
    dataset_name: str,
    docs: List[str],
    embeddings: Optional[NDArray[np.float32]],
    embedding_model: SentenceTransformer,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[BERTopic], Optional[List[int]], Optional[List[float]]]:
    if not docs:
        print(f"No docs provided for {dataset_name}. Skipping topic modeling.")
        return None, None, None

    params = params or {}
    vectorizer_model, umap_model, hdbscan_model, representation_model = create_submodels(params)

    print(f"Topic modeling for {dataset_name}...")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=params.get("min_topic_size", 7),
        nr_topics=params["nr_topics"],  # dealing with the ValueError: zero-size array
    )

    start_time = time.time()
    try:
        topics, probs = topic_model.fit_transform(docs, embeddings)
        return topic_model, topics, probs
    except ValueError as e:
        if "After pruning, no terms remain" in str(e):
            raise
        return None, None, None
    except Exception as e:
        print(f"Error during {dataset_name} topic modeling: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        print(f"{dataset_name} topic modeling completed in {elapsed_seconds:.3f} seconds")


def save_and_reload_model(name: str, model_dir: Path, topic_models: Dict[str, BERTopic]) -> None:
    save_path = Path(model_dir) / f"{name}.safetensors"
    topic_models[name].save(str(save_path), serialization="safetensors")
    print(f"Model saved: {save_path}")


def save_dataframe_inplace(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_csv(path, index=False)
        print(f"Saved updated dataframe back to {path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
