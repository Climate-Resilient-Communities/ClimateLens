"""
Dynamic Topic Modeling Script

This script performs topic modeling with temporal analysis using BERTopic.
It processes datasets, computes embeddings, trains topic models, and generates
dynamic visualizations showing how topics evolve over time.

Usage:
    python src/dynamic_topic_modeling.py
"""

import os
import re
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


def load_environment():
    """Load environment variables from .env file."""
    env_path = Path(__file__).resolve().parent.parent / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        print("Loading environment variables")
        data_dir = os.getenv("DATA_DIR")
        code_dir = os.getenv("CODE_DIR")
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    return data_dir, code_dir


def process_datasets(data_path, text_cols=('body', 'text')):
    """
    Load CSV datasets from specified directory.

    Args:
        data_path: Directory containing CSV files
        text_cols: Tuple of possible text column names to look for

    Returns:
        dfs: Dictionary of dataframes {dataset_name: dataframe}
        docs_dict: Dictionary of document lists {dataset_name: [doc1, doc2, ...]}
        datasets: Dictionary mapping names to file paths
    """
    datasets, dfs, docs_dict, failed = {}, {}, {}, []
    data_path = Path(data_path)

    for file_path in data_path.glob("*.csv"):
        name = re.sub(r"(_clean|filtered_)?\.csv$", "", file_path.stem)
        datasets[name] = file_path

        try:
            df = pd.read_csv(file_path)
            text_col = next((c for c in text_cols if c in df.columns), None)

            if not text_col:
                print(f"Skipping {name}. No {text_cols} column found.")
                failed.append(name)
                continue

            dfs[name] = df
            docs_dict[name] = df[df[text_col].notna()][text_col].tolist()
            print(f"Loaded {name} ({len(dfs[name])} rows) from: {file_path}")

        except Exception as e:
            print(f"Error loading {name}: {e}")
            failed.append(name)

    print(f"{len(dfs)}/{len(datasets)} datasets loaded successfully")
    if failed:
        print(f"Failed to load: {', '.join(failed)}")

    return dfs, docs_dict, datasets


def create_directories(code_dir):
    """
    Create output directories for models, visualizations, and DTM results.

    Args:
        code_dir: Base directory for outputs

    Returns:
        Tuple of (model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir)
    """
    directories = {
        "models": Path(code_dir) / "models",
        "IDM": Path(code_dir) / "visualizations" / "IDM",
        "hierarchies": Path(code_dir) / "visualizations" / "hierarchies",
        "barcharts": Path(code_dir) / "visualizations" / "barcharts",
        "dtm": Path(code_dir) / "visualizations" / "dtm",
    }

    for path in directories.values():
        os.makedirs(path, exist_ok=True)

    model_dir = directories["models"]
    IDM_dir = directories["IDM"]
    hierarchy_dir = directories["hierarchies"]
    barchart_dir = directories["barcharts"]
    dtm_dir = directories["dtm"]

    return model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir


def compute_embeddings(docs_dict):
    """
    Compute embeddings for all documents using sentence transformers.
    This function should only be called ONCE per pipeline run.

    Args:
        docs_dict: Dictionary of {dataset_name: list_of_documents}

    Returns:
        embeddings_dict: Dictionary of {dataset_name: embeddings_array}
    """
    embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings_dict = {}

    print("\nComputing embeddings for all datasets...")
    for name, docs in docs_dict.items():
        print(f'Computing {name} embeddings...')
        embeddings_dict[name] = embedding_model.encode(docs, show_progress_bar=True)
        print(f'{name} embeddings complete')

    return embeddings_dict


def create_submodels(params=None):
    """
    Create BERTopic sub-models (vectorizer, UMAP, HDBSCAN, representation).

    Args:
        params: Dictionary of hyperparameters

    Returns:
        Tuple of (vectorizer_model, umap_model, hdbscan_model, representation_model)
    """
    params = params or {
        "min_df": 0.05,
        "max_df": 0.9,
        "n_neighbors": 6,
        "min_cluster_size": 7,
        "min_topic_size": 7,
    }

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=params["min_df"],
        max_df=params["max_df"]
    )

    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=5,
        metric='cosine',
        low_memory=False,
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        metric='euclidean',
        prediction_data=True
    )

    mmr_model = MaximalMarginalRelevance(diversity=0.1)
    representation_model = mmr_model

    return vectorizer_model, umap_model, hdbscan_model, representation_model


def bert_model(dataset_name, docs, embeddings, params=None):
    """
    Create and fit a BERTopic model.

    Args:
        dataset_name: Name of the dataset
        docs: List of documents
        embeddings: Pre-computed embeddings
        params: Hyperparameter dictionary

    Returns:
        Tuple of (topic_model, topics, probs)
    """
    if not docs:
        print(f"No docs provided for {dataset_name}. Skipping topic modeling.")
        return None, None, None

    params = params or {}
    vectorizer_model, umap_model, hdbscan_model, representation_model = create_submodels(params)

    embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    print(f"\nTopic modeling for {dataset_name}...")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=params.get("min_topic_size", 7),
        nr_topics="auto",
    )

    start_time = time.time()
    try:
        topics, probs = topic_model.fit_transform(docs, embeddings)
        return topic_model, topics, probs
    except Exception as e:
        print(f"Error during {dataset_name} topic modeling: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        end_time = time.time()
        elapsed_hours = (end_time - start_time) / 3600
        print(f"{dataset_name} completed in {elapsed_hours:.3f} hours")


def prepare_timestamps(dfs, name):
    """
    Extract timestamps from dataset for dynamic topic modeling.

    Args:
        dfs: Dictionary of dataframes
        name: Dataset name

    Returns:
        List of datetime objects
    """
    df = dfs[name]

    # Try different timestamp column names
    if 'created_at' in df.columns:
        timestamps = pd.to_datetime(df['created_at']).tolist()
    elif 'created_utc' in df.columns:
        timestamps = pd.to_datetime(df['created_utc'], unit='s').tolist()
    elif 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp']).tolist()
    else:
        print(f"Warning: No timestamp column found for {name}. DTM will be skipped.")
        return None

    print(f"{name} - {len(timestamps)} documents")
    print(f"Date range: {min(timestamps)} to {max(timestamps)}")

    return timestamps


def run_dynamic_topic_modeling(topic_model, docs, timestamps, dataset_name, nr_bins=50, top_n_topics=10):
    """
    Perform dynamic topic modeling and generate visualization.

    Args:
        topic_model: Fitted BERTopic model
        docs: List of documents
        timestamps: List of datetime objects
        dataset_name: Name of the dataset
        nr_bins: Number of time bins to divide data into
        top_n_topics: Number of top topics to visualize

    Returns:
        Tuple of (topics_over_time DataFrame, plotly figure)
    """
    if timestamps is None:
        print(f"Skipping DTM for {dataset_name} - no timestamps available")
        return None, None

    print(f"\nAnalyzing topics over time for {dataset_name}...")

    try:
        topics_over_time = topic_model.topics_over_time(
            docs,
            timestamps,
            nr_bins=nr_bins
        )

        print(f"DTM analysis complete. Creating visualization...")

        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=top_n_topics
        )

        return topics_over_time, fig

    except Exception as e:
        print(f"Error during DTM for {dataset_name}: {e}")
        traceback.print_exc()
        return None, None


def save_dtm_outputs(dataset_name, topics_over_time, fig, dtm_dir):
    """
    Save dynamic topic modeling outputs (CSV and HTML visualization).

    Args:
        dataset_name: Name of the dataset
        topics_over_time: DataFrame with topics over time
        fig: Plotly figure
        dtm_dir: Directory to save outputs
    """
    if topics_over_time is None or fig is None:
        return

    # Save topics_over_time data
    csv_path = Path(dtm_dir) / f"{dataset_name}_topics_over_time.csv"
    topics_over_time.to_csv(csv_path, index=False)
    print(f"Saved DTM data: {csv_path}")

    # Save interactive visualization
    html_path = Path(dtm_dir) / f"{dataset_name}_topics_over_time.html"
    fig.write_html(str(html_path))
    print(f"Saved DTM visualization: {html_path}")


def save_model(topic_model, dataset_name, model_dir):
    """
    Save BERTopic model to disk.

    Args:
        topic_model: Fitted BERTopic model
        dataset_name: Name of the dataset
        model_dir: Directory to save model
    """
    save_path = Path(model_dir) / f"{dataset_name}_dtm.safetensors"
    topic_model.save(str(save_path), serialization="safetensors")
    print(f"Model saved: {save_path}")


def save_static_visualizations(topic_model, dataset_name, IDM_dir, hierarchy_dir, barchart_dir, top_n_topics=10):
    """
    Save static BERTopic visualizations (IDM, hierarchy, barchart).

    Args:
        topic_model: Fitted BERTopic model
        dataset_name: Name of the dataset
        IDM_dir: Directory for IDM visualizations
        hierarchy_dir: Directory for hierarchy visualizations
        barchart_dir: Directory for barchart visualizations
        top_n_topics: Number of topics to show in barchart
    """
    try:
        print(f"\nGenerating static visualizations for {dataset_name}...")

        figure_hierarchy = topic_model.visualize_hierarchy()
        figure_topics = topic_model.visualize_topics()
        figure_barchart = topic_model.visualize_barchart(top_n_topics=top_n_topics, n_words=10)

        figure_hierarchy.write_html(str(Path(hierarchy_dir) / f"{dataset_name}_hierarchy.html"))
        figure_topics.write_html(str(Path(IDM_dir) / f"{dataset_name}_IDM.html"))
        figure_barchart.write_html(str(Path(barchart_dir) / f"{dataset_name}_barchart.html"))

        print(f"Static visualizations saved for {dataset_name}")

    except Exception as e:
        print(f"Error saving visualizations for {dataset_name}: {e}")
        traceback.print_exc()


def main():
    """
    Main pipeline function that orchestrates the entire DTM workflow:
    1. Load environment variables
    2. Load datasets
    3. Create output directories
    4. Compute embeddings (once!)
    5. Train BERTopic models
    6. Save models and static visualizations
    7. Run dynamic topic modeling
    8. Save DTM outputs
    """
    print("="*60)
    print("Dynamic Topic Modeling Pipeline")
    print("="*60)

    # Step 1: Load environment
    data_dir, code_dir = load_environment()
    if not data_dir or not code_dir:
        raise EnvironmentError("DATA_DIR and CODE_DIR must be set in the .env file.")

    # Step 2: Load datasets
    print("\n[1/7] Loading datasets...")
    dfs, docs_dict, datasets = process_datasets(data_dir)

    if not dfs:
        print("No datasets loaded. Exiting.")
        return

    # Step 3: Create directories
    print("\n[2/7] Creating output directories...")
    model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir = create_directories(code_dir)

    # Step 4: Compute embeddings (ONCE!)
    print("\n[3/7] Computing embeddings...")
    embeddings_dict = compute_embeddings(docs_dict)

    # Step 5: Train BERTopic models
    print("\n[4/7] Training BERTopic models...")
    topic_models = {}

    for name, docs in docs_dict.items():
        # Set dataset-specific parameters
        if 'twitter' in name.lower():
            params = {
                "min_df": 0.05, "max_df": 0.90, "n_neighbors": 5,
                "min_cluster_size": 5, "min_topic_size": 5
            }
        else:
            params = {
                "min_df": 0.05, "max_df": 0.90, "n_neighbors": 6,
                "min_cluster_size": 7, "min_topic_size": 7
            }

        topic_model, topics, probs = bert_model(
            dataset_name=name,
            docs=docs,
            embeddings=embeddings_dict[name],
            params=params
        )

        if topic_model is None:
            print(f"Skipping {name} - model training failed")
            continue

        topic_models[name] = topic_model
        print(f"{name}: Found {len(set(topics))} topics")

    # Step 6: Save models and static visualizations
    print("\n[5/7] Saving models and static visualizations...")
    for name, topic_model in topic_models.items():
        save_model(topic_model, name, model_dir)
        save_static_visualizations(topic_model, name, IDM_dir, hierarchy_dir, barchart_dir)

    # Step 7: Run dynamic topic modeling
    print("\n[6/7] Running dynamic topic modeling...")
    for name, topic_model in topic_models.items():
        timestamps = prepare_timestamps(dfs, name)

        if timestamps is None:
            continue

        # Determine number of bins based on time range
        time_range = (max(timestamps) - min(timestamps)).days
        nr_bins = min(50, max(10, time_range // 30))  # ~1 bin per month, max 50

        topics_over_time, fig = run_dynamic_topic_modeling(
            topic_model=topic_model,
            docs=docs_dict[name],
            timestamps=timestamps,
            dataset_name=name,
            nr_bins=nr_bins,
            top_n_topics=10
        )

        # Step 8: Save DTM outputs
        if topics_over_time is not None:
            save_dtm_outputs(name, topics_over_time, fig, dtm_dir)

    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print(f"Outputs saved to: {code_dir}")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "="*60)
        print("ERROR: Pipeline failed")
        print("="*60)
        traceback.print_exc()
