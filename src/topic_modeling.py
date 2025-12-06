import os
import re
import time
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

#!pip install -q bertopic sentence-transformers umap-learn hdbscan #cohere
#import cohere
#from bertopic.representation import Cohere
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

#warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset-specific hyperparameters for BERTopic modeling.
# Twitter data uses more aggressive clustering (smaller min sizes) due to shorter texts.
# The "default" configuration applies to all other datasets (e.g., Reddit, news articles).
DATASET_PARAMS = {
    "twitter": {
        "min_df": 0.05,
        "max_df": 0.90,
        "n_neighbors": 5,
        "min_cluster_size": 5,
        "min_topic_size": 5,
        "nr_topics": 20
    },
    "default": {
        "min_df": 0.05,
        "max_df": 0.90,
        "n_neighbors": 6,
        "min_cluster_size": 7,
        "min_topic_size": 7,
        "nr_topics": 30
    }
}


# =============================================================================
# DYNAMIC TOPIC MODELING FUNCTIONS
# =============================================================================

def prepare_timestamps(dfs, name):
    """
    Extract and validate timestamps from a dataset.
    
    Handles different timestamp formats:
    - Twitter: 'created_at' column with datetime strings
    - Reddit: 'created_utc' column with Unix timestamps
    
    Args:
        dfs: Dictionary of dataframes
        name: Dataset name (key in dfs)
    
    Returns:
        List of datetime objects corresponding to each document, or None if no timestamps found
    """
    df = dfs[name]
    timestamps = None
    
    # Possible timestamp column names (prioritized)
    timestamp_cols = ['created_utc', 'created_at', 'timestamp', 'date', 'datetime']
    
    found_col = None
    for col in timestamp_cols:
        if col in df.columns:
            found_col = col
            break
    
    if not found_col:
        print(f"  No timestamp column found for {name}. Checked: {timestamp_cols}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Skipping Dynamic Topic Modeling for {name}.")
        return None
    
    print(f" Found timestamp column '{found_col}' for {name}")
    
    try:
        if found_col == 'created_utc':
            # Reddit uses Unix timestamps (seconds since epoch)
            timestamps = pd.to_datetime(df[found_col], unit='s', errors='coerce')
        elif found_col == 'created_at':
            # Twitter uses datetime strings
            timestamps = pd.to_datetime(df[found_col], errors='coerce')
        else:
            # Try automatic parsing for other column names
            timestamps = pd.to_datetime(df[found_col], errors='coerce')
        
        # Convert to Python datetime objects (list)
        timestamps = timestamps.tolist()
        
        # Validate timestamps
        valid_timestamps = [t for t in timestamps if pd.notna(t)]
        invalid_count = len(timestamps) - len(valid_timestamps)
        
        if invalid_count > 0:
            print(f"     {invalid_count}/{len(timestamps)} timestamps could not be parsed")
        
        if len(valid_timestamps) == 0:
            print(f"    No valid timestamps found for {name}. Skipping DTM.")
            return None
        
        # Print time range info
        min_date = min(valid_timestamps)
        max_date = max(valid_timestamps)
        time_span_days = (max_date - min_date).days
        
        print(f"    Time range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"    Time span: {time_span_days} days (~{time_span_days/365:.1f} years)")
        
        # Check if time span is meaningful for DTM
        if time_span_days < 7:
            print(f"     Time span less than 1 week. DTM may not be meaningful.")
        
        return timestamps
        
    except Exception as e:
        print(f"    Error parsing timestamps for {name}: {e}")
        traceback.print_exc()
        return None


def calculate_optimal_bins(timestamps, min_bins=10, max_bins=50):
    """
    Calculate optimal number of temporal bins based on data time span.
    
    Heuristic: ~1 bin per month, bounded by min/max limits.
    
    Args:
        timestamps: List of datetime objects
        min_bins: Minimum number of bins
        max_bins: Maximum number of bins
    
    Returns:
        Integer number of bins
    """
    valid_timestamps = [t for t in timestamps if pd.notna(t)]
    
    if len(valid_timestamps) < 2:
        return min_bins
    
    min_date = min(valid_timestamps)
    max_date = max(valid_timestamps)
    time_span_days = (max_date - min_date).days
    
    # Roughly 1 bin per month (30 days)
    suggested_bins = max(1, time_span_days // 30)
    
    # Clamp to min/max bounds
    optimal_bins = max(min_bins, min(max_bins, suggested_bins))
    
    return optimal_bins


def perform_dynamic_topic_modeling(topic_model, docs, timestamps, name, nr_bins=None, top_n_topics=10):
    """
    Perform Dynamic Topic Modeling analysis using BERTopic's topics_over_time.
    
    Args:
        topic_model: Trained BERTopic model
        docs: List of documents
        timestamps: List of datetime objects (same order as docs)
        name: Dataset name for logging
        nr_bins: Number of temporal bins (auto-calculated if None)
        top_n_topics: Number of topics to include in visualization
    
    Returns:
        Tuple of (topics_over_time DataFrame, Plotly figure) or (None, None) on failure
    """
    if timestamps is None:
        return None, None
    
    # Calculate optimal bins if not specified
    if nr_bins is None:
        nr_bins = calculate_optimal_bins(timestamps)
    
    print(f"\n Performing Dynamic Topic Modeling for {name}...")
    print(f"   Using {nr_bins} temporal bins, visualizing top {top_n_topics} topics")
    
    start_time = time.time()
    
    try:
        # BERTopic's topics_over_time handles binning and aggregation
        topics_over_time = topic_model.topics_over_time(
            docs=docs,
            timestamps=timestamps,
            nr_bins=nr_bins,
            datetime_format=None,  # Auto-detect format
            evolution_tuning=True,  # Fine-tune topic representations per time bin
            global_tuning=True      # Use global topic representations as reference
        )
        
        # Generate interactive visualization
        fig = topic_model.visualize_topics_over_time(
            topics_over_time=topics_over_time,
            top_n_topics=top_n_topics,
            normalize_frequency=False,
            title=f"Topic Evolution Over Time - {name}"
        )
        
        elapsed_time = time.time() - start_time
        print(f"    DTM completed in {elapsed_time:.2f} seconds")
        print(f"    Generated {len(topics_over_time)} temporal data points")
        
        return topics_over_time, fig
        
    except Exception as e:
        print(f"    Error during DTM for {name}: {e}")
        traceback.print_exc()
        return None, None


def save_dtm_outputs(topics_over_time, fig, name, dtm_dir):
    """
    Save DTM outputs: CSV data and HTML visualization.
    
    Args:
        topics_over_time: DataFrame from topics_over_time
        fig: Plotly figure from visualize_topics_over_time
        name: Dataset name for file naming
        dtm_dir: Output directory path
    """
    if topics_over_time is None or fig is None:
        print(f"     No DTM outputs to save for {name}")
        return
    
    try:
        # Save CSV data for further analysis
        csv_path = Path(dtm_dir) / f"{name}_topics_over_time.csv"
        topics_over_time.to_csv(csv_path, index=False)
        print(f"    Saved DTM data: {csv_path}")
        
        # Save interactive HTML visualization
        html_path = Path(dtm_dir) / f"{name}_topics_over_time.html"
        fig.write_html(str(html_path))
        print(f"    Saved DTM visualization: {html_path}")
        
    except Exception as e:
        print(f"    Error saving DTM outputs for {name}: {e}")
        traceback.print_exc()


# =============================================================================
# ORIGINAL FUNCTIONS (with minor updates)
# =============================================================================

def load_environment():
    JUPYTER = False
    try:
        import google.colab
        from google.colab import drive
        drive.mount("/content/drive")

        print("Installing dependencies...")
        print("Dependencies installed.")

        #Since its running locally (not in Colab),this code never executes
        base_path = "..."
        env_path = Path(base_path) / ".env"
        JUPYTER = True
    except ImportError:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        JUPYTER = False

    if env_path.exists():
        load_dotenv(env_path)
        print("Loading environment variables")
        data_dir, code_dir = os.getenv("DATA_DIR"), os.getenv("CODE_DIR")
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    return data_dir, code_dir, JUPYTER


def process_datasets(data_path, text_cols=('body', 'text')):
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
    """Create output directories including DTM folder."""
    directories = {
        "models": Path(code_dir) / "models",
        "IDM": Path(code_dir) / "visualizations" / "IDM",
        "hierarchies": Path(code_dir) / "visualizations" / "hierarchies",
        "barcharts": Path(code_dir) / "visualizations" / "barcharts",
        "dtm": Path(code_dir) / "visualizations" / "dtm",  # NEW: DTM directory
    }

    for path in directories.values():
        os.makedirs(path, exist_ok=True)

    return (
        directories["models"],
        directories["IDM"],
        directories["hierarchies"],
        directories["barcharts"],
        directories["dtm"]  # Return DTM directory
    )


def compute_embeddings(docs_dict):
    embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings_dict = {}
    for name, docs in docs_dict.items():
        print(f'Computing {name} embeddings:')
        embeddings_dict[name] = embedding_model.encode(docs, show_progress_bar=True)

    return embeddings_dict


def create_submodels(params=None):
    params = params or {
        "min_df": 0.05,
        "max_df": 0.9,
        "n_neighbors": 6,
        "min_cluster_size": 7,
        "min_topic_size": 7,
    }

    # Twitter-specific stopwords to filter artifacts that slip through preprocessing
    twitter_stopwords = ['https', 'http', 'co', 'rt', 'amp', 't', 'www', 'url', 'pic', 'twitter', 'com', 'ru']

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=params["min_df"],
        max_df=params["max_df"],
        stop_words=twitter_stopwords  # Fallback to catch any Twitter artifacts
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

    cohere_model = cohere_integration()
    if cohere_model:
        representation_model = [mmr_model, cohere_model]
        print(f"Using MMR + Cohere for representation")
    else:
        representation_model = mmr_model

    return vectorizer_model, umap_model, hdbscan_model, representation_model


def cohere_integration():
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        print("No COHERE_API_KEY found in .env file, skipping Cohere representation.")
        return None

    try:
        cohere_client = cohere.Client(cohere_api_key)
        custom_prompt = """
        I have a topic described by the following keywords:
        [KEYWORDS]

        The most representative documents for this topic are:
        [DOCUMENTS]

        Based on the information above, create a short topic label.
        Use 2-5 words maximum, no punctuation.

        Return only the label (2-5 words, no prefix)
        """
        return Cohere(
            cohere_client,
            model="command-r-08-2024",
            prompt=custom_prompt,
            nr_docs=4,
            diversity=0.1,
            delay_in_seconds=2
        )
    except Exception as e:
        print(f"Error initializing Cohere integration: {e}")
        return None


def bert_model(dataset_name, docs, embeddings, params=None):
    if not docs:
        print(f"No docs provided for {dataset_name}. Skipping topic modeling.")
        return None, None, None

    params = params or {}
    vectorizer_model, umap_model, hdbscan_model, representation_model = create_submodels(params)

    embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    print(f"Topic modeling for {dataset_name}...")

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
        print(f"{dataset_name} topic modeling completed in {elapsed_hours:.2f} hours using {embedding_model_name}")

def annotate_data(dfs, name, JUPYTER, topics_dict, probs_dict, topic_info_dict):
    dfs[name]["topic"] = topics_dict[name]
    dfs[name]["topic_proba"] = probs_dict[name]

    if JUPYTER:
        from IPython.display import display
        print("Processed data (sample):\n")
        display(dfs[name].sample(n=min(3, len(dfs[name]))))

        print(f"\nNumber of topics (including outlier): {len(topic_info_dict[name])}\n")
        display(topic_info_dict[name].sample(n=min(4, len(topic_info_dict[name]))))


def process_topic_merges(dfs, topic_info_dict, name, topic_col="topic", repr_docs_col="Representative_Docs"):
    # Drop any existing merge columns to avoid duplicates
    cols_to_drop = [c for c in dfs[name].columns if c.endswith('_x') or c.endswith('_y') or c in ['Name', 'Representation', 'Representative_Docs']]
    dfs[name] = dfs[name].drop(columns=cols_to_drop, errors='ignore')
    df = dfs[name].merge(
        topic_info_dict[name][["Topic", "Name", "Representation", repr_docs_col]],
        left_on=topic_col,
        right_on="Topic",
        how="left",
    )
    if "Topic" in df.columns:
      del df["Topic"]

    is_repr_col = f"is_representative{'_core' if 'core' in topic_col else ''}"
    df[is_repr_col] = df.apply(
        lambda row: 1
        if isinstance(row.get(repr_docs_col), list) and row.get("cleaned_text") in row.get(repr_docs_col)
        else 0,
        axis=1,
    )
    return df


def process_core_topics(dfs, name, core_topics, topics_dict, probs_dict):
    dfs[name]["core_topic"] = topics_dict[name]
    dfs[name]["core_topic_proba"] = probs_dict[name]

    core_topics = core_topics.rename(
        columns={
            "Name": "Name_core",
            "Representation": "Representation_core",
            "Representative_Docs": "Representative_Docs_core",
        }
    )

    dfs[name] = dfs[name].merge(
        core_topics[["Topic", "Name_core", "Representation_core", "Representative_Docs_core"]],
        left_on="core_topic",
        right_on="Topic",
        how="left",
    )

    dfs[name]["is_representative_core"] = dfs[name].apply(
        lambda row: 1
        if isinstance(row.get("Representative_Docs_core"), list)
        and row.get("cleaned_text") in row.get("Representative_Docs_core")
        else 0,
        axis=1,
    )

    return core_topics


def update_model(name, dfs, docs, topic_models, docs_dict, dirs, core_topics_dict, topics_dict, probs_dict, nr_topics=30):
    model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir = dirs  # Updated to include dtm_dir
    topic_model = topic_models[name]

    topic_model_clustered = topic_model.reduce_topics(docs_dict[name], nr_topics=nr_topics)
    print(f"New topics:\n{topic_model_clustered.topics_}")

    topic_model_clustered.update_topics(docs_dict[name], n_gram_range=(3, 5))

    core_topics = topic_model_clustered.get_topic_info()
    core_topics = process_core_topics(dfs, name, core_topics, topics_dict, probs_dict)
    core_topics_dict[name] = core_topics

    figure_hierarchy = topic_model_clustered.visualize_hierarchy()
    figure_topics = topic_model_clustered.visualize_topics()
    figure_barchart = topic_model_clustered.visualize_barchart(top_n_topics=len(core_topics), n_words=10)

    figure_hierarchy.write_html(os.path.join(hierarchy_dir, f"{name}HRC.html"))
    figure_topics.write_html(os.path.join(IDM_dir, f"{name}IDM.html"))
    figure_barchart.write_html(os.path.join(barchart_dir, f"{name}BRC.html"))

    return topic_model_clustered


def save_and_reload_model(name, model_dir, topic_models):
    save_path = Path(model_dir) / f"{name}.safetensors"
    topic_models[name].save(str(save_path), serialization="safetensors")
    print(f"Model saved: {save_path}")


def save_dataframe_inplace(path, df):
    try:
        df.to_csv(path, index=False)
        print(f"Saved updated dataframe back to {path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    data_dir, code_dir, JUPYTER = load_environment()
    if not data_dir or not code_dir:
        raise EnvironmentError("DATA_DIR and CODE_DIR must be set in the .env file.")

    dfs, docs_dict, datasets = process_datasets(data_dir)

    # Updated to include dtm_dir
    model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir = create_directories(code_dir)
    dirs = (model_dir, IDM_dir, hierarchy_dir, barchart_dir, dtm_dir)

    embeddings_dict = compute_embeddings(docs_dict)

    topic_models, topics_dict, probs_dict = {}, {}, {}
    topic_info_dict, core_topics_dict = {}, {}

    # Train models
    for name, docs in docs_dict.items():
        params = DATASET_PARAMS.get(name, DATASET_PARAMS["default"])
        topic_model, topics, probs = bert_model(
            dataset_name=name,
            docs=docs,
            embeddings=embeddings_dict[name],
            params=params
        )
        topic_models[name] = topic_model
        topics_dict[name] = topics
        probs_dict[name] = probs

    # Post-process & annotate
    for name in dfs.keys():
        topic_info_dict[name] = topic_models[name].get_topic_info()
        annotate_data(
            dfs, name, JUPYTER,
            topics_dict, probs_dict, topic_info_dict=topic_info_dict
        )
        process_topic_merges(dfs, topic_info_dict, name)

    # Update models and generate static visualizations
    for name in dfs.keys():
        params = DATASET_PARAMS.get(name, DATASET_PARAMS["default"])
        topic_models[name] = update_model(
            name=name,
            dfs=dfs,
            docs=docs_dict[name],
            topic_models=topic_models,
            docs_dict=docs_dict,
            dirs=dirs,
            core_topics_dict=core_topics_dict,
            topics_dict=topics_dict,
            probs_dict=probs_dict,
            nr_topics=params["nr_topics"]
        )
        save_dataframe_inplace(datasets[name], dfs[name])
        save_and_reload_model(name, model_dir, topic_models)

    # ==========================================================================
    # DYNAMIC TOPIC MODELING (NEW SECTION)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("DYNAMIC TOPIC MODELING")
    print("=" * 60)

    for name in dfs.keys():
        print(f"\n{'─' * 40}")
        print(f"Processing {name} for DTM...")
        
        # Step 1: Prepare timestamps
        timestamps = prepare_timestamps(dfs, name)
        
        if timestamps is None:
            print(f"   Skipping DTM for {name} (no valid timestamps)")
            continue
        
        # Step 2: Perform DTM using the already-trained model
        try:
            topics_over_time, fig = perform_dynamic_topic_modeling(
                topic_model=topic_models[name],
                docs=docs_dict[name],
                timestamps=timestamps,
                name=name,
                nr_bins=None,  # Auto-calculate
                top_n_topics=10
            )
            
            # Step 3: Save outputs
            if topics_over_time is not None and fig is not None:
                save_dtm_outputs(topics_over_time, fig, name, dtm_dir)
            
        except Exception as e:
            print(f"    DTM failed for {name}: {e}")
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("Pipeline finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Exception in pipeline:")
        traceback.print_exc()