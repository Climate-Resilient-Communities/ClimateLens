import os, re
import time
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from itertools import product

def load_environment():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        base = "/content/drive/MyDrive/ClimateLens/02 Code/02.01 MVP2/"
        env_path = Path(base) / "colab.env"
    except ImportError:
        env_path = Path(__file__).parent / ".env"

    if not env_path.exists():
        raise FileNotFoundError(env_path)

    load_dotenv(env_path)
    return os.getenv("DATA_DIR"), os.getenv("CODE_DIR")


DATA_DIR, CODE_DIR = load_environment()

def load_docs(data_dir, text_cols=("body", "text")):
    docs = {}

    for fp in Path(data_dir).glob("*.csv"):
        name = re.sub(r"(_clean|_filtered)?\.csv$", "", fp.name)
        df = pd.read_csv(fp)

        col = next((c for c in text_cols if c in df.columns), None)
        if not col:
            continue

        docs[name] = df[col].dropna().astype(str).tolist()
        print(f"Loaded {name}: {len(docs[name]):,} docs")

    return docs
docs_dict = load_docs(DATA_DIR)

print("Installing dependencies...")
!pip install -qq bertopic sentence-transformers umap-learn hdbscan # change to -vv for debugging!!
print("Dependencies installed.")

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

EMBEDDING_MODELS = {
    "minilm-L6": "sentence-transformers/all-MiniLM-L6-v2", #fastest
    "minilm-L12": "sentence-transformers/all-MiniLM-L12-v2", #fast
    #"distilroberta": "sentence-transformers/all-distilroberta-v1", #slow
    #"all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2", #slowest
}

def compute_all_embeddings(docs_dict, embedding_models):
    embeddings = {}

    for dataset, docs in docs_dict.items():
        for emb_name, emb_path in embedding_models.items():
            print(f"Computing embeddings | Dataset={dataset} | Model={emb_name}")
            model = SentenceTransformer(emb_path)
            emb = model.encode(
                docs,
                batch_size=32,
                show_progress_bar=True,
            )
            embeddings[(dataset, emb_name)] = emb

    return embeddings

embeddings_dict = compute_all_embeddings(docs_dict, EMBEDDING_MODELS)

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
    "ngram_range": (2, 3),
}

def run_experiment(
    docs,
    embeddings,
    embedding_name,
    dataset_name,
    umap_params,
    hdbscan_params,
    bertopic_params,
):
    vectorizer = CountVectorizer(
        ngram_range=FIXED_PARAMS["ngram_range"],
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

    # Create a copy of bertopic_params and remove 'mmr_value' before passing to BERTopic
    # as mmr_value is used in the representation_model and not a direct BERTopic parameter.
    bertopic_params_for_bertopic = bertopic_params.copy()
    if "mmr_value" in bertopic_params_for_bertopic:
        del bertopic_params_for_bertopic["mmr_value"]

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        top_n_words=FIXED_PARAMS["top_n_words"],
        nr_topics=FIXED_PARAMS["nr_topics"],
        **bertopic_params_for_bertopic,
    )

    start = time.time()
    topics, _ = topic_model.fit_transform(docs, embeddings)
    runtime = time.time() - start

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    outliers = topics.count(-1) / len(topics)

    return {
        "dataset": dataset_name,
        "embedding_model": embedding_name,
        "n_topics": n_topics,
        "outlier_frac": round(outliers, 4),
        "runtime_sec": round(runtime, 2),
        "topic_model": topic_model,
        "mmr_diversity": representation_model.diversity,
        **umap_params,
        **hdbscan_params,
        **bertopic_params,
    }

def grid_search(docs_dict, embeddings_dict, output_dir):
    results = []

    for dataset, docs in docs_dict.items():
        print(f"\nDataset: {dataset} | Sampled: {len(docs):,}")

        for emb_name, emb_path in EMBEDDING_MODELS.items():
          embeddings = embeddings_dict[(dataset, emb_name)]
          print(f"\nEmbedding model: {emb_name}")
          #embeddings = compute_embeddings(emb_path, docs)

          for umap_vals in product(*UMAP_GRID.values()):
              umap_params = dict(zip(UMAP_GRID.keys(), umap_vals))

              for hdb_vals in product(*HDBSCAN_GRID.values()):
                  hdb_params = dict(zip(HDBSCAN_GRID.keys(), hdb_vals))

                  for bert_vals in product(*BERTOPIC_GRID.values()):
                      bert_params = dict(zip(BERTOPIC_GRID.keys(), bert_vals))

                      # Constraint: min_topic_size >= min_cluster_size
                      if bert_params["min_topic_size"] < hdb_params["min_cluster_size"]:
                          continue

                      try:
                          print(
                              f"COMPUTING: UMAP={umap_params} | "
                              f"HDB={hdb_params} | "
                              f"BERT={bert_params}"
                          )

                          res = run_experiment(
                              docs,
                              embeddings,
                              emb_name,
                              dataset,
                              umap_params,
                              hdb_params,
                              bert_params,
                          )
                          results.append(res)

                      except Exception:
                          traceback.print_exc()

    df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"bertopic_reddit_grid_results_{ts}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved results to: {out_path}")
    return df, embeddings_dict, out_path, res

if __name__ == "__main__":
    docs_dict = load_docs(DATA_DIR)

    testing_dir = Path(CODE_DIR) / "testing" / "topic_modeling"
    results_df, embeddings_dict, out_path, res = grid_search(docs_dict, embeddings_dict, testing_dir)

    print("\nTop candidates:")
    print(
        results_df.sort_values(
            ["outlier_frac", "n_topics"],
            ascending=[True, False],
        ).head(10)
    )

results = pd.read_csv(out_path)

results[
    (results["outlier_frac"] > 0.12) &
    (results["outlier_frac"] < 0.25) &
    (results["n_topics"]>2)
].drop(
    columns=["topic_model", "mmr_diversity", "mmr_value"]
).sort_values(
    by="outlier_frac", ascending=True
)

def plot_barchart_from_results(
    results_df,
    model_selector,
    top_n_topics=10,
    n_words=10,
    width=2000,
    height=1200,
    title_prefix="BERTopic barchart"
):
    """
    model_selector:
        - int → row index in results_df
        - dict → column filters, e.g. {"embedding_model": "minilm-L6"}
        - callable → function that takes df and returns a filtered df
    """

    # Select row
    if isinstance(model_selector, int):
        row = results_df.iloc[model_selector]

    elif isinstance(model_selector, dict):
        df_filt = results_df.copy()
        for col, val in model_selector.items():
            df_filt = df_filt[df_filt[col] == val]

        if df_filt.empty:
            raise ValueError("No model matches the provided filters")

        row = df_filt.iloc[0]

    elif callable(model_selector):
        df_filt = model_selector(results_df)
        if df_filt.empty:
            raise ValueError("Callable selector returned no rows")
        row = df_filt.iloc[0]

    else:
        raise TypeError("model_selector must be int, dict, or callable")

    topic_model = row["topic_model"]

    fig = topic_model.visualize_barchart(
        top_n_topics=top_n_topics,
        n_words=n_words
    )

    fig.update_layout(
        width=width,
        height=height,
        title=(
            f"{title_prefix}<br>"
            f"Dataset={row['dataset']} | "
            f"Embedding={row['embedding_model']} | "
            f"Topics={row['n_topics']} | "
            f"Outliers={row['outlier_frac']}"
        )
    )

    return fig

fig = plot_barchart_from_results(
    results_df,
    model_selector=1
)
display(fig)