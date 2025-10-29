import os
import re
import time
import warnings
import traceback
from pathlib import Path

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

def load_environment():
  JUPYTER = False
  try:
    import google.colab
    from google.colab import drive
    drive.mount("/content/drive")

    print("Installing dependencies...")
    !pip install -q bertopic sentence-transformers umap-learn hdbscan #cohere
    print("Dependencies installed.")

    base_path = "..."
    env_path = Path(base_path) / ".env"
    JUPYTER=True
  except ImportError:
    env_path = Path(__file__).resolve().parent / ".env"
    JUPYTER=False

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
      datasets[name] = file_path  # only used for links

      try:
          df = pd.read_csv(file_path)
          text_col = next((c for c in text_cols if c in df.columns), None)

          if not text_col:
              print(f"Skipping {name}. No {text_cols} column found.")
              failed.append(name)
              continue

          dfs[name] = df
          docs_dict[name] = df[df[text_col].notna()][text_col].tolist()
          #docs_dict[name] = df[text_col].dropna().astype(str).tolist()
          print(f"Loaded {name} ({len(dfs[name])} rows) from: {file_path}")

      except Exception as e:
          print(f"Error loading {name}: {e}")
          failed.append(name)

  print(f"{len(dfs)}/{len(datasets)} datasets loaded successfully")
  if failed:
      print(f"Failed to load: {', '.join(failed)}")

  return dfs, docs_dict, datasets

def create_directories(code_dir):
  directories = {
      "models": Path(code_dir) / "models",
      "IDM": Path(code_dir) / "visualizations" / "IDM",
      "heirarchies": Path(code_dir) / "visualizations" / "heirarchies",
      "barcharts": Path(code_dir) / "visualizations" / "barcharts",
  }

  for path in directories.values():
    os.makedirs(path, exist_ok=True)

  IDM_dir = directories["IDM"]
  hierarchy_dir = directories["heirarchies"]
  barchart_dir = directories["barcharts"]
  model_dir = directories["models"]

  return model_dir, IDM_dir, hierarchy_dir, barchart_dir

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

  if cohere_integration():
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
  df = dfs[name].merge(
      topic_info_dict[name][["Topic", "Name", "Representation", repr_docs_col]],
      left_on=topic_col,
      right_on="Topic",
      how="left",
  )
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
  model_dir, IDM_dir, hierarchy_dir, barchart_dir = dirs
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
  #return BERTopic.load(str(save_path))

def save_dataframe_inplace(path, df):
    try:
        df.to_csv(path, index=False)
        print(f"Saved updated dataframe back to {path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

def main():
  data_dir, code_dir, JUPYTER = load_environment()
  if not data_dir or not code_dir:
    raise EnvironmentError("DATA_DIR and CODE_DIR must be set in the .env file.")

  dfs, docs_dict, datasets = process_datasets(data_dir)

  model_dir, IDM_dir, hierarchy_dir, barchart_dir = create_directories(code_dir)
  dirs = (model_dir, IDM_dir, hierarchy_dir, barchart_dir)

  embeddings_dict = compute_embeddings(docs_dict)

  topic_models, topics_dict, probs_dict = {}, {}, {} # 'name': 'model/topics/probs'
  topic_info_dict, core_topics_dict = {}, {} # 'name': 'topic info / core topics'

  for name, docs in docs_dict.items():
    params = (
        {"min_df": 0.05, "max_df": 0.90, "n_neighbors": 5,
         "min_cluster_size": 5, "min_topic_size": 5}
        if name == "twitter"
        else {"min_df": 0.05, "max_df": 0.90, "n_neighbors": 6,
              "min_cluster_size": 7, "min_topic_size": 7}
    )
    topic_model, topics, probs = bert_model(
        dataset_name=name,
        docs=docs,
        embeddings=embeddings_dict[name],
        params=params
        )
    topic_models[name] = topic_model
    topics_dict[name] = topics
    probs_dict[name] = probs

  #post-process & annotate
  for name in dfs.keys():
      topic_info_dict[name] = topic_models[name].get_topic_info()
      annotate_data(
          dfs, name, JUPYTER,
          topics_dict, probs_dict, topic_info_dict=topic_info_dict
      )

      process_topic_merges(dfs, topic_info_dict, name)

  for name in dfs.keys():
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
          nr_topics=30
      )
      save_dataframe_inplace(datasets[name], dfs[name])
      save_and_reload_model(name, model_dir, topic_models)

  print("Pipeline finished successfully.")

if __name__ == "__main__":
  try:
    main()
  except Exception:
    print("Exception in pipeline:")
    traceback.print_exc()