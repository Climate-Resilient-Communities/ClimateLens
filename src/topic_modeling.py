import os
import re
import time
import warnings
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
#import cohere

#warnings.filterwarnings("ignore")

def load_environment():
  try:
    import google.colab
    from google.colab import drive
    drive.mount("/content/drive")

    print("Installing dependencies...")
    !pip install -q bertopic sentence-transformers umap-learn hdbscan #cohere
    print("Environment setup complete.")

    base_path = "/content/drive/MyDrive/ClimateLens/02 Notebooks/02.01 MVP2/"
    env_path = Path(base_path) / ".env"
  except ImportError:
    env_path = Path(__file__).resolve().parent / ".env"

  if env_path.exists():
    load_dotenv(env_path)
    print("Loaded environment variables")
    data_dir, code_dir = os.getenv("DATA_DIR"), os.getenv("CODE_DIR")
  else:
    raise FileNotFoundError(f".env file not found at {env_path}")

  return data_dir, code_dir

def process_datasets(data_path):
    dfs, docs_dict, datasets, failed = {}, {}, {}, []

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)

        if os.path.isfile(file_path) and file.endswith(".csv"):
            file_name = re.sub(r"(_clean|filtered_)?\.csv$", "", file)
            datasets[file_name] = file_path

    for name, path in datasets.items():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            failed.append(name)
            continue

        text_col = next((col for col in ['body', 'text'] if col in df.columns), None)

        if text_col is None:
            print(f"Skipping {name}. No 'body' or 'text' column.")
            failed.append(name)
            continue

        dfs[name] = df
        docs_dict[name] = df[df[text_col].notna()][text_col].tolist()

        print(f'Loaded {name}')

    return dfs, docs_dict, datasets, failed

print(f"{len(dfs)}/{len(datasets)} Dataframes loaded successfully")
if failed:
    print(f"Failed to load (check errors): {', '.join(failed)}")
 
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
  hierarchy_dir = directories["hierarchies"]
  barchart_dir = directories["barcharts"]
  model_dir = directories["models"]

def compute_embeddings(docs_dict):
  topic_models, topics_dict, probs_dict = {}, {}, {}, # 'name': 'model/topics/probs'
  topic_info_dict, core_topics_dict = {}, {} # 'name' : 'topic info / core topics'

  embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
  embedding_model = SentenceTransformer(embedding_model_name)
  embeddings_dict = {}
  for name, docs in docs_dict.items():
      print(f'Computing {name} embeddings:\n')
      embeddings_dict[name] = embedding_model.encode(docs, show_progress_bar=True)
      print('\n')

def bert_model(dataset_name, min_df, max_df, n_neighbors, min_cluster_size, min_topic_size):
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        metric='cosine',
        low_memory=False,
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        prediction_data=True
    )

    # Create MMR model
    mmr_model = MaximalMarginalRelevance(diversity=0.1)

    # Cohere integration - commented out due to API deprecation (Sept 15, 2025)
    # TODO: Re-enable once BERTopic supports new Cohere Chat API
    # cohere_api_key = os.getenv("COHERE_API_KEY")
    # if cohere_api_key:
    #     cohere_client = cohere.Client(cohere_api_key)
    #     custom_prompt = """
    # I have a topic described by the following keywords:
    # [KEYWORDS]
    #
    # The most representative documents for this topic are:
    # [DOCUMENTS]
    #
    # Based on the information above, create a short topic label.
    # Use 2-5 words maximum, no punctuation.
    #
    # Return only the label (2-5 words, no prefix)
    # """
    #     cohere_model = Cohere(
    #         cohere_client,
    #         model="command-r-08-2024",
    #         prompt=custom_prompt,
    #         nr_docs=4,
    #         diversity=0.1,
    #         delay_in_seconds=2
    #     )
    #     representation_model = [mmr_model, cohere_model]
    #     print(f"✅ Using MMR + Cohere for {dataset_name}")
    # else:

    # Using MMR only until Cohere integration is fixed
    representation_model = mmr_model
    print(f"Using MMR for {dataset_name}")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        nr_topics='auto',
    )

    print(f"Fitting {dataset_name} model...\n")
    start_time = time.time()

    try:
        topics, probs = topic_model.fit_transform(
            docs_dict[dataset_name],
            embeddings_dict[dataset_name]
        )
        return topic_model, topics, probs

    except Exception as e:
        print(f'Error occured during {dataset_name} topic modeling: {e}')
        return None, None, None

    finally:
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 3600
        print(f"{dataset_name} topic modeling completed in {elapsed_time:.3f} hours using {embedding_model_name}")

from IPython.display import display

def annotate_data(name):
    dfs[name]['topic'] = topics_dict[name]
    dfs[name]['topic_proba'] = probs_dict[name]

    print("processed data:\n")
    display(dfs[name].sample(n=min(3, len(dfs[name]))))

    print(f'\nNumber of topics (including outlier): {len(topic_info_dict[name])}\n')
    display(topic_info_dict[name].sample(n=min(4, len(topic_info_dict[name])))) #uncomment if in jupyter notebook

def process_topic_merges(name, topic_col='topic', repr_docs_col='Representative_Docs'):
    df = dfs[name].merge(
        topic_info_dict[name][['Topic', 'Name', 'Representation', repr_docs_col]],
        left_on=topic_col,
        right_on='Topic',
        how='left'
    )
    del df['Topic']
    is_repr_col = f'is_representative{"_core" if "core" in topic_col else ""}'
    df[is_repr_col] = df.apply(
        lambda row: 1 if isinstance(row[repr_docs_col], list) and row['cleaned_text'] in row[repr_docs_col] else 0,
        axis=1
    )
    return df

def process_core_topics(name, core_topics):
    dfs[name]['core_topic'] = topics_dict[name]
    dfs[name]['core_topic_proba'] = probs_dict[name]

    core_topics = core_topics.rename(columns={
        'Name': 'Name_core',
        'Representation': 'Representation_core',
        'Representative_Docs': 'Representative_Docs_core'
    })

    dfs[name] = dfs[name].merge(
        core_topics[['Topic', 'Name_core','Representation_core','Representative_Docs_core']],
        left_on='core_topic',
        right_on='Topic',
        how='left'
    )
    del dfs[name]['Topic']
    dfs[name]['is_representative_core'] = dfs[name].apply(
        lambda row: 1 if isinstance(row['Representative_Docs_core'], list) and row['cleaned_text'] in row['Representative_Docs_core'] else 0,
        axis=1
    )

    return core_topics

def visualize_model(name):
    topic_model = topic_models[name]
    print(f"\nVisuals for {name}:\n")

    figure_hierarchy=topic_model.visualize_hierarchy()
    figure_topics=topic_model.visualize_topics()
    figure_barchart=topic_model.visualize_barchart(top_n_topics=10, n_words=10)

    display(figure_topics)
    display(figure_barchart)

def update_model(name, save=True):
    topic_model = topic_models[name]

    topic_model_clustered = topic_model.reduce_topics(docs_dict[name], nr_topics=30)
    print(f'New topics:\n{topic_model_clustered.topics_}')

    topic_model_clustered.update_topics(docs_dict[name], n_gram_range=(3,5))

    core_topics = topic_model_clustered.get_topic_info() # remove this and add core_topics_dict={}
    core_topics = process_core_topics(name, core_topics)
    core_topics_dict[name] = core_topics

    figure_hierarchy=topic_model_clustered.visualize_hierarchy()
    figure_topics=topic_model_clustered.visualize_topics()
    figure_barchart=topic_model_clustered.visualize_barchart(top_n_topics=len(core_topics), n_words=10)

    if save==True:
      figure_hierarchy.write_html(os.path.join(heirarchy_dir, f"{name}HRC.html"))
      figure_topics.write_html(os.path.join(IDM_dir, f"{name}IDM.html"))
      figure_barchart.write_html(os.path.join(barchart_dir, f"{name}BRC.html"))

    return topic_model_clustered

def save_and_reload_model(name):
    joined_path = os.path.join(model_dir, f"{name}.safetensors")
    topic_models[name].save(joined_path, serialization="safetensors")
    #return BERTopic.load(save_path) # immediately reload

for name in list(docs_dict.keys()):
    print("\n" + "="*50)
    print(f"Starting Topic Modeling for: {name}")
    print("="*50)

    try:
        if name == 'twitter':
            print(f"{name} Running BERT model with twitter parameters...")
            topic_model, topics, probs = bert_model(name, min_df=0.05, max_df=0.90,
                                                    n_neighbors=5, min_cluster_size=5, min_topic_size=5)
        else:
            print(f"{name} Running BERT model with reddit parameters...")
            topic_model, topics, probs = bert_model(name, min_df=0.05, max_df=0.90,
                                                    n_neighbors=6, min_cluster_size=7, min_topic_size=7)

        topic_models[name] = topic_model
        topics_dict[name] = topics
        probs_dict[name] = probs

        topic_info_dict[name] = topic_model.get_topic_info()

        print(f"{name} data annotation and topic merging starting...")
        annotate_data(name)
        process_topic_merges(name)

        n_topics = len(topic_model.get_topic_info()) - 1  #exclude outlier
        if n_topics > 30:
            print(f"Updating {name} model...")
            update_model(name)

        save_and_reload_model(name)

        print(f"{name} topic modeling complete!")

    except Exception as e:
        print(f"[{name}] Error encountered: {e}")
        traceback.print_exc()