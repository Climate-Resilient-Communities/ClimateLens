import os
import pandas as pd


def load_datasets(data_path, prefix, datasets):
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        
        if os.path.isfile(file_path):
            file_name = file.replace(prefix, "").replace(".csv", "")
            datasets[file_name] = file_path

datasets = {}
load_datasets("...", "filtered_", datasets)
load_datasets("...", "clean_", datasets)

print("Collected Datasets:")
for key, value in datasets.items():
    print(f'{key}: {value}\n')

def loading_datasets(datasets):
    dfs = {}
    docs_dict = {}

    for name, path in datasets.items():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        if "cleaned_text" in df.columns:
            text_col = "cleaned_text"
        else:
            print(f"Skipping {name}. No 'body' or 'text' column.")
            continue

        df = df.dropna(subset=[text_col])

        print(f'Loaded {name}')

        docs = list(df[text_col].values)

        dfs[name] = df
        docs_dict[name] = docs

    return dfs, docs_dict

dfs, docs_dict = loading_datasets(datasets) # datafames aren't standalone variables
print(f"{len(list(dfs.keys()))} Dataframes collected")

import numpy as np
import matplotlib.pyplot as plt
import copy # copy topic models
import time

paths = {
    "home": "...", # root path, where this file is
    "models": "...", #path for models
    "insights": "...", # static
    "interactives": "...",
}

os.makedirs(paths.get("models"), exist_ok=True)
os.makedirs(paths.get("insights"), exist_ok=True)
os.makedirs(paths.get("interactives"), exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

### modeling
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import Cohere
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

"""# Topic Modeling"""

topic_models = {} # 'name': 'topic model'
topics_dict = {} # 'name' : 'topics'
probs_dict = {} # 'name' : 'probs'
topic_info_dict = {} # 'name' : 'topic information' .get_topic_info()
core_topics_dict = {} # 'name' : 'processed core topics'

embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
embedding_model = SentenceTransformer(embedding_model_name)
embeddings_dict = {}
for name, docs in docs_dict.items(): #pre-calculating embeddings
    print(f'Computing {name} embeddings:\n')
    embeddings_dict[name] = embedding_model.encode(docs, show_progress_bar=True)
    print('\n')

def bert_model(dataset_name, min_df, max_df, n_neighbors, min_cluster_size, min_topic_size):
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), # must be a tuple with range
        min_df=min_df, # increase for larger data
        max_df=max_df
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5, # be careful with this
        metric='cosine',
        low_memory=False,
        random_state=42 # avoid stochastic behaviours
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        prediction_data=True
    )

    representation_model = MaximalMarginalRelevance(diversity=0.1)

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
    display(topic_info_dict[name].sample(n=min(4, len(topic_info_dict[name]))))

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
    figure_heatmap=topic_model.visualize_heatmap(n_clusters=int(len(topic_info_dict[name]))-2)

    #display(figure_hierarchy)
    display(figure_topics)
    display(figure_barchart)
    #display(figure_heatmap)

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
    figure_heatmap=topic_model_clustered.visualize_heatmap(n_clusters=int(len(core_topics)) - 2)

    visuals_path=paths.get("interactives")
    if save==True:
      figure_hierarchy.write_html(os.path.join(visuals_path, f"{name}hierarchy.html"))
      figure_topics.write_html(os.path.join(visuals_path, f"{name}topic_distance.html"))
      figure_barchart.write_html(os.path.join(visuals_path, f"{name}barchart.html"))
      figure_heatmap.write_html(os.path.join(visuals_path, f"{name}heatmap.html"))

    return topic_model_clustered

def save_and_reload_model(name):
    joined_path = os.path.join(paths.get("models"), f"{name}.safetensors")
    topic_models[name].save(joined_path, serialization="safetensors")
    #return BERTopic.load(save_path) # immediately reload

import traceback

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