"""
Baseline LDA topic modeling for comparison with BERTopic.

NOT production, only used for:
- sanity checks
- interpretability comparison
- baseline metrics
"""

import warnings

warnings.filterwarnings("ignore")


"""
# Preprocessing
"""

import os
import sys
import time  # start, stop times

import joblib  # saving/loading models

### Topic modeling, preprocessing, etc.
import nltk

# !pip install numpy==1.23.5 gensim==4.3.3
import pandas as pd

nltk.download("wordnet")
nltk.download("stopwords")

from lda_model import Evaluate, best_model
from visualization import hierarchy, visualize_topics

names = None  # EDIT

# add scripts directory to path
sys.path.insert(1, "../scripts/")

from google.colab import drive

drive.mount("/content/drive")

folder_path = "/content/drive/My Drive/KHP/Results/"
reddit_df = pd.read_csv(
    os.path.join(folder_path, "Climate_Reddit_Youth_Sentiment_Emotion_Labelled.csv")
)
twitter_df = pd.read_csv(
    os.path.join(folder_path, "Climate_Tweet_Youth_Sentiment_Emotion_Labelled.csv")
)

print(f"Reddit dataframe with {len(reddit_df)} entries: \n{reddit_df.head()}")
print(f"\n Twitter dataframe with {len(twitter_df)} entries: \n{twitter_df.head()}")

"""**Functions**"""


"""Reddit"""

start_time = time.time()
best_reddit_model, processed_info, X_reddit, reddit_terms, reddit_countvec = best_model(
    reddit_df, ntopics_list=[5, 15, 50, 75, 100, 150]
)
end_time = time.time()

reddit_training_time = end_time - start_time

print(f"The best Reddit LDA model has {best_reddit_model.n_components} distinct topics")
print(f"Training time: {reddit_training_time:.4f} seconds (with GridSearch)\n")

redditLDA_model_path = "/content/drive/My Drive/Notebooks/LDA Topic Modeling/LDA_reddit_model.pkl"
joblib.dump(best_reddit_model, redditLDA_model_path)  # saving the model

reddit_coherence_score, reddit_perplexity, reddit_topic_distribution = Evaluate(
    best_reddit_model, processed_info, X_reddit, reddit_terms
)

import sys

import joblib
import pyLDAvis  # notebooks
import pyLDAvis.lda_model  # communicate with sklearn

# add scripts directory to path
sys.path.insert(1, "../scripts/")

best_reddit_model = joblib.load(
    "/content/drive/My Drive/Notebooks/LDA Topic Modeling/Models/LDA_reddit_model.pkl"
)

reddit_data = visualize_topics(
    best_reddit_model, reddit_topic_distribution, X_reddit, reddit_countvec
)

pyLDAvis.enable_notebook()
pyLDAvis.display(reddit_data)

reddit_vis_path = (
    "/content/drive/My Drive/Notebooks/LDA Topic Modeling/Models/Reddit_pyLDAvis.html"
)
pyLDAvis.save_html(reddit_data, reddit_vis_path)

# Call the function to perform clustering and generate the dendrogram
agg_clustering_model_reddit = hierarchy(
    best_reddit_model,
    reddit_topic_distribution,
    dataset_name=names[0],
    n_clusters=None,
    distance_threshold=0.0,
)
# print(agg_clustering_model_reddit.labels_)

"""Twitter"""

start_time = time.time()
best_twitter_model, processed_info, X_twitter, twitter_terms, twitter_countvec = best_model(
    twitter_df, ntopics_list=[5, 15, 50, 75, 100, 150]
)
end_time = time.time()

twitter_training_time = end_time - start_time

print(f"The best Twitter LDA model has {best_twitter_model.n_components} distinct topics")
print(f"Training time: {twitter_training_time:.4f} seconds (with GridSearch)\n")

twitterLDA_model_path = (
    "/content/drive/My Drive/Notebooks/LDA Topic Modeling/LDA_twitter_model.pkl"
)
joblib.dump(best_twitter_model, twitterLDA_model_path)  # saving the model

# Call Evaluate with the necessary arguments
twitter_coherence_score, twitter_perplexity, twitter_topic_distribution = Evaluate(
    best_twitter_model, processed_info, X_twitter, twitter_terms
)

best_twitter_model = joblib.load(
    "/content/drive/My Drive/Notebooks/LDA Topic Modeling/LDA_twitter_model.pkl"
)

twitter_data = visualize_topics(
    best_twitter_model, twitter_topic_distribution, X_twitter, twitter_countvec
)

pyLDAvis.enable_notebook()
pyLDAvis.display(twitter_data)

twitter_vis_path = (
    "/content/drive/My Drive/Notebooks/LDA Topic Modeling/Models/Twitter_pyLDAvis.html"
)
pyLDAvis.save_html(twitter_data, twitter_vis_path)

agg_clustering_model_twitter = hierarchy(
    best_twitter_model,
    twitter_topic_distribution,
    dataset_name=names[1],
    n_clusters=None,
    distance_threshold=0.0,
)
# print(agg_clustering_model_twitter.labels_)
