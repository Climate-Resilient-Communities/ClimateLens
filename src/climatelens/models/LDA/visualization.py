import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis
import pyLDAvis.lda_model
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

sys.setrecursionlimit(10000)  # required for scipy dendrogram on large datasets


def visualize_topics(
    model: LatentDirichletAllocation,
    topic_distribution: np.ndarray,
    X: np.ndarray,
    countvec: CountVectorizer,
) -> Any:
    topic_term_dists = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    doc_lengths = X.sum(axis=1).tolist()
    term_frequency = np.asarray(X.sum(axis=0)).flatten().tolist()
    vocab = countvec.get_feature_names_out()

    return pyLDAvis.prepare(
        topic_term_dists,
        topic_distribution,
        doc_lengths,
        vocab,
        term_frequency,
    )


def hierarchy(
    topic_distribution: np.ndarray,
    dataset_name: str,
    n_clusters: int | None = None,
    distance_threshold: float = 0,
    linkage_method: str = "ward",
    figsize: tuple[int, int] = (50, 35),
) -> AgglomerativeClustering:
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage_method,
    )
    agg_clustering.fit(topic_distribution)

    children = agg_clustering.children_
    n_samples = len(topic_distribution)
    linkage_matrix = np.zeros((n_samples - 1, 4))

    for i, (left, right) in enumerate(children):
        linkage_matrix[i, 0] = left
        linkage_matrix[i, 1] = right
        linkage_matrix[i, 2] = agg_clustering.distances_[i]
        linkage_matrix[i, 3] = 2

    plt.figure(figsize=figsize)
    dendrogram(linkage_matrix, labels=[f"Doc {i + 1}" for i in range(n_samples)])
    plt.title(f"{dataset_name} Hierarchical Clustering")
    plt.xlabel("Documents")
    plt.ylabel("Distance")
    plt.show()

    return agg_clustering


def generate_word_cloud(
    model: LatentDirichletAllocation, terms: np.ndarray, top_n: int = 10
) -> WordCloud:
    top_words: list[str] = []
    for topic in model.components_:
        top_words += [terms[i] for i in topic.argsort()[: -top_n - 1 : -1]]
    return WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(top_words)
    )


def plot_topic_word_distributions(
    model: LatentDirichletAllocation,
    terms: np.ndarray,
    dataset_name: str,
    top_n: int = 1,
) -> None:
    topic_word_distributions = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    topics = np.argsort(-topic_word_distributions, axis=1)[:, :top_n]

    data = np.zeros((len(topics), top_n))
    for i, topic in enumerate(topics):
        data[i] = topic_word_distributions[i, topic]

    fig, ax = plt.subplots(figsize=(15, 30))
    ax.barh(range(len(topics)), data.sum(axis=1), color="gray", alpha=0.5)
    colors = plt.cm.get_cmap("tab20", len(topics))

    for i, topic in enumerate(topics):
        ax.barh(i, data[i], color=colors(i))
        words = ", ".join([terms[j] for j in topic])
        value = data[i].sum()
        ax.text(value + 0.01, i, words, va="center", fontsize=10, color="black")

    ax.set_title(f"{dataset_name} Topic Word Distribution")
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([f"Topic {i + 1}" for i in range(len(topics))])
    plt.xlabel("Word Distribution")
    plt.tight_layout()
    plt.show()


def plot_coherence_perplexity(
    coherence: list[float],
    perplexity: list[float],
    names: list[str],
) -> None:
    colors = ["r", "b"]
    for i in range(len(names)):
        plt.scatter(coherence[i], perplexity[i], color=colors[i], s=100)
        plt.text(coherence[i], perplexity[i], f"{names[i]}", fontsize=10, ha="right", va="bottom")

    plt.xlim(0, 1)
    plt.xlabel("Coherence")
    plt.ylabel("Perplexity")
    plt.title("Coherence & Perplexity of Reddit and Twitter LDA Models")
    plt.scatter(
        [], [], color="r",
        label=f"Reddit: Coherence={coherence[0]:.2f}, Perplexity={int(perplexity[0])}",
    )
    plt.scatter(
        [], [], color="b",
        label=f"Twitter: Coherence={coherence[1]:.2f}, Perplexity={int(perplexity[1])}",
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    plt.grid(True)
    plt.show()


def plot_document_topic_heatmap(topic_distribution: np.ndarray, dataset_name: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(topic_distribution, cmap="viridis", cbar_kws={"label": "Topic Probability"})
    plt.title(f"{dataset_name} Document-Topic Distribution")
    plt.xlabel("Topics")
    plt.ylabel("Documents")
    plt.show()