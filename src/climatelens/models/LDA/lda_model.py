import gensim
import numpy as np
import pandas as pd
from gensim.matutils import jensen_shannon
from gensim.models.coherencemodel import CoherenceModel
from preprocessing import preprocess, stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


def get_jensen_shannon(components: np.ndarray, ntopics: int) -> tuple[float, float]:
    topic_dists = components
    js_dists = []
    for i in range(ntopics):
        for j in range(ntopics):
            if i > j:
                js_dists.append(jensen_shannon(topic_dists[i, :], topic_dists[j, :]))
    return float(np.min(js_dists)), float(np.mean(js_dists))


class LDAwithCustomScore(LatentDirichletAllocation):
    def score(self, X: np.ndarray, y: None = None) -> float:
        components = self.components_
        ntopics = self.n_components
        return get_jensen_shannon(components, ntopics)[0]


def best_model(
    documents: pd.DataFrame, ntopics_list: list[int]
) -> tuple[LDAwithCustomScore, list[list[str]], np.ndarray, np.ndarray, CountVectorizer]:
    processed_info: list[list[str]] = []
    for allinfo in documents["text"].values:
        preprocessed, _ = preprocess(allinfo, stopwords)
        processed_info.append(preprocessed)

    countvec = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords, max_df=0.25, min_df=10)
    clean_text = [" ".join(text) for text in processed_info]
    X = countvec.fit_transform(clean_text).toarray()
    terms = countvec.get_feature_names_out()

    search_params = {"n_components": ntopics_list}
    lda = LDAwithCustomScore(random_state=0)
    model = GridSearchCV(lda, param_grid=search_params, cv=5)
    model.fit(X)

    return model.best_estimator_, processed_info, X, terms, countvec


def Evaluate(
    model: LDAwithCustomScore,
    processed_info: list[list[str]],
    X: np.ndarray,
    terms: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    topic_word_distributions = model.components_
    top_n_words = 10
    topics = [
        [terms[i] for i in topic.argsort()[: -top_n_words - 1 : -1]]
        for topic in topic_word_distributions
    ]

    dictionary = gensim.corpora.Dictionary(processed_info)
    coherence_model = CoherenceModel(
        topics=topics, texts=processed_info, dictionary=dictionary, coherence="c_v"
    )
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score}")

    perplexity = model.perplexity(X)
    print(f"Perplexity: {perplexity}")

    topic_distribution = model.transform(X)

    return coherence_score, perplexity, topic_distribution
