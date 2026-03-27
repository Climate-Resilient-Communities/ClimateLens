

def get_jensen_shannon(components, ntopics):
    topic_dists = components
    js_dists = []
    for i in range(ntopics):
        for j in range(ntopics):
            if i>j:
                js_dists.append(jensen_shannon(topic_dists[i,:], topic_dists[j,:]))

    return np.min(js_dists), np.mean(js_dists)

def get_jaccard(components, ntopics): # not being used
    topn = int(np.ceil(len(dictionary)*(10/100)))
    topic_word_probs = components
    top_terms = np.argsort(-1*topic_word_probs,axis=1)
    top_terms = 1*top_terms[:,0:topn]
    jdists = []
    for i in range(ntopics):
        for j in range(ntopics):
            if i > j:
                jdists.append(jaccard(top_terms[i,:], top_terms[j,:]))
    return np.min(jdists), np.mean(jdists)

class LDAwithCustomScore(LatentDirichletAllocation):
    def score(self, X, y=None):
        components = self.components_
        ntopics = self.n_components
        score = get_jensen_shannon(components, ntopics)[0]
        return score
    

"""# Topic Modeling

Selecting & evaluating the best model
"""

def best_model(documents, ntopics_list):
    processed_info = []
    for allinfo in documents['text'].values:
        preprocessed, stemdict = preprocess(allinfo, stopwords)
        processed_info.append(preprocessed)

    countvec = CountVectorizer(ngram_range=(1,1), stop_words=stopwords, max_df=.25, min_df=10)
    clean_text = [' '.join(text) for text in processed_info]
    X = countvec.fit_transform(clean_text).toarray()
    wft = np.sum(X, axis=0).T

    terms = countvec.get_feature_names_out()

    # Using grid search CV with pipeline to find the best model
    search_params = {'n_components': ntopics_list}
    lda = LDAwithCustomScore(random_state=0)
    model = GridSearchCV(lda, param_grid=search_params, cv=5)
    model.fit(X)

    return model.best_estimator_, processed_info, X, terms, countvec # return the best model

def Evaluate(model, processed_info, X, terms):
  topic_word_distributions = model.components_  # how often the top words in a topic appear together in documents
  top_n_words = 10
  topics = [[terms[i] for i in topic.argsort()[:-top_n_words - 1:-1]] for topic in topic_word_distributions]

  dictionary = gensim.corpora.Dictionary(processed_info)
  coherence_model = CoherenceModel(topics=topics, texts=processed_info,
                                   dictionary=dictionary, coherence='c_v')
  coherence_score = coherence_model.get_coherence()

  print(f'Coherence Score: {coherence_score}')

  perplexity = model.perplexity(X)  # how well the model generalizes to unseen data, lower is better
  print(f'Perplexity: {perplexity}')

  topic_distribution = model.transform(X) # transform the document-term matrix into topic distributions

  return coherence_score, perplexity, topic_distribution