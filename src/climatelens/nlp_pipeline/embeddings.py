from sentence_transformers import SentenceTransformer


def compute_embeddings(docs_dict):
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

    embeddings_dict = {}
    embedding_models = {}

    for name, docs in docs_dict.items():
        model_name = DEFAULT_EMBEDDING_MODEL

        if model_name not in embedding_models:
            embedding_models[model_name] = SentenceTransformer(model_name)

        embedding_model = embedding_models[model_name]

        print(f"Computing {name} embeddings using {model_name}...")
        embeddings_dict[name] = embedding_model.encode(
            docs, batch_size=128, show_progress_bar=True
        )
        embedding_models[name] = embedding_model

    return embeddings_dict, embedding_models
