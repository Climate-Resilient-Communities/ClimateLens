from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


def compute_embeddings(
    docs_dict: Dict[str, List[str]],
) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, SentenceTransformer]]:
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

    embeddings_dict: Dict[str, NDArray[np.float32]] = {}
    embedding_models: Dict[str, SentenceTransformer] = {}
    # cache models?
    # model_cache: Dict[str, SentenceTransformer] = {}

    for name, docs in docs_dict.items():
        model_name = DEFAULT_EMBEDDING_MODEL

        # Reuse model if already loaded
        if model_name not in embedding_models:
            embedding_models[model_name] = SentenceTransformer(model_name)

        embedding_model = embedding_models[model_name]

        print(f"Computing {name} embeddings using {model_name}...")
        embeddings_dict[name] = embedding_model.encode(
            docs, batch_size=128, show_progress_bar=True
        )
        embedding_models[name] = embedding_model

    return embeddings_dict, embedding_models
