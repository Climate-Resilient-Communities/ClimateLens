### this file is used within the create_submodels() function in topic_modeling.py

import os
from bertopic import BERTopic
import cohere
from bertopic.representation import Cohere
from bertopic.representation import MaximalMarginalRelevance
from pathlib import Path

#need to do env handeling later

mmr_model = MaximalMarginalRelevance(diversity=0.3)

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

cohere_model = cohere_integration()
    if cohere_model:
        representation_model = [mmr_model, cohere_model]
        print(f"Using MMR + Cohere for representation")
    else:
        representation_model = mmr_model