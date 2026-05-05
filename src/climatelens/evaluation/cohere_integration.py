### this file is used within create_submodels() in topic_modeling.py

import os

from dotenv import load_dotenv

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

from typing import List, Optional, Union

import cohere
from bertopic.representation import Cohere, MaximalMarginalRelevance

mmr_model = MaximalMarginalRelevance(diversity=0.3)


def cohere_integration(KEYWORDS: List[str], DOCUMENTS: List[str]) -> Optional[Cohere]:
    if not cohere_api_key:
        print("No COHERE_API_KEY found in .env file, skipping Cohere representation.")
        return None

    try:
        cohere_client = cohere.Client(cohere_api_key)
        custom_prompt = f"""
        I have a topic described by the following keywords:
        {KEYWORDS}

        The most representative documents for this topic are:
        {DOCUMENTS}

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
            delay_in_seconds=2,
        )
    except Exception as e:
        print(f"Error initializing Cohere integration: {e}")
        return None


cohere_model = cohere_integration()
if cohere_model:
    representation_model: List[Union[MaximalMarginalRelevance, Cohere]] = [mmr_model, cohere_model]
    print("Using MMR + Cohere for representation")
else:
    representation_model = mmr_model
