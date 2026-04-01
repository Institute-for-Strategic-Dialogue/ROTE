"""Shared fixtures for keyword_discovery tests."""

import json
import pytest
import pandas as pd

from tools.keyword_discovery.corpus import Corpus, load_corpus


@pytest.fixture
def small_corpus_csv() -> bytes:
    """A small CSV corpus as bytes."""
    rows = [
        "The globalist cabal controls the media and international bankers.",
        "Wake up sheeple, do your own research about the new world order.",
        "Cats are friendly and cute animals that love to play.",
        "The great replacement is a conspiracy theory promoted by extremists.",
        "I went to the store to buy groceries and came home.",
        "Globalist elites are pushing cultural marxism in our schools.",
        "Just asking questions about the rothschild family and their influence.",
        "The weather is nice today, perfect for a walk in the park.",
        "Urban youth and thugs are coded language for racial profiling.",
        "Do your own research before believing what the mainstream media says.",
        "Free helicopter rides is a reference to political violence.",
        "The cat sat on the mat and purred contentedly.",
        "The cabal and the new world order are controlling everything.",
        "Wake up to the truth about globalist agendas and soros-funded groups.",
        "I love reading books about history and science fiction.",
        "Race realist is a euphemism for scientific racism and bigotry.",
        "The identitarian movement promotes the great replacement theory.",
        "Coffee is my favourite morning drink, I have it every day.",
        "Crisis actors and false flags are common conspiracy claims.",
        "The globalist agenda threatens national sovereignty worldwide.",
    ]
    df = pd.DataFrame({"text": rows, "id": range(len(rows))})
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def small_corpus(small_corpus_csv) -> Corpus:
    """A loaded small Corpus object."""
    return load_corpus(small_corpus_csv, "test.csv", ngram_range=(1, 2), min_df=1)


@pytest.fixture
def dogwhistle_dict_sample() -> dict:
    """A small sample dogwhistle dictionary."""
    return {
        "globalist": {
            "decoded_meaning": "antisemitic trope",
            "category": "antisemitism",
            "confidence": "high",
            "related_terms": ["cosmopolitan elite", "international bankers"]
        },
        "great replacement": {
            "decoded_meaning": "white supremacist conspiracy theory",
            "category": "white supremacism",
            "confidence": "high",
            "related_terms": ["white genocide", "demographic replacement"]
        },
    }
