"""Unit tests for pure helpers in src/data_preprocessing.py."""

from __future__ import annotations

import src.climatelens.preprocessing.data_preprocessing as dp
import pytest


@pytest.fixture(scope="module")
def stopwords():
    return dp.build_custom_stopwords()


def test_build_custom_stopwords_preserves_negations(stopwords):
    for word in ("not", "no", "nor", "why", "what", "how"):
        assert word not in stopwords, f"{word!r} should be preserved"


def test_build_custom_stopwords_includes_twitter_artifacts(stopwords):
    for token in ("https", "http", "rt", "amp"):
        assert token in stopwords


def test_remove_consecutive_repeats_dedupes_adjacent():
    assert dp.remove_consecutive_repeats(["a", "a", "b", "b", "a"]) == ["a", "b", "a"]


def test_remove_consecutive_repeats_empty():
    assert dp.remove_consecutive_repeats([]) == []


def test_preprocess_text_strips_urls_and_handles(stopwords):
    raw = "RT @bob Check https://t.co/x www.example.com &amp; more"
    cleaned = dp.preprocess_text(raw, stopwords)
    assert "https" not in cleaned
    assert "@bob" not in cleaned
    assert "www" not in cleaned
    assert "&amp" not in cleaned
    assert "rt" not in cleaned.split()


def test_preprocess_text_lowercases_and_tokenizes(stopwords):
    cleaned = dp.preprocess_text("HELLO Climate Grief", stopwords)
    # "climate" is a project-stopword and should be removed.
    assert "climate" not in cleaned
    assert "grief" in cleaned
    assert cleaned == cleaned.lower()


def test_preprocess_text_drops_consecutive_repeats(stopwords):
    cleaned = dp.preprocess_text("grief grief anxiety anxiety", stopwords)
    # Consecutive dupes collapse to a single occurrence.
    assert cleaned.split() == ["grief", "anxiety"]


def test_highlight_issues_reports_slang_and_repeats():
    repeats, slang = dp.highlight_issues("damn that shit shit shit is bad")
    assert "shit" in slang
    assert "damn" in slang
    assert "shit" in repeats
