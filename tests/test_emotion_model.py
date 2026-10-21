"""Tests for the ClimateLens emotion classification stage."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
em = pytest.importorskip("src.climatelens.nlp_pipeline.emotion_classification", reason="heavy ML deps not installed")


@pytest.fixture
def models():
    return {
        "emotion_default": MagicMock(),
        "emotion_twitter": MagicMock(),
        "emotion_reddit": MagicMock(),
    }


# ------------------------------------------------------------------
# Model selection
# ------------------------------------------------------------------

def test_choose_emotion_model_uses_profile(models):
    assert em.choose_emotion_model("twitter", models) is models["emotion_twitter"]
    assert em.choose_emotion_model("reddit", models) is models["emotion_reddit"]


def test_choose_emotion_model_falls_back_to_default():
    models = {"emotion_default": "default"}

    assert em.choose_emotion_model("unknown", models) == "default"


def test_load_models_loads_unique_profiles(monkeypatch):
    pipeline = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(em, "pipeline", pipeline)
    monkeypatch.setattr(em.torch.cuda, "is_available", lambda: False)

    result = em.load_models(["twitter", "reddit", "twitter"])

    assert set(result) == {
        "emotion_default", "emotion_twitter", "emotion_reddit"
    }
    assert pipeline.call_count == 3


# ------------------------------------------------------------------
# Emotion analysis
# ------------------------------------------------------------------

def test_emotion_analysis_adds_top_three_emotions():
    df = pd.DataFrame({"body": ["Climate change worries me."]})

    def analyzer(texts, **kwargs):
        return [[
            {"label": "fear", "score": 0.9},
            {"label": "sadness", "score": 0.8},
            {"label": "anger", "score": 0.7},
        ]]

    result = em.emotion_analysis(df, analyzer, multi=True)

    assert result.loc[0, "emotion_label"] == "fear"
    assert result.loc[0, "emotion_proba"] == 0.9
    assert len(result.loc[0, "all_emotions"]) == 3
    assert "fear (0.90)" in result.loc[0, "top3_emotions"]


@pytest.mark.parametrize("text_column", ["body", "text"])
def test_emotion_analysis_detects_text_column(text_column):
    df = pd.DataFrame({text_column: ["First post", "Second post"]})
    analyzer = MagicMock(return_value=[
        {"label": "joy", "score": 0.9},
        {"label": "fear", "score": 0.8},
    ])

    result = em.emotion_analysis(df, analyzer, multi=False)

    assert result["emotion_label"].tolist() == ["joy", "fear"]
    assert analyzer.call_args.args[0] == ["First post", "Second post"]


def test_emotion_analysis_prefers_body_when_both_columns_exist():
    df = pd.DataFrame({"body": ["Reddit text"], "text": ["Twitter text"]})
    analyzer = MagicMock(return_value=[{"label": "joy", "score": 0.9}])

    em.emotion_analysis(df, analyzer, multi=False)

    assert analyzer.call_args.args[0] == ["Reddit text"]


def test_emotion_analysis_raises_without_text_column():
    df = pd.DataFrame({"subreddit": ["climate"]})

    with pytest.raises(ValueError, match="no valid text column"):
        em.emotion_analysis(df, MagicMock())


def test_emotion_analysis_processes_batches():
    df = pd.DataFrame({"body": ["a", "b", "c"]})
    analyzer = MagicMock(side_effect=[
        [{"label": "joy", "score": 0.9}] * 2,
        [{"label": "fear", "score": 0.8}],
    ])

    result = em.emotion_analysis(
        df, analyzer, batch_size=2, multi=False
    )

    assert analyzer.call_count == 2
    assert result["emotion_label"].tolist() == ["joy", "joy", "fear"]


def test_emotion_analysis_handles_empty_dataframe():
    df = pd.DataFrame({"body": []})
    analyzer = MagicMock(return_value=[])

    result = em.emotion_analysis(df, analyzer, multi=False)

    assert result.empty
    assert "emotion_label" in result.columns
    analyzer.assert_not_called()


# ------------------------------------------------------------------
# Dataset and source handling
# ------------------------------------------------------------------

def test_pick_source_dir_prefers_processed_data(tmp_path):
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    (processed / "reddit.csv").write_text("body\nhello\n")

    runtime = SimpleNamespace(
        processed_data_dir=processed,
        data_dir=raw,
    )

    assert em._pick_source_dir(runtime) == processed


def test_pick_source_dir_falls_back_to_raw_data(tmp_path):
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()

    runtime = SimpleNamespace(
        processed_data_dir=processed,
        data_dir=raw,
    )

    assert em._pick_source_dir(runtime) == raw


def test_process_one_uses_dataset_text_column(
    tmp_path, models, monkeypatch
):
    path = tmp_path / "reddit.csv"
    pd.DataFrame({"body": ["Climate post"]}).to_csv(path, index=False)

    spec = SimpleNamespace(
        name="reddit",
        path=path,
        text_column="body",
        cleaned_text_column="cleaned_text",
        emotion_profile="reddit",
    )

    expected = pd.DataFrame({"body": ["Climate post"]})
    analysis = MagicMock(return_value=expected)
    monkeypatch.setattr(em, "emotion_analysis", analysis)

    result = em._process_one(spec, models)

    assert result is expected
    assert analysis.call_args.kwargs["text_col"] == "body"
    assert analysis.call_args.kwargs["multi"] is True
    assert analysis.call_args.args[1] is models["emotion_reddit"]


def test_process_one_skips_missing_text_column(
    tmp_path, models, monkeypatch
):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"id": [1]}).to_csv(path, index=False)

    spec = SimpleNamespace(
        name="invalid",
        path=path,
        text_column="body",
        cleaned_text_column="cleaned_text",
        emotion_profile="default",
    )

    analysis = MagicMock()
    monkeypatch.setattr(em, "emotion_analysis", analysis)

    assert em._process_one(spec, models) is None
    analysis.assert_not_called()


def test_process_one_retries_single_label_after_type_error(
    tmp_path, models, monkeypatch
):
    path = tmp_path / "reddit.csv"
    pd.DataFrame({"body": ["Climate post"]}).to_csv(path, index=False)

    spec = SimpleNamespace(
        name="reddit",
        path=path,
        text_column="body",
        cleaned_text_column="cleaned_text",
        emotion_profile="reddit",
    )

    expected = pd.DataFrame({"body": ["Climate post"]})
    analysis = MagicMock(
        side_effect=[TypeError("multi unsupported"), expected]
    )
    monkeypatch.setattr(em, "emotion_analysis", analysis)

    assert em._process_one(spec, models) is expected
    assert analysis.call_count == 2
    assert analysis.call_args_list[0].kwargs["multi"] is True
    assert analysis.call_args_list[1].kwargs["multi"] is False
