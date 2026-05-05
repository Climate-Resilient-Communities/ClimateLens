"""
Emotion classification stage.

Reads the preprocessed datasets from ``runtime.processed_data_dir`` (with a
fallback to ``runtime.data_dir``), runs an emotion classifier, and
writes ``<name>_with_emotions.csv`` into ``runtime.output_data_dir``.

The model families are selected per-dataset via the ``emotion_profile`` field
in ``src/config/datasets.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

# Add src/ to sys.path so utils imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.datasets import DatasetSpec, discover_datasets  # noqa: E402
from utils.io_helpers import drop_missing_text, safe_write_csv  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402
from utils.runtime import load_runtime  # noqa: E402

log = get_logger(__name__)


# =============================================================================
# Model selection
# =============================================================================

_EMOTION_MODELS: Dict[str, str] = {
    "twitter": "boltuix/bert-emotion",  # 13 labels
    "reddit": "SamLowe/roberta-base-go_emotions",  # 28 labels
    "default": "SamLowe/roberta-base-go_emotions",
}


def load_models(profiles: Iterable[str]) -> Dict[str, object]:
    """
    Load each emotion model we'll need.

    ``profiles`` should be the ``emotion_profile`` field of each dataset being
    processed; we load each profile's model exactly once.
    """
    device = 0 if torch.cuda.is_available() else -1

    models = {}
    for profile in sorted(set(profiles) | {"default"}):
        model_name = _EMOTION_MODELS.get(profile, _EMOTION_MODELS["default"])
        key = f"emotion_{profile}"
        log.info("loading emotion model for profile=%s (%s)", profile, model_name)
        models[key] = pipeline("text-classification", model=model_name, top_k=None, device=device)
    return models


def choose_emotion_model(profile: str, models: Dict[str, object]):
    """Return the emotion pipeline registered for *profile* (or default)."""
    return models.get(f"emotion_{profile}", models["emotion_default"])


# =============================================================================
# Analysis
# =============================================================================


def emotion_analysis(
    df: pd.DataFrame,
    analyzer,
    text_col: Optional[str] = None,
    batch_size: int = 128,
    multi: bool = False,
) -> pd.DataFrame:
    """Run emotion classification and attach label/proba (and top-3 if ``multi``)."""
    if text_col is None:
        text_col = next((c for c in ("body", "text") if c in df.columns), None)
        if text_col is None:
            raise ValueError("no valid text column found for emotion analysis")

    texts = df[text_col].tolist()
    labels, confs, all_emotions = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="emotion"):
        batch = texts[i : i + batch_size]
        results = analyzer(batch, padding=True, truncation=True, max_length=512)

        if multi:
            for emotion_list in results:
                top_k = sorted(emotion_list, key=lambda x: x["score"], reverse=True)[:3]
                labels.append(top_k[0]["label"])
                confs.append(top_k[0]["score"])
                all_emotions.append(top_k)
        else:
            for r in results:
                labels.append(r["label"])
                confs.append(r["score"])

    df["emotion_label"] = labels
    df["emotion_proba"] = confs
    if multi:
        df["all_emotions"] = all_emotions
        df["top3_emotions"] = df["all_emotions"].apply(
            lambda lst: ", ".join(f"{x['label']} ({x['score']:.2f})" for x in lst)
        )

    return df


# =============================================================================
# Pipeline
# =============================================================================


def _pick_source_dir(runtime) -> Path:
    """Prefer preprocessed data; fall back to raw with a warning."""
    processed = runtime.processed_data_dir
    if any(processed.glob("*.csv")):
        return processed
    log.warning(
        "no CSVs in %s; falling back to %s. Did you run data_preprocessing.py first?",
        processed,
        runtime.data_dir,
    )
    return runtime.data_dir


def _process_one(spec: DatasetSpec, models: Dict[str, object]) -> Optional[pd.DataFrame]:
    """Load, emotion-classify, and return a single dataset."""
    df = pd.read_csv(spec.path)
    text_col = (
        spec.cleaned_text_column if spec.cleaned_text_column in df.columns else spec.text_column
    )
    if text_col not in df.columns:
        log.warning(
            "skipping %s: no text column (%r or %r) in %s",
            spec.name,
            spec.text_column,
            spec.cleaned_text_column,
            list(df.columns),
        )
        return None

    df = drop_missing_text(df, text_col)
    analyzer = choose_emotion_model(spec.emotion_profile, models)

    try:
        return emotion_analysis(df, analyzer, text_col=text_col, multi=True)
    except TypeError:
        log.warning("multi-emotion failed for %s, retrying single-label", spec.name)
        return emotion_analysis(df, analyzer, text_col=text_col, multi=False)


def main() -> int:
    runtime = load_runtime()
    log.info("emotion classification starting")
    log.info("paths:\n%s", runtime.describe())

    source_dir = _pick_source_dir(runtime)
    specs = discover_datasets(source_dir)
    if not specs:
        log.error("no datasets matched the registry in %s", source_dir)
        return 1

    models = load_models(profiles=[s.emotion_profile for s in specs])

    results: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        log.info("analyzing %s (%s)", spec.name, spec.path.name)
        try:
            df = _process_one(spec, models)
        except Exception:
            log.exception("emotion analysis failed for %s", spec.name)
            continue
        if df is None:
            continue

        results[spec.name] = df
        out = runtime.output_data_dir / f"{spec.name}_with_emotions.csv"
        safe_write_csv(df, out)
        log.info("wrote %s", out)

    if not results:
        log.error("no datasets produced emotion output")
        return 1

    # Lightweight visual summary per dataset.
    runtime.output_vis_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        log.warning("matplotlib not installed; skipping emotion count plots")

    if plt is not None:
        for name, df in results.items():
            try:
                fig = df["emotion_label"].value_counts().plot(kind="bar").get_figure()
                out = runtime.output_vis_dir / f"{name}_emotion_counts.png"
                fig.savefig(out, dpi=200, bbox_inches="tight")
                fig.clf()
                log.info("saved visualization: %s", out)
            except Exception:
                log.exception("visualization failed for %s", name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        log.exception("unhandled exception in emotion classification pipeline")
        raise
