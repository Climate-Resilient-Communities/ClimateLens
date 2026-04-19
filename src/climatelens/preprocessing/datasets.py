"""
Dataset registry loader.

Pipeline stages used to dispatch on substrings like ``"twitter" in name.lower()``
to decide which text column, timestamp column, and model to use. That's
fragile — a file rename silently broke routing.

Instead, every dataset now has a declarative entry in ``src/config/datasets.yaml``::

    - name: climate_twitter_sample
      source: twitter
      filename_patterns: ["*twitter*.csv"]
      text_column: text
      timestamp_column: created_at
      timestamp_unit: null
      topic_profile: twitter
      emotion_profile: twitter

The loader returns a list of :class:`DatasetSpec` objects describing the
datasets actually present on disk, so downstream stages can look up the right
columns / profiles without string sniffing.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Single dataset declared in the registry + resolved on-disk path."""

    name: str
    source: str
    path: Path
    text_column: str
    timestamp_column: Optional[str] = None
    timestamp_unit: Optional[str] = None  # e.g. "s" for unix-seconds
    topic_profile: str = "default"
    emotion_profile: str = "default"
    cleaned_text_column: str = "cleaned_text"
    filename_patterns: List[str] = field(default_factory=list)

    def processed_path(self, processed_dir: Path) -> Path:
        """Where to write the preprocessed copy of this dataset."""
        return processed_dir / f"{self.name}.csv"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "config" / "datasets.yaml"


def load_registry(registry_path: Optional[Path] = None) -> List[dict]:
    """Read the raw dataset registry (list of dicts) from YAML."""
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY
    if not path.exists():
        raise FileNotFoundError(f"Dataset registry not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"Dataset registry must be a YAML list; got {type(data)}")
    return data


def _match_one(data_dir: Path, patterns: Iterable[str]) -> Optional[Path]:
    """Find the first file in *data_dir* matching any of *patterns*."""
    for pattern in patterns:
        matches = sorted(p for p in data_dir.iterdir() if fnmatch.fnmatch(p.name, pattern))
        if matches:
            return matches[0]
    return None


def discover_datasets(
    data_dir: Path,
    registry_path: Optional[Path] = None,
    *,
    require_all: bool = False,
) -> List[DatasetSpec]:
    """
    Resolve every dataset in the registry against files present in *data_dir*.

    Args:
        data_dir:       Directory containing input CSVs.
        registry_path:  Override path to the YAML registry.
        require_all:    If ``True``, raise ``FileNotFoundError`` when any
                        registry entry has no matching file on disk.

    Returns:
        Datasets that were actually found on disk. The caller controls
        strictness via ``require_all``.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    registry = load_registry(registry_path)
    specs: List[DatasetSpec] = []
    missing: List[str] = []

    for entry in registry:
        patterns = entry.get("filename_patterns") or []
        resolved = _match_one(data_dir, patterns)
        if resolved is None:
            missing.append(entry.get("name", "<unnamed>"))
            continue

        specs.append(
            DatasetSpec(
                name=entry["name"],
                source=entry.get("source", "unknown"),
                path=resolved,
                text_column=entry.get("text_column", "text"),
                timestamp_column=entry.get("timestamp_column"),
                timestamp_unit=entry.get("timestamp_unit"),
                topic_profile=entry.get("topic_profile", "default"),
                emotion_profile=entry.get("emotion_profile", "default"),
                cleaned_text_column=entry.get("cleaned_text_column", "cleaned_text"),
                filename_patterns=list(patterns),
            )
        )

    if require_all and missing:
        raise FileNotFoundError(
            "Datasets declared in registry but missing on disk: " + ", ".join(missing)
        )

    return specs
