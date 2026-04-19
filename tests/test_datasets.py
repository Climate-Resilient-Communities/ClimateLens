"""Tests for src/utils/datasets.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils import datasets


def _write_registry(path: Path, entries):
    path.write_text(yaml.safe_dump(entries, sort_keys=False))


def test_load_registry_default_exists():
    assert datasets.DEFAULT_REGISTRY.exists()
    registry = datasets.load_registry()
    assert isinstance(registry, list)
    assert all("name" in entry and "filename_patterns" in entry for entry in registry)


def test_load_registry_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        datasets.load_registry(tmp_path / "missing.yaml")


def test_load_registry_rejects_non_list(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not: a list")
    with pytest.raises(ValueError):
        datasets.load_registry(path)


def test_discover_datasets_resolves_by_pattern(tmp_path):
    registry = [
        {
            "name": "twitter_sample",
            "source": "twitter",
            "filename_patterns": ["*twitter*.csv"],
            "text_column": "text",
            "timestamp_column": "created_at",
            "timestamp_unit": None,
            "topic_profile": "twitter",
            "emotion_profile": "twitter",
        },
    ]
    reg_path = tmp_path / "datasets.yaml"
    _write_registry(reg_path, registry)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "climate_twitter.csv").write_text("text\nhi\n")

    specs = datasets.discover_datasets(data_dir, reg_path)

    assert len(specs) == 1
    assert specs[0].name == "twitter_sample"
    assert specs[0].path.name == "climate_twitter.csv"
    assert specs[0].topic_profile == "twitter"


def test_discover_datasets_skips_missing_by_default(tmp_path):
    registry = [
        {"name": "a", "filename_patterns": ["a.csv"], "text_column": "text"},
        {"name": "b", "filename_patterns": ["b.csv"], "text_column": "text"},
    ]
    reg_path = tmp_path / "datasets.yaml"
    _write_registry(reg_path, registry)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.csv").write_text("text\nx\n")

    specs = datasets.discover_datasets(data_dir, reg_path)
    assert [s.name for s in specs] == ["a"]


def test_discover_datasets_require_all_raises(tmp_path):
    registry = [
        {"name": "a", "filename_patterns": ["a.csv"], "text_column": "text"},
        {"name": "b", "filename_patterns": ["b.csv"], "text_column": "text"},
    ]
    reg_path = tmp_path / "datasets.yaml"
    _write_registry(reg_path, registry)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.csv").write_text("text\nx\n")

    with pytest.raises(FileNotFoundError, match="b"):
        datasets.discover_datasets(data_dir, reg_path, require_all=True)


def test_discover_datasets_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        datasets.discover_datasets(tmp_path / "nope")


def test_processed_path_uses_registry_name(tmp_path):
    spec = datasets.DatasetSpec(
        name="my_ds",
        source="twitter",
        path=tmp_path / "raw.csv",
        text_column="text",
    )
    assert spec.processed_path(tmp_path / "out") == tmp_path / "out" / "my_ds.csv"
