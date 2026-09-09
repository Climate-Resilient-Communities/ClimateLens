"""
Command-line entry point for the ClimateLens pipeline (ticket #36).

Installed as the ``climatelens`` console script; see ``[project.scripts]`` in
pyproject.toml. Run ``climatelens run --help`` for options.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import click


def _load_pipelines() -> Tuple[Callable[..., int], Callable[..., None]]:
    """
    Import the pipeline callables on demand.

    Deliberately not a module-level import: the emotion stages pull in torch
    and transformers, and keeping that off the import path means ``--help``
    (and the CLI tests) work without the heavy ML stack installed. It also
    gives the tests a single seam to patch.
    """
    from climatelens.nlp_pipeline.emotion_classification import run_emotion_pipeline
    from climatelens.visualizations.emotion_visualization import (
        run_pipeline as run_visualization_pipeline,
    )

    return run_emotion_pipeline, run_visualization_pipeline


@click.group()
def cli() -> None:
    """ClimateLens pipeline commands."""


@cli.command()
@click.option(
    "--sample-size",
    type=click.IntRange(min=1),
    default=None,
    help="Process only this many rows per dataset, drawn as a seeded random sample.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Seed for --sample-size.",
)
def run(sample_size: Optional[int], seed: int) -> None:
    """
    Run emotion classification, then the emotion visualizations.

    Exits non-zero if either stage fails. Exceptions are left to propagate so
    the traceback reaches the caller rather than being reported as a success.
    """
    run_emotion_pipeline, run_visualization_pipeline = _load_pipelines()

    status = run_emotion_pipeline(sample_size=sample_size, seed=seed)
    if status:
        raise SystemExit(status)

    run_visualization_pipeline(sample_size=sample_size, seed=seed)


if __name__ == "__main__":  # pragma: no cover
    cli()
