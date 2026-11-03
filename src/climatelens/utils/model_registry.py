
"""Load and validate pinned model definitions from the ClimateLens registry."""

from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2] / "config" / "models.yaml"
)

SUPPORTED_LOADERS = {
    "transformers",
    "sentence_transformers",
}


def load_registry(
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Load and validate the complete model registry."""

    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY

    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Model registry must be a YAML mapping.")

    models = data.get("models")
    assignments = data.get("assignments", {})

    if not isinstance(models, dict) or not models:
        raise ValueError(
            "Model registry must contain a non-empty 'models' mapping."
        )

    if not isinstance(assignments, dict):
        raise ValueError("'assignments' must be a YAML mapping.")

    # Validate model definitions
    for name, config in models.items():

        if not isinstance(config, dict):
            raise ValueError(
                f"Model '{name}' must have a mapping configuration."
            )

        for field in ("model_id", "revision", "task", "loader"):
            value = config.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Model '{name}' must define a non-empty '{field}'."
                )

        # Validate immutable Hugging Face commit SHA
        revision = config["revision"]

        if (
            len(revision) != 40
            or any(c not in "0123456789abcdefABCDEF" for c in revision)
        ):
            raise ValueError(
                f"Model '{name}' revision must be a full "
                "40-character commit SHA."
            )

        # Validate loading interface
        if config["loader"] not in SUPPORTED_LOADERS:
            raise ValueError(
                f"Model '{name}' has unsupported loader: "
                f"{config['loader']}"
            )

        # Validate configuration and metadata
        for field in ("config", "metadata"):
            value = config.get(field, {})

            if not isinstance(value, dict):
                raise ValueError(
                    f"Model '{name}' field '{field}' must be a mapping."
                )

    # Validate dataset-to-model assignments
    for task, dataset_assignments in assignments.items():

        if not isinstance(dataset_assignments, dict):
            raise ValueError(
                f"Assignment '{task}' must be a mapping."
            )

        for dataset, model_name in dataset_assignments.items():

            if model_name not in models:
                raise ValueError(
                    f"Assignment '{task}.{dataset}' references "
                    f"unknown model '{model_name}'."
                )

    return {
        "models": models,
        "assignments": assignments,
    }


def load_model_registry(
    registry_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load model definitions only (backward-compatible interface)."""

    return load_registry(registry_path)["models"]


def get_model_config(
    model_name: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve a model configuration by its registry name."""

    registry = registry or load_registry()

    try:
        return registry["models"][model_name]
    except KeyError:
        raise KeyError(
            f"Model '{model_name}' not found in registry."
        ) from None


def resolve_model_assignment(
    task: str,
    dataset: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the model assigned to a task and dataset."""

    registry = registry or load_registry()

    assignments = registry["assignments"]

    if task not in assignments:
        raise KeyError(f"Task '{task}' has no model assignments.")

    task_assignments = assignments[task]

    model_name = task_assignments.get(
        dataset,
        task_assignments.get("default"),
    )

    if model_name is None:
        raise KeyError(
            f"No model assigned to task '{task}' "
            f"for dataset '{dataset}', and no default exists."
        )

    return get_model_config(model_name, registry)
