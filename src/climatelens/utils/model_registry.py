# Load pinned model definitions from the model registry

from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def load_model_registry(
    registry_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    # Load and validate model definitions from YAML
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY

    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Model registry must be a YAML mapping.")

    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("Model registry must contain a 'models' mapping.")

    for name, config in models.items():
        if not isinstance(config, dict):
            raise ValueError(f"Model '{name}' must have a mapping configuration.")

        for field in ("model_id", "revision", "task"):
            if not isinstance(config.get(field), str) or not config[field].strip():
                raise ValueError(f"Model '{name}' must define a non-empty '{field}'.")

        revision = config["revision"]
        if len(revision) != 40 or any(char not in "0123456789abcdefABCDEF" for char in revision):
            raise ValueError(f"Model '{name}' revision must be a full 40-character commit SHA.")

        model_config = config.get("config", {})
        if not isinstance(model_config, dict):
            raise ValueError(f"Model '{name}' field 'config' must be a mapping.")

    return models
