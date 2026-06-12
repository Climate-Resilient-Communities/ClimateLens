"""
Runtime/environment helpers for the ClimateLens pipeline.

Centralizes three concerns that were duplicated across every script:

1. Detecting where we're running (local dev, Jupyter/Colab, AzureML job).
2. Locating and loading the .env file.
3. Resolving canonical input/output directories.

Scripts should call :func:`load_runtime` once near the top of ``main()`` and
then read paths from the returned :class:`RuntimeConfig` instead of touching
``os.environ`` directly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

_AZUREML_SIGNALS = (
    "AZUREML_RUN_ID",
    "AZUREML_EXPERIMENT_ID",
    "AZUREML_OUTPUT_DIR",
)


def is_azureml() -> bool:
    """Return ``True`` when any AzureML environment variable is present."""
    return any(name in os.environ for name in _AZUREML_SIGNALS)


def is_colab() -> bool:
    """Return ``True`` when running under Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------


def find_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """
    Walk up from *start* looking for a ``.env`` file.

    Returns the path to the first ``.env`` found, or ``None`` if none exists.
    When *start* is ``None`` we start from the caller's working directory.
    """
    start = (start or Path.cwd()).resolve()
    for parent in [start] + list(start.parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def load_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """
    Locate and load a ``.env`` file. No-ops (returns ``None``) if none found.

    AzureML jobs never have a .env checked in, so callers should treat a
    missing file as non-fatal and fall back to real environment variables.
    """
    env_path = find_dotenv(start)
    if env_path is not None:
        load_dotenv(env_path, override=False)
    return env_path


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------

# Type alias for clarity
TimeLog = Tuple[str, float]
time_logs: List[TimeLog] = []


# timer decorator
def timer_dec(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:  # ensure we log time even if an exception occurs
            end_time = time.time()
            execution_time = end_time - start_time
            # instead of printing, we store the time in a list (or a file)
            log_time(func.__name__, execution_time)
        return result

    return wrapper


time_logs = []


def log_time(function_name: str, execution_time: float) -> None:
    print(f"Function '{function_name}' executed in {execution_time:.4f} seconds.")
    time_logs.append((function_name, execution_time))


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Canonical directory layout for one pipeline run.

    - ``data_dir``            : READ-ONLY source data (raw / filtered CSVs)
    - ``processed_data_dir``  : WRITE   preprocessed + topic-annotated CSVs
    - ``output_data_dir``     : WRITE   emotion-classified CSVs
    - ``output_vis_dir``      : WRITE   visualizations (HTML, PNG)
    - ``models_dir``          : WRITE   serialized topic models
    - ``env_path``            : The .env file that was loaded, if any
    - ``in_azureml``          : Whether we detected an AzureML job context
    """

    data_dir: Path
    processed_data_dir: Path
    output_data_dir: Path
    output_vis_dir: Path
    models_dir: Path
    env_path: Optional[Path] = None
    in_azureml: bool = False
    extras: Dict[str, Path] = field(default_factory=dict)

    def ensure_writable_dirs(self) -> None:
        """Create the write-side directories (idempotent)."""
        for p in (
            self.processed_data_dir,
            self.output_data_dir,
            self.output_vis_dir,
            self.models_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        """Render a human-readable path summary for logging."""
        lines = [
            f"  data_dir           = {self.data_dir}",
            f"  processed_data_dir = {self.processed_data_dir}",
            f"  output_data_dir    = {self.output_data_dir}",
            f"  output_vis_dir     = {self.output_vis_dir}",
            f"  models_dir         = {self.models_dir}",
            f"  in_azureml         = {self.in_azureml}",
        ]
        if self.env_path:
            lines.append(f"  env_path           = {self.env_path}")
        for k, v in self.extras.items():
            lines.append(f"  {k:<18} = {v}")
        return "\n".join(lines)


def _path_from_env(name: str, default: Path) -> Path:
    """Return the env var *name* as a Path, or *default* if unset/empty."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def load_runtime(
    *,
    search_from: Optional[Path] = None,
    extras: Iterable[str] = (),
    ensure_dirs: bool = True,
) -> RuntimeConfig:
    """
    Build a :class:`RuntimeConfig` for the current run.

    AzureML jobs get a fixed ``outputs/`` layout. Local runs pull directory
    paths from env vars with sensible defaults, so a freshly-cloned repo
    "just works" against the sample data committed in ``./data``.

    Args:
        search_from: Where to start looking for ``.env``. Defaults to cwd.
        extras:      Additional env var names to expose in ``RuntimeConfig.extras``.
        ensure_dirs: If ``True``, create all write directories eagerly.
    """
    env_path = None if is_azureml() else load_env_file(search_from)

    in_azureml = is_azureml()
    if in_azureml:
        data_dir = Path("./data")
        processed_data_dir = Path("./outputs/data")
        output_data_dir = Path("./outputs/data")
        output_vis_dir = Path("./outputs/visualizations")
        models_dir = Path("./outputs/models")
    else:
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = _path_from_env("DATA_DIR", repo_root / "data")
        processed_data_dir = _path_from_env(
            "PROCESSED_DATA_DIR", repo_root / "outputs" / "processed"
        )
        output_data_dir = _path_from_env("OUTPUT_DATA_DIR", repo_root / "outputs" / "data")
        output_vis_dir = _path_from_env(
            "OUTPUT_VIS_DIR",
            _path_from_env("OUTPUT_DIR", repo_root / "outputs" / "visualizations"),
        )
        models_dir = _path_from_env("MODELS_DIR", repo_root / "outputs" / "models")

    extras_map: Dict[str, Path] = {}
    for name in extras:
        value = os.environ.get(name)
        if value:
            extras_map[name] = Path(value).expanduser()

    cfg = RuntimeConfig(
        data_dir=data_dir,
        processed_data_dir=processed_data_dir,
        output_data_dir=output_data_dir,
        output_vis_dir=output_vis_dir,
        models_dir=models_dir,
        env_path=env_path,
        in_azureml=in_azureml,
        extras=extras_map,
    )

    if ensure_dirs:
        cfg.ensure_writable_dirs()

    return cfg
