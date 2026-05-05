from datetime import datetime
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo


def create_directories(base_dir: str, dir_names: List[str],
                       use_timestamp: bool = False) -> Dict[str, Path]:
    base_dir = Path(base_dir)

    if use_timestamp:
        current_time = datetime.now(
            ZoneInfo("America/New_York")
        ).strftime("%m %d - %H %M")
        base_dir = base_dir / current_time

    created_paths = {}

    for name in dir_names:
        path = base_dir / name
        path.mkdir(parents=True, exist_ok=True)
        created_paths[name] = path

    return created_paths

# example:
paths = create_directories(
    "outputs", # following AML Job convention
    [
        "models",
        "visualizations/IDM",
        "visualizations/hierarchies"
    ],
    use_timestamp=True
)
