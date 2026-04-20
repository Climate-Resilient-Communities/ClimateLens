from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def create_directories(code_dir):
    current_time = datetime.now(ZoneInfo("America/New_York")).strftime("%m %d - %H %M Hours")
    print(f"Current time: {current_time}")

    outputs_dir = Path(code_dir) / "outputs"
    base = outputs_dir / current_time

    directories = {
        "models": base / "models",
        "IDM": base / "visualizations" / "IDM",
        "hierarchies": base / "visualizations" / "hierarchies",
        "barcharts": base / "visualizations" / "barcharts",
        "dtm": base / "visualizations" / "dtm",
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return (
        directories["models"],
        directories["IDM"],
        directories["hierarchies"],
        directories["barcharts"],
        directories["dtm"],
    )
