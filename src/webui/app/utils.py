from pathlib import Path

def get_visualizations(folder):
    valid_extensions = {".html", ".png", ".jpg", ".jpeg", ".webp"}

    return sorted([
        f
        for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])