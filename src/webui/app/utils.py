import os
import glob

def get_files(folder, extensions=("html",)):
    files = []

    for ext in extensions:
        files.extend(
            glob.glob(os.path.join(folder, f"*.{ext}"))
        )

    return files