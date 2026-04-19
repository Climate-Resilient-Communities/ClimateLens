import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment():
    JUPYTER = False
    try:
        import google.colab
        from google.colab import drive
        drive.mount("/content/drive")

        base_path = "/content/drive/MyDrive/ClimateLens/02 Code/02.01 MVP2/"
        env_path = Path(base_path) / "colab.env"
        JUPYTER = True
    except ImportError:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        JUPYTER = False

    if env_path.exists():
        load_dotenv(env_path)
        data_dir, code_dir = os.getenv("DATA_DIR"), os.getenv("CODE_DIR")
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")
    print("Loaded environment variables")

    return data_dir, code_dir, JUPYTER
