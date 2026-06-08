from dotenv import load_dotenv

load_dotenv()

from climatelens.nlp_pipeline.emotion_classification import run_emotion_pipeline
from climatelens.visualizations.emotion_visualization import (
    run_pipeline as run_visualization_pipeline,
)


def main():
    print("Starting Emotion Classification")

    try:
        run_emotion_pipeline()
    except Exception:
        print(f"Exception in emotion pipeline: {Exception}")

    print("Starting Emotion Visualization")

    try:
        run_visualization_pipeline()
    except Exception:
        print(f"Exception in visualization pipeline: {Exception}")

    print("Done.")


if __name__ == "__main__":
    main()
