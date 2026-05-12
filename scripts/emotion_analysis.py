import traceback

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
        print("Exception in emotion pipeline:")
        traceback.print_exc()

    print("Starting Emotion Visualization")

    try:
        run_visualization_pipeline()
    except Exception:
        print("Exception in visualization pipeline:")
        traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()
