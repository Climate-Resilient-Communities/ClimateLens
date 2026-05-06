import traceback

from dotenv import load_dotenv

load_dotenv() # remove if not needed

from climatelens.nlp_pipeline.emotion_classification import run_emotion_pipeline


def main():
    print("Starting Emotion Classification")

    try:
        run_emotion_pipeline()
    except Exception:
        print("Exception in emotion pipeline:")
        traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()
