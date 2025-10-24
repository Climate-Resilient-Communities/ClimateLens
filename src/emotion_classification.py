import os
import re
import traceback
from dotenv import load_dotenv
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

def setup_environment():
    try:
        import google.colab
        from google.colab import drive
        drive.mount("/content/drive")

        print("Installing dependencies...")
        !pip install -q emoji==0.6.0

        print("Environment setup complete.")
        return True

    except ImportError:
        return False

def load_env():
  if setup_environment():
    base_path = "..."
    env_path = Path(base_path) / ".env"
  else:
    env_path = Path(__file__).resolve().parent / ".env"

  if env_path.exists():
    load_dotenv(env_path)
    print("Environment variables loaded:")
    print(f"DATA_DIR: {bool(data_dir)}, CODE_DIR: {bool(code_dir)}")
  else:
    raise FileNotFoundError(f".env file not found at {env_path}")

  return {
      "data_dir": os.getenv("DATA_DIR"),
      "code_dir": os.getenv("CODE_DIR"),
  }

def process_datasets(data_path):
    dfs, docs_dict, datasets, failed = {}, {}, {}, []

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)

        if os.path.isfile(file_path) and file.endswith(".csv"):
            file_name = re.sub(r"(_clean|filtered_)?\.csv$", "", file)
            datasets[file_name] = file_path

    for name, path in datasets.items():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            failed.append(name)
            continue

        text_col = next((col for col in ['body', 'text'] if col in df.columns), None)

        if text_col is None:
            print(f"Skipping {name}. No 'body' or 'text' column.")
            failed.append(name)
            continue

        dfs[name] = df
        docs_dict[name] = df[df[text_col].notna()][text_col].tolist()

        print(f'Loaded {name}')

        print(f"\n{len(dfs)}/{len(datasets)} datasets loaded successfully.")
        if failed:
          print("Failed to load:", ", ".join(failed))

    return dfs, docs_dict, datasets, failed

def load_models():
    sentiment_model = "finiteautomata/bertweet-base-sentiment-analysis" #heavy twitter-leaning
    emotion_reddit = "SamLowe/roberta-base-go_emotions" #28 labels
    emotion_twitter = "boltuix/bert-emotion" #13 labels
    emotion_general = "cirimus/modernbert-base-go-emotions" #heavy reddit-leaning, 28 labels

    print("Loading models...\n")

    sentiment_analyzer = pipeline("text-classification", model=sentiment_model)

    emotion_analyzer_twitter = pipeline(
        "text-classification",
        model=emotion_twitter,
        top_k=None # check to see if this works, even with single labels
    )

    emotion_analyzer_reddit = pipeline(
        "text-classification",
        model=emotion_reddit,
        top_k=None
    )

    return {
        "sentiment": sentiment_analyzer,
        "emotion_twitter": emotion_analyzer_twitter,
        "emotion_reddit": emotion_analyzer_reddit,
    }

def choose_emotion_model(dataset_name, models):
    if "twitter" in dataset_name.lower():
        return models["emotion_twitter"]
    return models["emotion_reddit"]

def sentiment_analysis(df, analyzer, text_col=None, batch_size=128):
    if text_col is None:
        text_col = next((col for col in ['body', 'text'] if col in df.columns), None)
        if text_col is None:
            raise ValueError("No valid text column found")

    texts = df[text_col].tolist()
    label, confidence = [], []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        results = analyzer(batch, padding=True, truncation=True)

        for result in results:
            label.append(result['label'])
            confidence.append(result['score'])

    df['sentiment_label'] = label
    df['sentiment_proba'] = confidence
    return df

def emotion_analysis(df, analyzer, text_col=None, batch_size=128, multi=False):
    if text_col is None:
        text_col = next((col for col in ['body', 'text'] if col in df.columns), None)
        if text_col is None:
            raise ValueError("No valid text column found")

    texts = df[text_col].tolist()
    label, confidence, all_emotions = [], [], []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        results = analyzer(batch, padding=True, truncation=True, max_length=512)

        if multi:
            for emotion_list in results:
                top_k = sorted(emotion_list, key=lambda x: x['score'], reverse=True)[:3]
                label.append(top_k[0]['label'])
                confidence.append(top_k[0]['score'])
                all_emotions.append(top_k)
        else:
            for result in results:
                label.append(result['label'])
                confidence.append(result['score'])

    df['emotion_label'] = label
    df['emotion_proba'] = confidence
    if multi:
        df['all_emotions'] = all_emotions
        df['top3_emotions'] = df['all_emotions'].apply(
            lambda lst: ', '.join(f"{x['label']} ({x['score']:.2f})" for x in lst)
            )

    return df

def main():
  env = load_env()
  data_dir, code_dir = env["data_dir"], env["code_dir"]

  if not data_dir or not code_dir:
    raise EnvironmentError("DATA_DIR and CODE_DIR must be set in the .env file.")

  dfs, docs_dict, datasets, failed = process_datasets(data_dir)

  models = load_models()

  for name, df in dfs.items():
    print(f"\n{'=' * 50}\nAnalyzing {name}\n{'=' * 50}")

    df = sentiment_analysis(df, analyzer=models["sentiment"])

    emotion_model = choose_emotion_model(name, models)
    try:
      df = emotion_analysis(df, emotion_model, multi=True)
    except TypeError as e:
      print(f"Multi-emotion analysis failed for {name}: {e}")
      print("\nRetrying without computing multiple emotions")
      df = emotion_analysis(df, emotion_model, multi=False)
    except Exception as e:
      print(f"Error occurred during analysis of {name}: {e}")
      print(traceback.format_exc())
      continue

    dfs[name] = df

    try:
        df.to_csv(datasets[name], index=False)
        print(f"\nSaved {name}")
    except Exception as e:
        print(f"Failed to save {name}: {e}")

  return dfs, docs_dict, datasets

if __name__ == "__main__":
  dfs, docs_dict, datasets = main()