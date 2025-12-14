import os
import re
import traceback
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

def load_environment() -> Dict[str, Path]:
    """
    Load environment variables.
    - Local: read from .env
    - AzureML: force outputs/ paths
    """

    # Detect AzureML
    in_azureml = (
        "AZUREML_RUN_ID" in os.environ
        or "AZUREML_EXPERIMENT_ID" in os.environ
        or "AZUREML_OUTPUT_DIR" in os.environ
    )

    if in_azureml:
        print("Running inside AzureML — using outputs/ directories")
        input_data_dir = Path("./data")               # READ ONLY
        output_data_dir = Path("./outputs/data")      # WRITE
        output_vis_dir  = Path("./outputs/visualizations")
    else:
        # Local dev: search for .env upward
        search_dir = Path(__file__).resolve().parent
        for parent in [search_dir] + list(search_dir.parents):
            env_path = parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded .env from: {env_path}")
                break
        else:
            raise FileNotFoundError("No .env found")

        input_data_dir  = Path(os.environ["DATA_DIR"])
        output_data_dir = Path(os.environ["OUTPUT_DATA_DIR"])
        output_vis_dir  = Path(os.environ["OUTPUT_VIS_DIR"])

        if not data_dir or not output_dir:
            raise KeyError("DATA_DIR or OUTPUT_DIR missing in .env")

    # Ensure directories exist
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    return {
        "input_data_dir": input_data_dir,
        "output_data_dir": output_data_dir,
        "output_vis_dir": output_vis_dir,
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

    print("Loading sentiment model...", flush=True)
    sentiment_analyzer = pipeline("text-classification", model=sentiment_model)


    print("Loading twitter emotion model...", flush=True)    
    emotion_analyzer_twitter = pipeline(
        "text-classification",
        model=emotion_twitter,
        top_k=None # check to see if this works, even with single labels
    )

    print("Loading reddit emotion model...", flush=True)
    emotion_analyzer_reddit = pipeline(
        "text-classification",
        model=emotion_reddit,
        top_k=None
    )

    print("All pipelines loaded")

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
    # ======================================
    # LOAD ENVIRONMENT (THIS IS THE KEY)
    # ======================================
    env = load_environment()
    input_data_dir  = env["input_data_dir"]     # data (input)
    output_data_dir = env["output_data_dir"]    # outputs/data (Azure)
    output_vis_dir  = env["output_vis_dir"]     # outputs/visualizations

    print("INPUT_DATA_DIR =", input_data_dir)
    print("OUTPUT_DATA_DIR =", output_data_dir)
    print("OUTPUT_VIS_DIR =", output_vis_dir)

    # ======================================
    # LOAD DATA
    # ======================================
    dfs, docs_dict, datasets, failed = process_datasets(input_data_dir)

    if not dfs:
        raise RuntimeError("No datasets loaded — aborting")

    # ======================================
    # LOAD MODELS
    # ======================================
    models = load_models()

    # ======================================
    # EMOTION ANALYSIS + SAVE DATA
    # ======================================
    for name, df in dfs.items():
        print(f"\n{'=' * 50}\nAnalyzing {name}\n{'=' * 50}")

        emotion_model = choose_emotion_model(name, models)

        try:
            df = emotion_analysis(df, emotion_model, multi=True)
        except TypeError:
            print(f"Multi-emotion failed for {name}, retrying single-label...")
            df = emotion_analysis(df, emotion_model, multi=False)
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            print(traceback.format_exc())
            continue

        dfs[name] = df

        out_path = output_data_dir / f"{name}_with_emotions.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved dataset: {out_path}")

    # ======================================
    # VISUALIZATIONS (outputs/visualizations)
    # ======================================
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    for name, df in dfs.items():
        try:
            fig = df["emotion_label"].value_counts().plot(kind="bar").get_figure()
            out_file = output_vis_dir / f"{name}_emotion_counts.png"
            fig.savefig(out_file, dpi=200, bbox_inches="tight")
            fig.clf()
            print(f"Saved visualization: {out_file}")
        except Exception as e:
            print(f"Visualization failed for {name}: {e}")

    return dfs, docs_dict, datasets

if __name__ == "__main__":
    dfs, docs_dict, datasets = main()