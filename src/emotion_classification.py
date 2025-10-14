import os
import re
from dotenv import load_dotenv
from pathlib import Path
import traceback

import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline
# !pip3 install emoji==0.6.0

import matplotlib.pyplot as plt
import seaborn as sns

def check_env(): #check working env
  try:
    import google.colab
    return True
  except ImportError:
    return False

def load_env():
  if check_env():
    base_path = "." # manually update this
    env_path = Path(base_path) / ".env"
  else:
    base_path = path(__file__).resolve().parent
    env_path = base_path / ".env"

  if env_path.exists():
    load_dotenv(env_path)
  else:
    print("No .env file found")

  return {
      "data_dir": os.getenv("DATA_DIR"),
      "code_dir": os.getenv("CODE_DIR"),
  }

def process_datasets(data_path):
    datasets = {}
    dfs = {}
    docs_dict = {}
    failed = []

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
        #docs_dict[name] = df[df[text_col]][text_col].tolist()

        print(f'Loaded {name}')

    return dfs, docs_dict, datasets, failed

"""## **Analysis**"""

directories = {
    "emotions": Path(code_dir) / "visualizations" / "emotions",
    "sentiments": Path(code_dir) / "visualizations" / "sentiments",
}

for path in directories.values():
  os.makedirs(path, exist_ok=True)

emotion_dir=directories.get("emotions")
sentiment_dir=directories.get("sentiments")

models ={
    "sentiment_model":"finiteautomata/bertweet-base-sentiment-analysis", #twitter
    "emotion_model1":"SamLowe/roberta-base-go_emotions", #reddit
    "emotion_model2":"cirimus/modernbert-base-go-emotions", #reddit
    "emotion_model3":"boltuix/bert-emotion" #twitter
    }

sentiment_model=models['sentiment_model']
emotion_model=models["emotion_model2"]

sentiment_analyzer = pipeline("text-classification", model=sentiment_model)

emotion_analyzer_single = pipeline("text-classification", model=emotion_model)
emotion_analyzer_multi = pipeline(
    "text-classification",
    model=emotion_model,
    return_all_scores=True
    )

def sentiment_analysis(df, text_col, batch_size=128):
    if text_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{text_col}' column")

    texts = df[text_col].tolist()
    label, confidence = [], []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        results = sentiment_analyzer(batch, truncation=True, padding=True)

        for result in results:
            label.append(result['label'])
            confidence.append(result['score'])

    df['sentiment_label'] = label
    df['sentiment_proba'] = confidence
    return df

def emotion_analysis(df, analyzer, text_col, batch_size=128):
    if text_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{text_col}' column")

    texts = df[text_col].tolist()
    label, confidence = [], []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        results = analyzer(batch, truncation=True, padding=True)

        if analyzer == emotion_analyzer_multi:
          for result in results:
            top_emotion = max(result['emotions'], key=lambda x: x['score'])
            results.append(result['emotions'])

            label.append(top_emotion['label'])
            confidence.append(top_emotion['score'])
            df['all_emotions'] = results
        else:
          for result in results:
            label.append(result['label'])
            confidence.append(result['score'])

    df['emotion_label'] = label
    df['emotion_proba'] = confidence
    return df

def main():
  env=load_env()
  data_dir = env["data_dir"]
  code_dir = env["code_dir"]

  if not data_dir or not code_dir:
    raise EnvironmentError("DATA_DIR and CODE_DIR must be set in the .env file.")

  print("Environment variables loaded:")
  print("DATA_DIR:", bool(data_dir))
  print("CODE_DIR:", bool(code_dir))

  if check_env():
    from google.colab import drive
    drive.mount("/content/drive")

  dfs, docs_dict, datasets, failed = process_datasets(data_dir)
  print(f"\n{len(dfs)}/{len(datasets)} datasets loaded successfully.")
  if failed:
    print("Failed to load:", ", ".join(failed))

  for name, df in dfs.items():
    print("\n" + "=" * 50)
    print(f"Analyzing {name}")
    print("=" * 50)
    text_col = 'body' if 'body' in df.columns else 'text'

    try:
      df = emotion_analysis(df, multi_emotion_analyzer, text_col=text_col)
    except TypeError as e:
      print(f"Multi-emotion failed for {name}: {e}")
      print("\nRetrying without computing multiple emotions\n")
      df = emotion_analysis(df, emotion_analyzer_single, text_col=text_col)
    except Exception as e:
      print(f"Error occurred during analysis of {name}: {e}")
      print(traceback.format_exc())
      continue

if __name__ == "__main__":
  main()

dfs['climate_twitter_sample']
#dfs['filtered_anticonsumption_comments']

dfs['filtered_anticonsumption_comments']

"""# Emotion Visualizations
*   Word clouds to represent overall emotional distribution

+ Time-series visualization to show how emotions change over time

"""



"""# **Sentiment Visualizations**
1. Distribution as Pie Charts
2. Sentiment Probability Histograms
3. Sentiment Probability Violin Plots
"""

sent_dist = Path(sentiment_dir) / "pie charts"
sent_vio = Path(sentiment_dir) / "violin distributions"
sent_his = Path(sentiment_dir) / "probability histograms"

os.makedirs(sent_dist, exist_ok=True)
os.makedirs(sent_vio, exist_ok=True)
os.makedirs(sent_his, exist_ok=True)

save_dir = sent_dist

def create_pie_plot(df,title):
    sentiment_counts = df['sentiment_label'].value_counts()
    sentiment_counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=['skyblue', 'lightcoral', 'lightgreen'],
        labels=sentiment_counts.index
    )
    plt.title(title)

for name, df in dfs.items(): # saving each seperately
    plt.figure(figsize=(6, 6))
    create_pie_plot(df, name)
    file_path = os.path.join(save_dir, f"{name}_sentiment_pie.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Saved: {file_path}")

# one big combined visual
n_datasets = len(dfs)
plt.figure(figsize=(6, 4 * n_datasets))

for idx, (name, df) in enumerate(dfs.items(), start=1):
    plt.subplot(n_datasets, 1, idx)
    create_pie_plot(df, name)

plt.tight_layout()
combined_path = os.path.join(save_dir, "all_sentiment_pies.png")
plt.savefig(combined_path, dpi=300)
plt.close()

print(f"Combined visualization saved to: {combined_path}")

"""## **Sentiment Probability Distribution by Sentiment Label**"""

save_dir=sent_vio

def create_violin_plot(df, title):
    sns.violinplot(
        data=df,
        x='sentiment_proba',
        y='sentiment_label',
        inner='box',
        palette='husl',
        hue='sentiment_label'
    )
    sns.despine(top=True, right=True, bottom=True, left=True)
    plt.title(title)

for name, df in dfs.items():
    plt.figure(figsize=(8, 6))
    create_violin_plot(df, name)
    file_path = os.path.join(save_dir, f"{name}_sentiment_violin.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Saved: {file_path}")

n_datasets = len(dfs)
plt.figure(figsize=(8, 6 * n_datasets))

for idx, (name, df) in enumerate(dfs.items(), start=1):
    plt.subplot(n_datasets, 1, idx)
    create_violin_plot(df, name)

plt.tight_layout()
combined_path = os.path.join(save_dir, "all_sentiment_violins.png")
plt.savefig(combined_path, dpi=300)
plt.close()

print(f"Combined visualization saved to: {combined_path}")

save_dir=sent_his

def create_histplot(df, title):
    sns.histplot(
        x='sentiment_proba',
        hue='sentiment_label',
        data=df,
        element='step'
    )
    plt.title(title)
    plt.xlabel('Sentiment Probability')
    plt.ylabel('Frequency')

for name, df in dfs.items():
    plt.figure(figsize=(12, 5))
    create_histplot(df, name)
    file_path = os.path.join(save_dir, f"{name}_sentiment_hist.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Saved: {file_path}")

n_datasets = len(dfs)
plt.figure(figsize=(12, 4 * n_datasets))

for idx, (name, df) in enumerate(dfs.items(), start=1):
    plt.subplot(n_datasets, 1, idx)
    create_histplot(df, name)

plt.tight_layout()
combined_path = os.path.join(save_dir, "all_sentiment_hists.png")
plt.savefig(combined_path, dpi=300)
plt.close()

print(f"Combined visualization saved to: {combined_path}")