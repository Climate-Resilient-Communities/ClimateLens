import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from transformers import pipeline

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

paths = {
    "home": "...", # root path, where this file is
    "models": "...", #path for models
    "insights": "...", # static
    "interactives": "...",
}

os.makedirs(paths.get("models"), exist_ok=True)
os.makedirs(paths.get("insights"), exist_ok=True)
os.makedirs(paths.get("interactives"), exist_ok=True)

def load_datasets(data_path, prefix, datasets):
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        
        if os.path.isfile(file_path):
            file_name = file.replace(prefix, "").replace(".csv", "")
            datasets[file_name] = file_path

datasets = {}
load_datasets("...", "filtered_", datasets)
load_datasets("...", "clean_", datasets)

print("Collected Datasets:")
for key, value in datasets.items():
    print(f'{key}: {value}\n')

def loading_datasets(datasets):
    dfs = {}
    docs_dict = {}

    for name, path in datasets.items():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        if "body" in df.columns:
            text_col = "body"
        elif "text" in df.columns:
            text_col = "text"
        else:
            print(f"Skipping {name}. No 'body' or 'text' column.")
            continue

        print(f'Loaded {name}')

        docs = list(df[df[text_col].notna()][text_col].values)

        dfs[name] = df
        docs_dict[name] = docs

    return dfs, docs_dict

dfs, docs_dict = loading_datasets(datasets) # datafames aren't standalone variables
print(f"{len(list(dfs.keys()))} Dataframes collected")

"""## **Sentiment & Emotion Analysis**"""

"""
- ***j-hartmann/emotion-english-distilroberta-base*** - A model fine-tuned for emotion recognition in English text.
- ***bhadresh-savani/bert-base-go-emotion*** - Another popular model trained on the GoEmotions dataset from Google, which includes 27 emotion labels.
"""

model_name = 'finiteautomata/bertweet-base-sentiment-analysis'
sentiment_analyzer = pipeline("text-classification", model=model_name)

model_name = 'SamLowe/roberta-base-go_emotions'
emotion_analyzer = pipeline("text-classification", model=model_name)

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

def emotion_analysis(df, text_col, batch_size=128):
    if text_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{text_col}' column")

    texts = df[text_col].tolist()
    label, confidence = [], []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        results = emotion_analyzer(batch, truncation=True, padding=True)

        for result in results:
            label.append(result['label'])
            confidence.append(result['score'])

    df['emotion_label'] = label
    df['emotion_proba'] = confidence
    return df

for name, df in dfs.items():
    if 'emotion_proba' in df.columns:
        print(f'{name} already computed. Passing to next file\n')
        continue

    print(f'Computing {name}')
    text_col = 'body' if 'body' in df.columns else 'text'

    if 'emotion' not in df.columns:
        df['emotion'] = None
    if 'sentiment' not in df.columns:
        df['sentiment'] = None

    non_null_idx = df[df[text_col].notna()].index

    subset_df = df.loc[non_null_idx]
    subset_df = sentiment_analysis(subset_df, text_col=text_col)
    subset_df = emotion_analysis(subset_df, text_col=text_col)

    df.loc[non_null_idx, 'sentiment_label'] = subset_df['sentiment_label']
    df.loc[non_null_idx, 'sentiment_proba'] = subset_df['sentiment_proba']
    df.loc[non_null_idx, 'emotion_label'] = subset_df['emotion_label']
    df.loc[non_null_idx, 'emotion_proba'] = subset_df['emotion_proba']

    df.to_csv(datasets[name], index=False)

"""# **Sentiment Distribution in Generated Texts**
- Neutral sentiment texts are more frequent than negative sentiment texts.
- Negative sentiment texts are more frequent than positive sentiment texts.
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sent_dist = os.path.join(paths.get("insights"), "sentiment distribution/")
os.makedirs(sent_dist, exist_ok=True)

sent_vio = os.path.join(paths.get("insights"), "sentiment probability violins/")
os.makedirs(sent_vio, exist_ok=True)

sent_his = os.path.join(paths.get("insights"), "sentiment probability histograms/")
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

"""# **Sentiment Probability Distribution by Sentiment Label**
- Positive sentiment texts are labeled with greater certainty than negative sentiment texts.
- Negative sentiment texts are labeled with greater certainty than neutral sentiment texts.
"""

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