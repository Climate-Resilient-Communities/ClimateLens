import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string, re, os
import pandas as pd

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

        print(f'loaded {name}')

        docs = list(df[text_col].values)

        dfs[name] = df
        docs_dict[name] = docs

    return dfs, docs_dict

dfs, docs_dict = loading_datasets(datasets)
print(f"{len(list(dfs.keys()))} Dataframes collected")

stop_words = set(stopwords.words("english"))

swear_variants = [
    'fuck', 'fucking', 'fucked', 'fuckin', 'fck', 'f*ck', 'f@ck',
    'shit', 'shitty', 'shitshow', 'bullshit', 'bs', 'sh*t',
    'ass', 'asshole', 'a**', 'arse',
    'bitch', 'b*tch',
    'damn', 'd*mn',
    'crap', 'dick', 'piss', 'prick', 'whore', 'slut', 'cunt', 'mf', 'motherfucker',
]

additional_stopwords = [
    'rt', 'tweet', 'repost', 'replied', 'comments', 'comment', 'upvote', 'downvote', 'subreddit',
    'thread', 'user', 'followers', 'post', 'share', 'like', 'reply', 'hashtag', 'hashtags','link',
    'bio', 'mention', 'tagged', 'followed', 'following', 'message', 'profile','climate', 'change',
    'global', 'warming', 'yes', 'great',
    'love', 'great', 'thank', 'you', 'good', 'like', 'go',
]

# Words to keep (negations, modals, interrogatives)
preserve_words =  {
    'not', 'no', 'nor', 'should', 'could', 'would', 'must', 'might', 'may',
    'don’t', 'do', 'does', 'did', 'why', 'what', 'how', 'if', 'that', 'this',
    'i', 'you', 'we', 'they', 'he', 'she', 'it'
}

# combine nltk stopwords with our swear variants and garbage content words
custom_stopwords = stop_words.union(swear_variants).union(additional_stopwords)

# Remove preserved words from the stopwords set
custom_stop_words = stop_words - preserve_words

print(custom_stopwords)

def remove_consecutive_repeats(tokens):
    if not tokens:
        return tokens
    cleaned = [tokens[0]]
    for i in range(1, len(tokens)):
        if tokens[i] != tokens[i-1]:
            cleaned.append(tokens[i])
    return cleaned

def highlight_issues(text):
    lowered = text.lower()
    repeated = re.findall(r'\b(\w+)\s+\1\b', lowered)
    slang = [word for word in swear_variants if word in lowered]
    return repeated, slang

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in custom_stopwords]
    tokens = remove_consecutive_repeats(tokens)
    return ' '.join(tokens)

def run_pipeline(datasets):
    dfs, docs_dict = loading_datasets(datasets)
    print(f"Dataframes collected:\n{list(dfs.keys())}")

    for name, df in dfs.items():
        print(f"\nProcessing dataset: {name}")

        # Preview issues
        text_col = "body" if "body" in df.columns else "text"
        samples = []
        for idx, row in df.head(10).iterrows():
            text = str(row[text_col])
            repeated, slang = highlight_issues(text)
            samples.append({
                "original_text": text,
                "repeated_words": repeated,
                "slang_terms": slang
            })
        peek_df = pd.DataFrame(samples)
        #print(peek_df)

        df['cleaned_text'] = df[text_col].astype(str).apply(preprocess_text)
        df.to_csv(datasets[name], index=False)
        print(f"{name} cleaning complete!")

if __name__ == "__main__":
    run_pipeline(datasets)