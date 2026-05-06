import pandas as pd

twitter = pd.read_csv('../data/climate_twitter_sample.csv')
print("TWITTER COLUMNS:", twitter.columns.tolist())
print(twitter.head(2))

print("\n" + "="*50 + "\n")

reddit = pd.read_csv('../data/filtered_anticonsumption_comments.csv')
print("REDDIT COLUMNS:", reddit.columns.tolist())
print(reddit.head(2))