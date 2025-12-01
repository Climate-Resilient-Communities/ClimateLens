#!/bin/bash
set -e

echo "-------------------------------------"
echo " AzureML run_scripts.sh starting... "
echo "-------------------------------------"

echo "Current working directory:"
pwd
echo "Directory contents:"
ls -R .

echo "-------------------------------------"
echo " Running cleaning/filtering scripts "
echo "-------------------------------------"

python code/reddit_data_filtering.py
python code/twitter_data_cleaner.py
python code/data_preprocessing.py

echo "-------------------------------------"
echo " Running topic modeling scripts "
echo "-------------------------------------"

python code/topic_modeling.py
python code/dynamic_topic_modeling.py

echo "-------------------------------------"
echo " Running emotion classification scripts "
echo "-------------------------------------"

python code/emotion_classification.py
python code/emotion_visualizations.py

echo " All scripts successfully completed! "