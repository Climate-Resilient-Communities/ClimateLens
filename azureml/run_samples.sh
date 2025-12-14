#!/bin/bash
set -e

echo "AzureML run_scripts.sh starting..."

echo "Current working directory:"
pwd
echo "Directory contents:"
ls -R .

echo "============================"
echo "Current working directory: $(pwd)"
echo "CODE_DIR: '${CODE_DIR}'"
echo "Directory contents:"
ls -al
echo "============================"

echo "Running cleaning/filtering scripts"

python code/reddit_data_filtering.py
python code/data_preprocessing.py

echo "Running topic modeling script"

python code/topic_modeling.py

echo "Running emotion classification scripts"

python code/emotion_classification.py
python code/emotion_visualizations.py

echo "All scripts successfully completed!"