#!/bin/bash
set -e
echo "Running ..." # run each python file one after another within the same job where this is called
python .py
echo "Running ..."
python .py
echo "Running ..."
python .py

echo "All scripts ran successfully."