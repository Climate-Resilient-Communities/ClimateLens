#!/bin/bash
set -e

CODE_DIR=$(pwd)
echo "Code directory: $CODE_DIR"

echo "Running scripts..." # run each python file one after another within the same job where this is called
python $CODE_DIR/script1.py
python $CODE_DIR/script1.py
python $CODE_DIR/script1.py
python $CODE_DIR/script1.py
python $CODE_DIR/script1.py
python $CODE_DIR/script1.py

echo "All scripts ran successfully."