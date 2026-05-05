import os
from pathlib import Path

# DEFAULT DIRECTORIES (local dev)
DATA_DIR = "./data"
OUTPUT_DIR = "./code/visualizations"

# DETECT IF RUNNING INSIDE AZURE ML JOB
IN_AZUREML = (
    "AZUREML_RUN_ID" in os.environ
    or "AZUREML_EXPERIMENT_ID" in os.environ
    or "AZUREML_OUTPUT_DIR" in os.environ
)

# OVERRIDE PATHS WHEN INSIDE AZURE ML
if IN_AZUREML:
    DATA_DIR = "outputs/data"
    OUTPUT_DIR = "outputs/visualizations"

# ENSURE DIRECTORIES EXIST
Path(DATA_DIR).mkdir(
    parents=True, exist_ok=True
)  # can exist_ok work with mkdir()? according to docs, it should only be makedirs()
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  # investigate this
# https://docs.python.org/3/library/os.html#os.mkdir
