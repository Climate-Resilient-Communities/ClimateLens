# 🌍 ClimateLens

Climate change is driving rising anxiety, yet we lack clear insight into how it appears in everyday language and have few tools for early detection. By analyzing linguistic patterns with NLP and modern machine learning methods, ClimateLens aims to identify climate anxiety signals in text, reveal how they manifest among youth, and provide a reusable, scalable detection pipeline with an interactive platform for exploring results.

The goal is to enable earlier awareness, support research into climate-related emotional expression, and help organizations transform climate-related concern into constructive engagement and resilience.

The production application is deployed on HuggingFace Spaces using Streamlit, where users can explore the models and visualizations interactively.

- 🌐 **Launch Web App:** https://huggingface.co/spaces/crc-sprout/ClimateLens  
- 📖 **Learn More:** https://crc.place/climatelens/


# ✨ Features
+ Tools for filtering and cleaning social media datasets relevant to climate discourse.
+ Machine learning pipelines for topic modeling, clustering, and classification of climate-related emotions and anxiety signals.
+ Interactive visualizations for exploring topics, emotions, and trends within climate-related text datasets.
+ A Streamlit interface hosted on HuggingFace Spaces for applying models and visualizing results.


# ⚙️ Setup

## 1. Clone the repository

```bash
git clone https://github.com/Climate-Resilient-Communities/ClimateLens.git
cd ClimateLens
````

## 2. Create a Python environment

ClimateLens requires **Python 3.10**.

Using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```


## 3. Install dependencies

Install the project in **editable mode**:

```bash
make install
```

This command will:

* upgrade `pip`
* install the ClimateLens package locally
* install all required dependencies

If you prefer manual installation:

```bash
pip install -e .
```

Editable installs allow you to modify the source code without reinstalling the package.


## 4. Verify installation

```bash
python -c "import climate_lens; print('ClimateLens installed successfully')"
```

# 🔐 Required Environment Variables

Create a `.env` file in the project root.

Example:

```
# Cohere API access
COHERE_API_KEY=your_cohere_key

# Directory configuration
DATA_DIR=your_data_directory_here
CODE_DIR=your_code_directory_here
```

Some scripts (such as `topic_modeling.py` and `emotion_classification.py`) require these environment variables to access external APIs and local data directories.

# 🛠 Developer Commands

The repository includes a **Makefile** to simplify development tasks.

Common commands:

```bash
make install      # install project dependencies
make lint         # check formatting and lint code
make format       # auto-format Python files
make clean        # remove compiled Python files
```


# 📂 Project Structure

```
ClimateLens/
│
├── src/
│   └── climate_lens/               # Core Python package
│        ├── preprocessing/         # Data cleaning pipelines
│        ├── models/                # ML models and classifiers
│        ├── topic_modeling/        # Topic modeling implementations
│        ├── evaluation/            # Metrics and evaluation logic
│        └── utils/                 # Shared helper utilities
│
├── notebooks/                      # Exploratory analysis notebooks
├── scripts/                        # Standalone scripts for running pipelines
│
├── data/                           # Example datasets
│   ├── climate_twitter_sample.csv
│   └── filtered_anticonsumption_comments.csv
│
├── azureml/                        # AzureML job configuration
│   ├── AML_job.py                  # AzureML job definition
│   ├── environment.yml             # AzureML compute environment
│   ├── run_scripts.sh              # Pipeline execution script
│   └── test_run_scripts.sh         # Script for validating pipeline execution
│
├── tests/                          # Unit tests
│
├── Makefile                        # Development automation commands
├── pyproject.toml                  # Python package configuration
├── requirements.txt                # Dependency list
├── README.md                       # Project documentation
├── LICENSE
└── .gitignore
```


# ☁️ Azure Machine Learning Execution

ClimateLens supports cloud execution through Azure Machine Learning (AzureML) for running the pipeline on remote VMs without instead of local machines.

### How it works

* AzureML mounts the code and data stored in your workspace
* A job runs the pipeline scripts sequentially
* No local uploads or `.env` access are required
* Logs stream back to your terminal during execution

The pipeline execution order is defined in:

```
run_scripts.sh
```

The job submission and configuration are handled by:

```
AML_job.py
```

These scripts together define and launch the AzureML job.


# 🤝 Contributing

This repository is currently maintained internally by Climate Resilient Communities.

Plans are underway to make ClimateLens fully open-source and open to community contributions in the future.


# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.