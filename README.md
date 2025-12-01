# 🌍 ClimateLens

Climate change is driving rising anxiety, yet we lack clear insight into how it appears in everyday language and have few tools for early detection. By analyzing linguistic patterns with NLP/LLM methods, ClimateLens aims to identify climate anxiety early, reveal how it manifests among youth, and provide a reusable, scalable detection model with an interactive platform for applying and visualizing results. The goal is to enable timely support, strengthen resilience, and turn climate-related fears into constructive engagement.

The production app is deployed on HuggingFace Spaces using Streamlit. All visualizations and explanations are present in the app.

- [🌐 Launch Webapp](https://huggingface.co/spaces/crc-sprout/ClimateLens)  
- [📖 Learn More](https://crc.place/climatelens/)

## ✨ Features
- **Data Collection** – tools for gathering and cleaning social media datasets.
- **NLP Models** – topic modeling and classification for detecting climate-related emotions.
- **Visualization** – interactive graphics and dashboards.
- **WebApp** – HuggingFace Space using Streamlit.

## 🔐 Required Environment Variables
```
# Cohere
COHERE_API_KEY=your_cohere_key

# Directories
DATA_DIR=your_data_directory_here
CODE_DIR=your_code_directory_here
```

Moreover, `topic_modeling.py` and `emotion_classification.py` both also require a manual entry for the .env file.

## 📂 Project Structure
```
ClimateLens/
├── azureml/                         # Azure Machine Learning job + environment setup
│   ├── AML_job.py                   # Defines AML job configuration and execution
│   ├── environment.yml              # Conda environment used for AML compute
│   ├── run_scripts.sh               # Shell script for running AML jobs end-to-end
│   └── test_run_scripts.sh          # Test script to validate AML job execution
│
├── data/                            # Sample input datasets
│   ├── climate_twitter_sample.csv   # Example climate-related Twitter posts
│   ├── filtered_anticonsumption_comments.csv  # Cleaned Reddit/Twitter anti-consumption data
│   └── README.md                     # Notes describing sample data contents/format
│
├── src/
│   ├── LDA/                          # Baseline LDA topic modeling implementation
│   │   └── ...                       # (LDA model scripts, topic extraction helpers, etc.)
│   ├── data_preprocessing.py         # Cleans raw social media text, normalizes fields, removes noise
│   ├── dynamic_topic_modeling.py     # Implements dynamic/temporal topic modeling (e.g., DTM/BERT-based)
│   ├── emotion_classification.py     # Emotion classifier pipeline (e.g., emotion embeddings + model)
│   ├── emotion_visualizations.ipynb  # Notebook for plotting emotion trends and visual insights
│   ├── reddit_data_filtering.py      # Filtering + preprocessing logic specialized for Reddit datasets
│   ├── topic_modeling.py             # Main topic modeling pipeline (BERTopic, LDA, clustering, etc.)
│   └── twitter_data_cleaner.py       # Specialized cleaning for Twitter text (URLs, mentions, tokens)
│   └── README.md                     # Explanation of source code structure & how to run modules
│
├── .gitignore
├── LICENSE
├── Makefile                         # Automation commands (e.g., setup, run, clean)
├── pyproject.toml                   # Build system + project metadata (modern Python packaging)
├── README.md                        # Main project documentation
├── requirements.txt                 # Python dependencies (runtime)
└── setup.cfg                        # Linting, formatting, and packaging configuration
```

# ⚙️ Azure ML Execution

ClimateLens supports cloud execution using Azure Machine Learning (AzureML).
All code and data should already live inside your AzureML Workspace, the jobs simply run the pipeline on a compute cluster without needing a web connection (AzureML compute instances are VMs, but JupyterNotebook requires a job to run without the web connection). Note that you must keep `AML_job.py` in the root directory outside of the azureml folder for everything to work as is.

### **How it works**

* AzureML mounts your existing workspace code and data
* A job runs your scripts in sequence using `run_all.sh`
* No local uploads or `.env` access are required
* Logs stream back to your terminal

```run_scripts.sh``` defines the order of your pipeline steps and ```AML_job.py``` submits the job to AzureML.

## 🤝 Contributing
This is an organization-only project for now, but efforts are underway to make this fully open-source.

## License
This project is licensed under the MIT License. See the LICENSE file for details.