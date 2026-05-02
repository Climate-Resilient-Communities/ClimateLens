# File Tree

```
climatelens
├── evaluation
│   └── cohere_integration.py - Integrates Cohere API for analysis.
├── models
│   ├── artifacts - Stores trained models
│   ├── LDA
│       ├── lda_model.py - Contains the LDA model implementation.
│       ├── lda.md - Documentation for the LDA model.
│       ├── preprocessing.py - Handles preprocessing of data for LDA modeling.
│       ├── run_lda.py - Script to train and run the LDA model.
│       └── visualization.py - Visualizes the topics produced by the LDA model.
├── nlp_pipeline
│   ├── dynamic_topic_modeling.py - Implements dynamic topic modeling.
│   ├── embeddings.py - Handles embedding generation.
│   ├── emotion_classification.py - Classifies emotions in text.
│   ├── postprocessing.py - Post-processes model outputs.
│   └── topic_modeling.py - Performs topic modeling.
├── preprocessing
│   ├── init.py
│   ├── data_preprocessing.py - Preprocesses raw data for analysis.
│   ├── reddit_tools.py
│   └── twitter_tools.py
├── utils
│   ├── create_directories.py - Creates necessary directories for file management.
│   ├── io_helpers.py - Helper functions for input/output operations.
│   ├── logging_config.py - Configures logging settings.
│   ├── process_datasets.py - Processes datasets for the project.
│   └── runtime.py - Manages runtime environment and processes.
├── visualizations
│   └── emotion_visualization.py - Visualizes emotion analysis results.
├── config
│   ├── datasets.yaml - Dataset registry (text/timestamp columns, profiles)
│   └── dataset_registry.py - Registers and manages datasets used in the project.
```
