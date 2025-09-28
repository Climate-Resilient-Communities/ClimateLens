# 🌍 ClimateLens

ClimateLens is an open-source research project dedicated to understanding and addressing climate anxiety among young people. Our goal is to identify how climate anxiety manifests in youth and to develop a reproducible, reusable, and interpretable detection model for early intervention. By analyzing language patterns and expressions linked to climate anxiety, the project uncovers common themes and concerns, offering insights into how youth experience and cope with this growing issue.

- [🌐 Launch Webapp](https://huggingface.co/spaces/crc-sprout/ClimateLens)  
- [📖 Learn More](https://crc.place/climatelens/)

## Problem Statement

Climate change is a significant source of anxiety. Despite its increasing prevalence, there is limited understanding of how climate anxiety manifests and few tools exist for early detection and intervention. Without timely support, these anxieties can escalate, worsening mental health outcomes and reducing overall well-being.

We hypothesize that by analyzing language for recurring themes and expressions, and developing an NLP/LLM-based model to detect climate anxiety, we can deliver actionable insights that guide timely interventions.

This project will:
+ Reveal how climate anxiety manifests in youth language.
+ Develop a practical, reusable detection tool.
+ Ensure scalability and openness through public datasets.
+ Provide an interactive platform for applying and visualizing results.

## Impact
By enabling early detection, ClimateLens empowers support networks and mental health professionals to act proactively. Our ultimate goals are to:
+ Strengthen youth resilience.
+ Foster a sense of agency.
+ Transform climate-related fears into constructive engagement.

## ✨ Features
- **Data Collection** – tools for gathering and cleaning social media datasets.  
- **NLP Models** – topic modeling and classification for detecting climate-related emotions.  
- **Visualization** – interactive graphics and dashboards.  
- **WebApp** – HuggingFace Space.  

## 🔐 Required Environment Variables
```
# Cohere
COHERE_API_KEY=your_cohere_key
```

## 📂 Project Structure
```
├── data/                 # sample datasets
│
├── src/
│   ├── LDA/              # LDA model (baseline only)
│   ├── models/           # pipelines and ML models
│   ├── utils/            # shared utilities
│   └── notebooks/        # exploratory/archived notebooks
│
├── LICENSE
├── Makefile
├── requirements.txt
└── README.md
```

## 🌐 WebApp
The production app is deployed on HuggingFace Spaces using Streamlit. All visualizations and explanations are present in the app.

## 🤝 Contributing
This is an organization-only project.

## License
This project is licensed under the MIT License – see the LICENSE file for details.