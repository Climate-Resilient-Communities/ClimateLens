# 🌳 Branching Strategy

This document outlines the branching strategy used in this repository. The goal is to keep development organized, reproducible, and easy to collaborate on while supporting both experimentation and production readiness.

## 🔒 `main`

Purpose:  
The `main` branch contains production-ready, stable code only.

Rules:
- No direct commits
- All changes must come through pull requests (PRs)
- Code in this branch should always run without errors

Use case:
- Final versions of models
- Stable pipeline execution
- Clean, reproducible results

## 🧪 `dev`

Purpose:  
The `dev` branch is the integration branch where all features are combined and tested together.

Rules:
- Feature branches merge into `dev`
- `dev` is tested before merging into `main`
- May contain minor instability, but should generally work end-to-end
- Document important decisions when merging into `dev` or `main`

Use case:
- Combining preprocessing + models + evaluation
- Testing full pipeline behavior
- Preparing for release into `main`

## 🌡️ `climate-anxiety`

Purpose:  
Development and improvement of the climate anxiety classification system.

Scope includes:
- Model training and refinement
- Error analysis and evaluation
- Feature engineering
- Experimentation with different approaches
- Inference logic

Why this branch exists:
- Keeps the core classification task isolated
- Supports iterative improvement and analysis without affecting other components

## 🚀 `youth-detector`

Purpose:  
Development of the youth-written content detection model.

Scope includes:
- Model architecture and training
- Feature engineering (youth-specific signals)
- Inference logic
- Supporting scripts and utilities

Why this branch exists:
- This is a core component of the system
- Requires independent iteration and experimentation before integration

## 📦 `productization`

Purpose:  
Focuses on making the project usable, reproducible, and presentable.

Scope includes:
- Documentation (README, MkDocs, usage guides)
- Instructions for running the pipeline
- Project structure improvements
- Reproducibility (configs, scripts, environment setup)
- Building dashboards

Why this branch exists:
- Separates engineering work from presentation and usability
- Helps transition the project from research → production-ready

## 🧪 `experiment/*`

Purpose:  
Used for temporary experimentation and research.

Examples:
- `experiment/bert-baseline`
- `experiment/youth-features-v2`
- `experiment/class-imbalance-fix`

Rules:
- Can branch off from `dev` or a feature branch
- Not required to be stable
- Should be deleted after completion (or merged if useful)

Use case:
- Testing new models or ideas
- Running ablation studies
- Trying risky changes without affecting main development

## 🎯 Guiding Principles

- Keep branches focused and single-purpose
- Prefer small, frequent merges over large ones
- Use `experiment/*` for anything uncertain or exploratory
- Keep `main` always clean and reliable