# **ClimateLens Contributions Guide**

Thank you for contributing to the ClimateLens Project\! These guidelines ensure our code remains impactful, maintainable, and aligned with the overarching goals of the project. Please review them carefully before making contributions.

ClimateLens aims to build reliable, transparent tools that help communities and decision-makers better understand and respond to climate anxiety. Our work prioritizes practical impact, technical rigor, and reproducibility. Every contribution should move us closer to a stable, usable Minimum Viable Product (MVP) that serves real-world needs.

We prioritize clarity and maintainability over clever or complex code. All contributions should align with our current MVP. Contributors are expected to update documentation when making significant changes.

| Version | 2.01 |
| :- | :- |
| **Last Edited** | 3/30/2026 |
| **By** | Karim El-Sharkawy |
| **Technical Lead** | Karim El-Sharkawy |

## **Getting Started**

### **1. Fork & Clone the Repository**

If you are an external contributor:

```bash
git clone https://github.com/[org]/[repo].git
cd [repo]
```

If you are a core contributor, you may work directly from the main repository.



### **2. Set Up the Environment**

Follow the instructions in the `README.md` to:

* Install dependencies
* Configure your environment
* Run the pipeline locally

> If anything is unclear or missing, please open an issue! Improving onboarding is part of the project.

### **3. Development Environments**

You may use:

* Local development (recommended)
* Google Colab (optional, for experimentation)

**Important:**

* All final code must run locally and be reproducible outside of Colab
* Do not rely on Colab-specific paths or hidden state
* Do **not** include sensitive data or credentials in code, commits, or documentation
  * Use environment variables for secrets

## **Git & Pull Request Workflow**

To maintain stability and clarity, all contributions must follow this workflow.

### **Branch Structure**

We use the following branches:

* **`main`** → Production-ready, stable code
* **`dev`** → Integration branch for active development
* **Domain branches:**
  * `climate-anxiety`
  * `youth-detector`
  * `productization`
* **`experiment/*`** → Temporary experimentation

**Rules:**

* No direct commits to `main` or `dev`
* All changes must go through Pull Requests

### **Issue-Based Development**

Before starting work:
* Check existing issues
* If none exist, create a new issue describing the problem or feature

Your issue should include:
* A clear description
* Scope of work
* Expected outcome

All branches and PRs must reference an issue.

### **Development Workflow**

#### **1. Sync with `dev`**

```bash
git checkout dev
git pull origin dev
```

#### **2. Create a Branch**

Examples:

**Feature work:**

```bash
git checkout -b feature/issue-12-short-description
```

**Domain-specific work:**

```bash
git checkout -b climate-anxiety/issue-34-model-tuning
```

**Experiments:**

```bash
git checkout -b experiment/bert-baseline
```

#### **3. Commit Changes**

* Keep commits focused
* Only include relevant files

Example:

```bash
git commit -m "Issue #12: Add data validation pipeline"
```

#### **4. Open a Pull Request**

* Push your branch
* Open a PR into **`dev`**
* Link the associated issue
* Clearly describe:

  * What you changed
  * Why it matters
  * How it was tested

### **Where Should Work Happen?**

* **`experiment/*`**

  * Early-stage or uncertain ideas
  * May be unstable
  * Delete after completion unless promoted

* **Domain branches**

  * Core system/model development
  * Iterative improvements before integration

* **`productization`**

  * Documentation, reproducibility, usability, dashboards

* **`dev`**

  * Full system integration
  * Should run end-to-end

* **`main`**

  * Stable, production-ready releases only

### **Code Review & Merge**

* All PRs must be reviewed before merging
* Address all feedback before approval
* Merge into `dev` only
* Delete branches after merge

**Comments should:**

* Explain *why*, not *what*
* Highlight assumptions or tradeoffs

### **Releasing to `main`**

* Managed by the technical lead
* Happens at MVP milestones
* `main` must always be:
  * Stable
  * Reproducible
  * Fully functional

### **Core Principles**

* No direct pushes to `main` or `dev`
* Keep branches small and focused
* Prefer frequent, incremental contributions
* Use `experiment/*` for uncertainty

## **Testing & Code Review**

* Ensure code runs end-to-end
* Add validation steps or tests when applicable
* Prioritize readability and maintainability

## **AI Guidelines**

* AI tools may be used for learning and debugging. It is desirable to not use any AI code at all if possible.
* All AI-generated code must be:
  * Reviewed
  * Understood
  * Refactored if needed

You are responsible for every line you commit.

## **Communication & Collaboration**

* For major changes, open an issue first to discuss
* Be respectful and constructive in reviews
* If you're unsure where something belongs, ask in an issue