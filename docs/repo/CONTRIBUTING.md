# **ClimateLens Contributing Guide**

Thank you for contributing to the **ClimateLens Project**! Your contributions help us build reliable tools to understand and respond to climate anxiety.

Please review these guidelines before making contributions to ensure clarity, maintainability, and alignment with our project goals.

## **Our Mission**

ClimateLens builds transparent, reproducible tools to help communities and decision-makers better understand climate anxiety. Every contribution should move us closer to a **stable, usable MVP** that serves real-world needs.

We prioritize:

* Practical impact over complexity
* Clear, maintainable code over clever tricks
* Updated documentation for any significant changes

## **Getting Started**

Follow these steps to begin contributing:

### **1. Fork & Clone the Repository**

```bash
git clone https://github.com/[org]/[repo].git
cd [repo]
```

> Core contributors may work directly from the main repository.

### **2. Set Up the Environment**

Follow the `README.md` to:

* Install dependencies
* Configure your environment
* Run the pipeline locally

> If anything is unclear, open an issue! Improving onboarding is part of the project.

### **3. Development Environments**

You may use:

* **Local development** (recommended)
* **Google Colab** (optional, for experimentation)

**Important:**

* All final code must run locally and be reproducible outside of Colab
* Avoid Colab-specific paths or hidden state
* Do **not** include sensitive data or credentials in code, commits, or documentation
* Use environment variables for secrets

## **Making Contributions**

1. **Check existing issues** before starting work.
2. If none exist, create a **new issue** describing the problem or feature.
3. Reference your issue in any branch or Pull Request (PR).

**Key reminders:**

* Keep commits focused and relevant
* Clearly describe what changed, why it matters, and how it was tested
* Respect code review feedback

## **Need the Full Workflow?**

For **branching, detailed workflow, PR rules, testing, and release guidelines**, please see our dedicated [DEVELOPMENT.md](DEVELOPMENT.md).

## **Communication & Collaboration**

* Open an issue to discuss major changes before starting
* Be respectful and constructive in reviews
* Ask questions if unsure where something belongs