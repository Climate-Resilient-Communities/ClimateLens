
#################################################################################
# GLOBALS
#################################################################################

PROJECT_NAME = climatelens
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = .venv/bin/python
PIP = $(PYTHON_INTERPRETER) -m pip

# Use the Windows virtual environment paths when running under Windows.
ifeq ($(OS),Windows_NT)
PYTHON_INTERPRETER = .venv/Scripts/python.exe
endif

.DEFAULT_GOAL := help

#################################################################################
# ENVIRONMENT AND INSTALLATION
#################################################################################

.PHONY: create_environment
create_environment:
	python -m venv .venv
	@echo "Virtual environment created."
	@echo "Activate with: source .venv/bin/activate (Mac/Linux)"
	@echo "Activate with: .venv\\Scripts\\activate (Windows)"

.PHONY: install
install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

.PHONY: install-dev
install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: install-ml
install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[ml,dev]"

.PHONY: install-lda
install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[lda,dev]"

#################################################################################
# CODE QUALITY
#################################################################################

.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m ruff check src tests
	$(PYTHON_INTERPRETER) -m ruff format --check src tests

.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m ruff check --fix src tests
	$(PYTHON_INTERPRETER) -m ruff format src tests

#################################################################################
# TESTING AND PIPELINE
#################################################################################

.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests/ -v

.PHONY: pipeline
pipeline:
	$(PYTHON_INTERPRETER) src/data_preprocessing.py
	$(PYTHON_INTERPRETER) src/topic_modeling.py
	$(PYTHON_INTERPRETER) src/emotion_classification.py
	$(PYTHON_INTERPRETER) src/emotion_visualizations.py

#################################################################################
# CLEANUP
#################################################################################

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# HELP
#################################################################################

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make create_environment  Create the virtual environment"
	@echo "  make install             Install the package"
	@echo "  make install-dev         Install development dependencies"
	@echo "  make lint                Run lint and formatting checks"
	@echo "  make format              Automatically format code"
	@echo "  make test                Run tests"
	@echo "  make pipeline            Run the full processing pipeline"
	@echo "  make clean               Remove Python cache files"
