#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = climatelens
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

## Create virtual environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> Virtual environment created."
	@echo ">>> Activate with:"
	@echo ">>> Windows: .venv\\Scripts\\activate"
	@echo ">>> Mac/Linux: source .venv/bin/activate"

## Install package
.PHONY: install
install:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install package + dev tools
.PHONY: install-dev
install-dev:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -e .[dev]

## Run linting
.PHONY: lint
lint:
	ruff check .

## Format code
.PHONY: format
format:
	ruff check --fix .
	black .

## Run tests
.PHONY: test
test:
	pytest

## Remove Python cache files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (lint + format check)
.PHONY: lint
lint:
	ruff check src tests
	ruff format --check src tests

## Auto-format source code with ruff
.PHONY: format
format:
	ruff check --fix src tests
	ruff format src tests

## Run the test suite
.PHONY: test
test:
	pytest tests/ -v

## Run the full pipeline locally (preprocessing -> topic modeling -> emotion)
.PHONY: pipeline
pipeline:
	$(PYTHON_INTERPRETER) src/data_preprocessing.py
	$(PYTHON_INTERPRETER) src/topic_modeling.py
	$(PYTHON_INTERPRETER) src/emotion_classification.py
	$(PYTHON_INTERPRETER) src/emotion_visualizations.py

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Project Setup Guidelines
# The setup guide for cloning the repo, setting up the environment, and installing dependencies is detailed in the `README.md`.
# Please refer to the `README.md` file for instructions on setting up the project locally.


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)