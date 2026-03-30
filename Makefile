PROJECT_NAME = climate-lens
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
	rm -rf .pytest_cache
	rm -rf .ruff_cache

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