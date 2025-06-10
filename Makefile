.PHONY: help env test test-cov test-simple pre-commit demo clean add add-dev

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

env:  ## Install package with dev dependencies
	uv pip install -e '.[dev]'

dep=%  # Usage: make add dep=package_name
add:  ## Add a new dependency (use dep=package_name)
	uv add $(dep)

add-dev:  ## Add a new dev dependency (use dep=package_name)
	uv add --dev $(dep)

test:  ## Run all tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov

test-simple:  ## Run tests without matplotlib comparison
	uv run pytest tests/ -v -o addopts=""

pre-commit:  ## Install pre-commit hooks
	uvx pre-commit install

demo:  ## Run presentation demos
	uv run python presentation_demos.py

clean:  ## Clean cache and temporary files
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
