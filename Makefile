.PHONY: help install install-dev test test-cov lint format type-check clean docs serve-docs build publish

# Default target
help:
	@echo "Available targets:"
	@echo "  install       Install package in current environment"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  test          Run test suite"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with ruff"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean build artifacts"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"
	@echo "  build         Build distribution packages"
	@echo "  publish       Publish to PyPI"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest

test-cov:
	pytest --cov=holo_code_gen --cov-report=html --cov-report=term

test-photonic:
	pytest -m photonic

test-foundry:
	pytest -m foundry

# Code quality
lint:
	ruff check .

format:
	ruff format .

type-check:
	mypy holo_code_gen

check-all: lint type-check test

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs && make html && python -m http.server 8000 -d _build/html

# Distribution
build:
	python -m build

publish:
	python -m twine upload dist/*

# Development shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"

ci: check-all
	@echo "CI checks passed!"

# Photonic-specific targets
simulate:
	@echo "Running photonic simulations..."
	pytest tests/simulation/ -v

generate-gds:
	@echo "Generating GDS files from examples..."
	python examples/generate_all_gds.py

validate-designs:
	@echo "Validating photonic designs..."
	python scripts/validate_designs.py