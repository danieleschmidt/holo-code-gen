.PHONY: help install install-dev test test-cov lint format type-check clean docs serve-docs build publish terragon-setup terragon-discovery terragon-continuous terragon-status

# Default target
help: ## Show this help message
	@echo "Holo-Code-Gen Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Terragon Autonomous SDLC Commands:"
	@echo "  \033[36mterragon-setup\033[0m      Setup Terragon autonomous system"
	@echo "  \033[36mterragon-discovery\033[0m  Run value discovery once"
	@echo "  \033[36mterragon-continuous\033[0m Start continuous value discovery"
	@echo "  \033[36mterragon-status\033[0m     Show Terragon system status"

# Installation  
install: ## Install package in current environment
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt
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

# Terragon Autonomous SDLC System
terragon-setup: ## Setup Terragon autonomous system
	@echo "Setting up Terragon Autonomous SDLC System..."
	python -m pip install pyyaml
	mkdir -p .terragon
	chmod +x .terragon/discovery-engine.py
	@echo "✅ Terragon system ready"

terragon-discovery: ## Run value discovery analysis once
	@echo "Running Terragon value discovery..."
	python .terragon/discovery-engine.py --update
	@echo "📊 Check BACKLOG.md for discovered items"

terragon-continuous: ## Start continuous value discovery
	@echo "Starting continuous Terragon value discovery..."
	@echo "This will run every hour. Press Ctrl+C to stop."
	python .terragon/discovery-engine.py --continuous

terragon-status: ## Show Terragon system status
	@echo "=== Terragon System Status ==="
	@echo "Configuration: $$([ -f .terragon/config.yaml ] && echo '✅ Present' || echo '❌ Missing')"
	@echo "Discovery Engine: $$([ -x .terragon/discovery-engine.py ] && echo '✅ Ready' || echo '❌ Not Ready')"
	@echo "Value Metrics: $$([ -f .terragon/value-metrics.json ] && echo '✅ Present' || echo '❌ Missing')"
	@echo "Backlog: $$([ -f BACKLOG.md ] && echo '✅ Present' || echo '❌ Missing')"