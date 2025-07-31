# Contributing to Holo-Code-Gen

We welcome contributions to Holo-Code-Gen! This document provides guidelines for contributing to the project.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run the test suite: `make test`
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/holo-code-gen.git
cd holo-code-gen

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all public APIs
- Write comprehensive docstrings
- Add unit tests for new functionality
- Keep line length under 88 characters

## Testing

- Run tests: `pytest`
- Check coverage: `pytest --cov=holo_code_gen`
- Run linting: `ruff check .`
- Format code: `ruff format .`

## Photonic Design Guidelines

- Use IMEC template library for standard components
- Validate all photonic designs with simulation
- Include process variation analysis
- Document optical performance metrics

## Submitting Changes

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Create detailed pull request description

## Code Review Process

- All changes require review from maintainers
- Photonic designs need domain expert approval
- Breaking changes require broader discussion

## Getting Help

- Check existing [issues](https://github.com/yourusername/holo-code-gen/issues)
- Join our [discussions](https://github.com/yourusername/holo-code-gen/discussions)
- Read the [documentation](docs/)

Thank you for contributing!