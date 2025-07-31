# Development Guide

This guide provides detailed information for developers working on Holo-Code-Gen.

## Architecture Overview

See [Architecture Documentation](architecture/) for detailed system design.

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Virtual environment manager (venv, conda, or similar)

### Quick Setup

```bash
git clone https://github.com/yourusername/holo-code-gen.git
cd holo-code-gen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

### Development Workflow

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and add tests
3. Run quality checks: `pytest && ruff check && mypy src/`
4. Commit with conventional commits
5. Push and create pull request

## Testing Strategy

### Test Categories

- **Unit Tests**: Fast, isolated component tests
- **Integration Tests**: Multi-component interactions
- **Simulation Tests**: Require photonic simulation tools
- **Foundry Tests**: Require PDK access

### Running Tests

```bash
# All tests
pytest

# Specific categories
pytest -m "not slow"
pytest -m simulation
pytest -m foundry

# With coverage
pytest --cov=src --cov-report=html
```

## Code Quality

### Automated Checks

- **Black**: Code formatting
- **Ruff**: Linting and code analysis
- **MyPy**: Type checking
- **Bandit**: Security scanning
- **Pre-commit**: Automated quality gates

### Manual Review

- Code review checklist
- Performance impact assessment
- Photonic domain validation
- Documentation completeness

## Photonic Development Guidelines

### Physical Units

- Wavelength: nanometers (nm)
- Dimensions: micrometers (μm)
- Power: milliwatts (mW)
- Loss: decibels (dB)

### Validation Requirements

- Design rule compliance
- Fabrication constraints
- Process variation tolerance
- Performance benchmarks

## Release Process

See [workflows/](workflows/) for CI/CD documentation and release procedures.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.