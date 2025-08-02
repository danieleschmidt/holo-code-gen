# Build System Documentation

## Overview

Holo-Code-Gen uses a modern Python build system based on `hatchling` with comprehensive automation through Make and Docker.

## Build Configuration

### pyproject.toml

The project uses PEP 517/518 compliant build configuration:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.version]
path = "holo_code_gen/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["holo_code_gen"]
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Install package in current environment |
| `make install-dev` | Install with development dependencies |
| `make test` | Run test suite |
| `make test-cov` | Run tests with coverage |
| `make lint` | Check code quality |
| `make format` | Format code |
| `make type-check` | Run type checking |
| `make build` | Build distribution packages |
| `make clean` | Clean build artifacts |
| `make ci` | Run all CI checks |

## Build Process

### 1. Development Build

```bash
# Clone repository
git clone https://github.com/yourusername/holo-code-gen.git
cd holo-code-gen

# Setup development environment
make dev-setup

# Verify installation
python -c "import holo_code_gen; print(holo_code_gen.__version__)"
```

### 2. Distribution Build

```bash
# Clean previous builds
make clean

# Run quality checks
make ci

# Build packages
make build

# Verify build artifacts
ls -la dist/
```

Output structure:
```
dist/
├── holo_code_gen-0.1.0-py3-none-any.whl
└── holo_code_gen-0.1.0.tar.gz
```

### 3. Docker Build

```bash
# Build runtime image
docker build -t holo-code-gen:latest .

# Build development image
docker build -t holo-code-gen:dev --target development .

# Multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 -t holo-code-gen:latest .
```

## Dependency Management

### Core Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **torch**: Machine learning framework
- **networkx**: Graph algorithms
- **pydantic**: Data validation
- **typer**: CLI framework
- **rich**: Terminal formatting
- **gdstk**: GDS file manipulation

### Optional Dependencies

#### Simulation Support
```bash
pip install holo-code-gen[simulation]
```
- **meep**: FDTD simulation
- **scikit-rf**: RF analysis
- **photonics-toolkit**: Photonic design tools

#### Foundry Support
```bash
pip install holo-code-gen[foundry]
```
- **klayout**: Layout editor API
- **siepic**: Silicon photonics EDA

#### Development Tools
```bash
pip install holo-code-gen[dev]
```
- **pytest**: Testing framework
- **black**: Code formatter
- **ruff**: Fast linter
- **mypy**: Type checker

## Build Optimization

### Performance Optimizations

1. **Multi-stage Docker builds** reduce image size
2. **Wheel caching** speeds up repeated builds
3. **Layer optimization** minimizes rebuild time
4. **Parallel testing** with pytest-xdist

### Security Optimizations

1. **Non-root user** in containers
2. **Minimal base images** reduce attack surface
3. **Dependency scanning** with safety/bandit
4. **Secret management** with environment variables

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: make ci
```

### Build Matrix

Test across multiple environments:

| Python | OS | Architecture |
|--------|----|----|
| 3.9 | Ubuntu | x64 |
| 3.10 | Ubuntu | x64 |
| 3.11 | Ubuntu | x64 |
| 3.12 | Ubuntu | x64 |
| 3.11 | macOS | x64 |
| 3.11 | Windows | x64 |

## Troubleshooting

### Common Build Issues

#### Import Errors
```bash
# Ensure proper installation
pip install -e .
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### Dependency Conflicts
```bash
# Use fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install -e ".[dev]"
```

#### Docker Build Failures
```bash
# Clear Docker cache
docker builder prune -a

# Verbose build
docker build --progress=plain --no-cache .
```

### Performance Debugging

#### Build Time Analysis
```bash
# Time each build step
time make clean
time make build

# Docker build timing
docker build --progress=plain . 2>&1 | ts
```

#### Memory Usage
```bash
# Monitor during build
docker build --memory=2g .

# Check system resources
docker system df
```

## Release Process

### Semantic Versioning

Version format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes

### Release Automation

Automated releases use semantic-release:

```bash
# Install semantic-release
npm install -g semantic-release

# Dry run
semantic-release --dry-run

# Execute release
semantic-release
```

Release triggers:
- `feat:` → Minor version bump
- `fix:` → Patch version bump
- `BREAKING CHANGE:` → Major version bump

## Advanced Configuration

### Custom Build Scripts

```python
# scripts/build_extensions.py
"""Custom build script for photonic extensions."""

def build_photonic_libs():
    """Build custom photonic simulation libraries."""
    pass

if __name__ == "__main__":
    build_photonic_libs()
```

### Environment-specific Builds

```bash
# Production build
export HOLO_BUILD_ENV=production
make build

# Development build
export HOLO_BUILD_ENV=development
make build
```

### Cross-compilation

```dockerfile
# Multi-arch Dockerfile
FROM --platform=$BUILDPLATFORM python:3.11-slim as builder
ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "Building on $BUILDPLATFORM for $TARGETPLATFORM"
```