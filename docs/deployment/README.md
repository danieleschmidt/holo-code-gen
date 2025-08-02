# Deployment Documentation

This directory contains comprehensive deployment and build documentation for the Holo-Code-Gen project.

## Contents

- [Build System](./build-system.md) - Comprehensive build configuration and processes
- [Docker Guide](./docker-guide.md) - Container deployment and development
- [CI/CD Pipelines](./cicd-pipelines.md) - Continuous integration and deployment
- [Release Process](./release-process.md) - Semantic versioning and release automation
- [Security Guidelines](./security-guidelines.md) - Security best practices for deployment

## Quick Start

### Local Development Build

```bash
# Install development dependencies
make install-dev

# Run full CI checks
make ci

# Build distribution packages
make build
```

### Docker Development

```bash
# Start development environment
docker-compose up holo-dev

# Run with simulation capabilities
docker-compose --profile simulation up

# Run with monitoring stack
docker-compose --profile monitoring up
```

### Production Deployment

```bash
# Build production image
docker build -t holo-code-gen:latest .

# Run production container
docker run -d --name holo-prod holo-code-gen:latest
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOLO_CODE_GEN_LOG_LEVEL` | Logging verbosity | `INFO` |
| `HOLO_CODE_GEN_CONFIG_PATH` | Configuration directory | `/app/config` |
| `HOLO_CODE_GEN_CACHE_DIR` | Cache directory | `/app/cache` |
| `HOLO_CODE_GEN_DEV_MODE` | Enable development mode | `false` |
| `HOLO_CODE_GEN_SIMULATION_MODE` | Enable simulation mode | `false` |
| `HOLO_CODE_GEN_PARALLEL_WORKERS` | Number of parallel workers | `4` |

## Health Checks

The application includes built-in health checks:

```bash
# Application health
curl http://localhost:8000/health

# Docker health check
docker inspect --format='{{.State.Health.Status}}' holo-code-gen
```