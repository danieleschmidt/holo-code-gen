# Docker Deployment Guide

## Overview

This guide covers Docker-based deployment strategies for Holo-Code-Gen, from local development to production environments.

## Docker Images

### Multi-stage Architecture

The Dockerfile implements a multi-stage build pattern:

1. **Builder stage**: Compiles dependencies and builds wheels
2. **Runtime stage**: Minimal production image
3. **Development stage**: Extended image with dev tools

### Image Variants

| Image | Target | Size | Use Case |
|-------|--------|------|----------|
| `holo-code-gen:latest` | runtime | ~200MB | Production |
| `holo-code-gen:dev` | development | ~500MB | Development |
| `holo-code-gen:slim` | runtime | ~150MB | Resource-constrained |

## Local Development

### Quick Start

```bash
# Start development environment
docker-compose up holo-dev

# Access development container
docker exec -it holo-code-gen-dev bash

# Run tests inside container
docker exec -it holo-code-gen-dev pytest

# Access Jupyter notebook
# Navigate to http://localhost:8888
```

### Development Workflow

```bash
# Start services
docker-compose up -d holo-dev

# Install additional packages
docker exec -it holo-code-gen-dev pip install new-package

# Code changes are reflected immediately via volume mounts
# No need to rebuild for code changes

# Run specific commands
docker exec -it holo-code-gen-dev make test
docker exec -it holo-code-gen-dev python -m holo_code_gen.cli --help
```

### Volume Mounts

Development setup includes strategic volume mounts:

```yaml
volumes:
  - .:/app                    # Source code (read-write)
  - holo-cache:/app/.cache   # Build cache (persistent)
  - holo-venv:/app/venv      # Virtual environment (persistent)
```

## Production Deployment

### Single Container

```bash
# Build production image
docker build -t holo-code-gen:prod .

# Run production container
docker run -d \
  --name holo-prod \
  --restart unless-stopped \
  -v /data:/app/data \
  -v /output:/app/output \
  -e HOLO_CODE_GEN_LOG_LEVEL=INFO \
  holo-code-gen:prod
```

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  holo-code-gen:
    image: holo-code-gen:latest
    restart: unless-stopped
    environment:
      - HOLO_CODE_GEN_LOG_LEVEL=INFO
      - HOLO_CODE_GEN_WORKERS=4
    volumes:
      - production-data:/app/data
      - production-output:/app/output
    healthcheck:
      test: ["CMD", "python", "-c", "import holo_code_gen"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'

volumes:
  production-data:
  production-output:
```

## Specialized Deployments

### Simulation Workloads

For compute-intensive photonic simulations:

```bash
# Start simulation service
docker-compose --profile simulation up -d

# Monitor resource usage
docker stats holo-simulation

# Scale simulation workers
docker-compose --profile simulation up --scale simulation=4
```

### High-Performance Computing

```dockerfile
# Dockerfile.hpc - HPC-optimized image
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install dependencies optimized for GPU compute
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libblas-dev \
    liblapack-dev \
    libcuda-dev

# GPU-accelerated photonic simulation
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt

# Configure for GPU workloads
ENV CUDA_VISIBLE_DEVICES=all
ENV HOLO_CODE_GEN_GPU_ENABLED=true
```

```bash
# Run with GPU support
docker run --gpus all \
  -v /data:/app/data \
  holo-code-gen:hpc python -m holo_code_gen.simulation.gpu_fdtd
```

## Monitoring and Observability

### Built-in Monitoring Stack

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access interfaces
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Health Checks

Application includes comprehensive health checks:

```bash
# Container health
docker inspect --format='{{.State.Health.Status}}' holo-code-gen

# Application endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics  # Prometheus metrics
```

### Log Management

```bash
# View logs
docker logs holo-code-gen -f

# Structured logging
docker logs holo-code-gen 2>&1 | jq .

# Log aggregation with ELK
docker-compose -f docker-compose.elk.yml up -d
```

## Security Considerations

### Container Security

1. **Non-root execution**: Containers run as user `holo` (UID 1000)
2. **Read-only filesystem**: Root filesystem mounted read-only
3. **Security scanning**: Images scanned with Trivy
4. **Minimal attack surface**: Alpine-based images available

```bash
# Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image holo-code-gen:latest
```

### Secrets Management

```bash
# Use Docker secrets
echo "sensitive_api_key" | docker secret create holo_api_key -

# Mount secrets
docker run -d \
  --secret holo_api_key \
  -e HOLO_API_KEY_FILE=/run/secrets/holo_api_key \
  holo-code-gen:latest
```

### Network Security

```yaml
# Secure network configuration
networks:
  holo-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
    internal: true  # No external access

  holo-public:
    driver: bridge
    # Public-facing network
```

## Performance Optimization

### Resource Allocation

```yaml
# Resource constraints
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

### Caching Strategies

```dockerfile
# Multi-layer caching
FROM python:3.11-slim as base

# Cache dependencies separately from code
COPY pyproject.toml .
RUN pip install -e .

# Copy code last (changes frequently)
COPY . .
```

### Image Optimization

```bash
# Multi-architecture builds
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t holo-code-gen:latest .

# Size optimization
docker build --target runtime --squash -t holo-code-gen:slim .
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker logs holo-code-gen

# Debug interactively
docker run -it --entrypoint /bin/bash holo-code-gen:latest

# Check resource constraints
docker system df
```

#### Permission Issues
```bash
# Fix volume permissions
docker run --rm -v $(pwd):/app alpine chown -R 1000:1000 /app

# Run as specific user
docker run --user 1000:1000 holo-code-gen:latest
```

#### Network Connectivity
```bash
# Test network
docker network ls
docker network inspect holo-network

# Debug DNS
docker run --rm --network holo-network alpine nslookup holo-code-gen
```

### Performance Debugging

#### Memory Issues
```bash
# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Analyze memory leaks
docker exec -it holo-code-gen python -m memory_profiler script.py
```

#### CPU Bottlenecks
```bash
# Profile CPU usage
docker exec -it holo-code-gen py-spy top --pid 1

# CPU limits
docker run --cpus=".5" holo-code-gen:latest
```

## Advanced Configurations

### Custom Entrypoints

```dockerfile
# Multi-purpose entrypoint
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

```bash
#!/bin/bash
# entrypoint.sh
case "$1" in
  "server")
    exec python -m holo_code_gen.server
    ;;
  "worker")
    exec python -m holo_code_gen.worker
    ;;
  "cli")
    shift
    exec python -m holo_code_gen.cli "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
```

### Environment-specific Configurations

```bash
# Development
docker run -e HOLO_ENV=development holo-code-gen:latest

# Staging
docker run -e HOLO_ENV=staging holo-code-gen:latest

# Production
docker run -e HOLO_ENV=production holo-code-gen:latest
```

## Migration Guide

### From Local to Container

1. **Identify dependencies**: List all system dependencies
2. **Extract configuration**: Move config to environment variables
3. **Data persistence**: Set up volume mounts
4. **Test thoroughly**: Validate functionality in container

### From Docker to Kubernetes

1. **Create manifests**: Convert docker-compose to K8s manifests
2. **Configure ingress**: Set up load balancing
3. **Persistent volumes**: Configure storage classes
4. **Secrets management**: Use K8s secrets
5. **Monitoring**: Deploy Prometheus operator