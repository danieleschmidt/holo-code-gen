# Multi-stage Dockerfile for Holo-Code-Gen
# Optimized for photonic circuit simulation and development

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels .

# Runtime stage
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd --gid 1000 holo && \
    useradd --uid 1000 --gid holo --shell /bin/bash --create-home holo

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/* && \
    rm -rf /tmp/wheels

# Copy application code
COPY --chown=holo:holo . /app
WORKDIR /app

# Install in development mode
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER holo

# Set up environment
ENV PYTHONPATH=/app
ENV HOLO_CODE_GEN_CONFIG_PATH=/app/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import holo_code_gen; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "holo_code_gen.cli", "--help"]

# Development stage
FROM runtime as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    git \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    pre-commit \
    jupyter

# Install simulation dependencies (optional)
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.5.0 || true

USER holo

# Development command
CMD ["/bin/bash"]