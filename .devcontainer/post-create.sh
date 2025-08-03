#!/bin/bash

# Holo-Code-Gen Development Environment Setup
set -e

echo "🚀 Setting up Holo-Code-Gen development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    graphviz \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
echo "⚙️ Installing holo-code-gen in development mode..."
pip install -e ".[dev]"

# Install additional scientific computing tools
echo "🧮 Installing scientific computing tools..."
pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    plotly \
    seaborn \
    scikit-learn \
    opencv-python-headless

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/models
mkdir -p data/circuits
mkdir -p data/benchmarks
mkdir -p output/gds
mkdir -p output/simulation
mkdir -p logs

# Setup environment file
echo "🔧 Creating environment configuration..."
cat > .env.example << EOF
# Holo-Code-Gen Environment Configuration

# Development settings
DEVELOPMENT=true
DEBUG=true
LOG_LEVEL=INFO

# Photonic simulation settings
MEEP_ENABLED=false
FDTD_SOLVER=meep
DEFAULT_WAVELENGTH=1550
DEFAULT_MATERIAL=silicon_nitride

# Template library settings
IMEC_LIBRARY_PATH=./templates/imec
CUSTOM_LIBRARY_PATH=./templates/custom

# Performance settings
MAX_THREADS=4
CACHE_ENABLED=true
CACHE_SIZE=1000

# Export settings
DEFAULT_EXPORT_FORMAT=gds
GDS_PRECISION=1e-9
NETLIST_FORMAT=spice

# Security settings
ALLOW_CUSTOM_TEMPLATES=true
VALIDATE_IMPORTS=true
EOF

# Create development scripts
echo "📝 Creating development scripts..."
mkdir -p scripts/dev

cat > scripts/dev/run_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "🧪 Running test suite..."

# Unit tests
echo "Running unit tests..."
pytest tests/unit/ -v --cov=holo_code_gen --cov-report=html

# Integration tests
echo "Running integration tests..."
pytest tests/integration/ -v

# Performance tests
echo "Running performance tests..."
pytest tests/performance/ -v -m "not slow"

# Check coverage
echo "Generating coverage report..."
coverage report --show-missing

echo "✅ All tests completed!"
EOF

cat > scripts/dev/lint_code.sh << 'EOF'
#!/bin/bash
set -e

echo "🔍 Running code quality checks..."

# Format with black
echo "Formatting code with black..."
black holo_code_gen/ tests/ examples/

# Sort imports
echo "Sorting imports with ruff..."
ruff --fix holo_code_gen/ tests/ examples/

# Lint with ruff
echo "Linting with ruff..."
ruff check holo_code_gen/ tests/ examples/

# Type checking with mypy
echo "Type checking with mypy..."
mypy holo_code_gen/

# Security scanning with bandit
echo "Security scanning with bandit..."
bandit -r holo_code_gen/ -c security/configs/bandit.yml

echo "✅ Code quality checks completed!"
EOF

cat > scripts/dev/build_docs.sh << 'EOF'
#!/bin/bash
set -e

echo "📚 Building documentation..."

# Create docs build directory
mkdir -p docs/_build

# Build API documentation
echo "Generating API documentation..."
sphinx-apidoc -f -o docs/api holo_code_gen/

# Build HTML documentation
echo "Building HTML documentation..."
sphinx-build docs/ docs/_build/html

# Build PDF documentation (if LaTeX available)
if command -v pdflatex &> /dev/null; then
    echo "Building PDF documentation..."
    sphinx-build -b latex docs/ docs/_build/latex
    cd docs/_build/latex && make
    cd -
fi

echo "✅ Documentation build completed!"
echo "📖 Open docs/_build/html/index.html to view documentation"
EOF

# Make scripts executable
chmod +x scripts/dev/*.sh

# Setup Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name holo-code-gen --display-name "Holo-Code-Gen"

# Create example data
echo "📊 Creating example data..."
python -c "
import torch
import torch.nn as nn
import json
from pathlib import Path

# Create example neural network
class ExampleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Save example model
model = ExampleMLP()
torch.save(model, 'data/models/example_mlp.pth')

# Create example configuration
config = {
    'model_name': 'example_mlp',
    'input_shape': [1, 784],
    'optimization_target': 'power',
    'power_budget': 500.0,
    'area_budget': 50.0
}

with open('data/models/example_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Created example model and configuration')
"

# Setup git hooks for development
echo "🔗 Setting up git hooks..."
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
set -e

echo "🚀 Running pre-push checks..."

# Run quick tests
pytest tests/unit/ -x --tb=short

# Check code quality
ruff check holo_code_gen/
mypy holo_code_gen/ --no-error-summary

echo "✅ Pre-push checks passed!"
EOF

chmod +x .git/hooks/pre-push

# Final setup message
echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  scripts/dev/run_tests.sh    - Run test suite"
echo "  scripts/dev/lint_code.sh    - Run code quality checks"
echo "  scripts/dev/build_docs.sh   - Build documentation"
echo "  holo-code-gen --help        - Show CLI help"
echo "  jupyter lab                 - Start Jupyter Lab"
echo ""
echo "Example usage:"
echo "  holo-code-gen compile data/models/example_mlp.pth"
echo "  holo-code-gen list-templates"
echo ""
echo "Happy coding! 🚀"