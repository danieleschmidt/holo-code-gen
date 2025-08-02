#!/bin/bash

# Holo-Code-Gen Development Container Post-Creation Script

set -e

echo "🚀 Setting up Holo-Code-Gen development environment..."

# Upgrade pip and install build tools
echo "📦 Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install the project in development mode with all dependencies
echo "🔧 Installing Holo-Code-Gen in development mode..."
pip install -e ".[dev,simulation,foundry]"

# Install additional development tools
echo "🛠️  Installing additional development tools..."
pip install \
    pre-commit \
    jupyter \
    ipykernel \
    notebook \
    jupyterlab \
    sphinx-autobuild \
    mkdocs \
    mkdocs-material

# Set up pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Install system dependencies for photonic simulation
echo "📡 Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    gfortran \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-openmpi-dev \
    libfftw3-dev \
    libgsl-dev \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    git-lfs

# Initialize git-lfs for large files
echo "📂 Initializing Git LFS..."
git lfs install

# Create useful directories
echo "📁 Creating development directories..."
mkdir -p \
    scratch \
    notebooks \
    experiments \
    benchmarks \
    test_outputs \
    simulation_data

# Set up Jupyter kernel
echo "🪐 Setting up Jupyter kernel..."
python -m ipykernel install --user --name holo-code-gen --display-name "Holo-Code-Gen"

# Generate initial documentation
echo "📚 Building initial documentation..."
if [ -f "docs/conf.py" ]; then
    cd docs && make html && cd ..
fi

# Run initial tests to verify setup
echo "🧪 Running initial test suite..."
python -m pytest tests/ -v --tb=short || echo "⚠️  Some tests failed - this is expected for initial setup"

# Create useful aliases
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Holo-Code-Gen development aliases
alias hcg='holo-code-gen'
alias hcg-test='python -m pytest'
alias hcg-lint='ruff check . && mypy holo_code_gen'
alias hcg-format='black . && ruff --fix .'
alias hcg-docs='cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8080'
alias hcg-notebook='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias hcg-clean='find . -type d -name "__pycache__" -exec rm -rf {} + && find . -name "*.pyc" -delete'

# Useful environment variables
export HOLO_CODE_GEN_DEBUG=1
export HOLO_CODE_GEN_LOG_LEVEL=DEBUG
EOF

# Set up zsh aliases if zsh is available
if [ -f ~/.zshrc ]; then
    cat >> ~/.zshrc << 'EOF'

# Holo-Code-Gen development aliases
alias hcg='holo-code-gen'
alias hcg-test='python -m pytest'
alias hcg-lint='ruff check . && mypy holo_code_gen'
alias hcg-format='black . && ruff --fix .'
alias hcg-docs='cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8080'
alias hcg-notebook='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias hcg-clean='find . -type d -name "__pycache__" -exec rm -rf {} + && find . -name "*.pyc" -delete'

# Useful environment variables
export HOLO_CODE_GEN_DEBUG=1
export HOLO_CODE_GEN_LOG_LEVEL=DEBUG
EOF
fi

# Set up git configuration for development
echo "⚙️  Configuring git for development..."
git config --global pull.rebase false
git config --global init.defaultBranch main
git config --global core.autocrlf input

# Create sample environment file
echo "🔐 Creating sample environment configuration..."
cat > .env.example << 'EOF'
# Holo-Code-Gen Environment Configuration

# Development settings
HOLO_CODE_GEN_DEBUG=true
HOLO_CODE_GEN_LOG_LEVEL=DEBUG
HOLO_CODE_GEN_PROFILE=false

# Simulation settings
HOLO_CODE_GEN_SIMULATION_BACKEND=meep
HOLO_CODE_GEN_SIMULATION_THREADS=4
HOLO_CODE_GEN_SIMULATION_MEMORY_LIMIT=8GB

# Template library settings
HOLO_CODE_GEN_TEMPLATE_PATH=./templates
HOLO_CODE_GEN_IMEC_LICENSE_KEY=your_license_key_here
HOLO_CODE_GEN_CUSTOM_TEMPLATE_PATH=./custom_templates

# Foundry PDK settings
HOLO_CODE_GEN_PDK_PATH=./pdk
HOLO_CODE_GEN_DEFAULT_PDK=IMEC_SiN_220nm
HOLO_CODE_GEN_GDS_PRECISION=1  # nm

# Performance settings
HOLO_CODE_GEN_CACHE_ENABLED=true
HOLO_CODE_GEN_CACHE_SIZE=1GB
HOLO_CODE_GEN_PARALLEL_JOBS=auto

# Security settings
HOLO_CODE_GEN_SAFE_MODE=true
HOLO_CODE_GEN_ALLOW_CUSTOM_CODE=false
HOLO_CODE_GEN_AUDIT_LOG=true

# Integration settings
HOLO_CODE_GEN_KLAYOUT_PATH=/usr/local/bin/klayout
HOLO_CODE_GEN_MEEP_PATH=/usr/local/bin/meep
HOLO_CODE_GEN_GDSTK_PRECISION=1e-9
EOF

# Print setup completion message
echo ""
echo "✅ Holo-Code-Gen development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Copy .env.example to .env and configure your settings"
echo "   2. Run 'hcg-test' to run the test suite"
echo "   3. Run 'hcg-docs' to start the documentation server"
echo "   4. Run 'hcg-notebook' to start Jupyter Lab"
echo "   5. Check out the examples/ directory for getting started"
echo ""
echo "🔗 Useful commands:"
echo "   hcg --help           Show CLI help"
echo "   hcg-lint             Run linting and type checking"
echo "   hcg-format           Format code with black and ruff"
echo "   hcg-clean            Clean up Python cache files"
echo ""
echo "Happy coding! 🧬✨"