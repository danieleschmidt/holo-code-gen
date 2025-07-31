# Installation Guide

## Prerequisites

- Python 3.9 or higher
- Git
- (Optional) CUDA for GPU acceleration
- (Optional) Foundry PDKs for advanced features

## Basic Installation

```bash
pip install holo-code-gen
```

## Development Installation

```bash
git clone https://github.com/yourusername/holo-code-gen.git
cd holo-code-gen
make install-dev
```

## Optional Dependencies

### Photonic Simulation
```bash
pip install holo-code-gen[simulation]
```

### Foundry Support
```bash
pip install holo-code-gen[foundry]
```

### Full Installation
```bash
pip install holo-code-gen[full]
```

## Verification

```bash
holo-code-gen --version
python -c "import holo_code_gen; print('Installation successful!')"
```