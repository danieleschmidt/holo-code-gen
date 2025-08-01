# Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for common issues encountered when using the holo-code-gen photonic neural network toolchain.

## Common Issues and Solutions

### Installation Issues

#### Issue: Package Installation Fails
**Symptoms:**
```bash
ERROR: Could not find a version that satisfies the requirement holo-code-gen
ERROR: Failed building wheel for gdstk
```

**Solutions:**
1. **Update pip and setuptools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential python3-dev libffi-dev
   ```

3. **Install system dependencies (macOS):**
   ```bash
   brew install gcc python3-dev
   xcode-select --install
   ```

4. **Use conda for complex dependencies:**
   ```bash
   conda create -n holo-code-gen python=3.9
   conda activate holo-code-gen
   conda install -c conda-forge gdstk numpy scipy
   pip install holo-code-gen
   ```

#### Issue: Optional Dependencies Not Available
**Symptoms:**
```bash
ModuleNotFoundError: No module named 'meep'
ImportError: cannot import name 'klayout'
```

**Solutions:**
1. **Install simulation dependencies:**
   ```bash
   pip install holo-code-gen[simulation]
   # or
   conda install -c conda-forge meep scikit-rf
   ```

2. **Install foundry dependencies:**
   ```bash
   pip install holo-code-gen[foundry]
   # Note: KLayout may require separate installation
   ```

3. **Check installation:**
   ```python
   import holo_code_gen
   print(holo_code_gen.check_dependencies())
   ```

### Compilation Issues

#### Issue: PhotonicCompiler Not Found
**Symptoms:**
```python
AttributeError: module 'holo_code_gen' has no attribute 'PhotonicCompiler'
```

**Solutions:**
1. **Check current implementation status:**
   ```python
   import holo_code_gen
   print(holo_code_gen.__version__)
   print(holo_code_gen.list_available_classes())
   ```

2. **Use development installation:**
   ```bash
   git clone https://github.com/yourusername/holo-code-gen
   cd holo-code-gen
   pip install -e ".[dev]"
   ```

3. **Check TODO items:**
   - Review BACKLOG.md for implementation status
   - Many core classes are still in development phase

#### Issue: Neural Network Model Not Supported
**Symptoms:**
```python
NotImplementedError: Layer type 'BatchNorm2d' not supported
UnsupportedModelError: Custom activation function not recognized
```

**Solutions:**
1. **Check supported layers:**
   ```python
   from holo_code_gen import get_supported_layers
   print(get_supported_layers())
   ```

2. **Use supported alternatives:**
   ```python
   # Instead of BatchNorm2d, use LayerNorm
   # Instead of custom activations, use ReLU/Tanh
   ```

3. **Implement custom mapping:**
   ```python
   from holo_code_gen.templates import register_custom_layer
   
   @register_custom_layer('BatchNorm2d')
   def map_batchnorm(layer_spec):
       # Custom photonic implementation
       return photonic_normalization_block(layer_spec)
   ```

### Simulation Issues

#### Issue: FDTD Simulation Convergence Problems
**Symptoms:**
```bash
Warning: FDTD simulation did not converge after 10000 steps
Error: Field values becoming unstable (NaN detected)
```

**Solutions:**
1. **Adjust simulation parameters:**
   ```python
   simulator = PhotonicSimulator(
       method="fdtd",
       resolution=40,  # Increase resolution
       time_step_factor=0.5,  # Smaller time steps
       max_iterations=20000
   )
   ```

2. **Check boundary conditions:**
   ```python
   simulator.set_boundary_conditions(
       x="PML",  # Perfectly Matched Layer
       y="PML",
       z="periodic"  # If applicable
   )
   ```

3. **Use adaptive stepping:**
   ```python
   simulator.enable_adaptive_stepping(
       tolerance=1e-6,
       min_step=0.001,
       max_step=0.1
   )
   ```

#### Issue: Memory Issues During Simulation
**Symptoms:**
```bash
MemoryError: Unable to allocate array
Process killed (OOM killer)
```

**Solutions:**
1. **Reduce simulation domain:**
   ```python
   simulator.set_domain_size(
       x_span=reduced_x,
       y_span=reduced_y,
       resolution=20  # Lower resolution
   )
   ```

2. **Use streaming processing:**
   ```python
   results = simulator.simulate_streaming(
       chunk_size=1000,
       save_intermediate=True
   )
   ```

3. **Enable memory monitoring:**
   ```python
   simulator.enable_memory_monitoring(
       max_memory="8GB",
       warning_threshold=0.8
   )
   ```

### Layout and GDS Issues

#### Issue: Design Rule Check (DRC) Violations
**Symptoms:**
```bash
DRC Error: Minimum width violation at (100.5, 200.3)
DRC Error: Spacing violation between layers
```

**Solutions:**
1. **Check process design rules:**
   ```python
   from holo_code_gen.fabrication import load_pdk_rules
   rules = load_pdk_rules("IMEC_SiN_220nm")
   print(rules.minimum_width)
   print(rules.minimum_spacing)
   ```

2. **Enable auto-fix:**
   ```python
   layout_generator = LayoutGenerator(
       auto_fix_drc=True,
       snap_to_grid=True,
       grid_size=0.005  # 5nm grid
   )
   ```

3. **Use design margin:**
   ```python
   layout_generator.set_design_margins(
       width_margin=0.02,  # 20nm extra
       spacing_margin=0.01  # 10nm extra
   )
   ```

#### Issue: GDS File Size Too Large
**Symptoms:**
```bash
Warning: GDS file size 250MB exceeds recommended limit
Error: File system space exhausted
```

**Solutions:**
1. **Enable compression:**
   ```python
   layout.export_gds(
       filename="circuit.gds",
       compression=True,
       precision=0.001  # 1nm precision
   )
   ```

2. **Use hierarchical design:**
   ```python
   # Create reusable cells
   basic_cell = layout.create_cell("basic_component")
   # Reference instead of copying
   layout.add_cell_reference(basic_cell, position=(0, 0))
   ```

3. **Optimize data representation:**
   ```python
   layout.optimize_for_size(
       merge_overlapping=True,
       remove_duplicates=True,
       compress_coordinates=True
   )
   ```

### Performance Issues

#### Issue: Slow Compilation Times
**Symptoms:**
```bash
Compilation taking >30 minutes for medium network
Memory usage growing continuously
```

**Solutions:**
1. **Enable parallel processing:**
   ```python
   compiler = PhotonicCompiler(
       parallel_compilation=True,
       max_workers=8,
       memory_limit="16GB"
   )
   ```

2. **Use caching:**
   ```python
   compiler.enable_caching(
       cache_dir="~/.holo_cache",
       max_cache_size="5GB"
   )
   ```

3. **Optimize for speed:**
   ```python
   compiler.set_optimization_level(
       level="fast",  # vs "balanced" or "best"
       max_iterations=100
   )
   ```

#### Issue: High Memory Usage
**Symptoms:**
```bash
Process using >16GB RAM
System becoming unresponsive
```

**Solutions:**
1. **Enable memory profiling:**
   ```python
   from holo_code_gen.utils import MemoryProfiler
   profiler = MemoryProfiler()
   profiler.start()
   # ... your code ...
   profiler.report()
   ```

2. **Use streaming processing:**
   ```python
   processor = StreamingProcessor(
       chunk_size=1024,
       max_memory="8GB"
   )
   ```

3. **Clear intermediate data:**
   ```python
   compiler.clear_intermediate_data()
   import gc; gc.collect()
   ```

### Template and Library Issues

#### Issue: Template Library Not Found
**Symptoms:**
```bash
FileNotFoundError: Template library 'IMEC_v2025' not found
ImportError: Cannot load template definitions
```

**Solutions:**
1. **Check template installation:**
   ```bash
   holo-code-gen list-templates
   ```

2. **Download templates:**
   ```bash
   holo-code-gen download-templates --library IMEC_v2025
   ```

3. **Set template path:**
   ```python
   import os
   os.environ['HOLO_TEMPLATE_PATH'] = '/path/to/templates'
   ```

#### Issue: Custom Template Registration Fails
**Symptoms:**
```python
TemplateError: Invalid template specification
ValidationError: Required parameters missing
```

**Solutions:**
1. **Validate template specification:**
   ```python
   from holo_code_gen.templates import validate_template
   is_valid, errors = validate_template(my_template)
   if not is_valid:
       print(errors)
   ```

2. **Use template builder:**
   ```python
   from holo_code_gen.templates import TemplateBuilder
   builder = TemplateBuilder()
   builder.validate_continuously = True
   template = builder.build_template(spec)
   ```

### Development and Testing Issues

#### Issue: Tests Failing
**Symptoms:**
```bash
FAILED tests/test_compiler.py::test_basic_compilation
ModuleNotFoundError in test files
```

**Solutions:**
1. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   # or
   pip install -r requirements-dev.txt
   ```

2. **Run specific test categories:**
   ```bash
   pytest tests/unit/  # Only unit tests
   pytest -m "not slow"  # Skip slow tests
   pytest --tb=short  # Shorter traceback
   ```

3. **Check test environment:**
   ```python
   pytest --collect-only  # Show what tests would run
   pytest --fixtures  # Show available fixtures
   ```

#### Issue: Pre-commit Hooks Failing
**Symptoms:**
```bash
black....................................................................Failed
ruff.....................................................................Failed
```

**Solutions:**
1. **Install pre-commit:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **Run hooks manually:**
   ```bash
   pre-commit run --all-files
   pre-commit run black --files modified_file.py
   ```

3. **Fix common issues:**
   ```bash
   black .  # Auto-format code
   ruff check --fix .  # Auto-fix linting issues
   ```

## Debugging Strategies

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger('holo_code_gen.compiler').setLevel(logging.DEBUG)
```

### Use Interactive Debugging
```python
import pdb; pdb.set_trace()  # Set breakpoint
# Or use ipdb for better interface
import ipdb; ipdb.set_trace()
```

### Profile Performance
```bash
# Time profiling
python -m cProfile -o profile.stats my_script.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
mprof run my_script.py
mprof plot
```

## Getting Help

### Check Documentation
1. **README.md** - Basic usage and installation
2. **docs/DEVELOPMENT.md** - Development setup
3. **docs/ARCHITECTURE.md** - System architecture
4. **BACKLOG.md** - Current implementation status

### Community Support
1. **GitHub Issues** - Report bugs and feature requests
2. **Discussions** - Ask questions and share ideas
3. **Stack Overflow** - Use tag `holo-code-gen`

### Diagnostic Information
When reporting issues, include:

```python
# Diagnostic script
import holo_code_gen
import sys
import platform

print(f"holo-code-gen version: {holo_code_gen.__version__}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")

# Check dependencies
try:
    import numpy; print(f"NumPy: {numpy.__version__}")
except ImportError: print("NumPy: Not installed")

try:
    import torch; print(f"PyTorch: {torch.__version__}")
except ImportError: print("PyTorch: Not installed")

# System resources
import psutil
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"CPU cores: {psutil.cpu_count()}")
```

### Log Collection
```bash
# Enable comprehensive logging
export HOLO_LOG_LEVEL=DEBUG
export HOLO_LOG_FILE=holo_debug.log
python your_script.py

# Collect relevant logs
tar -czf debug_info.tar.gz holo_debug.log *.py requirements.txt
```

---

This troubleshooting guide will be updated as new issues are discovered and resolved. If you encounter an issue not covered here, please report it so we can add it to the guide.