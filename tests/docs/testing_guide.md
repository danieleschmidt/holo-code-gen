# Holo-Code-Gen Testing Guide

## Overview

This guide covers the testing infrastructure and practices for Holo-Code-Gen, including unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini              # Pytest configuration  
├── coverage.ini             # Coverage configuration
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── e2e/                     # End-to-end tests
├── performance/             # Performance benchmarks
├── fixtures/                # Test data and fixtures
└── docs/                    # Testing documentation
```

## Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation:
- Neural network parsing
- Template system
- Component libraries
- Optimization algorithms
- Simulation interfaces

**Example:**
```python
def test_neural_network_parser(sample_neural_network):
    parser = NeuralNetworkParser()
    graph = parser.parse(sample_neural_network)
    assert graph.num_nodes() > 0
    assert graph.validate()
```

### Integration Tests (`tests/integration/`)
Test component interactions:
- Compiler pipeline stages
- Template-to-circuit mapping
- Optimization workflows
- Simulation integration

**Example:**
```python
@pytest.mark.integration
def test_compiler_pipeline(sample_neural_network, process_parameters):
    compiler = PhotonicCompiler(process_parameters)
    circuit = compiler.compile(sample_neural_network)
    assert circuit.is_valid()
    assert circuit.meets_constraints()
```

### End-to-End Tests (`tests/e2e/`)
Test complete workflows:
- Neural network to GDS compilation
- Multi-wavelength optimization
- Performance validation
- Error handling

**Example:**
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_full_compilation_workflow(sample_neural_network, temp_dir):
    # Complete compilation pipeline
    result = compile_neural_network_to_gds(
        network=sample_neural_network,
        output_dir=temp_dir
    )
    assert result.success
    assert (temp_dir / "layout.gds").exists()
```

### Performance Tests (`tests/performance/`)
Benchmark performance and scalability:
- Compilation speed
- Memory usage
- Simulation performance
- Optimization convergence

**Example:**
```python
@pytest.mark.performance
def test_compilation_performance(benchmark, large_neural_network):
    def compile_network():
        return PhotonicCompiler().compile(large_neural_network)
    
    result = benchmark(compile_network)
    assert result.total_time < 60  # seconds
```

## Test Fixtures

### Shared Fixtures (`conftest.py`)

#### Neural Networks
- `sample_neural_network`: Basic MLP for testing
- `sample_spiking_network`: Spiking neural network
- `large_neural_network`: Large network for performance tests

#### Photonic Components
- `photonic_wavelength`: Standard telecom wavelength (1550nm)
- `process_parameters`: Silicon photonic process specs
- `imec_template_config`: IMEC template library config

#### Testing Infrastructure
- `temp_dir`: Temporary directory for test outputs
- `test_data_dir`: Test fixtures and reference data
- `simulation_config`: Mock simulation configuration

### Custom Fixtures

Create domain-specific fixtures:

```python
@pytest.fixture
def custom_neural_architecture():
    return {
        "type": "transformer",
        "layers": 12,
        "heads": 8,
        "embedding_dim": 512
    }

@pytest.fixture  
def photonic_test_bench(simulation_config):
    return PhotonicTestBench(
        config=simulation_config,
        mock_mode=True
    )
```

## Test Markers

Use markers to categorize and filter tests:

```python
# Slow tests (skip in CI fast mode)
@pytest.mark.slow
def test_large_circuit_optimization():
    pass

# Tests requiring simulation software
@pytest.mark.photonic
def test_optical_simulation():
    pass

# Tests requiring foundry PDKs
@pytest.mark.foundry
def test_foundry_specific_rules():
    pass

# Performance benchmarks
@pytest.mark.performance
def test_compilation_speed():
    pass
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m "not slow"              # Skip slow tests
pytest -m "photonic and not slow" # Photonic tests, skip slow

# Run specific test files
pytest tests/unit/test_neural_network_parser.py
pytest tests/integration/test_compiler_integration.py

# Run with coverage
pytest --cov=holo_code_gen --cov-report=html
```

### Parallel Testing

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto                    # Auto-detect CPU count
pytest -n 4                       # Use 4 processes
```

### Development Testing

```bash
# Fast feedback during development
pytest -x                         # Stop on first failure
pytest --lf                       # Run last failed tests
pytest --ff                       # Run failures first

# Watch mode (requires pytest-watch)
pip install pytest-watch
ptw                               # Auto-run tests on file changes
```

## Mock Testing

### Simulation Mocking

For tests that don't require full simulation:

```python
@pytest.fixture
def mock_simulator(mocker):
    mock_sim = mocker.Mock()
    mock_sim.simulate.return_value = MockResults({
        "transmission": [0.95, 0.94, 0.93],
        "phase": [0.1, 0.15, 0.2]
    })
    return mock_sim

def test_circuit_analysis(mock_simulator):
    analyzer = CircuitAnalyzer(simulator=mock_simulator)
    results = analyzer.analyze_circuit(test_circuit)
    assert results.transmission_average > 0.9
```

### Template Library Mocking

```python
@pytest.fixture
def mock_template_library(mocker):
    mock_lib = mocker.Mock()
    mock_lib.get_component.return_value = MockComponent({
        "type": "ring_resonator",
        "area": 100,  # μm²
        "power": 5    # mW
    })
    return mock_lib
```

## Performance Testing

### Benchmarking with pytest-benchmark

```python
def test_neural_network_parsing_speed(benchmark, large_neural_network):
    parser = NeuralNetworkParser()
    
    # Benchmark the parsing operation
    result = benchmark(parser.parse, large_neural_network)
    
    # Verify result quality
    assert result.num_layers > 10
    assert result.total_parameters > 1000000

def test_memory_usage(memory_profiler, large_neural_network):
    """Test memory usage during compilation."""
    initial_memory = memory_profiler.current()
    
    compiler = PhotonicCompiler()
    circuit = compiler.compile(large_neural_network)
    
    peak_memory = memory_profiler.peak()
    assert (peak_memory - initial_memory) < 1000  # MB
```

### Performance Regression Testing

```python
# Store baseline performance metrics
PERFORMANCE_BASELINES = {
    "simple_mlp_compilation": {"max_time": 10.0, "max_memory": 100},
    "ring_resonator_simulation": {"max_time": 5.0, "max_memory": 50}
}

def test_performance_regression(benchmark):
    test_name = "simple_mlp_compilation"
    baseline = PERFORMANCE_BASELINES[test_name]
    
    result = benchmark(compile_simple_mlp)
    
    assert result.total_time < baseline["max_time"]
    # Memory check would require memory profiling
```

## Test Data Management

### Fixture Data

Store test data in `tests/fixtures/`:

```
fixtures/
├── neural_networks/
│   ├── simple_mlp.json
│   ├── conv_net.json
│   └── spiking_network.json
├── photonic_circuits/
│   ├── ring_resonator.gds
│   └── mzi_mesh.gds
└── simulation_data/
    ├── optical_responses.json
    └── thermal_profiles.json
```

### Loading Test Data

```python
def test_reference_circuit_compilation(test_data_dir):
    # Load reference neural network
    network_file = test_data_dir / "neural_networks" / "simple_mlp.json"
    with open(network_file) as f:
        reference_network = json.load(f)
    
    # Load expected output
    expected_gds = test_data_dir / "photonic_circuits" / "simple_mlp.gds"
    
    # Test compilation
    compiled_circuit = compile_network(reference_network)
    assert circuits_match(compiled_circuit.gds, expected_gds)
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        
    - name: Run tests
      run: |
        pytest -m "not slow" --cov=holo_code_gen
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Environment Setup

```bash
# Development environment
pip install -e ".[dev]"
pre-commit install

# CI environment  
pip install -e ".[dev]"
pytest --cov=holo_code_gen --cov-report=xml
```

## Best Practices

### Test Organization

1. **One test per behavior**: Each test should verify one specific behavior
2. **Descriptive names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent tests**: Tests should not depend on each other

### Fixture Design

1. **Reusable fixtures**: Create fixtures that can be used across multiple tests
2. **Parameterized fixtures**: Use parameterization for testing multiple scenarios
3. **Scoped fixtures**: Use appropriate fixture scopes (function, class, module, session)
4. **Cleanup**: Ensure fixtures clean up resources properly

### Mock Usage

1. **Mock external dependencies**: Mock file systems, network calls, expensive simulations
2. **Verify interactions**: Use mocks to verify component interactions
3. **Realistic mocks**: Ensure mocks behave like real components
4. **Mock boundaries**: Mock at architectural boundaries, not internal details

### Performance Testing

1. **Baseline establishment**: Establish performance baselines for regression testing
2. **Resource monitoring**: Monitor memory, CPU, and disk usage
3. **Scalability testing**: Test with various input sizes
4. **Environment consistency**: Use consistent test environments for reliable results

## Troubleshooting

### Common Issues

**Tests failing locally but passing in CI:**
- Check Python version differences
- Verify environment variable setup
- Ensure test isolation

**Slow test execution:**
- Use markers to skip slow tests during development
- Profile tests to identify bottlenecks
- Consider parallel test execution

**Memory issues in tests:**
- Check for memory leaks in test fixtures
- Use memory profiling to identify issues
- Ensure proper cleanup of resources

**Flaky tests:**
- Identify timing-dependent behavior
- Use proper mocking for external dependencies
- Ensure test isolation and cleanup

### Debugging Tests

```python
# Add debugging to tests
def test_circuit_compilation(sample_neural_network):
    import pdb; pdb.set_trace()  # Debugger breakpoint
    
    circuit = compile_network(sample_neural_network)
    
    # Add logging for debugging
    logger.debug(f"Circuit has {circuit.num_components} components")
    
    assert circuit.is_valid()
```

For more information, see the [Contributing Guide](../../CONTRIBUTING.md) and [Development Setup](../../docs/DEVELOPMENT.md).