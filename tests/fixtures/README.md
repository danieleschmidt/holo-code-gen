# Test Fixtures

This directory contains test data and fixtures used by the Holo-Code-Gen test suite.

## Directory Structure

```
fixtures/
├── neural_networks/     # Sample neural network models
├── photonic_circuits/   # Reference photonic circuit designs
├── gds_files/          # Sample GDS layout files
├── simulation_data/    # Expected simulation results
├── templates/          # Test photonic component templates
└── configs/           # Test configuration files
```

## Neural Networks

Sample neural network models for testing compilation:

- `simple_mlp.json` - Basic multi-layer perceptron
- `conv_net.json` - Convolutional neural network
- `spiking_network.json` - Spiking neural network
- `transformer_block.json` - Transformer attention block

## Photonic Circuits

Reference photonic circuit designs for validation:

- `ring_resonator.gds` - Single ring resonator
- `mzi_mesh.gds` - Mach-Zehnder interferometer mesh
- `weight_bank.gds` - Photonic weight bank
- `neural_layer.gds` - Complete photonic neural layer

## Simulation Data

Expected simulation results for regression testing:

- `optical_responses/` - Frequency domain responses
- `thermal_profiles/` - Temperature distributions
- `noise_analysis/` - Signal-to-noise ratios
- `performance_metrics/` - Throughput and power data

## Usage in Tests

Load fixtures using the `test_data_dir` fixture:

```python
def test_circuit_compilation(test_data_dir, sample_neural_network):
    # Load reference data
    reference_gds = test_data_dir / "photonic_circuits" / "neural_layer.gds"
    
    # Test compilation
    compiled_circuit = compile_network(sample_neural_network)
    
    # Validate against reference
    assert compare_layouts(compiled_circuit.gds, reference_gds)
```

## Adding New Fixtures

1. Create files in appropriate subdirectory
2. Use descriptive names with version numbers if applicable
3. Include metadata files (`.json`) describing the fixture
4. Update this README with fixture descriptions
5. Add corresponding test cases using the fixtures