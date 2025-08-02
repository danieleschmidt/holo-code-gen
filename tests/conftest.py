"""Pytest configuration and shared fixtures for Holo-Code-Gen."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Any, List
import json
import os


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_gds_file(temp_dir: Path) -> Path:
    """Create a sample GDS file for testing."""
    gds_file = temp_dir / "test_layout.gds"
    # Create minimal GDS content placeholder
    gds_file.write_bytes(b"HEADER 600\nBGNLIB\nLIBNAME TEST\nUNITS\nBGNSTR\nSTRNAME TEST_CELL\nENDSTR\nENDLIB")
    return gds_file


@pytest.fixture
def sample_neural_network() -> Dict[str, Any]:
    """Provide a sample neural network for testing."""
    return {
        "name": "sample_mlp",
        "input_shape": [784],
        "layers": [
            {
                "type": "linear",
                "input_size": 784,
                "output_size": 128,
                "weights_shape": [784, 128],
                "bias_shape": [128]
            },
            {
                "type": "activation",
                "function": "relu",
                "input_size": 128,
                "output_size": 128
            },
            {
                "type": "linear",
                "input_size": 128,
                "output_size": 10,
                "weights_shape": [128, 10],
                "bias_shape": [10]
            }
        ],
        "total_parameters": 784 * 128 + 128 + 128 * 10 + 10
    }


@pytest.fixture
def sample_spiking_network() -> Dict[str, Any]:
    """Provide a sample spiking neural network for testing."""
    return {
        "name": "sample_snn",
        "input_shape": [28, 28],
        "layers": [
            {
                "type": "lif_neuron",
                "input_size": 784,
                "output_size": 256,
                "tau": 10.0,
                "threshold": 1.0,
                "reset_voltage": 0.0
            },
            {
                "type": "lif_neuron", 
                "input_size": 256,
                "output_size": 128,
                "tau": 15.0,
                "threshold": 1.0,
                "reset_voltage": 0.0
            },
            {
                "type": "lif_neuron",
                "input_size": 128,
                "output_size": 10,
                "tau": 20.0,
                "threshold": 1.0,
                "reset_voltage": 0.0
            }
        ],
        "simulation_time": 100.0,  # ms
        "time_step": 0.1  # ms
    }


@pytest.fixture
def photonic_wavelength() -> float:
    """Standard telecom wavelength for testing."""
    return 1550.0  # nm


@pytest.fixture
def process_parameters() -> Dict[str, Any]:
    """Standard silicon photonic process parameters."""
    return {
        "process_name": "IMEC_SiN_220nm",
        "waveguide_width": 450,  # nm
        "waveguide_height": 220,  # nm
        "substrate": "SiN_on_SiO2",
        "min_bend_radius": 5.0,  # μm
        "coupling_gap": 200,  # nm
        "metal_layers": 3,
        "via_size": 150,  # nm
        "design_rules": {
            "min_width": 100,  # nm
            "min_space": 150,  # nm
            "min_enclosure": 50  # nm
        }
    }


@pytest.fixture
def imec_template_config() -> Dict[str, Any]:
    """IMEC template library configuration for testing."""
    return {
        "library_version": "v2025_07",
        "components": [
            {
                "name": "microring_resonator",
                "type": "passive_filter",
                "parameters": {
                    "radius": 10.0,  # μm
                    "coupling_gap": 200,  # nm
                    "q_factor": 10000
                }
            },
            {
                "name": "mzi_modulator",
                "type": "active_modulator",
                "parameters": {
                    "arm_length": 500,  # μm
                    "phase_efficiency": 1.0,  # π/V
                    "extinction_ratio": 20  # dB
                }
            }
        ]
    }


@pytest.fixture
def simulation_config() -> Dict[str, Any]:
    """Simulation configuration for testing."""
    return {
        "backend": "meep",
        "resolution": 20,  # points per wavelength
        "pml_layers": 10,
        "wavelength_range": [1500, 1600],  # nm
        "convergence_threshold": 1e-6,
        "max_iterations": 10000,
        "parallel_jobs": 1
    }


# Performance test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "photonic: marks tests requiring photonic simulation"
    )
    config.addinivalue_line(
        "markers", "foundry: marks tests requiring foundry PDK access"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


# Photonic-specific test utilities
class PhotonicTestUtils:
    """Utilities for photonic testing."""
    
    @staticmethod
    def validate_wavelength(wavelength: float) -> bool:
        """Validate wavelength is in telecom range."""
        return 1200 <= wavelength <= 1700
    
    @staticmethod
    def calculate_loss_budget(components: list) -> float:
        """Calculate total optical loss for component chain."""
        return sum(comp.get("loss_db", 0) for comp in components)
    
    @staticmethod
    def validate_design_rules(layout_data: dict) -> list:
        """Validate photonic design rules."""
        violations = []
        # Placeholder for design rule checking
        return violations


@pytest.fixture
def photonic_utils():
    """Provide photonic testing utilities."""
    return PhotonicTestUtils()