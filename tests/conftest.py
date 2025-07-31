"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from typing import Generator


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
    gds_file.write_bytes(b"HEADER 600\nBGNLIB\nENDLIB")
    return gds_file


@pytest.fixture
def sample_neural_network():
    """Provide a sample neural network for testing."""
    # Placeholder for neural network fixture
    return {
        "layers": [
            {"type": "linear", "input_size": 784, "output_size": 128},
            {"type": "activation", "function": "relu"},
            {"type": "linear", "input_size": 128, "output_size": 10}
        ]
    }


@pytest.fixture
def photonic_wavelength():
    """Standard telecom wavelength for testing."""
    return 1550.0  # nm


@pytest.fixture
def process_parameters():
    """Standard silicon photonic process parameters."""
    return {
        "waveguide_width": 450,  # nm
        "waveguide_height": 220,  # nm
        "substrate": "SOI",
        "min_bend_radius": 5.0,  # μm
        "coupling_gap": 200,  # nm
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