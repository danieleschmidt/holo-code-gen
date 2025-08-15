"""Holo-Code-Gen: HLS toolchain for photonic neural networks."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@yourcompany.com"

# Core API exports - import lazily to avoid dependency issues
def __getattr__(name):
    """Lazy import to avoid dependency issues during development."""
    if name == "PhotonicCompiler":
        from .compiler import PhotonicCompiler
        return PhotonicCompiler
    elif name == "SpikingPhotonicCompiler":
        from .compiler import SpikingPhotonicCompiler
        return SpikingPhotonicCompiler
    elif name == "IMECLibrary":
        from .templates import IMECLibrary
        return IMECLibrary
    elif name == "PhotonicComponent":
        from .templates import PhotonicComponent
        return PhotonicComponent
    elif name == "PowerOptimizer":
        from .optimization import PowerOptimizer
        return PowerOptimizer
    elif name == "YieldOptimizer":
        from .optimization import YieldOptimizer
        return YieldOptimizer
    elif name == "QuantumInspiredTaskPlanner":
        from .optimization import QuantumInspiredTaskPlanner
        return QuantumInspiredTaskPlanner
    elif name == "PhotonicQuantumAlgorithms":
        from .quantum_algorithms import PhotonicQuantumAlgorithms
        return PhotonicQuantumAlgorithms
    elif name == "PhotonicQuantumOptimizer":
        from .optimization import PhotonicQuantumOptimizer
        return PhotonicQuantumOptimizer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "__version__",
    "PhotonicCompiler",
    "SpikingPhotonicCompiler", 
    "IMECLibrary",
    "PhotonicComponent",
    "PowerOptimizer",
    "YieldOptimizer",
    "QuantumInspiredTaskPlanner",
    "PhotonicQuantumAlgorithms",
    "PhotonicQuantumOptimizer",
]