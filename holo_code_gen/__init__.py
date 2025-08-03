"""Holo-Code-Gen: HLS toolchain for photonic neural networks."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@yourcompany.com"

# Core API exports
from .compiler import PhotonicCompiler, SpikingPhotonicCompiler
from .templates import IMECLibrary, PhotonicComponent
from .optimization import PowerOptimizer, YieldOptimizer

__all__ = [
    "__version__",
    "PhotonicCompiler",
    "SpikingPhotonicCompiler",
    "IMECLibrary",
    "PhotonicComponent",
    "PowerOptimizer",
    "YieldOptimizer",
]