"""Database and data management components for Holo-Code-Gen."""

# Lazy imports to avoid dependency issues during development
def __getattr__(name):
    """Lazy import to avoid dependency issues."""
    if name == "CacheManager":
        from .cache import CacheManager
        return CacheManager
    elif name == "ComponentCache":
        from .cache import ComponentCache
        return ComponentCache
    elif name == "DesignDatabase":
        from .models import DesignDatabase
        return DesignDatabase
    elif name == "CircuitModel":
        from .models import CircuitModel
        return CircuitModel
    elif name == "OptimizationResult":
        from .models import OptimizationResult
        return OptimizationResult
    elif name == "SimulationResult":
        from .models import SimulationResult
        return SimulationResult
    elif name == "ModelPersistence":
        from .persistence import ModelPersistence
        return ModelPersistence
    elif name == "CircuitPersistence":
        from .persistence import CircuitPersistence
        return CircuitPersistence
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "CacheManager",
    "ComponentCache", 
    "DesignDatabase",
    "CircuitModel",
    "OptimizationResult",
    "SimulationResult",
    "ModelPersistence",
    "CircuitPersistence"
]