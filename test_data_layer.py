#!/usr/bin/env python3
"""Test comprehensive data layer functionality."""

import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Mock dependencies for testing
class MockTorch:
    class nn:
        class Module:
            def named_modules(self):
                return [("fc1", MockModule(784, 128)), ("fc2", MockModule(128, 10))]
            def parameters(self):
                return [MockTensor([128, 784]), MockTensor([10, 128])]
        
        class Linear(Module):
            def __init__(self, in_features, out_features):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = MockTensor([out_features, in_features])
    
    @staticmethod
    def save(model, path):
        with open(path, 'w') as f:
            f.write('mock_model_data')
    
    @staticmethod
    def load(path, map_location=None):
        return MockModule(128, 10)
    
    class onnx:
        @staticmethod
        def export(model, dummy_input, path, **kwargs):
            with open(path, 'w') as f:
                f.write('mock_onnx_data')
    
    __version__ = "2.0.0"

class MockTensor:
    def __init__(self, shape):
        self.shape = shape
    
    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    @property
    def requires_grad(self):
        return True

class MockModule:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = MockTensor([out_features, in_features])
        self.bias = MockTensor([out_features])
    
    def parameters(self):
        return [self.weight, self.bias]
    
    def named_modules(self):
        return [("linear", self)]
    
    def children(self):
        return []

# Patch imports
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MockTorch.nn
sys.modules['torch.onnx'] = MockTorch.onnx

# Now import our modules
from holo_code_gen.database.models import DesignDatabase, CircuitModel, OptimizationResult, SimulationResult
from holo_code_gen.database.persistence import ModelPersistence, CircuitPersistence
from holo_code_gen.database.cache import CacheManager, ComponentCache, SimulationCache
from holo_code_gen.repositories import RepositoryManager, CircuitRepository, OptimizationRepository, SimulationRepository, ModelRepository
from holo_code_gen.config import HoloCodeGenConfig, get_config
from holo_code_gen.circuit import PhotonicCircuit
from holo_code_gen.templates import IMECLibrary, PhotonicComponent, ComponentSpec


def test_database_models():
    """Test database models and operations."""
    print("=== Testing Database Models ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        db = DesignDatabase(db_path)
        
        # Test circuit model
        circuit = CircuitModel(
            name="Test Circuit",
            description="Test circuit for database",
            neural_network_source="test_model.pth",
            compilation_config={"wavelength": 1550.0, "process": "SiN_220nm"},
            components=[{"name": "test_component", "type": "waveguide"}],
            connections=[{"source": "input", "target": "output"}],
            performance_metrics={"power": 10.5, "area": 2.3}
        )
        
        # Save circuit
        circuit_id = db.save_circuit(circuit)
        print(f"Saved circuit: {circuit_id}")
        
        # Load circuit
        loaded_circuit = db.load_circuit(circuit_id)
        assert loaded_circuit is not None
        assert loaded_circuit.name == "Test Circuit"
        assert loaded_circuit.performance_metrics["power"] == 10.5
        print(f"Loaded circuit: {loaded_circuit.name}")
        
        # Test optimization result
        opt_result = OptimizationResult(
            circuit_id=circuit_id,
            optimizer_type="power",
            parameters={"budget": 100.0},
            initial_metrics={"power": 15.0},
            final_metrics={"power": 10.5},
            optimization_log=[{"iteration": 1, "power": 12.0}],
            execution_time=5.2,
            success=True
        )
        
        opt_id = db.save_optimization_result(opt_result)
        print(f"Saved optimization result: {opt_id}")
        
        # Load optimization results
        opt_results = db.load_optimization_results(circuit_id)
        assert len(opt_results) == 1
        assert opt_results[0].optimizer_type == "power"
        print(f"Loaded {len(opt_results)} optimization results")
        
        # Test simulation result
        sim_result = SimulationResult(
            circuit_id=circuit_id,
            simulation_type="fdtd",
            parameters={"resolution": 20, "wavelength": 1550.0},
            results={"transmission": 0.85, "reflection": 0.12},
            simulation_time=120.5,
            convergence_achieved=True
        )
        
        sim_id = db.save_simulation_result(sim_result)
        print(f"Saved simulation result: {sim_id}")
        
        # Load simulation results
        sim_results = db.load_simulation_results(circuit_id)
        assert len(sim_results) == 1
        assert sim_results[0].simulation_type == "fdtd"
        print(f"Loaded {len(sim_results)} simulation results")
        
        # Test search
        search_results = db.search_circuits("Test")
        assert len(search_results) == 1
        print(f"Search found {len(search_results)} circuits")
        
        # Test database stats
        stats = db.get_database_stats()
        print(f"Database stats: {stats}")
        assert stats['total_circuits'] == 1
        
        print("✓ Database models tests passed")


def test_persistence_layer():
    """Test persistence layer functionality."""
    print("\n=== Testing Persistence Layer ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test model persistence
        model_persistence = ModelPersistence(Path(temp_dir) / "models")
        
        # Create mock model
        mock_model = MockModule(784, 10)
        
        # Save PyTorch model
        model_path = model_persistence.save_pytorch_model(
            mock_model, 
            "test_model",
            metadata={"description": "Test neural network"}
        )
        print(f"Saved PyTorch model: {model_path}")
        
        # Load PyTorch model
        loaded_model = model_persistence.load_pytorch_model(model_path, MockModule(784, 10))
        assert loaded_model is not None
        print("Loaded PyTorch model successfully")
        
        # List models
        models = model_persistence.list_models()
        assert len(models) == 1
        assert models[0]['model_name'] == "test_model"
        print(f"Listed {len(models)} models")
        
        # Test circuit persistence
        circuit_persistence = CircuitPersistence(Path(temp_dir) / "circuits")
        
        # Create mock circuit
        circuit = PhotonicCircuit()
        library = IMECLibrary("imec_basic")
        
        # Add some components
        spec = ComponentSpec(
            name="test_component",
            component_type="waveguide",
            parameters={"length": 100.0, "width": 0.5}
        )
        component = PhotonicComponent(spec=spec, instance_id="wg1")
        circuit.components.append(component)
        circuit.connections.append(("input", "output"))
        
        # Save circuit
        circuit_id = circuit_persistence.save_circuit(
            circuit,
            "test_circuit",
            metadata={"description": "Test photonic circuit"}
        )
        print(f"Saved circuit: {circuit_id}")
        
        # Load circuit
        loaded_circuit = circuit_persistence.load_circuit(circuit_id)
        assert loaded_circuit is not None
        assert len(loaded_circuit.components) == 1
        print("Loaded circuit successfully")
        
        # List circuits
        circuits = circuit_persistence.list_circuits()
        assert len(circuits) == 1
        print(f"Listed {len(circuits)} circuits")
        
        print("✓ Persistence layer tests passed")


def test_caching_system():
    """Test caching system functionality."""
    print("\n=== Testing Caching System ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test generic cache manager
        cache = CacheManager(
            cache_dir=Path(temp_dir) / "cache",
            max_size_mb=10,
            default_ttl=60
        )
        
        # Test cache operations
        test_data = {"key": "value", "number": 42}
        cache.put("test_key", test_data)
        
        retrieved_data = cache.get("test_key")
        assert retrieved_data is not None
        assert retrieved_data["key"] == "value"
        print("Generic cache put/get working")
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['total_entries'] == 1
        print(f"Cache stats: {stats}")
        
        # Test component cache
        component_cache = ComponentCache(Path(temp_dir) / "component_cache")
        
        # Create mock component
        spec = ComponentSpec(
            name="test_ring",
            component_type="microring_resonator",
            parameters={"radius": 10.0, "gap": 0.2}
        )
        component = PhotonicComponent(spec=spec)
        
        # Cache component
        cache_key = component_cache.cache_component(component)
        assert cache_key != ""
        print(f"Cached component with key: {cache_key}")
        
        # Retrieve component
        cached_data = component_cache.get_component(cache_key)
        assert cached_data is not None
        assert cached_data['spec']['name'] == "test_ring"
        print("Component cache working")
        
        # Test simulation cache
        simulation_cache = SimulationCache(Path(temp_dir) / "simulation_cache")
        
        # Create mock circuit and simulation
        circuit = PhotonicCircuit()
        sim_params = {"wavelength": 1550.0, "resolution": 20}
        sim_results = {"transmission": 0.85, "loss": 1.2}
        
        # Cache simulation
        cache_key = simulation_cache.cache_simulation(circuit, sim_params, sim_results)
        assert cache_key != ""
        print(f"Cached simulation with key: {cache_key}")
        
        # Retrieve simulation
        cached_results = simulation_cache.get_simulation(circuit, sim_params)
        assert cached_results is not None
        assert cached_results["transmission"] == 0.85
        print("Simulation cache working")
        
        print("✓ Caching system tests passed")


def test_repositories():
    """Test repository layer functionality."""
    print("\n=== Testing Repository Layer ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize repository manager
        repo_manager = RepositoryManager(Path(temp_dir) / "test.db")
        
        # Test circuit repository
        circuit_id = repo_manager.circuits.create(
            name="Test Repository Circuit",
            description="Circuit created via repository",
            neural_network_source="test_model.pth",
            compilation_config={"wavelength": 1550.0}
        )
        print(f"Created circuit via repository: {circuit_id}")
        
        # Get circuit
        circuit = repo_manager.circuits.get_by_id(circuit_id)
        assert circuit is not None
        assert circuit.name == "Test Repository Circuit"
        print("Retrieved circuit via repository")
        
        # Update circuit
        success = repo_manager.circuits.update(circuit_id, description="Updated description")
        assert success
        print("Updated circuit via repository")
        
        # Search circuits
        results = repo_manager.circuits.search("Repository")
        assert len(results) == 1
        print(f"Search found {len(results)} circuits")
        
        # Test optimization repository
        opt_id = repo_manager.optimizations.create(
            circuit_id=circuit_id,
            optimizer_type="power",
            parameters={"budget": 100.0},
            initial_metrics={"power": 20.0},
            final_metrics={"power": 15.0},
            optimization_log=[{"iteration": 1, "power": 18.0}],
            execution_time=3.5,
            success=True
        )
        print(f"Created optimization result: {opt_id}")
        
        # Get optimization results for circuit
        opt_results = repo_manager.optimizations.get_by_circuit(circuit_id)
        assert len(opt_results) == 1
        print(f"Retrieved {len(opt_results)} optimization results")
        
        # Get optimization stats
        opt_stats = repo_manager.optimizations.get_optimization_stats(circuit_id)
        assert opt_stats['total_optimizations'] == 1
        assert opt_stats['success_rate'] == 100.0
        print(f"Optimization stats: {opt_stats}")
        
        # Test simulation repository
        sim_id = repo_manager.simulations.create(
            circuit_id=circuit_id,
            simulation_type="fdtd",
            parameters={"resolution": 20},
            results={"transmission": 0.9},
            simulation_time=60.0,
            convergence_achieved=True
        )
        print(f"Created simulation result: {sim_id}")
        
        # Get simulation results
        sim_results = repo_manager.simulations.get_by_circuit(circuit_id)
        assert len(sim_results) == 1
        print(f"Retrieved {len(sim_results)} simulation results")
        
        # Test model repository
        mock_model = MockModule(784, 10)
        model_path = repo_manager.models.save_pytorch_model(
            mock_model,
            "repository_test_model"
        )
        print(f"Saved model via repository: {model_path}")
        
        # List models
        models = repo_manager.models.list_models()
        assert len(models) == 1
        print(f"Listed {len(models)} models")
        
        # Test repository manager stats
        stats = repo_manager.get_database_stats()
        assert stats['database']['total_circuits'] == 1
        print(f"Repository manager stats: {stats}")
        
        # Test health check
        health = repo_manager.health_check()
        assert health['overall'] == True
        print(f"Health check: {health}")
        
        print("✓ Repository layer tests passed")


def test_configuration_system():
    """Test configuration management system."""
    print("\n=== Testing Configuration System ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test default configuration
        config = HoloCodeGenConfig()
        assert config.database.max_cache_size_mb == 1000
        assert config.simulation.default_wavelength == 1550.0
        print("Default configuration created")
        
        # Test configuration from environment
        import os
        os.environ['HOLO_MAX_CACHE_SIZE_MB'] = '2000'
        os.environ['HOLO_DEFAULT_WAVELENGTH'] = '1310.0'
        os.environ['HOLO_CODE_GEN_DEBUG'] = 'true'
        
        env_config = HoloCodeGenConfig.from_env()
        assert env_config.database.max_cache_size_mb == 2000
        assert env_config.simulation.default_wavelength == 1310.0
        assert env_config.debug == True
        print("Configuration loaded from environment")
        
        # Test configuration file save/load
        config_file = Path(temp_dir) / "config.json"
        config.to_file(config_file)
        
        loaded_config = HoloCodeGenConfig.from_file(config_file)
        assert loaded_config.database.max_cache_size_mb == config.database.max_cache_size_mb
        print("Configuration saved and loaded from file")
        
        # Test path expansion
        config.database.path = "~/test_db.db"
        config.expand_paths()
        assert not config.database.path.startswith("~")
        print("Path expansion working")
        
        # Test configuration validation
        config.simulation.threads = -1  # Invalid
        issues = config.validate()
        assert len(issues) > 0
        print(f"Configuration validation found {len(issues)} issues")
        
        # Test configuration summary
        summary = config.get_summary()
        assert 'database_path' in summary
        assert 'simulation_backend' in summary
        print(f"Configuration summary: {summary}")
        
        print("✓ Configuration system tests passed")


def test_integration():
    """Test integration between all data layer components."""
    print("\n=== Testing Data Layer Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up configuration
        config = HoloCodeGenConfig()
        config.database.path = str(Path(temp_dir) / "integration.db")
        config.database.cache_dir = str(Path(temp_dir) / "cache")
        config.database.models_dir = str(Path(temp_dir) / "models")
        config.database.circuits_dir = str(Path(temp_dir) / "circuits")
        config.expand_paths()
        config.create_directories()
        
        # Initialize repository manager with configuration
        repo_manager = RepositoryManager(Path(config.database.path))
        
        # Create and save a neural network model
        mock_model = MockModule(784, 128)
        model_path = repo_manager.models.save_pytorch_model(
            mock_model,
            "integration_test_model",
            metadata={"created_by": "integration_test"}
        )
        
        # Create circuit with model reference
        circuit_id = repo_manager.circuits.create(
            name="Integration Test Circuit",
            description="Full integration test circuit",
            neural_network_source=f"pytorch:{model_path}",
            compilation_config={
                "wavelength": config.simulation.default_wavelength,
                "process": config.simulation.default_process,
                "power_budget": config.optimization.default_power_budget_mw
            }
        )
        
        # Create and save a PhotonicCircuit instance
        circuit = PhotonicCircuit()
        library = IMECLibrary("imec_v2025_07")
        
        # Add components
        for i, comp_type in enumerate(["waveguide", "microring_weight_bank", "ring_modulator"]):
            component = library.get_component(comp_type)
            component.instance_id = f"comp_{i}"
            circuit.components.append(component)
        
        circuit.connections = [("comp_0", "comp_1"), ("comp_1", "comp_2")]
        circuit.generate_layout()
        circuit.calculate_metrics()
        
        # Save circuit instance
        success = repo_manager.circuits.save_circuit_instance(
            circuit, 
            circuit_id,
            metadata={"integration_test": True}
        )
        assert success
        print("Saved complete circuit instance")
        
        # Load circuit instance
        loaded_circuit = repo_manager.circuits.load_circuit_instance(circuit_id)
        assert loaded_circuit is not None
        assert len(loaded_circuit.components) == 3
        print("Loaded complete circuit instance")
        
        # Run optimization and save results
        opt_id = repo_manager.optimizations.create(
            circuit_id=circuit_id,
            optimizer_type="power",
            parameters={"budget": config.optimization.default_power_budget_mw},
            initial_metrics={"power": 25.0, "area": 5.0},
            final_metrics={"power": 18.0, "area": 5.0},
            optimization_log=[
                {"iteration": 1, "power": 22.0},
                {"iteration": 2, "power": 20.0},
                {"iteration": 3, "power": 18.0}
            ],
            execution_time=8.5,
            success=True
        )
        
        # Cache optimization result
        cache_key = repo_manager.optimizations.cache_optimization_result(
            circuit, "power", {"budget": 100.0}, {"final_power": 18.0}
        )
        assert cache_key != ""
        print("Cached optimization result")
        
        # Run simulation and save results
        sim_id = repo_manager.simulations.create(
            circuit_id=circuit_id,
            simulation_type="fdtd",
            parameters={
                "resolution": config.simulation.resolution,
                "wavelength": config.simulation.default_wavelength,
                "convergence_threshold": config.simulation.convergence_threshold
            },
            results={
                "transmission": 0.87,
                "reflection": 0.10,
                "loss": 1.3,
                "bandwidth": 50.0
            },
            simulation_time=180.0,
            convergence_achieved=True
        )
        
        # Cache simulation result
        sim_cache_key = repo_manager.simulations.cache_simulation_result(
            circuit,
            {"wavelength": 1550.0, "resolution": 20},
            {"transmission": 0.87}
        )
        assert sim_cache_key != ""
        print("Cached simulation result")
        
        # Export complete circuit data
        export_path = Path(temp_dir) / "exported_circuit.json"
        success = repo_manager.circuits.export_circuit(circuit_id, export_path)
        assert success
        print("Exported complete circuit data")
        
        # Import circuit data (create new circuit)
        imported_circuit_id = repo_manager.circuits.import_circuit(export_path)
        assert imported_circuit_id is not None
        print("Imported circuit data")
        
        # Verify imported data
        imported_circuit = repo_manager.circuits.get_by_id(imported_circuit_id)
        assert imported_circuit is not None
        assert imported_circuit.name == "Integration Test Circuit"
        
        imported_opt_results = repo_manager.optimizations.get_by_circuit(imported_circuit_id)
        assert len(imported_opt_results) == 1
        
        imported_sim_results = repo_manager.simulations.get_by_circuit(imported_circuit_id)
        assert len(imported_sim_results) == 1
        
        print("Verified imported data integrity")
        
        # Test comprehensive stats
        final_stats = repo_manager.get_database_stats()
        assert final_stats['database']['total_circuits'] == 2  # Original + imported
        print(f"Final database stats: {final_stats}")
        
        print("✓ Data layer integration tests passed")


def main():
    """Run all data layer tests."""
    print("Holo-Code-Gen Data Layer Tests")
    print("=" * 50)
    
    try:
        test_database_models()
        test_persistence_layer()
        test_caching_system()
        test_repositories()
        test_configuration_system()
        test_integration()
        
        print("\n" + "=" * 50)
        print("🎉 ALL DATA LAYER TESTS PASSED!")
        print("\nData layer components tested:")
        print("  ✓ Database models with SQLite persistence")
        print("  ✓ Model and circuit persistence layers")
        print("  ✓ Multi-level caching system")
        print("  ✓ Repository pattern with CRUD operations")
        print("  ✓ Configuration management system")
        print("  ✓ Full integration with export/import")
        print("\nThe data layer provides:")
        print("  • Complete persistence for all photonic designs")
        print("  • High-performance caching for optimization results")
        print("  • Flexible configuration management")
        print("  • Repository abstraction for clean data access")
        print("  • Import/export capabilities for data exchange")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())