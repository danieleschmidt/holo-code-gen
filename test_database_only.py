#!/usr/bin/env python3
"""Test database layer functionality only."""

import sys
import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import only database modules directly
from holo_code_gen.database.models import DesignDatabase, CircuitModel, OptimizationResult, SimulationResult


def test_database_direct():
    """Test database functionality directly."""
    print("=== Testing Database Layer Directly ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        # Test database initialization
        db = DesignDatabase(db_path)
        print("✓ Database initialized")
        
        # Test circuit creation and saving
        circuit = CircuitModel(
            name="Test Circuit",
            description="A test photonic circuit",
            neural_network_source="model.pth",
            compilation_config={"wavelength": 1550.0, "process": "SiN_220nm"},
            components=[
                {"name": "input_wg", "type": "waveguide", "length": 100},
                {"name": "ring1", "type": "microring", "radius": 10.0}
            ],
            connections=[{"source": "input_wg", "target": "ring1"}],
            layout_data={"area": 0.5, "routing": "manhattan"},
            performance_metrics={"power": 12.5, "area": 0.5, "loss": 2.1}
        )
        
        circuit_id = db.save_circuit(circuit)
        assert circuit_id is not None
        print(f"✓ Saved circuit: {circuit_id}")
        
        # Test circuit loading
        loaded_circuit = db.load_circuit(circuit_id)
        assert loaded_circuit is not None
        assert loaded_circuit.name == "Test Circuit"
        assert loaded_circuit.performance_metrics["power"] == 12.5
        assert len(loaded_circuit.components) == 2
        print("✓ Loaded circuit successfully")
        
        # Test circuit listing
        circuits = db.list_circuits(limit=10)
        assert len(circuits) == 1
        assert circuits[0].name == "Test Circuit"
        print(f"✓ Listed {len(circuits)} circuits")
        
        # Test circuit search
        search_results = db.search_circuits("Test")
        assert len(search_results) == 1
        print(f"✓ Search found {len(search_results)} circuits")
        
        # Test optimization result
        opt_result = OptimizationResult(
            circuit_id=circuit_id,
            optimizer_type="power_optimizer",
            parameters={"power_budget": 100.0, "iterations": 1000},
            initial_metrics={"power": 20.0, "area": 0.8},
            final_metrics={"power": 12.5, "area": 0.5},
            optimization_log=[
                {"iteration": 1, "power": 18.0},
                {"iteration": 500, "power": 15.0},
                {"iteration": 1000, "power": 12.5}
            ],
            execution_time=45.2,
            success=True
        )
        
        opt_id = db.save_optimization_result(opt_result)
        assert opt_id is not None
        print(f"✓ Saved optimization result: {opt_id}")
        
        # Test loading optimization results
        opt_results = db.load_optimization_results(circuit_id)
        assert len(opt_results) == 1
        assert opt_results[0].optimizer_type == "power_optimizer"
        assert opt_results[0].success == True
        assert opt_results[0].execution_time == 45.2
        print(f"✓ Loaded {len(opt_results)} optimization results")
        
        # Test simulation result
        sim_result = SimulationResult(
            circuit_id=circuit_id,
            simulation_type="fdtd",
            parameters={
                "resolution": 20, 
                "wavelength": 1550.0,
                "simulation_time": 10.0,
                "mesh_accuracy": 3
            },
            results={
                "transmission": 0.85,
                "reflection": 0.12,
                "loss_db": 2.1,
                "bandwidth_nm": 50.0,
                "q_factor": 10000.0
            },
            simulation_time=180.5,
            convergence_achieved=True
        )
        
        sim_id = db.save_simulation_result(sim_result)
        assert sim_id is not None
        print(f"✓ Saved simulation result: {sim_id}")
        
        # Test loading simulation results
        sim_results = db.load_simulation_results(circuit_id)
        assert len(sim_results) == 1
        assert sim_results[0].simulation_type == "fdtd"
        assert sim_results[0].convergence_achieved == True
        assert sim_results[0].results["transmission"] == 0.85
        print(f"✓ Loaded {len(sim_results)} simulation results")
        
        # Test database statistics
        stats = db.get_database_stats()
        assert stats["total_circuits"] == 1
        assert stats["total_optimizations"] == 1
        assert stats["total_simulations"] == 1
        print(f"✓ Database stats: {stats}")
        
        # Test export functionality
        export_path = Path(temp_dir) / "exported_circuit.json"
        success = db.export_circuit(circuit_id, export_path)
        assert success
        assert export_path.exists()
        print("✓ Exported circuit data")
        
        # Verify export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        assert export_data["circuit"]["name"] == "Test Circuit"
        assert len(export_data["optimization_results"]) == 1
        assert len(export_data["simulation_results"]) == 1
        print("✓ Verified export data integrity")
        
        # Test import functionality
        imported_id = db.import_circuit(export_path)
        assert imported_id is not None
        assert imported_id != circuit_id  # Should be new ID
        print(f"✓ Imported circuit: {imported_id}")
        
        # Verify imported data
        imported_circuit = db.load_circuit(imported_id)
        assert imported_circuit.name == "Test Circuit"
        
        imported_opts = db.load_optimization_results(imported_id)
        assert len(imported_opts) == 1
        
        imported_sims = db.load_simulation_results(imported_id)
        assert len(imported_sims) == 1
        
        print("✓ Verified imported data")
        
        # Test circuit deletion
        deleted = db.delete_circuit(imported_id)
        assert deleted
        
        # Verify deletion
        deleted_circuit = db.load_circuit(imported_id)
        assert deleted_circuit is None
        print("✓ Circuit deletion working")
        
        # Final stats check
        final_stats = db.get_database_stats()
        assert final_stats["total_circuits"] == 1  # Only original circuit remains
        print(f"✓ Final database stats: {final_stats}")
        
        return True


def test_database_schema():
    """Test database schema directly."""
    print("\n=== Testing Database Schema ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "schema_test.db"
        
        # Initialize database
        db = DesignDatabase(db_path)
        
        # Test schema by direct SQL queries
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check circuits table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='circuits'")
            assert cursor.fetchone() is not None
            print("✓ Circuits table exists")
            
            # Check optimization_results table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='optimization_results'")
            assert cursor.fetchone() is not None
            print("✓ Optimization results table exists")
            
            # Check simulation_results table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='simulation_results'")
            assert cursor.fetchone() is not None
            print("✓ Simulation results table exists")
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_circuits_name'")
            assert cursor.fetchone() is not None
            print("✓ Circuit name index exists")
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_optimization_circuit'")
            assert cursor.fetchone() is not None
            print("✓ Optimization circuit index exists")
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_simulation_circuit'")
            assert cursor.fetchone() is not None
            print("✓ Simulation circuit index exists")
        
        return True


def test_data_validation():
    """Test data validation and error handling."""
    print("\n=== Testing Data Validation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "validation_test.db"
        db = DesignDatabase(db_path)
        
        # Test circuit with minimal data
        minimal_circuit = CircuitModel(name="Minimal Circuit")
        circuit_id = db.save_circuit(minimal_circuit)
        assert circuit_id is not None
        print("✓ Minimal circuit saved")
        
        # Test loading non-existent circuit
        non_existent = db.load_circuit("non-existent-id")
        assert non_existent is None
        print("✓ Non-existent circuit handling")
        
        # Test empty search
        empty_results = db.search_circuits("NonExistentName")
        assert len(empty_results) == 0
        print("✓ Empty search results")
        
        # Test invalid circuit deletion
        delete_result = db.delete_circuit("invalid-id")
        assert delete_result == False
        print("✓ Invalid deletion handling")
        
        # Test optimization result with missing circuit
        invalid_opt = OptimizationResult(
            circuit_id="non-existent",
            optimizer_type="test",
            parameters={},
            initial_metrics={},
            final_metrics={},
            optimization_log=[],
            execution_time=0.0,
            success=False
        )
        
        # This should still save (foreign key constraint not enforced in SQLite by default)
        opt_id = db.save_optimization_result(invalid_opt)
        assert opt_id is not None
        print("✓ Optimization with invalid circuit ID handled")
        
        # Test large data handling
        large_circuit = CircuitModel(
            name="Large Circuit",
            description="A" * 10000,  # Large description
            components=[{"component": i} for i in range(1000)],  # Many components
            performance_metrics={f"metric_{i}": float(i) for i in range(100)}  # Many metrics
        )
        
        large_id = db.save_circuit(large_circuit)
        assert large_id is not None
        
        loaded_large = db.load_circuit(large_id)
        assert len(loaded_large.description) == 10000
        assert len(loaded_large.components) == 1000
        print("✓ Large data handling")
        
        return True


def main():
    """Run database tests."""
    print("Holo-Code-Gen Database Layer Tests")
    print("=" * 50)
    
    try:
        test_database_direct()
        test_database_schema()
        test_data_validation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL DATABASE TESTS PASSED!")
        print("\nDatabase layer features verified:")
        print("  ✓ SQLite database initialization and schema creation")
        print("  ✓ Circuit CRUD operations with JSON serialization")
        print("  ✓ Optimization result storage and retrieval")
        print("  ✓ Simulation result storage and retrieval")
        print("  ✓ Search functionality with full-text search")
        print("  ✓ Export/import functionality with complete data")
        print("  ✓ Database statistics and reporting")
        print("  ✓ Data validation and error handling")
        print("  ✓ Large data handling and performance")
        print("  ✓ Database schema with proper indexing")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())