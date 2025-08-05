#!/usr/bin/env python3
"""
Comprehensive test suite for Quantum-Inspired Task Planner

Tests all Generation 2 robustness and reliability features including
error handling, validation, security, and monitoring.
"""

import pytest
import json
from typing import Dict, Any

# Test imports
from holo_code_gen.optimization import QuantumInspiredTaskPlanner
from holo_code_gen.exceptions import ValidationError, CompilationError, SecurityError
from holo_code_gen.security import initialize_security


def setup_module():
    """Setup security components for testing."""
    initialize_security()


class TestQuantumTaskPlannerRobustness:
    """Test quantum task planner robustness and error handling."""
    
    def setup_method(self):
        """Setup test environment."""
        self.planner = QuantumInspiredTaskPlanner(
            coherence_time=1000.0,
            entanglement_fidelity=0.95
        )
    
    def test_initialization_validation(self):
        """Test initialization parameter validation."""
        # Test negative coherence time
        with pytest.raises(ValidationError) as exc_info:
            QuantumInspiredTaskPlanner(coherence_time=-100.0)
        assert "coherence_time" in str(exc_info.value)
        
        # Test invalid entanglement fidelity
        with pytest.raises(ValidationError) as exc_info:
            QuantumInspiredTaskPlanner(entanglement_fidelity=1.5)
        assert "entanglement_fidelity" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            QuantumInspiredTaskPlanner(entanglement_fidelity=-0.1)
        assert "entanglement_fidelity" in str(exc_info.value)
    
    def test_valid_quantum_algorithm_planning(self):
        """Test planning with valid quantum algorithm."""
        valid_algorithm = {
            "qubits": 3,
            "operations": [
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "Phase", "qubits": [2]}
            ],
            "measurements": [
                {"qubit": 0, "basis": "computational"},
                {"qubit": 1, "basis": "X"}
            ]
        }
        
        plan = self.planner.plan_quantum_circuit(valid_algorithm)
        
        # Verify plan structure
        assert "qubits" in plan
        assert "photonic_qubits" in plan
        assert "gate_sequence" in plan
        assert "entanglement_scheme" in plan
        assert "measurement_scheme" in plan
        assert "coherence_optimization" in plan
        assert "planning_metadata" in plan
        
        # Verify consistency
        assert plan["qubits"] == 3
        assert len(plan["photonic_qubits"]) == 3
        assert len(plan["gate_sequence"]) == 3
    
    def test_invalid_algorithm_input_validation(self):
        """Test input validation for invalid algorithms."""
        # Test non-dictionary input
        with pytest.raises(ValidationError) as exc_info:
            self.planner.plan_quantum_circuit("not a dict")
        assert "dictionary" in str(exc_info.value)
        
        # Test missing operations field
        with pytest.raises(ValidationError) as exc_info:
            self.planner.plan_quantum_circuit({"qubits": 2})
        assert "operations" in str(exc_info.value)
        
        # Test invalid operations type
        with pytest.raises(ValidationError) as exc_info:
            self.planner.plan_quantum_circuit({
                "qubits": 2,
                "operations": "not a list"
            })
        assert "list" in str(exc_info.value)
        
        # Test invalid qubit count
        with pytest.raises(ValidationError) as exc_info:
            self.planner.plan_quantum_circuit({
                "qubits": 0,
                "operations": []
            })
        assert "positive" in str(exc_info.value)
        
        # Test too many qubits
        with pytest.raises(ValidationError) as exc_info:
            self.planner.plan_quantum_circuit({
                "qubits": 100,  # Exceeds limit of 50
                "operations": []
            })
        assert "Too many qubits" in str(exc_info.value)
    
    def test_invalid_operation_validation(self):
        """Test validation of individual operations."""
        # Test non-dictionary operation
        with pytest.raises(ValidationError) as exc_info:
            self.planner.plan_quantum_circuit({
                "qubits": 2,
                "operations": ["not a dict"]
            })
        assert "dictionary" in str(exc_info.value)
        
        # Test missing gate field
        with pytest.raises(ValidationError) as exc_info:
            self.planner.plan_quantum_circuit({
                "qubits": 2,
                "operations": [{"qubits": [0]}]  # Missing gate
            })
        assert "gate" in str(exc_info.value)
    
    def test_resource_limit_enforcement(self):
        """Test resource limit enforcement."""
        # Test circuit complexity limits
        large_algorithm = {
            "qubits": 20,
            "operations": [{"gate": "Hadamard", "qubits": [i]} for i in range(100)]
        }
        
        # Should not raise error for reasonable circuit
        plan = self.planner.plan_quantum_circuit(large_algorithm)
        assert plan is not None
        
        # Test extremely large circuit (would be caught by security limiter)
        huge_algorithm = {
            "qubits": 60,  # Exceeds limit
            "operations": [{"gate": "Hadamard", "qubits": [0]}]
        }
        
        with pytest.raises(ValidationError):
            self.planner.plan_quantum_circuit(huge_algorithm)
    
    def test_photonic_plan_validation(self):
        """Test validation of generated photonic plans."""
        algorithm = {
            "qubits": 2,
            "operations": [
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]}
            ]
        }
        
        plan = self.planner.plan_quantum_circuit(algorithm)
        
        # Test that plan passes validation
        self.planner._validate_photonic_plan(plan)  # Should not raise
        
        # Test invalid plan validation
        invalid_plan = plan.copy()
        del invalid_plan["qubits"]  # Remove required field
        
        with pytest.raises(ValidationError) as exc_info:
            self.planner._validate_photonic_plan(invalid_plan)
        assert "missing required field" in str(exc_info.value)
        
        # Test qubit count mismatch
        mismatch_plan = plan.copy()
        mismatch_plan["qubits"] = 5  # Doesn't match photonic_qubits
        
        with pytest.raises(ValidationError) as exc_info:
            self.planner._validate_photonic_plan(mismatch_plan)
        assert "mismatch" in str(exc_info.value)
    
    def test_fidelity_calculation(self):
        """Test fidelity calculation robustness."""
        # Test valid gate sequence
        gate_sequence = [
            {"fidelity_estimate": 0.99},
            {"fidelity_estimate": 0.95},
            {"fidelity_estimate": 0.98}
        ]
        
        fidelity = self.planner._calculate_total_fidelity(gate_sequence)
        expected = 0.99 * 0.95 * 0.98
        assert abs(fidelity - expected) < 1e-10
        
        # Test with invalid fidelity values
        invalid_sequence = [
            {"fidelity_estimate": 1.5},  # Invalid > 1
            {"fidelity_estimate": 0.95}
        ]
        
        fidelity = self.planner._calculate_total_fidelity(invalid_sequence)
        # Should ignore invalid values
        assert fidelity == 0.95
        
        # Test empty sequence
        empty_fidelity = self.planner._calculate_total_fidelity([])
        assert empty_fidelity == 1.0
    
    def test_resource_usage_estimation(self):
        """Test resource usage estimation."""
        usage = self.planner._estimate_resource_usage(4, 10)
        
        # Verify all fields present
        assert "photonic_components" in usage
        assert "estimated_area_mm2" in usage
        assert "estimated_power_mw" in usage
        assert "complexity_score" in usage
        assert "fabrication_layers" in usage
        assert "wavelength_channels" in usage
        
        # Verify reasonable values
        assert usage["photonic_components"] > 0
        assert usage["estimated_area_mm2"] > 0
        assert usage["estimated_power_mw"] > 0
        assert 0 <= usage["complexity_score"] <= 10
    
    def test_planning_statistics(self):
        """Test planning statistics tracking."""
        initial_stats = self.planner.get_planning_statistics()
        assert initial_stats["circuits_planned"] == 0
        assert initial_stats["errors_detected"] == 0
        
        # Plan a circuit
        algorithm = {
            "qubits": 2,
            "operations": [{"gate": "Hadamard", "qubits": [0]}]
        }
        
        self.planner.plan_quantum_circuit(algorithm)
        
        updated_stats = self.planner.get_planning_statistics()
        assert updated_stats["circuits_planned"] == 1
        assert updated_stats["average_fidelity"] > 0
        
        # Test error tracking
        try:
            self.planner.plan_quantum_circuit({"invalid": "algorithm"})
        except ValidationError:
            pass
        
        error_stats = self.planner.get_planning_statistics()
        assert error_stats["errors_detected"] == 1
        assert error_stats["error_rate"] > 0
    
    def test_health_check(self):
        """Test health check functionality."""
        health = self.planner.health_check()
        
        # Verify health status structure
        assert "status" in health
        assert "planning_functional" in health
        assert "components_initialized" in health
        assert "statistics" in health
        assert "last_check" in health
        
        # Verify healthy status
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert health["planning_functional"] is True
        assert health["components_initialized"] is True
    
    def test_quantum_circuit_optimization(self):
        """Test quantum circuit optimization robustness."""
        algorithm = {
            "qubits": 4,
            "operations": [
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "Hadamard", "qubits": [1]},  # Can be parallel
                {"gate": "CNOT", "qubits": [0, 2]},
                {"gate": "Phase", "qubits": [3]}  # Can be parallel with CNOT
            ]
        }
        
        plan = self.planner.plan_quantum_circuit(algorithm)
        optimized_plan = self.planner.optimize_quantum_circuit(plan)
        
        # Verify optimization added features
        assert "error_correction" in optimized_plan
        assert "entanglement_scheme" in optimized_plan
        
        # Verify error correction details
        error_correction = optimized_plan["error_correction"]
        assert "scheme" in error_correction
        assert "logical_qubits" in error_correction
        assert "error_threshold" in error_correction
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test minimum valid circuit
        minimal_algorithm = {
            "qubits": 1,
            "operations": [{"gate": "Phase", "qubits": [0]}]
        }
        
        plan = self.planner.plan_quantum_circuit(minimal_algorithm)
        assert plan["qubits"] == 1
        assert len(plan["photonic_qubits"]) == 1
        
        # Test circuit with no measurements
        no_measurements = {
            "qubits": 2,
            "operations": [{"gate": "Hadamard", "qubits": [0]}]
            # No measurements field
        }
        
        plan = self.planner.plan_quantum_circuit(no_measurements)
        assert "measurement_scheme" in plan
        assert len(plan["measurement_scheme"]["measurements"]) == 0
    
    def test_coherence_time_warnings(self):
        """Test coherence time constraint handling."""
        # Create planner with very short coherence time
        short_coherence_planner = QuantumInspiredTaskPlanner(
            coherence_time=10.0,  # Very short
            entanglement_fidelity=0.95
        )
        
        # Plan circuit that exceeds coherence time
        long_algorithm = {
            "qubits": 2,
            "operations": [
                {"gate": "Toffoli", "qubits": [0, 1, 0]},  # Slow gate
                {"gate": "Toffoli", "qubits": [0, 1, 0]},  # Another slow gate
            ]
        }
        
        # Should still complete but with warnings
        plan = short_coherence_planner.plan_quantum_circuit(long_algorithm)
        
        coherence_opt = plan["coherence_optimization"]
        assert coherence_opt["coherence_ratio"] < 1.0  # Circuit time > coherence time
        assert len(coherence_opt["optimization_strategies"]) > 0


def test_integration_with_security_system():
    """Test integration with security and monitoring systems."""
    planner = QuantumInspiredTaskPlanner()
    
    # Test that security components are initialized
    assert planner.parameter_validator is not None
    assert planner.resource_limiter is not None
    assert planner.logger is not None
    
    # Test planning with monitoring
    algorithm = {
        "qubits": 3,
        "operations": [
            {"gate": "Hadamard", "qubits": [0]},
            {"gate": "CNOT", "qubits": [0, 1]}
        ]
    }
    
    plan = planner.plan_quantum_circuit(algorithm)
    
    # Verify monitoring metadata
    metadata = plan["planning_metadata"]
    assert "timestamp" in metadata
    assert "validation_passed" in metadata
    assert "resource_usage" in metadata
    assert metadata["validation_passed"] is True


def test_error_recovery_and_logging():
    """Test error recovery and comprehensive logging."""
    planner = QuantumInspiredTaskPlanner()
    
    # Test that errors are properly logged and statistics updated
    initial_errors = planner.planning_stats["errors_detected"]
    
    try:
        planner.plan_quantum_circuit(None)  # Should cause error
    except ValidationError:
        pass
    
    assert planner.planning_stats["errors_detected"] == initial_errors + 1
    
    # Test that valid operations still work after errors
    valid_algorithm = {
        "qubits": 2,
        "operations": [{"gate": "Hadamard", "qubits": [0]}]
    }
    
    plan = planner.plan_quantum_circuit(valid_algorithm)
    assert plan is not None


if __name__ == "__main__":
    print("🧪 Running Quantum Task Planner Robustness Tests")
    print("=" * 60)
    
    # Setup
    setup_module()
    
    # Create test instance
    test_class = TestQuantumTaskPlannerRobustness()
    test_class.setup_method()
    
    # Run key tests
    test_methods = [
        "test_initialization_validation",
        "test_valid_quantum_algorithm_planning", 
        "test_invalid_algorithm_input_validation",
        "test_resource_limit_enforcement",
        "test_planning_statistics",
        "test_health_check",
        "test_coherence_time_warnings"
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            print(f"Running {method_name}...")
            method = getattr(test_class, method_name)
            method()
            print(f"✅ {method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name} FAILED: {str(e)}")
            failed += 1
    
    # Run integration tests
    try:
        print("Running integration tests...")
        test_integration_with_security_system()
        test_error_recovery_and_logging()
        print("✅ Integration tests PASSED")
        passed += 2
    except Exception as e:
        print(f"❌ Integration tests FAILED: {str(e)}")
        failed += 2
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Generation 2 robustness tests passed!")
        print("✅ Error handling: Comprehensive")
        print("✅ Input validation: Robust")
        print("✅ Security integration: Complete")
        print("✅ Monitoring: Functional")
        print("✅ Resource limits: Enforced")
    else:
        print("⚠️  Some tests failed - review implementation")
    
    print("\n🚀 Quantum Task Planner Generation 2 Testing Complete!")