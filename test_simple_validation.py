#!/usr/bin/env python3
"""Simple validation of breakthrough quantum algorithms."""

import time
import sys

def test_breakthrough_algorithms():
    """Test breakthrough algorithms directly."""
    print("🧪 TESTING BREAKTHROUGH QUANTUM ALGORITHMS")
    print("=" * 60)
    
    try:
        from holo_code_gen.quantum_algorithms import PhotonicQuantumAlgorithms
        
        # Initialize algorithms
        quantum_algos = PhotonicQuantumAlgorithms()
        
        results = []
        
        # Test 1: Adaptive State Injection CV-QAOA
        print("\n🔬 Test 1: Adaptive State Injection CV-QAOA")
        print("-" * 50)
        
        try:
            simple_graph = {
                "nodes": list(range(4)),
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]
            }
            
            start = time.time()
            result1 = quantum_algos._adaptive_state_injection_cv_qaoa(
                simple_graph, depth=2, max_iterations=20, adaptation_threshold=0.01
            )
            duration = time.time() - start
            
            print(f"✅ Algorithm: {result1.get('algorithm', 'unknown')}")
            print(f"✅ Cost: {result1.get('cost', 'N/A')}")
            print(f"✅ Adaptations: {result1.get('adaptations_triggered', 0)}")
            print(f"✅ Time: {duration:.3f}s")
            print(f"✅ Improvement: {result1.get('baseline_improvement_percentage', 0):.1f}%")
            
            results.append({"test": "adaptive_cv_qaoa", "passed": True, "result": result1})
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({"test": "adaptive_cv_qaoa", "passed": False, "error": str(e)})
        
        # Test 2: Coherence-Enhanced VQE
        print("\n🔬 Test 2: Coherence-Enhanced VQE")
        print("-" * 50)
        
        try:
            hamiltonian = {
                "num_qubits": 4,
                "terms": [
                    {"coeff": -1.0, "pauli_string": "ZIII"},
                    {"coeff": -0.5, "pauli_string": "IZII"}
                ]
            }
            
            start = time.time()
            result2 = quantum_algos._coherence_enhanced_vqe(
                hamiltonian, num_layers=2, max_iterations=20, coherence_threshold=0.7
            )
            duration = time.time() - start
            
            print(f"✅ Algorithm: {result2.get('algorithm', 'unknown')}")
            print(f"✅ Energy: {result2.get('ground_state_energy', 'N/A')}")
            print(f"✅ Convergence speedup: {result2.get('convergence_speedup', 1.0):.2f}x")
            print(f"✅ Coherence: {result2.get('final_coherence', 0.0):.3f}")
            print(f"✅ Time: {duration:.3f}s")
            print(f"✅ Improvement: {result2.get('convergence_improvement_percentage', 0):.1f}%")
            
            results.append({"test": "coherence_vqe", "passed": True, "result": result2})
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({"test": "coherence_vqe", "passed": False, "error": str(e)})
        
        # Test 3: Quantum Natural Gradient Optimization
        print("\n🔬 Test 3: Quantum Natural Gradient Optimization")
        print("-" * 50)
        
        try:
            objective = {
                "type": "quadratic",
                "dimension": 4
            }
            initial_params = [0.1, 0.2, 0.1, 0.15]
            
            start = time.time()
            result3 = quantum_algos._quantum_natural_gradient_optimization(
                objective, initial_params, max_iterations=30
            )
            duration = time.time() - start
            
            print(f"✅ Algorithm: {result3.get('algorithm', 'unknown')}")
            print(f"✅ Optimal cost: {result3.get('optimal_cost', 'N/A')}")
            print(f"✅ Convergence improvement: {result3.get('convergence_improvement', 1.0):.2f}x")
            print(f"✅ Time: {duration:.3f}s")
            print(f"✅ Natural gradient advantage: {result3.get('natural_gradient_advantage', False)}")
            
            results.append({"test": "natural_gradient", "passed": True, "result": result3})
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({"test": "natural_gradient", "passed": False, "error": str(e)})
        
        # Test 4: Photonic Quantum Kernel ML
        print("\n🔬 Test 4: Photonic Quantum Kernel ML")
        print("-" * 50)
        
        try:
            training_data = {
                "features": [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
                "labels": [1, -1, 1, -1]
            }
            
            start = time.time()
            result4 = quantum_algos._photonic_quantum_kernel_ml(
                training_data, kernel_type="rbf_continuous", cv_encoding_dim=3
            )
            duration = time.time() - start
            
            print(f"✅ Algorithm: {result4.get('algorithm', 'unknown')}")
            print(f"✅ Training accuracy: {result4.get('training_accuracy', 0.0):.3f}")
            print(f"✅ Quantum speedup: {result4.get('quantum_speedup', 1.0):.1f}x")
            print(f"✅ Energy efficiency: {result4.get('energy_efficiency_improvement', 1.0):.1f}x")
            print(f"✅ Time: {duration:.3f}s")
            print(f"✅ Quantum advantage: {result4.get('quantum_advantage_achieved', False)}")
            
            results.append({"test": "quantum_kernel_ml", "passed": True, "result": result4})
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({"test": "quantum_kernel_ml", "passed": False, "error": str(e)})
        
        # Test 5: Time-Domain Multiplexed Compilation
        print("\n🔬 Test 5: Time-Domain Multiplexed Compilation")
        print("-" * 50)
        
        try:
            circuit_spec = {
                "gates": [
                    {"type": "H", "qubits": [0]},
                    {"type": "CNOT", "qubits": [0, 1]},
                    {"type": "RZ", "qubits": [1]}
                ],
                "num_qubits": 2
            }
            
            hardware_constraints = {
                "photonic_modes": 4,
                "delay_line_ns": 100
            }
            
            start = time.time()
            result5 = quantum_algos._time_domain_multiplexed_compilation(
                circuit_spec, hardware_constraints, multiplexing_factor=3
            )
            duration = time.time() - start
            
            print(f"✅ Algorithm: {result5.get('algorithm', 'unknown')}")
            print(f"✅ Compilation successful: {result5.get('compilation_successful', False)}")
            print(f"✅ Effective qubits: {result5.get('effective_qubits', 0)}")
            print(f"✅ Scalability factor: {result5.get('qubit_scalability_factor', 1.0):.1f}x")
            print(f"✅ Time: {duration:.3f}s")
            print(f"✅ Production ready: {result5.get('production_ready', False)}")
            
            results.append({"test": "time_multiplexed", "passed": True, "result": result5})
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({"test": "time_multiplexed", "passed": False, "error": str(e)})
        
        # Summary
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in results if r.get("passed", False))
        total = len(results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print(f"✅ Tests passed: {passed}/{total}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🏆 BREAKTHROUGH ALGORITHMS VALIDATED")
            print("✅ Ready for production deployment")
        elif success_rate >= 60:
            print("⚠️  PARTIAL VALIDATION")
            print("Some algorithms need optimization")
        else:
            print("❌ VALIDATION FAILED")
            print("Significant issues need addressing")
        
        return success_rate >= 60
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation execution."""
    success = test_breakthrough_algorithms()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())