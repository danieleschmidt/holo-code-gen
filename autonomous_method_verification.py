#!/usr/bin/env python3
"""AUTONOMOUS METHOD VERIFICATION"""

import sys
sys.path.append('/root/repo')

from holo_code_gen import QuantumInspiredTaskPlanner

def verify_methods():
    """Verify that all required methods exist."""
    planner = QuantumInspiredTaskPlanner()
    
    print("🔍 Verifying QuantumInspiredTaskPlanner methods...")
    
    # Check if methods exist
    methods_to_check = [
        '_plan_error_correction',
        '_parallel_plan_quantum_circuit',
        'plan_quantum_circuit',
        'optimize_quantum_circuit'
    ]
    
    results = {}
    for method_name in methods_to_check:
        exists = hasattr(planner, method_name)
        results[method_name] = exists
        status = "✅" if exists else "❌"
        print(f"   {status} {method_name}: {'EXISTS' if exists else 'MISSING'}")
    
    # Test basic functionality
    try:
        quantum_algorithm = {
            "name": "test_bell_state",
            "qubits": 2,
            "operations": [
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]}
            ]
        }
        
        print("\n🧪 Testing basic quantum circuit planning...")
        plan = planner.plan_quantum_circuit(quantum_algorithm)
        print("   ✅ Plan quantum circuit: SUCCESS")
        
        print("🧪 Testing quantum circuit optimization...")
        optimized = planner.optimize_quantum_circuit(plan)
        print("   ✅ Optimize quantum circuit: SUCCESS")
        
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        return False
    
    all_exist = all(results.values())
    print(f"\n🏆 Overall status: {'✅ ALL METHODS EXIST' if all_exist else '❌ MISSING METHODS'}")
    return all_exist

if __name__ == "__main__":
    verify_methods()