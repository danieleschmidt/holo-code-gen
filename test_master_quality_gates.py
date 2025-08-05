#!/usr/bin/env python3
"""
Master Quality Gates Test Suite

Comprehensive testing suite that validates all three generations of development:
- Generation 1: Basic functionality 
- Generation 2: Robustness and reliability
- Generation 3: Scaling and optimization

This test suite ensures production readiness across all quality gates.
"""

import sys
import time
import traceback
from typing import Dict, List, Any, Tuple
import subprocess
import json

# Import test modules
try:
    from test_quantum_task_planner import TestQuantumTaskPlannerRobustness, setup_module as setup_robustness
    from test_quantum_performance import TestQuantumPlannerPerformance, setup_module as setup_performance
except ImportError as e:
    print(f"⚠️  Could not import test modules: {e}")
    print("Running basic functionality tests only")

# Core imports
from holo_code_gen import QuantumInspiredTaskPlanner
from holo_code_gen.security import initialize_security
from holo_code_gen.exceptions import ValidationError, CompilationError


def run_quality_gate_1_basic_functionality() -> Tuple[bool, Dict[str, Any]]:
    """Run Quality Gate 1: Basic Functionality Tests"""
    print("🔧 QUALITY GATE 1: BASIC FUNCTIONALITY")
    print("=" * 50)
    
    results = {
        "gate_name": "Basic Functionality",
        "tests_passed": 0,
        "tests_failed": 0,
        "details": [],
        "overall_pass": False
    }
    
    try:
        # Initialize quantum task planner
        planner = QuantumInspiredTaskPlanner(
            coherence_time=1000.0,
            entanglement_fidelity=0.95
        )
        
        # Test 1: Basic quantum circuit planning
        print("Testing basic quantum circuit planning...")
        basic_algorithm = {
            "qubits": 2,
            "operations": [
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]}
            ],
            "measurements": [{"qubit": 0, "basis": "computational"}]
        }
        
        plan = planner.plan_quantum_circuit(basic_algorithm)
        if plan and "qubits" in plan and "gate_sequence" in plan:
            results["tests_passed"] += 1
            results["details"].append("✅ Basic circuit planning: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Basic circuit planning: FAILED")
        
        # Test 2: Photonic qubit mapping
        print("Testing photonic qubit mapping...")
        photonic_qubits = plan.get("photonic_qubits", [])
        if len(photonic_qubits) == 2 and all("wavelength" in q for q in photonic_qubits):
            results["tests_passed"] += 1
            results["details"].append("✅ Photonic qubit mapping: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Photonic qubit mapping: FAILED")
        
        # Test 3: Quantum gate to photonic component mapping
        print("Testing quantum gate mapping...")
        gate_sequence = plan.get("gate_sequence", [])
        if len(gate_sequence) == 2 and all("photonic_component" in g for g in gate_sequence):
            results["tests_passed"] += 1
            results["details"].append("✅ Quantum gate mapping: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Quantum gate mapping: FAILED")
        
        # Test 4: Entanglement planning
        print("Testing entanglement planning...")
        entanglement = plan.get("entanglement_scheme", {})
        if "pairs" in entanglement and entanglement["type"] == "bell_pairs":
            results["tests_passed"] += 1
            results["details"].append("✅ Entanglement planning: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Entanglement planning: FAILED")
        
        # Test 5: Quantum circuit optimization
        print("Testing quantum circuit optimization...")
        optimized_plan = planner.optimize_quantum_circuit(plan)
        if "error_correction" in optimized_plan and "gate_sequence" in optimized_plan:
            results["tests_passed"] += 1
            results["details"].append("✅ Circuit optimization: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Circuit optimization: FAILED")
        
        # Test 6: Coherence time analysis
        print("Testing coherence time analysis...")
        coherence_opt = plan.get("coherence_optimization", {})
        if "coherence_ratio" in coherence_opt and "estimated_circuit_time_ns" in coherence_opt:
            results["tests_passed"] += 1
            results["details"].append("✅ Coherence analysis: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Coherence analysis: FAILED")
        
        # Test 7: Complex quantum algorithm
        print("Testing complex quantum algorithm...")
        complex_algorithm = {
            "qubits": 4,
            "operations": [
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "CNOT", "qubits": [1, 2]},
                {"gate": "Toffoli", "qubits": [0, 1, 3]},
                {"gate": "Phase", "qubits": [2]}
            ],
            "measurements": [
                {"qubit": 0, "basis": "computational"},
                {"qubit": 3, "basis": "X"}
            ]
        }
        
        complex_plan = planner.plan_quantum_circuit(complex_algorithm)
        if complex_plan and len(complex_plan["gate_sequence"]) == 5:
            results["tests_passed"] += 1
            results["details"].append("✅ Complex algorithm planning: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Complex algorithm planning: FAILED")
        
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append(f"❌ Exception in basic functionality: {str(e)}")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
    
    # Determine overall pass/fail
    total_tests = results["tests_passed"] + results["tests_failed"]
    pass_rate = results["tests_passed"] / total_tests if total_tests > 0 else 0
    results["overall_pass"] = pass_rate >= 0.85  # 85% pass rate required
    results["pass_rate"] = pass_rate
    
    print(f"\nQuality Gate 1 Results:")
    print(f"  Tests Passed: {results['tests_passed']}")
    print(f"  Tests Failed: {results['tests_failed']}")
    print(f"  Pass Rate: {pass_rate:.1%}")
    print(f"  Overall: {'PASSED' if results['overall_pass'] else 'FAILED'}")
    
    return results["overall_pass"], results


def run_quality_gate_2_robustness() -> Tuple[bool, Dict[str, Any]]:
    """Run Quality Gate 2: Robustness and Reliability Tests"""
    print("\n🛡️  QUALITY GATE 2: ROBUSTNESS & RELIABILITY")
    print("=" * 55)
    
    results = {
        "gate_name": "Robustness & Reliability",
        "tests_passed": 0,
        "tests_failed": 0,
        "details": [],
        "overall_pass": False
    }
    
    try:
        # Initialize security
        initialize_security()
        
        # Run robustness tests
        test_class = TestQuantumTaskPlannerRobustness()
        test_class.setup_method()
        
        robustness_tests = [
            ("Initialization Validation", test_class.test_initialization_validation),
            ("Input Validation", test_class.test_invalid_algorithm_input_validation),
            ("Resource Limits", test_class.test_resource_limit_enforcement),
            ("Plan Validation", test_class.test_photonic_plan_validation),
            ("Statistics Tracking", test_class.test_planning_statistics),
            ("Health Check", test_class.test_health_check),
            ("Error Recovery", test_class.test_edge_cases)
        ]
        
        for test_name, test_method in robustness_tests:
            try:
                print(f"Running {test_name}...")
                test_method()
                results["tests_passed"] += 1
                results["details"].append(f"✅ {test_name}: PASSED")
            except Exception as e:
                results["tests_failed"] += 1
                results["details"].append(f"❌ {test_name}: FAILED - {str(e)}")
                print(f"  ❌ {test_name} failed: {str(e)}")
        
        # Additional robustness checks
        print("Running additional robustness checks...")
        
        # Test security integration
        planner = QuantumInspiredTaskPlanner()
        if (planner.parameter_validator and planner.resource_limiter and 
            planner.logger and planner.performance_monitor):
            results["tests_passed"] += 1
            results["details"].append("✅ Security Integration: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Security Integration: FAILED")
        
        # Test error handling
        error_handled = False
        try:
            planner.plan_quantum_circuit("invalid input")
        except ValidationError:
            error_handled = True
        except Exception:
            pass
        
        if error_handled:
            results["tests_passed"] += 1
            results["details"].append("✅ Error Handling: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Error Handling: FAILED")
        
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append(f"❌ Exception in robustness testing: {str(e)}")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
    
    # Determine overall pass/fail
    total_tests = results["tests_passed"] + results["tests_failed"]
    pass_rate = results["tests_passed"] / total_tests if total_tests > 0 else 0
    results["overall_pass"] = pass_rate >= 0.90  # 90% pass rate required for robustness
    results["pass_rate"] = pass_rate
    
    print(f"\nQuality Gate 2 Results:")
    print(f"  Tests Passed: {results['tests_passed']}")
    print(f"  Tests Failed: {results['tests_failed']}")
    print(f"  Pass Rate: {pass_rate:.1%}")
    print(f"  Overall: {'PASSED' if results['overall_pass'] else 'FAILED'}")
    
    return results["overall_pass"], results


def run_quality_gate_3_performance() -> Tuple[bool, Dict[str, Any]]:
    """Run Quality Gate 3: Performance and Scaling Tests"""
    print("\n🚀 QUALITY GATE 3: PERFORMANCE & SCALING")
    print("=" * 50)
    
    results = {
        "gate_name": "Performance & Scaling",
        "tests_passed": 0,
        "tests_failed": 0,
        "details": [],
        "overall_pass": False
    }
    
    try:
        # Initialize performance components
        try:
            from holo_code_gen.performance import initialize_performance
            initialize_performance()
        except:
            print("⚠️  Performance components not available - using basic implementations")
        
        # Run performance tests
        test_class = TestQuantumPlannerPerformance()
        test_class.setup_method()
        
        performance_tests = [
            ("Caching Performance", test_class.test_caching_performance),
            ("Memory Management", test_class.test_memory_management),
            ("Cache Warmup", test_class.test_cache_warmup),
            ("Concurrent Access", test_class.test_concurrent_access),
            ("Performance Metrics", test_class.test_comprehensive_performance_metrics)
        ]
        
        for test_name, test_method in performance_tests:
            try:
                print(f"Running {test_name}...")
                test_method()
                results["tests_passed"] += 1
                results["details"].append(f"✅ {test_name}: PASSED")
            except Exception as e:
                results["tests_failed"] += 1
                results["details"].append(f"❌ {test_name}: FAILED - {str(e)}")
                print(f"  ❌ {test_name} failed: {str(e)}")
        
        # Performance benchmarks
        print("Running performance benchmarks...")
        
        planner = QuantumInspiredTaskPlanner()
        
        # Test planning speed
        algorithm = {
            "qubits": 4,
            "operations": [{"gate": "Hadamard", "qubits": [i]} for i in range(4)],
            "measurements": [{"qubit": 0, "basis": "computational"}]
        }
        
        start_time = time.time()
        plan = planner.plan_quantum_circuit(algorithm)
        planning_time = (time.time() - start_time) * 1000  # ms
        
        if planning_time < 100:  # Should be under 100ms
            results["tests_passed"] += 1
            results["details"].append(f"✅ Planning Speed: PASSED ({planning_time:.1f}ms)")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"❌ Planning Speed: FAILED ({planning_time:.1f}ms)")
        
        # Test cache effectiveness
        start_time = time.time()
        plan2 = planner.plan_quantum_circuit(algorithm)  # Should be cached
        cached_time = (time.time() - start_time) * 1000  # ms
        
        if cached_time < planning_time * 0.5:  # Cached should be at least 2x faster
            results["tests_passed"] += 1
            results["details"].append(f"✅ Cache Effectiveness: PASSED ({cached_time:.1f}ms)")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"❌ Cache Effectiveness: FAILED ({cached_time:.1f}ms)")
        
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append(f"❌ Exception in performance testing: {str(e)}")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
    
    # Determine overall pass/fail
    total_tests = results["tests_passed"] + results["tests_failed"]
    pass_rate = results["tests_passed"] / total_tests if total_tests > 0 else 0
    results["overall_pass"] = pass_rate >= 0.80  # 80% pass rate required for performance
    results["pass_rate"] = pass_rate
    
    print(f"\nQuality Gate 3 Results:")
    print(f"  Tests Passed: {results['tests_passed']}")
    print(f"  Tests Failed: {results['tests_failed']}")
    print(f"  Pass Rate: {pass_rate:.1%}")
    print(f"  Overall: {'PASSED' if results['overall_pass'] else 'FAILED'}")
    
    return results["overall_pass"], results


def run_quality_gate_4_integration() -> Tuple[bool, Dict[str, Any]]:
    """Run Quality Gate 4: Integration and End-to-End Tests"""
    print("\n🔗 QUALITY GATE 4: INTEGRATION & END-TO-END")
    print("=" * 55)
    
    results = {
        "gate_name": "Integration & End-to-End",
        "tests_passed": 0,
        "tests_failed": 0,
        "details": [],
        "overall_pass": False
    }
    
    try:
        # Initialize all systems
        initialize_security()
        planner = QuantumInspiredTaskPlanner()
        
        # Test 1: Complete quantum algorithm workflow
        print("Testing complete quantum algorithm workflow...")
        quantum_search_algorithm = {
            "name": "quantum_search",
            "qubits": 4,
            "operations": [
                # Initialize superposition
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "Hadamard", "qubits": [1]},
                
                # Oracle operation
                {"gate": "CNOT", "qubits": [0, 2]},
                {"gate": "CNOT", "qubits": [1, 2]},
                {"gate": "Toffoli", "qubits": [0, 1, 3]},
                
                # Diffusion operator
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "Hadamard", "qubits": [1]},
                {"gate": "Phase", "qubits": [0]},
                {"gate": "Phase", "qubits": [1]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "Phase", "qubits": [1]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "Hadamard", "qubits": [1]}
            ],
            "measurements": [
                {"qubit": 0, "basis": "computational"},
                {"qubit": 1, "basis": "computational"},
                {"qubit": 2, "basis": "computational"},
                {"qubit": 3, "basis": "computational"}
            ]
        }
        
        # Plan the circuit
        plan = planner.plan_quantum_circuit(quantum_search_algorithm)
        
        # Optimize the circuit
        optimized_plan = planner.optimize_quantum_circuit(plan)
        
        # Validate the complete workflow
        if (optimized_plan and 
            len(optimized_plan["gate_sequence"]) == len(quantum_search_algorithm["operations"]) and
            "error_correction" in optimized_plan and
            "planning_metadata" in optimized_plan):
            results["tests_passed"] += 1
            results["details"].append("✅ Complete Workflow: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Complete Workflow: FAILED")
        
        # Test 2: Cross-component integration
        print("Testing cross-component integration...")
        
        # Verify all components are properly integrated
        component_checks = [
            ("Parameter Validator", planner.parameter_validator is not None),
            ("Resource Limiter", planner.resource_limiter is not None),
            ("Logger", planner.logger is not None),
            ("Performance Monitor", planner.performance_monitor is not None),
            ("Cache Manager", planner.cache_manager is not None)
        ]
        
        integration_passed = 0
        for component_name, check in component_checks:
            if check:
                integration_passed += 1
            else:
                print(f"  ❌ {component_name} not integrated")
        
        if integration_passed >= 4:  # At least 4 out of 5 components
            results["tests_passed"] += 1
            results["details"].append("✅ Component Integration: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Component Integration: FAILED")
        
        # Test 3: Resource usage validation
        print("Testing resource usage validation...")
        
        resource_usage = optimized_plan["planning_metadata"]["resource_usage"]
        resource_checks = [
            resource_usage.get("estimated_power_mw", 0) > 0,
            resource_usage.get("estimated_area_mm2", 0) > 0,
            0 <= resource_usage.get("complexity_score", -1) <= 10,
            resource_usage.get("photonic_components", 0) > 0
        ]
        
        if all(resource_checks):
            results["tests_passed"] += 1
            results["details"].append("✅ Resource Usage Validation: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Resource Usage Validation: FAILED")
        
        # Test 4: Error correction integration
        print("Testing error correction integration...")
        
        error_correction = optimized_plan.get("error_correction", {})
        if ("scheme" in error_correction and 
            "logical_qubits" in error_correction and
            "error_threshold" in error_correction):
            results["tests_passed"] += 1
            results["details"].append("✅ Error Correction Integration: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Error Correction Integration: FAILED")
        
        # Test 5: Performance metrics consistency
        print("Testing performance metrics consistency...")
        
        planning_stats = planner.get_planning_statistics()
        cache_stats = planner.get_cache_statistics()
        
        metrics_consistent = (
            planning_stats["circuits_planned"] > 0 and
            0 <= planning_stats["cache_hit_rate"] <= 1.0 and
            planning_stats["error_rate"] >= 0 and
            cache_stats["cache_hit_rate"] >= 0
        )
        
        if metrics_consistent:
            results["tests_passed"] += 1
            results["details"].append("✅ Metrics Consistency: PASSED")
        else:
            results["tests_failed"] += 1
            results["details"].append("❌ Metrics Consistency: FAILED")
        
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append(f"❌ Exception in integration testing: {str(e)}")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
    
    # Determine overall pass/fail
    total_tests = results["tests_passed"] + results["tests_failed"]
    pass_rate = results["tests_passed"] / total_tests if total_tests > 0 else 0
    results["overall_pass"] = pass_rate >= 0.80  # 80% pass rate required
    results["pass_rate"] = pass_rate
    
    print(f"\nQuality Gate 4 Results:")
    print(f"  Tests Passed: {results['tests_passed']}")
    print(f"  Tests Failed: {results['tests_failed']}")
    print(f"  Pass Rate: {pass_rate:.1%}")
    print(f"  Overall: {'PASSED' if results['overall_pass'] else 'FAILED'}")
    
    return results["overall_pass"], results


def run_all_quality_gates() -> Dict[str, Any]:
    """Run all quality gates and produce comprehensive report."""
    print("🏆 QUANTUM TASK PLANNER - MASTER QUALITY GATES")
    print("=" * 65)
    print("Running comprehensive test suite for production readiness validation")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all quality gates
    quality_gates = [
        ("QG1", "Basic Functionality", run_quality_gate_1_basic_functionality),
        ("QG2", "Robustness & Reliability", run_quality_gate_2_robustness),
        ("QG3", "Performance & Scaling", run_quality_gate_3_performance),
        ("QG4", "Integration & End-to-End", run_quality_gate_4_integration)
    ]
    
    overall_results = {
        "start_time": time.time(),
        "gates": {},
        "summary": {
            "total_gates": len(quality_gates),
            "gates_passed": 0,
            "gates_failed": 0,
            "overall_pass": False
        }
    }
    
    # Execute each quality gate
    for gate_code, gate_name, gate_function in quality_gates:
        print(f"\n{'='*65}")
        start_time = time.time()
        
        try:
            gate_passed, gate_results = gate_function()
            execution_time = time.time() - start_time
            
            gate_results["execution_time"] = execution_time
            gate_results["gate_code"] = gate_code
            
            overall_results["gates"][gate_code] = gate_results
            
            if gate_passed:
                overall_results["summary"]["gates_passed"] += 1
                print(f"\n🎉 {gate_code} ({gate_name}): PASSED in {execution_time:.1f}s")
            else:
                overall_results["summary"]["gates_failed"] += 1
                print(f"\n💥 {gate_code} ({gate_name}): FAILED in {execution_time:.1f}s")
                
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n💥 {gate_code} ({gate_name}): EXCEPTION in {execution_time:.1f}s")
            print(f"Exception: {str(e)}")
            
            overall_results["gates"][gate_code] = {
                "gate_name": gate_name,
                "overall_pass": False,
                "execution_time": execution_time,
                "exception": str(e),
                "details": [f"❌ Exception: {str(e)}"]
            }
            overall_results["summary"]["gates_failed"] += 1
    
    # Calculate overall results
    total_execution_time = time.time() - overall_results["start_time"]
    gates_passed = overall_results["summary"]["gates_passed"]
    total_gates = overall_results["summary"]["total_gates"]
    
    overall_results["summary"]["overall_pass"] = gates_passed == total_gates
    overall_results["summary"]["pass_rate"] = gates_passed / total_gates if total_gates > 0 else 0
    overall_results["summary"]["total_execution_time"] = total_execution_time
    overall_results["end_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    return overall_results


def print_final_report(results: Dict[str, Any]) -> None:
    """Print comprehensive final report."""
    print(f"\n{'='*65}")
    print("🏆 FINAL QUALITY GATES REPORT")
    print(f"{'='*65}")
    
    print(f"Execution Time: {results['summary']['total_execution_time']:.1f}s")
    print(f"End Time: {results['end_time']}")
    
    print(f"\n📊 OVERALL SUMMARY")
    print(f"{'='*25}")
    print(f"Total Quality Gates: {results['summary']['total_gates']}")
    print(f"Gates Passed: {results['summary']['gates_passed']}")
    print(f"Gates Failed: {results['summary']['gates_failed']}")
    print(f"Pass Rate: {results['summary']['pass_rate']:.1%}")
    print(f"Overall Status: {'✅ PASSED' if results['summary']['overall_pass'] else '❌ FAILED'}")
    
    print(f"\n📋 DETAILED RESULTS")
    print(f"{'='*30}")
    
    for gate_code, gate_result in results["gates"].items():
        status = "✅ PASSED" if gate_result.get("overall_pass", False) else "❌ FAILED"
        execution_time = gate_result.get("execution_time", 0)
        pass_rate = gate_result.get("pass_rate", 0)
        
        print(f"\n{gate_code}: {gate_result['gate_name']}")
        print(f"  Status: {status}")
        print(f"  Execution Time: {execution_time:.1f}s")
        print(f"  Pass Rate: {pass_rate:.1%}")
        
        if "details" in gate_result:
            print("  Details:")
            for detail in gate_result["details"][-5:]:  # Show last 5 details
                print(f"    {detail}")
    
    # Production readiness assessment
    print(f"\n🚀 PRODUCTION READINESS ASSESSMENT")
    print(f"{'='*40}")
    
    if results["summary"]["overall_pass"]:
        print("🎉 QUANTUM TASK PLANNER IS READY FOR PRODUCTION!")
        print("✅ All quality gates passed successfully")
        print("✅ Basic functionality: Operational")
        print("✅ Robustness & reliability: Validated")
        print("✅ Performance & scaling: Optimized")
        print("✅ Integration: Complete")
        
        print(f"\n🎯 KEY ACHIEVEMENTS:")
        print("• Quantum-inspired task planning implemented")
        print("• Comprehensive error handling and validation")
        print("• High-performance caching and parallel processing")
        print("• Security measures and resource management")
        print("• Full integration with photonic compilation pipeline")
        
    else:
        print("⚠️  QUANTUM TASK PLANNER REQUIRES FIXES BEFORE PRODUCTION")
        failed_gates = [code for code, result in results["gates"].items() 
                       if not result.get("overall_pass", False)]
        print(f"❌ Failed gates: {', '.join(failed_gates)}")
        print("🔧 Review failed tests and address issues before deployment")
    
    print(f"\n{'='*65}")


if __name__ == "__main__":
    print("🧪 Starting Master Quality Gates Test Suite")
    
    # Run all quality gates
    results = run_all_quality_gates()
    
    # Print final report
    print_final_report(results)
    
    # Save results to file
    try:
        with open("/root/repo/quality_gates_report.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: quality_gates_report.json")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    # Exit with appropriate code
    exit_code = 0 if results["summary"]["overall_pass"] else 1
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)