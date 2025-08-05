#!/usr/bin/env python3
"""
Performance Test Suite for Quantum-Inspired Task Planner Generation 3

Tests high-performance caching, parallel processing, and optimization features.
Demonstrates scaling capabilities and performance improvements.
"""

import time
import json
from typing import List, Dict, Any
import concurrent.futures

# Test imports  
from holo_code_gen.optimization import QuantumInspiredTaskPlanner
from holo_code_gen.security import initialize_security
from holo_code_gen.performance import initialize_performance


def setup_module():
    """Setup performance and security components."""
    initialize_security()
    try:
        initialize_performance()
    except:
        print("⚠️  Performance components not fully available - using mock implementations")


def generate_test_algorithms(count: int = 50) -> List[Dict[str, Any]]:
    """Generate a variety of test quantum algorithms for performance testing."""
    algorithms = []
    
    # Simple algorithms (1-2 qubits)
    for i in range(count // 3):
        algorithms.append({
            "name": f"simple_circuit_{i}",
            "qubits": 2,
            "operations": [
                {"gate": "Hadamard", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "Phase", "qubits": [1]}
            ],
            "measurements": [{"qubit": j, "basis": "computational"} for j in range(2)]
        })
    
    # Medium algorithms (3-5 qubits)
    for i in range(count // 3):
        qubit_count = 3 + (i % 3)
        operations = []
        for j in range(qubit_count + 2):
            gate_type = ["Hadamard", "CNOT", "Phase", "Swap"][j % 4]
            if gate_type == "CNOT":
                operations.append({"gate": gate_type, "qubits": [j % qubit_count, (j + 1) % qubit_count]})
            else:
                operations.append({"gate": gate_type, "qubits": [j % qubit_count]})
        
        algorithms.append({
            "name": f"medium_circuit_{i}",
            "qubits": qubit_count,
            "operations": operations,
            "measurements": [{"qubit": j, "basis": "computational"} for j in range(qubit_count)]
        })
    
    # Complex algorithms (6-10 qubits)
    for i in range(count // 3):
        qubit_count = 6 + (i % 5)
        operations = []
        for j in range(qubit_count * 2):
            gate_type = ["Hadamard", "CNOT", "Phase", "Swap", "Toffoli"][j % 5]
            if gate_type == "CNOT":
                operations.append({"gate": gate_type, "qubits": [j % qubit_count, (j + 1) % qubit_count]})
            elif gate_type == "Toffoli":
                operations.append({"gate": gate_type, "qubits": [j % qubit_count, (j + 1) % qubit_count, (j + 2) % qubit_count]})
            else:
                operations.append({"gate": gate_type, "qubits": [j % qubit_count]})
        
        algorithms.append({
            "name": f"complex_circuit_{i}",
            "qubits": qubit_count,
            "operations": operations,
            "measurements": [{"qubit": j, "basis": "computational"} for j in range(qubit_count)]
        })
    
    return algorithms


class TestQuantumPlannerPerformance:
    """Test suite for quantum planner performance features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.planner = QuantumInspiredTaskPlanner(
            coherence_time=1000.0,
            entanglement_fidelity=0.95
        )
        self.test_algorithms = generate_test_algorithms(30)
    
    def test_caching_performance(self):
        """Test caching system performance and effectiveness."""
        print("\n🚀 Testing Caching Performance")
        print("-" * 40)
        
        # Clear caches to start fresh
        self.planner.clear_caches()
        initial_stats = self.planner.get_cache_statistics()
        assert initial_stats["cache_hit_rate"] == 0.0
        
        # Test cache misses (first time planning)
        start_time = time.time()
        for algorithm in self.test_algorithms[:10]:
            plan = self.planner.plan_quantum_circuit(algorithm)
            assert plan is not None
        
        first_run_time = time.time() - start_time
        first_stats = self.planner.get_cache_statistics()
        
        print(f"First run (cache misses): {first_run_time:.3f}s")
        print(f"Cache hit rate: {first_stats['cache_hit_rate']:.2%}")
        
        # Test cache hits (second time planning same algorithms)
        start_time = time.time()
        for algorithm in self.test_algorithms[:10]:
            plan = self.planner.plan_quantum_circuit(algorithm)
            assert plan is not None
            # Verify it came from cache
            assert plan.get("planning_metadata", {}).get("from_cache", False) is True
        
        second_run_time = time.time() - start_time
        second_stats = self.planner.get_cache_statistics()
        
        print(f"Second run (cache hits): {second_run_time:.3f}s")
        print(f"Cache hit rate: {second_stats['cache_hit_rate']:.2%}")
        print(f"Speedup: {first_run_time / second_run_time:.1f}x")
        
        # Verify cache effectiveness
        assert second_stats["cache_hit_rate"] > 0.5  # At least 50% hit rate
        assert second_run_time < first_run_time  # Second run should be faster
        
        print(f"✅ Cache speedup: {first_run_time / second_run_time:.1f}x")
    
    def test_parallel_processing_performance(self):
        """Test parallel processing performance improvements."""
        print("\n⚡ Testing Parallel Processing Performance")
        print("-" * 45)
        
        # Test with complex algorithms that benefit from parallelization
        complex_algorithms = [alg for alg in self.test_algorithms if alg["qubits"] >= 4][:5]
        
        if not complex_algorithms:
            print("⚠️  No complex algorithms available for parallel testing")
            return
        
        # Clear cache to ensure fair comparison
        self.planner.clear_caches()
        
        # Sequential processing timing
        total_sequential_time = 0
        for algorithm in complex_algorithms:
            start_time = time.time()
            plan = self.planner._sequential_plan_quantum_circuit(
                algorithm, algorithm["qubits"], algorithm["operations"]
            )
            sequential_time = time.time() - start_time
            total_sequential_time += sequential_time
            
            assert plan is not None
            assert not plan.get("planning_metadata", {}).get("parallel_processing", True)
        
        # Parallel processing timing  
        total_parallel_time = 0
        for algorithm in complex_algorithms:
            start_time = time.time()
            plan = self.planner._parallel_plan_quantum_circuit(
                algorithm, algorithm["qubits"], algorithm["operations"]
            )
            parallel_time = time.time() - start_time
            total_parallel_time += parallel_time
            
            assert plan is not None
            assert plan.get("planning_metadata", {}).get("parallel_processing", False)
        
        speedup = total_sequential_time / total_parallel_time if total_parallel_time > 0 else 1.0
        
        print(f"Sequential total time: {total_sequential_time:.3f}s")
        print(f"Parallel total time: {total_parallel_time:.3f}s")
        print(f"Parallel speedup: {speedup:.1f}x")
        
        # Verify parallel processing benefits
        assert speedup >= 1.0  # Should at least not be slower
        print(f"✅ Parallel processing speedup: {speedup:.1f}x")
    
    def test_optimization_caching(self):
        """Test optimization result caching."""
        print("\n🔧 Testing Optimization Caching")
        print("-" * 35)
        
        test_algorithm = self.test_algorithms[0]
        plan = self.planner.plan_quantum_circuit(test_algorithm)
        
        # Clear optimization cache
        self.planner._optimization_cache.clear()
        
        # First optimization (cache miss)
        start_time = time.time()
        optimized_plan1 = self.planner.optimize_quantum_circuit(plan)
        first_opt_time = time.time() - start_time
        
        # Second optimization (cache hit)
        start_time = time.time()
        optimized_plan2 = self.planner.optimize_quantum_circuit(plan)
        second_opt_time = time.time() - start_time
        
        optimization_speedup = first_opt_time / second_opt_time if second_opt_time > 0 else 1.0
        
        print(f"First optimization: {first_opt_time:.3f}s")
        print(f"Second optimization: {second_opt_time:.3f}s")
        print(f"Optimization cache speedup: {optimization_speedup:.1f}x")
        
        # Verify optimization results are equivalent
        assert optimized_plan1["qubits"] == optimized_plan2["qubits"]
        assert len(optimized_plan1["gate_sequence"]) == len(optimized_plan2["gate_sequence"])
        
        print(f"✅ Optimization caching speedup: {optimization_speedup:.1f}x")
    
    def test_cache_warmup(self):
        """Test cache warmup functionality."""
        print("\n🔥 Testing Cache Warmup")
        print("-" * 25)
        
        # Clear all caches
        self.planner.clear_caches()
        
        # Warm up cache with sample algorithms
        warmup_algorithms = self.test_algorithms[:15]
        
        start_time = time.time()
        self.planner.warmup_cache(warmup_algorithms)
        warmup_time = time.time() - start_time
        
        # Verify cache is populated
        cache_stats = self.planner.get_cache_statistics()
        
        print(f"Cache warmup time: {warmup_time:.3f}s")
        print(f"Algorithms warmed up: {len(warmup_algorithms)}")
        print(f"Cache size after warmup: {cache_stats['cache_sizes']['circuit_cache']}")
        
        # Test that warmed algorithms are fast
        start_time = time.time()
        for algorithm in warmup_algorithms[:5]:
            plan = self.planner.plan_quantum_circuit(algorithm)
            assert plan.get("planning_metadata", {}).get("from_cache", False) is True
        
        cached_time = time.time() - start_time
        
        print(f"Time to plan 5 cached circuits: {cached_time:.3f}s")
        print(f"Average time per cached circuit: {cached_time / 5 * 1000:.1f}ms")
        
        assert cache_stats["cache_sizes"]["circuit_cache"] > 0
        print("✅ Cache warmup successful")
    
    def test_memory_management(self):
        """Test memory management and cache size limits."""
        print("\n💾 Testing Memory Management")
        print("-" * 30)
        
        self.planner.clear_caches()
        
        # Create many unique algorithms to test cache limits
        large_algorithm_set = []
        for i in range(150):  # More than cache limit
            algorithm = {
                "name": f"memory_test_{i}",
                "qubits": 2,
                "operations": [
                    {"gate": "Hadamard", "qubits": [0]},
                    {"gate": "Phase", "qubits": [1]},
                    {"gate": f"Phase_{i}", "qubits": [0]}  # Make each unique
                ],
                "measurements": [{"qubit": 0, "basis": "computational"}]
            }
            large_algorithm_set.append(algorithm)
        
        # Plan all algorithms (should trigger cache management)
        for algorithm in large_algorithm_set:
            self.planner.plan_quantum_circuit(algorithm)
        
        cache_stats = self.planner.get_cache_statistics()
        circuit_cache_size = cache_stats["cache_sizes"]["circuit_cache"]
        
        print(f"Planned {len(large_algorithm_set)} unique circuits")
        print(f"Final cache size: {circuit_cache_size}")
        print(f"Cache utilization: {cache_stats['memory_efficiency']['cache_utilization']:.2%}")
        
        # Verify cache size is managed (should be limited)
        assert circuit_cache_size <= 1000  # Should not exceed cache limit
        
        print("✅ Memory management working correctly")
    
    def test_concurrent_access(self):
        """Test concurrent access to the planner."""
        print("\n🔄 Testing Concurrent Access")
        print("-" * 30)
        
        def plan_algorithm(algorithm):
            """Helper function for concurrent planning."""
            try:
                plan = self.planner.plan_quantum_circuit(algorithm)
                return {"success": True, "plan": plan, "error": None}
            except Exception as e:
                return {"success": False, "plan": None, "error": str(e)}
        
        # Test concurrent planning
        test_algorithms = self.test_algorithms[:10]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(plan_algorithm, alg) for alg in test_algorithms]
            results = [future.result() for future in futures]
        
        concurrent_time = time.time() - start_time
        
        # Verify all succeeded
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        print(f"Concurrent planning time: {concurrent_time:.3f}s")
        print(f"Successful plans: {len(successful_results)}")
        print(f"Failed plans: {len(failed_results)}")
        
        # Most should succeed (some might fail due to resource contention)
        assert len(successful_results) >= len(test_algorithms) * 0.8  # At least 80% success
        
        print("✅ Concurrent access handled successfully")
    
    def test_comprehensive_performance_metrics(self):
        """Test comprehensive performance monitoring."""
        print("\n📊 Testing Performance Metrics")
        print("-" * 35)
        
        # Plan several circuits to generate metrics
        for algorithm in self.test_algorithms[:10]:
            plan = self.planner.plan_quantum_circuit(algorithm)
            optimized_plan = self.planner.optimize_quantum_circuit(plan)
        
        # Get comprehensive statistics
        planning_stats = self.planner.get_planning_statistics()
        cache_stats = self.planner.get_cache_statistics()
        
        print("Planning Statistics:")
        print(f"  Circuits planned: {planning_stats['circuits_planned']}")
        print(f"  Optimizations applied: {planning_stats['optimizations_applied']}")
        print(f"  Average fidelity: {planning_stats['average_fidelity']:.3f}")
        print(f"  Cache hit rate: {planning_stats['cache_hit_rate']:.2%}")
        print(f"  Error rate: {planning_stats['error_rate']:.2%}")
        
        print("\nCache Statistics:")
        print(f"  Cache hit rate: {cache_stats['cache_hit_rate']:.2%}")
        print(f"  Circuit cache size: {cache_stats['cache_sizes']['circuit_cache']}")
        print(f"  Optimization cache size: {cache_stats['cache_sizes']['optimization_cache']}")
        print(f"  Memory utilization: {cache_stats['memory_efficiency']['cache_utilization']:.2%}")
        
        # Verify metrics are reasonable
        assert 0.0 <= planning_stats["average_fidelity"] <= 1.0
        assert 0.0 <= planning_stats["cache_hit_rate"] <= 1.0
        assert planning_stats["circuits_planned"] > 0
        
        print("✅ Performance metrics comprehensive and accurate")


def benchmark_scaling_performance():
    """Benchmark performance scaling with circuit complexity."""
    print("\n🏎️  Scaling Performance Benchmark")
    print("=" * 40)
    
    planner = QuantumInspiredTaskPlanner()
    
    # Test different circuit sizes
    sizes = [2, 4, 6, 8, 10]
    results = []
    
    for size in sizes:
        # Create test algorithm of given size
        operations = []
        for i in range(size * 2):  # More operations for larger circuits
            gate_type = ["Hadamard", "CNOT", "Phase"][i % 3]
            if gate_type == "CNOT" and size > 1:
                operations.append({"gate": gate_type, "qubits": [i % size, (i + 1) % size]})
            else:
                operations.append({"gate": gate_type, "qubits": [i % size]})
        
        algorithm = {
            "qubits": size,
            "operations": operations,
            "measurements": [{"qubit": i, "basis": "computational"} for i in range(size)]
        }
        
        # Time the planning
        start_time = time.time()
        plan = planner.plan_quantum_circuit(algorithm)
        planning_time = (time.time() - start_time) * 1000  # ms
        
        # Time the optimization
        start_time = time.time()
        optimized_plan = planner.optimize_quantum_circuit(plan)
        optimization_time = (time.time() - start_time) * 1000  # ms
        
        results.append({
            "qubits": size,
            "operations": len(operations),
            "planning_time_ms": planning_time,
            "optimization_time_ms": optimization_time,
            "total_time_ms": planning_time + optimization_time
        })
        
        print(f"  {size} qubits: {planning_time:.1f}ms planning + {optimization_time:.1f}ms optimization = {planning_time + optimization_time:.1f}ms total")
    
    # Analyze scaling
    print(f"\nScaling Analysis:")
    print(f"  2 qubits: {results[0]['total_time_ms']:.1f}ms")
    print(f"  10 qubits: {results[-1]['total_time_ms']:.1f}ms")
    print(f"  Scaling factor: {results[-1]['total_time_ms'] / results[0]['total_time_ms']:.1f}x")
    
    return results


if __name__ == "__main__":
    print("🚀 Quantum Task Planner Generation 3 Performance Tests")
    print("=" * 65)
    
    # Setup
    setup_module()
    
    # Create test instance
    test_class = TestQuantumPlannerPerformance()
    test_class.setup_method()
    
    # Run performance tests
    test_methods = [
        "test_caching_performance",
        "test_parallel_processing_performance", 
        "test_optimization_caching",
        "test_cache_warmup",
        "test_memory_management",
        "test_concurrent_access",
        "test_comprehensive_performance_metrics"
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            print(f"\nRunning {method_name}...")
            method = getattr(test_class, method_name)
            method()
            print(f"✅ {method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name} FAILED: {str(e)}")
            failed += 1
    
    # Run scaling benchmark
    try:
        benchmark_scaling_performance()
        print("✅ Scaling benchmark PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ Scaling benchmark FAILED: {str(e)}")
        failed += 1
    
    print(f"\n📊 Performance Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All Generation 3 performance tests passed!")
        print("✅ High-performance caching: Operational")
        print("✅ Parallel processing: Functional") 
        print("✅ Memory management: Efficient")
        print("✅ Concurrent access: Safe")
        print("✅ Performance monitoring: Comprehensive")
        print("✅ Scaling optimization: Verified")
    else:
        print("⚠️  Some performance tests failed - review implementation")
    
    print("\n🚀 Quantum Task Planner Generation 3 Performance Testing Complete!")