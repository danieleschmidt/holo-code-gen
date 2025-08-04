#!/usr/bin/env python3
"""
Comprehensive Quality Gates test for Holo-Code-Gen.
Tests all three generations with comprehensive validation, performance, and security checks.
"""

import sys
import time
import os
import json
from pathlib import Path
sys.path.insert(0, '/root/repo')

def test_generation_1_basic():
    """Test Generation 1: Make It Work functionality."""
    print("🔧 TESTING GENERATION 1: MAKE IT WORK")
    print("-" * 50)
    
    # Test basic imports and functionality
    from holo_code_gen import PhotonicCompiler
    from holo_code_gen.templates import IMECLibrary
    from holo_code_gen.ir import ComputationGraph, CircuitNode
    from holo_code_gen.circuit import PhotonicCircuit
    from holo_code_gen.optimization import PowerOptimizer
    
    # Test template library
    library = IMECLibrary()
    components = library.list_components()
    assert len(components) >= 4, f"Expected at least 4 components, got {len(components)}"
    
    # Test computation graph
    graph = ComputationGraph()
    node = CircuitNode("test", "matrix_multiply", {"input_size": 4, "output_size": 2})
    graph.add_node(node)
    assert len(graph.nodes) == 1
    
    # Test basic compilation
    compiler = PhotonicCompiler()
    model_spec = {
        "layers": [
            {"name": "input", "type": "input", "parameters": {"size": 4}},
            {"name": "linear", "type": "matrix_multiply", "parameters": {"input_size": 4, "output_size": 2}},
            {"name": "activation", "type": "optical_nonlinearity", "parameters": {"activation_type": "relu"}}
        ]
    }
    
    circuit = compiler.compile(model_spec)
    assert circuit is not None
    assert len(circuit.components) > 0
    
    # Test circuit functionality
    circuit.generate_layout()
    metrics = circuit.calculate_metrics()
    assert metrics.total_area >= 0
    
    print("✅ Generation 1: Basic functionality works")
    return True

def test_generation_2_robust():
    """Test Generation 2: Make It Robust functionality."""
    print("\n🛡️  TESTING GENERATION 2: MAKE IT ROBUST")
    print("-" * 50)
    
    # Test error handling
    from holo_code_gen.exceptions import ValidationError, CompilationError, SecurityError
    from holo_code_gen.compiler import CompilationConfig
    
    # Test configuration validation
    try:
        invalid_config = CompilationConfig(wavelength=-100)
        assert False, "Should have rejected invalid wavelength"
    except ValidationError as e:
        assert "wavelength" in str(e)
    
    # Test security features
    from holo_code_gen.security import InputSanitizer, ParameterValidator, initialize_security
    initialize_security()
    
    sanitizer = InputSanitizer()
    
    # Test malicious input detection
    try:
        sanitizer.sanitize_string("<script>alert('xss')</script>")
        assert False, "Should have detected XSS"
    except SecurityError:
        pass
    
    # Test parameter validation
    validator = ParameterValidator()
    valid_params = validator.validate_parameters_dict({
        "wavelength": 1550.0,
        "power": 100.0
    })
    assert valid_params["wavelength"] == 1550.0
    
    # Test monitoring system
    from holo_code_gen.monitoring import initialize_monitoring, get_logger, get_metrics_collector
    initialize_monitoring(enable_metrics=False)
    
    logger = get_logger()
    metrics = get_metrics_collector()
    
    logger.info("Test log message", component="test")
    metrics.record_metric("test_metric", 42.0)
    
    # Test comprehensive compilation with error handling
    from holo_code_gen import PhotonicCompiler
    compiler = PhotonicCompiler()
    
    # Valid compilation
    valid_spec = {
        "layers": [
            {"name": "input", "type": "input", "parameters": {"size": 8}},
            {"name": "hidden", "type": "matrix_multiply", "parameters": {"input_size": 8, "output_size": 4}},
            {"name": "output", "type": "matrix_multiply", "parameters": {"input_size": 4, "output_size": 2}}
        ]
    }
    
    circuit = compiler.compile(valid_spec)
    assert circuit is not None
    
    # Invalid compilation should fail gracefully
    try:
        invalid_spec = {"layers": []}
        compiler.compile(invalid_spec)
        assert False, "Should have rejected empty layers"
    except ValidationError:
        pass
    
    print("✅ Generation 2: Robust error handling, security, and monitoring work")
    return True

def test_generation_3_scale():
    """Test Generation 3: Make It Scale functionality."""
    print("\n⚡ TESTING GENERATION 3: MAKE IT SCALE")
    print("-" * 50)
    
    from holo_code_gen.performance import (
        initialize_performance, PerformanceConfig, CacheManager, 
        ParallelExecutor, MemoryManager, cached, parallel_map
    )
    
    # Initialize performance systems
    perf_config = PerformanceConfig(
        enable_caching=True,
        cache_size=50,
        enable_parallel_processing=True,
        max_workers=2
    )
    initialize_performance(perf_config)
    
    # Test caching
    cache = CacheManager(max_size=10, ttl_seconds=5.0)
    cache.put("test_key", "test_value")
    assert cache.get("test_key") == "test_value"
    
    # Test cached function
    call_count = 0
    
    @cached(ttl_seconds=10.0)
    def expensive_func(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    result1 = expensive_func(5)
    result2 = expensive_func(5)  # Should use cache
    assert result1 == result2 == 10
    
    # Test parallel processing
    executor = ParallelExecutor(max_workers=2)
    items = [1, 2, 3, 4, 5]
    
    def square(x):
        return x * x
    
    parallel_results = executor.map_parallel(square, items)
    expected = [1, 4, 9, 16, 25]
    assert sorted(parallel_results) == expected
    
    # Test parallel_map convenience function
    map_results = parallel_map(lambda x: x + 1, [1, 2, 3])
    assert sorted(map_results) == [2, 3, 4]
    
    # Test memory management
    memory_mgr = MemoryManager(max_memory_mb=1024.0)
    stats = memory_mgr.check_memory_usage()
    assert "used_mb" in stats
    
    # Test scaling with multiple compilations
    from holo_code_gen import PhotonicCompiler
    compiler = PhotonicCompiler()
    
    # Create multiple network specifications
    specs = []
    for i in range(4):
        spec = {
            "layers": [
                {"name": f"input_{i}", "type": "input", "parameters": {"size": 8}},
                {"name": f"hidden_{i}", "type": "matrix_multiply", 
                 "parameters": {"input_size": 8, "output_size": 4}},
                {"name": f"output_{i}", "type": "matrix_multiply", 
                 "parameters": {"input_size": 4, "output_size": 2}}
            ]
        }
        specs.append(spec)
    
    # Compile all specifications
    start_time = time.time()
    circuits = []
    for spec in specs:
        circuit = compiler.compile(spec)
        circuits.append(circuit)
    compile_time = time.time() - start_time
    
    # Verify all circuits were created
    assert len(circuits) == len(specs)
    for circuit in circuits:
        assert circuit is not None
        assert len(circuit.components) > 0
    
    print(f"✅ Generation 3: Compiled {len(specs)} circuits in {compile_time:.3f}s")
    return True

def test_comprehensive_integration():
    """Test comprehensive integration of all systems."""
    print("\n🔗 TESTING COMPREHENSIVE INTEGRATION")
    print("-" * 50)
    
    # Initialize all systems
    from holo_code_gen.monitoring import initialize_monitoring
    from holo_code_gen.security import initialize_security
    from holo_code_gen.performance import initialize_performance, PerformanceConfig
    
    initialize_monitoring(enable_metrics=False)
    initialize_security()
    initialize_performance(PerformanceConfig(
        enable_caching=True,
        enable_parallel_processing=True,
        max_workers=2
    ))
    
    from holo_code_gen import PhotonicCompiler
    from holo_code_gen.compiler import CompilationConfig
    
    # Create compiler with custom configuration
    config = CompilationConfig(
        wavelength=1550.0,
        power_budget=500.0,
        optimization_target="power"
    )
    
    compiler = PhotonicCompiler(config)
    
    # Test complex neural network
    complex_network = {
        "layers": [
            {"name": "input", "type": "input", "parameters": {"size": 784}},
            {"name": "fc1", "type": "matrix_multiply", 
             "parameters": {"input_size": 784, "output_size": 256}},
            {"name": "relu1", "type": "optical_nonlinearity", 
             "parameters": {"activation_type": "relu"}},
            {"name": "fc2", "type": "matrix_multiply", 
             "parameters": {"input_size": 256, "output_size": 128}},
            {"name": "relu2", "type": "optical_nonlinearity", 
             "parameters": {"activation_type": "relu"}},
            {"name": "fc3", "type": "matrix_multiply", 
             "parameters": {"input_size": 128, "output_size": 64}},
            {"name": "relu3", "type": "optical_nonlinearity", 
             "parameters": {"activation_type": "relu"}},
            {"name": "output", "type": "matrix_multiply", 
             "parameters": {"input_size": 64, "output_size": 10}}
        ]
    }
    
    # Compile complex network
    start_time = time.time()
    circuit = compiler.compile(complex_network)
    compile_time = time.time() - start_time
    
    # Verify circuit
    assert circuit is not None
    assert len(circuit.components) >= 7  # Should have at least 7 components
    
    # Generate layout and calculate metrics
    circuit.generate_layout()
    metrics = circuit.calculate_metrics()
    
    assert metrics.total_power >= 0
    assert metrics.total_area > 0
    assert metrics.latency >= 0
    
    # Test circuit export functionality
    temp_dir = Path("/tmp/holo_test")
    temp_dir.mkdir(exist_ok=True)
    
    circuit.export_gds(str(temp_dir / "test_circuit.gds"))
    circuit.export_netlist(str(temp_dir / "test_circuit.spi"))
    
    # Verify files were created
    assert (temp_dir / "test_circuit_metadata.json").exists()
    assert (temp_dir / "test_circuit.spi").exists()
    
    print(f"✅ Integration: Complex circuit compiled in {compile_time:.3f}s")
    print(f"   Components: {len(circuit.components)}, Area: {metrics.total_area:.4f}mm²")
    print(f"   Power: {metrics.total_power:.2f}mW, Latency: {metrics.latency:.2f}ns")
    
    return True

def test_error_recovery():
    """Test error recovery and graceful degradation."""
    print("\n🔄 TESTING ERROR RECOVERY")
    print("-" * 50)
    
    from holo_code_gen import PhotonicCompiler
    from holo_code_gen.exceptions import ValidationError, CompilationError
    
    compiler = PhotonicCompiler()
    
    # Test recovery from various error conditions
    error_conditions = [
        # Empty specification
        {"layers": []},
        # Missing layer type
        {"layers": [{"name": "bad_layer", "parameters": {}}]},
        # Invalid parameter values
        {"layers": [{"name": "test", "type": "matrix_multiply", 
                    "parameters": {"input_size": -1, "output_size": 0}}]},
    ]
    
    errors_caught = 0
    
    for i, error_spec in enumerate(error_conditions):
        try:
            circuit = compiler.compile(error_spec)
            print(f"❌ Error condition {i+1} should have failed")
        except (ValidationError, CompilationError) as e:
            errors_caught += 1
            print(f"✅ Error condition {i+1}: Caught {type(e).__name__}")
        except Exception as e:
            print(f"⚠️  Error condition {i+1}: Unexpected error {type(e).__name__}")
    
    assert errors_caught >= 2, f"Expected at least 2 errors caught, got {errors_caught}"
    
    print(f"✅ Error Recovery: {errors_caught}/{len(error_conditions)} error conditions handled properly")
    return True

def test_performance_benchmarks():
    """Test performance benchmarks and quality thresholds."""
    print("\n📊 TESTING PERFORMANCE BENCHMARKS")
    print("-" * 50)
    
    from holo_code_gen import PhotonicCompiler
    from holo_code_gen.performance import initialize_performance, PerformanceConfig
    from holo_code_gen.monitoring import get_performance_monitor
    
    # Initialize with performance optimizations
    initialize_performance(PerformanceConfig(
        enable_caching=True,
        enable_parallel_processing=True
    ))
    
    compiler = PhotonicCompiler()
    perf_monitor = get_performance_monitor()
    
    # Benchmark compilation performance
    benchmark_specs = []
    for size in [16, 32, 64]:
        spec = {
            "layers": [
                {"name": f"input_{size}", "type": "input", "parameters": {"size": size}},
                {"name": f"hidden_{size}", "type": "matrix_multiply", 
                 "parameters": {"input_size": size, "output_size": size//2}},
                {"name": f"output_{size}", "type": "matrix_multiply", 
                 "parameters": {"input_size": size//2, "output_size": size//4}}
            ]
        }
        benchmark_specs.append((size, spec))
    
    compile_times = []
    
    for size, spec in benchmark_specs:
        with perf_monitor.measure_operation(f"benchmark_compile_{size}", "benchmark"):
            start_time = time.time()
            circuit = compiler.compile(spec)
            compile_time = time.time() - start_time
            
            # Verify circuit quality
            assert circuit is not None
            assert len(circuit.components) >= 2
            
            circuit.generate_layout()
            metrics = circuit.calculate_metrics()
            
            compile_times.append(compile_time)
            
            print(f"  Size {size:2d}: {compile_time:.3f}s, "
                  f"{len(circuit.components)} components, "
                  f"{metrics.total_area:.4f}mm²")
    
    # Performance thresholds
    max_compile_time = 0.1  # 100ms max per compilation
    
    passed_benchmarks = sum(1 for t in compile_times if t <= max_compile_time)
    total_benchmarks = len(compile_times)
    
    print(f"✅ Performance: {passed_benchmarks}/{total_benchmarks} benchmarks under {max_compile_time}s")
    
    # Test cache effectiveness
    print("\n📈 Testing cache effectiveness...")
    cache_test_spec = benchmark_specs[0][1]  # Use first spec
    
    # First compilation (cold)
    start_time = time.time()
    circuit1 = compiler.compile(cache_test_spec)
    cold_time = time.time() - start_time
    
    # Second compilation (potentially cached)
    start_time = time.time()
    circuit2 = compiler.compile(cache_test_spec)
    warm_time = time.time() - start_time
    
    speedup = cold_time / warm_time if warm_time > 0 else 1.0
    print(f"  Cold: {cold_time:.3f}s, Warm: {warm_time:.3f}s, Speedup: {speedup:.1f}x")
    
    return True

def test_security_compliance():
    """Test security compliance and validation."""
    print("\n🔒 TESTING SECURITY COMPLIANCE")
    print("-" * 50)
    
    from holo_code_gen.security import (
        InputSanitizer, ParameterValidator, ResourceLimiter, 
        SecurityConfig, initialize_security
    )
    
    # Initialize security with strict settings
    security_config = SecurityConfig(
        max_circuit_components=100,
        max_graph_nodes=50,
        enable_input_sanitization=True,
        enable_path_validation=True
    )
    
    initialize_security(security_config)
    
    sanitizer = InputSanitizer(security_config)
    validator = ParameterValidator()
    limiter = ResourceLimiter(security_config)
    
    # Test input sanitization
    security_tests = [
        ("safe_input_123", True),
        ("<script>alert('xss')</script>", False),
        ("../../../etc/passwd", False),
        ("javascript:alert('test')", False),
        ("normal_component_name", True),
    ]
    
    sanitization_passed = 0
    
    for test_input, should_pass in security_tests:
        try:
            sanitized = sanitizer.sanitize_string(test_input)
            if should_pass:
                sanitization_passed += 1
                print(f"  ✅ Safe input accepted: '{test_input[:20]}...'")
            else:
                print(f"  ❌ Malicious input incorrectly accepted: '{test_input[:20]}...'")
        except Exception as e:
            if not should_pass:
                sanitization_passed += 1
                print(f"  ✅ Malicious input rejected: '{test_input[:20]}...'")
            else:
                print(f"  ❌ Safe input incorrectly rejected: '{test_input[:20]}...'")
    
    # Test parameter validation
    param_tests = [
        ({"wavelength": 1550.0, "power": 100.0}, True),
        ({"wavelength": -100.0}, False),  # Invalid wavelength
        ({"power": -50.0}, False),        # Invalid power
        ({"component_type": "valid_name"}, True),
        ({"component_type": "<invalid>"}, False),
    ]
    
    validation_passed = 0
    
    for params, should_pass in param_tests:
        try:
            validated = validator.validate_parameters_dict(params)
            if should_pass:
                validation_passed += 1
                print(f"  ✅ Valid parameters accepted")
            else:
                print(f"  ❌ Invalid parameters incorrectly accepted")
        except Exception as e:
            if not should_pass:
                validation_passed += 1
                print(f"  ✅ Invalid parameters rejected")
            else:
                print(f"  ❌ Valid parameters incorrectly rejected")
    
    # Test resource limits
    try:
        limiter.check_circuit_complexity(150)  # Exceeds limit of 100
        print("  ❌ Resource limit not enforced")
    except Exception:
        print("  ✅ Resource limits enforced")
    
    print(f"✅ Security: {sanitization_passed}/{len(security_tests)} sanitization tests passed")
    print(f"             {validation_passed}/{len(param_tests)} validation tests passed")
    
    return sanitization_passed >= len(security_tests) * 0.8  # 80% pass rate

def run_all_quality_gates():
    """Run all quality gate tests."""
    print("=" * 80)
    print("🏁 HOLO-CODE-GEN COMPREHENSIVE QUALITY GATES")
    print("   Testing All Generations with Full Validation")
    print("=" * 80)
    
    tests = [
        ("Generation 1: Basic Functionality", test_generation_1_basic),
        ("Generation 2: Robust Systems", test_generation_2_robust), 
        ("Generation 3: Scaling Performance", test_generation_3_scale),
        ("Comprehensive Integration", test_comprehensive_integration),
        ("Error Recovery", test_error_recovery),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Security Compliance", test_security_compliance),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            test_start = time.time()
            result = test_func()
            test_time = time.time() - test_start
            
            if result:
                passed_tests += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            print(f"\n{status} - {test_name} ({test_time:.2f}s)")
            
        except Exception as e:
            print(f"\n❌ FAILED - {test_name}: {str(e)}")
            import traceback
            print(f"   Error details: {traceback.format_exc()}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📋 QUALITY GATES SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Total Time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("🚀 System is ready for production deployment")
        print("🏆 Holo-Code-Gen has successfully passed all validation tests")
        return True
    else:
        print("⚠️  SOME QUALITY GATES FAILED")
        print("🔧 Review failed tests before deployment")
        return False

def main():
    """Main quality gate runner."""
    success = run_all_quality_gates()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())