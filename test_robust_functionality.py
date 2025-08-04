#!/usr/bin/env python3
"""
Robust functionality test for Holo-Code-Gen Generation 2.
Tests error handling, security, monitoring, and validation.
"""

import sys
sys.path.insert(0, '/root/repo')

def test_error_handling():
    """Test comprehensive error handling."""
    print("Testing error handling...")
    
    from holo_code_gen.exceptions import (
        ValidationError, CompilationError, SecurityError, 
        validate_positive, validate_range, ErrorCodes
    )
    
    # Test validation functions
    try:
        validate_positive(-1, "test_param")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.error_code == ErrorCodes.INVALID_PARAMETER_VALUE
        print("✓ Negative value validation works")
    
    try:
        validate_range(150, 0, 100, "test_range")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.error_code == ErrorCodes.PARAMETER_OUT_OF_RANGE
        print("✓ Range validation works")
    
    # Test custom exception features
    try:
        raise CompilationError(
            "Test compilation error",
            error_code="TEST_ERROR",
            context={"component": "test"}
        )
    except CompilationError as e:
        assert e.error_code == "TEST_ERROR"
        assert e.context["component"] == "test"
        error_dict = e.to_dict()
        assert error_dict["error_type"] == "CompilationError"
        print("✓ Custom exception serialization works")

def test_security_features():
    """Test security validation and sanitization."""
    print("\nTesting security features...")
    
    from holo_code_gen.security import (
        InputSanitizer, ParameterValidator, ResourceLimiter,
        SecurityConfig, SecurityError
    )
    from holo_code_gen.exceptions import ValidationError
    
    # Test input sanitization
    sanitizer = InputSanitizer()
    
    # Test safe input
    safe_input = sanitizer.sanitize_string("valid_input_123")
    assert safe_input == "valid_input_123"
    print("✓ Safe input sanitization works")
    
    # Test malicious input detection
    try:
        sanitizer.sanitize_string("<script>alert('xss')</script>")
        assert False, "Should have detected malicious input"
    except SecurityError:
        print("✓ Malicious input detection works")
    
    # Test filename sanitization
    safe_filename = sanitizer.sanitize_filename("test_file.gds")
    assert safe_filename == "test_file.gds"
    print("✓ Safe filename validation works")
    
    try:
        sanitizer.sanitize_filename("../../../etc/passwd")
        assert False, "Should have detected path traversal"
    except SecurityError:
        print("✓ Path traversal detection works")
    
    # Test parameter validation
    validator = ParameterValidator()
    
    # Valid parameters
    valid_params = validator.validate_parameters_dict({
        "wavelength": 1550.0,
        "power": 100.0,
        "component_type": "microring_resonator"
    })
    assert valid_params["wavelength"] == 1550.0
    print("✓ Parameter validation works")
    
    # Invalid parameter range
    try:
        validator.validate_parameter("wavelength", 5000.0)  # Out of range
        assert False, "Should have rejected out-of-range wavelength"
    except ValidationError:
        print("✓ Parameter range validation works")

def test_monitoring_system():
    """Test monitoring and logging functionality."""
    print("\nTesting monitoring system...")
    
    from holo_code_gen.monitoring import (
        MetricsCollector, StructuredLogger, PerformanceMonitor,
        initialize_monitoring
    )
    import time
    
    # Initialize monitoring
    initialize_monitoring(enable_metrics=False)  # Disable file export for tests
    
    # Test metrics collection
    metrics = MetricsCollector(enable_export=False)
    metrics.record_metric("test_metric", 42.0, {"tag": "test"}, "count")
    metrics.increment_counter("test_counter", {"component": "test"})
    
    collected = metrics.get_metrics()
    assert len(collected) == 2
    assert collected[0].name == "test_metric"
    assert collected[0].value == 42.0
    print("✓ Metrics collection works")
    
    # Test structured logging
    logger = StructuredLogger("test_logger")
    logger.info("Test message", component="test", operation="test_op")
    logger.error("Test error", component="test", error="test_error")
    
    events = logger.get_events()
    assert len(events) >= 2
    print("✓ Structured logging works")
    
    # Test performance monitoring
    perf_monitor = PerformanceMonitor(metrics, logger)
    
    with perf_monitor.measure_operation("test_operation", "test_component"):
        time.sleep(0.01)  # Simulate work
    
    # Check that timing metrics were recorded
    timing_metrics = [m for m in metrics.get_metrics() if "duration" in m.name]
    assert len(timing_metrics) > 0
    print("✓ Performance monitoring works")

def test_robust_compilation():
    """Test robust compilation with error handling."""
    print("\nTesting robust compilation...")
    
    from holo_code_gen import PhotonicCompiler  
    from holo_code_gen.compiler import CompilationConfig
    from holo_code_gen.exceptions import ValidationError
    
    # Test configuration validation
    try:
        config = CompilationConfig(wavelength=-100)  # Invalid
        assert False, "Should have rejected negative wavelength"
    except ValidationError:
        print("✓ Configuration validation works")
    
    # Test valid configuration
    config = CompilationConfig(
        wavelength=1550.0,
        power_budget=500.0,
        template_library="imec_v2025_07"
    )
    
    compiler = PhotonicCompiler(config)
    print("✓ Compiler initialization with validation works")
    
    # Test dict-based compilation (since PyTorch might not be available)
    model_spec = {
        "layers": [
            {
                "name": "input",
                "type": "input",
                "parameters": {"size": 4},
                "output_shape": (4,)
            },
            {
                "name": "linear1",
                "type": "matrix_multiply",
                "parameters": {"input_size": 4, "output_size": 2},
                "input_shape": (4,),
                "output_shape": (2,)
            },
            {
                "name": "activation",
                "type": "optical_nonlinearity",
                "parameters": {"activation_type": "relu"},
                "input_shape": (2,),
                "output_shape": (2,)
            }
        ]
    }
    
    # This should work without PyTorch
    circuit = compiler.compile(model_spec)
    assert circuit is not None
    print("✓ Dict-based compilation works")
    
    # Test invalid model specification
    try:
        invalid_spec = {"layers": []}  # Empty layers
        compiler.compile(invalid_spec)
        assert False, "Should have rejected empty layers"
    except ValidationError:
        print("✓ Model specification validation works")

def test_resource_limits():
    """Test resource limit enforcement."""
    print("\nTesting resource limits...")
    
    from holo_code_gen.security import ResourceLimiter, SecurityConfig, SecurityError
    from pathlib import Path
    
    # Create test config with low limits
    config = SecurityConfig(
        max_circuit_components=5,
        max_graph_nodes=3
    )
    
    limiter = ResourceLimiter(config)
    
    # Test within limits
    limiter.check_circuit_complexity(3)
    limiter.check_graph_complexity(2)
    print("✓ Resource limits allow valid operations")
    
    # Test exceeding limits
    try:
        limiter.check_circuit_complexity(10)  # Exceeds limit of 5
        assert False, "Should have rejected too many components"
    except SecurityError:
        print("✓ Circuit complexity limits enforced")
    
    try:
        limiter.check_graph_complexity(5)  # Exceeds limit of 3
        assert False, "Should have rejected too many nodes"
    except SecurityError:
        print("✓ Graph complexity limits enforced")

def test_health_monitoring():
    """Test health check system."""
    print("\nTesting health monitoring...")
    
    from holo_code_gen.monitoring import HealthChecker, StructuredLogger
    
    logger = StructuredLogger("health_test")
    health_checker = HealthChecker(logger)
    
    # Register test health checks
    def healthy_check():
        return True
    
    def unhealthy_check():
        return False
    
    def error_check():
        raise Exception("Health check error")
    
    health_checker.register_health_check("healthy", healthy_check, "Always healthy")
    health_checker.register_health_check("unhealthy", unhealthy_check, "Always unhealthy")
    health_checker.register_health_check("error", error_check, "Always errors")
    
    # Run health checks
    results = health_checker.run_health_checks()
    
    assert not results["overall_healthy"]  # Should be unhealthy due to failures
    assert results["checks"]["healthy"]["status"] == "healthy"
    assert results["checks"]["unhealthy"]["status"] == "unhealthy"
    assert results["checks"]["error"]["status"] == "error"
    
    print("✓ Health check system works")

def test_comprehensive_integration():
    """Test comprehensive integration of all robust features."""
    print("\nTesting comprehensive integration...")
    
    from holo_code_gen import PhotonicCompiler
    from holo_code_gen.monitoring import initialize_monitoring
    from holo_code_gen.security import initialize_security
    
    # Initialize all systems
    initialize_monitoring(enable_metrics=False)
    initialize_security()
    
    # Create a realistic compilation scenario
    compiler = PhotonicCompiler()
    
    # Complex model specification
    complex_model = {
        "layers": [
            {"name": "input", "type": "input", "parameters": {"size": 784}},
            {"name": "fc1", "type": "matrix_multiply", 
             "parameters": {"input_size": 784, "output_size": 128}},
            {"name": "relu1", "type": "optical_nonlinearity", 
             "parameters": {"activation_type": "relu"}},
            {"name": "fc2", "type": "matrix_multiply", 
             "parameters": {"input_size": 128, "output_size": 64}},
            {"name": "relu2", "type": "optical_nonlinearity", 
             "parameters": {"activation_type": "relu"}},
            {"name": "output", "type": "matrix_multiply", 
             "parameters": {"input_size": 64, "output_size": 10}}
        ]
    }
    
    # Compile with full monitoring and security
    circuit = compiler.compile(complex_model)
    
    # Verify circuit was created successfully
    assert circuit is not None
    assert len(circuit.components) > 0
    
    # Generate layout and metrics with monitoring
    circuit.generate_layout()
    metrics = circuit.calculate_metrics()
    
    assert metrics.total_area > 0
    print(f"✓ Generated circuit: {len(circuit.components)} components, {metrics.total_area:.4f}mm²")
    
    print("✓ Comprehensive integration test passed")

def main():
    """Run all robust functionality tests."""
    print("=" * 70)
    print("HOLO-CODE-GEN GENERATION 2 ROBUST FUNCTIONALITY TEST")
    print("=" * 70)
    
    try:
        test_error_handling()
        test_security_features()
        test_monitoring_system()
        test_robust_compilation()
        test_resource_limits()
        test_health_monitoring()
        test_comprehensive_integration()
        
        print("\n" + "=" * 70)
        print("🎉 ALL ROBUST TESTS PASSED - GENERATION 2 IMPLEMENTATION COMPLETE!")
        print("🛡️  Security: Input sanitization, parameter validation, resource limits")
        print("📊 Monitoring: Metrics collection, structured logging, performance tracking")
        print("🚨 Error Handling: Comprehensive exceptions, validation, recovery")
        print("🔍 Health Checks: System monitoring, audit logging, diagnostics")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ROBUST TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())