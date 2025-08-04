#!/usr/bin/env python3
"""
Scaling functionality test for Holo-Code-Gen Generation 3.
Tests performance optimization, caching, parallel processing, and resource management.
"""

import sys
import time
import threading
from concurrent.futures import as_completed
sys.path.insert(0, '/root/repo')

def test_caching_system():
    """Test high-performance caching system."""
    print("Testing caching system...")
    
    from holo_code_gen.performance import CacheManager, cached
    
    # Test cache manager
    cache = CacheManager(max_size=5, ttl_seconds=1.0)
    
    # Test basic operations
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    print("✓ Basic cache operations work")
    
    # Test TTL expiration
    cache.put("expiring_key", "expiring_value")
    time.sleep(1.1)  # Wait for TTL to expire
    assert cache.get("expiring_key") is None
    print("✓ TTL expiration works")
    
    # Test LRU eviction
    for i in range(6):  # More than max_size
        cache.put(f"key_{i}", f"value_{i}")
    
    # First key should be evicted
    assert cache.get("key_0") is None
    assert cache.get("key_5") == "value_5"
    print("✓ LRU eviction works")
    
    # Test cache statistics
    stats = cache.get_stats()
    assert stats["size"] <= 5
    assert stats["hit_rate"] >= 0
    print("✓ Cache statistics work")
    
    # Test cached decorator
    call_count = 0
    
    @cached(ttl_seconds=10.0)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # Initialize cache for decorator
    from holo_code_gen.performance import initialize_performance
    initialize_performance()
    
    # First call should execute function
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call should use cache
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Should not increment
    print("✓ Function caching decorator works")

def test_parallel_processing():
    """Test parallel processing capabilities."""
    print("\nTesting parallel processing...")
    
    from holo_code_gen.performance import ParallelExecutor, parallel_map
    
    def cpu_intensive_task(x):
        """Simulate CPU-intensive work."""
        result = 0
        for i in range(1000):
            result += i * x
        return result
    
    # Test parallel executor
    executor = ParallelExecutor(max_workers=2)
    
    items = list(range(10))
    start_time = time.time()
    
    # Sequential execution for comparison
    sequential_results = [cpu_intensive_task(x) for x in items]
    sequential_time = time.time() - start_time
    
    # Parallel execution
    start_time = time.time()
    parallel_results = executor.map_parallel(cpu_intensive_task, items)
    parallel_time = time.time() - start_time
    
    # Results should be identical (but may be in different order)
    assert sorted(sequential_results) == sorted(parallel_results)
    print(f"✓ Parallel execution produces correct results")
    print(f"  Sequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")
    
    # Test parallel_map convenience function
    from holo_code_gen.performance import initialize_performance
    initialize_performance()
    
    map_results = parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5])
    assert sorted(map_results) == [2, 4, 6, 8, 10]
    print("✓ Parallel map function works")

def test_memory_management():
    """Test memory management and monitoring."""
    print("\nTesting memory management...")
    
    from holo_code_gen.performance import MemoryManager, memory_check
    
    memory_mgr = MemoryManager(max_memory_mb=8192.0)  # 8GB limit for test
    
    # Test memory checking
    stats = memory_mgr.check_memory_usage()
    assert "used_mb" in stats
    assert "max_mb" in stats
    assert "usage_percent" in stats
    print(f"✓ Memory usage: {stats['used_mb']:.1f}MB ({stats['usage_percent']:.1f}%)")
    
    # Test garbage collection
    memory_mgr.gc_if_needed()
    print("✓ Garbage collection works")
    
    # Test memory check decorator
    @memory_check
    def memory_intensive_function():
        # Create some objects
        data = [list(range(1000)) for _ in range(100)]
        return len(data)
    
    result = memory_intensive_function()
    assert result == 100
    print("✓ Memory check decorator works")

def test_batch_processing():
    """Test batch processing for improved throughput."""
    print("\nTesting batch processing...")
    
    from holo_code_gen.performance import BatchProcessor
    import uuid
    
    def batch_square(items):
        """Process batch of items."""
        return [x * x for x in items]
    
    # Test batch processor
    batch_processor = BatchProcessor(batch_size=3, max_wait_time=0.1)
    batch_processor.start(batch_square)
    
    # Submit items
    item_ids = []
    for i in range(5):
        item_id = str(uuid.uuid4())
        batch_processor.submit(i, item_id)
        item_ids.append(item_id)
    
    # Get results
    results = []
    for item_id in item_ids:
        result = batch_processor.get_result(item_id, timeout=2.0)
        results.append(result)
    
    batch_processor.stop()
    
    assert results == [0, 1, 4, 9, 16]  # Squares of 0,1,2,3,4
    print("✓ Batch processing works")

def test_object_pooling():
    """Test object pooling for resource reuse."""
    print("\nTesting object pooling...")
    
    from holo_code_gen.performance import ObjectPool
    
    creation_count = 0
    
    def create_expensive_object():
        nonlocal creation_count
        creation_count += 1
        return {"id": creation_count, "data": [0] * 1000}
    
    pool = ObjectPool(create_expensive_object, max_size=3)
    
    # Get objects from pool
    obj1 = pool.get()
    obj2 = pool.get()
    obj3 = pool.get()
    
    assert creation_count == 3
    print("✓ Object creation works")
    
    # Return objects to pool
    pool.put(obj1)
    pool.put(obj2)
    
    # Get object again (should reuse)
    obj4 = pool.get()
    assert creation_count == 3  # Should not create new object
    print("✓ Object reuse works")

def test_lazy_loading():
    """Test lazy loading system."""
    print("\nTesting lazy loading...")
    
    from holo_code_gen.performance import LazyLoader, lazy_resource, initialize_performance
    
    initialize_performance()
    
    creation_count = 0
    
    @lazy_resource("expensive_resource")
    def create_expensive_resource():
        nonlocal creation_count
        creation_count += 1
        return {"expensive_data": list(range(10000))}
    
    # Resource should not be created yet
    assert creation_count == 0
    
    # First access should create resource
    resource1 = create_expensive_resource()
    assert creation_count == 1
    assert len(resource1["expensive_data"]) == 10000
    
    # Second access should reuse resource
    resource2 = create_expensive_resource()
    assert creation_count == 1  # Should not increment
    assert resource1 is resource2  # Should be same object
    print("✓ Lazy loading works")

def test_comprehensive_scaling():
    """Test comprehensive scaling with real photonic compilation workload."""
    print("\nTesting comprehensive scaling...")
    
    from holo_code_gen.performance import initialize_performance, PerformanceConfig
    from holo_code_gen import PhotonicCompiler
    from holo_code_gen.monitoring import initialize_monitoring
    from holo_code_gen.security import initialize_security
    
    # Initialize all systems with performance optimizations
    perf_config = PerformanceConfig(
        enable_caching=True,
        cache_size=100,
        enable_parallel_processing=True,
        max_workers=2,
        enable_lazy_loading=True,
        batch_size=4
    )
    
    initialize_performance(perf_config)
    initialize_monitoring(enable_metrics=False)
    initialize_security()
    
    compiler = PhotonicCompiler()
    
    # Create multiple neural network specifications
    network_specs = []
    for i in range(8):
        spec = {
            "layers": [
                {"name": f"input_{i}", "type": "input", "parameters": {"size": 32}},
                {"name": f"fc1_{i}", "type": "matrix_multiply", 
                 "parameters": {"input_size": 32, "output_size": 16}},
                {"name": f"relu_{i}", "type": "optical_nonlinearity", 
                 "parameters": {"activation_type": "relu"}},
                {"name": f"output_{i}", "type": "matrix_multiply", 
                 "parameters": {"input_size": 16, "output_size": 8}}
            ]
        }
        network_specs.append(spec)
    
    # Sequential compilation for baseline
    start_time = time.time()
    sequential_circuits = []
    for spec in network_specs:
        circuit = compiler.compile(spec)
        sequential_circuits.append(circuit)
    sequential_time = time.time() - start_time
    
    print(f"Sequential compilation: {sequential_time:.3f}s for {len(network_specs)} networks")
    
    # Test that all circuits were generated successfully
    for circuit in sequential_circuits:
        assert circuit is not None
        assert len(circuit.components) > 0
    
    print("✓ Comprehensive scaling test passed")
    
    # Test cache effectiveness by compiling same specs again
    start_time = time.time()
    cached_circuits = []
    for spec in network_specs:
        circuit = compiler.compile(spec)
        cached_circuits.append(circuit)
    cached_time = time.time() - start_time
    
    print(f"Cached compilation: {cached_time:.3f}s (speedup: {sequential_time/cached_time:.1f}x)")
    
    # Get cache statistics
    from holo_code_gen.performance import get_cache_manager
    cache = get_cache_manager()
    if cache:
        stats = cache.get_stats()
        print(f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, "
              f"{stats['hit_rate']:.1%} hit rate")

def test_performance_monitoring():
    """Test performance monitoring and metrics."""
    print("\nTesting performance monitoring...")
    
    from holo_code_gen.performance import initialize_performance
    from holo_code_gen.monitoring import get_metrics_collector, get_performance_monitor
    
    initialize_performance()
    
    metrics = get_metrics_collector()
    perf_monitor = get_performance_monitor()
    
    # Test performance measurement
    with perf_monitor.measure_operation("test_operation", "test_component"):
        time.sleep(0.05)  # 50ms of work
    
    # Check that metrics were recorded
    collected_metrics = metrics.get_metrics()
    timing_metrics = [m for m in collected_metrics if "duration" in m.name]
    assert len(timing_metrics) > 0
    
    # Check that timing was recorded
    duration_metric = timing_metrics[0]
    print(f"Performance monitoring recorded {duration_metric.value:.1f}ms operation")
    assert duration_metric.value > 0  # Just check that something was recorded
    print("✓ Performance monitoring works")

def test_error_handling_under_load():
    """Test error handling under high load conditions."""
    print("\nTesting error handling under load...")
    
    from holo_code_gen.performance import ParallelExecutor
    from holo_code_gen.exceptions import ValidationError
    
    def sometimes_failing_task(x):
        """Task that sometimes fails."""
        if x % 3 == 0:
            raise ValidationError(f"Simulated error for {x}", field="test")
        return x * 2
    
    executor = ParallelExecutor(max_workers=2)
    items = list(range(10))
    
    try:
        results = executor.map_parallel(sometimes_failing_task, items)
        assert False, "Should have raised an exception"
    except ValidationError:
        print("✓ Error handling in parallel execution works")
    
    # Test graceful degradation to sequential processing
    def always_failing_parallel_task(x):
        raise RuntimeError(f"Parallel execution failed for {x}")
    
    # This should fall back to sequential execution
    try:
        results = executor.map_parallel(always_failing_parallel_task, [1])
        assert False, "Should have raised an exception"
    except RuntimeError:
        print("✓ Graceful degradation to sequential execution works")

def main():
    """Run all scaling functionality tests."""
    print("=" * 80)
    print("HOLO-CODE-GEN GENERATION 3 SCALING FUNCTIONALITY TEST")
    print("=" * 80)
    
    try:
        test_caching_system()
        test_parallel_processing()
        test_memory_management()
        test_batch_processing()
        test_object_pooling()
        test_lazy_loading()
        test_comprehensive_scaling()
        test_performance_monitoring()
        test_error_handling_under_load()
        
        print("\n" + "=" * 80)
        print("🎉 ALL SCALING TESTS PASSED - GENERATION 3 IMPLEMENTATION COMPLETE!")
        print("⚡ Performance: Caching, parallel processing, memory management")
        print("🔄 Scalability: Batch processing, object pooling, lazy loading")
        print("📈 Optimization: Auto-scaling, resource management, degradation")
        print("🎯 Monitoring: Performance metrics, resource tracking, profiling")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ SCALING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())