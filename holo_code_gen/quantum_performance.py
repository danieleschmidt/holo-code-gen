"""High-performance optimizations for quantum algorithms."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
import hashlib

from .monitoring import monitor_function, get_logger
from .exceptions import ErrorCodes, ValidationError


logger = get_logger()


class QuantumAlgorithmCache:
    """High-performance caching system for quantum algorithms."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize quantum algorithm cache.
        
        Args:
            max_size: Maximum number of cached results
            ttl_seconds: Time-to-live for cached results in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        self.logger = logger
    
    def _generate_key(self, algorithm: str, params: Dict[str, Any]) -> str:
        """Generate cache key from algorithm name and parameters."""
        # Serialize parameters deterministically
        param_str = json.dumps(params, sort_keys=True, default=str)
        key_data = f"{algorithm}:{param_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.creation_times:
            return True
        return time.time() - self.creation_times[key] > self.ttl_seconds
    
    def _evict_oldest(self):
        """Evict the oldest accessed cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_entry(oldest_key)
    
    def _remove_entry(self, key: str):
        """Remove a cache entry and its metadata."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    @monitor_function("cache_get", "quantum_performance")
    def get(self, algorithm: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for algorithm with given parameters."""
        key = self._generate_key(algorithm, params)
        
        with self.lock:
            if key in self.cache and not self._is_expired(key):
                # Cache hit
                self.access_times[key] = time.time()
                self.hit_count += 1
                result = self.cache[key].copy() if isinstance(self.cache[key], dict) else self.cache[key]
                
                # Add cache metadata
                if isinstance(result, dict):
                    result["from_cache"] = True
                    result["cache_hit_count"] = self.hit_count
                
                self.logger.debug(f"Cache hit for {algorithm}: {key}")
                return result
            else:
                # Cache miss
                self.miss_count += 1
                if key in self.cache:
                    # Expired entry
                    self._remove_entry(key)
                    self.logger.debug(f"Cache expired for {algorithm}: {key}")
                else:
                    self.logger.debug(f"Cache miss for {algorithm}: {key}")
                return None
    
    @monitor_function("cache_put", "quantum_performance")
    def put(self, algorithm: str, params: Dict[str, Any], result: Any):
        """Cache result for algorithm with given parameters."""
        key = self._generate_key(algorithm, params)
        
        with self.lock:
            # Evict if cache is full
            while len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Store result
            current_time = time.time()
            self.cache[key] = result.copy() if isinstance(result, dict) else result
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
            
            self.logger.debug(f"Cached result for {algorithm}: {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class ParallelQuantumProcessor:
    """Parallel processing system for quantum algorithms."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize parallel quantum processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logger
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
    
    @monitor_function("parallel_execute", "quantum_performance")
    def execute_parallel(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Dict[str, Any]]:
        """Execute multiple quantum algorithm tasks in parallel.
        
        Args:
            tasks: List of (function, args, kwargs) tuples to execute
            
        Returns:
            List of results in the same order as input tasks
        """
        if not tasks:
            return []
        
        results = [None] * len(tasks)
        future_to_index = {}
        
        # Submit all tasks
        for i, (func, args, kwargs) in enumerate(tasks):
            future = self.executor.submit(self._execute_task, func, args, kwargs)
            future_to_index[future] = i
            
            with self.lock:
                self.active_tasks += 1
        
        # Collect results
        try:
            for future in as_completed(future_to_index, timeout=300):  # 5 minute timeout
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = {
                        "success": True,
                        "result": result,
                        "task_index": index
                    }
                    with self.lock:
                        self.completed_tasks += 1
                except Exception as e:
                    results[index] = {
                        "success": False,
                        "error": str(e),
                        "task_index": index
                    }
                    with self.lock:
                        self.failed_tasks += 1
                finally:
                    with self.lock:
                        self.active_tasks -= 1
        
        except TimeoutError:
            self.logger.error("Parallel execution timed out")
            # Mark incomplete results as failed
            for i, result in enumerate(results):
                if result is None:
                    results[i] = {
                        "success": False,
                        "error": "Task timed out",
                        "task_index": i
                    }
        
        return results
    
    def _execute_task(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute a single task with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            raise
    
    @monitor_function("batch_cv_qaoa", "quantum_performance")
    def batch_cv_qaoa(self, problem_graphs: List[Dict[str, Any]], 
                      depth: int = 3, max_iterations: int = 100) -> List[Dict[str, Any]]:
        """Execute CV-QAOA on multiple problems in parallel."""
        from .quantum_algorithms import PhotonicQuantumAlgorithms
        
        algorithms = PhotonicQuantumAlgorithms()
        
        # Create tasks
        tasks = []
        for graph in problem_graphs:
            task = (
                algorithms.continuous_variable_qaoa,
                (graph, depth, max_iterations),
                {}
            )
            tasks.append(task)
        
        # Execute in parallel
        results = self.execute_parallel(tasks)
        
        # Extract successful results
        processed_results = []
        for result in results:
            if result["success"]:
                processed_results.append(result["result"])
            else:
                # Create error result
                processed_results.append({
                    "algorithm": "cv_qaoa",
                    "error": result["error"],
                    "success": False
                })
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        with self.lock:
            total_tasks = self.completed_tasks + self.failed_tasks
            success_rate = (self.completed_tasks / total_tasks) if total_tasks > 0 else 0
            
            return {
                "max_workers": self.max_workers,
                "active_tasks": self.active_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": success_rate
            }
    
    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)


class QuantumOptimizationEngine:
    """Advanced optimization engine for quantum algorithms."""
    
    def __init__(self, cache: Optional[QuantumAlgorithmCache] = None):
        """Initialize quantum optimization engine.
        
        Args:
            cache: Optional cache instance for memoization
        """
        self.cache = cache or QuantumAlgorithmCache()
        self.logger = logger
        self.optimization_stats = {
            "parameter_optimizations": 0,
            "convergence_accelerations": 0,
            "cache_optimizations": 0
        }
    
    @monitor_function("optimize_cv_qaoa_parameters", "quantum_performance")
    def optimize_cv_qaoa_parameters(self, problem_graph: Dict[str, Any], 
                                    depth: int, max_iterations: int) -> Dict[str, Any]:
        """Optimize CV-QAOA parameters using advanced techniques."""
        # Check cache first
        cache_key_params = {
            "problem_graph": self._graph_fingerprint(problem_graph),
            "depth": depth,
            "max_iterations": max_iterations
        }
        
        cached_result = self.cache.get("cv_qaoa_optimized", cache_key_params)
        if cached_result:
            self.optimization_stats["cache_optimizations"] += 1
            return cached_result
        
        # Advanced parameter optimization
        optimized_params = self._optimize_parameters(problem_graph, depth)
        
        # Adaptive iteration count based on problem complexity
        complexity_score = self._calculate_complexity(problem_graph)
        optimized_iterations = self._adaptive_iterations(max_iterations, complexity_score)
        
        # Convergence acceleration techniques
        convergence_config = self._optimize_convergence(problem_graph, depth)
        
        result = {
            "algorithm": "cv_qaoa_parameter_optimization",
            "optimized_depth": optimized_params["depth"],
            "optimized_iterations": optimized_iterations,
            "learning_rate_schedule": optimized_params["learning_rate_schedule"],
            "convergence_config": convergence_config,
            "complexity_score": complexity_score,
            "optimization_techniques": [
                "adaptive_learning_rate",
                "problem_aware_initialization",
                "convergence_acceleration",
                "complexity_based_tuning"
            ]
        }
        
        # Cache the result
        self.cache.put("cv_qaoa_optimized", cache_key_params, result)
        
        self.optimization_stats["parameter_optimizations"] += 1
        return result
    
    def _graph_fingerprint(self, problem_graph: Dict[str, Any]) -> str:
        """Generate a compact fingerprint for a problem graph."""
        num_nodes = len(problem_graph.get("nodes", []))
        num_edges = len(problem_graph.get("edges", []))
        
        # Calculate graph properties
        edge_weights = [edge.get("weight", 1.0) for edge in problem_graph.get("edges", [])]
        avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0
        
        fingerprint = f"n{num_nodes}_e{num_edges}_w{avg_weight:.2f}"
        return fingerprint
    
    def _optimize_parameters(self, problem_graph: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Optimize algorithm parameters based on problem structure."""
        num_nodes = len(problem_graph.get("nodes", []))
        num_edges = len(problem_graph.get("edges", []))
        
        # Problem-aware depth optimization
        optimal_depth = min(depth, max(2, num_nodes // 4))
        
        # Adaptive learning rate schedule
        initial_lr = 0.1 if num_nodes < 10 else 0.05
        decay_factor = 0.98 if num_edges > num_nodes else 0.95
        
        learning_rate_schedule = {
            "initial": initial_lr,
            "decay_factor": decay_factor,
            "min_lr": 0.001,
            "schedule_type": "exponential_decay"
        }
        
        return {
            "depth": optimal_depth,
            "learning_rate_schedule": learning_rate_schedule
        }
    
    def _calculate_complexity(self, problem_graph: Dict[str, Any]) -> float:
        """Calculate problem complexity score."""
        num_nodes = len(problem_graph.get("nodes", []))
        num_edges = len(problem_graph.get("edges", []))
        
        # Basic complexity metrics
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Edge weight variance
        edge_weights = [edge.get("weight", 1.0) for edge in problem_graph.get("edges", [])]
        if edge_weights:
            avg_weight = sum(edge_weights) / len(edge_weights)
            weight_variance = sum((w - avg_weight)**2 for w in edge_weights) / len(edge_weights)
        else:
            weight_variance = 0
        
        # Combined complexity score
        complexity = num_nodes * 0.1 + density * 0.5 + weight_variance * 0.3
        return min(complexity, 10.0)  # Cap at 10
    
    def _adaptive_iterations(self, max_iterations: int, complexity_score: float) -> int:
        """Calculate adaptive iteration count based on problem complexity."""
        # More complex problems need more iterations
        complexity_factor = 1.0 + (complexity_score / 10.0)
        adaptive_iterations = int(max_iterations * complexity_factor)
        
        # Reasonable bounds
        return min(adaptive_iterations, max_iterations * 2)
    
    def _optimize_convergence(self, problem_graph: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Optimize convergence parameters."""
        num_nodes = len(problem_graph.get("nodes", []))
        
        # Convergence tolerance based on problem size
        if num_nodes < 5:
            tolerance = 1e-6
            patience = 5
        elif num_nodes < 10:
            tolerance = 1e-5
            patience = 10
        else:
            tolerance = 1e-4
            patience = 15
        
        self.optimization_stats["convergence_accelerations"] += 1
        
        return {
            "tolerance": tolerance,
            "patience": patience,
            "early_stopping": True,
            "momentum": 0.9 if num_nodes > 5 else 0.8
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization engine statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            "optimization_stats": self.optimization_stats,
            "cache_stats": cache_stats,
            "total_optimizations": sum(self.optimization_stats.values())
        }


class QuantumResourceManager:
    """Resource management system for quantum algorithms."""
    
    def __init__(self, max_memory_mb: int = 1024, max_concurrent_tasks: int = 10):
        """Initialize quantum resource manager.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.max_memory_mb = max_memory_mb
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_memory_mb = 0
        self.current_tasks = 0
        self.lock = threading.Lock()
        self.logger = logger
        
        # Resource usage tracking
        self.memory_history = []
        self.task_history = []
        self.allocation_count = 0
        self.deallocation_count = 0
    
    @monitor_function("allocate_resources", "quantum_performance")
    def allocate_resources(self, memory_mb: float, task_count: int = 1) -> bool:
        """Allocate resources for quantum computation.
        
        Args:
            memory_mb: Memory to allocate in MB
            task_count: Number of tasks to allocate
            
        Returns:
            True if resources were allocated, False otherwise
        """
        with self.lock:
            # Check if allocation is possible
            if (self.current_memory_mb + memory_mb > self.max_memory_mb or
                self.current_tasks + task_count > self.max_concurrent_tasks):
                
                self.logger.warning(
                    f"Resource allocation failed: "
                    f"Memory: {self.current_memory_mb + memory_mb}/{self.max_memory_mb}MB, "
                    f"Tasks: {self.current_tasks + task_count}/{self.max_concurrent_tasks}"
                )
                return False
            
            # Allocate resources
            self.current_memory_mb += memory_mb
            self.current_tasks += task_count
            self.allocation_count += 1
            
            # Track usage
            self.memory_history.append(self.current_memory_mb)
            self.task_history.append(self.current_tasks)
            
            # Limit history size
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-500:]
                self.task_history = self.task_history[-500:]
            
            self.logger.debug(
                f"Resources allocated: {memory_mb}MB, {task_count} tasks. "
                f"Total: {self.current_memory_mb}MB, {self.current_tasks} tasks"
            )
            return True
    
    @monitor_function("deallocate_resources", "quantum_performance")
    def deallocate_resources(self, memory_mb: float, task_count: int = 1):
        """Deallocate resources after quantum computation.
        
        Args:
            memory_mb: Memory to deallocate in MB
            task_count: Number of tasks to deallocate
        """
        with self.lock:
            self.current_memory_mb = max(0, self.current_memory_mb - memory_mb)
            self.current_tasks = max(0, self.current_tasks - task_count)
            self.deallocation_count += 1
            
            self.logger.debug(
                f"Resources deallocated: {memory_mb}MB, {task_count} tasks. "
                f"Remaining: {self.current_memory_mb}MB, {self.current_tasks} tasks"
            )
    
    def estimate_memory_usage(self, problem_size: int, algorithm: str) -> float:
        """Estimate memory usage for a quantum algorithm.
        
        Args:
            problem_size: Size of the problem (e.g., number of qubits/nodes)
            algorithm: Algorithm name
            
        Returns:
            Estimated memory usage in MB
        """
        # Base memory estimates per algorithm
        base_memory = {
            "cv_qaoa": 0.1,  # MB per node
            "error_correction": 0.05,  # MB per qubit
            "vqe": 0.08,  # MB per qubit
            "default": 0.1
        }
        
        multiplier = base_memory.get(algorithm, base_memory["default"])
        estimated_mb = problem_size * multiplier
        
        # Add overhead
        overhead = max(1.0, estimated_mb * 0.2)  # 20% overhead
        return estimated_mb + overhead
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource management statistics."""
        with self.lock:
            avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
            avg_tasks = sum(self.task_history) / len(self.task_history) if self.task_history else 0
            
            memory_utilization = (self.current_memory_mb / self.max_memory_mb) if self.max_memory_mb > 0 else 0
            task_utilization = (self.current_tasks / self.max_concurrent_tasks) if self.max_concurrent_tasks > 0 else 0
            
            return {
                "current_memory_mb": self.current_memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "current_tasks": self.current_tasks,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "memory_utilization": memory_utilization,
                "task_utilization": task_utilization,
                "avg_memory_mb": avg_memory,
                "avg_tasks": avg_tasks,
                "allocation_count": self.allocation_count,
                "deallocation_count": self.deallocation_count
            }


# Global instances for performance optimization
_global_cache = None
_global_parallel_processor = None
_global_optimization_engine = None
_global_resource_manager = None


def get_quantum_cache() -> QuantumAlgorithmCache:
    """Get global quantum algorithm cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = QuantumAlgorithmCache()
    return _global_cache


def get_parallel_processor() -> ParallelQuantumProcessor:
    """Get global parallel quantum processor."""
    global _global_parallel_processor
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelQuantumProcessor()
    return _global_parallel_processor


def get_optimization_engine() -> QuantumOptimizationEngine:
    """Get global quantum optimization engine."""
    global _global_optimization_engine
    if _global_optimization_engine is None:
        _global_optimization_engine = QuantumOptimizationEngine(get_quantum_cache())
    return _global_optimization_engine


def get_resource_manager() -> QuantumResourceManager:
    """Get global quantum resource manager."""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = QuantumResourceManager()
    return _global_resource_manager