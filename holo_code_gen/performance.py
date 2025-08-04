"""Performance optimization and scaling for Holo-Code-Gen."""

import time
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import hashlib
import pickle
import json
from pathlib import Path
import queue
import weakref
from datetime import datetime, timedelta

from .exceptions import HoloCodeGenException, TimeoutError, ResourceLimitError
from .monitoring import get_logger, get_metrics_collector


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: float = 3600.0  # 1 hour
    enable_parallel_processing: bool = True
    max_workers: int = mp.cpu_count()
    max_memory_mb: float = 4096.0
    enable_lazy_loading: bool = True
    batch_size: int = 32
    prefetch_factor: int = 2
    enable_profiling: bool = False


class CacheManager:
    """High-performance caching with TTL and memory management."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        """Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}  # value, timestamp
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.logger = get_logger()
        self.metrics = get_metrics_collector()
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                self.metrics.increment_counter("cache_misses", {"operation": "get"})
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self._misses += 1
                self.metrics.increment_counter("cache_misses", {"operation": "expired"})
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            self._hits += 1
            self.metrics.increment_counter("cache_hits", {"operation": "get"})
            
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            current_time = time.time()
            
            # Check if we need to evict items
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = (value, current_time)
            self._access_times[key] = current_time
            
            self.metrics.increment_counter("cache_puts", {"operation": "put"})
    
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            
            self.metrics.increment_counter("cache_invalidations", {"operation": "invalidate"})
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self.metrics.increment_counter("cache_clears", {"operation": "clear"})
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find LRU item
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._evictions += 1
        
        self.metrics.increment_counter("cache_evictions", {"operation": "lru"})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size
            }


class ObjectPool:
    """Pool for expensive-to-create objects."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 10):
        """Initialize object pool.
        
        Args:
            factory: Function to create new objects
            max_size: Maximum pool size
        """
        self.factory = factory
        self.max_size = max_size
        self._pool = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created_count = 0
    
    def get(self) -> Any:
        """Get object from pool or create new one.
        
        Returns:
            Object from pool
        """
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created_count < self.max_size:
                    obj = self.factory()
                    self._created_count += 1
                    return obj
                else:
                    # Wait for object to become available
                    return self._pool.get(timeout=1.0)
    
    def put(self, obj: Any) -> None:
        """Return object to pool.
        
        Args:
            obj: Object to return to pool
        """
        try:
            self._pool.put_nowait(obj)
        except queue.Full:
            # Pool is full, just discard the object
            pass


class BatchProcessor:
    """Processes items in batches for better performance."""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 0.1):
        """Initialize batch processor.
        
        Args:
            batch_size: Size of batches to process
            max_wait_time: Maximum time to wait for batch to fill
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self._queue = queue.Queue()
        self._results = {}
        self._lock = threading.Lock()
        self._processor_thread = None
        self._stop_event = threading.Event()
    
    def start(self, processor_func: Callable[[List[Any]], List[Any]]) -> None:
        """Start batch processing.
        
        Args:
            processor_func: Function to process batches
        """
        self.processor_func = processor_func
        self._processor_thread = threading.Thread(target=self._process_batches)
        self._processor_thread.daemon = True
        self._processor_thread.start()
    
    def submit(self, item: Any, item_id: str) -> None:
        """Submit item for batch processing.
        
        Args:
            item: Item to process
            item_id: Unique identifier for the item
        """
        self._queue.put((item, item_id))
    
    def get_result(self, item_id: str, timeout: float = 10.0) -> Any:
        """Get result for processed item.
        
        Args:
            item_id: Item identifier
            timeout: Timeout in seconds
            
        Returns:
            Processing result
            
        Raises:
            TimeoutError: If result not available within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if item_id in self._results:
                    result = self._results.pop(item_id)
                    return result
            
            time.sleep(0.01)  # 10ms polling interval
        
        raise TimeoutError(
            f"Result for {item_id} not available within {timeout}s",
            timeout_seconds=timeout
        )
    
    def _process_batches(self) -> None:
        """Process items in batches."""
        batch = []
        batch_ids = []
        last_process_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Try to get item with timeout
                item, item_id = self._queue.get(timeout=0.01)
                batch.append(item)
                batch_ids.append(item_id)
                
                # Process batch if full or max wait time exceeded
                should_process = (
                    len(batch) >= self.batch_size or
                    time.time() - last_process_time > self.max_wait_time
                )
                
                if should_process and batch:
                    try:
                        results = self.processor_func(batch)
                        
                        # Store results
                        with self._lock:
                            for batch_id, result in zip(batch_ids, results):
                                self._results[batch_id] = result
                        
                        batch.clear()
                        batch_ids.clear()
                        last_process_time = time.time()
                        
                    except Exception as e:
                        # Store error for all items in batch
                        with self._lock:
                            for batch_id in batch_ids:
                                self._results[batch_id] = e
                        
                        batch.clear()
                        batch_ids.clear()
                
            except queue.Empty:
                # Process partial batch if max wait time exceeded
                if batch and time.time() - last_process_time > self.max_wait_time:
                    try:
                        results = self.processor_func(batch)
                        
                        with self._lock:
                            for batch_id, result in zip(batch_ids, results):
                                self._results[batch_id] = result
                        
                        batch.clear()
                        batch_ids.clear()
                        last_process_time = time.time()
                        
                    except Exception as e:
                        with self._lock:
                            for batch_id in batch_ids:
                                self._results[batch_id] = e
                        
                        batch.clear()
                        batch_ids.clear()
    
    def stop(self) -> None:
        """Stop batch processing."""
        self._stop_event.set()
        if self._processor_thread:
            self._processor_thread.join(timeout=1.0)


class ParallelExecutor:
    """High-performance parallel execution engine."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Whether to use processes instead of threads
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.logger = get_logger()
        self.metrics = get_metrics_collector()
    
    def map_parallel(self, func: Callable, items: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """Execute function on items in parallel.
        
        Args:
            func: Function to execute
            items: Items to process
            chunk_size: Size of chunks for processing
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        if len(items) == 1:
            # Single item, no need for parallelization
            return [func(items[0])]
        
        start_time = time.time()
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # Determine chunk size
                if chunk_size is None:
                    chunk_size = max(1, len(items) // (self.max_workers * 2))
                
                # Submit tasks
                futures = []
                for i in range(0, len(items), chunk_size):
                    chunk = items[i:i + chunk_size]
                    if len(chunk) == 1:
                        future = executor.submit(func, chunk[0])
                    else:
                        future = executor.submit(self._process_chunk, func, chunk)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parallel execution error: {e}")
                        raise
                
                duration = time.time() - start_time
                self.metrics.record_histogram(
                    "parallel_execution_duration", 
                    duration * 1000,  # Convert to ms
                    {"items": len(items), "workers": self.max_workers},
                    "ms"
                )
                
                return results
                
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            return [func(item) for item in items]
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]


class MemoryManager:
    """Manages memory usage and prevents out-of-memory errors."""
    
    def __init__(self, max_memory_mb: float = 4096.0):
        """Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.logger = get_logger()
        self._memory_warning_threshold = 0.8  # 80% of max memory
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = (memory_mb / self.max_memory_mb) * 100
            
            stats = {
                "used_mb": memory_mb,
                "max_mb": self.max_memory_mb,
                "usage_percent": memory_percent
            }
            
            # Log warnings for high memory usage
            if memory_percent > self._memory_warning_threshold * 100:
                self.logger.warning(
                    f"High memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)",
                    component="memory_manager"
                )
            
            # Raise error if memory limit exceeded
            if memory_mb > self.max_memory_mb:
                raise ResourceLimitError(
                    f"Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_mb:.1f}MB",
                    resource_type="memory",
                    limit=self.max_memory_mb,
                    current=memory_mb
                )
            
            return stats
            
        except ImportError:
            # psutil not available
            return {"used_mb": 0, "max_mb": self.max_memory_mb, "usage_percent": 0}
    
    def gc_if_needed(self) -> None:
        """Run garbage collection if memory usage is high."""
        import gc
        
        stats = self.check_memory_usage()
        if stats["usage_percent"] > self._memory_warning_threshold * 100:
            gc.collect()
            self.logger.info("Garbage collection triggered due to high memory usage")


class LazyLoader:
    """Lazy loading for expensive resources."""
    
    def __init__(self):
        """Initialize lazy loader."""
        self._cache = {}
        self._factories = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a lazy-loaded resource.
        
        Args:
            name: Resource name
            factory: Function to create the resource
        """
        with self._lock:
            self._factories[name] = factory
    
    def get(self, name: str) -> Any:
        """Get lazy-loaded resource.
        
        Args:
            name: Resource name
            
        Returns:
            Loaded resource
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        with self._lock:
            # Double-check pattern
            if name in self._cache:
                return self._cache[name]
            
            if name not in self._factories:
                raise ValueError(f"No factory registered for resource: {name}")
            
            # Create resource
            resource = self._factories[name]()
            self._cache[name] = resource
            
            return resource


# Global performance components
_cache_manager: Optional[CacheManager] = None
_parallel_executor: Optional[ParallelExecutor] = None
_memory_manager: Optional[MemoryManager] = None
_lazy_loader: Optional[LazyLoader] = None


def initialize_performance(config: Optional[PerformanceConfig] = None) -> None:
    """Initialize global performance components.
    
    Args:
        config: Performance configuration
    """
    global _cache_manager, _parallel_executor, _memory_manager, _lazy_loader
    
    perf_config = config or PerformanceConfig()
    
    if perf_config.enable_caching:
        _cache_manager = CacheManager(
            max_size=perf_config.cache_size,
            ttl_seconds=perf_config.cache_ttl_seconds
        )
    
    if perf_config.enable_parallel_processing:
        _parallel_executor = ParallelExecutor(
            max_workers=perf_config.max_workers,
            use_processes=False  # Use threads for now
        )
    
    _memory_manager = MemoryManager(max_memory_mb=perf_config.max_memory_mb)
    
    if perf_config.enable_lazy_loading:
        _lazy_loader = LazyLoader()


def get_cache_manager() -> Optional[CacheManager]:
    """Get global cache manager."""
    return _cache_manager


def get_parallel_executor() -> Optional[ParallelExecutor]:
    """Get global parallel executor."""
    return _parallel_executor


def get_memory_manager() -> Optional[MemoryManager]:
    """Get global memory manager."""
    return _memory_manager


def get_lazy_loader() -> Optional[LazyLoader]:
    """Get global lazy loader."""
    return _lazy_loader


def cached(ttl_seconds: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        ttl_seconds: Time-to-live for cached results
        key_func: Function to generate cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            if cache is None:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                cache_key = hashlib.md5(
                    json.dumps(key_data, sort_keys=True, default=str).encode()
                ).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        return wrapper
    return decorator


def parallel_map(func: Callable, items: List[Any], chunk_size: Optional[int] = None) -> List[Any]:
    """Execute function on items in parallel.
    
    Args:
        func: Function to execute
        items: Items to process
        chunk_size: Size of chunks for processing
        
    Returns:
        List of results
    """
    executor = get_parallel_executor()
    if executor is None or len(items) <= 1:
        return [func(item) for item in items]
    
    return executor.map_parallel(func, items, chunk_size)


def memory_check(func: Callable) -> Callable:
    """Decorator to check memory usage before function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        memory_mgr = get_memory_manager()
        if memory_mgr:
            memory_mgr.check_memory_usage()
            memory_mgr.gc_if_needed()
        
        result = func(*args, **kwargs)
        
        if memory_mgr:
            memory_mgr.check_memory_usage()
        
        return result
    return wrapper


def lazy_resource(name: str):
    """Decorator for lazy resource loading.
    
    Args:
        name: Resource name
    """
    def decorator(factory_func: Callable) -> Callable:
        loader = get_lazy_loader()
        if loader:
            loader.register(name, factory_func)
        
        def get_resource():
            if loader:
                return loader.get(name)
            else:
                return factory_func()
        
        return get_resource
    return decorator