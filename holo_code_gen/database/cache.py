"""Caching system for photonic components and optimization results."""

import os
import json
import pickle
import hashlib
import time
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    data: Any
    timestamp: float
    size_bytes: int
    access_count: int = 0
    last_accessed: float = 0.0
    
    def __post_init__(self):
        """Initialize last_accessed to current time."""
        if self.last_accessed == 0.0:
            self.last_accessed = time.time()


class CacheManager:
    """Generic caching system with TTL and size limits."""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_size_mb: int = 1000,
                 default_ttl: int = 3600,
                 cleanup_interval: int = 300):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_size_mb: Maximum cache size in megabytes
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.cache_dir = cache_dir or Path.home() / ".holo_code_gen" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._last_cleanup = time.time()
        
        # Load persistent cache
        self._load_persistent_cache()
        
    def _load_persistent_cache(self) -> None:
        """Load cache from disk."""
        cache_index_file = self.cache_dir / "cache_index.json"
        if cache_index_file.exists():
            try:
                with open(cache_index_file, 'r') as f:
                    index_data = json.load(f)
                
                # Load entries that haven't expired
                current_time = time.time()
                for key, metadata in index_data.items():
                    if current_time - metadata['timestamp'] < metadata.get('ttl', self.default_ttl):
                        cache_file = self.cache_dir / f"{key}.cache"
                        if cache_file.exists():
                            try:
                                with open(cache_file, 'rb') as f:
                                    data = pickle.load(f)
                                
                                entry = CacheEntry(
                                    key=key,
                                    data=data,
                                    timestamp=metadata['timestamp'],
                                    size_bytes=metadata['size_bytes'],
                                    access_count=metadata.get('access_count', 0),
                                    last_accessed=metadata.get('last_accessed', metadata['timestamp'])
                                )
                                self._memory_cache[key] = entry
                                
                            except Exception as e:
                                logger.warning(f"Failed to load cache entry {key}: {e}")
                                # Remove corrupted cache file
                                cache_file.unlink(missing_ok=True)
                        
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
    
    def _save_persistent_cache(self) -> None:
        """Save cache index to disk."""
        cache_index_file = self.cache_dir / "cache_index.json"
        
        index_data = {}
        for key, entry in self._memory_cache.items():
            index_data[key] = {
                'timestamp': entry.timestamp,
                'size_bytes': entry.size_bytes,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed,
                'ttl': self.default_ttl
            }
            
            # Save data to separate file
            cache_file = self.cache_dir / f"{key}.cache"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry.data, f)
            except Exception as e:
                logger.error(f"Failed to save cache entry {key}: {e}")
        
        try:
            with open(cache_index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        self._cleanup_if_needed()
        
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            
            # Check if expired
            if time.time() - entry.timestamp > self.default_ttl:
                self.remove(key)
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            logger.debug(f"Cache hit for key: {key}")
            return entry.data
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Put item in cache."""
        self._cleanup_if_needed()
        
        # Calculate data size
        try:
            data_bytes = pickle.dumps(data)
            size_bytes = len(data_bytes)
        except Exception as e:
            logger.error(f"Failed to serialize data for cache: {e}")
            return False
        
        # Check if we need to free space
        current_size = self.get_total_size()
        if current_size + size_bytes > self.max_size_bytes:
            if not self._free_space(size_bytes):
                logger.warning("Failed to free enough cache space")
                return False
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            size_bytes=size_bytes
        )
        
        self._memory_cache[key] = entry
        logger.debug(f"Cached item with key: {key} ({size_bytes} bytes)")
        
        return True
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        if key in self._memory_cache:
            del self._memory_cache[key]
            
            # Remove persistent file
            cache_file = self.cache_dir / f"{key}.cache"
            cache_file.unlink(missing_ok=True)
            
            logger.debug(f"Removed cache entry: {key}")
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        # Remove persistent cache files
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)
        
        cache_index_file = self.cache_dir / "cache_index.json"
        cache_index_file.unlink(missing_ok=True)
        
        logger.info("Cleared all cache entries")
    
    def get_total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(entry.size_bytes for entry in self._memory_cache.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._memory_cache)
        total_size = self.get_total_size()
        total_accesses = sum(entry.access_count for entry in self._memory_cache.values())
        
        return {
            'total_entries': total_entries,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_accesses': total_accesses,
            'hit_rate': 0.0,  # Would need to track misses for accurate calculation
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': total_size / self.max_size_bytes * 100
        }
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup expired entries if enough time has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._memory_cache.items():
            if current_time - entry.timestamp > self.default_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.remove(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _free_space(self, needed_bytes: int) -> bool:
        """Free cache space using LRU eviction."""
        current_size = self.get_total_size()
        target_size = current_size + needed_bytes
        
        if target_size <= self.max_size_bytes:
            return True
        
        # Sort by last access time (LRU)
        entries_by_access = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        freed_bytes = 0
        for key, entry in entries_by_access:
            if freed_bytes >= needed_bytes:
                break
            
            freed_bytes += entry.size_bytes
            self.remove(key)
        
        return freed_bytes >= needed_bytes
    
    def __del__(self):
        """Save cache when object is destroyed."""
        try:
            self._save_persistent_cache()
        except Exception:
            pass  # Ignore errors during cleanup


class ComponentCache(CacheManager):
    """Specialized cache for photonic components."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize component cache."""
        if cache_dir is None:
            cache_dir = Path.home() / ".holo_code_gen" / "component_cache"
        
        super().__init__(
            cache_dir=cache_dir,
            max_size_mb=500,  # Smaller cache for components
            default_ttl=7200,  # 2 hour TTL
            cleanup_interval=600  # 10 minute cleanup
        )
    
    def cache_component(self, component: 'PhotonicComponent') -> str:
        """Cache a photonic component and return cache key."""
        # Generate key from component specification
        key_data = {
            'name': component.spec.name,
            'type': component.spec.component_type,
            'parameters': sorted(component.spec.parameters.items()),
            'constraints': sorted(component.spec.constraints.items())
        }
        
        key = self._generate_key(key_data)
        
        # Cache component data
        cache_data = {
            'spec': asdict(component.spec),
            'instance_id': component.instance_id,
            'position': component.position,
            'orientation': component.orientation,
            'ports': component.ports,
            'connections': component.connections
        }
        
        if self.put(key, cache_data):
            logger.debug(f"Cached component: {component.spec.name}")
            return key
        else:
            logger.warning(f"Failed to cache component: {component.spec.name}")
            return ""
    
    def get_component(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached component data."""
        return self.get(key)
    
    def cache_optimization_result(self, circuit: 'PhotonicCircuit', 
                                 optimizer_type: str, 
                                 parameters: Dict[str, Any],
                                 result: Any) -> str:
        """Cache optimization result."""
        # Generate key from circuit and optimization parameters
        circuit_hash = self._hash_circuit(circuit)
        key_data = {
            'circuit_hash': circuit_hash,
            'optimizer_type': optimizer_type,
            'parameters': sorted(parameters.items())
        }
        
        key = self._generate_key(key_data)
        
        cache_data = {
            'optimizer_type': optimizer_type,
            'parameters': parameters,
            'result': result,
            'circuit_components': len(circuit.components),
            'timestamp': time.time()
        }
        
        if self.put(key, cache_data):
            logger.debug(f"Cached optimization result: {optimizer_type}")
            return key
        else:
            logger.warning(f"Failed to cache optimization result: {optimizer_type}")
            return ""
    
    def get_optimization_result(self, circuit: 'PhotonicCircuit',
                               optimizer_type: str,
                               parameters: Dict[str, Any]) -> Optional[Any]:
        """Get cached optimization result."""
        circuit_hash = self._hash_circuit(circuit)
        key_data = {
            'circuit_hash': circuit_hash,
            'optimizer_type': optimizer_type,
            'parameters': sorted(parameters.items())
        }
        
        key = self._generate_key(key_data)
        cached_data = self.get(key)
        
        if cached_data:
            logger.debug(f"Found cached optimization result: {optimizer_type}")
            return cached_data['result']
        
        return None
    
    def _hash_circuit(self, circuit: 'PhotonicCircuit') -> str:
        """Generate hash for circuit configuration."""
        circuit_data = {
            'components': [
                {
                    'type': comp.component_type,
                    'parameters': sorted(comp.spec.parameters.items())
                }
                for comp in circuit.components
            ],
            'connections': sorted(circuit.connections)
        }
        
        circuit_str = json.dumps(circuit_data, sort_keys=True, default=str)
        return hashlib.md5(circuit_str.encode()).hexdigest()
    
    def invalidate_circuit_cache(self, circuit: 'PhotonicCircuit') -> None:
        """Invalidate all cached results for a circuit."""
        circuit_hash = self._hash_circuit(circuit)
        
        # Find and remove all entries with matching circuit hash
        keys_to_remove = []
        for key, entry in self._memory_cache.items():
            if hasattr(entry.data, 'get') and entry.data.get('circuit_hash') == circuit_hash:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.remove(key)
        
        logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for circuit")


class SimulationCache(CacheManager):
    """Specialized cache for simulation results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize simulation cache."""
        if cache_dir is None:
            cache_dir = Path.home() / ".holo_code_gen" / "simulation_cache"
        
        super().__init__(
            cache_dir=cache_dir,
            max_size_mb=2000,  # Larger cache for simulation data
            default_ttl=86400,  # 24 hour TTL
            cleanup_interval=3600  # 1 hour cleanup
        )
    
    def cache_simulation(self, circuit: 'PhotonicCircuit',
                        simulation_params: Dict[str, Any],
                        results: Dict[str, Any]) -> str:
        """Cache simulation results."""
        circuit_hash = self._hash_circuit_for_simulation(circuit)
        key_data = {
            'circuit_hash': circuit_hash,
            'simulation_params': sorted(simulation_params.items())
        }
        
        key = self._generate_key(key_data)
        
        cache_data = {
            'simulation_params': simulation_params,
            'results': results,
            'circuit_components': len(circuit.components),
            'timestamp': time.time()
        }
        
        if self.put(key, cache_data):
            logger.debug("Cached simulation results")
            return key
        else:
            logger.warning("Failed to cache simulation results")
            return ""
    
    def get_simulation(self, circuit: 'PhotonicCircuit',
                      simulation_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached simulation results."""
        circuit_hash = self._hash_circuit_for_simulation(circuit)
        key_data = {
            'circuit_hash': circuit_hash,
            'simulation_params': sorted(simulation_params.items())
        }
        
        key = self._generate_key(key_data)
        cached_data = self.get(key)
        
        if cached_data:
            logger.debug("Found cached simulation results")
            return cached_data['results']
        
        return None
    
    def _hash_circuit_for_simulation(self, circuit: 'PhotonicCircuit') -> str:
        """Generate hash for circuit relevant to simulation."""
        # Include physical layout and component positions for simulation
        circuit_data = {
            'components': [
                {
                    'type': comp.component_type,
                    'parameters': sorted(comp.spec.parameters.items()),
                    'position': comp.position,
                    'orientation': comp.orientation
                }
                for comp in circuit.components
            ],
            'connections': sorted(circuit.connections),
            'layout': circuit.physical_layout if circuit.physical_layout else None
        }
        
        circuit_str = json.dumps(circuit_data, sort_keys=True, default=str)
        return hashlib.md5(circuit_str.encode()).hexdigest()