"""Data repositories providing high-level CRUD operations for photonic designs."""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from .database.models import DesignDatabase, CircuitModel, OptimizationResult, SimulationResult
from .database.persistence import ModelPersistence, CircuitPersistence  
from .database.cache import ComponentCache, SimulationCache
from .circuit import PhotonicCircuit
from .templates import PhotonicComponent

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Abstract base repository with common functionality."""
    
    def __init__(self, database: DesignDatabase):
        """Initialize base repository.
        
        Args:
            database: Design database instance
        """
        self.database = database
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def create(self, *args, **kwargs) -> str:
        """Create new entity."""
        pass
    
    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[Any]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def update(self, entity_id: str, **kwargs) -> bool:
        """Update entity."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete entity."""
        pass
    
    @abstractmethod
    def list_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List all entities with pagination."""
        pass


class CircuitRepository(BaseRepository):
    """Repository for managing photonic circuit designs."""
    
    def __init__(self, database: DesignDatabase, 
                 circuit_persistence: Optional[CircuitPersistence] = None,
                 component_cache: Optional[ComponentCache] = None):
        """Initialize circuit repository.
        
        Args:
            database: Design database instance
            circuit_persistence: Circuit persistence layer
            component_cache: Component cache for performance
        """
        super().__init__(database)
        self.circuit_persistence = circuit_persistence or CircuitPersistence()
        self.component_cache = component_cache or ComponentCache()
    
    def create(self, name: str, description: str = "", 
               neural_network_source: str = "",
               compilation_config: Optional[Dict[str, Any]] = None,
               **kwargs) -> str:
        """Create new circuit design.
        
        Args:
            name: Circuit name
            description: Circuit description
            neural_network_source: Source neural network description
            compilation_config: Compilation configuration
            **kwargs: Additional circuit properties
            
        Returns:
            Circuit ID
        """
        circuit_model = CircuitModel(
            name=name,
            description=description,
            neural_network_source=neural_network_source,
            compilation_config=compilation_config or {},
            **kwargs
        )
        
        circuit_id = self.database.save_circuit(circuit_model)
        self.logger.info(f"Created circuit: {name} ({circuit_id})")
        return circuit_id
    
    def get_by_id(self, circuit_id: str) -> Optional[CircuitModel]:
        """Get circuit by ID.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            Circuit model or None
        """
        return self.database.load_circuit(circuit_id)
    
    def get_by_name(self, name: str) -> Optional[CircuitModel]:
        """Get circuit by name.
        
        Args:
            name: Circuit name
            
        Returns:
            Circuit model or None (returns first match)
        """
        circuits = self.database.search_circuits(name, limit=1)
        return circuits[0] if circuits else None
    
    def update(self, circuit_id: str, **kwargs) -> bool:
        """Update circuit properties.
        
        Args:
            circuit_id: Circuit ID
            **kwargs: Properties to update
            
        Returns:
            True if successful
        """
        circuit = self.database.load_circuit(circuit_id)
        if circuit is None:
            self.logger.warning(f"Circuit not found: {circuit_id}")
            return False
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(circuit, key):
                setattr(circuit, key, value)
        
        circuit.updated_at = datetime.now()
        
        try:
            self.database.save_circuit(circuit)
            self.logger.info(f"Updated circuit: {circuit_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update circuit {circuit_id}: {e}")
            return False
    
    def delete(self, circuit_id: str) -> bool:
        """Delete circuit and associated data.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            True if successful
        """
        success = self.database.delete_circuit(circuit_id)
        
        if success:
            # Also delete from persistence layer
            self.circuit_persistence.delete_circuit(circuit_id)
            self.logger.info(f"Deleted circuit: {circuit_id}")
        
        return success
    
    def list_all(self, limit: int = 100, offset: int = 0) -> List[CircuitModel]:
        """List all circuits with pagination.
        
        Args:
            limit: Maximum number of circuits to return
            offset: Number of circuits to skip
            
        Returns:
            List of circuit models
        """
        return self.database.list_circuits(limit, offset)
    
    def search(self, query: str, limit: int = 50) -> List[CircuitModel]:
        """Search circuits by name or description.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching circuits
        """
        return self.database.search_circuits(query, limit)
    
    def save_circuit_instance(self, circuit: PhotonicCircuit, 
                             circuit_id: str,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save PhotonicCircuit instance to persistence layer.
        
        Args:
            circuit: PhotonicCircuit instance
            circuit_id: Circuit ID
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Update database model with circuit data
            circuit_model = self.database.load_circuit(circuit_id)
            if circuit_model is None:
                self.logger.error(f"Circuit model not found: {circuit_id}")
                return False
            
            # Extract circuit data
            circuit_model.components = [self._serialize_component(comp) for comp in circuit.components]
            circuit_model.connections = [{"source": src, "target": tgt} for src, tgt in circuit.connections]
            circuit_model.layout_data = circuit.physical_layout or {}
            
            if circuit.metrics:
                circuit_model.performance_metrics = {
                    "total_power": circuit.metrics.total_power,
                    "total_area": circuit.metrics.total_area,
                    "total_loss": circuit.metrics.total_loss,
                    "latency": circuit.metrics.latency,
                    "throughput": circuit.metrics.throughput,
                    "energy_efficiency": circuit.metrics.energy_efficiency
                }
            
            circuit_model.updated_at = datetime.now()
            
            # Save to database
            self.database.save_circuit(circuit_model)
            
            # Save to persistence layer
            persistent_id = self.circuit_persistence.save_circuit(
                circuit, 
                circuit_model.name,
                metadata
            )
            
            self.logger.info(f"Saved circuit instance: {circuit_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save circuit instance {circuit_id}: {e}")
            return False
    
    def load_circuit_instance(self, circuit_id: str) -> Optional[PhotonicCircuit]:
        """Load PhotonicCircuit instance from persistence layer.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            PhotonicCircuit instance or None
        """
        return self.circuit_persistence.load_circuit(circuit_id)
    
    def get_recent_circuits(self, days: int = 7, limit: int = 10) -> List[CircuitModel]:
        """Get recently created circuits.
        
        Args:
            days: Number of days to look back
            limit: Maximum circuits to return
            
        Returns:
            List of recent circuits
        """
        circuits = self.database.list_circuits(limit=limit * 2)  # Get extra to filter
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_circuits = [
            circuit for circuit in circuits 
            if circuit.created_at and circuit.created_at >= cutoff_date
        ]
        
        return recent_circuits[:limit]
    
    def get_circuits_by_performance(self, metric: str, min_value: float, 
                                   max_value: Optional[float] = None,
                                   limit: int = 50) -> List[CircuitModel]:
        """Get circuits filtered by performance metric.
        
        Args:
            metric: Performance metric name
            min_value: Minimum value
            max_value: Maximum value (optional)
            limit: Maximum results
            
        Returns:
            List of circuits matching criteria
        """
        circuits = self.database.list_circuits(limit=limit * 2)
        filtered_circuits = []
        
        for circuit in circuits:
            if metric in circuit.performance_metrics:
                value = circuit.performance_metrics[metric]
                if value >= min_value and (max_value is None or value <= max_value):
                    filtered_circuits.append(circuit)
        
        return filtered_circuits[:limit]
    
    def export_circuit(self, circuit_id: str, export_path: Path) -> bool:
        """Export circuit data to file.
        
        Args:
            circuit_id: Circuit ID
            export_path: Export file path
            
        Returns:
            True if successful
        """
        return self.database.export_circuit(circuit_id, export_path)
    
    def import_circuit(self, import_path: Path) -> Optional[str]:
        """Import circuit data from file.
        
        Args:
            import_path: Import file path
            
        Returns:
            New circuit ID or None
        """
        return self.database.import_circuit(import_path)
    
    def _serialize_component(self, component: PhotonicComponent) -> Dict[str, Any]:
        """Serialize component for database storage."""
        return {
            "name": component.spec.name,
            "component_type": component.spec.component_type,
            "parameters": component.spec.parameters,
            "constraints": component.spec.constraints,
            "performance": component.spec.performance,
            "instance_id": component.instance_id,
            "position": component.position,
            "orientation": component.orientation,
            "ports": component.ports,
            "connections": component.connections
        }


class OptimizationRepository(BaseRepository):
    """Repository for managing optimization results."""
    
    def __init__(self, database: DesignDatabase,
                 component_cache: Optional[ComponentCache] = None):
        """Initialize optimization repository.
        
        Args:
            database: Design database instance
            component_cache: Component cache for optimization results
        """
        super().__init__(database)
        self.component_cache = component_cache or ComponentCache()
    
    def create(self, circuit_id: str, optimizer_type: str,
               parameters: Dict[str, Any],
               initial_metrics: Dict[str, float],
               final_metrics: Dict[str, float],
               optimization_log: List[Dict[str, Any]],
               execution_time: float,
               success: bool,
               error_message: str = "",
               **kwargs) -> str:
        """Create optimization result.
        
        Args:
            circuit_id: Circuit ID
            optimizer_type: Type of optimizer used
            parameters: Optimization parameters
            initial_metrics: Initial performance metrics
            final_metrics: Final performance metrics
            optimization_log: Detailed optimization log
            execution_time: Execution time in seconds
            success: Whether optimization succeeded
            error_message: Error message if failed
            **kwargs: Additional properties
            
        Returns:
            Optimization result ID
        """
        result = OptimizationResult(
            circuit_id=circuit_id,
            optimizer_type=optimizer_type,
            parameters=parameters,
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
            optimization_log=optimization_log,
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            **kwargs
        )
        
        result_id = self.database.save_optimization_result(result)
        self.logger.info(f"Created optimization result: {optimizer_type} ({result_id})")
        return result_id
    
    def get_by_id(self, result_id: str) -> Optional[OptimizationResult]:
        """Get optimization result by ID."""
        # Note: Database doesn't have direct get by ID for optimization results
        # This would need to be implemented in the database layer
        results = self.database.load_optimization_results("")  # Get all
        for result in results:
            if result.id == result_id:
                return result
        return None
    
    def update(self, result_id: str, **kwargs) -> bool:
        """Update optimization result."""
        # Would need database support for individual optimization result updates
        self.logger.warning("Optimization result updates not implemented")
        return False
    
    def delete(self, result_id: str) -> bool:
        """Delete optimization result."""
        # Would need database support for individual optimization result deletion
        self.logger.warning("Individual optimization result deletion not implemented")
        return False
    
    def list_all(self, limit: int = 100, offset: int = 0) -> List[OptimizationResult]:
        """List all optimization results."""
        # Would need database support for listing all optimization results
        self.logger.warning("Listing all optimization results not efficiently supported")
        return []
    
    def get_by_circuit(self, circuit_id: str) -> List[OptimizationResult]:
        """Get optimization results for a circuit.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            List of optimization results
        """
        return self.database.load_optimization_results(circuit_id)
    
    def get_by_optimizer_type(self, circuit_id: str, 
                             optimizer_type: str) -> List[OptimizationResult]:
        """Get optimization results by optimizer type.
        
        Args:
            circuit_id: Circuit ID
            optimizer_type: Optimizer type
            
        Returns:
            List of matching optimization results
        """
        results = self.database.load_optimization_results(circuit_id)
        return [r for r in results if r.optimizer_type == optimizer_type]
    
    def get_successful_optimizations(self, circuit_id: str) -> List[OptimizationResult]:
        """Get successful optimization results for a circuit.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            List of successful optimization results
        """
        results = self.database.load_optimization_results(circuit_id)
        return [r for r in results if r.success]
    
    def get_best_optimization(self, circuit_id: str, 
                             metric: str = "final_power") -> Optional[OptimizationResult]:
        """Get best optimization result by metric.
        
        Args:
            circuit_id: Circuit ID
            metric: Metric to optimize for
            
        Returns:
            Best optimization result or None
        """
        results = self.get_successful_optimizations(circuit_id)
        if not results:
            return None
        
        # Find result with lowest value for the metric
        best_result = min(results, 
                         key=lambda r: r.final_metrics.get(metric, float('inf')))
        
        return best_result
    
    def get_optimization_stats(self, circuit_id: str) -> Dict[str, Any]:
        """Get optimization statistics for a circuit.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            Dictionary of optimization statistics
        """
        results = self.database.load_optimization_results(circuit_id)
        
        if not results:
            return {
                "total_optimizations": 0,
                "successful_optimizations": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "optimizer_types": []
            }
        
        successful = [r for r in results if r.success]
        
        stats = {
            "total_optimizations": len(results),
            "successful_optimizations": len(successful),
            "success_rate": len(successful) / len(results) * 100,
            "average_execution_time": sum(r.execution_time for r in results) / len(results),
            "optimizer_types": list(set(r.optimizer_type for r in results)),
            "total_execution_time": sum(r.execution_time for r in results)
        }
        
        if successful:
            stats["best_power_reduction"] = max(
                (r.initial_metrics.get("power", 0) - r.final_metrics.get("power", 0))
                for r in successful
            )
            stats["average_power_reduction"] = sum(
                (r.initial_metrics.get("power", 0) - r.final_metrics.get("power", 0))
                for r in successful
            ) / len(successful)
        
        return stats
    
    def cache_optimization_result(self, circuit: PhotonicCircuit,
                                 optimizer_type: str,
                                 parameters: Dict[str, Any],
                                 result: Any) -> str:
        """Cache optimization result for performance.
        
        Args:
            circuit: PhotonicCircuit instance
            optimizer_type: Optimizer type
            parameters: Optimization parameters
            result: Optimization result
            
        Returns:
            Cache key
        """
        return self.component_cache.cache_optimization_result(
            circuit, optimizer_type, parameters, result
        )
    
    def get_cached_optimization(self, circuit: PhotonicCircuit,
                               optimizer_type: str,
                               parameters: Dict[str, Any]) -> Optional[Any]:
        """Get cached optimization result.
        
        Args:
            circuit: PhotonicCircuit instance
            optimizer_type: Optimizer type
            parameters: Optimization parameters
            
        Returns:
            Cached result or None
        """
        return self.component_cache.get_optimization_result(
            circuit, optimizer_type, parameters
        )


class SimulationRepository(BaseRepository):
    """Repository for managing simulation results."""
    
    def __init__(self, database: DesignDatabase,
                 simulation_cache: Optional[SimulationCache] = None):
        """Initialize simulation repository.
        
        Args:
            database: Design database instance
            simulation_cache: Simulation cache for performance
        """
        super().__init__(database)
        self.simulation_cache = simulation_cache or SimulationCache()
    
    def create(self, circuit_id: str, simulation_type: str,
               parameters: Dict[str, Any],
               results: Dict[str, Any],
               simulation_time: float,
               convergence_achieved: bool,
               error_message: str = "",
               **kwargs) -> str:
        """Create simulation result.
        
        Args:
            circuit_id: Circuit ID
            simulation_type: Type of simulation
            parameters: Simulation parameters
            results: Simulation results
            simulation_time: Simulation time in seconds
            convergence_achieved: Whether simulation converged
            error_message: Error message if failed
            **kwargs: Additional properties
            
        Returns:
            Simulation result ID
        """
        result = SimulationResult(
            circuit_id=circuit_id,
            simulation_type=simulation_type,
            parameters=parameters,
            results=results,
            simulation_time=simulation_time,
            convergence_achieved=convergence_achieved,
            error_message=error_message,
            **kwargs
        )
        
        result_id = self.database.save_simulation_result(result)
        self.logger.info(f"Created simulation result: {simulation_type} ({result_id})")
        return result_id
    
    def get_by_id(self, result_id: str) -> Optional[SimulationResult]:
        """Get simulation result by ID."""
        # Would need database support
        self.logger.warning("Simulation result get by ID not implemented")
        return None
    
    def update(self, result_id: str, **kwargs) -> bool:
        """Update simulation result."""
        self.logger.warning("Simulation result updates not implemented")
        return False
    
    def delete(self, result_id: str) -> bool:
        """Delete simulation result."""
        self.logger.warning("Individual simulation result deletion not implemented")
        return False
    
    def list_all(self, limit: int = 100, offset: int = 0) -> List[SimulationResult]:
        """List all simulation results."""
        self.logger.warning("Listing all simulation results not efficiently supported")
        return []
    
    def get_by_circuit(self, circuit_id: str) -> List[SimulationResult]:
        """Get simulation results for a circuit.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            List of simulation results
        """
        return self.database.load_simulation_results(circuit_id)
    
    def get_by_simulation_type(self, circuit_id: str,
                              simulation_type: str) -> List[SimulationResult]:
        """Get simulation results by type.
        
        Args:
            circuit_id: Circuit ID
            simulation_type: Simulation type
            
        Returns:
            List of matching simulation results
        """
        results = self.database.load_simulation_results(circuit_id)
        return [r for r in results if r.simulation_type == simulation_type]
    
    def get_converged_simulations(self, circuit_id: str) -> List[SimulationResult]:
        """Get converged simulation results.
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            List of converged simulation results
        """
        results = self.database.load_simulation_results(circuit_id)
        return [r for r in results if r.convergence_achieved]
    
    def cache_simulation_result(self, circuit: PhotonicCircuit,
                               simulation_params: Dict[str, Any],
                               results: Dict[str, Any]) -> str:
        """Cache simulation result.
        
        Args:
            circuit: PhotonicCircuit instance
            simulation_params: Simulation parameters
            results: Simulation results
            
        Returns:
            Cache key
        """
        return self.simulation_cache.cache_simulation(circuit, simulation_params, results)
    
    def get_cached_simulation(self, circuit: PhotonicCircuit,
                             simulation_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached simulation result.
        
        Args:
            circuit: PhotonicCircuit instance
            simulation_params: Simulation parameters
            
        Returns:
            Cached results or None
        """
        return self.simulation_cache.get_simulation(circuit, simulation_params)


class ModelRepository:
    """Repository for managing neural network models."""
    
    def __init__(self, model_persistence: Optional[ModelPersistence] = None):
        """Initialize model repository.
        
        Args:
            model_persistence: Model persistence layer
        """
        self.model_persistence = model_persistence or ModelPersistence()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_pytorch_model(self, model: Any, model_name: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save PyTorch model.
        
        Args:
            model: PyTorch model
            model_name: Model name
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        return self.model_persistence.save_pytorch_model(model, model_name, metadata)
    
    def load_pytorch_model(self, model_path: str, model_class: Optional[Any] = None):
        """Load PyTorch model.
        
        Args:
            model_path: Path to model file
            model_class: Model class for loading
            
        Returns:
            Loaded model
        """
        return self.model_persistence.load_pytorch_model(model_path, model_class)
    
    def export_onnx_model(self, model: Any, model_name: str,
                         input_shape: tuple,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Export model to ONNX format.
        
        Args:
            model: PyTorch model
            model_name: Model name
            input_shape: Input shape for export
            metadata: Additional metadata
            
        Returns:
            Path to ONNX model
        """
        return self.model_persistence.save_onnx_model(model, model_name, input_shape, metadata)
    
    def list_models(self, framework: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models.
        
        Args:
            framework: Filter by framework
            
        Returns:
            List of model metadata
        """
        return self.model_persistence.list_models(framework)
    
    def delete_model(self, model_path: str) -> bool:
        """Delete model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        return self.model_persistence.delete_model(model_path)
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by name.
        
        Args:
            model_name: Model name
            
        Returns:
            Model metadata or None
        """
        models = self.list_models()
        for model in models:
            if model.get('model_name') == model_name:
                return model
        return None
    
    def get_recent_models(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently saved models.
        
        Args:
            days: Number of days to look back
            limit: Maximum models to return
            
        Returns:
            List of recent models
        """
        models = self.list_models()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_models = []
        for model in models:
            if 'saved_at' in model:
                try:
                    saved_date = datetime.fromisoformat(model['saved_at'].replace('Z', '+00:00'))
                    if saved_date.replace(tzinfo=None) >= cutoff_date:
                        recent_models.append(model)
                except (ValueError, AttributeError):
                    continue
        
        return recent_models[:limit]


class RepositoryManager:
    """Centralized manager for all repositories."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize repository manager.
        
        Args:
            db_path: Database file path
        """
        self.database = DesignDatabase(db_path)
        self.circuit_persistence = CircuitPersistence()
        self.model_persistence = ModelPersistence()
        self.component_cache = ComponentCache()
        self.simulation_cache = SimulationCache()
        
        # Initialize repositories
        self.circuits = CircuitRepository(
            self.database, 
            self.circuit_persistence,
            self.component_cache
        )
        
        self.optimizations = OptimizationRepository(
            self.database,
            self.component_cache
        )
        
        self.simulations = SimulationRepository(
            self.database,
            self.simulation_cache
        )
        
        self.models = ModelRepository(self.model_persistence)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        db_stats = self.database.get_database_stats()
        cache_stats = {
            'component_cache': self.component_cache.get_stats(),
            'simulation_cache': self.simulation_cache.get_stats()
        }
        
        return {
            'database': db_stats,
            'caches': cache_stats,
            'total_models': len(self.models.list_models())
        }
    
    def cleanup_caches(self) -> None:
        """Cleanup all caches.""" 
        self.component_cache.clear()
        self.simulation_cache.clear()
        self.logger.info("Cleaned up all caches")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            'database': True,
            'caches': True,
            'persistence': True,
            'errors': []
        }
        
        try:
            # Test database connection
            stats = self.database.get_database_stats()
            health['database'] = True
        except Exception as e:
            health['database'] = False
            health['errors'].append(f"Database error: {e}")
        
        try:
            # Test cache functionality
            cache_stats = self.component_cache.get_stats()
            health['caches'] = True
        except Exception as e:
            health['caches'] = False
            health['errors'].append(f"Cache error: {e}")
        
        try:
            # Test persistence layers
            models = self.models.list_models() 
            health['persistence'] = True
        except Exception as e:
            health['persistence'] = False
            health['errors'].append(f"Persistence error: {e}")
        
        health['overall'] = all([health['database'], health['caches'], health['persistence']])
        
        return health
    
    def create_circuit_with_model(self, model: Any, model_name: str, 
                                 circuit_name: str, circuit_description: str,
                                 compilation_config: Dict[str, Any]) -> Tuple[str, str]:
        """Create circuit with associated neural network model.
        
        Args:
            model: Neural network model
            model_name: Model name
            circuit_name: Circuit name  
            circuit_description: Circuit description
            compilation_config: Compilation configuration
            
        Returns:
            Tuple of (circuit_id, model_path)
        """
        # Save model
        model_path = self.models.save_pytorch_model(model, model_name)
        
        # Create circuit with model reference
        circuit_id = self.circuits.create(
            name=circuit_name,
            description=circuit_description,
            neural_network_source=f"pytorch:{model_path}",
            compilation_config=compilation_config
        )
        
        self.logger.info(f"Created circuit with model: {circuit_name} ({circuit_id})")
        return circuit_id, model_path