"""Monitoring, logging, and telemetry for Holo-Code-Gen."""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import traceback
from functools import wraps
from contextlib import contextmanager

from .exceptions import HoloCodeGenException


@dataclass
class MetricData:
    """Container for metric data points."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class LogEvent:
    """Container for structured log events."""
    level: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: str = ""
    operation: str = ""
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "context": self.context
        }


class MetricsCollector:
    """Advanced metrics collector with quantum algorithm monitoring."""
    
    def __init__(self, enable_export: bool = True, export_interval: float = 60.0):
        """Initialize enhanced metrics collector.
        
        Args:
            enable_export: Whether to enable metric export
            export_interval: Export interval in seconds
        """
        self.enable_export = enable_export
        self.export_interval = export_interval
        self._metrics: List[MetricData] = []
        self._lock = threading.Lock()
        self._export_thread: Optional[threading.Thread] = None
        self._stop_export = threading.Event()
        
        # Advanced quantum algorithm metrics
        self._quantum_algorithm_metrics = {
            'enhanced_qaoa': {
                'execution_count': 0,
                'total_execution_time': 0.0,
                'convergence_rate': 0.0,
                'average_cost_improvement': 0.0,
                'multi_objective_success_rate': 0.0
            },
            'vqe_plus': {
                'execution_count': 0,
                'total_execution_time': 0.0,
                'average_energy_accuracy': 0.0,
                'qng_success_rate': 0.0,
                'adaptive_ansatz_efficiency': 0.0
            },
            'quantum_neural_networks': {
                'compilation_count': 0,
                'circuit_depth_efficiency': 0.0,
                'parameter_optimization_success': 0.0
            }
        }
        
        # Real-time performance analytics
        self._performance_analytics = {
            'algorithm_comparison': {},
            'resource_utilization': [],
            'convergence_patterns': {},
            'optimization_trends': {}
        }
        
        if enable_export:
            self._start_export_thread()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None,
                     unit: str = "") -> None:
        """Record a metric data point.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for filtering/grouping
            unit: Unit of measurement
        """
        metric = MetricData(
            name=name,
            value=value,
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self._metrics.append(metric)
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            tags: Optional tags
        """
        self.record_metric(name, 1.0, tags, "count")
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None,
                        unit: str = "") -> None:
        """Record a histogram metric (for timing, sizes, etc).
        
        Args:
            name: Histogram name
            value: Measured value
            tags: Optional tags
            unit: Unit of measurement
        """
        self.record_metric(f"{name}_value", value, tags, unit)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None,
                    unit: str = "") -> None:
        """Record a gauge metric (current value).
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags
            unit: Unit of measurement
        """
        self.record_metric(f"{name}_gauge", value, tags, unit)
    
    def get_metrics(self, since: Optional[datetime] = None) -> List[MetricData]:
        """Get collected metrics.
        
        Args:
            since: Only return metrics after this timestamp
            
        Returns:
            List of metric data points
        """
        with self._lock:
            if since is None:
                return self._metrics.copy()
            else:
                return [m for m in self._metrics if m.timestamp >= since]
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export comprehensive metrics including quantum algorithm analytics."""
        metrics_data = []
        
        with self._lock:
            for metric in self._metrics:
                metrics_data.append(metric.to_dict())
        
        return {
            "metrics": metrics_data,
            "quantum_algorithm_metrics": self._quantum_algorithm_metrics.copy(),
            "performance_analytics": self._performance_analytics.copy(),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_metrics": len(metrics_data)
        }
    
    def record_quantum_algorithm_execution(self, algorithm_type: str, execution_data: Dict[str, Any]) -> None:
        """Record quantum algorithm execution metrics.
        
        Args:
            algorithm_type: Type of quantum algorithm (enhanced_qaoa, vqe_plus, etc.)
            execution_data: Execution results and metrics
        """
        if algorithm_type not in self._quantum_algorithm_metrics:
            # Create default structure based on algorithm type
            if algorithm_type == 'enhanced_qaoa':
                self._quantum_algorithm_metrics[algorithm_type] = {
                    'execution_count': 0,
                    'total_execution_time': 0.0,
                    'convergence_rate': 0.0,
                    'average_cost_improvement': 0.0,
                    'multi_objective_success_rate': 0.0
                }
            elif algorithm_type == 'vqe_plus':
                self._quantum_algorithm_metrics[algorithm_type] = {
                    'execution_count': 0,
                    'total_execution_time': 0.0,
                    'average_energy_accuracy': 0.0,
                    'qng_success_rate': 0.0,
                    'adaptive_ansatz_efficiency': 0.0
                }
            else:
                self._quantum_algorithm_metrics[algorithm_type] = {
                    'execution_count': 0,
                    'total_execution_time': 0.0,
                    'average_performance': 0.0,
                    'success_rate': 0.0
                }
        
        # Update algorithm-specific metrics
        metrics = self._quantum_algorithm_metrics[algorithm_type]
        metrics['execution_count'] += 1
        
        if 'execution_time' in execution_data:
            metrics['total_execution_time'] += execution_data['execution_time']
        
        # Algorithm-specific metrics
        if algorithm_type == 'enhanced_qaoa':
            if 'best_cost' in execution_data:
                self._update_qaoa_metrics(execution_data)
        elif algorithm_type == 'vqe_plus':
            if 'best_energy' in execution_data:
                self._update_vqe_metrics(execution_data)
        
        # Update performance analytics
        self._update_performance_analytics(algorithm_type, execution_data)
    
    def _update_qaoa_metrics(self, execution_data: Dict[str, Any]) -> None:
        """Update QAOA-specific metrics."""
        metrics = self._quantum_algorithm_metrics['enhanced_qaoa']
        
        # Calculate convergence rate
        convergence_data = execution_data.get('convergence_data', {})
        if 'costs' in convergence_data and len(convergence_data['costs']) > 1:
            initial_cost = convergence_data['costs'][0]
            final_cost = convergence_data['costs'][-1]
            improvement = abs(initial_cost - final_cost) / abs(initial_cost) if initial_cost != 0 else 0
            
            # Running average
            current_rate = metrics.get('average_cost_improvement', 0.0)
            count = metrics['execution_count']
            metrics['average_cost_improvement'] = (current_rate * (count - 1) + improvement) / count
        
        # Multi-objective success rate
        if 'multi_objective' in execution_data and execution_data['multi_objective']:
            cost_components = execution_data.get('cost_components', {})
            if len(cost_components) > 1:
                success = all(abs(cost) < 10.0 for cost in cost_components.values())  # Threshold-based success
                current_success = metrics.get('multi_objective_success_rate', 0.0)
                count = metrics['execution_count']
                metrics['multi_objective_success_rate'] = (current_success * (count - 1) + (1.0 if success else 0.0)) / count
    
    def _update_vqe_metrics(self, execution_data: Dict[str, Any]) -> None:
        """Update VQE-specific metrics."""
        metrics = self._quantum_algorithm_metrics['vqe_plus']
        
        # Energy accuracy (relative to expected ground state)
        best_energy = execution_data.get('best_energy', 0.0)
        energy_accuracy = 1.0 / (1.0 + abs(best_energy))  # Simplified accuracy metric
        
        current_accuracy = metrics.get('average_energy_accuracy', 0.0)
        count = metrics['execution_count']
        metrics['average_energy_accuracy'] = (current_accuracy * (count - 1) + energy_accuracy) / count
        
        # QNG success rate
        optimization_details = execution_data.get('optimization_details', {})
        qng_enabled = optimization_details.get('qng_enabled', False)
        converged = execution_data.get('convergence_data', {}).get('converged', False)
        
        if qng_enabled:
            current_qng_success = metrics.get('qng_success_rate', 0.0)
            metrics['qng_success_rate'] = (current_qng_success * (count - 1) + (1.0 if converged else 0.0)) / count
        
        # Adaptive ansatz efficiency
        ansatz_structure = execution_data.get('ansatz_structure', {})
        if ansatz_structure.get('type') == 'adaptive':
            parameter_count = ansatz_structure.get('parameter_count', 0)
            depth = ansatz_structure.get('depth', 1)
            efficiency = 1.0 / (1.0 + parameter_count / depth) if depth > 0 else 0.0
            
            current_efficiency = metrics.get('adaptive_ansatz_efficiency', 0.0)
            metrics['adaptive_ansatz_efficiency'] = (current_efficiency * (count - 1) + efficiency) / count
    
    def _update_performance_analytics(self, algorithm_type: str, execution_data: Dict[str, Any]) -> None:
        """Update performance analytics for trend analysis."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Algorithm comparison
        if 'algorithm_comparison' not in self._performance_analytics:
            self._performance_analytics['algorithm_comparison'] = {}
        
        comparison_data = {
            'execution_time': execution_data.get('execution_time', 0.0),
            'timestamp': timestamp,
            'success': execution_data.get('converged', True) or execution_data.get('convergence_data', {}).get('converged', True)
        }
        
        if algorithm_type not in self._performance_analytics['algorithm_comparison']:
            self._performance_analytics['algorithm_comparison'][algorithm_type] = []
        
        self._performance_analytics['algorithm_comparison'][algorithm_type].append(comparison_data)
        
        # Keep only recent entries (last 100)
        if len(self._performance_analytics['algorithm_comparison'][algorithm_type]) > 100:
            self._performance_analytics['algorithm_comparison'][algorithm_type] = \
                self._performance_analytics['algorithm_comparison'][algorithm_type][-100:]
        
        # Resource utilization tracking
        photonic_impl = execution_data.get('photonic_implementation', {})
        if photonic_impl:
            resource_data = {
                'required_modes': photonic_impl.get('required_modes', 0),
                'gate_count': photonic_impl.get('gate_count', 0),
                'circuit_depth': photonic_impl.get('circuit_depth', 0),
                'timestamp': timestamp
            }
            
            if 'resource_utilization' not in self._performance_analytics:
                self._performance_analytics['resource_utilization'] = []
            
            self._performance_analytics['resource_utilization'].append(resource_data)
            
            # Keep only recent entries
            if len(self._performance_analytics['resource_utilization']) > 1000:
                self._performance_analytics['resource_utilization'] = \
                    self._performance_analytics['resource_utilization'][-1000:]
    
    def get_algorithm_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive algorithm performance summary."""
        summary = {
            'quantum_algorithms': {},
            'performance_trends': {},
            'resource_efficiency': {},
            'optimization_insights': {}
        }
        
        # Quantum algorithm summaries
        for algo_type, metrics in self._quantum_algorithm_metrics.items():
            if isinstance(metrics, dict) and 'execution_count' in metrics and metrics['execution_count'] > 0:
                avg_time = metrics['total_execution_time'] / metrics['execution_count']
                summary['quantum_algorithms'][algo_type] = {
                    'executions': metrics['execution_count'],
                    'avg_execution_time': avg_time,
                    'performance_score': self._calculate_performance_score(algo_type, metrics)
                }
        
        # Performance trends
        comparison_data = self._performance_analytics.get('algorithm_comparison', {})
        for algo_type, executions in comparison_data.items():
            if len(executions) >= 5:  # Need at least 5 executions for trends
                recent_times = [e['execution_time'] for e in executions[-10:]]
                trend = 'improving' if len(recent_times) > 1 and recent_times[-1] < recent_times[0] else 'stable'
                summary['performance_trends'][algo_type] = {
                    'trend': trend,
                    'recent_avg_time': sum(recent_times) / len(recent_times),
                    'success_rate': sum(1 for e in executions[-20:] if e['success']) / min(20, len(executions))
                }
        
        # Resource efficiency analysis
        resource_data = self._performance_analytics.get('resource_utilization', [])
        if resource_data:
            recent_resources = resource_data[-50:]  # Last 50 executions
            avg_modes = sum(r['required_modes'] for r in recent_resources) / len(recent_resources)
            avg_depth = sum(r['circuit_depth'] for r in recent_resources) / len(recent_resources)
            
            summary['resource_efficiency'] = {
                'avg_required_modes': avg_modes,
                'avg_circuit_depth': avg_depth,
                'efficiency_score': self._calculate_efficiency_score(recent_resources)
            }
        
        return summary
    
    def _calculate_performance_score(self, algo_type: str, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score for an algorithm."""
        base_score = 0.5  # Base score
        
        # Execution count bonus (more executions = more confidence)
        execution_bonus = min(0.2, metrics['execution_count'] / 100.0)
        
        # Algorithm-specific scoring
        if algo_type == 'enhanced_qaoa':
            improvement_score = metrics.get('average_cost_improvement', 0.0) * 0.3
            multi_obj_score = metrics.get('multi_objective_success_rate', 0.0) * 0.2
            algo_score = improvement_score + multi_obj_score
        elif algo_type == 'vqe_plus':
            accuracy_score = metrics.get('average_energy_accuracy', 0.0) * 0.3
            qng_score = metrics.get('qng_success_rate', 0.0) * 0.15
            efficiency_score = metrics.get('adaptive_ansatz_efficiency', 0.0) * 0.15
            algo_score = accuracy_score + qng_score + efficiency_score
        else:
            algo_score = 0.0
        
        # Time penalty (faster is better)
        avg_time = metrics['total_execution_time'] / metrics['execution_count']
        time_penalty = min(0.1, avg_time / 100.0)  # Penalty for slow execution
        
        final_score = base_score + execution_bonus + algo_score - time_penalty
        return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
    
    def _calculate_efficiency_score(self, resource_data: List[Dict[str, Any]]) -> float:
        """Calculate resource efficiency score."""
        if not resource_data:
            return 0.0
        
        # Efficiency based on modes per gate and depth per mode ratios
        efficiency_scores = []
        for data in resource_data:
            modes = data.get('required_modes', 1)
            gates = data.get('gate_count', 1)
            depth = data.get('circuit_depth', 1)
            
            # Lower ratios are better (more efficient)
            gate_efficiency = 1.0 / (1.0 + gates / modes) if modes > 0 else 0.0
            depth_efficiency = 1.0 / (1.0 + depth / modes) if modes > 0 else 0.0
            
            combined_efficiency = (gate_efficiency + depth_efficiency) / 2.0
            efficiency_scores.append(combined_efficiency)
        
        return sum(efficiency_scores) / len(efficiency_scores)
    
    def _start_export_thread(self) -> None:
        """Start background thread for metric export."""
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()
    
    def _export_loop(self) -> None:
        """Background loop for periodic metric export."""
        while not self._stop_export.wait(self.export_interval):
            try:
                # Export metrics to file (in production, this would be to monitoring system)
                export_data = self.export_metrics()
                
                # Write to monitoring file
                metrics_file = Path("./monitoring_metrics.jsonl")
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(export_data) + "\n")
                
                # Clear old metrics to prevent memory buildup
                cutoff_time = datetime.now(timezone.utc)
                with self._lock:
                    self._metrics = [m for m in self._metrics 
                                   if (cutoff_time - m.timestamp).seconds < 3600]  # Keep 1 hour
                
            except Exception as e:
                # Log error but don't crash the export thread
                logging.error(f"Error exporting metrics: {e}")
    
    def shutdown(self) -> None:
        """Shutdown metrics collector."""
        if self._export_thread:
            self._stop_export.set()
            self._export_thread.join(timeout=5.0)


class StructuredLogger:
    """Structured logging with consistent format and context."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._setup_handler()
        self._log_events: List[LogEvent] = []
        self._lock = threading.Lock()
    
    def _setup_handler(self) -> None:
        """Set up log handler with structured format."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_event(self, level: str, message: str, component: str = "",
                  operation: str = "", duration_ms: Optional[float] = None,
                  error: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a structured event.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message
            component: Component name
            operation: Operation name
            duration_ms: Operation duration in milliseconds
            error: Error message if applicable
            context: Additional context data
        """
        event = LogEvent(
            level=level,
            message=message,
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            error=error,
            context=context or {}
        )
        
        # Store event for analysis
        with self._lock:
            self._log_events.append(event)
        
        # Log to standard logger
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Format message with context
        formatted_msg = message
        if component:
            formatted_msg = f"[{component}] {formatted_msg}"
        if operation:
            formatted_msg = f"[{operation}] {formatted_msg}"
        if duration_ms is not None:
            formatted_msg = f"{formatted_msg} (took {duration_ms:.2f}ms)"
        
        # Log to standard logger (avoid conflicts with LogRecord fields)
        extra_data = {k: v for k, v in event.to_dict().items() 
                     if k not in ['message', 'levelname', 'msg']}
        self.logger.log(log_level, formatted_msg, extra=extra_data)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log_event("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log_event("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log_event("ERROR", message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log_event("DEBUG", message, **kwargs)
    
    def get_events(self, level: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[LogEvent]:
        """Get logged events.
        
        Args:
            level: Filter by log level
            since: Only return events after this timestamp
            
        Returns:
            List of log events
        """
        with self._lock:
            events = self._log_events.copy()
        
        if level:
            events = [e for e in events if e.level == level.upper()]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events


class PerformanceMonitor:
    """Monitor performance metrics and resource usage."""
    
    def __init__(self, metrics_collector: MetricsCollector, logger: StructuredLogger):
        """Initialize performance monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            logger: Structured logger instance
        """
        self.metrics = metrics_collector
        self.logger = logger
        self._active_operations: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def measure_operation(self, operation: str, component: str = "",
                         tags: Optional[Dict[str, str]] = None):
        """Context manager for measuring operation performance.
        
        Args:
            operation: Operation name
            component: Component name
            tags: Additional tags for metrics
        """
        start_time = time.time()
        operation_id = f"{component}:{operation}" if component else operation
        
        with self._lock:
            self._active_operations[operation_id] = start_time
        
        try:
            self.logger.info(f"Starting {operation}", component=component, operation=operation)
            yield
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"Operation {operation} failed: {str(e)}",
                component=component,
                operation=operation,
                duration_ms=duration_ms,
                error=str(e)
            )
            
            # Record failure metric
            failure_tags = (tags or {}).copy()
            failure_tags.update({"operation": operation, "component": component, "status": "error"})
            self.metrics.increment_counter("operation_failures", failure_tags)
            
            raise
        
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            with self._lock:
                self._active_operations.pop(operation_id, None)
            
            # Record timing metrics
            timing_tags = (tags or {}).copy()
            timing_tags.update({"operation": operation, "component": component})
            self.metrics.record_histogram("operation_duration", duration_ms, timing_tags, "ms")
            
            # Log completion
            self.logger.info(
                f"Completed {operation}",
                component=component,
                operation=operation,
                duration_ms=duration_ms
            )
    
    def record_resource_usage(self, tags: Optional[Dict[str, str]] = None) -> None:
        """Record current resource usage metrics.
        
        Args:
            tags: Additional tags for metrics
        """
        try:
            import psutil
            import os
            
            # Memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.record_gauge("memory_usage", memory_mb, tags, "MB")
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.metrics.record_gauge("cpu_usage", cpu_percent, tags, "percent")
            
            # Active operations count
            with self._lock:
                active_ops = len(self._active_operations)
            self.metrics.record_gauge("active_operations", active_ops, tags, "count")
            
        except ImportError:
            # psutil not available, skip resource monitoring
            pass
        except Exception as e:
            self.logger.warning(f"Failed to collect resource metrics: {e}")


class HealthChecker:
    """System health monitoring and checks."""
    
    def __init__(self, logger: StructuredLogger):
        """Initialize health checker.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        self._health_status: Dict[str, Dict[str, Any]] = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], bool],
                            description: str = "") -> None:
        """Register a health check function.
        
        Args:
            name: Health check name
            check_func: Function that returns True if healthy
            description: Description of what this check validates
        """
        self._health_checks[name] = check_func
        self._health_status[name] = {
            "description": description,
            "status": "unknown",
            "last_check": None,
            "error": None
        }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks.
        
        Returns:
            Dictionary with health check results
        """
        overall_healthy = True
        
        for name, check_func in self._health_checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration_ms = (time.time() - start_time) * 1000
                
                self._health_status[name].update({
                    "status": "healthy" if is_healthy else "unhealthy",
                    "last_check": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": duration_ms,
                    "error": None
                })
                
                if not is_healthy:
                    overall_healthy = False
                    self.logger.warning(f"Health check {name} failed")
                
            except Exception as e:
                overall_healthy = False
                error_msg = str(e)
                
                self._health_status[name].update({
                    "status": "error",
                    "last_check": datetime.now(timezone.utc).isoformat(),
                    "error": error_msg
                })
                
                self.logger.error(f"Health check {name} raised exception: {error_msg}")
        
        return {
            "overall_healthy": overall_healthy,
            "checks": self._health_status.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status without running checks.
        
        Returns:
            Current health status
        """
        return {
            "checks": self._health_status.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global monitoring instances
_metrics_collector: Optional[MetricsCollector] = None
_logger: Optional[StructuredLogger] = None
_performance_monitor: Optional[PerformanceMonitor] = None
_health_checker: Optional[HealthChecker] = None


def initialize_monitoring(enable_metrics: bool = True, log_level: int = logging.INFO) -> None:
    """Initialize global monitoring components.
    
    Args:
        enable_metrics: Whether to enable metrics collection
        log_level: Logging level
    """
    global _metrics_collector, _logger, _performance_monitor, _health_checker
    
    _metrics_collector = MetricsCollector(enable_export=enable_metrics)
    _logger = StructuredLogger("holo_code_gen", log_level)
    _performance_monitor = PerformanceMonitor(_metrics_collector, _logger)
    _health_checker = HealthChecker(_logger)
    
    _logger.info("Monitoring system initialized", component="monitoring")


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    if _metrics_collector is None:
        initialize_monitoring()
    return _metrics_collector


def get_logger() -> StructuredLogger:
    """Get global structured logger."""
    if _logger is None:
        initialize_monitoring()
    return _logger


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    if _performance_monitor is None:
        initialize_monitoring()
    return _performance_monitor


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    if _health_checker is None:
        initialize_monitoring()
    return _health_checker


def monitor_function(operation: str, component: str = ""):
    """Decorator to monitor function performance.
    
    Args:
        operation: Operation name
        component: Component name
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.measure_operation(operation, component):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_exceptions(component: str = ""):
    """Decorator to log exceptions with context.
    
    Args:
        component: Component name
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HoloCodeGenException as e:
                logger = get_logger()
                logger.error(
                    f"Exception in {func.__name__}: {e.message}",
                    component=component,
                    operation=func.__name__,
                    error=e.error_code,
                    context=e.context
                )
                raise
            except Exception as e:
                logger = get_logger()
                logger.error(
                    f"Unexpected exception in {func.__name__}: {str(e)}",
                    component=component,
                    operation=func.__name__,
                    error="UNEXPECTED_ERROR",
                    context={"traceback": traceback.format_exc()}
                )
                raise
        return wrapper
    return decorator