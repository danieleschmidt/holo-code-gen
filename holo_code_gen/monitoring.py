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
    """Collects and manages metrics for monitoring."""
    
    def __init__(self, enable_export: bool = True, export_interval: float = 60.0):
        """Initialize metrics collector.
        
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
        """Export metrics in standardized format."""
        metrics_data = []
        
        with self._lock:
            for metric in self._metrics:
                metrics_data.append(metric.to_dict())
        
        return {
            "metrics": metrics_data,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_metrics": len(metrics_data)
        }
    
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