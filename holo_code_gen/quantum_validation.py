"""Advanced validation and robustness for quantum algorithms."""

from typing import Dict, List, Any, Optional, Tuple
import time
import json
from .exceptions import ValidationError, ErrorCodes, CompilationError, SecurityError
from .monitoring import get_logger, monitor_function


logger = get_logger()


class QuantumParameterValidator:
    """Advanced parameter validation for quantum algorithms."""
    
    def __init__(self):
        """Initialize quantum parameter validator."""
        self.logger = logger
        self.validation_cache = {}
    
    @monitor_function("parameter_validation", "quantum_validation")
    def validate_cv_qaoa_parameters(self, problem_graph: Dict[str, Any], 
                                   depth: int, max_iterations: int) -> Dict[str, Any]:
        """Validate CV-QAOA parameters with comprehensive checks."""
        validation_results = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "parameter_adjustments": {}
        }
        
        # Validate problem graph structure
        if not self._validate_problem_graph(problem_graph, validation_results):
            validation_results["validation_passed"] = False
        
        # Validate depth parameter
        if not self._validate_circuit_depth(depth, validation_results):
            validation_results["validation_passed"] = False
        
        # Validate iteration count
        if not self._validate_iteration_count(max_iterations, validation_results):
            validation_results["validation_passed"] = False
        
        # Security checks for resource limits
        if not self._validate_resource_limits(problem_graph, depth, max_iterations, validation_results):
            validation_results["validation_passed"] = False
        
        return validation_results
    
    def _validate_problem_graph(self, problem_graph: Dict[str, Any], 
                               validation_results: Dict[str, Any]) -> bool:
        """Validate problem graph structure and content."""
        if not isinstance(problem_graph, dict):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                "message": "Problem graph must be a dictionary",
                "field": "problem_graph"
            })
            return False
        
        # Check required fields
        required_fields = ["nodes", "edges"]
        for field in required_fields:
            if field not in problem_graph:
                validation_results["errors"].append({
                    "code": ErrorCodes.MISSING_REQUIRED_PARAMETER,
                    "message": f"Problem graph missing required field: {field}",
                    "field": field
                })
                return False
        
        # Validate nodes
        nodes = problem_graph["nodes"]
        if not isinstance(nodes, list) or len(nodes) == 0:
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                "message": "Nodes must be a non-empty list",
                "field": "nodes"
            })
            return False
        
        # Check node count limits for security
        if len(nodes) > 1000:
            validation_results["errors"].append({
                "code": ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
                "message": f"Too many nodes: {len(nodes)} > 1000",
                "field": "nodes"
            })
            return False
        
        # Validate edges
        edges = problem_graph["edges"]
        if not isinstance(edges, list):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                "message": "Edges must be a list",
                "field": "edges"
            })
            return False
        
        # Check edge count limits
        if len(edges) > 10000:
            validation_results["errors"].append({
                "code": ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
                "message": f"Too many edges: {len(edges)} > 10000",
                "field": "edges"
            })
            return False
        
        # Validate each edge
        node_set = set(nodes)
        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                validation_results["errors"].append({
                    "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                    "message": f"Edge {i} must be a dictionary",
                    "field": f"edges[{i}]"
                })
                return False
            
            if "nodes" not in edge:
                validation_results["errors"].append({
                    "code": ErrorCodes.MISSING_REQUIRED_PARAMETER,
                    "message": f"Edge {i} missing 'nodes' field",
                    "field": f"edges[{i}].nodes"
                })
                return False
            
            edge_nodes = edge["nodes"]
            if not isinstance(edge_nodes, list) or len(edge_nodes) != 2:
                validation_results["errors"].append({
                    "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                    "message": f"Edge {i} nodes must be a list of 2 elements",
                    "field": f"edges[{i}].nodes"
                })
                return False
            
            # Check if edge nodes exist in node list
            for node in edge_nodes:
                if node not in node_set:
                    validation_results["errors"].append({
                        "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                        "message": f"Edge {i} references non-existent node: {node}",
                        "field": f"edges[{i}].nodes"
                    })
                    return False
            
            # Validate edge weight if present
            if "weight" in edge:
                weight = edge["weight"]
                if not isinstance(weight, (int, float)):
                    validation_results["errors"].append({
                        "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                        "message": f"Edge {i} weight must be numeric",
                        "field": f"edges[{i}].weight"
                    })
                    return False
                
                if weight < 0:
                    validation_results["warnings"].append({
                        "message": f"Edge {i} has negative weight: {weight}",
                        "field": f"edges[{i}].weight"
                    })
        
        return True
    
    def _validate_circuit_depth(self, depth: int, validation_results: Dict[str, Any]) -> bool:
        """Validate circuit depth parameter."""
        if not isinstance(depth, int):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                "message": "Circuit depth must be an integer",
                "field": "depth"
            })
            return False
        
        if depth < 1:
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                "message": f"Circuit depth must be positive: {depth}",
                "field": "depth"
            })
            return False
        
        if depth > 100:
            validation_results["errors"].append({
                "code": ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
                "message": f"Circuit depth too large: {depth} > 100",
                "field": "depth"
            })
            return False
        
        if depth > 20:
            validation_results["warnings"].append({
                "message": f"Large circuit depth may impact performance: {depth}",
                "field": "depth"
            })
        
        return True
    
    def _validate_iteration_count(self, max_iterations: int, validation_results: Dict[str, Any]) -> bool:
        """Validate iteration count parameter."""
        if not isinstance(max_iterations, int):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                "message": "Max iterations must be an integer",
                "field": "max_iterations"
            })
            return False
        
        if max_iterations < 1:
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                "message": f"Max iterations must be positive: {max_iterations}",
                "field": "max_iterations"
            })
            return False
        
        if max_iterations > 10000:
            validation_results["errors"].append({
                "code": ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
                "message": f"Too many iterations: {max_iterations} > 10000",
                "field": "max_iterations"
            })
            return False
        
        if max_iterations > 1000:
            validation_results["warnings"].append({
                "message": f"Large iteration count may impact performance: {max_iterations}",
                "field": "max_iterations"
            })
        
        return True
    
    def _validate_resource_limits(self, problem_graph: Dict[str, Any], depth: int, 
                                 max_iterations: int, validation_results: Dict[str, Any]) -> bool:
        """Validate overall resource consumption limits."""
        num_nodes = len(problem_graph["nodes"])
        num_edges = len(problem_graph["edges"])
        
        # Estimate computational complexity
        complexity_estimate = num_nodes * depth * max_iterations + num_edges * depth
        
        if complexity_estimate > 1000000:  # 1M operations
            validation_results["errors"].append({
                "code": ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
                "message": f"Computational complexity too high: {complexity_estimate}",
                "field": "computational_complexity"
            })
            return False
        
        # Estimate memory usage
        memory_estimate = num_nodes * depth * 8 + num_edges * 4  # bytes
        
        if memory_estimate > 100 * 1024 * 1024:  # 100MB
            validation_results["errors"].append({
                "code": ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
                "message": f"Memory usage too high: {memory_estimate} bytes",
                "field": "memory_usage"
            })
            return False
        
        return True

    @monitor_function("error_correction_validation", "quantum_validation")
    def validate_error_correction_parameters(self, logical_qubits: int, error_rate: float, 
                                           code_type: str) -> Dict[str, Any]:
        """Validate error correction parameters."""
        validation_results = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "parameter_adjustments": {}
        }
        
        # Validate logical qubits
        if not isinstance(logical_qubits, int):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                "message": "Logical qubits must be an integer",
                "field": "logical_qubits"
            })
            validation_results["validation_passed"] = False
        elif logical_qubits <= 0:
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                "message": f"Logical qubits must be positive: {logical_qubits}",
                "field": "logical_qubits"
            })
            validation_results["validation_passed"] = False
        elif logical_qubits > 1000:
            validation_results["errors"].append({
                "code": ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
                "message": f"Too many logical qubits: {logical_qubits} > 1000",
                "field": "logical_qubits"
            })
            validation_results["validation_passed"] = False
        
        # Validate error rate
        if not isinstance(error_rate, (int, float)):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                "message": "Error rate must be numeric",
                "field": "error_rate"
            })
            validation_results["validation_passed"] = False
        elif not (0 < error_rate < 1):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                "message": f"Error rate must be between 0 and 1: {error_rate}",
                "field": "error_rate"
            })
            validation_results["validation_passed"] = False
        
        # Validate code type
        valid_codes = ["surface", "color", "repetition"]
        if not isinstance(code_type, str):
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_TYPE,
                "message": "Code type must be a string",
                "field": "code_type"
            })
            validation_results["validation_passed"] = False
        elif code_type not in valid_codes:
            validation_results["errors"].append({
                "code": ErrorCodes.INVALID_PARAMETER_VALUE,
                "message": f"Unsupported code type: {code_type}. Valid: {valid_codes}",
                "field": "code_type"
            })
            validation_results["validation_passed"] = False
        
        # Check threshold conditions
        if validation_results["validation_passed"]:
            thresholds = {
                "surface": 0.0109,
                "color": 0.0074,
                "repetition": 0.5
            }
            
            threshold = thresholds.get(code_type, 0.01)
            if error_rate > threshold:
                validation_results["errors"].append({
                    "code": ErrorCodes.THRESHOLD_EXCEEDED,
                    "message": f"Error rate {error_rate} exceeds {code_type} code threshold {threshold}",
                    "field": "error_rate"
                })
                validation_results["validation_passed"] = False
        
        return validation_results


class QuantumSecurityValidator:
    """Security validation for quantum algorithms."""
    
    def __init__(self):
        """Initialize quantum security validator."""
        self.logger = logger
        self.max_input_size = 1024 * 1024  # 1MB
        self.max_string_length = 10000
    
    @monitor_function("security_validation", "quantum_validation")
    def validate_input_safety(self, input_data: Any) -> Dict[str, Any]:
        """Validate input data for security concerns."""
        validation_results = {
            "security_passed": True,
            "threats_detected": [],
            "sanitized_data": input_data
        }
        
        # Check input size
        try:
            input_size = len(str(input_data))
            if input_size > self.max_input_size:
                validation_results["security_passed"] = False
                validation_results["threats_detected"].append({
                    "threat_type": "oversized_input",
                    "description": f"Input size {input_size} exceeds limit {self.max_input_size}",
                    "severity": "high"
                })
        except Exception:
            pass
        
        # Check for potentially malicious strings
        if isinstance(input_data, str):
            if len(input_data) > self.max_string_length:
                validation_results["security_passed"] = False
                validation_results["threats_detected"].append({
                    "threat_type": "oversized_string",
                    "description": f"String length {len(input_data)} exceeds limit {self.max_string_length}",
                    "severity": "medium"
                })
            
            # Check for suspicious patterns
            suspicious_patterns = [
                "__import__", "eval", "exec", "subprocess", "os.system",
                "open(", "file(", "input(", "raw_input", "compile"
            ]
            
            for pattern in suspicious_patterns:
                if pattern in input_data:
                    validation_results["security_passed"] = False
                    validation_results["threats_detected"].append({
                        "threat_type": "malicious_code",
                        "description": f"Suspicious pattern detected: {pattern}",
                        "severity": "critical"
                    })
        
        # Recursively check dictionary and list structures
        elif isinstance(input_data, dict):
            for key, value in input_data.items():
                key_validation = self.validate_input_safety(key)
                value_validation = self.validate_input_safety(value)
                
                if not key_validation["security_passed"]:
                    validation_results["security_passed"] = False
                    validation_results["threats_detected"].extend(key_validation["threats_detected"])
                
                if not value_validation["security_passed"]:
                    validation_results["security_passed"] = False
                    validation_results["threats_detected"].extend(value_validation["threats_detected"])
        
        elif isinstance(input_data, list):
            for item in input_data:
                item_validation = self.validate_input_safety(item)
                if not item_validation["security_passed"]:
                    validation_results["security_passed"] = False
                    validation_results["threats_detected"].extend(item_validation["threats_detected"])
        
        return validation_results
    
    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters to prevent injection attacks."""
        sanitized = {}
        
        for key, value in params.items():
            # Sanitize key
            safe_key = str(key)[:100]  # Limit key length
            safe_key = ''.join(c for c in safe_key if c.isalnum() or c in '_-.')
            
            # Sanitize value based on type
            if isinstance(value, str):
                # Remove potentially dangerous characters
                safe_value = value[:self.max_string_length]
                safe_value = ''.join(c for c in safe_value if ord(c) < 127 and c.isprintable())
            elif isinstance(value, (int, float)):
                # Clamp numeric values to reasonable ranges
                if isinstance(value, int):
                    safe_value = max(-1000000, min(1000000, value))
                else:
                    safe_value = max(-1e6, min(1e6, value))
            elif isinstance(value, (list, dict)):
                # Recursively sanitize collections
                if isinstance(value, list) and len(value) <= 10000:
                    safe_value = [self.sanitize_parameters({"item": item}).get("item", item) 
                                 for item in value[:1000]]
                elif isinstance(value, dict) and len(value) <= 1000:
                    safe_value = self.sanitize_parameters(value)
                else:
                    safe_value = None  # Drop oversized collections
            else:
                safe_value = value  # Keep other types as-is
            
            if safe_value is not None:
                sanitized[safe_key] = safe_value
        
        return sanitized


class QuantumHealthMonitor:
    """Health monitoring for quantum algorithm execution."""
    
    def __init__(self):
        """Initialize quantum health monitor."""
        self.logger = logger
        self.execution_stats = {}
        self.error_counts = {}
    
    @monitor_function("health_check", "quantum_validation")
    def check_algorithm_health(self, algorithm_name: str, 
                              execution_time: float, 
                              result: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of quantum algorithm execution."""
        health_status = {
            "status": "healthy",
            "alerts": [],
            "metrics": {},
            "recommendations": []
        }
        
        # Update execution statistics
        if algorithm_name not in self.execution_stats:
            self.execution_stats[algorithm_name] = {
                "total_executions": 0,
                "total_time": 0.0,
                "success_count": 0,
                "error_count": 0,
                "avg_execution_time": 0.0
            }
        
        stats = self.execution_stats[algorithm_name]
        stats["total_executions"] += 1
        stats["total_time"] += execution_time
        stats["avg_execution_time"] = stats["total_time"] / stats["total_executions"]
        
        # Check execution time
        if execution_time > 10.0:  # 10 seconds
            health_status["status"] = "warning"
            health_status["alerts"].append({
                "type": "performance",
                "message": f"Slow execution time: {execution_time:.2f}s",
                "severity": "medium"
            })
        
        # Check result quality
        if "converged" in result and not result["converged"]:
            health_status["status"] = "warning"
            health_status["alerts"].append({
                "type": "convergence",
                "message": "Algorithm did not converge",
                "severity": "medium"
            })
        
        # Check for numerical issues
        if "optimal_cost" in result:
            cost = result["optimal_cost"]
            if not isinstance(cost, (int, float)) or cost != cost:  # NaN check
                health_status["status"] = "error"
                health_status["alerts"].append({
                    "type": "numerical",
                    "message": "Invalid numerical result detected",
                    "severity": "high"
                })
        
        # Performance trend analysis
        if stats["total_executions"] > 10:
            recent_avg_time = execution_time
            historical_avg_time = stats["avg_execution_time"]
            
            if recent_avg_time > historical_avg_time * 2:
                health_status["alerts"].append({
                    "type": "performance_degradation",
                    "message": f"Performance degraded: {recent_avg_time:.2f}s vs {historical_avg_time:.2f}s avg",
                    "severity": "medium"
                })
        
        # Generate metrics
        health_status["metrics"] = {
            "execution_time": execution_time,
            "avg_execution_time": stats["avg_execution_time"],
            "total_executions": stats["total_executions"],
            "success_rate": (stats["success_count"] / stats["total_executions"]) if stats["total_executions"] > 0 else 0
        }
        
        # Generate recommendations
        if execution_time > 5.0:
            health_status["recommendations"].append("Consider reducing problem size or iteration count")
        
        if "converged" in result and not result["converged"]:
            health_status["recommendations"].append("Increase max_iterations or adjust learning rate")
        
        return health_status
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.execution_stats:
            return {
                "status": "unknown",
                "message": "No execution data available"
            }
        
        total_executions = sum(stats["total_executions"] for stats in self.execution_stats.values())
        total_errors = sum(self.error_counts.values())
        
        error_rate = total_errors / total_executions if total_executions > 0 else 0
        
        if error_rate > 0.1:  # >10% error rate
            status = "critical"
        elif error_rate > 0.05:  # >5% error rate
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "total_executions": total_executions,
            "error_rate": error_rate,
            "algorithms": list(self.execution_stats.keys()),
            "summary": {
                "avg_execution_time": sum(stats["avg_execution_time"] for stats in self.execution_stats.values()) / len(self.execution_stats),
                "most_used_algorithm": max(self.execution_stats.keys(), key=lambda k: self.execution_stats[k]["total_executions"]) if self.execution_stats else None
            }
        }