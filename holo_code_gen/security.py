"""Security utilities and validation for Holo-Code-Gen."""

import hashlib
import hmac
import secrets
import re
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import json
from dataclasses import dataclass

from .exceptions import SecurityError, ValidationError, ErrorCodes


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_file_size_mb: float = 100.0  # Maximum file size in MB
    allowed_file_extensions: Set[str] = None  # Allowed file extensions
    max_circuit_components: int = 10000  # Maximum components per circuit
    max_graph_nodes: int = 10000  # Maximum nodes per graph
    enable_input_sanitization: bool = True
    enable_path_validation: bool = True
    enable_resource_limits: bool = True
    
    def __post_init__(self):
        """Initialize default allowed extensions."""
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = {
                '.py', '.json', '.yaml', '.yml', '.gds', '.spi', '.spice',
                '.md', '.txt', '.csv', '.h5', '.hdf5'
            }


class InputSanitizer:
    """Sanitizes and validates user inputs."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize input sanitizer.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        
        # Compile regex patterns for efficiency  
        self._malicious_patterns = [
            re.compile(r'<script[^>]*>', re.IGNORECASE),  # Script tags
            re.compile(r'javascript:', re.IGNORECASE),    # JavaScript URLs
            re.compile(r'on\w+\s*=', re.IGNORECASE),      # Event handlers
            re.compile(r'eval\s*\(', re.IGNORECASE),      # eval() calls
            re.compile(r'exec\s*\(', re.IGNORECASE),      # exec() calls
            re.compile(r'__import__', re.IGNORECASE),     # Dynamic imports
            re.compile(r'\.\.[\\/]', re.IGNORECASE),      # Path traversal
        ]
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If malicious content detected
            ValidationError: If input is invalid
        """
        if not isinstance(value, str):
            raise ValidationError(
                "Input must be a string",
                field="input",
                value=type(value).__name__
            )
        
        # Check length
        if len(value) > max_length:
            raise ValidationError(
                f"Input too long: {len(value)} chars, max {max_length}",
                field="input",
                value=len(value)
            )
        
        # Check for malicious patterns
        for pattern in self._malicious_patterns:
            if pattern.search(value):
                raise SecurityError(
                    f"Malicious pattern detected in input: {pattern.pattern}",
                    error_code=ErrorCodes.MALICIOUS_INPUT_DETECTED
                )
        
        # Basic sanitization
        sanitized = value.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem operations.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
            
        Raises:
            SecurityError: If filename is malicious
        """
        if not filename:
            raise ValidationError("Filename cannot be empty", field="filename")
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise SecurityError(
                "Path traversal detected in filename",
                error_code=ErrorCodes.MALICIOUS_INPUT_DETECTED
            )
        
        # Check file extension
        path = Path(filename)
        if path.suffix.lower() not in self.config.allowed_file_extensions:
            raise SecurityError(
                f"File extension {path.suffix} not allowed",
                error_code=ErrorCodes.MALICIOUS_INPUT_DETECTED
            )
        
        # Remove dangerous characters
        safe_chars = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(safe_chars) > 255:
            safe_chars = safe_chars[:255]
        
        return safe_chars
    
    def validate_json_input(self, json_str: str, max_depth: int = 10) -> Dict[str, Any]:
        """Validate and parse JSON input safely.
        
        Args:
            json_str: JSON string
            max_depth: Maximum nesting depth
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON is invalid or malicious
        """
        try:
            # Basic sanitization
            sanitized_json = self.sanitize_string(json_str, max_length=1000000)  # 1MB max
            
            # Parse JSON
            data = json.loads(sanitized_json)
            
            # Check nesting depth
            if self._get_json_depth(data) > max_depth:
                raise ValidationError(
                    f"JSON nesting too deep: max {max_depth} levels",
                    field="json_depth"
                )
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON: {str(e)}",
                field="json_input",
                error_code=ErrorCodes.INVALID_PARAMETER_VALUE
            )
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Get maximum nesting depth of JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth


class ResourceLimiter:
    """Enforces resource limits to prevent abuse."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize resource limiter.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
    
    def check_file_size(self, file_path: Path) -> None:
        """Check if file size is within limits.
        
        Args:
            file_path: Path to file to check
            
        Raises:
            SecurityError: If file is too large
        """
        if not file_path.exists():
            return
        
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise SecurityError(
                f"File too large: {size_mb:.1f}MB, max {self.config.max_file_size_mb}MB",
                error_code=ErrorCodes.RESOURCE_LIMIT_EXCEEDED
            )
    
    def check_circuit_complexity(self, num_components: int) -> None:
        """Check if circuit complexity is within limits.
        
        Args:
            num_components: Number of components in circuit
            
        Raises:
            SecurityError: If circuit is too complex
        """
        if num_components > self.config.max_circuit_components:
            raise SecurityError(
                f"Circuit too complex: {num_components} components, "
                f"max {self.config.max_circuit_components}",
                error_code="CIRCUIT_COMPLEXITY_EXCEEDED"
            )
    
    def check_quantum_circuit_complexity(self, qubit_count: int, operation_count: int) -> None:
        """Check if quantum circuit complexity is within limits.
        
        Args:
            qubit_count: Number of qubits
            operation_count: Number of quantum operations
            
        Raises:
            SecurityError: If circuit is too complex
        """
        max_qubits = 50  # Reasonable limit for photonic quantum circuits
        max_operations = 1000  # Maximum quantum operations
        
        if qubit_count > max_qubits:
            raise SecurityError(
                f"Too many qubits: {qubit_count}, maximum allowed: {max_qubits}",
                error_code=ErrorCodes.RESOURCE_LIMIT_EXCEEDED
            )
        
        if operation_count > max_operations:
            raise SecurityError(
                f"Too many operations: {operation_count}, maximum allowed: {max_operations}",
                error_code=ErrorCodes.RESOURCE_LIMIT_EXCEEDED
            )
        
        # Check combined complexity (qubits × operations)
        complexity = qubit_count * operation_count
        max_complexity = 10000
        
        if complexity > max_complexity:
            raise SecurityError(
                f"Circuit too complex: {complexity}, maximum allowed: {max_complexity}",
                error_code=ErrorCodes.RESOURCE_LIMIT_EXCEEDED
            )
    
    def check_graph_complexity(self, num_nodes: int) -> None:
        """Check if graph complexity is within limits.
        
        Args:
            num_nodes: Number of nodes in graph
            
        Raises:
            SecurityError: If graph is too complex
        """
        if num_nodes > self.config.max_graph_nodes:
            raise SecurityError(
                f"Graph too complex: {num_nodes} nodes, max {self.config.max_graph_nodes}",
                error_code="GRAPH_COMPLEXITY_EXCEEDED"
            )


class ParameterValidator:
    """Validates parameters for security and correctness."""
    
    SAFE_PARAMETER_PATTERNS = {
        'component_type': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
        'node_name': re.compile(r'^[a-zA-Z0-9_\-\.]+$'),
        'wavelength': (1000.0, 2000.0),  # nm, reasonable range
        'power': (0.0, 10000.0),  # mW, reasonable range
        'radius': (1.0, 1000.0),  # μm, reasonable range
        'width': (0.1, 100.0),    # μm, reasonable range
        'length': (1.0, 100000.0), # μm, reasonable range
    }
    
    def validate_parameter(self, name: str, value: Any) -> Any:
        """Validate a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Validated value (may be converted)
            
        Raises:
            ValidationError: If parameter is invalid
        """
        if name in self.SAFE_PARAMETER_PATTERNS:
            pattern_or_range = self.SAFE_PARAMETER_PATTERNS[name]
            
            if isinstance(pattern_or_range, re.Pattern):
                # String pattern validation
                if not isinstance(value, str):
                    raise ValidationError(
                        f"Parameter {name} must be string",
                        field=name,
                        value=type(value).__name__
                    )
                
                if not pattern_or_range.match(value):
                    raise ValidationError(
                        f"Parameter {name} has invalid format",
                        field=name,
                        value=value
                    )
                
                return value
            
            elif isinstance(pattern_or_range, tuple):
                # Numeric range validation
                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Parameter {name} must be numeric",
                        field=name,
                        value=value
                    )
                
                min_val, max_val = pattern_or_range
                if not (min_val <= numeric_value <= max_val):
                    raise ValidationError(
                        f"Parameter {name} must be between {min_val} and {max_val}",
                        field=name,
                        value=numeric_value
                    )
                
                return numeric_value
        
        # Generic validation for unknown parameters
        if isinstance(value, str):
            if len(value) > 1000:  # Prevent extremely long strings
                raise ValidationError(
                    f"Parameter {name} too long: {len(value)} chars, max 1000",
                    field=name,
                    value=len(value)
                )
        
        return value
    
    def validate_parameters_dict(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a dictionary of parameters.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            Dictionary of validated parameters
            
        Raises:
            ValidationError: If any parameter is invalid
        """
        validated = {}
        
        for name, value in parameters.items():
            # Validate parameter name
            if not isinstance(name, str) or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                raise ValidationError(
                    f"Invalid parameter name: {name}",
                    field="parameter_name",
                    value=name
                )
            
            # Validate parameter value
            validated[name] = self.validate_parameter(name, value)
        
        return validated


class SecurityAuditor:
    """Performs security audits and checks."""
    
    def __init__(self):
        """Initialize security auditor."""
        self.audit_log: List[Dict[str, Any]] = []
    
    def audit_operation(self, operation: str, user: str = "system",
                       parameters: Optional[Dict[str, Any]] = None,
                       result: str = "success") -> None:
        """Log an operation for security auditing.
        
        Args:
            operation: Operation performed
            user: User who performed operation
            parameters: Operation parameters (will be sanitized)
            result: Operation result (success/failure)
        """
        from datetime import datetime, timezone
        
        # Sanitize parameters for logging
        safe_params = {}
        if parameters:
            for key, value in parameters.items():
                if isinstance(value, str) and len(value) > 100:
                    safe_params[key] = f"{value[:100]}... (truncated)"
                elif key.lower() in ['password', 'secret', 'key', 'token']:
                    safe_params[key] = "[REDACTED]"
                else:
                    safe_params[key] = str(value)[:200]  # Limit value length
        
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "user": user,
            "parameters": safe_params,
            "result": result
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only last 1000 entries to prevent memory bloat
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def get_audit_log(self, operation: Optional[str] = None,
                     user: Optional[str] = None,
                     result: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered audit log entries.
        
        Args:
            operation: Filter by operation
            user: Filter by user
            result: Filter by result
            
        Returns:
            Filtered audit log entries
        """
        filtered_log = self.audit_log.copy()
        
        if operation:
            filtered_log = [entry for entry in filtered_log 
                          if entry["operation"] == operation]
        
        if user:
            filtered_log = [entry for entry in filtered_log 
                          if entry["user"] == user]
        
        if result:
            filtered_log = [entry for entry in filtered_log 
                          if entry["result"] == result]
        
        return filtered_log
    
    def check_security_violations(self) -> List[Dict[str, Any]]:
        """Check for potential security violations in audit log.
        
        Returns:
            List of potential security violations
        """
        violations = []
        
        # Check for too many failures from same user
        failure_counts = {}
        for entry in self.audit_log[-100:]:  # Check last 100 entries
            if entry["result"] == "failure":
                user = entry["user"]
                failure_counts[user] = failure_counts.get(user, 0) + 1
        
        for user, count in failure_counts.items():
            if count > 10:  # More than 10 failures
                violations.append({
                    "type": "excessive_failures",
                    "user": user,
                    "count": count,
                    "severity": "high"
                })
        
        # Check for unusual operations
        operation_counts = {}
        for entry in self.audit_log[-50:]:  # Check last 50 entries
            op = entry["operation"]
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        for operation, count in operation_counts.items():
            if count > 20:  # Same operation more than 20 times
                violations.append({
                    "type": "excessive_operations",
                    "operation": operation,
                    "count": count,
                    "severity": "medium"
                })
        
        return violations


# Global security instances
_input_sanitizer: Optional[InputSanitizer] = None
_resource_limiter: Optional[ResourceLimiter] = None
_parameter_validator: Optional[ParameterValidator] = None
_security_auditor: Optional[SecurityAuditor] = None


def initialize_security(config: Optional[SecurityConfig] = None) -> None:
    """Initialize global security components.
    
    Args:
        config: Security configuration
    """
    global _input_sanitizer, _resource_limiter, _parameter_validator, _security_auditor
    
    security_config = config or SecurityConfig()
    
    _input_sanitizer = InputSanitizer(security_config)
    _resource_limiter = ResourceLimiter(security_config)
    _parameter_validator = ParameterValidator()
    _security_auditor = SecurityAuditor()


def get_input_sanitizer() -> InputSanitizer:
    """Get global input sanitizer."""
    if _input_sanitizer is None:
        initialize_security()
    return _input_sanitizer


def get_resource_limiter() -> ResourceLimiter:
    """Get global resource limiter."""
    if _resource_limiter is None:
        initialize_security()
    return _resource_limiter


def get_parameter_validator() -> ParameterValidator:
    """Get global parameter validator."""
    if _parameter_validator is None:
        initialize_security()
    return _parameter_validator


def get_security_auditor() -> SecurityAuditor:
    """Get global security auditor."""
    if _security_auditor is None:
        initialize_security()
    return _security_auditor


def secure_operation(operation: str, user: str = "system"):
    """Decorator for secure operations with auditing.
    
    Args:
        operation: Operation name
        user: User performing operation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            auditor = get_security_auditor()
            
            try:
                # Audit operation start
                auditor.audit_operation(
                    operation=operation,
                    user=user,
                    parameters={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                    result="started"
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Audit success
                auditor.audit_operation(
                    operation=operation,
                    user=user,
                    result="success"
                )
                
                return result
                
            except Exception as e:
                # Audit failure
                auditor.audit_operation(
                    operation=operation,
                    user=user,
                    parameters={"error": str(e)[:200]},
                    result="failure"
                )
                raise
        
        return wrapper
    return decorator