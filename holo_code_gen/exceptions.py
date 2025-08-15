"""Custom exceptions for Holo-Code-Gen."""

from typing import List, Optional, Any, Dict


class HoloCodeGenException(Exception):
    """Base exception for all Holo-Code-Gen errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """Initialize exception with message, error code, and context.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class CompilationError(HoloCodeGenException):
    """Errors during neural network compilation."""
    pass


class GraphValidationError(HoloCodeGenException):
    """Errors in computation graph validation."""
    
    def __init__(self, message: str, validation_issues: List[str], 
                 error_code: str = "GRAPH_VALIDATION_ERROR"):
        super().__init__(message, error_code)
        self.validation_issues = validation_issues
        self.context["validation_issues"] = validation_issues


class ComponentError(HoloCodeGenException):
    """Errors related to photonic components."""
    pass


class TemplateLibraryError(HoloCodeGenException):
    """Errors in template library operations."""
    pass


class LayoutGenerationError(HoloCodeGenException):
    """Errors during physical layout generation."""
    pass


class OptimizationError(HoloCodeGenException):
    """Errors during circuit optimization."""
    pass


class SimulationError(HoloCodeGenException):
    """Errors during photonic simulation."""
    pass


class ConfigurationError(HoloCodeGenException):
    """Configuration and parameter errors."""
    pass


class ResourceLimitError(HoloCodeGenException):
    """Errors when resource limits are exceeded."""
    
    def __init__(self, message: str, resource_type: str, limit: float, 
                 current: float, error_code: str = "RESOURCE_LIMIT_ERROR"):
        super().__init__(message, error_code)
        self.resource_type = resource_type
        self.limit = limit
        self.current = current
        self.context.update({
            "resource_type": resource_type,
            "limit": limit,
            "current": current
        })


class ProcessVariationError(HoloCodeGenException):
    """Errors related to manufacturing process variations."""
    pass


class SecurityError(HoloCodeGenException):
    """Security-related errors."""
    pass


class ValidationError(HoloCodeGenException):
    """Input/output validation errors."""
    
    def __init__(self, message: str, field: str, value: Any = None,
                 error_code: str = "VALIDATION_ERROR"):
        super().__init__(message, error_code)
        self.field = field
        self.value = value
        self.context.update({
            "field": field,
            "value": str(value) if value is not None else None
        })


class TimeoutError(HoloCodeGenException):
    """Operation timeout errors."""
    
    def __init__(self, message: str, timeout_seconds: float,
                 error_code: str = "TIMEOUT_ERROR"):
        super().__init__(message, error_code)
        self.timeout_seconds = timeout_seconds
        self.context["timeout_seconds"] = timeout_seconds


class DependencyError(HoloCodeGenException):
    """Missing or incompatible dependency errors."""
    
    def __init__(self, message: str, dependency: str, required_version: Optional[str] = None,
                 current_version: Optional[str] = None, error_code: str = "DEPENDENCY_ERROR"):
        super().__init__(message, error_code)
        self.dependency = dependency
        self.required_version = required_version
        self.current_version = current_version
        self.context.update({
            "dependency": dependency,
            "required_version": required_version,
            "current_version": current_version
        })


class CompatibilityError(HoloCodeGenException):
    """Compatibility errors between components or versions."""
    pass


class ExportError(HoloCodeGenException):
    """Errors during file export operations."""
    pass


class ImportError(HoloCodeGenException):
    """Errors during file import operations."""
    pass


# Error code constants
class ErrorCodes:
    """Standard error codes for consistent error handling."""
    
    # Compilation errors
    NEURAL_NETWORK_PARSE_ERROR = "E001"
    UNSUPPORTED_LAYER_TYPE = "E002"
    GRAPH_EXTRACTION_ERROR = "E003"
    COMPONENT_MAPPING_ERROR = "E004"
    
    # Validation errors
    INVALID_INPUT_SHAPE = "E101"
    INVALID_PARAMETER_VALUE = "E102"
    MISSING_REQUIRED_PARAMETER = "E103"
    PARAMETER_OUT_OF_RANGE = "E104"
    
    # Graph errors
    CYCLIC_GRAPH = "E201"
    DISCONNECTED_GRAPH = "E202"
    INVALID_NODE_CONNECTION = "E203"
    
    # Component errors
    COMPONENT_NOT_FOUND = "E301"
    INVALID_COMPONENT_TYPE = "E302"
    COMPONENT_SPEC_INVALID = "E303"
    
    # Layout errors
    LAYOUT_GENERATION_FAILED = "E401"
    PLACEMENT_CONSTRAINT_VIOLATION = "E402"
    ROUTING_FAILURE = "E403"
    AREA_CONSTRAINT_VIOLATION = "E404"
    
    # Optimization errors
    OPTIMIZATION_CONVERGENCE_FAILURE = "E501"
    CONSTRAINT_VIOLATION = "E502"
    INFEASIBLE_OPTIMIZATION = "E503"
    
    # Resource errors
    POWER_BUDGET_EXCEEDED = "E601"
    AREA_BUDGET_EXCEEDED = "E602"
    MEMORY_LIMIT_EXCEEDED = "E603"
    
    # Simulation errors
    SIMULATION_CONVERGENCE_FAILURE = "E701"
    NUMERICAL_INSTABILITY = "E702"
    
    # System errors
    FILE_NOT_FOUND = "E801"
    PERMISSION_DENIED = "E802"
    DISK_SPACE_EXCEEDED = "E803"
    TIMEOUT = "E804"
    
    # Security errors
    UNAUTHORIZED_ACCESS = "E901"
    INVALID_CREDENTIALS = "E902"
    MALICIOUS_INPUT_DETECTED = "E903"
    
    # Quantum-specific errors
    QUANTUM_PLANNING_ERROR = "E1001"
    QUANTUM_GATE_NOT_SUPPORTED = "E1002"
    COHERENCE_TIME_EXCEEDED = "E1003"
    ENTANGLEMENT_GENERATION_FAILED = "E1004"
    MEASUREMENT_SCHEME_INVALID = "E1005"
    ALGORITHM_EXECUTION_ERROR = "E1006"
    QUANTUM_STATE_ERROR = "E1007"
    ENTANGLEMENT_ERROR = "E1008"
    MEASUREMENT_ERROR = "E1009"
    DECOHERENCE_ERROR = "E1010"
    ERROR_CORRECTION_FAILED = "E1011"
    SYNDROME_EXTRACTION_ERROR = "E1012"
    LOGICAL_ERROR_RATE_EXCEEDED = "E1013"
    THRESHOLD_EXCEEDED = "E1014"
    PHOTONIC_LOSS_EXCEEDED = "E1015"
    WAVELENGTH_MISMATCH = "E1016"
    THERMAL_INSTABILITY = "E1017"
    MODE_MISMATCH = "E1018"
    DETECTOR_SATURATION = "E1019"
    
    # General error types
    VALIDATION_ERROR = "E1100"
    INVALID_PARAMETER_TYPE = "E1101"
    INCONSISTENT_PARAMETERS = "E1102"
    RESOURCE_LIMIT_EXCEEDED = "E1103"
    DEPENDENCY_ERROR = "E1104"


def handle_exception(func):
    """Decorator for consistent exception handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HoloCodeGenException:
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise HoloCodeGenException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
            ) from e
    return wrapper


def validate_range(value: float, min_val: float, max_val: float, param_name: str) -> None:
    """Validate that a numeric value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value  
        param_name: Parameter name for error messages
        
    Raises:
        ValidationError: If value is out of range
    """
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"{param_name} must be between {min_val} and {max_val}, got {value}",
            field=param_name,
            value=value,
            error_code=ErrorCodes.PARAMETER_OUT_OF_RANGE
        )


def validate_positive(value: float, param_name: str) -> None:
    """Validate that a numeric value is positive.
    
    Args:
        value: Value to validate
        param_name: Parameter name for error messages
        
    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(
            f"{param_name} must be positive, got {value}",
            field=param_name,
            value=value,
            error_code=ErrorCodes.INVALID_PARAMETER_VALUE
        )


def validate_non_negative(value: float, param_name: str) -> None:
    """Validate that a numeric value is non-negative.
    
    Args:
        value: Value to validate
        param_name: Parameter name for error messages
        
    Raises:
        ValidationError: If value is negative
    """
    if value < 0:
        raise ValidationError(
            f"{param_name} must be non-negative, got {value}",
            field=param_name,
            value=value,
            error_code=ErrorCodes.INVALID_PARAMETER_VALUE
        )


def validate_not_empty(value: str, param_name: str) -> None:
    """Validate that a string value is not empty.
    
    Args:
        value: String to validate
        param_name: Parameter name for error messages
        
    Raises:
        ValidationError: If string is empty or None
    """
    if not value or not value.strip():
        raise ValidationError(
            f"{param_name} cannot be empty",
            field=param_name,
            value=value,
            error_code=ErrorCodes.MISSING_REQUIRED_PARAMETER
        )


def validate_list_not_empty(value: List[Any], param_name: str) -> None:
    """Validate that a list is not empty.
    
    Args:
        value: List to validate
        param_name: Parameter name for error messages
        
    Raises:
        ValidationError: If list is empty or None
    """
    if not value:
        raise ValidationError(
            f"{param_name} cannot be empty",
            field=param_name,
            value=value,
            error_code=ErrorCodes.MISSING_REQUIRED_PARAMETER
        )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
        
    Raises:
        SecurityError: If filename contains malicious patterns
    """
    import os
    import re
    
    # Remove directory traversal patterns
    if ".." in filename or "/" in filename or "\\" in filename:
        raise SecurityError(
            "Filename contains invalid characters",
            error_code=ErrorCodes.MALICIOUS_INPUT_DETECTED
        )
    
    # Remove or replace dangerous characters
    safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Limit length
    if len(safe_filename) > 255:
        safe_filename = safe_filename[:255]
    
    return safe_filename