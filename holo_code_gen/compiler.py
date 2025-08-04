"""Core compiler functionality for neural network to photonic circuit translation."""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from .templates import IMECLibrary, PhotonicComponent
from .ir import ComputationGraph, CircuitNode
from .optimization import OptimizationTarget
from .exceptions import (
    CompilationError, GraphValidationError, ComponentError,
    ValidationError, ErrorCodes, validate_positive, validate_not_empty
)
from .monitoring import monitor_function, log_exceptions, get_logger, get_performance_monitor
from .security import get_parameter_validator, get_resource_limiter, secure_operation

# Optional PyTorch import for compatibility
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock classes for type hints
    class torch:
        class nn:
            class Module:
                pass
            Linear = Module
            Conv2d = Module
            ReLU = Module
            Sigmoid = Module


logger = get_logger()


@dataclass
class CompilationConfig:
    """Configuration for photonic compilation."""
    template_library: str = "imec_v2025_07"
    process: str = "SiN_220nm" 
    wavelength: float = 1550.0  # nm
    input_encoding: str = "amplitude"
    output_detection: str = "coherent"
    optimization_target: str = "power"
    max_optical_path: float = 10.0  # mm
    power_budget: float = 1000.0  # mW
    area_budget: float = 100.0  # mm²
    
    def __post_init__(self):
        """Validate configuration parameters."""
        validate_positive(self.wavelength, "wavelength")
        validate_positive(self.max_optical_path, "max_optical_path")
        validate_positive(self.power_budget, "power_budget")
        validate_positive(self.area_budget, "area_budget")
        validate_not_empty(self.template_library, "template_library")
        validate_not_empty(self.process, "process")
        
        # Validate enum-like parameters
        valid_encodings = ["amplitude", "phase", "coherent", "incoherent"]
        if self.input_encoding not in valid_encodings:
            raise ValidationError(
                f"Invalid input_encoding: {self.input_encoding}. Must be one of {valid_encodings}",
                field="input_encoding",
                value=self.input_encoding
            )
        
        valid_detections = ["coherent", "incoherent", "single_photon"]
        if self.output_detection not in valid_detections:
            raise ValidationError(
                f"Invalid output_detection: {self.output_detection}. Must be one of {valid_detections}",
                field="output_detection",
                value=self.output_detection
            )


class PhotonicCompiler:
    """Main compiler for neural networks to photonic circuits."""
    
    @log_exceptions("compiler")
    def __init__(self, config: Optional[CompilationConfig] = None):
        """Initialize photonic compiler.
        
        Args:
            config: Compilation configuration parameters
        """
        self.config = config or CompilationConfig()
        
        try:
            self.template_library = IMECLibrary(self.config.template_library)
        except Exception as e:
            raise CompilationError(
                f"Failed to initialize template library: {str(e)}",
                error_code=ErrorCodes.COMPONENT_NOT_FOUND,
                context={"template_library": self.config.template_library}
            )
        
        # Initialize security components
        self.parameter_validator = get_parameter_validator()
        self.resource_limiter = get_resource_limiter()
        
        logger.info(
            "PhotonicCompiler initialized",
            component="compiler",
            context={"template_library": self.config.template_library}
        )
        
    @monitor_function("compile", "compiler")
    @secure_operation("neural_network_compilation")
    @log_exceptions("compiler")
    def compile(
        self, 
        model: Union['torch.torch.nn.Module', Dict[str, Any]],
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> 'PhotonicCircuit':
        """Compile neural network to photonic circuit.
        
        Args:
            model: PyTorch neural network model or model specification dict
            input_shape: Expected input tensor shape
            **kwargs: Additional compilation parameters
            
        Returns:
            Compiled photonic circuit
            
        Raises:
            CompilationError: If compilation fails
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if model is None:
            raise ValidationError(
                "Model cannot be None",
                field="model",
                error_code=ErrorCodes.MISSING_REQUIRED_PARAMETER
            )
        
        # Support both PyTorch models and dict specifications
        if isinstance(model, dict):
            if not TORCH_AVAILABLE:
                # Use dict-based compilation when PyTorch not available
                return self._compile_from_dict(model, input_shape, **kwargs)
            else:
                raise CompilationError(
                    "Dict-based compilation not yet implemented when PyTorch is available",
                    error_code=ErrorCodes.UNSUPPORTED_LAYER_TYPE
                )
        
        if not TORCH_AVAILABLE:
            raise CompilationError(
                "PyTorch not available and model is not a dict specification",
                error_code=ErrorCodes.DEPENDENCY_ERROR
            )
        
        model_name = type(model).__name__
        logger.info(
            f"Starting compilation of {model_name}",
            component="compiler",
            operation="compile",
            context={"model_type": model_name, "input_shape": input_shape}
        )
        
        try:
            # Extract computation graph
            computation_graph = self._extract_graph(model, input_shape)
            logger.info(
                f"Extracted graph with {len(computation_graph.nodes)} nodes",
                component="compiler"
            )
            
            # Validate graph complexity
            self.resource_limiter.check_graph_complexity(len(computation_graph.nodes))
            
            # Map to photonic components  
            circuit_nodes = self._map_to_photonic(computation_graph)
            logger.info(
                f"Mapped to {len(circuit_nodes)} photonic components",
                component="compiler"
            )
            
            # Generate physical layout
            photonic_circuit = self._generate_circuit(circuit_nodes)
            logger.info("Generated photonic circuit layout", component="compiler")
            
            # Apply optimization
            optimized_circuit = self._optimize_circuit(photonic_circuit)
            logger.info("Applied circuit optimizations", component="compiler")
            
            return optimized_circuit
            
        except Exception as e:
            logger.error(
                f"Compilation failed: {str(e)}",
                component="compiler",
                operation="compile",
                error=str(e)
            )
            
            if isinstance(e, (CompilationError, ValidationError)):
                raise
            else:
                raise CompilationError(
                    f"Unexpected compilation error: {str(e)}",
                    error_code=ErrorCodes.GRAPH_EXTRACTION_ERROR,
                    context={"model_type": model_name}
                ) from e
    
    def _compile_from_dict(self, model_spec: Dict[str, Any], input_shape: Optional[tuple] = None,
                          **kwargs) -> 'PhotonicCircuit':
        """Compile neural network from dictionary specification.
        
        Args:
            model_spec: Model specification dictionary
            input_shape: Expected input tensor shape
            **kwargs: Additional compilation parameters
            
        Returns:
            Compiled photonic circuit
        """
        logger.info(
            "Starting dict-based compilation",
            component="compiler",
            context={"layers": len(model_spec.get("layers", []))}
        )
        
        # Create computation graph from specification
        graph = ComputationGraph()
        
        layers = model_spec.get("layers", [])
        if not layers:
            raise ValidationError(
                "Model specification must contain layers",
                field="layers",
                error_code=ErrorCodes.MISSING_REQUIRED_PARAMETER
            )
        
        # Add nodes for each layer
        for i, layer_spec in enumerate(layers):
            layer_type = layer_spec.get("type")
            if not layer_type:
                raise ValidationError(
                    f"Layer {i} missing type",
                    field=f"layers[{i}].type",
                    error_code=ErrorCodes.MISSING_REQUIRED_PARAMETER
                )
            
            # Validate parameters
            parameters = layer_spec.get("parameters", {})
            validated_params = self.parameter_validator.validate_parameters_dict(parameters)
            
            node = CircuitNode(
                name=layer_spec.get("name", f"layer_{i}"),
                node_type=layer_type,
                parameters=validated_params,
                input_shape=layer_spec.get("input_shape"),
                output_shape=layer_spec.get("output_shape")
            )
            graph.add_node(node)
        
        # Add connections
        node_list = list(graph.nodes.values())
        for i in range(len(node_list) - 1):
            graph.add_edge(node_list[i].name, node_list[i + 1].name)
        
        # Continue with standard compilation pipeline
        circuit_nodes = self._map_to_photonic(graph)
        photonic_circuit = self._generate_circuit(circuit_nodes)
        optimized_circuit = self._optimize_circuit(photonic_circuit)
        
        return optimized_circuit
        
    def _extract_graph(self, model: 'torch.torch.nn.Module', input_shape: Optional[tuple]) -> ComputationGraph:
        """Extract computation graph from neural network."""
        graph = ComputationGraph()
        
        # Trace model execution
        if input_shape:
            dummy_input = torch.randn(*input_shape)
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
                
        # Extract layers and connections
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid)):
                node_type = self._get_node_type(module)
                node = CircuitNode(
                    name=name,
                    node_type=node_type,
                    parameters=self._extract_parameters(module),
                    input_shape=self._get_input_shape(module),
                    output_shape=self._get_output_shape(module)
                )
                graph.add_node(node)
                
        # Add connections between nodes
        self._add_connections(graph, model)
        
        return graph
        
    def _get_node_type(self, module: torch.nn.Module) -> str:
        """Determine photonic node type for module."""
        if isinstance(module, nn.Linear):
            return "matrix_vector_multiply"
        elif isinstance(module, nn.Conv2d):
            return "convolution"
        elif isinstance(module, nn.ReLU):
            return "optical_nonlinearity"
        elif isinstance(module, nn.Sigmoid):
            return "sigmoid_activation"
        else:
            return "unknown"
            
    def _extract_parameters(self, module: torch.nn.Module) -> Dict[str, Any]:
        """Extract relevant parameters from module."""
        params = {}
        if hasattr(module, 'weight'):
            params['weight_matrix'] = module.weight.detach().numpy()
        if hasattr(module, 'bias') and module.bias is not None:
            params['bias_vector'] = module.bias.detach().numpy()
        if hasattr(module, 'in_features'):
            params['input_size'] = module.in_features
        if hasattr(module, 'out_features'):
            params['output_size'] = module.out_features
        return params
        
    def _get_input_shape(self, module: torch.nn.Module) -> Optional[tuple]:
        """Get input shape for module."""
        if hasattr(module, 'in_features'):
            return (module.in_features,)
        elif hasattr(module, 'in_channels'):
            return (module.in_channels,)
        return None
        
    def _get_output_shape(self, module: torch.nn.Module) -> Optional[tuple]:
        """Get output shape for module.""" 
        if hasattr(module, 'out_features'):
            return (module.out_features,)
        elif hasattr(module, 'out_channels'):
            return (module.out_channels,)
        return None
        
    def _add_connections(self, graph: ComputationGraph, model: torch.nn.Module) -> None:
        """Add connections between nodes in graph."""
        # Simple sequential connection for now
        nodes = list(graph.nodes.values())
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i].name, nodes[i+1].name)
            
    def _map_to_photonic(self, graph: ComputationGraph) -> List[CircuitNode]:
        """Map computation graph to photonic components."""
        circuit_nodes = []
        
        for node in graph.nodes.values():
            photonic_component = self._select_photonic_component(node)
            
            circuit_node = CircuitNode(
                name=f"photonic_{node.name}",
                node_type=photonic_component.component_type,
                parameters=self._adapt_parameters(node.parameters, photonic_component),
                photonic_component=photonic_component
            )
            circuit_nodes.append(circuit_node)
            
        return circuit_nodes
        
    def _select_photonic_component(self, node: CircuitNode) -> PhotonicComponent:
        """Select appropriate photonic component for computation node."""
        if node.node_type == "matrix_vector_multiply":
            return self.template_library.get_component("microring_weight_bank")
        elif node.node_type == "optical_nonlinearity":
            return self.template_library.get_component("ring_modulator")
        elif node.node_type == "convolution":
            return self.template_library.get_component("mzi_mesh")
        else:
            # Default to basic waveguide
            return self.template_library.get_component("waveguide")
            
    def _adapt_parameters(self, neural_params: Dict[str, Any], component: PhotonicComponent) -> Dict[str, Any]:
        """Adapt neural network parameters to photonic component parameters."""
        adapted = {}
        
        if 'weight_matrix' in neural_params:
            weights = neural_params['weight_matrix']
            # Convert to optical phases/amplitudes
            adapted['phase_matrix'] = np.angle(weights + 1j * np.abs(weights))
            adapted['amplitude_matrix'] = np.abs(weights)
            
        if 'input_size' in neural_params:
            adapted['num_inputs'] = neural_params['input_size']
            
        if 'output_size' in neural_params:
            adapted['num_outputs'] = neural_params['output_size']
            
        return adapted
        
    def _generate_circuit(self, circuit_nodes: List[CircuitNode]) -> 'PhotonicCircuit':
        """Generate physical photonic circuit layout."""
        from .circuit import PhotonicCircuit
        
        circuit = PhotonicCircuit(
            config=self.config,
            template_library=self.template_library
        )
        
        # Add components to circuit
        for node in circuit_nodes:
            circuit.add_component(node)
            
        # Generate physical layout
        circuit.generate_layout()
        
        return circuit
        
    def _optimize_circuit(self, circuit: 'PhotonicCircuit') -> 'PhotonicCircuit':
        """Apply optimization to photonic circuit.""" 
        from .optimization import PowerOptimizer
        
        if self.config.optimization_target == "power":
            optimizer = PowerOptimizer(
                power_budget=self.config.power_budget,
                circuit=circuit
            )
            return optimizer.optimize()
        else:
            # Return unoptimized circuit for now
            return circuit


class SpikingPhotonicCompiler(PhotonicCompiler):
    """Specialized compiler for spiking neural networks."""
    
    def __init__(self, config: Optional[CompilationConfig] = None):
        """Initialize spiking photonic compiler."""
        super().__init__(config)
        self.spike_encoding = getattr(config, 'spike_encoding', 'phase')
        self.spike_threshold = getattr(config, 'spike_threshold', 1.0)
        self.refractory_period = getattr(config, 'refractory_period', 2.0)
        
    def compile(
        self, 
        model: torch.nn.Module,
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> 'PhotonicCircuit':
        """Compile spiking neural network to photonic circuit."""
        logger.info(f"Starting spiking compilation of {type(model).__name__}")
        
        # Use spike-specific processing
        computation_graph = self._extract_spiking_graph(model, input_shape)
        circuit_nodes = self._map_to_spiking_photonic(computation_graph)
        photonic_circuit = self._generate_spiking_circuit(circuit_nodes)
        optimized_circuit = self._optimize_spiking_circuit(photonic_circuit)
        
        return optimized_circuit
        
    def _extract_spiking_graph(self, model: torch.nn.Module, input_shape: Optional[tuple]) -> ComputationGraph:
        """Extract computation graph with spiking semantics."""
        # For now, use base implementation with spiking annotations
        graph = self._extract_graph(model, input_shape)
        
        # Add spiking-specific metadata
        for node in graph.nodes.values():
            node.parameters['spike_threshold'] = self.spike_threshold
            node.parameters['refractory_period'] = self.refractory_period
            node.parameters['encoding'] = self.spike_encoding
            
        return graph
        
    def _map_to_spiking_photonic(self, graph: ComputationGraph) -> List[CircuitNode]:
        """Map spiking computation to photonic components."""
        circuit_nodes = []
        
        for node in graph.nodes.values():
            # Use spiking-specific components
            if node.node_type == "matrix_vector_multiply":
                component = self.template_library.get_component("photonic_lif_neuron")
            else:
                component = self.template_library.get_component("spike_detector")
                
            circuit_node = CircuitNode(
                name=f"spiking_{node.name}",
                node_type=f"spiking_{node.node_type}",
                parameters=self._adapt_spiking_parameters(node.parameters, component),
                photonic_component=component
            )
            circuit_nodes.append(circuit_node)
            
        return circuit_nodes
        
    def _adapt_spiking_parameters(self, neural_params: Dict[str, Any], component: PhotonicComponent) -> Dict[str, Any]:
        """Adapt spiking parameters to photonic implementation."""
        adapted = self._adapt_parameters(neural_params, component)
        
        # Add spiking-specific adaptations
        if 'spike_threshold' in neural_params:
            adapted['optical_threshold'] = neural_params['spike_threshold']
            
        if 'refractory_period' in neural_params:
            # Convert time to optical delay
            adapted['delay_length'] = neural_params['refractory_period'] * 0.3  # mm (c * time)
            
        return adapted
        
    def _generate_spiking_circuit(self, circuit_nodes: List[CircuitNode]) -> 'PhotonicCircuit':
        """Generate spiking photonic circuit with temporal dynamics."""
        circuit = self._generate_circuit(circuit_nodes)
        
        # Add temporal processing elements
        circuit.add_temporal_processing()
        
        return circuit
        
    def _optimize_spiking_circuit(self, circuit: 'PhotonicCircuit') -> 'PhotonicCircuit':
        """Optimize spiking circuit for temporal performance."""
        # Apply base optimization
        circuit = self._optimize_circuit(circuit)
        
        # Add spiking-specific optimizations
        circuit.optimize_temporal_dynamics()
        
        return circuit