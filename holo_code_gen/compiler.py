"""Core compiler functionality for neural network to photonic circuit translation."""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from .templates import IMECLibrary, PhotonicComponent
from .ir import ComputationGraph, CircuitNode
from .optimization import OptimizationTarget


logger = logging.getLogger(__name__)


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


class PhotonicCompiler:
    """Main compiler for neural networks to photonic circuits."""
    
    def __init__(self, config: Optional[CompilationConfig] = None):
        """Initialize photonic compiler.
        
        Args:
            config: Compilation configuration parameters
        """
        self.config = config or CompilationConfig()
        self.template_library = IMECLibrary(self.config.template_library)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging for compilation process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def compile(
        self, 
        model: nn.Module,
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> 'PhotonicCircuit':
        """Compile neural network to photonic circuit.
        
        Args:
            model: PyTorch neural network model
            input_shape: Expected input tensor shape
            **kwargs: Additional compilation parameters
            
        Returns:
            Compiled photonic circuit
        """
        logger.info(f"Starting compilation of {type(model).__name__}")
        
        # Extract computation graph
        computation_graph = self._extract_graph(model, input_shape)
        logger.info(f"Extracted graph with {len(computation_graph.nodes)} nodes")
        
        # Map to photonic components  
        circuit_nodes = self._map_to_photonic(computation_graph)
        logger.info(f"Mapped to {len(circuit_nodes)} photonic components")
        
        # Generate physical layout
        photonic_circuit = self._generate_circuit(circuit_nodes)
        logger.info("Generated photonic circuit layout")
        
        # Apply optimization
        optimized_circuit = self._optimize_circuit(photonic_circuit)
        logger.info("Applied circuit optimizations")
        
        return optimized_circuit
        
    def _extract_graph(self, model: nn.Module, input_shape: Optional[tuple]) -> ComputationGraph:
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
        
    def _get_node_type(self, module: nn.Module) -> str:
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
            
    def _extract_parameters(self, module: nn.Module) -> Dict[str, Any]:
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
        
    def _get_input_shape(self, module: nn.Module) -> Optional[tuple]:
        """Get input shape for module."""
        if hasattr(module, 'in_features'):
            return (module.in_features,)
        elif hasattr(module, 'in_channels'):
            return (module.in_channels,)
        return None
        
    def _get_output_shape(self, module: nn.Module) -> Optional[tuple]:
        """Get output shape for module.""" 
        if hasattr(module, 'out_features'):
            return (module.out_features,)
        elif hasattr(module, 'out_channels'):
            return (module.out_channels,)
        return None
        
    def _add_connections(self, graph: ComputationGraph, model: nn.Module) -> None:
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
        model: nn.Module,
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
        
    def _extract_spiking_graph(self, model: nn.Module, input_shape: Optional[tuple]) -> ComputationGraph:
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