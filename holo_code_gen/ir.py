"""Intermediate representation for photonic computation graphs."""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import networkx as nx
from enum import Enum


class NodeType(Enum):
    """Types of computation nodes in the IR."""
    MATRIX_MULTIPLY = "matrix_multiply"
    CONVOLUTION = "convolution"
    ACTIVATION = "activation"
    POOLING = "pooling"
    NORMALIZATION = "normalization"
    NONLINEARITY = "nonlinearity"
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class CircuitNode:
    """Represents a single computation node in the photonic circuit."""
    name: str
    node_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    photonic_component: Optional['PhotonicComponent'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node after initialization."""
        if not self.name:
            raise ValueError("Node name cannot be empty")
        if not self.node_type:
            raise ValueError("Node type cannot be empty")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set parameter value."""
        self.parameters[key] = value
    
    def get_complexity(self) -> float:
        """Estimate computational complexity of this node."""
        if self.node_type == "matrix_multiply":
            input_size = self.get_parameter('input_size', 1)
            output_size = self.get_parameter('output_size', 1)
            return float(input_size * output_size)
        elif self.node_type == "convolution":
            kernel_size = self.get_parameter('kernel_size', (3, 3))
            if isinstance(kernel_size, int):
                kernel_ops = kernel_size * kernel_size
            else:
                kernel_ops = kernel_size[0] * kernel_size[1]
            channels = self.get_parameter('channels', 1)
            return float(kernel_ops * channels)
        else:
            # Default complexity for other operations
            return 1.0


@dataclass
class CircuitEdge:
    """Represents a connection between computation nodes."""
    source: str
    target: str
    data_type: str = "optical"
    wavelength: Optional[float] = None
    bandwidth: Optional[float] = None
    loss: float = 0.0
    delay: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate edge after initialization."""
        if not self.source or not self.target:
            raise ValueError("Source and target nodes must be specified")
        if self.loss < 0:
            raise ValueError("Loss cannot be negative")
        if self.delay < 0:
            raise ValueError("Delay cannot be negative")


class ComputationGraph:
    """Hardware-agnostic representation of computation graph."""
    
    def __init__(self):
        """Initialize empty computation graph."""
        self.nodes: Dict[str, CircuitNode] = {}
        self.edges: Dict[tuple, CircuitEdge] = {}
        self._graph = nx.DiGraph()
        self.metadata: Dict[str, Any] = {}
    
    def add_node(self, node: CircuitNode) -> None:
        """Add computation node to graph."""
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name} already exists")
        
        self.nodes[node.name] = node
        self._graph.add_node(node.name, **node.parameters)
    
    def add_edge(self, source: str, target: str, edge: Optional[CircuitEdge] = None) -> None:
        """Add edge between nodes."""
        if source not in self.nodes:
            raise ValueError(f"Source node {source} does not exist")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} does not exist")
        
        if edge is None:
            edge = CircuitEdge(source=source, target=target)
        
        edge_key = (source, target)
        self.edges[edge_key] = edge
        self._graph.add_edge(source, target, **edge.metadata)
    
    def remove_node(self, name: str) -> None:
        """Remove node and all connected edges."""
        if name not in self.nodes:
            raise ValueError(f"Node {name} does not exist")
        
        # Remove all edges connected to this node
        edges_to_remove = [
            key for key in self.edges.keys() 
            if key[0] == name or key[1] == name
        ]
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
        
        # Remove node
        del self.nodes[name]
        self._graph.remove_node(name)
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove edge between nodes."""
        edge_key = (source, target)
        if edge_key not in self.edges:
            raise ValueError(f"Edge {source} -> {target} does not exist")
        
        del self.edges[edge_key]
        self._graph.remove_edge(source, target)
    
    def get_node(self, name: str) -> CircuitNode:
        """Get node by name."""
        if name not in self.nodes:
            raise ValueError(f"Node {name} does not exist")
        return self.nodes[name]
    
    def get_edge(self, source: str, target: str) -> CircuitEdge:
        """Get edge by source and target."""
        edge_key = (source, target)
        if edge_key not in self.edges:
            raise ValueError(f"Edge {source} -> {target} does not exist")
        return self.edges[edge_key]
    
    def get_predecessors(self, node_name: str) -> List[str]:
        """Get predecessor nodes."""
        return list(self._graph.predecessors(node_name))
    
    def get_successors(self, node_name: str) -> List[str]:
        """Get successor nodes."""
        return list(self._graph.successors(node_name))
    
    def topological_sort(self) -> List[str]:
        """Get topologically sorted node names."""
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXError as e:
            raise ValueError(f"Graph contains cycles: {e}")
    
    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph."""
        try:
            return list(nx.simple_cycles(self._graph))
        except nx.NetworkXError:
            return []
    
    def get_paths(self, source: str, target: str) -> List[List[str]]:
        """Get all simple paths between source and target."""
        try:
            return list(nx.all_simple_paths(self._graph, source, target))
        except nx.NetworkXError:
            return []
    
    def get_critical_path(self) -> List[str]:
        """Find critical path through the computation graph."""
        # For now, return longest path by number of nodes
        try:
            return list(nx.dag_longest_path(self._graph))
        except nx.NetworkXError:
            # If there are cycles, return topological sort
            return self.topological_sort()
    
    def estimate_latency(self) -> float:
        """Estimate total computation latency."""
        critical_path = self.get_critical_path()
        total_latency = 0.0
        
        for i, node_name in enumerate(critical_path):
            node = self.nodes[node_name]
            
            # Add node processing time (simplified)
            complexity = node.get_complexity()
            processing_time = complexity * 1e-9  # Assume 1ns per operation
            total_latency += processing_time
            
            # Add edge delay if not the last node
            if i < len(critical_path) - 1:
                next_node = critical_path[i + 1]
                edge_key = (node_name, next_node)
                if edge_key in self.edges:
                    total_latency += self.edges[edge_key].delay
        
        return total_latency
    
    def estimate_power(self) -> float:
        """Estimate total power consumption."""
        total_power = 0.0
        
        for node in self.nodes.values():
            # Simple power model based on complexity
            complexity = node.get_complexity()
            node_power = complexity * 1e-3  # Assume 1mW per operation
            total_power += node_power
        
        return total_power
    
    def estimate_area(self) -> float:
        """Estimate total chip area."""
        total_area = 0.0
        
        for node in self.nodes.values():
            # Simple area model based on complexity
            complexity = node.get_complexity()
            node_area = complexity * 1e-6  # Assume 1μm² per operation
            total_area += node_area
        
        return total_area
    
    def validate(self) -> List[str]:
        """Validate graph structure and return list of issues."""
        issues = []
        
        # Check for cycles
        cycles = self.find_cycles()
        if cycles:
            issues.append(f"Graph contains {len(cycles)} cycle(s)")
        
        # Check for disconnected components
        if not nx.is_weakly_connected(self._graph):
            issues.append("Graph has disconnected components")
        
        # Check for nodes without inputs (except input nodes)
        for node_name, node in self.nodes.items():
            if node.node_type != "input":
                predecessors = self.get_predecessors(node_name)
                if not predecessors:
                    issues.append(f"Node {node_name} has no inputs")
        
        # Check for nodes without outputs (except output nodes)
        for node_name, node in self.nodes.items():
            if node.node_type != "output":
                successors = self.get_successors(node_name)
                if not successors:
                    issues.append(f"Node {node_name} has no outputs")
        
        # Check parameter consistency
        for node in self.nodes.values():
            if node.input_shape and node.output_shape:
                if node.node_type == "matrix_multiply":
                    input_size = node.get_parameter('input_size')
                    if input_size and len(node.input_shape) > 0:
                        if input_size != node.input_shape[0]:
                            issues.append(f"Node {node.name} input size mismatch")
        
        return issues
    
    def optimize_structure(self) -> None:
        """Apply structural optimizations to the graph."""
        # Remove redundant nodes
        self._remove_identity_nodes()
        
        # Merge compatible sequential operations
        self._merge_sequential_ops()
        
        # Optimize data flow
        self._optimize_data_flow()
    
    def _remove_identity_nodes(self) -> None:
        """Remove nodes that don't perform useful computation."""
        nodes_to_remove = []
        
        for node_name, node in self.nodes.items():
            # Check if node is identity operation
            if (node.node_type == "activation" and 
                node.get_parameter('activation_type') == 'linear'):
                nodes_to_remove.append(node_name)
        
        for node_name in nodes_to_remove:
            self._bypass_node(node_name)
    
    def _bypass_node(self, node_name: str) -> None:
        """Bypass a node by connecting its inputs directly to outputs."""
        predecessors = self.get_predecessors(node_name)
        successors = self.get_successors(node_name)
        
        # Connect all predecessors to all successors
        for pred in predecessors:
            for succ in successors:
                if (pred, succ) not in self.edges:
                    self.add_edge(pred, succ)
        
        # Remove the bypassed node
        self.remove_node(node_name)
    
    def _merge_sequential_ops(self) -> None:
        """Merge compatible sequential operations."""
        # Find chains of matrix multiplications that can be merged
        for node_name in list(self.nodes.keys()):
            if node_name not in self.nodes:
                continue  # Node was already merged
                
            node = self.nodes[node_name]
            if node.node_type == "matrix_multiply":
                successors = self.get_successors(node_name)
                if len(successors) == 1:
                    next_node_name = successors[0]
                    next_node = self.nodes[next_node_name]
                    if (next_node.node_type == "matrix_multiply" and
                        len(self.get_predecessors(next_node_name)) == 1):
                        self._merge_matrix_multiplies(node_name, next_node_name)
    
    def _merge_matrix_multiplies(self, node1: str, node2: str) -> None:
        """Merge two sequential matrix multiplication nodes."""
        # Get weight matrices
        w1 = self.nodes[node1].get_parameter('weight_matrix')
        w2 = self.nodes[node2].get_parameter('weight_matrix')
        
        if w1 is not None and w2 is not None:
            # Compute combined weight matrix
            import numpy as np
            combined_weights = np.dot(w2, w1)
            
            # Update first node with combined weights
            self.nodes[node1].set_parameter('weight_matrix', combined_weights)
            self.nodes[node1].set_parameter('output_size', 
                                          self.nodes[node2].get_parameter('output_size'))
            
            # Redirect edges from second node to first node
            successors = self.get_successors(node2)
            for succ in successors:
                edge = self.get_edge(node2, succ)
                self.add_edge(node1, succ, edge)
            
            # Remove second node
            self.remove_node(node2)
    
    def _optimize_data_flow(self) -> None:
        """Optimize data flow patterns in the graph."""
        # Identify and optimize common patterns
        self._optimize_activation_placement()
        self._optimize_branch_merging()
    
    def _optimize_activation_placement(self) -> None:
        """Optimize placement of activation functions."""
        # Move activations closer to their most beneficial position
        pass  # Implementation would depend on specific optimization strategies
    
    def _optimize_branch_merging(self) -> None:
        """Optimize merging of parallel branches."""
        # Identify opportunities to merge parallel computation paths
        pass  # Implementation would depend on specific optimization strategies
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            'nodes': {name: {
                'name': node.name,
                'node_type': node.node_type,
                'parameters': node.parameters,
                'input_shape': node.input_shape,
                'output_shape': node.output_shape,
                'metadata': node.metadata
            } for name, node in self.nodes.items()},
            'edges': {f"{edge.source}->{edge.target}": {
                'source': edge.source,
                'target': edge.target,
                'data_type': edge.data_type,
                'wavelength': edge.wavelength,
                'bandwidth': edge.bandwidth,
                'loss': edge.loss,
                'delay': edge.delay,
                'metadata': edge.metadata
            } for edge in self.edges.values()},
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputationGraph':
        """Create graph from dictionary representation."""
        graph = cls()
        graph.metadata = data.get('metadata', {})
        
        # Add nodes
        for node_data in data.get('nodes', {}).values():
            node = CircuitNode(
                name=node_data['name'],
                node_type=node_data['node_type'],
                parameters=node_data.get('parameters', {}),
                input_shape=tuple(node_data['input_shape']) if node_data.get('input_shape') else None,
                output_shape=tuple(node_data['output_shape']) if node_data.get('output_shape') else None,
                metadata=node_data.get('metadata', {})
            )
            graph.add_node(node)
        
        # Add edges
        for edge_data in data.get('edges', {}).values():
            edge = CircuitEdge(
                source=edge_data['source'],
                target=edge_data['target'],
                data_type=edge_data.get('data_type', 'optical'),
                wavelength=edge_data.get('wavelength'),
                bandwidth=edge_data.get('bandwidth'),
                loss=edge_data.get('loss', 0.0),
                delay=edge_data.get('delay', 0.0),
                metadata=edge_data.get('metadata', {})
            )
            graph.add_edge(edge.source, edge.target, edge)
        
        return graph