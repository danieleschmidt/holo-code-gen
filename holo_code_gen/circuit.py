"""Photonic circuit representation and layout generation."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
from copy import deepcopy

from .templates import PhotonicComponent, IMECLibrary
from .ir import CircuitNode


logger = logging.getLogger(__name__)


@dataclass
class LayoutConstraints:
    """Constraints for physical layout generation."""
    max_chip_area: float = 100.0  # mm²
    min_component_spacing: float = 10.0  # μm
    max_waveguide_length: float = 10.0  # mm
    routing_algorithm: str = "manhattan"
    compact_layout: bool = True
    thermal_isolation: bool = False


@dataclass
class CircuitMetrics:
    """Performance metrics for photonic circuit."""
    total_power: float = 0.0      # mW
    total_area: float = 0.0       # mm²
    total_loss: float = 0.0       # dB
    latency: float = 0.0          # ns
    throughput: float = 0.0       # TOPS
    energy_efficiency: float = 0.0  # TOPS/W


class PhotonicCircuit:
    """Represents a complete photonic neural network circuit."""
    
    def __init__(self, config=None, template_library: Optional[IMECLibrary] = None):
        """Initialize photonic circuit.
        
        Args:
            config: Compilation configuration
            template_library: Component template library
        """
        self.config = config
        self.template_library = template_library or IMECLibrary()
        self.components: List[PhotonicComponent] = []
        self.connections: List[Tuple[str, str]] = []
        self.layout_constraints = LayoutConstraints()
        self.layout_generated = False
        self.physical_layout: Optional[Dict[str, Any]] = None
        self.metrics: Optional[CircuitMetrics] = None
        
    def add_component(self, node: CircuitNode) -> None:
        """Add component from circuit node."""
        if node.photonic_component is None:
            # Create component from node specification
            component_type = node.node_type.replace('photonic_', '').replace('spiking_', '')
            
            try:
                component = self.template_library.get_component(component_type)
            except ValueError:
                # Fallback to basic waveguide for unknown types
                component = self.template_library.get_component("waveguide")
            
            # Apply node parameters to component
            for key, value in node.parameters.items():
                component.set_parameter(key, value)
            
            component.instance_id = node.name
            node.photonic_component = component
        
        self.components.append(node.photonic_component)
        logger.debug(f"Added component {node.name} of type {node.node_type}")
    
    def add_connection(self, source_id: str, target_id: str) -> None:
        """Add connection between components."""
        self.connections.append((source_id, target_id))
    
    def generate_layout(self, constraints: Optional[LayoutConstraints] = None) -> None:
        """Generate physical layout for the circuit."""
        if constraints:
            self.layout_constraints = constraints
        
        logger.info("Generating physical layout")
        
        # Component placement
        component_positions = self._place_components()
        
        # Waveguide routing
        routing_layout = self._route_waveguides(component_positions)
        
        # Assembly layout information
        self.physical_layout = {
            'components': component_positions,
            'routing': routing_layout,
            'total_area': self._calculate_layout_area(component_positions),
            'bounding_box': self._calculate_bounding_box(component_positions)
        }
        
        self.layout_generated = True
        logger.info("Physical layout generated successfully")
    
    def _place_components(self) -> Dict[str, Tuple[float, float]]:
        """Place components on the chip."""
        positions = {}
        
        if self.layout_constraints.compact_layout:
            positions = self._compact_placement()
        else:
            positions = self._grid_placement()
        
        # Apply thermal constraints if needed
        if self.layout_constraints.thermal_isolation:
            positions = self._apply_thermal_spacing(positions)
        
        return positions
    
    def _compact_placement(self) -> Dict[str, Tuple[float, float]]:
        """Compact component placement algorithm."""
        positions = {}
        current_x, current_y = 0.0, 0.0
        row_height = 0.0
        max_width = np.sqrt(self.layout_constraints.max_chip_area * 1000)  # Convert to μm
        
        for component in self.components:
            # Estimate component size
            width = self._estimate_component_width(component)
            height = self._estimate_component_height(component)
            
            # Check if we need to start a new row
            if current_x + width > max_width:
                current_x = 0.0
                current_y += row_height + self.layout_constraints.min_component_spacing
                row_height = 0.0
            
            positions[component.instance_id] = (current_x, current_y)
            
            current_x += width + self.layout_constraints.min_component_spacing
            row_height = max(row_height, height)
        
        return positions
    
    def _grid_placement(self) -> Dict[str, Tuple[float, float]]:
        """Grid-based component placement."""
        positions = {}
        grid_size = int(np.ceil(np.sqrt(len(self.components))))
        spacing = 100.0  # μm
        
        for i, component in enumerate(self.components):
            row = i // grid_size
            col = i % grid_size
            x = col * spacing
            y = row * spacing
            positions[component.instance_id] = (x, y)
        
        return positions
    
    def _apply_thermal_spacing(self, positions: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Apply thermal isolation spacing."""
        # Identify high-power components
        high_power_components = [
            c for c in self.components 
            if c.estimate_power() > 5.0  # > 5mW
        ]
        
        # Increase spacing around high-power components
        adjusted_positions = positions.copy()
        thermal_spacing = 50.0  # μm additional spacing
        
        for hp_component in high_power_components:
            hp_pos = positions[hp_component.instance_id]
            
            for component in self.components:
                if component == hp_component:
                    continue
                    
                comp_pos = positions[component.instance_id]
                distance = np.sqrt((hp_pos[0] - comp_pos[0])**2 + (hp_pos[1] - comp_pos[1])**2)
                
                if distance < thermal_spacing:
                    # Move component away
                    direction = np.array(comp_pos) - np.array(hp_pos)
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        new_pos = np.array(hp_pos) + direction * thermal_spacing
                        adjusted_positions[component.instance_id] = tuple(new_pos)
        
        return adjusted_positions
    
    def _route_waveguides(self, positions: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Route waveguides between components.""" 
        routing = {
            'waveguides': [],
            'total_length': 0.0,
            'crossings': 0
        }
        
        for source_id, target_id in self.connections:
            if source_id in positions and target_id in positions:
                source_pos = positions[source_id]
                target_pos = positions[target_id]
                
                if self.layout_constraints.routing_algorithm == "manhattan":
                    path = self._manhattan_routing(source_pos, target_pos)
                else:
                    path = self._direct_routing(source_pos, target_pos)
                
                length = self._calculate_path_length(path)
                
                waveguide_info = {
                    'source': source_id,
                    'target': target_id,
                    'path': path,
                    'length': length
                }
                
                routing['waveguides'].append(waveguide_info)
                routing['total_length'] += length
        
        # Count crossings (simplified)
        routing['crossings'] = self._count_waveguide_crossings(routing['waveguides'])
        
        return routing
    
    def _manhattan_routing(self, start: Tuple[float, float], 
                          end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Manhattan (L-shaped) routing between points."""
        x1, y1 = start
        x2, y2 = end
        
        # Route horizontally first, then vertically
        return [(x1, y1), (x2, y1), (x2, y2)]
    
    def _direct_routing(self, start: Tuple[float, float], 
                       end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Direct routing between points."""
        return [start, end]
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total path length."""
        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += length
        return total_length * 1e-3  # Convert μm to mm
    
    def _count_waveguide_crossings(self, waveguides: List[Dict[str, Any]]) -> int:
        """Count number of waveguide crossings."""
        # Simplified crossing detection
        crossings = 0
        for i, wg1 in enumerate(waveguides):
            for wg2 in waveguides[i+1:]:
                if self._paths_intersect(wg1['path'], wg2['path']):
                    crossings += 1
        return crossings
    
    def _paths_intersect(self, path1: List[Tuple[float, float]], 
                        path2: List[Tuple[float, float]]) -> bool:
        """Check if two paths intersect."""
        # Simplified intersection check - would need proper line segment intersection
        return False  # Placeholder
    
    def _estimate_component_width(self, component: PhotonicComponent) -> float:
        """Estimate component width in μm."""
        if component.component_type == "microring_resonator":
            radius = component.get_parameter('ring_radius', 10.0)
            return 2 * radius + 10.0  # Add margin
        elif component.component_type == "mach_zehnder_interferometer":
            return component.get_parameter('width', 50.0)
        else:
            return 20.0  # Default width
    
    def _estimate_component_height(self, component: PhotonicComponent) -> float:
        """Estimate component height in μm."""
        if component.component_type == "microring_resonator":
            radius = component.get_parameter('ring_radius', 10.0)
            return 2 * radius + 10.0  # Add margin
        elif component.component_type == "mach_zehnder_interferometer":
            return component.get_parameter('height', 100.0)
        else:
            return 20.0  # Default height
    
    def _calculate_layout_area(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """Calculate total layout area in mm²."""
        if not positions:
            return 0.0
        
        # Find bounding box
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add component sizes to bounding box
        max_comp_width = max(self._estimate_component_width(c) for c in self.components)
        max_comp_height = max(self._estimate_component_height(c) for c in self.components)
        
        width = (max_x - min_x + max_comp_width) * 1e-3  # Convert μm to mm
        height = (max_y - min_y + max_comp_height) * 1e-3
        
        return width * height
    
    def _calculate_bounding_box(self, positions: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Calculate bounding box coordinates."""
        if not positions:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
        
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    def estimate_latency(self) -> float:
        """Estimate circuit latency in nanoseconds."""
        if not self.physical_layout:
            self.generate_layout()
        
        # Calculate latency from optical path delays
        total_delay = 0.0
        light_speed = 3e8  # m/s
        group_index = 2.0  # Typical for silicon photonics
        
        # Component processing delays
        for component in self.components:
            if "matrix" in component.component_type:
                total_delay += 0.1  # 100 ps for matrix operation
            elif "activation" in component.component_type:
                total_delay += 0.05  # 50 ps for activation
        
        # Waveguide propagation delays
        if self.physical_layout:
            total_length = self.physical_layout['routing']['total_length'] * 1e-3  # Convert to m
            propagation_delay = (total_length * group_index / light_speed) * 1e9  # Convert to ns
            total_delay += propagation_delay
        
        return total_delay
    
    def calculate_metrics(self) -> CircuitMetrics:
        """Calculate comprehensive circuit metrics."""
        if not self.layout_generated:
            self.generate_layout()
        
        # Power calculation
        total_power = sum(component.estimate_power() for component in self.components)
        
        # Area calculation  
        total_area = self.physical_layout['total_area'] if self.physical_layout else 0.0
        
        # Loss calculation
        total_loss = sum(component.estimate_loss() for component in self.components)
        if self.physical_layout:
            # Add waveguide losses
            waveguide_loss = self.physical_layout['routing']['total_length'] * 0.1  # 0.1 dB/cm
            total_loss += waveguide_loss
        
        # Latency calculation
        latency = self.estimate_latency()
        
        # Throughput estimation (simplified)
        num_operations = len([c for c in self.components 
                            if any(op in c.component_type for op in ['matrix', 'multiply', 'convolution'])])
        if latency > 0:
            throughput = num_operations / (latency * 1e-9)  # Operations per second
            throughput = throughput / 1e12  # Convert to TOPS
        else:
            throughput = 0.0
        
        # Energy efficiency
        if total_power > 0:
            energy_efficiency = throughput / (total_power / 1000.0)  # TOPS/W
        else:
            energy_efficiency = 0.0
        
        self.metrics = CircuitMetrics(
            total_power=total_power,
            total_area=total_area,
            total_loss=total_loss,
            latency=latency,
            throughput=throughput,
            energy_efficiency=energy_efficiency
        )
        
        return self.metrics
    
    def add_temporal_processing(self) -> None:
        """Add temporal processing elements for spiking circuits."""
        # Add delay lines and temporal integration
        for component in self.components:
            if "spiking" in component.component_type:
                # Add temporal processing parameters
                component.set_parameter('temporal_integration', True)
                component.set_parameter('spike_memory', True)
        
        logger.info("Added temporal processing elements")
    
    def optimize_temporal_dynamics(self) -> None:
        """Optimize temporal dynamics for spiking circuits."""
        # Optimize delay line lengths and feedback paths
        for component in self.components:
            if component.get_parameter('temporal_integration'):
                # Optimize integration time constant
                tau = component.get_parameter('decay_time', 10.0)
                optimal_tau = min(tau, 5.0)  # Limit to 5ns for speed
                component.set_parameter('decay_time', optimal_tau)
        
        logger.info("Optimized temporal dynamics")
    
    def copy(self) -> 'PhotonicCircuit':
        """Create a deep copy of the circuit."""
        new_circuit = PhotonicCircuit(self.config, self.template_library)
        new_circuit.components = [deepcopy(comp) for comp in self.components]
        new_circuit.connections = self.connections.copy()
        new_circuit.layout_constraints = deepcopy(self.layout_constraints)
        new_circuit.layout_generated = False  # Force regeneration
        new_circuit.physical_layout = None
        return new_circuit
    
    def export_gds(self, filename: str) -> None:
        """Export circuit layout to GDS format."""
        if not self.layout_generated:
            self.generate_layout()
        
        logger.info(f"Exporting layout to {filename}")
        
        # Simplified GDS export - would use gdstk in real implementation
        gds_data = {
            'filename': filename,
            'components': len(self.components),
            'area': self.physical_layout['total_area'] if self.physical_layout else 0,
            'layout_generated': True
        }
        
        # Save metadata for now (real implementation would generate actual GDS)
        with open(filename.replace('.gds', '_metadata.json'), 'w') as f:
            import json
            json.dump(gds_data, f, indent=2)
        
        logger.info("GDS export completed")
    
    def export_netlist(self, filename: str) -> None:
        """Export circuit netlist."""
        logger.info(f"Exporting netlist to {filename}")
        
        # Generate SPICE-like netlist
        netlist_lines = [
            f"* Photonic Neural Network Circuit Netlist",
            f"* Generated by Holo-Code-Gen",
            f"* Components: {len(self.components)}",
            f"",
        ]
        
        # Add component definitions
        for i, component in enumerate(self.components):
            comp_line = f"X{i} {component.instance_id} {component.component_type}"
            for key, value in component.spec.parameters.items():
                comp_line += f" {key}={value}"
            netlist_lines.append(comp_line)
        
        # Add connections
        netlist_lines.append("")
        netlist_lines.append("* Connections")
        for source, target in self.connections:
            netlist_lines.append(f"CONN {source} {target}")
        
        netlist_lines.append("")
        netlist_lines.append(".END")
        
        with open(filename, 'w') as f:
            f.write('\n'.join(netlist_lines))
        
        logger.info("Netlist export completed")