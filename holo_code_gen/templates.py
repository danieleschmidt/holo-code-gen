"""Photonic component template system and IMEC library integration."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import json


class ComponentType(Enum):
    """Types of photonic components."""
    WAVEGUIDE = "waveguide"
    MICRORING = "microring_resonator"
    MZI = "mach_zehnder_interferometer"
    PHASE_SHIFTER = "phase_shifter"
    COUPLER = "directional_coupler"
    SPLITTER = "power_splitter"
    DETECTOR = "photodetector"
    MODULATOR = "optical_modulator"
    AMPLIFIER = "optical_amplifier"
    FILTER = "optical_filter"
    WEIGHT_BANK = "weight_bank"
    MESH = "interferometer_mesh"


@dataclass
class ComponentSpec:
    """Specification for a photonic component."""
    name: str
    component_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    layout_info: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate component specification."""
        issues = []
        
        if not self.name:
            issues.append("Component name is required")
            
        if not self.component_type:
            issues.append("Component type is required")
            
        # Type-specific validations
        if self.component_type == "microring_resonator":
            if 'radius' not in self.parameters:
                issues.append("Microring requires radius parameter")
            elif self.parameters['radius'] <= 0:
                issues.append("Microring radius must be positive")
                
        if self.component_type == "mach_zehnder_interferometer":
            if 'arm_length' not in self.parameters:
                issues.append("MZI requires arm_length parameter")
                
        return issues


@dataclass
class PhotonicComponent:
    """Represents a specific photonic component instance."""
    spec: ComponentSpec
    instance_id: str = ""
    position: Tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0  # degrees
    ports: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        """Get component name."""
        return self.spec.name
    
    @property
    def component_type(self) -> str:
        """Get component type."""
        return self.spec.component_type
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get component parameter."""
        return self.spec.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set component parameter."""
        self.spec.parameters[key] = value
    
    def get_performance_metric(self, metric: str) -> Optional[float]:
        """Get performance metric value."""
        return self.spec.performance.get(metric)
    
    def estimate_loss(self) -> float:
        """Estimate optical loss for this component."""
        if self.component_type == "waveguide":
            length = self.get_parameter('length', 0.0)
            loss_per_cm = self.get_parameter('loss_per_cm', 0.1)  # dB/cm
            return length * loss_per_cm / 10.0  # Convert mm to cm
        elif self.component_type == "microring_resonator":
            return self.get_parameter('insertion_loss', 0.5)  # dB
        elif self.component_type == "mach_zehnder_interferometer":
            return self.get_parameter('insertion_loss', 1.0)  # dB
        else:
            return 0.1  # Default loss
    
    def estimate_power(self) -> float:
        """Estimate power consumption for this component."""
        if self.component_type == "phase_shifter":
            pi_power = self.get_parameter('pi_power', 10.0)  # mW for π shift
            phase = self.get_parameter('phase', 0.0)
            return pi_power * abs(phase) / np.pi
        elif self.component_type == "optical_modulator":
            return self.get_parameter('static_power', 5.0)  # mW
        else:
            return 0.0  # Passive components
    
    def estimate_area(self) -> float:
        """Estimate area footprint for this component."""
        if self.component_type == "microring_resonator":
            radius = self.get_parameter('radius', 10.0)  # μm
            return np.pi * (radius * 1e-3) ** 2  # mm²
        elif self.component_type == "mach_zehnder_interferometer":
            arm_length = self.get_parameter('arm_length', 100.0)  # μm
            width = self.get_parameter('width', 20.0)  # μm
            return (arm_length * width) * 1e-6  # mm²
        elif self.component_type == "waveguide":
            length = self.get_parameter('length', 100.0)  # μm
            width = self.get_parameter('width', 0.5)  # μm
            return (length * width) * 1e-6  # mm²
        else:
            return 0.01  # Default 0.01 mm²


class ComponentBuilder:
    """Builder for creating custom photonic components."""
    
    def __init__(self):
        """Initialize component builder."""
        self.spec = ComponentSpec(name="", component_type="")
        self.elements = []
    
    def set_name(self, name: str) -> 'ComponentBuilder':
        """Set component name."""
        self.spec.name = name
        return self
    
    def set_type(self, component_type: str) -> 'ComponentBuilder':
        """Set component type."""
        self.spec.component_type = component_type
        return self
    
    def add_parameter(self, key: str, value: Any) -> 'ComponentBuilder':
        """Add component parameter."""
        self.spec.parameters[key] = value
        return self
    
    def add_constraint(self, key: str, value: Any) -> 'ComponentBuilder':
        """Add component constraint."""
        self.spec.constraints[key] = value
        return self
    
    def add_performance_metric(self, metric: str, value: float) -> 'ComponentBuilder':
        """Add performance metric."""
        self.spec.performance[metric] = value
        return self
    
    def add_waveguide(self, start: Tuple[float, float], end: Tuple[float, float], 
                     width: float, bend_radius: float = 5.0) -> 'ComponentBuilder':
        """Add waveguide element."""
        length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        self.elements.append({
            'type': 'waveguide',
            'start': start,
            'end': end,
            'width': width,
            'bend_radius': bend_radius,
            'length': length
        })
        return self
    
    def add_ring_resonator(self, center: Tuple[float, float], radius: float,
                          gap: float, waveguide_port: str = "auto") -> 'ComponentBuilder':
        """Add ring resonator element."""
        self.elements.append({
            'type': 'ring_resonator',
            'center': center,
            'radius': radius,
            'gap': gap,
            'waveguide_port': waveguide_port
        })
        return self
    
    def add_phase_shifter(self, position: Tuple[float, float], length: float,
                         shifter_type: str = "thermal", power: float = 10.0) -> 'ComponentBuilder':
        """Add phase shifter element."""
        self.elements.append({
            'type': 'phase_shifter',
            'position': position,
            'length': length,
            'shifter_type': shifter_type,
            'pi_power': power
        })
        return self
    
    def build(self, name: Optional[str] = None, ports: Optional[List[str]] = None,
              parameters: Optional[List[str]] = None) -> PhotonicComponent:
        """Build the custom component."""
        if name:
            self.spec.name = name
        
        # Add layout information from elements
        self.spec.layout_info['elements'] = self.elements
        
        if ports:
            self.spec.layout_info['ports'] = ports
            
        if parameters:
            self.spec.layout_info['parameters'] = parameters
        
        # Validate specification
        issues = self.spec.validate()
        if issues:
            raise ValueError(f"Component validation failed: {', '.join(issues)}")
        
        return PhotonicComponent(spec=self.spec)


class IMECLibrary:
    """IMEC photonic component template library."""
    
    LIBRARY_VERSIONS = {
        "imec_v2025_07": "IMEC July 2025 Analog-Photonic IC Templates",
        "imec_neuromorphic": "IMEC Neuromorphic Computing Templates",
        "imec_basic": "IMEC Basic Component Library"
    }
    
    def __init__(self, version: str = "imec_v2025_07"):
        """Initialize IMEC library.
        
        Args:
            version: Library version to use
        """
        self.version = version
        self.components: Dict[str, ComponentSpec] = {}
        self._load_library()
    
    def _load_library(self) -> None:
        """Load component library from templates."""
        if self.version == "imec_v2025_07":
            self._load_imec_2025_library()
        elif self.version == "imec_neuromorphic":
            self._load_neuromorphic_library()
        elif self.version == "imec_basic":
            self._load_basic_library()
        else:
            raise ValueError(f"Unknown library version: {self.version}")
    
    def _load_imec_2025_library(self) -> None:
        """Load IMEC 2025 template library."""
        # Microring weight bank
        self.components["microring_weight_bank"] = ComponentSpec(
            name="microring_weight_bank",
            component_type="weight_bank",
            parameters={
                "num_weights": 64,
                "ring_radius": 10.0,  # μm
                "coupling_gap": 0.2,  # μm
                "waveguide_width": 0.45,  # μm
                "wavelength": 1550.0,  # nm
                "fsr": 20.0,  # nm (free spectral range)
                "finesse": 100.0,
                "q_factor": 10000.0
            },
            constraints={
                "min_gap": 0.15,  # μm
                "max_gap": 0.5,   # μm
                "min_radius": 5.0,  # μm
                "max_radius": 50.0  # μm
            },
            performance={
                "insertion_loss": 2.0,  # dB
                "extinction_ratio": 20.0,  # dB
                "bandwidth": 1.0,  # GHz
                "power_per_weight": 0.1  # mW
            }
        )
        
        # MZI mesh for matrix operations
        self.components["mzi_mesh"] = ComponentSpec(
            name="mzi_mesh",
            component_type="interferometer_mesh",
            parameters={
                "size": (4, 4),
                "splitting_ratio": 0.5,
                "phase_shifter_type": "thermal",
                "control_bits": 8,
                "arm_length": 100.0,  # μm
                "phase_range": 2*np.pi
            },
            constraints={
                "max_size": (16, 16),
                "min_splitting_ratio": 0.1,
                "max_splitting_ratio": 0.9
            },
            performance={
                "insertion_loss": 3.0,  # dB
                "uniformity": 0.1,  # dB
                "phase_noise": 0.01,  # radians
                "switching_speed": 1.0  # μs
            }
        )
        
        # Ring modulator for nonlinearity
        self.components["ring_modulator"] = ComponentSpec(
            name="ring_modulator",
            component_type="optical_modulator",
            parameters={
                "ring_radius": 15.0,  # μm
                "modulation_type": "electro_absorption",
                "voltage_range": (-2.0, 2.0),  # V
                "wavelength": 1550.0,  # nm
                "bandwidth": 10.0  # GHz
            },
            constraints={
                "max_voltage": 5.0,  # V
                "min_extinction": 5.0  # dB
            },
            performance={
                "extinction_ratio": 15.0,  # dB
                "insertion_loss": 1.5,  # dB
                "rise_time": 0.1,  # ns
                "power_consumption": 2.0  # mW
            }
        )
        
        # Basic waveguide
        self.components["waveguide"] = ComponentSpec(
            name="waveguide",
            component_type="waveguide",
            parameters={
                "width": 0.45,  # μm
                "thickness": 0.22,  # μm
                "material": "silicon_nitride",
                "cladding": "silicon_dioxide",
                "bend_radius": 5.0  # μm
            },
            constraints={
                "min_width": 0.3,  # μm
                "max_width": 2.0,  # μm
                "min_bend_radius": 3.0  # μm
            },
            performance={
                "loss_per_cm": 0.1,  # dB/cm
                "group_index": 2.0,
                "dispersion": 0.01,  # ps/nm/km
                "nonlinear_coefficient": 1e-18  # m²/W
            }
        )
    
    def _load_neuromorphic_library(self) -> None:
        """Load neuromorphic computing specific templates."""
        # Photonic LIF neuron
        self.components["photonic_lif_neuron"] = ComponentSpec(
            name="photonic_lif_neuron",
            component_type="spiking_neuron",
            parameters={
                "threshold_power": 1.0,  # mW
                "refractory_delay": 1.0,  # ns
                "decay_time": 10.0,  # ns
                "integration_length": 100.0,  # μm
                "feedback_gain": 0.9
            },
            performance={
                "spike_rate": 1e9,  # Hz
                "jitter": 0.1,  # ns
                "power_per_spike": 1e-12  # J
            }
        )
        
        # Spike detector
        self.components["spike_detector"] = ComponentSpec(
            name="spike_detector",
            component_type="photodetector",
            parameters={
                "responsivity": 0.8,  # A/W
                "dark_current": 1e-9,  # A
                "bandwidth": 10e9,  # Hz
                "area": 100.0  # μm²
            },
            performance={
                "noise_equivalent_power": 1e-12,  # W/√Hz
                "quantum_efficiency": 0.9,
                "detection_threshold": 1e-6  # W
            }
        )
    
    def _load_basic_library(self) -> None:
        """Load basic component library."""
        # Simple directional coupler
        self.components["directional_coupler"] = ComponentSpec(
            name="directional_coupler",
            component_type="coupler",
            parameters={
                "coupling_length": 50.0,  # μm
                "gap": 0.3,  # μm
                "coupling_ratio": 0.5
            },
            performance={
                "insertion_loss": 0.5,  # dB
                "directivity": 20.0,  # dB
                "bandwidth": 50.0  # nm
            }
        )
        
        # Power splitter
        self.components["power_splitter"] = ComponentSpec(
            name="power_splitter",
            component_type="splitter",
            parameters={
                "split_ratio": 0.5,
                "num_outputs": 2,
                "geometry": "y_branch"
            },
            performance={
                "insertion_loss": 0.3,  # dB
                "uniformity": 0.1,  # dB
                "return_loss": 30.0  # dB
            }
        )
    
    def get_component(self, name: str) -> PhotonicComponent:
        """Get component by name."""
        if name not in self.components:
            available = list(self.components.keys())
            raise ValueError(f"Component '{name}' not found. Available: {available}")
        
        spec = self.components[name]
        return PhotonicComponent(spec=spec)
    
    def list_components(self) -> List[str]:
        """List all available component names."""
        return list(self.components.keys())
    
    def get_components_by_type(self, component_type: str) -> List[str]:
        """Get component names by type."""
        return [
            name for name, spec in self.components.items()
            if spec.component_type == component_type
        ]
    
    def register_custom(self, component: PhotonicComponent) -> None:
        """Register custom component in library."""
        name = component.spec.name
        if name in self.components:
            raise ValueError(f"Component '{name}' already exists")
        
        self.components[name] = component.spec
    
    def save_library(self, filepath: Path) -> None:
        """Save library to file."""
        data = {
            'version': self.version,
            'components': {
                name: {
                    'name': spec.name,
                    'component_type': spec.component_type,
                    'parameters': spec.parameters,
                    'constraints': spec.constraints,
                    'performance': spec.performance,
                    'layout_info': spec.layout_info
                }
                for name, spec in self.components.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_library(cls, filepath: Path) -> 'IMECLibrary':
        """Load library from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        library = cls.__new__(cls)  # Create without calling __init__
        library.version = data['version']
        library.components = {}
        
        for name, comp_data in data['components'].items():
            spec = ComponentSpec(
                name=comp_data['name'],
                component_type=comp_data['component_type'],
                parameters=comp_data.get('parameters', {}),
                constraints=comp_data.get('constraints', {}),
                performance=comp_data.get('performance', {}),
                layout_info=comp_data.get('layout_info', {})
            )
            library.components[name] = spec
        
        return library
    
    def validate_all_components(self) -> Dict[str, List[str]]:
        """Validate all components in library."""
        validation_results = {}
        
        for name, spec in self.components.items():
            issues = spec.validate()
            if issues:
                validation_results[name] = issues
        
        return validation_results