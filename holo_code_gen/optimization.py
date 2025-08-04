"""Optimization algorithms for photonic neural networks."""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class OptimizationTarget(Enum):
    """Optimization targets for photonic circuits."""
    POWER = "power"
    AREA = "area"
    LATENCY = "latency"
    LOSS = "loss"
    YIELD = "yield"
    ENERGY_EFFICIENCY = "energy_efficiency"
    THROUGHPUT = "throughput"


@dataclass
class OptimizationConstraints:
    """Constraints for circuit optimization."""
    max_power: Optional[float] = None  # mW
    max_area: Optional[float] = None   # mm²
    max_latency: Optional[float] = None  # ns
    max_loss: Optional[float] = None    # dB
    min_yield: Optional[float] = None   # %
    temperature_range: Tuple[float, float] = (-40.0, 85.0)  # °C
    wavelength_range: Tuple[float, float] = (1500.0, 1600.0)  # nm
    process_variations: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize default process variations."""
        if self.process_variations is None:
            self.process_variations = {
                "waveguide_width": {"mean": 0, "std": 5},  # nm
                "ring_radius": {"mean": 0, "std": 50},      # nm
                "coupling_gap": {"mean": 0, "std": 10}      # nm
            }


@dataclass
class OptimizationMetrics:
    """Metrics for evaluating optimization results."""
    power_consumption: float = 0.0  # mW
    chip_area: float = 0.0          # mm²
    latency: float = 0.0            # ns
    optical_loss: float = 0.0       # dB
    energy_per_operation: float = 0.0  # pJ/op
    throughput: float = 0.0         # TOPS
    yield_estimate: float = 0.0     # %
    
    @property
    def tops_per_watt(self) -> float:
        """Calculate TOPS/W efficiency."""
        if self.power_consumption > 0:
            return self.throughput / (self.power_consumption / 1000.0)
        return 0.0
    
    @property
    def area_efficiency(self) -> float:
        """Calculate TOPS/mm² efficiency."""
        if self.chip_area > 0:
            return self.throughput / self.chip_area
        return 0.0


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    @abstractmethod
    def optimize(self, circuit: 'PhotonicCircuit', constraints: OptimizationConstraints) -> 'PhotonicCircuit':
        """Apply optimization strategy to circuit."""
        pass
    
    @abstractmethod
    def estimate_metrics(self, circuit: 'PhotonicCircuit') -> OptimizationMetrics:
        """Estimate performance metrics for circuit."""
        pass


class PowerOptimizer(OptimizationStrategy):
    """Power-aware optimization for photonic circuits."""
    
    def __init__(self, power_budget: float = 1000.0, circuit: Optional['PhotonicCircuit'] = None):
        """Initialize power optimizer.
        
        Args:
            power_budget: Maximum power budget in mW
            circuit: Circuit to optimize (for analysis)
        """
        self.power_budget = power_budget
        self.circuit = circuit
        self.optimization_strategies = [
            "wavelength_reuse",
            "adiabatic_switching", 
            "resonance_trimming",
            "thermal_management"
        ]
        
    def optimize(self, circuit: Optional['PhotonicCircuit'] = None, 
                constraints: Optional[OptimizationConstraints] = None,
                strategies: Optional[List[str]] = None,
                iterations: int = 1000) -> 'PhotonicCircuit':
        """Optimize circuit for power consumption.
        
        Args:
            circuit: Circuit to optimize
            constraints: Optimization constraints
            strategies: List of optimization strategies to apply
            iterations: Number of optimization iterations
            
        Returns:
            Optimized photonic circuit
        """
        if circuit is None:
            circuit = self.circuit
        if circuit is None:
            raise ValueError("No circuit provided for optimization")
            
        if constraints is None:
            constraints = OptimizationConstraints(max_power=self.power_budget)
            
        if strategies is None:
            strategies = self.optimization_strategies
            
        logger.info(f"Starting power optimization with budget {self.power_budget} mW")
        
        optimized_circuit = circuit.copy()
        
        # Apply each optimization strategy
        for strategy in strategies:
            logger.info(f"Applying {strategy} optimization")
            optimized_circuit = self._apply_strategy(optimized_circuit, strategy, constraints)
            
        # Iterative refinement
        for i in range(iterations):
            current_power = self._estimate_power(optimized_circuit)
            if current_power <= self.power_budget:
                break
                
            # Apply gradient-based optimization
            optimized_circuit = self._gradient_optimization_step(optimized_circuit, constraints)
            
            if i % 100 == 0:
                logger.info(f"Iteration {i}: Power = {current_power:.2f} mW")
        
        final_power = self._estimate_power(optimized_circuit)
        logger.info(f"Optimization complete. Final power: {final_power:.2f} mW")
        
        return optimized_circuit
    
    def _apply_strategy(self, circuit: 'PhotonicCircuit', strategy: str, 
                       constraints: OptimizationConstraints) -> 'PhotonicCircuit':
        """Apply specific optimization strategy."""
        if strategy == "wavelength_reuse":
            return self._optimize_wavelength_reuse(circuit)
        elif strategy == "adiabatic_switching":
            return self._optimize_adiabatic_switching(circuit)
        elif strategy == "resonance_trimming":
            return self._optimize_resonance_trimming(circuit)
        elif strategy == "thermal_management":
            return self._optimize_thermal_management(circuit, constraints)
        else:
            logger.warning(f"Unknown optimization strategy: {strategy}")
            return circuit
    
    def _optimize_wavelength_reuse(self, circuit: 'PhotonicCircuit') -> 'PhotonicCircuit':
        """Optimize wavelength allocation for reuse."""
        # Identify opportunities for wavelength reuse
        wavelength_map = {}
        for component in circuit.components:
            wavelength = component.get_parameter('wavelength', 1550.0)
            if wavelength not in wavelength_map:
                wavelength_map[wavelength] = []
            wavelength_map[wavelength].append(component)
        
        # Merge compatible operations on same wavelength
        optimized_circuit = circuit.copy()
        for wavelength, components in wavelength_map.items():
            if len(components) > 1:
                # Check if components can share wavelength
                compatible_components = self._find_compatible_components(components)
                if compatible_components:
                    optimized_circuit = self._merge_wavelength_operations(
                        optimized_circuit, compatible_components, wavelength
                    )
        
        return optimized_circuit
    
    def _optimize_adiabatic_switching(self, circuit: 'PhotonicCircuit') -> 'PhotonicCircuit':
        """Optimize for adiabatic switching to reduce power."""
        optimized_circuit = circuit.copy()
        
        # Find phase shifters and optimize switching protocols
        for component in optimized_circuit.components:
            if component.component_type == "phase_shifter":
                # Implement adiabatic switching
                current_power = component.get_parameter('pi_power', 10.0)
                adiabatic_power = current_power * 0.7  # 30% reduction typical
                component.set_parameter('pi_power', adiabatic_power)
                component.set_parameter('switching_type', 'adiabatic')
        
        return optimized_circuit
    
    def _optimize_resonance_trimming(self, circuit: 'PhotonicCircuit') -> 'PhotonicCircuit':
        """Optimize resonator trimming to reduce tuning power."""
        optimized_circuit = circuit.copy()
        
        # Optimize ring resonator parameters
        for component in optimized_circuit.components:
            if "ring" in component.component_type:
                # Optimize resonance wavelength to minimize tuning
                target_wavelength = component.get_parameter('wavelength', 1550.0)
                optimal_radius = self._calculate_optimal_radius(target_wavelength)
                component.set_parameter('ring_radius', optimal_radius)
                
                # Reduce tuning power requirement
                tuning_power = component.get_parameter('tuning_power', 5.0)
                optimized_power = tuning_power * 0.8  # 20% reduction
                component.set_parameter('tuning_power', optimized_power)
        
        return optimized_circuit
    
    def _optimize_thermal_management(self, circuit: 'PhotonicCircuit', 
                                   constraints: OptimizationConstraints) -> 'PhotonicCircuit':
        """Optimize thermal management to reduce power."""
        optimized_circuit = circuit.copy()
        
        # Distribute heat sources to minimize hotspots
        heat_sources = [c for c in optimized_circuit.components 
                       if c.estimate_power() > 1.0]  # Components > 1mW
        
        # Implement thermal-aware placement
        if len(heat_sources) > 1:
            optimized_circuit = self._thermal_aware_placement(optimized_circuit, heat_sources)
        
        # Add thermal isolation where needed
        for component in optimized_circuit.components:
            if component.estimate_power() > 5.0:  # High power components
                component.set_parameter('thermal_isolation', True)
        
        return optimized_circuit
    
    def _estimate_power(self, circuit: 'PhotonicCircuit') -> float:
        """Estimate total power consumption."""
        total_power = 0.0
        for component in circuit.components:
            total_power += component.estimate_power()
        return total_power
    
    def _gradient_optimization_step(self, circuit: 'PhotonicCircuit', 
                                  constraints: OptimizationConstraints) -> 'PhotonicCircuit':
        """Apply gradient-based optimization step."""
        # Simplified gradient descent on component parameters
        optimized_circuit = circuit.copy()
        learning_rate = 0.01
        
        for component in optimized_circuit.components:
            if component.component_type == "phase_shifter":
                current_phase = component.get_parameter('phase', 0.0)
                power_gradient = self._compute_power_gradient(component, 'phase')
                new_phase = current_phase - learning_rate * power_gradient
                new_phase = np.clip(new_phase, 0, 2*np.pi)
                component.set_parameter('phase', new_phase)
        
        return optimized_circuit
    
    def _compute_power_gradient(self, component: 'PhotonicComponent', parameter: str) -> float:
        """Compute power gradient with respect to parameter."""
        # Simplified gradient computation
        if parameter == 'phase' and component.component_type == "phase_shifter":
            current_phase = component.get_parameter('phase', 0.0)
            pi_power = component.get_parameter('pi_power', 10.0)
            # Power gradient: d(power)/d(phase) = pi_power/π for linear model
            return pi_power / np.pi
        return 0.0
    
    def estimate_metrics(self, circuit: 'PhotonicCircuit') -> OptimizationMetrics:
        """Estimate performance metrics for power optimization."""
        total_power = self._estimate_power(circuit)
        total_area = sum(c.estimate_area() for c in circuit.components)
        total_loss = sum(c.estimate_loss() for c in circuit.components)
        
        # Estimate latency from critical path
        latency = circuit.estimate_latency() if hasattr(circuit, 'estimate_latency') else 10.0
        
        # Estimate throughput (simplified)
        num_operations = len([c for c in circuit.components 
                            if 'matrix' in c.component_type or 'multiply' in c.component_type])
        if latency > 0:
            throughput = num_operations * 1e12 / (latency * 1e-9)  # TOPS
        else:
            throughput = 0.0
        
        return OptimizationMetrics(
            power_consumption=total_power,
            chip_area=total_area,
            latency=latency,
            optical_loss=total_loss,
            energy_per_operation=total_power * latency / num_operations if num_operations > 0 else 0,
            throughput=throughput / 1e12,  # Convert to TOPS
            yield_estimate=95.0  # Simplified yield estimate
        )
    
    def power_analysis(self) -> 'PowerAnalysisReport':
        """Generate detailed power analysis report."""
        return PowerAnalysisReport(self.circuit, self)
    
    # Helper methods
    def _find_compatible_components(self, components: List['PhotonicComponent']) -> List['PhotonicComponent']:
        """Find components compatible for wavelength sharing."""
        # Simplified compatibility check
        return components[:2] if len(components) >= 2 else []
    
    def _merge_wavelength_operations(self, circuit: 'PhotonicCircuit', 
                                   components: List['PhotonicComponent'], 
                                   wavelength: float) -> 'PhotonicCircuit':
        """Merge operations on the same wavelength."""
        # Simplified merging - return circuit unchanged for now
        return circuit
    
    def _calculate_optimal_radius(self, wavelength: float) -> float:
        """Calculate optimal ring radius for given wavelength."""
        # Simplified calculation: radius ∝ wavelength
        return wavelength * 0.01  # μm
    
    def _thermal_aware_placement(self, circuit: 'PhotonicCircuit', 
                               heat_sources: List['PhotonicComponent']) -> 'PhotonicCircuit':
        """Apply thermal-aware component placement."""
        # Simplified thermal placement - return circuit unchanged
        return circuit


class YieldOptimizer(OptimizationStrategy):
    """Yield optimization accounting for process variations."""
    
    def __init__(self, target_yield: float = 99.0):
        """Initialize yield optimizer.
        
        Args:
            target_yield: Target yield percentage
        """
        self.target_yield = target_yield
        self.process_variations = {
            "waveguide_width": {"mean": 0, "std": 5},  # nm
            "ring_radius": {"mean": 0, "std": 50},      # nm
            "coupling_gap": {"mean": 0, "std": 10}      # nm
        }
    
    def optimize(self, circuit: 'PhotonicCircuit', 
                constraints: Optional[OptimizationConstraints] = None) -> 'PhotonicCircuit':
        """Optimize circuit for manufacturing yield."""
        if constraints is None:
            constraints = OptimizationConstraints(min_yield=self.target_yield)
        
        logger.info(f"Starting yield optimization for {self.target_yield}% yield")
        
        # Run Monte Carlo yield analysis
        yield_results = self.monte_carlo(circuit, n_samples=10000)
        
        if yield_results.yield_percentage >= self.target_yield:
            return circuit
        
        # Apply yield optimization strategies
        optimized_circuit = self.optimize_for_yield(circuit, self.target_yield)
        
        return optimized_circuit
    
    def monte_carlo(self, circuit: 'PhotonicCircuit', n_samples: int = 10000,
                   performance_specs: Optional[Dict[str, float]] = None) -> 'YieldResults':
        """Run Monte Carlo yield analysis."""
        if performance_specs is None:
            performance_specs = {
                "ber": 1e-9,
                "power": self.target_yield,  # mW
                "latency": 10   # ns
            }
        
        logger.info(f"Running Monte Carlo analysis with {n_samples} samples")
        
        passed_samples = 0
        failed_reasons = []
        
        for i in range(n_samples):
            # Generate random process variations
            varied_circuit = self._apply_process_variations(circuit)
            
            # Check if circuit meets specifications
            meets_specs, reason = self._check_specifications(varied_circuit, performance_specs)
            
            if meets_specs:
                passed_samples += 1
            else:
                failed_reasons.append(reason)
        
        yield_percentage = (passed_samples / n_samples) * 100.0
        
        logger.info(f"Monte Carlo yield: {yield_percentage:.1f}%")
        
        return YieldResults(
            yield_percentage=yield_percentage,
            total_samples=n_samples,
            passed_samples=passed_samples,
            failed_reasons=failed_reasons
        )
    
    def optimize_for_yield(self, circuit: 'PhotonicCircuit', target_yield: float,
                          design_margins: bool = True) -> 'PhotonicCircuit':
        """Optimize circuit design for target yield."""
        optimized_circuit = circuit.copy()
        
        if design_margins:
            # Add design margins to critical parameters
            for component in optimized_circuit.components:
                if "ring" in component.component_type:
                    # Add margin to ring radius
                    radius = component.get_parameter('ring_radius', 10.0)
                    margin = 0.1  # 10% margin
                    robust_radius = radius * (1 + margin)
                    component.set_parameter('ring_radius', robust_radius)
                
                if component.component_type == "directional_coupler":
                    # Add margin to coupling gap
                    gap = component.get_parameter('gap', 0.3)
                    margin = 0.05  # 50nm margin
                    robust_gap = gap + margin
                    component.set_parameter('gap', robust_gap)
        
        # Apply robust design principles
        optimized_circuit = self._apply_robust_design(optimized_circuit)
        
        return optimized_circuit
    
    def _apply_process_variations(self, circuit: 'PhotonicCircuit') -> 'PhotonicCircuit':
        """Apply random process variations to circuit."""
        varied_circuit = circuit.copy()
        
        for component in varied_circuit.components:
            # Apply variations to relevant parameters
            for param, variation in self.process_variations.items():
                if param == "waveguide_width" and component.component_type == "waveguide":
                    current_width = component.get_parameter('width', 0.45)
                    delta = np.random.normal(variation['mean'], variation['std']) * 1e-3  # Convert nm to μm
                    new_width = current_width + delta
                    component.set_parameter('width', max(0.1, new_width))  # Minimum width constraint
                
                elif param == "ring_radius" and "ring" in component.component_type:
                    current_radius = component.get_parameter('ring_radius', 10.0)
                    delta = np.random.normal(variation['mean'], variation['std']) * 1e-3  # Convert nm to μm
                    new_radius = current_radius + delta
                    component.set_parameter('ring_radius', max(1.0, new_radius))  # Minimum radius
                
                elif param == "coupling_gap" and component.component_type == "directional_coupler":
                    current_gap = component.get_parameter('gap', 0.3)
                    delta = np.random.normal(variation['mean'], variation['std']) * 1e-3  # Convert nm to μm
                    new_gap = current_gap + delta
                    component.set_parameter('gap', max(0.1, new_gap))  # Minimum gap
        
        return varied_circuit
    
    def _check_specifications(self, circuit: 'PhotonicCircuit', 
                            specs: Dict[str, float]) -> Tuple[bool, str]:
        """Check if circuit meets performance specifications."""
        # Simplified specification checking
        total_power = sum(c.estimate_power() for c in circuit.components)
        
        if "power" in specs and total_power > specs["power"]:
            return False, f"Power {total_power:.1f} mW exceeds limit {specs['power']} mW"
        
        total_loss = sum(c.estimate_loss() for c in circuit.components)
        if "loss" in specs and total_loss > specs["loss"]:
            return False, f"Loss {total_loss:.1f} dB exceeds limit {specs['loss']} dB"
        
        # All specs passed
        return True, ""
    
    def _apply_robust_design(self, circuit: 'PhotonicCircuit') -> 'PhotonicCircuit':
        """Apply robust design principles to improve yield."""
        # Use redundancy for critical components
        critical_components = [c for c in circuit.components 
                             if c.component_type in ["phase_shifter", "ring_modulator"]]
        
        for component in critical_components:
            # Add backup/redundant elements where possible
            component.set_parameter('redundancy_level', 2)
            component.set_parameter('error_correction', True)
        
        return circuit
    
    def estimate_metrics(self, circuit: 'PhotonicCircuit') -> OptimizationMetrics:
        """Estimate metrics with yield consideration."""
        # Run simplified yield analysis
        yield_result = self.monte_carlo(circuit, n_samples=1000)
        
        # Base metrics
        total_power = sum(c.estimate_power() for c in circuit.components)
        total_area = sum(c.estimate_area() for c in circuit.components)
        total_loss = sum(c.estimate_loss() for c in circuit.components)
        
        return OptimizationMetrics(
            power_consumption=total_power,
            chip_area=total_area,
            optical_loss=total_loss,
            yield_estimate=yield_result.yield_percentage
        )


@dataclass
class YieldResults:
    """Results from yield analysis."""
    yield_percentage: float
    total_samples: int
    passed_samples: int
    failed_reasons: List[str]


class PowerAnalysisReport:
    """Detailed power analysis report."""
    
    def __init__(self, circuit: 'PhotonicCircuit', optimizer: PowerOptimizer):
        """Initialize power analysis report."""
        self.circuit = circuit
        self.optimizer = optimizer
        self._analyze()
    
    def _analyze(self) -> None:
        """Perform power analysis."""
        self.static_mW = 0.0
        self.dynamic_mW = 0.0
        self.component_breakdown = {}
        
        for component in self.circuit.components:
            power = component.estimate_power()
            self.component_breakdown[component.name] = power
            
            # Classify as static or dynamic
            if component.component_type in ["phase_shifter", "optical_modulator"]:
                self.dynamic_mW += power
            else:
                self.static_mW += power
    
    def plot_breakdown(self) -> None:
        """Plot power breakdown by component."""
        # Simplified plotting - would use matplotlib in real implementation
        print("Power Breakdown:")
        for component, power in self.component_breakdown.items():
            print(f"  {component}: {power:.2f} mW")
        print(f"Total Static: {self.static_mW:.2f} mW")
        print(f"Total Dynamic: {self.dynamic_mW:.2f} mW")
    
    @property
    def has_violations(self) -> bool:
        """Check if there are power budget violations."""
        total_power = self.static_mW + self.dynamic_mW
        return total_power > self.optimizer.power_budget