#!/usr/bin/env python3
"""Complete example demonstrating end-to-end photonic neural network compilation."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from holo_code_gen.compiler import PhotonicCompiler, SpikingPhotonicCompiler, CompilationConfig
from holo_code_gen.templates import IMECLibrary, ComponentBuilder
from holo_code_gen.optimization import PowerOptimizer, YieldOptimizer, OptimizationConstraints
from holo_code_gen.circuit import LayoutConstraints


def create_simple_neural_network():
    """Create a simple neural network for testing."""
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.activation1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.activation2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.activation1(x)
            x = self.fc2(x)
            x = self.activation2(x)
            x = self.fc3(x)
            return x
    
    return SimpleNN()


def create_spiking_neural_network():
    """Create a simple spiking neural network."""
    class SpikingNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            # Simple spiking behavior (placeholder)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x)) 
            x = self.fc3(x)
            return x
    
    return SpikingNN()


def demonstrate_basic_compilation():
    """Demonstrate basic neural network to photonic circuit compilation."""
    print("=== Basic Neural Network Compilation ===")
    
    # Create neural network
    model = create_simple_neural_network()
    print(f"Created neural network with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Configure compilation
    config = CompilationConfig(
        template_library="imec_v2025_07",
        process="SiN_220nm",
        wavelength=1550.0,
        input_encoding="amplitude",
        output_detection="coherent",
        optimization_target="power",
        power_budget=500.0,  # 500mW budget
        area_budget=50.0     # 50mm² budget
    )
    
    # Initialize compiler
    compiler = PhotonicCompiler(config)
    
    # Compile to photonic circuit
    input_shape = (1, 784)  # MNIST input shape
    photonic_circuit = compiler.compile(model, input_shape)
    
    print(f"Compiled to photonic circuit with {len(photonic_circuit.components)} components")
    
    # Calculate metrics
    metrics = photonic_circuit.calculate_metrics()
    print(f"Circuit Metrics:")
    print(f"  Power: {metrics.total_power:.2f} mW")
    print(f"  Area: {metrics.total_area:.2f} mm²")
    print(f"  Loss: {metrics.total_loss:.2f} dB")
    print(f"  Latency: {metrics.latency:.2f} ns")
    print(f"  Throughput: {metrics.throughput:.2f} TOPS")
    print(f"  Energy Efficiency: {metrics.energy_efficiency:.2f} TOPS/W")
    
    # Generate layout
    layout_constraints = LayoutConstraints(
        max_chip_area=50.0,
        routing_algorithm="manhattan",
        compact_layout=True
    )
    photonic_circuit.generate_layout(layout_constraints)
    print("Generated physical layout")
    
    # Export files
    photonic_circuit.export_gds("simple_nn_circuit.gds")
    photonic_circuit.export_netlist("simple_nn_circuit.spi")
    print("Exported GDS and netlist files")
    
    return photonic_circuit


def demonstrate_spiking_compilation():
    """Demonstrate spiking neural network compilation."""
    print("\n=== Spiking Neural Network Compilation ===")
    
    # Create spiking neural network
    model = create_spiking_neural_network()
    print(f"Created spiking neural network")
    
    # Configure spiking compilation
    config = CompilationConfig(
        template_library="imec_neuromorphic",
        spike_encoding="phase",
        spike_threshold=1.0,
        refractory_period=2.0,
        power_budget=100.0  # Lower power for spiking
    )
    
    # Initialize spiking compiler
    compiler = SpikingPhotonicCompiler(config)
    
    # Compile to spiking photonic circuit
    input_shape = (1, 784)
    spiking_circuit = compiler.compile(model, input_shape)
    
    print(f"Compiled to spiking circuit with {len(spiking_circuit.components)} components")
    
    # Calculate metrics
    metrics = spiking_circuit.calculate_metrics()
    print(f"Spiking Circuit Metrics:")
    print(f"  Power: {metrics.total_power:.2f} mW")
    print(f"  Area: {metrics.total_area:.2f} mm²")
    print(f"  Throughput: {metrics.throughput:.2f} TOPS")
    print(f"  Energy Efficiency: {metrics.energy_efficiency:.2f} TOPS/W")
    
    return spiking_circuit


def demonstrate_power_optimization():
    """Demonstrate power optimization."""
    print("\n=== Power Optimization ===")
    
    # Create and compile a circuit
    model = create_simple_neural_network()
    compiler = PhotonicCompiler()
    circuit = compiler.compile(model, (1, 784))
    
    initial_power = sum(c.estimate_power() for c in circuit.components)
    print(f"Initial power consumption: {initial_power:.2f} mW")
    
    # Apply power optimization
    power_optimizer = PowerOptimizer(power_budget=300.0, circuit=circuit)
    
    constraints = OptimizationConstraints(
        max_power=300.0,
        max_area=40.0,
        temperature_range=(-20.0, 70.0)
    )
    
    optimized_circuit = power_optimizer.optimize(
        circuit,
        constraints=constraints,
        strategies=["wavelength_reuse", "adiabatic_switching", "thermal_management"],
        iterations=100
    )
    
    optimized_power = sum(c.estimate_power() for c in optimized_circuit.components)
    power_reduction = ((initial_power - optimized_power) / initial_power) * 100
    
    print(f"Optimized power consumption: {optimized_power:.2f} mW")
    print(f"Power reduction: {power_reduction:.1f}%")
    
    # Generate power analysis report
    analysis = power_optimizer.power_analysis()
    analysis.plot_breakdown()
    
    return optimized_circuit


def demonstrate_yield_optimization():
    """Demonstrate yield optimization."""
    print("\n=== Yield Optimization ===")
    
    # Create circuit for yield analysis
    model = create_simple_neural_network()
    compiler = PhotonicCompiler()
    circuit = compiler.compile(model, (1, 784))
    
    # Create yield optimizer
    yield_optimizer = YieldOptimizer(target_yield=95.0)
    
    # Run Monte Carlo yield analysis
    performance_specs = {
        "power": 500.0,  # mW
        "loss": 10.0,    # dB
        "latency": 20.0  # ns
    }
    
    print("Running Monte Carlo yield analysis...")
    yield_results = yield_optimizer.monte_carlo(
        circuit, 
        n_samples=1000,  # Reduced for demo
        performance_specs=performance_specs
    )
    
    print(f"Initial yield: {yield_results.yield_percentage:.1f}%")
    print(f"Passed samples: {yield_results.passed_samples}/{yield_results.total_samples}")
    
    # Optimize for yield if needed
    if yield_results.yield_percentage < yield_optimizer.target_yield:
        print("Yield below target, applying yield optimization...")
        robust_circuit = yield_optimizer.optimize_for_yield(
            circuit, 
            target_yield=95.0,
            design_margins=True
        )
        
        # Re-analyze yield
        final_yield = yield_optimizer.monte_carlo(robust_circuit, n_samples=1000)
        print(f"Final yield after optimization: {final_yield.yield_percentage:.1f}%")
    else:
        print("Yield already meets target")
    
    return circuit


def demonstrate_custom_components():
    """Demonstrate custom component creation."""
    print("\n=== Custom Component Creation ===")
    
    # Create custom photonic neuron
    builder = ComponentBuilder()
    
    custom_neuron = builder.set_name("photonic_lif_neuron") \
                          .set_type("spiking_neuron") \
                          .add_parameter("threshold_power", 1.0) \
                          .add_parameter("decay_time", 10.0) \
                          .add_parameter("refractory_period", 2.0) \
                          .add_waveguide((0, 0), (100, 0), 0.5, 5.0) \
                          .add_ring_resonator((50, 10), 10.0, 0.2) \
                          .add_phase_shifter((75, 0), 50.0, "thermal", 10.0) \
                          .build(ports=["input", "output", "control"])
    
    print(f"Created custom component: {custom_neuron.name}")
    print(f"Component type: {custom_neuron.component_type}")
    print(f"Estimated power: {custom_neuron.estimate_power():.2f} mW")
    print(f"Estimated area: {custom_neuron.estimate_area():.6f} mm²") 
    
    # Register in library
    library = IMECLibrary("imec_neuromorphic")
    library.register_custom(custom_neuron)
    print(f"Registered in library. Available components: {library.list_components()}")
    
    return custom_neuron


def demonstrate_template_library():
    """Demonstrate template library usage."""
    print("\n=== Template Library Demonstration ===")
    
    # Load different library versions
    for version in ["imec_v2025_07", "imec_neuromorphic", "imec_basic"]:
        library = IMECLibrary(version)
        components = library.list_components()
        print(f"\n{version} library contains {len(components)} components:")
        for component_name in components:
            component = library.get_component(component_name)
            power = component.estimate_power()
            area = component.estimate_area()
            loss = component.estimate_loss()
            print(f"  {component_name}: {power:.2f}mW, {area:.6f}mm², {loss:.2f}dB")
    
    # Demonstrate component validation
    library = IMECLibrary("imec_v2025_07")
    validation_results = library.validate_all_components()
    
    if validation_results:
        print(f"\nValidation issues found:")
        for component, issues in validation_results.items():
            print(f"  {component}: {', '.join(issues)}")
    else:
        print("\nAll components validated successfully")


def main():
    """Run all demonstrations."""
    print("Holo-Code-Gen Complete Compilation Example")
    print("=" * 50)
    
    try:
        # Basic compilation
        basic_circuit = demonstrate_basic_compilation()
        
        # Spiking compilation  
        spiking_circuit = demonstrate_spiking_compilation()
        
        # Power optimization
        optimized_circuit = demonstrate_power_optimization()
        
        # Yield optimization
        yield_circuit = demonstrate_yield_optimization()
        
        # Custom components
        custom_component = demonstrate_custom_components()
        
        # Template library
        demonstrate_template_library()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nGenerated files:")
        print("  - simple_nn_circuit.gds")
        print("  - simple_nn_circuit.spi") 
        print("  - simple_nn_circuit_metadata.json")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())