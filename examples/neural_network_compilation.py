"""Example: Neural network to photonic circuit compilation."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import holo-code-gen components
from holo_code_gen import PhotonicCompiler, IMECLibrary, CompilationConfig
from holo_code_gen.optimization import PowerOptimizer, YieldOptimizer


def create_simple_mlp():
    """Create a simple multi-layer perceptron for demonstration."""
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x
    
    return SimpleMLP()


def basic_compilation_example():
    """Basic neural network compilation example."""
    print("=== Basic Neural Network Compilation ===")
    
    # Create neural network
    model = create_simple_mlp()
    print(f"Created MLP with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup compilation configuration
    config = CompilationConfig(
        template_library="imec_v2025_07",
        process="SiN_220nm",
        wavelength=1550.0,
        optimization_target="power",
        power_budget=500.0,  # 500 mW
        area_budget=50.0     # 50 mm²
    )
    
    # Initialize compiler
    compiler = PhotonicCompiler(config)
    
    # Compile to photonic circuit
    print("Compiling neural network to photonic circuit...")
    photonic_circuit = compiler.compile(
        model=model,
        input_shape=(1, 784)  # MNIST input shape
    )
    
    # Calculate metrics
    metrics = photonic_circuit.calculate_metrics()
    
    print("\\nCompilation Results:")
    print(f"  Power Consumption: {metrics.total_power:.2f} mW")
    print(f"  Chip Area: {metrics.total_area:.2f} mm²")
    print(f"  Latency: {metrics.latency:.2f} ns")
    print(f"  Energy Efficiency: {metrics.energy_efficiency:.2f} TOPS/W")
    print(f"  Throughput: {metrics.throughput:.2f} TOPS")
    
    # Export results
    output_dir = Path("output/basic_compilation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    photonic_circuit.export_gds(str(output_dir / "mlp_circuit.gds"))
    photonic_circuit.export_netlist(str(output_dir / "mlp_circuit.spi"))
    
    print(f"\\nFiles exported to {output_dir}/")
    
    return photonic_circuit


def power_optimization_example():
    """Power optimization example."""
    print("\\n=== Power Optimization Example ===")
    
    # Create and compile base circuit
    model = create_simple_mlp()
    config = CompilationConfig(power_budget=1000.0)
    compiler = PhotonicCompiler(config)
    base_circuit = compiler.compile(model, input_shape=(1, 784))
    
    base_power = base_circuit.calculate_metrics().total_power
    print(f"Base circuit power: {base_power:.2f} mW")
    
    # Apply power optimization
    power_optimizer = PowerOptimizer(power_budget=300.0)
    optimized_circuit = power_optimizer.optimize(
        base_circuit,
        strategies=["wavelength_reuse", "adiabatic_switching", "thermal_management"],
        iterations=500
    )
    
    optimized_power = optimized_circuit.calculate_metrics().total_power
    power_reduction = ((base_power - optimized_power) / base_power) * 100
    
    print(f"Optimized circuit power: {optimized_power:.2f} mW")
    print(f"Power reduction: {power_reduction:.1f}%")
    
    # Detailed power analysis
    power_report = power_optimizer.power_analysis()
    print("\\nPower Breakdown:")
    power_report.plot_breakdown()
    
    return optimized_circuit


def yield_optimization_example():
    """Manufacturing yield optimization example."""
    print("\\n=== Yield Optimization Example ===")
    
    # Create base circuit
    model = create_simple_mlp()
    compiler = PhotonicCompiler()
    circuit = compiler.compile(model, input_shape=(1, 784))
    
    # Setup yield optimizer
    yield_optimizer = YieldOptimizer(target_yield=99.0)
    
    # Run Monte Carlo yield analysis
    print("Running Monte Carlo yield analysis...")
    yield_results = yield_optimizer.monte_carlo(
        circuit,
        n_samples=5000,
        performance_specs={
            "power": 800.0,  # mW
            "loss": 10.0,    # dB
            "latency": 20.0  # ns
        }
    )
    
    print(f"Estimated yield: {yield_results.yield_percentage:.1f}%")
    print(f"Samples passed: {yield_results.passed_samples}/{yield_results.total_samples}")
    
    # Optimize for yield if needed
    if yield_results.yield_percentage < 99.0:
        print("\\nOptimizing for higher yield...")
        robust_circuit = yield_optimizer.optimize_for_yield(
            circuit,
            target_yield=99.0,
            design_margins=True
        )
        
        # Re-run analysis
        new_yield = yield_optimizer.monte_carlo(robust_circuit, n_samples=1000)
        print(f"Optimized yield: {new_yield.yield_percentage:.1f}%")
    
    return circuit


def spiking_neural_network_example():
    """Spiking neural network compilation example."""
    print("\\n=== Spiking Neural Network Example ===")
    
    # Create a simple spiking network (using regular PyTorch for demo)
    class SpikingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128) 
            self.fc3 = nn.Linear(128, 10)
            # In real implementation, these would be spiking layers
        
        def forward(self, x):
            # Simplified spiking dynamics
            x = torch.relu(self.fc1(x))  # Would be LIF neurons
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    snn_model = SpikingMLP()
    
    # Configure for spiking compilation
    config = CompilationConfig(
        template_library="imec_neuromorphic",
        spike_encoding="phase",
        spike_threshold=1.0,
        refractory_period=2.0
    )
    
    # Use spiking compiler
    from holo_code_gen import SpikingPhotonicCompiler
    spiking_compiler = SpikingPhotonicCompiler(config)
    
    print("Compiling spiking neural network...")
    spiking_circuit = spiking_compiler.compile(
        model=snn_model,
        input_shape=(1, 784)
    )
    
    metrics = spiking_circuit.calculate_metrics()
    print("\\nSpiking Circuit Results:")
    print(f"  Power: {metrics.total_power:.2f} mW")
    print(f"  Area: {metrics.total_area:.2f} mm²")
    print(f"  Latency: {metrics.latency:.2f} ns")
    print(f"  Energy/spike: {metrics.energy_per_operation:.2f} pJ")
    
    return spiking_circuit


def advanced_optimization_example():
    """Advanced multi-objective optimization example."""
    print("\\n=== Multi-Objective Optimization Example ===")
    
    # Create larger network
    class LargerMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    large_model = LargerMLP()
    
    # Compile with different optimization targets
    targets = ["power", "area", "latency"]
    results = {}
    
    for target in targets:
        print(f"\\nOptimizing for {target}...")
        config = CompilationConfig(optimization_target=target)
        compiler = PhotonicCompiler(config)
        circuit = compiler.compile(large_model, input_shape=(1, 784))
        
        metrics = circuit.calculate_metrics()
        results[target] = {
            'power': metrics.total_power,
            'area': metrics.total_area,
            'latency': metrics.latency,
            'efficiency': metrics.energy_efficiency
        }
    
    # Compare results
    print("\\nOptimization Comparison:")
    print("Target       Power(mW)  Area(mm²)  Latency(ns)  Efficiency(TOPS/W)")
    print("-" * 65)
    for target, metrics in results.items():
        print(f"{target:10s}   {metrics['power']:8.2f}  {metrics['area']:8.2f}  "
              f"{metrics['latency']:10.2f}  {metrics['efficiency']:13.2f}")
    
    return results


def component_library_example():
    """Demonstrate component library usage."""
    print("\\n=== Component Library Example ===")
    
    # Load different template libraries
    libraries = ["imec_v2025_07", "imec_neuromorphic", "imec_basic"]
    
    for lib_name in libraries:
        print(f"\\n{lib_name} Library:")
        try:
            library = IMECLibrary(lib_name)
            components = library.list_components()
            print(f"  Available components: {len(components)}")
            
            for comp_name in components[:3]:  # Show first 3
                component = library.get_component(comp_name)
                power = component.estimate_power()
                area = component.estimate_area()
                print(f"    {comp_name}: {power:.2f} mW, {area:.4f} mm²")
        
        except Exception as e:
            print(f"  Error loading library: {e}")
    
    # Create custom component
    print("\\nCreating custom component...")
    from holo_code_gen.templates import ComponentBuilder
    
    builder = ComponentBuilder()
    custom_neuron = (builder
                    .set_name("custom_photonic_neuron")
                    .set_type("spiking_neuron")
                    .add_parameter("threshold", 1.5)
                    .add_parameter("decay_time", 8.0)
                    .add_waveguide((0, 0), (100, 0), 0.5)
                    .add_ring_resonator((50, 10), 12.0, 0.25)
                    .add_phase_shifter((75, 0), 40.0)
                    .build())
    
    print(f"Created custom component: {custom_neuron.name}")
    print(f"  Power: {custom_neuron.estimate_power():.2f} mW")
    print(f"  Area: {custom_neuron.estimate_area():.4f} mm²")


def main():
    """Run all examples."""
    print("Holo-Code-Gen Neural Network Compilation Examples")
    print("=" * 50)
    
    try:
        # Basic compilation
        basic_circuit = basic_compilation_example()
        
        # Power optimization
        optimized_circuit = power_optimization_example()
        
        # Yield optimization
        yield_optimized_circuit = yield_optimization_example()
        
        # Spiking networks
        spiking_circuit = spiking_neural_network_example()
        
        # Advanced optimization
        optimization_results = advanced_optimization_example()
        
        # Component library
        component_library_example()
        
        print("\\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()