#!/usr/bin/env python3
"""
Basic functionality test for Holo-Code-Gen without PyTorch dependency.
Tests core components independently.
"""

import sys
sys.path.insert(0, '/root/repo')

def test_imports():
    """Test that core modules can be imported."""
    print("Testing imports...")
    
    # Test individual modules
    from holo_code_gen.ir import ComputationGraph, CircuitNode
    from holo_code_gen.circuit import PhotonicCircuit, CircuitMetrics
    from holo_code_gen.templates import IMECLibrary, PhotonicComponent
    from holo_code_gen.optimization import PowerOptimizer, YieldOptimizer
    
    print("✓ All core modules imported successfully")

def test_template_library():
    """Test IMEC template library functionality."""
    print("\nTesting template library...")
    
    from holo_code_gen.templates import IMECLibrary
    
    # Test library initialization
    library = IMECLibrary("imec_v2025_07")
    print(f"✓ Library initialized with version: {library.version}")
    
    # Test component listing
    components = library.list_components()
    print(f"✓ Found {len(components)} components: {components}")
    
    # Test getting specific components
    waveguide = library.get_component("waveguide")
    print(f"✓ Retrieved waveguide component: {waveguide.name}")
    
    microring = library.get_component("microring_weight_bank")
    print(f"✓ Retrieved microring component: {microring.name}")

def test_computation_graph():
    """Test computation graph functionality."""
    print("\nTesting computation graph...")
    
    from holo_code_gen.ir import ComputationGraph, CircuitNode
    
    # Create computation graph
    graph = ComputationGraph()
    
    # Add nodes
    input_node = CircuitNode(
        name="input",
        node_type="input",
        parameters={"size": 784},
        output_shape=(784,)
    )
    
    layer1_node = CircuitNode(
        name="layer1",
        node_type="matrix_multiply",
        parameters={"input_size": 784, "output_size": 128},
        input_shape=(784,),
        output_shape=(128,)
    )
    
    activation_node = CircuitNode(
        name="activation",
        node_type="activation",
        parameters={"activation_type": "relu"},
        input_shape=(128,),
        output_shape=(128,)
    )
    
    graph.add_node(input_node)
    graph.add_node(layer1_node)
    graph.add_node(activation_node)
    
    # Add edges
    graph.add_edge("input", "layer1")
    graph.add_edge("layer1", "activation")
    
    print(f"✓ Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Test graph operations
    topo_order = graph.topological_sort()
    print(f"✓ Topological order: {topo_order}")
    
    complexity = sum(node.get_complexity() for node in graph.nodes.values())
    print(f"✓ Total complexity: {complexity}")
    
    # Test validation
    issues = graph.validate()
    print(f"✓ Validation issues: {len(issues)}")

def test_photonic_circuit():
    """Test photonic circuit functionality."""
    print("\nTesting photonic circuit...")
    
    from holo_code_gen.circuit import PhotonicCircuit
    from holo_code_gen.ir import CircuitNode
    from holo_code_gen.templates import IMECLibrary
    
    # Create circuit
    library = IMECLibrary()
    circuit = PhotonicCircuit(template_library=library)
    
    # Create circuit nodes
    node1 = CircuitNode(
        name="matrix_mult_1",
        node_type="matrix_multiply",
        parameters={"input_size": 784, "output_size": 128}
    )
    
    node2 = CircuitNode(
        name="activation_1", 
        node_type="optical_nonlinearity",
        parameters={"activation_type": "relu"}
    )
    
    # Add to circuit
    circuit.add_component(node1)
    circuit.add_component(node2)
    circuit.add_connection("matrix_mult_1", "activation_1")
    
    print(f"✓ Created circuit with {len(circuit.components)} components")
    
    # Test layout generation
    circuit.generate_layout()
    print("✓ Layout generated successfully")
    
    # Test metrics calculation
    metrics = circuit.calculate_metrics()
    print(f"✓ Metrics: Power={metrics.total_power:.2f}mW, Area={metrics.total_area:.2f}mm²")

def test_optimization():
    """Test optimization functionality."""
    print("\nTesting optimization...")
    
    from holo_code_gen.optimization import PowerOptimizer, YieldOptimizer
    from holo_code_gen.circuit import PhotonicCircuit
    from holo_code_gen.templates import IMECLibrary
    from holo_code_gen.ir import CircuitNode
    
    # Create test circuit
    library = IMECLibrary()
    circuit = PhotonicCircuit(template_library=library)
    
    node = CircuitNode(
        name="test_node",
        node_type="phase_shifter",
        parameters={"pi_power": 10.0, "phase": 1.57}  # π/2
    )
    circuit.add_component(node)
    
    # Test power optimization
    power_optimizer = PowerOptimizer(power_budget=500.0)
    optimized = power_optimizer.optimize(circuit)
    
    print("✓ Power optimization completed")
    
    # Test yield optimization
    yield_optimizer = YieldOptimizer(target_yield=95.0)
    yield_result = yield_optimizer.monte_carlo(circuit, n_samples=100)
    
    print(f"✓ Yield analysis: {yield_result.yield_percentage:.1f}%")

def test_end_to_end_simple():
    """Test simplified end-to-end functionality without PyTorch."""
    print("\nTesting end-to-end functionality...")
    
    from holo_code_gen.ir import ComputationGraph, CircuitNode
    from holo_code_gen.circuit import PhotonicCircuit
    from holo_code_gen.templates import IMECLibrary
    from holo_code_gen.optimization import PowerOptimizer
    
    # Create computation graph
    graph = ComputationGraph()
    
    # Simple 2-layer network
    nodes = [
        CircuitNode("input", "input", {"size": 4}),
        CircuitNode("layer1", "matrix_multiply", {"input_size": 4, "output_size": 2}),
        CircuitNode("activation", "optical_nonlinearity"),
        CircuitNode("output", "output", {"size": 2})
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Connect nodes
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i].name, nodes[i+1].name)
    
    print(f"✓ Created computation graph with {len(nodes)} nodes")
    
    # Convert to photonic circuit
    library = IMECLibrary()
    circuit = PhotonicCircuit(template_library=library)
    
    for node in graph.nodes.values():
        if node.node_type != "input" and node.node_type != "output":
            circuit.add_component(node)
    
    # Generate layout
    circuit.generate_layout()
    
    # Calculate metrics
    metrics = circuit.calculate_metrics()
    
    # Optimize
    optimizer = PowerOptimizer(power_budget=100.0)
    optimized_circuit = optimizer.optimize(circuit)
    optimized_metrics = optimized_circuit.calculate_metrics()
    
    print(f"✓ Original: {metrics.total_power:.2f}mW, {metrics.total_area:.4f}mm²")
    print(f"✓ Optimized: {optimized_metrics.total_power:.2f}mW, {optimized_metrics.total_area:.4f}mm²")
    print("✓ End-to-end test completed successfully")

def main():
    """Run all tests."""
    print("=" * 60)
    print("HOLO-CODE-GEN BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        test_imports()
        test_template_library()
        test_computation_graph()
        test_photonic_circuit()
        test_optimization()
        test_end_to_end_simple()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - GENERATION 1 IMPLEMENTATION WORKING!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())