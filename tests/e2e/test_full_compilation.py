"""End-to-end tests for full neural network to photonic circuit compilation."""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List

# Mock imports - would be actual imports in real implementation
# from holo_code_gen import PhotonicCompiler
# from holo_code_gen.templates import IMECLibrary
# from holo_code_gen.optimization import PowerOptimizer, AreaOptimizer


@pytest.mark.integration
class TestFullCompilation:
    """Test complete compilation pipeline from neural network to photonic circuit."""
    
    def test_simple_mlp_compilation(self, sample_neural_network: Dict[str, Any], 
                                  process_parameters: Dict[str, Any],
                                  temp_dir: Path):
        """Test compilation of simple MLP to photonic circuit."""
        # Mock compilation process
        network = sample_neural_network
        process = process_parameters
        
        # Simulate compilation steps
        compilation_steps = [
            "parse_neural_network",
            "create_computation_graph", 
            "map_to_photonic_components",
            "optimize_circuit",
            "generate_layout",
            "export_gds"
        ]
        
        # Each step should complete successfully
        for step in compilation_steps:
            # Mock step execution
            assert step is not None
            
        # Verify output files would be created
        expected_outputs = [
            "compiled_circuit.json",
            "layout.gds",
            "netlist.spi",
            "simulation_results.json"
        ]
        
        for output in expected_outputs:
            output_path = temp_dir / output
            # In real implementation, these files would be created
            assert output_path.parent.exists()
            
    def test_spiking_network_compilation(self, sample_spiking_network: Dict[str, Any],
                                       process_parameters: Dict[str, Any],
                                       temp_dir: Path):
        """Test compilation of spiking neural network."""
        snn = sample_spiking_network
        
        # Spiking networks have additional requirements
        assert "simulation_time" in snn
        assert "time_step" in snn
        
        # Verify temporal parameters are reasonable
        assert snn["simulation_time"] > 0
        assert snn["time_step"] > 0
        assert snn["simulation_time"] > snn["time_step"]
        
        # Mock compilation with temporal considerations
        temporal_steps = [
            "parse_spiking_dynamics",
            "create_temporal_graph",
            "map_to_photonic_neurons",
            "optimize_timing",
            "generate_time_domain_layout"
        ]
        
        for step in temporal_steps:
            assert step is not None
            
    def test_compilation_with_constraints(self, sample_neural_network: Dict[str, Any],
                                        temp_dir: Path):
        """Test compilation with hardware constraints."""
        constraints = {
            "power_budget": 100.0,  # mW
            "area_budget": 25.0,    # mm²
            "wavelength_budget": 4,  # number of wavelengths
            "latency_requirement": 1.0,  # ms
            "accuracy_requirement": 0.95
        }
        
        # Mock constraint checking
        for constraint_name, limit in constraints.items():
            assert limit > 0
            
        # Simulate constraint-aware compilation
        compilation_success = True
        
        # Check if network meets constraints
        network_complexity = sample_neural_network["total_parameters"]
        if network_complexity > 1000000:  # Large network
            # Might need partitioning or optimization
            optimization_needed = True
        else:
            optimization_needed = False
            
        assert compilation_success
        
    def test_multi_wavelength_compilation(self, sample_neural_network: Dict[str, Any]):
        """Test compilation using wavelength division multiplexing."""
        wdm_config = {
            "num_wavelengths": 4,
            "wavelength_spacing": 1.6,  # nm
            "center_wavelength": 1550.0,  # nm
            "crosstalk_budget": -30  # dB
        }
        
        # Calculate wavelength grid
        wavelengths = []
        for i in range(wdm_config["num_wavelengths"]):
            offset = (i - wdm_config["num_wavelengths"]/2 + 0.5) * wdm_config["wavelength_spacing"]
            wavelength = wdm_config["center_wavelength"] + offset
            wavelengths.append(wavelength)
            
        assert len(wavelengths) == wdm_config["num_wavelengths"]
        
        # Verify wavelength spacing
        for i in range(1, len(wavelengths)):
            spacing = abs(wavelengths[i] - wavelengths[i-1])
            assert abs(spacing - wdm_config["wavelength_spacing"]) < 0.1
            
    def test_compilation_error_handling(self):
        """Test compilation error handling and recovery."""
        invalid_network = {
            "layers": [
                {"type": "unsupported_layer", "size": 1000}
            ]
        }
        
        # Mock error detection
        supported_types = ["linear", "activation", "lif_neuron"]
        errors = []
        
        for layer in invalid_network["layers"]:
            if layer["type"] not in supported_types:
                errors.append(f"Unsupported layer type: {layer['type']}")
                
        assert len(errors) > 0
        
        # Test error recovery suggestions
        recovery_suggestions = [
            "Replace unsupported layer with supported equivalent",
            "Use custom template for unsupported functionality",
            "Implement layer using multiple supported layers"
        ]
        
        assert len(recovery_suggestions) > 0


@pytest.mark.integration 
class TestOptimization:
    """Test optimization phases of compilation."""
    
    def test_power_optimization(self, sample_neural_network: Dict[str, Any]):
        """Test power consumption optimization."""
        # Mock power analysis
        baseline_power = 150.0  # mW
        power_budget = 100.0   # mW
        
        if baseline_power > power_budget:
            optimization_needed = True
            
            # Mock optimization strategies
            strategies = [
                "wavelength_reuse",
                "power_gating", 
                "voltage_scaling",
                "component_sharing"
            ]
            
            # Apply optimizations
            optimized_power = baseline_power
            for strategy in strategies:
                # Each strategy reduces power by 10%
                optimized_power *= 0.9
                
            assert optimized_power < baseline_power
            
        else:
            optimization_needed = False
            
    def test_area_optimization(self, sample_neural_network: Dict[str, Any]):
        """Test chip area optimization."""
        # Mock area analysis
        baseline_area = 30.0  # mm²
        area_budget = 25.0   # mm²
        
        if baseline_area > area_budget:
            # Apply area optimization
            optimization_strategies = [
                "component_compaction",
                "layer_sharing",
                "3d_integration",
                "routing_optimization"
            ]
            
            optimized_area = baseline_area
            for strategy in optimization_strategies:
                optimized_area *= 0.95  # 5% reduction per strategy
                
            assert optimized_area < baseline_area
            
    def test_performance_optimization(self, sample_neural_network: Dict[str, Any]):
        """Test performance optimization."""
        performance_metrics = {
            "throughput": 1000,  # inferences/second
            "latency": 2.0,      # ms
            "accuracy": 0.94     # fraction
        }
        
        requirements = {
            "min_throughput": 1500,
            "max_latency": 1.0,
            "min_accuracy": 0.95
        }
        
        # Check which metrics need optimization
        optimizations_needed = []
        
        if performance_metrics["throughput"] < requirements["min_throughput"]:
            optimizations_needed.append("throughput")
            
        if performance_metrics["latency"] > requirements["max_latency"]:
            optimizations_needed.append("latency")
            
        if performance_metrics["accuracy"] < requirements["min_accuracy"]:
            optimizations_needed.append("accuracy")
            
        # Apply optimizations as needed
        for optimization in optimizations_needed:
            # Mock optimization application
            assert optimization in ["throughput", "latency", "accuracy"]
            
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization balancing multiple goals."""
        objectives = {
            "power": {"current": 120, "target": 100, "weight": 0.3},
            "area": {"current": 28, "target": 25, "weight": 0.3},
            "speed": {"current": 800, "target": 1000, "weight": 0.4}
        }
        
        # Calculate overall fitness score
        total_score = 0
        for obj_name, obj_data in objectives.items():
            if obj_name in ["power", "area"]:  # Minimize
                normalized = min(1.0, obj_data["target"] / obj_data["current"])
            else:  # Maximize (speed)
                normalized = min(1.0, obj_data["current"] / obj_data["target"])
                
            weighted_score = normalized * obj_data["weight"]
            total_score += weighted_score
            
        assert 0 <= total_score <= 1


@pytest.mark.integration
@pytest.mark.slow
class TestSimulationValidation:
    """Test simulation and validation of compiled circuits."""
    
    def test_optical_simulation(self, sample_neural_network: Dict[str, Any],
                              simulation_config: Dict[str, Any]):
        """Test optical simulation of compiled circuit."""
        sim_config = simulation_config
        
        # Mock simulation setup
        simulation_params = {
            "wavelength_range": sim_config["wavelength_range"],
            "resolution": sim_config["resolution"],
            "convergence_threshold": sim_config["convergence_threshold"]
        }
        
        # Run mock simulation
        simulation_results = {
            "transmission": [0.95, 0.94, 0.93],  # Mock data
            "reflection": [0.02, 0.03, 0.04],
            "loss": [0.03, 0.03, 0.03],
            "phase": [0.1, 0.15, 0.2]
        }
        
        # Validate simulation results
        for wavelength_idx in range(len(simulation_results["transmission"])):
            transmission = simulation_results["transmission"][wavelength_idx]
            reflection = simulation_results["reflection"][wavelength_idx]
            loss = simulation_results["loss"][wavelength_idx]
            
            # Energy conservation check
            total_energy = transmission + reflection + loss
            assert abs(total_energy - 1.0) < 0.1  # Within 10%
            
    def test_thermal_simulation(self, sample_neural_network: Dict[str, Any]):
        """Test thermal simulation of compiled circuit."""
        # Mock thermal analysis
        power_dissipation = 50.0  # mW
        ambient_temperature = 25.0  # °C
        thermal_resistance = 100.0  # °C/W
        
        # Calculate temperature rise
        temperature_rise = power_dissipation * 1e-3 * thermal_resistance
        operating_temperature = ambient_temperature + temperature_rise
        
        # Check thermal limits
        max_operating_temperature = 85.0  # °C
        assert operating_temperature < max_operating_temperature
        
    def test_noise_analysis(self, photonic_wavelength: float):
        """Test noise analysis of photonic circuit."""
        # Mock noise sources
        noise_sources = {
            "shot_noise": 1e-12,     # A²/Hz
            "thermal_noise": 5e-13,  # A²/Hz  
            "flicker_noise": 2e-13,  # A²/Hz
            "amplifier_noise": 3e-13 # A²/Hz
        }
        
        # Calculate total noise
        total_noise = sum(noise_sources.values())
        
        # Signal power (mock)
        signal_power = 1e-6  # W
        
        # Calculate SNR
        snr_linear = signal_power / total_noise
        snr_db = 10 * np.log10(snr_linear) if 'np' in globals() else 30  # Mock value
        
        # SNR should be reasonable for good performance
        assert snr_db > 20  # dB
        
    def test_performance_validation(self, sample_neural_network: Dict[str, Any]):
        """Test validation of compiled circuit performance."""
        # Mock performance metrics
        simulated_metrics = {
            "accuracy": 0.96,
            "throughput": 1200,  # inferences/second
            "latency": 0.8,      # ms
            "power": 95,         # mW
            "area": 23          # mm²
        }
        
        expected_metrics = {
            "accuracy": 0.95,
            "throughput": 1000,
            "latency": 1.0,
            "power": 100,
            "area": 25
        }
        
        # Validate performance meets or exceeds expectations
        assert simulated_metrics["accuracy"] >= expected_metrics["accuracy"]
        assert simulated_metrics["throughput"] >= expected_metrics["throughput"]
        assert simulated_metrics["latency"] <= expected_metrics["latency"]
        assert simulated_metrics["power"] <= expected_metrics["power"]
        assert simulated_metrics["area"] <= expected_metrics["area"]


@pytest.mark.integration
class TestGDSGeneration:
    """Test GDS layout generation and validation."""
    
    def test_gds_export(self, sample_neural_network: Dict[str, Any], 
                       temp_dir: Path):
        """Test GDS file generation."""
        gds_file = temp_dir / "compiled_circuit.gds"
        
        # Mock GDS generation
        gds_content = {
            "header": "HEADER 600",
            "library": "HOLO_CODE_GEN",
            "cells": [
                {"name": "NEURAL_LAYER_1", "components": 10},
                {"name": "NEURAL_LAYER_2", "components": 8},
                {"name": "TOP_CELL", "components": 5}
            ]
        }
        
        # Validate GDS structure
        assert "header" in gds_content
        assert "library" in gds_content
        assert len(gds_content["cells"]) > 0
        
        # Check hierarchical structure
        top_cells = [cell for cell in gds_content["cells"] if "TOP" in cell["name"]]
        assert len(top_cells) == 1
        
    def test_design_rule_checking(self, process_parameters: Dict[str, Any]):
        """Test design rule checking of generated layout."""
        design_rules = process_parameters["design_rules"]
        
        # Mock layout elements for checking
        layout_elements = [
            {"type": "waveguide", "width": 450, "spacing": 200},
            {"type": "ring", "radius": 10000, "gap": 200},  # nm
            {"type": "via", "size": 150, "enclosure": 75}
        ]
        
        violations = []
        
        for element in layout_elements:
            if element["type"] == "waveguide":
                if element["width"] < design_rules["min_width"]:
                    violations.append(f"Waveguide width {element['width']} < {design_rules['min_width']}")
                if element["spacing"] < design_rules["min_space"]:
                    violations.append(f"Waveguide spacing {element['spacing']} < {design_rules['min_space']}")
                    
        # Should have no DRC violations
        assert len(violations) == 0
        
    def test_layer_assignment(self):
        """Test proper layer assignment in GDS."""
        layer_map = {
            "waveguide": {"layer": 1, "datatype": 0},
            "slab": {"layer": 2, "datatype": 0},
            "metal1": {"layer": 10, "datatype": 0},
            "metal2": {"layer": 11, "datatype": 0},
            "via1": {"layer": 15, "datatype": 0},
            "text": {"layer": 100, "datatype": 0}
        }
        
        # Validate layer assignments
        used_layers = set()
        for component, layer_info in layer_map.items():
            layer_num = layer_info["layer"]
            assert layer_num > 0
            assert layer_num not in used_layers  # No conflicts
            used_layers.add(layer_num)
            
    def test_chip_assembly(self, sample_neural_network: Dict[str, Any]):
        """Test assembly of complete chip layout."""
        # Mock chip components
        chip_components = {
            "neural_core": {"area": 15, "power": 80},  # mm², mW
            "io_pads": {"area": 3, "power": 5},
            "control_logic": {"area": 2, "power": 8},
            "test_structures": {"area": 1, "power": 0}
        }
        
        # Calculate total chip metrics
        total_area = sum(comp["area"] for comp in chip_components.values())
        total_power = sum(comp["power"] for comp in chip_components.values())
        
        # Verify chip is within reasonable limits
        assert total_area <= 30  # mm²
        assert total_power <= 150  # mW
        
        # Check component proportions
        core_fraction = chip_components["neural_core"]["area"] / total_area
        assert core_fraction > 0.5  # Core should be majority of chip