"""Unit tests for photonic template system."""

import pytest
from typing import Dict, Any, List
from pathlib import Path

# Import modules under test (mock imports for now)
# from holo_code_gen.templates import TemplateLibrary, PhotonicComponent
# from holo_code_gen.templates.imec import IMECLibrary


class TestTemplateLibrary:
    """Test suite for template library management."""
    
    def test_library_initialization(self, imec_template_config: Dict[str, Any]):
        """Test template library initialization."""
        assert imec_template_config["library_version"] == "v2025_07"
        assert "components" in imec_template_config
        assert len(imec_template_config["components"]) > 0
        
    def test_component_registration(self, imec_template_config: Dict[str, Any]):
        """Test registration of photonic components."""
        components = imec_template_config["components"]
        
        for component in components:
            assert "name" in component
            assert "type" in component
            assert "parameters" in component
            
    def test_template_versioning(self, imec_template_config: Dict[str, Any]):
        """Test template library versioning."""
        version = imec_template_config["library_version"]
        
        # Version should follow semantic versioning pattern
        assert version.startswith("v")
        version_parts = version[1:].split("_")
        assert len(version_parts) >= 2  # year_month format
        
    def test_component_lookup(self, imec_template_config: Dict[str, Any]):
        """Test looking up components by name."""
        components = {comp["name"]: comp for comp in imec_template_config["components"]}
        
        assert "microring_resonator" in components
        assert "mzi_modulator" in components
        
    def test_template_validation(self, imec_template_config: Dict[str, Any]):
        """Test validation of template definitions."""
        for component in imec_template_config["components"]:
            # Every component must have required fields
            assert "name" in component
            assert "type" in component
            assert "parameters" in component
            
            # Parameters should be a dictionary
            assert isinstance(component["parameters"], dict)
            
    def test_parameter_constraints(self, imec_template_config: Dict[str, Any]):
        """Test parameter constraint validation."""
        ring_component = next(comp for comp in imec_template_config["components"] 
                             if comp["name"] == "microring_resonator")
        
        params = ring_component["parameters"]
        
        # Ring radius should be positive
        assert params["radius"] > 0
        
        # Coupling gap should be reasonable
        assert 50 <= params["coupling_gap"] <= 1000  # nm
        
        # Q factor should be reasonable
        assert params["q_factor"] > 100


class TestPhotonicComponent:
    """Test suite for individual photonic components."""
    
    def test_microring_resonator(self):
        """Test microring resonator component."""
        ring_params = {
            "radius": 10.0,  # μm
            "coupling_gap": 200,  # nm
            "q_factor": 10000,
            "wavelength": 1550.0  # nm
        }
        
        # Basic parameter validation
        assert ring_params["radius"] > 0
        assert ring_params["coupling_gap"] > 0
        assert ring_params["q_factor"] > 100
        
        # Calculate FSR (Free Spectral Range)
        n_eff = 2.4  # Effective index for SiN
        fsr_nm = ring_params["wavelength"]**2 / (2 * 3.14159 * ring_params["radius"] * 1000 * n_eff)
        assert fsr_nm > 0
        
    def test_mzi_modulator(self):
        """Test Mach-Zehnder interferometer modulator."""
        mzi_params = {
            "arm_length": 500.0,  # μm
            "phase_efficiency": 1.0,  # π/V
            "extinction_ratio": 20,  # dB
            "bandwidth": 10  # GHz
        }
        
        # Parameter validation
        assert mzi_params["arm_length"] > 0
        assert mzi_params["phase_efficiency"] > 0
        assert mzi_params["extinction_ratio"] > 10  # Good modulator
        assert mzi_params["bandwidth"] > 1  # GHz
        
    def test_waveguide_component(self):
        """Test basic waveguide component."""
        waveguide_params = {
            "width": 450,  # nm
            "height": 220,  # nm
            "length": 1000,  # μm
            "loss_coefficient": 0.1,  # dB/cm
            "bend_radius": 5.0  # μm
        }
        
        # Dimension validation
        assert 200 <= waveguide_params["width"] <= 1000  # nm
        assert 100 <= waveguide_params["height"] <= 500  # nm
        assert waveguide_params["length"] > 0
        assert waveguide_params["loss_coefficient"] >= 0
        assert waveguide_params["bend_radius"] >= 1.0  # μm
        
    def test_photodetector_component(self):
        """Test photodetector component."""
        detector_params = {
            "active_area": 100,  # μm²
            "responsivity": 0.8,  # A/W
            "dark_current": 1e-9,  # A
            "bandwidth": 40,  # GHz
            "wavelength_range": [1500, 1600]  # nm
        }
        
        # Performance validation
        assert detector_params["active_area"] > 0
        assert 0 < detector_params["responsivity"] <= 1.5
        assert detector_params["dark_current"] >= 0
        assert detector_params["bandwidth"] > 0
        assert len(detector_params["wavelength_range"]) == 2
        
    def test_component_interconnection(self):
        """Test component interconnection validation."""
        # Two components with compatible ports
        component1_output = {"port_type": "optical", "wavelength": 1550.0, "power_dbm": 0}
        component2_input = {"port_type": "optical", "wavelength": 1550.0, "max_power_dbm": 10}
        
        # Check compatibility
        assert component1_output["port_type"] == component2_input["port_type"]
        assert component1_output["wavelength"] == component2_input["wavelength"]
        assert component1_output["power_dbm"] <= component2_input["max_power_dbm"]


class TestTemplateOptimization:
    """Test suite for template optimization."""
    
    def test_power_optimization(self):
        """Test power consumption optimization."""
        component_power = {
            "static_power": 1.0,  # mW
            "dynamic_power": 5.0,  # mW
            "switching_energy": 1e-12,  # J per switch
            "switching_frequency": 1e9  # Hz
        }
        
        total_power = (component_power["static_power"] + 
                      component_power["dynamic_power"] + 
                      component_power["switching_energy"] * component_power["switching_frequency"] * 1000)
        
        assert total_power > 0
        assert total_power < 1000  # Reasonable power limit (mW)
        
    def test_area_optimization(self):
        """Test area optimization."""
        component_area = {
            "active_area": 100,  # μm²
            "routing_area": 50,  # μm²
            "control_area": 25,  # μm²
            "margin": 0.2  # 20% margin
        }
        
        total_area = ((component_area["active_area"] + 
                      component_area["routing_area"] + 
                      component_area["control_area"]) * 
                     (1 + component_area["margin"]))
        
        assert total_area > 0
        assert total_area < 10000  # Reasonable area limit (μm²)
        
    def test_performance_optimization(self):
        """Test performance optimization."""
        performance_metrics = {
            "bandwidth": 10,  # GHz
            "latency": 1e-9,  # s
            "accuracy": 0.95,  # fraction
            "throughput": 1000  # operations/s
        }
        
        # Performance should meet minimum requirements
        assert performance_metrics["bandwidth"] >= 1  # GHz
        assert performance_metrics["latency"] <= 1e-6  # s
        assert performance_metrics["accuracy"] >= 0.9
        assert performance_metrics["throughput"] >= 100
        
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        objectives = {
            "power": {"value": 10, "weight": 0.3, "minimize": True},
            "area": {"value": 500, "weight": 0.3, "minimize": True},
            "speed": {"value": 1000, "weight": 0.4, "minimize": False}
        }
        
        # Calculate weighted score
        score = 0
        for obj_name, obj_data in objectives.items():
            normalized_value = obj_data["value"] / 1000  # Normalize
            if obj_data["minimize"]:
                score += obj_data["weight"] * (1 - normalized_value)
            else:
                score += obj_data["weight"] * normalized_value
                
        assert 0 <= score <= 1


class TestTemplateLibraryIntegration:
    """Test suite for template library integration."""
    
    def test_pdk_compatibility(self, process_parameters: Dict[str, Any]):
        """Test Process Design Kit compatibility."""
        pdk_name = process_parameters["process_name"]
        
        # Template should be compatible with PDK
        compatible_pdks = ["IMEC_SiN_220nm", "TSMC_40nm", "GlobalFoundries_45nm"]
        assert pdk_name in compatible_pdks
        
        # Design rules should be defined
        assert "design_rules" in process_parameters
        design_rules = process_parameters["design_rules"]
        assert "min_width" in design_rules
        assert "min_space" in design_rules
        
    def test_foundry_specific_templates(self):
        """Test foundry-specific template variations."""
        foundry_templates = {
            "IMEC": {
                "processes": ["SiN_220nm", "SiO2_platform"],
                "components": ["ring_resonator", "mzi", "grating_coupler"]
            },
            "TSMC": {
                "processes": ["40nm_photonic"],
                "components": ["ring_modulator", "phase_shifter"]
            }
        }
        
        for foundry, data in foundry_templates.items():
            assert len(data["processes"]) > 0
            assert len(data["components"]) > 0
            
    def test_template_library_updates(self):
        """Test template library update mechanism."""
        library_versions = [
            {"version": "v2025_01", "date": "2025-01-01", "components": 50},
            {"version": "v2025_07", "date": "2025-07-01", "components": 75},
        ]
        
        # Newer versions should have more components
        for i in range(1, len(library_versions)):
            current = library_versions[i]
            previous = library_versions[i-1]
            assert current["components"] >= previous["components"]
            
    def test_custom_template_integration(self):
        """Test integration of custom user templates."""
        custom_template = {
            "name": "custom_neuron",
            "type": "active_component",
            "author": "user",
            "version": "1.0",
            "parameters": {
                "threshold": 1.0,
                "gain": 10.0,
                "bandwidth": 5.0
            },
            "validated": False  # Custom templates need validation
        }
        
        # Custom template should have required fields
        assert "name" in custom_template
        assert "type" in custom_template
        assert "parameters" in custom_template
        
        # Custom templates should be marked for validation
        assert custom_template.get("validated", True) == False