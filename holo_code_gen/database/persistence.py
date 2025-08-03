"""Persistence layer for neural network models and photonic circuits."""

import json
import pickle
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handles persistence of neural network models and metadata."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize model persistence.
        
        Args:
            storage_dir: Directory for model storage
        """
        self.storage_dir = storage_dir or Path.home() / ".holo_code_gen" / "models"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / "pytorch").mkdir(exist_ok=True)
        (self.storage_dir / "onnx").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
    
    def save_pytorch_model(self, model: torch.nn.Module, 
                          model_name: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save PyTorch model with metadata.
        
        Args:
            model: PyTorch model to save
            model_name: Name for the model
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in model_name if c.isalnum() or c in "._-")
        filename = f"{safe_name}_{timestamp}.pth"
        
        model_path = self.storage_dir / "pytorch" / filename
        metadata_path = self.storage_dir / "metadata" / f"{filename}.json"
        
        try:
            # Save model
            torch.save(model.state_dict(), model_path)
            
            # Prepare metadata
            model_metadata = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "model_class": f"{type(model).__module__}.{type(model).__name__}",
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "model_size_mb": model_path.stat().st_size / (1024 * 1024),
                "saved_at": datetime.now().isoformat(),
                "framework": "pytorch",
                "framework_version": torch.__version__,
                "architecture": self._extract_architecture(model),
                "input_shape": self._infer_input_shape(model),
                "output_shape": self._infer_output_shape(model)
            }
            
            # Add user metadata
            if metadata:
                model_metadata.update(metadata)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)
            
            logger.info(f"Saved PyTorch model: {model_name}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save PyTorch model: {e}")
            # Cleanup partial files
            model_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
    
    def load_pytorch_model(self, model_path: str, 
                          model_class: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """Load PyTorch model from file.
        
        Args:
            model_path: Path to model file
            model_class: Model class for loading state dict
            
        Returns:
            Loaded PyTorch model
        """
        model_path = Path(model_path)
        metadata_path = self.storage_dir / "metadata" / f"{model_path.name}.json"
        
        try:
            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Load model
            if model_class is not None:
                # Load into provided model class
                model = model_class
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                # Try to load complete model (if saved with torch.save(model))
                try:
                    model = torch.load(model_path, map_location='cpu')
                except Exception:
                    # If that fails, we need the model class
                    raise ValueError("Model class required for loading state dict")
            
            logger.info(f"Loaded PyTorch model: {metadata.get('model_name', 'unknown')}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def save_onnx_model(self, model: torch.nn.Module,
                       model_name: str,
                       input_shape: tuple,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            model_name: Name for the model
            input_shape: Input tensor shape for export
            metadata: Additional metadata
            
        Returns:
            Path to exported ONNX model
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in model_name if c.isalnum() or c in "._-")
        filename = f"{safe_name}_{timestamp}.onnx"
        
        onnx_path = self.storage_dir / "onnx" / filename
        metadata_path = self.storage_dir / "metadata" / f"{filename}.json"
        
        try:
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Prepare metadata
            onnx_metadata = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "input_shape": list(input_shape),
                "model_size_mb": onnx_path.stat().st_size / (1024 * 1024),
                "exported_at": datetime.now().isoformat(),
                "framework": "onnx",
                "source_framework": "pytorch",
                "opset_version": 11
            }
            
            # Add user metadata
            if metadata:
                onnx_metadata.update(metadata)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(onnx_metadata, f, indent=2, default=str)
            
            logger.info(f"Exported ONNX model: {model_name}")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            # Cleanup partial files
            onnx_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
    
    def list_models(self, framework: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models.
        
        Args:
            framework: Filter by framework (pytorch, onnx)
            
        Returns:
            List of model metadata
        """
        models = []
        metadata_dir = self.storage_dir / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Filter by framework if specified
                if framework and metadata.get('framework') != framework:
                    continue
                
                # Add file path
                model_file = metadata_file.name.replace('.json', '')
                if metadata.get('framework') == 'pytorch':
                    metadata['file_path'] = str(self.storage_dir / "pytorch" / model_file)
                elif metadata.get('framework') == 'onnx':
                    metadata['file_path'] = str(self.storage_dir / "onnx" / model_file)
                
                models.append(metadata)
                
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        # Sort by saved date
        models.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        return models
    
    def delete_model(self, model_path: str) -> bool:
        """Delete model and associated metadata.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        model_path = Path(model_path)
        metadata_path = self.storage_dir / "metadata" / f"{model_path.name}.json"
        
        try:
            # Delete model file
            if model_path.exists():
                model_path.unlink()
            
            # Delete metadata
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Deleted model: {model_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def _extract_architecture(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extract model architecture information."""
        architecture = {
            "layers": [],
            "total_layers": 0,
            "layer_types": {}
        }
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_type = type(module).__name__
                layer_info = {
                    "name": name,
                    "type": layer_type,
                    "parameters": sum(p.numel() for p in module.parameters())
                }
                
                # Add type-specific information
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    layer_info.update({
                        "input_size": module.in_features,
                        "output_size": module.out_features
                    })
                elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    layer_info.update({
                        "input_channels": module.in_channels,
                        "output_channels": module.out_channels,
                        "kernel_size": getattr(module, 'kernel_size', None),
                        "stride": getattr(module, 'stride', None),
                        "padding": getattr(module, 'padding', None)
                    })
                
                architecture["layers"].append(layer_info)
                architecture["layer_types"][layer_type] = architecture["layer_types"].get(layer_type, 0) + 1
        
        architecture["total_layers"] = len(architecture["layers"])
        return architecture
    
    def _infer_input_shape(self, model: torch.nn.Module) -> Optional[List[int]]:
        """Infer input shape from model architecture."""
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                return [module.in_features]
            elif isinstance(module, torch.nn.Conv2d):
                return [module.in_channels, None, None]  # Height/width unknown
        return None
    
    def _infer_output_shape(self, model: torch.nn.Module) -> Optional[List[int]]:
        """Infer output shape from model architecture."""
        # Find the last linear or conv layer
        last_linear = None
        last_conv = None
        
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
            elif isinstance(module, torch.nn.Conv2d):
                last_conv = module
        
        if last_linear:
            return [last_linear.out_features]
        elif last_conv:
            return [last_conv.out_channels, None, None]
        
        return None


class CircuitPersistence:
    """Handles persistence of photonic circuits and layouts."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize circuit persistence.
        
        Args:
            storage_dir: Directory for circuit storage
        """
        self.storage_dir = storage_dir or Path.home() / ".holo_code_gen" / "circuits"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / "layouts").mkdir(exist_ok=True)
        (self.storage_dir / "gds").mkdir(exist_ok=True)
        (self.storage_dir / "netlists").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
    
    def save_circuit(self, circuit: 'PhotonicCircuit', 
                    circuit_name: str,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save photonic circuit with all associated data.
        
        Args:
            circuit: PhotonicCircuit to save
            circuit_name: Name for the circuit
            metadata: Additional metadata
            
        Returns:
            Circuit ID
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in circuit_name if c.isalnum() or c in "._-")
        circuit_id = f"{safe_name}_{timestamp}"
        
        # Save circuit data
        circuit_data = {
            "circuit_id": circuit_id,
            "circuit_name": circuit_name,
            "components": [self._serialize_component(comp) for comp in circuit.components],
            "connections": circuit.connections,
            "layout_constraints": circuit.layout_constraints.__dict__,
            "physical_layout": circuit.physical_layout,
            "metrics": circuit.metrics.__dict__ if circuit.metrics else None,
            "config": circuit.config.__dict__ if circuit.config else None,
            "saved_at": datetime.now().isoformat()
        }
        
        # Add user metadata
        if metadata:
            circuit_data.update(metadata)
        
        # Save to pickle file for complete data
        circuit_file = self.storage_dir / f"{circuit_id}.pkl"
        with open(circuit_file, 'wb') as f:
            pickle.dump(circuit_data, f)
        
        # Save human-readable metadata
        metadata_file = self.storage_dir / "metadata" / f"{circuit_id}.json"
        readable_metadata = {
            "circuit_id": circuit_id,
            "circuit_name": circuit_name,
            "num_components": len(circuit.components),
            "num_connections": len(circuit.connections),
            "layout_generated": circuit.layout_generated,
            "total_power": circuit.metrics.total_power if circuit.metrics else None,
            "total_area": circuit.metrics.total_area if circuit.metrics else None,
            "saved_at": circuit_data["saved_at"]
        }
        
        if metadata:
            readable_metadata.update({k: v for k, v in metadata.items() 
                                    if isinstance(v, (str, int, float, bool, type(None)))})
        
        with open(metadata_file, 'w') as f:
            json.dump(readable_metadata, f, indent=2, default=str)
        
        logger.info(f"Saved circuit: {circuit_name} ({circuit_id})")
        return circuit_id
    
    def load_circuit(self, circuit_id: str) -> Optional['PhotonicCircuit']:
        """Load photonic circuit from storage.
        
        Args:
            circuit_id: Circuit ID to load
            
        Returns:
            Loaded PhotonicCircuit or None
        """
        circuit_file = self.storage_dir / f"{circuit_id}.pkl"
        
        if not circuit_file.exists():
            logger.warning(f"Circuit file not found: {circuit_id}")
            return None
        
        try:
            with open(circuit_file, 'rb') as f:
                circuit_data = pickle.load(f)
            
            # Reconstruct circuit
            from ..circuit import PhotonicCircuit
            from ..templates import PhotonicComponent, ComponentSpec
            
            circuit = PhotonicCircuit()
            circuit.connections = circuit_data["connections"]
            circuit.layout_generated = circuit_data.get("layout_generated", False)
            circuit.physical_layout = circuit_data.get("physical_layout")
            
            # Reconstruct components
            for comp_data in circuit_data["components"]:
                component = self._deserialize_component(comp_data)
                circuit.components.append(component)
            
            # Reconstruct layout constraints
            if "layout_constraints" in circuit_data:
                from ..circuit import LayoutConstraints
                constraint_data = circuit_data["layout_constraints"]
                circuit.layout_constraints = LayoutConstraints(**constraint_data)
            
            # Reconstruct metrics
            if circuit_data.get("metrics"):
                from ..circuit import CircuitMetrics
                circuit.metrics = CircuitMetrics(**circuit_data["metrics"])
            
            logger.info(f"Loaded circuit: {circuit_data['circuit_name']} ({circuit_id})")
            return circuit
            
        except Exception as e:
            logger.error(f"Failed to load circuit {circuit_id}: {e}")
            return None
    
    def save_gds_layout(self, circuit_id: str, gds_data: bytes) -> str:
        """Save GDS layout data.
        
        Args:
            circuit_id: Circuit ID
            gds_data: GDS file data
            
        Returns:
            Path to GDS file
        """
        gds_file = self.storage_dir / "gds" / f"{circuit_id}.gds"
        
        with open(gds_file, 'wb') as f:
            f.write(gds_data)
        
        logger.info(f"Saved GDS layout: {circuit_id}")
        return str(gds_file)
    
    def save_netlist(self, circuit_id: str, netlist_data: str, format: str = "spice") -> str:
        """Save circuit netlist.
        
        Args:
            circuit_id: Circuit ID
            netlist_data: Netlist content
            format: Netlist format (spice, verilog, etc.)
            
        Returns:
            Path to netlist file
        """
        extension = {"spice": "spi", "verilog": "v", "cdl": "cdl"}.get(format, "txt")
        netlist_file = self.storage_dir / "netlists" / f"{circuit_id}.{extension}"
        
        with open(netlist_file, 'w') as f:
            f.write(netlist_data)
        
        logger.info(f"Saved {format} netlist: {circuit_id}")
        return str(netlist_file)
    
    def list_circuits(self) -> List[Dict[str, Any]]:
        """List available circuits."""
        circuits = []
        metadata_dir = self.storage_dir / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                circuits.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load circuit metadata: {e}")
        
        # Sort by saved date
        circuits.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        return circuits
    
    def delete_circuit(self, circuit_id: str) -> bool:
        """Delete circuit and all associated files.
        
        Args:
            circuit_id: Circuit ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Delete main circuit file
            circuit_file = self.storage_dir / f"{circuit_id}.pkl"
            circuit_file.unlink(missing_ok=True)
            
            # Delete metadata
            metadata_file = self.storage_dir / "metadata" / f"{circuit_id}.json"
            metadata_file.unlink(missing_ok=True)
            
            # Delete GDS file
            gds_file = self.storage_dir / "gds" / f"{circuit_id}.gds"
            gds_file.unlink(missing_ok=True)
            
            # Delete netlist files
            for netlist_file in (self.storage_dir / "netlists").glob(f"{circuit_id}.*"):
                netlist_file.unlink(missing_ok=True)
            
            logger.info(f"Deleted circuit: {circuit_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete circuit {circuit_id}: {e}")
            return False
    
    def _serialize_component(self, component: 'PhotonicComponent') -> Dict[str, Any]:
        """Serialize photonic component for storage."""
        return {
            "spec": {
                "name": component.spec.name,
                "component_type": component.spec.component_type,
                "parameters": component.spec.parameters,
                "constraints": component.spec.constraints,
                "performance": component.spec.performance,
                "layout_info": component.spec.layout_info
            },
            "instance_id": component.instance_id,
            "position": component.position,
            "orientation": component.orientation,
            "ports": component.ports,
            "connections": component.connections
        }
    
    def _deserialize_component(self, comp_data: Dict[str, Any]) -> 'PhotonicComponent':
        """Deserialize photonic component from storage."""
        from ..templates import PhotonicComponent, ComponentSpec
        
        spec = ComponentSpec(
            name=comp_data["spec"]["name"],
            component_type=comp_data["spec"]["component_type"],
            parameters=comp_data["spec"]["parameters"],
            constraints=comp_data["spec"]["constraints"],
            performance=comp_data["spec"]["performance"],
            layout_info=comp_data["spec"]["layout_info"]
        )
        
        component = PhotonicComponent(
            spec=spec,
            instance_id=comp_data["instance_id"],
            position=comp_data["position"],
            orientation=comp_data["orientation"],
            ports=comp_data["ports"],
            connections=comp_data["connections"]
        )
        
        return component