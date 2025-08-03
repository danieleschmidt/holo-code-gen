"""Data models for design database and persistence."""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CircuitModel:
    """Data model for photonic circuit."""
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    neural_network_source: str = ""  # Original neural network model
    compilation_config: Dict[str, Any] = None
    components: List[Dict[str, Any]] = None
    connections: List[Dict[str, Any]] = None
    layout_data: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.compilation_config is None:
            self.compilation_config = {}
        if self.components is None:
            self.components = []
        if self.connections is None:
            self.connections = []
        if self.layout_data is None:
            self.layout_data = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class OptimizationResult:
    """Data model for optimization results."""
    id: Optional[str] = None
    circuit_id: str = ""
    optimizer_type: str = ""
    parameters: Dict[str, Any] = None
    initial_metrics: Dict[str, float] = None
    final_metrics: Dict[str, float] = None
    optimization_log: List[Dict[str, Any]] = None
    execution_time: float = 0.0
    success: bool = False
    error_message: str = ""
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parameters is None:
            self.parameters = {}
        if self.initial_metrics is None:
            self.initial_metrics = {}
        if self.final_metrics is None:
            self.final_metrics = {}
        if self.optimization_log is None:
            self.optimization_log = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SimulationResult:
    """Data model for simulation results."""
    id: Optional[str] = None
    circuit_id: str = ""
    simulation_type: str = ""  # FDTD, thermal, etc.
    parameters: Dict[str, Any] = None
    results: Dict[str, Any] = None
    simulation_time: float = 0.0
    convergence_achieved: bool = False
    error_message: str = ""
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parameters is None:
            self.parameters = {}
        if self.results is None:
            self.results = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class DesignDatabase:
    """SQLite database for storing photonic circuit designs and results."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize design database.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = Path.home() / ".holo_code_gen" / "designs.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS circuits (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    neural_network_source TEXT,
                    compilation_config TEXT,
                    components TEXT,
                    connections TEXT,
                    layout_data TEXT,
                    performance_metrics TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id TEXT PRIMARY KEY,
                    circuit_id TEXT,
                    optimizer_type TEXT,
                    parameters TEXT,
                    initial_metrics TEXT,
                    final_metrics TEXT,
                    optimization_log TEXT,
                    execution_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (circuit_id) REFERENCES circuits (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_results (
                    id TEXT PRIMARY KEY,
                    circuit_id TEXT,
                    simulation_type TEXT,
                    parameters TEXT,
                    results TEXT,
                    simulation_time REAL,
                    convergence_achieved BOOLEAN,
                    error_message TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (circuit_id) REFERENCES circuits (id)
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_circuits_name ON circuits (name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_optimization_circuit ON optimization_results (circuit_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_simulation_circuit ON simulation_results (circuit_id)")
            
            conn.commit()
    
    def save_circuit(self, circuit: CircuitModel) -> str:
        """Save circuit to database."""
        if circuit.id is None:
            import uuid
            circuit.id = str(uuid.uuid4())
        
        circuit.updated_at = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO circuits 
                (id, name, description, neural_network_source, compilation_config,
                 components, connections, layout_data, performance_metrics,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                circuit.id,
                circuit.name,
                circuit.description,
                circuit.neural_network_source,
                json.dumps(circuit.compilation_config),
                json.dumps(circuit.components),
                json.dumps(circuit.connections),
                json.dumps(circuit.layout_data),
                json.dumps(circuit.performance_metrics),
                circuit.created_at.isoformat(),
                circuit.updated_at.isoformat()
            ))
            conn.commit()
        
        logger.debug(f"Saved circuit: {circuit.name} ({circuit.id})")
        return circuit.id
    
    def load_circuit(self, circuit_id: str) -> Optional[CircuitModel]:
        """Load circuit from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM circuits WHERE id = ?
            """, (circuit_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return CircuitModel(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                neural_network_source=row['neural_network_source'],
                compilation_config=json.loads(row['compilation_config']),
                components=json.loads(row['components']),
                connections=json.loads(row['connections']),
                layout_data=json.loads(row['layout_data']),
                performance_metrics=json.loads(row['performance_metrics']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
    
    def list_circuits(self, limit: int = 100, offset: int = 0) -> List[CircuitModel]:
        """List circuits in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM circuits 
                ORDER BY updated_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            circuits = []
            for row in cursor.fetchall():
                circuit = CircuitModel(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    neural_network_source=row['neural_network_source'],
                    compilation_config=json.loads(row['compilation_config']),
                    components=json.loads(row['components']),
                    connections=json.loads(row['connections']),
                    layout_data=json.loads(row['layout_data']),
                    performance_metrics=json.loads(row['performance_metrics']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                circuits.append(circuit)
            
            return circuits
    
    def delete_circuit(self, circuit_id: str) -> bool:
        """Delete circuit and associated results."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete associated optimization results
            conn.execute("DELETE FROM optimization_results WHERE circuit_id = ?", (circuit_id,))
            
            # Delete associated simulation results
            conn.execute("DELETE FROM simulation_results WHERE circuit_id = ?", (circuit_id,))
            
            # Delete circuit
            cursor = conn.execute("DELETE FROM circuits WHERE id = ?", (circuit_id,))
            
            conn.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted circuit: {circuit_id}")
            
            return deleted
    
    def save_optimization_result(self, result: OptimizationResult) -> str:
        """Save optimization result to database."""
        if result.id is None:
            import uuid
            result.id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO optimization_results
                (id, circuit_id, optimizer_type, parameters, initial_metrics,
                 final_metrics, optimization_log, execution_time, success,
                 error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.id,
                result.circuit_id,
                result.optimizer_type,
                json.dumps(result.parameters),
                json.dumps(result.initial_metrics),
                json.dumps(result.final_metrics),
                json.dumps(result.optimization_log),
                result.execution_time,
                result.success,
                result.error_message,
                result.created_at.isoformat()
            ))
            conn.commit()
        
        logger.debug(f"Saved optimization result: {result.optimizer_type} ({result.id})")
        return result.id
    
    def load_optimization_results(self, circuit_id: str) -> List[OptimizationResult]:
        """Load optimization results for a circuit."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM optimization_results 
                WHERE circuit_id = ?
                ORDER BY created_at DESC
            """, (circuit_id,))
            
            results = []
            for row in cursor.fetchall():
                result = OptimizationResult(
                    id=row['id'],
                    circuit_id=row['circuit_id'],
                    optimizer_type=row['optimizer_type'],
                    parameters=json.loads(row['parameters']),
                    initial_metrics=json.loads(row['initial_metrics']),
                    final_metrics=json.loads(row['final_metrics']),
                    optimization_log=json.loads(row['optimization_log']),
                    execution_time=row['execution_time'],
                    success=bool(row['success']),
                    error_message=row['error_message'],
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                results.append(result)
            
            return results
    
    def save_simulation_result(self, result: SimulationResult) -> str:
        """Save simulation result to database."""
        if result.id is None:
            import uuid
            result.id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO simulation_results
                (id, circuit_id, simulation_type, parameters, results,
                 simulation_time, convergence_achieved, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.id,
                result.circuit_id,
                result.simulation_type,
                json.dumps(result.parameters),
                json.dumps(result.results),
                result.simulation_time,
                result.convergence_achieved,
                result.error_message,
                result.created_at.isoformat()
            ))
            conn.commit()
        
        logger.debug(f"Saved simulation result: {result.simulation_type} ({result.id})")
        return result.id
    
    def load_simulation_results(self, circuit_id: str) -> List[SimulationResult]:
        """Load simulation results for a circuit."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM simulation_results 
                WHERE circuit_id = ?
                ORDER BY created_at DESC
            """, (circuit_id,))
            
            results = []
            for row in cursor.fetchall():
                result = SimulationResult(
                    id=row['id'],
                    circuit_id=row['circuit_id'],
                    simulation_type=row['simulation_type'],
                    parameters=json.loads(row['parameters']),
                    results=json.loads(row['results']),
                    simulation_time=row['simulation_time'],
                    convergence_achieved=bool(row['convergence_achieved']),
                    error_message=row['error_message'],
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                results.append(result)
            
            return results
    
    def search_circuits(self, query: str, limit: int = 50) -> List[CircuitModel]:
        """Search circuits by name or description."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM circuits 
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY updated_at DESC 
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))
            
            circuits = []
            for row in cursor.fetchall():
                circuit = CircuitModel(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    neural_network_source=row['neural_network_source'],
                    compilation_config=json.loads(row['compilation_config']),
                    components=json.loads(row['components']),
                    connections=json.loads(row['connections']),
                    layout_data=json.loads(row['layout_data']),
                    performance_metrics=json.loads(row['performance_metrics']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                circuits.append(circuit)
            
            return circuits
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count circuits
            cursor = conn.execute("SELECT COUNT(*) FROM circuits")
            stats['total_circuits'] = cursor.fetchone()[0]
            
            # Count optimization results
            cursor = conn.execute("SELECT COUNT(*) FROM optimization_results")
            stats['total_optimizations'] = cursor.fetchone()[0]
            
            # Count simulation results
            cursor = conn.execute("SELECT COUNT(*) FROM simulation_results")
            stats['total_simulations'] = cursor.fetchone()[0]
            
            # Database file size
            stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            # Recent activity
            cursor = conn.execute("""
                SELECT COUNT(*) FROM circuits 
                WHERE created_at >= datetime('now', '-7 days')
            """)
            stats['circuits_last_week'] = cursor.fetchone()[0]
            
            return stats
    
    def export_circuit(self, circuit_id: str, export_path: Path) -> bool:
        """Export circuit and associated data to JSON file."""
        circuit = self.load_circuit(circuit_id)
        if circuit is None:
            return False
        
        # Load associated results
        optimization_results = self.load_optimization_results(circuit_id)
        simulation_results = self.load_simulation_results(circuit_id)
        
        export_data = {
            'circuit': asdict(circuit),
            'optimization_results': [asdict(result) for result in optimization_results],
            'simulation_results': [asdict(result) for result in simulation_results],
            'export_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported circuit {circuit.name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export circuit: {e}")
            return False
    
    def import_circuit(self, import_path: Path) -> Optional[str]:
        """Import circuit from JSON file."""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Import circuit
            circuit_data = import_data['circuit']
            
            # Convert datetime strings back to datetime objects
            if 'created_at' in circuit_data and isinstance(circuit_data['created_at'], str):
                circuit_data['created_at'] = datetime.fromisoformat(circuit_data['created_at'])
            if 'updated_at' in circuit_data and isinstance(circuit_data['updated_at'], str):
                circuit_data['updated_at'] = datetime.fromisoformat(circuit_data['updated_at'])
                
            circuit = CircuitModel(**circuit_data)
            circuit.id = None  # Generate new ID
            circuit_id = self.save_circuit(circuit)
            
            # Import optimization results
            for result_data in import_data.get('optimization_results', []):
                # Convert datetime strings back to datetime objects
                if 'created_at' in result_data and isinstance(result_data['created_at'], str):
                    result_data['created_at'] = datetime.fromisoformat(result_data['created_at'])
                    
                result = OptimizationResult(**result_data)
                result.id = None  # Generate new ID
                result.circuit_id = circuit_id
                self.save_optimization_result(result)
            
            # Import simulation results
            for result_data in import_data.get('simulation_results', []):
                # Convert datetime strings back to datetime objects
                if 'created_at' in result_data and isinstance(result_data['created_at'], str):
                    result_data['created_at'] = datetime.fromisoformat(result_data['created_at'])
                    
                result = SimulationResult(**result_data)
                result.id = None  # Generate new ID
                result.circuit_id = circuit_id
                self.save_simulation_result(result)
            
            logger.info(f"Imported circuit {circuit.name} from {import_path}")
            return circuit_id
            
        except Exception as e:
            logger.error(f"Failed to import circuit: {e}")
            return None