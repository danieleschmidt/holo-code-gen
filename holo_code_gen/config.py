"""Configuration management system for Holo-Code-Gen."""

import os
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "~/.holo_code_gen/designs.db"
    cache_dir: str = "~/.holo_code_gen/cache"
    models_dir: str = "~/.holo_code_gen/models"
    circuits_dir: str = "~/.holo_code_gen/circuits"
    max_cache_size_mb: int = 1000
    cache_ttl_seconds: int = 3600
    cleanup_interval_seconds: int = 300


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    backend: str = "meep"
    threads: int = 4
    memory_limit: str = "8GB"
    gpu_enabled: bool = False
    resolution: int = 20
    convergence_threshold: float = 1e-6
    max_iterations: int = 10000
    timeout_seconds: int = 3600
    default_wavelength: float = 1550.0
    default_process: str = "SiN_220nm"


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    default_power_budget_mw: float = 1000.0
    default_area_budget_mm2: float = 100.0
    iterations: int = 1000
    convergence_tolerance: float = 1e-6
    enable_parallel: bool = True
    max_parallel_optimizations: int = 4
    max_parallel_simulations: int = 2


@dataclass
class TemplateConfig:
    """Template library configuration."""
    default_library_version: str = "imec_v2025_07"
    custom_templates_dir: str = "~/.holo_code_gen/custom_templates"
    validation_enabled: bool = True
    imec_license_key: Optional[str] = None
    imec_auto_update: bool = False
    strict_mode: bool = False


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_encryption: bool = False
    encryption_key_file: str = "~/.holo_code_gen/keys/encryption.key"
    enable_access_control: bool = False
    max_file_size_mb: int = 1000
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".pth", ".onnx", ".json", ".pkl", ".gds", ".spi"
    ])
    safe_mode: bool = True
    allow_custom_code: bool = False
    sandbox_execution: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: str = "~/.holo_code_gen/logs/holo_code_gen.log"
    max_size_mb: int = 10
    backup_count: int = 5
    enable_debug: bool = False
    audit_log: bool = True
    audit_log_path: str = "~/.holo_code_gen/logs/audit.log"


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_memory_usage_mb: int = 8192
    max_cpu_cores: int = 8
    max_disk_usage_gb: int = 50
    max_concurrent_compilations: int = 4
    parallel_jobs: Union[str, int] = "auto"
    max_workers: int = 8
    lazy_loading: bool = True
    garbage_collection: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 4
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class BackupConfig:
    """Backup configuration."""
    enable_auto_backup: bool = True
    interval_hours: int = 24
    retention_days: int = 30
    backup_directory: str = "~/.holo_code_gen/backups"


@dataclass
class HoloCodeGenConfig:
    """Main configuration class."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    
    # Development settings
    development_mode: bool = False
    debug: bool = False
    profile: bool = False
    verbose: bool = False
    test_mode: bool = False
    
    @classmethod
    def from_env(cls) -> 'HoloCodeGenConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Database configuration
        config.database.path = os.getenv('HOLO_DATABASE_PATH', config.database.path)
        config.database.cache_dir = os.getenv('HOLO_CACHE_DIR', config.database.cache_dir)
        config.database.models_dir = os.getenv('HOLO_MODELS_DIR', config.database.models_dir)
        config.database.circuits_dir = os.getenv('HOLO_CIRCUITS_DIR', config.database.circuits_dir)
        config.database.max_cache_size_mb = int(os.getenv('HOLO_MAX_CACHE_SIZE_MB', config.database.max_cache_size_mb))
        config.database.cache_ttl_seconds = int(os.getenv('HOLO_CACHE_TTL_SECONDS', config.database.cache_ttl_seconds))
        config.database.cleanup_interval_seconds = int(os.getenv('HOLO_CLEANUP_INTERVAL_SECONDS', config.database.cleanup_interval_seconds))
        
        # Simulation configuration  
        config.simulation.backend = os.getenv('HOLO_CODE_GEN_SIMULATION_BACKEND', config.simulation.backend)
        config.simulation.threads = int(os.getenv('HOLO_CODE_GEN_SIMULATION_THREADS', config.simulation.threads))
        config.simulation.memory_limit = os.getenv('HOLO_CODE_GEN_SIMULATION_MEMORY_LIMIT', config.simulation.memory_limit)
        config.simulation.gpu_enabled = os.getenv('HOLO_CODE_GEN_SIMULATION_GPU_ENABLED', '').lower() == 'true'
        config.simulation.resolution = int(os.getenv('HOLO_CODE_GEN_SIMULATION_RESOLUTION', config.simulation.resolution))
        config.simulation.convergence_threshold = float(os.getenv('HOLO_CODE_GEN_SIMULATION_CONVERGENCE_THRESHOLD', config.simulation.convergence_threshold))
        config.simulation.max_iterations = int(os.getenv('HOLO_CODE_GEN_SIMULATION_MAX_ITERATIONS', config.simulation.max_iterations))
        config.simulation.timeout_seconds = int(os.getenv('HOLO_SIMULATION_TIMEOUT_SECONDS', config.simulation.timeout_seconds))
        config.simulation.default_wavelength = float(os.getenv('HOLO_DEFAULT_WAVELENGTH', config.simulation.default_wavelength))
        config.simulation.default_process = os.getenv('HOLO_DEFAULT_PROCESS', config.simulation.default_process)
        
        # Optimization configuration
        config.optimization.default_power_budget_mw = float(os.getenv('HOLO_DEFAULT_POWER_BUDGET_MW', config.optimization.default_power_budget_mw))
        config.optimization.default_area_budget_mm2 = float(os.getenv('HOLO_DEFAULT_AREA_BUDGET_MM2', config.optimization.default_area_budget_mm2))
        config.optimization.iterations = int(os.getenv('HOLO_OPTIMIZATION_ITERATIONS', config.optimization.iterations))
        config.optimization.convergence_tolerance = float(os.getenv('HOLO_CONVERGENCE_TOLERANCE', config.optimization.convergence_tolerance))
        config.optimization.enable_parallel = os.getenv('HOLO_ENABLE_PARALLEL_OPTIMIZATION', '').lower() == 'true'
        config.optimization.max_parallel_optimizations = int(os.getenv('HOLO_MAX_PARALLEL_OPTIMIZATIONS', config.optimization.max_parallel_optimizations))
        config.optimization.max_parallel_simulations = int(os.getenv('HOLO_MAX_PARALLEL_SIMULATIONS', config.optimization.max_parallel_simulations))
        
        # Template configuration
        config.templates.default_library_version = os.getenv('HOLO_DEFAULT_LIBRARY_VERSION', config.templates.default_library_version)
        config.templates.custom_templates_dir = os.getenv('HOLO_CUSTOM_TEMPLATES_DIR', config.templates.custom_templates_dir)
        config.templates.validation_enabled = os.getenv('HOLO_TEMPLATE_VALIDATION_ENABLED', '').lower() != 'false'
        config.templates.imec_license_key = os.getenv('HOLO_CODE_GEN_IMEC_LICENSE_KEY')
        config.templates.imec_auto_update = os.getenv('HOLO_CODE_GEN_IMEC_AUTO_UPDATE', '').lower() == 'true'
        config.templates.strict_mode = os.getenv('HOLO_CODE_GEN_STRICT_TEMPLATE_MODE', '').lower() == 'true'
        
        # Security configuration
        config.security.enable_encryption = os.getenv('HOLO_ENABLE_ENCRYPTION', '').lower() == 'true'
        config.security.encryption_key_file = os.getenv('HOLO_ENCRYPTION_KEY_FILE', config.security.encryption_key_file)
        config.security.enable_access_control = os.getenv('HOLO_ENABLE_ACCESS_CONTROL', '').lower() == 'true'
        config.security.max_file_size_mb = int(os.getenv('HOLO_MAX_FILE_SIZE_MB', config.security.max_file_size_mb))
        config.security.safe_mode = os.getenv('HOLO_CODE_GEN_SAFE_MODE', '').lower() != 'false'
        config.security.allow_custom_code = os.getenv('HOLO_CODE_GEN_ALLOW_CUSTOM_CODE', '').lower() == 'true'
        config.security.sandbox_execution = os.getenv('HOLO_CODE_GEN_SANDBOX_EXECUTION', '').lower() != 'false'
        
        # Logging configuration
        config.logging.level = os.getenv('HOLO_LOG_LEVEL', config.logging.level)
        config.logging.log_file = os.getenv('HOLO_LOG_FILE', config.logging.log_file)
        config.logging.max_size_mb = int(os.getenv('HOLO_LOG_MAX_SIZE_MB', config.logging.max_size_mb))
        config.logging.backup_count = int(os.getenv('HOLO_LOG_BACKUP_COUNT', config.logging.backup_count))
        config.logging.enable_debug = os.getenv('HOLO_ENABLE_DEBUG_LOGGING', '').lower() == 'true'
        config.logging.audit_log = os.getenv('HOLO_CODE_GEN_AUDIT_LOG', '').lower() != 'false'
        config.logging.audit_log_path = os.getenv('HOLO_CODE_GEN_AUDIT_LOG_PATH', config.logging.audit_log_path)
        
        # Performance configuration
        config.performance.max_memory_usage_mb = int(os.getenv('HOLO_MAX_MEMORY_USAGE_MB', config.performance.max_memory_usage_mb))
        config.performance.max_cpu_cores = int(os.getenv('HOLO_MAX_CPU_CORES', config.performance.max_cpu_cores))
        config.performance.max_disk_usage_gb = int(os.getenv('HOLO_MAX_DISK_USAGE_GB', config.performance.max_disk_usage_gb))
        config.performance.max_concurrent_compilations = int(os.getenv('HOLO_MAX_CONCURRENT_COMPILATIONS', config.performance.max_concurrent_compilations))
        config.performance.max_workers = int(os.getenv('HOLO_CODE_GEN_MAX_WORKERS', config.performance.max_workers))
        config.performance.lazy_loading = os.getenv('HOLO_CODE_GEN_LAZY_LOADING', '').lower() != 'false'
        config.performance.garbage_collection = os.getenv('HOLO_CODE_GEN_GARBAGE_COLLECTION', '').lower() != 'false'
        
        # API configuration
        config.api.host = os.getenv('HOLO_API_HOST', config.api.host)
        config.api.port = int(os.getenv('HOLO_API_PORT', config.api.port))
        config.api.workers = int(os.getenv('HOLO_API_WORKERS', config.api.workers))
        config.api.enable_cors = os.getenv('HOLO_ENABLE_CORS', '').lower() != 'false'
        
        # Backup configuration
        config.backup.enable_auto_backup = os.getenv('HOLO_ENABLE_AUTO_BACKUP', '').lower() != 'false'
        config.backup.interval_hours = int(os.getenv('HOLO_BACKUP_INTERVAL_HOURS', config.backup.interval_hours))
        config.backup.retention_days = int(os.getenv('HOLO_BACKUP_RETENTION_DAYS', config.backup.retention_days))
        config.backup.backup_directory = os.getenv('HOLO_BACKUP_DIRECTORY', config.backup.backup_directory)
        
        # Development settings
        config.development_mode = os.getenv('HOLO_DEVELOPMENT_MODE', '').lower() == 'true'
        config.debug = os.getenv('HOLO_CODE_GEN_DEBUG', '').lower() == 'true'
        config.profile = os.getenv('HOLO_CODE_GEN_PROFILE', '').lower() == 'true'
        config.verbose = os.getenv('HOLO_CODE_GEN_VERBOSE', '').lower() == 'true'
        config.test_mode = os.getenv('HOLO_TEST_MODE', '').lower() == 'true'
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'HoloCodeGenConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            config = cls()
            
            # Apply configuration from file
            for section_name, section_data in config_data.items():
                if hasattr(config, section_name):
                    section = getattr(config, section_name)
                    if hasattr(section, '__dataclass_fields__'):
                        # Update dataclass fields
                        for field_name, value in section_data.items():
                            if hasattr(section, field_name):
                                setattr(section, field_name, value)
                    else:
                        # Direct attribute
                        setattr(config, section_name, section_data)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_path}: {e}")
            return cls()
    
    def to_file(self, config_path: Union[str, Path]) -> bool:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = {}
            
            # Convert dataclass sections to dictionaries
            for field_name in self.__dataclass_fields__:
                field_value = getattr(self, field_name)
                if hasattr(field_value, '__dataclass_fields__'):
                    config_dict[field_name] = {
                        sub_field: getattr(field_value, sub_field)
                        for sub_field in field_value.__dataclass_fields__
                    }
                else:
                    config_dict[field_name] = field_value
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            return False
    
    def expand_paths(self) -> None:
        """Expand all path-like configuration values."""
        def expand_path(path_str: str) -> str:
            """Expand ~ and environment variables in path."""
            if path_str.startswith('~'):
                return str(Path(path_str).expanduser().resolve())
            return str(Path(os.path.expandvars(path_str)).resolve())
        
        # Expand database paths
        self.database.path = expand_path(self.database.path)
        self.database.cache_dir = expand_path(self.database.cache_dir)
        self.database.models_dir = expand_path(self.database.models_dir)
        self.database.circuits_dir = expand_path(self.database.circuits_dir)
        
        # Expand template paths
        self.templates.custom_templates_dir = expand_path(self.templates.custom_templates_dir)
        
        # Expand security paths
        if self.security.encryption_key_file:
            self.security.encryption_key_file = expand_path(self.security.encryption_key_file)
        
        # Expand logging paths
        self.logging.log_file = expand_path(self.logging.log_file)
        self.logging.audit_log_path = expand_path(self.logging.audit_log_path)
        
        # Expand backup paths
        self.backup.backup_directory = expand_path(self.backup.backup_directory)
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.database.cache_dir,
            self.database.models_dir,
            self.database.circuits_dir,
            self.templates.custom_templates_dir,
            self.backup.backup_directory,
            Path(self.logging.log_file).parent,
            Path(self.logging.audit_log_path).parent
        ]
        
        if self.security.encryption_key_file:
            directories.append(Path(self.security.encryption_key_file).parent)
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            except Exception as e:
                logger.warning(f"Failed to create directory {directory}: {e}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate paths exist or can be created
        required_dirs = [
            self.database.cache_dir,
            self.database.models_dir,
            self.database.circuits_dir
        ]
        
        for directory in required_dirs:
            path = Path(directory)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {directory}: {e}")
        
        # Validate numeric ranges
        if self.simulation.threads <= 0:
            issues.append("Simulation threads must be positive")
        
        if self.optimization.iterations <= 0:
            issues.append("Optimization iterations must be positive")
        
        if self.performance.max_memory_usage_mb <= 0:
            issues.append("Max memory usage must be positive")
        
        # Validate file extensions
        if not all(ext.startswith('.') for ext in self.security.allowed_file_extensions):
            issues.append("File extensions must start with '.'")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level.upper() not in valid_log_levels:
            issues.append(f"Invalid log level: {self.logging.level}")
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'database_path': self.database.path,
            'simulation_backend': self.simulation.backend,
            'template_library': self.templates.default_library_version,
            'debug_mode': self.debug,
            'development_mode': self.development_mode,
            'test_mode': self.test_mode,
            'security_enabled': self.security.safe_mode,
            'cache_size_mb': self.database.max_cache_size_mb,
            'max_workers': self.performance.max_workers,
            'api_host': self.api.host,
            'api_port': self.api.port
        }


# Global configuration instance
_config: Optional[HoloCodeGenConfig] = None


def get_config() -> HoloCodeGenConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = HoloCodeGenConfig.from_env()
        _config.expand_paths()
        _config.create_directories()
        
        # Validate configuration
        issues = _config.validate()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
    
    return _config


def set_config(config: HoloCodeGenConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset global configuration instance."""
    global _config
    _config = None