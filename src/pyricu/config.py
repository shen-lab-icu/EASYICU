"""Configuration models for ICU data sources.

The R :mod:`ricu` package organises access to heterogeneous datasets through
JSON configuration files that describe identifier columns, default table
metadata, and loader specific hints.  The classes below provide a Pythonic
representation of the same ideas with validation via :mod:`pydantic`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import json

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class IdentifierConfig(BaseModel):
    """Describe one identifier system (e.g. patient, admission, icu stay)."""

    name: str
    id: str = Field(..., alias="id")
    position: int = Field(..., alias="position")
    start: Optional[str] = None
    end: Optional[str] = None
    table: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class TableDefaults(BaseModel):
    """Default metadata that describes how to interpret a table."""

    id_var: Optional[str] = None
    index_var: Optional[str] = None
    val_var: Optional[str] = None
    unit_var: Optional[str] = None
    dur_var: Optional[str] = None
    time_vars: List[str] = Field(default_factory=list, alias="time_vars")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("time_vars", mode="before")
    def _ensure_list(cls, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(item) for item in value]
        raise TypeError("time_vars must be a string or an iterable of strings")


class DatasetOptions(BaseModel):
    """Optional declaration describing a PyArrow Dataset layout for a table."""

    path: Optional[str] = None
    format: str = "parquet"
    partitioning: Optional[str] = None
    options: Dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class TableConfig(BaseModel):
    """Configuration for one logical table within a data source."""

    name: str
    defaults: TableDefaults = Field(default_factory=TableDefaults)
    files: List[Mapping[str, Any]] = Field(default_factory=list)
    num_rows: Optional[int] = Field(default=None, alias="num_rows")
    columns: Optional[Mapping[str, object]] = Field(default=None, alias="cols")
    extra: Dict[str, object] = Field(default_factory=dict)
    dataset: Optional[DatasetOptions] = None

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def _extract_known_fields(cls, values: Mapping[str, object]) -> Mapping[str, object]:
        values = dict(values)
        known = {"defaults", "files", "num_rows", "cols", "dataset"}
        extra = {k: v for k, v in values.items() if k not in known and k != "name"}
        values["extra"] = extra
        defaults = values.get("defaults")
        if defaults in (None, [], ()):
            values["defaults"] = {}
        values["files"] = cls._coerce_files(values.get("files"))
        dataset_payload = values.get("dataset")
        if dataset_payload:
            if isinstance(dataset_payload, DatasetOptions):
                values["dataset"] = dataset_payload
            elif isinstance(dataset_payload, Mapping):
                values["dataset"] = DatasetOptions(**dataset_payload)
            else:
                raise TypeError("dataset must be a mapping describing dataset options")
        return values

    @staticmethod
    def _coerce_files(raw: object) -> List[Mapping[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            normalised: List[Mapping[str, Any]] = []
            for entry in raw:
                if isinstance(entry, Mapping):
                    normalised.append(dict(entry))
                elif isinstance(entry, str):
                    normalised.append({"path": entry})
                else:
                    raise TypeError("files entries must be mappings or strings")
            return normalised
        if isinstance(raw, Mapping):
            return [dict(raw)]
        if isinstance(raw, str):
            return [{"path": raw}]
        raise TypeError("files must be string, mapping, or list of them")

    def first_file(self) -> Optional[str]:
        """Return the first file entry if one is defined."""
        if not self.files:
            return None
        file_entry = self.files[0]
        path = file_entry.get("path") or file_entry.get("file")
        return path


class DataSourceConfig(BaseModel):
    """Complete configuration for a data source (e.g. ``mimic_demo``)."""

    name: str
    class_prefix: List[str] = Field(default_factory=list, alias="class_prefix")
    id_configs: Dict[str, IdentifierConfig] = Field(default_factory=dict)
    tables: Dict[str, TableConfig] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    def _normalise_payload(cls, values: Mapping[str, object]) -> Mapping[str, object]:
        # Convert id_cfg and tables dictionaries into strongly typed objects.
        id_cfg_raw = values.get("id_cfg", {})
        tables_raw = values.get("tables", {})

        id_cfg = {
            key: IdentifierConfig(name=key, **cfg)
            for key, cfg in id_cfg_raw.items()
        }

        tables = {
            key: TableConfig(name=key, **cfg)
            for key, cfg in tables_raw.items()
        }

        transformed = dict(values)
        transformed["id_configs"] = id_cfg
        transformed["tables"] = tables
        return transformed

    def list_tables(self) -> List[str]:
        return sorted(self.tables.keys())

    def get_table(self, name: str) -> TableConfig:
        try:
            return self.tables[name]
        except KeyError as error:
            raise KeyError(f"{self.name} has no table named '{name}'") from error

    def get_default_id(self) -> Optional[str]:
        if not self.id_configs:
            return None
        # Choose identifier with smallest position (higher priority).
        sorted_ids = sorted(self.id_configs.values(), key=lambda cfg: cfg.position)
        return sorted_ids[0].id

    def id_options(self) -> Dict[str, IdentifierConfig]:
        return dict(self.id_configs)


class DataSourceRegistry:
    """Container holding multiple :class:`DataSourceConfig` objects."""

    def __init__(self, configs: Iterable[DataSourceConfig] | None = None) -> None:
        self._configs: Dict[str, DataSourceConfig] = {}
        if configs:
            for cfg in configs:
                self.register(cfg)

    def register(self, config: DataSourceConfig) -> None:
        self._configs[config.name] = config

    def get(self, name: str) -> DataSourceConfig:
        try:
            return self._configs[name]
        except KeyError as error:
            raise KeyError(f"Unknown data source '{name}'") from error

    def __contains__(self, name: object) -> bool:
        return name in self._configs

    def __iter__(self) -> Iterator[DataSourceConfig]:
        yield from self._configs.values()

    @classmethod
    def from_payload(cls, payload: Iterable[Mapping[str, object]]) -> "DataSourceRegistry":
        configs = [DataSourceConfig(**entry) for entry in payload]
        return cls(configs=configs)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "DataSourceRegistry":
        path = Path(file_path)
        with path.open("r", encoding="utf8") as handle:
            payload = json.load(handle)
        return cls.from_payload(payload)


# ============================================================================
# Global configuration management (R ricu config-utils.R)
# ============================================================================

class GlobalConfig:
    """Global configuration manager (R ricu .ricu_env).
    
    Manages runtime configuration options similar to R's .ricu_env.
    Provides get/set methods for configuration options.
    
    Attributes:
        _config: Dictionary of configuration options
        
    Examples:
        >>> config = GlobalConfig()
        >>> config.set('data_dir', '/path/to/data')
        >>> config.get('data_dir')
        '/path/to/data'
    """
    
    def __init__(self):
        """Initialize with default configuration."""
        self._config = {
            'data_dir': None,
            'cache_dir': None,
            'download_timeout': 300,  # seconds
            'verbose': True,
            'max_workers': 4,
            'chunk_size': 10000,
            'progress_bar': True,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value (R ricu get_config).
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Examples:
            >>> global_config.get('data_dir')
            '/path/to/data'
            >>> global_config.get('unknown_key', 'default')
            'default'
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (R ricu set_config).
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Examples:
            >>> global_config.set('data_dir', '/new/path')
            >>> global_config.set('verbose', False)
        """
        self._config[key] = value
    
    def update(self, **kwargs) -> None:
        """Update multiple configuration values (R ricu set_config).
        
        Args:
            **kwargs: Key-value pairs to update
            
        Examples:
            >>> global_config.update(data_dir='/path', verbose=True)
        """
        self._config.update(kwargs)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.
        
        Returns:
            Dictionary of all configuration options
        """
        return dict(self._config)
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self.__init__()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        items = [f"{k}={repr(v)}" for k, v in self._config.items()]
        return f"GlobalConfig({', '.join(items)})"


# Global configuration instance
global_config = GlobalConfig()


def get_config(key: str, default: Any = None) -> Any:
    """Get global configuration value (R ricu get_config).
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value
        
    Examples:
        >>> get_config('data_dir')
        '/path/to/data'
        >>> get_config('unknown', 'default')
        'default'
    """
    return global_config.get(key, default)


def set_config(**kwargs) -> None:
    """Set global configuration values (R ricu set_config).
    
    Args:
        **kwargs: Configuration key-value pairs
        
    Examples:
        >>> set_config(data_dir='/path/to/data')
        >>> set_config(verbose=True, max_workers=8)
    """
    global_config.update(**kwargs)


def reset_config() -> None:
    """Reset configuration to defaults (R ricu reset_config).
    
    Examples:
        >>> reset_config()
    """
    global_config.reset()


def list_config() -> Dict[str, Any]:
    """List all configuration options (R ricu list_config).
    
    Returns:
        Dictionary of all configuration options
        
    Examples:
        >>> config = list_config()
        >>> print(config['data_dir'])
    """
    return global_config.get_all()


def load_src_cfg(src_name: str) -> DataSourceConfig:
    """
    加载数据源配置 - 对应 R ricu load_src_cfg
    
    Args:
        src_name: 数据源名称 (如 'mimic_demo', 'eicu_demo', 'eicu', 'miiv')
        
    Returns:
        DataSourceConfig 对象
        
    Examples:
        >>> cfg = load_src_cfg('mimic_demo')
        >>> print(cfg.name)
        'mimic_demo'
        >>> cfg = load_src_cfg('eicu')
        >>> 'vitalperiodic' in cfg.tables
        True
    """
    # 使用 load_data_sources 加载数据源注册表
    from .resources import load_data_sources
    registry = load_data_sources()
    
    try:
        return registry.get(src_name)
    except KeyError:
        # 如果数据源不存在，创建一个基本配置
        # 注意: id_cfg 和 tables 必须是字典
        return DataSourceConfig(
            name=src_name,
            id_cfg={},
            tables={}
        )


# ============================================================================
# Persistent Configuration Storage (R ricu config persistence)
# ============================================================================

import os
from pathlib import Path
import json
from typing import Any, Dict, Optional


def get_config_dir() -> Path:
    """Get pyricu configuration directory (R ricu config_paths).
    
    Returns directory for storing persistent configuration.
    Default: ~/.pyricu/config
    
    Returns:
        Path to config directory
    """
    # Try XDG_CONFIG_HOME first (Linux standard)
    xdg_config = os.environ.get('XDG_CONFIG_HOME')
    if xdg_config:
        config_dir = Path(xdg_config) / 'pyricu'
    else:
        # Fall back to home directory
        config_dir = Path.home() / '.pyricu' / 'config'
    
    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    
    return config_dir


def get_config_file(name: str = 'pyricu') -> Path:
    """Get path to persistent configuration file (R ricu get_config_file).
    
    Args:
        name: Configuration file name (without extension)
        
    Returns:
        Path to configuration JSON file
    """
    return get_config_dir() / f'{name}.json'


def save_config(config: Optional[Dict[str, Any]] = None, 
                name: str = 'pyricu') -> None:
    """Save configuration to persistent storage (R ricu save_config).
    
    Args:
        config: Configuration dictionary (if None, uses current global config)
        name: Configuration file name
        
    Examples:
        >>> save_config({'data_dir': '/path/to/data', 'verbose': True})
        >>> save_config()  # Save current global config
    """
    if config is None:
        config = global_config.get_all()
    
    config_file = get_config_file(name)
    
    # Convert Path objects to strings for JSON serialization
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(config_file, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration saved to {config_file}")


def load_config(name: str = 'pyricu', 
                merge: bool = True) -> Optional[Dict[str, Any]]:
    """Load configuration from persistent storage (R ricu load_config).
    
    Args:
        name: Configuration file name
        merge: If True, merge with current config; if False, replace
        
    Returns:
        Configuration dictionary (or None if file doesn't exist)
        
    Examples:
        >>> config = load_config()
        >>> print(config['data_dir'])
    """
    config_file = get_config_file(name)
    
    if not config_file.exists():
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Convert string paths back to Path objects
    for key in ['data_dir', 'cache_dir']:
        if key in config and isinstance(config[key], str):
            config[key] = Path(config[key])
    
    if merge:
        # Merge with existing config
        global_config.update(**config)
    
    return config


def delete_config(name: str = 'pyricu') -> None:
    """Delete persistent configuration file (R ricu delete_config).
    
    Args:
        name: Configuration file name
        
    Examples:
        >>> delete_config()
    """
    config_file = get_config_file(name)
    
    if config_file.exists():
        config_file.unlink()
        print(f"Configuration file deleted: {config_file}")
    else:
        print(f"Configuration file not found: {config_file}")


def list_config_files() -> List[str]:
    """List all saved configuration files (R ricu list_config_files).
    
    Returns:
        List of configuration file names (without extension)
        
    Examples:
        >>> files = list_config_files()
        >>> print(files)
        ['pyricu', 'custom']
    """
    config_dir = get_config_dir()
    
    if not config_dir.exists():
        return []
    
    config_files = list(config_dir.glob('*.json'))
    return [f.stem for f in config_files]


def export_config(config: Dict[str, Any], 
                  output_file: Path) -> None:
    """Export configuration to a custom location (R ricu export_config).
    
    Args:
        config: Configuration dictionary
        output_file: Output file path
        
    Examples:
        >>> export_config(global_config.get_all(), Path('my_config.json'))
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration exported to {output_file}")


def import_config(input_file: Path, 
                  merge: bool = True) -> Dict[str, Any]:
    """Import configuration from a custom location (R ricu import_config).
    
    Args:
        input_file: Input file path
        merge: If True, merge with current config; if False, replace
        
    Returns:
        Loaded configuration dictionary
        
    Examples:
        >>> config = import_config(Path('my_config.json'))
    """
    input_file = Path(input_file)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {input_file}")
    
    with open(input_file, 'r') as f:
        config = json.load(f)
    
    # Convert string paths back to Path objects
    for key in ['data_dir', 'cache_dir']:
        if key in config and isinstance(config[key], str):
            config[key] = Path(config[key])
    
    if merge:
        global_config.update(**config)
    
    return config


# ============================================================================
# Configuration validation and migration
# ============================================================================

def validate_config(config: Optional[Dict[str, Any]] = None) -> bool:
    """Validate configuration (R ricu validate_config).
    
    Checks if configuration values are valid and paths exist.
    
    Args:
        config: Configuration dictionary (if None, uses current global config)
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> if validate_config():
        ...     print("Configuration is valid")
    """
    if config is None:
        config = global_config.get_all()
    
    valid = True
    
    # Check data_dir exists if specified
    if 'data_dir' in config and config['data_dir']:
        data_dir = Path(config['data_dir'])
        if not data_dir.exists():
            print(f"Warning: data_dir does not exist: {data_dir}")
            valid = False
    
    # Check cache_dir exists if specified
    if 'cache_dir' in config and config['cache_dir']:
        cache_dir = Path(config['cache_dir'])
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Cannot create cache_dir: {cache_dir}: {e}")
                valid = False
    
    # Check numeric values are valid
    if 'max_workers' in config:
        if not isinstance(config['max_workers'], int) or config['max_workers'] < 1:
            print(f"Warning: max_workers must be a positive integer: {config['max_workers']}")
            valid = False
    
    return valid


def migrate_config(old_version: str = '0.1', 
                   new_version: str = '0.2') -> Dict[str, Any]:
    """Migrate configuration between versions (R ricu migrate_config).
    
    Handles configuration format changes between versions.
    
    Args:
        old_version: Old configuration version
        new_version: New configuration version
        
    Returns:
        Migrated configuration dictionary
    """
    config = global_config.get_all()
    
    # Add version tracking
    config['_version'] = new_version
    
    # Perform version-specific migrations
    # (Add logic here as configuration format evolves)
    
    return config


# ============================================================================
# Auto-load configuration on import
# ============================================================================

def _auto_load_config():
    """Automatically load saved configuration when module is imported."""
    try:
        saved_config = load_config(merge=True)
        if saved_config:
            pass  # Config already merged
    except Exception:
        pass  # Silently ignore errors during auto-load


# Auto-load on import
_auto_load_config()
