"""Data source utility functions (R ricu config-utils.R).

Provides functions for accessing data source metadata and checking availability.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from pathlib import Path

from .config import DataSourceConfig, DataSourceRegistry
from .datasource import ICUDataSource
try:
    from .data_env import SrcEnv, DataEnv, get_src_env
except ImportError:
    SrcEnv = None
    DataEnv = None
    def get_src_env(name):
        return None

def src_name(x: Union[str, DataSourceConfig, ICUDataSource, SrcEnv, DataEnv]) -> str:
    """Get data source name (R ricu src_name).
    
    Args:
        x: Data source identifier (name string, config, datasource, or env)
        
    Returns:
        Data source name string
        
    Examples:
        >>> src_name('mimic_demo')
        'mimic_demo'
        >>> src_name(data_source)
        'mimic_demo'
    """
    if isinstance(x, str):
        return x
    if isinstance(x, DataSourceConfig):
        return x.name
    if isinstance(x, ICUDataSource):
        return x.config.name
    if SrcEnv is not None and isinstance(x, SrcEnv):
        return x.name
    if DataEnv is not None and isinstance(x, DataEnv):
        return x.name
    
    # Try to get from registry
    try:
        registry = DataSourceRegistry.get_default()
        if isinstance(x, str):
            config = registry.get(x)
            if config:
                return config.name
    except Exception:
        pass
    
    raise TypeError(f"Cannot extract source name from {type(x)}")

def src_prefix(x: Union[str, DataSourceConfig, ICUDataSource, SrcEnv]) -> list[str]:
    """Get data source class prefix (R ricu src_prefix).
    
    Args:
        x: Data source identifier
        
    Returns:
        List of class prefix strings
        
    Examples:
        >>> src_prefix('mimic_demo')
        ['mimic']
    """
    if isinstance(x, DataSourceConfig):
        return x.class_prefix
    if isinstance(x, ICUDataSource):
        return x.config.class_prefix
    if SrcEnv is not None and isinstance(x, SrcEnv):
        if hasattr(x, 'config') and x.config:
            return x.config.class_prefix
        # Try to get from registry
        try:
            registry = DataSourceRegistry.get_default()
            config = registry.get(x.name)
            if config:
                return config.class_prefix
        except Exception:
            pass
    
    # Try to get from registry
    if isinstance(x, str):
        try:
            registry = DataSourceRegistry.get_default()
            config = registry.get(x)
            if config:
                return config.class_prefix
        except Exception:
            pass
    
    return []

def src_extra_cfg(x: Union[str, DataSourceConfig, ICUDataSource, SrcEnv]) -> Dict[str, Any]:
    """Get extra configuration for data source (R ricu src_extra_cfg).
    
    Args:
        x: Data source identifier
        
    Returns:
        Dictionary of extra configuration
        
    Examples:
        >>> src_extra_cfg('mimic_demo')
        {}
    """
    if isinstance(x, DataSourceConfig):
        # DataSourceConfig doesn't have extra_cfg, return empty dict
        return {}
    if isinstance(x, ICUDataSource):
        return {}
    if SrcEnv is not None and isinstance(x, SrcEnv):
        if hasattr(x, 'config') and x.config:
            return {}
    
    return {}

def src_data_avail(src: str, data_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if data source data is available (R ricu src_data_avail).
    
    Args:
        src: Data source name
        data_dir: Optional data directory to check
        
    Returns:
        True if data is available, False otherwise
        
    Examples:
        >>> src_data_avail('mimic_demo', '/path/to/data')
        True
    """
    from .file_utils import src_data_dir, dir_exists
    
    try:
        if data_dir is None:
            data_dir = src_data_dir(src)
        else:
            data_dir = Path(data_dir)
        
        return dir_exists(data_dir)
    except Exception:
        return False

def src_tbl_avail(src: str, tbl: str, data_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if a table is available for a data source (R ricu src_tbl_avail).
    
    Args:
        src: Data source name
        tbl: Table name
        data_dir: Optional data directory to check
        
    Returns:
        True if table is available, False otherwise
        
    Examples:
        >>> src_tbl_avail('mimic_demo', 'patients')
        True
    """
    from .file_utils import src_data_dir, file_exists
    from .config import DataSourceRegistry
    
    try:
        # Get config
        registry = DataSourceRegistry.get_default()
        config = registry.get(src)
        if not config:
            return False
        
        table_cfg = config.get_table(tbl)
        
        # Check if files exist
        if data_dir is None:
            data_dir = src_data_dir(src)
        else:
            data_dir = Path(data_dir)
        
        for file_entry in table_cfg.files:
            file_path = file_entry.get('path') or file_entry.get('file')
            if file_path:
                full_path = data_dir / file_path
                if file_exists(full_path):
                    return True
        
        return False
    except Exception:
        return False

def is_data_avail(src: str, data_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if data is available (R ricu is_data_avail).
    
    Alias for src_data_avail.
    
    Args:
        src: Data source name
        data_dir: Optional data directory to check
        
    Returns:
        True if data is available, False otherwise
    """
    return src_data_avail(src, data_dir)

def is_tbl_avail(src: str, tbl: str, data_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if a table is available (R ricu is_tbl_avail).
    
    Alias for src_tbl_avail.
    
    Args:
        src: Data source name
        tbl: Table name
        data_dir: Optional data directory to check
        
    Returns:
        True if table is available, False otherwise
    """
    return src_tbl_avail(src, tbl, data_dir)

def is_src_tbl(x: Any) -> bool:
    """Check if object is a source table (R ricu is_src_tbl).
    
    Args:
        x: Object to check
        
    Returns:
        True if object is a source table, False otherwise
        
    Note:
        This is a simplified check. In R, src_tbl is a specific class.
        In Python, we check if it's a table name string or has table-like attributes.
    """
    # Check if it's a string (table name)
    if isinstance(x, str):
        return True
    
    # Check if it has table-like attributes
    if hasattr(x, 'table_name') or hasattr(x, 'name'):
        return True
    
    # Check if it's an ICUDataSource with table info
    from .datasource import ICUDataSource
    if isinstance(x, ICUDataSource):
        return True
    
    return False

