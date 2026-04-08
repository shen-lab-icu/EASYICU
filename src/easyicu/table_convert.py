"""Table type conversion utilities (R ricu tbl-utils.R).

Provides functions for converting between different table types and configurations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import pandas as pd

from .table import IdTbl, TsTbl, WinTbl, as_id_tbl, as_ts_tbl, as_win_tbl
from .config import TableConfig, IdentifierConfig, DataSourceConfig

def reclass_tbl(x: Union[pd.DataFrame, IdTbl, TsTbl, WinTbl], 
                to: Optional[str] = None) -> Union[pd.DataFrame, IdTbl, TsTbl, WinTbl]:
    """Reclassify table (R ricu reclass_tbl).
    
    Convert table to different class or type.
    
    Args:
        x: Table to reclassify
        to: Target class name ('id_tbl', 'ts_tbl', 'win_tbl', or None for DataFrame)
        
    Returns:
        Reclassified table
        
    Examples:
        >>> reclass_tbl(df, 'id_tbl')
    """
    if to is None:
        # Convert to plain DataFrame
        if isinstance(x, (IdTbl, TsTbl, WinTbl)):
            return x.data
        return x
    
    if to == 'id_tbl':
        if isinstance(x, pd.DataFrame):
            return as_id_tbl(x)
        return x
    
    if to == 'ts_tbl':
        if isinstance(x, pd.DataFrame):
            return as_ts_tbl(x)
        if isinstance(x, IdTbl):
            return as_ts_tbl(x)
        return x
    
    if to == 'win_tbl':
        if isinstance(x, pd.DataFrame):
            # Try to convert via ts_tbl
            ts = as_ts_tbl(x)
            return as_win_tbl(ts)
        if isinstance(x, IdTbl):
            ts = as_ts_tbl(x)
            return as_win_tbl(ts)
        if isinstance(x, TsTbl):
            return as_win_tbl(x)
        return x
    
    raise ValueError(f"Unknown target class: {to}")

def unclass_tbl(x: Union[IdTbl, TsTbl, WinTbl]) -> pd.DataFrame:
    """Unclassify table (R ricu unclass_tbl).
    
    Convert table class to plain DataFrame.
    
    Args:
        x: Table to unclassify
        
    Returns:
        Plain DataFrame
        
    Examples:
        >>> unclass_tbl(id_tbl)
    """
    if isinstance(x, (IdTbl, TsTbl, WinTbl)):
        return x.data.copy()
    return x

def as_col_cfg(x: Any) -> Dict[str, Any]:
    """Convert to column configuration (R ricu as_col_cfg).
    
    Args:
        x: Object to convert
        
    Returns:
        Column configuration dictionary
    """
    if isinstance(x, TableConfig):
        defaults = x.defaults
        return {
            'id_var': defaults.id_var,
            'index_var': defaults.index_var,
            'val_var': defaults.val_var,
            'unit_var': defaults.unit_var,
            'dur_var': defaults.dur_var,
            'time_vars': defaults.time_vars,
        }
    
    if isinstance(x, (IdTbl, TsTbl, WinTbl)):
        from .table_meta import id_vars, index_var, dur_var
        cfg = {}
        id_cols = id_vars(x)
        if id_cols:
            cfg['id_var'] = id_cols[0] if isinstance(id_cols, list) else id_cols
        if isinstance(x, (TsTbl, WinTbl)):
            idx = index_var(x)
            if idx:
                cfg['index_var'] = idx
        if isinstance(x, WinTbl):
            dur = dur_var(x)
            if dur:
                cfg['dur_var'] = dur
        return cfg
    
    return {}

def as_id_cfg(x: Any) -> Dict[str, IdentifierConfig]:
    """Convert to ID configuration (R ricu as_id_cfg).
    
    Args:
        x: Object to convert
        
    Returns:
        Dictionary of identifier configurations
    """
    if isinstance(x, DataSourceConfig):
        return x.id_configs
    
    if hasattr(x, 'config'):
        config = x.config
        if isinstance(config, DataSourceConfig):
            return config.id_configs
    
    return {}

def as_src_cfg(x: Any) -> DataSourceConfig:
    """Convert to source configuration (R ricu as_src_cfg).
    
    Args:
        x: Object to convert
        
    Returns:
        DataSourceConfig object
    """
    if isinstance(x, DataSourceConfig):
        return x
    
    if hasattr(x, 'config'):
        config = x.config
        if isinstance(config, DataSourceConfig):
            return config
    
    # Try to load from registry
    if isinstance(x, str):
        from .resources import load_data_sources
        registry = load_data_sources()
        return registry.get(x)
    
    raise TypeError(f"Cannot convert {type(x)} to DataSourceConfig")

def as_tbl_cfg(x: Any, table_name: Optional[str] = None) -> TableConfig:
    """Convert to table configuration (R ricu as_tbl_cfg).
    
    Args:
        x: Object to convert
        table_name: Optional table name
        
    Returns:
        TableConfig object
    """
    if isinstance(x, TableConfig):
        return x
    
    if isinstance(x, DataSourceConfig):
        if table_name is None:
            raise ValueError("table_name required when x is DataSourceConfig")
        return x.get_table(table_name)
    
    if hasattr(x, 'config'):
        config = x.config
        if isinstance(config, DataSourceConfig):
            if table_name is None:
                # Try to infer from x
                if hasattr(x, 'table_name'):
                    table_name = x.table_name
                else:
                    raise ValueError("Cannot determine table_name")
            return config.get_table(table_name)
    
    raise TypeError(f"Cannot convert {type(x)} to TableConfig")

def as_src_tbl(x: Any, src: Optional[str] = None) -> Any:
    """Convert to source table (R ricu as_src_tbl).
    
    Args:
        x: Object to convert (table name string or table object)
        src: Data source name (if x is string)
        
    Returns:
        Source table representation
        
    Note:
        In Python, this is simplified - we return the table name or object as-is.
    """
    if isinstance(x, str):
        if src:
            from .datasource import ICUDataSource
            from .resources import load_data_sources
            registry = load_data_sources()
            config = registry.get(src)
            if config:
                return ICUDataSource(config)
        return x
    
    return x

def as_ptype(x: Any) -> type:
    """Convert to prototype type (R ricu as_ptype).
    
    Args:
        x: Object to convert
        
    Returns:
        Prototype type (class)
    """
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame
    if isinstance(x, IdTbl):
        return IdTbl
    if isinstance(x, TsTbl):
        return TsTbl
    if isinstance(x, WinTbl):
        return WinTbl
    
    return type(x)

