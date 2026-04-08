"""ID mapping system (R ricu data-utils.R).

Provides functions for mapping between different ID systems and getting ID origins.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import pandas as pd

from .table import IdTbl, as_id_tbl
from .src_utils import src_name
from .config import DataSourceConfig
from .datasource import ICUDataSource
from .assertions import assert_that, is_string

# Cache for ID maps
_id_map_cache: Dict[str, pd.DataFrame] = {}
_id_origin_cache: Dict[str, pd.DataFrame] = {}

def id_map(
    x: Union[str, DataSourceConfig, ICUDataSource],
    id_var: str,
    win_var: str,
    in_time: Optional[str] = None,
    out_time: Optional[str] = None,
) -> IdTbl:
    """Create ID mapping between two ID systems (R ricu id_map).
    
    Maps between different ID systems (e.g., hospital admission to ICU stay).
    
    Args:
        x: Data source identifier
        id_var: Source ID type (e.g., 'icustay_id')
        win_var: Target ID type (e.g., 'hadm_id')
        in_time: Optional name for input time column
        out_time: Optional name for output time column
        
    Returns:
        IdTbl with mapping between ID systems
        
    Examples:
        >>> id_map('mimic_demo', 'icustay_id', 'hadm_id')
    """
    from .data_env import get_src_env, as_src_env
    try:
        from .assertions import assert_that, is_string, is_flag
    except ImportError:
        def assert_that(cond, msg=None):
            if not cond:
                raise AssertionError(msg or "Assertion failed")
        def is_string(x):
            return isinstance(x, str)
        def is_flag(x):
            return isinstance(x, bool)
    
    assert_that(is_string(id_var), is_string(win_var))
    
    # Get source environment
    if isinstance(x, str):
        src_env = get_src_env(x) or as_src_env(x)
    else:
        src_env = as_src_env(x)
    
    src = src_name(src_env)
    key = f"{src}:{id_var}_{win_var}"
    
    # Check cache
    if key in _id_map_cache:
        res = _id_map_cache[key].copy()
    else:
        res = id_map_helper(src_env, id_var, win_var)
        _id_map_cache[key] = res.copy()
    
    # Rename time columns if needed
    if in_time or out_time:
        res = res.copy()
        if in_time:
            # Find start column
            start_cols = [c for c in res.columns if '_start' in c or c == 'start']
            if start_cols:
                res = res.rename(columns={start_cols[0]: in_time})
        if out_time:
            # Find end column
            end_cols = [c for c in res.columns if '_end' in c or c == 'end']
            if end_cols:
                res = res.rename(columns={end_cols[0]: out_time})
    
    return as_id_tbl(res, id_vars=id_var)

def id_map_helper(
    x: Union[str, DataSourceConfig, ICUDataSource],
    id_var: str,
    win_var: str,
) -> pd.DataFrame:
    """Helper function for ID mapping (R ricu id_map_helper).
    
    Args:
        x: Data source identifier
        id_var: Source ID type
        win_var: Target ID type
        
    Returns:
        DataFrame with ID mapping
    """
    from .data_env import get_src_env, as_src_env
    from .config import DataSourceConfig
    
    # Get source environment
    if isinstance(x, str):
        src_env = get_src_env(x) or as_src_env(x)
    else:
        src_env = as_src_env(x)
    
    # Get ID windows
    id_wins = id_windows(src_env)
    
    # Get ID configurations
    if isinstance(src_env, DataSourceConfig):
        config = src_env
    elif hasattr(src_env, 'config'):
        config = src_env.config
    else:
        from .resources import load_data_sources
        registry = load_data_sources()
        config = registry.get(src_name(src_env))
    
    if not config:
        raise ValueError(f"Cannot get configuration for {src_name(src_env)}")
    
    id_configs = config.id_configs
    
    # Find mapping
    if id_var not in id_configs or win_var not in id_configs:
        raise ValueError(f"ID types {id_var} or {win_var} not found in configuration")
    
    # Get columns from id_windows
    id_col_name = id_configs[id_var].id
    win_col_name = id_configs[win_var].id
    start_col = f"{win_var}_start"
    end_col = f"{win_var}_end"
    
    # Extract mapping from id_windows
    if all(col in id_wins.columns for col in [id_col_name, win_col_name, start_col, end_col]):
        result = id_wins[[id_col_name, win_col_name, start_col, end_col]].copy()
        result = result.rename(columns={start_col: 'start', end_col: 'end'})
        return result
    else:
        # Fallback: try to construct from tables
        # This is a simplified implementation
        raise NotImplementedError(f"ID mapping from {id_var} to {win_var} requires table lookup")

def id_origin(
    x: Union[str, DataSourceConfig, ICUDataSource],
    id: str,
    origin_name: Optional[str] = None,
    copy: bool = True,
) -> IdTbl:
    """Get ID origin times (R ricu id_origin).
    
    Returns admission/origin timestamps for each ID.
    
    Args:
        x: Data source identifier
        id: ID type name
        origin_name: Optional name for origin column
        copy: Whether to return a copy
        
    Returns:
        IdTbl with ID and origin times
        
    Examples:
        >>> id_origin('mimic_demo', 'icustay_id')
    """
    from .data_env import get_src_env, as_src_env
    try:
        from .assertions import assert_that, is_string, is_flag
    except ImportError:
        def assert_that(cond, msg=None):
            if not cond:
                raise AssertionError(msg or "Assertion failed")
        def is_string(x):
            return isinstance(x, str)
        def is_flag(x):
            return isinstance(x, bool)
    
    assert_that(is_string(id), is_flag(copy))
    
    # Get source environment
    if isinstance(x, str):
        src_env = get_src_env(x) or as_src_env(x)
    else:
        src_env = as_src_env(x)
    
    src = src_name(src_env)
    key = f"{src}:{id}"
    
    # Check cache
    if key in _id_origin_cache:
        res = _id_origin_cache[key]
    else:
        res = id_orig_helper(src_env, id)
        _id_origin_cache[key] = res
    
    # Rename if needed
    if origin_name:
        res = res.copy() if copy else res
        # Find origin column (typically last time column)
        time_cols = [c for c in res.columns if c != id and 
                    pd.api.types.is_datetime64_any_dtype(res[c])]
        if time_cols:
            res = res.rename(columns={time_cols[0]: origin_name})
    elif copy:
        res = res.copy()
    
    return res

def id_orig_helper(
    x: Union[str, DataSourceConfig, ICUDataSource],
    id: str,
) -> IdTbl:
    """Helper function for ID origin (R ricu id_orig_helper).
    
    Args:
        x: Data source identifier
        id: ID type name
        
    Returns:
        IdTbl with ID and origin times
    """
    from .data_env import get_src_env, as_src_env
    from .table_convert import as_id_cfg
    
    assert_that(is_string(id))
    
    # Get source environment
    if isinstance(x, str):
        src_env = get_src_env(x) or as_src_env(x)
    else:
        src_env = as_src_env(x)
    
    # Get ID configuration
    id_cfg = as_id_cfg(src_env)
    if id not in id_cfg:
        raise ValueError(f"ID type '{id}' not found in configuration")
    
    cfg = id_cfg[id]
    
    # Get table
    table_name = cfg.table
    if not table_name:
        raise ValueError(f"No table configured for ID type '{id}'")
    
    # Load table
    if isinstance(src_env, ICUDataSource):
        data_source = src_env
    elif hasattr(src_env, 'load_table'):
        data_source = src_env
    else:
        from .resources import load_data_sources
        registry = load_data_sources()
        config = registry.get(src_name(src_env))
        if not config:
            raise ValueError(f"Cannot load data source {src_name(src_env)}")
        data_source = ICUDataSource(config)
    
    table = data_source.load_table(table_name)
    df = table.data
    
    # Get start column
    start_col = cfg.start
    if start_col and start_col in df.columns:
        result = df[[cfg.id, start_col]].drop_duplicates()
    else:
        # No start column, use ID only with zero time
        result = df[[cfg.id]].drop_duplicates()
        result['origin'] = pd.Timestamp('1970-01-01')  # Default origin
    
    return as_id_tbl(result, id_vars=cfg.id)

def id_windows(
    x: Union[str, DataSourceConfig, ICUDataSource],
    copy: bool = True,
) -> IdTbl:
    """Get ID windows for all ID systems (R ricu id_windows).
    
    Returns a table containing all available ID systems with their time windows.
    
    Args:
        x: Data source identifier
        copy: Whether to return a copy
        
    Returns:
        IdTbl with all ID windows
        
    Examples:
        >>> id_windows('mimic_demo')
    """
    from .data_env import get_src_env, as_src_env
    
    # Get source environment
    if isinstance(x, str):
        src_env = get_src_env(x) or as_src_env(x)
    else:
        src_env = as_src_env(x)
    
    src_name(src_env)
    
    # Check cache (would need to implement)
    res = id_win_helper(src_env)
    
    if copy:
        res = res.copy()
    
    return res

def id_win_helper(
    x: Union[str, DataSourceConfig, ICUDataSource],
) -> IdTbl:
    """Helper function for ID windows (R ricu id_win_helper).
    
    Args:
        x: Data source identifier
        
    Returns:
        IdTbl with ID windows
        
    Note:
        This is a simplified implementation. Full implementation would
        need to handle different data sources (MIMIC, eICU, HiRID, etc.)
        with their specific ID window logic.
    """
    from .data_env import get_src_env, as_src_env
    from .table_convert import as_id_cfg
    
    # Get source environment
    if isinstance(x, str):
        src_env = get_src_env(x) or as_src_env(x)
    else:
        src_env = as_src_env(x)
    
    # Get ID configurations
    id_cfg = as_id_cfg(src_env)
    if not id_cfg:
        raise ValueError("No ID configurations found")
    
    # Sort by position (descending)
    sorted_ids = sorted(id_cfg.items(), key=lambda x: x[1].position, reverse=True)
    
    # Build ID windows table
    # This is a simplified implementation
    # Full implementation would need to merge tables and handle time conversions
    
    result_rows = []
    for id_name, cfg in sorted_ids:
        # Get table
        table_name = cfg.table
        if not table_name:
            continue
        
        # Load table
        if isinstance(src_env, ICUDataSource):
            data_source = src_env
        elif hasattr(src_env, 'load_table'):
            data_source = src_env
        else:
            from .resources import load_data_sources
            registry = load_data_sources()
            config = registry.get(src_name(src_env))
            if not config:
                continue
            data_source = ICUDataSource(config)
        
        try:
            table = data_source.load_table(table_name)
            df = table.data
            
            # Get start and end columns
            start_col = cfg.start if cfg.start in df.columns else None
            end_col = cfg.end if cfg.end in df.columns else None
            
            # Extract ID windows
            id_col = cfg.id
            if id_col in df.columns:
                if start_col and end_col:
                    windows = df[[id_col, start_col, end_col]].drop_duplicates()
                    windows = windows.rename(columns={
                        start_col: f"{id_name}_start",
                        end_col: f"{id_name}_end"
                    })
                elif start_col:
                    windows = df[[id_col, start_col]].drop_duplicates()
                    windows = windows.rename(columns={start_col: f"{id_name}_start"})
                    windows[f"{id_name}_end"] = windows[f"{id_name}_start"]
                else:
                    windows = df[[id_col]].drop_duplicates()
                    windows[f"{id_name}_start"] = pd.Timedelta(0)
                    windows[f"{id_name}_end"] = pd.Timedelta(0)
                
                result_rows.append(windows)
        except Exception:
            # Skip if table not available
            continue
    
    # Merge all ID windows
    if not result_rows:
        raise ValueError("No ID windows found")
    
    # Start with first ID
    result = result_rows[0]
    for window_df in result_rows[1:]:
        # Merge on common ID columns
        common_cols = set(result.columns) & set(window_df.columns)
        if common_cols:
            result = result.merge(window_df, on=list(common_cols), how='outer')
        else:
            # Cartesian product (shouldn't happen in practice)
            result = result.merge(window_df, how='cross')
    
    # Use first ID as primary
    first_id = sorted_ids[0][1].id
    return as_id_tbl(result, id_vars=first_id)

def as_src_env(x: Any) -> Any:
    """Convert to source environment (R ricu as_src_env).
    
    Args:
        x: Object to convert
        
    Returns:
        Source environment object
    """
    from .data_env import get_src_env
    
    if isinstance(x, str):
        return get_src_env(x)
    
    # If already a source environment, return as-is
    if hasattr(x, 'name') and hasattr(x, 'load_table'):
        return x
    
    # Try to convert from config
    from .table_convert import as_src_cfg
    try:
        config = as_src_cfg(x)
        from .datasource import ICUDataSource
        return ICUDataSource(config)
    except Exception:
        pass
    
    raise TypeError(f"Cannot convert {type(x)} to source environment")

