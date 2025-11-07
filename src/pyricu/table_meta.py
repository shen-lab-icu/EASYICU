"""Table metadata accessors (R ricu tbl-utils.R).

Provides functions for accessing table metadata like ID columns, index columns,
data columns, and time-related metadata.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import pandas as pd

from .table import IdTbl, TsTbl, WinTbl, ICUTable, id_vars, index_var, dur_var, meta_vars, data_vars
from .assertions import assert_that, is_string


def id_var(x: Union[IdTbl, TsTbl, WinTbl, ICUTable]) -> str:
    """Get single ID variable name (R ricu id_var).
    
    Args:
        x: Table object
        
    Returns:
        Single ID variable name as string
        
    Raises:
        AssertionError: If multiple ID variables exist
        
    Examples:
        >>> tbl = IdTbl(pd.DataFrame({'id': [1, 2], 'val': [10, 20]}), id_columns=['id'])
        >>> id_var(tbl)
        'id'
    """
    res = id_vars(x)
    assert_that(is_string(res), msg="Multiple ID variables exist, use id_vars() instead")
    return res


def id_col(x: Union[IdTbl, TsTbl, WinTbl, ICUTable]) -> pd.Series:
    """Get single ID column as Series (R ricu id_col).
    
    Args:
        x: Table object
        
    Returns:
        ID column as pandas Series
        
    Examples:
        >>> tbl = IdTbl(pd.DataFrame({'id': [1, 2], 'val': [10, 20]}), id_columns=['id'])
        >>> id_col(tbl)
        0    1
        1    2
        Name: id, dtype: int64
    """
    return x.data[id_var(x)]


def index_col(x: Union[TsTbl, WinTbl]) -> pd.Series:
    """Get index column as Series (R ricu index_col).
    
    Args:
        x: Time series table object
        
    Returns:
        Index column as pandas Series
        
    Raises:
        TypeError: If object is not a time series table
        
    Examples:
        >>> tbl = TsTbl(...)
        >>> index_col(tbl)
    """
    if not isinstance(x, (TsTbl, WinTbl)):
        raise TypeError(f"index_col requires ts_tbl or win_tbl, got {type(x)}")
    idx_var = index_var(x)
    if idx_var is None:
        raise ValueError("Table has no index variable")
    return x.data[idx_var]


def dur_col(x: WinTbl) -> pd.Series:
    """Get duration column as Series (R ricu dur_col).
    
    Args:
        x: Window table object
        
    Returns:
        Duration column as pandas Series
        
    Raises:
        TypeError: If object is not a window table
        
    Examples:
        >>> tbl = WinTbl(...)
        >>> dur_col(tbl)
    """
    if not isinstance(x, WinTbl):
        raise TypeError(f"dur_col requires win_tbl, got {type(x)}")
    dur = dur_var(x)
    if dur is None:
        raise ValueError("Table has no duration variable")
    return x.data[dur]


def dur_unit(x: WinTbl) -> str:
    """Get duration unit (R ricu dur_unit).
    
    Args:
        x: Window table object
        
    Returns:
        Duration unit as string (e.g., 'minutes', 'hours')
        
    Examples:
        >>> dur_unit(win_tbl)
        'minutes'
    """
    col = dur_col(x)
    if hasattr(col, 'dtype'):
        if pd.api.types.is_timedelta64_dtype(col):
            # Extract unit from timedelta
            return 'minutes'  # Default, could be improved
    return 'minutes'  # Default


def data_var(x: Union[IdTbl, TsTbl, WinTbl, ICUTable]) -> str:
    """Get single data variable name (R ricu data_var).
    
    Args:
        x: Table object
        
    Returns:
        Single data variable name as string
        
    Raises:
        AssertionError: If multiple or no data variables exist
        
    Examples:
        >>> tbl = IdTbl(pd.DataFrame({'id': [1, 2], 'val': [10, 20]}), id_columns=['id'])
        >>> data_var(tbl)
        'val'
    """
    res = data_vars(x)
    assert_that(is_string(res), msg="Multiple data variables exist, use data_vars() instead")
    return res


def data_col(x: Union[IdTbl, TsTbl, WinTbl, ICUTable]) -> pd.Series:
    """Get single data column as Series (R ricu data_col).
    
    Args:
        x: Table object
        
    Returns:
        Data column as pandas Series
        
    Examples:
        >>> tbl = IdTbl(pd.DataFrame({'id': [1, 2], 'val': [10, 20]}), id_columns=['id'])
        >>> data_col(tbl)
        0    10
        1    20
        Name: val, dtype: int64
    """
    return x.data[data_var(x)]


def time_unit(x: Union[TsTbl, WinTbl]) -> str:
    """Get time unit of interval (R ricu time_unit).
    
    Args:
        x: Time series table object
        
    Returns:
        Time unit as string (e.g., 'minutes', 'hours')
        
    Examples:
        >>> time_unit(ts_tbl)
        'minutes'
    """
    interval_obj = interval(x)
    if hasattr(interval_obj, 'unit'):
        return interval_obj.unit
    # Try to infer from timedelta
    if isinstance(interval_obj, pd.Timedelta):
        if interval_obj >= pd.Timedelta(days=1):
            return 'days'
        elif interval_obj >= pd.Timedelta(hours=1):
            return 'hours'
        elif interval_obj >= pd.Timedelta(minutes=1):
            return 'minutes'
        else:
            return 'seconds'
    return 'minutes'  # Default


def time_step(x: Union[TsTbl, WinTbl]) -> float:
    """Get time step size in time_unit (R ricu time_step).
    
    Args:
        x: Time series table object
        
    Returns:
        Time step size as numeric value
        
    Examples:
        >>> time_step(ts_tbl)
        60.0  # 60 minutes
    """
    interval_obj = interval(x)
    if isinstance(interval_obj, pd.Timedelta):
        unit = time_unit(x)
        if unit == 'days':
            return interval_obj.total_seconds() / (24 * 3600)
        elif unit == 'hours':
            return interval_obj.total_seconds() / 3600
        elif unit == 'minutes':
            return interval_obj.total_seconds() / 60
        else:
            return interval_obj.total_seconds()
    return 1.0  # Default


def interval(x: Union[TsTbl, WinTbl]) -> pd.Timedelta:
    """Get time series interval (R ricu interval).
    
    Args:
        x: Time series table object
        
    Returns:
        Time interval as pandas Timedelta
        
    Examples:
        >>> interval(ts_tbl)
        Timedelta('0 days 01:00:00')  # 1 hour
    """
    if not isinstance(x, (TsTbl, WinTbl)):
        raise TypeError(f"interval requires ts_tbl or win_tbl, got {type(x)}")
    
    if hasattr(x, '_interval') and x._interval is not None:
        return x._interval
    
    # Try to infer from index column
    idx_var = index_var(x)
    if idx_var and idx_var in x.data.columns:
        idx_col = x.data[idx_var]
        if pd.api.types.is_timedelta64_dtype(idx_col):
            # Calculate interval from differences
            diffs = idx_col.diff().dropna()
            if len(diffs) > 0:
                positive_diffs = diffs[diffs > pd.Timedelta(0)]
                if len(positive_diffs) > 0:
                    return positive_diffs.min()
    
    # Default: 1 hour
    return pd.Timedelta(hours=1)


def id_var_opts(x: Any) -> list[str]:
    """Get ID variable options (R ricu id_var_opts).
    
    Args:
        x: Configuration object or table
        
    Returns:
        List of available ID variable names
        
    Examples:
        >>> id_var_opts(config)
        ['subject_id', 'hadm_id', 'icustay_id']
    """
    # Try to get from config
    if hasattr(x, 'id_configs'):
        return list(x.id_configs.keys())
    if hasattr(x, 'config'):
        config = x.config
        if hasattr(config, 'id_configs'):
            return list(config.id_configs.keys())
    
    # Try to get from table
    if isinstance(x, (IdTbl, TsTbl, WinTbl)):
        return id_vars(x) if isinstance(id_vars(x), list) else [id_vars(x)]
    
    return []


def default_vars(x: Any) -> dict[str, Any]:
    """Get default variables (R ricu default_vars).
    
    Args:
        x: Configuration object or table
        
    Returns:
        Dictionary of default variable names
        
    Examples:
        >>> default_vars(table_cfg)
        {'id_var': 'icustay_id', 'index_var': 'time', ...}
    """
    from .config import TableConfig
    
    if isinstance(x, TableConfig):
        defaults = x.defaults
        result = {}
        if defaults.id_var:
            result['id_var'] = defaults.id_var
        if defaults.index_var:
            result['index_var'] = defaults.index_var
        if defaults.val_var:
            result['val_var'] = defaults.val_var
        if defaults.unit_var:
            result['unit_var'] = defaults.unit_var
        if defaults.dur_var:
            result['dur_var'] = defaults.dur_var
        if defaults.time_vars:
            result['time_vars'] = defaults.time_vars
        return result
    
    # Try to get from table
    if isinstance(x, (IdTbl, TsTbl, WinTbl, ICUTable)):
        result = {}
        id_cols = id_vars(x)
        if id_cols:
            result['id_var'] = id_cols[0] if isinstance(id_cols, list) else id_cols
        if isinstance(x, (TsTbl, WinTbl)):
            idx_var = index_var(x)
            if idx_var:
                result['index_var'] = idx_var
        if isinstance(x, WinTbl):
            dur = dur_var(x)
            if dur:
                result['dur_var'] = dur
        return result
    
    return {}

