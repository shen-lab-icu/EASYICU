"""Data manipulation tools (R ricu data-utils.R, tbl-utils.R).

Provides utility functions for data manipulation.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import pandas as pd

from .table import IdTbl, TsTbl, WinTbl
from .ts_utils import has_gaps

def unmerge(tbl: Union[IdTbl, TsTbl, WinTbl, pd.DataFrame]) -> pd.DataFrame:
    """Unmerge table (R ricu unmerge).
    
    Inverse operation of merging - splits merged data back into separate tables.
    
    Args:
        tbl: Table to unmerge
        
    Returns:
        Unmerged DataFrame(s)
        
    Note:
        This is a simplified implementation. Full unmerge would require
        tracking merge history.
    """
    if isinstance(tbl, (IdTbl, TsTbl, WinTbl)):
        return tbl.data.copy()
    return tbl.copy()

def rm_na(tbl: Union[IdTbl, TsTbl, WinTbl, pd.DataFrame], 
          columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Remove NA values (R ricu rm_na).
    
    Args:
        tbl: Table to clean
        columns: Optional list of columns to check for NA
        
    Returns:
        DataFrame with NA rows removed
        
    Examples:
        >>> rm_na(df, columns=['value'])
    """
    if isinstance(tbl, (IdTbl, TsTbl, WinTbl)):
        df = tbl.data.copy()
    else:
        df = tbl.copy()
    
    if columns:
        df = df.dropna(subset=columns)
    else:
        df = df.dropna()
    
    return df

def change_dur_unit(
    tbl: WinTbl,
    unit: str = 'minutes'
) -> WinTbl:
    """Change duration unit (R ricu change_dur_unit).
    
    Args:
        tbl: Window table
        unit: Target unit ('minutes', 'hours', 'days', 'seconds')
        
    Returns:
        WinTbl with duration in new unit
        
    Examples:
        >>> change_dur_unit(win_tbl, 'hours')
    """
    from .table_meta import dur_var, dur_col
    
    dur_v = dur_var(tbl)
    if dur_v is None:
        return tbl
    
    dur_c = dur_col(tbl)
    
    # Convert to target unit
    if pd.api.types.is_timedelta64_dtype(dur_c):
        if unit == 'minutes':
            new_dur = dur_c / pd.Timedelta(minutes=1)
        elif unit == 'hours':
            new_dur = dur_c / pd.Timedelta(hours=1)
        elif unit == 'days':
            new_dur = dur_c / pd.Timedelta(days=1)
        elif unit == 'seconds':
            new_dur = dur_c / pd.Timedelta(seconds=1)
        else:
            raise ValueError(f"Unknown unit: {unit}")
        
        new_data = tbl.data.copy()
        new_data[dur_v] = new_dur
    else:
        # Already numeric, assume minutes
        if unit == 'hours':
            new_data = tbl.data.copy()
            new_data[dur_v] = new_data[dur_v] / 60
        elif unit == 'days':
            new_data = tbl.data.copy()
            new_data[dur_v] = new_data[dur_v] / (60 * 24)
        elif unit == 'seconds':
            new_data = tbl.data.copy()
            new_data[dur_v] = new_data[dur_v] * 60
        else:
            new_data = tbl.data.copy()
    
    return WinTbl(
        data=new_data,
        id_vars=tbl.id_vars,
        index_var=tbl.index_var if isinstance(tbl, WinTbl) else None,
        dur_var=dur_v,
        interval=tbl.interval if hasattr(tbl, 'interval') else None
    )

def has_no_gaps(tbl: Union[TsTbl, WinTbl]) -> bool:
    """Check if table has no gaps (R ricu has_no_gaps).
    
    Args:
        tbl: Time series table
        
    Returns:
        True if table has no gaps, False otherwise
        
    Examples:
        >>> has_no_gaps(ts_tbl)
        True
    """
    if not isinstance(tbl, (TsTbl, WinTbl)):
        raise TypeError("has_no_gaps requires ts_tbl or win_tbl")
    
    return not has_gaps(tbl)

def load_src_cfg(src: str) -> Any:
    """Load source configuration (R ricu load_src_cfg).
    
    Args:
        src: Source name
        
    Returns:
        DataSourceConfig object
        
    Examples:
        >>> cfg = load_src_cfg('mimic_demo')
    """
    from .resources import load_data_sources
    registry = load_data_sources()
    return registry.get(src)

