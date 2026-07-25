"""Data manipulation tools (R ricu data-utils.R, tbl-utils.R).

Provides utility functions for data manipulation.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import pandas as pd

from ..table import IdTbl, TsTbl, WinTbl
from .ts_utils import has_gaps

#: Seconds per supported duration unit, so any pair converts by one ratio
#: instead of a per-unit branch that has to be kept consistent by hand.
_DUR_UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}

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
    # NB: `..table`, not `.table` — this module lives in easyicu/io/, and the
    # relative import was wrong at HEAD, so change_dur_unit() raised
    # ModuleNotFoundError for every caller despite being exported as public API.
    from ..table.meta import dur_var, dur_col
    from ..table.duration import (
        UNIT_TIMEDELTA,
        DurationUnitError,
        get_dur_var_unit,
        set_dur_var_unit,
    )

    if unit not in _DUR_UNIT_SECONDS:
        raise ValueError(f"Unknown unit: {unit}")

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
        # A numeric duration carries no unit of its own — the source unit must
        # come from the declaration, never from "assume minutes". Assuming here
        # is exactly the guess the dur_var contract exists to remove: converting
        # an hours-valued column "to hours" would still divide it by 60.
        source_unit = get_dur_var_unit(tbl.data)
        if source_unit is None:
            raise DurationUnitError(
                f"change_dur_unit({unit!r}) needs the CURRENT unit of numeric "
                f"column {dur_v!r}, which is undeclared. Declare it at the "
                "producer with set_dur_var_unit(frame, ...) before converting."
            )
        if source_unit == UNIT_TIMEDELTA:
            raise DurationUnitError(
                f"column {dur_v!r} declares unit 'timedelta' but holds a numeric "
                "dtype; the producer must convert before the unit is consumed"
            )
        new_data = tbl.data.copy()
        factor = _DUR_UNIT_SECONDS[source_unit] / _DUR_UNIT_SECONDS[unit]
        new_data[dur_v] = new_data[dur_v] * factor

    # Record the unit we just converted TO. Leaving the old declaration in
    # place made the value and its label disagree, and the next consumer would
    # convert again from the stale unit.
    set_dur_var_unit(new_data, unit)

    return WinTbl(
        data=new_data,
        id_vars=tbl.id_vars,
        index_var=tbl.index_var if isinstance(tbl, WinTbl) else None,
        dur_var=dur_v,
        dur_unit=unit,
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
    from ..resources import load_data_sources
    registry = load_data_sources()
    return registry.get(src)

