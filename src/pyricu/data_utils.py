"""Data manipulation utilities for ICU tables.

This module provides utilities for data transformation, aggregation,
merging, and ID system conversions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from .table import ICUTable

def stay_windows(
    data: pd.DataFrame,
    id_cols: List[str],
    start_col: str = "start",
    end_col: str = "end",
    interval: Optional[pd.Timedelta] = None,
    left_closed: bool = True,
    right_closed: bool = False,
) -> pd.DataFrame:
    """Extract stay windows from ICU data (R ricu stay_windows).
    
    Computes time windows for patient stays, optionally subdividing long
    stays into smaller intervals for rolling analysis.
    
    Args:
        data: Input DataFrame with patient stay data
        id_cols: ID column names
        start_col: Name of start time column
        end_col: Name of end time column  
        interval: Optional time interval for windowing (None = whole stay)
        left_closed: Whether interval includes start time
        right_closed: Whether interval includes end time
        
    Returns:
        DataFrame with stay windows (id_cols, start_col, end_col)
        
    Examples:
        >>> # Get whole stay windows
        >>> windows = stay_windows(
        ...     admissions, 
        ...     id_cols=['icustay_id'],
        ...     start_col='intime',
        ...     end_col='outtime'
        ... )
        >>>
        >>> # Get 6-hour windows
        >>> windows = stay_windows(
        ...     admissions,
        ...     id_cols=['icustay_id'],
        ...     start_col='intime',
        ...     end_col='outtime',
        ...     interval=pd.Timedelta(hours=6)
        ... )
    """
    # Ensure time columns are datetime
    data = data.copy()
    
    if start_col not in data.columns or end_col not in data.columns:
        raise ValueError(f"Columns '{start_col}' and '{end_col}' must exist in data")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data[start_col]):
        data[start_col] = pd.to_datetime(data[start_col])
    if not pd.api.types.is_datetime64_any_dtype(data[end_col]):
        data[end_col] = pd.to_datetime(data[end_col])
    
    # Ensure ID columns exist
    for col in id_cols:
        if col not in data.columns:
            raise ValueError(f"ID column '{col}' not found in data")
    
    # Group by ID columns
    grouped = data.groupby(id_cols, dropna=False)
    
    results = []
    for ids, group in grouped:
        # Get stay boundaries
        stay_start = group[start_col].min()
        stay_end = group[end_col].max()
        
        if pd.isna(stay_start) or pd.isna(stay_end):
            continue
        
        if interval is not None:
            # Create windowed intervals
            current = stay_start
            while current < stay_end:
                window_end = min(current + interval, stay_end)
                
                # Apply closure rules
                if not left_closed and current == stay_start:
                    current = current + pd.Timedelta(microseconds=1)
                if not right_closed and window_end == stay_end:
                    window_end = window_end - pd.Timedelta(microseconds=1)
                
                window = {start_col: current, end_col: window_end}
                
                # Add ID columns
                if isinstance(ids, tuple):
                    for i, col in enumerate(id_cols):
                        window[col] = ids[i]
                else:
                    window[id_cols[0]] = ids
                
                results.append(window)
                current += interval
        else:
            # Single window for entire stay
            window = {start_col: stay_start, end_col: stay_end}
            
            # Add ID columns
            if isinstance(ids, tuple):
                for i, col in enumerate(id_cols):
                    window[col] = ids[i]
            else:
                window[id_cols[0]] = ids
            
            results.append(window)
    
    if not results:
        return pd.DataFrame(columns=id_cols + [start_col, end_col])
    
    result_df = pd.DataFrame(results)
    
    # Reorder columns: ID columns first
    col_order = id_cols + [start_col, end_col]
    result_df = result_df[col_order]
    
    return result_df

def id_windows(
    data: pd.DataFrame,
    id_cols: List[str],
    index_col: str,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    """Extract ID-specific time windows (R ricu id_windows).
    
    Similar to stay_windows but operates on time series data, creating
    windows based on actual observation times rather than stay boundaries.
    
    Args:
        data: Input time series DataFrame
        id_cols: ID column names
        index_col: Time index column
        interval: Window size
        
    Returns:
        DataFrame with windows (id_cols, start, end)
        
    Examples:
        >>> # Get 4-hour windows from vitals data
        >>> windows = id_windows(
        ...     vitals,
        ...     id_cols=['icustay_id'],
        ...     index_col='charttime',
        ...     interval=pd.Timedelta(hours=4)
        ... )
    """
    # Ensure time column is datetime
    data = data.copy()
    
    if index_col not in data.columns:
        raise ValueError(f"Index column '{index_col}' not found in data")
    
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])
    
    # Get time ranges per ID
    time_ranges = data.groupby(id_cols, dropna=False)[index_col].agg(['min', 'max'])
    time_ranges = time_ranges.reset_index()
    time_ranges.columns = id_cols + ['start', 'end']
    
    # Use stay_windows to create intervals
    windows = stay_windows(
        time_ranges,
        id_cols=id_cols,
        start_col='start',
        end_col='end',
        interval=interval
    )
    
    return windows

def id_origin(
    data: pd.DataFrame,
    id_col: str,
) -> pd.DataFrame:
    """Get origin (first observation time) for each ID (R ricu id_origin).
    
    Args:
        data: Input DataFrame with time index
        id_col: ID column name
        
    Returns:
        DataFrame with id_col and 'origin' column
    """
    # Find time column
    time_cols = [
        col for col in data.columns
        if pd.api.types.is_datetime64_any_dtype(data[col]) or
           pd.api.types.is_timedelta64_dtype(data[col])
    ]
    
    if not time_cols:
        raise ValueError("No time column found in data")
    
    time_col = time_cols[0]
    
    # Get first time per ID
    origins = data.groupby(id_col, dropna=False)[time_col].min().reset_index()
    origins.columns = [id_col, 'origin']
    
    return origins

def upgrade_id(
    data: pd.DataFrame,
    from_id: str,
    to_id: str,
    id_map: pd.DataFrame,
) -> pd.DataFrame:
    """Upgrade from lower-level ID to higher-level ID (R ricu upgrade_id).
    
    Converts data indexed by a fine-grained ID (e.g., icustay_id) to a
    coarser ID (e.g., hadm_id or subject_id) using a mapping table.
    
    Args:
        data: Input DataFrame
        from_id: Current ID column name
        to_id: Target ID column name
        id_map: Mapping DataFrame with both from_id and to_id columns
        
    Returns:
        DataFrame with upgraded ID system
        
    Examples:
        >>> # Upgrade from icustay to hospital admission
        >>> upgraded = upgrade_id(
        ...     icustay_data,
        ...     from_id='icustay_id',
        ...     to_id='hadm_id',
        ...     id_map=icustays[['icustay_id', 'hadm_id']]
        ... )
    """
    if from_id not in data.columns:
        raise ValueError(f"Column '{from_id}' not found in data")
    
    if from_id not in id_map.columns or to_id not in id_map.columns:
        raise ValueError(f"Mapping must contain both '{from_id}' and '{to_id}'")
    
    # Merge with mapping
    result = pd.merge(
        data,
        id_map[[from_id, to_id]].drop_duplicates(),
        on=from_id,
        how='left'
    )
    
    # Remove old ID column
    result = result.drop(columns=[from_id])
    
    # Reorder: new ID first
    cols = [to_id] + [c for c in result.columns if c != to_id]
    result = result[cols]
    
    return result

def change_id_type(
    table: ICUTable,
    target_id: str,
    id_map: pd.DataFrame,
) -> ICUTable:
    """Convert table to use a different ID system.

    Args:
        table: Input ICU table
        target_id: Target ID column name
        id_map: Mapping table containing current and target IDs

    Returns:
        ICUTable with converted ID system
    """
    if not table.id_columns:
        raise ValueError("Table has no ID columns")

    current_id = table.id_columns[0]

    if current_id == target_id:
        return table

    # Merge with ID mapping
    merged = pd.merge(
        table.data,
        id_map[[current_id, target_id]],
        on=current_id,
        how="inner",
    )

    # Replace ID column
    merged = merged.drop(columns=[current_id])

    new_id_columns = [target_id] + table.id_columns[1:]

    return ICUTable(
        data=merged,
        id_columns=new_id_columns,
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )

def merge_tables(
    tables: Sequence[ICUTable],
    *,
    how: str = "outer",
    on: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Merge multiple ICU tables into a single wide-format DataFrame.

    Args:
        tables: Sequence of ICU tables to merge
        how: Merge method ('inner', 'outer', 'left', 'right')
        on: Columns to merge on (default: ID + index columns)

    Returns:
        Merged DataFrame
    """
    if not tables:
        return pd.DataFrame()

    if len(tables) == 1:
        return tables[0].data

    # Determine merge keys
    if on is None:
        on = list(tables[0].id_columns)
        if tables[0].index_column:
            on.append(tables[0].index_column)

    # Start with first table
    result = tables[0].data.copy()

    # Merge remaining tables
    for table in tables[1:]:
        # Validate merge keys
        if set(on) - set(table.data.columns):
            raise ValueError(f"Table missing merge keys: {set(on) - set(table.data.columns)}")

        result = pd.merge(result, table.data, on=on, how=how)

    return result

def aggregate_table(
    table: ICUTable,
    *,
    aggregation: str | Mapping[str, Any] = "mean",
    by: Optional[List[str]] = None,
) -> ICUTable:
    """Aggregate table data by grouping columns.

    Args:
        table: Input ICU table
        aggregation: Aggregation method or dict mapping columns to methods
        by: Columns to group by (default: ID + index columns)

    Returns:
        Aggregated ICUTable
    """
    if by is None:
        by = list(table.id_columns)
        if table.index_column:
            by.append(table.index_column)

    if not by:
        return table

    # Prepare aggregation specification
    if isinstance(aggregation, str):
        agg_dict = {}
        for col in table.data.columns:
            if col not in by:
                agg_dict[col] = aggregation
    else:
        agg_dict = dict(aggregation)

    # Perform aggregation
    grouped = table.data.groupby(by, as_index=False, dropna=False)
    aggregated = grouped.agg(agg_dict)

    return ICUTable(
        data=aggregated,
        id_columns=table.id_columns,
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )

def filter_table(
    table: ICUTable,
    condition: Any,
) -> ICUTable:
    """Filter table rows based on a condition.

    Args:
        table: Input ICU table
        condition: Boolean mask or query string

    Returns:
        Filtered ICUTable
    """
    if isinstance(condition, str):
        filtered = table.data.query(condition)
    else:
        filtered = table.data[condition]

    return ICUTable(
        data=filtered.reset_index(drop=True),
        id_columns=table.id_columns,
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )

def pivot_table(
    table: ICUTable,
    values: str,
    index: List[str],
    columns: str,
    *,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Create a pivot table from ICU data.

    Args:
        table: Input ICU table
        values: Column to aggregate
        index: Columns to use as index
        columns: Column to pivot on
        aggfunc: Aggregation function

    Returns:
        Pivoted DataFrame
    """
    return pd.pivot_table(
        table.data,
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
    )

def rename_columns(
    table: ICUTable,
    mapping: Mapping[str, str],
) -> ICUTable:
    """Rename columns in a table.

    Args:
        table: Input ICU table
        mapping: Dictionary mapping old names to new names

    Returns:
        ICUTable with renamed columns
    """
    df = table.data.rename(columns=mapping)

    # Update metadata if renamed columns were in metadata
    id_columns = [mapping.get(col, col) for col in table.id_columns]
    index_column = mapping.get(table.index_column) if table.index_column else None
    value_column = mapping.get(table.value_column) if table.value_column else None
    unit_column = mapping.get(table.unit_column) if table.unit_column else None
    time_columns = [mapping.get(col, col) for col in table.time_columns]

    return ICUTable(
        data=df,
        id_columns=id_columns,
        index_column=index_column,
        value_column=value_column,
        unit_column=unit_column,
        time_columns=time_columns,
    )

def select_columns(
    table: ICUTable,
    columns: Iterable[str],
) -> ICUTable:
    """Select specific columns from a table.

    Args:
        table: Input ICU table
        columns: Columns to keep

    Returns:
        ICUTable with selected columns
    """
    cols = list(columns)
    df = table.data[cols]

    # Update metadata to only include selected columns
    id_columns = [col for col in table.id_columns if col in cols]
    index_column = table.index_column if table.index_column in cols else None
    value_column = table.value_column if table.value_column in cols else None
    unit_column = table.unit_column if table.unit_column in cols else None
    time_columns = [col for col in table.time_columns if col in cols]

    return ICUTable(
        data=df,
        id_columns=id_columns,
        index_column=index_column,
        value_column=value_column,
        unit_column=unit_column,
        time_columns=time_columns,
    )

def add_column(
    table: ICUTable,
    name: str,
    values: Any,
) -> ICUTable:
    """Add a new column to the table.

    Args:
        table: Input ICU table
        name: Column name
        values: Column values (scalar, Series, or array)

    Returns:
        ICUTable with new column
    """
    df = table.data.copy()
    df[name] = values

    return ICUTable(
        data=df,
        id_columns=table.id_columns,
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )

def drop_columns(
    table: ICUTable,
    columns: Iterable[str],
) -> ICUTable:
    """Drop columns from the table.

    Args:
        table: Input ICU table
        columns: Columns to drop

    Returns:
        ICUTable without specified columns
    """
    df = table.data.drop(columns=list(columns))

    # Update metadata
    id_columns = [col for col in table.id_columns if col not in columns]
    index_column = None if table.index_column in columns else table.index_column
    value_column = None if table.value_column in columns else table.value_column
    unit_column = None if table.unit_column in columns else table.unit_column
    time_columns = [col for col in table.time_columns if col not in columns]

    return ICUTable(
        data=df,
        id_columns=id_columns,
        index_column=index_column,
        value_column=value_column,
        unit_column=unit_column,
        time_columns=time_columns,
    )

def sort_table(
    table: ICUTable,
    by: List[str],
    ascending: bool = True,
) -> ICUTable:
    """Sort table by specified columns.

    Args:
        table: Input ICU table
        by: Columns to sort by
        ascending: Sort order

    Returns:
        Sorted ICUTable
    """
    df = table.data.sort_values(by=by, ascending=ascending).reset_index(drop=True)

    return ICUTable(
        data=df,
        id_columns=table.id_columns,
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )

# ============================================================================
# Additional utilities from R ricu
# ============================================================================

def id_origin(
    data: pd.DataFrame,
    id_var: str = "icustay_id",
    origin_name: Optional[str] = None,
) -> pd.DataFrame:
    """Get ID origin times (R ricu id_origin).
    
    Returns admission/origin timestamps for each ID.
    
    Args:
        data: Input DataFrame
        id_var: ID column name
        origin_name: Name for origin column (default: use metadata)
        
    Returns:
        DataFrame with ID and origin times
        
    Examples:
        >>> origins = id_origin(admissions, id_var='icustay_id')
    """
    if not hasattr(data, '_metadata') or 'origin_col' not in getattr(data, '_metadata', {}):
        # Try to infer origin column (typically 'intime', 'admittime', etc.)
        time_cols = [
            col for col in data.columns
            if 'time' in col.lower() or 'date' in col.lower()
        ]
        
        if not time_cols:
            raise ValueError("Cannot determine origin column - no time columns found")
        
        # Prefer columns with 'in' or 'admit' in name
        origin_candidates = [c for c in time_cols if 'in' in c.lower() or 'admit' in c.lower()]
        origin_col = origin_candidates[0] if origin_candidates else time_cols[0]
    else:
        origin_col = data._metadata.get('origin_col')
    
    result = data[[id_var, origin_col]].drop_duplicates()
    
    if origin_name:
        result = result.rename(columns={origin_col: origin_name})
    
    return result

def id_windows(
    data: pd.DataFrame,
    id_var: str = "icustay_id",
    start_var: str = "start",
    end_var: str = "end",
) -> pd.DataFrame:
    """Get ID windows (R ricu id_windows).
    
    Returns time windows for each ID.
    
    Args:
        data: Input DataFrame
        id_var: ID column name
        start_var: Start time column name
        end_var: End time column name
        
    Returns:
        DataFrame with id, start, and end columns
        
    Examples:
        >>> windows = id_windows(admissions, 'icustay_id', 'intime', 'outtime')
    """
    if start_var not in data.columns or end_var not in data.columns:
        raise ValueError(f"Columns '{start_var}' and '{end_var}' must exist")
    
    # Group by ID and get min start, max end
    result = data.groupby(id_var, dropna=False).agg({
        start_var: 'min',
        end_var: 'max'
    }).reset_index()
    
    result = result.rename(columns={start_var: 'start', end_var: 'end'})
    
    return result

def id_map(
    data: pd.DataFrame,
    source_id: str,
    target_id: str,
    start_col: str = "start",
    end_col: str = "end",
) -> pd.DataFrame:
    """Create ID mapping between two ID systems (R ricu id_map).
    
    Maps between different ID systems (e.g., hospital admission to ICU stay).
    
    Args:
        data: DataFrame containing both ID systems
        source_id: Source ID column name
        target_id: Target ID column name
        start_col: Start time column for target ID
        end_col: End time column for target ID
        
    Returns:
        DataFrame mapping source_id to target_id with time windows
        
    Examples:
        >>> # Map from hospital admission to ICU stays
        >>> mapping = id_map(
        ...     admissions,
        ...     source_id='hadm_id',
        ...     target_id='icustay_id',
        ...     start_col='intime',
        ...     end_col='outtime'
        ... )
    """
    # Select required columns
    cols = list(set([source_id, target_id, start_col, end_col]))
    result = data[cols].copy()
    
    # Remove duplicates
    result = result.drop_duplicates()
    
    # Sort by source_id and start time
    result = result.sort_values([source_id, start_col])
    
    return result

def change_id(
    data: pd.DataFrame,
    current_id: str,
    target_id: str,
    mapping: pd.DataFrame,
    keep_old_id: bool = False,
) -> pd.DataFrame:
    """Change between ID systems (R ricu change_id).
    
    Converts data from one ID system to another using a mapping.
    
    Args:
        data: Input DataFrame with current_id
        current_id: Current ID column name
        target_id: Target ID column name
        mapping: DataFrame mapping current_id to target_id
        keep_old_id: Whether to keep the old ID column
        
    Returns:
        DataFrame with target_id instead of (or in addition to) current_id
        
    Examples:
        >>> # Change from hadm_id to icustay_id
        >>> result = change_id(
        ...     lab_data,
        ...     current_id='hadm_id',
        ...     target_id='icustay_id',
        ...     mapping=id_mapping
        ... )
    """
    if current_id not in data.columns:
        raise ValueError(f"Current ID column '{current_id}' not found")
    
    if target_id not in mapping.columns or current_id not in mapping.columns:
        raise ValueError(f"Mapping must contain both '{current_id}' and '{target_id}'")
    
    # Merge with mapping
    result = data.merge(
        mapping[[current_id, target_id]].drop_duplicates(),
        on=current_id,
        how='left'
    )
    
    # Remove old ID if requested
    if not keep_old_id:
        result = result.drop(columns=[current_id])
    
    return result

def merge_patid(
    data: pd.DataFrame,
    patient_ids: Optional[List[Any]] = None,
    id_col: str = "subject_id",
) -> pd.DataFrame:
    """Filter data by patient IDs (R ricu merge_patid).
    
    Args:
        data: Input DataFrame
        patient_ids: List of patient IDs to keep (None = keep all)
        id_col: Patient ID column name
        
    Returns:
        Filtered DataFrame
    """
    if patient_ids is None or len(patient_ids) == 0:
        return data
    
    if id_col not in data.columns:
        # Try to find patient ID column
        possible_cols = [c for c in data.columns if 'subject' in c.lower() or 'patient' in c.lower()]
        if possible_cols:
            id_col = possible_cols[0]
        else:
            raise ValueError(f"Patient ID column '{id_col}' not found")
    
    return data[data[id_col].isin(patient_ids)].copy()

def re_time(
    times: pd.Series,
    interval: pd.Timedelta,
) -> pd.Series:
    """Discretize times by flooring to the requested interval (ricu ``re_time``)."""

    if times is None or interval is None:
        return times

    from .ts_utils import round_to_interval

    return round_to_interval(times, interval)

