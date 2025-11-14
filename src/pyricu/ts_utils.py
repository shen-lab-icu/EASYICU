"""Time series utilities for ICU data processing.

This module provides utilities for handling time-indexed data, including
interval alignment, windowing, and time-based transformations.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional, Union, List, Iterable
import logging

import pandas as pd
import numpy as np

from .table import ICUTable


def change_interval(
    table: ICUTable | pd.DataFrame,
    interval: pd.Timedelta | timedelta = None,
    *,
    new_interval: pd.Timedelta | timedelta = None,
    time_col: str = None,
    aggregation: Optional[str] = None,
    fill_gaps: bool = False,
) -> ICUTable | pd.DataFrame:
    """Change the time resolution of a time series table.
    
    Replicates R ricu's change_interval behavior: round times to interval,
    aggregate同一时间点的多个值, and optionally fill gaps to create
    a complete time series (matching R ricu's fill_gaps + aggregate).

    Args:
        table: Input ICU table or DataFrame with time index
        interval: Target time interval (alternative name: new_interval)
        new_interval: Target time interval (alternative to interval)
        time_col: Time column name (for DataFrame input)
        aggregation: Aggregation method ('mean', 'median', 'first', 'last', etc.)
        fill_gaps: Whether to fill missing time points (default True, matches R ricu)

    Returns:
        New ICUTable or DataFrame with adjusted time resolution
    """
    # Handle parameter aliases
    if interval is None and new_interval is None:
        raise ValueError("Either 'interval' or 'new_interval' must be specified")
    
    target_interval = interval if interval is not None else new_interval
    
    # Convert timedelta to pd.Timedelta if needed
    if isinstance(target_interval, timedelta):
        target_interval = pd.Timedelta(target_interval)
    
    def _fill_time_gaps(
        df: pd.DataFrame,
        id_cols: List[str],
        time_column: str,
        target_interval: pd.Timedelta,
        numeric_time: bool,
    ) -> pd.DataFrame:
        """Vectorized gap filler that relies on pandas reindexing."""
        if df.empty:
            return df

        def _build_grid(start_value, end_value):
            if numeric_time:
                step = target_interval.total_seconds() / 3600.0
                return np.arange(start_value, end_value + step, step)
            start_dt = pd.to_datetime(start_value)
            end_dt = pd.to_datetime(end_value)
            if pd.isna(start_dt) or pd.isna(end_dt):
                return None
            return pd.date_range(start=start_dt, end=end_dt, freq=target_interval)

        def _expand_group(group: pd.DataFrame) -> pd.DataFrame:
            usable = group[time_column].dropna()
            if usable.empty:
                return group
            grid = _build_grid(usable.iloc[0], usable.iloc[-1])
            if grid is None or len(grid) == 0:
                return group
            expanded = (
                group.set_index(time_column)
                .reindex(grid)
                .reset_index()
                .rename(columns={"index": time_column})
            )
            for col in id_cols:
                if col in group.columns:
                    expanded[col] = group[col].iloc[0]
            return expanded

        if id_cols:
            filled = (
                df.groupby(id_cols, dropna=False, sort=False, group_keys=False)
                .apply(_expand_group)
                .reset_index(drop=True)
            )
        else:
            filled = _expand_group(df)

        for col in df.columns:
            if col not in filled.columns:
                filled[col] = np.nan
        return filled[df.columns]

    # Handle DataFrame input
    if isinstance(table, pd.DataFrame):
        if time_col is None:
            # Try to guess time column
            for col in ['datetime', 'time', 'charttime', 'index_time']:
                if col in table.columns:
                    time_col = col
                    break
        
        if time_col is None:
            raise ValueError("time_col must be specified for DataFrame input")
        
        df = table.copy()
        
        # Round times to the specified interval
        df[time_col] = pd.to_datetime(df[time_col]).dt.floor(target_interval)
        
        if aggregation:
            # Group by time and aggregate
            numeric_cols = df.select_dtypes(include=['number']).columns
            agg_dict = {col: aggregation for col in numeric_cols if col != time_col}
            if agg_dict:
                try:
                    df = df.groupby(time_col, as_index=False).agg(agg_dict)
                except Exception:
                    # 聚合失败时退化为去重，避免抛错
                    df = df.drop_duplicates(subset=[time_col], keep="first")
            else:
                # 无可聚合的数值列，退化为去重
                df = df.drop_duplicates(subset=[time_col], keep="first")
        else:
            # Just drop duplicates
            df = df.drop_duplicates(subset=[time_col], keep="first")
        
        return df
    
    # Handle ICUTable input
    if not table.index_column or table.index_column not in table.data.columns:
        return table

    df = table.data.copy()

    numeric_time = pd.api.types.is_numeric_dtype(df[table.index_column])

    # Handle numeric time (hours since admission) differently from datetime
    if numeric_time:
        # Time is already in hours (since admission), use floor to round down
        # R ricu uses floor() not round(): round_to(x, to=1) = floor(x)
        # e.g., if interval is 1 hour, floor 11.7 -> 11.0, floor 11.3 -> 11.0
        # 🔧 FIX: Handle None values to prevent division errors
        total_seconds = target_interval.total_seconds()
        if total_seconds is None:
            raise ValueError(f"Invalid target_interval: {target_interval}")
        hours_per_interval = total_seconds / 3600.0
        if hours_per_interval == 1.0:
            # Special case: if interval is 1 hour, use floor directly
            # 🔧 FIX: Handle None/NaN values to prevent division errors
            df[table.index_column] = np.floor(pd.to_numeric(df[table.index_column], errors='coerce'))
        else:
            # General case: floor(x / to) * to (matches R ricu round_to)
            # 🔧 FIX: Handle None/NaN values to prevent division errors
            numeric_time = pd.to_numeric(df[table.index_column], errors='coerce')
            df[table.index_column] = np.floor(numeric_time / hours_per_interval) * hours_per_interval
    else:
        # Time is datetime
        # 注意：如果时间应该是相对于入院时间的小时数，这里不应该出现datetime
        # 但为了安全，我们将datetime舍入到指定的interval，然后保持为datetime
        # （理想情况下，_align_time_to_admission应该已经将时间转换为小时数）
        df[table.index_column] = pd.to_datetime(df[table.index_column], errors='coerce')
        df[table.index_column] = df[table.index_column].dt.floor(target_interval)

    # Group by ID columns and rounded time, and aggregate
    group_cols = list(table.id_columns) + [table.index_column]

    # Filter group_cols to only include columns that actually exist in the dataframe
    # This handles cases where ID columns were filtered out during processing (e.g., eICU infusiondrug)
    existing_group_cols = [col for col in group_cols if col in df.columns]
    if len(existing_group_cols) != len(group_cols):
        missing_cols = set(group_cols) - set(existing_group_cols)
        import logging
        logging.debug(f"change_interval: Missing columns {missing_cols} in dataframe. Using available columns: {existing_group_cols}")
    group_cols = existing_group_cols

    # Ensure we have at least one grouping column
    if not group_cols:
        # If no valid grouping columns, create a dummy group column
        df['__dummy_group'] = 1
        group_cols = ['__dummy_group']
        import logging
        logging.debug("change_interval: No valid grouping columns found. Using dummy group column.")

    if aggregation:
        # Apply aggregation - only aggregate numeric columns
        agg_dict = {}
        for col in df.columns:
            if col not in group_cols:
                # Only aggregate numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    agg_dict[col] = aggregation
                else:
                    # For non-numeric columns, keep first value
                    agg_dict[col] = 'first'

        if agg_dict:
            try:
                df = df.groupby(group_cols, as_index=False).agg(agg_dict)
            except Exception:
                # 聚合失败时退化为去重，避免抛错
                df = df.drop_duplicates(subset=group_cols, keep="first")
        else:
            # 无可聚合列，退化为去重
            df = df.drop_duplicates(subset=group_cols, keep="first")
    else:
        # Just drop duplicates, keeping first occurrence
        df = df.drop_duplicates(subset=group_cols, keep="first")
    
    # CRITICAL: Fill gaps in time series (replicates R ricu fill_gaps)
    # This ensures all time points are present, even if no measurements exist
    if fill_gaps:
        df = _fill_time_gaps(
            df,
            list(table.id_columns),
            table.index_column,
            target_interval,
            numeric_time,
        )

    return ICUTable(
        data=df,
        id_columns=table.id_columns,
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )


def stay_windows(
    table: ICUTable,
    start_var: str,
    end_var: str,
    *,
    left_closed: bool = True,
    right_closed: bool = False,
) -> pd.DataFrame:
    """Extract stay windows (start/end times) for patient stays.

    Args:
        table: Input ICU table
        start_var: Column name for start time
        end_var: Column name for end time
        left_closed: Whether interval includes start time
        right_closed: Whether interval includes end time

    Returns:
        DataFrame with ID columns, start time, and end time
    """
    df = table.data.copy()

    required_cols = set(table.id_columns) | {start_var, end_var}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df[list(table.id_columns) + [start_var, end_var]].copy()
    result = result.drop_duplicates()

    return result


def expand_intervals(
    table: ICUTable,
    interval: pd.Timedelta,
    start_col: Optional[str] = None,
    end_col: Optional[str] = None,
) -> ICUTable:
    """Expand a table by creating rows for each time point in an interval.

    Args:
        table: Input ICU table
        interval: Time step for expansion
        start_col: Column containing start times (default: index_column)
        end_col: Column containing end times

    Returns:
        Expanded ICUTable with regular time intervals
    """
    if start_col is None:
        start_col = table.index_column

    if start_col is None or end_col is None:
        raise ValueError("Both start and end columns must be specified")

    df = table.data.copy()

    expanded_rows = []
    for _, row in df.iterrows():
        start = pd.to_datetime(row[start_col])
        end = pd.to_datetime(row[end_col])

        # Generate time range
        time_range = pd.date_range(start=start, end=end, freq=interval)

        for time_point in time_range:
            new_row = row.copy()
            new_row[start_col] = time_point
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)

    return ICUTable(
        data=expanded_df,
        id_columns=table.id_columns,
        index_column=start_col,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )


def expand(
    data: pd.DataFrame,
    start_var: str,
    end_var: str,
    step_size: pd.Timedelta,
    id_cols: Optional[list] = None,
    keep_vars: Optional[list] = None,
) -> pd.DataFrame:
    """Expand intervals into individual time points (R ricu expand).
    
    Converts duration data (with start/end times) into time series data
    by creating a row for each time step within each interval.
    
    Args:
        data: Input DataFrame with interval data
        start_var: Column name for interval start times
        end_var: Column name for interval end times (or duration)
        step_size: Time step for expansion (e.g., pd.Timedelta(hours=1))
        id_cols: ID columns to group by (inferred if None)
        keep_vars: Additional columns to keep (values repeated)
        
    Returns:
        Expanded DataFrame with one row per time step
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1],
        ...     'start': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 03:00']),
        ...     'end': pd.to_datetime(['2020-01-01 02:00', '2020-01-01 05:00']),
        ...     'value': [10, 20]
        ... })
        >>> expand(df, 'start', 'end', pd.Timedelta(hours=1), 
        ...        id_cols=['id'], keep_vars=['value'])
    """
    if id_cols is None:
        # Infer ID columns (columns with 'id' in name)
        id_cols = [col for col in data.columns if 'id' in col.lower()]
    
    if keep_vars is None:
        keep_vars = []
    
    # Ensure start and end are datetime
    data = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[start_var]):
        data[start_var] = pd.to_datetime(data[start_var])
    
    # Handle end_var as duration or absolute time
    if pd.api.types.is_timedelta64_dtype(data[end_var]):
        data['_end_abs'] = data[start_var] + data[end_var]
        end_col = '_end_abs'
    else:
        if not pd.api.types.is_datetime64_any_dtype(data[end_var]):
            data[end_var] = pd.to_datetime(data[end_var])
        end_col = end_var
    
    expanded_rows = []
    for _, row in data.iterrows():
        start = row[start_var]
        end = row[end_col]
        
        if pd.isna(start) or pd.isna(end) or start > end:
            continue
        
        # Generate time range
        time_range = pd.date_range(start=start, end=end, freq=step_size)
        
        for time_point in time_range:
            new_row = {start_var: time_point}
            
            # Add ID columns
            for col in id_cols:
                if col in row:
                    new_row[col] = row[col]
            
            # Add keep_vars
            for col in keep_vars:
                if col in row:
                    new_row[col] = row[col]
            
            expanded_rows.append(new_row)
    
    if not expanded_rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(expanded_rows)
    
    # Clean up temporary column
    if '_end_abs' in data.columns:
        data.drop('_end_abs', axis=1, inplace=True)
    
    return result


def collapse(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    start_var: str = "start",
    end_var: str = "end",
    **agg_funcs
) -> pd.DataFrame:
    """Collapse time series into intervals (inverse of expand).
    
    Groups time series data and represents each group as an interval
    with start/end times plus aggregated values.
    
    Args:
        data: Input time series DataFrame
        id_cols: Columns to group by
        index_col: Time index column
        start_var: Name for output start time column
        end_var: Name for output end time column
        **agg_funcs: Aggregation functions for other columns (e.g., value='mean')
        
    Returns:
        Collapsed DataFrame with intervals
        
    Examples:
        >>> ts_data = pd.DataFrame({
        ...     'id': [1]*5,
        ...     'time': pd.date_range('2020-01-01', periods=5, freq='H'),
        ...     'value': [10, 12, 11, 13, 14]
        ... })
        >>> collapse(ts_data, ['id'], 'time', value='mean')
    """
    if not id_cols:
        raise ValueError("id_cols cannot be empty")
    
    # Build aggregation dict
    agg_dict = {
        index_col: ['min', 'max']
    }
    
    for col, func in agg_funcs.items():
        if col in data.columns:
            agg_dict[col] = func
    
    # Group and aggregate
    result = data.groupby(id_cols, as_index=False).agg(agg_dict)
    
    # Flatten column names
    result.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                      for col in result.columns.values]
    
    # Rename min/max to start/end
    result = result.rename(columns={
        f'{index_col}_min': start_var,
        f'{index_col}_max': end_var
    })
    
    return result


def fill_gaps(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    interval: pd.Timedelta,
    limits: Optional[pd.DataFrame] = None,
    method: str = "none",
) -> pd.DataFrame:
    """Fill time gaps in a time series (R ricu fill_gaps).

    Args:
        data: Input time series DataFrame
        id_cols: ID columns to group by
        index_col: Time index column
        interval: Expected time interval between observations
        limits: DataFrame with start/end times per ID (optional)
        method: Fill method ('ffill', 'bfill', 'interpolate', or 'none')

    Returns:
        DataFrame with filled gaps
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1, 1],
        ...     'time': pd.to_datetime(['2020-01-01 00:00', 
        ...                             '2020-01-01 02:00',  # gap!
        ...                             '2020-01-01 03:00']),
        ...     'value': [10, 20, 30]
        ... })
        >>> fill_gaps(df, ['id'], 'time', pd.Timedelta(hours=1))
    """
    data = data.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])
    
    filled_groups = []
    
    for id_vals, group in data.groupby(id_cols):
        if len(group) == 0:
            continue
        
        # Determine time range
        if limits is not None:
            # Use limits DataFrame
            if isinstance(id_vals, tuple):
                mask = True
                for i, col in enumerate(id_cols):
                    mask = mask & (limits[col] == id_vals[i])
            else:
                mask = (limits[id_cols[0]] == id_vals)
            
            limit_row = limits[mask]
            if len(limit_row) > 0:
                min_time = pd.to_datetime(limit_row.iloc[0]['start'])
                max_time = pd.to_datetime(limit_row.iloc[0]['end'])
            else:
                min_time = group[index_col].min()
                max_time = group[index_col].max()
        else:
            min_time = group[index_col].min()
            max_time = group[index_col].max()
        
        # Create complete time index
        complete_index = pd.date_range(start=min_time, end=max_time, freq=interval)
        
        # Reindex
        group = group.set_index(index_col)
        group = group.reindex(complete_index)
        
        # Fill ID columns
        for i, id_col in enumerate(id_cols):
            if isinstance(id_vals, tuple):
                group[id_col] = id_vals[i]
            else:
                group[id_col] = id_vals
        
        # Apply fill method
        if method == "ffill":
            group = group.fillna(method="ffill")
        elif method == "bfill":
            group = group.fillna(method="bfill")
        elif method == "interpolate":
            numeric_cols = group.select_dtypes(include="number").columns
            group[numeric_cols] = group[numeric_cols].interpolate()
        
        group = group.reset_index()
        group.rename(columns={"index": index_col}, inplace=True)
        filled_groups.append(group)
    
    if not filled_groups:
        return pd.DataFrame()
    
    result = pd.concat(filled_groups, ignore_index=True)
    return result


def replace_na(
    data: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "ffill",
    value: Any = None,
    limit: Optional[int] = None,
    max_gap: Optional[pd.Timedelta] = None,
    index_col: Optional[str] = None,
    id_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Replace NA values with specified method (R ricu replace_na).
    
    More comprehensive than pandas fillna, supporting time-aware gap limits
    and grouped operations.
    
    Args:
        data: Input DataFrame
        columns: Columns to fill (None = all numeric columns)
        method: Fill method:
            - 'ffill': Forward fill (last observation carried forward)
            - 'bfill': Backward fill (next observation carried backward)
            - 'interpolate': Linear interpolation
            - 'const': Constant value
            - 'mean': Group mean
            - 'median': Group median
            - 'mode': Group mode
        value: Constant value when method='const'
        limit: Maximum number of consecutive NAs to fill
        max_gap: Maximum time gap to fill (requires index_col)
        index_col: Time index column for gap checking
        id_cols: ID columns for grouped operations
        
    Returns:
        DataFrame with NAs replaced
        
    Examples:
        >>> # Forward fill with gap limit
        >>> replace_na(df, method='ffill', max_gap=pd.Timedelta(hours=4),
        ...            index_col='time', id_cols=['patient_id'])
        >>>
        >>> # Interpolate numeric columns
        >>> replace_na(df, method='interpolate')
        >>>
        >>> # Fill with constant
        >>> replace_na(df, columns=['value'], method='const', value=0)
    """
    data = data.copy()
    
    # Determine columns to fill
    if columns is None:
        if method in ['interpolate', 'mean', 'median']:
            columns = data.select_dtypes(include='number').columns.tolist()
        else:
            columns = [c for c in data.columns if c not in (id_cols or [])]
    
    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    # Filter to existing columns
    columns = [c for c in columns if c in data.columns]
    
    if not columns:
        return data
    
    # Group-based filling if id_cols specified
    if id_cols and any(col in data.columns for col in id_cols):
        existing_id_cols = [c for c in id_cols if c in data.columns]
        
        def fill_group(group):
            return _fill_na_single(
                group, columns, method, value, limit, max_gap, index_col
            )
        
        data = data.groupby(existing_id_cols, group_keys=False).apply(fill_group, include_groups=False)
        return data.reset_index(drop=True)
    else:
        return _fill_na_single(data, columns, method, value, limit, max_gap, index_col)


def _fill_na_single(
    data: pd.DataFrame,
    columns: list,
    method: str,
    value: Any,
    limit: Optional[int],
    max_gap: Optional[pd.Timedelta],
    index_col: Optional[str],
) -> pd.DataFrame:
    """Helper function to fill NAs in a single group."""
    
    for col in columns:
        if col not in data.columns:
            continue
        
        if method == "ffill":
            if max_gap is not None and index_col is not None:
                # Time-aware forward fill
                data = data.sort_values(index_col)
                time_diffs = data[index_col].diff()
                
                # Mark positions where gap exceeds max_gap
                large_gap = time_diffs > max_gap
                
                # Create groups separated by large gaps
                gap_groups = large_gap.cumsum()
                
                # Fill within each gap group
                data[col] = data.groupby(gap_groups)[col].fillna(
                    method='ffill', limit=limit
                )
            else:
                data[col] = data[col].fillna(method='ffill', limit=limit)
        
        elif method == "bfill":
            if max_gap is not None and index_col is not None:
                # Time-aware backward fill
                data = data.sort_values(index_col)
                time_diffs = data[index_col].diff(-1).abs()
                
                large_gap = time_diffs > max_gap
                gap_groups = large_gap[::-1].cumsum()[::-1]
                
                data[col] = data.groupby(gap_groups)[col].fillna(
                    method='bfill', limit=limit
                )
            else:
                data[col] = data[col].fillna(method='bfill', limit=limit)
        
        elif method == "interpolate":
            if index_col is not None:
                # Time-aware interpolation
                data = data.sort_values(index_col)
            
            if limit is not None:
                data[col] = data[col].interpolate(limit=limit, limit_direction='both')
            else:
                data[col] = data[col].interpolate()
        
        elif method == "const":
            if value is None:
                raise ValueError("value must be specified when method='const'")
            data[col] = data[col].fillna(value)
        
        elif method == "mean":
            data[col] = data[col].fillna(data[col].mean())
        
        elif method == "median":
            data[col] = data[col].fillna(data[col].median())
        
        elif method == "mode":
            mode_val = data[col].mode()
            if len(mode_val) > 0:
                data[col] = data[col].fillna(mode_val[0])
        
        else:
            raise ValueError(
                f"Unknown method: {method}. Must be one of: "
                f"'ffill', 'bfill', 'interpolate', 'const', 'mean', 'median', 'mode'"
            )
    
    return data


def slide(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    before: pd.Timedelta,
    after: pd.Timedelta = pd.Timedelta(0),
    agg_func: Optional[dict] = None,
    full_window: bool = False,
) -> pd.DataFrame:
    """Apply sliding window aggregation (R ricu slide).
    
    For each time point, creates a window spanning [time - before, time + after]
    and aggregates values within that window.
    
    Args:
        data: Input time series DataFrame
        id_cols: ID columns to group by
        index_col: Time index column
        before: Time to look back
        after: Time to look forward (default 0)
        agg_func: Dict mapping column names to aggregation functions
                  e.g., {'value': 'mean', 'count': 'sum'}
                  Can also accept strings like 'max', 'min', 'mean', 'sum'
                  or callable functions
        full_window: If True, only include rows where the full window is available
        
    Returns:
        DataFrame with windowed aggregations
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1]*10,
        ...     'time': pd.date_range('2020-01-01', periods=10, freq='H'),
        ...     'value': range(10)
        ... })
        >>> slide(df, ['id'], 'time', pd.Timedelta(hours=2), 
        ...       agg_func={'value': 'mean'})
    """
    if agg_func is None:
        agg_func = {}
    
    data = data.copy()
    
    # Handle both datetime and numeric (hours) time columns
    is_numeric_time = pd.api.types.is_numeric_dtype(data[index_col])
    
    if not is_numeric_time:
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
            data[index_col] = pd.to_datetime(data[index_col])
    
    # Convert before/after to compatible units
    if before is None:
        before_val = 24.0  # Default to 24 hours
    elif is_numeric_time:
        # Time is in hours, convert Timedelta to hours
        before_val = before.total_seconds() / 3600.0
    else:
        # Time is datetime, use Timedelta directly
        before_val = before

    if after is None:
        after_val = 0.0  # Default to 0 hours
    elif is_numeric_time:
        # Time is in hours, convert Timedelta to hours
        after_val = after.total_seconds() / 3600.0
    else:
        # Time is datetime, use Timedelta directly
        after_val = after
    
    results = []
    
    for id_vals, group in data.groupby(id_cols):
        group = group.sort_values(index_col)
        
        # For full_window check, we need group time range
        if full_window and len(group) > 0:
            group_start = group[index_col].min()
            group_end = group[index_col].max()
        
        for idx, row in group.iterrows():
            center_time = row[index_col]
            window_start = center_time - before_val
            window_end = center_time + after_val
            
            # Check if full window is available
            if full_window:
                if window_start < group_start or window_end > group_end:
                    continue
            
            # Select data in window
            window_data = group[
                (group[index_col] >= window_start) &
                (group[index_col] <= window_end)
            ]
            
            if len(window_data) == 0:
                continue
            
            # Build result row
            result_row = {index_col: center_time}
            
            # Add ID columns
            for i, col in enumerate(id_cols):
                if isinstance(id_vals, tuple):
                    result_row[col] = id_vals[i]
                else:
                    result_row[col] = id_vals
            
            # Apply aggregations
            for col, func in agg_func.items():
                if col in window_data.columns:
                    # Handle string function names
                    if isinstance(func, str):
                        if func == 'mean':
                            result_row[col] = window_data[col].mean()
                        elif func == 'sum':
                            result_row[col] = window_data[col].sum()
                        elif func == 'min':
                            result_row[col] = window_data[col].min()
                        elif func == 'max':
                            result_row[col] = window_data[col].max()
                        elif func == 'count':
                            result_row[col] = window_data[col].count()
                        elif func == 'std':
                            result_row[col] = window_data[col].std()
                        elif func == 'first':
                            result_row[col] = window_data[col].iloc[0] if len(window_data) > 0 else None
                        elif func == 'last':
                            result_row[col] = window_data[col].iloc[-1] if len(window_data) > 0 else None
                        elif func == 'max_or_na':
                            # Special handling for max_or_na
                            vals = window_data[col].dropna()
                            result_row[col] = vals.max() if len(vals) > 0 else None
                        elif func == 'min_or_na':
                            # Special handling for min_or_na
                            vals = window_data[col].dropna()
                            result_row[col] = vals.min() if len(vals) > 0 else None
                        else:
                            raise ValueError(f"Unknown aggregation function: {func}")
                    elif callable(func):
                        # Handle callable functions
                        result_row[col] = func(window_data[col])
                    else:
                        raise ValueError(f"agg_func values must be string or callable, got {type(func)}")
            
            results.append(result_row)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def slide_index(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    index: list,
    before: pd.Timedelta,
    after: pd.Timedelta = pd.Timedelta(0),
    agg_func: Optional[dict] = None,
) -> pd.DataFrame:
    """Apply sliding window aggregation at specific indices (R ricu slide_index).
    
    Similar to slide(), but only creates windows at specified time points
    instead of every observation.
    
    Args:
        data: Input time series DataFrame
        id_cols: ID columns to group by
        index_col: Time index column
        index: List of time points to center windows on
        before: Time to look back
        after: Time to look forward (default 0)
        agg_func: Dict mapping column names to aggregation functions
        
    Returns:
        DataFrame with windowed aggregations at specified indices
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1]*10,
        ...     'time': pd.date_range('2020-01-01', periods=10, freq='H'),
        ...     'value': range(10)
        ... })
        >>> # Aggregate at specific time points
        >>> slide_index(df, ['id'], 'time', 
        ...            [pd.Timestamp('2020-01-01 04:00'), pd.Timestamp('2020-01-01 08:00')],
        ...            pd.Timedelta(hours=2), agg_func={'value': 'mean'})
    """
    if agg_func is None:
        agg_func = {}
    
    data = data.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])
    
    # Convert index times to datetime if needed
    index_times = [pd.to_datetime(t) if not isinstance(t, pd.Timestamp) else t 
                   for t in index]
    
    results = []
    
    for id_vals, group in data.groupby(id_cols):
        group = group.sort_values(index_col)
        
        for center_time in index_times:
            window_start = center_time - before
            window_end = center_time + after
            
            # Select data in window
            window_data = group[
                (group[index_col] >= window_start) &
                (group[index_col] <= window_end)
            ]
            
            if len(window_data) == 0:
                continue
            
            # Build result row
            result_row = {index_col: center_time}
            
            # Add ID columns
            for i, col in enumerate(id_cols):
                if isinstance(id_vals, tuple):
                    result_row[col] = id_vals[i]
                else:
                    result_row[col] = id_vals
            
            # Apply aggregations
            for col, func in agg_func.items():
                if col in window_data.columns:
                    if func == 'mean':
                        result_row[col] = window_data[col].mean()
                    elif func == 'sum':
                        result_row[col] = window_data[col].sum()
                    elif func == 'min':
                        result_row[col] = window_data[col].min()
                    elif func == 'max':
                        result_row[col] = window_data[col].max()
                    elif func == 'count':
                        result_row[col] = window_data[col].count()
                    elif func == 'std':
                        result_row[col] = window_data[col].std()
                    elif func == 'first':
                        result_row[col] = window_data[col].iloc[0]
                    elif func == 'last':
                        result_row[col] = window_data[col].iloc[-1]
                    elif callable(func):
                        result_row[col] = func(window_data[col])
            
            results.append(result_row)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def hop(
    data: pd.DataFrame,
    windows: pd.DataFrame,
    id_cols: list,
    index_col: str,
    lwr_col: str = "start",
    upr_col: str = "end",
    agg_func: Optional[dict] = None,
    left_closed: bool = True,
    right_closed: bool = True,
) -> pd.DataFrame:
    """Apply aggregation over custom time windows (R ricu hop).
    
    Unlike slide(), this allows specifying arbitrary windows for each ID.
    
    Args:
        data: Input time series DataFrame
        windows: DataFrame with window definitions (must have id_cols, lwr_col, upr_col)
        id_cols: ID columns
        index_col: Time index column in data
        lwr_col: Column name for window start times
        upr_col: Column name for window end times
        agg_func: Dict mapping column names to aggregation functions
        left_closed: Whether window includes start time
        right_closed: Whether window includes end time
        
    Returns:
        DataFrame with aggregations per window
    """
    if agg_func is None:
        agg_func = {}
    
    data = data.copy()
    windows = windows.copy()
    
    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])
    if not pd.api.types.is_datetime64_any_dtype(windows[lwr_col]):
        windows[lwr_col] = pd.to_datetime(windows[lwr_col])
    if not pd.api.types.is_datetime64_any_dtype(windows[upr_col]):
        windows[upr_col] = pd.to_datetime(windows[upr_col])
    
    results = []
    
    for _, window_row in windows.iterrows():
        # Extract window definition
        window_start = window_row[lwr_col]
        window_end = window_row[upr_col]
        
        # Build filter for this ID
        id_filter = True
        for col in id_cols:
            id_filter = id_filter & (data[col] == window_row[col])
        
        # Filter by time window
        if left_closed and right_closed:
            time_filter = (data[index_col] >= window_start) & (data[index_col] <= window_end)
        elif left_closed and not right_closed:
            time_filter = (data[index_col] >= window_start) & (data[index_col] < window_end)
        elif not left_closed and right_closed:
            time_filter = (data[index_col] > window_start) & (data[index_col] <= window_end)
        else:
            time_filter = (data[index_col] > window_start) & (data[index_col] < window_end)
        
        window_data = data[id_filter & time_filter]
        
        if len(window_data) == 0:
            continue
        
        # Build result row
        result_row = {lwr_col: window_start, upr_col: window_end}
        
        # Add ID columns
        for col in id_cols:
            result_row[col] = window_row[col]
        
        # Apply aggregations
        for col, func in agg_func.items():
            if col in window_data.columns:
                if func == 'mean':
                    result_row[col] = window_data[col].mean()
                elif func == 'sum':
                    result_row[col] = window_data[col].sum()
                elif func == 'min':
                    result_row[col] = window_data[col].min()
                elif func == 'max':
                    result_row[col] = window_data[col].max()
                elif func == 'count':
                    result_row[col] = window_data[col].count()
                elif func == 'std':
                    result_row[col] = window_data[col].std()
                elif callable(func):
                    result_row[col] = func(window_data[col])
        
        results.append(result_row)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def has_gaps(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    interval: pd.Timedelta,
) -> bool:
    """Check if time series has missing time steps.
    
    Args:
        data: Input DataFrame
        id_cols: ID columns
        index_col: Time index column
        interval: Expected time interval
        
    Returns:
        True if there are gaps, False otherwise
    """
    for _, group in data.groupby(id_cols):
        if len(group) < 2:
            continue
        
        times = sorted(group[index_col])
        expected_len = int((times[-1] - times[0]) / interval) + 1
        
        if len(times) != expected_len:
            return True
    
    return False


def is_regular(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    interval: pd.Timedelta,
) -> bool:
    """Check if time series is regular (no gaps, sorted, unique).
    
    Args:
        data: Input DataFrame
        id_cols: ID columns
        index_col: Time index column
        interval: Expected time interval
        
    Returns:
        True if regular, False otherwise
    """
    # Check for duplicates
    if data.duplicated(subset=id_cols + [index_col]).any():
        return False
    
    # Check if sorted
    if not data[index_col].is_monotonic_increasing:
        return False
    
    # Check for gaps
    return not has_gaps(data, id_cols, index_col, interval)


def slide_windows(
    table: ICUTable,
    window_size: pd.Timedelta,
    step_size: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Create sliding windows over time series data.

    Args:
        table: Input ICU table
        window_size: Size of each window
        step_size: Step between windows (default: same as window_size)

    Returns:
        DataFrame with windowed data
    """
    if step_size is None:
        step_size = window_size

    if not table.index_column or not table.value_column:
        raise ValueError("Table must have index and value columns")

    df = table.data.copy()
    windows = []

    for id_vals, group in df.groupby(list(table.id_columns)):
        group = group.sort_values(table.index_column)

        min_time = group[table.index_column].min()
        max_time = group[table.index_column].max()

        current_time = min_time
        while current_time + window_size <= max_time:
            window_end = current_time + window_size
            window_data = group[
                (group[table.index_column] >= current_time)
                & (group[table.index_column] < window_end)
            ]

            if len(window_data) > 0:
                window_info = {
                    "window_start": current_time,
                    "window_end": window_end,
                    "n_obs": len(window_data),
                }

                # Add ID columns
                for i, id_col in enumerate(table.id_columns):
                    if isinstance(id_vals, tuple):
                        window_info[id_col] = id_vals[i]
                    else:
                        window_info[id_col] = id_vals

                # Add aggregated values
                if pd.api.types.is_numeric_dtype(window_data[table.value_column]):
                    window_info[f"{table.value_column}_mean"] = window_data[
                        table.value_column
                    ].mean()
                    window_info[f"{table.value_column}_std"] = window_data[
                        table.value_column
                    ].std()
                    window_info[f"{table.value_column}_min"] = window_data[
                        table.value_column
                    ].min()
                    window_info[f"{table.value_column}_max"] = window_data[
                        table.value_column
                    ].max()

                windows.append(window_info)

            current_time += step_size

    return pd.DataFrame(windows)


def locf(
    data: pd.DataFrame,
    id_cols: Optional[list] = None,
    max_gap: Optional[pd.Timedelta] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Last observation carried forward (R ricu locf).
    
    Fill missing values by propagating the last valid observation forward.
    Optionally respects a maximum time gap.
    
    Args:
        data: Input DataFrame
        id_cols: ID columns for grouping (None = no grouping)
        max_gap: Maximum time gap to carry forward (None = no limit)
        index_col: Time index column (required if max_gap specified)
        
    Returns:
        DataFrame with forward-filled values
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1, 1],
        ...     'time': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 02:00', '2020-01-01 04:00']),
        ...     'value': [10, np.nan, np.nan]
        ... })
        >>> locf(df, id_cols=['id'], max_gap=pd.Timedelta(hours=3), index_col='time')
    """
    data = data.copy()
    
    if id_cols and any(col in data.columns for col in id_cols):
        existing_id_cols = [c for c in id_cols if c in data.columns]
        
        def fill_group(group):
            if max_gap is not None and index_col is not None:
                # Time-aware forward fill
                group = group.sort_values(index_col)
                time_diffs = group[index_col].diff()
                
                # Mark positions where gap exceeds max_gap
                large_gap = time_diffs > max_gap
                gap_groups = large_gap.cumsum()
                
                # Fill within each gap group
                for col in group.columns:
                    if col not in existing_id_cols and col != index_col:
                        group[col] = group.groupby(gap_groups)[col].fillna(method='ffill')
            else:
                # Simple forward fill
                group = group.fillna(method='ffill')
            
            return group
        
        data = data.groupby(existing_id_cols, group_keys=False).apply(fill_group, include_groups=False)
    else:
        # No grouping
        if max_gap is not None and index_col is not None:
            data = data.sort_values(index_col)
            time_diffs = data[index_col].diff()
            large_gap = time_diffs > max_gap
            gap_groups = large_gap.cumsum()
            
            for col in data.columns:
                if col != index_col:
                    data[col] = data.groupby(gap_groups)[col].fillna(method='ffill')
        else:
            data = data.fillna(method='ffill')
    
    return data


def locb(
    data: pd.DataFrame,
    id_cols: Optional[list] = None,
    max_gap: Optional[pd.Timedelta] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Last observation carried backward (R ricu locb).
    
    Fill missing values by propagating the next valid observation backward.
    Optionally respects a maximum time gap.
    
    Args:
        data: Input DataFrame
        id_cols: ID columns for grouping (None = no grouping)
        max_gap: Maximum time gap to carry backward (None = no limit)
        index_col: Time index column (required if max_gap specified)
        
    Returns:
        DataFrame with backward-filled values
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1, 1],
        ...     'time': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 02:00', '2020-01-01 04:00']),
        ...     'value': [np.nan, np.nan, 10]
        ... })
        >>> locb(df, id_cols=['id'], max_gap=pd.Timedelta(hours=3), index_col='time')
    """
    data = data.copy()
    
    if id_cols and any(col in data.columns for col in id_cols):
        existing_id_cols = [c for c in id_cols if c in data.columns]
        
        def fill_group(group):
            if max_gap is not None and index_col is not None:
                # Time-aware backward fill
                group = group.sort_values(index_col)
                time_diffs = group[index_col].diff(-1).abs()
                
                # Mark positions where gap exceeds max_gap
                large_gap = time_diffs > max_gap
                gap_groups = large_gap[::-1].cumsum()[::-1]
                
                # Fill within each gap group
                for col in group.columns:
                    if col not in existing_id_cols and col != index_col:
                        group[col] = group.groupby(gap_groups)[col].fillna(method='bfill')
            else:
                # Simple backward fill
                group = group.fillna(method='bfill')
            
            return group
        
        data = data.groupby(existing_id_cols, group_keys=False).apply(fill_group, include_groups=False)
    else:
        # No grouping
        if max_gap is not None and index_col is not None:
            data = data.sort_values(index_col)
            time_diffs = data[index_col].diff(-1).abs()
            large_gap = time_diffs > max_gap
            gap_groups = large_gap[::-1].cumsum()[::-1]
            
            for col in data.columns:
                if col != index_col:
                    data[col] = data.groupby(gap_groups)[col].fillna(method='bfill')
        else:
            data = data.fillna(method='bfill')
    
    return data


def calc_dur(
    data: pd.DataFrame,
    start_col: str,
    end_col: str,
    dur_col: str = "duration",
) -> pd.DataFrame:
    """Calculate duration from start and end times (R ricu calc_dur).
    
    Args:
        data: Input DataFrame
        start_col: Start time column
        end_col: End time column
        dur_col: Name for output duration column
        
    Returns:
        DataFrame with added duration column
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'start': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 10:00']),
        ...     'end': pd.to_datetime(['2020-01-01 02:00', '2020-01-01 14:00'])
        ... })
        >>> calc_dur(df, 'start', 'end')
    """
    data = data.copy()
    
    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(data[start_col]):
        data[start_col] = pd.to_datetime(data[start_col])
    if not pd.api.types.is_datetime64_any_dtype(data[end_col]):
        data[end_col] = pd.to_datetime(data[end_col])
    
    # Calculate duration
    data[dur_col] = data[end_col] - data[start_col]
    
    return data


def remove_gaps(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    data_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Remove time steps that consist entirely of NA values (R ricu remove_gaps).
    
    Inverse of fill_gaps - removes rows where all data columns are NA.
    
    Args:
        data: Input DataFrame
        id_cols: ID columns
        index_col: Time index column
        data_cols: Columns to check for NA (None = all except ID and index)
        
    Returns:
        DataFrame with gap rows removed
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1, 1],
        ...     'time': pd.date_range('2020-01-01', periods=3, freq='H'),
        ...     'value': [10, np.nan, 30]
        ... })
        >>> remove_gaps(df, ['id'], 'time', ['value'])
    """
    if data_cols is None:
        # All columns except ID and index
        data_cols = [c for c in data.columns if c not in id_cols and c != index_col]
    
    if not data_cols:
        return data
    
    # Keep rows where at least one data column is not NA
    mask = data[data_cols].notna().any(axis=1)
    
    return data[mask].copy()


def hours(n: float) -> pd.Timedelta:
    """Create a timedelta representing n hours."""
    return pd.Timedelta(hours=n)


def minutes(n: float) -> pd.Timedelta:
    """Create a timedelta representing n minutes."""
    return pd.Timedelta(minutes=n)


def mins(n: float) -> pd.Timedelta:
    """Create a timedelta representing n minutes (alias for minutes)."""
    return pd.Timedelta(minutes=n)


def days(n: float) -> pd.Timedelta:
    """Create a timedelta representing n days."""
    return pd.Timedelta(days=n)


def secs(n: float) -> pd.Timedelta:
    """Create a timedelta representing n seconds."""
    return pd.Timedelta(seconds=n)


def weeks(n: float) -> pd.Timedelta:
    """Create a timedelta representing n weeks."""
    return pd.Timedelta(weeks=n)


def merge_ranges(
    data: pd.DataFrame,
    id_cols: list,
    start_col: str,
    end_col: str,
    max_gap: pd.Timedelta = pd.Timedelta(0),
) -> pd.DataFrame:
    """Merge overlapping or nearby time ranges (R ricu merge_ranges).
    
    Combines intervals that overlap or are within max_gap of each other.
    
    Args:
        data: Input DataFrame with time ranges
        id_cols: ID columns
        start_col: Start time column
        end_col: End time column
        max_gap: Maximum gap to merge across (default 0 = only overlapping)
        
    Returns:
        DataFrame with merged ranges
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1, 1],
        ...     'start': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 02:00', '2020-01-01 05:00']),
        ...     'end': pd.to_datetime(['2020-01-01 03:00', '2020-01-01 04:00', '2020-01-01 07:00'])
        ... })
        >>> merge_ranges(df, ['id'], 'start', 'end', max_gap=pd.Timedelta(hours=1))
    """
    data = data.copy()
    
    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(data[start_col]):
        data[start_col] = pd.to_datetime(data[start_col])
    if not pd.api.types.is_datetime64_any_dtype(data[end_col]):
        data[end_col] = pd.to_datetime(data[end_col])
    
    # Extend end times by max_gap for merging
    if max_gap != pd.Timedelta(0):
        data[end_col] = data[end_col] + max_gap
    
    results = []
    
    for id_vals, group in data.groupby(id_cols):
        # Sort by start time
        group = group.sort_values(start_col).reset_index(drop=True)
        
        if len(group) == 0:
            continue
        
        # Merge overlapping intervals
        merged = []
        current_start = group.iloc[0][start_col]
        current_end = group.iloc[0][end_col]
        
        for idx in range(1, len(group)):
            next_start = group.iloc[idx][start_col]
            next_end = group.iloc[idx][end_col]
            
            if next_start <= current_end:
                # Overlapping - extend current interval
                current_end = max(current_end, next_end)
            else:
                # No overlap - save current and start new
                row = {start_col: current_start, end_col: current_end}
                if isinstance(id_vals, tuple):
                    for i, col in enumerate(id_cols):
                        row[col] = id_vals[i]
                else:
                    row[id_cols[0]] = id_vals
                merged.append(row)
                
                current_start = next_start
                current_end = next_end
        
        # Add last interval
        row = {start_col: current_start, end_col: current_end}
        if isinstance(id_vals, tuple):
            for i, col in enumerate(id_cols):
                row[col] = id_vals[i]
        else:
            row[id_cols[0]] = id_vals
        merged.append(row)
        
        results.extend(merged)
    
    if not results:
        return pd.DataFrame(columns=id_cols + [start_col, end_col])
    
    result_df = pd.DataFrame(results)
    
    # Adjust end times back
    if max_gap != pd.Timedelta(0):
        result_df[end_col] = result_df[end_col] - max_gap
    
    return result_df


def group_measurements(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    max_gap: pd.Timedelta = pd.Timedelta(hours=6),
    group_col: str = "grp_var",
) -> pd.DataFrame:
    """Group measurements separated by gaps (R ricu group_measurements).
    
    Creates group IDs for sequences of measurements separated by gaps
    larger than max_gap.
    
    Args:
        data: Input time series DataFrame
        id_cols: ID columns
        index_col: Time index column
        max_gap: Maximum gap within a group
        group_col: Name for output group column
        
    Returns:
        DataFrame with added group column
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1]*5,
        ...     'time': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 01:00',
        ...                             '2020-01-01 10:00', '2020-01-01 11:00', '2020-01-01 12:00']),
        ...     'value': [1, 2, 3, 4, 5]
        ... })
        >>> group_measurements(df, ['id'], 'time', max_gap=pd.Timedelta(hours=6))
        # Returns groups: [1, 1, 2, 2, 2]
    """
    data = data.copy()
    
    # Ensure sorted
    data = data.sort_values(id_cols + [index_col])
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])
    
    def calc_groups(group):
        # Calculate time differences
        time_diffs = group[index_col].diff()
        
        # Mark where gaps exceed max_gap
        large_gaps = time_diffs > max_gap
        
        # Cumulative sum creates group IDs
        group[group_col] = large_gaps.cumsum()
        
        return group
    
    data = data.groupby(id_cols, group_keys=False).apply(calc_groups, include_groups=False)
    
    return data.reset_index(drop=True)


def create_intervals(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    overhang: pd.Timedelta = pd.Timedelta(hours=1),
    max_len: pd.Timedelta = pd.Timedelta(hours=6),
    end_col: str = "endtime",
) -> pd.DataFrame:
    """Create intervals from time series observations (R ricu create_intervals).
    
    For each observation, creates an interval ending at the next observation
    or at overhang/max_len.
    
    Args:
        data: Input time series DataFrame
        id_cols: ID columns
        index_col: Time index column (becomes start time)
        overhang: Default duration for last observation
        max_len: Maximum interval length
        end_col: Name for end time column
        
    Returns:
        DataFrame with added end time column
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1, 1],
        ...     'time': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 02:00', '2020-01-01 10:00']),
        ...     'value': [10, 20, 30]
        ... })
        >>> create_intervals(df, ['id'], 'time')
    """
    data = data.copy()
    
    # Ensure sorted
    data = data.sort_values(id_cols + [index_col])
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])
    
    def calc_endtimes(group):
        # Calculate time to next observation
        next_times = group[index_col].shift(-1)
        durations = next_times - group[index_col]
        
        # For last observation, use overhang
        durations = durations.fillna(overhang)
        
        # Cap at max_len
        durations = durations.clip(upper=max_len)
        
        # Calculate end times
        group[end_col] = group[index_col] + durations
        
        return group
    
    data = data.groupby(id_cols, group_keys=False).apply(calc_endtimes, include_groups=False)
    
    return data.reset_index(drop=True)


# Re-export for compatibility with ICUTable interface
from .table import ICUTable  # noqa: E402


def fill_gaps_table(
    table: ICUTable,
    interval: pd.Timedelta,
    method: str = "ffill",
) -> ICUTable:
    """Fill time gaps in a time series table (ICUTable wrapper).

    Args:
        table: Input ICU table
        interval: Expected time interval between observations
        method: Fill method ('ffill', 'bfill', 'interpolate', or None)

    Returns:
        ICUTable with filled gaps
    """
    if not table.index_column or not table.id_columns:
        return table

    df_filled = fill_gaps(
        table.data,
        list(table.id_columns),
        table.index_column,
        interval,
        method=method,
    )

    return ICUTable(
        data=df_filled,
        id_columns=table.id_columns,
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=table.time_columns,
    )


# 新增函数 - 完全复刻 R ricu


def round_to_interval(
    times: Union[pd.Series, timedelta],
    interval: timedelta
):
    """将时间舍入到最近的间隔 (R ricu re_time)"""
    if isinstance(times, (timedelta, pd.Timedelta)):
        interval_ns = pd.Timedelta(interval).value
        times_ns = pd.Timedelta(times).value
        rounded_ns = np.round(times_ns / interval_ns) * interval_ns
        return pd.Timedelta(rounded_ns, unit='ns')
    
    elif isinstance(times, pd.Series):
        interval_ns = pd.Timedelta(interval).value
        times_ns = times.dt.total_seconds() * 1e9
        rounded_ns = np.round(times_ns / interval_ns) * interval_ns
        return pd.to_timedelta(rounded_ns, unit='ns')
    
    return times


def aggregate_data(
    df: pd.DataFrame,
    by: Union[str, List[str]],
    agg_func: str = 'mean',
    value_col: str = 'value'
) -> pd.DataFrame:
    """聚合数据 (R ricu aggregate)"""
    if isinstance(by, str):
        by = [by]
    
    return df.groupby(by, as_index=False).agg({value_col: agg_func})
