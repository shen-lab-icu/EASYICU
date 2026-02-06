"""Time series utilities for ICU data processing.

This module provides utilities for handling time-indexed data, including
interval alignment, windowing, and time-based transformations.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional, Union, List, Sequence

import pandas as pd
import numpy as np

# 🚀 Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op decorator when numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .table import ICUTable

def _safe_group_apply(grouped, func):
    """Call groupby.apply explicitly keeping group columns in the result."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*DataFrameGroupBy.apply.*', category=FutureWarning)
        try:
            # include_groups=True preserves ID columns for pandas ≥2.1 while
            # maintaining backwards compatibility with earlier versions.
            return grouped.apply(func, include_groups=True)
        except TypeError:  # pandas < 2.1
            return grouped.apply(func)

def change_interval(
    table: ICUTable | pd.DataFrame,
    interval: pd.Timedelta | timedelta = None,
    *,
    new_interval: pd.Timedelta | timedelta = None,
    time_col: str = None,
    aggregation: Optional[str] = None,
    fill_gaps: bool = False,
    fill_method: str = "none",
    copy: bool = True,
    is_window_concept: bool = False,
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
        copy: Whether to copy the input data (default True). Set to False for performance if input can be modified.

    Returns:
        New ICUTable or DataFrame with adjusted time resolution
    """
    # Handle parameter aliases
    if interval is None and new_interval is None:
        raise ValueError("Either 'interval' or 'new_interval' must be specified")

    target_interval = pd.to_timedelta(interval if interval is not None else new_interval)

    def _detect_time_columns(df: pd.DataFrame) -> List[str]:
        # 🔧 FIX 2024-12-17: Match R ricu behavior exactly
        # R ricu's time_vars.data.frame returns ALL difftime columns:
        #   time_vars.data.frame <- function(x) colnames(x)[lgl_ply(x, is_difftime)]
        # This means duration columns (norepi_dur, epi_dur, etc.) ARE included
        # and re_time (floor to interval) is applied to them.
        #
        # Exception: endtime columns for window concepts need to be preserved
        # for expand() to work correctly.
        endtime_patterns = ('endtime', 'end_time', 'stop', 'stoptime', 'end')
        return [
            col
            for col in df.columns
            if (pd.api.types.is_datetime64_any_dtype(df[col])
                # Include ALL timedelta columns (durations) - matches R ricu
                or pd.api.types.is_timedelta64_dtype(df[col]))
            # 🔧 排除 endtime 类型的列，它们用于窗口展开
            and col.lower() not in endtime_patterns
            and not any(col.lower().endswith(pat) for pat in endtime_patterns)
        ]

    def _round_time_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = round_to_interval(df[col], target_interval)
        return df

    # Handle DataFrame input
    if isinstance(table, pd.DataFrame):
        df = table.copy() if copy else table

        detected_time_cols = _detect_time_columns(df)
        primary_time_col = time_col or (detected_time_cols[0] if detected_time_cols else None)

        if primary_time_col is None:
            raise ValueError("time_col must be specified or inferable from DataFrame")

        time_cols = [primary_time_col] + [
            col for col in detected_time_cols if col != primary_time_col
        ]

        df = _round_time_columns(df, time_cols)
        
        if aggregation:
            # Group by time and aggregate
            numeric_cols = df.select_dtypes(include=['number']).columns
            agg_dict = {col: aggregation for col in numeric_cols if col != primary_time_col}
            if agg_dict:
                try:
                    df = df.groupby(primary_time_col, as_index=False).agg(agg_dict)
                except Exception:
                    # 聚合失败时退化为去重，避免抛错
                    df = df.drop_duplicates(subset=[primary_time_col], keep="first")
            else:
                # 无可聚合的数值列，退化为去重
                df = df.drop_duplicates(subset=[primary_time_col], keep="first")
        else:
            # Just drop duplicates
            df = df.drop_duplicates(subset=[primary_time_col], keep="first")
        
        return df
    
    # Handle ICUTable input
    if not table.index_column or table.index_column not in table.data.columns:
        return table

    df = table.data.copy() if copy else table.data
    
    # 🔧 FIX: 窗口概念由调用者明确指定，不再基于列名自动检测
    # 这修复了 AUMC drugitems 表的 'stop' 列被错误识别为窗口结束列的问题
    has_endtime = is_window_concept
    
    endtime_patterns = ('endtime', 'end_time', 'stop', 'stoptime')
    
    time_cols: List[str] = []
    if table.index_column:
        time_cols.append(table.index_column)
    # 🔧 FIX: 从 time_columns 中排除 endtime 类型的列
    for tc in (table.time_columns or []):
        if tc.lower() not in endtime_patterns and not any(tc.lower().endswith(pat) for pat in endtime_patterns):
            if tc not in time_cols:
                time_cols.append(tc)

    detected = _detect_time_columns(df)
    for col in detected:
        if col not in time_cols:
            time_cols.append(col)

    # 🔧 FIX: 窗口概念不取整时间，保留原始值给 expand_interval_rows
    if not has_endtime:
        df = _round_time_columns(df, time_cols)

    # Group by ID columns and rounded time, and aggregate
    group_cols = list(table.id_columns) + [table.index_column]

    # Filter group_cols to only include columns that actually exist in the dataframe
    # This handles cases where ID columns were filtered out during processing (e.
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
    
    # 🔧 对于窗口概念（有 endtime），完全跳过聚合，只做时间取整
    # 聚合将在 expand_interval_rows 展开后进行
    if has_endtime:
        # 窗口概念：只取整时间，不聚合
        # 时间已经在上面的 _round_time_columns 中取整了
        pass
    elif aggregation is False:
        # 🔧 FIX 2024-12-17: When aggregation is explicitly False, skip ALL aggregation/dedup
        # This is critical for vaso60 sub-concepts which need ALL rows (even duplicates)
        # for the callback's own max aggregation logic
        pass
    elif aggregation:
        # 普通概念：正常聚合
        agg_dict = {}
        for col in df.columns:
            if col not in group_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    agg_dict[col] = aggregation
                else:
                    # For non-numeric columns, keep first value
                    agg_dict[col] = 'first'

        if agg_dict:
            # Optimization: If all aggregations are 'first', use drop_duplicates (much faster)
            if all(agg == 'first' for agg in agg_dict.values()):
                df = df.drop_duplicates(subset=group_cols, keep='first')
            else:
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
    
    if fill_gaps:
        fill_fn = globals().get("fill_gaps")
        if not callable(fill_fn):
            raise RuntimeError("fill_gaps helper is not available")
        # 🔧 CRITICAL FIX: Use appropriate fill method based on concept type
        # - "ffill": For medication rate concepts (locf behavior)
        # - "none": For urine/vent_ind (only fill time points, not values)
        df = fill_fn(
            df,
            list(table.id_columns),
            table.index_column,
            target_interval,
            method=fill_method,
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
    admission_times: Optional[pd.DataFrame] = None,
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
        admission_times: DataFrame with id and intime columns for R ricu-compatible
                        floor behavior. When provided, floor is applied to
                        relative time (time - intime), not absolute datetime.
        
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
    
    # 🔧 CRITICAL FIX: Handle numeric time columns (hours since admission)
    # When align_to_admission=True, time columns are converted to float (hours)
    # We need to handle both datetime and numeric time formats
    is_numeric_time = pd.api.types.is_numeric_dtype(data[start_var])
    
    if is_numeric_time:
        # Numeric time - detect if it's in hours or minutes
        # eICU's infusionoffset is in MINUTES (values typically 100-10000)
        # MIIV's charttime aligned is in HOURS (values typically 0-168)
        # We use the step_size unit to determine the data unit
        
        # 🔧 CRITICAL FIX 2025-02-12: Match step_size to data unit
        # If step_size is in minutes (e.g., 1 min), and data appears to be in minutes
        # (values > 24), use minutes. Otherwise use hours.
        is_minute_data = start_var.lower() == 'infusionoffset'
        
        if is_minute_data:
            # Data is in MINUTES, step_size should be in MINUTES
            step_val = step_size.total_seconds() / 60.0
        else:
            # Data is in HOURS, step_size should be in HOURS
            step_val = step_size.total_seconds() / 3600.0
        
        # Determine end column
        if end_var not in data.columns:
            raise ValueError(f"End variable '{end_var}' not found in data")
        
        end_col = end_var
        
        # Filter valid rows (non-NA, start <= end)
        valid_mask = data[start_var].notna() & data[end_col].notna() & (data[start_var] <= data[end_col])
        valid_data = data[valid_mask].copy()
        
        if len(valid_data) == 0:
            result_cols = [start_var] + id_cols + keep_vars
            return pd.DataFrame(columns=result_cols)
        
        # 🔧 CRITICAL FIX 2024-12-16: Match R ricu expand() behavior exactly
        # 
        # R ricu's expand() uses:
        #   seq(start, end, step)  -- using original values, not floored!
        #
        # The key insight is that R's seq(from, to, by) generates:
        #   from, from+by, from+2*by, ... until value exceeds 'to'
        #
        # Example: start=2.595556, end=2.612222, step=1
        #   seq(2.595556, 2.612222, 1) = [2.595556] (only 1 value)
        #   Because 2.595556 + 1 = 3.595556 > 2.612222
        #
        # Example: start=-0.15, end=5.57, step=1
        #   seq(-0.15, 5.57, 1) = [-0.15, 0.85, 1.85, 2.85, 3.85, 4.85] (6 values)
        #   
        # After change_interval/aggregate, values are floored:
        #   floor([-0.15, 0.85, 1.85, 2.85, 3.85, 4.85]) = [-1, 0, 1, 2, 3, 4]
        #
        # 🔧 R ricu clips end < 0 to 0, BUT ONLY for win_tbl when end_var is not present
        # (i.e., when endtime is calculated from duration, not when it's already provided)
        # R ricu code: 
        #   if (is_win_tbl(x) && !end_var %in% colnames(x)) {
        #     x <- x[get(end_var) < 0, c(end_var) := as.difftime(0, units = time_unit)]
        #   }
        # Since expand_intervals always calls create_intervals first (which adds endtime),
        # we should NOT clip end values here - they are already computed correctly.
        #
        # Get original start values (NO floor!)
        start_values = valid_data[start_var].values.copy()
        
        # Get end values - do NOT clip to 0, R ricu only does that for win_tbl without endtime
        end_values = valid_data[end_col].values.copy()
        
        # 🔧 FIX 2024-12-16: Use original end values, NOT floor(end)!
        # R's seq(start, end, step) generates values starting from start,
        # incrementing by step, as long as value <= end.
        # This means seq(2.59, 2.61, 1) = [2.59] (1 value, not 0)
        #
        # Calculate number of points using R's seq() behavior:
        # seq(start, end, step) = [start, start+step, ..., start+n*step]
        # where start + n*step <= end and start + (n+1)*step > end
        # So n = floor((end - start) / step), and count = n + 1
        diff = end_values - start_values
        counts = np.floor(diff / step_val).astype(int) + 1
        counts = np.maximum(counts, 1)  # 🔧 FIX: R seq() always returns at least 1 value when start <= end
        
        # Filter out rows with 0 counts
        row_mask = counts > 0
        if not row_mask.any():
            result_cols = [start_var] + id_cols + keep_vars
            return pd.DataFrame(columns=result_cols)
            
        valid_data = valid_data[row_mask]
        counts = counts[row_mask]
        start_values = start_values[row_mask]
        
        # Repeat rows
        # valid_data.index.repeat(counts) repeats the index, then loc selects rows
        expanded_df = valid_data.loc[valid_data.index.repeat(counts)].reset_index(drop=True)
        
        # Repeat original start times for offset calculation
        start_expanded = np.repeat(start_values, counts)
        
        # Generate time offsets: 0, 1, 2... for each group
        # Using list comprehension with numpy is much faster than pandas iterrows
        offsets = np.concatenate([np.arange(c) for c in counts])
        
        # Calculate new times: start + offset * step (using original start, not floored)
        # The values will be floored later by change_interval
        expanded_df[start_var] = start_expanded + offsets * step_val
        
        # Select columns - remove duplicates while preserving order
        result_cols = [start_var]
        seen = {start_var}
        for col in id_cols + keep_vars:
            if col not in seen:
                result_cols.append(col)
                seen.add(col)
        
        # Ensure columns exist, remove duplicated column names from expanded_df
        if expanded_df.columns.duplicated().any():
            expanded_df = expanded_df.loc[:, ~expanded_df.columns.duplicated()]
        
        available_cols = [c for c in result_cols if c in expanded_df.columns]
        return expanded_df[available_cols]

    
    # 🔧 CRITICAL FIX 2024-12: Match R ricu expand() behavior for datetime
    # 
    # R ricu's expand() uses floor for BOTH numeric and datetime:
    #   seq(floor(start), floor(end), by = step)
    # 
    # For datetime, this means flooring to the hour:
    #   04:41:00 → 04:00:00
    #   05:35:00 → 05:00:00
    #
    # The interval 04:41 to 05:35 should expand to hours 4, 5 (2 points)
    # NOT just the start time 04:41 (1 point)
    #
    # 🔧 CRITICAL FIX 2024-11-30: R ricu floor behavior with admission times
    #
    # R ricu's load_mihi() converts datetime to relative time (difftime) BEFORE callbacks.
    # This means expand() operates on relative hours, and floor() is applied to relative hours.
    #
    # Example (patient 30000484):
    #   - admission: 2136-01-14 17:23:32
    #   - starttime: 2136-01-15 06:39:00 -> relative 13.26 hours -> floor -> 13
    #   - endtime:   2136-01-15 07:00:00 -> relative 13.61 hours -> floor -> 13
    #   - Result: 1 row at hour 13
    #
    # Old pyricu behavior (WRONG):
    #   - starttime: 06:39 -> floor -> 06:00 (absolute)
    #   - After time alignment: 06:00 relative -> 12.61 hours -> hour 12
    #   - Result: 1 row at hour 12 (wrong!)
    #
    # When admission_times is provided, we use relative-time-aware floor.
    
    # Ensure start and end are datetime
    if not pd.api.types.is_datetime64_any_dtype(data[start_var]):
        data = data.copy()
        data[start_var] = pd.to_datetime(data[start_var])
    
    # Handle end_var as duration or absolute time
    if pd.api.types.is_timedelta64_dtype(data[end_var]):
        data['_end_abs'] = data[start_var] + data[end_var]
        end_col = '_end_abs'
    else:
        if not pd.api.types.is_datetime64_any_dtype(data[end_var]):
            data[end_var] = pd.to_datetime(data[end_var])
        end_col = end_var
    
    # Filter valid rows (non-NA, start <= end)
    valid_mask = data[start_var].notna() & data[end_col].notna() & (data[start_var] <= data[end_col])
    valid_data = data[valid_mask].copy()
    
    if len(valid_data) == 0:
        result_cols = [start_var] + id_cols + keep_vars
        return pd.DataFrame(columns=result_cols)
    
    # 🔧 CRITICAL FIX 2024-11-30: Use relative-time-aware floor when admission_times provided
    step_hours = step_size.total_seconds() / 3600.0
    
    if admission_times is not None and not admission_times.empty:
        # Find the ID column to join on
        id_col_for_join = None
        for col in id_cols:
            if col in valid_data.columns and col in admission_times.columns:
                id_col_for_join = col
                break
        
        if id_col_for_join is not None:
            # Merge admission times
            intime_col = 'intime' if 'intime' in admission_times.columns else None
            if intime_col is None:
                for col in admission_times.columns:
                    if 'intime' in col.lower() or 'admittime' in col.lower():
                        intime_col = col
                        break
            
            if intime_col is not None:
                # Merge and calculate relative hours
                valid_data = valid_data.merge(
                    admission_times[[id_col_for_join, intime_col]].drop_duplicates(),
                    on=id_col_for_join,
                    how='left'
                )
                
                # Ensure intime is datetime
                valid_data[intime_col] = pd.to_datetime(valid_data[intime_col])
                
                # Calculate relative hours
                start_rel = (valid_data[start_var] - valid_data[intime_col]).dt.total_seconds() / 3600.0
                end_rel = (valid_data[end_col] - valid_data[intime_col]).dt.total_seconds() / 3600.0
                
                # Floor relative hours (R ricu behavior)
                start_floored_rel = np.floor(start_rel / step_hours) * step_hours
                end_floored_rel = np.floor(end_rel / step_hours) * step_hours
                
                # Calculate counts based on floored relative hours
                diff_hours = end_floored_rel - start_floored_rel
                counts = (diff_hours / step_hours).astype(int) + 1
                counts = np.maximum(counts, 0)
                
                # Filter out rows with 0 counts
                row_mask = counts > 0
                if not row_mask.any():
                    result_cols = [start_var] + id_cols + keep_vars
                    return pd.DataFrame(columns=result_cols)
                    
                valid_data = valid_data[row_mask].copy()
                counts = counts[row_mask]
                start_floored_rel = start_floored_rel[row_mask]
                intime_values = valid_data[intime_col].values
                
                # Repeat rows
                expanded_df = valid_data.loc[valid_data.index.repeat(counts)].reset_index(drop=True)
                
                # Repeat start floored relative hours and intime for offset calculation
                start_floored_rel_expanded = np.repeat(start_floored_rel.values, counts)
                intime_expanded = np.repeat(intime_values, counts)
                
                # Generate time offsets: 0, 1, 2, ... for each row
                offsets = np.concatenate([np.arange(c) for c in counts])
                
                # Calculate new absolute times from floored relative time + intime
                # 🔧 PERFORMANCE FIX: Use numpy timedelta64 instead of pd.to_timedelta
                new_rel_hours = start_floored_rel_expanded + offsets * step_hours
                new_rel_seconds = (new_rel_hours * 3600).astype('timedelta64[s]')
                new_abs_times = pd.to_datetime(intime_expanded) + new_rel_seconds
                expanded_df[start_var] = new_abs_times
                
                # Remove temporary intime column
                if intime_col in expanded_df.columns:
                    expanded_df = expanded_df.drop(columns=[intime_col])
                
                # Select columns - remove duplicates while preserving order
                result_cols = [start_var]
                seen = {start_var}
                for col in id_cols + keep_vars:
                    if col not in seen and col != intime_col:
                        result_cols.append(col)
                        seen.add(col)
                
                # Ensure columns exist, remove duplicated column names from expanded_df
                if expanded_df.columns.duplicated().any():
                    expanded_df = expanded_df.loc[:, ~expanded_df.columns.duplicated()]
                
                available_cols = [c for c in result_cols if c in expanded_df.columns]
                result = expanded_df[available_cols]
                
                # Clean up temporary column
                if '_end_abs' in data.columns:
                    data.drop('_end_abs', axis=1, inplace=True)
                
                return result
    
    # Fallback: Original datetime floor behavior (when no admission_times)
    # 🔧 CRITICAL FIX: Floor both start and end to step boundaries
    # This matches R ricu's seq(floor(start), floor(end), by)
    # For hourly steps, floor to the hour (remove minutes/seconds)
    start_floored = valid_data[start_var].dt.floor(step_size)
    end_floored = valid_data[end_col].dt.floor(step_size)
    
    # Calculate number of points: (floor_end - floor_start) / step + 1
    diff = end_floored - start_floored
    counts = (diff / step_size).astype(int) + 1
    counts = np.maximum(counts, 0)
    
    # Filter out rows with 0 counts
    row_mask = counts > 0
    if not row_mask.any():
        result_cols = [start_var] + id_cols + keep_vars
        return pd.DataFrame(columns=result_cols)
        
    valid_data = valid_data[row_mask]
    counts = counts[row_mask]
    start_floored = start_floored[row_mask]
    
    # Repeat rows
    expanded_df = valid_data.loc[valid_data.index.repeat(counts)].reset_index(drop=True)
    
    # Repeat floored start times for offset calculation
    start_floored_expanded = start_floored.repeat(counts).values
    
    # Generate time offsets: 0, 1, 2, ... for each row
    offsets = np.concatenate([np.arange(c) for c in counts])
    
    # Calculate new times: floored_start + offset * step_size
    # Use floored start time to generate aligned hour boundaries
    # 🔧 PERFORMANCE FIX: Use numpy timedelta64 instead of pd.to_timedelta
    # pd.to_timedelta is very slow for large arrays due to type inference
    step_seconds = int(step_size.total_seconds())
    offsets_td = (offsets * step_seconds).astype('timedelta64[s]')
    expanded_df[start_var] = start_floored_expanded + offsets_td
    
    # Select columns - remove duplicates while preserving order
    result_cols = [start_var]
    seen = {start_var}
    for col in id_cols + keep_vars:
        if col not in seen:
            result_cols.append(col)
            seen.add(col)
    
    # Ensure columns exist, remove duplicated column names from expanded_df
    if expanded_df.columns.duplicated().any():
        expanded_df = expanded_df.loc[:, ~expanded_df.columns.duplicated()]
    
    available_cols = [c for c in result_cols if c in expanded_df.columns]
    result = expanded_df[available_cols]
    
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
    limits: Optional[Any] = None,
    method: str = "none",
) -> pd.DataFrame:
    """Fill time gaps in a time series (R ricu fill_gaps).

    Args:
        data: Input time series DataFrame
        id_cols: ID columns to group by
        index_col: Time index column
        interval: Expected time interval between observations
        limits: Either a DataFrame with per-ID start/end bounds, an
            `ICUTable`, or any length-2 sequence specifying global
            lower/upper bounds (matching ricu's difftime vector support)
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
    # 🚀 优化: 只在需要修改时才复制
    if index_col not in data.columns:
        return data

    interval = pd.to_timedelta(interval)
    if interval <= pd.Timedelta(0):
        raise ValueError("interval must be a positive timedelta")

    id_cols = [col for col in (id_cols or []) if col in data.columns]

    def _normalize_time(series: pd.Series) -> tuple[pd.Series, str]:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, errors="coerce"), "datetime"
        if pd.api.types.is_timedelta64_dtype(series):
            return pd.to_timedelta(series, errors="coerce"), "timedelta"
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce"), "numeric"
        as_dt = pd.to_datetime(series, errors="coerce")
        if as_dt.notna().any():
            return as_dt, "datetime"
        as_td = pd.to_timedelta(series, errors="coerce")
        if as_td.notna().any():
            return as_td, "timedelta"
        return pd.to_numeric(series, errors="coerce"), "numeric"

    def _coerce_limits(values: pd.Series, kind: str) -> pd.Series:
        if kind == "datetime":
            return pd.to_datetime(values, errors="coerce")
        if kind == "timedelta":
            return pd.to_timedelta(values, errors="coerce")
        return pd.to_numeric(values, errors="coerce")

    # Optimization: Pre-build limits lookup dictionary
    limits_lookup = {}

    def _select_limits(id_vals: Any) -> tuple[Any, Any]:
        """Lookup per-id fill limits while tolerating pandas tuple keys."""
        if not limits_lookup:
            return None, None

        if not id_cols:
            entry = limits_lookup.get(None)
            if entry:
                return entry["start"], entry["end"]
            return None, None

        entry = limits_lookup.get(id_vals)
        if entry:
            return entry["start"], entry["end"]

        # pandas >=2.1 emits tuple keys even for single column groupbys when
        # dropna=False; handle both tuple and scalar variants so limits stay in sync
        if isinstance(id_vals, tuple):
            if len(id_vals) == 1:
                entry = limits_lookup.get(id_vals[0])
                if entry:
                    return entry["start"], entry["end"]
        elif len(id_cols) == 1:
            entry = limits_lookup.get((id_vals,))
            if entry:
                return entry["start"], entry["end"]

        return None, None

    time_series, time_kind = _normalize_time(data[index_col])
    data[index_col] = time_series

    def _is_sequence_like(obj: Any) -> bool:
        if isinstance(obj, (pd.Series, pd.Index, np.ndarray)):
            return True
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            return True
        return False

    def _prepare_limits(limits_obj: Any) -> Optional[pd.DataFrame]:
        if limits_obj is None:
            return None
        if isinstance(limits_obj, pd.DataFrame):
            return limits_obj.copy()
        if hasattr(limits_obj, "data") and isinstance(getattr(limits_obj, "data"), pd.DataFrame):
            return getattr(limits_obj, "data").copy()
        if _is_sequence_like(limits_obj):
            seq = list(limits_obj)
            if len(seq) != 2:
                raise ValueError("limits sequences must contain exactly two entries (start, end)")
            if id_cols:
                base = data[id_cols].drop_duplicates().reset_index(drop=True)
            else:
                base = pd.DataFrame(index=[0])
            base = base.copy()
            base["start"] = seq[0]
            base["end"] = seq[1]
            if not id_cols:
                base = base[["start", "end"]]
            return base
        raise TypeError("limits must be a DataFrame, ICUTable, or length-2 sequence")

    limits_df = _prepare_limits(limits)
    if limits_df is not None:
        for required_col in ("start", "end"):
            if required_col not in limits_df.columns:
                raise ValueError("limits must contain 'start' and 'end' columns")
        limits_df["start"] = _coerce_limits(limits_df["start"], time_kind)
        limits_df["end"] = _coerce_limits(limits_df["end"], time_kind)
        
        # Build lookup dictionary for O(1) access
        if id_cols:
            if len(id_cols) == 1:
                limits_lookup = limits_df.set_index(id_cols[0])[['start', 'end']].to_dict('index')
            else:
                limits_lookup = limits_df.set_index(id_cols)[['start', 'end']].to_dict('index')
        else:
            if not limits_df.empty:
                limits_lookup = {None: {'start': limits_df.iloc[0]['start'], 'end': limits_df.iloc[0]['end']}}

    step_hours = interval / pd.Timedelta(hours=1)

    def _build_range(start, end):
        if pd.isna(start) or pd.isna(end) or start > end:
            return None
        if time_kind == "datetime":
            return pd.date_range(start=start, end=end, freq=interval)
        if time_kind == "timedelta":
            return pd.timedelta_range(start=start, end=end, freq=interval)
        if step_hours <= 0:
            raise ValueError("interval must not be zero")
        return np.arange(start, end + step_hours, step_hours)

    def _assign_ids(frame: pd.DataFrame, id_vals: Any) -> pd.DataFrame:
        if not id_cols:
            return frame
        if isinstance(id_vals, tuple):
            for idx, col in enumerate(id_cols):
                frame[col] = id_vals[idx]
        else:
            frame[id_cols[0]] = id_vals
            for extra in id_cols[1:]:
                if extra not in frame.columns:
                    frame[extra] = np.nan
        return frame

    if id_cols:
        grouped = data.groupby(id_cols, dropna=False, sort=False)
    else:
        grouped = [(None, data)]

    filled_groups: List[pd.DataFrame] = []

    for id_vals, group in grouped:
        if group.empty:
            continue
        group = group.sort_values(index_col)
        observed = group[index_col].dropna()
        if observed.empty:
            continue

        min_time = observed.iloc[0]
        max_time = observed.iloc[-1]
        limit_min, limit_max = _select_limits(id_vals)
        if limit_min is not None and limit_max is not None:
            min_time, max_time = limit_min, limit_max

        full_range = _build_range(min_time, max_time)
        if full_range is None or len(full_range) == 0:
            continue

        if isinstance(full_range, np.ndarray):
            reindex_target = pd.Index(full_range, name=index_col)
        else:
            reindex_target = full_range

        group = group.set_index(index_col)

        # Preserve original observations even when limits introduce
        # off-grid timestamps by ensuring the reindex target includes
        # the existing index values.
        if hasattr(reindex_target, "union"):
            target_index = reindex_target.union(group.index)
        else:
            target_index = pd.Index(reindex_target, name=index_col).union(group.index)

        group = group.reindex(target_index)
        group = _assign_ids(group, id_vals)

        if method == "ffill":
            group = group.ffill()
        elif method == "bfill":
            group = group.bfill()
        elif method == "interpolate":
            numeric_cols = group.select_dtypes(include="number").columns
            group[numeric_cols] = group[numeric_cols].interpolate()
        elif method == "zero":
            # Fill numeric columns with 0 (useful for vent_ind, urine)
            numeric_cols = group.select_dtypes(include="number").columns
            # Exclude index column if it's numeric
            cols_to_fill = [c for c in numeric_cols if c != index_col]
            group[cols_to_fill] = group[cols_to_fill].fillna(0)
        elif method == "none":
            # R ricu's "none" method: only fill time gaps, don't fill data values
            # This is the default R behavior when method is not specified
            pass
        else:
            raise ValueError(f"Unknown fill method: {method}")

        group = group.reset_index()
        group.rename(columns={"index": index_col}, inplace=True)
        filled_groups.append(group)

    if not filled_groups:
        return pd.DataFrame(columns=data.columns)

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
        
        data = data.groupby(existing_id_cols, group_keys=False).apply(fill_group, include_groups=True)
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

# 🚀 Numba-accelerated window computation core
@jit(nopython=True, cache=True)
def _compute_window_bounds(times: np.ndarray, before_val: float, after_val: float):
    """Compute window start/end indices for each time point.
    
    Returns:
        window_starts, window_ends: arrays of indices marking window boundaries
    """
    n = len(times)
    window_starts = np.empty(n, dtype=np.int64)
    window_ends = np.empty(n, dtype=np.int64)
    
    for i in range(n):
        center_time = times[i]
        start_time = center_time - before_val
        end_time = center_time + after_val
        
        # Binary search for window start
        left, right = 0, n
        while left < right:
            mid = (left + right) // 2
            if times[mid] < start_time:
                left = mid + 1
            else:
                right = mid
        window_starts[i] = left
        
        # Binary search for window end
        left, right = 0, n
        while left < right:
            mid = (left + right) // 2
            if times[mid] <= end_time:
                left = mid + 1
            else:
                right = mid
        window_ends[i] = left
    
    return window_starts, window_ends

def slide(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    before: pd.Timedelta,
    after: pd.Timedelta = pd.Timedelta(0),
    agg_func: Optional[dict] = None,
    full_window: bool = False,
) -> pd.DataFrame:
    """Apply sliding window aggregation (R ricu slide) - VECTORIZED VERSION.
    
    🚀 Performance: Fully vectorized using pandas rolling() API.
    Expected 2-5x faster than loop-based version.
    
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
                  Supports: 'max', 'min', 'mean', 'sum', 'count', 'std', 
                           'first', 'last', 'max_or_na', 'min_or_na'
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
    
    if len(data) == 0:
        return pd.DataFrame()
    
    # 🚀 优化策略选择：after=0 时使用向量化 rolling，否则用循环
    # rolling() 原生不支持 forward window，仅支持 backward
    if after == pd.Timedelta(0):
        # print(f"🚀 使用向量化 slide (vectorized path)", flush=True)
        return _slide_vectorized(data, id_cols, index_col, before, agg_func, full_window)
    else:
        # print(f"⚠️ 使用循环 slide (loop path, after={after})", flush=True)
        # Fallback to loop-based version for forward windows
        return _slide_loop(data, id_cols, index_col, before, after, agg_func, full_window)

def _slide_vectorized(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    before: pd.Timedelta,
    agg_func: dict,
    full_window: bool = False,
) -> pd.DataFrame:
    """Vectorized slide implementation using pandas rolling() - FAST PATH.
    
    🚀 Uses pandas native rolling() for maximum performance.
    Only works when after=0 (no forward looking window).
    """
    if len(data) == 0:
        return pd.DataFrame()
    
    # Handle both datetime and numeric (hours) time columns
    is_numeric_time = pd.api.types.is_numeric_dtype(data[index_col])
    
    if not is_numeric_time:
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
            # 🚀 优化：避免copy（调用者通常不再使用原数据）
            data[index_col] = pd.to_datetime(data[index_col])
    
    # Convert before to compatible units
    if before is None:
        window_size = 24.0 if is_numeric_time else pd.Timedelta(hours=24)
    elif is_numeric_time:
        window_size = before.total_seconds() / 3600.0
    else:
        window_size = before
    
    # Map string function names to pandas aggregation methods
    agg_map = {}
    for col, func in agg_func.items():
        if col not in data.columns:
            continue
        
        if isinstance(func, str):
            if func in ['max', 'min', 'mean', 'sum', 'std', 'count']:
                agg_map[col] = func
            elif func == 'max_or_na':
                agg_map[col] = 'max'  # pandas max already handles NA
            elif func == 'min_or_na':
                agg_map[col] = 'min'  # pandas min already handles NA
            elif func == 'first':
                agg_map[col] = lambda x: x.iloc[0] if len(x) > 0 else np.nan
            elif func == 'last':
                agg_map[col] = lambda x: x.iloc[-1] if len(x) > 0 else np.nan
            else:
                # Fallback to loop version for unknown functions
                return _slide_loop(data, id_cols, index_col, before, pd.Timedelta(0), agg_func, full_window)
        elif callable(func):
            agg_map[col] = func
        else:
            raise ValueError(f"agg_func values must be string or callable, got {type(func)}")
    
    if not agg_map:
        return pd.DataFrame()
    
    # 🚀 核心优化：使用 pandas rolling() 向量化计算
    results = []
    
    for id_vals, group in data.groupby(id_cols, sort=False):
        if len(group) == 0:
            continue
        
        # Sort by time and set time as index for rolling
        group = group.sort_values(index_col).reset_index(drop=True)
        
        # Set time as index for rolling
        if is_numeric_time:
            # For numeric time (hours), convert to timedelta for rolling
            # print(f"🚀 Numeric time - converting to timedelta for vectorized rolling", flush=True)
            
            # Convert numeric hours to timedelta
            group_with_td = group.copy()
            group_with_td['__temp_time__'] = pd.to_timedelta(group[index_col], unit='h')
            group_indexed = group_with_td.set_index('__temp_time__')
            
            # Convert window size to timedelta
            window_td = pd.Timedelta(hours=window_size)
            
            # Apply rolling aggregation
            rolled = group_indexed.rolling(
                window=window_td,
                closed='both',
                min_periods=1 if not full_window else None
            )
            
            # Compute aggregations
            agg_results = {}
            for col, agg_fn in agg_map.items():
                if col in group_indexed.columns and col != '__temp_time__':
                    if isinstance(agg_fn, str):
                        agg_results[col] = rolled[col].agg(agg_fn)
                    else:
                        agg_results[col] = rolled[col].apply(agg_fn, raw=False)
            
            # Reconstruct result with original index_col
            result_group = pd.DataFrame(agg_results, index=group_indexed.index)
            result_group = result_group.reset_index(drop=True)
            result_group[index_col] = group[index_col].values
            
            # Add ID columns
            for i, id_col in enumerate(id_cols):
                if isinstance(id_vals, tuple):
                    result_group[id_col] = id_vals[i]
                else:
                    result_group[id_col] = id_vals
            
            # Filter for full_window if needed
            if full_window:
                group_start = group[index_col].min()
                group[index_col].max()
                mask = (result_group[index_col] - window_size >= group_start)
                result_group = result_group[mask]
        else:
            # For datetime, use time-based rolling (FAST!)
            group_indexed = group.set_index(index_col)
            
            # Apply rolling aggregation
            rolled = group_indexed.rolling(
                window=window_size, 
                closed='both',  # Include both endpoints: [time-before, time]
                min_periods=1 if not full_window else None
            )
            
            # Compute aggregations
            agg_results = {}
            for col, agg_fn in agg_map.items():
                if col in group_indexed.columns:
                    if isinstance(agg_fn, str):
                        agg_results[col] = rolled[col].agg(agg_fn)
                    else:
                        agg_results[col] = rolled[col].apply(agg_fn, raw=False)
            
            # Reconstruct result with ID columns
            result_group = pd.DataFrame(agg_results, index=group_indexed.index)
            result_group = result_group.reset_index()
            
            # Add ID columns
            for i, id_col in enumerate(id_cols):
                if isinstance(id_vals, tuple):
                    result_group[id_col] = id_vals[i]
                else:
                    result_group[id_col] = id_vals
            
            # Filter for full_window if needed
            if full_window:
                group_start = group_indexed.index.min()
                group_indexed.index.max()
                mask = (result_group[index_col] - window_size >= group_start)
                result_group = result_group[mask]
        
        results.append(result_group)
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)

def _slide_loop(
    data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    before: pd.Timedelta,
    after: pd.Timedelta,
    agg_func: dict,
    full_window: bool = False,
) -> pd.DataFrame:
    """Loop-based slide implementation - FALLBACK for complex cases.
    
    Used when:
    - after != 0 (forward looking window)
    - Custom aggregation functions
    - Numeric time columns
    """
    if len(data) == 0:
        return pd.DataFrame()
    
    # Handle both datetime and numeric (hours) time columns
    is_numeric_time = pd.api.types.is_numeric_dtype(data[index_col])
    
    if not is_numeric_time:
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
            data = data.copy()
            data[index_col] = pd.to_datetime(data[index_col])
    
    # Convert before/after to compatible units
    if before is None:
        before_val = 24.0 if is_numeric_time else pd.Timedelta(hours=24)
    elif is_numeric_time:
        before_val = before.total_seconds() / 3600.0
    else:
        before_val = before

    if after is None:
        after_val = 0.0 if is_numeric_time else pd.Timedelta(0)
    elif is_numeric_time:
        after_val = after.total_seconds() / 3600.0
    else:
        after_val = after
    
    results = []
    
    for id_vals, group in data.groupby(id_cols, sort=False):
        if len(group) == 0:
            continue
        
        result_group = _slide_loop_single_group(
            group, id_cols, index_col, before, after, 
            agg_func, full_window, id_vals, is_numeric_time,
            before_val, after_val
        )
        results.append(result_group)
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)

def _slide_loop_single_group(
    group: pd.DataFrame,
    id_cols: list,
    index_col: str,
    before: pd.Timedelta,
    after: pd.Timedelta,
    agg_func: dict,
    full_window: bool,
    id_vals,
    is_numeric_time: bool,
    before_val = None,
    after_val = None,
) -> pd.DataFrame:
    """Process single group with loop-based sliding window."""
    
    group = group.sort_values(index_col).reset_index(drop=True)
    
    # Convert time column to numeric for comparison
    if is_numeric_time:
        times_numeric = group[index_col].values.astype(np.float64)
        if before_val is None:
            before_numeric = before.total_seconds() / 3600.0 if before else 24.0
        else:
            before_numeric = float(before_val)
        if after_val is None:
            after_numeric = after.total_seconds() / 3600.0 if after else 0.0
        else:
            after_numeric = float(after_val)
    else:
        # Convert datetime to Unix timestamp (float64)
        times_numeric = group[index_col].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        before_numeric = (before_val if before_val else before).total_seconds()
        after_numeric = (after_val if after_val else after).total_seconds()
    
    n = len(times_numeric)
    
    # For full_window check
    if full_window:
        group_start = times_numeric[0]
        group_end = times_numeric[-1]
    
    # 🚀 Numba加速：批量计算所有窗口边界
    window_starts, window_ends = _compute_window_bounds(
        times_numeric, before_numeric, after_numeric
    )
    
    # 遍历每个时间点进行聚合
    results = []
    for i in range(n):
        window_start_idx = window_starts[i]
        window_end_idx = window_ends[i]
        
        # Check if full window is available
        if full_window:
            window_start_time = times_numeric[i] - before_numeric
            window_end_time = times_numeric[i] + after_numeric
            if window_start_time < group_start or window_end_time > group_end:
                continue
        
        if window_start_idx >= window_end_idx:
            continue
        
        # Build result row
        result_row = {index_col: group.iloc[i][index_col]}
        
        # Add ID columns
        for j, col in enumerate(id_cols):
            if isinstance(id_vals, tuple):
                result_row[col] = id_vals[j]
            else:
                result_row[col] = id_vals
        
        # 🚀 优化：直接使用iloc批量索引
        window_data = group.iloc[window_start_idx:window_end_idx]
        
        # Apply aggregations
        for col, func in agg_func.items():
            if col not in group.columns:
                continue
                
            col_data = window_data[col]
            
            # Handle string function names
            if isinstance(func, str):
                if func == 'mean':
                    result_row[col] = col_data.mean()
                elif func == 'sum':
                    result_row[col] = col_data.sum()
                elif func == 'min':
                    result_row[col] = col_data.min()
                elif func == 'max':
                    result_row[col] = col_data.max()
                elif func == 'count':
                    result_row[col] = col_data.count()
                elif func == 'std':
                    result_row[col] = col_data.std()
                elif func == 'first':
                    result_row[col] = col_data.iloc[0] if len(col_data) > 0 else None
                elif func == 'last':
                    result_row[col] = col_data.iloc[-1] if len(col_data) > 0 else None
                elif func == 'max_or_na':
                    # 🚀 优化：直接用 max(skipna=True)，避免函数调用开销
                    result = col_data.max(skipna=True)
                    result_row[col] = result if not pd.isna(result) else np.nan
                elif func == 'min_or_na':
                    # 🚀 优化：直接用 min(skipna=True)，避免函数调用开销
                    result = col_data.min(skipna=True)
                    result_row[col] = result if not pd.isna(result) else np.nan
                else:
                    raise ValueError(f"Unknown aggregation function: {func}")
            elif callable(func):
                result_row[col] = func(col_data)
            else:
                raise ValueError(f"agg_func values must be string or callable, got {type(func)}")
        
        results.append(result_row)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)

# Remove old duplicated code below
def _old_slide_code_placeholder():
    """This is a placeholder to mark where old code was removed."""
    pass

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
        
        data = data.groupby(existing_id_cols, group_keys=False).apply(fill_group, include_groups=True)
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
        
        data = data.groupby(existing_id_cols, group_keys=False).apply(fill_group, include_groups=True)
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
    if not id_cols:
        raise ValueError("group_measurements requires at least one id column")

    max_gap = pd.to_timedelta(max_gap)

    data = data.copy()

    # Ensure sorted for predictable diff behaviour
    data = data.sort_values(id_cols + [index_col])

    # Ensure datetime comparisons
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])

    indexed = data.set_index(id_cols)
    level_order = list(range(len(id_cols)))

    time_diffs = indexed.groupby(level=level_order, sort=False)[index_col].diff()
    gap_flags = (time_diffs > max_gap).fillna(False).astype(int)
    group_ids = gap_flags.groupby(level=level_order, sort=False).cumsum()

    indexed[group_col] = group_ids

    result = indexed.reset_index()
    return result.reset_index(drop=True)

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
    if not id_cols:
        raise ValueError("create_intervals requires at least one id column")

    overhang = pd.to_timedelta(overhang)
    max_len = pd.to_timedelta(max_len)

    data = data.copy()

    # Ensure sorted for predictable shift semantics
    data = data.sort_values(id_cols + [index_col])

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
        data[index_col] = pd.to_datetime(data[index_col])

    indexed = data.set_index(id_cols)
    level_order = list(range(len(id_cols)))

    next_times = indexed.groupby(level=level_order, sort=False)[index_col].shift(-1)
    durations = next_times - indexed[index_col]

    # For last observation, use overhang
    durations = durations.fillna(overhang)

    # Cap at max_len
    durations = durations.clip(upper=max_len)

    indexed[end_col] = indexed[index_col] + durations

    result = indexed.reset_index()
    return result.reset_index(drop=True)

# Re-export for compatibility with ICUTable interface

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
    times: Union[pd.Series, pd.Index, pd.Timestamp, pd.Timedelta, timedelta, float, int],
    interval: timedelta,
) -> Union[pd.Series, pd.Index, pd.Timestamp, pd.Timedelta, float, int]:
    """Floor timestamps/durations/numerics to the nearest interval (ricu ``re_time``)."""

    if interval is None:
        return times

    interval_td = pd.to_timedelta(interval)
    if interval_td <= pd.Timedelta(0):
        raise ValueError("interval must be a positive timedelta")

    step_ns = interval_td.value
    step_hours = interval_td / pd.Timedelta(hours=1)

    def _floor_ns(int_values: np.ndarray) -> np.ndarray:
        return (int_values // step_ns) * step_ns

    def _floor_numeric(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if step_hours == 0:
            raise ValueError("interval must not be zero")
        floored = np.floor(numeric / step_hours) * step_hours
        res = pd.Series(floored, index=values.index, name=values.name)
        res[numeric.isna()] = np.nan
        return res

    def _series_from(values: np.ndarray, template: pd.Series, converter) -> pd.Series:
        series = converter(values)
        return pd.Series(series, index=template.index, name=template.name)

    if isinstance(times, pd.Series):
        if times.empty:
            return times

        if pd.api.types.is_datetime64_any_dtype(times):
            # 🚀 Optimization: Use dt.floor for datetime series (much faster)
            return times.dt.floor(interval_td)

        if pd.api.types.is_timedelta64_dtype(times):
            # 🚀 Optimization: Use dt.floor for timedelta series
            return times.dt.floor(interval_td)

        if pd.api.types.is_numeric_dtype(times):
            return _floor_numeric(times)

        as_dt = pd.to_datetime(times, errors="coerce")
        if as_dt.notna().any():
            # 🚀 Optimization: Use dt.floor for mixed/object datetime series
            result = as_dt.dt.floor(interval_td)
            result[as_dt.isna()] = pd.NaT
            return result

        as_td = pd.to_timedelta(times, errors="coerce")
        if as_td.notna().any():
            # 🚀 Optimization: Use dt.floor for mixed/object timedelta series
            result = as_td.dt.floor(interval_td)
            result[as_td.isna()] = pd.NaT
            return result

        return times

    if isinstance(times, pd.DatetimeIndex):
        floored = _floor_ns(times.astype("int64", copy=False))
        return pd.to_datetime(floored)

    if isinstance(times, pd.TimedeltaIndex):
        floored = _floor_ns(times.astype("int64", copy=False))
        return pd.to_timedelta(floored, unit="ns")

    if isinstance(times, pd.Timestamp):
        return pd.Timestamp(_floor_ns(np.array([times.value]))[0])

    if isinstance(times, (pd.Timedelta, timedelta)):
        td = pd.to_timedelta(times)
        return pd.to_timedelta(_floor_ns(np.array([td.value]))[0], unit="ns")

    if isinstance(times, (float, int)):
        if step_hours == 0:
            raise ValueError("interval must not be zero")
        return float(np.floor(times / step_hours) * step_hours)

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
