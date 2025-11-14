"""Item callback utilities (R ricu callback-itm.R).

Provides function factories and utilities for creating item callback functions
that handle data transformations during concept loading.
"""

from typing import Callable, Union, Optional, Dict, Any, List
import logging
import re
import operator
import pandas as pd
import numpy as np


def transform_fun(func: Callable, **kwargs) -> Callable:
    """Create a callback that transforms the value column (R ricu transform_fun).
    
    Args:
        func: Function to apply to values
        **kwargs: Additional arguments passed to func
        
    Returns:
        Callback function
        
    Examples:
        >>> # Divide by 2
        >>> divide_2 = transform_fun(lambda x: x / 2)
        >>> 
        >>> # Subtract 3
        >>> subtract_3 = transform_fun(lambda x: x - 3)
    """
    def callback(data: pd.DataFrame, val_col: str = 'value', **cb_kwargs) -> pd.DataFrame:
        if val_col not in data.columns:
            return data
        
        data = data.copy()
        data[val_col] = func(data[val_col], **kwargs)
        return data
    
    return callback


def binary_op(op: Callable, y: Any) -> Callable:
    """Create a binary operation function (R ricu binary_op).

    Args:
        op: Binary operator function (e.g., operator.add, operator.mul)
        y: Second operand

    Returns:
        Unary function that applies op(x, y)

    Examples:
        >>> import operator
        >>> times_2 = binary_op(operator.mul, 2)
        >>> times_2(5)  # Returns 10
    """
    def safe_binary_op(x: Any) -> Any:
        # Handle None values and ensure numeric types for division
        if x is None:
            return None

        # Convert to numeric if needed for division operations
        if op in (operator.truediv, operator.floordiv):
            try:
                x = pd.to_numeric(x, errors='coerce')
                y_val = pd.to_numeric(y, errors='coerce')
                if pd.isna(x) or pd.isna(y_val):
                    return None
                # Special handling for division by zero
                if y_val == 0:
                    return None
                return op(x, y_val)
            except (ValueError, TypeError):
                return None
        else:
            try:
                return op(x, y)
            except (TypeError, ZeroDivisionError):
                return None

    return safe_binary_op


def comp_na(op: Callable, y: Any) -> Callable:
    """Create a comparison that handles NA values (R ricu comp_na).
    
    Args:
        op: Comparison operator (e.g., operator.gt, operator.eq)
        y: Value to compare against
        
    Returns:
        Function that returns False for NA, op(x, y) otherwise
        
    Examples:
        >>> import operator
        >>> gte_4 = comp_na(operator.ge, 4)
        >>> gte_4(pd.Series([1, 4, 5, np.nan]))
        # Returns: [False, True, True, False]
    """
    def compare(x):
        if isinstance(x, pd.Series):
            return ~x.isna() & op(x, y)
        elif isinstance(x, np.ndarray):
            return ~pd.isna(x) & op(x, y)
        elif pd.isna(x):
            return False
        else:
            return op(x, y)
    
    return compare


def set_val(val: Any) -> Callable:
    """Create a function that sets all values to a constant (R ricu set_val).
    
    Args:
        val: Value to set
        
    Returns:
        Function that replaces all values with val
        
    Examples:
        >>> set_true = set_val(True)
        >>> set_true(pd.Series([1, 2, 3]))
        # Returns: [True, True, True]
    """
    def setter(x):
        if isinstance(x, pd.Series):
            return pd.Series([val] * len(x), index=x.index)
        elif isinstance(x, np.ndarray):
            return np.full_like(x, val)
        else:
            return val
    
    return setter


def apply_map(mapping: Dict[Any, Any], var: str = 'val_col') -> Callable:
    """Create a callback that maps values (R ricu apply_map).
    
    Args:
        mapping: Dictionary mapping old values to new values
        var: Name of the parameter containing the column name to map
        
    Returns:
        Callback function
        
    Examples:
        >>> # Map numeric codes to labels
        >>> code_map = apply_map({1: 'male', 2: 'female'})
        >>> df = pd.DataFrame({'sex_code': [1, 2, 1]})
        >>> code_map(df, val_col='sex_code')
    """
    def callback(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        col_name = kwargs.get(var, 'value')

        if col_name not in data.columns:
            return data

        data = data.copy()
        values = data[col_name]

        # First try direct mapping with the original keys
        mapped = values.map(mapping)

        # Fallback: many ricu mappings specify keys as strings even when the
        # source column is numeric (e.g. itemid == 225792). Try again using the
        # string representation of the source values.
        needs_str = mapped.isna() & values.notna()
        if needs_str.any():
            str_mapping = {str(k): v for k, v in mapping.items()}
            str_mapped = values.astype(str).map(str_mapping)
            mapped = mapped.where(~needs_str, str_mapped)

        # Identify rows where a mapping actually existed (either numeric or string)
        direct_keys = pd.Index(mapping.keys())
        str_keys = pd.Index([str(k) for k in mapping.keys()])
        mask = values.isin(direct_keys) | values.astype(str).isin(str_keys)

        if mask.any():
            # Cast to object to allow inserting strings into numeric columns
            if data[col_name].dtype != object:
                data[col_name] = data[col_name].astype(object)
            data.loc[mask, col_name] = mapped[mask]

            # Check if mapping values are numeric and ensure float type to match ricu.R
            mapped_values = [v for v in mapping.values() if isinstance(v, (int, float))]
            if mapped_values and all(isinstance(v, (int, float)) for v in mapping.values()):
                # For pure numeric mappings, ensure float type
                try:
                    data[col_name] = pd.to_numeric(data[col_name], errors='coerce').astype(float)
                except:
                    # Keep as is if conversion fails
                    pass
        return data
    
    return callback


def convert_unit(
    func: Union[Callable, list],
    new_unit: Union[str, list],
    regex: Optional[Union[str, list]] = None,
    ignore_case: bool = True,
) -> Callable:
    """Create a callback for unit conversion (R ricu convert_unit).
    
    Args:
        func: Conversion function(s)
        new_unit: New unit name(s) after conversion
        regex: Regex pattern(s) to match current units (None = all)
        ignore_case: Whether to ignore case in regex matching
        
    Returns:
        Callback function
        
    Examples:
        >>> # Convert Fahrenheit to Celsius
        >>> f_to_c = convert_unit(
        ...     func=lambda x: (x - 32) * 5/9,
        ...     new_unit='degC',
        ...     regex='degF'
        ... )
    """
    # Normalize to lists
    if not isinstance(func, list):
        func = [func]
    if not isinstance(new_unit, list):
        new_unit = [new_unit]
    if regex is not None and not isinstance(regex, list):
        regex = [regex]
    
    if regex is None:
        regex = [None] * len(func)
    
    def callback(
        data: pd.DataFrame,
        val_col: str = 'value',
        unit_col: str = 'unit',
        **kwargs
    ) -> pd.DataFrame:
        if val_col not in data.columns:
            if 'valuenum' in data.columns:
                val_col = 'valuenum'
            else:
                return data
        if unit_col not in data.columns:
            if 'valueuom' in data.columns:
                unit_col = 'valueuom'
            else:
                unit_col = None
        
        data = data.copy()
        
        for f, new_u, rgx in zip(func, new_unit, regex):
            if rgx is None:
                # Apply to all rows
                data[val_col] = f(data[val_col])
                if unit_col:
                    data[unit_col] = new_u
            else:
                # Apply to matching rows
                if unit_col is None:
                    break
                mask = data[unit_col].str.contains(
                    rgx, case=not ignore_case, na=False, regex=True
                )
                data.loc[mask, val_col] = f(data.loc[mask, val_col])
                if unit_col:
                    data.loc[mask, unit_col] = new_u
        
        return data
    
    return callback


def combine_callbacks(*callbacks: Callable) -> Callable:
    """Combine multiple callbacks into one (R ricu combine_callbacks).
    
    Args:
        *callbacks: Callback functions to combine
        
    Returns:
        Combined callback function
        
    Examples:
        >>> cb1 = transform_fun(lambda x: x * 2)
        >>> cb2 = transform_fun(lambda x: x + 1)
        >>> combined = combine_callbacks(cb1, cb2)
        >>> # Applies cb1, then cb2
    """
    def combined_callback(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for callback in callbacks:
            data = callback(data, **kwargs)
        return data
    
    return combined_callback


# Common transformations
def fahr_to_cels(temp: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """Convert Fahrenheit to Celsius."""
    return (temp - 32) * 5 / 9


def silent_as_numeric(x: pd.Series) -> pd.Series:
    """Convert to numeric, suppressing warnings."""
    return pd.to_numeric(x, errors='coerce')


def force_type(target_type: str) -> Callable:
    """Create a function that forces type conversion.
    
    Args:
        target_type: Target type ('int', 'float', 'str', 'bool')
        
    Returns:
        Conversion function
    """
    type_map = {
        'int': lambda x: pd.to_numeric(x, errors='coerce').astype('Int64'),
        'float': lambda x: pd.to_numeric(x, errors='coerce'),
        'str': lambda x: x.astype(str),
        'bool': lambda x: x.astype(bool),
    }
    
    if target_type not in type_map:
        raise ValueError(f"Unknown type: {target_type}")
    
    return type_map[target_type]


# Database-specific helpers
def eicu_age(data: pd.DataFrame, val_col: str = 'age', **kwargs) -> pd.DataFrame:
    """Process eICU age (handles '> 89')."""
    data = data.copy()
    data[val_col] = data[val_col].replace('> 89', '90')
    data[val_col] = pd.to_numeric(data[val_col], errors='coerce')
    # Ensure float type to match ricu.R
    data[val_col] = data[val_col].astype(float)
    return data


def mimic_age(data: pd.DataFrame, val_col: str = 'age', **kwargs) -> pd.DataFrame:
    """Process MIMIC age (convert from days, cap at 90)."""
    data = data.copy()
    # Convert from days to years
    data[val_col] = data[val_col] / 365.25
    # Cap at 90
    data[val_col] = data[val_col].clip(upper=90)
    return data


def percent_as_numeric(x: Union[str, pd.Series]) -> Union[float, pd.Series]:
    """Convert percent strings to numeric (e.g., '50%' -> 50)."""
    if isinstance(x, pd.Series):
        return x.str.replace('%', '').astype(float)
    else:
        return float(str(x).replace('%', ''))


def distribute_amount(
    data: pd.DataFrame,
    val_col: str = 'value',
    unit_col: str = 'unit',
    end_col: str = 'endtime',
    index_col: str = 'time',
    **kwargs
) -> pd.DataFrame:
    """Distribute total amount over duration to get rate.
    
    For drug administrations given as total amount over a duration,
    converts to rate per hour.
    """
    data = data.copy()
    
    # Calculate duration in hours
    # 🔧 FIX: Handle None values to prevent division errors
    time_diff = pd.to_datetime(data[end_col], errors='coerce') - pd.to_datetime(data[index_col], errors='coerce')
    duration = time_diff.dt.total_seconds() / 3600
    
    # Avoid division by zero
    duration = duration.replace(0, 1)
    
    # Calculate rate
    # 🔧 FIX: Handle None values to prevent division errors
    data[val_col] = pd.to_numeric(data[val_col], errors='coerce') / duration
    data[unit_col] = data[unit_col] + '/hr'
    
    return data


def aggregate_fun(agg_func: str, new_unit: str) -> Callable:
    """Create aggregation callback.
    
    Args:
        agg_func: Aggregation function ('sum', 'mean', 'max', 'min')
        new_unit: Unit after aggregation
        
    Returns:
        Callback function
    """
    def callback(
        data: pd.DataFrame,
        val_col: str = 'value',
        unit_col: str = 'unit',
        id_cols: list = None,
        **kwargs
    ) -> pd.DataFrame:
        if id_cols is None:
            id_cols = [c for c in data.columns if 'id' in c.lower()]
        
        agg_dict = {val_col: agg_func}
        result = data.groupby(id_cols, as_index=False).agg(agg_dict)
        result[unit_col] = new_unit
        
        return result
    
    return callback


def fwd_concept(concept_name: str) -> Callable:
    """Forward reference to another concept (R ricu fwd_concept).
    
    Returns a callback that retrieves a previously loaded concept from
    the data dictionary. This allows concepts to reference other concepts
    in their definitions.
    
    Args:
        concept_name: Name of the concept to forward reference
        
    Returns:
        Callback function that retrieves the concept
        
    Examples:
        >>> # In concept definition:
        >>> # "callback": "fwd_concept('mech_vent')"
        >>> cb = fwd_concept('mech_vent')
        >>> # Later during loading:
        >>> result = cb(data_dict={'mech_vent': df_mech_vent})
    """
    def _fwd_callback(data_dict: dict, **kwargs) -> pd.DataFrame:
        """Retrieve referenced concept from data dictionary.
        
        Args:
            data_dict: Dictionary of already loaded concepts
            **kwargs: Additional arguments (ignored)
            
        Returns:
            The referenced concept DataFrame
            
        Raises:
            ValueError: If referenced concept not found
        """
        if concept_name not in data_dict:
            raise ValueError(
                f"Concept '{concept_name}' not found in data_dict. "
                f"Available concepts: {list(data_dict.keys())}"
            )
        return data_dict[concept_name]
    
    return _fwd_callback


def ts_to_win_tbl(
    duration_col: str = 'duration',
    start_col: str = 'start',
    end_col: str = 'end',
) -> Callable:
    """Convert time series table to window table (R ricu ts_to_win_tbl).
    
    Creates a callback that adds duration/window columns to a time series,
    converting it to a windowed format suitable for interval-based queries.
    
    Args:
        duration_col: Name of duration column to create
        start_col: Name of start time column
        end_col: Name of end time column
        
    Returns:
        Callback function
        
    Examples:
        >>> # Convert drug administration to windows
        >>> cb = ts_to_win_tbl(duration_col='drug_duration')
        >>> result = cb(drug_data, index_col='time')
    """
    def _ts_to_win_callback(
        data: pd.DataFrame,
        index_col: str = 'time',
        **kwargs
    ) -> pd.DataFrame:
        """Add window columns to time series data.
        
        Args:
            data: Input time series DataFrame
            index_col: Time index column name
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with added window columns
        """
        data = data.copy()
        
        # If duration_col already exists, use it to compute end
        if duration_col in data.columns:
            if pd.api.types.is_timedelta64_dtype(data[duration_col]):
                data[end_col] = data[index_col] + data[duration_col]
            else:
                # Assume duration is already in same units as time
                data[end_col] = data[duration_col]
        else:
            # Create duration from start/end if they exist
            if start_col in data.columns and end_col in data.columns:
                data[duration_col] = data[end_col] - data[start_col]
            elif start_col in data.columns:
                data[duration_col] = data[index_col] - data[start_col]
            elif end_col in data.columns:
                data[duration_col] = data[end_col] - data[index_col]
        
        # Ensure start column exists
        if start_col not in data.columns and index_col in data.columns:
            data[start_col] = data[index_col]
        
        return data
    
    return _ts_to_win_callback


def locf(max_gap: Optional[pd.Timedelta] = None) -> Callable:
    """Last observation carried forward (R ricu locf).
    
    Creates a callback that performs forward filling of missing values,
    optionally limiting the maximum gap to fill.
    
    Args:
        max_gap: Maximum time gap to fill (None = unlimited)
        
    Returns:
        Callback function
        
    Examples:
        >>> # Fill gaps up to 4 hours
        >>> cb = locf(max_gap=pd.Timedelta(hours=4))
        >>> result = cb(data, index_col='time', val_col='hr')
    """
    def _locf_callback(
        data: pd.DataFrame,
        val_col: str = 'value',
        index_col: Optional[str] = None,
        id_cols: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Apply last observation carried forward.
        
        Args:
            data: Input DataFrame
            val_col: Value column to fill
            index_col: Time index column (for gap checking)
            id_cols: ID columns for grouping
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with forward filled values
        """
        data = data.copy()
        
        if id_cols:
            # Group by IDs and fill within each group
            def fill_group(group):
                if max_gap is not None and index_col is not None:
                    # Fill only within max_gap
                    group = group.sort_values(index_col)
                    time_diff = group[index_col].diff()
                    
                    # Create mask for fillable positions
                    can_fill = time_diff <= max_gap
                    
                    # Forward fill with limit
                    filled = group[val_col].fillna(method='ffill')
                    group[val_col] = group[val_col].where(can_fill, filled)
                else:
                    # Simple forward fill
                    group[val_col] = group[val_col].fillna(method='ffill')
                
                return group
            
            data = data.groupby(id_cols).apply(fill_group, include_groups=False).reset_index(drop=True)
        else:
            # Simple forward fill
            data[val_col] = data[val_col].fillna(method='ffill')
        
        return data
    
    return _locf_callback


def locb(max_gap: Optional[pd.Timedelta] = None) -> Callable:
    """Last observation carried backward (R ricu locb).
    
    Creates a callback that performs backward filling of missing values,
    optionally limiting the maximum gap to fill.
    
    Args:
        max_gap: Maximum time gap to fill (None = unlimited)
        
    Returns:
        Callback function
    """
    def _locb_callback(
        data: pd.DataFrame,
        val_col: str = 'value',
        index_col: Optional[str] = None,
        id_cols: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Apply last observation carried backward.
        
        Args:
            data: Input DataFrame
            val_col: Value column to fill
            index_col: Time index column (for gap checking)
            id_cols: ID columns for grouping
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with backward filled values
        """
        data = data.copy()
        
        if id_cols:
            # Group by IDs and fill within each group
            def fill_group(group):
                if max_gap is not None and index_col is not None:
                    # Fill only within max_gap
                    group = group.sort_values(index_col)
                    time_diff = group[index_col].diff(-1).abs()
                    
                    # Create mask for fillable positions
                    can_fill = time_diff <= max_gap
                    
                    # Backward fill with limit
                    filled = group[val_col].fillna(method='bfill')
                    group[val_col] = group[val_col].where(can_fill, filled)
                else:
                    # Simple backward fill
                    group[val_col] = group[val_col].fillna(method='bfill')
                
                return group
            
            data = data.groupby(id_cols).apply(fill_group, include_groups=False).reset_index(drop=True)
        else:
            # Simple backward fill
            data[val_col] = data[val_col].fillna(method='bfill')
        
        return data
    
    return _locb_callback


def vent_flag(
    data: pd.DataFrame,
    val_col: str = "value",
    index_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    **kwargs,
) -> pd.DataFrame:
    """Filter to ventilated rows and coerce value column to boolean."""

    if val_col not in data.columns:
        return data.copy()

    frame = data.copy()
    mask = frame[val_col].astype(bool)
    frame = frame.loc[mask].copy()
    frame[val_col] = True

    # Ensure index/id columns are preserved if not already present after filtering
    if index_var and index_var not in frame.columns and index_var in data.columns:
        frame[index_var] = data.loc[frame.index, index_var]

    if id_cols:
        for col in id_cols:
            if col not in frame.columns and col in data.columns:
                frame[col] = data.loc[frame.index, col]

    return frame


def eicu_duration_callback(gap_length: pd.Timedelta) -> Callable:
    """Create callback equivalent to R's eicu_duration(gap_length)."""
    from .ts_utils import group_measurements

    if not isinstance(gap_length, pd.Timedelta):
        gap_length = pd.to_timedelta(gap_length)

    def _callback(
        data: pd.DataFrame,
        val_col: str = "value",
        index_var: Optional[str] = None,
        id_cols: Optional[list] = None,
        group_col: str = "__grp",
        **kwargs,
    ) -> pd.DataFrame:
        if data.empty:
            return data.copy()

        frame = data.copy()

        if id_cols is None or not id_cols:
            # Find ID columns, but for eICU, prioritize patient-level IDs
            # Check for patientunitstayid first (eICU specific)
            if "patientunitstayid" in frame.columns:
                id_cols = ["patientunitstayid"]
            else:
                # Fall back to general ID column search
                id_cols = [col for col in frame.columns if "id" in col.lower()]

        # For eICU infusion data, if no ID columns exist, create a dummy one
        # This allows the callback to work even when ID columns were filtered out
        if not any(col in frame.columns for col in id_cols) and not frame.empty:
            import logging
            logging.debug(f"No ID columns found in eICU duration callback. Available columns: {list(frame.columns)}. Creating dummy grouping for duration calculation.")
            # Use a constant group ID for all rows (treat as single patient/time series)
            frame["__dummy_patient_id"] = 1
            id_cols = ["__dummy_patient_id"]

        if index_var is None or index_var not in frame.columns:
            # eICU uses 'offset' columns, other databases use 'time' columns
            time_cols = [col for col in frame.columns if "time" in col.lower() or "offset" in col.lower()]
            if not time_cols:
                raise ValueError("Cannot determine time column for eICU duration callback")
            index_var = time_cols[0]

        # eICU offset is numeric (minutes), datetime conversion will be handled later if needed
        # Just ensure it's numeric and not null
        if 'offset' in index_var.lower():
            # offset is already numeric (minutes from ICU admission)
            frame[index_var] = pd.to_numeric(frame[index_var], errors="coerce")
        else:
            # Other databases use datetime
            frame[index_var] = pd.to_datetime(frame[index_var], errors="coerce")
        
        frame = frame.dropna(subset=[index_var])
        
        if frame.empty:
            return frame

        # Add group column using group_measurements
        grouped = group_measurements(
            frame,
            id_cols=id_cols,
            index_col=index_var,
            max_gap=gap_length,
            group_col=group_col,
        )

        # Calculate duration per group (R calc_dur logic)
        # For each id_cols + group_col group, calculate max(time) - min(time)
        # Make sure all ID columns actually exist in the frame
        valid_id_cols = []
        for col in id_cols:
            if col in frame.columns:
                valid_id_cols.append(col)
            else:
                # If an expected ID column is missing, log it and continue
                # This can happen in eICU where some ID columns are filtered out during processing
                import logging
                logging.warning(f"Expected ID column '{col}' not found in data frame. Available columns: {list(frame.columns)}")

        if not valid_id_cols:
            raise ValueError("No valid ID columns found in data frame")

        groupby_cols = valid_id_cols + [group_col]

        # Ensure all groupby columns exist in the grouped dataframe
        existing_groupby_cols = []
        for col in groupby_cols:
            if col in grouped.columns:
                existing_groupby_cols.append(col)
            else:
                import logging
                logging.debug(f"GroupBy column '{col}' not found in grouped data. Available columns: {list(grouped.columns)}")

        if not existing_groupby_cols:
            raise ValueError("No valid GroupBy columns found in grouped data frame")

        result = grouped.groupby(existing_groupby_cols).agg({
            index_var: ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        result.columns = existing_groupby_cols + ['min_time', 'max_time']
        
        # Calculate duration
        if 'offset' in index_var.lower():
            # For offset (numeric minutes), duration is direct subtraction
            result[val_col] = result['max_time'] - result['min_time']
            # Use min_time as the time point for this duration measurement
            result[index_var] = result['min_time']
        else:
            # For datetime, duration is timedelta converted to minutes
            # 🔧 FIX: Handle None values to prevent division errors
            time_diff = pd.to_datetime(result['max_time'], errors='coerce') - pd.to_datetime(result['min_time'], errors='coerce')
            result[val_col] = time_diff.dt.total_seconds() / 60
            result[index_var] = result['min_time']
        
        # Keep existing_groupby_cols + index_var + val_col (drop group_col, min_time, max_time)
        # But only keep columns that actually exist
        final_cols = []
        for col in existing_groupby_cols:
            if col in result.columns:
                final_cols.append(col)

        # Add index_var and val_col if they exist and are not the same as existing columns
        for col in [index_var, val_col]:
            if col in result.columns and col not in final_cols:
                final_cols.append(col)

        if final_cols:
            result = result[final_cols]
        else:
            # If no valid columns, return empty dataframe with correct structure
            result = pd.DataFrame(columns=[index_var, val_col])
        
        return result

    return _callback


def mimic_rate_mv(
    data: pd.DataFrame,
    val_col: str = 'value',
    unit_col: Optional[str] = None,
    stop_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """MIMIC MetaVision infusion rate callback (R ricu mimic_rate_mv).
    
    Expands inputevents_mv data with start/stop times into time series.
    This is used for continuous infusion medications like vasopressors.
    
    Args:
        data: Input DataFrame with infusion data
        val_col: Value column (infusion rate)
        unit_col: Unit column (rate units)
        stop_var: End time variable for expansion
        id_cols: ID columns for grouping
        **kwargs: Additional arguments
        
    Returns:
        Expanded DataFrame with time series data
        
    Note:
        This is a simplified version that expands intervals.
        In ricu, it calls expand(x, index_var(x), stop_var, keep_vars = ...)
    """
    # Handle empty data - preserve column structure
    if data.empty:
        return data
    
    from .ts_utils import expand
    
    # Infer ID columns if not provided
    if id_cols is None:
        id_cols = [col for col in data.columns if 'id' in col.lower()]
    
    # Infer index variable (time column)
    time_cols = [col for col in data.columns if 'time' in col.lower() and col != stop_var]
    if not time_cols:
        # Fallback to common names
        time_cols = [col for col in ['charttime', 'starttime'] if col in data.columns]
    
    if not time_cols:
        # No time column found, return as-is
        return data
    
    index_var = time_cols[0]
    
    # Prepare keep_vars
    keep_vars = list(id_cols) + [val_col]
    if unit_col and unit_col in data.columns:
        keep_vars.append(unit_col)
    
    # 确保 index_var (starttime) 被保留在 keep_vars 中，因为它是时间索引
    if index_var not in keep_vars:
        keep_vars.append(index_var)
    
    # Remove duplicates
    keep_vars = [col for col in keep_vars if col in data.columns]
    
    # Expand intervals if stop_var exists
    if stop_var and stop_var in data.columns:
        # Expand with 1-hour steps (standard for ICU data)
        step_size = pd.Timedelta(hours=1)
        expanded = expand(
            data,
            start_var=index_var,
            end_var=stop_var,
            step_size=step_size,
            id_cols=id_cols,
            keep_vars=keep_vars
        )
        return expanded
    else:
        # No expansion needed
        return data


def calc_dur(
    data: pd.DataFrame,
    val_col: str,
    min_var: str,
    max_var: str,
    grp_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    unit_col: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Calculate duration for grouped events (R ricu calc_dur).
    
    Computes duration as the difference between max and min timestamps
    within each group (patient + grp_var).
    
    Args:
        data: Input DataFrame
        val_col: Output column name for duration
        min_var: Column with minimum time (start time)
        max_var: Column with maximum time (end time)
        grp_var: Optional grouping variable (e.g., linkorderid)
        id_cols: ID columns for patient grouping
        unit_col: Optional unit column to preserve
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with duration column
        
    Example:
        For each patient's medication infusion group, calculate:
        duration = max(endtime) - min(starttime)
    """
    if data.empty:
        data = data.copy()
        # Infer index variable (time column)
        time_cols = [col for col in data.columns if 'time' in col.lower()]
        index_var = time_cols[0] if time_cols else min_var
        data[val_col] = data[index_var]
        return data
    
    # Infer ID columns if not provided
    if id_cols is None:
        id_cols = [col for col in data.columns if 'id' in col.lower()]
    
    # Infer index variable (time column) - use min_var as the index output
    index_var = min_var
    
    # Build grouping columns
    group_cols = list(id_cols)
    if grp_var and grp_var in data.columns:
        group_cols.append(grp_var)
    
    # Collect all time columns to preserve them
    time_cols_to_keep = [col for col in data.columns if 'time' in col.lower()]
    
    # Build aggregation dict
    # Exclude group_cols from aggregation to avoid duplicates
    agg_dict = {
        min_var: 'min',
        max_var: 'max'
    }
    # Add time columns for aggregation (keep first value), but exclude group_cols
    for col in time_cols_to_keep:
        if col not in agg_dict and col != min_var and col != max_var and col not in group_cols:
            agg_dict[col] = 'first'
    
    # Add unit column if specified, but exclude if it's in group_cols
    if unit_col and unit_col in data.columns and unit_col not in group_cols:
        agg_dict[unit_col] = 'first'
    
    # Group and aggregate
    if group_cols:
        # Ensure datetime columns are properly typed before aggregation
        data = data.copy()
        data[min_var] = pd.to_datetime(data[min_var], errors='coerce')
        data[max_var] = pd.to_datetime(data[max_var], errors='coerce')
        
        # Drop rows where min_var or max_var is NaT
        data = data.dropna(subset=[min_var, max_var])
        
        if data.empty:
            # Return empty result with correct structure
            result = pd.DataFrame(columns=group_cols + [index_var, val_col])
            if unit_col and unit_col in data.columns:
                result[unit_col] = []
            return result
        
        # Remove duplicate columns before groupby (keep first occurrence)
        # This prevents "cannot insert X, already exists" errors
        data = data.loc[:, ~data.columns.duplicated(keep='first')]
        
        # Select only the columns we need for groupby and aggregation
        # This avoids pandas trying to insert duplicate columns
        # First, ensure we have a clean DataFrame with unique column names
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else str(col) 
                           for col in data.columns.values]
            # Remove any duplicates that might remain
            data = data.loc[:, ~data.columns.duplicated(keep='first')]
        
        cols_to_use = list(group_cols) + [min_var, max_var]
        if unit_col and unit_col in data.columns and unit_col not in cols_to_use:
            cols_to_use.append(unit_col)
        # Add time columns that are not already included
        for col in time_cols_to_keep:
            if col not in cols_to_use and col != min_var and col != max_var:
                cols_to_use.append(col)
        
        # Only use columns that actually exist and are unique
        cols_to_use = [col for col in cols_to_use if col in data.columns]
        # Remove duplicates while preserving order
        cols_to_use = list(dict.fromkeys(cols_to_use))
        data_subset = data[cols_to_use].copy()
        
        # Ensure group_cols are all valid 1D columns
        valid_group_cols = [col for col in group_cols if col in data_subset.columns]
        if len(valid_group_cols) != len(group_cols):
            # Some group cols are missing, use only valid ones
            group_cols = valid_group_cols
        
        # Build aggregation dict for subset (exclude group_cols from aggregation)
        agg_dict_subset = {
            min_var: 'min',
            max_var: 'max'
        }
        for col in time_cols_to_keep:
            if col in data_subset.columns and col not in agg_dict_subset and col != min_var and col != max_var and col not in group_cols:
                agg_dict_subset[col] = 'first'
        if unit_col and unit_col in data_subset.columns and unit_col not in group_cols:
            agg_dict_subset[unit_col] = 'first'
        
        # Groupby without as_index, then manually reset index
        # This gives us more control over handling duplicate columns
        grouped = data_subset.groupby(group_cols, dropna=False)
        result = grouped.agg(agg_dict_subset)
        
        # Reset index manually, using level numbers to avoid name conflicts
        if isinstance(result.index, pd.MultiIndex):
            # Get index values and convert to DataFrame columns
            index_df = pd.DataFrame(
                {result.index.names[i]: result.index.get_level_values(i) 
                 for i in range(len(result.index.names))},
                index=result.index
            )
            # Check which index columns are already in result.columns
            cols_to_add = []
            for col in index_df.columns:
                if col not in result.columns:
                    cols_to_add.append(col)
            
            # Add only new columns
            if cols_to_add:
                result = pd.concat([index_df[cols_to_add], result], axis=1)
            
            # Remove index
            result = result.reset_index(drop=True)
        else:
            # Single index
            if result.index.name:
                if result.index.name not in result.columns:
                    result[result.index.name] = result.index.values
            result = result.reset_index(drop=True)
        
        # Calculate duration: max - min
        # After aggregation, max_var contains max(endtime), min_var contains min(starttime)
        result[val_col] = result[max_var] - result[min_var]
        
        # Rename min_var to index_var (for consistency with time column naming)
        # The index_var should be the start time (min_var)
        # Note: min_var already contains the aggregated min(starttime), which becomes the index_var
        if min_var != index_var and min_var in result.columns:
            result = result.rename(columns={min_var: index_var})
        
        # Keep group cols, time cols, unit col, and value column
        # Ensure all necessary columns are present
        keep_cols = []
        for col in group_cols + [index_var] + [col for col in time_cols_to_keep if col != min_var] + [val_col]:
            if col in result.columns:
                keep_cols.append(col)
        if unit_col and unit_col in result.columns:
            keep_cols.append(unit_col)
        # Remove duplicates while preserving order
        keep_cols = list(dict.fromkeys(keep_cols))
        # Only keep columns that actually exist
        result = result[[col for col in keep_cols if col in result.columns]]
    else:
        # No grouping, just compute overall min/max
        result = pd.DataFrame({
            index_var: [data[min_var].min()],
            val_col: [data[max_var].max() - data[min_var].min()]
        })
        # Add ID columns if they exist
        for col in id_cols:
            if col in data.columns:
                result[col] = data[col].iloc[0]
    
    return result


def mimic_dur_inmv(
    data: pd.DataFrame,
    val_col: str = 'value',
    grp_var: Optional[str] = None,
    stop_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    unit_col: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """MIMIC MetaVision infusion duration callback (R ricu mimic_dur_inmv).
    
    Calculates total infusion duration for each medication group.
    
    Args:
        data: Input DataFrame with infusion data
        val_col: Output column for duration
        grp_var: Grouping variable (e.g., linkorderid)
        stop_var: End time variable
        id_cols: ID columns for patient grouping
        unit_col: Optional unit column to preserve
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with duration calculations
    """
    # Remove duplicate columns first (keep first occurrence)
    data = data.loc[:, ~data.columns.duplicated(keep='first')]
    
    # Infer index variable (start time)
    time_cols = [col for col in data.columns if 'time' in col.lower() and col != stop_var]
    if not time_cols:
        time_cols = [col for col in ['charttime', 'starttime'] if col in data.columns]
    
    if not time_cols:
        return data
    
    index_var = time_cols[0]
    
    # Call calc_dur
    result = calc_dur(
        data,
        val_col=val_col,
        min_var=index_var,
        max_var=stop_var or index_var,
        grp_var=grp_var,
        id_cols=id_cols,
        unit_col=unit_col
    )
    
    # Ensure val_col is timedelta type (not datetime)
    if val_col in result.columns and not pd.api.types.is_timedelta64_dtype(result[val_col]):
        # If it's datetime type, it's likely a bug - duration should be timedelta
        # For now, convert to NaT to avoid crashing
        if pd.api.types.is_datetime64_any_dtype(result[val_col]):
            result[val_col] = pd.Series([pd.NaT] * len(result), dtype='timedelta64[ns]')
        else:
            # Try to convert numeric or string to timedelta
            result[val_col] = pd.to_timedelta(result[val_col], errors='coerce')
    
    return result


def mimic_dur_incv(
    data: pd.DataFrame,
    val_col: str = 'value',
    grp_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    unit_col: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """MIMIC CareVue infusion duration callback (R ricu mimic_dur_incv).
    
    For CareVue system where there's no stop time, calculates duration
    using the same time column for both start and end.
    
    Args:
        data: Input DataFrame with infusion data
        val_col: Output column for duration
        grp_var: Grouping variable (e.g., linkorderid)
        id_cols: ID columns for patient grouping
        unit_col: Optional unit column to preserve
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with duration calculations
    """
    # Infer index variable (time column)
    time_cols = [col for col in data.columns if 'time' in col.lower()]
    if not time_cols:
        time_cols = [col for col in ['charttime', 'starttime'] if col in data.columns]
    
    if not time_cols:
        return data
    
    index_var = time_cols[0]
    
    # Call calc_dur with same column for min and max
    result = calc_dur(
        data,
        val_col=val_col,
        min_var=index_var,
        max_var=index_var,
        grp_var=grp_var,
        id_cols=id_cols,
        unit_col=unit_col
    )
    
    # Ensure val_col is timedelta type (not datetime)
    if val_col in result.columns and not pd.api.types.is_timedelta64_dtype(result[val_col]):
        # If it's datetime type, it's likely a bug - duration should be timedelta
        # For now, convert to NaT to avoid crashing
        if pd.api.types.is_datetime64_any_dtype(result[val_col]):
            result[val_col] = pd.Series([pd.NaT] * len(result), dtype='timedelta64[ns]')
        else:
            # Try to convert numeric or string to timedelta
            result[val_col] = pd.to_timedelta(result[val_col], errors='coerce')
    
    return result



def create_intervals(
    data: pd.DataFrame,
    by_cols: Optional[list] = None,
    overhang: pd.Timedelta = pd.Timedelta(hours=1),
    max_len: pd.Timedelta = pd.Timedelta(hours=6),
    end_var: str = 'endtime',
    **kwargs
) -> pd.DataFrame:
    """Create intervals for CareVue infusion data (R ricu create_intervals).
    
    When stop times are not available, creates estimated end times based on
    subsequent measurements or default overhang period.
    
    Args:
        data: Input DataFrame
        by_cols: Columns to group by
        overhang: Default duration to add if no next measurement
        max_len: Maximum interval length
        end_var: Output column name for end time
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with end time column added
    """
    if data.empty:
        data = data.copy()
        data[end_var] = pd.NaT
        return data
    
    # Infer time column
    time_cols = [col for col in data.columns if 'time' in col.lower()]
    if not time_cols:
        return data
    
    index_var = time_cols[0]
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(data[index_var]):
        data = data.copy()
        data[index_var] = pd.to_datetime(data[index_var])
    else:
        data = data.copy()
    
    # Infer by_cols if not provided
    if by_cols is None:
        by_cols = [col for col in data.columns if 'id' in col.lower()]
    
    # Sort by grouping columns and time
    sort_cols = by_cols + [index_var]
    data = data.sort_values(sort_cols)
    
    # Calculate end times
    if by_cols:
        # Group and calculate next time or add overhang
        def calc_end(group):
            group = group.copy()
            # Get next time within group
            group[end_var] = group[index_var].shift(-1)
            # For last row, use overhang
            mask = group[end_var].isna()
            group.loc[mask, end_var] = group.loc[mask, index_var] + overhang
            # Cap at max_len
            duration = group[end_var] - group[index_var]
            too_long = duration > max_len
            group.loc[too_long, end_var] = group.loc[too_long, index_var] + max_len
            return group
        
        data = data.groupby(by_cols, group_keys=False).apply(calc_end, include_groups=False)
    else:
        # No grouping
        data[end_var] = data[index_var].shift(-1)
        mask = data[end_var].isna()
        data.loc[mask, end_var] = data.loc[mask, index_var] + overhang
        duration = data[end_var] - data[index_var]
        too_long = duration > max_len
        data.loc[too_long, end_var] = data.loc[too_long, index_var] + max_len
    
    return data


def expand_intervals(
    data: pd.DataFrame,
    keep_vars: Optional[list] = None,
    grp_var: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Expand CareVue intervals into time series (R ricu expand_intervals).
    
    Creates intervals using create_intervals and then expands them.
    
    Args:
        data: Input DataFrame
        keep_vars: Variables to keep in expansion
        grp_var: Optional grouping variable
        **kwargs: Additional arguments
        
    Returns:
        Expanded DataFrame
    """
    from .ts_utils import expand
    
    # Infer ID columns
    id_cols = [col for col in data.columns if 'id' in col.lower()]
    
    # Build by_cols for create_intervals
    by_cols = list(id_cols)
    if grp_var and grp_var in data.columns:
        by_cols.append(grp_var)
    
    # Create intervals
    data = create_intervals(
        data,
        by_cols=by_cols,
        overhang=pd.Timedelta(hours=1),
        max_len=pd.Timedelta(hours=6),
        end_var='endtime'
    )
    
    # Infer index variable
    time_cols = [col for col in data.columns if 'time' in col.lower() and col != 'endtime']
    if not time_cols:
        return data
    
    index_var = time_cols[0]
    
    # Prepare keep_vars
    if keep_vars is None:
        keep_vars = []
    elif isinstance(keep_vars, str):
        keep_vars = [keep_vars]
    
    keep_vars = list(id_cols) + list(keep_vars)
    keep_vars = [v for v in keep_vars if v in data.columns and v not in [index_var, 'endtime']]
    
    # Expand
    expanded = expand(
        data,
        start_var=index_var,
        end_var='endtime',
        step_size=pd.Timedelta(hours=1),
        id_cols=id_cols,
        keep_vars=keep_vars
    )
    
    return expanded


def mimic_rate_cv(
    data: pd.DataFrame,
    val_col: str = 'value',
    grp_var: Optional[str] = None,
    unit_col: Optional[str] = None,
    id_cols: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """MIMIC CareVue infusion rate callback (R ricu mimic_rate_cv).
    
    For CareVue system, creates intervals and expands into time series.
    
    Args:
        data: Input DataFrame with infusion data
        val_col: Value column (infusion rate)
        grp_var: Grouping variable (e.g., linkorderid)
        unit_col: Unit column (rate units)
        id_cols: ID columns for grouping
        **kwargs: Additional arguments
        
    Returns:
        Expanded DataFrame with time series data
    """
    # Build keep_vars
    keep_vars = [val_col]
    if unit_col and unit_col in data.columns:
        keep_vars.append(unit_col)
    
    # Call expand_intervals
    return expand_intervals(data, keep_vars=keep_vars, grp_var=grp_var)


def hirid_vent(
    data: pd.DataFrame,
    index_var: str = 'datetime',
    id_cols: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """HiRID ventilation callback (R ricu hirid_vent).
    
    Creates duration windows for ventilation events using padded_capped_diff.
    Duration is calculated based on time differences, capped at 12 hours.
    
    Args:
        data: Input DataFrame with ventilation events
        index_var: Time index column
        id_cols: ID columns for grouping
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with 'dur_var' column for event durations
    """
    if id_cols is None:
        id_cols = [col for col in data.columns if 'id' in col.lower()]
    
    if not id_cols:
        raise ValueError("Cannot determine ID columns")
    
    data = data.copy()
    
    # Sort by ID and time
    data = data.sort_values(id_cols + [index_var])
    
    # Calculate durations within each ID group
    def calc_duration(group):
        """Calculate padded and capped time differences."""
        if len(group) == 0:
            return group
        
        # Calculate time differences (convert to hours)
        time_diffs = group[index_var].diff()
        
        # Replace first diff (NaN) with 4 hours (padding)
        # Cap all diffs at 12 hours
        durations = time_diffs.fillna(pd.Timedelta(hours=4))
        durations = durations.clip(upper=pd.Timedelta(hours=12))
        
        group['dur_var'] = durations
        return group
    
    data = data.groupby(id_cols, group_keys=False).apply(calc_duration, include_groups=False)
    
    return data


def grp_amount_to_rate(
    grp_var: str,
    unit_val: Union[str, dict],
    filt_fun: Optional[Callable] = None
) -> Callable:
    """Create callback for converting drug amounts to rates (R ricu grp_amount_to_rate).
    
    Converts cumulative drug amounts into infusion rates by taking
    differences within groups.
    
    Args:
        grp_var: Grouping variable (e.g., linkorderid)
        unit_val: Unit to assign to rates (string or mapping)
        filt_fun: Optional filter function to apply first
        
    Returns:
        Callback function
        
    Examples:
        >>> # Convert cumulative norepinephrine to rate
        >>> norepi_callback = grp_amount_to_rate(
        ...     grp_var='linkorderid',
        ...     unit_val='mcg/min'
        ... )
    """
    def callback(
        data: pd.DataFrame,
        val_col: str = 'value',
        unit_col: str = 'unit',
        index_var: str = 'datetime',
        id_cols: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        if id_cols is None:
            id_cols = [col for col in data.columns if 'id' in col.lower()]
        
        data = data.copy()
        
        # Apply filter if provided
        if filt_fun is not None:
            data = data[filt_fun(data)]
        
        # Sort by ID, group, and time
        sort_cols = id_cols + [grp_var, index_var] if grp_var in data.columns else id_cols + [index_var]
        data = data.sort_values(sort_cols)
        
        # Calculate rate within each group
        def calc_rate(group):
            if len(group) <= 1:
                group[val_col] = np.nan
                return group
            
            # Calculate time diff (in hours)
            time_diff = group[index_var].diff().dt.total_seconds() / 3600
            
            # Calculate amount diff
            amount_diff = group[val_col].diff()
            
            # Rate = amount_diff / time_diff
            group[val_col] = amount_diff / time_diff
            
            # Remove first row (NaN rate)
            return group.iloc[1:]
        
        if grp_var in data.columns:
            group_cols = id_cols + [grp_var]
        else:
            group_cols = id_cols
        
        data = data.groupby(group_cols, group_keys=False).apply(calc_rate, include_groups=False)
        
        # Set units
        if isinstance(unit_val, dict):
            # Map units based on some condition
            for key, val in unit_val.items():
                mask = data[grp_var] == key if grp_var in data.columns else slice(None)
                data.loc[mask, unit_col] = val
        else:
            data[unit_col] = unit_val
        
        return data
    
    return callback


def aumc_drug(
    data: pd.DataFrame,
    val_col: str = 'value',
    unit_col: str = 'unit',
    item_col: str = 'itemid',
    **kwargs
) -> pd.DataFrame:
    """AmsterdamUMCdb drug callback (R ricu aumc_drug).
    
    Handles special processing for AmsterdamUMCdb drug administration data.
    This may include unit conversions, rate calculations, etc.
    
    Args:
        data: Input DataFrame with drug data
        val_col: Value column
        unit_col: Unit column
        item_col: Item ID column
        **kwargs: Additional arguments
        
    Returns:
        Processed DataFrame
    """
    data = data.copy()
    
    # AmsterdamUMCdb-specific drug processing
    # This is highly data-specific and would need actual AUMC data structure
    # Placeholder implementation
    
    # Example: Convert doses to rates based on duration
    # Example: Standardize units
    
    return data


def ts_to_win_tbl(win_dur: pd.Timedelta) -> Callable:
    """Create callback to convert time series to windowed table (R ricu ts_to_win_tbl).
    
    Adds a constant duration to all events, creating a window table.
    
    Args:
        win_dur: Window duration to apply to all events
        
    Returns:
        Callback function
        
    Examples:
        >>> # Create 1-hour windows for all events
        >>> hourly_windows = ts_to_win_tbl(pd.Timedelta(hours=1))
    """
    def callback(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        data = data.copy()
        data['dur_var'] = win_dur
        return data
    
    return callback


def fwd_concept(concept_name: str) -> Callable:
    """Create callback that forwards to another concept (R ricu fwd_concept).
    
    Loads a different concept and uses its result. Useful for hierarchical
    concept definitions.
    
    Args:
        concept_name: Name of concept to load
        
    Returns:
        Callback function
        
    Examples:
        >>> # Use GCS total instead of recomputing
        >>> gcs_callback = fwd_concept('gcs')
    """
    def callback(data: pd.DataFrame, src: str = None, **kwargs) -> pd.DataFrame:
        from .load_concepts import load_concept
        
        if src is None:
            raise ValueError("Source must be specified for fwd_concept")
        
        # Load the referenced concept
        concept_data = load_concept(
            concept_name,
            src,
            aggregate=False,
            verbose=False,
            **kwargs
        )
        
        # Rename data column to 'val_var'
        if concept_data is not None and len(concept_data) > 0:
            # Find data column (non-ID, non-time)
            data_cols = [col for col in concept_data.columns 
                        if col not in ['id', 'datetime', 'time'] and 
                        'id' not in col.lower()]
            
            if data_cols:
                concept_data = concept_data.rename(columns={data_cols[0]: 'val_var'})
        
        return concept_data
    
    return callback


def dex_to_10(id_list: list, factor_list: list) -> Callable:
    """Create callback to convert dexmedetomidine concentrations (R ricu dex_to_10).
    
    Converts drug concentrations from various forms (e.g., 4 mcg/ml) to a 
    standard concentration (e.g., 10 mcg/ml equivalent).
    
    Args:
        id_list: List of item IDs or sub_var values to match
        factor_list: Corresponding conversion factors
        
    Returns:
        Callback function
        
    Examples:
        >>> # Convert 4 mcg/ml dex to 10 mcg/ml equivalent: multiply by 4/10
        >>> dex_cb = dex_to_10(
        ...     id_list=[[221668]],  # Item ID for 4 mcg/ml
        ...     factor_list=[0.4]    # 4/10 = 0.4
        ... )
    """
    if len(id_list) != len(factor_list):
        raise ValueError("id_list and factor_list must have the same length")
    
    def callback(
        data: pd.DataFrame,
        sub_var: str,
        val_col: str = 'value',
        **kwargs
    ) -> pd.DataFrame:
        """Apply conversion factors based on item IDs.
        
        Args:
            data: Input DataFrame
            sub_var: Column containing item IDs to match
            val_col: Value column to transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed DataFrame
        """
        data = data.copy()
        
        for ids, factor in zip(id_list, factor_list):
            # Ensure ids is a list
            if not isinstance(ids, (list, tuple)):
                ids = [ids]
            
            # Create mask for matching rows
            mask = data[sub_var].isin(ids)
            
            # Apply factor
            data.loc[mask, val_col] = data.loc[mask, val_col] * factor
        
        return data
    
    return callback


def mimv_rate(
    data: pd.DataFrame,
    val_col: str = 'value',
    unit_col: str = 'unit',
    dur_var: str = 'duration',
    amount_var: str = 'amount',
    auom_var: str = 'amountuom',
    **kwargs
) -> pd.DataFrame:
    """MIMIC MetaVision rate calculation callback (R ricu mimv_rate).
    
    For MIMIC-III/IV MetaVision inputevents, calculates infusion rate from
    amount when rate is not directly available.
    
    Args:
        data: Input DataFrame
        val_col: Rate column (output)
        unit_col: Unit column (output)
        dur_var: Duration column
        amount_var: Amount column
        auom_var: Amount unit of measure column
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with calculated rates
        
    Note:
        Only calculates rate where val_col is NA.
        Rate = amount / duration
        Unit = amountuom / duration_unit
    """
    data = data.copy()
    
    # Find rows where rate is NA
    mask = data[val_col].isna()
    
    if not mask.any():
        return data
    
    # Ensure duration is timedelta
    if dur_var in data.columns:
        if not pd.api.types.is_timedelta64_dtype(data[dur_var]):
            data[dur_var] = pd.to_timedelta(data[dur_var], errors='coerce')
        
        # Convert duration to hours for rate calculation
        dur_hours = data.loc[mask, dur_var].dt.total_seconds() / 3600
        
        # Avoid division by zero
        dur_hours = dur_hours.replace(0, np.nan)
        
        # Calculate rate
        if amount_var in data.columns and auom_var in data.columns:
            data.loc[mask, val_col] = data.loc[mask, amount_var] / dur_hours
            
            # Create unit string (e.g., "mg/hour")
            # Extract unit suffix from duration (e.g., "hour" from timedelta)
            data.loc[mask, unit_col] = data.loc[mask, auom_var] + '/hour'
    
    return data


def grp_amount_to_rate(
    grp_var: str,
    unit_val: Union[str, dict],
    filt_fun: Optional[Callable] = None
) -> Callable:
    """Create callback for converting drug amounts to rates (R ricu grp_amount_to_rate).
    
    DEPRECATED: Use grp_mount_to_rate instead for consistency with R ricu naming.
    Kept for backwards compatibility.
    """
    import warnings
    warnings.warn(
        "grp_amount_to_rate is deprecated, use grp_mount_to_rate instead",
        DeprecationWarning
    )
    
    return grp_mount_to_rate(
        min_dur=pd.Timedelta(minutes=1),
        extra_dur=pd.Timedelta(minutes=0),
        unit_val=unit_val,
        grp_var=grp_var,
        filt_fun=filt_fun
    )


def grp_mount_to_rate(
    min_dur: pd.Timedelta,
    extra_dur: pd.Timedelta,
    unit_val: Optional[Union[str, dict]] = None,
    grp_var: Optional[str] = None,
    filt_fun: Optional[Callable] = None
) -> Callable:
    """Create callback for converting grouped amounts to rates (R ricu grp_mount_to_rate).
    
    Aggregates drug amounts by group (e.g., linkorderid in MIMIC), calculates
    total duration, and converts to infusion rate.
    
    Args:
        min_dur: Minimum duration for zero-duration infusions
        extra_dur: Extra duration to add to all infusions
        unit_val: Unit to assign to rates (string or mapping)
        grp_var: Optional explicit grouping variable name
        filt_fun: Optional filter function to apply first
        
    Returns:
        Callback function
        
    Examples:
        >>> # Convert cumulative norepinephrine to rate with 1 min padding
        >>> norepi_cb = grp_mount_to_rate(
        ...     min_dur=pd.Timedelta(minutes=1),
        ...     extra_dur=pd.Timedelta(minutes=0),
        ...     unit_val='mcg/min',
        ...     grp_var='linkorderid'
        ... )
    """
    # Capture grp_var and unit_val in closure
    closure_grp_var = grp_var
    closure_unit_val = unit_val
    
    def callback(
        data: pd.DataFrame,
        val_col: str = 'value',
        unit_col: str = 'unit',
        index_var: Optional[str] = None,
        id_cols: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        if data.empty:
            return data
        
        # Use closure variable, but allow override from kwargs
        nonlocal closure_grp_var
        grp_var_to_use = kwargs.get('grp_var', closure_grp_var)
        
        # Infer index_var if not provided
        if index_var is None:
            time_cols = [col for col in data.columns if 'time' in col.lower()]
            index_var = time_cols[0] if time_cols else 'time'
        
        # Infer ID columns if not provided
        if id_cols is None:
            id_cols = [col for col in data.columns if 'id' in col.lower()]
        
        data = data.copy()
        
        # Apply filter if provided
        if filt_fun is not None:
            data = data[filt_fun(data)]
        
        # Build grouping columns
        group_cols = list(id_cols)
        if grp_var_to_use and grp_var_to_use in data.columns:
            group_cols.append(grp_var_to_use)
        
        # Sort by group and time
        sort_cols = group_cols + [index_var]
        data = data.sort_values(sort_cols)
        
        # Aggregate by group
        agg_dict = {
            index_var: ['min', 'max'],
            val_col: 'sum'
        }
        
        # Keep first unit if available
        if unit_col in data.columns:
            agg_dict[unit_col] = lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
        
        result = data.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        
        # Flatten column names
        result.columns = [
            col[0] if col[1] == '' or col[1] == '<lambda>' else f"{col[0]}_{col[1]}"
            for col in result.columns
        ]
        
        # Calculate duration
        min_time_col = f"{index_var}_min"
        max_time_col = f"{index_var}_max"
        
        result['dur_var'] = result[max_time_col] - result[min_time_col]
        
        # Apply min_dur for zero-duration events
        zero_dur_mask = result['dur_var'] == pd.Timedelta(0)
        result.loc[zero_dur_mask, 'dur_var'] = min_dur
        
        # Add extra_dur to all durations
        result['dur_var'] = result['dur_var'] + extra_dur
        
        # Calculate rate: amount / duration (convert to hours for rate/hour)
        dur_hours = result['dur_var'].dt.total_seconds() / 3600
        result[val_col] = result[f"{val_col}_sum"] / dur_hours
        
        # Set units
        if closure_unit_val is not None:
            if isinstance(closure_unit_val, dict):
                # Map units based on group variable
                for key, val in closure_unit_val.items():
                    mask = result[grp_var_to_use] == key if grp_var_to_use in result.columns else slice(None)
                    result.loc[mask, unit_col] = val
            else:
                result[unit_col] = closure_unit_val
        elif unit_col in result.columns:
            # Append rate unit to existing unit
            base_unit = result.get(f"{unit_col}_<lambda>", result.get(unit_col, 'unit'))
            result[unit_col] = base_unit.astype(str) + '/hour'
        
        # Rename min time back to index_var
        result = result.rename(columns={min_time_col: index_var})
        
        # Select output columns
        output_cols = group_cols + [index_var, 'dur_var', val_col]
        if unit_col in result.columns:
            output_cols.append(unit_col)
        
        result = result[[col for col in output_cols if col in result.columns]]
        
        return result
    
    return callback


def padded_capped_diff(
    times: pd.Series,
    padding: pd.Timedelta,
    cap: pd.Timedelta
) -> pd.Series:
    """Calculate time differences with padding and capping (R ricu padded_capped_diff).
    
    Used for calculating event durations with sensible defaults.
    
    Args:
        times: Series of timestamps
        padding: Default duration for first event
        cap: Maximum allowed duration
        
    Returns:
        Series of durations
        
    Examples:
        >>> times = pd.to_datetime(['2020-01-01 00:00', '2020-01-01 02:00', 
        ...                         '2020-01-01 20:00'])
        >>> padded_capped_diff(times, pd.Timedelta(hours=4), pd.Timedelta(hours=12))
        # Returns: [4 hours, 2 hours, 12 hours (capped from 18)]
    """
    # Ensure we're working with a Series, not Index
    if isinstance(times, pd.DatetimeIndex):
        times = pd.Series(times)
    
    diffs = times.diff()
    
    # Replace first diff (NaN) with padding
    diffs = diffs.fillna(padding)
    
    # Cap at maximum
    diffs = diffs.clip(upper=cap)
    
    return diffs


# ============================================================================
# Additional callback utilities from R ricu
# ============================================================================

def combine_date_time(
    data: pd.DataFrame,
    time_var: str,
    date_shift: pd.Timedelta = pd.Timedelta(hours=12),
    index_var: str = 'time',
    **kwargs
) -> pd.DataFrame:
    """Combine date and time columns (R ricu combine_date_time).
    
    When time_var is NA, uses index_var + date_shift as the time.
    
    Args:
        data: Input DataFrame
        time_var: Time variable column name
        date_shift: Shift to apply when time is NA
        index_var: Index variable column name
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with combined time
    """
    data = data.copy()
    
    if time_var not in data.columns or index_var not in data.columns:
        return data
    
    # Where time_var is NA, use index_var + date_shift
    mask = data[time_var].isna()
    data.loc[mask, index_var] = data.loc[mask, index_var] + date_shift
    
    return data


def add_concept(
    data: pd.DataFrame,
    env,
    concept: str,
    var_name: Optional[str] = None,
    aggregate: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Add another concept to current data (R ricu add_concept).
    
    Loads a referenced concept and merges it with the current data.
    Used when one concept depends on another (e.g., vasopressor rates
    need weight).
    
    Args:
        data: Current data DataFrame
        env: Data source environment
        concept: Name of concept to load
        var_name: Variable name for merged concept (default: concept name)
        aggregate: Aggregation method for concept loading
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with added concept
        
    Examples:
        >>> # Add weight to vasopressor data
        >>> data = add_concept(vaso_data, env, 'weight')
    """
    from .load_concepts import load_concept
    
    if var_name is None:
        var_name = concept
    
    # Determine source name
    if hasattr(env, 'name'):
        src = env.name
    elif isinstance(env, str):
        src = env
    else:
        raise ValueError("Cannot determine source name from env")
    
    # Load the concept
    concept_data = load_concept(
        concept,
        src,
        aggregate=aggregate,
        verbose=False,
        **kwargs
    )
    
    if concept_data is None or len(concept_data) == 0:
        # Return original data if concept not available
        return data
    
    # Rename value column to var_name if different
    value_cols = [col for col in concept_data.columns 
                  if col not in ['id', 'datetime', 'time'] and 'id' not in col.lower()]
    
    if value_cols and value_cols[0] != var_name:
        concept_data = concept_data.rename(columns={value_cols[0]: var_name})
    
    # Merge with current data
    # Find common ID and time columns
    id_cols = [col for col in data.columns if 'id' in col.lower()]
    time_cols = [col for col in data.columns if col in ['datetime', 'time', 'charttime']]
    
    merge_cols = []
    for col in id_cols + time_cols:
        if col in data.columns and col in concept_data.columns:
            merge_cols.append(col)
    
    if not merge_cols:
        # Cannot merge, return original
        return data
    
    # Perform merge
    result = pd.merge(data, concept_data, on=merge_cols, how='left')
    
    return result


def add_weight(
    data: pd.DataFrame,
    env,
    var_name: str = 'weight',
    **kwargs
) -> pd.DataFrame:
    """Add weight concept to data (R ricu add_weight).
    
    Special case of add_concept for weight, with fallback handling.
    
    Args:
        data: Current data DataFrame
        env: Data source environment
        var_name: Variable name for weight (default: 'weight')
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with weight added
        
    Examples:
        >>> # Add weight to vasopressor rate calculation
        >>> vaso_data = add_weight(vaso_data, env)
        >>> vaso_data['rate_per_kg'] = vaso_data['rate'] / vaso_data['weight']
    """
    # Check if weight already exists
    if var_name in data.columns:
        # Weight exists, but may have NAs - fill from concept
        temp_var = f"__{var_name}_temp__"
        data = add_concept(data, env, 'weight', var_name=temp_var, **kwargs)
        
        if temp_var in data.columns:
            # Convert existing weight to numeric
            data[var_name] = pd.to_numeric(data[var_name], errors='coerce')
            
            # Fill NAs from loaded weight
            mask = data[var_name].isna()
            data.loc[mask, var_name] = data.loc[mask, temp_var]
            
            # Drop temp column
            data = data.drop(columns=[temp_var])
        
        return data
    else:
        # Weight doesn't exist, add it
        return add_concept(data, env, 'weight', var_name=var_name, **kwargs)


def blood_cell_ratio(
    data: pd.DataFrame,
    val_col: str = 'value',
    unit_col: str = 'unit',
    env=None,
    **kwargs
) -> pd.DataFrame:
    """Convert blood cell counts to ratios (R ricu blood_cell_ratio).
    
    Converts absolute cell counts to percentages by dividing by WBC.
    
    Args:
        data: Input DataFrame with cell counts
        val_col: Value column name
        unit_col: Unit column name
        env: Data source environment
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with ratios instead of absolute counts
    """
    if env is None:
        # Cannot convert without WBC, return as-is
        return data
    
    # Add WBC concept
    data = add_concept(data, env, 'wbc', var_name='wbc')
    
    if 'wbc' not in data.columns:
        # WBC not available, return as-is
        return data
    
    data = data.copy()
    
    # Convert to ratio
    data[val_col] = 100 * data[val_col] / data['wbc']
    data[unit_col] = '%'
    
    # Drop WBC column
    data = data.drop(columns=['wbc'])
    
    return data


def silent_as_numeric(x: Union[pd.Series, np.ndarray, Any]) -> Union[pd.Series, np.ndarray, float]:
    """Convert to numeric, suppressing warnings (R ricu silent_as_num).
    
    Args:
        x: Data to convert
        
    Returns:
        Numeric data, with non-convertible values as NaN
    """
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors='coerce')
    elif isinstance(x, np.ndarray):
        return pd.to_numeric(pd.Series(x), errors='coerce').values
    else:
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan


def eicu_extract_unit(x: Union[str, pd.Series]) -> Union[str, pd.Series]:
    """Extract unit from eICU medication strings (R ricu eicu_extract_unit).
    
    eICU often stores units in parentheses, like "Drug Name (mg/hr)".
    
    Args:
        x: String or Series with units in parentheses
        
    Returns:
        Extracted unit(s)
        
    Examples:
        >>> eicu_extract_unit("Norepinephrine (mcg/kg/min)")
        'mcg/kg/min'
        >>> eicu_extract_unit("Drug")
        nan
    """
    if isinstance(x, pd.Series):
        # Extract text within parentheses
        units = x.str.extract(r'\(([^)]+)\)')[0]
        # Return NaN for empty strings
        units = units.replace('', np.nan)
        return units
    else:
        # Single string
        match = re.search(r'\(([^)]+)\)', str(x))
        if match:
            unit = match.group(1)
            return unit if unit else np.nan
        return np.nan


def sub_trans(regex: str, repl: str) -> Callable:
    """Create a substitution transform function (R ricu sub_trans).
    
    Returns a function that performs regex substitution.
    
    Args:
        regex: Regular expression pattern
        repl: Replacement string
        
    Returns:
        Function that performs substitution
        
    Examples:
        >>> convert_hr_to_min = sub_trans(r'/hr$', '/min')
        >>> convert_hr_to_min('mg/hr')
        'mg/min'
    """
    
    def transformer(x: Union[str, pd.Series]) -> Union[str, pd.Series]:
        if isinstance(x, pd.Series):
            return x.str.replace(regex, repl, regex=True, case=False)
        else:
            return re.sub(regex, repl, str(x), flags=re.IGNORECASE)
    
    return transformer


def get_one_unique(x: Union[pd.Series, list], na_rm: bool = False) -> Any:
    """Get single unique value or NA (R ricu get_one_unique).
    
    If there's exactly one unique value, return it.
    If there are multiple unique values, return NA.
    
    Args:
        x: Data to check
        na_rm: Whether to remove NA before checking
        
    Returns:
        Single unique value or NA
        
    Examples:
        >>> get_one_unique([1, 1, 1])
        1
        >>> get_one_unique([1, 2, 3])
        nan
    """
    if isinstance(x, pd.Series):
        if na_rm:
            x = x.dropna()
        unique_vals = x.unique()
    else:
        if na_rm:
            x = [v for v in x if not pd.isna(v)]
        unique_vals = list(set(x))
    
    if len(unique_vals) == 1:
        return unique_vals[0]
    else:
        return np.nan


def units_to_unit(x: pd.Timedelta) -> str:
    """Convert timedelta to unit string (R ricu units_to_unit).
    
    Removes 's' from unit name (e.g., 'hours' -> 'hour').
    
    Args:
        x: Timedelta object
        
    Returns:
        Unit string without 's'
        
    Examples:
        >>> units_to_unit(pd.Timedelta(hours=1))
        'hour'
    """
    # Get resolution (e.g., 'H' for hours, 'T' for minutes)
    resolution = x.resolution_string
    
    # Map to full names
    unit_map = {
        'D': 'day',
        'H': 'hour',
        'T': 'min',
        'S': 'sec',
        'L': 'millisec',
        'U': 'microsec',
        'N': 'nanosec'
    }
    
    return unit_map.get(resolution, 'hour')


def eicu_rate_kg_callback(ml_to_mcg: float) -> Callable:
    """eICU dose rate conversion with weight normalization (R ricu eicu_rate_kg).
    
    Converts various dose rate units to mcg/kg/min, handling:
    - /hr -> /min (divide by 60)
    - mg/ -> mcg/ (multiply by 1000)
    - units/kg/min -> NA (not convertible)
    - ml/ -> mcg/ (multiply by ml_to_mcg)
    - nanograms/ -> mcg/ (divide by 1000)
    - Unknown/ml units -> NA
    - Add /kg/ for non-kg-normalized rates using patient weight
    
    Args:
        ml_to_mcg: Conversion factor from ml to mcg (drug concentration)
        
    Returns:
        Callback function
        
    Examples:
        >>> # Norepinephrine: 1 ml = 1600 mcg
        >>> norepi_callback = eicu_rate_kg_callback(ml_to_mcg=1600)
    """
    def callback(
        frame: pd.DataFrame,
        val_var: str,
        sub_var: str,
        weight_var: str,
        concept_name: str,
    ) -> pd.DataFrame:
        """Apply eICU rate/kg conversion.
        
        Args:
            frame: Input dataframe
            val_var: Value column name
            sub_var: Sub-variable column (contains unit info)
            weight_var: Weight column name (from patient table)
            concept_name: Output concept name
            
        Returns:
            Converted dataframe
        """
        frame = frame.copy()
        
        # Convert values to numeric
        if val_var in frame.columns:
            frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
        
        # Extract unit from sub_var (e.g., "drugname (mcg/kg/min)")
        if sub_var in frame.columns:
            # eICU format: "drugname (unit)" or just "unit"
            def extract_unit(s):
                if pd.isna(s):
                    return None
                s = str(s)
                # Match pattern like "(unit)" or just return as-is
                match = re.search(r'\(([^)]+)\)$', s)
                if match:
                    return match.group(1)
                # Check if it's already a unit-like string
                if '/' in s or s.lower() in ['mg', 'mcg', 'ml', 'units']:
                    return s
                return None
            
            frame['unit_var'] = frame[sub_var].apply(extract_unit)
        else:
            # No sub_var, assume all are same unit or unknown
            frame['unit_var'] = 'Unknown'
        
        # Load weight if needed (merge from patient table)
        # For now, we'll use a placeholder - in practice, this should join with patient table
        if weight_var not in frame.columns:
            # Try to get weight from patient table via patientunitstayid
            # This is a simplified version - full implementation would join tables
            frame[weight_var] = 70.0  # Default weight kg as fallback
        
        # 🔧 FIX: Convert weight to numeric (handle empty strings in infusiondrug.patientweight)
        if weight_var in frame.columns:
            frame[weight_var] = pd.to_numeric(frame[weight_var], errors='coerce')
        
        # Normalize to /kg/ if not already
        frame['is_per_kg'] = frame['unit_var'].str.contains('/kg/', case=False, na=False)
        
        # For non-kg rates, divide by weight and update unit
        # 🔧 FIX: Also check weight is not NaN after numeric conversion
        mask_non_kg = ~frame['is_per_kg'] & frame[val_var].notna() & frame[weight_var].notna()
        if mask_non_kg.any():
            # 🔧 FIX: Convert val_var to float before division to avoid dtype mismatch warning
            frame[val_var] = frame[val_var].astype(float)
            frame.loc[mask_non_kg, val_var] = frame.loc[mask_non_kg, val_var] / frame.loc[mask_non_kg, weight_var]
            frame.loc[mask_non_kg, 'unit_var'] = frame.loc[mask_non_kg, 'unit_var'].apply(
                lambda u: u.replace('/', '/kg/') if u and '/' in u else f'{u}/kg' if u else 'Unknown/kg'
            )
        
        # Now apply unit conversions to standardize to mcg/kg/min
        def convert_to_mcg_kg_min(row):
            val = row[val_var]
            unit = row.get('unit_var', '')
            
            if pd.isna(val) or not unit:
                return val
            
            unit = str(unit).strip()
            
            # /hr -> /min
            if '/hr' in unit or '/hour' in unit:
                val = val / 60
                unit = re.sub(r'/h(ou)?r', '/min', unit, flags=re.IGNORECASE)
            
            # mg -> mcg
            if unit.startswith('mg/') or 'mg/kg' in unit:
                val = val * 1000
                unit = unit.replace('mg/', 'mcg/').replace('mg/kg', 'mcg/kg')
            
            # ml -> mcg
            if unit.startswith('ml/') or 'ml/kg' in unit:
                val = val * ml_to_mcg
                unit = unit.replace('ml/', 'mcg/').replace('ml/kg', 'mcg/kg')
            
            # nanograms -> mcg
            if unit.startswith('nanograms/') or 'nanograms/kg' in unit:
                val = val / 1000
                unit = unit.replace('nanograms/', 'mcg/').replace('nanograms/kg', 'mcg/kg')
            
            # Set to NA for incompatible units
            if 'units/' in unit.lower() or unit.lower() in ['unknown', 'ml', 'unknown/kg', 'ml/kg']:
                return np.nan
            
            return val
        
        frame[concept_name] = frame.apply(convert_to_mcg_kg_min, axis=1)
        
        # Clean up temporary columns
        frame = frame.drop(columns=['unit_var', 'is_per_kg'], errors='ignore')
        
        return frame
    
    return callback


def _aumc_get_id_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if isinstance(col, str) and col.lower().endswith('id')]


def _aumc_normalize_mass_units(df: pd.DataFrame, unit_col: Optional[str], val_col: str) -> None:
    if not unit_col:
        return
    if unit_col not in df.columns:
        df[unit_col] = 'mcg'
        return

    df[unit_col] = df[unit_col].astype(str).str.strip()
    units_lower = df[unit_col].str.lower()

    mask_mg = units_lower.isin({'mg', 'milligram', 'milligrams'})
    if mask_mg.any():
        df.loc[mask_mg, val_col] = df.loc[mask_mg, val_col] * 1_000.0
        df.loc[mask_mg, unit_col] = 'mcg'

    mask_g = units_lower.isin({'g', 'gram', 'grams'})
    if mask_g.any():
        df.loc[mask_g, val_col] = df.loc[mask_g, val_col] * 1_000_000.0
        df.loc[mask_g, unit_col] = 'mcg'

    mask_micro = units_lower.isin({'µg', 'μg', 'ug', 'microgram', 'micrograms'})
    if mask_micro.any():
        df.loc[mask_micro, unit_col] = 'mcg'

    mask_mcg = units_lower.isin({'mcg', 'mcgs'})
    if mask_mcg.any():
        df.loc[mask_mcg, unit_col] = 'mcg'


def _aumc_normalize_rate_units(df: pd.DataFrame, rate_uom_col: Optional[str], val_col: str, default: str = 'min') -> Optional[str]:
    if not rate_uom_col:
        return None
    if rate_uom_col not in df.columns:
        df[rate_uom_col] = default
        return rate_uom_col

    df[rate_uom_col] = df[rate_uom_col].astype(str).str.strip()
    rate_lower = df[rate_uom_col].str.lower()

    mask_hour = rate_lower.isin({'uur', 'u', 'hour', 'hours', 'h'})
    if mask_hour.any():
        df.loc[mask_hour, val_col] = df.loc[mask_hour, val_col] / 60.0
        df.loc[mask_hour, rate_uom_col] = 'min'

    mask_day = rate_lower.isin({'dag', 'dagen', 'day', 'days', 'd'})
    if mask_day.any():
        df.loc[mask_day, val_col] = df.loc[mask_day, val_col] / (24.0 * 60.0)
        df.loc[mask_day, rate_uom_col] = 'min'

    mask_sec = rate_lower.isin({'sec', 'seconde', 'second', 'seconds', 's'})
    if mask_sec.any():
        df.loc[mask_sec, val_col] = df.loc[mask_sec, val_col] * 60.0
        df.loc[mask_sec, rate_uom_col] = 'min'

    df[rate_uom_col] = df[rate_uom_col].replace({'nan': 'min', 'none': 'min'}).fillna('min')
    return rate_uom_col


def aumc_rate_kg(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: str,
    unit_col: Optional[str],
    rel_weight_col: Optional[str],
    rate_unit_col: Optional[str],
    index_col: Optional[str],
    stop_col: Optional[str],
    default_weight: float = 70.0,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    df = frame.copy()

    if val_col not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + [concept_name])

    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[val_col])
    if df.empty:
        return df

    _aumc_normalize_mass_units(df, unit_col, val_col)
    rate_unit_col = _aumc_normalize_rate_units(df, rate_unit_col, val_col) or rate_unit_col

    if 'weight' not in df.columns:
        df['weight'] = default_weight
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(default_weight)

    if rel_weight_col and rel_weight_col in df.columns:
        rel_mask = df[rel_weight_col].fillna(False).astype(bool)
    else:
        rel_mask = pd.Series(False, index=df.index)

    mask_non_perkg = (~rel_mask) & (df['weight'] > 0)
    if mask_non_perkg.any():
        df.loc[mask_non_perkg, val_col] = df.loc[mask_non_perkg, val_col] / df.loc[mask_non_perkg, 'weight']

    if unit_col and unit_col in df.columns:
        df[unit_col] = df[unit_col].astype(str).replace({'µg': 'mcg', 'μg': 'mcg', 'ug': 'mcg'})
    else:
        unit_col = None

    if rate_unit_col and rate_unit_col in df.columns:
        df[rate_unit_col] = df[rate_unit_col].astype(str)
        if unit_col and unit_col in df.columns:
            df[unit_col] = df[unit_col] + '/kg/' + df[rate_unit_col]
    elif unit_col and unit_col in df.columns:
        df[unit_col] = df[unit_col] + '/kg/min'

    base_time = pd.Timestamp('2000-01-01')
    if index_col and index_col in df.columns and pd.api.types.is_numeric_dtype(df[index_col]):
        df[index_col] = base_time + pd.to_timedelta(pd.to_numeric(df[index_col], errors='coerce'), unit='ms')

    df[concept_name] = df[val_col]

    id_cols = _aumc_get_id_columns(df)
    result_cols = list(dict.fromkeys(id_cols))
    
    # 🔧 CRITICAL FIX: 确保时间列总是包含在返回中(即使为空或不存在)
    # aumc_rate_kg回调在R中调用expand(),保留index_var(时间列)
    # Python中必须显式保留,否则vaso60回调会失败(rate_df没有时间列,dur_df有)
    if index_col:
        # 即使index_col不在df.columns中,也需要确保它存在
        # 如果不存在,创建一个空的时间列(NaT)
        if index_col not in df.columns:
            df[index_col] = pd.NaT
        result_cols.append(index_col)
    
    result_cols.append(concept_name)
    if unit_col and unit_col in df.columns:
        result_cols.append(unit_col)
    if rate_unit_col and rate_unit_col in df.columns:
        result_cols.append(rate_unit_col)

    return df[result_cols].dropna(subset=[concept_name])


def aumc_rate_units_callback(mcg_to_units: float) -> Callable:
    def callback(
        frame: pd.DataFrame,
        val_col: str,
        unit_col: Optional[str],
        rate_unit_col: Optional[str],
        stop_col: Optional[str],
        concept_name: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame

        df = frame.copy()

        if val_col not in df.columns:
            return pd.DataFrame(columns=list(df.columns) + [concept_name])

        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[val_col])
        if df.empty:
            return df

        if unit_col and unit_col in df.columns:
            df[unit_col] = df[unit_col].astype(str).str.strip()
            lower = df[unit_col].str.lower()

            mask_micro = lower.isin({'µg', 'μg', 'ug', 'microgram', 'micrograms'})
            if mask_micro.any():
                df.loc[mask_micro, unit_col] = 'mcg'

            mask_mg = lower.isin({'mg', 'milligram', 'milligrams'})
            if mask_mg.any():
                df.loc[mask_mg, val_col] = df.loc[mask_mg, val_col] * 1000.0
                df.loc[mask_mg, unit_col] = 'mcg'

            lower = df[unit_col].str.lower()
            mask_mcg = lower.isin({'mcg', 'microgram', 'micrograms'})
            if mask_mcg.any():
                df.loc[mask_mcg, val_col] = df.loc[mask_mcg, val_col] * mcg_to_units
                df.loc[mask_mcg, unit_col] = 'units'
        else:
            unit_col = None

        rate_unit_col = _aumc_normalize_rate_units(df, rate_unit_col, val_col) or rate_unit_col
        if rate_unit_col and rate_unit_col in df.columns:
            df[rate_unit_col] = df[rate_unit_col].astype(str)

        if unit_col and unit_col in df.columns:
            if rate_unit_col and rate_unit_col in df.columns:
                df[unit_col] = df[unit_col] + '/' + df[rate_unit_col]
            else:
                df[unit_col] = df[unit_col] + '/min'

        base_time = pd.Timestamp('2000-01-01')
        index_col = next((col for col in ['start', 'charttime', 'time'] if col in df.columns), None)
        if index_col and pd.api.types.is_numeric_dtype(df[index_col]):
            df[index_col] = base_time + pd.to_timedelta(pd.to_numeric(df[index_col], errors='coerce'), unit='ms')

        df[concept_name] = df[val_col]

        id_cols = _aumc_get_id_columns(df)
        result_cols = list(dict.fromkeys(id_cols))
        if index_col and index_col in df.columns:
            result_cols.append(index_col)
        result_cols.append(concept_name)
        if unit_col and unit_col in df.columns:
            result_cols.append(unit_col)
        if rate_unit_col and rate_unit_col in df.columns:
            result_cols.append(rate_unit_col)

        return df[result_cols].dropna(subset=[concept_name])

    return callback


def aumc_dur(
    frame: pd.DataFrame,
    *,
    val_col: str,
    stop_var: Optional[str],
    grp_var: Optional[str],
    index_var: Optional[str],
    concept_name: str,
) -> pd.DataFrame:
    if frame.empty or not stop_var or stop_var not in frame.columns:
        return frame

    df = frame.copy()

    start_col = index_var if index_var and index_var in df.columns else None
    if not start_col:
        start_col = next((col for col in ['start', 'charttime', 'time'] if col in df.columns), None)
    if not start_col:
        return df

    base_time = pd.Timestamp('2000-01-01')
    start_numeric = pd.to_numeric(df[start_col], errors='coerce')
    stop_numeric = pd.to_numeric(df[stop_var], errors='coerce')
    df['__start_dt'] = base_time + pd.to_timedelta(start_numeric, unit='ms')
    df['__stop_dt'] = base_time + pd.to_timedelta(stop_numeric, unit='ms')

    id_cols = _aumc_get_id_columns(df)
    group_var = grp_var if grp_var and grp_var in df.columns else None

    result = calc_dur(
        df,
        val_col=val_col,
        min_var='__start_dt',
        max_var='__stop_dt',
        grp_var=group_var,
        id_cols=id_cols,
    )

    if val_col in result.columns and pd.api.types.is_timedelta64_dtype(result[val_col]):
        result[val_col] = result[val_col].dt.total_seconds() / 3600.0

    if '__start_dt' in result.columns:
        result = result.rename(columns={'__start_dt': start_col})
    if '__stop_dt' in result.columns:
        result = result.drop(columns=['__stop_dt'])

    result[concept_name] = result[val_col]

    return result
