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

logger = logging.getLogger(__name__)

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
                except Exception:
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

# 注意: silent_as_numeric 的完整版本在第2489行定义（支持多种输入类型）

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
    interval_hours: float = 1.0,  # 默认 1 小时间隔
    admission_times: pd.DataFrame = None,  # 🔧 For proper relative time floor behavior
    **kwargs
) -> pd.DataFrame:
    """Distribute total amount over duration to get rate, then expand.
    
    For drug administrations given as total amount over a duration,
    converts to rate per hour AND expands to hourly time points.
    
    R ricu 逻辑 (distribute_amount):
    1. 过滤掉 endtime - starttime < 0 的行
    2. 对于 duration == 0 的行，设置 endtime = starttime + 1hr
    3. 计算速率 = amount / duration * 1hr
    4. 调用 expand() 展开时间窗口到每个小时
    5. 设置单位为 units/hr
    
    Args:
        admission_times: DataFrame with id_col and 'intime' columns for relative time calculation.
                        Required for proper floor() behavior matching R ricu.
    """
    data = data.copy()
    
    # 检测 ID 列
    id_cols = [c for c in data.columns if c.lower().endswith('id') or c.lower() == 'stay_id']
    
    # 确保时间列存在
    if index_col not in data.columns or end_col not in data.columns:
        return data
    
    # 将时间转换为小时（如果是数值）或 datetime
    start_time = data[index_col].copy()
    end_time = data[end_col].copy()
    
    # 判断时间是数值（小时）还是 datetime
    is_numeric = pd.api.types.is_numeric_dtype(start_time)
    
    if is_numeric:
        # 时间已经是小时数
        pass
    else:
        # 转换为 datetime 然后计算小时差
        start_time = pd.to_datetime(start_time, errors='coerce')
        end_time = pd.to_datetime(end_time, errors='coerce')
        
        # 假设时间已经是相对于某个参考点的
        # 转换为相对小时数（如果需要）
        if start_time.notna().any():
            # 保持原始逻辑
            pass
    
    if is_numeric:
        # 计算时间差（小时）
        time_diff_hours = end_time - start_time
        
        # 过滤掉 endtime - starttime < 0 的行
        valid_mask = time_diff_hours >= 0
        data = data[valid_mask].copy()
        start_time = start_time[valid_mask]
        end_time = end_time[valid_mask]
        time_diff_hours = time_diff_hours[valid_mask]
        
        if data.empty:
            return data
        
        # 对于 duration == 0 的行，设置 endtime = starttime + 1
        zero_duration_mask = time_diff_hours == 0
        if zero_duration_mask.any():
            end_time = end_time.copy()
            end_time.loc[zero_duration_mask] = start_time.loc[zero_duration_mask] + 1.0
            data.loc[zero_duration_mask, end_col] = end_time.loc[zero_duration_mask]
            time_diff_hours = end_time - start_time
        
        # 计算速率 = amount / duration_hours
        time_diff_hours = time_diff_hours.replace(0, 1)  # 避免除以零
        data[val_col] = pd.to_numeric(data[val_col], errors='coerce') / time_diff_hours
        
        # 向量化展开时间窗口
        start_hrs = np.floor(start_time.values).astype(int)
        end_hrs = np.floor(end_time.values).astype(int)
        n_points = np.maximum(end_hrs - start_hrs + 1, 1)
        
        if n_points.sum() > 0:
            # Find ID columns present in data
            valid_id_cols = [c for c in id_cols if c in data.columns]
            
            # Build expanded arrays
            time_vals = np.concatenate([np.arange(s, s + n) for s, n in zip(start_hrs, n_points)])
            val_vals = np.repeat(data[val_col].values, n_points)
            
            result_dict = {index_col: time_vals, val_col: val_vals}
            for c in valid_id_cols:
                result_dict[c] = np.repeat(data[c].values, n_points)
            
            result = pd.DataFrame(result_dict)
            if unit_col and unit_col in data.columns:
                result[unit_col] = 'units/hr'
            return result
        else:
            return data
    
    else:
        # datetime 逻辑 - 需要转换为相对时间然后展开
        # 
        # R ricu 的处理流程：
        # 1. load_difftime() 将 datetime 转换为相对时间（分钟）
        # 2. change_interval(floor) 将分钟转换为小时并 floor
        # 3. distribute_amount 接收 floor 后的时间:
        #    - 如果 end - start == 0，设置 end = start + 1
        #    - 计算速率 = amount / (end - start) * 1hr
        # 4. expand() 展开到每个小时
        #
        # 关键点：R ricu 计算速率时使用的是 floor 后的持续时间！
        
        start_time = pd.to_datetime(data[index_col], errors='coerce')
        end_time = pd.to_datetime(data[end_col], errors='coerce')
        
        time_diff = end_time - start_time
        valid_mask = time_diff >= pd.Timedelta(0)
        data = data[valid_mask].copy()
        start_time = start_time[valid_mask]
        end_time = end_time[valid_mask]
        
        if data.empty:
            return data
        
        expanded_rows = []
        
        # 检测 ID 列 - 优先使用标准的患者 ID 列
        standard_id_cols = ['icustay_id', 'stay_id', 'admissionid', 'patientid', 'patientunitstayid', 'hadm_id', 'subject_id']
        id_col = None
        for col in standard_id_cols:
            if col in data.columns:
                id_col = col
                break
        if id_col is None:
            for col in id_cols:
                if col in data.columns:
                    id_col = col
                    break
        
        # 获取每个患者的 intime 用于计算相对时间
        intime_map = {}
        if admission_times is not None and id_col is not None:
            for _, row in admission_times.iterrows():
                if id_col in row.index and 'intime' in row.index:
                    patient_id = row[id_col]
                    intime = pd.to_datetime(row['intime'], errors='coerce')
                    if pd.notna(intime):
                        if intime.tzinfo is None:
                            intime = intime.tz_localize('UTC')
                        intime_map[patient_id] = intime
        
        # Vectorized path for rows with intime
        if intime_map and id_col is not None:
            # Ensure timezone consistency
            starts_dt = start_time.copy()
            ends_dt = end_time.copy()
            if starts_dt.dt.tz is None:
                starts_dt = starts_dt.dt.tz_localize('UTC')
            if ends_dt.dt.tz is None:
                ends_dt = ends_dt.dt.tz_localize('UTC')
            
            # Map intime to each row
            intimes = data[id_col].map(intime_map)
            has_intime = intimes.notna()
            amounts = pd.to_numeric(data[val_col], errors='coerce')
            has_amount = amounts.notna()
            valid = has_intime & has_amount
            
            if valid.any():
                d_valid = data.loc[valid].copy()
                intimes_v = pd.to_datetime(intimes.loc[valid])
                starts_v = starts_dt.loc[valid]
                ends_v = ends_dt.loc[valid]
                amounts_v = amounts.loc[valid].values
                
                # Compute relative hours and floor
                start_rel = (starts_v.values - intimes_v.values).astype('timedelta64[s]').astype(float) / 3600
                end_rel = (ends_v.values - intimes_v.values).astype('timedelta64[s]').astype(float) / 3600
                start_floor = np.floor(start_rel).astype(int)
                end_floor = np.floor(end_rel).astype(int)
                
                # If end == start, set end = start + 1
                same_mask = end_floor == start_floor
                end_floor[same_mask] = start_floor[same_mask] + 1
                
                # Rate = amount / duration
                duration = (end_floor - start_floor).astype(float)
                duration[duration == 0] = 1.0
                rates = amounts_v / duration
                
                # Expand
                n_points = np.maximum(end_floor - start_floor + 1, 1)
                total = n_points.sum()
                if total > 0:
                    valid_ids = [c for c in id_cols if c in d_valid.columns]
                    time_vals = np.concatenate([np.arange(s, s + n, dtype=float) for s, n in zip(start_floor, n_points)])
                    rate_vals = np.repeat(rates, n_points)
                    result_dict = {index_col: time_vals, val_col: rate_vals}
                    for c in valid_ids:
                        result_dict[c] = np.repeat(d_valid[c].values, n_points)
                    expanded_rows = [pd.DataFrame(result_dict)]
            
            # Handle rows without intime (fallback)
            no_intime = (~has_intime) & has_amount
            if no_intime.any():
                for idx in data.index[no_intime]:
                    row = data.loc[idx]
                    row_start = start_time.loc[idx]
                    row_end = end_time.loc[idx]
                    if pd.isna(row_start) or pd.isna(row_end):
                        continue
                    if row_start.tzinfo is None:
                        row_start = row_start.tz_localize('UTC')
                    if row_end.tzinfo is None:
                        row_end = row_end.tz_localize('UTC')
                    amount = amounts.loc[idx]
                    start_fl = row_start.floor('h')
                    end_fl = row_end.floor('h')
                    if start_fl == end_fl:
                        end_fl = start_fl + pd.Timedelta(hours=1)
                    dur = (end_fl - start_fl).total_seconds() / 3600
                    rate = amount / dur
                    time_points = pd.date_range(start=start_fl, end=end_fl, freq='h')
                    for t in time_points:
                        new_row = {c: row[c] for c in id_cols if c in row.index}
                        new_row[index_col] = t
                        new_row[val_col] = rate
                        expanded_rows.append(new_row)
        else:
            # No intime_map: fallback to per-row datetime processing
            for idx, row in data.iterrows():
                row_start = start_time.loc[idx]
                row_end = end_time.loc[idx]
                if pd.isna(row_start) or pd.isna(row_end):
                    continue
                if row_start.tzinfo is None:
                    row_start = row_start.tz_localize('UTC')
                if row_end.tzinfo is None:
                    row_end = row_end.tz_localize('UTC')
                amount = pd.to_numeric(row[val_col], errors='coerce')
                if pd.isna(amount):
                    continue
                start_fl = row_start.floor('h')
                end_fl = row_end.floor('h')
                if start_fl == end_fl:
                    end_fl = start_fl + pd.Timedelta(hours=1)
                dur = (end_fl - start_fl).total_seconds() / 3600
                rate = amount / dur
                time_points = pd.date_range(start=start_fl, end=end_fl, freq='h')
                for t in time_points:
                    new_row = {c: row[c] for c in id_cols if c in row.index}
                    new_row[index_col] = t
                    new_row[val_col] = rate
                    expanded_rows.append(new_row)
        
        if expanded_rows:
            if isinstance(expanded_rows[0], pd.DataFrame):
                result = pd.concat(expanded_rows, ignore_index=True)
            else:
                result = pd.DataFrame(expanded_rows)
            if unit_col and unit_col in data.columns:
                result[unit_col] = 'units/hr'
            return result
        else:
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
                    filled = group[val_col].ffill()
                    group[val_col] = group[val_col].where(can_fill, filled)
                else:
                    # Simple forward fill
                    group[val_col] = group[val_col].ffill()
                
                return group
            
            # 🔧 FIX pandas 3.0: groupby().apply() drops group columns.
            # Re-add group columns after apply.
            _grp_backup = data[id_cols].copy()
            data = data.groupby(id_cols).apply(fill_group).reset_index(drop=True)
            for _gc in id_cols:
                if _gc not in data.columns:
                    data[_gc] = _grp_backup[_gc].values
        else:
            # Simple forward fill
            data[val_col] = data[val_col].ffill()
        
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
                    filled = group[val_col].bfill()
                    group[val_col] = group[val_col].where(can_fill, filled)
                else:
                    # Simple backward fill
                    group[val_col] = group[val_col].bfill()
                
                return group
            
            # 🔧 FIX pandas 3.0: groupby().apply() drops group columns.
            # Re-add group columns after apply.
            _grp_backup = data[id_cols].copy()
            data = data.groupby(id_cols).apply(fill_group).reset_index(drop=True)
            for _gc in id_cols:
                if _gc not in data.columns:
                    data[_gc] = _grp_backup[_gc].values
        else:
            # Simple backward fill
            data[val_col] = data[val_col].bfill()
        
        return data
    
    return _locb_callback

def vent_flag(
    data: pd.DataFrame,
    val_col: str = "value",
    index_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    **kwargs,
) -> pd.DataFrame:
    """Filter to ventilated rows and use val_col as new time index.
    
    This replicates R ricu's vent_flag behavior exactly:
    ```R
    vent_flag <- function(x, val_var, ...) {
      x <- x[as.logical(get(val_var)), ]
      set(x, j = c(index_var(x), val_var),
          value = list(x[[val_var]], rep(TRUE, nrow(x))))
    }
    ```
    
    The key insight is that val_var (e.g., ventstartoffset=1566) becomes
    the new time index, and the value column is set to TRUE.
    """
    if val_col not in data.columns:
        return data.copy()

    frame = data.copy()
    
    # 🔥 R ricu: x <- x[as.logical(get(val_var)), ]
    # 过滤只保留 val_col 为真值的行（非零、非NA）
    numeric_val = pd.to_numeric(frame[val_col], errors='coerce')
    mask = numeric_val.notna() & (numeric_val != 0)
    frame = frame.loc[mask].copy()
    
    if frame.empty:
        return frame
    
    # 🔥 R ricu: set(x, j = c(index_var(x), val_var), value = list(x[[val_var]], rep(TRUE, nrow(x))))
    # 这意味着：
    # 1. index_var 列被设置为 val_col 的原始值（时间戳）
    # 2. val_col 列被设置为 TRUE
    
    # 保存 val_col 的原始值（这将成为新的时间索引）
    original_val = numeric_val.loc[frame.index]
    
    # 如果 index_var 存在，用 val_col 的值替换它
    if index_var and index_var in frame.columns:
        frame[index_var] = original_val.values
    elif index_var:
        # 如果 index_var 不存在，创建它
        frame[index_var] = original_val.values
    
    # 将 val_col 设置为 TRUE
    frame[val_col] = True
    
    # Ensure id columns are preserved
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

        # Handle numeric offset vs datetime properly
        is_offset = 'offset' in index_var.lower()
        if is_offset:
            # eICU offset is numeric (minutes from ICU admission)
            frame[index_var] = pd.to_numeric(frame[index_var], errors="coerce")
            frame = frame.dropna(subset=[index_var])
            
            if frame.empty:
                return frame
            
            # CRITICAL: R ricu converts minutes to hours BEFORE grouping.
            # floor(offset/60) → hour-level, then groups on unique hours.
            # This affects gap detection: e.g., 310min gap < 5h (strict >) stays together,
            # but at minute-level 310 > 300 would split incorrectly.
            frame['_hour'] = (frame[index_var] // 60).astype(int)
            # Keep original offset for min tracking
            frame['_orig_offset'] = frame[index_var]
            # Deduplicate per (patient, hour) — keep first occurrence per hour
            dedup_cols = [c for c in id_cols if c in frame.columns] + ['_hour']
            frame = frame.drop_duplicates(subset=dedup_cols, keep='first')
            
            # For group_measurements, convert HOURS to datetime (so gap unit matches hours)
            base_time = pd.Timestamp('2000-01-01')
            frame['__temp_time'] = base_time + pd.to_timedelta(frame['_hour'], unit='h')
            temp_index_var = '__temp_time'
        else:
            # Other databases use datetime
            frame[index_var] = pd.to_datetime(frame[index_var], errors="coerce")
            frame = frame.dropna(subset=[index_var])
            
            if frame.empty:
                return frame
            
            temp_index_var = index_var

        # Add group column using group_measurements
        grouped = group_measurements(
            frame,
            id_cols=id_cols,
            index_col=temp_index_var,
            max_gap=gap_length,
            group_col=group_col,
        )
        
        # If we used temporary time column, drop it now but keep original index_var
        if is_offset:
            grouped = grouped.drop(columns=['__temp_time'], errors='ignore')

        # Calculate duration per group (R calc_dur logic)
        # Following R ricu's calc_dur implementation
        # Simplify to match R ricu exactly
        
        # Make sure all ID columns actually exist in grouped dataframe
        valid_id_cols = [col for col in id_cols if col in grouped.columns]
        
        if not valid_id_cols:
            import logging
            logging.warning(f"No valid ID columns found in grouped data. Available columns: {list(grouped.columns)}")
            # Return empty with correct structure
            return pd.DataFrame(columns=list(id_cols) + [index_var, val_col])

        groupby_cols = valid_id_cols + [group_col]
        
        # R ricu: res <- x[, list(min(min_var), max(max_var)), by = c(id_vars, grp_var)]
        result = grouped.groupby(groupby_cols, dropna=False).agg({
            index_var: ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        result.columns = groupby_cols + ['min_time', 'max_time']
        
        # R ricu: res <- res[, c(val_var) := get(val_var) - get(index_var)]
        # Calculate duration: max - min
        if is_offset:
            # eICU: use _hour for duration (integer hour difference, matching R ricu)
            # R does: floor(max_offset/60) - floor(min_offset/60)
            result_hr = grouped.groupby(groupby_cols, dropna=False).agg(
                min_hour=('_hour', 'min'),
                max_hour=('_hour', 'max'),
            ).reset_index()
            result[val_col] = result_hr['max_hour'] - result_hr['min_hour']
            result[val_col] = result[val_col].astype(float)
        else:
            # Other databases: datetime difference gives timedelta, convert to hours
            result[val_col] = (result['max_time'] - result['min_time']).dt.total_seconds() / 3600.0
        
        # Use min_time as the index_var value for this duration
        result[index_var] = result['min_time']
        
        # Return columns: id_vars + index_var + val_var (drop group_col)
        # R ricu returns: id_vars, grp_var, index_var, val_var
        # But for duration concepts, we typically don't need grp_var in final output
        final_cols = valid_id_cols + [index_var, val_col]
        result = result[final_cols]
        
        return result

    return _callback

def mimic_rate_mv(
    data: pd.DataFrame,
    val_col: str = 'value',
    unit_col: Optional[str] = None,
    stop_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    admission_times: Optional[pd.DataFrame] = None,
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
        admission_times: DataFrame with id and intime columns for time alignment
        **kwargs: Additional arguments
        
    Returns:
        Expanded DataFrame with time series data
        
    Note:
        This is a simplified version that expands intervals.
        In ricu, it calls expand(x, index_var(x), stop_var, keep_vars = ...)
        
        🔧 CRITICAL FIX 2024-11-30: R ricu converts datetime to relative time 
        BEFORE calling expand(). This affects the floor() behavior:
        - R ricu: 06:39 -> relative 13.26h -> floor -> 13
        - Old easyicu: 06:39 -> floor -> 06:00 -> relative 12.61h -> 12
        
        We now pass admission_times to expand() to fix this discrepancy.
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
    
    # Prepare keep_vars - IMPORTANT: do NOT include stop_var, matching R ricu behavior
    # R ricu mimic_rate_mv: keep_vars = c(id_vars(x), val_var, unit_var) - no stop_var
    # Keeping stop_var would cause double expand in _load_single_concept
    keep_vars = list(id_cols) + [val_col]
    if unit_col and unit_col in data.columns:
        keep_vars.append(unit_col)
    # NOTE: DO NOT add stop_var to keep_vars - it should be removed after expand
    
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
            keep_vars=keep_vars,
            admission_times=admission_times,  # 🔧 Pass admission times for proper floor behavior
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
    admission_times: Optional[pd.DataFrame] = None,
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
        admission_times: DataFrame with id_col and intime columns for relative time calculation
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
    # 🔧 FIX 2025-02-10: R ricu's calc_dur uses id_vars(x) which returns only the PRIMARY patient ID column
    # (e.g., icustay_id), NOT all columns containing "id" in their name.
    # Using all "id" columns causes over-grouping and wrong results.
    if id_cols is None:
        # Try to find the primary patient ID column in priority order
        primary_id_candidates = ['icustay_id', 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        id_cols = []
        for cand in primary_id_candidates:
            if cand in data.columns:
                id_cols = [cand]
                break
        # If no standard ID found, fall back to first column with 'id' (but NOT grp_var or technical IDs)
        if not id_cols:
            technical_ids = {'row_id', 'subject_id', 'hadm_id', 'itemid', 'cgid', 'orderid', 'linkorderid'}
            for col in data.columns:
                if 'id' in col.lower() and col.lower() not in technical_ids and col != grp_var:
                    id_cols = [col]
                    break
    
    # 🔧 FIX: Exclude grp_var from id_cols to avoid duplication
    if grp_var and grp_var in id_cols:
        id_cols = [col for col in id_cols if col != grp_var]
    
    # Infer index variable (time column) - use min_var as the index output
    index_var = min_var
    
    # Build grouping columns - ensure no duplicates
    group_cols = list(id_cols)
    if grp_var and grp_var in data.columns and grp_var not in group_cols:
        group_cols.append(grp_var)
    # Remove any duplicates while preserving order
    group_cols = list(dict.fromkeys(group_cols))
    
    # Collect all time columns to preserve them
    time_cols_to_keep = [col for col in data.columns if 'time' in col.lower()]
    
    # Group and aggregate
    # 🔧 FIX: R ricu calc_dur creates two separate columns:
    #   - index_var = min(min_var)
    #   - val_var = max(max_var)
    # Then computes: val_var = val_var - index_var (duration)
    # 
    # When min_var == max_var (e.g., CareVue), we need to handle this specially
    # because pandas agg() with same key would only keep one result.
    
    if group_cols:
        data = data.copy()
        
        # Check if time columns are already numeric (relative hours) or datetime
        min_is_numeric = pd.api.types.is_numeric_dtype(data[min_var])
        max_is_numeric = pd.api.types.is_numeric_dtype(data[max_var])
        
        if not min_is_numeric:
            data[min_var] = pd.to_datetime(data[min_var], errors='coerce')
        if not max_is_numeric and max_var != min_var:
            data[max_var] = pd.to_datetime(data[max_var], errors='coerce')
        
        # Drop rows where min_var is NaN/NaT
        data = data.dropna(subset=[min_var])
        
        if data.empty:
            result = pd.DataFrame(columns=list(id_cols) + [index_var, val_col])
            if unit_col and unit_col in data.columns:
                result[unit_col] = []
            return result
        
        # Remove duplicate columns
        data = data.loc[:, ~data.columns.duplicated(keep='first')]
        
        # 🔧 FIX: Select only needed columns for groupby to avoid conflicts
        cols_needed = list(group_cols) + [min_var]
        if max_var != min_var and max_var in data.columns:
            cols_needed.append(max_var)
        if unit_col and unit_col in data.columns:
            cols_needed.append(unit_col)
        cols_needed = list(dict.fromkeys(cols_needed))  # Remove duplicates
        data_subset = data[cols_needed].copy()
        
        # 🔧 FIX: Handle MultiIndex columns
        if isinstance(data_subset.columns, pd.MultiIndex):
            data_subset.columns = ['_'.join(map(str, col)).strip('_') for col in data_subset.columns.values]
        
        # 🔧 FIX: Use named aggregation with unique names that won't conflict with group_cols
        agg_funcs = {
            '_calc_dur_min_time': (min_var, 'min'),
            '_calc_dur_max_time': (max_var if max_var in data_subset.columns else min_var, 'max'),
        }
        
        # Add unit column if specified (but NOT if it's a group column)
        if unit_col and unit_col in data_subset.columns and unit_col not in group_cols:
            agg_funcs['_calc_dur_unit'] = (unit_col, 'first')
        
        # Perform groupby with named aggregation
        grouped = data_subset.groupby(group_cols, dropna=False)
        agg_result = grouped.agg(**agg_funcs)
        
        
        # 🔧 FIX: Check for column conflicts before reset_index
        # If any index level name exists as a column, drop that column first
        for level_name in agg_result.index.names:
            if level_name is not None and level_name in agg_result.columns:
                logger.debug("Dropping conflicting column %s before reset_index", level_name)
                agg_result = agg_result.drop(columns=[level_name])
        
        result = agg_result.reset_index()
        
        # 🔧 FIX: R ricu's duration calculation uses floor(end_hours) - floor(start_hours)
        # Now we have: _calc_dur_min_time = min(start), _calc_dur_max_time = max(end)
        # Compute: val_col = floor(_calc_dur_max_time) - floor(_calc_dur_min_time)
        import numpy as np
        
        # Check if times are already in numeric hours or datetime
        min_time_col = result['_calc_dur_min_time']
        max_time_col = result['_calc_dur_max_time']
        
        if pd.api.types.is_numeric_dtype(min_time_col) and pd.api.types.is_numeric_dtype(max_time_col):
            # Times are already in relative hours - use floor(end_h) - floor(start_h)
            min_hours = min_time_col.astype(float)
            max_hours = max_time_col.astype(float)
            result[val_col] = np.floor(max_hours) - np.floor(min_hours)
        else:
            # Times are datetime - need to compute relative hours using intime
            id_col = None
            for col in id_cols if id_cols else []:
                if col in result.columns:
                    id_col = col
                    break
            
            if admission_times is not None and id_col is not None and 'intime' in admission_times.columns:
                intime_df = admission_times[[id_col, 'intime']].drop_duplicates()
                intime_df['intime'] = pd.to_datetime(intime_df['intime'], errors='coerce')
                result = result.merge(intime_df, on=id_col, how='left')
                
                result['_calc_dur_min_time'] = pd.to_datetime(result['_calc_dur_min_time'], errors='coerce')
                result['_calc_dur_max_time'] = pd.to_datetime(result['_calc_dur_max_time'], errors='coerce')
                
                min_hours = (result['_calc_dur_min_time'] - result['intime']).dt.total_seconds() / 3600.0
                max_hours = (result['_calc_dur_max_time'] - result['intime']).dt.total_seconds() / 3600.0
                
                result[val_col] = np.floor(max_hours) - np.floor(min_hours)
                result = result.drop(columns=['intime'], errors='ignore')
            else:
                # Fallback: no intime available, use floor(duration)
                duration_td = result['_calc_dur_max_time'] - result['_calc_dur_min_time']
                
                if pd.api.types.is_timedelta64_dtype(duration_td):
                    duration_hours = duration_td.dt.total_seconds() / 3600.0
                else:
                    duration_hours = duration_td.apply(
                        lambda x: x.total_seconds() / 3600.0 if hasattr(x, 'total_seconds') else float(x)
                    )
                
                result[val_col] = np.floor(duration_hours)
        
        # Rename _calc_dur_min_time to index_var (start time)
        result = result.rename(columns={'_calc_dur_min_time': index_var})
        
        # Handle unit column
        if '_calc_dur_unit' in result.columns:
            result = result.rename(columns={'_calc_dur_unit': unit_col})
        
        # 🔧 FIX: R ricu calc_dur only returns: id_vars + index_var (start time) + val_var (duration)
        # It does NOT keep endtime/max_var column!
        keep_cols = list(id_cols) + [index_var, val_col]
        if unit_col and unit_col in result.columns:
            keep_cols.append(unit_col)
        keep_cols = list(dict.fromkeys(keep_cols))
        result = result[[col for col in keep_cols if col in result.columns]]
    else:
        # No grouping, just compute overall min/max
        import numpy as np
        min_time = data[min_var].min()
        max_time = data[max_var].max()
        
        # Check if times are already in numeric hours or datetime
        if pd.api.types.is_numeric_dtype(data[min_var]) and pd.api.types.is_numeric_dtype(data[max_var]):
            # Times are already in relative hours
            min_hours = float(min_time)
            max_hours = float(max_time)
        else:
            # Times are datetime - compute duration and convert
            duration_td = max_time - min_time
            if hasattr(duration_td, 'total_seconds'):
                min_hours = 0  # Reference point
                max_hours = duration_td.total_seconds() / 3600.0
            else:
                min_hours = float(min_time)
                max_hours = float(max_time)
        
        result = pd.DataFrame({
            index_var: [min_time],
            val_col: [np.floor(max_hours) - np.floor(min_hours)]
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
    admission_times: Optional[pd.DataFrame] = None,
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
        admission_times: DataFrame with id_col and intime columns for relative time calculation
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
    
    # Call calc_dur (returns hours as numeric, not timedelta)
    result = calc_dur(
        data,
        val_col=val_col,
        min_var=index_var,
        max_var=stop_var or index_var,
        grp_var=grp_var,
        id_cols=id_cols,
        unit_col=unit_col,
        admission_times=admission_times
    )
    
    # calc_dur now returns hours as numeric (matching ricu behavior)
    # No need to convert to timedelta
    
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
    
    # Call calc_dur with same column for min and max (returns hours as numeric)
    result = calc_dur(
        data,
        val_col=val_col,
        min_var=index_var,
        max_var=index_var,
        grp_var=grp_var,
        id_cols=id_cols,
        unit_col=unit_col
    )
    
    # calc_dur now returns hours as numeric (matching ricu behavior)
    # No need to convert to timedelta
    
    return result

def create_intervals(
    data: pd.DataFrame,
    by_cols: Optional[list] = None,
    overhang: pd.Timedelta = pd.Timedelta(hours=1),
    max_len: pd.Timedelta = pd.Timedelta(hours=6),
    end_var: str = 'endtime',
    interval: pd.Timedelta = pd.Timedelta(hours=1),  # Add interval parameter
    **kwargs
) -> pd.DataFrame:
    """Create intervals for CareVue infusion data (R ricu create_intervals).
    
    When stop times are not available, creates estimated end times based on
    subsequent measurements or default overhang period.
    
    R ricu logic:
    1. Calculate diff to next time (or use overhang for last record)
    2. Truncate diff to [0, max_len]
    3. Subtract interval (typically 1 hour)
    4. endtime = start + adjusted_diff
    
    Args:
        data: Input DataFrame
        by_cols: Columns to group by
        overhang: Default duration to add if no next measurement
        max_len: Maximum interval length
        end_var: Output column name for end time
        interval: Time interval to subtract from diff (default 1 hour)
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with end time column added
    """
    if data.empty:
        data = data.copy()
        data[end_var] = pd.NaT
        return data
    
    # Infer time column - support eICU's infusionoffset and HiRID's givenat/datetime
    time_col_patterns = ['time', 'offset', 'charttime', 'starttime', 'givenat', 'datetime']
    time_cols = []
    for col in data.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in time_col_patterns):
            time_cols.append(col)
    
    if not time_cols:
        return data
    
    index_var = time_cols[0]
    
    # Check if time column is numeric (hours since admission) vs datetime
    is_numeric_time = pd.api.types.is_numeric_dtype(data[index_var])
    
    data = data.copy()
    
    # Infer by_cols if not provided
    if by_cols is None:
        by_cols = [col for col in data.columns if 'id' in col.lower()]
    
    # Sort by grouping columns and time
    sort_cols = by_cols + [index_var]
    data = data.sort_values(sort_cols)
    
    # Convert overhang, max_len, and interval to appropriate units for numeric time
    if is_numeric_time:
        # 🔧 FIX: Don't assume time is in hours - detect actual unit
        # For eICU infusionoffset, time is in MINUTES
        # We use total_seconds() / 60 to get minutes
        # This works because the caller (expand_intervals) passes Timedelta
        # with appropriate resolution (minutes for eICU, hours for others)
        
        # Check if this looks like minute-based data (eICU)
        is_minute_based = index_var.lower() == 'infusionoffset'
        
        if is_minute_based:
            # Convert to minutes
            overhang_val = overhang.total_seconds() / 60.0
            max_len_val = max_len.total_seconds() / 60.0
            interval_val = interval.total_seconds() / 60.0
        else:
            # Assume hours (for MIIV and others)
            overhang_val = overhang.total_seconds() / 3600.0
            max_len_val = max_len.total_seconds() / 3600.0
            interval_val = interval.total_seconds() / 3600.0
        
        # R ricu logic (matching ricu 0.6.3 behavior):
        # NOTE: R ricu 0.6.3's trunc_time has a bug - it doesn't assign the result!
        # So max_len truncation is NOT applied in practice.
        # We match this behavior to produce identical results to gold standard.
        # 
        # 1. diff = next_time - start (or overhang for last record)
        # 2. diff = trunc(diff, 0, max_len)  # In ricu 0.6.3: max_len NOT applied due to bug
        # 3. diff = diff - interval
        # 4. endtime = start + diff
        
        if by_cols:
            # 🔧 FIX pandas 3.0: groupby().apply() drops group columns.
            # Use vectorized groupby().shift() instead.
            next_times = data.groupby(by_cols)[index_var].shift(-1)
            diff = next_times - data[index_var]
            diff = diff.fillna(overhang_val)
            diff = diff.clip(lower=0)  # Only apply lower bound (ricu 0.6.3 bug)
            diff = diff - interval_val
            diff = diff.clip(lower=0)
            data[end_var] = data[index_var] + diff
        else:
            next_times = data[index_var].shift(-1)
            diff = next_times - data[index_var]
            diff = diff.fillna(overhang_val)
            # NOTE: max_len NOT applied - matching ricu 0.6.3 bug
            diff = diff.clip(lower=0)
            diff = diff - interval_val
            diff = diff.clip(lower=0)
            data[end_var] = data[index_var] + diff
    else:
        # Original datetime logic
        if not pd.api.types.is_datetime64_any_dtype(data[index_var]):
            data[index_var] = pd.to_datetime(data[index_var])
        
        if by_cols:
            # 🔧 FIX pandas 3.0: groupby().apply() drops group columns.
            # Use vectorized groupby().shift() instead.
            next_times = data.groupby(by_cols)[index_var].shift(-1)
            diff = next_times - data[index_var]
            diff = diff.fillna(overhang)
            diff = diff.clip(lower=pd.Timedelta(0))
            diff = diff - interval
            diff = diff.clip(lower=pd.Timedelta(0))
            data[end_var] = data[index_var] + diff
        else:
            next_times = data[index_var].shift(-1)
            diff = next_times - data[index_var]
            diff = diff.fillna(overhang)
            # NOTE: max_len NOT applied - matching ricu 0.6.3 bug
            diff = diff.clip(lower=pd.Timedelta(0))
            diff = diff - interval
            diff = diff.clip(lower=pd.Timedelta(0))
            data[end_var] = data[index_var] + diff
    
    return data


def expand_intervals(
    data: pd.DataFrame,
    keep_vars: Optional[list] = None,
    grp_var: Optional[str] = None,
    id_cols: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """Expand CareVue intervals into time series (R ricu expand_intervals).
    
    Creates intervals using create_intervals and then expands them.
    
    R ricu behavior:
    - create_intervals groups by (id_vars, grp_var) to create intervals per infusion
    - expand only keeps id_vars in output, NOT grp_var
    - When multiple infusions have overlapping times, expand produces duplicate rows
    - These duplicates are aggregated later by aggregate() using median for numeric values
    - CRITICAL: R ricu expands at MINUTE resolution first, then aggregates to hours
      This ensures continuous hourly output even when measurements are sparse
    
    Args:
        data: Input DataFrame
        keep_vars: Variables to keep in expansion
        grp_var: Optional grouping variable (e.g., infusionid) - used for interval creation
                but NOT kept in output
        id_cols: Explicit list of ID columns to use for grouping. If None, auto-detect.
                 Pass this to avoid false-positive ID column detection (e.g. row_id, itemid).
        **kwargs: Additional arguments
        
    Returns:
        Expanded DataFrame with duplicates aggregated by median
    """
    from .ts_utils import expand
    import numpy as np
    
    if id_cols is None:
        # 🔧 FIX: 只使用标准患者/住院 ID 列进行分组
        # 不再用 endswith('id') 模式匹配，避免误匹配 row_id, itemid, cgid, orderid 等
        standard_id_cols = [
            'patientunitstayid', 'stay_id', 'icustay_id', 'hadm_id',
            'admissionid', 'patientid', 'CaseID', 'subject_id'
        ]
        id_cols = [col for col in standard_id_cols if col in data.columns]
    
    if grp_var and grp_var in id_cols:
        id_cols = [c for c in id_cols if c != grp_var]
    
    # Build by_cols for create_intervals - INCLUDE grp_var for grouping
    by_cols = list(id_cols)
    if grp_var and grp_var in data.columns:
        by_cols.append(grp_var)
    
    # Infer index variable - support eICU's infusionoffset and HiRID's givenat/datetime
    time_col_patterns = ['time', 'offset', 'charttime', 'starttime', 'givenat', 'datetime']
    time_cols = []
    for col in data.columns:
        if col == 'endtime':
            continue
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in time_col_patterns):
            time_cols.append(col)
    
    if not time_cols:
        return data
    
    index_var = time_cols[0]
    
    # 🔧 CRITICAL FIX: Detect eICU minute-based data
    # eICU uses infusionoffset in MINUTES since admission
    # R ricu expands at MINUTE resolution first, then aggregates to hours
    # This ensures continuous hourly output
    is_eicu_minutes = index_var.lower() == 'infusionoffset' and pd.api.types.is_numeric_dtype(data[index_var])
    
    if is_eicu_minutes:
        # OPTIMIZED: Compute hourly output directly without minute-level expansion.
        # For each interval, calculate which hours it covers and assign the rate value.
        # This avoids creating 60× more rows than needed.
        data = data.copy()
        
        # Create intervals at minute resolution (for correct interval boundaries)
        data = create_intervals(
            data,
            by_cols=by_cols,
            overhang=pd.Timedelta(minutes=60),
            max_len=pd.Timedelta(minutes=360),
            end_var='endtime',
            interval=pd.Timedelta(minutes=1)
        )
        
        # Prepare keep_vars
        if keep_vars is None:
            keep_vars = []
        elif isinstance(keep_vars, str):
            keep_vars = [keep_vars]
        keep_vars = list(id_cols) + list(keep_vars)
        keep_vars = [v for v in keep_vars if v in data.columns and v != index_var]
        
        # Direct hour-level expansion: for each row, generate one entry per covered hour
        starts_min = data[index_var].values.astype(float)
        ends_min = data['endtime'].values.astype(float)
        
        # Filter valid intervals (start <= end)
        valid = starts_min <= ends_min
        if not valid.any():
            cols = list(id_cols) + [index_var] + [v for v in keep_vars if v not in id_cols]
            return pd.DataFrame(columns=cols)
        
        starts_min = starts_min[valid]
        ends_min = ends_min[valid]
        data_valid = data[valid].reset_index(drop=True)
        
        start_hours = np.floor(starts_min / 60.0).astype(int)
        end_hours = np.floor(ends_min / 60.0).astype(int)
        counts = end_hours - start_hours + 1
        counts = np.maximum(counts, 1)
        
        total = counts.sum()
        hours_arr = np.empty(total, dtype=np.int64)
        row_idx = np.empty(total, dtype=np.intp)
        pos = 0
        for i in range(len(counts)):
            c = counts[i]
            hours_arr[pos:pos+c] = np.arange(start_hours[i], start_hours[i]+c)
            row_idx[pos:pos+c] = i
            pos += c
        
        # Build expanded DataFrame
        exp_data = {}
        for col in id_cols:
            if col in data_valid.columns:
                exp_data[col] = data_valid[col].values[row_idx]
        exp_data[index_var] = hours_arr * 60  # Convert back to minutes
        for v in keep_vars:
            if v not in id_cols and v in data_valid.columns:
                exp_data[v] = data_valid[v].values[row_idx]
        
        expanded = pd.DataFrame(exp_data)
        
        # Aggregate to hourly: median for numeric, first for others
        if len(expanded) > 0:
            group_cols = list(id_cols) + [index_var]
            if expanded.duplicated(subset=group_cols).any():
                val_cols = [c for c in expanded.columns if c not in group_cols]
                if val_cols:
                    agg_dict = {}
                    for col in val_cols:
                        if pd.api.types.is_numeric_dtype(expanded[col]):
                            agg_dict[col] = 'median'
                        else:
                            agg_dict[col] = 'first'
                    if agg_dict:
                        expanded = expanded.groupby(group_cols, as_index=False).agg(agg_dict)
        
        return expanded
    
    # 🔧 FIX: R ricu's flow for CareVue data:
    #   1. load_difftime() → change_interval(x, 1h) floors RELATIVE times to hours FIRST
    #   2. callback (mimic_rate_cv) receives ALL rows (including NA rate/amount-only rows)
    #   3. create_intervals(x) on ALL rows — NA rows provide time references for diffs
    #   4. expand(x) — rows where start > end (from 0-diff within same hour) auto-filtered
    #   5. After expand, NA value rows are gone (their intervals had start > end)
    #
    # CRITICAL: R ricu does NOT filter NA rows before create_intervals.
    # The NA rows at later time points provide the diff reference needed to compute
    # correct endtimes. E.g. rate=0 at h=41 + NA at h=48 → diff=7 → endtime=47 → 7 rows.
    # Without the NA row: rate=0 is last → diff=overhang(1) → endtime=41 → only 1 row.
    #
    # CRITICAL: R ricu floors RELATIVE time (time since admission), not absolute datetime.
    # Example: charttime=19:30, intime=14:22 → relative=5.13h → floor=5
    # If we floor absolute datetime: 19:30→19:00 → relative=4.63h → floor=4 (WRONG!)

    data = data.copy()
    
    if data.empty:
        return data
    
    is_datetime = pd.api.types.is_datetime64_any_dtype(data[index_var])
    is_numeric = pd.api.types.is_numeric_dtype(data[index_var])
    
    # Convert datetime to relative-hour-boundary datetimes using admission_times
    # (matching R ricu's load_difftime which floors relative time before callbacks).
    # IMPORTANT: Keep the column as datetime (not float) to avoid type conflicts
    # when multi-source data is concatenated (CareVue datetime + MetaVision datetime).
    admission_times = kwargs.get('admission_times', None)
    converted_to_relative_datetime = False
    if is_datetime and admission_times is not None and id_cols:
        id_col = id_cols[0]
        if id_col in admission_times.columns and 'intime' in admission_times.columns:
            at = admission_times[[id_col, 'intime']].drop_duplicates(subset=[id_col])
            if not pd.api.types.is_datetime64_any_dtype(at['intime']):
                at['intime'] = pd.to_datetime(at['intime'])
            data = data.merge(at, on=id_col, how='left')
            rel_hours = (data[index_var] - data['intime']).dt.total_seconds() / 3600.0
            floored_rel = np.floor(rel_hours)
            data[index_var] = data['intime'] + pd.to_timedelta(floored_rel, unit='h')
            data = data.drop(columns=['intime'])
            converted_to_relative_datetime = True
            # is_datetime stays True — column is still datetime
    
    # Fallback floor for data without admission_times
    if is_datetime and not converted_to_relative_datetime:
        data[index_var] = pd.to_datetime(data[index_var], errors='coerce').dt.floor('h')

    # Create intervals on ALL rows (including NA value rows — they provide time refs)
    data = create_intervals(
        data,
        by_cols=by_cols,
        overhang=pd.Timedelta(hours=1),
        max_len=pd.Timedelta(hours=6),
        end_var='endtime'
    )
    
    # Prepare keep_vars - EXCLUDE grp_var to match R behavior
    if keep_vars is None:
        keep_vars = []
    elif isinstance(keep_vars, str):
        keep_vars = [keep_vars]
    
    keep_vars = list(id_cols) + list(keep_vars)
    keep_vars = [v for v in keep_vars if v in data.columns and v != index_var]
    # NOTE: DO NOT add 'endtime' to keep_vars - it should be removed after expand
    # R ricu expand_intervals does NOT keep endtime in output
    # Keeping it would cause double expand in _load_single_concept
    # Also DO NOT add grp_var - R ricu expand drops it after intervals are created
    
    # Expand with step_size=1 hour
    # Pass admission_times so expand uses relative-hour-aware flooring (not absolute-hour)
    expanded = expand(
        data,
        start_var=index_var,
        end_var='endtime',
        step_size=pd.Timedelta(hours=1),
        id_cols=id_cols,
        keep_vars=keep_vars,
        admission_times=admission_times
    )
    
    # 🔧 After expand, filter out rows where value columns are NA
    # Only for MIMIC CareVue (when admission_times is provided), not for HiRID
    # (NA rows from amount-only CareVue records that had start <= end by chance)
    if admission_times is not None and keep_vars and len(expanded) > 0:
        val_keep = [v for v in keep_vars if v in expanded.columns and v not in id_cols]
        if val_keep:
            mask = expanded[val_keep].notna().any(axis=1)
            expanded = expanded[mask]
    
    # 🔧 CRITICAL: Aggregate duplicate (patient, time) combinations
    # When multiple infusions overlap in time, expand produces multiple rows
    # R ricu's aggregate() uses median for numeric values
    if len(expanded) > 0:
        group_cols = list(id_cols) + [index_var]
        if expanded.duplicated(subset=group_cols).any():
            # Find numeric columns to aggregate
            val_cols = [c for c in expanded.columns if c not in group_cols]
            if val_cols:
                # Use median for numeric columns (R ricu default for numeric)
                agg_dict = {}
                for col in val_cols:
                    if pd.api.types.is_numeric_dtype(expanded[col]):
                        agg_dict[col] = 'median'
                    else:
                        agg_dict[col] = 'first'
                if agg_dict:
                    expanded = expanded.groupby(group_cols, as_index=False).agg(agg_dict)
    
    return expanded

def mimic_rate_cv(
    data: pd.DataFrame,
    val_col: str = 'value',
    grp_var: Optional[str] = None,
    unit_col: Optional[str] = None,
    id_cols: Optional[list] = None,
    admission_times: Optional[pd.DataFrame] = None,
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
        admission_times: DataFrame with id and intime columns for relative time
        **kwargs: Additional arguments
        
    Returns:
        Expanded DataFrame with time series data
    """
    # Build keep_vars
    keep_vars = [val_col]
    if unit_col and unit_col in data.columns:
        keep_vars.append(unit_col)
    
    # Call expand_intervals — pass id_cols and admission_times to avoid false-positive detection
    return expand_intervals(data, keep_vars=keep_vars, grp_var=grp_var, id_cols=id_cols,
                            admission_times=admission_times)

# 注意: hirid_vent 的完整版本在第4104行定义（支持展开到小时级别）

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
        
        # 🔧 FIX pandas 3.0: groupby().apply() drops group columns.
        _grp_backup = data[group_cols].copy()
        data = data.groupby(group_cols, group_keys=False).apply(calc_rate)
        for _gc in group_cols:
            if _gc not in data.columns:
                # Re-add from original (align by index)
                data[_gc] = _grp_backup.loc[data.index, _gc].values if len(data) == len(_grp_backup) else _grp_backup[_gc].iloc[:len(data)].values
        
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
        
        # 检测charttime的类型，确保dur_var与其兼容
        # 如果charttime是数值型（小时），则dur_var也应该是数值型（小时）
        # 如果charttime是datetime型，则dur_var应该是Timedelta
        index_col = None
        for col in ['charttime', 'starttime', 'start', 'time']:
            if col in data.columns:
                index_col = col
                break
        
        if index_col and index_col in data.columns:
            if pd.api.types.is_numeric_dtype(data[index_col]):
                # charttime是数值型（小时），dur_var也用小时
                data['dur_var'] = win_dur.total_seconds() / 3600.0
            else:
                # charttime是datetime型，dur_var用Timedelta
                data['dur_var'] = win_dur
        else:
            # 默认使用Timedelta
            data['dur_var'] = win_dur
            
        return data
    
    return callback

# 注意: fwd_concept 已在第607行定义，此处删除重复定义
# 实际的 fwd_concept 处理逻辑在 concept.py 的 _load_fwd_concept 方法中

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
    
    For MIMIC-III/IV MetaVision inputevents, extracts the infusion rate from
    the `rate` column, falling back to amount/duration when rate is 0 or NA.
    
    This mirrors R ricu's mimv_rate which reads the `rate` column directly
    from inputevents and fills missing values by computing amount/duration.
    
    Args:
        data: Input DataFrame (must contain a 'rate' column from inputevents)
        val_col: Output column name where computed rate will be stored
        unit_col: Unit column (output)
        dur_var: Duration column
        amount_var: Amount column (fallback when rate is 0/NA)
        auom_var: Amount unit of measure column
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with val_col set to the infusion rate (mL/hr or drug/hr)
    """
    data = data.copy()

    # Step 1: if the table has a dedicated 'rate' column (MIMIC inputevents),
    # copy it into val_col. This is the primary source of truth.
    if 'rate' in data.columns and val_col != 'rate':
        data[val_col] = pd.to_numeric(data['rate'], errors='coerce')

    # Step 2: where val_col is still NA or 0, fall back to amount / duration
    mask = data[val_col].isna() | (data[val_col] == 0)

    if mask.any() and dur_var in data.columns:
        dur_series = data.loc[mask, dur_var]

        # Determine duration in hours based on dtype:
        # - timedelta64: use .dt.total_seconds() / 3600
        # - datetime64: this means dur_var holds an end-time, not duration;
        #   but concept.py may also store pre-computed minutes as datetime64
        #   after DuckDB processing – convert to numeric first
        # - numeric (float/int): concept.py stores as minutes (total_seconds/60)
        # - string datetime: try parsing as timedelta then as datetime diff
        if pd.api.types.is_timedelta64_dtype(dur_series):
            dur_hours = dur_series.dt.total_seconds() / 3600
        elif pd.api.types.is_datetime64_any_dtype(dur_series):
            # datetime64 column – try interpreting as numeric minutes
            # (concept.py sometimes stores duration as minutes in datetime col)
            dur_numeric = pd.to_numeric(dur_series, errors='coerce')
            if dur_numeric.notna().any():
                dur_hours = dur_numeric / 60.0
            else:
                dur_hours = pd.Series(np.nan, index=dur_series.index)
        elif pd.api.types.is_numeric_dtype(dur_series):
            # Stored in MINUTES by concept.py (see concept.py:2730, 7457)
            dur_hours = dur_series / 60.0
        else:
            # String or datetime: try converting to timedelta
            try:
                converted = pd.to_timedelta(dur_series, errors='coerce')
            except TypeError:
                converted = pd.to_timedelta(dur_series.astype(str), errors='coerce')
            dur_hours = converted.dt.total_seconds() / 3600

        dur_hours = dur_hours.replace(0, np.nan)
        dur_hours = dur_hours.where(dur_hours > 0, np.nan)

        if amount_var in data.columns:
            data.loc[mask, val_col] = (
                pd.to_numeric(data.loc[mask, amount_var], errors='coerce') / dur_hours
            )
        if auom_var in data.columns and unit_col in data.columns:
            data.loc[mask, unit_col] = data.loc[mask, auom_var].astype(str) + '/hour'

    # Step 3: drop rows that still have no usable rate
    data = data[data[val_col].notna() & (data[val_col] > 0)]

    return data

# 注意: grp_amount_to_rate 已在第1807行定义，此处删除重复的 deprecated wrapper

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
        
        # Detect if time is in float hours (HiRID) or datetime
        time_is_float = result['dur_var'].dtype in ['float64', 'float32', 'int64', 'int32']
        
        if time_is_float:
            # Time is in hours, dur_var is in hours (float)
            # Convert min_dur and extra_dur from Timedelta to hours
            min_dur_hours = min_dur.total_seconds() / 3600
            extra_dur_hours = extra_dur.total_seconds() / 3600
            
            # Apply min_dur for zero-duration events
            zero_dur_mask = result['dur_var'] == 0
            result.loc[zero_dur_mask, 'dur_var'] = min_dur_hours
            
            # Add extra_dur to all durations
            result['dur_var'] = result['dur_var'] + extra_dur_hours
            
            # Calculate rate: amount / duration (dur_var is already in hours)
            dur_hours = result['dur_var']
        else:
            # Time is datetime/timedelta
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
            # Use 'hr' instead of 'hour' to match R ricu conventions (ml/hr, mcg/kg/hr, etc.)
            base_unit = result.get(f"{unit_col}_<lambda>", result.get(unit_col, 'unit'))
            result[unit_col] = base_unit.astype(str) + '/hr'
        
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
    from .api import load_concept
    
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
    # pandas <1.5 returned upper-case codes ('H'/'T'/'S'/...); pandas >=1.5
    # returns 'h'/'min'/'s'/... — accept both forms so this helper stays
    # correct across pandas versions instead of silently defaulting every
    # sub-day Timedelta to 'hour'.
    resolution = x.resolution_string

    unit_map = {
        # pandas 1.x (upper-case)
        'D': 'day',
        'H': 'hour',
        'T': 'min',
        'S': 'sec',
        'L': 'millisec',
        'U': 'microsec',
        'N': 'nanosec',
        # pandas 2.x (lower-case, plus 'min' written out)
        'h': 'hour',
        'min': 'min',
        's': 'sec',
        'ms': 'millisec',
        'us': 'microsec',
        'ns': 'nanosec',
    }

    return unit_map.get(resolution, 'hour')

def eicu_rate_kg_callback(ml_to_mcg: float) -> Callable:
    """eICU dose rate conversion with weight normalization (R ricu eicu_rate_kg).
    
    Converts various dose rate units to mcg/kg/min, following R ricu logic:
    1. First apply unit conversions:
       - /hr -> /min (divide by 60)
       - mg/ -> mcg/ (multiply by 1000)
       - units/ -> NA (not convertible)
       - ml/ -> mcg/ (multiply by ml_to_mcg)
       - nanograms/ -> mcg/ (divide by 1000)
       - Unknown/ml -> NA
    2. Then for non-/kg/ rates, divide by patient weight (from patient table)
    
    Args:
        ml_to_mcg: Conversion factor from ml to mcg (drug concentration)
        
    Returns:
        Callback function
        
    Examples:
        >>> # Norepinephrine: ml_to_mcg=32 (standard concentration)
        >>> norepi_callback = eicu_rate_kg_callback(ml_to_mcg=32)
    """
    def callback(
        frame: pd.DataFrame,
        val_var: str,
        sub_var: str,
        weight_var: str,
        concept_name: str,
        data_source=None,
        patient_ids=None,
    ) -> pd.DataFrame:
        """Apply eICU rate/kg conversion following R ricu logic.
        
        Args:
            frame: Input dataframe
            val_var: Value column name (e.g., 'drugrate')
            sub_var: Sub-variable column containing unit info (e.g., 'drugname')
            weight_var: Weight column name (from patient table)
            concept_name: Output concept name
            data_source: ICUDataSource for loading weight concept
            patient_ids: Patient IDs for weight loading
            
        Returns:
            Converted dataframe
        """
        frame = frame.copy()
        
        # Convert values to numeric
        if val_var in frame.columns:
            frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
        
        # Extract unit from sub_var (e.g., "Norepinephrine (mcg/min)" -> "mcg/min")
        if sub_var in frame.columns:
            # 🚀 Vectorized: str.extract + fallback for bare units
            _sub = frame[sub_var].astype(str)
            _extracted = _sub.str.extract(r'\(([^)]+)\)$', expand=False)
            # Fallback: if no parenthesized unit, check if value itself looks like a unit
            _bare_unit = _sub.str.contains(r'/', na=False) | _sub.str.lower().isin(['mg', 'mcg', 'ml', 'units'])
            _fallback = _bare_unit & _extracted.isna()
            _extracted[_fallback] = _sub[_fallback]
            # NaN original → None
            _extracted[frame[sub_var].isna()] = None
            frame['unit_var'] = _extracted
        else:
            frame['unit_var'] = 'Unknown'
        
        # Get weight from patient table (following R ricu add_weight logic)
        # First check if weight_var exists in frame
        if weight_var in frame.columns:
            frame['_weight'] = pd.to_numeric(frame[weight_var], errors='coerce')
        else:
            frame['_weight'] = np.nan
        
        # Load weight from patient table if data_source is available
        if data_source is not None and frame['_weight'].isna().any():
            try:
                from .datasource import FilterSpec, FilterOp
                
                # Determine ID column
                id_col = None
                for candidate in ['patientunitstayid', 'stay_id', 'hadm_id', 'icustay_id']:
                    if candidate in frame.columns:
                        id_col = candidate
                        break
                
                if id_col:
                    # Load weight concept
                    patient_list = frame[id_col].unique().tolist()
                    weight_table = data_source.load_table(
                        'patient',
                        columns=['patientunitstayid', 'admissionweight'],
                        filters=[FilterSpec(column='patientunitstayid', op=FilterOp.IN, value=patient_list)]
                    )
                    
                    # Extract DataFrame from ICUTable if needed
                    if hasattr(weight_table, 'data'):
                        weight_df = weight_table.data
                    else:
                        weight_df = weight_table
                    
                    if weight_df is not None and len(weight_df) > 0:
                        weight_df = weight_df.rename(columns={'admissionweight': '_loaded_weight'})
                        weight_df['_loaded_weight'] = pd.to_numeric(weight_df['_loaded_weight'], errors='coerce')
                        
                        # Merge weight
                        frame = frame.merge(
                            weight_df[['patientunitstayid', '_loaded_weight']],
                            on='patientunitstayid',
                            how='left'
                        )
                        
                        # Fill NaN weights with loaded weight
                        mask = frame['_weight'].isna()
                        frame.loc[mask, '_weight'] = frame.loc[mask, '_loaded_weight']
                        frame = frame.drop(columns=['_loaded_weight'], errors='ignore')
            except Exception as e:
                logging.debug(f"Failed to load weight from patient table: {e}")
        
        # Fill remaining missing weights with default 70 kg
        # Also replace 0 or negative weights with default (prevents division by zero)
        frame['_weight'] = frame['_weight'].fillna(70.0)
        frame.loc[frame['_weight'] <= 0, '_weight'] = 70.0
        
        # 🚀 Vectorized unit conversion (replaces per-row apply, ~50x faster)
        _val = pd.to_numeric(frame[val_var], errors='coerce').values.astype(np.float64).copy()
        _unit = frame['unit_var'].astype(str).str.strip().str.lower()
        _wt = frame['_weight'].values.astype(np.float64).copy()
        _wt_bad = np.isnan(_wt) | (_wt <= 0)
        _wt[_wt_bad] = 70.0

        # Incompatible units → NaN (include NaN/empty units — original code: `if not unit: return np.nan`)
        _unit_raw_na = frame['unit_var'].isna()
        _invalid = _unit_raw_na | _unit.str.startswith('units/') | _unit.isin(['unknown', 'ml', '', 'nan', 'none'])
        _val[_invalid.values] = np.nan
        _val[np.isnan(_val)] = np.nan  # preserve original NaN

        # Step 1: /hr → /min (÷60)  — applied before mg/ml prefix checks
        _hr = _unit.str.contains('/hr', na=False).values & ~_invalid.values
        _val[_hr] /= 60
        # Update unit strings for subsequent prefix checks (mg/hr → mg/min, etc.)
        _unit_arr = _unit.values.copy()  # numpy object array for fast mutation
        if _hr.any():
            _unit_arr[_hr] = np.array([u.replace('/hr', '/min') for u in _unit_arr[_hr]])

        # Vectorized prefix checks on (possibly updated) unit strings
        _unit_s = pd.Series(_unit_arr)

        # Step 2: mg/ → mcg/ (×1000)
        _mg = _unit_s.str.startswith('mg/').values & ~_invalid.values
        _val[_mg] *= 1000
        if _mg.any():
            _unit_arr[_mg] = np.array(['mcg' + u[2:] for u in _unit_arr[_mg]])

        # Step 3: ml/ → mcg/ (×ml_to_mcg)
        _ml = _unit_s.str.startswith('ml/').values & ~_invalid.values
        _val[_ml] *= ml_to_mcg
        if _ml.any():
            _unit_arr[_ml] = np.array(['mcg' + u[2:] for u in _unit_arr[_ml]])

        # Step 4: nanograms/ → mcg/ (÷1000)
        _ng = _unit_s.str.startswith('nanograms/').values & ~_invalid.values
        _val[_ng] /= 1000

        # Step 5: non-/kg/ units → ÷weight
        _unit_final = pd.Series(_unit_arr)
        _no_kg = ~_unit_final.str.contains('/kg/', na=False).values & ~_invalid.values
        _val[_no_kg] /= _wt[_no_kg]

        frame[concept_name] = _val
        
        # Clean up temporary columns
        frame = frame.drop(columns=['unit_var', '_weight'], errors='ignore')
        
        # Expand intervals to match R ricu's expand_intervals behavior
        # For eICU, infusionoffset is in minutes - need to convert to hours and expand
        
        # Check for infusionoffset column (eICU-specific)
        time_col = None
        for candidate in ['infusionoffset', 'charttime', 'starttime']:
            if candidate in frame.columns:
                time_col = candidate
                break
        
        if time_col is None:
            return frame
        
        # Determine ID column
        id_col = None
        for candidate in ['patientunitstayid', 'stay_id', 'hadm_id', 'icustay_id']:
            if candidate in frame.columns:
                id_col = candidate
                break
        
        if id_col is None:
            return frame
        
        # Remove rows with NaN concept values (already converted)
        frame = frame[frame[concept_name].notna()].copy()
        
        if len(frame) == 0:
            result_cols = [id_col, time_col, concept_name]
            return pd.DataFrame(columns=result_cols)
        
        # R ricu expand_intervals logic for eICU:
        # 1. Convert minutes to hours (floor division)
        # 2. Aggregate by hour (take max if multiple values)
        # 3. Create intervals: diff = min(next_hour - current_hour, max_len) - interval
        # 4. Expand each record to [current_hour, current_hour + diff]
        
        # Step 0: Sort by time so groupby.agg('last') picks the chronologically last record
        frame = frame.sort_values([id_col, time_col]).reset_index(drop=True)
        
        # Step 1: Convert to hours
        frame['_hour'] = (frame[time_col] // 60).astype(int)
        
        # Step 2: Aggregate by patient and hour
        # CRITICAL: R ricu uses 'last' (chronological order), not 'max'.
        # When infusion stops, a rate=0 record is written. Using 'max' would
        # incorrectly keep the pre-stop rate, while 'last' correctly picks up 0.
        hourly = frame.groupby([id_col, '_hour'], as_index=False).agg({
            concept_name: 'last'
        })
        hourly = hourly.sort_values([id_col, '_hour'])
        
        # Step 3: Create intervals using R ricu's create_intervals logic
        # R: endtime = padded_diff(hour, overhang=1)  # diff to next, or 1 for last
        # R: endtime = trunc(endtime, 0, max_len=6) - interval=1
        # R: endtime = hour + endtime
        overhang = 1  # hours
        max_len = 6   # hours  
        interval = 1  # hours (time step)
        
        def create_intervals_r(group):
            group = group.copy()
            # padded_diff: next - current, or overhang for last
            group['_diff'] = group['_hour'].shift(-1) - group['_hour']
            group.loc[group['_diff'].isna(), '_diff'] = overhang
            # trunc to max_len
            group['_diff'] = group['_diff'].clip(upper=max_len)
            # subtract interval (key step to avoid overlap!)
            group['_diff'] = group['_diff'] - interval
            # Calculate end hour
            group['_end_hour'] = group['_hour'] + group['_diff']
            return group
        
        # 🔧 FIX pandas 3.0: groupby().apply() drops group columns.
        # Use vectorized groupby().shift() instead.
        hourly = hourly.copy()
        hourly['_diff'] = hourly.groupby(id_col)['_hour'].shift(-1) - hourly['_hour']
        hourly.loc[hourly['_diff'].isna(), '_diff'] = overhang
        hourly['_diff'] = hourly['_diff'].clip(upper=max_len)
        hourly['_diff'] = hourly['_diff'] - interval
        hourly['_end_hour'] = hourly['_hour'] + hourly['_diff']
        
        # Step 4: Vectorized expand using numpy repeat + arange
        start_hours = hourly['_hour'].values.astype(int)
        end_hours = hourly['_end_hour'].values.astype(int)
        n_points = np.maximum(end_hours - start_hours + 1, 1)
        total_points = n_points.sum()
        
        if total_points > 0:
            # Repeat each row's patient_id and value n_points times
            patient_ids_exp = np.repeat(hourly[id_col].values, n_points)
            values_exp = np.repeat(hourly[concept_name].values, n_points)
            # Generate time points (in minutes for eICU _align_time_to_admission)
            time_points = np.concatenate([
                np.arange(s, s + n) * 60
                for s, n in zip(start_hours, n_points)
            ])
            expanded = pd.DataFrame({
                id_col: patient_ids_exp,
                time_col: time_points,
                concept_name: values_exp,
            })
        
        # If multiple values at same hour (from overlapping intervals), take max
        # R ricu default aggregation for rate concepts is 'max'
        expanded = expanded.groupby([id_col, time_col], as_index=False).agg({
            concept_name: 'max'
        })
        
        # Sort final result
        expanded = expanded.sort_values([id_col, time_col]).reset_index(drop=True)
        
        # 🔧 CRITICAL FIX: Apply LOCF (last observation carried forward) to fill gaps
        # R ricu's expand_intervals creates the interval data, then locf fills gaps
        # This ensures continuous time series from min to max hour
        def fill_gaps_locf(group):
            if len(group) < 2:
                return group
            
            # Get hour range
            min_hour = int(group[time_col].min() / 60)  # Convert back from minutes
            max_hour = int(group[time_col].max() / 60)
            
            # Create complete hourly grid
            all_hours = list(range(min_hour, max_hour + 1))
            all_minutes = [h * 60 for h in all_hours]
            
            # Create grid dataframe - get id from group.name (set by groupby)
            grid = pd.DataFrame({
                id_col: group.name if not isinstance(group.name, tuple) else group.name[0],
                time_col: all_minutes
            })
            
            # Merge with data
            merged = grid.merge(group[[time_col, concept_name]], on=time_col, how='left')
            
            # Forward fill (locf)
            merged[concept_name] = merged[concept_name].ffill()
            
            return merged
        
        expanded = expanded.groupby(id_col, group_keys=False).apply(fill_gaps_locf, include_groups=False)
        expanded = expanded.reset_index(drop=True)
        
        return expanded
    
    return callback

def eicu_rate_units_callback(ml_to_mcg: float, mcg_to_units: float) -> Callable:
    """Convert eICU medication rates to units/min (R ricu eicu_rate_units).

    Args:
        ml_to_mcg: Conversion factor from millilitres to micrograms.
        mcg_to_units: Conversion factor from micrograms to drug-specific units.

    Returns:
        Callback that normalises rate units and expands durations to hourly intervals.
    """

    if ml_to_mcg <= 0 or mcg_to_units <= 0:
        raise ValueError("Conversion factors must be positive numbers")

    def _normalize_units(frame: pd.DataFrame, val_var: str, unit_col: str) -> pd.DataFrame:
        work = frame.copy()
        work[unit_col] = work[unit_col].fillna("")

        # 1) '/hr' -> '/min'
        mask = work[unit_col].str.contains(r"/hr$", case=False, na=False)
        if mask.any():
            work.loc[mask, val_var] = work.loc[mask, val_var] / 60.0
            work.loc[mask, unit_col] = work.loc[mask, unit_col].str.replace(
                r"/hr$", "/min", regex=True, flags=re.IGNORECASE
            )

        # 2) 'mg/' -> 'mcg/'
        mask = work[unit_col].str.contains(r"^mg/", case=False, na=False)
        if mask.any():
            work.loc[mask, val_var] = work.loc[mask, val_var] * 1000.0
            work.loc[mask, unit_col] = work.loc[mask, unit_col].str.replace(
                r"^mg/", "mcg/", regex=True, flags=re.IGNORECASE
            )

        # 3) Entries with '/kg/' are not convertible → mark as missing units/min
        mask = work[unit_col].str.contains(r"/kg/", case=False, na=False)
        if mask.any():
            work.loc[mask, val_var] = np.nan
            work.loc[mask, unit_col] = "units/min"

        # 4) 'ml/' -> 'mcg/' using concentration factor
        mask = work[unit_col].str.contains(r"^ml/", case=False, na=False)
        if mask.any():
            work.loc[mask, val_var] = work.loc[mask, val_var] * ml_to_mcg
            work.loc[mask, unit_col] = work.loc[mask, unit_col].str.replace(
                r"^ml/", "mcg/", regex=True, flags=re.IGNORECASE
            )

        # 5) 'mcg/' -> 'units/' using microgram-to-unit factor
        mask = work[unit_col].str.contains(r"^mcg/", case=False, na=False)
        if mask.any():
            work.loc[mask, val_var] = work.loc[mask, val_var] * mcg_to_units
            work.loc[mask, unit_col] = work.loc[mask, unit_col].str.replace(
                r"^mcg/", "units/", regex=True, flags=re.IGNORECASE
            )

        return work

    def callback(
        frame: pd.DataFrame,
        val_var: str,
        sub_var: Optional[str],
        concept_name: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame

        work = frame.copy()
        
        # 🔧 FIX: 回调可能在列重命名后调用，val_var 可能已被重命名为 concept_name
        # 优先使用 concept_name（重命名后的列），然后是 val_var
        actual_val_var = None
        if concept_name in work.columns:
            actual_val_var = concept_name
        elif val_var in work.columns:
            actual_val_var = val_var
        else:
            # 尝试其他常见列名
            for col in ['drugrate', 'rate', 'value']:
                if col in work.columns:
                    actual_val_var = col
                    break
        
        if actual_val_var is None:
            # 如果都找不到，返回空 DataFrame
            return pd.DataFrame(columns=frame.columns)
        
        # 使用找到的值列，并创建统一的列名用于后续处理
        work[val_var] = pd.to_numeric(work[actual_val_var], errors="coerce")

        if sub_var and sub_var in work.columns:
            work["unit_var"] = eicu_extract_unit(work[sub_var])
        else:
            work["unit_var"] = np.nan

        work = _normalize_units(work, val_var, "unit_var")

        # Expand into hourly windows so that easyicu matches ricu's exposure logic.
        expanded = expand_intervals(work, keep_vars=[val_var, "unit_var"])
        return expanded

    return callback

def eicu_rate_mass_callback(target_unit: str) -> Callable:
    """eICU rate conversion for mass-rate drugs (no weight normalization).

    Mirrors :func:`eicu_rate_kg_callback` but omits the final ÷weight step,
    because the concept is a bare mass/time rate (e.g. mcg/hour fentanyl,
    mg/hour midazolam) rather than a weight-indexed rate.

    Strategy
    --------
    1. Parse the unit from ``drugname`` using the same ``(mcg/hr)``-style
       parenthesized suffix pattern used elsewhere in this module.
    2. Drop rows whose unit is NOT mass/time. Specifically incompatible:
         - ``/kg/`` anything — these rows need weight re-normalization, which
           would be a separate callback; mixing in this mass-rate pipeline
           would silently inflate per-kg-rate patients.
         - ``ml/`` anything — we don't know the drug concentration for every
           eICU drugname variant, so can't convert to mass rate safely.
         - Empty / ``unknown`` / ``nan`` / ``()`` / ``cont`` — unusable.
         - ``units/`` anything — for non-mass drugs.
    3. Convert the remaining mass/time rate to ``target_unit`` using the
       same conversion factors as the kg variant.
    4. Run the same interval-expansion logic as ``eicu_rate_kg_callback`` so
       rates align to the ricu hourly grid with LOCF gap filling.

    Supported target units
    ----------------------
    ``mcg/hour``, ``mcg/min``, ``mg/hour``, ``mg/min``

    Examples
    --------
    >>> # Fentanyl in mcg/hour (drugnames like "fentanyl (mcg/hr)")
    >>> fen_cb = eicu_rate_mass_callback("mcg/hour")
    >>> # Midazolam in mg/hour
    >>> mid_cb = eicu_rate_mass_callback("mg/hour")
    """
    target = target_unit.lower().replace("hr", "hour").strip()
    if target not in {"mcg/hour", "mcg/min", "mg/hour", "mg/min"}:
        raise ValueError(
            f"eicu_rate_mass_callback: unsupported target_unit {target_unit!r}. "
            "Must be one of mcg/hour, mcg/min, mg/hour, mg/min."
        )
    # Split into (mass, time)
    target_mass, target_time = target.split("/")

    def callback(
        frame: pd.DataFrame,
        val_var: str,
        sub_var: str,
        concept_name: str,
        data_source=None,
        patient_ids=None,
    ) -> pd.DataFrame:
        """Apply eICU mass-rate conversion (no ÷weight) and expand intervals."""
        frame = frame.copy()

        if val_var in frame.columns:
            frame[val_var] = pd.to_numeric(frame[val_var], errors="coerce")

        # Extract unit from drugname (same pattern as eicu_rate_kg)
        if sub_var in frame.columns:
            _sub = frame[sub_var].astype(str)
            _extracted = _sub.str.extract(r"\(([^)]+)\)$", expand=False)
            _bare_unit = _sub.str.contains(r"/", na=False) | _sub.str.lower().isin(
                ["mg", "mcg", "ml", "units"]
            )
            _fallback = _bare_unit & _extracted.isna()
            _extracted[_fallback] = _sub[_fallback]
            _extracted[frame[sub_var].isna()] = None
            frame["unit_var"] = _extracted
        else:
            frame["unit_var"] = None

        _val = pd.to_numeric(frame[val_var], errors="coerce").values.astype(np.float64).copy()
        _unit = frame["unit_var"].astype(str).str.strip().str.lower()
        _unit_raw_na = frame["unit_var"].isna()

        # Row-level invalid flags → NaN (will be dropped before expand)
        _invalid = (
            _unit_raw_na
            | _unit.str.contains("/kg/", na=False)  # would need kg re-norm
            | _unit.str.startswith("ml/", na=False)  # no concentration
            | _unit.str.startswith("units/", na=False)  # non-mass
            | _unit.isin(["unknown", "ml", "", "nan", "none", "cont"])
        )
        _val[_invalid.values] = np.nan

        # Apply unit conversions row-wise to reach target.
        # Start unit string (post-lowercase). Build working array we can mutate.
        _unit_arr = _unit.values.copy()
        _valid = ~_invalid.values

        # 1) Normalize time suffix
        _hr = _unit.str.contains("/hr", na=False).values & _valid
        _hour = _unit.str.contains("/hour", na=False).values & _valid
        # /hr → /min if target_time is min (×1/60), no-op if target is hour
        if target_time == "min":
            mask_need_to_min = (_hr | _hour) & _valid
            _val[mask_need_to_min] /= 60.0
            if mask_need_to_min.any():
                _unit_arr[mask_need_to_min] = np.array(
                    [u.replace("/hr", "/min").replace("/hour", "/min")
                     for u in _unit_arr[mask_need_to_min]]
                )
        else:  # target_time == "hour"
            _per_min = _unit.str.endswith("/min", na=False).values & _valid
            _val[_per_min] *= 60.0
            if _per_min.any():
                _unit_arr[_per_min] = np.array(
                    [u.replace("/min", "/hour") for u in _unit_arr[_per_min]]
                )
            # Normalize /hr label
            _hr_rows = _hr & ~_hour
            if _hr_rows.any():
                _unit_arr[_hr_rows] = np.array(
                    [u.replace("/hr", "/hour") for u in _unit_arr[_hr_rows]]
                )

        # 2) Normalize mass prefix (after time normalization)
        _unit_s = pd.Series(_unit_arr)
        if target_mass == "mcg":
            _mg = _unit_s.str.startswith("mg/").values & _valid
            _val[_mg] *= 1000.0
            if _mg.any():
                _unit_arr[_mg] = np.array(["mcg" + u[2:] for u in _unit_arr[_mg]])
            _ng = _unit_s.str.startswith("nanograms/").values & _valid
            _val[_ng] /= 1000.0
        else:  # target_mass == "mg"
            _mcg = _unit_s.str.startswith("mcg/").values & _valid
            _val[_mcg] /= 1000.0
            if _mcg.any():
                _unit_arr[_mcg] = np.array(["mg" + u[3:] for u in _unit_arr[_mcg]])

        # 3) Final guard: only keep rows that now land exactly on the target unit.
        # We require the resulting mass/time tokens to match.
        _final_unit = pd.Series(_unit_arr)
        target_norm = f"{target_mass}/{target_time}"
        _ok = _final_unit.str.lower().str.fullmatch(re.escape(target_norm)).fillna(False).values
        _val[~_ok] = np.nan

        frame[concept_name] = _val
        frame = frame.drop(columns=["unit_var"], errors="ignore")

        # Find time and ID columns (matches eicu_rate_kg_callback)
        time_col = None
        for candidate in ["infusionoffset", "charttime", "starttime"]:
            if candidate in frame.columns:
                time_col = candidate
                break
        if time_col is None:
            return frame
        id_col = None
        for candidate in [
            "patientunitstayid", "stay_id", "hadm_id", "icustay_id"
        ]:
            if candidate in frame.columns:
                id_col = candidate
                break
        if id_col is None:
            return frame

        # Drop rows with NaN concept value (incompatible units filtered out)
        frame = frame[frame[concept_name].notna()].copy()
        if len(frame) == 0:
            return pd.DataFrame(columns=[id_col, time_col, concept_name])

        # R ricu-compatible interval expansion — identical to eicu_rate_kg
        frame = frame.sort_values([id_col, time_col]).reset_index(drop=True)
        frame["_hour"] = (frame[time_col] // 60).astype(int)
        hourly = frame.groupby([id_col, "_hour"], as_index=False).agg(
            {concept_name: "last"}
        )
        hourly = hourly.sort_values([id_col, "_hour"])

        overhang = 1
        max_len = 6
        interval = 1

        hourly = hourly.copy()
        hourly["_diff"] = hourly.groupby(id_col)["_hour"].shift(-1) - hourly["_hour"]
        hourly.loc[hourly["_diff"].isna(), "_diff"] = overhang
        hourly["_diff"] = hourly["_diff"].clip(upper=max_len)
        hourly["_diff"] = hourly["_diff"] - interval
        hourly["_end_hour"] = hourly["_hour"] + hourly["_diff"]

        start_hours = hourly["_hour"].values.astype(int)
        end_hours = hourly["_end_hour"].values.astype(int)
        n_points = np.maximum(end_hours - start_hours + 1, 1)
        total_points = n_points.sum()

        if total_points == 0:
            return pd.DataFrame(columns=[id_col, time_col, concept_name])

        patient_ids_exp = np.repeat(hourly[id_col].values, n_points)
        values_exp = np.repeat(hourly[concept_name].values, n_points)
        time_points = np.concatenate([
            np.arange(s, s + n) * 60
            for s, n in zip(start_hours, n_points)
        ])
        expanded = pd.DataFrame({
            id_col: patient_ids_exp,
            time_col: time_points,
            concept_name: values_exp,
        })
        expanded = expanded.groupby([id_col, time_col], as_index=False).agg(
            {concept_name: "max"}
        )
        expanded = expanded.sort_values([id_col, time_col]).reset_index(drop=True)

        # LOCF gap-fill across each patient's [min, max] hour range
        def fill_gaps_locf(group):
            if len(group) < 2:
                return group
            min_hour = int(group[time_col].min() / 60)
            max_hour = int(group[time_col].max() / 60)
            all_minutes = [h * 60 for h in range(min_hour, max_hour + 1)]
            grid = pd.DataFrame({
                id_col: group.name if not isinstance(group.name, tuple) else group.name[0],
                time_col: all_minutes,
            })
            merged = grid.merge(
                group[[time_col, concept_name]], on=time_col, how="left"
            )
            merged[concept_name] = merged[concept_name].ffill()
            return merged

        expanded = expanded.groupby(id_col, group_keys=False).apply(
            fill_gaps_locf, include_groups=False
        )
        expanded = expanded.reset_index(drop=True)
        return expanded

    return callback


def _infer_interval_from_series(series: pd.Series) -> pd.Timedelta:
    """Best-effort detection of interval spacing for offset/time columns."""

    values = series.dropna()
    if values.empty:
        return pd.Timedelta(hours=1)

    if pd.api.types.is_datetime64_any_dtype(values):
        ordered = values.sort_values()
        diffs = ordered.diff()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if not diffs.empty:
            return diffs.min()

    if pd.api.types.is_timedelta64_dtype(values):
        ordered = values.sort_values()
        diffs = ordered.diff()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if not diffs.empty:
            return diffs.min()

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if not numeric.empty:
        ordered = numeric.sort_values()
        diffs = ordered.diff()
        diffs = diffs[diffs > 0]
        if not diffs.empty and diffs.min() > 0:
            minutes = diffs.min()
            return pd.to_timedelta(minutes, unit="m")

    return pd.Timedelta(hours=1)

def eicu_dex_med(
    frame: pd.DataFrame,
    val_var: str,
    dur_var: str,
    concept_name: str,
) -> pd.DataFrame:
    """Dexmedetomidine eICU infusion normalisation (R ricu eicu_dex_med).
    
    R ricu logic (callback-itm.R line 856-872):
    1. Split dosage into value and unit
    2. If unit is mg, multiply by 2
    3. Filter: duration > 0 (set to 1 min if <= 0)
    4. Filter: duration <= 12 hours
    5. rate = value / duration_minutes * 5
    """

    if val_var not in frame.columns or dur_var not in frame.columns:
        return frame

    work = frame.copy()

    # Split textual dose "<value> <unit>" into numeric value + unit column
    tokens = (
        work[val_var]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.split(" ", n=1, expand=True)
    )
    # Handle case where split produces only one column (no space in value)
    if tokens.shape[1] >= 1:
        work[val_var] = tokens.iloc[:, 0]
        work["unit_var"] = tokens.iloc[:, 1] if tokens.shape[1] > 1 else np.nan
    else:
        work["unit_var"] = np.nan

    work[val_var] = pd.to_numeric(
        work[val_var].astype(str).str.replace(r"^(.+-|Manual)", "", regex=True), errors="coerce"
    )

    mg_mask = work["unit_var"].astype(str).str.contains(r"^m?g.*m?", case=False, na=False)
    if mg_mask.any():
        work.loc[mg_mask, val_var] = work.loc[mg_mask, val_var] * 2.0

    # dur_var可能是:
    # 1. {concept_name}_dur - 已计算的duration，单位是小时（dur_is_end逻辑）
    # 2. drugstopoffset - 原始offset，单位是分钟
    # 需要判断并统一转换为分钟
    
    dur_vals = pd.to_numeric(work[dur_var], errors="coerce")
    
    # 判断dur_var是否是已计算的duration列（单位是小时）
    is_computed_duration = dur_var.endswith('_dur')
    
    if is_computed_duration:
        # dur_var是已计算的duration，单位是小时，转换为分钟
        duration_minutes = dur_vals * 60.0
    else:
        # eICU medication 的原始 dur_var 往往是 stop offset，而不是持续时间。
        # 对 D50 这种短时给药，RICU 使用 stop-start 作为持续时间；若直接把 stop offset
        # 当 duration，会在负 offset/极小值场景下退化成 1 分钟，从而把速率放大到 7500 ml/hr。
        start_candidates = [
            "drugstartoffset",
            "startoffset",
            f"{concept_name}_start",
            "start",
        ]
        start_col = next((col for col in start_candidates if col in work.columns), None)
        if start_col is not None:
            start_vals = pd.to_numeric(work[start_col], errors="coerce")
            duration_minutes = dur_vals - start_vals
        else:
            duration_minutes = dur_vals

    # Filter: duration <= 0 set to 1 min
    duration_minutes = duration_minutes.fillna(1.0)
    duration_minutes = duration_minutes.where(duration_minutes > 0, 1.0)

    # Filter: duration <= 12 hours (720 minutes)
    mask = duration_minutes <= 720.0
    work = work.loc[mask].copy()
    duration_minutes = duration_minutes.loc[mask]

    # rate = value / duration_minutes * 5 (ml/min)
    # 🔧 FIX 2025-02-03: 转换为 ml/hr 以匹配概念定义 (unit: "ml/hr")
    # R ricu 输出的是 ml/min，但概念定义是 ml/hr
    duration_minutes = duration_minutes.where(duration_minutes > 0, 1.0)
    rate_ml_min = work[val_var] / duration_minutes * 5.0
    work[val_var] = rate_ml_min * 60.0  # 转换为 ml/hr
    work["unit_var"] = "ml/hr"
    
    # 保存duration（小时，与charttime单位一致，用于后续expand）
    work[dur_var] = duration_minutes / 60.0

    return work

def eicu_dex_inf(
    frame: pd.DataFrame,
    val_var: str,
    index_var: Optional[str],
) -> pd.DataFrame:
    """Normalize eICU dex infusion TS rows to win-table compatible rows."""

    if frame.empty or val_var not in frame.columns:
        return frame

    work = frame.copy()
    work[val_var] = pd.to_numeric(work[val_var], errors="coerce")

    idx_col = index_var
    if not idx_col or idx_col not in work.columns:
        candidates = [
            col
            for col in work.columns
            if col.lower().endswith("offset") or col.lower().endswith("time")
        ]
        idx_col = candidates[0] if candidates else None

    interval = pd.Timedelta(hours=1)
    if idx_col and idx_col in work.columns:
        interval = _infer_interval_from_series(work[idx_col])

    # 🔧 FIX: Output dur_var in MINUTES (eICU native unit) so that
    # _align_time_to_admission's auto-detect correctly converts dur_var / 60 → hours.
    # Previously output hours → auto-detect would double-convert (hours/60 = wrong).
    interval_minutes = interval.total_seconds() / 60.0
    work["dur_var"] = interval_minutes
    work["unit_var"] = "ml/hr"

    return work

def _aumc_get_id_columns(df: pd.DataFrame) -> List[str]:
    """Get the actual patient/stay ID columns, not all columns ending with 'id'.
    
    This is used for grouping in aggregation. We only want the true identifier
    columns (admissionid, icustay_id, stay_id, etc.), not item/order IDs.
    """
    # True ID columns for ICU data
    true_id_cols = ['admissionid', 'icustay_id', 'stay_id', 'subject_id', 'patientid', 
                    'hadm_id', 'patientunitstayid']
    return [col for col in df.columns if col.lower() in [c.lower() for c in true_id_cols]]

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

def _aumc_normalize_rate_units(df: pd.DataFrame, rate_uom_col: Optional[str], val_col: str, 
                               default: str = 'min', interval_mins: float = 60.0) -> Optional[str]:
    """
    Normalize AUMC rate units to per-minute.
    
    This function handles:
    1. Converting 'uur' (hour) to min: divide value by 60
    2. Converting 'dag' (day) to min: divide value by 1440
    3. Converting bolus doses (NA rate_uom) to per-minute: divide value by interval
    
    R ricu's aumc_rate_units does this (callback-itm.R lines 599-602):
        x <- x[is.na(get(rate_uom)), c(val_var, rate_uom) := list(
          sum(get(val_var)) * frac, "min"), by = c(meta_vars(x))
        ]
    where frac = 1 / interval_in_minutes (typically 1/60 for hourly interval)
    
    Args:
        df: DataFrame to modify in-place
        rate_uom_col: Name of the rate unit column
        val_col: Name of the value column  
        default: Default rate unit if column doesn't exist
        interval_mins: Interval in minutes for bolus dose conversion (default 60)
    """
    if not rate_uom_col:
        return None
    if rate_uom_col not in df.columns:
        # If no rate_uom column exists, treat all as bolus doses
        # Convert by dividing by interval (e.g., 60 mins) to get per-minute rate
        df[val_col] = df[val_col] / interval_mins
        df[rate_uom_col] = 'min'
        return rate_uom_col

    # Convert to string and handle NA values
    # First identify actual NA/None values before converting to string
    is_na_mask = df[rate_uom_col].isna()
    
    df[rate_uom_col] = df[rate_uom_col].astype(str).str.strip()
    rate_lower = df[rate_uom_col].str.lower()
    
    # Expand NA mask to include string versions of NA
    is_na_mask = is_na_mask | rate_lower.isin({'nan', 'none', 'nat', ''})
    
    # Handle bolus doses (NA rate_uom) - R ricu divides by interval
    # This is the key fix: bolus doses need to be converted to per-minute rate
    if is_na_mask.any():
        df.loc[is_na_mask, val_col] = df.loc[is_na_mask, val_col] / interval_mins
        df.loc[is_na_mask, rate_uom_col] = 'min'

    # Recalculate rate_lower after NA handling
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

    # Final cleanup - ensure all are 'min'
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

    # 🔧 FIX: Match R ricu rm_na(x, c(unit_var, rate_uom), "any")
    # Remove rows where unit_col or rate_unit_col is NA
    # This is critical for AUMC epi_rate where some records have no doserateunit
    na_cols = []
    if unit_col and unit_col in df.columns:
        na_cols.append(unit_col)
    if rate_unit_col and rate_unit_col in df.columns:
        na_cols.append(rate_unit_col)
    if na_cols:
        df = df.dropna(subset=na_cols, how='any')
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
    # 🚀 FIX: 不要在这里转换时间单位！
    # datasource.py 已经把 AUMC 时间从毫秒转换为分钟
    # _align_time_to_admission (concept.py) 会统一把分钟转换为小时
    # 如果这里也做转换，会导致时间被除以 60 两次，变得非常小
    # 保持时间列为分钟，让 _align_time_to_admission 统一处理

    df[concept_name] = df[val_col]

    id_cols = _aumc_get_id_columns(df)
    result_cols = list(dict.fromkeys(id_cols))
    
    # 确保时间列总是包含在返回中(即使为空或不存在)
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

    result = df[result_cols].dropna(subset=[concept_name])
    
    # 🔧 CRITICAL: Call expand() like R ricu does
    # R ricu: expand(res, index_var(x), stop_var, keep_vars = c(id_vars(x), val_var, unit_var))
    # This expands interval data (start/stop) into hourly time points
    # Without this, we get only ~40 rows instead of ~1000 rows
    if stop_col and stop_col in df.columns and index_col and index_col in df.columns:
        # Add stop_col to result for expand
        result[stop_col] = df.loc[result.index, stop_col]
        
        # Time is in minutes (from datasource), expand at 60-minute intervals
        # This matches R ricu's 1-hour interval
        step_minutes = 60.0  # 1 hour = 60 minutes
        
        # Vectorized expand: floor start/stop to hour boundaries, compute repeat counts
        starts = pd.to_numeric(result[index_col], errors='coerce')
        stops = pd.to_numeric(result[stop_col], errors='coerce')
        valid = starts.notna() & stops.notna() & (stops > starts)
        result_valid = result.loc[valid].copy()
        starts_v = starts.loc[valid].values
        stops_v = stops.loc[valid].values
        
        start_hours = np.floor(starts_v / step_minutes) * step_minutes
        stop_hours = np.floor(stops_v / step_minutes) * step_minutes
        # Number of hourly time points per row (inclusive)
        n_points = ((stop_hours - start_hours) / step_minutes).astype(int) + 1
        n_points = np.maximum(n_points, 1)
        
        if n_points.sum() > 0:
            # Repeat each row n_points times
            row_indices = np.repeat(np.arange(len(result_valid)), n_points)
            expanded = result_valid.iloc[row_indices].reset_index(drop=True)
            
            # Generate time points for each expanded row
            time_values = np.concatenate([
                np.arange(sh, sh + n * step_minutes, step_minutes)
                for sh, n in zip(start_hours, n_points)
            ])
            expanded[index_col] = time_values
            
            # Drop stop_col from result (not needed after expand)
            if stop_col in expanded.columns:
                expanded = expanded.drop(columns=[stop_col])
            result = expanded
            
            # 🔧 FIX 2024-12-01: Do NOT aggregate in expand()
            # R ricu's expand() does NOT aggregate by default (aggregate=FALSE)
            # Aggregation should be done at a higher level based on the concept's
            # aggregate parameter (e.g., 'max' for dopa60 in sofa_cardio)
            # 
            # Previous code used mean aggregation here, which caused:
            # - dopa60 at time=1 = mean(5.33, 4.44, 3.56) = 4.44 (incorrect)
            # - sofa_cardio score = 2 (because 4.44 <= 5)
            #
            # Correct behavior:
            # - dopa60 at time=1 should keep all values (5.33, 4.44, 3.56)
            # - External aggregation with max gives 5.33
            # - sofa_cardio score = 3 (because 5.33 > 5)
            #
            # Note: This may result in duplicate rows at the same time point,
            # which is expected and will be handled by external aggregation.
        else:
            result = pd.DataFrame(columns=[c for c in result.columns if c != stop_col])
    
    return result

def aumc_rate_mass(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: str,
    unit_col: Optional[str],
    rate_unit_col: Optional[str],
    index_col: Optional[str],
    stop_col: Optional[str],
    target_unit: str,
) -> pd.DataFrame:
    """AUMC non-kg mass-rate callback (analogue of aumc_rate_kg without ÷weight).

    Implements R ricu-style pipeline for AUMC ``drugitems`` rows that express a
    mass-rate (mg/hour, mcg/hour) rather than a kg-normalized rate:

      1. Normalize mass column (`mg`/`µg`/`g` → target mass unit).
      2. Normalize rate-time column (`uur` → hour, `dag` → day, `min` → min).
      3. Pure unit filter: only rows whose final `unit/rate_unit` exactly
         matches ``target_unit`` survive. Everything else becomes NaN and is
         dropped before interval expansion. Crucially:
           * rows where ``doserateperkg == 1`` are excluded (this would need
             kg re-normalization, which this callback does not perform);
           * rows whose ``rate_unit`` is missing are excluded (raw/bolus dose).
      4. Interval-expand from ``index_col`` to ``stop_col`` at a 1-hour
         resolution — identical to ``aumc_rate_kg``'s expand step, guaranteeing
         output schema parity.

    Supported target units
    ----------------------
    ``mcg/hour``, ``mcg/min``, ``mg/hour``, ``mg/min``.

    Examples
    --------
    >>> # Fentanyl AUMC itemid 7219 — target mcg/hour
    >>> cb = aumc_rate_mass
    >>> # Dispatched via concept-dict callback
    ...  "aumc_rate_mass(target_unit = \"mcg/hour\")"
    """
    target = target_unit.lower().replace("hr", "hour").strip()
    if target not in {"mcg/hour", "mcg/min", "mg/hour", "mg/min"}:
        raise ValueError(
            f"aumc_rate_mass: unsupported target_unit {target_unit!r}"
        )
    target_mass, target_time = target.split("/")

    if frame.empty:
        return frame

    df = frame.copy()

    if val_col not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + [concept_name])

    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    if df.empty:
        return df

    # Drop rows missing unit / rate_unit (raw bolus dose cannot be converted to rate)
    na_cols = []
    if unit_col and unit_col in df.columns:
        na_cols.append(unit_col)
    if rate_unit_col and rate_unit_col in df.columns:
        na_cols.append(rate_unit_col)
    if na_cols:
        df = df.dropna(subset=na_cols, how="any")
        if df.empty:
            return df

    # Drop rows whose doserateperkg is true (would need kg normalization)
    if "doserateperkg" in df.columns:
        per_kg_mask = pd.to_numeric(df["doserateperkg"], errors="coerce").fillna(0).astype(bool)
        if per_kg_mask.any():
            df = df.loc[~per_kg_mask].copy()
            if df.empty:
                return df

    # Normalize mass units to mcg internally (reuse existing helper)
    _aumc_normalize_mass_units(df, unit_col, val_col)
    rate_unit_col = _aumc_normalize_rate_units(df, rate_unit_col, val_col) or rate_unit_col

    # At this point mass is in mcg and time is in 'min'. Convert to target.
    # Convert mcg → target mass
    if target_mass == "mg":
        df[val_col] = df[val_col] / 1000.0
    # target mcg → no-op

    # Convert /min → target time
    if target_time == "hour":
        df[val_col] = df[val_col] * 60.0

    # Set final unit string for downstream bookkeeping
    if unit_col and unit_col in df.columns:
        df[unit_col] = f"{target_mass}/{target_time}"

    df[concept_name] = df[val_col]

    id_cols = _aumc_get_id_columns(df)
    result_cols = list(dict.fromkeys(id_cols))
    if index_col:
        if index_col not in df.columns:
            df[index_col] = pd.NaT
        result_cols.append(index_col)
    result_cols.append(concept_name)
    if unit_col and unit_col in df.columns:
        result_cols.append(unit_col)

    result = df[result_cols].dropna(subset=[concept_name])

    # Interval expand (identical to aumc_rate_kg)
    if stop_col and stop_col in df.columns and index_col and index_col in df.columns:
        result[stop_col] = df.loc[result.index, stop_col]

        step_minutes = 60.0
        starts = pd.to_numeric(result[index_col], errors="coerce")
        stops = pd.to_numeric(result[stop_col], errors="coerce")
        valid = starts.notna() & stops.notna() & (stops > starts)
        result_valid = result.loc[valid].copy()
        starts_v = starts.loc[valid].values
        stops_v = stops.loc[valid].values

        start_hours = np.floor(starts_v / step_minutes) * step_minutes
        stop_hours = np.floor(stops_v / step_minutes) * step_minutes
        n_points = ((stop_hours - start_hours) / step_minutes).astype(int) + 1
        n_points = np.maximum(n_points, 1)

        if n_points.sum() > 0:
            row_indices = np.repeat(np.arange(len(result_valid)), n_points)
            expanded = result_valid.iloc[row_indices].reset_index(drop=True)
            time_values = np.concatenate([
                np.arange(sh, sh + n * step_minutes, step_minutes)
                for sh, n in zip(start_hours, n_points)
            ])
            expanded[index_col] = time_values
            if stop_col in expanded.columns:
                expanded = expanded.drop(columns=[stop_col])
            result = expanded
        else:
            result = pd.DataFrame(columns=[c for c in result.columns if c != stop_col])

    return result


def sic_rate_mass(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: str,
    index_col: Optional[str],
    stop_col: Optional[str],
    target_unit: str,
) -> pd.DataFrame:
    """SIC non-kg mass-rate callback for sedatives/analgesics.

    Unlike vasopressors (where ``AmountPerMinute`` is a genuine g/min rate),
    SIC sedatives store the **total bolus dose** in both ``Amount`` and
    ``AmountPerMinute`` (see audit report 2026-05-13). The actual infusion
    rate must be computed as:

        rate = Amount / duration_seconds × conversion_factor

    where ``duration_seconds = OffsetDrugEnd − Offset``.

    SIC ``Amount`` is in **grams** (confirmed by d_references unit='g' for
    DrugID 1480/1495/1499/1549). Conversion to target:

      - mcg/hour: Amount_g × 1e6 / duration_seconds × 3600
      - mg/hour:  Amount_g × 1e3 / duration_seconds × 3600
      - mcg/min:  Amount_g × 1e6 / duration_seconds × 60
      - mg/min:   Amount_g × 1e3 / duration_seconds × 60

    Rows with zero or negative duration are dropped (single-push boluses
    with no meaningful rate).

    After rate computation, intervals are expanded to an hourly grid
    (matching ``sic_rate_kg``'s expand logic).

    Parameters
    ----------
    target_unit : str
        One of ``mcg/hour``, ``mcg/min``, ``mg/hour``, ``mg/min``.
    """
    target = target_unit.lower().replace("hr", "hour").strip()
    if target not in {"mcg/hour", "mcg/min", "mg/hour", "mg/min"}:
        raise ValueError(f"sic_rate_mass: unsupported target_unit {target_unit!r}")
    target_mass, target_time = target.split("/")

    # Mass multiplier: grams → target mass
    mass_mult = {"mcg": 1e6, "mg": 1e3}[target_mass]
    # Time multiplier: per-second → per-target_time
    time_mult = {"hour": 3600.0, "min": 60.0}[target_time]

    if frame.empty:
        return frame

    df = frame.copy()

    # Resolve val_col (may have been renamed to concept_name)
    actual_val = val_col if val_col in df.columns else (
        concept_name if concept_name in df.columns else None
    )
    if actual_val is None:
        return pd.DataFrame(columns=list(df.columns) + [concept_name])

    # Use Amount column (total dose in grams) — NOT AmountPerMinute
    # because for sedatives Amount == AmountPerMinute (audit finding).
    # If val_col was set to AmountPerMinute in the dict, that's fine —
    # the values are the same. We just need the duration to compute rate.
    df[actual_val] = pd.to_numeric(df[actual_val], errors="coerce")
    df = df.dropna(subset=[actual_val])
    if df.empty:
        return df

    # Compute duration in seconds
    if not index_col:
        for cand in ["Offset", "OffsetDrugStart", "start", "charttime"]:
            if cand in df.columns:
                index_col = cand
                break
    if not stop_col:
        for cand in ["OffsetDrugEnd", "stop", "endtime"]:
            if cand in df.columns:
                stop_col = cand
                break

    if not (index_col and stop_col and index_col in df.columns and stop_col in df.columns):
        return pd.DataFrame(columns=list(df.columns) + [concept_name])

    starts = pd.to_numeric(df[index_col], errors="coerce")
    stops = pd.to_numeric(df[stop_col], errors="coerce")
    duration_sec = stops - starts

    # Drop zero/negative duration (bolus pushes with no meaningful rate)
    valid = duration_sec > 0
    df = df.loc[valid].copy()
    duration_sec = duration_sec.loc[valid]
    if df.empty:
        return df

    # Compute rate: Amount_g × mass_mult / duration_sec × time_mult
    df[concept_name] = df[actual_val] * mass_mult / duration_sec * time_mult

    # Drop implausible (NaN, inf, negative)
    df = df[df[concept_name].notna() & np.isfinite(df[concept_name]) & (df[concept_name] > 0)]
    if df.empty:
        return df

    # Expand intervals to hourly grid (same logic as sic_rate_kg in concept.py)
    _PATIENT_ID_COLS = ["CaseID", "stay_id", "icustay_id", "patientunitstayid",
                        "admissionid", "patientid"]
    id_cols = [c for c in _PATIENT_ID_COLS if c in df.columns]
    keep_cols = id_cols + [concept_name]

    expanded_rows = []
    for _, row in df.iterrows():
        start_val = pd.to_numeric(row.get(index_col), errors="coerce")
        stop_val = pd.to_numeric(row.get(stop_col), errors="coerce")
        if pd.isna(start_val) or pd.isna(stop_val) or stop_val <= start_val:
            continue
        start_hour = int(start_val // 3600)
        stop_hour = int(stop_val // 3600)
        for t in range(start_hour, stop_hour + 1):
            new_row = {index_col: t}
            for c in keep_cols:
                if c in row.index:
                    new_row[c] = row[c]
            expanded_rows.append(new_row)

    if expanded_rows:
        result = pd.DataFrame(expanded_rows)
        if concept_name in result.columns:
            result[concept_name] = pd.to_numeric(result[concept_name], errors="coerce")
        return result
    else:
        return pd.DataFrame(columns=id_cols + [index_col, concept_name])


def aumc_rate_units_callback(mcg_to_units: float) -> Callable:
    """
    AUMC rate units callback - converts dose units and expands intervals.
    
    This callback matches R ricu's aumc_rate_units function (callback-itm.R lines 580-608):
    1. Converts µg → mcg, mg → mcg → units (using mcg_to_units factor)
    2. Converts rate units: dag → min (/1440), uur → min (/60)
    3. Handles bolus doses (NA rate_uom) by dividing by interval (60 min)
    4. Expands intervals from start to stop time
    
    R ricu code:
        to_units <- convert_unit(...)  # µg→mcg, mg→mcg, mcg→units
        to_min <- convert_unit(...)    # dag→uur (/24), uur→min (/60)
        x[is.na(rate_uom), ...] <- sum(val) * frac, by = meta_vars  # bolus handling
        x <- to_units(to_min(x, val_var, rate_uom), val_var, unit_var)
        expand(x, index_var, stop_var, ...)  # interval expansion
    
    Args:
        mcg_to_units: Conversion factor from mcg to units (e.g., 0.53 for vasopressin)
    """
    from .ts_utils import expand
    
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

        # Find time columns
        index_col = next((col for col in ['start', 'charttime', 'time'] if col in df.columns), None)
        
        # Set concept value
        df[concept_name] = df[val_col]

        id_cols = _aumc_get_id_columns(df)
        
        # 🔧 CRITICAL FIX: Expand intervals from start to stop (R ricu expand)
        # R ricu calls: expand(x, index_var(x), stop_var, keep_vars = ...)
        # This creates hourly rows from start to stop time
        if stop_col and stop_col in df.columns and index_col:
            # Prepare for expansion
            keep_vars = [concept_name]
            if unit_col and unit_col in df.columns:
                keep_vars.append(unit_col)
            
            # AUMC times are in minutes at this point (converted from ms in datasource.py)
            # Convert to hours for expand, but DON'T modify the original df yet
            # because _align_time_to_admission will also convert to hours
            # 
            # Actually, we need to convert to hours for expand() to work correctly
            # with step_size=1 hour. But then we need to return the data in minutes
            # so that _align_time_to_admission can convert it properly.
            #
            # Solution: Convert to hours for expand, which will create hourly rows,
            # then the result is already in hours, so _align_time_to_admission
            # should NOT divide by 60 again.
            #
            # Wait, that's wrong. Let me trace the flow:
            # 1. datasource.py: ms -> minutes (floor)
            # 2. aumc_rate_units_callback: minutes -> (expand with step=1h) -> hours
            # 3. _align_time_to_admission: minutes -> hours (/ 60)
            #
            # The bug is that expand() outputs times in hours, but 
            # _align_time_to_admission expects minutes and divides by 60 again.
            #
            # Fix: expand() should output times in minutes (same as input),
            # and _align_time_to_admission will convert to hours.
            
            if pd.api.types.is_numeric_dtype(df[index_col]):
                # Save original times in minutes
                start_min = df[index_col].copy()
                stop_min = df[stop_col].copy()
                
                # Convert to hours for expand (step_size is in hours)
                df[index_col] = df[index_col] / 60.0
                df[stop_col] = df[stop_col] / 60.0
                
                # 🔧 FIX: R ricu calls change_interval(re_time) BEFORE callback,
                # which floors start times to hour boundaries. This changes expand
                # behavior: seq(floor(start), stop, 1h) generates more rows when
                # an interval crosses an hour boundary. Stop is NOT floored.
                df[index_col] = np.floor(df[index_col])
                
                try:
                    df = expand(
                        df,
                        start_var=index_col,
                        end_var=stop_col,
                        step_size=pd.Timedelta(hours=1),
                        id_cols=id_cols,
                        keep_vars=keep_vars,
                    )
                    # After expand, times are in hours (integer hours)
                    # Convert back to minutes for _align_time_to_admission
                    if index_col in df.columns:
                        df[index_col] = df[index_col] * 60.0
                except Exception:
                    # If expand fails, restore original times and continue
                    df[index_col] = start_min
                    df[stop_col] = stop_min
            else:
                try:
                    df = expand(
                        df,
                        start_var=index_col,
                        end_var=stop_col,
                        step_size=pd.Timedelta(hours=1),
                        id_cols=id_cols,
                        keep_vars=keep_vars,
                    )
                except Exception:
                    pass
                pass

        result_cols = list(dict.fromkeys(id_cols))
        if index_col and index_col in df.columns:
            result_cols.append(index_col)
        result_cols.append(concept_name)
        if unit_col and unit_col in df.columns:
            result_cols.append(unit_col)
        if rate_unit_col and rate_unit_col in df.columns:
            result_cols.append(rate_unit_col)
        
        # Filter to only existing columns
        result_cols = [c for c in result_cols if c in df.columns]

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
    """
    Calculate duration for AUMC database items.
    
    NOTE: AUMC times are preprocessed in datasource.py and converted from 
    milliseconds to INTEGER MINUTES (floor(ms / 60000)) to match R ricu's as.integer().
    
    R ricu's calc_dur behavior:
    1. Times are first processed by re_time which floors to interval (1 hour)
    2. Then calc_dur computes: duration = max(stop_floor_hours) - min(start_floor_hours)
    
    So: duration = floor(max_stop_min/60) - floor(min_start_min/60)
    
    IMPORTANT: This function returns times in MINUTES to allow _align_time_to_admission
    to perform the final conversion to hours. Only the duration value is in hours.
    
    Args:
        frame: Input dataframe with AUMC data (times in INTEGER MINUTES)
        val_col: Name of the value column (will be replaced with duration)
        stop_var: Column name containing stop timestamps in MINUTES
        grp_var: Column name for grouping (e.g., 'orderid')
        index_var: Column name containing start timestamps in MINUTES
        concept_name: Name of the concept being calculated
        
    Returns:
        DataFrame with:
        - duration column (concept_name) in HOURS (integer)
        - start column (index_var) in MINUTES (to be converted by _align_time_to_admission)
    """
    if frame.empty or not stop_var or stop_var not in frame.columns:
        return frame

    df = frame.copy()

    # Find start column
    start_col = index_var if index_var and index_var in df.columns else None
    if not start_col:
        start_col = next((col for col in ['start', 'charttime', 'time'] if col in df.columns), None)
    if not start_col:
        return df

    # Get patient ID columns
    id_cols = _aumc_get_id_columns(df)
    
    # Prepare grouping columns
    group_cols = list(id_cols)
    if grp_var and grp_var in df.columns:
        if grp_var not in group_cols:
            group_cols.append(grp_var)
    
    # Times are in INTEGER MINUTES (converted in datasource.py)
    df[start_col] = pd.to_numeric(df[start_col], errors='coerce')
    df[stop_var] = pd.to_numeric(df[stop_var], errors='coerce')
    
    # Group by patient (and orderid if available) and aggregate start/stop
    grouped = df.groupby(group_cols, as_index=False).agg({
        start_col: 'min',  # earliest start time (minutes)
        stop_var: 'max',   # latest stop time (minutes)
    })
    
    # R ricu uses floor(hours) for both start and stop before computing duration
    # duration = floor(max_stop/60) - floor(min_start/60)
    # 🚀 Vectorized: avoid per-element lambda
    start_hours_floor = np.floor(grouped[start_col].values / 60.0)
    stop_hours_floor = np.floor(grouped[stop_var].values / 60.0)
    duration_hours = stop_hours_floor - start_hours_floor
    
    # Create a clean result with the duration in HOURS
    grouped[concept_name] = duration_hours.astype(float)
    
    # IMPORTANT: Keep start time in MINUTES (not hours!)
    # _align_time_to_admission will convert minutes to hours later
    # This prevents double conversion: aumc_dur converts to hours, then _align_time_to_admission divides by 60 again
    # Start time stays as grouped[start_col] which is already in minutes
    
    # Keep only necessary columns for the result
    result_cols = group_cols + [concept_name]
    # Also keep start_col if it's the index column (e.g., 'start')
    if start_col not in result_cols:
        result_cols.append(start_col)
    
    result = grouped[result_cols]
    
    return result


def hirid_rate_kg(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: str,
    unit_col: Optional[str],
    grp_var: Optional[str],
    index_col: Optional[str],
    default_weight: float = 70.0,
    interval_minutes: float = 60.0,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    HiRID rate per kg callback - converts dose to mcg/kg/min.
    
    Implements R ricu's hirid_rate_kg:
    1. Convert mg to µg (multiply by 1000)
    2. Filter to only µg unit records
    3. Group by (patientid, time, infusionid) and sum doses
    4. Get patient weight
    5. Calculate rate = dose_per_interval / (interval_minutes * weight)
    
    Args:
        interval_minutes: The concept's interval in minutes. Default is 60 (1 hour).
            R ricu uses frac = 1 / interval(x), where interval(x) is the concept's
            interval attribute. For dobu_rate (no interval defined), default 60min
            is used. For dobu60 (interval="00:01:00"), 1min is used.
            
            This affects the rate calculation:
            - interval=60min: groups data by hour, sums doses, rate = sum/60/weight
            - interval=1min: each point is independent, rate = dose/1/weight
              (this results in 60x higher values, matching R ricu's behavior)
    
    Then expand intervals to hourly time points.
    """
    # Create empty result with concept_name column
    empty_result = pd.DataFrame(columns=list(frame.columns) + [concept_name])
    
    if frame.empty:
        return empty_result
    
    df = frame.copy()
    
    # Handle case where val_col was renamed to concept_name before callback
    # This happens when the frame is renamed (givendose -> norepi_rate) before callback
    actual_val_col = val_col
    if val_col not in df.columns:
        if concept_name in df.columns:
            actual_val_col = concept_name
        else:
            return empty_result
    
    df[actual_val_col] = pd.to_numeric(df[actual_val_col], errors='coerce')
    df = df.dropna(subset=[actual_val_col])
    if df.empty:
        return empty_result
    
    # Step 1: Convert mg to µg
    if unit_col and unit_col in df.columns:
        mg_mask = df[unit_col].astype(str).str.lower().str.strip() == 'mg'
        if mg_mask.any():
            df.loc[mg_mask, actual_val_col] = df.loc[mg_mask, actual_val_col] * 1000
            df.loc[mg_mask, unit_col] = 'µg'
    
    # Step 2: Filter to only µg unit records
    if unit_col and unit_col in df.columns:
        unit_series = df[unit_col].astype(str).str.lower().str.strip()
        # Accept µg, ug, mcg variations
        ug_mask = unit_series.isin(['µg', 'ug', 'mcg', 'μg'])
        df = df[ug_mask]
    
    if df.empty:
        return empty_result
    
    # Identify patient ID column
    id_col = None
    for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid']:
        if cand in df.columns:
            id_col = cand
            break
    
    if id_col is None:
        return df
    
    # Step 3: Get patient weight
    if 'weight' not in df.columns:
        df['weight'] = default_weight
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(default_weight)
    
    # Detect time column if not specified
    if not index_col:
        for cand in ['datetime', 'givenat', 'charttime', 'time']:
            if cand in df.columns:
                index_col = cand
                break
    
    if index_col and index_col in df.columns:
        # Floor time based on interval_minutes to match R ricu's change_interval behavior
        time_series = df[index_col]
        if pd.api.types.is_numeric_dtype(time_series):
            if interval_minutes >= 60:
                df['_time_bin'] = np.floor(time_series).astype(int)
            else:
                time_in_minutes = time_series * 60
                df['_time_bin'] = np.floor(time_in_minutes).astype(int) / 60
        else:
            df['_time_bin'] = time_series
    else:
        df['_time_bin'] = 0
    
    # Step 4: Group by (patientid, time_bin, infusionid) and sum doses
    group_cols = [id_col, '_time_bin']
    if grp_var and grp_var in df.columns:
        group_cols.append(grp_var)
    
    # Aggregate weight (take first value per patient)
    weight_map = df.groupby(id_col)['weight'].first().to_dict()
    
    grouped = df.groupby(group_cols, as_index=False).agg({
        actual_val_col: 'sum',
    })
    
    # Map weight back
    grouped['weight'] = grouped[id_col].map(weight_map).fillna(default_weight)
    
    # Step 5: Calculate rate = dose / interval_minutes / weight
    grouped[concept_name] = grouped[actual_val_col] / interval_minutes / grouped['weight']
    
    # Rename _time_bin to index_col for consistency
    grouped = grouped.rename(columns={'_time_bin': index_col if index_col else 'datetime'})
    
    # Set unit
    if unit_col:
        grouped[unit_col] = 'mcg/kg/min'
    
    # Keep only necessary columns
    result_cols = [id_col, index_col if index_col else 'datetime', concept_name]
    if grp_var and grp_var in grouped.columns:
        result_cols.append(grp_var)
    if unit_col:
        result_cols.append(unit_col)
    
    # Filter to existing columns
    result_cols = [c for c in result_cols if c in grouped.columns]
    result = grouped[result_cols].dropna(subset=[concept_name])
    
    # Apply filter_bounds BEFORE expand_intervals to remove outlier rates per
    # infusionid that would otherwise distort the cross-infusion median.
    # R ricu applies filter_bounds after change_interval (which only re-discretizes,
    # NOT aggregates) and before aggregate(median). Since easyicu's expand_intervals 
    # already aggregates overlapping infusions with median, filter_bounds must run here.
    if value_min is not None:
        result = result[result[concept_name] >= value_min]
    if value_max is not None:
        result = result[result[concept_name] <= value_max]
    
    # 🔧 R ricu's hirid_rate_kg calls expand_intervals (callback-itm.R:523) which
    # LOCF-expands per-infusion rates at the ts_tbl interval (1 min for HiRID).
    # However, EasyICU's expand_intervals only supports hourly step size.
    # The per-infusion LOCF + MEDIAN collapse is instead handled inside
    # _callback_vaso60 which has full control over the minute-level expansion.
    # For standalone norepi_rate (not via vaso60), the expand_intervals at hourly
    # resolution + median aggregation is close enough.
    
    return result


def hirid_rate(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: str,
    unit_col: Optional[str],
    grp_var: Optional[str],
    index_col: Optional[str],
) -> pd.DataFrame:
    """
    HiRID rate callback - converts dose to rate per minute (no weight normalization).
    
    Implements R ricu's hirid_rate:
    1. Get the most frequent unit in the data
    2. Filter to only records with that unit
    3. Group by (patientid, time, infusionid) and sum doses
    4. Calculate rate = dose_per_interval / interval_minutes
    5. Append /min to the unit
    
    This differs from hirid_rate_kg in that:
    - No weight normalization
    - Uses the most common unit from data, not forcing µg
    - Output unit is "{original_unit}/min"
    
    Used for drugs like vasopressin (adh_rate) that don't need weight-based dosing.
    
    HiRID interval is 1 hour = 60 minutes, so:
    rate = givendose / 60
    """
    empty_result = pd.DataFrame(columns=list(frame.columns) + [concept_name])
    
    if frame.empty:
        return empty_result
    
    df = frame.copy()
    
    # Handle case where val_col was renamed to concept_name before callback
    actual_val_col = val_col
    if val_col not in df.columns:
        if concept_name in df.columns:
            actual_val_col = concept_name
        else:
            return empty_result
    
    df[actual_val_col] = pd.to_numeric(df[actual_val_col], errors='coerce')
    df = df.dropna(subset=[actual_val_col])
    if df.empty:
        return empty_result
    
    # Step 1: Get the most frequent unit
    target_unit = None
    if unit_col and unit_col in df.columns:
        unit_counts = df[unit_col].value_counts()
        if len(unit_counts) > 0:
            target_unit = unit_counts.index[0]
    
    # Step 2: Filter to only that unit
    if target_unit is not None and unit_col in df.columns:
        old_len = len(df)
        df = df[df[unit_col] == target_unit]
        if len(df) < old_len:
            pass  # Lost some rows due to unexpected units
    
    if df.empty:
        return empty_result
    
    # Identify patient ID column
    id_col = None
    for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid']:
        if cand in df.columns:
            id_col = cand
            break
    
    if id_col is None:
        return df
    
    # Detect time column if not specified
    if not index_col:
        for cand in ['datetime', 'givenat', 'charttime', 'time']:
            if cand in df.columns:
                index_col = cand
                break
    
    if index_col and index_col in df.columns:
        # Floor time to hours
        time_series = df[index_col]
        if pd.api.types.is_numeric_dtype(time_series):
            df['_hour'] = np.floor(time_series).astype(int)
        else:
            df['_hour'] = time_series
    else:
        df['_hour'] = 0
    
    # Step 3: Group by (patientid, hour, infusionid) and sum doses
    group_cols = [id_col, '_hour']
    if grp_var and grp_var in df.columns:
        group_cols.append(grp_var)
    
    grouped = df.groupby(group_cols, as_index=False).agg({
        actual_val_col: 'sum',  # Sum doses within each hour
    })
    
    # Step 4: Calculate rate = dose / interval_minutes
    # HiRID interval is 1 hour = 60 minutes
    interval_minutes = 60.0
    grouped[concept_name] = grouped[actual_val_col] / interval_minutes
    
    # Rename _hour to datetime for consistency
    grouped = grouped.rename(columns={'_hour': index_col if index_col else 'datetime'})
    
    # Step 5: Set unit to "{original_unit}/min"
    output_unit = f"{target_unit}/min" if target_unit else "units/min"
    if unit_col:
        grouped[unit_col] = output_unit
    
    # Keep only necessary columns
    result_cols = [id_col, index_col if index_col else 'datetime', concept_name]
    if grp_var and grp_var in grouped.columns:
        result_cols.append(grp_var)
    if unit_col:
        result_cols.append(unit_col)
    
    # Filter to existing columns
    result_cols = [c for c in result_cols if c in grouped.columns]
    result = grouped[result_cols].dropna(subset=[concept_name])
    
    return result


def hirid_rate_mass(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: str,
    unit_col: Optional[str],
    grp_var: Optional[str],
    index_col: Optional[str],
    target_unit: str,
) -> pd.DataFrame:
    """HiRID mass-rate callback (analogue of ``hirid_rate_kg`` without ÷weight).

    HiRID pharma stores each administration event's dose in a native mass unit
    (``doseunit``: µg for fentanyl, mg for midazolam, ...). Unlike the vasopressin
    ``hirid_rate`` variant, this callback locks to a caller-specified target
    unit rather than inferring from the most-frequent doseunit.

    Steps (mirroring ``hirid_rate_kg``):

    1. Filter to rows whose ``doseunit`` is compatible with ``target_unit`` —
       i.e. can be converted to the target mass via a unit factor. Rows with
       other or missing units are dropped.
    2. Group by (patientid, hour-floor, infusionid) and SUM the doses given
       within each hour.
    3. Divide by 60 (HiRID pharma events accumulate within an hourly grid;
       the summed dose across 1 hour becomes the per-minute rate).
    4. Convert to ``target_unit`` via mass + time scaling.
    5. Return ``(patientid, hour, concept_name, unit)`` columns.

    Parameters
    ----------
    target_unit : str
        One of ``mcg/hour``, ``mcg/min``, ``mg/hour``, ``mg/min``.

    Examples
    --------
    >>> # Fentanyl: doseunit µg, target mcg/hour
    >>> # dispatched via concept-dict callback "hirid_rate_mass(target_unit = \"mcg/hour\")"
    """
    target = target_unit.lower().replace("hr", "hour").strip()
    if target not in {"mcg/hour", "mcg/min", "mg/hour", "mg/min"}:
        raise ValueError(
            f"hirid_rate_mass: unsupported target_unit {target_unit!r}"
        )
    target_mass, target_time = target.split("/")

    empty = pd.DataFrame(columns=list(frame.columns) + [concept_name])
    if frame.empty:
        return empty

    df = frame.copy()

    # Handle val_col rename collision (same pattern as hirid_rate)
    actual_val_col = val_col
    if val_col not in df.columns:
        if concept_name in df.columns:
            actual_val_col = concept_name
        else:
            return empty

    df[actual_val_col] = pd.to_numeric(df[actual_val_col], errors="coerce")
    df = df.dropna(subset=[actual_val_col])
    if df.empty:
        return empty

    # ── Step 1: unit-compatibility filter ──
    # HiRID doseunit is one of: µg, mg, g (rare)
    if unit_col and unit_col in df.columns:
        unit_lower = df[unit_col].astype(str).str.strip().str.lower()
        # Map each row's unit to multiplier that converts its mass to target_mass
        #   µg → mcg: 1.0,  mg → mcg: 1000, g → mcg: 1e6
        #   µg → mg: 0.001, mg → mg: 1.0,   g → mg: 1000
        mass_mult = pd.Series(np.nan, index=df.index)
        is_mcg = unit_lower.isin({"µg", "μg", "ug", "mcg"})
        is_mg = unit_lower.eq("mg")
        is_g = unit_lower.eq("g")
        if target_mass == "mcg":
            mass_mult[is_mcg] = 1.0
            mass_mult[is_mg] = 1000.0
            mass_mult[is_g] = 1_000_000.0
        else:  # target_mass == "mg"
            mass_mult[is_mcg] = 0.001
            mass_mult[is_mg] = 1.0
            mass_mult[is_g] = 1000.0

        valid = mass_mult.notna()
        df = df.loc[valid].copy()
        mass_mult = mass_mult.loc[valid]
        if df.empty:
            return empty
        df[actual_val_col] = df[actual_val_col] * mass_mult
    # If no unit column, assume values are already in target_mass — risky but
    # consistent with how hirid_rate handles missing units.

    # ── Step 2: identify ID and time columns ──
    id_col = None
    for cand in ["patientid", "stay_id", "admissionid", "patientunitstayid"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        return df

    if not index_col:
        for cand in ["datetime", "givenat", "charttime", "time"]:
            if cand in df.columns:
                index_col = cand
                break

    if index_col and index_col in df.columns:
        time_series = df[index_col]
        if pd.api.types.is_numeric_dtype(time_series):
            df["_hour"] = np.floor(time_series).astype(int)
        else:
            df["_hour"] = time_series
    else:
        df["_hour"] = 0

    # ── Step 3: group-sum per (patient, hour, infusion) ──
    group_cols = [id_col, "_hour"]
    if grp_var and grp_var in df.columns:
        group_cols.append(grp_var)
    grouped = df.groupby(group_cols, as_index=False).agg({actual_val_col: "sum"})

    # ── Step 4: sum over hour → rate per target_time ──
    # An hour's total dose / 60 = per-minute rate; multiply if target_time = hour
    grouped[concept_name] = grouped[actual_val_col] / 60.0
    if target_time == "hour":
        grouped[concept_name] = grouped[concept_name] * 60.0  # back to per-hour

    grouped = grouped.rename(columns={"_hour": index_col if index_col else "datetime"})

    if unit_col:
        grouped[unit_col] = f"{target_mass}/{target_time}"

    result_cols = [id_col, index_col if index_col else "datetime", concept_name]
    if grp_var and grp_var in grouped.columns:
        result_cols.append(grp_var)
    if unit_col:
        result_cols.append(unit_col)
    result_cols = [c for c in result_cols if c in grouped.columns]
    result = grouped[result_cols].dropna(subset=[concept_name])
    return result


def hirid_urine(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: str,
    unit_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    HiRID urine callback - converts cumulative urine output to incremental values.
    
    Implements R ricu's hirid_urine:
    1. Group by patient ID
    2. Calculate diff of cumulative values
    3. Set negative diffs to 0 (cumulative can't decrease)
    4. Set unit to "mL"
    
    Args:
        frame: DataFrame with urine data
        concept_name: Name of the output column
        val_col: Name of the value column (cumulative urine)
        unit_col: Name of the unit column
        
    Returns:
        DataFrame with incremental urine values
    """
    if frame.empty:
        return frame
    
    df = frame.copy()
    
    # Handle case where val_col was renamed to concept_name before callback
    actual_val_col = val_col
    if val_col not in df.columns:
        if concept_name in df.columns:
            actual_val_col = concept_name
        else:
            return df
    
    # Identify patient ID column
    id_col = None
    for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid', 'subject_id']:
        if cand in df.columns:
            id_col = cand
            break
    
    if id_col is None:
        # No ID column, can't do diff per patient
        return df
    
    # Convert to numeric
    df[actual_val_col] = pd.to_numeric(df[actual_val_col], errors='coerce')
    
    # Sort by patient and time
    time_col = None
    for cand in ['datetime', 'charttime', 'time', 'givenat']:
        if cand in df.columns:
            time_col = cand
            break
    
    if time_col:
        df = df.sort_values([id_col, time_col])
    else:
        df = df.sort_values([id_col])
    
    # Calculate diff per patient
    def do_diff(x):
        """Calculate diff and set negative values to 0."""
        if len(x) == 0:
            return x
        result = x.diff()
        result.iloc[0] = x.iloc[0]  # First value is the first cumulative
        result = result.clip(lower=0)  # Negative diffs become 0
        return result
    
    df[actual_val_col] = df.groupby(id_col)[actual_val_col].transform(do_diff)
    
    # Set unit to mL
    if unit_col and unit_col in df.columns:
        df[unit_col] = 'mL'
    
    # Rename to concept_name if needed
    if actual_val_col != concept_name:
        df = df.rename(columns={actual_val_col: concept_name})
    
    return df


def hirid_vent(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: Optional[str] = None,
    index_col: Optional[str] = None,
    dur_var: str = 'dur_var',
    padding_hours: float = 4.0,
    max_gap_hours: float = 12.0,
    expand_to_hourly: bool = True,
) -> pd.DataFrame:
    """
    HiRID ventilation callback - converts time series to window table with durations.
    
    Implements R ricu's hirid_vent:
    1. Calculate time differences between consecutive records per patient
    2. Pad the last difference with padding (in data's time unit)
    3. Cap differences > max_gap to padding
    4. Store as dur_var for window table processing (in MINUTES per R ricu)
    5. Round datetime to integer hours (per R ricu)
    
    Args:
        frame: DataFrame with ventilation records
        concept_name: Name of the output column
        val_col: Name of the value column
        index_col: Name of the time/index column
        dur_var: Name of the duration column to create
        padding_hours: Duration to use for last record and capped gaps (in hours)
        max_gap_hours: Maximum allowed gap between records (in hours)
        expand_to_hourly: If True, expand windows to hourly time series
        
    Returns:
        DataFrame with window table format (dur_var in minutes, datetime as integer hours)
    """
    if frame.empty:
        return frame
    
    df = frame.copy()
    
    # Identify patient ID column
    id_col = None
    for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid', 'subject_id']:
        if cand in df.columns:
            id_col = cand
            break
    
    if id_col is None:
        # No ID column, add default duration (in minutes)
        df[dur_var] = padding_hours * 60
        return df
    
    # Detect time column
    actual_index_col = index_col
    if not actual_index_col:
        for cand in ['datetime', 'charttime', 'time', 'givenat']:
            if cand in df.columns:
                actual_index_col = cand
                break
    
    if not actual_index_col or actual_index_col not in df.columns:
        # No time column, add default duration (in minutes)
        df[dur_var] = padding_hours * 60
        return df
    
    # Sort by patient and time
    df = df.sort_values([id_col, actual_index_col])
    
    # R ricu uses the data's time unit (which is hours for HiRID after conversion)
    # padding and max_gap are in hours, dur_var output is in MINUTES
    padding_minutes = padding_hours * 60  # 4 hours = 240 minutes
    max_gap_minutes = max_gap_hours * 60  # 12 hours = 720 minutes
    
    # Calculate padded_capped_diff per patient
    # R: padded_diff <- function(x, final) c(diff(x), final)
    # R: padded_capped_diff <- function(x, final, max) { res <- padded_diff(x, final); res[res > max] <- final; res }
    def padded_capped_diff_minutes(time_series: pd.Series) -> pd.Series:
        """
        Calculate time diffs in MINUTES, pad last with padding_minutes, cap at max_gap_minutes.
        """
        if len(time_series) == 0:
            return pd.Series(dtype=float)
        
        if len(time_series) == 1:
            return pd.Series([padding_minutes], index=time_series.index)
        
        # Get values - should already be float hours (after HiRID time conversion)
        time_vals = time_series.values
        
        # Handle datetime/timedelta types (shouldn't happen after conversion, but just in case)
        if np.issubdtype(time_vals.dtype, np.datetime64):
            time_vals = (time_vals - time_vals[0]).astype('timedelta64[m]').astype(float)
        elif np.issubdtype(time_vals.dtype, np.timedelta64):
            time_vals = time_vals.astype('timedelta64[m]').astype(float)
        else:
            # Already numeric (hours) - convert to minutes
            time_vals = np.asarray(time_vals, dtype=float) * 60
        
        # Calculate diff (in minutes)
        diff_vals = np.diff(time_vals)
        
        # Pad with final value (in minutes)
        padded = np.append(diff_vals, padding_minutes)
        
        # Cap values > max_gap_minutes
        padded[padded > max_gap_minutes] = padding_minutes
        
        return pd.Series(padded, index=time_series.index)
    
    # Apply per patient
    df[dur_var] = df.groupby(id_col)[actual_index_col].transform(padded_capped_diff_minutes)
    
    # Convert dur_var from minutes to hours to match index unit
    # The index is already in hours; dur_var must be in the same unit for
    # correct endtime computation (start_hours + dur_hours) in _load_single_concept
    df[dur_var] = df[dur_var] / 60.0
    
    # Round datetime to integer hours (R ricu behavior)
    if actual_index_col in df.columns and not np.issubdtype(df[actual_index_col].dtype, np.datetime64):
        # If float hours, floor to integer
        df[actual_index_col] = np.floor(df[actual_index_col]).astype(int)
    
    # Expand to hourly time series (like R ricu's expand())
    if expand_to_hourly:
        df = _expand_hirid_vent_to_hourly(
            df, 
            id_col=id_col, 
            index_col=actual_index_col, 
            dur_col=dur_var,
            value_col=val_col,
            concept_name=concept_name,
        )
    
    return df


def _expand_hirid_vent_to_hourly(
    df: pd.DataFrame,
    id_col: str,
    index_col: str,
    dur_col: str,
    value_col: Optional[str],
    concept_name: str,
) -> pd.DataFrame:
    """
    Expand window table to hourly time series.
    
    Implements R ricu's expand() for win_tbl:
    - For each window (start_time, duration), generate rows for each hour
    - Each row has the patient ID, hour index, and value
    
    Args:
        df: DataFrame with windows (id, start_time, duration, value)
        id_col: Patient ID column
        index_col: Start time column
        dur_col: Duration column (in hours)
        value_col: Value column (optional)
        concept_name: Name for the output value column
        
    Returns:
        Expanded DataFrame with hourly rows
    """
    if df.empty:
        return df
    
    # 🚀 PERF 2026-05-11: Vectorized expansion via np.repeat — replaces an
    # iterrows loop that ran O(N × avg_duration_hours) at Python speed.
    # On HiRID 200-patient respiratory module this was the dominant cost
    # (vent_ind/mech_vent/adv_resp totalling ~500s; expected <5s after fix).

    # Ensure index is numeric (hours from ICU admission)
    time_col_data = df[index_col]
    if pd.api.types.is_datetime64_any_dtype(time_col_data):
        df = df.copy()
        df['_start_hours'] = df.groupby(id_col)[index_col].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600
        )
        start_col = '_start_hours'
    elif hasattr(time_col_data.dtype, 'kind') and time_col_data.dtype.kind == 'm':  # timedelta
        df = df.copy()
        df['_start_hours'] = time_col_data.dt.total_seconds() / 3600
        start_col = '_start_hours'
    else:
        start_col = index_col

    starts = pd.to_numeric(df[start_col], errors='coerce').to_numpy(dtype=float, copy=False)
    durs = pd.to_numeric(df[dur_col], errors='coerce').to_numpy(dtype=float, copy=False)
    valid = np.isfinite(starts) & np.isfinite(durs) & (durs > 0)

    if not valid.any():
        return pd.DataFrame(columns=[id_col, index_col, concept_name])

    df_v = df.iloc[valid]
    starts = starts[valid]
    durs = durs[valid]

    starts_floor = np.floor(starts).astype(np.int64)
    ends = starts + durs
    # Number of integer hours h such that floor(start) <= h < end
    # Equivalent to the original `while current_hour < end_hours` semantics.
    n_per_row = np.ceil(ends - starts_floor).astype(np.int64)
    # Guard against rounding edge cases (end == floor(start)) -> n=0
    n_per_row = np.maximum(n_per_row, 0)

    total = int(n_per_row.sum())
    if total == 0:
        return pd.DataFrame(columns=[id_col, index_col, concept_name])

    ids = df_v[id_col].to_numpy(copy=False)
    if value_col and value_col in df_v.columns:
        values = df_v[value_col].to_numpy(copy=False)
    else:
        values = np.full(len(df_v), True)

    expanded_ids = np.repeat(ids, n_per_row)
    expanded_values = np.repeat(values, n_per_row)
    # hour offsets: 0..n-1 per row, concatenated
    hour_offsets = np.concatenate([np.arange(n, dtype=np.int64) for n in n_per_row]) \
        if total > 0 else np.empty(0, dtype=np.int64)
    expanded_hours = np.repeat(starts_floor, n_per_row) + hour_offsets

    result = pd.DataFrame({
        id_col: expanded_ids,
        index_col: expanded_hours.astype(float),
        concept_name: expanded_values,
    })

    # Remove duplicates (same patient, same hour) and sort
    result = result.drop_duplicates(subset=[id_col, index_col], keep='first')
    result = result.sort_values([id_col, index_col]).reset_index(drop=True)

    return result


def hirid_duration(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    val_col: Optional[str] = None,
    index_col: Optional[str] = None,
    grp_var: Optional[str] = None,
) -> pd.DataFrame:
    """
    HiRID duration callback - calculates infusion durations.
    
    Implements R ricu's hirid_duration via calc_dur:
    For each (patient, infusion) group: duration = max(time) - min(time)
    
    Args:
        frame: DataFrame with infusion records
        concept_name: Name of the output column (duration)
        val_col: Name of the value column (not used, for compatibility)
        index_col: Name of the time/index column
        grp_var: Grouping variable (e.g., infusionid)
        
    Returns:
        DataFrame with duration per patient (and per group if grp_var specified)
    """
    # 🔧 FIX: Return empty DataFrame with all expected columns when input is empty
    # This prevents "Missing expected columns" errors downstream
    if frame.empty:
        # Build expected columns list
        empty_cols = ['patientid', 'givenat', concept_name]
        if grp_var:
            empty_cols.append(grp_var)
        return pd.DataFrame(columns=empty_cols)
    
    df = frame.copy()
    
    # Identify patient ID column
    id_col = None
    for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid', 'subject_id']:
        if cand in df.columns:
            id_col = cand
            break
    
    if id_col is None:
        # No ID column, return empty with expected structure
        empty_cols = ['patientid', 'givenat', concept_name]
        if grp_var:
            empty_cols.append(grp_var)
        return pd.DataFrame(columns=empty_cols)
    
    # Detect time column
    actual_index_col = index_col
    if not actual_index_col:
        for cand in ['datetime', 'charttime', 'time', 'givenat']:
            if cand in df.columns:
                actual_index_col = cand
                break
    
    if not actual_index_col or actual_index_col not in df.columns:
        # No time column, return empty with expected structure
        empty_cols = [id_col, 'givenat', concept_name]
        if grp_var:
            empty_cols.append(grp_var)
        return pd.DataFrame(columns=empty_cols)
    
    # Convert time to numeric if needed
    time_series = df[actual_index_col]
    if pd.api.types.is_datetime64_any_dtype(time_series):
        # Convert to hours (assuming times are in same units)
        df['_time_numeric'] = (time_series - time_series.min()).dt.total_seconds() / 3600.0
    elif hasattr(time_series.dtype, 'kind') and time_series.dtype.kind == 'm':
        # timedelta type
        df['_time_numeric'] = time_series.dt.total_seconds() / 3600.0
    else:
        df['_time_numeric'] = pd.to_numeric(time_series, errors='coerce')
    
    # Group by (patient_id, grp_var)
    group_cols = [id_col]
    if grp_var and grp_var in df.columns:
        group_cols.append(grp_var)
    
    # Calculate duration: floor(max(time)) - floor(min(time))
    # 🔧 FIX: R ricu's calc_dur operates on times that have already been floored
    # by dt_round_min in load_mihi. So the effective calculation is:
    # duration = floor(max_hours) - floor(min_hours)
    # NOT: duration = max_hours - min_hours
    # 
    # Example: Patient 2, infusion 289451
    #   min_time = 2.8333 (01:25 - 22:35 = 2:50 = 2.833h)
    #   max_time = 17.0833 (15:40 - 22:35 = 17:05 = 17.083h)
    #   Wrong: 17.0833 - 2.8333 = 14.25 → floor = 14
    #   Correct: floor(17.0833) - floor(2.8333) = 17 - 2 = 15
    result = df.groupby(group_cols, as_index=False).agg(
        _min_time=('_time_numeric', 'min'),
        _max_time=('_time_numeric', 'max'),
    )
    
    result[concept_name] = np.floor(result['_max_time']) - np.floor(result['_min_time'])
    
    # Add index_col as the start time (min_time)
    result[actual_index_col] = result['_min_time']
    
    # Drop temp columns
    result = result.drop(columns=['_min_time', '_max_time'])
    
    # Note: R ricu does NOT filter out duration=0 records
    # Keep all durations including 0 for consistency
    result = result[result[concept_name] >= 0]
    
    return result
