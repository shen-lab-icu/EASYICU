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
        
        # 展开时间窗口
        expanded_rows = []
        for idx, row in data.iterrows():
            start_hr = int(np.floor(start_time.loc[idx]))
            end_hr = int(np.floor(end_time.loc[idx]))
            
            # 展开每个小时
            for hr in range(start_hr, end_hr + 1):
                new_row = {c: row[c] for c in id_cols if c in row.index}
                new_row[index_col] = hr
                new_row[val_col] = row[val_col]
                expanded_rows.append(new_row)
        
        if expanded_rows:
            result = pd.DataFrame(expanded_rows)
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
        
        # 检测 ID 列
        id_col = None
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
                        # 确保时区一致
                        if intime.tzinfo is None:
                            intime = intime.tz_localize('UTC')
                        intime_map[patient_id] = intime
        
        for idx, row in data.iterrows():
            row_start = start_time.loc[idx]
            row_end = end_time.loc[idx]
            
            if pd.isna(row_start) or pd.isna(row_end):
                continue
            
            # 确保时区一致
            if row_start.tzinfo is None:
                row_start = row_start.tz_localize('UTC')
            if row_end.tzinfo is None:
                row_end = row_end.tz_localize('UTC')
            
            # 获取患者的 intime
            patient_id = row[id_col] if id_col and id_col in row.index else None
            intime = intime_map.get(patient_id) if patient_id is not None else None
            
            # 获取原始值
            amount = pd.to_numeric(row[val_col], errors='coerce')
            if pd.isna(amount):
                continue
            
            if intime is not None:
                # 🔧 正确的 R ricu 兼容逻辑：
                # 1. 计算相对时间（小时）
                start_rel_hours = (row_start - intime).total_seconds() / 3600
                end_rel_hours = (row_end - intime).total_seconds() / 3600
                
                # 2. floor 相对时间
                start_floor = int(np.floor(start_rel_hours))
                end_floor = int(np.floor(end_rel_hours))
                
                # 3. 如果 floor 后 end == start，设置 end = start + 1
                #    (R ricu: x <- x[get(end_var) - get(idx) == 0, c(end_var) := get(idx) + hr])
                if end_floor == start_floor:
                    end_floor = start_floor + 1
                
                # 4. 计算速率 = amount / (end_floor - start_floor) * 1hr
                #    注意：使用 floor 后的持续时间，不是原始持续时间！
                duration_hours = end_floor - start_floor
                rate = amount / duration_hours  # 已经是 units/hr
                
                # 5. 生成时间点 seq(start_floor, end_floor, 1)
                #    🔧 FIX: 直接返回相对小时数，而不是转换回绝对 datetime
                #    这样避免后续 change_interval 再次 floor 导致时间偏移
                for hr in range(start_floor, end_floor + 1):
                    new_row = {c: row[c] for c in id_cols if c in row.index}
                    # 直接存储相对小时数（float）
                    new_row[index_col] = float(hr)
                    new_row[val_col] = rate
                    expanded_rows.append(new_row)
            else:
                # Fallback: 使用绝对 datetime floor（不那么精确但可以工作）
                start_floor = row_start.floor('h')
                end_floor = row_end.floor('h')
                
                # 如果 floor 后相同，加 1 小时
                if start_floor == end_floor:
                    end_floor = start_floor + pd.Timedelta(hours=1)
                
                # 计算持续时间（小时）
                duration_hours = (end_floor - start_floor).total_seconds() / 3600
                rate = amount / duration_hours
                
                time_points = pd.date_range(start=start_floor, end=end_floor, freq='h')
                
                for t in time_points:
                    new_row = {c: row[c] for c in id_cols if c in row.index}
                    new_row[index_col] = t
                    new_row[val_col] = rate
                    expanded_rows.append(new_row)
        
        if expanded_rows:
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
            
            data = data.groupby(id_cols).apply(fill_group, include_groups=True).reset_index(drop=True)
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
            
            data = data.groupby(id_cols).apply(fill_group, include_groups=True).reset_index(drop=True)
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
            
            # For numeric offsets, convert to datetime temporarily for group_measurements
            # which expects datetime. We'll convert back later.
            base_time = pd.Timestamp('2000-01-01')
            frame['__temp_time'] = base_time + pd.to_timedelta(frame[index_var], unit='min')
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
        # For numeric offset (minutes), convert to hours; for datetime, result is already timedelta
        if is_offset:
            # eICU: offset in minutes, duration should be in hours
            result[val_col] = (result['max_time'] - result['min_time']) / 60.0
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
        - Old pyricu: 06:39 -> floor -> 06:00 -> relative 12.61h -> 12
        
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
        data = data.copy()
        
        # Check if time columns are already numeric (relative hours) or datetime
        min_is_numeric = pd.api.types.is_numeric_dtype(data[min_var])
        max_is_numeric = pd.api.types.is_numeric_dtype(data[max_var])
        
        if not min_is_numeric:
            # Try to convert to datetime if not already numeric
            data[min_var] = pd.to_datetime(data[min_var], errors='coerce')
        if not max_is_numeric:
            data[max_var] = pd.to_datetime(data[max_var], errors='coerce')
        
        # Drop rows where min_var or max_var is NaN/NaT
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
        
        # 🔧 FIX: R ricu's duration calculation uses floor(end_hours) - floor(start_hours)
        # This is because R ricu applies round_to (floor) to all time columns before calc_dur
        # 
        # R ricu flow:
        # 1. load_mihi: dt_round_min <- function(x, y) round_to(difftime(x, y, units = "mins"))
        #    This floors all time columns to integer minutes relative to origin
        # 2. change_interval: re_time floors minutes to hours
        # 3. calc_dur: computes max(end) - min(start) on already-floored hours
        #
        # So effectively: duration = floor(max_end_hours) - floor(min_start_hours)
        
        # Calculate duration: floor(max_end) - floor(min_start) in hours
        # After aggregation, max_var contains max(endtime), min_var contains min(starttime)
        import numpy as np
        
        # Check if times are already in numeric hours (converted earlier) or datetime
        if pd.api.types.is_numeric_dtype(result[min_var]) and pd.api.types.is_numeric_dtype(result[max_var]):
            # Times are already in relative hours - use floor(end_h) - floor(start_h)
            min_hours = result[min_var].astype(float)
            max_hours = result[max_var].astype(float)
            result[val_col] = np.floor(max_hours) - np.floor(min_hours)
        else:
            # Times are datetime - need to compute relative hours using intime
            # R ricu formula: floor(end_hours) - floor(start_hours) where hours are relative to intime
            
            # Find ID column
            id_col = None
            for col in id_cols if id_cols else []:
                if col in result.columns:
                    id_col = col
                    break
            
            if admission_times is not None and id_col is not None and 'intime' in admission_times.columns:
                # Merge intime into result
                intime_df = admission_times[[id_col, 'intime']].drop_duplicates()
                intime_df['intime'] = pd.to_datetime(intime_df['intime'], errors='coerce')
                result_with_intime = result.merge(intime_df, on=id_col, how='left')
                
                # Convert datetime to relative hours
                result[min_var] = pd.to_datetime(result[min_var], errors='coerce')
                result[max_var] = pd.to_datetime(result[max_var], errors='coerce')
                
                min_hours = (result_with_intime[min_var] - result_with_intime['intime']).dt.total_seconds() / 3600.0
                max_hours = (result_with_intime[max_var] - result_with_intime['intime']).dt.total_seconds() / 3600.0
                
                # Use floor(end_h) - floor(start_h) formula
                result[val_col] = np.floor(max_hours) - np.floor(min_hours)
            else:
                # Fallback: no intime available, use floor(duration)
                duration_td = result[max_var] - result[min_var]
                
                if pd.api.types.is_timedelta64_dtype(duration_td):
                    duration_hours = duration_td.dt.total_seconds() / 3600.0
                else:
                    # duration_td might be object dtype containing timedeltas
                    duration_hours = duration_td.apply(
                        lambda x: x.total_seconds() / 3600.0 if hasattr(x, 'total_seconds') else float(x)
                    )
                
                # Without intime, we can only floor the duration directly
                # This is less accurate than floor(end_h) - floor(start_h)
                result[val_col] = np.floor(duration_hours)
        
        # Rename min_var to index_var (for consistency with time column naming)
        # The index_var should be the start time (min_var)
        # Note: min_var already contains the aggregated min(starttime), which becomes the index_var
        if min_var != index_var and min_var in result.columns:
            result = result.rename(columns={min_var: index_var})
        
        # 🔧 FIX: R ricu calc_dur only returns: id_vars + index_var (start time) + val_var (duration)
        # It does NOT keep endtime/max_var column!
        # Keeping endtime causes _load_single_concept to mistakenly expand duration concepts
        # as if they were window concepts (win_tbl).
        keep_cols = list(id_cols) + [index_var, val_col]
        if unit_col and unit_col in result.columns:
            keep_cols.append(unit_col)
        # Remove duplicates while preserving order
        keep_cols = list(dict.fromkeys(keep_cols))
        # Only keep columns that actually exist
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
        # Assume numeric time is in HOURS (expand_intervals converts eICU minutes to hours first)
        overhang_hours = overhang.total_seconds() / 3600.0  # 1 hour
        max_len_hours = max_len.total_seconds() / 3600.0    # 6 hours
        interval_hours = interval.total_seconds() / 3600.0  # 1 hour
        
        # R ricu logic:
        # 1. diff = next_time - start (or overhang for last record)
        # 2. diff = trunc(diff, 0, max_len)
        # 3. diff = diff - interval
        # 4. endtime = start + diff
        
        if by_cols:
            def calc_end(group):
                group = group.copy()
                # Step 1: Calculate diff to next time (padded_diff)
                next_times = group[index_var].shift(-1)
                diff = next_times - group[index_var]
                # For last row, use overhang
                diff = diff.fillna(overhang_hours)
                
                # Step 2: Truncate to [0, max_len]
                diff = diff.clip(lower=0, upper=max_len_hours)
                
                # Step 3: Subtract interval
                diff = diff - interval_hours
                
                # Ensure diff is not negative
                diff = diff.clip(lower=0)
                
                # Step 4: endtime = start + diff
                group[end_var] = group[index_var] + diff
                return group
            
            data = data.groupby(by_cols, group_keys=False).apply(calc_end, include_groups=True)
        else:
            next_times = data[index_var].shift(-1)
            diff = next_times - data[index_var]
            diff = diff.fillna(overhang_hours)
            diff = diff.clip(lower=0, upper=max_len_hours)
            diff = diff - interval_hours
            diff = diff.clip(lower=0)
            data[end_var] = data[index_var] + diff
    else:
        # Original datetime logic
        if not pd.api.types.is_datetime64_any_dtype(data[index_var]):
            data[index_var] = pd.to_datetime(data[index_var])
        
        if by_cols:
            def calc_end(group):
                group = group.copy()
                next_times = group[index_var].shift(-1)
                diff = next_times - group[index_var]
                diff = diff.fillna(overhang)
                diff = diff.clip(lower=pd.Timedelta(0), upper=max_len)
                diff = diff - interval
                diff = diff.clip(lower=pd.Timedelta(0))
                group[end_var] = group[index_var] + diff
                return group
            
            data = data.groupby(by_cols, group_keys=False).apply(calc_end, include_groups=True)
        else:
            next_times = data[index_var].shift(-1)
            diff = next_times - data[index_var]
            diff = diff.fillna(overhang)
            diff = diff.clip(lower=pd.Timedelta(0), upper=max_len)
            diff = diff - interval
            diff = diff.clip(lower=pd.Timedelta(0))
            data[end_var] = data[index_var] + diff
    
    return data


def expand_intervals(
    data: pd.DataFrame,
    keep_vars: Optional[list] = None,
    grp_var: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Expand CareVue intervals into time series (R ricu expand_intervals).
    
    Creates intervals using create_intervals and then expands them.
    
    R ricu behavior:
    - create_intervals groups by (id_vars, grp_var) to create intervals per infusion
    - expand only keeps id_vars in output, NOT grp_var
    - When multiple infusions have overlapping times, expand produces duplicate rows
    - These duplicates are aggregated later by aggregate() using median for numeric values
    
    Args:
        data: Input DataFrame
        keep_vars: Variables to keep in expansion
        grp_var: Optional grouping variable (e.g., infusionid) - used for interval creation
                but NOT kept in output
        **kwargs: Additional arguments
        
    Returns:
        Expanded DataFrame with duplicates aggregated by median
    """
    from .ts_utils import expand
    import numpy as np
    
    # Infer ID columns - EXCLUDE grp_var from id_cols
    # R ricu's id_vars() returns only the patient-level ID, not infusion-level IDs
    id_cols = [col for col in data.columns if 'id' in col.lower()]
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
    # R ricu uses hours() for overhang/max_len but converts to minutes internally
    # We need to convert minute data to hours for proper expansion, then back to minutes
    is_eicu_minutes = index_var.lower() == 'infusionoffset' and pd.api.types.is_numeric_dtype(data[index_var])
    
    if is_eicu_minutes:
        # Convert minutes to hours for proper interval creation and expansion
        data = data.copy()
        data[index_var] = data[index_var] / 60.0  # Minutes to hours
    
    # Create intervals (overhang=1 hour, max_len=6 hours)
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
    expanded = expand(
        data,
        start_var=index_var,
        end_var='endtime',
        step_size=pd.Timedelta(hours=1),
        id_cols=id_cols,
        keep_vars=keep_vars
    )
    
    if is_eicu_minutes and len(expanded) > 0:
        # 🔧 CRITICAL: Round time to floor hour and deduplicate
        # R ricu uses re_time(floor) to round times to integer hours
        # Then keeps only unique hour values per patient
        expanded = expanded.copy()
        
        # Floor to integer hours
        expanded[index_var] = np.floor(expanded[index_var])
        
        # Deduplicate by patient and hour, keeping last value (LOCF style)
        dedup_cols = list(id_cols) + [index_var]
        expanded = expanded.drop_duplicates(subset=dedup_cols, keep='last')
        
        # Convert hours back to minutes for output (integer hours * 60)
        expanded[index_var] = (expanded[index_var] * 60).astype(int)
    
    # 🔧 CRITICAL FIX: Aggregate duplicate (patient, time) combinations
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
    
    data = data.groupby(id_cols, group_keys=False).apply(calc_duration, include_groups=True)
    
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
        
        data = data.groupby(group_cols, group_keys=False).apply(calc_rate, include_groups=True)
        
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
            def extract_unit(s):
                if pd.isna(s):
                    return None
                s = str(s)
                match = re.search(r'\(([^)]+)\)$', s)
                if match:
                    return match.group(1)
                if '/' in s or s.lower() in ['mg', 'mcg', 'ml', 'units']:
                    return s
                return None
            
            frame['unit_var'] = frame[sub_var].apply(extract_unit)
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
        
        # Apply unit conversions FIRST (following R ricu logic)
        # Then divide by weight for non-/kg/ units
        def convert_value(row):
            val = row[val_var]
            unit = row.get('unit_var', '')
            weight = row.get('_weight', 70.0)
            
            # Ensure weight is positive to prevent division by zero
            if pd.isna(weight) or weight <= 0:
                weight = 70.0
            
            if pd.isna(val) or not unit:
                return np.nan
            
            unit = str(unit).strip().lower()
            
            # Check for incompatible units first
            if unit.startswith('units/') or unit in ['unknown', 'ml', '']:
                return np.nan
            
            # Step 1: /hr -> /min (divide by 60)
            if '/hr' in unit:
                val = val / 60
                unit = unit.replace('/hr', '/min')
            
            # Step 2: mg/ -> mcg/ (multiply by 1000)
            if unit.startswith('mg/'):
                val = val * 1000
                unit = 'mcg' + unit[2:]
            
            # Step 3: ml/ -> mcg/ (multiply by ml_to_mcg)
            if unit.startswith('ml/'):
                val = val * ml_to_mcg
                unit = 'mcg' + unit[2:]
            
            # Step 4: nanograms/ -> mcg/ (divide by 1000)
            if unit.startswith('nanograms/'):
                val = val / 1000
                unit = 'mcg' + unit[9:]
            
            # Step 5: For non-/kg/ units, divide by weight
            if '/kg/' not in unit:
                val = val / weight
            
            return val
        
        frame[concept_name] = frame.apply(convert_value, axis=1)
        
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
        
        # Step 1: Convert to hours
        frame['_hour'] = (frame[time_col] // 60).astype(int)
        
        # Step 2: Aggregate by patient and hour (take max)
        hourly = frame.groupby([id_col, '_hour'], as_index=False).agg({
            concept_name: 'max'
        })
        hourly = hourly.sort_values([id_col, '_hour'])
        
        # Step 3: Create intervals using R ricu's create_intervals logic
        # R: endtime = padded_diff(hour, overhang=1)  # diff to next, or 1 for last
        # R: endtime = trunc(endtime, 0, max_len=6) - interval=1
        # R: endtime = hour + endtime
        overhang = 1  # hours
        max_len = 6   # hours  
        interval = 1  # hours (time step)
        
        def create_intervals_ricu(group):
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
        
        hourly = hourly.groupby(id_col, group_keys=False, sort=False).apply(
            create_intervals_ricu, include_groups=True
        )
        
        # Step 4: Expand each record using R's seq(start, end, step=1)
        # NOTE: Return time in MINUTES (hour * 60) so that _align_time_to_admission
        # can perform the standard minutes -> hours conversion for eICU data
        expanded_rows = []
        for _, row in hourly.iterrows():
            start_hour = int(row['_hour'])
            end_hour = int(row['_end_hour'])
            value = row[concept_name]
            patient_id = row[id_col]
            
            # R seq(start, end, 1) includes both endpoints
            # Return time in minutes (hour * 60) for consistency with eICU offset format
            for hour in range(start_hour, end_hour + 1):
                expanded_rows.append({
                    id_col: patient_id,
                    time_col: hour * 60,  # Convert hours back to minutes for _align_time_to_admission
                    concept_name: value
                })
        
        if not expanded_rows:
            result_cols = [id_col, time_col, concept_name]
            return pd.DataFrame(columns=result_cols)
        
        expanded = pd.DataFrame(expanded_rows)
        
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
            
            # Create grid dataframe
            grid = pd.DataFrame({
                id_col: group[id_col].iloc[0],
                time_col: all_minutes
            })
            
            # Merge with data
            merged = grid.merge(group[[time_col, concept_name]], on=time_col, how='left')
            
            # Forward fill (locf)
            merged[concept_name] = merged[concept_name].ffill()
            
            return merged
        
        expanded = expanded.groupby(id_col, group_keys=False).apply(fill_gaps_locf)
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
        if val_var not in work.columns:
            work[val_var] = pd.to_numeric(work.iloc[:, 0], errors="coerce")
        else:
            work[val_var] = pd.to_numeric(work[val_var], errors="coerce")

        if sub_var and sub_var in work.columns:
            work["unit_var"] = eicu_extract_unit(work[sub_var])
        else:
            work["unit_var"] = np.nan

        work = _normalize_units(work, val_var, "unit_var")

        # Expand into hourly windows so that pyricu matches ricu's exposure logic.
        expanded = expand_intervals(work, keep_vars=[val_var, "unit_var"])
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
        # dur_var是原始的offset，需要看情况处理
        # 如果传入的是drugstopoffset，它本身就是停止时间偏移量，不是duration
        # 这种情况不应该发生（因为我们优先使用{concept}_dur列）
        # 但作为回退，假设它是分钟
        duration_minutes = dur_vals

    # Filter: duration <= 0 set to 1 min
    duration_minutes = duration_minutes.fillna(1.0)
    duration_minutes = duration_minutes.where(duration_minutes > 0, 1.0)

    # Filter: duration <= 12 hours (720 minutes)
    mask = duration_minutes <= 720.0
    work = work.loc[mask].copy()
    duration_minutes = duration_minutes.loc[mask]

    # rate = value / duration_minutes * 5
    duration_minutes = duration_minutes.where(duration_minutes > 0, 1.0)
    work[val_var] = work[val_var] / duration_minutes * 5.0
    work["unit_var"] = "ml/min"
    
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

    # 🔧 FIX: 使用数值类型（小时）而不是Timedelta，与eicu_dex_med保持一致
    # 这样在合并两个表时不会有类型冲突
    interval_hours = interval.total_seconds() / 3600.0
    work["dur_var"] = interval_hours
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
        
        # Expand intervals into hourly points
        expanded_rows = []
        for _, row in result.iterrows():
            start_min = row[index_col]
            stop_min = row[stop_col]
            
            if pd.isna(start_min) or pd.isna(stop_min):
                continue
            if stop_min <= start_min:
                continue
            
            # 🔧 CRITICAL FIX 2024-11-29: Match R ricu expand() behavior
            # R ricu uses floor(start) to floor(end) for hourly intervals
            # Example: start=1602 min (26.7h), stop=2480 min (41.33h)
            # → floor to hours: 26h to 41h → 16 rows
            # Previous bug: np.arange(start, stop, step) gave 15 rows (stop exclusive)
            #
            # Floor start and stop to hour boundaries
            start_hour = np.floor(start_min / step_minutes) * step_minutes
            stop_hour = np.floor(stop_min / step_minutes) * step_minutes
            
            # Generate time points from floor(start) to floor(stop) inclusive
            # Add step_minutes to stop_hour to make it inclusive
            time_points = np.arange(start_hour, stop_hour + step_minutes, step_minutes)
            if len(time_points) == 0:
                time_points = np.array([start_hour])
            
            for t in time_points:
                new_row = row.copy()
                new_row[index_col] = t
                expanded_rows.append(new_row)
        
        if expanded_rows:
            result = pd.DataFrame(expanded_rows)
            # Drop stop_col from result (not needed after expand)
            if stop_col in result.columns:
                result = result.drop(columns=[stop_col])
            
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
                except Exception as e:
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
                except Exception as e:
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
    start_hours_floor = (grouped[start_col] / 60.0).apply(lambda x: int(x) if pd.notna(x) else x)
    stop_hours_floor = (grouped[stop_var] / 60.0).apply(lambda x: int(x) if pd.notna(x) else x)
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
        old_len = len(df)
        df = df[ug_mask]
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
        # 🔧 FIX: Floor time based on interval_minutes to match R ricu's change_interval behavior
        # - interval=60min (1h): floor to hours, so multiple 15-min records in same hour get summed
        # - interval=1min: floor to minutes, so each 15-min record stays separate (no aggregation)
        time_series = df[index_col]
        if pd.api.types.is_numeric_dtype(time_series):
            # Already in hours (from datetime conversion)
            if interval_minutes >= 60:
                # Floor to hours
                df['_time_bin'] = np.floor(time_series).astype(int)
            else:
                # Floor to minutes: time_hours * 60 -> minutes, floor, back to hours for consistency
                # But keep original time for minute-level grouping
                time_in_minutes = time_series * 60
                df['_time_bin'] = np.floor(time_in_minutes).astype(int) / 60  # Keep in hours but with minute precision
        else:
            # This shouldn't happen for HiRID after datetime conversion
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
        actual_val_col: 'sum',  # Sum doses within each hour
    })
    
    # Map weight back
    grouped['weight'] = grouped[id_col].map(weight_map).fillna(default_weight)
    
    # Step 5: Calculate rate = dose / interval_minutes / weight
    # 🔧 FIX: R ricu uses frac = 1 / interval(x), where interval(x) is the concept's
    # interval attribute. This affects the grouping and rate calculation:
    # - interval=60min: data is grouped by hour, doses summed, rate = sum/60/weight
    # - interval=1min: each point is independent (no aggregation beyond same minute),
    #   rate = dose/1/weight (60x higher values)
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
    
    # 🔧 FIX: R ricu calls expand_intervals at the end of hirid_rate_kg
    # This creates time intervals and expands to fill gaps (forward-fill)
    # expand_intervals(x, val_var, grp_var) creates intervals using create_intervals
    # with overhang=1h, max_len=6h, then expands with 1h step_size
    keep_vars = [concept_name]
    if unit_col and unit_col in result.columns:
        keep_vars.append(unit_col)
    result = expand_intervals(result, keep_vars=keep_vars, grp_var=grp_var)
    
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
    2. Pad the last difference with padding_hours
    3. Cap differences > max_gap_hours to padding_hours
    4. Store as dur_var for window table processing
    5. Expand windows to hourly time series (like R ricu's expand())
    
    Args:
        frame: DataFrame with ventilation records
        concept_name: Name of the output column
        val_col: Name of the value column
        index_col: Name of the time/index column
        dur_var: Name of the duration column to create
        padding_hours: Duration to use for last record and capped gaps
        max_gap_hours: Maximum allowed gap between records
        expand_to_hourly: If True, expand windows to hourly time series
        
    Returns:
        DataFrame with hourly expanded ventilation indicator
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
        # No ID column, add default duration
        df[dur_var] = padding_hours
        return df
    
    # Detect time column
    actual_index_col = index_col
    if not actual_index_col:
        for cand in ['datetime', 'charttime', 'time', 'givenat']:
            if cand in df.columns:
                actual_index_col = cand
                break
    
    if not actual_index_col or actual_index_col not in df.columns:
        # No time column, add default duration
        df[dur_var] = padding_hours
        return df
    
    # Sort by patient and time
    df = df.sort_values([id_col, actual_index_col])
    
    # Calculate padded_capped_diff per patient
    def padded_capped_diff(time_series: pd.Series) -> pd.Series:
        """
        Calculate time diffs, pad last with padding_hours, cap at max_gap_hours.
        
        R: padded_diff <- function(x, final) c(diff(x), final)
        R: padded_capped_diff <- function(x, final, max) { res <- padded_diff(x, final); res[res > max] <- final; res }
        """
        if len(time_series) == 0:
            return pd.Series(dtype=float)
        
        if len(time_series) == 1:
            return pd.Series([padding_hours], index=time_series.index)
        
        # Get values and convert to hours if needed
        time_vals = time_series.values
        
        # Handle datetime/timedelta types
        if np.issubdtype(time_vals.dtype, np.datetime64):
            # Convert to hours relative to first value
            time_vals = (time_vals - time_vals[0]).astype('timedelta64[h]').astype(float)
        elif np.issubdtype(time_vals.dtype, np.timedelta64):
            # Already timedelta, convert to hours
            time_vals = time_vals.astype('timedelta64[h]').astype(float)
        else:
            # Assume already numeric (hours)
            time_vals = np.asarray(time_vals, dtype=float)
        
        # Calculate diff
        diff_vals = np.diff(time_vals)
        
        # Pad with final value
        padded = np.append(diff_vals, padding_hours)
        
        # Cap values > max_gap_hours
        padded[padded > max_gap_hours] = padding_hours
        
        return pd.Series(padded, index=time_series.index)
    
    # Apply per patient
    df[dur_var] = df.groupby(id_col)[actual_index_col].transform(padded_capped_diff)
    
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
    
    records = []
    
    # Ensure index is numeric (hours from ICU admission)
    time_col_data = df[index_col]
    if pd.api.types.is_datetime64_any_dtype(time_col_data):
        # Convert datetime to hours relative to first time per patient
        # This should already be done upstream, but handle it just in case
        df = df.copy()
        df['_start_hours'] = df.groupby(id_col)[index_col].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600
        )
        start_col = '_start_hours'
    elif hasattr(time_col_data.dtype, 'kind') and time_col_data.dtype.kind == 'm':  # timedelta
        # Convert timedelta to hours
        df = df.copy()
        df['_start_hours'] = time_col_data.dt.total_seconds() / 3600
        start_col = '_start_hours'
    else:
        # Already numeric
        start_col = index_col
    
    for _, row in df.iterrows():
        patient_id = row[id_col]
        start_hours = float(row[start_col])
        duration_hours = float(row[dur_col])
        
        if np.isnan(start_hours) or np.isnan(duration_hours) or duration_hours <= 0:
            continue
        
        # Get value
        if value_col and value_col in row.index:
            value = row[value_col]
        else:
            value = True  # Default indicator value
        
        # Generate hourly rows
        end_hours = start_hours + duration_hours
        current_hour = int(np.floor(start_hours))
        
        while current_hour < end_hours:
            records.append({
                id_col: patient_id,
                index_col: float(current_hour),
                concept_name: value,
            })
            current_hour += 1
    
    if not records:
        return pd.DataFrame(columns=[id_col, index_col, concept_name])
    
    result = pd.DataFrame(records)
    
    # Remove duplicates (same patient, same hour)
    result = result.drop_duplicates(subset=[id_col, index_col], keep='first')
    
    # Sort by patient and time
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
    if frame.empty:
        return pd.DataFrame(columns=[concept_name])
    
    df = frame.copy()
    
    # Identify patient ID column
    id_col = None
    for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid', 'subject_id']:
        if cand in df.columns:
            id_col = cand
            break
    
    if id_col is None:
        # No ID column, return empty
        return pd.DataFrame(columns=[concept_name])
    
    # Detect time column
    actual_index_col = index_col
    if not actual_index_col:
        for cand in ['datetime', 'charttime', 'time', 'givenat']:
            if cand in df.columns:
                actual_index_col = cand
                break
    
    if not actual_index_col or actual_index_col not in df.columns:
        # No time column, return empty
        return pd.DataFrame(columns=[concept_name])
    
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
