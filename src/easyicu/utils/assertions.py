"""Data validation and assertion utilities (R ricu assertions.R).

Provides assertion functions for validating data types, structures, and values,
similar to R's assertthat package but adapted for Python/pandas.

完全复刻 R ricu 的断言系统，包括所有验证函数。
"""

from typing import Any, Optional, Union, List, Callable
import pandas as pd
import numpy as np
from pathlib import Path

class RicuAssertionError(Exception):
    """Custom assertion error for ricu-style assertions."""
    pass

def assert_that(condition: bool, msg: Optional[str] = None, env=None) -> bool:
    """Assert a condition is True (R ricu assert_that).
    
    完全复刻 R ricu 的 assert_that 函数。
    
    Args:
        condition: Boolean condition to check
        msg: Optional error message
        env: Environment (for compatibility, not used in Python)
        
    Returns:
        True if condition is met
        
    Raises:
        RicuAssertionError: If condition is False
        
    Examples:
        >>> assert_that(True)
        True
        >>> assert_that(1 + 1 == 2)
        True
        >>> assert_that(False, "This should fail")
        RicuAssertionError: This should fail
    """
    if not condition:
        if msg is None:
            msg = "Assertion failed"
        raise RicuAssertionError(msg)
    return True

def is_string(x: Any) -> bool:
    """Check if x is a string (R assertthat::is.string)."""
    return isinstance(x, str)

def is_flag(x: Any) -> bool:
    """Check if x is a boolean (R assertthat::is.flag)."""
    return isinstance(x, bool)

def is_scalar(x: Any) -> bool:
    """Check if x is a scalar value (R ricu is_scalar)."""
    # 包含 timedelta 类型
    return isinstance(x, (int, float, str, bool, np.integer, np.floating, 
                         pd.Timedelta, np.timedelta64)) and \
           not isinstance(x, (list, tuple, np.ndarray))

def is_number(x: Any) -> bool:
    """Check if x is a numeric scalar (R assertthat::is.number)."""
    return isinstance(x, (int, float, np.integer, np.floating)) and is_scalar(x)

def is_count(x: Any) -> bool:
    """Check if x is a positive integer (R assertthat::is.count)."""
    return isinstance(x, (int, np.integer)) and x > 0

def is_intish(x: Union[float, np.ndarray, pd.Series]) -> bool:
    """Check if numeric value(s) are integer-like (R ricu is_intish).
    
    Returns True if all values equal their truncated versions.
    """
    if isinstance(x, (pd.Series, np.ndarray)):
        return np.all(x == np.trunc(x)) and not np.any(pd.isna(x))
    else:
        return x == int(x) and not pd.isna(x)

def no_na(x: Union[pd.Series, pd.DataFrame, np.ndarray]) -> bool:
    """Check if data contains no NA values (R ricu no_na)."""
    if isinstance(x, pd.DataFrame):
        return not x.isna().any().any()
    elif isinstance(x, (pd.Series, np.ndarray)):
        return not pd.isna(x).any()
    else:
        return not pd.isna(x)

def has_length(x: Any, length: Optional[int] = None) -> bool:
    """Check if x has expected length (R ricu has_length).
    
    Args:
        x: Object to check
        length: Expected length (if None, just checks length > 0)
    """
    try:
        actual_len = len(x)
    except TypeError:
        return False
    
    if length is None:
        return actual_len > 0
    else:
        return actual_len == length

def has_rows(x: pd.DataFrame) -> bool:
    """Check if DataFrame has at least one row (R ricu has_rows)."""
    return len(x) > 0

def has_cols(x: pd.DataFrame, cols: Optional[Union[str, List[str]]] = None) -> bool:
    """Check if DataFrame has expected columns.
    
    Args:
        x: DataFrame to check
        cols: Column name(s) to check for (if None, just checks ncols > 0)
    """
    if cols is None:
        return len(x.columns) > 0
    
    if isinstance(cols, str):
        cols = [cols]
    
    return all(col in x.columns for col in cols)

def has_name(x: Any, name: str) -> bool:
    """Check if object has an attribute (R assertthat::has_name)."""
    return hasattr(x, name)

def are_in(x: List[str], opts: List[str], na_rm: bool = False) -> bool:
    """Check if all values are in options (R ricu are_in).
    
    Args:
        x: Values to check
        opts: Valid options
        na_rm: Whether to ignore NA values
    """
    if not has_length(x) or not has_length(opts):
        return False
    
    if na_rm:
        x = [v for v in x if v is not None and not pd.isna(v)]
    
    return all(v in opts for v in x)

def is_unique(x: Union[pd.Series, List, np.ndarray, pd.DataFrame]) -> bool:
    """Check if all values are unique (R ricu is_unique).
    
    对于 DataFrame，检查行是否唯一。
    对于 Series/List/Array，检查值是否唯一。
    """
    if isinstance(x, pd.DataFrame):
        # 检查行是否唯一
        return not x.duplicated().any()
    elif isinstance(x, pd.Series):
        return not x.duplicated().any()
    else:
        return len(x) == len(set(x))

def is_sorted(x: Union[pd.Series, List, np.ndarray]) -> bool:
    """Check if values are sorted (R ricu is_sorted)."""
    if isinstance(x, pd.Series):
        x = x.values
    elif isinstance(x, list):
        x = np.array(x)
    
    return np.all(x[:-1] <= x[1:])

def is_disjoint(x: set, y: set) -> bool:
    """Check if two sets are disjoint (no common elements)."""
    return len(x & y) == 0

def is_subset(x: set, y: set) -> bool:
    """Check if x is a subset of y."""
    return x <= y

def is_directory(path: Union[str, Path]) -> bool:
    """Check if path is a directory (R assertthat::is.dir)."""
    return Path(path).is_dir()

def is_file(path: Union[str, Path]) -> bool:
    """Check if path is a file."""
    return Path(path).is_file()

def is_data_frame(x: Any) -> bool:
    """Check if x is a DataFrame."""
    return isinstance(x, pd.DataFrame)

def is_series(x: Any) -> bool:
    """Check if x is a Series."""
    return isinstance(x, pd.Series)

def not_null(x: Any) -> bool:
    """Check if x is not None (R ricu not_null)."""
    return x is not None

def null_or(x: Any, condition: Callable) -> bool:
    """Check if x is None or satisfies condition (R ricu null_or)."""
    return x is None or condition(x)

# Validation helpers for common patterns

def validate_data_frame(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    min_rows: int = 0,
    no_na_cols: Optional[List[str]] = None,
) -> bool:
    """Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_cols: Columns that must be present
        min_rows: Minimum number of rows
        no_na_cols: Columns that must not have NA values
        
    Returns:
        True if all validations pass
        
    Raises:
        AssertionError: If validation fails
    """
    assert_that(is_data_frame(df), "Input must be a DataFrame")
    
    if min_rows > 0:
        assert_that(len(df) >= min_rows, 
                   f"DataFrame must have at least {min_rows} rows, got {len(df)}")
    
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        assert_that(len(missing) == 0,
                   f"Missing required columns: {missing}")
    
    if no_na_cols:
        for col in no_na_cols:
            assert_that(col in df.columns,
                       f"Column '{col}' not found in DataFrame")
            assert_that(no_na(df[col]),
                       f"Column '{col}' contains NA values")
    
    return True

def validate_id_tbl(
    df: pd.DataFrame,
    id_vars: List[str],
) -> bool:
    """Validate id_tbl structure (R ricu validate_tbl for id_tbl).
    
    Args:
        df: DataFrame to validate
        id_vars: ID variable names
        
    Returns:
        True if valid
    """
    assert_that(is_data_frame(df), "Must be a DataFrame")
    assert_that(has_length(id_vars), "id_vars must be non-empty")
    assert_that(has_cols(df, id_vars), f"Missing ID columns: {id_vars}")
    
    # Check for unique column names
    assert_that(len(df.columns) == len(set(df.columns)),
               "Column names must be unique")
    
    # Check for NA in ID columns
    for id_var in id_vars:
        assert_that(no_na(df[id_var]),
                   f"ID column '{id_var}' contains NA values")
    
    return True

def validate_ts_tbl(
    df: pd.DataFrame,
    id_vars: List[str],
    index_var: str,
    interval: Optional[pd.Timedelta] = None,
) -> bool:
    """Validate ts_tbl structure (R ricu validate_tbl for ts_tbl).
    
    Args:
        df: DataFrame to validate
        id_vars: ID variable names
        index_var: Time index variable name
        interval: Expected time interval
        
    Returns:
        True if valid
    """
    # First validate as id_tbl
    validate_id_tbl(df, id_vars)
    
    # Check index_var
    assert_that(is_string(index_var), "index_var must be a string")
    assert_that(index_var in df.columns,
               f"Index column '{index_var}' not found")
    assert_that(index_var not in id_vars,
               f"Index column '{index_var}' cannot be an ID column")
    
    # Check index column type
    assert_that(pd.api.types.is_timedelta64_dtype(df[index_var]) or
               pd.api.types.is_datetime64_any_dtype(df[index_var]),
               f"Index column '{index_var}' must be datetime or timedelta")
    
    # Check for NA in index
    assert_that(no_na(df[index_var]),
               f"Index column '{index_var}' contains NA values")
    
    # Optionally validate interval
    if interval is not None:
        # Check that all time differences are multiples of interval
        for id_group, group_df in df.groupby(id_vars):
            if len(group_df) > 1:
                sorted_times = group_df[index_var].sort_values()
                diffs = sorted_times.diff().dropna()
                
                if len(diffs) > 0:
                    # Check if all differences are multiples of interval
                    remainders = diffs % interval
                    assert_that(all(remainders == pd.Timedelta(0)),
                               f"Time differences are not multiples of interval {interval}")
    
    return True

def validate_win_tbl(
    df: pd.DataFrame,
    id_vars: List[str],
    index_var: str,
    dur_var: str,
    interval: Optional[pd.Timedelta] = None,
) -> bool:
    """Validate win_tbl structure (R ricu validate_tbl for win_tbl).
    
    Args:
        df: DataFrame to validate
        id_vars: ID variable names
        index_var: Time index variable name
        dur_var: Duration variable name
        interval: Expected time interval
        
    Returns:
        True if valid
    """
    # First validate as ts_tbl
    validate_ts_tbl(df, id_vars, index_var, interval)
    
    # Check dur_var
    assert_that(is_string(dur_var), "dur_var must be a string")
    assert_that(dur_var in df.columns,
               f"Duration column '{dur_var}' not found")
    assert_that(dur_var not in id_vars + [index_var],
               f"Duration column '{dur_var}' cannot be an ID or index column")
    
    # Check duration column type
    assert_that(pd.api.types.is_timedelta64_dtype(df[dur_var]),
               f"Duration column '{dur_var}' must be timedelta")
    
    return True

# Convenient assertion wrappers

def assert_string(x: Any, msg: Optional[str] = None):
    """Assert x is a string."""
    assert_that(is_string(x), msg or f"{x} is not a string")

def assert_flag(x: Any, msg: Optional[str] = None):
    """Assert x is a boolean."""
    assert_that(is_flag(x), msg or f"{x} is not a boolean")

def assert_scalar(x: Any, msg: Optional[str] = None):
    """Assert x is a scalar."""
    assert_that(is_scalar(x), msg or f"{x} is not a scalar")

def assert_number(x: Any, msg: Optional[str] = None):
    """Assert x is a number."""
    assert_that(is_number(x), msg or f"{x} is not a number")

def assert_count(x: Any, msg: Optional[str] = None):
    """Assert x is a positive integer."""
    assert_that(is_count(x), msg or f"{x} is not a positive integer")

def assert_no_na(x: Any, msg: Optional[str] = None):
    """Assert x contains no NA values."""
    assert_that(no_na(x), msg or "Data contains NA values")

def assert_has_rows(df: pd.DataFrame, msg: Optional[str] = None):
    """Assert DataFrame has at least one row."""
    assert_that(has_rows(df), msg or "DataFrame has no rows")

def assert_has_cols(df: pd.DataFrame, cols: Union[str, List[str]], 
                   msg: Optional[str] = None):
    """Assert DataFrame has required columns."""
    if isinstance(cols, str):
        cols = [cols]
    assert_that(has_cols(df, cols), 
               msg or f"DataFrame missing columns: {cols}")

# R ricu 特有的断言函数

def all_equal(x: Any, y: Any) -> bool:
    """Check if x and y are equal (R ricu all_equal).
    
    类似 R 的 all.equal，进行深度比较。
    """
    try:
        if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
            return x.equals(y)
        elif isinstance(x, pd.Series) and isinstance(y, pd.Series):
            return x.equals(y)
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.array_equal(x, y, equal_nan=True)
        else:
            return x == y
    except Exception:
        return False

def same_length(x: Any, y: Any) -> bool:
    """Check if x and y have the same length (R ricu same_length)."""
    try:
        return len(x) == len(y)
    except TypeError:
        return False

def are_equal(x: Any, y: Any) -> bool:
    """Alias for all_equal (R assertthat::are_equal)."""
    return all_equal(x, y)

def has_attr(x: Any, attr: str) -> bool:
    """Check if object has attribute (R assertthat::has_attr)."""
    return hasattr(x, attr)

def is_difftime(x: Any) -> bool:
    """Check if x is a timedelta type (R ricu is_difftime)."""
    if isinstance(x, pd.Series):
        return pd.api.types.is_timedelta64_dtype(x)
    elif isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.timedelta64)
    else:
        return isinstance(x, (pd.Timedelta, np.timedelta64))

def is_interval(x: Any, length: Optional[int] = None) -> bool:
    """Check if x is a valid time interval (R ricu is_interval).
    
    时间间隔必须是 timedelta 类型且非负。
    """
    if not is_difftime(x):
        return False
    
    if is_scalar(x):
        if length is not None and length != 1:
            return False
        try:
            # 直接比较标量 timedelta
            td = pd.Timedelta(x) if not isinstance(x, pd.Timedelta) else x
            return td >= pd.Timedelta(0)
        except Exception:
            return False
    else:
        if length is not None and len(x) != length:
            return False
        try:
            # 数组/Series 的比较
            if isinstance(x, pd.Series):
                return (x >= pd.Timedelta(0)).all()
            else:
                return all(x >= pd.Timedelta(0))
        except Exception:
            return False

def obeys_interval(x: Union[pd.Series, np.ndarray], 
                   interval: pd.Timedelta,
                   na_rm: bool = True,
                   tolerance: pd.Timedelta = pd.Timedelta(milliseconds=1)) -> bool:
    """Check if time values obey interval (R ricu obeys_interval).
    
    检查时间序列是否符合给定的间隔。
    
    Args:
        x: 时间序列
        interval: 期望的时间间隔
        na_rm: 是否忽略 NA 值
        tolerance: 允许的误差
    """
    if not is_difftime(x):
        return False
    
    if isinstance(x, pd.Series):
        x = x.values
    
    # 检查每个时间点是否是 interval 的整数倍
    if na_rm:
        x = x[~pd.isna(x)]
    
    try:
        # 转换为纳秒进行比较
        x_ns = x.astype('timedelta64[ns]').astype('int64')
        interval_ns = interval.value
        tolerance_ns = tolerance.value
        
        remainders = x_ns % interval_ns
        return all(remainders < tolerance_ns)
    except Exception:
        return False

def same_unit(x: Union[pd.Timedelta, pd.Series], 
              y: Union[pd.Timedelta, pd.Series]) -> bool:
    """Check if x and y have the same time unit (R ricu same_unit).
    
    在 Python/pandas 中，timedelta 会自动标准化，所以这个检查总是返回 True。
    保留此函数是为了 API 兼容性。
    """
    return is_difftime(x) and is_difftime(y)

def same_time(x: Union[pd.Timedelta, pd.Series],
              y: Union[pd.Timedelta, pd.Series],
              tolerance: pd.Timedelta = pd.Timedelta(milliseconds=1)) -> bool:
    """Check if x and y are on the same time scale (R ricu same_time).
    
    检查两个时间值是否相等（在误差范围内）。
    """
    if not same_unit(x, y):
        return False
    
    try:
        diff = abs(x - y)
        if is_scalar(diff):
            return diff < tolerance
        else:
            return all(diff < tolerance)
    except Exception:
        return False

def has_interval(df: pd.DataFrame, interval: pd.Timedelta, 
                index_var: Optional[str] = None) -> bool:
    """Check if DataFrame has expected time interval (R ricu has_interval).
    
    Args:
        df: DataFrame to check
        interval: Expected time interval
        index_var: Time index column name (if None, try to detect)
    """
    if index_var is None:
        # 尝试检测时间列
        time_cols = [col for col in df.columns 
                    if pd.api.types.is_timedelta64_dtype(df[col]) or
                       pd.api.types.is_datetime64_any_dtype(df[col])]
        if len(time_cols) != 1:
            return False
        index_var = time_cols[0]
    
    if index_var not in df.columns:
        return False
    
    return obeys_interval(df[index_var], interval)

def has_time_cols(df: pd.DataFrame, cols: Union[str, List[str]], 
                 length: Optional[int] = None) -> bool:
    """Check if specified columns are time columns (R ricu has_time_cols).
    
    Args:
        df: DataFrame to check
        cols: Column name(s) to check
        length: Expected number of columns (if None, just check has_length)
    """
    if isinstance(cols, str):
        cols = [cols]
    
    if not has_cols(df, cols):
        return False
    
    if length is not None and len(cols) != length:
        return False
    
    return all(is_difftime(df[col]) for col in cols)

def all_fun(x: Any, fun: Callable, *args, na_rm: bool = False, **kwargs) -> bool:
    """Check if function returns True for all elements (R ricu all_fun).
    
    Args:
        x: Iterable to check
        fun: Function to apply
        *args: Positional arguments to fun
        na_rm: Whether to ignore NA values
        **kwargs: Keyword arguments to fun
    """
    try:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if na_rm:
                x = x.dropna()
            results = x.apply(lambda item: fun(item, *args, **kwargs))
            return all(results)
        else:
            if na_rm:
                x = [item for item in x if not pd.isna(item)]
            return all(fun(item, *args, **kwargs) for item in x)
    except Exception:
        return False

def all_null(x: Any) -> bool:
    """Check if all elements are None/null (R ricu all_null)."""
    return all_fun(x, lambda item: item is None or pd.isna(item))

def evals_to_fun(x: Any) -> bool:
    """Check if x is or evaluates to a function (R ricu evals_to_fun)."""
    if callable(x):
        return True
    
    if isinstance(x, str):
        try:
            # 尝试将字符串作为函数名
            import builtins
            return hasattr(builtins, x) and callable(getattr(builtins, x))
        except Exception:
            return False
    
    return False

# 为 id_tbl 和 ts_tbl 特定的验证

def is_id_tbl(x: Any) -> bool:
    """Check if x is an id_tbl (R ricu is_id_tbl)."""
    return isinstance(x, pd.DataFrame) and hasattr(x, 'attrs') and 'id_vars' in x.attrs

def is_ts_tbl(x: Any) -> bool:
    """Check if x is a ts_tbl (R ricu is_ts_tbl)."""
    return (is_id_tbl(x) and 
            'index_var' in x.attrs and 
            'interval' in x.attrs)

def is_win_tbl(x: Any) -> bool:
    """Check if x is a win_tbl (R ricu is_win_tbl)."""
    return is_ts_tbl(x) and 'dur_var' in x.attrs

# 断言包装器（抛出异常而不是返回 False）

def assert_unique(x: Any, msg: Optional[str] = None):
    """Assert all values are unique."""
    assert_that(is_unique(x), msg or "Values are not unique")

def assert_sorted(x: Any, msg: Optional[str] = None):
    """Assert values are sorted."""
    assert_that(is_sorted(x), msg or "Values are not sorted")

def assert_disjoint(x: set, y: set, msg: Optional[str] = None):
    """Assert sets are disjoint."""
    assert_that(is_disjoint(x, y), msg or "Sets are not disjoint")

def assert_interval(x: Any, length: Optional[int] = None, msg: Optional[str] = None):
    """Assert x is a valid interval."""
    assert_that(is_interval(x, length), msg or "Not a valid interval")

def assert_id_tbl(x: Any, msg: Optional[str] = None):
    """Assert x is an id_tbl."""
    assert_that(is_id_tbl(x), msg or "Not an id_tbl")

def assert_ts_tbl(x: Any, msg: Optional[str] = None):
    """Assert x is a ts_tbl."""
    assert_that(is_ts_tbl(x), msg or "Not a ts_tbl")

def assert_win_tbl(x: Any, msg: Optional[str] = None):
    """Assert x is a win_tbl."""
    assert_that(is_win_tbl(x), msg or "Not a win_tbl")
