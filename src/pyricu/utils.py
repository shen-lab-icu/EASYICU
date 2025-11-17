"""Miscellaneous utility functions (R ricu utils-misc.R).

Collection of general-purpose utility functions used throughout pyricu,
corresponding to R ricu's utils-misc.R.
"""

from typing import Any, Callable, List, Optional, Union, TypeVar, Dict, Tuple
import pandas as pd
import numpy as np
from functools import reduce as functools_reduce
import hashlib
import pickle
import warnings

T = TypeVar('T')


# ============================================================================
# Numeric operations
# ============================================================================

def round_to(x: Union[float, np.ndarray], to: float = 1.0) -> Union[float, np.ndarray]:
    """Round to nearest multiple (R ricu round_to).
    
    Args:
        x: Value(s) to round
        to: Round to nearest multiple of this value
        
    Returns:
        Rounded value(s)
        
    Examples:
        >>> round_to(3.7, 5)
        5.0
        >>> round_to(12.3, 10)
        10.0
    """
    return np.round(x / to) * to


def is_val(x: Union[Any, np.ndarray, pd.Series], val: Any) -> Union[bool, np.ndarray]:
    """Check if value equals val, ignoring NA (R ricu is_val)."""
    if isinstance(x, (np.ndarray, pd.Series)):
        return ~pd.isna(x) & (x == val)
    else:
        return not pd.isna(x) and x == val


def not_val(x: Union[Any, np.ndarray, pd.Series], val: Any) -> Union[bool, np.ndarray]:
    """Check if value not equals val, ignoring NA (R ricu not_val)."""
    if isinstance(x, (np.ndarray, pd.Series)):
        return ~pd.isna(x) & (x != val)
    else:
        return not pd.isna(x) and x != val


def val_or_na(x: Union[Any, np.ndarray, pd.Series], val: Any) -> Union[bool, np.ndarray]:
    """Check if value equals val or is NA (R ricu val_or_na)."""
    if isinstance(x, (np.ndarray, pd.Series)):
        return pd.isna(x) | (x == val)
    else:
        return pd.isna(x) or x == val


# ============================================================================
# Logical operations
# ============================================================================

def is_true(x: Union[bool, np.ndarray, pd.Series]) -> Union[bool, np.ndarray]:
    """Check if value is True (not NA and True) (R ricu is_true)."""
    if isinstance(x, (np.ndarray, pd.Series)):
        return ~pd.isna(x) & x
    else:
        return not pd.isna(x) and x


def is_false(x: Union[bool, np.ndarray, pd.Series]) -> Union[bool, np.ndarray]:
    """Check if value is False (not NA or False) (R ricu is_false)."""
    if isinstance(x, (np.ndarray, pd.Series)):
        return ~(pd.isna(x) | x)
    else:
        return not (pd.isna(x) or x)


# ============================================================================
# Sequence operations
# ============================================================================

def first_elem(x: Union[List, np.ndarray, pd.Series]) -> Any:
    """Get first element (R ricu first_elem)."""
    if isinstance(x, pd.Series):
        return x.iloc[0] if len(x) > 0 else None
    elif isinstance(x, np.ndarray):
        return x[0] if len(x) > 0 else None
    else:
        return x[0] if len(x) > 0 else None


def last_elem(x: Union[List, np.ndarray, pd.Series]) -> Any:
    """Get last element (R ricu last_elem)."""
    if isinstance(x, pd.Series):
        return x.iloc[-1] if len(x) > 0 else None
    elif isinstance(x, np.ndarray):
        return x[-1] if len(x) > 0 else None
    else:
        return x[-1] if len(x) > 0 else None


def rep_along(x: Any, times: Union[List, int]) -> List:
    """Repeat x along times (R ricu rep_along)."""
    if isinstance(times, int):
        return [x] * times
    else:
        return [x] * len(times)


# ============================================================================
# Functional programming utilities
# ============================================================================

def reduce(func: Callable, x: List, *args, **kwargs) -> Any:
    """Reduce a list with a function (R ricu reduce)."""
    return functools_reduce(lambda a, b: func(a, b, *args, **kwargs), x)


def map_list(func: Callable, *iterables) -> List:
    """Map function over iterables, returning list (R ricu map)."""
    return list(map(func, *iterables))


def do_call(func: Callable, args: List, kwargs: Optional[Dict] = None) -> Any:
    """Call function with args and kwargs (R ricu do_call)."""
    if kwargs is None:
        kwargs = {}
    return func(*args, **kwargs)


# ============================================================================
# List operations
# ============================================================================

def unlst(x: List, recursive: bool = False, use_names: bool = False) -> List:
    """Unlist (flatten) a list (R ricu unlst)."""
    if not recursive:
        return [item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])]
    else:
        result = []
        for item in x:
            if isinstance(item, list):
                result.extend(unlst(item, recursive=True))
            else:
                result.append(item)
        return result


def lst_inv(x: Dict[Any, List]) -> Dict[Any, Any]:
    """Invert a dictionary of lists (R ricu lst_inv).
    
    Args:
        x: Dict with list values
        
    Returns:
        Inverted dict where list items become keys
        
    Examples:
        >>> lst_inv({'a': [1, 2], 'b': [3]})
        {1: 'a', 2: 'a', 3: 'b'}
    """
    result = {}
    for key, values in x.items():
        for value in values:
            result[value] = key
    return result


# ============================================================================
# Coalescing and default values
# ============================================================================

def coalesce(*args) -> Any:
    """Return first non-None, non-NA value (R ricu coalesce).
    
    Examples:
        >>> coalesce(None, np.nan, 5, 10)
        5
    """
    for arg in args:
        if arg is not None and not pd.isna(arg):
            return arg
    return None


def rep_arg(arg: Any, names: List[str]) -> Dict[str, Any]:
    """Replicate argument for each name (R ricu rep_arg).
    
    Args:
        arg: Single value or dict
        names: List of names
        
    Returns:
        Dict mapping names to arg values
    """
    if isinstance(arg, dict):
        return {name: arg.get(name, arg.get('default')) for name in names}
    else:
        return {name: arg for name in names}


# ============================================================================
# Set operations
# ============================================================================

def set_names(obj: Union[List, Dict], names: List[str]) -> Union[List, Dict]:
    """Set names for object (R ricu set_names)."""
    if isinstance(obj, dict):
        return {name: val for name, val in zip(names, obj.values())}
    else:
        return dict(zip(names, obj))


def new_names(old_names: List[str] = None, n: int = 1, 
              prefix: str = "V") -> List[str]:
    """Generate new unique names (R ricu new_names).
    
    Args:
        old_names: Existing names to avoid
        n: Number of new names to generate
        prefix: Prefix for new names
        
    Returns:
        List of n new unique names
    """
    if old_names is None:
        old_names = []
    
    i = 1
    new = []
    while len(new) < n:
        candidate = f"{prefix}{i}"
        if candidate not in old_names and candidate not in new:
            new.append(candidate)
        i += 1
    
    return new


# ============================================================================
# Time utilities
# ============================================================================

def ms_as_mins(x: Union[int, float]) -> pd.Timedelta:
    """Convert milliseconds to minutes (R ricu ms_as_mins)."""
    return pd.Timedelta(minutes=x / 60000)


def min_as_mins(x: Union[int, float]) -> pd.Timedelta:
    """Convert numeric minutes to timedelta (R ricu min_as_mins)."""
    return pd.Timedelta(minutes=x)


# ============================================================================
# Apply-like functions
# ============================================================================

def chr_ply(x: List, func: Callable, *args, length: int = 1, 
           use_names: bool = False, **kwargs) -> List[str]:
    """Apply function expecting string results (R ricu chr_ply)."""
    return [str(func(item, *args, **kwargs)) for item in x]


def lgl_ply(x: List, func: Callable, *args, length: int = 1,
           use_names: bool = False, **kwargs) -> List[bool]:
    """Apply function expecting boolean results (R ricu lgl_ply)."""
    return [bool(func(item, *args, **kwargs)) for item in x]


def int_ply(x: List, func: Callable, *args, length: int = 1,
           use_names: bool = False, **kwargs) -> List[int]:
    """Apply function expecting integer results (R ricu int_ply)."""
    return [int(func(item, *args, **kwargs)) for item in x]


def dbl_ply(x: List, func: Callable, *args, length: int = 1,
           use_names: bool = False, **kwargs) -> List[float]:
    """Apply function expecting float results (R ricu dbl_ply)."""
    return [float(func(item, *args, **kwargs)) for item in x]


def col_ply(df: pd.DataFrame, cols: List[str], func: Callable,
           ply_func: Callable = lgl_ply, *args, **kwargs) -> Any:
    """Apply function to specified columns (R ricu col_ply)."""
    return ply_func([df[col] for col in cols], func, *args, **kwargs)


# ============================================================================
# Extract utilities
# ============================================================================

def lst_xtr(x: List[Dict], key: str) -> List:
    """Extract key from list of dicts (R ricu lst_xtr)."""
    return [item.get(key) for item in x]


def chr_xtr(x: List[Dict], key: str, length: int = 1) -> List[str]:
    """Extract key as strings (R ricu chr_xtr)."""
    return chr_ply(x, lambda item: item.get(key, ""), length=length)


def lgl_xtr(x: List[Dict], key: str, length: int = 1) -> List[bool]:
    """Extract key as booleans (R ricu lgl_xtr)."""
    return lgl_ply(x, lambda item: item.get(key, False), length=length)


def int_xtr(x: List[Dict], key: str, length: int = 1) -> List[int]:
    """Extract key as integers (R ricu int_xtr)."""
    return int_ply(x, lambda item: item.get(key, 0), length=length)


def dbl_xtr(x: List[Dict], key: str, length: int = 1) -> List[float]:
    """Extract key as floats (R ricu dbl_xtr)."""
    return dbl_ply(x, lambda item: item.get(key, 0.0), length=length)


# ============================================================================
# String utilities
# ============================================================================

def concat(*args, sep: str = ", ") -> str:
    """Concatenate strings (R ricu concat)."""
    return sep.join(str(arg) for arg in args)


def quote_bt(x: str) -> str:
    """Quote string with backticks (R ricu quote_bt)."""
    return f"`{x}`"


def enbraket(x: str) -> str:
    """Surround with brackets (R ricu enbraket)."""
    return f"[{x}]"


# ============================================================================
# Formatting utilities
# ============================================================================

def prcnt(x: Union[int, float, List], tot: Optional[Union[int, float]] = None) -> str:
    """Format as percentage (R ricu prcnt)."""
    if isinstance(x, list):
        x = sum(x)
    if tot is None:
        tot = 100
    
    return f"{(x / tot * 100):.1f}%"


def big_mark(x: Union[int, float], sep: str = ",") -> str:
    """Format number with thousands separator (R ricu big_mark)."""
    return f"{x:,}".replace(",", sep)


# ============================================================================
# System utilities
# ============================================================================

def sys_name() -> str:
    """Get system name (R ricu sys_name)."""
    import platform
    return platform.system()


def sys_env(var: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable (R ricu sys_env)."""
    import os
    return os.environ.get(var, default)


# ============================================================================
# Null handling
# ============================================================================

def not_null(x: Any) -> bool:
    """Check if not None (R ricu not_null)."""
    return x is not None


def null_or(x: Any, func: Callable) -> bool:
    """Check if None or satisfies condition (R ricu null_or)."""
    return x is None or func(x)


# ============================================================================
# Aggregation utilities
# ============================================================================

def agg_or_na(agg_func: Callable) -> Callable:
    """Create aggregation function that returns NA if input is all NA (R ricu agg_or_na).
    
    Args:
        agg_func: Aggregation function (e.g., np.mean, np.sum)
        
    Returns:
        Wrapped function that handles all-NA case
    """
    def wrapper(x, *args, **kwargs):
        if isinstance(x, pd.Series):
            if x.isna().all():
                return np.nan
        elif isinstance(x, np.ndarray):
            if np.all(np.isnan(x)):
                return np.nan
        
        return agg_func(x, *args, **kwargs)
    
    return wrapper


def min_or_na(x: Union[pd.Series, np.ndarray, List]) -> Any:
    """Return minimum or NA if all values are NA (R ricu min_or_na).
    
    Unlike base min() which returns Inf for all-NA input with na.rm=TRUE,
    this returns NA.
    
    Args:
        x: Array-like data
        
    Returns:
        Minimum value or NA
        
    Examples:
        >>> min_or_na([1, 2, 3])
        1
        >>> min_or_na([np.nan, np.nan])
        nan
    """
    if isinstance(x, pd.Series):
        if x.isna().all():
            return np.nan
        return x.min(skipna=True)
    elif isinstance(x, np.ndarray):
        x_clean = x[~pd.isna(x)]
        if len(x_clean) == 0:
            return np.nan
        return np.min(x_clean)
    else:
        x_clean = [v for v in x if not pd.isna(v)]
        if len(x_clean) == 0:
            return np.nan
        return min(x_clean)


def max_or_na(x: Union[pd.Series, np.ndarray, List]) -> Any:
    """Return maximum or NA if all values are NA (R ricu max_or_na).
    
    🚀 Optimized: Single-pass computation without isna().all() check.
    
    Unlike base max() which returns -Inf for all-NA input with na.rm=TRUE,
    this returns NA.
    
    Args:
        x: Array-like data
        
    Returns:
        Maximum value or NA
        
    Examples:
        >>> max_or_na([1, 2, 3])
        3
        >>> max_or_na([np.nan, np.nan])
        nan
    """
    if isinstance(x, pd.Series):
        # 🚀 优化：直接用max(skipna=True)，空序列返回nan
        result = x.max(skipna=True)
        return result if not pd.isna(result) else np.nan
    elif isinstance(x, np.ndarray):
        # 🚀 优化：使用np.nanmax，更快且单次遍历
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmax(x)
        return result if not (np.isnan(result) if isinstance(result, (int, float)) else False) else np.nan
    else:
        x_clean = [v for v in x if not pd.isna(v)]
        if len(x_clean) == 0:
            return np.nan
        return max(x_clean)


# Commonly used aggregation functions
mean_or_na = agg_or_na(np.nanmean)
sum_or_na = agg_or_na(np.nansum)


# ============================================================================
# Pandas-specific utilities
# ============================================================================

def safe_concat(dfs: List[pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    """Safely concatenate DataFrames, handling empty list."""
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, *args, **kwargs)


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, 
              *args, **kwargs) -> pd.DataFrame:
    """Safely merge DataFrames, handling empty cases."""
    if len(left) == 0 or len(right) == 0:
        return pd.DataFrame()
    return pd.merge(left, right, *args, **kwargs)


# ============================================================================
# Additional utilities from R ricu
# ============================================================================

def split_msg(msg: str, width: int = 80) -> List[str]:
    """Split message into lines of max width (R ricu split_msg).
    
    Args:
        msg: Message to split
        width: Maximum line width
        
    Returns:
        List of lines
    """
    import textwrap
    return textwrap.wrap(msg, width=width)


def fmt_class(obj: Any) -> str:
    """Format object class name (R ricu fmt_class).
    
    Returns:
        String representation of class name
    """
    return obj.__class__.__name__


def is_dt(x: Any) -> bool:
    """Check if datetime type (R ricu is_dt).
    
    Args:
        x: Object to check
        
    Returns:
        True if x is datetime-like
    """
    if isinstance(x, pd.Series):
        return pd.api.types.is_datetime64_any_dtype(x)
    elif isinstance(x, pd.DataFrame):
        return any(pd.api.types.is_datetime64_any_dtype(x[col]) for col in x.columns)
    else:
        return isinstance(x, (pd.Timestamp, np.datetime64, pd.DatetimeIndex))


def is_unique(x: Union[List, np.ndarray, pd.Series]) -> bool:
    """Check if all elements are unique (R ricu is_unique).
    
    Args:
        x: Sequence to check
        
    Returns:
        True if all elements are unique
    """
    if isinstance(x, pd.Series):
        return x.is_unique
    elif isinstance(x, np.ndarray):
        return len(x) == len(np.unique(x))
    else:
        return len(x) == len(set(x))


def all_equal(x: Any, y: Any) -> bool:
    """Check if two values are equal (R ricu all_equal).
    
    Handles NA values and arrays properly.
    
    Args:
        x: First value
        y: Second value
        
    Returns:
        True if x equals y (or both are NA)
    """
    if pd.isna(x) and pd.isna(y):
        return True
    if pd.isna(x) or pd.isna(y):
        return False
    
    if isinstance(x, (np.ndarray, pd.Series)):
        if not isinstance(y, (np.ndarray, pd.Series)):
            return False
        return np.array_equal(x, y, equal_nan=True)
    
    return x == y


def digest_lst(x: List) -> str:
    """Create hash digest of list (R ricu digest_lst).
    
    Args:
        x: List to hash
        
    Returns:
        MD5 hash as hex string
    """
    serialized = pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(serialized).hexdigest()


def digest(*args) -> str:
    """Create hash digest of arguments (R ricu digest).
    
    Args:
        *args: Values to hash
        
    Returns:
        MD5 hash as hex string
    """
    return digest_lst(list(args))


def cat_line(*args) -> None:
    """Print line (R ricu cat_line).
    
    Args:
        *args: Values to print
    """
    line = ' '.join(str(arg) for arg in args).rstrip()
    print(line)


def unlst_str(x: List[str]) -> List[str]:
    """Unlist strings (R ricu unlst_str).
    
    Ensures all elements are strings.
    
    Args:
        x: List to process
        
    Returns:
        List of strings
    """
    return [str(item) for item in x]


def has_length(x: Any, length: Optional[int] = None) -> bool:
    """Check if object has length (R ricu has_length).
    
    Args:
        x: Object to check
        length: Expected length (if None, just checks if has length > 0)
        
    Returns:
        True if has specified length
    """
    try:
        actual_len = len(x)
        if length is None:
            return actual_len > 0
        return actual_len == length
    except TypeError:
        return False


def is_scalar(x: Any) -> bool:
    """Check if scalar value (R ricu is.scalar).
    
    Args:
        x: Value to check
        
    Returns:
        True if x is scalar
    """
    return np.isscalar(x) or (isinstance(x, (pd.Series, np.ndarray)) and len(x) == 1)


def xtr_null(x: Dict, key: str, default: Any = None) -> Any:
    """Extract from dict or return default (R ricu xtr_null).
    
    Args:
        x: Dictionary to extract from
        key: Key to extract
        default: Default value if key not found
        
    Returns:
        Value from dict or default
    """
    return x.get(key, default)


def chr_xtr_null(x: List[Dict], key: str, length: int = 1, 
                 default: str = "") -> List[str]:
    """Extract key as strings with null handling (R ricu chr_xtr_null).
    
    Args:
        x: List of dicts
        key: Key to extract
        length: Expected length
        default: Default value
        
    Returns:
        List of string values
    """
    return [str(xtr_null(item, key, default)) for item in x]


def lgl_xtr_null(x: List[Dict], key: str, length: int = 1,
                 default: bool = False) -> List[bool]:
    """Extract key as booleans with null handling (R ricu lgl_xtr_null)."""
    return [bool(xtr_null(item, key, default)) for item in x]


def int_xtr_null(x: List[Dict], key: str, length: int = 1,
                 default: int = 0) -> List[int]:
    """Extract key as integers with null handling (R ricu int_xtr_null)."""
    result = []
    for item in x:
        val = xtr_null(item, key, default)
        try:
            result.append(int(val) if val is not None else default)
        except (ValueError, TypeError):
            result.append(default)
    return result


def dbl_xtr_null(x: List[Dict], key: str, length: int = 1,
                 default: float = 0.0) -> List[float]:
    """Extract key as floats with null handling (R ricu dbl_xtr_null)."""
    result = []
    for item in x:
        val = xtr_null(item, key, default)
        try:
            result.append(float(val) if val is not None else default)
        except (ValueError, TypeError):
            result.append(default)
    return result


def bullet(text: str, level: int = 1) -> str:
    """Create bulleted text (R ricu bullet).
    
    Args:
        text: Text to bullet
        level: Bullet level (1, 2, or 3)
        
    Returns:
        Bulleted string
    """
    bullets = {1: "•", 2: "○", 3: "-"}
    bullet_char = bullets.get(level, "•")
    return f"{bullet_char} {text}"


def ensure_list(x: Any) -> List:
    """Ensure value is a list (pyricu utility).
    
    Args:
        x: Value to convert
        
    Returns:
        List containing x (or x itself if already a list)
    """
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    elif isinstance(x, (tuple, set)):
        return list(x)
    else:
        return [x]


def flatten_list(x: List[Any]) -> List[Any]:
    """Flatten nested list one level (pyricu utility).
    
    Args:
        x: Nested list
        
    Returns:
        Flattened list
    """
    result = []
    for item in x:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


# ============================================================================
# NA/NULL replacement utilities
# ============================================================================

def replace_na(
    x: Union[pd.Series, pd.DataFrame, np.ndarray, List],
    val: Any,
    by_ref: bool = False
) -> Union[pd.Series, pd.DataFrame, np.ndarray, List]:
    """Replace NA values with a constant (R ricu replace_na).
    
    Args:
        x: Data with potential NA values
        val: Replacement value
        by_ref: If True, modify in place (for DataFrame)
        
    Returns:
        Data with NA values replaced
        
    Examples:
        >>> replace_na([1, np.nan, 3], 0)
        [1.0, 0.0, 3.0]
        
        >>> s = pd.Series([1, np.nan, 3])
        >>> replace_na(s, 0)
        0    1.0
        1    0.0
        2    3.0
        dtype: float64
        
        >>> df = pd.DataFrame({'a': [1, np.nan], 'b': [3, 4]})
        >>> replace_na(df, 0)
           a  b
        0  1  3
        1  0  4
    """
    if isinstance(x, pd.DataFrame):
        if by_ref:
            x.fillna(val, inplace=True)
            return x
        else:
            return x.fillna(val)
    elif isinstance(x, pd.Series):
        return x.fillna(val)
    elif isinstance(x, np.ndarray):
        result = x.copy()
        result[pd.isna(result)] = val
        return result
    elif isinstance(x, list):
        return [val if pd.isna(v) else v for v in x]
    else:
        return val if pd.isna(x) else x


def has_interval(x: pd.DataFrame, id_vars: List[str], index_var: str) -> bool:
    """Check if time series has regular intervals (R ricu has_interval).
    
    Args:
        x: DataFrame with time series data
        id_vars: ID column names
        index_var: Time index column name
        
    Returns:
        True if all time intervals are consistent
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'id': [1, 1, 1],
        ...     'time': pd.timedelta_range(0, periods=3, freq='1H'),
        ...     'value': [1, 2, 3]
        ... })
        >>> has_interval(df, ['id'], 'time')
        True
    """
    if not isinstance(x, pd.DataFrame):
        return False
    
    if index_var not in x.columns:
        return False
    
    if not pd.api.types.is_timedelta64_dtype(x[index_var]) and \
       not pd.api.types.is_datetime64_any_dtype(x[index_var]):
        return False
    
    # Check each group
    for _, group in x.groupby(id_vars):
        if len(group) < 2:
            continue
        
        sorted_times = group[index_var].sort_values()
        diffs = sorted_times.diff().dropna()
        
        if len(diffs) == 0:
            continue
        
        # Check if all diffs are equal
        if not all(diffs == diffs.iloc[0]):
            return False
    
    return True


def is_regular(x: Union[pd.Series, pd.DatetimeIndex, pd.TimedeltaIndex]) -> bool:
    """Check if time index is regular (R ricu is_regular).
    
    Args:
        x: Time series data or index
        
    Returns:
        True if intervals are regular
        
    Examples:
        >>> times = pd.timedelta_range(0, periods=10, freq='1H')
        >>> is_regular(times)
        True
        
        >>> times = pd.to_timedelta([0, 1, 3, 6], unit='H')  # Irregular
        >>> is_regular(times)
        False
    """
    if isinstance(x, pd.Series):
        x = x.values
    
    if len(x) < 2:
        return True
    
    diffs = np.diff(x)
    
    # Check if all differences are equal
    return np.all(diffs == diffs[0])


def binary_op(op: Callable, value: Any) -> Callable:
    """Create unary function from binary operation (R ricu binary_op).
    
    Fixes the second argument of a binary operation.
    
    Args:
        op: Binary operation function
        value: Value to fix as second argument
        
    Returns:
        Unary function
        
    Examples:
        >>> multiply_by_2 = binary_op(lambda x, y: x * y, 2)
        >>> multiply_by_2(5)
        10
        
        >>> add_10 = binary_op(lambda x, y: x + y, 10)
        >>> add_10(5)
        15
    """
    def unary_op(x):
        return op(x, value)
    return unary_op


def identity(x: Any) -> Any:
    """Identity function - returns input unchanged (R ricu identity).
    
    Args:
        x: Any value
        
    Returns:
        The same value unchanged
        
    Examples:
        >>> identity(5)
        5
        >>> identity([1, 2, 3])
        [1, 2, 3]
    """
    return x


def compact(x: List) -> List:
    """Remove NULL/None values from list (R ricu compact).
    
    Args:
        x: List potentially containing None values
        
    Returns:
        List with None values removed
        
    Examples:
        >>> compact([1, None, 2, None, 3])
        [1, 2, 3]
    """
    return [item for item in x if item is not None]


def keep_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Keep only specified columns (R dplyr::select).
    
    Args:
        df: DataFrame
        cols: Columns to keep
        
    Returns:
        DataFrame with only specified columns
    """
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def drop_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop specified columns (R dplyr::select with -).
    
    Args:
        df: DataFrame
        cols: Columns to drop
        
    Returns:
        DataFrame with specified columns removed
    """
    return df.drop(columns=[c for c in cols if c in df.columns])


def rename_cols(df: pd.DataFrame, rename_dict: Dict[str, str],
                by_ref: bool = False) -> pd.DataFrame:
    """Rename columns (R dplyr::rename).
    
    Args:
        df: DataFrame
        rename_dict: Mapping from old to new column names
        by_ref: If True, rename in place
        
    Returns:
        DataFrame with renamed columns
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> rename_cols(df, {'a': 'x', 'b': 'y'})
           x  y
        0  1  3
        1  2  4
    """
    if by_ref:
        df.rename(columns=rename_dict, inplace=True)
        return df
    else:
        return df.rename(columns=rename_dict)


def arrange(df: pd.DataFrame, by: Union[str, List[str]], 
           ascending: Union[bool, List[bool]] = True) -> pd.DataFrame:
    """Sort DataFrame by columns (R dplyr::arrange).
    
    Args:
        df: DataFrame to sort
        by: Column name(s) to sort by
        ascending: Sort order(s)
        
    Returns:
        Sorted DataFrame
        
    Examples:
        >>> df = pd.DataFrame({'a': [3, 1, 2], 'b': [1, 2, 3]})
        >>> arrange(df, 'a')
           a  b
        1  1  2
        2  2  3
        0  3  1
    """
    return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)


def distinct(df: pd.DataFrame, subset: Optional[List[str]] = None, 
            keep: str = 'first') -> pd.DataFrame:
    """Keep distinct/unique rows (R dplyr::distinct).
    
    Args:
        df: DataFrame
        subset: Column names to consider for uniqueness
        keep: Which duplicate to keep ('first', 'last', False)
        
    Returns:
        DataFrame with duplicates removed
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 1, 2]})
        >>> distinct(df)
           a  b
        0  1  1
        2  2  2
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
