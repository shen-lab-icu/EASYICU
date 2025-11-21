"""DataFrame utility functions for common data manipulation tasks.

This module provides reusable utility functions for working with ICU data,
reducing code duplication across callback functions.
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np

# Known ID column names by database
KNOWN_ID_COLUMNS = {
    'miiv': ['stay_id', 'subject_id', 'hadm_id'],
    'eicu': ['patientunitstayid', 'patientid'],
    'aumc': ['admissionid', 'patientid'],
    'hirid': ['patientid'],
    'sic': ['patientid', 'icustay_id'],
}

# Known time column names by database
KNOWN_TIME_COLUMNS = {
    'miiv': ['charttime', 'starttime', 'endtime'],
    'eicu': ['observationoffset', 'nursingchartoffset', 'offset'],
    'aumc': ['measuredat', 'start', 'stop'],
    'hirid': ['datetime', 'charttime'],
    'sic': ['datetime', 'charttime'],
}

# Generic fallback patterns
GENERIC_ID_PATTERNS = ['_id', 'id']
GENERIC_TIME_PATTERNS = ['time', 'offset', 'date']

def infer_id_columns(df: pd.DataFrame, database: Optional[str] = None) -> List[str]:
    """Infer ID columns from DataFrame.
    
    Args:
        df: Input DataFrame
        database: Optional database name for database-specific inference
        
    Returns:
        List of ID column names
        
    Examples:
        >>> df = pd.DataFrame({'stay_id': [1, 2], 'value': [10, 20]})
        >>> infer_id_columns(df, 'miiv')
        ['stay_id']
    """
    if df.empty:
        return []
    
    # Try database-specific columns first
    if database and database in KNOWN_ID_COLUMNS:
        for col in KNOWN_ID_COLUMNS[database]:
            if col in df.columns:
                # Return first match (primary ID)
                return [col]
    
    # Fall back to generic pattern matching
    id_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Check if column ends with '_id' or is exactly 'id'
        if any(pattern in col_lower for pattern in GENERIC_ID_PATTERNS):
            id_cols.append(col)
    
    return id_cols

def infer_time_column(df: pd.DataFrame, database: Optional[str] = None) -> Optional[str]:
    """Infer primary time column from DataFrame.
    
    Args:
        df: Input DataFrame
        database: Optional database name for database-specific inference
        
    Returns:
        Time column name or None if not found
        
    Examples:
        >>> df = pd.DataFrame({'charttime': pd.date_range('2020-01-01', periods=3)})
        >>> infer_time_column(df, 'miiv')
        'charttime'
    """
    if df.empty:
        return None
    
    # Try database-specific columns first
    if database and database in KNOWN_TIME_COLUMNS:
        for col in KNOWN_TIME_COLUMNS[database]:
            if col in df.columns:
                return col
    
    # Fall back to generic pattern matching
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in GENERIC_TIME_PATTERNS):
            # Prefer datetime dtype columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
    
    # If no datetime column found, return first time-like named column
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in GENERIC_TIME_PATTERNS):
            return col
    
    return None

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has unique column names.
    
    Handles MultiIndex columns and duplicate column names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with unique column names
        
    Examples:
        >>> df = pd.DataFrame([[1, 2]], columns=['a', 'a'])
        >>> result = ensure_unique_columns(df)
        >>> list(result.columns)
        ['a']
    """
    if df.empty:
        return df
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join(str(c) for c in col).strip('_')
            for col in df.columns.values
        ]
    
    # Remove duplicate columns (keep first)
    return df.loc[:, ~df.columns.duplicated(keep='first')]

def safe_merge_weight(
    data: pd.DataFrame,
    weight: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    database: Optional[str] = None
) -> pd.DataFrame:
    """Safely merge weight data with primary data.
    
    Handles cases where ID columns don't match between datasets.
    
    Args:
        data: Primary data DataFrame
        weight: Weight data DataFrame
        id_cols: Optional explicit ID columns
        database: Optional database name
        
    Returns:
        Merged DataFrame with weight column
    """
    if weight.empty:
        return data
    
    if id_cols is None:
        id_cols = infer_id_columns(data, database)
    
    # Check if weight has matching ID columns
    common_ids = [col for col in id_cols if col in weight.columns]
    
    if common_ids:
        # Standard merge
        result = pd.merge(data, weight, on=common_ids, how='left', suffixes=('', '_weight'))
    else:
        # Broadcast weight (assume single patient or no ID match needed)
        result = data.copy()
        if 'weight' in weight.columns and len(weight) > 0:
            result['weight'] = weight['weight'].iloc[0]
    
    return result

def filter_time_range(
    df: pd.DataFrame,
    min_time: Optional[float] = None,
    max_time: Optional[float] = None,
    time_col: Optional[str] = None,
    database: Optional[str] = None
) -> pd.DataFrame:
    """Filter DataFrame by time range.
    
    Args:
        df: Input DataFrame
        min_time: Minimum time (inclusive), None for no lower bound
        max_time: Maximum time (exclusive), None for no upper bound
        time_col: Time column name, inferred if None
        database: Database name for column inference
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    if time_col is None:
        time_col = infer_time_column(df, database)
    
    if time_col is None or time_col not in df.columns:
        return df
    
    result = df.copy()
    
    if min_time is not None:
        result = result[result[time_col] >= min_time]
    
    if max_time is not None:
        result = result[result[time_col] < max_time]
    
    return result

def standardize_column_dtype(
    df: pd.DataFrame,
    col: str,
    target_dtype: str,
    errors: str = 'coerce'
) -> pd.DataFrame:
    """Standardize column data type.
    
    Args:
        df: Input DataFrame
        col: Column name
        target_dtype: Target dtype ('float', 'int', 'datetime', 'timedelta')
        errors: How to handle conversion errors ('coerce', 'raise', 'ignore')
        
    Returns:
        DataFrame with standardized column dtype
    """
    if df.empty or col not in df.columns:
        return df
    
    df = df.copy()
    
    if target_dtype == 'float':
        df[col] = pd.to_numeric(df[col], errors=errors)
    elif target_dtype == 'int':
        df[col] = pd.to_numeric(df[col], errors=errors)
        if errors == 'coerce':
            df[col] = df[col].astype('Int64')  # Nullable integer
    elif target_dtype == 'datetime':
        df[col] = pd.to_datetime(df[col], errors=errors)
    elif target_dtype == 'timedelta':
        if not pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = pd.to_timedelta(df[col], errors=errors)
    
    return df

def get_grouped_aggregate(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    agg_func: str = 'sum',
    sort: bool = True
) -> pd.DataFrame:
    """Aggregate data by groups.
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        value_col: Column to aggregate
        agg_func: Aggregation function ('sum', 'mean', 'min', 'max', 'first', 'last')
        sort: Whether to sort by group columns
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty:
        return df
    
    if sort:
        df = df.sort_values(group_cols)
    
    agg_dict = {value_col: agg_func}
    result = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    return result

def remove_na_rows(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    how: str = 'any'
) -> pd.DataFrame:
    """Remove rows with NA values in specified columns.
    
    Args:
        df: Input DataFrame
        cols: Column name(s) to check for NA
        how: 'any' or 'all' - remove if any or all specified cols are NA
        
    Returns:
        DataFrame with NA rows removed
    """
    if df.empty:
        return df
    
    if isinstance(cols, str):
        cols = [cols]
    
    # Only check columns that exist
    cols = [col for col in cols if col in df.columns]
    
    if not cols:
        return df
    
    if how == 'any':
        return df.dropna(subset=cols, how='any')
    elif how == 'all':
        return df.dropna(subset=cols, how='all')
    else:
        raise ValueError(f"Invalid how parameter: {how}. Must be 'any' or 'all'")

def forward_fill_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    value_cols: Union[str, List[str]],
    max_gap: Optional[pd.Timedelta] = None,
    time_col: Optional[str] = None
) -> pd.DataFrame:
    """Forward fill values within groups.
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        value_cols: Column(s) to forward fill
        max_gap: Maximum time gap to fill, None for unlimited
        time_col: Time column for gap checking
        
    Returns:
        DataFrame with forward filled values
    """
    if df.empty:
        return df
    
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    
    df = df.copy()
    
    def fill_group(group):
        if max_gap is not None and time_col is not None:
            # Fill only within max_gap
            group = group.sort_values(time_col)
            for col in value_cols:
                if col in group.columns:
                    # Simple forward fill
                    group[col] = group[col].fillna(method='ffill')
        else:
            # Simple forward fill
            for col in value_cols:
                if col in group.columns:
                    group[col] = group[col].fillna(method='ffill')
        return group
    
    result = df.groupby(group_cols, group_keys=False).apply(fill_group)
    
    return result
