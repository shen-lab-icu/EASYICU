"""Low-level data loading functions (R ricu data-load.R).

Provides load_src, load_difftime, load_id, load_ts, and load_win functions
for loading data from ICU data sources.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Mapping, Optional, Union
from pathlib import Path

import pandas as pd

from .table import IdTbl, TsTbl, WinTbl, as_id_tbl, as_ts_tbl, as_win_tbl, id_vars
from .datasource import ICUDataSource, FilterSpec, FilterOp
from .config import DataSourceConfig
from .src_utils import src_name
from .ts_utils import change_interval, hours, mins, minutes
from .table_meta import id_var_opts, default_vars, time_vars


def load_src(
    x: Union[str, ICUDataSource, Any],
    rows: Optional[Callable] = None,
    cols: Optional[Iterable[str]] = None,
    src: Optional[Union[str, ICUDataSource, DataSourceConfig]] = None,
    **kwargs
) -> pd.DataFrame:
    """Load data from source table (R ricu load_src).
    
    This is the lowest level data loading function that loads a subset of
    rows/columns from a tabular data source.
    
    Args:
        x: Table name (string) or source table object
        rows: Optional row filter function or expression
        cols: Optional list of column names to load
        src: Data source name (if x is a string)
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with loaded data
        
    Examples:
        >>> load_src('labevents', src='mimic_demo', cols=['itemid', 'value'])
        >>> load_src(data_source, rows=lambda df: df['itemid'] == 50809)
    """
    # Extract keyword arguments relevant for ICUDataSource initialisation
    datasource_kw_keys = {"base_path", "table_sources", "registry", "default_format", "enable_cache"}
    datasource_kwargs = {
        key: kwargs.pop(key) for key in list(kwargs.keys()) if key in datasource_kw_keys
    }

    # Optional explicit table name can be supplied via kwargs
    table_kw = kwargs.pop("table", None) or kwargs.pop("table_name", None)

    # load_table currently accepts only 'verbose'
    load_table_kwargs = {}
    if "verbose" in kwargs:
        load_table_kwargs["verbose"] = bool(kwargs.pop("verbose"))

    if kwargs:
        raise TypeError(
            f"Unsupported keyword arguments for load_src: {sorted(kwargs.keys())}"
        )

    table_name: Optional[str] = table_kw if isinstance(table_kw, str) else None
    data_source: Optional[ICUDataSource] = None

    if isinstance(x, str):
        if table_name is not None and table_name != x:
            raise ValueError(
                f"Conflicting table names provided: '{x}' (positional) vs '{table_name}'"
            )
        table_name = x
    elif isinstance(x, ICUDataSource):
        data_source = x
    else:
        raise TypeError(
            "load_src expects either a table name (str) or ICUDataSource as first argument"
        )

    if table_name is None:
        table_name = getattr(data_source, "table_name", None)
        if table_name is None:
            raise ValueError("Table name must be provided when x is an ICUDataSource.")

    if data_source is None:
        if src is None:
            raise ValueError("src argument required when x is a string")
        if isinstance(src, ICUDataSource):
            data_source = src
        elif isinstance(src, DataSourceConfig):
            data_source = ICUDataSource(src, **datasource_kwargs)
        elif isinstance(src, str):
            from .resources import load_data_sources

            registry = load_data_sources()
            config = registry.get(src)
            if config is None:
                raise ValueError(f"Data source '{src}' not found")
            data_source = ICUDataSource(config, **datasource_kwargs)
        else:
            raise TypeError(
                "src must be a data source name, DataSourceConfig, or ICUDataSource instance"
            )
    else:
        if datasource_kwargs:
            raise ValueError(
                "Data source keyword arguments such as base_path may only be provided when "
                "src is a string or DataSourceConfig"
            )

    columns = list(cols) if cols is not None else None

    filter_specs: List[FilterSpec] = []
    post_filter: Optional[Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]] = None

    if rows is not None:
        if callable(rows):
            post_filter = rows
        elif isinstance(rows, FilterSpec):
            filter_specs.append(rows)
        elif isinstance(rows, Iterable) and all(isinstance(item, FilterSpec) for item in rows):
            filter_specs.extend(rows)  # type: ignore[arg-type]
        elif isinstance(rows, Mapping):
            for column, value in rows.items():
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                    filter_specs.append(FilterSpec(column=column, op=FilterOp.IN, value=value))
                else:
                    filter_specs.append(FilterSpec(column=column, op=FilterOp.EQ, value=value))
        else:
            raise TypeError(
                "rows must be a callable, FilterSpec, iterable of FilterSpec, or mapping of filters"
            )

    table = data_source.load_table(
        table_name,
        columns=columns,
        filters=filter_specs or None,
        **load_table_kwargs,
    )
    frame = table.data.copy()

    if post_filter is not None:
        filtered = post_filter(frame)
        if isinstance(filtered, pd.DataFrame):
            frame = filtered
        else:
            mask = pd.Series(filtered)
            if mask.dtype != bool:
                mask = mask.astype(bool, errors="ignore")
            frame = frame.loc[mask]

    return frame


def load_difftime(
    x: Union[str, ICUDataSource, Any],
    rows: Optional[Callable] = None,
    cols: Optional[Iterable[str]] = None,
    id_hint: Optional[str] = None,
    time_vars: Optional[Iterable[str]] = None,
    src: Optional[str] = None,
    **kwargs
) -> IdTbl:
    """Load data with timestamps converted to difftime (R ricu load_difftime).
    
    Loads data and converts timestamp columns to relative time (difftime).
    Times are relative to the origin provided by the ID system.
    
    Args:
        x: Table name (string) or source table object
        rows: Optional row filter function
        cols: Optional list of column names to load
        id_hint: Suggested ID column (may not be honored)
        time_vars: Columns to treat as timestamps
        src: Data source name (if x is a string)
        **kwargs: Additional arguments
        
    Returns:
        IdTbl with time columns as Timedelta
        
    Examples:
        >>> load_difftime('labevents', src='mimic_demo', id_hint='icustay_id')
    """
    # Load raw data
    data = load_src(x, rows=rows, cols=cols, src=src, **kwargs)
    
    # Get data source for config
    if isinstance(x, str):
        if src is None:
            raise ValueError("src argument required when x is a string")
        from .resources import load_data_sources
        registry = load_data_sources()
        config = registry.get(src)
        if not config:
            raise ValueError(f"Data source '{src}' not found")
        data_source = ICUDataSource(config)
    elif isinstance(x, ICUDataSource):
        data_source = x
        config = x.config
    else:
        raise TypeError(f"Cannot determine data source from {type(x)}")
    
    # Determine ID column
    if id_hint and id_hint in data.columns:
        id_col = id_hint
    else:
        # Try to resolve from config
        id_col = _resolve_id_hint(data, config, id_hint)
    
    # Determine time variables
    if time_vars is None:
        # Get from config if available
        if hasattr(x, 'time_vars'):
            time_vars_list = x.time_vars
        else:
            # Try to infer from data
            time_vars_list = [col for col in data.columns 
                            if pd.api.types.is_datetime64_any_dtype(data[col])]
    else:
        time_vars_list = list(time_vars)
    
    # Filter time_vars to those present in data
    time_vars_list = [col for col in time_vars_list if col in data.columns]
    
    # Convert timestamps to relative time
    if time_vars_list and id_col:
        # Get origin times
        try:
            from .data_env import get_src_env
            from .data_utils import id_origin
            src_env = get_src_env(src_name(config))
            if src_env:
                # Try to get origin from data source configuration
                # For now, skip origin lookup - will be implemented in data_env
                origin_df = None
                if origin_df is not None and len(origin_df) > 0:
                    # Get origin column name
                    origin_cols = [col for col in origin_df.columns 
                                  if col != id_col and pd.api.types.is_datetime64_any_dtype(origin_df[col])]
                    if origin_cols:
                        origin_col = origin_cols[0]
                        # Merge with origin
                        data = data.merge(origin_df[[id_col, origin_col]], on=id_col, how='left')
                        
                        # Convert to relative time
                        for time_col in time_vars_list:
                            if pd.api.types.is_datetime64_any_dtype(data[time_col]):
                                data[time_col] = data[time_col] - data[origin_col]
                                # Convert to minutes
                                data[time_col] = data[time_col] / pd.Timedelta(minutes=1)
                                data[time_col] = pd.to_timedelta(data[time_col], unit='minutes')
                        
                        # Drop origin column
                        if origin_col in data.columns:
                            data = data.drop(columns=[origin_col])
        except Exception:
            # If id_origin fails, try direct conversion (for eICU-like sources)
            for time_col in time_vars_list:
                if pd.api.types.is_numeric_dtype(data[time_col]):
                    # Assume already in minutes
                    data[time_col] = pd.to_timedelta(data[time_col], unit='minutes')
    
    # Return as IdTbl
    return as_id_tbl(data, id_vars=id_col)


def load_id(
    x: Union[str, ICUDataSource, Any],
    rows: Optional[Callable] = None,
    cols: Optional[Iterable[str]] = None,
    id_var: Optional[str] = None,
    src: Optional[str] = None,
    **kwargs
) -> IdTbl:
    """Load data as id_tbl (R ricu load_id).
    
    Loads data and returns as IdTbl with specified ID variable.
    Guaranteed to return data with requested id_var.
    
    Args:
        x: Table name (string) or source table object
        rows: Optional row filter function
        cols: Optional list of column names to load
        id_var: Requested ID variable (guaranteed to be honored)
        src: Data source name (if x is a string)
        **kwargs: Additional arguments
        
    Returns:
        IdTbl with specified ID variable
        
    Examples:
        >>> load_id('patients', src='mimic_demo', id_var='subject_id')
    """
    # Load with difftime
    tbl = load_difftime(x, rows=rows, cols=cols, id_hint=id_var, src=src, **kwargs)
    
    # Change ID if needed
    if id_var and id_vars(tbl) != [id_var] if isinstance(id_vars(tbl), list) else id_vars(tbl) != id_var:
        from .table import change_id
        tbl = change_id(tbl, id_var)
    
    return tbl


def load_ts(
    x: Union[str, ICUDataSource, Any],
    rows: Optional[Callable] = None,
    cols: Optional[Iterable[str]] = None,
    id_var: Optional[str] = None,
    index_var: Optional[str] = None,
    interval: Optional[pd.Timedelta] = None,
    time_vars: Optional[Iterable[str]] = None,
    src: Optional[str] = None,
    **kwargs
) -> TsTbl:
    """Load data as ts_tbl (R ricu load_ts).
    
    Loads time series data and returns as TsTbl with specified ID and index variables.
    
    Args:
        x: Table name (string) or source table object
        rows: Optional row filter function
        cols: Optional list of column names to load
        id_var: ID variable
        index_var: Index variable (time column)
        interval: Time series interval
        time_vars: Time variables to convert
        src: Data source name (if x is a string)
        **kwargs: Additional arguments
        
    Returns:
        TsTbl with specified metadata
        
    Examples:
        >>> load_ts('vitals', src='mimic_demo', id_var='icustay_id', 
        ...         index_var='charttime', interval=hours(1))
    """
    # Load as id_tbl first
    tbl = load_id(x, rows=rows, cols=cols, id_var=id_var, src=src, **kwargs)
    
    # Determine index variable
    if index_var is None:
        # Try to get from defaults
        if isinstance(x, ICUDataSource):
            # Try to get from table config
            pass
        # Default: use first time column
        time_cols = [col for col in tbl.data.columns 
                     if pd.api.types.is_timedelta64_dtype(tbl.data[col])]
        if time_cols:
            index_var = time_cols[0]
        else:
            raise ValueError("Cannot determine index_var")
    
    # Convert to ts_tbl
    ts_tbl = as_ts_tbl(tbl, index_var=index_var)
    
    # Change interval if specified
    if interval is not None:
        ts_tbl = change_interval(ts_tbl, interval)
    
    return ts_tbl


def load_win(
    x: Union[str, ICUDataSource, Any],
    rows: Optional[Callable] = None,
    cols: Optional[Iterable[str]] = None,
    id_var: Optional[str] = None,
    index_var: Optional[str] = None,
    dur_var: Optional[str] = None,
    interval: Optional[pd.Timedelta] = None,
    src: Optional[str] = None,
    **kwargs
) -> WinTbl:
    """Load data as win_tbl (R ricu load_win).
    
    Loads windowed time series data and returns as WinTbl.
    
    Args:
        x: Table name (string) or source table object
        rows: Optional row filter function
        cols: Optional list of column names to load
        id_var: ID variable
        index_var: Index variable (time column)
        dur_var: Duration variable
        interval: Time series interval
        src: Data source name (if x is a string)
        **kwargs: Additional arguments
        
    Returns:
        WinTbl with specified metadata
        
    Examples:
        >>> load_win('ventilation', src='mimic_demo', id_var='icustay_id',
        ...          index_var='starttime', dur_var='duration')
    """
    # Load as ts_tbl first
    ts_tbl = load_ts(x, rows=rows, cols=cols, id_var=id_var, 
                     index_var=index_var, interval=interval, src=src, **kwargs)
    
    # Determine duration variable
    if dur_var is None:
        # Try to infer from column names
        dur_candidates = ['duration', 'dur', 'dur_var']
        for candidate in dur_candidates:
            if candidate in ts_tbl.data.columns:
                dur_var = candidate
                break
        
        if dur_var is None:
            raise ValueError("Cannot determine dur_var")
    
    # Convert to win_tbl
    win_tbl = as_win_tbl(ts_tbl, dur_var=dur_var)
    
    return win_tbl


def _resolve_id_hint(data: pd.DataFrame, config: DataSourceConfig, hint: Optional[str]) -> str:
    """Resolve ID column from hint (R ricu resolve_id_hint).
    
    Args:
        data: DataFrame
        config: Data source configuration
        hint: Suggested ID column name
        
    Returns:
        Resolved ID column name
    """
    if hint and hint in data.columns:
        return hint
    
    # Try to get from config
    if config and config.id_configs:
        id_opts = list(config.id_configs.keys())
        # Find first ID option present in data
        for id_opt in id_opts:
            id_cfg = config.id_configs[id_opt]
            if id_cfg.id in data.columns:
                return id_cfg.id
    
    # Fallback: use first column
    if len(data.columns) > 0:
        return data.columns[0]
    
    raise ValueError("Cannot resolve ID column")
