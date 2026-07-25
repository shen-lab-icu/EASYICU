"""Advanced low-level data loading (interface design inspired by R ricu).

``load_src``, ``load_difftime``, ``load_id``, ``load_ts`` and ``load_win`` read
a *named source table* into a typed ICU table, with timestamps expressed
relative to the ID system's declared origin.

**For clinical variables, use :func:`easyicu.load_concepts` instead.** That is
the supported path: it resolves concepts across all six databases through the
concept dictionary, applies the callback chain, and is what every extraction in
this project runs on. The functions here are for reading a table the concept
layer does not cover — a custom export, a source-specific table, an ad-hoc
join — and they hand back raw source columns with no concept harmonisation,
no unit normalisation and no bounds checking.

These entry points are shaped like ricu's, but this is not a compatibility
layer and no ricu-API equivalence is claimed; ``pyproject.toml`` says
"inspired by", and the real inheritance from ricu is the concept dictionary.

Nothing inside EasyICU calls this module, which is how three signature errors
and a dead relative-time conversion survived here until 2026-07-29. Treat it as
a supported but lightly-travelled road: changes need their own tests, because
the extraction suite will not exercise them.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Mapping, Optional, Union

import pandas as pd

from ..table import IdTbl, TsTbl, WinTbl, as_id_tbl, as_ts_tbl, as_win_tbl, id_vars
from ..datasource import ICUDataSource, FilterSpec, FilterOp
from ..config import DataSourceConfig, IdentifierConfig
from .ts_utils import change_interval


class TimeOriginError(ValueError):
    """A time column could not be expressed relative to a known origin.

    Raised instead of returning the column untouched. A ``charttime`` that is
    still an absolute date, wearing the same name and type a relative one
    would, is indistinguishable downstream from a converted column — the
    windows, the interval flooring and every score built on them would be
    computed against year zero rather than against admission.
    """


#: Time units this loader will accept for a numeric time column.
VALID_TIME_UNITS = frozenset({"seconds", "minutes", "hours", "days"})

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
            from ..resources import load_data_sources

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

def _resolve_source(
    x: Union[str, ICUDataSource, Any], src: Optional[str]
) -> "tuple[DataSourceConfig, Optional[ICUDataSource]]":
    """The config, and a source object able to load the origin/map tables."""

    if isinstance(x, ICUDataSource):
        return x.config, x
    if isinstance(src, ICUDataSource):
        return src.config, src
    if isinstance(src, DataSourceConfig):
        return src, ICUDataSource(src)
    if isinstance(x, str):
        if src is None:
            raise ValueError("src argument required when x is a string")
        from ..resources import load_data_sources

        registry = load_data_sources()
        config = registry.get(src)
        if not config:
            raise ValueError(f"Data source '{src}' not found")
        return config, ICUDataSource(config)
    raise TypeError(f"Cannot determine data source from {type(x)}")


def _id_config_for(config: DataSourceConfig, id_col: str) -> Optional[IdentifierConfig]:
    for cfg in config.id_configs.values():
        if cfg.id == id_col:
            return cfg
    return None


def _load_origin(
    data_source: Optional[ICUDataSource],
    config: DataSourceConfig,
    id_col: str,
) -> "tuple[pd.DataFrame, str]":
    """``(id, origin)`` for one ID system, from that system's own table.

    ``id_cfg`` already records where each identifier's clock starts —
    ``icustay.start = intime`` for MIMIC-IV, ``icustay.start =
    unitadmitoffset`` for eICU. Reading it is the whole conversion; the
    previous code set ``origin_df = None`` with a note that this would be
    implemented later, which left every absolute timestamp absolute.
    """

    cfg = _id_config_for(config, id_col)
    if cfg is None:
        raise TimeOriginError(
            f"no ID system in {config.name!r} is keyed on {id_col!r}, so the "
            "origin its timestamps are relative to is unknown"
        )
    if not cfg.table or not cfg.start:
        raise TimeOriginError(
            f"ID system {cfg.name!r} of {config.name!r} declares no "
            f"{'table' if not cfg.table else 'start column'}, so no origin can "
            "be read for it"
        )
    if data_source is None:
        raise TimeOriginError(
            f"origin table {cfg.table!r} cannot be loaded without a data source"
        )
    origin = data_source.load_table(cfg.table, columns=[cfg.id, cfg.start]).data
    if cfg.start not in origin.columns or cfg.id not in origin.columns:
        raise TimeOriginError(
            f"origin table {cfg.table!r} does not carry {cfg.id!r} and "
            f"{cfg.start!r}"
        )
    return origin[[cfg.id, cfg.start]].drop_duplicates(subset=[cfg.id]), cfg.start


def _to_relative(
    data: pd.DataFrame,
    time_cols: List[str],
    origin: pd.Series,
    *,
    time_unit: Optional[str],
    source: str,
) -> pd.DataFrame:
    """Subtract the origin and return timedeltas."""

    for time_col in time_cols:
        column = data[time_col]
        if pd.api.types.is_datetime64_any_dtype(column) and pd.api.types.is_datetime64_any_dtype(origin):
            data[time_col] = column - origin
            continue
        if pd.api.types.is_numeric_dtype(column) and pd.api.types.is_numeric_dtype(origin):
            if time_unit is None:
                raise TimeOriginError(
                    f"{source}: {time_col!r} is a numeric offset, so its unit "
                    "cannot be read off the column. Pass time_unit='minutes' "
                    "(or 'seconds', 'hours', 'days'). Inferring it from the "
                    "values is what shifted concept windows by 60x before."
                )
            data[time_col] = pd.to_timedelta(column - origin, unit=time_unit)
            continue
        raise TimeOriginError(
            f"{source}: {time_col!r} and its origin are not the same kind of "
            f"time ({column.dtype} against {origin.dtype}), so subtracting one "
            "from the other would not give an elapsed time"
        )
    return data


def load_difftime(
    x: Union[str, ICUDataSource, Any],
    rows: Optional[Callable] = None,
    cols: Optional[Iterable[str]] = None,
    id_hint: Optional[str] = None,
    time_vars: Optional[Iterable[str]] = None,
    src: Optional[str] = None,
    time_unit: Optional[str] = None,
    **kwargs
) -> IdTbl:
    """Load data with timestamps converted to difftime (R ricu load_difftime).

    Loads data and converts timestamp columns to relative time (difftime).
    Times are relative to the origin the ID system declares.

    Args:
        x: Table name (string) or source table object
        rows: Optional row filter function
        cols: Optional list of column names to load
        id_hint: Suggested ID column (may not be honored)
        time_vars: Columns to treat as timestamps
        src: Data source name (if x is a string)
        time_unit: Unit of a NUMERIC time column ('seconds'/'minutes'/'hours'/
            'days'). Required for numeric offsets such as eICU's ``*offset``
            columns; a datetime column is self-describing and needs nothing.
        **kwargs: Additional arguments

    Returns:
        IdTbl with time columns as Timedelta

    Raises:
        TimeOriginError: a time column could not be made relative.

    Examples:
        >>> load_difftime('labevents', src='mimic_demo', id_hint='icustay_id')
    """
    if time_unit is not None and time_unit not in VALID_TIME_UNITS:
        raise ValueError(
            f"unknown time_unit {time_unit!r}; expected one of "
            + ", ".join(sorted(VALID_TIME_UNITS))
        )

    # Load raw data
    data = load_src(x, rows=rows, cols=cols, src=src, **kwargs)

    config, data_source = _resolve_source(x, src)

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
            time_vars_list = list(x.time_vars)
        else:
            # Try to infer from data
            time_vars_list = [col for col in data.columns
                            if pd.api.types.is_datetime64_any_dtype(data[col])]
    else:
        time_vars_list = list(time_vars)

    # Filter time_vars to those present in data
    time_vars_list = [col for col in time_vars_list if col in data.columns]
    # An origin column is a time, not an observation of one.
    time_vars_list = [col for col in time_vars_list if col != id_col]

    already_relative = [
        col for col in time_vars_list if pd.api.types.is_timedelta64_dtype(data[col])
    ]
    pending = [col for col in time_vars_list if col not in already_relative]

    if pending:
        if not id_col:
            raise TimeOriginError(
                f"{config.name}: no ID column was resolved, so "
                f"{pending} cannot be expressed relative to an admission"
            )
        origin_frame, origin_col = _load_origin(data_source, config, id_col)
        merged_origin = f"__origin__{origin_col}"
        data = data.merge(
            origin_frame.rename(columns={origin_col: merged_origin}),
            on=id_col,
            how="left",
        )
        data = _to_relative(
            data,
            pending,
            data[merged_origin],
            time_unit=time_unit,
            source=config.name,
        )
        data = data.drop(columns=[merged_origin])

    # Return as IdTbl
    return as_id_tbl(data, id_vars=id_col)


def _load_id_map(
    data_source: Optional[ICUDataSource],
    config: DataSourceConfig,
    from_id: str,
    to_id: str,
) -> pd.DataFrame:
    """A two-column map between two ID systems of one source.

    Both identifiers live together in whichever id-system table is granular
    enough to carry them — MIMIC-IV's ``icustays`` holds ``subject_id``,
    ``hadm_id`` and ``stay_id`` at once. The most granular table is tried
    first because it is the one that can express the relation without loss.
    """

    if data_source is None:
        raise ValueError(
            f"changing {from_id!r} to {to_id!r} needs a data source to read "
            "the ID map from"
        )
    candidates = sorted(
        (cfg for cfg in config.id_configs.values() if cfg.table),
        key=lambda cfg: cfg.position,
        reverse=True,
    )
    tried: List[str] = []
    for cfg in candidates:
        tried.append(str(cfg.table))
        try:
            frame = data_source.load_table(
                cfg.table, columns=[from_id, to_id]
            ).data
        except Exception:
            continue
        if from_id in frame.columns and to_id in frame.columns:
            return frame[[from_id, to_id]].drop_duplicates()
    raise ValueError(
        f"no table of {config.name!r} carries both {from_id!r} and {to_id!r} "
        f"(tried {tried}), so the two ID systems cannot be related"
    )


def load_id(
    x: Union[str, ICUDataSource, Any],
    rows: Optional[Callable] = None,
    cols: Optional[Iterable[str]] = None,
    id_var: Optional[str] = None,
    src: Optional[str] = None,
    on_many_to_many: Optional[str] = None,
    agg_funcs: Optional[Mapping[str, Any]] = None,
    on_unmapped: str = "error",
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
        on_many_to_many: Passed to :func:`easyicu.table.change_id` when the two
            ID systems are related many-to-many.
        agg_funcs: How to combine each column when several source rows collapse
            onto one target id. Required for integer-valued columns, whose mean
            is not the quantity they measure.
        on_unmapped: What to do with rows the ID map does not cover. Defaults to
            ``'error'``; ``'drop'`` removes them, ``'keep'`` accepts a null id.
        **kwargs: Additional arguments

    Returns:
        IdTbl with specified ID variable

    Examples:
        >>> load_id('patients', src='mimic_demo', id_var='subject_id')
    """
    # Load with difftime
    tbl = load_difftime(x, rows=rows, cols=cols, id_hint=id_var, src=src, **kwargs)

    if not id_var:
        return tbl

    current = id_vars(tbl)
    current_list = [current] if isinstance(current, str) else list(current or [])
    if current_list == [id_var]:
        return tbl
    if len(current_list) != 1:
        raise ValueError(
            f"cannot change a table keyed on {current_list} to {id_var!r}: "
            "only a single-column ID can be remapped"
        )

    from ..table import change_id

    config, data_source = _resolve_source(x, src)
    id_map = _load_id_map(data_source, config, current_list[0], id_var)
    return as_id_tbl(
        change_id(
            tbl.data,
            id_map,
            current_list[0],
            id_var,
            agg_funcs=dict(agg_funcs) if agg_funcs else None,
            on_many_to_many=on_many_to_many,
            on_unmapped=on_unmapped,
        ),
        id_vars=id_var,
    )

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
    
    # Convert to ts_tbl. ``as_ts_tbl`` requires the id columns: the index alone
    # does not say whose series it is.
    ts_tbl = as_ts_tbl(tbl.data, id_vars=id_vars(tbl), index_var=index_var)

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
    duration_unit: Optional[str] = None,
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
        duration_unit: Unit of a NUMERIC duration column ('seconds'/'minutes'/
            'hours'/'days'). Required for numeric durations — a bare number
            carries no unit, and inferring one is what inflated windows 60x.
            Not needed for a timedelta column, which is self-describing.
        **kwargs: Additional arguments

    Returns:
        WinTbl with specified metadata

    Raises:
        DurationUnitError: numeric duration loaded without ``duration_unit``.

    Examples:
        >>> load_win('ventilation', src='mimic_demo', id_var='icustay_id',
        ...          index_var='starttime', dur_var='duration',
        ...          duration_unit='minutes')
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
    
    # A numeric duration is meaningless without a unit, so demand one here
    # rather than letting a downstream consumer guess.
    from ..table.duration import (
        UNIT_TIMEDELTA,
        VALID_DUR_VAR_UNITS,
        DurationUnitError,
        set_dur_var_unit,
    )

    duration_is_timedelta = pd.api.types.is_timedelta64_dtype(ts_tbl.data[dur_var])
    if duration_unit is not None:
        if duration_unit not in VALID_DUR_VAR_UNITS:
            raise DurationUnitError(
                f"unknown duration_unit {duration_unit!r}; expected one of "
                + ", ".join(sorted(VALID_DUR_VAR_UNITS))
            )
        set_dur_var_unit(ts_tbl.data, duration_unit)
    elif duration_is_timedelta:
        duration_unit = UNIT_TIMEDELTA
        set_dur_var_unit(ts_tbl.data, UNIT_TIMEDELTA)
    else:
        raise DurationUnitError(
            f"load_win: duration column {dur_var!r} is numeric, so its unit "
            "cannot be inferred. Pass duration_unit='minutes' (or 'hours', "
            "'seconds', 'days')."
        )

    # Convert to win_tbl, carrying the id and index the series was built on.
    win_tbl = as_win_tbl(
        ts_tbl.data,
        id_vars=ts_tbl.id_vars,
        index_var=ts_tbl.index_var,
        dur_var=dur_var,
    )
    win_tbl.dur_unit = duration_unit
    set_dur_var_unit(win_tbl.data, duration_unit)

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
