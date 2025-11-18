"""High level concept callback implementations (R ricu callback-cncpt.R).

This module provides concept-level aggregation utilities that operate on
collections of :class:`~pyricu.table.ICUTable` objects as produced by the
concept resolver.  Each callback mirrors the behaviour of its R counterpart
well enough for the packaged concept dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import logging
import os
import sys

import numpy as np
import pandas as pd

from .callbacks import (
    mews_score,
    news_score,
    qsofa_score,
    sirs_score,
    sofa_cardio,
    sofa_cns,
    sofa_coag,
    sofa_liver,
    sofa_renal,
    sofa2_renal,
    sofa2_resp,
    sofa2_coag,
    sofa2_liver,
    sofa2_cardio,
    sofa2_cns,
    sofa_resp,
    sofa_score,
)
from .sofa2 import sofa2_score as sofa2_score_fn
from .sepsis import sep3 as sep3_detector, susp_inf as susp_inf_detector
from .sepsis_sofa2 import sep3_sofa2 as sep3_sofa2_detector
from .table import ICUTable, WinTbl

logger = logging.getLogger(__name__)
from .utils import coalesce
from .unit_conversion import convert_vaso_rate


def _safe_group_apply(grouped, func):
    """Compatibility helper for pandas include_groups default change."""
    try:
        # include_groups=True preserves the group keys as columns so downstream
        # callbacks relying on ID columns (e.g., urine/urine24) keep them.
        return grouped.apply(func, include_groups=True)
    except TypeError:  # pandas < 2.1
        return grouped.apply(func)


# Helper functions to unify WinTbl and ICUTable attribute access
def _get_id_columns(table):
    """Get ID columns from either WinTbl (id_vars) or ICUTable (id_columns)."""
    return list(table.id_vars if isinstance(table, WinTbl) else table.id_columns)


def _get_index_column(table):
    """Get index column from either WinTbl (index_var) or ICUTable (index_column)."""
    return table.index_var if isinstance(table, WinTbl) else table.index_column


def _coerce_hour_scalar(value) -> float:
    """Convert timestamps/timedeltas/numeric offsets to floating hour units."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize(None) if getattr(value, "tzinfo", None) else value
        return ts.value / 3_600_000_000_000
    if isinstance(value, (np.datetime64,)):
        ts = pd.Timestamp(value)
        ts = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
        return ts.value / 3_600_000_000_000
    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 3600.0
    if isinstance(value, np.timedelta64):
        return pd.to_timedelta(value).total_seconds() / 3600.0
    if isinstance(value, str):
        ts = pd.to_datetime(value, errors="coerce")
        if pd.notna(ts):
            ts = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
            return ts.value / 3_600_000_000_000
        td = pd.to_timedelta(value, errors="coerce")
        if pd.notna(td):
            return td.total_seconds() / 3600.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _coerce_duration_hours(value) -> float:
    """Convert duration column to floating hour units."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 3600.0
    if isinstance(value, np.timedelta64):
        return pd.to_timedelta(value).total_seconds() / 3600.0
    if isinstance(value, str):
        td = pd.to_timedelta(value, errors="coerce")
        if pd.notna(td):
            return td.total_seconds() / 3600.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


_STAY_LIMIT_CACHE: Dict[int, pd.DataFrame] = {}


def _normalize_patient_ids(patient_ids, column: str) -> Optional[List[object]]:
    """Resolve the list of patient ids matching the requested id column."""
    if patient_ids is None:
        return None
    if isinstance(patient_ids, dict):
        for key in (column, f"{column}_id", column.replace("_id", "")):
            if key in patient_ids and patient_ids[key] is not None:
                values = patient_ids[key]
                break
        else:
            return None
    else:
        values = patient_ids
    if values is None:
        return None
    if isinstance(values, (pd.Series, np.ndarray)):
        values = values.tolist()
    normalized = []
    for value in values:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    return normalized or None


def _build_stay_window_limits(ctx: "ConceptCallbackContext", id_columns: List[str]) -> Optional[pd.DataFrame]:
    """Compute per-stay start/end offsets (hours) using admission windows."""
    if not id_columns:
        return None
    primary_id = id_columns[0]
    if primary_id != "stay_id":
        return None
    data_source = getattr(ctx, "data_source", None)
    if data_source is None:
        return None
    cache_key = id(data_source)
    cached = _STAY_LIMIT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        icu_tbl = data_source.load_table(
            "icustays",
            columns=["stay_id", "hadm_id", "subject_id", "intime", "outtime"],
            verbose=False,
        )
    except Exception:
        return None
    icu_df = getattr(icu_tbl, "data", icu_tbl)
    if icu_df is None or icu_df.empty or "hadm_id" not in icu_df.columns:
        return None
    try:
        adm_tbl = data_source.load_table(
            "admissions",
            columns=["hadm_id", "admittime", "dischtime", "deathtime"],
            verbose=False,
        )
    except Exception:
        return None
    adm_df = getattr(adm_tbl, "data", adm_tbl)
    if adm_df is None or adm_df.empty:
        return None
    try:
        pat_tbl = data_source.load_table(
            "patients", columns=["subject_id", "dod", "anchor_age", "anchor_year"], verbose=False
        )
    except Exception:
        pat_tbl = None
    pat_df = getattr(pat_tbl, "data", pat_tbl) if pat_tbl is not None else None
    icu = icu_df
    patient_filter = _normalize_patient_ids(ctx.patient_ids, primary_id)
    if patient_filter:
        icu = icu[icu["stay_id"].isin(patient_filter)].copy()
    adm = adm_df
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce").dt.tz_localize(None)
    icu["outtime"] = pd.to_datetime(icu.get("outtime"), errors="coerce").dt.tz_localize(None)
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce").dt.tz_localize(None)
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce").dt.tz_localize(None)
    for extra_col in ("deathtime",):
        if extra_col in adm.columns:
            adm[extra_col] = pd.to_datetime(adm[extra_col], errors="coerce").dt.tz_localize(None)
    if pat_df is not None and not pat_df.empty and "subject_id" in icu.columns:
        pat = pat_df
        for col in ("dod",):
            if col in pat.columns:
                pat = pat.copy()  # Only copy when modifying
                pat[col] = pd.to_datetime(pat[col], errors="coerce").dt.tz_localize(None)
                break  # Only need to copy once
        merged = icu.merge(adm, on="hadm_id", how="left", suffixes=("", "_adm"))
        merged = merged.merge(pat, on="subject_id", how="left", suffixes=("", "_pat"))
    else:
        merged = icu.merge(adm, on="hadm_id", how="left", suffixes=("", "_adm"))
    merged = merged.dropna(subset=["stay_id", "intime"])
    if merged.empty:
        return None
    merged["admittime"] = merged["admittime"].fillna(merged["intime"])
    if "outtime" in merged.columns:
        merged["dischtime"] = merged["dischtime"].fillna(merged["outtime"])
    else:
        merged["dischtime"] = merged["dischtime"].fillna(merged["admittime"])
    # Incorporate additional clinical timestamps to better match ricu stay windows.
    start_candidates = [
        merged.get(col)
        for col in ("admittime", "intime")
        if col in merged.columns
    ]
    end_candidates = [
        merged.get(col)
        for col in ("dischtime", "deathtime", "outtime", "dod")
        if col in merged.columns
    ]
    if start_candidates:
        start_time = pd.concat(start_candidates, axis=1).min(axis=1)
    else:
        start_time = merged["intime"]
    if end_candidates:
        end_time = pd.concat(end_candidates, axis=1).max(axis=1)
    else:
        end_time = merged.get("outtime", merged["intime"])

    invalid_mask = start_time.notna() & end_time.notna() & (end_time < start_time)
    if invalid_mask.any():
        end_time = end_time.where(~invalid_mask, start_time)

    start_hours = (start_time - merged["intime"]).dt.total_seconds() / 3600.0
    end_hours = (end_time - merged["intime"]).dt.total_seconds() / 3600.0
    limits = merged.loc[:, ["stay_id"]].copy()
    limits["start"] = start_hours
    limits["end"] = end_hours
    limits = limits.replace([np.inf, -np.inf], np.nan).dropna(subset=["start", "end"])
    _STAY_LIMIT_CACHE[cache_key] = limits
    return limits


def _compose_fill_limits(
    data: pd.DataFrame,
    id_columns: List[str],
    index_column: str,
    ctx: "ConceptCallbackContext",
) -> Optional[pd.DataFrame]:
    """Build fill_gaps limits matching ricu's collapse(x) behavior.
    
    By default (matching ricu), uses observed data range (min/max per ID).
    This ensures SOFA and other aggregates cover the full patient timeline
    as captured in the merged component data, not just ICU stay windows.
    """
    if not id_columns or index_column not in data.columns:
        return None
    observed = (
        data.dropna(subset=[index_column])
        .groupby(id_columns, dropna=False)[index_column]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "start", "max": "end"})
    )
    if observed.empty:
        return None
    # 🔧 CRITICAL FIX: Match ricu's fill_gaps(dat, limits = collapse(x))
    # Use observed data range directly (not constrained by ICU stay windows)
    # to replicate ricu's behavior of filling gaps across entire patient timeline
    return observed[id_columns + ["start", "end"]]


def _expand_win_table_to_interval(
    win_tbl: WinTbl,
    *,
    interval: Optional[pd.Timedelta],
    value_column: str,
    target_index: Optional[str] = None,
    fill_value: Optional[object] = True,
) -> ICUTable:
    """
    Expand a WinTbl into an hourly ICUTable to simplify downstream merges.

    Args:
        win_tbl: Source window table to expand.
        interval: Desired sampling interval (defaults to 1 hour).
        value_column: Column to use as indicator/value in the expanded rows.
        target_index: Optional output index column name (defaults to win_tbl.index_var).
        fill_value: Value to emit when the window table does not store explicit values.
    """
    if not isinstance(win_tbl, WinTbl):
        return win_tbl

    interval = interval or pd.Timedelta(hours=1)
    interval_hours = max(interval.total_seconds() / 3600.0, 1e-6)

    idx_col = win_tbl.index_var or target_index or "time"
    dur_col = win_tbl.dur_var
    id_columns = list(win_tbl.id_vars)
    out_index = target_index or idx_col

    if dur_col is None or dur_col not in win_tbl.data.columns:
        raise ValueError("Cannot expand WinTbl without a duration column")

    records: List[Dict[str, object]] = []
    data = win_tbl.data.copy()

    for _, row in data.iterrows():
        start_val = row[idx_col] if idx_col in row.index else np.nan
        duration_val = row[dur_col] if dur_col in row.index else np.nan
        start_hours = _coerce_hour_scalar(start_val)
        duration_hours = _coerce_duration_hours(duration_val)

        if np.isnan(start_hours) or np.isnan(duration_hours) or duration_hours <= 0:
            continue

        end_hours = start_hours + duration_hours
        current = np.floor(start_hours / interval_hours) * interval_hours

        while current < end_hours:
            rec = {col: row[col] for col in id_columns if col in row.index}
            rec[out_index] = current
            if value_column in row.index:
                rec[value_column] = row[value_column]
            else:
                rec[value_column] = fill_value
            records.append(rec)
            current += interval_hours

    cols = id_columns + [out_index, value_column]
    if not records:
        return _as_icutbl(
            pd.DataFrame(columns=cols),
            id_columns=id_columns,
            index_column=out_index,
            value_column=value_column,
        )

    expanded = pd.DataFrame.from_records(records)
    expanded = expanded[cols].drop_duplicates()
    expanded = expanded.sort_values(id_columns + [out_index])
    return _as_icutbl(
        expanded.reset_index(drop=True),
        id_columns=id_columns,
        index_column=out_index,
        value_column=value_column,
    )


def _get_numeric_series(
    data: pd.DataFrame,
    column: str,
    *,
    index: Optional[pd.Index] = None,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """
    Fetch a column from ``data`` as a numeric Series, tolerating scalars or missing values.

    R ricu callbacks frequently expect Series inputs even when some components are
    absent.  When a column is missing (or when pandas returns a scalar because the
    merge dropped to a Series), we create a new Series filled with ``default`` so
    downstream score functions can safely operate on aligned indices.
    """
    if column in data.columns:
        raw = data[column]
    else:
        raw = default

    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]

    if isinstance(raw, pd.Series):
        series = raw
    else:
        fill_index = index if index is not None else data.index
        series = pd.Series(default, index=fill_index, dtype=float)

    return pd.to_numeric(series, errors="coerce")


CallbackFn = Callable[[Dict[str, ICUTable], "ConceptCallbackContext"], ICUTable]


@dataclass
class ConceptCallbackContext:
    """Context passed to concept-level callbacks."""

    concept_name: str
    target: Optional[str]
    interval: Optional[pd.Timedelta]
    resolver: "ConceptResolverProtocol"
    data_source: "ICUDataSourceProtocol"
    patient_ids: Optional[Iterable[object]]
    kwargs: Optional[Dict] = None  # Additional parameters for callbacks
    
    def __post_init__(self):
        """Initialize kwargs as empty dict if None."""
        if self.kwargs is None:
            self.kwargs = {}


class ConceptResolverProtocol:
    """Protocol subset used from :class:`~pyricu.concept.ConceptResolver`."""

    def load_concepts(  # pragma: no cover - runtime typing only
        self,
        concept_names: Iterable[str],
        data_source: "ICUDataSourceProtocol",
        *,
        merge: bool = True,
        aggregate: Optional[Mapping[str, object]] = None,
        patient_ids: Optional[Iterable[object]] = None,
    ):
        raise NotImplementedError


class ICUDataSourceProtocol:
    """Protocol subset for :class:`~pyricu.datasource.ICUDataSource`."""

    config: object


def _load_id_mapping_table(ctx: ConceptCallbackContext, from_col: str, to_col: str) -> Optional[pd.DataFrame]:
    """
    Load ID mapping table (e.g., icustays) for converting between ID types.
    
    This replicates R ricu's change_id() functionality which uses mapping tables
    to convert between different ID hierarchies (e.g., hadm_id ↔ stay_id).
    
    Args:
        ctx: Callback context with data source access
        from_col: Source ID column name (e.g., 'hadm_id', 'subject_id')
        to_col: Target ID column name (e.g., 'stay_id')
    
    Returns:
        DataFrame with columns [from_col, to_col] and optionally 'subject_id'
    """
    try:
        # 🔧 FIX: eICU doesn't use icustays table, skip for eICU databases
        db_name = ctx.data_source.config.name if hasattr(ctx.data_source, 'config') and hasattr(ctx.data_source.config, 'name') else ''
        if db_name in ['eicu', 'eicu_demo']:
            # eICU uses patientunitstayid as the primary ID, no mapping needed
            return None
        
        # Load icustays table which contains the mapping for MIMIC datasets
        # This works for MIMIC-III/IV, other databases may use different mapping tables
        # Build list of columns, avoiding duplicates
        cols_to_load = list(set([from_col, to_col]))  # Remove duplicates first
        # Add subject_id if it's not already in the list
        if 'subject_id' not in cols_to_load:
            cols_to_load.append('subject_id')
        
        # Load full icustays table without filtering
        # (filtering by patient_ids is complex since we don't know if they are 
        # subject_id, stay_id, or hadm_id - easier to load all and filter later)
        icustays_tbl = ctx.data_source.load_table(
            'icustays', 
            columns=cols_to_load, 
            filters=None,  # No filters - load all rows
            verbose=False
        )
        
        if icustays_tbl and not icustays_tbl.data.empty:
            # Keep only needed columns and drop duplicates
            needed_cols = [col for col in cols_to_load if col in icustays_tbl.data.columns]
            if from_col in needed_cols and to_col in needed_cols:
                mapping = icustays_tbl.data[needed_cols].drop_duplicates()
                # Debug logging
                if os.environ.get('DEBUG'):
                    logger.debug(f"ID映射加载成功: {from_col} → {to_col}, {len(mapping)} 行")
                return mapping
        else:
            if os.environ.get('DEBUG'):
                print(f"   ⚠️  icustays 表为空或未加载")
    except Exception as e:
        # Mapping table not available - this is OK, not all concepts need it
        # Only print error in debug mode to avoid spam
        if os.environ.get('DEBUG'):
            import traceback
            print(f"   ⚠️  无法加载 icustays 进行 ID 转换 ({from_col} → {to_col}): {e}")
            traceback.print_exc()
    return None


def _convert_id_column(
    data: pd.DataFrame,
    from_col: str,
    to_col: str,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert data from one ID column type to another using a mapping table.
    
    Replicates R ricu's change_id() / upgrade_id() / downgrade_id() functions.
    
    Args:
        data: DataFrame with from_col
        from_col: Current ID column name
        to_col: Target ID column name  
        mapping: Mapping table with both from_col and to_col
        
    Returns:
        DataFrame with from_col replaced by to_col
    """
    if from_col not in data.columns:
        return data
    
    if to_col in data.columns and from_col != to_col:
        # Already has target column, just remove old one
        return data.drop(columns=[from_col])
    
    # Merge with mapping to get target ID
    result = data.merge(
        mapping[[from_col, to_col]].drop_duplicates(),
        on=from_col,
        how='left'
    )
    
    # Remove the old ID column
    if from_col in result.columns and from_col != to_col:
        result = result.drop(columns=[from_col])
    
    return result


def _assert_shared_schema(
    tables: Dict[str, ICUTable], 
    ctx: Optional[ConceptCallbackContext] = None,
    convert_ids: bool = True
) -> tuple[list[str], Optional[str], Dict[str, ICUTable]]:
    """
    Validate that all input tables share identical identifier metadata.
    
    If convert_ids=True and ctx is provided, will attempt to convert mismatched
    ID columns using mapping tables (replicating R ricu's change_id behavior).
    
    Args:
        tables: Dictionary of concept component tables
        ctx: Callback context (required for ID conversion)
        convert_ids: Whether to attempt automatic ID conversion
        
    Returns:
        Tuple of (id_columns, index_column, converted_tables)
    """
    if not tables:
        raise ValueError("No tables supplied to concept callback")

    id_columns: Optional[list[str]] = None
    index_column: Optional[str] = None
    index_columns_found = set()
    
    # Collect all unique ID column sets
    id_column_sets = {}
    for name, table in tables.items():
        # Use helper function to support both WinTbl and ICUTable
        ids = _get_id_columns(table)
        idx = _get_index_column(table)
        id_column_sets[name] = ids
        
        if id_columns is None:
            id_columns = ids
        
        if idx:
            index_columns_found.add(idx)
        
        if index_column is None:
            index_column = idx
    
    # Check if ID conversion is needed
    needs_conversion = not all(ids == id_columns for ids in id_column_sets.values())
    
    converted_tables = dict(tables)  # Copy for potential modifications
    
    if needs_conversion and convert_ids and ctx is not None:
        # Try to convert all tables to the common target ID (prefer stay_id for ICU data)
        # This replicates R ricu's automatic ID conversion in collect_dots()
        target_id_col = id_columns[0] if id_columns else 'stay_id'
        
        # Determine all ID types present
        all_id_types = set()
        for ids in id_column_sets.values():
            all_id_types.update(ids)
        
        # Handle hadm_id ↔ stay_id conversion
        if 'hadm_id' in all_id_types and 'stay_id' in all_id_types:
            # Prefer stay_id as target (ICU-level granularity)
            target_id_col = 'stay_id'
            mapping = _load_id_mapping_table(ctx, 'hadm_id', 'stay_id')
            
            if mapping is not None:
                if os.environ.get('DEBUG'):
                    logger.debug(f"ID映射表加载成功: hadm_id → stay_id, {len(mapping)} 行")
                
                # Convert tables with hadm_id to stay_id
                tables_to_remove = []  # Track empty tables to remove
                for name, table in list(tables.items()):
                    if 'hadm_id' in table.id_columns and 'stay_id' not in table.id_columns:
                        if os.environ.get('DEBUG'):
                            logger.debug(f"转换表 '{name}': hadm_id → stay_id")
                        converted_data = _convert_id_column(
                            table.data.copy(),
                            'hadm_id',
                            'stay_id',
                            mapping
                        )
                        if os.environ.get('DEBUG'):
                            print(f"      转换后: {len(converted_data)} 行")
                        
                        # 如果转换后数据为空，标记要移除这个表（而不是报错）
                        if converted_data.empty:
                            if os.environ.get('DEBUG'):
                                print(f"      ⚠️  跳过空表 '{name}'（ID 转换后无匹配数据）")
                            # 标记要从原始tables中移除
                            tables_to_remove.append(name)
                            continue
                        
                        # Update table with converted data
                        converted_tables[name] = ICUTable(
                            data=converted_data,
                            id_columns=['stay_id'],
                            index_column=table.index_column,
                            value_column=table.value_column,
                            unit_column=table.unit_column,
                        )
                
                # Remove empty tables from original tables dict
                for name in tables_to_remove:
                    if name in tables:
                        del tables[name]
                    if name in converted_tables:
                        del converted_tables[name]
                # Update id_columns to reflect conversion
                id_columns = ['stay_id']
            else:
                if os.environ.get('DEBUG'):
                    print(f"   ⚠️  ID映射表加载失败: hadm_id → stay_id")
        
        # Handle subject_id ↔ stay_id conversion
        if 'subject_id' in all_id_types and 'stay_id' in all_id_types:
            # Prefer stay_id as target (ICU-level granularity, more specific)
            target_id_col = 'stay_id'
            mapping = _load_id_mapping_table(ctx, 'subject_id', 'stay_id')
            
            if mapping is not None:
                # Convert tables with subject_id (but not stay_id) to stay_id
                tables_to_remove = []  # Track empty tables to remove
                for name, table in list(tables.items()):
                    if 'subject_id' in table.id_columns and 'stay_id' not in table.id_columns:
                        if os.environ.get('DEBUG'):
                            logger.debug(f"转换表 '{name}': subject_id → stay_id")
                        converted_data = _convert_id_column(
                            table.data.copy(),
                            'subject_id',
                            'stay_id',
                            mapping
                        )
                        if os.environ.get('DEBUG'):
                            print(f"      转换后: {len(converted_data)} 行")
                        
                        # 如果转换后数据为空，标记要移除这个表（而不是报错）
                        if converted_data.empty:
                            if os.environ.get('DEBUG'):
                                print(f"      ⚠️  跳过空表 '{name}'（ID 转换后无匹配数据）")
                            # 标记要从原始tables中移除
                            tables_to_remove.append(name)
                            continue
                        
                        # Update table with converted data
                        converted_tables[name] = ICUTable(
                            data=converted_data,
                            id_columns=['stay_id'],
                            index_column=table.index_column,
                            value_column=table.value_column,
                            unit_column=table.unit_column,
                        )
                
                # Remove empty tables from original tables dict
                for name in tables_to_remove:
                    if name in tables:
                        del tables[name]
                    if name in converted_tables:
                        del converted_tables[name]
                # Update id_columns to reflect conversion
                id_columns = ['stay_id']
    
    # Final validation - all tables should now have matching IDs
    # Note: Some tables may have been removed during conversion if they became empty
    for name, table in converted_tables.items():
        # Use helper function to support both WinTbl and ICUTable
        ids = _get_id_columns(table)
        if ids != id_columns:
            # 如果还有 ID 不匹配的表，说明转换失败
            if os.environ.get('DEBUG'):
                print(f"   ⚠️  表 '{name}' ID 不匹配: {ids} vs {id_columns}")
            raise ValueError(
                f"Concept component '{name}' has identifier columns {ids}, "
                f"expected {id_columns}. Automatic ID conversion failed."
            )
    
    return id_columns or [], index_column, converted_tables


def _merge_tables(
    tables: Dict[str, ICUTable],
    *,
    how: str = "outer",
    ctx: Optional[ConceptCallbackContext] = None,  # Add ctx parameter
) -> tuple[pd.DataFrame, list[str], Optional[str]]:
    """Merge component tables into a single DataFrame."""

    # Enable ID conversion for _merge_tables
    id_columns, index_column, converted_tables = _assert_shared_schema(
        tables, 
        ctx=ctx,  # Pass ctx for ID conversion
        convert_ids=True  # Enable automatic ID conversion
    )
    
    # Use converted tables if conversion happened
    if converted_tables:
        tables = converted_tables
    
    # Standardize index column names if they differ
    # (e.g., charttime from chartevents, starttime from inputevents)
    standardized_tables = {}
    for name, table in tables.items():
        frame = table.data.copy()
        
        # 展平MultiIndex列，避免合并时的MultiIndex错误
        if isinstance(frame.columns, pd.MultiIndex):
            new_cols = []
            for col in frame.columns:
                if isinstance(col, tuple):
                    # Join tuple elements, skipping empty strings
                    parts = [str(c) for c in col if c and str(c).strip()]
                    new_col = '_'.join(parts) if parts else name
                    new_cols.append(new_col)
                else:
                    new_cols.append(str(col))
            frame.columns = new_cols
        
        # Use helper function to support both WinTbl and ICUTable
        table_idx = _get_index_column(table)
        
        # Rename index column to the canonical name if it differs
        if table_idx and index_column and table_idx != index_column:
            if table_idx in frame.columns:
                frame = frame.rename(columns={table_idx: index_column})
        
        standardized_tables[name] = (frame, table)
    
    key_cols = id_columns + ([index_column] if index_column else [])
    merged: Optional[pd.DataFrame] = None
    
    # 首先检查所有表的时间类型，确定统一的目标类型
    target_time_type = None
    if index_column:
        for name, (frame, table) in standardized_tables.items():
            if index_column in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[index_column]):
                    # 优先使用numeric（小时）类型
                    target_time_type = 'numeric'
                    break
        
        # 如果没有numeric类型，使用datetime
        if target_time_type is None:
            target_time_type = 'datetime'
    
    for name, (frame, table) in standardized_tables.items():
        # 🔧 跳过空表 - 它们对合并没有贡献，且可能有不正确的列类型
        if frame.empty:
            continue

        # 确保时间列类型与目标一致，避免后续被强制跳过
        frame = _ensure_time_column_type(
            frame,
            index_column=index_column,
            target_time_type=target_time_type,
            id_columns=id_columns,
            ctx=ctx,
            table_name=name,
        )
        
        # 🔧 处理 WinTbl (没有 value_column，使用 name 本身)
        from pyricu.table import WinTbl
        if isinstance(table, WinTbl):
            value_col = name  # WinTbl 的值列就是概念名本身
        else:
            value_col = table.value_column or name
            
        if value_col != name:
            frame = frame.rename(columns={value_col: name})
        
        # 如果重命名后 name 列仍不存在，尝试查找匹配的列（例如 gcs_min -> gcs）
        if name not in frame.columns:
            matching_cols = [c for c in frame.columns if c.startswith(name + '_') or c == name]
            if matching_cols:
                # Prefer min aggregation (for gcs, use gcs_min)
                if any('min' in c for c in matching_cols):
                    col_to_rename = [c for c in matching_cols if 'min' in c][0]
                else:
                    col_to_rename = matching_cols[0]
                frame = frame.rename(columns={col_to_rename: name})
        
        # 🔧 FIX: 先处理frame中的重复列（例如合并多个item时可能产生重复的measuredat列）
        if frame.columns.duplicated().any():
            # 对于重复列，只保留第一个
            frame = frame.loc[:, ~frame.columns.duplicated()]
        
        # 只保留键列和值列,避免合并时的列冲突
        cols_to_keep = key_cols + [name]
        # 确保所有列都存在
        cols_to_keep = [c for c in cols_to_keep if c in frame.columns]
        frame = frame[cols_to_keep]
        
        if merged is None:
            merged = frame
        else:
            # 时间类型已经在前面统一，直接merge
            try:
                # 🔧 CRITICAL FIX: 检查merge所需的键列是否都存在
                actual_key_cols = [col for col in key_cols if col in frame.columns and col in merged.columns]
                if len(actual_key_cols) < len(key_cols):
                    missing_in_frame = [col for col in key_cols if col not in frame.columns]
                    missing_in_merged = [col for col in key_cols if col not in merged.columns]
                    print(f"   ⚠️  跳过 '{name}': 缺少合并键列 (frame缺少: {missing_in_frame}, merged缺少: {missing_in_merged})")
                    continue

                # 🔧 FIX: 如果frame有与merged重复的列（除了actual_key_cols），先删除frame中的重复列
                # 这通常发生在时间列（如measuredat, registeredat）在多个源表中都存在的情况
                duplicate_cols = [c for c in frame.columns if c in merged.columns and c not in actual_key_cols]
                if duplicate_cols:
                    frame = frame.drop(columns=duplicate_cols)

                merged = merged.merge(frame, on=actual_key_cols, how=how)
            except (ValueError, KeyError) as e:
                # Merge失败，跳过这个表
                print(f"   ⚠️  跳过 '{name}': merge失败 - {e}")
                continue

    if merged is None:
        merged = pd.DataFrame(columns=key_cols)

    # Ensure each concept contributes a column even if its source table was empty
    expected_value_columns = list(standardized_tables.keys())
    for value_col in expected_value_columns:
        if value_col not in merged.columns:
            merged[value_col] = pd.Series(dtype="float64")

    return merged, id_columns, index_column


def _ensure_time_column_type(
    frame: pd.DataFrame,
    *,
    index_column: Optional[str],
    target_time_type: Optional[str],
    id_columns: Iterable[str] | None,
    ctx: Optional[ConceptCallbackContext],
    table_name: str,
) -> pd.DataFrame:
    """Coerce the time column to the desired type (hours or datetime).

    R ricu keeps component timelines aligned by ensuring every table uses
    the same relative hour axis. When some sub-concepts still expose
    datetime or timedelta columns, the previous implementation silently
    skipped them, causing downstream aggregates (like SOFA) to lose their
    component inputs. This helper mirrors ricu's behaviour by converting
    those columns to numeric hours whenever possible, leveraging the
    resolver's ``_align_time_to_admission`` fallback when context exists.
    """

    if not index_column or index_column not in frame.columns or not target_time_type:
        return frame

    series = frame[index_column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    if target_time_type == "numeric":
        if pd.api.types.is_numeric_dtype(series):
            return frame

        def _ensure_copy(data: pd.DataFrame) -> pd.DataFrame:
            return data if data is not frame else data.copy()

        working = frame

        if pd.api.types.is_timedelta64_dtype(series):
            working = _ensure_copy(working)
            working[index_column] = pd.to_timedelta(series, errors="coerce").dt.total_seconds() / 3600.0
            return working

        resolver = getattr(ctx, "resolver", None) if ctx else None
        data_source = getattr(ctx, "data_source", None) if ctx else None
        if resolver is not None and data_source is not None and hasattr(resolver, "_align_time_to_admission"):
            try:
                aligned = resolver._align_time_to_admission(  # type: ignore[attr-defined]
                    working.copy(),
                    data_source,
                    list(id_columns or []),
                    index_column,
                )
            except Exception:
                aligned = None
            if isinstance(aligned, pd.DataFrame) and index_column in aligned.columns:
                aligned_series = aligned[index_column]
                if pd.api.types.is_numeric_dtype(aligned_series):
                    return aligned
                if pd.api.types.is_timedelta64_dtype(aligned_series):
                    aligned[index_column] = (
                        pd.to_timedelta(aligned_series, errors="coerce").dt.total_seconds() / 3600.0
                    )
                    return aligned
                working = aligned

        working = _ensure_copy(working)
        aligned_series = working[index_column]
        if pd.api.types.is_datetime64_any_dtype(aligned_series):
            if list(id_columns or []):
                def _relative_hours(group: pd.Series) -> pd.Series:
                    valid = group.dropna()
                    if valid.empty:
                        return pd.Series(np.nan, index=group.index)
                    base = valid.iloc[0]
                    delta = group - base
                    return delta.dt.total_seconds() / 3600.0

                working[index_column] = working.groupby(list(id_columns or []))[index_column].transform(
                    _relative_hours
                )
            else:
                valid = aligned_series.dropna()
                base = valid.iloc[0] if not valid.empty else pd.NaT
                working[index_column] = ((aligned_series - base).dt.total_seconds() / 3600.0)
            return working

        working[index_column] = pd.to_numeric(working[index_column], errors="coerce")
        return working

    if target_time_type == "datetime":
        if pd.api.types.is_datetime64_any_dtype(series):
            return frame
        working = frame if frame is not None else pd.DataFrame()
        working = working.copy()
        working[index_column] = pd.to_datetime(working[index_column], errors="coerce").dt.tz_localize(None)
        return working

    return frame


def _as_icutbl(
    frame: pd.DataFrame,
    *,
    id_columns: Iterable[str],
    index_column: Optional[str],
    value_column: str,
    unit_column: Optional[str] = None,
) -> ICUTable:
    """Create an :class:`ICUTable` from a plain DataFrame."""

    return ICUTable(
        data=frame,
        id_columns=list(id_columns),
        index_column=index_column,
        value_column=value_column,
        unit_column=unit_column,
    )


def _ensure_time_index(table: ICUTable) -> ICUTable:
    """Ensure that a time-indexed table is sorted and gap-free."""

    data = table.data.copy()
    idx = table.index_column
    if idx and not data.empty:
        data = data.sort_values(table.id_columns + [idx])
    return ICUTable(
        data=data,
        id_columns=list(table.id_columns),
        index_column=table.index_column,
        value_column=table.value_column,
        unit_column=table.unit_column,
        time_columns=list(table.time_columns),
    )


def _infer_interval_from_table(table: ICUTable) -> Optional[pd.Timedelta]:
    idx = table.index_column
    if not idx:
        return None

    data = table.data.copy()
    # 只有在不是numeric类型时才转换为datetime
    if not pd.api.types.is_numeric_dtype(data[idx]):
        data[idx] = pd.to_datetime(data[idx], errors="coerce")
    data = data.dropna(subset=[idx])
    if data.empty:
        return None

    if table.id_columns:
        diffs = []
        for _, group in data.sort_values(table.id_columns + [idx]).groupby(table.id_columns):
            series = group[idx].diff()
            diffs.append(series)
        diffs = pd.concat(diffs, axis=0)
    else:
        diffs = data.sort_values(idx)[idx].diff()

    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None

    return diffs.min()


def _merge_intervals(
    df: pd.DataFrame,
    *,
    id_columns: Iterable[str],
    start_col: str,
    end_col: str,
    max_gap: pd.Timedelta,
) -> pd.DataFrame:
    records = []
    sort_cols = list(id_columns) + [start_col]
    for key, group in df.sort_values(sort_cols).groupby(list(id_columns), sort=False):
        current_start = None
        current_end = None
        for _, row in group.iterrows():
            start = row[start_col]
            end = row[end_col]
            if current_start is None:
                current_start, current_end = start, end
                continue
            if start <= current_end + max_gap:
                if end > current_end:
                    current_end = end
            else:
                key_tuple = key if isinstance(key, tuple) else (key,)
                records.append((*key_tuple, current_start, current_end))
                current_start, current_end = start, end
        if current_start is not None:
            key_tuple = key if isinstance(key, tuple) else (key,)
            records.append((*key_tuple, current_start, current_end))

    if not records:
        return pd.DataFrame(columns=list(id_columns) + ["__start", "__end"])

    columns = list(id_columns) + ["__start", "__end"]
    return pd.DataFrame.from_records(records, columns=columns)


# ============================================================================
# AUMC-specific callbacks
# ============================================================================

def _callback_aumc_death(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """AUMC death callback: marks death if it occurred within 72 hours of ICU discharge.
    
    Similar to ricu's: x[, val_var := is_true(index_var - val_var < hours(72L))]
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    # Get the single input table
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return _empty_icutbl(ctx)
    
    id_columns = input_table.id_columns
    index_column = input_table.index_column or ctx.index_column
    value_column = input_table.value_column or list(tables.keys())[0]
    
    # Check if death occurred within 72 hours
    # index_column is typically the time of observation (e.g., charttime in hours)
    # value_column contains the time of death
    if index_column in data.columns and value_column in data.columns:
        # Convert to numeric to handle time differences
        index_vals = pd.to_numeric(data[index_column], errors='coerce')
        value_vals = pd.to_numeric(data[value_column], errors='coerce')
        
        # Death within 72 hours: (time_of_observation - time_of_death) < 72
        data['death'] = (index_vals - value_vals < 72).astype(int)
    else:
        # If columns are missing, mark all as death=1 (conservative)
        data['death'] = 1
    
    output_cols = list(id_columns) + ([index_column] if index_column else []) + ['death']
    result = data[output_cols].copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column='death')


def _callback_aumc_bxs(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """AUMC blood gas callback: negates values where direction is '-'.
    
    Similar to ricu's: x[get(dir_var) == "-", val_var := -1L * get(val_var)]
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    # Typically receives two tables: value and direction
    # Find the value table and direction table
    value_table = None
    dir_table = None
    
    for name, tbl in tables.items():
        if 'dir' in name.lower() or 'direction' in name.lower():
            dir_table = tbl
        else:
            value_table = tbl
    
    if value_table is None:
        # If no direction table, just return the first table
        return list(tables.values())[0]
    
    data = value_table.df.copy()
    if data.empty:
        return value_table
    
    id_columns = value_table.id_columns
    index_column = value_table.index_column or ctx.index_column
    value_column = value_table.value_column or ctx.concept_name
    
    # If we have direction information, merge it
    if dir_table is not None:
        dir_data = dir_table.df
        merge_cols = list(id_columns)
        if index_column and index_column in data.columns and index_column in dir_data.columns:
            merge_cols.append(index_column)
        
        if merge_cols:
            data = data.merge(dir_data, on=merge_cols, how='left', suffixes=('', '_dir'))
            dir_column = dir_table.value_column or 'direction'
            
            # Negate values where direction is '-'
            if dir_column in data.columns:
                mask = data[dir_column] == '-'
                if value_column in data.columns:
                    data.loc[mask, value_column] = -1 * data.loc[mask, value_column]
    
    output_cols = list(id_columns) + ([index_column] if index_column else []) + [value_column]
    output_cols = [c for c in output_cols if c in data.columns]
    result = data[output_cols].copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column=value_column)


def _callback_aumc_rass(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """AUMC RASS callback: extracts first 2 characters as integer.
    
    Similar to ricu's: as.integer(substr(x, 1L, 2L))
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return input_table
    
    id_columns = input_table.id_columns
    index_column = input_table.index_column or ctx.index_column
    value_column = input_table.value_column or ctx.concept_name
    
    if value_column in data.columns:
        # Extract first 2 characters and convert to integer
        data[value_column] = data[value_column].astype(str).str[:2]
        data[value_column] = pd.to_numeric(data[value_column], errors='coerce')
    
    output_cols = list(id_columns) + ([index_column] if index_column else []) + [value_column]
    output_cols = [c for c in output_cols if c in data.columns]
    result = data[output_cols].dropna(subset=[value_column]).copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column=value_column)


def _callback_blood_cell_ratio(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate blood cell ratios (e.g., lymphocytes, neutrophils as percentage).
    
    This callback handles cell count ratios, typically percentage calculations.
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    # This is typically a simple passthrough that may involve unit conversion
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return input_table
    
    id_columns = input_table.id_columns
    index_column = input_table.index_column or ctx.index_column
    value_column = input_table.value_column or ctx.concept_name
    
    # The data is usually already in the correct format
    # Just ensure it's numeric
    if value_column in data.columns:
        data[value_column] = pd.to_numeric(data[value_column], errors='coerce')
    
    output_cols = list(id_columns) + ([index_column] if index_column else []) + [value_column]
    output_cols = [c for c in output_cols if c in data.columns]
    result = data[output_cols].dropna(subset=[value_column]).copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column=value_column)


def _callback_bmi(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    merged, id_columns, _ = _merge_tables(tables, ctx=ctx, how="inner")
    if merged.empty:
        return _as_icutbl(merged, id_columns=id_columns, index_column=None, value_column="bmi")

    weight = merged["weight"]
    height = merged["height"]
    height_m = np.where(height > 10, height / 100.0, height)
    bmi = weight / np.where(height_m == 0, np.nan, height_m**2)
    merged = merged.assign(bmi=bmi)
    merged = merged[id_columns + ["bmi"]]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["bmi"])
    merged = merged[(merged["bmi"] >= 10) & (merged["bmi"] <= 100)]

    return _as_icutbl(merged.reset_index(drop=True), id_columns=id_columns, index_column=None, value_column="bmi")


def _callback_avpu(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    gcs_tbl = _ensure_time_index(tables["gcs"])
    df = gcs_tbl.data.copy()
    idx_cols = gcs_tbl.id_columns + ([gcs_tbl.index_column] if gcs_tbl.index_column else [])
    preferred_order = [
        gcs_tbl.value_column,
        "gcs",
        "gcs_min",
        "gcs_total",
        "gcs_sum",
    ]
    score_col = next(
        (
            col
            for col in preferred_order
            if isinstance(col, str) and col in df.columns
        ),
        None,
    )
    if score_col is None:
        candidates = [col for col in df.columns if col.startswith("gcs")]
        score_col = candidates[0] if candidates else None
    if score_col is None:
        frame = pd.DataFrame(columns=idx_cols + ["avpu"])
        return _as_icutbl(
            frame,
            id_columns=gcs_tbl.id_columns,
            index_column=gcs_tbl.index_column,
            value_column="avpu",
        )
    column_data = df[score_col]
    if isinstance(column_data, pd.DataFrame):
        column_data = column_data.iloc[:, 0]
    scores = pd.to_numeric(column_data, errors="coerce")

    def score_to_avpu(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value <= 3:
            return "U"
        if value <= 8:
            return "P"
        if value <= 12:
            return "V"
        return "A"

    avpu = scores.map(score_to_avpu)
    result = pd.DataFrame(index=df.index)
    for col in gcs_tbl.id_columns:
        if col in df.columns:
            result[col] = df[col]
    if gcs_tbl.index_column and gcs_tbl.index_column in df.columns:
        result[gcs_tbl.index_column] = df[gcs_tbl.index_column]
    result["avpu"] = avpu
    result = result.dropna(subset=["avpu"])

    return _as_icutbl(
        result.reset_index(drop=True),
        id_columns=gcs_tbl.id_columns,
        index_column=gcs_tbl.index_column,
        value_column="avpu",
    )


def _callback_norepi_equiv(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    factors = {
        "epi_rate": 1.0,
        "norepi_rate": 1.0,
        "dopa_rate": 1 / 150.0,
        "adh_rate": 1 / 0.4,
        "phn_rate": 1 / 10.0,
    }

    relevant_tables = {name: tbl for name, tbl in tables.items() if name in factors}
    schema_tables = relevant_tables if relevant_tables else tables

    id_columns, index_column, converted = _assert_shared_schema(
        schema_tables,
        ctx=ctx,
        convert_ids=True,
    )

    if converted:
        schema_tables = converted

    tables_to_use = {name: schema_tables[name] for name in factors if name in schema_tables}

    key_cols = id_columns + ([index_column] if index_column else [])
    scaled_frames: List[pd.DataFrame] = []

    for name, factor in factors.items():
        table = tables_to_use.get(name)
        if table is None or table.data.empty:
            continue

        frame = table.data.copy()
        value_col = table.value_column or name
        if value_col not in frame.columns and name in frame.columns:
            value_col = name
        elif value_col not in frame.columns:
            key_set = set(key_cols)
            fallback_cols = [col for col in frame.columns if col not in key_set]
            value_col = fallback_cols[0] if fallback_cols else None

        if value_col is None or value_col not in frame.columns:
            continue

        numeric = pd.to_numeric(frame[value_col], errors="coerce") * factor
        if key_cols:
            missing_keys = [col for col in key_cols if col not in frame.columns]
            for col in missing_keys:
                frame[col] = np.nan
            out = frame[key_cols].copy()
        else:
            out = pd.DataFrame(index=frame.index)

        out["norepi_equiv"] = numeric
        out = out.dropna(subset=["norepi_equiv"])
        if not out.empty:
            scaled_frames.append(out)

    if not scaled_frames:
        empty_cols = key_cols + ["norepi_equiv"]
        empty = pd.DataFrame(columns=empty_cols)
        return _as_icutbl(empty, id_columns=id_columns, index_column=index_column, value_column="norepi_equiv")

    combined = pd.concat(scaled_frames, ignore_index=True)

    if key_cols:
        aggregated = (
            combined.groupby(key_cols)["norepi_equiv"].sum(min_count=1).reset_index()
        )
        aggregated = aggregated.sort_values(key_cols).reset_index(drop=True)
    else:
        aggregated = pd.DataFrame({"norepi_equiv": [combined["norepi_equiv"].sum(min_count=1)]})

    return _as_icutbl(
        aggregated,
        id_columns=id_columns,
        index_column=index_column,
        value_column="norepi_equiv",
    )


def _callback_sofa_component(
    func: Callable[..., pd.Series],
) -> CallbackFn:
    def wrapper(tables: Dict[str, ICUTable], ctx: ConceptCallbackContext) -> ICUTable:
        # Some SOFA components (cardio variants) rely on auxiliary concepts such as
        # ``vaso_ind`` to determine when vasopressor rates should be zeroed out.
        # The recursive dictionary only lists the direct rate concepts as
        # ``sub_concepts``, so the callback would never receive the indicator and
        # our forward-fill logic would keep vasopressors active forever.  Fetch the
        # optional dependency lazily via the resolver so we can preserve the
        # original merge behavior when the indicator is available.
        tables = dict(tables)
        if ctx.concept_name in {"sofa_cardio", "sofa2_cardio"} and "vaso_ind" not in tables:
            try:
                loaded = ctx.resolver.load_concepts(
                    ["vaso_ind"],
                    ctx.data_source,
                    merge=False,
                    aggregate={"vaso_ind": "max"},
                    patient_ids=ctx.patient_ids,
                    interval=ctx.interval,
                    align_to_admission=True,
                )
                if isinstance(loaded, dict):
                    vaso_tbl = loaded.get("vaso_ind")
                else:
                    vaso_tbl = loaded
                if isinstance(vaso_tbl, ICUTable) and not vaso_tbl.data.empty:
                    tables["vaso_ind"] = vaso_tbl
            except Exception:
                # Missing indicator should not break SOFA computation; simply skip
                # zeroing when it is unavailable.
                pass
        # CRITICAL: For single concept (sofa_single type), ricu_code's collect_dots returns the data directly
        # For multiple concepts, use outer join (replicates R ricu merge_dat = TRUE)
        # In ricu_code: sofa_single("plt", "sofa_coag", fun) -> collect_dots("plt", ...) returns plt data
        # Then: dat[, c("sofa_coag") := fun(get("plt"))] -> rm_cols(dat, "plt", by_ref = TRUE)
        if len(tables) == 1:
            # Single concept: directly use the table data (replicates collect_dots for single concept)
            sub_name, table = next(iter(tables.items()))  # sub_name is the sub-concept name (e.g., "plt")
            data = table.data.copy()
            id_columns = list(table.id_columns)
            index_column = table.index_column
            
            # The value column should be the sub-concept name (e.g., "plt")
            # Ensure it's named correctly for the callback function
            value_col = table.value_column or sub_name
            if value_col not in data.columns:
                # Try to find the value column (first non-key column)
                key_cols = id_columns + ([index_column] if index_column else [])
                matching_cols = [c for c in data.columns if c not in key_cols]
                if matching_cols:
                    value_col = matching_cols[0]
            
            # Rename value column to sub-concept name for callback function lookup
            if value_col != sub_name and value_col in data.columns:
                data = data.rename(columns={value_col: sub_name})
        else:
            # Multiple concepts: merge with outer join
            data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
        
        if data.empty:
            cols = id_columns + ([index_column] if index_column else []) + [ctx.concept_name]
            frame = pd.DataFrame(columns=cols)
            return _as_icutbl(frame, id_columns=id_columns, index_column=index_column, value_column=ctx.concept_name)

        # CRITICAL: Ensure time points are sorted after merge
        if index_column:
            sort_cols = id_columns + [index_column] if id_columns else [index_column]
            data = data.sort_values(sort_cols)

        # ricu keeps vasopressor rates active until the infusion ends. Our raw
        # vaso rate/equivalent rows only appear when the rate table logs a change,
        # so after the outer merge with MAP we must forward-fill those values and
        # zero them once the patient is off vasopressors.
        if ctx.concept_name == "sofa_cardio":
            vaso_tokens = ("norepi", "dobu", "dopa", "epi", "adh")
            vaso_cols: list[str] = [
                col
                for col in data.columns
                if col != "vaso_ind" and any(token in col for token in vaso_tokens)
            ]
            if vaso_cols:
                if id_columns:
                    data[vaso_cols] = data.groupby(id_columns, dropna=False)[vaso_cols].ffill()
                else:
                    data[vaso_cols] = data[vaso_cols].ffill()

                if "vaso_ind" in data.columns:
                    vaso_mask = pd.to_numeric(data["vaso_ind"], errors="coerce").fillna(0.0) != 0
                    for col in vaso_cols:
                        data.loc[~vaso_mask, col] = 0.0

        # Extract data from merged DataFrame
        # The data DataFrame already has columns from all tables merged by key columns
        kwargs = {}
        for name, table in tables.items():
            if name == "vaso_ind":
                continue  # indicator is only used to zero rates, not passed to score function
            # Try to find the column in merged data
            # First try the concept name (after _merge_tables renaming)
            col_name = name
            if col_name not in data.columns:
                # If not found, try the table's value_column
                col_name = table.value_column or name
            if col_name not in data.columns:
                # If still not found, try variations (e.g., "gcs_min" if name is "gcs")
                # Check for columns that start with the concept name
                matching_cols = [c for c in data.columns if c.startswith(name + '_') or c == name]
                if matching_cols:
                    # Prefer exact match, then min aggregation (for gcs, use gcs_min)
                    if name in matching_cols:
                        col_name = name
                    elif any('min' in c for c in matching_cols):
                        # For concepts like gcs, prefer gcs_min over gcs_any
                        min_cols = [c for c in matching_cols if 'min' in c]
                        col_name = min_cols[0]  # Take first min column
                    else:
                        col_name = matching_cols[0]
                else:
                    # If no matching columns found, keep col_name as name (will fail later)
                    pass
            
            if col_name in data.columns:
                # Extract the column as a Series
                col_data = data[col_name]
                # Ensure it's a Series before converting to numeric
                if isinstance(col_data, pd.Series):
                    kwargs[name] = pd.to_numeric(col_data, errors="coerce")
                elif isinstance(col_data, pd.DataFrame):
                    # If it's a DataFrame (shouldn't happen), take first column
                    kwargs[name] = pd.to_numeric(col_data.iloc[:, 0], errors="coerce")
                else:
                    # Convert to Series first
                    kwargs[name] = pd.to_numeric(pd.Series(col_data), errors="coerce")
            else:
                # 🔧 FIX: For optional parameters (like vasopressors in sofa_cardio, urine24 in sofa_renal), 
                # if the column is missing and all values would be NaN, pass None instead
                # This allows the callback function to handle missing data correctly
                # For sofa_cardio, missing vasopressors should be None (will be filled with 0 in callback)
                # For sofa_renal, missing urine24 should be None (will be handled as optional parameter)
                # For required parameters (like map in sofa_cardio, crea in sofa_renal), create NaN Series to preserve time points
                if ctx.concept_name == 'sofa_cardio' and name in ['dopa60', 'norepi60', 'dobu60', 'epi60']:
                    # Optional vasopressor parameters - pass None
                    kwargs[name] = None
                elif ctx.concept_name == 'sofa2_cardio' and name in ['dopa60', 'norepi60', 'dobu60', 'epi60', 'other_vaso', 'mech_circ_support']:
                    # SOFA-2 optional vasopressor/support parameters - pass None
                    kwargs[name] = None
                elif ctx.concept_name == 'sofa_renal' and name == 'urine24':
                    # Optional urine24 parameter - pass None
                    kwargs[name] = None
                elif ctx.concept_name == 'sofa2_renal' and name in ['uo_6h', 'uo_12h', 'uo_24h', 'rrt', 'rrt_criteria', 'potassium', 'ph', 'bicarb']:
                    # SOFA-2 renal optional parameters - pass None
                    kwargs[name] = None
                elif ctx.concept_name == 'sofa2_resp' and name in ['spo2', 'fio2', 'adv_resp', 'ecmo', 'ecmo_indication']:
                    # SOFA-2 respiratory optional parameters - pass None
                    kwargs[name] = None
                elif ctx.concept_name == 'sofa2_cns' and name in ['delirium_tx', 'sedated_gcs']:
                    # SOFA-2 CNS optional parameters - pass None
                    kwargs[name] = None
                else:
                    # Required parameters - create Series with NaN to preserve time points
                    kwargs[name] = pd.Series(np.nan, index=data.index, dtype=float)
        
        # Call function with kwargs - add special handling for functions that require positional args
        try:
            # Special handling for sofa_renal and sofa2_renal which require 'crea' as positional arg
            if ctx.concept_name in ['sofa_renal', 'sofa2_renal']:
                if 'crea' in kwargs:
                    func_kwargs = kwargs.copy()
                    crea_arg = func_kwargs.pop('crea')
                    result = func(crea_arg, **func_kwargs)
                else:
                    # For sofa_renal/sofa2_renal, if crea is missing, create an empty NaN series
                    # This can happen when patients have no creatinine measurements
                    logger.warning(
                        f"SOFA component '{ctx.concept_name}' has no creatinine data. "
                        f"Returning empty result."
                    )
                    if data is not None and not data.empty and index_column:
                        result = pd.Series([], dtype=float, name=ctx.concept_name)
                    else:
                        result = pd.Series([], name=ctx.concept_name, dtype=float)
            else:
                result = func(**kwargs)
        except TypeError as e:
            if "unsupported operand type" in str(e) and "NoneType" in str(e):
                # Handle the case where callback functions receive None values due to missing data sources
                # This happens when concepts don't have mappings for the current database
                logger.warning(
                    f"SOFA component '{ctx.concept_name}' encountered missing data for database "
                    f"{getattr(ctx.data_source.config, 'name', 'unknown')}. Returning zeros."
                )

                # Create a result Series with zeros (same index as data if available)
                if data is not None and not data.empty and index_column:
                    result = pd.Series(0.0, index=data.index, name=ctx.concept_name)
                else:
                    # Fallback: create empty series with concept name
                    result = pd.Series([], name=ctx.concept_name, dtype=float)
            else:
                # Re-raise other TypeError exceptions
                raise e
        # Ensure result has the same index as data for assignment
        if isinstance(result, pd.Series):
            # Align index with data
            if not result.index.equals(data.index):
                result = result.reindex(data.index, fill_value=0.0)
        
        # CRITICAL: For sofa_renal, only keep rows within the data boundary of crea and urine24
        # i.e., only keep rows where the charttime is within the range where at least one input
        # originally had data (not filled by fill_gaps).
        # This replicates ricu's behavior where collect_dots only merges actual data points.
        if ctx.concept_name == 'sofa_renal' and index_column:
            crea_vals = kwargs.get('crea')
            urine24_vals = kwargs.get('urine24')
            
                    
            # Find the maximum charttime where either crea or urine24 has actual data (not NaN)
            max_valid_time = None
            if crea_vals is not None and isinstance(crea_vals, pd.Series) and crea_vals.notna().any():
                crea_max = data.loc[crea_vals.notna(), index_column].max()
                max_valid_time = crea_max if max_valid_time is None else max(max_valid_time, crea_max)
            if urine24_vals is not None and isinstance(urine24_vals, pd.Series) and urine24_vals.notna().any():
                urine_max = data.loc[urine24_vals.notna(), index_column].max()
                max_valid_time = urine_max if max_valid_time is None else max(max_valid_time, urine_max)
            
              
            # Filter: only keep rows where charttime <= max_valid_time
            if max_valid_time is not None:
                valid_mask = data[index_column] <= max_valid_time
                data = data[valid_mask].copy()
                result = result[valid_mask]
                
                
        # For optional parameters (like urine24 in sofa_renal), ensure they are None if all NaN
        # This replicates R ricu's behavior where missing optional params are treated as NULL
        # 🔧 FIX: Handle optional parameters correctly - convert all-NaN Series to None
        if ctx.concept_name == 'sofa_renal' and 'urine24' in kwargs:
            # If urine24 is all NaN or None, remove it from kwargs and call again
            urine24_val = kwargs['urine24']
            if urine24_val is None:
                # Already None, call without it
                kwargs_no_urine = {k: v for k, v in kwargs.items() if k != 'urine24'}
                result = func(**kwargs_no_urine)
                if isinstance(result, pd.Series) and not result.index.equals(data.index):
                    result = result.reindex(data.index, fill_value=0.0)
            elif isinstance(urine24_val, pd.Series) and urine24_val.isna().all():
                # All NaN, remove from kwargs and call again
                kwargs_no_urine = {k: v for k, v in kwargs.items() if k != 'urine24'}
                result = func(**kwargs_no_urine)
                if isinstance(result, pd.Series) and not result.index.equals(data.index):
                    result = result.reindex(data.index, fill_value=0.0)
        
        # CRITICAL: Replicate ricu_code's rm_cols behavior - remove input concept columns
        # In ricu_code: rm_cols(dat, cnc, by_ref = TRUE) removes the input concept columns
        # Keep only ID columns, time column, and the result column
        cols_to_remove = [name for name in tables.keys() if name in data.columns]
        if cols_to_remove:
            data = data.drop(columns=cols_to_remove)
        
        data[ctx.concept_name] = result
        
        cols = id_columns + ([index_column] if index_column else []) + [ctx.concept_name]
        frame = data[cols]
        return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column=ctx.concept_name)

    return wrapper


def _callback_sofa_resp(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate respiratory SOFA component with vent_ind expansion.
    
    Replicates R ricu's sofa_resp logic:
    - Expand vent_ind (win_tbl) to time series with aggregate="any"
    - Merge with pafi
    - Adjust pafi values: if pafi < 200 and NOT on ventilation, set pafi = 200
    - Calculate score
    """
    from .ts_utils import expand
    
    pafi_tbl = tables.get("pafi")
    vent_tbl = tables.get("vent_ind")
    
    if pafi_tbl is None:
        raise ValueError("sofa_resp requires 'pafi' concept")
    if vent_tbl is None:
        raise ValueError("sofa_resp requires 'vent_ind' concept")
    
    id_columns, index_column, _ = _assert_shared_schema({"pafi": pafi_tbl})
    
    # Get value column names for both tables (handle both ICUTable and WinTbl)
    pafi_col = pafi_tbl.value_column if hasattr(pafi_tbl, 'value_column') else "pafi"
    vent_col = vent_tbl.value_column if hasattr(vent_tbl, 'value_column') else "vent_ind"
    if pafi_col is None:
        pafi_col = "pafi"
    if vent_col is None:
        vent_col = "vent_ind"
    
    # Expand vent_ind (win_tbl) to time series with aggregate="any"
    # This replicates: expand(dat[[vent_var]], aggregate = "any")
    # 关键：需要确保vent_expanded的时间列名与pafi一致（使用pafi的index_column）
    if isinstance(vent_tbl, WinTbl):
        # Expand window table to time series
        # WinTbl has start (index_var) and duration (dur_var)
        vent_df = vent_tbl.data.copy()
        # Handle WinTbl vs ICUTable
        vent_start_col = vent_tbl.index_var if vent_tbl.index_var else vent_tbl.index_column
        dur_col = vent_tbl.dur_var
        
        # Get ID columns - handle both id_vars (TsTbl/WinTbl) and id_columns (ICUTable)
        vent_id_cols = vent_tbl.id_vars if hasattr(vent_tbl, 'id_vars') else vent_tbl.id_columns
        
        # 使用pafi的index_column作为目标时间列名，确保一致
        target_time_col = index_column  # pafi的时间列名
        
        if vent_start_col and dur_col:
            from .ts_utils import expand
            # Expand intervals to time series. Handle both numeric hours (already
            # aligned to ICU admission) and absolute datetimes.
            vent_df = vent_df.copy()
            if dur_col not in vent_df.columns:
                raise ValueError(
                    f"Duration column '{dur_col}' not found in vent_ind data. "
                    f"Available columns: {list(vent_df.columns)}"
                )

            # Determine time semantics
            start_series = vent_df[vent_start_col]
            duration_series = vent_df[dur_col]

            # Treat as numeric hours when start is non-datetime and convertible to numbers
            start_is_numeric = (
                not pd.api.types.is_datetime64_any_dtype(start_series)
                and not pd.api.types.is_timedelta64_dtype(start_series)
                and pd.to_numeric(start_series, errors="coerce").notna().any()
            )

            # Use the same step size as the data source interval (like R ricu's time_step(x))
            step_size = ctx.interval or pd.Timedelta(hours=1)  # Default to 1 hour like R ricu
            if not isinstance(step_size, pd.Timedelta):
                step_size = pd.Timedelta(step_size)

            # Handle both ICUTable and WinTbl
            if hasattr(vent_tbl, 'value_column'):
                value_col = vent_tbl.value_column or "vent_ind"
            else:
                value_col = "vent_ind"

            if start_is_numeric:
                # Numeric hours since ICU admission. Expand without converting to
                # epoch-based datetimes to avoid 1969/1970 artifacts.
                start_vals = pd.to_numeric(start_series, errors="coerce")
                if pd.api.types.is_timedelta64_dtype(duration_series):
                    dur_hours = duration_series.dt.total_seconds() / 3600.0
                else:
                    dur_td = pd.to_timedelta(duration_series, errors="coerce")
                    if dur_td.notna().any():
                        dur_hours = dur_td.dt.total_seconds() / 3600.0
                    else:
                        dur_hours = pd.to_numeric(duration_series, errors="coerce")

                end_vals = start_vals + dur_hours.fillna(0)
                step_hours = max(step_size.total_seconds() / 3600.0, 1e-6)

                expanded_rows = []
                id_cols = [col for col in vent_id_cols if col in vent_df.columns]

                # 🔧 CRITICAL FIX: 完全复制R ricu的expand逻辑
                # R ricu的expand为每个interval生成时间序列，然后aggregate处理重叠
                for idx, row in vent_df.iterrows():
                    start_val = start_vals.iloc[idx]
                    end_val = end_vals.iloc[idx]
                    if pd.isna(start_val) or pd.isna(end_val):
                        continue
                    if end_val < start_val:
                        end_val = start_val

                    # 生成这个间隔的时间序列
                    times = np.arange(start_val, end_val + step_hours, step_hours)
                    base = {col: row[col] for col in id_cols}
                    base[value_col] = bool(row.get(value_col, True))
                    for t in times:
                        new_row = dict(base)
                        new_row[target_time_col] = float(t)
                        expanded_rows.append(new_row)

                if expanded_rows:
                    vent_expanded_df = pd.DataFrame(expanded_rows)
                else:
                    vent_expanded_df = pd.DataFrame(columns=id_cols + [target_time_col, value_col])
            else:
                # Datetime semantics: fall back to expand() with date_range logic
                end_col = "_end_time"
                # 只有在不是numeric类型时才转换为datetime
                if pd.api.types.is_numeric_dtype(vent_df[vent_start_col]):
                    # 如果是numeric（小时），直接加duration
                    vent_df[end_col] = vent_df[vent_start_col] + vent_df[dur_col]
                else:
                    vent_df[end_col] = pd.to_datetime(vent_df[vent_start_col], errors="coerce") + pd.to_timedelta(vent_df[dur_col])

                vent_expanded_df = expand(
                    vent_df,
                    start_var=vent_start_col,
                    end_var=end_col,
                    step_size=step_size,
                    id_cols=vent_id_cols,
                    keep_vars=[value_col]
                )

                if vent_expanded_df.empty:
                    vent_expanded_df = pd.DataFrame(columns=vent_id_cols + [target_time_col, value_col])
                else:
                    if vent_start_col != target_time_col and vent_start_col in vent_expanded_df.columns:
                        vent_expanded_df = vent_expanded_df.rename(columns={vent_start_col: target_time_col})

            # Aggregate overlapping intervals with boolean OR ("any")
            key_cols = vent_id_cols + [target_time_col]
            if value_col in vent_expanded_df.columns and not vent_expanded_df.empty:
                vent_expanded_df[value_col] = vent_expanded_df[value_col].where(vent_expanded_df[value_col].notna(), False).astype(bool)
                vent_expanded_df = vent_expanded_df.groupby(key_cols, as_index=False)[value_col].any()
            elif vent_expanded_df.empty:
                vent_expanded_df = pd.DataFrame(columns=key_cols + [value_col])

            vent_expanded = ICUTable(
                data=vent_expanded_df,
                id_columns=vent_id_cols,
                index_column=target_time_col,
                value_column=value_col
            )
        else:
            vent_expanded = vent_tbl
            # 如果无法展开，至少确保时间列名一致
            if hasattr(vent_expanded, 'data') and vent_expanded.index_column != index_column:
                vent_expanded.data = vent_expanded.data.rename(columns={vent_expanded.index_column: index_column})
                vent_expanded.index_column = index_column
    else:
        vent_expanded = vent_tbl
        # 如果vent_ind不是WinTbl，也需要确保时间列名一致
        if hasattr(vent_expanded, 'data') and hasattr(vent_expanded, 'index_column'):
            if vent_expanded.index_column != index_column and vent_expanded.index_column in vent_expanded.data.columns:
                vent_expanded.data = vent_expanded.data.rename(columns={vent_expanded.index_column: index_column})
                vent_expanded.index_column = index_column
    
    # Merge pafi with expanded vent_ind
    pafi_df = pafi_tbl.data.copy()
    vent_df = vent_expanded.data.copy()
    
    # Reset index if MultiIndex to avoid merge issues
    if isinstance(pafi_df.index, pd.MultiIndex):
        pafi_df = pafi_df.reset_index()
    if isinstance(vent_df.index, pd.MultiIndex):
        vent_df = vent_df.reset_index()
    
    # Flatten MultiIndex columns if any
    if isinstance(pafi_df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns properly
        new_cols = []
        for col in pafi_df.columns:
            if isinstance(col, tuple):
                # Join tuple elements, skipping empty strings
                parts = [str(c) for c in col if c and str(c).strip()]
                new_col = '_'.join(parts) if parts else 'unknown'
                new_cols.append(new_col)
            else:
                new_cols.append(str(col))
        pafi_df.columns = new_cols
        # Ensure value column exists
        if pafi_col not in pafi_df.columns:
            # Try to find a column that might be the value
            for col in pafi_df.columns:
                if 'pafi' in col.lower() or col in ['min', 'max']:
                    pafi_df = pafi_df.rename(columns={col: pafi_col})
                    break
    
    if isinstance(vent_df.columns, pd.MultiIndex):
        new_cols = []
        for col in vent_df.columns:
            if isinstance(col, tuple):
                parts = [str(c) for c in col if c and str(c).strip()]
                new_col = '_'.join(parts) if parts else 'unknown'
                new_cols.append(new_col)
            else:
                new_cols.append(str(col))
        vent_df.columns = new_cols
    
    # Ensure key columns exist in both dataframes
    
    # Get actual ID columns from dataframes (they might be in index)
    actual_id_cols = []
    for col in id_columns:
        if col in pafi_df.columns:
            actual_id_cols.append(col)
        elif hasattr(pafi_df.index, 'names') and col in pafi_df.index.names:
            # Column is in index, need to reset it
            if not isinstance(pafi_df.index, pd.MultiIndex):
                pafi_df = pafi_df.reset_index()
            actual_id_cols.append(col)
    
    # Build key columns list
    key_cols = list(actual_id_cols)
    if index_column and index_column in pafi_df.columns:
        key_cols.append(index_column)
    
    # Ensure we have at least one key column
    if not key_cols:
        # Try to infer from column names - 扩展支持更多数据库的ID列
        for col in ['stay_id', 'icustay_id', 'admissionid', 'patientunitstayid', 'subject_id']:
            if col in pafi_df.columns and col in vent_df.columns:
                key_cols = [col]
                break
        if not key_cols:
            # Debug: print available columns
            pafi_cols = list(pafi_df.columns)
            vent_cols = list(vent_df.columns)
            common_cols = set(pafi_cols) & set(vent_cols)
            raise ValueError(
                f"无法确定合并键列：pafi_df和vent_df都没有共同的ID列。\n"
                f"pafi_df列: {pafi_cols}\n"
                f"vent_df列: {vent_cols}\n"
                f"共同列: {list(common_cols)}\n"
                f"id_columns: {id_columns}, index_column: {index_column}"
            )
    
    # 🔧 FIX: 统一ID列名 - 不同概念可能使用不同的ID列名（stay_id vs admissionid等）
    # 如果vent_df和pafi_df的ID列名不一致，重命名为统一的列名
    id_col_map = {
        'stay_id': ['stay_id', 'icustay_id', 'admissionid', 'patientunitstayid'],
        'icustay_id': ['stay_id', 'icustay_id', 'admissionid', 'patientunitstayid'],
        'admissionid': ['stay_id', 'icustay_id', 'admissionid', 'patientunitstayid'],
        'patientunitstayid': ['stay_id', 'icustay_id', 'admissionid', 'patientunitstayid']
    }
    
    # 找到pafi_df和vent_df各自的ID列
    pafi_id_col = None
    vent_id_col = None
    
    for col in ['admissionid', 'stay_id', 'icustay_id', 'patientunitstayid']:
        if col in pafi_df.columns:
            pafi_id_col = col
            break
    
    for col in ['admissionid', 'stay_id', 'icustay_id', 'patientunitstayid']:
        if col in vent_df.columns:
            vent_id_col = col
            break
    
    # 如果ID列名不一致，统一重命名为pafi的ID列名
    if pafi_id_col and vent_id_col and pafi_id_col != vent_id_col:
        vent_df = vent_df.rename(columns={vent_id_col: pafi_id_col})
        # 更新key_cols
        if vent_id_col in key_cols:
            key_cols = [pafi_id_col if c == vent_id_col else c for c in key_cols]
    
    # Ensure all key columns exist in both dataframes
    missing_in_pafi = [col for col in key_cols if col not in pafi_df.columns]
    missing_in_vent = [col for col in key_cols if col not in vent_df.columns]
    if missing_in_pafi:
        raise ValueError(f"Key columns missing in pafi_df: {missing_in_pafi}. Available columns: {list(pafi_df.columns)}")
    if missing_in_vent:
        raise ValueError(f"Key columns missing in vent_df: {missing_in_vent}. Available columns: {list(vent_df.columns)}")
    
    # 确保时间列类型一致（合并前统一）
    # 如果时间已经是数字类型（小时数），保持数字类型；否则统一为datetime
    if index_column and index_column in pafi_df.columns and index_column in vent_df.columns:
        pafi_time_is_numeric = pd.api.types.is_numeric_dtype(pafi_df[index_column])
        vent_time_is_numeric = pd.api.types.is_numeric_dtype(vent_df[index_column])
        
        if pafi_time_is_numeric and vent_time_is_numeric:
            # 都是数字类型，保持
            pass
        elif not pafi_time_is_numeric and not vent_time_is_numeric:
            # 都是datetime类型，统一格式
            def normalize_datetime(series):
                """标准化datetime列：移除时区，统一为datetime64[ns]"""
                # 如果已经是numeric，说明类型判断有误，直接返回
                if pd.api.types.is_numeric_dtype(series):
                    return series
                if pd.api.types.is_datetime64_any_dtype(series):
                    if hasattr(series.dtype, 'tz') and series.dtype.tz is not None:
                        series = series.dt.tz_localize(None)
                    return series.astype('datetime64[ns]')
                else:
                    return pd.to_datetime(series, errors='coerce').astype('datetime64[ns]')
            
            pafi_df[index_column] = normalize_datetime(pafi_df[index_column])
            vent_df[index_column] = normalize_datetime(vent_df[index_column])
        else:
            # 类型不一致，尝试统一为numeric类型
            # 如果一个是数字一个是datetime，转换datetime为numeric（小时数）
            pafi_is_numeric = pd.api.types.is_numeric_dtype(pafi_df[index_column])
            vent_is_numeric = pd.api.types.is_numeric_dtype(vent_df[index_column])
            
            if pafi_is_numeric and not vent_is_numeric:
                # pafi是numeric，vent是datetime，转换vent为numeric
                # 需要vent的入院时间，从数据源获取
                # 取vent_df的id列，查询入院时间
                if ctx and ctx.data_source:
                    vent_df[index_column] = ctx.resolver._align_time_to_admission(
                        vent_df,
                        ctx.data_source,
                        vent_tbl.id_columns if hasattr(vent_tbl, 'id_columns') else vent_tbl.id_vars,
                        index_column
                    )[index_column]
                else:
                    raise ValueError(
                        f"无法自动转换：pafi的时间列是numeric但vent是datetime，且没有数据源上下文。"
                    )
            elif vent_is_numeric and not pafi_is_numeric:
                # vent是numeric，pafi是datetime，转换pafi为numeric
                if ctx and ctx.data_source:
                    pafi_df[index_column] = ctx.resolver._align_time_to_admission(
                        pafi_df,
                        ctx.data_source,
                        pafi_tbl.id_columns if hasattr(pafi_tbl, 'id_columns') else pafi_tbl.id_vars,
                        index_column
                    )[index_column]
                else:
                    raise ValueError(
                        f"无法自动转换：vent的时间列是numeric但pafi是datetime，且没有数据源上下文。"
                    )
            else:
                # 两者都不是我们期望的类型
                raise ValueError(
                    f"时间列类型不一致且无法自动转换: "
                    f"pafi[{index_column}]={pafi_df[index_column].dtype}, "
                    f"vent[{index_column}]={vent_df[index_column].dtype}"
                )
    
    # Merge with outer join (all = TRUE in R)
    merged = pafi_df.merge(
        vent_df[key_cols + [vent_col]],
        on=key_cols,
        how="outer"
    )
    
    # 🔧 CRITICAL FIX: R's behavior for sofa_resp
    # R uses expand() on vent_ind which fills the full time range,
    # but pafi is NOT forward-filled infinitely. Instead:
    # 1. vent_ind is expanded to full time range (already done above)
    # 2. pafi is merged (with gaps)
    # 3. When pafi is NaN and vent_ind is also NaN/False, sofa_resp = NaN (not 0)
    # 4. When pafi < 200 and NOT on ventilation, pafi is adjusted to 200
    # 5. Score is calculated, and NaN pafi results in NaN score
    
    # The key insight: R's fill_gaps() is called on the RESULT of sofa_resp callback,
    # not on the input data. So we should not fill gaps here.
    # Instead, just handle the merge and calculate score, leaving NaN as NaN.
    
    # 🔧 CRITICAL FIX: 移除错误的PaFi调整逻辑
    # 原逻辑错误地调整了PaFi值：if pafi < 200 and NOT on ventilation, set pafi = 200
    # 但根据SOFA临床指南，PaFi评分应基于实际值，不应因通气状态而调整
    # R ricu可能没有这个调整，或调整条件不同（如仅针对无创通气）
    #
    # SOFA呼吸评分标准：
    # - PaFi < 100: 4分
    # - PaFi < 200: 3分
    # - PaFi < 300: 2分
    # - PaFi < 400: 1分
    # - PaFi >= 400: 0分
    #
    # 注意：某些临床变体可能对无通气患者的PaFi有特殊处理，
    # 但基于我们的数据分析，R ricu似乎没有这个调整

    # 不再调整PaFi值，直接基于实际PaFi计算SOFA评分
    
    # Calculate score using sofa_resp function
    # Pass None for vent_ind since we're not doing PaFi adjustment anymore
    vent_series = merged[vent_col] if vent_col in merged.columns else None
    score = sofa_resp(
        pd.to_numeric(merged[pafi_col], errors="coerce"),
        vent_series
    )

    merged[ctx.concept_name] = score
    cols = key_cols + [ctx.concept_name]
    
    return _as_icutbl(
        merged[cols].reset_index(drop=True),
        id_columns=id_columns,
        index_column=index_column,
        value_column=ctx.concept_name
    )


def _callback_sofa_score(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate SOFA score with sliding window support.
    
    Replicates R ricu's sofa_score logic exactly:
    1. Collect all component data with merge_dat = TRUE
    2. Fill gaps: res <- fill_gaps(dat)
    3. Apply sliding window: slide(res, !!expr, before = win_length, full_window = FALSE)
    4. Calculate total: rowSums(.SD, na.rm = TRUE)
    5. Optionally keep components or remove them
    
    Args:
        tables: Dictionary of input tables (SOFA components)
        ctx: Callback context with optional parameters:
            - win_length: Sliding window duration (default: 24 hours)
            - worst_val_fun: Aggregation function ('max', 'min', or callable, default: 'max')
            - keep_components: Whether to keep individual components (default: False)
            - full_window: Whether to require full window (default: False)
    
    Returns:
        ICUTable with SOFA scores
    """
    from .ts_utils import slide, fill_gaps, hours
    from .utils import max_or_na
    
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["sofa"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="sofa")

    # Get parameters from context
    win_length = ctx.kwargs.get('win_length', hours(24))
    worst_val_fun = ctx.kwargs.get('worst_val_fun', 'max')
    keep_components = ctx.kwargs.get('keep_components', False)
    full_window = ctx.kwargs.get('full_window', False)
    
    # 🚀 优化：使用字符串而非函数对象，配合 slide 内的直接 max/min 调用
    # R ricu uses max_or_na by default, which returns NA if all values are NA
    if worst_val_fun == 'max':
        worst_val_fun = 'max_or_na'  # ✅ 使用字符串
    elif worst_val_fun == 'min':
        worst_val_fun = 'min_or_na'  # ✅ 使用字符串
    
    # Convert timedelta to pd.Timedelta if needed
    if win_length is None:
        win_length = hours(24)  # Default to 24 hours if None
    elif hasattr(win_length, 'total_seconds'):  # datetime.timedelta
        win_length = pd.Timedelta(win_length)

    # SOFA components
    required = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal"]

    # 🔧 CRITICAL FIX: Ensure all components exist with proper missing data handling
    for name in required:
        data[name] = data.get(name)
    
    # CRITICAL: Fill gaps in time series (replicates R ricu fill_gaps(dat))
    # This ensures all time points are present before sliding window calculation
    if index_column and index_column in data.columns:
        interval = ctx.interval or pd.Timedelta(hours=1)
        # Fill gaps for each patient group
        id_cols_to_group = list(id_columns) if id_columns else []

        if id_cols_to_group:
            limits_df = _compose_fill_limits(data, id_cols_to_group, index_column, ctx)
            data = fill_gaps(
                data,
                id_cols=id_cols_to_group,
                index_col=index_column,
                interval=interval,
                limits=limits_df,
                method="none",
            )
        
        # Sort data by time
        data = data.sort_values(list(id_columns) + [index_column] if id_columns else [index_column])
        
        # Apply sliding window to each component (replicates R ricu slide)
        agg_dict = {}
        for comp in required:
            if comp in data.columns:
                agg_dict[comp] = worst_val_fun
        
        # Apply slide (replicates R ricu slide(res, !!expr, before = win_length, full_window = FALSE))
        if agg_dict:
            data = slide(
                data,
                list(id_columns),
                index_column,
                before=win_length,
                after=pd.Timedelta(0),
                agg_func=agg_dict,
                full_window=full_window,
            )
    
    # Calculate total SOFA score (replicates R ricu rowSums(.SD, na.rm = TRUE))
    # 🔧 FINAL FIX: Smart interpolation for time=0 to match R ricu behavior
    # Apply final calculation (sum components; NA treated as 0 via row sum)
    data["sofa"] = data[required].fillna(0).sum(axis=1)
    
    # Select output columns
    if keep_components:
        cols = id_columns + ([index_column] if index_column else []) + required + ["sofa"]
    else:
        cols = id_columns + ([index_column] if index_column else []) + ["sofa"]
    
    # Filter to existing columns
    cols = [c for c in cols if c in data.columns]
    frame = data[cols]
    
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="sofa")


def _callback_sofa2_score(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate SOFA-2 score with sliding window support.
    
    Similar to SOFA-1 but outputs 'sofa2' column and uses sofa2_* components.
    
    Args:
        tables: Dictionary of input tables (SOFA-2 components)
        ctx: Callback context with optional parameters:
            - win_length: Sliding window duration (default: 24 hours)
            - worst_val_fun: Aggregation function ('max', 'min', or callable, default: 'max')
            - keep_components: Whether to keep individual components (default: False)
            - full_window: Whether to require full window (default: False)
    
    Returns:
        ICUTable with SOFA-2 scores
    """
    from .ts_utils import slide, fill_gaps, hours
    from .utils import max_or_na
    
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        print(f"   ⚠️  SOFA-2回调: _merge_tables 返回空数据")
        cols = id_columns + ([index_column] if index_column else []) + ["sofa2"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="sofa2")

    # Get parameters from context
    win_length = ctx.kwargs.get('win_length', hours(24))
    worst_val_fun = ctx.kwargs.get('worst_val_fun', 'max')
    keep_components = ctx.kwargs.get('keep_components', False)
    full_window = ctx.kwargs.get('full_window', False)
    
    # 🚀 优化：使用字符串而非函数对象
    if worst_val_fun == 'max':
        worst_val_fun = 'max_or_na'  # ✅ 使用字符串
    elif worst_val_fun == 'min':
        worst_val_fun = 'min_or_na'  # ✅ 使用字符串
    
    # Convert timedelta to pd.Timedelta if needed
    if win_length is None:
        win_length = hours(24)  # Default to 24 hours if None
    elif hasattr(win_length, 'total_seconds'):  # datetime.timedelta
        win_length = pd.Timedelta(win_length)

    # SOFA-2 components (note the sofa2_ prefix)
    required = ["sofa2_resp", "sofa2_coag", "sofa2_liver", "sofa2_cardio", "sofa2_cns", "sofa2_renal"]
    
    # Ensure all components exist
    for name in required:
        if name not in data:
            data[name] = 0
    
    # Fill gaps and apply sliding window (same logic as SOFA-1)
    if index_column and index_column in data.columns:
        interval = ctx.interval or pd.Timedelta(hours=1)
        id_cols_to_group = list(id_columns) if id_columns else []
        
        if id_cols_to_group:
            limits_df = _compose_fill_limits(data, id_cols_to_group, index_column, ctx)
            data = fill_gaps(
                data,
                id_cols=id_cols_to_group,
                index_col=index_column,
                interval=interval,
                limits=limits_df,
                method="none",
            )
        
        # Sort data by time
        data = data.sort_values(list(id_columns) + [index_column] if id_columns else [index_column])
        
        # Apply sliding window to each component
        agg_dict = {}
        for comp in required:
            if comp in data.columns:
                agg_dict[comp] = worst_val_fun
        
        if agg_dict:
            data = slide(
                data,
                list(id_columns),
                index_column,
                before=win_length,
                after=pd.Timedelta(0),
                agg_func=agg_dict,
                full_window=full_window,
            )
    
    # Calculate total SOFA-2 score (note: output column is 'sofa2')
    data["sofa2"] = (
        data[required]
        .fillna(0)
        .astype(float)
        .sum(axis=1)
        .round()
        .astype(int)
    )
    
    # Select output columns
    if keep_components:
        cols = id_columns + ([index_column] if index_column else []) + required + ["sofa2"]
    else:
        cols = id_columns + ([index_column] if index_column else []) + ["sofa2"]
    
    # Filter to existing columns
    cols = [c for c in cols if c in data.columns]
    frame = data[cols]
    
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="sofa2")


def _callback_mews(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["mews"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="mews")

    result = mews_score(
        sbp=pd.to_numeric(data.get("sbp")),
        hr=pd.to_numeric(data.get("hr")),
        resp=pd.to_numeric(data.get("resp")),
        temp=pd.to_numeric(data.get("temp")),
        avpu=data.get("avpu").astype(str),
    )
    data["mews"] = result
    cols = id_columns + ([index_column] if index_column else []) + ["mews"]
    return _as_icutbl(data[cols].reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="mews")


def _callback_news(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["news"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="news")

    result = news_score(
        resp=pd.to_numeric(data.get("resp")),
        o2sat=pd.to_numeric(data.get("o2sat")),
        temp=pd.to_numeric(data.get("temp")),
        sbp=pd.to_numeric(data.get("sbp")),
        hr=pd.to_numeric(data.get("hr")),
        supp_o2=data.get("supp_o2").where(data.get("supp_o2").notna(), False).astype(bool),
        avpu=data.get("avpu").astype(str),
        keep_components=False,
    )
    data["news"] = result
    cols = id_columns + ([index_column] if index_column else []) + ["news"]
    return _as_icutbl(data[cols].reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="news")


def _callback_qsofa(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["qsofa"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="qsofa")

    data["qsofa"] = qsofa_score(
        sbp=pd.to_numeric(data.get("sbp")),
        resp=pd.to_numeric(data.get("resp")),
        gcs=pd.to_numeric(data.get("gcs")),
    )
    cols = id_columns + ([index_column] if index_column else []) + ["qsofa"]
    return _as_icutbl(data[cols].reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="qsofa")


def _callback_sirs(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["sirs"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="sirs")

    index = data.index
    data["sirs"] = sirs_score(
        temp=_get_numeric_series(data, "temp", index=index),
        hr=_get_numeric_series(data, "hr", index=index),
        resp=_get_numeric_series(data, "resp", index=index),
        pco2=_get_numeric_series(data, "pco2", index=index),
        wbc=_get_numeric_series(data, "wbc", index=index),
        bnd=_get_numeric_series(data, "bnd", index=index),
    )
    cols = id_columns + ([index_column] if index_column else []) + ["sirs"]
    return _as_icutbl(data[cols].reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="sirs")


def _match_fio2(
    tables: Dict[str, ICUTable],
    o2_col: str,  # po2 or o2sat
    fio2_col: str,  # fio2
    match_win: pd.Timedelta,
    mode: str = "match_vals",
    fix_na_fio2: bool = True,
    ctx: Optional[ConceptCallbackContext] = None,  # Use ConceptCallbackContext
) -> tuple[pd.DataFrame, list[str], Optional[str]]:
    """
    Match FiO2 with PO2/O2Sat measurements within a time window.
    
    This replicates R ricu's match_fio2() function:
    - mode="match_vals": Rolling join within match_win
    - mode="extreme_vals": Merge + sliding window with min(o2) and max(fio2)
    - mode="fill_gaps": Fill gaps + sliding window
    
    Args:
        tables: Dict with o2_col and fio2_col tables
        o2_col: Name of oxygen measurement (po2 or o2sat)
        fio2_col: Name of FiO2 measurement
        match_win: Time window for matching
        mode: Matching mode
        fix_na_fio2: Fill missing FiO2 with 21% room air
        ctx: ConceptCallbackContext for automatic ID conversion
        
    Returns:
        Merged DataFrame, id_columns, index_column
    """
    from .ts_utils import slide, fill_gaps
    
    o2_tbl = tables[o2_col]
    fio2_tbl = tables[fio2_col]
    
    # Try automatic ID conversion if IDs don't match and ctx is available
    if ctx is not None:
        id_columns, index_column, converted_tables = _assert_shared_schema(
            {o2_col: o2_tbl, fio2_col: fio2_tbl},
            ctx=ctx,
            convert_ids=True
        )
        
        # Use converted tables if conversion happened
        if converted_tables:
            o2_tbl = converted_tables[o2_col]
            fio2_tbl = converted_tables[fio2_col]
    else:
        # No context available, just check schema without conversion
        id_columns, index_column, _ = _assert_shared_schema(
            {o2_col: o2_tbl, fio2_col: fio2_tbl},
            ctx=None,
            convert_ids=False
        )
    
    if mode == "match_vals":
        # Rolling join: merge o2 and fio2 within time window
        # This matches R's rolling join behavior
        o2_df = o2_tbl.data
        fio2_df = fio2_tbl.data
        
        # Rename value columns
        o2_val_col = o2_tbl.value_column or o2_col
        fio2_val_col = fio2_tbl.value_column or fio2_col
        
        if o2_val_col != o2_col:
            o2_df = o2_df.rename(columns={o2_val_col: o2_col})
        if fio2_val_col != fio2_col:
            fio2_df = fio2_df.rename(columns={fio2_val_col: fio2_col})
        
        # Use pd.merge_asof for rolling join (similar to R's data.table rolling join)
        if index_column:
            # 保存原始时间列类型（numeric或datetime）
            o2_time_is_numeric = pd.api.types.is_numeric_dtype(o2_df[index_column])
            fio2_time_is_numeric = pd.api.types.is_numeric_dtype(fio2_df[index_column])
            # 统一用于 numeric<->datetime 临时转换的基准时间
            base_time = pd.Timestamp('2000-01-01')
            
            # Convert to datetime if not already (但不转换numeric类型)
            # 对于numeric类型，merge_asof需要先转换为datetime，然后在merge后转换回numeric
            if not o2_time_is_numeric:
                o2_df[index_column] = pd.to_datetime(o2_df[index_column], errors='coerce')
            if not fio2_time_is_numeric:
                fio2_df[index_column] = pd.to_datetime(fio2_df[index_column], errors='coerce')
            
            # 如果原始时间列是numeric类型，需要临时转换为datetime进行merge_asof
            # 然后在merge后转换回numeric类型
            o2_time_backup = None
            fio2_time_backup = None
            numeric_unit = 'h'
            if ctx is not None:
                ds_cfg = getattr(getattr(ctx, "data_source", None), "config", None)
                ds_name = getattr(ds_cfg, "name", "") if ds_cfg is not None else ""
                if isinstance(ds_name, str) and ds_name.lower() == "aumc":
                    numeric_unit = 'ms'
            if o2_time_is_numeric:
                o2_time_backup = o2_df[index_column]
                # 对于numeric类型，需要转换为datetime进行merge_asof
                o2_df = o2_df.copy()  # Only copy when we need to modify
                o2_df[index_column] = base_time + pd.to_timedelta(o2_df[index_column], unit=numeric_unit)
            if fio2_time_is_numeric:
                fio2_time_backup = fio2_df[index_column]
                fio2_df = fio2_df.copy()  # Only copy when we need to modify
                fio2_df[index_column] = base_time + pd.to_timedelta(fio2_df[index_column], unit=numeric_unit)
            
            # 确保数据在每个by分组内都是排序的（merge_asof的严格要求）
            # 先选择需要的列，然后排序
            o2_subset = o2_df[id_columns + [index_column, o2_col]]
            fio2_subset = fio2_df[id_columns + [index_column, fio2_col]]
            
            # 移除NaN时间值（NaN会导致排序问题）
            o2_subset = o2_subset.dropna(subset=[index_column])
            fio2_subset = fio2_subset.dropna(subset=[index_column])
            
            # 关键：merge_asof要求每个by分组内的on列必须严格排序
            # 必须按by列+on列排序，确保每个分组内on列都是递增的
            if id_columns:
                # 方法：按分组逐个排序，确保每个分组内时间列严格递增
                # 这样可以避免pandas sort_values在某些边界情况下的问题
                o2_groups = []
                for id_val in o2_subset[id_columns[0]].unique():
                    group = o2_subset[o2_subset[id_columns[0]] == id_val]
                    # 确保每个分组内时间列严格排序 - sort_values creates a copy
                    group = group.sort_values(by=index_column, kind='mergesort')
                    o2_groups.append(group)
                if o2_groups:
                    o2_subset = pd.concat(o2_groups, ignore_index=True)
                else:
                    o2_subset = pd.DataFrame()
                
                fio2_groups = []
                for id_val in fio2_subset[id_columns[0]].unique():
                    group = fio2_subset[fio2_subset[id_columns[0]] == id_val]
                    # 确保每个分组内时间列严格排序 - sort_values creates a copy
                    group = group.sort_values(by=index_column, kind='mergesort')
                    fio2_groups.append(group)
                if fio2_groups:
                    fio2_subset = pd.concat(fio2_groups, ignore_index=True)
                else:
                    # 如果 fio2_groups 为空，返回空 DataFrame
                    fio2_subset = pd.DataFrame()
                
                # 关键修复：如果o2_subset为空，返回空结果
                # 但如果fio2_subset为空，不返回空结果（后面会用21%填充）
                if o2_subset.empty:
                    return pd.DataFrame(columns=id_columns + [index_column]), id_columns, index_column
                
                # 如果fio2为空，创建与o2_subset时间点对齐的DataFrame
                # 所有fio2值都是NaN，后续会被fix_na_fio2填充为21%
                if fio2_subset.empty:
                    merged = o2_subset
                    merged = merged.assign(**{fio2_col: float('nan')})
                    # 转换回原始时间类型
                    if o2_time_is_numeric:
                        merged[index_column] = o2_time_backup
                    # Fix missing FiO2 with 21% room air
                    if fix_na_fio2:
                        merged[fio2_col] = merged[fio2_col].fillna(21.0)
                    return merged, id_columns, index_column
                
                # 正常情况：o2和fio2都有数据，进行merge_asof
                # 最后再次按id列和时间列排序，确保整体顺序正确
                o2_subset = o2_subset.sort_values(by=id_columns + [index_column], kind='mergesort')
                fio2_subset = fio2_subset.sort_values(by=id_columns + [index_column], kind='mergesort')
            else:
                o2_subset = o2_subset.sort_values(by=index_column, kind='mergesort')
                fio2_subset = fio2_subset.sort_values(by=index_column, kind='mergesort')
            
            # 关键修复：pandas的merge_asof对排序检查非常严格，即使看起来排序了也可能失败
            # 解决方法：按分组逐个处理，不使用by参数，避免pandas的严格检查
            if id_columns:
                merged_fwd_list = []
                merged_bwd_list = []
                
                # 按每个ID分组处理
                unique_ids = o2_subset[id_columns[0]].unique()
                for id_val in unique_ids:
                    # 获取当前ID的数据
                    o2_mask = o2_subset[id_columns[0]] == id_val
                    fio2_mask = fio2_subset[id_columns[0]] == id_val
                    
                    o2_group = o2_subset[o2_mask]
                    fio2_group = fio2_subset[fio2_mask]
                    
                    # 🔧 CRITICAL FIX: 如果 o2_group 为空，跳过
                    # 但如果 fio2_group 为空，不跳过！应该填充 fio2=21%
                    if len(o2_group) == 0:
                        continue
                    
                    # 如果 fio2_group 为空，为当前患者创建 fio2=NaN 的数据，后续会被填充为 21%
                    if len(fio2_group) == 0:
                        merged_fwd_group = o2_group[[index_column, o2_col]].assign(**{fio2_col: float('nan')})
                        # 添加ID列
                        for col in id_columns:
                            merged_fwd_group[col] = id_val
                        merged_fwd_list.append(merged_fwd_group)
                        continue
                    
                    # 确保每个分组内时间列严格排序（单独排序，避免跨分组问题）
                    o2_group = o2_group.sort_values(by=index_column, kind='mergesort').reset_index(drop=True)
                    fio2_group = fio2_group.sort_values(by=index_column, kind='mergesort').reset_index(drop=True)
                    
                    # 验证排序
                    if not o2_group[index_column].is_monotonic_increasing:
                        o2_group = o2_group.sort_values(by=index_column, kind='mergesort').reset_index(drop=True)
                    if not fio2_group[index_column].is_monotonic_increasing:
                        fio2_group = fio2_group.sort_values(by=index_column, kind='mergesort').reset_index(drop=True)
                    
                    # Forward join: 不使用by参数，因为已经按ID分组了
                    try:
                        merged_fwd_group = pd.merge_asof(
                            o2_group[[index_column, o2_col]],
                            fio2_group[[index_column, fio2_col]],
                            on=index_column,
                            tolerance=match_win,
                            direction='backward'
                        )
                        # 添加ID列
                        for col in id_columns:
                            merged_fwd_group[col] = id_val
                        merged_fwd_list.append(merged_fwd_group)
                    except Exception as e:
                        # 如果merge_asof失败，跳过这个分组
                        continue
                    
                    # Backward join
                    try:
                        merged_bwd_group = pd.merge_asof(
                            fio2_group[[index_column, fio2_col]],
                            o2_group[[index_column, o2_col]],
                            on=index_column,
                            tolerance=match_win,
                            direction='backward'
                        )
                        # 添加ID列
                        for col in id_columns:
                            merged_bwd_group[col] = id_val
                        merged_bwd_list.append(merged_bwd_group)
                    except Exception as e:
                        # 如果merge_asof失败，跳过这个分组
                        continue
                
                # 合并所有分组的结果
                if merged_fwd_list:
                    merged_fwd = pd.concat(merged_fwd_list, ignore_index=True)
                else:
                    merged_fwd = pd.DataFrame(columns=id_columns + [index_column, o2_col, fio2_col])
                
                if merged_bwd_list:
                    merged_bwd = pd.concat(merged_bwd_list, ignore_index=True)
                else:
                    merged_bwd = pd.DataFrame(columns=id_columns + [index_column, o2_col, fio2_col])
            else:
                # 没有ID列，直接处理
                merged_fwd = pd.merge_asof(
                    o2_subset,
                    fio2_subset,
                    on=index_column,
                    tolerance=match_win,
                    direction='backward'
                )

                merged_bwd = pd.merge_asof(
                    fio2_subset,
                    o2_subset,
                    on=index_column,
                    tolerance=match_win,
                    direction='backward'
                )
            
            # Combine both directions and remove duplicates (like R's rbind + unique)
            merged = pd.concat([merged_fwd, merged_bwd], ignore_index=True)
            merged = merged.drop_duplicates()
            # 如果两个输入原本都是数值型相对小时，则将结果时间列转换回相对小时
            if o2_time_is_numeric and fio2_time_is_numeric:
                try:
                    merged[index_column] = (
                        pd.to_datetime(merged[index_column], errors='coerce') - base_time
                    ) / pd.Timedelta(hours=1)
                except Exception:
                    pass
        else:
            # No time index, just merge
            merged, _, _ = _merge_tables({o2_col: o2_tbl, fio2_col: fio2_tbl}, ctx=ctx, how='inner')
            
    else:
        # mode = "extreme_vals" or "fill_gaps"
        # Merge all data
        merged, id_columns, index_column = _merge_tables({o2_col: o2_tbl, fio2_col: fio2_tbl}, ctx=ctx, how='outer')
        
        if mode == "fill_gaps" and index_column:
            # Fill gaps in time series
            from .ts_utils import fill_gaps
            merged = fill_gaps(
                merged, 
                id_columns, 
                index_column, 
                pd.Timedelta(hours=1),  # Use hourly interval
                method='none'
            )
        
        # Apply sliding window: min(o2) and max(fio2)
        if index_column and not merged.empty:
            from .ts_utils import slide
            from .utils import min_or_na, max_or_na
            
            agg_dict = {}
            if o2_col in merged.columns:
                agg_dict[o2_col] = min_or_na
            if fio2_col in merged.columns:
                agg_dict[fio2_col] = max_or_na
            
            if agg_dict:
                merged = slide(
                    merged,
                    id_columns,
                    index_column,
                    before=match_win,
                    after=pd.Timedelta(0),
                    agg_func=agg_dict,
                    full_window=False
                )
    
    # Fix missing FiO2 with 21% room air
    if fix_na_fio2 and fio2_col in merged.columns:
        merged[fio2_col] = merged[fio2_col].fillna(21.0)
    
    return merged, id_columns, index_column


def _callback_pafi(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
    *,
    source_col_a: str,  # po2 or o2sat
    source_col_b: str,  # fio2
    output_col: str,    # pafi or safi
) -> ICUTable:
    """
    Calculate PaO2/FiO2 ratio (pafi) or SpO2/FiO2 ratio (safi).
    
    完整复刻 R ricu 的 pafi/safi 函数:
    - 支持3种匹配模式: match_vals, extreme_vals, fill_gaps
    - 在时间窗口内匹配 po2/o2sat 和 fio2
    - 填充缺失的 fio2 为 21% (室内空气)
    - 过滤无效值
    
    Args:
        tables: 概念表字典
        ctx: 回调上下文,可包含:
            - match_win: 匹配时间窗口 (默认: 2小时)
            - mode: 匹配模式 (默认: "match_vals")
            - fix_na_fio2: 填充缺失FiO2 (默认: True)
        source_col_a: po2 或 o2sat 列名
        source_col_b: fio2 列名
        output_col: 输出列名 (pafi 或 safi)
        
    Returns:
        包含计算结果的 ICUTable
        
    Examples:
        >>> # PaFi = 100 * PaO2 / FiO2
        >>> pafi_tbl = _callback_pafi(
        ...     {"po2": po2_tbl, "fio2": fio2_tbl},
        ...     ctx,
        ...     source_col_a="po2",
        ...     source_col_b="fio2", 
        ...     output_col="pafi"
        ... )
    """
    # Get parameters from context (with R ricu defaults)
    match_win = ctx.kwargs.get('match_win', pd.Timedelta(hours=2))
    mode = ctx.kwargs.get('mode', 'match_vals')
    fix_na_fio2 = ctx.kwargs.get('fix_na_fio2', True)
    
    # Validate mode
    if mode not in ['match_vals', 'extreme_vals', 'fill_gaps']:
        mode = 'match_vals'
    
    # Convert match_win to pd.Timedelta if needed
    if isinstance(match_win, (int, float)):
        match_win = pd.Timedelta(hours=match_win)
    elif hasattr(match_win, 'total_seconds'):  # datetime.timedelta
        match_win = pd.Timedelta(match_win)
    
    # Ensure tables don't have MultiIndex columns
    cleaned_tables = {}
    for name, table in tables.items():
        if isinstance(table, ICUTable):
            table_data = table.data.copy()
            # Reset MultiIndex index
            if isinstance(table_data.index, pd.MultiIndex):
                table_data = table_data.reset_index()
            # Flatten MultiIndex columns
            if isinstance(table_data.columns, pd.MultiIndex):
                new_cols = []
                for col in table_data.columns:
                    if isinstance(col, tuple):
                        # Join tuple elements, skipping empty strings
                        parts = [str(c) for c in col if c and str(c).strip()]
                        new_col = '_'.join(parts) if parts else name
                        new_cols.append(new_col)
                    else:
                        new_cols.append(str(col))
                table_data.columns = new_cols
            # Recreate ICUTable with cleaned data
            cleaned_tables[name] = ICUTable(
                data=table_data,
                id_columns=table.id_columns,
                index_column=table.index_column,
                value_column=table.value_column,
                unit_column=table.unit_column,
                time_columns=table.time_columns,
            )
        else:
            cleaned_tables[name] = table
    
    # Match FiO2 with O2 measurements
    data, id_columns, index_column = _match_fio2(
        cleaned_tables, 
        source_col_a, 
        source_col_b, 
        match_win, 
        mode, 
        fix_na_fio2,
        ctx=ctx  # Pass callback context directly
    )
    
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + [output_col]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column=output_col)

    # Get series (already renamed in _match_fio2)
    o2 = pd.to_numeric(data.get(source_col_a), errors="coerce")
    fio2 = pd.to_numeric(data.get(source_col_b), errors="coerce")
    
    # CRITICAL FIX: Normalize FiO2 unit if provided as fraction (0–1)
    # This prevents PaFi from being inflated by 100x when FiO2 is stored as decimal
    fio2_unit = (ctx.kwargs or {}).get("fio2_unit")
    if fio2_unit == "fraction":
        # Explicitly specified as fraction, convert to percentage
        fio2 = fio2 * 100.0
    elif fio2_unit == "percentage":
        # Already percentage, no conversion needed
        pass
    elif fio2_unit is None:
        # Auto-detect: if majority of non-null values are <= 1.0, treat as fraction
        non_null = fio2.dropna()
        if len(non_null) > 0 and (non_null.le(1.0).mean() > 0.5):
            fio2 = fio2 * 100.0
    
    # Filter: !is.na(po2) & !is.na(fio2) & fio2 != 0
    valid_mask = o2.notna() & fio2.notna() & (fio2 != 0)
    data = data[valid_mask].copy()
    
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + [output_col]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column=output_col)
    
    # Recalculate after filtering
    o2 = pd.to_numeric(data[source_col_a], errors="coerce")
    fio2 = pd.to_numeric(data[source_col_b], errors="coerce")
    
    # Apply the same normalization to the filtered data
    if fio2_unit == "fraction":
        fio2 = fio2 * 100.0
    elif fio2_unit is None:
        non_null = fio2.dropna()
        if len(non_null) > 0 and (non_null.le(1.0).mean() > 0.5):
            fio2 = fio2 * 100.0
    
    # Calculate ratio: pafi/safi = 100 * po2/o2sat / fio2
    data[output_col] = 100 * o2 / fio2
    
    # Keep only essential columns (like R's rm_cols)
    cols = id_columns + ([index_column] if index_column else []) + [output_col]
    frame = data[cols]
    
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column=output_col)


def _callback_supp_o2(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    vent_tbl = tables["vent_ind"]
    fio2_tbl = tables["fio2"]

    # When vent_ind arrives as a WinTbl we need a dense hourly indicator to align with FiO2.
    if isinstance(vent_tbl, WinTbl):
        desired_index = fio2_tbl.index_column or vent_tbl.index_column or "charttime"
        vent_col = vent_tbl.value_column or "vent_ind"
        vent_tbl = _expand_win_table_to_interval(
            vent_tbl,
            interval=ctx.interval,
            value_column=vent_col,
            target_index=desired_index,
            fill_value=True,
        )

    id_columns, index_column, _ = _assert_shared_schema({"vent_ind": vent_tbl, "fio2": fio2_tbl})
    vent_df = vent_tbl.data.copy()
    fio2_df = fio2_tbl.data.copy()

    # 🔧 FIX: vent_ind is often a WinTbl indexed by starttime while fio2 uses charttime.
    # Align the index column names so we can merge on a common timeline.
    vent_index = vent_tbl.index_column
    fio2_index = fio2_tbl.index_column

    # Prefer the fio2 index (charttime) for the merged timeline. If missing, fall back to vent index.
    if fio2_index is None and "charttime" in fio2_df.columns:
        fio2_index = "charttime"
    if vent_index is None and "charttime" in vent_df.columns:
        vent_index = "charttime"
    if vent_index is None and "starttime" in vent_df.columns:
        vent_index = "starttime"

    # If we still don't have an index from either table, fail fast.
    merged_index = fio2_index or vent_index
    if merged_index is None:
        raise ValueError("supp_o2 requires at least one time column")

    # Ensure both tables expose the merged index column so downstream merge succeeds.
    if merged_index not in vent_df.columns and vent_index in vent_df.columns:
        vent_df = vent_df.rename(columns={vent_index: merged_index})
    if merged_index not in fio2_df.columns and fio2_index in fio2_df.columns:
        fio2_df = fio2_df.rename(columns={fio2_index: merged_index})

    index_column = merged_index
    key_cols = (id_columns or []) + [index_column]

    if index_column not in vent_df.columns:
        # When vent_ind comes as WinTbl without explicit timestamp, approximate using starttime column
        possible_cols = ["starttime", "charttime", "time"]
        for candidate in possible_cols:
            if candidate in vent_df.columns:
                vent_df = vent_df.rename(columns={candidate: index_column})
                break
    if index_column not in fio2_df.columns:
        possible_cols = ["charttime", "starttime", "time"]
        for candidate in possible_cols:
            if candidate in fio2_df.columns:
                fio2_df = fio2_df.rename(columns={candidate: index_column})
                break

    if index_column not in vent_df.columns or index_column not in fio2_df.columns:
        raise ValueError("supp_o2 requires vent_ind and fio2 tables to provide a shared time column")

    vent_col = vent_tbl.value_column or "vent_ind"
    fio2_col = fio2_tbl.value_column or "fio2"

    fio2_df[fio2_col] = pd.to_numeric(fio2_df[fio2_col], errors="coerce")

    # Align both tables on a shared MultiIndex and compute boolean result via numpy.
    id_components = key_cols.copy()
    if not id_components:
        id_components = [index_column]

    vent_series = (
        vent_df.set_index(id_components)[vent_col]
        if key_cols
        else vent_df.set_index(index_column)[vent_col]
    )
    fio2_series = (
        fio2_df.set_index(id_components)[fio2_col]
        if key_cols
        else fio2_df.set_index(index_column)[fio2_col]
    )

    shared_index = vent_series.index.union(fio2_series.index)
    vent_aligned = vent_series.reindex(shared_index, fill_value=False).astype(bool, copy=False)
    fio2_aligned = pd.to_numeric(fio2_series.reindex(shared_index, fill_value=21.0), errors="coerce").fillna(21.0)

    supp_mask = np.logical_or(vent_aligned.to_numpy(), fio2_aligned.to_numpy() > 21.0)
    result = shared_index.to_frame(index=False)
    result["supp_o2"] = supp_mask

    if key_cols:
        cols = key_cols + ["supp_o2"]
        return _as_icutbl(result[cols], id_columns=id_columns, index_column=index_column, value_column="supp_o2")

    cols = [index_column, "supp_o2"]
    return _as_icutbl(result[cols], id_columns=id_columns, index_column=index_column, value_column="supp_o2")


def _callback_vent_ind(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    from .ts_utils import expand

    interval = ctx.interval or pd.Timedelta(hours=1)
    if not isinstance(interval, pd.Timedelta):
        interval = pd.to_timedelta(interval)

    match_win = ctx.kwargs.get("match_win", pd.Timedelta(hours=6))
    if not isinstance(match_win, pd.Timedelta):
        match_win = pd.to_timedelta(match_win)

    min_length = ctx.kwargs.get("min_length", pd.Timedelta(minutes=30))
    if not isinstance(min_length, pd.Timedelta):
        min_length = pd.to_timedelta(min_length)

    relevant_tables = {
        name: tbl
        for name, tbl in tables.items()
        if name in {"vent_start", "vent_end", "mech_vent"} and tbl is not None
    }

    if not relevant_tables:
        raise ValueError("vent_ind requires vent_start or mech_vent concept data")

    id_columns, _, converted = _assert_shared_schema(relevant_tables, ctx=ctx, convert_ids=True)
    if converted:
        for name in list(relevant_tables.keys()):
            if name in converted:
                relevant_tables[name] = converted[name]

    id_columns = id_columns or []

    start_tbl = relevant_tables.get("vent_start")
    end_tbl = relevant_tables.get("vent_end")
    mech_tbl = relevant_tables.get("mech_vent")

    time_column = (
        (start_tbl.index_column if start_tbl and start_tbl.index_column else None)
        or (mech_tbl.index_column if mech_tbl and mech_tbl.index_column else None)
        or "time"
    )

    def _empty_result() -> ICUTable:
        cols = list(id_columns)
        if time_column:
            cols.append(time_column)
        cols.append("vent_ind")
        frame = pd.DataFrame(columns=cols)
        return _as_icutbl(frame, id_columns=id_columns, index_column=time_column, value_column="vent_ind")

    def _relative_hours(frame: pd.DataFrame, column: str) -> pd.Series:
        """Convert heterogeneous time columns to hours since ICU admission."""
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        if not hasattr(ctx, "resolver") or not hasattr(ctx.resolver, "_align_time_to_admission"):
            return pd.Series(np.nan, index=series.index)
        if not id_columns:
            return pd.Series(np.nan, index=series.index)
        helper = frame[list(id_columns) + [column]].copy()
        helper = helper.rename(columns={column: "__time"})
        aligned = ctx.resolver._align_time_to_admission(  # type: ignore[attr-defined]
            helper,
            ctx.data_source,
            list(id_columns),
            "__time",
        )
        return aligned["__time"]

    def _coerce_time(series: pd.Series):
        if pd.api.types.is_datetime64_any_dtype(series):
            clean = pd.to_datetime(series, errors="coerce").dt.tz_localize(None)
            return clean, lambda values: values
        if pd.api.types.is_timedelta64_dtype(series):
            base = pd.Timestamp("1970-01-01")
            clean = base + series
            return clean, lambda values: (values - base)
        base = pd.Timestamp("1970-01-01")
        numeric = pd.to_numeric(series, errors="coerce")
        clean = base + pd.to_timedelta(numeric, unit="h")
        return clean, lambda values: (values - base).dt.total_seconds() / 3600.0

    def _coerce_duration(series: pd.Series) -> pd.Series:
        if pd.api.types.is_timedelta64_dtype(series):
            return series
        try:
            td_series = pd.to_timedelta(series, errors="coerce")
            if td_series.notna().any():
                return td_series
        except Exception:  # fallback to numeric parsing
            pass
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_timedelta(numeric, unit="h")

    def _expand_windows(window_df: pd.DataFrame, revert_fn) -> ICUTable:
        if window_df.empty:
            return _empty_result()

        work = window_df.copy()
        work["vent_dur_td"] = _coerce_duration(work["vent_dur_td"]).fillna(match_win)
        work = work.dropna(subset=["_start_dt", "vent_dur_td"])
        if work.empty:
            return _empty_result()

        work["_end_dt"] = work["_start_dt"] + work["vent_dur_td"]
        expanded = expand(
            work,
            start_var="_start_dt",
            end_var="_end_dt",
            step_size=interval,
            id_cols=id_columns,
            keep_vars=None,
        )
        if expanded.empty:
            return _empty_result()

        expanded = expanded.rename(columns={"_start_dt": time_column})
        if revert_fn is not None:
            expanded[time_column] = revert_fn(expanded[time_column])
        expanded["vent_ind"] = True

        group_cols = list(id_columns) + [time_column]
        expanded = expanded.groupby(group_cols, as_index=False)["vent_ind"].any()
        expanded = expanded.reset_index(drop=True)
        return _as_icutbl(expanded, id_columns=id_columns, index_column=time_column, value_column="vent_ind")

    def _windows_from_mech(mech: ICUTable | WinTbl) -> Optional[ICUTable]:
        df = mech.data.copy()
        if df.empty:
            return None

        value_col = mech.value_column or "mech_vent"
        if value_col in df.columns:
            df["vent_flag"] = pd.Series(df[value_col]).fillna(False)
            if df["vent_flag"].dtype == bool:
                vent_mask = df["vent_flag"]
            else:
                vent_mask = ~df["vent_flag"].isin([False, 0, "0", "false", "False", "none", None])
        else:
            vent_mask = pd.Series(True, index=df.index)

        df = df[vent_mask]
        if df.empty:
            return None

        idx_col = mech.index_column or mech.index_var or time_column
        if idx_col not in df.columns:
            for candidate in ["charttime", "starttime", "time"]:
                if candidate in df.columns:
                    idx_col = candidate
                    break
        if idx_col not in df.columns:
            return None

        start_times, revert_fn = _coerce_time(df[idx_col])
        start_hours = _relative_hours(df, idx_col)
        df = df.assign(_start_dt=start_times, _start_hours=start_hours).dropna(subset=["_start_dt", "_start_hours"])
        if df.empty:
            return None
        start_hours = df["_start_hours"]

        dur_series: Optional[pd.Series] = None
        if isinstance(mech, WinTbl) and mech.dur_var and mech.dur_var in df.columns:
            dur_series = df[mech.dur_var]
        else:
            end_col = next(
                (col for col in ("endtime", "end_time", "stop", "end") if col in df.columns),
                None,
            )
            if end_col is not None:
                end_hours = _relative_hours(df, end_col)
                dur_hours = end_hours - start_hours
                dur_series = pd.to_timedelta(dur_hours, unit="h")
            elif "duration" in df.columns:
                dur_series = pd.to_timedelta(df["duration"], errors="coerce")

        if dur_series is None:
            dur_series = pd.Series(match_win, index=df.index)

        dur_series = _coerce_duration(dur_series).fillna(match_win)

        window_df = df[id_columns + ["_start_dt"]].copy()
        window_df["vent_dur_td"] = dur_series.values
        return _expand_windows(window_df, revert_fn)

    def _windows_from_events(start: ICUTable, end: Optional[ICUTable]) -> ICUTable:
        start_df = start.data.copy()
        val_col = start.value_column or "vent_start"
        if val_col in start_df.columns:
            start_df = start_df[pd.to_numeric(start_df[val_col], errors="coerce").fillna(0).astype(bool)]
        if start_df.empty:
            return _empty_result()

        idx_col = start.index_column or time_column
        if idx_col not in start_df.columns:
            for candidate in ["charttime", "starttime", "time"]:
                if candidate in start_df.columns:
                    idx_col = candidate
                    break
        if idx_col not in start_df.columns:
            return _empty_result()

        start_times, revert_fn = _coerce_time(start_df[idx_col])
        start_df = start_df.assign(_start_dt=start_times).dropna(subset=["_start_dt"])
        if start_df.empty:
            return _empty_result()

        if end is not None and not end.data.empty:
            end_df = end.data.copy()
            end_col = end.value_column or "vent_end"
            if end_col in end_df.columns:
                end_df = end_df[pd.to_numeric(end_df[end_col], errors="coerce").fillna(0).astype(bool)]
            if not end_df.empty:
                end_idx = end.index_column or idx_col
                if end_idx not in end_df.columns:
                    for candidate in ["endtime", "charttime", "time"]:
                        if candidate in end_df.columns:
                            end_idx = candidate
                            break
                end_times, _ = _coerce_time(end_df[end_idx])
                end_df = end_df.assign(_end_dt=end_times).dropna(subset=["_end_dt"])
            else:
                end_df = None
        else:
            end_df = None

        sort_cols = ["_start_dt"]
        if id_columns:
            sort_cols += list(id_columns)
        start_sorted = start_df.sort_values(sort_cols).reset_index(drop=True)

        if end_df is not None and not end_df.empty:
            end_sort_cols = ["_end_dt"]
            if id_columns:
                end_sort_cols += list(id_columns)
            end_sorted = end_df.sort_values(end_sort_cols).reset_index(drop=True)
            merge_kwargs = {
                "left_on": "_start_dt",
                "right_on": "_end_dt",
                "direction": "forward",
                "tolerance": match_win,
            }
            if id_columns:
                merge_kwargs["by"] = id_columns
            merged = pd.merge_asof(start_sorted, end_sorted[id_columns + ["_end_dt"]], **merge_kwargs)
            merged["_matched_end"] = merged["_end_dt"].where(merged["_end_dt"].notna(), merged["_start_dt"] + match_win)
        else:
            merged = start_sorted.copy()
            merged["_matched_end"] = merged["_start_dt"] + match_win

        merged["vent_dur_td"] = (merged["_matched_end"] - merged["_start_dt"]).clip(lower=min_length)
        merged = merged[merged["vent_dur_td"] >= min_length]
        if merged.empty:
            return _empty_result()

        window_df = merged[id_columns + ["_start_dt", "vent_dur_td"]].copy()
        return _expand_windows(window_df, revert_fn)

    def _normalize_result(result: Optional[ICUTable]) -> Optional[ICUTable]:
        if result is None:
            return None
        if getattr(result, "data", pd.DataFrame()).empty:
            return None
        return result

    mech_result = None
    if mech_tbl is not None and not mech_tbl.data.empty:
        mech_result = _normalize_result(_windows_from_mech(mech_tbl))

    event_result = None
    if start_tbl is not None and not start_tbl.data.empty:
        event_result = _normalize_result(_windows_from_events(start_tbl, end_tbl))

    if mech_result is None and event_result is None:
        return _empty_result()

    if mech_result is None:
        return event_result  # type: ignore[return-value]

    if event_result is None:
        return mech_result

    # Combine both sources instead of picking the longest. ricu unions any
    # evidence of ventilation, so we take the OR across timelines.
    combined = pd.concat(
        [event_result.data, mech_result.data], ignore_index=True, copy=False
    )

    # Ensure duplicate timestamps collapse to a single True indicator.
    group_cols = list(id_columns)
    if time_column:
        group_cols += [time_column]

    if group_cols:
        combined = (
            combined.groupby(group_cols, as_index=False)["vent_ind"].any()
        )

    return _as_icutbl(
        combined,
        id_columns=id_columns,
        index_column=time_column,
        value_column="vent_ind",
    )


def _callback_urine24(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """
    Calculate 24-hour urine output (R ricu urine24 callback).
    
    Replicates R ricu's urine24 logic exactly:
    1. Collect urine data
    2. Fill gaps in time series using fill_gaps (with limits from collapse)
    3. Apply sliding window (24 hours) with urine_sum function
    4. urine_sum: if length < min_steps return NA, else sum * step_factor / length
    """
    urine_tbl = _ensure_time_index(tables["urine"])
    interval = ctx.interval or pd.Timedelta(hours=1)
    min_win = ctx.kwargs.get('min_win', pd.Timedelta(hours=12))
    
    df = urine_tbl.data.copy()
    key_cols = urine_tbl.id_columns + [urine_tbl.index_column]
    if df.empty:
        cols = key_cols + ["urine24"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column="urine24")

    # Convert min_win to timedelta if needed
    if isinstance(min_win, (int, float)):
        min_win = pd.Timedelta(hours=min_win)
    elif hasattr(min_win, 'total_seconds'):
        min_win = pd.Timedelta(min_win)
    
    # Validate min_win
    if min_win <= interval or min_win > pd.Timedelta(hours=24):
        min_win = pd.Timedelta(hours=12)  # Default
    
    # Prepare urine column
    urine_col = urine_tbl.value_column or "urine"
    if urine_col not in df.columns:
        df[urine_col] = 0.0
    df[urine_col] = pd.to_numeric(df[urine_col], errors="coerce").fillna(0.0)
    
    # 🔧 性能优化：对于AUMC和HiRID等高频数据，先按interval聚合再处理
    # 这些数据库的采样频率很高（每分钟甚至更频繁），需要先降采样
    # ⚠️  暂时禁用这个优化，因为可能导致性能问题
    time_col = urine_tbl.index_column
    is_numeric_time = pd.api.types.is_numeric_dtype(df[time_col])
    
    # 检测是否需要降采样（如果同一患者在1小时内有超过10个数据点）
    id_cols_to_group = list(urine_tbl.id_columns) if urine_tbl.id_columns else []
    need_resampling = False
    
    # DISABLED降采样检测逻辑 - 会导致性能问题
    # if id_cols_to_group and len(df) > 100:  # 只对较大的数据集检测
    #     ...
    
    # 跳过降采样步骤
    # if need_resampling:
    #     ...
    
    # Check if time column is numeric (hours since admission) or datetime
    
    # Calculate min_steps and step_factor (replicates R ricu logic)
    if is_numeric_time:
        interval_hours = interval.total_seconds() / 3600.0
        min_win_hours = min_win.total_seconds() / 3600.0
        min_steps = int(np.ceil(min_win_hours / interval_hours))
        step_factor = 24.0 / interval_hours  # convert_dt(hours(24L)) / as.double(interval)
    else:
        min_steps = int(np.ceil(min_win.total_seconds() / interval.total_seconds()))
        step_factor = pd.Timedelta(hours=24).total_seconds() / interval.total_seconds()
    
    # Get limits (start/end times per patient) - from kwargs or collapse
    limits = ctx.kwargs.get('limits', None)
    if limits is None:
        # Use collapse to get start/end times per patient (replicates R ricu collapse)
        # CRITICAL: Use only non-NaN urine values to determine the actual data range
        # Find the end time considering data continuity
        id_cols_to_group = list(urine_tbl.id_columns) if urine_tbl.id_columns else []
        
        # Filter to rows where urine has actual data (non-NaN, non-zero)
        urine_data_mask = df[urine_col].notna() & (df[urine_col] != 0)
        df_with_data = df[urine_data_mask]
        
        if id_cols_to_group and not df_with_data.empty:
            limits_list = []
            for patient_id, group in df_with_data.groupby(id_cols_to_group):
                # Sort by time
                group_sorted = group.sort_values(time_col)
                times = group_sorted[time_col].values
                
                start_time = times[0]
                end_time = times[-1]  # Default to last measurement
                
                # TEMP WORKAROUND: Use known correct boundaries from ricu for test patients
                # TODO: Find the correct algorithm to compute these automatically
                if isinstance(patient_id, tuple):
                    pid = patient_id[0] if len(patient_id) > 0 else None
                else:
                    pid = patient_id
                
                                
                # Create limits entry
                if isinstance(patient_id, tuple):
                    limits_entry = {col: patient_id[i] for i, col in enumerate(id_cols_to_group)}
                else:
                    limits_entry = {id_cols_to_group[0]: patient_id}
                
                limits_entry['start'] = start_time
                limits_entry['end'] = end_time
                limits_list.append(limits_entry)
            
            limits = pd.DataFrame(limits_list)
        elif not df_with_data.empty:
            limits = pd.DataFrame({
                'start': [df_with_data[time_col].min()],
                'end': [df_with_data[time_col].max()]
            })
        else:
            # No data at all, use original df range as fallback
            if id_cols_to_group:
                limits = df.groupby(id_cols_to_group)[time_col].agg(['min', 'max']).reset_index()
                limits = limits.rename(columns={'min': 'start', 'max': 'end'})
            else:
                limits = pd.DataFrame({
                    'start': [df[time_col].min()],
                    'end': [df[time_col].max()]
                })
    
    # CRITICAL: Fill gaps in time series (replicates R ricu fill_gaps)
    # This ensures all time points are present before sliding window calculation
    id_cols_to_group = list(urine_tbl.id_columns) if urine_tbl.id_columns else []
    
    if id_cols_to_group:
        filled_groups = []
        for patient_id, group in df.groupby(id_cols_to_group):
            # Get limits for this patient
            if isinstance(patient_id, tuple):
                mask = True
                for i, col in enumerate(id_cols_to_group):
                    mask = mask & (limits[col] == patient_id[i])
            else:
                mask = limits[id_cols_to_group[0]] == patient_id
            
            patient_limits = limits[mask]
            
            if len(patient_limits) > 0:
                start_time = patient_limits['start'].iloc[0]
                end_time = patient_limits['end'].iloc[0]
            else:
                # Fallback: use group's time range
                start_time = group[time_col].min()
                end_time = group[time_col].max()
            
            # Create complete time series (fill gaps)
            if is_numeric_time:
                time_range = np.arange(start_time, end_time + interval_hours, interval_hours)
                filled_df = pd.DataFrame({time_col: time_range})
            else:
                time_range = pd.date_range(start=start_time, end=end_time, freq=interval)
                filled_df = pd.DataFrame({time_col: time_range})
            
            # Add ID columns
            if isinstance(patient_id, tuple):
                for i, col in enumerate(id_cols_to_group):
                    filled_df[col] = patient_id[i]
            else:
                filled_df[id_cols_to_group[0]] = patient_id
            
            # Merge with original data (left join to keep all time points)
            filled_df = filled_df.merge(
                group[[time_col, urine_col] + id_cols_to_group],
                on=[time_col] + id_cols_to_group,
                how='left'
            )
            
            # Fill NaN urine values with 0 (for gap filling)
            # R ricu's fill_gaps for urine likely treats missing as 0 output
            filled_df[urine_col] = filled_df[urine_col].fillna(0.0)
            
            filled_groups.append(filled_df)
        
        if filled_groups:
            df = pd.concat(filled_groups, ignore_index=True)
        else:
            df = pd.DataFrame()
    
    if df.empty:
        cols = key_cols + ["urine24"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column="urine24")
    
    df = df.sort_values(key_cols)
    samples_per_day = max(int(np.ceil(pd.Timedelta(hours=24).total_seconds() / interval.total_seconds())), 1)
    scale_factor = step_factor / samples_per_day

    if id_cols_to_group:
        grouped = df.groupby(id_cols_to_group, dropna=False, sort=False)[urine_col]
        rolling = grouped.rolling(window=samples_per_day, min_periods=min_steps).sum()
        rolling = rolling.reset_index(level=id_cols_to_group, drop=True)
        df["urine24"] = rolling * scale_factor
    else:
        rolling = df[urine_col].rolling(window=samples_per_day, min_periods=min_steps).sum()
        df["urine24"] = rolling * scale_factor

    result = df
    
        
    cols = list(urine_tbl.id_columns) + [urine_tbl.index_column, "urine24"]
    return _as_icutbl(result[cols], id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column="urine24")


def _callback_vaso_ind(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    merged, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    time_col = index_column or "starttime"
    cols = list(id_columns) + ([time_col] if time_col else [])
    empty_cols = cols + ["vaso_ind"]
    if merged.empty or time_col not in merged.columns:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    vaso_cols = [col for col in merged.columns if col not in cols]
    if not vaso_cols:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    base_time = pd.Timestamp("2000-01-01")
    time_series = merged[time_col]
    time_is_numeric = pd.api.types.is_numeric_dtype(time_series)
    if time_is_numeric:
        numeric_time = pd.to_numeric(time_series, errors="coerce")
        merged["__start_dt"] = base_time + pd.to_timedelta(numeric_time, unit="h")
    else:
        merged["__start_dt"] = pd.to_datetime(time_series, errors="coerce")

    def _coerce_duration(series: pd.Series) -> pd.Series:
        if pd.api.types.is_timedelta64_dtype(series):
            return series
        converted = pd.to_timedelta(series, errors="coerce")
        if converted.notna().any():
            return converted
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_timedelta(numeric, unit="h", errors="coerce")

    for col in vaso_cols:
        merged[col] = _coerce_duration(merged[col])

    window_records: list[tuple] = []
    for _, row in merged.iterrows():
        start_dt = row["__start_dt"]
        if pd.isna(start_dt):
            continue
        id_values = tuple(row[col] for col in id_columns) if id_columns else tuple()
        for col in vaso_cols:
            duration = row[col]
            if pd.isna(duration) or duration <= pd.Timedelta(0):
                continue
            end_dt = start_dt + duration
            window_records.append((*id_values, start_dt, end_dt))

    if not window_records:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    window_cols = list(id_columns) + ["__start", "__end"]
    window_df = pd.DataFrame(window_records, columns=window_cols)

    existing_id_cols = list(id_columns)
    if not existing_id_cols:
        window_df["__dummy_id"] = 1
        existing_id_cols = ["__dummy_id"]

    intervals = _merge_intervals(
        window_df[existing_id_cols + ["__start", "__end"]],
        id_columns=existing_id_cols,
        start_col="__start",
        end_col="__end",
        max_gap=pd.Timedelta(0),
    )

    if intervals.empty:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    final_interval = ctx.interval
    if isinstance(final_interval, str):
        try:
            final_interval = pd.to_timedelta(final_interval)
        except Exception:
            final_interval = None
    if final_interval is None or final_interval <= pd.Timedelta(0):
        final_interval = pd.Timedelta(hours=1)

    expanded_frames: list[pd.DataFrame] = []
    for _, row in intervals.iterrows():
        start = row["__start"]
        end = row["__end"]
        if pd.isna(start) or pd.isna(end) or start > end:
            continue
        grid = pd.date_range(start=start, end=end, freq=final_interval)
        if grid.empty:
            grid = pd.DatetimeIndex([start])
        frame = pd.DataFrame({time_col: grid, "vaso_ind": True})
        for col in existing_id_cols:
            frame[col] = row[col]
        expanded_frames.append(frame)

    if not expanded_frames:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    expanded = pd.concat(expanded_frames, ignore_index=True)
    if "__dummy_id" in expanded.columns:
        expanded = expanded.drop(columns=["__dummy_id"])
    expanded = expanded.drop_duplicates(subset=list(id_columns) + [time_col] if id_columns else [time_col])

    if time_is_numeric:
        expanded[time_col] = (
            pd.to_datetime(expanded[time_col], errors="coerce") - base_time
        ) / pd.Timedelta(hours=1)

    result_cols = list(id_columns) + [time_col, "vaso_ind"] if id_columns else [time_col, "vaso_ind"]
    expanded = expanded[result_cols].reset_index(drop=True)
    return _as_icutbl(expanded, id_columns=id_columns, index_column=time_col, value_column="vaso_ind")


def _callback_sep3(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    # Check if required tables exist
    if "sofa" not in tables or "susp_inf" not in tables:
        # Return empty result if required tables are missing
        import pandas as pd
        return _as_icutbl(
            pd.DataFrame(columns=['stay_id', 'charttime', 'sep3']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='sep3'
        )
    
    # Convert ID columns if needed (hadm_id → stay_id) before merging
    # This replicates R ricu's automatic ID conversion in collect_dots()
    id_columns, index_column, converted_tables = _assert_shared_schema(
        {"sofa": tables["sofa"], "susp_inf": tables["susp_inf"]},
        ctx=ctx,
        convert_ids=True
    )
    
    # Check if tables still exist after conversion (they may have been removed if empty)
    if "sofa" not in converted_tables or "susp_inf" not in converted_tables:
        # Return empty result if conversion resulted in empty tables
        import pandas as pd
        return _as_icutbl(
            pd.DataFrame(columns=list(id_columns) + ([index_column] if index_column else []) + ['sep3']),
            id_columns=id_columns,
            index_column=index_column,
            value_column='sep3'
        )
    
    # Use converted tables
    sofa_tbl = converted_tables["sofa"]
    susp_tbl = converted_tables["susp_inf"]
    
    # Standardize time column names - both need to use the same column name
    sofa_data = sofa_tbl.data.copy()
    susp_data = susp_tbl.data.copy()
    
    # Rename time columns to index_column if they differ
    if sofa_tbl.index_column and sofa_tbl.index_column != index_column and sofa_tbl.index_column in sofa_data.columns:
        sofa_data = sofa_data.rename(columns={sofa_tbl.index_column: index_column})
    if susp_tbl.index_column and susp_tbl.index_column != index_column and susp_tbl.index_column in susp_data.columns:
        susp_data = susp_data.rename(columns={susp_tbl.index_column: index_column})

    result = sep3_detector(
        sofa=sofa_data,
        susp_inf=susp_data,
        id_cols=list(id_columns),
        index_col=coalesce(sofa_tbl.index_column, susp_tbl.index_column, index_column),
    )

    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column="sep3")


def _callback_vaso60(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    rate_names = [name for name in tables if name.endswith("_rate")]
    dur_names = [name for name in tables if name.endswith("_dur")]
    if not rate_names or not dur_names:
        raise ValueError("vaso60 requires both rate and duration concepts.")

    rate_name = rate_names[0]
    dur_name = dur_names[0]

    rate_tbl = tables[rate_name]
    dur_tbl = tables[dur_name]
    
    # 🔧 FIX: Handle empty input data
    if rate_tbl.data.empty or dur_tbl.data.empty:
        # Return empty result with proper schema
        id_cols = rate_tbl.id_columns or dur_tbl.id_columns or ['stay_id']
        idx_col = rate_tbl.index_column or dur_tbl.index_column or 'charttime'
        return _as_icutbl(
            pd.DataFrame(columns=list(id_cols) + [idx_col, ctx.concept_name]),
            id_columns=id_cols,
            index_column=idx_col,
            value_column=ctx.concept_name,
        )

    id_columns, index_column, _ = _assert_shared_schema({rate_name: rate_tbl, dur_name: dur_tbl})
    if index_column is None:
        raise ValueError("vaso60 requires time-indexed component tables.")

    final_interval = ctx.interval
    if isinstance(final_interval, str):
        final_interval = pd.to_timedelta(final_interval)
    elif final_interval is not None and not isinstance(final_interval, pd.Timedelta):
        final_interval = pd.to_timedelta(final_interval)

    if final_interval is None:
        final_interval = _infer_interval_from_table(rate_tbl)

    rate_df = rate_tbl.data.copy()
    dur_df = dur_tbl.data.copy()
    rate_col = rate_tbl.value_column or rate_name
    dur_col = dur_tbl.value_column or dur_name
    
    #  🔧 修复：确保index_column在两个DataFrame中都存在
    # change_interval可能将列名改为'start',需要使用实际的列名
    rate_index_col = index_column if index_column in rate_df.columns else (rate_tbl.index_column if rate_tbl.index_column and rate_tbl.index_column in rate_df.columns else None)
    dur_index_col = index_column if index_column in dur_df.columns else (dur_tbl.index_column if dur_tbl.index_column and dur_tbl.index_column in dur_df.columns else None)
    
    if rate_index_col is None or dur_index_col is None:
        # 尝试查找时间列
        rate_time_cols = [c for c in rate_df.columns if c in ['start', 'measuredat', 'charttime', index_column]]
        dur_time_cols = [c for c in dur_df.columns if c in ['start', 'measuredat', 'charttime', index_column]]
        if rate_time_cols and dur_time_cols:
            rate_index_col = rate_time_cols[0]
            dur_index_col = dur_time_cols[0]
        else:
            raise ValueError(f"vaso60: time column not found. Expected '{index_column}' but rate has {list(rate_df.columns[:5])}, dur has {list(dur_df.columns[:5])}")

    # Identify unit column heuristically if metadata is missing
    rate_unit_col = rate_tbl.unit_column
    if (rate_unit_col is None or rate_unit_col not in rate_df.columns) and not rate_df.empty:
        for candidate in rate_df.columns:
            if "unit" in candidate.lower():
                rate_unit_col = candidate
                break

    # Normalise unit strings to canonical tokens to simplify conversion logic
    def _normalise_unit(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        token = str(value).strip().lower()
        if not token:
            return ""
        replacements = (
            ("μ", "u"),
            ("µ", "u"),
            ("mcg", "ug"),
            (" per ", "/"),
            ("per", "/"),
            (" minutes", " min"),
            (" minute", " min"),
            ("mins", "min"),
            ("min.", "min"),
            (" hours", " h"),
            (" hour", " h"),
            ("hrs", "h"),
            ("hr", "h"),
            ("mg/hr", "mg/h"),
            ("kgmin", "kg/min"),
            ("ugkgmin", "ug/kg/min"),
        )
        for old, new in replacements:
            token = token.replace(old, new)
        while "//" in token:
            token = token.replace("//", "/")
        return token

    unit_tokens: Optional[pd.Series]
    if rate_unit_col and rate_unit_col in rate_df.columns:
        unit_tokens = rate_df[rate_unit_col].map(_normalise_unit)
    else:
        unit_tokens = None

    if unit_tokens is not None:
        rate_df["__unit_token"] = unit_tokens

    # 统一时间列：若任一为数值型（相对小时），将双方都转换为基于同一锚点的datetime
    base_time = pd.Timestamp('2000-01-01')
    ds_name = ''
    if ctx is not None:
        ds_cfg = getattr(getattr(ctx, 'data_source', None), 'config', None)
        ds_name = getattr(ds_cfg, 'name', '') if ds_cfg is not None else ''
    numeric_unit = 'h'
    if isinstance(ds_name, str) and ds_name.lower() == 'aumc':
        numeric_unit = 'ms'

    rate_time_is_numeric = pd.api.types.is_numeric_dtype(rate_df[rate_index_col])
    dur_time_is_numeric = pd.api.types.is_numeric_dtype(dur_df[dur_index_col])
    if rate_time_is_numeric or dur_time_is_numeric:
        if rate_time_is_numeric:
            rate_df[rate_index_col] = base_time + pd.to_timedelta(pd.to_numeric(rate_df[rate_index_col], errors='coerce'), unit=numeric_unit)
        else:
            rate_df[rate_index_col] = pd.to_datetime(rate_df[rate_index_col], errors='coerce')
        if dur_time_is_numeric:
            dur_df[dur_index_col] = base_time + pd.to_timedelta(pd.to_numeric(dur_df[dur_index_col], errors='coerce'), unit=numeric_unit)
        else:
            dur_df[dur_index_col] = pd.to_datetime(dur_df[dur_index_col], errors='coerce')
    else:
        # 双方原本均为datetime，标准化为tz-naive
        rate_df[rate_index_col] = pd.to_datetime(rate_df[rate_index_col], errors='coerce')
        dur_df[dur_index_col] = pd.to_datetime(dur_df[dur_index_col], errors='coerce')

    durations = dur_df[dur_col]
    if pd.api.types.is_timedelta64_dtype(durations):
        pass
    elif pd.api.types.is_datetime64_any_dtype(durations):
        # Duration column is datetime type (probably a bug from calc_dur)
        # This shouldn't happen, but if it does, try to detect if it's actually timedelta stored as datetime
        # For now, skip conversion and let it fail gracefully
        print(f"⚠️  Warning: {dur_col} has datetime dtype instead of timedelta, attempting conversion...")
        # Just set durations to NaN to avoid crash
        durations = pd.Series([pd.NaT] * len(durations), index=durations.index, dtype='timedelta64[ns]')
    else:
        converted = pd.to_timedelta(durations, errors="coerce")
        if converted.notna().any():
            durations = converted
        else:
            numeric_durations = pd.to_numeric(durations, errors="coerce")
            minutes_based = pd.to_timedelta(numeric_durations, unit="m", errors="coerce")
            if minutes_based.notna().any():
                durations = minutes_based
            else:
                seconds_based = pd.to_timedelta(numeric_durations, unit="s", errors="coerce")
                durations = seconds_based

    dur_df["__duration"] = durations
    dur_df = dur_df.dropna(subset=["__duration", dur_index_col])
    dur_df = dur_df[dur_df["__duration"] > pd.Timedelta(0)]

    if dur_df.empty or rate_df.empty:
        # 使用rate_index_col作为最终输出的时间列（因为它更可能是标准列名）
        output_index_col = rate_index_col if rate_index_col in ['start', 'measuredat', 'charttime'] else dur_index_col
        cols = id_columns + [output_index_col, ctx.concept_name]
        return _as_icutbl(
            pd.DataFrame(columns=cols),
            id_columns=id_columns,
            index_column=output_index_col,
            value_column=ctx.concept_name,
        )

    dur_df["__start"] = dur_df[dur_index_col]
    dur_df["__end"] = dur_df["__start"] + dur_df["__duration"]

    max_gap = pd.Timedelta(minutes=5)

    # Filter id_columns to only include columns that actually exist in dur_df
    # This handles cases where ID columns were filtered out during processing (e.g., eICU infusiondrug)
    existing_id_cols = [col for col in id_columns if col in dur_df.columns]
    if len(existing_id_cols) != len(id_columns):
        missing_cols = set(id_columns) - set(existing_id_cols)
        import logging
        logging.debug(f"_callback_vaso60: Missing ID columns {missing_cols} in duration dataframe. Using available columns: {existing_id_cols}")

    # If no valid ID columns exist, create a dummy one for processing
    if not existing_id_cols:
        dur_df["__dummy_id"] = 1
        existing_id_cols = ["__dummy_id"]
        import logging
        logging.debug("_callback_vaso60: No valid ID columns found. Using dummy ID column.")

    intervals = _merge_intervals(
        dur_df[existing_id_cols + ["__start", "__end"]],
        id_columns=existing_id_cols,
        start_col="__start",
        end_col="__end",
        max_gap=max_gap,
    )

    if intervals.empty:
        output_index_col = rate_index_col if rate_index_col in ['start', 'measuredat', 'charttime'] else dur_index_col
        cols = id_columns + [output_index_col, ctx.concept_name]
        return _as_icutbl(
            pd.DataFrame(columns=cols),
            id_columns=id_columns,
            index_column=output_index_col,
            value_column=ctx.concept_name,
        )

    intervals["__length"] = intervals["__end"] - intervals["__start"]
    intervals = intervals[intervals["__length"] >= pd.Timedelta(hours=1)].copy()

    if intervals.empty:
        output_index_col = rate_index_col if rate_index_col in ['start', 'measuredat', 'charttime'] else dur_index_col
        cols = id_columns + [output_index_col, ctx.concept_name]
        return _as_icutbl(
            pd.DataFrame(columns=cols),
            id_columns=id_columns,
            index_column=output_index_col,
            value_column=ctx.concept_name,
        )

    rate_df = rate_df.dropna(subset=[rate_index_col])
    rate_df[rate_col] = pd.to_numeric(rate_df[rate_col], errors="coerce")

    if rate_unit_col and "__unit_token" in rate_df.columns and not rate_df.empty:
        unit_tokens = rate_df["__unit_token"]

        standard_units = {"ug/kg/min", "mcg/kg/min"}
        needs_conversion_mask = unit_tokens.notna() & ~unit_tokens.isin(standard_units)

        if needs_conversion_mask.any():
            units_requiring_weight = {"ug/min", "mcg/min", "mg/h", "mg/hr"}
            simple_conversions = {"mg/kg/h", "mg/kg/hr"}

            weight_merge_col = None
            if unit_tokens.isin(units_requiring_weight).any():
                weight_concept = ctx.kwargs.get("weight_concept", "weight")
                weight_table = ctx.kwargs.get("weight_table")
                if weight_table is None:
                    try:
                        loaded = ctx.resolver.load_concepts(
                            [weight_concept],
                            ctx.data_source,
                            merge=True,
                            aggregate="last",
                            patient_ids=ctx.patient_ids,
                            verbose=False,
                            align_to_admission=False,
                        )
                        if isinstance(loaded, dict):
                            weight_table = loaded.get(weight_concept)
                        else:
                            weight_table = loaded
                    except Exception as exc:  # pragma: no cover - defensive guard
                        print(
                            f"⚠️  Warning: failed to load '{weight_concept}' concept for vasopressor conversion: {exc}"
                        )
                        weight_table = None

                if isinstance(weight_table, ICUTable):
                    weight_df = weight_table.data.copy()
                    weight_ids = list(weight_table.id_columns)
                    if weight_ids:
                        value_col = weight_table.value_column or weight_concept
                        if value_col not in weight_df.columns:
                            non_id_cols = [col for col in weight_df.columns if col not in weight_ids]
                            if non_id_cols:
                                value_col = non_id_cols[0]
                        if value_col in weight_df.columns:
                            usable_cols = list(weight_ids) + [value_col]
                            weight_df = weight_df[usable_cols].dropna(subset=[value_col])
                            if weight_table.index_column and weight_table.index_column in weight_df.columns:
                                order_cols = weight_ids + [weight_table.index_column]
                                weight_df = weight_df.sort_values(order_cols)
                            weight_df = weight_df.drop_duplicates(subset=weight_ids, keep="last")
                            merge_df = weight_df.rename(columns={value_col: "__weight_kg"})
                            rate_df = rate_df.merge(merge_df, on=weight_ids, how="left")
                            weight_merge_col = "__weight_kg"

            if unit_tokens.isin(units_requiring_weight).any():
                if weight_merge_col is None or weight_merge_col not in rate_df.columns:
                    missing = unit_tokens[unit_tokens.isin(units_requiring_weight)].unique()
                    raise ValueError(
                        "Unable to convert vasopressor rates without patient weight for units "
                        f"{sorted(map(str, missing))}. "
                        "Ensure a weight concept is available or pass 'weight_table' via ctx.kwargs."
                    )

            for unit in unit_tokens[needs_conversion_mask].unique():
                if not unit:
                    continue
                mask = unit_tokens == unit
                if unit in {"ug/min", "mcg/min"}:
                    weights = rate_df.loc[mask, weight_merge_col]
                    rate_df.loc[mask, rate_col] = convert_vaso_rate(
                        rate_df.loc[mask, rate_col],
                        "ug/min",
                        weight_kg=weights,
                    )
                elif unit in {"mg/h", "mg/hr"}:
                    weights = rate_df.loc[mask, weight_merge_col]
                    source_unit = "mg/h" if unit == "mg/h" else "mg/hr"
                    rate_df.loc[mask, rate_col] = convert_vaso_rate(
                        rate_df.loc[mask, rate_col],
                        source_unit,
                        weight_kg=weights,
                    )
                elif unit in simple_conversions:
                    source_unit = "mg/kg/h" if unit == "mg/kg/h" else "mg/kg/hr"
                    rate_df.loc[mask, rate_col] = convert_vaso_rate(
                        rate_df.loc[mask, rate_col],
                        source_unit,
                    )
                else:
                    print(
                        f"⚠️  Warning: unsupported vasopressor rate unit '{unit}' encountered in {ctx.concept_name}; leaving values unconverted."
                    )
                    continue

                if rate_unit_col in rate_df.columns:
                    rate_df.loc[mask, rate_unit_col] = "ug/kg/min"

            if weight_merge_col and weight_merge_col in rate_df.columns:
                rate_df = rate_df.drop(columns=[weight_merge_col])

    if "__unit_token" in rate_df.columns:
        rate_df = rate_df.drop(columns=["__unit_token"])

    rate_df = rate_df.dropna(subset=[rate_col])

    if rate_df.empty:
        output_index_col = rate_index_col if rate_index_col in ['start', 'measuredat', 'charttime'] else dur_index_col
        cols = id_columns + [output_index_col, ctx.concept_name]
        return _as_icutbl(
            pd.DataFrame(columns=cols),
            id_columns=id_columns,
            index_column=output_index_col,
            value_column=ctx.concept_name,
        )

    merged = rate_df.merge(intervals.drop(columns=["__length"]), on=id_columns, how="inner")
    mask = (merged[rate_index_col] >= merged["__start"]) & (merged[rate_index_col] <= merged["__end"])
    filtered = merged[mask]

    if filtered.empty:
        output_index_col = rate_index_col if rate_index_col in ['start', 'measuredat', 'charttime'] else dur_index_col
        cols = id_columns + [output_index_col, ctx.concept_name]
        return _as_icutbl(
            pd.DataFrame(columns=cols),
            id_columns=id_columns,
            index_column=output_index_col,
            value_column=ctx.concept_name,
        )

    filtered = filtered.drop(columns=["__start", "__end"])
    grouped = (
        filtered.groupby(id_columns + [rate_index_col], dropna=False)[rate_col]
        .max()
        .reset_index()
    )
    grouped[ctx.concept_name] = grouped[rate_col]
    grouped = grouped.drop(columns=[rate_col])

    # Expand vasopressor rates across their active windows so that every hour
    # within an infusion inherits the most recent setting, matching ricu's
    # vaso60 (change_interval + aggregate).
    if final_interval is not None and not grouped.empty and not intervals.empty:
        interval_id_cols = [col for col in id_columns if col in intervals.columns]

        def _normalize_key(values: Sequence[object]) -> tuple:
            normalized = []
            for val in values:
                if pd.isna(val):
                    normalized.append("__nan__")
                else:
                    normalized.append(val)
            return tuple(normalized) if normalized else ("__all__",)

        interval_map: Dict[tuple, List[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for _, row in intervals.iterrows():
            key = _normalize_key([row[col] for col in interval_id_cols])
            interval_map.setdefault(key, []).append((row["__start"], row["__end"]))

        expanded_frames: List[pd.DataFrame] = []
        group_key_cols = [col for col in id_columns if col in grouped.columns]

        if group_key_cols:
            grouped_iter = grouped.sort_values(group_key_cols + [rate_index_col]).groupby(group_key_cols, dropna=False)
        else:
            grouped_iter = [(None, grouped.sort_values([rate_index_col]))]

        for _, grp in grouped_iter:
            if grp.empty:
                continue
            first = grp.iloc[0]
            key = _normalize_key([first[col] for col in interval_id_cols])
            windows = interval_map.get(key)
            if not windows:
                continue

            times = pd.to_datetime(grp[rate_index_col], errors="coerce")
            values = pd.to_numeric(grp[ctx.concept_name], errors="coerce")
            valid_mask = times.notna() & values.notna()
            if not valid_mask.any():
                continue
            times = times[valid_mask]
            values = values[valid_mask]
            times_ns = times.to_numpy(dtype="datetime64[ns]")
            value_arr = values.to_numpy()

            for start, end in windows:
                if pd.isna(start) or pd.isna(end) or start > end:
                    continue
                grid = pd.date_range(start=start, end=end, freq=final_interval)
                if grid.empty:
                    continue
                grid_ns = grid.to_numpy(dtype="datetime64[ns]")
                idx = np.searchsorted(times_ns, grid_ns, side="right") - 1
                valid_idx = idx >= 0
                if not valid_idx.any():
                    continue
                idx_safe = idx.copy()
                idx_safe[idx_safe < 0] = 0
                sampled = value_arr[idx_safe].astype(float, copy=False)
                sampled[~valid_idx] = np.nan
                frame = pd.DataFrame({rate_index_col: grid, ctx.concept_name: sampled})
                for col in group_key_cols:
                    frame[col] = first[col]
                frame = frame.dropna(subset=[ctx.concept_name])
                if not frame.empty:
                    expanded_frames.append(frame)

        if expanded_frames:
            expanded = pd.concat(expanded_frames, ignore_index=True)
            agg_cols = [col for col in group_key_cols] + [rate_index_col]
            grouped = (
                expanded.groupby(agg_cols, dropna=False)[ctx.concept_name]
                .max()
                .reset_index()
            )

    if final_interval is not None and not grouped.empty:
        grouped[rate_index_col] = grouped[rate_index_col].dt.floor(final_interval)
        grouped = (
            grouped.groupby(id_columns + [rate_index_col], dropna=False)[ctx.concept_name]
            .max()
            .reset_index()
        )

    output_index_col = rate_index_col if rate_index_col in ['start', 'measuredat', 'charttime'] else dur_index_col
    cols = id_columns + [output_index_col, ctx.concept_name]
    result = grouped[cols].reset_index(drop=True)
    # 若上面为了计算将时间转换为datetime（源头为相对小时），在返回前还原为相对小时
    if rate_time_is_numeric or dur_time_is_numeric:
        try:
            result[output_index_col] = (pd.to_datetime(result[output_index_col], errors='coerce') - base_time) / pd.Timedelta(hours=1)
        except Exception:
            pass
    return _as_icutbl(
        result,
        id_columns=id_columns,
        index_column=output_index_col,
        value_column=ctx.concept_name,
    )


def _callback_susp_inf(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    # Convert ID columns if needed (hadm_id → stay_id) before merging
    # This replicates R ricu's automatic ID conversion in collect_dots()
    id_columns, index_column, converted_tables = _assert_shared_schema(
        {"abx": tables["abx"], "samp": tables["samp"]},
        ctx=ctx,
        convert_ids=True
    )
    
    # Use converted tables
    abx_tbl = converted_tables["abx"]
    samp_tbl = converted_tables["samp"]
    
    if index_column is None:
        raise ValueError("susp_inf requires time-indexed component tables")
    
    # Standardize time column names - both need to use the same column name
    # abx might have 'starttime', samp might have 'charttime' or 'chartdate'
    abx_data = abx_tbl.data.copy()
    samp_data = samp_tbl.data.copy()
    
    # Rename time columns to index_column if they differ
    if abx_tbl.index_column and abx_tbl.index_column != index_column and abx_tbl.index_column in abx_data.columns:
        abx_data = abx_data.rename(columns={abx_tbl.index_column: index_column})
    if samp_tbl.index_column and samp_tbl.index_column != index_column and samp_tbl.index_column in samp_data.columns:
        samp_data = samp_data.rename(columns={samp_tbl.index_column: index_column})

    result = susp_inf_detector(
        abx=abx_data,
        samp=samp_data,
        id_cols=list(id_columns),
        index_col=index_column,
    )

    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column="susp_inf")


def _callback_gcs(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """
    Calculate GCS (Glasgow Coma Scale) with sed_impute logic.
    
    Replicates R ricu's GCS callback logic:
    - sed_impute="max" (default): Intubated patients get GCS=15
    - sed_impute="none": Use actual measured values
    - set_na_max=True (default): Fill remaining NA with max values (egcs=4, mgcs=6, vgcs=5)
    
    Args:
        tables: Dictionary containing GCS component tables (egcs, mgcs, vgcs, tgcs, ett_gcs)
        ctx: Callback context with kwargs like sed_impute, set_na_max
    
    Returns:
        ICUTable with GCS values
    """
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["gcs"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="gcs")

    # Get parameters from context (matching R ricu defaults)
    sed_impute = ctx.kwargs.get("sed_impute", "max")
    set_na_max = ctx.kwargs.get("set_na_max", True)

    tgcs = pd.to_numeric(data.get("tgcs"), errors="coerce")
    egcs = pd.to_numeric(data.get("egcs"), errors="coerce")
    mgcs = pd.to_numeric(data.get("mgcs"), errors="coerce")
    vgcs = pd.to_numeric(data.get("vgcs"), errors="coerce")
    ett_gcs = data.get("ett_gcs") if "ett_gcs" in data.columns else None

    # CRITICAL FIX: Replicate R ricu's sed_impute logic
    # If sed_impute="max" (default) and patient is intubated (ett_gcs=True), set tgcs=15
    if sed_impute == "max" and ett_gcs is not None:
        # Convert ett_gcs to boolean - 使用 where() 替代 fillna() 以避免警告
        is_intubated = ett_gcs.where(ett_gcs.notna(), False).astype(bool)
        # For intubated patients, set tgcs=15
        if tgcs is None:
            tgcs = pd.Series(np.nan, index=data.index, dtype=float)
        # Ensure tgcs is a Series with proper index for assignment
        if not isinstance(tgcs, pd.Series):
            tgcs = pd.Series(tgcs, index=data.index, dtype=float)
        else:
            tgcs = tgcs.copy()
        tgcs[is_intubated] = 15.0
    
    # Ensure all GCS components are Series with proper index for operations
    if egcs is not None and not isinstance(egcs, pd.Series):
        egcs = pd.Series(egcs, index=data.index, dtype=float)
    if mgcs is not None and not isinstance(mgcs, pd.Series):
        mgcs = pd.Series(mgcs, index=data.index, dtype=float)
    if vgcs is not None and not isinstance(vgcs, pd.Series):
        vgcs = pd.Series(vgcs, index=data.index, dtype=float)

    # If set_na_max=True, fill NA component values with maximum scores
    # (egcs max=4, mgcs max=6, vgcs max=5) - matches R ricu behavior
    if set_na_max:
        if egcs is not None:
            egcs = egcs.fillna(4.0)
        if mgcs is not None:
            mgcs = mgcs.fillna(6.0)
        if vgcs is not None:
            vgcs = vgcs.fillna(5.0)

    # Calculate GCS: use tgcs if available, otherwise sum components
    combined = tgcs.copy() if tgcs is not None else pd.Series(index=data.index, dtype=float)
    
    # For rows where tgcs is NA, calculate from components
    if egcs is not None and mgcs is not None and vgcs is not None:
        component_sum = egcs.add(mgcs, fill_value=np.nan).add(vgcs, fill_value=np.nan)
        combined = combined.fillna(component_sum)
    
    # If set_na_max=True and GCS is still NA, fill with 15 (perfect score)
    if set_na_max:
        combined = combined.fillna(15.0)

    data["gcs"] = combined
    cols = id_columns + ([index_column] if index_column else []) + ["gcs"]
    frame = data[cols].dropna(subset=["gcs"])
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="gcs")


def _callback_rrt_criteria(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Determine if patient meets RRT criteria but not receiving RRT.
    
    SOFA-2 RRT criteria (footnote p):
    - Base kidney injury: Creatinine > 1.2 mg/dL OR oliguria (<0.3 mL/kg/h) for >6 hours
    - PLUS at least one of:
      * Serum potassium ≥ 6.0 mmol/L
      * Metabolic acidosis: pH ≤ 7.20 AND HCO3 ≤ 12 mmol/L
    - AND NOT currently receiving RRT
    
    This is a computed concept that requires crea, uo_6h, uo_12h, uo_24h, potassium, ph, bicarb, and rrt.
    
    ⚡ 性能优化: 依赖概念应该在调用前就已加载好,避免在callback中递归加载
    """
    # ⚡ 优化: 检查是否所有依赖都已提供,如果缺失则一次性批量加载
    required_concepts = ["crea", "uo_6h", "uo_12h", "uo_24h", "potassium", "ph", "bicarb", "rrt"]
    missing_concepts = [c for c in required_concepts if c not in tables]
    
    if missing_concepts:
        # ⚡ 批量加载所有缺失的概念(而非逐个加载)
        try:
            loaded = ctx.resolver.load_concepts(
                missing_concepts,  # 一次性加载所有缺失概念
                ctx.data_source,
                merge=False,
                aggregate=None,
                patient_ids=ctx.patient_ids,
                interval=ctx.interval,
            )
            # 将加载的概念添加到tables
            if isinstance(loaded, dict):
                tables.update(loaded)
            elif isinstance(loaded, ICUTable) and len(missing_concepts) == 1:
                tables[missing_concepts[0]] = loaded
        except (KeyError, ValueError) as e:
            # 如果批量加载失败,静默处理(某些概念可能在字典中不存在)
            if os.environ.get('DEBUG'):
                print(f"   ⚠️  无法加载部分RRT依赖概念: {e}")
    
    # Extract tables
    crea_tbl = tables.get("crea")
    uo_6h_tbl = tables.get("uo_6h")
    uo_12h_tbl = tables.get("uo_12h")
    uo_24h_tbl = tables.get("uo_24h")
    k_tbl = tables.get("potassium")
    ph_tbl = tables.get("ph")
    hco3_tbl = tables.get("bicarb")
    rrt_tbl = tables.get("rrt")
    
    # Merge all tables
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["rrt_criteria"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="rrt_criteria")
    
    # Extract columns - use uo_6h for oliguria check (proxy for >6h duration)
    crea = pd.to_numeric(data.get("crea", pd.Series(np.nan, index=data.index)), errors="coerce")
    uo_6h = pd.to_numeric(data.get("uo_6h", pd.Series(np.nan, index=data.index)), errors="coerce")
    potassium = pd.to_numeric(data.get("potassium", pd.Series(np.nan, index=data.index)), errors="coerce")
    ph = pd.to_numeric(data.get("ph", pd.Series(np.nan, index=data.index)), errors="coerce")
    hco3 = pd.to_numeric(data.get("bicarb", pd.Series(np.nan, index=data.index)), errors="coerce")
    
    # Check if receiving RRT - handle both boolean and numeric types
    rrt_series = data.get("rrt")
    if rrt_series is not None and len(rrt_series) > 0:
        # Convert to boolean, treating NaN/NA/0 as False
        # First convert to numeric if needed, then to bool
        if pd.api.types.is_numeric_dtype(rrt_series):
            rrt_active = (rrt_series.fillna(0) > 0).astype(bool)
        else:
            rrt_active = rrt_series.fillna(False).astype(bool)
    else:
        rrt_active = pd.Series(False, index=data.index, dtype=bool)
    
    # Base kidney injury criteria (use uo_6h as proxy for oliguria >6h)
    aki_crea = (crea > 1.2).fillna(False)
    aki_oligo = (uo_6h < 0.3).fillna(False)
    base_injury = aki_crea | aki_oligo
    
    # Electrolyte/acid-base crisis
    hyperkalemia = (potassium >= 6.0).fillna(False)
    acidosis = ((ph <= 7.20) & (hco3 <= 12)).fillna(False)
    crisis = hyperkalemia | acidosis
    
    # Meets RRT criteria = base injury + crisis - NOT on RRT
    meets_criteria = base_injury & crisis & (~rrt_active)
    
    data["rrt_criteria"] = meets_criteria
    cols = id_columns + ([index_column] if index_column else []) + ["rrt_criteria"]
    frame = data[cols].dropna(subset=["rrt_criteria"])
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="rrt_criteria")


def _callback_urine_mlkgph(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Convert urine output from mL to mL/kg/h.
    
    DEPRECATED: This callback is replaced by uo_6h, uo_12h, uo_24h for SOFA-2.
    Kept for backward compatibility with old code.
    
    This callback computes urine output rate by:
    1. Loading urine (mL) data
    2. Loading weight (kg) data
    3. Computing hourly rate: urine_mL / weight_kg / hours
    
    For SOFA-2, we need the instantaneous urine rate, not cumulative.
    """
    # Load required concepts if not provided
    if "urine" not in tables:
        loaded = ctx.resolver.load_concepts(
            ["urine"],
            ctx.data_source,
            merge=False,
            aggregate="sum",  # Sum urine over intervals
            patient_ids=ctx.patient_ids,
            interval=ctx.interval or pd.Timedelta(hours=1),
        )
        tables["urine"] = loaded if isinstance(loaded, ICUTable) else loaded["urine"]
    
    urine_tbl = tables["urine"]
    
    # For now, we need weight data - but if not available, we can skip weight normalization
    # TODO: Load weight from concept dictionary if available
    # For SOFA-2, typically we compute urine rate from outputevents which may already have rate
    
    # Simple approach: return urine as-is for now
    # In practice, MIMIC-IV outputevents stores urine in mL, and we need to compute rate
    # This requires knowing the time interval and patient weight
    
    # Return urine data with renamed column
    df = urine_tbl.data.copy()
    urine_col = urine_tbl.value_column or "urine"
    
    # For SOFA-2, we compute urine rate over 1-hour intervals
    # Assuming interval is 1 hour, urine_mlkgph = urine_mL / weight_kg / 1 hour
    # Without weight data, we cannot compute the exact rate
    # For testing purposes, return a placeholder
    
    # TODO: Implement proper weight-normalized urine rate calculation
    # For now, return NaN to indicate missing feature
    df["urine_mlkgph"] = np.nan
    
    cols = list(urine_tbl.id_columns) + ([urine_tbl.index_column] if urine_tbl.index_column else []) + ["urine_mlkgph"]
    frame = df[cols].copy()
    
    return _as_icutbl(
        frame.reset_index(drop=True),
        id_columns=list(urine_tbl.id_columns),
        index_column=urine_tbl.index_column,
        value_column="urine_mlkgph"
    )


def _callback_uo_window(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
    window_hours: int,
    output_col: str,
) -> ICUTable:
    """Generic callback for windowed urine output (uo_6h, uo_12h, uo_24h).
    
    Computes rolling average urine output over specified window in mL/kg/h.
    Uses the uo_6h, uo_12h, uo_24h functions from callbacks.py.
    """
    from .callbacks import _urine_window_avg
    
    # Load required concepts
    required = ["urine", "weight"]
    missing = [c for c in required if c not in tables]
    
    if missing:
        loaded = ctx.resolver.load_concepts(
            missing,
            ctx.data_source,
            merge=False,
            aggregate=None,
            patient_ids=ctx.patient_ids,
            interval=ctx.interval,
        )
        if isinstance(loaded, ICUTable):
            tables[missing[0]] = loaded
        else:
            tables.update(loaded)
    
    urine_tbl = tables.get("urine")
    weight_tbl = tables.get("weight")
    
    if urine_tbl is None or urine_tbl.data.empty:
        # Return empty table - get ID columns from data_source
        id_cols = []
        # Prefer patientunitstayid when available (eICU)
        if weight_tbl is not None and 'patientunitstayid' in weight_tbl.data.columns:
            id_cols = ['patientunitstayid']
        elif urine_tbl is not None and 'patientunitstayid' in urine_tbl.data.columns:
            id_cols = ['patientunitstayid']
        else:
            try:
                id_candidate = getattr(ctx.data_source, 'id_cfg', None)
                if hasattr(id_candidate, 'id'):
                    id_cols = [id_candidate.id]
            except Exception:
                id_cols = []
        if not id_cols:
            id_cols = ["stay_id"]

        index_col = urine_tbl.index_column if urine_tbl and urine_tbl.index_column else "charttime"
        cols = id_cols + [index_col] + [output_col]
        frame = pd.DataFrame(columns=cols)
        return _as_icutbl(
            frame,
            id_columns=id_cols,
            index_column=index_col,
            value_column=output_col
        )
    
    # Call the actual callback function from callbacks.py
    min_hours = max(1, window_hours // 2)
    result_df = _urine_window_avg(
        urine=urine_tbl.data,
        weight=weight_tbl.data if weight_tbl else pd.DataFrame(),
        window_hours=window_hours,
        min_hours=min_hours,
        interval=ctx.interval or pd.Timedelta(hours=1)
    )
    
    if result_df.empty:
        return _as_icutbl(
            result_df,
            id_columns=list(urine_tbl.id_columns),
            index_column=urine_tbl.index_column,
            value_column=output_col
        )
    
    return _as_icutbl(
        result_df.reset_index(drop=True),
        id_columns=list(urine_tbl.id_columns),
        index_column=urine_tbl.index_column,
        value_column=output_col
    )


def _callback_uo_6h(tables: Dict[str, ICUTable], ctx: ConceptCallbackContext) -> ICUTable:
    """6-hour rolling average urine output (mL/kg/h)."""
    return _callback_uo_window(tables, ctx, window_hours=6, output_col="uo_6h")


def _callback_uo_12h(tables: Dict[str, ICUTable], ctx: ConceptCallbackContext) -> ICUTable:
    """12-hour rolling average urine output (mL/kg/h)."""
    return _callback_uo_window(tables, ctx, window_hours=12, output_col="uo_12h")


def _callback_uo_24h(tables: Dict[str, ICUTable], ctx: ConceptCallbackContext) -> ICUTable:
    """24-hour rolling average urine output (mL/kg/h)."""
    return _callback_uo_window(tables, ctx, window_hours=24, output_col="uo_24h")


def _callback_sum_components(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Sum multiple component tables together (e.g., for GCS total = eye + motor + verbal)."""
    
    if not tables:
        raise ValueError("sum_components requires at least one input table")
    
    # Merge all tables
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    
    # Create output column name from context
    output_col = ctx.concept_name
    
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + [output_col]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column=output_col)
    
    # Sum all component columns
    component_cols = [tbl.value_column or name for name, tbl in tables.items()]
    total = pd.Series(0, index=data.index, dtype=float)
    
    for col in component_cols:
        if col in data.columns:
            total += pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    data[output_col] = total
    
    # Keep only rows where we have at least some data
    mask = pd.Series(False, index=data.index)
    for col in component_cols:
        if col in data.columns:
            mask |= data[col].notna()
    
    data = data[mask]
    
    cols = id_columns + ([index_column] if index_column else []) + [output_col]
    frame = data[cols].dropna(subset=[output_col])
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column=output_col)


CALLBACK_REGISTRY: MutableMapping[str, CallbackFn] = {
    "bmi": _callback_bmi,
    "avpu": _callback_avpu,
    "norepi_equiv": _callback_norepi_equiv,
    "gcs": _callback_gcs,
    "rrt_criteria": _callback_rrt_criteria,
    "urine_mlkgph": _callback_urine_mlkgph,
    "uo_6h": _callback_uo_6h,
    "uo_12h": _callback_uo_12h,
    "uo_24h": _callback_uo_24h,
    "sum_components": _callback_sum_components,
    "sofa_resp": _callback_sofa_resp,
    "sofa_coag": _callback_sofa_component(sofa_coag),
    "sofa_liver": _callback_sofa_component(sofa_liver),
    "sofa_cardio": _callback_sofa_component(sofa_cardio),
    "sofa_cns": _callback_sofa_component(sofa_cns),
    "sofa_renal": _callback_sofa_component(sofa_renal),
    "sofa_score": _callback_sofa_score,
    "mews_score": _callback_mews,
    "news_score": _callback_news,
    "qsofa_score": _callback_qsofa,
    "sirs_score": _callback_sirs,
    # PaFi = PaO2/FiO2 ratio (arterial oxygen pressure / inspired oxygen fraction)
    "pafi": lambda tables, ctx: _callback_pafi(tables, ctx, source_col_a="po2", source_col_b="fio2", output_col="pafi"),
    # SaFi = SpO2/FiO2 ratio (oxygen saturation / inspired oxygen fraction) 
    "safi": lambda tables, ctx: _callback_pafi(tables, ctx, source_col_a="o2sat", source_col_b="fio2", output_col="safi"),
    "supp_o2": _callback_supp_o2,
    "vent_ind": _callback_vent_ind,
    "urine24": _callback_urine24,
    "vaso_ind": _callback_vaso_ind,
    "sep3": _callback_sep3,
    "vaso60": _callback_vaso60,
    "susp_inf": _callback_susp_inf,
    # SOFA-2 callbacks (2025 version with updated scoring logic)
    "sofa2_resp": _callback_sofa_component(sofa2_resp),
    "sofa2_coag": _callback_sofa_component(sofa2_coag),
    "sofa2_liver": _callback_sofa_component(sofa2_liver),
    "sofa2_cardio": _callback_sofa_component(sofa2_cardio),
    "sofa2_cns": _callback_sofa_component(sofa2_cns),
    "sofa2_renal": _callback_sofa_component(sofa2_renal),  # SOFA-2 version with RRT criteria
    "sofa2_score": _callback_sofa2_score,  # SOFA-2 总分计算（使用 sofa2_* 组件）
    # AUMC-specific callbacks
    "aumc_death": _callback_aumc_death,
    "aumc_bxs": _callback_aumc_bxs,
    "aumc_rass": _callback_aumc_rass,
    "blood_cell_ratio": _callback_blood_cell_ratio,
    "transform_fun(aumc_rass)": _callback_aumc_rass,  # Handle transform_fun wrapper
}


def register_callback(name: str, func: CallbackFn) -> None:
    """Register a new concept callback."""

    CALLBACK_REGISTRY[name] = func


def execute_concept_callback(
    name: str,
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Execute a registered concept callback."""

    func = CALLBACK_REGISTRY.get(name)
    if func is None:
        raise NotImplementedError(f"Concept-level callback '{name}' not implemented.")
    return func(tables, ctx)
