"""High level concept callback implementations (R ricu callback-cncpt.R).

This module provides concept-level aggregation utilities that operate on
collections of :class:`~easyicu.table.ICUTable` objects as produced by the
concept resolver.  Each callback mirrors the behaviour of its R counterpart
well enough for the packaged concept dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional
import logging
import os

import numpy as np
import pandas as pd

# Debug mode flag - can be set to True for verbose debugging output
DEBUG_MODE = False

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
)
from .sepsis import sep3 as sep3_detector, susp_inf as susp_inf_detector
from .sepsis_sofa2 import sep3_sofa2 as sep3_sofa2_detector
from .table import ICUTable, WinTbl
from .utils import coalesce, compute_patient_ids_hash as _compute_patient_ids_hash  # 🔧 统一的 patient_ids hash 函数

logger = logging.getLogger(__name__)
_SUSP_INF_UNSUPPORTED_WARNED: set[str] = set()

from .unit_conversion import convert_vaso_rate

def _standardize_fio2_units(fio2_df: pd.DataFrame, fio2_col: str, database: str) -> pd.DataFrame:
    """将FiO2标准化为百分比形式（0-100）以实现跨数据库兼容性

    Args:
        fio2_df: FiO2数据DataFrame
        fio2_col: FiO2列名
        database: 数据库名称

    Returns:
        标准化后的DataFrame
    """
    if fio2_df.empty or fio2_col not in fio2_df.columns:
        return fio2_df

    # 创建副本避免修改原数据
    result_df = fio2_df.copy()

    # 获取非空的FiO2值进行分析
    fio2_values = result_df[fio2_col].dropna()

    if len(fio2_values) == 0:
        return result_df

    max_val = fio2_values.max()
    min_val = fio2_values.min()

    # 数据库特定的单位转换逻辑
    if database.lower() == 'miiv':
        # MIMIC-IV: 如果最大值<=1.0且中位数>0.1，认为是分数形式，需要转换为百分比
        if max_val <= 1.0 and min_val >= 0.0 and fio2_values.median() > 0.1:
            result_df[fio2_col] = result_df[fio2_col] * 100
            logger.debug(f"MIMIC-IV FiO2从分数形式转换为百分比形式 (max_val: {max_val}, median: {fio2_values.median()})")
        # 如果值在0-1之间但有些异常值，检查大部分数据
        elif max_val <= 1.5 and (fio2_values.quantile(0.95) <= 1.0) and fio2_values.median() > 0.1:
            result_df[fio2_col] = result_df[fio2_col] * 100
            logger.debug(f"MIMIC-IV FiO2从分数形式转换为百分比形式 (95%分位数: {fio2_values.quantile(0.95)}, median: {fio2_values.median()})")

    elif database.lower() == 'eicu':
        # eICU: 通常已经是百分比形式，但进行验证
        if max_val <= 1.0 and min_val >= 0.0 and fio2_values.median() > 0.1:
            result_df[fio2_col] = result_df[fio2_col] * 100
            logger.debug(f"eICU FiO2从分数形式转换为百分比形式 (max_val: {max_val}, median: {fio2_values.median()})")

    elif database.lower() == 'aumc':
        # AUMC: 特殊处理 - 已知大部分是百分比形式，只有少数itemid可能是分数
        # 检查是否存在明显的分数形式数据（如0.21, 0.4等典型的分数值）
        fraction_like_values = fio2_values[(fio2_values > 0.1) & (fio2_values < 1.0)]

        if len(fraction_like_values) > 0:
            # 如果有>20%的值看起来像分数形式，则全部转换
            fraction_ratio = len(fraction_like_values) / len(fio2_values)
            if fraction_ratio > 0.2:
                result_df[fio2_col] = result_df[fio2_col] * 100
                logger.debug(f"AUMC FiO2从分数形式转换为百分比形式 (fraction_ratio: {fraction_ratio:.2f})")
            else:
                # 否则只转换明显是分数的值，保留已经是百分比的值
                mask = (result_df[fio2_col] > 0.1) & (result_df[fio2_col] < 1.0)
                result_df.loc[mask, fio2_col] = result_df.loc[mask, fio2_col] * 100
                logger.debug(f"AUMC FiO2选择性转换：{mask.sum()}个值从分数转为百分比")

        # 特殊处理：将可疑的0值和异常值设为NaN，让后续逻辑处理
        # AUMC中0.0通常表示缺失值而不是真实的FiO2值
        zero_mask = result_df[fio2_col] == 0.0
        if zero_mask.sum() > 0:
            result_df.loc[zero_mask, fio2_col] = float('nan')
            logger.debug(f"AUMC FiO2: 将{zero_mask.sum()}个0值设为NaN")

    # 验证转换后的值在合理范围内（0-100）
    converted_values = result_df[fio2_col].dropna()
    if len(converted_values) > 0:
        conv_max = converted_values.max()
        conv_min = converted_values.min()

        # 如果转换后的值超出合理范围，发出警告
        if conv_max > 100 or conv_min < 0:
            logger.warning(f"数据库 {database} FiO2值超出合理范围 [0,100]: min={conv_min}, max={conv_max}")

        # 记录转换信息
        if max_val <= 1.0:
            logger.info(f"数据库 {database} FiO2单位已标准化为百分比形式")

    return result_df

def _safe_group_apply(grouped, func):
    """Compatibility helper for pandas include_groups default change."""
    try:
        # pandas 2.1+: use include_groups=False (group keys excluded from func input)
        # pandas 3.0: include_groups=True removed entirely; False is correct
        return grouped.apply(func)
    except TypeError:  # pandas < 2.1 doesn't have include_groups at all
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
    expand_forward: bool = False,
) -> Optional[pd.DataFrame]:
    """Build fill_gaps limits matching ricu's collapse(x) behavior.
    
    Returns the observed data range (min/max per ID) without expansion.
    This matches R ricu's collapse() function which simply computes:
    - start = min(index_var) per ID
    - end = max(index_var) per ID
    
    Args:
        expand_forward: Deprecated, kept for compatibility. Should always be False.
                       R ricu's collapse() does NOT expand the time range.
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
    
    # R ricu's collapse() simply returns min/max without any expansion
    # Do NOT expand end time - this was a bug that caused AUMC sofa to generate
    # millions of extra rows (e.g., patient 14 with admittedat=57961 hours 
    # would generate rows from 0 to 115890 instead of just 57911 to 57979)
    
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

    data = win_tbl.data.copy()

    # 🚀 向量化展开 — 替代 iterrows 循环，50-100x 加速
    starts_raw = data[idx_col].apply(_coerce_hour_scalar).values
    durs_raw = data[dur_col].apply(_coerce_duration_hours).values
    
    # Filter valid rows
    valid_mask = ~np.isnan(starts_raw) & ~np.isnan(durs_raw) & (durs_raw > 0)
    if not valid_mask.any():
        cols = id_columns + [out_index, value_column]
        return _as_icutbl(
            pd.DataFrame(columns=cols),
            id_columns=id_columns,
            index_column=out_index,
            value_column=value_column,
        )
    
    starts = starts_raw[valid_mask]
    durs = durs_raw[valid_mask]
    ends = starts + durs
    
    # Compute aligned start times and number of points per row
    aligned_starts = np.floor(starts / interval_hours) * interval_hours
    n_points = np.maximum(1, np.ceil((ends - aligned_starts) / interval_hours).astype(int))
    total_points = n_points.sum()
    
    # Pre-allocate and fill time array
    expanded_times = np.empty(total_points, dtype=np.float64)
    row_indices = np.empty(total_points, dtype=np.intp)
    pos = 0
    valid_indices = np.where(valid_mask)[0]
    for i in range(len(starts)):
        n = n_points[i]
        times = aligned_starts[i] + np.arange(n) * interval_hours
        expanded_times[pos:pos+n] = times
        row_indices[pos:pos+n] = valid_indices[i]
        pos += n
    expanded_times = expanded_times[:pos]
    row_indices = row_indices[:pos]
    
    # Build result DataFrame using numpy repeat (vectorized)
    result_dict = {}
    for col in id_columns:
        if col in data.columns:
            result_dict[col] = data[col].values[row_indices]
    result_dict[out_index] = expanded_times
    if value_column in data.columns:
        result_dict[value_column] = data[value_column].values[row_indices]
    else:
        result_dict[value_column] = fill_value
    
    expanded = pd.DataFrame(result_dict)
    expanded = expanded.drop_duplicates()
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
    """Protocol subset used from :class:`~easyicu.concept.ConceptResolver`."""

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
    """Protocol subset for :class:`~easyicu.datasource.ICUDataSource`."""

    config: object


def _empty_icutbl(ctx: ConceptCallbackContext) -> ICUTable:
    """Create an empty ICUTable with proper column structure from context.
    
    This is used when callbacks receive empty or missing input data but need
    to return a valid ICUTable structure.
    
    Args:
        ctx: Callback context containing data source configuration
    
    Returns:
        Empty ICUTable with appropriate ID columns
    """
    # Get default ID column from data source config
    id_col = 'stay_id'
    if hasattr(ctx.data_source, 'config') and hasattr(ctx.data_source.config, 'name'):
        db_name = ctx.data_source.config.name
        if db_name in ['eicu', 'eicu_demo']:
            id_col = 'patientunitstayid'
        elif db_name == 'aumc':
            id_col = 'admissionid'
        elif db_name == 'hirid':
            id_col = 'patientid'
        elif db_name == 'mimic':
            id_col = 'icustay_id'
        elif db_name == 'sic':
            id_col = 'CaseID'
    
    empty_df = pd.DataFrame(columns=[id_col, 'charttime', ctx.concept_name])
    return ICUTable(
        data=empty_df,
        id_columns=[id_col],
        index_column='charttime',
        value_column=ctx.concept_name,
    )


def _load_concept_for_callback(ctx: ConceptCallbackContext, concept_name: str) -> Optional[pd.DataFrame]:
    """Load a concept within a callback context.
    
    This is used when callbacks need to load additional concepts (e.g., weight for BMI).
    
    Args:
        ctx: Callback context with resolver access
        concept_name: Name of the concept to load
    
    Returns:
        DataFrame with the loaded concept data, or None if not available
    """
    try:
        if hasattr(ctx.resolver, 'load_concepts'):
            result = ctx.resolver.load_concepts(
                [concept_name],
                ctx.data_source,
                merge=True,
                patient_ids=ctx.patient_ids,
            )
            if isinstance(result, dict) and concept_name in result:
                table = result[concept_name]
                return table.df if hasattr(table, 'df') else table
            elif hasattr(result, 'df'):
                return result.df
            elif isinstance(result, pd.DataFrame):
                return result
    except Exception as e:
        logger.debug(f"Failed to load concept '{concept_name}' in callback: {e}")
    return None


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
        # eICU doesn't use icustays table, skip for eICU databases
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
                print("   ⚠️  icustays 表为空或未加载")
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
    
    # Normalize ID columns for case-insensitive comparison (SICdb uses CaseID vs caseid)
    def normalize_id_cols(ids):
        if not ids:
            return []
        return [col.lower() for col in ids]
    
    # Check if ID conversion is needed (case-insensitive)
    ref_normalized = normalize_id_cols(id_columns)
    needs_conversion = not all(normalize_id_cols(ids) == ref_normalized for ids in id_column_sets.values())
    
    converted_tables = dict(tables)  # Initialize copy for potential modifications
    
    # If difference is only case, normalize all tables to use the same case
    if needs_conversion:
        # Check if the difference is only case
        all_normalized = set()
        for ids in id_column_sets.values():
            all_normalized.update(normalize_id_cols(ids))
        
        # If all normalized IDs are the same, it's just a case issue
        if len(all_normalized) == 1 or (len(all_normalized) == 0 and len(id_column_sets) == 0):
            # Standardize to the first table's ID column case
            target_id_col = id_columns[0] if id_columns else None
            if target_id_col:
                for name, table in list(tables.items()):
                    table_ids = _get_id_columns(table)
                    if table_ids and table_ids[0].lower() == target_id_col.lower() and table_ids[0] != target_id_col:
                        # Rename the column to match the target case
                        data = table.data if hasattr(table, 'data') else table.df
                        if table_ids[0] in data.columns:
                            data = data.rename(columns={table_ids[0]: target_id_col})
                            converted_tables[name] = ICUTable(
                                data=data,
                                id_columns=[target_id_col],
                                index_column=table.index_column,
                                value_column=table.value_column,
                                unit_column=table.unit_column,
                            )
                # Update id_columns to the normalized form
                id_columns = [target_id_col]
                needs_conversion = False  # Case normalized, no real conversion needed
    
    if needs_conversion and convert_ids and ctx is not None:
        # Try to convert all tables to the common target ID (prefer stay_id/icustay_id for ICU data)
        # This replicates R ricu's automatic ID conversion in collect_dots()
        
        # Determine all ID types present
        all_id_types = set()
        for ids in id_column_sets.values():
            all_id_types.update(ids)
        
        # Determine target ID column based on database and available types
        # MIMIC-III uses icustay_id, MIMIC-IV uses stay_id
        if 'icustay_id' in all_id_types:
            target_id_col = 'icustay_id'
        elif 'stay_id' in all_id_types:
            target_id_col = 'stay_id'
        else:
            target_id_col = id_columns[0] if id_columns else 'stay_id'
        
        # Handle hadm_id ↔ icustay_id conversion (MIMIC-III)
        if 'hadm_id' in all_id_types and 'icustay_id' in all_id_types:
            # Prefer icustay_id as target (ICU-level granularity)
            target_id_col = 'icustay_id'
            mapping = _load_id_mapping_table(ctx, 'hadm_id', 'icustay_id')
            
            if mapping is not None:
                if os.environ.get('DEBUG'):
                    logger.debug(f"ID映射表加载成功: hadm_id → icustay_id, {len(mapping)} 行")
                
                # Convert tables with hadm_id to icustay_id
                tables_to_remove = []
                for name, table in list(tables.items()):
                    if 'hadm_id' in table.id_columns and 'icustay_id' not in table.id_columns:
                        if os.environ.get('DEBUG'):
                            logger.debug(f"转换表 '{name}': hadm_id → icustay_id")
                        converted_data = _convert_id_column(
                            table.data.copy(),
                            'hadm_id',
                            'icustay_id',
                            mapping
                        )
                        
                        if converted_data.empty:
                            tables_to_remove.append(name)
                            continue
                        
                        converted_tables[name] = ICUTable(
                            data=converted_data,
                            id_columns=['icustay_id'],
                            index_column=table.index_column,
                            value_column=table.value_column,
                            unit_column=table.unit_column,
                        )
                
                for name in tables_to_remove:
                    if name in tables:
                        del tables[name]
                    if name in converted_tables:
                        del converted_tables[name]
                id_columns = ['icustay_id']
        
        # Handle hadm_id ↔ stay_id conversion (MIMIC-IV)
        elif 'hadm_id' in all_id_types and 'stay_id' in all_id_types:
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
                    print("   ⚠️  ID映射表加载失败: hadm_id → stay_id")
        
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
        # Avoid full data copy if possible
        frame = table.data
        
        # 展平MultiIndex列，避免合并时的MultiIndex错误
        if isinstance(frame.columns, pd.MultiIndex):
            # Must copy (shallow) if we are modifying columns
            frame = frame.copy(deep=False)
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
        # 跳过空表 - 它们对合并没有贡献，且可能有不正确的列类型
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
        
        # 处理 WinTbl (没有 value_column，使用 name 本身)
        from easyicu.table import WinTbl
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
        
        # 先处理frame中的重复列（例如合并多个item时可能产生重复的measuredat列）
        if frame.columns.duplicated().any():
            # 对于重复列，只保留第一个
            frame = frame.loc[:, ~frame.columns.duplicated()]
        
        # 只保留键列和值列,避免合并时的列冲突
        cols_to_keep = key_cols + [name]
        # 确保所有列都存在
        cols_to_keep = [c for c in cols_to_keep if c in frame.columns]
        frame = frame[cols_to_keep]
        
        standardized_tables[name] = (frame, table)  # 🚀 update with processed frame

    # 🚀 FAST PATH: pd.concat(axis=1) for outer join with clean key columns
    # Replaces N-1 iterative pd.merge with single pd.concat — ~3-5x faster for SOFA (6 components)
    _use_concat_fast_path = (
        how == "outer"
        and len(standardized_tables) > 1
        and all(
            set(key_cols).issubset(frame.columns)
            for frame, _ in standardized_tables.values()
            if not frame.empty
        )
    )
    
    if _use_concat_fast_path:
        indexed_frames = []
        for name, (frame, _) in standardized_tables.items():
            if frame.empty:
                continue
            # 如果同一 key 出现多行，先去重再走 concat fast path。
            # AUMC vasopressors 这类回调在 batch 场景下可能会暴露重复 key，
            # 直接 set_index 会触发 pandas InvalidIndexError。
            if frame.duplicated(subset=key_cols).any():
                frame = frame.drop_duplicates(subset=key_cols, keep='last')
            indexed = frame.set_index(key_cols, drop=True)
            # Only keep the value column (concept name)
            if name in indexed.columns:
                indexed = indexed[[name]]
            indexed_frames.append(indexed)
        
        if indexed_frames:
            if len(indexed_frames) == 1:
                merged = indexed_frames[0].reset_index()
            else:
                merged = pd.concat(indexed_frames, axis=1, join="outer", sort=False, copy=False).reset_index()
        else:
            merged = None
    else:
        # FALLBACK: iterative merge (for inner join or edge cases)
        merged = None
        for name, (frame, table) in standardized_tables.items():
            if frame.empty:
                continue
            
            if merged is None:
                merged = frame
            else:
                try:
                    actual_key_cols = [col for col in key_cols if col in frame.columns and col in merged.columns]
                    if len(actual_key_cols) < len(key_cols):
                        missing_in_frame = [col for col in key_cols if col not in frame.columns]
                        missing_in_merged = [col for col in key_cols if col not in merged.columns]
                        logging.debug(f"跳过 '{name}': 缺少合并键列 (frame缺少: {missing_in_frame}, merged缺少: {missing_in_merged})")
                        continue

                    duplicate_cols = [c for c in frame.columns if c in merged.columns and c not in actual_key_cols]
                    if duplicate_cols:
                        frame = frame.drop(columns=duplicate_cols)

                    merged = merged.merge(frame, on=actual_key_cols, how=how)
                except (ValueError, KeyError) as e:
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
            # Filter id_columns to only those actually present in the DataFrame
            actual_id_cols = [col for col in (id_columns or []) if col in working.columns]
            if actual_id_cols:
                def _relative_hours(group: pd.Series) -> pd.Series:
                    valid = group.dropna()
                    if valid.empty:
                        return pd.Series(np.nan, index=group.index)
                    base = valid.iloc[0]
                    delta = group - base
                    return delta.dt.total_seconds() / 3600.0

                working[index_column] = working.groupby(actual_id_cols)[index_column].transform(
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
    if df.empty:
        return pd.DataFrame(columns=list(id_columns) + ["__start", "__end"])

    # Sort by ID and start time
    sort_cols = list(id_columns) + [start_col]
    df = df.sort_values(sort_cols).copy()

    # Vectorized interval merging
    # 1. Calculate running maximum of end time per group
    #    (groupby().cummax() is efficient)
    df['cum_max_end'] = df.groupby(list(id_columns))[end_col].cummax()
    
    # 2. Get previous row's cumulative max end
    #    (shift globally, but we'll handle group boundaries via mask)
    prev_max_end = df['cum_max_end'].shift()
    
    # 3. Identify start of new interval groups
    #    Condition: Current start > Previous Max End + Gap
    #    OR: It's the first row of a patient (ID change)
    
    # Check gap condition
    gap_condition = df[start_col] > (prev_max_end + max_gap)
    
    # Check ID change (first row of each ID group)
    # Since we sorted by ID, ~duplicated(keep='first') identifies the first row of each group
    is_first_row = ~df.duplicated(subset=id_columns, keep='first')
    
    # Combine conditions
    is_new_group = gap_condition | is_first_row
    
    # 4. Assign group IDs
    df['group_id'] = is_new_group.cumsum()
    
    # 5. Aggregate to find min start and max end for each group
    agg_dict = {start_col: 'min', end_col: 'max'}
    
    # Group by ID columns + group_id
    # We include id_columns in groupby to preserve them in the result
    merged = df.groupby(list(id_columns) + ['group_id'], as_index=False).agg(agg_dict)
    
    # Drop the temporary group_id
    merged = merged.drop(columns=['group_id'])
    
    # Rename columns to match expected output if needed (but here they are already correct)
    # The caller expects __start and __end columns, which are preserved if start_col/end_col are __start/__end
    
    return merged

# ============================================================================
# AUMC-specific callbacks
# ============================================================================

def _callback_aumc_death(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """AUMC death callback: marks death if it occurred within 72 hours of ICU discharge.
    
    R ricu logic: x[, val_var := is_true(index_var - val_var < hours(72L))]
    where index_var = dateofdeath, val_var = dischargedat
    
    - If dateofdeath is NA: death = FALSE (survived)
    - If dateofdeath is not NA and (dateofdeath - dischargedat) < 72h: death = TRUE
    - Time is in milliseconds in AUMC, 72 hours = 72 * 3600 * 1000 ms
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    # Get the single input table
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return _empty_icutbl(ctx)
    
    id_columns = input_table.id_columns
    # For AUMC death: index_var = dateofdeath, val_var = dischargedat
    index_column = input_table.index_column or ctx.index_column  # dateofdeath
    value_column = input_table.value_column  # dischargedat
    
    # R ricu logic: is_true(dateofdeath - dischargedat < 72 hours)
    # is_true returns TRUE only if the condition is TRUE (not NA)
    # If dateofdeath is NA, result is FALSE (survived)
    
    # AUMC times are in milliseconds, 72 hours = 72 * 3600 * 1000 = 259200000 ms
    hours_72_ms = 72 * 3600 * 1000
    
    if index_column in data.columns and value_column in data.columns:
        # dateofdeath and dischargedat
        dateofdeath = pd.to_numeric(data[index_column], errors='coerce')
        dischargedat = pd.to_numeric(data[value_column], errors='coerce')
        
        # is_true: returns TRUE only if condition is TRUE (not NA)
        # If dateofdeath is NA, the subtraction result is NA, so is_true returns FALSE
        diff = dateofdeath - dischargedat
        # death = TRUE if dateofdeath is not NA AND (dateofdeath - dischargedat) < 72h
        data['death'] = (~dateofdeath.isna() & (diff < hours_72_ms)).astype(bool)
    else:
        # If columns are missing, mark all as death=False (conservative - assume survived)
        data['death'] = False
    
    output_cols = list(id_columns) + ['death']
    result = data[output_cols].copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=None, value_column='death')

def _callback_sic_death(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """SICdb death callback: marks death based on OffsetOfDeath in cases table.

    SICdb stores OffsetOfDeath in seconds from ICU admission.
    If OffsetOfDeath is NaN, patient survived.
    If OffsetOfDeath is not NaN, patient died.

    Returns DataFrame with CaseID, charttime (OffsetOfDeath in hours), death (bool).
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)

    input_table = list(tables.values())[0]
    data = input_table.df.copy()

    if data.empty:
        return _empty_icutbl(ctx)

    id_columns = input_table.id_columns

    # OffsetOfDeath is the index_var in concept-dict.json
    offset_col = input_table.index_column or 'OffsetOfDeath'

    if offset_col not in data.columns:
        # Fall back: look for any column with "death" in name
        for c in data.columns:
            if 'death' in c.lower():
                offset_col = c
                break

    if offset_col in data.columns:
        offset_vals = pd.to_numeric(data[offset_col], errors='coerce')
        # death = TRUE if OffsetOfDeath is not NaN
        data['death'] = (~offset_vals.isna()).astype(bool)
        # Convert offset from seconds to hours for charttime
        data['charttime'] = offset_vals / 3600.0
    else:
        data['death'] = False
        data['charttime'] = np.nan

    output_cols = list(id_columns) + ['charttime', 'death']
    output_cols = [c for c in output_cols if c in data.columns]
    result = data[output_cols].copy()

    return _as_icutbl(result, id_columns=id_columns, index_column='charttime', value_column='death')


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

def _callback_aumc_dur(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """AUMC duration callback: calc duration from start to stop by group.
    
    Replicates ricu's aumc_dur which calls calc_dur(x, val_var, index_var(x), stop_var, grp_var).
    
    IMPORTANT: AUMC times are already in MINUTES (converted by datasource.py load_table).
    R ricu flow:
    1. ms_as_mins: as.integer(x / 6e4) - floors to integer minutes (done in datasource.py)
    2. re_time: round_to(x, 60) - floors to hours (60 min intervals)
    3. calc_dur: max(stop_hours) - min(start_hours)
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return input_table
    
    id_columns = input_table.id_columns
    index_column = input_table.index_column or ctx.index_column
    value_column = ctx.concept_name  # e.g., dopa_dur
    
    # Get metadata from item definition
    item_def = getattr(ctx, 'item_definition', None) if hasattr(ctx, 'item_definition') else None
    stop_var = None
    grp_var = None
    
    if item_def:
        stop_var = item_def.get('stop_var')
        grp_var = item_def.get('grp_var')
    
    if not stop_var or stop_var not in data.columns:
        # Fall back to common AUMC stop column names
        for candidate in ['stop', 'endtime', 'stoptime']:
            if candidate in data.columns:
                stop_var = candidate
                break
    
    if not stop_var or stop_var not in data.columns:
        # Can't calculate duration without stop time
        logger.warning(f"aumc_dur: stop_var not found for {ctx.concept_name}, columns: {data.columns.tolist()}")
        return input_table
    
    if not index_column or index_column not in data.columns:
        logger.warning(f"aumc_dur: index_column '{index_column}' not found, columns: {data.columns.tolist()}")
        return input_table
    
    # Build grouping columns
    # NOTE: We use grp_var (orderid) to calculate per-order duration initially,
    # but the final merge_ranges is done in vaso60 callback, not here.
    group_cols = list(id_columns)
    if grp_var and grp_var in data.columns:
        group_cols.append(grp_var)
    
    # Ensure numeric types for time columns
    data[index_column] = pd.to_numeric(data[index_column], errors='coerce')
    data[stop_var] = pd.to_numeric(data[stop_var], errors='coerce')
    
    # Drop rows with NaN times
    data = data.dropna(subset=[index_column, stop_var])
    
    if data.empty:
        result = pd.DataFrame(columns=group_cols + [index_column, value_column])
        return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column=value_column)
    
    # AUMC times are already in MINUTES (converted by datasource.py from ms)
    # Just use them directly as integers
    data['_start_mins'] = data[index_column].astype(int)
    data['_stop_mins'] = data[stop_var].astype(int)
    
    # Group and aggregate: min(start), max(stop) in minutes
    if group_cols:
        agg_dict = {
            '_start_mins': 'min',  # min start time (integer minutes)
            '_stop_mins': 'max'    # max stop time (integer minutes)
        }
        
        grouped = data.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    else:
        # No grouping - aggregate over entire dataset per row
        grouped = data.copy()
    
    # R ricu re_time floors to hours: round_to(mins, 60) = floor(mins/60)*60
    # Then calc_dur: duration = max_stop_hours - min_start_hours
    # Combined: duration = floor(max_stop_mins/60) - floor(min_start_mins/60)
    start_hours_floor = (grouped['_start_mins'] / 60.0).apply(lambda x: int(x) if pd.notna(x) else x)
    stop_hours_floor = (grouped['_stop_mins'] / 60.0).apply(lambda x: int(x) if pd.notna(x) else x)
    duration_hours = stop_hours_floor - start_hours_floor
    
    grouped[value_column] = duration_hours.astype(float)
    
    # Index column is start time in hours (floored)
    grouped[index_column] = start_hours_floor.astype(float)
    
    # Keep group cols, index_column, and value column
    keep_cols = group_cols + [index_column, value_column]
    result = grouped[[col for col in keep_cols if col in grouped.columns]].copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column=value_column)

def _callback_blood_cell_ratio(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate blood cell ratios (e.g., lymphocytes, neutrophils as percentage).
    
    R ricu logic:
      blood_cell_ratio <- function(x, val_var, unit_var, env, ...) {
        x <- add_concept(x, env, "wbc")
        x <- x[, c(val_var, "wbc", unit_var) := list(
          100 * get(val_var) / get("wbc"), NULL, "%"
        )]
        x
      }
    
    This callback:
    1. Loads WBC (white blood cell count) concept
    2. Calculates percentage: 100 * cell_count / wbc
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
    
    # Ensure numeric
    if value_column in data.columns:
        data[value_column] = pd.to_numeric(data[value_column], errors='coerce')
    
    # Try to load WBC concept for ratio calculation
    try:
        # Get patient IDs from current data
        patient_ids = None
        if id_columns and id_columns[0] in data.columns:
            patient_ids = data[id_columns[0]].unique().tolist()
        
        # Load WBC concept
        from easyicu.api import load_concepts
        wbc_df = load_concepts(
            ['wbc'],
            database=ctx.database if hasattr(ctx, 'database') else None,
            patient_ids={id_columns[0]: patient_ids} if patient_ids and id_columns else None,
            verbose=False
        )
        
        if wbc_df is not None and not wbc_df.empty and 'wbc' in wbc_df.columns:
            # Find matching time columns in both dataframes
            # WBC from load_concepts uses 'charttime', but source data may use 'measuredat'
            time_col_candidates = ['charttime', 'measuredat', 'measuredat_minutes', 'datetime', 'observationoffset']
            
            data_time_col = None
            wbc_time_col = None
            for col in time_col_candidates:
                if col in data.columns and data_time_col is None:
                    data_time_col = col
                if col in wbc_df.columns and wbc_time_col is None:
                    wbc_time_col = col
            
            # Also check the declared index_column
            if index_column and index_column in data.columns:
                data_time_col = index_column
            
            if data_time_col and wbc_time_col:
                # Normalize time columns to numeric for merge_asof
                data_sorted = data.copy()
                wbc_sorted = wbc_df.copy()
                
                # Convert to numeric if needed
                data_sorted['_time_numeric'] = pd.to_numeric(data_sorted[data_time_col], errors='coerce')
                wbc_sorted['_time_numeric'] = pd.to_numeric(wbc_sorted[wbc_time_col], errors='coerce')
                
                # Drop rows with invalid times
                data_sorted = data_sorted.dropna(subset=['_time_numeric'])
                wbc_sorted = wbc_sorted.dropna(subset=['_time_numeric'])
                
                if not data_sorted.empty and not wbc_sorted.empty:
                    # Sort by patient ID and time
                    id_col = id_columns[0]
                    data_sorted = data_sorted.sort_values([id_col, '_time_numeric'])
                    wbc_sorted = wbc_sorted.sort_values([id_col, '_time_numeric'])
                    
                    # Use merge_asof to match WBC within 24 hours (1440 minutes)
                    merged = pd.merge_asof(
                        data_sorted,
                        wbc_sorted[[id_col, '_time_numeric', 'wbc']].rename(columns={'_time_numeric': '_wbc_time'}),
                        by=id_col,
                        left_on='_time_numeric',
                        right_on='_wbc_time',
                        direction='nearest',
                        tolerance=1440  # 24 hours in minutes
                    )
                    
                    # Calculate ratio: 100 * cell_count / wbc
                    if 'wbc' in merged.columns:
                        valid_wbc = merged['wbc'].notna() & (merged['wbc'] > 0)
                        merged.loc[valid_wbc, value_column] = 100 * merged.loc[valid_wbc, value_column] / merged.loc[valid_wbc, 'wbc']
                        merged = merged.drop(columns=['wbc', '_time_numeric', '_wbc_time'], errors='ignore')
                        data = merged
            else:
                # Fallback: merge only on patient ID (use latest WBC per patient)
                id_col = id_columns[0]
                wbc_latest = wbc_df.sort_values(wbc_time_col if wbc_time_col else id_col).groupby(id_col).last().reset_index()[[id_col, 'wbc']]
                merged = pd.merge(data, wbc_latest, on=id_col, how='left')
                
                if 'wbc' in merged.columns:
                    valid_wbc = merged['wbc'].notna() & (merged['wbc'] > 0)
                    merged.loc[valid_wbc, value_column] = 100 * merged.loc[valid_wbc, value_column] / merged.loc[valid_wbc, 'wbc']
                    merged = merged.drop(columns=['wbc'], errors='ignore')
                    data = merged
                    
    except Exception as e:
        # If WBC loading fails, just pass through the data as-is
        import logging
        logging.debug(f"blood_cell_ratio: WBC loading failed: {e}, using passthrough")
    
    output_cols = list(id_columns) + ([index_column] if index_column else []) + [value_column]
    output_cols = [c for c in output_cols if c in data.columns]
    result = data[output_cols].dropna(subset=[value_column]).copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column=value_column)

# ============================================================================
# MIMIC-III-specific callbacks
# ============================================================================

def _callback_mimic_age(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """MIMIC-III age callback: convert dob to age in years at ICU admission.
    
    R ricu logic for mimic_age (transform_fun wrapper):
      mimic_age <- function(x) {
        x <- as.double(x, units = "days") / -365
        ifelse(x > 90, 90, x)
      }
    
    In MIMIC-III, age is calculated from date of birth (dob) to ICU admission (intime).
    R ricu's change_id mechanism converts dob to (intime - dob) time difference.
    Then mimic_age converts days to years and caps at 90.
    
    MIMIC-III patients >= 89 years old at admission have shifted dob to obfuscate age.
    These patients get age = 90 (capped).
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return _empty_icutbl(ctx)
    
    id_columns = input_table.id_columns
    
    # Determine database and stay-level ID column
    db_name = getattr(ctx, 'db_name', 'mimic')
    stay_id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
    
    # Check if 'age' column already exists (some MIMIC-III setups have it)
    if 'age' in data.columns:
        data['age'] = pd.to_numeric(data['age'], errors='coerce')
        # Cap at 90 (R ricu: ifelse(x > 90, 90, x))
        data.loc[data['age'] > 90, 'age'] = 90
    elif 'anchor_age' in data.columns:
        # MIMIC-IV style anchor_age
        data['age'] = pd.to_numeric(data['anchor_age'], errors='coerce')
        data.loc[data['age'] > 90, 'age'] = 90
    elif 'dob' in data.columns:
        # MIMIC-III: Calculate age from dob and intime
        # Need to load icustays to get intime for each patient
        data_source = getattr(ctx, 'data_source', None)
        
        if data_source is None:
            # Cannot calculate age without data source
            return _empty_icutbl(ctx)
        
        try:
            # Load icustays to get intime
            icustays = data_source.load_table(
                'icustays', 
                columns=['subject_id', stay_id_col, 'intime'],
                verbose=False
            )
            if hasattr(icustays, 'data'):
                icustays = icustays.data
            
            # Merge patients with icustays to get intime
            data = data.merge(icustays, on='subject_id', how='inner')
            
            # Parse datetime columns
            dob = pd.to_datetime(data['dob'], errors='coerce')
            intime = pd.to_datetime(data['intime'], errors='coerce')
            
            # R ricu formula: as.double(x, units = "days") / -365
            # The negative is because R ricu passes (intime - dob) but in negative form
            # Actually in R, the dob column is directly used and change_id calculates
            # intime - dob as a difftime. So age = (intime - dob).days / 365
            age_days = (intime - dob).dt.days
            age_years = age_days / 365.0  # R ricu uses 365, not 365.25
            
            # Cap at 90 (R ricu: ifelse(x > 90, 90, x))
            age_years = np.where(age_years > 90, 90, age_years)
            data['age'] = age_years
            
            # Update id_columns to include stay-level ID
            if stay_id_col in data.columns:
                id_columns = [stay_id_col]
            
        except Exception as e:
            # If loading icustays fails, try with admittime if available
            if 'admittime' in data.columns:
                dob = pd.to_datetime(data['dob'], errors='coerce')
                admittime = pd.to_datetime(data['admittime'], errors='coerce')
                age_days = (admittime - dob).dt.days
                age_years = age_days / 365.0
                age_years = np.where(age_years > 90, 90, age_years)
                data['age'] = age_years
            else:
                # Cannot calculate age
                return _empty_icutbl(ctx)
    else:
        # Cannot calculate age
        return _empty_icutbl(ctx)
    
    # Remove missing ages
    data = data.dropna(subset=['age'])
    
    output_cols = list(id_columns) + ['age']
    result = data[[c for c in output_cols if c in data.columns]].copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=None, value_column='age')

def _callback_mimic_abx_presc(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """MIMIC-III antibiotic prescription callback.
    
    R ricu logic:
      mimic_abx_presc <- function(x, val_var, ...) {
        idx <- index_var(x)
        x <- x[, c(idx, val_var) := list(get(idx) + mins(720L), TRUE)]
        x
      }
    
    This callback:
    1. Shifts the time index forward by 720 minutes (12 hours)
    2. Sets the value to TRUE (antibiotic was prescribed)
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return _empty_icutbl(ctx)
    
    id_columns = input_table.id_columns
    index_column = input_table.index_column or ctx.index_column
    
    # Shift time forward by 720 minutes (12 hours)
    if index_column and index_column in data.columns:
        # Assuming time is in minutes (or convert if needed)
        data[index_column] = pd.to_numeric(data[index_column], errors='coerce') + 720
    
    # Set value to TRUE
    data['abx'] = True
    
    output_cols = list(id_columns) + ([index_column] if index_column and index_column in data.columns else []) + ['abx']
    result = data[[c for c in output_cols if c in data.columns]].copy()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column='abx')

def _callback_mimic_kg_rate(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """MIMIC-III weight-normalized rate callback.
    
    R ricu logic:
      mimic_kg_rate <- function(x, val_var, unit_var, env, ...) {
        x <- add_weight(x, env, "weight")
        x <- x[, c(val_var, unit_var) := list(
          get(val_var) / get("weight"), sub("mcgmin", "mcg/kg/min", get(unit_var))
        )]
        x
      }
    
    This callback:
    1. Adds patient weight to the data
    2. Divides the rate value by weight
    3. Updates the unit from mcgmin to mcg/kg/min
    """
    if not tables or len(tables) == 0:
        return _empty_icutbl(ctx)
    
    input_table = list(tables.values())[0]
    data = input_table.df.copy()
    
    if data.empty:
        return _empty_icutbl(ctx)
    
    id_columns = input_table.id_columns
    index_column = input_table.index_column or ctx.index_column
    value_column = input_table.value_column or ctx.concept_name
    unit_column = input_table.unit_column
    
    # Determine the primary ID column for MIMIC-III
    id_col = 'icustay_id' if 'icustay_id' in id_columns else (id_columns[0] if id_columns else 'stay_id')
    
    # Try to load weight data and join
    try:
        weight_df = _load_concept_for_callback(ctx, 'weight')
        if weight_df is not None and not weight_df.empty:
            # Get weight ID column
            weight_id_col = 'icustay_id' if 'icustay_id' in weight_df.columns else (
                'stay_id' if 'stay_id' in weight_df.columns else id_col
            )
            
            # Get first (or median) weight per patient
            weight_agg = weight_df.groupby(weight_id_col)['weight'].first().reset_index()
            
            # Merge weight into data
            if id_col in data.columns:
                data = data.merge(weight_agg, left_on=id_col, right_on=weight_id_col, how='left')
                
                # Divide rate by weight
                if 'weight' in data.columns and value_column in data.columns:
                    data[value_column] = pd.to_numeric(data[value_column], errors='coerce')
                    data['weight'] = pd.to_numeric(data['weight'], errors='coerce')
                    data[value_column] = data[value_column] / data['weight']
                    
                    # Update unit if present
                    if unit_column and unit_column in data.columns:
                        data[unit_column] = data[unit_column].str.replace('mcgmin', 'mcg/kg/min', regex=False)
                    
                    # Drop weight column
                    data = data.drop(columns=['weight'], errors='ignore')
    except Exception:
        # If weight loading fails, return data without weight normalization
        pass
    
    # Build output columns
    output_cols = list(id_columns) + ([index_column] if index_column and index_column in data.columns else []) + [value_column]
    if unit_column and unit_column in data.columns:
        output_cols.append(unit_column)
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


def _callback_anion_gap(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Derived concept: serum anion gap = Na - (Cl + HCO3).

    Inputs are ``na``, ``cl`` and ``bicar`` (all in mEq/L). The three
    components are joined on (id, time) and the difference is computed
    row-wise. Rows missing any component are dropped. Values outside a
    permissive physiological window are filtered to suppress lab noise.

    Normal range: 8-16 mEq/L. Elevated AG (>16) indicates metabolic
    acidosis with unmeasured anions (lactate, ketones, uremia, toxins);
    low AG (<4) typically reflects hypoalbuminemia or lab error.
    """
    merged, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="inner")
    if merged.empty:
        keep = id_columns + ([index_column] if index_column else [])
        empty = merged[keep].copy() if all(c in merged.columns for c in keep) else pd.DataFrame(columns=keep)
        empty["anion_gap"] = pd.Series(dtype="float64")
        return _as_icutbl(
            empty,
            id_columns=id_columns,
            index_column=index_column,
            value_column="anion_gap",
        )

    na = pd.to_numeric(merged["na"], errors="coerce")
    cl = pd.to_numeric(merged["cl"], errors="coerce")
    bicar = pd.to_numeric(merged["bicar"], errors="coerce")
    anion_gap = na - (cl + bicar)

    keep = id_columns + ([index_column] if index_column else [])
    out = merged[keep].copy()
    out["anion_gap"] = anion_gap
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["anion_gap"])
    # Permissive physiological filter; standard reference is 8-16 mEq/L
    # but we keep abnormal values up to 40 (severe metabolic acidosis)
    # and down to -10 (lab error / extreme hypoalbuminemia).
    out = out[(out["anion_gap"] >= -10) & (out["anion_gap"] <= 50)]

    return _as_icutbl(
        out.reset_index(drop=True),
        id_columns=id_columns,
        index_column=index_column,
        value_column="anion_gap",
    )


def _callback_pulse_pressure(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Derived concept: arterial pulse pressure = SBP - DBP (mmHg).

    Inputs are ``sbp`` and ``dbp`` (paired vitals, typically recorded at
    the same timestamp). Rows missing either component are dropped.
    Values outside a permissive physiological window are filtered.

    Normal range: 30-50 mmHg. Narrow pulse pressure (<25) suggests
    cardiogenic shock, tamponade, or severe hypovolemia; wide pulse
    pressure (>60) suggests aortic regurgitation, sepsis vasoplegia,
    or anemia.
    """
    merged, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="inner")
    if merged.empty:
        keep = id_columns + ([index_column] if index_column else [])
        empty = merged[keep].copy() if all(c in merged.columns for c in keep) else pd.DataFrame(columns=keep)
        empty["pulse_pressure"] = pd.Series(dtype="float64")
        return _as_icutbl(
            empty,
            id_columns=id_columns,
            index_column=index_column,
            value_column="pulse_pressure",
        )

    sbp = pd.to_numeric(merged["sbp"], errors="coerce")
    dbp = pd.to_numeric(merged["dbp"], errors="coerce")
    pp = sbp - dbp

    keep = id_columns + ([index_column] if index_column else [])
    out = merged[keep].copy()
    out["pulse_pressure"] = pp
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["pulse_pressure"])
    # Permissive physiological filter; extreme but plausible values kept
    # (very narrow PP in shock, very wide PP in severe AR).
    out = out[(out["pulse_pressure"] >= 0) & (out["pulse_pressure"] <= 200)]

    return _as_icutbl(
        out.reset_index(drop=True),
        id_columns=id_columns,
        index_column=index_column,
        value_column="pulse_pressure",
    )


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
        # R ricu: findInterval(x, c(2, 3, 9, 13, 15), left.open=TRUE)
        # (2,3] -> U, (3,9] -> P, (9,13] -> V, (13,15] -> A, else NA
        if value <= 2:
            return None
        if value <= 3:
            return "U"
        if value <= 9:
            return "P"
        if value <= 13:
            return "V"
        if value <= 15:
            return "A"
        return None

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

    # R ricu uses median aggregation for numeric data (see tbl-utils.R aggregate.id_tbl)
    if key_cols:
        aggregated = (
            combined.groupby(key_cols)["norepi_equiv"].median().reset_index()
        )
        aggregated = aggregated.sort_values(key_cols).reset_index(drop=True)
    else:
        aggregated = pd.DataFrame({"norepi_equiv": [combined["norepi_equiv"].median()]})

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
        # NOTE: R ricu's sofa_cardio does NOT merge vaso_ind into the score data.
        # vaso_ind is listed in depends_on only for dependency ordering.
        # The scoring function uses only the concepts field: map, dopa60, norepi60, dobu60, epi60.
        # We explicitly remove vaso_ind from tables if present.
        tables = dict(tables)
        tables.pop("vaso_ind", None)
        # CRITICAL: For single concept (sofa_single type), ricu_code's collect_dots returns the data directly
        # For multiple concepts, use outer join (replicates R ricu merge_dat = TRUE)
        # In ricu_code: sofa_single("plt", "sofa_coag", fun) -> collect_dots("plt", .
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

        # 🚀 PERF: Skip sort here — scoring functions are per-row operations (thresholds),
        # don't depend on order. fill_gaps in _callback_sofa_score will sort later.

        # NOTE: R ricu's sofa_cardio does NOT forward-fill vasopressor values.
        # It simply merges the data and calculates scores directly.
        # Forward-fill was incorrectly added here which caused vasopressor values
        # to persist beyond the end of infusion, inflating SOFA scores.
        # Removed forward-fill to match R ricu behavior.

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
                if ctx.concept_name == "sofa2_resp" and name == "ecmo_indication":
                    kwargs[name] = col_data if isinstance(col_data, pd.Series) else pd.Series(col_data)
                    continue
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
                # For optional parameters (like vasopressors in sofa_cardio, urine24 in sofa_renal), 
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
                elif ctx.concept_name == 'sofa2_cns' and name in ['delirium_tx', 'delirium_positive', 'motor_response']:
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
        
        # NOTE: ricu uses merge(all=TRUE) which produces all time points from both tables.
        # Even if urine24 values are NA, the time points are preserved in the merged result.
        # We should NOT filter based on NA values - the outer merge handles this correctly.
        # The sofa_renal score at those time points will be 0 (based on available data).
        
        # For optional parameters (like urine24 in sofa_renal), ensure they are None if all NaN
        # This replicates R ricu's behavior where missing optional params are treated as NULL
        # Handle optional parameters correctly - convert all-NaN Series to None
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
        
        # Remove duplicate timestamps (can occur when merging tables with outer join
        # or when raw data has multiple records at same timestamp)
        # Keep first occurrence for each (admissionid, measuredat) pair
        dedup_cols = list(id_columns) + ([index_column] if index_column else [])
        frame = frame.drop_duplicates(subset=dedup_cols, keep='first')
        
        return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column=ctx.concept_name)

    return wrapper


def _expand_wintbl_to_hourly(vent_df, vent_id_cols, start_col, dur_col, time_col_name):
    """Vectorized expansion of WinTbl vent_ind data to hourly time points.
    
    Replaces iterrows() loop with numpy vectorized operations.
    Each window (start, dur) is expanded to cover all integer hours in [floor(start), floor(start+dur/60)].
    """
    if vent_df.empty:
        return pd.DataFrame(columns=list(vent_id_cols) + [time_col_name, 'vent_ind'])
    
    starts = pd.to_numeric(vent_df[start_col], errors='coerce').values
    if dur_col in vent_df.columns:
        durs = pd.to_numeric(vent_df[dur_col], errors='coerce').fillna(0).values
    else:
        durs = np.zeros(len(vent_df))
    
    ends = np.where(durs > 0, starts + durs / 60.0, starts)
    start_hours = np.floor(starts).astype(int)
    end_hours = np.floor(ends).astype(int)
    counts = np.maximum(end_hours - start_hours + 1, 1)
    
    # Build expanded arrays
    total = counts.sum()
    hours_arr = np.empty(total, dtype=np.float64)
    row_indices = np.empty(total, dtype=np.intp)
    pos = 0
    for i in range(len(counts)):
        c = counts[i]
        hours_arr[pos:pos+c] = np.arange(start_hours[i], start_hours[i] + c)
        row_indices[pos:pos+c] = i
        pos += c
    
    # Build result DataFrame
    id_data = {}
    for col in vent_id_cols:
        if col in vent_df.columns:
            id_data[col] = vent_df[col].values[row_indices]
    id_data[time_col_name] = hours_arr
    id_data['vent_ind'] = True
    
    result = pd.DataFrame(id_data)
    # Aggregate by id + time (any=True for overlapping windows)
    group_cols = [c for c in list(vent_id_cols) + [time_col_name] if c in result.columns]
    result = result.groupby(group_cols, as_index=False).agg({'vent_ind': 'any'})
    return result


def _callback_sofa_resp(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate respiratory SOFA component with vent_ind expansion.
    
    Replicates R ricu's sofa_resp logic exactly:
      dat <- merge(dat[[pafi_var]], expand(dat[[vent_var]], aggregate = "any"), all = TRUE)
      dat <- dat[is_true(get(pafi_var) < 200) & !is_true(get(vent_var)), c(pafi_var) := 200]
      dat <- dat[, c("sofa_resp") := score_calc(get(pafi_var))]
    
    Key points:
    1. expand() only expands vent_ind windows to hourly points WITHIN the window
    2. merge(all=TRUE) is a full outer join
    3. Adjust: pafi < 200 and NOT ventilated → set pafi = 200
    4. score_calc uses is_true() so NA pafi → score 0
    """
    pafi_tbl = tables.get("pafi")
    vent_tbl = tables.get("vent_ind")
    
    if pafi_tbl is None:
        raise ValueError("sofa_resp requires 'pafi' concept")
    if vent_tbl is None:
        raise ValueError("sofa_resp requires 'vent_ind' concept")
    
    # Get pafi data - this provides the base timeline
    pafi_df = pafi_tbl.data.copy()
    pafi_index = pafi_tbl.index_column or "charttime"
    
    # Detect ID columns
    id_columns = pafi_tbl.id_columns if hasattr(pafi_tbl, 'id_columns') and pafi_tbl.id_columns else []
    if not id_columns:
        id_columns = [c for c in pafi_df.columns if c in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']]
    
    # Find pafi value column
    pafi_col = 'pafi'
    if 'pafi' not in pafi_df.columns:
        for col in pafi_df.columns:
            if col not in id_columns and col != pafi_index and 'pafi' in col.lower():
                pafi_df = pafi_df.rename(columns={col: 'pafi'})
                break
        else:
            # Fallback: use first non-id, non-index column
            for col in pafi_df.columns:
                if col not in id_columns and col != pafi_index:
                    pafi_df = pafi_df.rename(columns={col: 'pafi'})
                    break
    
    # Ensure numeric pafi
    pafi_df['pafi'] = pd.to_numeric(pafi_df['pafi'], errors='coerce')
    
    # Expand vent_ind WinTbl to hourly points WITHIN windows only (vectorized)
    if isinstance(vent_tbl, WinTbl):
        vent_df = vent_tbl.data.copy()
        vent_id_cols = vent_tbl.id_vars if hasattr(vent_tbl, 'id_vars') else id_columns
        start_col = vent_tbl.index_var if hasattr(vent_tbl, 'index_var') else "starttime"
        dur_col = vent_tbl.dur_var if hasattr(vent_tbl, 'dur_var') else "dur_var"
        
        vent_df = _expand_wintbl_to_hourly(vent_df, vent_id_cols, start_col, dur_col, pafi_index)
    else:
        vent_df = vent_tbl.data.copy()
        vent_index = vent_tbl.index_column if hasattr(vent_tbl, 'index_column') and vent_tbl.index_column else pafi_index
        # Rename vent index to match pafi
        if vent_index != pafi_index and vent_index in vent_df.columns:
            vent_df = vent_df.rename(columns={vent_index: pafi_index})
        # Find vent_ind column
        if 'vent_ind' not in vent_df.columns:
            for col in vent_df.columns:
                if col not in id_columns and col != pafi_index:
                    vent_df = vent_df.rename(columns={col: 'vent_ind'})
                    break
    
    # Merge with full outer join (R: merge(..., all=TRUE))
    merge_cols = [c for c in id_columns if c in vent_df.columns and c in pafi_df.columns] + [pafi_index]
    merge_cols = [c for c in merge_cols if c in vent_df.columns and c in pafi_df.columns]
    
    if merge_cols:
        result = pd.merge(pafi_df, vent_df, on=merge_cols, how='outer')
    else:
        result = pd.merge(pafi_df, vent_df, on=[pafi_index], how='outer')
    
    # Fill NaN vent_ind with False (not ventilated)
    result['vent_ind'] = pd.Series(
        pd.array(result['vent_ind'], dtype='boolean').fillna(False).to_numpy(dtype=bool),
        index=result.index,
    )
    
    # R ricu adjustment: if pafi < 200 and NOT ventilated, set pafi = 200
    # This limits score to max 2 for non-ventilated patients
    # R: dat[is_true(get(pafi_var) < 200) & !is_true(get(vent_var)), c(pafi_var) := 200]
    adj_mask = (result['pafi'] < 200) & (~result['vent_ind'])
    # is_true(pafi < 200) returns False for NA, so NA pafi is not adjusted
    adj_mask = adj_mask.fillna(False)
    result.loc[adj_mask, 'pafi'] = 200.0
    
    # R ricu score_calc using is_true() — NA pafi → all is_true checks FALSE → score 0
    pafi_vals = result['pafi']
    score = pd.Series(0, index=result.index, dtype="Int64")
    score = score.where(~(pafi_vals < 400).fillna(False), 1)
    score = score.where(~(pafi_vals < 300).fillna(False), 2)
    score = score.where(~(pafi_vals < 200).fillna(False), 3)
    score = score.where(~(pafi_vals < 100).fillna(False), 4)
    result['sofa_resp'] = score
    
    # Select output columns
    output_cols = [c for c in id_columns if c in result.columns] + [pafi_index, 'sofa_resp']
    result = result[output_cols].reset_index(drop=True)
    
    return _as_icutbl(result, id_columns=id_columns, index_column=pafi_index, value_column="sofa_resp")

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

    # Ensure all components exist with proper missing data handling
    for name in required:
        data[name] = data.get(name)
    
    # Fill gaps using data's inherent interval (matches R ricu fill_gaps(dat))
    # R infers interval from data; we infer from median time diff in merged data
    if index_column and index_column in data.columns:
        id_cols_to_group = list(id_columns) if id_columns else []
        data = data.sort_values(list(id_columns) + [index_column] if id_columns else [index_column])
        
        # 🚀 Vectorized interval inference (replaces per-patient loop)
        # cProfile: old per-patient loop contributed ~2s for 5000 patients
        if id_cols_to_group and len(data) > 1:
            _diffs = data.groupby(id_cols_to_group, sort=False)[index_column].diff().dropna()
            if pd.api.types.is_numeric_dtype(_diffs):
                _pos = _diffs[_diffs > 0]
            else:
                _pos = _diffs[_diffs > pd.Timedelta(0)]
            if len(_pos) > 0:
                inferred_interval = _pos.median()
                # Handle numeric (hours) vs timedelta
                if isinstance(inferred_interval, (int, float)):
                    interval = pd.Timedelta(hours=max(1, round(inferred_interval)))
                else:
                    inferred_hours = round(inferred_interval.total_seconds() / 3600)
                    interval = pd.Timedelta(hours=max(1, inferred_hours))
                
                # Fill gaps with inferred interval
                # Replicate R ricu's collapse() → fill_gaps() → expand() interaction
                limits_df = _compose_fill_limits(data, id_cols_to_group, index_column, ctx)
                if limits_df is not None and 'start' in limits_df.columns and 'end' in limits_df.columns:
                    limits_df = limits_df.copy()
                    limits_df['end'] = limits_df['end'] - limits_df['start']
                data = fill_gaps(
                    data,
                    id_cols=id_cols_to_group,
                    index_col=index_column,
                    interval=interval,
                    limits=limits_df,
                    method="none",
                )
                # 🚀 fill_gaps fast path returns sorted data, skip redundant sort
        
        # R ricu's sofa_score does NOT apply any LOCF to components.
        # The 24h sliding window (slide with max_or_na) naturally handles gaps:
        # - If data exists within the 24h window → max value is used
        # - If no data in the 24h window → NA → treated as 0 in rowSums
        # GCS has a 6h LOCF applied in its own callback (gcs), not here.
        
        # Apply sliding window to each component (replicates R ricu slide)
        agg_dict = {}
        for comp in required:
            if comp in data.columns:
                agg_dict[comp] = worst_val_fun
        
        # Apply slide (replicates R ricu slide(res, !
        if agg_dict:
            data = slide(
                data,
                list(id_columns),
                index_column,
                before=win_length,
                after=pd.Timedelta(0),
                agg_func=agg_dict,
                full_window=full_window,
                _pre_sorted=True,  # 🚀 data already sorted by fill_gaps
            )
    
    # Calculate total SOFA score (replicates R ricu rowSums(.SD, na.rm = TRUE))
    # R's na.rm=TRUE means skip NA, NOT fill with 0
    # sum(axis=1, skipna=True) matches R behavior: only sum non-NA values
    data["sofa"] = data[required].sum(axis=1, skipna=True)
    
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
    
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        print("   ⚠️  SOFA-2回调: _merge_tables 返回空数据")
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
        id_cols_to_group = list(id_columns) if id_columns else []
        data = data.sort_values(list(id_columns) + [index_column] if id_columns else [index_column])
        
        # 🚀 Vectorized interval inference (same optimization as SOFA-1)
        if id_cols_to_group and len(data) > 1:
            _diffs = data.groupby(id_cols_to_group, sort=False)[index_column].diff().dropna()
            if pd.api.types.is_numeric_dtype(_diffs):
                _pos = _diffs[_diffs > 0]
            else:
                _pos = _diffs[_diffs > pd.Timedelta(0)]
            if len(_pos) > 0:
                inferred_interval = _pos.median()
                if isinstance(inferred_interval, (int, float)):
                    interval = pd.Timedelta(hours=max(1, round(inferred_interval)))
                else:
                    inferred_hours = round(inferred_interval.total_seconds() / 3600)
                    interval = pd.Timedelta(hours=max(1, inferred_hours))
                
                limits_df = _compose_fill_limits(data, id_cols_to_group, index_column, ctx)
                data = fill_gaps(
                    data,
                    id_cols=id_cols_to_group,
                    index_col=index_column,
                    interval=interval,
                    limits=limits_df,
                    method="none",
                )
                # 🚀 fill_gaps fast path returns sorted data, skip redundant sort
        
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
                _pre_sorted=True,  # 🚀 data already sorted by fill_gaps
            )
    
    # Calculate total SOFA-2 score (note: output column is 'sofa2')
    # R's na.rm=TRUE means skip NA, NOT fill with 0
    data["sofa2"] = (
        data[required]
        .sum(axis=1, skipna=True)
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
    """Calculate MEWS score with 24-hour LOCF as in R ricu."""
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["mews"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="mews")

    # Apply 24-hour LOCF to input columns (matching R ricu slide + locf)
    # R ricu: cnc <- c("hr", "avpu", "temp", "sbp", "resp")
    #         slide(res, lapply(.SD, locf), before = win_length, .SDcols = cnc)
    value_cols = ["hr", "avpu", "temp", "sbp", "resp"]
    data = _apply_locf_24h(data, id_columns, index_column, value_cols, win_length_hours=24.0)

    # Handle avpu NaN: R ricu fifelse(x == "A", 0L, 3L) returns NA for NA input,
    # and rowSums(.SD, na.rm=TRUE) treats NA as 0. So we preprocess NaN -> "A".
    avpu_col = data.get("avpu")
    if avpu_col is not None:
        avpu_col = avpu_col.astype(str)
        # Treat NaN (converted to "nan") and empty as "A" (score 0)
        avpu_col = avpu_col.replace({"nan": "A", "None": "A", "": "A", "<NA>": "A"})
    else:
        avpu_col = pd.Series("A", index=data.index)

    result = mews_score(
        sbp=pd.to_numeric(data.get("sbp")),
        hr=pd.to_numeric(data.get("hr")),
        resp=pd.to_numeric(data.get("resp")),
        temp=pd.to_numeric(data.get("temp")),
        avpu=avpu_col,
    )
    data["mews"] = result
    cols = id_columns + ([index_column] if index_column else []) + ["mews"]
    return _as_icutbl(data[cols].reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="mews")

def _callback_news(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate NEWS score with 24-hour LOCF as in R ricu."""
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["news"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="news")

    # R ricu: res <- res[is.na(get("supp_o2")), c("supp_o2") := FALSE]
    # Fill NA supp_o2 with FALSE BEFORE LOCF (so TRUE values propagate, but NA → FALSE stays FALSE)
    if "supp_o2" in data.columns:
        data["supp_o2"] = data["supp_o2"].fillna(False)

    # Apply 24-hour LOCF to input columns (matching R ricu slide + locf)
    # R ricu: cnc <- c("hr", "avpu", "supp_o2", "o2sat", "temp", "sbp", "resp")
    #         slide(res, lapply(.SD, locf), before = win_length, .SDcols = cnc)
    value_cols = ["hr", "avpu", "supp_o2", "o2sat", "temp", "sbp", "resp"]
    data = _apply_locf_24h(data, id_columns, index_column, value_cols, win_length_hours=24.0)

    # Handle avpu NaN: R ricu fifelse(x == "A", 0L, 3L) returns NA for NA input,
    # and rowSums(.SD, na.rm=TRUE) treats NA as 0. So we preprocess NaN -> "A".
    avpu_col = data.get("avpu")
    if avpu_col is not None:
        avpu_col = avpu_col.astype(str)
        # Treat NaN (converted to "nan") and empty as "A" (score 0)
        avpu_col = avpu_col.replace({"nan": "A", "None": "A", "": "A", "<NA>": "A"})
    else:
        avpu_col = pd.Series("A", index=data.index)

    # supp_o2 should already be filled with False for NA values (done before LOCF)
    supp_o2_col = data.get("supp_o2")
    if supp_o2_col is not None:
        supp_o2_col = supp_o2_col.astype(bool)
    else:
        supp_o2_col = pd.Series(False, index=data.index)

    result = news_score(
        resp=pd.to_numeric(data.get("resp")),
        o2sat=pd.to_numeric(data.get("o2sat")),
        temp=pd.to_numeric(data.get("temp")),
        sbp=pd.to_numeric(data.get("sbp")),
        hr=pd.to_numeric(data.get("hr")),
        supp_o2=supp_o2_col,
        avpu=avpu_col,
        keep_components=False,
    )
    data["news"] = result
    cols = id_columns + ([index_column] if index_column else []) + ["news"]
    return _as_icutbl(data[cols].reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column="news")


def _apply_locf_window(
    data: pd.DataFrame,
    id_columns: List[str],
    index_column: Optional[str],
    value_columns: List[str],
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """
    Apply Last Observation Carried Forward (LOCF) within a sliding window.
    
    This is an alias for _apply_locf_24h with a configurable window size.
    Used by GCS callback with default 6-hour window.
    """
    return _apply_locf_24h(
        data=data,
        id_columns=id_columns,
        index_column=index_column,
        value_columns=value_columns,
        win_length_hours=window_hours,
    )


def _apply_locf_24h(
    data: pd.DataFrame,
    id_columns: List[str],
    index_column: Optional[str],
    value_columns: List[str],
    win_length_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Apply Last Observation Carried Forward (LOCF) within a 24-hour sliding window.
    
    This replicates the R ricu behavior:
    - slide(res, !!exp, before = win_length, .SDcols = cnc)
    - where exp = substitute(lapply(.SD, fun), list(fun = locf))
    
    For each time point, look backward within the window at ORIGINAL observations
    (not LOCF-filled values) and take the last non-NA value. This prevents
    cascading propagation beyond the original window.
    
    Args:
        data: DataFrame with measurements
        id_columns: List of ID columns for grouping (e.g., ['stay_id'])
        index_column: Time column name (e.g., 'charttime')
        value_columns: List of columns to apply LOCF on
        win_length_hours: Window length in hours (default 24)
        
    Returns:
        DataFrame with LOCF applied to specified columns
    """
    if data.empty or not index_column or not value_columns:
        return data
    
    # Ensure data is sorted by id and time
    sort_cols = id_columns + [index_column]
    data = data.sort_values(sort_cols).reset_index(drop=True)
    
    # Convert index_column to numeric (hours) if it's not already
    time_col = data[index_column]
    if pd.api.types.is_timedelta64_dtype(time_col):
        time_hours = time_col.dt.total_seconds() / 3600
    elif pd.api.types.is_numeric_dtype(time_col):
        time_hours = time_col  # Assume already in hours
    else:
        try:
            time_hours = pd.to_timedelta(time_col).dt.total_seconds() / 3600
        except Exception:
            # If cannot convert, use simple forward fill without time limit
            for col in value_columns:
                if col in data.columns:
                    data[col] = data.groupby(id_columns, dropna=False)[col].ffill()
            return data
    
    data["_time_hours_"] = time_hours
    
    # ⚡ Fully vectorized LOCF — no groupby.apply(), no per-patient Python dispatch.
    # Uses pandas ffill + time window mask to achieve O(n) total complexity.
    for col in value_columns:
        if col not in data.columns:
            continue
        
        # Save original non-NA positions and values
        valid_mask = data[col].notna()
        
        # Record the time of each valid observation
        data["_last_valid_time_"] = np.where(valid_mask, data["_time_hours_"], np.nan)
        
        # Forward-fill the last-valid-time within each patient group
        data["_last_valid_time_"] = data.groupby(id_columns, dropna=False, sort=False)["_last_valid_time_"].ffill()
        
        # Forward-fill the value within each patient group
        data[col] = data.groupby(id_columns, dropna=False, sort=False)[col].ffill()
        
        # Mask values that are outside the time window
        outside_window = (data["_time_hours_"] - data["_last_valid_time_"]) > win_length_hours
        data.loc[outside_window, col] = np.nan
    
    data = data.drop(columns=["_time_hours_", "_last_valid_time_"], errors="ignore")
    
    return data


def _callback_qsofa(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate qSOFA score with 24-hour LOCF as in R ricu."""
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["qsofa"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="qsofa")

    # Apply 24-hour LOCF to input columns (matching R ricu slide + locf)
    value_cols = ["sbp", "resp", "gcs"]
    data = _apply_locf_24h(data, id_columns, index_column, value_cols, win_length_hours=24.0)

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
    """Calculate SIRS score with 24-hour LOCF as in R ricu."""
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["sirs"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="sirs")

    # Apply 24-hour LOCF to input columns (matching R ricu slide + locf)
    value_cols = ["temp", "hr", "resp", "wbc", "pco2", "bnd"]
    data = _apply_locf_24h(data, id_columns, index_column, value_cols, win_length_hours=24.0)

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


def _match_fio2_fallback_loop(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    id_columns: list,
    index_column: str,
    left_col: str,
    right_col: str,
    match_win: pd.Timedelta,
    direction: str = 'forward'
) -> pd.DataFrame:
    """
    Fallback loop for merge_asof when the by parameter fails.
    
    🚀 OPTIMIZATION 2025-01-31: Instead of looping through each patient,
    we add a large offset to each patient's time values to make the global
    time column monotonically increasing. This allows using merge_asof's
    optimized C implementation for the entire dataset at once.
    """
    if left_df.empty or right_df.empty:
        return pd.DataFrame(columns=id_columns + [index_column, left_col, right_col])
    
    # 按 key + time 排序
    left_sorted = left_df.sort_values(by=id_columns + [index_column]).reset_index(drop=True)
    right_sorted = right_df.sort_values(by=id_columns + [index_column]).reset_index(drop=True)
    
    # 🔧 Align ID column dtypes (int32 vs int64 from different DuckDB sources)
    for col in id_columns:
        if col in left_sorted.columns and col in right_sorted.columns:
            if left_sorted[col].dtype != right_sorted[col].dtype:
                common_dtype = np.result_type(left_sorted[col].dtype, right_sorted[col].dtype)
                left_sorted[col] = left_sorted[col].astype(common_dtype)
                right_sorted[col] = right_sorted[col].astype(common_dtype)

    # 创建 key 到整数索引的映射（用于计算偏移量）
    all_keys = pd.concat([left_sorted[id_columns[0]], right_sorted[id_columns[0]]]).unique()
    key_to_idx = {k: i for i, k in enumerate(sorted(all_keys))}
    
    # 计算足够大的偏移量：比最大时间范围大很多
    time_range_left = left_sorted[index_column].max() - left_sorted[index_column].min()
    time_range_right = right_sorted[index_column].max() - right_sorted[index_column].min()
    
    # 处理 Timedelta 和数值类型
    def to_numeric(val):
        if isinstance(val, pd.Timedelta):
            return val.total_seconds() / 3600.0  # 转换为小时
        return val
    
    time_range = max(to_numeric(time_range_left), to_numeric(time_range_right))
    if pd.isna(time_range) or time_range <= 0:
        time_range = 1000000.0
    large_offset = time_range * 10  # 10倍时间范围作为偏移
    
    # 添加全局单调时间列
    left_sorted = left_sorted.copy()
    right_sorted = right_sorted.copy()
    
    # 检测时间列类型并转换为数值（小时）
    is_timedelta = pd.api.types.is_timedelta64_dtype(left_sorted[index_column])
    is_datetime = pd.api.types.is_datetime64_any_dtype(left_sorted[index_column])
    
    if is_timedelta:
        left_sorted['_time_numeric'] = left_sorted[index_column].dt.total_seconds() / 3600.0
        right_sorted['_time_numeric'] = right_sorted[index_column].dt.total_seconds() / 3600.0
        time_col_for_merge = '_time_numeric'
    elif is_datetime:
        # 转换为从最小时间开始的小时数
        min_time = min(left_sorted[index_column].min(), right_sorted[index_column].min())
        left_sorted['_time_numeric'] = (left_sorted[index_column] - min_time).dt.total_seconds() / 3600.0
        right_sorted['_time_numeric'] = (right_sorted[index_column] - min_time).dt.total_seconds() / 3600.0
        time_col_for_merge = '_time_numeric'
    else:
        # 已经是数值类型
        time_col_for_merge = index_column
    
    left_sorted['_time_global'] = (
        left_sorted[time_col_for_merge] + 
        left_sorted[id_columns[0]].map(key_to_idx) * large_offset
    )
    right_sorted['_time_global'] = (
        right_sorted[time_col_for_merge] + 
        right_sorted[id_columns[0]].map(key_to_idx) * large_offset
    )
    
    # tolerance 已经是小时单位（因为 _time_global 是小时）
    effective_tolerance = match_win
    if isinstance(match_win, pd.Timedelta):
        effective_tolerance = match_win.total_seconds() / 3600.0
    
    try:
        # 批量 merge_asof - 使用 _time_global 作为 on 列，by 列保持原样
        merged = pd.merge_asof(
            left_sorted[[*id_columns, '_time_global', left_col]],
            right_sorted[[*id_columns, '_time_global', right_col]],
            on='_time_global',
            by=id_columns,
            tolerance=effective_tolerance,
            direction='backward'
        )
        
        # 恢复原始时间列
        time_numeric_restored = merged['_time_global'] - merged[id_columns[0]].map(key_to_idx) * large_offset
        
        if is_timedelta:
            # 从数值（小时）转换回 Timedelta
            merged[index_column] = pd.to_timedelta(time_numeric_restored, unit='h')
        elif is_datetime:
            # 从数值（小时）转换回 datetime
            merged[index_column] = min_time + pd.to_timedelta(time_numeric_restored, unit='h')
        else:
            # 数值类型直接使用
            merged[index_column] = time_numeric_restored
        
        # 删除临时列
        merged = merged.drop(columns=['_time_global'])
        
        return merged[id_columns + [index_column, left_col, right_col]]
        
    except Exception as e:
        # 如果批量方法失败，回退到原始的逐个循环方法
        logger.debug(f"Batch merge_asof failed: {e}, falling back to per-patient loop")
        return _match_fio2_fallback_loop_original(
            left_df, right_df, id_columns, index_column,
            left_col, right_col, match_win, direction
        )


def _match_fio2_fallback_loop_original(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    id_columns: list,
    index_column: str,
    left_col: str,
    right_col: str,
    match_win: pd.Timedelta,
    direction: str = 'forward'
) -> pd.DataFrame:
    """
    Original fallback loop - processes each patient individually.
    Used when the optimized batch method fails.
    """
    result_list = []
    
    # 转换 tolerance 为数值类型（如果时间列是数值）
    effective_tolerance = match_win
    if pd.api.types.is_numeric_dtype(left_df[index_column]):
        if isinstance(match_win, pd.Timedelta):
            effective_tolerance = match_win.total_seconds() / 3600.0
    
    unique_ids = left_df[id_columns[0]].unique()
    for id_val in unique_ids:
        left_mask = left_df[id_columns[0]] == id_val
        right_mask = right_df[id_columns[0]] == id_val
        
        left_group = left_df[left_mask].sort_values(by=index_column).reset_index(drop=True)
        right_group = right_df[right_mask]
        
        if len(right_group) == 0:
            continue
            
        right_group = right_group.sort_values(by=index_column).reset_index(drop=True)
        
        try:
            merged = pd.merge_asof(
                left_group[[index_column, left_col]],
                right_group[[index_column, right_col]],
                on=index_column,
                tolerance=effective_tolerance,
                direction='backward'
            )
            for col in id_columns:
                merged[col] = id_val
            result_list.append(merged)
        except Exception:
            continue
    
    if result_list:
        return pd.concat(result_list, ignore_index=True)
    else:
        return pd.DataFrame(columns=id_columns + [index_column, left_col, right_col])


def _match_fio2(
    tables: Dict[str, ICUTable],
    o2_col: str,  # po2 or o2sat
    fio2_col: str,  # fio2
    match_win: pd.Timedelta,
    mode: str = "match_vals",
    fix_na_fio2: bool = True,
    ctx: Optional[ConceptCallbackContext] = None,  # Use ConceptCallbackContext
    database: str = None,  # 数据库名称，用于FiO2单位转换
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
    
    # 🔧 FIX 2025-01-31: 提前检查空数据并返回空结果
    # 当两个输入表都为空时，没必要继续处理，直接返回空结果
    o2_empty = (not hasattr(o2_tbl, 'data') or o2_tbl.data is None or len(o2_tbl.data) == 0)
    fio2_empty = (not hasattr(fio2_tbl, 'data') or fio2_tbl.data is None or len(fio2_tbl.data) == 0)
    
    if o2_empty and fio2_empty:
        # 两个输入都为空，返回空 DataFrame
        # 从 ctx.data_source.config 获取默认的 ID 列和时间列
        default_id_col = 'stay_id'  # 通用默认值
        default_idx_col = 'charttime'
        if ctx is not None and hasattr(ctx, 'data_source') and ctx.data_source is not None:
            cfg = ctx.data_source.config
            # 优先使用 icustay 的 ID（如 AUMC 的 admissionid）
            if hasattr(cfg, 'id_configs') and 'icustay' in cfg.id_configs:
                default_id_col = cfg.id_configs['icustay'].id
            elif hasattr(cfg, 'stay_id'):
                default_id_col = cfg.stay_id
            if hasattr(cfg, 'index_column'):
                default_idx_col = cfg.index_column
        empty_df = pd.DataFrame(columns=[default_id_col, default_idx_col, o2_col, fio2_col])
        return empty_df, [default_id_col], default_idx_col
    
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
        # 🚀 Optimized: avoid full-table copies, skip datetime conversion for numeric time,
        # use single merge_asof (matching R ricu's single rolling join)

        # Detect time columns and value columns
        o2_idx_col = o2_tbl.index_column
        fio2_idx_col = fio2_tbl.index_column
        o2_val_col = o2_tbl.value_column or o2_col
        fio2_val_col = fio2_tbl.value_column or fio2_col

        time_col_priority = ['charttime', 'measuredat_minutes', 'datetime', 'givenat', 'measuredat']

        def _detect_time_col(df, declared):
            if declared and declared in df.columns and not df[declared].isna().all():
                return declared
            for col in time_col_priority:
                if col in df.columns and not df[col].isna().all():
                    return col
            return declared

        o2_actual_time = _detect_time_col(o2_tbl.data, o2_idx_col)
        fio2_actual_time = _detect_time_col(fio2_tbl.data, fio2_idx_col)
        unified_time_col = 'charttime'
        index_column = unified_time_col

        # 🚀 Select only needed columns first, THEN copy (avoid copying wide tables)
        o2_cols_needed = list(set(id_columns + [o2_actual_time, o2_val_col]))
        fio2_cols_needed = list(set(id_columns + [fio2_actual_time, fio2_val_col]))
        o2_subset = o2_tbl.data[[c for c in o2_cols_needed if c in o2_tbl.data.columns]].copy()
        fio2_subset = fio2_tbl.data[[c for c in fio2_cols_needed if c in fio2_tbl.data.columns]].copy()

        # Rename to unified column names
        if o2_actual_time != unified_time_col and o2_actual_time in o2_subset.columns:
            o2_subset.rename(columns={o2_actual_time: unified_time_col}, inplace=True)
        if fio2_actual_time != unified_time_col and fio2_actual_time in fio2_subset.columns:
            fio2_subset.rename(columns={fio2_actual_time: unified_time_col}, inplace=True)
        if o2_val_col != o2_col and o2_val_col in o2_subset.columns:
            o2_subset.rename(columns={o2_val_col: o2_col}, inplace=True)
        if fio2_val_col != fio2_col and fio2_val_col in fio2_subset.columns:
            fio2_subset.rename(columns={fio2_val_col: fio2_col}, inplace=True)

        # FiO2 unit standardization
        if database is not None and not fio2_subset.empty and fio2_col in fio2_subset.columns:
            fio2_subset = _standardize_fio2_units(fio2_subset, fio2_col, database)

        # dropna + sort
        o2_subset = o2_subset.dropna(subset=[unified_time_col])
        fio2_subset = fio2_subset.dropna(subset=[unified_time_col])

        if o2_subset.empty:
            return pd.DataFrame(columns=id_columns + [index_column]), id_columns, index_column

        if fio2_subset.empty:
            merged = o2_subset.copy()
            merged[fio2_col] = float('nan')
            if fix_na_fio2:
                merged[fio2_col] = 21.0
            return merged, id_columns, index_column

        # 🚀 Determine tolerance: use numeric tolerance for numeric time columns
        o2_time_is_numeric = pd.api.types.is_numeric_dtype(o2_subset[unified_time_col])
        fio2_time_is_numeric = pd.api.types.is_numeric_dtype(fio2_subset[unified_time_col])

        if o2_time_is_numeric and fio2_time_is_numeric:
            # Time is in hours — use numeric tolerance directly (skip datetime conversion)
            effective_tolerance = match_win.total_seconds() / 3600.0
        else:
            # Time is datetime — convert numeric side if mixed
            base_time = pd.Timestamp('2000-01-01')
            if o2_time_is_numeric:
                o2_subset[unified_time_col] = base_time + pd.to_timedelta(o2_subset[unified_time_col], unit='h')
            else:
                o2_subset[unified_time_col] = pd.to_datetime(o2_subset[unified_time_col], errors='coerce')
            if fio2_time_is_numeric:
                fio2_subset[unified_time_col] = base_time + pd.to_timedelta(fio2_subset[unified_time_col], unit='h')
            else:
                fio2_subset[unified_time_col] = pd.to_datetime(fio2_subset[unified_time_col], errors='coerce')
            effective_tolerance = match_win

        sort_cols = id_columns + [unified_time_col]
        # 🔧 Align by-column dtypes before merge_asof (int32 vs int64 from different DuckDB sources)
        for col in id_columns:
            if col in o2_subset.columns and col in fio2_subset.columns:
                if o2_subset[col].dtype != fio2_subset[col].dtype:
                    common_dtype = np.result_type(o2_subset[col].dtype, fio2_subset[col].dtype)
                    o2_subset[col] = o2_subset[col].astype(common_dtype)
                    fio2_subset[col] = fio2_subset[col].astype(common_dtype)

        o2_subset = o2_subset.sort_values(by=sort_cols, kind='mergesort').reset_index(drop=True)
        fio2_subset = fio2_subset.sort_values(by=sort_cols, kind='mergesort').reset_index(drop=True)

        # 🚀 Single backward merge_asof with per-patient time offset
        # R ricu: merge(o2, fio2, roll = match_win) = single backward rolling join.
        # Per-patient offset avoids 'by' parameter (which fails on dtype mismatches)
        # and cross-patient isolation is guaranteed by offset >> tolerance.
        merge_cols = id_columns + [unified_time_col, o2_col, fio2_col]
        id_col_name = id_columns[0] if id_columns else None

        if id_col_name is not None:
            # Compute numeric hours for offset computation
            if pd.api.types.is_numeric_dtype(o2_subset[unified_time_col]):
                o2_t = o2_subset[unified_time_col].values.astype(np.float64)
                fio2_t = fio2_subset[unified_time_col].values.astype(np.float64)
            else:
                _base = pd.Timestamp('2000-01-01')
                o2_t = (o2_subset[unified_time_col] - _base).dt.total_seconds().values / 3600.0
                fio2_t = (fio2_subset[unified_time_col] - _base).dt.total_seconds().values / 3600.0
            tol_hours = match_win.total_seconds() / 3600.0

            # Build per-patient offset (data already sorted by pid+time → global time is sorted)
            all_pids = np.union1d(o2_subset[id_col_name].unique(), fio2_subset[id_col_name].unique())
            pid_rank = dict(zip(sorted(all_pids), range(len(all_pids))))
            o2_prank = o2_subset[id_col_name].map(pid_rank).values
            fio2_prank = fio2_subset[id_col_name].map(pid_rank).values
            tr = max(np.ptp(o2_t[np.isfinite(o2_t)]) if np.isfinite(o2_t).any() else 0,
                     np.ptp(fio2_t[np.isfinite(fio2_t)]) if np.isfinite(fio2_t).any() else 0)
            offset = max(tr, 1000000.0) * 10 + 100

            o2_g = o2_t + o2_prank * offset
            fio2_g = fio2_t + fio2_prank * offset

            # Bidirectional merge_asof on global time (no 'by' parameter, no failure)
            # Both use direction='backward': perspective 1 finds fio2 before o2,
            # perspective 2 finds o2 before fio2. Together they cover all temporal orderings.
            left_o = pd.DataFrame({'_g': o2_g, o2_col: o2_subset[o2_col].values})
            right_f = pd.DataFrame({'_g': fio2_g, fio2_col: fio2_subset[fio2_col].values})
            result_fwd = pd.merge_asof(left_o, right_f, on='_g', tolerance=tol_hours, direction='backward')
            left_f = pd.DataFrame({'_g': fio2_g, fio2_col: fio2_subset[fio2_col].values})
            right_o = pd.DataFrame({'_g': o2_g, o2_col: o2_subset[o2_col].values})
            result_bwd = pd.merge_asof(left_f, right_o, on='_g', tolerance=tol_hours, direction='backward')

            merged = pd.concat([
                pd.DataFrame({
                    id_col_name: o2_subset[id_col_name].values,
                    unified_time_col: o2_subset[unified_time_col].values,
                    o2_col: result_fwd[o2_col].values,
                    fio2_col: result_fwd[fio2_col].values,
                }),
                pd.DataFrame({
                    id_col_name: fio2_subset[id_col_name].values,
                    unified_time_col: fio2_subset[unified_time_col].values,
                    o2_col: result_bwd[o2_col].values,
                    fio2_col: result_bwd[fio2_col].values,
                }),
            ], ignore_index=True)
        else:
            o2_sorted = o2_subset[[unified_time_col, o2_col]].sort_values(unified_time_col)
            fio2_sorted = fio2_subset[[unified_time_col, fio2_col]].sort_values(unified_time_col)
            fwd = pd.merge_asof(o2_sorted, fio2_sorted, on=unified_time_col, tolerance=effective_tolerance, direction='backward')
            bwd = pd.merge_asof(fio2_sorted, o2_sorted, on=unified_time_col, tolerance=effective_tolerance, direction='backward')
            merged = pd.concat([fwd, bwd], ignore_index=True)

        # Convert datetime back to numeric if we converted above
        if not (o2_time_is_numeric and fio2_time_is_numeric) and (o2_time_is_numeric or fio2_time_is_numeric):
            try:
                merged[unified_time_col] = (
                    pd.to_datetime(merged[unified_time_col], errors='coerce') - base_time
                ) / pd.Timedelta(hours=1)
            except Exception:
                pass
            
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
    database: str = None,  # 数据库名称，用于FiO2单位转换
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
        ctx=ctx,  # Pass callback context directly
        database=database  # Pass database for FiO2 unit conversion
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
    """
    R ricu logic:
    res <- merge(res[[fio2_var]], expand(res[[vent_var]], aggregate = "any"), all = TRUE)
    res <- res[, c("supp_o2", vent_var, fio2_var) := list(
      is_true(get(vent_var) | get(fio2_var) > 21), NULL, NULL
    )]
    
    Key points:
    1. expand() only expands vent_ind windows to hourly points WITHIN the window
    2. merge(all=TRUE) is a full outer join
    3. is_true() only keeps TRUE values
    """
    vent_tbl = tables["vent_ind"]
    fio2_tbl = tables["fio2"]
    
    # Get fio2 data - this provides the base timeline
    fio2_df = fio2_tbl.data.copy()
    fio2_index = fio2_tbl.index_column or "charttime"
    
    # Detect ID columns
    id_columns = fio2_tbl.id_columns if hasattr(fio2_tbl, 'id_columns') and fio2_tbl.id_columns else []
    if not id_columns:
        id_columns = [c for c in fio2_df.columns if c in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']]
    
    # Expand vent_ind WinTbl to hourly points WITHIN windows only (vectorized)
    if isinstance(vent_tbl, WinTbl):
        vent_df = vent_tbl.data.copy()
        vent_id_cols = vent_tbl.id_vars if hasattr(vent_tbl, 'id_vars') else id_columns
        start_col = vent_tbl.index_var if hasattr(vent_tbl, 'index_var') else "starttime"
        dur_col = vent_tbl.dur_var if hasattr(vent_tbl, 'dur_var') else "dur_var"
        
        vent_df = _expand_wintbl_to_hourly(vent_df, vent_id_cols, start_col, dur_col, fio2_index)
    else:
        vent_df = vent_tbl.data.copy()
        vent_index = vent_tbl.index_column if hasattr(vent_tbl, 'index_column') and vent_tbl.index_column else fio2_index
        # Rename vent index to match fio2
        if vent_index != fio2_index and vent_index in vent_df.columns:
            vent_df = vent_df.rename(columns={vent_index: fio2_index})
        # Find vent_ind column
        vent_col = 'vent_ind'
        if 'vent_ind' not in vent_df.columns:
            for col in vent_df.columns:
                if col not in id_columns and col != fio2_index:
                    vent_col = col
                    vent_df = vent_df.rename(columns={col: 'vent_ind'})
                    break
    
    # Prepare fio2 - find value column
    fio2_col = 'fio2'
    if 'fio2' not in fio2_df.columns:
        for col in fio2_df.columns:
            if col not in id_columns and col != fio2_index:
                fio2_col = col
                fio2_df = fio2_df.rename(columns={col: 'fio2'})
                break
    
    # Ensure numeric fio2
    fio2_df['fio2'] = pd.to_numeric(fio2_df['fio2'], errors='coerce')
    
    # Merge with full outer join (R: merge(..., all=TRUE))
    merge_cols = [c for c in id_columns if c in vent_df.columns and c in fio2_df.columns] + [fio2_index]
    merge_cols = [c for c in merge_cols if c in vent_df.columns and c in fio2_df.columns]
    
    if merge_cols:
        result = pd.merge(fio2_df, vent_df, on=merge_cols, how='outer')
    else:
        result = pd.merge(fio2_df, vent_df, on=[fio2_index], how='outer')
    
    # Fill NaN values
    result['vent_ind'] = result['vent_ind'].fillna(False)
    result['fio2'] = result['fio2'].fillna(21.0)
    
    # Calculate supp_o2: is_true(vent_ind | fio2 > 21)
    supp_mask = result['vent_ind'].astype(bool) | (result['fio2'] > 21.0)
    
    # R ricu's is_true() only keeps TRUE values
    result = result[supp_mask].copy()
    result['supp_o2'] = True
    
    # Select output columns
    output_cols = [c for c in id_columns if c in result.columns] + [fio2_index, 'supp_o2']
    result = result[output_cols].drop_duplicates()
    
    return _as_icutbl(result, id_columns=id_columns, index_column=fio2_index, value_column="supp_o2")

def _callback_supp_o2_aumc(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """AUMC-specific supplemental oxygen callback.

    AUMC database lacks FiO2 data (itemid 12279 is empty), so we rely only on
    mechanical ventilation indicator to determine supplemental oxygen use.
    """
    vent_tbl = tables["vent_ind"]

    # Handle ID and time columns
    id_columns = vent_tbl.id_columns or []
    index_column = vent_tbl.index_column or "starttime"
    vent_col = vent_tbl.value_column or "vent_ind"

    vent_df = vent_tbl.data.copy()

    # For AUMC, supplemental oxygen is equivalent to mechanical ventilation
    # since we don't have reliable FiO2 data
    vent_df["supp_o2"] = vent_df[vent_col].astype(bool)

    result_cols = id_columns + ([index_column] if index_column else []) + ["supp_o2"]
    result_df = vent_df[result_cols].reset_index(drop=True)

    return _as_icutbl(
        result_df,
        id_columns=id_columns,
        index_column=index_column,
        value_column="supp_o2"
    )

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

        # 🔧 FIX 2024-11-29: Match R ricu expand() behavior
        # R ricu: end = re_time(start + dur, interval) -- NO subtraction!
        # R seq(start, end, step) is INCLUSIVE on both ends
        # easyicu expand() also uses inclusive end (after ts_utils fix)
        # So end = start + dur is correct
        work["_end_dt"] = work["_start_dt"] + work["vent_dur_td"]
        
        # 🔧 FIX 2024-11-30: Match R ricu's end < 0 correction
        # R ricu code: x <- x[get(end_var) < 0, c(end_var) := as.difftime(0, units = time_unit)]
        # This ensures windows with negative end times extend to time 0
        # Example: start=-5h, dur=1h → original_end=-4h → corrected_end=0h → covers -5,-4,-3,-2,-1,0
        #
        # The base timestamp represents time 0 (ICU admission)
        # _coerce_time converts numeric hours to datetime as: base + timedelta(hours=value)
        # So time 0 = base, negative times < base
        base = pd.Timestamp("1970-01-01")
        negative_end_mask = work["_end_dt"] < base
        if negative_end_mask.any():
            work.loc[negative_end_mask, "_end_dt"] = base
        
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

    def _windows_from_mech_as_wintbl(mech: ICUTable | WinTbl) -> Optional[WinTbl]:
        """Return win_tbl format for mech_vent data (matching R ricu behavior).
        
        R ricu's vent_ind callback with mech_vent:
        1. vent_ind = !is.na(mech_vent)
        2. change_interval(res, final_int)
        3. return res (preserves win_tbl format with starttime + dur_var)
        """
        df = mech.data.copy()
        if df.empty:
            return None

        # 🔧 FIX 2026-03-10: For WinTbl mech_vent, every row IS a ventilation window
        # by definition. R ricu: vent_ind = rep(TRUE, .N) — ignore the value column.
        # The mech_vent value may be NaN (e.g., MIIV procedureevents), but the row's
        # existence means mechanical ventilation is happening.
        if isinstance(mech, WinTbl):
            vent_mask = pd.Series(True, index=df.index)
        else:
            value_col = "mech_vent"
            if hasattr(mech, 'value_column') and mech.value_column:
                value_col = mech.value_column
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

        # 获取时间列和持续时间列
        # 🔧 FIX 2025-02-14: WinTbl uses index_var, ICUTable uses index_column
        if isinstance(mech, WinTbl):
            idx_col = mech.index_var
        elif hasattr(mech, 'index_column') and mech.index_column:
            idx_col = mech.index_column
        else:
            idx_col = None
        if idx_col is None or idx_col not in df.columns:
            for candidate in ["charttime", "starttime", "time"]:
                if candidate in df.columns:
                    idx_col = candidate
                    break
        if idx_col not in df.columns:
            return None

        # 获取 dur_var 列
        dur_col = None
        if isinstance(mech, WinTbl) and mech.dur_var and mech.dur_var in df.columns:
            dur_col = mech.dur_var
        else:
            # 🔧 FIX 2025-02-14: Check for 'dur_var' column first (R ricu naming convention)
            for candidate in ["dur_var", "mech_vent_dur", "duration", "dur", "endtime", "end_time", "stop", "end"]:
                if candidate in df.columns:
                    dur_col = candidate
                    break

        # 🔧 FIX: 转换时间为相对小时数
        time_values = df[idx_col]
        if pd.api.types.is_datetime64_any_dtype(time_values):
            # 需要转换为相对小时数
            time_hours = _relative_hours(df, idx_col)
        elif pd.api.types.is_timedelta64_dtype(time_values):
            time_hours = time_values.dt.total_seconds() / 3600.0
        else:
            time_hours = pd.to_numeric(time_values, errors="coerce")

        # 🔧 FIX: dur_var must be in HOURS to match start_time unit (hours)
        # After _align_time_to_admission, numeric dur_var from concept loading is already in hours.
        # For timedelta and match_win fallback, convert to hours explicitly.
        if dur_col is not None:
            dur_values = df[dur_col]
            if pd.api.types.is_timedelta64_dtype(dur_values):
                dur_hours = dur_values.dt.total_seconds() / 3600.0
            else:
                # dur_var already converted to hours by _align_time_to_admission
                dur_hours = pd.to_numeric(dur_values, errors="coerce")
        else:
            # 默认持续时间为 match_win（转换为小时）
            dur_hours = match_win.total_seconds() / 3600.0

        # 🔧 创建 win_tbl 格式输出
        result_df = pd.DataFrame()
        for col in id_columns:
            if col in df.columns:
                result_df[col] = df[col].values
        result_df["starttime"] = time_hours.values
        result_df["dur_var"] = dur_hours if isinstance(dur_hours, (int, float)) else dur_hours.values
        result_df["vent_ind"] = True
        
        # 按时间聚合 (change_interval 行为)
        # 取整到小时
        result_df["starttime"] = (result_df["starttime"] // 1).astype(int)
        
        # 按 ID 和 starttime 分组，取 max dur_var
        group_cols = list(id_columns) + ["starttime"]
        result_df = result_df.groupby(group_cols, as_index=False).agg({
            "dur_var": "max",
            "vent_ind": "any"
        }).reset_index(drop=True)

        return WinTbl(
            data=result_df,
            id_vars=list(id_columns),
            index_var="starttime",
            dur_var="dur_var"
        )

    def _windows_from_mech(mech: ICUTable | WinTbl) -> Optional[ICUTable]:
        df = mech.data.copy()
        if df.empty:
            return None

        # 🔧 FIX 2026-03-10: For WinTbl, every row IS a ventilation window by definition.
        if isinstance(mech, WinTbl):
            vent_mask = pd.Series(True, index=df.index)
        else:
            value_col = getattr(mech, 'value_column', None) or "mech_vent"
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

        # 🔧 CRITICAL FIX 2024-12-16: Check if mech_vent is already expanded
        # R ricu behavior: if mech_vent has data, use it directly as vent_ind = !is.na(mech_vent)
        # Do NOT re-expand already-expanded data!
        #
        # Detection: if mech_vent has NO duration/endtime columns, it's already expanded
        duration_cols = [col for col in ("mech_vent_dur", "duration", "dur", "endtime", "end_time", "stop", "end") 
                        if col in df.columns]
        
        if not duration_cols:
            # Already expanded - just set vent_ind = True for all rows
            result = df[[idx_col] + id_columns].copy()
            result["vent_ind"] = True
            result = result.rename(columns={idx_col: time_column})
            
            # Group by ID and time to remove duplicates
            group_cols = list(id_columns) + [time_column]
            result = result.groupby(group_cols, as_index=False)["vent_ind"].any()
            result = result.reset_index(drop=True)
            return _as_icutbl(result, id_columns=id_columns, index_column=time_column, value_column="vent_ind")

        # Not expanded yet - need to expand windows
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
            # 🔥 首先检查 mech_vent_dur 列（MIMIC-IV mech_vent 使用这个列名）
            dur_col = next(
                (col for col in ("mech_vent_dur", "duration", "dur") if col in df.columns),
                None,
            )
            if dur_col is not None:
                # 🔧 FIX: 根据列类型决定如何转换
                col_data = df[dur_col]
                
                # Case 1: 如果已经是 timedelta 类型（MIIV），直接使用
                if pd.api.types.is_timedelta64_dtype(col_data):
                    dur_series = col_data
                else:
                    # Case 2: 数值类型（AUMC/eICU 经过 concept.py 转换后是小时）
                    # 需要指定单位为小时
                    dur_values = pd.to_numeric(col_data, errors="coerce")
                    dur_series = pd.to_timedelta(dur_values, unit="h")
            else:
                # 其次检查 endtime 列
                end_col = next(
                    (col for col in ("endtime", "end_time", "stop", "end") if col in df.columns),
                    None,
                )
                if end_col is not None:
                    end_hours = _relative_hours(df, end_col)
                    dur_hours = end_hours - start_hours
                    dur_series = pd.to_timedelta(dur_hours, unit="h")

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
            # 🔥 R ricu 的 calc_dur 逻辑:
            # calc_dur <- function(x, y) fifelse(is.na(y), x + match_win, y - x)
            # 其中 x = vent_start.time (ICU 入院后的小时数), y = vent_end.time (如果没匹配到是 NA)
            # 如果没匹配到 vent_end: dur = start_hours + match_win（不是 match_win！）
            # 如果匹配到 vent_end: dur = end - start
            # 这样窗口结束时间 = start + dur = 2*start + match_win（如果没匹配）
            # 这导致"连锁"效应：密集的 vent_start 事件会产生相互重叠的大窗口
            
            # 获取 start 时间相对于 epoch 的小时数（这就是 R ricu 的 ICU 入院后时间）
            # 因为 _coerce_time 将数值型时间转换为 epoch + timedelta(hours=value)
            epoch = pd.Timestamp("1970-01-01")
            start_hours = (merged["_start_dt"] - epoch).dt.total_seconds() / 3600.0
            
            # 如果匹配到 vent_end: dur = end - start
            # 如果没匹配到: dur = start_hours + match_win_hours（R ricu 的行为）
            matched_mask = merged["_end_dt"].notna()
            match_win_hours = match_win.total_seconds() / 3600.0
            
            # 初始化持续时间列为 timedelta 类型（避免 FutureWarning）
            merged["vent_dur_td"] = pd.to_timedelta(pd.Series(dtype=float), unit="h")
            
            # 匹配到的情况: dur = end - start
            if matched_mask.any():
                matched_dur = merged.loc[matched_mask, "_end_dt"] - merged.loc[matched_mask, "_start_dt"]
                merged.loc[matched_mask, "vent_dur_td"] = matched_dur.values
            
            # 没匹配到的情况: dur = start_hours + match_win (R ricu 的 calc_dur 行为)
            # 使用 start_hours（相对于 epoch），这等于 ICU 入院后的小时数
            if (~matched_mask).any():
                unmatched_dur = pd.to_timedelta(start_hours.loc[~matched_mask] + match_win_hours, unit="h")
                merged.loc[~matched_mask, "vent_dur_td"] = unmatched_dur.values
        else:
            merged = start_sorted.copy()
            # 当没有任何 vent_end 数据时，也使用 R ricu 的 calc_dur 逻辑
            epoch = pd.Timestamp("1970-01-01")
            start_hours = (merged["_start_dt"] - epoch).dt.total_seconds() / 3600.0
            match_win_hours = match_win.total_seconds() / 3600.0
            merged["vent_dur_td"] = pd.to_timedelta(start_hours + match_win_hours, unit="h")

        # 🔧 FIX 2024-11-30: Match R ricu's min_length filter
        # R ricu code: res <- res[get(var) >= min_length, ]
        # This FILTERS OUT rows where dur < min_length, NOT clips them!
        # Example: start_hours=-7, dur = -7 + 6 = -1 hour → filtered out (not kept)
        # Previously we used clip() which would keep these rows with dur=min_length
        merged["vent_dur_td"] = pd.to_timedelta(merged["vent_dur_td"], errors="coerce")
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

    # 🔥 R ricu vent_ind 逻辑:
    # 如果 mech_vent 有数据 → 只使用 mech_vent，返回 win_tbl 格式
    # 否则 → 使用 vent_start + vent_end 匹配，返回 win_tbl 格式
    # 
    # 🔧 CRITICAL FIX 2025-02-14: R ricu returns win_tbl format, NOT expanded ts_tbl!
    # Gold standard: stay_id, starttime, dur_var, vent_ind
    # EasyICU was: stay_id, charttime, vent_ind (expanded hourly)
    # 
    # 参考 R 代码:
    #   if (has_rows(res[[3L]])) {  # mech_vent
    #     res <- res[[3L]][, c("vent_ind", "mech_vent") := ...]
    #     res <- change_interval(res, final_int, by_ref = TRUE)
    #     return(res)  # Returns win_tbl format (mech_vent is already win_tbl)
    #   }
    
    # 🔧 PRIORITY: Try win_tbl format first (matches R ricu behavior)
    if mech_tbl is not None and not mech_tbl.data.empty:
        wintbl_result = _windows_from_mech_as_wintbl(mech_tbl)
        if wintbl_result is not None and not wintbl_result.data.empty:
            return wintbl_result
    
    # Fallback to expanded format if win_tbl not available
    mech_result = None
    if mech_tbl is not None and not mech_tbl.data.empty:
        mech_result = _normalize_result(_windows_from_mech(mech_tbl))
    
    # 🔥 关键修复: 如果 mech_vent 有结果，直接返回，不合并 vent_start/vent_end
    if mech_result is not None:
        return mech_result

    # 只有当 mech_vent 没有数据时，才使用 vent_start/vent_end
    event_result = None
    if start_tbl is not None and not start_tbl.data.empty:
        event_result = _normalize_result(_windows_from_events(start_tbl, end_tbl))

    if event_result is None:
        return _empty_result()

    return event_result

def _urine24_batch(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str,
    time_step: float,
    window_steps: int,
    min_steps: int,
    step_factor: float,
) -> pd.DataFrame:
    """Batch-vectorized urine24 computation for all patients at once.
    
    🚀 Performance: Replaces per-patient loop (4796 calls to process_patient = 13.3s)
    with batch operations:
    - 1 groupby().agg() for patient stats
    - 1 merge for urine values onto grid
    - 1 groupby().rolling() for rolling sum
    
    Expected speedup: 13.3s → ~2-3s (5-6x faster)
    
    Args:
        df: Input DataFrame with urine data
        id_col: Patient ID column name
        time_col: Time column name (numeric)
        urine_col: Urine value column name
        time_step: Time step size (in time column units)
        window_steps: Rolling window size in steps (typically 24)
        min_steps: Minimum window steps for non-NA output (typically 12)
        step_factor: Scaling factor (typically 24.0)
    """
    # Step 1: Compute per-patient stats
    stats = df.groupby(id_col)[time_col].agg(['min', 'max']).reset_index()
    stats['ricu_end'] = stats['max'] - stats['min']  # ricu buggy behavior: duration as end
    
    # Step 2: Generate time grids for all patients (vectorized with numpy)
    # Each patient needs: arange(start, ricu_end + step, step)
    grid_sizes = np.maximum(1, np.floor(
        (stats['ricu_end'].values - stats['min'].values) / time_step + 1
    ).astype(int))
    
    # Handle edge case: ricu_end < start (duration = 0)
    grid_sizes = np.where(stats['ricu_end'].values >= stats['min'].values, grid_sizes, 1)
    
    total_points = grid_sizes.sum()
    
    # Pre-allocate arrays for speed
    all_times = np.empty(total_points, dtype=np.float64)
    all_ids = np.empty(total_points, dtype=df[id_col].dtype)
    
    offset = 0
    for i in range(len(stats)):
        n = grid_sizes[i]
        start = stats.iloc[i]['min']
        end = stats.iloc[i]['ricu_end']
        if end >= start:
            times = np.arange(start, end + time_step * 0.5, time_step)[:n]
        else:
            times = np.array([start])
            n = 1
        actual_n = len(times)
        all_times[offset:offset + actual_n] = times
        all_ids[offset:offset + actual_n] = stats.iloc[i][id_col]
        offset += actual_n
    
    full_grid = pd.DataFrame({
        id_col: all_ids[:offset],
        time_col: all_times[:offset],
    })
    
    # Step 3: Merge original urine values onto grid (single merge!)
    orig = df[[id_col, time_col, urine_col]].drop_duplicates([id_col, time_col])
    full_grid = full_grid.merge(orig, on=[id_col, time_col], how='left')
    full_grid[urine_col] = full_grid[urine_col].fillna(0.0)
    full_grid = full_grid.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Step 4: Rolling sum using groupby().rolling() (single operation!)
    rolling_sum = (
        full_grid
        .groupby(id_col, sort=True)[urine_col]
        .rolling(window=window_steps, min_periods=min_steps, center=False)
        .sum()
    )
    # Drop group level from MultiIndex to align with full_grid
    if isinstance(rolling_sum.index, pd.MultiIndex):
        rolling_sum = rolling_sum.droplevel(0)
    rolling_sum_vals = rolling_sum.values
    
    # Step 5: Calculate urine24 = rolling_sum * step_factor / actual_window_length
    cumcount = full_grid.groupby(id_col).cumcount().values + 1
    actual_window_lens = np.minimum(cumcount, window_steps)
    
    full_grid['urine24'] = np.where(
        (actual_window_lens >= min_steps) & np.isfinite(rolling_sum_vals),
        rolling_sum_vals * step_factor / actual_window_lens,
        np.nan
    )
    
    return full_grid[[id_col, time_col, 'urine24']]


def _callback_urine24(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """
    Calculate 24-hour urine output (R ricu urine24 callback).
    
    Replicates R ricu's urine24 logic:
    1. fill_gaps: Expand time series to hourly grid
       - Uses collapse(min, max) to get the full time range
       - Fills all hours between min and max
    2. slide: Apply sliding window with urine_sum function
       - Window is 24 hours lookback (left_closed=True)
       - min_win = 12 hours (minimum window length for non-NA output)
       - Formula: sum(x) * step_factor / length(x)
       - step_factor = 24 (converts to 24h equivalent)
       - length(x) = number of rows in window (not number of non-zero values)
    """
    # Load urine if not in tables - 🚀 优化：使用 get_raw_concept 缓存
    if "urine" not in tables:
        try:
            urine_tbl = None
            if hasattr(ctx.resolver, 'get_raw_concept'):
                urine_tbl = ctx.resolver.get_raw_concept("urine", ctx.data_source, ctx.patient_ids)
            
            if urine_tbl is not None:
                tables["urine"] = urine_tbl
            else:
                loaded = ctx.resolver.load_concepts(
                    ["urine"],
                    ctx.data_source,
                    merge=False,
                    aggregate=None,
                    patient_ids=ctx.patient_ids,
                    interval=None,  # Load raw data without interval aggregation
                )
                if isinstance(loaded, dict):
                    tables.update(loaded)
                elif isinstance(loaded, ICUTable):
                    tables["urine"] = loaded
        except (KeyError, ValueError):
            # Return empty table if urine cannot be loaded
            cols = ["urine24"]
            return _as_icutbl(pd.DataFrame(columns=cols), id_columns=[], index_column=None, value_column="urine24")
    
    urine_tbl = _ensure_time_index(tables["urine"])
    interval = ctx.interval or pd.Timedelta(hours=1)
    ctx.kwargs.get('min_win', pd.Timedelta(hours=12))
    
    df = urine_tbl.data.copy()
    key_cols = list(urine_tbl.id_columns) + [urine_tbl.index_column]
    if df.empty:
        cols = key_cols + ["urine24"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column="urine24")

    # Prepare columns
    urine_col = urine_tbl.value_column or "urine"
    time_col = urine_tbl.index_column
    id_cols = list(urine_tbl.id_columns) if urine_tbl.id_columns else []
    
    if urine_col not in df.columns:
        df[urine_col] = 0.0
    df[urine_col] = pd.to_numeric(df[urine_col], errors="coerce").fillna(0.0)
    
    is_numeric_time = pd.api.types.is_numeric_dtype(df[time_col])
    interval_hours = interval.total_seconds() / 3600.0
    
    # Detect numeric time unit: SIC uses seconds (3600s/hr), others use hours (1.0/hr)
    # Use per-patient diffs to avoid inter-patient zero-diffs contaminating the median
    numeric_time_step = interval_hours  # default: 1 hour step
    if is_numeric_time and len(df) > 1 and id_cols:
        _per_pt_diffs = df.sort_values([id_cols[0], time_col]).groupby(id_cols[0])[time_col].diff().dropna()
        _pos_diffs = _per_pt_diffs[_per_pt_diffs > 0]
        if len(_pos_diffs) > 0:
            median_diff = _pos_diffs.median()
            # If median positive per-patient diff >= 60, time is in seconds (e.g., SIC: 3600)
            if median_diff >= 60:
                numeric_time_step = median_diff  # e.g., 3600 for SIC
    
    # Constants for ricu algorithm
    # min_win = 12 hours，应转换为“步数”而不是固定写死 12。
    # 例如 interval=6h 时，min_steps 应为 2，而不是 12。
    step_factor = 24.0  # step_factor = 24
    
    # Compute window size in integer steps (same for all patients)
    # CRITICAL: R ricu's slide_index(.before = hours(24)) uses a CLOSED interval [t-24h, t],
    # which on an hourly grid includes 25 entries (not 24).
    # We add +1 to window_steps to match R's closed-interval semantics.
    # min_steps stays at ceil(12/step) without +1 to match R's min_win behavior.
    if is_numeric_time:
        step_hours = numeric_time_step / 3600.0 if numeric_time_step >= 60 else float(numeric_time_step)
        window_steps = max(1, int(np.ceil(24.0 / step_hours)) + 1)
        min_steps = max(1, int(np.ceil(12.0 / step_hours)))
    else:
        step_hours = interval.total_seconds() / 3600.0
        window_steps = max(1, int(np.ceil(24.0 / step_hours)) + 1)
        min_steps = max(1, int(np.ceil(12.0 / step_hours)))

    min_steps = min(min_steps, window_steps)
    
    n_patients = df[id_cols[0]].nunique() if id_cols else 1
    
    # 🚀 BATCH FAST PATH: process all patients in a single merge + rolling
    # cProfile: per-patient loop (4796 calls to process_patient) = 13.3s
    # Batch: single merge + single groupby().rolling() ≈ 2-3s
    if id_cols and len(id_cols) == 1 and is_numeric_time and n_patients > 20:
        try:
            result_df = _urine24_batch(
                df, id_cols[0], time_col, urine_col,
                numeric_time_step, window_steps, min_steps, step_factor
            )
            output_cols = id_cols + [time_col, 'urine24']
            available_cols = [c for c in output_cols if c in result_df.columns]
            return _as_icutbl(
                result_df[available_cols],
                id_columns=urine_tbl.id_columns,
                index_column=urine_tbl.index_column,
                value_column="urine24"
            )
        except Exception:
            pass  # Fall through to per-patient loop
    
    # === PER-PATIENT LOOP (fallback for small groups, datetime, multi-id) ===
    def process_patient(group):
        """Process urine24 for a single patient (ricu-compatible)."""
        group = group.sort_values(time_col).reset_index(drop=True)
        original_times = group[time_col].values
        original_urine = group[urine_col].values
        
        if len(original_times) == 0:
            return pd.DataFrame(columns=[time_col, urine_col, 'urine24'] + id_cols)
        
        start_time = original_times[0]
        actual_end_time = original_times[-1]
        duration = actual_end_time - start_time
        
        # CRITICAL: Match ricu's buggy behavior
        if is_numeric_time:
            ricu_end_time = duration
            time_grid = np.arange(start_time, ricu_end_time + numeric_time_step, numeric_time_step)
        else:
            ricu_end_time = pd.Timedelta(hours=duration) if isinstance(duration, (int, float)) else duration
            time_grid = pd.date_range(start=start_time, end=start_time + ricu_end_time, freq=interval)
        
        filled_df = pd.DataFrame({time_col: time_grid})
        orig_df = pd.DataFrame({time_col: original_times, urine_col: original_urine})
        filled_df = filled_df.merge(orig_df, on=time_col, how='left')
        filled_df[urine_col] = filled_df[urine_col].fillna(0.0)
        filled_df = filled_df.sort_values(time_col).reset_index(drop=True)
        
        n = len(filled_df)
        urine_vals = filled_df[urine_col].values
        
        rolling_sum = pd.Series(urine_vals).rolling(
            window=window_steps, min_periods=min_steps, center=False
        ).sum()
        
        positions = np.arange(n) + 1
        actual_window_lens = np.minimum(positions, window_steps)
        
        urine24_values = np.where(
            (actual_window_lens >= min_steps) & pd.notna(rolling_sum.values),
            rolling_sum.values * step_factor / actual_window_lens,
            np.nan
        )
        
        filled_df['urine24'] = urine24_values
        
        for col in id_cols:
            if col in group.columns:
                filled_df[col] = group[col].iloc[0]
            elif col in df.columns:
                filled_df[col] = df[col].iloc[0]
        
        return filled_df
    
    # Process each patient
    if id_cols:
        df = df.sort_values(id_cols + [time_col]).reset_index(drop=True)
        result_dfs = []
        for keys, group in df.groupby(id_cols, sort=False):
            patient_result = process_patient(group)
            if isinstance(keys, tuple):
                for i, col in enumerate(id_cols):
                    if col not in patient_result.columns:
                        patient_result[col] = keys[i]
            else:
                if id_cols[0] not in patient_result.columns:
                    patient_result[id_cols[0]] = keys
            result_dfs.append(patient_result)
        
        if result_dfs:
            result_df = pd.concat(result_dfs, ignore_index=True)
        else:
            result_df = pd.DataFrame(columns=id_cols + [time_col, 'urine24'])
    else:
        result_df = process_patient(df)
    
    output_cols = id_cols + [time_col, 'urine24']
    available_cols = [c for c in output_cols if c in result_df.columns]
    
    return _as_icutbl(
        result_df[available_cols], 
        id_columns=urine_tbl.id_columns, 
        index_column=urine_tbl.index_column, 
        value_column="urine24"
    )



def _callback_vaso_ind(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """R ricu vaso_ind callback 的精确复制.
    
    R ricu 的 vaso_ind 回调有一个特殊行为：
    1. 计算 pmax(dopa_dur, norepi_dur, dobu_dur, epi_dur) 作为 vaso_ind 列
    2. 调用 expand(res, index_var(res), "vaso_ind")
    3. 由于 "vaso_ind" 列已存在，expand 直接使用它作为 end_var
    4. 这导致 seq(starttime, duration) 而不是 seq(starttime, starttime+duration)
    5. R 的 seq(1, 4.18, 1) = 1,2,3,4（不包含超过 4.18 的值）
    
    我们需要精确复制这个行为。
    """
    # When upstream concepts request hourly alignment (ctx.interval != None),
    # the duration tables may already have their start times floored to the hour.
    # 🚀 优化：使用 get_raw_concept 缓存原始数据，避免重复加载
    if ctx.interval:
        refreshed: Dict[str, ICUTable] = {}
        for name, tbl in tables.items():
            raw_tbl = tbl
            # 尝试从缓存获取原始数据
            if hasattr(ctx.resolver, 'get_raw_concept'):
                cached_raw = ctx.resolver.get_raw_concept(name, ctx.data_source, ctx.patient_ids)
                if cached_raw is not None and not cached_raw.data.empty:
                    raw_tbl = cached_raw
            refreshed[name] = raw_tbl
        tables = refreshed

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
    
    # 🔧 FIX 2025-01: Database-specific time units for vaso_ind
    # After _align_time_to_admission, all databases use hours for relative time
    ds_name = ''
    if ctx is not None:
        ds_cfg = getattr(getattr(ctx, 'data_source', None), 'config', None)
        ds_name = getattr(ds_cfg, 'name', '') if ds_cfg is not None else ''
    numeric_unit = 'h'  # All databases use hours after _align_time_to_admission
    
    if time_is_numeric:
        numeric_time = pd.to_numeric(time_series, errors="coerce")
        merged["__start_dt"] = base_time + pd.to_timedelta(numeric_time, unit=numeric_unit)
    else:
        merged["__start_dt"] = pd.to_datetime(time_series, errors="coerce")

    def _coerce_duration(series: pd.Series) -> pd.Series:
        if pd.api.types.is_timedelta64_dtype(series):
            return series
        # Check if it's datetime type (bug in some duration columns)
        if pd.api.types.is_datetime64_any_dtype(series):
            # This might be a datetime column mistakenly used as duration
            # Try to interpret as offset from base time
            dt_series = pd.to_datetime(series, errors="coerce")
            base = pd.Timestamp("2000-01-01")
            # If values are close to base_time, they might represent durations stored as timestamps
            time_diffs = (dt_series - base).dt.total_seconds()
            # Check if these look like reasonable durations (< 1 year in seconds)
            if time_diffs.notna().any() and (time_diffs[time_diffs.notna()].abs() < 365*24*3600).all():
                return pd.to_timedelta(time_diffs, unit="s", errors="coerce")
            # Otherwise, return NaT for all invalid entries
            return pd.Series([pd.NaT] * len(series), index=series.index, dtype='timedelta64[ns]')
        # For numeric values, assume duration is in HOURS (consistent with eICU/ricu conventions)
        # pd.to_timedelta on raw numbers defaults to nanoseconds, which is wrong
        # First try to convert to numeric - if successful, interpret as hours
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return pd.to_timedelta(numeric, unit="h", errors="coerce")
        # Last resort: try string parsing (e.g., "1 hour", "30 minutes")
        converted = pd.to_timedelta(series, errors="coerce")
        return converted

    for col in vaso_cols:
        merged[col] = _coerce_duration(merged[col])

    # 采用R ricu的pmax逻辑 - 对每行的所有duration取max,只有当某行至少有一个valid duration时才创建vaso_ind
    # R: res <- res[, c("vaso_ind", cnc) := list(pmax(get("dopa_dur"), .
    # 计算每行的max duration (跳过NA)
    merged["__max_duration"] = merged[vaso_cols].max(axis=1, skipna=True)
    
    # 将 duration 转换为小时数
    merged["__duration_hours"] = merged["__max_duration"].dt.total_seconds() / 3600
    # 获取 start 的小时数（相对于 base_time）
    merged["__start_hours"] = (merged["__start_dt"] - base_time).dt.total_seconds() / 3600
    
    # R ricu 的 expand 函数只保留 start <= end 的行
    # 对于 vaso_ind，end_var = vaso_ind = duration（pmax 结果）
    # 所以当 start=0, duration=0 时，0 <= 0 为 TRUE，会保留这一行
    # 修复：允许 duration >= 0（而不是 duration > 0）
    valid_mask = (
        merged["__max_duration"].notna() & 
        (merged["__max_duration"] >= pd.Timedelta(0))
    )
    valid_rows = merged[valid_mask].copy()
    
    if valid_rows.empty:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    # R ricu vaso_ind 的特殊行为：
    # 1. vaso_ind 列被设置为 pmax(durations)，即 duration 值
    # 2. expand(res, index_var(res), "vaso_ind") 被调用
    # 3. 由于 "vaso_ind" 列已存在，expand 直接使用它作为 end_var
    # 4. 所以实际执行的是 seq(starttime, duration) 而不是 seq(starttime, starttime+duration)
    # 5. R 的 seq(1, 4.18, 1) = [1, 2, 3, 4] (不包含超过 4.18 的值)
    # 6. R 的 seq(0, 0, 1) = [0] (当 start=end 时返回单个值)
    #
    # 我们需要模拟这个行为：对于每一行，生成从 start_hour 到 floor(duration_hour) 的序列
    # 注意：R seq(a, b, 1) 生成的是 a, a+1, a+2, ... 直到 <= b
    
    expanded_records: list[tuple] = []
    for _, row in valid_rows.iterrows():
        start_hours = row["__start_hours"]
        duration_hours = row["__duration_hours"]
        # R expand 只检查 start <= end，所以 duration >= 0 都可以
        if pd.isna(start_hours) or pd.isna(duration_hours) or duration_hours < 0:
            continue
        id_values = tuple(row[col] for col in id_columns) if id_columns else tuple()
        
        # R ricu 的行为：seq(start, duration, 1)
        # 例如：start=0, duration=0 → seq(0, 0, 1) = [0]
        # 例如：start=1, duration=4.18 → seq(1, 4.18, 1) = [1, 2, 3, 4]
        # 例如：start=1, duration=6.05 → seq(1, 6.05, 1) = [1, 2, 3, 4, 5, 6]
        start_int = int(start_hours)
        # R 的 seq 行为：生成 start, start+1, start+2, ... 直到值 <= end
        # 所以最大值是 start + floor(end - start) = start + floor(duration - start + start) 的最大整数 <= duration
        # 简化：最大值是 floor(duration)，但不能小于 start
        end_int = int(duration_hours)  # floor(duration)
        
        # 当 start > duration 时，R 会返回空（不满足 start <= end）
        # 但前面的 valid_mask 已经处理了 duration >= 0 的情况
        # 实际上对于 vaso_ind，start 和 duration 应该都 >= 0
        if start_int > duration_hours + 1e-9:
            # start > duration，跳过（R expand 的 start <= end 条件）
            continue
        
        # 生成序列 [start_int, start_int+1, ..., end_int] 如果 end_int >= start_int
        # R seq(1, 4.18, 1) 意味着 seq(start=1, to=4.18, by=1)
        # 结果是 [1, 2, 3, 4] 因为 5 > 4.18
        for hour in range(start_int, end_int + 1):
            if hour <= duration_hours + 1e-9:  # 包含 duration_hours 本身（如果是整数）
                expanded_records.append((*id_values, float(hour)))
    
    if not expanded_records:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    result_cols = list(id_columns) + [time_col]
    expanded = pd.DataFrame(expanded_records, columns=result_cols)
    expanded["vaso_ind"] = True
    
    # 去重（同一患者同一小时只保留一条记录）
    expanded = expanded.drop_duplicates(subset=list(id_columns) + [time_col] if id_columns else [time_col])
    
    result_cols = list(id_columns) + [time_col, "vaso_ind"] if id_columns else [time_col, "vaso_ind"]
    expanded = expanded[result_cols].reset_index(drop=True)
    return _as_icutbl(expanded, id_columns=id_columns, index_column=time_col, value_column="vaso_ind")

def _callback_vaso_ind_rate(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Vasopressor indicator based on rate data (alternative for eICU where duration calculation fails).

    This callback uses vasopressor rate data instead of duration data to determine
    when vasopressors were administered. It's specifically designed for eICU database
    where the duration calculation has issues.
    """
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

    # Create result data: for each time point where any vaso drug has rate > 0, set vaso_ind = True
    result_rows = []

    # Get unique time points
    merged[time_col].dropna().unique()

    # Get interval for time grid generation
    final_interval = ctx.interval
    if isinstance(final_interval, str):
        try:
            final_interval = pd.to_timedelta(final_interval)
        except Exception:
            final_interval = pd.Timedelta(hours=1)
    elif final_interval is None or final_interval <= pd.Timedelta(0):
        final_interval = pd.Timedelta(hours=1)

    # For each ID combination and time point, check if any vaso drug is active
    id_groups = merged[list(id_columns)].drop_duplicates() if id_columns else [pd.Series([None])]

    for _, id_group in id_groups.iterrows() if id_columns else [(None, None)]:
        # Filter data for this ID group
        if id_columns:
            mask = pd.Series([True] * len(merged))
            for col in id_columns:
                mask = mask & (merged[col] == id_group[col])
            group_data = merged[mask]
        else:
            group_data = merged

        if group_data.empty:
            continue

        # Get time range for this ID group
        min_time = group_data[time_col].min()
        max_time = group_data[time_col].max()

        if pd.isna(min_time) or pd.isna(max_time):
            continue

        # Create time grid
        if pd.api.types.is_numeric_dtype(group_data[time_col]):
            time_grid = np.arange(min_time, max_time + final_interval.total_seconds()/3600,
                                 final_interval.total_seconds()/3600)
        else:
            time_grid = pd.date_range(start=min_time, end=max_time, freq=final_interval)

        # For each time point, check if any vaso drug is active
        for time_point in time_grid:
            # Check if any vaso drug has rate > 0 at this time point (or nearest time)
            # Handle both numeric and datetime time columns
            if pd.api.types.is_numeric_dtype(group_data[time_col]):
                # Numeric time column
                time_diff = abs(group_data[time_col] - time_point)
                threshold = final_interval.total_seconds()/7200  # half interval
            else:
                # Datetime/timedelta time column
                # Convert time_point to timedelta if it's numeric hours
                if isinstance(time_point, (int, float)):
                    time_point_td = pd.Timedelta(hours=time_point)
                elif hasattr(time_point, 'total_seconds'):  # Already timedelta-like
                    time_point_td = time_point
                else:
                    # Try to convert from datetime string to timedelta (relative to some base)
                    try:
                        # Check if it's a datetime string that needs conversion
                        if isinstance(time_point, str) and ('-' in time_point or ':' in time_point):
                            # This looks like a datetime string, convert to timedelta relative to start of day
                            time_dt = pd.to_datetime(time_point)
                            time_point_td = pd.Timedelta(hours=time_dt.hour, minutes=time_dt.minute,
                                                      seconds=time_dt.second, microseconds=time_dt.microsecond)
                        else:
                            # Try direct timedelta conversion
                            time_point_td = pd.to_timedelta(time_point)
                    except Exception:
                        # If all conversions fail, use numeric conversion
                        time_point_td = pd.Timedelta(hours=float(str(time_point)))

                # Ensure both operands are timedelta for subtraction
                time_col_vals = pd.to_timedelta(group_data[time_col]) if not pd.api.types.is_timedelta64_dtype(group_data[time_col]) else group_data[time_col]
                time_diff = abs(time_col_vals - time_point_td)
                threshold = final_interval / 2

            time_mask = time_diff <= threshold
            nearby_data = group_data[time_mask]

            has_vaso = False
            for _, row in nearby_data.iterrows():
                for col in vaso_cols:
                    val = row.get(col)
                    if pd.notna(val) and float(val) > 0:
                        has_vaso = True
                        break
                if has_vaso:
                    break

            # Create result row
            result_row = {time_col: time_point, "vaso_ind": has_vaso}
            if id_columns:
                for col in id_columns:
                    result_row[col] = id_group[col]
            result_rows.append(result_row)

    if not result_rows:
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=id_columns,
            index_column=time_col,
            value_column="vaso_ind",
        )

    result_df = pd.DataFrame(result_rows)
    result_cols = list(id_columns) + [time_col, "vaso_ind"] if id_columns else [time_col, "vaso_ind"]
    result_df = result_df[result_cols].reset_index(drop=True)

    return _as_icutbl(result_df, id_columns=id_columns, index_column=time_col, value_column="vaso_ind")

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

def _callback_sep3_sofa2(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    if "sofa2" not in tables or "susp_inf" not in tables:
        return _as_icutbl(
            pd.DataFrame(columns=['stay_id', 'charttime', 'sep3_sofa2']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='sep3_sofa2'
        )

    id_columns, index_column, converted_tables = _assert_shared_schema(
        {"sofa2": tables["sofa2"], "susp_inf": tables["susp_inf"]},
        ctx=ctx,
        convert_ids=True
    )

    if "sofa2" not in converted_tables or "susp_inf" not in converted_tables:
        return _as_icutbl(
            pd.DataFrame(columns=list(id_columns) + ([index_column] if index_column else []) + ['sep3_sofa2']),
            id_columns=id_columns,
            index_column=index_column,
            value_column='sep3_sofa2'
        )

    sofa2_tbl = converted_tables["sofa2"]
    susp_tbl = converted_tables["susp_inf"]

    sofa2_data = sofa2_tbl.data.copy()
    susp_data = susp_tbl.data.copy()

    if sofa2_tbl.index_column and sofa2_tbl.index_column != index_column and sofa2_tbl.index_column in sofa2_data.columns:
        sofa2_data = sofa2_data.rename(columns={sofa2_tbl.index_column: index_column})
    if susp_tbl.index_column and susp_tbl.index_column != index_column and susp_tbl.index_column in susp_data.columns:
        susp_data = susp_data.rename(columns={susp_tbl.index_column: index_column})

    result = sep3_sofa2_detector(
        sofa2=sofa2_data,
        susp_inf_df=susp_data,
        id_cols=list(id_columns),
        index_col=coalesce(sofa2_tbl.index_column, susp_tbl.index_column, index_column),
    )

    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column="sep3_sofa2")

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
    
    # Handle empty input data
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
    
    # 修复：确保index_column在两个DataFrame中都存在
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
            # 🔧 FIX 2025-01: More precise unit column matching
            # Avoid matching ID columns like 'patientunitstayid' which contain 'unit'
            # Only match columns that look like unit columns: 'unit', 'rate_unit', 'drugunit' etc.
            candidate_lower = candidate.lower()
            if candidate_lower in id_columns:
                continue  # Skip ID columns
            # Match 'unit' as a word boundary, not just substring
            if candidate_lower == 'unit' or candidate_lower.endswith('_unit') or candidate_lower.startswith('unit_') or 'rateunit' in candidate_lower or 'drugunit' in candidate_lower:
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
    
    # 🔧 FIX 2025-01: Database-specific time units
    # After sic_dur/sic_rate_kg callbacks, SIC medication Offset is already converted to hours
    # All databases use hours for relative time at this point
    numeric_unit = 'h'  # All databases: hours (SIC medication Offset already converted from seconds)

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

    # 🔧 FIX for ricu compatibility (confirmed from R source callback-cncpt.R, callback-itm.R):
    # R ricu pipeline: hirid_rate_kg → expand_intervals(LOCF per infusion) → MEDIAN per
    # (patient, minute) → vaso60 [non-equi join, change_interval(1h), aggregate("max")].
    # expand_intervals = create_intervals(overhang=1h, max_len=6h) + expand(step=1min)
    # Since EasyICU loads sub-concepts with aggregate=False, we replicate this here.
    rate_df[rate_index_col] = rate_df[rate_index_col].dt.floor('min')

    # Detect infusion group column for per-infusion LOCF
    _infusion_col = None
    for _candidate in ['infusionid', 'orderid', 'linkorderid']:
        if _candidate in rate_df.columns:
            _infusion_col = _candidate
            break

    if _infusion_col is not None and len(rate_df) > 0:
        # Vectorized per-infusion LOCF expansion at minute level
        # (R ricu create_intervals overhang=1h, max_len=6h + expand step=1min)
        _overhang_min = 60
        _max_len_min = 360
        _step = pd.Timedelta(minutes=1)
        _group_cols = id_columns + [_infusion_col]
        rate_df = rate_df.sort_values(_group_cols + [rate_index_col]).reset_index(drop=True)
        # next_time per group
        _next_time = rate_df.groupby(_group_cols, dropna=False)[rate_index_col].shift(-1)
        _gap_min = (_next_time - rate_df[rate_index_col]).dt.total_seconds() / 60.0
        # For last row in group: overhang; otherwise min(gap-1, max_len-1, overhang-1)
        _dur_min = np.where(
            _gap_min.isna(),
            _overhang_min - 1,
            np.minimum.reduce([
                _gap_min.fillna(0).values - 1,
                np.full(len(rate_df), _max_len_min - 1, dtype=float),
                np.full(len(rate_df), _overhang_min - 1, dtype=float),
            ]),
        )
        _dur_min = np.maximum(_dur_min, 0).astype(np.int64)
        _repeat = (_dur_min + 1).astype(np.int64)  # minutes per row (inclusive)
        _total = int(_repeat.sum())
        if _total > 0:
            # Fully vectorized repeat + per-row 0..k-1 offsets
            _idx_repeat = np.repeat(np.arange(len(rate_df)), _repeat)
            # offsets = position within each repeat group (0, 1, ..., _repeat[i]-1)
            _group_start = np.repeat(np.cumsum(_repeat) - _repeat, _repeat)
            _offsets = np.arange(_total) - _group_start
            _expanded = rate_df.iloc[_idx_repeat].reset_index(drop=True)
            _expanded[rate_index_col] = _expanded[rate_index_col] + pd.to_timedelta(_offsets, unit='m')
            rate_df = _expanded
        # Drop infusion column — R ricu expand drops grp_var
        if _infusion_col in rate_df.columns:
            rate_df = rate_df.drop(columns=[_infusion_col])

        # MEDIAN per (patient, minute) — ONLY when LOCF was applied.
        # For LOCF-expanded data, multiple values at same minute = concurrent
        # overlapping infusions → MEDIAN gives the "typical" rate (R ricu behavior).
        # For non-LOCF DBs (SIC/MIMIC/etc.), multiple values at same time represent
        # sequential rate changes → keep them and let final MAX aggregation pick.
        rate_df = (
            rate_df.groupby(id_columns + [rate_index_col], dropna=False)[rate_col]
            .median()
            .reset_index()
        )
    # R ricu's vaso60 uses minute-precision start: end = start + duration
    dur_df[dur_index_col] = dur_df[dur_index_col].dt.floor('min')
    dur_df['__dur_time_min'] = dur_df[dur_index_col]

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
        # Duration is numeric (hours) - this is the common case for MIMIC-IV and most databases
        # Convert numeric durations to timedelta
        # IMPORTANT: pd.to_timedelta() without unit treats numbers as NANOSECONDS, not hours!
        # We need to explicitly specify the unit based on the database
        numeric_durations = pd.to_numeric(durations, errors="coerce")
        
        is_aumc = isinstance(ds_name, str) and ds_name.lower() == 'aumc'
        is_miiv = isinstance(ds_name, str) and ds_name.lower() in ('miiv', 'mimic', 'mimic_demo')
        
        if is_aumc or is_miiv:
            # AUMC and MIMIC: duration is in hours
            durations = pd.to_timedelta(numeric_durations, unit="h", errors="coerce")
        else:
            # Other databases: try hours first (most common for ICU data),
            # then fall back to minutes if hours gives unreasonably large values
            hours_based = pd.to_timedelta(numeric_durations, unit="h", errors="coerce")
            # Check if values are reasonable (< 1000 hours = ~41 days max stay)
            if hours_based.notna().any() and (hours_based.max() < pd.Timedelta(hours=1000) if hours_based.notna().any() else True):
                durations = hours_based
            else:
                # Try minutes
                minutes_based = pd.to_timedelta(numeric_durations, unit="m", errors="coerce")
                if minutes_based.notna().any():
                    durations = minutes_based
                else:
                    # Fall back to seconds
                    durations = pd.to_timedelta(numeric_durations, unit="s", errors="coerce")

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

    # R ricu vaso60: start at minute precision, end = start + duration
    dur_df["__start"] = dur_df[dur_index_col]
    dur_df["__end"] = dur_df[dur_index_col] + dur_df["__duration"]

    max_gap = pd.Timedelta(minutes=5)

    # Filter id_columns to only include columns that actually exist in dur_df
    # This handles cases where ID columns were filtered out during processing (e.
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

    # 🔧 FIX: Carry forward rate values by +1 hour for MIMIC CareVue data
    # R ricu's expand_intervals expands rate records at 1-minute resolution.
    # A rate charted at fractional relative hour H (e.g., 30.866h) is expanded
    # for ~59 minutes, spanning into hour H+1. After flooring to hours and
    # taking max, hour H+1 gets the HIGHER of old and new rate.
    # easyicu's expand_intervals uses 1-hour steps, losing this cross-boundary effect.
    # This only affects MIMIC CareVue (mimic_rate_cv → expand_intervals path).
    # Other databases use different callbacks with proper start/stop expansion.
    if ds_name.lower() in ('mimic', 'mimic_demo'):
        rate_shifted = rate_df.copy()
        rate_shifted[rate_index_col] = rate_shifted[rate_index_col] + pd.Timedelta(hours=1)
        rate_df = pd.concat([rate_df, rate_shifted], ignore_index=True)

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

    if final_interval is not None and not grouped.empty:
        # 🔧 R ricu's vaso60: change_interval(res, final_int) then aggregate(res, "max")
        # = floor to hours + MAX per (patient, hour). No LOCF, no MEDIAN.
        grouped[rate_index_col] = grouped[rate_index_col].dt.floor(final_interval)
        grouped = (
            grouped.groupby(id_columns + [rate_index_col], dropna=False)[ctx.concept_name]
            .max()
            .reset_index()
        )

    output_index_col = rate_index_col if rate_index_col in ['start', 'measuredat', 'charttime'] else dur_index_col
    cols = id_columns + [output_index_col, ctx.concept_name]
    avail_cols = [c for c in cols if c in grouped.columns]
    result = grouped[avail_cols].reset_index(drop=True)
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
    """Detect suspected infection (疑似感染).
    
    Supports multiple detection modes via si_mode:
    
    - "and": Both ABX and sampling required (Sepsis-3 standard, default for MIMIC-IV/AUMC)
    - "or": Either ABX or sampling
    - "abx": Only ABX required
    - "samp": Only sampling required
    - "icd_abx": ICD infection diagnosis (定人) + antibiotics (定时) - eICU新策略
    - "auto": Automatically select based on database
    
    Database-specific defaults (si_mode="auto"):
    - MIMIC-IV: "and" (ABX + 血培养, microlab coverage ~95%)
    - eICU: "icd_abx" (ICD感染诊断 + 抗生素, microlab coverage only 1.5%)
    - AUMC: "and" (ABX + 血培养, procedureorderitems coverage ~33%)
    - HiRID: "and" (default)
    
    Args:
        tables: Dictionary with component ICUTable objects:
            - 'abx': Antibiotic data (required)
            - 'samp': Body fluid sampling data (required for "and"/"or"/"samp" modes)
            - 'infection_icd': ICD infection diagnosis data (required for "icd_abx" mode)
        ctx: Callback context with kwargs:
            - si_mode: Detection mode ("and", "or", "abx", "samp", "icd_abx", "auto")
            - abx_win: Time window after ABX for sampling (default 24h)
            - samp_win: Time window after sampling for ABX (default 72h)
            - abx_min_count: Minimum antibiotic administrations required
            - positive_cultures: Whether to require positive cultures
    """
    import logging
    logger = logging.getLogger("easyicu")
    
    # Determine database name
    ds_name = ""
    if ctx is not None and getattr(ctx, "data_source", None) is not None:
        source_cfg = getattr(ctx.data_source, "config", None)
        if source_cfg is not None and hasattr(source_cfg, "name"):
            ds_name = getattr(source_cfg, "name", "") or ""
        else:
            ds_name = getattr(ctx.data_source, "name", "") or ""
    ds_name = ds_name.lower()
    
    # Get si_mode from context kwargs, default to "auto"
    si_mode = ctx.kwargs.get("si_mode", "auto") if ctx and ctx.kwargs else "auto"
    
    # Auto mode: select si_mode based on database
    if si_mode == "auto":
        # Database-specific defaults:
        # - eICU: Use "icd_abx" (ICD感染诊断定人 + 抗生素定时) due to sparse microlab (1.5%)
        # - MIMIC-IV/AUMC: Use "and" (ABX + 血培养, Sepsis-3 standard)
        if ds_name in {"eicu", "eicu_demo"}:
            si_mode = "icd_abx"
            logger.info(f"susp_inf: Using si_mode='icd_abx' for {ds_name} (ICD感染诊断 + 抗生素)")
        else:
            si_mode = "and"
            logger.debug(f"susp_inf: Using si_mode='and' for {ds_name}")

    unsupported_strict_si_dbs = {"hirid", "sic", "sicdb"}
    samp_data = getattr(tables.get("samp"), "data", None)
    infection_icd_data = getattr(tables.get("infection_icd"), "data", None)
    has_empty_sampling_support = samp_data is not None and samp_data.empty
    has_empty_icd_support = infection_icd_data is None or infection_icd_data.empty

    if (
        ds_name in unsupported_strict_si_dbs
        and si_mode in {"and", "or", "samp"}
        and has_empty_sampling_support
        and has_empty_icd_support
        and ds_name not in _SUSP_INF_UNSUPPORTED_WARNED
    ):
        logger.warning(
            "susp_inf: 数据库 '%s' 当前未提供 `samp`/`infection_icd` 感染证据，严格疑似感染定义不受支持；"
            "本次将返回空结果。若需使用抗生素代理，请显式传入 si_mode='abx'。",
            ds_name,
        )
        _SUSP_INF_UNSUPPORTED_WARNED.add(ds_name)
    
    # ===== eICU新策略: icd_abx (ICD感染诊断定人 + 抗生素定时) =====
    if si_mode == "icd_abx":
        # 需要 infection_icd 和 abx 两个概念
        if "infection_icd" not in tables or "abx" not in tables:
            raise ValueError(
                f"si_mode='icd_abx' requires 'infection_icd' and 'abx' concepts. "
                f"Available: {list(tables.keys())}"
            )
        
        # 获取感染诊断数据 (定人 - 只需要患者有感染诊断即可)
        infection_tbl = tables["infection_icd"]
        abx_tbl = tables["abx"]
        
        # 转换ID列
        id_columns, index_column, converted_tables = _assert_shared_schema(
            {"infection_icd": infection_tbl, "abx": abx_tbl},
            ctx=ctx,
            convert_ids=True
        )
        
        infection_data = converted_tables["infection_icd"].data.copy()
        abx_data = converted_tables["abx"].data.copy()
        
        if index_column is None:
            raise ValueError("susp_inf requires time-indexed component tables")
        
        # 统一时间列名
        abx_idx = converted_tables["abx"].index_column
        if abx_idx and abx_idx != index_column and abx_idx in abx_data.columns:
            abx_data = abx_data.rename(columns={abx_idx: index_column})
        
        infection_idx = converted_tables["infection_icd"].index_column
        if infection_idx and infection_idx != index_column and infection_idx in infection_data.columns:
            infection_data = infection_data.rename(columns={infection_idx: index_column})
        
        # ICD感染诊断"定人" - 获取有感染诊断的患者列表
        id_col_list = list(id_columns)
        infection_patients = infection_data[id_col_list].drop_duplicates()
        
        # 抗生素"定时" - 获取使用抗生素的时间点
        abx_events = abx_data[id_col_list + [index_column]].drop_duplicates()
        
        # 合并: 有感染诊断的患者 + 使用抗生素的时间点
        # 这意味着: 患者必须有感染诊断，抗生素使用时间即为疑似感染时间
        result = abx_events.merge(infection_patients, on=id_col_list, how="inner")
        result['susp_inf'] = True
        
        logger.info(
            f"susp_inf (icd_abx): {len(infection_patients)} patients with infection ICD, "
            f"{len(result)} suspected infection events"
        )
        
        return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column="susp_inf")
    
    # ===== 原有策略: and/or/abx/samp =====
    # 需要 abx 和 samp 两个概念
    if "abx" not in tables or "samp" not in tables:
        raise ValueError(
            f"si_mode='{si_mode}' requires 'abx' and 'samp' concepts. "
            f"Available: {list(tables.keys())}"
        )
    
    # Convert ID columns if needed (hadm_id → stay_id) before merging
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
    
    # Standardize time column names
    abx_data = abx_tbl.data.copy()
    samp_data = samp_tbl.data.copy()
    
    if abx_tbl.index_column and abx_tbl.index_column != index_column and abx_tbl.index_column in abx_data.columns:
        abx_data = abx_data.rename(columns={abx_tbl.index_column: index_column})
    if samp_tbl.index_column and samp_tbl.index_column != index_column and samp_tbl.index_column in samp_data.columns:
        samp_data = samp_data.rename(columns={samp_tbl.index_column: index_column})
    
    # Get other parameters from kwargs
    abx_win = ctx.kwargs.get("abx_win", pd.Timedelta(hours=24)) if ctx and ctx.kwargs else pd.Timedelta(hours=24)
    samp_win = ctx.kwargs.get("samp_win", pd.Timedelta(hours=72)) if ctx and ctx.kwargs else pd.Timedelta(hours=72)
    abx_min_count = ctx.kwargs.get("abx_min_count", 1) if ctx and ctx.kwargs else 1
    positive_cultures = ctx.kwargs.get("positive_cultures", False) if ctx and ctx.kwargs else False
    keep_components = ctx.kwargs.get("keep_components", False) if ctx and ctx.kwargs else False
    
    # Convert string timedelta if needed
    if isinstance(abx_win, str):
        abx_win = pd.Timedelta(abx_win)
    if isinstance(samp_win, str):
        samp_win = pd.Timedelta(samp_win)

    result = susp_inf_detector(
        abx=abx_data,
        samp=samp_data,
        id_cols=list(id_columns),
        index_col=index_column,
        si_mode=si_mode,
        abx_win=abx_win,
        samp_win=samp_win,
        abx_min_count=abx_min_count,
        positive_cultures=positive_cultures,
        keep_components=keep_components,
    )

    return _as_icutbl(result, id_columns=id_columns, index_column=index_column, value_column="susp_inf")

def _callback_gcs(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """
    Calculate GCS (Glasgow Coma Scale) with sed_impute logic.
    
    Replicates R ricu's GCS callback logic:
    - valid_win = hours(6L): Apply LOCF within a 6-hour window
    - sed_impute="max" (default): Intubated patients get GCS=15
    - sed_impute="none": Use actual measured values
    - set_na_max=True (default): Fill remaining NA with max values (egcs=4, mgcs=6, vgcs=5)
    
    Args:
        tables: Dictionary containing GCS component tables (egcs, mgcs, vgcs, tgcs, ett_gcs)
        ctx: Callback context with kwargs like sed_impute, set_na_max
    
    Returns:
        ICUTable with GCS values
    """
    # 🔧 FIX: R ricu separates ett_gcs from the main GCS components merge
    # R code: sed <- res[[cnc[5L]]]  # Store ett_gcs separately
    #         res <- reduce(merge, res[cnc[-5L]], all = TRUE)  # Merge only egcs, vgcs, mgcs, tgcs
    # This prevents ett_gcs (which is expanded hourly from mech_vent) from adding extra time points
    
    # Separate ett_gcs from other components
    ett_gcs_table = tables.pop("ett_gcs", None)
    
    # Merge only the GCS components (egcs, vgcs, mgcs, tgcs)
    data, id_columns, index_column = _merge_tables(tables, ctx=ctx, how="outer")
    ds_name = ""
    if ctx is not None and getattr(ctx, "data_source", None) is not None:
        source_cfg = getattr(ctx.data_source, "config", None)
        if source_cfg is not None and hasattr(source_cfg, "name"):
            ds_name = getattr(source_cfg, "name", "") or ""
        else:
            ds_name = getattr(ctx.data_source, "name", "") or ""
    ignore_tgcs = ds_name.lower() in {"miiv", "mimiciv"}
    if data.empty:
        cols = id_columns + ([index_column] if index_column else []) + ["gcs"]
        return _as_icutbl(pd.DataFrame(columns=cols), id_columns=id_columns, index_column=index_column, value_column="gcs")

    # Get parameters from context (matching R ricu defaults)
    sed_impute = ctx.kwargs.get("sed_impute", "max")
    set_na_max = ctx.kwargs.get("set_na_max", True)
    valid_win = ctx.kwargs.get("valid_win", 6.0)  # 6 hours, default in R ricu
    
    # CRITICAL: Apply LOCF within valid_win before processing
    # R ricu: slide(res, !!expr, before = valid_win) where expr = substitute(lapply(.SD, fun), list(fun = locf))
    gcs_components = ["egcs", "vgcs", "mgcs", "tgcs"]
    available_components = [c for c in gcs_components if c in data.columns]
    if available_components and index_column:
        data = _apply_locf_window(
            data=data,
            id_columns=id_columns,
            index_column=index_column,
            value_columns=available_components,
            window_hours=valid_win,
        )

    tgcs = None if ignore_tgcs else pd.to_numeric(data.get("tgcs"), errors="coerce")
    egcs = pd.to_numeric(data.get("egcs"), errors="coerce")
    mgcs = pd.to_numeric(data.get("mgcs"), errors="coerce")
    vgcs = pd.to_numeric(data.get("vgcs"), errors="coerce")
    
    # Ensure all GCS components are Series (pd.to_numeric may return scalar for single-row data)
    # Use repeat() to broadcast scalar to match data.index length
    if tgcs is not None and not isinstance(tgcs, pd.Series):
        tgcs = pd.Series(np.repeat(tgcs, len(data.index)), index=data.index, dtype=float)
    if egcs is not None and not isinstance(egcs, pd.Series):
        egcs = pd.Series(np.repeat(egcs, len(data.index)), index=data.index, dtype=float)
    if mgcs is not None and not isinstance(mgcs, pd.Series):
        mgcs = pd.Series(np.repeat(mgcs, len(data.index)), index=data.index, dtype=float)
    if vgcs is not None and not isinstance(vgcs, pd.Series):
        vgcs = pd.Series(np.repeat(vgcs, len(data.index)), index=data.index, dtype=float)
    
    # 🔧 FIX: Get ett_gcs from the separated table, not from merged data
    # R ricu: sed <- res[[cnc[5L]]] - ett_gcs is kept separate
    def _time_to_hours(frame: pd.DataFrame, column: str) -> pd.Series:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds() / 3600.0
        if pd.api.types.is_datetime64_any_dtype(series):
            if id_columns:
                return frame.groupby(list(id_columns), dropna=False)[column].transform(
                    lambda x: (pd.to_datetime(x, errors="coerce") - pd.to_datetime(x, errors="coerce").min()).dt.total_seconds() / 3600.0
                )
            dt = pd.to_datetime(series, errors="coerce")
            return (dt - dt.min()).dt.total_seconds() / 3600.0
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().any():
            return coerced
        dt = pd.to_datetime(series, errors="coerce")
        if id_columns:
            return frame.groupby(list(id_columns), dropna=False)[column].transform(
                lambda x: (pd.to_datetime(x, errors="coerce") - pd.to_datetime(x, errors="coerce").min()).dt.total_seconds() / 3600.0
            )
        return (dt - dt.min()).dt.total_seconds() / 3600.0

    def _window_match_indicator(base_df: pd.DataFrame, window_df: pd.DataFrame, *, time_col: str, window_time_col: str, dur_col: str) -> pd.Series:
        matched = pd.Series(False, index=base_df.index, dtype=bool)
        base_hours = _time_to_hours(base_df, time_col)
        window_hours = _time_to_hours(window_df, window_time_col)
        dur_hours = pd.to_numeric(window_df[dur_col], errors="coerce")

        if id_columns:
            grouped = base_df.groupby(list(id_columns), dropna=False, sort=False).groups
            # ⚡ Pre-group window_df by patient — O(W) once, instead of O(P × W)
            win_grouped = window_df.groupby(list(id_columns), dropna=False, sort=False).groups
            for key, idx in grouped.items():
                win_idx = win_grouped.get(key)
                if win_idx is None or len(win_idx) == 0:
                    continue
                starts = window_hours.loc[win_idx].to_numpy(dtype=float)
                durs = dur_hours.loc[win_idx].to_numpy(dtype=float)
                times = base_hours.loc[idx].to_numpy(dtype=float)
                # Vectorized interval coverage — avoid per-window Python loop
                valid = ~(np.isnan(starts) | np.isnan(durs))
                if not valid.any():
                    continue
                fs = np.floor(starts[valid])
                fe = np.floor(fs + durs[valid])
                # Broadcasting: times[i] vs all windows — O(T × W_patient)
                if len(fs) <= 50:
                    # Small number of windows: vectorized broadcasting
                    coverage = np.zeros(len(times), dtype=bool)
                    for s, e in zip(fs, fe):
                        coverage |= (times >= s) & (times <= e)
                else:
                    # Many windows: sort and scan
                    coverage = np.zeros(len(times), dtype=bool)
                    order = np.argsort(fs)
                    fs_sorted, fe_sorted = fs[order], fe[order]
                    for s, e in zip(fs_sorted, fe_sorted):
                        coverage |= (times >= s) & (times <= e)
                matched.loc[idx] = coverage
            return matched

        starts = window_hours.to_numpy(dtype=float)
        durs = dur_hours.to_numpy(dtype=float)
        times = base_hours.to_numpy(dtype=float)
        coverage = np.zeros(len(times), dtype=bool)
        for start, dur in zip(starts, durs):
            if np.isnan(start) or np.isnan(dur):
                continue
            fs = np.floor(start)
            fe = np.floor(fs + dur)
            coverage |= (times >= fs) & (times <= fe)
        return pd.Series(coverage, index=base_df.index, dtype=bool)

    ett_gcs = None
    if ett_gcs_table is not None:
        if hasattr(ett_gcs_table, 'data'):
            ett_df = ett_gcs_table.data
        else:
            ett_df = ett_gcs_table
        if 'ett_gcs' in ett_df.columns and not ett_df.empty:
            merge_cols = list(id_columns) + ([index_column] if index_column else [])
            ett_time_col = _get_index_column(ett_gcs_table) if hasattr(ett_gcs_table, 'data') else None
            if not ett_time_col or ett_time_col not in ett_df.columns:
                # Detect actual time column in ett_df
                for cand in [index_column, 'datetime', 'charttime', 'starttime', 'time']:
                    if cand in ett_df.columns:
                        ett_time_col = cand
                        break
            dur_col = getattr(ett_gcs_table, 'dur_var', None) if hasattr(ett_gcs_table, 'dur_var') else None
            
            # 🔧 FIX: Handle time column name mismatch between ett_gcs and GCS data
            if ett_time_col and ett_time_col != index_column and ett_time_col in ett_df.columns:
                ett_df = ett_df.rename(columns={ett_time_col: index_column})
                ett_time_col = index_column
            
            if dur_col and ett_time_col and dur_col in ett_df.columns and ett_time_col in ett_df.columns and index_column in data.columns:
                ett_true = ett_df[ett_df['ett_gcs'].fillna(False)].copy()
                if not ett_true.empty:
                    ett_gcs = _window_match_indicator(
                        data,
                        ett_true,
                        time_col=index_column,
                        window_time_col=ett_time_col,
                        dur_col=dur_col,
                    )
                    data['ett_gcs'] = ett_gcs
            elif all(c in ett_df.columns for c in merge_cols):
                ett_subset = ett_df[merge_cols + ['ett_gcs']].copy()
                # R ricu: sed <- sed[is_true(get(cnc[5L])), ] - only keep TRUE rows
                # Then inner join with data to find intubated time points
                ett_true = ett_subset[ett_subset['ett_gcs'].fillna(False)]
                if not ett_true.empty:
                    # Mark which rows in data are intubated
                    data = data.merge(ett_true[merge_cols + ['ett_gcs']], on=merge_cols, how='left')
                    ett_gcs = data.get("ett_gcs")

    # CRITICAL FIX: Replicate R ricu's sed_impute logic
    # If sed_impute="max" (default) and patient is intubated, set tgcs=15
    # 🔧 FIX 2025-02: For MIIV/MIMICIV, ett_gcs pipeline (mech_vent→vent_ind) is
    # unreliable — use vgcs==1 ("No verbal response" = likely intubated) as proxy.
    # This matches R's sed_impute behavior ~98% of the time for these databases.
    if sed_impute == "max":
        is_intubated = None
        if ignore_tgcs:
            # MIIV/MIMICIV: prefer actual ett_gcs data when available (it correctly
            # distinguishes intubated vgcs=1 from non-intubated vgcs=1).
            # Only fall back to vgcs==1 proxy if ett_gcs is unavailable.
            if ett_gcs is not None and isinstance(ett_gcs, pd.Series) and ett_gcs.any():
                is_intubated = ett_gcs.where(ett_gcs.notna(), False).astype(bool)
            elif vgcs is not None and isinstance(vgcs, pd.Series) and len(vgcs) > 0:
                is_intubated = (vgcs == 1.0)
        elif ett_gcs is not None:
            is_intubated = ett_gcs.where(ett_gcs.notna(), False).astype(bool)
        
        if is_intubated is not None and is_intubated.any():
            if tgcs is None:
                tgcs = pd.Series(np.nan, index=data.index, dtype=float)
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
    # R ricu unconditionally fills all NA components with max values
    if set_na_max:
        if egcs is not None:
            egcs = egcs.fillna(4.0)
        if mgcs is not None:
            mgcs = mgcs.fillna(6.0)
        if vgcs is not None:
            vgcs = vgcs.fillna(5.0)

    # Calculate GCS: use tgcs if available AND valid (>=3), otherwise sum components
    combined = pd.Series(index=data.index, dtype=float)
    
    if tgcs is not None:
        valid_tgcs = tgcs.where((tgcs >= 3) | tgcs.isna())
        combined = valid_tgcs.copy()
    
    # For rows where tgcs is NA or invalid (<3), calculate from components
    if egcs is not None and mgcs is not None and vgcs is not None:
        component_sum = egcs.add(mgcs, fill_value=np.nan).add(vgcs, fill_value=np.nan)
        combined = combined.fillna(component_sum)
    
    # Final NA fill with max GCS=15
    if set_na_max:
        combined = combined.fillna(15.0)

    # Use ctx.concept_name to support both 'gcs' and 'tgcs' concepts
    output_col = ctx.concept_name if ctx is not None else "gcs"
    data[output_col] = combined
    cols = id_columns + ([index_column] if index_column else []) + [output_col]
    frame = data[cols].dropna(subset=[output_col])
    
    # Remove duplicate timestamps (outer merge may create duplicates)
    # Keep first occurrence for each (admissionid, measuredat) pair
    dedup_cols = list(id_columns) + ([index_column] if index_column else [])
    frame = frame.drop_duplicates(subset=dedup_cols, keep='first')
    
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column=output_col)

from .callbacks import uo_6h as calc_uo_6h, uo_12h as calc_uo_12h, uo_24h as calc_uo_24h, uo_all_windows

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
    # 注意：uo_6h/12h/24h 是计算概念，依赖 urine 和 weight
    # 为了避免递归调用 load_concepts 导致重复加载 urine/weight，我们手动处理这些依赖
    
    direct_deps = ["crea", "potassium", "ph", "bicarb", "rrt"]
    uo_deps = ["uo_6h", "uo_12h", "uo_24h"]
    
    # 检查缺失的直接依赖
    missing_direct = [c for c in direct_deps if c not in tables]
    
    # 检查缺失的UO依赖
    missing_uo = [c for c in uo_deps if c not in tables]
    
    # 如果有缺失的UO依赖，我们需要 urine 和 weight
    if missing_uo:
        if "urine" not in tables:
            missing_direct.append("urine")
        if "weight" not in tables:
            missing_direct.append("weight")
    
    if missing_direct:
        # ⚡ 批量加载所有缺失的基础概念
        try:
            loaded = ctx.resolver.load_concepts(
                missing_direct,
                ctx.data_source,
                merge=False,
                aggregate=None,
                patient_ids=ctx.patient_ids,
                interval=ctx.interval,
            )
            # 将加载的概念添加到tables
            if isinstance(loaded, dict):
                tables.update(loaded)
            elif isinstance(loaded, ICUTable) and len(missing_direct) == 1:
                tables[missing_direct[0]] = loaded
        except (KeyError, ValueError) as e:
            if os.environ.get('DEBUG'):
                print(f"   ⚠️  无法加载部分RRT依赖概念: {e}")
    
    # 手动计算缺失的 UO 概念，避免递归调用 load_concepts
    if missing_uo and "urine" in tables and "weight" in tables:
        urine_tbl = tables["urine"]
        weight_tbl = tables["weight"]
        
        # 确保数据不为空
        if not urine_tbl.data.empty and not weight_tbl.data.empty:
            # 提取DataFrame并确保列名正确
            urine_df = urine_tbl.data.copy()
            weight_df = weight_tbl.data.copy()
            
            # 确保urine列名为'urine'
            urine_val_col = urine_tbl.value_column or "urine"
            if urine_val_col != "urine" and urine_val_col in urine_df.columns:
                urine_df = urine_df.rename(columns={urine_val_col: "urine"})
            elif "urine" not in urine_df.columns:
                # 尝试找到值列
                cols = [c for c in urine_df.columns if c not in urine_tbl.id_columns and c != urine_tbl.index_column]
                if cols:
                    urine_df = urine_df.rename(columns={cols[0]: "urine"})
            
            # 确保weight列名为'weight'
            weight_val_col = weight_tbl.value_column or "weight"
            if weight_val_col != "weight" and weight_val_col in weight_df.columns:
                weight_df = weight_df.rename(columns={weight_val_col: "weight"})
            elif "weight" not in weight_df.columns:
                # 尝试找到值列
                cols = [c for c in weight_df.columns if c not in weight_tbl.id_columns and c != weight_tbl.index_column]
                if cols:
                    weight_df = weight_df.rename(columns={cols[0]: "weight"})
            
            # ⚡ PERF: Compute all 3 UO windows in a single merge+sort+groupby
            uo_results = uo_all_windows(urine_df, weight_df, interval=ctx.interval)
            for uo_name in missing_uo:
                if uo_name in uo_results:
                    tables[uo_name] = _as_icutbl(uo_results[uo_name], id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column=uo_name)
    
    # 🔧 FIX: 如果所有依赖都加载失败，返回空表而不是报错
    if not tables:
        # 使用数据库特定的默认 ID 列
        default_id_col = 'stay_id'
        if ctx and hasattr(ctx, 'data_source') and ctx.data_source:
            db_name = getattr(ctx.data_source.config, 'name', '')
            if db_name == 'eicu':
                default_id_col = 'patientunitstayid'
            elif db_name == 'aumc':
                default_id_col = 'admissionid'
            elif db_name == 'hirid':
                default_id_col = 'patientid'
            elif db_name in ['sic', 'sicdb']:
                default_id_col = 'CaseID'
            elif db_name == 'mimic':
                default_id_col = 'icustay_id'
        return _as_icutbl(
            pd.DataFrame(columns=[default_id_col, 'charttime', 'rrt_criteria']),
            id_columns=[default_id_col],
            index_column='charttime',
            value_column='rrt_criteria'
        )
    
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
        if pd.api.types.is_bool_dtype(rrt_series) or str(rrt_series.dtype) == 'boolean':
            # Boolean type - fill NA with False
            rrt_active = rrt_series.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(rrt_series):
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
    """Compute weight-normalized urine output rate (mL/kg/h) over a 1-hour window.

    Historical name kept for backward compatibility; the SOFA-2 pipeline
    prefers ``uo_6h`` / ``uo_12h`` / ``uo_24h``. Previously this callback
    returned NaN, silently dropping a real feature; it now delegates to
    the same windowed-average helper used by ``uo_6h`` so callers that
    still resolve ``urine_mlkgph`` receive a real value.
    """
    result = _callback_uo_window(tables, ctx, window_hours=1, output_col="uo_1h")
    df = result.data.rename(columns={"uo_1h": "urine_mlkgph"})
    return _as_icutbl(
        df.reset_index(drop=True),
        id_columns=list(result.id_columns),
        index_column=result.index_column,
        value_column="urine_mlkgph",
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
    
    # Load required concepts - 🚀 优化：使用 get_raw_concept 缓存
    required = ["urine", "weight"]
    missing = [c for c in required if c not in tables]
    
    if missing:
        # 优先从缓存获取
        for concept in missing[:]:  # 使用副本迭代
            if hasattr(ctx.resolver, 'get_raw_concept'):
                cached = ctx.resolver.get_raw_concept(concept, ctx.data_source, ctx.patient_ids)
                if cached is not None:
                    tables[concept] = cached
                    missing.remove(concept)
        
        # 剩余的批量加载
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
    """Sum multiple component tables together (e.g., for GCS total = eye + motor + verbal).
    
    R ricu's sum_components() is a plain sum of sub-concepts. NaN propagates:
    if ANY sub-component is NaN at a given time point, the result is NaN.
    The set_na_max logic (filling missing GCS components with max values)
    belongs exclusively in the gcs() callback, NOT here.
    """
    
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
    
    # Plain sum with NaN propagation (matches R ricu's sum_components behavior).
    # If any component is NaN, the result is NaN for that row.
    available_cols = [c for c in component_cols if c in data.columns]
    if available_cols:
        data[output_col] = data[available_cols].apply(
            lambda row: row.sum() if row.notna().all() else np.nan, axis=1
        )
    else:
        data[output_col] = np.nan
    
    # Keep only rows where we have at least some data
    mask = pd.Series(False, index=data.index)
    for col in component_cols:
        if col in data.columns:
            mask |= data[col].notna()
    
    data = data[mask]

    cols = id_columns + ([index_column] if index_column else []) + [output_col]
    frame = data[cols].dropna(subset=[output_col])
    return _as_icutbl(frame.reset_index(drop=True), id_columns=id_columns, index_column=index_column, value_column=output_col)

def _callback_miiv_icu_patients_filter(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Filter MIMIC-IV cohorts so demographics align with ICU stay IDs."""

    from ..datasource import ICUDataSource

    database = ctx.database if ctx.database else "miiv"
    if database != "miiv":
        return next(iter(tables.values()))

    try:
        ds = ICUDataSource.get_instance(database)
        icustays_df = ds.load_table("icustays", columns=["stay_id", "subject_id"])
        if icustays_df.empty:
            main_table = next(iter(tables.values()))
            from ..table import IdTbl

            cols = ["stay_id"] + [col for col in main_table.columns if col != "subject_id"]
            return IdTbl(pd.DataFrame(columns=cols), id_vars=["stay_id"])

        main_table = next(iter(tables.values()))
        data_df = main_table.to_pandas()
        if "subject_id" in data_df.columns and "subject_id" in icustays_df.columns:
            merged = icustays_df.merge(
                data_df.astype({ "subject_id": icustays_df["subject_id"].dtype }),
                on="subject_id",
                how="inner",
            )
            merged = merged.drop(columns=["subject_id"], errors="ignore")
            merged = merged.set_index("stay_id")
            from ..table import IdTbl

            return IdTbl(merged, id_vars=["stay_id"])
        return main_table
    except Exception:
        return next(iter(tables.values()))

def _callback_driving_pressure(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate driving pressure (Plateau Pressure - PEEP).
    
    Driving pressure is a key ventilator parameter associated with mortality
    in ARDS patients. It represents the pressure applied to expand the lungs
    beyond PEEP.
    
    Args:
        tables: Dictionary containing 'plateau_pres' and 'peep' tables
        ctx: Callback context with database and other information
        
    Returns:
        ICUTable with driving_pres column (cmH2O)
        
    References:
        Amato et al., NEJM 2015 - Driving Pressure and Survival in ARDS
    """
    from easyicu.callbacks import driving_pressure
    import logging
    logger = logging.getLogger(__name__)
    
    # Get the input tables
    plateau_tbl = tables.get('plateau_pres')
    peep_tbl = tables.get('peep')
    
    if plateau_tbl is None or peep_tbl is None:
        # Return empty table
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'driving_pres']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='driving_pres'
        )
    
    # Convert to pandas DataFrame
    if hasattr(plateau_tbl, 'to_dataframe'):
        plateau_df = plateau_tbl.to_dataframe()
    elif hasattr(plateau_tbl, 'data'):
        plateau_df = plateau_tbl.data
    else:
        plateau_df = plateau_tbl
    
    if hasattr(peep_tbl, 'to_dataframe'):
        peep_df = peep_tbl.to_dataframe()
    elif hasattr(peep_tbl, 'data'):
        peep_df = peep_tbl.data
    else:
        peep_df = peep_tbl
    
    # Ensure DataFrames (not ICUTable)
    if hasattr(plateau_df, 'data'):
        plateau_df = plateau_df.data
    if hasattr(peep_df, 'data'):
        peep_df = peep_df.data
    
    # Debug: check columns
    logger.debug(f"driving_pres callback: plateau_df columns={plateau_df.columns.tolist()}, shape={plateau_df.shape}")
    logger.debug(f"driving_pres callback: peep_df columns={peep_df.columns.tolist()}, shape={peep_df.shape}")
    
    if plateau_df.empty or peep_df.empty:
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'driving_pres']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='driving_pres'
        )
    
    # Get database name
    database = None
    if hasattr(ctx, 'data_source') and ctx.data_source and hasattr(ctx.data_source, 'config'):
        database = getattr(ctx.data_source.config, 'name', None)
    
    # Call the driving_pressure function
    result = driving_pressure(
        plateau_pres=plateau_df,
        peep=peep_df,
        match_win=pd.Timedelta(hours=1),
        database=database
    )
    
    if result.empty:
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'driving_pres']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='driving_pres'
        )
    
    # Detect ID and time columns
    id_col = 'stay_id'
    for col in ['stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']:
        if col in result.columns:
            id_col = col
            break
    
    time_col = 'charttime'
    for col in ['charttime', 'measuredat', 'measuredat_minutes', 'observationoffset', 'datetime', 'registeredat']:
        if col in result.columns:
            time_col = col
            break
    
    return ICUTable(
        data=result,
        id_columns=[id_col],
        index_column=time_col,
        value_column='driving_pres'
    )


def _callback_kdigo_aki(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate KDIGO AKI staging based on creatinine and urine output.
    
    KDIGO stages:
    - Stage 0: No AKI
    - Stage 1: Creatinine >=0.3 mg/dL increase in 48h OR >=1.5x baseline, or UO <0.5 mL/kg/h for 6-12h
    - Stage 2: Creatinine >=2x baseline, or UO <0.5 mL/kg/h for >=12h  
    - Stage 3: Creatinine >=3x baseline OR >=4.0 mg/dL OR RRT, or UO <0.3 mL/kg/h for >=24h or anuria
    
    Args:
        tables: Dictionary containing 'crea', 'urine', 'weight', and optionally 'rrt' tables
        ctx: Callback context
        
    Returns:
        ICUTable with aki_stage column (0-3)
    """
    from easyicu.kdigo_aki import kdigo_stages, _detect_id_col, _detect_time_col
    
    # Extract DataFrames from tables
    crea_tbl = tables.get('crea')
    urine_tbl = tables.get('urine')
    weight_tbl = tables.get('weight')
    rrt_tbl = tables.get('rrt')
    
    # Convert to DataFrames
    def to_df(tbl):
        if tbl is None:
            return None
        if hasattr(tbl, 'to_dataframe'):
            return tbl.to_dataframe()
        if hasattr(tbl, 'data'):
            return tbl.data
        return tbl
    
    crea_df = to_df(crea_tbl)
    urine_df = to_df(urine_tbl)
    weight_df = to_df(weight_tbl)
    rrt_df = to_df(rrt_tbl)
    
    if crea_df is None or crea_df.empty:
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'aki_stage', 'aki']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='aki_stage'
        )
    
    # Detect ID and time columns using the same helpers as kdigo_aki.py
    id_col = _detect_id_col(crea_df) or 'stay_id'
    time_col = _detect_time_col(crea_df) or 'charttime'
    
    # Calculate KDIGO stages
    result = kdigo_stages(
        crea_df=crea_df,
        urine_df=urine_df,
        weight_df=weight_df,
        rrt_df=rrt_df,
        id_col=id_col,
        time_col=time_col,
    )
    
    if result.empty:
        return ICUTable(
            data=pd.DataFrame(columns=[id_col, time_col, 'aki_stage', 'aki']),
            id_columns=[id_col],
            index_column=time_col,
            value_column='aki_stage'
        )
    
    return ICUTable(
        data=result,
        id_columns=[id_col],
        index_column=time_col,
        value_column='aki_stage'
    )


def _callback_kdigo_creatinine(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate KDIGO AKI creatinine-based staging.
    
    Args:
        tables: Dictionary containing 'crea' table
        ctx: Callback context
        
    Returns:
        ICUTable with aki_stage_creat column (0-3)
    """
    from easyicu.kdigo_aki import kdigo_creatinine
    
    crea_tbl = tables.get('crea')
    if crea_tbl is None:
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'aki_stage_creat']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='aki_stage_creat'
        )
    
    if hasattr(crea_tbl, 'to_dataframe'):
        crea_df = crea_tbl.to_dataframe()
    elif hasattr(crea_tbl, 'data'):
        crea_df = crea_tbl.data
    else:
        crea_df = crea_tbl
    
    if crea_df.empty:
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'aki_stage_creat']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='aki_stage_creat'
        )
    
    # Detect ID and time columns
    id_col = 'stay_id'
    for col in ['stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']:
        if col in crea_df.columns:
            id_col = col
            break
    
    time_col = 'charttime'
    for col in ['charttime', 'measuredat', 'measuredat_minutes', 'observationoffset', 'datetime', 'registeredat', 'Offset', 'offset']:
        if col in crea_df.columns:
            time_col = col
            break
    
    result = kdigo_creatinine(crea_df, id_col=id_col, time_col=time_col)
    
    return ICUTable(
        data=result,
        id_columns=[id_col],
        index_column=time_col,
        value_column='aki_stage_creat'
    )


def _callback_kdigo_uo(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
) -> ICUTable:
    """Calculate KDIGO AKI urine output-based staging.
    
    Args:
        tables: Dictionary containing 'urine' and 'weight' tables
        ctx: Callback context
        
    Returns:
        ICUTable with aki_stage_uo column (0-3)
    """
    from easyicu.kdigo_aki import kdigo_uo
    
    urine_tbl = tables.get('urine')
    weight_tbl = tables.get('weight')
    
    if urine_tbl is None:
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'aki_stage_uo']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='aki_stage_uo'
        )
    
    def to_df(tbl):
        if tbl is None:
            return None
        if hasattr(tbl, 'to_dataframe'):
            return tbl.to_dataframe()
        if hasattr(tbl, 'data'):
            return tbl.data
        return tbl
    
    urine_df = to_df(urine_tbl)
    weight_df = to_df(weight_tbl)
    
    if urine_df is None or urine_df.empty:
        return ICUTable(
            data=pd.DataFrame(columns=['stay_id', 'charttime', 'aki_stage_uo']),
            id_columns=['stay_id'],
            index_column='charttime',
            value_column='aki_stage_uo'
        )
    
    # Detect ID and time columns
    id_col = 'stay_id'
    for col in ['stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']:
        if col in urine_df.columns:
            id_col = col
            break
    
    time_col = 'charttime'
    for col in ['charttime', 'measuredat', 'measuredat_minutes', 'observationoffset', 'datetime', 'registeredat', 'intakeoutputoffset', 'intakeoutputentryoffset', 'Offset', 'offset']:
        if col in urine_df.columns:
            time_col = col
            break
    
    result = kdigo_uo(urine_df, weight_df, id_col=id_col, time_col=time_col)
    
    return ICUTable(
        data=result,
        id_columns=[id_col],
        index_column=time_col,
        value_column='aki_stage_uo'
    )


def _callback_simple_passthrough(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext,
    concept_name: str,
) -> ICUTable:
    """Simple passthrough callback for concepts that just need to load from source.
    
    Some concepts like tco2 and ca have a concept-level callback defined in the
    dictionary but don't require any special aggregation - they just load directly
    from the source table. This callback handles those cases.
    
    Args:
        tables: Dictionary of loaded tables (should be empty when callback is called)
        ctx: Callback context
        concept_name: Name of the concept to load
        
    Returns:
        The loaded concept data
    """
    # Load the concept normally through the resolver
    from .concept import ConceptResolver
    
    resolver = ctx.resolver if ctx.resolver else ConceptResolver()
    
    # Load the concept without the callback to avoid infinite recursion
    # merge=False returns a dict of {concept_name: ICUTable}
    result_dict = resolver.load_concepts(
        [concept_name],
        data_source=ctx.data_source,
        interval=ctx.interval,
        patient_ids=ctx.patient_ids,
        merge=False,  # Don't merge, return dict
        _bypass_callback=True,  # Special flag to skip callback
    )
    
    # Return the ICUTable for this concept
    return result_dict[concept_name]


def _callback_fluid_balance_admitted(
    tables: Dict[str, "ICUTable"],
    ctx: "ConceptCallbackContext",
) -> "ICUTable":
    """Compute cumulative fluid balance = cumulative_input_mL - cumulative_urine_mL.

    Phase A implementation (approved defaults):
      - Input: all inputevents rows where amountuom == 'mL', summed per hour.
      - Output: urine concept (already loaded as dependency).
      - Result: cumulative(input) - cumulative(urine) per stay.

    This callback directly loads inputevents from the data source because the
    standard concept-dict pattern doesn't support unit-based row filtering.
    """
    import numpy as np

    # Get urine data from dependencies
    urine_tbl = tables.get("urine")
    if urine_tbl is None or urine_tbl.data.empty:
        # Return empty with correct schema
        empty_cols = ["stay_id", "charttime", "fluid_balance_admitted"]
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="fluid_balance_admitted",
        )

    urine_df = urine_tbl.data.copy()
    id_col = urine_tbl.id_columns[0] if urine_tbl.id_columns else "stay_id"
    time_col = urine_tbl.index_column or "charttime"
    urine_val_col = "urine" if "urine" in urine_df.columns else urine_tbl.value_column

    # Load total input (mL-valued rows) directly from data source
    data_source = ctx.data_source
    patient_ids = ctx.patient_ids

    try:
        # Determine database type
        db_name = ""
        if hasattr(data_source, "config") and hasattr(data_source.config, "name"):
            db_name = data_source.config.name.lower()

        # Choose table and unit filter based on database
        if db_name in ("miiv", "mimic"):
            table_name = "inputevents" if db_name == "miiv" else "inputevents_mv"
            unit_col = "amountuom"
            unit_value = "mL" if db_name == "miiv" else "ml"
            amount_col = "amount"
        else:
            # Other databases not yet supported for fluid balance
            empty_cols = [id_col, time_col, "fluid_balance_admitted"]
            return _as_icutbl(
                pd.DataFrame(columns=empty_cols),
                id_columns=[id_col],
                index_column=time_col,
                value_column="fluid_balance_admitted",
            )

        # Load inputevents with amount and unit columns
        from .datasource import FilterSpec, FilterOp

        # Get patient IDs from urine data
        stay_ids = urine_df[id_col].unique().tolist()

        input_df = data_source.load_table(
            table_name,
            columns=[id_col, "starttime", amount_col, unit_col],
            filters=[
                FilterSpec(column=id_col, op=FilterOp.IN, value=stay_ids),
            ],
        )

        # Extract DataFrame from ICUTable if needed
        if hasattr(input_df, "data"):
            input_df = input_df.data

        if input_df is None or input_df.empty:
            empty_cols = [id_col, time_col, "fluid_balance_admitted"]
            return _as_icutbl(
                pd.DataFrame(columns=empty_cols),
                id_columns=[id_col],
                index_column=time_col,
                value_column="fluid_balance_admitted",
            )

        # Filter to mL-valued rows only (approved default: stance A)
        input_df = input_df[input_df[unit_col].astype(str).str.strip().str.lower() == unit_value.lower()].copy()
        input_df[amount_col] = pd.to_numeric(input_df[amount_col], errors="coerce")
        input_df = input_df.dropna(subset=[amount_col])

        if input_df.empty:
            empty_cols = [id_col, time_col, "fluid_balance_admitted"]
            return _as_icutbl(
                pd.DataFrame(columns=empty_cols),
                id_columns=[id_col],
                index_column=time_col,
                value_column="fluid_balance_admitted",
            )

        # Convert starttime to hours (same as urine's time column)
        time_input_col = "starttime"
        if pd.api.types.is_numeric_dtype(input_df[time_input_col]):
            input_df["_hour"] = np.floor(input_df[time_input_col]).astype(int)
        else:
            # datetime → relative hours would need intime; for now use numeric
            input_df["_hour"] = pd.to_numeric(input_df[time_input_col], errors="coerce")
            input_df["_hour"] = np.floor(input_df["_hour"]).astype(int)

        # Sum input per (stay, hour)
        input_hourly = input_df.groupby([id_col, "_hour"], as_index=False).agg(
            {amount_col: "sum"}
        ).rename(columns={"_hour": time_col, amount_col: "input_ml"})

        # Prepare urine hourly
        urine_df[urine_val_col] = pd.to_numeric(urine_df[urine_val_col], errors="coerce").fillna(0)
        urine_df[time_col] = pd.to_numeric(urine_df[time_col], errors="coerce")
        urine_df["_hour"] = np.floor(urine_df[time_col]).astype(int)
        urine_hourly = urine_df.groupby([id_col, "_hour"], as_index=False).agg(
            {urine_val_col: "sum"}
        ).rename(columns={"_hour": time_col, urine_val_col: "urine_ml"})

        # Merge input and urine on (stay, hour) — outer join
        merged = pd.merge(input_hourly, urine_hourly, on=[id_col, time_col], how="outer")
        merged["input_ml"] = merged["input_ml"].fillna(0)
        merged["urine_ml"] = merged["urine_ml"].fillna(0)
        merged = merged.sort_values([id_col, time_col])

        # Compute cumulative balance per stay
        merged["cum_input"] = merged.groupby(id_col)["input_ml"].cumsum()
        merged["cum_urine"] = merged.groupby(id_col)["urine_ml"].cumsum()
        merged["fluid_balance_admitted"] = merged["cum_input"] - merged["cum_urine"]

        result = merged[[id_col, time_col, "fluid_balance_admitted"]].copy()

        return _as_icutbl(
            result,
            id_columns=[id_col],
            index_column=time_col,
            value_column="fluid_balance_admitted",
        )

    except Exception as e:
        logger.warning(f"fluid_balance_admitted callback failed: {e}")
        empty_cols = [id_col, time_col, "fluid_balance_admitted"]
        return _as_icutbl(
            pd.DataFrame(columns=empty_cols),
            id_columns=[id_col],
            index_column=time_col,
            value_column="fluid_balance_admitted",
        )


def _callback_fluid_balance_hourly(
    tables: Dict[str, "ICUTable"],
    ctx: "ConceptCallbackContext",
) -> "ICUTable":
    """Compute hourly fluid balance = total_input_ml - urine.

    Both dependencies go through the standard loading pipeline, so their time
    columns are already aligned to relative hours. This callback simply merges
    them on (id, time) and subtracts.
    """
    input_tbl = tables.get("total_input_ml")
    urine_tbl = tables.get("urine")

    # Determine schema from whichever table is available
    ref_tbl = input_tbl or urine_tbl
    if ref_tbl is None:
        return _as_icutbl(
            pd.DataFrame(columns=["stay_id", "charttime", "fluid_balance"]),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="fluid_balance",
        )

    id_col = ref_tbl.id_columns[0] if ref_tbl.id_columns else "stay_id"
    declared_time_col = ref_tbl.index_column or "charttime"

    # Detect actual time column in the DataFrame (may differ from declared)
    def detect_time_col(df, declared):
        if declared in df.columns:
            return declared
        # Common time-column candidates after pipeline normalization
        for candidate in ("charttime", "starttime", "start", "time", "measuredat",
                          "givenat", "Offset", "datetime"):
            if candidate in df.columns:
                return candidate
        return declared

    # Extract DataFrames
    def to_df(tbl, val_name):
        if tbl is None or tbl.data.empty:
            return pd.DataFrame(columns=[id_col, "charttime", val_name])
        df = tbl.data.copy()
        # Detect time column from this specific DataFrame
        if "charttime" in df.columns:
            pass  # already canonical
        else:
            # Find first available time-like column and rename
            for candidate in (declared_time_col, "starttime", "start", "time",
                              "measuredat", "givenat", "Offset", "datetime"):
                if candidate and candidate in df.columns:
                    df = df.rename(columns={candidate: "charttime"})
                    break
        # Rename value column
        vc = tbl.value_column or val_name
        if vc in df.columns and vc != val_name:
            df = df.rename(columns={vc: val_name})
        elif val_name not in df.columns:
            # Try to find the value column
            non_meta = [c for c in df.columns if c not in [id_col, "charttime"]]
            if non_meta:
                df = df.rename(columns={non_meta[0]: val_name})
        # Defensive: if still missing required columns, return empty
        if "charttime" not in df.columns or val_name not in df.columns:
            return pd.DataFrame(columns=[id_col, "charttime", val_name])
        return df[[id_col, "charttime", val_name]].copy()

    input_df = to_df(input_tbl, "input_ml")
    urine_df = to_df(urine_tbl, "urine_ml")
    time_col = "charttime"  # canonical post-rename

    # Ensure numeric
    input_df["input_ml"] = pd.to_numeric(input_df["input_ml"], errors="coerce").fillna(0)
    urine_df["urine_ml"] = pd.to_numeric(urine_df["urine_ml"], errors="coerce").fillna(0)

    # Merge on (id, time) — outer join
    merged = pd.merge(input_df, urine_df, on=[id_col, time_col], how="outer")
    merged["input_ml"] = merged["input_ml"].fillna(0)
    merged["urine_ml"] = merged["urine_ml"].fillna(0)

    # Hourly balance = input - output
    merged["fluid_balance"] = merged["input_ml"] - merged["urine_ml"]
    merged = merged.sort_values([id_col, time_col])

    result = merged[[id_col, time_col, "fluid_balance"]].copy()

    return _as_icutbl(
        result,
        id_columns=[id_col],
        index_column=time_col,
        value_column="fluid_balance",
    )


def _callback_fluid_balance_cumulative(
    tables: Dict[str, "ICUTable"],
    ctx: "ConceptCallbackContext",
) -> "ICUTable":
    """Cumulative fluid balance = running sum of hourly fluid_balance per stay."""
    fb_tbl = tables.get("fluid_balance")
    if fb_tbl is None or fb_tbl.data.empty:
        return _as_icutbl(
            pd.DataFrame(columns=["stay_id", "charttime", "fluid_balance_cumulative"]),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="fluid_balance_cumulative",
        )

    df = fb_tbl.data.copy()
    id_col = fb_tbl.id_columns[0] if fb_tbl.id_columns else "stay_id"
    declared_time_col = fb_tbl.index_column or "charttime"
    val_col = fb_tbl.value_column or "fluid_balance"

    # Detect time col from actual DataFrame
    if declared_time_col not in df.columns:
        for candidate in ("charttime", "starttime", "start", "time"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "charttime"})
                break
    elif declared_time_col != "charttime":
        df = df.rename(columns={declared_time_col: "charttime"})
    time_col = "charttime"

    if val_col not in df.columns:
        # Fallback to first non-meta column
        non_meta = [c for c in df.columns if c not in [id_col, time_col]]
        if non_meta:
            val_col = non_meta[0]

    df[val_col] = pd.to_numeric(df[val_col], errors="coerce").fillna(0)
    df = df.sort_values([id_col, time_col])
    df["fluid_balance_cumulative"] = df.groupby(id_col)[val_col].cumsum()
    result = df[[id_col, time_col, "fluid_balance_cumulative"]].copy()

    return _as_icutbl(
        result,
        id_columns=[id_col],
        index_column=time_col,
        value_column="fluid_balance_cumulative",
    )


CALLBACK_REGISTRY: MutableMapping[str, CallbackFn] = {
    "bmi": _callback_bmi,
    "anion_gap": _callback_anion_gap,
    "pulse_pressure": _callback_pulse_pressure,
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
    "pafi": lambda tables, ctx: _callback_pafi(tables, ctx, source_col_a="po2", source_col_b="fio2", output_col="pafi",
                                         database=getattr(ctx.data_source.config, 'name', '') if hasattr(ctx.data_source, 'config') and hasattr(ctx.data_source.config, 'name') else None),
    # SaFi = SpO2/FiO2 ratio (oxygen saturation / inspired oxygen fraction)
    "safi": lambda tables, ctx: _callback_pafi(tables, ctx, source_col_a="o2sat", source_col_b="fio2", output_col="safi",
                                         database=getattr(ctx.data_source.config, 'name', '') if hasattr(ctx.data_source, 'config') and hasattr(ctx.data_source.config, 'name') else None),
    "supp_o2": _callback_supp_o2,
    "supp_o2_aumc": _callback_supp_o2_aumc,
    "vent_ind": _callback_vent_ind,
    "urine24": _callback_urine24,
    "fluid_balance_admitted": _callback_fluid_balance_admitted,
    "fluid_balance_hourly": _callback_fluid_balance_hourly,
    "fluid_balance_cumulative": _callback_fluid_balance_cumulative,
    "vaso_ind": _callback_vaso_ind,
    "vaso_ind_rate": _callback_vaso_ind_rate,
    "sep3": _callback_sep3,
    "sep3_sofa2": _callback_sep3_sofa2,
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
    "sic_death": _callback_sic_death,
    "aumc_bxs": _callback_aumc_bxs,
    "aumc_rass": _callback_aumc_rass,
    "aumc_dur": _callback_aumc_dur,
    "blood_cell_ratio": _callback_blood_cell_ratio,
    "transform_fun(aumc_rass)": _callback_aumc_rass,  # Handle transform_fun wrapper
    "miiv_icu_patients_filter": _callback_miiv_icu_patients_filter,  # Filter MIMIC-IV patients to ICU only
    # MIMIC-III-specific callbacks
    "mimic_age": _callback_mimic_age,
    "transform_fun(mimic_age)": _callback_mimic_age,  # Handle transform_fun wrapper
    "mimic_abx_presc": _callback_mimic_abx_presc,
    "mimic_kg_rate": _callback_mimic_kg_rate,
    # Simple passthrough callbacks for concepts that just load from source
    "tco2": lambda tables, ctx: _callback_simple_passthrough(tables, ctx, "tco2"),
    "ca": lambda tables, ctx: _callback_simple_passthrough(tables, ctx, "ca"),
    # Ventilator parameters
    "driving_pressure": _callback_driving_pressure,
    # KDIGO AKI callbacks (loaded from kdigo_aki module)
    "kdigo_aki": _callback_kdigo_aki,
    "kdigo_creatinine": _callback_kdigo_creatinine,
    "kdigo_uo": _callback_kdigo_uo,
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

def _callback_miiv_icu_patients_filter(
    tables: Dict[str, ICUTable],
    ctx: ConceptCallbackContext
) -> ICUTable:
    """Filter MIMIC-IV patients data to only include ICU patients.

    This callback connects patients table with icustays table to ensure
    that demographic data (age, sex) only includes ICU patients,
    matching the ID system used by other concepts (stay_id).

    Args:
        tables: Dictionary containing loaded tables
        ctx: Callback context with database and other information

    Returns:
        Filtered table with ICU patients only
    """
    from ..datasource import ICUDataSource

    # Get database name from context
    database = ctx.database if ctx.database else 'miiv'

    if database != 'miiv':
        # For non-MIMIC-IV databases, return first table unchanged
        return next(iter(tables.values()))

    try:
        # Get ICUDataSource instance
        ds = ICUDataSource.get_instance(database)

        # Load icustays table to get mapping between subject_id and stay_id
        icustays_df = ds.load_table('icustays', columns=['stay_id', 'subject_id'])

        if icustays_df.empty:
            # If no icustays data, return empty table with expected structure
            main_table = next(iter(tables.values()))
            from ..table import IdTbl
            empty_df = pd.DataFrame(columns=['stay_id'] + [col for col in main_table.columns if col != 'subject_id'])
            return IdTbl(empty_df, id_vars=['stay_id'])

        # Get the main table (patients data)
        main_table = next(iter(tables.values()))
        data_df = main_table.to_pandas()

        # Merge patients data with icustays to filter only ICU patients
        if 'subject_id' in data_df.columns and 'subject_id' in icustays_df.columns:
            # Ensure both subject_id columns are the same type for proper merging
            data_copy = data_df.copy()
            icustays_copy = icustays_df.copy()

            data_copy['subject_id'] = data_copy['subject_id'].astype(icustays_copy['subject_id'].dtype)

            # Merge to keep only ICU patients
            merged = pd.merge(
                icustays_copy[['stay_id', 'subject_id']],
                data_copy,
                on='subject_id',
                how='inner'
            )

            # Set stay_id as the primary ID column
            merged = merged.set_index('stay_id')

            # Remove subject_id column as stay_id is now primary
            merged = merged.drop(columns=['subject_id'], errors='ignore')

            # Convert back to ICUTable
            return IdTbl(merged, id_vars=['stay_id'])
        else:
            # If expected columns not found, return original table
            return main_table

    except Exception:
        # If any error occurs during filtering, return original table
        # This ensures the system doesn't break if icustays table is unavailable
        return next(iter(tables.values()))

# miiv_icu_patients_filter is imported from callbacks.py at the top of this file
# No need to redefine it here



# Note: _apply_convert_unit_post_agg was removed (dead code).
# Unit conversion is now handled inline in DuckDB via CASE WHEN + regexp_matches
# in datasource.py's load_bucketed_table_aggregated().
