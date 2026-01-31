"""High level concept callback implementations (R ricu callback-cncpt.R).

This module provides concept-level aggregation utilities that operate on
collections of :class:`~pyricu.table.ICUTable` objects as produced by the
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
from .table import ICUTable, WinTbl
from .utils import coalesce, compute_patient_ids_hash as _compute_patient_ids_hash  # 🔧 统一的 patient_ids hash 函数

logger = logging.getLogger(__name__)
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
        
        # 先处理frame中的重复列（例如合并多个item时可能产生重复的measuredat列）
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
                # 检查merge所需的键列是否都存在
                actual_key_cols = [col for col in key_cols if col in frame.columns and col in merged.columns]
                if len(actual_key_cols) < len(key_cols):
                    missing_in_frame = [col for col in key_cols if col not in frame.columns]
                    missing_in_merged = [col for col in key_cols if col not in merged.columns]
                    # 使用 logging.debug 代替 print，避免在每个 chunk 都打印重复警告
                    logging.debug(f"跳过 '{name}': 缺少合并键列 (frame缺少: {missing_in_frame}, merged缺少: {missing_in_merged})")
                    continue

                # 如果frame有与merged重复的列（除了actual_key_cols），先删除frame中的重复列
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
    
    In MIMIC-III, age is calculated from date of birth (dob) to admittime.
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
    
    # MIMIC-III patients table has dob, admissions has admittime
    # The age is typically already pre-calculated as anchor_age in newer versions
    # or needs to be computed from dob - admittime
    
    # Check if 'age' column already exists (some MIMIC-III setups have it)
    if 'age' in data.columns:
        data['age'] = pd.to_numeric(data['age'], errors='coerce')
        # Cap at 90
        data.loc[data['age'] > 90, 'age'] = 90
    elif 'anchor_age' in data.columns:
        # MIMIC-IV style anchor_age
        data['age'] = pd.to_numeric(data['anchor_age'], errors='coerce')
        data.loc[data['age'] > 90, 'age'] = 90
    elif 'dob' in data.columns and 'admittime' in data.columns:
        # Calculate age from dob and admittime
        dob = pd.to_datetime(data['dob'], errors='coerce')
        admittime = pd.to_datetime(data['admittime'], errors='coerce')
        # Age in days then convert to years
        age_days = (admittime - dob).dt.days
        age_years = age_days / 365.25
        # If age > 90, cap at 90 (MIMIC de-identification)
        age_years = np.where(age_years > 90, 90, age_years)
        data['age'] = age_years
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
        # 🚀 优化：优先使用 _raw_concept_cache 中已预缓存的 vaso_ind
        tables = dict(tables)
        if ctx.concept_name in {"sofa_cardio", "sofa2_cardio"} and "vaso_ind" not in tables:
            try:
                vaso_tbl = None
                # 优先从 _raw_concept_cache 获取
                if hasattr(ctx.resolver, '_raw_concept_cache') and hasattr(ctx.resolver, '_cache_lock'):
                    # 🔧 使用统一的 hash 函数
                    patient_ids_hash = _compute_patient_ids_hash(ctx.patient_ids)
                    
                    cache_key = ("vaso_ind", patient_ids_hash)
                    with ctx.resolver._cache_lock:
                        if cache_key in ctx.resolver._raw_concept_cache:
                            vaso_tbl = ctx.resolver._raw_concept_cache[cache_key]
                            if hasattr(vaso_tbl, 'copy'):
                                vaso_tbl = vaso_tbl.copy()
                
                # 如果缓存未命中，则加载
                if vaso_tbl is None:
                    loaded = ctx.resolver.load_concepts(
                        ["vaso_ind"],
                        ctx.data_source,
                        merge=False,
                        aggregate={"vaso_ind": "max"},
                        patient_ids=ctx.patient_ids,
                        interval=None,
                        align_to_admission=True,
                        ricu_compatible=False,
                    )
                    if isinstance(loaded, dict):
                        vaso_tbl = loaded.get("vaso_ind")
                    else:
                        vaso_tbl = loaded
                
                # Handle both ICUTable and DataFrame returns
                if isinstance(vaso_tbl, ICUTable) and not vaso_tbl.data.empty:
                    tables["vaso_ind"] = vaso_tbl
                elif isinstance(vaso_tbl, pd.DataFrame) and not vaso_tbl.empty:
                    id_cols = [c for c in vaso_tbl.columns if c in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid']]
                    time_cols = [c for c in vaso_tbl.columns if 'time' in c.lower() and c not in id_cols]
                    index_col = time_cols[0] if time_cols else None
                    value_col = 'vaso_ind' if 'vaso_ind' in vaso_tbl.columns else None
                    tables["vaso_ind"] = _as_icutbl(vaso_tbl, id_columns=id_cols, index_column=index_col, value_column=value_col)
            except Exception as e:
                logger.debug(f"sofa_cardio: vaso_ind load exception: {e}")
                pass
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

        # CRITICAL: Ensure time points are sorted after merge
        if index_column:
            sort_cols = id_columns + [index_column] if id_columns else [index_column]
            data = data.sort_values(sort_cols)

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

                # 完全复制R ricu的expand逻辑
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
    
    # 统一ID列名 - 不同概念可能使用不同的ID列名（stay_id vs admissionid等）
    # 如果vent_df和pafi_df的ID列名不一致，重命名为统一的列名
    
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
                        "无法自动转换：pafi的时间列是numeric但vent是datetime，且没有数据源上下文。"
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
                        "无法自动转换：vent的时间列是numeric但pafi是datetime，且没有数据源上下文。"
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
    
    # R's behavior for sofa_resp
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
    
    # 移除错误的PaFi调整逻辑
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
        
        # Infer interval from data (median time difference per patient)
        if id_cols_to_group and len(data) > 1:
            time_diffs = []
            for _, group in data.groupby(id_cols_to_group):
                if len(group) > 1:
                    sorted_times = group[index_column].sort_values()
                    diffs = sorted_times.diff().dropna()
                    time_diffs.extend(diffs.tolist())
            if time_diffs:
                inferred_interval = pd.Series(time_diffs).median()
                # Handle numeric (hours) vs timedelta
                if isinstance(inferred_interval, (int, float)):
                    # Numeric time in hours
                    interval = pd.Timedelta(hours=max(1, round(inferred_interval)))
                else:
                    # Timedelta
                    inferred_hours = round(inferred_interval.total_seconds() / 3600)
                    interval = pd.Timedelta(hours=max(1, inferred_hours))
                
                # Fill gaps with inferred interval
                # ✅ FIX: Use default expand_forward=True to match ricu's symmetric timeline
                # This generates complete hourly grid covering the full patient timeline
                limits_df = _compose_fill_limits(data, id_cols_to_group, index_column, ctx)
                data = fill_gaps(
                    data,
                    id_cols=id_cols_to_group,
                    index_col=index_column,
                    interval=interval,
                    limits=limits_df,
                    method="none",
                )
                data = data.sort_values(list(id_columns) + [index_column] if id_columns else [index_column])
        
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
        
        # Infer interval from data (same as SOFA-1)
        if id_cols_to_group and len(data) > 1:
            time_diffs = []
            for _, group in data.groupby(id_cols_to_group):
                if len(group) > 1:
                    sorted_times = group[index_column].sort_values()
                    diffs = sorted_times.diff().dropna()
                    time_diffs.extend(diffs.tolist())
            
            if time_diffs:
                inferred_interval = pd.Series(time_diffs).median()
                # Handle numeric (hours) vs timedelta
                if isinstance(inferred_interval, (int, float)):
                    interval = pd.Timedelta(hours=max(1, round(inferred_interval)))
                else:
                    inferred_hours = round(inferred_interval.total_seconds() / 3600)
                    interval = pd.Timedelta(hours=max(1, inferred_hours))
                
                # Fill gaps with inferred interval
                # ✅ FIX: Use default expand_forward=True to match ricu's symmetric timeline
                limits_df = _compose_fill_limits(data, id_cols_to_group, index_column, ctx)
                data = fill_gaps(
                    data,
                    id_cols=id_cols_to_group,
                    index_col=index_column,
                    interval=interval,
                    limits=limits_df,
                    method="none",
                )
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
    
    # For each patient, apply LOCF within the time window
    # IMPORTANT: Use ORIGINAL values only, not values that were already LOCF-filled
    def locf_within_window(group):
        group = group.sort_values("_time_hours_")
        times = group["_time_hours_"].values
        n = len(times)
        
        for col in value_columns:
            if col not in group.columns:
                continue
            # Keep original values separate to avoid cascading propagation
            original_values = group[col].values.copy()
            result_values = original_values.copy()
            
            for i in range(n):
                if pd.isna(result_values[i]):
                    # Look backward within the window for the LAST ORIGINAL non-NA value
                    # This matches R locf: last_elem(x[!is.na(x)])
                    last_valid = np.nan
                    for j in range(i - 1, -1, -1):
                        if times[i] - times[j] <= win_length_hours:
                            if pd.notna(original_values[j]):
                                last_valid = original_values[j]
                                break  # Found the most recent original value
                        else:
                            break  # Outside window, stop looking
                    result_values[i] = last_valid
            
            group[col] = result_values
        return group
    
    result = data.groupby(id_columns, dropna=False, group_keys=False).apply(locf_within_window)
    result = result.drop(columns=["_time_hours_"], errors="ignore")
    
    return result


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
        # This matches R's rolling join behavior
        o2_df = o2_tbl.data.copy()
        fio2_df = fio2_tbl.data.copy()
        
        # CRITICAL FIX: Standardize time column names before processing
        # o2_tbl and fio2_tbl may have different index_column names (e.g., 'charttime' vs 'measuredat_minutes')
        # We need to rename them to a common name for merge_asof to work
        o2_idx_col = o2_tbl.index_column
        fio2_idx_col = fio2_tbl.index_column
        
        # 🔧 FIX 2025-01-30: 智能检测并统一时间列
        # 问题场景：
        #   - _assert_shared_schema 返回 index_column='measuredat'（来自 ICUTable.index_column 属性）
        #   - 但实际数据列是 'measuredat_minutes'（来自 DuckDB 聚合）
        # 解决方案：检测数据中实际存在的时间列，并统一重命名为 charttime
        
        time_col_priority = ['charttime', 'measuredat_minutes', 'datetime', 'givenat', 'measuredat']
        
        def detect_actual_time_col(df, declared_idx_col):
            """检测数据中实际的时间列"""
            # 优先使用声明的 index_column（如果在数据中存在且有有效值）
            if declared_idx_col and declared_idx_col in df.columns:
                if not df[declared_idx_col].isna().all():
                    return declared_idx_col
            # 按优先级查找有有效值的时间列
            for col in time_col_priority:
                if col in df.columns and not df[col].isna().all():
                    return col
            # 回退到声明的列（即使全是 NaN）
            return declared_idx_col
        
        o2_actual_time_col = detect_actual_time_col(o2_df, o2_idx_col)
        fio2_actual_time_col = detect_actual_time_col(fio2_df, fio2_idx_col)
        
        # 统一使用 'charttime' 作为标准时间列名
        unified_time_col = 'charttime'
        
        # 删除冗余的时间列，只保留实际使用的那个
        for df_ref, actual_col in [(o2_df, o2_actual_time_col), (fio2_df, fio2_actual_time_col)]:
            cols_to_drop = [col for col in time_col_priority 
                           if col in df_ref.columns and col != actual_col]
            if cols_to_drop:
                df_ref.drop(columns=cols_to_drop, inplace=True)
            
            # 重命名为统一的时间列名
            if actual_col and actual_col != unified_time_col and actual_col in df_ref.columns:
                df_ref.rename(columns={actual_col: unified_time_col}, inplace=True)
        
        # 更新 index_column 为统一的时间列名
        index_column = unified_time_col
        o2_idx_col = unified_time_col
        fio2_idx_col = unified_time_col
        
        # Rename value columns
        o2_val_col = o2_tbl.value_column or o2_col
        fio2_val_col = fio2_tbl.value_column or fio2_col
        
        if o2_val_col != o2_col:
            o2_df = o2_df.rename(columns={o2_val_col: o2_col})
        if fio2_val_col != fio2_col:
            fio2_df = fio2_df.rename(columns={fio2_val_col: fio2_col})
        
        # Use pd.merge_asof for rolling join (similar to R's data.table rolling join)
        if index_column:
            # 时间列已在上面统一为 unified_time_col，无需再次重命名
            
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
            # 🔧 FIX: 经过 downsampling 后，所有数据库的时间都已转换为小时
            # AUMC 原始数据是毫秒 -> datasource.py 转换为分钟 -> downsampling 转换为小时
            # 所以这里统一使用小时单位，不需要数据库特定处理
            numeric_unit = 'h'  # 所有数据库在 downsampling 后都使用小时
            if o2_time_is_numeric:
                o2_time_backup = o2_df[index_column]
                # 对于numeric类型，需要转换为datetime进行merge_asof
                o2_df = o2_df.copy()  # Only copy when we need to modify
                o2_df[index_column] = base_time + pd.to_timedelta(o2_df[index_column], unit=numeric_unit)
            if fio2_time_is_numeric:
                fio2_df[index_column]
                fio2_df = fio2_df.copy()  # Only copy when we need to modify
                fio2_df[index_column] = base_time + pd.to_timedelta(fio2_df[index_column], unit=numeric_unit)
            
            # 确保数据在每个by分组内都是排序的（merge_asof的严格要求）
            # 先选择需要的列，然后排序
            o2_subset = o2_df[id_columns + [index_column, o2_col]]
            fio2_subset = fio2_df[id_columns + [index_column, fio2_col]]

            # 新增：数据库自适应的FiO2单位标准化
            if database is not None and not fio2_subset.empty:
                logger.debug(f"开始FiO2单位标准化: database={database}, fio2_col={fio2_col}, 数据行数={len(fio2_subset)}")
                # 调试：显示原始数据范围
                if fio2_col in fio2_subset.columns:
                    orig_values = fio2_subset[fio2_col].dropna()
                    if len(orig_values) > 0:
                        logger.debug(f"原始FiO2值范围: {orig_values.min():.3f} - {orig_values.max():.3f}")

                fio2_subset = _standardize_fio2_units(fio2_subset, fio2_col, database)

                # 调试：显示转换后数据范围
                if fio2_col in fio2_subset.columns:
                    conv_values = fio2_subset[fio2_col].dropna()
                    if len(conv_values) > 0:
                        logger.debug(f"转换后FiO2值范围: {conv_values.min():.3f} - {conv_values.max():.3f}")
            else:
                logger.debug(f"跳过FiO2单位标准化: database={database}, fio2_subset.empty={fio2_subset.empty}")

            # 移除NaN时间值（NaN会导致排序问题）
            o2_subset = o2_subset.dropna(subset=[index_column])
            fio2_subset = fio2_subset.dropna(subset=[index_column])
            
            # 关键：merge_asof要求每个by分组内的on列必须严格排序
            # 必须按by列+on列排序，确保每个分组内on列都是递增的
            if id_columns:
                # 🚀 性能优化：使用全局排序替代逐个分组排序
                # 原始循环方式在2000患者时耗时严重 (O(N*M))
                # 全局排序 (O(N*M*log(N*M))) 在Pandas中通常更快，因为是C层实现
                
                # 确保排序稳定 (kind='mergesort')
                sort_cols = id_columns + [index_column]
                
                if not o2_subset.empty:
                    o2_subset = o2_subset.sort_values(by=sort_cols, kind='mergesort')
                
                if not fio2_subset.empty:
                    fio2_subset = fio2_subset.sort_values(by=sort_cols, kind='mergesort')
                
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
            
            # 🚀 OPTIMIZATION 2025-01-31: 使用 merge_asof 的 by 参数进行批量处理
            # 之前的实现对每个患者ID分别调用 merge_asof (2*N 次调用)
            # 优化后使用 by 参数，只需要 2 次调用
            if id_columns:
                # 处理 fio2 为空的患者
                o2_patient_ids = set(o2_subset[id_columns[0]].unique())
                fio2_patient_ids = set(fio2_subset[id_columns[0]].unique())
                
                # 找出只有 o2 数据但没有 fio2 数据的患者
                patients_without_fio2 = o2_patient_ids - fio2_patient_ids
                
                # 对于没有 fio2 的患者，直接创建 NaN 结果
                if patients_without_fio2:
                    o2_no_fio2 = o2_subset[o2_subset[id_columns[0]].isin(patients_without_fio2)]
                    merged_no_fio2 = o2_no_fio2.assign(**{fio2_col: float('nan')})
                else:
                    merged_no_fio2 = pd.DataFrame(columns=id_columns + [index_column, o2_col, fio2_col])
                
                # 对于有 fio2 数据的患者，使用 merge_asof 的 by 参数进行批量处理
                patients_with_fio2 = o2_patient_ids & fio2_patient_ids
                
                if patients_with_fio2:
                    o2_with_fio2 = o2_subset[o2_subset[id_columns[0]].isin(patients_with_fio2)].copy()
                    fio2_with_fio2 = fio2_subset[fio2_subset[id_columns[0]].isin(patients_with_fio2)].copy()
                    
                    # 🔧 FIX: merge_asof 的 by 参数要求每个 by 组内的 on 列必须单调递增
                    # 先按 by+on 排序，然后重置索引以确保符合 merge_asof 的严格要求
                    o2_with_fio2 = o2_with_fio2.sort_values(
                        by=id_columns + [index_column], 
                        kind='mergesort'
                    ).reset_index(drop=True)
                    fio2_with_fio2 = fio2_with_fio2.sort_values(
                        by=id_columns + [index_column],
                        kind='mergesort'
                    ).reset_index(drop=True)
                    
                    # 🔧 FIX: 当时间列是数值类型（小时）时，tolerance 也需要转换为数值
                    effective_tolerance = match_win
                    if pd.api.types.is_numeric_dtype(o2_with_fio2[index_column]):
                        # 时间列已经是小时单位，tolerance 也转换为小时
                        effective_tolerance = match_win.total_seconds() / 3600.0
                    
                    try:
                        # Forward join: 使用 by 参数批量处理
                        merged_fwd = pd.merge_asof(
                            o2_with_fio2[[*id_columns, index_column, o2_col]],
                            fio2_with_fio2[[*id_columns, index_column, fio2_col]],
                            on=index_column,
                            by=id_columns,
                            tolerance=effective_tolerance,
                            direction='backward'
                        )
                    except Exception:
                        # 如果 by 参数失败（例如 pandas 的全局排序要求），回退到逐个处理
                        # 这是预期行为，因为 pandas 的 merge_asof 即使使用 by 参数
                        # 也要求 on 列全局单调递增，这在多患者数据中很难满足
                        merged_fwd = _match_fio2_fallback_loop(
                            o2_with_fio2, fio2_with_fio2, id_columns, index_column, 
                            o2_col, fio2_col, match_win, 'forward'
                        )
                    
                    try:
                        # Backward join: 使用 by 参数批量处理
                        merged_bwd = pd.merge_asof(
                            fio2_with_fio2[[*id_columns, index_column, fio2_col]],
                            o2_with_fio2[[*id_columns, index_column, o2_col]],
                            on=index_column,
                            by=id_columns,
                            tolerance=effective_tolerance,
                            direction='backward'
                        )
                    except Exception:
                        # 回退到逐个处理
                        merged_bwd = _match_fio2_fallback_loop(
                            fio2_with_fio2, o2_with_fio2, id_columns, index_column,
                            fio2_col, o2_col, match_win, 'backward'
                        )
                    
                    # 合并两个方向的结果
                    merge_cols = id_columns + [index_column, o2_col, fio2_col]
                    merged_fwd = merged_fwd[merge_cols] if not merged_fwd.empty else pd.DataFrame(columns=merge_cols)
                    merged_bwd = merged_bwd[merge_cols] if not merged_bwd.empty else pd.DataFrame(columns=merge_cols)
                    
                    merged_with_fio2 = pd.concat([merged_fwd, merged_bwd], ignore_index=True)
                    merged_with_fio2 = merged_with_fio2.drop_duplicates()
                else:
                    merged_with_fio2 = pd.DataFrame(columns=id_columns + [index_column, o2_col, fio2_col])
                
                # 合并有 fio2 和无 fio2 的结果
                merged = pd.concat([merged_with_fio2, merged_no_fio2], ignore_index=True)
            else:
                # 没有ID列，直接处理
                merged_fwd = pd.merge_asof(
                    o2_subset,
                    fio2_subset,
                    on=index_column,
                    tolerance=match_win,
                    direction='backward'
                )
                merged = merged_fwd
            
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

    # vent_ind is often a WinTbl indexed by starttime while fio2 uses charttime.
    # Align the index column names so we can merge on a common timeline.
    vent_index = vent_tbl.index_column
    fio2_index = fio2_tbl.index_column

    # Prefer the fio2 index (charttime) for the merged timeline.
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

    # 🔧 FIX: 在 reindex 前需要处理原始 Series 中的重复索引
    # 当同一时间点有多个值时，取最后一个（或第一个），避免 reindex 时报错
    # "cannot assemble with duplicate keys"
    if vent_series.index.duplicated().any():
        vent_series = vent_series[~vent_series.index.duplicated(keep='last')]
    if fio2_series.index.duplicated().any():
        fio2_series = fio2_series[~fio2_series.index.duplicated(keep='last')]

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
        # pyricu expand() also uses inclusive end (after ts_utils fix)
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
    # 如果 mech_vent 有数据 → 只使用 mech_vent，忽略 vent_start/vent_end
    # 否则 → 使用 vent_start + vent_end 匹配
    # 
    # 参考 R 代码:
    #   if (has_rows(res[[3L]])) {  # mech_vent
    #     assert_that(nrow(res[[1L]]) == 0L, nrow(res[[2L]]) == 0L)  # vent_start/end should be empty
    #     res <- res[[3L]][, c("vent_ind", "mech_vent") := ...]
    #     return(res)
    #   }
    #   # else: use vent_start/vent_end
    
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
    
    # Constants for ricu algorithm
    min_steps = 12  # min_win = 12 hours
    step_factor = 24.0  # step_factor = 24
    
    def process_patient(group):
        """Process urine24 for a single patient.
        
        IMPORTANT: This replicates ricu's fill_gaps behavior where the collapse() function
        returns a win_tbl with start=min and end=duration (not absolute end time).
        When fill_gaps uses expand() with this win_tbl, it incorrectly treats the duration
        as an absolute end time because 'end' is already in colnames, so the duration->end
        calculation is skipped.
        
        For patient 141179 with urine times [11, 15, 23, 28]:
        - collapse returns start=11, end=17 (duration = 28-11 = 17)
        - fill_gaps erroneously generates seq(11, 17) = [11,12,13,14,15,16,17]
        - This is different from the expected [11..28] range
        
        We replicate this behavior to match ricu's output exactly.
        """
        group = group.sort_values(time_col).reset_index(drop=True)
        original_times = group[time_col].values
        original_urine = group[urine_col].values
        
        if len(original_times) == 0:
            return pd.DataFrame(columns=[time_col, urine_col, 'urine24'] + id_cols)
        
        # Step 1: Get min and max times (ricu's collapse behavior)
        start_time = original_times[0]  # min
        actual_end_time = original_times[-1]  # max
        
        # CRITICAL: Match ricu's buggy behavior
        # ricu's collapse() stores duration in 'end' column, but fill_gaps/expand
        # mistakenly treats it as absolute end time when 'end' is already a column
        # Duration = max - min
        duration = actual_end_time - start_time
        
        # ricu uses this duration value directly as the end time (buggy behavior we need to match)
        # So time_grid = range(start_time, duration)
        # For patient 141179: start=11, duration=17, so grid = seq(11, 17) = [11..17]
        if is_numeric_time:
            # Use duration as the absolute end time (matching ricu's bug)
            ricu_end_time = duration  # NOT start_time + duration
            time_grid = np.arange(start_time, ricu_end_time + interval_hours, interval_hours)
        else:
            ricu_end_time = pd.Timedelta(hours=duration) if isinstance(duration, (int, float)) else duration
            time_grid = pd.date_range(start=start_time, end=start_time + ricu_end_time, freq=interval)
        
        # Step 3: Build filled DataFrame with urine values
        filled_df = pd.DataFrame({time_col: time_grid})
        
        # Merge with original urine values
        orig_df = pd.DataFrame({time_col: original_times, urine_col: original_urine})
        filled_df = filled_df.merge(orig_df, on=time_col, how='left')
        filled_df[urine_col] = filled_df[urine_col].fillna(0.0)
        filled_df = filled_df.sort_values(time_col).reset_index(drop=True)
        
        # Step 4: Compute urine24 using vectorized sliding window
        # 🚀 OPTIMIZATION 2025-01-31: Replace per-row loop with pandas rolling
        n = len(filled_df)
        urine_vals = filled_df[urine_col].values
        
        if is_numeric_time:
            window_size = int(24.0 / interval_hours)
        else:
            window_size = int(pd.Timedelta(hours=24) / interval)
        
        # Use pandas rolling for vectorized computation
        rolling_sum = pd.Series(urine_vals).rolling(
            window=window_size, min_periods=min_steps, center=False
        ).sum()
        
        # Calculate actual window lengths for all positions
        positions = np.arange(n) + 1
        actual_window_lens = np.minimum(positions, window_size)
        
        # Calculate urine24: sum * step_factor / window_length
        urine24_values = np.where(
            (actual_window_lens >= min_steps) & pd.notna(rolling_sum.values),
            rolling_sum.values * step_factor / actual_window_lens,
            np.nan
        )
        
        filled_df['urine24'] = urine24_values
        
        # Add ID columns
        for col in id_cols:
            if col in group.columns:
                filled_df[col] = group[col].iloc[0]
        
        return filled_df
    
    # Process each patient using groupby.apply
    if id_cols:
        df = df.sort_values(id_cols + [time_col]).reset_index(drop=True)
        result_df = df.groupby(id_cols, sort=False, group_keys=False).apply(
            process_patient, include_groups=False
        ).reset_index(drop=True)
    else:
        result_df = process_patient(df)
    
    # Return only the required columns
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
    if time_is_numeric:
        numeric_time = pd.to_numeric(time_series, errors="coerce")
        merged["__start_dt"] = base_time + pd.to_timedelta(numeric_time, unit="h")
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
    
    # 🔧 FIX: All databases use HOURS for relative time (not minutes)
    # The dobu_rate/dobu_dur data shows start=26,27,28... which are hours
    numeric_unit = 'h'

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

    # 🔧 FIX for ricu compatibility: R ricu applies change_interval (re_time with floor)
    # to sub-concepts BEFORE passing them to the vaso60 callback.
    # This means the start times are floored to whole hours.
    # We need to replicate this behavior to match ricu's join conditions.
    # Example: dur_starttime=13.26h should become 13:00, not 13:15.
    rate_df[rate_index_col] = rate_df[rate_index_col].dt.floor('h')
    dur_df[dur_index_col] = dur_df[dur_index_col].dt.floor('h')

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

    dur_df["__start"] = dur_df[dur_index_col]
    dur_df["__end"] = dur_df["__start"] + dur_df["__duration"]

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

    # 🔧 FIX: R ricu's vaso60 does NOT expand/fill missing hours with max values.
    # It simply joins rate to duration windows and keeps the ACTUAL hourly rate values.
    # The only aggregation is when the same hour has multiple rate values (takes max).
    # The previous expansion code incorrectly replaced all hourly values with the window max.
    #
    # R ricu logic:
    # 1. Join rate where rate.time >= dur.start AND rate.time <= dur.end
    # 2. change_interval() to floor times to hours (already done above)
    # 3. aggregate("max") - only aggregates if same hour has multiple values
    #
    # The `grouped` variable already contains the correct hourly values from step 1 and 3.
    # No expansion is needed.

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
    logger = logging.getLogger("pyricu")
    
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
    ett_gcs = None
    if ett_gcs_table is not None:
        if hasattr(ett_gcs_table, 'data'):
            ett_df = ett_gcs_table.data
        else:
            ett_df = ett_gcs_table
        if 'ett_gcs' in ett_df.columns and not ett_df.empty:
            # Merge ett_gcs to data on id and time columns (left join to preserve data's time points)
            merge_cols = list(id_columns) + ([index_column] if index_column else [])
            if all(c in ett_df.columns for c in merge_cols):
                ett_subset = ett_df[merge_cols + ['ett_gcs']].copy()
                # R ricu: sed <- sed[is_true(get(cnc[5L])), ] - only keep TRUE rows
                # Then inner join with data to find intubated time points
                ett_true = ett_subset[ett_subset['ett_gcs'].fillna(False)]
                if not ett_true.empty:
                    # Mark which rows in data are intubated
                    data = data.merge(ett_true[merge_cols + ['ett_gcs']], on=merge_cols, how='left')
                    ett_gcs = data.get("ett_gcs")

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

    # Calculate GCS: use tgcs if available AND valid (>=3), otherwise sum components
    # tgcs computed by sum_components may have incorrect values when vgcs is missing
    # GCS minimum is 3 (E1+M1+V1), so if tgcs<3, it's invalid and should be recalculated
    combined = pd.Series(index=data.index, dtype=float)
    
    if tgcs is not None:
        # Use tgcs where it's valid (>=3 or NaN)
        valid_tgcs = tgcs.where((tgcs >= 3) | tgcs.isna())
        combined = valid_tgcs.copy()
    
    # For rows where tgcs is NA or invalid (<3), calculate from components
    if egcs is not None and mgcs is not None and vgcs is not None:
        component_sum = egcs.add(mgcs, fill_value=np.nan).add(vgcs, fill_value=np.nan)
        combined = combined.fillna(component_sum)
    
    # If set_na_max=True and GCS is still NA, fill with 15 (perfect score)
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

from .callbacks import uo_6h as calc_uo_6h, uo_12h as calc_uo_12h, uo_24h as calc_uo_24h

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
            
            # 计算并封装为 ICUTable
            if "uo_6h" in missing_uo:
                df = calc_uo_6h(urine_df, weight_df, interval=ctx.interval)
                tables["uo_6h"] = _as_icutbl(df, id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column="uo_6h")
            
            if "uo_12h" in missing_uo:
                df = calc_uo_12h(urine_df, weight_df, interval=ctx.interval)
                tables["uo_12h"] = _as_icutbl(df, id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column="uo_12h")
                
            if "uo_24h" in missing_uo:
                df = calc_uo_24h(urine_df, weight_df, interval=ctx.interval)
                tables["uo_24h"] = _as_icutbl(df, id_columns=urine_tbl.id_columns, index_column=urine_tbl.index_column, value_column="uo_24h")
    
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
    
    For GCS (tgcs), this implements R ricu's set_na_max=TRUE behavior:
    - egcs NA -> 4 (max eye response)
    - vgcs NA -> 5 (max verbal response)  
    - mgcs NA -> 6 (max motor response)
    - tgcs NA -> 15 (max total)
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
    
    # 🔧 FIX: R ricu set_na_max=TRUE behavior for GCS
    # When computing tgcs, fill missing GCS components with their max values:
    # egcs=4, vgcs=5, mgcs=6
    # See R ricu callback-cncpt.R gcs() function
    try:
        ricu_mode = bool(ctx.kwargs.get('ricu_compatible', True)) if ctx and getattr(ctx, 'kwargs', None) is not None else True
    except Exception:
        ricu_mode = True
    
    is_tgcs = output_col and output_col.lower() == 'tgcs'
    
    # GCS component max values (R ricu set_na_max defaults)
    gcs_max_values = {
        'egcs': 4.0,  # Eye response max
        'vgcs': 5.0,  # Verbal response max
        'mgcs': 6.0,  # Motor response max
    }
    
    total = pd.Series(0, index=data.index, dtype=float)
    
    for col in component_cols:
        if col in data.columns:
            col_values = pd.to_numeric(data[col], errors='coerce')
            
            # Apply set_na_max for tgcs in ricu_compatible mode
            if ricu_mode and is_tgcs and col.lower() in gcs_max_values:
                max_val = gcs_max_values[col.lower()]
                col_values = col_values.fillna(max_val)
            else:
                col_values = col_values.fillna(0)
            
            total += col_values
    
    data[output_col] = total
    
    # Keep only rows where we have at least some data
    mask = pd.Series(False, index=data.index)
    for col in component_cols:
        if col in data.columns:
            mask |= data[col].notna()
    
    data = data[mask]
    
    # 🔧 FIX: R ricu set_na_max: 如果 tgcs 仍然是 NA，设置为 15
    # 注意: 移除了 ett_gcs 加载逻辑，因为：
    # 1. R ricu 的 sum_components() 函数只是简单求和，不涉及 sedation imputation
    # 2. ett_gcs 的 sedation imputation 只在完整的 gcs() callback 中使用
    # 3. 加载 ett_gcs 需要扫描大量数据（57个桶），严重影响性能（从2s变成200s+）
    if ricu_mode and is_tgcs:
        if output_col in data.columns:
            data[output_col] = data[output_col].fillna(15.0)

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
    from pyricu.callbacks import driving_pressure
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
    for col in ['charttime', 'measuredat_minutes', 'observationoffset', 'datetime']:
        if col in result.columns:
            time_col = col
            break
    
    return ICUTable(
        data=result,
        id_columns=[id_col],
        index_column=time_col,
        value_column='driving_pres'
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
    "pafi": lambda tables, ctx: _callback_pafi(tables, ctx, source_col_a="po2", source_col_b="fio2", output_col="pafi",
                                         database=getattr(ctx.data_source.config, 'name', '') if hasattr(ctx.data_source, 'config') and hasattr(ctx.data_source.config, 'name') else None),
    # SaFi = SpO2/FiO2 ratio (oxygen saturation / inspired oxygen fraction)
    "safi": lambda tables, ctx: _callback_pafi(tables, ctx, source_col_a="o2sat", source_col_b="fio2", output_col="safi",
                                         database=getattr(ctx.data_source.config, 'name', '') if hasattr(ctx.data_source, 'config') and hasattr(ctx.data_source.config, 'name') else None),
    "supp_o2": _callback_supp_o2,
    "supp_o2_aumc": _callback_supp_o2_aumc,
    "vent_ind": _callback_vent_ind,
    "urine24": _callback_urine24,
    "vaso_ind": _callback_vaso_ind,
    "vaso_ind_rate": _callback_vaso_ind_rate,
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
