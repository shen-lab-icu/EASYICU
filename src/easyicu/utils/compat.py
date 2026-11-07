"""
ricu兼容层 - 实现与R ricu一致的数据提取行为

该模块提供了与R ricu包load_concepts函数完全一致的数据提取逻辑，包括：
1. 时间网格对齐 - 所有概念对齐到共同的时间网格（默认1小时）
2. 窗口展开 - 将start/end时间窗口展开为逐小时记录
3. 概念合并 - 使用outer join合并多个概念
4. 静态概念填充 - 静态值（age, sex等）填充到所有时间点

用法示例:
    >>> from easyicu import load_concepts
    >>> 
    >>> # 提取生命体征（与ricu.R一致）
    >>> vitals = load_concepts(
    ...     ['hr', 'sbp', 'dbp', 'temp'],
    ...     database='miiv',
    ...     patient_ids=[30041748, 30046525],
    ...     interval='1h',  # 默认值，与ricu的hours(1L)一致
    ...     r_compatible=True  # 启用完整的ricu兼容模式
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# 概念模块定义（与ricu.R中的模块对应）
# ============================================================================

@dataclass
class ConceptModule:
    """概念模块定义，对应ricu.R中的数据提取分组"""
    name: str
    concepts: List[str]
    id_column: str = "stay_id"
    time_column: Optional[str] = "charttime"  # None表示静态概念（无时间维度）
    description: str = ""


# 与ricu.R中extract_data函数的模块对应
COMPAT_MODULES: Dict[str, ConceptModule] = {
    "demo": ConceptModule(
        name="demo",
        concepts=["age", "bmi", "height", "sex", "weight"],
        time_column=None,  # 静态概念
        description="基础人口统计学",
    ),
    "outcome": ConceptModule(
        name="outcome",
        concepts=[
            "death", "los_icu", "qsofa", "sirs", "sofa", 
            "sofa_cardio", "sofa_cns", "sofa_coag", "sofa_liver", 
            "sofa_renal", "sofa_resp"
        ],
        time_column="index_var",
        description="结局和SOFA评分",
    ),
    "vital": ConceptModule(
        name="vital",
        concepts=["dbp", "etco2", "hr", "map", "sbp", "temp"],
        description="生命体征",
    ),
    "neu": ConceptModule(
        name="neu",
        concepts=["avpu", "egcs", "gcs", "mgcs", "rass", "vgcs"],
        description="神经系统评估",
    ),
    "output": ConceptModule(
        name="output",
        concepts=["urine", "urine24"],
        description="尿量",
    ),
    "resp": ConceptModule(
        name="resp",
        concepts=[
            "ett_gcs", "mech_vent", "o2sat", "sao2", "pafi", 
            "resp", "safi", "supp_o2", "vent_ind"
        ],
        description="呼吸系统",
    ),
    "lab": ConceptModule(
        name="lab",
        concepts=[
            "alb", "alp", "alt", "ast", "bicar", "bili", "bili_dir", 
            "bun", "ca", "ck", "ckmb", "cl", "crea", "crp", "glu",
            "k", "mg", "na", "phos", "tnt"
        ],
        description="实验室检查",
    ),
    "blood": ConceptModule(
        name="blood",
        concepts=["be", "cai", "fio2", "hbco", "lact", "methb", "pco2", "ph", "po2", "tco2"],
        description="血气分析",
    ),
    "hematology": ConceptModule(
        name="hematology",
        concepts=[
            "bnd", "esr", "fgn", "hgb", "inr_pt", "lymph", "mch", 
            "mchc", "mcv", "neut", "plt", "ptt", "wbc"
        ],
        description="血液学检查",
    ),
    "med": ConceptModule(
        name="med",
        concepts=[
            "abx", "adh_rate", "cort", "dex", "dobu_dur", "dobu_rate", 
            "dobu60", "epi_dur", "epi_rate", "ins", "norepi_dur", 
            "norepi_equiv", "norepi_rate", "vaso_ind"
        ],
        time_column="starttime",
        description="药物治疗",
    ),
}


# 静态概念列表（target=id_tbl，需要填充到所有时间点的概念）
# 注意：death 不是静态概念，它是 lgl_cncpt，只在死亡时刻有值
STATIC_CONCEPTS = {"age", "sex", "bmi", "height", "weight", "los_icu"}

# 窗口型概念（需要展开start/end时间的概念）
# 包括：
# - 机械通气指标: mech_vent, vent_ind, supp_o2
# - 血管活性药物速率: *_rate, vaso_ind
# - dex: 输液概念，有 dur_var（aumc 使用 stop）
# - ett_gcs: 使用 ts_to_win_tbl(mins(360L)) 展开为 6 小时窗口
# 注意：ins 不在这里，因为 ricu 中它是 ts_tbl 而不是 win_tbl
WINDOW_CONCEPTS = {
    "mech_vent", "vent_ind", "supp_o2",
    "rrt",  # Sources with explicit treatment intervals expand; point-only sources remain points.
    "other_vaso",  # Likewise (2026-11-04 audit round 3, M NEW-2).
    "mech_circ_support",  # MIIV IABP procedure intervals beside charted device rows.
    "norepi_rate", "epi_rate", "dobu_rate", "adh_rate",
    "dopa_rate", "phn_rate", "vaso_ind",
    # Sedation/analgesia rate concepts (2026-05-13)
    "propofol_rate", "fentanyl_rate", "midazolam_rate",
    # 注意: "dex" 不应该在这里！它的 target: "win_tbl" 意味着返回原始格式，不展开
    # ett_gcs windows are recorded airway intervals in SICdb (median 16.9 h) and
    # HiRID (median 4 h); as in ricu, a merged export expands them over their
    # duration (2026-11-04 audit round 3, F-31). The resolver now publishes
    # ett_gcs on ricu's inclusive grid before the merge
    # (window_expansion.INCLUSIVE_ANY_WINDOW_CONCEPTS, r13gcs).
    "ett_gcs",
}

# Window ends a merged export recognises; a window without one is expanded
# over its declared ``dur_var`` instead.
INTERVAL_END_COLUMNS = ("endtime", "stop", "end_time", "end")


def merged_window_duration_column(name: str, frame: pd.DataFrame, dur_var: Optional[str]) -> Optional[str]:
    """Return the duration column a merged export expands ``name`` over, if any.

    A declared hours duration is authoritative, including when the frame
    retains an auxiliary absolute source end timestamp. Both endpoints must
    use the same clock before overlap is computed.
    """
    from ..table.duration import UNIT_HOURS, get_dur_var_unit

    if name not in WINDOW_CONCEPTS or not dur_var or dur_var not in frame.columns:
        return None
    if get_dur_var_unit(frame) != UNIT_HOURS:
        return None
    return dur_var

# Concepts that intentionally combine point observations with explicit
# treatment windows in at least one database.  When a multi-source frame has
# an end-time column, the generic window expander must expand only rows with a
# real end time and retain the point rows.  Treating the whole frame as a
# window table drops every point row (and, when source time columns differ,
# can drop the window rows too).
MIXED_POINT_WINDOW_CONCEPTS = {
    "rrt",
    # Infusions with an end time (MIIV, MIMIC-III MetaVision, AUMC, SICdb)
    # beside charted points (CareVue, eICU, HiRID); 2026-11-04 audit round 3,
    # M NEW-2.
    "other_vaso",
    # MIMIC-IV procedureevents IABP intervals beside charted device settings
    # (2026-11-04 audit round 3, full-dictionary M).
    "mech_circ_support",
    # MIMIC-III MetaVision antipsychotic infusions beside CareVue
    # prescriptions without an end time; the expansion dropped the
    # prescriptions (2026-11-04 audit round 4, R4-006).
    "delirium_tx_proxy",
}

# 点事件概念（不应展开为连续时间序列）
POINT_EVENT_CONCEPTS = {
    "abx", "samp", "cort", "dobu60", "susp_inf", "sep3", "avpu",
    "vent_end", "vent_start",  # Ventilation events: uses set_val(TRUE), point events
    "furosemide",  # Loop diuretic: lgl_cncpt with set_val(TRUE), point events from drug administration tables
    # New medication lgl_cncpt concepts (2026-05-12)
    "propofol", "midazolam", "dexmedetomidine",  # Sedation
    "fentanyl", "morphine",                       # Analgesia
    "heparin",                                    # Anticoagulation
    "mannitol",                                   # Osmotic diuretic
    "amiodarone",                                 # Antiarrhythmic
    "milrinone",                                  # Inotrope
    "rocuronium",                                 # Neuromuscular blocker
    # Batch 2 (2026-05-13)
    "lorazepam",                                  # Benzodiazepine
    "ketamine",                                   # Sedation/analgesia
    "vecuronium", "cisatracurium",                # Neuromuscular blockers
    "nitroglycerin",                              # Vasodilator
    # Batch 3 (2026-05-13)
    "pantoprazole",                               # PPI
    "vancomycin", "meropenem",                    # Specific antibiotics
    "calcium_iv",                                 # IV electrolyte
    # Batch 4 (2026-05-13)
    "potassium_iv", "magnesium_iv",               # IV electrolytes
    "albumin_iv",                                 # Colloid
    "packed_rbc",                                 # Blood product
    # Batch 5 (2026-05-13)
    "bicarbonate",                                # Acidosis correction
    "dextrose50",                                 # Hypoglycemia rescue
    "ffp", "platelets",                           # Blood products
    # Batch 6 (2026-05-13)
    "levetiracetam",                              # Antiepileptic
    "dexamethasone",                              # Specific corticosteroid
    "octreotide",                                 # Somatostatin analog
    "neostigmine",                                # NMB reversal
    # Batch 7 (2026-05-14)
    "phenytoin", "labetalol", "esmolol",          # Antiepileptic + beta blockers
    "diltiazem", "nicardipine",                   # Calcium channel blockers
    # Batch 8 (2026-05-14)
    "warfarin", "apixaban", "enoxaparin",         # Anticoagulants
    "aspirin",                                    # Antiplatelet
    "insulin",                                    # Insulin (boolean companion to ins rate)
    # Diuretic family (2026-11-03 extraction audit #19)
    "loop_diuretic", "thiazide_diuretic",
    "potassium_sparing_diuretic", "acetazolamide",
    # A dose attributed to its start: expanding over a source end time would
    # repeat the same milligrams in every hour of the infusion.
    "loop_diuretic_furosemide_eq",
}

# 时长概念（已编码持续时间，不需要展开）
DURATION_CONCEPTS = {
    "norepi_dur", "epi_dur", "dobu_dur", "dopa_dur"
}

# 🔧 FIX 2025-02-13: win_tbl 目标概念（返回事件级别数据，不进行时间聚合）
# 这些概念的 target: "win_tbl"，应该返回 starttime + dur_var 格式，不取整时间
WIN_TBL_CONCEPTS = {
    "dex",      # Dextrose: target="win_tbl" in concept-dict.json
    "ett_gcs",  # ETT GCS: uses ts_to_win_tbl callback
    "mech_vent",  # Mechanical ventilation: returns event-level data
}


# ============================================================================
# 时间处理工具
# ============================================================================

def time_to_hours(
    series: pd.Series, 
    id_series: Optional[pd.Series] = None,
    intime_lookup: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """将时间列转换为相对小时数
    
    Args:
        series: 时间序列（datetime64或已是数值）
        id_series: 对应的ID序列（用于分组计算相对时间）
        intime_lookup: 包含stay_id和intime的查找表
        
    Returns:
        相对于ICU入院的小时数
    """
    if series.empty:
        return series
    
    # 已经是数值类型
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # timedelta类型
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds() / 3600.0
    
    # datetime类型
    if pd.api.types.is_datetime64_any_dtype(series):
        # 移除时区信息
        clean = series.copy()
        if hasattr(clean.dtype, 'tz') and clean.dt.tz is not None:
            clean = clean.dt.tz_localize(None)
        
        # 如果有intime查找表，使用它
        if intime_lookup is not None and id_series is not None:
            # 需要返回相对于每个患者intime的小时数
            # 这需要在调用方处理
            pass
        
        # 按ID分组计算相对时间
        if id_series is not None:
            return clean.groupby(id_series).transform(
                lambda s: (s - s.min()).dt.total_seconds() / 3600.0
            )
        
        # 全局相对时间
        return (clean - clean.min()).dt.total_seconds() / 3600.0
    
    # 尝试转换为数值
    return pd.to_numeric(series, errors="coerce")


def round_to_interval(time_series: pd.Series, interval_hours: float = 1.0) -> pd.Series:
    """将时间四舍五入到指定间隔
    
    Args:
        time_series: 时间序列（小时数）
        interval_hours: 间隔（小时）
        
    Returns:
        四舍五入后的时间序列
    """
    if time_series.empty:
        return time_series
    
    # 使用floor而非round，与ricu行为一致
    return np.floor(time_series / interval_hours) * interval_hours


# ============================================================================
# 窗口展开
# ============================================================================

def expand_interval_rows(
    df: pd.DataFrame,
    concept_name: str,
    id_col: str = "id",
    time_col: str = "time",
    value_col: str = "value",
    endtime_col: str = "endtime",
    duration_col: str = "duration",
    interval_hours: float = 1.0,
    max_span_hours: float = 24 * 365,  # 最大展开范围
) -> pd.DataFrame:
    """展开时间窗口为逐小时记录
    
    将有start/end时间的记录按半开区间的正重叠展开为逐小时记录。
    
    Args:
        df: 输入DataFrame
        concept_name: 概念名称（用于判断是否需要展开）
        id_col: ID列名
        time_col: 开始时间列名
        value_col: 值列名
        endtime_col: 结束时间列名
        duration_col: 持续时间列名
        interval_hours: 时间间隔（小时）
        max_span_hours: 最大展开时长
        
    Returns:
        展开后的DataFrame
    """
    concept_lower = concept_name.lower()
    
    # 时长概念不展开
    if concept_lower.endswith("_dur") or concept_lower in DURATION_CONCEPTS:
        return df.drop(columns=[endtime_col, duration_col], errors="ignore")
    
    # 点事件概念不展开
    if concept_lower in POINT_EVENT_CONCEPTS:
        return df.drop(columns=[endtime_col, duration_col], errors="ignore")
    
    # 检查是否有时间列
    if time_col not in df.columns:
        return df.drop(columns=[endtime_col, duration_col], errors="ignore")
    
    # 检查是否有结束时间或持续时间
    has_end = endtime_col in df.columns and df[endtime_col].notna().any()
    has_duration = duration_col in df.columns and df[duration_col].notna().any()
    
    # 没有窗口信息，不展开
    if not has_end and not has_duration:
        return df.drop(columns=[endtime_col, duration_col], errors="ignore")
    
    # 只处理有值的行
    working = df.copy()
    if value_col in working.columns:
        has_value = working[value_col].notna()
        working = working[has_value].copy()
        if working.empty:
            return pd.DataFrame(columns=[id_col, time_col, value_col])
    
    # 确保时间是数值类型
    if not pd.api.types.is_numeric_dtype(working[time_col]):
        working[time_col] = time_to_hours(working[time_col])
    
    # 处理结束时间
    if has_end and not pd.api.types.is_numeric_dtype(working[endtime_col]):
        working[endtime_col] = time_to_hours(working[endtime_col])
    
    # 处理持续时间
    if has_duration:
        if pd.api.types.is_timedelta64_dtype(working[duration_col]):
            working[duration_col] = working[duration_col].dt.total_seconds() / 3600.0
        working[duration_col] = pd.to_numeric(working[duration_col], errors="coerce")
    
    # 计算结束时间
    starts = pd.to_numeric(working[time_col], errors="coerce")
    if has_end:
        ends = pd.to_numeric(working[endtime_col], errors="coerce")
    elif has_duration:
        ends = starts + working[duration_col].fillna(0)
    else:
        ends = starts
    
    # The native merged path shares the same positive-overlap owner as the
    # resolver. Only rows without an interval are point observations.
    from ..concept.window_expansion import expand_wintbl_vectorized

    working["__span"] = (ends - starts).clip(upper=max_span_hours)
    expanded = expand_wintbl_vectorized(
        working, idx_col=time_col, dur_col="__span", id_cols=[id_col],
        value_columns=[value_col], interval_hours=interval_hours,
    )
    missing_end = ends.isna()
    if missing_end.any():
        points = working.loc[missing_end, [id_col, time_col, value_col]].copy()
        points[time_col] = np.floor(points[time_col] / interval_hours) * interval_hours
        expanded = pd.concat([expanded, points], ignore_index=True)
    if expanded.empty:
        return pd.DataFrame(columns=[id_col, time_col, value_col])

    # 🔧 FIX: 按(id, time)聚合，根据数据类型选择聚合函数
    # 参考: ricu/R/tbl-utils.R 第 741-751 行:
    #   - numeric → median
    #   - logical → sum (或 any)  
    #   - character → first
    value_dtype = expanded[value_col].dtype
    # Pandas treats bool as a numeric dtype, so logical must be tested first.
    # Otherwise ``median`` silently converts an in-use flag to float 0/1.
    if pd.api.types.is_bool_dtype(value_dtype):
        agg_func = 'any'
    elif pd.api.types.is_numeric_dtype(value_dtype):
        agg_func = 'median'
    else:
        if concept_lower == 'mech_vent':
            return pool_mech_vent_levels(expanded, [id_col, time_col], value_col)
        agg_func = 'first'
    
    expanded = expanded.groupby([id_col, time_col], as_index=False).agg({value_col: agg_func})
    
    return expanded


MECH_VENT_MIXED_LEVEL = 'invasive|noninvasive'


def pool_mech_vent_levels(frame: pd.DataFrame, keys: List[str], value_col: str) -> pd.DataFrame:
    """Return one mech_vent level per key, pooled by category membership.

    Rule R5 allows invasive and non-invasive ventilation in the same cell; the
    cell then carries the declared mixed level.  Pooling is order-free: which
    source row came first or last cannot decide the category (audit R7-F013).
    Missing levels contribute no membership.  Vectorized reductions keep this
    affordable on full source tables.
    """
    work = frame[[*keys, value_col]].copy()
    work['__imv'] = work[value_col].isin(['invasive', MECH_VENT_MIXED_LEVEL])
    work['__niv'] = work[value_col].isin(['noninvasive', MECH_VENT_MIXED_LEVEL])
    pooled = work.groupby(keys, as_index=False).agg(
        {value_col: 'first', '__imv': 'max', '__niv': 'max'}
    )
    labels = pooled[value_col].astype('object')
    labels = labels.mask(pooled.__imv, 'invasive').mask(pooled.__niv, 'noninvasive')
    pooled[value_col] = labels.mask(pooled.__imv & pooled.__niv, MECH_VENT_MIXED_LEVEL)
    return pooled[[*keys, value_col]]


# ============================================================================
# 时间网格对齐
# ============================================================================

def build_time_grid(
    series_dict: Dict[str, pd.DataFrame],
    id_col: str = "id",
    time_col: str = "time",
) -> Optional[pd.DataFrame]:
    """构建所有概念的统一时间网格
    
    注意：这个函数确保包含所有患者，即使他们只有静态概念数据（无时间列）。
    对于只有静态数据的患者，在网格中创建一个 time=NaN 的占位行。
    
    Args:
        series_dict: 概念名称到DataFrame的映射
        id_col: ID列名
        time_col: 时间列名
        
    Returns:
        包含所有(id, time)组合的DataFrame，或None（如果没有数据）
    """
    time_frames = []
    static_ids = set()
    
    for name, df in series_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if id_col not in df.columns:
            continue
            
        if time_col in df.columns:
            # 有时间数据的概念
            time_frames.append(df[[id_col, time_col]])
        else:
            # 静态概念：收集患者ID
            static_ids.update(df[id_col].dropna().unique())
    
    if not time_frames and not static_ids:
        return None
    
    if time_frames:
        grid = (
            pd.concat(time_frames, ignore_index=True)
            .dropna(subset=[id_col, time_col])
            .drop_duplicates()
            .sort_values([id_col, time_col])
            .reset_index(drop=True)
        )
        # 确保静态概念的患者也在网格中
        grid_ids = set(grid[id_col].unique())
        missing_ids = static_ids - grid_ids
        if missing_ids:
            # 为缺失的患者添加一个 time=NaN 的占位行
            # 这样在后续的 left join 中，他们的静态数据可以被保留
            missing_rows = pd.DataFrame({
                id_col: list(missing_ids),
                time_col: [np.nan] * len(missing_ids)
            })
            grid = pd.concat([grid, missing_rows], ignore_index=True)
            grid = grid.sort_values([id_col, time_col]).reset_index(drop=True)
    else:
        # 只有静态数据，创建一个只有ID的网格（time=NaN）
        grid = pd.DataFrame({
            id_col: list(static_ids),
            time_col: [np.nan] * len(static_ids)
        })
    
    return grid if not grid.empty else None


def align_to_grid(
    concept_data: Dict[str, pd.DataFrame],
    grid: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "time",
    value_col: str = "value",
) -> Dict[str, pd.DataFrame]:
    """将所有概念对齐到统一的时间网格
    
    Args:
        concept_data: 概念名称到DataFrame的映射
        grid: 时间网格DataFrame
        id_col: ID列名
        time_col: 时间列名
        value_col: 值列名
        
    Returns:
        对齐后的概念数据字典
    """
    if grid is None or grid.empty:
        return concept_data

    use_lightweight = _should_use_lightweight_alignment(grid, concept_data)
    aligned = {}

    if use_lightweight:
        for name, df in concept_data.items():
            if df is None or df.empty:
                aligned[name] = pd.DataFrame(columns=[id_col, time_col, name])
                continue

            if time_col not in df.columns:
                aligned[name] = df
                continue

            df_copy = df.copy()
            df_copy[id_col] = pd.to_numeric(df_copy[id_col], errors="coerce")
            df_copy[time_col] = pd.to_numeric(df_copy[time_col], errors="coerce")
            df_copy = df_copy.dropna(subset=[id_col, time_col])
            aligned[name] = df_copy

        return aligned

    grid_copy = grid.copy()
    grid_copy[id_col] = pd.to_numeric(grid_copy[id_col], errors="coerce")
    grid_copy[time_col] = pd.to_numeric(grid_copy[time_col], errors="coerce")
    grid_copy = grid_copy.dropna(subset=[id_col, time_col]).drop_duplicates()

    for name, df in concept_data.items():
        if df is None or df.empty:
            placeholder = grid_copy.copy()
            placeholder[value_col] = np.nan
            aligned[name] = placeholder
            continue

        if time_col not in df.columns:
            aligned[name] = df
            continue

        df_copy = df.copy()
        df_copy[id_col] = pd.to_numeric(df_copy[id_col], errors="coerce")
        df_copy[time_col] = pd.to_numeric(df_copy[time_col], errors="coerce")
        df_copy = df_copy.dropna(subset=[id_col, time_col])
        result = grid_copy.merge(df_copy, on=[id_col, time_col], how="left")
        aligned[name] = result

    return aligned


def _should_use_lightweight_alignment(
    grid: pd.DataFrame,
    concept_data: Dict[str, pd.DataFrame],
) -> bool:
    """Prefer low-copy alignment on memory-constrained systems and large grids."""
    try:
        from .runtime.memory_manager import get_available_memory_mb
        available_mb = float(get_available_memory_mb())
    except Exception:
        available_mb = 8 * 1024.0

    time_series_count = sum(
        1 for df in concept_data.values()
        if isinstance(df, pd.DataFrame) and not df.empty and "time" in df.columns
    )
    grid_rows = len(grid)

    if available_mb <= 32 * 1024:
        return True
    if available_mb <= 64 * 1024 and grid_rows >= 1_000_000 and time_series_count >= 4:
        return True
    return False


# ============================================================================
# 主要接口
# ============================================================================

def merge_concepts_r_style(
    concept_data: Dict[str, pd.DataFrame],
    id_col: str = "stay_id",
    time_col: str = "charttime",
    interval_hours: float = 1.0,
    *,
    consume_input: bool = False,
    dense_status_columns: Iterable[str] = (),
    logical_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """以ricu风格合并多个概念数据
    
    实现与R ricu的load_concepts(..., interval=hours(1L))一致的行为：
    1. 构建统一时间网格
    2. 对齐所有概念到网格
    3. 使用outer join合并
    
    Args:
        concept_data: 概念名称到DataFrame的映射
        id_col: ID列名
        time_col: 时间列名
        interval_hours: 时间间隔（小时）
        consume_input: 内部提取热路径可设为 ``True``，逐概念转移所有权并
            清空传入字典，从而避免最终宽表合并时仍保留整套上游帧。公开调用
            默认保留输入，维持既有 API 语义。
        dense_status_columns: Source-bound statuses whose nulls retain record identity.
        logical_columns: Concepts whose declared value kind is logical. Their
            hourly merge is a logical any whatever values one batch holds.
        
    Returns:
        合并后的宽格式DataFrame
    """
    if not concept_data:
        return pd.DataFrame()
    dense_status_columns = frozenset(dense_status_columns)
    logical_columns = frozenset(logical_columns)
    
    # 标准化列名
    # 🚀 优化：不再 .copy()，调用者 _to_r_format_merged_enhanced 已经复制过
    # 列重命名使用 .rename() 返回新对象，不会修改原始 DataFrame
    normalized = {}
    requested_names = list(concept_data)
    # The old implementation materialised a union ``grid`` containing every
    # concept's (id, time) pair, but the optimized merge below never consumed
    # that frame.  On eICU respiratory this retained tens of millions of key
    # rows beside all prepared concept frames and the final wide result.  Keep
    # only the exact boolean needed to select the legacy no-grid branch.
    has_grid_rows = False
    for name in requested_names:
        if consume_input and isinstance(concept_data, MutableMapping):
            df = concept_data.pop(name)
        else:
            df = concept_data[name]
        if df is None or df.empty:
            normalized[name] = pd.DataFrame(columns=["id", "time", name])
            continue
        
        df_copy = df
        
        # 检测和重命名ID列
        id_candidates = [id_col, "stay_id", "subject_id", "patientunitstayid", "admissionid", "patientid"]
        found_id = None
        for cand in id_candidates:
            if cand in df_copy.columns:
                found_id = cand
                break
        
        if found_id and found_id != "id":
            df_copy = df_copy.rename(columns={found_id: "id"})
        
        # 检测和重命名时间列
        # 🔧 FIX: 添加 eICU 的时间列（包括 intakeoutputoffset）和 death 的 deathtime
        # 🔧 FIX: 添加 start 列（区间格式数据的开始时间）
        # 🔧 FIX: 添加 measuredat_minutes（AUMC DuckDB聚合后返回的时间列）
        # 🔧 FIX 2025-01-30: measuredat_minutes 应该在 measuredat 之前，因为 DuckDB 聚合后返回的是 measuredat_minutes
        time_candidates = [time_col, "charttime", "time", "starttime", "start", "index_var", 
                          "datetime", "givenat",  # HiRID time columns
                          "nursingchartoffset", "labresultoffset", "observationoffset",
                          "measuredat_minutes", "measuredat",  # AUMC time columns: measuredat_minutes first!
                          "respchartoffset", "intakeoutputoffset",
                          "infusionoffset", "drugstartoffset", "deathtime",
                          "unitdischargeoffset", "dateofdeath"]
        found_time = None
        for cand in time_candidates:
            if cand in df_copy.columns:
                found_time = cand
                break
        
        if found_time and found_time != "time":
            df_copy = df_copy.rename(columns={found_time: "time"})
        
        # 🔧 FIX: 删除其他可能导致笛卡尔积的额外时间列
        extra_time_cols = ["intakeoutputentryoffset"]
        for col in extra_time_cols:
            if col in df_copy.columns and col != found_time:
                df_copy = df_copy.drop(columns=[col])
        
        # 转换时间为小时数
        if "time" in df_copy.columns and not pd.api.types.is_numeric_dtype(df_copy["time"]):
            df_copy["time"] = time_to_hours(df_copy["time"], df_copy.get("id"))
        
        # 🔧 FIX: 窗口概念不取整时间，保留原始值给 expand_interval_rows 处理
        # R ricu 的 expand() 使用原始浮点时间来计算 seq()
        # 取整将在 expand_interval_rows 内部进行
        # 🔧 FIX 2025-02-13: 同时跳过 WIN_TBL_CONCEPTS，它们保留原始事件时间不取整
        is_window_concept = name in WINDOW_CONCEPTS or name.endswith("_rate")
        is_win_tbl_concept = name in WIN_TBL_CONCEPTS or name in DURATION_CONCEPTS or name.endswith("_dur")
        if "time" in df_copy.columns and not is_window_concept and not is_win_tbl_concept:
            df_copy["time"] = round_to_interval(df_copy["time"], interval_hours)
        
        # Durations are exact elapsed hours. Flooring their start would shift
        # the reconstructed stop and disagree with per-concept loads.
        
        # 确保有值列
        if name not in df_copy.columns:
            value_candidates = ["value", "valuenum", name]
            for cand in value_candidates:
                if cand in df_copy.columns and cand != name:
                    df_copy = df_copy.rename(columns={cand: name})
                    break
        
        # 🔧 FIX: 标准化窗口概念的列名
        # mech_vent 等概念返回 start/stop/{name}_dur，需要重命名为 time/endtime/duration
        if name in WINDOW_CONCEPTS or name.endswith("_rate"):
            # 重命名 start -> time (如果还没有 time 列)
            if "start" in df_copy.columns and "time" not in df_copy.columns:
                df_copy = df_copy.rename(columns={"start": "time"})
            # 重命名 stop -> endtime
            if "stop" in df_copy.columns:
                df_copy = df_copy.rename(columns={"stop": "endtime"})
            # 重命名 {name}_dur -> duration
            dur_col = f"{name}_dur"
            if dur_col in df_copy.columns:
                df_copy = df_copy.rename(columns={dur_col: "duration"})
        
        # 窗口展开
        if name in WINDOW_CONCEPTS or name.endswith("_rate"):
            df_copy = expand_interval_rows(
                df_copy, name, 
                id_col="id", time_col="time", value_col=name,
                interval_hours=interval_hours
            )
        
        if name in dense_status_columns and name in df_copy:
            df_copy = df_copy.assign(**{name: df_copy[name].astype("boolean")})
        normalized[name] = df_copy
        if not has_grid_rows and "id" in df_copy.columns:
            if "time" in df_copy.columns:
                valid_time = df_copy["time"].notna()
                has_grid_rows = bool(
                    valid_time.any()
                    and df_copy.loc[valid_time, "id"].notna().any()
                )
            else:
                has_grid_rows = bool(df_copy["id"].notna().any())

    if not has_grid_rows and not dense_status_columns:
        # 没有时间数据，简单合并
        if len(normalized) == 1:
            name = list(normalized.keys())[0]
            df = list(normalized.values())[0]
            # 重命名列以匹配输出
            if "id" in df.columns and id_col != "id":
                df = df.rename(columns={"id": id_col})
            if "time" in df.columns and time_col != "time":
                df = df.rename(columns={"time": time_col})
            return df
        
        # 多个概念都为空的情况
        all_empty = all(df.empty if df is not None else True for df in normalized.values())
        if all_empty:
            # 返回包含所有概念名的空 DataFrame
            return pd.DataFrame(columns=[id_col, time_col] + list(normalized.keys()))
        
        merged = None
        for name, df in normalized.items():
            if df is None or df.empty:
                continue
            if merged is None:
                merged = df.copy()
            else:
                # 按ID合并，避免重复列
                merge_cols = ["id"] if "id" in merged.columns and "id" in df.columns else []
                if merge_cols:
                    # 只选择需要的列：ID + 概念名
                    cols_to_add = [c for c in df.columns if c not in merged.columns or c in merge_cols]
                    df_subset = df[cols_to_add].copy()
                    merged = merged.merge(df_subset, on=merge_cols, how="outer", suffixes=('', '_dup'))
                    # 删除重复列
                    merged = merged[[c for c in merged.columns if not c.endswith('_dup')]]
                else:
                    # 没有公共ID列，添加概念列
                    if name in df.columns and name not in merged.columns:
                        merged[name] = np.nan
        
        if merged is not None:
            # 重命名列以匹配输出
            if "id" in merged.columns and id_col != "id":
                merged = merged.rename(columns={"id": id_col})
            if "time" in merged.columns and time_col != "time":
                merged = merged.rename(columns={"time": time_col})
            return merged
        
        return pd.DataFrame(columns=[id_col, time_col] + list(normalized.keys()))
    
    # 🚀 优化合并：直接使用 reduce outer merge 替代 grid + align + iterative merge
    # 原路径：build_time_grid (concat+dedup) + align_to_grid (N left joins) + iterative merge (N left joins) = 2N+1 操作
    # 新路径：prepare each concept (dedup) + reduce outer merge (N-1 merges) = N 操作
    # 性能提升：MIIV 6 vitals ~50s → ~8s
    
    normalized_names = list(normalized)
    indexed_frames = []
    boolean_concepts = []
    static_concepts = {}  # name → df (no time column, per-patient)
    
    for name in normalized_names:
        # The production caller transfers ownership of its concept frames to
        # this function.  Pop each normalized frame as soon as it is prepared
        # so pd.concat never overlaps the originals, prepared frames and the
        # final wide frame.  This also helps the non-consuming public path by
        # dropping this function's extra references; the caller's dictionary
        # itself remains untouched unless ``consume_input=True``.
        df = normalized.pop(name)
        if df is None or df.empty:
            continue
        
        if "time" not in df.columns:
            # 静态概念
            if "id" in df.columns and name in df.columns:
                # A source row whose value became missing after a sentinel or
                # bounds transform carries no observation.  Retaining it as a
                # static row manufactures an all-null (id, time=NULL) record
                # when no other concept is available for that stay.
                # A static status owner may also emit the recorded event
                # coordinate as ``<concept>_time``.  It is a value companion,
                # not the row index: hospital-death status must cover every
                # admission even when no death clock was recorded.  Preserve
                # that closed, owner-issued sidecar through the public merge.
                static_columns = ["id", name]
                companion_time = f"{name}_time"
                if companion_time in df.columns:
                    static_columns.append(companion_time)
                static = df[static_columns].dropna(subset=["id"])
                if name in dense_status_columns:
                    static = static.groupby("id", as_index=False).agg(
                        {column: "max" if column == name else "first"
                         for column in static_columns if column != "id"}
                    )
                else:
                    static = static.dropna(subset=[name]).drop_duplicates(subset=["id"], keep="last")
                if not static.empty:
                    static_concepts[name] = static
            continue
        
        keep_cols = ["id", "time"]
        if name in df.columns:
            keep_cols.append(name)
        from ..concept_output_sources import SOURCE_ITEM_OUTPUT_COLUMNS, SPECIMEN_STATUS_OUTPUT_COLUMNS

        keep_cols.extend(SOURCE_ITEM_OUTPUT_COLUMNS.get(name, ()))
        keep_cols.extend(SPECIMEN_STATUS_OUTPUT_COLUMNS.get(name, ()))
        # Some clinical concept owners emit evidence receipts beside the
        # public value.  They are part of that value's meaning, not separate
        # selectable concepts.  The R-style merger previously projected only
        # ``name`` and silently discarded these sidecars before the native-v2
        # Arrow writer could preserve them.  Keep only the closed owner-issued
        # receipt contract here; arbitrary callback helper columns remain
        # excluded.
        keep_cols.extend(
            column
            for column in (
                f"{name}_observed",
                f"{name}_available",
                f"{name}_time",
            )
            if column in df.columns
        )
        if name in {"urine24", "urine24_est", "uo_6h", "uo_12h", "uo_24h", "kdigo_uo", "aki_stage_uo", "kdigo_aki", "aki_stage_uo_reference"}:
            # RU-02: availability belongs to the window value, including its
            # separate full-window and >6h assessments. Preserve it in exports.
            from ..scores.urine_output import urine_availability_columns

            keep_cols.extend(c for c in urine_availability_columns(name) if c not in normalized_names)
        keep_cols = [c for c in keep_cols if c in df.columns]
        if len(keep_cols) <= 2:
            continue
        
        # 布尔概念检测和聚合
        is_boolean_col = False
        if name in df.columns:
            col = df[name]
            # Declared logical concepts, not the first rows of this batch,
            # decide the boolean merge.
            is_boolean_col = col.dtype == bool or col.dtype == 'boolean' or name in logical_columns
        
        if is_boolean_col:
            # ⚡ PERF: 向量化布尔聚合替代逐组 Python 函数
            # bool 列中 True=1, False=0, max() 相当于 any()，且自动跳过 NaN
            from ..concept.value_kind import coerce_declared_values

            _bool_df = df[keep_cols].copy()
            _bool_df[name] = coerce_declared_values(_bool_df[name], "logical").astype('float32')
            prepared = _bool_df.groupby(
                ["id", "time"], as_index=False, dropna=name not in dense_status_columns
            ).agg({name: 'max'})
            # 将 max 结果转回布尔: 1.0→True, 0.0→False, NaN→NaN
            prepared[name] = prepared[name].map({1.0: True, 0.0: False})
            if name in dense_status_columns:
                prepared[name] = prepared[name].astype("boolean")
            boolean_concepts.append(name)
        else:
            prepared = df[keep_cols].drop_duplicates(subset=["id", "time"], keep="last")
            if name == "mech_vent" and name in keep_cols:
                # Point evidence keeps every level of an hour (eICU, R6-F039);
                # publication pools them instead of keeping the last row.
                observed = df.loc[df[name].notna(), ["id", "time", name]]
                if observed.duplicated(["id", "time"]).any():
                    pooled = pool_mech_vent_levels(observed, ["id", "time"], name)
                    prepared = prepared.drop(columns=[name]).merge(
                        pooled, on=["id", "time"], how="left", validate="one_to_one"
                    )[keep_cols]

        # Missing values must not contribute a physical event key.  The outer
        # merge will still place NA at keys supplied by other concepts, but a
        # value-less source row can no longer create an otherwise empty module
        # row on its own. Urine scoring receipts still report incomplete
        # coverage when the descriptive rate is unavailable. Dense statuses
        # retain the source identity, including an unrecorded clock.
        required = ["id"] if name in dense_status_columns else ["id", "time"]
        has_urine_assessment = name in {"uo_6h", "uo_12h", "uo_24h"} and f"{name}_assessment_covered_h" in prepared
        if name not in dense_status_columns and not has_urine_assessment:
            required.append(name)
        prepared = prepared.dropna(subset=required)
        if prepared.empty:
            continue
        
        # Retain only the indexed representation.  Keeping both ``prepared``
        # and ``indexed`` lists until the final concat duplicated the key
        # columns for every concept at the peak-memory boundary.
        indexed_frames.append(prepared.set_index(["id", "time"], drop=True))
    
    if not indexed_frames:
        if not static_concepts:
            return pd.DataFrame(columns=[id_col, time_col] + normalized_names)
        # 仅静态概念：按 id 合并
        from functools import reduce
        static_dfs = list(static_concepts.values())
        if len(static_dfs) == 1:
            merged = static_dfs[0].copy()
        else:
            merged = reduce(lambda left, right: left.merge(right, on="id", how="outer", suffixes=('', '_drop')), static_dfs)
            merged = merged[[c for c in merged.columns if not c.endswith('_drop')]]
        merged = merged.sort_values("id", ignore_index=True)
        merged = merged.rename(columns={"id": id_col})
        return merged
    
    # 🚀 优化合并：pd.concat(axis=1) 替代 N-1 次 reduce outer merge。
    # ``indexed_frames`` 已在上面逐概念构建，避免同时保留两套列表。
    if len(indexed_frames) == 1:
        merged = indexed_frames[0].reset_index()
    else:
        # 🚀 sort=True: 在 concat 过程中排序 (id, time) 索引，
        # 比 concat 后再 sort_values 更快（利用部分有序性）
        merged = pd.concat(indexed_frames, axis=1, join="outer", sort=True, copy=False).reset_index()
    del indexed_frames  # 释放索引帧
    
    # 合并静态概念 (outer join 确保仅有静态数据的患者也被保留)
    for name, sdf in static_concepts.items():
        merged = merged.merge(sdf, on="id", how="outer", suffixes=('', '_drop'))
        merged = merged[[c for c in merged.columns if not c.endswith('_drop')]]
    
    # 🚀 已在 pd.concat(sort=True) 中排序，仅需处理静态概念合并后的新行
    if static_concepts:
        merged = merged.sort_values(["id", "time"], ignore_index=True, na_position="last")
    
    # 重命名列以匹配ricu输出
    merged = merged.rename(columns={"id": id_col, "time": time_col})
    
    return merged


def get_module_concepts(module_name: str) -> List[str]:
    """获取模块中的所有概念"""
    module = COMPAT_MODULES.get(module_name)
    if module:
        return module.concepts
    return []


def find_module_for_concept(concept_name: str) -> Optional[str]:
    """查找概念所属的模块"""
    for module_name, module in COMPAT_MODULES.items():
        if concept_name in module.concepts:
            return module_name
    return None
