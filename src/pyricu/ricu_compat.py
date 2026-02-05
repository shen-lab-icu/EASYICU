"""
ricu兼容层 - 实现与R ricu一致的数据提取行为

该模块提供了与R ricu包load_concepts函数完全一致的数据提取逻辑，包括：
1. 时间网格对齐 - 所有概念对齐到共同的时间网格（默认1小时）
2. 窗口展开 - 将start/end时间窗口展开为逐小时记录
3. 概念合并 - 使用outer join合并多个概念
4. 静态概念填充 - 静态值（age, sex等）填充到所有时间点

用法示例:
    >>> from pyricu import load_concepts
    >>> 
    >>> # 提取生命体征（与ricu.R一致）
    >>> vitals = load_concepts(
    ...     ['hr', 'sbp', 'dbp', 'temp'],
    ...     database='miiv',
    ...     patient_ids=[30041748, 30046525],
    ...     interval='1h',  # 默认值，与ricu的hours(1L)一致
    ...     ricu_compatible=True  # 启用完整的ricu兼容模式
    ... )
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

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
RICU_MODULES: Dict[str, ConceptModule] = {
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
    "norepi_rate", "epi_rate", "dobu_rate", "adh_rate",
    "dopa_rate", "phn_rate", "vaso_ind",
    "dex",  # dex 在 aumc/eicu 有 dur_var，需要展开
    "ett_gcs",     # FIX: ett_gcs 使用 ts_to_win_tbl 展开窗口
}

# 点事件概念（不应展开为连续时间序列）
POINT_EVENT_CONCEPTS = {
    "abx", "samp", "cort", "dobu60", "susp_inf", "sep3", "avpu",
    "rrt",  # Renal replacement therapy: uses set_val(TRUE), point events from chartevents + procedureevents
    "vent_end", "vent_start",  # Ventilation events: uses set_val(TRUE), point events
}

# 时长概念（已编码持续时间，不需要展开）
DURATION_CONCEPTS = {
    "norepi_dur", "epi_dur", "dobu_dur", "dopa_dur"
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
    
    将有start/end时间的记录展开为每小时一条记录，与ricu的expand()行为一致。
    
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
    
    # 🔧 注意：R ricu 不对原始 endtime 做 floor
    # 只有在 endtime 不在列中时，R ricu 才会用 re_time(start + dur, interval) 计算
    # 对于 MIIV inputevents，endtime 已经在列中，所以直接使用原始值
    # seq(start, end, step) 会产生所有 <= end 的时间点
    
    # 展开
    records = []
    for idx, (start, end, value, stay_id) in enumerate(
        zip(starts, ends, working.get(value_col), working.get(id_col))
    ):
        if pd.isna(start) or pd.isna(stay_id):
            continue
        if pd.isna(end):
            end = start
        if end < start:
            end = start
        
        span = min(end - start, max_span_hours)
        if span <= 0:
            records.append({id_col: stay_id, time_col: float(math.floor(start)), value_col: value})
            continue
        
        # 🔧 FIX: 使用 R seq(start, end, step) 的行为
        # R 的 seq(17.84, 20, 1) 产生 [17.84, 18.84, 19.84]
        # 然后取 floor 得到 [17, 18, 19]
        # 
        # 实现：从 start 开始，每次加 1，直到超过 end
        time_points = []
        current = start
        while current <= end + 1e-9:  # 加小量避免浮点误差
            time_points.append(math.floor(current))
            current += interval_hours
        
        # 去重（因为 floor 可能产生重复）
        time_points = sorted(set(time_points))
        
        for hour in time_points:
            records.append({id_col: stay_id, time_col: float(hour), value_col: value})
    
    if not records:
        return df.drop(columns=[endtime_col, duration_col], errors="ignore")
    
    expanded = pd.DataFrame.from_records(records)
    
    # 🔧 FIX: 按(id, time)聚合，根据数据类型选择聚合函数
    # 参考: ricu/R/tbl-utils.R 第 741-751 行:
    #   - numeric → median
    #   - logical → sum (或 any)  
    #   - character → first
    value_dtype = expanded[value_col].dtype
    if pd.api.types.is_numeric_dtype(value_dtype):
        agg_func = 'median'
    elif pd.api.types.is_bool_dtype(value_dtype):
        agg_func = 'any'
    else:
        # object/string/category → first
        agg_func = 'first'
    
    expanded = expanded.groupby([id_col, time_col], as_index=False).agg({value_col: agg_func})
    
    return expanded


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
    
    aligned = {}
    grid_copy = grid.copy()
    grid_copy[id_col] = pd.to_numeric(grid_copy[id_col], errors="coerce")
    grid_copy[time_col] = pd.to_numeric(grid_copy[time_col], errors="coerce")
    grid_copy = grid_copy.dropna(subset=[id_col, time_col]).drop_duplicates()
    
    for name, df in concept_data.items():
        if df is None or df.empty:
            # 创建空占位符
            placeholder = grid_copy.copy()
            placeholder[value_col] = np.nan
            aligned[name] = placeholder
            continue
        
        if time_col not in df.columns:
            # 静态概念，不需要时间对齐
            aligned[name] = df
            continue
        
        df_copy = df.copy()
        df_copy[id_col] = pd.to_numeric(df_copy[id_col], errors="coerce")
        df_copy[time_col] = pd.to_numeric(df_copy[time_col], errors="coerce")
        df_copy = df_copy.dropna(subset=[id_col, time_col])
        
        # 左连接到网格
        result = grid_copy.merge(df_copy, on=[id_col, time_col], how="left")
        
        # 静态概念填充 - 使用概念名称作为值列，而不是默认的 value_col
        concept_value_col = name if name in result.columns else value_col
        if name in STATIC_CONCEPTS and concept_value_col in result.columns:
            for patient_id in result[id_col].unique():
                if pd.isna(patient_id):
                    continue
                patient_mask = result[id_col] == patient_id
                patient_values = result.loc[patient_mask, concept_value_col]
                non_na = patient_values.dropna()
                if len(non_na) > 0 and non_na.nunique() == 1:
                    result.loc[patient_mask, concept_value_col] = non_na.iloc[0]
        
        aligned[name] = result
    
    return aligned


# ============================================================================
# 主要接口
# ============================================================================

def merge_concepts_ricu_style(
    concept_data: Dict[str, pd.DataFrame],
    id_col: str = "stay_id",
    time_col: str = "charttime",
    interval_hours: float = 1.0,
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
        
    Returns:
        合并后的宽格式DataFrame
    """
    if not concept_data:
        return pd.DataFrame()
    
    # 标准化列名
    normalized = {}
    for name, df in concept_data.items():
        if df is None or df.empty:
            normalized[name] = pd.DataFrame(columns=["id", "time", name])
            continue
        
        df_copy = df.copy()
        
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
        is_window_concept = name in WINDOW_CONCEPTS or name.endswith("_rate")
        if "time" in df_copy.columns and not is_window_concept:
            df_copy["time"] = round_to_interval(df_copy["time"], interval_hours)
        
        # 🔧 NOTE: Duration 概念的值（如 dobu_dur）已经在 calc_dur 中使用 floor(end_h) - floor(start_h) 计算
        # 不需要再对 duration 值做额外处理
        # R ricu 的 calc_dur 在时间已经被 floor 到小时后计算 max(end) - min(start)
        
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
        
        normalized[name] = df_copy
    
    # 构建时间网格
    grid = build_time_grid(normalized, id_col="id", time_col="time")
    
    if grid is None or grid.empty:
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
    
    # 对齐到网格并合并
    aligned = align_to_grid(normalized, grid, id_col="id", time_col="time")
    
    # 按时间网格合并
    merged = grid.copy()
    boolean_concepts = []  # 跟踪布尔概念，以便后续 fillna(False)
    for name, df in aligned.items():
        if df is None or df.empty:
            merged[name] = np.nan
            continue
        
        if "time" not in df.columns:
            # 静态概念，直接按ID合并
            if "id" in df.columns and name in df.columns:
                static = df[["id", name]].drop_duplicates()
                merged = merged.merge(static, on="id", how="left", suffixes=('', '_drop'))
                # 删除重复列
                merged = merged[[c for c in merged.columns if not c.endswith('_drop')]]
            continue
        
        # 选择需要的列：只保留 id, time, 和概念名列
        keep_cols = ["id", "time"]
        if name in df.columns:
            keep_cols.append(name)
        
        keep_cols = [c for c in keep_cols if c in df.columns]
        if len(keep_cols) <= 2:  # 只有id和time，没有值
            merged[name] = np.nan
            continue
        
        # FIX: 对于布尔型概念（如 ett_gcs），使用 any() 聚合而不是 drop_duplicates(keep="last")
        # 因为同一时间点可能有多个值（TRUE 和 FALSE），应该取 any(TRUE) = TRUE
        # 注意：left join 后 dtype 可能从 bool 变为 object（因为 NaN），需要特殊检测
        is_boolean_col = False
        if name in df.columns:
            col = df[name]
            # 检查 dtype 或者检查非 NA 值是否都是布尔
            if col.dtype == bool or col.dtype == 'boolean':
                is_boolean_col = True
            elif col.dtype == object:
                # 检查非 NA 值是否为布尔型
                non_na = col.dropna()
                if len(non_na) > 0:
                    is_boolean_col = all(isinstance(v, (bool, np.bool_)) for v in non_na.head(100))
        
        if is_boolean_col:
            # 重新聚合：如果同一 (id, time) 有任何 TRUE，则为 TRUE
            # 但要保留全 NA 组为 NA（不转为 FALSE）
            def bool_agg_with_na(x):
                """布尔聚合，保留全 NA 为 NA"""
                non_na = x.dropna()
                if len(non_na) == 0:
                    return np.nan
                return non_na.any()
            
            to_merge = df[keep_cols].groupby(["id", "time"], as_index=False).agg({name: bool_agg_with_na})
            boolean_concepts.append(name)  # 记录布尔概念
        else:
            to_merge = df[keep_cols].drop_duplicates(subset=["id", "time"], keep="last")
        
        merged = merged.merge(to_merge, on=["id", "time"], how="left", suffixes=('', '_drop'))
        # 删除重复列
        merged = merged[[c for c in merged.columns if not c.endswith('_drop')]]
    
    # 重命名列以匹配ricu输出
    merged = merged.rename(columns={"id": id_col, "time": time_col})
    
    return merged


def get_module_concepts(module_name: str) -> List[str]:
    """获取模块中的所有概念"""
    module = RICU_MODULES.get(module_name)
    if module:
        return module.concepts
    return []


def find_module_for_concept(concept_name: str) -> Optional[str]:
    """查找概念所属的模块"""
    for module_name, module in RICU_MODULES.items():
        if concept_name in module.concepts:
            return module_name
    return None
