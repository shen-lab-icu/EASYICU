"""Sepsis-3 and suspected infection detection.

This module implements the Sepsis-3 criteria from Singer et al. (2016)
and suspected infection (SI) detection, following R ricu's implementation.

References:
    Singer M, Deutschman CS, Seymour CW, et al. The Third International
    Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA.
    2016;315(8):801–810. doi:10.1001/jama.2016.0287
"""

from typing import Optional, Callable, Literal, List
import pandas as pd
import numpy as np


def delta_cummin(x: pd.Series) -> pd.Series:
    """Calculate delta from cumulative minimum (R ricu delta_cummin).
    
    For Sepsis-3, this represents the increase in SOFA score from the
    minimum value seen up to the current time point.
    
    This is the recommended default for Sepsis-3 detection as it captures
    the maximum increase from any prior low point.
    
    Args:
        x: SOFA score series
        
    Returns:
        Delta from cumulative minimum
        
    Examples:
        >>> sofa = pd.Series([2, 1, 4, 3, 5])
        >>> delta_cummin(sofa)
        # Returns: [0, 0, 3, 2, 4]  # Increase from cumulative minimum
    """
    # Use integer.max instead of inf to match R ricu
    # R ricu uses .Machine$integer.max (2^31-1 = 2147483647)
    # This ensures exact compatibility with R ricu's behavior
    integer_max = 2147483647  # .Machine$integer.max in R
    x_filled = x.fillna(integer_max)
    cummin = x_filled.cummin()
    
    # Calculate delta
    result = x - cummin
    
    # Handle cases where x was NaN - 使用 numpy 向量化操作避免 __repr__ 开销
    na_mask = x.isna()
    if na_mask.any():
        result = result.where(~na_mask, np.nan)
    
    return result

def delta_start(x: pd.Series) -> pd.Series:
    """Calculate delta from start value (R ricu delta_start).
    
    Represents SOFA score increase from the first non-NA measurement.
    
    Args:
        x: SOFA score series
        
    Returns:
        Delta from first value
        
    Examples:
        >>> sofa = pd.Series([2, 1, 4, 3, 5])
        >>> delta_start(sofa)
        # Returns: [0, -1, 2, 1, 3]  # Increase from first value (2)
    """
    # Match R ricu behavior - return NaN if all values are NA
    non_na = x.dropna()
    if len(non_na) == 0:
        return pd.Series([np.nan] * len(x), index=x.index)
    first_val = non_na.iloc[0]
    return x - first_val

def delta_min(x: pd.Series, shifts: Optional[List[int]] = None) -> pd.Series:
    """Calculate delta from minimum over shifted windows (R ricu delta_min).
    
    Represents SOFA score increase from the minimum value in a
    sliding window. Default window is previous 24 hours (shifts 0-23).
    
    Args:
        x: SOFA score series (hourly resolution expected)
        shifts: List of shift amounts in hours (default: 0-23 for 24-hour window)
        
    Returns:
        Delta from windowed minimum
        
    Examples:
        >>> # Hourly SOFA scores
        >>> sofa = pd.Series([2, 3, 1, 4, 2, 5])
        >>> delta_min(sofa, shifts=[0, 1, 2])  # 3-hour window
        # Returns minimum over current + 2 prior hours for each time point
    """
    if shifts is None:
        shifts = list(range(24))  # Default: 24-hour window
    
    if len(x) == 0:
        return x
    
    # Calculate minimum across all shifts
    shifted_vals = [x.shift(s) for s in shifts]
    
    if not shifted_vals:
        return x - x
    
    # Stack and find minimum
    stacked = pd.concat(shifted_vals, axis=1)
    windowed_min = stacked.min(axis=1, skipna=True)
    
    return x - windowed_min

def susp_inf(
    abx: pd.DataFrame,
    samp: pd.DataFrame,
    id_cols: list,
    index_col: str,
    abx_count_win: pd.Timedelta = pd.Timedelta(hours=24),
    abx_min_count: int = 1,
    positive_cultures: bool = False,
    si_mode: Literal["and", "or", "abx", "samp", "icd_abx"] = "and",
    abx_win: pd.Timedelta = pd.Timedelta(hours=24),
    samp_win: pd.Timedelta = pd.Timedelta(hours=72),
    keep_components: bool = False,
) -> pd.DataFrame:
    """Detect suspected infection (R ricu susp_inf).
    
    Suspected infection is defined as co-occurrence of antibiotic treatment
    and body-fluid sampling within specified time windows.
    
    Implementation follows R ricu's susp_inf function:
    1. Process antibiotics with si_abx():
       - Count antibiotics in rolling window (abx_count_win)
       - Filter by minimum count (abx_min_count)
    2. Process samples with si_samp():
       - Aggregate sampling events
       - Optionally filter for positive cultures
    3. Combine using si_mode:
       - "and": Both ABX and sampling required (si_and)
       - "or": Either ABX or sampling (si_or)
       - "abx": Only ABX required
       - "samp": Only sampling required
       - "icd_abx": ICD infection diagnosis + ABX (eICU新策略，在callback中处理)
    
    Time window logic (si_mode="and"):
    - ABX followed by sampling: sampling within [abx_time, abx_time + abx_win)
    - Sampling followed by ABX: ABX within [samp_time, samp_time + samp_win)
    
    Args:
        abx: Antibiotic data (must have id_cols, index_col, 'abx' column)
        samp: Sampling data (must have id_cols, index_col, 'samp' column)
        id_cols: ID columns for merging
        index_col: Time index column
        abx_count_win: Window for counting antibiotic administrations
        abx_min_count: Minimum antibiotic administrations required
        positive_cultures: Whether to require positive cultures
        si_mode: Detection mode ('and', 'or', 'abx', 'samp')
        abx_win: Time window after ABX for sampling (default 24h)
        samp_win: Time window after sampling for ABX (default 72h)
        keep_components: Whether to keep individual component times
        
    Returns:
        DataFrame with suspected infection events
    """
    # Process antibiotic data (si_abx in R ricu)
    abx_processed = _process_abx(abx, id_cols, index_col, abx_count_win, abx_min_count)
    
    # Process sampling data (si_samp in R ricu)
    samp_processed = _process_samp(samp, positive_cultures)
    
    # Combine based on mode
    if si_mode == "and":
        result = _si_and(abx_processed, samp_processed, id_cols, index_col,
                        abx_win, samp_win, keep_components)
    elif si_mode == "or":
        result = _si_or(abx_processed, samp_processed, id_cols, index_col,
                       keep_components)
    elif si_mode == "abx":
        result = abx_processed.copy()
        result['susp_inf'] = True
    elif si_mode == "samp":
        result = samp_processed.copy()
        result['susp_inf'] = True
    elif si_mode == "icd_abx":
        # icd_abx模式在callback中处理，这里只作为fallback
        # 如果直接调用susp_inf函数且模式为icd_abx，则使用abx作为时间点
        result = abx_processed.copy()
        result['susp_inf'] = True
    else:
        raise ValueError(f"Unknown si_mode: {si_mode}")
    
    return result

def _process_abx(
    abx: pd.DataFrame,
    id_cols: list,
    index_col: str,
    count_win: pd.Timedelta,
    min_count: int,
) -> pd.DataFrame:
    """Process antibiotic data for SI detection."""
    if abx.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'abx'])
    
    # 确保abx列存在
    if 'abx' not in abx.columns:
        # 如果abx列不存在，创建它（假设所有行都是abx事件）
        abx = abx.copy()
        abx['abx'] = True
    
    if min_count > 1:
        # Count antibiotics in rolling window
        from .ts_utils import slide
        abx = slide(
            abx, id_cols, index_col,
            before=pd.Timedelta(0),
            after=count_win,
            agg_func={'abx': 'sum'}
        )
    
    # Filter by minimum count
    abx = abx[abx['abx'] >= min_count].copy()
    return abx

def _process_samp(samp: pd.DataFrame, positive_only: bool) -> pd.DataFrame:
    """Process sampling data for SI detection."""
    if samp.empty:
        return pd.DataFrame()
    
    # 确保samp列存在
    if 'samp' not in samp.columns:
        # 如果samp列不存在，创建它（假设所有行都是采样事件）
        samp = samp.copy()
        samp['samp'] = True
    
    if positive_only:
        # Require positive cultures (samp > 0)
        samp = samp[samp['samp'] > 0].copy()
    else:
        # Just require any sampling (non-NA)
        samp = samp[samp['samp'].notna()].copy()
    
    return samp

def _si_and(
    abx: pd.DataFrame,
    samp: pd.DataFrame,
    id_cols: list,
    index_col: str,
    abx_win: pd.Timedelta,
    samp_win: pd.Timedelta,
    keep_components: bool,
) -> pd.DataFrame:
    """Detect SI when both antibiotic AND sampling occur.

    Vectorized merge implementation: O(n log n) instead of O(n²).

    Time unit contract: when ``index_col`` is numeric, it is assumed to be in
    HOURS since ICU admission (the post-``_align_time_to_admission`` invariant
    used everywhere else in EasyICU, including ``sep3_sofa2``). The 24h/72h
    windows are converted to hours accordingly.
    """
    if abx.empty or samp.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'susp_inf'])
    
    # Deduplicate columns (MIMIC-III samp can have duplicate 'charttime')
    def _dedup_cols(df, keep_set):
        seen = set()
        keep_idx = []
        for i, col in enumerate(df.columns):
            if col in keep_set and col not in seen:
                keep_idx.append(i)
                seen.add(col)
        return df.iloc[:, keep_idx].copy()
    
    abx_needed = set(id_cols + [index_col, 'abx', 'susp_inf']) & set(abx.columns)
    abx = _dedup_cols(abx, abx_needed)
    
    samp_needed = set(id_cols + [index_col, 'samp', 'org_itemid']) & set(samp.columns)
    samp = _dedup_cols(samp, samp_needed)
    
    # Determine time type and window values
    time_is_numeric = pd.api.types.is_numeric_dtype(abx[index_col])

    if not time_is_numeric:
        if not pd.api.types.is_datetime64_any_dtype(abx[index_col]):
            abx = abx.copy()
            abx[index_col] = pd.to_datetime(abx[index_col], errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(samp[index_col]):
            samp = samp.copy()
            samp[index_col] = pd.to_datetime(samp[index_col], errors='coerce')
        abx_win_val = abx_win
        samp_win_val = samp_win
    else:
        # Numeric time axis is in hours (post _align_time_to_admission).
        abx_win_val = abx_win.total_seconds() / 3600.0
        samp_win_val = samp_win.total_seconds() / 3600.0
    
    # Drop rows with NA times
    abx = abx.dropna(subset=[index_col])
    samp = samp.dropna(subset=[index_col])
    
    if abx.empty or samp.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'susp_inf'])
    
    # ⚡ Vectorized: merge on id_cols, then filter by time window
    abx_slim = abx[id_cols + [index_col]].copy()
    abx_slim = abx_slim.rename(columns={index_col: '_abx_time'})
    samp_slim = samp[id_cols + [index_col]].copy()
    samp_slim = samp_slim.rename(columns={index_col: '_samp_time'})
    
    # Inner merge on patient IDs — O(n) with hash join
    merged = abx_slim.merge(samp_slim, on=id_cols, how='inner')
    
    if merged.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'susp_inf'])
    
    # Method 1: ABX → sampling (samp_time in [abx_time, abx_time + abx_win))
    m1 = merged[(merged['_samp_time'] >= merged['_abx_time']) &
                (merged['_samp_time'] < merged['_abx_time'] + abx_win_val)]
    
    # Method 2: Sampling → ABX (abx_time in [samp_time, samp_time + samp_win))
    m2 = merged[(merged['_abx_time'] >= merged['_samp_time']) &
                (merged['_abx_time'] < merged['_samp_time'] + samp_win_val)]
    
    parts = []
    if not m1.empty:
        r1 = m1[id_cols].copy()
        r1[index_col] = m1['_abx_time'].values
        if keep_components:
            r1['abx_time'] = m1['_abx_time'].values
            r1['samp_time'] = m1['_samp_time'].values
        parts.append(r1)
    
    if not m2.empty:
        r2 = m2[id_cols].copy()
        r2[index_col] = m2['_samp_time'].values
        if keep_components:
            r2['abx_time'] = m2['_abx_time'].values
            r2['samp_time'] = m2['_samp_time'].values
        parts.append(r2)
    
    if not parts:
        return pd.DataFrame(columns=id_cols + [index_col, 'susp_inf'])
    
    result_df = pd.concat(parts, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=id_cols + [index_col])
    result_df['susp_inf'] = True
    return result_df

def _si_or(
    abx: pd.DataFrame,
    samp: pd.DataFrame,
    id_cols: list,
    index_col: str,
    keep_components: bool,
) -> pd.DataFrame:
    """Detect SI when either antibiotic OR sampling occurs.
    
    Following R ricu's si_or logic:
    - Merge abx and samp with outer join
    - Keep rows where abx OR samp is TRUE
    """
    merge_cols = id_cols + [index_col]
    
    # Deduplicate columns (same as _si_and for MIMIC-III compatibility)
    def _dedup_cols(df, keep_set):
        seen = set()
        keep_idx = []
        for i, col in enumerate(df.columns):
            if col in keep_set and col not in seen:
                keep_idx.append(i)
                seen.add(col)
        return df.iloc[:, keep_idx].copy()
    
    if not abx.empty and abx.columns.duplicated().any():
        abx = _dedup_cols(abx, set(merge_cols + ['abx', 'susp_inf']) & set(abx.columns))
    if not samp.empty and samp.columns.duplicated().any():
        samp = _dedup_cols(samp, set(merge_cols + ['samp', 'org_itemid']) & set(samp.columns))
    
    # Handle empty DataFrames
    abx_empty = abx.empty or not all(c in abx.columns for c in merge_cols)
    samp_empty = samp.empty or not all(c in samp.columns for c in merge_cols)
    
    if abx_empty and samp_empty:
        # Both empty
        return pd.DataFrame(columns=merge_cols + ['susp_inf'])
    
    if abx_empty:
        # Only samp data
        result = samp[merge_cols].copy()
        result['susp_inf'] = True
        if keep_components:
            result['samp_time'] = result[index_col]
            result['abx_time'] = pd.NaT
        return result
    
    if samp_empty:
        # Only abx data
        result = abx[merge_cols].copy()
        result['susp_inf'] = True
        if keep_components:
            result['abx_time'] = result[index_col]
            result['samp_time'] = pd.NaT
        return result
    
    # Both have data - do outer merge
    abx_prep = abx[merge_cols].copy()
    abx_prep['_abx_flag'] = True
    
    samp_prep = samp[merge_cols].copy()
    samp_prep['_samp_flag'] = True
    
    # Outer merge (like R's merge(..., all = TRUE))
    result = pd.merge(abx_prep, samp_prep, on=merge_cols, how='outer')
    
    # Keep rows where abx OR samp occurred
    result['_abx_flag'] = result['_abx_flag'].fillna(False).infer_objects(copy=False)
    result['_samp_flag'] = result['_samp_flag'].fillna(False).infer_objects(copy=False)
    result = result[result['_abx_flag'] | result['_samp_flag']].copy()
    
    # Add component times if requested
    if keep_components:
        result['abx_time'] = result[index_col].where(result['_abx_flag'])
        result['samp_time'] = result[index_col].where(result['_samp_flag'])
    
    # Clean up flags
    result = result.drop(columns=['_abx_flag', '_samp_flag'])
    result['susp_inf'] = True
    
    return result

def sep3(
    sofa: pd.DataFrame,
    susp_inf: pd.DataFrame,
    id_cols: list,
    index_col: str,
    si_window: Literal["first", "last", "any"] = "first",
    delta_fun: Callable = delta_cummin,
    sofa_thresh: int = 2,
    si_lwr: pd.Timedelta = pd.Timedelta(hours=48),
    si_upr: pd.Timedelta = pd.Timedelta(hours=24),
    keep_components: bool = False,
) -> pd.DataFrame:
    """Detect Sepsis-3 (R ricu sep3) - 向量化优化版本.
    
    Sepsis-3 is defined as a ≥2 point increase in SOFA score within
    the suspected infection window.
    
    Args:
        sofa: SOFA score data (must have 'sofa' column)
        susp_inf: Suspected infection data
        id_cols: ID columns
        index_col: Time index column
        si_window: Which SI window to use ('first', 'last', 'any')
        delta_fun: Function to calculate SOFA delta
        sofa_thresh: Required SOFA increase (default 2)
        si_lwr: Time before SI (default 48h)
        si_upr: Time after SI (default 24h)
        keep_components: Whether to keep delta_sofa, samp_time, abx_time
        
    Returns:
        DataFrame with Sepsis-3 events
    """
    # Filter SI events where susp_inf == TRUE (or 1.0 for numeric columns)
    if 'susp_inf' in susp_inf.columns:
        # Handle both boolean and numeric (1.0/0.0) susp_inf columns
        susp_inf_col = susp_inf['susp_inf'].fillna(0)
        if pd.api.types.is_numeric_dtype(susp_inf_col):
            si_events = susp_inf[susp_inf_col > 0].copy()
        else:
            si_events = susp_inf[susp_inf_col.astype(bool)].copy()
    else:
        si_events = susp_inf.copy()
    
    if si_events.empty or sofa.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'sep3'])
    
    # Determine if time is numeric (hours) or datetime
    si_time_is_numeric = pd.api.types.is_numeric_dtype(si_events[index_col])
    
    # Convert Timedelta to hours if time is numeric
    if si_time_is_numeric:
        si_lwr_val = si_lwr.total_seconds() / 3600.0
        si_upr_val = si_upr.total_seconds() / 3600.0
    else:
        si_lwr_val = si_lwr
        si_upr_val = si_upr
        # Ensure time columns are datetime
        if index_col in si_events.columns:
            si_events[index_col] = pd.to_datetime(si_events[index_col], errors='coerce')
        if index_col in sofa.columns:
            sofa = sofa.copy()
            sofa[index_col] = pd.to_datetime(sofa[index_col], errors='coerce')
    
    # Apply si_window filter: "first", "last", or "any"
    if si_window in ["first", "last"]:
        if si_window == "first":
            si_events = si_events.sort_values(index_col).groupby(id_cols, as_index=False).first()
        else:
            si_events = si_events.sort_values(index_col).groupby(id_cols, as_index=False).last()
    
    # Calculate SI windows
    si_events = si_events.copy()
    si_events['_si_lwr'] = si_events[index_col] - si_lwr_val
    si_events['_si_upr'] = si_events[index_col] + si_upr_val
    si_events['_si_time'] = si_events[index_col]  # 保存原始 SI 时间
    
    # 准备 SOFA 数据
    sofa_prep = sofa.copy()
    sofa_prep['_sofa_time'] = sofa_prep[index_col]
    
    # ========== 向量化 merge 替代 iterrows() ==========
    # 使用 cross join + filter 的方式，对于中等数据集效率更高
    
    # 首先按 id_cols 分组计算 delta_sofa
    # 优化：对于 delta_cummin，直接用向量化操作避免 transform 的开销
    sofa_prep = sofa_prep.sort_values(id_cols + [index_col])
    
    if delta_fun is delta_cummin or delta_fun.__name__ == 'delta_cummin':
        # 向量化计算 delta_cummin: x - cummin(x) per group
        # 使用 expanding().min() 代替 cummin() 可以利用分组
        integer_max = 2147483647
        sofa_filled = sofa_prep['sofa'].fillna(integer_max)
        # 计算每组内的 cummin
        cummin_vals = sofa_filled.groupby([sofa_prep[c] for c in id_cols], sort=False).cummin()
        sofa_prep['_delta_sofa'] = sofa_prep['sofa'] - cummin_vals
        # NaN 的位置保持 NaN
        sofa_prep.loc[sofa_prep['sofa'].isna(), '_delta_sofa'] = np.nan
    else:
        # 其他 delta 函数使用 transform
        sofa_prep['_delta_sofa'] = sofa_prep.groupby(id_cols)['sofa'].transform(delta_fun)
    
    # Merge SI events with SOFA on id_cols
    merged = si_events.merge(sofa_prep, on=id_cols, how='inner', suffixes=('_si', '_sofa'))
    
    if merged.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'sep3'])
    
    # Filter by time window: _si_lwr <= _sofa_time <= _si_upr
    in_window = (merged['_sofa_time'] >= merged['_si_lwr']) & (merged['_sofa_time'] <= merged['_si_upr'])
    merged = merged[in_window]
    
    if merged.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'sep3'])
    
    # Filter by threshold
    meets_thresh = merged['_delta_sofa'] >= sofa_thresh
    sep3_events = merged[meets_thresh].copy()
    
    if sep3_events.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'sep3'])
    
    # Take first occurrence per patient (earliest SOFA time meeting criteria)
    sep3_events = sep3_events.sort_values('_sofa_time')
    first_sep3 = sep3_events.groupby(id_cols, as_index=False).first()
    
    # Build result
    result = first_sep3[id_cols].copy()
    result[index_col] = first_sep3['_sofa_time']
    result['sep3'] = True
    
    if keep_components:
        result['delta_sofa'] = first_sep3['_delta_sofa']
        if 'samp_time' in first_sep3.columns:
            result['samp_time'] = first_sep3['samp_time']
        if 'abx_time' in first_sep3.columns:
            result['abx_time'] = first_sep3['abx_time']
    
    return result.reset_index(drop=True)

# 别名函数 - 为了兼容性
def label_sep3(
    sofa_data: pd.DataFrame,
    susp_inf_data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    delta_sofa: int = 2,
    keep_components: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Sepsis-3 标注 - 别名函数，调用 sep3()
    
    这是 R ricu label_sep3 的 Python 实现。
    
    Args:
        sofa_data: SOFA 评分数据
        susp_inf_data: 疑似感染数据
        id_cols: 患者 ID 列
        index_col: 时间索引列
        delta_sofa: SOFA 评分增量阈值（默认 2）
        keep_components: 是否保留组件
        **kwargs: 其他参数
        
    Returns:
        Sepsis-3 标注结果
        
    Examples:
        >>> sep3_labels = label_sep3(sofa_df, si_df, ['stay_id'], 'charttime')
        >>> sep3_labels = label_sep3(sofa_df, si_df, ['stay_id'], 'charttime', delta_sofa=3)
    """
    return sep3(
        sofa=sofa_data,
        susp_inf=susp_inf_data,
        id_cols=id_cols,
        index_col=index_col,
        sofa_thresh=delta_sofa,
        keep_components=keep_components,
        **kwargs
    )

def _prepare_series(df: pd.DataFrame, required_cols: List[str], label: str) -> pd.DataFrame:
    """Ensure required columns exist and return a copy containing them."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{label} 缺少必要列: {missing}")
    return df[required_cols].copy()

def compute_sepsis3_onset(
    sofa_df: pd.DataFrame,
    susp_inf_df: pd.DataFrame,
    *,
    id_col: str,
    sofa_time_col: str,
    si_time_col: str,
    sofa_score_col: str = 'sofa',
    sofa_system: str = 'SOFA-1',
    delta_fun: Callable = delta_cummin,
    sofa_thresh: int = 2,
    si_window: Literal['first', 'last', 'any'] = 'first',
    si_lwr: pd.Timedelta = pd.Timedelta(hours=48),
    si_upr: pd.Timedelta = pd.Timedelta(hours=24),
) -> pd.DataFrame:
    """Compute Sepsis-3 onset time for a specific SOFA system."""

    if sofa_df.empty or susp_inf_df.empty:
        return pd.DataFrame(columns=[id_col, 'onset_time', 'delta_sofa', 'sofa_system'])

    sofa_required = [id_col, sofa_time_col, sofa_score_col]
    si_required = [id_col, si_time_col]
    sofa_ready = _prepare_series(sofa_df, sofa_required, f"SOFA数据[{sofa_system}]")
    susp_ready = _prepare_series(susp_inf_df, si_required, "疑似感染数据")

    # 确保susp_inf列存在
    if 'susp_inf' in susp_inf_df.columns:
        susp_ready['susp_inf'] = susp_inf_df['susp_inf'].values
        susp_ready['susp_inf'] = susp_ready['susp_inf'].fillna(True)
    else:
        susp_ready['susp_inf'] = True

    sofa_norm = sofa_ready.rename(columns={
        id_col: '_id',
        sofa_time_col: '_time',
        sofa_score_col: 'sofa'
    })
    susp_norm = susp_ready.rename(columns={
        id_col: '_id',
        si_time_col: '_time'
    })

    result = sep3(
        sofa=sofa_norm,
        susp_inf=susp_norm,
        id_cols=['_id'],
        index_col='_time',
        si_window=si_window,
        delta_fun=delta_fun,
        sofa_thresh=sofa_thresh,
        si_lwr=si_lwr,
        si_upr=si_upr,
        keep_components=True
    )

    if result.empty:
        return pd.DataFrame(columns=[id_col, 'onset_time', 'delta_sofa', 'sofa_system'])

    renamed = result.rename(columns={'_id': id_col, '_time': 'onset_time'})
    if 'delta_sofa' not in renamed.columns:
        renamed['delta_sofa'] = sofa_thresh
    renamed['sofa_system'] = sofa_system
    columns = [id_col, 'onset_time', 'delta_sofa', 'sofa_system']
    if 'sep3' in renamed.columns:
        columns.append('sep3')
    else:
        renamed['sep3'] = True
        columns.append('sep3')

    return renamed[columns]

def compare_sepsis_onsets(
    sofa1_onset: pd.DataFrame,
    sofa2_onset: pd.DataFrame,
    id_col: str,
    tolerance_hours: float = 1.0,
) -> pd.DataFrame:
    """Compare Sepsis-3 onset times between SOFA-1 and SOFA-2."""

    if sofa1_onset.empty and sofa2_onset.empty:
        return pd.DataFrame(columns=[id_col, 'onset_time_sofa1', 'onset_time_sofa2', 'time_diff_hours', 'agreement'])

    def _prepare(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[id_col, f'onset_time_{suffix}'])
        return df[[id_col, 'onset_time']].drop_duplicates().rename(columns={'onset_time': f'onset_time_{suffix}'})

    s1 = _prepare(sofa1_onset, 'sofa1')
    s2 = _prepare(sofa2_onset, 'sofa2')

    merged = pd.merge(s1, s2, on=id_col, how='outer')

    def _calc_diff(row):
        t1 = row.get('onset_time_sofa1')
        t2 = row.get('onset_time_sofa2')
        if pd.isna(t1) or pd.isna(t2):
            return np.nan
        if isinstance(t1, pd.Timestamp) and isinstance(t2, pd.Timestamp):
            delta = (t2 - t1) / pd.Timedelta(hours=1)
            return float(delta)
        try:
            return float(t2) - float(t1)
        except Exception:
            return np.nan

    merged['time_diff_hours'] = merged.apply(_calc_diff, axis=1)
    merged['agreement'] = merged['time_diff_hours'].abs() <= tolerance_hours

    def _earlier(row):
        if pd.isna(row.get('onset_time_sofa1')) and pd.isna(row.get('onset_time_sofa2')):
            return 'unknown'
        if pd.isna(row.get('onset_time_sofa1')):
            return 'SOFA-2'
        if pd.isna(row.get('onset_time_sofa2')):
            return 'SOFA-1'
        diff = row.get('time_diff_hours')
        if pd.isna(diff):
            return 'unknown'
        if diff < 0:
            return 'SOFA-2 earlier'
        elif diff > 0:
            return 'SOFA-1 earlier'
        return 'same'

    merged['earlier_onset'] = merged.apply(_earlier, axis=1)

    return merged
