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
    
    # Handle cases where x was NaN
    result[x.isna()] = np.nan
    
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
    si_mode: Literal["and", "or", "abx", "samp"] = "and",
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
    
    Simple iterrows implementation (correctness-focused)
    """
    if abx.empty or samp.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'susp_inf'])
    
    # Determine time type
    time_is_numeric = pd.api.types.is_numeric_dtype(abx[index_col])
    
    if not time_is_numeric:
        # 🚀 优化：仅在需要时copy（datetime转换可能已在上游完成）
        if not pd.api.types.is_datetime64_any_dtype(abx[index_col]):
            abx = abx.copy()
            abx[index_col] = pd.to_datetime(abx[index_col], errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(samp[index_col]):
            samp = samp.copy()
            samp[index_col] = pd.to_datetime(samp[index_col], errors='coerce')
        abx_win_val = abx_win
        samp_win_val = samp_win
    else:
        abx_win_val = abx_win.total_seconds() / 3600.0
        samp_win_val = samp_win.total_seconds() / 3600.0
    
    results = []
    
    # Method 1: ABX → sampling
    for _, abx_row in abx.iterrows():
        abx_time = abx_row[index_col]
        if pd.isna(abx_time):
            continue
        
        samp_mask = pd.Series(True, index=samp.index)
        for col in id_cols:
            samp_mask &= (samp[col] == abx_row[col])
        
        samp_subset = samp[samp_mask]
        if samp_subset.empty:
            continue
        
        if time_is_numeric:
            samp_in_win = samp_subset[
                (samp_subset[index_col] >= abx_time) &
                (samp_subset[index_col] < abx_time + abx_win_val)
            ]
        else:
            samp_in_win = samp_subset[
                (samp_subset[index_col] >= abx_time) &
                (samp_subset[index_col] < abx_time + abx_win_val)
            ]
        
        if not samp_in_win.empty:
            result_row = {col: abx_row[col] for col in id_cols}
            result_row[index_col] = abx_time
            
            if keep_components:
                result_row['abx_time'] = abx_time
                result_row['samp_time'] = samp_in_win.iloc[0][index_col]
            
            results.append(result_row)
    
    # Method 2: Sampling → ABX
    for _, samp_row in samp.iterrows():
        samp_time = samp_row[index_col]
        if pd.isna(samp_time):
            continue
        
        abx_mask = pd.Series(True, index=abx.index)
        for col in id_cols:
            abx_mask &= (abx[col] == samp_row[col])
        
        abx_subset = abx[abx_mask]
        if abx_subset.empty:
            continue
        
        if time_is_numeric:
            abx_in_win = abx_subset[
                (abx_subset[index_col] >= samp_time) &
                (abx_subset[index_col] < samp_time + samp_win_val)
            ]
        else:
            abx_in_win = abx_subset[
                (abx_subset[index_col] >= samp_time) &
                (abx_subset[index_col] < samp_time + samp_win_val)
            ]
        
        if not abx_in_win.empty:
            result_row = {col: samp_row[col] for col in id_cols}
            result_row[index_col] = samp_time
            
            if keep_components:
                result_row['abx_time'] = abx_in_win.iloc[0][index_col]
                result_row['samp_time'] = samp_time
            
            results.append(result_row)
    
    if not results:
        return pd.DataFrame(columns=id_cols + [index_col, 'susp_inf'])
    
    result_df = pd.DataFrame(results)
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
    # Always use merge to match R ricu behavior
    merge_cols = id_cols + [index_col]
    
    # Prepare data with flags
    abx_prep = abx[merge_cols].copy()
    abx_prep['_abx_flag'] = True
    
    samp_prep = samp[merge_cols].copy()
    samp_prep['_samp_flag'] = True
    
    # Outer merge (like R's merge(..., all = TRUE))
    result = pd.merge(abx_prep, samp_prep, on=merge_cols, how='outer')
    
    # Keep rows where abx OR samp occurred
    result['_abx_flag'] = result['_abx_flag'].fillna(False)
    result['_samp_flag'] = result['_samp_flag'].fillna(False)
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
    # Filter SI events where susp_inf == TRUE
    if 'susp_inf' in susp_inf.columns:
        si_events = susp_inf[susp_inf['susp_inf'] == True].copy()
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
    sofa_prep = sofa_prep.sort_values(id_cols + [index_col])
    sofa_prep['_delta_sofa'] = sofa_prep.groupby(id_cols)['sofa'].transform(delta_fun)
    
    # Merge SI events with SOFA on id_cols
    merged = si_events.merge(sofa_prep, on=id_cols, how='inner', suffixes=('_si', '_sofa'))
    
    if merged.empty:
        return pd.DataFrame(columns=id_cols + [index_col, 'sep3'])
    
    # Filter by time window: _si_lwr <= _sofa_time <= _si_upr
    time_col_sofa = f'{index_col}_sofa' if f'{index_col}_sofa' in merged.columns else '_sofa_time'
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
    delta_sofa: int = 2,
    si_mode: str = "abx_ind",
    keep_components: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Sepsis-3 标注 - 别名函数，调用 sep3()
    
    这是 R ricu label_sep3 的 Python 实现。
    
    Args:
        sofa_data: SOFA 评分数据
        susp_inf_data: 疑似感染数据
        delta_sofa: SOFA 评分增量阈值（默认 2）
        si_mode: 疑似感染模式
        keep_components: 是否保留组件
        **kwargs: 其他参数
        
    Returns:
        Sepsis-3 标注结果
        
    Examples:
        >>> # 使用 SOFA 和疑似感染数据
        >>> sep3_labels = label_sep3(sofa_df, si_df)
        >>> 
        >>> # 自定义阈值
        >>> sep3_labels = label_sep3(sofa_df, si_df, delta_sofa=3)
    """
    return sep3(
        sofa_data=sofa_data,
        susp_inf_data=susp_inf_data,
        delta_sofa=delta_sofa,
        si_mode=si_mode,
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
