"""Sepsis detection based on SOFA-2 (2025 consensus).

This module mirrors sepsis.py but uses SOFA-2 scores instead of SOFA-1.
It provides a sep3_sofa2() function for Sepsis-3 detection using SOFA-2 scoring.

The suspected infection (SI) detection logic is reused from pyricu.sepsis.susp_inf.

Sepsis-3 (using SOFA-2) is defined as a ≥2 point increase in SOFA-2 score
within the suspected infection window.

References:
    Moreno et al. (2025). SOFA-2 Consensus Statement. JAMA Network Open.
    Singer M, Deutschman CS, Seymour CW, et al. The Third International
    Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA.
    2016;315(8):801–810.
"""

from __future__ import annotations

from typing import Callable, List, Literal, Optional

import numpy as np
import pandas as pd

# 重用 sepsis.py 中的工具函数
from .sepsis import (
    delta_cummin,
    delta_min,
    delta_start,
    susp_inf,
    _process_abx,
    _process_samp,
    _si_and,
    _si_or
)

def sep3_sofa2(
    sofa2: pd.DataFrame,
    susp_inf_df: pd.DataFrame,
    id_cols: list,
    index_col: str,
    si_window: Literal["first", "last", "any"] = "first",
    delta_fun: Callable[[pd.Series], pd.Series] = delta_cummin,
    sofa_thresh: int = 2,
    si_lwr: pd.Timedelta = pd.Timedelta(hours=48),
    si_upr: pd.Timedelta = pd.Timedelta(hours=24),
    keep_components: bool = False,
) -> pd.DataFrame:
    """Detect Sepsis-3 using SOFA-2 scores (ΔSOFA-2 ≥ threshold within SI window).

    This function mirrors sep3() from sepsis.py but uses SOFA-2 scores instead of SOFA-1.
    The detection criteria remain the same: ≥2 point increase in score within the
    suspected infection window.
    
    Implementation mirrors R ricu's sep3 function:
    1. Filter SI events where susp_inf == TRUE
    2. Apply si_window filter ("first", "last", or "any")
    3. Create time windows: [si_time - si_lwr, si_time + si_upr]
    4. Join with SOFA-2 scores using the window
    5. Calculate delta_sofa2 using delta_fun within each window
    6. Keep events where delta_sofa2 >= sofa_thresh
    7. Take first occurrence per patient

    Args:
        sofa2: SOFA-2 score data (must have 'sofa2' column)
        susp_inf_df: Suspected infection data
        id_cols: ID columns for patient identification
        index_col: Time index column
        si_window: Which SI window to use ('first', 'last', 'any')
        delta_fun: Function to calculate SOFA-2 delta (default: delta_cummin)
        sofa_thresh: Required SOFA-2 increase (default 2)
        si_lwr: Time before SI (default 48h)
        si_upr: Time after SI (default 24h)
        keep_components: Whether to keep delta_sofa2, samp_time, abx_time
        
    Returns:
        DataFrame with Sepsis-3 events using SOFA-2 (sep3_sofa2 column = True)
        
    Examples:
        >>> # Basic Sepsis-3 detection with SOFA-2
        >>> sep3_sofa2_events = sep3_sofa2(sofa2_df, si_df, ['stay_id'], 'charttime')
        >>> 
        >>> # Custom threshold and keep components
        >>> sep3_sofa2_events = sep3_sofa2(
        ...     sofa2_df, si_df, ['stay_id'], 'charttime',
        ...     sofa_thresh=3, keep_components=True
        ... )
    """
    # Filter SI events
    if 'susp_inf' in susp_inf_df.columns:
        si_events = susp_inf_df[susp_inf_df['susp_inf'] == True].copy()
    else:
        # If no susp_inf column, assume all rows are SI events
        si_events = susp_inf_df.copy()
    
    if si_events.empty:
        return pd.DataFrame(columns=id_cols + [index_col, "sep3_sofa2"])

    # Determine if time is numeric (hours) or datetime
    si_time_is_numeric = pd.api.types.is_numeric_dtype(si_events[index_col])
    
    # Convert Timedelta to hours if time is numeric
    if si_time_is_numeric:
        si_lwr_hours = si_lwr.total_seconds() / 3600.0
        si_upr_hours = si_upr.total_seconds() / 3600.0
    else:
        # Ensure time columns are datetime
        if index_col in si_events.columns:
            si_events[index_col] = pd.to_datetime(si_events[index_col], errors='coerce')
        if index_col in sofa2.columns:
            sofa2 = sofa2.copy()
            sofa2[index_col] = pd.to_datetime(sofa2[index_col], errors='coerce')

    # Apply si_window filter: "first", "last", or "any"
    if si_window in ("first", "last"):
        grp = si_events.groupby(id_cols, as_index=False)
        si_events = grp.first() if si_window == "first" else grp.last()
        si_events = si_events.reset_index(drop=True)

    # Define SI windows - matches R ricu logic
    # R ricu: [si_time - si_lwr, si_time + si_upr]
    if si_time_is_numeric:
        si_events['si_lwr_time'] = si_events[index_col] - si_lwr_hours
        si_events['si_upr_time'] = si_events[index_col] + si_upr_hours
    else:
        si_events["si_lwr_time"] = pd.to_datetime(si_events[index_col], errors="coerce") - si_lwr
        si_events["si_upr_time"] = pd.to_datetime(si_events[index_col], errors="coerce") + si_upr

    # ============ VECTORIZED IMPLEMENTATION ============
    # Use merge instead of iterrows for better performance
    
    # Prepare SI events with unique row identifier
    si_events = si_events.reset_index(drop=True)
    si_events['_si_idx'] = si_events.index
    si_events['_si_time'] = si_events[index_col]
    
    # Prepare SOFA-2 data
    sofa2_prep = sofa2.copy()
    
    # Perform merge on ID columns
    merged = si_events.merge(sofa2_prep, on=id_cols, how='inner', suffixes=('_si', '_sofa'))
    
    if merged.empty:
        return pd.DataFrame(columns=id_cols + [index_col, "sep3_sofa2"])
    
    # Determine SOFA time column name after merge
    sofa_time_col = f"{index_col}_sofa" if f"{index_col}_sofa" in merged.columns else index_col
    
    # Apply time window filter: si_lwr_time <= sofa_time <= si_upr_time
    time_mask = (merged[sofa_time_col] >= merged['si_lwr_time']) & (merged[sofa_time_col] <= merged['si_upr_time'])
    merged = merged[time_mask].copy()
    
    if merged.empty:
        return pd.DataFrame(columns=id_cols + [index_col, "sep3_sofa2"])
    
    # Sort by SI index and SOFA time to ensure proper ordering for delta calculation
    merged = merged.sort_values(['_si_idx', sofa_time_col])
    
    # Calculate delta_sofa2 within each SI event window
    # Group by patient IDs and SI index
    group_cols = id_cols + ['_si_idx']
    
    # 优化：对于 delta_cummin，直接用向量化操作避免逐组循环
    if 'sofa2' in merged.columns:
        if delta_fun is delta_cummin or getattr(delta_fun, '__name__', None) == 'delta_cummin':
            # 向量化计算 delta_cummin: x - cummin(x) per group
            integer_max = 2147483647
            sofa_filled = merged['sofa2'].fillna(integer_max)
            # 计算每组内的 cummin
            cummin_vals = sofa_filled.groupby([merged[c] for c in group_cols], sort=False).cummin()
            merged['delta_sofa2'] = merged['sofa2'] - cummin_vals
            # NaN 的位置保持 NaN
            merged.loc[merged['sofa2'].isna(), 'delta_sofa2'] = np.nan
        else:
            # 其他 delta 函数使用逐组循环
            merged['delta_sofa2'] = np.nan
            for key, group_idx in merged.groupby(group_cols).groups.items():
                group_sofa = merged.loc[group_idx, 'sofa2']
                merged.loc[group_idx, 'delta_sofa2'] = delta_fun(group_sofa).values
    else:
        merged['delta_sofa2'] = np.nan
    
    if len(merged) == 0:
        return pd.DataFrame(columns=id_cols + [index_col, "sep3_sofa2"])
    
    # Filter by threshold: delta_sofa2 >= sofa_thresh
    sep_rows = merged[merged["delta_sofa2"] >= sofa_thresh].copy()
    
    if sep_rows.empty:
        return pd.DataFrame(columns=id_cols + [index_col, "sep3_sofa2"])
    
    # Take first occurrence per SI event (earliest SOFA time meeting threshold)
    sep_rows = sep_rows.sort_values(sofa_time_col)
    first_per_si = sep_rows.groupby(group_cols, as_index=False).first()
    
    # Build output DataFrame
    out_cols = id_cols + [sofa_time_col]
    out = first_per_si[out_cols].copy()
    out = out.rename(columns={sofa_time_col: index_col})
    out["sep3_sofa2"] = True
    
    if keep_components:
        out["delta_sofa2"] = first_per_si["delta_sofa2"].values
        if "samp_time" in first_per_si.columns:
            out["samp_time"] = first_per_si["samp_time"].values
        elif "samp_time_si" in first_per_si.columns:
            out["samp_time"] = first_per_si["samp_time_si"].values
        if "abx_time" in first_per_si.columns:
            out["abx_time"] = first_per_si["abx_time"].values
        elif "abx_time_si" in first_per_si.columns:
            out["abx_time"] = first_per_si["abx_time_si"].values
    
    # Keep only first Sepsis-3 event per patient
    out = out.sort_values(index_col).groupby(id_cols, as_index=False).first()
    
    return out

def label_sep3_sofa2(
    sofa2_data: pd.DataFrame,
    susp_inf_data: pd.DataFrame,
    id_cols: list,
    index_col: str,
    delta_sofa2: int = 2,
    si_mode: str = "and",
    keep_components: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Sepsis-3 labeling using SOFA-2 - wrapper function calling sep3_sofa2().
    
    This is the Python implementation of R ricu label_sep3, providing
    a user-friendly interface for Sepsis-3 detection using SOFA-2 scores.
    
    Args:
        sofa2_data: SOFA-2 score data
        susp_inf_data: Suspected infection data
        id_cols: Patient ID columns
        index_col: Time index column
        delta_sofa2: SOFA-2 score increase threshold (default 2)
        si_mode: Suspected infection mode (passed to susp_inf if needed)
        keep_components: Whether to keep component columns
        **kwargs: Additional arguments passed to sep3_sofa2()
        
    Returns:
        Sepsis-3 labeling results using SOFA-2
        
    Examples:
        >>> # Basic usage
        >>> sep3_sofa2_labels = label_sep3_sofa2(sofa2_df, si_df, ['stay_id'], 'charttime')
        >>> 
        >>> # Custom threshold
        >>> sep3_sofa2_labels = label_sep3_sofa2(
        ...     sofa2_df, si_df, ['stay_id'], 'charttime',
        ...     delta_sofa2=3
        ... )
    """
    return sep3_sofa2(
        sofa2=sofa2_data,
        susp_inf_df=susp_inf_data,
        id_cols=id_cols,
        index_col=index_col,
        sofa_thresh=delta_sofa2,
        keep_components=keep_components,
        **kwargs
    )

__all__ = [
    "sep2",
    "label_sep2",
    "susp_inf",
    "delta_cummin",
    "delta_min",
    "delta_start",
]
