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
    sofa2_time_is_numeric = pd.api.types.is_numeric_dtype(sofa2[index_col]) if index_col in sofa2.columns else False
    
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
        si_events['si_lwr'] = si_events[index_col] - si_lwr_hours
        si_events['si_upr'] = si_events[index_col] + si_upr_hours
    else:
        si_events["si_lwr"] = pd.to_datetime(si_events[index_col], errors="coerce") - si_lwr
        si_events["si_upr"] = pd.to_datetime(si_events[index_col], errors="coerce") + si_upr

    # Prepare SOFA-2 data with join columns
    # R ricu uses: sofa[, c("join_time1", "join_time2") := list(time, time)]
    sofa2_prep = sofa2.copy()
    sofa2_prep['join_time1'] = sofa2_prep[index_col]
    sofa2_prep['join_time2'] = sofa2_prep[index_col]

    results = []
    
    # For each SI event, find SOFA-2 scores in window and calculate delta
    # R ricu: sofa[susp, list(delta_sofa = delta_fun(sofa), ...), 
    #              on = .(id, join_time1 >= si_lwr, join_time2 <= si_upr), 
    #              by = .EACHI, nomatch = NULL]
    for _, si_row in si_events.iterrows():
        # Match by patient ID
        id_match = pd.Series(True, index=sofa2_prep.index)
        for col in id_cols:
            if col in sofa2_prep.columns:
                id_match = id_match & (sofa2_prep[col] == si_row[col])

        # Match by time window: join_time1 >= si_lwr AND join_time2 <= si_upr
        # This is equivalent to: si_lwr <= time <= si_upr
        time_match = (
            (sofa2_prep['join_time1'] >= si_row['si_lwr']) &
            (sofa2_prep['join_time2'] <= si_row['si_upr'])
        ) if index_col in sofa2_prep.columns else pd.Series(False, index=sofa2_prep.index)

        window = sofa2_prep[id_match & time_match].copy()
        
        if window.empty:
            continue

        # Sort by time and calculate SOFA-2 delta using delta_fun
        # R ricu: delta_sofa = delta_fun(sofa)
        window = window.sort_values(index_col)
        
        if 'sofa2' in window.columns:
            window["delta_sofa2"] = delta_fun(window["sofa2"])
        else:
            window["delta_sofa2"] = np.nan

        # Filter by threshold: delta_sofa2 >= sofa_thresh
        # R ricu: res[delta_sofa >= sofa_thresh]
        sep2_rows = window[window["delta_sofa2"] >= sofa_thresh]
        
        if sep2_rows.empty:
            continue

        # Take first occurrence (earliest time meeting threshold)
        # R ricu: res[, head(.SD, n = 1L), by = id_vars(res)]
        first_hit = sep2_rows.iloc[0]
        
        row = {index_col: first_hit[index_col]}
        for col in id_cols:
            row[col] = si_row[col]
        
        row["sep3_sofa2"] = True
        
        if keep_components:
            row["delta_sofa2"] = first_hit["delta_sofa2"]
            if "samp_time" in si_row:
                row["samp_time"] = si_row["samp_time"]
            if "abx_time" in si_row:
                row["abx_time"] = si_row["abx_time"]
        
        results.append(row)

    if not results:
        return pd.DataFrame(columns=id_cols + [index_col, "sep3_sofa2"])

    # Keep only first Sepsis-3 event per patient
    # R ricu: res[, head(.SD, n = 1L), by = id_vars(res)]
    out = pd.DataFrame(results)
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
