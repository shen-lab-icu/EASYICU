"""
KDIGO AKI (Acute Kidney Injury) Implementation for EasyICU.

This module implements the KDIGO (Kidney Disease: Improving Global Outcomes) 
criteria for staging Acute Kidney Injury (AKI) across all supported ICU databases.

KDIGO AKI Staging Criteria:
===========================

**Stage 1:**
- Creatinine: ≥0.3 mg/dL increase within 48h OR ≥1.5-1.9x baseline (within 7 days)
- Urine Output: <0.5 mL/kg/h for 6-12 hours

**Stage 2:**
- Creatinine: ≥2.0-2.9x baseline
- Urine Output: <0.5 mL/kg/h for ≥12 hours

**Stage 3:**
- Creatinine: ≥3.0x baseline OR ≥4.0 mg/dL (with acute increase ≥0.3 or ≥1.5x)
- Urine Output: <0.3 mL/kg/h for ≥24 hours OR anuria for ≥12 hours
- OR initiation of RRT (Renal Replacement Therapy)

References:
-----------
1. KDIGO Clinical Practice Guideline for Acute Kidney Injury (2012)
2. MIT-LCP MIMIC-IV concepts: https://github.com/MIT-LCP/mimic-iv/tree/master/concepts/organfailure
3. AmsterdamUMCdb: https://github.com/AmsterdamUMC/AmsterdamUMCdb

Author: EasyICU Team
Date: 2026-01-26
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import logging

from easyicu.io.ts_utils import _infer_numeric_time_unit
from easyicu.urine_weight_linkage import resolve_unkeyed_single_entity_weight

logger = logging.getLogger(__name__)


def kdigo_creatinine(
    crea_df: pd.DataFrame,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    value_col: str = 'crea',
) -> pd.DataFrame:
    """Calculate creatinine-based AKI staging using KDIGO criteria.
    
    For each creatinine measurement, calculates:
    - creat_low_past_48hr: Lowest creatinine in past 48 hours (for acute rise ≥0.3)
    - creat_low_past_7day: Lowest creatinine in past 7 days (baseline for fold increase)
    - aki_stage_creat: KDIGO AKI stage based on creatinine criteria
    
    Staging Logic:
    - Stage 3: creat ≥ 3x baseline (7-day) OR (creat ≥ 4.0 with acute rise ≥0.3/48h or ≥1.5x)
    - Stage 2: creat ≥ 2x baseline (7-day)
    - Stage 1: creat ≥ 1.5x baseline (7-day) OR creat ≥ (48h min + 0.3)
    - Stage 0: No AKI after at least one eligible baseline comparison
    - ``<NA>``: Not assessable from the available baseline history
    
    Args:
        crea_df: DataFrame with creatinine values
        id_col: Column name for patient ID (auto-detected if None)
        time_col: Column name for time (auto-detected if None)
        value_col: Column name for creatinine values
        
    Returns:
        DataFrame with columns: id_col, time_col, crea, creat_low_past_48hr, 
        creat_low_past_7day, aki_stage_creat
    """
    if crea_df.empty:
        return pd.DataFrame()
    
    # Auto-detect columns
    id_col = _detect_id_col(crea_df, id_col)
    time_col = _detect_time_col(crea_df, time_col)
    
    if id_col is None or time_col is None:
        raise ValueError(f"Could not detect ID or time columns. Found columns: {crea_df.columns.tolist()}")
    
    # Ensure numeric creatinine values
    df = crea_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Remove invalid values
    df = df[df[value_col].notna() & (df[value_col] > 0) & (df[value_col] <= 150)]
    
    if df.empty:
        return pd.DataFrame()
    
    # Sort by ID and time
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Convert time to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        if pd.api.types.is_numeric_dtype(df[time_col]):
            # Assume minutes from admission
            pass  # Keep as numeric
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Calculate rolling minimum creatinine for 48h and 7 days
    # Vectorized: use searchsorted for O(N log N) window boundaries per patient
    
    # Detect time unit and convert to hours for uniform processing
    time_unit = _detect_time_unit(df[time_col], time_col)
    logger.debug(f"Creatinine baseline calculation using time unit: {time_unit}")
    
    if time_unit == 'datetime':
        ref_time = df[time_col].min()
        df['_hours'] = (df[time_col] - ref_time) / pd.Timedelta(hours=1)
    elif time_unit == 'seconds':
        df['_hours'] = df[time_col].astype(np.float64) / 3600.0
    elif time_unit == 'hours':
        df['_hours'] = df[time_col].astype(np.float64)
    else:  # minutes
        df['_hours'] = df[time_col].astype(np.float64) / 60.0
    
    creat_low_48hr = np.full(len(df), np.nan)
    creat_low_7day = np.full(len(df), np.nan)
    
    # Process each patient with vectorized operations
    for _pid, _idx in df.groupby(id_col, sort=False).indices.items():
        _idx_sorted = _idx[np.argsort(df['_hours'].values[_idx])]
        hours = df['_hours'].values[_idx_sorted]
        creas = df[value_col].values[_idx_sorted]
        n = len(hours)
        if n < 2:
            continue
        # For each i, find the leftmost j where hours[i] - hours[j] <= window
        # i.e. hours[j] >= hours[i] - window
        # searchsorted gives first index >= target
        bounds_48 = np.searchsorted(hours, hours - 48.0, side='left')
        bounds_7d = np.searchsorted(hours, hours - 168.0, side='left')
        
        for i in range(1, n):
            left_48 = bounds_48[i]
            if left_48 < i:  # at least one previous measurement in window
                creat_low_48hr[_idx_sorted[i]] = np.min(creas[left_48:i])
            left_7d = bounds_7d[i]
            if left_7d < i:
                creat_low_7day[_idx_sorted[i]] = np.min(creas[left_7d:i])
    
    df['creat_low_past_48hr'] = creat_low_48hr
    df['creat_low_past_7day'] = creat_low_7day
    df.drop(columns=['_hours'], inplace=True)
    
    result = df
    
    # Calculate AKI stage based on creatinine
    result['aki_stage_creat'] = _calc_aki_stage_creat(
        result[value_col],
        result['creat_low_past_48hr'],
        result['creat_low_past_7day']
    )
    
    # Rename columns for clarity
    result = result.rename(columns={value_col: 'crea'})
    
    return result[[id_col, time_col, 'crea', 'creat_low_past_48hr', 
                   'creat_low_past_7day', 'aki_stage_creat']]


def _calc_aki_stage_creat(
    creat: pd.Series,
    creat_low_48hr: pd.Series,
    creat_low_7day: pd.Series
) -> pd.Series:
    """Calculate KDIGO AKI stage from creatinine values.
    
    KDIGO Creatinine Criteria:
    - Stage 3: creat ≥ 3.0x baseline OR (creat ≥ 4.0 with acute increase)
    - Stage 2: creat ≥ 2.0x baseline (and < 3.0x)
    - Stage 1: creat ≥ 1.5x baseline OR (creat ≥ baseline + 0.3 within 48h)
    - Stage 0: No AKI
    """
    # A first creatinine value has no prior value to compare against.  It is
    # not evidence of "no AKI" and must remain unknown rather than becoming a
    # stage 0 through an integer default.
    stage = pd.Series(pd.NA, index=creat.index, dtype="Int64")
    assessable = creat.notna() & (
        creat_low_48hr.notna() | creat_low_7day.notna()
    )
    stage.loc[assessable] = 0
    
    # Stage 1: ≥1.5x baseline (7-day) OR ≥0.3 increase in 48h
    mask_1_fold = creat >= (creat_low_7day * 1.5)
    mask_1_abs = creat >= (creat_low_48hr + 0.3)
    stage[mask_1_fold | mask_1_abs] = 1
    
    # Stage 2: ≥2.0x baseline (7-day)
    mask_2 = creat >= (creat_low_7day * 2.0)
    stage[mask_2] = 2
    
    # Stage 3: ≥3.0x baseline OR (≥4.0 with acute increase)
    mask_3_fold = creat >= (creat_low_7day * 3.0)
    # For creat ≥ 4.0, require acute increase (≥0.3 in 48h or ≥1.5x baseline)
    mask_3_abs = (creat >= 4.0) & (
        (creat_low_48hr <= 3.7) |  # Can have 0.3 increase to reach 4.0
        (creat >= creat_low_7day * 1.5)  # Or 1.5x baseline
    )
    stage[mask_3_fold | mask_3_abs] = 3
    
    return stage


def kdigo_uo(
    urine_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    urine_col: str = 'urine',
    weight_col: str = 'weight',
    source_is_rate: bool = False,
    interval: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Calculate urine output-based AKI staging using KDIGO criteria.
    
    For each time point, calculates:
    - uo_rt_6hr: Urine output rate over past 6 hours (mL/kg/h)
    - uo_rt_12hr: Urine output rate over past 12 hours (mL/kg/h)
    - uo_rt_24hr: Urine output rate over past 24 hours (mL/kg/h)
    - aki_stage_uo: KDIGO AKI stage based on urine output criteria
    
    Staging Logic:
    - Stage 3: UO < 0.3 mL/kg/h for ≥24h OR anuria for ≥12h
    - Stage 2: UO < 0.5 mL/kg/h for ≥12h
    - Stage 1: UO < 0.5 mL/kg/h for 6-12h
    - Stage 0: No AKI
    
    Args:
        urine_df: DataFrame with urine output values (mL), or hourly-equivalent
            values when ``source_is_rate`` is true
        weight_df: DataFrame with patient weight (kg)
        id_col: Column name for patient ID (auto-detected if None)
        time_col: Column name for time (auto-detected if None)
        urine_col: Column name for urine output values
        weight_col: Column name for weight values
        source_is_rate: Treat urine values as a directly recorded rate source.
            HiRID variable 10020000 requires this path.
        interval: Extraction-bin width represented by each rate-source value.
        
    Returns:
        DataFrame with uo rates and aki_stage_uo
    """
    if urine_df.empty:
        return pd.DataFrame()
    
    # Auto-detect columns
    id_col = _detect_id_col(urine_df, id_col)
    time_col = _detect_time_col(urine_df, time_col)
    
    if id_col is None or time_col is None:
        raise ValueError("Could not detect ID or time columns")
    
    # Calculate UO rates using simplified windowed average
    result = _calculate_uo_rates_simple(
        urine_df, weight_df, 
        id_col, time_col, 
        urine_col, weight_col,
        source_is_rate=source_is_rate,
        interval=interval,
    )
    
    if result.empty:
        return pd.DataFrame()
    
    # Calculate AKI stage based on urine output.  A missing complete window
    # stays unassessable; it is never coerced into a non-oliguric stage 0.
    result['aki_stage_uo'] = _calc_aki_stage_uo(
        result.get('uo_rt_6hr'),
        result.get('uo_rt_12hr'),
        result.get('uo_rt_24hr')
    )
    result['uo_assessable'] = result['aki_stage_uo'].notna()
    result['uo_assessment_reason'] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result.loc[
        ~result['uo_assessable'], 'uo_assessment_reason'
    ] = "uo_window_or_weight_unavailable"
    
    # Rename columns for consistency
    result = result.rename(columns={
        'uo_6h': 'uo_rt_6hr',
        'uo_12h': 'uo_rt_12hr',
        'uo_24h': 'uo_rt_24hr'
    })
    
    return result


def _detect_time_unit(time_series: pd.Series, time_col: str | None = None) -> str:
    """Detect the unit of a numeric time series.
    
    Returns:
        'seconds': Time values are in seconds (e.g., SICdb)
        'hours': Time values are already in hours
        'minutes': Time values are in minutes (e.g., MIIV, AUMC, eICU)
        'datetime': Time values are datetime objects
    """
    if pd.api.types.is_datetime64_any_dtype(time_series):
        return 'datetime'

    # KDIGO accepts both raw source offsets and normalized concept time axes.
    # Reuse ts_utils' inference, but do not let the generic "charttime" name
    # override numeric spacing because older callers pass minute offsets under
    # that column name.
    index_hint = time_col or getattr(time_series, "name", None)
    if str(index_hint).lower() == "charttime":
        index_hint = None
    inferred = _infer_numeric_time_unit(time_series, index_hint)
    if inferred == 's':
        return 'seconds'
    if inferred == 'h':
        return 'hours'
    return 'minutes'


def _calculate_uo_rates_simple(
    urine_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str = 'urine',
    weight_col: str = 'weight',
    source_is_rate: bool = False,
    interval: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Calculate urine output rates using simplified time-windowed averages.
    
    This function handles datetime, minutes (MIIV, AUMC, eICU), and 
    seconds (SICdb) time columns automatically.
    
    Args:
        urine_df: DataFrame with urine output values (mL), or hourly-equivalent
            rate values when ``source_is_rate`` is true
        weight_df: DataFrame with patient weight (kg)
        id_col: Column name for patient ID
        time_col: Column name for time
        urine_col: Column name for urine output values
        weight_col: Column name for weight values
        source_is_rate: Use time-weighted observed-interval rate coverage
            instead of event-volume summation.
        interval: Extraction-bin width represented by each rate-source value.
        
    Returns:
        DataFrame with columns: id_col, time_col, uo_rt_6hr, uo_rt_12hr, uo_rt_24hr
    """
    if urine_df.empty or weight_df.empty:
        return pd.DataFrame()

    if source_is_rate:
        # HiRID 10020000 is OUTurine/h (mL/h), whereas the other databases
        # expose voided volume events. Reusing the event-volume denominator
        # here would divide a rate by charting gaps and create false oliguria.
        # KDIGO staging requires the complete 6/12/24-hour duration, so its
        # thresholds are stricter than the descriptive UO concepts.
        from easyicu.callbacks import _urine_rate_window_avg_multi

        rate_urine = urine_df.copy()
        rate_weight = weight_df.copy()
        if urine_col != "urine" and urine_col in rate_urine.columns:
            rate_urine = rate_urine.rename(columns={urine_col: "urine"})
        if weight_col != "weight" and weight_col in rate_weight.columns:
            rate_weight = rate_weight.rename(columns={weight_col: "weight"})

        windowed = _urine_rate_window_avg_multi(
            rate_urine,
            rate_weight,
            windows=[(6, 6), (12, 12), (24, 24)],
            interval=interval,
        )
        result = windowed["uo_6h"].merge(
            windowed["uo_12h"],
            on=[id_col, time_col],
            how="outer",
            sort=False,
        ).merge(
            windowed["uo_24h"],
            on=[id_col, time_col],
            how="outer",
            sort=False,
        )
        return result.rename(
            columns={
                "uo_6h": "uo_rt_6hr",
                "uo_12h": "uo_rt_12hr",
                "uo_24h": "uo_rt_24hr",
            }
        )
    
    # Copy data
    urine = urine_df.copy()
    weight = weight_df.copy()
    
    # Ensure urine values are numeric
    if urine_col in urine.columns:
        urine[urine_col] = pd.to_numeric(urine[urine_col], errors='coerce')
    else:
        # Try to find urine column
        for col in ['urine', 'value', 'valuenum']:
            if col in urine.columns:
                urine_col = col
                urine[urine_col] = pd.to_numeric(urine[urine_col], errors='coerce')
                break
    
    # Detect weight column in weight_df
    _detect_time_col(weight_df)
    weight_id_col = _detect_id_col(weight_df)
    
    if weight_col not in weight.columns:
        for col in ['weight', 'value', 'valuenum']:
            if col in weight.columns:
                weight_col = col
                break

    if weight_col in weight.columns:
        weight[weight_col] = pd.to_numeric(weight[weight_col], errors='coerce')
    
    # Resolve weight only through a patient key.  The exceptional unkeyed
    # one-entity path is explicitly proved below; selecting ``iloc[0]`` from a
    # multi-patient table would silently apply one patient's weight to another.
    global_weight = np.nan
    if weight_id_col and weight_id_col in weight.columns:
        weight_per_patient = weight.groupby(weight_id_col)[weight_col].first().to_dict()
    else:
        resolution = resolve_unkeyed_single_entity_weight(
            urine,
            weight,
            urine_id_columns=[id_col],
            weight_column=weight_col,
        )
        global_weight = resolution.weight if resolution.weight is not None else np.nan
        if resolution.diagnostic_code:
            logger.info(
                "Leaving urine-output rates unassessable: %s",
                resolution.diagnostic_code,
            )
        weight_per_patient = {}
    
    # Sort urine by patient and time
    urine = urine.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Determine time unit (datetime, hours, minutes, or seconds)
    time_unit = _detect_time_unit(urine[time_col], time_col)
    logger.debug(f"Detected time unit for UO calculation: {time_unit}")

    # Vectorized UO rate calculation using cumsum + searchsorted (O(N log N))
    # Convert time to minutes for uniform window computation
    urine = urine.copy()
    if time_unit == 'datetime':
        _ref = urine[time_col].min()
        urine['_min'] = (urine[time_col] - _ref) / pd.Timedelta(minutes=1)
    elif time_unit == 'seconds':
        urine['_min'] = urine[time_col].astype(np.float64) / 60.0
    elif time_unit == 'hours':
        urine['_min'] = urine[time_col].astype(np.float64) * 60.0
    else:
        urine['_min'] = urine[time_col].astype(np.float64)
    
    # Map weights to each row
    if weight_per_patient:
        urine['_wt'] = urine[id_col].map(weight_per_patient)
    else:
        urine['_wt'] = global_weight
    urine['_wt'] = pd.to_numeric(urine['_wt'], errors='coerce')
    urine.loc[urine['_wt'] <= 0, '_wt'] = np.nan
    
    urine[urine_col] = urine[urine_col].astype(np.float64)
    urine = urine.sort_values([id_col, '_min']).reset_index(drop=True)
    
    # Initialize output columns
    n_total = len(urine)
    uo_6h = np.full(n_total, np.nan)
    uo_12h = np.full(n_total, np.nan)
    uo_24h = np.full(n_total, np.nan)
    
    windows_min = np.array([360.0, 720.0, 1440.0])  # 6h, 12h, 24h in minutes
    
    # Process each patient with vectorized searchsorted + cumsum
    for _pid, _idx in urine.groupby(id_col, sort=False).indices.items():
        _idx_sorted = _idx[np.argsort(urine['_min'].values[_idx])]
        times_min = urine['_min'].values[_idx_sorted]
        u_vals = urine[urine_col].values[_idx_sorted]
        wt = urine['_wt'].values[_idx_sorted[0]]
        n = len(times_min)
        
        # Replicate MIT-LCP's hours_since_previous_row logic:
        # first row defaults to 1 hour, subsequent rows use the elapsed time
        # from the previous urine charting row.
        hours_prev = np.empty(n, dtype=np.float64)
        hours_prev[0] = 1.0
        if n > 1:
            hours_prev[1:] = np.maximum(np.diff(times_min) / 60.0, 0.0)

        # Replace NaN with 0 for cumsum
        u_clean = np.where(np.isnan(u_vals), 0.0, u_vals)
        cum_u = np.concatenate([[0.0], np.cumsum(u_clean)])
        cum_h = np.concatenate([[0.0], np.cumsum(hours_prev)])
        idx_arr = np.arange(n)
        
        for w_idx in range(3):
            window = windows_min[w_idx]
            # For each i, find left boundary: first j where times_min[j] >= times_min[i] - window
            lefts = np.searchsorted(times_min, times_min - window, side='left')
            # Clip: only look at j <= i (backward looking)
            lefts = np.minimum(lefts, idx_arr)
            # Windowed sums via prefix sums
            total_u = cum_u[idx_arr + 1] - cum_u[lefts]
            total_h = cum_h[idx_arr + 1] - cum_h[lefts]

            if w_idx == 0:
                valid = (total_h >= 6.0) & (total_h < 12.0)
            elif w_idx == 1:
                valid = total_h >= 12.0
            else:
                valid = total_h >= 24.0

            rates = np.full(n, np.nan, dtype=np.float64)
            np.divide(
                total_u,
                wt * total_h,
                out=rates,
                where=valid & (total_h > 0),
            )
            
            if w_idx == 0:
                uo_6h[_idx_sorted] = rates
            elif w_idx == 1:
                uo_12h[_idx_sorted] = rates
            else:
                uo_24h[_idx_sorted] = rates
    
    urine['uo_rt_6hr'] = uo_6h
    urine['uo_rt_12hr'] = uo_12h
    urine['uo_rt_24hr'] = uo_24h
    
    return urine[[id_col, time_col, 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr']]


def _calc_aki_stage_uo(
    uo_6h: pd.Series,
    uo_12h: pd.Series,
    uo_24h: pd.Series
) -> pd.Series:
    """Calculate KDIGO AKI stage from urine output rates.
    
    KDIGO Urine Output Criteria:
    - Stage 3: UO < 0.3 mL/kg/h for ≥24h OR anuria (0) for ≥12h
    - Stage 2: UO < 0.5 mL/kg/h for ≥12h
    - Stage 1: UO < 0.5 mL/kg/h for 6-12h (i.e., 6h avg < 0.5 but 12h avg ≥ 0.5)
    - Stage 0: No AKI after at least one complete eligible window
    - ``<NA>``: Not assessable from the available urine/weight history
    """
    if uo_6h is None:
        return pd.Series(dtype="Int64")
    
    stage = pd.Series(pd.NA, index=uo_6h.index, dtype="Int64")
    
    uo_6h_num = pd.to_numeric(uo_6h, errors='coerce')
    uo_12h_num = pd.to_numeric(uo_12h, errors='coerce') if uo_12h is not None else pd.Series(np.nan, index=uo_6h.index)
    uo_24h_num = pd.to_numeric(uo_24h, errors='coerce') if uo_24h is not None else pd.Series(np.nan, index=uo_6h.index)

    # Any complete UO window can establish a non-oliguric stage 0.  Missing
    # windows cannot: they may reflect insufficient coverage, missing weight,
    # or an unprovable linkage.
    stage.loc[
        uo_6h_num.notna() | uo_12h_num.notna() | uo_24h_num.notna()
    ] = 0
    
    # Stage 1: UO < 0.5 for 6h but NOT for 12h
    mask_1 = (uo_6h_num < 0.5) & ((uo_12h_num >= 0.5) | uo_12h_num.isna())
    stage[mask_1] = 1
    
    # Stage 2: UO < 0.5 for ≥12h
    mask_2 = uo_12h_num < 0.5
    stage[mask_2] = 2
    
    # Stage 3: UO < 0.3 for ≥24h OR anuria (0) for ≥12h
    mask_3_oliguria = uo_24h_num < 0.3
    mask_3_anuria = uo_12h_num < 0.01
    stage[mask_3_oliguria | mask_3_anuria] = 3
    
    return stage


def _rrt_active_from_initiation(
    result: pd.DataFrame,
    rrt_view: pd.DataFrame,
    id_col: str,
    time_col: str,
) -> pd.Series:
    """Return True for rows at or after the first documented active RRT time."""
    active = pd.Series(False, index=result.index, dtype=bool)
    if result.empty or rrt_view.empty:
        return active

    rrt = rrt_view.copy()
    if pd.api.types.is_bool_dtype(rrt["rrt"]) or str(rrt["rrt"].dtype) == "boolean":
        rrt["_rrt_active"] = rrt["rrt"].fillna(False).astype(bool)
    elif pd.api.types.is_numeric_dtype(rrt["rrt"]):
        rrt["_rrt_active"] = pd.to_numeric(rrt["rrt"], errors="coerce").fillna(0) > 0
    else:
        rrt["_rrt_active"] = (
            rrt["rrt"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "t", "yes", "y", "active"})
        )

    starts = (
        rrt.loc[rrt["_rrt_active"], [id_col, time_col]]
        .dropna(subset=[id_col, time_col])
        .groupby(id_col, sort=False)[time_col]
        .min()
    )
    if starts.empty:
        return active

    for pid, start_time in starts.items():
        row_mask = result[id_col] == pid
        active.loc[row_mask] = result.loc[row_mask, time_col] >= start_time
    return active


def kdigo_stages(
    crea_df: Optional[pd.DataFrame] = None,
    urine_df: Optional[pd.DataFrame] = None,
    weight_df: Optional[pd.DataFrame] = None,
    rrt_df: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    crea_col: str = 'crea',
    urine_col: str = 'urine',
    weight_col: str = 'weight',
    urine_source_is_rate: bool = False,
    interval: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Calculate component-neutral KDIGO AKI staging.
    
    This is the main function for KDIGO AKI staging. It combines:
    - Creatinine-based staging (baseline and acute rise)
    - Urine output-based staging (6h, 12h, 24h rates)
    - RRT initiation (automatic Stage 3)
    
    The final AKI stage is the maximum of assessable creatinine, urine-output,
    and documented RRT stages.  ``<NA>`` means the available source evidence
    cannot establish either AKI or no AKI; it is intentionally not a stage 0.
    
    Args:
        crea_df: DataFrame with creatinine values (optional)
        urine_df: DataFrame with urine output values (optional)
        weight_df: DataFrame with patient weight (optional, required if urine_df provided)
        rrt_df: DataFrame with RRT indicator (optional)
        id_col: Column name for patient ID
        time_col: Column name for time
        crea_col: Column name for creatinine values
        urine_col: Column name for urine values
        weight_col: Column name for weight values
        urine_source_is_rate: Treat urine as a direct rate source (HiRID).
        interval: Extraction-bin width represented by each rate-source value.
        
    Returns:
        DataFrame with combined AKI staging including:
        - aki_stage_creat: Creatinine-based stage (0-3)
        - aki_stage_uo: Urine output-based stage (0-3)
        - aki_stage: Final combined stage (0-3, or ``<NA>`` when indeterminate)
        - aki: Boolean indicator (True if aki_stage > 0)
        - creat_assessable, uo_assessable, rrt_observed, aki_assessable,
          aki_assessment_reason: component/overall ascertainment receipt
    """
    source_frames = tuple(
        frame
        for frame in (crea_df, urine_df, rrt_df)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    )
    if not source_frames:
        return pd.DataFrame()

    # Do not let one component decide whether a patient enters the KDIGO
    # table.  The first available component supplies the public timeline key;
    # every component then contributes its own rows to the union spine below.
    anchor = next(
        frame
        for frame in (crea_df, urine_df, rrt_df)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    )
    id_col = _detect_id_col(anchor, id_col)
    time_col = _detect_time_col(anchor, time_col)
    if id_col is None or time_col is None:
        raise ValueError(
            "Could not detect a KDIGO ID/time key from available components. "
            f"Found columns: {list(anchor.columns)}"
        )

    crea_staging = pd.DataFrame()
    if isinstance(crea_df, pd.DataFrame) and not crea_df.empty:
        crea_staging = kdigo_creatinine(crea_df, id_col, time_col, crea_col)

    if crea_staging.empty:
        result = anchor[[id_col, time_col]].copy()
        result['aki_stage_creat'] = pd.Series(
            pd.NA, index=result.index, dtype="Int64"
        )
    else:
        result = crea_staging.copy()
    result['creat_assessable'] = result['aki_stage_creat'].notna()
    
    # Calculate urine output-based staging if data available
    if urine_df is not None and weight_df is not None and not urine_df.empty:
        try:
            uo_staging = kdigo_uo(
                urine_df,
                weight_df,
                id_col,
                time_col,
                urine_col,
                weight_col,
                source_is_rate=urine_source_is_rate,
                interval=interval,
            )

            if not uo_staging.empty:
                # 2026-05-20 fix: kdigo_uo auto-detects its own time column
                # from `urine_df` (hirid → 'datetime', eicu →
                # 'observationoffset', miiv/mimic → 'charttime') — the
                # outer `time_col` passed in is just a hint and may not
                # match what's actually in urine_df. The previous code
                # then tried to `merge(..., on=[id_col, time_col])` with
                # a time_col that wasn't present in uo_staging and
                # silently lost uo_rt_6hr/12hr/24hr on hirid+eicu. Detect
                # uo_staging's actual time col and align names before merge.
                uo_time_col = time_col if time_col in uo_staging.columns else \
                              _detect_time_col(uo_staging)
                if uo_time_col and uo_time_col != time_col:
                    uo_staging = uo_staging.rename(columns={uo_time_col: time_col})
                missing_cols = [c for c in [id_col, time_col] if c not in uo_staging.columns]
                if missing_cols:
                    raise KeyError(
                        f"uo_staging is missing merge key columns {missing_cols}; "
                        f"available cols: {list(uo_staging.columns)}"
                    )
                result = result.merge(
                    uo_staging[[
                        id_col,
                        time_col,
                        'uo_rt_6hr',
                        'uo_rt_12hr',
                        'uo_rt_24hr',
                        'aki_stage_uo',
                        'uo_assessable',
                        'uo_assessment_reason',
                    ]],
                    on=[id_col, time_col],
                    how='outer'
                )
        except Exception as e:
            logger.warning(f"Failed to calculate UO-based AKI staging: {e}")
            result['aki_stage_uo'] = pd.Series(pd.NA, index=result.index, dtype="Int64")
            result['uo_assessable'] = False
            result['uo_assessment_reason'] = "uo_calculation_error"
    else:
        result['aki_stage_uo'] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result['uo_assessable'] = False
        result['uo_assessment_reason'] = "urine_or_weight_unavailable"

    def _component_timeline(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=[id_col, time_col])
        source_id = _detect_id_col(frame, id_col)
        source_time = _detect_time_col(frame, time_col)
        if source_id is None or source_time is None:
            return pd.DataFrame(columns=[id_col, time_col])
        return (
            frame[[source_id, source_time]]
            .rename(columns={source_id: id_col, source_time: time_col})
            .dropna(subset=[id_col, time_col])
        )

    # Retain rows from every observed KDIGO component.  In particular, an RRT
    # initiation must survive even when no contemporaneous creatinine exists.
    spine = (
        pd.concat(
            [
                result[[id_col, time_col]],
                _component_timeline(urine_df),
                _component_timeline(rrt_df),
            ],
            ignore_index=True,
        )
        .dropna(subset=[id_col, time_col])
        .drop_duplicates()
        .sort_values([id_col, time_col], kind="stable")
        .reset_index(drop=True)
    )
    result = spine.merge(result, on=[id_col, time_col], how='left', sort=False)

    # An outer UO merge can add times that have no creatinine row.  Preserve
    # the distinction between an unavailable baseline and a proven negative.
    result['creat_assessable'] = result['aki_stage_creat'].notna()
    result['uo_assessable'] = result['uo_assessable'].fillna(False).astype(bool)
    result['uo_assessment_reason'] = result['uo_assessment_reason'].astype("string")
    result.loc[
        ~result['uo_assessable'] & result['uo_assessment_reason'].isna(),
        'uo_assessment_reason',
    ] = "uo_window_or_weight_unavailable"
    
    # Handle RRT - automatic Stage 3
    result['aki_stage_rrt'] = pd.Series(0, index=result.index, dtype="Int64")
    if rrt_df is not None and not rrt_df.empty:
        rrt_id_col = _detect_id_col(rrt_df, id_col)
        rrt_time_col = _detect_time_col(rrt_df, time_col)
        rrt_col = _detect_value_col(rrt_df, 'rrt')
        if rrt_id_col and rrt_time_col and rrt_col:
            rrt_view = rrt_df[[rrt_id_col, rrt_time_col, rrt_col]].rename(
                columns={rrt_id_col: id_col, rrt_time_col: time_col, rrt_col: 'rrt'}
            )
            rrt_mask = _rrt_active_from_initiation(result, rrt_view, id_col, time_col)
            result["rrt"] = rrt_mask
            # RRT is its own component; do not rewrite the creatinine stage.
            result.loc[rrt_mask, 'aki_stage_rrt'] = 3
        else:
            logger.warning(
                "Skipping RRT merge because ID/time/value columns could not be detected: %s",
                list(rrt_df.columns),
            )
    
    # Calculate combined AKI stage.  Inactive/undocumented RRT is deliberately
    # excluded as negative evidence: a zero placeholder must not turn a row
    # with unknown creatinine and UO components into "no AKI".
    result['aki_stage_creat'] = result['aki_stage_creat'].astype("Int64")
    result['aki_stage_uo'] = result['aki_stage_uo'].astype("Int64")
    result['aki_stage_rrt'] = result['aki_stage_rrt'].astype("Int64")
    result['rrt_observed'] = (result['aki_stage_rrt'] > 0).astype("boolean")
    rrt_positive = result['aki_stage_rrt'].where(
        result['aki_stage_rrt'] > 0, pd.NA
    )
    components = pd.concat(
        [result['aki_stage_creat'], result['aki_stage_uo'], rrt_positive],
        axis=1,
    )
    result['aki_stage'] = components.max(axis=1, skipna=True).astype("Int64")
    result['aki'] = (result['aki_stage'] > 0).astype("boolean")
    result['aki_assessable'] = result['aki_stage'].notna()
    result['aki_assessment_reason'] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result.loc[result['aki_stage'] > 0, 'aki_assessment_reason'] = "positive_component"
    result.loc[result['aki_stage'] == 0, 'aki_assessment_reason'] = (
        "assessable_components_negative"
    )
    result.loc[
        ~result['aki_assessable'], 'aki_assessment_reason'
    ] = "no_assessable_component"
    
    return result


def load_kdigo_aki(
    database: str,
    data_path: Optional[str] = None,
    patient_ids: Optional[List] = None,
    max_patients: Optional[int] = None,
    verbose: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load KDIGO AKI staging for a given database using EasyICU concepts.
    
    This is the high-level API function that:
    1. Loads required concepts (crea, urine, weight, rrt) from the database
    2. Calculates KDIGO AKI staging using the loaded data
    3. Returns a unified DataFrame with AKI staging results
    
    Args:
        database: Database name ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        data_path: Path to the database (uses default if None)
        patient_ids: List of patient IDs to load (loads all if None)
        max_patients: Maximum number of patients to load
        verbose: Print progress messages
        preloaded_data: Optional dict of {concept_name: DataFrame} to skip re-loading
        
    Returns:
        DataFrame with KDIGO AKI staging including:
        - id_col, time_col
        - crea, creat_low_past_48hr, creat_low_past_7day
        - aki_stage_creat, aki_stage_uo, aki_stage, aki
        
    Example:
        >>> from easyicu.kdigo_aki import load_kdigo_aki
        >>> aki_df = load_kdigo_aki('miiv', max_patients=100)
        >>> summary = summarize_aki(aki_df)
        >>> print(summary['aki_prevalence_among_assessable'])
    """
    from easyicu.api import load_concepts  # was `from .api` -> resolved to non-existent easyicu.scores.api

    _pre = preloaded_data or {}
    
    def _load_or_reuse(concept):
        if concept in _pre and isinstance(_pre[concept], pd.DataFrame):
            return _pre[concept]
        try:
            return load_concepts(
                concepts=[concept], database=database, data_path=data_path,
                patient_ids=patient_ids, max_patients=max_patients, verbose=verbose
            )
        except Exception as e:
            logger.warning(f"Failed to load {concept}: {e}")
            return None
    
    if verbose:
        logger.info(f"Loading KDIGO AKI data for {database}...")
    
    # Load all KDIGO components before deciding whether the patient has an
    # assessable timeline.  Creatinine is not a membership prerequisite.
    crea_df = _load_or_reuse('crea')
    urine_df = _load_or_reuse('urine')
    weight_df = _load_or_reuse('weight')
    rrt_df = _load_or_reuse('rrt')

    if not any(
        isinstance(frame, pd.DataFrame) and not frame.empty
        for frame in (crea_df, urine_df, rrt_df)
    ):
        logger.warning("No creatinine, urine-output, or RRT data found for %s", database)
        return pd.DataFrame()
    
    # Calculate KDIGO AKI staging
    result = kdigo_stages(
        crea_df=crea_df,
        urine_df=urine_df,
        weight_df=weight_df,
        rrt_df=rrt_df,
        crea_col='crea',
        urine_col='urine',
        weight_col='weight',
        urine_source_is_rate=str(database).lower() == 'hirid',
        interval=pd.Timedelta(hours=1),
    )
    
    if verbose and not result.empty:
        n_total = len(result)
        n_assessable = int(result['aki_assessable'].sum())
        n_positive = int(result['aki'].eq(True).fillna(False).sum())
        n_negative = int(result['aki'].eq(False).fillna(False).sum())
        n_indeterminate = n_total - n_assessable
        prevalence = (
            100.0 * n_positive / n_assessable if n_assessable else float("nan")
        )
        stage_dist = result['aki_stage'].value_counts().sort_index()
        logger.info(f"KDIGO AKI Results for {database}:")
        logger.info(f"  Total rows: {n_total:,}")
        logger.info(f"  AKI positive: {n_positive:,}")
        logger.info(f"  AKI negative (assessable): {n_negative:,}")
        logger.info(f"  AKI indeterminate: {n_indeterminate:,}")
        logger.info(
            "  Prevalence among assessable: %.1f%% (coverage: %d/%d, %.1f%%)",
            prevalence,
            n_assessable,
            n_total,
            100.0 * n_assessable / n_total,
        )
        logger.info(f"  Stage distribution: {stage_dist.to_dict()}")
    
    return result


# ============================================================================
# Helper Functions
# ============================================================================

def _detect_id_col(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    """Detect the patient ID column in a DataFrame."""
    if hint and hint in df.columns:
        return hint
    
    # Priority order for ID columns
    id_candidates = [
        'stay_id', 'icustay_id', 'patientunitstayid', 
        'admissionid', 'patientid', 'CaseID', 'hadm_id'
    ]
    
    for col in id_candidates:
        if col in df.columns:
            return col
    
    # Fallback: look for columns ending with '_id'
    for col in df.columns:
        if col.endswith('_id'):
            return col
    
    return None


def _detect_time_col(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    """Detect the time column in a DataFrame."""
    if hint and hint in df.columns:
        return hint
    
    # Priority order for time columns
    time_candidates = [
        'charttime', 'starttime', 'measuredat', 'measuredat_minutes',
        'observationoffset', 'labresultoffset', 'datetime',
        'nursingchartoffset', 'OffsetOfDataFloat', 'Offset',
        'intakeoutputoffset', 'intakeoutputentryoffset',  # eICU urine
        'registeredat',  # AUMC
    ]
    
    for col in time_candidates:
        if col in df.columns:
            return col
    
    return None


def _detect_value_col(df: pd.DataFrame, concept: str) -> Optional[str]:
    """Detect the value column for a given concept."""
    if concept in df.columns:
        return concept
    
    # Common value column names
    candidates = ['value', 'valuenum', concept.lower(), concept.upper()]
    for col in candidates:
        if col in df.columns:
            return col
    
    return None


# ============================================================================
# Convenience Functions for Specific Stages
# ============================================================================

def get_aki_incidence(
    aki_df: pd.DataFrame,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Get first AKI occurrence for each patient.
    
    Returns the first time point at which each patient developed AKI
    (aki_stage > 0), along with the maximum AKI stage reached.
    
    Args:
        aki_df: DataFrame from kdigo_stages or load_kdigo_aki
        id_col: Patient ID column (auto-detected if None)
        
    Returns:
        DataFrame with columns: id_col, first_aki_time, max_aki_stage
    """
    if aki_df.empty:
        return pd.DataFrame()
    
    id_col = _detect_id_col(aki_df, id_col)
    time_col = _detect_time_col(aki_df)
    
    # Get first AKI occurrence
    aki_only = aki_df[aki_df['aki']].copy()
    
    if aki_only.empty:
        return pd.DataFrame()
    
    first_aki = aki_only.groupby(id_col).agg({
        time_col: 'min',
        'aki_stage': 'max'
    }).reset_index()
    
    first_aki = first_aki.rename(columns={
        time_col: 'first_aki_time',
        'aki_stage': 'max_aki_stage'
    })
    
    return first_aki


def summarize_aki(aki_df: pd.DataFrame, id_col: Optional[str] = None) -> Dict[str, Any]:
    """Generate explicit patient-level AKI ascertainment statistics.

    The prevalence denominator is patients with at least one assessable KDIGO
    component.  Indeterminate patients remain visible rather than being
    silently counted as non-AKI.
    
    Args:
        aki_df: DataFrame from kdigo_stages or load_kdigo_aki
        id_col: Patient ID column (auto-detected if None)
        
    Returns:
        Dictionary with summary statistics
    """
    if aki_df.empty:
        return {'error': 'Empty DataFrame'}
    
    id_col = _detect_id_col(aki_df, id_col)
    
    n_patients = aki_df[id_col].nunique()
    n_measurements = len(aki_df)
    
    if id_col is None:
        return {'error': 'Could not detect patient ID column'}

    assessable_rows = (
        aki_df['aki_assessable']
        if 'aki_assessable' in aki_df.columns
        else aki_df['aki'].notna()
    )
    per_patient = pd.DataFrame(
        {
            id_col: aki_df[id_col],
            'positive': aki_df['aki'].eq(True).fillna(False),
            'assessable': assessable_rows.fillna(False),
        }
    ).groupby(id_col, dropna=True, sort=False).agg(
        positive=('positive', 'any'),
        assessable=('assessable', 'any'),
    )
    n_aki_patients = int(per_patient['positive'].sum())
    n_assessable_patients = int(per_patient['assessable'].sum())
    n_indeterminate_patients = int((~per_patient['assessable']).sum())
    n_negative_patients = n_assessable_patients - n_aki_patients
    prevalence_among_assessable = (
        float(n_aki_patients / n_assessable_patients)
        if n_assessable_patients > 0
        else None
    )
    
    # Stage distribution (at measurement level)
    stage_dist = aki_df['aki_stage'].value_counts().sort_index().to_dict()
    
    # Max stage per patient
    max_stage_per_patient = aki_df.groupby(id_col)['aki_stage'].max()
    max_stage_dist = max_stage_per_patient.value_counts().sort_index().to_dict()
    
    return {
        'n_patients': n_patients,
        'n_measurements': n_measurements,
        'aki_positive_patients': n_aki_patients,
        'aki_negative_assessable_patients': n_negative_patients,
        'aki_indeterminate_patients': n_indeterminate_patients,
        'n_assessable_patients': n_assessable_patients,
        'aki_prevalence_among_assessable': prevalence_among_assessable,
        'ascertainment_coverage': (
            float(n_assessable_patients / n_patients) if n_patients > 0 else None
        ),
        # Retained for callers that used the historical key; its denominator
        # is now explicit above and is never the whole cohort by implication.
        'aki_rate': prevalence_among_assessable,
        'aki_rate_denominator': 'assessable_patients',
        'stage_distribution_measurements': stage_dist,
        'max_stage_distribution_patients': max_stage_dist,
    }
