"""
KDIGO AKI (Acute Kidney Injury) Implementation for PyRICU.

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

Author: PyRICU Team
Date: 2026-01-26
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import logging

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
    - Stage 0: No AKI
    
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
    results = []
    
    # Detect time unit for numeric time columns
    time_unit = _detect_time_unit(df[time_col])
    logger.debug(f"Creatinine baseline calculation using time unit: {time_unit}")
    
    for patient_id, group in df.groupby(id_col):
        group = group.sort_values(time_col).copy()
        
        # Calculate time differences
        time_vals = group[time_col].values
        crea_vals = group[value_col].values
        
        creat_low_48hr = np.full(len(group), np.nan)
        creat_low_7day = np.full(len(group), np.nan)
        
        for i in range(len(group)):
            current_time = time_vals[i]
            
            # Find measurements in past 48 hours (excluding current)
            if time_unit == 'datetime':
                time_diff_hr = (current_time - time_vals[:i]) / np.timedelta64(1, 'h') if i > 0 else np.array([])
            elif time_unit == 'seconds':
                # SICdb uses seconds
                time_diff_hr = (current_time - time_vals[:i]) / 3600 if i > 0 else np.array([])  # Convert to hours
            else:
                # Default: minutes
                time_diff_hr = (current_time - time_vals[:i]) / 60 if i > 0 else np.array([])  # Convert to hours
            
            mask_48 = (time_diff_hr > 0) & (time_diff_hr <= 48)
            mask_7d = (time_diff_hr > 0) & (time_diff_hr <= 168)  # 7 days = 168 hours
            
            if i > 0 and mask_48.any():
                creat_low_48hr[i] = np.nanmin(crea_vals[:i][mask_48])
            
            if i > 0 and mask_7d.any():
                creat_low_7day[i] = np.nanmin(crea_vals[:i][mask_7d])
        
        group['creat_low_past_48hr'] = creat_low_48hr
        group['creat_low_past_7day'] = creat_low_7day
        
        results.append(group)
    
    result = pd.concat(results, ignore_index=True)
    
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
    stage = pd.Series(0, index=creat.index, dtype=int)
    
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
        urine_df: DataFrame with urine output values (mL)
        weight_df: DataFrame with patient weight (kg)
        id_col: Column name for patient ID (auto-detected if None)
        time_col: Column name for time (auto-detected if None)
        urine_col: Column name for urine output values
        weight_col: Column name for weight values
        
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
        urine_col, weight_col
    )
    
    if result.empty:
        return pd.DataFrame()
    
    # Calculate AKI stage based on urine output
    result['aki_stage_uo'] = _calc_aki_stage_uo(
        result.get('uo_rt_6hr'),
        result.get('uo_rt_12hr'),
        result.get('uo_rt_24hr')
    )
    
    # Rename columns for consistency
    result = result.rename(columns={
        'uo_6h': 'uo_rt_6hr',
        'uo_12h': 'uo_rt_12hr',
        'uo_24h': 'uo_rt_24hr'
    })
    
    return result


def _detect_time_unit(time_series: pd.Series) -> str:
    """Detect the unit of a numeric time series.
    
    Returns:
        'seconds': Time values are in seconds (e.g., SICdb)
        'minutes': Time values are in minutes (e.g., MIIV, AUMC, eICU)
        'datetime': Time values are datetime objects
    """
    if pd.api.types.is_datetime64_any_dtype(time_series):
        return 'datetime'
    
    # For numeric time, check the magnitude
    # If max value > 50000, likely seconds (50000 seconds = ~14 hours)
    # Typical ICU stays are 1-30 days = 1440-43200 minutes = 86400-2592000 seconds
    max_val = time_series.max()
    
    if max_val > 100000:  # > 100000 seconds = 27.8 hours is reasonable for seconds
        return 'seconds'
    elif max_val > 50000:  # Ambiguous zone, but lean towards seconds
        return 'seconds'
    else:
        return 'minutes'


def _calculate_uo_rates_simple(
    urine_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str = 'urine',
    weight_col: str = 'weight',
) -> pd.DataFrame:
    """Calculate urine output rates using simplified time-windowed averages.
    
    This function handles datetime, minutes (MIIV, AUMC, eICU), and 
    seconds (SICdb) time columns automatically.
    
    Args:
        urine_df: DataFrame with urine output values (mL)
        weight_df: DataFrame with patient weight (kg)
        id_col: Column name for patient ID
        time_col: Column name for time
        urine_col: Column name for urine output values
        weight_col: Column name for weight values
        
    Returns:
        DataFrame with columns: id_col, time_col, uo_rt_6hr, uo_rt_12hr, uo_rt_24hr
    """
    if urine_df.empty or weight_df.empty:
        return pd.DataFrame()
    
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
    
    # Get first weight per patient for simplicity
    if weight_id_col and weight_id_col in weight.columns:
        weight_per_patient = weight.groupby(weight_id_col)[weight_col].first().to_dict()
    else:
        # Single weight value
        weight_per_patient = {None: weight[weight_col].iloc[0] if len(weight) > 0 else 70.0}
    
    # Sort urine by patient and time
    urine = urine.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Determine time unit (datetime, minutes, or seconds)
    time_unit = _detect_time_unit(urine[time_col])
    logger.debug(f"Detected time unit for UO calculation: {time_unit}")
    
    # Define window sizes and conversion factor to get hours
    if time_unit == 'datetime':
        # Will convert to minutes in the loop
        to_minutes_factor = 1.0  # Already handled specially
    elif time_unit == 'seconds':
        # SICdb uses seconds
        to_minutes_factor = 1.0 / 60.0  # 1 second = 1/60 minute
    else:
        # Default: minutes
        to_minutes_factor = 1.0
    
    results = []
    
    for patient_id, group in urine.groupby(id_col):
        group = group.sort_values(time_col).copy()
        
        # Get patient weight
        pt_weight = weight_per_patient.get(patient_id)
        if pt_weight is None:
            pt_weight = weight_per_patient.get(None, 70.0)  # Default weight
        
        if pd.isna(pt_weight) or pt_weight <= 0:
            pt_weight = 70.0  # Default
        
        time_vals = group[time_col].values
        urine_vals = group[urine_col].values
        
        n = len(group)
        uo_6h = np.full(n, np.nan)
        uo_12h = np.full(n, np.nan)
        uo_24h = np.full(n, np.nan)
        
        for i in range(n):
            current_time = time_vals[i]
            
            # Calculate time differences in MINUTES
            if time_unit == 'datetime':
                time_diffs_min = (current_time - time_vals[:i+1]) / np.timedelta64(1, 'm')
            else:
                # Numeric time (either seconds or minutes)
                time_diffs_raw = current_time - time_vals[:i+1]
                time_diffs_min = time_diffs_raw * to_minutes_factor
            
            # 6-hour window (0 to 360 minutes)
            mask_6h = (time_diffs_min >= 0) & (time_diffs_min < 360)
            if mask_6h.any():
                total_urine_6h = np.nansum(urine_vals[:i+1][mask_6h])
                hours_6h = max(time_diffs_min[mask_6h].max() / 60, 1.0)  # At least 1 hour
                uo_6h[i] = total_urine_6h / (pt_weight * hours_6h)
            
            # 12-hour window (0 to 720 minutes)
            mask_12h = (time_diffs_min >= 0) & (time_diffs_min < 720)
            if mask_12h.any():
                total_urine_12h = np.nansum(urine_vals[:i+1][mask_12h])
                hours_12h = max(time_diffs_min[mask_12h].max() / 60, 1.0)
                uo_12h[i] = total_urine_12h / (pt_weight * hours_12h)
            
            # 24-hour window (0 to 1440 minutes)
            mask_24h = (time_diffs_min >= 0) & (time_diffs_min < 1440)
            if mask_24h.any():
                total_urine_24h = np.nansum(urine_vals[:i+1][mask_24h])
                hours_24h = max(time_diffs_min[mask_24h].max() / 60, 1.0)
                uo_24h[i] = total_urine_24h / (pt_weight * hours_24h)
        
        group['uo_rt_6hr'] = uo_6h
        group['uo_rt_12hr'] = uo_12h
        group['uo_rt_24hr'] = uo_24h
        
        results.append(group[[id_col, time_col, 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr']])
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)


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
    - Stage 0: No AKI
    """
    if uo_6h is None:
        return pd.Series(0, dtype=int)
    
    stage = pd.Series(0, index=uo_6h.index, dtype=int)
    
    uo_6h_num = pd.to_numeric(uo_6h, errors='coerce')
    uo_12h_num = pd.to_numeric(uo_12h, errors='coerce') if uo_12h is not None else pd.Series(np.nan, index=uo_6h.index)
    uo_24h_num = pd.to_numeric(uo_24h, errors='coerce') if uo_24h is not None else pd.Series(np.nan, index=uo_6h.index)
    
    # Stage 1: UO < 0.5 for 6h but NOT for 12h
    mask_1 = (uo_6h_num < 0.5) & ((uo_12h_num >= 0.5) | uo_12h_num.isna())
    stage[mask_1] = 1
    
    # Stage 2: UO < 0.5 for ≥12h
    mask_2 = uo_12h_num < 0.5
    stage[mask_2] = 2
    
    # Stage 3: UO < 0.3 for ≥24h OR anuria (0) for ≥12h
    mask_3_oliguria = uo_24h_num < 0.3
    mask_3_anuria = uo_12h_num == 0
    stage[mask_3_oliguria | mask_3_anuria] = 3
    
    return stage


def kdigo_stages(
    crea_df: pd.DataFrame,
    urine_df: Optional[pd.DataFrame] = None,
    weight_df: Optional[pd.DataFrame] = None,
    rrt_df: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    crea_col: str = 'crea',
    urine_col: str = 'urine',
    weight_col: str = 'weight',
) -> pd.DataFrame:
    """Calculate combined KDIGO AKI staging using creatinine and urine output.
    
    This is the main function for KDIGO AKI staging. It combines:
    - Creatinine-based staging (baseline and acute rise)
    - Urine output-based staging (6h, 12h, 24h rates)
    - RRT initiation (automatic Stage 3)
    
    The final AKI stage is the MAXIMUM of creatinine and urine output stages.
    
    Args:
        crea_df: DataFrame with creatinine values
        urine_df: DataFrame with urine output values (optional)
        weight_df: DataFrame with patient weight (optional, required if urine_df provided)
        rrt_df: DataFrame with RRT indicator (optional)
        id_col: Column name for patient ID
        time_col: Column name for time
        crea_col: Column name for creatinine values
        urine_col: Column name for urine values
        weight_col: Column name for weight values
        
    Returns:
        DataFrame with combined AKI staging including:
        - aki_stage_creat: Creatinine-based stage (0-3)
        - aki_stage_uo: Urine output-based stage (0-3)
        - aki_stage: Final combined stage (0-3)
        - aki: Boolean indicator (True if aki_stage > 0)
    """
    if crea_df.empty:
        return pd.DataFrame()
    
    # Auto-detect columns
    id_col = _detect_id_col(crea_df, id_col)
    time_col = _detect_time_col(crea_df, time_col)
    
    # Calculate creatinine-based staging
    crea_staging = kdigo_creatinine(crea_df, id_col, time_col, crea_col)
    
    if crea_staging.empty:
        return pd.DataFrame()
    
    result = crea_staging.copy()
    
    # Calculate urine output-based staging if data available
    if urine_df is not None and weight_df is not None and not urine_df.empty:
        try:
            uo_staging = kdigo_uo(urine_df, weight_df, id_col, time_col, urine_col, weight_col)
            
            if not uo_staging.empty:
                # Merge UO staging with creatinine staging
                result = result.merge(
                    uo_staging[[id_col, time_col, 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr', 'aki_stage_uo']],
                    on=[id_col, time_col],
                    how='outer'
                )
        except Exception as e:
            logger.warning(f"Failed to calculate UO-based AKI staging: {e}")
            result['aki_stage_uo'] = 0
    else:
        result['aki_stage_uo'] = 0
    
    # Handle RRT - automatic Stage 3
    result['aki_stage_rrt'] = 0  # Default: no RRT
    if rrt_df is not None and not rrt_df.empty:
        rrt_col = _detect_value_col(rrt_df, 'rrt')
        if rrt_col:
            result = result.merge(
                rrt_df[[id_col, time_col, rrt_col]].rename(columns={rrt_col: 'rrt'}),
                on=[id_col, time_col],
                how='left'
            )
            # RRT = Stage 3
            rrt_mask = result['rrt'].fillna(False).astype(bool)
            result.loc[rrt_mask, 'aki_stage_creat'] = np.maximum(
                result.loc[rrt_mask, 'aki_stage_creat'].fillna(0), 3
            )
            # aki_stage_rrt: 3 if RRT active, 0 otherwise
            result.loc[rrt_mask, 'aki_stage_rrt'] = 3
    
    # Calculate combined AKI stage
    result['aki_stage_creat'] = result['aki_stage_creat'].fillna(0).astype(int)
    result['aki_stage_uo'] = result['aki_stage_uo'].fillna(0).astype(int)
    result['aki_stage_rrt'] = result['aki_stage_rrt'].fillna(0).astype(int)
    result['aki_stage'] = np.maximum(result['aki_stage_creat'], result['aki_stage_uo'])
    
    # Boolean AKI indicator
    result['aki'] = result['aki_stage'] > 0
    
    return result


def load_kdigo_aki(
    database: str,
    data_path: Optional[str] = None,
    patient_ids: Optional[List] = None,
    max_patients: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load KDIGO AKI staging for a given database using PyRICU concepts.
    
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
        
    Returns:
        DataFrame with KDIGO AKI staging including:
        - id_col, time_col
        - crea, creat_low_past_48hr, creat_low_past_7day
        - aki_stage_creat, aki_stage_uo, aki_stage, aki
        
    Example:
        >>> from pyricu.kdigo_aki import load_kdigo_aki
        >>> aki_df = load_kdigo_aki('miiv', max_patients=100)
        >>> print(f"AKI prevalence: {aki_df['aki'].mean():.1%}")
    """
    from .api import load_concepts
    
    if verbose:
        logger.info(f"Loading KDIGO AKI data for {database}...")
    
    # Load creatinine
    crea_df = load_concepts(
        concepts=['crea'],
        database=database,
        data_path=data_path,
        patient_ids=patient_ids,
        max_patients=max_patients,
        verbose=verbose
    )
    
    if crea_df.empty:
        logger.warning(f"No creatinine data found for {database}")
        return pd.DataFrame()
    
    # Detect ID column for this database
    id_col = _detect_id_col(crea_df)
    time_col = _detect_time_col(crea_df)
    
    # Load urine output and weight
    urine_df = None
    weight_df = None
    
    try:
        urine_df = load_concepts(
            concepts=['urine'],
            database=database,
            data_path=data_path,
            patient_ids=patient_ids,
            max_patients=max_patients,
            verbose=verbose
        )
    except Exception as e:
        logger.warning(f"Failed to load urine data: {e}")
    
    try:
        weight_df = load_concepts(
            concepts=['weight'],
            database=database,
            data_path=data_path,
            patient_ids=patient_ids,
            max_patients=max_patients,
            verbose=verbose
        )
    except Exception as e:
        logger.warning(f"Failed to load weight data: {e}")
    
    # Load RRT data
    rrt_df = None
    try:
        rrt_df = load_concepts(
            concepts=['rrt'],
            database=database,
            data_path=data_path,
            patient_ids=patient_ids,
            max_patients=max_patients,
            verbose=verbose
        )
    except Exception as e:
        logger.warning(f"Failed to load RRT data: {e}")
    
    # Calculate KDIGO AKI staging
    result = kdigo_stages(
        crea_df=crea_df,
        urine_df=urine_df,
        weight_df=weight_df,
        rrt_df=rrt_df,
        id_col=id_col,
        time_col=time_col,
        crea_col='crea',
        urine_col='urine',
        weight_col='weight'
    )
    
    if verbose and not result.empty:
        aki_rate = result['aki'].mean() * 100
        stage_dist = result['aki_stage'].value_counts().sort_index()
        logger.info(f"KDIGO AKI Results for {database}:")
        logger.info(f"  Total rows: {len(result):,}")
        logger.info(f"  AKI prevalence: {aki_rate:.1f}%")
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
        'charttime', 'measuredat', 'measuredat_minutes',
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
    """Generate summary statistics for AKI staging results.
    
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
    
    # Patient-level AKI (any AKI during stay)
    patient_aki = aki_df.groupby(id_col)['aki'].any()
    n_aki_patients = patient_aki.sum()
    
    # Stage distribution (at measurement level)
    stage_dist = aki_df['aki_stage'].value_counts().sort_index().to_dict()
    
    # Max stage per patient
    max_stage_per_patient = aki_df.groupby(id_col)['aki_stage'].max()
    max_stage_dist = max_stage_per_patient.value_counts().sort_index().to_dict()
    
    return {
        'n_patients': n_patients,
        'n_measurements': n_measurements,
        'aki_patients': int(n_aki_patients),
        'aki_rate': float(n_aki_patients / n_patients) if n_patients > 0 else 0.0,
        'stage_distribution_measurements': stage_dist,
        'max_stage_distribution_patients': max_stage_dist,
    }
