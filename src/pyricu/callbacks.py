"""Callback functions for clinical scores (SOFA, SIRS, qSOFA, etc.).

This module provides implementations of clinical scoring systems commonly
used in intensive care settings, replicating the functionality of R ricu's
callback functions.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .table import ICUTable
from .common_utils import SeriesUtils

# Import missing callbacks
from .callbacks_missing import (
    rrt_criteria,
    sum_components,
    blood_cell_ratio,
    aumc_bxs,
    blood_cell_count,
    delta_cummin,
    delta_start,
    delta_min,
)
def _standardize_fio2_units(fio2_df: pd.DataFrame, database: str) -> pd.DataFrame:
    """将FiO2标准化为百分比形式（0-100）以实现跨数据库兼容性

    Args:
        fio2_df: FiO2数据DataFrame，包含fio2列
        database: 数据库名称（'miiv', 'eicu', 'aumc'等）

    Returns:
        标准化后的DataFrame
    """
    if 'fio2' not in fio2_df.columns or fio2_df.empty:
        return fio2_df

    # 获取非空值进行分析
    values = fio2_df['fio2'].dropna()
    if len(values) == 0:
        return fio2_df

    # 基于分析结果的数据库特定转换
    if database == 'miiv':
        # MIMIC-IV分析显示pafi计算准确，推断不需要转换
        # 但为了安全起见，添加自动检测
        max_val = values.max()
        if max_val <= 1.0:  # 分数形式
            fio2_df['fio2'] = fio2_df['fio2'] * 100

    elif database == 'eicu':
        # eICU已确认为百分比形式（21-100），不需要转换
        # 但保持一致性检查
        max_val = values.max()
        if max_val <= 1.0:  # 如果意外发现是分数形式
            fio2_df['fio2'] = fio2_df['fio2'] * 100

    elif database == 'aumc':
        # AUMC有percent_as_numeric转换机制，基本确认为百分比形式
        # 但同样添加安全检查
        max_val = values.max()
        if max_val <= 1.0:  # 如果意外发现是分数形式
            fio2_df['fio2'] = fio2_df['fio2'] * 100

    return fio2_df

def _is_true_safe(series: pd.Series) -> pd.Series:
    """Safely convert series to boolean, handling different dtypes.

    Replicates R's is_true: non-NA and True.
    Handles Float64 dtype which can't use fillna(False).
    """
    return SeriesUtils.is_true(series)

def sofa_score(
    data_dict: Dict[str, pd.DataFrame],
    *,
    keep_components: bool = False,
    win_length: pd.Timedelta = None,
    worst_val_fun: str = 'max',
) -> pd.DataFrame:
    """Calculate SOFA (Sequential Organ Failure Assessment) score.

    Replicates R ricu's sofa_score logic:
    1. Merge all components
    2. fill_gaps() - Fill time gaps to create continuous time series
    3. slide() - Apply sliding window to get worst value over win_length
    4. Sum components to get total SOFA score

    The SOFA score aggregates 6 organ system scores:
    - Respiratory (PaO2/FiO2 ratio)
    - Coagulation (platelets)
    - Liver (bilirubin)
    - Cardiovascular (MAP and vasopressors)
    - CNS (Glasgow Coma Score)
    - Renal (creatinine and urine output)

    Args:
        data_dict: Dictionary containing required concept DataFrames:
            - sofa_resp: Respiratory component
            - sofa_coag: Coagulation component
            - sofa_liver: Liver component
            - sofa_cardio: Cardiovascular component
            - sofa_cns: CNS component
            - sofa_renal: Renal component
        keep_components: If True, keep individual components in output
        win_length: Time window for worst value calculation (default 24 hours)
        worst_val_fun: Function to apply over window ('max' or 'min')

    Returns:
        DataFrame with SOFA score and optionally components
    """
    from .ts_utils import fill_gaps, slide
    
    required = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']
    
    # Set default window length
    if win_length is None:
        win_length = pd.Timedelta(hours=24)
    
    # Step 1: Merge all component dataframes using outer joins.
    result = None
    for comp in required:
        if comp in data_dict and not data_dict[comp].empty:
            df = data_dict[comp]
            if result is None:
                result = df
            else:
                result = pd.merge(result, df, on=['stay_id', 'time'], how='outer')

    if result is None or result.empty:
        return pd.DataFrame(columns=['stay_id', 'time', 'sofa'])

    # Step 2: Use fill_gaps to create continuous time series
    # CRITICAL: fill_gaps should only fill within the range where we have data,
    # matching R ricu's behavior where time points are determined by component data.
    # We DON'T create arbitrary time grids beyond data boundaries.
    from .ts_utils import fill_gaps as ricu_fill_gaps
    
    # Apply fill_gaps which will create hourly grid within data range for each patient
    # For miiv database, enforce ricu-style time filtering (time >= 0)
    if 'time' in result.columns and len(result) > 0:
        # Check if this is miiv data by examining data source patterns
        # If we have very negative time values, assume miiv and apply ricu-style filtering
        min_time = result['time'].min()
        if min_time < -1000:  # Likely miiv data with pre-ICU records
            # Filter to only include time >= 0 (ricu-style)
            result = result[result['time'] >= 0].copy()

    aligned_result = ricu_fill_gaps(result, limits=None)

    # Step 3: Apply sliding window to get the "worst" value (max).
    # CRITICAL: Slide on the gapped data, BEFORE forward-filling.
    # This matches R ricu's logic where `slide` operates on the output of `fill_gaps`.
    id_cols = ['stay_id']
    sofa_components = [comp for comp in required if comp in aligned_result.columns]
    
    if win_length.total_seconds() > 0:
        agg_dict = {comp: worst_val_fun for comp in sofa_components}
        
        slid_result = slide(
            data=aligned_result,
            id_cols=id_cols,
            index_col='time',
            before=win_length,
            after=pd.Timedelta(0),
            agg_func=agg_dict,
            full_window=False
        )
    else:
        slid_result = aligned_result.copy()

    # Step 5: Forward-fill the results AFTER sliding.
    # This fills in the hours where the entire 24h window had no data.
    slid_components = [c for c in sofa_components if c in slid_result.columns]
    if slid_components:
        slid_result[slid_components] = slid_result.groupby(id_cols)[slid_components].ffill()

    # CRITICAL: Filter out rows where ALL components are NaN
    # This replicates ricu's behavior where time points only exist if at least one component has data
    # Without this, fill_gaps creates rows beyond component data boundaries
    if slid_components:
        # Keep rows where at least one component is non-NaN
        has_data_mask = slid_result[slid_components].notna().any(axis=1)
        slid_result = slid_result[has_data_mask].copy()

    # Step 6: Calculate the total SOFA score.
    slid_result['sofa'] = slid_result[slid_components].fillna(0).sum(axis=1).astype(int)
    
    # Final column selection.
    final_cols = ['stay_id', 'time', 'sofa']
    if keep_components:
        final_cols.extend(sofa_components)
    
    # Ensure all required columns exist before returning, adding NaN if they were dropped.
    for col in final_cols:
        if col not in slid_result.columns:
            slid_result[col] = np.nan
            
    return slid_result[final_cols]

def sofa_resp(pafi: pd.Series, vent_ind: Optional[pd.Series] = None) -> pd.Series:
    """Calculate respiratory SOFA component.

    Replicates R ricu's sofa_resp logic:
    - Merge pafi with expanded vent_ind (aggregate="any")
    - If pafi < 200 and NOT on ventilation, set pafi = 200
    - Then calculate score based on pafi thresholds

    Score based on PaO2/FiO2 ratio (after adjustment):
    - 0: >= 400
    - 1: < 400
    - 2: < 300
    - 3: < 200
    - 4: < 100

    Args:
        pafi: PaO2/FiO2 ratio values (should already be adjusted if vent_ind provided)
        vent_ind: Ventilation indicator (optional, should be expanded from win_tbl)

    Returns:
        Series with respiratory SOFA scores
    """

    pafi_num = pd.to_numeric(pafi, errors="coerce")
    idx = pafi_num.index
    if vent_ind is not None:
        vent_mask = SeriesUtils.is_true(vent_ind.reindex(idx, copy=False))
    else:
        vent_mask = pd.Series(False, index=idx)

    adj = pafi_num.copy()
    mask = SeriesUtils.is_true(adj < 200) & (~vent_mask)
    adj[mask] = 200

    score = pd.Series(0, index=idx, dtype=float)
    mask_100 = SeriesUtils.is_true(adj < 100)
    mask_200 = SeriesUtils.is_true(adj < 200) & ~mask_100
    mask_300 = SeriesUtils.is_true(adj < 300) & ~mask_200 & ~mask_100
    mask_400 = SeriesUtils.is_true(adj < 400) & ~mask_300 & ~mask_200 & ~mask_100

    score[mask_100] = 4
    score[mask_200] = 3
    score[mask_300] = 2
    score[mask_400] = 1

    # R ricu returns 0 if pafi is missing (via is_true checks failing)
    # So we should NOT set missing values to NaN if we want to match R's behavior
    # missing_mask = pafi_num.isna() & ~vent_mask
    # score[missing_mask] = np.nan

    return score

def sofa_coag(plt: pd.Series) -> pd.Series:
    """Calculate coagulation SOFA component.
    
    Replicates R ricu's sofa_coag logic:
    sofa_coag <- sofa_single("plt", "sofa_coag", function(x) 4L - findInterval(x, c(20, 50, 100, 150)))

    Score based on platelet count (×10³/mm³):
    - 4: < 20
    - 3: 20-49
    - 2: 50-99
    - 1: 100-149
    - 0: >= 150

    Args:
        plt: Platelet count values

    Returns:
        Series with coagulation SOFA scores
    """
    # Replicate R's findInterval logic: findInterval(x, c(20, 50, 100, 150))
    # Returns: 0 if x < 20, 1 if 20 <= x < 50, 2 if 50 <= x < 100, 3 if 100 <= x < 150, 4 if x >= 150
    # Then: score = 4 - findInterval
    # Replicate R's findInterval logic: findInterval(x, c(20, 50, 100, 150))
    # Returns: 0 if x < 20, 1 if 20 <= x < 50, 2 if 50 <= x < 100, 3 if 100 <= x < 150, 4 if x >= 150
    # Then: score = 4 - findInterval
    score = pd.Series(0, index=plt.index, dtype=int)
    
    # Handle NaN values (keep as 0)
    valid_mask = plt.notna()
    
    # Apply findInterval logic
    # x < 20: findInterval = 0, score = 4 - 0 = 4
    score[valid_mask & (plt < 20)] = 4
    
    # 20 <= x < 50: findInterval = 1, score = 4 - 1 = 3
    score[valid_mask & (plt >= 20) & (plt < 50)] = 3
    
    # 50 <= x < 100: findInterval = 2, score = 4 - 2 = 2
    score[valid_mask & (plt >= 50) & (plt < 100)] = 2
    
    # 100 <= x < 150: findInterval = 3, score = 4 - 3 = 1
    score[valid_mask & (plt >= 100) & (plt < 150)] = 1
    
    # x >= 150: findInterval = 4, score = 4 - 4 = 0 (already set)
    
    return score

def sofa_liver(bili: pd.Series) -> pd.Series:
    """Calculate liver SOFA component.
    
    Replicates R ricu's sofa_liver logic:
    sofa_liver <- sofa_single("bili", "sofa_liver", function(x) findInterval(x, c(1.2, 2, 6, 12)))

    Score based on bilirubin (mg/dL):
    - 0: < 1.2
    - 1: 1.2-1.9
    - 2: 2.0-5.9
    - 3: 6.0-11.9
    - 4: >= 12.0

    Args:
        bili: Bilirubin values

    Returns:
        Series with liver SOFA scores
    """
    # Replicate R's findInterval logic: findInterval(x, c(1.2, 2, 6, 12))
    # Returns: 0 if x < 1.2, 1 if 1.2 <= x < 2, 2 if 2 <= x < 6, 3 if 6 <= x < 12, 4 if x >= 12
    score = pd.Series(0, index=bili.index, dtype=int)
    
    # Handle NaN values (keep as 0)
    valid_mask = bili.notna()
    
    # Apply findInterval logic directly
    # x < 1.2: findInterval = 0, score = 0 (already set)
    
    # 1.2 <= x < 2: findInterval = 1, score = 1
    score[valid_mask & (bili >= 1.2) & (bili < 2.0)] = 1
    
    # 2 <= x < 6: findInterval = 2, score = 2
    score[valid_mask & (bili >= 2.0) & (bili < 6.0)] = 2
    
    # 6 <= x < 12: findInterval = 3, score = 3
    score[valid_mask & (bili >= 6.0) & (bili < 12.0)] = 3
    
    # x >= 12: findInterval = 4, score = 4
    score[valid_mask & (bili >= 12.0)] = 4
    
    return score

def sofa_cardio(
    map: pd.Series,
    dopa60: Optional[pd.Series] = None,
    norepi60: Optional[pd.Series] = None,
    dobu60: Optional[pd.Series] = None,
    epi60: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate cardiovascular SOFA component.
    
    Replicates R ricu's fifelse chain logic with proper priority:
    - 4: dopa > 15 | epi > 0.1 | norepi > 0.1
    - 3: dopa > 5 | (epi > 0 & epi <= 0.1) | (norepi > 0 & norepi <= 0.1)
    - 2: (dopa > 0 & dopa <= 5) | dobu > 0
    - 1: map < 70
    - 0: otherwise

    IMPORTANT: Vasopressor rates MUST be in μg/kg/min
    
    Unit conversion requirements:
    - If your data is in μg/min (not weight-adjusted), divide by patient weight (kg)
    - If your data is in mg/h, convert: (mg/h * 1000 μg/mg) / (60 min/h * weight_kg)
    - Common MIMIC-IV sources (inputevents) may store rates in various units
    
    The thresholds (e.g., dopa > 15, norepi > 0.1) assume μg/kg/min.
    Incorrect units will cause severe scoring errors.

    Args:
        map: Mean arterial pressure values (mmHg)
        dopa60: Dopamine rate (μg/kg/min) - REQUIRED UNIT
        norepi60: Norepinephrine rate (μg/kg/min) - REQUIRED UNIT
        dobu60: Dobutamine rate (μg/kg/min) - REQUIRED UNIT
        epi60: Epinephrine rate (μg/kg/min) - REQUIRED UNIT

    Returns:
        Series with cardiovascular SOFA scores
        
    Warnings:
        This function does NOT perform unit conversion. Ensure vasopressor
        rates are standardized to μg/kg/min before calling this function.
        Use pyricu.unit_conversion or concept-level callbacks for standardization.
    """
    # Default zero for missing vasopressor data
    dopa = dopa60.fillna(0) if dopa60 is not None else pd.Series(0, index=map.index)
    norepi = norepi60.fillna(0) if norepi60 is not None else pd.Series(0, index=map.index)
    dobu = dobu60.fillna(0) if dobu60 is not None else pd.Series(0, index=map.index)
    epi = epi60.fillna(0) if epi60 is not None else pd.Series(0, index=map.index)
    
    # Chain of fifelse (if-else) with priority from highest to lowest
    # Score 4: highest priority
    score = pd.Series(0, index=map.index, dtype=int)
    score[SeriesUtils.is_true((dopa > 15) | (epi > 0.1) | (norepi > 0.1))] = 4
    
    # Score 3: second priority (only if not already 4)
    mask3 = SeriesUtils.is_true((dopa > 5) | ((epi > 0) & (epi <= 0.1)) | ((norepi > 0) & (norepi <= 0.1)))
    score[mask3 & (score != 4)] = 3
    
    # Score 2: third priority (only if not already 3 or 4)
    mask2 = SeriesUtils.is_true(((dopa > 0) & (dopa <= 5)) | (dobu > 0))
    score[mask2 & (score < 3)] = 2
    
    # Score 1: lowest priority (only if not already 2, 3, or 4)
    mask1 = SeriesUtils.is_true(map < 70)
    score[mask1 & (score < 2)] = 1
    
    return score

def sofa_cns(gcs: pd.Series) -> pd.Series:
    """Calculate CNS SOFA component.

    Score based on Glasgow Coma Score:
    - 0: 15
    - 1: 13-14
    - 2: 10-12
    - 3: 6-9
    - 4: < 6

    Args:
        gcs: Glasgow Coma Score values

    Returns:
        Series with CNS SOFA scores
    """
    g = pd.to_numeric(gcs, errors="coerce")
    score = pd.Series(np.nan, index=g.index, dtype=float)
    valid = g.notna()
    score.loc[valid] = 0

    # Only apply thresholds where gcs is not NaN
    mask = g < 6
    score[mask] = 4

    # Score 3: 6-9
    mask = (g >= 6) & (g < 10)
    score[mask] = 3
    
    # Score 2: 10-12
    mask = (g >= 10) & (g < 13)
    score[mask] = 2
    
    # Score 1: 13-14
    mask = (g >= 13) & (g < 15)
    score[mask] = 1
    
    return score

def sofa_renal(
    crea: pd.Series,
    urine24: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate renal SOFA component.
    
    Replicates R ricu's sofa_renal fifelse chain logic:
    - 4: cre >= 5 | uri < 200
    - 3: (cre >= 3.5 & cre < 5) | uri < 500
    - 2: cre >= 2 & cre < 3.5
    - 1: cre >= 1.2 & cre < 2
    - 0: otherwise (when at least one input is non-NA)
    
    CRITICAL: In ricu, sofa_renal is calculated using collect_dots with merge_dat=TRUE,
    which means it only produces output at time points where at least one of the input
    concepts (crea or urine24) has data. The score itself follows fifelse logic that
    returns 0L as the final else, but this is only executed when there IS data.

    Args:
        crea: Creatinine values (mg/dL)
        urine24: 24-hour urine output values (mL/day, optional)

    Returns:
        Series with renal SOFA scores (0-4, with 0 as baseline when data exists)
    """
    
    # Convert to numeric, preserving NaN
    crea_num = pd.to_numeric(crea, errors="coerce")
    
    # Handle urine24: only use if provided
    if urine24 is not None:
        uri_num = pd.to_numeric(urine24, errors="coerce")
    else:
        # If not provided, create Series of NaN (won't match any threshold)
        uri_num = pd.Series(np.nan, index=crea.index)
    
    score = pd.Series(0, index=crea.index, dtype=int)

    # Score 1: crea >= 1.2 & crea < 2 (only based on crea)
    mask1 = SeriesUtils.is_true((crea_num >= 1.2) & (crea_num < 2))
    score[mask1] = 1
    
    # Score 2: crea >= 2 & crea < 3.5 (only based on crea)
    mask2 = SeriesUtils.is_true((crea_num >= 2) & (crea_num < 3.5))
    score[mask2] = 2
    
    # Score 3: (crea >= 3.5 & crea < 5) OR urine24 < 500
    mask_crea_3 = SeriesUtils.is_true((crea_num >= 3.5) & (crea_num < 5))
    mask_uri_3 = SeriesUtils.is_true(uri_num < 500)
    score[mask_crea_3 | mask_uri_3] = 3
    
    # Score 4: crea >= 5 OR urine24 < 200
    mask_crea_4 = SeriesUtils.is_true(crea_num >= 5)
    mask_uri_4 = SeriesUtils.is_true(uri_num < 200)
    score[mask_crea_4 | mask_uri_4] = 4
    
    return score

def sofa2_renal(
    crea: pd.Series,
    rrt: Optional[pd.Series] = None,
    rrt_criteria: Optional[pd.Series] = None,
    uo_6h: Optional[pd.Series] = None,
    uo_12h: Optional[pd.Series] = None,
    uo_24h: Optional[pd.Series] = None,
    potassium: Optional[pd.Series] = None,
    ph: Optional[pd.Series] = None,
    bicarb: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate SOFA-2 renal component with accurate duration-based urine output scoring.
    
    SOFA-2 renal scoring (2025 version):
    - Score 0: Creatinine ≤1.20 mg/dL
    - Score 1: Creatinine >1.20, ≤2.0 mg/dL OR UO <0.5 mL/kg/h for 6-12h
    - Score 2: Creatinine >2.0, ≤3.5 mg/dL OR UO <0.5 mL/kg/h for ≥12h
    - Score 3: Creatinine >3.5 mg/dL OR UO <0.3 mL/kg/h for ≥24h OR anuria (0 mL) for ≥12h
    - Score 4: Receiving RRT OR meets RRT criteria
    
    RRT criteria (SOFA-2 footnote p):
    - Creatinine > 1.2 mg/dL OR oliguria (<0.3 mL/kg/h) for >6 hours
    - PLUS at least one of:
      * Serum potassium ≥ 6.0 mmol/L
      * Metabolic acidosis: pH ≤ 7.20 AND HCO3 ≤ 12 mmol/L
    
    Args:
        crea: Serum creatinine (mg/dL)
        rrt: Boolean - currently receiving RRT
        rrt_criteria: Boolean - meets RRT criteria but not receiving it
        uo_6h: 6-hour average urine output (mL/kg/h)
        uo_12h: 12-hour average urine output (mL/kg/h)
        uo_24h: 24-hour average urine output (mL/kg/h)
        potassium: Serum potassium (mmol/L) - for RRT criteria
        ph: Arterial pH - for RRT criteria
        bicarb: Serum bicarbonate (mmol/L) - for RRT criteria
        
    Returns:
        Series with SOFA-2 renal scores (0-4)
    """
    crea_num = pd.to_numeric(crea, errors="coerce")
    score = pd.Series(0, index=crea.index, dtype=int)
    
    # Convert urine output parameters
    uo_6h_num = pd.to_numeric(uo_6h, errors="coerce") if uo_6h is not None else pd.Series(np.nan, index=crea.index)
    uo_12h_num = pd.to_numeric(uo_12h, errors="coerce") if uo_12h is not None else pd.Series(np.nan, index=crea.index)
    uo_24h_num = pd.to_numeric(uo_24h, errors="coerce") if uo_24h is not None else pd.Series(np.nan, index=crea.index)
    
    # Score 1: Creatinine >1.2, ≤2.0 OR UO <0.5 mL/kg/h for 6-12h
    # UO for 6-12h means: 6h average <0.5 BUT 12h average ≥0.5 (or missing)
    mask1_crea = _is_true_safe((crea_num > 1.2) & (crea_num <= 2.0))
    mask1_uo = _is_true_safe(uo_6h_num < 0.5) & ~_is_true_safe(uo_12h_num < 0.5)
    score[mask1_crea | mask1_uo] = 1
    
    # Score 2: Creatinine >2.0, ≤3.5 OR UO <0.5 mL/kg/h for ≥12h
    # UO for ≥12h means: 12h average <0.5
    mask2_crea = _is_true_safe((crea_num > 2.0) & (crea_num <= 3.5))
    mask2_uo = _is_true_safe(uo_12h_num < 0.5)
    score[mask2_crea | mask2_uo] = 2
    
    # Score 3: Creatinine >3.5 OR UO <0.3 mL/kg/h for ≥24h OR anuria for ≥12h
    mask3_crea = _is_true_safe(crea_num > 3.5)
    mask3_uo_24h = _is_true_safe(uo_24h_num < 0.3)  # UO <0.3 for ≥24h
    mask3_anuria_12h = _is_true_safe(uo_12h_num == 0)  # Anuria for ≥12h
    score[mask3_crea | mask3_uo_24h | mask3_anuria_12h] = 3
    
    # Score 4: RRT or meets RRT criteria
    mask4_rrt = _is_true_safe(rrt) if rrt is not None else pd.Series(False, index=crea.index)
    mask4_rrt_crit = _is_true_safe(rrt_criteria) if rrt_criteria is not None else pd.Series(False, index=crea.index)
    
    # If we have potassium and pH/bicarb data, check RRT criteria
    if not mask4_rrt_crit.any() and potassium is not None and ph is not None and bicarb is not None:
        k_num = pd.to_numeric(potassium, errors="coerce")
        ph_num = pd.to_numeric(ph, errors="coerce")
        hco3_num = pd.to_numeric(bicarb, errors="coerce")
        
        # Base kidney injury: crea > 1.2 OR oliguria (<0.3 mL/kg/h for >6h)
        # Use 6h average as proxy for "oliguria for >6h"
        base_injury = _is_true_safe(crea_num > 1.2) | _is_true_safe(uo_6h_num < 0.3)
        
        # Electrolyte/acid-base crisis
        hyperkalemia = _is_true_safe(k_num >= 6.0)
        acidosis = _is_true_safe((ph_num <= 7.20) & (hco3_num <= 12))
        
        # Meets RRT criteria
        mask4_rrt_crit = base_injury & (hyperkalemia | acidosis)
    
    # Apply score 4
    score[mask4_rrt | mask4_rrt_crit] = 4
    
    return score

def sofa2_resp(
    pafi: pd.Series,
    spo2: Optional[pd.Series] = None,
    fio2: Optional[pd.Series] = None,
    adv_resp: Optional[pd.Series] = None,
    ecmo: Optional[pd.Series] = None,
    ecmo_indication: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate SOFA-2 respiratory component.
    
    SOFA-2 respiratory scoring (2025 version):
    - Score 4: P/F ≤75 with advanced support OR ECMO (respiratory indication)
    - Score 3: P/F ≤150 with advanced support
    - Score 2: P/F ≤225
    - Score 1: P/F ≤300
    - Score 0: P/F >300
    
    Advanced support required for scores 3-4 (unless unavailable/ceiling of treatment).
    If PaO2/FiO2 unavailable, use SpO2/FiO2 with adjusted cutoffs.
    
    Args:
        pafi: PaO2/FiO2 ratio
        spo2: SpO2 (oxygen saturation) for SaFi calculation
        fio2: FiO2 (fraction of inspired oxygen) for SaFi calculation
        adv_resp: Advanced respiratory support (IMV/NIV/HFNC/CPAP/BiPAP)
        ecmo: ECMO in use
        ecmo_indication: ECMO indication ('respiratory' or 'cardiovascular')
    
    Returns:
        Series with SOFA-2 respiratory scores (0-4)
    """
    pf = pd.to_numeric(pafi, errors="coerce")
    idx = pf.index
    score = pd.Series(0, index=idx, dtype=int)
    
    # Calculate SaFi if PaFi unavailable
    safi = None
    if spo2 is not None and fio2 is not None:
        s = pd.to_numeric(spo2, errors="coerce")
        f = pd.to_numeric(fio2, errors="coerce")
        # Only use SaFi when SpO2 < 98% (per SOFA-2 footnote f)
        safi = s / f
        safi[s >= 98] = np.nan
    
    # Use SaFi if PaFi unavailable (with adjusted cutoffs per footnote f)
    # SaFi cutoffs: >300 / ≤300 / ≤250 / ≤200 with vent / ≤120 with vent or ECMO
    ratio = pf.copy()
    use_safi = pf.isna() & (safi is not None and safi.notna())
    if use_safi.any() and safi is not None:
        ratio[use_safi] = safi[use_safi]
    
    # Check for advanced respiratory support
    on_adv_resp = _is_true_safe(adv_resp) if adv_resp is not None else pd.Series(False, index=idx)
    
    # Check for ECMO (respiratory indication auto-scores 4)
    on_ecmo_resp = pd.Series(False, index=idx)
    if ecmo is not None and ecmo_indication is not None:
        on_ecmo = _is_true_safe(ecmo)
        is_resp_indication = (ecmo_indication == 'respiratory')
        on_ecmo_resp = on_ecmo & is_resp_indication
    elif ecmo is not None:
        # Default ECMO to respiratory if indication unknown
        on_ecmo_resp = _is_true_safe(ecmo)
    
    # Score 1: P/F ≤300 (or SaFi ≤300)
    score[ratio <= 300] = 1
    
    # Score 2: P/F ≤225 (or SaFi ≤250)
    mask2 = (ratio <= 225) if safi is None else ((ratio <= 225) | ((use_safi) & (ratio <= 250)))
    score[mask2] = 2
    
    # Score 3: P/F ≤150 (or SaFi ≤200) WITH advanced support
    mask3_ratio = (ratio <= 150) if safi is None else ((ratio <= 150) | ((use_safi) & (ratio <= 200)))
    mask3 = mask3_ratio & on_adv_resp
    score[mask3] = 3
    
    # Score 4: P/F ≤75 (or SaFi ≤120) WITH advanced support OR ECMO (respiratory)
    mask4_ratio = (ratio <= 75) if safi is None else ((ratio <= 75) | ((use_safi) & (ratio <= 120)))
    mask4 = (mask4_ratio & on_adv_resp) | on_ecmo_resp
    score[mask4] = 4
    
    return score

def sofa2_coag(plt: pd.Series) -> pd.Series:
    """Calculate SOFA-2 coagulation component.
    
    SOFA-2 hemostasis scoring (updated thresholds from SOFA-1):
    - Score 4: platelets ≤50
    - Score 3: platelets ≤80  (was ≤50 in SOFA-1)
    - Score 2: platelets ≤100 (was ≤100 in SOFA-1)
    - Score 1: platelets ≤150 (was ≤150 in SOFA-1)
    - Score 0: platelets >150
    
    Args:
        plt: Platelet count (×10³/μL)
        
    Returns:
        Series with SOFA-2 coagulation scores (0-4)
    """
    p = pd.to_numeric(plt, errors="coerce")
    score = pd.Series(0, index=p.index, dtype=int)
    
    valid = p.notna()
    score[valid & (p <= 150)] = 1
    score[valid & (p <= 100)] = 2
    score[valid & (p <= 80)] = 3
    score[valid & (p <= 50)] = 4
    
    return score

def sofa2_liver(bili: pd.Series) -> pd.Series:
    """Calculate SOFA-2 liver component.
    
    SOFA-2 liver scoring (relaxed 1-point threshold):
    - Score 4: bilirubin >12 mg/dL (>205 μmol/L)
    - Score 3: bilirubin ≤12, >6.0 mg/dL (≤205, >102.6 μmol/L)
    - Score 2: bilirubin ≤6.0, >3.0 mg/dL (≤102.6, >51.3 μmol/L)
    - Score 1: bilirubin ≤3.0, >1.2 mg/dL (≤51.3, >20.6 μmol/L) - RELAXED from ≤1.9 in SOFA-1
    - Score 0: bilirubin ≤1.2 mg/dL (≤20.6 μmol/L)
    
    Args:
        bili: Total bilirubin (mg/dL)
        
    Returns:
        Series with SOFA-2 liver scores (0-4)
    """
    b = pd.to_numeric(bili, errors="coerce")
    score = pd.Series(0, index=b.index, dtype=int)
    
    valid = b.notna()
    # Use findInterval-like logic: check thresholds in order
    score[valid & (b > 1.2)] = 1  # Relaxed threshold (was 1.9 in SOFA-1)
    score[valid & (b > 3.0)] = 2  # SOFA-2 new intermediate threshold
    score[valid & (b > 6.0)] = 3  # Fixed: should be > 6.0, not >= 6.0
    score[valid & (b > 12.0)] = 4
    
    return score

def sofa2_cardio(
    map: pd.Series,
    norepi60: Optional[pd.Series] = None,
    epi60: Optional[pd.Series] = None,
    dopa60: Optional[pd.Series] = None,
    dobu60: Optional[pd.Series] = None,
    other_vaso: Optional[pd.Series] = None,
    mech_circ_support: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate SOFA-2 cardiovascular component.
    
    SOFA-2 cardiovascular scoring (combined catecholamine dosing):
    - Score 4: High-dose vasopressor (NE+Epi >0.4 μg/kg/min) 
               OR medium-dose + other vaso/inotrope
               OR mechanical circulatory support
    - Score 3: Medium-dose (NE+Epi 0.2-0.4) OR low-dose + other vaso/inotrope
    - Score 2: Low-dose (NE+Epi ≤0.2) OR any other vaso/inotrope
    - Score 1: MAP <70 mmHg, no vasopressors
    - Score 0: MAP ≥70 mmHg
    
    Dopamine used only if NE/Epi unavailable (equipotency: >40 = high, >20 = medium, ≤20 = low).
    
    Args:
        map: Mean arterial pressure (mmHg)
        norepi60: Norepinephrine dose (μg/kg/min)
        epi60: Epinephrine dose (μg/kg/min)
        dopa60: Dopamine dose (μg/kg/min, backup)
        dobu60: Dobutamine dose (μg/kg/min)
        other_vaso: Other vasopressors/inotropes (vasopressin/phenylephrine/milrinone)
        mech_circ_support: Mechanical circulatory support (IABP/LVAD/Impella/VA-ECMO)
        
    Returns:
        Series with SOFA-2 cardiovascular scores (0-4)
    """
    map_num = pd.to_numeric(map, errors="coerce")
    idx = map_num.index
    score = pd.Series(0, index=idx, dtype=int)
    
    # Convert vasopressor doses to numeric
    ne = pd.to_numeric(norepi60, errors="coerce").fillna(0) if norepi60 is not None else pd.Series(0, index=idx)
    epi = pd.to_numeric(epi60, errors="coerce").fillna(0) if epi60 is not None else pd.Series(0, index=idx)
    dopa = pd.to_numeric(dopa60, errors="coerce").fillna(0) if dopa60 is not None else pd.Series(0, index=idx)
    dobu = pd.to_numeric(dobu60, errors="coerce").fillna(0) if dobu60 is not None else pd.Series(0, index=idx)
    
    # SOFA-2: Combined norepinephrine + epinephrine dose
    combined_cate = ne + epi
    
    # Check for other vasopressors/inotropes
    has_other_vaso = _is_true_safe(other_vaso) | (dobu > 0) if other_vaso is not None else (dobu > 0)
    
    # Check for mechanical support
    has_mech_support = _is_true_safe(mech_circ_support) if mech_circ_support is not None else pd.Series(False, index=idx)
    
    # Score 1: MAP <70, no vasopressors
    score[map_num < 70] = 1
    
    # Use dopamine only if NE/Epi not available
    use_dopamine = (combined_cate == 0) & (dopa > 0)
    
    # Score 2: Low-dose vasopressor OR any other vaso/inotrope
    low_dose_cate = (combined_cate > 0) & (combined_cate <= 0.2)
    low_dose_dopa = use_dopamine & (dopa <= 20)
    score[low_dose_cate | low_dose_dopa | has_other_vaso] = 2
    
    # Score 3: Medium-dose OR low-dose + other vaso
    medium_dose_cate = (combined_cate > 0.2) & (combined_cate <= 0.4)
    medium_dose_dopa = use_dopamine & (dopa > 20) & (dopa <= 40)
    score[medium_dose_cate | medium_dose_dopa | (low_dose_cate & has_other_vaso)] = 3
    
    # Score 4: High-dose OR medium-dose + other vaso OR mechanical support
    high_dose_cate = combined_cate > 0.4
    high_dose_dopa = use_dopamine & (dopa > 40)
    score[high_dose_cate | high_dose_dopa | (medium_dose_cate & has_other_vaso) | has_mech_support] = 4
    
    return score

def sofa2_cns(
    gcs: pd.Series,
    delirium_tx: Optional[pd.Series] = None,
    delirium_positive: Optional[pd.Series] = None,
    motor_response: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate SOFA-2 CNS (brain) component.
    
    SOFA-2 brain scoring:
    - Score 4: GCS 3-5 (or extension/no response to pain, myoclonus)
    - Score 3: GCS 6-8 (or flexion to pain)
    - Score 2: GCS 9-12 (or withdrawal to pain)
    - Score 1: GCS 13-14 (or localizing to pain) OR delirium treatment/positive CAM-ICU
    - Score 0: GCS 15 (unless on delirium treatment or positive CAM-ICU)
    
    Special rules:
    - If receiving delirium treatment OR has positive CAM-ICU and GCS=15, score 1 point
    - Motor response scale can substitute full GCS if unavailable
    
    Args:
        gcs: Glasgow Coma Scale (3-15)
        delirium_tx: Receiving delirium treatment (haloperidol, etc.)
        delirium_positive: CAM-ICU assessment positive for delirium (itemid 228332)
        motor_response: GCS Motor component (1-6) for alternative scoring
        
    Returns:
        Series with SOFA-2 CNS scores (0-4)
    """
    # Start with regular GCS
    g = pd.to_numeric(gcs, errors="coerce")
    
    idx = g.index
    score = pd.Series(0, index=idx, dtype=int)
    
    valid = g.notna()
    score[valid & (g >= 13) & (g <= 14)] = 1
    score[valid & (g >= 9) & (g <= 12)] = 2
    score[valid & (g >= 6) & (g <= 8)] = 3
    score[valid & (g >= 3) & (g <= 5)] = 4
    
    # SOFA-2 NEW: Delirium rule
    # If receiving delirium treatment OR has positive CAM-ICU and GCS==15, upgrade to 1pt
    has_delirium = pd.Series(False, index=idx)
    
    if delirium_tx is not None:
        has_delirium = has_delirium | _is_true_safe(delirium_tx)
    
    if delirium_positive is not None:
        has_delirium = has_delirium | _is_true_safe(delirium_positive)
    
    if has_delirium.any():
        mask = (g == 15) & has_delirium
        score[mask] = np.maximum(score[mask], 1)
    
    return score

def sirs_score(
    temp: pd.Series,
    hr: pd.Series,
    resp: pd.Series,
    pco2: Optional[pd.Series] = None,
    wbc: Optional[pd.Series] = None,
    bnd: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate SIRS (Systemic Inflammatory Response Syndrome) score.

    SIRS criteria (score 1 for each, total max 4):
    - Temperature < 36°C or > 38°C
    - Heart rate > 90 bpm
    - Respiratory rate > 20/min OR PaCO2 < 32 mmHg (combined as one criterion)
    - WBC < 4 or > 12 OR >10% bands (combined as one criterion)

    This matches the R ricu implementation where:
    - rspi: resp > 20 | pco2 < 32 → 1 point
    - wbcn: wbc < 4 | wbc > 12 | bnd > 10 → 1 point

    Args:
        temp: Temperature values (°C)
        hr: Heart rate values
        resp: Respiratory rate values
        pco2: PaCO2 values (optional)
        wbc: White blood cell count (optional)
        bnd: Band forms percentage (optional)

    Returns:
        Series with SIRS scores (0-4)
    """
    score = pd.Series(0, index=temp.index, dtype=int)
    
    # Temperature criterion: temp < 36 or temp > 38
    score[(temp < 36) | (temp > 38)] += 1
    
    # Heart rate criterion: hr > 90
    score[hr > 90] += 1
    
    # Respiratory criterion: resp > 20 OR pco2 < 32 (combined as one point)
    # Match R ricu: rspi <- function(re, pa) fifelse(re > 20 | pa < 32, 1L, 0L)
    resp_crit = (resp > 20)
    if pco2 is not None:
        resp_crit = resp_crit | (pco2 < 32)
    score[resp_crit] += 1
    
    # WBC criterion: wbc < 4 OR wbc > 12 OR bnd > 10 (combined as one point)
    # Match R ricu: wbcn <- function(wb, ba) fifelse(wb < 4 | wb > 12 | ba > 10, 1L, 0L)
    wbc_crit = pd.Series(False, index=temp.index)
    if wbc is not None:
        wbc_crit = wbc_crit | (wbc < 4) | (wbc > 12)
    if bnd is not None:
        wbc_crit = wbc_crit | (bnd > 10)
    score[wbc_crit] += 1
    
    return score

def qsofa_score(
    sbp: pd.Series,
    resp: pd.Series,
    gcs: pd.Series,
) -> pd.Series:
    """Calculate qSOFA (quick SOFA) score.

    qSOFA criteria (score 1 for each):
    - SBP ≤ 100 mmHg
    - Respiratory rate ≥ 22/min
    - Altered mentation (GCS ≤ 13)

    Args:
        sbp: Systolic blood pressure values
        resp: Respiratory rate values
        gcs: Glasgow Coma Score values

    Returns:
        Series with qSOFA scores (0-3)
    """
    score = pd.Series(0, index=sbp.index, dtype=int)
    
    score[sbp <= 100] += 1
    score[resp >= 22] += 1
    score[gcs <= 13] += 1  # ricu uses gcs <= 13 for altered mentation
    
    return score

def apache_ii_score(
    age: pd.Series,
    temp: pd.Series,
    map_val: pd.Series,
    hr: pd.Series,
    resp: pd.Series,
    pao2: Optional[pd.Series] = None,
    ph: Optional[pd.Series] = None,
    na: Optional[pd.Series] = None,
    k: Optional[pd.Series] = None,
    crea: Optional[pd.Series] = None,
    hct: Optional[pd.Series] = None,
    wbc: Optional[pd.Series] = None,
    gcs: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate APACHE II score (simplified version).

    Note: This is a simplified implementation. Full APACHE II requires
    additional chronic health and admission diagnosis information.

    Args:
        age: Age in years
        temp: Temperature (°C)
        map_val: Mean arterial pressure
        hr: Heart rate
        resp: Respiratory rate
        pao2: PaO2 (optional)
        ph: Arterial pH (optional)
        na: Sodium (optional)
        k: Potassium (optional)
        crea: Creatinine (optional)
        hct: Hematocrit (optional)
        wbc: White blood cell count (optional)
        gcs: Glasgow Coma Score (optional)

    Returns:
        Series with APACHE II scores
    """
    score = pd.Series(0, index=age.index, dtype=float)
    
    # Age points
    score[age >= 45] += 2
    score[age >= 55] += 3
    score[age >= 65] += 5
    score[age >= 75] += 6
    
    # Temperature
    score[temp >= 41] += 4
    score[(temp >= 39) & (temp < 41)] += 3
    score[(temp >= 38.5) & (temp < 39)] += 1
    score[(temp >= 36) & (temp < 38.5)] += 0
    score[(temp >= 34) & (temp < 36)] += 1
    score[(temp >= 32) & (temp < 34)] += 2
    score[(temp >= 30) & (temp < 32)] += 3
    score[temp < 30] += 4
    
    # Additional components would be added here...
    # This is a placeholder for demonstration
    
    return score.astype(int)

def news_score(
    resp: pd.Series,
    temp: pd.Series,
    sbp: pd.Series,
    hr: pd.Series,
    o2sat: pd.Series = None,
    spo2: pd.Series = None,
    supp_o2: pd.Series = None,
    avpu: pd.Series = None,
    gcs: pd.Series = None,
    keep_components: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate NEWS (National Early Warning Score).
    
    NEWS is an early warning score system for detecting acute illness deterioration.
    
    Components:
    - Respiratory rate: 8-24 breaths/min
    - Oxygen saturation: 91-100%
    - Temperature: 35-39°C
    - Systolic BP: 90-219 mmHg
    - Heart rate: 40-130 bpm
    - Supplemental O2: Yes/No
    - Consciousness level: A=Alert, V=Voice, P=Pain, U=Unresponsive
    
    Args:
        resp: Respiratory rate
        temp: Temperature (°C)
        sbp: Systolic blood pressure
        hr: Heart rate
        o2sat: Oxygen saturation (%) - alias: spo2
        spo2: Oxygen saturation (%) - alias for o2sat
        supp_o2: Supplemental oxygen (boolean), optional
        avpu: AVPU score (A/V/P/U), optional
        gcs: Glasgow Coma Scale (alternative to AVPU), optional
        keep_components: Return individual component scores
        
    Returns:
        NEWS total score or DataFrame with components
    """
    # Handle parameter aliases
    if o2sat is None and spo2 is not None:
        o2sat = spo2
    
    # Create default values if not provided
    if o2sat is None:
        o2sat = pd.Series(100, index=resp.index)
    if supp_o2 is None:
        supp_o2 = pd.Series(False, index=resp.index)
    if avpu is None and gcs is None:
        avpu = pd.Series("A", index=resp.index)
    elif avpu is None and gcs is not None:
        # Convert GCS to AVPU (rough approximation)
        avpu = pd.Series("A", index=gcs.index)
        avpu[gcs < 15] = "V"
        avpu[gcs < 13] = "P"
        avpu[gcs < 9] = "U"
    # Initialize component scores
    resp_score = pd.Series(0, index=resp.index, dtype=int)
    resp_score[resp <= 8] = 3
    resp_score[(resp > 8) & (resp <= 11)] = 1
    resp_score[(resp > 11) & (resp <= 20)] = 0
    resp_score[(resp > 20) & (resp <= 24)] = 2
    resp_score[resp > 24] = 3
    
    o2sat_score = pd.Series(0, index=o2sat.index, dtype=int)
    o2sat_score[o2sat <= 91] = 3
    o2sat_score[(o2sat > 91) & (o2sat <= 93)] = 2
    o2sat_score[(o2sat > 93) & (o2sat <= 95)] = 1
    o2sat_score[o2sat > 95] = 0
    
    temp_score = pd.Series(0, index=temp.index, dtype=int)
    temp_score[temp <= 35] = 3
    temp_score[(temp > 35) & (temp <= 36)] = 1
    temp_score[(temp > 36) & (temp <= 38.4)] = 0
    temp_score[(temp > 38.4) & (temp <= 39)] = 1
    temp_score[temp > 39] = 2
    
    sbp_score = pd.Series(0, index=sbp.index, dtype=int)
    sbp_score[sbp <= 90] = 3
    sbp_score[(sbp > 90) & (sbp <= 100)] = 2
    sbp_score[(sbp > 100) & (sbp <= 110)] = 1
    sbp_score[(sbp > 110) & (sbp <= 219)] = 0
    sbp_score[sbp > 219] = 3
    
    hr_score = pd.Series(0, index=hr.index, dtype=int)
    hr_score[hr <= 40] = 3
    hr_score[(hr > 40) & (hr <= 50)] = 1
    hr_score[(hr > 50) & (hr <= 90)] = 0
    hr_score[(hr > 90) & (hr <= 110)] = 1
    hr_score[(hr > 110) & (hr <= 130)] = 2
    hr_score[hr > 130] = 3
    
    supp_o2_score = supp_o2.apply(lambda x: 2 if x else 0).astype(int)
    
    avpu_score = avpu.apply(lambda x: 0 if x == "A" else 3).astype(int)
    
    # Calculate total
    news_total = (
        resp_score + o2sat_score + temp_score + sbp_score +
        hr_score + supp_o2_score + avpu_score
    )
    
    if keep_components:
        return pd.DataFrame({
            "news": news_total,
            "resp_comp": resp_score,
            "o2sat_comp": o2sat_score,
            "temp_comp": temp_score,
            "sbp_comp": sbp_score,
            "hr_comp": hr_score,
            "supp_o2_comp": supp_o2_score,
            "avpu_comp": avpu_score
        })
    
    return news_total

def mews_score(
    sbp: pd.Series,
    hr: pd.Series,
    resp: pd.Series,
    temp: pd.Series,
    avpu: pd.Series = None,
    gcs: pd.Series = None,
    keep_components: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate MEWS (Modified Early Warning Score).
    
    MEWS is a simplified early warning score system.
    
    Components:
    - Systolic BP: 70-199 mmHg
    - Heart rate: 40-129 bpm
    - Respiratory rate: 9-29 breaths/min
    - Temperature: 35-38.4°C
    - Consciousness level: A=0, V=1, P=2, U=3
    
    Args:
        sbp: Systolic blood pressure
        hr: Heart rate
        resp: Respiratory rate
        temp: Temperature (°C)
        avpu: AVPU score (A/V/P/U), optional
        gcs: Glasgow Coma Scale (alternative to AVPU), optional
        keep_components: Return individual component scores
        
    Returns:
        MEWS total score or DataFrame with components
    """
    # Handle missing consciousness assessment
    if avpu is None and gcs is None:
        avpu = pd.Series("A", index=sbp.index)
    elif avpu is None and gcs is not None:
        # Convert GCS to AVPU
        avpu = pd.Series("A", index=gcs.index)
        avpu[gcs < 15] = "V"
        avpu[gcs < 13] = "P"
        avpu[gcs < 9] = "U"
    # Systolic BP score
    sbp_score = pd.Series(0, index=sbp.index, dtype=int)
    sbp_score[sbp <= 70] = 3
    sbp_score[(sbp > 70) & (sbp <= 80)] = 2
    sbp_score[(sbp > 80) & (sbp <= 100)] = 1
    sbp_score[(sbp > 100) & (sbp <= 199)] = 0
    sbp_score[sbp > 199] = 2
    
    # Heart rate score
    hr_score = pd.Series(0, index=hr.index, dtype=int)
    hr_score[hr <= 40] = 2
    hr_score[(hr > 40) & (hr <= 50)] = 1
    hr_score[(hr > 50) & (hr <= 100)] = 0
    hr_score[(hr > 100) & (hr <= 110)] = 1
    hr_score[(hr > 110) & (hr <= 129)] = 2
    hr_score[hr > 129] = 3
    
    # Respiratory rate score
    resp_score = pd.Series(0, index=resp.index, dtype=int)
    resp_score[resp <= 9] = 2
    resp_score[(resp > 9) & (resp <= 14)] = 0
    resp_score[(resp > 14) & (resp <= 20)] = 1
    resp_score[(resp > 20) & (resp <= 29)] = 2
    resp_score[resp > 29] = 3
    
    # Temperature score
    temp_score = pd.Series(0, index=temp.index, dtype=int)
    temp_score[temp <= 35] = 2
    temp_score[(temp > 35) & (temp <= 38.4)] = 0
    temp_score[temp > 38.4] = 2
    
    # AVPU score
    avpu_map = {"A": 0, "V": 1, "P": 2, "U": 3}
    avpu_score = avpu.map(avpu_map).fillna(0).astype(int)
    
    # Calculate total
    mews_total = sbp_score + hr_score + resp_score + temp_score + avpu_score
    
    if keep_components:
        return pd.DataFrame({
            "mews": mews_total,
            "sbp_comp": sbp_score,
            "hr_comp": hr_score,
            "resp_comp": resp_score,
            "temp_comp": temp_score,
            "avpu_comp": avpu_score
        })
    
    return mews_total

def ts_to_win_tbl(win_dur):
    """
    Convert time series table to window table by adding duration variable.
    
    This is a higher-order function that returns a callback function.
    The returned function adds a duration column and marks the table as window-based.
    
    Args:
        win_dur: Window duration (e.g., from mins(1))
        
    Returns:
        Callback function that transforms the data
    """
    def callback(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Add duration column to data."""
        # Add duration column
        result = data.copy()
        result['dur_var'] = win_dur
        return result
    
    return callback

def pafi(
    po2: pd.DataFrame,
    fio2: pd.DataFrame,
    match_win: pd.Timedelta = pd.Timedelta(hours=2),
    mode: str = "match_vals",
    fix_na_fio2: bool = True,
    database: str = None,
) -> pd.DataFrame:
    """Calculate PaO2/FiO2 ratio (P/F ratio) from oxygen partial pressure and FiO2.
    
    The P/F ratio is a key indicator of respiratory function used in SOFA scoring.
    This function matches PaO2 and FiO2 measurements within a time window and
    calculates the ratio.
    
    Args:
        po2: DataFrame with PaO2 (partial pressure of oxygen) measurements
        fio2: DataFrame with FiO2 (fraction of inspired oxygen) measurements
        match_win: Time window for matching PaO2 and FiO2 (default: 2 hours)
        mode: Matching mode ('match_vals', 'extreme_vals', or 'fill_gaps')
        fix_na_fio2: If True, assume FiO2=21% (room air) when missing
        
    Returns:
        DataFrame with pafi column (PaO2/FiO2 * 100)
        
    Examples:
        >>> # Load PaO2 and FiO2 data
        >>> po2_data = load_concept('po2', 'mimic_demo')
        >>> fio2_data = load_concept('fio2', 'mimic_demo')
        >>> pf_ratio = pafi(po2_data, fio2_data)
    """
    # Identify ID and time columns
    id_cols = [col for col in po2.columns if 'id' in col.lower() and col in fio2.columns]
    time_cols = [col for col in po2.columns if 'time' in col.lower() and col in fio2.columns]
    
    if not time_cols:
        raise ValueError("No time column found in input DataFrames")
    
    time_col = time_cols[0]
    
    # Identify value columns (non-ID, non-time)
    po2_val_col = [col for col in po2.columns if col not in id_cols + time_cols][0]
    fio2_val_col = [col for col in fio2.columns if col not in id_cols + time_cols][0]
    
    # Prepare data
    po2_df = po2[id_cols + [time_col, po2_val_col]].copy()
    fio2_df = fio2[id_cols + [time_col, fio2_val_col]].copy()
    
    po2_df = po2_df.rename(columns={po2_val_col: 'po2'})
    fio2_df = fio2_df.rename(columns={fio2_val_col: 'fio2'})

    # 新增：数据库自适应的FiO2单位标准化
    if database is not None:
        fio2_df = _standardize_fio2_units(fio2_df, database)

    # Merge based on mode
    if mode == "match_vals":
        # 使用 left join 而不是 inner join，保留所有 po2 数据
        # 这样即使 fio2 缺失，也能在后面填充为 21（室内空气）
        result = pd.merge(po2_df, fio2_df, on=id_cols + [time_col], how='left')
    
    elif mode == "extreme_vals":
        # Match within window, take extreme values
        result = pd.merge_asof(
            po2_df.sort_values(id_cols + [time_col]),
            fio2_df.sort_values(id_cols + [time_col]),
            on=time_col,
            by=id_cols,
            tolerance=match_win,
            direction='nearest'
        )
    
    elif mode == "fill_gaps":
        # Forward fill FiO2 values
        result = pd.merge_asof(
            po2_df.sort_values(id_cols + [time_col]),
            fio2_df.sort_values(id_cols + [time_col]),
            on=time_col,
            by=id_cols,
            direction='backward'
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Fix missing FiO2 (assume room air = 21%)
    if fix_na_fio2:
        result.loc[result['fio2'].isna(), 'fio2'] = 21.0
    
    # Convert to numeric, handling None and string values
    result['po2'] = pd.to_numeric(result['po2'], errors='coerce')
    result['fio2'] = pd.to_numeric(result['fio2'], errors='coerce')

    # Remove rows with missing or invalid data
    result = result[
        result['po2'].notna() &
        result['fio2'].notna() &
        (result['fio2'] > 0)  # Use > 0 instead of != 0 for safety
    ]

    # Calculate P/F ratio with safe division
    result['pafi'] = 100 * result['po2'] / result['fio2']
    
    # Remove intermediate columns
    result = result.drop(columns=['po2', 'fio2'])
    
    return result

def safi(
    o2sat: pd.DataFrame,
    fio2: pd.DataFrame,
    match_win: pd.Timedelta = pd.Timedelta(hours=2),
    mode: str = "match_vals",
    fix_na_fio2: bool = True,
    database: str = None,
) -> pd.DataFrame:
    """Calculate SaO2/FiO2 ratio (S/F ratio) from oxygen saturation and FiO2.
    
    The S/F ratio is an alternative to P/F ratio when arterial blood gas is not
    available. It uses pulse oximetry (SpO2) instead of PaO2.
    
    Args:
        o2sat: DataFrame with O2 saturation (SpO2) measurements
        fio2: DataFrame with FiO2 (fraction of inspired oxygen) measurements
        match_win: Time window for matching O2sat and FiO2 (default: 2 hours)
        mode: Matching mode ('match_vals', 'extreme_vals', or 'fill_gaps')
        fix_na_fio2: If True, assume FiO2=21% (room air) when missing
        
    Returns:
        DataFrame with safi column (SaO2/FiO2 * 100)
        
    Examples:
        >>> # Load SpO2 and FiO2 data
        >>> spo2_data = load_concept('o2sat', 'mimic_demo')
        >>> fio2_data = load_concept('fio2', 'mimic_demo')
        >>> sf_ratio = safi(spo2_data, fio2_data)
    """
    # Identify ID and time columns
    id_cols = [col for col in o2sat.columns if 'id' in col.lower() and col in fio2.columns]
    time_cols = [col for col in o2sat.columns if 'time' in col.lower() and col in fio2.columns]
    
    if not time_cols:
        raise ValueError("No time column found in input DataFrames")
    
    time_col = time_cols[0]
    
    # Identify value columns (non-ID, non-time)
    o2sat_val_col = [col for col in o2sat.columns if col not in id_cols + time_cols][0]
    fio2_val_col = [col for col in fio2.columns if col not in id_cols + time_cols][0]
    
    # Prepare data
    o2sat_df = o2sat[id_cols + [time_col, o2sat_val_col]].copy()
    fio2_df = fio2[id_cols + [time_col, fio2_val_col]].copy()
    
    o2sat_df = o2sat_df.rename(columns={o2sat_val_col: 'o2sat'})
    fio2_df = fio2_df.rename(columns={fio2_val_col: 'fio2'})

    # 新增：数据库自适应的FiO2单位标准化
    if database is not None:
        fio2_df = _standardize_fio2_units(fio2_df, database)

    # Merge based on mode
    if mode == "match_vals":
        # 使用 left join 而不是 inner join，保留所有 o2sat 数据
        # 这样即使 fio2 缺失，也能在后面填充为 21（室内空气）
        result = pd.merge(o2sat_df, fio2_df, on=id_cols + [time_col], how='left')
    
    elif mode == "extreme_vals":
        # Match within window, take extreme values
        result = pd.merge_asof(
            o2sat_df.sort_values(id_cols + [time_col]),
            fio2_df.sort_values(id_cols + [time_col]),
            on=time_col,
            by=id_cols,
            tolerance=match_win,
            direction='nearest'
        )
    
    elif mode == "fill_gaps":
        # Forward fill FiO2 values
        result = pd.merge_asof(
            o2sat_df.sort_values(id_cols + [time_col]),
            fio2_df.sort_values(id_cols + [time_col]),
            on=time_col,
            by=id_cols,
            direction='backward'
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Fix missing FiO2 (assume room air = 21%)
    if fix_na_fio2:
        result.loc[result['fio2'].isna(), 'fio2'] = 21.0
    
    # Convert to numeric, handling None and string values
    result['o2sat'] = pd.to_numeric(result['o2sat'], errors='coerce')
    result['fio2'] = pd.to_numeric(result['fio2'], errors='coerce')

    # Remove rows with missing or invalid data
    result = result[
        result['o2sat'].notna() &
        result['fio2'].notna() &
        (result['fio2'] > 0)  # Use > 0 instead of != 0 for safety
    ]

    # Calculate S/F ratio with safe division
    result['safi'] = 100 * result['o2sat'] / result['fio2']
    
    # Remove intermediate columns
    result = result.drop(columns=['o2sat', 'fio2'])
    
    return result

def uo_6h(urine: pd.DataFrame, weight: pd.DataFrame, interval: pd.Timedelta = None) -> pd.DataFrame:
    """Calculate 6-hour average urine output in mL/kg/h.
    
    This function computes a rolling 6-hour average of urine output, normalized
    by patient weight, for SOFA-2 renal scoring.
    
    SOFA-2 criterion: UO <0.5 mL/kg/h for 6-12 hours → Score 1
    
    Args:
        urine: DataFrame with urine output (mL)
        weight: DataFrame with patient weight (kg)
        interval: Time interval for data (if None, inferred from data)
        
    Returns:
        DataFrame with 'uo_6h' column (mL/kg/h)
    """
    return _urine_window_avg(urine, weight, window_hours=6, min_hours=3, interval=interval)

def uo_12h(urine: pd.DataFrame, weight: pd.DataFrame, interval: pd.Timedelta = None) -> pd.DataFrame:
    """Calculate 12-hour average urine output in mL/kg/h.
    
    This function computes a rolling 12-hour average of urine output, normalized
    by patient weight, for SOFA-2 renal scoring.
    
    SOFA-2 criteria:
    - UO <0.5 mL/kg/h for ≥12 hours → Score 2
    - Anuria (0 mL) for ≥12 hours → Score 4
    
    Args:
        urine: DataFrame with urine output (mL)
        weight: DataFrame with patient weight (kg)
        interval: Time interval for data (if None, inferred from data)
        
    Returns:
        DataFrame with 'uo_12h' column (mL/kg/h)
    """
    return _urine_window_avg(urine, weight, window_hours=12, min_hours=6, interval=interval)

def uo_24h(urine: pd.DataFrame, weight: pd.DataFrame, interval: pd.Timedelta = None) -> pd.DataFrame:
    """Calculate 24-hour average urine output in mL/kg/h.
    
    This function computes a rolling 24-hour average of urine output, normalized
    by patient weight, for SOFA-2 renal scoring.
    
    SOFA-2 criterion: UO <0.3 mL/kg/h for ≥24 hours → Score 3
    
    Args:
        urine: DataFrame with urine output (mL)
        weight: DataFrame with patient weight (kg)
        interval: Time interval for data (if None, inferred from data)
        
    Returns:
        DataFrame with 'uo_24h' column (mL/kg/h)
    """
    return _urine_window_avg(urine, weight, window_hours=24, min_hours=12, interval=interval)

def _urine_window_avg(
    urine: pd.DataFrame,
    weight: pd.DataFrame,
    window_hours: int,
    min_hours: int,
    interval: pd.Timedelta = None
) -> pd.DataFrame:
    """Internal function to calculate windowed average urine output.
    
    Args:
        urine: DataFrame with urine output (mL)
        weight: DataFrame with patient weight (kg)
        window_hours: Window size in hours
        min_hours: Minimum hours of data required in window
        interval: Time interval for data
        
    Returns:
        DataFrame with averaged urine output (mL/kg/h)
    """
    # Determine ID and time columns
    id_cols = [col for col in urine.columns if col.endswith('_id') or col in ['stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']]
    
    # 不同数据库使用不同的时间列名
    if 'charttime' in urine.columns:
        time_col = 'charttime'
    elif 'measuredat' in urine.columns:
        time_col = 'measuredat'
    elif 'nursingchartoffset' in urine.columns:
        time_col = 'nursingchartoffset'
    elif 'observationoffset' in urine.columns:
        time_col = 'observationoffset'
    else:
        # 尝试找到任何时间相关的列
        time_like_cols = [col for col in urine.columns if 'time' in col.lower() or 'offset' in col.lower()]
        if time_like_cols:
            time_col = time_like_cols[0]
        else:
            raise ValueError(f"Cannot find time column in urine data. Available columns: {urine.columns.tolist()}")
    
    val_col = 'urine'
    result_col = f'uo_{window_hours}h'
    
    # 如果没有找到ID列，尝试常见的ID列名
    if not id_cols:
        for potential_id in ['admissionid', 'stay_id', 'patientunitstayid', 'patientid', 'icustay_id']:
            if potential_id in urine.columns:
                id_cols = [potential_id]
                break
    
    # Merge urine and weight data
    if id_cols:
        # 检查weight表是否有相同的ID列
        # 对于AUMC，urine有admissionid但weight只有patientid
        # 需要先join admissions表获取admissionid->patientid映射
        common_ids = [col for col in id_cols if col in weight.columns]
        if common_ids:
            merged = pd.merge(urine, weight, on=common_ids, how='left', suffixes=('', '_weight'))
        else:
            # ID列不匹配，尝试广播weight（假设只有一个患者）
            merged = urine.copy()
            if 'weight' in weight.columns and len(weight) > 0:
                merged['weight'] = weight['weight'].iloc[0]
    else:
        # 如果还是没有ID列，直接使用urine数据（假设只有一个患者）
        merged = urine.copy()
        if 'weight' in weight.columns and len(weight) > 0:
            merged['weight'] = weight['weight'].iloc[0]
    
    # Handle weight time column
    if 'charttime_weight' in merged.columns:
        # Use forward fill to propagate weight values
        merged = merged.sort_values(id_cols + [time_col])
        for id_col in id_cols:
            merged['weight'] = merged.groupby(id_col)['weight'].ffill()
    
    # Remove rows without weight
    merged = merged[merged['weight'].notna() & (merged['weight'] > 0)]
    
    if len(merged) == 0:
        # Return empty DataFrame with correct structure
        result = urine[id_cols + [time_col]].copy()
        result[result_col] = np.nan
        return result
    
    # Sort by ID and time
    merged = merged.sort_values(id_cols + [time_col])
    
    # Infer interval if not provided
    if interval is None:
        # Try to infer from data
        if len(merged) > 1:
            time_diffs = merged.groupby(id_cols)[time_col].diff()
            valid_diffs = time_diffs[time_diffs.notna() & (time_diffs > pd.Timedelta(0))]
            if len(valid_diffs) > 0:
                interval = valid_diffs.median()
            else:
                interval = pd.Timedelta(hours=1)
        else:
            interval = pd.Timedelta(hours=1)
    
    # Calculate window parameters (use lowercase 'h' to avoid deprecation warning)
    window_str = f'{window_hours}h'  # Pandas rolling window format
    min_periods = max(1, int(min_hours / (interval.total_seconds() / 3600)))
    
    # PERFORMANCE FIX: Use simple mean instead of complex rolling window
    # For SOFA scoring, we just need to know if average UO over the period meets threshold
    # This is much faster than time-based rolling windows
    
    # Simple approach: Calculate average UO per patient over recent window
    # This is an approximation but much faster for large datasets
    def calc_uo_rate_fast(group):
        # Sort by time
        group = group.sort_values(time_col)
        
        # Use backward-looking average over last N measurements
        # Approximate window_hours by number of measurements
        n_measurements = max(1, int(window_hours / (interval.total_seconds() / 3600)))
        
        # Rolling sum of urine (count-based, much faster than time-based)
        urine_sum = group[val_col].rolling(
            window=n_measurements,
            min_periods=max(1, min_periods),
            closed='right'
        ).sum()
        
        # Use most recent weight (forward fill already done)
        weight_val = group['weight']
        
        # Calculate rate: total_mL / (kg * hours) = mL/kg/h
        rate = urine_sum / (weight_val * window_hours)
        
        group[result_col] = rate
        return group
    
    result = merged.groupby(id_cols, group_keys=False).apply(
        calc_uo_rate_fast,
        include_groups=True,
    )

    # 如果未来的 pandas 默认移除了 ID 列，这里仍兜底补齐
    if id_cols:
        missing_cols = [col for col in id_cols if col not in result.columns]
        if missing_cols:
            ids_df = merged[id_cols].reset_index(drop=True)
            result = result.reset_index(drop=True)
            for col in missing_cols:
                result[col] = ids_df[col]
    
    # Keep only relevant columns
    keep_cols = id_cols + [time_col, result_col]
    result = result[[col for col in keep_cols if col in result.columns]]

    return result

def miiv_icu_patients_filter(data: Union[IdTbl, pd.DataFrame], **kwargs) -> Union[IdTbl, pd.DataFrame]:
    """Filter MIMIC-IV patients data to only include ICU patients.

    This callback connects patients table with icustays table to ensure
    that demographic data (age, sex) only includes ICU patients,
    matching the ID system used by other concepts (stay_id).

    Args:
        data: DataFrame from patients table with subject_id and demographic data
        **kwargs: Additional keyword arguments (database name, etc.)

    Returns:
        DataFrame filtered to ICU patients with stay_id as primary ID
    """
    from .datasource import ICUDataSource

    # Get database name from kwargs or default to miiv
    database = kwargs.get('database', 'miiv')

    if database != 'miiv':
        # For non-MIMIC-IV databases, return data unchanged
        return data

    try:
        # Get ICUDataSource instance
        ds = ICUDataSource.get_instance(database)

        # Load icustays table to get mapping between subject_id and stay_id
        icustays = ds.load_table('icustays', columns=['stay_id', 'subject_id'])

        if icustays.empty:
            # If no icustays data, return empty DataFrame with expected structure
            return pd.DataFrame(columns=['stay_id'] + [col for col in data.columns if col != 'subject_id'])

        # Merge patients data with icustays to filter only ICU patients
        # Convert subject_id to same type for proper merging
        if 'subject_id' in data.columns and 'subject_id' in icustays.columns:
            data_copy = data.copy()
            icustays_copy = icustays.copy()

            # Ensure both subject_id columns are the same type
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

            return merged
        else:
            # If expected columns not found, return original data
            return data

    except Exception:
        # If any error occurs during filtering, return original data
        # This ensures the system doesn't break if icustays table is unavailable
        return data
