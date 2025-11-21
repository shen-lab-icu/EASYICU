"""Clinical scoring systems (R ricu callback-scores.R).

This module implements various clinical severity scores including:
- SIRS (Systemic Inflammatory Response Syndrome)
- qSOFA (quick Sequential Organ Failure Assessment)
- NEWS (National Early Warning Score)
- MEWS (Modified Early Warning Score)

References:
    R ricu: https://github.com/eth-mds/ricu
"""

from typing import Optional, Literal, Callable
import pandas as pd
import numpy as np

def _locf(series: pd.Series) -> any:
    """Last observation carried forward (R ricu locf)."""
    # Get last non-NA value
    non_na = series.dropna()
    if len(non_na) > 0:
        return non_na.iloc[-1]
    return np.nan

def _map_vals(values: list, breaks: list) -> Callable:
    """Map continuous values to discrete scores based on breaks.
    
    Args:
        values: Score values to assign
        breaks: Break points for binning
        
    Returns:
        Function that maps input to score
    """
    def mapper(x):
        if pd.isna(x):
            return np.nan
        
        # Find appropriate bin
        for i, brk in enumerate(breaks):
            if x <= brk:
                return values[i]
        
        # If greater than all breaks
        return values[-1]
    
    return mapper

def sirs_score(
    temp: pd.DataFrame,
    hr: pd.DataFrame,
    resp: pd.DataFrame,
    wbc: pd.DataFrame,
    pco2: pd.DataFrame,
    bnd: pd.DataFrame,
    id_cols: list,
    index_col: str,
    win_length: pd.Timedelta = pd.Timedelta(hours=24),
    keep_components: bool = False,
) -> pd.DataFrame:
    """Calculate SIRS score (R ricu sirs_score).
    
    The SIRS (Systemic Inflammatory Response Syndrome) score is a commonly used
    assessment tool used to track a patient's well-being in an ICU.
    
    SIRS criteria (≥2 indicates SIRS):
    1. Temperature: <36°C or >38°C (1 point)
    2. Heart rate: >90 bpm (1 point)
    3. Respiratory rate: >20/min or PaCO2 <32 mmHg (1 point)
    4. WBC: <4 or >12 x10^9/L or >10% bands (1 point)
    
    Args:
        temp: Temperature data (°C)
        hr: Heart rate data (bpm)
        resp: Respiratory rate data (/min)
        wbc: White blood cell count (x10^9/L)
        pco2: Partial pressure of CO2 (mmHg)
        bnd: Band neutrophils (%)
        id_cols: ID columns
        index_col: Time index column
        win_length: Window for last observation carried forward
        keep_components: Whether to keep individual component scores
        
    Returns:
        DataFrame with SIRS scores
    """
    from .ts_utils import slide
    
    # Merge all data
    dfs = [temp, hr, resp, wbc, pco2, bnd]
    data = dfs[0].copy()
    
    for df in dfs[1:]:
        data = pd.merge(data, df, on=id_cols + [index_col], how='outer')
    
    data = data.sort_values(id_cols + [index_col])
    
    # Apply LOCF (last observation carried forward) within window
    agg_dict = {}
    for col in ['temp', 'hr', 'resp', 'wbc', 'pco2', 'bnd']:
        if col in data.columns:
            agg_dict[col] = _locf
    
    if agg_dict:
        data = slide(data, id_cols, index_col, before=win_length, 
                    after=pd.Timedelta(0), agg_func=agg_dict)
    
    # Calculate component scores
    def temp_score(x):
        if pd.isna(x):
            return 0
        return 1 if (x < 36 or x > 38) else 0
    
    def hr_score(x):
        if pd.isna(x):
            return 0
        return 1 if x > 90 else 0
    
    def resp_score(resp_val, pco2_val):
        if pd.isna(resp_val) and pd.isna(pco2_val):
            return 0
        if (not pd.isna(resp_val) and resp_val > 20) or \
           (not pd.isna(pco2_val) and pco2_val < 32):
            return 1
        return 0
    
    def wbc_score(wbc_val, bnd_val):
        if pd.isna(wbc_val) and pd.isna(bnd_val):
            return 0
        if (not pd.isna(wbc_val) and (wbc_val < 4 or wbc_val > 12)) or \
           (not pd.isna(bnd_val) and bnd_val > 10):
            return 1
        return 0
    
    # Apply scoring
    data['temp_comp'] = data['temp'].apply(temp_score)
    data['hr_comp'] = data['hr'].apply(hr_score)
    data['resp_comp'] = data.apply(lambda row: resp_score(row.get('resp'), row.get('pco2')), axis=1)
    data['wbc_comp'] = data.apply(lambda row: wbc_score(row.get('wbc'), row.get('bnd')), axis=1)
    
    # Calculate total SIRS score
    component_cols = ['temp_comp', 'hr_comp', 'resp_comp', 'wbc_comp']
    data['sirs'] = data[component_cols].sum(axis=1)
    
    # Clean up
    result_cols = id_cols + [index_col, 'sirs']
    if keep_components:
        result_cols.extend(component_cols)
    else:
        data = data.drop(columns=component_cols, errors='ignore')
    
    # Remove intermediate columns
    data = data.drop(columns=['temp', 'hr', 'resp', 'wbc', 'pco2', 'bnd'], errors='ignore')
    
    return data[result_cols]

def qsofa_score(
    gcs: pd.DataFrame,
    sbp: pd.DataFrame,
    resp: pd.DataFrame,
    id_cols: list,
    index_col: str,
    win_length: pd.Timedelta = pd.Timedelta(hours=24),
    keep_components: bool = False,
) -> pd.DataFrame:
    """Calculate qSOFA score (R ricu qsofa_score).
    
    The qSOFA (quick Sequential Organ Failure Assessment) is a bedside
    prompt that may identify patients at greater risk of poor outcomes.
    
    qSOFA criteria (≥2 suggests sepsis):
    1. GCS ≤13 (1 point)
    2. SBP ≤100 mmHg (1 point)
    3. Respiratory rate ≥22/min (1 point)
    
    Args:
        gcs: Glasgow Coma Scale data
        sbp: Systolic blood pressure data (mmHg)
        resp: Respiratory rate data (/min)
        id_cols: ID columns
        index_col: Time index column
        win_length: Window for last observation carried forward
        keep_components: Whether to keep individual component scores
        
    Returns:
        DataFrame with qSOFA scores
    """
    from .ts_utils import slide
    
    # Merge data
    data = gcs.copy()
    data = pd.merge(data, sbp, on=id_cols + [index_col], how='outer')
    data = pd.merge(data, resp, on=id_cols + [index_col], how='outer')
    data = data.sort_values(id_cols + [index_col])
    
    # Apply LOCF
    agg_dict = {
        'gcs': _locf,
        'sbp': _locf,
        'resp': _locf,
    }
    
    data = slide(data, id_cols, index_col, before=win_length,
                after=pd.Timedelta(0), agg_func=agg_dict)
    
    # Calculate component scores
    data['gcs_comp'] = (data['gcs'] <= 13).astype(int)
    data['sbp_comp'] = (data['sbp'] <= 100).astype(int)
    data['resp_comp'] = (data['resp'] >= 22).astype(int)
    
    # Calculate total qSOFA score
    component_cols = ['gcs_comp', 'sbp_comp', 'resp_comp']
    data['qsofa'] = data[component_cols].sum(axis=1)
    
    # Clean up
    result_cols = id_cols + [index_col, 'qsofa']
    if keep_components:
        result_cols.extend(component_cols)
    
    return data[result_cols]

def news_score(
    hr: pd.DataFrame,
    avpu: pd.DataFrame,
    supp_o2: pd.DataFrame,
    o2sat: pd.DataFrame,
    temp: pd.DataFrame,
    sbp: pd.DataFrame,
    resp: pd.DataFrame,
    id_cols: list,
    index_col: str,
    win_length: pd.Timedelta = pd.Timedelta(hours=24),
    keep_components: bool = False,
) -> pd.DataFrame:
    """Calculate NEWS score (R ricu news_score).
    
    The NEWS (National Early Warning Score) is used to detect clinical
    deterioration in acutely ill patients.
    
    Args:
        hr: Heart rate data (bpm)
        avpu: AVPU consciousness scale (A/V/P/U)
        supp_o2: Supplemental oxygen (boolean)
        o2sat: Oxygen saturation (%)
        temp: Temperature data (°C)
        sbp: Systolic blood pressure (mmHg)
        resp: Respiratory rate (/min)
        id_cols: ID columns
        index_col: Time index column
        win_length: Window for last observation carried forward
        keep_components: Whether to keep individual component scores
        
    Returns:
        DataFrame with NEWS scores
    """
    from .ts_utils import slide
    
    # Merge all data
    dfs = [hr, avpu, supp_o2, o2sat, temp, sbp, resp]
    data = dfs[0].copy()
    
    for df in dfs[1:]:
        data = pd.merge(data, df, on=id_cols + [index_col], how='outer')
    
    # Fill missing supp_o2 with False
    if 'supp_o2' in data.columns:
        data['supp_o2'] = data['supp_o2'].fillna(False)
    
    data = data.sort_values(id_cols + [index_col])
    
    # Apply LOCF
    agg_dict = {}
    for col in ['hr', 'avpu', 'supp_o2', 'o2sat', 'temp', 'sbp', 'resp']:
        if col in data.columns:
            agg_dict[col] = _locf
    
    if agg_dict:
        data = slide(data, id_cols, index_col, before=win_length,
                    after=pd.Timedelta(0), agg_func=agg_dict)
    
    # Define scoring functions (NEWS scoring rules)
    def resp_map(x):
        if pd.isna(x): return 0
        if x <= 8: return 3
        if x <= 11: return 1
        if x <= 20: return 0
        if x <= 24: return 2
        return 3
    
    def o2sat_map(x):
        if pd.isna(x): return 0
        if x <= 91: return 3
        if x <= 93: return 2
        if x <= 95: return 1
        return 0
    
    def temp_map(x):
        if pd.isna(x): return 0
        if x <= 35: return 3
        if x <= 36: return 1
        if x <= 38: return 0
        if x <= 39: return 1
        return 2
    
    def sbp_map(x):
        if pd.isna(x): return 0
        if x <= 90: return 3
        if x <= 100: return 2
        if x <= 110: return 1
        if x <= 219: return 0
        return 3
    
    def hr_map(x):
        if pd.isna(x): return 0
        if x <= 40: return 3
        if x <= 50: return 1
        if x <= 90: return 0
        if x <= 110: return 1
        if x <= 130: return 2
        return 3
    
    def avpu_map(x):
        if pd.isna(x): return 0
        return 0 if x == 'A' else 3
    
    def supp_o2_map(x):
        if pd.isna(x): return 0
        return 2 if x else 0
    
    # Apply scoring
    data['resp_comp'] = data['resp'].apply(resp_map)
    data['o2sat_comp'] = data['o2sat'].apply(o2sat_map)
    data['temp_comp'] = data['temp'].apply(temp_map)
    data['sbp_comp'] = data['sbp'].apply(sbp_map)
    data['hr_comp'] = data['hr'].apply(hr_map)
    data['avpu_comp'] = data['avpu'].apply(avpu_map)
    data['supp_o2_comp'] = data['supp_o2'].apply(supp_o2_map)
    
    # Calculate total NEWS score
    component_cols = ['resp_comp', 'o2sat_comp', 'temp_comp', 'sbp_comp', 
                     'hr_comp', 'avpu_comp', 'supp_o2_comp']
    data['news'] = data[component_cols].sum(axis=1)
    
    # Clean up
    result_cols = id_cols + [index_col, 'news']
    if keep_components:
        result_cols.extend(component_cols)
    
    return data[result_cols]

def mews_score(
    hr: pd.DataFrame,
    avpu: pd.DataFrame,
    temp: pd.DataFrame,
    sbp: pd.DataFrame,
    resp: pd.DataFrame,
    id_cols: list,
    index_col: str,
    win_length: pd.Timedelta = pd.Timedelta(hours=24),
    keep_components: bool = False,
) -> pd.DataFrame:
    """Calculate MEWS score (R ricu mews_score).
    
    The MEWS (Modified Early Warning Score) is used to track patient
    deterioration in acute care settings.
    
    Args:
        hr: Heart rate data (bpm)
        avpu: AVPU consciousness scale (A/V/P/U)
        temp: Temperature data (°C)
        sbp: Systolic blood pressure (mmHg)
        resp: Respiratory rate (/min)
        id_cols: ID columns
        index_col: Time index column
        win_length: Window for last observation carried forward
        keep_components: Whether to keep individual component scores
        
    Returns:
        DataFrame with MEWS scores
    """
    from .ts_utils import slide
    
    # Merge data
    dfs = [hr, avpu, temp, sbp, resp]
    data = dfs[0].copy()
    
    for df in dfs[1:]:
        data = pd.merge(data, df, on=id_cols + [index_col], how='outer')
    
    data = data.sort_values(id_cols + [index_col])
    
    # Apply LOCF
    agg_dict = {}
    for col in ['hr', 'avpu', 'temp', 'sbp', 'resp']:
        if col in data.columns:
            agg_dict[col] = _locf
    
    if agg_dict:
        data = slide(data, id_cols, index_col, before=win_length,
                    after=pd.Timedelta(0), agg_func=agg_dict)
    
    # Define MEWS scoring functions
    def sbp_map(x):
        if pd.isna(x): return 0
        if x <= 70: return 3
        if x <= 80: return 2
        if x <= 100: return 1
        if x <= 199: return 0
        return 2
    
    def hr_map(x):
        if pd.isna(x): return 0
        if x <= 40: return 2
        if x <= 50: return 1
        if x <= 100: return 0
        if x <= 110: return 1
        if x <= 129: return 2
        return 3
    
    def resp_map(x):
        if pd.isna(x): return 0
        if x <= 9: return 2
        if x <= 14: return 0
        if x <= 20: return 1
        if x <= 29: return 2
        return 3
    
    def temp_map(x):
        if pd.isna(x): return 0
        if x <= 35: return 2
        if x <= 38.4: return 0
        return 2
    
    def avpu_map(x):
        if pd.isna(x): return 0
        avpu_scores = {'A': 0, 'V': 1, 'P': 2, 'U': 3}
        return avpu_scores.get(x, 0)
    
    # Apply scoring
    data['sbp_comp'] = data['sbp'].apply(sbp_map)
    data['hr_comp'] = data['hr'].apply(hr_map)
    data['resp_comp'] = data['resp'].apply(resp_map)
    data['temp_comp'] = data['temp'].apply(temp_map)
    data['avpu_comp'] = data['avpu'].apply(avpu_map)
    
    # Calculate total MEWS score
    component_cols = ['sbp_comp', 'hr_comp', 'resp_comp', 'temp_comp', 'avpu_comp']
    data['mews'] = data[component_cols].sum(axis=1)
    
    # Clean up
    result_cols = id_cols + [index_col, 'mews']
    if keep_components:
        result_cols.extend(component_cols)
    
    return data[result_cols]
