"""
Circulatory Failure Assessment Module (circEWS Definition)

This module implements the circulatory failure definition from:
    Hyland, S.L. et al. Early prediction of circulatory failure in the 
    intensive care unit using machine learning. Nat Med (2020).
    https://doi.org/10.1038/s41591-020-0789-4

Circulatory failure is defined based on three components:
1. Elevated lactate (≥ 2 mmol/L)
2. Low mean arterial pressure (MAP ≤ 65 mmHg)
3. Vasopressor/inotrope use at different levels

Event Classification:
- Event 0: Stable (no circulatory failure)
- Event 1: Lactate ≥ 2 AND (MAP ≤ 65 OR Level 1 drugs)
- Event 2: Lactate ≥ 2 AND Level 2 drugs (0 < norepi/epi < 0.1 μg/kg/min)
- Event 3: Lactate ≥ 2 AND Level 3 drugs (norepi/epi ≥ 0.1 μg/kg/min OR vasopressin)

Drug Levels:
- Level 1: dobutamine, milrinone, levosimendan, theophylline, dopamine, phenylephrine
- Level 2: norepinephrine or epinephrine between 0 and 0.1 μg/kg/min
- Level 3: norepinephrine or epinephrine ≥ 0.1 μg/kg/min, or any vasopressin
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import warnings


# ============================================================================
# Constants
# ============================================================================

# Thresholds
LACTATE_THRESHOLD = 2.0  # mmol/L
MAP_THRESHOLD = 65.0  # mmHg
NOREPI_EPI_LEVEL2_THRESHOLD = 0.1  # μg/kg/min

# Window parameters (from circEWS paper)
DEFAULT_WINDOW_SIZE_MINUTES = 45
DEFAULT_GRID_SIZE_MINUTES = 5
WINDOW_FRACTION_THRESHOLD = 2.0 / 3.0  # ≥ 2/3 of window

# Drug categories for each database
LEVEL1_DRUGS = {
    'miiv': ['dobu', 'mili', 'dopa', 'phenyl'],  # dobutamine, milrinone, dopamine, phenylephrine
    'eicu': ['dobu', 'mili', 'dopa', 'phenyl'],
    'aumc': ['dobu', 'mili', 'dopa', 'levo'],  # levosimendan available in AUMC
    'hirid': ['dobu', 'mili', 'dopa', 'levo', 'theo'],  # theophylline in HiRID
    'mimic': ['dobu', 'mili', 'dopa', 'phenyl'],
    'sic': ['dobu', 'mili', 'dopa'],
}

LEVEL2_3_DRUGS = {
    'norepi': 'norepi_rate',  # norepinephrine rate (μg/kg/min)
    'epi': 'epi_rate',  # epinephrine rate (μg/kg/min)
    'vaso': 'vaso_rate',  # vasopressin rate
}


# ============================================================================
# Core Functions
# ============================================================================

def circ_failure_event(
    lactate: float,
    map_value: float,
    norepi_rate: float = 0.0,
    epi_rate: float = 0.0,
    vaso_rate: float = 0.0,
    level1_drug_present: bool = False,
) -> int:
    """
    Determine circulatory failure event level for a single time point.
    
    Parameters
    ----------
    lactate : float
        Lactate level in mmol/L
    map_value : float
        Mean arterial pressure in mmHg
    norepi_rate : float
        Norepinephrine infusion rate in μg/kg/min
    epi_rate : float
        Epinephrine infusion rate in μg/kg/min
    vaso_rate : float
        Vasopressin infusion rate (any units, >0 = present)
    level1_drug_present : bool
        Whether any Level 1 drug (dobutamine, milrinone, etc.) is present
        
    Returns
    -------
    int
        Event level: 0 (stable), 1, 2, or 3 (most severe)
    """
    # Check lactate criterion
    lactate_elevated = lactate >= LACTATE_THRESHOLD if pd.notna(lactate) else False
    
    if not lactate_elevated:
        # Without elevated lactate, check if MAP/drugs indicate potential instability
        # but cannot confirm circulatory failure → Event 0
        return 0
    
    # Lactate is elevated, now check drug levels
    
    # Level 3: norepi/epi ≥ 0.1 μg/kg/min OR any vasopressin
    if (pd.notna(norepi_rate) and norepi_rate >= NOREPI_EPI_LEVEL2_THRESHOLD) or \
       (pd.notna(epi_rate) and epi_rate >= NOREPI_EPI_LEVEL2_THRESHOLD) or \
       (pd.notna(vaso_rate) and vaso_rate > 0):
        return 3
    
    # Level 2: 0 < norepi/epi < 0.1 μg/kg/min
    if (pd.notna(norepi_rate) and 0 < norepi_rate < NOREPI_EPI_LEVEL2_THRESHOLD) or \
       (pd.notna(epi_rate) and 0 < epi_rate < NOREPI_EPI_LEVEL2_THRESHOLD):
        return 2
    
    # Level 1: MAP ≤ 65 OR Level 1 drugs present
    map_low = map_value <= MAP_THRESHOLD if pd.notna(map_value) else False
    if map_low or level1_drug_present:
        return 1
    
    # Elevated lactate but no other criteria → technically unstable
    # Following circEWS: if MAP > 65 and no drugs, still Event 0
    return 0


def calculate_circ_failure_status(
    df: pd.DataFrame,
    id_col: str = 'stay_id',
    time_col: str = 'charttime',
    lactate_col: str = 'lact',
    map_col: str = 'map',
    norepi_rate_col: Optional[str] = 'norepi_rate',
    epi_rate_col: Optional[str] = 'epi_rate',
    vaso_rate_col: Optional[str] = 'vaso_rate',
    level1_cols: Optional[List[str]] = None,
    window_size_minutes: int = DEFAULT_WINDOW_SIZE_MINUTES,
    grid_size_minutes: int = DEFAULT_GRID_SIZE_MINUTES,
    use_rolling_window: bool = True,
) -> pd.DataFrame:
    """
    Calculate circulatory failure status for a patient dataframe.
    
    This implements the rolling window approach from circEWS:
    - Uses a centered 45-minute window (default)
    - Labels time point as Event 1/2/3 if ≥ 2/3 of window meets criteria
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data with lactate, MAP, and drug infusion rates
    id_col : str
        Column name for patient identifier
    time_col : str
        Column name for time
    lactate_col : str
        Column name for lactate values
    map_col : str
        Column name for MAP values
    norepi_rate_col : str, optional
        Column name for norepinephrine rate
    epi_rate_col : str, optional
        Column name for epinephrine rate
    vaso_rate_col : str, optional
        Column name for vasopressin rate
    level1_cols : list, optional
        Column names for Level 1 drugs
    window_size_minutes : int
        Rolling window size in minutes
    grid_size_minutes : int
        Data resampling interval in minutes
    use_rolling_window : bool
        If True, use rolling window. If False, use point-in-time assessment.
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        - circ_event: Event level (0, 1, 2, 3)
        - circ_failure: Boolean (True if event > 0)
        - lactate_elevated: Boolean
        - map_low: Boolean
        - level1_drugs: Boolean
        - level2_drugs: Boolean
        - level3_drugs: Boolean
    """
    df = df.copy()
    
    # Initialize output columns
    df['lactate_elevated'] = False
    df['map_low'] = False
    df['level1_drugs'] = False
    df['level2_drugs'] = False
    df['level3_drugs'] = False
    df['circ_event'] = 0
    df['circ_failure'] = False
    
    if df.empty:
        return df
    
    # Calculate component conditions
    if lactate_col in df.columns:
        df['lactate_elevated'] = df[lactate_col] >= LACTATE_THRESHOLD
        
    if map_col in df.columns:
        df['map_low'] = df[map_col] <= MAP_THRESHOLD
        
    # Level 1 drugs
    if level1_cols:
        for col in level1_cols:
            if col in df.columns:
                df['level1_drugs'] = df['level1_drugs'] | (df[col] > 0)
    
    # Level 2 drugs (0 < norepi/epi < 0.1)
    if norepi_rate_col and norepi_rate_col in df.columns:
        df['level2_drugs'] = df['level2_drugs'] | (
            (df[norepi_rate_col] > 0) & 
            (df[norepi_rate_col] < NOREPI_EPI_LEVEL2_THRESHOLD)
        )
    if epi_rate_col and epi_rate_col in df.columns:
        df['level2_drugs'] = df['level2_drugs'] | (
            (df[epi_rate_col] > 0) & 
            (df[epi_rate_col] < NOREPI_EPI_LEVEL2_THRESHOLD)
        )
    
    # Level 3 drugs (norepi/epi ≥ 0.1 OR vasopressin)
    if norepi_rate_col and norepi_rate_col in df.columns:
        df['level3_drugs'] = df['level3_drugs'] | (
            df[norepi_rate_col] >= NOREPI_EPI_LEVEL2_THRESHOLD
        )
    if epi_rate_col and epi_rate_col in df.columns:
        df['level3_drugs'] = df['level3_drugs'] | (
            df[epi_rate_col] >= NOREPI_EPI_LEVEL2_THRESHOLD
        )
    if vaso_rate_col and vaso_rate_col in df.columns:
        df['level3_drugs'] = df['level3_drugs'] | (df[vaso_rate_col] > 0)
    
    # Calculate event levels
    def get_event_level(row):
        if not row.get('lactate_elevated', False):
            return 0
        if row.get('level3_drugs', False):
            return 3
        if row.get('level2_drugs', False):
            return 2
        if row.get('map_low', False) or row.get('level1_drugs', False):
            return 1
        return 0
    
    if use_rolling_window:
        # Apply rolling window per patient
        window_steps = window_size_minutes // grid_size_minutes
        
        def apply_rolling_window(group):
            if len(group) < window_steps:
                # Not enough data for rolling window, use point assessment
                group['circ_event'] = group.apply(get_event_level, axis=1)
            else:
                # Use rolling window
                # For each position, check if ≥ 2/3 of window meets criteria
                for i in range(len(group)):
                    half_window = window_steps // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(group), i + half_window + 1)
                    
                    window_df = group.iloc[start_idx:end_idx]
                    
                    # Check if MAP or drugs present in ≥ 2/3 of window
                    map_or_drugs = (
                        window_df['map_low'] | 
                        window_df['level1_drugs'] | 
                        window_df['level2_drugs'] | 
                        window_df['level3_drugs']
                    )
                    
                    if map_or_drugs.mean() >= WINDOW_FRACTION_THRESHOLD:
                        # Determine highest drug level in window
                        if window_df['level3_drugs'].any() and window_df['lactate_elevated'].mean() >= WINDOW_FRACTION_THRESHOLD:
                            group.iloc[i, group.columns.get_loc('circ_event')] = 3
                        elif window_df['level2_drugs'].any() and window_df['lactate_elevated'].mean() >= WINDOW_FRACTION_THRESHOLD:
                            group.iloc[i, group.columns.get_loc('circ_event')] = 2
                        elif window_df['lactate_elevated'].mean() >= WINDOW_FRACTION_THRESHOLD:
                            group.iloc[i, group.columns.get_loc('circ_event')] = 1
                        else:
                            group.iloc[i, group.columns.get_loc('circ_event')] = 0
                    else:
                        group.iloc[i, group.columns.get_loc('circ_event')] = 0
                        
            return group
        
        df = df.groupby(id_col, group_keys=False).apply(apply_rolling_window)
    else:
        # Simple point-in-time assessment
        df['circ_event'] = df.apply(get_event_level, axis=1)
    
    # Set circ_failure flag
    df['circ_failure'] = df['circ_event'] > 0
    
    return df


def load_circ_failure(
    database: str,
    data_path: Optional[str] = None,
    max_patients: Optional[int] = None,
    patient_ids: Optional[List] = None,
    use_rolling_window: bool = False,  # Default to simple for speed
    verbose: bool = True,
) -> pd.DataFrame:
    """
    High-level API to load and calculate circulatory failure status.
    
    This function loads the required concepts (lactate, MAP, vasopressors)
    and calculates circulatory failure status according to circEWS definition.
    
    Parameters
    ----------
    database : str
        Database name: 'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic'
    data_path : str, optional
        Path to database files
    max_patients : int, optional
        Maximum number of patients to load
    patient_ids : list, optional
        Specific patient IDs to load
    use_rolling_window : bool
        If True, use 45-min rolling window (slower but more accurate).
        If False, use point-in-time assessment (faster).
    verbose : bool
        Print progress information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with circulatory failure status for each time point
    """
    from pyricu.api import load_concepts
    
    # Determine ID column based on database
    id_col_map = {
        'miiv': 'stay_id',
        'eicu': 'patientunitstayid',
        'aumc': 'admissionid',
        'hirid': 'patientid',
        'mimic': 'icustay_id',
        'sic': 'CaseID',
    }
    id_col = id_col_map.get(database, 'stay_id')
    
    # Concepts to load
    # Core concepts - note: lactate is 'lact' not 'lac'
    core_concepts = ['lact', 'map']  # lactate and MAP
    
    # Vasopressor concepts - try each one individually
    # Note: Not all concepts are available in all databases
    optional_concepts = ['norepi_rate', 'epi_rate', 'dobu_rate', 'dopa_rate']
    
    if verbose:
        print(f"Loading circulatory failure data for {database}...")
    
    # First load core concepts
    try:
        df = load_concepts(
            concepts=core_concepts,
            database=database,
            data_path=data_path,
            max_patients=max_patients,
            patient_ids=patient_ids,
            verbose=verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load core concepts (lact, map): {e}")
    
    if df.empty:
        if verbose:
            print("No core data loaded")
        return df
    
    # Try to load optional concepts one by one
    loaded_optional = []
    for concept in optional_concepts:
        try:
            opt_df = load_concepts(
                concepts=[concept],
                database=database,
                data_path=data_path,
                max_patients=max_patients,
                patient_ids=patient_ids,
                verbose=False,
            )
            if not opt_df.empty and concept in opt_df.columns:
                # Merge with main dataframe
                id_cols = [c for c in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID'] if c in df.columns]
                time_cols = [c for c in ['charttime', 'datetime', 'measuredat', 'observationoffset'] if c in df.columns]
                
                if id_cols and time_cols:
                    merge_cols = id_cols + time_cols
                    # Keep only new concept column
                    opt_df = opt_df[[c for c in merge_cols if c in opt_df.columns] + [concept]]
                    df = pd.merge(df, opt_df, on=[c for c in merge_cols if c in opt_df.columns], how='left')
                    loaded_optional.append(concept)
        except Exception:
            pass  # Ignore missing optional concepts
    
    if verbose and loaded_optional:
        print(f"Loaded optional concepts: {loaded_optional}")
    
    if verbose:
        print(f"Loaded {len(df)} rows for {df[id_col].nunique()} patients")
    
    # Determine time column
    time_col = 'charttime'
    if time_col not in df.columns:
        time_candidates = ['datetime', 'measuredat', 'observationoffset', 'time']
        for col in time_candidates:
            if col in df.columns:
                time_col = col
                break
    
    # Map column names
    lactate_col = 'lact' if 'lact' in df.columns else ('lac' if 'lac' in df.columns else 'lactate')
    map_col = 'map' if 'map' in df.columns else None
    
    # Find drug columns (based on what was loaded)
    norepi_col = 'norepi_rate' if 'norepi_rate' in df.columns else None
    epi_col = 'epi_rate' if 'epi_rate' in df.columns else None
    vaso_col = None  # Not commonly available
    level1_cols = [c for c in ['dobu_rate', 'dopa_rate'] if c in df.columns]
    
    # Calculate circulatory failure status
    result = calculate_circ_failure_status(
        df=df,
        id_col=id_col,
        time_col=time_col,
        lactate_col=lactate_col,
        map_col=map_col,
        norepi_rate_col=norepi_col,
        epi_rate_col=epi_col,
        vaso_rate_col=vaso_col,
        level1_cols=level1_cols,
        use_rolling_window=use_rolling_window,
    )
    
    if verbose:
        event_counts = result['circ_event'].value_counts().sort_index()
        print(f"\nCirculatory failure event distribution:")
        for event, count in event_counts.items():
            pct = 100 * count / len(result)
            print(f"  Event {event}: {count:,} ({pct:.1f}%)")
        
        failure_rate = result['circ_failure'].mean()
        print(f"\nOverall circulatory failure rate: {failure_rate:.1%}")
    
    return result


def summarize_circ_failure(df: pd.DataFrame, id_col: str = 'stay_id') -> pd.DataFrame:
    """
    Generate summary statistics for circulatory failure data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with circulatory failure status
    id_col : str
        Patient identifier column
        
    Returns
    -------
    pd.DataFrame
        Summary statistics including:
        - Total patients
        - Patients with any circulatory failure
        - Event distribution
        - Time in each state
    """
    if df.empty:
        return pd.DataFrame()
    
    # Detect ID column
    if id_col not in df.columns:
        for col in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
            if col in df.columns:
                id_col = col
                break
    
    summary = {
        'total_observations': len(df),
        'total_patients': df[id_col].nunique(),
        'patients_with_circ_failure': df[df['circ_failure']][id_col].nunique() if 'circ_failure' in df.columns else 0,
    }
    
    if 'circ_event' in df.columns:
        for event in [0, 1, 2, 3]:
            count = (df['circ_event'] == event).sum()
            summary[f'event_{event}_count'] = count
            summary[f'event_{event}_pct'] = 100 * count / len(df) if len(df) > 0 else 0
    
    if 'circ_failure' in df.columns:
        summary['circ_failure_rate'] = df['circ_failure'].mean()
        
    return pd.DataFrame([summary])


def get_circ_failure_incidence(
    df: pd.DataFrame,
    id_col: str = 'stay_id',
    time_col: str = 'charttime',
    min_event_level: int = 1,
) -> pd.DataFrame:
    """
    Get the first circulatory failure event for each patient.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with circulatory failure status
    id_col : str
        Patient identifier column
    time_col : str
        Time column
    min_event_level : int
        Minimum event level to consider (1, 2, or 3)
        
    Returns
    -------
    pd.DataFrame
        First circulatory failure time for each patient who had an event
    """
    if df.empty or 'circ_event' not in df.columns:
        return pd.DataFrame()
    
    # Detect columns
    if id_col not in df.columns:
        for col in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
            if col in df.columns:
                id_col = col
                break
                
    if time_col not in df.columns:
        for col in ['charttime', 'datetime', 'measuredat', 'observationoffset']:
            if col in df.columns:
                time_col = col
                break
    
    # Filter to events at or above minimum level
    events = df[df['circ_event'] >= min_event_level].copy()
    
    if events.empty:
        return pd.DataFrame()
    
    # Get first event per patient
    first_events = events.groupby(id_col).agg({
        time_col: 'min',
        'circ_event': 'first',
    }).reset_index()
    
    first_events.columns = [id_col, 'first_circ_failure_time', 'first_event_level']
    
    return first_events


# ============================================================================
# Utility Functions
# ============================================================================

def validate_circ_failure_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate circulatory failure data quality.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with patient data
        
    Returns
    -------
    dict
        Validation results including missing rates and warnings
    """
    results = {
        'valid': True,
        'warnings': [],
        'missing_rates': {},
    }
    
    required_cols = ['lactate_elevated', 'map_low', 'circ_event']
    for col in required_cols:
        if col not in df.columns:
            results['warnings'].append(f"Missing required column: {col}")
            results['valid'] = False
    
    # Check missing rates for key columns
    key_cols = ['lact', 'lac', 'lactate', 'map', 'norepi_rate', 'epi_rate', 'vaso_rate']
    for col in key_cols:
        if col in df.columns:
            missing_rate = df[col].isna().mean()
            results['missing_rates'][col] = missing_rate
            if missing_rate > 0.5:
                results['warnings'].append(f"High missing rate for {col}: {missing_rate:.1%}")
    
    return results


# ============================================================================
# Module-level convenience
# ============================================================================

__all__ = [
    'circ_failure_event',
    'calculate_circ_failure_status',
    'load_circ_failure',
    'summarize_circ_failure',
    'get_circ_failure_incidence',
    'validate_circ_failure_data',
    'LACTATE_THRESHOLD',
    'MAP_THRESHOLD',
    'NOREPI_EPI_LEVEL2_THRESHOLD',
]
