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
from typing import Optional, List, Dict, Any


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
    if pd.isna(lactate) or pd.isna(map_value):
        raise ValueError("circulatory failure requires measured lactate and MAP")
    if any(pd.isna(value) for value in (norepi_rate, epi_rate, vaso_rate)):
        raise ValueError("circulatory failure drug-rate evidence is incomplete")
    if pd.isna(level1_drug_present):
        raise ValueError("circulatory failure Level-1 drug evidence is incomplete")

    # Check lactate criterion
    lactate_elevated = lactate >= LACTATE_THRESHOLD
    
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
    map_low = map_value <= MAP_THRESHOLD
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
    
    if df.empty:
        for column in (
            'lactate_elevated', 'map_low', 'level1_drugs', 'level2_drugs',
            'level3_drugs', 'circ_failure',
        ):
            df[column] = pd.Series(dtype="boolean")
        df['circ_event'] = pd.Series(dtype="Int64")
        return df

    # Core criteria are required. Treating a missing lactate or MAP column as
    # "not elevated / not low" silently under-detects circulatory failure.
    if lactate_col is None or lactate_col not in df.columns:
        raise ValueError("circulatory failure requires a lactate column")
    if map_col is None or map_col not in df.columns:
        raise ValueError("circulatory failure requires a MAP column")

    def nullable_flag(source: pd.Series, comparison: pd.Series) -> pd.Series:
        return comparison.astype("boolean").mask(source.isna(), pd.NA)

    # Calculate component conditions while preserving row-level unknowns.
    df['lactate_elevated'] = nullable_flag(
        df[lactate_col], df[lactate_col] >= LACTATE_THRESHOLD
    )
    df['map_low'] = nullable_flag(
        df[map_col], df[map_col] <= MAP_THRESHOLD
    )
    df['level1_drugs'] = pd.Series(False, index=df.index, dtype="boolean")
    df['level2_drugs'] = pd.Series(False, index=df.index, dtype="boolean")
    df['level3_drugs'] = pd.Series(False, index=df.index, dtype="boolean")

    evidence_columns = [lactate_col, map_col]
        
    # Level 1 drugs
    if level1_cols:
        for col in level1_cols:
            if col in df.columns:
                evidence_columns.append(col)
                df['level1_drugs'] = df['level1_drugs'] | nullable_flag(
                    df[col], df[col] > 0
                )
    
    # Level 2 drugs (0 < norepi/epi < 0.1)
    if norepi_rate_col and norepi_rate_col in df.columns:
        evidence_columns.append(norepi_rate_col)
        df['level2_drugs'] = df['level2_drugs'] | nullable_flag(
            df[norepi_rate_col],
            (df[norepi_rate_col] > 0)
            & (df[norepi_rate_col] < NOREPI_EPI_LEVEL2_THRESHOLD),
        )
    if epi_rate_col and epi_rate_col in df.columns:
        evidence_columns.append(epi_rate_col)
        df['level2_drugs'] = df['level2_drugs'] | nullable_flag(
            df[epi_rate_col],
            (df[epi_rate_col] > 0)
            & (df[epi_rate_col] < NOREPI_EPI_LEVEL2_THRESHOLD),
        )
    
    # Level 3 drugs (norepi/epi ≥ 0.1 OR vasopressin)
    if norepi_rate_col and norepi_rate_col in df.columns:
        df['level3_drugs'] = df['level3_drugs'] | nullable_flag(
            df[norepi_rate_col],
            df[norepi_rate_col] >= NOREPI_EPI_LEVEL2_THRESHOLD,
        )
    if epi_rate_col and epi_rate_col in df.columns:
        df['level3_drugs'] = df['level3_drugs'] | nullable_flag(
            df[epi_rate_col],
            df[epi_rate_col] >= NOREPI_EPI_LEVEL2_THRESHOLD,
        )
    if vaso_rate_col and vaso_rate_col in df.columns:
        evidence_columns.append(vaso_rate_col)
        df['level3_drugs'] = df['level3_drugs'] | nullable_flag(
            df[vaso_rate_col], df[vaso_rate_col] > 0
        )

    # A present-but-missing input is unknown, not evidence of absence.  Columns
    # not supplied by the caller remain outside this row contract; callers that
    # possess a drug stream must pass it explicitly.
    evidence_columns = list(dict.fromkeys(evidence_columns))
    df['_input_complete'] = df[evidence_columns].notna().all(axis=1)

    # Calculate event levels
    def get_event_level(row):
        if not bool(row['_input_complete']):
            return pd.NA
        if not bool(row['lactate_elevated']):
            return 0
        if bool(row['level3_drugs']):
            return 3
        if bool(row['level2_drugs']):
            return 2
        if bool(row['map_low']) or bool(row['level1_drugs']):
            return 1
        return 0
    
    if use_rolling_window:
        # Apply rolling window per patient.
        window_steps = window_size_minutes // grid_size_minutes

        # Resolve the point-in-time event level first (lactate AND MAP/drugs at
        # the SAME timepoint, per get_event_level), then label each center as
        # Event k when the centered window sustains level >= k for >= 2/3 of its
        # timepoints. This keeps the rolling label consistent with the documented
        # Event 1/2/3 definition and the point-in-time path, instead of
        # decoupling sustained lactate (>= 2/3) from single-point drug presence
        # (the previous logic used `.any()` for drug level, which could promote a
        # window to Event 2/3 from a single drugged timepoint).
        df['_point_event'] = df.apply(get_event_level, axis=1).astype("Int64")

        def apply_rolling_window(group):
            point = group['_point_event'].astype("Int64")
            n = len(point)
            if n < window_steps:
                # Not enough data for a full window: use point assessment.
                group['circ_event'] = point.array
                return group
            half_window = window_steps // 2
            events = pd.array([pd.NA] * n, dtype="Int64")
            for i in range(n):
                start_idx = max(0, i - half_window)
                end_idx = min(n, i + half_window + 1)
                win = point.iloc[start_idx:end_idx]
                if win.isna().any():
                    continue
                values = win.astype(int).to_numpy()
                events[i] = 0
                for k in (3, 2, 1):
                    if (values >= k).mean() >= WINDOW_FRACTION_THRESHOLD:
                        events[i] = k
                        break
            group['circ_event'] = events
            return group

        # 🔧 FIX pandas 3.0: groupby().apply() drops group columns
        _id_backup = df[[id_col]].copy()
        df = df.groupby(id_col, group_keys=False).apply(apply_rolling_window)
        if id_col not in df.columns:
            df[id_col] = _id_backup[id_col].values
        df = df.drop(columns=['_point_event'], errors='ignore')
        df['circ_event'] = df['circ_event'].astype("Int64")
    else:
        # Simple point-in-time assessment
        df['circ_event'] = df.apply(get_event_level, axis=1).astype("Int64")
    
    # Set circ_failure flag
    df['circ_failure'] = df['circ_event'].gt(0).astype("boolean")
    df = df.drop(columns=['_input_complete'], errors='ignore')
    
    return df


def load_circ_failure(
    database: str,
    data_path: Optional[str] = None,
    max_patients: Optional[int] = None,
    patient_ids: Optional[List] = None,
    use_rolling_window: bool = False,  # Default to simple for speed
    verbose: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
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
    from easyicu.api import load_concepts
    
    _pre = preloaded_data or {}
    
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
    core_concepts = ['lact', 'map']
    optional_concepts = [
        'norepi_rate',
        'epi_rate',
        'adh_rate',
        'dobu_rate',
        'dopa_rate',
        'phn_rate',
        'milrinone',
        'levo_rate',
        'theo_rate',
    ]
    all_needed = core_concepts + optional_concepts
    
    if verbose:
        print(f"Loading circulatory failure data for {database}...")
    
    # Batch-load all concepts not already in preloaded_data
    to_load = [c for c in all_needed if c not in _pre or not isinstance(_pre.get(c), pd.DataFrame) or _pre[c].empty]
    
    loaded_dfs = dict(_pre)  # start from preloaded
    if to_load:
        try:
            batch_result = load_concepts(
                concepts=to_load,
                database=database,
                data_path=data_path,
                max_patients=max_patients,
                patient_ids=patient_ids,
                verbose=verbose,
                merge=False,
            )
            if isinstance(batch_result, dict):
                for c, cdf in batch_result.items():
                    if hasattr(cdf, 'data'):
                        cdf = cdf.data
                    if isinstance(cdf, pd.DataFrame) and not cdf.empty:
                        loaded_dfs[c] = cdf
        except Exception:
            # Fallback: load one by one
            for c in to_load:
                try:
                    r = load_concepts(concepts=[c], database=database, data_path=data_path,
                                      max_patients=max_patients, patient_ids=patient_ids, verbose=False)
                    if isinstance(r, pd.DataFrame) and not r.empty:
                        loaded_dfs[c] = r
                except Exception:
                    pass
    
    # Normalize time columns to 'charttime' (merge=False returns raw column names
    # like 'measuredat_minutes' / 'start' / 'starttime' that differ from merge=True's 'charttime')
    _time_aliases = ['measuredat_minutes', 'measuredat', 'datetime',
                     'observationoffset', 'Offset', 'starttime', 'start',
                     'givenat', 'enteredentryat']
    for _cname in list(loaded_dfs.keys()):
        _cdf = loaded_dfs[_cname]
        if 'charttime' not in _cdf.columns:
            for _alias in _time_aliases:
                if _alias in _cdf.columns:
                    loaded_dfs[_cname] = _cdf.rename(columns={_alias: 'charttime'})
                    break
    
    # Core concepts are non-optional; a silent partial load would score
    # circulatory failure without its lactate or MAP evidence.
    missing_core = [c for c in core_concepts if c not in loaded_dfs]
    if missing_core:
        raise ValueError(
            "circulatory failure could not load required core concepts: "
            + ", ".join(missing_core)
        )

    # Build merged dataframe from core concepts
    core_dfs = [loaded_dfs[c] for c in core_concepts]
    if not core_dfs:
        if verbose:
            print("No core data loaded")
        return pd.DataFrame()
    
    # Merge core concepts
    df = core_dfs[0]
    id_cols = [c for c in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID'] if c in df.columns]
    time_cols = [c for c in ['charttime', 'datetime', 'measuredat', 'measuredat_minutes', 'observationoffset', 'start'] if c in df.columns]
    merge_cols = id_cols + time_cols
    
    for cdf in core_dfs[1:]:
        mcols = [c for c in merge_cols if c in cdf.columns]
        if mcols:
            df = pd.merge(df, cdf[mcols + [c for c in cdf.columns if c not in mcols and c not in df.columns]],
                         on=mcols, how='outer')
    
    if df.empty:
        return df
    
    # Merge optional concepts
    loaded_optional = []
    for concept in optional_concepts:
        if concept in loaded_dfs:
            opt_df = loaded_dfs[concept]
            if concept in opt_df.columns:
                mcols = [c for c in merge_cols if c in opt_df.columns]
                if mcols:
                    opt_df = opt_df[mcols + [concept]]
                    df = pd.merge(df, opt_df, on=mcols, how='left')
                    loaded_optional.append(concept)
    
    if verbose and loaded_optional:
        print(f"Loaded optional concepts: {loaded_optional}")
    
    if verbose:
        print(f"Loaded {len(df)} rows for {df[id_col].nunique()} patients")
    
    # Determine time column
    time_col = 'charttime'
    if time_col not in df.columns:
        time_candidates = ['datetime', 'measuredat', 'measuredat_minutes', 'observationoffset', 'start', 'time']
        for col in time_candidates:
            if col in df.columns:
                time_col = col
                break
    
    # Map column names
    lactate_col = 'lact' if 'lact' in df.columns else ('lac' if 'lac' in df.columns else 'lactate')
    map_col = 'map' if 'map' in df.columns else None
    if map_col is None:
        raise ValueError("circulatory failure requires a MAP column")
    
    # Find drug columns (based on what was loaded)
    norepi_col = 'norepi_rate' if 'norepi_rate' in df.columns else None
    epi_col = 'epi_rate' if 'epi_rate' in df.columns else None
    vaso_col = 'adh_rate' if 'adh_rate' in df.columns else None
    level1_cols = [
        c
        for c in [
            'dobu_rate', 'dopa_rate', 'phn_rate', 'milrinone', 'levo_rate',
            'theo_rate',
        ]
        if c in df.columns
    ]
    
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
        print("\nCirculatory failure event distribution:")
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
    
    # Get first event per patient. Sort by time first so the level always
    # describes the same row as the minimum time.
    events = events.sort_values(time_col)
    first_events = events.groupby(id_col, sort=False).agg({
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
