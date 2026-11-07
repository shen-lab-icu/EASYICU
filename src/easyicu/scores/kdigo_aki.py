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
4. ETH ICU features (prior-only rolling AKI baseline): https://github.com/eth-mds/icu-features

Author: EasyICU Team
Date: 2026-01-26
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Mapping
import pandas as pd
import numpy as np
import logging

from easyicu.utils.thresholds import REL_EPS, ge, le, lt
from easyicu.urine_weight_linkage import (
    ConflictingKeyedWeightError,
    resolve_keyed_unique_weights,
    resolve_unkeyed_single_entity_weight,
)

logger = logging.getLogger(__name__)


KDIGO_ASCERTAINMENT_STATES = frozenset(
    {
        "positive",
        "negative_complete",
        "partial_no_observed_positive",
        "indeterminate",
    }
)
KDIGO_OBSERVATION_COVERAGE_STATES = frozenset(
    {"complete", "partial", "indeterminate"}
)


class KDIGOComponentError(RuntimeError):
    """A technical component failure that must not masquerade as missing data."""

    def __init__(self, *, component: str, reason_code: str, message: str) -> None:
        self.component = component
        self.reason_code = reason_code
        super().__init__(
            f"{message} (component={component}, reason_code={reason_code})"
        )


class KDIGOComponentLoadError(KDIGOComponentError):
    """A source could not be loaded reliably."""


class KDIGOComponentSchemaError(KDIGOComponentError):
    """A non-empty component did not satisfy the required table contract."""


class KDIGOComponentCalculationError(KDIGOComponentError):
    """A declared component failed while its phenotype was being calculated."""


@dataclass(frozen=True)
class KDIGOComponentSources:
    """Run-level receipt: whether this installation provides each component source.

    ``True`` means the database maps the component and at least one of its
    defined source tables is present, so a stay without rows lacks *patient*
    data.  ``False`` means the source itself is absent.  The receipt describes
    the installation, never the stays that happen to share an extraction
    batch: a batch in which no stay has rows must label those stays exactly as
    a whole-cohort run would (2026-11-06, batch-invariant reason codes).
    """

    creatinine: bool
    urine: bool
    rrt: bool


#: The general concept each KDIGO component input reads.
_COMPONENT_SOURCE_CONCEPTS = (
    ("creatinine", "crea"),
    ("urine", "urine"),
    ("rrt", "rrt"),
)


def _concept_source_present(concept: str, data_source: Any) -> Optional[bool]:
    """Whether ``data_source`` maps ``concept`` and has one of its tables.

    ``None`` means the data source cannot answer (no config, resolver or
    dictionary entry, or a mapping without source tables).
    """

    from easyicu.resources import load_dictionary

    config = getattr(data_source, "config", None)
    resolve = getattr(data_source, "resolve_loader_from_disk", None)
    definition = load_dictionary().get(concept)
    if config is None or resolve is None or definition is None:
        return None
    try:
        sources = definition.for_data_source(config)
    except (AttributeError, TypeError):
        return None
    if not sources:
        return False
    tables = sorted({str(s.table) for s in sources if getattr(s, "table", None)})
    if not tables:
        return None
    return any(resolve(table) is not None for table in tables)


def kdigo_component_sources(data_source: Any) -> Optional[KDIGOComponentSources]:
    """The installation-level source receipt for the KDIGO components.

    Returns ``None`` when any component cannot be decided; callers then keep
    the historical per-call labels instead of guessing.
    """

    flags = {
        component: _concept_source_present(concept, data_source)
        for component, concept in _COMPONENT_SOURCE_CONCEPTS
    }
    if any(flag is None for flag in flags.values()):
        return None
    return KDIGOComponentSources(**{k: bool(v) for k, v in flags.items()})


def kdigo_component_sources_for_path(
    database: str, data_path: Optional[str]
) -> Optional[KDIGOComponentSources]:
    """:func:`kdigo_component_sources` for a registered database at ``data_path``.

    A path that is not an installation directory (an in-memory or stub loader
    carries a nominal path) cannot answer, so the receipt is ``None`` there.
    """

    from pathlib import Path

    if data_path is None or not Path(data_path).is_dir():
        return None
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_data_sources

    try:
        config = load_data_sources().get(database)
    except KeyError:
        return None
    return kdigo_component_sources(ICUDataSource(config, base_path=data_path))


def _component_source_present(
    component: str,
    frame: Optional[pd.DataFrame],
    sources: Optional[KDIGOComponentSources],
) -> bool:
    """Whether ``component``'s source exists, from the receipt when given.

    Without a receipt the historical per-call inference (rows present) is kept
    for direct callers that pass whole-cohort frames; batched extraction must
    pass the receipt.  Rows for a source the receipt declares absent are a
    contract violation and fail loudly.
    """

    has_rows = isinstance(frame, pd.DataFrame) and not frame.empty
    if sources is None:
        return has_rows
    present = bool(getattr(sources, component))
    if has_rows and not present:
        raise KDIGOComponentSchemaError(
            component=component,
            reason_code=f"kdigo_{component}_rows_without_declared_source",
            message=(
                f"{component} rows were supplied although the source receipt "
                "declares the source absent"
            ),
        )
    return present


def _strict_numeric_component(
    values: pd.Series,
    *,
    component: str,
    reason_code: str,
    field_name: str,
) -> pd.Series:
    """Convert declared numeric evidence without turning bad text into missingness."""

    numeric = pd.to_numeric(values, errors="coerce")
    conversion_loss = values.notna() & numeric.isna()
    nonfinite = numeric.notna() & ~np.isfinite(numeric.astype(float))
    if conversion_loss.any() or nonfinite.any():
        raise KDIGOComponentSchemaError(
            component=component,
            reason_code=reason_code,
            message=f"KDIGO {field_name} contains a non-numeric or non-finite value",
        )
    return numeric


def _strict_time_component(
    values: pd.Series,
    *,
    component: str,
    reason_code: str,
) -> pd.Series:
    """Preserve a wholly numeric or wholly datetime time axis, never a mixed one."""

    if pd.api.types.is_datetime64_any_dtype(
        values
    ) or pd.api.types.is_timedelta64_dtype(values):
        return values
    if pd.api.types.is_numeric_dtype(values):
        return _strict_numeric_component(
            values,
            component=component,
            reason_code=reason_code,
            field_name="time axis",
        )
    nonmissing = values.notna()
    numeric = pd.to_numeric(values, errors="coerce")
    if bool(numeric.notna().eq(nonmissing).all()):
        return _strict_numeric_component(
            values,
            component=component,
            reason_code=reason_code,
            field_name="time axis",
        )
    parsed = pd.to_datetime(values, errors="coerce")
    if bool(parsed.notna().eq(nonmissing).all()):
        return parsed
    raise KDIGOComponentSchemaError(
        component=component,
        reason_code=reason_code,
        message="KDIGO time axis contains an invalid or mixed encoding",
    )


def _valid_weight_patient_ids(
    urine_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    *,
    urine_id_col: str,
    weight_col: str,
) -> set[Any]:
    """Return patients with a valid keyed or proven single-entity weight."""

    candidate = weight_col if weight_col in weight_df.columns else _detect_value_col(
        weight_df, "weight"
    )
    if candidate is None:
        return set()
    numeric = _strict_numeric_component(
        weight_df[candidate],
        component="weight",
        reason_code="kdigo_weight_numeric_encoding_invalid",
        field_name="weight",
    )
    valid_weight = weight_df.loc[numeric.gt(0)].copy()
    if valid_weight.empty:
        return set()
    weight_id_col = _detect_id_col(valid_weight, urine_id_col)
    if weight_id_col is not None:
        return set(valid_weight[weight_id_col].dropna().tolist())
    resolution = resolve_unkeyed_single_entity_weight(
        urine_df,
        valid_weight,
        urine_id_columns=[urine_id_col],
        weight_column=candidate,
    )
    if resolution.weight is None:
        return set()
    patient_ids = urine_df[urine_id_col].dropna().unique().tolist()
    return set(patient_ids) if len(patient_ids) == 1 else set()


def kdigo_creatinine(
    crea_df: pd.DataFrame,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    value_col: str = 'crea',
    time_unit: Optional[str] = None,
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
        raise KDIGOComponentSchemaError(
            component="creatinine",
            reason_code="kdigo_creatinine_keys_unresolved",
            message=(
                "Could not detect creatinine ID/time columns; found "
                f"{crea_df.columns.tolist()!r}"
            ),
        )
    if value_col not in crea_df.columns:
        raise KDIGOComponentSchemaError(
            component="creatinine",
            reason_code="kdigo_creatinine_value_column_missing",
            message=f"Creatinine source is missing {value_col!r}",
        )
    
    # Ensure numeric creatinine values
    df = crea_df.copy()
    df[value_col] = _strict_numeric_component(
        df[value_col],
        component="creatinine",
        reason_code="kdigo_creatinine_numeric_encoding_invalid",
        field_name="creatinine",
    )
    invalid_range = df[value_col].notna() & (
        (df[value_col] <= 0) | (df[value_col] > 150)
    )
    if invalid_range.any():
        # A physiologically impossible numeric value is a bad *observation*,
        # not evidence that the complete component table has the wrong
        # contract.  Failing the whole component here made one malformed
        # record turn every stay in a source into an unavailable KDIGO result.
        # Keep the fail-closed behaviour for non-numeric/non-finite encodings
        # above, but exclude only the impossible measurements below.
        logger.warning(
            "Dropping %d creatinine observation(s) outside (0, 150] mg/dL "
            "before KDIGO staging",
            int(invalid_range.sum()),
        )
        df.loc[invalid_range, value_col] = np.nan
    df[time_col] = _strict_time_component(
        df[time_col],
        component="creatinine",
        reason_code="kdigo_creatinine_time_encoding_invalid",
    )
    df = df.dropna(subset=[id_col, time_col, value_col])
    
    if df.empty:
        return pd.DataFrame()
    
    # Sort by ID and time
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Calculate rolling minimum creatinine for 48h and 7 days
    # Vectorized: use searchsorted for O(N log N) window boundaries per patient
    
    # Detect time unit and convert to hours for uniform processing
    time_unit = _resolve_time_unit(df[time_col], time_col, time_unit)
    logger.debug(f"Creatinine baseline calculation using time unit: {time_unit}")
    
    if time_unit == 'datetime':
        ref_time = df[time_col].min()
        df['_hours'] = (df[time_col] - ref_time) / pd.Timedelta(hours=1)
    elif time_unit == 'timedelta':
        df['_hours'] = df[time_col] / pd.Timedelta(hours=1)
    elif time_unit == 'seconds':
        df['_hours'] = df[time_col].astype(np.float64) / 3600.0
    elif time_unit == 'hours':
        df['_hours'] = df[time_col].astype(np.float64)
    else:  # minutes
        df['_hours'] = df[time_col].astype(np.float64) / 60.0
    
    creat_low_48hr = np.full(len(df), np.nan)
    creat_low_7day = np.full(len(df), np.nan)
    baseline_n_48hr = np.zeros(len(df), dtype=np.int64)
    baseline_n_7day = np.zeros(len(df), dtype=np.int64)
    baseline_source = np.full(len(df), "unavailable", dtype=object)
    pre_icu_history_observed = np.zeros(len(df), dtype=bool)
    relative_icu_time = time_unit != 'datetime'
    
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
        # An earlier value exactly 48 h / 7 d back is inside the window; the
        # elapsed time is float hours, so the edge moves out by the shared
        # decimal-threshold eps (utils.thresholds).
        bounds_48 = np.searchsorted(hours, hours - 48.0 * (1 + REL_EPS), side='left')
        bounds_7d = np.searchsorted(hours, hours - 168.0 * (1 + REL_EPS), side='left')
        
        for i in range(1, n):
            left_48 = bounds_48[i]
            if left_48 < i:  # at least one previous measurement in window
                window_48 = creas[left_48:i]
                baseline_n_48hr[_idx_sorted[i]] = len(window_48)
                creat_low_48hr[_idx_sorted[i]] = np.min(window_48)
            left_7d = bounds_7d[i]
            if left_7d < i:
                window_7d = creas[left_7d:i]
                baseline_n_7day[_idx_sorted[i]] = len(window_7d)
                baseline_offset = int(np.argmin(window_7d))
                baseline_position = left_7d + baseline_offset
                creat_low_7day[_idx_sorted[i]] = window_7d[baseline_offset]
                if relative_icu_time:
                    baseline_source[_idx_sorted[i]] = (
                        "observed_pre_icu"
                        if hours[baseline_position] < 0
                        else "observed_prior_icu"
                    )
                    pre_icu_history_observed[_idx_sorted[i]] = bool(
                        np.any(hours[left_7d:i] < 0)
                    )
                else:
                    baseline_source[_idx_sorted[i]] = "observed_prior_time_unknown"
    
    df['creat_low_past_48hr'] = creat_low_48hr
    df['creat_low_past_7day'] = creat_low_7day
    df['creat_baseline_n_48h'] = pd.Series(
        baseline_n_48hr, index=df.index, dtype="Int64"
    )
    df['creat_baseline_n_7d'] = pd.Series(
        baseline_n_7day, index=df.index, dtype="Int64"
    )
    df['creat_baseline_source'] = pd.Series(
        baseline_source, index=df.index, dtype="string"
    )
    if relative_icu_time:
        df['creat_pre_icu_history_observed'] = pd.Series(
            pre_icu_history_observed, index=df.index, dtype="boolean"
        )
    else:
        df['creat_pre_icu_history_observed'] = pd.Series(
            pd.NA, index=df.index, dtype="boolean"
        )
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
    
    return result[[
        id_col,
        time_col,
        'crea',
        'creat_low_past_48hr',
        'creat_low_past_7day',
        'creat_baseline_n_48h',
        'creat_baseline_n_7d',
        'creat_baseline_source',
        'creat_pre_icu_history_observed',
        'aki_stage_creat',
    ]]


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
    # A rise of exactly 0.3 mg/dL or exactly 1.5x is AKI; the sums and
    # products put that boundary a few ulps either side (utils.thresholds).
    mask_1_fold = ge(creat, creat_low_7day * 1.5)
    mask_1_abs = ge(creat, creat_low_48hr + 0.3)
    stage[mask_1_fold | mask_1_abs] = 1
    
    # Stage 2: ≥2.0x baseline (7-day)
    mask_2 = ge(creat, creat_low_7day * 2.0)
    stage[mask_2] = 2
    
    # Stage 3: ≥3.0x baseline OR (≥4.0 with acute increase)
    mask_3_fold = ge(creat, creat_low_7day * 3.0)
    # For creat ≥ 4.0, require acute increase (≥0.3 in 48h or ≥1.5x baseline)
    mask_3_abs = ge(creat, 4.0) & (
        ge(creat, creat_low_48hr + 0.3) |
        ge(creat, creat_low_7day * 1.5)  # Or 1.5x baseline
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
    time_unit: Optional[str] = None,
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
        raise KDIGOComponentSchemaError(
            component="urine_output",
            reason_code="kdigo_urine_output_keys_unresolved",
            message="Could not detect urine-output ID/time columns",
        )
    
    # Calculate UO rates using simplified windowed average
    result = _calculate_uo_rates_simple(
        urine_df, weight_df, 
        id_col, time_col, 
        urine_col, weight_col,
        source_is_rate=source_is_rate,
        interval=interval,
        time_unit=time_unit,
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
    valid_weight_ids = _valid_weight_patient_ids(
        urine_df,
        weight_df,
        urine_id_col=id_col,
        weight_col=weight_col,
    )
    result['weight_available'] = result[id_col].isin(valid_weight_ids)
    result['uo_assessment_reason'] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result.loc[
        ~result['uo_assessable'] & ~result['weight_available'],
        'uo_assessment_reason',
    ] = "missing_weight"
    result.loc[
        ~result['uo_assessable'] & result['weight_available'],
        'uo_assessment_reason',
    ] = "insufficient_window"
    result.loc[result['aki_stage_uo'] == 0, 'uo_assessment_reason'] = (
        "criterion_negative"
    )
    result.loc[result['aki_stage_uo'] > 0, 'uo_assessment_reason'] = (
        "criterion_positive"
    )
    
    # Rename columns for consistency
    result = result.rename(columns={
        'uo_6h': 'uo_rt_6hr',
        'uo_12h': 'uo_rt_12hr',
        'uo_24h': 'uo_rt_24hr'
    })
    
    return result


def _detect_time_unit(time_series: pd.Series, time_col: str | None = None) -> str:
    """Return the unit a time series declares by its dtype.

    Returns 'datetime' or 'timedelta'. A numeric series declares nothing and
    raises: the caller must pass ``time_unit``.
    """
    if pd.api.types.is_datetime64_any_dtype(time_series):
        return 'datetime'
    if pd.api.types.is_timedelta64_dtype(time_series):
        return 'timedelta'

    # A numeric axis carries no unit of its own. Neither its name nor its
    # spacing is a safe substitute: a sparse hourly series and a minute-offset
    # series can have the same values, and the old spacing guess depended on
    # which stays shared the frame (lead-r13 magnitude-inference sweep, #10).
    # Every numeric axis needs an explicit time_unit.
    raise ValueError(
        "numeric KDIGO time axis is ambiguous; pass time_unit explicitly"
    )


def _resolve_time_unit(
    time_series: pd.Series,
    time_col: str | None,
    explicit: Optional[str],
) -> str:
    if explicit is None:
        return _detect_time_unit(time_series, time_col)
    aliases = {
        "h": "hours",
        "hour": "hours",
        "hours": "hours",
        "min": "minutes",
        "minute": "minutes",
        "minutes": "minutes",
        "s": "seconds",
        "second": "seconds",
        "seconds": "seconds",
        "datetime": "datetime",
        "timedelta": "timedelta",
    }
    normalized = aliases.get(str(explicit).strip().lower())
    if normalized is None:
        raise ValueError(
            "time_unit must be one of hours, minutes, seconds, datetime, or "
            "timedelta"
        )
    return normalized


# Longest charting interval one urine record may be spread over (R4-001).
_UO_MAX_GAP_HOURS = 12.0


def _calculate_uo_rates_simple(
    urine_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str = 'urine',
    weight_col: str = 'weight',
    source_is_rate: bool = False,
    interval: Optional[pd.Timedelta] = None,
    time_unit: Optional[str] = None,
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
    if urine_df.empty:
        return pd.DataFrame()

    urine = urine_df.copy()
    # An extraction batch in which no stay has a weight row still yields the
    # collection-window coverage and an unassessable rate per urine row, as
    # those stays would in a whole-cohort frame.
    weight = (
        weight_df.copy()
        if isinstance(weight_df, pd.DataFrame)
        else pd.DataFrame()
    )
    if urine_col not in urine.columns:
        urine_col = next(
            (column for column in ("urine", "value", "valuenum") if column in urine),
            "",
        )
    if not urine_col:
        raise KDIGOComponentSchemaError(
            component="urine_output",
            reason_code="kdigo_urine_output_value_column_missing",
            message="Urine-output source has no detectable value column",
        )
    urine[urine_col] = _strict_numeric_component(
        urine[urine_col],
        component="urine_output",
        reason_code="kdigo_urine_output_numeric_encoding_invalid",
        field_name="urine output",
    )
    negative_output = urine[urine_col].notna() & (urine[urine_col] < 0)
    if negative_output.any():
        # Ignore impossible volumes but retain their collection endpoints.
        # Removing the row would bridge a hole with a false valid interval.
        logger.warning(
            "Ignoring %d negative urine-output volume(s); retaining their "
            "collection breakpoints for KDIGO staging",
            int(negative_output.sum()),
        )
        urine.loc[negative_output, urine_col] = np.nan
    urine[time_col] = _strict_time_component(
        urine[time_col],
        component="urine_output",
        reason_code="kdigo_urine_output_time_encoding_invalid",
    )
    urine = urine.dropna(subset=[id_col, time_col])
    if urine.empty:
        return pd.DataFrame()

    # Multiple channels at one collection end share one duration.
    aggregation = {urine_col: "mean" if source_is_rate else "sum"}
    if '_urine_available_at' in urine:
        aggregation['_urine_available_at'] = 'max'
    if '_urine_gap_h' in urine:
        aggregation['_urine_gap_h'] = 'max'
    groups = urine.groupby([id_col, time_col], sort=True)
    values = groups.agg(aggregation)
    # A valid zero contributes coverage. An invalid same-time channel makes
    # the collection incomplete, matching the shared assessment owner.
    values[urine_col] = values[urine_col].where(groups[urine_col].count() == groups.size())
    urine = values.reset_index()

    weight_rows = not weight.empty
    if weight_rows and weight_col not in weight.columns:
        weight_col = next(
            (column for column in ("weight", "value", "valuenum") if column in weight),
            "",
        )
    if weight_rows and not weight_col:
        raise KDIGOComponentSchemaError(
            component="weight",
            reason_code="kdigo_weight_value_column_missing",
            message="Weight source has no detectable value column",
        )
    if weight_rows:
        weight[weight_col] = _strict_numeric_component(
            weight[weight_col],
            component="weight",
            reason_code="kdigo_weight_numeric_encoding_invalid",
            field_name="weight",
        )
    weight_id_col = _detect_id_col(weight) if weight_rows else None

    keyed_weight = None
    if weight_id_col and weight_id_col in weight.columns:
        try:
            keyed_weight = resolve_keyed_unique_weights(
                weight,
                id_columns=[weight_id_col],
                weight_column=weight_col,
            )
        except ConflictingKeyedWeightError as exc:
            raise KDIGOComponentCalculationError(
                component="weight",
                reason_code="kdigo_weight_values_conflict",
                message=(
                    "KDIGO urine-output staging cannot choose among "
                    "different valid keyed weights without a clinical selector"
                ),
            ) from exc

    # Resolve weight only through a patient key.  The exceptional unkeyed
    # one-entity path is explicitly proved below; selecting ``iloc[0]`` from a
    # multi-patient table would silently apply one patient's weight to another.
    global_weight = np.nan
    if keyed_weight is not None:
        weight_per_patient = keyed_weight.set_index(weight_id_col)[
            weight_col
        ].to_dict()
    elif not weight_rows:
        weight_per_patient = {}
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
    time_unit = _resolve_time_unit(urine[time_col], time_col, time_unit)
    logger.debug(f"Detected time unit for UO calculation: {time_unit}")

    # Vectorized UO rate calculation using cumsum + searchsorted (O(N log N))
    # Convert time to minutes for uniform window computation
    urine = urine.copy()
    if time_unit == 'datetime':
        _ref = urine[time_col].min()
        urine['_min'] = (urine[time_col] - _ref) / pd.Timedelta(minutes=1)
    elif time_unit == 'timedelta':
        urine['_min'] = urine[time_col] / pd.Timedelta(minutes=1)
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
    
    from .urine_output import collection_windows, KDIGO_EVIDENCE_COLUMNS

    output = [f"uo_rt_{w}hr" for w in (6, 12, 24)]
    arrays = {name: np.full(len(urine), np.nan) for name in [*output, *KDIGO_EVIDENCE_COLUMNS]}
    width = pd.Timedelta(interval or pd.Timedelta(hours=1)).total_seconds() / 3600
    minutes = urine['_min'].to_numpy()
    all_amounts = urine[urine_col].to_numpy()
    weights = urine['_wt'].to_numpy()
    source_times = urine['_urine_available_at'].to_numpy() if '_urine_available_at' in urine else None
    record_gaps = urine['_urine_gap_h'].to_numpy() if '_urine_gap_h' in urine else None
    for positions in urine.groupby(id_col, sort=False).indices.values():
        ends = minutes[positions] / 60.0
        amounts = all_amounts[positions]
        if source_is_rate:
            amounts = amounts / width
        integrated = collection_windows(
            ends, amounts, (6, 12, 24), source_is_rate=source_is_rate,
            max_gap=_UO_MAX_GAP_HOURS,
            source_times=source_times[positions] if source_times is not None else None,
            record_gaps=record_gaps[positions] if record_gaps is not None else None,
        )
        stage_available = np.full(len(ends), np.nan)
        wt = weights[positions]
        for window in (6, 12, 24):
            volume, covered, available = integrated[window]
            valid = ge(covered, float(window)) & np.isfinite(wt) & (wt > 0)
            name = f"uo_rt_{window}hr"
            arrays[name][positions] = np.divide(
                volume, wt * covered, out=np.full(len(ends), np.nan), where=valid,
            )
            arrays[f"{name}_covered_h"][positions] = covered
            stage_available = np.fmax(stage_available, np.where(valid, available, np.nan))
        arrays["aki_stage_uo_available_at"][positions] = stage_available
    for name, values in arrays.items():
        urine[name] = values
    return urine[[id_col, time_col, *output, *KDIGO_EVIDENCE_COLUMNS]]


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
    # Rates are volume / (weight x hours): exactly 0.5 is not < 0.5.
    mask_1 = lt(uo_6h_num, 0.5) & (ge(uo_12h_num, 0.5) | uo_12h_num.isna())
    stage[mask_1] = 1
    
    # Stage 2: UO < 0.5 for ≥12h
    mask_2 = lt(uo_12h_num, 0.5)
    stage[mask_2] = 2
    
    # Stage 3: UO < 0.3 for ≥24h OR anuria (0) for ≥12h
    mask_3_oliguria = lt(uo_24h_num, 0.3)
    # Anuria is zero output (R4-061); 0.01 mL/kg/h was a non-zero cut-off.
    mask_3_anuria = le(uo_12h_num, 0.0)
    stage[mask_3_oliguria | mask_3_anuria] = 3
    
    return stage


def _normalise_rrt_indicator(values: pd.Series) -> pd.Series:
    """Return explicit RRT observations as nullable booleans.

    Missing observations remain missing.  Unknown encodings are schema errors;
    interpreting them as ``False`` would turn a decoding defect into negative
    clinical evidence.
    """

    if pd.api.types.is_bool_dtype(values) or str(values.dtype) == "boolean":
        return values.astype("boolean")
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if (numeric.dropna() < 0).any():
            raise KDIGOComponentSchemaError(
                component="rrt",
                reason_code="kdigo_rrt_indicator_invalid",
                message="RRT indicator contains a negative numeric value",
            )
        result = pd.Series(pd.NA, index=values.index, dtype="boolean")
        result.loc[numeric.notna()] = numeric.loc[numeric.notna()] > 0
        return result

    normalised = values.astype("string").str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y", "active"}
    falsy = {"0", "false", "f", "no", "n", "inactive"}
    unknown = normalised.notna() & ~normalised.isin(truthy | falsy)
    if unknown.any():
        raise KDIGOComponentSchemaError(
            component="rrt",
            reason_code="kdigo_rrt_indicator_invalid",
            message="RRT indicator contains an unrecognised value",
        )
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    result.loc[normalised.isin(truthy)] = True
    result.loc[normalised.isin(falsy)] = False
    return result


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
    rrt["_rrt_active"] = _normalise_rrt_indicator(rrt["rrt"])

    starts = (
        rrt.loc[rrt["_rrt_active"].fillna(False), [id_col, time_col]]
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


def _rrt_ascertainment_from_observations(
    result: pd.DataFrame,
    rrt_view: pd.DataFrame,
    id_col: str,
    time_col: str,
    *,
    source_complete: bool = False,
) -> pd.Series:
    """Classify RRT evidence, including a completed positive-event search."""

    ascertainment = pd.Series(
        "negative" if source_complete else "indeterminate",
        index=result.index,
        dtype="string",
    )
    if result.empty or rrt_view.empty:
        return ascertainment

    rrt = rrt_view.copy().dropna(subset=[id_col, time_col])
    rrt["_rrt_active"] = _normalise_rrt_indicator(rrt["rrt"])
    for patient_id, observations in rrt.groupby(id_col, sort=False):
        patient_rows = result[id_col] == patient_id
        if not patient_rows.any():
            continue
        active_times = observations.loc[
            observations["_rrt_active"].fillna(False), time_col
        ].tolist()
        negative_times = observations.loc[
            observations["_rrt_active"].eq(False).fillna(False), time_col
        ].tolist()
        for row_index, row_time in result.loc[patient_rows, time_col].items():
            if any(observed_time <= row_time for observed_time in active_times):
                ascertainment.loc[row_index] = "positive"
            elif any(observed_time <= row_time for observed_time in negative_times):
                ascertainment.loc[row_index] = "negative"
    return ascertainment


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
    time_unit: Optional[str] = None,
    interval: Optional[pd.Timedelta] = None,
    observation_window_coverage: Optional[Mapping[Any, str]] = None,
    rrt_source_complete: bool = False,
    component_sources: Optional[KDIGOComponentSources] = None,
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
        observation_window_coverage: Explicit patient-level coverage receipt.
            Values are ``complete``, ``partial``, or ``indeterminate``.  In the
            absence of this receipt the function will never infer that a
            component-negative row completely excludes AKI.
        rrt_source_complete: Whether the dedicated active-RRT source was
            successfully searched for the requested cohort. Because the
            source is positive-event encoded, no event then constitutes a
            criterion-negative RRT component rather than missing data.
        component_sources: Run-level receipt of which component sources the
            installation provides.  It decides ``source_absent`` versus
            ``patient_data_absent``; without it an empty frame is read as an
            absent source, which is only valid for whole-cohort callers.
        
    Returns:
        DataFrame with combined AKI staging including:
        - aki_stage_creat: Creatinine-based stage (0-3)
        - aki_stage_uo: Urine output-based stage (0-3)
        - aki_stage: Final combined stage (0-3, or ``<NA>`` when indeterminate)
        - aki: Boolean indicator (True if aki_stage > 0)
        - aki_severe: Nullable severe-AKI indicator (KDIGO stage 2-3)
        - aki_severe_creat, aki_severe_uo, aki_severe_rrt: component indicators
        - aki_severe_ascertainment, aki_severe_assessable: severe-AKI receipt
        - creatinine_ascertainment, urine_ascertainment, rrt_ascertainment
        - observation_window_coverage, aki_ascertainment
        - aki_assessable, aki_assessment_reason: compatibility/diagnostic fields
    """
    for component, frame in (
        ("creatinine", crea_df),
        ("urine", urine_df),
        ("weight", weight_df),
        ("rrt", rrt_df),
    ):
        if frame is not None and not isinstance(frame, pd.DataFrame):
            raise KDIGOComponentSchemaError(
                component=component,
                reason_code=f"kdigo_{component}_source_not_dataframe",
                message=f"{component} source must be a pandas DataFrame when supplied",
            )
    crea_present = _component_source_present("creatinine", crea_df, component_sources)
    urine_present = _component_source_present("urine", urine_df, component_sources)
    rrt_present = _component_source_present("rrt", rrt_df, component_sources)
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
        raise KDIGOComponentSchemaError(
            component="combined_aki",
            reason_code="kdigo_combined_timeline_keys_unresolved",
            message=(
                "Could not detect a KDIGO ID/time key from available components; "
                f"found columns {list(anchor.columns)!r}"
            ),
        )
    anchor_keys = anchor[[id_col, time_col]].copy()
    anchor_keys[time_col] = _strict_time_component(
        anchor_keys[time_col],
        component="combined_aki",
        reason_code="kdigo_combined_time_encoding_invalid",
    )
    anchor_keys = anchor_keys.dropna(subset=[id_col, time_col]).drop_duplicates()

    crea_staging = pd.DataFrame()
    if isinstance(crea_df, pd.DataFrame) and not crea_df.empty:
        missing = [
            column
            for column in (id_col, time_col, crea_col)
            if column not in crea_df.columns
        ]
        if missing:
            raise KDIGOComponentSchemaError(
                component="creatinine",
                reason_code="kdigo_creatinine_schema_invalid",
                message=f"Non-empty creatinine source is missing {missing!r}",
            )
        try:
            crea_staging = kdigo_creatinine(
                crea_df, id_col, time_col, crea_col, time_unit=time_unit
            )
        except KDIGOComponentError:
            raise
        except Exception as exc:
            raise KDIGOComponentCalculationError(
                component="creatinine",
                reason_code="kdigo_creatinine_calculation_failed",
                message="Creatinine KDIGO staging failed",
            ) from exc

    if crea_staging.empty:
        result = anchor_keys
        result['aki_stage_creat'] = pd.Series(
            pd.NA, index=result.index, dtype="Int64"
        )
    else:
        result = crea_staging.copy()
    result['creat_assessable'] = result['aki_stage_creat'].notna()
    
    # Calculate urine output-based staging whenever urine rows exist.  A stay
    # without a valid weight is labelled per row by ``kdigo_uo``; deciding it
    # from an empty weight frame would make the label (and the window
    # coverage receipts) depend on which stays share an extraction batch.
    if urine_df is not None and not urine_df.empty:
        try:
            uo_staging = kdigo_uo(
                urine_df,
                weight_df if isinstance(weight_df, pd.DataFrame) else pd.DataFrame(),
                id_col,
                time_col,
                urine_col,
                weight_col,
                source_is_rate=urine_source_is_rate,
                interval=interval,
                time_unit=time_unit,
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
                        'weight_available',
                        *[c for c in uo_staging if c.endswith('_covered_h') or c == 'aki_stage_uo_available_at'],
                    ]],
                    on=[id_col, time_col],
                    how='outer'
                )
            else:
                result['aki_stage_uo'] = pd.Series(
                    pd.NA, index=result.index, dtype="Int64"
                )
                result['uo_assessable'] = False
                result['uo_assessment_reason'] = "patient_data_absent"
                result['weight_available'] = False
        except KDIGOComponentError:
            raise
        except Exception as exc:
            raise KDIGOComponentCalculationError(
                component="urine_output",
                reason_code="kdigo_urine_output_calculation_failed",
                message="Urine-output KDIGO staging failed",
            ) from exc
    else:
        result['aki_stage_uo'] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result['uo_assessable'] = False
        result['uo_assessment_reason'] = (
            "patient_data_absent" if urine_present else "source_absent"
        )
        result['weight_available'] = False

    def _component_timeline(
        frame: Optional[pd.DataFrame], *, component: str
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=[id_col, time_col])
        source_id = _detect_id_col(frame, id_col)
        source_time = _detect_time_col(frame, time_col)
        if source_id is None or source_time is None:
            raise KDIGOComponentSchemaError(
                component=component,
                reason_code=f"kdigo_{component}_timeline_schema_invalid",
                message=(
                    "Non-empty KDIGO component has no detectable ID/time keys; "
                    f"available columns are {list(frame.columns)!r}"
                ),
            )
        timeline = (
            frame[[source_id, source_time]]
            .rename(columns={source_id: id_col, source_time: time_col})
            .dropna(subset=[id_col, time_col])
        )
        timeline[time_col] = _strict_time_component(
            timeline[time_col],
            component=component,
            reason_code=f"kdigo_{component}_time_encoding_invalid",
        )
        return timeline

    # Retain rows from every observed KDIGO component.  In particular, an RRT
    # initiation must survive even when no contemporaneous creatinine exists.
    try:
        spine = (
            pd.concat(
                [
                    result[[id_col, time_col]],
                    _component_timeline(urine_df, component="urine"),
                    _component_timeline(rrt_df, component="rrt"),
                ],
                ignore_index=True,
            )
            .dropna(subset=[id_col, time_col])
            .drop_duplicates()
            .sort_values([id_col, time_col], kind="stable")
            .reset_index(drop=True)
        )
    except KDIGOComponentError:
        raise
    except Exception as exc:
        raise KDIGOComponentSchemaError(
            component="combined_aki",
            reason_code="kdigo_component_time_axes_incompatible",
            message="KDIGO component ID/time axes cannot form one timeline",
        ) from exc
    result = spine.merge(result, on=[id_col, time_col], how='left', sort=False)

    # An outer UO merge can add times that have no creatinine row.  Preserve
    # the distinction between an unavailable baseline and a proven negative.
    result['creat_assessable'] = result['aki_stage_creat'].notna()
    result['uo_assessable'] = result['uo_assessable'].fillna(False).astype(bool)
    result['weight_available'] = result['weight_available'].fillna(False).astype(bool)
    result['uo_assessment_reason'] = result['uo_assessment_reason'].astype("string")
    result.loc[
        ~result['uo_assessable'] & result['uo_assessment_reason'].isna(),
        'uo_assessment_reason',
    ] = "patient_data_absent"
    
    # Handle RRT - automatic Stage 3
    result['aki_stage_rrt'] = pd.Series(0, index=result.index, dtype="Int64")
    rrt_ascertainment = pd.Series(
        "indeterminate", index=result.index, dtype="string"
    )
    rrt_id_col: Optional[str] = None
    if rrt_df is not None and not rrt_df.empty:
        try:
            rrt_id_col = _detect_id_col(rrt_df, id_col)
            rrt_time_col = _detect_time_col(rrt_df, time_col)
            rrt_col = _detect_value_col(rrt_df, 'rrt')
            if not (rrt_id_col and rrt_time_col and rrt_col):
                raise KDIGOComponentSchemaError(
                    component="rrt",
                    reason_code="kdigo_rrt_schema_invalid",
                    message=(
                        "Non-empty RRT source has no detectable ID/time/value contract; "
                        f"available columns are {list(rrt_df.columns)!r}"
                    ),
                )
            rrt_view = rrt_df[[rrt_id_col, rrt_time_col, rrt_col]].rename(
                columns={rrt_id_col: id_col, rrt_time_col: time_col, rrt_col: 'rrt'}
            )
            rrt_view[time_col] = _strict_time_component(
                rrt_view[time_col],
                component="rrt",
                reason_code="kdigo_rrt_time_encoding_invalid",
            )
            rrt_mask = _rrt_active_from_initiation(
                result, rrt_view, id_col, time_col
            )
            rrt_ascertainment = _rrt_ascertainment_from_observations(
                result,
                rrt_view,
                id_col,
                time_col,
                source_complete=rrt_source_complete,
            )
            result["rrt"] = rrt_mask
            # RRT is its own component; do not rewrite the creatinine stage.
            result.loc[rrt_mask, 'aki_stage_rrt'] = 3
        except KDIGOComponentError:
            raise
        except Exception as exc:
            raise KDIGOComponentCalculationError(
                component="rrt",
                reason_code="kdigo_rrt_calculation_failed",
                message="RRT KDIGO staging failed",
            ) from exc
    
    # Calculate combined AKI stage.  Inactive/undocumented RRT is deliberately
    # excluded as negative evidence: a zero placeholder must not turn a row
    # with unknown creatinine and UO components into "no AKI".
    result['aki_stage_creat'] = result['aki_stage_creat'].astype("Int64")
    result['aki_stage_uo'] = result['aki_stage_uo'].astype("Int64")
    result['aki_stage_rrt'] = result['aki_stage_rrt'].astype("Int64")
    result['rrt_observed'] = (result['aki_stage_rrt'] > 0).astype("boolean")
    result['creatinine_ascertainment'] = pd.Series(
        "indeterminate", index=result.index, dtype="string"
    )
    result.loc[
        result['aki_stage_creat'] == 0, 'creatinine_ascertainment'
    ] = "negative"
    result.loc[
        result['aki_stage_creat'] > 0, 'creatinine_ascertainment'
    ] = "positive"
    result['creatinine_ascertainment_reason'] = pd.Series(
        "patient_data_absent" if crea_present else "source_absent",
        index=result.index,
        dtype="string",
    )
    if crea_df is not None and not crea_df.empty:
        creat_patients = set(crea_df[id_col].dropna().tolist())
        creat_present = result[id_col].isin(creat_patients)
        result.loc[creat_present, 'creatinine_ascertainment_reason'] = (
            "insufficient_baseline"
        )
        result.loc[
            result['aki_stage_creat'] == 0, 'creatinine_ascertainment_reason'
        ] = "criterion_negative"
        result.loc[
            result['aki_stage_creat'] > 0, 'creatinine_ascertainment_reason'
        ] = "criterion_positive"
        result.loc[
            ~creat_present, 'creatinine_ascertainment_reason'
        ] = "patient_data_absent"
    result['urine_ascertainment'] = pd.Series(
        "indeterminate", index=result.index, dtype="string"
    )
    result.loc[result['aki_stage_uo'] == 0, 'urine_ascertainment'] = "negative"
    result.loc[result['aki_stage_uo'] > 0, 'urine_ascertainment'] = "positive"
    result['urine_ascertainment_reason'] = result[
        'uo_assessment_reason'
    ].astype("string")
    result['rrt_ascertainment'] = rrt_ascertainment
    result['rrt_ascertainment_reason'] = pd.Series(
        (
            "source_searched_no_active_rrt"
            if rrt_source_complete
            else "patient_data_absent"
            if rrt_present
            else "source_absent"
        ),
        index=result.index,
        dtype="string",
    )
    if rrt_df is not None and not rrt_df.empty:
        assert rrt_id_col is not None
        rrt_patients = set(rrt_df[rrt_id_col].dropna().tolist())
        rrt_present = result[id_col].isin(rrt_patients)
        result.loc[rrt_present, 'rrt_ascertainment_reason'] = (
            "source_searched_no_active_rrt"
            if rrt_source_complete
            else "no_observation_at_or_before_time"
        )
        result.loc[
            result['rrt_ascertainment'] == "negative",
            'rrt_ascertainment_reason',
        ] = "criterion_negative"
        result.loc[
            result['rrt_ascertainment'] == "positive",
            'rrt_ascertainment_reason',
        ] = "criterion_positive"
        result.loc[~rrt_present, 'rrt_ascertainment_reason'] = (
            "source_searched_no_active_rrt"
            if rrt_source_complete
            else "patient_data_absent"
        )
    elif rrt_source_complete:
        # A successfully queried positive-event source can legitimately be
        # empty for a cohort. Preserve that completed search as negative RRT
        # evidence instead of silently reverting to unknown.
        result['rrt_ascertainment'] = pd.Series(
            "negative", index=result.index, dtype="string"
        )
    rrt_positive = result['aki_stage_rrt'].where(
        result['aki_stage_rrt'] > 0, pd.NA
    )
    components = pd.concat(
        [result['aki_stage_creat'], result['aki_stage_uo'], rrt_positive],
        axis=1,
    )
    result['aki_stage'] = components.max(axis=1, skipna=True).astype("Int64")

    supplied_coverage = dict(observation_window_coverage or {})
    invalid_coverage = sorted(
        {
            str(value)
            for value in supplied_coverage.values()
            if str(value) not in KDIGO_OBSERVATION_COVERAGE_STATES
        }
    )
    if invalid_coverage:
        raise KDIGOComponentSchemaError(
            component="observation_window",
            reason_code="kdigo_observation_window_coverage_invalid",
            message=f"Unknown observation-window coverage states: {invalid_coverage!r}",
        )
    any_component_observed = pd.concat(
        [
            result['creatinine_ascertainment'],
            result['urine_ascertainment'],
            result['rrt_ascertainment'],
        ],
        axis=1,
    ).ne("indeterminate").any(axis=1)
    result['observation_window_coverage'] = result[id_col].map(
        supplied_coverage
    ).astype("string")
    all_components_known = pd.concat(
        [
            result['creatinine_ascertainment'],
            result['urine_ascertainment'],
            result['rrt_ascertainment'],
        ],
        axis=1,
    ).ne("indeterminate").all(axis=1)
    result.loc[
        result['observation_window_coverage'].isna() & all_components_known,
        'observation_window_coverage',
    ] = "complete"
    result.loc[
        result['observation_window_coverage'].isna() & any_component_observed,
        'observation_window_coverage',
    ] = "partial"
    result.loc[
        result['observation_window_coverage'].isna(),
        'observation_window_coverage',
    ] = "indeterminate"

    component_states = result[
        [
            'creatinine_ascertainment',
            'urine_ascertainment',
            'rrt_ascertainment',
        ]
    ]
    any_positive = component_states.eq("positive").any(axis=1)
    all_negative = component_states.eq("negative").all(axis=1)
    any_negative = component_states.eq("negative").any(axis=1)
    complete_negative = all_negative & result['observation_window_coverage'].eq(
        "complete"
    )

    result['aki_ascertainment'] = pd.Series(
        "indeterminate", index=result.index, dtype="string"
    )
    result.loc[
        any_negative & ~any_positive, 'aki_ascertainment'
    ] = "partial_no_observed_positive"
    result.loc[complete_negative, 'aki_ascertainment'] = "negative_complete"
    result.loc[any_positive, 'aki_ascertainment'] = "positive"

    result['aki'] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result.loc[result['aki_ascertainment'] == "positive", 'aki'] = True
    result.loc[result['aki_ascertainment'] == "negative_complete", 'aki'] = False
    result['aki_assessable'] = result['aki_ascertainment'].isin(
        {"positive", "negative_complete"}
    )
    result['aki_assessment_reason'] = result['aki_ascertainment'].copy()

    # Publish severe AKI (KDIGO stage 2-3) as a separate, nullable endpoint.
    # A positive component is sufficient to establish severe AKI even when
    # the other components are missing.  A negative combined endpoint needs
    # all three components and a complete observation-window receipt; absence
    # of an RRT event row or of a urine/creatinine window is not negative
    # evidence.  "Incident" severe AKI is intentionally not encoded here:
    # incidence depends on a study-specific baseline and follow-up anchor.
    result['aki_severe_creat'] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    creat_known = result['aki_stage_creat'].notna()
    result.loc[creat_known, 'aki_severe_creat'] = (
        result.loc[creat_known, 'aki_stage_creat'] >= 2
    )

    result['aki_severe_uo'] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    urine_known = result['aki_stage_uo'].notna()
    result.loc[urine_known, 'aki_severe_uo'] = (
        result.loc[urine_known, 'aki_stage_uo'] >= 2
    )

    result['aki_severe_rrt'] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result.loc[
        result['rrt_ascertainment'] == "negative", 'aki_severe_rrt'
    ] = False
    result.loc[
        result['rrt_ascertainment'] == "positive", 'aki_severe_rrt'
    ] = True

    severe_components = result[
        ['aki_severe_creat', 'aki_severe_uo', 'aki_severe_rrt']
    ]
    any_severe = severe_components.eq(True).any(axis=1)
    all_below_severe = severe_components.eq(False).all(axis=1)
    any_below_severe = severe_components.eq(False).any(axis=1)
    complete_severe_negative = all_below_severe & result[
        'observation_window_coverage'
    ].eq("complete")

    result['aki_severe_ascertainment'] = pd.Series(
        "indeterminate", index=result.index, dtype="string"
    )
    result.loc[
        any_below_severe & ~any_severe, 'aki_severe_ascertainment'
    ] = "partial_no_observed_positive"
    result.loc[
        complete_severe_negative, 'aki_severe_ascertainment'
    ] = "negative_complete"
    result.loc[any_severe, 'aki_severe_ascertainment'] = "positive"

    result['aki_severe'] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result.loc[
        result['aki_severe_ascertainment'] == "positive", 'aki_severe'
    ] = True
    result.loc[
        result['aki_severe_ascertainment'] == "negative_complete", 'aki_severe'
    ] = False
    result['aki_severe_assessable'] = result[
        'aki_severe_ascertainment'
    ].isin({"positive", "negative_complete"})
    
    return result


def load_kdigo_aki(
    database: str,
    data_path: Optional[str] = None,
    patient_ids: Optional[List] = None,
    max_patients: Optional[int] = None,
    verbose: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    observation_window_coverage: Optional[Mapping[Any, str]] = None,
    rrt_source_complete: bool = False,
) -> pd.DataFrame:
    """Load KDIGO AKI staging for a given database using EasyICU concepts.
    
    This is the high-level API function that:
    1. Loads phenotype-specific creatinine, urine-output, and active-RRT inputs
    2. Calculates KDIGO AKI staging using the loaded data
    3. Returns a unified DataFrame with AKI staging results
    
    Args:
        database: Database name ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        data_path: Path to the database (uses default if None)
        patient_ids: List of patient IDs to load (loads all if None)
        max_patients: Maximum number of patients to load
        verbose: Print progress messages
        preloaded_data: Optional dict of {concept_name: DataFrame} to skip re-loading
        observation_window_coverage: Explicit patient-level complete/partial/
            indeterminate coverage receipt.  Without it, component-negative
            rows remain partial rather than being promoted to complete negatives.
        rrt_source_complete: Explicit receipt that the dedicated positive-event
            RRT source was successfully searched for the requested cohort.
        
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
    legacy_input_names = {
        "kdigo_creatinine_input": "crea",
        "kdigo_urine_input": "urine",
        "acute_rrt_input": "rrt",
    }
    
    def _load_or_reuse(concept):
        if concept in _pre:
            if not isinstance(_pre[concept], pd.DataFrame):
                raise KDIGOComponentSchemaError(
                    component=concept,
                    reason_code="kdigo_preloaded_component_not_dataframe",
                    message=f"Preloaded {concept} source is not a pandas DataFrame",
                )
            return _pre[concept]
        legacy_name = legacy_input_names.get(concept)
        if legacy_name in _pre:
            if not isinstance(_pre[legacy_name], pd.DataFrame):
                raise KDIGOComponentSchemaError(
                    component=concept,
                    reason_code="kdigo_preloaded_component_not_dataframe",
                    message=f"Preloaded {legacy_name} source is not a pandas DataFrame",
                )
            return _pre[legacy_name]
        try:
            frame = load_concepts(
                concepts=[concept], database=database, data_path=data_path,
                patient_ids=patient_ids, max_patients=max_patients, verbose=verbose,
                r_compatible=concept != "kdigo_urine_input",
            )
            if concept == "kdigo_urine_input" and isinstance(frame, pd.DataFrame):
                clock = _detect_time_col(frame)
                if clock is not None:
                    frame = frame.rename(columns={clock: "charttime"})
            return frame
        except Exception as exc:
            raise KDIGOComponentLoadError(
                component=concept,
                reason_code="kdigo_component_load_failed",
                message=f"Failed to load KDIGO component {concept!r}",
            ) from exc
    
    if verbose:
        logger.info(f"Loading KDIGO AKI data for {database}...")
    
    # Load all KDIGO components before deciding whether the patient has an
    # assessable timeline.  Creatinine is not a membership prerequisite.
    crea_df = _load_or_reuse('kdigo_creatinine_input')
    urine_df = _load_or_reuse('kdigo_urine_input')
    weight_df = _load_or_reuse('weight')
    rrt_df = _load_or_reuse('acute_rrt_input')

    def _normalise_input_column(
        frame: pd.DataFrame, source_name: str, target_name: str
    ) -> pd.DataFrame:
        if target_name in frame.columns or source_name not in frame.columns:
            return frame
        return frame.rename(columns={source_name: target_name})

    crea_df = _normalise_input_column(
        crea_df, "kdigo_creatinine_input", "crea"
    )
    urine_df = _normalise_input_column(
        urine_df, "kdigo_urine_input", "urine"
    )
    rrt_df = _normalise_input_column(rrt_df, "acute_rrt_input", "rrt")

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
        time_unit="hours",
        interval=pd.Timedelta(hours=1),
        observation_window_coverage=observation_window_coverage,
        rrt_source_complete=rrt_source_complete,
        component_sources=kdigo_component_sources_for_path(
            database, data_path
        ),
    )
    
    if verbose and not result.empty:
        n_total = len(result)
        n_assessable = int(result['aki_assessable'].sum())
        n_positive = int(result['aki'].eq(True).fillna(False).sum())
        n_negative = int(result['aki'].eq(False).fillna(False).sum())
        n_partial = int(
            result['aki_ascertainment']
            .eq("partial_no_observed_positive")
            .sum()
        )
        n_indeterminate = int(result['aki_ascertainment'].eq("indeterminate").sum())
        prevalence = (
            100.0 * n_positive / n_assessable if n_assessable else float("nan")
        )
        stage_dist = result['aki_stage'].value_counts().sort_index()
        logger.info(f"KDIGO AKI Results for {database}:")
        logger.info(f"  Total rows: {n_total:,}")
        logger.info(f"  AKI positive: {n_positive:,}")
        logger.info(f"  AKI negative (complete ascertainment): {n_negative:,}")
        logger.info(f"  AKI partial, no observed positive: {n_partial:,}")
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
        'admissionid', 'patientid', 'CaseID', 'hadm_id',
        'INP_NO', 'patient_SN',
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
    if id_col is None or time_col is None:
        raise KDIGOComponentSchemaError(
            component="combined_aki",
            reason_code="kdigo_public_api_keys_unresolved",
            message="Could not detect patient ID/time columns for AKI incidence",
        )
    missing = [column for column in ("aki", "aki_stage") if column not in aki_df]
    if missing:
        raise KDIGOComponentSchemaError(
            component="combined_aki",
            reason_code="kdigo_public_api_schema_invalid",
            message=f"AKI incidence input is missing {missing!r}",
        )
    
    # Get first AKI occurrence
    aki_only = aki_df.loc[aki_df['aki'].eq(True).fillna(False)].copy()
    
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

    Prevalence is reported only among definitive phenotypes (positive or
    complete negative).  Partial ascertainment and indeterminate patients are
    separate patient-level strata and never silently enter the negative group.
    
    Args:
        aki_df: DataFrame from kdigo_stages or load_kdigo_aki
        id_col: Patient ID column (auto-detected if None)
        
    Returns:
        Dictionary with summary statistics
    """
    if aki_df.empty:
        return {'error': 'Empty DataFrame'}
    
    id_col = _detect_id_col(aki_df, id_col)
    if id_col is None:
        raise KDIGOComponentSchemaError(
            component="combined_aki",
            reason_code="kdigo_public_api_id_unresolved",
            message="Could not detect patient ID column for AKI summary",
        )
    required = {"aki_stage", "aki_ascertainment"}
    missing = sorted(required - set(aki_df.columns))
    if missing:
        raise KDIGOComponentSchemaError(
            component="combined_aki",
            reason_code="kdigo_ascertainment_receipt_missing",
            message=f"AKI summary input is missing {missing!r}",
        )

    invalid_states = sorted(
        set(aki_df['aki_ascertainment'].dropna().astype(str))
        - KDIGO_ASCERTAINMENT_STATES
    )
    if invalid_states:
        raise KDIGOComponentSchemaError(
            component="combined_aki",
            reason_code="kdigo_ascertainment_state_invalid",
            message=f"AKI summary input has unknown states {invalid_states!r}",
        )

    n_patients = int(aki_df[id_col].nunique())
    n_measurements = len(aki_df)
    patient_flags = pd.DataFrame(
        {
            id_col: aki_df[id_col],
            "positive": aki_df['aki_ascertainment'].eq("positive"),
            "negative_complete": aki_df['aki_ascertainment'].eq(
                "negative_complete"
            ),
            "partial": aki_df['aki_ascertainment'].eq(
                "partial_no_observed_positive"
            ),
        }
    ).groupby(id_col, dropna=True, sort=False).any()
    patient_state = pd.Series(
        "indeterminate", index=patient_flags.index, dtype="string"
    )
    patient_state.loc[patient_flags["partial"]] = (
        "partial_no_observed_positive"
    )
    patient_state.loc[patient_flags["negative_complete"]] = "negative_complete"
    patient_state.loc[patient_flags["positive"]] = "positive"

    n_aki_patients = int(patient_state.eq("positive").sum())
    n_negative_patients = int(patient_state.eq("negative_complete").sum())
    n_partial_patients = int(
        patient_state.eq("partial_no_observed_positive").sum()
    )
    n_indeterminate_patients = int(patient_state.eq("indeterminate").sum())
    n_definitive_patients = n_aki_patients + n_negative_patients
    prevalence_definitive = (
        float(n_aki_patients / n_definitive_patients)
        if n_definitive_patients > 0
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
        'aki_negative_complete_patients': n_negative_patients,
        'aki_partial_no_observed_positive_patients': n_partial_patients,
        'aki_indeterminate_patients': n_indeterminate_patients,
        'n_definitive_phenotype_patients': n_definitive_patients,
        'aki_prevalence_among_definitive_phenotypes': prevalence_definitive,
        'definitive_phenotype_coverage': (
            float(n_definitive_patients / n_patients) if n_patients > 0 else None
        ),
        'partial_ascertainment_fraction': (
            float(n_partial_patients / n_patients) if n_patients > 0 else None
        ),
        # Compatibility aliases now point only to definitive phenotypes.  They
        # do not recreate the old "any component available" denominator.
        'aki_negative_assessable_patients': n_negative_patients,
        'n_assessable_patients': n_definitive_patients,
        'aki_prevalence_among_assessable': prevalence_definitive,
        'ascertainment_coverage': (
            float(n_definitive_patients / n_patients) if n_patients > 0 else None
        ),
        'aki_rate': prevalence_definitive,
        'aki_rate_denominator': 'definitive_phenotype_patients',
        'stage_distribution_measurements': stage_dist,
        'max_stage_distribution_patients': max_stage_dist,
    }


# --- Public cross-package aliases (thin wrappers, no logic change) ---
# Private names are kept for backward compatibility; cross-package callers
# must use the public names below.
detect_id_col = _detect_id_col
detect_time_col = _detect_time_col
detect_value_col = _detect_value_col
resolve_time_unit = _resolve_time_unit
strict_numeric_component = _strict_numeric_component


__all__ = [
    "KDIGOComponentSources",
    "kdigo_component_sources",
    "kdigo_component_sources_for_path",
    "detect_id_col",
    "detect_time_col",
    "detect_value_col",
    "resolve_time_unit",
    "strict_numeric_component",
]
