"""Quality metric calculations and patient selector helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

_PROTECTED_NAMES = {
    '_quality_detect_time_col',
    '_quality_to_hour_bins',
    '_get_quality_cohort_patient_count',
    '_count_quality_event_occurrences',
    '_choose_concept_value_column',
    '_get_concept_numeric_value_columns',
    '_expected_observation_count',
    '_compute_quality_out_of_physio_rate',
    '_compute_quality_duplicate_timestamp_rate',
    '_summarize_quality_temporal_density',
    '_filter_patient_selector_options',
    '_patient_selector',
    '_get_quality_cohort_patient_ids',
    '_get_quality_los_by_patient',
    '_format_quality_density',
    '_get_quality_denominator_note',
    '_smd_severity_tag',
    '_compute_smd_continuous',
    '_compute_smd_binary',
    '_vectorized_expected_per_patient',
    '_build_quality_metric_profile',
    '_cohort_cache_fingerprint',
    '_los_cache_fingerprint',
    '_build_quality_metric_profile_cached',
    '_compute_quality_missing_rate',
    '_APP_CONTEXT',
    '_PROTECTED_NAMES',
    '_install_app_context',
    'Any',
    'Optional',
    'np',
    'pd',
}
_APP_CONTEXT: dict[str, Any] = {}


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app constants and Streamlit state needed by transitional helpers."""
    _APP_CONTEXT.clear()
    _APP_CONTEXT.update(app_context)
    for name, value in app_context.items():
        if name.startswith('__') or name in _PROTECTED_NAMES:
            continue
        globals()[name] = value


def _quality_detect_time_col(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely time column for quality-rate calculations."""
    for col in QUALITY_TIME_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _quality_to_hour_bins(series: pd.Series, col_name: str) -> Optional[pd.Series]:
    """Normalize common EasyICU time formats to hourly bins."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.floor('H')
    if pd.api.types.is_object_dtype(series):
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().any():
            return parsed.dt.floor('H')
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().any():
            col_lower = col_name.lower()
            if 'second' in col_lower:
                return (numeric / 3600).floordiv(1)
            if 'minute' in col_lower or 'offset' in col_lower:
                return (numeric / 60).floordiv(1)
            return numeric.floordiv(1)
        return None
    if pd.api.types.is_numeric_dtype(series):
        col_lower = col_name.lower()
        if 'second' in col_lower:
            return (series / 3600).floordiv(1)
        if 'minute' in col_lower or 'offset' in col_lower:
            return (series / 60).floordiv(1)
        return series.floordiv(1)
    return None


def _get_quality_cohort_patient_count(state: dict[str, Any]) -> int:
    """Choose the cohort denominator shown in the current Quick Visualization session."""
    patient_ids = state.get('patient_ids') or []
    if patient_ids:
        return len(patient_ids)

    all_patient_count = int(state.get('all_patient_count') or 0)
    if all_patient_count > 0:
        return all_patient_count

    mock_params = state.get('mock_params', {}) or {}
    mock_patient_count = int(mock_params.get('n_patients') or 0)
    if mock_patient_count > 0:
        return mock_patient_count

    id_col = state.get('id_col')
    max_patients_found = 0
    if id_col:
        for df in state.get('loaded_concepts', {}).values():
            if isinstance(df, pd.DataFrame) and not df.empty and id_col in df.columns:
                max_patients_found = max(max_patients_found, int(df[id_col].nunique()))
    if max_patients_found > 0:
        return max_patients_found

    patient_limit = int(state.get('patient_limit') or 0)
    return patient_limit if patient_limit > 0 else 0


def _count_quality_event_occurrences(series: pd.Series) -> int:
    """Count event occurrences instead of treating all non-null rows as observed values."""
    if pd.api.types.is_bool_dtype(series):
        return int(series.fillna(False).sum())
    if pd.api.types.is_numeric_dtype(series):
        return int((series.fillna(0) > 0).sum())
    return int(series.notna().sum())


def _choose_concept_value_column(concept: str, df: pd.DataFrame) -> Optional[str]:
    """Pick the most clinically useful numeric value column for a concept frame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    value_cols = [c for c in numeric_cols if c not in QUALITY_EXCLUDE_COLUMNS]
    if not value_cols:
        return None
    if concept in value_cols:
        return concept

    for candidate in PRIMARY_VALUE_COLUMN_HINTS.get(concept, []):
        if candidate in value_cols:
            return candidate

    return value_cols[0]


def _get_concept_numeric_value_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric value columns after excluding IDs and time-like metadata."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    numeric_cols = df.select_dtypes(include=['number']).columns
    return [c for c in numeric_cols if c not in QUALITY_EXCLUDE_COLUMNS]


def _expected_observation_count(
    concept: str,
    patient_df: pd.DataFrame,
    los_icu: Optional[float],
    *,
    demo_hours: Optional[int] = None,
    fallback_hours: Optional[int] = None,
) -> tuple[int, str]:
    """Return the expected hourly observation denominator and its provenance tag."""
    if not isinstance(patient_df, pd.DataFrame):
        return 0, 'empty'

    time_col = _quality_detect_time_col(patient_df)
    if time_col is None and concept in QUALITY_STATIC_BOOLEAN_EVENTS:
        return 1, 'static'
    if time_col is None:
        return 1, 'static'

    if demo_hours is not None and int(demo_hours) > 0:
        return int(demo_hours), 'demo'

    los_value = pd.to_numeric(pd.Series([los_icu]), errors='coerce').iloc[0]
    if pd.notna(los_value) and float(los_value) > 0:
        return max(1, int(np.ceil(float(los_value) * 24))), 'los'

    if fallback_hours is not None and int(fallback_hours) > 0:
        return int(fallback_hours), '72h'

    return 72, '72h'


def _compute_quality_out_of_physio_rate(concept: str, df: pd.DataFrame) -> float:
    """Measure the share of non-null rows that are outside harmonized physiologic bounds."""
    bounds = PHYSIOLOGIC_RANGES.get(concept)
    value_col = _choose_concept_value_column(concept, df)
    if bounds is None or value_col is None or value_col not in df.columns:
        return 0.0

    values = pd.to_numeric(df[value_col], errors='coerce').dropna()
    if values.empty:
        return 0.0

    lower, upper = bounds
    out_of_range = ((values < lower) | (values > upper)).mean() * 100
    return float(out_of_range)


def _compute_quality_duplicate_timestamp_rate(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
) -> float:
    """Count duplicate rows where the same patient and concept share the same timestamp."""
    if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns:
        return 0.0

    time_col = _quality_detect_time_col(df)
    value_col = _choose_concept_value_column(concept, df)
    if time_col is None or value_col is None or value_col not in df.columns:
        return 0.0

    observed = df[[id_col, time_col, value_col]].dropna(subset=[time_col, value_col]).copy()
    if observed.empty:
        return 0.0

    duplicate_rows = observed.duplicated(subset=[id_col, time_col], keep='first').sum()
    return float(duplicate_rows / len(observed) * 100)


def _summarize_quality_temporal_density(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
    fallback_hours: Optional[int] = None,
) -> dict[str, float]:
    """Summarize records-per-patient-hour using median and IQR to resist ICU long tails.

    Vectorized: one groupby size() pass plus an aligned division, no per-patient loop.
    """
    empty = {'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'n_patients': 0}
    if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns:
        return empty

    value_col = _choose_concept_value_column(concept, df)
    if value_col is None or value_col not in df.columns:
        return empty

    seen_patient_ids = df[id_col].dropna().unique().tolist()
    if not seen_patient_ids:
        return empty

    expected_per_patient, _sources = _vectorized_expected_per_patient(
        seen_patient_ids,
        los_by_patient=los_by_patient,
        demo_hours=demo_hours,
        fallback_hours=fallback_hours,
    )

    value_not_na = df.loc[df[value_col].notna(), [id_col]]
    if value_not_na.empty:
        return empty
    obs_counts = (
        value_not_na.groupby(id_col, observed=False)
        .size()
        .astype('float64')
        .reindex(pd.Index(seen_patient_ids), fill_value=0)
    )

    expected = expected_per_patient.astype('float64')
    valid = expected > 0
    if not valid.any():
        return empty

    densities = (obs_counts[valid] / expected[valid]).replace([np.inf, -np.inf], np.nan).dropna()
    if densities.empty:
        return empty

    return {
        'median': float(densities.median()),
        'q25': float(densities.quantile(0.25)),
        'q75': float(densities.quantile(0.75)),
        'n_patients': int(len(densities)),
    }


def _filter_patient_selector_options(
    patient_ids: list[Any],
    *,
    query: str = "",
    max_display: int = 200,
) -> list[Any]:
    """Filter and cap patient selector options so large cohorts stay responsive."""
    unique_patient_ids = list(dict.fromkeys(patient_ids))
    trimmed_query = str(query or "").strip()
    if trimmed_query:
        unique_patient_ids = [pid for pid in unique_patient_ids if trimmed_query in str(pid)]
    return unique_patient_ids[:max(1, int(max_display))]


def _patient_selector(
    *,
    patient_ids: list[Any],
    state_key: str,
    label: str,
    lang: str,
    max_display: int = 200,
    default_patient: Any = None,
) -> Any:
    """Render a searchable patient selector with a capped option list."""
    search_label = "Search Patient ID" if lang == 'en' else "搜索患者ID"
    search_placeholder = "Type to filter..." if lang == 'en' else "输入ID过滤..."
    search_query = st.text_input(
        search_label,
        key=f"{state_key}_search",
        placeholder=search_placeholder,
    )
    options = _filter_patient_selector_options(
        patient_ids,
        query=search_query,
        max_display=max_display,
    )
    if default_patient is not None and default_patient not in options and default_patient in patient_ids:
        options = [default_patient] + options[: max(0, max_display - 1)]
    if not options:
        options = _filter_patient_selector_options(patient_ids, max_display=max_display)
    if not options:
        return None

    select_kwargs: dict[str, Any] = {
        'label': label,
        'options': options,
        'key': state_key,
    }
    if state_key not in st.session_state and default_patient in options:
        select_kwargs['index'] = options.index(default_patient)
    return st.selectbox(**select_kwargs)


def _get_quality_cohort_patient_ids(state: dict[str, Any]) -> list[Any]:
    """Return the current patient universe for quality metrics whenever it is known."""
    patient_ids = state.get('patient_ids') or []
    if patient_ids:
        return list(dict.fromkeys(patient_ids))

    id_col = state.get('id_col')
    if not id_col:
        return []

    candidate_frames: list[pd.DataFrame] = []
    loaded_concepts = state.get('loaded_concepts', {}) or {}
    for concept_name in ('age', 'sex', 'death', 'los_icu'):
        frame = loaded_concepts.get(concept_name)
        if isinstance(frame, pd.DataFrame) and not frame.empty and id_col in frame.columns:
            candidate_frames.append(frame)
    if not candidate_frames:
        for frame in loaded_concepts.values():
            if isinstance(frame, pd.DataFrame) and not frame.empty and id_col in frame.columns:
                candidate_frames.append(frame)
                if len(candidate_frames) >= 3:
                    break

    patient_pool: list[Any] = []
    for frame in candidate_frames:
        patient_pool.extend(frame[id_col].dropna().tolist())
    return list(dict.fromkeys(patient_pool))


def _get_quality_los_by_patient(state: dict[str, Any]) -> Optional[pd.Series]:
    """Build a per-patient LOS series in days when available for denominator estimation."""
    loaded_concepts = state.get('loaded_concepts', {}) or {}
    los_df = loaded_concepts.get('los_icu')
    id_col = state.get('id_col')
    if not isinstance(los_df, pd.DataFrame) or los_df.empty or not id_col or id_col not in los_df.columns:
        return None
    if 'los_icu' not in los_df.columns:
        return None

    los_copy = los_df[[id_col, 'los_icu']].copy()
    los_copy['los_icu'] = pd.to_numeric(los_copy['los_icu'], errors='coerce')
    los_copy = los_copy.dropna(subset=['los_icu'])
    if los_copy.empty:
        return None
    return los_copy.groupby(id_col, observed=False)['los_icu'].max()


def _format_quality_density(summary: dict[str, float], lang: str) -> str:
    """Format median/IQR records-per-patient-hour text for the quality table."""
    if not summary or int(summary.get('n_patients', 0)) == 0:
        return '-' if lang == 'en' else '—'
    return f"{summary['median']:.2f} [{summary['q25']:.2f}-{summary['q75']:.2f}]"


def _get_quality_denominator_note(tag: str, lang: str) -> str:
    """Explain denominator provenance tags shown in the quality table."""
    notes = {
        'd=los': "LOS-based expected hours" if lang == 'en' else "按患者 ICU LOS 估算期望小时数",
        'd=72h': "72 h fallback window" if lang == 'en' else "使用 72 小时兜底窗口",
        'd=demo': "demo simulation horizon" if lang == 'en' else "演示数据预设时间窗",
        'd=static': "single observation per patient" if lang == 'en' else "每位患者单次静态观测",
        'd=mixed': "mixed LOS / fallback denominators" if lang == 'en' else "混合使用 LOS 与兜底分母",
    }
    return notes.get(str(tag or '').lower(), tag)


def _smd_severity_tag(value: float, lang: str) -> str:
    """Attach an interpretable balance flag next to SMD values."""
    abs_value = abs(float(value))
    if abs_value > 0.25:
        return "🔴 large" if lang == 'en' else "🔴 较大"
    if abs_value > 0.10:
        return "🟠 mild" if lang == 'en' else "🟠 轻度"
    return "🟢 balanced" if lang == 'en' else "🟢 平衡"


def _compute_smd_continuous(series1: pd.Series, series2: pd.Series) -> float:
    """Compute standardized mean difference for continuous variables."""
    values1 = pd.to_numeric(series1, errors='coerce').dropna()
    values2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(values1) < 2 or len(values2) < 2:
        return 0.0

    sd1 = float(values1.std(ddof=1))
    sd2 = float(values2.std(ddof=1))
    pooled_sd = np.sqrt((sd1 ** 2 + sd2 ** 2) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((values1.mean() - values2.mean()) / pooled_sd)


def _compute_smd_binary(series1: pd.Series, series2: pd.Series) -> float:
    """Compute standardized mean difference for binary variables."""
    values1 = pd.to_numeric(series1, errors='coerce').dropna()
    values2 = pd.to_numeric(series2, errors='coerce').dropna()
    if values1.empty or values2.empty:
        return 0.0

    p1 = float(values1.mean())
    p2 = float(values2.mean())
    p_bar = (p1 + p2) / 2
    denom = np.sqrt(p_bar * (1 - p_bar))
    if denom == 0:
        return 0.0
    return float((p1 - p2) / denom)


def _vectorized_expected_per_patient(
    patient_universe: list[Any],
    *,
    los_by_patient: Optional[pd.Series],
    demo_hours: Optional[int],
    fallback_hours: Optional[int],
) -> tuple[pd.Series, pd.Series]:
    """Return (expected_count, source_tag) Series indexed by patient id.

    Vectorizes the per-patient branch of `_expected_observation_count` for the
    time-series case where time_col is already known to exist on the frame.
    """
    universe_index = pd.Index(patient_universe)
    fallback = int(fallback_hours) if fallback_hours and int(fallback_hours) > 0 else 72

    if demo_hours is not None and int(demo_hours) > 0:
        expected = pd.Series(int(demo_hours), index=universe_index, dtype='int64')
        sources = pd.Series('demo', index=universe_index)
        return expected, sources

    if isinstance(los_by_patient, pd.Series) and not los_by_patient.empty:
        los_aligned = pd.to_numeric(los_by_patient.reindex(universe_index), errors='coerce')
    else:
        los_aligned = pd.Series(np.nan, index=universe_index, dtype='float64')

    expected = pd.Series(fallback, index=universe_index, dtype='int64')
    sources = pd.Series('72h', index=universe_index)

    los_valid = los_aligned.notna() & (los_aligned > 0)
    if los_valid.any():
        los_hours = np.ceil(los_aligned[los_valid].astype('float64') * 24).astype('int64')
        los_hours = np.maximum(1, los_hours)
        expected.loc[los_valid] = los_hours
        sources.loc[los_valid] = 'los'

    return expected, sources


def _build_quality_metric_profile(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    cohort_patient_count: int,
    time_grid_size: int,
    cohort_patient_ids: Optional[list[Any]] = None,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
) -> dict[str, Any]:
    """Compute one concept-level QC profile shared by the table and chart views.

    Performance: replaces the old O(P * N) per-patient loop with a single
    vectorized pass that (a) computes expected counts via aligned Series
    operations and (b) folds temporal density into the same groupby pass.
    """
    profile = {
        'missing_rate': 100.0,
        'out_of_physio_rate': 0.0,
        'duplicate_rate': 0.0,
        'denominator_tag': 'd=72h',
        'expected_observations': 0,
        'observed_observations': 0,
        'temporal_density': {'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'n_patients': 0},
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        return profile

    value_col = _choose_concept_value_column(concept, df)
    time_col = _quality_detect_time_col(df)
    n_patients = int(df[id_col].nunique()) if id_col in df.columns else 0
    cohort_patient_count = int(cohort_patient_count or 0)

    if concept in QUALITY_STATIC_BOOLEAN_EVENTS and not time_col:
        denominator = cohort_patient_count or n_patients
        if denominator > 0:
            profile['missing_rate'] = float(max(0.0, min(100.0, (1 - min(n_patients, denominator) / denominator) * 100)))
            profile['expected_observations'] = denominator
            profile['observed_observations'] = min(n_patients, denominator)
            profile['denominator_tag'] = 'd=static'
        return profile

    if value_col is None or value_col not in df.columns:
        if cohort_patient_count > 0 and n_patients > 0:
            patient_coverage_missing = (1 - min(n_patients, cohort_patient_count) / cohort_patient_count) * 100
            profile['missing_rate'] = float(max(0.0, min(100.0, patient_coverage_missing)))
        return profile

    profile['out_of_physio_rate'] = _compute_quality_out_of_physio_rate(concept, df)
    profile['duplicate_rate'] = _compute_quality_duplicate_timestamp_rate(concept=concept, df=df, id_col=id_col)

    raw_na_rate = float(df[value_col].isna().mean() * 100)
    if concept in QUALITY_DEMOGRAPHIC_STATIC:
        profile['missing_rate'] = raw_na_rate
        profile['denominator_tag'] = 'd=static'
        return profile

    if time_col and id_col in df.columns:
        seen_patient_ids = df[id_col].dropna().unique().tolist()
        patient_universe = cohort_patient_ids or seen_patient_ids
        patient_universe = list(dict.fromkeys(patient_universe))
        fallback_hours = time_grid_size if time_grid_size > 0 else None

        expected_per_patient, source_per_patient = _vectorized_expected_per_patient(
            patient_universe,
            los_by_patient=los_by_patient,
            demo_hours=demo_hours,
            fallback_hours=fallback_hours,
        )

        expected_total = int(expected_per_patient.sum())
        unique_sources = sorted(set(source_per_patient.tolist()))

        if not cohort_patient_ids and cohort_patient_count > len(seen_patient_ids):
            missing_patient_count = cohort_patient_count - len(seen_patient_ids)
            default_expected, default_source = _expected_observation_count(
                concept=concept,
                patient_df=df,
                los_icu=None,
                demo_hours=demo_hours,
                fallback_hours=fallback_hours,
            )
            if default_expected > 0:
                expected_total += missing_patient_count * default_expected
                if default_source not in unique_sources:
                    unique_sources = sorted(set(unique_sources + [default_source]))

        source_label = unique_sources[0] if len(unique_sources) == 1 else 'mixed'

        hour_bins = _quality_to_hour_bins(df[time_col], time_col)
        if hour_bins is not None:
            if concept in QUALITY_EVENT_TIME_SERIES:
                if pd.api.types.is_bool_dtype(df[value_col]):
                    observed_mask = df[value_col].astype('boolean').fillna(False)
                elif pd.api.types.is_numeric_dtype(df[value_col]):
                    observed_mask = df[value_col].fillna(0) > 0
                else:
                    observed_mask = df[value_col].notna()
            else:
                observed_mask = df[value_col].notna()

            observed = df.loc[observed_mask, [id_col]].copy()
            observed['_hour_bin'] = hour_bins.loc[observed.index]
            observed = observed.dropna(subset=['_hour_bin'])
            observed_total = int(observed.drop_duplicates(subset=[id_col, '_hour_bin']).shape[0])

            if expected_total > 0:
                coverage_missing = (1 - observed_total / expected_total) * 100
                profile['missing_rate'] = float(max(raw_na_rate, max(0.0, min(100.0, coverage_missing))))
                profile['expected_observations'] = expected_total
                profile['observed_observations'] = observed_total
                profile['denominator_tag'] = f"d={source_label}"

        # Temporal density: vectorized per-patient count using fast groupby+size,
        # aligned against expected_per_patient. Replaces the old O(P * N) loop.
        seen_index = pd.Index(seen_patient_ids)
        if len(seen_index) > 0:
            value_not_na = df.loc[df[value_col].notna(), [id_col]]
            if not value_not_na.empty:
                obs_counts = (
                    value_not_na.groupby(id_col, observed=False)
                    .size()
                    .astype('float64')
                )
            else:
                obs_counts = pd.Series(dtype='float64')
            obs_counts_aligned = obs_counts.reindex(seen_index, fill_value=0)
            expected_for_seen = expected_per_patient.reindex(seen_index)
            if expected_for_seen.isna().any():
                expected_for_seen = expected_for_seen.fillna(
                    int(fallback_hours) if fallback_hours and int(fallback_hours) > 0 else 72
                )
            expected_for_seen = expected_for_seen.astype('float64')
            valid_mask = expected_for_seen > 0
            if valid_mask.any():
                densities = obs_counts_aligned[valid_mask] / expected_for_seen[valid_mask]
                densities = densities.replace([np.inf, -np.inf], np.nan).dropna()
                if len(densities) > 0:
                    profile['temporal_density'] = {
                        'median': float(densities.median()),
                        'q25': float(densities.quantile(0.25)),
                        'q75': float(densities.quantile(0.75)),
                        'n_patients': int(len(densities)),
                    }
        return profile

    if cohort_patient_count > 0 and n_patients > 0:
        patient_coverage_missing = (1 - min(n_patients, cohort_patient_count) / cohort_patient_count) * 100
        profile['missing_rate'] = float(max(raw_na_rate, max(0.0, min(100.0, patient_coverage_missing))))
        profile['expected_observations'] = cohort_patient_count
        profile['observed_observations'] = min(n_patients, cohort_patient_count)
    else:
        profile['missing_rate'] = raw_na_rate

    return profile


def _cohort_cache_fingerprint(cohort_patient_ids: Optional[list[Any]]) -> tuple[Any, ...]:
    """Cheap O(1-ish) fingerprint to key the per-concept quality cache."""
    if not cohort_patient_ids:
        return (0,)
    head = str(cohort_patient_ids[0]) if len(cohort_patient_ids) else ''
    tail = str(cohort_patient_ids[-1]) if len(cohort_patient_ids) else ''
    return (len(cohort_patient_ids), head, tail)


def _los_cache_fingerprint(los_by_patient: Optional[pd.Series]) -> tuple[Any, ...]:
    """Cheap fingerprint for the LOS series used in quality denominators."""
    if not isinstance(los_by_patient, pd.Series) or los_by_patient.empty:
        return (0,)
    try:
        head_idx = str(los_by_patient.index[0])
    except Exception:
        head_idx = ''
    try:
        # sum() is vectorized C; keeps the fingerprint sensitive to content edits
        values_sum = float(pd.to_numeric(los_by_patient, errors='coerce').fillna(0).sum())
    except Exception:
        values_sum = 0.0
    return (len(los_by_patient), head_idx, round(values_sum, 4))


def _build_quality_metric_profile_cached(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    cohort_patient_count: int,
    time_grid_size: int,
    cohort_patient_ids: Optional[list[Any]] = None,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
) -> dict[str, Any]:
    """Session-scoped cache wrapper around `_build_quality_metric_profile`.

    The cache is keyed by a cheap structural fingerprint of the inputs so that
    re-rendering the Quality page (language toggles, tab switches, sidebar
    interactions) does not re-run the whole QC pipeline for every concept.
    The cache lives on `st.session_state` and is naturally invalidated when
    `loaded_concepts` is rebuilt (df identity changes).
    """
    try:
        cache = st.session_state.setdefault('_quality_profile_cache', {})
    except Exception:
        # When called outside a Streamlit run (e.g. tests) just skip caching.
        return _build_quality_metric_profile(
            concept=concept,
            df=df,
            id_col=id_col,
            cohort_patient_count=cohort_patient_count,
            time_grid_size=time_grid_size,
            cohort_patient_ids=cohort_patient_ids,
            los_by_patient=los_by_patient,
            demo_hours=demo_hours,
        )

    key = (
        str(concept),
        str(id_col),
        id(df),
        tuple(df.shape) if isinstance(df, pd.DataFrame) else (0, 0),
        int(cohort_patient_count or 0),
        int(time_grid_size or 0),
        int(demo_hours) if demo_hours else None,
        _cohort_cache_fingerprint(cohort_patient_ids),
        _los_cache_fingerprint(los_by_patient),
    )

    cached = cache.get(key)
    if cached is not None:
        return cached

    result = _build_quality_metric_profile(
        concept=concept,
        df=df,
        id_col=id_col,
        cohort_patient_count=cohort_patient_count,
        time_grid_size=time_grid_size,
        cohort_patient_ids=cohort_patient_ids,
        los_by_patient=los_by_patient,
        demo_hours=demo_hours,
    )
    # Guard against unbounded growth across long sessions.
    if len(cache) > 512:
        cache.clear()
    cache[key] = result
    return result


def _compute_quality_missing_rate(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    cohort_patient_count: int,
    time_grid_size: int,
    cohort_patient_ids: Optional[list[Any]] = None,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
) -> float:
    """Compute a consistent concept-level missing rate for both table and chart views."""
    profile = _build_quality_metric_profile_cached(
        concept=concept,
        df=df,
        id_col=id_col,
        cohort_patient_count=cohort_patient_count,
        time_grid_size=time_grid_size,
        cohort_patient_ids=cohort_patient_ids,
        los_by_patient=los_by_patient,
        demo_hours=demo_hours,
    )
    return float(profile['missing_rate'])
