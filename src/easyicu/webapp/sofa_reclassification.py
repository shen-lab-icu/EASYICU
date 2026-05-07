"""SOFA-1 vs SOFA-2 reclassification calculations and helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

_PROTECTED_NAMES = {
    'SOFA_RECLASS_ORGANS',
    'SOFA_RECLASS_ANALYSIS_MODES',
    '_generate_mock_sofa_timeseries_concepts',
    '_demo_cohort_fingerprint',
    '_get_demo_sofa_timeseries_concepts',
    '_get_sofa_reclassification_mode_availability',
    '_sofa_severity_group',
    '_build_sofa_reclassification_stats',
    '_build_reclassification_df_from_loaded_concepts',
    '_get_sofa_reclassification_source',
    '_render_reclassification_cards',
    '_render_reclassification_snapshot',
    '_APP_CONTEXT',
    '_PROTECTED_NAMES',
    '_install_app_context',
    'Any',
    'Dict',
    'np',
    'pd',
}
_APP_CONTEXT: dict[str, Any] = {}


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose remaining app helpers and Streamlit state for this transitional module."""
    _APP_CONTEXT.clear()
    _APP_CONTEXT.update(app_context)
    for name, value in app_context.items():
        if name.startswith('__') or name in _PROTECTED_NAMES:
            continue
        globals()[name] = value


SOFA_RECLASS_ORGANS = [
    ('resp', 'Respiratory', '呼吸'),
    ('coag', 'Coagulation', '凝血'),
    ('liver', 'Liver', '肝脏'),
    ('cardio', 'Cardiovascular', '循环'),
    ('cns', 'Neurological', '神经'),
    ('renal', 'Renal', '肾脏'),
]

SOFA_RECLASS_ANALYSIS_MODES = {
    'worst_icu': {
        'label_en': 'Worst ICU score',
        'label_zh': 'ICU期间最高分',
        'description_en': 'Patient-level maximum SOFA-1 and maximum SOFA-2 across the ICU stay.',
        'description_zh': '按患者汇总 ICU 全程 SOFA-1 和 SOFA-2 的最高值。',
    },
    'first24_worst': {
        'label_en': 'First 24h paired worst',
        'label_zh': '首24小时配对最高分',
        'description_en': 'Patient-level maximum from time-aligned SOFA-1/SOFA-2 points during the first 24 ICU hours.',
        'description_zh': '仅使用入 ICU 后 0-24 小时内同一时间点配对的 SOFA-1/SOFA-2，并按患者取最高值。',
    },
    'time_aligned': {
        'label_en': 'Time-aligned points',
        'label_zh': '同时间点配对',
        'description_en': 'Row-level comparison at the same stay_id and charttime; denominator is paired time points.',
        'description_zh': '在相同 stay_id 和 charttime 上逐点比较；分母为配对时间点。',
    },
}


def _generate_mock_sofa_timeseries_concepts(cohort_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create paired SOFA-1/SOFA-2 time-series concepts from demo cohort rows.

    Fully vectorized over patients × time-points × organs — no iterrows loop.
    """
    if not isinstance(cohort_df, pd.DataFrame) or cohort_df.empty or 'stay_id' not in cohort_df.columns:
        return {}

    valid = cohort_df.dropna(subset=['stay_id']).reset_index(drop=True)
    if valid.empty:
        return {}

    rng = np.random.default_rng(20260424)
    time_points = np.array([-6, 0, 6, 12, 18, 24, 36, 48, 60, 72], dtype=int)
    n_times = int(len(time_points))
    organ_keys = [key for key, _label_en, _label_zh in SOFA_RECLASS_ORGANS]
    n_organs = int(len(organ_keys))

    stay_ids = valid['stay_id'].to_numpy()
    n_patients = int(len(stay_ids))

    peak_choices = np.array([0, 6, 12, 18, 24, 36], dtype=int)
    peak_times = rng.choice(peak_choices, size=n_patients)  # (P,)

    distances = np.abs(time_points[None, :] - peak_times[:, None]) / 18.0  # (P, T)
    recovery = np.minimum(2, np.floor(distances)).astype(np.int64)  # (P, T)

    # Gather per-patient organ peaks for SOFA-1 and SOFA-2
    sofa1_peaks = np.zeros((n_patients, n_organs), dtype=np.int64)
    sofa2_peaks = np.zeros((n_patients, n_organs), dtype=np.int64)
    for j, key in enumerate(organ_keys):
        sofa1_col = f'sofa1_{key}'
        sofa2_col = f'sofa2_{key}'
        if sofa1_col in valid.columns:
            s1 = pd.to_numeric(valid[sofa1_col], errors='coerce').fillna(0).to_numpy()
        else:
            s1 = np.zeros(n_patients, dtype=np.float64)
        if sofa2_col in valid.columns:
            s2_raw = pd.to_numeric(valid[sofa2_col], errors='coerce')
            s2 = s2_raw.fillna(pd.Series(s1, index=valid.index)).to_numpy()
        else:
            s2 = s1
        sofa1_peaks[:, j] = np.clip(s1, 0, 4).astype(np.int64)
        sofa2_peaks[:, j] = np.clip(s2, 0, 4).astype(np.int64)

    # values[j, p, t] = clip(peak[p, j] - recovery[p, t] - noise[j, p, t], 0, 4)
    noise1 = rng.integers(0, 2, size=(n_organs, n_patients, n_times))
    noise2 = rng.integers(0, 2, size=(n_organs, n_patients, n_times))

    s1_peaks_ex = sofa1_peaks.T[:, :, None]  # (n_organs, P, 1)
    s2_peaks_ex = sofa2_peaks.T[:, :, None]
    recovery_ex = recovery[None, :, :]       # (1, P, T)

    values1 = np.clip(s1_peaks_ex - recovery_ex - noise1, 0, 4).astype(np.int64)
    values2 = np.clip(s2_peaks_ex - recovery_ex - noise2, 0, 4).astype(np.int64)

    # Restore the exact peak at peak_idx per patient (match original semantics).
    peak_idx = np.argmin(np.abs(time_points[None, :] - peak_times[:, None]), axis=1)  # (P,)
    patient_range = np.arange(n_patients)
    for j in range(n_organs):
        values1[j, patient_range, peak_idx] = sofa1_peaks[:, j]
        values2[j, patient_range, peak_idx] = sofa2_peaks[:, j]

    sofa1_total = np.clip(values1.sum(axis=0), 0, 24).astype(np.int64)  # (P, T)
    sofa2_total = np.clip(values2.sum(axis=0), 0, 24).astype(np.int64)

    stay_repeated = np.repeat(stay_ids, n_times)
    time_tiled = np.tile(time_points, n_patients)

    rows_by_concept: Dict[str, pd.DataFrame] = {}
    for j, key in enumerate(organ_keys):
        rows_by_concept[f'sofa_{key}'] = pd.DataFrame({
            'stay_id': stay_repeated,
            'charttime': time_tiled,
            f'sofa_{key}': values1[j].ravel(),
        })
        rows_by_concept[f'sofa2_{key}'] = pd.DataFrame({
            'stay_id': stay_repeated,
            'charttime': time_tiled,
            f'sofa2_{key}': values2[j].ravel(),
        })

    rows_by_concept['sofa'] = pd.DataFrame({
        'stay_id': stay_repeated,
        'charttime': time_tiled,
        'sofa': sofa1_total.ravel(),
    })
    rows_by_concept['sofa2'] = pd.DataFrame({
        'stay_id': stay_repeated,
        'charttime': time_tiled,
        'sofa2': sofa2_total.ravel(),
    })

    if 'mortality' in valid.columns:
        death_vals = valid['mortality'].fillna(False).astype(bool).astype(np.int64).to_numpy()
    else:
        death_vals = np.zeros(n_patients, dtype=np.int64)
    rows_by_concept['death'] = pd.DataFrame({
        'stay_id': stay_ids,
        'death': death_vals,
    })

    if 'los_days' in valid.columns:
        los_days = pd.to_numeric(valid['los_days'], errors='coerce')
    else:
        los_days = pd.Series(np.nan, index=valid.index, dtype='float64')
    if 'los_hours' in valid.columns:
        los_hours = pd.to_numeric(valid['los_hours'], errors='coerce')
        los_days = los_days.fillna(los_hours / 24)
    rows_by_concept['los_icu'] = pd.DataFrame({
        'stay_id': stay_ids,
        'los_icu': los_days.to_numpy(),
    })

    return rows_by_concept


def _demo_cohort_fingerprint(cohort_df: pd.DataFrame) -> tuple[Any, ...]:
    """Cheap cohort fingerprint so cached demo SOFA series survive reruns."""
    if not isinstance(cohort_df, pd.DataFrame) or cohort_df.empty or 'stay_id' not in cohort_df.columns:
        return ('empty',)
    try:
        head_id = str(cohort_df['stay_id'].iloc[0])
    except Exception:
        head_id = ''
    try:
        tail_id = str(cohort_df['stay_id'].iloc[-1])
    except Exception:
        tail_id = ''
    return (id(cohort_df), int(len(cohort_df)), head_id, tail_id)


def _get_demo_sofa_timeseries_concepts() -> Dict[str, pd.DataFrame]:
    """Return cached demo SOFA time-series concepts for the current session."""
    demo_sources = []
    dash_df = st.session_state.get('dash_demographics')
    if st.session_state.get('dash_is_demo') and isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
        demo_sources.append(dash_df)
    reclass_df = st.session_state.get('reclass_demo_df')
    if isinstance(reclass_df, pd.DataFrame) and not reclass_df.empty:
        demo_sources.append(reclass_df)
    if not demo_sources:
        return {}

    source_df = demo_sources[0]
    cache = st.session_state.setdefault('_demo_sofa_ts_cache', {})
    fingerprint = _demo_cohort_fingerprint(source_df)
    cached = cache.get(fingerprint)
    if cached is not None:
        return cached

    result = _generate_mock_sofa_timeseries_concepts(source_df)
    # Keep cache small; this helper is called on nearly every rerun.
    if len(cache) > 8:
        cache.clear()
    cache[fingerprint] = result
    return result


def _get_sofa_reclassification_mode_availability(loaded_concepts: Dict[str, Any]) -> Dict[str, list[str]]:
    """Report which SOFA sensitivity definitions are actually available in the current session."""
    available = ['worst_icu']
    locked: list[str] = []

    for mode in ['first24_worst', 'time_aligned']:
        if _build_reclassification_df_from_loaded_concepts(loaded_concepts, mode=mode).empty:
            locked.append(mode)
        else:
            available.append(mode)

    return {'available': available, 'locked': locked}


def _sofa_severity_group(series: pd.Series) -> pd.Series:
    """Map SOFA scores to compact severity groups used across cohort plots."""
    return pd.cut(
        pd.to_numeric(series, errors='coerce'),
        bins=[-np.inf, 3, 6, 10, np.inf],
        labels=['0-2', '3-5', '6-9', '>=10'],
        right=False,
    )


def _build_sofa_reclassification_stats(df: pd.DataFrame, lang: str = 'en') -> Dict[str, Any]:
    """Summarize cohort-level severity changes between SOFA-1 and SOFA-2."""
    sofa1 = _cohort_numeric_series(df, ['sofa1_max', 'sofa1', 'sofa'])
    sofa2 = _cohort_numeric_series(df, ['sofa2_max', 'sofa2'])
    analysis_unit = 'patients'
    if 'analysis_unit' in df.columns and df['analysis_unit'].notna().any():
        analysis_unit = str(df['analysis_unit'].dropna().iloc[0])
    is_timepoint_unit = analysis_unit == 'timepoints'
    denominator_label = "Paired points" if is_timepoint_unit and lang == 'en' else (
        "配对时间点" if is_timepoint_unit else ("Patients" if lang == 'en' else "患者数")
    )
    denominator_hint = "time-aligned rows" if is_timepoint_unit and lang == 'en' else (
        "同时间点记录" if is_timepoint_unit else ("paired SOFA" if lang == 'en' else "双SOFA记录")
    )
    empty_summary = pd.DataFrame(columns=['group', 'patients', 'pct', 'mortality', 'median_los'])
    empty_matrix = pd.DataFrame(columns=['SOFA-1', 'SOFA-2', 'patients'])
    empty_organ = pd.DataFrame(columns=['organ', 'mean_delta', 'mean_abs_delta', 'up', 'down'])
    empty_rows = pd.DataFrame(columns=['stay_id', 'sofa1', 'sofa2', 'delta', 'group'])
    empty_metrics = {
        'patients': '0',
        'denominator': '0',
        'denominator_label': denominator_label,
        'denominator_hint': denominator_hint,
        'patient_count': '0',
        'discordant_pct': 'NA',
        'up_pct': 'NA',
        'down_pct': 'NA',
        'median_delta': 'NA',
    }

    if sofa1 is None or sofa2 is None:
        return {
            'available': False,
            'rows': empty_rows,
            'summary': empty_summary,
            'matrix': empty_matrix,
            'organ': empty_organ,
            'metrics': empty_metrics,
        }

    work = pd.DataFrame({
        'stay_id': df['stay_id'] if 'stay_id' in df.columns else np.arange(len(df)),
        'sofa1': sofa1,
        'sofa2': sofa2,
    }).dropna(subset=['sofa1', 'sofa2']).copy()
    if work.empty:
        return {
            'available': False,
            'rows': empty_rows,
            'summary': empty_summary,
            'matrix': empty_matrix,
            'organ': empty_organ,
            'metrics': empty_metrics,
        }

    if 'charttime' in df.columns:
        work['charttime'] = df.loc[work.index, 'charttime'].to_numpy()
    work['delta'] = work['sofa2'] - work['sofa1']
    group_labels = {
        'up': 'Up-classified' if lang == 'en' else '上调分层',
        'same': 'Same' if lang == 'en' else '不变',
        'down': 'Down-classified' if lang == 'en' else '下调分层',
    }
    work['group'] = np.select(
        [work['delta'] > 0, work['delta'] < 0],
        [group_labels['up'], group_labels['down']],
        default=group_labels['same'],
    )
    work['SOFA-1'] = _sofa_severity_group(work['sofa1'])
    work['SOFA-2'] = _sofa_severity_group(work['sofa2'])

    mortality_series = _cohort_bool_series(df, ['mortality'])
    survived = _cohort_bool_series(df, ['survived'])
    if mortality_series is None and survived is not None:
        mortality_series = ~survived
    if mortality_series is not None:
        work['death'] = mortality_series.reindex(work.index).fillna(False).astype(bool).to_numpy()
    else:
        work['death'] = False

    los_hours = _cohort_numeric_series(df, ['los_hours'])
    los_days = los_hours / 24 if los_hours is not None else _cohort_numeric_series(df, ['los_days'])
    if los_days is not None:
        work['los_days'] = los_days.reindex(work.index).to_numpy()
    else:
        work['los_days'] = np.nan

    order = [group_labels['up'], group_labels['same'], group_labels['down']]
    summary = work.groupby('group', observed=False).agg(
        patients=('stay_id', 'count'),
        deaths=('death', 'sum'),
        median_los=('los_days', 'median'),
    ).reindex(order).fillna({'patients': 0, 'deaths': 0}).reset_index()
    summary['patients'] = summary['patients'].astype(int)
    summary['pct'] = np.where(len(work) > 0, (summary['patients'] / len(work) * 100).round(1), 0.0)
    summary['mortality'] = np.where(
        summary['patients'] > 0,
        (summary['deaths'] / summary['patients'] * 100).round(1),
        0.0,
    )
    summary['median_los'] = summary['median_los'].fillna(0).round(1)
    summary = summary[['group', 'patients', 'pct', 'mortality', 'median_los']]

    matrix = work.groupby(['SOFA-1', 'SOFA-2'], observed=False).size().reset_index(name='patients')

    organ_rows = []
    for key, label_en, label_zh in SOFA_RECLASS_ORGANS:
        sofa1_col = f'sofa1_{key}'
        sofa2_col = f'sofa2_{key}'
        if sofa1_col not in df.columns or sofa2_col not in df.columns:
            continue
        organ_delta = pd.to_numeric(df[sofa2_col], errors='coerce') - pd.to_numeric(df[sofa1_col], errors='coerce')
        organ_rows.append({
            'organ': label_en if lang == 'en' else label_zh,
            'mean_delta': round(float(organ_delta.mean()), 2),
            'mean_abs_delta': round(float(organ_delta.abs().mean()), 2),
            'up': int((organ_delta > 0).sum()),
            'down': int((organ_delta < 0).sum()),
        })
    organ = pd.DataFrame(organ_rows, columns=['organ', 'mean_delta', 'mean_abs_delta', 'up', 'down'])
    if not organ.empty:
        organ = organ.sort_values('mean_abs_delta', ascending=True)

    up_pct = summary.loc[summary['group'] == group_labels['up'], 'pct'].iloc[0]
    down_pct = summary.loc[summary['group'] == group_labels['down'], 'pct'].iloc[0]
    discordant_pct = round(float(up_pct + down_pct), 1)
    metrics = {
        'patients': f"{len(work):,}",
        'denominator': f"{len(work):,}",
        'denominator_label': denominator_label,
        'denominator_hint': denominator_hint,
        'patient_count': f"{work['stay_id'].nunique():,}",
        'discordant_pct': f"{discordant_pct:.1f}%",
        'up_pct': f"{up_pct:.1f}%",
        'down_pct': f"{down_pct:.1f}%",
        'median_delta': f"{work['delta'].median():.1f}",
    }

    return {
        'available': True,
        'rows': work,
        'summary': summary,
        'matrix': matrix,
        'organ': organ,
        'metrics': metrics,
    }


def _build_reclassification_df_from_loaded_concepts(
    loaded_concepts: Dict[str, Any],
    mode: str = 'worst_icu',
) -> pd.DataFrame:
    """Build SOFA-1/SOFA-2 comparison data from loaded Quick Visualization concepts."""
    if not loaded_concepts:
        return pd.DataFrame()
    if mode not in SOFA_RECLASS_ANALYSIS_MODES:
        mode = 'worst_icu'

    def _concept_frame(concept: str, output_col: str, *, require_time: bool = False) -> pd.DataFrame:
        concept_df = loaded_concepts.get(concept)
        if not isinstance(concept_df, pd.DataFrame) or concept_df.empty:
            return pd.DataFrame()
        id_col = next((c for c in ['stay_id', 'patient_id', 'subject_id'] if c in concept_df.columns), None)
        time_col = next((c for c in ['charttime', 'time', 'hours_from_admit'] if c in concept_df.columns), None)
        value_col = concept if concept in concept_df.columns else None
        if id_col is None or value_col is None or (require_time and time_col is None):
            return pd.DataFrame()
        cols = [id_col, value_col]
        if time_col:
            cols.insert(1, time_col)
        result = concept_df[cols].copy()
        rename_map = {id_col: 'stay_id', value_col: output_col}
        if time_col:
            rename_map[time_col] = 'charttime'
        result = result.rename(columns=rename_map)
        result[output_col] = pd.to_numeric(result[output_col], errors='coerce')
        result = result.dropna(subset=['stay_id', output_col])
        if require_time:
            result['charttime'] = pd.to_numeric(result['charttime'], errors='coerce')
            result = result.dropna(subset=['charttime'])
            result = result.groupby(['stay_id', 'charttime'], as_index=False)[output_col].max()
        return result

    def _max_feature_frame(concept: str, output_col: str) -> pd.DataFrame:
        concept_df = _concept_frame(concept, output_col)
        if concept_df.empty:
            return pd.DataFrame()
        return (
            concept_df[['stay_id', output_col]]
            .groupby('stay_id', as_index=False)[output_col]
            .max()
        )

    def _paired_feature_frame(sofa1_concept: str, sofa2_concept: str, sofa1_col: str, sofa2_col: str) -> pd.DataFrame:
        sofa1_frame = _concept_frame(sofa1_concept, sofa1_col, require_time=True)
        sofa2_frame = _concept_frame(sofa2_concept, sofa2_col, require_time=True)
        if sofa1_frame.empty or sofa2_frame.empty:
            return pd.DataFrame()
        return sofa1_frame.merge(sofa2_frame, on=['stay_id', 'charttime'], how='inner')

    def _merge_outcomes(result: pd.DataFrame) -> pd.DataFrame:
        if result.empty:
            return result
        for concept, output_col in [('death', 'mortality'), ('los_icu', 'los_days')]:
            concept_frame = _max_feature_frame(concept, output_col)
            if not concept_frame.empty:
                result = result.merge(concept_frame, on='stay_id', how='left')
        return result

    if mode in {'first24_worst', 'time_aligned'}:
        result = _paired_feature_frame('sofa', 'sofa2', 'sofa1_max', 'sofa2_max')
        if result.empty:
            return pd.DataFrame()

        for key, _label_en, _label_zh in SOFA_RECLASS_ORGANS:
            organ_pair = _paired_feature_frame(f'sofa_{key}', f'sofa2_{key}', f'sofa1_{key}', f'sofa2_{key}')
            if not organ_pair.empty:
                result = result.merge(organ_pair, on=['stay_id', 'charttime'], how='left')

        if mode == 'first24_worst':
            result = result[(result['charttime'] >= 0) & (result['charttime'] <= 24)].copy()
            if result.empty:
                return pd.DataFrame()
            numeric_cols = [c for c in result.columns if c not in {'stay_id', 'charttime'}]
            result = result.groupby('stay_id', as_index=False)[numeric_cols].max()
            result['analysis_unit'] = 'patients'
            result['analysis_mode'] = mode
            return _merge_outcomes(result)

        result = result.sort_values(['stay_id', 'charttime']).reset_index(drop=True)
        result['analysis_unit'] = 'timepoints'
        result['analysis_mode'] = mode
        return _merge_outcomes(result)

    result = _max_feature_frame('sofa', 'sofa1_max')
    sofa2 = _max_feature_frame('sofa2', 'sofa2_max')
    if result.empty or sofa2.empty:
        return pd.DataFrame()
    result = result.merge(sofa2, on='stay_id', how='inner')

    for key, _label_en, _label_zh in SOFA_RECLASS_ORGANS:
        sofa1_part = _max_feature_frame(f'sofa_{key}', f'sofa1_{key}')
        sofa2_part = _max_feature_frame(f'sofa2_{key}', f'sofa2_{key}')
        if not sofa1_part.empty:
            result = result.merge(sofa1_part, on='stay_id', how='left')
        if not sofa2_part.empty:
            result = result.merge(sofa2_part, on='stay_id', how='left')

    for concept, output_col in [('death', 'mortality'), ('los_icu', 'los_days')]:
        concept_frame = _max_feature_frame(concept, output_col)
        if not concept_frame.empty:
            result = result.merge(concept_frame, on='stay_id', how='left')
    result['analysis_unit'] = 'patients'
    result['analysis_mode'] = mode
    return result


def _get_sofa_reclassification_source(lang: str = 'en', mode: str = 'worst_icu') -> tuple[pd.DataFrame, str]:
    """Return the best available patient-level dataset for SOFA reclassification UI."""
    dash_df = st.session_state.get('dash_demographics')
    if mode == 'worst_icu' and isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
        stats = _build_sofa_reclassification_stats(dash_df, lang=lang)
        if stats.get('available'):
            return dash_df, "Cohort Snapshot data" if lang == 'en' else "队列快照数据"

    loaded_df = _build_reclassification_df_from_loaded_concepts(st.session_state.get('loaded_concepts', {}), mode=mode)
    if not loaded_df.empty:
        mode_label = SOFA_RECLASS_ANALYSIS_MODES.get(mode, SOFA_RECLASS_ANALYSIS_MODES['worst_icu'])
        label = mode_label['label_en'] if lang == 'en' else mode_label['label_zh']
        return loaded_df, ("Loaded Quick Visualization concepts · " + label) if lang == 'en' else ("快速可视化已载入特征 · " + label)

    demo_concepts = _get_demo_sofa_timeseries_concepts()
    demo_timeseries_df = _build_reclassification_df_from_loaded_concepts(demo_concepts, mode=mode)
    if not demo_timeseries_df.empty:
        mode_label = SOFA_RECLASS_ANALYSIS_MODES.get(mode, SOFA_RECLASS_ANALYSIS_MODES['worst_icu'])
        label = mode_label['label_en'] if lang == 'en' else mode_label['label_zh']
        return demo_timeseries_df, ("Demo SOFA time series · " + label) if lang == 'en' else ("演示SOFA时间序列 · " + label)

    demo_df = st.session_state.get('reclass_demo_df')
    if mode == 'worst_icu' and isinstance(demo_df, pd.DataFrame) and not demo_df.empty:
        return demo_df, "Demo reclassification cohort" if lang == 'en' else "演示重新分层队列"

    return pd.DataFrame(), ""


def _render_reclassification_cards(reclass: Dict[str, Any], lang: str = 'en'):
    """Render compact metric cards for SOFA reclassification summaries."""
    metrics = reclass['metrics']
    cols = st.columns(5)
    cards = [
        (metrics.get('denominator', metrics['patients']), metrics.get('denominator_label', "Patients" if lang == 'en' else "患者数"), metrics.get('denominator_hint', "paired SOFA" if lang == 'en' else "双SOFA记录"), "#2563eb", "👥"),
        (metrics['discordant_pct'], "Discordant" if lang == 'en' else "重新分层", "SOFA-2 != SOFA-1" if lang == 'en' else "SOFA-2 != SOFA-1", "#ea580c", "⇄"),
        (metrics['up_pct'], "Up-classified" if lang == 'en' else "上调分层", "higher SOFA-2" if lang == 'en' else "SOFA-2更高", "#e11d48", "↑"),
        (metrics['down_pct'], "Down-classified" if lang == 'en' else "下调分层", "lower SOFA-2" if lang == 'en' else "SOFA-2更低", "#0f766e", "↓"),
        (metrics['median_delta'], "Median delta" if lang == 'en' else "Delta中位数", "SOFA-2 - SOFA-1" if lang == 'en' else "SOFA-2 - SOFA-1", "#475569", "Δ"),
    ]
    for col, (value, label, hint, color, icon) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid #cddbeb;border-left:4px solid {color};
                            border-radius:16px;padding:11px 13px;min-height:92px;box-shadow:0 8px 24px rgba(15,31,68,.045)">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
                        <span style="width:24px;height:24px;border-radius:7px;background:{color};color:white;display:inline-flex;align-items:center;justify-content:center;font-size:.82rem;font-weight:900">{icon}</span>
                        <span style="font-size:.68rem;font-weight:850;color:#60718a;letter-spacing:.07em;text-transform:uppercase">{label}</span>
                    </div>
                    <div style="font-size:1.5rem;font-weight:900;line-height:1.05;color:{color};letter-spacing:-.02em">{value}</div>
                    <div style="font-size:.68rem;color:#60718a;margin-top:4px;font-weight:700">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_reclassification_snapshot(reclass: Dict[str, Any], lang: str = 'en', key_prefix: str = 'reclass'):
    """Render a compact dashboard-friendly SOFA reclassification snapshot."""
    import plotly.express as px

    summary = reclass.get('summary', pd.DataFrame())
    if summary.empty:
        st.warning("No SOFA reclassification data available" if lang == 'en' else "没有可用的SOFA重新分层数据")
        return
    unit_pct_label = "Paired points (%)" if reclass.get('metrics', {}).get('denominator_label') == "Paired points" else (
        "配对时间点占比 (%)" if reclass.get('metrics', {}).get('denominator_label') == "配对时间点" else ("Patients (%)" if lang == 'en' else "患者占比 (%)")
    )

    fig = px.bar(
        summary.sort_values('pct', ascending=True),
        x='pct',
        y='group',
        orientation='h',
        text=summary.sort_values('pct', ascending=True)['pct'].map(lambda x: f"{x:.1f}%"),
        color='mortality',
        color_continuous_scale=['#dbeafe', '#ef4444'],
        range_color=[0, 100],
        labels={
            'pct': unit_pct_label,
            'group': "",
            'mortality': "Mortality %" if lang == 'en' else "死亡率 %",
        },
        template='plotly_white',
    )
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_layout(
        height=315,
        margin=dict(l=10, r=45, t=12, b=35),
        coloraxis_colorbar=dict(title="Mortality %" if lang == 'en' else "死亡率 %"),
        font=dict(size=13, color='#111827'),
    )
    fig.update_xaxes(range=[0, max(10, float(summary['pct'].max()) * 1.22)], gridcolor='#e5e7eb')
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_snapshot", config=_get_plotly_chart_config())
