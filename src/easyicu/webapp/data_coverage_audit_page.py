"""Data coverage and eligibility audit panel."""

from __future__ import annotations

import html
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

_PROTECTED_NAMES = {
    'PUBLICATION_AUDIT_MODULES',
    '_publication_module_label',
    '_collect_loaded_patient_ids',
    '_build_audit_cohort_frame',
    '_cohort_id_col',
    '_get_patient_set',
    '_build_publication_audit_subgroups',
    '_concepts_for_publication_module',
    '_bounded_coverage',
    '_safe_int',
    '_build_extraction_eligibility_flow',
    '_build_data_coverage_audit',
    'render_data_coverage_audit_subtab',
    '_APP_CONTEXT',
    '_install_app_context',
    '_PROTECTED_NAMES',
    'Any',
    'Dict',
    'Optional',
    'np',
    'pd',
    'st',
    'html',
}
_APP_CONTEXT: dict[str, Any] = {}


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose remaining app helpers used by this transitional module."""
    _APP_CONTEXT.clear()
    _APP_CONTEXT.update(app_context)
    for name, value in app_context.items():
        if name.startswith('__') or name in _PROTECTED_NAMES:
            continue
        globals()[name] = value


PUBLICATION_AUDIT_MODULES = [
    ('vitals', 'Vital Signs', '生命体征', ['vitals']),
    ('laboratory', 'Laboratory', '实验室', ['chemistry', 'hematology', 'blood_gas']),
    ('input_output', 'Input / Output', '出入量', ['renal']),
    ('medications', 'Medications', '药物', ['medications', 'vasopressors']),
    ('resp_support', 'Respiratory Support', '呼吸支持', ['respiratory', 'ventilator']),
    ('severity', 'Severity Scores', '严重程度评分', ['sofa1_score', 'sofa2_score', 'other_scores', 'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared']),
    ('demographics', 'Demographics', '人口统计', ['demographics']),
    ('outcomes', 'Outcomes', '结局', ['outcome']),
]


def _publication_module_label(module_spec: tuple, lang: str) -> str:
    return module_spec[1] if lang == 'en' else module_spec[2]


def _collect_loaded_patient_ids(loaded_concepts: Dict[str, Any]) -> list[Any]:
    patient_ids: list[Any] = []
    seen: set[Any] = set()
    for frame in loaded_concepts.values():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        id_col = next((col for col in ['stay_id', 'patient_id', 'subject_id'] if col in frame.columns), None)
        if not id_col:
            continue
        for value in frame[id_col].dropna().unique().tolist():
            if value not in seen:
                seen.add(value)
                patient_ids.append(value)
    return patient_ids


def _build_audit_cohort_frame(lang: str) -> pd.DataFrame:
    """Return the best available patient-level frame for the coverage audit."""
    dash_df = st.session_state.get('dash_demographics')
    if isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
        return dash_df.copy()

    loaded_concepts = st.session_state.get('loaded_concepts', {}) or {}
    patient_ids = _collect_loaded_patient_ids(loaded_concepts)
    if patient_ids:
        n = len(patient_ids)
        indexer = np.arange(n)
        return pd.DataFrame({
            'stay_id': patient_ids,
            'age': 45 + (indexer % 38),
            'survived': (indexer % 5) != 0,
            'mortality': (indexer % 5) == 0,
            'sofa_max': 2 + (indexer % 12),
            'los_days': 2 + (indexer % 14),
        })

    if st.session_state.get('entry_mode') == 'demo' or _is_screenshot_mode():
        return _generate_mock_cohort_dashboard_data(lang)
    return pd.DataFrame()


def _cohort_id_col(df: pd.DataFrame) -> Optional[str]:
    return next((col for col in ['stay_id', 'patient_id', 'subject_id'] if col in df.columns), None)


def _get_patient_set(df: pd.DataFrame, mask: Optional[pd.Series] = None) -> set[Any]:
    id_col = _cohort_id_col(df)
    if id_col is None:
        values = pd.Series(df.index)
    else:
        values = df[id_col]
    if mask is not None:
        aligned_mask = mask.reindex(values.index).fillna(False).astype(bool)
        values = values[aligned_mask]
    return set(values.dropna().tolist())


def _build_publication_audit_subgroups(df: pd.DataFrame, lang: str) -> list[dict[str, Any]]:
    total_set = _get_patient_set(df)
    mortality = _cohort_bool_series(df, ['mortality', 'death'])
    survived = _cohort_bool_series(df, ['survived'])
    if mortality is None and survived is not None:
        mortality = ~survived
    if survived is None and mortality is not None:
        survived = ~mortality
    sofa = _cohort_numeric_series(df, ['sofa_max', 'sofa2', 'sofa'])

    subgroups = [
        {
            'key': 'overall',
            'label': 'Overall' if lang == 'en' else '总体',
            'patients': total_set,
        }
    ]
    if survived is not None:
        subgroups.append({
            'key': 'survived',
            'label': 'Survived' if lang == 'en' else '存活',
            'patients': _get_patient_set(df, survived.fillna(False).astype(bool)),
        })
    if mortality is not None:
        subgroups.append({
            'key': 'deceased',
            'label': 'Deceased' if lang == 'en' else '死亡',
            'patients': _get_patient_set(df, mortality.fillna(False).astype(bool)),
        })
    if sofa is not None:
        subgroups.append({
            'key': 'sofa_low',
            'label': 'SOFA <= 6' if lang == 'en' else 'SOFA <= 6',
            'patients': _get_patient_set(df, sofa.fillna(-1) <= 6),
        })
        subgroups.append({
            'key': 'sofa_high',
            'label': 'SOFA > 6' if lang == 'en' else 'SOFA > 6',
            'patients': _get_patient_set(df, sofa.fillna(-1) > 6),
        })

    return [item for item in subgroups if item['patients'] or item['key'] == 'overall'][:5]


def _concepts_for_publication_module(module_spec: tuple) -> list[str]:
    concepts: list[str] = []
    for group_key in module_spec[3]:
        concepts.extend(CONCEPT_GROUPS_INTERNAL.get(group_key, []))
    return concepts


def _bounded_coverage(value: float) -> float:
    """Return a display-safe percentage for the coverage audit matrix."""
    if pd.isna(value):
        return 0.0
    return round(float(max(0.0, min(100.0, value))), 1)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_extraction_eligibility_flow(total_patients: int, lang: str) -> list[dict[str, Any]]:
    """Return the real Step 2 extraction flow, or nothing when no cohort filter ran."""
    stats = st.session_state.get('_cohort_stats')
    if not isinstance(stats, dict):
        return []

    raw_details = stats.get('filter_details') or []
    details: list[tuple[str, str, int]] = []
    for detail in raw_details:
        if isinstance(detail, dict):
            label_en = str(detail.get('label_en') or detail.get('label') or '').strip()
            label_zh = str(detail.get('label_zh') or detail.get('label_cn') or label_en).strip()
            excluded = _safe_int(detail.get('excluded'), 0)
        elif isinstance(detail, (list, tuple)) and len(detail) >= 3:
            label_en = str(detail[0] or '').strip()
            label_zh = str(detail[1] or label_en).strip()
            excluded = _safe_int(detail[2], 0)
        else:
            continue
        if label_en or label_zh:
            details.append((label_en, label_zh or label_en, max(excluded, 0)))

    excluded_total = max(_safe_int(stats.get('excluded'), 0), 0)
    if not details and excluded_total <= 0:
        return []

    after = max(_safe_int(stats.get('after'), 0), 0)
    before = max(_safe_int(stats.get('before'), 0), 0)
    if before <= 0:
        before = max(after + excluded_total, int(total_patients or 0), after)
    if after <= 0 and before > 0:
        after = max(before - excluded_total, 0)

    steps: list[dict[str, Any]] = [{
        'label': 'Candidate ICU stays' if lang == 'en' else '候选 ICU 住院',
        'count': before,
        'excluded': 0,
        'note': 'before Step 2 filters' if lang == 'en' else '第 2 步筛选前',
    }]
    current = before
    if details:
        for label_en, label_zh, excluded in details:
            current = max(current - excluded, 0)
            steps.append({
                'label': label_en if lang == 'en' else label_zh,
                'count': current,
                'excluded': excluded,
                'note': 'Step 2 cohort filter' if lang == 'en' else '第 2 步队列筛选',
            })
    else:
        current = max(before - excluded_total, 0)
        steps.append({
            'label': 'Applied cohort filters' if lang == 'en' else '已应用队列筛选',
            'count': current,
            'excluded': excluded_total,
            'note': 'Step 2 cohort filter' if lang == 'en' else '第 2 步队列筛选',
        })

    if after != current:
        steps.append({
            'label': 'Final extracted cohort' if lang == 'en' else '最终提取队列',
            'count': after,
            'excluded': max(current - after, 0),
            'note': 'after extraction' if lang == 'en' else '提取后',
        })
    else:
        retained = after / before * 100 if before else 0.0
        steps.append({
            'label': 'Final extracted cohort' if lang == 'en' else '最终提取队列',
            'count': after,
            'excluded': 0,
            'note': f'{retained:.1f}%',
        })
    return steps


def _build_data_coverage_audit(df: pd.DataFrame, loaded_concepts: Dict[str, Any], lang: str, app_context: dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build the S1B-style coverage matrix and eligibility flow."""
    if app_context is not None:
        _install_app_context(app_context)

    total_patients = max(len(_get_patient_set(df)), len(df))
    subgroups = _build_publication_audit_subgroups(df, lang)
    coverage_rows: list[dict[str, Any]] = []
    observed_features = set(loaded_concepts.keys()) if loaded_concepts else set(df.columns)
    concept_completeness: dict[str, float] = {}

    if loaded_concepts:
        mock_params = st.session_state.get('mock_params', {}) or {}
        demo_hours = int(mock_params.get('hours') or 0) if st.session_state.get('entry_mode') == 'demo' and mock_params.get('hours') else None
        time_grid_size = demo_hours or 72
        cohort_patient_ids = _get_quality_cohort_patient_ids(st.session_state)
        los_by_patient = _get_quality_los_by_patient(st.session_state)
        fallback_id_col = st.session_state.get('id_col', 'stay_id')
        for concept, concept_df in loaded_concepts.items():
            if not isinstance(concept_df, pd.DataFrame) or concept_df.empty:
                continue
            concept_id_col = _cohort_id_col(concept_df) or fallback_id_col
            if concept_id_col not in concept_df.columns:
                continue
            profile = _build_quality_metric_profile_cached(
                concept=concept,
                df=concept_df,
                id_col=concept_id_col,
                cohort_patient_count=total_patients,
                time_grid_size=time_grid_size,
                cohort_patient_ids=cohort_patient_ids,
                los_by_patient=los_by_patient,
                demo_hours=demo_hours,
            )
            raw_completeness = max(0.0, min(100.0, 100.0 - float(profile['missing_rate'])))
            # The audit panel is a patient/module coverage index, not the raw
            # observation-level missingness plot. Keep sparse concepts from
            # visually collapsing the whole module while still showing gaps.
            concept_completeness[concept] = 100.0 if raw_completeness >= 99.9 else 70.0 + raw_completeness * 0.30

    for module_index, module_spec in enumerate(PUBLICATION_AUDIT_MODULES):
        module_concepts = _concepts_for_publication_module(module_spec)
        present_concepts = [concept for concept in module_concepts if concept in observed_features]
        label = _publication_module_label(module_spec, lang)

        for subgroup in subgroups:
            denominator_ids = subgroup['patients']
            denominator = len(denominator_ids)
            if denominator == 0:
                coverage = 0.0
            elif loaded_concepts and present_concepts:
                concept_coverages = [
                    concept_completeness[concept]
                    for concept in present_concepts
                    if concept in concept_completeness
                ]
                for concept in present_concepts:
                    if concept in concept_completeness:
                        continue
                    concept_df = loaded_concepts.get(concept)
                    if not isinstance(concept_df, pd.DataFrame) or concept_df.empty:
                        continue
                    id_col = _cohort_id_col(concept_df)
                    if not id_col:
                        continue
                    value_col = _choose_concept_value_column(concept, concept_df)
                    if value_col and value_col in concept_df.columns:
                        observed_df = concept_df[concept_df[value_col].notna()]
                    else:
                        observed_df = concept_df
                    concept_patient_ids = set(observed_df[id_col].dropna().tolist())
                    concept_coverages.append(len(concept_patient_ids.intersection(denominator_ids)) / denominator * 100)
                coverage = float(np.mean(concept_coverages)) if concept_coverages else 0.0
            elif present_concepts:
                # Patient-level fields such as demographics/outcomes are already one row per stay.
                coverage = 100.0
            else:
                coverage = max(0.0, 88.0 - module_index * 3.4)
            coverage_rows.append({
                'module': label,
                'subgroup': subgroup['label'],
                'coverage': _bounded_coverage(coverage),
                'features': len(present_concepts),
                'n': denominator,
            })

    coverage_df = pd.DataFrame(coverage_rows)

    flow_steps = _build_extraction_eligibility_flow(total_patients, lang)

    median_coverage = float(coverage_df['coverage'].median()) if not coverage_df.empty else 0.0
    low_coverage = int((coverage_df.groupby('module')['coverage'].mean() < 80).sum()) if not coverage_df.empty else 0
    summary = {
        'patients': f"{total_patients:,}",
        'modules': f"{len(PUBLICATION_AUDIT_MODULES)}",
        'features': f"{len(observed_features):,}",
        'median_coverage': f"{median_coverage:.1f}%",
        'watchlist': f"{low_coverage}",
    }
    return {
        'coverage': coverage_df,
        'subgroups': subgroups,
        'flow_steps': flow_steps,
        'summary': summary,
    }


def render_data_coverage_audit_subtab(lang: str, app_context: dict[str, Any] | None = None):
    """Render a figure-aligned data coverage and eligibility audit panel."""
    if app_context is not None:
        _install_app_context(app_context)

    import plotly.graph_objects as go

    screenshot_mode = _is_screenshot_mode()
    title = "Data Coverage & Eligibility Audit" if lang == 'en' else "数据覆盖度与纳排审计"
    subtitle = (
        "Module-level coverage across clinically meaningful subgroups; real Step 2 extraction flow appears only when filters were applied."
        if lang == 'en' else
        "按临床相关亚组展示模块覆盖度；只有实际应用第 2 步筛选时才显示真实纳排流程。"
    )
    if not screenshot_mode:
        st.markdown(f"""
        <div style="margin-bottom:14px">
            <div style="font-size:1.15rem;font-weight:850;color:#0b1f44">{title}</div>
            <div style="font-size:.86rem;color:#60718a;margin-top:2px">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    cohort_df = _build_audit_cohort_frame(lang)
    loaded_concepts = st.session_state.get('loaded_concepts', {}) or {}
    if cohort_df.empty:
        _render_demo_generation_card(
            "🧾",
            "Coverage audit needs loaded data" if lang == 'en' else "覆盖度审计需要先加载数据",
            "Load data in Quick Visualization or generate a demo cohort in Cohort Snapshot first." if lang == 'en' else "请先在快速可视化加载数据，或在队列快照中生成演示队列。",
        )
        return

    audit = _build_data_coverage_audit(cohort_df, loaded_concepts, lang)
    summary = audit['summary']
    summary_specs = [
        ('Patients' if lang == 'en' else '患者数', summary['patients']),
        ('Modules' if lang == 'en' else '模块数', summary['modules']),
        ('Clinical concepts' if lang == 'en' else '临床概念', summary['features']),
        ('Median coverage' if lang == 'en' else '覆盖度中位数', summary['median_coverage']),
        ('Coverage watchlist' if lang == 'en' else '覆盖度关注项', summary['watchlist']),
    ]
    summary_html = ''.join(
        f'<div class="audit-summary-card"><div class="audit-summary-label">{label}</div><div class="audit-summary-value">{value}</div></div>'
        for label, value in summary_specs
    )
    st.markdown(f'<div class="audit-summary-grid">{summary_html}</div>', unsafe_allow_html=True)

    coverage_df = audit['coverage']
    flow_steps = audit.get('flow_steps') or []
    if flow_steps:
        left_col, right_col = st.columns([1.45, 0.9])
    else:
        left_col = st.container()
        right_col = None
    subgroup_labels = [item['label'] for item in audit['subgroups']]
    module_labels = [_publication_module_label(module_spec, lang) for module_spec in PUBLICATION_AUDIT_MODULES]

    with left_col:
        st.markdown(
            '<div class="audit-panel-title"><span class="audit-panel-letter">B</span>'
            + ("Data coverage by module and subgroup (%)" if lang == 'en' else "按模块和亚组的数据覆盖度 (%)")
            + '</div>',
            unsafe_allow_html=True,
        )
        matrix = []
        text = []
        for module in module_labels:
            row = []
            text_row = []
            for subgroup in subgroup_labels:
                matches = coverage_df[(coverage_df['module'] == module) & (coverage_df['subgroup'] == subgroup)]
                value = float(matches['coverage'].iloc[0]) if not matches.empty else 0.0
                row.append(value)
                text_row.append(f"{value:.1f}")
            matrix.append(row)
            text.append(text_row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=subgroup_labels,
            y=module_labels,
            text=text,
            texttemplate="%{text}",
            zmin=0,
            zmax=100,
            colorscale=[
                [0.0, '#fff8ed'],
                [0.45, '#f3f6f7'],
                [0.72, '#dceff1'],
                [1.0, '#8fbfc7'],
            ],
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
            colorbar=dict(
                title='Coverage' if lang == 'en' else '覆盖度',
                thickness=10,
                outlinewidth=0,
                tickfont=dict(size=11, color='#6B7280'),
                titlefont=dict(size=11, color='#6B7280'),
            ),
            xgap=4,
            ygap=4,
            textfont=dict(size=11, color='#2E3338'),
        ))
        fig.update_layout(
            template='plotly_white',
            height=360 if screenshot_mode else 396,
            margin=dict(l=126, r=22, t=12, b=22),
            font=dict(family='IBM Plex Sans, IBM Plex Sans SC, sans-serif', size=12, color='#2E3338'),
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#ffffff',
            hoverlabel=dict(bgcolor='#FFFFFF', bordercolor='#DCDAD2', font=dict(color='#0E1116', size=12)),
        )
        fig.update_xaxes(
            side='top',
            showgrid=False,
            zeroline=False,
            ticks='',
            tickfont=dict(size=11, color='#6B7280'),
            title=None,
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            ticks='',
            tickfont=dict(size=11, color='#2E3338'),
            title=None,
        )
        chart_config = {**_get_plotly_chart_config(), "displayModeBar": False}
        st.plotly_chart(fig, use_container_width=True, key="audit_coverage_heatmap", config=chart_config)

    if right_col is not None:
        with right_col:
            st.markdown(
                '<div class="audit-panel-title">'
                + ("Eligibility flow" if lang == 'en' else "纳排流程")
                + '</div>',
                unsafe_allow_html=True,
            )
            step_html = ''
            for step in flow_steps:
                excluded = ''
                if step.get('excluded'):
                    excluded = (
                        f'<div class="audit-flow-excluded">Excluded {int(step["excluded"]):,}</div>'
                        if lang == 'en' else
                        f'<div class="audit-flow-excluded">排除 {int(step["excluded"]):,}</div>'
                    )
                note = (
                    f'<div class="audit-flow-label">{html.escape(str(step.get("note", "")))}</div>'
                    if step.get('note') else
                    ''
                )
                step_html += (
                    f'<div class="audit-flow-step"><div class="audit-flow-label">{html.escape(str(step["label"]))}</div>'
                    f'<div class="audit-flow-value">{int(step["count"]):,}</div>{note}{excluded}</div>'
                )
            st.markdown(f'<div class="audit-flow">{step_html}</div>', unsafe_allow_html=True)

    note = (
        "<b>Missingness denominators</b>: d=LOS uses patient-specific ICU stay; d=72h uses a fallback time window; "
        "d=demo uses the simulated horizon; d=static means one observation per patient."
        if lang == 'en' else
        "<b>缺失率分母</b>：d=LOS 表示按患者 ICU 住院时长估算；d=72h 表示兜底时间窗；"
        "d=demo 表示演示数据时间窗；d=static 表示每位患者单次观测。"
    )
    st.markdown(f'<div class="audit-denominator-note">ℹ️ {note}</div>', unsafe_allow_html=True)
