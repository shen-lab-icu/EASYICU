"""Export manifest and PDF report helpers for the Streamlit app."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_APP_CONTEXT: dict[str, Any] = {}


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Store app globals needed by transitional extraction helpers."""
    if app_context is _APP_CONTEXT:
        return
    context_copy = dict(app_context)
    _APP_CONTEXT.clear()
    _APP_CONTEXT.update(context_copy)


def _ctx(name: str) -> Any:
    try:
        return _APP_CONTEXT[name]
    except KeyError as exc:
        raise RuntimeError(f"EasyICU export report helper missing app context: {name}") from exc


def _generate_cohort_prefix(app_context: dict[str, Any] | None = None) -> str:
    """根据队列筛选条件生成文件名前缀。

    Returns:
        筛选条件前缀字符串，如 "age18-80_firstICU_los24h"，无筛选则返回空字符串
    """
    if app_context is not None:
        _install_app_context(app_context)

    if not st.session_state.get('cohort_enabled', False):
        return ""

    cf = st.session_state.get('cohort_filter', {})
    parts = []

    # 年龄
    age_min = cf.get('age_min')
    age_max = cf.get('age_max')
    if age_min is not None or age_max is not None:
        age_str = f"age{int(age_min) if age_min else 0}-{int(age_max) if age_max else 'inf'}"
        parts.append(age_str)

    # 首次入ICU
    first_icu = cf.get('first_icu_stay')
    if first_icu is True:
        parts.append("firstICU")
    elif first_icu is False:
        parts.append("readmit")

    # 住院时长
    los_min = cf.get('los_min')
    if los_min is not None and los_min > 0:
        parts.append(f"los{int(los_min)}h")

    # 性别
    gender = cf.get('gender')
    if gender is not None:
        parts.append(f"sex{gender}")

    # 存活状态
    survived = cf.get('survived')
    if survived is True:
        parts.append("survived")
    elif survived is False:
        parts.append("deceased")

    # Sepsis
    has_sepsis = cf.get('has_sepsis')
    if has_sepsis is True:
        parts.append("sepsis")
    elif has_sepsis is False:
        parts.append("noSepsis")

    disease_cohort = cf.get('disease_cohort')
    if disease_cohort and disease_cohort != 'none':
        parts.append(disease_cohort)

    icd_include_query = str(cf.get('icd_include_query', cf.get('icd_query', ''))).strip()
    if icd_include_query:
        token = _ctx("_split_query_tokens")(icd_include_query)
        if token:
            parts.append(f"icdIn{token[0][:10]}")
    icd_exclude_query = str(cf.get('icd_exclude_query', '')).strip()
    if icd_exclude_query:
        token = _ctx("_split_query_tokens")(icd_exclude_query)
        if token:
            parts.append(f"icdEx{token[0][:10]}")

    return "_".join(parts)

def _write_export_manifest(
    export_dir: Path,
    *,
    exported_files: list[str],
    patient_count: int,
    concept_count: int,
    export_format: str,
    unavailable_concepts: list[str] | None = None,
    unsupported_concepts: list[str] | None = None,
    empty_data_concepts: list[str] | None = None,
    failed_concepts: list[str] | None = None,
    note: str | None = None,
    app_context: dict[str, Any] | None = None,
) -> list[str]:
    """Write a lightweight export manifest for reproducibility."""
    if app_context is not None:
        _install_app_context(app_context)

    cohort_filter = st.session_state.get('cohort_filter', {}) if st.session_state.get('cohort_enabled') else {}
    cohort_filter = {
        key: value
        for key, value in cohort_filter.items()
        if value not in (None, '', 'none', False)
    }

    manifest = {
        'easyicu_version': '1.0.0',
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'database': st.session_state.get('database', 'unknown'),
        'entry_mode': st.session_state.get('entry_mode', 'unknown'),
        'export_dir': str(export_dir),
        'export_format': export_format,
        'merge_mode': st.session_state.get('export_merge_mode', 'separate'),
        'filter_by_patient': bool(st.session_state.get('export_filter_patient_enabled', False)),
        'patient_limit_requested': int(st.session_state.get('patient_limit', 0) or 0),
        'patient_limit_effective': int(
            st.session_state.get(
                '_export_effective_patient_limit',
                st.session_state.get('patient_limit', 0) if st.session_state.get('export_filter_patient_enabled') else 0,
            ) or 0
        ),
        'include_row_index': bool(st.session_state.get('export_include_index', False)),
        'add_timestamp_to_filename': bool(st.session_state.get('export_add_timestamp', False)),
        'patient_count': int(patient_count or 0),
        'concept_count': int(concept_count or 0),
        'selected_concepts': list(st.session_state.get('selected_concepts', [])),
        'selected_groups': list(st.session_state.get('selected_groups', [])),
        'cohort_enabled': bool(st.session_state.get('cohort_enabled', False)),
        'cohort_filter': cohort_filter,
        'cohort_suffix': _generate_cohort_prefix(_APP_CONTEXT),
        'sepsis_runtime_options': _ctx('_get_sepsis_runtime_options')(),
        'exported_files': [Path(path).name for path in exported_files if path],
        'unavailable_concepts': unavailable_concepts or [],
        'unsupported_concepts': unsupported_concepts or [],
        'empty_data_concepts': empty_data_concepts or [],
        'failed_concepts': failed_concepts or [],
        'note': note or '',
    }

    json_path = export_dir / 'easyicu_export_manifest.json'
    txt_path = export_dir / 'easyicu_export_manifest.txt'

    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2, default=str)

    lines = [
        "EasyICU Export Manifest",
        f"Exported at: {manifest['exported_at']}",
        f"Database: {manifest['database']}",
        f"Entry mode: {manifest['entry_mode']}",
        f"Export directory: {manifest['export_dir']}",
        f"Export format: {manifest['export_format']}",
        f"Merge mode: {manifest['merge_mode']}",
        f"Filter by patient: {manifest['filter_by_patient']}",
        f"Patient limit requested: {manifest['patient_limit_requested']}",
        f"Patient limit effective: {manifest['patient_limit_effective']}",
        f"Include row index: {manifest['include_row_index']}",
        f"Add timestamp to filename: {manifest['add_timestamp_to_filename']}",
        f"Patients: {manifest['patient_count']}",
        f"Concepts: {manifest['concept_count']}",
    ]
    if manifest['cohort_suffix']:
        lines.append(f"Cohort suffix: {manifest['cohort_suffix']}")
    if manifest['cohort_filter']:
        lines.append("Cohort filter:")
        for key, value in manifest['cohort_filter'].items():
            lines.append(f"  - {key}: {value}")
    if manifest['selected_groups']:
        lines.append(f"Selected groups: {', '.join(manifest['selected_groups'])}")
    if manifest['selected_concepts']:
        lines.append("Selected concepts:")
        lines.extend([f"  - {concept}" for concept in manifest['selected_concepts']])
    if manifest['exported_files']:
        lines.append("Exported files:")
        lines.extend([f"  - {name}" for name in manifest['exported_files']])
    if note:
        lines.append(f"Note: {note}")

    with open(txt_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(lines) + "\n")

    return [str(json_path), str(txt_path)]

def _build_quick_viz_pdf_report(*, lang: str, preview_data: dict[str, pd.DataFrame], concepts_to_export: list[str], app_context: dict[str, Any] | None = None) -> bytes:
    """Create a compact one-file PDF summary for Quick Visualization."""
    if app_context is not None:
        _install_app_context(app_context)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    id_col = st.session_state.get('id_col', 'stay_id')
    database = st.session_state.get('database', 'unknown')
    export_dir = st.session_state.get('viz_confirmed_path') or st.session_state.get('last_export_dir') or "-"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_rows = 0
    patient_ids = set()
    summary_rows: list[dict[str, object]] = []
    for concept in concepts_to_export:
        df = preview_data.get(concept)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        rows = len(df)
        total_rows += rows

        if id_col in df.columns:
            concept_patients = set(df[id_col].dropna().astype(str))
            patient_ids |= concept_patients
            patient_count = len(concept_patients)
        else:
            patient_count = 0

        value_col = concept if concept in df.columns else None
        if value_col is None:
            candidate_cols = [
                col for col in df.columns
                if col not in {id_col, 'time', 'charttime', 'starttime', 'endtime', '_concept'}
            ]
            value_col = candidate_cols[0] if candidate_cols else None

        missing_pct = 0.0
        if value_col and value_col in df.columns:
            valid = pd.to_numeric(df[value_col], errors='coerce') if df[value_col].dtype == 'object' else df[value_col]
            missing_pct = float(valid.isna().mean() * 100) if len(valid) else 0.0

        summary_rows.append({
            'concept': concept,
            'rows': rows,
            'patients': patient_count,
            'missing_pct': missing_pct,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('rows', ascending=False).head(10)
    coverage_df = pd.DataFrame(summary_rows).sort_values('patients', ascending=False).head(10)

    total_patients = len(patient_ids)
    concept_count = len(summary_rows)

    title_text = "EasyICU Quick Visualization Report" if lang == 'en' else "EasyICU 快速可视化报告"
    subtitle_text = (
        f"Database: {database.upper()}   •   Concepts: {concept_count}   •   Patients: {total_patients}   •   Records: {total_rows:,}"
        if lang == 'en' else
        f"数据库：{database.upper()}   •   特征：{concept_count}   •   患者：{total_patients}   •   记录：{total_rows:,}"
    )
    meta_lines = [
        f"Generated at: {generated_at}" if lang == 'en' else f"生成时间：{generated_at}",
        f"Export directory: {export_dir}" if lang == 'en' else f"导出目录：{export_dir}",
        (
            f"Selected concepts: {', '.join(concepts_to_export[:10])}" + (" ..." if len(concepts_to_export) > 10 else "")
            if lang == 'en' else
            f"所选特征：{', '.join(concepts_to_export[:10])}" + (" ..." if len(concepts_to_export) > 10 else "")
        ),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "bar"}], [{"type": "bar"}, {"type": "table"}]],
        column_widths=[0.38, 0.62],
        row_heights=[0.42, 0.58],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=(
            "Overview" if lang == 'en' else "概览",
            "Top Concepts by Records" if lang == 'en' else "记录数最高的特征",
            "Top Concepts by Patient Coverage" if lang == 'en' else "患者覆盖最高的特征",
            "Detail Table" if lang == 'en' else "明细表",
        ),
    )

    overview_labels = [
        "Patients" if lang == 'en' else "患者数",
        "Concepts" if lang == 'en' else "特征数",
        "Records" if lang == 'en' else "记录数",
    ]
    overview_values = [total_patients, concept_count, total_rows]
    fig.add_trace(
        go.Pie(
            labels=overview_labels,
            values=[max(v, 1) for v in overview_values],
            hole=0.62,
            marker=dict(colors=["#2563eb", "#0ea5e9", "#14b8a6"]),
            textinfo="label+value",
            sort=False,
        ),
        row=1,
        col=1,
    )

    if not summary_df.empty:
        fig.add_trace(
            go.Bar(
                x=summary_df['rows'],
                y=summary_df['concept'],
                orientation='h',
                marker_color="#2563eb",
                hovertemplate="%{y}: %{x:,}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    if not coverage_df.empty:
        fig.add_trace(
            go.Bar(
                x=coverage_df['patients'],
                y=coverage_df['concept'],
                orientation='h',
                marker_color="#0f766e",
                hovertemplate="%{y}: %{x:,}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    table_df = pd.DataFrame(summary_rows).sort_values(['patients', 'rows'], ascending=[False, False]).head(8)
    if table_df.empty:
        table_df = pd.DataFrame([{
            'concept': '-',
            'rows': 0,
            'patients': 0,
            'missing_pct': 0.0,
        }])
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Concept" if lang == 'en' else "特征",
                    "Rows" if lang == 'en' else "记录",
                    "Patients" if lang == 'en' else "患者",
                    "Missing %" if lang == 'en' else "缺失率",
                ],
                fill_color="#e0f2fe",
                align="left",
                font=dict(size=13, color="#0f172a"),
            ),
            cells=dict(
                values=[
                    table_df['concept'],
                    table_df['rows'].map(lambda x: f"{int(x):,}"),
                    table_df['patients'].map(lambda x: f"{int(x):,}"),
                    table_df['missing_pct'].map(lambda x: f"{x:.1f}%"),
                ],
                fill_color="#ffffff",
                align="left",
                font=dict(size=12, color="#0f172a"),
                height=28,
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_layout(
        width=1440,
        height=1020,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text=f"{title_text}<br><sup>{subtitle_text}</sup>",
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            font=dict(size=24, color="#0f172a"),
        ),
        margin=dict(l=40, r=40, t=120, b=50),
        font=dict(family="Arial, sans-serif", color="#0f172a", size=13),
    )

    meta_text = "<br>".join(meta_lines)
    fig.add_annotation(
        x=0.0,
        y=1.08,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text=meta_text,
        font=dict(size=12, color="#475569"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=8,
    )

    return fig.to_image(format="pdf")
