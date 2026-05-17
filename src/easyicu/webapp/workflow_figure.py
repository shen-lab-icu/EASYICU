"""Live extraction workflow figure rendering."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st

from easyicu.webapp.components.constants import get_all_concepts


def _html_escape_text(value: Any, default: str = "—") -> str:
    """Escape short UI values for HTML snippets."""
    if value is None:
        return html.escape(default)
    text = str(value).strip()
    return html.escape(text if text else default)

def _workflow_field(label: str, value: Any, suffix: str = "⌄") -> str:
    suffix_html = f"<span>{html.escape(suffix)}</span>" if suffix else ""
    return (
        '<div class="workflow-field">'
        f'<div class="workflow-label">{html.escape(label)}</div>'
        f'<div class="workflow-input"><span>{_html_escape_text(value)}</span>{suffix_html}</div>'
        '</div>'
    )

def _workflow_status(done: bool, done_text: str, todo_text: str) -> str:
    if done:
        return (
            '<div class="workflow-status">'
            '<span class="workflow-check-dot">✓</span>'
            f'<span>{html.escape(done_text)}</span>'
            '</div>'
        )
    return (
        '<div class="workflow-status warn">'
        '<span class="workflow-check-dot" style="background:#f59e0b">!</span>'
        f'<span>{html.escape(todo_text)}</span>'
        '</div>'
    )

def _render_extraction_pipeline_figure(
    *,
    lang: str,
    step1_done: bool,
    step2_done: bool,
    step3_done: bool,
    step4_done: bool,
) -> None:
    """Render the live extraction workflow using the same visual logic as Figure 2."""
    is_en = lang == 'en'
    db_display_names = {
        'mock': 'Demo ICU',
        'miiv': 'MIMIC-IV',
        'eicu': 'eICU-CRD',
        'aumc': 'AmsterdamUMCdb',
        'hirid': 'HiRID',
        'mimic': 'MIMIC-III',
        'sic': 'SICdb',
    }
    database = st.session_state.get('database', 'mock' if st.session_state.get('use_mock_data') else 'miiv')
    db_label = db_display_names.get(database, str(database).upper())
    data_path = (
        "Auto-generated demo data"
        if st.session_state.get('use_mock_data', False)
        else st.session_state.get('data_path', '')
    )
    cohort_filter = st.session_state.get('cohort_filter', {}) or {}
    age_min = cohort_filter.get('age_min') or 18
    age_max = cohort_filter.get('age_max') or 120
    los_min = cohort_filter.get('los_min') or 24
    gender = cohort_filter.get('gender') or ("Any" if is_en else "不限")
    survived = cohort_filter.get('survived')
    survival_text = (
        "Any" if survived is None else ("Survived" if survived else "Deceased")
    ) if is_en else (
        "不限" if survived is None else ("存活" if survived else "死亡")
    )
    cohort_name = cohort_filter.get('disease_cohort') or 'none'
    cohort_display = {
        'none': 'No disease filter' if is_en else '不限制疾病队列',
        'sepsis': 'Sepsis-3 cohort',
        'aki': 'AKI cohort (KDIGO)',
        'circ_failure': 'Circulatory failure',
        'mech_vent': 'Mechanical ventilation',
        'rrt': 'Renal replacement therapy',
        'ards': 'ARDS cohort',
        'pneumonia': 'Pneumonia cohort',
        'heart_failure': 'Heart failure cohort',
        'ami': 'Acute myocardial infarction',
        'stroke': 'Stroke cohort',
    }.get(cohort_name, str(cohort_name))
    include_query = cohort_filter.get('icd_include_query') or ("N17-18" if cohort_name == 'aki' else "—")
    exclude_query = cohort_filter.get('icd_exclude_query') or ("C34" if cohort_name == 'aki' else "—")

    selected_groups = list(st.session_state.get('selected_groups') or [])
    if not selected_groups:
        selected_groups = [
            "Vital Signs",
            "Laboratory",
            "Renal & Urine Output",
            "SOFA Scores",
        ] if is_en else [
            "生命体征",
            "实验室检验",
            "肾脏与尿量",
            "SOFA 评分",
        ]
    group_chips = "".join(
        f'<div class="workflow-input" style="min-height:30px;padding:0.25rem 0.45rem;font-size:0.68rem">{html.escape(group)}</div>'
        for group in selected_groups[:6]
    )

    selected_concepts = list(st.session_state.get('selected_concepts') or [])
    concept_preview = selected_concepts[:12] or ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp", "creatinine", "uo_24h", "aki_stage", "sofa", "sofa2"]
    concepts_html = "".join(
        f'<div class="workflow-concept"><span class="workflow-tick">✓</span>{html.escape(str(concept))}</div>'
        for concept in concept_preview
    )
    more_count = max(0, len(selected_concepts) - len(concept_preview))
    if more_count:
        concepts_html += f'<div class="workflow-input" style="min-height:26px;font-size:0.66rem;justify-content:center">+ {more_count} more</div>'

    export_path = st.session_state.get('export_path') or (
        "/exports/vital_signs_aki/" if is_en else "/exports/vital_signs_aki/"
    )
    export_format = st.session_state.get('export_format') or "Parquet"
    patient_limit = st.session_state.get('patient_limit', 0)
    patient_limit_text = "All patients" if not patient_limit else f"{int(patient_limit):,}"
    export_result = st.session_state.get('_export_success_result') or {}
    export_files = list(export_result.get('files') or [])
    export_files_html = "".join(
        f'<div>▧ {html.escape(Path(str(file_name)).name)}</div>'
        for file_name in export_files[:4]
    )
    if len(export_files) > 4:
        export_files_html += f'<div style="text-align:center;color:#60718a">… ({len(export_files) - 4} more files)</div>'
    if not export_files:
        export_files_html = (
            f'<div style="color:#94a3b8">{"No files exported yet" if is_en else "尚未导出文件"}</div>'
        )

    title = "EasyICU Data Preparation Workflow" if is_en else "EasyICU 数据准备流程"
    subtitle = (
        "Progress overview only — run each step using the sidebar on the left. "
        "Panels A–D mirror your sidebar configuration; panel E reviews the export."
        if is_en else
        "仅为流程进度总览——每个步骤请使用左侧侧边栏操作。"
        "A–D 面板镜像侧边栏配置，E 面板用于复核导出。"
    )
    summary_title = "Export summary" if is_en else "导出摘要"
    summary_status = (
        "Export completed successfully"
        if step4_done else
        ("Ready for export once the sidebar confirmation is clicked" if step3_done else "Complete the active sidebar step to unlock export")
    )
    if not is_en:
        summary_status = "导出已完成" if step4_done else ("确认侧边栏设置后即可导出" if step3_done else "请完成当前侧边栏步骤以解锁导出")
    summary_ready = bool(step4_done or step3_done)
    summary_strip_class = "workflow-success-strip" if summary_ready else "workflow-success-strip warn"
    summary_icon = "✓" if summary_ready else "!"

    import time as _time_module

    def _fmt_duration(seconds: float | None) -> str:
        if not seconds:
            return "—"
        minutes, secs = divmod(int(seconds), 60)
        if is_en:
            return f"{minutes} min {secs} sec" if minutes else f"{secs} sec"
        return f"{minutes} 分 {secs} 秒" if minutes else f"{secs} 秒"

    start_time_text = "—"
    if export_result.get('start_time'):
        start_time_text = _time_module.strftime(
            "%H:%M:%S", _time_module.localtime(export_result['start_time'])
        )
    total_size_text = "—"
    if step4_done and export_files:
        try:
            total_bytes = sum(
                Path(str(f)).stat().st_size
                for f in export_files
                if Path(str(f)).exists()
            )
            if total_bytes:
                total_size_text = f"{total_bytes / (1024 * 1024):.1f} MB"
        except OSError:
            total_size_text = "—"

    stats = [
        ("Start time" if is_en else "开始时间", start_time_text),
        ("Duration" if is_en else "耗时", _fmt_duration(export_result.get('total_time'))),
        ("Files" if is_en else "文件数", str(len(export_files)) if export_files else "—"),
        ("Total size" if is_en else "总大小", total_size_text),
    ]
    stats_html = "".join(
        f'<div class="workflow-mini-stat"><div class="workflow-mini-label">{html.escape(label)}</div><div class="workflow-mini-value">{html.escape(value)}</div></div>'
        for label, value in stats
    )

    st.markdown(
        f'''
        <div class="workflow-figure-shell">
            <div class="workflow-figure-title">{html.escape(title)}</div>
            <div class="workflow-figure-subtitle">{html.escape(subtitle)}</div>
            <div class="workflow-pipeline-grid">
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">A</div>
                        <div><div class="workflow-card-title">{"Data source configuration" if is_en else "数据源配置"}</div><div class="workflow-card-kicker">Step 1</div></div>
                    </div>
                    {_workflow_field("Select database" if is_en else "选择数据库", db_label)}
                    {_workflow_field("Data path" if is_en else "数据路径", data_path, suffix="")}
                    <div class="workflow-button">⌕ {"Validate path" if is_en else "验证路径"}</div>
                    {_workflow_status(step1_done, "Path validated" if is_en else "路径已确认", "Confirm data source" if is_en else "请确认数据源")}
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">B</div>
                        <div><div class="workflow-card-title">{"Cohort definition" if is_en else "队列定义"}</div><div class="workflow-card-kicker">Step 2</div></div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0.45rem;align-items:end">
                        {_workflow_field("Age range (years)" if is_en else "年龄范围", age_min, suffix="")}
                        <div style="padding-bottom:0.62rem;color:#60718a;font-weight:800">to</div>
                        {_workflow_field("", age_max, suffix="")}
                    </div>
                    {_workflow_field("ICU stay (hours)" if is_en else "ICU 住院时长", f"≥ {los_min}")}
                    {_workflow_field("Gender" if is_en else "性别", gender)}
                    {_workflow_field("Survival status" if is_en else "存活状态", survival_text)}
                    {_workflow_field("Clinical cohort" if is_en else "疾病队列", cohort_display)}
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.45rem">
                        {_workflow_field("ICD include" if is_en else "ICD 纳入", include_query, suffix="")}
                        {_workflow_field("ICD exclude" if is_en else "ICD 排除", exclude_query, suffix="")}
                    </div>
                    {_workflow_status(step2_done, "Cohort defined" if is_en else "队列已定义", "Confirm cohort" if is_en else "请确认队列")}
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">C</div>
                        <div><div class="workflow-card-title">{"Concept selection" if is_en else "概念选择"}</div><div class="workflow-card-kicker">Step 3</div></div>
                    </div>
                    <div class="workflow-label">{"Select modules" if is_en else "选择模块"}</div>
                    <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.38rem;margin-bottom:0.58rem">{group_chips}</div>
                    <div class="workflow-label">{"Select clinical concepts" if is_en else "选择临床概念"}</div>
                    <div class="workflow-concepts">{concepts_html}</div>
                    <div class="workflow-input" style="margin-top:0.65rem;background:#f5f7ff">{f"{len(get_all_concepts())} concepts available" if not selected_concepts else f"{len(selected_concepts)} concepts selected"}</div>
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">D</div>
                        <div><div class="workflow-card-title">{"Data export" if is_en else "数据导出"}</div><div class="workflow-card-kicker">Step 4</div></div>
                    </div>
                    {_workflow_field("Export path" if is_en else "导出路径", export_path, suffix="▣")}
                    {_workflow_field("Export format" if is_en else "导出格式", export_format)}
                    {_workflow_field("Patient limit" if is_en else "患者上限", patient_limit_text)}
                    <div class="workflow-button">⇧ {"Export data" if is_en else "导出数据"}</div>
                    <div class="workflow-status warn" style="font-weight:700">ⓘ {"Large exports run in the background." if is_en else "大规模导出将在后台运行。"}</div>
                </div>
            </div>
            <div class="workflow-summary-panel">
                <div class="workflow-card-head" style="margin-bottom:0.5rem">
                    <div class="workflow-badge">E</div>
                    <div><div class="workflow-card-title">{html.escape(summary_title)}</div><div class="workflow-card-kicker">Preview-before-commit</div></div>
                </div>
                <div class="workflow-summary-grid">
                    <div>
                        <div class="{summary_strip_class}"><span class="workflow-check-dot" style="background:{'#2ca25f' if summary_ready else '#f59e0b'}">{summary_icon}</span>{html.escape(summary_status)}</div>
                        <div class="workflow-stat-row">{stats_html}</div>
                    </div>
                    <div>
                        <div class="workflow-label">{"Exported files (Parquet)" if is_en else "导出文件 (Parquet)"}</div>
                        <div class="workflow-file-list">{export_files_html}</div>
                    </div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
