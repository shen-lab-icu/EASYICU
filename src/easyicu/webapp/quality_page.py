"""Data-quality page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_quality_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _quality_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _quality_rate_tone(value: float, *, warn: float = 5.0, danger: float = 20.0) -> str:
    if value >= danger:
        return "danger"
    if value >= warn:
        return "warning"
    return "ready"


def _quality_notice_html(tone: str, kicker: str, title: str, body: str, meta: str = "") -> str:
    meta_html = f'<em>{_quality_escape(meta)}</em>' if meta else ""
    return (
        f'<div class="eu-quality-notice {tone}">'
        f'<span>{_quality_escape(kicker)}</span>'
        f'<b>{_quality_escape(title)}</b>'
        f'<p>{_quality_escape(body)}</p>'
        f'{meta_html}'
        '</div>'
    )


def _render_quality_notice(tone: str, kicker: str, title: str, body: str, meta: str = "") -> None:
    st.markdown(_quality_notice_html(tone, kicker, title, body, meta), unsafe_allow_html=True)


def _quality_contract_row_html(index: str, label: str, detail: str, tone: str) -> str:
    return (
        f'<div class="eu-quality-contract-row {tone}">'
        f'<span>{_quality_escape(index)}</span>'
        '<div>'
        f'<b>{_quality_escape(label)}</b>'
        f'<em>{_quality_escape(detail)}</em>'
        '</div>'
        '</div>'
    )


def _quality_contract_html(
    *,
    lang: str,
    concept_count: int,
    patient_count: int,
    total_records: int,
    overall_missing: float,
    overall_outliers: float,
    overall_duplicates: float,
) -> str:
    is_en = lang == "en"
    rows = [
        _quality_contract_row_html(
            "01",
            "Local concept scope" if is_en else "本地概念范围",
            (
                f"{concept_count} concepts · {patient_count} ICU stays · {total_records:,} records"
                if is_en
                else f"{concept_count} 个概念 · {patient_count} 个 ICU stay · {total_records:,} 条记录"
            ),
            "ready" if concept_count and total_records else "neutral",
        ),
        _quality_contract_row_html(
            "02",
            "Missingness gate" if is_en else "缺失率关口",
            (
                f"{overall_missing:.1f}% weighted missing"
                if is_en
                else f"加权缺失 {overall_missing:.1f}%"
            ),
            _quality_rate_tone(overall_missing),
        ),
        _quality_contract_row_html(
            "03",
            "Physiologic range" if is_en else "生理范围",
            (
                f"{overall_outliers:.1f}% out-of-physio values"
                if is_en
                else f"越出生理范围 {overall_outliers:.1f}%"
            ),
            _quality_rate_tone(overall_outliers, warn=1.0, danger=5.0),
        ),
        _quality_contract_row_html(
            "04",
            "Temporal integrity" if is_en else "时序完整性",
            (
                f"{overall_duplicates:.1f}% duplicate patient-time rows"
                if is_en
                else f"重复患者-时间行 {overall_duplicates:.1f}%"
            ),
            _quality_rate_tone(overall_duplicates, warn=0.5, danger=2.0),
        ),
    ]
    return (
        '<div class="eu-quality-contract">'
        '<div class="eu-quality-contract-head">'
        f'<span>{_quality_escape("QC ledger" if is_en else "质控账本")}</span>'
        f'<b>{_quality_escape("local export -> denominator -> gate -> chart" if is_en else "本地导出 -> 分母 -> 关口 -> 图表")}</b>'
        '</div>'
        f'<p>{_quality_escape("Quality review stays tied to loaded local concepts, denominator definitions, and patient-time integrity before modeling." if is_en else "质控审阅绑定已加载本地概念、分母定义与患者-时间完整性，再进入建模。")}</p>'
        '<div class="eu-quality-contract-list">'
        + "".join(rows)
        + '</div>'
        '</div>'
    )


def _apply_quality_plot_style(
    fig: Any,
    *,
    title: str,
    height: int,
    x_title: str = "",
    y_title: str = "",
    showlegend: bool = False,
    margin: dict[str, int] | None = None,
) -> Any:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=13, color="#0e1116")),
        height=height,
        showlegend=showlegend,
        margin=margin or dict(l=76, r=28, t=46, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(
            family="IBM Plex Sans, PingFang SC, Hiragino Sans GB, system-ui, sans-serif",
            size=12,
            color="#2e3338",
        ),
        hoverlabel=dict(bgcolor="#0e1116", font=dict(color="#ffffff", size=12)),
    )
    fig.update_xaxes(
        title=x_title,
        gridcolor="#eeeee8",
        zerolinecolor="#dcdad2",
        tickfont=dict(size=11, color="#6b7280"),
        title_font=dict(size=12, color="#2e3338"),
        showline=True,
        linecolor="#e7e5df",
    )
    fig.update_yaxes(
        title=y_title,
        gridcolor="#eeeee8",
        zerolinecolor="#dcdad2",
        tickfont=dict(size=11, color="#6b7280"),
        title_font=dict(size=12, color="#2e3338"),
        showline=True,
        linecolor="#e7e5df",
    )
    return fig


def _render_quality_panel_switcher(lang: str, screenshot_mode: bool = False) -> str:
    panel_options = {
        "missingness": "Missingness" if lang == 'en' else "缺失分析",
        "outliers": "Out-of-Physio" if lang == 'en' else "生理范围越界",
        "temporal": "Temporal Integrity" if lang == 'en' else "时序完整性",
    }
    state_key = "quality_active_panel"
    if st.session_state.get(state_key) not in panel_options:
        st.session_state[state_key] = "missingness"

    if screenshot_mode:
        return "missingness"

    label = "Quality panel" if lang == 'en' else "质控面板"
    with st.container(key="quality_panel_switcher"):
        st.markdown(
            f'<div class="inline-control-label">{html.escape(label)}</div>',
            unsafe_allow_html=True,
        )
        return st.radio(
            label,
            options=list(panel_options.keys()),
            format_func=lambda key: panel_options[key],
            horizontal=True,
            key=state_key,
            label_visibility="collapsed",
        )


def render_quality_page(app_context: dict[str, Any] | None = None):
    """渲染数据质量页面。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')
    screenshot_mode = _is_screenshot_mode()

    page_title = "Data Quality" if lang == 'en' else "数据质量评估"
    page_sub = "Missing rate analysis, coverage badges & explainable causes" if lang == 'en' else "缺失率分析、覆盖度标识与可解释原因"
    st.markdown(f'''
    <div class="eu-subhead">
        <div class="t">{page_title}</div>
        <div class="s">{page_sub}</div>
    </div>
    ''', unsafe_allow_html=True)

    if screenshot_mode:
        focus_note = (
            "Figure preset: keeping summary cards and charts prominent while moving the detailed report out of the way."
            if lang == 'en'
            else "截图预设：优先突出摘要卡片和图表，并弱化详细报告的存在感。"
        )
        st.markdown(f'<div class="compact-inline-notice info">{focus_note}</div>', unsafe_allow_html=True)

    if len(st.session_state.loaded_concepts) == 0:
        _render_quality_notice(
            "info",
            "Quality workspace" if lang == "en" else "质控工作台",
            "Local export required" if lang == "en" else "需要本地导出",
            (
                'Load concepts from Data Extraction or use Demo Mode before reviewing missingness, physiologic ranges, and temporal integrity.'
                if lang == "en"
                else "请先从 Data Extraction 加载概念，或使用演示模式，再审阅缺失率、生理范围和时序完整性。"
            ),
            'Data Extraction -> Patient Review -> Data Quality',
        )
        return

    mock_params = st.session_state.get('mock_params', {}) or {}
    demo_hours = int(mock_params.get('hours') or 0) if st.session_state.get('entry_mode') == 'demo' and mock_params.get('hours') else None
    time_grid_size = demo_hours or 72
    id_col = st.session_state.get('id_col', 'stay_id')
    total_patients_in_session = _get_quality_cohort_patient_count(st.session_state)
    cohort_patient_ids = _get_quality_cohort_patient_ids(st.session_state)
    los_by_patient = _get_quality_los_by_patient(st.session_state)

    records_col = "Records" if lang == 'en' else "记录数"
    patients_col = "ICU stays" if lang == 'en' else "ICU stay"
    missing_col = "Missing %" if lang == 'en' else "缺失率"
    denom_col = "Denom" if lang == 'en' else "分母"
    out_col = "% Out-of-physio" if lang == 'en' else "越出生理范围%"
    dup_col = "Dup TS %" if lang == 'en' else "重复时间戳%"
    density_col = "Density / h" if lang == 'en' else "密度 / 小时"
    coverage_col = "Coverage" if lang == 'en' else "覆盖度"
    cause_col = "Likely Cause" if lang == 'en' else "可能原因"

    quality_rows: list[dict[str, Any]] = []
    total_records = 0
    total_expected = 0.0
    total_missing_weight = 0.0
    total_outlier_weight = 0.0
    total_duplicate_weight = 0.0

    for concept, df in st.session_state.loaded_concepts.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        n_records = len(df)
        n_patients = df[id_col].nunique() if id_col in df.columns else 0
        profile = _build_quality_metric_profile_cached(
            concept=concept,
            df=df,
            id_col=id_col,
            cohort_patient_count=total_patients_in_session,
            time_grid_size=time_grid_size,
            cohort_patient_ids=cohort_patient_ids,
            los_by_patient=los_by_patient,
            demo_hours=demo_hours,
        )

        total_records += n_records
        weight = float(profile['expected_observations'] or n_records or 1)
        total_expected += weight
        total_missing_weight += weight * (profile['missing_rate'] / 100)
        total_outlier_weight += n_records * (profile['out_of_physio_rate'] / 100)
        total_duplicate_weight += n_records * (profile['duplicate_rate'] / 100)

        badge, _current_supported, _n_db = _get_concept_coverage_summary(
            concept,
            current_database=st.session_state.get('database', ''),
        )
        cause_text, _cause_color = _get_missing_cause_tag(
            concept,
            profile['missing_rate'] / 100.0,
            current_database=st.session_state.get('database', ''),
            has_observed_rows=n_records > 0,
        )

        quality_rows.append({
            'Concept': concept,
            records_col: f"{n_records:,}",
            patients_col: n_patients,
            missing_col: f"{profile['missing_rate']:.1f}%",
            denom_col: profile['denominator_tag'],
            out_col: f"{profile['out_of_physio_rate']:.1f}%",
            dup_col: f"{profile['duplicate_rate']:.1f}%",
            density_col: _format_quality_density(profile['temporal_density'], lang),
            coverage_col: badge,
            cause_col: cause_text,
            '_records': n_records,
            '_patients': n_patients,
            '_missing_rate': float(profile['missing_rate']),
            '_out_rate': float(profile['out_of_physio_rate']),
            '_dup_rate': float(profile['duplicate_rate']),
            '_density_median': float(profile['temporal_density'].get('median', 0.0)),
            '_density_q25': float(profile['temporal_density'].get('q25', 0.0)),
            '_density_q75': float(profile['temporal_density'].get('q75', 0.0)),
            '_denominator_tag': profile['denominator_tag'],
        })

    quality_df = pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame()
    overall_missing = (total_missing_weight / total_expected * 100) if total_expected > 0 else 0.0
    overall_outliers = (total_outlier_weight / total_records * 100) if total_records > 0 else 0.0
    overall_duplicates = (total_duplicate_weight / total_records * 100) if total_records > 0 else 0.0

    records_label = "Total Records" if lang == 'en' else "总记录数"
    missing_label = "Weighted Missing" if lang == 'en' else "加权缺失率"
    outlier_label = "Out-of-physio" if lang == 'en' else "越出生理范围"
    duplicate_label = "Duplicate TS" if lang == 'en' else "重复时间戳"

    def _metric_color(value: float) -> str:
        if value < 5:
            return "#0f766e"
        if value < 20:
            return "#b45309"
        return "#b91c1c"

    st.markdown(f'''
    <div class="quality-summary-grid">
        <div class="quality-summary-card">
            <div class="quality-summary-label">{html.escape(records_label)}</div>
            <div class="quality-summary-value">{total_records:,}</div>
        </div>
        <div class="quality-summary-card">
            <div class="quality-summary-label">{html.escape(missing_label)}</div>
            <div class="quality-summary-value" style="color:{_metric_color(overall_missing)}">{overall_missing:.1f}%</div>
        </div>
        <div class="quality-summary-card">
            <div class="quality-summary-label">{html.escape(outlier_label)}</div>
            <div class="quality-summary-value" style="color:{_metric_color(overall_outliers)}">{overall_outliers:.1f}%</div>
        </div>
        <div class="quality-summary-card">
            <div class="quality-summary-label">{html.escape(duplicate_label)}</div>
            <div class="quality-summary-value" style="color:{_metric_color(overall_duplicates)}">{overall_duplicates:.1f}%</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(
        _quality_contract_html(
            lang=lang,
            concept_count=len(quality_df),
            patient_count=total_patients_in_session,
            total_records=total_records,
            overall_missing=overall_missing,
            overall_outliers=overall_outliers,
            overall_duplicates=overall_duplicates,
        ),
        unsafe_allow_html=True,
    )

    def _quality_action_for_row(row: Any) -> str:
        concept = str(row.get('Concept') or "").lower()
        denom = str(row.get('_denominator_tag') or "")
        missing_rate = float(row.get('_missing_rate') or 0.0)
        is_demo = st.session_state.get('entry_mode') == 'demo' or st.session_state.get('use_mock_data') or denom == "demo"
        if missing_rate < 25:
            return (
                "No action unless this is a required endpoint."
                if lang == 'en' else
                "除非是必需终点，否则通常无需处理。"
            )
        if is_demo:
            if concept in {"cort", "abx", "vaso_ind", "rrt", "ecmo", "mech_circ_support"}:
                return (
                    "Demo event sparsity: treat as a workflow stress test; do not model this exposure from the demo cohort."
                    if lang == 'en' else
                    "演示事件稀疏：只作为工作流压力测试；不要用演示队列建模该暴露。"
                )
            if concept in {"samp", "cult", "culture"} or "samp" in concept or "cult" in concept:
                return (
                    "Demo sampling sparsity: check the infection-window definition before using this as suspected-infection evidence."
                    if lang == 'en' else
                    "演示采样稀疏：作为疑似感染证据前，先检查感染时间窗定义。"
                )
            if "vent" in concept:
                return (
                    "Demo support indicator: verify ventilation eligibility/window before using it as an exposure or subgroup."
                    if lang == 'en' else
                    "演示支持治疗指标：作为暴露或亚组前，先确认通气资格和时间窗。"
                )
            return (
                "Demo sparsity: use this to review denominator handling, not to draw clinical conclusions."
                if lang == 'en' else
                "演示稀疏性：用于检查分母处理，不用于临床结论。"
            )
        if concept in {"cort", "abx", "vaso_ind", "rrt", "ecmo", "mech_circ_support"}:
            return (
                "Sparse treatment/event variable: report availability, consider an availability indicator, and prespecify the exposure window."
                if lang == 'en' else
                "稀疏治疗/事件变量：报告可用性，考虑可用性指示变量，并预先指定暴露时间窗。"
            )
        if concept in {"samp", "cult", "culture"} or "samp" in concept or "cult" in concept:
            return (
                "Sampling variable: audit microbiology/lab source coverage and infection-window logic before Sepsis-3 grouping."
                if lang == 'en' else
                "采样变量：用于 Sepsis-3 分组前，先审计微生物/化验来源覆盖和感染时间窗逻辑。"
            )
        if "vent" in concept:
            return (
                "Respiratory support variable: verify device table coverage and align the exposure window to the cohort index time."
                if lang == 'en' else
                "呼吸支持变量：确认设备表覆盖，并把暴露时间窗对齐到队列 index time。"
            )
        if denom == "static":
            return (
                "Confirm the static source table before using this field as an adjustment variable."
                if lang == 'en' else
                "作为调整变量前，先确认静态来源表覆盖。"
            )
        if denom in {"LOS", "72h", "demo"}:
            return (
                "Check whether sparse measurement is expected; model availability or narrow the analysis window."
                if lang == 'en' else
                "先判断稀疏测量是否符合预期；必要时建模可用性或收窄分析窗口。"
            )
        return (
            "Review source coverage before modeling."
            if lang == 'en' else
            "建模前先复核来源覆盖。"
        )

    def _render_quality_top_issues() -> None:
        if quality_df.empty:
            return
        high_missing = (
            quality_df.sort_values(['_missing_rate', '_records'], ascending=[False, False])
            .head(3)
            .copy()
        )
        issue_cards = []
        for _, row in high_missing.iterrows():
            concept = str(row.get('Concept') or "")
            missing_rate = float(row.get('_missing_rate') or 0.0)
            records = int(row.get('_records') or 0)
            denom = str(row.get('_denominator_tag') or "")
            denom_label = denom if denom.startswith("d=") else f"d={denom or 'unknown'}"
            issue_cards.append(
                '<div class="quality-issue-card">'
                f'<b>{html.escape(concept)}</b>'
                f'<span>{missing_rate:.1f}% {html.escape("missing" if lang == "en" else "缺失")} · {records:,} {html.escape("records" if lang == "en" else "条记录")} · {html.escape(denom_label)}</span>'
                f'<em>{html.escape(_quality_action_for_row(row))}</em>'
                '</div>'
            )
        outlier_concepts = int((quality_df['_out_rate'] > 0).sum())
        duplicate_concepts = int((quality_df['_dup_rate'] > 0).sum())
        footer_note = (
            f"{outlier_concepts} concepts have physiologic-range flags; {duplicate_concepts} have duplicate timestamps."
            if lang == 'en' else
            f"{outlier_concepts} 个概念存在生理范围标记；{duplicate_concepts} 个概念存在重复时间戳。"
        )
        st.markdown(
            '<div class="quality-issue-panel">'
            f'<div class="quality-issue-head"><span>{html.escape("Top quality issues" if lang == "en" else "优先质控问题")}</span>'
            f'<em>{html.escape("review before modeling" if lang == "en" else "建模前先复核")}</em></div>'
            f'<div class="quality-issue-grid">{"".join(issue_cards)}</div>'
            f'<p>{html.escape(footer_note)}</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    _render_quality_top_issues()

    detail_title = "Detailed QC Report" if lang == 'en' else "详细质控报告"
    denom_caption = (
        "Missingness denominator tags: d=LOS uses stay-specific ICU time, d=72h uses the fallback window, d=demo uses the demo horizon, d=static means one observation per ICU stay."
        if lang == 'en'
        else "缺失率分母说明：d=LOS 表示按 ICU stay 时长估算，d=72h 表示 72 小时兜底窗口，d=demo 表示演示数据预设时间窗，d=static 表示每个 ICU stay 一次静态观测。"
    )

    def _render_quality_detail_report() -> None:
        if quality_df.empty:
            return
        display_cols = ['Concept', records_col, patients_col, missing_col, denom_col, out_col, dup_col, density_col, coverage_col, cause_col]
        _st_dataframe_compat(
            st,
            quality_df[display_cols],
            width="stretch",
            hide_index=True,
        )
        st.caption(denom_caption)
        if not screenshot_mode:
            _render_ai_context_button(
                'ai_why_missing',
                context=f"database={st.session_state.get('database', '')}; loaded_concepts={len(st.session_state.get('loaded_concepts', {}))}; explain missingness, physiologic range outliers, and temporal integrity issues from the current QC summary",
            )

    active_quality_panel = _render_quality_panel_switcher(lang, screenshot_mode=screenshot_mode)

    if active_quality_panel == "missingness":
        if screenshot_mode:
            sort_order = 'desc'
        else:
            sort_label = "Sort by" if lang == 'en' else "排序方式"
            if 'missing_chart_sort_order' not in st.session_state:
                st.session_state['missing_chart_sort_order'] = 'desc'
            sort_options = {
                'desc': 'Missing Rate (High → Low)' if lang == 'en' else '缺失率 (从高到低)',
                'asc': 'Missing Rate (Low → High)' if lang == 'en' else '缺失率 (从低到高)',
                'alpha': 'Alphabetical (A → Z)' if lang == 'en' else '首字母排序 (A → Z)',
            }
            sort_order = st.radio(
                sort_label,
                options=list(sort_options.keys()),
                format_func=lambda x: sort_options[x],
                horizontal=True,
                key='missing_chart_sort_order',
            )

        if quality_df.empty:
            _render_quality_notice(
                "info",
                "Missingness gate" if lang == "en" else "缺失率关口",
                "No quality metrics available" if lang == "en" else "当前没有可用的质量指标",
                (
                    "Loaded concepts did not produce QC rows. Recheck the selected concept tables and patient identifier."
                    if lang == "en"
                    else "已加载概念没有生成质控行，请复核概念表和患者标识列。"
                ),
            )
        else:
            import plotly.express as px

            missing_plot_df = quality_df[['Concept', '_missing_rate', '_records', '_patients', '_denominator_tag']].copy()
            missing_rate_label = "Missing Rate (%)" if lang == 'en' else "缺失率 (%)"
            denom_hover = "Denominator" if lang == 'en' else "分母来源"
            missing_plot_df[missing_rate_label] = missing_plot_df['_missing_rate']
            missing_plot_df[records_col] = missing_plot_df['_records']
            missing_plot_df[patients_col] = missing_plot_df['_patients']
            missing_plot_df[denom_hover] = missing_plot_df['_denominator_tag'].apply(lambda x: _get_quality_denominator_note(x, lang))

            if sort_order == 'desc':
                missing_plot_df = missing_plot_df.sort_values(missing_rate_label, ascending=False)
            elif sort_order == 'alpha':
                missing_plot_df = missing_plot_df.sort_values('Concept', ascending=True)
            else:
                missing_plot_df = missing_plot_df.sort_values(missing_rate_label, ascending=True)

            st.caption(denom_caption)
            if missing_plot_df[missing_rate_label].sum() == 0:
                _render_quality_notice(
                    "ready",
                    "Missingness gate" if lang == "en" else "缺失率关口",
                    "Missingness is negligible" if lang == "en" else "缺失率很低",
                    (
                        "Loaded concepts are complete enough for downstream review; keep denominator notes in the report."
                        if lang == "en"
                        else "已加载概念足够完整，可进入下游审阅；报告中仍保留分母说明。"
                    ),
                )
            else:
                # In screenshot mode keep a compact fixed size; in the
                # interactive web view show ALL concepts so nothing is hidden.
                # The chart height auto-scales and Streamlit's scrollable
                # container handles overflow.
                screenshot_limit = 18
                total_quality_concepts = len(missing_plot_df)
                if screenshot_mode:
                    chart_df = missing_plot_df.head(screenshot_limit).copy()
                else:
                    chart_df = missing_plot_df.copy()
                chart_df['_missing_bin'] = pd.cut(
                    chart_df[missing_rate_label],
                    bins=[-0.001, 25, 50, 75, 100],
                    labels=['< 25', '25–50', '50–75', '75–100'],
                    include_lowest=True,
                ).astype(str)
                bin_label = "Missing rate bin" if lang == 'en' else "缺失率区间"
                chart_df[bin_label] = chart_df['_missing_bin']
                fig = px.bar(
                    chart_df,
                    x=missing_rate_label,
                    y='Concept',
                    orientation='h',
                    color=bin_label,
                    color_discrete_map={
                        '< 25': '#0f766e',
                        '25–50': '#b45309',
                        '50–75': '#c2410c',
                        '75–100': '#b91c1c',
                    },
                    hover_data=[records_col, patients_col, denom_hover],
                )
                fig = _apply_quality_plot_style(
                    fig,
                    title='Missingness by concept' if lang == 'en' else '各概念缺失率',
                    height=max(340, len(chart_df) * 34 + 110),
                    x_title=missing_rate_label,
                    y_title="",
                    showlegend=True,
                    margin=dict(l=92, r=160, t=44, b=44),
                )
                fig.update_layout(
                    legend=dict(
                        title='Missing rate (%)' if lang == 'en' else '缺失率 (%)',
                        orientation='v',
                        x=1.02,
                        y=0.72,
                        bgcolor='rgba(255,255,255,0.92)',
                        bordercolor='#e7e5df',
                        borderwidth=1,
                        font=dict(size=12, color='#2e3338'),
                    ),
                    yaxis=dict(autorange='reversed', title=""),
                )
                fig.update_xaxes(range=[0, 100])
                if total_quality_concepts > len(chart_df):
                    fig.add_annotation(
                        xref='paper', yref='paper', x=1.0, y=1.08,
                        text=(
                            f"Showing {len(chart_df)} of {total_quality_concepts}"
                            if lang == 'en'
                            else f"显示 {len(chart_df)} / {total_quality_concepts}"
                        ),
                        showarrow=False,
                        font=dict(size=11, color='#6b7280'),
                        align='right',
                )
                st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

    elif active_quality_panel == "outliers":
        if quality_df.empty:
            _render_quality_notice(
                "info",
                "Physiologic range" if lang == "en" else "生理范围",
                "No quality metrics available" if lang == "en" else "当前没有可用的质量指标",
                (
                    "Range QC needs loaded numeric concept profiles."
                    if lang == "en"
                    else "生理范围质控需要已加载的数值型概念画像。"
                ),
            )
        else:
            import plotly.express as px

            range_note = (
                "Out-of-physio % uses harmonized physiologic bounds after unit normalization when available. It highlights implausible values, not ordinary clinical abnormalities."
                if lang == 'en'
                else "越出生理范围比例基于统一的生理合理区间，并在可用时按单位归一化后计算。它提示不合理值，而不是一般性的临床异常。"
            )
            st.caption(range_note)

            outlier_df = quality_df[['Concept', '_out_rate', '_records']].copy()
            outlier_rate_label = "% Out-of-physio" if lang == 'en' else "越出生理范围 (%)"
            outlier_df[outlier_rate_label] = outlier_df['_out_rate']
            outlier_df[records_col] = outlier_df['_records']
            outlier_df = outlier_df.sort_values(outlier_rate_label, ascending=False)

            if outlier_df[outlier_rate_label].max() <= 0:
                _render_quality_notice(
                    "ready",
                    "Physiologic range" if lang == "en" else "生理范围",
                    "No range flags detected" if lang == "en" else "未发现越界值",
                    (
                        "Loaded numeric concepts currently stay inside configured physiologic QC ranges."
                        if lang == "en"
                        else "当前已加载数值概念均在配置的生理质控范围内。"
                    ),
                )
            else:
                fig = px.bar(
                    outlier_df,
                    x=outlier_rate_label,
                    y='Concept',
                    orientation='h',
                    color=outlier_rate_label,
                    color_continuous_scale=['#ccfbf1', '#fbbf24', '#b91c1c'],
                    hover_data=[records_col],
                )
                fig = _apply_quality_plot_style(
                    fig,
                    title='Physiologic Range QC' if lang == 'en' else '生理范围质控',
                    height=max(320, len(outlier_df) * 36),
                    x_title=outlier_rate_label,
                    y_title="",
                    showlegend=False,
                    margin=dict(l=90, r=20, t=44, b=40),
                )
                fig.update_yaxes(autorange='reversed', title="")
                st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

    elif active_quality_panel == "temporal":
        if quality_df.empty:
            _render_quality_notice(
                "info",
                "Temporal integrity" if lang == "en" else "时序完整性",
                "No quality metrics available" if lang == "en" else "当前没有可用的质量指标",
                (
                    "Temporal QC needs loaded concept profiles with patient-time rows."
                    if lang == "en"
                    else "时序质控需要已加载概念的患者-时间记录画像。"
                ),
            )
        else:
            import plotly.express as px

            temporal_note = (
                "Duplicate timestamps are counted only when the same patient and concept repeat at the same timestamp. Density is records / patient / expected hour, summarized as median [IQR]."
                if lang == 'en'
                else "重复时间戳仅在同一患者的同一概念于同一时间重复时计为问题。密度定义为 records / patient / expected hour，并汇总为 median [IQR]。"
            )
            st.caption(temporal_note)

            temporal_cols = st.columns(2)
            with temporal_cols[0]:
                duplicate_df = quality_df[['Concept', '_dup_rate', '_records']].copy()
                duplicate_rate_label = "Duplicate TS (%)" if lang == 'en' else "重复时间戳 (%)"
                duplicate_df[duplicate_rate_label] = duplicate_df['_dup_rate']
                duplicate_df[records_col] = duplicate_df['_records']
                duplicate_df = duplicate_df.sort_values(duplicate_rate_label, ascending=False)
                if duplicate_df[duplicate_rate_label].max() <= 0:
                    _render_quality_notice(
                        "ready",
                        "Temporal integrity" if lang == "en" else "时序完整性",
                        "No duplicate patient-time rows" if lang == "en" else "未发现重复患者-时间行",
                        (
                            "Loaded concepts do not show duplicate timestamps for the same patient and concept."
                            if lang == "en"
                            else "当前已加载概念没有同一患者同一概念的重复时间戳。"
                        ),
                    )
                else:
                    fig_dup = px.bar(
                        duplicate_df,
                        x=duplicate_rate_label,
                        y='Concept',
                        orientation='h',
                        color=duplicate_rate_label,
                        color_continuous_scale=['#ccfbf1', '#fbbf24', '#b91c1c'],
                        hover_data=[records_col],
                    )
                    fig_dup = _apply_quality_plot_style(
                        fig_dup,
                        title='Duplicate Timestamp Rate' if lang == 'en' else '重复时间戳比例',
                        height=max(320, len(duplicate_df) * 34),
                        x_title=duplicate_rate_label,
                        y_title="",
                        showlegend=False,
                        margin=dict(l=90, r=20, t=44, b=40),
                    )
                    fig_dup.update_yaxes(autorange='reversed', title="")
                    st.plotly_chart(fig_dup, use_container_width=True, config=_get_plotly_chart_config())

            with temporal_cols[1]:
                density_df = quality_df[['Concept', '_density_median', '_density_q25', '_density_q75', '_missing_rate', '_dup_rate', '_denominator_tag']].copy()
                density_label = "Median records / patient / hour" if lang == 'en' else "中位 records / patient / hour"
                missing_label = "Missing Rate (%)" if lang == 'en' else "缺失率 (%)"
                dup_label = "Duplicate TS (%)" if lang == 'en' else "重复时间戳 (%)"
                iqr_label = "IQR" if lang == 'en' else "IQR"

                density_df = density_df[density_df['_density_median'] > 0].copy()
                if density_df.empty:
                    _render_quality_notice(
                        "info",
                        "Density profile" if lang == "en" else "密度画像",
                        "Time-stamped concepts required" if lang == "en" else "需要带时间戳的概念",
                        (
                            "Density summaries appear after loaded concepts include usable patient-time observations."
                            if lang == "en"
                            else "已加载概念包含可用患者-时间观测后，才会显示密度摘要。"
                        ),
                    )
                else:
                    has_duplicates = float(density_df['_dup_rate'].max() or 0) > 0
                    density_df['_iqr_text'] = density_df.apply(
                        lambda r: f"{r['_density_median']:.2f} [{r['_density_q25']:.2f}-{r['_density_q75']:.2f}]",
                        axis=1,
                    )

                    if has_duplicates:
                        # Keep a scatter but make it readable: hover-only labels for
                        # the bulk of concepts, always-on labels for the top-N outliers
                        # (highest density or highest duplicate rate).
                        outlier_keys = set(density_df.nlargest(5, '_density_median')['Concept'].tolist())
                        outlier_keys |= set(density_df.nlargest(5, '_dup_rate')['Concept'].tolist())
                        density_df['_label'] = density_df['Concept'].where(density_df['Concept'].isin(outlier_keys), '')
                        fig_density = px.scatter(
                            density_df,
                            x='_density_median',
                            y='_dup_rate',
                            size='_missing_rate',
                            color='_missing_rate',
                            text='_label',
                            hover_name='Concept',
                            hover_data={'_density_median': ':.2f', '_dup_rate': ':.2f', '_missing_rate': ':.1f', '_iqr_text': True, '_label': False},
                            color_continuous_scale=['#0f766e', '#fbbf24', '#b91c1c'],
                            labels={
                                '_density_median': density_label,
                                '_dup_rate': dup_label,
                                '_missing_rate': missing_label,
                                '_iqr_text': iqr_label,
                            },
                        )
                        fig_density.update_traces(textposition='top center', textfont=dict(size=11))
                        fig_density = _apply_quality_plot_style(
                            fig_density,
                            title='Temporal Density vs Duplicate Rate' if lang == 'en' else '时序密度与重复率',
                            height=420,
                            x_title=density_label,
                            y_title=dup_label,
                            margin=dict(l=30, r=20, t=44, b=40),
                        )
                        st.plotly_chart(fig_density, use_container_width=True, config=_get_plotly_chart_config())
                    else:
                        # No duplicate signal to split against: a stacked scatter at y=0
                        # with 167 overlapping labels is unreadable. Pivot to a
                        # density-ranked bar chart colored by missingness, capped to
                        # the top-K concepts so the chart stays readable.
                        top_k = 25
                        ranked = density_df.sort_values('_density_median', ascending=False).head(top_k).copy()
                        ranked = ranked.sort_values('_density_median', ascending=True)
                        fig_density = px.bar(
                            ranked,
                            x='_density_median',
                            y='Concept',
                            orientation='h',
                            color='_missing_rate',
                            color_continuous_scale=['#0f766e', '#fbbf24', '#b91c1c'],
                            hover_data={
                                '_density_median': ':.2f',
                                '_density_q25': ':.2f',
                                '_density_q75': ':.2f',
                                '_missing_rate': ':.1f',
                                '_denominator_tag': True,
                                '_iqr_text': True,
                            },
                            labels={
                                '_density_median': density_label,
                                '_missing_rate': missing_label,
                                '_density_q25': "Q25",
                                '_density_q75': "Q75",
                                '_denominator_tag': "Denom" if lang == 'en' else "分母",
                                '_iqr_text': iqr_label,
                            },
                            title=(
                                f"Top {len(ranked)} concepts by density"
                                if lang == 'en'
                                else f"密度排名前 {len(ranked)} 的概念"
                            ),
                        )
                        fig_density = _apply_quality_plot_style(
                            fig_density,
                            title=(
                                f"Top {len(ranked)} concepts by density"
                                if lang == 'en'
                                else f"密度排名前 {len(ranked)} 的概念"
                            ),
                            height=max(320, len(ranked) * 22),
                            x_title=density_label,
                            y_title="",
                            margin=dict(l=90, r=20, t=44, b=40),
                            showlegend=False,
                        )
                        fig_density.update_yaxes(title="")
                        total_concepts = int(len(quality_df[quality_df['_density_median'] > 0]))
                        if total_concepts > len(ranked):
                            fig_density.add_annotation(
                                xref='paper', yref='paper', x=1.0, y=1.02,
                                text=(
                                    f"Showing {len(ranked)} of {total_concepts} time-stamped concepts"
                                    if lang == 'en'
                                    else f"显示 {len(ranked)} / {total_concepts} 个带时间戳的概念"
                                ),
                                showarrow=False,
                                font=dict(size=11, color='#6b7280'),
                                align='right',
                            )
                        st.plotly_chart(fig_density, use_container_width=True, config=_get_plotly_chart_config())

    if not quality_df.empty and not screenshot_mode:
        with st.expander(detail_title, expanded=False):
            _render_quality_detail_report()
