"""Cohort severity-reclassification subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {
        'render_severity_reclassification_subtab',
        "_install_app_context",
        "_render_section_heading",
    }
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _render_section_heading(
    title: str,
    eyebrow: str | None = None,
    subtitle: str | None = None,
) -> None:
    eyebrow_html = (
        f'<span>{html.escape(eyebrow)}</span>'
        if eyebrow else ""
    )
    subtitle_html = (
        f'<p>{html.escape(subtitle)}</p>'
        if subtitle else ""
    )
    st.markdown(
        '<div class="eu-native-section-heading">'
        f'{eyebrow_html}<b>{html.escape(title)}</b>{subtitle_html}'
        '</div><div class="eu-native-section-heading-after" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )


def _render_chart_heading(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    eyebrow_html = f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        '<div class="eu-chart-heading">'
        f'{eyebrow_html}'
        f'<div class="title">{html.escape(title)}</div>'
        f'<div class="subtitle">{html.escape(subtitle)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _reclass_notice_html(tone: str, kicker: str, title: str, body: str = "", meta: str = "") -> str:
    body_html = f'<p>{html.escape(body)}</p>' if body else ""
    meta_html = f'<em>{html.escape(meta)}</em>' if meta else ""
    return (
        f'<div class="eu-cohort-loader-notice {html.escape(tone)}">'
        f'<span>{html.escape(kicker)}</span>'
        f'<b>{html.escape(title)}</b>'
        f'{body_html}'
        f'{meta_html}'
        '</div>'
    )


def _render_reclass_notice(tone: str, kicker: str, title: str, body: str = "", meta: str = "") -> None:
    st.markdown(
        _reclass_notice_html(tone, kicker, title, body, meta),
        unsafe_allow_html=True,
    )


RECLASS_CHART = {
    "ink": "#1d2935",
    "muted": "#65727f",
    "grid": "#e8e2d8",
    "axis": "#d9d2c7",
    "plot": "#fbfaf7",
    "teal": "#0f766e",
    "teal_soft": "#d8ece8",
    "teal_line": "#a9cbc5",
    "teal_mid": "#72aaa0",
    "rose": "#9f3a57",
    "rose_soft": "#f3dbe2",
    "slate": "#697684",
    "slate_soft": "#e4e7ea",
}


def _style_reclass_figure(fig, *, height: int, margin: dict[str, int] | None = None, legend_y: float = 1.12):
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=margin or dict(l=64, r=58, t=42, b=58),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=RECLASS_CHART["plot"],
        font=dict(
            family='IBM Plex Sans, Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            size=12,
            color=RECLASS_CHART["ink"],
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=legend_y,
            xanchor='right',
            x=1,
            bgcolor='rgba(251,250,247,0.96)',
            bordercolor='rgba(217,210,199,0.88)',
            borderwidth=1,
            font=dict(size=11, color=RECLASS_CHART["ink"]),
        ),
        hoverlabel=dict(bgcolor='#102a2d', font_size=12, font_color='#FFFFFF'),
    )
    fig.update_xaxes(
        gridcolor=RECLASS_CHART["grid"],
        zeroline=False,
        linecolor=RECLASS_CHART["axis"],
        tickfont=dict(size=11, color=RECLASS_CHART["muted"]),
        title_font=dict(size=12, color=RECLASS_CHART["muted"]),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=RECLASS_CHART["grid"],
        zeroline=False,
        linecolor=RECLASS_CHART["axis"],
        tickfont=dict(size=11, color=RECLASS_CHART["muted"]),
        title_font=dict(size=12, color=RECLASS_CHART["muted"]),
        automargin=True,
    )
    return fig


def render_severity_reclassification_subtab(lang: str, app_context: dict[str, Any] | None = None):
    """Render cohort-level SOFA-1 to SOFA-2 severity reclassification analysis."""
    if app_context is not None:
        _install_app_context(app_context)
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    screenshot_mode = _is_screenshot_mode()
    if st.session_state.get('entry_mode') == 'demo':
        _ensure_cohort_demo_workspace(st.session_state, lang=lang)

    # P0-2: If real workspace loaded SOFA concepts, ensure loaded_concepts is seeded
    if (st.session_state.get('entry_mode') == 'real'
        and _cohort_real_workspace_ready(st.session_state)):
        ws_concepts = st.session_state.get('_cohort_real_ws_concepts', {})
        if ws_concepts:
            existing = dict(st.session_state.get('loaded_concepts', {}) or {})
            for k, v in ws_concepts.items():
                if k not in existing:
                    existing[k] = v
            st.session_state['loaded_concepts'] = existing

    title = "SOFA-1 vs SOFA-2 Definition Sensitivity" if lang == 'en' else "SOFA-1 与 SOFA-2 定义敏感性"
    subtitle = (
        "Dedicated reclassification analysis under an explicit aggregation rule: ICU worst score, first-24h paired worst score, or time-aligned paired points."
        if lang == 'en' else
        "专门分析 SOFA-1 切换到 SOFA-2 后的重新分层；可选择 ICU 全程最高分、首24小时配对最高分，或同时间点配对。"
    )
    if not screenshot_mode:
        _render_section_heading(
            title,
            "Sensitivity" if lang == 'en' else "敏感性",
            subtitle,
        )

    mode_labels = {
        key: cfg['label_en'] if lang == 'en' else cfg['label_zh']
        for key, cfg in SOFA_RECLASS_ANALYSIS_MODES.items()
    }
    loaded_concepts = st.session_state.get('loaded_concepts', {})
    demo_concepts = _get_demo_sofa_timeseries_concepts()
    mode_availability = _get_sofa_reclassification_mode_availability(loaded_concepts)
    if mode_availability['locked'] and demo_concepts:
        demo_availability = _get_sofa_reclassification_mode_availability(demo_concepts)
        if len(demo_availability['available']) > len(mode_availability['available']):
            mode_availability = demo_availability
    available_modes = mode_availability['available']
    locked_modes = mode_availability['locked']
    mode_state_key = "sofa_reclass_analysis_mode_key"
    if st.session_state.get(mode_state_key) not in available_modes:
        st.session_state[mode_state_key] = available_modes[0]

    if screenshot_mode:
        mode = 'worst_icu' if 'worst_icu' in available_modes else available_modes[0]
        st.session_state[mode_state_key] = mode
    else:
        mode = st.radio(
            "Analysis definition" if lang == 'en' else "分析口径",
            available_modes,
            format_func=lambda key: mode_labels[key],
            horizontal=True,
            key=mode_state_key,
            help=(
                "Time-aligned modes require loaded SOFA-1/SOFA-2 time-series concepts from Quick Visualization."
                if lang == 'en' else
                "同时间点相关模式需要先在快速可视化中载入 SOFA-1/SOFA-2 时间序列。"
            ),
        )
    mode_cfg = SOFA_RECLASS_ANALYSIS_MODES[mode]
    if not screenshot_mode:
        st.caption(mode_cfg['description_en'] if lang == 'en' else mode_cfg['description_zh'])
    if locked_modes and not screenshot_mode:
        locked_text = ", ".join(mode_labels[key] for key in locked_modes)
        _render_reclass_notice(
            "warning",
            "Definition gate" if lang == 'en' else "定义门禁",
            "Time-aligned modes are locked" if lang == 'en' else "同时间点口径暂未解锁",
            (
                f"{locked_text} {'is' if len(locked_modes) == 1 else 'are'} locked: "
                "this mode needs time-aligned `sofa` and `sofa2` concepts. "
                "Load both via Patient Review > Previously Exported Data "
                "or generate them in Demo Mode."
                if lang == 'en' else
                f"{locked_text} 暂未解锁："
                "该口径需要同时间点的 `sofa` 与 `sofa2` 概念。"
                "请在患者审阅 > 之前导出的结果文件中同时加载二者，"
                "或在演示模式中生成。"
            ),
        )

    source_df, source_label = _get_sofa_reclassification_source(lang, mode=mode)
    if source_df.empty:
        if st.session_state.get('entry_mode') == 'real' and not screenshot_mode:
            st.markdown(
                f"""
                <div class="viz-demo-load-card">
                    <div class="viz-demo-load-kicker">REAL DATA</div>
                    <div class="viz-demo-load-title">{html.escape('Load SOFA-1/SOFA-2 from current data source' if lang == 'en' else '从当前真实数据加载 SOFA-1/SOFA-2')}</div>
                    <div class="viz-demo-load-subtitle">{html.escape('Uses the validated sidebar database path and prepares paired SOFA concepts for this sensitivity panel.' if lang == 'en' else '使用侧边栏已验证的数据路径，并为本敏感性分析准备配对 SOFA 特征。')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                max_stays = st.number_input(
                    "Max stays to load" if lang == 'en' else "最大加载 stay 数",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100,
                    key="real_sofa_reclass_max_stays",
                    help="Use a limited subset for interactive review; full extraction remains available in the export workflow."
                         if lang == 'en' else "交互式审阅建议先使用子集；完整提取可在导出流程中执行。",
                )
            with col_b:
                _compact_spacer(28)
                if st.button(
                    "Load real SOFA concepts" if lang == 'en' else "加载真实 SOFA 特征",
                    type="primary",
                    use_container_width=True,
                    key="real_sofa_reclass_load",
                ):
                    try:
                        from easyicu import load_concepts
                        from easyicu.patient_filter import PatientFilter

                        database = _default_real_database()
                        data_path = _default_real_data_root()
                        if not data_path or not Path(data_path).exists():
                            _render_reclass_notice(
                                "danger",
                                "Real SOFA load" if lang == 'en' else "真实 SOFA 加载",
                                "Validated path required" if lang == 'en' else "需要已验证路径",
                                "Please validate a real data path in the sidebar first."
                                if lang == 'en' else
                                "请先在侧边栏验证真实数据路径。",
                            )
                            return

                        concepts = [
                            'sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal',
                            'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
                            'death', 'los_icu',
                        ]
                        progress_bar = st.progress(0, text="Loading demographics..." if lang == 'en' else "正在加载人口统计...")
                        try:
                            demographics = PatientFilter(database=database, data_path=data_path, verbose=False)._load_demographics()
                            id_col = 'stay_id' if 'stay_id' in demographics.columns else 'patient_id'
                            patient_ids = demographics[id_col].dropna().astype(int).head(int(max_stays)).tolist()
                            progress_bar.progress(20, text=f"Loading {len(concepts)} concepts for {len(patient_ids):,} stays..."
                                                  if lang == 'en' else f"正在加载 {len(concepts)} 个特征 ({len(patient_ids):,} stays)...")
                        except Exception as e:
                            progress_bar.empty()
                            _render_reclass_notice(
                                "danger",
                                "Real SOFA load" if lang == 'en' else "真实 SOFA 加载",
                                "Demographics load failed" if lang == 'en' else "人口统计加载失败",
                                str(e),
                            )
                            return

                        try:
                            concept_df = load_concepts(
                                concepts=concepts,
                                database=database,
                                data_path=data_path,
                                patient_ids=patient_ids,
                                verbose=False,
                                **_get_sepsis_runtime_options(),
                            )
                            progress_bar.progress(80, text="Splitting concept frames..." if lang == 'en' else "正在拆分特征表...")
                        except Exception as e:
                            progress_bar.empty()
                            _render_reclass_notice(
                                "danger",
                                "Real SOFA load" if lang == 'en' else "真实 SOFA 加载",
                                "Concept extraction failed" if lang == 'en' else "概念提取失败",
                                str(e),
                            )
                            return

                        if concept_df is None or concept_df.empty:
                            progress_bar.empty()
                            _render_reclass_notice(
                                "warning",
                                "Real SOFA load" if lang == 'en' else "真实 SOFA 加载",
                                "No paired SOFA data returned" if lang == 'en' else "未返回配对 SOFA 数据",
                                "No paired SOFA data were returned for the selected stays."
                                if lang == 'en' else
                                "所选 stay 未返回可用的配对 SOFA 数据。",
                            )
                            return

                        split_concepts = dict(st.session_state.get('loaded_concepts', {}) or {})
                        detected_id_col = next((col for col in ['stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid'] if col in concept_df.columns), None)
                        time_cols = [col for col in ['charttime', 'time'] if col in concept_df.columns]
                        base_cols = ([detected_id_col] if detected_id_col else []) + time_cols
                        loaded_ok: list[str] = []
                        failed_concepts: list[str] = []
                        for concept in concepts:
                            if concept not in concept_df.columns:
                                failed_concepts.append(concept)
                                continue
                            keep_cols = base_cols + [concept]
                            split_concepts[concept] = concept_df[keep_cols].dropna(subset=[concept]).copy()
                            loaded_ok.append(concept)

                        st.session_state['loaded_concepts'] = split_concepts
                        st.session_state['loaded_data_origin'] = 'real_sofa_reclassification'
                        progress_bar.progress(100, text="Done!" if lang == 'en' else "完成！")

                        loaded_meta = ", ".join(loaded_ok[:10])
                        if len(loaded_ok) > 10:
                            loaded_meta += f" +{len(loaded_ok) - 10}"
                        _render_reclass_notice(
                            "ready",
                            "Real SOFA load" if lang == 'en' else "真实 SOFA 加载",
                            "SOFA concepts loaded" if lang == 'en' else "SOFA 特征已加载",
                            f"Loaded {len(loaded_ok)} concept frames for {len(patient_ids):,} stays."
                            if lang == 'en' else
                            f"已为 {len(patient_ids):,} 个 stay 加载 {len(loaded_ok)} 个特征表。",
                            loaded_meta,
                        )
                        if failed_concepts:
                            _render_reclass_notice(
                                "warning",
                                "Real SOFA load" if lang == 'en' else "真实 SOFA 加载",
                                f"{len(failed_concepts)} concepts missing"
                                if lang == 'en' else
                                f"{len(failed_concepts)} 个概念缺失",
                                "Not found in extraction result."
                                if lang == 'en' else
                                "这些概念未出现在提取结果中。",
                                ", ".join(failed_concepts[:10]),
                            )
                        st.rerun()
                    except Exception as e:
                        _render_reclass_notice(
                            "danger",
                            "Real SOFA load" if lang == 'en' else "真实 SOFA 加载",
                            "SOFA concept load failed" if lang == 'en' else "SOFA 特征加载失败",
                            str(e),
                        )
                        return
            return

        if mode != 'worst_icu':
            _render_reclass_notice(
                "warning",
                "Definition gate" if lang == 'en' else "定义门禁",
                "Definition not available" if lang == 'en' else "当前口径不可用",
                "This definition is not available for the current session yet. Load real time-series concepts: `sofa`, `sofa2`, and optionally organ components in Quick Visualization first."
                if lang == 'en' else
                "当前会话还不能使用这个口径。请先在快速可视化中载入真实时间序列特征：`sofa`、`sofa2`，以及可选的器官组成分。",
            )
            return
        _render_reclass_notice(
            "pending",
            "Demo source" if lang == 'en' else "演示来源",
            "Generate SOFA-1 vs SOFA-2 Demo" if lang == 'en' else "生成 SOFA-1 vs SOFA-2 演示",
            "Create a patient-level paired SOFA-1/SOFA-2 cohort to inspect definition-driven reclassification." if lang == 'en' else "生成患者级SOFA-1/SOFA-2配对队列，用于查看定义差异导致的重新分层。",
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "Generate Reclassification Demo" if lang == 'en' else "生成重新分层演示",
                type="primary",
                use_container_width=True,
                key="reclass_generate_demo_btn",
            ):
                st.session_state['reclass_demo_df'] = _generate_mock_cohort_dashboard_data(lang)
                st.rerun()
        _render_reclass_notice(
            "info",
            "Source options" if lang == 'en' else "来源选项",
            "No paired SOFA source loaded yet" if lang == 'en' else "尚未加载配对 SOFA 来源",
            "Use Cohort Snapshot demo data, loaded Quick Visualization concepts, or generate demo data here."
            if lang == 'en' else
            "可使用队列快照演示数据、快速可视化已载入特征，或在此生成演示数据。",
        )
        return

    reclass = _build_sofa_reclassification_stats(source_df, lang=lang)
    if not reclass.get('available'):
        _render_reclass_notice(
            "warning",
            "Source check" if lang == 'en' else "来源检查",
            "Paired SOFA columns missing" if lang == 'en' else "缺少配对 SOFA 列",
            "The selected dataset does not contain paired SOFA-1 and SOFA-2 columns."
            if lang == 'en' else
            "当前数据不包含配对的 SOFA-1 和 SOFA-2 列。",
        )
        return

    if not screenshot_mode:
        _render_reclass_notice(
            "info",
            "Analysis source" if lang == 'en' else "分析来源",
            "Paired SOFA source selected" if lang == 'en' else "已选择配对 SOFA 来源",
            source_label,
        )
    _render_reclassification_cards(reclass, lang)
    if reclass['metrics'].get('denominator_label') == "Paired points":
        st.caption(
            f"Patient coverage: {reclass['metrics'].get('patient_count', '0')} unique stays represented in the paired time points."
            if lang == 'en' else
            f"患者覆盖：这些配对时间点来自 {reclass['metrics'].get('patient_count', '0')} 个唯一 ICU stay。"
        )
        st.caption(
            "Outcome rates in this mode are row-weighted by paired time points; use the first-24h or worst-score modes for patient-level outcome interpretation."
            if lang == 'en' else
            "此模式下的结局率按配对时间点加权；若要解释患者级结局，请使用首24小时或最高分模式。"
        )
    _render_compact_divider()

    rows = reclass['rows']
    summary = reclass['summary']
    matrix = reclass['matrix']
    organ = reclass['organ']
    unit_label = reclass['metrics'].get('denominator_label', "Patients" if lang == 'en' else "患者数")

    col1, col2 = st.columns([1, 1.06], gap="large")
    with col1:
        _render_chart_heading(
            "Reclassification matrix" if lang == 'en' else "重新分层矩阵",
            "Rows are SOFA-1 groups; columns are SOFA-2 groups." if lang == 'en' else "行表示 SOFA-1 分层，列表示 SOFA-2 分层。",
            "Agreement" if lang == 'en' else "一致性",
        )
        heatmap = matrix.pivot(index='SOFA-1', columns='SOFA-2', values='patients').fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns.astype(str),
            y=heatmap.index.astype(str),
            colorscale=[
                [0.0, RECLASS_CHART["plot"]],
                [0.32, RECLASS_CHART["teal_soft"]],
                [0.72, RECLASS_CHART["teal_mid"]],
                [1.0, RECLASS_CHART["teal"]],
            ],
            text=heatmap.values.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=12, color=RECLASS_CHART["ink"]),
            hovertemplate=f"SOFA-1: %{{y}}<br>SOFA-2: %{{x}}<br>{unit_label}: %{{z}}<extra></extra>",
            colorbar=dict(
                title=dict(
                    text=unit_label,
                    font=dict(size=12, color=RECLASS_CHART["ink"]),
                ),
                thickness=10,
                len=0.76,
                outlinewidth=0,
                tickfont=dict(size=11, color=RECLASS_CHART["muted"]),
            ),
        ))
        _style_reclass_figure(fig, height=410, margin=dict(l=72, r=58, t=24, b=64))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="SOFA-2 group" if lang == 'en' else "SOFA-2分层")
        fig.update_yaxes(title_text="SOFA-1 group" if lang == 'en' else "SOFA-1分层")
        st.plotly_chart(fig, use_container_width=True, key="reclass_matrix", config=_get_plotly_chart_config())

    with col2:
        _render_chart_heading(
            "Outcome by reclassification group" if lang == 'en' else "重新分层组别与结局",
            "Patient counts and mortality for up-classified, unchanged, and down-classified groups." if lang == 'en' else "展示上调、不变、下调分层患者数及死亡率。",
            "Outcome" if lang == 'en' else "结局",
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=summary['group'],
                y=summary['patients'],
                name=unit_label,
                marker=dict(
                    color=RECLASS_CHART["teal_soft"],
                    line=dict(color=RECLASS_CHART["teal_line"], width=1),
                ),
                text=summary['patients'],
                textposition='outside',
                cliponaxis=False,
                textfont=dict(size=12, color=RECLASS_CHART["ink"]),
                hovertemplate=f"%{{x}}<br>{unit_label}: %{{y}}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=summary['group'],
                y=summary['mortality'],
                name="Mortality %" if lang == 'en' else "死亡率 %",
                mode='lines+markers+text',
                text=summary['mortality'].map(lambda x: f"{x:.1f}%"),
                textposition='top center',
                marker=dict(color=RECLASS_CHART["rose"], size=7),
                line=dict(width=2.4, color=RECLASS_CHART["rose"]),
                cliponaxis=False,
                textfont=dict(size=12, color=RECLASS_CHART["rose"]),
                hovertemplate='%{x}<br>%{y:.1f}% mortality<extra></extra>' if lang == 'en' else '%{x}<br>%{y:.1f}% 死亡率<extra></extra>',
            ),
            secondary_y=True,
        )
        _style_reclass_figure(
            fig,
            height=410,
            margin=dict(l=58, r=72, t=54, b=74),
            legend_y=1.14,
        )
        fig.update_yaxes(
            title_text=unit_label,
            secondary_y=False,
            range=[0, max(5, float(summary['patients'].max()) * 1.30)],
        )
        fig.update_yaxes(
            title_text="Mortality %" if lang == 'en' else "死亡率 %",
            secondary_y=True,
            range=[0, 100],
            showgrid=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="reclass_outcome", config=_get_plotly_chart_config())

    st.markdown('<div class="eu-chart-row-gap"></div>', unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1.06], gap="large")
    with col3:
        _render_chart_heading(
            "SOFA-2 minus SOFA-1" if lang == 'en' else "SOFA-2 减 SOFA-1",
            "Distribution of paired score differences; zero means unchanged severity band." if lang == 'en' else "配对评分差值分布；零表示严重程度分层未改变。",
            "Delta" if lang == 'en' else "变化",
        )
        delta_counts = (
            rows.groupby(['delta', 'group'], dropna=False)
            .size()
            .reset_index(name='count')
            .sort_values(['delta', 'group'])
        )
        fig = go.Figure()
        color_map = {
            'Up-classified': RECLASS_CHART["rose"],
            'Same': RECLASS_CHART["slate"],
            'Down-classified': RECLASS_CHART["teal"],
            '上调分层': RECLASS_CHART["rose"],
            '不变': RECLASS_CHART["slate"],
            '下调分层': RECLASS_CHART["teal"],
        }
        for group_name in [name for name in color_map if name in set(delta_counts['group'])]:
            group_df = delta_counts[delta_counts['group'] == group_name]
            fig.add_trace(
                go.Bar(
                    x=group_df['delta'],
                    y=group_df['count'],
                    name=group_name,
                    marker=dict(color=color_map[group_name], line=dict(color='rgba(29,41,53,0.12)', width=1)),
                    hovertemplate=f"Delta: %{{x}}<br>{unit_label}: %{{y}}<extra></extra>",
                )
            )
        fig.add_vline(x=0, line_color=RECLASS_CHART["ink"], line_dash='dash', opacity=0.72)
        _style_reclass_figure(
            fig,
            height=360,
            margin=dict(l=58, r=34, t=54, b=58),
            legend_y=1.14,
        )
        fig.update_layout(barmode='stack', bargap=0.14, legend_title_text="")
        fig.update_xaxes(title_text="SOFA delta" if lang == 'en' else "SOFA差值")
        fig.update_yaxes(title_text=unit_label)
        st.plotly_chart(fig, use_container_width=True, key="reclass_delta_hist", config=_get_plotly_chart_config())

    with col4:
        _render_chart_heading(
            "Organ contributors" if lang == 'en' else "器官评分贡献",
            "Average absolute contribution of each SOFA component to the score difference." if lang == 'en' else "各 SOFA 器官组成分对评分差值的平均绝对贡献。",
            "Components" if lang == 'en' else "组成分",
        )
        if not organ.empty:
            organ_plot = organ.sort_values('mean_abs_delta', ascending=True)
            colors = [
                RECLASS_CHART["rose"] if val > 0 else RECLASS_CHART["teal"] if val < 0 else RECLASS_CHART["slate"]
                for val in organ_plot['mean_delta']
            ]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=organ_plot['mean_abs_delta'],
                    y=organ_plot['organ'],
                    orientation='h',
                    text=organ_plot['mean_abs_delta'].map(lambda x: f"{x:.2f}"),
                    textposition='outside',
                    cliponaxis=False,
                    textfont=dict(size=12, color=RECLASS_CHART["ink"]),
                    marker=dict(color=colors, line=dict(color='rgba(29,41,53,0.12)', width=1)),
                    hovertemplate='%{y}<br>Mean |delta|: %{x:.2f}<extra></extra>' if lang == 'en' else '%{y}<br>平均|差值|: %{x:.2f}<extra></extra>',
                    name="Mean |delta|" if lang == 'en' else "平均|差值|",
                )
            )
            _style_reclass_figure(
                fig,
                height=360,
                margin=dict(l=118, r=76, t=24, b=58),
            )
            fig.update_layout(showlegend=False)
            fig.update_xaxes(
                title_text="Mean |delta|" if lang == 'en' else "平均|差值|",
                range=[0, max(0.2, float(organ_plot['mean_abs_delta'].max()) * 1.28)],
            )
            fig.update_yaxes(title_text="")
            st.plotly_chart(fig, use_container_width=True, key="reclass_organ_contrib", config=_get_plotly_chart_config())
        else:
            _render_reclass_notice(
                "info",
                "Organ contributors" if lang == 'en' else "器官贡献",
                "Organ-level columns are unavailable" if lang == 'en' else "缺少器官级组成列",
                "Organ-level SOFA component columns are not available."
                if lang == 'en' else
                "当前数据没有器官级 SOFA 组成列。",
            )

    table_title = "Time-point reclassification table" if reclass['metrics'].get('denominator_label') == "Paired points" and lang == 'en' else (
        "时间点重新分层表" if reclass['metrics'].get('denominator_label') == "配对时间点" else ("Patient-level reclassification table" if lang == 'en' else "患者级重新分层表")
    )
    with st.expander(table_title, expanded=False):
        table_cols = ['stay_id']
        if 'charttime' in rows.columns:
            table_cols.append('charttime')
        table_cols.extend(['sofa1', 'sofa2', 'delta', 'SOFA-1', 'SOFA-2', 'group'])
        table = rows[table_cols].copy()
        _dataframe_compat(table, width="stretch", hide_index=True)
