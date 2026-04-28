"""Cohort severity-reclassification subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_severity_reclassification_subtab', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_severity_reclassification_subtab(lang: str, app_context: dict[str, Any] | None = None):
    """Render cohort-level SOFA-1 to SOFA-2 severity reclassification analysis."""
    if app_context is not None:
        _install_app_context(app_context)
    
    import plotly.express as px
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
        st.markdown(f"### 🧭 {title}")
        st.caption(subtitle)

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
        st.caption(
            (
                f"Load `sofa` and `sofa2` in Quick Visualization to unlock: {locked_text}."
                if lang == 'en' else
                f"先在快速可视化中载入 `sofa` 和 `sofa2`，即可解锁：{locked_text}。"
            )
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
                    "🚀 " + ("Load real SOFA concepts" if lang == 'en' else "加载真实 SOFA 特征"),
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
                            st.error("Please validate a real data path in the sidebar first." if lang == 'en' else "请先在侧边栏验证真实数据路径。")
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
                            st.error(f"Failed to load demographics: {e}")
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
                            st.error(f"Concept extraction failed: {e}")
                            return

                        if concept_df is None or concept_df.empty:
                            progress_bar.empty()
                            st.warning("No paired SOFA data were returned for the selected stays." if lang == 'en' else "所选 stay 未返回可用的配对 SOFA 数据。")
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

                        # Show per-concept results
                        st.success(
                            f"Loaded {len(loaded_ok)} concept frames for {len(patient_ids):,} stays."
                            if lang == 'en' else
                            f"已为 {len(patient_ids):,} 个 stay 加载 {len(loaded_ok)} 个特征表。"
                        )
                        if loaded_ok:
                            st.caption("✅ " + ", ".join(f"`{c}`" for c in loaded_ok))
                        if failed_concepts:
                            st.warning(
                                f"⚠️ {len(failed_concepts)} concepts not found in extraction result: "
                                + ", ".join(f"`{c}`" for c in failed_concepts)
                                if lang == 'en' else
                                f"⚠️ {len(failed_concepts)} 个概念在提取结果中缺失："
                                + ", ".join(f"`{c}`" for c in failed_concepts)
                            )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading real SOFA concepts: {e}")
                        return
            return

        if mode != 'worst_icu':
            st.warning(
                "This definition is not available for the current session yet. Load real time-series concepts: `sofa`, `sofa2`, and optionally organ components in Quick Visualization first."
                if lang == 'en' else
                "当前会话还不能使用这个口径。请先在快速可视化中载入真实时间序列特征：`sofa`、`sofa2`，以及可选的器官组成分。"
            )
            return
        _render_demo_generation_card(
            "🧭",
            "Generate SOFA-1 vs SOFA-2 Demo" if lang == 'en' else "生成 SOFA-1 vs SOFA-2 演示",
            "Create a patient-level paired SOFA-1/SOFA-2 cohort to inspect definition-driven reclassification." if lang == 'en' else "生成患者级SOFA-1/SOFA-2配对队列，用于查看定义差异导致的重新分层。",
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 " + ("Generate Reclassification Demo" if lang == 'en' else "生成重新分层演示"),
                type="primary",
                use_container_width=True,
                key="reclass_generate_demo_btn",
            ):
                st.session_state['reclass_demo_df'] = _generate_mock_cohort_dashboard_data(lang)
                st.rerun()
        st.info("Use Cohort Snapshot demo data, loaded Quick Visualization concepts, or generate demo data here." if lang == 'en' else "可使用队列快照演示数据、快速可视化已载入特征，或在此生成演示数据。")
        return

    reclass = _build_sofa_reclassification_stats(source_df, lang=lang)
    if not reclass.get('available'):
        st.warning("The selected dataset does not contain paired SOFA-1 and SOFA-2 columns." if lang == 'en' else "当前数据不包含配对的SOFA-1和SOFA-2列。")
        return

    if not screenshot_mode:
        st.info(("Source: " if lang == 'en' else "数据来源：") + source_label)
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

    col1, col2 = st.columns([1, 1.1])
    with col1:
        st.markdown("##### " + ("Reclassification Matrix" if lang == 'en' else "重新分层矩阵"))
        heatmap = matrix.pivot(index='SOFA-1', columns='SOFA-2', values='patients').fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns.astype(str),
            y=heatmap.index.astype(str),
            colorscale='Blues',
            text=heatmap.values.astype(int),
            texttemplate="%{text}",
            hovertemplate=f"SOFA-1: %{{y}}<br>SOFA-2: %{{x}}<br>{unit_label}: %{{z}}<extra></extra>",
        ))
        fig.update_layout(
            template='plotly_white',
            height=360,
            margin=dict(l=45, r=20, t=12, b=45),
            xaxis_title="SOFA-2 group" if lang == 'en' else "SOFA-2分层",
            yaxis_title="SOFA-1 group" if lang == 'en' else "SOFA-1分层",
            font=dict(size=13, color='#111827'),
        )
        st.plotly_chart(fig, use_container_width=True, key="reclass_matrix", config=_get_plotly_chart_config())

    with col2:
        st.markdown("##### " + ("Outcome by Reclassification Group" if lang == 'en' else "重新分层组别与结局"))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=summary['group'],
                y=summary['patients'],
                name=unit_label,
                marker_color='rgba(37, 99, 235, 0.58)',
                text=summary['patients'],
                textposition='outside',
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
                marker_color='#e11d48',
                line=dict(width=3),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            template='plotly_white',
            height=360,
            margin=dict(l=20, r=20, t=12, b=60),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            font=dict(size=13, color='#111827'),
        )
        fig.update_yaxes(title_text=unit_label, secondary_y=False, gridcolor='#e5e7eb')
        fig.update_yaxes(title_text="Mortality %" if lang == 'en' else "死亡率 %", secondary_y=True, range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, key="reclass_outcome", config=_get_plotly_chart_config())

    col3, col4 = st.columns([1, 1.1])
    with col3:
        st.markdown("##### " + ("SOFA-2 minus SOFA-1" if lang == 'en' else "SOFA-2 减 SOFA-1"))
        fig = px.histogram(
            rows,
            x='delta',
            color='group',
            nbins=17,
            color_discrete_map={
                'Up-classified': '#e11d48',
                'Same': '#64748b',
                'Down-classified': '#0f766e',
                '上调分层': '#e11d48',
                '不变': '#64748b',
                '下调分层': '#0f766e',
            },
            labels={'delta': "SOFA delta" if lang == 'en' else "SOFA差值", 'count': unit_label},
            template='plotly_white',
        )
        fig.add_vline(x=0, line_color='#111827', line_dash='dash')
        fig.update_layout(height=330, margin=dict(l=20, r=20, t=12, b=40), font=dict(size=13, color='#111827'), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, key="reclass_delta_hist", config=_get_plotly_chart_config())

    with col4:
        st.markdown("##### " + ("Organ Contributors" if lang == 'en' else "器官评分贡献"))
        if not organ.empty:
            fig = px.bar(
                organ,
                x='mean_abs_delta',
                y='organ',
                orientation='h',
                text='mean_abs_delta',
                color='mean_delta',
                color_continuous_scale=['#0f766e', '#f8fafc', '#e11d48'],
                labels={'mean_abs_delta': "Mean |delta|" if lang == 'en' else "平均|差值|", 'organ': "", 'mean_delta': "Mean delta" if lang == 'en' else "平均差值"},
                template='plotly_white',
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition='outside', cliponaxis=False)
            fig.update_layout(height=330, margin=dict(l=10, r=45, t=12, b=40), font=dict(size=13, color='#111827'))
            fig.update_xaxes(range=[0, max(0.2, float(organ['mean_abs_delta'].max()) * 1.25)], gridcolor='#e5e7eb')
            st.plotly_chart(fig, use_container_width=True, key="reclass_organ_contrib", config=_get_plotly_chart_config())
        else:
            st.info("Organ-level SOFA component columns are not available." if lang == 'en' else "当前数据没有器官级SOFA组成列。")

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
