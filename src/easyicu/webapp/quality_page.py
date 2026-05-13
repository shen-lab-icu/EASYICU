"""Data-quality page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_quality_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_quality_page(app_context: dict[str, Any] | None = None):
    """渲染数据质量页面。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')
    screenshot_mode = _is_screenshot_mode()

    page_title = "Data Quality" if lang == 'en' else "数据质量评估"
    page_sub = "Missing rate analysis, coverage badges & explainable causes" if lang == 'en' else "缺失率分析、覆盖度标识与可解释原因"
    st.markdown(f'''
    <div style="margin-bottom:20px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{page_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{page_sub}</div>
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
        _no_data_msg = "Load data to begin quality analysis." if lang == 'en' else "请先加载数据以进行质量分析。"
        _tip_msg = 'Try "Demo Mode" for a quick start.' if lang == 'en' else '选择「演示模式」快速开始。'
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">📊</div>
            <div style="font-weight:600;color:#111827;margin-bottom:4px">{_no_data_msg}</div>
            <div style="font-size:.85rem;color:#9ca3af">{_tip_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    mock_params = st.session_state.get('mock_params', {}) or {}
    demo_hours = int(mock_params.get('hours') or 0) if st.session_state.get('entry_mode') == 'demo' and mock_params.get('hours') else None
    time_grid_size = demo_hours or 72
    id_col = st.session_state.get('id_col', 'stay_id')
    total_patients_in_session = _get_quality_cohort_patient_count(st.session_state)
    cohort_patient_ids = _get_quality_cohort_patient_ids(st.session_state)
    los_by_patient = _get_quality_los_by_patient(st.session_state)

    records_col = "Records" if lang == 'en' else "记录数"
    patients_col = "Patients" if lang == 'en' else "患者数"
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
            return "#10b981"
        if value < 20:
            return "#f59e0b"
        return "#ef4444"

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

    detail_title = "Detailed QC Report" if lang == 'en' else "详细质控报告"
    denom_caption = (
        "Missingness denominator tags: d=LOS uses patient-specific ICU stay, d=72h uses the fallback window, d=demo uses the demo horizon, d=static means one observation per patient."
        if lang == 'en'
        else "缺失率分母说明：d=LOS 表示按患者 ICU 住院时长估算，d=72h 表示 72 小时兜底窗口，d=demo 表示演示数据预设时间窗，d=static 表示每位患者一次静态观测。"
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

    tab1_label = "📊 Missingness" if lang == 'en' else "📊 缺失分析"
    tab2_label = "🧪 Out-of-Physio" if lang == 'en' else "🧪 生理范围越界"
    tab3_label = "⏱️ Temporal Integrity" if lang == 'en' else "⏱️ 时序完整性"
    tab1, tab2, tab3 = st.tabs([tab1_label, tab2_label, tab3_label])

    with tab1:
        if screenshot_mode:
            sort_order = 'desc'
        else:
            sort_label = "Sort by" if lang == 'en' else "排序方式"
            if 'missing_chart_sort_order' not in st.session_state:
                st.session_state['missing_chart_sort_order'] = 'desc'
            sort_options = {
                'desc': '📉 Missing Rate (High → Low)' if lang == 'en' else '📉 缺失率 (从高到低)',
                'asc': '📈 Missing Rate (Low → High)' if lang == 'en' else '📈 缺失率 (从低到高)',
                'alpha': '🔤 Alphabetical (A → Z)' if lang == 'en' else '🔤 首字母排序 (A → Z)',
            }
            sort_order = st.radio(
                sort_label,
                options=list(sort_options.keys()),
                format_func=lambda x: sort_options[x],
                horizontal=True,
                key='missing_chart_sort_order',
            )

        if quality_df.empty:
            st.info("No quality metrics available." if lang == 'en' else "当前没有可用的质量指标。")
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
                good_msg = "✅ Missingness is negligible across loaded concepts." if lang == 'en' else "✅ 当前已加载概念几乎没有缺失。"
                st.success(good_msg)
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
                        '< 25': '#f59e0b',
                        '25–50': '#fb923c',
                        '50–75': '#f97316',
                        '75–100': '#ef4444',
                    },
                    hover_data=[records_col, patients_col, denom_hover],
                    title='Missingness by concept' if lang == 'en' else '各概念缺失率',
                )
                fig.update_layout(
                    template="plotly_white",
                    height=max(340, len(chart_df) * 34 + 110),
                    showlegend=True,
                    legend=dict(
                        title='Missing rate (%)' if lang == 'en' else '缺失率 (%)',
                        orientation='v',
                        x=1.02,
                        y=0.72,
                        bgcolor='rgba(255,255,255,0.92)',
                        bordercolor='#dbeafe',
                        borderwidth=1,
                        font=dict(size=12, color='#0b1f44'),
                    ),
                    yaxis_title="",
                    yaxis=dict(autorange='reversed'),
                    margin=dict(l=92, r=160, t=44, b=44),
                    font=dict(size=12, color='#0b1f44'),
                    xaxis=dict(range=[0, 100], title=missing_rate_label, gridcolor='#e8eef7'),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                )
                if total_quality_concepts > len(chart_df):
                    fig.add_annotation(
                        xref='paper', yref='paper', x=1.0, y=1.08,
                        text=(
                            f"Showing {len(chart_df)} of {total_quality_concepts}"
                            if lang == 'en'
                            else f"显示 {len(chart_df)} / {total_quality_concepts}"
                        ),
                        showarrow=False,
                        font=dict(size=11, color='#60718a'),
                        align='right',
                    )
                st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

    with tab2:
        if quality_df.empty:
            st.info("No quality metrics available." if lang == 'en' else "当前没有可用的质量指标。")
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
                st.success("✅ No loaded concept currently exceeds the physiologic QC ranges." if lang == 'en' else "✅ 当前已加载概念没有明显越出生理范围的值。")
            else:
                fig = px.bar(
                    outlier_df,
                    x=outlier_rate_label,
                    y='Concept',
                    orientation='h',
                    color=outlier_rate_label,
                    color_continuous_scale=['#dbeafe', '#f59e0b', '#dc2626'],
                    hover_data=[records_col],
                    title='🧪 Physiologic Range QC' if lang == 'en' else '🧪 生理范围质控',
                )
                fig.update_layout(
                    template='plotly_white',
                    height=max(320, len(outlier_df) * 36),
                    showlegend=False,
                    yaxis_title='',
                    yaxis=dict(autorange='reversed'),
                    margin=dict(l=90, r=20, t=44, b=40),
                    font=dict(size=13, color='#111827'),
                )
                st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

    with tab3:
        if quality_df.empty:
            st.info("No quality metrics available." if lang == 'en' else "当前没有可用的质量指标。")
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
                    st.success("✅ No duplicate patient-time rows were detected in the loaded concepts." if lang == 'en' else "✅ 当前已加载概念未检测到重复的患者-时间行。")
                else:
                    fig_dup = px.bar(
                        duplicate_df,
                        x=duplicate_rate_label,
                        y='Concept',
                        orientation='h',
                        color=duplicate_rate_label,
                        color_continuous_scale=['#bfdbfe', '#f59e0b', '#dc2626'],
                        hover_data=[records_col],
                        title='🧬 Duplicate Timestamp Rate' if lang == 'en' else '🧬 重复时间戳比例',
                    )
                    fig_dup.update_layout(
                        template='plotly_white',
                        height=max(320, len(duplicate_df) * 34),
                        showlegend=False,
                        yaxis_title='',
                        yaxis=dict(autorange='reversed'),
                        margin=dict(l=90, r=20, t=44, b=40),
                        font=dict(size=13, color='#111827'),
                    )
                    st.plotly_chart(fig_dup, use_container_width=True, config=_get_plotly_chart_config())

            with temporal_cols[1]:
                density_df = quality_df[['Concept', '_density_median', '_density_q25', '_density_q75', '_missing_rate', '_dup_rate', '_denominator_tag']].copy()
                density_label = "Median records / patient / hour" if lang == 'en' else "中位 records / patient / hour"
                missing_label = "Missing Rate (%)" if lang == 'en' else "缺失率 (%)"
                dup_label = "Duplicate TS (%)" if lang == 'en' else "重复时间戳 (%)"
                iqr_label = "IQR" if lang == 'en' else "IQR"

                density_df = density_df[density_df['_density_median'] > 0].copy()
                if density_df.empty:
                    st.info("Density summaries need time-stamped concepts." if lang == 'en' else "密度摘要需要带时间戳的概念。")
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
                            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
                            labels={
                                '_density_median': density_label,
                                '_dup_rate': dup_label,
                                '_missing_rate': missing_label,
                                '_iqr_text': iqr_label,
                            },
                            title='⏱️ Temporal Density vs Duplicate Rate' if lang == 'en' else '⏱️ 时序密度与重复率',
                        )
                        fig_density.update_traces(textposition='top center', textfont=dict(size=11))
                        fig_density.update_layout(
                            template='plotly_white',
                            height=420,
                            margin=dict(l=30, r=20, t=44, b=40),
                            font=dict(size=13, color='#111827'),
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
                            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
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
                                f"⏱️ Top {len(ranked)} concepts by density"
                                if lang == 'en'
                                else f"⏱️ 密度排名前 {len(ranked)} 的概念"
                            ),
                        )
                        fig_density.update_layout(
                            template='plotly_white',
                            height=max(320, len(ranked) * 22),
                            margin=dict(l=90, r=20, t=44, b=40),
                            font=dict(size=12, color='#111827'),
                            yaxis_title='',
                            showlegend=False,
                        )
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
        with st.expander(f"📋 {detail_title}", expanded=False):
            _render_quality_detail_report()
