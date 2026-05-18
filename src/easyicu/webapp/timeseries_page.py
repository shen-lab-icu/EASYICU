"""Time-series page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat
from easyicu.webapp.ui_helpers import StatCard, render_stat_grid


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_timeseries_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_timeseries_page(app_context: dict[str, Any] | None = None):
    """渲染时序分析页面。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')
    screenshot_mode = _is_screenshot_mode()

    _ts_title = "Time Series Analysis" if lang == 'en' else "时序数据分析"
    _ts_sub = "Interactive visualization, single & multi-patient comparison" if lang == 'en' else "交互式可视化，支持单/多患者对比"
    _hdr_col1, _hdr_col2 = st.columns([3, 1])
    with _hdr_col1:
        st.markdown(f'''
        <div style="margin-bottom:16px">
            <div style="font-size:1.4rem;font-weight:800;color:#111827">{_ts_title}</div>
            <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_ts_sub}</div>
        </div>
        ''', unsafe_allow_html=True)
    with _hdr_col2:
        if screenshot_mode:
            st.session_state['_ts_show_thresholds'] = True
        else:
            _show_thresh = st.toggle(get_text('threshold_lines'), value=True, key="ts_show_thresholds")
            st.session_state['_ts_show_thresholds'] = _show_thresh

    if screenshot_mode:
        focus_hint = (
            "Figure preset: showing up to 4 representative trajectories and hiding chart toolbars."
            if lang == 'en'
            else "截图预设：默认仅展示最多 4 条代表性轨迹，并隐藏图表工具条。"
        )
        st.markdown(f'<div class="compact-inline-notice info">{focus_hint}</div>', unsafe_allow_html=True)

    if len(st.session_state.loaded_concepts) == 0:
        _msg = "Load data to begin time series analysis." if lang == 'en' else "请先加载数据以进行时序分析。"
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">📈</div>
            <div style="font-weight:600;color:#111827">{_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    # Concept 选择区域
    available_concepts = list(st.session_state.loaded_concepts.keys())

    # 分析模式选择 — 新增 Clinical Lanes 模式
    mode_label = "Analysis Mode" if lang == 'en' else "分析模式"
    mode_lanes = get_text('clinical_lanes')
    mode_single = "Single Patient" if lang == 'en' else "单患者分析"
    mode_multi = "Multi-Patient Comparison" if lang == 'en' else "多患者比较"
    if screenshot_mode:
        analysis_mode = mode_lanes
    else:
        analysis_mode = st.radio(
            mode_label,
            options=[mode_lanes, mode_single, mode_multi],
            horizontal=True,
            key="ts_mode"
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ============ Clinical Lanes View (默认) ============
    if analysis_mode == mode_lanes:
        import plotly.graph_objects as go

        if not st.session_state.patient_ids:
            st.warning("No patient data" if lang == 'en' else "无患者数据")
        else:
            if screenshot_mode:
                _lane_pid = st.session_state.get('lane_patient_select') or st.session_state.patient_ids[0]
            else:
                _lane_pid = _patient_selector(
                    patient_ids=st.session_state.patient_ids,
                    state_key="lane_patient_select",
                    label="Patient" if lang == 'en' else "患者",
                    lang=lang,
                    max_display=200,
                    default_patient=st.session_state.get('lane_patient_select', st.session_state.patient_ids[0]),
                )

            id_col = st.session_state.get('id_col', 'stay_id')
            _show_thresh = st.session_state.get('_ts_show_thresholds', True)
            screenshot_concepts = set(_select_timeseries_screenshot_concepts(available_concepts)) if screenshot_mode else None

            for lane_name, lane_concepts in CLINICAL_LANES.items():
                _lane_avail = [c for c in lane_concepts if c in available_concepts]
                if screenshot_mode and screenshot_concepts is not None:
                    _lane_avail = [c for c in _lane_avail if c in screenshot_concepts]
                if not _lane_avail:
                    continue

                lane_label = get_text(f'lane_{lane_name}')
                st.markdown(f"#### {lane_label}")

                _n_cols = min(len(_lane_avail), 3)
                _cols = st.columns(_n_cols)
                for idx, cname in enumerate(_lane_avail[:6]):
                    with _cols[idx % _n_cols]:
                        df = st.session_state.loaded_concepts.get(cname)
                        if df is None or not hasattr(df, 'columns'):
                            continue
                        if id_col not in df.columns or cname not in df.columns:
                            continue
                        pdf = df[df[id_col] == _lane_pid].copy()
                        if pdf.empty:
                            if not screenshot_mode:
                                st.caption(f"{cname}: no data")
                            continue

                        _tcol = None
                        for tc in ['charttime', 'time', 'hour', 'datetime', 'measuredat_minutes',
                                   'observationoffset', 'starttime']:
                            if tc in pdf.columns:
                                _tcol = tc
                                break

                        if _tcol is None:
                            if not screenshot_mode:
                                st.caption(f"{cname}: no time column")
                            continue

                        try:
                            plot_pdf = _prepare_timeseries_plot_df(pdf, _tcol, cname)
                            if plot_pdf.empty:
                                if not screenshot_mode:
                                    st.caption(f"{cname}: no valid time series")
                                continue
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=plot_pdf[_tcol], y=plot_pdf[cname],
                                mode='lines+markers', name=cname,
                                line=dict(width=1.5), marker=dict(size=3)
                            ))
                            fig = _add_clinical_thresholds(fig, cname, _show_thresh)

                            _unit = CLINICAL_THRESHOLDS.get(cname, {}).get('unit', '')
                            fig.update_layout(
                                title=dict(text=f"{cname}" + (f" ({_unit})" if _unit else ""), font=dict(size=12)),
                                height=200, margin=dict(l=40, r=10, t=30, b=25),
                                showlegend=False, xaxis=dict(title=""), yaxis=dict(title=""),
                            )
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"lane_{lane_name}_{cname}_{_lane_pid}",
                                config=_get_plotly_chart_config(),
                            )
                        except Exception:
                            if not screenshot_mode:
                                st.caption(f"{cname}: render error")

                st.markdown("---")
        return

    if analysis_mode == mode_single:
        # 顶部控制面板 - 🔧 FIX: 添加模块筛选，方便用户在100+特征中找到想要的
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 0.9])

        with col1:
            # 🔧 FIX: 先选择模块，再选择特征
            module_label = "📂 Select Module" if lang == 'en' else "📂 选择模块"
            all_modules_opt = "All Modules" if lang == 'en' else "全部模块"

            # 获取模块列表 - 🔧 FIX (2026-02-05): 只显示支持时序分析的模块
            module_options = [all_modules_opt]
            for grp_key in CONCEPT_GROUPS_INTERNAL:
                # 跳过不支持时序分析的模块（demographics, outcome）
                if grp_key not in TIME_SERIES_COMPATIBLE_MODULES:
                    continue
                grp_concepts = CONCEPT_GROUPS_INTERNAL[grp_key]
                # 检查该模块是否有已加载的概念
                if any(c in available_concepts for c in grp_concepts):
                    display_name = CONCEPT_GROUPS_DISPLAY.get(grp_key, grp_key)
                    module_options.append(display_name)

            selected_module = st.selectbox(
                module_label,
                options=module_options,
                key="ts_module"
            )

        with col2:
            # 根据选择的模块过滤概念
            if selected_module == all_modules_opt:
                filtered_concepts = available_concepts
            else:
                # 找到对应的 group_key
                selected_grp_key = None
                for grp_key, display in CONCEPT_GROUPS_DISPLAY.items():
                    if display == selected_module:
                        selected_grp_key = grp_key
                        break
                if selected_grp_key:
                    grp_concepts = CONCEPT_GROUPS_INTERNAL.get(selected_grp_key, [])
                    filtered_concepts = [c for c in available_concepts if c in grp_concepts]
                else:
                    filtered_concepts = available_concepts

            concept_label = "📋 Select Concept" if lang == 'en' else "📋 选择 Concept"
            concept_help = "Select data type to visualize" if lang == 'en' else "选择要可视化的数据类型"
            selected_concept = st.selectbox(
                concept_label,
                options=filtered_concepts if filtered_concepts else available_concepts,
                key="ts_concept",
                help=concept_help
            )

        with col3:
            if st.session_state.patient_ids:
                patient_label = "👤 Select Patient" if lang == 'en' else "👤 选择患者"
                patient_id = _patient_selector(
                    patient_ids=st.session_state.patient_ids,
                    state_key="ts_patient",
                    label=patient_label,
                    lang=lang,
                    max_display=200,
                    default_patient=st.session_state.get('ts_patient', st.session_state.patient_ids[0]),
                )
            else:
                patient_id = None
                no_patient_msg = "No patients found" if lang == 'en' else "未找到患者"
                st.warning(no_patient_msg)

        with col4:
            chart_label = "📊 Chart Type" if lang == 'en' else "📊 图表类型"
            line_opt = "Line Chart" if lang == 'en' else "折线图"
            scatter_opt = "Scatter Plot" if lang == 'en' else "散点图"
            area_opt = "Area Chart" if lang == 'en' else "面积图"
            chart_type = st.selectbox(
                chart_label,
                options=[line_opt, scatter_opt, area_opt],
                key="ts_chart_type"
            )

        with col5:
            value_label = "🧪 Value Column" if lang == 'en' else "🧪 数值列"
            concept_df = st.session_state.loaded_concepts.get(selected_concept)
            value_options = _get_concept_numeric_value_columns(concept_df)
            preferred_value_col = _choose_concept_value_column(selected_concept, concept_df) if isinstance(concept_df, pd.DataFrame) else None
            if len(value_options) > 1:
                default_index = value_options.index(preferred_value_col) if preferred_value_col in value_options else 0
                selected_value_col = st.selectbox(
                    value_label,
                    options=value_options,
                    index=default_index,
                    key="ts_value_column",
                )
            else:
                selected_value_col = preferred_value_col
                st.markdown(
                    f'<div class="inline-control-label">{value_label}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(preferred_value_col or ("No numeric column" if lang == 'en' else "无数值列"))

        with col6:
            show_stats_label = "Show Statistics" if lang == 'en' else "显示统计"
            show_stats = st.checkbox(show_stats_label, value=True, key="ts_show_stats")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 主图表区域
        if selected_concept and patient_id:
            df = st.session_state.loaded_concepts[selected_concept]

            # 确保是 DataFrame
            if not isinstance(df, pd.DataFrame):
                format_warn = f"Data format not supported: {type(df).__name__}" if lang == 'en' else f"数据格式不支持: {type(df).__name__}"
                st.warning(format_warn)
                return

            # 过滤数据
            id_col = st.session_state.id_col
            if id_col and id_col in df.columns:
                patient_df = df[df[id_col] == patient_id].copy()
            else:
                patient_df = df.copy()

            # 显示图表
            if len(patient_df) > 0:
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go

                    # 确定数值列
                    value_cols = _get_concept_numeric_value_columns(patient_df)

                    # 检测时间列 - 支持多种命名
                    time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat']
                    time_col = None
                    for tc in time_candidates:
                        if tc in patient_df.columns:
                            time_col = tc
                            break

                    if value_cols:
                        value_col = selected_value_col if selected_value_col in value_cols else _choose_concept_value_column(selected_concept, patient_df)

                        if time_col:
                            plot_df = _prepare_timeseries_plot_df(patient_df, time_col, value_col)
                            if plot_df.empty:
                                st.info("ℹ️ No valid time series after sorting/cleaning" if lang == 'en' else "ℹ️ 清理后没有可用的时序数据")
                                return
                            # 根据图表类型创建图表
                            line_type = "Line Chart" if lang == 'en' else "折线图"
                            scatter_type = "Scatter Plot" if lang == 'en' else "散点图"
                            patient_label = "Patient" if lang == 'en' else "患者"
                            chart_title = f"📈 {selected_concept.upper()} - {patient_label} {patient_id}"

                            if chart_type == line_type:
                                fig = px.line(
                                    plot_df, x=time_col, y=value_col,
                                    title=chart_title,
                                    markers=True
                                )
                            elif chart_type == scatter_type:
                                fig = px.scatter(
                                    plot_df, x=time_col, y=value_col,
                                    title=chart_title,
                                    size_max=10
                                )
                            else:  # 面积图
                                fig = px.area(
                                    plot_df, x=time_col, y=value_col,
                                    title=chart_title
                                )

                            # 美化图表
                            time_label = "Time (hours)" if lang == 'en' else "时间 (小时)"
                            fig.update_layout(
                                template="plotly_white",
                                hovermode="x unified",
                                xaxis_title=time_label,
                                yaxis_title=value_col.upper(),
                                font=dict(size=14, color='black'),
                                title_font_size=16,
                                showlegend=False,
                                height=450,
                                margin=dict(l=50, r=30, t=50, b=50),
                            )
                            fig.update_traces(
                                line=dict(width=2, color='#1f77b4'),
                                marker=dict(size=6)
                            )
                            fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))

                            st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())
                        else:
                            # 🔧 只有数值没有时间列（静态数据/单点数据）
                            st.info("ℹ️ Static value (No time series data)" if lang == 'en' else "ℹ️ 静态数值（无时间序列数据）")
                            if len(patient_df) == 1:
                                val = patient_df[value_col].iloc[0]
                                st.metric(label=value_col.upper(), value=f"{val}")
                            else:
                                _st_dataframe_compat(st, patient_df[[value_col]], width="stretch")

                        # 显示统计信息
                        if show_stats:
                            stat_title = "#### 📊 Statistical Summary" if lang == 'en' else "#### 📊 统计摘要"
                            st.markdown(stat_title)
                            values = patient_df[value_col]
                            if lang == 'en':
                                stats = [
                                    ("Min", f"{values.min():.2f}", "📉"),
                                    ("Max", f"{values.max():.2f}", "📈"),
                                    ("Mean", f"{values.mean():.2f}", "📊"),
                                    ("Std Dev", f"{values.std():.2f}", "📐"),
                                    ("Records", f"{len(values)}", "📝"),
                                ]
                            else:
                                stats = [
                                    ("最小值", f"{values.min():.2f}", "📉"),
                                    ("最大值", f"{values.max():.2f}", "📈"),
                                    ("平均值", f"{values.mean():.2f}", "📊"),
                                    ("标准差", f"{values.std():.2f}", "📐"),
                                    ("记录数", f"{len(values)}", "📝"),
                                ]
                            render_stat_grid(
                                [StatCard(label=f"{icon} {label}", value=value) for label, value, icon in stats],
                                columns=5,
                                compact=True,
                            )
                    else:
                        # 🔧 FIX: 检测是否有布尔列（包括pandas boolean和numpy bool）
                        bool_cols = []
                        for col in patient_df.columns:
                            dtype_str = str(patient_df[col].dtype).lower()
                            if 'bool' in dtype_str:
                                bool_cols.append(col)

                        if bool_cols:
                            if lang == 'en':
                                warn_msg = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart."
                            else:
                                warn_msg = f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                        else:
                            warn_msg = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart." if lang == 'en' else f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                        st.warning(warn_msg)
                        # 🔧 显示数据表格预览，将布尔列转换为字符串
                        display_patient_df = patient_df.head(20).copy()
                        for col in display_patient_df.columns:
                            dtype_str = str(display_patient_df[col].dtype).lower()
                            if 'bool' in dtype_str:
                                display_patient_df[col] = display_patient_df[col].astype(str)
                        st.dataframe(display_patient_df, use_container_width=True)

                except Exception as e:
                    err_msg = f"Chart rendering failed: {e}" if lang == 'en' else f"图表渲染失败: {e}"
                    st.warning(err_msg)
                    if 'time' in patient_df.columns:
                        chart_df = patient_df.set_index('time')
                        fallback_value_col = selected_value_col if selected_value_col in chart_df.columns else _choose_concept_value_column(selected_concept, chart_df.reset_index())
                        if fallback_value_col and fallback_value_col in chart_df.columns:
                            st.line_chart(chart_df[fallback_value_col])
            else:
                no_data_msg = f"ℹ️ No {selected_concept} data for patient {patient_id}" if lang == 'en' else f"ℹ️ 患者 {patient_id} 无 {selected_concept} 数据"
                st.info(no_data_msg)

        # 数据表格预览
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        preview_label = "📋 Data Table Preview" if lang == 'en' else "📋 数据表格预览"
        with st.expander(preview_label, expanded=True):  # 🔧 FIX: 默认展开
            if selected_concept in st.session_state.loaded_concepts:
                df = st.session_state.loaded_concepts[selected_concept]
                if isinstance(df, pd.DataFrame):
                    if patient_id:
                        id_col = st.session_state.id_col
                        if id_col in df.columns:
                            df = df[df[id_col] == patient_id]
                    _st_dataframe_compat(st, df.head(50), width="stretch", hide_index=True)
                else:
                    format_msg = "Data format does not support preview" if lang == 'en' else "数据格式不支持预览"
                    st.info(format_msg)

    else:  # 多患者比较模式
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 0.9])

        with col1:
            # 🔧 FIX: 先选择模块，再选择特征
            module_label = "📂 Select Module" if lang == 'en' else "📂 选择模块"
            all_modules_opt = "All Modules" if lang == 'en' else "全部模块"

            # 🔧 FIX (2026-02-05): 只显示支持时序分析的模块（排除静态数据模块）
            module_options = [all_modules_opt]
            for grp_key in CONCEPT_GROUPS_INTERNAL:
                # 跳过不支持时序分析的模块（demographics, outcome）
                if grp_key not in TIME_SERIES_COMPATIBLE_MODULES:
                    continue
                grp_concepts = CONCEPT_GROUPS_INTERNAL[grp_key]
                if any(c in available_concepts for c in grp_concepts):
                    display_name = CONCEPT_GROUPS_DISPLAY.get(grp_key, grp_key)
                    module_options.append(display_name)

            selected_module_multi = st.selectbox(
                module_label,
                options=module_options,
                key="ts_module_multi"
            )

        with col2:
            # 根据选择的模块过滤概念
            if selected_module_multi == all_modules_opt:
                filtered_concepts_multi = available_concepts
            else:
                selected_grp_key = None
                for grp_key, display in CONCEPT_GROUPS_DISPLAY.items():
                    if display == selected_module_multi:
                        selected_grp_key = grp_key
                        break
                if selected_grp_key:
                    grp_concepts = CONCEPT_GROUPS_INTERNAL.get(selected_grp_key, [])
                    filtered_concepts_multi = [c for c in available_concepts if c in grp_concepts]
                else:
                    filtered_concepts_multi = available_concepts

            concept_label = "📋 Select Concept" if lang == 'en' else "📋 选择 Concept"
            selected_concept = st.selectbox(
                concept_label,
                options=filtered_concepts_multi if filtered_concepts_multi else available_concepts,
                key="ts_concept_multi"
            )

        with col3:
            if st.session_state.patient_ids:
                compare_label = "👥 Select patients to compare (max 5)" if lang == 'en' else "👥 选择要比较的患者 (最多5个)"
                compare_search = st.text_input(
                    "🔍 Search Patient IDs" if lang == 'en' else "🔍 搜索患者ID",
                    key="ts_compare_search",
                    placeholder="Type to filter..." if lang == 'en' else "输入ID过滤...",
                )
                compare_options = _filter_patient_selector_options(
                    st.session_state.patient_ids,
                    query=compare_search,
                    max_display=200,
                )
                default_compare = [pid for pid in st.session_state.patient_ids[:3] if pid in compare_options]
                compare_patients = st.multiselect(
                    compare_label,
                    options=compare_options,
                    default=default_compare,
                    max_selections=5,
                    key="ts_compare_patients"
                )
            else:
                compare_patients = []

        with col4:
            compare_df = st.session_state.loaded_concepts.get(selected_concept)
            compare_value_options = _get_concept_numeric_value_columns(compare_df)
            compare_preferred_value = _choose_concept_value_column(selected_concept, compare_df) if isinstance(compare_df, pd.DataFrame) else None
            value_label = "🧪 Value Column" if lang == 'en' else "🧪 数值列"
            if len(compare_value_options) > 1:
                default_index = compare_value_options.index(compare_preferred_value) if compare_preferred_value in compare_value_options else 0
                compare_value_col = st.selectbox(
                    value_label,
                    options=compare_value_options,
                    index=default_index,
                    key="ts_compare_value_column",
                )
            else:
                compare_value_col = compare_preferred_value
                st.markdown(
                    f'<div class="inline-control-label">{value_label}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(compare_preferred_value or ("No numeric column" if lang == 'en' else "无数值列"))

        with col5:
            normalize_label = "Normalize" if lang == 'en' else "归一化比较"
            normalize_help = "Normalize values to 0-1 range for comparison" if lang == 'en' else "将数值归一化到0-1范围便于比较"
            normalize = st.checkbox(normalize_label, value=False, key="ts_normalize",
                                   help=normalize_help)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if selected_concept and compare_patients:
            try:
                import plotly.graph_objects as go

                df = st.session_state.loaded_concepts[selected_concept]

                # 确保是 DataFrame
                if not isinstance(df, pd.DataFrame):
                    format_warn = f"Data format not supported for multi-patient comparison: {type(df).__name__}" if lang == 'en' else f"数据格式不支持多患者比较: {type(df).__name__}"
                    st.warning(format_warn)
                    return

                id_col = st.session_state.id_col

                # 确定数值列
                value_cols = _get_concept_numeric_value_columns(df)

                # 检测时间列
                time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat']
                time_col = None
                for tc in time_candidates:
                    if tc in df.columns:
                        time_col = tc
                        break

                has_trend_data = bool(value_cols and time_col and id_col in df.columns)
                if has_trend_data:
                    value_col = compare_value_col if compare_value_col in value_cols else _choose_concept_value_column(selected_concept, df)

                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    comparison_stats = []

                    for i, pid in enumerate(compare_patients):
                        patient_df = df[df[id_col] == pid].sort_values(time_col)
                        if len(patient_df) == 0:
                            continue

                        y_values = patient_df[value_col].values
                        if normalize and len(y_values) > 0:
                            y_min, y_max = y_values.min(), y_values.max()
                            if y_max > y_min:
                                y_values = (y_values - y_min) / (y_max - y_min)

                        patient_label = f"Patient {pid}" if lang == 'en' else f"患者 {pid}"
                        fig.add_trace(go.Scatter(
                            x=patient_df[time_col],
                            y=y_values,
                            mode='lines+markers',
                            name=patient_label,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4),
                        ))

                        if lang == 'en':
                            comparison_stats.append({
                                'Patient': pid,
                                'Mean': f"{patient_df[value_col].mean():.2f}",
                                'Max': f"{patient_df[value_col].max():.2f}",
                                'Min': f"{patient_df[value_col].min():.2f}",
                                'Records': len(patient_df),
                            })
                        else:
                            comparison_stats.append({
                                '患者': pid,
                                '平均值': f"{patient_df[value_col].mean():.2f}",
                                '最大值': f"{patient_df[value_col].max():.2f}",
                                '最小值': f"{patient_df[value_col].min():.2f}",
                                '记录数': len(patient_df),
                            })

                    chart_title = (
                        f"📊 {selected_concept.upper()} Multi-Patient Comparison"
                        if lang == 'en'
                        else f"📊 {selected_concept.upper()} 多患者比较"
                    )
                    x_axis_label = "Time (hours)" if lang == 'en' else "时间 (小时)"
                    y_suffix = " (Normalized)" if lang == 'en' else " (归一化)"
                    fig.update_layout(
                        template="plotly_white",
                        title=chart_title,
                        xaxis_title=x_axis_label,
                        yaxis_title=f"{value_col}" + (y_suffix if normalize else ""),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        height=450,
                        font=dict(size=14, color='black'),
                    )
                    fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

                    if comparison_stats:
                        compare_stats_title = "#### 📊 Comparison Statistics" if lang == 'en' else "#### 📊 比较统计"
                        st.markdown(compare_stats_title)
                        _st_dataframe_compat(
                            st,
                            pd.DataFrame(comparison_stats),
                            width="stretch",
                            hide_index=True,
                        )
                else:
                    # 🔧 FIX: 检测是否有布尔列（包括pandas boolean和numpy bool）
                    bool_cols = []
                    for col in df.columns:
                        dtype_str = str(df[col].dtype).lower()
                        if 'bool' in dtype_str:
                            bool_cols.append(col)

                    if bool_cols:
                        if lang == 'en':
                            format_warn = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart."
                        else:
                            format_warn = f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                    else:
                        format_warn = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart." if lang == 'en' else f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                    st.warning(format_warn)

            except Exception as e:
                err_msg = f"Comparison chart rendering failed: {e}" if lang == 'en' else f"比较图表渲染失败: {e}"
                st.error(err_msg)
