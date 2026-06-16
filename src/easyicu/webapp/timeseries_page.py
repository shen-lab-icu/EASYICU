"""Time-series page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat
from easyicu.webapp.ui_helpers import StatCard, render_stat_grid


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_timeseries_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _plain_display_label(label: str) -> str:
    """Remove leading decorative symbols from legacy labels."""
    text = str(label or "").strip()
    while text and not (text[0].isalnum() or "\u4e00" <= text[0] <= "\u9fff"):
        text = text[1:].lstrip()
    return text or str(label or "")


_TS_SERIES_COLORS = ["#0f766e", "#334155", "#64748b", "#b45309", "#7c3aed"]


def _ts_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _ts_notice_html(tone: str, kicker: str, title: str, body: str, meta: str = "") -> str:
    meta_html = f'<em>{_ts_escape(meta)}</em>' if meta else ""
    return (
        f'<div class="eu-ts-notice {tone}">'
        f'<span>{_ts_escape(kicker)}</span>'
        f'<b>{_ts_escape(title)}</b>'
        f'<p>{_ts_escape(body)}</p>'
        f'{meta_html}'
        '</div>'
    )


def _render_ts_notice(tone: str, kicker: str, title: str, body: str, meta: str = "") -> None:
    st.markdown(_ts_notice_html(tone, kicker, title, body, meta), unsafe_allow_html=True)


def _ts_contract_row_html(index: str, label: str, detail: str, tone: str) -> str:
    return (
        f'<div class="eu-ts-contract-row {tone}">'
        f'<span>{_ts_escape(index)}</span>'
        '<div>'
        f'<b>{_ts_escape(label)}</b>'
        f'<em>{_ts_escape(detail)}</em>'
        '</div>'
        '</div>'
    )


def _timeseries_contract_html(
    *,
    lang: str,
    mode: str,
    concept_count: int,
    patient_count: int,
    lane_count: int,
) -> str:
    is_en = lang == "en"
    rows = [
        _ts_contract_row_html(
            "01",
            "Patient scope" if is_en else "患者范围",
            f"{patient_count} patients" if is_en else f"{patient_count} 位患者",
            "ready" if patient_count else "warn",
        ),
        _ts_contract_row_html(
            "02",
            "Loaded signals" if is_en else "已加载信号",
            f"{concept_count} concepts" if is_en else f"{concept_count} 个概念",
            "ready" if concept_count else "neutral",
        ),
        _ts_contract_row_html(
            "03",
            "Clinical lanes" if is_en else "临床分组",
            f"{lane_count} lanes available" if is_en else f"{lane_count} 个分组可用",
            "ready" if lane_count else "neutral",
        ),
        _ts_contract_row_html(
            "04",
            "Review mode" if is_en else "审阅模式",
            mode,
            "ready",
        ),
    ]
    return (
        '<div class="eu-ts-contract">'
        '<div class="eu-ts-contract-head">'
        f'<span>{_ts_escape("Trajectory ledger" if is_en else "轨迹账本")}</span>'
        f'<b>{_ts_escape("Patient -> signal -> lane -> chart" if is_en else "患者 -> 信号 -> 分组 -> 图表")}</b>'
        '</div>'
        f'<p>{_ts_escape("Every trajectory view is tied to loaded local exports and the selected patient scope." if is_en else "每个轨迹视图都绑定到已加载的本地导出与当前患者范围。")}</p>'
        '<div class="eu-ts-contract-list">'
        + "".join(rows)
        + '</div>'
        '</div>'
    )


def _ts_lane_header_html(label: str, count: int, patient_id: object, lang: str) -> str:
    detail = (
        f"{count} signals for patient {patient_id}"
        if lang == "en" else
        f"患者 {patient_id} · {count} 个信号"
    )
    return (
        '<div class="eu-ts-lane-head">'
        f'<span>{_ts_escape("Clinical lane" if lang == "en" else "临床分组")}</span>'
        f'<b>{_ts_escape(label)}</b>'
        f'<em>{_ts_escape(detail)}</em>'
        '</div>'
    )


def _ts_static_value_html(label: str, value: object, lang: str) -> str:
    return (
        '<div class="eu-ts-static-value">'
        f'<span>{_ts_escape("Static value" if lang == "en" else "静态值")}</span>'
        f'<b>{_ts_escape(value)}</b>'
        f'<em>{_ts_escape(label)}</em>'
        '</div>'
    )


def _apply_ts_plot_style(
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
        margin=margin or dict(l=48, r=24, t=48, b=42),
        showlegend=showlegend,
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
        <div class="eu-subhead">
            <div class="t">{_ts_title}</div>
            <div class="s">{_ts_sub}</div>
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
        _render_ts_notice(
            "info",
            "Trajectory workspace" if lang == "en" else "轨迹工作台",
            "Local export required" if lang == "en" else "需要本地导出",
            (
                "Load exported module files before reviewing patient trajectories."
                if lang == "en" else
                "请先加载导出的模块文件，再审阅患者轨迹。"
            ),
            "Data Tables -> Time Series" if lang == "en" else "数据表 -> 时序分析",
        )
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

    _patient_count = len(st.session_state.get('patient_ids', []))
    _lane_count = sum(
        1
        for _concepts in CLINICAL_LANES.values()
        if any(_c in available_concepts for _c in _concepts)
    )
    st.markdown(
        _timeseries_contract_html(
            lang=lang,
            mode=analysis_mode,
            concept_count=len(available_concepts),
            patient_count=_patient_count,
            lane_count=_lane_count,
        ),
        unsafe_allow_html=True,
    )

    # ============ Clinical Lanes View (默认) ============
    if analysis_mode == mode_lanes:
        import plotly.graph_objects as go

        if not st.session_state.patient_ids:
            _render_ts_notice(
                "warning",
                "Patient scope" if lang == "en" else "患者范围",
                "No patient data" if lang == 'en' else "无患者数据",
                (
                    "Load a review workspace with patient identifiers before rendering trajectories."
                    if lang == "en" else
                    "请先加载包含患者标识的审阅工作区，再渲染轨迹。"
                ),
            )
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
            lane_rows = []

            for lane_name, lane_concepts in CLINICAL_LANES.items():
                _lane_avail = [c for c in lane_concepts if c in available_concepts]
                if screenshot_mode and screenshot_concepts is not None:
                    _lane_avail = [c for c in _lane_avail if c in screenshot_concepts]
                if not _lane_avail:
                    continue
                lane_rows.append((lane_name, get_text(f'lane_{lane_name}'), _lane_avail))

            if not lane_rows:
                _render_ts_notice(
                    "info",
                    "Clinical lanes" if lang == "en" else "临床分组",
                    "No time-series lanes available." if lang == 'en' else "当前没有可用的时序分组。",
                    (
                        "Loaded modules do not include compatible longitudinal signals."
                        if lang == "en" else
                        "已加载模块不包含可用于纵向审阅的信号。"
                    ),
                )
                return

            if screenshot_mode:
                lanes_to_render = lane_rows
            else:
                lane_state_key = "ts_active_lane"
                lane_keys = [name for name, _label, _concepts in lane_rows]
                if st.session_state.get(lane_state_key) not in lane_keys:
                    st.session_state[lane_state_key] = lane_keys[0]
                lane_label_map = {name: label for name, label, _concepts in lane_rows}
                st.markdown(
                    f'<div class="inline-control-label">{"Clinical lane" if lang == "en" else "临床分组"}</div>',
                    unsafe_allow_html=True,
                )
                active_lane = st.radio(
                    "Clinical lane" if lang == 'en' else "临床分组",
                    options=lane_keys,
                    format_func=lambda key: lane_label_map.get(key, key),
                    horizontal=True,
                    key=lane_state_key,
                    label_visibility="collapsed",
                )
                lanes_to_render = [row for row in lane_rows if row[0] == active_lane]

            for lane_name, lane_label, _lane_avail in lanes_to_render:
                st.markdown(
                    _ts_lane_header_html(lane_label, len(_lane_avail), _lane_pid, lang),
                    unsafe_allow_html=True,
                )

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
                                line=dict(width=1.6, color=_TS_SERIES_COLORS[idx % len(_TS_SERIES_COLORS)]),
                                marker=dict(size=3.5, color=_TS_SERIES_COLORS[idx % len(_TS_SERIES_COLORS)])
                            ))
                            fig = _add_clinical_thresholds(fig, cname, _show_thresh)

                            _unit = CLINICAL_THRESHOLDS.get(cname, {}).get('unit', '')
                            fig = _apply_ts_plot_style(
                                fig,
                                title=f"{cname}" + (f" ({_unit})" if _unit else ""),
                                height=200,
                                margin=dict(l=40, r=10, t=30, b=25),
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

                st.markdown('<div class="eu-ts-divider"></div>', unsafe_allow_html=True)
        return

    if analysis_mode == mode_single:
        # 顶部控制面板 - 🔧 FIX: 添加模块筛选，方便用户在100+特征中找到想要的
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 0.9])

        with col1:
            # 🔧 FIX: 先选择模块，再选择特征
            module_label = "Select Module" if lang == 'en' else "选择模块"
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
                    display_name = _plain_display_label(CONCEPT_GROUPS_DISPLAY.get(grp_key, grp_key))
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
                    if _plain_display_label(display) == selected_module:
                        selected_grp_key = grp_key
                        break
                if selected_grp_key:
                    grp_concepts = CONCEPT_GROUPS_INTERNAL.get(selected_grp_key, [])
                    filtered_concepts = [c for c in available_concepts if c in grp_concepts]
                else:
                    filtered_concepts = available_concepts

            concept_label = "Select Concept" if lang == 'en' else "选择 Concept"
            concept_help = "Select data type to visualize" if lang == 'en' else "选择要可视化的数据类型"
            selected_concept = st.selectbox(
                concept_label,
                options=filtered_concepts if filtered_concepts else available_concepts,
                key="ts_concept",
                help=concept_help
            )

        with col3:
            if st.session_state.patient_ids:
                patient_label = "Select Patient" if lang == 'en' else "选择患者"
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
                _render_ts_notice(
                    "warning",
                    "Patient scope" if lang == "en" else "患者范围",
                    no_patient_msg,
                    "Load a patient-level review workspace first." if lang == "en" else "请先加载患者级审阅工作区。",
                )

        with col4:
            chart_label = "Chart Type" if lang == 'en' else "图表类型"
            line_opt = "Line Chart" if lang == 'en' else "折线图"
            scatter_opt = "Scatter Plot" if lang == 'en' else "散点图"
            area_opt = "Area Chart" if lang == 'en' else "面积图"
            chart_type = st.selectbox(
                chart_label,
                options=[line_opt, scatter_opt, area_opt],
                key="ts_chart_type"
            )

        with col5:
            value_label = "Value Column" if lang == 'en' else "数值列"
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
                _render_ts_notice(
                    "warning",
                    "Signal format" if lang == "en" else "信号格式",
                    format_warn,
                    "Use exported module tables for trajectory review." if lang == "en" else "请使用导出的模块表进行轨迹审阅。",
                )
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
                                _render_ts_notice(
                                    "info",
                                    "Trajectory check" if lang == "en" else "轨迹检查",
                                    "No valid time series after sorting/cleaning" if lang == 'en' else "清理后没有可用的时序数据",
                                    "Try a different signal or patient." if lang == "en" else "请尝试其他信号或患者。",
                                )
                                return
                            # 根据图表类型创建图表
                            line_type = "Line Chart" if lang == 'en' else "折线图"
                            scatter_type = "Scatter Plot" if lang == 'en' else "散点图"
                            patient_label = "Patient" if lang == 'en' else "患者"
                            chart_title = f"{selected_concept.upper()} - {patient_label} {patient_id}"

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
                            fig = _apply_ts_plot_style(
                                fig,
                                title=chart_title,
                                height=450,
                                x_title=time_label,
                                y_title=value_col.upper(),
                                margin=dict(l=50, r=30, t=50, b=50),
                            )
                            fig.update_traces(
                                line=dict(width=2.2, color=_TS_SERIES_COLORS[0]),
                                marker=dict(size=6, color=_TS_SERIES_COLORS[0])
                            )

                            st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())
                        else:
                            _render_ts_notice(
                                "info",
                                "Static signal" if lang == "en" else "静态信号",
                                "Static value, no time series data" if lang == 'en' else "静态值，无时序数据",
                                "EasyICU shows the bound value instead of drawing a false trend." if lang == "en" else "EasyICU 展示已绑定值，不绘制伪趋势。",
                            )
                            if len(patient_df) == 1:
                                val = patient_df[value_col].iloc[0]
                                st.markdown(
                                    _ts_static_value_html(value_col.upper(), val, lang),
                                    unsafe_allow_html=True,
                                )
                            else:
                                _st_dataframe_compat(st, patient_df[[value_col]], width="stretch")

                        # 显示统计信息
                        if show_stats:
                            stat_title = "#### Statistical Summary" if lang == 'en' else "#### 统计摘要"
                            st.markdown(stat_title)
                            values = patient_df[value_col]
                            if lang == 'en':
                                stats = [
                                    ("Min", f"{values.min():.2f}"),
                                    ("Max", f"{values.max():.2f}"),
                                    ("Mean", f"{values.mean():.2f}"),
                                    ("Std Dev", f"{values.std():.2f}"),
                                    ("Records", f"{len(values)}"),
                                ]
                            else:
                                stats = [
                                    ("最小值", f"{values.min():.2f}"),
                                    ("最大值", f"{values.max():.2f}"),
                                    ("平均值", f"{values.mean():.2f}"),
                                    ("标准差", f"{values.std():.2f}"),
                                    ("记录数", f"{len(values)}"),
                                ]
                            render_stat_grid(
                                [StatCard(label=label, value=value) for label, value in stats],
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
                                warn_msg = f"{selected_concept.upper()} is a Boolean (True/False) feature."
                            else:
                                warn_msg = f"{selected_concept.upper()} 是布尔类型（True/False）特征。"
                        else:
                            warn_msg = f"{selected_concept.upper()} does not expose a numeric value column." if lang == 'en' else f"{selected_concept.upper()} 没有可用数值列。"
                        _render_ts_notice(
                            "warning",
                            "Chart gate" if lang == "en" else "图表关口",
                            warn_msg,
                            (
                                "Time Series requires numeric values; the raw rows are shown for review."
                                if lang == "en" else
                                "时序分析需要数值型数据；下方保留原始行供复核。"
                            ),
                        )
                        display_patient_df = patient_df.head(20).copy()
                        for col in display_patient_df.columns:
                            dtype_str = str(display_patient_df[col].dtype).lower()
                            if 'bool' in dtype_str:
                                display_patient_df[col] = display_patient_df[col].astype(str)
                        st.dataframe(display_patient_df, use_container_width=True)

                except Exception as e:
                    err_msg = f"Chart rendering failed: {e}" if lang == 'en' else f"图表渲染失败: {e}"
                    _render_ts_notice(
                        "warning",
                        "Chart render" if lang == "en" else "图表渲染",
                        err_msg,
                        "Fallback chart will be attempted when a time column is available." if lang == "en" else "若存在时间列，将尝试回退图表。",
                    )
                    if 'time' in patient_df.columns:
                        chart_df = patient_df.set_index('time')
                        fallback_value_col = selected_value_col if selected_value_col in chart_df.columns else _choose_concept_value_column(selected_concept, chart_df.reset_index())
                        if fallback_value_col and fallback_value_col in chart_df.columns:
                            st.line_chart(chart_df[fallback_value_col])
            else:
                no_data_msg = f"No {selected_concept} data for patient {patient_id}" if lang == 'en' else f"患者 {patient_id} 无 {selected_concept} 数据"
                _render_ts_notice(
                    "info",
                    "Signal availability" if lang == "en" else "信号可用性",
                    no_data_msg,
                    "Choose another patient or signal." if lang == "en" else "请选择其他患者或信号。",
                )

        # 数据表格预览
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        preview_label = "Data Table Preview" if lang == 'en' else "数据表格预览"
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
                    _render_ts_notice(
                        "info",
                        "Preview" if lang == "en" else "预览",
                        format_msg,
                        "Only tabular module exports can be previewed here." if lang == "en" else "此处仅预览表格型模块导出。",
                    )

    else:  # 多患者比较模式
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 0.9])

        with col1:
            # 🔧 FIX: 先选择模块，再选择特征
            module_label = "Select Module" if lang == 'en' else "选择模块"
            all_modules_opt = "All Modules" if lang == 'en' else "全部模块"

            # 🔧 FIX (2026-02-05): 只显示支持时序分析的模块（排除静态数据模块）
            module_options = [all_modules_opt]
            for grp_key in CONCEPT_GROUPS_INTERNAL:
                # 跳过不支持时序分析的模块（demographics, outcome）
                if grp_key not in TIME_SERIES_COMPATIBLE_MODULES:
                    continue
                grp_concepts = CONCEPT_GROUPS_INTERNAL[grp_key]
                if any(c in available_concepts for c in grp_concepts):
                    display_name = _plain_display_label(CONCEPT_GROUPS_DISPLAY.get(grp_key, grp_key))
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
                    if _plain_display_label(display) == selected_module_multi:
                        selected_grp_key = grp_key
                        break
                if selected_grp_key:
                    grp_concepts = CONCEPT_GROUPS_INTERNAL.get(selected_grp_key, [])
                    filtered_concepts_multi = [c for c in available_concepts if c in grp_concepts]
                else:
                    filtered_concepts_multi = available_concepts

            concept_label = "Select Concept" if lang == 'en' else "选择 Concept"
            selected_concept = st.selectbox(
                concept_label,
                options=filtered_concepts_multi if filtered_concepts_multi else available_concepts,
                key="ts_concept_multi"
            )

        with col3:
            if st.session_state.patient_ids:
                compare_label = "Select patients to compare (max 5)" if lang == 'en' else "选择要比较的患者 (最多5个)"
                compare_search = st.text_input(
                    "Search Patient IDs" if lang == 'en' else "搜索患者ID",
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
            value_label = "Value Column" if lang == 'en' else "数值列"
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
                    _render_ts_notice(
                        "warning",
                        "Signal format" if lang == "en" else "信号格式",
                        format_warn,
                        "Use exported module tables for comparison." if lang == "en" else "请使用导出的模块表进行比较。",
                    )
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
                            line=dict(color=_TS_SERIES_COLORS[i % len(_TS_SERIES_COLORS)], width=2),
                            marker=dict(size=4, color=_TS_SERIES_COLORS[i % len(_TS_SERIES_COLORS)]),
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
                        f"{selected_concept.upper()} Multi-Patient Comparison"
                        if lang == 'en'
                        else f"{selected_concept.upper()} 多患者比较"
                    )
                    x_axis_label = "Time (hours)" if lang == 'en' else "时间 (小时)"
                    y_suffix = " (Normalized)" if lang == 'en' else " (归一化)"
                    fig = _apply_ts_plot_style(
                        fig,
                        title=chart_title,
                        height=450,
                        x_title=x_axis_label,
                        y_title=f"{value_col}" + (y_suffix if normalize else ""),
                        showlegend=True,
                    )
                    fig.update_layout(
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

                    if comparison_stats:
                        compare_stats_title = "Comparison statistics" if lang == 'en' else "比较统计"
                        st.markdown(
                            f'<div class="eu-ts-table-title">{_ts_escape(compare_stats_title)}</div>',
                            unsafe_allow_html=True,
                        )
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
                            format_warn = f"{selected_concept.upper()} is a Boolean (True/False) feature."
                        else:
                            format_warn = f"{selected_concept.upper()} 是布尔类型（True/False）特征。"
                    else:
                        format_warn = f"{selected_concept.upper()} does not expose a numeric value column." if lang == 'en' else f"{selected_concept.upper()} 没有可用数值列。"
                    _render_ts_notice(
                        "warning",
                        "Chart gate" if lang == "en" else "图表关口",
                        format_warn,
                        (
                            "Multi-patient comparison requires numeric values and a time column."
                            if lang == "en" else
                            "多患者比较需要数值列和时间列。"
                        ),
                    )

            except Exception as e:
                err_msg = f"Comparison chart rendering failed: {e}" if lang == 'en' else f"比较图表渲染失败: {e}"
                _render_ts_notice(
                    "danger",
                    "Comparison render" if lang == "en" else "比较渲染",
                    err_msg,
                    "Review the selected concept table before retrying." if lang == "en" else "请先复核所选概念表后再重试。",
                )
