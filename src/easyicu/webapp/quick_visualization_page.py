"""Quick visualization page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_quick_visualization_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _load_demo_review_workspace_from_state() -> tuple[int, int]:
    """Hydrate Quick Viz on explicit user request."""
    params_getter = globals().get("get_mock_params_with_cohort")
    if callable(params_getter):
        params = dict(params_getter())
    else:
        params = dict(st.session_state.get("mock_params") or {})

    try:
        params["n_patients"] = int(params.get("n_patients") or 50)
    except (TypeError, ValueError):
        params["n_patients"] = 50
    try:
        params["hours"] = int(params.get("hours") or 48)
    except (TypeError, ValueError):
        params["hours"] = 48
    params["demo_profile"] = "lite"

    demo_generator = globals().get("generate_lightweight_demo_data", generate_mock_data)
    mock_data, patient_ids = demo_generator(**params)
    st.session_state.mock_params = params
    st.session_state.loaded_concepts = mock_data
    st.session_state.loaded_data_origin = "demo_viz"
    st.session_state.patient_ids = sorted(patient_ids) if patient_ids else []
    st.session_state.id_col = "stay_id"
    st.session_state.time_col = "time"
    st.session_state.selected_concepts = list(mock_data.keys())
    st.session_state.trigger_export = False
    st.session_state["_exporting_in_progress"] = False
    st.session_state.viz_data_source_mode = "demo"
    return len(mock_data), len(st.session_state.patient_ids)


def _quick_viz_panel_options(lang: str) -> list[tuple[str, str]]:
    labels = {
        "Data Tables": get_text("review_tables"),
        "Time Series": get_text("review_trends"),
        "Patient Overview": get_text("review_patients"),
        "Data Quality": get_text("review_quality"),
    }
    return [(key, labels[key]) for key in labels]


def _render_quick_viz_panel_switcher(lang: str) -> str:
    """Render a lazy panel switcher and return the active panel key.

    Streamlit tabs eagerly execute every tab body on every rerun. The
    Quick Visualization panels are dataframe/chart heavy, so this
    segmented radio keeps the visual four-panel affordance while rendering
    only the selected panel.
    """
    panel_options = _quick_viz_panel_options(lang)
    panel_keys = [key for key, _label in panel_options]
    label_map = dict(panel_options)
    state_key = "quick_viz_active_panel"
    if st.session_state.get(state_key) not in panel_keys:
        st.session_state[state_key] = panel_keys[0]

    switcher_label = "Review panel" if lang == "en" else "审阅面板"
    with st.container(key="qv_panel_switcher"):
        st.markdown(
            f'<div class="inline-control-label">{html.escape(switcher_label)}</div>',
            unsafe_allow_html=True,
        )
        active_panel = st.radio(
            switcher_label,
            options=panel_keys,
            format_func=lambda key: label_map.get(key, key),
            horizontal=True,
            key=state_key,
            label_visibility="collapsed",
        )
    return active_panel


def render_quick_visualization_page(app_context: dict[str, Any] | None = None):
    """渲染快速可视化主页面 - 包含数据加载区域和四个子模块。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')
    entry_mode = st.session_state.get('entry_mode', 'none')
    if _sync_quick_viz_screenshot_mode(st.session_state, lang=lang):
        st.rerun()
    screenshot_mode = bool(st.session_state.get('screenshot_mode', False))
    figure_panel = st.session_state.get('_figure_target_panel') if screenshot_mode else None
    direct_figure_panel = figure_panel in {'Data Tables', 'Time Series', 'Patient Overview', 'Data Quality'}

    # Shell-A declutter: the topbar breadcrumb + each subtab's own header
    # already name the page, so there is no separate page header here. The
    # screenshot-mode toggle was removed at the user's request.

    viz_notices = st.session_state.pop('_viz_notices', [])
    for notice in viz_notices[:3]:
        level = str(notice.get('level') or 'info')
        message = str(notice.get('message') or '').strip()
        if message:
            st.markdown(
                f'<div class="compact-inline-notice {level}">{message}</div>',
                unsafe_allow_html=True,
            )

    if screenshot_mode and not direct_figure_panel:
        screenshot_notice = (
            "Figure preset active: compact layout, hidden side chrome, and screenshot-first defaults."
            if lang == 'en'
            else "截图预设已启用：界面更紧凑、隐藏侧边栏干扰，并自动切到更适合论文配图的默认视图。"
        )
        st.markdown(f'<div class="compact-inline-notice info">{screenshot_notice}</div>', unsafe_allow_html=True)

    if 'viz_export_path' not in st.session_state:
        st.session_state.viz_export_path = ""
    recent_export_path = st.session_state.get('viz_confirmed_path') or st.session_state.get('last_export_dir') or ""
    if recent_export_path and (st.session_state.get('_prefer_exported_viz') or not st.session_state.get('viz_export_path')):
        st.session_state.viz_export_path = recent_export_path
        st.session_state['viz_export_path_input'] = recent_export_path
        st.session_state['viz_data_source_mode'] = "exported"
        st.session_state['_prefer_exported_viz'] = False

    auto_viz_request = st.session_state.pop('_viz_auto_load_export', None)
    if auto_viz_request and auto_viz_request.get('path'):
        auto_path = auto_viz_request.get('path')
        if Path(auto_path).exists():
            with st.spinner("Refreshing with newly exported files..." if lang == 'en' else "正在使用最新导出文件刷新..."):
                load_from_exported(
                    auto_path,
                    max_patients=auto_viz_request.get('max_patients', 100),
                    selected_files=auto_viz_request.get('selected_files'),
                )
            st.session_state['_viz_auto_load_notice'] = (
                f"✅ Auto-loaded exported files from `{auto_path}`"
                if lang == 'en' else
                f"✅ 已自动加载最新导出文件：`{auto_path}`"
            )
            recent_export_path = auto_path

    auto_notice = st.session_state.pop('_viz_auto_load_notice', None)
    if auto_notice:
        st.success(auto_notice)

    data_loaded = len(st.session_state.loaded_concepts) > 0
    show_data_loader = not data_loaded
    if show_data_loader:
        with st.container(key="eu_qv_loader"):
            st.markdown(
                '<div class="eu-qv-loader-head">'
                '<div>'
                f'<div class="k">{html.escape("Quick Visualization" if lang == "en" else "快速可视化")}</div>'
                f'<div class="t">{html.escape("Load a review workspace" if lang == "en" else "加载审阅工作区")}</div>'
                f'<div class="s">{html.escape("Start with exported EasyICU tables or generate a compact demo set; review tabs appear immediately after loading." if lang == "en" else "从已导出的 EasyICU 表格开始，或生成一个紧凑演示集；加载后直接进入审阅子页。")}</div>'
                '</div>'
                '<span class="eu-qv-loader-badge">Data Tables · Time Series · Patient · Quality</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            allow_demo = entry_mode != 'real'
            source_options = ["exported"] + (["demo"] if allow_demo else [])
            source_labels = {
                "exported": "Previously exported data" if lang == 'en' else "加载之前导出的结果文件",
                "demo": "Demo data" if lang == 'en' else "模拟数据",
            }
            st.session_state.viz_data_source_mode = _resolve_viz_data_source_mode(
                current_mode=st.session_state.get('viz_data_source_mode'),
                recent_export_path=recent_export_path,
                allow_demo=allow_demo,
                entry_mode=entry_mode,
            )
            current_source = st.radio(
                "Data Source" if lang == 'en' else "数据来源",
                options=source_options,
                format_func=lambda value: source_labels[value],
                horizontal=True,
                key="viz_data_source_mode",
            )

            if current_source == "exported":
                export_path = _directory_input(
                    "Folder Containing Exported Data Files" if lang == 'en' else "存放导出结果文件的文件夹",
                    value=st.session_state.get('viz_export_path') or recent_export_path,
                    input_key="viz_export_path_input",
                    button_key="viz_export_path_browse",
                    help="Choose the folder that contains EasyICU exported CSV / Parquet / Excel files" if lang == 'en' else "选择存放 EasyICU 导出 CSV / Parquet / Excel 文件的文件夹",
                )
                st.session_state.viz_export_path = export_path

                if export_path:
                    export_dir = Path(export_path)
                    if export_dir.exists() and export_dir.is_dir():
                        available_files = sorted(
                            list(export_dir.glob('*.csv'))
                            + list(export_dir.glob('*.parquet'))
                            + list(export_dir.glob('*.xlsx')),
                            key=lambda path: path.name,
                        )
                        file_names = list(dict.fromkeys(file.stem for file in available_files))

                        if file_names:
                            st.success(
                                f"✅ Found {len(file_names)} data files" if lang == 'en' else f"✅ 发现 {len(file_names)} 个数据文件"
                            )
                            selected_files = st.multiselect(
                                "Select Tables to Load" if lang == 'en' else "选择要加载的表格",
                                options=file_names,
                                default=file_names,
                                key="viz_selected_files",
                            )

                            patient_options = [50, 100, 200, 500, -1]
                            option_labels = {
                                50: "50 (Recommended)" if lang == 'en' else "50 (推荐)",
                                100: "100",
                                200: "200",
                                500: "500 (Slow)" if lang == 'en' else "500 (较慢)",
                                -1: "All (May Lag)" if lang == 'en' else "全部 (可能卡顿)",
                            }
                            max_patients_opt = st.selectbox(
                                "Max Patients to Load" if lang == 'en' else "最大加载患者数",
                                options=patient_options,
                                index=0,
                                format_func=lambda value: option_labels[value],
                                key="viz_max_patients",
                            )
                            max_patients = None if max_patients_opt == -1 else max_patients_opt

                            if selected_files:
                                if st.button(
                                    "Load selected data" if lang == 'en' else "加载所选数据",
                                    type="primary",
                                    use_container_width=True,
                                    key="viz_load_files",
                                ):
                                    with st.spinner("Loading data..." if lang == 'en' else "正在加载数据..."):
                                        load_from_exported(export_path, max_patients=max_patients, selected_files=selected_files)
                                    st.rerun()
                            else:
                                st.warning("⚠️ Please select at least one file" if lang == 'en' else "⚠️ 请至少选择一个文件")
                        else:
                            st.warning(
                                "⚠️ No data files found in this directory (CSV/Parquet/Excel)"
                                if lang == 'en'
                                else "⚠️ 该目录下未找到数据文件 (CSV/Parquet/Excel)"
                            )
                    else:
                        st.error("❌ Directory does not exist" if lang == 'en' else "❌ 目录不存在")

            elif current_source == "demo":
                _viz_demo_title = (
                    "Generate a lightweight demo review workspace"
                    if lang == 'en' else
                    "生成轻量演示审阅工作区"
                )
                _viz_demo_subtitle = (
                    "Loads a fast core ICU concept set for tables, trends, patient overview, and quality checks."
                    if lang == 'en' else
                    "加载轻量核心 ICU 概念集，用于表格、趋势、患者概览和质量检查。"
                )
                st.markdown(
                    f"""
                    <div class="viz-demo-load-card">
                        <div class="viz-demo-load-kicker">DEMO REVIEW</div>
                        <div class="viz-demo-load-title">{html.escape(_viz_demo_title)}</div>
                        <div class="viz-demo-load-subtitle">{html.escape(_viz_demo_subtitle)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    n_patients = st.slider(
                        "Number of Patients" if lang == 'en' else "患者数量",
                        10,
                        120,
                        50,
                        key="viz_demo_patients",
                    )
                with col2:
                    hours = st.slider(
                        "Data Duration (hours)" if lang == 'en' else "数据时长(小时)",
                        24,
                        96,
                        48,
                        key="viz_demo_hours",
                    )

                feature_hint = (
                    "Fast demo profile: core vitals, labs, SOFA/SOFA-2, Sepsis-3, AKI, interventions, demographics, and outcomes."
                    if lang == 'en'
                    else "轻量演示配置：核心生命体征、实验室、SOFA/SOFA-2、Sepsis-3、AKI、干预、人口学和结局。"
                )
                st.caption(feature_hint)

                if st.button(
                    "Generate and load demo workspace" if lang == 'en' else "生成并加载演示工作区",
                    type="primary",
                    use_container_width=True,
                    key="viz_load_demo",
                ):
                    with st.spinner(
                        "Generating lightweight demo data..." if lang == 'en' else "正在生成轻量演示数据..."
                    ):
                        params = get_mock_params_with_cohort()
                        params['n_patients'] = n_patients
                        params['hours'] = hours
                        params['demo_profile'] = 'lite'
                        demo_generator = globals().get("generate_lightweight_demo_data", generate_mock_data)
                        mock_data, patient_ids = demo_generator(**params)
                        st.session_state.loaded_concepts = mock_data
                        st.session_state.loaded_data_origin = 'demo_viz'
                        st.session_state.patient_ids = patient_ids
                        st.session_state.id_col = 'stay_id'
                        st.session_state.time_col = 'time'
                        st.session_state.selected_concepts = list(mock_data.keys())
                    st.rerun()

    if data_loaded:
        if figure_panel in {'Data Tables', 'Time Series', 'Patient Overview', 'Data Quality'}:
            render_quick_figure_panel(figure_panel)
            return

        active_panel = _render_quick_viz_panel_switcher(lang)
        if active_panel == "Data Tables":
            render_data_table_subtab()
        elif active_panel == "Time Series":
            render_timeseries_page()
        elif active_panel == "Patient Overview":
            render_patient_page()
        elif active_panel == "Data Quality":
            render_quality_page()
    else:
        empty_title = "Preview workspace awaits data" if lang == 'en' else "预览工作区等待数据"
        empty_subtitle = (
            "Generate demo data or load exported files above; the review tabs will appear here as a compact multi-view workspace."
            if lang == 'en' else
            "请在上方生成演示数据或加载导出文件；随后这里会显示紧凑的多视角审阅界面。"
        )
        no_data_msg = f"""
        <div class="viz-empty-state">
            <div class="viz-empty-icon">Data</div>
            <div class="viz-empty-title">{html.escape(empty_title)}</div>
            <div class="viz-empty-subtitle">{html.escape(empty_subtitle)}</div>
        </div>
        """
        st.markdown(no_data_msg, unsafe_allow_html=True)
