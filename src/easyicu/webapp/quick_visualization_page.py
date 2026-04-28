"""Quick visualization page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_quick_visualization_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


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

    screenshot_title = "📸 Screenshot Mode" if lang == 'en' else "📸 截图模式"
    screenshot_hint = (
        "Hide sidebar and AI dock, reduce chart chrome, and apply figure-friendly defaults."
        if lang == 'en'
        else "隐藏侧边栏和 AI 浮窗、减少图表工具条，并应用更适合论文截图的默认视图。"
    )

    if not direct_figure_panel:
        header_cols = st.columns([3.1, 1.3])
        with header_cols[0]:
            _viz_title = get_text('quick_viz')
            hint_text = (
                "Generate demo data or load previously exported result files for interactive analysis"
                if entry_mode == 'demo'
                else "Load previously exported result files for interactive analysis"
            )
            if lang != 'en':
                hint_text = "生成模拟数据或从之前导出的结果文件中加载，进行交互式分析" if entry_mode == 'demo' else "从之前导出的结果文件中加载，进行交互式分析"

            st.markdown(
                f'''
                <div class="compact-section-title">{_viz_title}</div>
                <div class="compact-section-desc">{hint_text}</div>
                ''',
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            if not _is_screenshot_mode():
                st.toggle(
                    screenshot_title,
                    value=st.session_state.get('screenshot_mode', False),
                    key='screenshot_mode',
                    help=screenshot_hint,
                )
                st.caption(screenshot_hint)

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

    data_loaded = len(st.session_state.loaded_concepts) > 0
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

    show_data_loader = not (screenshot_mode and data_loaded)
    if show_data_loader:
        expander_label = "⚙️ Data Loading Settings" if lang == 'en' else "⚙️ 数据加载设置"
        with st.expander(expander_label, expanded=not data_loaded):
            allow_demo = entry_mode != 'real'
            source_options = ["exported"] + (["demo"] if allow_demo else [])
            source_labels = {
                "exported": "📁 Previously Exported Data" if lang == 'en' else "📁 加载之前导出的结果文件",
                "demo": "🧪 Demo Data" if lang == 'en' else "🧪 模拟数据",
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
                                    "🔍 Load Data" if lang == 'en' else "🔍 加载数据",
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
                    "Generate one complete demo review workspace"
                    if lang == 'en' else
                    "生成完整演示审阅工作区"
                )
                _viz_demo_subtitle = (
                    "Loads representative tables, time series, patient overview, and quality metrics together for the Figure 3-style multi-view review."
                    if lang == 'en' else
                    "一次性加载代表性表格、时间序列、患者概览和质量指标，用于 Figure 3 风格的多视角审阅。"
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
                        200,
                        50,
                        key="viz_demo_patients",
                    )
                with col2:
                    hours = st.slider(
                        "Data Duration (hours)" if lang == 'en' else "数据时长(小时)",
                        24,
                        168,
                        72,
                        key="viz_demo_hours",
                    )

                feature_hint = (
                    "Will generate ~160+ features across all modules (Vitals, Labs, SOFA, Sepsis, AKI, etc.)"
                    if lang == 'en'
                    else "将生成约160+个特征，覆盖所有模块（生命体征、实验室、SOFA、脓毒症、AKI等）"
                )
                st.caption(f"💡 {feature_hint}")

                if st.button(
                    "🚀 Generate & Load All Demo Data" if lang == 'en' else "🚀 生成并加载全部模拟数据",
                    type="primary",
                    use_container_width=True,
                    key="viz_load_demo",
                ):
                    with st.spinner(
                        "Generating all mock data (~160+ features)..." if lang == 'en' else "正在生成全部模拟数据（约160+特征）..."
                    ):
                        params = get_mock_params_with_cohort()
                        params['n_patients'] = n_patients
                        params['hours'] = hours
                        mock_data, patient_ids = generate_mock_data(**params)
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

        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            get_text('review_tables'),
            get_text('review_trends'),
            get_text('review_patients'),
            get_text('review_quality'),
        ])

        with sub_tab1:
            render_data_table_subtab()
        with sub_tab2:
            render_timeseries_page()
        with sub_tab3:
            render_patient_page()
        with sub_tab4:
            render_quality_page()
    else:
        empty_title = "Preview workspace awaits data" if lang == 'en' else "预览工作区等待数据"
        empty_subtitle = (
            "Generate demo data or load exported files above; the review tabs will appear here as a compact Figure 3-style interface."
            if lang == 'en' else
            "请在上方生成演示数据或加载导出文件；随后这里会显示 Figure 3 风格的紧凑审阅界面。"
        )
        no_data_msg = f"""
        <div class="viz-empty-state">
            <div class="viz-empty-icon">📊</div>
            <div class="viz-empty-title">{html.escape(empty_title)}</div>
            <div class="viz-empty-subtitle">{html.escape(empty_subtitle)}</div>
        </div>
        """
        st.markdown(no_data_msg, unsafe_allow_html=True)
