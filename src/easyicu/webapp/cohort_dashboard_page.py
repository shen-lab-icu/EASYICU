"""Cohort dashboard subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from typing import Any

from easyicu.webapp.cohort_workspace import _bundle_from_raw_schema, _seed_workspace_state


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {
        'render_cohort_dashboard_subtab',
        "_install_app_context",
        "_render_section_heading",
    }
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _render_section_heading(title: str, eyebrow: str | None = None) -> None:
    eyebrow_html = (
        f'<span>{html.escape(eyebrow)}</span>'
        if eyebrow else ""
    )
    st.markdown(
        '<div class="eu-native-section-heading">'
        f'{eyebrow_html}<b>{html.escape(title)}</b>'
        '</div>',
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


def _render_metric_grid(cards: list[tuple[str, str, str, str, str]], *, accent_value: bool = False) -> None:
    body = []
    value_class = "kpi-value accent" if accent_value else "kpi-value"
    for value, label, hint, accent, icon in cards:
        body.append(
            f'<div class="eu-cohort-kpi" style="--accent:{html.escape(accent)}">'
            '<div class="kpi-top">'
            f'<span class="kpi-icon">{html.escape(icon)}</span>'
            f'<span class="kpi-label">{html.escape(label)}</span>'
            '</div>'
            f'<div class="{value_class}">{html.escape(str(value))}</div>'
            f'<div class="kpi-hint">{html.escape(hint)}</div>'
            '</div>'
        )
    st.markdown('<div class="eu-cohort-kpi-grid">' + "".join(body) + '</div>', unsafe_allow_html=True)


SHELL_CHART = {
    "ink": "#1d2935",
    "muted": "#65727f",
    "grid": "#e8e2d8",
    "axis": "#d9d2c7",
    "plot": "#fbfaf7",
    "teal": "#0f766e",
    "teal_soft": "#d8ece8",
    "teal_line": "#a9cbc5",
    "rose": "#9f3a57",
    "rose_soft": "#f3dbe2",
}


def _style_readout_figure(fig, *, height: int, margin: dict[str, int] | None = None, legend_y: float = 1.12):
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=margin or dict(l=60, r=56, t=42, b=54),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=SHELL_CHART["plot"],
        font=dict(
            family='IBM Plex Sans, Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            size=12,
            color=SHELL_CHART["ink"],
        ),
        bargap=0.34,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=legend_y,
            xanchor='right',
            x=1,
            bgcolor='rgba(251,250,247,0.96)',
            bordercolor='rgba(217,210,199,0.88)',
            borderwidth=1,
            font=dict(size=11, color=SHELL_CHART["ink"]),
        ),
        hoverlabel=dict(bgcolor='#102a2d', font_size=12, font_color='#FFFFFF'),
    )
    fig.update_xaxes(
        gridcolor=SHELL_CHART["grid"],
        zeroline=False,
        linecolor=SHELL_CHART["axis"],
        tickfont=dict(size=11, color=SHELL_CHART["muted"]),
        title_font=dict(size=12, color=SHELL_CHART["muted"]),
        automargin=True,
        ticks="",
        showline=True,
    )
    fig.update_yaxes(
        gridcolor=SHELL_CHART["grid"],
        zeroline=False,
        linecolor=SHELL_CHART["axis"],
        tickfont=dict(size=11, color=SHELL_CHART["muted"]),
        title_font=dict(size=12, color=SHELL_CHART["muted"]),
        automargin=True,
        ticks="",
        showline=True,
    )
    return fig


def render_cohort_dashboard_subtab(lang: str, app_context: dict[str, Any] | None = None):
    """队列仪表板子标签页 - 使用Plotly实现交互式可视化"""
    if app_context is not None:
        _install_app_context(app_context)
    

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    screenshot_mode = _is_screenshot_mode()

    snapshot_subtitle = (
        "One-cohort clinical profile: phenotype burden, baseline distribution, severity anchor, outcome, and loaded-module coverage."
        if lang == 'en' else
        "单一队列的临床画像：表型负担、基线分布、严重程度锚点、结局与已加载模块覆盖度。"
    )

    # 获取入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')

    # ========== Demo模式：复用队列分析顶层的一次性共享演示工作区 ==========
    if entry_mode == 'demo':
        _ensure_cohort_demo_workspace(st.session_state, lang=lang)
        # Shared demo state is announced once at the Cohort Analysis level.

    # ========== Real Data：如果共享工作区已就绪，跳过独立配置 ==========
    elif entry_mode == 'real' and _cohort_real_workspace_ready(st.session_state):
        _sync_real_data_panel_defaults(root_key="dash_data_root", db_key="dash_db_select")

    # ========== Real Data模式：显示数据配置 ==========
    else:
        with st.expander("⚙️ " + ("Data Configuration" if lang == 'en' else "数据配置"), expanded=True):
            # 数据源选择 — 支持3种模式
            _src_label = "Data Source" if lang == 'en' else "数据来源"
            _allow_demo = entry_mode != 'real'
            _src_keys = ["raw", "exported"] + (["demo"] if _allow_demo else [])
            _src_labels = {
                "raw": "📂 Raw Database" if lang == 'en' else "📂 原始数据库",
                "exported": "📦 Previously Exported Results" if lang == 'en' else "📦 之前导出的结果文件",
                "demo": "🧪 Demo Data" if lang == 'en' else "🧪 模拟数据",
            }
            _default_src = "demo" if _allow_demo and entry_mode == 'demo' else "raw"
            dash_src = st.radio(
                _src_label, _src_keys,
                index=_src_keys.index(_default_src),
                format_func=lambda x: _src_labels[x],
                horizontal=True, key="dash_data_source"
            )

            if dash_src == "demo":
                # ===== 模拟数据模式 =====
                load_btn = st.button(
                    "🚀 " + ("Generate Demo Snapshot" if lang == 'en' else "生成演示快照"),
                    type="primary", key="dash_load_demo_btn"
                )
                if load_btn:
                    demo_df = _generate_mock_cohort_dashboard_data(lang)
                    st.session_state['dash_demographics'] = demo_df
                    st.session_state['dash_loaded_db'] = 'Demo'
                    st.session_state['dash_is_demo'] = True
                    st.rerun()

            elif dash_src == "raw":
                # ===== 原始数据库模式 =====
                _sync_real_data_panel_defaults(root_key="dash_data_root", db_key="dash_db_select")
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    data_root = _directory_input(
                        "📁 " + ("ICU Data Root" if lang == 'en' else "ICU数据根目录"),
                        value=st.session_state.get('dash_data_root', ''),
                        input_key="dash_data_root",
                        button_key="dash_data_root_browse",
                        placeholder="/path/to/icudb" if os.name != 'nt' else "D:\\data\\icudb",
                        help="Root directory containing database folders" if lang == 'en' else "包含数据库文件夹的根目录"
                    )
                    render_directory_structure_guide(lang)

                with col2:
                    db_options = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
                    default_db = st.session_state.get('dash_db_select') or _default_real_database()
                    selected_db = st.selectbox(
                        "🏥 " + ("Database" if lang == 'en' else "数据库"),
                        options=list(db_options.keys()),
                        index=list(db_options.keys()).index(default_db) if default_db in db_options else 0,
                        format_func=lambda x: db_options[x],
                        key="dash_db_select"
                    )

                with col3:
                    max_patients = st.number_input(
                        "👥 " + ("Max Patients" if lang == 'en' else "最大患者数"),
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                        key="dash_max_patients"
                    )

                data_root_str = str(data_root or '').strip()
                full_data_path = find_database_path(data_root_str, selected_db) if data_root_str else ''
                path_ok = bool(full_data_path) and os.path.exists(full_data_path)

                if not data_root_str:
                    st.info("ℹ️ " + ("Enter the ICU data root above to validate the database path."
                                     if lang == 'en' else "请在上方填写 ICU 数据根目录以验证数据库路径。"))
                elif path_ok:
                    st.success(f"✅ " + (f"Path valid: `{full_data_path}`" if lang == 'en' else f"路径有效: `{full_data_path}`"))
                else:
                    st.warning(f"⚠️ " + (f"Path not found: `{full_data_path}`" if lang == 'en' else f"路径不存在: `{full_data_path}`"))

                load_btn = st.button(
                    "🚀 " + ("Load Snapshot Data" if lang == 'en' else "加载快照数据"),
                    type="primary",
                    disabled=not path_ok,
                    key="dash_load_btn"
                )

                if load_btn:
                    try:
                        with st.spinner(
                            "Loading snapshot demographics, phenotypes, and SOFA..."
                            if lang == 'en' else "正在加载快照所需的人口统计、表型与 SOFA..."
                        ):
                            ok, msg, bundle = _bundle_from_raw_schema(
                                selected_db,
                                data_root_str,
                                lang=lang,
                                max_patients=int(max_patients),
                                load_concepts=True,
                            )
                            if not ok or bundle is None:
                                st.error(f"❌ {msg}")
                            else:
                                _seed_workspace_state(st.session_state, bundle)
                                st.session_state['dash_data_root'] = data_root_str
                                st.session_state['dash_db_select'] = selected_db
                                st.session_state['dash_loaded_path'] = bundle.resolved_path or full_data_path
                                st.success(f"✅ {msg}")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif dash_src == "exported":
                # ===== 导出文件模式 =====
                col1, col2 = st.columns([3, 2])
                with col1:
                    export_root = _directory_input(
                        "📦 " + ("Folder with Exported Data Files" if lang == 'en' else "存放导出结果文件的文件夹"),
                        value=st.session_state.get('export_path', ''),
                        input_key="dash_export_root",
                        button_key="dash_export_root_browse",
                        placeholder="/path/to/easyicu_export" if os.name != 'nt' else "D:\\easyicu_export",
                        help="Choose the folder that contains EasyICU exported result folders" if lang == 'en' else "选择包含 EasyICU 导出结果子文件夹的目录"
                    )

                _export_folders = _scan_export_folders(export_root if 'export_root' in dir() else st.session_state.get('dash_export_root', ''))

                with col2:
                    if _export_folders:
                        folder_options = {f[0]: f"📁 {f[0]} ({f[1]} files)" for f in _export_folders}
                        selected_folder = st.selectbox(
                            "📁 " + ("Select an Export Result Folder" if lang == 'en' else "选择一批导出结果"),
                            options=list(folder_options.keys()),
                            format_func=lambda x: folder_options[x],
                            key="dash_export_folder"
                        )
                    elif export_root and os.path.isdir(export_root):
                        st.warning("⚠️ " + ("No valid export folders found (need demographics_*.parquet)" if lang == 'en' else "未找到有效的导出文件夹（需要 demographics_*.parquet）"))
                        selected_folder = None
                    else:
                        selected_folder = None

                if _export_folders and selected_folder:
                    selected_path = os.path.join(export_root, selected_folder)
                    st.success(f"✅ `{selected_path}`")

                    load_btn = st.button(
                        "🚀 " + ("Load Exported Result Files" if lang == 'en' else "加载这批导出结果文件"),
                        type="primary",
                        key="dash_load_export_btn"
                    )

                    if load_btn:
                        try:
                            with st.spinner("Loading..." if lang == 'en' else "加载中..."):
                                demographics_df = _load_demographics_from_export(selected_path)
                                _detected_db = demographics_df.attrs.get('detected_db', 'unknown')

                                st.session_state['dash_demographics'] = demographics_df
                                st.session_state['dash_loaded_db'] = _detected_db
                                st.session_state['dash_loaded_path'] = selected_path
                                st.session_state['dash_is_demo'] = False

                            st.success(f"✅ Loaded {len(demographics_df):,} patients from exported result files" if lang == 'en' else f"✅ 已从这批导出结果文件中加载 {len(demographics_df):,} 名患者")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    _render_compact_divider()

    # ========== 仪表板内容 ==========
    if 'dash_demographics' not in st.session_state:
        st.info("👆 " + ("Configure data source and click 'Load' to view the cohort snapshot" if lang == 'en' else "配置数据源并点击'加载'查看队列快照"))
        return

    df = st.session_state['dash_demographics']

    try:
        if not screenshot_mode:
            _render_chart_heading(
                "Snapshot summary" if lang == 'en' else "快照概览",
                snapshot_subtitle,
                "Snapshot" if lang == 'en' else "快照",
            )

        review = _build_cohort_dashboard_review_stats(
            df,
            loaded_concepts=st.session_state.get('loaded_concepts', {}),
            lang=lang,
        )

        # ========== 顶部指标卡片 ==========
        metrics = review['metrics']
        card_specs = [
            (metrics['patients'], "Patients" if lang == 'en' else "患者数", "cohort size" if lang == 'en' else "队列规模", "#2563eb", "N"),
            (metrics['features'], "Loaded features" if lang == 'en' else "已载入特征", "available signal" if lang == 'en' else "可用信号", "#0891b2", "▦"),
            (metrics['median_sofa'], "Median SOFA" if lang == 'en' else "SOFA中位数", "severity anchor" if lang == 'en' else "严重程度锚点", "#0f766e", "⌁"),
            (metrics['phenotype_burden'], "Top phenotype" if lang == 'en' else "最高表型占比", "max prevalence" if lang == 'en' else "最高患病/干预率", "#7c3aed", "◆"),
            (metrics['mortality'], "Mortality" if lang == 'en' else "死亡率", "outcome check" if lang == 'en' else "结局校验", "#e11d48", "↯"),
            (metrics['median_los'], "Median LOS" if lang == 'en' else "LOS中位数", "resource use" if lang == 'en' else "资源占用", "#475569", "▣"),
        ]
        _render_metric_grid(card_specs)

        # ========== 图表行1: 临床表型和严重程度 ==========
        chart_col1, chart_col2 = st.columns([1, 1.08], gap="large")

        with chart_col1:
            _render_chart_heading(
                "Clinical phenotype prevalence" if lang == 'en' else "临床表型占比",
                "Share of patients carrying each phenotype or support signal." if lang == 'en' else "各临床表型或器官支持信号在当前队列中的占比。",
                "Phenotype" if lang == 'en' else "表型",
            )
            phenotype_df = review['phenotype']
            if not phenotype_df.empty:
                plot_df = phenotype_df.sort_values('pct', ascending=True)
                colors = [SHELL_CHART["teal_soft"]] * len(plot_df)
                if colors:
                    colors[-1] = SHELL_CHART["teal"]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=plot_df['pct'],
                        y=plot_df['label'],
                        orientation='h',
                        marker=dict(color=colors, line=dict(color=SHELL_CHART["teal_line"], width=1)),
                        text=plot_df['pct'].map(lambda x: f"{x:.1f}%"),
                        textposition='outside',
                        textfont=dict(size=11, color=SHELL_CHART["ink"]),
                        cliponaxis=False,
                        hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>',
                        name="Prevalence" if lang == 'en' else "占比",
                    )
                )
                _style_readout_figure(
                    fig,
                    height=380,
                    margin=dict(l=118, r=76, t=24, b=58),
                )
                fig.update_layout(showlegend=False)
                fig.update_xaxes(
                    title_text="Prevalence (%)" if lang == 'en' else "占比 (%)",
                    range=[0, max(10, float(plot_df['pct'].max()) * 1.26)],
                )
                fig.update_yaxes(title_text="")
                st.plotly_chart(fig, use_container_width=True, key="dash_phenotype_prevalence", config=_get_plotly_chart_config())
            else:
                st.warning("No clinical phenotype columns found" if lang == 'en' else "未找到临床表型列")

        with chart_col2:
            _render_chart_heading(
                "SOFA severity anchor and outcome" if lang == 'en' else "SOFA严重程度锚点与结局",
                "Patient count by SOFA band with mortality overlaid on a separate scale." if lang == 'en' else "按 SOFA 分层展示患者数，并在独立刻度上叠加死亡率。",
                "Severity" if lang == 'en' else "严重程度",
            )
            severity_df = review['severity']
            if not severity_df.empty and severity_df['patients'].sum() > 0:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(
                        x=severity_df['sofa_group'].astype(str),
                        y=severity_df['patients'],
                        name="Patients" if lang == 'en' else "患者数",
                        marker=dict(
                            color='rgba(15,118,110,0.24)',
                            line=dict(color='rgba(15,118,110,0.54)', width=1),
                        ),
                        text=severity_df['patients'],
                        textposition='outside',
                        textfont=dict(size=11, color=SHELL_CHART["ink"]),
                        cliponaxis=False,
                        hovertemplate='%{x}<br>%{y} patients<extra></extra>' if lang == 'en' else '%{x}<br>%{y} 名患者<extra></extra>',
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=severity_df['sofa_group'].astype(str),
                        y=severity_df['mortality'],
                        name="Mortality %" if lang == 'en' else "死亡率 %",
                        mode='lines+markers+text',
                        text=severity_df['mortality'].map(lambda x: f"{x:.1f}%"),
                        textposition='top center',
                        textfont=dict(size=11, color=SHELL_CHART["rose"]),
                        marker=dict(
                            color=SHELL_CHART["plot"],
                            size=8,
                            line=dict(color=SHELL_CHART["rose"], width=2),
                        ),
                        line=dict(width=2.4, color=SHELL_CHART["rose"], shape='spline', smoothing=0.45),
                        cliponaxis=False,
                        hovertemplate='%{x}<br>%{y:.1f}% mortality<extra></extra>' if lang == 'en' else '%{x}<br>%{y:.1f}% 死亡率<extra></extra>',
                    ),
                    secondary_y=True,
                )
                _style_readout_figure(
                    fig,
                    height=380,
                    margin=dict(l=58, r=70, t=52, b=58),
                    legend_y=1.14,
                )
                fig.update_yaxes(
                    title_text="Patients" if lang == 'en' else "患者数",
                    secondary_y=False,
                    range=[0, max(5, float(severity_df['patients'].max()) * 1.28)],
                )
                fig.update_yaxes(
                    title_text="Mortality %" if lang == 'en' else "死亡率 %",
                    secondary_y=True,
                    range=[0, 100],
                    showgrid=False,
                )
                fig.update_xaxes(title_text="SOFA group" if lang == 'en' else "SOFA分层")
                st.plotly_chart(fig, use_container_width=True, key="dash_severity_outcome", config=_get_plotly_chart_config())
            else:
                st.warning("No SOFA severity column found" if lang == 'en' else "未找到SOFA严重程度列")

        reclass = review.get('reclassification') or {}
        if reclass.get('available'):
            discordant_pct = reclass.get('metrics', {}).get('discordant_pct', '')
            teaser_en = (
                f"Under SOFA-2, {discordant_pct} of patients reclassify. "
                "Open the **SOFA reclassification** panel for the matrix, organ contributors, and mortality breakdown."
            )
            teaser_zh = (
                f"在 SOFA-2 下，共 {discordant_pct} 的患者发生重新分层。"
                "请切换到 **SOFA 重分层** 面板查看重分类矩阵、器官贡献度与死亡率分解。"
            )
            st.info(teaser_en if lang == 'en' else teaser_zh, icon="🧭")

    except Exception as e:
        st.error(f"Render error: {e}")
        import traceback
        st.code(traceback.format_exc())
