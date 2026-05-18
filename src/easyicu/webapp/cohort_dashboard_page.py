"""Cohort dashboard subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_cohort_dashboard_subtab', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_cohort_dashboard_subtab(lang: str, app_context: dict[str, Any] | None = None):
    """队列仪表板子标签页 - 使用Plotly实现交互式可视化"""
    if app_context is not None:
        _install_app_context(app_context)
    

    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    screenshot_mode = _is_screenshot_mode()

    snapshot_title = "Cohort Snapshot" if lang == 'en' else "队列快照"
    snapshot_subtitle = (
        "One-cohort clinical profile: phenotype burden, baseline distribution, severity anchor, outcome, and loaded-module coverage."
        if lang == 'en' else
        "单一队列的临床画像：表型负担、基线分布、严重程度锚点、结局与已加载模块覆盖度。"
    )
    if not screenshot_mode:
        st.markdown("### 🎯 " + snapshot_title)
        st.caption(snapshot_subtitle)

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
                        from easyicu.patient_filter import PatientFilter

                        with st.spinner("Loading demographics..." if lang == 'en' else "正在加载..."):
                            pf = PatientFilter(database=selected_db, data_path=full_data_path)
                            demographics_df = pf._load_demographics()

                            if len(demographics_df) > max_patients:
                                demographics_df = demographics_df.head(max_patients)

                            st.session_state['dash_demographics'] = demographics_df
                            st.session_state['dash_loaded_db'] = selected_db
                            st.session_state['dash_loaded_path'] = full_data_path
                            st.session_state['dash_is_demo'] = False

                        st.success(f"✅ Loaded {len(demographics_df):,} patients" if lang == 'en' else f"✅ 已加载 {len(demographics_df):,} 名患者")
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
        review = _build_cohort_dashboard_review_stats(
            df,
            loaded_concepts=st.session_state.get('loaded_concepts', {}),
            lang=lang,
        )

        # ========== 顶部指标卡片 ==========
        st.markdown("#### " + ("📊 Cohort Snapshot Summary" if lang == 'en' else "📊 队列快照摘要"))

        metric_cols = st.columns(6)

        def metric_card(value, label, hint, accent, icon):
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #cddbeb;border-left:4px solid {accent};
                        padding:10px 11px;border-radius:16px;color:#0b1f44;min-height:94px;
                        display:flex;flex-direction:column;justify-content:center;box-shadow:0 8px 24px rgba(15,31,68,.045)">
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
                    <span style="width:24px;height:24px;border-radius:7px;background:{accent};color:#fff;display:inline-flex;align-items:center;justify-content:center;font-size:.82rem;font-weight:900">{icon}</span>
                    <span style="font-size:.66rem;font-weight:850;color:#60718a;letter-spacing:.07em;text-transform:uppercase">{label}</span>
                </div>
                <div style="font-size:1.55rem;font-weight:900;line-height:1.05;color:#0b1f44;letter-spacing:-.02em">{value}</div>
                <div style="font-size:.66rem;color:#60718a;margin-top:4px;font-weight:700">{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        metrics = review['metrics']
        card_specs = [
            (metrics['patients'], "Patients" if lang == 'en' else "患者数", "cohort size" if lang == 'en' else "队列规模", "#2563eb", "👥"),
            (metrics['features'], "Loaded features" if lang == 'en' else "已载入特征", "available signal" if lang == 'en' else "可用信号", "#0891b2", "▦"),
            (metrics['median_sofa'], "Median SOFA" if lang == 'en' else "SOFA中位数", "severity anchor" if lang == 'en' else "严重程度锚点", "#0f766e", "⌁"),
            (metrics['phenotype_burden'], "Top phenotype" if lang == 'en' else "最高表型占比", "max prevalence" if lang == 'en' else "最高患病/干预率", "#7c3aed", "◆"),
            (metrics['mortality'], "Mortality" if lang == 'en' else "死亡率", "outcome check" if lang == 'en' else "结局校验", "#e11d48", "↯"),
            (metrics['median_los'], "Median LOS" if lang == 'en' else "LOS中位数", "resource use" if lang == 'en' else "资源占用", "#475569", "▣"),
        ]
        with metric_cols[0]:
            metric_card(*card_specs[0])
        with metric_cols[1]:
            metric_card(*card_specs[1])
        with metric_cols[2]:
            metric_card(*card_specs[2])
        with metric_cols[3]:
            metric_card(*card_specs[3])
        with metric_cols[4]:
            metric_card(*card_specs[4])
        with metric_cols[5]:
            metric_card(*card_specs[5])

        _render_compact_divider()

        # ========== 图表行1: 临床表型和严重程度 ==========
        chart_col1, chart_col2 = st.columns([1, 1.15])

        with chart_col1:
            st.markdown("##### " + ("Clinical Phenotype Prevalence" if lang == 'en' else "临床表型占比"))
            phenotype_df = review['phenotype']
            if not phenotype_df.empty:
                fig = px.bar(
                    phenotype_df,
                    x='pct',
                    y='label',
                    orientation='h',
                    text=phenotype_df['pct'].map(lambda x: f"{x:.1f}%"),
                    color='pct',
                    color_continuous_scale=['#dbeafe', '#0f766e'],
                    labels={'pct': "Prevalence (%)" if lang == 'en' else "占比 (%)", 'label': ""},
                    template='plotly_white',
                )
                fig.update_traces(textposition='outside', cliponaxis=False)
                fig.update_layout(
                    height=330,
                    margin=dict(l=10, r=40, t=12, b=30),
                    coloraxis_showscale=False,
                    font=dict(size=13, color='#111827'),
                )
                fig.update_xaxes(range=[0, max(10, float(phenotype_df['pct'].max()) * 1.18)], gridcolor='#e5e7eb')
                st.plotly_chart(fig, use_container_width=True, key="dash_phenotype_prevalence", config=_get_plotly_chart_config())
            else:
                st.warning("No clinical phenotype columns found" if lang == 'en' else "未找到临床表型列")

        with chart_col2:
            st.markdown("##### " + ("SOFA Severity Anchor & Outcome" if lang == 'en' else "SOFA严重程度锚点与结局"))
            severity_df = review['severity']
            if not severity_df.empty and severity_df['patients'].sum() > 0:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(
                        x=severity_df['sofa_group'].astype(str),
                        y=severity_df['patients'],
                        name="Patients" if lang == 'en' else "患者数",
                        marker_color='rgba(37, 99, 235, 0.55)',
                        text=severity_df['patients'],
                        textposition='outside',
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
                        marker_color='#e11d48',
                        line=dict(width=3),
                    ),
                    secondary_y=True,
                )
                fig.update_layout(
                    template='plotly_white',
                    height=330,
                    margin=dict(l=20, r=20, t=12, b=35),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    font=dict(size=13, color='#111827'),
                )
                fig.update_yaxes(title_text="Patients" if lang == 'en' else "患者数", secondary_y=False, gridcolor='#e5e7eb')
                fig.update_yaxes(title_text="Mortality %" if lang == 'en' else "死亡率 %", secondary_y=True, range=[0, 100])
                fig.update_xaxes(title_text="SOFA group" if lang == 'en' else "SOFA分层")
                st.plotly_chart(fig, use_container_width=True, key="dash_severity_outcome", config=_get_plotly_chart_config())
            else:
                st.warning("No SOFA severity column found" if lang == 'en' else "未找到SOFA严重程度列")

        # 图表行 2 被删除（2026-05 Phase C 去重）：
        #   - Baseline Distributions (Age + LOS days) 移到 Patient Review →
        #     Patient Overview（避免与个人级 trend 重复）。
        #   - Data Coverage by Module 移到 Cohort Statistics → Coverage tab
        #     （那里是 coverage 的主页，避免双份）。
        # 这里仅保留指向 SOFA Δ 的 1 行 teaser，让 Snapshot 真正变成
        # "one-page cohort profile" 而不是杂烩。
        reclass = review.get('reclassification') or {}
        if reclass.get('available'):
            discordant_pct = reclass.get('metrics', {}).get('discordant_pct', '')
            teaser_en = (
                f"Under SOFA-2, {discordant_pct} of patients reclassify — open the "
                "**SOFA-1 vs SOFA-2** tab for the matrix, organ contributors, and mortality breakdown."
            )
            teaser_zh = (
                f"在 SOFA-2 下，共 {discordant_pct} 的患者发生重新分层 —— "
                "切换到 **SOFA-1 vs SOFA-2** 标签查看重分类矩阵、器官贡献度与死亡率。"
            )
            st.info(teaser_en if lang == 'en' else teaser_zh, icon="🧭")

    except Exception as e:
        st.error(f"Render error: {e}")
        import traceback
        st.code(traceback.format_exc())
