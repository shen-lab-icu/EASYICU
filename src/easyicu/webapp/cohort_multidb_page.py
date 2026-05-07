"""Cohort multi-database distribution subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_multidb_distribution_subtab', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_multidb_distribution_subtab(lang: str, app_context: dict[str, Any] | None = None):
    """多数据库特征分布对比子标签页"""
    if app_context is not None:
        _install_app_context(app_context)
    
    import plotly.graph_objects as go
    screenshot_mode = _is_screenshot_mode()

    title = "Cross-Database Benchmark" if lang == 'en' else "跨库分布基准"
    subtitle = (
        "Figure 3-style comparison of harmonized feature distributions across ICU databases; kept separate from the S1 cohort audit."
        if lang == 'en' else
        "对应 Figure 3 风格的跨 ICU 数据库标准化特征分布对照；与补充图 S1 的队列审计保持分工。"
    )
    if not screenshot_mode:
        st.markdown(f"""
        <div style="margin-bottom:14px">
            <div style="font-size:1.15rem;font-weight:850;color:#0b1f44">📈 {title}</div>
            <div style="font-size:.86rem;color:#60718a;margin-top:2px">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    # 获取入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')

    # ========== Demo模式：复用队列分析顶层的一次性共享演示工作区 ==========
    if entry_mode == 'demo':
        _ensure_cohort_demo_workspace(st.session_state, lang=lang)
        # Shared demo state is announced once at the Cohort Analysis level.

    # ========== Real Data模式 ==========
    if entry_mode != 'demo':
        # 如果共享工作区已就绪，确保路径/数据库同步
        if _cohort_real_workspace_ready(st.session_state):
            _sync_real_data_panel_defaults(root_key="multidb_data_root", multi_db_key="multidb_selected")
        else:
            # 配置区域
            _sync_real_data_panel_defaults(root_key="multidb_data_root", multi_db_key="multidb_selected")
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            data_root = _directory_input(
                "🗂️ " + ("ICU Data Root" if lang == 'en' else "ICU数据根目录"),
                value=st.session_state.get('multidb_data_root', ''),
                input_key="multidb_data_root",
                button_key="multidb_data_root_browse",
                placeholder="/path/to/icudb" if os.name != 'nt' else "D:\\data\\icudb",
                help="Root directory containing database folders (mimiciv, eicu, aumc, hirid)" if lang == 'en' else "包含数据库文件夹的根目录"
            )
            # 添加目录结构指南
            render_directory_structure_guide(lang)

        with col2:
            # 数据库选择
            db_options = ['miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic']
            db_labels = {'miiv': 'MIMIC-IV 🟢', 'eicu': 'eICU 🟠', 'aumc': 'Amsterdam 🔵', 'hirid': 'HiRID 🔴', 'mimic': 'MIMIC-III 🟣', 'sic': 'SICdb ⚫'}
            default_dbs = st.session_state.get('multidb_selected') or [_default_real_database()]
            selected_dbs = st.multiselect(
                "🏥 " + ("Databases" if lang == 'en' else "数据库"),
                options=db_options,
                default=[db for db in default_dbs if db in db_options] or ['miiv'],
                format_func=lambda x: db_labels.get(x, x),
                key="multidb_selected"
            )

        with col3:
            max_patients = st.number_input(
                "👥 " + ("Max Patients" if lang == 'en' else "最大患者数"),
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                key="multidb_max_patients"
            )

        # 特征选择
        feature_groups = {
            "Vital Signs": ['hr', 'sbp', 'dbp', 'map', 'resp', 'temp', 'o2sat'],
            "Laboratory": ['glu', 'na', 'k', 'crea', 'bili', 'lact'],
            "Hematology": ['hgb', 'plt', 'wbc'],
            "Blood Gas": ['ph', 'po2', 'pco2', 'fio2'],
            "SOFA-2 Scores": ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
        }

        col1, col2 = st.columns([1, 3])
        with col1:
            selected_group = st.selectbox(
                "📋 " + ("Feature Group" if lang == 'en' else "特征分组"),
                options=list(feature_groups.keys()),
                key="multidb_group"
            )

        with col2:
            available_features = feature_groups.get(selected_group, [])
            selected_features = st.multiselect(
                "🔬 " + ("Select Features" if lang == 'en' else "选择特征"),
                options=available_features,
                default=available_features[:4],
                key="multidb_features"
            )

        # 加载按钮
        load_btn = st.button(
            "🚀 " + ("Load & Generate" if lang == 'en' else "加载并生成"),
            type="primary",
            key="multidb_load"
        )

        _render_compact_divider()

        if load_btn and selected_dbs and selected_features:
            try:
                from easyicu.cohort_visualization import MultiDatabaseDistribution

                with st.spinner("Loading data from databases..." if lang == 'en' else "正在从数据库加载数据..."):
                    mdd = MultiDatabaseDistribution(data_root=data_root, language=lang)
                    data = mdd.load_feature_data(
                        concepts=selected_features,
                        databases=selected_dbs,
                        max_patients=max_patients,
                    )
                    st.session_state['multidb_data'] = data
                    st.session_state['multidb_concepts'] = selected_features
                    st.session_state['multidb_is_demo'] = False
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return

    # 显示结果
    if 'multidb_data' in st.session_state and st.session_state.get('multidb_data'):
        data = st.session_state['multidb_data']
        concepts = st.session_state.get('multidb_concepts', ['hr', 'sbp', 'temp', 'resp'])
        if screenshot_mode:
            # Paper figures need the distribution signal, not an exhaustive grid.
            screenshot_priority = ['hr', 'sbp', 'map', 'resp', 'temp', 'spo2', 'crea', 'lact']
            prioritized = [concept for concept in screenshot_priority if concept in concepts]
            concepts = prioritized or concepts[:8]

        # 数据量统计
        stat_cols = st.columns(len(data))
        db_colors = {'miiv': '🟢', 'eicu': '🟠', 'aumc': '🔵', 'hirid': '🔴', 'mimic': '🟣', 'sic': '⚫'}
        for i, (db, df) in enumerate(data.items()):
            with stat_cols[i]:
                st.metric(
                    label=f"{db_colors.get(db, '')} {db.upper()}",
                    value=f"{len(df):,}",
                    delta="records"
                )

        # 生成分布图
        try:
            from easyicu.cohort_visualization import MultiDatabaseDistribution
            # Demo模式使用默认路径
            _data_root = st.session_state.get('multidb_data_root', os.environ.get('EASYICU_DATA_PATH', ''))
            mdd = MultiDatabaseDistribution(data_root=_data_root, language=lang)

            # 网格图
            n_cols = min(4, len(concepts))
            fig = mdd.create_distribution_grid(data, concepts, cols=n_cols)
            if not screenshot_mode:
                fig.update_layout(
                    title_text="Multi-Database Feature<br>Distribution Comparison",
                    margin=dict(t=132, b=62, l=72, r=24),
                )
            st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

            if screenshot_mode:
                return

            # 单特征详细对比
            _render_compact_divider()
            st.markdown("#### " + ("Detailed Single Feature View" if lang == 'en' else "单特征详细视图"))

            selected_single = st.selectbox(
                "Select feature" if lang == 'en' else "选择特征",
                options=concepts,
                key="multidb_single_feature"
            )

            if selected_single:
                fig_single, stats_df = mdd.create_single_feature_comparison(data, selected_single)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig_single, use_container_width=True, config=_get_plotly_chart_config())
                with col2:
                    st.markdown("**Statistics**" if lang == 'en' else "**统计信息**")
                    _st_dataframe_compat(
                        st,
                        stats_df.style.format({
                            'Mean': '{:.2f}',
                            'Std': '{:.2f}',
                            'Median': '{:.2f}',
                            'Q25': '{:.2f}',
                            'Q75': '{:.2f}',
                        }),
                        width='stretch',
                        hide_index=True
                    )
        except Exception as e:
            st.error(f"Error generating chart: {e}")
    else:
        # 占位提示
        st.info(
            "👆 Select databases and features, then click 'Load & Generate'"
            if lang == 'en' else
            "👆 选择数据库和特征，然后点击'加载并生成'"
        )
