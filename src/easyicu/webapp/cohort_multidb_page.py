"""Cohort multi-database distribution subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from pathlib import Path
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

    title = "Cross-Database Benchmark" if lang == 'en' else "跨数据库分布对比"
    subtitle = (
        "Comparison of harmonized feature distributions across ICU databases; kept separate from the cohort audit."
        if lang == 'en' else
        "比较多个 ICU 数据库中标准化特征的分布；与队列统计保持分工。"
    )
    if not screenshot_mode:
        st.markdown(f"""
        <div class="eu-crossdb-distribution-heading">
            <div class="eu-crossdb-distribution-title">{html.escape(title)}</div>
            <div class="eu-crossdb-distribution-subtitle">{html.escape(subtitle)}</div>
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
        # Exports-only workspace covers a single database. Cross-DB needs ≥2
        # DBs to actually compare — but the original config form below still
        # works if the user has a multi-DB root on disk. Show a helpful hint,
        # not a wall.
        if st.session_state.get('_cohort_real_ws_origin') == 'loaded_exports':
            db_label_for_msg = {
                'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC',
                'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb',
            }.get(st.session_state.get('_cohort_real_ws_db', ''), '')

            # Best-effort: detect sibling DB folders next to the sidebar path.
            sidebar_path = st.session_state.get('data_path') or ''
            sibling_dbs: list[str] = []
            sibling_root: str = ''
            if sidebar_path:
                try:
                    p = Path(sidebar_path)
                    parent = p if p.is_dir() and any(
                        (p / name).is_dir() for name in
                        ('mimiciv', 'mimic-iv', 'eicu', 'eicu-crd', 'aumc',
                         'amsterdamumcdb', 'hirid', 'mimic', 'mimiciii', 'sic')
                    ) else p.parent
                    if parent.is_dir():
                        for name in parent.iterdir():
                            if not name.is_dir():
                                continue
                            lower = name.name.lower()
                            for tag in ('mimiciv', 'mimic-iv', 'mimic_iv',
                                        'eicu', 'aumc', 'amsterdamumcdb',
                                        'hirid', 'mimiciii', 'mimic-iii',
                                        'mimic_iii', 'mimic3', 'sic'):
                                if tag in lower:
                                    sibling_dbs.append(name.name)
                                    break
                        if len(sibling_dbs) >= 2:
                            sibling_root = str(parent)
                except (OSError, PermissionError, ValueError):
                    pass

            if lang == 'en':
                base = (
                    f"💡 **Cross-DB compares multiple databases.** Your loaded "
                    f"module exports cover only **{db_label_for_msg or 'one database'}**, "
                    "so a fair cross-database comparison needs additional DB roots. "
                    "Fill **ICU Data Root** below with a folder that contains "
                    "**≥2 database subfolders** (e.g. `mimiciv/`, `eicu/`, `aumc/`), "
                    "pick the databases in **Databases**, then click **Load & Generate**."
                )
                if sibling_dbs:
                    base += (
                        f"\n\n🔎 Detected near your sidebar path: "
                        f"`{sibling_root}` with subfolders "
                        + ", ".join(f"`{n}`" for n in sibling_dbs[:6])
                        + ". You can paste that into **ICU Data Root** below."
                    )
                st.info(base)
            else:
                base = (
                    f"💡 **跨数据库对比需要至少两个数据库。** 你加载的模块导出只覆盖 "
                    f"**{db_label_for_msg or '单个数据库'}**，因此还需要其它数据库根目录。"
                    "请在下方 **ICU 数据根目录** 中填入包含 "
                    "**≥2 个数据库子目录**（如 `mimiciv/`、`eicu/`、`aumc/`）的文件夹，"
                    "在 **数据库** 中勾选要对比的库，然后点击 **加载并生成**。"
                )
                if sibling_dbs:
                    base += (
                        f"\n\n🔎 在你侧边栏路径附近发现：`{sibling_root}` 包含子目录 "
                        + "、".join(f"`{n}`" for n in sibling_dbs[:6])
                        + "。可直接粘贴到下方 **ICU 数据根目录**。"
                    )
                st.info(base)
            _render_compact_divider()
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
            multiselect_kwargs = {
                "label": "🏥 " + ("Databases" if lang == 'en' else "数据库"),
                "options": db_options,
                "format_func": lambda x: db_labels.get(x, x),
                "key": "multidb_selected",
            }
            if 'multidb_selected' not in st.session_state:
                default_dbs = [_default_real_database()]
                multiselect_kwargs["default"] = [db for db in default_dbs if db in db_options] or ['miiv']
            selected_dbs = st.multiselect(**multiselect_kwargs)

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

        # When the user picks ≥6 features the 4×N grid becomes very dense
        # and individual subplots get tiny. Point them at the dedicated
        # "Detailed Single Feature View" further down the page instead
        # (2026-05 Phase D polish).
        if len(selected_features) >= 6:
            st.caption(
                (f"💡 You selected {len(selected_features)} features — the grid will get crowded. "
                 "For a focused side-by-side comparison, scroll down to "
                 "**Detailed Single Feature View** after generation."
                 if lang == 'en' else
                 f"💡 你选择了 {len(selected_features)} 个特征——网格会比较密集。"
                 "若想专注对比，请在生成后滚动到下方的 **单特征详细视图**。")
            )

        # 加载按钮
        load_btn = st.button(
            "🚀 " + ("Load & Generate" if lang == 'en' else "加载并生成"),
            type="primary",
            key="multidb_load"
        )

        _render_compact_divider()

        if load_btn:
            if len(selected_dbs or []) < 2:
                st.session_state.pop('multidb_data', None)
                st.session_state.pop('multidb_concepts', None)
                st.session_state.pop('multidb_is_demo', None)
                st.warning(
                    "Select at least two databases before loading a cross-database comparison."
                    if lang == 'en' else
                    "请至少选择两个数据库后再加载跨数据库对比。"
                )
            elif not selected_features:
                st.warning(
                    "Select at least one feature before loading."
                    if lang == 'en' else
                    "请至少选择一个特征后再加载。"
                )
            else:
                try:
                    from easyicu.cohort_visualization import MultiDatabaseDistribution

                    with st.spinner("Loading data from databases..." if lang == 'en' else "正在从数据库加载数据..."):
                        mdd = MultiDatabaseDistribution(data_root=data_root, language=lang)
                        data = mdd.load_feature_data(
                            concepts=selected_features,
                            databases=selected_dbs,
                            max_patients=max_patients,
                        )
                    if len(data or {}) < 2:
                        st.session_state.pop('multidb_data', None)
                        st.session_state.pop('multidb_concepts', None)
                        st.session_state.pop('multidb_is_demo', None)
                        st.warning(
                            "Loaded fewer than two databases. Check that ICU Data Root contains separate folders for the selected databases."
                            if lang == 'en' else
                            "实际加载到的数据库少于两个。请检查 ICU 数据根目录是否包含所选数据库各自的子文件夹。"
                        )
                    else:
                        st.session_state['multidb_data'] = data
                        st.session_state['multidb_concepts'] = selected_features
                        st.session_state['multidb_is_demo'] = False
                except Exception as e:
                    st.session_state.pop('multidb_data', None)
                    st.session_state.pop('multidb_concepts', None)
                    st.session_state.pop('multidb_is_demo', None)
                    st.error(f"Error loading data: {e}")
                    return

    # 显示结果
    if 'multidb_data' in st.session_state and st.session_state.get('multidb_data'):
        data = st.session_state['multidb_data']
        concepts = st.session_state.get('multidb_concepts', ['hr', 'sbp', 'temp', 'resp'])
        is_demo_preview = bool(st.session_state.get('multidb_is_demo')) or entry_mode == 'demo'
        if is_demo_preview:
            concepts = [concept for concept in concepts if concept in {'hr', 'sbp', 'map', 'temp', 'spo2', 'lact'}][:6]
            trimmed_data = {}
            for db, df in list(data.items())[:6]:
                if hasattr(df, "loc") and 'concept' in getattr(df, "columns", []):
                    trimmed_data[db] = df.loc[df['concept'].isin(concepts)].head(320).copy()
                else:
                    trimmed_data[db] = df
            data = trimmed_data
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
            if is_demo_preview and not screenshot_mode:
                st.caption(
                    "Demo preview uses a compact sample so the page opens quickly; real data mode keeps the full selectable comparison."
                    if lang == 'en' else
                    "演示预览只加载紧凑样本，保证页面快速打开；真实数据模式仍保留完整可选对比。"
                )

            # 网格图
            n_cols = min(4, len(concepts))
            fig = mdd.create_distribution_grid(data, concepts, cols=n_cols)
            if not screenshot_mode:
                fig.update_layout(
                    title=dict(
                        text="Multi-Database Feature Distribution Comparison",
                        x=0.5,
                        xanchor="center",
                        y=0.985,
                        yanchor="top",
                        font=dict(size=18),
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.08,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=12, color="black"),
                    ),
                    margin=dict(t=80, b=110, l=72, r=24),
                )
            st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

            if screenshot_mode:
                return
            if is_demo_preview:
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
