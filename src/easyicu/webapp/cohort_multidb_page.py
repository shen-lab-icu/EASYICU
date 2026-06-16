"""Cohort multi-database distribution subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


_DB_FOLDER_HINTS = (
    'mimiciv', 'mimic-iv', 'mimic_iv',
    'eicu', 'eicu-crd',
    'aumc', 'amsterdamumcdb',
    'hirid',
    'mimiciii', 'mimic-iii', 'mimic_iii', 'mimic3',
    'sic',
)


def _detect_sibling_database_root(sidebar_path: str) -> tuple[str, list[str]]:
    """Return a nearby root path and DB-like child folders, if any."""
    if not sidebar_path:
        return "", []
    try:
        p = Path(sidebar_path)
        parent = p if p.is_dir() and any(
            (p / name).is_dir() for name in _DB_FOLDER_HINTS
        ) else p.parent
        if not parent.is_dir():
            return "", []
        sibling_dbs: list[str] = []
        for name in parent.iterdir():
            if not name.is_dir():
                continue
            lower = name.name.lower()
            if any(tag in lower for tag in _DB_FOLDER_HINTS):
                sibling_dbs.append(name.name)
        return (str(parent), sibling_dbs) if sibling_dbs else ("", [])
    except (OSError, PermissionError, ValueError):
        return "", []


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_multidb_distribution_subtab', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _render_crossdb_micro_heading(kicker: str, title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<p>{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        '<div class="eu-crossdb-micro-heading">'
        f'<span>{html.escape(kicker)}</span>'
        f'<b>{html.escape(title)}</b>'
        f'{subtitle_html}'
        '</div>',
        unsafe_allow_html=True,
    )


def _crossdb_loader_notice_html(tone: str, kicker: str, title: str, body: str = "", meta: str = "") -> str:
    body_html = f'<p>{html.escape(body)}</p>' if body else ""
    meta_html = f'<em>{html.escape(meta)}</em>' if meta else ""
    return (
        f'<div class="eu-crossdb-loader-notice {html.escape(tone)}">'
        f'<span>{html.escape(kicker)}</span>'
        f'<b>{html.escape(title)}</b>'
        f'{body_html}'
        f'{meta_html}'
        '</div>'
    )


def _render_crossdb_loader_notice(tone: str, kicker: str, title: str, body: str = "", meta: str = "") -> None:
    st.markdown(
        _crossdb_loader_notice_html(tone, kicker, title, body, meta),
        unsafe_allow_html=True,
    )


def _render_crossdb_metric_grid(items: list[tuple[str, str, str, str]]) -> None:
    cards_html = "".join(
        (
            f'<div class="eu-crossdb-metric-card {html.escape(tone)}">'
            f'<span>{html.escape(label)}</span>'
            f'<b>{html.escape(value)}</b>'
            f'<em>{html.escape(note)}</em>'
            '</div>'
        )
        for label, value, note, tone in items
    )
    st.markdown(
        f'<div class="eu-crossdb-metric-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def _render_crossdb_db_cards(
    db_options: list[str],
    db_labels: dict[str, str],
    selected_dbs: list[str] | tuple[str, ...] | None,
    *,
    lang: str,
) -> None:
    selected = set(selected_dbs or [])
    add_label = "add" if lang == "en" else "添加"
    selected_label = "selected" if lang == "en" else "已选择"
    rows = []
    for db_id in db_options:
        is_selected = db_id in selected
        rows.append(
            '<div class="eu-crossdb-db-card '
            f'{"selected" if is_selected else "available"}">'
            f'<span>{html.escape(db_id.upper())}</span>'
            f'<b>{html.escape(db_labels.get(db_id, db_id))}</b>'
            f'<em>{html.escape(selected_label if is_selected else add_label)}</em>'
            '</div>'
        )
    st.markdown(
        '<div class="eu-crossdb-db-grid">'
        + "".join(rows)
        + '</div>',
        unsafe_allow_html=True,
    )


def _crossdb_setup_contract_html(
    *,
    data_root: str,
    selected_dbs: list[str] | tuple[str, ...] | None,
    db_labels: dict[str, str],
    selected_group: str,
    selected_features: list[str] | tuple[str, ...] | None,
    max_patients: int,
    lang: str,
) -> str:
    db_count = len(selected_dbs or [])
    feature_count = len(selected_features or [])
    root_ready = bool(str(data_root or "").strip())
    db_ready = db_count >= 2
    feature_ready = feature_count > 0
    load_ready = root_ready and db_ready and feature_ready
    selected_names = ", ".join(db_labels.get(db, db) for db in (selected_dbs or [])) or (
        "none selected" if lang == "en" else "尚未选择"
    )
    feature_names = ", ".join(selected_features or []) or (
        "none selected" if lang == "en" else "尚未选择"
    )
    nodes = [
        (
            "01",
            "Source root" if lang == "en" else "数据根目录",
            "ready" if root_ready else "pending",
            data_root or ("waiting for local folder" if lang == "en" else "等待本地文件夹"),
        ),
        (
            "02",
            "Database selection" if lang == "en" else "数据库选择",
            "ready" if db_ready else "warning",
            (
                f"{db_count} selected · {selected_names}"
                if lang == "en" else
                f"已选择 {db_count} 个 · {selected_names}"
            ),
        ),
        (
            "03",
            "Feature scope" if lang == "en" else "特征范围",
            "ready" if feature_ready else "warning",
            (
                f"{selected_group} · {feature_count} concepts · {feature_names}"
                if lang == "en" else
                f"{selected_group} · {feature_count} 个概念 · {feature_names}"
            ),
        ),
        (
            "04",
            "Load gate" if lang == "en" else "加载门禁",
            "ready" if load_ready else "pending",
            (
                f"ready up to {int(max_patients):,} patients"
                if load_ready and lang == "en" else
                f"最多 {int(max_patients):,} 位患者，已就绪"
                if load_ready else
                "needs root, >=2 databases, and >=1 concept"
                if lang == "en" else
                "需要根目录、至少两个数据库和至少一个概念"
            ),
        ),
    ]
    nodes_html = "".join(
        (
            f'<div class="eu-crossdb-setup-node {html.escape(tone)}">'
            f'<span>{html.escape(num)}</span>'
            f'<div><b>{html.escape(title)}</b><p>{html.escape(copy)}</p></div>'
            '</div>'
        )
        for num, title, tone, copy in nodes
    )
    return (
        '<div class="eu-crossdb-setup-contract">'
        f'<div class="eu-crossdb-setup-head"><span>{html.escape("Run contract" if lang == "en" else "运行合同")}</span>'
        f'<b>{html.escape("Cross-database loader readiness" if lang == "en" else "跨数据库加载就绪度")}</b></div>'
        f'<div class="eu-crossdb-setup-grid">{nodes_html}</div>'
        '</div>'
    )


def _render_crossdb_setup_contract(
    *,
    data_root: str,
    selected_dbs: list[str] | tuple[str, ...] | None,
    db_labels: dict[str, str],
    selected_group: str,
    selected_features: list[str] | tuple[str, ...] | None,
    max_patients: int,
    lang: str,
) -> None:
    st.markdown(
        _crossdb_setup_contract_html(
            data_root=data_root,
            selected_dbs=selected_dbs,
            db_labels=db_labels,
            selected_group=selected_group,
            selected_features=selected_features,
            max_patients=max_patients,
            lang=lang,
        ),
        unsafe_allow_html=True,
    )


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
            sibling_root, sibling_dbs = _detect_sibling_database_root(sidebar_path)

            if lang == 'en':
                base = (
                    f"Your loaded module exports cover only {db_label_for_msg or 'one database'}. "
                    "For a fair cross-database comparison, enter an ICU Data Root that contains "
                    "at least two database subfolders, select the databases, then click Load & Generate."
                )
                detected = (
                    f"Detected near sidebar path: {sibling_root} · "
                    + ", ".join(sibling_dbs[:6])
                ) if sibling_dbs else ""
                _render_crossdb_loader_notice(
                    "warning",
                    "Source scope",
                    "Additional database roots required",
                    base,
                    detected,
                )
            else:
                base = (
                    f"你加载的模块导出只覆盖 {db_label_for_msg or '单个数据库'}。"
                    "要做公平的跨数据库对比，请在 ICU 数据根目录中填入包含至少两个数据库子目录的文件夹，"
                    "勾选数据库后点击加载并生成。"
                )
                detected = (
                    f"在侧边栏路径附近发现：{sibling_root} · "
                    + "、".join(sibling_dbs[:6])
                ) if sibling_dbs else ""
                _render_crossdb_loader_notice(
                    "warning",
                    "来源范围",
                    "需要更多数据库根目录",
                    base,
                    detected,
                )
            _render_compact_divider()
        # 如果共享工作区已就绪，确保路径/数据库同步
        if _cohort_real_workspace_ready(st.session_state):
            _sync_real_data_panel_defaults(root_key="multidb_data_root", multi_db_key="multidb_selected")
        else:
            # 配置区域
            _sync_real_data_panel_defaults(root_key="multidb_data_root", multi_db_key="multidb_selected")
        _render_crossdb_micro_heading(
            "Operational loader" if lang == 'en' else "操作加载器",
            "ICU Data Root + database selection" if lang == 'en' else "ICU 数据根目录与数据库选择",
            "Use this panel only when you have at least two database folders available for a fair cross-database comparison."
            if lang == 'en' else
            "仅在你有至少两个数据库文件夹时使用该面板，以保证跨数据库对比成立。",
        )
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            data_root = _directory_input(
                "ICU Data Root" if lang == 'en' else "ICU 数据根目录",
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
            db_labels = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'Amsterdam', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
            multiselect_kwargs = {
                "label": "Databases" if lang == 'en' else "数据库",
                "options": db_options,
                "format_func": lambda x: db_labels.get(x, x),
                "key": "multidb_selected",
            }
            if 'multidb_selected' not in st.session_state:
                default_dbs = [_default_real_database()]
                multiselect_kwargs["default"] = [db for db in default_dbs if db in db_options] or ['miiv']
            selected_dbs = st.multiselect(**multiselect_kwargs)
            _render_crossdb_db_cards(db_options, db_labels, selected_dbs, lang=lang)

        with col3:
            max_patients = st.number_input(
                "Max Patients" if lang == 'en' else "最大患者数",
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

        _render_crossdb_micro_heading(
            "Feature scope" if lang == 'en' else "特征范围",
            "Select harmonized concepts" if lang == 'en' else "选择标准化概念",
            "Keep the grid lean; use the detailed single-feature view for close distribution review." if lang == 'en' else "保持网格克制；精细分布复核放到单特征详细视图。",
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_group = st.selectbox(
                "Feature Group" if lang == 'en' else "特征分组",
                options=list(feature_groups.keys()),
                key="multidb_group"
            )

        with col2:
            available_features = feature_groups.get(selected_group, [])
            selected_features = st.multiselect(
                "Select Features" if lang == 'en' else "选择特征",
                options=available_features,
                default=available_features[:4],
                key="multidb_features"
            )

        # When the user picks ≥6 features the 4×N grid becomes very dense
        # and individual subplots get tiny. Point them at the dedicated
        # "Detailed Single Feature View" further down the page instead
        # (2026-05 Phase D polish).
        if len(selected_features) >= 6:
            crowd_msg = (
                f"Tip: you selected {len(selected_features)} features. The grid will get dense; use the detailed single-feature view after generation for close review."
                if lang == 'en' else
                f"你选择了 {len(selected_features)} 个特征，网格会变密；生成后使用单特征详细视图做精细复核。"
            )
            st.markdown(
                f'<div class="compact-inline-notice warn">{html.escape(crowd_msg)}</div>',
                unsafe_allow_html=True,
            )

        _render_crossdb_setup_contract(
            data_root=data_root,
            selected_dbs=selected_dbs,
            db_labels=db_labels,
            selected_group=selected_group,
            selected_features=selected_features,
            max_patients=max_patients,
            lang=lang,
        )

        # 加载按钮
        load_btn = st.button(
            "Load & Generate" if lang == 'en' else "加载并生成",
            type="primary",
            key="multidb_load"
        )

        _render_compact_divider()

        if load_btn:
            if len(selected_dbs or []) < 2:
                st.session_state.pop('multidb_data', None)
                st.session_state.pop('multidb_concepts', None)
                st.session_state.pop('multidb_is_demo', None)
                _render_crossdb_loader_notice(
                    "warning",
                    "Load gate" if lang == "en" else "加载门禁",
                    "Select at least two databases" if lang == "en" else "请至少选择两个数据库",
                    "Select at least two databases before loading a cross-database comparison."
                    if lang == 'en' else
                    "请至少选择两个数据库后再加载跨数据库对比。",
                )
            elif not selected_features:
                _render_crossdb_loader_notice(
                    "warning",
                    "Load gate" if lang == "en" else "加载门禁",
                    "Select at least one feature" if lang == "en" else "请至少选择一个特征",
                    "Select at least one feature before loading."
                    if lang == 'en' else
                    "请至少选择一个特征后再加载。",
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
                        _render_crossdb_loader_notice(
                            "warning",
                            "Load result" if lang == "en" else "加载结果",
                            "Loaded fewer than two databases" if lang == "en" else "实际加载到的数据库少于两个",
                            "Check that ICU Data Root contains separate folders for the selected databases."
                            if lang == 'en' else
                            "请检查 ICU 数据根目录是否包含所选数据库各自的子文件夹。",
                        )
                    else:
                        st.session_state['multidb_data'] = data
                        st.session_state['multidb_concepts'] = selected_features
                        st.session_state['multidb_is_demo'] = False
                except Exception as e:
                    st.session_state.pop('multidb_data', None)
                    st.session_state.pop('multidb_concepts', None)
                    st.session_state.pop('multidb_is_demo', None)
                    _render_crossdb_loader_notice(
                        "danger",
                        "Load result" if lang == "en" else "加载结果",
                        "Error loading data" if lang == "en" else "数据加载失败",
                        str(e),
                    )
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
        _render_crossdb_micro_heading(
            "Loaded frames" if lang == 'en' else "已加载数据帧",
            "Database coverage" if lang == 'en' else "数据库覆盖",
            "Record counts reflect the feature rows available to the distribution benchmark." if lang == 'en' else "记录数表示可用于分布基准的特征行。",
        )
        _render_crossdb_metric_grid([
            (
                str(db).upper(),
                f"{len(df):,}",
                "records" if lang == 'en' else "行记录",
                "accent" if idx == 0 else "neutral",
            )
            for idx, (db, df) in enumerate(data.items())
        ])

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
            _render_crossdb_micro_heading(
                "Distribution grid" if lang == 'en' else "分布网格",
                "Multi-Database Feature Distribution Comparison" if lang == 'en' else "多数据库特征分布对比",
            )
            n_cols = min(4, len(concepts))
            fig = mdd.create_distribution_grid(data, concepts, cols=n_cols)
            if not screenshot_mode:
                fig.update_layout(
                    title=None,
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
            _render_crossdb_micro_heading(
                "Focused review" if lang == 'en' else "聚焦复核",
                "Detailed Single Feature View" if lang == 'en' else "单特征详细视图",
                "Use this view when the overview grid is too dense or when one concept needs database-by-database inspection."
                if lang == 'en' else
                "当总览网格过密，或某个概念需要逐数据库检查时使用此视图。",
            )

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
                    st.markdown(
                        '<div class="eu-crossdb-stats-table-title">'
                        f'{html.escape("Statistics" if lang == "en" else "统计信息")}'
                        '</div>',
                        unsafe_allow_html=True,
                    )
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
            _render_crossdb_loader_notice(
                "danger",
                "Chart render" if lang == "en" else "图表渲染",
                "Error generating chart" if lang == "en" else "图表生成失败",
                str(e),
            )
    else:
        # 占位提示
        _render_crossdb_loader_notice(
            "pending",
            "Benchmark workspace" if lang == "en" else "基准工作区",
            "Cross-database comparison is waiting for input" if lang == "en" else "跨数据库对比等待输入",
            "Select databases and features, then click Load & Generate."
            if lang == 'en' else
            "选择数据库和特征，然后点击加载并生成。",
        )
