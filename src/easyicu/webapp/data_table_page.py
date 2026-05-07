"""Data-table subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_data_table_subtab', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_data_table_subtab(app_context: dict[str, Any] | None = None):
    """渲染数据大表子模块 - 让用户按模块查看已加载的数据。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')

    page_copy = _get_data_table_page_copy(lang)
    page_title = page_copy["title"]
    st.markdown(f'<div class="compact-section-title">{page_title}</div>', unsafe_allow_html=True)

    page_desc = page_copy["description"]
    st.markdown(f'<div class="compact-section-desc">{page_desc}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    if len(st.session_state.loaded_concepts) == 0:
        no_data_msg = "Please load data first in the settings above." if lang == 'en' else "请先在上方设置中加载数据。"
        st.warning(no_data_msg)
        return

    # 🔧 FIX (2026-02-12): 使用内部分组定义来构建映射
    # 由于列名已在 load_from_exported() 中规范化，这里直接使用列名查找分组
    concept_to_group_display = {}
    concept_to_group_key = {}
    for group_key, concepts in CONCEPT_GROUPS_INTERNAL.items():
        # 获取显示名称
        display_name = CONCEPT_GROUP_NAMES.get(group_key, (group_key, group_key))
        group_display = display_name[0] if lang == 'en' else display_name[1]

        for c in concepts:
            if c not in concept_to_group_display:
                concept_to_group_display[c] = group_display
                concept_to_group_key[c] = group_key

    # 🔧 FIX (2026-02-12): 列名已在 load_from_exported() 中规范化并去重
    # 每个列就是一个唯一的 concept，直接分组即可
    loaded_by_module: Dict[str, Dict[str, Any]] = {}

    for column_name in st.session_state.loaded_concepts.keys():
        # 使用列名查找分组（列名已经是规范化后的）
        group_display = concept_to_group_display.get(column_name)
        group_key = concept_to_group_key.get(column_name)
        if group_display:
            if group_display not in loaded_by_module:
                loaded_by_module[group_display] = {
                    'group_key': group_key,
                    'concepts': [],
                }
            loaded_by_module[group_display]['concepts'].append(column_name)

    # 🔧 FIX (2026-02-12): Features = Concepts = 列数（已去重）
    unique_feature_count = len(st.session_state.loaded_concepts)

    patient_count = len(st.session_state.patient_ids) if st.session_state.patient_ids else 0
    loaded_summary = (
        f"{len(loaded_by_module)} modules loaded · {unique_feature_count} features · {patient_count} patients"
        if lang == 'en'
        else f"已加载 {len(loaded_by_module)} 个模块 · {unique_feature_count} 个特征 · {patient_count} 名患者"
    )
    st.markdown(f'<div class="preview-hint-line">{loaded_summary}</div>', unsafe_allow_html=True)

    # 模块选择器 - 🔧 放大标题
    module_select_label = "Select Module to View" if lang == 'en' else "选择要查看的模块"
    st.markdown(f'<div style="font-size:1.02rem;font-weight:800;color:#111827;margin:0 0 0.35rem 0;line-height:1.2">📦 {module_select_label}</div>', unsafe_allow_html=True)
    module_options = list(loaded_by_module.keys())

    if not module_options:
        no_module_msg = "No modules found in loaded data." if lang == 'en' else "加载的数据中没有找到模块。"
        st.info(no_module_msg)
        return

    selected_module = st.selectbox(
        "Select Module",
        options=module_options,
        key="data_table_module_select",
        label_visibility="collapsed"
    )

    if selected_module:
        module_meta = loaded_by_module[selected_module]
        module_key = module_meta.get('group_key', '')
        module_concepts = module_meta['concepts']
        preview_meta = _build_module_preview_metadata(module_key, selected_module, module_concepts, lang=lang)

        preview_cols = st.columns([2.5, 0.9, 0.9])
        tags_html = "".join(
            f'<span class="module-feature-chip">{html.escape(tag)}</span>'
            for tag in preview_meta['tags']
        )
        if preview_meta['overflow_count']:
            tags_html += f'<span class="module-feature-chip muted">+{preview_meta["overflow_count"]}</span>'

        with preview_cols[0]:
            selected_label = "Selected Module" if lang == 'en' else "当前模块"
            st.markdown(
                f'''
                <div class="module-preview-card">
                    <div class="eyebrow">{selected_label}</div>
                    <div class="title">{html.escape(selected_module)}</div>
                    <div class="summary">{html.escape(preview_meta["summary"])}</div>
                    <div class="module-feature-chip-row">{tags_html}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        with preview_cols[1]:
            st.markdown(
                f'<div class="mini-stat-card"><div class="mini-label">{"Features" if lang == "en" else "特征数"}</div><div class="mini-value">{len(module_concepts)}</div></div>',
                unsafe_allow_html=True,
            )
        with preview_cols[2]:
            st.markdown(
                f'<div class="mini-stat-card"><div class="mini-label">{"Patients" if lang == "en" else "患者数"}</div><div class="mini-value">{patient_count}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        # 特征选择器（单选或多选合并）- 默认合并全部放第一个
        # 🔧 放大标题
        view_mode_label = "Preview Mode" if lang == 'en' else "预览模式"
        st.markdown(f'<div style="font-size:1.02rem;font-weight:800;color:#111827;margin:0 0 0.35rem 0;line-height:1.2">👁️ {view_mode_label}</div>', unsafe_allow_html=True)
        view_modes = ["Merge All (Wide Table)", "Single Feature"] if lang == 'en' else ["合并全部（宽表）", "单个特征"]

        view_mode = st.radio("View Mode", view_modes, horizontal=True, key="data_table_view_mode", index=0, label_visibility="collapsed")

        if view_mode == view_modes[1]:
            # 单个特征模式 (现在是第二个选项)
            feature_control_cols = st.columns([1.7, 0.78])
            feature_select_label = "Select Feature" if lang == 'en' else "选择特征"
            with feature_control_cols[0]:
                selected_feature = st.selectbox(
                    feature_select_label,
                    options=sorted(module_concepts),
                    key="data_table_feature_select"
                )
            with feature_control_cols[1]:
                st.markdown(
                    f'<div class="inline-control-label">{"Preview rows" if lang == "en" else "预览行数"}</div>',
                    unsafe_allow_html=True,
                )
                max_rows = st.selectbox(
                    "Preview rows" if lang == 'en' else "预览行数",
                    options=[500, 1000, 2000, 5000, 10000],
                    index=1,
                    key="single_feature_max_rows",
                    label_visibility="collapsed",
                )

            if selected_feature and selected_feature in st.session_state.loaded_concepts:
                df = st.session_state.loaded_concepts[selected_feature]

                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    feature_copy = _get_single_feature_preview_copy(selected_feature, lang)
                    st.markdown(
                        f'''
                        <div class="preview-toolbar">
                            <div class="preview-toolbar-main">
                                <div class="preview-toolbar-title">{feature_copy["title"]}</div>
                                <div class="preview-toolbar-note">{feature_copy["description"]}</div>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

                    stat_cols = st.columns(3)
                    size_kb = df.memory_usage(deep=True).sum() / 1024
                    stats = [
                        ("Feature Rows" if lang == 'en' else "特征行数", f"{len(df):,}"),
                        ("Columns" if lang == 'en' else "列数", len(df.columns)),
                        ("Memory" if lang == 'en' else "内存占用", f"{size_kb:.1f} KB"),
                    ]
                    for idx, (label, value) in enumerate(stats):
                        with stat_cols[idx]:
                            st.markdown(
                                f'<div class="tiny-stat-card"><div class="tiny-label">{label}</div><div class="tiny-value">{value}</div></div>',
                                unsafe_allow_html=True,
                            )

                    column_chip_html = "".join(
                        f'<span class="module-feature-chip">{html.escape(str(column_name))}</span>'
                        for column_name in df.columns[:8]
                    )
                    if len(df.columns) > 8:
                        column_chip_html += f'<span class="module-feature-chip muted">+{len(df.columns) - 8}</span>'
                    st.markdown(
                        f'''
                        <div class="subtle-preview-note">{"Columns included in this preview" if lang == "en" else "当前预览包含的列"}</div>
                        <div class="module-feature-chip-row">{column_chip_html}</div>
                        ''',
                        unsafe_allow_html=True,
                    )

                    cols_info_label = "Column Details" if lang == 'en' else "列详情"
                    with st.expander(f"📊 {cols_info_label}", expanded=False):
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Type': [str(df[c].dtype) for c in df.columns],
                            'Non-Null': [df[c].notna().sum() for c in df.columns],
                            'Null %': [f"{df[c].isna().mean()*100:.1f}%" for c in df.columns]
                        })
                        st.dataframe(col_info, hide_index=True, use_container_width=True)

                    # 添加搜索/过滤选项
                    filter_expander_label = "🔎 Preview Filters" if lang == 'en' else "🔎 预览筛选"
                    with st.expander(filter_expander_label, expanded=False):
                        # 患者过滤
                        id_col = st.session_state.get('id_col', 'stay_id')
                        if id_col in df.columns:
                            unique_ids = df[id_col].unique().tolist()
                            filter_patient_label = "Filter by Patient ID" if lang == 'en' else "按患者ID过滤"
                            selected_ids = st.multiselect(
                                filter_patient_label,
                                options=unique_ids[:100],  # 最多显示100个选项
                                default=[],
                                key=f"filter_ids_{selected_feature}"
                            )
                            if selected_ids:
                                df = df[df[id_col].isin(selected_ids)]

                    # 显示数据（限制行数以防卡顿）
                    display_df = df.head(max_rows) if len(df) > max_rows else df
                    # 🔧 FIX: 将布尔列转换为字符串"True"/"False"显示，而非复选框图标
                    display_df = display_df.copy()
                    for col in display_df.columns:
                        dtype_str = str(display_df[col].dtype).lower()
                        if 'bool' in dtype_str:
                            display_df[col] = display_df[col].astype(str)
                    st.dataframe(display_df, use_container_width=True, height=500)

                    if len(df) > max_rows:
                        truncate_msg = (
                            f"Showing first {max_rows:,} preview rows."
                            if lang == 'en'
                            else f"当前显示前 {max_rows:,} 行预览。"
                        )
                        st.caption(truncate_msg)
                    # 不提供下载按钮，因为数据是用户导入的
                else:
                    empty_msg = f"No data available for {selected_feature}" if lang == 'en' else f"{selected_feature} 没有可用数据"
                    st.info(empty_msg)

        else:
            # 合并全部模式（宽表）
            merge_control_cols = st.columns([2.5, 0.95])
            with merge_control_cols[0]:
                preview_hint = (
                    "Representative columns are prioritized below. Switch to Single Feature for full detail."
                    if lang == 'en'
                    else "下方优先展示代表性列；如需完整细节可切换到单个特征。"
                )
                st.markdown(
                    f'''
                    <div class="preview-toolbar">
                        <div class="preview-toolbar-main">
                            <div class="preview-toolbar-title">🧭 {"Merged Preview Table" if lang == "en" else "合并预览表"}</div>
                            <div class="preview-toolbar-note">{preview_hint}</div>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            with merge_control_cols[1]:
                st.markdown(
                    f'<div class="inline-control-label">{"Rows per feature" if lang == "en" else "每特征行数"}</div>',
                    unsafe_allow_html=True,
                )
                max_rows_per_feature = st.selectbox(
                    "Max rows" if lang == 'en' else "最大行数",
                    options=[1000, 2000, 5000, 10000],
                    index=1,
                    key="merge_max_rows",
                    label_visibility="collapsed",
                )

            # 收集该模块的所有数据
            dfs_to_merge = []
            id_col = st.session_state.get('id_col', 'stay_id')
            metadata_cols = {'valueuom', 'unit', 'units', 'category', 'type', 'dur_var', 'entertime', 'intakeoutputentryoffset'}
            unified_time_col = 'charttime'

            for concept_name in module_concepts:
                if concept_name in st.session_state.loaded_concepts:
                    df = st.session_state.loaded_concepts[concept_name]
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        df_copy = df.copy()

                        # 统一时间列名，避免不同数据库/概念的时间别名导致只按 ID 合并
                        if unified_time_col in df_copy.columns:
                            other_time_cols = [tc for tc in PREVIEW_TIME_COLUMNS if tc in df_copy.columns and tc != unified_time_col]
                            if other_time_cols:
                                df_copy = df_copy.drop(columns=other_time_cols)
                        else:
                            for tc in PREVIEW_TIME_COLUMNS:
                                if tc in df_copy.columns:
                                    df_copy = df_copy.rename(columns={tc: unified_time_col})
                                    break

                        drop_meta_cols = [c for c in df_copy.columns if c in metadata_cols]
                        if drop_meta_cols:
                            df_copy = df_copy.drop(columns=drop_meta_cols)

                        value_cols = [c for c in df_copy.columns if c not in [id_col, unified_time_col]]
                        if len(value_cols) == 1 and value_cols[0] != concept_name:
                            df_copy = df_copy.rename(columns={value_cols[0]: concept_name})
                        dfs_to_merge.append(df_copy)

            if len(dfs_to_merge) == 0:
                no_data_msg = "No data to merge in this module." if lang == 'en' else "该模块没有可合并的数据。"
                st.warning(no_data_msg)
            elif len(dfs_to_merge) == 1:
                merged_df = dfs_to_merge[0]
                display_merged = merged_df.head(1000).copy()
                # 🔧 FIX: 将布尔列转换为字符串"True"/"False"显示
                for col in display_merged.columns:
                    dtype_str = str(display_merged[col].dtype).lower()
                    if 'bool' in dtype_str:
                        display_merged[col] = display_merged[col].astype(str)
                st.dataframe(display_merged, use_container_width=True, height=500)
            else:
                from functools import reduce
                merge_cols = [id_col, unified_time_col]

                if not any(id_col in df.columns for df in dfs_to_merge):
                    no_common_msg = "Cannot merge: patient ID column is missing in all features." if lang == 'en' else "无法合并：所有特征都缺少患者 ID 列。"
                    st.warning(no_common_msg)
                else:
                    try:
                        merging_msg = "Merging data..." if lang == 'en' else "正在合并数据..."
                        with st.spinner(merging_msg):
                            MAX_ROWS_PER_DF = max_rows_per_feature
                            total_rows_before = sum(len(df) for df in dfs_to_merge)

                            dynamic_frames = []
                            static_frames = []
                            seen_value_cols = set()

                            for df in dfs_to_merge:
                                if id_col not in df.columns:
                                    continue

                                df_proc = df.copy()
                                if len(df_proc) > MAX_ROWS_PER_DF:
                                    df_proc = df_proc.head(MAX_ROWS_PER_DF)

                                value_cols_in_df = [c for c in df_proc.columns if c not in [id_col, unified_time_col]]
                                if not value_cols_in_df:
                                    continue

                                duplicate_cols = [vc for vc in value_cols_in_df if vc in seen_value_cols]
                                if duplicate_cols:
                                    df_proc = df_proc.drop(columns=duplicate_cols, errors='ignore')
                                    value_cols_in_df = [c for c in value_cols_in_df if c not in duplicate_cols]
                                seen_value_cols.update(value_cols_in_df)
                                if not value_cols_in_df:
                                    continue

                                has_time = unified_time_col in df_proc.columns and not df_proc[unified_time_col].isna().all()
                                if has_time:
                                    dynamic_frames.append(df_proc.drop_duplicates(subset=merge_cols, keep='last'))
                                else:
                                    static_frames.append(df_proc.drop_duplicates(subset=[id_col], keep='last'))

                            merged_df = None

                            if dynamic_frames:
                                stacked_frames = []
                                for df_proc in dynamic_frames:
                                    value_cols_in_df = [c for c in df_proc.columns if c not in merge_cols]
                                    for value_col in value_cols_in_df:
                                        single_val_df = df_proc[merge_cols + [value_col]].copy()
                                        single_val_df['_concept'] = str(value_col)
                                        single_val_df['_value'] = single_val_df[value_col]
                                        single_val_df.drop(columns=[value_col], inplace=True)
                                        stacked_frames.append(single_val_df)

                                if stacked_frames:
                                    stacked = pd.concat(stacked_frames, ignore_index=True)
                                    merged_df = stacked.pivot_table(
                                        index=merge_cols,
                                        columns='_concept',
                                        values='_value',
                                        aggfunc='first'
                                    ).reset_index()

                            if static_frames:
                                static_merged = static_frames[0] if len(static_frames) == 1 else reduce(
                                    lambda left, right: pd.merge(left, right, on=[id_col], how='outer'),
                                    static_frames
                                )
                                if merged_df is not None and id_col in merged_df.columns:
                                    merged_df = pd.merge(merged_df, static_merged, on=[id_col], how='left')
                                else:
                                    merged_df = static_merged

                            if merged_df is None:
                                no_data_msg = "No unique data columns to merge." if lang == 'en' else "没有唯一的数据列可合并。"
                                st.warning(no_data_msg)
                                return

                        sampled_for_preview = total_rows_before > MAX_ROWS_PER_DF * len(dfs_to_merge)
                        # 🔧 显示截断提示
                        max_rows = 1000
                        display_df = merged_df.head(max_rows).copy() if len(merged_df) > max_rows else merged_df.copy()
                        preview_columns = _select_preview_columns(
                            display_df,
                            module_key=module_key,
                            module_concepts=module_concepts,
                            id_col=id_col,
                            max_columns=10,
                        )
                        if preview_columns:
                            display_df = display_df[preview_columns].copy()
                        # 🔧 FIX: 将布尔列转换为字符串"True"/"False"显示
                        for col in display_df.columns:
                            dtype_str = str(display_df[col].dtype).lower()
                            if 'bool' in dtype_str:
                                display_df[col] = display_df[col].astype(str)
                        summary_wrap_cols = st.columns([2.15, 1.0])
                        with summary_wrap_cols[0]:
                            summary_cols = st.columns([1.12, 0.9, 0.9])
                            with summary_cols[0]:
                                summary_badge = (
                                    f"⚠️ Sample preview · max {MAX_ROWS_PER_DF:,} rows per feature"
                                    if sampled_for_preview and lang == 'en'
                                    else (
                                        f"⚠️ 采样预览 · 每个特征最多 {MAX_ROWS_PER_DF:,} 行"
                                        if sampled_for_preview
                                        else (
                                            "✅ Merged preview ready"
                                            if lang == 'en'
                                            else "✅ 合并预览已就绪"
                                        )
                                    )
                                )
                                badge_class = "preview-badge warning" if sampled_for_preview else "preview-badge"
                                st.markdown(f'<div class="{badge_class}">{summary_badge}</div>', unsafe_allow_html=True)
                            preview_stats = [
                                ("Preview Rows" if lang == 'en' else "预览行数", f"{len(display_df):,}"),
                                ("Preview Columns" if lang == 'en' else "预览列数", len(display_df.columns)),
                            ]
                            for idx, (label, value) in enumerate(preview_stats, start=1):
                                with summary_cols[idx]:
                                    st.markdown(
                                        f'<div class="tiny-stat-card"><div class="tiny-label">{label}</div><div class="tiny-value">{value}</div></div>',
                                        unsafe_allow_html=True,
                                    )
                        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
                        st.dataframe(display_df, use_container_width=True, height=500)

                        if len(merged_df) > max_rows:
                            truncate_msg = (
                                f"Showing first {max_rows:,} preview rows."
                                if lang == 'en'
                                else f"当前显示前 {max_rows:,} 行预览。"
                            )
                            st.caption(truncate_msg)
                    # 不提供下载按钮，因为数据是用户导入的
                    except Exception as e:
                        err_msg = f"Error merging data: {e}" if lang == 'en' else f"合并数据时出错: {e}"
                        st.error(err_msg)
