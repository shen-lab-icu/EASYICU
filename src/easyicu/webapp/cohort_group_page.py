"""Cohort group-comparison subtab rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to the extracted cohort group page."""
    protected = {"render_group_comparison_subtab", "_install_app_context"}
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


def render_group_comparison_subtab(lang: str, app_context: dict[str, Any] | None = None):
    """分组对比子标签页 - 带独立数据加载配置"""
    if app_context is not None:
        _install_app_context(app_context)

    screenshot_mode = _is_screenshot_mode()

    if not screenshot_mode:
        _render_section_heading(
            "Group Contrast Table" if lang == 'en' else "组间对照表",
            "Analysis table" if lang == 'en' else "分析表",
        )

    # 获取当前入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')

    # ========== Demo模式：复用队列分析顶层的一次性共享演示工作区 ==========
    if entry_mode == 'demo':
        _ensure_cohort_demo_workspace(st.session_state, lang=lang)
        # Demo data is prepared once at the Cohort Analysis level; keep
        # individual panels focused on analysis rather than repeated setup.

    # ========== Real Data：如果共享工作区已就绪，跳过独立配置 ==========
    elif entry_mode == 'real' and _cohort_real_workspace_ready(st.session_state):
        # Data already seeded by the shared workspace; nothing extra needed.
        _sync_real_data_panel_defaults(root_key="grp_data_root", db_key="grp_db_select")

    # ========== Real Data模式：显示完整数据配置 ==========
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
            grp_src = st.radio(
                _src_label, _src_keys,
                index=_src_keys.index(_default_src),
                format_func=lambda x: _src_labels[x],
                horizontal=True, key="grp_data_source"
            )

            if grp_src == "demo":
                # ===== 模拟数据模式 =====
                n_patients = st.slider(
                    "👥 " + ("Number of Patients" if lang == 'en' else "患者数量"),
                    min_value=50, max_value=500, value=st.session_state.mock_params.get('n_patients', 100),
                    key="grp_demo_patients_inline"
                )
                load_btn = st.button(
                    "🚀 " + ("Generate Demo Data" if lang == 'en' else "生成模拟数据"),
                    type="primary", key="grp_load_demo_btn"
                )
                if load_btn:
                    st.session_state.mock_params['n_patients'] = n_patients
                    demographics_df = _generate_mock_demographics(n_patients, lang)
                    st.session_state['grp_demographics'] = demographics_df
                    st.session_state['grp_loaded_db'] = 'demo'
                    st.session_state['grp_is_demo'] = True
                    st.session_state.pop('grp_feature_data', None)
                    st.rerun()

            elif grp_src == "raw":
                # ===== 原始数据库模式 =====
                _sync_real_data_panel_defaults(root_key="grp_data_root", db_key="grp_db_select")
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    data_root = _directory_input(
                        "📁 " + ("ICU Data Root" if lang == 'en' else "ICU数据根目录"),
                        value=st.session_state.get('grp_data_root', ''),
                        input_key="grp_data_root",
                        button_key="grp_data_root_browse",
                        placeholder="/path/to/icudb" if os.name != 'nt' else "D:\\data\\icudb",
                        help="Root directory containing database folders (mimiciv, eicu, aumc, hirid)" if lang == 'en' else "包含数据库文件夹的根目录"
                    )
                    render_directory_structure_guide(lang)

                with col2:
                    db_options = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
                    default_db = st.session_state.get('grp_db_select') or _default_real_database()
                    selected_db = st.selectbox(
                        "🏥 " + ("Database" if lang == 'en' else "数据库"),
                        options=list(db_options.keys()),
                        index=list(db_options.keys()).index(default_db) if default_db in db_options else 0,
                        format_func=lambda x: db_options[x],
                        key="grp_db_select"
                    )

                with col3:
                    max_patients = st.number_input(
                        "👥 " + ("Max Patients" if lang == 'en' else "最大患者数"),
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                        key="grp_max_patients"
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
                    "🚀 " + ("Load Patient Demographics" if lang == 'en' else "加载患者人口统计学数据"),
                    type="primary",
                    disabled=not path_ok,
                    key="grp_load_btn"
                )

                if load_btn:
                    try:
                        from easyicu.patient_filter import PatientFilter

                        with st.spinner("Loading demographics..." if lang == 'en' else "正在加载人口统计学数据..."):
                            pf = PatientFilter(database=selected_db, data_path=full_data_path)
                            demographics_df = pf._load_demographics()

                            if len(demographics_df) > max_patients:
                                demographics_df = demographics_df.head(max_patients)

                            st.session_state['grp_demographics'] = demographics_df
                            st.session_state['grp_loaded_db'] = selected_db
                            st.session_state['grp_loaded_path'] = full_data_path
                            st.session_state['grp_is_demo'] = False

                        st.success(f"✅ Loaded {len(demographics_df):,} patients" if lang == 'en' else f"✅ 已加载 {len(demographics_df):,} 名患者")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif grp_src == "exported":
                # ===== 导出文件模式 =====
                col1, col2 = st.columns([3, 2])
                with col1:
                    export_root = _directory_input(
                        "📦 " + ("Folder with Exported Data Files" if lang == 'en' else "存放导出结果文件的文件夹"),
                        value=st.session_state.get('export_path', ''),
                        input_key="grp_export_root",
                        button_key="grp_export_root_browse",
                        placeholder="/path/to/easyicu_export" if os.name != 'nt' else "D:\\easyicu_export",
                        help="Choose the folder that contains EasyICU exported result folders" if lang == 'en' else "选择包含 EasyICU 导出结果子文件夹的目录"
                    )

                # 扫描导出文件夹
                _export_folders = _scan_export_folders(export_root if 'export_root' in dir() else st.session_state.get('grp_export_root', ''))

                with col2:
                    if _export_folders:
                        folder_options = {f[0]: f"📁 {f[0]} ({f[1]} files)" for f in _export_folders}
                        selected_folder = st.selectbox(
                            "📁 " + ("Select an Export Result Folder" if lang == 'en' else "选择一批导出结果"),
                            options=list(folder_options.keys()),
                            format_func=lambda x: folder_options[x],
                            key="grp_export_folder"
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
                        key="grp_load_export_btn"
                    )

                    if load_btn:
                        try:
                            with st.spinner("Loading..." if lang == 'en' else "加载中..."):
                                demographics_df = _load_demographics_from_export(selected_path)
                                _detected_db = demographics_df.attrs.get('detected_db', 'unknown')

                                st.session_state['grp_demographics'] = demographics_df
                                st.session_state['grp_loaded_db'] = _detected_db
                                st.session_state['grp_loaded_path'] = selected_path
                                st.session_state['grp_is_demo'] = False

                            st.success(f"✅ Loaded {len(demographics_df):,} patients from exported result files" if lang == 'en' else f"✅ 已从这批导出结果文件中加载 {len(demographics_df):,} 名患者")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    if not screenshot_mode:
        _render_compact_divider()

    # ========== 分组对比区域 ==========
    if 'grp_demographics' not in st.session_state:
        st.info("👆 " + ("Configure data source and click 'Load' to start" if lang == 'en' else "配置数据源并点击'加载'开始"))
        return

    demographics_df = st.session_state['grp_demographics']
    database = st.session_state.get('grp_loaded_db', 'miiv')
    data_path = st.session_state.get('grp_loaded_path', '')

    if not screenshot_mode:
        # 显示数据概览
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients" if lang == 'en' else "患者总数", f"{len(demographics_df):,}")
        with col2:
            avg_age = demographics_df['age'].mean() if 'age' in demographics_df.columns else 0
            st.metric("Mean Age" if lang == 'en' else "平均年龄", f"{avg_age:.1f}")
        with col3:
            male_pct = (demographics_df['gender'] == 'M').mean() * 100 if 'gender' in demographics_df.columns else 0
            st.metric("Male %" if lang == 'en' else "男性占比", f"{male_pct:.1f}%")
        with col4:
            mortality = (1 - demographics_df['survived'].mean()) * 100 if 'survived' in demographics_df.columns else 0
            st.metric("Mortality" if lang == 'en' else "死亡率", f"{mortality:.1f}%")

        _render_compact_divider()

        # 对比模式选择
        _render_section_heading(
            "Select Comparison Mode" if lang == 'en' else "选择对比模式",
            "Comparison" if lang == 'en' else "对比",
        )

    compare_options = {
        'survival': ('Survived vs Deceased', '存活 vs 死亡'),
        'age': ('Age Groups', '年龄分组'),
        'gender': ('Male vs Female', '男性 vs 女性'),
        'los': ('Short vs Long Stay', '短住院 vs 长住院'),
        'sepsis': ('Sepsis vs Non-sepsis', '脓毒症 vs 非脓毒症'),
        'custom': ('Custom Threshold', '自定义阈值'),
    }

    if screenshot_mode:
        compare_mode = 'survival'
        st.session_state["group_comp_mode"] = compare_mode
    else:
        compare_mode = st.radio(
            "Comparison Mode" if lang == 'en' else "对比模式",
            options=list(compare_options.keys()),
            format_func=lambda x: compare_options[x][0] if lang == 'en' else compare_options[x][1],
            horizontal=True,
            key="group_comp_mode"
        )

    # 根据模式显示额外配置
    if compare_mode == 'age' and not screenshot_mode:
        age_threshold = st.slider(
            "Age Threshold" if lang == 'en' else "年龄阈值",
            min_value=30, max_value=90, value=65, step=5,
            key="group_comp_age_threshold"
        )
    elif compare_mode == 'los' and 'los_hours' in demographics_df.columns and not screenshot_mode:
        median_los = demographics_df['los_hours'].median()
        los_threshold = st.slider(
            "LOS Threshold (hours)" if lang == 'en' else "住院时长阈值（小时）",
            min_value=24,
            max_value=int(min(500, demographics_df['los_hours'].quantile(0.95))),
            value=int(median_los),
            step=12,
            key="group_comp_los_threshold"
        )
    elif compare_mode == 'sepsis' and not screenshot_mode:
        sepsis_note = (
            "Sepsis grouping uses Sepsis-3 labels (`sep3_sofa2` preferred, fallback `sep3_sofa1`)."
            if lang == 'en'
            else "脓毒症分组优先使用 Sepsis-3 标签 `sep3_sofa2`，回退到 `sep3_sofa1`。"
        )
        st.caption(sepsis_note)
    elif compare_mode == 'custom' and not screenshot_mode:
        custom_note = (
            "Choose the threshold variable after selecting feature modules below."
            if lang == 'en'
            else "请在下方选择特征模块后，再指定阈值变量。"
        )
        st.caption(custom_note)

    # 定义所有可用的特征模块
    FEATURE_MODULES = {
        'demographic': {
            'name_en': 'Demographics',
            'name_zh': '人口统计学',
            'features': [
                ('age', 'Age (years)', '年龄 (岁)', 'continuous'),
                ('gender', 'Male', '男性', 'binary', 'M'),
                ('weight', 'Weight (kg)', '体重 (kg)', 'continuous'),
                ('height', 'Height (cm)', '身高 (cm)', 'continuous'),
                ('los_days', 'ICU LOS (days)', 'ICU住院时长 (天)', 'continuous'),
                ('first_icu_stay', 'First ICU Stay', '首次ICU入住', 'binary', True),
            ],
            'default': True
        },
        'outcome': {
            'name_en': 'Outcomes',
            'name_zh': '结局指标',
            'features': [
                ('mortality', 'ICU Mortality', 'ICU死亡率', 'binary_survival'),
            ],
            'default': True
        },
        'vital': {
            'name_en': 'Vital Signs',
            'name_zh': '生命体征',
            'features': [
                ('hr', 'Heart Rate (bpm)', '心率 (bpm)', 'continuous'),
                ('sbp', 'Systolic BP (mmHg)', '收缩压 (mmHg)', 'continuous'),
                ('dbp', 'Diastolic BP (mmHg)', '舒张压 (mmHg)', 'continuous'),
                ('map', 'Mean Arterial Pressure (mmHg)', '平均动脉压 (mmHg)', 'continuous'),
                ('resp', 'Respiratory Rate', '呼吸频率', 'continuous'),
                ('temp', 'Temperature (°C)', '体温 (°C)', 'continuous'),
                ('o2sat', 'SpO2 (%)', '血氧饱和度 (%)', 'continuous'),
            ],
            'default': True
        },
        'lab': {
            'name_en': 'Laboratory',
            'name_zh': '实验室检查',
            'features': [
                ('glu', 'Glucose (mg/dL)', '血糖 (mg/dL)', 'continuous'),
                ('na', 'Sodium (mEq/L)', '钠 (mEq/L)', 'continuous'),
                ('k', 'Potassium (mEq/L)', '钾 (mEq/L)', 'continuous'),
                ('crea', 'Creatinine (mg/dL)', '肌酐 (mg/dL)', 'continuous'),
                ('bili', 'Bilirubin (mg/dL)', '胆红素 (mg/dL)', 'continuous'),
                ('lact', 'Lactate (mmol/L)', '乳酸 (mmol/L)', 'continuous'),
                ('alb', 'Albumin (g/dL)', '白蛋白 (g/dL)', 'continuous'),
                ('bun', 'BUN (mg/dL)', '尿素氮 (mg/dL)', 'continuous'),
            ],
            'default': True
        },
        'hematology': {
            'name_en': 'Hematology',
            'name_zh': '血液学',
            'features': [
                ('hgb', 'Hemoglobin (g/dL)', '血红蛋白 (g/dL)', 'continuous'),
                ('plt', 'Platelets (K/uL)', '血小板 (K/uL)', 'continuous'),
                ('wbc', 'WBC (K/uL)', '白细胞 (K/uL)', 'continuous'),
                ('inr_pt', 'INR', 'INR', 'continuous'),
            ],
            'default': True
        },
        'blood_gas': {
            'name_en': 'Blood Gas',
            'name_zh': '血气分析',
            'features': [
                ('ph', 'pH', 'pH值', 'continuous'),
                ('po2', 'PaO2 (mmHg)', 'PaO2 (mmHg)', 'continuous'),
                ('pco2', 'PaCO2 (mmHg)', 'PaCO2 (mmHg)', 'continuous'),
                ('fio2', 'FiO2 (%)', 'FiO2 (%)', 'continuous'),
                ('pafi', 'P/F Ratio', 'P/F比值', 'continuous'),
                ('safi', 'S/F Ratio', 'S/F比值', 'continuous'),
            ],
            'default': True
        },
        'sofa': {
            'name_en': 'SOFA Scores',
            'name_zh': 'SOFA评分',
            'features': [
                ('sofa', 'SOFA Score', 'SOFA评分', 'continuous'),
                ('sofa_resp', 'SOFA Respiratory', 'SOFA呼吸', 'continuous'),
                ('sofa_coag', 'SOFA Coagulation', 'SOFA凝血', 'continuous'),
                ('sofa_liver', 'SOFA Liver', 'SOFA肝脏', 'continuous'),
                ('sofa_cardio', 'SOFA Cardiovascular', 'SOFA心血管', 'continuous'),
                ('sofa_cns', 'SOFA CNS', 'SOFA神经', 'continuous'),
                ('sofa_renal', 'SOFA Renal', 'SOFA肾脏', 'continuous'),
            ],
            'default': True
        },
    }

    def _merge_feature_frame(target_df: pd.DataFrame, concept_name: str, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Merge one concept-level feature frame into the cohort table with a unified ID column."""
        if concept_name in target_df.columns or not isinstance(feat_df, pd.DataFrame) or feat_df.empty:
            return target_df
        feat_df_copy = feat_df.copy()
        feat_id_col = None
        for col in ['stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
            if col in feat_df_copy.columns:
                feat_id_col = col
                break
        if feat_id_col is None or concept_name not in feat_df_copy.columns:
            return target_df

        target_id_col = 'stay_id' if 'stay_id' in target_df.columns else 'patient_id'
        feat_df_copy[feat_id_col] = pd.to_numeric(feat_df_copy[feat_id_col], errors='coerce')
        feat_df_copy = feat_df_copy.dropna(subset=[feat_id_col])
        if feat_df_copy.empty:
            return target_df

        feat_df_copy[feat_id_col] = feat_df_copy[feat_id_col].astype(int)
        if feat_id_col != target_id_col:
            feat_df_copy[target_id_col] = feat_df_copy[feat_id_col]
        return target_df.merge(feat_df_copy[[target_id_col, concept_name]], on=target_id_col, how='left')

    def _format_smd_value(value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return '-'
        return f"{value:.2f} {_smd_severity_tag(float(value), lang)}"

    default_modules = [k for k, v in FEATURE_MODULES.items() if v.get('default', False)]
    if screenshot_mode:
        selected_modules = [m for m in ['demographic', 'outcome', 'vital', 'lab', 'sofa'] if m in FEATURE_MODULES]
    else:
        _render_compact_divider()

        # ========== 特征模块选择 ==========
        _render_section_heading(
            "Select Feature Modules" if lang == 'en' else "选择特征模块",
            "Features" if lang == 'en' else "变量",
        )

        # 模块多选
        selected_modules = st.multiselect(
            "Select feature modules" if lang == 'en' else "选择特征模块",
            options=list(FEATURE_MODULES.keys()),
            default=default_modules,
            format_func=lambda x: FEATURE_MODULES[x]['name_en'] if lang == 'en' else FEATURE_MODULES[x]['name_zh'],
            key="grp_feature_modules",
            help="Click to add/remove modules. All available modules are listed above."
                 if lang == 'en' else "点击可添加或移除模块，上方列出了所有可用模块"
        )

    # 显示将要加载的特征
    if selected_modules:
        concepts_to_load = []
        for mod in selected_modules:
            if mod not in ['demographic', 'outcome']:  # 这些从 demographics 表获取
                for feat in FEATURE_MODULES[mod]['features']:
                    concepts_to_load.append(feat[0])

        if concepts_to_load and not screenshot_mode:
            with st.expander("🔬 " + (f"Features to load: {len(concepts_to_load)}" if lang == 'en' else f"待加载特征: {len(concepts_to_load)}个"), expanded=False):
                st.caption(", ".join(concepts_to_load))

    custom_variable_options: dict[str, str] = {}
    for column_name in ['age', 'los_days', 'los_hours', 'sofa_max', 'weight', 'height', 'bmi']:
        if column_name in demographics_df.columns and pd.api.types.is_numeric_dtype(demographics_df[column_name]):
            custom_variable_options[column_name] = column_name
    for mod_key, mod_info in FEATURE_MODULES.items():
        for feat_info in mod_info['features']:
            feat_key = feat_info[0]
            feat_label = feat_info[1] if lang == 'en' else feat_info[2]
            feat_type = feat_info[3]
            if feat_type == 'continuous' and feat_key not in custom_variable_options:
                custom_variable_options[feat_key] = feat_label

    if compare_mode == 'custom' and custom_variable_options and not screenshot_mode:
        custom_cols = st.columns([1.4, 1.2])
        default_custom_var = 'sofa_max' if 'sofa_max' in custom_variable_options else next(iter(custom_variable_options))
        with custom_cols[0]:
            custom_var = st.selectbox(
                "Threshold Variable" if lang == 'en' else "阈值变量",
                options=list(custom_variable_options.keys()),
                index=list(custom_variable_options.keys()).index(st.session_state.get('group_comp_custom_feature', default_custom_var)) if st.session_state.get('group_comp_custom_feature', default_custom_var) in custom_variable_options else 0,
                format_func=lambda x: custom_variable_options[x],
                key="group_comp_custom_feature",
            )
        with custom_cols[1]:
            threshold_placeholder = (
                "Threshold will be estimated from the loaded variable distribution below."
                if lang == 'en'
                else "阈值会在下方根据已加载变量分布自动估计。"
            )
            st.markdown(
                f'<div class="compact-inline-notice info" style="margin-top: 1.6rem;">{threshold_placeholder}</div>',
                unsafe_allow_html=True,
            )

    if not screenshot_mode:
        _render_compact_divider()

    # 执行分组
    try:
        base_df = demographics_df.copy()
        group1_ids, group2_ids = [], []
        group1_name, group2_name = "", ""
        show_mortality = True

        # 检测ID列名（支持stay_id或patient_id）
        id_col = 'stay_id' if 'stay_id' in base_df.columns else 'patient_id'
        base_df[id_col] = pd.to_numeric(base_df[id_col], errors='coerce')
        base_df = base_df.dropna(subset=[id_col]).copy()
        base_df[id_col] = base_df[id_col].astype(int)

        concepts_to_load = []
        for mod in selected_modules:
            if mod not in ['demographic', 'outcome']:
                for feat in FEATURE_MODULES[mod]['features']:
                    concepts_to_load.append(feat[0])

        grouping_concepts_required: list[str] = []
        custom_var = st.session_state.get('group_comp_custom_feature')
        if compare_mode == 'sepsis':
            grouping_concepts_required = ['sep3_sofa2', 'sep3_sofa1']
        elif compare_mode == 'custom' and custom_var and custom_var not in base_df.columns:
            grouping_concepts_required = [custom_var]

        concepts_to_load = list(dict.fromkeys(concepts_to_load + grouping_concepts_required))
        feature_data = st.session_state.get('grp_feature_data', {})
        all_patient_ids = base_df[id_col].astype(int).unique().tolist()

        if concepts_to_load:
            missing_concepts = [c for c in concepts_to_load if c not in feature_data]
            if missing_concepts:
                if entry_mode == 'demo' or database == 'demo':
                    reused = _build_group_feature_data_from_loaded_concepts(
                        all_patient_ids,
                        missing_concepts,
                        st.session_state.get('loaded_concepts', {}) or {},
                        id_col=id_col,
                    )
                    if reused:
                        feature_data = {**feature_data, **reused}
                        st.session_state['grp_feature_data'] = feature_data
                    missing_concepts = [c for c in missing_concepts if c not in feature_data]
                    if missing_concepts:
                        auto_load_msg = "Auto-loading simulated features for demo mode..." if lang == 'en' else "演示模式自动加载模拟特征数据..."
                        with st.spinner(auto_load_msg):
                            generated = _build_mock_group_feature_data(
                                all_patient_ids,
                                missing_concepts,
                                id_col=id_col,
                            )
                            feature_data = {**feature_data, **generated}
                            st.session_state['grp_feature_data'] = feature_data
                else:
                    st.info(f"🔬 " + (f"{len(missing_concepts)} features need to be loaded: " if lang == 'en' else f"需要加载 {len(missing_concepts)} 个特征: ") + ", ".join(missing_concepts[:5]) + ("..." if len(missing_concepts) > 5 else ""))
                    load_features_btn = st.button(
                        "🚀 " + (f"Load {len(missing_concepts)} Features" if lang == 'en' else f"加载 {len(missing_concepts)} 个特征"),
                        type="primary",
                        key="grp_load_features"
                    )
                    if load_features_btn:
                        try:
                            from easyicu import load_concepts

                            with st.spinner(f"Loading {len(missing_concepts)} features for {len(all_patient_ids)} patients..." if lang == 'en' else f"正在加载 {len(missing_concepts)} 个特征..."):
                                progress_bar = st.progress(0)
                                loaded_count = 0

                                for i, concept in enumerate(missing_concepts):
                                    try:
                                        df_concept = load_concepts(
                                            concepts=[concept],
                                            database=database,
                                            data_path=data_path,
                                            patient_ids=all_patient_ids,
                                            verbose=False,
                                            **_get_sepsis_runtime_options(),
                                        )
                                        if df_concept is not None and len(df_concept) > 0 and concept in df_concept.columns:
                                            feat_id_col = None
                                            for col in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'hadm_id']:
                                                if col in df_concept.columns:
                                                    feat_id_col = col
                                                    break
                                            if feat_id_col is None:
                                                feat_id_col = df_concept.columns[0]
                                            agg_func = 'max' if concept.startswith('sep3_') else 'mean'
                                            agg_df = df_concept.groupby(feat_id_col, as_index=False)[concept].agg(agg_func)
                                            agg_df.columns = [id_col, concept]
                                            agg_df[id_col] = pd.to_numeric(agg_df[id_col], errors='coerce')
                                            agg_df = agg_df.dropna(subset=[id_col])
                                            if not agg_df.empty:
                                                agg_df[id_col] = agg_df[id_col].astype(int)
                                                feature_data[concept] = agg_df
                                                loaded_count += 1
                                    except Exception:
                                        pass

                                    progress_bar.progress((i + 1) / len(missing_concepts))

                                progress_bar.empty()
                                st.session_state['grp_feature_data'] = feature_data
                                st.success(f"✅ " + (f"Loaded {loaded_count}/{len(missing_concepts)} features" if lang == 'en' else f"已加载 {loaded_count}/{len(missing_concepts)} 个特征"))
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading features: {e}")
                    if compare_mode in {'sepsis', 'custom'}:
                        dependency_msg = (
                            "Load the grouping variable first to compute this comparison."
                            if lang == 'en'
                            else "请先加载分组所需变量后再计算此对比。"
                        )
                        st.warning(dependency_msg)
                        return

        analysis_df = base_df.copy()
        for concept in concepts_to_load:
            if concept in feature_data:
                analysis_df = _merge_feature_frame(analysis_df, concept, feature_data[concept])

        if compare_mode == 'survival':
            if 'survived' not in analysis_df.columns:
                st.warning("Survival data not available" if lang == 'en' else "无存活状态数据")
                return

            survived_df = analysis_df[analysis_df['survived'] == 1]
            deceased_df = analysis_df[analysis_df['survived'] == 0]
            group1_ids = survived_df[id_col].tolist()
            group2_ids = deceased_df[id_col].tolist()
            group1_name = 'Survived' if lang == 'en' else '存活'
            group2_name = 'Deceased' if lang == 'en' else '死亡'
            show_mortality = False

        elif compare_mode == 'age':
            threshold = st.session_state.get('group_comp_age_threshold', 65)
            young_df = analysis_df[analysis_df['age'] < threshold]
            old_df = analysis_df[analysis_df['age'] >= threshold]
            group1_ids = young_df[id_col].tolist()
            group2_ids = old_df[id_col].tolist()
            group1_name = f'Age < {threshold}' if lang == 'en' else f'年龄 < {threshold}'
            group2_name = f'Age ≥ {threshold}' if lang == 'en' else f'年龄 ≥ {threshold}'

        elif compare_mode == 'gender':
            if 'gender' not in analysis_df.columns:
                st.warning("Gender data not available" if lang == 'en' else "无性别数据")
                return
            male_df = analysis_df[analysis_df['gender'] == 'M']
            female_df = analysis_df[analysis_df['gender'] == 'F']
            group1_ids = male_df[id_col].tolist()
            group2_ids = female_df[id_col].tolist()
            group1_name = 'Male' if lang == 'en' else '男性'
            group2_name = 'Female' if lang == 'en' else '女性'

        elif compare_mode == 'los':
            if 'los_hours' not in analysis_df.columns:
                st.warning("Length of stay data not available" if lang == 'en' else "无住院时长数据")
                return
            threshold = st.session_state.get('group_comp_los_threshold', int(analysis_df['los_hours'].median()))
            short_df = analysis_df[analysis_df['los_hours'] < threshold]
            long_df = analysis_df[analysis_df['los_hours'] >= threshold]
            group1_ids = short_df[id_col].tolist()
            group2_ids = long_df[id_col].tolist()
            group1_name = f'LOS < {threshold}h' if lang == 'en' else f'住院 < {threshold}h'
            group2_name = f'LOS ≥ {threshold}h' if lang == 'en' else f'住院 ≥ {threshold}h'
        elif compare_mode == 'sepsis':
            sepsis_col = next((c for c in ['sep3_sofa2', 'sep3_sofa1'] if c in analysis_df.columns), None)
            if sepsis_col is None:
                st.warning("Sepsis-3 labels are not available for grouping." if lang == 'en' else "当前没有可用于分组的 Sepsis-3 标签。")
                return
            sepsis_mask = pd.to_numeric(analysis_df[sepsis_col], errors='coerce').fillna(0) > 0
            non_sepsis_df = analysis_df[~sepsis_mask]
            sepsis_df = analysis_df[sepsis_mask]
            group1_ids = non_sepsis_df[id_col].tolist()
            group2_ids = sepsis_df[id_col].tolist()
            group1_name = 'Non-sepsis' if lang == 'en' else '非脓毒症'
            group2_name = 'Sepsis' if lang == 'en' else '脓毒症'
        elif compare_mode == 'custom':
            custom_var = st.session_state.get('group_comp_custom_feature')
            if not custom_var or custom_var not in analysis_df.columns:
                st.warning("Threshold variable is not available." if lang == 'en' else "阈值变量当前不可用。")
                return
            custom_values = pd.to_numeric(analysis_df[custom_var], errors='coerce')
            valid_values = custom_values.dropna()
            if valid_values.empty:
                st.warning("Threshold variable has no numeric values." if lang == 'en' else "阈值变量没有可用的数值。")
                return
            min_value = float(valid_values.min())
            max_value = float(valid_values.max())
            default_threshold = float(valid_values.median())
            if np.isclose(min_value, max_value):
                threshold = min_value
                st.info("All patients share the same threshold value; the split may be degenerate." if lang == 'en' else "所有患者的阈值变量相同，分组可能退化。")
            elif pd.api.types.is_integer_dtype(valid_values) or np.allclose(valid_values, np.round(valid_values)):
                slider_default = int(round(st.session_state.get('group_comp_custom_threshold', default_threshold)))
                slider_default = min(max(slider_default, int(np.floor(min_value))), int(np.ceil(max_value)))
                threshold = st.slider(
                    "Threshold" if lang == 'en' else "阈值",
                    min_value=int(np.floor(min_value)),
                    max_value=int(np.ceil(max_value)),
                    value=slider_default,
                    step=1,
                    key="group_comp_custom_threshold",
                )
            else:
                step = max((max_value - min_value) / 100, 0.1)
                slider_default = float(st.session_state.get('group_comp_custom_threshold', default_threshold))
                slider_default = min(max(slider_default, float(min_value)), float(max_value))
                threshold = st.slider(
                    "Threshold" if lang == 'en' else "阈值",
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=slider_default,
                    step=float(step),
                    key="group_comp_custom_threshold",
                )
            lower_df = analysis_df[custom_values < threshold]
            higher_df = analysis_df[custom_values >= threshold]
            custom_label = custom_variable_options.get(custom_var, custom_var)
            group1_ids = lower_df[id_col].tolist()
            group2_ids = higher_df[id_col].tolist()
            group1_name = f"{custom_label} < {threshold:g}"
            group2_name = f"{custom_label} ≥ {threshold:g}"

        # 分组统计概览
        _render_section_heading(
            "Group Overview" if lang == 'en' else "分组概览",
            "Summary" if lang == 'en' else "概览",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(group1_name, f"{len(group1_ids):,}")
        with col2:
            st.metric(group2_name, f"{len(group2_ids):,}")
        with col3:
            total = len(group1_ids) + len(group2_ids)
            pct1 = len(group1_ids) / total * 100 if total > 0 else 0
            st.metric("Ratio" if lang == 'en' else "比例", f"{pct1:.1f}% / {100-pct1:.1f}%")

        if len(group1_ids) == 0 or len(group2_ids) == 0:
            st.warning("One group is empty, please adjust criteria" if lang == 'en' else "其中一个分组为空，请调整条件")
            return

        _render_compact_divider()

        # ========== 基线特征对比表 (Table One) ==========
        _render_section_heading(
            "Baseline Characteristics Comparison" if lang == 'en' else "基线特征对比表",
            "Table one" if lang == 'en' else "表一",
        )

        from scipy import stats

        # 获取两组数据 - 使用动态ID列
        group1_df = analysis_df[analysis_df[id_col].isin(group1_ids)].copy()
        group2_df = analysis_df[analysis_df[id_col].isin(group2_ids)].copy()

        def format_continuous(series, name):
            """格式化连续变量: mean ± std (median [IQR])"""
            valid = series.dropna()
            if len(valid) == 0:
                return '-'
            mean, std = valid.mean(), valid.std()
            median = valid.median()
            q25, q75 = valid.quantile(0.25), valid.quantile(0.75)
            return f"{mean:.1f} ± {std:.1f} ({median:.1f} [{q25:.1f}-{q75:.1f}])"

        def format_categorical(series, category, total):
            """格式化分类变量: n (%)"""
            n = (series == category).sum()
            pct = n / total * 100 if total > 0 else 0
            return f"{n:,} ({pct:.1f}%)"

        def calc_pvalue_continuous(s1, s2):
            """连续变量 p 值 (Mann-Whitney U)"""
            v1, v2 = s1.dropna(), s2.dropna()
            if len(v1) < 2 or len(v2) < 2:
                return '-'
            try:
                stat, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                return f"{p:.3f}" if p >= 0.001 else "<0.001"
            except Exception:
                return '-'

        def calc_pvalue_categorical(s1, s2, categories):
            """分类变量 p 值 (Chi-square)"""
            try:
                obs1 = [int((s1 == c).sum()) for c in categories]
                obs2 = [int((s2 == c).sum()) for c in categories]
                # 去除全0的类别
                valid_idx = [i for i in range(len(categories)) if obs1[i] + obs2[i] > 0]
                if len(valid_idx) < 2:
                    return '-'
                table = [[obs1[i], obs2[i]] for i in valid_idx]
                chi2, p, dof, expected = stats.chi2_contingency(table)
                return f"{p:.3f}" if p >= 0.001 else "<0.001"
            except Exception:
                return '-'

        def calc_smd_continuous(s1, s2):
            try:
                return _format_smd_value(_compute_smd_continuous(s1, s2))
            except Exception:
                return '-'

        def calc_smd_binary(s1, s2, positive_category):
            try:
                binary1 = (s1 == positive_category).astype(int)
                binary2 = (s2 == positive_category).astype(int)
                return _format_smd_value(_compute_smd_binary(binary1, binary2))
            except Exception:
                return '-'

        # 构建表格数据 - 根据选中的模块动态生成
        table_data = []

        # 样本量 (总是显示)
        table_data.append({
            'Module': '',
            'Characteristic': 'N' if lang == 'en' else '样本量',
            group1_name: f"{len(group1_df):,}",
            group2_name: f"{len(group2_df):,}",
            'p-value': '',
            'SMD': '',
        })

        # 遍历选中的模块
        for mod_key in selected_modules:
            mod_info = FEATURE_MODULES[mod_key]
            mod_name = mod_info['name_en'] if lang == 'en' else mod_info['name_zh']
            is_first_in_module = True

            for feat_info in mod_info['features']:
                feat_key = feat_info[0]
                feat_name_en = feat_info[1]
                feat_name_zh = feat_info[2]
                feat_type = feat_info[3]

                feat_display = feat_name_en if lang == 'en' else feat_name_zh
                module_display = mod_name if is_first_in_module else ''
                is_first_in_module = False

                # 处理不同类型的特征
                if mod_key == 'demographic':
                    if feat_key == 'age' and 'age' in group1_df.columns:
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_continuous(group1_df['age'], 'age'),
                            group2_name: format_continuous(group2_df['age'], 'age'),
                            'p-value': calc_pvalue_continuous(group1_df['age'], group2_df['age']),
                            'SMD': calc_smd_continuous(group1_df['age'], group2_df['age']),
                        })
                    elif feat_key == 'gender' and 'gender' in group1_df.columns:
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_categorical(group1_df['gender'], 'M', len(group1_df)),
                            group2_name: format_categorical(group2_df['gender'], 'M', len(group2_df)),
                            'p-value': calc_pvalue_categorical(group1_df['gender'], group2_df['gender'], ['M', 'F']),
                            'SMD': calc_smd_binary(group1_df['gender'], group2_df['gender'], 'M'),
                        })
                    elif feat_key == 'los_days' and 'los_hours' in group1_df.columns:
                        g1_los = group1_df['los_hours'] / 24
                        g2_los = group2_df['los_hours'] / 24
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_continuous(g1_los, 'los'),
                            group2_name: format_continuous(g2_los, 'los'),
                            'p-value': calc_pvalue_continuous(g1_los, g2_los),
                            'SMD': calc_smd_continuous(g1_los, g2_los),
                        })
                    elif feat_key == 'first_icu_stay' and 'first_icu_stay' in group1_df.columns:
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_categorical(group1_df['first_icu_stay'], True, len(group1_df)),
                            group2_name: format_categorical(group2_df['first_icu_stay'], True, len(group2_df)),
                            'p-value': calc_pvalue_categorical(group1_df['first_icu_stay'], group2_df['first_icu_stay'], [True, False]),
                            'SMD': calc_smd_binary(group1_df['first_icu_stay'], group2_df['first_icu_stay'], True),
                        })

                elif mod_key == 'outcome':
                    if feat_key == 'mortality' and 'survived' in group1_df.columns and show_mortality:
                        mort1 = (1 - group1_df['survived'].mean()) * 100
                        mort2 = (1 - group2_df['survived'].mean()) * 100
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: f"{int((group1_df['survived']==0).sum()):,} ({mort1:.1f}%)",
                            group2_name: f"{int((group2_df['survived']==0).sum()):,} ({mort2:.1f}%)",
                            'p-value': calc_pvalue_categorical(group1_df['survived'], group2_df['survived'], [0, 1]),
                            'SMD': calc_smd_binary(group1_df['survived'], group2_df['survived'], 0),
                        })

                else:
                    # 从加载的特征数据获取
                    # 首先尝试从 group_df 的列获取
                    if feat_key in group1_df.columns:
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_continuous(group1_df[feat_key], feat_key),
                            group2_name: format_continuous(group2_df[feat_key], feat_key),
                            'p-value': calc_pvalue_continuous(group1_df[feat_key], group2_df[feat_key]),
                            'SMD': calc_smd_continuous(group1_df[feat_key], group2_df[feat_key]),
                        })
                    # 如果没在 group_df 中，尝试直接从 feature_data 获取
                    elif feat_key in feature_data:
                        feat_df = feature_data[feat_key]
                        # 检测ID列
                        feat_id_col = None
                        for col in ['stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
                            if col in feat_df.columns:
                                feat_id_col = col
                                break
                        if feat_id_col is None:
                            feat_id_col = id_col
                        # 按组筛选
                        g1_ids_set = set(group1_df[id_col].astype(int).tolist())
                        g2_ids_set = set(group2_df[id_col].astype(int).tolist())
                        g1_vals = feat_df[feat_df[feat_id_col].astype(int).isin(g1_ids_set)][feat_key]
                        g2_vals = feat_df[feat_df[feat_id_col].astype(int).isin(g2_ids_set)][feat_key]

                        if len(g1_vals) > 0 or len(g2_vals) > 0:
                            table_data.append({
                                'Module': module_display,
                                'Characteristic': feat_display,
                                group1_name: format_continuous(g1_vals, feat_key) if len(g1_vals) > 0 else 'N/A',
                                group2_name: format_continuous(g2_vals, feat_key) if len(g2_vals) > 0 else 'N/A',
                                'p-value': calc_pvalue_continuous(g1_vals, g2_vals) if len(g1_vals) > 0 and len(g2_vals) > 0 else '-',
                                'SMD': calc_smd_continuous(g1_vals, g2_vals) if len(g1_vals) > 0 and len(g2_vals) > 0 else '-',
                            })
                        else:
                            table_data.append({
                                'Module': module_display,
                                'Characteristic': feat_display,
                                group1_name: 'No data',
                                group2_name: 'No data',
                                'p-value': '-',
                                'SMD': '-',
                            })
                    elif feat_key in concepts_to_load:
                        # 特征需要加载但尚未加载
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: '⏳ 待加载',
                            group2_name: '⏳ 待加载',
                            'p-value': '-',
                            'SMD': '-',
                        })

        # 显示表格
        result_df = pd.DataFrame(table_data)

        if screenshot_mode:
            figure_df = result_df.head(16).fillna('').astype(str)
            st.markdown(
                f'<div class="figure-table">{figure_df.to_html(index=False, escape=True)}</div>',
                unsafe_allow_html=True,
            )
        else:
            table_height = min(760, max(460, 42 + 35 * (len(result_df) + 1)))
            # 使用 Streamlit 表格并应用样式
            _st_dataframe_compat(
                st,
                result_df,
                width='stretch',
                height=table_height,
                hide_index=True,
                column_config={
                    'Module': st.column_config.TextColumn('Module' if lang == 'en' else '模块', width='small'),
                    'Characteristic': st.column_config.TextColumn('Characteristic' if lang == 'en' else '特征', width='medium'),
                    group1_name: st.column_config.TextColumn(group1_name, width='medium'),
                    group2_name: st.column_config.TextColumn(group2_name, width='medium'),
                    'p-value': st.column_config.TextColumn('p-value', width='small'),
                    'SMD': st.column_config.TextColumn('SMD', width='small'),
                }
            )

            # 统计方法说明
            st.markdown("---")
            stats_note = """**Statistical Methods:**
- Continuous variables: Mean ± SD (Median [IQR]), Mann-Whitney U test, SMD with pooled SD
- Categorical variables: n (%), Chi-square test, binary SMD with pooled proportion
- SMD flags: 🟠 |SMD| > 0.10, 🔴 |SMD| > 0.25
- p < 0.05 considered statistically significant""" if lang == 'en' else """**统计方法说明：**
- 连续变量：Mean ± SD (Median [IQR])，Mann-Whitney U 检验，SMD 使用合并标准差
- 分类变量：n (%)，卡方检验，二分类 SMD 使用合并比例
- SMD 标记：🟠 |SMD| > 0.10，🔴 |SMD| > 0.25
- p < 0.05 认为具有统计学显著性"""
            st.caption(stats_note)

            # 🔧 FIX (2026-02-04): 简化导出逻辑，使用 UTF-8 BOM 编码确保 Excel 正确显示
            # 无需手动替换特殊字符，utf-8-sig 编码可以正确处理
            export_df = result_df.copy()

            # 只清理 emoji（这些可能导致问题）
            for col in export_df.columns:
                if export_df[col].dtype == 'object':
                    export_df[col] = export_df[col].apply(lambda x: strip_emoji(str(x)) if pd.notna(x) else x)

            # 使用 BytesIO 确保编码正确传递
            import io
            buffer = io.BytesIO()
            export_df.to_csv(buffer, index=False, encoding='utf-8-sig')
            csv_bytes = buffer.getvalue()

            st.download_button(
                label="📥 " + ("Download Table (CSV)" if lang == 'en' else "下载表格 (CSV)"),
                data=csv_bytes,
                file_name=f"baseline_comparison_{group1_name}_vs_{group2_name}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
