"""Sidebar rendering workflow for the EasyICU Streamlit app.

This is a transitional extraction: the sidebar still calls app-level helpers,
but the long Streamlit rendering block no longer lives in app.py.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import os

import streamlit as st


_PROTECTED_CONTEXT_NAMES = {"render_sidebar", "_install_app_context"}


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to the extracted sidebar."""
    for name, value in app_context.items():
        if not name.startswith("__") and name not in _PROTECTED_CONTEXT_NAMES:
            globals()[name] = value


def _ensure_default_directory_input_value(
    *,
    input_key: str,
    default_key: str,
    default_value: str,
) -> None:
    """Keep an empty/default directory input aligned without overwriting custom values."""
    previous_default = st.session_state.get(default_key)
    current_value = st.session_state.get(input_key)
    if not current_value:
        st.session_state[input_key] = default_value
    elif previous_default is not None and current_value == previous_default and previous_default != default_value:
        st.session_state[input_key] = default_value
    st.session_state[default_key] = default_value


def render_sidebar(app_context: dict[str, Any] | None = None):
    """渲染侧边栏 - 根据entry_mode显示不同内容。"""
    if app_context is not None:
        _install_app_context(app_context)

    # 使用双语特征分组
    concept_groups = get_concept_groups()

    # 所有可用的 concepts 列表（用于自定义选择）
    all_available_concepts = sorted(set(c for group_concepts in concept_groups.values() for c in group_concepts))

    # 获取当前模式
    entry_mode = st.session_state.get('entry_mode', 'none')

    with st.sidebar:
        # � 展开/收起按钮
        expand_col1, expand_col2 = st.columns([3, 1])
        with expand_col2:
            if st.session_state.sidebar_expanded:
                expand_label = "⬅️" if st.session_state.language == 'en' else "⬅️"
                expand_help = "Collapse sidebar" if st.session_state.language == 'en' else "收起侧边栏"
            else:
                expand_label = "⤢" if st.session_state.language == 'en' else "⤢"
                expand_help = "Expand to full width" if st.session_state.language == 'en' else "展开到全屏"

            if st.button(expand_label, key="toggle_sidebar_expand", help=expand_help):
                st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
                st.rerun()

        # �🔙 返回入口页面按钮（始终显示，除非在入口页）
        if entry_mode != 'none':
            back_label = "🔙 Back to Mode Selection" if st.session_state.language == 'en' else "🔙 返回模式选择"
            if st.button(back_label, key="back_to_entry", use_container_width=True):
                st.session_state.entry_mode = 'none'
                # 清空所有数据
                st.session_state.loaded_concepts = {}
                st.session_state.loaded_data_origin = 'none'
                st.session_state.patient_ids = []
                st.session_state.use_mock_data = False
                # 清理Cohort相关缓存
                for key in ['group_a_data', 'group_b_data', 'multidb_data', 'dash_demographics',
                            'multidb_is_demo', 'dash_is_demo', 'cohort_is_demo']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            st.markdown("---")

        # 显示当前模式标识 - 精简 pill 样式
        if entry_mode == 'demo':
            mode_badge = "Demo Mode" if st.session_state.language == 'en' else "演示模式"
            st.markdown(f"""
            <div style="background:#ecfdf5;border:1px solid #a7f3d0;
                        padding:8px 14px;border-radius:8px;color:#065f46;margin-bottom:12px;
                        display:flex;align-items:center;gap:8px;font-size:.9rem">
                <span style="width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block"></span>
                <b>{mode_badge}</b>
            </div>
            """, unsafe_allow_html=True)
        elif entry_mode == 'real':
            mode_badge = "Real Data" if st.session_state.language == 'en' else "真实数据"
            st.markdown(f"""
            <div style="background:#eef2ff;border:1px solid #c7d2fe;
                        padding:8px 14px;border-radius:8px;color:#3730a3;margin-bottom:12px;
                        display:flex;align-items:center;gap:8px;font-size:.9rem">
                <span style="width:8px;height:8px;border-radius:50%;background:#6366f1;display:inline-block"></span>
                <b>{mode_badge}</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"## {get_text('app_title')}")

        # 语言切换 - 更紧凑的布局
        lang = st.selectbox(
            "🌐 Language",
            options=['EN', 'ZH'],
            index=0 if st.session_state.language == 'en' else 1,
            key="lang_select",
        )
        if (lang == 'EN' and st.session_state.language != 'en') or \
           (lang == 'ZH' and st.session_state.language != 'zh'):
            st.session_state.language = 'en' if lang == 'EN' else 'zh'
            st.rerun()

        st.markdown("---")

        # ============ AI 助手设置（放在侧边栏最上方，方便用户看到）============
        try:
            from easyicu.webapp.llm_chat import render_llm_settings
            render_llm_settings()
        except Exception:
            pass  # silently skip if module unavailable

        st.markdown("---")

        # ============ 侧边栏仅用于数据提取导出模式 ============
        # 快速可视化功能已移至主页面的 "Quick Visualization" 标签页

        sidebar_title = "📤 Data Extraction" if st.session_state.language == 'en' else "📤 数据提取导出"
        st.markdown(f"### {sidebar_title}")

        # 🔧 FIX (2026-02-03): 导出完成后显示"重新提取"按钮，而非Step 1-4
        if st.session_state.get('export_completed', False):
            # 显示导出成功信息
            success_msg = "✅ Export Completed!" if st.session_state.language == 'en' else "✅ 导出完成！"
            export_dir = st.session_state.get('last_export_dir', '')
            st.success(success_msg)
            if export_dir:
                path_msg = f"📂 {export_dir}"
                st.info(path_msg)

            # 显示导出统计
            result = st.session_state.get('_export_success_result', {})
            if result:
                n_files = len(result.get('files', []))
                n_patients = result.get('patient_count', 0)
                stats_label = f"📊 {n_files} files, {n_patients} patients" if st.session_state.language == 'en' else f"📊 {n_files} 个文件, {n_patients} 个患者"
                st.caption(stats_label)
                if result.get('note'):
                    st.info(result['note'])

            # 显示队列筛选统计
            cohort_stats = st.session_state.get('_cohort_stats')
            if cohort_stats and cohort_stats.get('excluded', 0) > 0:
                n_before = cohort_stats['before']
                n_excluded = cohort_stats['excluded']
                n_after = cohort_stats['after']
                details = cohort_stats.get('filter_details', [])
                if st.session_state.language == 'en':
                    cohort_info = f"👥 **Cohort Selection**: {n_before} candidates → **{n_after} patients** exported ({n_excluded} excluded)"
                    if details:
                        reasons = ", ".join(f"{label_en}: -{cnt}" for label_en, _, cnt in details if cnt > 0)
                        if reasons:
                            cohort_info += f"\n\nExclusion reasons: {reasons}"
                else:
                    cohort_info = f"👥 **队列筛选**: {n_before} 候选 → 最终导出 **{n_after} 位患者**（排除 {n_excluded} 人）"
                    if details:
                        reasons = "、".join(f"{label_cn}: -{cnt}人" for _, label_cn, cnt in details if cnt > 0)
                        if reasons:
                            cohort_info += f"\n\n排除原因: {reasons}"
                st.info(cohort_info)

            st.markdown("---")

            # 重新提取按钮
            restart_label = "🔄 Start New Extraction" if st.session_state.language == 'en' else "🔄 重新提取"
            restart_help = "Reset all settings and start a new extraction" if st.session_state.language == 'en' else "重置所有设置并开始新的数据提取"
            if st.button(restart_label, type="primary", use_container_width=True, key="restart_extraction", help=restart_help):
                # 重置所有导出相关状态
                st.session_state.export_completed = False
                st.session_state.trigger_export = False
                st.session_state.step1_confirmed = False
                st.session_state.step2_confirmed = False
                st.session_state.selected_concepts = []
                st.session_state.concept_checkboxes = {}
                st.session_state.selected_groups = []
                st.session_state.loaded_concepts = {}
                st.session_state.loaded_data_origin = 'none'
                # 🔧 FIX (2026-02-15): 重置采样参数，避免上次提取的 patient_limit/patient_ids 泄露到新提取
                st.session_state.patient_limit = 1000  # 重置为默认值
                st.session_state.patient_ids = []
                st.session_state.all_patient_count = 0
                st.session_state.pop('_viz_auto_load_export', None)
                st.session_state.pop('_export_success_result', None)
                # 🔧 FIX (2026-02-15): 清除 easyicu 内部缓存，避免上次提取的数据影响新提取
                try:
                    from easyicu.cache_manager import clear_easyicu_cache
                    clear_easyicu_cache()
                except Exception:
                    pass
                try:
                    from easyicu.api import clear_global_loader
                    clear_global_loader()
                except Exception:
                    pass
                # 清理导出结果
                if '_export_success_result' in st.session_state:
                    del st.session_state['_export_success_result']
                if '_cohort_stats' in st.session_state:
                    del st.session_state['_cohort_stats']
                if '_skipped_modules' in st.session_state:
                    del st.session_state['_skipped_modules']
                if '_overwrite_modules' in st.session_state:
                    del st.session_state['_overwrite_modules']
                if '_exporting_in_progress' in st.session_state:
                    del st.session_state['_exporting_in_progress']
                if '_viz_import_export_auto_trigger' in st.session_state:
                    del st.session_state['_viz_import_export_auto_trigger']
                st.rerun()

            # 返回首页按钮
            home_label = "🏠 Back to Home" if st.session_state.language == 'en' else "🏠 返回首页"
            if st.button(home_label, use_container_width=True, key="back_to_home_after_export"):
                st.session_state.active_page = 'home_extract'
                st.rerun()

            return  # 不显示后续Step内容

        # ============ 步骤1: 数据源选择 ============
        # 🆕 根据entry_mode决定显示内容，不再允许切换

        if entry_mode == 'demo':
            # ===== DEMO 模式：只显示模拟数据参数，不显示数据库选择 =====
            st.markdown(f"### 📊 {get_text('step1')}")
            demo_title = "Demo Mode" if st.session_state.language == 'en' else "演示模式"
            demo_desc = "Auto-generated simulated ICU data" if st.session_state.language == 'en' else "自动生成模拟ICU数据"
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                        padding:10px 14px;border-radius:10px;margin:6px 0 10px">
                <div style="font-weight:600;color:#166534;font-size:.92rem">{demo_title}</div>
                <div style="color:#4ade80;font-size:.78rem;margin-top:2px">{demo_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.database = 'mock'
            st.session_state.use_mock_data = True

            # 模拟数据参数
            n_patients_label = "Number of Patients" if st.session_state.language == 'en' else "患者数量"
            hours_label = "Data Duration (hours)" if st.session_state.language == 'en' else "数据时长(小时)"
            if 'demo_mode_patients' not in st.session_state:
                st.session_state.demo_mode_patients = st.session_state.mock_params.get('n_patients', 100)
            if 'demo_mode_hours' not in st.session_state:
                st.session_state.demo_mode_hours = st.session_state.mock_params.get('hours', 72)

            n_patients = st.slider(
                n_patients_label,
                50,
                500,
                key="demo_mode_patients",
            )
            hours = st.slider(
                hours_label,
                24,
                168,
                key="demo_mode_hours",
            )
            # 🔧 注意: mock_params 需要在 Step 2 (Cohort Selection) 之后更新
            # 这里只保存基本参数，cohort_filter 在 Step 2 之后的函数中动态获取
            st.session_state.mock_params = {'n_patients': n_patients, 'hours': hours}

            # ✅ Step 1 确认按钮
            step1_confirm_label = "✅ Confirm Data Source" if st.session_state.language == 'en' else "✅ 确认数据源配置"
            if st.button(step1_confirm_label, type="primary", use_container_width=True, key="step1_confirm_demo"):
                st.session_state.step1_confirmed = True
                step1_done_msg = "✅ Step 1 completed! Proceed to Step 2: Cohort Selection" if st.session_state.language == 'en' else "✅ 步骤1已完成！请继续步骤2: 队列筛选"
                st.success(step1_done_msg)

        elif entry_mode == 'real':
            # ===== REAL DATA 模式：只显示数据库选择，不显示Demo选项 =====
            st.markdown(f"### 📊 {get_text('step1')}")

            # 🔧 自动检测数据库：根据路径中的关键词自动选择
            def detect_database_from_path(path: str) -> str:
                """根据路径自动检测数据库类型"""
                if not path:
                    return st.session_state.get('database', 'miiv')
                path_lower = path.lower()
                if 'hirid' in path_lower:
                    return 'hirid'
                elif 'eicu' in path_lower:
                    return 'eicu'
                elif 'aumc' in path_lower or 'amsterdam' in path_lower:
                    return 'aumc'
                elif 'mimiciii' in path_lower or 'mimic-iii' in path_lower or 'mimic_iii' in path_lower or 'mimic3' in path_lower:
                    return 'mimic'
                elif 'mimiciv' in path_lower or 'mimic-iv' in path_lower or 'mimic_iv' in path_lower or 'mimic4' in path_lower:
                    return 'miiv'
                elif 'sic' in path_lower:
                    return 'sic'
                return st.session_state.get('database', 'miiv')

            db_options = ['miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic']
            detected_db = detect_database_from_path(st.session_state.get('data_path', ''))
            default_idx = db_options.index(detected_db) if detected_db in db_options else 0

            db_label = "Select Database" if st.session_state.language == 'en' else "选择数据库"
            database = st.selectbox(
                db_label,
                options=db_options,
                index=default_idx,
                format_func=lambda x: {
                    'miiv': 'MIMIC-IV', 'eicu': 'eICU-CRD',
                    'aumc': 'AmsterdamUMCdb', 'hirid': 'HiRID',
                    'mimic': 'MIMIC-III', 'sic': 'SICdb'
                }.get(x, x)
            )
            st.session_state.database = database
            st.session_state.use_mock_data = False

            # 数据库简称 → 显示名映射
            _db_display_names = {
                'miiv': 'MIMIC-IV', 'eicu': 'eICU-CRD',
                'aumc': 'AmsterdamUMCdb', 'hirid': 'HiRID',
                'mimic': 'MIMIC-III', 'sic': 'SICdb'
            }
            _db_display = _db_display_names.get(database, database)

            import platform
            _is_win = platform.system() == 'Windows'
            _placeholder = f"D:\\data\\{database}" if _is_win else f"/path/to/{database}"
            _hint = f"Enter the path to your {_db_display} data directory" if st.session_state.language == 'en' else f"请输入 {_db_display} 数据目录的路径"

            path_label = "Data Path" if st.session_state.language == 'en' else "数据路径"
            data_path = _directory_input(
                path_label,
                value=st.session_state.data_path or "",
                input_key="sidebar_data_path_input",
                button_key="sidebar_data_path_browse",
                placeholder=_placeholder,
                help=_hint,
            )

            # 🔧 当路径变化时自动检测并更新数据库
            if data_path and data_path != st.session_state.get('_last_data_path', ''):
                detected_db = detect_database_from_path(data_path)
                if detected_db != database:
                    st.session_state.database = detected_db
                    st.session_state._last_data_path = data_path
                    st.rerun()
                st.session_state._last_data_path = data_path

            # 验证按钮
            validate_btn = "🔍 Validate Data Path" if st.session_state.language == 'en' else "🔍 验证数据路径"
            if st.button(validate_btn, use_container_width=True, key="validate_path"):
                if not data_path:
                    err_msg = "❌ Please enter data path" if st.session_state.language == 'en' else "❌ 请输入数据路径"
                    st.error(err_msg)
                elif not Path(data_path).exists():
                    err_msg = "❌ Path does not exist" if st.session_state.language == 'en' else "❌ 路径不存在"
                    st.error(err_msg)
                else:
                    # 检查数据库所需文件
                    validation_result = validate_database_path(data_path, database)
                    st.session_state.last_validation = validation_result
                    st.session_state.last_validated_path = data_path

                    if validation_result['valid']:
                        st.session_state.data_path = data_path
                        st.session_state.path_validated = True
                        st.success(validation_result['message'])
                    else:
                        st.session_state.path_validated = False
                        st.error(validation_result['message'])
                        if validation_result.get('suggestion'):
                            st.info(validation_result['suggestion'])

            # 显示当前验证状态和转换按钮
            last_validation = st.session_state.get('last_validation', {})
            last_path = st.session_state.get('last_validated_path', '')

            if st.session_state.get('path_validated') and st.session_state.data_path == data_path:
                validated_msg = "✅ Path validated" if st.session_state.language == 'en' else "✅ 路径已验证"
                st.success(validated_msg)
            elif last_validation.get('can_convert') and last_path == data_path:
                # 显示转换按钮
                convert_btn = "🔄 Convert & Setup" if st.session_state.language == 'en' else "🔄 转换并设置"
                if st.button(convert_btn, use_container_width=True, type="primary", key="convert_csv"):
                    st.session_state.show_convert_dialog = True
                    st.session_state.convert_source_path = data_path
                    st.rerun()
                convert_hint = "💡 One-click: convert → validate → ready" if st.session_state.language == 'en' else "💡 一键完成：转换 → 验证 → 就绪"
                st.caption(convert_hint)
                if last_validation.get('download_url'):
                    st.link_button(
                        last_validation.get('download_label') or (
                            "Open database download page"
                            if st.session_state.language == 'en' else
                            "打开数据库下载页"
                        ),
                        last_validation['download_url'],
                        use_container_width=True,
                    )
                    if last_validation.get('download_note'):
                        st.caption(last_validation['download_note'])
            elif last_validation and (not last_validation.get('valid')) and last_path == data_path:
                if last_validation.get('download_url'):
                    st.link_button(
                        last_validation.get('download_label') or (
                            "Open database download page"
                            if st.session_state.language == 'en' else
                            "打开数据库下载页"
                        ),
                        last_validation['download_url'],
                        use_container_width=True,
                    )
                    if last_validation.get('download_note'):
                        st.caption(last_validation['download_note'])
            elif data_path and Path(data_path).exists():
                validate_hint = "💡 Click the button above to validate data format" if st.session_state.language == 'en' else "💡 点击上方按钮验证数据格式"
                st.caption(validate_hint)

        st.markdown("---")

        # ============ 步骤2: 队列筛选（新增） ============
        step2_cohort_title = "Step 2: Cohort Selection" if st.session_state.language == 'en' else "步骤2: 队列筛选"
        st.markdown(f"### 👥 {step2_cohort_title}")

        # 🔧 FIX (2026-02-03): 检查步骤依赖 - Step1必须先完成
        use_mock = st.session_state.get('use_mock_data', False)
        if use_mock:
            step1_complete = st.session_state.get('step1_confirmed', False)
        else:
            step1_complete = st.session_state.data_path and Path(st.session_state.data_path).exists()

        if not step1_complete:
            if use_mock:
                step_dep_msg = (
                    "ℹ️ Confirm Step 1 to configure the extraction workflow. Cohort demo panels are already available in Cohort Analysis."
                    if st.session_state.language == 'en' else
                    "ℹ️ 如需配置提取流程，请先确认步骤1。队列分析的演示面板已可直接使用。"
                )
                st.info(step_dep_msg)
            else:
                # 提示用户先完成Step1
                step_dep_msg = "⚠️ Please complete Step 1 first" if st.session_state.language == 'en' else "⚠️ 请先完成步骤1"
                st.warning(step_dep_msg)
            return  # 不渲染后续内容

        # 初始化队列筛选的 session state
        if 'cohort_filter' not in st.session_state:
            st.session_state.cohort_filter = {
                'age_min': None,
                'age_max': None,
                'first_icu_stay': None,
                'los_min': None,
                'los_max': None,
                'gender': None,
                'survived': None,
                'has_sepsis': None,
                'disease_cohort': 'none',
                'icd_query': '',
                'icd_include_query': '',
                'icd_exclude_query': '',
                'icd_mode': 'include',
            }
        # Upgrade older sessions that only had a single ICD box.
        if 'icd_include_query' not in st.session_state.cohort_filter:
            legacy_query = str(st.session_state.cohort_filter.get('icd_query', '')).strip()
            legacy_mode = st.session_state.cohort_filter.get('icd_mode', 'include')
            st.session_state.cohort_filter['icd_include_query'] = legacy_query if legacy_mode != 'exclude' else ''
            st.session_state.cohort_filter['icd_exclude_query'] = legacy_query if legacy_mode == 'exclude' else ''
        else:
            st.session_state.cohort_filter.setdefault('icd_exclude_query', '')
        # Keep legacy keys for backward compatibility with existing export/session logic.
        st.session_state.cohort_filter.setdefault('icd_query', '')
        st.session_state.cohort_filter.setdefault('icd_mode', 'include')
        st.session_state.setdefault('sepsis_si_mode', 'auto')
        st.session_state.setdefault('sepsis_abx_win_hours', 24)
        st.session_state.setdefault('sepsis_samp_win_hours', 72)
        st.session_state.setdefault('sepsis_positive_cultures', False)
        if 'cohort_enabled' not in st.session_state:
            st.session_state.cohort_enabled = False
        if 'filtered_patient_count' not in st.session_state:
            st.session_state.filtered_patient_count = None

        # 启用队列筛选开关 - 使用 key 参数让 Streamlit 自动管理状态
        cohort_toggle_label = "Enable Cohort Filtering" if st.session_state.language == 'en' else "启用队列筛选"
        cohort_help = "Filter patients by demographics and clinical criteria" if st.session_state.language == 'en' else "根据人口统计学和临床标准筛选患者"
        st.toggle(cohort_toggle_label, key="cohort_enabled", help=cohort_help)

        # 从 session_state 获取当前值（由 toggle 的 key 自动更新）
        cohort_enabled = st.session_state.cohort_enabled

        if cohort_enabled:
            # 紧凑年龄筛选：0 / 120 代表不限制，避免单独 expander 占高
            age_col1, age_col2 = st.columns(2)
            with age_col1:
                age_min = st.number_input(
                    "🎂 Min Age" if st.session_state.language == 'en' else "🎂 最小年龄",
                    min_value=0,
                    max_value=120,
                    value=0 if st.session_state.cohort_filter['age_min'] is None else int(st.session_state.cohort_filter['age_min']),
                    key="cohort_age_min"
                )
                st.session_state.cohort_filter['age_min'] = age_min if age_min > 0 else None
            with age_col2:
                age_max = st.number_input(
                    "🎂 Max Age" if st.session_state.language == 'en' else "🎂 最大年龄",
                    min_value=0,
                    max_value=120,
                    value=120 if st.session_state.cohort_filter['age_max'] is None else int(st.session_state.cohort_filter['age_max']),
                    key="cohort_age_max"
                )
                st.session_state.cohort_filter['age_max'] = age_max if age_max < 120 else None
            age_hint = "0 / 120 means no age limit" if st.session_state.language == 'en' else "0 / 120 表示不限制年龄"
            st.caption(age_hint)

            # 紧凑筛选布局：下拉框替代横向 radio，减少截图高度
            first_icu_label = "🏥 First ICU" if st.session_state.language == 'en' else "🏥 首次 ICU"
            first_icu_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'yes': 'Yes (First ICU only)' if st.session_state.language == 'en' else '是（仅首次）',
                'no': 'No (Readmissions only)' if st.session_state.language == 'en' else '否（仅再入院）',
            }
            current_first_icu = st.session_state.cohort_filter.get('first_icu_stay')
            current_first_icu_key = 'any'
            if current_first_icu is True:
                current_first_icu_key = 'yes'
            elif current_first_icu is False:
                current_first_icu_key = 'no'

            los_label = "⏱️ Min ICU Stay (hours)" if st.session_state.language == 'en' else "⏱️ 最短住院时长（小时）"
            los_help = "Minimum ICU stay duration to include patients (default 0h = no lower bound)" if st.session_state.language == 'en' else "纳入患者的最短ICU住院时长（默认0小时，表示不设下限）"

            compact_row1_col1, compact_row1_col2 = st.columns([1.3, 1.0])
            with compact_row1_col1:
                first_icu_val = st.selectbox(
                    first_icu_label,
                    options=list(first_icu_options.keys()),
                    format_func=lambda x: first_icu_options[x],
                    index=list(first_icu_options.keys()).index(current_first_icu_key),
                    key="cohort_first_icu"
                )
            with compact_row1_col2:
                los_min = st.number_input(
                    los_label, min_value=0, max_value=10000,
                    value=0 if st.session_state.cohort_filter.get('los_min') is None else int(st.session_state.cohort_filter['los_min']),
                    help=los_help,
                    key="cohort_los_min"
                )

            if first_icu_val == 'yes':
                st.session_state.cohort_filter['first_icu_stay'] = True
            elif first_icu_val == 'no':
                st.session_state.cohort_filter['first_icu_stay'] = False
            else:
                st.session_state.cohort_filter['first_icu_stay'] = None

            st.session_state.cohort_filter['los_min'] = los_min if los_min > 0 else None
            st.session_state.cohort_filter['los_max'] = None  # 不再使用max

            gender_label = "👤 Gender" if st.session_state.language == 'en' else "👤 性别"
            gender_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'M': 'Male' if st.session_state.language == 'en' else '男性',
                'F': 'Female' if st.session_state.language == 'en' else '女性',
            }
            current_gender_key = st.session_state.cohort_filter.get('gender') or 'any'

            survival_label = "💚 Survival Status" if st.session_state.language == 'en' else "💚 存活状态"
            survival_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'survived': 'Survived' if st.session_state.language == 'en' else '存活',
                'deceased': 'Deceased' if st.session_state.language == 'en' else '死亡',
            }
            current_survival = st.session_state.cohort_filter.get('survived')
            current_survival_key = 'any'
            if current_survival is True:
                current_survival_key = 'survived'
            elif current_survival is False:
                current_survival_key = 'deceased'

            compact_row2_col1, compact_row2_col2 = st.columns(2)
            with compact_row2_col1:
                gender_val = st.selectbox(
                    gender_label,
                    options=list(gender_options.keys()),
                    format_func=lambda x: gender_options[x],
                    index=list(gender_options.keys()).index(current_gender_key),
                    key="cohort_gender"
                )
            with compact_row2_col2:
                survival_val = st.selectbox(
                    survival_label,
                    options=list(survival_options.keys()),
                    format_func=lambda x: survival_options[x],
                    index=list(survival_options.keys()).index(current_survival_key),
                    key="cohort_survival"
                )

            st.session_state.cohort_filter['gender'] = gender_val if gender_val != 'any' else None

            if survival_val == 'survived':
                st.session_state.cohort_filter['survived'] = True
            elif survival_val == 'deceased':
                st.session_state.cohort_filter['survived'] = False
            else:
                st.session_state.cohort_filter['survived'] = None

            # 疾病队列筛选（任务导向）
            disease_label = "🩺 Clinical Cohort" if st.session_state.language == 'en' else "🩺 疾病队列"
            supported_diseases = _get_supported_disease_cohorts(st.session_state.get('database', 'miiv'))
            disease_options_en = {
                'none': 'Any / No disease filter',
                'sepsis': 'Sepsis-3 cohort',
                'aki': 'AKI cohort (KDIGO)',
                'circ_failure': 'Circulatory failure cohort',
                'mech_vent': 'Mechanical ventilation cohort',
                'rrt': 'Renal replacement therapy cohort',
                'ards': 'ARDS cohort',
                'pneumonia': 'Pneumonia cohort',
                'heart_failure': 'Heart failure cohort',
                'ami': 'AMI cohort',
                'stroke': 'Stroke cohort',
            }
            disease_options_zh = {
                'none': '不限 / 不做疾病筛选',
                'sepsis': '脓毒症队列（Sepsis-3）',
                'aki': 'AKI 队列（KDIGO）',
                'circ_failure': '循环衰竭队列',
                'mech_vent': '机械通气队列',
                'rrt': '肾脏替代治疗队列',
                'ards': 'ARDS 队列',
                'pneumonia': '肺炎队列',
                'heart_failure': '心力衰竭队列',
                'ami': '急性心肌梗死队列',
                'stroke': '卒中队列',
            }
            disease_options = disease_options_en if st.session_state.language == 'en' else disease_options_zh
            current_disease = st.session_state.cohort_filter.get('disease_cohort', 'none')
            if current_disease not in supported_diseases:
                current_disease = 'none'
            disease_choice = st.selectbox(
                disease_label,
                options=supported_diseases,
                format_func=lambda x: disease_options.get(x, x),
                index=supported_diseases.index(current_disease),
                key="cohort_disease_cohort",
            )
            st.session_state.cohort_filter['disease_cohort'] = disease_choice
            st.session_state.cohort_filter['has_sepsis'] = True if disease_choice == 'sepsis' else None

            if disease_choice != 'none':
                disease_cfg = DISEASE_COHORT_CONFIG.get(disease_choice, {})
                disease_desc = disease_cfg.get('description_en') if st.session_state.language == 'en' else disease_cfg.get('description_zh')
                if disease_desc:
                    st.caption(disease_desc)
                if disease_cfg.get('icd_tokens'):
                    if _supports_icd_filter(st.session_state.get('database')):
                        icd_note = (
                            f"Template ICD prefixes: {', '.join(disease_cfg['icd_tokens'])}"
                            if st.session_state.language == 'en' else
                            f"模板 ICD 前缀：{', '.join(disease_cfg['icd_tokens'])}"
                        )
                        st.caption(icd_note)
                    else:
                        _no_icd_warn = (
                            f"⚠️ This cohort requires ICD codes, but {st.session_state.get('database', '')} "
                            "does not have ICD tables. The disease filter will not take effect."
                            if st.session_state.language == 'en' else
                            f"⚠️ 此队列需要 ICD 编码，但 {st.session_state.get('database', '')} "
                            "没有 ICD 诊断表。疾病筛选将不会生效。"
                        )
                        st.warning(_no_icd_warn)

            if disease_choice == 'sepsis':
                sepsis_title = "🦠 Sepsis suspected-infection settings" if st.session_state.language == 'en' else "🦠 脓毒症疑似感染设置"
                with st.expander(sepsis_title, expanded=False):
                    sepsis_mode_cfg = {
                        key: (value['label_en'] if st.session_state.language == 'en' else value['label_zh'])
                        for key, value in SEPSIS_MODE_CONFIG.items()
                    }
                    sepsis_mode = st.selectbox(
                        "Detection mode" if st.session_state.language == 'en' else "判定模式",
                        options=list(SEPSIS_MODE_CONFIG.keys()),
                        format_func=lambda x: sepsis_mode_cfg[x],
                        index=list(SEPSIS_MODE_CONFIG.keys()).index(st.session_state.get('sepsis_si_mode', 'auto')),
                        key="sepsis_si_mode",
                    )
                    mode_desc = SEPSIS_MODE_CONFIG[sepsis_mode]['desc_en'] if st.session_state.language == 'en' else SEPSIS_MODE_CONFIG[sepsis_mode]['desc_zh']
                    st.caption(mode_desc)

                    sepsis_col1, sepsis_col2 = st.columns(2)
                    with sepsis_col1:
                        st.number_input(
                            "ABX → sampling window (hours)" if st.session_state.language == 'en' else "抗生素后采样时间窗（小时）",
                            min_value=1,
                            max_value=168,
                            value=int(st.session_state.get('sepsis_abx_win_hours', 24)),
                            key="sepsis_abx_win_hours",
                        )
                    with sepsis_col2:
                        st.number_input(
                            "Sampling → ABX window (hours)" if st.session_state.language == 'en' else "采样后抗生素时间窗（小时）",
                            min_value=1,
                            max_value=168,
                            value=int(st.session_state.get('sepsis_samp_win_hours', 72)),
                            key="sepsis_samp_win_hours",
                        )

                    st.checkbox(
                        "Require positive cultures only" if st.session_state.language == 'en' else "仅使用阳性培养",
                        value=bool(st.session_state.get('sepsis_positive_cultures', False)),
                        key="sepsis_positive_cultures",
                    )
                    sepsis_ref = (
                        "Reference note: EasyICU supports multiple suspected-infection definitions. "
                        "The strict ABX + sampling window follows common Sepsis-3 operational logic; "
                        "eICU defaults to ICD + antibiotics because microbiology coverage is sparse."
                        if st.session_state.language == 'en' else
                        "参考说明：EasyICU 支持多种疑似感染定义。严格的“抗生素 + 采样”时间窗对应常见的 Sepsis-3 操作化逻辑；"
                        "eICU 默认采用“ICD + 抗生素”，因为微生物采样覆盖较稀疏。"
                    )
                    st.info(sepsis_ref)
                    _render_sepsis_ai_button(st.session_state.language)

            if _supports_icd_filter(st.session_state.get('database')):
                icd_include_label = "🧾 ICD include" if st.session_state.language == 'en' else "🧾 ICD 包含"
                icd_exclude_label = "🧾 ICD exclude" if st.session_state.language == 'en' else "🧾 ICD 排除"
                icd_help = (
                    "Use commas for multiple ICD prefixes, e.g. A41,A42. Simple ranges like A41-42 are also supported. For eICU, keywords also work."
                    if st.session_state.language == 'en' else
                    "对于 MIMIC 数据库，可用逗号输入多个 ICD 前缀，如 A41,A42；也支持 A41-42 这种简单范围写法。对于 eICU，也可输入关键词。"
                )
                icd_col1, icd_col2 = st.columns(2)
                with icd_col1:
                    icd_include_value = st.text_input(
                        icd_include_label,
                        value=st.session_state.cohort_filter.get('icd_include_query', ''),
                        help=icd_help,
                        key="cohort_icd_include_query",
                        placeholder="A41,A42 or A41-42" if st.session_state.language == 'en' else "A41,A42 或 A41-42",
                    )
                    st.session_state.cohort_filter['icd_include_query'] = icd_include_value.strip()
                with icd_col2:
                    icd_exclude_value = st.text_input(
                        icd_exclude_label,
                        value=st.session_state.cohort_filter.get('icd_exclude_query', ''),
                        help=icd_help,
                        key="cohort_icd_exclude_query",
                        placeholder="I50,C34 or I50-51" if st.session_state.language == 'en' else "I50,C34 或 I50-51",
                    )
                    st.session_state.cohort_filter['icd_exclude_query'] = icd_exclude_value.strip()

                # 兼容旧逻辑：保留 legacy 单字段，但优先使用 include/exclude
                st.session_state.cohort_filter['icd_query'] = st.session_state.cohort_filter['icd_include_query']
                st.session_state.cohort_filter['icd_mode'] = 'include'

                icd_preview_label = "🔍 Preview ICD Match" if st.session_state.language == 'en' else "🔍 预览 ICD 匹配"
                preview_specs = [
                    ("include", icd_include_value.strip(), "include"),
                    ("exclude", icd_exclude_value.strip(), "exclude"),
                ]
                has_preview_input = any(
                    _split_query_tokens(preview_value) for _, preview_value, _ in preview_specs
                )
                if has_preview_input and st.session_state.get('data_path'):
                    if st.button(icd_preview_label, key="icd_preview_btn_combined"):
                        for preview_key, preview_value, _preview_mode in preview_specs:
                            _icd_tokens_preview = _split_query_tokens(preview_value)
                            if not _icd_tokens_preview:
                                st.session_state.pop(f'_icd_preview_cache_{preview_key}', None)
                                continue
                            _icd_preview_result = _preview_icd_match(
                                Path(st.session_state.data_path),
                                st.session_state.get('database', 'miiv'),
                                _icd_tokens_preview,
                            )
                            st.session_state[f'_icd_preview_cache_{preview_key}'] = _icd_preview_result
            else:
                st.session_state.cohort_filter['icd_query'] = ""
                st.session_state.cohort_filter['icd_include_query'] = ""
                st.session_state.cohort_filter['icd_exclude_query'] = ""
                st.session_state.cohort_filter['icd_mode'] = 'include'
                _clear_icd_preview_state()

            # 显示当前筛选条件摘要
            filter_summary = []
            cf = st.session_state.cohort_filter
            if cf['age_min'] is not None or cf['age_max'] is not None:
                age_range = f"{cf['age_min'] or 0}-{cf['age_max'] or '∞'}"
                filter_summary.append(f"Age: {age_range}" if st.session_state.language == 'en' else f"年龄: {age_range}")
            if cf['first_icu_stay'] is not None:
                filter_summary.append(f"First ICU: {'Yes' if cf['first_icu_stay'] else 'No'}" if st.session_state.language == 'en' else f"首次入ICU: {'是' if cf['first_icu_stay'] else '否'}")
            # 🔧 ADD (2026-02-05): 显示 Min ICU Stay 筛选条件
            if cf.get('los_min') is not None:
                filter_summary.append(f"Min ICU Stay: {cf['los_min']}h" if st.session_state.language == 'en' else f"最短住院: {cf['los_min']}小时")
            if cf['gender'] is not None:
                filter_summary.append(f"Gender: {cf['gender']}" if st.session_state.language == 'en' else f"性别: {'男' if cf['gender']=='M' else '女'}")
            if cf['survived'] is not None:
                filter_summary.append(f"Survived: {'Yes' if cf['survived'] else 'No'}" if st.session_state.language == 'en' else f"存活: {'是' if cf['survived'] else '否'}")
            if cf['has_sepsis'] is not None:
                filter_summary.append(f"Sepsis: {'Yes' if cf['has_sepsis'] else 'No'}" if st.session_state.language == 'en' else f"脓毒症: {'是' if cf['has_sepsis'] else '否'}")
            if cf.get('disease_cohort') and cf.get('disease_cohort') != 'none':
                disease_cfg = DISEASE_COHORT_CONFIG.get(cf['disease_cohort'], {})
                disease_label_summary = disease_cfg.get('label_en') if st.session_state.language == 'en' else disease_cfg.get('label_zh')
                if disease_label_summary:
                    filter_summary.append(disease_label_summary)
            if _supports_icd_filter(st.session_state.get('database')) and cf.get('icd_include_query'):
                filter_summary.append(
                    f"ICD include: {cf['icd_include_query']}"
                    if st.session_state.language == 'en' else
                    f"ICD 包含: {cf['icd_include_query']}"
                )
            if _supports_icd_filter(st.session_state.get('database')) and cf.get('icd_exclude_query'):
                filter_summary.append(
                    f"ICD exclude: {cf['icd_exclude_query']}"
                    if st.session_state.language == 'en' else
                    f"ICD 排除: {cf['icd_exclude_query']}"
                )

            if filter_summary:
                summary_text = " | ".join(filter_summary)
                st.info(f"📋 {summary_text}")
                # 🔧 在演示模式下提示用户过滤器将应用于模拟数据生成
                if st.session_state.get('use_mock_data', False):
                    demo_filter_hint = "✨ These filters will be applied when generating mock data" if st.session_state.language == 'en' else "✨ 这些筛选条件将在生成模拟数据时应用"
                    st.caption(demo_filter_hint)
            else:
                no_filter_msg = "No filters applied (will load all patients)" if st.session_state.language == 'en' else "未设置筛选条件（将加载所有患者）"
                st.caption(no_filter_msg)

            # ✅ Step 2 确认按钮
            step2_confirm_label = "✅ Confirm Cohort Selection" if st.session_state.language == 'en' else "✅ 确认队列筛选"
            if st.button(step2_confirm_label, type="primary", use_container_width=True, key="step2_confirm"):
                _clear_icd_preview_state()
                st.session_state.step2_confirmed = True
                step2_done_msg = "✅ Step 2 completed!" if st.session_state.language == 'en' else "✅ 步骤2已完成！"
                st.success(step2_done_msg)
        else:
            # 队列筛选禁用时的提示
            disabled_msg = "💡 Enable cohort filtering to select specific patient populations" if st.session_state.language == 'en' else "💡 启用队列筛选可选择特定患者人群"
            st.caption(disabled_msg)

            # ✅ Step 2 确认按钮（即使禁用筛选也需要确认）
            step2_confirm_label = "✅ Confirm (No Filtering)" if st.session_state.language == 'en' else "✅ 确认（不筛选）"
            if st.button(step2_confirm_label, type="primary", use_container_width=True, key="step2_confirm_no_filter"):
                st.session_state.step2_confirmed = True
                step2_done_msg = "✅ Step 2 completed! Proceed to Step 3: Select Features" if st.session_state.language == 'en' else "✅ 步骤2已完成！请继续步骤3: 选择特征"
                st.success(step2_done_msg)

        st.markdown("---")

        # ============ 步骤3: Concept 选择 ============
        step3_title = "Step 3: Select Features" if st.session_state.language == 'en' else "步骤3: 选择特征"
        st.markdown(f"### 🔧 {step3_title}")

        # 🔧 FIX (2026-02-05): 检查步骤依赖 - Step2必须先确认，否则不显示特征选择
        step2_complete = st.session_state.get('step2_confirmed', False)
        if not step2_complete:
            # 提示用户先完成Step2，不显示后续内容
            step_dep_msg = "⚠️ Please complete Step 2 first (click Confirm Cohort Selection button)" if st.session_state.language == 'en' else "⚠️ 请先完成步骤2（点击确认队列筛选按钮）"
            st.warning(step_dep_msg)
            return  # 不再显示Step 3的内容

        # 初始化 session state
        if 'concept_checkboxes' not in st.session_state:
            st.session_state.concept_checkboxes = {}
        if 'selected_groups' not in st.session_state:
            st.session_state.selected_groups = []

        selected_concepts = []

        # 使用 multiselect 管理类别选择
        valid_defaults = [g for g in st.session_state.selected_groups if g in concept_groups]

        cat_label = "Select Feature Categories" if st.session_state.language == 'en' else "选择特征类别"
        cat_help = "Multi-select, click × to remove" if st.session_state.language == 'en' else "可多选，点击 × 删除"
        cat_placeholder = "Click to select..." if st.session_state.language == 'en' else "点击选择..."

        # 添加 ALL 按钮
        col_select, col_all = st.columns([4, 1])
        with col_all:
            all_label = "ALL" if st.session_state.language == 'en' else "全选"
            if st.button(all_label, key="select_all_groups", use_container_width=True):
                st.session_state.selected_groups = list(concept_groups.keys())
                # 自动选中所有概念
                for grp in concept_groups.keys():
                    for concept in concept_groups.get(grp, []):
                        st.session_state.concept_checkboxes[concept] = True
                st.rerun()

        with col_select:
            current_selection = st.multiselect(
                cat_label,
                options=list(concept_groups.keys()),
                default=valid_defaults,
                help=cat_help,
                placeholder=cat_placeholder
            )

        # 检测变化并更新
        if current_selection != st.session_state.selected_groups:
            added_groups = set(current_selection) - set(st.session_state.selected_groups)
            for grp in added_groups:
                for concept in concept_groups.get(grp, []):
                    st.session_state.concept_checkboxes[concept] = True

            removed_groups = set(st.session_state.selected_groups) - set(current_selection)
            for grp in removed_groups:
                for concept in concept_groups.get(grp, []):
                    if concept in st.session_state.concept_checkboxes:
                        del st.session_state.concept_checkboxes[concept]

            st.session_state.selected_groups = current_selection
            st.rerun()

        # 显示已选类别的详细特征配置
        if st.session_state.selected_groups:
            import hashlib

            detail_label = "🎯 Feature Detail Configuration" if st.session_state.language == 'en' else "🎯 特征详细配置"
            with st.expander(detail_label, expanded=True):
                for group_name in st.session_state.selected_groups:
                    if group_name not in concept_groups:
                        continue
                    key_hash = hashlib.md5(group_name.encode()).hexdigest()[:8]

                    st.markdown(f"**{group_name}**")
                    group_concepts = concept_groups.get(group_name, [])
                    cols = st.columns(3)
                    for cidx, concept in enumerate(group_concepts):
                        with cols[cidx % 3]:
                            default_val = st.session_state.concept_checkboxes.get(concept, True)
                            checked = st.checkbox(concept, value=default_val, key=f"cb_{key_hash}_{concept}")
                            st.session_state.concept_checkboxes[concept] = checked
                    st.markdown("---")

            # 收集所有选中的 concepts
            for group_name in st.session_state.selected_groups:
                for concept in concept_groups.get(group_name, []):
                    if st.session_state.concept_checkboxes.get(concept, True):
                        selected_concepts.append(concept)

            selected_concepts = list(set(selected_concepts))
            selected_msg = f"✅ {len(selected_concepts)} features selected" if st.session_state.language == 'en' else f"✅ 已选 {len(selected_concepts)} 个特征"
            st.success(selected_msg)

        st.session_state.selected_concepts = selected_concepts

        # 🔧 ADD (2026-02-05): 确认选择按钮 - 只有点击后才能进入Step 4
        if len(selected_concepts) > 0:
            step3_confirm_label = "✅ Confirm Selection" if st.session_state.language == 'en' else "✅ 确认选择"
            if st.button(step3_confirm_label, type="primary", use_container_width=True, key="step3_confirm_selection"):
                st.session_state.step3_confirmed = True
                step3_done_msg = "✅ Step 3 completed! Proceed to Step 4: Export Data" if st.session_state.language == 'en' else "✅ 步骤3已完成！请继续步骤4: 导出数据"
                st.success(step3_done_msg)
                st.rerun()

            # 显示已确认状态
            if st.session_state.get('step3_confirmed', False):
                step3_confirmed_msg = "✅ Selection confirmed" if st.session_state.language == 'en' else "✅ 已确认选择"
                st.info(step3_confirmed_msg)
                if st.session_state.get('loaded_data_origin') == 'preview':
                    preview_export_hint = (
                        "ℹ️ Preview only affects the visualization tabs. Confirm Export will re-extract data from the source database using your current cohort filters and patient limit."
                        if st.session_state.language == 'en' else
                        "ℹ️ Preview 只影响可视化标签页。点击确认导出时，会按照当前队列筛选和患者数量限制，重新从源数据库提取数据。"
                    )
                    st.markdown(f'<div class="compact-inline-notice info">{preview_export_hint}</div>', unsafe_allow_html=True)

                # ============ 🔍 Preview Sample Button ============
                st.markdown("---")
                preview_title = "👁️ Quick Preview" if st.session_state.language == 'en' else "👁️ 快速预览"
                st.markdown(f"**{preview_title}**")
                preview_desc = (
                    "Load a small sample into Quick Visualization. This is optional and does not replace the final export."
                    if st.session_state.language == 'en' else
                    "将一小部分样本加载到快速可视化。此步骤可选，不会替代最终导出。"
                )
                st.caption(preview_desc)
                preview_patient_options = [10, 20, 50, 100]
                if st.session_state.get('preview_n_patients') not in preview_patient_options:
                    st.session_state.pop('preview_n_patients', None)

                preview_slider_kwargs = {
                    'options': preview_patient_options,
                    'key': "preview_n_patients",
                }
                if 'preview_n_patients' not in st.session_state:
                    preview_slider_kwargs['value'] = 10

                preview_n = st.select_slider(
                    get_text('preview_patients'),
                    **preview_slider_kwargs,
                )

                preview_btn_label = "👁️ Run Quick Preview" if st.session_state.language == 'en' else "👁️ 运行快速预览"
                if st.button(preview_btn_label, key="sidebar_preview_btn", use_container_width=True):
                    st.session_state['_preview_requested'] = True
                    st.session_state['_preview_n'] = preview_n
                    st.session_state['_scroll_to_tab'] = 'viz'
                    st.rerun()
        else:
            # 如果没有选中任何概念，重置确认状态
            st.session_state.step3_confirmed = False

        st.markdown("---")

        # ============ 步骤4: 直接导出 ============
        step4_title = "Step 4: Export Data" if st.session_state.language == 'en' else "步骤4: 导出数据"
        st.markdown(f"### 💾 {step4_title}")

        # 🔧 FIX (2026-02-05): 检查步骤依赖 - Step3必须先确认（点击确认选择按钮）
        step3_complete = st.session_state.get('step3_confirmed', False) and len(st.session_state.get('selected_concepts', [])) > 0
        if not step3_complete:
            # 提示用户先完成Step3并点击确认按钮
            step_dep_msg = "⚠️ Please complete Step 3 first (select features and click Confirm Selection)" if st.session_state.language == 'en' else "⚠️ 请先完成步骤3（选择特征并点击确认选择）"
            st.warning(step_dep_msg)
            # 不再继续显示Step4的内容
            return

        # 导出路径配置 - 实时根据数据库显示子目录，添加时间戳后缀
        import platform
        from datetime import datetime
        if platform.system() == 'Windows':
            base_export_path = r'D:\easyicu_export'
        else:
            base_export_path = os.path.expanduser('~/easyicu_export')
        db_name = st.session_state.get('database', 'mock')
        # 生成带时间戳的默认目录名（只保留年月日）
        timestamp_suffix = datetime.now().strftime('%Y%m%d')
        cohort_suffix = _generate_cohort_prefix()
        default_dir_name = f"{db_name}_{timestamp_suffix}"
        if cohort_suffix:
            default_dir_name = f"{default_dir_name}_{cohort_suffix[:48]}"
        default_export_path = str(Path(base_export_path) / default_dir_name)
        export_input_key = "sidebar_export_path_input"
        export_default_key = "_sidebar_export_path_default"
        _ensure_default_directory_input_value(
            input_key=export_input_key,
            default_key=export_default_key,
            default_value=default_export_path,
        )

        export_path = _directory_input(
            "Export Path" if st.session_state.language == 'en' else "导出路径",
            value=default_export_path,
            input_key=export_input_key,
            button_key="sidebar_export_path_browse",
            placeholder="Select export directory" if st.session_state.language == 'en' else "选择导出目录",
            help=(f"Data will be exported to this directory (Current database: {db_name.upper()})" if st.session_state.language == 'en' else f"数据将导出到此目录（当前数据库: {db_name.upper()}）")
        )
        st.session_state.export_path = export_path

        # 检查路径并提供创建选项
        if export_path:
            if Path(export_path).exists():
                path_ok_msg = "✅ Path valid" if st.session_state.language == 'en' else "✅ 路径有效"
                st.success(path_ok_msg)
            else:
                col_create, col_info = st.columns([1, 2])
                with col_create:
                    create_btn = "📁 Create Directory" if st.session_state.language == 'en' else "📁 创建目录"
                    if st.button(create_btn, key="create_export_dir"):
                        try:
                            Path(export_path).mkdir(parents=True, exist_ok=True)
                            ok_msg = "✅ Directory created" if st.session_state.language == 'en' else "✅ 目录已创建"
                            st.success(ok_msg)
                            st.rerun()
                        except Exception as e:
                            err_msg = f"Creation failed: {e}" if st.session_state.language == 'en' else f"创建失败: {e}"
                            st.error(err_msg)
                with col_info:
                    not_exist_msg = "Path does not exist" if st.session_state.language == 'en' else "路径不存在"
                    st.caption(not_exist_msg)

        # 导出格式选择（优先Parquet）
        format_label = "Export Format" if st.session_state.language == 'en' else "导出格式"
        format_help = "Parquet format is smaller and faster to load, recommended" if st.session_state.language == 'en' else "Parquet格式体积小、加载快，推荐使用"
        export_format = st.selectbox(
            format_label,
            options=['Parquet', 'CSV', 'Excel'],
            index=0,
            help=format_help
        )
        st.session_state.export_format = export_format

        # 🚀 患者数量限制（性能优化选项）
        limit_label = "Patient Limit" if st.session_state.language == 'en' else "患者数量限制"
        limit_help = "Limit number of patients to speed up loading. 0 = no limit (full data, requires large memory)" if st.session_state.language == 'en' else "限制加载的患者数量以加速。0 = 不限制（全量数据，需要大内存。超过5万患者时自动分批）"
        patient_limit_options = [100, 1000, 5000, 10000, 20000, 50000, 0]
        patient_limit_labels = {
            100: "100 (quick test)" if st.session_state.language == 'en' else "100（快速测试）",
            1000: "1,000",
            5000: "5,000",
            10000: "10,000",
            20000: "20,000",
            50000: "50,000",
            0: "All patients (auto-batch)" if st.session_state.language == 'en' else "全部患者（自动分批）"
        }
        current_limit = st.session_state.get('patient_limit', 1000)  # 🔧 FIX: 默认1000患者（全量太慢）
        if current_limit not in patient_limit_options:
            current_limit = 1000  # 🔧 FIX: 默认1000患者
        patient_limit = st.selectbox(
            limit_label,
            options=patient_limit_options,
            index=patient_limit_options.index(current_limit),
            format_func=lambda x: patient_limit_labels.get(x, str(x)),
            help=limit_help
        )
        st.session_state.patient_limit = patient_limit

        # 导出按钮
        use_mock = st.session_state.get('use_mock_data', False)
        has_loaded_data = len(st.session_state.get('loaded_concepts', {})) > 0
        loaded_data_origin = st.session_state.get('loaded_data_origin', 'none')
        is_viz_import_mode = has_loaded_data and loaded_data_origin == 'exported_files'
        can_export = (use_mock or is_viz_import_mode or (st.session_state.data_path and Path(st.session_state.data_path).exists())) and selected_concepts and export_path and Path(export_path).exists()

        # 仅对真正的“已导出文件导入模式”自动复用 loaded_concepts；Preview 不应污染导出状态机
        if is_viz_import_mode and not selected_concepts:
            selected_concepts = list(st.session_state.loaded_concepts.keys())
            st.session_state.selected_concepts = selected_concepts
            can_export = export_path and Path(export_path).exists()

        export_btn = "📥 Export Data" if st.session_state.language == 'en' else "📥 导出数据"
        if st.session_state.get('_exporting_in_progress', False):
            export_pending_msg = (
                "⏳ Export has started. EasyICU is re-extracting data from the source database, not reusing the preview sample."
                if st.session_state.language == 'en' else
                "⏳ 导出已经开始。EasyICU 正在从源数据库重新提取数据，不会直接复用刚才的预览样本。"
            )
            st.info(export_pending_msg)
        if can_export:
            # ============ Final Sanity Review ============
            sanity_title = get_text('sanity_title')
            with st.expander(sanity_title, expanded=True):
                _sc1, _sc2, _sc3 = st.columns(3)
                _n_feats = len(selected_concepts)
                _n_mods = len(set(g for g, cs in CONCEPT_GROUPS_INTERNAL.items() if any(c in selected_concepts for c in cs)))
                _pat_lim = st.session_state.get('patient_limit', 0)
                _pat_str = str(_pat_lim) if _pat_lim and _pat_lim > 0 else ("All" if st.session_state.language == 'en' else "全部")
                with _sc1:
                    st.markdown(
                        f'<div class="compact-summary-card"><div class="summary-label">{get_text("sanity_patients")}</div><div class="summary-value">{_pat_str}</div></div>',
                        unsafe_allow_html=True,
                    )
                with _sc2:
                    st.markdown(
                        f'<div class="compact-summary-card"><div class="summary-label">{get_text("sanity_features")}</div><div class="summary-value">{_n_feats}</div></div>',
                        unsafe_allow_html=True,
                    )
                with _sc3:
                    st.markdown(
                        f'<div class="compact-summary-card"><div class="summary-label">{get_text("sanity_format")}</div><div class="summary-value">{export_format}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**{get_text('sanity_path')}:** `{export_path}`")
                st.markdown(f"**{get_text('sanity_modules')}:** {_n_mods}")

            _col_ok, _col_back = st.columns(2)
            with _col_ok:
                if st.button(get_text('sanity_confirm'), type="primary", use_container_width=True, key="final_export_btn"):
                    st.session_state.trigger_export = True
                    st.session_state.export_completed = False
                    st.session_state['_exporting_in_progress'] = True
                    st.session_state['_scroll_to_top'] = True
                    st.rerun()
            with _col_back:
                if st.button(get_text('sanity_back'), use_container_width=True, key="sanity_back_btn"):
                    st.session_state.step3_confirmed = False
                    st.rerun()
        else:
            st.button(export_btn, type="primary", use_container_width=True, disabled=True)
            if not selected_concepts:
                feat_warn = "⚠️ Please select features first" if st.session_state.language == 'en' else "⚠️ 请先选择特征"
                st.caption(feat_warn)
            elif not use_mock and not st.session_state.data_path:
                path_warn = "⚠️ Please set data path first" if st.session_state.language == 'en' else "⚠️ 请先设置数据路径"
                st.caption(path_warn)

        # ============ 系统资源信息 ============
        st.markdown("---")
        resources = get_system_resources()
        perf_title = "⚡ Performance" if st.session_state.language == 'en' else "⚡ 性能配置"
        with st.expander(perf_title, expanded=False):
            if st.session_state.language == 'en':
                st.markdown(f"""
                **System Resources:**
                - 🖥️ CPU: {resources['cpu_count']} cores
                - 💾 RAM: {resources['total_memory_gb']} GB total
                - 📊 Available: {resources['available_memory_gb']} GB

                **Auto-optimized:**
                - Workers: {resources['recommended_workers']}
                - Backend: {resources['recommended_backend']}
                """)
            else:
                st.markdown(f"""
                **系统资源:**
                - 🖥️ CPU: {resources['cpu_count']} 核心
                - 💾 内存: {resources['total_memory_gb']} GB 总计
                - 📊 可用: {resources['available_memory_gb']} GB

                **自动优化配置:**
                - 并行数: {resources['recommended_workers']}
                - 后端: {resources['recommended_backend']}
                """)
