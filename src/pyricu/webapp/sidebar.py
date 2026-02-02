"""侧边栏模块。

处理侧边栏渲染和交互逻辑。
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from pyricu.webapp.system_utils import get_system_resources


def _lazy_load_app_module():
    """延迟加载 app 模块以避免循环导入。"""
    from pyricu.webapp import app
    return app


def get_concept_groups():
    """从 app 模块获取概念分组。"""
    app_mod = _lazy_load_app_module()
    return app_mod.get_concept_groups()


def get_text(key: str) -> str:
    """从 app 模块获取文本。"""
    app_mod = _lazy_load_app_module()
    return app_mod.get_text(key)


def validate_database_path(data_path: str, database: str) -> dict:
    """从 app 模块验证数据库路径。"""
    app_mod = _lazy_load_app_module()
    return app_mod.validate_database_path(data_path, database)


def render_sidebar():
    """渲染侧边栏 - 根据entry_mode显示不同内容。"""
    # 使用双语特征分组
    concept_groups = get_concept_groups()
    
    # 所有可用的 concepts 列表（用于自定义选择）
    all_available_concepts = sorted(set(c for group_concepts in concept_groups.values() for c in group_concepts))
    
    # 获取当前模式
    entry_mode = st.session_state.get('entry_mode', 'none')
    
    with st.sidebar:
        # 🔙 返回入口页面按钮（始终显示，除非在入口页）
        if entry_mode != 'none':
            back_label = "🔙 Back to Mode Selection" if st.session_state.language == 'en' else "🔙 返回模式选择"
            if st.button(back_label, key="back_to_entry", use_container_width=True):
                st.session_state.entry_mode = 'none'
                # 清空所有数据
                st.session_state.loaded_concepts = {}
                st.session_state.patient_ids = []
                st.session_state.use_mock_data = False
                # 清理Cohort相关缓存
                for key in ['group_a_data', 'group_b_data', 'multidb_data', 'dash_demographics',
                            'multidb_is_demo', 'dash_is_demo', 'cohort_is_demo']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            st.markdown("---")
        
        # 显示当前模式标识
        if entry_mode == 'demo':
            mode_badge = "🎭 Demo Mode" if st.session_state.language == 'en' else "🎭 演示模式"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981, #059669); 
                        padding: 12px 16px; border-radius: 10px; color: white; margin-bottom: 15px; text-align: center;">
                <b style="font-size: 1.1rem;">{mode_badge}</b>
            </div>
            """, unsafe_allow_html=True)
        elif entry_mode == 'real':
            mode_badge = "📊 Real Data Mode" if st.session_state.language == 'en' else "📊 真实数据模式"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3b82f6, #2563eb); 
                        padding: 12px 16px; border-radius: 10px; color: white; margin-bottom: 15px; text-align: center;">
                <b style="font-size: 1.1rem;">{mode_badge}</b>
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
        
        # ============ 侧边栏仅用于数据提取导出模式 ============
        # 快速可视化功能已移至主页面的 "Quick Visualization" 标签页
        
        sidebar_title = "📤 Data Extraction" if st.session_state.language == 'en' else "📤 数据提取导出"
        st.markdown(f"### {sidebar_title}")
        
        # ============ 步骤1: 数据源选择 ============
        # 🆕 根据entry_mode决定显示内容，不再允许切换
        
        if entry_mode == 'demo':
            # ===== DEMO 模式：只显示模拟数据参数，不显示数据库选择 =====
            st.markdown(f"### 📊 {get_text('step1')}")
            demo_title = "✨ Demo Mode" if st.session_state.language == 'en' else "✨ 演示模式"
            demo_desc = "System generates simulated ICU data for exploration" if st.session_state.language == 'en' else "系统生成模拟ICU数据供体验"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981, #059669); 
                        padding: 12px 16px; border-radius: 10px; color: white; margin: 8px 0;">
                <b>{demo_title}</b><br>
                <small>{demo_desc}</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.database = 'mock'
            st.session_state.use_mock_data = True
            
            # 模拟数据参数
            n_patients_label = "Number of Patients" if st.session_state.language == 'en' else "患者数量"
            hours_label = "Data Duration (hours)" if st.session_state.language == 'en' else "数据时长(小时)"
            n_patients = st.slider(n_patients_label, 50, 500, st.session_state.mock_params.get('n_patients', 100))
            hours = st.slider(hours_label, 24, 168, st.session_state.mock_params.get('hours', 72))
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
            
            # 根据操作系统和数据库设置默认路径
            import platform
            if platform.system() == 'Windows':
                default_paths = {
                    'miiv': r'D:\mimic-iv-3.1',
                    'eicu': r'D:\eicu-crd-2.0',
                    'aumc': r'D:\amsterdamumcdb-1.0.2',
                    'hirid': r'D:\hirid-1.1.1',
                    'mimic': r'D:\mimic-iii-1.4',
                    'sic': r'D:\sicdb-1.0.6',
                }
            else:
                default_paths = {
                    'miiv': '/home/zhuhb/icudb/mimiciv/3.1',
                    'eicu': '/home/zhuhb/icudb/eicu/2.0.1',
                    'aumc': '/home/zhuhb/icudb/aumc/1.0.2',
                    'hirid': '/home/zhuhb/icudb/hirid/1.1.1',
                    'mimic': '/home/zhuhb/icudb/mimiciii/1.4',
                    'sic': '/home/zhuhb/icudb/sicdb/1.0.6',
                }
            default_path = default_paths.get(database, '')
            path_label = "Data Path" if st.session_state.language == 'en' else "数据路径"
            data_path = st.text_input(
                path_label,
                value=st.session_state.data_path or default_path,
                placeholder=f"/path/to/{database}",
                on_change=lambda: None  # 触发 rerun 以检测新数据库
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
            if st.button(validate_btn, width="stretch", key="validate_path"):
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
                        st.success(f"✅ {validation_result['message']}")
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
                convert_btn = "🔄 Convert to Parquet" if st.session_state.language == 'en' else "🔄 转换为Parquet"
                if st.button(convert_btn, width="stretch", type="primary", key="convert_csv"):
                    st.session_state.show_convert_dialog = True
                    st.session_state.convert_source_path = data_path
                    st.rerun()
                convert_hint = "💡 Converting to Parquet enables faster data loading" if st.session_state.language == 'en' else "💡 转换为Parquet格式可大幅加速数据加载"
                st.caption(convert_hint)
            elif data_path and Path(data_path).exists():
                validate_hint = "💡 Click the button above to validate data format" if st.session_state.language == 'en' else "💡 点击上方按钮验证数据格式"
                st.caption(validate_hint)
        
        st.markdown("---")
        
        # ============ 步骤2: 队列筛选（新增） ============
        step2_cohort_title = "Step 2: Cohort Selection" if st.session_state.language == 'en' else "步骤2: 队列筛选"
        st.markdown(f"### 👥 {step2_cohort_title}")
        
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
            }
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
            # 年龄筛选
            age_label = "🎂 Age Range" if st.session_state.language == 'en' else "🎂 年龄范围"
            with st.expander(age_label, expanded=True):
                age_col1, age_col2 = st.columns(2)
                with age_col1:
                    age_min_label = "Min Age" if st.session_state.language == 'en' else "最小年龄"
                    age_min = st.number_input(
                        age_min_label, min_value=0, max_value=120, 
                        value=18 if st.session_state.cohort_filter['age_min'] is None else int(st.session_state.cohort_filter['age_min']),
                        key="cohort_age_min"
                    )
                    if age_min > 0:
                        st.session_state.cohort_filter['age_min'] = age_min
                    else:
                        st.session_state.cohort_filter['age_min'] = None
                with age_col2:
                    age_max_label = "Max Age" if st.session_state.language == 'en' else "最大年龄"
                    age_max = st.number_input(
                        age_max_label, min_value=0, max_value=120, 
                        value=100 if st.session_state.cohort_filter['age_max'] is None else int(st.session_state.cohort_filter['age_max']),
                        key="cohort_age_max"
                    )
                    if age_max < 120:
                        st.session_state.cohort_filter['age_max'] = age_max
                    else:
                        st.session_state.cohort_filter['age_max'] = None
            
            # 首次入ICU筛选
            first_icu_label = "🏥 First ICU Stay Only" if st.session_state.language == 'en' else "🏥 仅首次入ICU"
            first_icu_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'yes': 'Yes (First ICU only)' if st.session_state.language == 'en' else '是（仅首次）',
                'no': 'No (Readmissions only)' if st.session_state.language == 'en' else '否（仅再入院）',
            }
            first_icu_val = st.radio(
                first_icu_label,
                options=list(first_icu_options.keys()),
                format_func=lambda x: first_icu_options[x],
                index=0,
                horizontal=True,
                key="cohort_first_icu"
            )
            if first_icu_val == 'yes':
                st.session_state.cohort_filter['first_icu_stay'] = True
            elif first_icu_val == 'no':
                st.session_state.cohort_filter['first_icu_stay'] = False
            else:
                st.session_state.cohort_filter['first_icu_stay'] = None
            
            # 住院时长筛选（只需要最短时长，默认24小时）
            los_label = "⏱️ Min ICU Stay (hours)" if st.session_state.language == 'en' else "⏱️ 最短住院时长（小时）"
            los_help = "Minimum ICU stay duration to include patients (default 24h)" if st.session_state.language == 'en' else "纳入患者的最短ICU住院时长（默认24小时）"
            los_min = st.number_input(
                los_label, min_value=0, max_value=10000, value=24,
                help=los_help,
                key="cohort_los_min"
            )
            st.session_state.cohort_filter['los_min'] = los_min if los_min > 0 else None
            st.session_state.cohort_filter['los_max'] = None  # 不再使用max
            
            # 性别筛选
            gender_label = "👤 Gender" if st.session_state.language == 'en' else "👤 性别"
            gender_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'M': 'Male' if st.session_state.language == 'en' else '男性',
                'F': 'Female' if st.session_state.language == 'en' else '女性',
            }
            gender_val = st.radio(
                gender_label,
                options=list(gender_options.keys()),
                format_func=lambda x: gender_options[x],
                index=0,
                horizontal=True,
                key="cohort_gender"
            )
            st.session_state.cohort_filter['gender'] = gender_val if gender_val != 'any' else None
            
            # 存活状态筛选
            survival_label = "💚 Survival Status" if st.session_state.language == 'en' else "💚 存活状态"
            survival_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'survived': 'Survived' if st.session_state.language == 'en' else '存活',
                'deceased': 'Deceased' if st.session_state.language == 'en' else '死亡',
            }
            survival_val = st.radio(
                survival_label,
                options=list(survival_options.keys()),
                format_func=lambda x: survival_options[x],
                index=0,
                horizontal=True,
                key="cohort_survival"
            )
            if survival_val == 'survived':
                st.session_state.cohort_filter['survived'] = True
            elif survival_val == 'deceased':
                st.session_state.cohort_filter['survived'] = False
            else:
                st.session_state.cohort_filter['survived'] = None
            
            # 🔧 移除 Sepsis 筛选器（太复杂，用户可能不理解）
            # 直接设置为 None（不筛选）
            st.session_state.cohort_filter['has_sepsis'] = None
            
            # 显示当前筛选条件摘要
            filter_summary = []
            cf = st.session_state.cohort_filter
            if cf['age_min'] is not None or cf['age_max'] is not None:
                age_range = f"{cf['age_min'] or 0}-{cf['age_max'] or '∞'}"
                filter_summary.append(f"Age: {age_range}" if st.session_state.language == 'en' else f"年龄: {age_range}")
            if cf['first_icu_stay'] is not None:
                filter_summary.append(f"First ICU: {'Yes' if cf['first_icu_stay'] else 'No'}" if st.session_state.language == 'en' else f"首次入ICU: {'是' if cf['first_icu_stay'] else '否'}")
            if cf['gender'] is not None:
                filter_summary.append(f"Gender: {cf['gender']}" if st.session_state.language == 'en' else f"性别: {'男' if cf['gender']=='M' else '女'}")
            if cf['survived'] is not None:
                filter_summary.append(f"Survived: {'Yes' if cf['survived'] else 'No'}" if st.session_state.language == 'en' else f"存活: {'是' if cf['survived'] else '否'}")
            if cf['has_sepsis'] is not None:
                filter_summary.append(f"Sepsis: {'Yes' if cf['has_sepsis'] else 'No'}" if st.session_state.language == 'en' else f"脓毒症: {'是' if cf['has_sepsis'] else '否'}")
            
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
                st.session_state.step2_confirmed = True
                step2_done_msg = "✅ Step 2 completed! Proceed to Step 3: Select Features" if st.session_state.language == 'en' else "✅ 步骤2已完成！请继续步骤3: 选择特征"
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
            if st.button(all_label, key="select_all_groups", width='stretch'):
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
        
        st.markdown("---")
        
        # ============ 步骤4: 直接导出 ============
        step4_title = "Step 4: Export Data" if st.session_state.language == 'en' else "步骤4: 导出数据"
        st.markdown(f"### 💾 {step4_title}")
        
        # 导出路径配置 - 实时根据数据库显示子目录，添加时间戳后缀
        import platform
        from datetime import datetime
        if platform.system() == 'Windows':
            base_export_path = r'D:\pyicu_export'
        else:
            base_export_path = os.path.expanduser('~/pyricu_export')
        db_name = st.session_state.get('database', 'mock')
        # 生成带时间戳的默认目录名（只保留年月日）
        timestamp_suffix = datetime.now().strftime('%Y%m%d')
        default_export_path = str(Path(base_export_path) / f"{db_name}_{timestamp_suffix}")
        
        export_path = st.text_input(
            "Export Path" if st.session_state.language == 'en' else "导出路径",
            value=default_export_path,
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
        limit_help = "Limit number of patients to speed up loading. 0 = no limit (full data, may be slow)" if st.session_state.language == 'en' else "限制加载的患者数量以加速。0 = 不限制（全量数据，可能较慢）"
        patient_limit_options = [100, 1000, 5000, 10000, 20000, 50000, 0]
        patient_limit_labels = {
            100: "100 (quick test)" if st.session_state.language == 'en' else "100（快速测试）",
            1000: "1,000",
            5000: "5,000", 
            10000: "10,000",
            20000: "20,000",
            50000: "50,000",
            0: "All patients (slower)" if st.session_state.language == 'en' else "全部患者（较慢）"
        }
        current_limit = st.session_state.get('patient_limit', 0)  # 默认全量
        if current_limit not in patient_limit_options:
            current_limit = 0  # 🔧 FIX: 默认全量加载
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
        can_export = (use_mock or (st.session_state.data_path and Path(st.session_state.data_path).exists())) and selected_concepts and export_path and Path(export_path).exists()
        
        export_btn = "📥 Export Data" if st.session_state.language == 'en' else "📥 导出数据"
        if can_export:
            if st.button(export_btn, type="primary", width="stretch"):
                st.session_state.trigger_export = True
                st.session_state.export_completed = False
                st.session_state['_exporting_in_progress'] = True  # 🆕 标记导出正在进行
                st.rerun()
        else:
            st.button(export_btn, type="primary", width="stretch", disabled=True)
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


