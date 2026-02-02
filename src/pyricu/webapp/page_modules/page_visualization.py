"""可视化页面模块。

包含快速可视化和传统可视化模式。
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List


def _lazy_load_app_module():
    """延迟加载 app 模块以避免循环导入。"""
    from pyricu.webapp import app
    return app


def get_text(key: str) -> str:
    """从 app 模块获取文本。"""
    app_mod = _lazy_load_app_module()
    return app_mod.get_text(key)


def render_quick_visualization_page():
    """渲染快速可视化主页面 - 包含数据加载区域和四个子模块。"""
    lang = st.session_state.get('language', 'en')
    entry_mode = st.session_state.get('entry_mode', 'none')
    
    # ============ 顶部：数据加载区域 ============
    st.markdown(f"### {get_text('quick_viz')}")
    
    # 根据 entry_mode 显示不同提示
    if entry_mode == 'demo':
        hint_text = "Generate demo data or load from exported files for interactive analysis" if lang == 'en' else "生成模拟数据或从已导出文件加载，进行交互式分析"
    else:
        hint_text = "Load data from exported files for interactive analysis" if lang == 'en' else "从已导出的数据文件加载，进行交互式分析"
    st.caption(hint_text)
    
    # 检查是否已加载数据
    data_loaded = len(st.session_state.loaded_concepts) > 0
    
    # 数据加载配置区域（折叠式，加载后默认收起）
    expander_label = "⚙️ Data Loading Settings" if lang == 'en' else "⚙️ 数据加载设置"
    with st.expander(expander_label, expanded=not data_loaded):
        
        # 真实数据模式：只能导入文件，不能使用 Demo
        if entry_mode == 'real':
            # 直接进入导出文件模式，不显示数据源选择
            st.session_state.viz_data_source = 1  # 强制设为文件模式
            
            info_msg = "📁 Load data from exported CSV/Parquet files" if lang == 'en' else "📁 从已导出的 CSV/Parquet 文件加载数据"
            st.info(info_msg)
        else:
            # 演示模式：可以选择 Demo 或 导出文件
            source_label = "Data Source" if lang == 'en' else "数据来源"
            st.markdown(f"**{source_label}**")
            
            # 初始化数据源选择 - 默认为导出文件模式（因为用户可能先用提取器导出过）
            if 'viz_data_source' not in st.session_state:
                st.session_state.viz_data_source = 1  # 默认导出文件
            
            # 使用两个按钮替代 radio，避免双击问题（导出文件优先显示）
            src_col1, src_col2 = st.columns(2)
            with src_col1:
                file_label = "📁 Exported Files" if lang == 'en' else "📁 已导出文件"
                file_type = "primary" if st.session_state.viz_data_source == 1 else "secondary"
                if st.button(file_label, key="viz_src_file", type=file_type, use_container_width=True):
                    st.session_state.viz_data_source = 1
                    st.rerun()
            with src_col2:
                demo_label = "🎭 Demo Data" if lang == 'en' else "🎭 模拟数据"
                demo_type = "primary" if st.session_state.viz_data_source == 0 else "secondary"
                if st.button(demo_label, key="viz_src_demo", type=demo_type, use_container_width=True):
                    st.session_state.viz_data_source = 0
                    st.rerun()
        
        # 🔧 根据数据源选择显示不同UI (导出文件模式优先检查)
        if st.session_state.viz_data_source == 1 or entry_mode == 'real':
            # ===== 导出文件模式 =====
            st.markdown("---")
            import platform
            
            # 🔧 默认路径：优先使用用户在数据提取器中保存的路径
            # 🔧 FIX: 使用 last_export_full_dir（包含cohort子目录）而非 last_export_dir
            if st.session_state.get('last_export_full_dir'):
                # 优先使用最后一次导出的完整目录（含cohort子目录）
                default_base_path = st.session_state['last_export_full_dir']
            elif st.session_state.get('last_export_dir'):
                # 其次使用导出根目录
                default_base_path = st.session_state['last_export_dir']
            elif st.session_state.get('export_path'):
                # 其次使用数据提取器中设置的导出路径
                default_base_path = st.session_state['export_path']
            elif platform.system() == 'Windows':
                default_base_path = r'D:\pyicu_export'
            else:
                default_base_path = os.path.expanduser('~/pyricu_export')
            
            # 🔧 数据库选择 - 根据入口模式提供不同选项
            db_select_label = "📊 Database" if lang == 'en' else "📊 数据库"
            
            # Real Data模式：只有6个真实数据库，无mock
            if entry_mode == 'real':
                db_options = ['(Auto Detect)', 'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic']
                db_labels = {
                    '(Auto Detect)': '(Auto Detect)' if lang == 'en' else '(自动检测)',
                    'miiv': 'MIMIC-IV 🟢',
                    'eicu': 'eICU 🟠',
                    'aumc': 'Amsterdam 🔵',
                    'hirid': 'HiRID 🔴',
                    'mimic': 'MIMIC-III 🟣',
                    'sic': 'SICdb ⚫',
                }
            else:
                # Demo模式：包含mock选项
                db_options = ['(Auto Detect)', 'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic', 'mock']
                db_labels = {
                    '(Auto Detect)': '(Auto Detect)' if lang == 'en' else '(自动检测)',
                    'miiv': 'MIMIC-IV 🟢',
                    'eicu': 'eICU 🟠',
                    'aumc': 'Amsterdam 🔵',
                    'hirid': 'HiRID 🔴',
                    'mimic': 'MIMIC-III 🟣',
                    'sic': 'SICdb ⚫',
                    'mock': '🎭 Mock/Demo',
                }
            
            # 路径输入行：路径输入 + 数据库选择
            path_col1, path_col2 = st.columns([3, 1])
            
            with path_col1:
                path_label = "Export Directory Path" if lang == 'en' else "导出数据目录路径"
                path_help = "Enter root export folder or specific database folder" if lang == 'en' else "输入导出根目录或具体数据库文件夹"
                
                # 🔧 FIX: 优先使用刚导出的路径，避免widget key冲突
                default_export_path = st.session_state.get('last_export_dir') or st.session_state.get('viz_export_path') or default_base_path
                
                # 🔧 FIX: 使用动态版本号key，确保导出后刷新显示
                path_version = st.session_state.get('_viz_export_path_version', 0)
                export_path = st.text_input(
                    path_label,
                    value=default_export_path,
                    help=path_help,
                    key=f"viz_export_path_input_v{path_version}"
                )
            st.session_state.viz_export_path = export_path
            
            with path_col2:
                selected_db = st.selectbox(
                    db_select_label,
                    options=db_options,
                    format_func=lambda x: db_labels.get(x, x),
                    key="viz_export_db_select",
                    help="Filter by database or auto-detect" if lang == 'en' else "按数据库筛选或自动检测"
                )
            
            # 🔧 智能目录搜索：根据路径和数据库选择，动态查找可用目录
            def find_export_directories(base_path: str, db_filter: str) -> list:
                """智能搜索导出数据目录"""
                result = []
                base = Path(base_path)
                
                if not base.exists():
                    return result
                
                # 如果指定了数据库，只搜索匹配的子目录
                if db_filter and db_filter != '(Auto Detect)':
                    # 直接检查 base_path 是否就是目标目录
                    if base.name == db_filter:
                        files = list(base.glob('*.csv')) + list(base.glob('*.parquet'))
                        if files:
                            result.append(('', len(files)))  # 当前目录
                    # 检查子目录
                    for subdir in base.iterdir():
                        if subdir.is_dir() and (subdir.name == db_filter or subdir.name.startswith(f'{db_filter}_')):
                            files = list(subdir.glob('*.csv')) + list(subdir.glob('*.parquet'))
                            if files:
                                result.append((subdir.name, len(files)))
                else:
                    # 自动检测：扫描所有子目录
                    # 先检查当前目录
                    files = list(base.glob('*.csv')) + list(base.glob('*.parquet'))
                    if files:
                        result.append(('(Current Dir)' if lang == 'en' else '(当前目录)', len(files)))
                    
                    # 检查子目录
                    for subdir in sorted(base.iterdir()):
                        if subdir.is_dir():
                            files = list(subdir.glob('*.csv')) + list(subdir.glob('*.parquet'))
                            if files:
                                result.append((subdir.name, len(files)))
                
                return result
            
            # 查找可用目录
            available_dirs = find_export_directories(export_path, selected_db)
            
            # 如果找到多个目录，提供选择
            if len(available_dirs) > 1:
                dir_options = [d[0] for d in available_dirs]
                dir_labels = {d[0]: f"{d[0]} ({d[1]} files)" for d in available_dirs}
                
                selected_subdir = st.selectbox(
                    "📂 " + ("Select Export Folder" if lang == 'en' else "选择导出文件夹"),
                    options=dir_options,
                    format_func=lambda x: dir_labels.get(x, x),
                    key="viz_export_subdir"
                )
                
                # 更新实际路径
                if selected_subdir and selected_subdir not in ['(Current Dir)', '(当前目录)']:
                    actual_path = os.path.join(export_path, selected_subdir)
                else:
                    actual_path = export_path
            elif len(available_dirs) == 1:
                # 只有一个目录，直接使用
                if available_dirs[0][0] not in ['(Current Dir)', '(当前目录)']:
                    actual_path = os.path.join(export_path, available_dirs[0][0])
                else:
                    actual_path = export_path
                st.success(f"✅ " + (f"Found export folder: {available_dirs[0][0]} ({available_dirs[0][1]} files)" if lang == 'en' else f"找到导出文件夹：{available_dirs[0][0]}（{available_dirs[0][1]}个文件）"))
            else:
                actual_path = export_path
            
            # 检查路径并显示可用文件
            if actual_path and Path(actual_path).exists():
                available_files = list(Path(actual_path).glob('*.csv')) + \
                                  list(Path(actual_path).glob('*.parquet')) + \
                                  list(Path(actual_path).glob('*.xlsx'))
                
                if available_files:
                    file_names = [f.stem for f in available_files]
                    found_msg = f"✅ Found {len(available_files)} data files" if lang == 'en' else f"✅ 发现 {len(available_files)} 个数据文件"
                    st.success(found_msg)
                    
                    # 文件选择
                    select_label = "Select Tables to Load" if lang == 'en' else "选择要加载的表格"
                    
                    # 🔧 FIX: 使用带版本号的 key 来强制刷新 multiselect
                    # 每次点击 All/Clear 按钮，版本号递增，multiselect 会重新创建
                    if '_viz_select_version_v2' not in st.session_state:
                        st.session_state._viz_select_version_v2 = 0
                    
                    # 🔧 保存当前文件列表到 session_state，让回调能访问
                    st.session_state._current_filenames_v2 = file_names.copy()
                    
                    # 初始化默认选中 - 默认全选
                    ms_key = f"viz_file_multiselect_v{st.session_state._viz_select_version_v2}"
                    if ms_key not in st.session_state:
                        # 新版本的 key，需要初始化默认值
                        default_selection = file_names.copy()  # 默认全选
                    else:
                        # 已存在的 key，过滤掉无效文件
                        existing = st.session_state.get(ms_key, [])
                        default_selection = [f for f in existing if f in file_names] or file_names.copy()
                    
                    # 🔧 FIX: 回调函数 - 全选
                    def select_all_v2():
                        version = st.session_state._viz_select_version_v2 + 1
                        st.session_state._viz_select_version_v2 = version
                        # 设置下一个版本的 multiselect key 的默认值
                        new_key = f"viz_file_multiselect_v{version}"
                        st.session_state[new_key] = st.session_state._current_filenames_v2.copy()
                    
                    # 🔧 FIX: 回调函数 - 清空
                    def clear_all_v2():
                        version = st.session_state._viz_select_version_v2 + 1
                        st.session_state._viz_select_version_v2 = version
                        new_key = f"viz_file_multiselect_v{version}"
                        st.session_state[new_key] = []
                    
                    col_all, col_clear = st.columns(2)
                    with col_all:
                        all_label = "✅ ALL" if lang == 'en' else "✅ 全选"
                        st.button(all_label, key="viz_select_all_v2", use_container_width=True, 
                                 on_click=select_all_v2, type="primary")
                    with col_clear:
                        clear_label = "❌ Clear" if lang == 'en' else "❌ 清空"
                        st.button(clear_label, key="viz_clear_all_v2", use_container_width=True,
                                 on_click=clear_all_v2)
                    
                    selected_files = st.multiselect(
                        select_label,
                        options=file_names,
                        default=default_selection,
                        key=ms_key
                    )
                    
                    # 患者数量限制
                    patient_limit_label = "Max Patients to Load" if lang == 'en' else "最大加载患者数"
                    patient_options = [50, 100, 200, 500, -1]
                    option_labels = {
                        50: "50 (Fast)" if lang == 'en' else "50 (快速)",
                        100: "100 (Recommended)" if lang == 'en' else "100 (推荐)",
                        200: "200" if lang == 'en' else "200",
                        500: "500 (Slow)" if lang == 'en' else "500 (较慢)",
                        -1: "All (May Lag)" if lang == 'en' else "全部 (可能卡顿)"
                    }
                    max_patients_opt = st.selectbox(
                        patient_limit_label,
                        options=patient_options,
                        index=1,
                        format_func=lambda x: option_labels[x],
                        key="viz_max_patients"
                    )
                    max_patients = None if max_patients_opt == -1 else max_patients_opt
                    
                    # 加载按钮
                    load_btn_label = "🔍 Load Data" if lang == 'en' else "🔍 加载数据"
                    if selected_files:
                        if st.button(load_btn_label, type="primary", use_container_width=True, key="viz_load_files"):
                            loading_msg = "Loading data..." if lang == 'en' else "正在加载数据..."
                            with st.spinner(loading_msg):
                                load_from_exported(actual_path, selected_files=selected_files, max_patients=max_patients)
                            st.rerun()
                    else:
                        st.button(load_btn_label, type="primary", use_container_width=True, disabled=True, key="viz_load_disabled")
                        warn_msg = "⚠️ Please select at least one file" if lang == 'en' else "⚠️ 请至少选择一个文件"
                        st.warning(warn_msg)
                else:
                    warn_msg = "⚠️ No data files found in this directory (CSV/Parquet/Excel)" if lang == 'en' else "⚠️ 该目录下未找到数据文件 (CSV/Parquet/Excel)"
                    st.warning(warn_msg)
            elif export_path:
                err_msg = "❌ Directory does not exist" if lang == 'en' else "❌ 目录不存在"
                st.error(err_msg)
        
        # ===== Demo 模式 (当选择模拟数据且非real模式) =====
        elif st.session_state.viz_data_source == 0 and entry_mode != 'real':
            st.markdown("---")
            demo_info = "Generate ALL simulated ICU features for full exploration" if lang == 'en' else "生成全部模拟ICU特征供完整体验"
            st.info(f"✨ {demo_info}")
            
            col1, col2 = st.columns(2)
            with col1:
                n_patients_label = "Number of Patients" if lang == 'en' else "患者数量"
                n_patients = st.slider(n_patients_label, 10, 200, 50, key="viz_demo_patients")
            with col2:
                hours_label = "Data Duration (hours)" if lang == 'en' else "数据时长(小时)"
                hours = st.slider(hours_label, 24, 168, 72, key="viz_demo_hours")
            
            # 显示将生成的特征数量提示
            feature_hint = "Will generate ~160+ features across all modules (Vitals, Labs, SOFA, Sepsis, AKI, etc.)" if lang == 'en' else "将生成约160+个特征，覆盖所有模块（生命体征、实验室、SOFA、脓毒症、AKI等）"
            st.caption(f"💡 {feature_hint}")
            
            load_btn_label = "🚀 Generate & Load All Demo Data" if lang == 'en' else "🚀 生成并加载全部模拟数据"
            if st.button(load_btn_label, type="primary", use_container_width=True, key="viz_load_demo"):
                loading_msg = "Generating all mock data (~160+ features)..." if lang == 'en' else "正在生成全部模拟数据（约160+特征）..."
                with st.spinner(loading_msg):
                    # 🔧 使用 get_mock_params_with_cohort 获取完整参数（包含 cohort_filter）
                    params = get_mock_params_with_cohort()
                    params['n_patients'] = n_patients  # 使用当前 slider 的值
                    params['hours'] = hours
                    mock_data, patient_ids = generate_mock_data(**params)
                    st.session_state.loaded_concepts = mock_data
                    st.session_state.patient_ids = patient_ids
                    st.session_state.id_col = 'stay_id'
                    st.session_state.time_col = 'time'
                st.rerun()
    
    # 显示已加载数据状态
    if data_loaded:
        st.markdown("---")
        status_cols = st.columns(3)
        with status_cols[0]:
            feat_count = len(st.session_state.loaded_concepts)
            feat_label = "Features" if lang == 'en' else "特征"
            st.metric(feat_label, feat_count)
        with status_cols[1]:
            pat_count = len(st.session_state.patient_ids) if st.session_state.patient_ids else 0
            pat_label = "Patients" if lang == 'en' else "患者"
            st.metric(pat_label, pat_count)
        with status_cols[2]:
            status_label = "Status" if lang == 'en' else "状态"
            st.metric(status_label, "✅ Ready" if lang == 'en' else "✅ 就绪")
        
        st.markdown("---")
        
        # ============ 下方：四个子模块 Tabs ============
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            get_text('sub_data_table'),
            get_text('sub_timeseries'),
            get_text('sub_patient_view'),
            get_text('sub_data_quality'),
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
        # 未加载数据时显示提示
        st.markdown("---")
        no_data_msg = """
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 16px; margin: 20px 0;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
            <h3 style="color: #495057; margin-bottom: 10px;">""" + ("No Data Loaded" if lang == 'en' else "尚未加载数据") + """</h3>
            <p style="color: #6c757d;">""" + ("Please configure data source above and click Load button" if lang == 'en' else "请在上方配置数据来源，然后点击加载按钮") + """</p>
        </div>
        """
        st.markdown(no_data_msg, unsafe_allow_html=True)


def render_visualization_mode_legacy():
    
    # 数据目录选择 - 支持选择已导出的文件夹
    import platform
    
    # 允许用户自定义基础搜索路径
    if 'viz_base_path' not in st.session_state:
        if platform.system() == 'Windows':
            st.session_state.viz_base_path = r'D:\pyicu_export'
        else:
            st.session_state.viz_base_path = os.path.expanduser('~/pyricu_export')
    
    # 基础路径配置
    base_path_label = "Base search directory" if st.session_state.language == 'en' else "基础搜索目录"
    base_path_help = "Directory containing exported data folders" if st.session_state.language == 'en' else "包含已导出数据文件夹的目录"
    
    with st.expander("⚙️ " + ("Path Settings" if st.session_state.language == 'en' else "路径设置"), expanded=True):
        new_base_path = st.text_input(
            base_path_label,
            value=st.session_state.viz_base_path,
            key="viz_base_path_input",
            help=base_path_help
        )
        
        col_update, col_reset = st.columns(2)
        with col_update:
            update_btn = "🔄 Update & Scan" if st.session_state.language == 'en' else "🔄 更新并扫描"
            if st.button(update_btn, width='stretch'):
                st.session_state.viz_base_path = new_base_path
                st.rerun()
        
        with col_reset:
            reset_btn = "↩️ Reset Default" if st.session_state.language == 'en' else "↩️ 重置默认"
            if st.button(reset_btn, width='stretch'):
                if platform.system() == 'Windows':
                    st.session_state.viz_base_path = r'D:\pyicu_export'
                else:
                    st.session_state.viz_base_path = os.path.expanduser('~/pyricu_export')
                st.rerun()
    
    base_export_path = st.session_state.viz_base_path
    
    # 扫描已有的导出文件夹
    available_folders = []
    if Path(base_export_path).exists():
        available_folders = sorted(
            [d.name for d in Path(base_export_path).iterdir() if d.is_dir()],
            reverse=True  # 最新的在前
        )
    else:
        path_not_exist_msg = f"⚠️ Base path does not exist: {base_export_path}" if st.session_state.language == 'en' else f"⚠️ 基础路径不存在: {base_export_path}"
        st.warning(path_not_exist_msg)
    
    # 文件夹筛选器
    selected_folder_path = None  # 🔧 在外部初始化，确保作用域正确
    
    # 初始化已确认的路径（存储在session_state中）
    if 'viz_confirmed_path' not in st.session_state:
        st.session_state.viz_confirmed_path = None
    
    if available_folders:
        filter_label = "Filter by database" if st.session_state.language == 'en' else "按数据库筛选"
        db_prefixes = ['miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic', 'mock', 'all']
        db_options = ['All'] + [p for p in db_prefixes if any(f.startswith(p) for f in available_folders)]
        db_filter = st.selectbox(
            filter_label,
            options=db_options,
            index=0,
            key="viz_db_filter"
        )
        
        # 过滤文件夹列表
        if db_filter != 'All':
            filtered_folders = [f for f in available_folders if f.startswith(db_filter)]
        else:
            filtered_folders = available_folders
        
        # 文件夹选择器
        if filtered_folders:
            folder_label = "Select exported folder" if st.session_state.language == 'en' else "选择导出文件夹"
            selected_folder = st.selectbox(
                folder_label,
                options=filtered_folders,
                index=0,
                key="viz_folder_select",
                help="Folders are sorted by timestamp (newest first)" if st.session_state.language == 'en' else "文件夹按时间戳排序（最新在前）"
            )
            
            # 🔧 构建完整路径
            if selected_folder:
                selected_folder_path = str(Path(base_export_path) / selected_folder)
                # 显示当前选择的路径
                current_path_msg = f"📂 Selected: `{selected_folder_path}`" if st.session_state.language == 'en' else f"📂 已选择: `{selected_folder_path}`"
                st.info(current_path_msg)
                
                # 🔧 添加确认按钮
                confirm_label = "✅ Confirm and Use This Folder" if st.session_state.language == 'en' else "✅ 确认使用此文件夹"
                if st.button(confirm_label, key="confirm_filter_path", type="primary", width='stretch'):
                    st.session_state.viz_confirmed_path = selected_folder_path
                    st.rerun()
        else:
            no_folder_msg = "No folders match the filter" if st.session_state.language == 'en' else "没有符合筛选条件的文件夹"
            st.info(no_folder_msg)
    
    # 🔧 确定最终使用的 data_dir
    if st.session_state.viz_confirmed_path:
        # 使用已确认的路径
        data_dir = st.session_state.viz_confirmed_path
        manual_expanded = False
    elif st.session_state.get('last_export_dir'):
        data_dir = st.session_state.get('last_export_dir')
        manual_expanded = True
    else:
        data_dir = st.session_state.get('export_path', str(Path(base_export_path) / 'miiv'))
        manual_expanded = True
    
    # 仍然提供手动输入选项
    manual_label = "Or enter path manually" if st.session_state.language == 'en' else "或手动输入路径"
    with st.expander(manual_label, expanded=False):
        manual_note = "💡 Use this to specify a custom path" if st.session_state.language == 'en' else "💡 使用此选项指定自定义路径"
        st.caption(manual_note)
        
        manual_data_dir = st.text_input(
            get_text('data_dir'),
            value="" if not manual_expanded else data_dir,  # 🔧 Filter模式时清空，避免混淆
            placeholder="Enter full path to exported data directory" if st.session_state.language == 'en' else "输入导出数据目录的完整路径",
            key="viz_data_dir_manual",
            help="Directory containing exported CSV/Parquet/Excel files" if st.session_state.language == 'en' else "包含已导出的 CSV/Parquet/Excel 文件的目录"
        )
        
        # 🔧 添加手动路径确认按钮
        if manual_data_dir and manual_data_dir.strip():
            manual_confirm_label = "✅ Confirm and Use Manual Path" if st.session_state.language == 'en' else "✅ 确认使用手动路径"
            if st.button(manual_confirm_label, key="confirm_manual_path", type="primary", width='stretch'):
                st.session_state.viz_confirmed_path = manual_data_dir.strip()
                st.rerun()
    
    # 🔧 显示最终确认的路径
    if st.session_state.viz_confirmed_path:
        final_path_msg = f"🎯 Active path: `{st.session_state.viz_confirmed_path}`" if st.session_state.language == 'en' else f"🎯 当前激活路径: `{st.session_state.viz_confirmed_path}`"
        st.success(final_path_msg)
        data_dir = st.session_state.viz_confirmed_path
    else:
        hint_msg = "⚠️ Please select a folder and click Confirm button" if st.session_state.language == 'en' else "⚠️ 请选择文件夹并点击确认按钮"
        st.warning(hint_msg)
        data_dir = None  # 未确认时不设置路径
    
    # 添加路径检查按钮
    check_btn = "🔍 Check Path" if st.session_state.language == 'en' else "🔍 检查路径"
    if st.button(check_btn, key="check_viz_path", width="stretch"):
        if data_dir:
            if Path(data_dir).exists():
                files = list(Path(data_dir).glob('*.csv')) + list(Path(data_dir).glob('*.parquet')) + list(Path(data_dir).glob('*.xlsx'))
                if files:
                    ok_msg = f"✅ Path valid! Found {len(files)} data files" if st.session_state.language == 'en' else f"✅ 路径有效！发现 {len(files)} 个数据文件"
                    st.success(ok_msg)
                else:
                    warn_msg = "⚠️ Directory exists but no data files found" if st.session_state.language == 'en' else "⚠️ 目录存在但未找到数据文件"
                    st.warning(warn_msg)
            else:
                err_msg = "❌ Path does not exist" if st.session_state.language == 'en' else "❌ 路径不存在"
                st.error(err_msg)
        else:
            warn_msg = "⚠️ Please enter a path first" if st.session_state.language == 'en' else "⚠️ 请先输入路径"
            st.warning(warn_msg)
    
    if data_dir and Path(data_dir).exists():
        # 扫描可用文件
        available_files = list(Path(data_dir).glob('*.csv')) + \
                          list(Path(data_dir).glob('*.parquet')) + \
                          list(Path(data_dir).glob('*.xlsx'))
        
        if available_files:
            file_names = [f.stem for f in available_files]
            found_msg = f"✅ Found {len(available_files)} data files" if st.session_state.language == 'en' else f"✅ 发现 {len(available_files)} 个数据文件"
            st.success(found_msg)
            
            # 让用户选择要加载的表格
            select_label = "Select Tables to Load" if st.session_state.language == 'en' else "选择要加载的表格"
            select_help = "Select tables to load for visualization (max 3 recommended)" if st.session_state.language == 'en' else "选择要加载到可视化的表格（建议不超过3个以保证流畅性）"
            
            # 🔧 FIX: 使用带版本号的 key 来强制刷新 multiselect（与 Export Directory 模式统一）
            if '_viz_select_version_filter' not in st.session_state:
                st.session_state._viz_select_version_filter = 0
            
            # 保存当前文件列表到 session_state
            st.session_state._current_filenames_filter = file_names.copy()
            
            # 确定 multiselect 的 key 和默认值
            ms_key_filter = f"viz_files_select_filter_v{st.session_state._viz_select_version_filter}"
            if ms_key_filter not in st.session_state:
                default_selection_filter = file_names.copy()  # 默认全选
            else:
                existing = st.session_state.get(ms_key_filter, [])
                default_selection_filter = [f for f in existing if f in file_names] or file_names.copy()
            
            # 🔧 FIX: 回调函数 - 全选
            def select_all_filter():
                version = st.session_state._viz_select_version_filter + 1
                st.session_state._viz_select_version_filter = version
                new_key = f"viz_files_select_filter_v{version}"
                st.session_state[new_key] = st.session_state._current_filenames_filter.copy()
            
            # 🔧 FIX: 回调函数 - 清空
            def clear_all_filter():
                version = st.session_state._viz_select_version_filter + 1
                st.session_state._viz_select_version_filter = version
                new_key = f"viz_files_select_filter_v{version}"
                st.session_state[new_key] = []
            
            # 添加 ALL / Clear 按钮
            col_all, col_clear = st.columns(2)
            with col_all:
                all_label = "✅ ALL" if st.session_state.language == 'en' else "✅ 全选"
                st.button(all_label, key="select_all_tables_filter", use_container_width=True, 
                         on_click=select_all_filter, type="primary")
            with col_clear:
                clear_label = "❌ Clear" if st.session_state.language == 'en' else "❌ 清空"
                st.button(clear_label, key="clear_all_tables_filter", use_container_width=True,
                         on_click=clear_all_filter)
            
            selected_files = st.multiselect(
                select_label,
                options=file_names,
                default=default_selection_filter,
                help=select_help,
                key=ms_key_filter,
            )
            
            if selected_files:
                selected_msg = f"{len(selected_files)} tables selected" if st.session_state.language == 'en' else f"已选 {len(selected_files)} 个表格"
                st.caption(selected_msg)
                
                # 患者数量选择器
                st.markdown("---")
                patient_limit_label = "Patients to Load" if st.session_state.language == 'en' else "加载患者数量"
                
                # 使用 selectbox 代替 slider，提供预设选项和"全部"选项
                patient_options = [50, 100, 200, 500, -1]  # -1 表示全部
                option_labels = {
                    50: "50 (Fast)" if st.session_state.language == 'en' else "50 (快速)",
                    100: "100 (Recommended)" if st.session_state.language == 'en' else "100 (推荐)",
                    200: "200 (Slow)" if st.session_state.language == 'en' else "200 (较慢)",
                    500: "500 (Very Slow)" if st.session_state.language == 'en' else "500 (很慢)",
                    -1: "🔓 All (May Lag!)" if st.session_state.language == 'en' else "🔓 全部 (可能卡顿！)"
                }
                
                selected_option = st.selectbox(
                    patient_limit_label,
                    options=patient_options,
                    index=1,  # 默认选择100
                    format_func=lambda x: option_labels[x],
                    key="viz_max_patients"
                )
                
                # 根据选择显示警告
                if selected_option == -1:
                    all_warn = "⚠️ Loading ALL patients may cause UI lag or crash for large datasets!" if st.session_state.language == 'en' else "⚠️ 加载全部患者可能导致界面卡顿甚至崩溃！大数据集请谨慎使用"
                    st.warning(all_warn)
                    max_patients = None  # None 表示不限制
                elif selected_option >= 200:
                    perf_warn = "⚠️ High patient count may cause slow performance" if st.session_state.language == 'en' else "⚠️ 患者数较多，性能可能下降"
                    st.warning(perf_warn)
                    max_patients = selected_option
                else:
                    max_patients = selected_option
                
                st.markdown("---")
                
                # 显示加载状态
                is_loaded = len(st.session_state.loaded_concepts) > 0
                if is_loaded:
                    loaded_msg = f"📊 {len(st.session_state.loaded_concepts)} features, {len(st.session_state.patient_ids)} patients loaded" if st.session_state.language == 'en' else f"📊 已加载 {len(st.session_state.loaded_concepts)} 个特征，{len(st.session_state.patient_ids)} 个患者"
                    st.info(loaded_msg)
                
                if st.button(get_text('load_data'), type="primary", width="stretch"):
                    loading_msg = "Loading data..." if st.session_state.language == 'en' else "正在加载数据..."
                    with st.spinner(loading_msg):
                        load_from_exported(data_dir, selected_files=selected_files, max_patients=max_patients)
                    st.rerun()
            else:
                st.button(get_text('load_data'), type="primary", width="stretch", disabled=True)
                warn_msg = "⚠️ Please select at least one table" if st.session_state.language == 'en' else "⚠️ 请选择至少一个表格"
                st.caption(warn_msg)
            
            # 显示文件预览
            with st.expander(get_text('file_list'), expanded=False):
                for f in available_files[:10]:
                    st.caption(f"• {f.name}")
                if len(available_files) > 10:
                    more_msg = f"... {len(available_files)} files total" if st.session_state.language == 'en' else f"... 共 {len(available_files)} 个文件"
                    st.caption(more_msg)
        else:
            st.warning(get_text('no_files'))
            format_msg = "Supported formats: CSV, Parquet, Excel" if st.session_state.language == 'en' else "支持格式：CSV、Parquet、Excel"
            st.caption(format_msg)
    elif data_dir:
        st.error(get_text('dir_not_exist'))
        check_msg = "Please check if the path is correct" if st.session_state.language == 'en' else "请检查路径是否正确"
        st.caption(check_msg)
    
    st.markdown("---")
    
    # 显示已加载数据的状态
    if len(st.session_state.loaded_concepts) > 0:
        st.markdown(f"### {get_text('loaded_data')}")
        feat_msg = f"✅ {len(st.session_state.loaded_concepts)} features" if st.session_state.language == 'en' else f"✅ {len(st.session_state.loaded_concepts)} 个特征"
        pat_msg = f"✅ {len(st.session_state.patient_ids)} patients" if st.session_state.language == 'en' else f"✅ {len(st.session_state.patient_ids)} 个患者"
        st.success(feat_msg)
        st.success(pat_msg)
        
        with st.expander(get_text('view_features'), expanded=False):
            for concept in sorted(st.session_state.loaded_concepts.keys()):
                st.caption(f"• {concept}")
    else:
        st.info(get_text('load_hint'))


