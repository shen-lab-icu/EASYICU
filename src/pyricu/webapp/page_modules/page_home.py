"""首页模块。

包含首页渲染和不同模式的处理逻辑。
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


def get_concept_groups():
    """从 app 模块获取概念分组。"""
    app_mod = _lazy_load_app_module()
    return app_mod.get_concept_groups()


def render_home():
    """渲染首页 - 引导式教程，根据用户进度动态显示。"""
    lang = st.session_state.language
    
    # 如果已加载数据，直接显示数据概览
    if len(st.session_state.loaded_concepts) > 0:
        render_data_overview()
        return
    
    # 标题已经在main()中渲染，这里不再重复
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 获取当前模式 - 使用app_mode（'extract'或'viz'）
    current_mode = st.session_state.get('app_mode', 'extract')
    is_viz_mode = current_mode == 'viz'
    
    if is_viz_mode:
        # ============ 快速可视化模式教程 ============
        render_home_viz_mode(lang)
    else:
        # ============ 数据提取导出模式教程 ============
        render_home_extract_mode(lang)


def render_home_viz_mode(lang):
    """渲染快速可视化模式的首页教程。"""
    # 进度指示器
    col1, col2 = st.columns(2)
    
    # 检查状态
    viz_dir = st.session_state.get('viz_data_dir', '')
    has_files = False
    if viz_dir and Path(viz_dir).exists():
        files = list(Path(viz_dir).glob('*.csv')) + list(Path(viz_dir).glob('*.parquet')) + list(Path(viz_dir).glob('*.xlsx'))
        has_files = len(files) > 0
    
    step1_done = has_files
    step2_done = len(st.session_state.loaded_concepts) > 0
    
    done_text = "✅ Done" if lang == 'en' else "✅ 完成"
    in_progress_text = "🔵 In Progress" if lang == 'en' else "🔵 进行中"
    waiting_text = "⏳ Waiting" if lang == 'en' else "⏳ 等待"
    
    with col1:
        status = done_text if step1_done else in_progress_text
        color = "#28a745" if step1_done else "#ffc107"
        step_label = "Step 1" if lang == 'en' else "步骤 1"
        step_desc = "Select Data Directory" if lang == 'en' else "选择数据目录"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if step1_done:
            status = done_text if step2_done else in_progress_text
            color = "#28a745" if step2_done else "#ffc107"
        else:
            status = waiting_text
            color = "#6c757d"
        step_label = "Step 2" if lang == 'en' else "步骤 2"
        step_desc = "Load & Visualize" if lang == 'en' else "加载并可视化"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 教程内容
    if not step1_done:
        task_header = "📍 Current Task: Select Data Directory" if lang == 'en' else "📍 当前任务：选择数据目录"
        st.markdown(f"## {task_header}")
        
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 Please specify the data directory in the left sidebar</h4>
                <p style="color:#333; margin-bottom:12px">
                    Quick Visualization mode loads data from previously exported files:
                </p>
                <ul style="color:#444; font-size:0.9rem;">
                    <li>Enter the path to the directory containing exported data files</li>
                    <li>Supported formats: <b>CSV, Parquet, Excel</b></li>
                    <li>If you haven't exported data yet, switch to "Data Extraction" mode first</li>
                </ul>
                <p style="color:#b45309; margin-top:12px;">
                    <b>💡 Tip:</b> Default path is <code>~/pyricu_export/miiv</code>
                </p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 请在左侧边栏指定数据目录</h4>
                <p style="color:#333; margin-bottom:12px">
                    快速可视化模式从已导出的文件加载数据：
                </p>
                <ul style="color:#444; font-size:0.9rem;">
                    <li>输入包含已导出数据文件的目录路径</li>
                    <li>支持的格式：<b>CSV、Parquet、Excel</b></li>
                    <li>如果您还没有导出过数据，请先切换到「数据提取导出」模式</li>
                </ul>
                <p style="color:#b45309; margin-top:12px;">
                    <b>💡 提示：</b> 默认路径是 <code>~/pyricu_export/miiv</code>
                </p>
            </div>
            ''', unsafe_allow_html=True)
    else:
        task_header = "📍 Current Task: Load Data" if lang == 'en' else "📍 当前任务：加载数据"
        st.markdown(f"## {task_header}")
        
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 Click "Load Data" in the left sidebar</h4>
                <p style="color:#333; margin-bottom:12px">
                    Data files found! You can now:
                </p>
                <ul style="color:#444; font-size:0.9rem;">
                    <li>Select specific tables to load (recommended ≤ 3 for best performance)</li>
                    <li>Click <b>"Load Data"</b> button to load into memory</li>
                    <li>After loading, use the tabs above to explore and visualize</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 在左侧边栏点击「加载数据」</h4>
                <p style="color:#333; margin-bottom:12px">
                    已发现数据文件！您现在可以：
                </p>
                <ul style="color:#444; font-size:0.9rem;">
                    <li>选择要加载的表格（建议不超过3个以保证流畅性）</li>
                    <li>点击 <b>「加载数据」</b> 按钮将数据加载到内存</li>
                    <li>加载完成后，使用上方的标签页进行探索和可视化</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
    
    # 功能预览
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    preview_title = "🎯 What You Can Do After Loading" if lang == 'en' else "🎯 加载后可用功能"
    st.markdown(f"### {preview_title}")
    
    if lang == 'en':
        features = [
            ("�", "Data Tables", "Browse and merge features"),
            ("📈", "Time Series", "Interactive visualization"),
            ("🏥", "Patient View", "Patient dashboard"),
            ("📊", "Data Quality", "Missing rate analysis"),
        ]
    else:
        features = [
            ("📋", "数据大表", "浏览与合并特征"),
            ("📈", "时序分析", "交互式可视化"),
            ("🏥", "患者视图", "患者仪表盘"),
            ("📊", "数据质量", "缺失率分析"),
        ]
    
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f'''
            <div class="feature-card" style="text-align:center;min-height:160px;display:flex;flex-direction:column;justify-content:center;padding:20px">
                <div style="font-size:2.5rem">{icon}</div>
                <div style="font-weight:600;color:#4fc3f7;margin:10px 0 6px 0;font-size:1.1rem">{title}</div>
                <div style="font-size:0.95rem;color:#333;line-height:1.5">{desc}</div>
            </div>
            ''', unsafe_allow_html=True)


def render_home_extract_mode(lang):
    """渲染数据提取导出模式的首页教程。"""
    
    # 计算当前步骤完成状态（4个步骤）
    # Step 1: Demo模式需要点击Confirm按钮，Real Data模式需要有效路径
    if st.session_state.get('use_mock_data', False):
        step1_done = st.session_state.get('step1_confirmed', False)
    else:
        step1_done = st.session_state.data_path and Path(st.session_state.data_path).exists()
    step2_done = st.session_state.get('step2_confirmed', False)
    step3_done = len(st.session_state.get('selected_concepts', [])) > 0
    # Step 4 只在真正导出完成后才算完成
    step4_done = st.session_state.get('export_completed', False)
    
    # ============ 进度指示器 ============
    # 添加锚点和大标题
    st.markdown('<div id="progress"></div>', unsafe_allow_html=True)
    progress_title = "📋 Progress" if lang == 'en' else "📋 进度"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.8rem;">{progress_title}</h2>', unsafe_allow_html=True)
    
    # 🆕 添加说明文字
    if lang == 'en':
        progress_desc = """
        <div style="font-size: 1.15rem; color: #333; margin-bottom: 20px; line-height: 1.6;">
            👈 <b>Simply click through the left sidebar</b> to complete the 4 steps below. 
            You'll easily define your ICU cohort, select features, and extract data!
        </div>
        """
    else:
        progress_desc = """
        <div style="font-size: 1.15rem; color: #333; margin-bottom: 20px; line-height: 1.6;">
            👈 <b>只需通过左侧边栏点击</b>，完成下面的4个步骤，
            即可轻松完成ICU数据的队列定义、特征选择和数据提取！
        </div>
        """
    st.markdown(progress_desc, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 状态文本
    done_text = "✅ Done" if lang == 'en' else "✅ 完成"
    in_progress_text = "🔵 In Progress" if lang == 'en' else "🔵 进行中"
    waiting_text = "⏳ Waiting" if lang == 'en' else "⏳ 等待"
    
    with col1:
        status = done_text if step1_done else in_progress_text
        color = "#28a745" if step1_done else "#ffc107"
        step_label = "Step 1" if lang == 'en' else "步骤 1"
        step_desc = "Data Source" if lang == 'en' else "配置数据源"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if step1_done:
            status = done_text if step2_done else in_progress_text
            color = "#28a745" if step2_done else "#ffc107"
        else:
            status = waiting_text
            color = "#6c757d"
        step_label = "Step 2" if lang == 'en' else "步骤 2"
        step_desc = "Cohort Selection" if lang == 'en' else "队列筛选"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        if step1_done and step2_done:
            status = done_text if step3_done else in_progress_text
            color = "#28a745" if step3_done else "#ffc107"
        else:
            status = waiting_text
            color = "#6c757d"
        step_label = "Step 3" if lang == 'en' else "步骤 3"
        step_desc = "Select Features" if lang == 'en' else "选择特征"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        if step1_done and step2_done and step3_done:
            status = done_text if step4_done else in_progress_text
            color = "#28a745" if step4_done else "#ffc107"
        else:
            status = waiting_text
            color = "#6c757d"
        step_label = "Step 4" if lang == 'en' else "步骤 4"
        step_desc = "Export Data" if lang == 'en' else "导出数据"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ============ 动态引导内容 ============
    # 添加引导锚点和动态标题（根据当前步骤变化）
    st.markdown('<div id="guide"></div>', unsafe_allow_html=True)
    
    # 🆕 动态Guide标题，根据Progress自动转换
    if not step1_done:
        guide_step = "Data Source" if lang == 'en' else "数据源配置"
    elif not step2_done:
        guide_step = "Cohort Selection" if lang == 'en' else "队列筛选"
    elif not step3_done:
        guide_step = "Select Features" if lang == 'en' else "特征选择"
    elif not step4_done:
        guide_step = "Export Data" if lang == 'en' else "数据导出"
    else:
        guide_step = "Complete" if lang == 'en' else "完成"
    
    guide_title = f"📍 Guide: {guide_step}" if lang == 'en' else f"📍 引导: {guide_step}"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.8rem;">{guide_title}</h2>', unsafe_allow_html=True)
    
    if not step1_done:
        # 步骤1引导：配置数据源
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card" style="font-size: 1.1rem; line-height: 1.8;">
                <h3 style="color: #667eea; margin-bottom: 15px;">👈 Configure Data Source in the Left Sidebar</h3>
                <p style="margin-bottom: 15px;">Choose one of the following modes to get started:</p>
                <div style="background: rgba(16, 185, 129, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color: #10b981;">🎭 Demo Mode (Recommended for First-time Users)</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>No real data required - system generates realistic simulated ICU data</li>
                        <li>Perfect for learning how PyRICU works</li>
                        <li>Adjust patient count (50-500) and data duration (24-168 hours)</li>
                        <li>Click <b>"✅ Confirm Data Source"</b> when ready</li>
                    </ul>
                </div>
                <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #3b82f6;">📊 Real Data Mode (For Research)</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>Supports MIMIC-IV, eICU, AUMC, HiRID, MIMIC-III, SICdb</li>
                        <li>Enter your local database path</li>
                        <li>Click "Validate Path" to verify data format</li>
                        <li>All processing is done locally - your data stays secure 🔒</li>
                    </ul>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card" style="font-size: 1.1rem; line-height: 1.8;">
                <h3 style="color: #667eea; margin-bottom: 15px;">👈 在左侧边栏配置数据源</h3>
                <p style="margin-bottom: 15px;">选择以下任一模式开始使用：</p>
                <div style="background: rgba(16, 185, 129, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color: #10b981;">🎭 演示模式（推荐新用户使用）</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>无需真实数据 - 系统会生成逼真的模拟ICU数据</li>
                        <li>非常适合学习PyRICU的工作方式</li>
                        <li>可调整患者数量（50-500）和数据时长（24-168小时）</li>
                        <li>设置完成后点击 <b>"✅ 确认数据源配置"</b></li>
                    </ul>
                </div>
                <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #3b82f6;">📊 真实数据模式（用于科研）</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>支持 MIMIC-IV、eICU、AUMC、HiRID、MIMIC-III、SICdb</li>
                        <li>输入您本地的数据库路径</li>
                        <li>点击"验证路径"确认数据格式</li>
                        <li>所有处理都在本地完成 - 您的数据安全无忧 🔒</li>
                    </ul>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
    elif not step2_done:
        # 步骤2引导：队列筛选
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card" style="font-size: 1.1rem; line-height: 1.8;">
                <h3 style="color: #667eea; margin-bottom: 15px;">👈 Configure Cohort Selection in the Left Sidebar</h3>
                <p style="margin-bottom: 15px;">Define your study cohort by filtering patients:</p>
                <div style="background: rgba(99, 102, 241, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color: #6366f1;">🔧 Available Filters</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li><b>Age Range</b> - Filter patients by age (e.g., 18-65 years)</li>
                        <li><b>Gender</b> - Select Male, Female, or Any</li>
                        <li><b>Survival Status</b> - Include survivors, non-survivors, or all</li>
                        <li><b>ICU Stay Duration</b> - Minimum length of stay in hours</li>
                    </ul>
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #10b981;">💡 Tips</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>Enable "Cohort Filtering" toggle to activate filters</li>
                        <li>You can skip this step by clicking <b>"✅ Confirm (No Filtering)"</b></li>
                        <li>Filters will be applied when generating/loading data</li>
                    </ul>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card" style="font-size: 1.1rem; line-height: 1.8;">
                <h3 style="color: #667eea; margin-bottom: 15px;">👈 在左侧边栏配置队列筛选</h3>
                <p style="margin-bottom: 15px;">通过筛选患者来定义您的研究队列：</p>
                <div style="background: rgba(99, 102, 241, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color: #6366f1;">🔧 可用的筛选条件</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li><b>年龄范围</b> - 按年龄筛选患者（如 18-65 岁）</li>
                        <li><b>性别</b> - 选择男性、女性或不限</li>
                        <li><b>存活状态</b> - 包含存活者、死亡者或全部</li>
                        <li><b>ICU住院时长</b> - 最短住院时长（小时）</li>
                    </ul>
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #10b981;">💡 提示</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>启用"队列筛选"开关来激活筛选功能</li>
                        <li>可以点击 <b>"✅ 确认（不筛选）"</b> 跳过此步骤</li>
                        <li>筛选条件将在生成/加载数据时应用</li>
                    </ul>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
    elif not step3_done:
        # 步骤3引导：选择特征
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card" style="font-size: 1.1rem; line-height: 1.8;">
                <h3 style="color: #0369a1; margin-bottom: 15px;">👈 Select Features in the Left Sidebar</h3>
                <p style="margin-bottom: 15px;">PyRICU provides <b>150+ comprehensive ICU clinical features</b>, covering:</p>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 15px;">
                    <div style="flex: 1; min-width: 200px; background: rgba(59, 130, 246, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #1d4ed8;">📊 Vital Signs</b>
                        <p style="color: #1e40af; margin-top: 5px; font-size: 0.95rem;">Heart rate, blood pressure, temperature, SpO2, respiratory rate</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(16, 185, 129, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #047857;">🧪 Laboratory Tests</b>
                        <p style="color: #065f46; margin-top: 5px; font-size: 0.95rem;">Blood chemistry, hematology, coagulation, blood gas analysis</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(251, 191, 36, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #b45309;">💊 Medications</b>
                        <p style="color: #92400e; margin-top: 5px; font-size: 0.95rem;">Vasopressors, sedatives, antibiotics, fluid therapy</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(139, 92, 246, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #6d28d9;">🏥 Clinical Scores</b>
                        <p style="color: #5b21b6; margin-top: 5px; font-size: 0.95rem;">SOFA, GCS, urine output, organ failure indicators</p>
                    </div>
                </div>
                <div style="background: rgba(251, 191, 36, 0.2); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color: #b45309;">🔥 Quick Selection Methods</h4>
                    <ul style="margin-left: 20px; margin-top: 10px; color: #78350f;">
                        <li><b>By Category</b> - Expand a group and select entire group or individual features</li>
                        <li><b>Custom</b> - Mix and match based on your research needs</li>
                    </ul>
                </div>
                <div style="background: rgba(139, 92, 246, 0.2); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #6d28d9;">📖 Need Help Choosing?</h4>
                    <p style="margin-top: 10px; color: #5b21b6;">
                        👇 Check the <b>Data Dictionary</b> below for detailed descriptions of each feature!
                    </p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card" style="font-size: 1.1rem; line-height: 1.8;">
                <h3 style="color: #0369a1; margin-bottom: 15px;">👈 在左侧边栏选择特征</h3>
                <p style="margin-bottom: 15px;">PyRICU 提供 <b>150+ 全面的 ICU 临床特征</b>，涵盖：</p>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 15px;">
                    <div style="flex: 1; min-width: 200px; background: rgba(59, 130, 246, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #1d4ed8;">📊 生命体征</b>
                        <p style="color: #1e40af; margin-top: 5px; font-size: 0.95rem;">心率、血压、体温、血氧饱和度、呼吸频率</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(16, 185, 129, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #047857;">🧪 实验室检验</b>
                        <p style="color: #065f46; margin-top: 5px; font-size: 0.95rem;">血生化、血常规、凝血功能、血气分析</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(251, 191, 36, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #b45309;">💊 药物治疗</b>
                        <p style="color: #92400e; margin-top: 5px; font-size: 0.95rem;">血管活性药、镇静药、抗生素、液体治疗</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(139, 92, 246, 0.15); padding: 12px; border-radius: 8px;">
                        <b style="color: #6d28d9;">🏥 临床评分</b>
                        <p style="color: #5b21b6; margin-top: 5px; font-size: 0.95rem;">SOFA 评分、GCS 评分、尿量、器官衰竭指标</p>
                    </div>
                </div>
                <div style="background: rgba(251, 191, 36, 0.2); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color: #b45309;">🔥 快速选择方法</h4>
                    <ul style="margin-left: 20px; margin-top: 10px; color: #78350f;">
                        <li><b>按类别</b> - 展开某个分组，选择整组或单个特征</li>
                        <li><b>自定义</b> - 根据研究需求自由组合</li>
                    </ul>
                </div>
                <div style="background: rgba(139, 92, 246, 0.2); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #6d28d9;">📖 不知道该选什么？</h4>
                    <p style="margin-top: 10px; color: #5b21b6;">
                        👇 查看下方的 <b>数据字典</b>，了解每个特征的详细描述！
                    </p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
    elif not step4_done:
        # Step 4 Guide: Export Data
        # 🆕 检查是否正在导出或刚完成导出
        exporting_in_progress = st.session_state.get('_exporting_in_progress', False)
        
        if exporting_in_progress:
            # 🆕 导出正在进行中，显示进度标题
            if lang == 'en':
                st.markdown('''<div class="highlight-card" style="border-left: 4px solid #ff9800; background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%);">
<h3 style="color: #ff9800; margin-bottom: 10px;">⏳ Export in Progress...</h3>
<p style="color: #555; margin: 0; font-size: 1.1rem;">Please wait while your data is being exported. Progress details will appear below.</p>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div class="highlight-card" style="border-left: 4px solid #ff9800; background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%);">
<h3 style="color: #ff9800; margin-bottom: 10px;">⏳ 导出进行中...</h3>
<p style="color: #555; margin: 0; font-size: 1.1rem;">请稍候，数据正在导出中。进度详情将显示在下方。</p>
</div>''', unsafe_allow_html=True)
        else:
            # 显示导出教程
            if lang == 'en':
                export_guide_html = '''<div class="highlight-card" style="border-left: 4px solid #28a745;">
<h3 style="color: #28a745; margin-bottom: 15px;">📥 How to Export Data</h3>
<div style="display: flex; gap: 25px; flex-wrap: wrap;">
<div style="flex: 1; min-width: 280px;">
<ol style="color: #1a1a1a; font-size: 1.1rem; line-height: 1.8;">
<li>Go to <b>"Data Export"</b> tab above</li>
<li>Select export format (CSV/Parquet/Excel)</li>
<li>Choose save location</li>
<li>Click <b>"Export Data"</b> button</li>
</ol>
<p style="color: #28a745; margin-top: 10px; font-size: 1rem;">✅ Best for large datasets - saves directly to disk without loading to memory</p>
</div>
</div>
</div>'''
                st.markdown(export_guide_html, unsafe_allow_html=True)
            else:
                export_guide_html = '''<div class="highlight-card" style="border-left: 4px solid #28a745;">
<h3 style="color: #28a745; margin-bottom: 15px;">📥 如何导出数据</h3>
<div style="display: flex; gap: 25px; flex-wrap: wrap;">
<div style="flex: 1; min-width: 280px;">
<ol style="color: #1a1a1a; font-size: 1.1rem; line-height: 1.8;">
<li>点击上方 <b>"数据导出"</b> 标签页</li>
<li>选择导出格式（CSV/Parquet/Excel）</li>
<li>选择保存位置</li>
<li>点击 <b>"导出数据"</b> 按钮</li>
</ol>
<p style="color: #28a745; margin-top: 10px; font-size: 1rem;">✅ 适合大数据集 - 直接保存到磁盘，不占用内存</p>
</div>
<div style="flex: 1; min-width: 280px;">
<ol style="color: #1a1a1a; font-size: 1.1rem; line-height: 1.8;">
<li>点击上方 <b>"数据导出"</b> 标签页</li>
<li>选择导出格式（CSV/Parquet/Excel）</li>
<li>选择保存位置</li>
<li>点击 <b>"导出数据"</b> 按钮</li>
</ol>
<p style="color: #28a745; margin-top: 10px; font-size: 1rem;">✅ 适合大数据集 - 直接保存到磁盘，不占用内存</p>
</div>
</div>
</div>'''
                st.markdown(export_guide_html, unsafe_allow_html=True)
            
            # 显示当前选择摘要
            selected = st.session_state.get('selected_concepts', [])
            if st.session_state.get('use_mock_data', False):
                source_info = "🎭 Demo Mode" if lang == 'en' else "🎭 演示模式"
            else:
                source_info = f"📊 {st.session_state.get('data_path', '')}"
            
            source_label = "Data Source" if lang == 'en' else "数据源"
            feat_label = "Selected Features" if lang == 'en' else "已选特征"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="stat-label">{source_label}</div>
                    <div style="font-weight:600">{source_info}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="stat-label">{feat_label}</div>
                    <div class="stat-number">{len(selected)}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # 🆕 导出进度区域（无论是否正在导出都创建，导出时内容会填充进来）
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        export_section = st.container()
        st.session_state['_export_progress_container'] = export_section
    
    else:
        # 所有步骤完成 - Guide: Complete
        
        # 🆕 首先检查是否有刚完成的导出结果要显示
        export_result = st.session_state.get('_export_success_result')
        if export_result:
            # 显示导出成功消息
            exported_files = export_result['files']
            export_dir = export_result['export_dir']
            total_elapsed = export_result['total_time']
            module_times = export_result.get('module_times', {})
            
            success_msg = f"✅ Successfully exported {len(exported_files)} files to `{export_dir}`" if lang == 'en' else f"✅ 成功导出 {len(exported_files)} 个文件到 `{export_dir}`"
            st.success(success_msg)
            
            # 显示时间统计
            time_stats_title = "⏱️ Export Time Statistics" if lang == 'en' else "⏱️ 导出耗时统计"
            with st.expander(time_stats_title, expanded=False):
                for mod_name, mod_time in module_times.items():
                    if mod_time >= 60:
                        time_str = f"{mod_time/60:.1f} min"
                    else:
                        time_str = f"{mod_time:.1f} s"
                    st.text(f"  • {mod_name}: {time_str}")
                
                if total_elapsed >= 60:
                    total_str = f"{total_elapsed/60:.1f} min"
                else:
                    total_str = f"{total_elapsed:.1f} s"
                total_msg = f"**Total: {total_str}**" if lang == 'en' else f"**总计: {total_str}**"
                st.markdown(total_msg)
            
            # 显示导出的文件列表
            view_files_label = "📁 View Exported Files" if lang == 'en' else "📁 查看导出文件"
            with st.expander(view_files_label, expanded=True):
                for f in exported_files[:10]:
                    st.caption(f"• {Path(f).name}")
                if len(exported_files) > 10:
                    more_msg = f"... and {len(exported_files) - 10} more files" if lang == 'en' else f"... 及其他 {len(exported_files) - 10} 个文件"
                    st.caption(more_msg)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            # 清除导出结果，避免重复显示
            del st.session_state['_export_success_result']
        
        # 显示状态概览卡片
        col1, col2, col3, col4 = st.columns(4)
        
        db_label = "Database" if lang == 'en' else "数据库"
        feat_label = "Loaded Features" if lang == 'en' else "已加载特征"
        patient_label = "Patients" if lang == 'en' else "患者数量"
        status_label = "Status" if lang == 'en' else "数据状态"
        ready_status = "✅ Ready" if lang == 'en' else "✅ 就绪"
        
        with col1:
            db_display = "🎭 DEMO" if st.session_state.get('use_mock_data', False) else st.session_state.get('database', 'N/A').upper()
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{db_label}</div>
                <div class="stat-number" style="font-size:1.8rem">{db_display}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            # 显示已选择的特征数（selected_concepts），而非已加载的（loaded_concepts 可能为空）
            n_concepts = len(st.session_state.get('selected_concepts', []))
            if n_concepts == 0:
                n_concepts = len(st.session_state.loaded_concepts)
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{feat_label}</div>
                <div class="stat-number">{n_concepts}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            # 显示患者数：优先使用导出时记录的实际数量（cohort filter 后的真实数量）
            n_patients = 0
            id_col = st.session_state.get('id_col', 'stay_id')
            
            # 🔧 DEBUG: 打印各个来源的值
            print(f"[DEBUG Guide] _exported_patient_count: {st.session_state.get('_exported_patient_count')}")
            print(f"[DEBUG Guide] patient_ids len: {len(st.session_state.patient_ids) if st.session_state.patient_ids else 0}")
            print(f"[DEBUG Guide] mock_params: {st.session_state.get('mock_params')}")
            
            # 最高优先级：导出时记录的实际患者数（filter 后的真实数量）
            if st.session_state.get('_exported_patient_count'):
                n_patients = st.session_state['_exported_patient_count']
            
            # 其次：从已加载数据中计算唯一患者数
            if n_patients == 0 and st.session_state.loaded_concepts:
                all_ids = set()
                for df in st.session_state.loaded_concepts.values():
                    if isinstance(df, pd.DataFrame) and id_col in df.columns:
                        all_ids.update(df[id_col].unique())
                if all_ids:
                    n_patients = len(all_ids)
            
            # 然后：使用 patient_ids 列表
            if n_patients == 0 and st.session_state.patient_ids:
                n_patients = len(st.session_state.patient_ids)
            
            # 最后：用 mock_params（仅用于显示预期值）
            if n_patients == 0:
                mock_params = st.session_state.get('mock_params', {})
                if mock_params.get('n_patients'):
                    n_patients = mock_params['n_patients']
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{patient_label}</div>
                <div class="stat-number">{n_patients:,}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{status_label}</div>
                <div class="stat-number" style="color:#28a745">{ready_status}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # 🆕 What's Next? 两个选项
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        next_step_title = "🔄 What's Next?" if lang == 'en' else "🔄 下一步？"
        st.markdown(f"### {next_step_title}")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            # Option A: Quick Visualization
            if lang == 'en':
                st.markdown('''<div class="highlight-card" style="border-left: 4px solid #0277bd;">
<h4 style="color: #0277bd; margin-bottom: 12px;">📈 Option A: Quick Visualization</h4>
<p style="color: #1a1a1a; margin-bottom: 15px;">Explore your data with interactive visualizations:</p>
<ul style="color: #2a2a2a; margin: 10px 0 0 15px; font-size: 0.95rem; line-height: 1.8;">
<li><b>Data Tables Explorer</b> — Browse and explore loaded data by module, view complete data tables with sorting and filtering</li>
<li><b>Time Series Analysis</b> — Visualize clinical trends over time with multi-feature overlay, interactive zoom, and customizable aggregation</li>
<li><b>Patient Overview</b> — Comprehensive single-patient dashboard showing all clinical trajectories and key events</li>
<li><b>Data Quality Assessment</b> — Analyze missing rates, temporal coverage, and data completeness across all features</li>
</ul>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div class="highlight-card" style="border-left: 4px solid #0277bd;">
<h4 style="color: #0277bd; margin-bottom: 12px;">📈 选项 A：快速可视化</h4>
<p style="color: #1a1a1a; margin-bottom: 15px;">通过交互式可视化探索数据：</p>
<ul style="color: #2a2a2a; margin: 10px 0 0 15px; font-size: 0.95rem; line-height: 1.8;">
<li><b>数据表浏览器</b> — 按模块浏览和探索已加载数据，查看完整数据表并支持排序筛选</li>
<li><b>时序分析</b> — 可视化临床指标随时间的变化趋势，支持多特征叠加、交互缩放和自定义聚合</li>
<li><b>患者概览</b> — 综合单患者仪表盘，展示所有临床轨迹和关键事件</li>
<li><b>数据质量评估</b> — 分析所有特征的缺失率、时间覆盖度和数据完整性</li>
</ul>
</div>''', unsafe_allow_html=True)
            
            # Option A 按钮
            viz_label = "📈 Go to Visualization" if lang == 'en' else "📈 前往可视化"
            if st.button(viz_label, use_container_width=True, key="goto_viz_home", type="primary"):
                st.session_state['_scroll_to_tab'] = 'viz'
                st.rerun()
        
        with col_opt2:
            # Option B: Cohort Analysis
            if lang == 'en':
                st.markdown('''<div class="highlight-card" style="border-left: 4px solid #6d28d9;">
<h4 style="color: #6d28d9; margin-bottom: 12px;">🔬 Option B: Cohort Analysis</h4>
<p style="color: #1a1a1a; margin-bottom: 15px;">Perform statistical analysis on your cohort:</p>
<ul style="color: #2a2a2a; margin: 10px 0 0 15px; font-size: 0.95rem; line-height: 1.8;">
<li><b>Group Comparison Analysis</b> — Compare subgroups with statistical tests</li>
<li><b>Multi-Database Feature Distribution</b> — Compare feature distributions across different ICU databases</li>
<li><b>Cohort Dashboard</b> — Interactive overview of cohort demographics, outcomes, and key clinical characteristics</li>
</ul>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div class="highlight-card" style="border-left: 4px solid #6d28d9;">
<h4 style="color: #6d28d9; margin-bottom: 12px;">🔬 选项 B：队列分析</h4>
<p style="color: #1a1a1a; margin-bottom: 15px;">对队列进行统计分析：</p>
<ul style="color: #2a2a2a; margin: 10px 0 0 15px; font-size: 0.95rem; line-height: 1.8;">
<li><b>组间比较分析</b> — 使用统计检验（t检验、卡方检验、Mann-Whitney U）比较亚组并生成 Table 1</li>
<li><b>多数据库特征分布</b> — 比较不同ICU数据库（MIMIC、eICU等）间的特征分布差异</li>
<li><b>队列仪表盘</b> — 队列人口统计学、结局和关键临床特征的交互式概览</li>
</ul>
</div>''', unsafe_allow_html=True)
            
            # Option B 按钮
            cohort_label = "🔬 Go to Cohort Analysis" if lang == 'en' else "🔬 前往队列分析"
            if st.button(cohort_label, use_container_width=True, key="goto_cohort_home", type="primary"):
                st.session_state['_scroll_to_tab'] = 'cohort'
                st.rerun()
        
        # 🆕 在 Guide: Complete 下方创建导出进度区域
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        export_section = st.container()
        st.session_state['_export_progress_container'] = export_section
    
    # ============ 数据字典展示 ============
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    # 添加字典锚点和大标题
    st.markdown('<div id="dictionary"></div>', unsafe_allow_html=True)
    dict_header = "📖 Data Dictionary" if lang == 'en' else "📖 数据字典"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.6rem;">{dict_header}</h2>', unsafe_allow_html=True)
    
    # 添加数据字典说明
    if lang == 'en':
        st.markdown('''
        <div style="background: rgba(102, 126, 234, 0.15); padding: 18px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #667eea;">
            <p style="color: #333; font-size: 1.15rem; margin: 0; line-height: 1.7;">
                📚 <b>Reference Guide</b>: This dictionary contains all 150+ ICU clinical features available in PyRICU. 
                Each feature includes its code name, full description, and measurement unit. 
                Use this to understand what data you're extracting and make informed selections.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="background: rgba(102, 126, 234, 0.15); padding: 18px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #667eea;">
            <p style="color: #333; font-size: 1.15rem; margin: 0; line-height: 1.7;">
                📚 <b>参考指南</b>：本字典包含 PyRICU 提供的全部 150+ ICU 临床特征。
                每个特征包括代码名称、完整描述和测量单位。
                使用此字典了解您正在提取的数据，做出明智的选择。
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    render_home_data_dictionary(lang)
    
    # 页脚信息
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if lang == 'en':
        st.markdown('''
        <div style="text-align:center;color:#aaa;font-size:0.85rem">
            <p>🏥 PyRICU - Python Re-Implementation of RICU | 
            📦 <a href="https://github.com/your-repo/pyricu" style="color:#4fc3f7">GitHub</a> | 
            📖 <a href="#" style="color:#4fc3f7">Docs</a></p>
            <p>All data processing is done locally, no data is uploaded to any server 🔒</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align:center;color:#aaa;font-size:0.85rem">
            <p>🏥 PyRICU - Python Re-Implementation of RICU | 
            📦 <a href="https://github.com/your-repo/pyricu" style="color:#4fc3f7">GitHub</a> | 
            📖 <a href="#" style="color:#4fc3f7">文档</a></p>
            <p>所有数据处理均在本地完成，不会上传到任何服务器 🔒</p>
        </div>
        ''', unsafe_allow_html=True)


def render_home_data_dictionary(lang):
    """在首页渲染完整的数据字典。"""
    dict_title = "📖 Complete Data Dictionary" if lang == 'en' else "📖 完整数据字典"
    
    with st.expander(dict_title, expanded=True):
        dict_intro = "PyRICU provides 150+ ICU clinical features, organized by category. Click each category to view detailed descriptions." if lang == 'en' else "PyRICU 提供 150+ ICU 临床特征，按类别组织。点击各类别查看详细说明。"
        st.caption(dict_intro)
        
        # 获取分组
        concept_groups = get_concept_groups()
        
        # 使用 tabs 展示各分类
        group_names = list(concept_groups.keys())
        tabs = st.tabs(group_names[:8])  # 前8个分类
        
        for i, tab in enumerate(tabs):
            with tab:
                group_name = group_names[i]
                concepts = concept_groups[group_name]
                _render_home_dict_table(concepts, lang)
        
        # 其余分类用expander
        if len(group_names) > 8:
            more_title = "📂 More Categories" if lang == 'en' else "📂 更多类别"
            st.markdown(f"#### {more_title}")
            for group_name in group_names[8:]:
                feat_text = "features" if lang == 'en' else "个特征"
                with st.expander(f"{group_name} ({len(concept_groups[group_name])} {feat_text})"):
                    _render_home_dict_table(concept_groups[group_name], lang)


def _render_home_dict_table(concepts, lang):
    """为首页数据字典渲染表格。"""
    rows = []
    for concept in concepts:
        if concept in CONCEPT_DICTIONARY:
            eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
            # 获取详细描述
            if concept in CONCEPT_DESCRIPTIONS:
                eng_desc, chn_desc = CONCEPT_DESCRIPTIONS[concept]
            else:
                eng_desc, chn_desc = eng_name, chn_name  # 用名称作为默认描述
            
            if lang == 'en':
                rows.append({
                    'Code': concept,
                    'Full Name': eng_name,
                    'Description': eng_desc,
                    'Unit': unit if unit else '-'
                })
            else:
                rows.append({
                    '代码': concept,
                    '全称': eng_name,
                    '说明': chn_desc,
                    '单位': unit if unit else '-'
                })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True, height=300)
