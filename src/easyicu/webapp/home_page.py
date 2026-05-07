"""Home and loaded-data overview renderers for the EasyICU Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat
from easyicu.webapp.i18n import get_text
from easyicu.webapp.page_header import render_page_header
from easyicu.webapp.services import count_unique_concepts
from easyicu.webapp.ui_helpers import (
    FeatureCard,
    StatCard,
    render_feature_grid,
    render_inline_heading,
    render_kicker,
    render_stat_grid,
    render_status_banner,
    render_steps,
)


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to extracted home renderers."""
    protected = {"render_home", "render_data_overview", "render_home_viz_mode", "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_data_overview():
    """渲染已加载数据的概览页面。"""
    lang = st.session_state.language

    render_page_header(
        get_text('page_data_overview_title'),
        get_text('page_data_overview_subtitle'),
        icon="📊",
        kicker=get_text('page_data_overview_kicker'),
    )

    # 准备就绪提示 - 使用成功横幅
    db_display = "DEMO" if st.session_state.get('use_mock_data', False) else st.session_state.get('database', 'N/A').upper()
    # 🔧 FIX (2026-02-04): 统计唯一概念数
    n_concepts = count_unique_concepts(list(st.session_state.loaded_concepts.keys()))
    # 计算实际患者数
    n_patients = 0
    if st.session_state.loaded_concepts:
        all_ids = set()
        id_col = st.session_state.get('id_col', 'stay_id')
        for df in st.session_state.loaded_concepts.values():
            if isinstance(df, pd.DataFrame) and id_col in df.columns:
                all_ids.update(df[id_col].unique())
        n_patients = len(all_ids) if all_ids else len(st.session_state.patient_ids)
    else:
        n_patients = len(st.session_state.patient_ids)

    _ready_title = "Data Ready" if lang == 'en' else "数据就绪"
    _ready_sub = "Your data is loaded and ready for analysis." if lang == 'en' else "数据已加载，可以开始分析。"
    _lbl_db = "Database" if lang == 'en' else "数据库"
    _lbl_feat = "Concepts" if lang == 'en' else "已加载概念"
    _lbl_pat = "Patients" if lang == 'en' else "患者数量"
    _lbl_status = "Status" if lang == 'en' else "状态"
    _status_val = "Ready" if lang == 'en' else "就绪"

    render_status_banner(_ready_title, _ready_sub, tone="success")

    # 状态概览 - 4 个统计卡片
    render_stat_grid(
        [
            StatCard(_lbl_db, db_display, "primary"),
            StatCard(_lbl_feat, str(n_concepts)),
            StatCard(_lbl_pat, f"{n_patients:,}"),
            StatCard(_lbl_status, f"● {_status_val}", "success"),
        ]
    )

    # 快捷导航卡片
    if lang == 'en':
        features = [
            ("📈", "Time Series", "Interactive time series visualization with single & multi-patient comparison"),
            ("🏥", "Patient Overview", "Multi-dimensional patient dashboard for comprehensive assessment"),
            ("📊", "Data Quality", "Missing rate analysis, distribution statistics & completeness reports"),
        ]
    else:
        features = [
            ("📈", "时序分析", "交互式时间序列可视化，支持单患者/多患者对比"),
            ("🏥", "患者视图", "单患者多维度仪表盘，全面了解患者状态"),
            ("📊", "数据质量", "缺失率分析、数据分布统计及完整度报告"),
        ]

    _nav_title = "Start Exploring" if lang == 'en' else "开始探索"
    _nav_hint = "Select a tab above to begin:" if lang == 'en' else "选择上方标签页开始："
    render_inline_heading(_nav_title, _nav_hint)
    render_feature_grid([FeatureCard(icon, title, desc) for icon, title, desc in features])

    # 数据摘要
    summary_label = "Data Summary" if lang == 'en' else "数据摘要"
    render_inline_heading(summary_label)

    concept_stats = []
    for name, df in st.session_state.loaded_concepts.items():
        if isinstance(df, pd.DataFrame):
            n_records = len(df)
            n_pts = df[st.session_state.id_col].nunique() if st.session_state.id_col in df.columns else 0
            concept_stats.append({
                'Feature' if lang == 'en' else 'Concept': name,
                'Records' if lang == 'en' else '记录数': f"{n_records:,}",
                'Patients' if lang == 'en' else '患者数': n_pts,
            })

    if concept_stats:
        _st_dataframe_compat(st, pd.DataFrame(concept_stats), width="stretch", hide_index=True)


def render_home(app_context: dict[str, Any] | None = None):
    """渲染首页 - 引导式教程，根据用户进度动态显示。"""
    if app_context is not None:
        _install_app_context(app_context)

    lang = st.session_state.language

    # 如果已加载数据，直接显示数据概览
    if len(st.session_state.loaded_concepts) > 0:
        render_data_overview()
        return

    render_page_header(
        get_text('page_tutorial_title'),
        get_text('page_tutorial_subtitle'),
        icon="📚",
        kicker=get_text('page_tutorial_kicker'),
    )

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
    # 检查状态
    viz_dir = st.session_state.get('viz_data_dir', '')
    has_files = False
    if viz_dir and Path(viz_dir).exists():
        files = list(Path(viz_dir).glob('*.csv')) + list(Path(viz_dir).glob('*.parquet')) + list(Path(viz_dir).glob('*.xlsx'))
        has_files = len(files) > 0

    step1_done = has_files
    step2_done = len(st.session_state.loaded_concepts) > 0

    # 进度指示器 - 使用统一的 step-indicator 样式
    _steps_viz = [
        ("Select Directory" if lang == 'en' else "选择目录", "Set data path" if lang == 'en' else "设置数据路径"),
        ("Load & Visualize" if lang == 'en' else "加载可视化", "Explore data" if lang == 'en' else "浏览数据"),
    ]
    _cur_viz = 2 if step2_done else (1 if step1_done else 0)

    render_steps(_steps_viz, current_index=_cur_viz)

    # 教程内容 - 使用更干净的卡片样式
    if not step1_done:
        _task = "Select Data Directory" if lang == 'en' else "选择数据目录"
        st.markdown(f'''
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:20px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
                <div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:0.85rem;font-weight:700">1</div>
                <span style="font-weight:700;font-size:1.05rem;color:#111827">{_task}</span>
            </div>
        ''', unsafe_allow_html=True)
        if lang == 'en':
            st.markdown('''
                <p style="color:#4b5563;margin-bottom:12px;font-size:.92rem">Specify the data directory in the <b>left sidebar</b>. Quick Visualization loads from previously exported files:</p>
                <ul style="color:#6b7280;font-size:.88rem;line-height:1.8;padding-left:20px">
                    <li>Enter the path to exported data files</li>
                    <li>Supported: <b>CSV, Parquet, Excel</b></li>
                    <li>No exports yet? Use "Data Extraction" mode first</li>
                </ul>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;margin-top:14px;font-size:.85rem;color:#92400e">
                    💡 Default path: <code style="background:#fef3c7;padding:2px 6px;border-radius:4px">~/easyicu_export/miiv</code>
                </div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''
                <p style="color:#4b5563;margin-bottom:12px;font-size:.92rem">在<b>左侧边栏</b>指定数据目录。快速可视化从已导出文件加载数据：</p>
                <ul style="color:#6b7280;font-size:.88rem;line-height:1.8;padding-left:20px">
                    <li>输入已导出数据文件的目录路径</li>
                    <li>支持格式：<b>CSV、Parquet、Excel</b></li>
                    <li>还没导出数据？请先使用「数据提取导出」模式</li>
                </ul>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;margin-top:14px;font-size:.85rem;color:#92400e">
                    💡 默认路径：<code style="background:#fef3c7;padding:2px 6px;border-radius:4px">~/easyicu_export/miiv</code>
                </div>
            </div>''', unsafe_allow_html=True)
    else:
        _task = "Load Data" if lang == 'en' else "加载数据"
        st.markdown(f'''
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:20px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
                <div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:0.85rem;font-weight:700">2</div>
                <span style="font-weight:700;font-size:1.05rem;color:#111827">{_task}</span>
            </div>
        ''', unsafe_allow_html=True)
        if lang == 'en':
            st.markdown('''
                <p style="color:#4b5563;margin-bottom:12px;font-size:.92rem">Data files found! Click <b>"Load Data"</b> in the sidebar:</p>
                <ul style="color:#6b7280;font-size:.88rem;line-height:1.8;padding-left:20px">
                    <li>Select specific tables to load (≤ 3 recommended)</li>
                    <li>Click the <b>Load Data</b> button</li>
                    <li>After loading, use the tabs above to explore</li>
                </ul>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''
                <p style="color:#4b5563;margin-bottom:12px;font-size:.92rem">发现数据文件！在侧边栏点击<b>「加载数据」</b>：</p>
                <ul style="color:#6b7280;font-size:.88rem;line-height:1.8;padding-left:20px">
                    <li>选择要加载的表格（建议不超过 3 个）</li>
                    <li>点击<b>「加载数据」</b>按钮</li>
                    <li>加载后使用上方标签页探索</li>
                </ul>
            </div>''', unsafe_allow_html=True)

    # 功能预览 - 使用统一的网格卡片样式
    _preview_title = "After Loading" if lang == 'en' else "加载后可用"
    if lang == 'en':
        features = [
            ("📋", "Data Tables", "Browse & merge"),
            ("📈", "Time Series", "Interactive charts"),
            ("🏥", "Patient Overview", "Patient dashboard"),
            ("📊", "Data Quality", "Missing analysis"),
        ]
    else:
        features = [
            ("📋", "数据大表", "浏览与合并"),
            ("📈", "时序分析", "交互式图表"),
            ("🏥", "患者视图", "患者仪表盘"),
            ("📊", "数据质量", "缺失率分析"),
        ]

    render_kicker(_preview_title)
    render_feature_grid([FeatureCard(icon, title, desc) for icon, title, desc in features], columns=4, muted=True)
