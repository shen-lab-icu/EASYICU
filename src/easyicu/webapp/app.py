"""EasyICU Streamlit 主应用。

本地 ICU 数据分析和可视化平台。
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import os
import threading
from typing import Dict, Any, Optional, List

# 🚀 性能优化：禁用自动缓存清除，保持表缓存在多次加载间复用
os.environ['EASYICU_AUTO_CLEAR_CACHE'] = 'False'

# 尝试导入美化组件
try:
    from streamlit_extras.metric_cards import style_metric_cards
    HAS_EXTRAS = True
except ImportError:
    HAS_EXTRAS = False

# 页面配置
st.set_page_config(
    page_title="EasyICU Data Explorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
# 初始化侧边栏展开状态
if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = False

# 侧边栏宽度设置 - 根据展开状态动态调整
sidebar_width = "100vw" if st.session_state.sidebar_expanded else "clamp(320px, 24vw, 540px)"
sidebar_min_width = "100vw" if st.session_state.sidebar_expanded else "clamp(300px, 22vw, 500px)"
main_display = "none" if st.session_state.sidebar_expanded else "block"

st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        min-width: {sidebar_min_width};
        max-width: {sidebar_width};
        width: {sidebar_width} !important;
        transition: all 0.3s ease;
    }}
    [data-testid="stSidebar"] > div {{
        width: 100% !important;
    }}
    /* 隐藏侧边栏折叠按钮 */
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    button[kind="headerNoPadding"] {{
        display: none !important;
    }}
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    /* 展开时隐藏右侧主内容 */
    [data-testid="stMain"] {{
        display: {main_display} !important;
    }}
</style>
""", unsafe_allow_html=True)

# 🎨 现代化 CSS 样式系统 — Premium Design System v1
st.markdown("""
<style>
    /* ============================================================
       EasyICU Design System v1 — Premium Medical Analytics UI
       ============================================================ */

    /* ============ 谷歌字体导入（必须在所有规则之前） ============ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ============ 全局设计令牌 ============ */
    :root {
        /* 品牌主色 — 医疗蓝青 */
        --primary-color: #2563eb;
        --primary-dark: #1d4ed8;
        --primary-light: #60a5fa;
        --secondary-color: #0891b2;
        --accent-color: #06b6d4;

        /* 渐变系统 */
        --gradient-primary: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
        --gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        --gradient-info: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%);
        --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        --gradient-hero: #385d90;
        --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));

        /* 语义色 */
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --info-color: #06b6d4;

        /* 阴影系统 — 分层深度 */
        --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-soft: 0 4px 16px rgba(0,0,0,0.06);
        --shadow-card: 0 1px 3px rgba(0,0,0,0.06), 0 6px 16px rgba(0,0,0,0.04);
        --shadow-hover: 0 8px 30px rgba(37,99,235,0.12), 0 4px 12px rgba(0,0,0,0.05);
        --shadow-glow: 0 0 20px rgba(56,93,144,0.18), 0 4px 16px rgba(56,93,144,0.10);
        --shadow-elevated: 0 12px 40px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.03);

        /* 圆角 */
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --radius-2xl: 32px;

        /* 动画 */
        --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
        --transition-smooth: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        --transition-fast: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        --transition-spring: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);

        /* 排版 */
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;

        /* 浅色主题 */
        --bg-primary: #f8fafc;
        --bg-secondary: #ffffff;
        --bg-tertiary: #f1f5f9;
        --card-bg-light: #ffffff;
        --text-primary-light: #0f172a;
        --text-secondary-light: #64748b;
        --text-tertiary-light: #94a3b8;
        --border-light: rgba(37,99,235,0.08);
        --border-subtle: #e2e8f0;
        --fluid-body: clamp(0.98rem, 0.12vw + 0.94rem, 1.08rem);
        --fluid-small: clamp(0.88rem, 0.08vw + 0.84rem, 0.98rem);
    }
    
    /* ============ 全局排版 ============ */
    html, body, .stApp, *, *::before, *::after {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color-scheme: light !important;
    }

    /* 强制浅色背景 — 覆盖系统/浏览器深色模式 */
    html, body {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMain"],
    .main {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, rgba(248,250,252,0.97), rgba(241,245,249,0.97)) !important;
        color: #0f172a !important;
    }
    /* Streamlit 内部组件强制浅色 */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: inherit !important;
    }
    /* 全局标签、文本强制深色 — 覆盖 Streamlit 暗色 variables */
    .stApp label,
    .stApp .stMarkdown p,
    .stApp .stMarkdown li,
    .stApp .stMarkdown span,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
    }
    /* Streamlit secondary button 强制浅色 */
    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border-color: #e2e8f0 !important;
    }
    /* Streamlit selectbox / input / text_input 强制浅色 */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    /* Radio / Checkbox / NumberInput / Slider 强制浅色 */
    [data-testid="stRadio"],
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] div[role="radiogroup"],
    [data-testid="stRadio"] div[role="radiogroup"] label {
        color: #0f172a !important;
        background-color: transparent !important;
    }
    [data-testid="stCheckbox"],
    [data-testid="stCheckbox"] label {
        color: #0f172a !important;
    }
    [data-testid="stNumberInput"],
    [data-testid="stNumberInput"] label {
        color: #0f172a !important;
    }
    [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    [data-testid="stSlider"],
    [data-testid="stSlider"] label {
        color: #0f172a !important;
    }
    [data-testid="stMultiSelect"],
    [data-testid="stMultiSelect"] label {
        color: #0f172a !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"],
    [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: var(--gradient-primary) !important;
        border: none !important;
        box-shadow: 0 6px 16px rgba(37,99,235,0.18) !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    /* Tab list 强制浅色 */
    div[data-baseweb="tab-list"] {
        background: rgba(241,245,249,0.8) !important;
    }
    div[data-baseweb="tab-list"] button {
        color: #64748b !important;
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        color: white !important;
    }
    /* Expander 强制浅色 — 包括 summary bar */
    details[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border-color: #e2e8f0 !important;
    }
    details[data-testid="stExpander"] summary {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    details[data-testid="stExpander"] summary span {
        color: #0f172a !important;
    }
    details[data-testid="stExpander"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    /* 对内容元素应用字体 */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stAlert, div[data-testid="stMetric"],
    div[data-baseweb="select"], div[data-baseweb="input"],
    div[data-baseweb="textarea"], div[data-baseweb="tab-list"],
    h1, h2, h3, h4, h5, h6, label,
    input, textarea, select, option, td, th {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ============ 页面头部 ============ */
    .block-container {
        padding-top: 0.5rem !important;
        margin-top: 0 !important;
        max-width: clamp(1040px, 92vw, 1960px) !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
    }

    /* ============ 现代化标签页 — Pill 风格 ============ */
    div[data-baseweb="tab-list"] {
        gap: 6px !important;
        margin-top: 0 !important;
        padding: 6px !important;
        background: rgba(241,245,249,0.8) !important;
        border-radius: var(--radius-xl) !important;
        border: 1px solid var(--border-subtle);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    div[data-baseweb="tab-list"] button {
        font-size: clamp(1rem, 0.12vw + 0.96rem, 1.16rem) !important;
        font-weight: 700 !important;
        padding: clamp(11px, 0.2vw + 10px, 15px) clamp(18px, 0.45vw + 14px, 32px) !important;
        border-radius: var(--radius-lg) !important;
        transition: var(--transition-fast) !important;
        border: none !important;
        background: transparent !important;
        color: var(--text-secondary-light) !important;
        letter-spacing: 0.01em;
    }

    div[data-baseweb="tab-list"] button:hover {
        background: rgba(37,99,235,0.08) !important;
        color: var(--primary-color) !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: 0 2px 12px rgba(37,99,235,0.24) !important;
        border: none !important;
    }

    div[data-baseweb="tab-list"] button p {
        font-size: clamp(1rem, 0.12vw + 0.96rem, 1.16rem) !important;
        font-weight: 700 !important;
    }

    /* Tab 下划线隐藏 */
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    
    /* ============ Metric 卡片 — 毛玻璃风格 ============ */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(37,99,235,0.06);
        border-radius: var(--radius-lg);
        padding: 1.2rem 1.5rem;
        box-shadow: var(--shadow-xs);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: var(--gradient-primary);
        border-radius: 3px 0 0 3px;
        opacity: 0.8;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
        border-color: rgba(37,99,235,0.15);
    }

    div[data-testid="stMetric"] label {
        font-weight: 600 !important;
        color: var(--text-secondary-light) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: var(--text-primary-light) !important;
        background: none !important;
        -webkit-text-fill-color: unset !important;
    }

    
    /* ============ 主标题 — 精致排版 ============ */
    .main-header {
        font-size: clamp(2rem, 1.55rem + 0.9vw, 3rem);
        font-weight: 800;
        color: var(--text-primary-light);
        margin-top: 0;
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }

    .sub-header {
        font-size: clamp(1.04rem, 0.94rem + 0.28vw, 1.34rem);
        color: var(--text-tertiary-light);
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    
    /* ============ 功能卡片 — 精致玻璃 ============ */
    .metric-card, .feature-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 1.4rem;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-card);
        border: 1px solid rgba(37,99,235,0.06);
        transition: var(--transition-smooth);
        color: var(--text-primary-light);
    }

    .metric-card:hover, .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
        border-color: rgba(37,99,235,0.15);
    }

    
    .feature-card h4 {
        color: var(--primary-color);
        margin-bottom: 0.6rem;
        font-weight: 700;
        font-size: 1rem;
    }

    /* ============ 按钮 — 精致渐变 ============ */
    .stButton > button[kind="primary"] {
        background: var(--gradient-primary) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.65rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em;
        box-shadow: 0 2px 8px rgba(37,99,235,0.22) !important;
        transition: var(--transition-smooth) !important;
    }

    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] *,
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-glow) !important;
    }

    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* 侧边栏按钮 */
    [data-testid="stSidebar"] .stButton button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: var(--radius-md) !important;
        letter-spacing: 0.01em;
        transition: var(--transition-smooth) !important;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        box-shadow: var(--shadow-glow) !important;
        transform: translateY(-1px) !important;
    }

    /* ============ 状态提示框 — 更现代 ============ */
    .success-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(52,211,153,0.04));
        border-left: 3px solid var(--success-color);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #065f46;
        font-size: 0.9rem;
    }

    .warning-box {
        background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(251,191,36,0.04));
        border-left: 3px solid var(--warning-color);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #92400e;
        font-size: 0.9rem;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(34,211,238,0.04));
        border-left: 3px solid var(--info-color);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #0e7490;
        font-size: 0.9rem;
    }

    
    /* ============ 分隔线 — 微妙 ============ */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 0.9rem 0;
        border: none;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 0.9rem 0;
    }

    /* ============ 统计数字 ============ */
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--primary-color);
        letter-spacing: -0.02em;
    }

    .stat-label {
        font-size: 0.78rem;
        color: var(--text-tertiary-light);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    
    /* ============ 患者卡片 ============ */
    .patient-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-subtle);
        margin-bottom: 1rem;
        color: var(--text-primary-light);
        transition: var(--transition-smooth);
    }

    .patient-card:hover {
        border-color: rgba(99,102,241,0.2);
        box-shadow: var(--shadow-soft);
        transform: translateY(-1px);
    }

    
    .patient-card.critical { border-color: var(--danger-color); border-width: 2px; }
    .patient-card.warning { border-color: var(--warning-color); border-width: 2px; }
    .patient-card.stable { border-color: var(--success-color); border-width: 2px; }

    /* ============ 侧边栏 — 精致 ============ */
    /* 注意: 侧边栏宽度由顶部动态 CSS 控制 (sidebar_expanded 状态) */

    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, rgba(248,250,252,0.97), rgba(241,245,249,0.97)) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    
    .sidebar-header {
        background: var(--gradient-primary);
        border-radius: var(--radius-lg);
        padding: 1rem 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 2px 12px rgba(99,102,241,0.2);
    }

    .sidebar-header h3 {
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    /* ============ SOFA2 徽章 ============ */
    .sofa2-badge {
        background: linear-gradient(135deg, #ef4444, #f97316);
        color: white;
        padding: 3px 10px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 6px;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 6px rgba(239,68,68,0.25);
    }

    /* ============ 数据表格 ============ */
    .dataframe {
        border-radius: var(--radius-md) !important;
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] {
        border-radius: var(--radius-md);
        border: 1px solid var(--border-subtle);
        overflow: hidden;
    }

    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td {
        color: #000000 !important;
    }

    [data-testid="stDataFrame"] thead th {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    div[data-testid="stDataFrame"] * {
        color: #000000 !important;
    }

    div[data-testid="stDataFrame"] div[role="columnheader"] {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    .dvn-scroller div[class*="header"],
    [class*="headerCell"] {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    
    /* ============ 进度条 ============ */
    .progress-bar {
        height: 6px;
        background: var(--bg-tertiary);
        border-radius: 100px;
        overflow: hidden;
    }

    .progress-bar-fill {
        height: 100%;
        background: var(--gradient-primary);
        border-radius: 100px;
        transition: width 0.5s var(--ease-out-expo);
    }

    /* ============ 高亮卡片 ============ */
    .highlight-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(139,92,246,0.02));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        color: #312e81;
    }

    .highlight-card h4 { color: var(--primary-color); margin-bottom: 0.8rem; font-weight: 700; }
    .highlight-card p, .highlight-card li { color: #4338ca; }
    .highlight-card b { color: var(--primary-dark); }

    
    /* ============ 动画 ============ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeInUp 0.5s var(--ease-out-expo);
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .animate-pulse { animation: pulse 2.5s infinite; }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    /* ============ 输入控件 ============ */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: var(--radius-md) !important;
        border-color: var(--border-subtle) !important;
        transition: var(--transition-fast) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* ============ Expander ============ */
    details[data-testid="stExpander"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        overflow: hidden;
    }

    details[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* ============ Tooltip ============ */
    [data-baseweb="tooltip"] {
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-elevated) !important;
    }

    /* ============ Streamlit Alert 美化 ============ */
    div[data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        border: none !important;
        font-size: 0.88rem !important;
    }

    /* ============ 入口页面 Hero ============ */
    .hero-container {
        background: var(--gradient-hero);
        border-radius: var(--radius-2xl);
        padding: clamp(2.5rem, 2rem + 1.5vw, 4rem) clamp(1.5rem, 1rem + 1vw, 3rem) clamp(2rem, 1.5rem + 1.5vw, 3.5rem);
        margin: 0 auto 2rem;
        max-width: min(900px, 85%);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: none;
        animation: none;
    }

    .hero-title {
        font-size: clamp(2rem, 1.5rem + 1.5vw, 3rem);
        font-weight: 900;
        color: rgba(255,255,255,0.98);
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 1px 0 rgba(0,0,0,0.06);
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.82);
        font-weight: 500;
        letter-spacing: 0.02em;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.30);
        border-radius: 100px;
        padding: 6px 16px;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.92);
        font-weight: 600;
        margin-bottom: 1.2rem;
        letter-spacing: 0.04em;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 30px rgba(15,34,48,0.12);
        position: relative;
        z-index: 1;
    }

    /* ============ 入口模式卡片 — Glass ============ */
    .mode-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: var(--radius-xl);
        padding: 2.5rem 2rem;
        text-align: center;
        cursor: pointer;
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }

    .mode-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255,255,255,0.2);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }

    .mode-card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }

    .mode-card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.6rem;
    }

    .mode-card-desc {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.6);
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    .mode-card-tag {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .mode-card-tag.green {
        background: rgba(16,185,129,0.2);
        color: #34d399;
        border: 1px solid rgba(16,185,129,0.3);
    }

    .mode-card-tag.blue {
        background: rgba(99,102,241,0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.3);
    }

    /* ============ 入口页紧凑总览 ============ */
    .entry-overview {
        max-width: min(1200px, 92vw);
        margin: 1rem auto 0;
        padding: 0;
        background: transparent;
        border: none;
        border-radius: 0;
        box-shadow: none;
    }

    .entry-overview-lead {
        text-align: center;
        color: var(--text-secondary-light);
        font-size: clamp(0.92rem, 0.12vw + 0.88rem, 1.03rem);
        line-height: 1.7;
        max-width: 920px;
        margin: 0.85rem auto 0;
    }

    .entry-overview-panel {
        margin-top: 0.55rem;
        padding: 0.4rem 0 0.2rem;
        border-radius: 0;
        background: transparent;
        border: none;
    }

    .entry-overview-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }

    .entry-overview-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #14263d;
        line-height: 1.3;
        margin-bottom: 0.18rem;
    }

    .entry-overview-subtitle {
        font-size: 0.84rem;
        line-height: 1.55;
        color: #6c8099;
        max-width: 760px;
    }

    .entry-overview-kicker {
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7185a2;
        white-space: nowrap;
    }

    .entry-overview-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: clamp(0.7rem, 0.35rem + 0.5vw, 1rem);
    }

    .entry-overview-item {
        display: flex;
        gap: 0.72rem;
        align-items: flex-start;
        padding: 0.15rem 0.1rem;
        border-radius: 0;
        background: transparent;
        border: none;
    }

    .entry-overview-item.ai {
        background: transparent;
        border: none;
    }

    .entry-overview-icon {
        font-size: 1rem;
        line-height: 1.2;
        margin-top: 0.02rem;
        flex-shrink: 0;
    }

    .entry-overview-item-title {
        font-size: 0.92rem;
        font-weight: 800;
        color: #14263d;
        line-height: 1.28;
        margin-bottom: 0.16rem;
    }

    .entry-overview-item-desc {
        font-size: 0.8rem;
        color: #6f829b;
        line-height: 1.48;
    }

    .entry-db-inline {
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(148,163,184,0.18);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .entry-db-inline-label {
        font-size: 0.75rem;
        color: var(--text-tertiary-light);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .entry-db-inline-list {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.86rem;
        color: var(--text-secondary-light);
        font-weight: 600;
    }

    
    @media (max-width: 768px) {
        .entry-overview-head { flex-direction: column; }
        .entry-overview-grid { grid-template-columns: 1fr; }
        .entry-db-inline { flex-direction: column; align-items: flex-start; }
        .hero-title { font-size: 2rem; }
    }

    /* ============ 步骤指示器 — 精致 ============ */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: clamp(10px, 0.3vw + 8px, 16px);
        padding: clamp(12px, 0.32vw + 10px, 18px) clamp(14px, 0.45vw + 11px, 24px);
        border-radius: var(--radius-md);
        margin-bottom: 8px;
        transition: var(--transition-fast);
        border: 1px solid transparent;
    }

    .step-indicator.active {
        background: rgba(99,102,241,0.06);
        border-color: rgba(99,102,241,0.12);
    }

    .step-indicator.done {
        background: rgba(16,185,129,0.06);
        border-color: rgba(16,185,129,0.12);
    }

    .step-dot {
        width: clamp(28px, 0.45vw + 24px, 40px);
        height: clamp(28px, 0.45vw + 24px, 40px);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: clamp(0.78rem, 0.12vw + 0.74rem, 1rem);
        font-weight: 700;
        flex-shrink: 0;
    }

    .step-dot.pending {
        background: var(--bg-tertiary);
        color: var(--text-tertiary-light);
        border: 2px solid var(--border-subtle);
    }

    .step-dot.active {
        background: var(--gradient-primary);
        color: white;
        border: none;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3);
    }

    .step-dot.done {
        background: var(--success-color);
        color: white;
        border: none;
    }

    .step-text {
        font-size: clamp(0.98rem, 0.16vw + 0.92rem, 1.18rem);
        font-weight: 600;
        color: var(--text-primary-light);
    }

    .step-text small {
        display: block;
        font-size: clamp(0.88rem, 0.10vw + 0.84rem, 1rem);
        font-weight: 500;
        color: var(--text-tertiary-light);
        margin-top: 4px;
    }

    
    /* ============ 响应式适配 — 多分辨率 ============ */

    /* 小屏 (≤1366px, 13-14" 笔记本) */
    @media (max-width: 1366px) {
        .block-container { max-width: 97.5% !important; }
        .main-header { font-size: 1.95rem; }
        .sub-header { font-size: 1.02rem; }
        .step-indicator { padding: 12px 14px; gap: 10px; }
        .step-dot { width: 28px; height: 28px; font-size: 0.78rem; }
        .step-text { font-size: 0.96rem; }
        .step-text small { font-size: 0.86rem; }
        div[data-testid="stMetric"] { padding: 1rem 1.2rem; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        .highlight-card { padding: 1.1rem 1.3rem; }
        .mode-card { padding: 2rem 1.5rem; }
        .mode-card-title { font-size: 1.15rem; }
        div[data-baseweb="tab-list"] button { font-size: 1rem !important; padding: 11px 18px !important; }
        div[data-baseweb="tab-list"] button p { font-size: 1rem !important; }
    }

    /* 大屏 (≥1920px, 24-27" 显示器) */
    @media (min-width: 1920px) {
        .block-container { max-width: min(94vw, 2140px) !important; }
        .hero-container { max-width: min(900px, 65%); }
        .main-header { font-size: 2.45rem; }
        .sub-header { font-size: 1.18rem; }
        .step-indicator { padding: 16px 24px; gap: 14px; }
        .step-dot { width: 36px; height: 36px; font-size: 0.92rem; }
        .step-text { font-size: 1.12rem; }
        .step-text small { font-size: 0.96rem; }
        div[data-testid="stMetric"] { padding: 1.4rem 1.8rem; }
        div[data-baseweb="tab-list"] button { padding: 14px 30px !important; font-size: 1.14rem !important; }
        div[data-baseweb="tab-list"] button p { font-size: 1.14rem !important; }
    }

    /* 超大屏 (≥2560px, 27"+ 2K/4K) */
    @media (min-width: 2560px) {
        .block-container { max-width: min(95vw, 2460px) !important; }
        .hero-container { max-width: min(1000px, 55%); }
        .main-header { font-size: 2.9rem; }
        .sub-header { font-size: 1.32rem; }
        .step-indicator { padding: 18px 28px; gap: 16px; }
        .step-dot { width: 40px; height: 40px; font-size: 1rem; }
        .step-text { font-size: 1.22rem; }
        .step-text small { font-size: 1.02rem; }
        div[data-testid="stMetric"] { padding: 1.6rem 2rem; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.8rem; }
        div[data-baseweb="tab-list"] button { padding: 16px 34px !important; font-size: 1.2rem !important; }
        div[data-baseweb="tab-list"] button p { font-size: 1.2rem !important; }
        .highlight-card { padding: 1.6rem 2rem; }
        .features-grid { gap: 1.4rem; }
        .feature-item { padding: 1.8rem 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .compact-summary-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-left: 3px solid #60a5fa;
        border-radius: 12px;
        padding: 0.68rem 0.82rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        min-height: 64px;
    }
    .compact-summary-card .summary-label {
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.18rem;
    }
    .compact-summary-card .summary-value {
        font-size: 1.22rem;
        line-height: 1.1;
        color: #0f172a;
        font-weight: 800;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .compact-inline-notice {
        border-radius: 10px;
        padding: 0.48rem 0.72rem;
        margin: 0.22rem 0 0.42rem;
        font-size: 0.84rem;
        line-height: 1.45;
        border: 1px solid transparent;
    }
    .compact-inline-notice.info {
        background: rgba(59,130,246,0.08);
        border-color: rgba(59,130,246,0.14);
        color: #1d4ed8;
    }
    .compact-inline-notice.success {
        background: rgba(16,185,129,0.09);
        border-color: rgba(16,185,129,0.15);
        color: #047857;
    }
    .compact-inline-notice.warning {
        background: rgba(245,158,11,0.10);
        border-color: rgba(245,158,11,0.18);
        color: #b45309;
    }
    .compact-section-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: #111827;
        margin: 0 0 0.18rem 0;
        line-height: 1.25;
    }
    .compact-section-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 0.55rem;
        line-height: 1.45;
    }
    .mini-stat-card {
        background: #fff;
        border: 1px solid #dbeafe;
        border-left: 3px solid #60a5fa;
        border-radius: 12px;
        padding: 0.52rem 0.68rem;
        min-height: 48px;
    }
    .mini-stat-card .mini-label {
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 0.1rem;
    }
    .mini-stat-card .mini-value {
        font-size: 1rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
    }
    .tiny-stat-card {
        background: #fff;
        border: 1px solid #dbeafe;
        border-left: 3px solid #60a5fa;
        border-radius: 10px;
        padding: 0.42rem 0.58rem;
        min-height: 42px;
    }
    .tiny-stat-card .tiny-label {
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 0.08rem;
    }
    .tiny-stat-card .tiny-value {
        font-size: 0.94rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.12;
    }
    .server-browser-box {
        background: rgba(255,255,255,0.96);
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 0.7rem 0.8rem 0.8rem;
        margin-top: 0.45rem;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
    }
    .server-browser-path {
        font-size: 0.78rem;
        color: #334155;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.45rem 0.55rem;
        margin: 0.45rem 0 0.55rem;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)


# 数据字典定义 - 特征缩写及其含义
CONCEPT_DICTIONARY = {
    # 生命体征
    'hr': ('Heart Rate', '心率', 'bpm'),
    'map': ('Mean Arterial Pressure', '平均动脉压', 'mmHg'),
    'sbp': ('Systolic Blood Pressure', '收缩压', 'mmHg'),
    'dbp': ('Diastolic Blood Pressure', '舒张压', 'mmHg'),
    'temp': ('Temperature', '体温', '°C'),
    'etco2': ('End-Tidal CO2', '呼气末二氧化碳', 'mmHg'),
    'resp': ('Respiratory Rate', '呼吸频率', 'breaths/min'),
    
    # 呼吸系统
    'pafi': ('PaO2/FiO2 Ratio', '氧合指数', 'mmHg'),
    'safi': ('SpO2/FiO2 Ratio', '脉氧/吸氧比', ''),
    'supp_o2': ('Supplemental Oxygen', '辅助吸氧', 'boolean'),
    'vent_ind': ('Ventilation Duration Windows', '机械通气时间窗', 'boolean'),
    'o2sat': ('Oxygen Saturation (SpO2)', '血氧饱和度', '%'),
    'sao2': ('Arterial Oxygen Saturation', '动脉血氧饱和度', '%'),
    'mech_vent': ('Mechanical Ventilation', '机械通气', 'boolean'),
    'ett_gcs': ('Intubation/Tracheostomy Status', '气管插管/切开状态', 'boolean'),
    'fio2': ('Fraction of Inspired Oxygen', '吸入氧浓度', '%'),
    
    # 血气分析
    'be': ('Base Excess', '碱剩余', 'mEq/L'),
    'cai': ('Ionized Calcium', '离子钙', 'mmol/L'),
    'hbco': ('Carboxyhemoglobin', '碳氧血红蛋白', '%'),
    'lact': ('Lactate', '乳酸', 'mmol/L'),
    'methb': ('Methemoglobin', '高铁血红蛋白', '%'),
    'pco2': ('Partial Pressure of CO2', '二氧化碳分压', 'mmHg'),
    'ph': ('Blood pH', '血液pH值', ''),
    'po2': ('Partial Pressure of O2', '氧分压', 'mmHg'),
    'tco2': ('Total CO2', '总二氧化碳', 'mEq/L'),
    
    # 实验室检查
    'alb': ('Albumin', '白蛋白', 'g/dL'),
    'alp': ('Alkaline Phosphatase', '碱性磷酸酶', 'IU/L'),
    'alt': ('Alanine Aminotransferase', '谷丙转氨酶', 'IU/L'),
    'ast': ('Aspartate Aminotransferase', '谷草转氨酶', 'IU/L'),
    'bicar': ('Bicarbonate', '碳酸氢根', 'mEq/L'),
    'bili': ('Total Bilirubin', '总胆红素', 'mg/dL'),
    'bili_dir': ('Direct Bilirubin', '直接胆红素', 'mg/dL'),
    'bun': ('Blood Urea Nitrogen', '血尿素氮', 'mg/dL'),
    'ca': ('Calcium', '钙', 'mg/dL'),
    'ck': ('Creatine Kinase', '肌酸激酶', 'IU/L'),
    'ckmb': ('CK-MB', '肌酸激酶同工酶', 'ng/mL'),
    'cl': ('Chloride', '氯', 'mEq/L'),
    'crea': ('Creatinine', '肌酐', 'mg/dL'),
    'crp': ('C-Reactive Protein', 'C反应蛋白', 'mg/L'),
    'glu': ('Glucose', '血糖', 'mg/dL'),
    'k': ('Potassium', '钾', 'mEq/L'),
    'mg': ('Magnesium', '镁', 'mg/dL'),
    'na': ('Sodium', '钠', 'mEq/L'),
    'phos': ('Phosphorus', '磷', 'mg/dL'),
    'tnt': ('Troponin T', '肌钙蛋白T', 'ng/mL'),
    
    # 血液学
    'bnd': ('Band Neutrophils', '杆状核中性粒细胞', '%'),
    'esr': ('Erythrocyte Sedimentation Rate', '红细胞沉降率', 'mm/hr'),
    'fgn': ('Fibrinogen', '纤维蛋白原', 'mg/dL'),
    'hgb': ('Hemoglobin', '血红蛋白', 'g/dL'),
    'inr_pt': ('INR (Prothrombin Time)', '国际标准化比值', ''),
    'lymph': ('Lymphocytes', '淋巴细胞', '%'),
    'mch': ('Mean Corpuscular Hemoglobin', '平均红细胞血红蛋白含量', 'pg'),
    'mchc': ('Mean Corpuscular Hemoglobin Concentration', '平均红细胞血红蛋白浓度', 'g/dL'),
    'mcv': ('Mean Corpuscular Volume', '平均红细胞体积', 'fL'),
    'neut': ('Neutrophils', '中性粒细胞', '%'),
    'plt': ('Platelets', '血小板', '×10³/μL'),
    'ptt': ('Partial Thromboplastin Time', '部分凝血活酶时间', 'sec'),
    'wbc': ('White Blood Cells', '白细胞', '×10³/μL'),
    
    # 药物治疗
    'abx': ('Antibiotics', '抗生素使用', 'boolean'),
    'adh_rate': ('Vasopressin Rate', '血管加压素速率', 'units/min'),
    'cort': ('Corticosteroids', '糖皮质激素', 'boolean'),
    'dex': ('Dextrose (D10)', '葡萄糖（10%）', 'mL/hr'),
    'dobu_dur': ('Dobutamine Duration', '多巴酚丁胺持续时间', 'hours'),
    'dobu_rate': ('Dobutamine Rate', '多巴酚丁胺速率', 'mcg/kg/min'),
    'dobu60': ('Dobutamine >60min', '多巴酚丁胺>60分钟', 'boolean'),
    'epi_dur': ('Epinephrine Duration', '肾上腺素持续时间', 'hours'),
    'epi_rate': ('Epinephrine Rate', '肾上腺素速率', 'mcg/kg/min'),
    'ins': ('Insulin', '胰岛素', 'units/hr'),
    'norepi_dur': ('Norepinephrine Duration', '去甲肾上腺素持续时间', 'hours'),
    'norepi_equiv': ('Norepinephrine Equivalent', '去甲肾上腺素当量', 'mcg/kg/min'),
    'norepi_rate': ('Norepinephrine Rate', '去甲肾上腺素速率', 'mcg/kg/min'),
    'vaso_ind': ('Vasopressor Indicator', '血管活性药物指示', 'boolean'),
    
    # 尿量
    'urine': ('Urine Output', '尿量', 'mL'),
    'urine24': ('24h Urine Output', '24小时尿量', 'mL/24h'),
    
    # 神经系统
    'avpu': ('AVPU Scale', 'AVPU意识评分', ''),
    'egcs': ('Eye Component of GCS', 'GCS眼睛评分', ''),
    'gcs': ('Glasgow Coma Scale', '格拉斯哥昏迷评分', ''),
    'mgcs': ('Motor Component of GCS', 'GCS运动评分', ''),
    'rass': ('Richmond Agitation-Sedation Scale', 'RASS镇静评分', ''),
    'tgcs': ('Total GCS', 'GCS总分', ''),
    'vgcs': ('Verbal Component of GCS', 'GCS语言评分', ''),
    
    # 人口统计
    'age': ('Age', '年龄', 'years'),
    'bmi': ('Body Mass Index', '体重指数', 'kg/m²'),
    'height': ('Height', '身高', 'cm'),
    'sex': ('Sex', '性别', ''),
    'weight': ('Weight', '体重', 'kg'),
    
    # SOFA-1 评分
    'sofa': ('SOFA Score (Total)', 'SOFA总分', '0-24'),
    'sofa_resp': ('SOFA Respiratory', 'SOFA呼吸评分', '0-4'),
    'sofa_coag': ('SOFA Coagulation', 'SOFA凝血评分', '0-4'),
    'sofa_liver': ('SOFA Liver', 'SOFA肝脏评分', '0-4'),
    'sofa_cardio': ('SOFA Cardiovascular', 'SOFA心血管评分', '0-4'),
    'sofa_cns': ('SOFA Central Nervous System', 'SOFA神经评分', '0-4'),
    'sofa_renal': ('SOFA Renal', 'SOFA肾脏评分', '0-4'),
    'qsofa': ('Quick SOFA', '快速SOFA评分', '0-3'),
    'sirs': ('SIRS Criteria', 'SIRS标准', '0-4'),
    'mews': ('Modified Early Warning Score', '改良早期预警评分', '0-14'),
    'news': ('National Early Warning Score', '国家早期预警评分', '0-20'),
    'death': ('In-hospital Mortality', '院内死亡', 'boolean'),
    'los_icu': ('ICU Length of Stay', 'ICU住院时长', 'days'),
    'los_hosp': ('Hospital Length of Stay', '住院时长', 'days'),
    
    # SOFA-2 评分 (2025年新标准)
    'sofa2': ('SOFA-2 Score (Total)', 'SOFA-2总分 (2025新标准)', '0-24'),
    'sofa2_resp': ('SOFA-2 Respiratory', 'SOFA-2呼吸评分', '0-4'),
    'sofa2_coag': ('SOFA-2 Coagulation', 'SOFA-2凝血评分', '0-4'),
    'sofa2_liver': ('SOFA-2 Liver', 'SOFA-2肝脏评分', '0-4'),
    'sofa2_cardio': ('SOFA-2 Cardiovascular', 'SOFA-2心血管评分', '0-4'),
    'sofa2_cns': ('SOFA-2 Central Nervous System', 'SOFA-2神经评分', '0-4'),
    'sofa2_renal': ('SOFA-2 Renal', 'SOFA-2肾脏评分', '0-4'),
    
    # Sepsis 诊断
    'sep3_sofa1': ('Sepsis-3 (SOFA-1 based)', 'Sepsis-3诊断 (基于传统SOFA)', 'boolean'),
    'sep3_sofa2': ('Sepsis-3 (SOFA-2 based)', 'Sepsis-3诊断 (基于SOFA-2, 2025新标准)', 'boolean'),
    'susp_inf': ('Suspected Infection (ICD or Abx+Culture timing)', '疑似感染 (ICD诊断码或抗生素+培养时间窗)', 'boolean'),
    'infection_icd': ('ICD Infection Diagnosis (eICU only, Angus 2001)', 'ICD感染诊断 (仅eICU, Angus标准)', 'boolean'),
    
    # 呼吸系统 (扩展)
    'spo2': ('Peripheral Oxygen Saturation', '脉搏血氧饱和度', '%'),
    'vent_start': ('Ventilation Start Time', '通气开始时间', 'datetime'),
    'vent_end': ('Ventilation End Time', '通气结束时间', 'datetime'),
    'ecmo': ('ECMO in Use', 'ECMO使用中', 'boolean'),
    'ecmo_indication': ('ECMO Indication', 'ECMO适应症 (呼吸/心血管)', ''),
    'adv_resp': ('Advanced Respiratory Support', '高级呼吸支持 (IMV/NIV/HFNC)', 'boolean'),
    
    # 呼吸机参数 (Ventilator Parameters)
    'peep': ('Positive End-Expiratory Pressure', '呼气末正压', 'cmH2O'),
    'tidal_vol': ('Tidal Volume (Observed)', '潮气量（实测）', 'mL'),
    'tidal_vol_set': ('Tidal Volume (Set)', '潮气量（设定）', 'mL'),
    'pip': ('Peak Inspiratory Pressure', '吸气峰压', 'cmH2O'),
    'plateau_pres': ('Plateau Pressure', '平台压', 'cmH2O'),
    'mean_airway_pres': ('Mean Airway Pressure', '平均气道压', 'cmH2O'),
    'minute_vol': ('Minute Ventilation', '分钟通气量', 'L/min'),
    'vent_rate': ('Ventilator Respiratory Rate', '呼吸机频率', '/min'),
    'compliance': ('Static Compliance', '静态肺顺应性', 'mL/cmH2O'),
    'driving_pres': ('Driving Pressure', '驱动压', 'cmH2O'),
    'ps': ('Pressure Support', '压力支持', 'cmH2O'),
    
    # 血液学 (扩展)
    'basos': ('Basophils', '嗜碱性粒细胞', '%'),
    'eos': ('Eosinophils', '嗜酸性粒细胞', '%'),
    'hba1c': ('Hemoglobin A1C', '糖化血红蛋白', '%'),
    'hct': ('Hematocrit', '红细胞压积', '%'),
    'pt': ('Prothrombin Time', '凝血酶原时间', 'sec'),
    'rbc': ('Red Blood Cell Count', '红细胞计数', '×10⁶/μL'),
    'rdw': ('Red Cell Distribution Width', '红细胞分布宽度', '%'),
    
    # 生化 (扩展)
    'tri': ('Troponin I', '肌钙蛋白I', 'ng/mL'),
    
    # 药物 (扩展)
    'dopa_rate': ('Dopamine Rate', '多巴胺速率', 'mcg/kg/min'),
    'dopa_dur': ('Dopamine Duration', '多巴胺持续时间', 'hours'),
    'dopa60': ('Dopamine >60min', '多巴胺>60分钟', 'boolean'),
    'norepi60': ('Norepinephrine >60min', '去甲肾上腺素>60分钟', 'boolean'),
    'epi60': ('Epinephrine >60min', '肾上腺素>60分钟', 'boolean'),
    'phn_rate': ('Phenylephrine Rate', '去氧肾上腺素速率', 'mcg/kg/min'),
    
    # 肾脏与尿量率
    'rrt': ('Renal Replacement Therapy', '肾脏替代治疗', 'boolean'),
    'rrt_criteria': ('RRT Criteria Met', '满足RRT标准', 'boolean'),
    'uo_6h': ('Average Urine Output Rate (past 6h)', '过去6小时平均尿量率', 'mL/kg/h'),
    'uo_12h': ('Average Urine Output Rate (past 12h)', '过去12小时平均尿量率', 'mL/kg/h'),
    'uo_24h': ('Average Urine Output Rate (past 24h)', '过去24小时平均尿量率', 'mL/kg/h'),
    
    # KDIGO AKI (急性肾损伤) - 🔧 2026-02-04: 移除重复的 kdigo_aki/kdigo_creat/kdigo_uo
    'aki': ('Acute Kidney Injury', '急性肾损伤', 'boolean'),
    'aki_stage': ('AKI Stage (KDIGO)', 'AKI分期（KDIGO标准）', '0-3'),
    'aki_stage_creat': ('AKI Stage (Creatinine)', 'AKI分期（肌酐）', '0-3'),
    'aki_stage_uo': ('AKI Stage (Urine Output)', 'AKI分期（尿量）', '0-3'),
    'aki_stage_rrt': ('AKI Stage (RRT)', 'AKI分期（RRT）', '0-3'),
    # 🔧 2026-02-12: 添加规范化后的 KDIGO 扩展列
    'creat_low_past_48hr': ('Lowest Creatinine in Past 48h', '过去48小时内最低肌酐', 'mg/dL'),
    'creat_low_past_7day': ('Baseline Creatinine (7-day lowest)', '基线肌酐（7天内最低值）', 'mg/dL'),
    'uo_rt_6hr': ('Urine Output Rate (6h rolling window)', '尿量率（6小时滚动窗口）', 'mL/kg/h'),
    'uo_rt_12hr': ('Urine Output Rate (12h rolling window)', '尿量率（12小时滚动窗口）', 'mL/kg/h'),
    'uo_rt_24hr': ('Urine Output Rate (24h rolling window)', '尿量率（24小时滚动窗口）', 'mL/kg/h'),
    
    # 神经 (扩展)
    'sedated_gcs': ('GCS Before Sedation', '镇静前GCS', ''),
    
    # 心血管 (扩展)
    'mech_circ_support': ('Mechanical Circulatory Support', '机械循环支持 (IABP/LVAD/Impella)', 'boolean'),
    'other_vaso': ('Other Vasopressors', '其他血管活性药物', 'boolean'),
    'circ_failure': ('Circulatory Failure', '循环衰竭', 'boolean'),
    'circ_event': ('Circulatory Failure Event Level', '循环衰竭事件等级', '0-3'),
    
    # 神经系统 SOFA-2 扩展
    'motor_response': ('GCS Motor Response', 'GCS运动反应', '1-6'),
    'delirium_positive': ('Delirium Positive (CAM-ICU)', '谵妄阳性（CAM-ICU）', 'boolean'),
    'delirium_tx': ('Delirium Treatment', '谵妄治疗', 'boolean'),
    
    # 人口统计 (扩展)
    'adm': ('Admission Type', '入院类型', ''),
    
    # 微生物
    'samp': ('Body Fluid Sampling (for infection workup)', '体液采样 (用于感染检查)', 'boolean'),
}

# 特征详细描述（英文和中文）
CONCEPT_DESCRIPTIONS = {
    # SOFA-2
    'sofa2': ('Total SOFA-2 score (2025 new standard), sum of 6 organ systems (0-24)', 'SOFA-2总分（2025年新标准），6个器官系统评分之和（0-24分）'),
    'sofa2_resp': ('Respiratory: PaO2/FiO2 (or SpO2/FiO2 if unavailable), scores 3-4 require advanced respiratory support (IMV/NIV/HFNC) or ECMO', '呼吸系统：基于氧合指数，3-4分需要高级呼吸支持（IMV/NIV/HFNC）或ECMO'),
    'sofa2_coag': ('Coagulation: platelet count with updated thresholds (≤50→4, ≤80→3, ≤100→2, ≤150→1)', '凝血系统：基于血小板计数，使用更新的阈值（≤50→4分，≤80→3分，≤100→2分，≤150→1分）'),
    'sofa2_liver': ('Liver: bilirubin with relaxed 1-point threshold (>1.2 mg/dL instead of >1.9)', '肝脏：基于胆红素，1分阈值放宽（>1.2 mg/dL，原为>1.9）'),
    'sofa2_cardio': ('Cardiovascular: combined NE+Epi dose, other vasopressors/inotropes, or mechanical circulatory support (IABP/LVAD/Impella)', '心血管：基于去甲肾+肾上腺素联合剂量、其他血管活性药物或机械循环支持'),
    'sofa2_cns': ('Neurological: GCS score, with delirium (CAM-ICU+ or treatment) adding 1 point if GCS=15', '神经系统：基于GCS评分，若GCS=15但有谵妄（CAM-ICU阳性或接受治疗）则加1分'),
    'sofa2_renal': ('Renal: creatinine and urine output (6h/12h/24h windows), score 4 for RRT or meeting RRT criteria', '肾脏：基于肌酐和尿量（6h/12h/24h窗口），接受RRT或满足RRT标准则为4分'),
    
    # Sepsis
    'sep3_sofa2': ('Sepsis-3 diagnosis: suspected infection + SOFA-2 ≥2 point increase from baseline', '基于SOFA-2的Sepsis-3诊断：疑似感染 + SOFA-2较基线升高≥2分'),
    'sep3_sofa1': ('Sepsis-3 diagnosis: suspected infection + traditional SOFA ≥2 point increase', '基于传统SOFA的Sepsis-3诊断：疑似感染 + SOFA较基线升高≥2分'),
    'susp_inf': ('Suspected infection: (1) ICD infection diagnosis codes (eICU only) OR (2) antibiotics started within 72h of culture OR culture within 24h of antibiotics. Combines infection_icd, abx, and samp concepts.', '疑似感染：(1) ICD感染诊断码（仅eICU可用）或 (2) 培养后72小时内开始抗生素 或 抗生素后24小时内进行培养。由infection_icd、abx和samp概念组合而成'),
    'infection_icd': ('Infection diagnosis based on Angus 2001 ICD criteria (explicit infection codes). ONLY available in eICU database.', '基于Angus 2001 ICD标准的感染诊断（显性感染编码）。仅eICU数据库可用'),
    'samp': ('Body fluid sampling (blood, urine, sputum, etc.) for culture-based infection workup. Used as a marker for suspected infection when combined with antibiotic timing.', '体液采样（血液、尿液、痰液等）用于培养检查。与抗生素时间窗结合作为疑似感染的标志'),
    
    # Vitals
    'hr': ('Heart rate in beats per minute', '每分钟心跳次数'),
    'map': ('Mean arterial pressure = (SBP + 2×DBP) / 3', '平均动脉压 = (收缩压 + 2×舒张压) / 3'),
    'sbp': ('Systolic blood pressure (peak pressure during heartbeat)', '收缩压（心脏收缩时的最高压力）'),
    'dbp': ('Diastolic blood pressure (pressure between heartbeats)', '舒张压（心脏舒张时的最低压力）'),
    'temp': ('Body temperature in Celsius', '体温（摄氏度）'),
    'resp': ('Respiratory rate (breaths per minute)', '呼吸频率（每分钟呼吸次数）'),
    
    # Respiratory
    'pafi': ('PaO2/FiO2 ratio - key oxygenation index for ARDS/SOFA scoring', '氧合指数 - ARDS/SOFA评分的关键指标'),
    'safi': ('SpO2/FiO2 ratio - non-invasive alternative to PaFi (used when SpO2<98%)', '脉氧/吸氧比 - PaFi的非侵入性替代（当SpO2<98%时使用）'),
    'fio2': ('Fraction of inspired oxygen (21-100%)', '吸入氧浓度（21-100%）'),
    'vent_ind': ('Mechanical ventilation indicator (boolean)', '机械通气指示（布尔值）'),
    'ecmo_indication': ("ECMO indication type: 'respiratory' (for lung failure, auto-scores 4 in SOFA-2 resp) or 'cardiovascular' (for heart failure, scores in SOFA-2 cardio as mech_circ_support)", "ECMO适应症类型：'respiratory'（肺衰竭，SOFA-2呼吸评分自动为4分）或'cardiovascular'（心衰，计入SOFA-2心血管的机械循环支持）"),
    'adv_resp': ('Advanced respiratory support indicator: IMV (invasive mechanical ventilation), NIV (non-invasive ventilation), HFNC (high-flow nasal cannula), CPAP, or BiPAP - required for SOFA-2 respiratory scores 3-4', '高级呼吸支持指示：IMV（有创机械通气）、NIV（无创通气）、HFNC（经鼻高流量）、CPAP或BiPAP - SOFA-2呼吸评分3-4分的必要条件'),
    
    # Blood gas
    'lact': ('Lactate - marker of tissue hypoperfusion and shock', '乳酸 - 组织低灌注和休克的标志物'),
    'ph': ('Blood acidity/alkalinity (normal 7.35-7.45)', '血液酸碱度（正常7.35-7.45）'),
    'pco2': ('Partial pressure of CO2 in arterial blood', '动脉血中二氧化碳分压'),
    'po2': ('Partial pressure of O2 in arterial blood', '动脉血中氧分压'),
    
    # Labs
    'crea': ('Serum creatinine - kidney function marker, key for SOFA renal scoring', '血清肌酐 - 肾功能标志物，SOFA肾脏评分关键指标'),
    'bili': ('Total bilirubin - liver function marker, key for SOFA liver scoring', '总胆红素 - 肝功能标志物，SOFA肝脏评分关键指标'),
    'plt': ('Platelet count - coagulation marker, key for SOFA coagulation scoring', '血小板计数 - 凝血功能标志物，SOFA凝血评分关键指标'),
    'wbc': ('White blood cell count - infection/inflammation marker', '白细胞计数 - 感染/炎症标志物'),
    
    # Vasopressors
    'norepi_rate': ('Norepinephrine infusion rate in μg/kg/min (weight-adjusted)', '去甲肾上腺素输注速率（μg/kg/min，体重校正）'),
    'norepi_equiv': ('Norepinephrine equivalent dose - standardized vasopressor potency', '去甲肾上腺素当量 - 标准化血管活性药物效价'),
    'vaso_ind': ('Any vasopressor use indicator (boolean)', '任何血管活性药物使用指示（布尔值）'),
    'other_vaso': ('Other vasopressors/inotropes: vasopressin, phenylephrine, milrinone (combined with dobutamine in SOFA-2 cardio scoring as "has_other_vaso")', '其他血管活性药物：血管加压素、去氧肾上腺素、米力农（在SOFA-2心血管评分中与多巴酚丁胺合并为"has_other_vaso"）'),
    
    # Neurological
    'gcs': ('Glasgow Coma Scale total score (3-15), key for SOFA CNS scoring', '格拉斯哥昏迷评分总分（3-15分），SOFA神经评分关键指标'),
    
    # Outcomes
    'death': ('In-hospital mortality (0=survived, 1=died)', '院内死亡（0=存活，1=死亡）'),
    'los_icu': ('ICU length of stay in days', 'ICU住院时长（天）'),
    'los_hosp': ('Hospital length of stay in days', '总住院时长（天）'),
    
    # AKI
    'aki': ('Acute Kidney Injury (KDIGO Stage ≥1)', '急性肾损伤（KDIGO分期≥1）'),
    'aki_stage': ('KDIGO AKI stage (0-3): max of creatinine and urine output criteria', 'KDIGO AKI分期（0-3）：肌酐和尿量标准的最大值'),
    'aki_stage_creat': ('AKI stage based on creatinine: ≥1.5x baseline or ≥0.3 mg/dL increase in 48h', '基于肌酐的AKI分期：较基线升高≥1.5倍 或 48h内升高≥0.3 mg/dL'),
    'aki_stage_uo': ('AKI stage based on urine output: <0.5 mL/kg/h for 6h (Stage 1), 12h (Stage 2), or <0.3 for 24h (Stage 3)', '基于尿量的AKI分期：<0.5 mL/kg/h持续6h(1期)、12h(2期) 或 <0.3持续24h(3期)'),
    
    # Circulatory failure
    'circ_failure': ('Circulatory failure (circEWS definition): lactate ≥2 mmol/L with hypotension/vasopressors', '循环衰竭（circEWS定义）：乳酸≥2 mmol/L伴低血压或血管活性药物'),
    'circ_event': ('Circulatory failure event level (0-3): based on lactate, MAP, and vasopressor tier', '循环衰竭事件等级（0-3）：基于乳酸、MAP和血管活性药物等级'),
    
    # Other scores
    'qsofa': ('Quick SOFA (0-3): RR≥22 + altered mental status + SBP≤100', '快速SOFA（0-3分）：呼吸频率≥22 + 意识改变 + 收缩压≤100'),
    'sirs': ('SIRS criteria (0-4): temp + HR + RR/PaCO2 + WBC/bands', 'SIRS标准（0-4分）：体温 + 心率 + 呼吸/PaCO2 + 白细胞/杆状核'),
}

# 🔧 FIX (2026-02-09): 随机患者采样，避免 eICU 等多中心数据库的采样偏差
# 使用固定种子保证可复现
def _sample_patient_ids_random(all_ids: list, n: int, seed: int = 42) -> list:
    """从患者ID列表中随机采样n个，使用固定种子保证可复现。
    
    修复 eICU 等多中心数据库的采样偏差问题：
    - 旧方法：all_ids[:n] 按ID排序取前N个 → 可能全部来自同一家医院
    - 新方法：随机采样 → 覆盖多家医院，确保各种特征（GCS、血管活性药等）有数据
    """
    import random
    if len(all_ids) <= n:
        return all_ids
    rng = random.Random(seed)
    return sorted(rng.sample(all_ids, n))


def _get_patient_id_table_files(database: str) -> list:
    """返回数据库特定的患者ID表文件查找列表。
    
    不同数据库的患者ID存储在不同的表中：
    - MIIV/MIMIC-III: icustays.parquet
    - eICU: patient.parquet
    - AUMC: admissions.parquet
    - HiRID: general.parquet
    - SICdb: cases.parquet
    
    返回按优先级排序的文件列表，确保所有数据库都能正确找到患者ID。
    """
    # 数据库特定的主表
    db_specific = {
        'hirid': ['general.parquet'],
        'sic': ['cases.parquet'],
        'aumc': ['admissions.parquet'],
    }
    specific = db_specific.get(database, [])
    # 通用查找列表
    generic = ['icustays.parquet', 'patient.parquet', 'admissions.parquet', 'general.parquet', 'cases.parquet']
    # 合并：先查数据库特定的，再查通用的（去重）
    result = list(specific)
    for f in generic:
        if f not in result:
            result.append(f)
    return result


# 全局特征分组定义 - 供侧边栏和数据字典共用
# 使用英文key，并提供双语显示名称
CONCEPT_GROUPS_INTERNAL = {
    'sofa2_score': ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
    'sofa1_score': ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
    'sepsis3_sofa2': ['sep3_sofa2'],  # 🔧 共享概念移到单独的 sepsis_shared 模块
    'sepsis3_sofa1': ['sep3_sofa1'],  # 🔧 共享概念移到单独的 sepsis_shared 模块
    'sepsis_shared': ['susp_inf', 'infection_icd', 'samp'],  # Sepsis共享概念（已移除sep3）
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],  # 🔧 etco2 移到 ventilator
    'respiratory': ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication', 'adv_resp'],
    'ventilator': ['peep', 'tidal_vol', 'tidal_vol_set', 'pip', 'plateau_pres', 'mean_airway_pres', 'minute_vol', 'vent_rate', 'etco2', 'compliance', 'driving_pres', 'ps'],
    'blood_gas': ['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2'],
    'chemistry': ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri'],
    'hematology': ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt', 'ptt', 'rbc', 'rdw', 'wbc'],
    'vasopressors': ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind', 'other_vaso'],
    'medications': ['abx', 'cort', 'dex', 'ins'],
    # 🔧 2026-02-04: 移除重复的 kdigo_aki/kdigo_creat/kdigo_uo，只保留 aki_* 规范名
    'renal': ['urine', 'urine24', 'uo_6h', 'uo_12h', 'uo_24h', 'rrt', 'rrt_criteria', 'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
              # 规范化后的列名（从 kdigo_* 展开列规范化而来）
              'creat_low_past_48hr', 'creat_low_past_7day', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
    'neurological': ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'tgcs', 'vgcs', 'sedated_gcs', 'motor_response', 'delirium_positive', 'delirium_tx'],
    'circulatory': ['mech_circ_support', 'circ_failure', 'circ_event'],  # 🔧 添加循环衰竭特征
    'demographics': ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
    'other_scores': ['qsofa', 'sirs', 'mews', 'news'],
    'outcome': ['death', 'los_icu', 'los_hosp'],
}

# 双语显示名称映射（优化：更清晰的命名区分评分vs诊断，包含准确特征数量）
CONCEPT_GROUP_NAMES = {
    'sofa2_score': ('⭐ SOFA-2 Scores', '⭐ SOFA-2 评分'),
    'sofa1_score': ('📊 SOFA-1 Scores', '📊 SOFA-1 评分'),
    'sepsis3_sofa2': ('🦠 Sepsis-3 (SOFA-2 based)', '🦠 Sepsis-3 (基于SOFA-2)'),
    'sepsis3_sofa1': ('🦠 Sepsis-3 (SOFA-1 based)', '🦠 Sepsis-3 (基于SOFA-1)'),
    'sepsis_shared': ('🦠 Sepsis Shared Concepts', '🦠 Sepsis 共享概念'),
    'vitals': ('❤️ Vital Signs', '❤️ 生命体征'),
    'respiratory': ('💨 Respiratory System', '💨 呼吸系统'),
    'ventilator': ('🌬️ Ventilator Parameters', '🌬️ 呼吸机参数'),
    'blood_gas': ('🩸 Blood Gas Analysis', '🩸 血气分析'),
    'chemistry': ('🧪 Lab - Chemistry', '🧪 实验室-生化'),
    'hematology': ('🔬 Lab - Hematology', '🔬 实验室-血液学'),
    'vasopressors': ('💉 Vasopressors', '💉 血管活性药物'),
    'medications': ('💊 Other Medications', '💊 其他药物'),
    'renal': ('🚰 Renal & Urine Output', '🚰 肾脏与尿量'),
    'neurological': ('🧠 Neurological', '🧠 神经系统'),
    'circulatory': ('❤️‍🩹 Circulatory System', '❤️‍🩹 循环系统'),
    'demographics': ('👤 Demographics', '👤 人口统计'),
    'other_scores': ('📈 Other Scores', '📈 其他评分'),
    'outcome': ('🎯 Outcome', '🎯 结局'),
}

# 用于时序分析页面的显示名称映射（英文版本）
CONCEPT_GROUPS_DISPLAY = {
    'sofa2_score': '⭐ SOFA-2 Scores',
    'sofa1_score': '📊 SOFA-1 Scores',
    'sepsis3_sofa2': '🦠 Sepsis-3 (SOFA-2)',
    'sepsis3_sofa1': '🦠 Sepsis-3 (SOFA-1)',
    'sepsis_shared': '🦠 Sepsis Shared',
    'vitals': '❤️ Vital Signs',
    'respiratory': '💨 Respiratory',
    'ventilator': '🌬️ Ventilator',
    'blood_gas': '🩸 Blood Gas',
    'chemistry': '🧪 Chemistry',
    'hematology': '🔬 Hematology',
    'vasopressors': '💉 Vasopressors',
    'medications': '💊 Medications',
    'renal': '🚰 Renal',
    'neurological': '🧠 Neurological',
    'circulatory': '❤️‍🩹 Circulatory',
    'demographics': '👤 Demographics',
    'other_scores': '📈 Other Scores',
    'outcome': '🎯 Outcome',
}

# ============ 临床阈值线（用于时序图表默认标注） ============
CLINICAL_THRESHOLDS = {
    'hr':   {'lines': [60, 100], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Bradycardia', 'Tachycardia'], 'unit': 'bpm'},
    'map':  {'lines': [65], 'colors': ['#ef4444'], 'labels': ['Hypotension'], 'unit': 'mmHg'},
    'sbp':  {'lines': [90, 140], 'colors': ['#ef4444', '#f59e0b'], 'labels': ['Hypotension', 'Hypertension'], 'unit': 'mmHg'},
    'spo2': {'lines': [94], 'colors': ['#ef4444'], 'labels': ['Hypoxemia'], 'unit': '%'},
    'temp': {'lines': [36, 38], 'colors': ['#3b82f6', '#ef4444'], 'labels': ['Hypothermia', 'Fever'], 'unit': '°C'},
    'resp': {'lines': [12, 20], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Bradypnea', 'Tachypnea'], 'unit': '/min'},
    'lact': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Elevated'], 'unit': 'mmol/L'},
    'crea': {'lines': [1.2], 'colors': ['#f59e0b'], 'labels': ['Elevated'], 'unit': 'mg/dL'},
    'ph':   {'lines': [7.35, 7.45], 'colors': ['#ef4444', '#ef4444'], 'labels': ['Acidosis', 'Alkalosis'], 'unit': ''},
    'glu':  {'lines': [70, 180], 'colors': ['#ef4444', '#f59e0b'], 'labels': ['Hypoglycemia', 'Hyperglycemia'], 'unit': 'mg/dL'},
    'k':    {'lines': [3.5, 5.0], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Hypokalemia', 'Hyperkalemia'], 'unit': 'mEq/L'},
    'na':   {'lines': [135, 145], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Hyponatremia', 'Hypernatremia'], 'unit': 'mEq/L'},
    'plt':  {'lines': [150], 'colors': ['#ef4444'], 'labels': ['Thrombocytopenia'], 'unit': '×10³/µL'},
    'hgb':  {'lines': [7], 'colors': ['#ef4444'], 'labels': ['Severe Anemia'], 'unit': 'g/dL'},
    'inr_pt': {'lines': [1.5], 'colors': ['#f59e0b'], 'labels': ['Coagulopathy'], 'unit': ''},
    'pafi': {'lines': [300, 200, 100], 'colors': ['#f59e0b', '#ef4444', '#7f1d1d'], 'labels': ['Mild ARDS', 'Moderate ARDS', 'Severe ARDS'], 'unit': 'mmHg'},
    'bili': {'lines': [1.2], 'colors': ['#f59e0b'], 'labels': ['Elevated'], 'unit': 'mg/dL'},
    'sofa': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Organ Dysfunction'], 'unit': 'points'},
    'sofa2': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Organ Dysfunction'], 'unit': 'points'},
    'gcs':  {'lines': [8], 'colors': ['#ef4444'], 'labels': ['Severe Impairment'], 'unit': 'points'},
    'qsofa': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Positive qSOFA'], 'unit': 'points'},
}

# 临床概念分道映射
CLINICAL_LANES = {
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    'labs': ['lact', 'crea', 'bili', 'plt', 'hgb', 'wbc', 'inr_pt', 'glu', 'k', 'na', 'alb', 'crp', 'tnt', 'ph', 'po2', 'pco2'],
    'interventions': ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'fio2', 'peep', 'ins', 'abx', 'cort', 'rrt'],
    'scores': ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'mews', 'news', 'pafi', 'safi'],
}

# 跨库概念可用性（用于 harmonization badge）
CONCEPT_DB_COVERAGE = {
    'hr': 6, 'map': 6, 'sbp': 6, 'dbp': 6, 'resp': 6, 'spo2': 6, 'temp': 6,
    'glu': 6, 'crea': 6, 'bili': 6, 'plt': 6, 'hgb': 6, 'wbc': 6, 'na': 6, 'k': 6,
    'age': 6, 'sex': 6, 'weight': 6, 'height': 6, 'death': 6, 'los_icu': 6,
    'sofa': 6, 'sofa2': 6, 'gcs': 6,
    'lact': 5, 'alb': 5, 'crp': 5, 'fio2': 5, 'po2': 5, 'pco2': 5, 'ph': 5,
    'pafi': 5, 'safi': 5, 'urine': 5,
    'peep': 4, 'tidal_vol': 4, 'ins': 4,
    'mech_vent': 3, 'vent_ind': 3, 'ecmo': 2, 'rrt': 4,
}


def _get_coverage_badge(concept_name: str) -> str:
    """返回概念跨库可用性的 HTML badge。"""
    n = CONCEPT_DB_COVERAGE.get(concept_name, 0)
    if n >= 6:
        color, bg = '#059669', 'rgba(5,150,105,0.1)'
        label = get_text('coverage_badge_full')
    elif n >= 4:
        color, bg = '#d97706', 'rgba(217,119,6,0.1)'
        label = get_text('coverage_badge_caveat')
    elif n >= 2:
        color, bg = '#dc2626', 'rgba(220,38,38,0.1)'
        label = get_text('coverage_badge_partial')
    else:
        color, bg = '#6b7280', 'rgba(107,114,128,0.1)'
        label = get_text('coverage_badge_dbspec')
    return f'<span style="display:inline-block;font-size:0.7rem;font-weight:600;color:{color};background:{bg};padding:1px 8px;border-radius:20px;margin-left:6px;">{label} ({n}/6)</span>'


def _get_missing_cause_tag(concept_name: str, missing_rate: float) -> tuple:
    """为缺失率提供可解释的原因标签。返回 (text, color)。"""
    sparse_events = {'ecmo', 'ecmo_indication', 'mech_circ_support', 'cort', 'rrt',
                     'abx', 'vaso_ind', 'mech_vent', 'vent_ind', 'vent_start', 'vent_end',
                     'sep3_sofa1', 'sep3_sofa2', 'circ_failure', 'circ_event'}
    n_db = CONCEPT_DB_COVERAGE.get(concept_name, 0)
    if n_db == 0:
        return get_text('missing_cause_db'), '#dc2626'
    if concept_name in sparse_events and missing_rate > 0.7:
        return get_text('missing_cause_sparse'), '#6b7280'
    if missing_rate > 0.8:
        return get_text('missing_cause_cohort'), '#d97706'
    return get_text('missing_cause_normal'), '#059669'


# 🔧 ADD (2026-02-05): 支持时序分析的模块（排除静态数据模块）
# 静态数据模块（demographics, outcome）的值不是连续变化的，不适合时序分析
TIME_SERIES_COMPATIBLE_MODULES = {
    'sofa2_score',      # SOFA评分随时间变化
    'sofa1_score',
    'sepsis3_sofa2',    # Sepsis状态随时间变化
    'sepsis3_sofa1',
    'sepsis_shared',
    'vitals',           # 生命体征（心率、血压等）
    'respiratory',      # 呼吸系统
    'ventilator',       # 呼吸机参数
    'blood_gas',        # 血气分析
    'chemistry',        # 生化检验
    'hematology',       # 血液学
    'vasopressors',     # 血管活性药物
    'medications',      # 药物
    'renal',            # 肾脏与尿量
    'neurological',     # 神经系统（GCS等）
    'circulatory',      # 循环系统
    'other_scores',     # 其他评分
    # 排除: 'demographics' - 静态数据（年龄、性别、身高、体重等）
    # 排除: 'outcome' - 静态数据（死亡、住院时长等）
}

def get_concept_groups():
    """根据当前语言返回带正确显示名称的特征分组。"""
    lang = st.session_state.get('language', 'en')
    result = {}
    for key, concepts in CONCEPT_GROUPS_INTERNAL.items():
        en_name, zh_name = CONCEPT_GROUP_NAMES.get(key, (key, key))
        display_name = en_name if lang == 'en' else zh_name
        result[display_name] = concepts
    return result


def _group_label_from_key(group_key: str, lang: str) -> str | None:
    names = CONCEPT_GROUP_NAMES.get(group_key)
    if not names:
        return None
    return names[0] if lang == 'en' else names[1]


def _materialize_feature_preset(payload: dict):
    """Apply a prepared feature preset into sidebar state."""
    lang = st.session_state.get('language', 'en')
    requested_group_keys = [
        g for g in payload.get('group_keys', [])
        if isinstance(g, str) and g in CONCEPT_GROUPS_INTERNAL
    ]
    selected_groups = []
    selected_concepts = set()
    concept_checkboxes = {}

    for group_key in requested_group_keys:
        label = _group_label_from_key(group_key, lang)
        if label:
            selected_groups.append(label)
        for concept in CONCEPT_GROUPS_INTERNAL.get(group_key, []):
            concept_checkboxes[concept] = True
            selected_concepts.add(concept)

    for concept in payload.get('concepts', []):
        if not isinstance(concept, str):
            continue
        selected_concepts.add(concept)
        concept_checkboxes[concept] = True
        for group_key, group_concepts in CONCEPT_GROUPS_INTERNAL.items():
            if concept in group_concepts:
                label = _group_label_from_key(group_key, lang)
                if label and label not in selected_groups:
                    selected_groups.append(label)

    if selected_groups:
        st.session_state.selected_groups = selected_groups
    if selected_concepts:
        st.session_state.concept_checkboxes = concept_checkboxes
        st.session_state.selected_concepts = sorted(selected_concepts)
        st.session_state.step3_confirmed = False


def _preset_ready_for_materialize() -> bool:
    """Only apply feature presets once the workflow is ready to expose Step 3."""
    entry_mode = st.session_state.get('entry_mode', 'none')
    if entry_mode != 'real':
        return False
    data_path = st.session_state.get('data_path')
    if not data_path:
        return False
    if not st.session_state.get('path_validated', False):
        return False
    if not st.session_state.get('step2_confirmed', False):
        return False
    return True


def _maybe_materialize_pending_preset():
    """Auto-apply a pending assistant preset once the user reaches the right step."""
    payload = st.session_state.get('_assistant_pending_feature_preset')
    if not isinstance(payload, dict):
        return
    if payload.get('kind') != 'feature_preset':
        return
    if not _preset_ready_for_materialize():
        return

    _materialize_feature_preset(payload)
    st.session_state.pop('_assistant_pending_feature_preset', None)
    notice_en = payload.get('apply_notice_en') or "Your AI preset is now applied in Step 3. Review the checked features, then confirm selection."
    notice_zh = payload.get('apply_notice_zh') or "AI 预设已应用到步骤3。请检查已勾选特征，然后确认选择。"
    lang = st.session_state.get('language', 'en')
    st.session_state['_assistant_notice'] = notice_en if lang == 'en' else notice_zh


def _apply_assistant_preset():
    """Consume a chat-triggered preset and stage or apply it safely."""
    payload = st.session_state.pop('_assistant_preset_request', None)
    if not isinstance(payload, dict):
        return

    if payload.get('kind') != 'feature_preset':
        return

    lang = st.session_state.get('language', 'en')
    database = payload.get('database')
    if isinstance(database, str) and database:
        st.session_state.entry_mode = 'real'
        st.session_state.use_mock_data = False
        st.session_state.database = database

    if _preset_ready_for_materialize():
        _materialize_feature_preset(payload)
        notice_en = payload.get('apply_notice_en') or "Prepared your sidebar feature preset. Review Step 3 and confirm selection."
        notice_zh = payload.get('apply_notice_zh') or "已应用侧边栏特征预设。请检查步骤3并确认选择。"
        st.session_state['_assistant_notice'] = notice_en if lang == 'en' else notice_zh
    else:
        st.session_state['_assistant_pending_feature_preset'] = payload
        notice_en = payload.get('notice_en') or "Prepared a sidebar preset from the AI assistant."
        notice_zh = payload.get('notice_zh') or "已根据 AI 助手建议预设侧边栏。"
        st.session_state['_assistant_notice'] = notice_en if lang == 'en' else notice_zh
    st.session_state['_scroll_to_top'] = True


# 🔧 列名规范化映射：将重复的展开列名统一为简短的规范名称
# 这些列来自 kdigo_aki, kdigo_creat, kdigo_uo 等复合概念的展开
# 规范化后每个唯一的数据列只保留一份，避免重复
COLUMN_NORMALIZATION_MAP = {
    # kdigo_aki_ 前缀的列 -> 规范名
    'kdigo_aki_aki': 'aki',
    'kdigo_aki_aki_stage': 'aki_stage',
    'kdigo_aki_aki_stage_creat': 'aki_stage_creat',
    'kdigo_aki_aki_stage_uo': 'aki_stage_uo',
    'kdigo_aki_crea': 'crea',  # 注意：crea 在 chemistry 模块也有，需要区分
    'kdigo_aki_creat_low_past_48hr': 'creat_low_past_48hr',
    'kdigo_aki_creat_low_past_7day': 'creat_low_past_7day',
    'kdigo_aki_rrt': 'rrt',
    'kdigo_aki_uo_rt_6hr': 'uo_rt_6hr',
    'kdigo_aki_uo_rt_12hr': 'uo_rt_12hr',
    'kdigo_aki_uo_rt_24hr': 'uo_rt_24hr',
    # kdigo_creat_ 前缀的列 -> 规范名（与 kdigo_aki_ 重复）
    'kdigo_creat_aki_stage_creat': 'aki_stage_creat',
    'kdigo_creat_crea': 'crea',
    'kdigo_creat_creat_low_past_48hr': 'creat_low_past_48hr',
    'kdigo_creat_creat_low_past_7day': 'creat_low_past_7day',
    # kdigo_uo_ 前缀的列 -> 规范名（与 kdigo_aki_ 重复）
    'kdigo_uo_aki_stage_uo': 'aki_stage_uo',
    'kdigo_uo_uo_rt_6hr': 'uo_rt_6hr',
    'kdigo_uo_uo_rt_12hr': 'uo_rt_12hr',
    'kdigo_uo_uo_rt_24hr': 'uo_rt_24hr',
}

# 🔧 反向映射：规范名 -> 所有原始列名（用于查找数据）
NORMALIZED_TO_ORIGINAL_MAP = {}
for orig, norm in COLUMN_NORMALIZATION_MAP.items():
    if norm not in NORMALIZED_TO_ORIGINAL_MAP:
        NORMALIZED_TO_ORIGINAL_MAP[norm] = []
    NORMALIZED_TO_ORIGINAL_MAP[norm].append(orig)


def normalize_column_name(col_name: str) -> str:
    """将列名规范化为统一的简短名称。
    
    对于重复的展开列（如 kdigo_aki_aki, kdigo_creat_crea），返回规范名（如 aki, crea）。
    对于普通列名，直接返回原名。
    
    Args:
        col_name: 原始列名
        
    Returns:
        规范化后的列名
    """
    return COLUMN_NORMALIZATION_MAP.get(col_name, col_name)


def count_unique_columns(column_names: list) -> int:
    """统计唯一列数量（规范化后去重）。
    
    每个唯一的数据列算作一个 concept。
    
    Args:
        column_names: 列名列表
        
    Returns:
        唯一列数量
    """
    normalized = set()
    for col in column_names:
        normalized.add(normalize_column_name(col))
    return len(normalized)


# 🔧 保持向后兼容：旧函数名指向新实现
def map_column_to_concept(col_name: str) -> str:
    """将列名映射到概念名（向后兼容，现在使用规范化）。"""
    return normalize_column_name(col_name)


def count_unique_concepts(column_names: list) -> int:
    """统计唯一概念数量（向后兼容，现在使用规范化）。"""
    return count_unique_columns(column_names)


def get_unique_concepts(column_names: list) -> set:
    """获取唯一概念集合（规范化后去重）。
    
    Args:
        column_names: 列名列表
        
    Returns:
        唯一概念集合
    """
    concepts = set()
    for col in column_names:
        concept = normalize_column_name(col)
        concepts.add(concept)
    return concepts

# 保持向后兼容的CONCEPT_GROUPS（默认中文）
CONCEPT_GROUPS = {
    "⭐ SOFA-2 评分 (2025新标准)": ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
    "⭐ Sepsis-3 诊断 (基于SOFA-2)": ['sep3_sofa2', 'susp_inf', 'infection_icd', 'samp'],
    "Sepsis-3 诊断 (基于SOFA-1)": ['sep3_sofa1', 'susp_inf', 'infection_icd', 'samp'],
    "生命体征 (vitals)": ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    "呼吸支持 (respiratory)": ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication', 'adv_resp'],
    "呼吸机参数 (ventilator)": ['peep', 'tidal_vol', 'tidal_vol_set', 'pip', 'plateau_pres', 'mean_airway_pres', 'minute_vol', 'vent_rate', 'etco2', 'compliance', 'driving_pres', 'ps'],
    "血气分析 (blood gas)": ['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2'],
    "实验室-生化 (chemistry)": ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri'],
    "实验室-血液学 (hematology)": ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt', 'ptt', 'rbc', 'rdw', 'wbc'],
    "血管活性药物 (vasopressors)": ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind', 'other_vaso'],
    "其他药物 (medications)": ['abx', 'cort', 'dex', 'ins'],
    # 🔧 2026-02-04: 移除重复的 kdigo_* 概念
    "肾脏与尿量 (renal)": ['urine', 'urine24', 'uo_6h', 'uo_12h', 'uo_24h', 'rrt', 'rrt_criteria', 'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt'],
    "神经系统 (neurological)": ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'tgcs', 'vgcs', 'sedated_gcs', 'motor_response', 'delirium_positive', 'delirium_tx'],
    "循环支持 (circulatory)": ['mech_circ_support', 'circ_failure', 'circ_event'],
    "人口统计 (demographics)": ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
    "SOFA-1 评分 (传统)": ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
    "其他评分 (scores)": ['qsofa', 'sirs', 'mews', 'news'],
    "结局 (outcome)": ['death', 'los_icu', 'los_hosp'],
}

# 🆕 特殊概念定义：这些概念不在 concept-dict.json 中，需要通过专用模块加载
# 格式: 概念名 -> (加载函数模块, 函数名, 输出列名列表)
SPECIAL_CONCEPTS = {
    # KDIGO AKI 相关概念 - 通过 kdigo_aki.py 加载
    'aki': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['aki']),
    'aki_stage': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['aki_stage']),
    'aki_stage_creat': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['aki_stage_creat']),
    'aki_stage_uo': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['aki_stage_uo']),
    'aki_stage_rrt': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['aki_stage_rrt']),
    # KDIGO AKI 输出列 - 也通过 kdigo_aki.py 加载
    'uo_rt_6hr': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['uo_rt_6hr']),
    'uo_rt_12hr': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['uo_rt_12hr']),
    'uo_rt_24hr': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['uo_rt_24hr']),
    'creat_low_past_48hr': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['creat_low_past_48hr']),
    'creat_low_past_7day': ('easyicu.kdigo_aki', 'load_kdigo_aki', ['creat_low_past_7day']),
    # Sepsis-3 诊断 - 通过专用函数加载
    'sep3_sofa1': ('easyicu.webapp.app', '_load_sep3_diagnosis', ['sep3_sofa1']),
    'sep3_sofa2': ('easyicu.webapp.app', '_load_sep3_diagnosis', ['sep3_sofa2']),
    # 循环衰竭相关概念 - 通过 circ_failure.py 加载
    'circ_failure': ('easyicu.circ_failure', 'load_circ_failure', ['circ_failure']),
    'circ_event': ('easyicu.circ_failure', 'load_circ_failure', ['circ_event']),
}

# 特殊概念的分组（同一模块的概念可以一起加载）
SPECIAL_CONCEPT_GROUPS = {
    'kdigo_aki': ['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
                  'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr', 'creat_low_past_48hr', 'creat_low_past_7day'],
    'sepsis3': ['sep3_sofa1', 'sep3_sofa2'],
    'circ_failure': ['circ_failure', 'circ_event'],
}


def _load_sep3_diagnosis(
    database: str,
    data_path: str = None,
    patient_ids: list = None,
    max_patients: int = None,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    计算 Sepsis-3 诊断 (sep3_sofa1 和 sep3_sofa2)。
    
    sep3_sofa1 = suspected infection + SOFA-1 ≥ 2
    sep3_sofa2 = suspected infection + SOFA-2 ≥ 2
    
    Returns:
        DataFrame with columns: [id_col, charttime, sep3_sofa1, sep3_sofa2]
    """
    from easyicu.api import load_concepts
    
    load_kwargs = {
        'data_path': data_path,
        'database': database,
        'verbose': verbose,
        'merge': True,
    }
    if max_patients:
        load_kwargs['max_patients'] = max_patients
    if patient_ids:
        load_kwargs['patient_ids'] = patient_ids
    
    # Load susp_inf + sofa + sofa2
    try:
        merged = load_concepts(concepts=['susp_inf', 'sofa', 'sofa2'], **load_kwargs)
    except Exception:
        try:
            merged = load_concepts(concepts=['susp_inf', 'sofa'], **load_kwargs)
        except Exception:
            return pd.DataFrame()
    
    if not isinstance(merged, pd.DataFrame) or merged.empty:
        return pd.DataFrame()
    
    # Detect ID and time columns
    id_col = None
    for c in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
        if c in merged.columns:
            id_col = c
            break
    time_col = None
    for c in ['charttime', 'time', 'starttime', 'datetime', 'Offset', 'measuredat_minutes', 'measuredat']:
        if c in merged.columns:
            time_col = c
            break
    
    if id_col is None or time_col is None:
        return pd.DataFrame()
    
    result_cols = [id_col, time_col]
    
    # sep3_sofa1: susp_inf == 1 AND sofa >= 2
    if 'susp_inf' in merged.columns and 'sofa' in merged.columns:
        susp = merged['susp_inf'].fillna(0).astype(bool)
        sofa_ok = merged['sofa'].fillna(0) >= 2
        merged['sep3_sofa1'] = (susp & sofa_ok).astype(int)
        result_cols.append('sep3_sofa1')
    
    # sep3_sofa2: susp_inf == 1 AND sofa2 >= 2
    if 'susp_inf' in merged.columns and 'sofa2' in merged.columns:
        susp = merged['susp_inf'].fillna(0).astype(bool)
        sofa2_ok = merged['sofa2'].fillna(0) >= 2
        merged['sep3_sofa2'] = (susp & sofa2_ok).astype(int)
        result_cols.append('sep3_sofa2')
    
    if len(result_cols) <= 2:
        return pd.DataFrame()
    
    # Only keep rows where susp_inf is present (infection window)
    if 'susp_inf' in merged.columns:
        mask = merged['susp_inf'].fillna(0).astype(bool)
        result = merged.loc[mask, result_cols].copy()
    else:
        result = merged[result_cols].copy()
    
    return result


# 本地特殊加载函数注册表（不需要通过 importlib 动态导入）
_LOCAL_SPECIAL_LOADERS = {
    '_load_sep3_diagnosis': _load_sep3_diagnosis,
}


def load_special_concepts(
    concepts: list,
    database: str,
    data_path: str,
    patient_ids: dict = None,
    max_patients: int = None,
    verbose: bool = False
) -> dict:
    """
    加载不在 concept-dict.json 中的特殊概念。
    
    这些概念需要通过专用模块（如 kdigo_aki.py, circ_failure.py）加载。
    
    Args:
        concepts: 要加载的概念列表
        database: 数据库名称 ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        data_path: 数据路径
        patient_ids: 患者ID过滤器 dict
        max_patients: 最大患者数
        verbose: 是否显示详细信息
        
    Returns:
        dict: {concept_name: DataFrame} 格式的结果
    """
    results = {}
    
    # 按特殊概念分组进行加载，避免重复调用
    loaded_groups = set()
    
    for concept in concepts:
        if concept not in SPECIAL_CONCEPTS:
            continue
            
        # 检查这个概念属于哪个分组
        for group_name, group_concepts in SPECIAL_CONCEPT_GROUPS.items():
            if concept in group_concepts and group_name not in loaded_groups:
                # 加载这个分组的数据
                try:
                    module_name, func_name, _ = SPECIAL_CONCEPTS[concept]
                    
                    # 🔧 先检查本地加载函数注册表
                    if func_name in _LOCAL_SPECIAL_LOADERS:
                        load_func = _LOCAL_SPECIAL_LOADERS[func_name]
                    else:
                        # 动态导入模块
                        import importlib
                        module = importlib.import_module(module_name)
                        load_func = getattr(module, func_name)
                    
                    # 准备加载参数
                    load_kwargs = {
                        'database': database,
                        'data_path': data_path,
                        'verbose': verbose,
                    }
                    if max_patients:
                        load_kwargs['max_patients'] = max_patients
                    if patient_ids:
                        # 提取患者ID列表
                        id_col = list(patient_ids.keys())[0] if patient_ids else None
                        if id_col:
                            load_kwargs['patient_ids'] = patient_ids[id_col]
                    
                    # 调用加载函数
                    df = load_func(**load_kwargs)
                    
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # 为这个分组中的每个概念创建结果
                        for gc in group_concepts:
                            if gc in concepts:
                                _, _, output_cols = SPECIAL_CONCEPTS[gc]
                                # 检查 DataFrame 中是否有对应的列
                                available_cols = [c for c in output_cols if c in df.columns]
                                if available_cols:
                                    results[gc] = df
                    
                    loaded_groups.add(group_name)
                    
                except Exception as e:
                    if verbose:
                        print(f"Failed to load special concept {concept}: {e}")
                    continue
                break
    
    return results


def _has_any_source_recursive(concept_name, database, concept_dict, visited=None):
    """递归检查概念或其子概念是否在目标数据库有数据源。"""
    if visited is None:
        visited = set()
    if concept_name in visited:
        return False
    visited.add(concept_name)

    concept_def = concept_dict.get(concept_name)
    if not concept_def:
        return False
    if concept_name in SPECIAL_CONCEPTS:
        return True
    if concept_def.sources.get(database):
        return True
    if concept_def.sub_concepts:
        return any(_has_any_source_recursive(sub_concept, database, concept_dict, visited) for sub_concept in concept_def.sub_concepts)
    return False


def load_preview_concepts(
    concepts: list,
    database: str,
    data_path: str,
    max_patients: int = 20,
    verbose: bool = False,
):
    """为 Preview Sample 加载概念，支持普通概念与特殊概念混合。"""
    from easyicu import load_concepts
    from easyicu.concept import load_dictionary

    concept_dict = load_dictionary(include_sofa2=True)
    normal_concepts = []
    special_concepts = []
    unsupported_concepts = []

    for concept in concepts or []:
        if concept in SPECIAL_CONCEPTS:
            special_concepts.append(concept)
            continue
        if concept in concept_dict and _has_any_source_recursive(concept, database, concept_dict):
            normal_concepts.append(concept)
        else:
            unsupported_concepts.append(concept)

    loaded = {}

    if normal_concepts:
        normal_result = load_concepts(
            normal_concepts,
            database=database,
            data_path=data_path,
            max_patients=max_patients,
            merge=False,
            verbose=verbose,
        )
        if isinstance(normal_result, dict):
            loaded.update({k: v.data if hasattr(v, 'data') else v for k, v in normal_result.items() if v is not None})
        elif hasattr(normal_result, 'columns'):
            for concept in normal_concepts:
                if concept in normal_result.columns:
                    loaded[concept] = normal_result

    if special_concepts:
        special_result = load_special_concepts(
            concepts=special_concepts,
            database=database,
            data_path=data_path,
            max_patients=max_patients,
            verbose=verbose,
        )
        for concept, df in (special_result or {}).items():
            if hasattr(df, 'data'):
                df = df.data
            if isinstance(df, pd.DataFrame) and not df.empty:
                loaded[concept] = df

    return {
        'loaded_concepts': loaded,
        'unsupported_concepts': unsupported_concepts,
        'requested_normal': normal_concepts,
        'requested_special': special_concepts,
    }


def render_data_dictionary():
    """Render data dictionary (aligned with sidebar groups)."""
    lang = st.session_state.get('language', 'en')
    
    # 双语标题
    title = "### 📖 Data Dictionary" if lang == 'en' else "### 📖 数据字典"
    st.markdown(title)
    
    caption = "Feature abbreviations, English names, Chinese meanings, and units (aligned with module categories)" if lang == 'en' else "每个特征的缩写、英文名称、中文含义及单位（与左侧模块分类一致）"
    st.caption(caption)
    
    # 🔍 搜索框
    search_placeholder = "Search by code, name or description... (e.g. hr, heart rate, lactate)" if lang == 'en' else "按代码、名称或描述搜索... (如 hr、heart rate、心率)"
    search_query = st.text_input(
        "🔍 Search" if lang == 'en' else "🔍 搜索",
        placeholder=search_placeholder,
        key="dict_page_search_input",
    )
    
    # 获取双语分组
    concept_groups = get_concept_groups()
    
    # 如果有搜索词，展示搜索结果
    if search_query and search_query.strip():
        query = search_query.strip().lower()
        matched_rows = []
        for cat_name, concepts in concept_groups.items():
            for concept in concepts:
                if concept in CONCEPT_DICTIONARY:
                    eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
                    eng_desc, chn_desc = CONCEPT_DESCRIPTIONS.get(concept, ('', ''))
                    if lang == 'en':
                        searchable = f"{concept} {eng_name} {eng_desc}".lower()
                    else:
                        searchable = f"{concept} {eng_name} {chn_name} {eng_desc} {chn_desc}".lower()
                    if query in searchable:
                        if lang == 'en':
                            matched_rows.append({
                                'Code': concept,
                                'Full Name': eng_name,
                                'Category': cat_name,
                                'Description': eng_desc if eng_desc else eng_name,
                                'Unit': unit if unit else '-'
                            })
                        else:
                            matched_rows.append({
                                '代码': concept,
                                '全称': eng_name,
                                '类别': cat_name,
                                '说明': chn_desc if chn_desc else chn_name,
                                '单位': unit if unit else '-'
                            })
        
        if matched_rows:
            n = len(matched_rows)
            result_text = f"Found **{n}** matching feature(s)" if lang == 'en' else f"找到 **{n}** 个匹配特征"
            st.success(result_text)
            st.dataframe(pd.DataFrame(matched_rows), width="stretch", hide_index=True, height=min(400, 50 + 35 * n))
        else:
            no_result = "No matching features found." if lang == 'en' else "未找到匹配的特征。"
            st.warning(no_result)
    else:
        # 无搜索词时，使用分类选择器
        all_label = "All" if lang == 'en' else "全部"
        select_label = "Select Category" if lang == 'en' else "选择类别查看"
        
        selected_category = st.selectbox(
            select_label,
            options=[all_label] + list(concept_groups.keys()),
            index=0,
            key="dict_category_select"
        )
        
        if selected_category == all_label:
            # 显示所有类别
            for cat_name, concepts in concept_groups.items():
                feat_label = "features" if lang == 'en' else "个特征"
                with st.expander(f"📁 {cat_name} ({len(concepts)} {feat_label})", expanded=False):
                    _render_category_table(concepts, lang)
        else:
            # 只显示选中的类别
            st.markdown(f"#### {selected_category}")
            _render_category_table(concept_groups[selected_category], lang)


def _render_category_table(concepts, lang='en'):
    """Render feature table for a single category with detailed descriptions."""
    rows = []
    for concept in concepts:
        if concept in CONCEPT_DICTIONARY:
            eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
            # 获取详细描述
            if concept in CONCEPT_DESCRIPTIONS:
                eng_desc, chn_desc = CONCEPT_DESCRIPTIONS[concept]
            else:
                eng_desc, chn_desc = '', ''
            
            if lang == 'en':
                rows.append({
                    'Abbr': concept,
                    'Full Name': eng_name,
                    'Description': eng_desc if eng_desc else chn_name,
                    'Unit': unit if unit else '-'
                })
            else:
                rows.append({
                    '缩写': concept,
                    '全名': eng_name,
                    '详细说明': chn_desc if chn_desc else chn_name,
                    '单位': unit if unit else '-'
                })
    
    if rows:
        df = pd.DataFrame(rows)
        if lang == 'en':
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    'Abbr': st.column_config.TextColumn('Abbr', width='small'),
                    'Full Name': st.column_config.TextColumn('Full Name', width='medium'),
                    'Description': st.column_config.TextColumn('Description', width='large'),
                    'Unit': st.column_config.TextColumn('Unit', width='small'),
                }
            )
        else:
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    '缩写': st.column_config.TextColumn('缩写', width='small'),
                    '全名': st.column_config.TextColumn('全名', width='medium'),
                    '详细说明': st.column_config.TextColumn('详细说明', width='large'),
                    '单位': st.column_config.TextColumn('单位', width='small'),
                }
            )


def check_data_status(data_path: str, database: str) -> dict:
    """检查数据目录的状态，返回文件统计信息。"""
    from pathlib import Path
    
    path = Path(data_path)
    result = {
        'ready': False,
        'parquet_count': 0,
        'csv_count': 0,
        'csv_files': [],
        'parquet_files': [],
        'missing_tables': [],
    }
    
    # 统计 parquet 文件（包括分片目录）
    parquet_files = list(path.glob('*.parquet'))
    # 检查分片目录（如 chartevents/1.parquet）
    for subdir in path.iterdir():
        if subdir.is_dir():
            shard_files = list(subdir.glob('[0-9]*.parquet'))
            if shard_files:
                result['parquet_count'] += 1
                result['parquet_files'].append(subdir.name)
    
    result['parquet_count'] += len(parquet_files)
    result['parquet_files'].extend([f.stem for f in parquet_files])
    
    # 统计 CSV 文件
    csv_files = list(path.glob('*.csv')) + list(path.glob('*.csv.gz'))
    result['csv_count'] = len(csv_files)
    result['csv_files'] = [f.name for f in csv_files]
    
    # 检查是否有足够的 parquet 文件（至少需要一些核心表）
    core_tables = {
        'miiv': ['icustays', 'patients', 'admissions'],
        'eicu': ['patient', 'apachepatientresult'],
        'aumc': ['admissions', 'drugitems'],
        'hirid': ['general_table', 'observations'],
    }
    
    required = core_tables.get(database, [])
    found = set(f.lower() for f in result['parquet_files'])
    
    # 如果有 parquet 文件，检查核心表是否存在
    if result['parquet_count'] > 0:
        missing = [t for t in required if t not in found]
        if len(missing) <= 1:  # 允许缺少1个核心表
            result['ready'] = True
        else:
            result['missing_tables'] = missing
    
    return result


def convert_data_with_progress(data_path: str, database: str):
    """带进度条的数据转换功能。"""
    import time
    
    lang = st.session_state.get('language', 'en')
    
    conv_title = "🔄 Data Conversion" if lang == 'en' else "🔄 数据转换"
    st.markdown(f"### {conv_title}")
    
    warn_msg = "⚠️ **Note**: Converting large datasets may take a long time (30min~2hrs), please be patient." if lang == 'en' else "⚠️ **注意**：转换大型数据集可能需要较长时间（30分钟~2小时），请耐心等待。"
    st.warning(warn_msg)
    
    info_msg = "💡 Using DuckDB for memory-efficient conversion. Large tables will be bucket-partitioned automatically." if lang == 'en' else "💡 使用 DuckDB 进行内存安全转换，大表将自动进行分桶优化。"
    st.info(info_msg)
    
    # 定义需要分桶转换的大表
    BUCKET_TABLES = {
        'miiv': {
            'chartevents': ('itemid', 100),
            'labevents': ('itemid', 100),
            'inputevents': ('itemid', 50),
        },
        'eicu': {
            'nursecharting': ('nursingchartcelltypevalname', 30),  # 按字符串hash
            'lab': ('labname', 50),
        },
        'aumc': {
            'numericitems': ('itemid', 100),
            'listitems': ('itemid', 50),
        },
        'hirid': {
            'observations': ('variableid', 100),
            'pharma': ('pharmaid', 50),
        },
        'mimic': {
            'chartevents': ('itemid', 100),
            'labevents': ('itemid', 100),
        },
        'sic': {
            'data_float_h': ('dataid', 50),
            'laboratory': ('laboratoryid', 50),
        },
    }
    
    try:
        from easyicu.duckdb_converter import DuckDBConverter
        from easyicu.bucket_converter import convert_to_buckets, BucketConfig
        import gc
        
        # 自动检测可用内存，预留 3GB 给 OS/Python
        from easyicu.memory_manager import get_available_memory_mb
        _avail_gb = get_available_memory_mb() / 1024
        _duckdb_mem_gb = max(2.0, _avail_gb - 3.0)
        
        converter = DuckDBConverter(
            data_path=data_path, 
            memory_limit_gb=_duckdb_mem_gb,
            verbose=True
        )
        
        # 获取需要转换的文件列表
        csv_files = converter._find_csv_files()
        total_files = len(csv_files)
        
        if total_files == 0:
            err_msg = "No CSV files found to convert" if lang == 'en' else "未找到需要转换的 CSV 文件"
            st.error(err_msg)
            return
        
        # 分类文件：大表用分桶，小表用普通转换
        bucket_tables_config = BUCKET_TABLES.get(database, {})
        bucket_files = []
        normal_files = []
        
        for csv_file in csv_files:
            stem = csv_file.stem.lower().replace('.csv', '')
            if stem in bucket_tables_config:
                bucket_files.append((csv_file, bucket_tables_config[stem]))
            else:
                normal_files.append(csv_file)
        
        detect_msg = f"📊 Detected **{len(normal_files)}** normal + **{len(bucket_files)}** large tables" if lang == 'en' else f"📊 共检测到 **{len(normal_files)}** 个普通表 + **{len(bucket_files)}** 个大表"
        st.markdown(detect_msg)
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_container = st.container()
        
        converted = 0
        skipped = 0
        failed = 0
        total = len(normal_files) + len(bucket_files)
        current = 0
        
        # 1. 先转换普通表
        for csv_file in normal_files:
            current += 1
            file_name = csv_file.name
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            
            processing_msg = f"**Processing**: `{file_name}` ({file_size_mb:.1f} MB) [{current}/{total}]" if lang == 'en' else f"**正在处理**: `{file_name}` ({file_size_mb:.1f} MB) [{current}/{total}]"
            status_text.markdown(processing_msg)
            
            parquet_path = converter._get_parquet_path(csv_file)
            if parquet_path.exists():
                skipped += 1
                with details_container:
                    st.caption(f"⏭️ {file_name} (exists)")
            else:
                try:
                    result = converter.convert_file(csv_file)
                    if result['status'] == 'success':
                        converted += 1
                        with details_container:
                            st.caption(f"✅ {file_name}: {result['row_count']:,} rows")
                    else:
                        failed += 1
                        with details_container:
                            st.caption(f"❌ {file_name}: {result.get('error', 'unknown')[:40]}")
                except Exception as e:
                    failed += 1
                    with details_container:
                        st.caption(f"❌ {file_name}: {str(e)[:40]}")
            
            progress_bar.progress(current / total)
            gc.collect()
        
        # 2. 分桶转换大表
        for csv_file, (partition_col, num_buckets) in bucket_files:
            current += 1
            file_name = csv_file.name
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            stem = csv_file.stem.lower().replace('.csv', '')
            
            processing_msg = f"**Bucketing**: `{file_name}` ({file_size_mb:.1f} MB) → {num_buckets} buckets [{current}/{total}]" if lang == 'en' else f"**分桶转换**: `{file_name}` ({file_size_mb:.1f} MB) → {num_buckets} 个桶 [{current}/{total}]"
            status_text.markdown(processing_msg)
            
            # 检查分桶目录是否已存在
            bucket_dir = csv_file.parent / f"{stem}_bucket"
            if bucket_dir.exists() and list(bucket_dir.glob('bucket_id=*')):
                skipped += 1
                with details_container:
                    st.caption(f"⏭️ {file_name} (bucket exists)")
            else:
                try:
                    # AUMC 使用 latin-1 编码（µmol 等特殊字符）
                    _encoding = 'latin-1' if database == 'aumc' else None
                    config = BucketConfig(
                        num_buckets=num_buckets,
                        partition_col=partition_col,
                        memory_limit='8GB',
                        encoding=_encoding,
                    )
                    result = convert_to_buckets(
                        source_path=csv_file,
                        output_dir=bucket_dir,
                        config=config,
                        overwrite=True
                    )
                    if result.success:
                        converted += 1
                        with details_container:
                            st.caption(f"✅ {file_name} → {result.num_buckets} buckets, {result.total_rows:,} rows")
                    else:
                        failed += 1
                        with details_container:
                            st.caption(f"❌ {file_name}: {result.error[:40] if result.error else 'unknown'}")
                except Exception as e:
                    failed += 1
                    with details_container:
                        st.caption(f"❌ {file_name}: {str(e)[:40]}")
            
            progress_bar.progress(current / total)
            gc.collect()
        
        # 转换完成
        progress_bar.progress(1.0)
        status_text.empty()
        
        if lang == 'en':
            summary = f"""
            ✅ **Conversion Complete!**
            - Successfully converted: {converted} files
            - Already existed/skipped: {skipped} files
            - Failed: {failed} files
            """
        else:
            summary = f"""
            ✅ **转换完成！**
            - 成功转换: {converted} 个文件
            - 已存在跳过: {skipped} 个文件
            - 转换失败: {failed} 个文件
            """
        st.success(summary)
        
        if failed == 0:
            st.balloons()
            all_done_msg = "🎉 All data converted successfully, you can now load the data!" if lang == 'en' else "🎉 所有数据已转换完成，现在可以加载数据了！"
            st.info(all_done_msg)
        else:
            partial_msg = "Some files failed to convert, but you can still try loading the converted data." if lang == 'en' else "部分文件转换失败，但您仍可以尝试加载已转换的数据。"
            st.warning(partial_msg)
            
    except ImportError as e:
        import_err = f"Data converter module not installed: {e}" if lang == 'en' else f"数据转换模块未安装: {e}"
        st.error(import_err)
    except Exception as e:
        conv_err = f"Conversion error: {str(e)}" if lang == 'en' else f"转换过程出错: {str(e)}"
        st.error(conv_err)


# ============ 🚀 智能硬件检测与动态并行配置 ============

def get_system_resources():
    """检测系统硬件资源。
    
    使用统一的 parallel_config 模块，确保代码端和 Web 端配置一致。
    
    Returns:
        dict: 包含 cpu_count, memory_gb, recommended_workers, recommended_backend
    """
    try:
        from ..parallel_config import get_global_config
        config = get_global_config()
        
        # 根据配置选择后端
        if config.cpu_count >= 16 and config.total_memory_gb >= 32:
            recommended_backend = "loky"
        else:
            recommended_backend = "thread"
        
        return {
            'cpu_count': config.cpu_count,
            'total_memory_gb': round(config.total_memory_gb, 1),
            'available_memory_gb': round(config.available_memory_gb, 1),
            'recommended_workers': config.max_workers,
            'recommended_backend': recommended_backend,
            'performance_tier': config.performance_tier,
            'buckets_per_batch': config.buckets_per_batch,
        }
    except ImportError:
        # Fallback: 直接检测（兼容旧版本）
        import os
        try:
            import psutil
            mem_info = psutil.virtual_memory()
            total_memory_gb = mem_info.total / (1024 ** 3)
            available_memory_gb = mem_info.available / (1024 ** 3)
        except:
            total_memory_gb = 8
            available_memory_gb = 4
        
        cpu_count = os.cpu_count() or 4
        max_workers_by_memory = int(available_memory_gb / 2)
        max_workers_by_cpu = int(cpu_count * 0.75)
        recommended_workers = min(max_workers_by_memory, max_workers_by_cpu, 64)
        recommended_workers = max(recommended_workers, 1)
        
        if cpu_count >= 16 and total_memory_gb >= 32:
            recommended_backend = "loky"
        else:
            recommended_backend = "thread"
        
        return {
            'cpu_count': cpu_count,
            'total_memory_gb': round(total_memory_gb, 1),
            'available_memory_gb': round(available_memory_gb, 1),
            'recommended_workers': recommended_workers,
            'recommended_backend': recommended_backend,
        }


def get_optimal_parallel_config(num_patients: int = None, task_type: str = 'load'):
    """根据系统资源和任务规模返回最优的并行配置。
    
    Args:
        num_patients: 要处理的患者数量，None 表示未知/全量
        task_type: 任务类型 ('load', 'export', 'preview')
    
    Returns:
        tuple: (parallel_workers, parallel_backend)
    """
    resources = get_system_resources()
    base_workers = resources['recommended_workers']
    backend = resources['recommended_backend']
    
    # 根据任务类型调整
    if task_type == 'preview':
        # 预览只需少量数据，不需要太多并行
        workers = min(base_workers, 4)
        backend = "thread"  # 预览用线程更快启动
    elif task_type == 'load':
        # 数据加载根据患者数量调整
        if num_patients is None or num_patients >= 50000:
            workers = base_workers  # 全量使用推荐配置
        elif num_patients >= 10000:
            workers = min(base_workers, max(8, base_workers // 2))
        elif num_patients >= 2000:
            workers = min(base_workers, 4)
        else:
            workers = 1  # 少量患者不需要并行
    elif task_type == 'export':
        # 🔧 FIX(2026-02-09): 导出任务也限制并行，避免 DuckDB 连接竞争和死锁
        # 之前使用 base_workers (64) 导致 SIC/MIMIC-III 加载卡住
        workers = min(base_workers, 4)
    else:
        workers = min(base_workers, 8)
    
    # Streamlit webapp 环境下，线程通常更安全
    # 🔧 FIX(2026-02-09): 所有任务都使用线程，避免 loky 多进程死锁
    if backend == "loky":
        backend = "thread"  # webapp 中统一使用线程
    
    return workers, backend


def init_session_state():
    """初始化 session state。"""
    # 🆕 入口模式：'none' (入口页), 'demo' (演示模式), 'real' (真实数据模式)
    if 'entry_mode' not in st.session_state:
        st.session_state.entry_mode = 'none'
    if 'data_path' not in st.session_state:
        st.session_state.data_path = None
    if 'database' not in st.session_state:
        st.session_state.database = 'miiv'
    if 'loaded_concepts' not in st.session_state:
        st.session_state.loaded_concepts = {}
    if 'loaded_data_origin' not in st.session_state:
        st.session_state.loaded_data_origin = 'none'
    if 'patient_ids' not in st.session_state:
        st.session_state.patient_ids = []
    if 'all_patient_count' not in st.session_state:
        st.session_state.all_patient_count = 0
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None
    if 'use_mock_data' not in st.session_state:
        st.session_state.use_mock_data = False
    if 'id_col' not in st.session_state:
        st.session_state.id_col = 'stay_id'
    # 新增：用于简化流程的状态
    if 'selected_concepts' not in st.session_state:
        st.session_state.selected_concepts = []
    if 'export_completed' not in st.session_state:
        st.session_state.export_completed = False
    if 'mock_params' not in st.session_state:
        st.session_state.mock_params = {'n_patients': 100, 'hours': 72}
    if 'trigger_export' not in st.session_state:
        st.session_state.trigger_export = False
    if 'export_format' not in st.session_state:
        st.session_state.export_format = 'Parquet'  # 默认Parquet
    if 'export_path' not in st.session_state:
        st.session_state.export_path = os.path.expanduser('~/easyicu_export')
    if 'path_validated' not in st.session_state:
        st.session_state.path_validated = False
    if 'language' not in st.session_state:
        st.session_state.language = 'en'  # 默认英文
    # 🚀 性能优化：患者数量限制
    # 全量 MIIV 约 5万患者/4000万行，加载需 ~50s；100患者约2s
    # 🔧 FIX 2025-01-28: 默认全量加载（0=不限制），满足大多数用户需求
    if 'patient_limit' not in st.session_state:
        st.session_state.patient_limit = 0  # 默认全量加载
    if 'available_patient_ids' not in st.session_state:
        st.session_state.available_patient_ids = None
    # 🆕 步骤确认状态
    if 'step1_confirmed' not in st.session_state:
        st.session_state.step1_confirmed = False
    if 'step2_confirmed' not in st.session_state:
        st.session_state.step2_confirmed = False
    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = False
    if 'sidebar_preview_enabled' not in st.session_state:
        st.session_state.sidebar_preview_enabled = False


# ============ 辅助函数：获取完整的 mock_params（包含最新的 cohort_filter） ============
def get_mock_params_with_cohort():
    """
    获取完整的 mock_params，包含最新的 cohort_filter。
    
    由于 Streamlit 的渲染顺序，Step 1 (数据源) 在 Step 2 (队列筛选) 之前执行，
    所以 mock_params 中的 cohort_filter 可能不是最新的。
    
    此函数确保在调用 generate_mock_data 时使用最新的 cohort_filter。
    """
    params = st.session_state.get('mock_params', {'n_patients': 100, 'hours': 72}).copy()
    
    # 如果启用了队列筛选，添加最新的 cohort_filter
    if st.session_state.get('cohort_enabled', False):
        cohort_filter = st.session_state.get('cohort_filter', None)
        if cohort_filter:
            params['cohort_filter'] = cohort_filter
    
    return params


# ============ 辅助函数：加载后按队列条件过滤已提取数据中的 None 值患者 ============

def _post_filter_cohort_data(data: dict, database: str) -> dict:
    """Remove patients from loaded concept data whose cohort-critical features are None.
    
    After load_concepts(), certain patients may have None for features like 'death'
    or 'los_icu' because the concept extraction pipeline differs from the cohort
    filter (e.g., multi-stay death attribution). This function removes such patients
    so the exported data is consistent with the cohort criteria.
    
    Args:
        data: dict of {concept_name: DataFrame} loaded by load_concepts
        database: Database name for ID column detection
    
    Returns:
        Filtered data dict with inconsistent patients removed
    """
    cf = st.session_state.get('cohort_filter', {})
    if not cf or not data:
        return data
    
    # Determine ID column
    id_col_map = {
        'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid',
        'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID',
    }
    id_col = id_col_map.get(database, 'stay_id')
    
    # Detect actual ID column from data
    id_candidates = [id_col, 'stay_id', 'icustay_id', 'hadm_id',
                     'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
    actual_id_col = None
    for df in data.values():
        if isinstance(df, pd.DataFrame):
            for c in id_candidates:
                if c in df.columns:
                    actual_id_col = c
                    break
            if actual_id_col:
                break
    if not actual_id_col:
        return data
    
    # Collect all patient IDs across all concepts
    all_patient_ids = set()
    for df in data.values():
        if isinstance(df, pd.DataFrame) and actual_id_col in df.columns:
            all_patient_ids.update(df[actual_id_col].dropna().unique())
    
    if not all_patient_ids:
        return data
    
    # Determine which patients to exclude based on cohort filter + loaded data
    exclude_ids = set()
    
    # 1. Survival filter: check death column value
    #    Mock data: death=0 (survived) or death=1 (deceased) for ALL patients
    #    Real data: death column may only exist for deceased patients (NaN = survived)
    if cf.get('survived') is not None and 'death' in data:
        death_df = data['death']
        if isinstance(death_df, pd.DataFrame) and actual_id_col in death_df.columns:
            # Get value column (last column or 'death')
            val_col = 'death' if 'death' in death_df.columns else death_df.columns[-1]
            # Convert death values to numeric for comparison
            death_valid = death_df[death_df[val_col].notna()].copy()
            death_vals = pd.to_numeric(death_valid[val_col], errors='coerce')
            
            # Patients who died (death value == 1 or True)
            died_ids = set(death_valid.loc[death_vals == 1, actual_id_col].unique())
            # Patients who survived (death value == 0 or False, or no death record)
            survived_ids = all_patient_ids - died_ids
            
            if not cf['survived']:
                # Deceased filter: keep only patients who died
                exclude_ids |= survived_ids
            else:
                # Survived filter: keep only patients who survived
                exclude_ids |= died_ids
    
    # 2. Min LOS filter: patients must have los >= threshold
    if cf.get('los_min') is not None and 'los_icu' in data:
        los_df = data['los_icu']
        if isinstance(los_df, pd.DataFrame) and actual_id_col in los_df.columns:
            val_col = 'los_icu' if 'los_icu' in los_df.columns else los_df.columns[-1]
            los_valid = los_df[los_df[val_col].notna()].copy()
            # LOS is in days, threshold is in hours
            los_hours = pd.to_numeric(los_valid[val_col], errors='coerce') * 24
            los_ok_ids = set(los_valid.loc[los_hours >= cf['los_min'], actual_id_col].unique())
            exclude_ids |= (all_patient_ids - los_ok_ids)
    
    # 3. Age filter: patients must have age within range
    if (cf.get('age_min') is not None or cf.get('age_max') is not None) and 'age' in data:
        age_df = data['age']
        if isinstance(age_df, pd.DataFrame) and actual_id_col in age_df.columns:
            val_col = 'age' if 'age' in age_df.columns else age_df.columns[-1]
            age_valid = age_df[age_df[val_col].notna()].copy()
            age_vals = pd.to_numeric(age_valid[val_col], errors='coerce')
            age_mask = pd.Series(True, index=age_valid.index)
            if cf.get('age_min') is not None:
                age_mask &= (age_vals >= cf['age_min'])
            if cf.get('age_max') is not None:
                age_mask &= (age_vals <= cf['age_max'])
            age_ok_ids = set(age_valid.loc[age_mask, actual_id_col].unique())
            exclude_ids |= (all_patient_ids - age_ok_ids)
    
    # 4. Gender filter: patients must have matching sex
    if cf.get('gender') is not None and 'sex' in data:
        sex_df = data['sex']
        if isinstance(sex_df, pd.DataFrame) and actual_id_col in sex_df.columns:
            val_col = 'sex' if 'sex' in sex_df.columns else sex_df.columns[-1]
            sex_valid = sex_df[sex_df[val_col].notna()].copy()
            sex_vals = sex_valid[val_col].astype(str).str.strip().str.upper()
            target = cf['gender'].upper()  # 'M' or 'F'
            # Match both short ('M','F') and long ('MALE','FEMALE') formats
            if target == 'M':
                target_variants = {'M', 'MALE', 'MAN', 'MÄNNLICH'}
            else:
                target_variants = {'F', 'FEMALE', 'WOMAN', 'WEIBLICH', 'VROUW', 'W'}
            sex_ok_ids = set(sex_valid.loc[sex_vals.isin(target_variants), actual_id_col].unique())
            exclude_ids |= (all_patient_ids - sex_ok_ids)
    
    if not exclude_ids:
        return data
    
    # Remove excluded patients from all concept DataFrames
    n_excluded = len(exclude_ids)
    n_total = len(all_patient_ids)
    n_remaining = n_total - n_excluded
    print(f"[COHORT POST-FILTER] Removing {n_excluded}/{n_total} patients with inconsistent cohort feature values")
    
    filtered_data = {}
    for concept, df in data.items():
        if isinstance(df, pd.DataFrame) and actual_id_col in df.columns:
            filtered_data[concept] = df[~df[actual_id_col].isin(exclude_ids)].copy()
        else:
            filtered_data[concept] = df
    
    # 🔧 Update _cohort_stats so displayed message matches actual patient count
    cohort_stats = st.session_state.get('_cohort_stats')
    if cohort_stats:
        cohort_stats['after'] = n_remaining
        cohort_stats['excluded'] = cohort_stats['before'] - n_remaining
        # Add post-filter detail
        lang = st.session_state.get('language', 'en')
        detail_label_en = f"Data consistency check: -{n_excluded}"
        detail_label_cn = f"数据一致性检查: -{n_excluded}"
        cohort_stats.setdefault('filter_details', []).append(
            (detail_label_en, detail_label_cn, n_excluded)
        )
        st.session_state['_cohort_stats'] = cohort_stats
    
    return filtered_data


# ============ 辅助函数：真正的 Cohort 筛选（读取 Parquet 元数据过滤患者） ============

def apply_cohort_filter(data_path, database, candidate_ids=None):
    """
    Apply cohort filters from st.session_state to real patient data.
    
    Reads ICU metadata tables (icustays, patients, admissions) and filters
    patient IDs based on the active cohort criteria (age, first_icu_stay,
    los_min, gender, survived).
    
    Args:
        data_path: Path to the database directory (e.g. /home/zhuhb/icudb/mimiciv/3.1)
        database: Database name ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        candidate_ids: Optional pre-filtered list of IDs to further filter
    
    Returns:
        dict with keys: id_col, filtered_ids, total_before, total_after, filter_details
        or None if cohort filtering is disabled / no active filter
    """
    # Check if filtering is enabled
    if not st.session_state.get('cohort_enabled', False):
        return None
    
    cf = st.session_state.get('cohort_filter', {})
    if not cf:
        return None
    
    # Check if any filter is actually active
    has_active = (
        cf.get('age_min') is not None or
        cf.get('age_max') is not None or
        cf.get('first_icu_stay') is not None or
        cf.get('los_min') is not None or
        cf.get('gender') is not None or
        cf.get('survived') is not None
    )
    if not has_active:
        return None
    
    data_path = Path(data_path)
    
    # Database-specific configuration
    DB_META = {
        'miiv': {
            'id_col': 'stay_id', 'subject_col': 'subject_id',
            'icu_table': 'icustays.parquet', 'patient_table': 'patients.parquet',
            'admission_table': 'admissions.parquet',
        },
        'eicu': {
            'id_col': 'patientunitstayid', 'subject_col': 'uniquepid',
            'icu_table': 'patient.parquet', 'patient_table': None,
            'admission_table': None,
        },
        'aumc': {
            'id_col': 'admissionid', 'subject_col': 'patientid',
            'icu_table': 'admissions.parquet', 'patient_table': None,
            'admission_table': None,
        },
        'hirid': {
            'id_col': 'patientid', 'subject_col': 'patientid',
            'icu_table': 'general.parquet', 'patient_table': None,
            'admission_table': None,
            'icu_table_fallback': 'general_table.csv',  # fallback if parquet missing
        },
        'mimic': {
            'id_col': 'icustay_id', 'subject_col': 'subject_id',
            'icu_table': 'icustays.parquet', 'patient_table': 'patients.parquet',
            'admission_table': 'admissions.parquet',
        },
        'sic': {
            'id_col': 'CaseID', 'subject_col': 'PatientID',
            'icu_table': 'cases.parquet', 'patient_table': None,
            'admission_table': None,
        },
    }
    
    meta = DB_META.get(database)
    if not meta:
        print(f"[COHORT] Unknown database: {database}")
        return None
    
    id_col = meta['id_col']
    subject_col = meta['subject_col']
    
    # Load ICU stays table
    icu_path = data_path / meta['icu_table']
    if not icu_path.exists():
        # Try fallback path (e.g., HiRID general_table.csv)
        fallback = meta.get('icu_table_fallback')
        if fallback:
            icu_path = data_path / fallback
        if not icu_path.exists():
            print(f"[COHORT] ICU table not found: {icu_path}")
            return None
    
    if str(icu_path).endswith('.csv') or str(icu_path).endswith('.csv.gz'):
        icu_df = pd.read_csv(icu_path)
    else:
        icu_df = pd.read_parquet(icu_path)
    
    # Normalize column names to lowercase for comparison (except for SICdb)
    if database != 'sic':
        icu_df.columns = [c.lower() for c in icu_df.columns]
        id_col_lower = id_col.lower()
        subject_col_lower = subject_col.lower()
    else:
        id_col_lower = id_col
        subject_col_lower = subject_col
    
    # Load optional tables
    patient_df = None
    admission_df = None
    if meta.get('patient_table'):
        pt_path = data_path / meta['patient_table']
        if pt_path.exists():
            patient_df = pd.read_parquet(pt_path)
            if database != 'sic':
                patient_df.columns = [c.lower() for c in patient_df.columns]
    if meta.get('admission_table'):
        adm_path = data_path / meta['admission_table']
        if adm_path.exists():
            admission_df = pd.read_parquet(adm_path)
            if database != 'sic':
                admission_df.columns = [c.lower() for c in admission_df.columns]
    
    # Start with all IDs
    if candidate_ids is not None:
        mask = icu_df[id_col_lower].isin(candidate_ids)
        icu_df = icu_df[mask].copy()
        icu_df = icu_df.reset_index(drop=True)  # reset index so merge-based filters align
    
    total_before = len(icu_df)
    keep_mask = pd.Series(True, index=icu_df.index)
    filter_details = []  # list of (label_en, label_cn, excluded_count)
    
    # ---------- Age Filter ----------
    if cf.get('age_min') is not None or cf.get('age_max') is not None:
        age_series = _get_age_series(icu_df, database, patient_df, admission_df,
                                     id_col_lower, subject_col_lower)
        if age_series is not None:
            before_count = keep_mask.sum()
            age_valid = age_series.notna()
            if cf.get('age_min') is not None:
                keep_mask &= age_valid & (age_series >= cf['age_min'])
            if cf.get('age_max') is not None:
                keep_mask &= age_valid & (age_series <= cf['age_max'])
            excluded = int(before_count - keep_mask.sum())
            age_range = f"{cf.get('age_min', 0)}-{cf.get('age_max', '∞')}"
            filter_details.append((f"Age {age_range}", f"年龄 {age_range}", excluded))
    
    # ---------- First ICU Stay Filter ----------
    if cf.get('first_icu_stay') is not None:
        first_mask = _get_first_icu_mask(icu_df, database, id_col_lower, subject_col_lower)
        if first_mask is not None:
            before_count = keep_mask.sum()
            first_valid = first_mask.notna()
            if cf['first_icu_stay']:
                keep_mask &= first_valid & first_mask.fillna(False).astype(bool)
            else:
                keep_mask &= first_valid & ~first_mask.fillna(True).astype(bool)
            excluded = int(before_count - keep_mask.sum())
            en_label = "First ICU stay only" if cf['first_icu_stay'] else "Non-first ICU stay only"
            cn_label = "仅首次ICU入住" if cf['first_icu_stay'] else "仅非首次ICU入住"
            filter_details.append((en_label, cn_label, excluded))
    
    # ---------- Min LOS Filter ----------
    if cf.get('los_min') is not None:
        los_series = _get_los_hours_series(icu_df, database)
        if los_series is not None:
            before_count = keep_mask.sum()
            keep_mask &= los_series.notna() & (los_series >= cf['los_min'])
            excluded = int(before_count - keep_mask.sum())
            filter_details.append((f"LOS ≥ {cf['los_min']}h", f"住院时长 ≥ {cf['los_min']}h", excluded))
    
    # ---------- Gender Filter ----------
    if cf.get('gender') is not None:
        sex_series = _get_sex_series(icu_df, database, patient_df,
                                     id_col_lower, subject_col_lower)
        if sex_series is not None:
            before_count = keep_mask.sum()
            keep_mask &= sex_series.notna() & (sex_series == cf['gender'])
            excluded = int(before_count - keep_mask.sum())
            gender_en = "Male" if cf['gender'] == 'M' else "Female"
            gender_cn = "男性" if cf['gender'] == 'M' else "女性"
            filter_details.append((f"{gender_en} only", f"仅{gender_cn}", excluded))
    
    # ---------- Survival Filter ----------
    if cf.get('survived') is not None:
        death_series = _get_death_series(icu_df, database, patient_df, admission_df,
                                         id_col_lower, subject_col_lower)
        if death_series is not None:
            before_count = keep_mask.sum()
            death_valid = death_series.notna()
            death_bool = pd.array(death_series.fillna(False), dtype=bool)
            if cf['survived']:
                keep_mask &= death_valid & ~death_bool  # survived = known not dead
            else:
                keep_mask &= death_valid & death_bool   # deceased = known dead
            excluded = int(before_count - keep_mask.sum())
            en_label = "Survived only" if cf['survived'] else "Deceased only"
            cn_label = "仅存活" if cf['survived'] else "仅死亡"
            filter_details.append((en_label, cn_label, excluded))
    
    filtered_ids = icu_df.loc[keep_mask, id_col_lower].unique().tolist()
    total_after = len(filtered_ids)
    
    pct = total_after / total_before * 100 if total_before > 0 else 0
    print(f"[COHORT] {database}: {total_before} → {total_after} patients ({pct:.1f}% retained)")
    
    return {
        'id_col': id_col,    # original case (e.g. CaseID)
        'filtered_ids': filtered_ids,
        'total_before': total_before,
        'total_after': total_after,
        'filter_details': filter_details,
    }


def _get_age_series(icu_df, database, patient_df, admission_df, id_col, subject_col):
    """Return a Series of ages aligned with icu_df index."""
    try:
        if database == 'miiv':
            # MIIV: anchor_age in patients + anchor_year; admittime in admissions
            if patient_df is not None and admission_df is not None:
                merged = icu_df[[id_col, 'hadm_id']].merge(
                    admission_df[['hadm_id', 'admittime']], on='hadm_id', how='left'
                )
                merged = merged.merge(
                    patient_df[['subject_id', 'anchor_age', 'anchor_year']],
                    left_on=icu_df[subject_col].values, right_on='subject_id', how='left'
                )
                admittime = pd.to_datetime(merged['admittime'])
                age = merged['anchor_age'] + (admittime.dt.year - merged['anchor_year'])
                return age.reindex(icu_df.index)
            return None
        
        elif database == 'eicu':
            # eICU: age column directly in patient table (ICU table)
            if 'age' in icu_df.columns:
                age = icu_df['age'].copy()
                # eICU stores "> 89" as string
                age = pd.to_numeric(age, errors='coerce')
                return age
            return None
        
        elif database == 'aumc':
            # AUMC: agegroup column (e.g. "18-39", "40-49", ...)
            if 'agegroup' in icu_df.columns:
                def parse_aumc_age(ag):
                    if pd.isna(ag):
                        return None
                    s = str(ag)
                    if '-' in s:
                        parts = s.split('-')
                        try:
                            return (int(parts[0]) + int(parts[1])) / 2
                        except ValueError:
                            return None
                    if s.startswith('80'):
                        return 85
                    try:
                        return float(s)
                    except ValueError:
                        return None
                return icu_df['agegroup'].map(parse_aumc_age)
            return None
        
        elif database == 'hirid':
            # HiRID: age column directly in general_table
            if 'age' in icu_df.columns:
                return pd.to_numeric(icu_df['age'], errors='coerce')
            return None
        
        elif database == 'mimic':
            # MIMIC-III: dob in patients, intime in icustays → age = intime.year - dob.year
            if patient_df is not None and 'dob' in patient_df.columns:
                merged = icu_df.merge(
                    patient_df[['subject_id', 'dob']], on='subject_id', how='left'
                )
                intime = pd.to_datetime(merged['intime'])
                dob = pd.to_datetime(merged['dob'])
                age = (intime - dob).dt.days / 365.25
                age = age.clip(upper=90)
                return age.reindex(icu_df.index)
            return None
        
        elif database == 'sic':
            # SICdb: AgeOnAdmission column
            age_col = None
            for c in icu_df.columns:
                if c.lower() == 'ageonadmission':
                    age_col = c
                    break
            if age_col:
                return pd.to_numeric(icu_df[age_col], errors='coerce')  # already in years
            return None
        
        return None
    except Exception as e:
        print(f"[COHORT] _get_age_series error ({database}): {e}")
        return None


def _get_first_icu_mask(icu_df, database, id_col, subject_col):
    """Return a boolean Series: True where the row is the patient's first ICU stay."""
    try:
        if database == 'miiv':
            # Earliest intime per subject_id
            if 'intime' in icu_df.columns:
                intime = pd.to_datetime(icu_df['intime'])
                first_intime = intime.groupby(icu_df[subject_col]).transform('min')
                return intime == first_intime
            return None
        
        elif database == 'eicu':
            # unitvisitnumber == 1
            if 'unitvisitnumber' in icu_df.columns:
                return icu_df['unitvisitnumber'] == 1
            return None
        
        elif database == 'aumc':
            # admissioncount == 1
            if 'admissioncount' in icu_df.columns:
                return icu_df['admissioncount'] == 1
            return None
        
        elif database == 'hirid':
            # HiRID: each patient has exactly one entry — all True
            return pd.Series(True, index=icu_df.index)
        
        elif database == 'mimic':
            # MIMIC-III: earliest intime per subject_id
            if 'intime' in icu_df.columns:
                intime = pd.to_datetime(icu_df['intime'])
                first_intime = intime.groupby(icu_df[subject_col]).transform('min')
                return intime == first_intime
            return None
        
        elif database == 'sic':
            # SICdb: OffsetAfterFirstAdmission == 0
            offset_col = None
            for c in icu_df.columns:
                if c.lower() == 'offsetafterfirstadmission':
                    offset_col = c
                    break
            if offset_col:
                return icu_df[offset_col] == 0
            return None
        
        return None
    except Exception as e:
        print(f"[COHORT] _get_first_icu_mask error ({database}): {e}")
        return None


def _get_los_hours_series(icu_df, database):
    """Return a Series of Length of Stay in hours."""
    try:
        if database == 'miiv':
            if 'los' in icu_df.columns:
                return pd.to_numeric(icu_df['los'], errors='coerce') * 24  # stored in days
            elif 'intime' in icu_df.columns and 'outtime' in icu_df.columns:
                dt = pd.to_datetime(icu_df['outtime']) - pd.to_datetime(icu_df['intime'])
                return dt.dt.total_seconds() / 3600
            return None
        
        elif database == 'eicu':
            # unitdischargeoffset is in minutes from admission
            if 'unitdischargeoffset' in icu_df.columns:
                return pd.to_numeric(icu_df['unitdischargeoffset'], errors='coerce') / 60
            return None
        
        elif database == 'aumc':
            if 'admittedat' in icu_df.columns and 'dischargedat' in icu_df.columns:
                # stored in milliseconds from some epoch
                admitted = pd.to_numeric(icu_df['admittedat'], errors='coerce')
                discharged = pd.to_numeric(icu_df['dischargedat'], errors='coerce')
                return (discharged - admitted) / 1000 / 3600  # ms -> hours
            return None
        
        elif database == 'hirid':
            # HiRID general_table doesn't have reliable LOS — return None to skip filter
            return None
        
        elif database == 'mimic':
            if 'los' in icu_df.columns:
                return pd.to_numeric(icu_df['los'], errors='coerce') * 24  # stored in days
            elif 'intime' in icu_df.columns and 'outtime' in icu_df.columns:
                dt = pd.to_datetime(icu_df['outtime']) - pd.to_datetime(icu_df['intime'])
                return dt.dt.total_seconds() / 3600
            return None
        
        elif database == 'sic':
            # SICdb: TimeOfStay in seconds
            tos_col = None
            for c in icu_df.columns:
                if c.lower() == 'timeofstay':
                    tos_col = c
                    break
            if tos_col:
                return pd.to_numeric(icu_df[tos_col], errors='coerce') / 3600  # seconds -> hours
            return None
        
        return None
    except Exception as e:
        print(f"[COHORT] _get_los_hours_series error ({database}): {e}")
        return None


def _get_sex_series(icu_df, database, patient_df, id_col, subject_col):
    """Return a Series of sex normalized to 'M'/'F'."""
    try:
        SEX_MAP_M = {'m', 'male', 'man', 'männlich', 'Man', 'Male'}
        SEX_MAP_F = {'f', 'female', 'woman', 'weiblich', 'Vrouw', 'Female'}
        
        def normalize_sex(s):
            if pd.isna(s):
                return None
            s_str = str(s).strip()
            if s_str.lower() in {x.lower() for x in SEX_MAP_M}:
                return 'M'
            if s_str.lower() in {x.lower() for x in SEX_MAP_F}:
                return 'F'
            return None
        
        if database == 'miiv':
            if patient_df is not None and 'gender' in patient_df.columns:
                merged = icu_df[[subject_col]].merge(
                    patient_df[[subject_col, 'gender']], on=subject_col, how='left'
                )
                return merged['gender'].map(normalize_sex).reindex(icu_df.index)
            return None
        
        elif database == 'eicu':
            if 'gender' in icu_df.columns:
                return icu_df['gender'].map(normalize_sex)
            return None
        
        elif database == 'aumc':
            if 'gender' in icu_df.columns:
                return icu_df['gender'].map(normalize_sex)
            return None
        
        elif database == 'hirid':
            if 'sex' in icu_df.columns:
                return icu_df['sex'].map(normalize_sex)
            return None
        
        elif database == 'mimic':
            if patient_df is not None and 'gender' in patient_df.columns:
                merged = icu_df[[subject_col]].merge(
                    patient_df[[subject_col, 'gender']], on=subject_col, how='left'
                )
                return merged['gender'].map(normalize_sex).reindex(icu_df.index)
            return None
        
        elif database == 'sic':
            sex_col = None
            for c in icu_df.columns:
                if c.lower() == 'sex':
                    sex_col = c
                    break
            if sex_col:
                def sic_sex(v):
                    if pd.isna(v):
                        return None
                    v_int = int(v) if isinstance(v, (int, float)) else None
                    # SICdb uses 735=Male, 736=Female
                    if v_int == 735 or v_int == 0 or str(v).lower() in {'m', 'male', '0'}:
                        return 'M'
                    if v_int == 736 or v_int == 1 or str(v).lower() in {'f', 'female', '1', 'w'}:
                        return 'F'
                    return normalize_sex(v)
                return icu_df[sex_col].map(sic_sex)
            return None
        
        return None
    except Exception as e:
        print(f"[COHORT] _get_sex_series error ({database}): {e}")
        return None


def _pick_death_stay(merged, dead_mask, id_col, deathtime_col, intime_col, outtime_col):
    """For multi-stay admissions, pick the ICU stay to which death should be attributed.
    
    The death concept assigns the death event (using deathtime as the index) to
    a specific ICU stay via a rolling join.  This helper replicates that logic:
      1. If deathtime falls within [intime, outtime] → that stay.
      2. Otherwise the last ICU stay whose intime ≤ deathtime.
      3. Fallback: the very last ICU stay in the admission.
    """
    dead_rows = merged[dead_mask].copy()
    if dead_rows.empty:
        return set()
    
    dt = pd.to_datetime(dead_rows[deathtime_col], errors='coerce')
    it = pd.to_datetime(dead_rows[intime_col], errors='coerce')
    ot = pd.to_datetime(dead_rows[outtime_col], errors='coerce')
    
    dead_rows = dead_rows.copy()
    dead_rows['_dt'] = dt
    dead_rows['_it'] = it
    dead_rows['_ot'] = ot
    dead_rows['_in_stay'] = (it <= dt) & (dt <= ot)
    
    result_ids = set()
    for hadm, grp in dead_rows.groupby('hadm_id'):
        if len(grp) == 1:
            result_ids.add(grp.iloc[0][id_col])
            continue
        # 1. deathtime within the ICU stay
        in_stay = grp[grp['_in_stay']]
        if len(in_stay) > 0:
            result_ids.add(in_stay.iloc[0][id_col])
            continue
        # 2. last stay whose intime ≤ deathtime
        before = grp[grp['_it'] <= grp['_dt']]
        if len(before) > 0:
            result_ids.add(before.sort_values('_it').iloc[-1][id_col])
            continue
        # 3. fallback: last ICU stay overall
        result_ids.add(grp.sort_values('_it').iloc[-1][id_col])
    return result_ids


def _get_death_series(icu_df, database, patient_df, admission_df, id_col, subject_col):
    """Return a boolean Series: True where patient died in hospital/ICU.
    
    IMPORTANT: This must match the EasyICU 'death' concept definition exactly,
    so that filtering for 'deceased' patients guarantees death=True in the output.
    
    Concept definitions (concept-dict.json):
      - miiv/mimic: admissions.hospital_expire_flag == 1, index_var=deathtime
      - eicu: patient.hospitaldischargestatus == 'Expired'
      - aumc: aumc_death callback → dateofdeath not null AND (dateofdeath - dischargedat) < 72h
      - hirid: hirid_death callback → discharge_status == 'dead' in general table
      - sic: no death concept defined
    """
    try:
        if database == 'miiv':
            # Concept: admissions table, hospital_expire_flag == 1, index_var = deathtime
            # Must have BOTH flag=1 AND non-null deathtime (concept needs timestamp)
            # For multi-stay admissions, death is only attributed to the ICU stay
            # where deathtime falls (matching the concept's rolling-join behavior).
            if admission_df is not None and 'hospital_expire_flag' in admission_df.columns:
                merge_cols = ['hadm_id', 'hospital_expire_flag']
                if 'deathtime' in admission_df.columns:
                    merge_cols.append('deathtime')
                merged = icu_df.merge(
                    admission_df[merge_cols].drop_duplicates('hadm_id'),
                    on='hadm_id', how='left'
                )
                dead_base = (merged['hospital_expire_flag'] == 1)
                if 'deathtime' in merged.columns:
                    dead_base = dead_base & merged['deathtime'].notna()
                # For multi-stay admissions, only attribute death to the correct stay
                if 'deathtime' in merged.columns and 'intime' in merged.columns:
                    dead_stay_ids = _pick_death_stay(merged, dead_base, id_col, 'deathtime', 'intime', 'outtime')
                    return merged[id_col].isin(dead_stay_ids).reindex(icu_df.index).fillna(False)
                return dead_base.fillna(False).reindex(icu_df.index)
            return None
        
        elif database == 'eicu':
            # Concept: patient.hospitaldischargestatus == 'Expired'
            # (NOT unitdischargestatus — concept uses hospitaldischargestatus)
            if 'hospitaldischargestatus' in icu_df.columns:
                return (icu_df['hospitaldischargestatus'].astype(str).str.strip() == 'Expired')
            # Fallback to unit status only if hospital status is missing
            if 'unitdischargestatus' in icu_df.columns:
                return icu_df['unitdischargestatus'].str.lower().str.contains('expire', na=False)
            return None
        
        elif database == 'aumc':
            # Concept: aumc_death callback → dateofdeath not null AND
            #   (dateofdeath - dischargedat) < 72 hours (in milliseconds)
            if 'dateofdeath' in icu_df.columns and 'dischargedat' in icu_df.columns:
                dateofdeath = pd.to_numeric(icu_df['dateofdeath'], errors='coerce')
                dischargedat = pd.to_numeric(icu_df['dischargedat'], errors='coerce')
                hours_72_ms = 72 * 3600 * 1000
                diff = dateofdeath - dischargedat
                return (dateofdeath.notna() & (diff < hours_72_ms)).fillna(False)
            # Fallback: dateofdeath not null
            if 'dateofdeath' in icu_df.columns:
                return icu_df['dateofdeath'].notna()
            if 'destination' in icu_df.columns:
                return icu_df['destination'].str.lower().str.contains('overleden', na=False)
            return None
        
        elif database == 'hirid':
            # Concept: hirid_death callback → discharge_status == 'dead' from general table
            if 'discharge_status' in icu_df.columns:
                ds = icu_df['discharge_status']
                if ds.dtype == object:
                    return ds.str.lower().str.strip() == 'dead'
                else:
                    return ds == 1
            return None
        
        elif database == 'mimic':
            # Concept: admissions.hospital_expire_flag == 1, index_var = deathtime
            # Same multi-stay logic as MIIV.
            if admission_df is not None and 'hospital_expire_flag' in admission_df.columns:
                if 'hadm_id' in icu_df.columns:
                    merge_cols = ['hadm_id', 'hospital_expire_flag']
                    if 'deathtime' in admission_df.columns:
                        merge_cols.append('deathtime')
                    merged = icu_df.merge(
                        admission_df[merge_cols].drop_duplicates('hadm_id'),
                        on='hadm_id', how='left'
                    )
                    dead_base = (merged['hospital_expire_flag'] == 1)
                    if 'deathtime' in merged.columns:
                        dead_base = dead_base & merged['deathtime'].notna()
                    if 'deathtime' in merged.columns and 'intime' in merged.columns:
                        dead_stay_ids = _pick_death_stay(merged, dead_base, id_col, 'deathtime', 'intime', 'outtime')
                        return merged[id_col].isin(dead_stay_ids).reindex(icu_df.index).fillna(False)
                    return dead_base.fillna(False).reindex(icu_df.index)
            # Alternative: dod in patients
            if patient_df is not None and 'dod' in patient_df.columns:
                merged = icu_df[[subject_col]].merge(
                    patient_df[[subject_col, 'dod']], on=subject_col, how='left'
                )
                return merged['dod'].notna().reindex(icu_df.index)
            return None
        
        elif database == 'sic':
            # No death concept defined in concept-dict.json
            # Use OffsetOfDeath > 0 as best available approximation
            death_col = None
            for c in icu_df.columns:
                if c.lower() == 'offsetofdeath':
                    death_col = c
                    break
            if death_col:
                return icu_df[death_col].notna() & (pd.to_numeric(icu_df[death_col], errors='coerce') > 0)
            return None
        
        return None
    except Exception as e:
        print(f"[COHORT] _get_death_series error ({database}): {e}")
        return None


# ============ 国际化文本 ============
TEXTS = {
    'en': {
        'app_title': '🏥 EasyICU Data Explorer',
        'app_subtitle': 'Local ICU Data Analytics Platform',
        'select_mode': '🎯 Select Mode',
        'mode_extract': '💾 Data Extraction (New Data)',
        'mode_viz': '📊 Quick Visualization (Existing Data)',
        'step1': 'Step 1: Data Source',
        'step2': 'Step 2: Cohort Selection',
        'step3': 'Step 3: Select Features',
        'step4': 'Step 4: Export Data',
        'demo_mode': '🎭 Demo Mode',
        'real_data': '📁 Real Data',
        'demo_mode_desc': 'System generates simulated ICU data',
        'select_database': 'Select Database',
        'data_path': 'Data Path',
        'validate_path': '✅ Validate Path',
        'path_valid': '✅ Path Valid',
        'path_invalid': '❌ Path Invalid',
        'feature_groups': 'Feature Groups',
        'export_path': 'Export Path',
        'export_format': 'Export Format',
        'export_data': '💾 Export Data',
        'quick_viz': '📈 Quick Visualization',
        'load_data': '🔍 Load Data',
        'loading': 'Loading...',
        'data_loaded': '✅ Data Loaded',
        'features_loaded': 'features loaded',
        'patients_loaded': 'patients loaded',
        'select_tables': 'Select Tables to Load',
        'found_files': 'Found {n} data files',
        'no_files': 'No data files found in this directory',
        'dir_not_exist': 'Directory does not exist',
        'data_dir': '📁 Data Directory',
        'file_list': '📋 File List',
        'loaded_data': '📊 Loaded Data',
        'view_features': 'View Feature List',
        'load_hint': '💡 Select a data directory and load data to start visualization',
        'home': '📚 Tutorial',
        'quick_visualization': '📊 Quick Visualization',
        'cohort_compare': '📊 Cohort Analysis',
        'sub_data_table': '📋 Data Tables',
        'sub_timeseries': '📈 Time Series',
        'sub_patient_view': '🏥 Patient View',
        'sub_data_quality': '📊 Data Quality',
        'ready': '🎉 Ready!',
        'ready_desc': 'Data loaded, you can start exploring.',
        'database': 'Database',
        'features': 'Features',
        'patients': 'Patients',
        'status': 'Status',
        'start_analysis': '🚀 Start Analysis',
        'select_tab': 'Select a tab above to explore data:',
        'data_summary': '📋 Data Summary',
        'n_patients': 'Number of Patients',
        'n_hours': 'Data Duration (hours)',
        'current_task': '📍 Current Task',
        'configure_source': 'Configure Data Source',
        'select_features': 'Select Features',
        'export_or_preview': 'Export Data or Load Preview',
        'data_dict': '📖 Data Dictionary',
        'view_desc': 'View Feature Descriptions',
        'preview_btn': '👁️ Preview Sample',
        'preview_patients': 'Preview Patients',
        'sanity_title': '📋 Export Summary',
        'sanity_patients': 'Patients',
        'sanity_features': 'Features',
        'sanity_modules': 'Modules',
        'sanity_format': 'Format',
        'sanity_path': 'Export Path',
        'sanity_missing_hotspots': 'High Missingness (>50%)',
        'sanity_unsupported': 'Unsupported in This DB',
        'sanity_confirm': '✅ Confirm & Export',
        'sanity_back': '↩️ Go Back & Modify',
        'review_tables': '📋 Review Tables',
        'review_trends': '📈 Review Trends',
        'review_patients': '🏥 Review Patients',
        'review_quality': '📊 Review Quality',
        'clinical_lanes': 'Clinical Lanes',
        'lane_vitals': '❤️ Vital Signs',
        'lane_labs': '🧪 Labs',
        'lane_interventions': '💉 Interventions',
        'lane_scores': '📊 Scores',
        'threshold_lines': 'Show Clinical Thresholds',
        'patient_summary': 'Patient Summary',
        'demographics_header': 'Demographics',
        'icu_los_label': 'ICU LOS',
        'mortality_label': 'Mortality',
        'key_supports': 'Key Supports',
        'score_summary': 'Score Timeline',
        'missing_cause_db': 'Not available in this database',
        'missing_cause_cohort': 'Not measured in this cohort',
        'missing_cause_sparse': 'Sparse clinical event',
        'missing_cause_normal': 'Within expected range',
        'coverage_badge_full': 'Fully Harmonized',
        'coverage_badge_caveat': 'Harmonized (caveats)',
        'coverage_badge_partial': 'Partially Available',
        'coverage_badge_dbspec': 'DB-Specific',
        'ai_why_unavailable': 'Why is this concept unavailable?',
        'ai_suggest_related': 'Suggest related concepts',
        'ai_explain_score': 'Explain this score',
        'ai_why_missing': 'Why is missingness high?',
        'ai_assistant': '🤖 AI Assistant',
    },
    'zh': {
        'app_title': '🏥 EasyICU 数据探索器',
        'app_subtitle': '本地 ICU 数据分析与可视化平台',
        'select_mode': '🎯 选择操作模式',
        'mode_extract': '💾 数据提取导出（新数据）',
        'mode_viz': '📊 快速可视化（已有数据）',
        'step1': '步骤1: 数据源',
        'step2': '步骤2: 队列筛选',
        'step3': '步骤3: 选择特征',
        'step4': '步骤4: 导出数据',
        'demo_mode': '🎭 演示模式',
        'real_data': '📁 真实数据',
        'demo_mode_desc': '系统生成模拟ICU数据供体验',
        'select_database': '选择数据库',
        'data_path': '数据路径',
        'validate_path': '✅ 验证路径',
        'path_valid': '✅ 路径有效',
        'path_invalid': '❌ 路径无效',
        'feature_groups': '特征分组',
        'export_path': '导出路径',
        'export_format': '导出格式',
        'export_data': '💾 导出数据',
        'quick_viz': '📈 快速可视化',
        'load_data': '🔍 加载数据',
        'loading': '加载中...',
        'data_loaded': '✅ 数据已加载',
        'features_loaded': '个特征已加载',
        'patients_loaded': '个患者已加载',
        'select_tables': '选择要加载的表格',
        'found_files': '发现 {n} 个数据文件',
        'no_files': '该目录下没有找到数据文件',
        'dir_not_exist': '目录不存在',
        'data_dir': '📁 数据目录',
        'file_list': '📋 文件列表',
        'loaded_data': '📊 已加载数据',
        'view_features': '查看特征列表',
        'load_hint': '💡 选择数据目录并加载数据后，即可在右侧进行可视化分析',
        'home': '📚 教程',
        'quick_visualization': '📊 快速可视化',
        'cohort_compare': '📊 队列分析',
        'sub_data_table': '📋 数据大表',
        'sub_timeseries': '📈 时序分析',
        'sub_patient_view': '🏥 患者视图',
        'sub_data_quality': '📊 数据质量',
        'ready': '🎉 准备就绪！',
        'ready_desc': '数据已加载，您可以开始探索分析了。',
        'database': '数据库',
        'features': '特征',
        'patients': '患者',
        'status': '状态',
        'start_analysis': '🚀 开始分析',
        'select_tab': '选择上方的标签页开始探索数据：',
        'data_summary': '📋 数据摘要',
        'n_patients': '患者数量',
        'n_hours': '数据时长(小时)',
        'current_task': '📍 当前任务',
        'configure_source': '配置数据源',
        'select_features': '选择特征',
        'export_or_preview': '导出数据或加载预览',
        'data_dict': '📖 数据字典',
        'view_desc': '查看特征说明',
        'preview_btn': '👁️ 预览样本',
        'preview_patients': '预览患者数',
        'sanity_title': '📋 导出摘要',
        'sanity_patients': '患者数',
        'sanity_features': '特征数',
        'sanity_modules': '模块数',
        'sanity_format': '格式',
        'sanity_path': '导出路径',
        'sanity_missing_hotspots': '高缺失率 (>50%)',
        'sanity_unsupported': '当前数据库不支持',
        'sanity_confirm': '✅ 确认导出',
        'sanity_back': '↩️ 返回修改',
        'review_tables': '📋 数据审查',
        'review_trends': '📈 趋势审查',
        'review_patients': '🏥 患者审查',
        'review_quality': '📊 质量审查',
        'clinical_lanes': '临床分道视图',
        'lane_vitals': '❤️ 生命体征',
        'lane_labs': '🧪 实验室',
        'lane_interventions': '💉 药物干预',
        'lane_scores': '📊 评分',
        'threshold_lines': '显示临床阈值线',
        'patient_summary': '患者摘要',
        'demographics_header': '人口统计',
        'icu_los_label': 'ICU 住院时长',
        'mortality_label': '死亡率',
        'key_supports': '关键支持治疗',
        'score_summary': '评分时间线',
        'missing_cause_db': '该数据库无此变量',
        'missing_cause_cohort': '该队列未测量',
        'missing_cause_sparse': '稀疏临床事件',
        'missing_cause_normal': '在预期范围内',
        'coverage_badge_full': '完全统一',
        'coverage_badge_caveat': '统一(有注意事项)',
        'coverage_badge_partial': '部分可用',
        'coverage_badge_dbspec': '数据库特有',
        'ai_why_unavailable': '为什么此概念不可用？',
        'ai_suggest_related': '推荐相关概念',
        'ai_explain_score': '解释此评分',
        'ai_why_missing': '为什么缺失率高？',
        'ai_assistant': '🤖 AI 助手',
    }
}

def get_text(key: str) -> str:
    """根据当前语言获取文本。"""
    lang = st.session_state.get('language', 'en')
    return TEXTS.get(lang, TEXTS['en']).get(key, key)


def strip_emoji(text: str) -> str:
    """移除字符串中的emoji字符，用于CSV导出等场景防止乱码。"""
    import re
    # 匹配更全面的emoji范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Misc symbols (includes 🧪 etc)
        "\U00002B50-\U00002B55"  # stars
        "\U0001F004-\U0001F0CF"  # mahjong
        "\U0000203C-\U00003299"  # misc symbols
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()


def safe_format_number(val, decimals: int = 0) -> str:
    """安全地格式化数值，处理非数值类型（如字符串、NaN等）。
    
    Args:
        val: 要格式化的值
        decimals: 小数位数
        
    Returns:
        格式化后的字符串
    """
    import numpy as np
    
    # 处理 None 和 NaN
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    
    # 如果是字符串类型，直接返回
    if isinstance(val, (str, np.str_)):
        return str(val)
    
    # 尝试数值格式化
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def validate_database_path(data_path: str, database: str) -> dict:
    """
    验证数据路径是否包含指定数据库所需的文件。
    严格检查每个模块所需的所有表。
    
    返回:
        dict: {'valid': bool, 'message': str, 'suggestion': str (可选)}
    """
    path = Path(data_path)
    lang = st.session_state.get('language', 'en')
    
    # 各数据库需要的核心表（Parquet格式）- 包括分片目录
    # 分为必需表和可选表
    required_parquet_tables = {
        'miiv': {
            'core': ['icustays', 'patients', 'admissions'],  # 核心ID表
            'clinical': ['chartevents', 'labevents', 'inputevents', 'outputevents'],  # 临床数据
            'medication': ['prescriptions', 'ingredientevents'],  # 药物数据
            'other': ['procedureevents', 'd_items', 'd_labitems'],  # 其他
        },
        'eicu': {
            'core': ['patient'],
            'clinical': ['vitalperiodic', 'lab', 'nursecharting'],
            'medication': ['infusiondrug', 'medication'],
        },
        'aumc': {
            'core': ['admissions'],
            'clinical': ['numericitems', 'listitems'],
            'medication': ['drugitems'],
        },
        'hirid': {
            'core': ['general_table'],
            'clinical': ['observations'],
            'medication': ['pharma_records'],
        },
        'mimic': {  # MIMIC-III
            'core': ['icustays', 'patients', 'admissions'],
            'clinical': ['chartevents', 'labevents', 'outputevents'],
            'medication': ['prescriptions', 'inputevents_cv', 'inputevents_mv'],
        },
        'sic': {  # SICdb
            'core': ['cases'],
            'clinical': ['data_float_h', 'laboratory'],
            'medication': ['medication'],
        },
    }
    
    # 各数据库需要的核心表（CSV/GZ格式 - 原始文件）
    required_csv_files = {
        'miiv': ['icustays.csv', 'chartevents.csv', 'labevents.csv', 'prescriptions.csv', 'inputevents.csv'],
        'eicu': ['patient.csv', 'vitalPeriodic.csv', 'lab.csv'],
        'aumc': ['admissions.csv', 'numericitems.csv', 'drugitems.csv'],
        'hirid': ['general_table.csv', 'pharma_records.csv'],
        'mimic': ['icustays.csv', 'chartevents.csv', 'labevents.csv', 'prescriptions.csv'],
        'sic': ['cases.csv', 'data_float_h.csv', 'laboratory.csv', 'medication.csv'],
    }
    
    db_name = {
        'miiv': 'MIMIC-IV', 'eicu': 'eICU-CRD',
        'aumc': 'AmsterdamUMCdb', 'hirid': 'HiRID',
        'mimic': 'MIMIC-III', 'sic': 'SICdb'
    }.get(database, database.upper())
    
    # 检查Parquet文件和分片目录
    parquet_files = list(path.rglob('*.parquet'))
    parquet_names = set(f.name.lower().replace('.parquet', '') for f in parquet_files)
    
    # 对于某些数据库（如 HiRID），某些核心表可能是 CSV 格式
    csv_files = list(path.glob('*.csv'))
    csv_names = set(f.name.lower().replace('.csv', '') for f in csv_files)
    
    # 检查分片目录（如 chartevents/1.parquet）
    parquet_dirs = set()
    for pf in parquet_files:
        try:
            if pf.parent != path:
                rel = pf.parent.relative_to(path)
                # 如果是 xxx/1.parquet 格式，记录 xxx
                if pf.stem.isdigit():
                    parquet_dirs.add(pf.parent.name.lower())
        except ValueError:
            pass
    
    # 检查分桶目录（如 chartevents_bucket/bucket_id=*/data.parquet）
    bucket_dirs = set()
    for subdir in path.iterdir():
        if subdir.is_dir() and subdir.name.endswith('_bucket'):
            # 检查是否有 parquet 文件
            bucket_parquets = list(subdir.rglob('*.parquet'))
            if bucket_parquets:
                # 去掉 _bucket 后缀得到表名
                table_name = subdir.name[:-7]  # remove '_bucket'
                bucket_dirs.add(table_name.lower())
    
    # 分别统计 Parquet 可用和仅 CSV 可用的表
    parquet_available = parquet_names | parquet_dirs | bucket_dirs
    
    # HiRID 特殊处理：别名映射
    if database == 'hirid':
        if 'pharma' in parquet_available:
            parquet_available.add('pharma_records')
        # general.parquet (R ricu convention) 或 general_table.parquet (直接转换)
        if 'general' in parquet_available or 'general_table' in parquet_available:
            parquet_available.add('general_table')
    if database == 'hirid' and 'pharma' in csv_names:
        csv_names.add('pharma_records')
    
    # 检查各类别的表
    db_tables = required_parquet_tables.get(database, {})
    found_tables = []       # Parquet 可用
    csv_only_tables = []    # 仅 CSV 可用（无 Parquet）
    missing_tables = []     # 完全缺失
    missing_by_category = {}
    
    for category, tables in db_tables.items():
        for table in tables:
            tl = table.lower()
            if tl in parquet_available:
                found_tables.append(table)
            elif tl in csv_names:
                csv_only_tables.append(table)
            else:
                missing_tables.append(table)
                if category not in missing_by_category:
                    missing_by_category[category] = []
                missing_by_category[category].append(table)
    
    total_required = sum(len(tables) for tables in db_tables.values())
    
    # 如果所有表都有 Parquet（最佳情况）
    if len(missing_tables) == 0 and len(csv_only_tables) == 0:
        bucket_info = f", {len(bucket_dirs)} bucketed" if bucket_dirs else ""
        msg = f'✅ {db_name}: All {total_required} required tables found ({len(parquet_files)} Parquet files{bucket_info})' if lang == 'en' else f'✅ {db_name}: 所有 {total_required} 个必需表已找到 ({len(parquet_files)} 个 Parquet 文件{bucket_info})'
        return {
            'valid': True,
            'message': msg
        }
    
    # 有表仅通过 CSV 找到（无 Parquet）→ 需要转换
    if len(missing_tables) == 0 and len(csv_only_tables) > 0:
        csv_list = ', '.join(csv_only_tables[:5])
        if len(csv_only_tables) > 5:
            csv_list += f' (+{len(csv_only_tables)-5} more)'
        if lang == 'en':
            msg = f'⚠️ {db_name}: {len(csv_only_tables)} tables only found as CSV (no Parquet): {csv_list}. Click Validate & Setup to auto-convert.'
            sug = '💡 Click "Validate & Setup" to auto-convert CSV files'
        else:
            msg = f'⚠️ {db_name}: {len(csv_only_tables)} 个表仅有 CSV 格式（无 Parquet）: {csv_list}。点击「验证并设置」自动转换。'
            sug = '💡 点击「验证并设置」自动转换 CSV 文件'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
        }
    
    # 核心表缺失是严重问题
    core_missing = missing_by_category.get('core', [])
    if core_missing:
        missing_str = ', '.join(core_missing)
        csv_hint = ""
        if csv_only_tables:
            csv_hint_tables = ', '.join(csv_only_tables[:3])
            csv_hint = f" ({csv_hint_tables} only as CSV)" if lang == 'en' else f"（{csv_hint_tables} 仅有CSV格式）"
        if lang == 'en':
            msg = f'❌ {db_name}: Missing core tables: {missing_str}{csv_hint}'
            sug = f'💡 Core tables are required. Please ensure data is properly converted.'
        else:
            msg = f'❌ {db_name}: 缺少核心表: {missing_str}{csv_hint}'
            sug = f'💡 核心表是必需的，请确保数据已正确转换。'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'missing_tables': missing_tables + csv_only_tables,
        }
    
    # 部分表缺失（非核心）或仅有 CSV
    all_need_convert = missing_tables + csv_only_tables
    if len(found_tables) > 0 or len(csv_only_tables) > 0:
        parts = []
        if missing_tables:
            parts.append(', '.join(missing_tables[:3]) + (' (missing)' if lang == 'en' else '（缺失）'))
        if csv_only_tables:
            parts.append(', '.join(csv_only_tables[:3]) + (' (CSV only)' if lang == 'en' else '（仅CSV）'))
        detail_str = '; '.join(parts)
        if lang == 'en':
            msg = f'⚠️ {db_name}: {len(found_tables)}/{total_required} tables ready (Parquet), need conversion: {detail_str}'
            sug = f'💡 Click "Validate & Setup" to auto-convert missing/CSV tables'
        else:
            msg = f'⚠️ {db_name}: {len(found_tables)}/{total_required} 个表就绪（Parquet），需要转换: {detail_str}'
            sug = f'💡 点击「验证并设置」自动转换缺失或CSV格式的表'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'missing_tables': all_need_convert,
        }
    
    # 检查是否存在 CSV 文件（可能需要转换）
    csv_files = list(path.rglob('*.csv')) + list(path.rglob('*.csv.gz'))
    csv_names = [f.name.lower().replace('.gz', '') for f in csv_files]
    
    required_csvs = required_csv_files.get(database, [])
    found_csvs = []
    for req in required_csvs:
        if req.lower() in csv_names:
            found_csvs.append(req)
    
    if len(found_csvs) >= len(required_csvs) // 2:
        # 找到 CSV 文件但没有 Parquet - 需要转换
        msg = f'⚠️ Found {db_name} raw CSV files ({len(csv_files)} files), need to convert to Parquet' if lang == 'en' else f'⚠️ 找到 {db_name} 原始 CSV 文件 ({len(csv_files)} 个)，需要转换为 Parquet 格式'
        sug = '💡 Click "Validate & Setup" to auto-convert' if lang == 'en' else '💡 点击「验证并设置」自动转换'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path)
        }
    
    # HiRID 特殊检测：可能只有 tar.gz 归档文件（尚未解压）
    if database == 'hirid':
        tar_gz_files = list(path.glob('*.tar.gz'))
        raw_stage = path / 'raw_stage'
        if raw_stage.exists():
            tar_gz_files.extend(raw_stage.glob('*.tar.gz'))
        hirid_archives = [f.name for f in tar_gz_files if any(
            kw in f.name.lower() for kw in ['observation', 'pharma', 'reference']
        )]
        if hirid_archives:
            archive_list = ', '.join(hirid_archives[:5])
            msg = (f'⚠️ {db_name}: Found archives ({archive_list}) that need extraction and conversion' if lang == 'en' else
                   f'⚠️ {db_name}: 发现归档文件 ({archive_list}) 需要解压和转换')
            sug = '💡 Click "Validate & Setup" to auto-extract and convert' if lang == 'en' else '💡 点击「验证并设置」自动解压和转换'
            return {
                'valid': False,
                'message': msg,
                'suggestion': sug,
                'can_convert': True,
                'csv_path': str(path)
            }
    
    # 检查是否是子目录结构
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    subdir_names = [d.name.lower() for d in subdirs]
    
    # 检查常见的子目录结构
    expected_subdirs = {
        'miiv': ['hosp', 'icu', 'ed'],
        'eicu': ['eicu-crd'],
        'aumc': ['amsterdamumc'],
        'hirid': ['hirid'],
    }
    
    for expected in expected_subdirs.get(database, []):
        if expected.lower() in subdir_names:
            # 找到预期子目录
            lang = st.session_state.get('language', 'en')
            msg = f'⚠️ Detected {db_name} directory structure, but data may be in subdirectory' if lang == 'en' else f'⚠️ 检测到 {db_name} 目录结构，但数据可能在子目录中'
            sug = f'💡 Try path: {path / expected}' if lang == 'en' else f'💡 请尝试路径: {path / expected}'
            return {
                'valid': False,
                'message': msg,
                'suggestion': sug
            }
    
    # 完全找不到相关文件
    lang = st.session_state.get('language', 'en')
    msg = f'❌ Required data files for {db_name} not found in this path' if lang == 'en' else f'❌ 在此路径下未找到 {db_name} 所需的数据文件'
    sug = '💡 Please verify: 1) Path is correct 2) Database type matches 3) Data is downloaded' if lang == 'en' else '💡 请确认: 1) 路径是否正确 2) 数据库类型是否匹配 3) 数据是否已下载'
    return {
        'valid': False,
        'message': msg,
        'suggestion': sug
    }


def _choose_directory_dialog(initial_dir: str = "") -> str | None:
    """Try to open a native folder picker for local desktop usage."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    start_dir = initial_dir or os.path.expanduser("~")
    try:
        start_dir = str(Path(start_dir).expanduser())
    except Exception:
        start_dir = os.path.expanduser("~")
    if not Path(start_dir).exists():
        start_dir = str(Path(start_dir).parent)

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(initialdir=start_dir)
        return selected or None
    except Exception:
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _closest_existing_dir(path_str: str, fallback: str = "") -> Path:
    candidate = (path_str or fallback or os.path.expanduser("~")).strip()
    try:
        path = Path(candidate).expanduser()
    except Exception:
        path = Path(os.path.expanduser("~"))
    if path.is_file():
        path = path.parent
    while not path.exists() and path != path.parent:
        path = path.parent
    if not path.exists():
        path = Path(os.path.expanduser("~"))
    return path


@st.dialog("Browse Server Folders / 浏览服务器目录", width="large")
def _render_directory_browser_dialog(
    *,
    input_key: str,
    button_key: str,
    value: str = "",
) -> None:
    lang = st.session_state.get("language", "en")
    browser_open_key = f"{button_key}_open"
    browser_cwd_key = f"{button_key}_cwd"
    browser_filter_key = f"{button_key}_filter"
    browser_show_hidden_key = f"{button_key}_show_hidden"
    browser_new_folder_key = f"{button_key}_new_folder"
    pending_input_key = f"{input_key}__pending_value"

    current_dir = _closest_existing_dir(
        st.session_state.get(browser_cwd_key, ""),
        st.session_state.get(input_key, value or ""),
    )
    st.session_state[browser_cwd_key] = str(current_dir)
    if browser_filter_key not in st.session_state:
        st.session_state[browser_filter_key] = ""
    if browser_show_hidden_key not in st.session_state:
        st.session_state[browser_show_hidden_key] = False
    if browser_new_folder_key not in st.session_state:
        st.session_state[browser_new_folder_key] = ""

    header_hint = (
        "Choose a folder on the server running EasyICU."
        if lang == "en" else
        "选择运行 EasyICU 的服务器上的目录。"
    )
    st.caption(header_hint)
    st.markdown(f'<div class="server-browser-path">{current_dir}</div>', unsafe_allow_html=True)

    nav_cols = st.columns([1, 1, 1.4, 1.4])
    with nav_cols[0]:
        up_label = "⬆ Up" if lang == "en" else "⬆ 上级"
        if st.button(up_label, key=f"{button_key}_dlg_up", use_container_width=True):
            st.session_state[browser_cwd_key] = str(current_dir.parent if current_dir != current_dir.parent else current_dir)
            st.rerun()
    with nav_cols[1]:
        home_label = "🏠 Home" if lang == "en" else "🏠 主目录"
        if st.button(home_label, key=f"{button_key}_dlg_home", use_container_width=True):
            st.session_state[browser_cwd_key] = str(Path.home())
            st.rerun()
    with nav_cols[2]:
        select_label = "✅ Use This Folder" if lang == "en" else "✅ 使用当前目录"
        if st.button(select_label, key=f"{button_key}_dlg_select", use_container_width=True):
            st.session_state[pending_input_key] = str(current_dir)
            st.session_state[browser_open_key] = False
            st.rerun()
    with nav_cols[3]:
        close_label = "✕ Close" if lang == "en" else "✕ 关闭"
        if st.button(close_label, key=f"{button_key}_dlg_close", use_container_width=True):
            st.session_state[browser_open_key] = False
            st.rerun()

    tools_col1, tools_col2 = st.columns([1.4, 2.2])
    with tools_col1:
        st.checkbox(
            "Show hidden folders" if lang == "en" else "显示隐藏目录",
            key=browser_show_hidden_key,
            help="Hidden folders starting with '.' are hidden by default." if lang == "en" else "默认隐藏以 . 开头的目录。",
        )
    with tools_col2:
        create_cols = st.columns([2.4, 1])
        with create_cols[0]:
            st.text_input(
                "New folder name" if lang == "en" else "新建文件夹名称",
                key=browser_new_folder_key,
                placeholder="e.g. exports_20260415" if lang == "en" else "例如 exports_20260415",
                label_visibility="collapsed",
            )
        with create_cols[1]:
            create_label = "📁 Create" if lang == "en" else "📁 创建"
            if st.button(create_label, key=f"{button_key}_dlg_create", use_container_width=True):
                new_folder_name = str(st.session_state.get(browser_new_folder_key, "")).strip()
                if not new_folder_name:
                    st.warning("Please enter a folder name first." if lang == "en" else "请先输入文件夹名称。")
                elif any(sep in new_folder_name for sep in ('/', '\\')) or new_folder_name in {'.', '..'}:
                    st.warning("Folder name cannot contain path separators." if lang == "en" else "文件夹名称不能包含路径分隔符。")
                else:
                    try:
                        target_dir = current_dir / new_folder_name
                        target_dir.mkdir(parents=False, exist_ok=False)
                        st.session_state[browser_cwd_key] = str(target_dir)
                        st.session_state[pending_input_key] = str(target_dir)
                        st.session_state[browser_new_folder_key] = ""
                        st.success(f"Created folder: {new_folder_name}" if lang == "en" else f"已创建文件夹：{new_folder_name}")
                        st.rerun()
                    except FileExistsError:
                        st.warning(f"Folder already exists: {new_folder_name}" if lang == "en" else f"文件夹已存在：{new_folder_name}")
                    except Exception as exc:
                        st.error(f"Create folder failed: {exc}" if lang == "en" else f"创建文件夹失败：{exc}")

    st.text_input(
        "Directory Filter" if lang == "en" else "目录筛选",
        key=browser_filter_key,
        placeholder="Filter subfolders..." if lang == "en" else "筛选子目录...",
    )
    dir_filter = st.session_state.get(browser_filter_key, "").strip().lower()
    show_hidden = bool(st.session_state.get(browser_show_hidden_key, False))

    try:
        subdirs = [p for p in sorted(current_dir.iterdir(), key=lambda p: p.name.lower()) if p.is_dir()]
    except Exception as exc:
        st.error(f"Browse error: {exc}")
        subdirs = []

    if not show_hidden:
        subdirs = [p for p in subdirs if not p.name.startswith('.')]

    if dir_filter:
        subdirs = [p for p in subdirs if dir_filter in p.name.lower()]

    browser_list = st.container(height=460, border=True)
    with browser_list:
        if not subdirs:
            empty_msg = "No subdirectories found" if lang == "en" else "没有可用子目录"
            st.caption(empty_msg)
        else:
            shown_subdirs = subdirs[:120]
            for subdir in shown_subdirs:
                if st.button(
                    f"📁 {subdir.name}",
                    key=f"{button_key}_dlg_dir_{hash(str(subdir))}",
                    use_container_width=True,
                ):
                    st.session_state[browser_cwd_key] = str(subdir)
                    st.rerun()
            if len(subdirs) > len(shown_subdirs):
                more_msg = (
                    f"Showing first {len(shown_subdirs)} directories. Narrow with the filter above."
                    if lang == "en" else
                    f"当前仅显示前 {len(shown_subdirs)} 个目录。可用上方筛选缩小范围。"
                )
                st.caption(more_msg)


def _directory_input(
    label: str,
    *,
    input_key: str,
    button_key: str,
    value: str = "",
    placeholder: str = "",
    help: str | None = None,
    label_visibility: str = "visible",
) -> str:
    """Text input with a modal server-side directory browser."""
    lang = st.session_state.get("language", "en")
    browse_label = "📂"
    pending_input_key = f"{input_key}__pending_value"

    if pending_input_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(pending_input_key)
    elif input_key not in st.session_state:
        st.session_state[input_key] = value or ""

    browser_open_key = f"{button_key}_open"
    browser_cwd_key = f"{button_key}_cwd"
    browser_filter_key = f"{button_key}_filter"

    col_input, col_button = st.columns([8, 1.2])
    with col_input:
        typed_value = st.text_input(
            label,
            value=st.session_state.get(input_key, value or ""),
            key=input_key,
            placeholder=placeholder,
            help=help,
            label_visibility=label_visibility,
        ).strip()
    with col_button:
        if label_visibility == "visible":
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        browse_help = (
            "Browse directories on the server running EasyICU"
            if lang == "en" else
            "浏览运行 EasyICU 的服务器目录"
        )
        if st.button(browse_label, key=button_key, use_container_width=True, help=browse_help):
            st.session_state[browser_open_key] = True
            st.session_state[browser_cwd_key] = str(_closest_existing_dir(typed_value, value))
            st.rerun()

    if st.session_state.get(browser_open_key, False):
        _render_directory_browser_dialog(input_key=input_key, button_key=button_key, value=value)

    return typed_value


def generate_mock_data(n_patients=10, hours=72, cohort_filter=None):
    """生成模拟 ICU 数据用于演示。
    
    Args:
        n_patients: 要生成的患者数量
        hours: 数据时长（小时）
        cohort_filter: 队列过滤器字典，支持以下字段：
            - age_min/age_max: 年龄范围
            - gender: 'M' 或 'F'
            - survived: True/False
            - has_sepsis: True/False
            - los_min: 最短住院时长（小时）
    """
    data = {}
    
    # 🔧 如果有过滤器，根据过滤条件计算需要的初始患者数
    # 性别过滤约50%通过，存活过滤约85%通过，sepsis过滤约30%/70%通过
    initial_multiplier = 1
    if cohort_filter:
        # 估算每个过滤器的通过率
        if cohort_filter.get('gender') is not None:
            initial_multiplier *= 2.5  # 性别过滤约50%通过
        if cohort_filter.get('survived') is not None:
            if cohort_filter['survived']:
                initial_multiplier *= 1.3  # 存活约85%
            else:
                initial_multiplier *= 8  # 死亡约15%
        if cohort_filter.get('has_sepsis') is not None:
            if cohort_filter['has_sepsis']:
                initial_multiplier *= 4  # sepsis约30%
            else:
                initial_multiplier *= 1.5  # 非sepsis约70%
        if cohort_filter.get('age_min') is not None or cohort_filter.get('age_max') is not None:
            initial_multiplier *= 1.5  # 年龄范围过滤
        initial_multiplier = max(3, int(initial_multiplier))  # 最少3倍
    
    initial_n = n_patients * initial_multiplier
    all_patient_ids = list(range(10001, 10001 + initial_n))
    
    np.random.seed(42)
    time_points = np.arange(0, hours, 1)
    
    # 🔧 FIX (2026-02-03): 添加患者级随机采样时间生成函数
    def get_random_sample_times(pid, base_interval, jitter=0.3, min_samples=3):
        """为每个患者生成随机采样时间点
        
        Args:
            pid: 患者ID，用作随机种子的一部分
            base_interval: 基础采样间隔（小时）
            jitter: 间隔的随机抖动比例 (0-1)
            min_samples: 最少采样次数
        
        Returns:
            该患者的随机采样时间点列表
        """
        rng = np.random.RandomState(pid * 17 + 31)  # 每个患者有独立的随机状态
        sample_times = [0]  # 从0开始
        current_time = 0
        
        while current_time < hours - base_interval:
            # 在基础间隔上添加随机抖动
            interval = base_interval * (1 + rng.uniform(-jitter, jitter))
            interval = max(1, interval)  # 至少1小时间隔
            current_time += interval
            if current_time < hours:
                sample_times.append(int(current_time))
        
        # 确保至少有最少采样次数
        if len(sample_times) < min_samples:
            sample_times = list(np.linspace(0, hours-1, min_samples, dtype=int))
        
        return sample_times

    # 1. 预先生成患者元数据（用于后续过滤）
    patient_meta = {}
    for pid in all_patient_ids:
        # 年龄 (40-85岁)
        age = np.random.uniform(40, 85)
        # 性别
        sex = np.random.choice(['M', 'F'])
        # 死亡率 15%
        death = 1 if np.random.random() < 0.15 else 0
        # ICU住院时长 (1-14天转换为小时)
        los_icu = np.random.uniform(24, 14*24)  # 改为小时
        # 30% 概率患 sepsis
        is_septic = np.random.random() < 0.3
        # 发病时间随机分布在 10h ~ hours-10h 之间
        onset = np.random.choice(range(10, max(11, hours-10))) if is_septic else -999
        
        # 确定感染窗口 (samp time)
        samp_time = -1
        if is_septic:
            # 采样时间通常在发病前后
            samp_time = onset + np.random.randint(-4, 4)
            samp_time = max(0, min(hours-1, samp_time))
            
        patient_meta[pid] = {
            'age': age,
            'sex': sex,
            'death': death,
            'los_icu': los_icu,
            'is_septic': is_septic,
            'onset': onset,
            'samp_time': samp_time
        }
    
    # 2. 应用队列过滤器
    filtered_patient_ids = all_patient_ids
    if cohort_filter:
        filtered_patient_ids = []
        for pid in all_patient_ids:
            meta = patient_meta[pid]
            include = True
            
            # 年龄过滤
            if cohort_filter.get('age_min') is not None:
                if meta['age'] < cohort_filter['age_min']:
                    include = False
            if cohort_filter.get('age_max') is not None:
                if meta['age'] > cohort_filter['age_max']:
                    include = False
            
            # 性别过滤
            if cohort_filter.get('gender') is not None:
                if meta['sex'] != cohort_filter['gender']:
                    include = False
            
            # 存活状态过滤
            if cohort_filter.get('survived') is not None:
                if cohort_filter['survived'] and meta['death'] == 1:
                    include = False
                elif not cohort_filter['survived'] and meta['death'] == 0:
                    include = False
            
            # Sepsis过滤
            if cohort_filter.get('has_sepsis') is not None:
                if cohort_filter['has_sepsis'] and not meta['is_septic']:
                    include = False
                elif not cohort_filter['has_sepsis'] and meta['is_septic']:
                    include = False
            
            # 住院时长过滤
            if cohort_filter.get('los_min') is not None:
                if meta['los_icu'] < cohort_filter['los_min']:
                    include = False
            
            if include:
                filtered_patient_ids.append(pid)
        
        # 如果过滤后患者不够，发出警告但仍继续
        if len(filtered_patient_ids) < n_patients:
            print(f"Warning: Only {len(filtered_patient_ids)} patients match cohort criteria (requested {n_patients})")
    
    # 3. 限制到请求的患者数量
    patient_ids = filtered_patient_ids[:n_patients]
    
    # 为了兼容后续代码，创建 patient_sepsis_meta
    patient_sepsis_meta = {pid: patient_meta[pid] for pid in patient_ids}
    
    # 心率（使用患者级随机采样，模拟10%缺失率）
    hr_records = []
    for pid in patient_ids:
        base_hr = np.random.uniform(70, 90)
        # 如果 septic, 心率在发病后升高
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间，而非固定间隔
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            # 10%概率缺失（在随机采样基础上再添加随机缺失）
            if np.random.random() < 0.9:
                hr = base_hr + np.sin(t / 6) * 10 + np.random.normal(0, 5)
                if meta['is_septic'] and t >= meta['onset']:
                    hr += 20 # 发病后心率增加
                    
                hr_records.append({'stay_id': pid, 'time': t, 'hr': max(40, min(150, hr))})
    data['hr'] = pd.DataFrame(hr_records)
    
# MAP（使用患者级随机采样，模拟10%缺失率）
    map_records = []
    for pid in patient_ids:
        base_map = np.random.uniform(65, 85)
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if np.random.random() < 0.9:
                map_val = base_map + np.cos(t / 8) * 8 + np.random.normal(0, 4)
                if meta['is_septic'] and t >= meta['onset']:
                    map_val -= 15 # 发病后血压下降
                    
                map_records.append({'stay_id': pid, 'time': t, 'map': max(40, min(120, map_val))})
    data['map'] = pd.DataFrame(map_records)

    # SBP（使用患者级随机采样，模拟10%缺失率）
    sbp_records = []
    for pid in patient_ids:
        base_sbp = np.random.uniform(110, 140)
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if np.random.random() < 0.9:
                sbp_val = base_sbp + np.sin(t / 5) * 15 + np.random.normal(0, 8)
                if meta['is_septic'] and t >= meta['onset']:
                    sbp_val -= 20
                    
                sbp_records.append({'stay_id': pid, 'time': t, 'sbp': max(70, min(200, sbp_val))})
    data['sbp'] = pd.DataFrame(sbp_records)
    
    # 体温（使用患者级随机采样，约每4小时）
    temp_records = []
    for pid in patient_ids:
        base_temp = np.random.uniform(36.5, 37.5)
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            temp_val = base_temp + np.random.normal(0, 0.3)
            # 随机发热
            if np.random.random() < 0.1:
                temp_val += 1.5
            # Sepsis 发热
            if meta['is_septic'] and t >= meta['onset']:
                 temp_val += 1.2
                 
            temp_records.append({'stay_id': pid, 'time': t, 'temp': max(35, min(41, temp_val))})
    data['temp'] = pd.DataFrame(temp_records)
    
    # 呼吸（使用患者级随机采样，模拟15%缺失率）
    resp_records = []
    for pid in patient_ids:
        base_resp = np.random.uniform(14, 18)
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            # 15%概率缺失
            if np.random.random() < 0.85:
                resp_val = base_resp + np.random.normal(0, 2)
                if meta['is_septic'] and t >= meta['onset']:
                    resp_val += 8
                    
                resp_records.append({'stay_id': pid, 'time': t, 'resp': max(8, min(40, resp_val))})
    data['resp'] = pd.DataFrame(resp_records)
    
    # SpO2（使用患者级随机采样，模拟10%缺失率）
    spo2_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if np.random.random() < 0.9:
                spo2_val = 97 + np.random.normal(0, 2)
                if np.random.random() < 0.05:
                    spo2_val -= 10
                spo2_records.append({'stay_id': pid, 'time': t, 'spo2': max(80, min(100, spo2_val))})
    data['spo2'] = pd.DataFrame(spo2_records)
    
    # EtCO2 (End-Tidal CO2，模拟30%缺失率 - 需要特殊监测）
    etco2_records = []
    for pid in patient_ids:
        base_etco2 = np.random.uniform(35, 42)
        # 仅40%患者有EtCO2监测
        if np.random.random() < 0.4:
            for t in time_points:
                if np.random.random() < 0.7:
                    etco2_val = base_etco2 + np.random.normal(0, 3)
                    etco2_records.append({'stay_id': pid, 'time': t, 'etco2': max(20, min(60, etco2_val))})
    data['etco2'] = pd.DataFrame(etco2_records) if etco2_records else pd.DataFrame(columns=['stay_id', 'time', 'etco2'])
    
    # O2Sat (Oxygen Saturation - alias for spo2)
    data['o2sat'] = data['spo2'].rename(columns={'spo2': 'o2sat'}).copy() if not data['spo2'].empty else pd.DataFrame(columns=['stay_id', 'time', 'o2sat'])
    data['sao2'] = data['spo2'].rename(columns={'spo2': 'sao2'}).copy() if not data['spo2'].empty else pd.DataFrame(columns=['stay_id', 'time', 'sao2'])
    
    # SOFA（使用患者级随机采样，约每6小时）
    sofa_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.3)
        for t in sample_times:
            # 基础分布
            probs = [0.6, 0.3, 0.1, 0.0, 0.0] 
            
            # 如果是 sepsis 患者且处于发病期，概率向高分偏移
            if meta['is_septic'] and t >= meta['onset']:
                probs = [0.1, 0.2, 0.3, 0.25, 0.15]
                
            sofa_resp = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_coag = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_liver = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_cardio = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_cns = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_renal = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_cns + sofa_renal
            
            sofa_records.append({
                'stay_id': pid, 'time': t, 'sofa': sofa_total,
                'sofa_resp': sofa_resp, 'sofa_coag': sofa_coag, 'sofa_liver': sofa_liver,
                'sofa_cardio': sofa_cardio, 'sofa_cns': sofa_cns, 'sofa_renal': sofa_renal,
            })
    data['sofa'] = pd.DataFrame(sofa_records)
    # 🔧 FIX: 从父 DF 中提取子组件为独立 DF，然后剥离父 DF 中的子组件列
    # 避免合并时产生 _x/_y 后缀冲突
    _sofa_sub_cols = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']
    for _sc in _sofa_sub_cols:
        data[_sc] = data['sofa'][['stay_id', 'time', _sc]].copy()
    data['sofa'] = data['sofa'][['stay_id', 'time', 'sofa']].copy()
    
    # 肌酐（使用患者级随机采样，约每8小时）
    crea_records = []
    for pid in patient_ids:
        base_crea = np.random.uniform(0.8, 1.2)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=8, jitter=0.3)
        for t in sample_times:
            crea_val = base_crea + np.random.normal(0, 0.2)
            crea_records.append({'stay_id': pid, 'time': t, 'crea': max(0.3, crea_val)})
    data['crea'] = pd.DataFrame(crea_records)
    
    # 胆红素（使用患者级随机采样，约每12小时）
    bili_records = []
    for pid in patient_ids:
        base_bili = np.random.uniform(0.5, 1.5)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.3)
        for t in sample_times:
            bili_val = base_bili + np.random.normal(0, 0.3)
            bili_records.append({'stay_id': pid, 'time': t, 'bili': max(0.1, bili_val)})
    data['bili'] = pd.DataFrame(bili_records)
    
    # 血糖 (Glucose，使用患者级随机采样，约每4小时)
    glu_records = []
    for pid in patient_ids:
        base_glu = np.random.uniform(80, 120)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.3)
        for t in sample_times:
            glu_val = base_glu + np.random.normal(0, 15)
            glu_records.append({'stay_id': pid, 'time': t, 'glu': max(40, min(400, glu_val))})
    data['glu'] = pd.DataFrame(glu_records)
    
    # 乳酸（使用患者级随机采样，约每6小时）
    lac_records = []
    for pid in patient_ids:
        base_lac = np.random.uniform(1.0, 2.0)
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.3)
        for t in sample_times:
            lac_val = base_lac + np.random.normal(0, 0.5)
            if meta['is_septic'] and t >= meta['onset']:
                lac_val += 3.0 # 乳酸升高
                
            lac_records.append({'stay_id': pid, 'time': t, 'lact': max(0.5, lac_val)})  # 🔧 改为 lact（标准名称）
    data['lact'] = pd.DataFrame(lac_records)  # 🔧 改为 lact（与 CONCEPT_GROUPS_INTERNAL 一致）
    
    # 血小板（使用患者级随机采样，约每12小时）
    plt_records = []
    for pid in patient_ids:
        base_plt = np.random.uniform(150, 300)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.3)
        for t in sample_times:
            plt_val = base_plt + np.random.normal(0, 30)
            plt_records.append({'stay_id': pid, 'time': t, 'plt': max(10, plt_val)})
    data['plt'] = pd.DataFrame(plt_records)
    
    # 去甲肾上腺素（使用患者级随机采样）
    norepi_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if 12 <= t <= 48 and np.random.random() < 0.6:
                rate = np.random.uniform(0.05, 0.3)
                norepi_records.append({'stay_id': pid, 'time': t, 'norepi_rate': rate})
    data['norepi_rate'] = pd.DataFrame(norepi_records) if norepi_records else pd.DataFrame(
        columns=['stay_id', 'time', 'norepi_rate'])
    
    # SOFA-2 评分 (2025新标准，使用患者级随机采样，约每6小时)
    sofa2_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.3)
        for t in sample_times:
            # 基础分布
            probs = [0.55, 0.3, 0.1, 0.05, 0.0]
            if meta['is_septic'] and t >= meta['onset']:
                probs = [0.1, 0.2, 0.3, 0.25, 0.15]
                
            sofa2_resp = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_coag = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_liver = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_cardio = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_cns = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_renal = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_total = sofa2_resp + sofa2_coag + sofa2_liver + sofa2_cardio + sofa2_cns + sofa2_renal
            
            sofa2_records.append({
                'stay_id': pid, 'time': t, 'sofa2': sofa2_total,
                'sofa2_resp': sofa2_resp, 'sofa2_coag': sofa2_coag, 'sofa2_liver': sofa2_liver,
                'sofa2_cardio': sofa2_cardio, 'sofa2_cns': sofa2_cns, 'sofa2_renal': sofa2_renal,
            })
    data['sofa2'] = pd.DataFrame(sofa2_records)
    # 🔧 FIX: 提取子组件并剥离父 DF，避免合并 _x/_y 冲突
    _sofa2_sub_cols = ['sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal']
    for _sc in _sofa2_sub_cols:
        data[_sc] = data['sofa2'][['stay_id', 'time', _sc]].copy()
    data['sofa2'] = data['sofa2'][['stay_id', 'time', 'sofa2']].copy()
    
    # Sepsis-3 诊断数据 (严格基于 SOFA 变化)
    sep3_sofa2_records = []
    
    # 先把 sofa2 转换为 (stay_id, time) 索引以便查询
    sofa2_lookup = data['sofa2'].set_index(['stay_id', 'time'])['sofa2']
    
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        
        # 1. 疑似感染 susp_inf
        # 定义: 在发病前24h到发病后72h为“疑似感染窗口”
        # 这里简化：只要是septic患者，我们在 onset 之后标记疑似
        # samp 仅在具体的采样点为 1
        
        for t in time_points:
            # 默认值
            susp_inf_val = 0
            samp_val = 0
            sep3_val = 0
            infection_icd_val = 0
            
            if meta['is_septic']:
                # infection_icd 通常是静态诊断，整程为1
                infection_icd_val = 1
                
                # samp: 仅在采样点有值 (稀疏)
                if t == meta['samp_time']:
                    samp_val = 1
                    
                # susp_inf: 模拟有一个 suspicions window
                # 此处设为：samp_time 前24h 到 后72h
                samp_t = meta['samp_time']
                if samp_t >= 0 and (samp_t - 24 <= t <= samp_t + 72):
                    susp_inf_val = 1
                    
                # Sepsis-3: susp_inf=1 AND (current_sofa >= 2)
                # 真实标准是 delta_sofa >= 2，假设 baseline=0，则 absolute>=2
                try:
                    # 注意：sofa数据是每6小时采样的，中间时间点可能在dataframe里没有
                    # 这里通过 lookup 查找最近的有效值，或者因为我们的 time_points 包含所有点
                    # 但 sofa2_df 是 dense 的吗？ generate logic 是 time_points[::6]
                    # 我们需要插值或对齐。为简化 mock，我们在生成 sofa2 时只生成了部分点
                    # 但 export 要求对齐。这里我们简单处理：若无法查到准确sofa，则沿用上一个
                    pass 
                except:
                    pass
            
            sep3_sofa2_records.append({
                'stay_id': pid,
                'time': t,
                'sep3_sofa2': 0, # 稍后计算
                'susp_inf': susp_inf_val,
                'infection_icd': infection_icd_val,
                'samp': samp_val,
            })
            
    # 计算 Sep3 结果 (需要合并 sofa 数据)
    sep3_df = pd.DataFrame(sep3_sofa2_records)
    
    # 将 Sep3 表和 SOFA2 表合并计算最终 Sep3 状态
    # SOFA2 是每6小时一点，我们先 forward fill 到每小时
    sofa2_full = pd.DataFrame({'stay_id': patient_ids, 'key': 1}).merge(pd.DataFrame({'time': time_points, 'key': 1}), on='key').drop(columns=['key'])
    sofa2_source = data['sofa2'][['stay_id', 'time', 'sofa2']]
    sofa2_interpolated = sofa2_full.merge(sofa2_source, on=['stay_id', 'time'], how='left')
    sofa2_interpolated['sofa2'] = sofa2_interpolated.groupby('stay_id')['sofa2'].ffill().fillna(0)
    
    # 合并
    sep3_final = sep3_df.merge(sofa2_interpolated, on=['stay_id', 'time'], how='left')
    
    # 应用 Sepsis3 规则: susp_inf == 1 AND sofa2 >= 2
    sep3_final['sep3_sofa2'] = ((sep3_final['susp_inf'] == 1) & (sep3_final['sofa2'] >= 2)).astype(int)
    
    # 更新到 data
    # 🔧 保留所有时间点（包括0和1），模拟真实的稀疏性和缺失率
    # sep3_sofa2: 保留所有疑似感染窗口内的时间点（含0和1）
    sep3_with_context = sep3_final[
        (sep3_final['susp_inf'] == 1) |  # 疑似感染窗口
        (sep3_final['infection_icd'] == 1)  # 有感染诊断
    ].copy()
    
    # 对于 sep3_sofa2，保留疑似感染窗口内的所有记录（包含0值，模拟缺失率）
    data['sep3_sofa2'] = sep3_with_context[['stay_id', 'time', 'sep3_sofa2']] if len(sep3_with_context) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'sep3_sofa2'])
    data['susp_inf'] = sep3_with_context[['stay_id', 'time', 'susp_inf']] if len(sep3_with_context) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'susp_inf'])
    data['infection_icd'] = sep3_with_context[['stay_id', 'time', 'infection_icd']] if len(sep3_with_context) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'infection_icd'])
    data['samp'] = sep3_final[sep3_final['samp'] == 1][['stay_id', 'time', 'samp']] if (sep3_final['samp'] == 1).any() else pd.DataFrame(columns=['stay_id', 'time', 'samp'])
    
    # 🔧 删除组合概念别名（与 CONCEPT_GROUPS_INTERNAL 保持一致）
    # 删除: sep3_sofa2_susp_inf, sep3_sofa2_samp, sep3_sofa2_infection_icd
    
    # Sepsis-3 (SOFA-1) 同理
    sofa1_source = data['sofa'][['stay_id', 'time', 'sofa']]
    sofa1_interpolated = sofa2_full.merge(sofa1_source, on=['stay_id', 'time'], how='left')
    sofa1_interpolated['sofa'] = sofa1_interpolated.groupby('stay_id')['sofa'].ffill().fillna(0)
    
    sep3_sofa1_final = sep3_final[['stay_id', 'time', 'susp_inf', 'infection_icd']].merge(sofa1_interpolated, on=['stay_id', 'time'], how='left')
    sep3_sofa1_final['sep3_sofa1'] = ((sep3_sofa1_final['susp_inf'] == 1) & (sep3_sofa1_final['sofa'] >= 2)).astype(int)
    
    # sep3_sofa1: 保留所有在感染窗口内的记录（包括 0 和 1），模拟真实缺失率
    sep3_sofa1_in_window = sep3_sofa1_final[(sep3_sofa1_final['susp_inf'] == 1) | (sep3_sofa1_final['infection_icd'] == 1)]
    data['sep3_sofa1'] = sep3_sofa1_in_window[['stay_id', 'time', 'sep3_sofa1']] if len(sep3_sofa1_in_window) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'sep3_sofa1'])
    
    # 🔧 SOFA-1 子组件已在上方提取并剥离，此处无需重复
    
    # ============ 补充更多常用概念 ============
    
    # DBP (舒张压，模拟10%缺失率）
    dbp_records = []
    for pid in patient_ids:
        base_dbp = np.random.uniform(60, 80)
        for t in time_points:
            if np.random.random() < 0.9:
                dbp_val = base_dbp + np.sin(t / 5) * 8 + np.random.normal(0, 5)
                dbp_records.append({'stay_id': pid, 'time': t, 'dbp': max(40, min(110, dbp_val))})
    data['dbp'] = pd.DataFrame(dbp_records)
    
    # GCS (格拉斯哥昏迷评分，使用患者级随机采样，约每4小时)
    gcs_records = []
    for pid in patient_ids:
        base_gcs = np.random.choice([15, 14, 13, 12, 10, 8], p=[0.5, 0.2, 0.1, 0.08, 0.07, 0.05])
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            gcs_val = base_gcs + np.random.choice([-1, 0, 0, 0, 1], p=[0.1, 0.3, 0.3, 0.2, 0.1])
            gcs_records.append({'stay_id': pid, 'time': t, 'gcs': max(3, min(15, gcs_val))})
    data['gcs'] = pd.DataFrame(gcs_records)
    
    # 血气分析：pH, pco2, po2, lact（使用患者级随机采样，约每6小时）
    ph_records = []
    pco2_records = []
    po2_records = []
    for pid in patient_ids:
        base_ph = np.random.uniform(7.35, 7.45)
        base_pco2 = np.random.uniform(35, 45)
        base_po2 = np.random.uniform(80, 100)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        for t in sample_times:
            ph_records.append({'stay_id': pid, 'time': t, 'ph': base_ph + np.random.normal(0, 0.03)})
            pco2_records.append({'stay_id': pid, 'time': t, 'pco2': base_pco2 + np.random.normal(0, 3)})
            po2_records.append({'stay_id': pid, 'time': t, 'po2': max(60, base_po2 + np.random.normal(0, 10))})
    data['ph'] = pd.DataFrame(ph_records)
    data['pco2'] = pd.DataFrame(pco2_records)
    data['po2'] = pd.DataFrame(po2_records)
    # 🔧 lact 已在上方直接生成（不再需要从 lac 创建别名）
    
    # 呼吸系统：pafi, fio2, vent_ind（使用患者级随机采样，约每4小时）
    pafi_records = []
    fio2_records = []
    vent_ind_records = []
    for pid in patient_ids:
        base_fio2 = np.random.choice([0.21, 0.3, 0.4, 0.5], p=[0.4, 0.3, 0.2, 0.1])
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            fio2_val = base_fio2 + np.random.uniform(-0.05, 0.05)
            fio2_val = max(0.21, min(1.0, fio2_val))
            po2_val = 80 + np.random.normal(0, 15)
            pafi_val = po2_val / fio2_val
            vent = 1 if fio2_val > 0.3 else 0
            pafi_records.append({'stay_id': pid, 'time': t, 'pafi': pafi_val})
            fio2_records.append({'stay_id': pid, 'time': t, 'fio2': fio2_val * 100})  # 转为百分比
            vent_ind_records.append({'stay_id': pid, 'time': t, 'vent_ind': vent})
    data['pafi'] = pd.DataFrame(pafi_records)
    data['fio2'] = pd.DataFrame(fio2_records)
    data['vent_ind'] = pd.DataFrame(vent_ind_records)
    
    # 尿量（使用患者级随机采样，模拟30%缺失率）
    urine_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            # 30%概率无记录（缺失）
            if np.random.random() < 0.7:
                urine_val = np.random.uniform(30, 100)
                urine_records.append({'stay_id': pid, 'time': t, 'urine': urine_val})
    data['urine'] = pd.DataFrame(urine_records) if urine_records else pd.DataFrame(columns=['stay_id', 'time', 'urine'])
    
    # WBC (白细胞，使用患者级随机采样，约每12小时)
    wbc_records = []
    for pid in patient_ids:
        base_wbc = np.random.uniform(6, 12)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        for t in sample_times:
            wbc_val = base_wbc + np.random.normal(0, 2)
            wbc_records.append({'stay_id': pid, 'time': t, 'wbc': max(1, wbc_val)})
    data['wbc'] = pd.DataFrame(wbc_records)
    
    # 结局数据 (outcome) - 使用预先生成的元数据
    death_records = []
    los_icu_records = []
    los_hosp_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        death_records.append({'stay_id': pid, 'death': meta['death']})
        los_icu_records.append({'stay_id': pid, 'los_icu': meta['los_icu'] / 24})  # 转为天
        los_hosp = meta['los_icu'] / 24 + np.random.uniform(0, 10)  # 住院时间 >= ICU时间
        los_hosp_records.append({'stay_id': pid, 'los_hosp': los_hosp})
    data['death'] = pd.DataFrame(death_records)
    data['los_icu'] = pd.DataFrame(los_icu_records)
    data['los_hosp'] = pd.DataFrame(los_hosp_records)
    
    # 人口统计 (demographics) - 使用预先生成的元数据
    age_records = []
    weight_records = []
    height_records = []
    sex_records = []
    bmi_records = []
    adm_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        weight_val = np.random.uniform(50, 100)
        height_val = np.random.uniform(150, 190)
        bmi_val = weight_val / (height_val / 100) ** 2
        
        age_records.append({'stay_id': pid, 'age': meta['age']})
        weight_records.append({'stay_id': pid, 'weight': weight_val})
        height_records.append({'stay_id': pid, 'height': height_val})
        sex_records.append({'stay_id': pid, 'sex': meta['sex']})
        bmi_records.append({'stay_id': pid, 'bmi': bmi_val})
        adm_records.append({'stay_id': pid, 'adm': 1})  # 所有患者均有入院记录
    
    data['age'] = pd.DataFrame(age_records)
    data['weight'] = pd.DataFrame(weight_records)
    data['height'] = pd.DataFrame(height_records)
    data['sex'] = pd.DataFrame(sex_records)
    data['bmi'] = pd.DataFrame(bmi_records)
    data['adm'] = pd.DataFrame(adm_records)
    
    # 其他评分（使用患者级随机采样，约每6小时）
    qsofa_records = []
    sirs_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        for t in sample_times:
            qsofa_records.append({'stay_id': pid, 'time': t, 'qsofa': np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])})
            sirs_records.append({'stay_id': pid, 'time': t, 'sirs': np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.25, 0.25, 0.2, 0.1])})
    data['qsofa'] = pd.DataFrame(qsofa_records)
    data['sirs'] = pd.DataFrame(sirs_records)
    
    # 药物：抗生素使用
    abx_records = []
    for pid in patient_ids:
        abx_records.append({'stay_id': pid, 'abx': 1 if np.random.random() < 0.7 else 0})
    data['abx'] = pd.DataFrame(abx_records)
    
    # 🔧 FIX (2026-02-03): 药物：皮质类固醇 (corticosteroids) - 只记录发生的事件（NaN/1格式）
    # 只有发生时才记录1，没有发生时不生成记录（而不是生成0）
    cort_records = []
    for pid in patient_ids:
        if np.random.random() < 0.25:  # 25%患者使用皮质类固醇
            start_time = np.random.uniform(0, 24)
            cort_records.append({'stay_id': pid, 'time': start_time, 'cort': 1})
    data['cort'] = pd.DataFrame(cort_records) if cort_records else pd.DataFrame(columns=['stay_id', 'time', 'cort'])
    
    # ============ KDIGO AKI 急性肾损伤数据（使用患者级随机采样，约每4小时） ============
    aki_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        baseline_crea = np.random.uniform(0.6, 1.2)
        
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            # 基线肌酐附近波动
            crea = baseline_crea * (1 + np.random.normal(0, 0.1))
            
            # Sepsis 患者在发病后可能发生 AKI
            if meta['is_septic'] and t >= meta['onset']:
                # 30% 概率发生 AKI
                if np.random.random() < 0.3:
                    crea = baseline_crea * np.random.uniform(1.5, 3.0)
            
            # 计算 AKI 分期
            ratio = crea / baseline_crea
            if ratio >= 3.0 or crea >= 4.0:
                aki_stage = 3
            elif ratio >= 2.0:
                aki_stage = 2
            elif ratio >= 1.5 or crea >= baseline_crea + 0.3:
                aki_stage = 1
            else:
                aki_stage = 0
            
            aki_records.append({
                'stay_id': pid, 'time': t,
                'crea': round(crea, 2),
                'creat_low_past_7day': round(baseline_crea, 2),
                'aki_stage': aki_stage,
                'aki': 1 if aki_stage > 0 else 0
            })
    aki_full = pd.DataFrame(aki_records)
    data['aki'] = aki_full[['stay_id', 'time', 'aki']].copy()
    # 🔧 FIX: 先从完整 AKI 表派生子组件，再保留精简后的父 DF
    data['aki_stage'] = aki_full[['stay_id', 'time', 'aki_stage']].copy()
    data['creat_low_past_7day'] = aki_full[['stay_id', 'time', 'creat_low_past_7day']].copy()
    # 🔧 添加完整的AKI子特征（基于肌酐、尿量、RRT定义的）
    data['aki_stage_creat'] = aki_full[['stay_id', 'time', 'aki_stage']].copy()
    data['aki_stage_creat'].columns = ['stay_id', 'time', 'aki_stage_creat']
    # 尿量定义的AKI（随机生成，因为demo数据简化）
    aki_uo_records = []
    for _, row in aki_full.iterrows():
        # 尿量AKI通常与肌酐AKI相关但不完全一致
        uo_stage = max(0, row['aki_stage'] - np.random.randint(0, 2)) if row['aki_stage'] > 0 else 0
        aki_uo_records.append({'stay_id': row['stay_id'], 'time': row['time'], 'aki_stage_uo': uo_stage})
    data['aki_stage_uo'] = pd.DataFrame(aki_uo_records)
    # RRT定义的AKI（仅接受RRT的患者为Stage 3）
    aki_rrt_records = []
    for _, row in data['aki_stage'].iterrows():
        rrt_stage = 3 if row['aki_stage'] == 3 and np.random.random() < 0.3 else 0
        aki_rrt_records.append({'stay_id': row['stay_id'], 'time': row['time'], 'aki_stage_rrt': rrt_stage})
    data['aki_stage_rrt'] = pd.DataFrame(aki_rrt_records)
    
    # ============ 新增 KDIGO 相关特征 (2026-02-04) ============
    # creat_low_past_48hr: 过去48小时内最低肌酐（通常与 creat_low_past_7day 相似或稍高）
    creat_48hr_records = []
    for _, row in data['creat_low_past_7day'].iterrows():
        # 48hr内的最低肌酐通常略高于7天内的最低值
        baseline = row['creat_low_past_7day']
        creat_48hr = round(baseline * np.random.uniform(1.0, 1.15), 2)
        creat_48hr_records.append({'stay_id': row['stay_id'], 'time': row['time'], 'creat_low_past_48hr': creat_48hr})
    data['creat_low_past_48hr'] = pd.DataFrame(creat_48hr_records)
    
    # 尿量率（mL/kg/h）：基于患者体重的尿量产出率
    # 正常值: 0.5-1.5 mL/kg/h，AKI时 <0.5 mL/kg/h（Stage 1）, <0.3（Stage 2/3）
    uo_rate_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        patient_weight = data['weight'][data['weight']['stay_id'] == pid]['weight'].iloc[0] if len(data['weight'][data['weight']['stay_id'] == pid]) > 0 else 70
        
        # 使用与AKI相同的时间点
        patient_aki = data['aki_stage'][data['aki_stage']['stay_id'] == pid]
        for _, row in patient_aki.iterrows():
            t = row['time']
            aki_stage = row['aki_stage']
            
            # 根据AKI分期生成尿量率
            if aki_stage == 0:
                base_uo_rate = np.random.uniform(0.6, 1.5)  # 正常
            elif aki_stage == 1:
                base_uo_rate = np.random.uniform(0.3, 0.5)  # Stage 1: <0.5
            elif aki_stage == 2:
                base_uo_rate = np.random.uniform(0.15, 0.35)  # Stage 2: <0.3
            else:
                base_uo_rate = np.random.uniform(0.0, 0.2)  # Stage 3: <0.3或无尿
            
            # 6hr, 12hr, 24hr 窗口的尿量率（略有变化）
            uo_6hr = round(base_uo_rate * np.random.uniform(0.9, 1.1), 3)
            uo_12hr = round(base_uo_rate * np.random.uniform(0.85, 1.05), 3)
            uo_24hr = round(base_uo_rate * np.random.uniform(0.8, 1.0), 3)  # 24hr窗口通常更平滑
            
            uo_rate_records.append({
                'stay_id': pid, 'time': t,
                'uo_rt_6hr': uo_6hr,
                'uo_rt_12hr': uo_12hr,
                'uo_rt_24hr': uo_24hr
            })
    uo_rate_df = pd.DataFrame(uo_rate_records)
    data['uo_rt_6hr'] = uo_rate_df[['stay_id', 'time', 'uo_rt_6hr']].copy()
    data['uo_rt_12hr'] = uo_rate_df[['stay_id', 'time', 'uo_rt_12hr']].copy()
    data['uo_rt_24hr'] = uo_rate_df[['stay_id', 'time', 'uo_rt_24hr']].copy()
    
    # ============ 循环衰竭 (circEWS) 数据 ============
    circ_failure_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        
        for t in time_points:
            # 基线乳酸和MAP
            base_lact = np.random.uniform(0.8, 1.5)
            base_map = np.random.uniform(75, 95)
            
            lact = base_lact + np.random.normal(0, 0.3)
            map_val = base_map + np.random.normal(0, 5)
            
            # Sepsis 患者发病后可能发生循环衰竭
            if meta['is_septic'] and t >= meta['onset']:
                if np.random.random() < 0.4:
                    lact = np.random.uniform(2.5, 8.0)
                    map_val = np.random.uniform(50, 70)
            
            # 计算循环衰竭事件等级
            lactate_elevated = lact >= 2.0
            map_low = map_val <= 65
            
            if lactate_elevated and map_low:
                circ_event = np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25])
            elif lactate_elevated:
                circ_event = 1 if np.random.random() < 0.3 else 0
            else:
                circ_event = 0
            
            circ_failure_records.append({
                'stay_id': pid, 'time': t,
                'lact': round(lact, 2),
                'map': round(map_val, 1),
                'circ_event': circ_event,
                'circ_failure': 1 if circ_event > 0 else 0
            })
    data['circ_failure'] = pd.DataFrame(circ_failure_records)
    # 🔧 FIX: 提取 circ_event 并剥离父 DF
    data['circ_event'] = data['circ_failure'][['stay_id', 'time', 'circ_event']].copy()
    data['circ_failure'] = data['circ_failure'][['stay_id', 'time', 'circ_failure']].copy()
    
    # ============ 呼吸机参数（使用患者级随机采样） ============
    peep_records = []
    tidal_vol_records = []
    tidal_vol_set_records = []
    pip_records = []
    plateau_pres_records = []
    mean_airway_pres_records = []
    minute_vol_records = []
    vent_rate_records = []
    compliance_records = []
    driving_pres_records = []
    ps_records = []
    
    for pid in patient_ids:
        # 仅60%患者有呼吸机参数（模拟非所有患者都需要机械通气）
        has_vent = np.random.random() < 0.6
        if has_vent:
            # 呼吸机开始时间随机在6-24小时之间
            vent_start = np.random.choice(range(6, min(24, hours)))
            # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
            sample_times = get_random_sample_times(pid, base_interval=2, jitter=0.4)
            for t in sample_times:
                if t >= vent_start:
                    # 再有20%概率该时间点缺失记录（设备故障/记录缺失）
                    if np.random.random() < 0.8:
                        peep = np.random.uniform(5, 15)
                        tidal_vol = np.random.uniform(350, 550)
                        tidal_vol_set = np.random.uniform(400, 600)
                        pip = np.random.uniform(15, 35)
                        plateau = np.random.uniform(18, 30)
                        mean_airway = np.random.uniform(10, 20)
                        minute_vol = np.random.uniform(6, 12)
                        rate = np.random.uniform(12, 20)
                        compliance = tidal_vol / max(1, plateau - peep)
                        driving = plateau - peep
                        ps = np.random.uniform(5, 15)
                        
                        peep_records.append({'stay_id': pid, 'time': t, 'peep': peep})
                        tidal_vol_records.append({'stay_id': pid, 'time': t, 'tidal_vol': tidal_vol})
                        # tidal_vol_set较少记录（50%概率）
                        if np.random.random() < 0.5:
                            tidal_vol_set_records.append({'stay_id': pid, 'time': t, 'tidal_vol_set': tidal_vol_set})
                        pip_records.append({'stay_id': pid, 'time': t, 'pip': pip})
                        # plateau_pres和compliance较少记录（40%概率）
                        if np.random.random() < 0.4:
                            plateau_pres_records.append({'stay_id': pid, 'time': t, 'plateau_pres': plateau})
                            compliance_records.append({'stay_id': pid, 'time': t, 'compliance': compliance})
                            driving_pres_records.append({'stay_id': pid, 'time': t, 'driving_pres': driving})
                        # mean_airway和minute_vol中等记录率（60%概率）
                        if np.random.random() < 0.6:
                            mean_airway_pres_records.append({'stay_id': pid, 'time': t, 'mean_airway_pres': mean_airway})
                            minute_vol_records.append({'stay_id': pid, 'time': t, 'minute_vol': minute_vol})
                        vent_rate_records.append({'stay_id': pid, 'time': t, 'vent_rate': rate})
                        ps_records.append({'stay_id': pid, 'time': t, 'ps': ps})
    
    data['peep'] = pd.DataFrame(peep_records) if peep_records else pd.DataFrame(columns=['stay_id', 'time', 'peep'])
    data['tidal_vol'] = pd.DataFrame(tidal_vol_records) if tidal_vol_records else pd.DataFrame(columns=['stay_id', 'time', 'tidal_vol'])
    data['tidal_vol_set'] = pd.DataFrame(tidal_vol_set_records) if tidal_vol_set_records else pd.DataFrame(columns=['stay_id', 'time', 'tidal_vol_set'])
    data['pip'] = pd.DataFrame(pip_records) if pip_records else pd.DataFrame(columns=['stay_id', 'time', 'pip'])
    data['plateau_pres'] = pd.DataFrame(plateau_pres_records) if plateau_pres_records else pd.DataFrame(columns=['stay_id', 'time', 'plateau_pres'])
    data['mean_airway_pres'] = pd.DataFrame(mean_airway_pres_records) if mean_airway_pres_records else pd.DataFrame(columns=['stay_id', 'time', 'mean_airway_pres'])
    data['minute_vol'] = pd.DataFrame(minute_vol_records) if minute_vol_records else pd.DataFrame(columns=['stay_id', 'time', 'minute_vol'])
    data['vent_rate'] = pd.DataFrame(vent_rate_records) if vent_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'vent_rate'])
    data['compliance'] = pd.DataFrame(compliance_records) if compliance_records else pd.DataFrame(columns=['stay_id', 'time', 'compliance'])
    data['driving_pres'] = pd.DataFrame(driving_pres_records) if driving_pres_records else pd.DataFrame(columns=['stay_id', 'time', 'driving_pres'])
    data['ps'] = pd.DataFrame(ps_records) if ps_records else pd.DataFrame(columns=['stay_id', 'time', 'ps'])
    
    # ============ 补充更多实验室检查（使用患者级随机采样，约每12小时） ============
    alp_records = []
    bun_records = []
    alt_records = []
    ast_records = []
    ca_records = []
    mg_records = []
    cl_records = []
    ck_records = []
    ckmb_records = []
    tri_records = []
    tnt_records = []
    crp_records = []
    bicar_records = []
    bili_dir_records = []
    alb_records = []
    be_records = []
    cai_records = []
    tco2_records = []
    
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        for t in sample_times:
            alp_records.append({'stay_id': pid, 'time': t, 'alp': np.random.uniform(40, 120)})
            bun_records.append({'stay_id': pid, 'time': t, 'bun': np.random.uniform(10, 40)})
            alt_records.append({'stay_id': pid, 'time': t, 'alt': np.random.uniform(10, 60)})
            ast_records.append({'stay_id': pid, 'time': t, 'ast': np.random.uniform(10, 60)})
            ca_records.append({'stay_id': pid, 'time': t, 'ca': np.random.uniform(8.5, 10.5)})
            mg_records.append({'stay_id': pid, 'time': t, 'mg': np.random.uniform(1.5, 2.5)})
            cl_records.append({'stay_id': pid, 'time': t, 'cl': np.random.uniform(95, 110)})
            ck_records.append({'stay_id': pid, 'time': t, 'ck': np.random.uniform(50, 300)})
            ckmb_records.append({'stay_id': pid, 'time': t, 'ckmb': np.random.uniform(0, 10)})
            tri_records.append({'stay_id': pid, 'time': t, 'tri': np.random.uniform(0, 0.5)})
            tnt_records.append({'stay_id': pid, 'time': t, 'tnt': np.random.uniform(0, 0.5)})
            crp_records.append({'stay_id': pid, 'time': t, 'crp': np.random.uniform(5, 100)})
            bicar_records.append({'stay_id': pid, 'time': t, 'bicar': np.random.uniform(22, 28)})
            bili_dir_records.append({'stay_id': pid, 'time': t, 'bili_dir': np.random.uniform(0.1, 0.5)})
            alb_records.append({'stay_id': pid, 'time': t, 'alb': np.random.uniform(3.0, 4.5)})
            be_records.append({'stay_id': pid, 'time': t, 'be': np.random.uniform(-3, 3)})
            cai_records.append({'stay_id': pid, 'time': t, 'cai': np.random.uniform(1.1, 1.3)})
            tco2_records.append({'stay_id': pid, 'time': t, 'tco2': np.random.uniform(23, 29)})
    
    data['alp'] = pd.DataFrame(alp_records)
    data['bun'] = pd.DataFrame(bun_records)
    data['alt'] = pd.DataFrame(alt_records)
    data['ast'] = pd.DataFrame(ast_records)
    data['ca'] = pd.DataFrame(ca_records)
    data['mg'] = pd.DataFrame(mg_records)
    data['cl'] = pd.DataFrame(cl_records)
    data['ck'] = pd.DataFrame(ck_records)
    data['ckmb'] = pd.DataFrame(ckmb_records)
    data['tri'] = pd.DataFrame(tri_records)
    data['tnt'] = pd.DataFrame(tnt_records)
    data['crp'] = pd.DataFrame(crp_records)
    data['bicar'] = pd.DataFrame(bicar_records)
    data['bili_dir'] = pd.DataFrame(bili_dir_records)
    data['alb'] = pd.DataFrame(alb_records)
    data['be'] = pd.DataFrame(be_records)
    data['cai'] = pd.DataFrame(cai_records)
    data['tco2'] = pd.DataFrame(tco2_records)
    # 🔧 删除别名概念（与 CONCEPT_GROUPS_INTERNAL 保持一致）
    # 删除: bicarb (bicar的别名), potassium (k的别名)
    
    # ============ 血液学扩展（使用患者级随机采样，约每12小时） ============
    hct_records = []
    rbc_records = []
    rdw_records = []
    mcv_records = []
    mch_records = []
    mchc_records = []
    neut_records = []
    lymph_records = []
    eos_records = []
    basos_records = []
    bnd_records = []
    inr_pt_records = []
    ptt_records = []
    pt_records = []
    fgn_records = []
    esr_records = []
    hba1c_records = []
    
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        for t in sample_times:
            hct_records.append({'stay_id': pid, 'time': t, 'hct': np.random.uniform(30, 45)})
            rbc_records.append({'stay_id': pid, 'time': t, 'rbc': np.random.uniform(3.5, 5.5)})
            rdw_records.append({'stay_id': pid, 'time': t, 'rdw': np.random.uniform(11, 15)})
            mcv_records.append({'stay_id': pid, 'time': t, 'mcv': np.random.uniform(80, 100)})
            mch_records.append({'stay_id': pid, 'time': t, 'mch': np.random.uniform(27, 32)})
            mchc_records.append({'stay_id': pid, 'time': t, 'mchc': np.random.uniform(32, 36)})
            neut_records.append({'stay_id': pid, 'time': t, 'neut': np.random.uniform(40, 75)})
            lymph_records.append({'stay_id': pid, 'time': t, 'lymph': np.random.uniform(20, 40)})
            eos_records.append({'stay_id': pid, 'time': t, 'eos': np.random.uniform(1, 5)})
            basos_records.append({'stay_id': pid, 'time': t, 'basos': np.random.uniform(0, 2)})
            bnd_records.append({'stay_id': pid, 'time': t, 'bnd': np.random.uniform(0, 10)})
            inr_pt_records.append({'stay_id': pid, 'time': t, 'inr_pt': np.random.uniform(0.9, 1.3)})
            ptt_records.append({'stay_id': pid, 'time': t, 'ptt': np.random.uniform(25, 35)})
            pt_records.append({'stay_id': pid, 'time': t, 'pt': np.random.uniform(11, 14)})
            fgn_records.append({'stay_id': pid, 'time': t, 'fgn': np.random.uniform(200, 400)})
            esr_records.append({'stay_id': pid, 'time': t, 'esr': np.random.uniform(5, 25)})
            hba1c_records.append({'stay_id': pid, 'time': t, 'hba1c': np.random.uniform(5.0, 7.0)})
    
    data['hct'] = pd.DataFrame(hct_records)
    data['rbc'] = pd.DataFrame(rbc_records)
    data['rdw'] = pd.DataFrame(rdw_records)
    data['mcv'] = pd.DataFrame(mcv_records)
    data['mch'] = pd.DataFrame(mch_records)
    data['mchc'] = pd.DataFrame(mchc_records)
    data['neut'] = pd.DataFrame(neut_records)
    data['lymph'] = pd.DataFrame(lymph_records)
    data['eos'] = pd.DataFrame(eos_records)
    data['basos'] = pd.DataFrame(basos_records)
    data['bnd'] = pd.DataFrame(bnd_records)
    data['inr_pt'] = pd.DataFrame(inr_pt_records)
    data['ptt'] = pd.DataFrame(ptt_records)
    data['pt'] = pd.DataFrame(pt_records)
    data['fgn'] = pd.DataFrame(fgn_records)
    data['esr'] = pd.DataFrame(esr_records)
    data['hba1c'] = pd.DataFrame(hba1c_records)
    
    # ============ 更多药物 ============
    dopa_rate_records = []
    dopa_dur_records = []
    dopa60_records = []
    epi_dur_records = []
    epi_rate_records = []
    epi60_records = []
    norepi_rate_records = []
    norepi_dur_records = []
    norepi60_records = []
    adh_rate_records = []
    phn_rate_records = []
    dobu_rate_records = []
    dobu_dur_records = []
    dobu60_records = []
    ins_records = []
    dex_records = []
    
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=3, jitter=0.4)
        for t in sample_times:
            if np.random.random() < 0.3:
                dopa_rate_records.append({'stay_id': pid, 'time': t, 'dopa_rate': np.random.uniform(2, 10)})
                epi_rate_records.append({'stay_id': pid, 'time': t, 'epi_rate': np.random.uniform(0.01, 0.1)})
                norepi_rate_records.append({'stay_id': pid, 'time': t, 'norepi_rate': np.random.uniform(0.01, 1.0)})
                dobu_rate_records.append({'stay_id': pid, 'time': t, 'dobu_rate': np.random.uniform(2, 10)})
                adh_rate_records.append({'stay_id': pid, 'time': t, 'adh_rate': np.random.uniform(0.01, 0.04)})
                phn_rate_records.append({'stay_id': pid, 'time': t, 'phn_rate': np.random.uniform(0.1, 0.5)})
        
        dopa_dur_records.append({'stay_id': pid, 'dopa_dur': np.random.uniform(0, 48)})
        epi_dur_records.append({'stay_id': pid, 'epi_dur': np.random.uniform(0, 24)})
        norepi_dur_records.append({'stay_id': pid, 'norepi_dur': np.random.uniform(0, 72)})
        dobu_dur_records.append({'stay_id': pid, 'dobu_dur': np.random.uniform(0, 36)})
        dopa60_records.append({'stay_id': pid, 'dopa60': 1 if np.random.random() < 0.4 else 0})
        epi60_records.append({'stay_id': pid, 'epi60': 1 if np.random.random() < 0.3 else 0})
        norepi60_records.append({'stay_id': pid, 'norepi60': 1 if np.random.random() < 0.5 else 0})
        dobu60_records.append({'stay_id': pid, 'dobu60': 1 if np.random.random() < 0.3 else 0})
        ins_records.append({'stay_id': pid, 'ins': np.random.uniform(0, 10)})
        dex_records.append({'stay_id': pid, 'dex': np.random.uniform(0, 1.5)})
    
    data['dopa_rate'] = pd.DataFrame(dopa_rate_records) if dopa_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'dopa_rate'])
    data['dopa_dur'] = pd.DataFrame(dopa_dur_records)
    data['dopa60'] = pd.DataFrame(dopa60_records)
    data['epi_rate'] = pd.DataFrame(epi_rate_records) if epi_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'epi_rate'])
    data['epi_dur'] = pd.DataFrame(epi_dur_records)
    data['epi60'] = pd.DataFrame(epi60_records)
    data['norepi_rate'] = pd.DataFrame(norepi_rate_records) if norepi_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'norepi_rate'])
    data['norepi_dur'] = pd.DataFrame(norepi_dur_records)
    data['norepi60'] = pd.DataFrame(norepi60_records)
    data['adh_rate'] = pd.DataFrame(adh_rate_records) if adh_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'adh_rate'])
    data['phn_rate'] = pd.DataFrame(phn_rate_records) if phn_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'phn_rate'])
    data['dobu_rate'] = pd.DataFrame(dobu_rate_records) if dobu_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'dobu_rate'])
    data['dobu_dur'] = pd.DataFrame(dobu_dur_records)
    data['dobu60'] = pd.DataFrame(dobu60_records)
    data['ins'] = pd.DataFrame(ins_records)
    data['dex'] = pd.DataFrame(dex_records)
    data['norepi_equiv'] = data['norepi_rate'].copy() if 'norepi_rate' in data else pd.DataFrame()
    
    # vaso_ind (血管活性药物指示)
    vaso_ind_records = []
    for pid in patient_ids:
        vaso_ind_records.append({'stay_id': pid, 'vaso_ind': 1 if np.random.random() < 0.6 else 0})
    data['vaso_ind'] = pd.DataFrame(vaso_ind_records)
    
    # ============ 神经和其他支持（使用患者级随机采样） ============
    rass_records = []
    avpu_records = []
    egcs_records = []
    mgcs_records = []
    vgcs_records = []
    tgcs_records = []
    sedated_gcs_records = []
    
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        for t in sample_times:
            rass_records.append({'stay_id': pid, 'time': t, 'rass': np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], p=[0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.05, 0.05, 0.03, 0.02])})
            egcs_records.append({'stay_id': pid, 'time': t, 'egcs': np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])})
            mgcs_records.append({'stay_id': pid, 'time': t, 'mgcs': np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25])})
            vgcs_records.append({'stay_id': pid, 'time': t, 'vgcs': np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.2, 0.25, 0.3])})
            avpu_records.append({'stay_id': pid, 'time': t, 'avpu': np.random.choice(['A', 'V', 'P', 'U'], p=[0.6, 0.2, 0.1, 0.1])})
        tgcs_records.append({'stay_id': pid, 'tgcs': np.random.choice([15, 14, 13, 12, 10, 8, 6], p=[0.5, 0.2, 0.1, 0.08, 0.07, 0.03, 0.02])})
        sedated_gcs_records.append({'stay_id': pid, 'sedated_gcs': np.random.choice([15, 14, 13], p=[0.7, 0.2, 0.1])})
    
    data['rass'] = pd.DataFrame(rass_records)
    data['avpu'] = pd.DataFrame(avpu_records)
    data['egcs'] = pd.DataFrame(egcs_records)
    data['mgcs'] = pd.DataFrame(mgcs_records)
    data['vgcs'] = pd.DataFrame(vgcs_records)
    data['tgcs'] = pd.DataFrame(tgcs_records)
    data['sedated_gcs'] = pd.DataFrame(sedated_gcs_records)
    
    # ============ 其他指标（使用高效循环而非列表推导式）============
    # 静态指标（每患者一个值）
    static_records = {
        'rrt': [], 'ecmo': [], 'height': [], 'bmi': [], 'sex': [], 'adm': [], 
        'los_hosp': [], 'vent_start': [], 'vent_end': [], 'cort': []
    }
    
    # RRT改为时间序列数据（仅10%患者使用）
    rrt_records = []
    rrt_patient_ids = set()  # 记录有RRT的患者ID
    for pid in patient_ids:
        # 10%患者接受RRT
        if np.random.random() < 0.1:
            rrt_patient_ids.add(pid)
            # RRT开始时间随机在12-48小时之间
            rrt_start = np.random.choice(range(12, min(48, hours)))
            for t in time_points:
                if t >= rrt_start:
                    rrt_records.append({'stay_id': pid, 'time': t, 'rrt': 1})
    data['rrt'] = pd.DataFrame(rrt_records) if rrt_records else pd.DataFrame(columns=['stay_id', 'time', 'rrt'])
    
    # rrt_criteria 也改为时间序列（与rrt相同，但列名不同）
    if rrt_records:
        data['rrt_criteria'] = data['rrt'].rename(columns={'rrt': 'rrt_criteria'}).copy()
    else:
        data['rrt_criteria'] = pd.DataFrame(columns=['stay_id', 'time', 'rrt_criteria'])
    
    for pid in patient_ids:
        # 保留静态版本用于其他用途（但不覆盖时间序列版本）
        static_records['rrt'].append({'stay_id': pid, 'rrt_static': 1 if pid in rrt_patient_ids else 0})
        # 🔧 FIX (2026-02-03): ecmo只记录发生的事件（NaN/1格式）
        # 只有5%患者使用ECMO，只在发生时记录1
        if np.random.random() < 0.05:
            static_records['ecmo'].append({'stay_id': pid, 'ecmo': 1})
        # 🔧 注意：height, bmi, sex, adm, los_hosp 已在前面使用 patient_sepsis_meta 正确生成
        # 这里只生成那些前面没有生成的静态字段
        static_records['vent_start'].append({'stay_id': pid, 'vent_start': np.random.choice(time_points[:min(24, len(time_points))])})
        static_records['vent_end'].append({'stay_id': pid, 'vent_end': np.random.choice(time_points[-min(24, len(time_points)):])})
        # 🔧 FIX (2026-02-03): cort只记录发生的事件（NaN/1格式）
        if np.random.random() < 0.3:
            static_records['cort'].append({'stay_id': pid, 'cort': 1})
    
    # 只为非RRT且未在前面生成的静态指标创建DataFrame
    # 🔧 跳过已正确生成的: rrt(时间序列), sex, age, death, los_icu, los_hosp, weight, height, bmi, adm
    already_generated = {'rrt', 'sex', 'age', 'death', 'los_icu', 'los_hosp', 'weight', 'height', 'bmi', 'adm'}
    for key, records in static_records.items():
        if key not in already_generated:
            # 🔧 FIX: 如果记录为空，创建带正确列名的空DataFrame
            if records:
                data[key] = pd.DataFrame(records)
            else:
                # 根据key确定列名
                if key == 'ecmo':
                    data[key] = pd.DataFrame(columns=['stay_id', 'ecmo'])
                elif key == 'cort':
                    data[key] = pd.DataFrame(columns=['stay_id', 'cort'])
                else:
                    data[key] = pd.DataFrame(records)
    
    # 🔧 注意: ecmo, ecmo_indication, mech_circ_support 在后面的代码中单独生成
    # （约第3298-3320行），此处不再复制，避免生成顺序问题
    
    # 时间序列指标（使用患者级随机采样）
    mews_records = []
    news_records = []
    hbco_records = []
    methb_records = []
    k_records = []
    na_records = []
    phos_records = []
    hgb_records = []
    safi_records = []
    
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times_6h = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        sample_times_12h = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        sample_times_4h = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        
        for t in sample_times_6h:
            mews_records.append({'stay_id': pid, 'time': t, 'mews': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])})
            news_records.append({'stay_id': pid, 'time': t, 'news': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[0.25, 0.2, 0.18, 0.15, 0.1, 0.07, 0.03, 0.02])})
        
        for t in sample_times_12h:
            if 'k' not in data:
                k_records.append({'stay_id': pid, 'time': t, 'k': np.random.uniform(3.5, 5.0)})
            if 'na' not in data:
                na_records.append({'stay_id': pid, 'time': t, 'na': np.random.uniform(135, 145)})
            if 'phos' not in data:
                phos_records.append({'stay_id': pid, 'time': t, 'phos': np.random.uniform(2.5, 4.5)})
            if 'hgb' not in data:
                hgb_records.append({'stay_id': pid, 'time': t, 'hgb': np.random.uniform(10, 15)})
            hbco_records.append({'stay_id': pid, 'time': t, 'hbco': np.random.uniform(0, 5)})
            methb_records.append({'stay_id': pid, 'time': t, 'methb': np.random.uniform(0, 2)})
        
        for t in sample_times_4h:
            safi_records.append({'stay_id': pid, 'time': t, 'safi': np.random.uniform(200, 450)})
    
    data['mews'] = pd.DataFrame(mews_records)
    data['news'] = pd.DataFrame(news_records)
    data['hbco'] = pd.DataFrame(hbco_records)
    data['methb'] = pd.DataFrame(methb_records)
    data['safi'] = pd.DataFrame(safi_records)
    
    if k_records:
        data['k'] = pd.DataFrame(k_records)
    if na_records:
        data['na'] = pd.DataFrame(na_records)
    if phos_records:
        data['phos'] = pd.DataFrame(phos_records)
    if hgb_records:
        data['hgb'] = pd.DataFrame(hgb_records)
    
    # 数据复制和别名
    data['mech_vent'] = data['vent_ind'].copy() if 'vent_ind' in data and not data['vent_ind'].empty else pd.DataFrame(columns=['stay_id', 'time', 'mech_vent'])
    
    # vent_start 和 vent_end (机械通气起止时间)
    vent_start_records = []
    vent_end_records = []
    if 'vent_ind' in data and not data['vent_ind'].empty:
        for pid in patient_ids:
            pid_vent = data['vent_ind'][data['vent_ind']['stay_id'] == pid].copy()
            if len(pid_vent) > 0 and (pid_vent['vent_ind'] == 1).any():
                vent_times = pid_vent[pid_vent['vent_ind'] == 1]['time']
                if len(vent_times) > 0:
                    start_t = vent_times.min()
                    end_t = vent_times.max() + 4  # 假设每次测量持续4小时
                    vent_start_records.append({'stay_id': pid, 'time': start_t, 'vent_start': 1})
                    vent_end_records.append({'stay_id': pid, 'time': end_t, 'vent_end': 1})
    data['vent_start'] = pd.DataFrame(vent_start_records) if vent_start_records else pd.DataFrame(columns=['stay_id', 'time', 'vent_start'])
    data['vent_end'] = pd.DataFrame(vent_end_records) if vent_end_records else pd.DataFrame(columns=['stay_id', 'time', 'vent_end'])
    
    # ECMO (体外膜肺氧合) - 罕见事件，约3%患者
    ecmo_records = []
    ecmo_indication_records = []
    for pid in patient_ids:
        if np.random.random() < 0.03:  # 3%概率使用ECMO
            ecmo_start = np.random.uniform(12, 48)
            ecmo_indication = np.random.choice(['ARDS', 'Cardiogenic_shock', 'Bridge_to_transplant'])
            ecmo_records.append({'stay_id': pid, 'time': ecmo_start, 'ecmo': 1})
            ecmo_indication_records.append({'stay_id': pid, 'time': ecmo_start, 'ecmo_indication': ecmo_indication})
    data['ecmo'] = pd.DataFrame(ecmo_records) if ecmo_records else pd.DataFrame(columns=['stay_id', 'time', 'ecmo'])
    data['ecmo_indication'] = pd.DataFrame(ecmo_indication_records) if ecmo_indication_records else pd.DataFrame(columns=['stay_id', 'time', 'ecmo_indication'])
    
    # 🔧 FIX (2026-02-04): mech_circ_support - 机械循环支持（IABP/LVAD/Impella/VA-ECMO）
    # 真实数据中非常罕见，约2-3%的ICU患者使用（比ECMO稍多，因为包括IABP等）
    # 这里在ecmo生成之后更新mech_circ_support，确保反映正确的缺失率
    mech_circ_records = []
    for pid in patient_ids:
        # 2.5%概率使用机械循环支持（包括ECMO + IABP + LVAD等）
        if np.random.random() < 0.025:
            mcs_start = np.random.uniform(12, 48)
            mech_circ_records.append({'stay_id': pid, 'time': mcs_start, 'mech_circ_support': 1})
    data['mech_circ_support'] = pd.DataFrame(mech_circ_records) if mech_circ_records else pd.DataFrame(columns=['stay_id', 'time', 'mech_circ_support'])
    
    if 'fio2' in data and not data['fio2'].empty:
        data['supp_o2'] = data['fio2'].copy()
        data['supp_o2']['supp_o2'] = (data['supp_o2']['fio2'] > 21).astype(int)
        data['supp_o2'] = data['supp_o2'][['stay_id', 'time', 'supp_o2']]
    else:
        data['supp_o2'] = pd.DataFrame(columns=['stay_id', 'time', 'supp_o2'])
    
    # spo2/sao2 别名处理（避免循环引用）
    if 'spo2' in data and not data['spo2'].empty:
        # spo2已经存在，创建o2sat和sao2别名
        if 'o2sat' not in data or data['o2sat'].empty:
            data['o2sat'] = data['spo2'].rename(columns={'spo2': 'o2sat'}).copy()
        if 'sao2' not in data or data['sao2'].empty:
            data['sao2'] = data['spo2'].rename(columns={'spo2': 'sao2'}).copy()
    
    data['ett_gcs'] = data['gcs'].copy() if 'gcs' in data and not data['gcs'].empty else pd.DataFrame(columns=['stay_id', 'time', 'ett_gcs'])
    # urine24: 24小时累计尿量，每6小时一个记录点（模拟真实采样频率）
    urine24_records = []
    if 'urine' in data and not data['urine'].empty:
        for pid in patient_ids:
            pid_urine = data['urine'][data['urine']['stay_id'] == pid]
            for t in time_points[::6]:  # 每6小时记录一次
                # 计算过去24小时的尿量
                recent_urine = pid_urine[(pid_urine['time'] >= max(0, t-24)) & (pid_urine['time'] <= t)]
                if len(recent_urine) > 0:  # 只有当有数据时才记录
                    urine24_records.append({
                        'stay_id': pid,
                        'time': t,
                        'urine24': recent_urine['urine'].sum()
                    })
    data['urine24'] = pd.DataFrame(urine24_records) if urine24_records else pd.DataFrame(columns=['stay_id', 'time', 'urine24'])
    
    # === 🆕 新增 12 个缺失的概念（2026-02-03）===
    
    # 1. uo_6h, uo_12h, uo_24h: 6/12/24小时尿量率 (mL/kg/h)
    uo_6h_records = []
    uo_12h_records = []
    uo_24h_records = []
    if 'urine' in data and not data['urine'].empty and 'weight' in data and not data['weight'].empty:
        weight_dict = data['weight'].set_index('stay_id')['weight'].to_dict()
        for pid in patient_ids:
            if pid not in weight_dict:
                continue
            weight = weight_dict[pid]
            pid_urine = data['urine'][data['urine']['stay_id'] == pid]
            
            for t in time_points[::3]:  # 每3小时采样一次
                # 6小时尿量率
                recent_6h = pid_urine[(pid_urine['time'] >= max(0, t-6)) & (pid_urine['time'] <= t)]
                if len(recent_6h) > 0:
                    uo_6h = recent_6h['urine'].sum() / weight / 6.0
                    uo_6h_records.append({'stay_id': pid, 'time': t, 'uo_6h': uo_6h})
                
                # 12小时尿量率
                recent_12h = pid_urine[(pid_urine['time'] >= max(0, t-12)) & (pid_urine['time'] <= t)]
                if len(recent_12h) > 0:
                    uo_12h = recent_12h['urine'].sum() / weight / 12.0
                    uo_12h_records.append({'stay_id': pid, 'time': t, 'uo_12h': uo_12h})
                
                # 24小时尿量率
                recent_24h = pid_urine[(pid_urine['time'] >= max(0, t-24)) & (pid_urine['time'] <= t)]
                if len(recent_24h) > 0:
                    uo_24h = recent_24h['urine'].sum() / weight / 24.0
                    uo_24h_records.append({'stay_id': pid, 'time': t, 'uo_24h': uo_24h})
    
    data['uo_6h'] = pd.DataFrame(uo_6h_records) if uo_6h_records else pd.DataFrame(columns=['stay_id', 'time', 'uo_6h'])
    data['uo_12h'] = pd.DataFrame(uo_12h_records) if uo_12h_records else pd.DataFrame(columns=['stay_id', 'time', 'uo_12h'])
    data['uo_24h'] = pd.DataFrame(uo_24h_records) if uo_24h_records else pd.DataFrame(columns=['stay_id', 'time', 'uo_24h'])
    
    # 🔧 2026-02-04: 移除了重复的 kdigo_creat/kdigo_uo/kdigo_aki 创建代码
    # 这些概念与 aki_stage_creat/aki_stage_uo/aki_stage 完全重复，只保留后者
    
    # 3. motor_response: GCS运动反应分项（从gcs中提取）
    motor_response_records = []
    if 'gcs' in data and not data['gcs'].empty:
        for pid in patient_ids:
            pid_gcs = data['gcs'][data['gcs']['stay_id'] == pid]
            for _, row in pid_gcs.iterrows():
                # motor response 通常是 GCS 中的一部分 (1-6分)
                # 这里简化为 GCS/3 取整（模拟）
                motor_score = max(1, min(6, int(row['gcs'] / 3)))
                motor_response_records.append({
                    'stay_id': pid,
                    'time': row['time'],
                    'motor_response': motor_score
                })
    data['motor_response'] = pd.DataFrame(motor_response_records) if motor_response_records else pd.DataFrame(columns=['stay_id', 'time', 'motor_response'])
    
    # 4. delirium_positive: 谵妄阳性（基于RASS和GCS评估）
    delirium_positive_records = []
    if 'rass' in data and not data['rass'].empty:
        for pid in patient_ids:
            pid_rass = data['rass'][data['rass']['stay_id'] == pid]
            for _, row in pid_rass.iterrows():
                # 谵妄通常出现在 RASS > 0 且 < 4，或波动性意识状态
                # 这里简化为 RASS 在 1-3 时约30%几率阳性
                is_delirium = 1 if (1 <= row['rass'] <= 3 and np.random.random() < 0.3) else 0
                delirium_positive_records.append({
                    'stay_id': pid,
                    'time': row['time'],
                    'delirium_positive': is_delirium
                })
    data['delirium_positive'] = pd.DataFrame(delirium_positive_records) if delirium_positive_records else pd.DataFrame(columns=['stay_id', 'time', 'delirium_positive'])
    
    # 5. delirium_tx: 谵妄治疗（通常使用抗精神病药物）
    delirium_tx_records = []
    if 'delirium_positive' in data and not data['delirium_positive'].empty:
        # 假设约50%的谵妄阳性患者会接受治疗
        delirium_pts = data['delirium_positive'][data['delirium_positive']['delirium_positive'] == 1]['stay_id'].unique()
        for pid in delirium_pts:
            if np.random.random() < 0.5:  # 50%接受治疗
                treatment_start = np.random.uniform(12, 60)
                delirium_tx_records.append({
                    'stay_id': pid,
                    'time': treatment_start,
                    'delirium_tx': 1
                })
    data['delirium_tx'] = pd.DataFrame(delirium_tx_records) if delirium_tx_records else pd.DataFrame(columns=['stay_id', 'time', 'delirium_tx'])
    
    # 6. adv_resp: 高级呼吸支持（机械通气 + PEEP > 5）
    adv_resp_records = []
    if 'vent_ind' in data and not data['vent_ind'].empty and 'peep' in data and not data['peep'].empty:
        # 合并 vent_ind 和 peep
        vent_peep = pd.merge(
            data['vent_ind'],
            data['peep'],
            on=['stay_id', 'time'],
            how='inner'
        )
        for _, row in vent_peep.iterrows():
            # 高级呼吸支持 = 机械通气 + PEEP > 5
            is_adv = 1 if (row['vent_ind'] == 1 and row['peep'] > 5) else 0
            adv_resp_records.append({
                'stay_id': row['stay_id'],
                'time': row['time'],
                'adv_resp': is_adv
            })
    data['adv_resp'] = pd.DataFrame(adv_resp_records) if adv_resp_records else pd.DataFrame(columns=['stay_id', 'time', 'adv_resp'])
    
    # 7. other_vaso: 其他血管活性药物（不包括常见的norepi/epi/dopa/dobu）
    # 示例：血管加压素(vasopressin)、去甲肾上腺素(phenylephrine)等
    other_vaso_records = []
    if 'phn_rate' in data and not data['phn_rate'].empty:
        data['other_vaso'] = data['phn_rate'].copy()
        data['other_vaso'] = data['other_vaso'].rename(columns={'phn_rate': 'other_vaso'})
        data['other_vaso']['other_vaso'] = (data['other_vaso']['other_vaso'] > 0).astype(int)
    else:
        # 生成少量记录（约10%患者）
        for pid in patient_ids:
            if np.random.random() < 0.1:
                start_time = np.random.uniform(6, 48)
                for t in range(int(start_time), min(72, int(start_time + 24)), 4):
                    other_vaso_records.append({
                        'stay_id': pid,
                        'time': float(t),
                        'other_vaso': 1
                    })
        data['other_vaso'] = pd.DataFrame(other_vaso_records) if other_vaso_records else pd.DataFrame(columns=['stay_id', 'time', 'other_vaso'])
    
    # 🔧 已删除 sep3 别名概念（2026-02-13）：不再提取 sep3 特征
    # 保留 sep3_sofa1 和 sep3_sofa2 作为独立的 Sepsis-3 诊断概念
    
    return data, patient_ids


def render_quick_visualization_page():
    """渲染快速可视化主页面 - 包含数据加载区域和四个子模块。"""
    lang = st.session_state.get('language', 'en')
    entry_mode = st.session_state.get('entry_mode', 'none')

    _viz_title = get_text('quick_viz')
    hint_text = (
        "Generate demo data or load previously exported result files for interactive analysis"
        if entry_mode == 'demo'
        else "Load previously exported result files for interactive analysis"
    )
    if lang != 'en':
        hint_text = "生成模拟数据或从之前导出的结果文件中加载，进行交互式分析" if entry_mode == 'demo' else "从之前导出的结果文件中加载，进行交互式分析"

    st.markdown(
        f'''
        <div class="compact-section-title">{_viz_title}</div>
        <div class="compact-section-desc">{hint_text}</div>
        ''',
        unsafe_allow_html=True,
    )

    viz_notices = st.session_state.pop('_viz_notices', [])
    for notice in viz_notices[:3]:
        level = str(notice.get('level') or 'info')
        message = str(notice.get('message') or '').strip()
        if message:
            st.markdown(
                f'<div class="compact-inline-notice {level}">{message}</div>',
                unsafe_allow_html=True,
            )

    data_loaded = len(st.session_state.loaded_concepts) > 0
    if 'viz_export_path' not in st.session_state:
        st.session_state.viz_export_path = ""
    recent_export_path = st.session_state.get('viz_confirmed_path') or st.session_state.get('last_export_dir') or ""
    if recent_export_path and (st.session_state.get('_prefer_exported_viz') or not st.session_state.get('viz_export_path')):
        st.session_state.viz_export_path = recent_export_path
        st.session_state['viz_export_path_input'] = recent_export_path
        st.session_state['viz_data_source_mode'] = "exported"
        st.session_state['_prefer_exported_viz'] = False

    expander_label = "⚙️ Data Loading Settings" if lang == 'en' else "⚙️ 数据加载设置"
    with st.expander(expander_label, expanded=not data_loaded):
        allow_demo = entry_mode != 'real'
        source_options = ["exported"] + (["demo"] if allow_demo else [])
        source_labels = {
            "exported": "📁 Previously Exported Data" if lang == 'en' else "📁 加载之前导出的结果文件",
            "demo": "🧪 Demo Data" if lang == 'en' else "🧪 模拟数据",
        }
        default_source = "exported" if recent_export_path else ("demo" if allow_demo and entry_mode == 'demo' else "exported")
        if 'viz_data_source_mode' not in st.session_state:
            st.session_state.viz_data_source_mode = default_source
        elif st.session_state.get('viz_data_source_mode') not in source_options:
            st.session_state.viz_data_source_mode = default_source
        current_source = st.radio(
            "Data Source" if lang == 'en' else "数据来源",
            options=source_options,
            index=source_options.index(st.session_state.get('viz_data_source_mode', default_source)),
            format_func=lambda value: source_labels[value],
            horizontal=True,
            key="viz_data_source_mode",
        )

        if current_source == "exported":
            export_path = _directory_input(
                "Folder Containing Exported Data Files" if lang == 'en' else "存放导出结果文件的文件夹",
                value=st.session_state.get('viz_export_path') or recent_export_path,
                input_key="viz_export_path_input",
                button_key="viz_export_path_browse",
                help="Choose the folder that contains EasyICU exported CSV / Parquet / Excel files" if lang == 'en' else "选择存放 EasyICU 导出 CSV / Parquet / Excel 文件的文件夹",
            )
            st.session_state.viz_export_path = export_path

            if export_path:
                export_dir = Path(export_path)
                if export_dir.exists() and export_dir.is_dir():
                    available_files = sorted(
                        list(export_dir.glob('*.csv'))
                        + list(export_dir.glob('*.parquet'))
                        + list(export_dir.glob('*.xlsx')),
                        key=lambda path: path.name,
                    )
                    file_names = list(dict.fromkeys(file.stem for file in available_files))

                    if file_names:
                        st.success(
                            f"✅ Found {len(file_names)} data files" if lang == 'en' else f"✅ 发现 {len(file_names)} 个数据文件"
                        )
                        selected_files = st.multiselect(
                            "Select Tables to Load" if lang == 'en' else "选择要加载的表格",
                            options=file_names,
                            default=file_names,
                            key="viz_selected_files",
                        )

                        patient_options = [50, 100, 200, 500, -1]
                        option_labels = {
                            50: "50 (Fast)" if lang == 'en' else "50 (快速)",
                            100: "100 (Recommended)" if lang == 'en' else "100 (推荐)",
                            200: "200",
                            500: "500 (Slow)" if lang == 'en' else "500 (较慢)",
                            -1: "All (May Lag)" if lang == 'en' else "全部 (可能卡顿)",
                        }
                        max_patients_opt = st.selectbox(
                            "Max Patients to Load" if lang == 'en' else "最大加载患者数",
                            options=patient_options,
                            index=1,
                            format_func=lambda value: option_labels[value],
                            key="viz_max_patients",
                        )
                        max_patients = None if max_patients_opt == -1 else max_patients_opt

                        if selected_files:
                            if st.button(
                                "🔍 Load Data" if lang == 'en' else "🔍 加载数据",
                                type="primary",
                                use_container_width=True,
                                key="viz_load_files",
                            ):
                                with st.spinner("Loading data..." if lang == 'en' else "正在加载数据..."):
                                    load_from_exported(export_path, max_patients=max_patients, selected_files=selected_files)
                                st.rerun()
                        else:
                            st.warning("⚠️ Please select at least one file" if lang == 'en' else "⚠️ 请至少选择一个文件")
                    else:
                        st.warning(
                            "⚠️ No data files found in this directory (CSV/Parquet/Excel)"
                            if lang == 'en'
                            else "⚠️ 该目录下未找到数据文件 (CSV/Parquet/Excel)"
                        )
                else:
                    st.error("❌ Directory does not exist" if lang == 'en' else "❌ 目录不存在")

        elif current_source == "demo":
            st.info(
                "✨ Generate ALL simulated ICU features for full exploration"
                if lang == 'en'
                else "✨ 生成全部模拟 ICU 特征供完整体验"
            )

            col1, col2 = st.columns(2)
            with col1:
                n_patients = st.slider(
                    "Number of Patients" if lang == 'en' else "患者数量",
                    10,
                    200,
                    50,
                    key="viz_demo_patients",
                )
            with col2:
                hours = st.slider(
                    "Data Duration (hours)" if lang == 'en' else "数据时长(小时)",
                    24,
                    168,
                    72,
                    key="viz_demo_hours",
                )

            feature_hint = (
                "Will generate ~160+ features across all modules (Vitals, Labs, SOFA, Sepsis, AKI, etc.)"
                if lang == 'en'
                else "将生成约160+个特征，覆盖所有模块（生命体征、实验室、SOFA、脓毒症、AKI等）"
            )
            st.caption(f"💡 {feature_hint}")

            if st.button(
                "🚀 Generate & Load All Demo Data" if lang == 'en' else "🚀 生成并加载全部模拟数据",
                type="primary",
                use_container_width=True,
                key="viz_load_demo",
            ):
                with st.spinner(
                    "Generating all mock data (~160+ features)..." if lang == 'en' else "正在生成全部模拟数据（约160+特征）..."
                ):
                    params = get_mock_params_with_cohort()
                    params['n_patients'] = n_patients
                    params['hours'] = hours
                    mock_data, patient_ids = generate_mock_data(**params)
                    st.session_state.loaded_concepts = mock_data
                    st.session_state.loaded_data_origin = 'demo_viz'
                    st.session_state.patient_ids = patient_ids
                    st.session_state.id_col = 'stay_id'
                    st.session_state.time_col = 'time'
                    st.session_state.selected_concepts = list(mock_data.keys())
                st.rerun()

    if data_loaded:
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            get_text('review_tables'),
            get_text('review_trends'),
            get_text('review_patients'),
            get_text('review_quality'),
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
        no_data_msg = f"""
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 16px; margin: 20px 0;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
            <h3 style="color: #495057; margin-bottom: 10px;">{"No Data Loaded" if lang == 'en' else "尚未加载数据"}</h3>
            <p style="color: #6c757d;">{"Please configure data source above and click Load button" if lang == 'en' else "请在上方配置数据来源，然后点击加载按钮"}</p>
        </div>
        """
        st.markdown(no_data_msg, unsafe_allow_html=True)


def render_entry_page():
    """渲染入口选择页面 - Premium Hero 设计"""
    lang = st.session_state.get('language', 'en')

    # 语言切换（右上角, 紧凑）
    col_lang = st.columns([8, 1])[1]
    with col_lang:
        lang_select = st.selectbox(
            "🌐",
            options=['EN', 'ZH'],
            index=0 if lang == 'en' else 1,
            key="entry_lang_select",
            label_visibility="collapsed"
        )
        if (lang_select == 'EN' and lang != 'en') or (lang_select == 'ZH' and lang != 'zh'):
            st.session_state.language = 'en' if lang_select == 'EN' else 'zh'
            st.rerun()

    # ===== Hero Section =====
    _hero_title = "EasyICU" if lang == 'en' else "EasyICU"
    _hero_subtitle = "ICU Data Analytics Platform · Extract · Visualize · Export" if lang == 'en' else "ICU 数据分析平台 · 提取 · 可视化 · 导出"
    _hero_badge = "v1.0 · Open Source · 6 Databases" if lang == 'en' else "v1.0 · 开源 · 支持 6 大数据库"

    st.markdown(f"""
    <div class="hero-container animate-fade-in">
        <div class="hero-badge">{_hero_badge}</div>
        <div class="hero-title">🏥 {_hero_title}</div>
        <div class="hero-subtitle">{_hero_subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

    # ===== 入口按钮 CSS（保持 Streamlit 按钮可点击，纯 CSS 装饰） =====
    st.markdown("""
    <style>
    /* 入口按钮 — 深色玻璃卡片风格 */
    .entry-btn-wrap div[data-testid="stButton"] { height: 100%; }
    .entry-btn-wrap div[data-testid="stButton"] > button {
        min-height: 220px !important;
        height: 100% !important;
        padding: 2.5rem 2rem !important;
        font-size: 1.15rem !important;
        white-space: pre-line !important;
        line-height: 1.7 !important;
        border-radius: var(--radius-xl) !important;
        transition: var(--transition-smooth) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        text-align: center !important;
        font-weight: 500 !important;
        position: relative !important;
        backdrop-filter: blur(12px) !important;
    }
    .entry-btn-wrap div[data-testid="stButton"] > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.12) !important;
    }
    .entry-btn-wrap div[data-testid="stButton"] > button:active {
        transform: translateY(-1px) !important;
    }
    /* 左列 Demo = 柔和浅蓝 */
    .entry-btn-wrap.demo-col div[data-testid="stButton"] > button {
        background: #7ea9cd !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.26) !important;
        box-shadow: 0 4px 20px rgba(76,112,150,0.18) !important;
    }
    .entry-btn-wrap.demo-col div[data-testid="stButton"] > button:hover {
        background: #739dc0 !important;
        box-shadow: 0 8px 28px rgba(76,112,150,0.24) !important;
    }
    /* 右列 Real = 浅灰蓝 */
    .entry-btn-wrap.real-col div[data-testid="stButton"] > button {
        background: #f5f8fc !important;
        color: #1f2937 !important;
        border: 1px solid #d7e2ee !important;
        box-shadow: 0 4px 20px rgba(15,23,42,0.05) !important;
    }
    .entry-btn-wrap.real-col div[data-testid="stButton"] > button:hover {
        background: #eef4fa !important;
        box-shadow: 0 8px 24px rgba(15,23,42,0.08) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===== 模式选择提示 =====
    _choose = "Choose Your Mode" if lang == 'en' else "选择使用模式"
    st.markdown(f"<p style='text-align:center;color:var(--text-secondary-light);font-weight:600;font-size:0.82rem;letter-spacing:0.06em;text-transform:uppercase;margin:1.5rem 0 1rem;'>{_choose}</p>", unsafe_allow_html=True)

    # ===== 两列按钮（居中容器） =====
    _, col1, col2, _ = st.columns([1, 4, 4, 1], gap="large")

    with col1:
        st.markdown('<div class="entry-btn-wrap demo-col">', unsafe_allow_html=True)
        if lang == 'en':
            demo_label = "🧪\n\nDemo Mode\n\nExplore with simulated ICU data\nNo database required\n\n✨ Quick Start"
        else:
            demo_label = "🧪\n\n演示模式\n\n使用模拟 ICU 数据体验全部功能\n无需真实数据库\n\n✨ 快速开始"
        if st.button(demo_label, key="entry_demo_btn", use_container_width=True, type="primary"):
            st.session_state.entry_mode = 'demo'
            st.session_state.use_mock_data = True
            st.session_state.database = 'mock'
            st.session_state.loaded_concepts = {}
            st.session_state.loaded_data_origin = 'none'
            st.session_state.patient_ids = []
            for key in ['group_a_data', 'group_b_data', 'multidb_data', 'dash_demographics',
                        'multidb_is_demo', 'dash_is_demo', 'cohort_is_demo']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="entry-btn-wrap real-col">', unsafe_allow_html=True)
        if lang == 'en':
            real_label = "📊\n\nReal Data Mode\n\nConnect to local ICU databases\nMIMIC · eICU · AUMC · HiRID · SICdb\n\n🔬 Research Ready"
        else:
            real_label = "📊\n\n真实数据模式\n\n连接本地 ICU 数据库\nMIMIC · eICU · AUMC · HiRID · SICdb\n\n🔬 科研就绪"
        if st.button(real_label, key="entry_real_btn", use_container_width=True, type="secondary"):
            st.session_state.entry_mode = 'real'
            st.session_state.use_mock_data = False
            st.session_state.loaded_concepts = {}
            st.session_state.loaded_data_origin = 'none'
            st.session_state.patient_ids = []
            for key in ['group_a_data', 'group_b_data', 'multidb_data', 'dash_demographics',
                        'multidb_is_demo', 'dash_is_demo', 'cohort_is_demo']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    try:
        import streamlit_shadcn_ui as ui
    except Exception:
        ui = None

    if lang == 'en':
        lead_text = "EasyICU connects cohort design, standardized concept extraction, and review-ready analytics in one local-first ICU workflow."
        stat_pills = ["6 ICU databases", "167 clinical concepts", "Local-first processing", "Demo + Real Data", "AI-assisted workflow"]
        tab_options = ["Workflow", "Clinical layer", "Execution"]
        tab_default = "Workflow"
        dbs_label = "Supported Databases"
        overview = {
            "Workflow": (
                "User-facing workflow",
                "Move from study design to analysis-ready data without leaving the same interface.",
                [
                    ("🧲", "Cohort design", "Filter by age, ICU stay, outcomes, and study constraints."),
                    ("🧩", "Concept selection", "Choose from 167 ICU features across 19 clinical groups."),
                    ("👁️", "Visual review", "Inspect time series, patient view, quality reports, and cohorts."),
                ],
            ),
            "Clinical layer": (
                "Clinical intelligence",
                "Standardized concepts and computable rules sit behind every extraction step.",
                [
                    ("📚", "Standardized concept library", "One interface spanning all 6 supported public ICU databases."),
                    ("🧠", "Computable rules & scores", "SOFA, SOFA-2, Sepsis-3, KDIGO-AKI, qSOFA and more."),
                    ("🤖", "AI assistant", "Guided setup, troubleshooting, concept planning, and evidence-backed answers."),
                ],
            ),
            "Execution": (
                "Execution engine",
                "Fast, reproducible extraction with temporal alignment and scalable processing controls.",
                [
                    ("⚙️", "Structured extraction", "Export module-aware datasets to CSV, Parquet, or Excel."),
                    ("⏱️", "Temporal harmonization", "Align intervals and time axes across heterogeneous ICU tables."),
                    ("🚀", "Caching & parallel processing", "Selective access, memory control, and faster multi-module runs."),
                ],
            ),
        }
    else:
        lead_text = "EasyICU 将队列设计、标准化概念提取和复核分析整合为一个本地优先的 ICU 数据工作流。"
        stat_pills = ["支持 6 大 ICU 数据库", "167 个临床概念", "本地优先处理", "演示 + 真实数据模式", "AI 辅助工作流"]
        tab_options = ["用户工作流", "临床智能层", "执行能力层"]
        tab_default = "用户工作流"
        dbs_label = "支持的数据库"
        overview = {
            "用户工作流": (
                "用户工作流",
                "从研究设计到分析数据导出，尽量在同一界面内完成。",
                [
                    ("🧲", "队列设计", "按年龄、ICU 住院时长、结局和研究条件筛选患者。"),
                    ("🧩", "概念选择", "从 19 个分组、167 个 ICU 特征中选择目标变量。"),
                    ("👁️", "可视化复核", "查看时序图、患者视图、质量报告和队列分析。"),
                ],
            ),
            "临床智能层": (
                "临床智能层",
                "标准化概念和可计算规则支撑提取结果的一致性与可解释性。",
                [
                    ("📚", "标准化概念库", "统一接口覆盖 6 个公开 ICU 数据库。"),
                    ("🧠", "可计算规则与评分", "内置 SOFA、SOFA-2、Sepsis-3、KDIGO-AKI、qSOFA 等。"),
                    ("🤖", "AI 助手", "引导配置、报错排查、特征规划和证据支持回答。"),
                ],
            ),
            "执行能力层": (
                "执行能力层",
                "提供结构化导出、时间统一和更适合大数据的执行机制。",
                [
                    ("⚙️", "结构化提取", "按模块导出为 CSV、Parquet 或 Excel。"),
                    ("⏱️", "时间维度统一", "处理多数据库时间轴、间隔和患者级对齐。"),
                    ("🚀", "缓存与并行", "选择性读取、内存控制和更快的多模块运行。"),
                ],
            ),
        }

    st.markdown(f'<div class="entry-overview animate-fade-in">', unsafe_allow_html=True)

    if ui is not None:
        ui.badges(
            badge_list=[(item, "secondary") for item in stat_pills],
            class_name="flex flex-wrap gap-2 justify-center",
            key=f"entry_badges_{lang}",
        )
        selected_tab = ui.tabs(
            options=tab_options,
            default_value=tab_default,
            key=f"entry_tabs_{lang}",
        )
    else:
        selected_tab = st.radio(
            "",
            options=tab_options,
            index=tab_options.index(tab_default),
            horizontal=True,
            label_visibility="collapsed",
            key=f"entry_tabs_fallback_{lang}",
        )

    panel_title, panel_subtitle, panel_items = overview[selected_tab]
    items_html = ""
    for icon, title, desc in panel_items:
        ai_class = " ai" if ("AI" in title or "助手" in title) else ""
        items_html += f"""
        <div class="entry-overview-item{ai_class}">
            <span class="entry-overview-icon">{icon}</span>
            <div>
                <div class="entry-overview-item-title">{title}</div>
                <div class="entry-overview-item-desc">{desc}</div>
            </div>
        </div>"""

    st.markdown(f"""
    <div class="entry-overview-panel">
        <div class="entry-overview-head">
            <div>
                <div class="entry-overview-title">{panel_title}</div>
                <div class="entry-overview-subtitle">{panel_subtitle}</div>
            </div>
            <div class="entry-overview-kicker">EasyICU</div>
        </div>
        <div class="entry-overview-grid">{items_html}</div>
        <div class="entry-db-inline">
            <div class="entry-db-inline-label">{dbs_label}</div>
            <div class="entry-db-inline-list">
                <span>MIMIC-IV</span><span>MIMIC-III</span><span>eICU-CRD</span><span>AmsterdamUMCdb</span><span>HiRID</span><span>SICdb</span>
            </div>
        </div>
    </div>
    <div class="entry-overview-lead">{lead_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """渲染侧边栏 - 根据entry_mode显示不同内容。"""
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
                if st.button(convert_btn, width="stretch", type="primary", key="convert_csv"):
                    st.session_state.show_convert_dialog = True
                    st.session_state.convert_source_path = data_path
                    st.rerun()
                convert_hint = "💡 One-click: convert → validate → ready" if st.session_state.language == 'en' else "💡 一键完成：转换 → 验证 → 就绪"
                st.caption(convert_hint)
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
                    # 🔧 ADD (2026-02-05): 添加"不限制"选项
                    no_limit_min_label = "No Limit" if st.session_state.language == 'en' else "不限制"
                    age_min_no_limit = st.checkbox(no_limit_min_label, value=st.session_state.cohort_filter['age_min'] is None, key="cohort_age_min_no_limit")
                    if age_min_no_limit:
                        st.session_state.cohort_filter['age_min'] = None
                        st.caption("✓ " + ("No minimum age limit" if st.session_state.language == 'en' else "无最小年龄限制"))
                    else:
                        age_min = st.number_input(
                            age_min_label, min_value=0, max_value=120, 
                            value=18 if st.session_state.cohort_filter['age_min'] is None else int(st.session_state.cohort_filter['age_min']),
                            key="cohort_age_min"
                        )
                        st.session_state.cohort_filter['age_min'] = age_min if age_min > 0 else None
                with age_col2:
                    age_max_label = "Max Age" if st.session_state.language == 'en' else "最大年龄"
                    # 🔧 ADD (2026-02-05): 添加"不限制"选项
                    no_limit_max_label = "No Limit" if st.session_state.language == 'en' else "不限制"
                    age_max_no_limit = st.checkbox(no_limit_max_label, value=st.session_state.cohort_filter['age_max'] is None, key="cohort_age_max_no_limit")
                    if age_max_no_limit:
                        st.session_state.cohort_filter['age_max'] = None
                        st.caption("✓ " + ("No maximum age limit" if st.session_state.language == 'en' else "无最大年龄限制"))
                    else:
                        age_max = st.number_input(
                            age_max_label, min_value=0, max_value=120, 
                            value=100 if st.session_state.cohort_filter['age_max'] is None else int(st.session_state.cohort_filter['age_max']),
                            key="cohort_age_max"
                        )
                        st.session_state.cohort_filter['age_max'] = age_max if age_max < 120 else None
            
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
            # 🔧 ADD (2026-02-05): 显示 Min ICU Stay 筛选条件
            if cf.get('los_min') is not None:
                filter_summary.append(f"Min ICU Stay: {cf['los_min']}h" if st.session_state.language == 'en' else f"最短住院: {cf['los_min']}小时")
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
                preview_toggle_label = "Enable quick preview before export (optional)" if st.session_state.language == 'en' else "启用导出前快速预览（可选）"
                preview_toggle_help = "Preview is optional. Turn this on only when you want to inspect a small sample first." if st.session_state.language == 'en' else "预览不是必需步骤。只有在你想先看一小部分样本时再开启。"
                preview_enabled = st.toggle(
                    preview_toggle_label,
                    value=st.session_state.get('sidebar_preview_enabled', False),
                    key="sidebar_preview_enabled",
                    help=preview_toggle_help,
                )

                if preview_enabled:
                    preview_desc = "Preview a small sample before full extraction" if st.session_state.language == 'en' else "在全量提取前预览一小部分患者数据"
                    st.caption(preview_desc)
                    if 'preview_n_patients' not in st.session_state:
                        st.session_state.preview_n_patients = 10
                    
                    preview_n = st.select_slider(
                        get_text('preview_patients'),
                        options=[10, 20, 50, 100],
                        value=10,
                        key="preview_n_patients"
                    )

                    if st.session_state.get('_preview_requested', False):
                        preview_loading_msg = (
                            f"⏳ Preview request received. Loading {st.session_state.get('_preview_n', 10)} patients now..."
                            if st.session_state.language == 'en' else
                            f"⏳ 已收到预览请求，正在加载 {st.session_state.get('_preview_n', 10)} 位患者，请稍候..."
                        )
                        st.info(preview_loading_msg)
                    
                    if st.button(get_text('preview_btn'), key="sidebar_preview_btn", use_container_width=True):
                        st.session_state['_preview_requested'] = True
                        st.session_state['_preview_n'] = preview_n
                        st.session_state['_scroll_to_tab'] = 'viz'
                        st.rerun()
                else:
                    optional_msg = "Skip preview if you already know the configuration and want to export directly." if st.session_state.language == 'en' else "如果你已经确认配置无误，可以跳过预览，直接导出。"
                    st.caption(optional_msg)
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
        default_export_path = str(Path(base_export_path) / f"{db_name}_{timestamp_suffix}")
        
        export_path = _directory_input(
            "Export Path" if st.session_state.language == 'en' else "导出路径",
            value=default_export_path,
            input_key="sidebar_export_path_input",
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


def load_from_exported(export_dir: str, max_patients: int = 100, selected_files: list = None):
    """从已导出的数据文件加载数据（限制患者数用于快速预览）。
    
    从宽表中提取每个特征列，使其可以单独选择和可视化。
    
    Args:
        export_dir: 导出目录路径
        max_patients: 最大患者数限制（默认100）
        selected_files: 要加载的文件名列表（不含扩展名），None表示全部加载
    """
    try:
        import time
        load_start = time.time()
        
        export_path = Path(export_dir)
        raw_data = {}  # 原始文件数据
        
        # ID列和时间列，不作为特征
        id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 
                        'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        time_candidates = ['time', 'charttime', 'starttime', 'endtime', 
                          'datetime', 'timestamp', 'index', 'Offset',
                          'measuredat_minutes', 'measuredat']
        exclude_cols = set(id_candidates + time_candidates)
        
        # 扫描并加载选中的数据文件
        for file in export_path.iterdir():
            file_stem = file.stem
            
            # 如果指定了文件列表，只加载选中的
            if selected_files is not None and file_stem not in selected_files:
                continue
            
            if file.suffix == '.csv':
                df = pd.read_csv(file)
                raw_data[file_stem] = df
            elif file.suffix == '.parquet':
                df = pd.read_parquet(file)
                raw_data[file_stem] = df
            elif file.suffix == '.xlsx':
                df = pd.read_excel(file)
                raw_data[file_stem] = df
        
        if not raw_data:
            lang = st.session_state.get('language', 'en')
            warn_msg = "⚠️ No valid data files found" if lang == 'en' else "⚠️ 未找到有效的数据文件"
            st.warning(warn_msg)
            return
        
        # 从宽表中提取每个特征列作为单独的concept
        data = {}
        
        # 首先确定全局ID列（用于患者筛选）
        id_col_found = 'stay_id'
        for file_name, df in raw_data.items():
            if isinstance(df, pd.DataFrame):
                for col in id_candidates:
                    if col in df.columns:
                        id_col_found = col
                        break
                break
        
        # 从每个宽表中提取特征列
        # 注意：每个文件可能有不同的时间列，需要单独检测
        # 🔧 2026-02-12: 添加列名规范化和去重逻辑
        for file_name, df in raw_data.items():
            if isinstance(df, pd.DataFrame):
                # 为当前文件找时间列（每个文件单独检测）
                file_time_col = None
                for col in time_candidates:
                    if col in df.columns:
                        file_time_col = col
                        break
                
                # 获取特征列（排除ID列、时间列和元数据列如_concept）
                meta_cols = {'_concept'}
                feature_cols = [c for c in df.columns if c not in exclude_cols and c not in meta_cols]
                
                # 为每个特征创建单独的DataFrame
                for feat_col in feature_cols:
                    # 🔧 规范化列名（去重）
                    normalized_col = normalize_column_name(feat_col)
                    
                    # 如果规范化后的列名已存在，跳过（保留第一个遇到的）
                    if normalized_col in data:
                        continue
                    
                    # 保留ID列、该文件的时间列和该特征列
                    keep_cols = []
                    if id_col_found in df.columns:
                        keep_cols.append(id_col_found)
                    if file_time_col and file_time_col in df.columns:
                        keep_cols.append(file_time_col)
                    keep_cols.append(feat_col)
                    
                    feat_df = df[keep_cols].copy()
                    # 🔧 重命名特征列为规范化后的名称
                    if feat_col != normalized_col:
                        feat_df = feat_df.rename(columns={feat_col: normalized_col})
                    data[normalized_col] = feat_df
        
        # 获取患者列表
        patient_ids = set()
        
        for concept_df in data.values():
            if isinstance(concept_df, pd.DataFrame):
                if id_col_found in concept_df.columns:
                    patient_ids.update(concept_df[id_col_found].unique())
        
        all_patient_count = len(patient_ids)
        
        # 限制患者数用于可视化预览（max_patients=None 表示加载全部）
        if max_patients is None or max_patients <= 0:
            preview_patient_ids = sorted(list(patient_ids))
            is_limited = False
        else:
            preview_patient_ids = sorted(list(patient_ids))[:max_patients]
            is_limited = all_patient_count > max_patients
        
        # 筛选数据只保留限制的患者
        filtered_data = {}
        for concept_name, df in data.items():
            if isinstance(df, pd.DataFrame) and id_col_found in df.columns:
                filtered_df = df[df[id_col_found].isin(preview_patient_ids)]
                # 即使DataFrame为空也保留，确保特征数量一致
                filtered_data[concept_name] = filtered_df
            else:
                # 对于没有ID列的DataFrame（如静态指标），直接保留
                filtered_data[concept_name] = df
        
        st.session_state.loaded_concepts = filtered_data
        st.session_state.loaded_data_origin = 'exported_files'
        st.session_state.patient_ids = preview_patient_ids
        st.session_state.all_patient_count = all_patient_count
        st.session_state.id_col = id_col_found
        
        # 🔧 FIX (2026-02-03): 设置 selected_concepts 以便侧边栏的导出按钮可用
        st.session_state.selected_concepts = list(filtered_data.keys())
        
        # 🔧 FIX (2026-02-12): 规范化后每列就是一个概念，直接统计列数
        # 由于在加载时已经去重，这里直接使用 len(filtered_data)
        unique_concept_count = len(filtered_data)
        
        # 🔧 FIX (2026-02-03): Load Data后重置导出触发状态，避免白屏
        # 注意：不应该重置 export_completed，因为 Quick Visualization 的 Load Data
        # 是独立于侧边栏数据提取器的功能，不应该影响导出完成状态
        st.session_state.trigger_export = False
        st.session_state['_exporting_in_progress'] = False
        # 清理跳过/覆盖模块状态（这些是导出过程中的临时状态，可以安全清理）
        if '_skipped_modules' in st.session_state:
            del st.session_state['_skipped_modules']
        if '_overwrite_modules' in st.session_state:
            del st.session_state['_overwrite_modules']
        if '_viz_import_export_auto_trigger' in st.session_state:
            del st.session_state['_viz_import_export_auto_trigger']
        
        load_elapsed = time.time() - load_start
        
        # 显示提示信息
        # 🔧 FIX (2026-02-12): 规范化后 concepts = columns (已去重)
        lang = st.session_state.get('language', 'en')
        if lang == 'en':
            st.success(f"✅ Loaded {unique_concept_count} concepts, {len(preview_patient_ids)}/{all_patient_count} patients ({load_elapsed:.1f}s)")
            if is_limited:
                st.info(f"💡 For better performance, preview is limited to {max_patients} patients. Full data has been exported to disk.")
        else:
            st.success(f"✅ 已加载 {unique_concept_count} 个概念，{len(preview_patient_ids)}/{all_patient_count} 个患者 ({load_elapsed:.1f}秒)")
            if is_limited:
                st.info(f"💡 为保证流畅性，可视化预览仅加载前 {max_patients} 个患者。完整数据已导出到磁盘，可使用Python/R进行完整分析。")
        
    except Exception as e:
        lang = st.session_state.get('language', 'en')
        err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
        st.error(err_msg)


def load_data():
    """Load data with parallel acceleration support - optimized batch loading."""
    lang = st.session_state.get('language', 'en')
    
    if not st.session_state.data_path:
        err_msg = "Please set data path first" if lang == 'en' else "请先设置数据路径"
        st.error(err_msg)
        return
    
    if not st.session_state.selected_concepts:
        err_msg = "Please select at least one concept" if lang == 'en' else "请选择至少一个 Concept"
        st.error(err_msg)
        return
    
    # 显示加载提示
    n_selected = len(st.session_state.selected_concepts)
    if lang == 'en':
        st.info(f"⏳ Loading {n_selected} features in batch mode, please wait...")
        spinner_msg = "Batch loading data, please wait..."
    else:
        st.info(f"⏳ 批量加载 {n_selected} 个特征数据，请稍候...")
        spinner_msg = "正在批量加载数据，请稍候..."
    
    with st.spinner(spinner_msg):
        try:
            # 动态导入以避免循环导入
            from easyicu import load_concepts
            import time
            import os
            
            concepts_list = st.session_state.selected_concepts
            n_concepts = len(concepts_list)
            
            load_start = time.time()
            
            # 🚀 优化：真正的批量加载 - 一次调用加载所有concepts
            # 🚀 性能优化：参照 extract_baseline_features.py 的配置
            patient_limit = st.session_state.get('patient_limit', 0)
            
            # 获取数据库信息
            data_path = Path(st.session_state.data_path)
            database = st.session_state.get('database', 'miiv')
            id_col_map = {
                'miiv': 'stay_id',
                'eicu': 'patientunitstayid', 
                'aumc': 'admissionid',
                'hirid': 'patientid',
                'mimic': 'icustay_id',
                'sic': 'CaseID'
            }
            id_col = id_col_map.get(database, 'stay_id')
            patient_ids_filter = None
            
            # 👥 先从数据库选patient_limit个患者，再对这些患者做人群筛选
            try:
                # Step 1: 先选patient_limit个患者作为候选集
                candidate_ids = None
                if patient_limit and patient_limit > 0:
                    try:
                        icustays_files = _get_patient_id_table_files(database)
                        for f in icustays_files:
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_patient_ids = icustays_df[id_col].unique().tolist()
                                    candidate_ids = _sample_patient_ids_random(all_patient_ids, patient_limit)
                                    break
                    except Exception:
                        pass
                
                # Step 2: 在候选集上应用人群筛选
                data_path_for_cohort = st.session_state.data_path
                database_for_cohort = st.session_state.get('database', 'miiv')
                cohort_result = apply_cohort_filter(data_path_for_cohort, database_for_cohort, candidate_ids=candidate_ids)
                if cohort_result is not None:
                    cohort_id_col = cohort_result['id_col']
                    filtered_ids = cohort_result['filtered_ids']
                    id_col = cohort_id_col
                    patient_ids_filter = {id_col: filtered_ids}
                    # Save cohort stats for completion message
                    n_before = len(candidate_ids) if candidate_ids else 0
                    n_after = len(filtered_ids)
                    st.session_state['_cohort_stats'] = {
                        'before': n_before, 'after': n_after, 'excluded': n_before - n_after,
                        'filter_details': cohort_result.get('filter_details', []),
                    }
                    # 🚫 Zero patients after cohort filter — abort extraction
                    if n_after == 0:
                        lang = st.session_state.get('language', 'en')
                        details_parts = []
                        for fd in cohort_result.get('filter_details', []):
                            label = fd[0] if lang == 'en' else fd[1]
                            details_parts.append(f"{label} (excluded {fd[2]})")
                        details_str = ", ".join(details_parts) if details_parts else ""
                        if lang == 'en':
                            msg = f"No patients meet the cohort criteria. {n_before} candidates were all excluded."
                            if details_str:
                                msg += f" Filters applied: {details_str}."
                            msg += " Please adjust your cohort filters in Step 2 and try again."
                        else:
                            msg = f"没有患者满足队列筛选条件。{n_before} 名候选患者全部被排除。"
                            if details_str:
                                msg += f" 已应用的筛选条件: {details_str}。"
                            msg += " 请在步骤2中调整筛选条件后重试。"
                        st.error(msg)
                        return
                elif candidate_ids is not None:
                    # No cohort filter active, use the candidate set directly
                    patient_ids_filter = {id_col: candidate_ids}
                    st.session_state['_cohort_stats'] = None
                else:
                    st.session_state['_cohort_stats'] = None
            except Exception as _cohort_err:
                print(f"[COHORT] Error in load_data: {_cohort_err}")
                # Fallback: just apply patient_limit without cohort filter
                if patient_limit and patient_limit > 0:
                    try:
                        icustays_files = _get_patient_id_table_files(database)
                        for f in icustays_files:
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_patient_ids = icustays_df[id_col].unique().tolist()
                                    sample_ids = _sample_patient_ids_random(all_patient_ids, patient_limit)
                                    patient_ids_filter = {id_col: sample_ids}
                                    break
                    except Exception:
                        pass
            
            # �🚀 智能并行配置：根据系统资源和患者数量动态调整
            num_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
            parallel_workers, parallel_backend = get_optimal_parallel_config(num_patients, task_type='load')
            
            try:
                # 🔧 按模块分组加载概念，显示进度条，跳过不可用/超时的概念
                data = {}
                failed_concepts = []
                empty_concepts = []  # 🆕 跟踪返回空结果的概念
                
                # 🚀 FIX 2026-02: 按模块分组加载 + 进度条 + 单模块超时
                # 解决全量AUMC (21K患者) 加载10+分钟用户感觉"卡住"的问题
                import signal
                
                # 将概念按模块分组
                concept_to_module = {}
                for mod_key, mod_concepts in CONCEPT_GROUPS_INTERNAL.items():
                    for c in mod_concepts:
                        concept_to_module[c] = mod_key
                
                module_concept_map = {}
                special_concepts_to_load = []
                for c in concepts_list:
                    if c in SPECIAL_CONCEPTS:
                        special_concepts_to_load.append(c)
                        continue
                    mod = concept_to_module.get(c, '_other')
                    if mod not in module_concept_map:
                        module_concept_map[mod] = []
                    module_concept_map[mod].append(c)
                
                # 模块加载优先级：快的先加载
                MODULE_LOAD_ORDER = [
                    'vitals', 'demographics', 'outcome',
                    'chemistry', 'hematology', 'blood_gas',
                    'medications', 'ventilator', 'respiratory',
                    'vasopressors', 'renal', 'neurological',
                    'other_scores', 'circulatory',
                    'sofa1_score', 'sofa2_score',
                    'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared',
                    '_other',
                ]
                ordered_modules = [m for m in MODULE_LOAD_ORDER if m in module_concept_map]
                for m in module_concept_map:
                    if m not in ordered_modules:
                        ordered_modules.append(m)
                
                total_modules = len(ordered_modules) + (1 if special_concepts_to_load else 0)
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # 单模块超时限制 (秒) - 防止单个模块阻塞整体流程
                MODULE_TIMEOUT = 300  # 5分钟
                
                def _process_load_result(result, concept_names):
                    """处理 load_concepts 返回结果（复用逻辑）"""
                    if isinstance(result, dict):
                        for cname, df in result.items():
                            if hasattr(df, 'to_pandas'):
                                df = df.to_pandas()
                            elif hasattr(df, 'dataframe'):
                                df = df.dataframe()
                            elif hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                                df = df.data
                            if isinstance(df, pd.DataFrame) and len(df) > 0:
                                data[cname] = df
                            elif isinstance(df, pd.Series):
                                data[cname] = df.to_frame().reset_index()
                            else:
                                empty_concepts.append(cname)
                    elif isinstance(result, pd.DataFrame):
                        if len(result) > 0:
                            for concept in concept_names:
                                data[concept] = result
                        else:
                            empty_concepts.extend(concept_names)
                    for c in concept_names:
                        if c not in data and c not in empty_concepts:
                            empty_concepts.append(c)
                
                for mod_idx, mod_key in enumerate(ordered_modules):
                    mod_concepts = module_concept_map[mod_key]
                    mod_display = CONCEPT_GROUP_NAMES.get(mod_key, (mod_key, mod_key))
                    mod_name = mod_display[1] if lang != 'en' else mod_display[0]
                    
                    elapsed = time.time() - load_start
                    progress_pct = (mod_idx) / total_modules
                    progress_placeholder.progress(progress_pct, text=f"{'Loading' if lang == 'en' else '正在加载'} {mod_name} ({mod_idx+1}/{total_modules}) - {elapsed:.0f}s")
                    status_placeholder.caption(f"📦 {mod_name}: {', '.join(mod_concepts[:5])}{'...' if len(mod_concepts) > 5 else ''}")
                    
                    mod_start = time.time()
                    try:
                        load_kwargs = {
                            'data_path': st.session_state.data_path,
                            'database': st.session_state.get('database'),
                            'concepts': mod_concepts,
                            'verbose': False,
                            'merge': False,
                            'concept_workers': 1,
                            'parallel_workers': parallel_workers,
                            'parallel_backend': parallel_backend,
                        }
                        if patient_ids_filter:
                            load_kwargs['patient_ids'] = patient_ids_filter
                        
                        # 🔧 FIX (2026-02-20): 使用后台线程执行，主线程每2秒发送UI更新保活WebSocket
                        # 解决全量患者导出时长时间无UI更新导致 "Connection timed out" 的问题
                        _thr_result = {'done': False, 'result': None, 'error': None}
                        
                        def _load_in_thread(fn, kw, holder):
                            try:
                                holder['result'] = fn(**kw)
                            except Exception as _e:
                                holder['error'] = _e
                            finally:
                                holder['done'] = True
                        
                        _t = threading.Thread(
                            target=_load_in_thread,
                            args=(load_concepts, load_kwargs, _thr_result),
                            daemon=True,
                        )
                        _t.start()
                        
                        # 主线程每2秒发送UI更新 → 保活WebSocket连接
                        _ka_tick = 0
                        _spinners = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
                        while not _thr_result['done']:
                            time.sleep(2)
                            _ka_tick += 1
                            _mod_elapsed = time.time() - mod_start
                            _total_elapsed = time.time() - load_start
                            _sp = _spinners[_ka_tick % 10]
                            if lang == 'en':
                                _ka_msg = f"{_sp} **Loading {mod_name}** ({len(mod_concepts)} concepts, module {mod_idx+1}/{total_modules}, elapsed {_total_elapsed:.0f}s, module {_mod_elapsed:.0f}s)"
                            else:
                                _ka_msg = f"{_sp} **正在加载 {mod_name}** ({len(mod_concepts)} 个概念, 模块 {mod_idx+1}/{total_modules}, 已用 {_total_elapsed:.0f}s, 本模块 {_mod_elapsed:.0f}s)"
                            status_placeholder.caption(_ka_msg)
                        
                        _t.join(timeout=5)
                        
                        if _thr_result['error']:
                            raise _thr_result['error']
                        
                        _process_load_result(_thr_result['result'], mod_concepts)
                        
                    except Exception as mod_e:
                        # 模块加载失败，逐个概念回退
                        for concept in mod_concepts:
                            try:
                                single_kwargs = {
                                    'data_path': st.session_state.data_path,
                                    'database': st.session_state.get('database'),
                                    'concepts': [concept],
                                    'verbose': False,
                                    'merge': False,
                                    'concept_workers': 1,
                                }
                                if patient_ids_filter:
                                    single_kwargs['patient_ids'] = patient_ids_filter
                                result = load_concepts(**single_kwargs)
                                _process_load_result(result, [concept])
                            except Exception:
                                failed_concepts.append(concept)
                                continue
                    
                    progress_placeholder.progress((mod_idx + 1) / total_modules, text=f"✅ {mod_name} ({mod_idx+1}/{total_modules})")
                
                # 加载特殊概念 (AKI, circ_failure, sep3) — 也用 threading 保活
                if special_concepts_to_load:
                    progress_placeholder.progress((len(ordered_modules)) / total_modules, text=f"{'Loading special concepts...' if lang == 'en' else '正在加载特殊概念...'}")
                    try:
                        _sp_result = {'done': False, 'result': None, 'error': None}
                        
                        def _load_special_thread(fn, kw, holder):
                            try:
                                holder['result'] = fn(**kw)
                            except Exception as _e:
                                holder['error'] = _e
                            finally:
                                holder['done'] = True
                        
                        _sp_kwargs = dict(
                            concepts=special_concepts_to_load,
                            database=st.session_state.get('database', 'miiv'),
                            data_path=st.session_state.data_path,
                            patient_ids=patient_ids_filter,
                            max_patients=patient_limit if patient_limit and patient_limit > 0 else None,
                            verbose=False,
                        )
                        
                        _sp_t = threading.Thread(
                            target=_load_special_thread,
                            args=(load_special_concepts, _sp_kwargs, _sp_result),
                            daemon=True,
                        )
                        _sp_t.start()
                        
                        _sp_tick = 0
                        _sp_start = time.time()
                        while not _sp_result['done']:
                            time.sleep(2)
                            _sp_tick += 1
                            _sp_elapsed = time.time() - _sp_start
                            _sp_char = _spinners[_sp_tick % 10]
                            _sp_msg = f"{_sp_char} {'Loading special concepts' if lang == 'en' else '正在加载特殊概念'}... {_sp_elapsed:.0f}s"
                            status_placeholder.caption(_sp_msg)
                        
                        _sp_t.join(timeout=5)
                        
                        if _sp_result['error']:
                            raise _sp_result['error']
                        
                        special_data = _sp_result['result']
                        for cname, df in special_data.items():
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                data[cname] = df
                        failed_special = [c for c in special_concepts_to_load if c not in data]
                        failed_concepts.extend(failed_special)
                    except Exception:
                        failed_concepts.extend(special_concepts_to_load)
                
                # 完成进度条
                progress_placeholder.progress(1.0, text=f"✅ {'Done!' if lang == 'en' else '加载完成!'}")
                status_placeholder.empty()
                
                if failed_concepts:
                    skip_msg = f"⚠️ Skipped {len(failed_concepts)} unavailable: {', '.join(failed_concepts[:5])}" if lang == 'en' else f"⚠️ 跳过 {len(failed_concepts)} 个不可用: {', '.join(failed_concepts[:5])}"
                    st.warning(skip_msg)
                
                # 🆕 显示空结果概念提示
                if empty_concepts:
                    empty_msg = f"ℹ️ {len(empty_concepts)} concepts returned empty (not configured or no data): {', '.join(empty_concepts[:8])}" if lang == 'en' else f"ℹ️ {len(empty_concepts)} 个概念返回空结果（未配置或无数据）: {', '.join(empty_concepts[:8])}"
                    st.info(empty_msg)
                    
            except Exception as batch_err:
                # 加载完全失败
                batch_err_msg = f"⚠️ Loading failed: {batch_err}" if lang == 'en' else f"⚠️ 加载失败: {batch_err}"
                st.warning(batch_err_msg)
                data = {}
            
            load_elapsed = time.time() - load_start
            
            if not data:
                warn_msg = "⚠️ Failed to load any data, please check data path and concept selection" if lang == 'en' else "⚠️ 未能加载任何数据，请检查数据路径和 Concept 选择"
                st.warning(warn_msg)
                return
            
            # 🔧 POST-FILTER: Remove patients whose cohort-critical features are None
            database = st.session_state.get('database', 'miiv')
            data = _post_filter_cohort_data(data, database)
            
            st.session_state.loaded_concepts = data
            st.session_state.loaded_data_origin = 'quick_load'
            
            # 获取患者列表 - 统计所有患者数，但UI选择器限制显示数量
            patient_ids = set()
            id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 
                           'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
            
            for concept_df in data.values():
                if isinstance(concept_df, pd.DataFrame):
                    for col in id_candidates:
                        if col in concept_df.columns:
                            patient_ids.update(concept_df[col].unique())
                            break
            
            # 保存完整患者列表用于统计，UI选择器用截断列表
            all_patient_ids = sorted(list(patient_ids))
            st.session_state.all_patient_count = len(all_patient_ids)  # 保存真实患者数
            st.session_state.patient_ids = all_patient_ids[:5000]  # UI选择器限制5000个
            
            if lang == 'en':
                st.success(f"✅ Loaded {len(data)} concepts, {len(all_patient_ids)} patients ({load_elapsed:.1f}s)")
            else:
                st.success(f"✅ 成功加载 {len(data)} 个 Concepts，{len(all_patient_ids)} 个患者 ({load_elapsed:.1f}秒)")
            
        except Exception as e:
            err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
            st.error(err_msg)


def load_data_for_preview(max_patients: int = 50):
    """Load limited data for preview visualization (memory-friendly version)."""
    lang = st.session_state.get('language', 'en')
    
    if not st.session_state.data_path:
        err_msg = "Please set data path first" if lang == 'en' else "请先设置数据路径"
        st.error(err_msg)
        return
    
    selected = st.session_state.get('selected_concepts', [])
    if not selected:
        err_msg = "Please select at least one feature" if lang == 'en' else "请选择至少一个特征"
        st.error(err_msg)
        return
    
    try:
        from easyicu import load_concepts
        import time
        
        load_start = time.time()
        data = {}
        
        # 只加载前5个concept作为预览
        preview_concepts = selected[:5]
        
        # 先选max_patients个患者，再对这些患者做人群筛选
        patient_ids_filter = None
        id_col = 'stay_id'
        try:
            data_path = Path(st.session_state.data_path)
            database = st.session_state.get('database', 'miiv')
            id_col_map = {'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid', 'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID'}
            id_col = id_col_map.get(database, 'stay_id')
            
            # Step 1: 先选max_patients个患者作为候选集
            candidate_ids = None
            for f in _get_patient_id_table_files(database):
                fp = data_path / f
                if fp.exists():
                    icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                    if id_col in icustays_df.columns:
                        candidate_ids = _sample_patient_ids_random(icustays_df[id_col].unique().tolist(), max_patients)
                        break
            
            # Step 2: 在候选集上应用人群筛选
            try:
                data_path_for_cohort = st.session_state.data_path
                database_for_cohort = st.session_state.get('database', 'miiv')
                cohort_result = apply_cohort_filter(data_path_for_cohort, database_for_cohort, candidate_ids=candidate_ids)
                if cohort_result is not None:
                    cohort_id_col = cohort_result['id_col']
                    filtered_ids = cohort_result['filtered_ids']
                    id_col = cohort_id_col
                    patient_ids_filter = {id_col: filtered_ids}
                    # 🚫 Zero patients after cohort filter — abort preview
                    if len(filtered_ids) == 0:
                        lang = st.session_state.get('language', 'en')
                        n_before = len(candidate_ids) if candidate_ids else 0
                        if lang == 'en':
                            st.error(f"No patients meet the cohort criteria. {n_before} candidates were all excluded. Please adjust your cohort filters in Step 2.")
                        else:
                            st.error(f"没有患者满足队列筛选条件。{n_before} 名候选患者全部被排除。请在步骤2中调整筛选条件。")
                        return
                elif candidate_ids is not None:
                    patient_ids_filter = {id_col: candidate_ids}
            except Exception as _cohort_err:
                print(f"[COHORT] Error in load_data_for_preview: {_cohort_err}")
                if candidate_ids is not None:
                    patient_ids_filter = {id_col: candidate_ids}
        except Exception:
            pass
        
        try:
            load_kwargs = {
                'data_path': st.session_state.data_path,
                'database': st.session_state.get('database'),
                'concepts': preview_concepts,
                'verbose': False,
                'merge': False,
                'concept_workers': 1,
                'parallel_workers': 1,  # 预览数据少，不需要并行
                'parallel_backend': "thread",
            }
            if patient_ids_filter:
                load_kwargs['patient_ids'] = patient_ids_filter
            
            result = load_concepts(**load_kwargs)
            
            if isinstance(result, dict):
                for concept, df in result.items():
                    # 🔧 处理各种返回类型
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                    elif hasattr(df, 'dataframe'):
                        df = df.dataframe()
                    elif hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                        df = df.data
                    
                    # 保留所有特征，包括空DataFrame（确保特征数量一致）
                    if isinstance(df, pd.DataFrame):
                        data[concept] = df
                    elif isinstance(df, pd.Series):
                        data[concept] = df.to_frame().reset_index()
            elif isinstance(result, pd.DataFrame):
                # 单概念加载返回 DataFrame（即使为空也保留）
                data[preview_concepts[0]] = result
        except Exception:
            # 批量失败，回退到逐个加载
            for concept in preview_concepts:
                try:
                    df = load_concepts(
                        data_path=st.session_state.data_path,
                        database=st.session_state.get('database'),
                        concepts=[concept],
                        verbose=False,
                        merge=True,
                    )
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                    elif hasattr(df, 'dataframe'):
                        df = df.dataframe()
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        data[concept] = df
                except Exception:
                    pass
        
        if not data:
            lang = st.session_state.get('language', 'en')
            warn_msg = "⚠️ Failed to load any data" if lang == 'en' else "⚠️ 未能加载任何数据"
            st.warning(warn_msg)
            return
        
        # 🔧 POST-FILTER: Remove patients whose cohort-critical features are None
        database = st.session_state.get('database', 'miiv')
        data = _post_filter_cohort_data(data, database)
        
        # 获取患者列表并限制数量
        patient_ids = set()
        id_col_found = 'stay_id'
        id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 
                       'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        
        for concept_df in data.values():
            if isinstance(concept_df, pd.DataFrame):
                for col in id_candidates:
                    if col in concept_df.columns:
                        patient_ids.update(concept_df[col].unique())
                        id_col_found = col
                        break
        
        all_patient_count = len(patient_ids)
        preview_patient_ids = sorted(list(patient_ids))[:max_patients]
        
        # 筛选数据只保留限制的患者
        filtered_data = {}
        for concept_name, df in data.items():
            if isinstance(df, pd.DataFrame) and id_col_found in df.columns:
                filtered_df = df[df[id_col_found].isin(preview_patient_ids)]
                # 保留所有特征，包括空DataFrame（确保特征数量一致）
                filtered_data[concept_name] = filtered_df
            else:
                # 对于没有ID列的DataFrame（如静态指标），直接保留
                filtered_data[concept_name] = df
        
        st.session_state.loaded_concepts = filtered_data
        st.session_state.loaded_data_origin = 'quick_preview'
        st.session_state.patient_ids = preview_patient_ids
        st.session_state.all_patient_count = all_patient_count
        st.session_state.id_col = id_col_found
        
        load_elapsed = time.time() - load_start
        
        # 🔧 FIX (2026-02-04): 统计唯一概念数
        unique_concept_count = count_unique_concepts(list(filtered_data.keys()))
        
        lang = st.session_state.get('language', 'en')
        if lang == 'en':
            st.success(f"✅ Preview data loaded: {unique_concept_count} concepts ({len(filtered_data)} columns), {len(preview_patient_ids)}/{all_patient_count} patients ({load_elapsed:.1f}s)")
            if all_patient_count > max_patients:
                st.info(f"💡 For better performance, visualization is limited to {max_patients} patients. Export data first for full analysis with Python/R.")
        else:
            st.success(f"✅ 预览数据已加载：{unique_concept_count} 个概念（{len(filtered_data)} 列），{len(preview_patient_ids)}/{all_patient_count} 个患者 ({load_elapsed:.1f}秒)")
            if all_patient_count > max_patients:
                st.info(f"💡 为保证流畅性，可视化仅加载前 {max_patients} 个患者。建议先导出数据，再用Python/R工具进行完整分析。")
        
    except Exception as e:
        lang = st.session_state.get('language', 'en')
        err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
        st.error(err_msg)


def render_data_overview():
    """渲染已加载数据的概览页面。"""
    lang = st.session_state.language
    
    # 标题已经在main()中渲染，这里不再重复
    
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

    st.markdown(f'''
    <div style="background:linear-gradient(135deg,#ecfdf5 0%,#d1fae5 100%);border:1px solid #a7f3d0;border-radius:16px;padding:24px 28px;margin-bottom:24px;display:flex;align-items:center;gap:16px">
        <div style="width:48px;height:48px;border-radius:12px;background:#10b981;display:flex;align-items:center;justify-content:center;flex-shrink:0">
            <span style="color:#fff;font-size:1.4rem">✓</span>
        </div>
        <div>
            <div style="font-weight:700;font-size:1.15rem;color:#065f46">{_ready_title}</div>
            <div style="color:#047857;font-size:0.92rem;margin-top:2px">{_ready_sub}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # 状态概览 - 4 个统计卡片
    st.markdown(f'''
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:clamp(10px,.5rem + .5vw,20px);margin-bottom:28px">
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9ca3af;margin-bottom:8px">{_lbl_db}</div>
            <div style="font-size:1.5rem;font-weight:800;color:#6366f1">{db_display}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9ca3af;margin-bottom:8px">{_lbl_feat}</div>
            <div style="font-size:1.5rem;font-weight:800;color:#111827">{n_concepts}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9ca3af;margin-bottom:8px">{_lbl_pat}</div>
            <div style="font-size:1.5rem;font-weight:800;color:#111827">{n_patients:,}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9ca3af;margin-bottom:8px">{_lbl_status}</div>
            <div style="font-size:1.2rem;font-weight:700;color:#10b981">● {_status_val}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # 快捷导航卡片
    if lang == 'en':
        features = [
            ("📈", "Time Series", "Interactive time series visualization with single & multi-patient comparison"),
            ("🏥", "Patient View", "Multi-dimensional patient dashboard for comprehensive assessment"),
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
    st.markdown(f'''
    <div style="margin-bottom:8px">
        <span style="font-size:1.1rem;font-weight:700;color:#111827">{_nav_title}</span>
        <span style="color:#9ca3af;font-size:0.88rem;margin-left:8px">{_nav_hint}</span>
    </div>
    ''', unsafe_allow_html=True)

    _cards_html = ''
    for icon, title, desc in features:
        _cards_html += f'''
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:22px 20px;
                     transition:all .2s ease;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
            <div style="font-weight:700;color:#111827;font-size:1rem;margin-bottom:6px">{title}</div>
            <div style="font-size:0.85rem;color:#6b7280;line-height:1.55">{desc}</div>
        </div>'''
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px">{_cards_html}</div>', unsafe_allow_html=True)

    # 数据摘要
    summary_label = "Data Summary" if lang == 'en' else "数据摘要"
    st.markdown(f'''
    <div style="font-size:1.05rem;font-weight:700;color:#111827;margin-bottom:12px">{summary_label}</div>
    ''', unsafe_allow_html=True)
    
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
        st.dataframe(pd.DataFrame(concept_stats), width="stretch", hide_index=True)


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

    _steps_html = ''
    for idx, (title, desc) in enumerate(_steps_viz):
        if idx < _cur_viz:
            _dot = '<div class="step-dot done">✓</div>'
        elif idx == _cur_viz:
            _dot = f'<div class="step-dot active">{idx+1}</div>'
        else:
            _dot = f'<div class="step-dot">{idx+1}</div>'
        _steps_html += f'<div class="step-indicator"><div style="display:flex;align-items:center;gap:10px">{_dot}<div class="step-text"><div>{title}</div><small>{desc}</small></div></div></div>'
    st.markdown(f'<div style="display:flex;gap:32px;margin-bottom:28px">{_steps_html}</div>', unsafe_allow_html=True)

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
            ("🏥", "Patient View", "Patient dashboard"),
            ("📊", "Data Quality", "Missing analysis"),
        ]
    else:
        features = [
            ("📋", "数据大表", "浏览与合并"),
            ("📈", "时序分析", "交互式图表"),
            ("🏥", "患者视图", "患者仪表盘"),
            ("📊", "数据质量", "缺失率分析"),
        ]

    st.markdown(f'''
    <div style="font-size:0.88rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px">{_preview_title}</div>
    ''', unsafe_allow_html=True)
    _cards = ''
    for icon, title, desc in features:
        _cards += f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:18px 14px;text-align:center">
            <div style="font-size:1.6rem;margin-bottom:6px">{icon}</div>
            <div style="font-weight:600;color:#111827;font-size:.9rem">{title}</div>
            <div style="font-size:.78rem;color:#9ca3af;margin-top:3px">{desc}</div>
        </div>'''
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:clamp(8px,.4rem + .4vw,16px)">{_cards}</div>', unsafe_allow_html=True)


def render_home_extract_mode(lang):
    """渲染数据提取导出模式的首页教程。"""
    
    # 计算当前步骤完成状态（4个步骤）
    # Step 1: Demo模式需要点击Confirm按钮，Real Data模式需要有效路径
    if st.session_state.get('use_mock_data', False):
        step1_done = st.session_state.get('step1_confirmed', False)
    else:
        step1_done = st.session_state.data_path and Path(st.session_state.data_path).exists()
    step2_done = st.session_state.get('step2_confirmed', False)
    # 🔧 FIX (2026-02-05): Step 3 必须点击确认按钮后才算完成
    step3_done = st.session_state.get('step3_confirmed', False) and len(st.session_state.get('selected_concepts', [])) > 0
    # Step 4 只在真正导出完成后才算完成
    step4_done = st.session_state.get('export_completed', False)
    
    # ============ 进度指示器 — Stepper 风格 ============
    st.markdown('<div id="progress"></div>', unsafe_allow_html=True)

    if lang == 'en':
        st.markdown("""
        <p style="font-size:1.1rem;color:var(--text-secondary-light);margin:0.35rem 0 0.7rem;line-height:1.7;font-weight:500;">
            👈 Follow the <b>4 steps in the left sidebar</b> to define your cohort, select features, and export data.
        </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="font-size:1.1rem;color:var(--text-secondary-light);margin:0.35rem 0 0.7rem;line-height:1.7;font-weight:500;">
            👈 按照<b>左侧边栏的 4 个步骤</b>操作，即可完成 ICU 数据的队列定义、特征选择和导出。
        </p>
        """, unsafe_allow_html=True)

    # 步骤数据
    _steps = [
        (step1_done, "1", ("Data Source", "配置数据源"), ("Configure database path", "选择数据路径")),
        (step2_done, "2", ("Cohort Selection", "队列筛选"), ("Define patient criteria", "定义患者筛选条件")),
        (step3_done, "3", ("Select Features", "选择特征"), ("Choose clinical variables", "选择临床变量")),
        (step4_done, "4", ("Export Data", "导出数据"), ("Generate output files", "输出数据文件")),
    ]

    # 确定当前活跃步骤
    _current_step = 0
    for i, (done, *_) in enumerate(_steps):
        if not done:
            _current_step = i
            break
    else:
        _current_step = 4  # 全部完成

    # 横向 stepper HTML
    _stepper_items = []
    for i, (done, num, (title_en, title_zh), (desc_en, desc_zh)) in enumerate(_steps):
        _title = title_en if lang == 'en' else title_zh
        _desc = desc_en if lang == 'en' else desc_zh
        if done:
            _dot_cls = "done"
            _dot_content = "✓"
            _row_cls = "done"
        elif i == _current_step:
            _dot_cls = "active"
            _dot_content = num
            _row_cls = "active"
        else:
            _dot_cls = "pending"
            _dot_content = num
            _row_cls = ""
        _stepper_items.append(f"""
        <div class="step-indicator {_row_cls}">
            <div class="step-dot {_dot_cls}">{_dot_content}</div>
            <div class="step-text" style="font-size:1.08rem;font-weight:700;line-height:1.25;color:#1f2937">{_title}<small style="display:block;font-size:0.96rem;font-weight:500;line-height:1.45;color:#7c8faa;margin-top:6px">{_desc}</small></div>
        </div>""")

    col1, col2, col3, col4 = st.columns(4)
    for col, item_html in zip([col1, col2, col3, col4], _stepper_items):
        with col:
            st.markdown(item_html, unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    
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
    
    guide_title_text = f"Guide: {guide_step}" if lang == 'en' else f"引导: {guide_step}"
    st.markdown(f'''
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
        <div style="width:6px;height:32px;border-radius:3px;background:linear-gradient(180deg,#6366f1,#8b5cf6)"></div>
        <span style="font-size:1.7rem;font-weight:800;color:#111827;letter-spacing:-0.02em">{guide_title_text}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    if not step1_done:
        # 步骤1引导：配置数据源
        if lang == 'en':
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:800;color:#111827;font-size:1.42rem;margin-bottom:18px;letter-spacing:-0.02em">Configure Data Source in the Sidebar</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#166534;margin-bottom:10px;font-size:1.12rem">🎭 Demo Mode</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>No real data needed — generates simulated ICU data</li>
                            <li>Adjust patients (50-500) & duration (24-168h)</li>
                            <li>Click <b>"Confirm Data Source"</b> when ready</li>
                        </ul>
                    </div>
                    <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#3730a3;margin-bottom:10px;font-size:1.12rem">📊 Real Data Mode</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>MIMIC-IV, eICU, AUMC, HiRID, MIMIC-III, SICdb</li>
                            <li>Enter your local database path</li>
                            <li>All processing is local — data stays secure 🔒</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:800;color:#111827;font-size:1.42rem;margin-bottom:18px;letter-spacing:-0.02em">在侧边栏配置数据源</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#166534;margin-bottom:10px;font-size:1.12rem">🎭 演示模式</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>无需真实数据 — 自动生成模拟ICU数据</li>
                            <li>可调整患者数量(50-500)和时长(24-168h)</li>
                            <li>设置后点击<b>「确认数据源配置」</b></li>
                        </ul>
                    </div>
                    <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#3730a3;margin-bottom:10px;font-size:1.12rem">📊 真实数据模式</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>支持 MIMIC-IV、eICU、AUMC、HiRID、MIMIC-III、SICdb</li>
                            <li>输入本地数据库路径</li>
                            <li>所有处理本地完成 — 数据安全 🔒</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
    elif not step2_done:
        # 步骤2引导：队列筛选
        if lang == 'en':
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">Configure Cohort Selection</div>
                <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px;margin-bottom:12px">
                    <div style="font-weight:600;color:#3730a3;margin-bottom:8px">Available Filters</div>
                    <ul style="color:#4b5563;font-size:.92rem;line-height:1.7;padding-left:18px;margin:0">
                        <li><b>Age Range</b> — e.g., 18-65 years</li>
                        <li><b>Gender</b> — Male, Female, or Any</li>
                        <li><b>Survival Status</b> — Survivors, non-survivors, or all</li>
                        <li><b>ICU Stay</b> — Minimum length of stay</li>
                    </ul>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 Click <b>"Confirm (No Filtering)"</b> to skip this step
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">配置队列筛选</div>
                <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px;margin-bottom:12px">
                    <div style="font-weight:600;color:#3730a3;margin-bottom:8px">可用筛选条件</div>
                    <ul style="color:#4b5563;font-size:.92rem;line-height:1.7;padding-left:18px;margin:0">
                        <li><b>年龄范围</b> — 如 18-65 岁</li>
                        <li><b>性别</b> — 男/女/不限</li>
                        <li><b>存活状态</b> — 存活/死亡/全部</li>
                        <li><b>ICU住院时长</b> — 最短住院时长</li>
                    </ul>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 点击<b>「确认（不筛选）」</b>可跳过此步骤
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
    elif not step3_done:
        # 步骤3引导：选择特征
        if lang == 'en':
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">Select Features — 167 ICU Clinical Features</div>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px">
                    <div style="background:#eff6ff;border-radius:8px;padding:12px"><b style="color:#1d4ed8">📊 Vital Signs</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">HR, BP, Temp, SpO2, Resp</div></div>
                    <div style="background:#ecfdf5;border-radius:8px;padding:12px"><b style="color:#047857">🧪 Lab Tests</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">Chemistry, CBC, Coag, ABG</div></div>
                    <div style="background:#fffbeb;border-radius:8px;padding:12px"><b style="color:#b45309">💊 Medications</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">Vasopressors, Sedatives, ABX</div></div>
                    <div style="background:#f5f3ff;border-radius:8px;padding:12px"><b style="color:#6d28d9">🏥 Scores</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">SOFA, GCS, AKI, Sepsis-3</div></div>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 Select by category or pick individual features. Check the Data Dictionary below for details.
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">选择特征 — 167 个 ICU 临床特征</div>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px">
                    <div style="background:#eff6ff;border-radius:8px;padding:12px"><b style="color:#1d4ed8">📊 生命体征</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">心率、血压、体温、SpO2、呼吸</div></div>
                    <div style="background:#ecfdf5;border-radius:8px;padding:12px"><b style="color:#047857">🧪 实验室检验</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">生化、血常规、凝血、血气</div></div>
                    <div style="background:#fffbeb;border-radius:8px;padding:12px"><b style="color:#b45309">💊 药物治疗</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">血管活性药、镇静药、抗生素</div></div>
                    <div style="background:#f5f3ff;border-radius:8px;padding:12px"><b style="color:#6d28d9">🏥 临床评分</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">SOFA、GCS、AKI、Sepsis-3</div></div>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 按类别选择或选取单个特征，查看下方数据字典了解详情
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
    elif not step4_done:
        # Step 4 Guide: Export Data
        # 🆕 检查是否正在导出或刚完成导出
        exporting_in_progress = st.session_state.get('_exporting_in_progress', False)
        
        if exporting_in_progress:
            # 🆕 导出正在进行中，显示进度标题
            _exp_msg = ('Export in Progress...', 'Please wait while your data is being exported.') if lang == 'en' else ('导出进行中...', '请稍候，数据正在导出中，进度详情将显示在下方。')
            st.markdown(f'''<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:14px;padding:20px 24px;margin-bottom:12px">
<div style="font-weight:700;color:#b45309;font-size:1.05rem">⏳ {_exp_msg[0]}</div>
<div style="color:#92400e;font-size:.9rem;margin-top:4px">{_exp_msg[1]}</div>
</div>''', unsafe_allow_html=True)
        else:
            # 显示导出教程
            _steps_html = '<ol style="color:#374151;font-size:.92rem;line-height:2;margin:0;padding-left:18px"><li>{}</li><li>{}</li><li>{}</li><li>{}</li></ol>'
            _tip = ''
            if lang == 'en':
                _steps_html = _steps_html.format('Go to <b>"Data Export"</b> tab above', 'Select export format (CSV / Parquet / Excel)', 'Choose save location', 'Click <b>"Export Data"</b> button')
                _tip = '✅ Best for large datasets — saves directly to disk'
            else:
                _steps_html = _steps_html.format('点击上方 <b>"数据导出"</b> 标签页', '选择导出格式（CSV / Parquet / Excel）', '选择保存位置', '点击 <b>"导出数据"</b> 按钮')
                _tip = '✅ 适合大数据集 — 直接保存到磁盘，不占用内存'
            st.markdown(f'''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
<div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">{"How to Export Data" if lang=="en" else "如何导出数据"}</div>
{_steps_html}
<div style="background:#ecfdf5;border-radius:8px;padding:10px 14px;margin-top:12px;font-size:.85rem;color:#047857">{_tip}</div>
</div>''', unsafe_allow_html=True)
            
            # 显示当前选择摘要
            selected = st.session_state.get('selected_concepts', [])
            if st.session_state.get('use_mock_data', False):
                source_info = "Demo Mode" if lang == 'en' else "演示模式"
            else:
                source_info = f"{st.session_state.get('data_path', '')}"
            
            source_label = "Data Source" if lang == 'en' else "数据源"
            feat_label = "Selected Features" if lang == 'en' else "已选特征"
            
            st.markdown(f'''
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:14px">
                    <div style="font-size:.75rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{source_label}</div>
                    <div style="font-weight:600;color:#111827;margin-top:4px;font-size:.9rem">{source_info}</div>
                </div>
                <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:14px">
                    <div style="font-size:.75rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{feat_label}</div>
                    <div style="font-weight:700;color:#6366f1;margin-top:4px;font-size:1.2rem">{len(selected)}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # 🆕 导出进度区域（无论是否正在导出都创建，导出时内容会填充进来）
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
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
            # 🔧 FIX (2026-02-04): 使用保存的概念数
            concept_count = export_result.get('concept_count', len(exported_files))
            
            success_msg = f"✅ Successfully exported {len(exported_files)} files to `{export_dir}`" if lang == 'en' else f"✅ 成功导出 {concept_count} 个概念（{len(exported_files)} 个文件）到 `{export_dir}`"
            st.success(success_msg)
            
            # 🆕 显示队列筛选统计（在导出成功消息之后）
            cohort_stats = st.session_state.get('_cohort_stats')
            if cohort_stats and cohort_stats.get('excluded', 0) > 0:
                n_before = cohort_stats['before']
                n_excluded = cohort_stats['excluded']
                n_after = cohort_stats['after']
                details = cohort_stats.get('filter_details', [])
                if lang == 'en':
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
                # 使用多列布局显示文件列表
                files_to_show = exported_files[:12]  # 最多显示12个
                num_cols = 3  # 每行3个文件
                for i in range(0, len(files_to_show), num_cols):
                    cols = st.columns(num_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(files_to_show):
                            with col:
                                st.markdown(f"<p style='color: #1e1e1e; font-size: 0.9rem; margin: 2px 0;'>• {Path(files_to_show[idx]).name}</p>", unsafe_allow_html=True)
                if len(exported_files) > 12:
                    more_msg = f"... and {len(exported_files) - 12} more files" if lang == 'en' else f"... 及其他 {len(exported_files) - 12} 个文件"
                    st.markdown(f"<p style='color: #1e1e1e; font-size: 0.9rem; margin: 2px 0;'>{more_msg}</p>", unsafe_allow_html=True)
            
            # 🆕 显示被选择但未能提取的特征（区分无数据源 vs 无数据）
            unavailable_concepts = export_result.get('unavailable_concepts', [])
            unsupported_list = export_result.get('unsupported_concepts', [])
            empty_data_list = export_result.get('empty_data_concepts', [])
            if unavailable_concepts:
                # 🔧 FIX(2026-02-09): 分别显示无数据源和无数据的特征
                unsupported_in_unavail = [c for c in unavailable_concepts if c in unsupported_list]
                empty_or_other = [c for c in unavailable_concepts if c not in unsupported_list]
                
                parts_html = []
                if unsupported_in_unavail:
                    concepts_formatted = ', '.join(sorted(unsupported_in_unavail))
                    if lang == 'en':
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>🚫 Not configured ({len(unsupported_in_unavail)}):</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')
                    else:
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>🚫 该数据库未配置 ({len(unsupported_in_unavail)})：</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')
                if empty_or_other:
                    concepts_formatted = ', '.join(sorted(empty_or_other))
                    if lang == 'en':
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>📭 No data for selected patients ({len(empty_or_other)}):</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')
                    else:
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>📭 所选患者无数据 ({len(empty_or_other)})：</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')
                
                body = ''.join(parts_html)
                tip = '💡 <i>Try increasing the patient sample size or selecting All Patients to get more features.</i>' if lang == 'en' else '💡 <i>尝试增大患者样本量或选择全部患者以获取更多特征。</i>'
                unavailable_msg = f"""<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:12px;padding:16px 20px;margin-top:15px">
<div style="font-weight:600;color:#92400e;margin-bottom:8px">{len(unavailable_concepts)} {"selected features were not extracted" if lang=="en" else "个已选特征未被提取"}:</div>
{body}
<div style="margin-top:10px;font-size:.85rem;color:#6b7280">{tip}</div>
</div>"""
                st.markdown(unavailable_msg, unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            # 🔧 FIX (2026-02-04): 在删除前保存概念数和患者数，供后面的卡片使用
            st.session_state['_last_export_concept_count'] = export_result.get('concept_count', len(exported_files))
            st.session_state['_last_export_patient_count'] = export_result.get('patient_count', 0)
            # 清除导出结果，避免重复显示
            del st.session_state['_export_success_result']
        
        # 显示状态概览卡片
        db_label = "Database" if lang == 'en' else "数据库"
        feat_label = "Loaded Concepts" if lang == 'en' else "已加载概念"
        patient_label = "Patients" if lang == 'en' else "患者数量"
        status_label = "Status" if lang == 'en' else "数据状态"
        ready_status = "Ready" if lang == 'en' else "就绪"
        
        db_display = "DEMO" if st.session_state.get('use_mock_data', False) else st.session_state.get('database', 'N/A').upper()
        
        # 计算概念数
        export_result = st.session_state.get('_export_success_result')
        if export_result and 'concept_count' in export_result:
            n_concepts = export_result['concept_count']
        elif '_last_export_concept_count' in st.session_state:
            n_concepts = st.session_state['_last_export_concept_count']
        elif st.session_state.loaded_concepts:
            n_concepts = len(st.session_state.loaded_concepts)
        elif st.session_state.get('selected_concepts'):
            n_concepts = len(st.session_state.selected_concepts)
        else:
            n_concepts = 0
        
        # 计算患者数
        n_patients = 0
        id_col = st.session_state.get('id_col', 'stay_id')
        if st.session_state.get('_exported_patient_count'):
            n_patients = st.session_state['_exported_patient_count']
        if n_patients == 0 and st.session_state.loaded_concepts:
            all_ids = set()
            for df in st.session_state.loaded_concepts.values():
                if isinstance(df, pd.DataFrame) and id_col in df.columns:
                    all_ids.update(df[id_col].unique())
            if all_ids:
                n_patients = len(all_ids)
        if n_patients == 0 and st.session_state.patient_ids:
            n_patients = len(st.session_state.patient_ids)
        if n_patients == 0:
            mock_params = st.session_state.get('mock_params', {})
            if mock_params.get('n_patients'):
                n_patients = mock_params['n_patients']
        
        st.markdown(f'''
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:clamp(8px,.4rem + .4vw,16px);margin-bottom:16px">
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px">
                <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{db_label}</div>
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-top:6px">{db_display}</div>
            </div>
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px">
                <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{feat_label}</div>
                <div style="font-weight:700;color:#6366f1;font-size:1.4rem;margin-top:6px">{n_concepts}</div>
            </div>
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px">
                <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{patient_label}</div>
                <div style="font-weight:700;color:#111827;font-size:1.4rem;margin-top:6px">{n_patients:,}</div>
            </div>
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px">
                <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{status_label}</div>
                <div style="font-weight:700;color:#10b981;font-size:1.15rem;margin-top:6px">✓ {ready_status}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # 🆕 What's Next? 两个选项
        next_step_title = "What's Next?" if lang == 'en' else "下一步？"
        st.markdown(f'<div style="font-weight:700;color:#111827;font-size:1.1rem;margin:20px 0 12px">{next_step_title}</div>', unsafe_allow_html=True)
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            # Option A: Quick Visualization
            if lang == 'en':
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 22px">
<div style="font-weight:700;color:#0369a1;margin-bottom:10px">📈 Quick Visualization</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.88rem;line-height:1.9">
<li><b>Data Tables Explorer</b> — Browse loaded data by module</li>
<li><b>Time Series Analysis</b> — Clinical trends over time</li>
<li><b>Patient Overview</b> — Single-patient dashboard</li>
<li><b>Data Quality</b> — Missing rates &amp; completeness</li>
</ul>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 22px">
<div style="font-weight:700;color:#0369a1;margin-bottom:10px">📈 快速可视化</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.88rem;line-height:1.9">
<li><b>数据表浏览器</b> — 按模块浏览已加载数据</li>
<li><b>时序分析</b> — 临床指标随时间变化趋势</li>
<li><b>患者概览</b> — 单患者综合仪表盘</li>
<li><b>数据质量</b> — 缺失率与完整性分析</li>
</ul>
</div>''', unsafe_allow_html=True)
            
            # Option A 按钮
            viz_label = "📈 Go to Visualization" if lang == 'en' else "📈 前往可视化"
            if st.button(viz_label, use_container_width=True, key="goto_viz_home", type="primary"):
                if st.session_state.get('last_export_dir') or st.session_state.get('viz_confirmed_path'):
                    st.session_state['viz_data_source_mode'] = 'exported'
                    st.session_state['_prefer_exported_viz'] = True
                st.session_state['_scroll_to_tab'] = 'viz'
                st.rerun()
        
        with col_opt2:
            # Option B: Cohort Analysis
            if lang == 'en':
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 22px">
<div style="font-weight:700;color:#6d28d9;margin-bottom:10px">🔬 Cohort Analysis</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.88rem;line-height:1.9">
<li><b>Group Comparison</b> — Statistical tests between subgroups</li>
<li><b>Multi-DB Distribution</b> — Compare across databases</li>
<li><b>Cohort Dashboard</b> — Demographics &amp; outcomes overview</li>
</ul>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 22px">
<div style="font-weight:700;color:#6d28d9;margin-bottom:10px">🔬 队列分析</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.88rem;line-height:1.9">
<li><b>组间比较</b> — 统计检验比较亚组</li>
<li><b>多数据库分布</b> — 跨数据库特征分布比较</li>
<li><b>队列仪表盘</b> — 人口统计学与结局概览</li>
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
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    # 添加字典锚点和大标题
    st.markdown('<div id="dictionary"></div>', unsafe_allow_html=True)
    dict_header = "📖 Data Dictionary" if lang == 'en' else "📖 数据字典"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.6rem;">{dict_header}</h2>', unsafe_allow_html=True)
    
    # 添加数据字典说明
    if lang == 'en':
        st.markdown('''
        <div style="background: rgba(102, 126, 234, 0.15); padding: 18px; border-radius: 12px; margin-bottom: 14px; border-left: 4px solid #667eea;">
            <p style="color: #333; font-size: 1.15rem; margin: 0; line-height: 1.7;">
                📚 <b>Reference Guide</b>: This dictionary contains all 167 ICU clinical features available in EasyICU, organized into 19 categories. 
                Each feature includes its code name, full description, and measurement unit. 
                Use this to understand what data you're extracting and make informed selections.
                Note that some features may not be available in all ICU databases.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="background: rgba(102, 126, 234, 0.15); padding: 18px; border-radius: 12px; margin-bottom: 14px; border-left: 4px solid #667eea;">
            <p style="color: #333; font-size: 1.15rem; margin: 0; line-height: 1.7;">
                📚 <b>参考指南</b>：本字典包含 EasyICU 提供的全部 167 个 ICU 临床特征，分为 19 个类别。
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
            <p>🏥 EasyICU - ICU Data Analysis Toolkit | 
            📦 <a href="https://github.com/shen-lab-icu/EASYICU" style="color:#4fc3f7">GitHub</a> | 
            📖 <a href="https://github.com/shen-lab-icu/EASYICU/blob/main/README.md" style="color:#4fc3f7">Docs</a></p>
            <p>All data processing is done locally, no data is uploaded to any server 🔒</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align:center;color:#aaa;font-size:0.85rem">
            <p>🏥 EasyICU - ICU 数据分析工具包 | 
            📦 <a href="https://github.com/shen-lab-icu/EASYICU" style="color:#4fc3f7">GitHub</a> | 
            📖 <a href="https://github.com/shen-lab-icu/EASYICU/blob/main/README.md" style="color:#4fc3f7">Docs</a></p>
            <p>所有数据处理均在本地完成，不会上传到任何服务器 🔒</p>
        </div>
        ''', unsafe_allow_html=True)


def render_home_data_dictionary(lang):
    """在首页渲染完整的数据字典。"""
    dict_title = "📖 Complete Data Dictionary" if lang == 'en' else "📖 完整数据字典"
    
    with st.expander(dict_title, expanded=True):
        
        # 🔍 搜索框
        search_placeholder = "Search by code, name or description... (e.g. hr, heart rate, lactate)" if lang == 'en' else "按代码、名称或描述搜索... (如 hr、heart rate、心率)"
        search_query = st.text_input(
            "🔍 Search" if lang == 'en' else "🔍 搜索",
            placeholder=search_placeholder,
            key="dict_search_input",
        )
        
        # 获取分组
        concept_groups = get_concept_groups()
        
        # 如果有搜索词，展示搜索结果
        if search_query and search_query.strip():
            query = search_query.strip().lower()
            matched_rows = []
            for group_name, concepts in concept_groups.items():
                for concept in concepts:
                    if concept in CONCEPT_DICTIONARY:
                        eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
                        eng_desc, chn_desc = CONCEPT_DESCRIPTIONS.get(concept, ('', ''))
                        # 匹配 code、英文名、中文名、描述
                        if lang == 'en':
                            searchable = f"{concept} {eng_name} {eng_desc}".lower()
                        else:
                            searchable = f"{concept} {eng_name} {chn_name} {eng_desc} {chn_desc}".lower()
                        if query in searchable:
                            if lang == 'en':
                                matched_rows.append({
                                    'Code': concept,
                                    'Full Name': eng_name,
                                    'Category': group_name,
                                    'Description': eng_desc if eng_desc else eng_name,
                                    'Unit': unit if unit else '-'
                                })
                            else:
                                matched_rows.append({
                                    '代码': concept,
                                    '全称': eng_name,
                                    '类别': group_name,
                                    '说明': chn_desc if chn_desc else chn_name,
                                    '单位': unit if unit else '-'
                                })
            
            if matched_rows:
                n = len(matched_rows)
                result_text = f"Found **{n}** matching feature(s)" if lang == 'en' else f"找到 **{n}** 个匹配特征"
                st.success(result_text)
                st.dataframe(pd.DataFrame(matched_rows), width="stretch", hide_index=True, height=min(300, 50 + 35 * n))
            else:
                no_result = "No matching features found." if lang == 'en' else "未找到匹配的特征。"
                st.warning(no_result)
        else:
            # 无搜索词时，按类别展示
            categories_title = "📂 Categories" if lang == 'en' else "📂 类别"
            st.markdown(f"#### {categories_title}")
            
            for group_name in concept_groups.keys():
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


def _add_clinical_thresholds(fig, concept_name: str, show: bool = True):
    """在 Plotly 时序图上添加临床阈值参考线。"""
    if not show:
        return fig
    thresholds = CLINICAL_THRESHOLDS.get(concept_name)
    if not thresholds:
        return fig
    for val, color, label in zip(thresholds['lines'], thresholds['colors'], thresholds['labels']):
        fig.add_hline(
            y=val, line_dash="dot", line_color=color, line_width=1.5,
            opacity=0.7,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color=color,
        )
    return fig


def render_timeseries_page():
    """渲染时序分析页面。"""
    lang = st.session_state.get('language', 'en')
    
    _ts_title = "Time Series Analysis" if lang == 'en' else "时序数据分析"
    _ts_sub = "Interactive visualization, single & multi-patient comparison" if lang == 'en' else "交互式可视化，支持单/多患者对比"
    _hdr_col1, _hdr_col2 = st.columns([3, 1])
    with _hdr_col1:
        st.markdown(f'''
        <div style="margin-bottom:16px">
            <div style="font-size:1.4rem;font-weight:800;color:#111827">{_ts_title}</div>
            <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_ts_sub}</div>
        </div>
        ''', unsafe_allow_html=True)
    with _hdr_col2:
        _show_thresh = st.toggle(get_text('threshold_lines'), value=True, key="ts_show_thresholds")
        st.session_state['_ts_show_thresholds'] = _show_thresh
    
    if len(st.session_state.loaded_concepts) == 0:
        _msg = "Load data to begin time series analysis." if lang == 'en' else "请先加载数据以进行时序分析。"
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">📈</div>
            <div style="font-weight:600;color:#111827">{_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # Concept 选择区域
    available_concepts = list(st.session_state.loaded_concepts.keys())
    
    # 分析模式选择 — 新增 Clinical Lanes 模式
    mode_label = "Analysis Mode" if lang == 'en' else "分析模式"
    mode_lanes = get_text('clinical_lanes')
    mode_single = "Single Patient" if lang == 'en' else "单患者分析"
    mode_multi = "Multi-Patient Comparison" if lang == 'en' else "多患者比较"
    analysis_mode = st.radio(
        mode_label,
        options=[mode_lanes, mode_single, mode_multi],
        horizontal=True,
        key="ts_mode"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ============ Clinical Lanes View (默认) ============
    if analysis_mode == mode_lanes:
        import plotly.graph_objects as go
        
        if not st.session_state.patient_ids:
            st.warning("No patient data" if lang == 'en' else "无患者数据")
        else:
            _lane_pid = st.selectbox(
                "Patient" if lang == 'en' else "患者",
                options=st.session_state.patient_ids,
                key="lane_patient_select"
            )
            
            id_col = st.session_state.get('id_col', 'stay_id')
            _show_thresh = st.session_state.get('_ts_show_thresholds', True)
            
            for lane_name, lane_concepts in CLINICAL_LANES.items():
                _lane_avail = [c for c in lane_concepts if c in available_concepts]
                if not _lane_avail:
                    continue
                    
                lane_label = get_text(f'lane_{lane_name}')
                st.markdown(f"#### {lane_label}")
                
                _n_cols = min(len(_lane_avail), 3)
                _cols = st.columns(_n_cols)
                for idx, cname in enumerate(_lane_avail[:6]):
                    with _cols[idx % _n_cols]:
                        df = st.session_state.loaded_concepts.get(cname)
                        if df is None or not hasattr(df, 'columns'):
                            continue
                        if id_col not in df.columns or cname not in df.columns:
                            continue
                        pdf = df[df[id_col] == _lane_pid].copy()
                        if pdf.empty:
                            st.caption(f"{cname}: no data")
                            continue
                        
                        _tcol = None
                        for tc in ['charttime', 'time', 'hour', 'datetime', 'measuredat_minutes',
                                   'observationoffset', 'starttime']:
                            if tc in pdf.columns:
                                _tcol = tc
                                break
                        
                        if _tcol is None:
                            st.caption(f"{cname}: no time column")
                            continue
                        
                        try:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=pdf[_tcol], y=pd.to_numeric(pdf[cname], errors='coerce'),
                                mode='lines+markers', name=cname,
                                line=dict(width=1.5), marker=dict(size=3)
                            ))
                            fig = _add_clinical_thresholds(fig, cname, _show_thresh)
                            
                            _unit = CLINICAL_THRESHOLDS.get(cname, {}).get('unit', '')
                            fig.update_layout(
                                title=dict(text=f"{cname}" + (f" ({_unit})" if _unit else ""), font=dict(size=12)),
                                height=200, margin=dict(l=40, r=10, t=30, b=25),
                                showlegend=False, xaxis=dict(title=""), yaxis=dict(title=""),
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"lane_{lane_name}_{cname}_{_lane_pid}")
                        except Exception:
                            st.caption(f"{cname}: render error")
                
                st.markdown("---")
        return
    
    if analysis_mode == mode_single:
        # 顶部控制面板 - 🔧 FIX: 添加模块筛选，方便用户在100+特征中找到想要的
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            # 🔧 FIX: 先选择模块，再选择特征
            module_label = "📂 Select Module" if lang == 'en' else "📂 选择模块"
            all_modules_opt = "All Modules" if lang == 'en' else "全部模块"
            
            # 获取模块列表 - 🔧 FIX (2026-02-05): 只显示支持时序分析的模块
            module_options = [all_modules_opt]
            for grp_key in CONCEPT_GROUPS_INTERNAL:
                # 跳过不支持时序分析的模块（demographics, outcome）
                if grp_key not in TIME_SERIES_COMPATIBLE_MODULES:
                    continue
                grp_concepts = CONCEPT_GROUPS_INTERNAL[grp_key]
                # 检查该模块是否有已加载的概念
                if any(c in available_concepts for c in grp_concepts):
                    display_name = CONCEPT_GROUPS_DISPLAY.get(grp_key, grp_key)
                    module_options.append(display_name)
            
            selected_module = st.selectbox(
                module_label,
                options=module_options,
                key="ts_module"
            )
        
        with col2:
            # 根据选择的模块过滤概念
            if selected_module == all_modules_opt:
                filtered_concepts = available_concepts
            else:
                # 找到对应的 group_key
                selected_grp_key = None
                for grp_key, display in CONCEPT_GROUPS_DISPLAY.items():
                    if display == selected_module:
                        selected_grp_key = grp_key
                        break
                if selected_grp_key:
                    grp_concepts = CONCEPT_GROUPS_INTERNAL.get(selected_grp_key, [])
                    filtered_concepts = [c for c in available_concepts if c in grp_concepts]
                else:
                    filtered_concepts = available_concepts
            
            concept_label = "📋 Select Concept" if lang == 'en' else "📋 选择 Concept"
            concept_help = "Select data type to visualize" if lang == 'en' else "选择要可视化的数据类型"
            selected_concept = st.selectbox(
                concept_label,
                options=filtered_concepts if filtered_concepts else available_concepts,
                key="ts_concept",
                help=concept_help
            )
        
        with col3:
            if st.session_state.patient_ids:
                patient_label = "👤 Select Patient" if lang == 'en' else "👤 选择患者"
                # 🔧 FIX: 支持用户输入搜索患者ID
                patient_search = st.text_input(
                    "🔍 Search Patient ID" if lang == 'en' else "🔍 搜索患者ID",
                    key="ts_patient_search",
                    placeholder="Type to filter..." if lang == 'en' else "输入ID过滤..."
                )
                
                # 过滤患者列表
                all_patients = st.session_state.patient_ids[:500]  # 限制前500个
                if patient_search:
                    filtered_patients = [p for p in all_patients if str(patient_search) in str(p)]
                else:
                    filtered_patients = all_patients[:100]
                
                patient_id = st.selectbox(
                    patient_label,
                    options=filtered_patients if filtered_patients else all_patients[:100],
                    key="ts_patient"
                )
            else:
                patient_id = None
                no_patient_msg = "No patients found" if lang == 'en' else "未找到患者"
                st.warning(no_patient_msg)
        
        with col4:
            chart_label = "📊 Chart Type" if lang == 'en' else "📊 图表类型"
            line_opt = "Line Chart" if lang == 'en' else "折线图"
            scatter_opt = "Scatter Plot" if lang == 'en' else "散点图"
            area_opt = "Area Chart" if lang == 'en' else "面积图"
            chart_type = st.selectbox(
                chart_label,
                options=[line_opt, scatter_opt, area_opt],
                key="ts_chart_type"
            )
        
        with col5:
            show_stats_label = "Show Statistics" if lang == 'en' else "显示统计"
            show_stats = st.checkbox(show_stats_label, value=True, key="ts_show_stats")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # 主图表区域
        if selected_concept and patient_id:
            df = st.session_state.loaded_concepts[selected_concept]
            
            # 确保是 DataFrame
            if not isinstance(df, pd.DataFrame):
                format_warn = f"Data format not supported: {type(df).__name__}" if lang == 'en' else f"数据格式不支持: {type(df).__name__}"
                st.warning(format_warn)
                return
            
            # 过滤数据
            id_col = st.session_state.id_col
            if id_col and id_col in df.columns:
                patient_df = df[df[id_col] == patient_id].copy()
            else:
                patient_df = df.copy()
            
            # 显示图表
            if len(patient_df) > 0:
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # 确定数值列
                    numeric_cols = patient_df.select_dtypes(include=['number']).columns
                    # 排除ID列和所有可能的时间列
                    exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'index', 'time', 
                                   'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                                   'patientunitstayid', 'admissionid', 'patientid', 'CaseID', 'Offset',
                                   'measuredat_minutes', 'measuredat']
                    value_cols = [c for c in numeric_cols if c not in exclude_cols]
                    
                    # 检测时间列 - 支持多种命名
                    time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat']
                    time_col = None
                    for tc in time_candidates:
                        if tc in patient_df.columns:
                            time_col = tc
                            break
                    
                    if value_cols:
                        value_col = value_cols[0]
                        
                        if time_col:
                            # 根据图表类型创建图表
                            line_type = "Line Chart" if lang == 'en' else "折线图"
                            scatter_type = "Scatter Plot" if lang == 'en' else "散点图"
                            patient_label = "Patient" if lang == 'en' else "患者"
                            chart_title = f"📈 {selected_concept.upper()} - {patient_label} {patient_id}"
                            
                            if chart_type == line_type:
                                fig = px.line(
                                    patient_df, x=time_col, y=value_col,
                                    title=chart_title,
                                    markers=True
                                )
                            elif chart_type == scatter_type:
                                fig = px.scatter(
                                    patient_df, x=time_col, y=value_col,
                                    title=chart_title,
                                    size_max=10
                                )
                            else:  # 面积图
                                fig = px.area(
                                    patient_df, x=time_col, y=value_col,
                                    title=chart_title
                                )
                            
                            # 美化图表
                            time_label = "Time (hours)" if lang == 'en' else "时间 (小时)"
                            fig.update_layout(
                                template="plotly_white",
                                hovermode="x unified",
                                xaxis_title=time_label,
                                yaxis_title=value_col.upper(),
                                font=dict(size=14, color='black'),
                                title_font_size=16,
                                showlegend=False,
                                margin=dict(l=50, r=30, t=50, b=50),
                            )
                            fig.update_traces(
                                line=dict(width=2, color='#1f77b4'),
                                marker=dict(size=6)
                            )
                            fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # 🔧 只有数值没有时间列（静态数据/单点数据）
                            st.info("ℹ️ Static value (No time series data)" if lang == 'en' else "ℹ️ 静态数值（无时间序列数据）")
                            if len(patient_df) == 1:
                                val = patient_df[value_col].iloc[0]
                                st.metric(label=value_col.upper(), value=f"{val}")
                            else:
                                st.dataframe(patient_df[[value_col]], width="stretch")

                        # 显示统计信息
                        if show_stats:
                            stat_title = "#### 📊 Statistical Summary" if lang == 'en' else "#### 📊 统计摘要"
                            st.markdown(stat_title)
                            stat_cols = st.columns(5)
                            values = patient_df[value_col]
                            if lang == 'en':
                                stats = [
                                    ("Min", f"{values.min():.2f}", "📉"),
                                    ("Max", f"{values.max():.2f}", "📈"),
                                    ("Mean", f"{values.mean():.2f}", "📊"),
                                    ("Std Dev", f"{values.std():.2f}", "📐"),
                                    ("Records", f"{len(values)}", "📝"),
                                ]
                            else:
                                stats = [
                                    ("最小值", f"{values.min():.2f}", "📉"),
                                    ("最大值", f"{values.max():.2f}", "📈"),
                                    ("平均值", f"{values.mean():.2f}", "📊"),
                                    ("标准差", f"{values.std():.2f}", "📐"),
                                    ("记录数", f"{len(values)}", "📝"),
                                ]
                            for i, (label, value, icon) in enumerate(stats):
                                with stat_cols[i]:
                                    st.metric(f"{icon} {label}", value)
                    else:
                        # 🔧 FIX: 检测是否有布尔列（包括pandas boolean和numpy bool）
                        bool_cols = []
                        for col in patient_df.columns:
                            dtype_str = str(patient_df[col].dtype).lower()
                            if 'bool' in dtype_str:
                                bool_cols.append(col)
                        
                        if bool_cols:
                            if lang == 'en':
                                warn_msg = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart."
                            else:
                                warn_msg = f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                        else:
                            warn_msg = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart." if lang == 'en' else f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                        st.warning(warn_msg)
                        # 🔧 显示数据表格预览，将布尔列转换为字符串
                        display_patient_df = patient_df.head(20).copy()
                        for col in display_patient_df.columns:
                            dtype_str = str(display_patient_df[col].dtype).lower()
                            if 'bool' in dtype_str:
                                display_patient_df[col] = display_patient_df[col].astype(str)
                        st.dataframe(display_patient_df, use_container_width=True)
                        
                except Exception as e:
                    err_msg = f"Chart rendering failed: {e}" if lang == 'en' else f"图表渲染失败: {e}"
                    st.warning(err_msg)
                    if 'time' in patient_df.columns:
                        chart_df = patient_df.set_index('time')
                        value_cols = [c for c in chart_df.columns if c not in [id_col]]
                        if value_cols:
                            st.line_chart(chart_df[value_cols[0]])
            else:
                no_data_msg = f"ℹ️ No {selected_concept} data for patient {patient_id}" if lang == 'en' else f"ℹ️ 患者 {patient_id} 无 {selected_concept} 数据"
                st.info(no_data_msg)
        
        # 数据表格预览
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        preview_label = "📋 Data Table Preview" if lang == 'en' else "📋 数据表格预览"
        with st.expander(preview_label, expanded=True):  # 🔧 FIX: 默认展开
            if selected_concept in st.session_state.loaded_concepts:
                df = st.session_state.loaded_concepts[selected_concept]
                if isinstance(df, pd.DataFrame):
                    if patient_id:
                        id_col = st.session_state.id_col
                        if id_col in df.columns:
                            df = df[df[id_col] == patient_id]
                    st.dataframe(df.head(50), width="stretch", hide_index=True)  # 🔧 FIX: use width instead of use_container_width
                else:
                    format_msg = "Data format does not support preview" if lang == 'en' else "数据格式不支持预览"
                    st.info(format_msg)
    
    else:  # 多患者比较模式
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
        
        with col1:
            # 🔧 FIX: 先选择模块，再选择特征
            module_label = "📂 Select Module" if lang == 'en' else "📂 选择模块"
            all_modules_opt = "All Modules" if lang == 'en' else "全部模块"
            
            # 🔧 FIX (2026-02-05): 只显示支持时序分析的模块（排除静态数据模块）
            module_options = [all_modules_opt]
            for grp_key in CONCEPT_GROUPS_INTERNAL:
                # 跳过不支持时序分析的模块（demographics, outcome）
                if grp_key not in TIME_SERIES_COMPATIBLE_MODULES:
                    continue
                grp_concepts = CONCEPT_GROUPS_INTERNAL[grp_key]
                if any(c in available_concepts for c in grp_concepts):
                    display_name = CONCEPT_GROUPS_DISPLAY.get(grp_key, grp_key)
                    module_options.append(display_name)
            
            selected_module_multi = st.selectbox(
                module_label,
                options=module_options,
                key="ts_module_multi"
            )
        
        with col2:
            # 根据选择的模块过滤概念
            if selected_module_multi == all_modules_opt:
                filtered_concepts_multi = available_concepts
            else:
                selected_grp_key = None
                for grp_key, display in CONCEPT_GROUPS_DISPLAY.items():
                    if display == selected_module_multi:
                        selected_grp_key = grp_key
                        break
                if selected_grp_key:
                    grp_concepts = CONCEPT_GROUPS_INTERNAL.get(selected_grp_key, [])
                    filtered_concepts_multi = [c for c in available_concepts if c in grp_concepts]
                else:
                    filtered_concepts_multi = available_concepts
            
            concept_label = "📋 Select Concept" if lang == 'en' else "📋 选择 Concept"
            selected_concept = st.selectbox(
                concept_label,
                options=filtered_concepts_multi if filtered_concepts_multi else available_concepts,
                key="ts_concept_multi"
            )
        
        with col3:
            if st.session_state.patient_ids:
                compare_label = "👥 Select patients to compare (max 5)" if lang == 'en' else "👥 选择要比较的患者 (最多5个)"
                compare_patients = st.multiselect(
                    compare_label,
                    options=st.session_state.patient_ids[:50],
                    default=st.session_state.patient_ids[:3],
                    max_selections=5,
                    key="ts_compare_patients"
                )
            else:
                compare_patients = []
        
        with col4:
            normalize_label = "Normalize" if lang == 'en' else "归一化比较"
            normalize_help = "Normalize values to 0-1 range for comparison" if lang == 'en' else "将数值归一化到0-1范围便于比较"
            normalize = st.checkbox(normalize_label, value=False, key="ts_normalize",
                                   help=normalize_help)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if selected_concept and compare_patients:
            try:
                import plotly.graph_objects as go
                
                df = st.session_state.loaded_concepts[selected_concept]
                
                # 确保是 DataFrame
                if not isinstance(df, pd.DataFrame):
                    format_warn = f"Data format not supported for multi-patient comparison: {type(df).__name__}" if lang == 'en' else f"数据格式不支持多患者比较: {type(df).__name__}"
                    st.warning(format_warn)
                    return
                
                id_col = st.session_state.id_col
                
                # 确定数值列
                numeric_cols = df.select_dtypes(include=['number']).columns
                # 排除ID列和所有可能的时间列
                exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'index', 'time',
                               'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                               'patientunitstayid', 'admissionid', 'patientid', 'CaseID', 'Offset',
                               'measuredat_minutes', 'measuredat']
                value_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                # 检测时间列
                time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat']
                time_col = None
                for tc in time_candidates:
                    if tc in df.columns:
                        time_col = tc
                        break
                
                if value_cols and time_col and id_col in df.columns:
                    value_col = value_cols[0]
                    
                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    
                    comparison_stats = []
                    
                    for i, pid in enumerate(compare_patients):
                        patient_df = df[df[id_col] == pid].sort_values(time_col)
                        
                        if len(patient_df) > 0:
                            y_values = patient_df[value_col].values
                            
                            # 归一化
                            if normalize and len(y_values) > 0:
                                y_min, y_max = y_values.min(), y_values.max()
                                if y_max > y_min:
                                    y_values = (y_values - y_min) / (y_max - y_min)
                            
                            patient_label = f"Patient {pid}" if lang == 'en' else f"患者 {pid}"
                            fig.add_trace(go.Scatter(
                                x=patient_df[time_col],
                                y=y_values,
                                mode='lines+markers',
                                name=patient_label,
                                line=dict(color=colors[i % len(colors)], width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Build stats with language-aware column names
                            if lang == 'en':
                                comparison_stats.append({
                                    'Patient': pid,
                                    'Mean': f"{patient_df[value_col].mean():.2f}",
                                    'Max': f"{patient_df[value_col].max():.2f}",
                                    'Min': f"{patient_df[value_col].min():.2f}",
                                    'Records': len(patient_df)
                                })
                            else:
                                comparison_stats.append({
                                    '患者': pid,
                                    '平均值': f"{patient_df[value_col].mean():.2f}",
                                    '最大值': f"{patient_df[value_col].max():.2f}",
                                    '最小值': f"{patient_df[value_col].min():.2f}",
                                    '记录数': len(patient_df)
                                })
                    
                    # Language-aware chart labels
                    chart_title = f"📊 {selected_concept.upper()} Multi-Patient Comparison" if lang == 'en' else f"📊 {selected_concept.upper()} 多患者比较"
                    x_axis_label = "Time (hours)" if lang == 'en' else "时间 (小时)"
                    y_suffix = " (Normalized)" if lang == 'en' else " (归一化)"
                    fig.update_layout(
                        template="plotly_white",
                        title=chart_title,
                        xaxis_title=x_axis_label,
                        yaxis_title=f"{value_col}" + (y_suffix if normalize else ""),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        height=450,
                        font=dict(size=14, color='black'),
                    )
                    fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 比较统计表
                    if comparison_stats:
                        compare_stats_title = "#### 📊 Comparison Statistics" if lang == 'en' else "#### 📊 比较统计"
                        st.markdown(compare_stats_title)
                        st.dataframe(pd.DataFrame(comparison_stats), width="stretch", hide_index=True)
                else:
                    # 🔧 FIX: 检测是否有布尔列（包括pandas boolean和numpy bool）
                    bool_cols = []
                    for col in df.columns:
                        dtype_str = str(df[col].dtype).lower()
                        if 'bool' in dtype_str:
                            bool_cols.append(col)
                    
                    if bool_cols:
                        if lang == 'en':
                            format_warn = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart."
                        else:
                            format_warn = f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                    else:
                        format_warn = f"⚠️ **{selected_concept.upper()}** is a Boolean (True/False) feature. Time Series Analysis requires numeric values and cannot display boolean data as a chart." if lang == 'en' else f"⚠️ **{selected_concept.upper()}** 是布尔类型（True/False）特征。时序分析需要数值型数据，无法将布尔数据显示为图表。"
                    st.warning(format_warn)
                    
            except Exception as e:
                err_msg = f"Comparison chart rendering failed: {e}" if lang == 'en' else f"比较图表渲染失败: {e}"
                st.error(err_msg)


def render_patient_page():
    """渲染患者视图页面。"""
    lang = st.session_state.get('language', 'en')
    loaded_concepts_map = st.session_state.loaded_concepts

    def _patient_concept_frame(concept_name, patient_id, id_col_name):
        frame = loaded_concepts_map.get(concept_name)
        if not isinstance(frame, pd.DataFrame) or id_col_name not in frame.columns:
            return None
        patient_frame = frame[frame[id_col_name] == patient_id].copy()
        if patient_frame.empty:
            return None
        return patient_frame

    def _latest_patient_value(concept_name, patient_id, id_col_name):
        patient_frame = _patient_concept_frame(concept_name, patient_id, id_col_name)
        if patient_frame is None:
            return None
        value_col = concept_name if concept_name in patient_frame.columns else None
        if value_col is None:
            excluded_cols = {id_col_name, 'time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat'}
            candidates = [c for c in patient_frame.columns if c not in excluded_cols]
            if not candidates:
                return None
            value_col = candidates[-1]
        series = patient_frame[value_col].dropna()
        if series.empty:
            return None
        return series.iloc[-1]
    
    _pat_title = "Patient Overview" if lang == 'en' else "患者综合视图"
    _pat_sub = "Multi-dimensional patient dashboard" if lang == 'en' else "多维度患者仪表盘"
    st.markdown(f'''
    <div style="margin-bottom:16px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{_pat_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_pat_sub}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        _msg = "Load data to view patient dashboards." if lang == 'en' else "请先加载数据以查看患者视图。"
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">🏥</div>
            <div style="font-weight:600;color:#111827">{_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    if not st.session_state.patient_ids:
        warn_msg = "⚠️ No patient data found" if lang == 'en' else "⚠️ 未找到患者数据"
        st.warning(warn_msg)
        return
    
    # ============ Patient Summary Header (审稿式 Case Review) ============
    def _render_patient_summary_card(pid):
        """渲染患者摘要卡片"""
        loaded = st.session_state.loaded_concepts
        id_col = st.session_state.get('id_col', 'stay_id')
        _age = _los = _mort = _sex = "—"
        _supports = []
        
        for cname, df in loaded.items():
            if df is None or not hasattr(df, 'columns') or id_col not in df.columns:
                continue
            pdf = df[df[id_col] == pid]
            if pdf.empty or cname not in pdf.columns:
                continue
            v = pdf[cname].dropna()
            if len(v) == 0:
                continue
            if cname == 'age':
                _age = f"{float(v.iloc[0]):.0f}"
            elif cname == 'sex':
                _sex = str(v.iloc[0])
            elif cname == 'los_icu':
                _los = f"{float(v.iloc[0]):.1f}d"
            elif cname == 'death':
                _mort = "☠️ Yes" if float(v.iloc[0]) == 1 else "✅ No"
            elif cname == 'mech_vent' and float(v.iloc[0]) > 0:
                _supports.append("🌬️ MV")
            elif cname == 'rrt' and float(v.iloc[0]) > 0:
                _supports.append("🚰 RRT")
            elif cname in ('norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate') and float(v.max()) > 0:
                _supports.append("💉 Vasopressors")
        
        _supports = list(set(_supports))
        _supports_str = ", ".join(_supports) if _supports else "—"
        _demo_lbl = get_text('demographics_header')
        _los_lbl = get_text('icu_los_label')
        _mort_lbl = get_text('mortality_label')
        _sup_lbl = get_text('key_supports')
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border:1px solid #bae6fd;border-radius:14px;padding:16px 20px;margin-bottom:16px">
            <div style="font-size:1.1rem;font-weight:800;color:#0369a1;margin-bottom:10px">📋 {get_text('patient_summary')} — Patient {pid}</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px">
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_demo_lbl}</div><div style="font-weight:600">{_sex}, {_age}y</div></div>
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_los_lbl}</div><div style="font-weight:600">{_los}</div></div>
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_mort_lbl}</div><div style="font-weight:600">{_mort}</div></div>
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_sup_lbl}</div><div style="font-weight:600">{_supports_str}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 患者选择面板
    select_title = "Patient Selection" if lang == 'en' else "患者选择"
    st.markdown(f'''
    <div style="font-size:1.05rem;font-weight:700;color:#111827;margin-bottom:10px">{select_title}</div>
    ''', unsafe_allow_html=True)
    
    # 快速导航按钮
    first_btn = "⏮️ First" if lang == 'en' else "⏮️ 首位"
    prev_btn = "⬅️ Previous" if lang == 'en' else "⬅️ 上一位"
    next_btn = "➡️ Next" if lang == 'en' else "➡️ 下一位"
    last_btn = "⏭️ Last" if lang == 'en' else "⏭️ 末位"
    rand_btn = "🎲 Random" if lang == 'en' else "🎲 随机"
    first_help = "Jump to first patient" if lang == 'en' else "跳转到第一位患者"
    prev_help = "Previous patient" if lang == 'en' else "上一位患者"
    next_help = "Next patient" if lang == 'en' else "下一位患者"
    last_help = "Jump to last patient" if lang == 'en' else "跳转到最后一位患者"
    rand_help = "Random select a patient" if lang == 'en' else "随机选择一位患者"
    
    nav_cols = st.columns(6)
    with nav_cols[0]:
        if st.button(first_btn, width="stretch", help=first_help):
            st.session_state.patient_view_id = st.session_state.patient_ids[0]
            st.rerun()
    with nav_cols[1]:
        if st.button(prev_btn, width="stretch", help=prev_help):
            current_idx = st.session_state.patient_ids.index(st.session_state.get('patient_view_id', st.session_state.patient_ids[0]))
            if current_idx > 0:
                st.session_state.patient_view_id = st.session_state.patient_ids[current_idx - 1]
                st.rerun()
    with nav_cols[2]:
        if st.button(next_btn, width="stretch", help=next_help):
            current_idx = st.session_state.patient_ids.index(st.session_state.get('patient_view_id', st.session_state.patient_ids[0]))
            if current_idx < len(st.session_state.patient_ids) - 1:
                st.session_state.patient_view_id = st.session_state.patient_ids[current_idx + 1]
                st.rerun()
    with nav_cols[3]:
        if st.button(last_btn, width="stretch", help=last_help):
            st.session_state.patient_view_id = st.session_state.patient_ids[-1]
            st.rerun()
    with nav_cols[4]:
        if st.button(rand_btn, width="stretch", help=rand_help):
            import random
            st.session_state.patient_view_id = random.choice(st.session_state.patient_ids)
            st.rerun()
    with nav_cols[5]:
        # 显示当前位置
        current_idx = st.session_state.patient_ids.index(st.session_state.get('patient_view_id', st.session_state.patient_ids[0]))
        st.markdown(f"<div style='text-align:center;padding:0.5rem;background:rgba(30,40,50,0.6);border-radius:4px'>{current_idx + 1}/{len(st.session_state.patient_ids)}</div>", unsafe_allow_html=True)
    
    # ============ Render Patient Summary Card ============
    _current_pid = st.session_state.get('patient_view_id', st.session_state.patient_ids[0] if st.session_state.patient_ids else None)
    if _current_pid is not None:
        try:
            _render_patient_summary_card(_current_pid)
        except Exception:
            pass
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        pat_id_label = "👤 Patient ID" if lang == 'en' else "👤 患者 ID"
        patient_id = st.selectbox(
            pat_id_label,
            options=st.session_state.patient_ids[:100],
            key="patient_view_id"
        )
    
    with col2:
        view_label = "📋 View Mode" if lang == 'en' else "📋 显示模式"
        view_options = ["Dashboard", "Category View", "Data Table"] if lang == 'en' else ["综合仪表盘", "分类视图", "数据表格"]
        view_mode = st.selectbox(
            view_label,
            options=view_options,
            key="patient_view_mode"
        )
    
    with col3:
        # 数据概览 - 显示更详细的可用数据信息
        id_col = st.session_state.id_col
        available_concepts = [k for k, v in st.session_state.loaded_concepts.items() 
                             if isinstance(v, pd.DataFrame) and id_col in v.columns 
                             and patient_id in v[id_col].values]
        n_concepts = len(available_concepts)
        
        # 统计各类别数据
        vitals_list = ['hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2']
        labs_list = ['bili', 'crea', 'lac', 'plt', 'wbc', 'hgb', 'inr_pt', 'ptt']
        scores_list = ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'sep3_sofa1', 'sep3_sofa2']
        
        n_vitals = len([c for c in available_concepts if c in vitals_list])
        n_labs = len([c for c in available_concepts if c in labs_list])
        n_scores = len([c for c in available_concepts if c in scores_list])
        
        data_label = "Available Data" if lang == 'en' else "可用数据"
        st.markdown(f'''
        <div class="metric-card" style="padding:0.5rem 1rem">
            <div class="stat-label">{data_label}</div>
            <div style="display:flex;gap:1rem;font-size:0.9rem">
                <span>📊 {n_concepts} total</span>
                <span>❤️ {n_vitals} vitals</span>
                <span>🧪 {n_labs} labs</span>
                <span>📈 {n_scores} scores</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 判断视图模式
    dashboard_mode = "Dashboard" if lang == 'en' else "综合仪表盘"
    category_mode = "Category View" if lang == 'en' else "分类视图"
    table_mode = "Data Table" if lang == 'en' else "数据表格"
    
    if patient_id:
        st.session_state.selected_patient = patient_id
        id_col = st.session_state.id_col
        
        if view_mode == dashboard_mode:
            # 自定义综合仪表盘
            dash_title = "### 📊 Dashboard" if lang == 'en' else "### 📊 综合仪表盘"
            st.markdown(dash_title)
            
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # 收集所有生命体征数据
                vitals = ['hr', 'map', 'sbp', 'resp', 'spo2']
                vitals_data = {}
                time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat']
                
                for v in vitals:
                    if v in st.session_state.loaded_concepts:
                        df = st.session_state.loaded_concepts[v]
                        if isinstance(df, pd.DataFrame) and id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                            if len(patient_df) > 0:
                                # 检测时间列
                                time_col = None
                                for tc in time_candidates:
                                    if tc in patient_df.columns:
                                        time_col = tc
                                        break
                                if time_col:
                                    vitals_data[v] = (patient_df, time_col)
                
                if vitals_data:
                    # 创建多行子图
                    n_vitals = len(vitals_data)
                    fig = make_subplots(
                        rows=n_vitals, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[v.upper() for v in vitals_data.keys()]
                    )
                    
                    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
                    
                    for i, (name, (df, time_col)) in enumerate(vitals_data.items(), 1):
                        value_col = name if name in df.columns else df.columns[-1]
                        fig.add_trace(
                            go.Scatter(
                                x=df[time_col], y=df[value_col],
                                mode='lines+markers',
                                name=name.upper(),
                                line=dict(color=colors[(i-1) % len(colors)], width=2),
                                marker=dict(size=4)
                            ),
                            row=i, col=1
                        )
                    
                    vitals_title = f"Patient {patient_id} Vital Signs Trend" if lang == 'en' else f"患者 {patient_id} 生命体征趋势"
                    fig.update_layout(
                        height=150 * n_vitals + 100,
                        template="plotly_white",
                        showlegend=False,
                        title_text=vitals_title,
                        title_font_size=16,
                        margin=dict(l=50, r=30, t=60, b=50),
                        font=dict(size=14, color='black'),
                    )
                    fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    no_vitals = "ℹ️ No standard vital signs are present in the current loaded features" if lang == 'en' else "ℹ️ 当前已加载特征中不包含标准生命体征"
                    st.info(no_vitals)
                
                # SOFA 评分趋势
                if 'sofa' in st.session_state.loaded_concepts:
                    sofa_df = st.session_state.loaded_concepts['sofa']
                    if isinstance(sofa_df, pd.DataFrame) and id_col in sofa_df.columns:
                        patient_sofa = sofa_df[sofa_df[id_col] == patient_id]
                        # 检测时间列
                        sofa_time_col = None
                        for tc in time_candidates:
                            if tc in patient_sofa.columns:
                                sofa_time_col = tc
                                break
                        
                        if len(patient_sofa) > 0 and sofa_time_col:
                            sofa_trend = "#### 📈 SOFA Score Trend" if lang == 'en' else "#### 📈 SOFA 评分趋势"
                            st.markdown(sofa_trend)
                            
                            # SOFA 分解堆叠图
                            sofa_components = ['sofa_resp', 'sofa_coag', 'sofa_liver', 
                                             'sofa_cardio', 'sofa_cns', 'sofa_renal']
                            available_components = [c for c in sofa_components if c in patient_sofa.columns]
                            
                            if available_components:
                                fig = go.Figure()
                                colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff', '#5f27cd']
                                
                                for i, comp in enumerate(available_components):
                                    fig.add_trace(go.Bar(
                                        x=patient_sofa[sofa_time_col],
                                        y=patient_sofa[comp],
                                        name=comp.replace('sofa_', '').upper(),
                                        marker_color=colors[i]
                                    ))
                                
                                time_label = "Time" if lang == 'en' else "时间"
                                score_label = "SOFA Score" if lang == 'en' else "SOFA 分数"
                                fig.update_layout(
                                    barmode='stack',
                                    template="plotly_white",
                                    height=350,
                                    xaxis_title=time_label,
                                    yaxis_title=score_label,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                    font=dict(size=14, color='black'),
                                )
                                fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                                fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                # ============ SOFA-1 vs SOFA-2 对比图表 ============
                has_sofa1 = 'sofa' in st.session_state.loaded_concepts
                has_sofa2 = 'sofa2' in st.session_state.loaded_concepts
                
                if has_sofa1 and has_sofa2:
                    compare_title = "#### 🔄 SOFA-1 vs SOFA-2 Comparison" if lang == 'en' else "#### 🔄 SOFA-1 与 SOFA-2 对比"
                    st.markdown(compare_title)
                    
                    sofa1_df = st.session_state.loaded_concepts['sofa']
                    sofa2_df = st.session_state.loaded_concepts['sofa2']
                    
                    # 获取患者数据
                    if isinstance(sofa1_df, pd.DataFrame) and id_col in sofa1_df.columns:
                        patient_sofa1 = sofa1_df[sofa1_df[id_col] == patient_id].copy()
                    else:
                        patient_sofa1 = pd.DataFrame()
                    
                    if isinstance(sofa2_df, pd.DataFrame) and id_col in sofa2_df.columns:
                        patient_sofa2 = sofa2_df[sofa2_df[id_col] == patient_id].copy()
                    else:
                        patient_sofa2 = pd.DataFrame()
                    
                    if len(patient_sofa1) > 0 and len(patient_sofa2) > 0:
                        # 检测时间列
                        time_col1 = None
                        time_col2 = None
                        for tc in time_candidates:
                            if tc in patient_sofa1.columns and time_col1 is None:
                                time_col1 = tc
                            if tc in patient_sofa2.columns and time_col2 is None:
                                time_col2 = tc
                        
                        if time_col1 and time_col2:
                            # 1. 总分对比折线图
                            total_compare = "**Total Score Comparison**" if lang == 'en' else "**总分对比**"
                            st.markdown(total_compare)
                            
                            fig_total = go.Figure()
                            
                            # SOFA-1 总分
                            if 'sofa' in patient_sofa1.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa1[time_col1],
                                    y=patient_sofa1['sofa'],
                                    mode='lines+markers',
                                    name='SOFA-1 (Traditional)',
                                    line=dict(color='#1f77b4', width=3),
                                    marker=dict(size=8)
                                ))
                            
                            # SOFA-2 总分
                            if 'sofa2' in patient_sofa2.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa2[time_col2],
                                    y=patient_sofa2['sofa2'],
                                    mode='lines+markers',
                                    name='SOFA-2 (2025 New)',
                                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                                    marker=dict(size=8, symbol='diamond')
                                ))
                            
                            time_label = "Time (hours from ICU admission)" if lang == 'en' else "时间 (ICU入院后小时)"
                            score_label = "Total SOFA Score" if lang == 'en' else "SOFA 总分"
                            fig_total.update_layout(
                                template="plotly_white",
                                height=300,
                                xaxis_title=time_label,
                                yaxis_title=score_label,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                hovermode='x unified',
                                font=dict(size=14, color='black'),
                            )
                            fig_total.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            fig_total.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            
                            st.plotly_chart(fig_total, use_container_width=True)
                            
                            # 2. 子器官评分对比（6个子图）
                            organ_compare = "**Organ-specific Score Comparison**" if lang == 'en' else "**各器官评分对比**"
                            st.markdown(organ_compare)
                            
                            # 定义器官映射
                            organ_pairs = [
                                ('sofa_resp', 'sofa2_resp', 'Respiratory', '呼吸'),
                                ('sofa_coag', 'sofa2_coag', 'Coagulation', '凝血'),
                                ('sofa_liver', 'sofa2_liver', 'Liver', '肝脏'),
                                ('sofa_cardio', 'sofa2_cardio', 'Cardiovascular', '心血管'),
                                ('sofa_cns', 'sofa2_cns', 'Neurological', '神经'),
                                ('sofa_renal', 'sofa2_renal', 'Renal', '肾脏'),
                            ]
                            
                            # 🔧 检查器官评分列是否存在于各自的 DataFrame 中
                            # 如果不存在，尝试从其他加载的 concepts 中获取
                            def get_organ_data(patient_df, organ_col, time_col, loaded_concepts, id_col, patient_id):
                                """获取器官评分数据，优先从 sofa/sofa2 DataFrame，否则从单独加载的 concept"""
                                try:
                                    if organ_col in patient_df.columns and time_col in patient_df.columns:
                                        return patient_df[[time_col, organ_col]].copy()
                                    # 尝试从单独加载的 concept 获取
                                    if organ_col in loaded_concepts:
                                        organ_df = loaded_concepts[organ_col]
                                        if isinstance(organ_df, pd.DataFrame) and id_col in organ_df.columns:
                                            patient_organ = organ_df[organ_df[id_col] == patient_id].copy()
                                            if len(patient_organ) > 0 and organ_col in patient_organ.columns:
                                                # 找时间列
                                                for tc in ['time', 'charttime', 'starttime']:
                                                    if tc in patient_organ.columns:
                                                        return patient_organ[[tc, organ_col]].rename(columns={tc: time_col})
                                except Exception:
                                    pass
                                return None
                            
                            # 创建 2x3 子图
                            from plotly.subplots import make_subplots
                            
                            fig_organs = make_subplots(
                                rows=2, cols=3,
                                subplot_titles=[p[2] if lang == 'en' else p[3] for p in organ_pairs],
                                vertical_spacing=0.15,
                                horizontal_spacing=0.08
                            )
                            
                            has_any_data = False
                            for idx, (sofa1_col, sofa2_col, en_name, zh_name) in enumerate(organ_pairs):
                                row = idx // 3 + 1
                                col = idx % 3 + 1
                                
                                # SOFA-1 器官评分
                                sofa1_organ = get_organ_data(patient_sofa1, sofa1_col, time_col1, 
                                                            st.session_state.loaded_concepts, id_col, patient_id)
                                if sofa1_organ is not None and len(sofa1_organ) > 0:
                                    has_any_data = True
                                    fig_organs.add_trace(
                                        go.Scatter(
                                            x=sofa1_organ[time_col1],
                                            y=sofa1_organ[sofa1_col],
                                            mode='lines+markers',
                                            name='SOFA-1' if idx == 0 else None,
                                            legendgroup='sofa1',
                                            showlegend=(idx == 0),
                                            line=dict(color='#1f77b4', width=2),
                                            marker=dict(size=5)
                                        ),
                                        row=row, col=col
                                    )
                                
                                # SOFA-2 器官评分
                                sofa2_organ = get_organ_data(patient_sofa2, sofa2_col, time_col2,
                                                            st.session_state.loaded_concepts, id_col, patient_id)
                                if sofa2_organ is not None and len(sofa2_organ) > 0:
                                    has_any_data = True
                                    fig_organs.add_trace(
                                        go.Scatter(
                                            x=sofa2_organ[time_col2],
                                            y=sofa2_organ[sofa2_col],
                                            mode='lines+markers',
                                            name='SOFA-2' if idx == 0 else None,
                                            legendgroup='sofa2',
                                            showlegend=(idx == 0),
                                            line=dict(color='#ff7f0e', width=2, dash='dash'),
                                            marker=dict(size=5, symbol='diamond')
                                        ),
                                        row=row, col=col
                                    )
                            
                            if has_any_data:
                                fig_organs.update_layout(
                                    height=500,
                                    template="plotly_white",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                                    hovermode='x unified',
                                    font=dict(size=14, color='black'),
                                )
                                
                                # 更新 y 轴范围 (0-4)
                                for i in range(1, 7):
                                    fig_organs.update_yaxes(range=[0, 4.5], row=(i-1)//3+1, col=(i-1)%3+1,
                                                           tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                                fig_organs.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                                
                                st.plotly_chart(fig_organs, use_container_width=True)
                            else:
                                no_organ_msg = "ℹ️ Organ-specific scores not available in current data. Load individual organ concepts (e.g., sofa_resp, sofa2_resp) to see detailed comparison." if lang == 'en' else "ℹ️ 当前数据中无法获取器官子评分。请加载单独的器官概念（如 sofa_resp, sofa2_resp）以查看详细对比。"
                                st.info(no_organ_msg)
                            
                            # 3. 差异分析表格
                            diff_title = "**Score Difference (SOFA-2 - SOFA-1)**" if lang == 'en' else "**评分差异 (SOFA-2 - SOFA-1)**"
                            st.markdown(diff_title)
                            
                            # 计算最新时间点的差异
                            latest_sofa1 = patient_sofa1.iloc[-1] if len(patient_sofa1) > 0 else {}
                            latest_sofa2 = patient_sofa2.iloc[-1] if len(patient_sofa2) > 0 else {}
                            
                            diff_data = []
                            for sofa1_col, sofa2_col, en_name, zh_name in organ_pairs:
                                val1 = latest_sofa1.get(sofa1_col, 0) if isinstance(latest_sofa1, dict) or hasattr(latest_sofa1, 'get') else (latest_sofa1[sofa1_col] if sofa1_col in latest_sofa1.index else 0)
                                val2 = latest_sofa2.get(sofa2_col, 0) if isinstance(latest_sofa2, dict) or hasattr(latest_sofa2, 'get') else (latest_sofa2[sofa2_col] if sofa2_col in latest_sofa2.index else 0)
                                diff = val2 - val1
                                organ_name = en_name if lang == 'en' else zh_name
                                diff_data.append({
                                    'Organ' if lang == 'en' else '器官': organ_name,
                                    'SOFA-1': int(val1),
                                    'SOFA-2': int(val2),
                                    'Diff' if lang == 'en' else '差异': int(diff)
                                })
                            
                            # 总分差异
                            total1 = latest_sofa1.get('sofa', 0) if isinstance(latest_sofa1, dict) or hasattr(latest_sofa1, 'get') else (latest_sofa1['sofa'] if 'sofa' in latest_sofa1.index else 0)
                            total2 = latest_sofa2.get('sofa2', 0) if isinstance(latest_sofa2, dict) or hasattr(latest_sofa2, 'get') else (latest_sofa2['sofa2'] if 'sofa2' in latest_sofa2.index else 0)
                            diff_data.append({
                                'Organ' if lang == 'en' else '器官': '**Total**' if lang == 'en' else '**总分**',
                                'SOFA-1': int(total1),
                                'SOFA-2': int(total2),
                                'Diff' if lang == 'en' else '差异': int(total2 - total1)
                            })
                            
                            diff_df = pd.DataFrame(diff_data)
                            st.dataframe(diff_df, width="stretch", hide_index=True)
                    else:
                        no_compare = "ℹ️ Need both SOFA-1 and SOFA-2 data for comparison" if lang == 'en' else "ℹ️ 需要同时有 SOFA-1 和 SOFA-2 数据才能对比"
                        st.info(no_compare)
                
                # Dashboard 快速摘要面板
                summary_title = "#### 📋 Quick Summary" if lang == 'en' else "#### 📋 快速摘要"
                st.markdown(summary_title)
                
                summary_cols = st.columns(4)
                not_selected_status = "Not selected ⚪" if lang == 'en' else "未选择 ⚪"
                
                # Sepsis 状态
                with summary_cols[0]:
                    sepsis_status = not_selected_status
                    sepsis_color = "#6c757d"
                    
                    found_sep = False
                    if 'sep3_sofa2' in st.session_state.loaded_concepts:
                        sep_df = st.session_state.loaded_concepts['sep3_sofa2']
                        concept_key = 'sep3_sofa2'
                        found_sep = True
                    elif 'sep3_sofa1' in st.session_state.loaded_concepts:
                        sep_df = st.session_state.loaded_concepts['sep3_sofa1']
                        concept_key = 'sep3_sofa1'
                        found_sep = True
                    
                    if found_sep:
                        sepsis_status = "Unknown"
                        if isinstance(sep_df, pd.DataFrame) and id_col in sep_df.columns:
                            patient_sep = sep_df[sep_df[id_col] == patient_id]
                            if len(patient_sep) > 0 and concept_key in patient_sep.columns:
                                if patient_sep[concept_key].max() == 1:
                                    sepsis_status = "Sepsis ⚠️" if lang == 'en' else "脓毒症 ⚠️"
                                    sepsis_color = "#dc3545"
                                else:
                                    sepsis_status = "No Sepsis ✅" if lang == 'en' else "无脓毒症 ✅"
                                    sepsis_color = "#28a745"
                            else:
                                sepsis_status = "No Records" if lang == 'en' else "无记录"

                    st.markdown(f"**Sepsis-3**" if lang == 'en' else f"**脓毒症-3**")
                    st.markdown(f"<span style='color:{sepsis_color};font-weight:bold'>{sepsis_status}</span>", unsafe_allow_html=True)
                
                # 机械通气
                with summary_cols[1]:
                    vent_status = not_selected_status
                    vent_concepts = ['vent_ind', 'mech_vent', 'vent_start']
                    
                    # 检查是否有相关 concept 被加载
                    found_vent = any(c in st.session_state.loaded_concepts for c in vent_concepts)
                    
                    if found_vent:
                        vent_status = "Unknown"
                        if 'vent_ind' in st.session_state.loaded_concepts:
                            vent_df = st.session_state.loaded_concepts['vent_ind']
                            if isinstance(vent_df, pd.DataFrame) and id_col in vent_df.columns:
                                patient_vent = vent_df[vent_df[id_col] == patient_id]
                                if len(patient_vent) > 0 and 'vent_ind' in patient_vent.columns:
                                    vent_status = "Yes ✅" if patient_vent['vent_ind'].max() == 1 else "No ❌"
                                else:
                                    vent_status = "No Records" if lang == 'en' else "无记录"
                    
                    st.markdown(f"**Mechanical Vent**" if lang == 'en' else f"**机械通气**")
                    st.markdown(vent_status)
                
                # 血管活性药物
                with summary_cols[2]:
                    vaso_status = not_selected_status
                    vaso_concepts = ['norepi_rate', 'epi_rate', 'dopa_rate', 'vaso_ind']
                    
                    found_vaso = any(c in st.session_state.loaded_concepts for c in vaso_concepts)
                    
                    if found_vaso:
                        vaso_status = "No ❌"
                        for vc in vaso_concepts:
                            if vc in st.session_state.loaded_concepts:
                                vdf = st.session_state.loaded_concepts[vc]
                                if isinstance(vdf, pd.DataFrame) and id_col in vdf.columns:
                                    pvdf = vdf[vdf[id_col] == patient_id]
                                    if len(pvdf) > 0:
                                        val_col = vc if vc in pvdf.columns else pvdf.columns[-1]
                                        if pvdf[val_col].max() > 0:
                                            vaso_status = "Yes ✅"
                                            break
                    
                    st.markdown(f"**Vasopressors**" if lang == 'en' else f"**血管活性药**")
                    st.markdown(vaso_status)
                
                # GCS
                with summary_cols[3]:
                    gcs_val = "Not selected" if lang == 'en' else "未选择"
                    gcs_color = "#6c757d"
                    
                    if 'gcs' in st.session_state.loaded_concepts:
                        gcs_val = "N/A"
                        gcs_df = st.session_state.loaded_concepts['gcs']
                        if isinstance(gcs_df, pd.DataFrame) and id_col in gcs_df.columns:
                            patient_gcs = gcs_df[gcs_df[id_col] == patient_id]
                            if len(patient_gcs) > 0 and 'gcs' in patient_gcs.columns:
                                val = patient_gcs['gcs'].iloc[-1]
                                try:
                                    val_num = float(val)
                                    gcs_color = "#28a745" if val_num >= 13 else ("#ffc107" if val_num >= 9 else "#dc3545")
                                    gcs_val = safe_format_number(val_num, 0)
                                except (ValueError, TypeError):
                                    gcs_val = str(val)
                                    gcs_color = "#6c757d"
                            else:
                                gcs_val = "No Records" if lang == 'en' else "无记录"
                    # 尝试从 sofa_cns 推断
                    elif 'sofa_cns' in st.session_state.loaded_concepts or 'sofa2_cns' in st.session_state.loaded_concepts:
                        cns_col = 'sofa_cns' if 'sofa_cns' in st.session_state.loaded_concepts else 'sofa2_cns'
                        cns_df = st.session_state.loaded_concepts[cns_col]
                        if isinstance(cns_df, pd.DataFrame) and id_col in cns_df.columns:
                            patient_cns = cns_df[cns_df[id_col] == patient_id]
                            if len(patient_cns) > 0 and cns_col in patient_cns.columns:
                                cns_score = patient_cns[cns_col].iloc[-1]
                                # 0:15, 1:13-14, 2:10-12, 3:6-9, 4:<6
                                if cns_score == 0: gcs_val, gcs_color = "15 (est)", "#28a745"
                                elif cns_score == 1: gcs_val, gcs_color = "13-14 (est)", "#28a745"
                                elif cns_score == 2: gcs_val, gcs_color = "10-12 (est)", "#ffc107"
                                elif cns_score == 3: gcs_val, gcs_color = "6-9 (est)", "#dc3545"
                                elif cns_score == 4: gcs_val, gcs_color = "<6 (est)", "#dc3545"
                    
                    st.markdown("**GCS**")
                    st.markdown(f"<span style='color:{gcs_color};font-weight:bold;font-size:1.2rem'>{gcs_val}</span>", unsafe_allow_html=True)

                snapshot_candidates = []
                snapshot_excluded = {
                    'sep3_sofa1', 'sep3_sofa2', 'vent_ind', 'mech_vent', 'vent_start',
                    'norepi_rate', 'epi_rate', 'dopa_rate', 'vaso_ind', 'gcs'
                }
                for concept_name in available_concepts:
                    if concept_name in snapshot_excluded:
                        continue
                    latest_value = _latest_patient_value(concept_name, patient_id, id_col)
                    if latest_value is None:
                        continue
                    try:
                        formatted = safe_format_number(float(latest_value), 2)
                    except Exception:
                        formatted = str(latest_value)
                    snapshot_candidates.append((concept_name, formatted))

                if snapshot_candidates:
                    snapshot_title = "#### 🧩 Loaded Feature Snapshot" if lang == 'en' else "#### 🧩 已加载特征快照"
                    st.markdown(snapshot_title)
                    visible_snapshots = snapshot_candidates[:8]
                    snap_cols = st.columns(min(4, len(visible_snapshots)))
                    for idx, (concept_name, formatted) in enumerate(visible_snapshots):
                        with snap_cols[idx % len(snap_cols)]:
                            st.markdown(
                                f'<div class="tiny-stat-card"><div class="tiny-label">{concept_name}</div><div class="tiny-value">{formatted}</div></div>',
                                unsafe_allow_html=True,
                            )
                    if len(snapshot_candidates) > len(visible_snapshots):
                        more_msg = f"Showing {len(visible_snapshots)} of {len(snapshot_candidates)} loaded features for this patient." if lang == 'en' else f"当前展示该患者 {len(snapshot_candidates)} 个已加载特征中的前 {len(visible_snapshots)} 个。"
                        st.caption(more_msg)
                            
            except Exception as e:
                err_msg = f"Dashboard rendering failed: {e}" if lang == 'en' else f"综合仪表盘渲染失败: {e}"
                st.warning(err_msg)
                switch_msg = "Please try switching to 'Category View'" if lang == 'en' else "请尝试切换到「分类视图」"
                st.info(switch_msg)
        
        elif view_mode == category_mode:
            # 时间列候选（提前定义，避免UnboundLocalError）
            time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat']
            
            # 生命体征
            vitals_title = "### ❤️ Vital Signs" if lang == 'en' else "### ❤️ 生命体征"
            st.markdown(vitals_title)
            vitals = ['hr', 'map', 'sbp', 'resp', 'temp', 'spo2']
            vitals_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                          if k in vitals and isinstance(v, pd.DataFrame)}
            
            if vitals_data:
                cols = st.columns(min(3, len(vitals_data)))
                
                for i, (concept, df) in enumerate(vitals_data.items()):
                    with cols[i % 3]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            # 显示最新值
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            latest_val = patient_df[value_col].iloc[-1]
                            st.metric(concept.upper(), safe_format_number(latest_val, 1))
                            
                            # 小型趋势图 - 检测时间列
                            time_col = None
                            for tc in time_candidates:
                                if tc in patient_df.columns:
                                    time_col = tc
                                    break
                            if time_col:
                                st.line_chart(patient_df.set_index(time_col)[value_col], height=120)
            else:
                no_vitals = "ℹ️ No standard vital signs are present in the current loaded features" if lang == 'en' else "ℹ️ 当前已加载特征中不包含标准生命体征"
                st.info(no_vitals)
            
            # SOFA/SOFA2 评分
            sofa_concepts = ['sofa', 'sofa2']
            sofa_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in sofa_concepts and isinstance(v, pd.DataFrame)}
            
            if sofa_data:
                sofa_title = "### 📊 SOFA Score" if lang == 'en' else "### 📊 SOFA 评分"
                st.markdown(sofa_title)
                
                for sofa_key, sofa_df in sofa_data.items():
                    if id_col in sofa_df.columns:
                        patient_sofa = sofa_df[sofa_df[id_col] == patient_id]
                    else:
                        patient_sofa = sofa_df
                    
                    if len(patient_sofa) > 0:
                        latest = patient_sofa.iloc[-1]
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            sofa_val = latest.get(sofa_key, 0)
                            sofa_color = "#28a745" if sofa_val < 6 else ("#ffc107" if sofa_val < 10 else "#dc3545")
                            label = f"Latest {sofa_key.upper()}" if lang == 'en' else f"最新 {sofa_key.upper()}"
                            st.markdown(f'''
                            <div class="metric-card" style="text-align:center">
                                <div class="stat-label">{label}</div>
                                <div class="stat-number" style="color:{sofa_color}">{sofa_val}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col2:
                            sofa_time_col = None
                            for tc in time_candidates:
                                if tc in patient_sofa.columns:
                                    sofa_time_col = tc
                                    break
                            if sofa_key in patient_sofa.columns and sofa_time_col:
                                st.line_chart(patient_sofa.set_index(sofa_time_col)[sofa_key], height=150)
            
            # Sepsis-3 诊断状态
            sepsis_concepts = ['sep3_sofa1', 'sep3_sofa2', 'susp_inf', 'infection_icd']
            sepsis_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                          if k in sepsis_concepts and isinstance(v, pd.DataFrame)}
            
            if sepsis_data:
                sepsis_title = "### 🦠 Sepsis-3 Status" if lang == 'en' else "### 🦠 Sepsis-3 诊断"
                st.markdown(sepsis_title)
                cols = st.columns(len(sepsis_data))
                for i, (concept, df) in enumerate(sepsis_data.items()):
                    with cols[i]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            val = patient_df[value_col].iloc[-1] if len(patient_df) > 0 else 0
                            if val == 1:
                                st.markdown(f"✅ **{concept}**: Yes" if lang == 'en' else f"✅ **{concept}**: 是")
                            else:
                                st.markdown(f"❌ **{concept}**: No" if lang == 'en' else f"❌ **{concept}**: 否")
            
            # 实验室检查 - 扩展更多指标
            labs = ['bili', 'crea', 'lac', 'lact', 'plt', 'wbc', 'hgb', 'hct', 'inr_pt', 'ptt', 'alb', 'glu', 'na', 'k', 'cl', 'bun']
            labs_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in labs and isinstance(v, pd.DataFrame)}
            
            if labs_data:
                labs_title = "### 🧪 Laboratory Tests" if lang == 'en' else "### 🧪 实验室检查"
                st.markdown(labs_title)
                cols = st.columns(min(4, len(labs_data)))
                for i, (concept, df) in enumerate(labs_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(
                                label=concept.upper(),
                                value=f"{patient_df[value_col].iloc[-1]:.2f}",
                                delta=f"{patient_df[value_col].iloc[-1] - patient_df[value_col].iloc[0]:.2f}" if len(patient_df) > 1 else None
                            )
                            lab_time_col = None
                            for tc in time_candidates:
                                if tc in patient_df.columns:
                                    lab_time_col = tc
                                    break
                            if lab_time_col:
                                st.line_chart(patient_df.set_index(lab_time_col)[value_col], height=120)
            
            # 血气分析
            blood_gas = ['ph', 'pco2', 'po2', 'pafi', 'safi', 'be', 'hco3', 'bicar', 'fio2']
            bg_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                      if k in blood_gas and isinstance(v, pd.DataFrame)}
            
            if bg_data:
                bg_title = "### 🩸 Blood Gas Analysis" if lang == 'en' else "### 🩸 血气分析"
                st.markdown(bg_title)
                cols = st.columns(min(4, len(bg_data)))
                for i, (concept, df) in enumerate(bg_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.2f}")
            
            # 血管活性药物
            vasopressors = ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'adh_rate', 'phn_rate', 'vaso_ind']
            vaso_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in vasopressors and isinstance(v, pd.DataFrame)}
            
            if vaso_data:
                vaso_title = "### 💉 Vasopressors" if lang == 'en' else "### 💉 血管活性药物"
                st.markdown(vaso_title)
                cols = st.columns(min(4, len(vaso_data)))
                for i, (concept, df) in enumerate(vaso_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'vaso_ind':
                                val = patient_df[value_col].max()
                                st.markdown(f"**{concept}**: {'Yes ✅' if val == 1 else 'No ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.3f}")
                                vaso_time_col = None
                                for tc in time_candidates:
                                    if tc in patient_df.columns:
                                        vaso_time_col = tc
                                        break
                                if vaso_time_col:
                                    st.line_chart(patient_df.set_index(vaso_time_col)[value_col], height=100)
            
            # 呼吸支持
            resp_support = ['vent_ind', 'fio2', 'spo2', 'pafi', 'safi', 'resp']
            resp_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in resp_support and isinstance(v, pd.DataFrame) and k not in bg_data}  # 避免重复
            
            if resp_data:
                resp_title = "### 💨 Respiratory Support" if lang == 'en' else "### 💨 呼吸支持"
                st.markdown(resp_title)
                cols = st.columns(min(4, len(resp_data)))
                for i, (concept, df) in enumerate(resp_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'vent_ind':
                                val = patient_df[value_col].max()
                                st.markdown(f"**Mechanical Vent**: {'Yes ✅' if val == 1 else 'No ❌'}" if lang == 'en' else f"**机械通气**: {'是 ✅' if val == 1 else '否 ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=safe_format_number(patient_df[value_col].iloc[-1], 1))
            
            # 神经系统
            neuro = ['gcs', 'egcs', 'mgcs', 'vgcs', 'rass', 'avpu']
            neuro_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                         if k in neuro and isinstance(v, pd.DataFrame)}
            
            if neuro_data:
                neuro_title = "### 🧠 Neurological" if lang == 'en' else "### 🧠 神经系统"
                st.markdown(neuro_title)
                cols = st.columns(min(4, len(neuro_data)))
                for i, (concept, df) in enumerate(neuro_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            val = patient_df[value_col].iloc[-1]
                            # GCS 颜色编码
                            if concept == 'gcs':
                                try:
                                    val_num = float(val)
                                    color = "#28a745" if val_num >= 13 else ("#ffc107" if val_num >= 9 else "#dc3545")
                                    st.markdown(f"<div style='color:{color};font-size:1.5rem;font-weight:bold'>GCS: {safe_format_number(val_num, 0)}</div>", unsafe_allow_html=True)
                                except (ValueError, TypeError):
                                    st.markdown(f"<div style='font-size:1.5rem;font-weight:bold'>GCS: {val}</div>", unsafe_allow_html=True)
                            else:
                                st.metric(label=concept.upper(), value=safe_format_number(val, 0))
            
            # 肾脏功能
            renal = ['urine', 'urine24', 'crea', 'bun', 'rrt']
            renal_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                         if k in renal and isinstance(v, pd.DataFrame) and k not in labs_data}
            
            if renal_data:
                renal_title = "### 🚰 Renal Function" if lang == 'en' else "### 🚰 肾脏功能"
                st.markdown(renal_title)
                cols = st.columns(min(4, len(renal_data)))
                for i, (concept, df) in enumerate(renal_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'rrt':
                                val = patient_df[value_col].max()
                                st.markdown(f"**RRT**: {'Yes ✅' if val == 1 else 'No ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=safe_format_number(patient_df[value_col].iloc[-1], 1))
            
            # 其他评分
            other_scores = ['qsofa', 'sirs', 'mews', 'news']
            score_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                         if k in other_scores and isinstance(v, pd.DataFrame)}
            
            if score_data:
                score_title = "### 📈 Other Scores" if lang == 'en' else "### 📈 其他评分"
                st.markdown(score_title)
                cols = st.columns(min(4, len(score_data)))
                for i, (concept, df) in enumerate(score_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(label=concept.upper(), value=safe_format_number(patient_df[value_col].iloc[-1], 0))
        
        elif view_mode == table_mode:
            table_title = "### 📋 Patient Data Table" if lang == 'en' else "### 📋 患者数据表格"
            st.markdown(table_title)
            for concept, df in st.session_state.loaded_concepts.items():
                if id_col in df.columns:
                    patient_df = df[df[id_col] == patient_id]
                else:
                    patient_df = df
                
                if len(patient_df) > 0:
                    records_label = "records" if lang == 'en' else "条记录"
                    with st.expander(f"{concept} ({len(patient_df)} {records_label})", expanded=False):
                        st.dataframe(patient_df, width="stretch")


def render_data_table_subtab():
    """渲染数据大表子模块 - 让用户按模块查看已加载的数据。"""
    lang = st.session_state.get('language', 'en')
    
    page_title = "📋 Data Tables Explorer" if lang == 'en' else "📋 数据大表浏览"
    st.markdown(f'<div class="compact-section-title">{page_title}</div>', unsafe_allow_html=True)
    
    page_desc = "Browse and explore your loaded data by module. Select a module to view the complete data table." if lang == 'en' else "按模块浏览和探索已加载的数据。选择一个模块查看完整数据表。"
    st.markdown(f'<div class="compact-section-desc">{page_desc}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        no_data_msg = "Please load data first in the settings above." if lang == 'en' else "请先在上方设置中加载数据。"
        st.warning(no_data_msg)
        return
    
    # 按模块分组已加载的概念
    concept_groups = get_concept_groups()
    
    # 🔧 FIX (2026-02-12): 使用内部分组定义来构建映射
    # 由于列名已在 load_from_exported() 中规范化，这里直接使用列名查找分组
    concept_to_group = {}
    for group_key, concepts in CONCEPT_GROUPS_INTERNAL.items():
        # 获取显示名称
        display_name = CONCEPT_GROUP_NAMES.get(group_key, (group_key, group_key))
        group_display = display_name[0] if lang == 'en' else display_name[1]
        
        for c in concepts:
            if c not in concept_to_group:
                concept_to_group[c] = group_display
    
    # 🔧 FIX (2026-02-12): 列名已在 load_from_exported() 中规范化并去重
    # 每个列就是一个唯一的 concept，直接分组即可
    loaded_by_module = {}
    
    for column_name in st.session_state.loaded_concepts.keys():
        # 使用列名查找分组（列名已经是规范化后的）
        group = concept_to_group.get(column_name)
        if group:
            if group not in loaded_by_module:
                loaded_by_module[group] = []
            loaded_by_module[group].append(column_name)
    
    # 🔧 FIX (2026-02-12): Features = Concepts = 列数（已去重）
    unique_feature_count = len(st.session_state.loaded_concepts)
    
    # 显示模块统计
    total_rows = sum(
        len(df) for df in st.session_state.loaded_concepts.values() 
        if isinstance(df, pd.DataFrame)
    )
    stats_items = [
        ("Modules" if lang == 'en' else "模块数", len(loaded_by_module)),
        ("Features" if lang == 'en' else "特征数", unique_feature_count),
        ("Patients" if lang == 'en' else "患者数", len(st.session_state.patient_ids) if st.session_state.patient_ids else 0),
        ("Total Rows" if lang == 'en' else "总行数", f"{total_rows:,}"),
    ]
    stats_cols = st.columns(4)
    for idx, (label, value) in enumerate(stats_items):
        with stats_cols[idx]:
            st.markdown(
                f'<div class="mini-stat-card"><div class="mini-label">{label}</div><div class="mini-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    
    # 模块选择器 - 🔧 放大标题
    module_select_label = "Select Module to View" if lang == 'en' else "选择要查看的模块"
    st.markdown(f'<div style="font-size:1.02rem;font-weight:800;color:#111827;margin-bottom:0.4rem">📦 {module_select_label}</div>', unsafe_allow_html=True)
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
        module_concepts = loaded_by_module[selected_module]
        
        # 显示该模块包含的特征
        features_in_module = f"**Features in this module ({len(module_concepts)}):** " + ", ".join(sorted(module_concepts))
        st.caption(features_in_module)
        
        # 特征选择器（单选或多选合并）- 默认合并全部放第一个
        # 🔧 放大标题
        view_mode_label = "View Mode" if lang == 'en' else "查看模式"
        st.markdown(f'<div style="font-size:1.02rem;font-weight:800;color:#111827;margin-bottom:0.4rem">👁️ {view_mode_label}</div>', unsafe_allow_html=True)
        view_modes = ["Merge All (Wide Table)", "Single Feature"] if lang == 'en' else ["合并全部（宽表）", "单个特征"]
        
        view_mode = st.radio("View Mode", view_modes, horizontal=True, key="data_table_view_mode", index=0, label_visibility="collapsed")
        
        if view_mode == view_modes[1]:
            # 单个特征模式 (现在是第二个选项)
            feature_select_label = "Select Feature" if lang == 'en' else "选择特征"
            selected_feature = st.selectbox(
                feature_select_label,
                options=sorted(module_concepts),
                key="data_table_feature_select"
            )
            
            if selected_feature and selected_feature in st.session_state.loaded_concepts:
                df = st.session_state.loaded_concepts[selected_feature]
                
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    # 显示数据统计
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rows_label = "Rows" if lang == 'en' else "行数"
                        st.metric(rows_label, f"{len(df):,}")
                    with col2:
                        cols_label = "Columns" if lang == 'en' else "列数"
                        st.metric(cols_label, len(df.columns))
                    with col3:
                        size_kb = df.memory_usage(deep=True).sum() / 1024
                        size_label = "Memory" if lang == 'en' else "内存占用"
                        st.metric(size_label, f"{size_kb:.1f} KB")
                    
                    # 显示列信息
                    cols_info_label = "Columns" if lang == 'en' else "列信息"
                    with st.expander(f"📊 {cols_info_label}: {', '.join(df.columns.tolist())}", expanded=False):
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Type': [str(df[c].dtype) for c in df.columns],
                            'Non-Null': [df[c].notna().sum() for c in df.columns],
                            'Null %': [f"{df[c].isna().mean()*100:.1f}%" for c in df.columns]
                        })
                        st.dataframe(col_info, hide_index=True, use_container_width=True)
                    
                    # 显示数据表
                    st.markdown("---")
                    table_title = f"📋 {selected_feature} Data Table" if lang == 'en' else f"📋 {selected_feature} 数据表"
                    st.markdown(f"### {table_title}")
                    
                    # 添加搜索/过滤选项
                    filter_expander_label = "🔍 Filter Options" if lang == 'en' else "🔍 过滤选项"
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
                        
                        # 行数限制
                        max_rows_label = "Max rows to display" if lang == 'en' else "最大显示行数"
                        max_rows = st.slider(max_rows_label, 100, 10000, 1000, step=100, key=f"max_rows_{selected_feature}")
                    
                    # 显示数据（限制行数以防卡顿）
                    display_df = df.head(max_rows) if len(df) > max_rows else df
                    # 🔧 FIX: 将布尔列转换为字符串"True"/"False"显示，而非复选框图标
                    display_df = display_df.copy()
                    converted_cols = []
                    for col in display_df.columns:
                        dtype_str = str(display_df[col].dtype).lower()
                        if 'bool' in dtype_str:
                            display_df[col] = display_df[col].astype(str)
                            converted_cols.append(col)
                    # 调试：显示转换信息
                    if converted_cols:
                        st.caption(f"🔧 DEBUG: 已将布尔列 {converted_cols} 转换为字符串显示")
                    st.dataframe(display_df, use_container_width=True, height=500)
                    
                    if len(df) > max_rows:
                        truncate_msg = f"⚠️ Showing first {max_rows:,} of {len(df):,} rows. Adjust 'Max rows' in Filter Options to see more." if lang == 'en' else f"⚠️ 显示前 {max_rows:,} 行（共 {len(df):,} 行）。在过滤选项中调整最大行数可查看更多。"
                        st.caption(truncate_msg)
                    # 不提供下载按钮，因为数据是用户导入的
                else:
                    empty_msg = f"No data available for {selected_feature}" if lang == 'en' else f"{selected_feature} 没有可用数据"
                    st.info(empty_msg)
        
        else:
            # 合并全部模式（宽表）
            merge_control_cols = st.columns([2.8, 1.1])
            with merge_control_cols[0]:
                merge_info = "Merging all features in this module into a wide table (joined by patient ID and time)" if lang == 'en' else "将该模块的所有特征合并为宽表（按患者ID和时间连接）"
                sample_hint = "Large datasets will be sampled for performance" if lang == 'en' else "大数据集将被采样以保证性能"
                st.markdown(f'<div class="compact-inline-notice info">ℹ️ {merge_info}<br><span style="font-size:0.74rem;opacity:0.82">{sample_hint}</span></div>', unsafe_allow_html=True)
            with merge_control_cols[1]:
                max_rows_per_feature = st.selectbox(
                    "Max rows" if lang == 'en' else "最大行数",
                    options=[1000, 2000, 5000, 10000],
                    index=1,
                    key="merge_max_rows"
                )
            
            # 收集该模块的所有数据
            dfs_to_merge = []
            id_col = st.session_state.get('id_col', 'stay_id')
            time_col_candidates = [
                'charttime', 'time', 'starttime', 'start', 'endtime', 'itemtime',
                'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat',
                'givenat', 'enteredentryat', 'intakeoutputoffset', 'observationoffset',
                'nursingchartoffset', 'labresultoffset', 'respchartoffset'
            ]
            metadata_cols = {'valueuom', 'unit', 'units', 'category', 'type', 'dur_var', 'entertime', 'intakeoutputentryoffset'}
            unified_time_col = 'charttime'
            
            for concept_name in module_concepts:
                if concept_name in st.session_state.loaded_concepts:
                    df = st.session_state.loaded_concepts[concept_name]
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        df_copy = df.copy()

                        # 统一时间列名，避免不同数据库/概念的时间别名导致只按 ID 合并
                        if unified_time_col in df_copy.columns:
                            other_time_cols = [tc for tc in time_col_candidates if tc in df_copy.columns and tc != unified_time_col]
                            if other_time_cols:
                                df_copy = df_copy.drop(columns=other_time_cols)
                        else:
                            for tc in time_col_candidates:
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
                        
                        # 🔧 显示截断提示
                        if total_rows_before > MAX_ROWS_PER_DF * len(dfs_to_merge):
                            truncate_warn = f"⚠️ Data was sampled (max {MAX_ROWS_PER_DF:,} rows per feature) for performance. Total rows: {total_rows_before:,}" if lang == 'en' else f"⚠️ 数据已采样（每特征最多 {MAX_ROWS_PER_DF:,} 行）以保证性能。原始总行数：{total_rows_before:,}"
                            st.info(truncate_warn)
                        
                        # 显示合并结果统计（紧凑版）
                        result_stats = [
                            ("Rows" if lang == 'en' else "行数", f"{len(merged_df):,}"),
                            ("Columns" if lang == 'en' else "列数", len(merged_df.columns)),
                            ("Features" if lang == 'en' else "特征数", len(module_concepts)),
                        ]
                        result_cols = st.columns(3)
                        for idx, (label, value) in enumerate(result_stats):
                            with result_cols[idx]:
                                st.markdown(
                                    f'<div class="tiny-stat-card"><div class="tiny-label">{label}</div><div class="tiny-value">{value}</div></div>',
                                    unsafe_allow_html=True,
                                )
                        
                        # 显示数据
                        max_rows = 1000
                        display_df = merged_df.head(max_rows).copy() if len(merged_df) > max_rows else merged_df.copy()
                        # 🔧 FIX: 将布尔列转换为字符串"True"/"False"显示
                        for col in display_df.columns:
                            dtype_str = str(display_df[col].dtype).lower()
                            if 'bool' in dtype_str:
                                display_df[col] = display_df[col].astype(str)
                        st.dataframe(display_df, use_container_width=True, height=500)
                        
                        if len(merged_df) > max_rows:
                            truncate_msg = f"⚠️ Showing first {max_rows:,} of {len(merged_df):,} rows." if lang == 'en' else f"⚠️ 显示前 {max_rows:,} 行（共 {len(merged_df):,} 行）。"
                            st.caption(truncate_msg)
                    # 不提供下载按钮，因为数据是用户导入的
                    except Exception as e:
                        err_msg = f"Error merging data: {e}" if lang == 'en' else f"合并数据时出错: {e}"
                        st.error(err_msg)


def _render_ai_context_button(question_key: str, context: str = "", concept: str = ""):
    """渲染上下文感知的 AI 助手小按钮。点击后将问题发送到全局 AI 助手。"""
    label = get_text(question_key)
    btn_key = f"ai_ctx_{question_key}_{concept}_{hash(context) % 10000}"
    if st.button(f"🤖 {label}", key=btn_key, help=label):
        full_q = f"{label}"
        if concept:
            full_q += f" (concept: {concept})"
        if context:
            full_q += f" Context: {context}"
        st.session_state['_ai_pending_question'] = full_q
        st.session_state['_floating_ai_open'] = True
        st.toast("💬 Question sent to AI Assistant" if st.session_state.get('language') == 'en' else "💬 问题已发送到 AI 助手")


def render_quality_page():
    """渲染数据质量页面。"""
    lang = st.session_state.get('language', 'en')
    
    page_title = "Data Quality" if lang == 'en' else "数据质量评估"
    page_sub = "Missing rate analysis, coverage badges & explainable causes" if lang == 'en' else "缺失率分析、覆盖度标识与可解释原因"
    st.markdown(f'''
    <div style="margin-bottom:20px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{page_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{page_sub}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        _no_data_msg = "Load data to begin quality analysis." if lang == 'en' else "请先加载数据以进行质量分析。"
        _tip_msg = 'Try "Demo Mode" for a quick start.' if lang == 'en' else '选择「演示模式」快速开始。'
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">📊</div>
            <div style="font-weight:600;color:#111827;margin-bottom:4px">{_no_data_msg}</div>
            <div style="font-size:.85rem;color:#9ca3af">{_tip_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # 总体数据质量概览
    
    total_records = 0
    total_missing = 0
    quality_data = []
    
    # 🔧 改进的缺失率计算（2026-01-29 v3 重新设计）
    # 核心原则：
    # 1. 人口统计学静态概念：每患者一条记录，缺失率 = NA值比例（这些确实只需要1条）
    # 2. 所有其他概念（包括事件型）：缺失率 = 1 - (实际记录数/患者 / 72)
    #    72是完整的时间网格（72小时=72个时间点）
    #    例如：abx有1条 → 缺失率 = (72-1)/72 = 98.6%
    
    # 只有人口统计学数据才是真正的"静态"（每患者只需要1条记录）
    demographic_static = [
        'death', 'los_icu', 'los_hosp', 'age', 'weight', 'height', 'sex', 'bmi'
    ]

    # 事件型时间序列：只统计事件发生的时间点（避免全量0导致0%缺失）
    # 🔧 包含所有布尔事件型概念：sepsis相关、感染相关、RRT、循环衰竭等
    event_time_series = [
        # 循环衰竭
        'circ_failure', 'circ_event',
        # Sepsis-3 诊断
        'sep3_sofa2', 'sep3_sofa1', 'sepsis_sofa2',
        # 感染相关
        'susp_inf', 'infection_icd', 'samp',
        # 肾替代治疗
        'rrt', 'rrt_criteria',
        # AKI标志
        'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
        # 机械通气
        'mech_vent', 'vent_ind', 'vent_start', 'vent_end',
        # ECMO 和机械循环支持
        'ecmo', 'ecmo_indication', 'mech_circ_support',
        # 药物事件
        'abx', 'cort',
        # 血管活性药物指示
        'vaso_ind',
    ]
    
    # 🔧 FIX (2026-02-04): 静态布尔事件（每患者最多1条，只有发生时才记录）
    # 缺失率 = 1 - (有记录的患者数 / 总患者数)
    # 🔧 mech_circ_support 是非常罕见的治疗（约2-3%患者），缺失率应该约97-98%
    static_boolean_events = [
        'ecmo', 'ecmo_indication', 'mech_circ_support',  # ECMO/机械循环支持（罕见，约2-3%）
        'cort',  # 皮质类固醇（约25-30%）
        'abx',   # 抗生素（静态版本，约70%）
        'vaso_ind',  # 血管活性药物指示（约50-60%）
    ]
    
    # 🔧 完整时间网格大小：优先使用模拟数据的时长参数，否则默认72小时
    mock_params = st.session_state.get('mock_params', {})
    time_grid_size = mock_params.get('hours', 72) if mock_params else 72

    def _detect_time_col(df: pd.DataFrame) -> Optional[str]:
        # 🔧 添加 'time' 作为首选候选（模拟数据使用 'time' 列表示小时数）
        time_candidates = [
            'time',  # 🔧 模拟数据使用的时间列（小时数）
            'charttime', 'datetime', 'measuredat', 'measuredat_minutes',
            'observationoffset', 'Offset', 'starttime', 'endtime', 'givenat', 'timestamp',
        ]
        for col in time_candidates:
            if col in df.columns:
                return col
        return None

    def _to_hour_bins(series: pd.Series, col_name: str) -> Optional[pd.Series]:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.dt.floor('H')
        if pd.api.types.is_object_dtype(series):
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().any():
                return parsed.dt.floor('H')
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.notna().any():
                col_lower = col_name.lower()
                if 'second' in col_lower:
                    return (numeric / 3600).floordiv(1)
                if 'minute' in col_lower or 'offset' in col_lower:
                    return (numeric / 60).floordiv(1)
                return numeric.floordiv(1)
            return None
        if pd.api.types.is_numeric_dtype(series):
            col_lower = col_name.lower()
            if 'second' in col_lower:
                return (series / 3600).floordiv(1)
            if 'minute' in col_lower or 'offset' in col_lower:
                return (series / 60).floordiv(1)
            return series.floordiv(1)
        return None

    def _calc_time_missing(
        df: pd.DataFrame,
        id_col: str,
        time_col: Optional[str],
        time_grid_size: int,
        event_mask: Optional[pd.Series] = None,
    ) -> Optional[float]:
        if time_col is None or id_col not in df.columns:
            return None
        data = df.loc[event_mask] if event_mask is not None else df
        if data.empty:
            return 100.0
        time_series = data[time_col]
        hour_bins = _to_hour_bins(time_series, time_col)
        if hour_bins is None:
            return None
        tmp = data[[id_col]].copy()
        tmp['_hour_bin'] = hour_bins
        tmp = tmp.dropna(subset=['_hour_bin'])
        if tmp.empty:
            return None

        # 🔧 简化计算：直接用 (1 - 实际覆盖率) 作为缺失率
        # 每个患者的唯一小时数
        unique_hours_per_patient = tmp.groupby(id_col)['_hour_bin'].nunique()
        
        # 总患者数
        n_patients_in_data = len(unique_hours_per_patient)
        if n_patients_in_data == 0:
            return 100.0
        
        # 平均每患者的唯一小时数
        avg_unique_hours = unique_hours_per_patient.mean()
        
        # 缺失率 = 1 - (平均唯一小时数 / 时间网格大小)
        coverage = avg_unique_hours / time_grid_size
        missing_rate = max(0.0, 1.0 - coverage) * 100
        
        return float(missing_rate)
    
    # 获取总患者数（用于计算患者覆盖率）
    # 🔧 FIX (2026-02-04): 改进总患者数获取逻辑
    # 对于静态布尔事件，需要从非静态布尔事件的概念中获取总患者数
    # 否则会导致 n_patients == total_patients，缺失率错误地显示为 0%
    
    # 首先尝试从 mock_params 获取（Demo Mode 最准确）
    mock_params = st.session_state.get('mock_params', {})
    total_patients_in_session = mock_params.get('n_patients', 0)
    
    # 如果 mock_params 没有，尝试从 patient_limit 获取
    if total_patients_in_session == 0:
        total_patients_in_session = st.session_state.get('patient_limit', 0)
    
    # 如果仍然为 0，从数据中获取最大的患者数
    if total_patients_in_session == 0:
        # 尝试从非静态布尔事件的概念中获取最大患者数
        max_patients_found = 0
        for concept, df in st.session_state.loaded_concepts.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0 and st.session_state.id_col in df.columns:
                concept_patients = df[st.session_state.id_col].nunique()
                # 优先使用非静态布尔事件的概念患者数
                if concept not in static_boolean_events:
                    max_patients_found = max(max_patients_found, concept_patients)
        
        if max_patients_found > 0:
            total_patients_in_session = max_patients_found
        else:
            # 如果所有概念都是静态布尔事件，默认使用 50
            total_patients_in_session = 50
    
    for concept, df in st.session_state.loaded_concepts.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            numeric_cols = df.select_dtypes(include=['number']).columns
            # 排除ID列和所有可能的时间列，只保留真正的数值列
            exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'time', 'index',
                           'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                           'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
            value_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            n_records = len(df)
            n_patients = df[st.session_state.id_col].nunique() if st.session_state.id_col in df.columns else 0
            
            # 只有人口统计学数据才是真正的静态概念
            is_demographic = concept in demographic_static
            main_col = concept if concept in df.columns else (value_cols[0] if value_cols else None)
            time_col = _detect_time_col(df)
            
            # 🔧 FIX (2026-02-03): 判断是否为静态布尔事件
            is_static_boolean = concept in static_boolean_events
            
            # 计算缺失率
            if is_demographic:
                # 人口统计学静态概念：缺失率 = NA值比例（这些确实只需要1条/患者）
                if value_cols:
                    main_col = concept if concept in df.columns else (value_cols[0] if value_cols else None)
                    if main_col and main_col in df.columns:
                        missing_rate = df[main_col].isna().mean() * 100
                    else:
                        missing_rate = df[value_cols].isna().mean().mean() * 100
                else:
                    missing_rate = 0
            elif is_static_boolean:
                # 🔧 FIX (2026-02-03): 静态布尔事件：只有发生时才记录
                # 缺失率 = 1 - (有记录的患者数 / 总患者数)
                # 例如：5%患者使用ECMO → 缺失率 = 95%
                patients_with_event = n_patients  # 有记录的患者数
                # 总患者数从session获取
                total_patients = total_patients_in_session
                if total_patients > 0:
                    missing_rate = (1 - patients_with_event / total_patients) * 100
                else:
                    missing_rate = 0
            else:
                # 🔧 FIX (2026-02-03): 修复从宽表导入时的缺失率计算
                # 核心问题：宽表可能有完整的时间网格（72行/患者），但值列有大量NaN
                # 解决方案：优先检查值列的NaN比例
                
                if n_patients > 0:
                    # 🔧 先检查值列的NaN比例（对于从宽表导入的数据更准确）
                    na_rate_in_column = None
                    if main_col and main_col in df.columns:
                        na_rate_in_column = df[main_col].isna().mean() * 100
                    
                    # 计算每患者平均记录数
                    records_per_patient = n_records / n_patients
                    
                    # 对于事件型数据，只统计非零记录
                    if concept in event_time_series and main_col and main_col in df.columns:
                        col_data = df[main_col]
                        # 检查数据类型，bool必须在numeric之前（nullable bool也算numeric）
                        if pd.api.types.is_bool_dtype(col_data):
                            event_count = col_data.fillna(False).sum()
                        elif pd.api.types.is_numeric_dtype(col_data):
                            event_count = (col_data.fillna(0) > 0).sum()
                        else:
                            # 字符串或其他类型，统计非空非零记录
                            event_count = col_data.notna().sum()
                        records_per_patient = event_count / n_patients if n_patients > 0 else 0
                    
                    # 🔧 FIX: 如果值列NaN比例较高（>5%），优先使用NaN比例作为缺失率
                    # 这对从宽表导入的数据更准确
                    if na_rate_in_column is not None and na_rate_in_column > 5:
                        missing_rate = na_rate_in_column
                    else:
                        # 缺失率 = 1 - (每患者记录数 / 时间网格大小)
                        # 例如：每患者9条记录，时间网格72 → 缺失率 = 1 - 9/72 = 87.5%
                        coverage = records_per_patient / time_grid_size
                        missing_rate = max(0, min(100, (1 - coverage) * 100))
                else:
                    # 无患者数或计算失败
                    missing_rate = 100
            
            total_records += n_records
            total_missing += n_records * (missing_rate / 100)
            
            records_col = "Records" if lang == 'en' else "记录数"
            patients_col = "Patients" if lang == 'en' else "患者数"
            missing_col = "Missing %" if lang == 'en' else "缺失率"
            coverage_col = "Coverage" if lang == 'en' else "覆盖度"
            cause_col = "Likely Cause" if lang == 'en' else "可能原因"
            
            _cause_text, _cause_color = _get_missing_cause_tag(concept, missing_rate / 100.0)
            _n_db = CONCEPT_DB_COVERAGE.get(concept, 0)
            _badge = f"{_n_db}/6 DBs"
            
            quality_data.append({
                'Concept': concept,
                records_col: f"{n_records:,}",
                patients_col: n_patients,
                missing_col: f"{missing_rate:.1f}%",
                coverage_col: _badge,
                cause_col: _cause_text,
            })
    
    # 总体统计
    overall_missing = (total_missing / total_records * 100) if total_records > 0 else 0
    
    records_label = "Total Records" if lang == 'en' else "总记录数"
    missing_label = "Avg Missing" if lang == 'en' else "平均缺失率"
    items_label = "Data Items" if lang == 'en' else "数据项数"
    
    # 缺失率颜色映射
    _miss_color = "#10b981" if overall_missing < 20 else ("#f59e0b" if overall_missing < 50 else "#ef4444")
    
    st.markdown(f'''
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px">
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9ca3af;margin-bottom:6px">{records_label}</div>
            <div style="font-size:1.4rem;font-weight:800;color:#111827">{total_records:,}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9ca3af;margin-bottom:6px">{missing_label}</div>
            <div style="font-size:1.4rem;font-weight:800;color:{_miss_color}">{overall_missing:.1f}%</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9ca3af;margin-bottom:6px">{items_label}</div>
            <div style="font-size:1.4rem;font-weight:800;color:#111827">{len(quality_data)}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 详细数据表
    detail_title = "Detailed Report" if lang == 'en' else "详细报告"
    st.markdown(f'''
    <div style="font-size:1.05rem;font-weight:700;color:#111827;margin-bottom:10px">{detail_title}</div>
    ''', unsafe_allow_html=True)
    
    if quality_data:
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(
            quality_df, 
            width="stretch", 
            hide_index=True,
        )
        _render_ai_context_button(
            'ai_why_missing',
            context=f"database={st.session_state.get('database', '')}; loaded_concepts={len(st.session_state.get('loaded_concepts', {}))}; explain the main missingness patterns and likely reasons based on the current dataset summary",
        )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 可视化分析
    tab1_label = "📊 Missing Rate Chart" if lang == 'en' else "📊 缺失率图表"
    tab2_label = "📈 Value Distribution" if lang == 'en' else "📈 数值分布"
    tab1, tab2 = st.tabs([tab1_label, tab2_label])
    
    with tab1:
        # 排序方式选择
        sort_label = "Sort by" if lang == 'en' else "排序方式"
        sort_options = {
            'asc': '📈 Missing Rate (Low → High)' if lang == 'en' else '📈 缺失率 (从低到高)',
            'desc': '📉 Missing Rate (High → Low)' if lang == 'en' else '📉 缺失率 (从高到低)',
            'alpha': '🔤 Alphabetical (A → Z)' if lang == 'en' else '🔤 首字母排序 (A → Z)',
        }
        sort_order = st.radio(
            sort_label,
            options=list(sort_options.keys()),
            format_func=lambda x: sort_options[x],
            horizontal=True,
            key='missing_chart_sort_order',
        )
        
        # 缺失率条形图
        try:
            import plotly.express as px
            
            missing_data = []
            # 只有人口统计学数据才是真正的"静态"
            demographic_static = [
                'death', 'los_icu', 'los_hosp', 'age', 'weight', 'height', 'sex', 'bmi'
            ]

            # 事件型时间序列：只统计事件发生的时间点（避免全量0导致0%缺失）
            # 🔧 包含所有布尔事件型概念：sepsis相关、感染相关、RRT、循环衰竭等
            event_time_series = [
                # 循环衰竭
                'circ_failure', 'circ_event',
                # Sepsis-3 诊断
                'sep3_sofa2', 'sep3_sofa1', 'sepsis_sofa2',
                # 感染相关
                'susp_inf', 'infection_icd', 'samp',
                # 肾替代治疗
                'rrt', 'rrt_criteria',
                # AKI标志
                'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
                # 机械通气
                'mech_vent', 'vent_ind', 'vent_start', 'vent_end',
                # ECMO
                'ecmo', 'ecmo_indication',
                # 药物事件
                'abx', 'cort',
                # 血管活性药物指示
                'vaso_ind',
            ]
            
            # 🔧 FIX (2026-02-04): 静态布尔事件（每患者最多1条，只有发生时才记录）
            # 缺失率 = 1 - (有记录的患者数 / 总患者数)
            # mech_circ_support 是非常罕见的治疗（约2-3%患者），缺失率应该约97-98%
            static_boolean_events_chart = [
                'ecmo', 'ecmo_indication', 'mech_circ_support',  # ECMO/机械循环支持（罕见，约2-3%）
                'cort',  # 皮质类固醇（约25-30%）
                'abx',   # 抗生素（静态版本，约70%）
                'vaso_ind',  # 血管活性药物指示（约50-60%）
            ]
            
            # 🔧 完整时间网格大小：优先使用模拟数据的时长参数，否则默认72小时
            mock_params = st.session_state.get('mock_params', {})
            time_grid_size = mock_params.get('hours', 72) if mock_params else 72

            def _detect_time_col(df: pd.DataFrame) -> Optional[str]:
                # 🔧 添加 'time' 作为首选候选（模拟数据使用 'time' 列表示小时数）
                time_candidates = [
                    'time',  # 🔧 模拟数据使用的时间列（小时数）
                    'charttime', 'datetime', 'measuredat', 'measuredat_minutes',
                    'observationoffset', 'Offset', 'starttime', 'endtime', 'givenat', 'timestamp',
                ]
                for col in time_candidates:
                    if col in df.columns:
                        return col
                return None

            def _to_hour_bins(series: pd.Series, col_name: str) -> Optional[pd.Series]:
                if pd.api.types.is_datetime64_any_dtype(series):
                    return series.dt.floor('H')
                if pd.api.types.is_object_dtype(series):
                    parsed = pd.to_datetime(series, errors='coerce')
                    if parsed.notna().any():
                        return parsed.dt.floor('H')
                    numeric = pd.to_numeric(series, errors='coerce')
                    if numeric.notna().any():
                        col_lower = col_name.lower()
                        if 'second' in col_lower:
                            return (numeric / 3600).floordiv(1)
                        if 'minute' in col_lower or 'offset' in col_lower:
                            return (numeric / 60).floordiv(1)
                        return numeric.floordiv(1)
                    return None
                if pd.api.types.is_numeric_dtype(series):
                    col_lower = col_name.lower()
                    if 'second' in col_lower:
                        return (series / 3600).floordiv(1)
                    if 'minute' in col_lower or 'offset' in col_lower:
                        return (series / 60).floordiv(1)
                    return series.floordiv(1)
                return None

            def _calc_time_missing(
                df: pd.DataFrame,
                id_col: str,
                time_col: Optional[str],
                time_grid_size: int,
                event_mask: Optional[pd.Series] = None,
            ) -> Optional[float]:
                if time_col is None or id_col not in df.columns:
                    return None
                data = df.loc[event_mask] if event_mask is not None else df
                if data.empty:
                    return 100.0
                time_series = data[time_col]
                hour_bins = _to_hour_bins(time_series, time_col)
                if hour_bins is None:
                    return None
                tmp = data[[id_col]].copy()
                tmp['_hour_bin'] = hour_bins
                tmp = tmp.dropna(subset=['_hour_bin'])
                if tmp.empty:
                    return None

                # 🔧 优化：使用向量化操作代替 groupby().apply()
                unique_hours_per_patient = tmp.groupby(id_col)['_hour_bin'].nunique()
                
                if pd.api.types.is_numeric_dtype(tmp['_hour_bin']):
                    total_hours = time_grid_size
                    missing_rates = 1.0 - (unique_hours_per_patient / total_hours)
                else:
                    time_ranges = tmp.groupby(id_col)['_hour_bin'].agg(lambda x: (x.max() - x.min()) / pd.Timedelta(hours=1) + 1)
                    time_ranges = time_ranges.clip(upper=time_grid_size)
                    missing_rates = 1.0 - (unique_hours_per_patient / time_ranges.clip(lower=1))
                
                return float(missing_rates.clip(lower=0).mean() * 100)
            
            # 🔧 FIX (2026-02-04): 改进总患者数获取逻辑（图表部分）
            # 首先尝试从 mock_params 获取（Demo Mode 最准确）
            mock_params = st.session_state.get('mock_params', {})
            total_patients_chart = mock_params.get('n_patients', 0)
            
            # 如果 mock_params 没有，尝试从 patient_limit 获取
            if total_patients_chart == 0:
                total_patients_chart = st.session_state.get('patient_limit', 0)
            
            # 如果仍然为 0，从数据中获取最大的患者数
            if total_patients_chart == 0:
                max_patients_found = 0
                for concept, df in st.session_state.loaded_concepts.items():
                    if isinstance(df, pd.DataFrame) and len(df) > 0 and st.session_state.id_col in df.columns:
                        concept_patients = df[st.session_state.id_col].nunique()
                        if concept not in static_boolean_events_chart:
                            max_patients_found = max(max_patients_found, concept_patients)
                
                if max_patients_found > 0:
                    total_patients_chart = max_patients_found
                else:
                    total_patients_chart = 50
            
            for concept, df in st.session_state.loaded_concepts.items():
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'time', 'index',
                                   'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                                   'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                    value_cols = [c for c in numeric_cols if c not in exclude_cols]
                    if value_cols:
                        n_records = len(df)
                        n_patients = df[st.session_state.id_col].nunique() if st.session_state.id_col in df.columns else 0
                        
                        # 只有人口统计学数据才是静态
                        is_demographic = concept in demographic_static
                        # 🔧 FIX (2026-02-04): 静态布尔事件需要特殊处理
                        is_static_boolean_chart = concept in static_boolean_events_chart
                        
                        main_col = concept if concept in df.columns else value_cols[0]

                        # 计算缺失率
                        if is_demographic:
                            # 人口统计学：只看NA比例
                            if main_col in df.columns:
                                final_missing_rate = df[main_col].isna().mean() * 100
                            else:
                                final_missing_rate = df[value_cols].isna().mean().mean() * 100
                        elif is_static_boolean_chart:
                            # 🔧 FIX (2026-02-04): 静态布尔事件：缺失率 = 1 - (有记录的患者数 / 总患者数)
                            # 例如：2.5%患者使用机械循环支持 → 缺失率 = 97.5%
                            patients_with_event = n_patients  # 有记录的患者数
                            total_patients = total_patients_chart
                            if total_patients > 0:
                                final_missing_rate = (1 - patients_with_event / total_patients) * 100
                            else:
                                final_missing_rate = 0
                                final_missing_rate = df[value_cols].isna().mean().mean() * 100
                        else:
                            # 🔧 简化的缺失率计算：1 - (每患者记录数 / 时间网格)
                            # 与详情表保持一致，使用每个概念实际的患者数
                            if n_patients > 0:
                                # 每患者平均记录数
                                records_per_patient = n_records / n_patients
                                
                                # 对于事件型数据，只计算事件发生的记录
                                if concept in event_time_series and main_col in df.columns:
                                    _col = df[main_col]
                                    if pd.api.types.is_bool_dtype(_col):
                                        event_count = _col.fillna(False).sum()
                                    elif pd.api.types.is_numeric_dtype(_col):
                                        event_count = (_col.fillna(0) > 0).sum()
                                    else:
                                        event_count = _col.notna().sum()
                                    records_per_patient = event_count / n_patients
                                
                                coverage = records_per_patient / time_grid_size
                                final_missing_rate = max(0, min(100, (1 - coverage) * 100))
                            else:
                                final_missing_rate = 100

                        missing_rate_label = "Missing Rate (%)" if lang == 'en' else "空值比例 (%)"
                        records_label_2 = "Records" if lang == 'en' else "记录数"
                        
                        missing_data.append({
                            'Concept': concept, 
                            missing_rate_label: final_missing_rate,
                            records_label_2: len(df)
                        })
            
            if missing_data:
                missing_df = pd.DataFrame(missing_data)
                missing_rate_col = "Missing Rate (%)" if lang == 'en' else "空值比例 (%)"
                
                if sort_order == 'desc':
                    missing_df = missing_df.sort_values(missing_rate_col, ascending=False)
                elif sort_order == 'alpha':
                    missing_df = missing_df.sort_values('Concept', ascending=True)
                else:  # 'asc'
                    missing_df = missing_df.sort_values(missing_rate_col, ascending=True)
                
                # 检查是否全是0
                if missing_df[missing_rate_col].sum() == 0:
                    # 所有数据无缺失，显示成功信息
                    good_msg = "✅ Excellent data quality: No missing values in numeric columns" if lang == 'en' else "✅ 数据质量良好：所有数值列均无空值 (NA/NaN)"
                    st.success(good_msg)
                    
                    # 显示概念列表
                    concepts_loaded = f"**Loaded Concepts ({len(missing_df)} total):**" if lang == 'en' else f"**已加载概念 ({len(missing_df)} 个)：**"
                    st.markdown(concepts_loaded)
                    concept_list = ", ".join(missing_df['Concept'].tolist())
                    st.write(concept_list)
                else:
                    # 有缺失值，绘制条形图
                    chart_title = '📉 Missing Rate Analysis by Concept' if lang == 'en' else '📉 各 Concept 空值比例分析'
                    fig = px.bar(
                        missing_df, x=missing_rate_col, y='Concept',
                        orientation='h',
                        title=chart_title,
                        color=missing_rate_col,
                        color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
                        hover_data=[records_label_2 if lang == 'en' else '记录数']
                    )
                    fig.update_layout(
                        template="plotly_white",
                        height=max(300, len(missing_data) * 40),
                        showlegend=False,
                        yaxis_title="",
                        yaxis=dict(autorange='reversed'),
                        margin=dict(l=100, r=30, t=50, b=50),
                        font=dict(size=14, color='black'),
                    )
                    fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            err_msg = f"Chart rendering failed: {e}" if lang == 'en' else f"图表渲染失败: {e}"
            st.warning(err_msg)
    
    with tab2:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            select_concept_label = "Select Concept" if lang == 'en' else "选择 Concept"
            concept = st.selectbox(
                select_concept_label,
                options=list(st.session_state.loaded_concepts.keys()),
                key="quality_concept"
            )
        
        with col2:
            if concept:
                df = st.session_state.loaded_concepts[concept]
                
                if isinstance(df, pd.DataFrame):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    non_id_cols = [c for c in numeric_cols if c not in ['stay_id', 'hadm_id', 'time', 'index']]
                    
                    if non_id_cols:
                        try:
                            import plotly.express as px
                            import plotly.graph_objects as go
                            
                            value_col = non_id_cols[0]
                            
                            dist_title = f"📊 {concept.upper()} Value Distribution" if lang == 'en' else f"📊 {concept.upper()} 数值分布"
                            fig = px.histogram(
                                df, x=value_col, nbins=50,
                                title=dist_title,
                                marginal="box"
                            )
                            fig.update_layout(
                                template="plotly_white",
                                height=400,
                                showlegend=False,
                                font=dict(size=14, color='black'),
                            )
                            fig.update_traces(marker_color='#1f77b4')
                            fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 统计摘要
                            summary_label = "**Statistical Summary:**" if lang == 'en' else "**统计摘要:**"
                            st.markdown(summary_label)
                            col_a, col_b, col_c, col_d, col_e = st.columns(5)
                            col_a.metric("Min", f"{df[value_col].min():.2f}")
                            col_b.metric("Max", f"{df[value_col].max():.2f}")
                            col_c.metric("Mean", f"{df[value_col].mean():.2f}")
                            col_d.metric("Median", f"{df[value_col].median():.2f}")
                            col_e.metric("Std", f"{df[value_col].std():.2f}")
                            
                        except Exception as e:
                            err_msg = f"Distribution chart rendering failed: {e}" if lang == 'en' else f"分布图渲染失败: {e}"
                            st.warning(err_msg)


def _scan_export_folders(root: str):
    """扫描导出根目录，返回包含 demographics 文件的子文件夹列表"""
    folders = []
    if not root or not os.path.isdir(root):
        return folders
    try:
        for entry in sorted(os.listdir(root)):
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue
            has_demo = any(f.startswith('demographics') and f.endswith(('.parquet', '.csv'))
                          for f in os.listdir(entry_path))
            if has_demo:
                files = [f for f in os.listdir(entry_path) if f.endswith(('.parquet', '.csv', '.xlsx'))]
                folders.append((entry, len(files), entry_path))
    except OSError:
        pass
    return folders


def _load_demographics_from_export(folder_path: str) -> pd.DataFrame:
    """从导出文件夹加载 demographics 数据"""
    demo_files = [f for f in os.listdir(folder_path)
                  if f.startswith('demographics') and f.endswith(('.parquet', '.csv'))]
    if not demo_files:
        raise FileNotFoundError("No demographics file found")
    
    demo_path = os.path.join(folder_path, demo_files[0])
    if demo_path.endswith('.parquet'):
        df = pd.read_parquet(demo_path)
    else:
        df = pd.read_csv(demo_path)
    
    # 标准化列名: sex → gender
    if 'sex' in df.columns and 'gender' not in df.columns:
        df = df.rename(columns={'sex': 'gender'})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.upper().str[0]
    
    # 检测数据库来源
    folder_name = os.path.basename(folder_path).lower()
    detected_db = 'unknown'
    for db_key in ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic']:
        if db_key in folder_name:
            detected_db = db_key
            break
    df.attrs['detected_db'] = detected_db
    return df


def _generate_mock_demographics(n_patients: int, lang: str = 'en') -> pd.DataFrame:
    """生成模拟的人口统计学数据用于Cohort Comparison演示。
    
    🔧 改进：复用 generate_mock_data 的逻辑，确保数据一致性。
    
    Args:
        n_patients: 患者数量
        lang: 语言
        
    Returns:
        包含人口统计学数据的DataFrame
    """
    # 🔧 使用统一的 generate_mock_data 函数生成基础数据
    # 注意：generate_mock_data 返回 (data_dict, patient_ids) 元组
    mock_data_tuple = generate_mock_data(n_patients=n_patients, hours=72)
    mock_data = mock_data_tuple[0] if isinstance(mock_data_tuple, tuple) else mock_data_tuple
    
    # 提取需要的人口统计学字段
    age_df = mock_data.get('age', pd.DataFrame(columns=['stay_id', 'age']))
    sex_df = mock_data.get('sex', pd.DataFrame(columns=['stay_id', 'sex']))
    death_df = mock_data.get('death', pd.DataFrame(columns=['stay_id', 'death']))
    los_icu_df = mock_data.get('los_icu', pd.DataFrame(columns=['stay_id', 'los_icu']))
    sofa_df = mock_data.get('sofa', pd.DataFrame(columns=['stay_id', 'time', 'sofa']))
    
    # 创建基础 DataFrame
    patient_ids = age_df['stay_id'].tolist() if 'stay_id' in age_df.columns else list(range(1, n_patients + 1))
    
    df = pd.DataFrame({'stay_id': patient_ids})
    
    # 合并年龄
    if not age_df.empty and 'age' in age_df.columns:
        df = df.merge(age_df[['stay_id', 'age']], on='stay_id', how='left')
    else:
        df['age'] = np.clip(np.random.normal(65, 15, len(df)), 18, 95).astype(int)
    
    # 合并性别
    if not sex_df.empty and 'sex' in sex_df.columns:
        df = df.merge(sex_df[['stay_id', 'sex']], on='stay_id', how='left')
        df['gender'] = df['sex']
    else:
        df['gender'] = np.random.choice(['M', 'F'], len(df), p=[0.55, 0.45])
    
    # 合并死亡状态
    if not death_df.empty and 'death' in death_df.columns:
        df = df.merge(death_df[['stay_id', 'death']], on='stay_id', how='left')
        df['survived'] = (1 - df['death']).astype(int)
    else:
        df['survived'] = np.random.choice([0, 1], len(df), p=[0.15, 0.85])
    
    # 合并LOS
    if not los_icu_df.empty and 'los_icu' in los_icu_df.columns:
        df = df.merge(los_icu_df[['stay_id', 'los_icu']], on='stay_id', how='left')
        df['los_days'] = df['los_icu']
        df['los_hours'] = (df['los_icu'] * 24).astype(int)
    else:
        df['los_hours'] = np.clip(np.random.lognormal(4.5, 0.8, len(df)), 24, 1000).astype(int)
        df['los_days'] = df['los_hours'] / 24
    
    # 计算 SOFA max
    if not sofa_df.empty and 'sofa' in sofa_df.columns:
        sofa_max = sofa_df.groupby('stay_id')['sofa'].max().reset_index()
        sofa_max.columns = ['stay_id', 'sofa_max']
        df = df.merge(sofa_max, on='stay_id', how='left')
        df['sofa_max'] = df['sofa_max'].fillna(0).astype(int)
    else:
        df['sofa_max'] = np.random.choice(range(0, 20), len(df))
    
    # 首次ICU入住
    df['first_icu_stay'] = np.random.choice([True, False], len(df), p=[0.65, 0.35])
    
    # 选择需要的列
    result_cols = ['stay_id', 'age', 'gender', 'los_hours', 'los_days', 'first_icu_stay', 'survived', 'sofa_max']
    available_cols = [c for c in result_cols if c in df.columns]
    
    return df[available_cols]


def _path_looks_like_database(path: str) -> bool:
    """检查路径是否看起来像数据库目录（包含 parquet/csv 文件或已知子目录）"""
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    entries_lower = [e.lower() for e in entries]
    # 包含 parquet 文件
    if any(e.endswith('.parquet') for e in entries_lower):
        return True
    # 包含已知子目录（MIMIC hosp/icu, eICU 表名, HiRID 分桶等）
    known_dirs = {'hosp', 'icu', 'observations_bucket', 'pharma_bucket',
                  'observation_tables', 'pharma_records', 'reference_data'}
    if known_dirs & set(entries_lower):
        return True
    # 包含 csv/csv.gz 文件
    if any(e.endswith('.csv') or e.endswith('.csv.gz') for e in entries_lower):
        return True
    # 包含以 _bucket 结尾的子目录（分桶数据）
    if any(e.endswith('_bucket') for e in entries_lower):
        return True
    return False


def find_database_path(root: str, db_name: str) -> str:
    """智能检测数据库路径，支持多种目录命名方式
    
    支持以下场景:
    - root=根目录, db_name=数据库 → root/alias[/version]
    - root=数据库目录本身 → 直接返回 root（当 root 目录名匹配别名或包含数据文件）
    - root=版本目录 → 直接返回 root（当 root 目录包含 parquet/csv 文件）
    
    Args:
        root: ICU数据根目录，或直接的数据库路径
        db_name: 数据库名称（miiv, eicu, aumc, hirid, mimic, sic）
        
    Returns:
        完整的数据库路径
    """
    # 定义每个数据库可能的目录名称和版本号
    db_aliases = {
        'miiv': ['mimiciv', 'mimic-iv', 'miiv', 'mimic_iv', 'mimic-iv-3.1'],
        'eicu': ['eicu', 'eicu-crd', 'eicu_crd'],
        'aumc': ['aumc', 'amsterdamumc', 'amsterdam'],
        'hirid': ['hirid', 'hi-rid'],
        'mimic': ['mimiciii', 'mimic-iii', 'mimic3', 'mimic_iii'],
        'sic': ['sicdb', 'sic', 'sic-db'],
    }
    
    aliases = db_aliases.get(db_name, [db_name])
    
    # ===== 优先检查: root 本身就是数据库目录 =====
    if os.path.isdir(root):
        root_basename = os.path.basename(os.path.normpath(root)).lower()
        # 1) root 目录名精确匹配或包含数据库别名
        matched = root_basename in aliases
        if not matched:
            for alias in aliases:
                if alias in root_basename or root_basename.startswith(alias):
                    matched = True
                    break
        if matched:
            # 可能还有版本子目录
            try:
                subdirs = [d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d))
                           and d[0].isdigit()]
            except OSError:
                subdirs = []
            if subdirs:
                subdirs.sort(reverse=True)
                return os.path.join(root, subdirs[0])
            return root
        # 2) root 目录包含数据文件（用户直接指向了版本目录或扁平数据库目录）
        if _path_looks_like_database(root):
            return root
    
    # ===== 常规搜索: root 是父目录 =====
    for alias in aliases:
        # 尝试直接目录
        direct_path = os.path.join(root, alias)
        if os.path.isdir(direct_path):
            # 检查是否有版本子目录
            try:
                subdirs = [d for d in os.listdir(direct_path)
                           if os.path.isdir(os.path.join(direct_path, d))
                           and d[0].isdigit()]
            except OSError:
                subdirs = []
            if subdirs:
                subdirs.sort(reverse=True)
                return os.path.join(direct_path, subdirs[0])
            else:
                return direct_path
        
        # 尝试带版本的固定路径
        default_versions = {
            'mimiciv': '3.1', 'mimic-iv': '3.1', 'miiv': '3.1',
            'eicu': '2.0.1', 'eicu-crd': '2.0.1',
            'aumc': '1.0.2',
            'hirid': '1.1.1',
            'mimiciii': '1.4', 'mimic-iii': '1.4',
            'sicdb': '1.0.6', 'sic': '1.0.6',
        }
        if alias in default_versions:
            versioned_path = os.path.join(root, alias, default_versions[alias])
            if os.path.isdir(versioned_path):
                return versioned_path
    
    # ===== 模糊匹配: 扫描 root 下目录名是否部分匹配 =====
    if os.path.isdir(root):
        try:
            for entry in os.listdir(root):
                entry_path = os.path.join(root, entry)
                if not os.path.isdir(entry_path):
                    continue
                entry_lower = entry.lower()
                # 检查目录名是否包含任何别名（如 "my_sic_data" 包含 "sic"）
                for alias in aliases:
                    if alias in entry_lower:
                        # 再检查版本子目录
                        try:
                            subdirs = [d for d in os.listdir(entry_path)
                                       if os.path.isdir(os.path.join(entry_path, d))
                                       and d[0].isdigit()]
                        except OSError:
                            subdirs = []
                        if subdirs:
                            subdirs.sort(reverse=True)
                            return os.path.join(entry_path, subdirs[0])
                        return entry_path
        except OSError:
            pass
    
    # 回退：返回 root 本身（而非拼接不存在的路径）
    return root


def render_directory_structure_guide(lang: str = 'en'):
    """渲染目录结构指南弹窗"""
    with st.popover("📂 " + ("Directory Structure Guide" if lang == 'en' else "目录结构指南")):
        struct_info = """
**Expected directory structure:**

```
icudb/                    ← Your ICU Data Root
├── mimiciv/              ← or mimic-iv/, miiv/
│   └── 3.1/              ← version folder (optional)
├── eicu/
│   └── 2.0.1/
├── aumc/
│   └── 1.0.2/
├── hirid/
│   └── 1.1.1/
├── mimiciii/             ← or mimic-iii/, mimic/
│   └── 1.4/
└── sicdb/                ← or sic/
    └── 1.0.6/
```

**Tips:**
- Version folders (3.1, 2.0.1, etc.) are optional
- Database folder names can vary (mimiciv, mimic-iv, miiv)
- System will auto-detect the correct path
""" if lang == 'en' else """
**期望的目录结构：**

```
icudb/                    ← 你的ICU数据根目录
├── mimiciv/              ← 或 mimic-iv/, miiv/
│   └── 3.1/              ← 版本文件夹（可选）
├── eicu/
│   └── 2.0.1/
├── aumc/
│   └── 1.0.2/
├── hirid/
│   └── 1.1.1/
├── mimiciii/             ← 或 mimic-iii/, mimic/
│   └── 1.4/
└── sicdb/                ← 或 sic/
    └── 1.0.6/
```

**提示：**
- 版本文件夹 (3.1, 2.0.1 等) 是可选的
- 数据库文件夹名称可以变化 (mimiciv, mimic-iv, miiv)
- 系统会自动检测正确的路径
"""
        st.markdown(struct_info)


def _generate_mock_multidb_data(lang: str = 'en') -> Dict[str, pd.DataFrame]:
    """生成模拟的多数据库特征分布数据用于演示。
    
    Args:
        lang: 语言
        
    Returns:
        字典，键为数据库名，值为特征数据DataFrame（长格式，含concept和value列）
    """
    np.random.seed(42)
    
    # 🔧 扩展特征列表，涵盖更多临床指标
    # 🔧 FIX: 模拟6个数据库（添加 MIMIC-III 和 SICdb）
    databases = {
        'miiv': {
            # Vital Signs
            'hr': (80, 15), 'sbp': (120, 20), 'dbp': (70, 12), 'map': (85, 15),
            'temp': (37.2, 0.5), 'resp': (18, 4), 'spo2': (96, 3),
            # Laboratory
            'glu': (140, 50), 'na': (140, 5), 'k': (4.2, 0.6), 'crea': (1.2, 0.8),
            'bili': (1.5, 1.2), 'lact': (2.2, 1.5),
            # Hematology
            'hgb': (11, 2), 'plt': (200, 80), 'wbc': (12, 5),
            # Blood Gas
            'ph': (7.38, 0.08), 'po2': (90, 20), 'pco2': (40, 8), 'fio2': (45, 20),
            # SOFA-2
            'sofa2': (5.2, 3.8), 'sofa2_resp': (1.2, 1.1), 'sofa2_coag': (0.8, 0.9),
            'sofa2_liver': (0.6, 0.8), 'sofa2_cardio': (1.0, 1.2), 'sofa2_cns': (0.8, 1.0), 'sofa2_renal': (0.8, 1.0),
        },
        'eicu': {
            'hr': (85, 18), 'sbp': (125, 25), 'dbp': (72, 14), 'map': (88, 18),
            'temp': (37.0, 0.6), 'resp': (20, 5), 'spo2': (95, 4),
            'glu': (150, 60), 'na': (139, 6), 'k': (4.0, 0.7), 'crea': (1.4, 1.0),
            'bili': (1.8, 1.5), 'lact': (2.5, 1.8),
            'hgb': (10.5, 2.2), 'plt': (180, 90), 'wbc': (13, 6),
            'ph': (7.36, 0.09), 'po2': (85, 22), 'pco2': (42, 10), 'fio2': (50, 25),
            # SOFA-2
            'sofa2': (6.0, 4.2), 'sofa2_resp': (1.4, 1.2), 'sofa2_coag': (0.9, 1.0),
            'sofa2_liver': (0.7, 0.9), 'sofa2_cardio': (1.2, 1.3), 'sofa2_cns': (0.9, 1.1), 'sofa2_renal': (0.9, 1.1),
        },
        'aumc': {
            'hr': (75, 12), 'sbp': (115, 18), 'dbp': (65, 10), 'map': (80, 12),
            'temp': (37.4, 0.4), 'resp': (16, 3), 'spo2': (97, 2),
            'glu': (130, 45), 'na': (141, 4), 'k': (4.3, 0.5), 'crea': (1.0, 0.6),
            'bili': (1.2, 1.0), 'lact': (1.8, 1.2),
            'hgb': (11.5, 1.8), 'plt': (220, 70), 'wbc': (11, 4),
            'ph': (7.40, 0.06), 'po2': (95, 18), 'pco2': (38, 6), 'fio2': (40, 18),
            # SOFA-2
            'sofa2': (4.5, 3.5), 'sofa2_resp': (1.0, 1.0), 'sofa2_coag': (0.7, 0.8),
            'sofa2_liver': (0.5, 0.7), 'sofa2_cardio': (0.9, 1.1), 'sofa2_cns': (0.7, 0.9), 'sofa2_renal': (0.7, 0.9),
        },
        'hirid': {
            'hr': (78, 14), 'sbp': (118, 22), 'dbp': (68, 11), 'map': (83, 14),
            'temp': (37.3, 0.5), 'resp': (17, 4), 'spo2': (96, 3),
            'glu': (135, 48), 'na': (140, 5), 'k': (4.1, 0.6), 'crea': (1.1, 0.7),
            'bili': (1.4, 1.1), 'lact': (2.0, 1.4),
            'hgb': (11.2, 2.0), 'plt': (210, 75), 'wbc': (11.5, 4.5),
            'ph': (7.39, 0.07), 'po2': (92, 19), 'pco2': (39, 7), 'fio2': (42, 19),
            # SOFA-2
            'sofa2': (4.8, 3.6), 'sofa2_resp': (1.1, 1.0), 'sofa2_coag': (0.7, 0.9),
            'sofa2_liver': (0.5, 0.7), 'sofa2_cardio': (1.0, 1.1), 'sofa2_cns': (0.7, 0.9), 'sofa2_renal': (0.8, 1.0),
        },
        # 🆕 MIMIC-III
        'mimic': {
            'hr': (82, 16), 'sbp': (122, 21), 'dbp': (71, 13), 'map': (86, 16),
            'temp': (37.1, 0.5), 'resp': (19, 4), 'spo2': (95, 3),
            'glu': (145, 55), 'na': (139, 5), 'k': (4.1, 0.6), 'crea': (1.3, 0.9),
            'bili': (1.6, 1.3), 'lact': (2.3, 1.6),
            'hgb': (10.8, 2.1), 'plt': (190, 85), 'wbc': (12.5, 5.5),
            'ph': (7.37, 0.08), 'po2': (88, 21), 'pco2': (41, 9), 'fio2': (48, 22),
            # SOFA-2
            'sofa2': (5.5, 4.0), 'sofa2_resp': (1.3, 1.1), 'sofa2_coag': (0.8, 0.9),
            'sofa2_liver': (0.6, 0.8), 'sofa2_cardio': (1.1, 1.2), 'sofa2_cns': (0.8, 1.0), 'sofa2_renal': (0.9, 1.0),
        },
        # 🆕 SICdb
        'sic': {
            'hr': (77, 13), 'sbp': (116, 19), 'dbp': (67, 11), 'map': (82, 13),
            'temp': (37.3, 0.4), 'resp': (17, 3), 'spo2': (97, 2),
            'glu': (132, 46), 'na': (141, 4), 'k': (4.2, 0.5), 'crea': (1.05, 0.65),
            'bili': (1.3, 1.0), 'lact': (1.9, 1.3),
            'hgb': (11.3, 1.9), 'plt': (215, 72), 'wbc': (11.2, 4.2),
            'ph': (7.40, 0.06), 'po2': (93, 18), 'pco2': (38, 6), 'fio2': (41, 18),
            # SOFA-2
            'sofa2': (4.2, 3.3), 'sofa2_resp': (1.0, 1.0), 'sofa2_coag': (0.6, 0.8),
            'sofa2_liver': (0.5, 0.7), 'sofa2_cardio': (0.8, 1.0), 'sofa2_cns': (0.6, 0.8), 'sofa2_renal': (0.7, 0.9),
        },
    }
    
    result = {}
    for db_name, features in databases.items():
        n_records_per_feat = np.random.randint(300, 600)
        
        # 生成长格式数据（concept + value）
        rows = []
        for feat, (mean, std) in features.items():
            values = np.random.normal(mean, std, n_records_per_feat)
            # Clip SOFA scores to valid ranges
            if feat == 'sofa2':
                values = np.clip(np.round(values), 0, 24).astype(int)
            elif feat.startswith('sofa2_'):
                values = np.clip(np.round(values), 0, 4).astype(int)
            patient_ids = np.random.randint(1000, 9999, n_records_per_feat)
            for pid, val in zip(patient_ids, values):
                rows.append({
                    'stay_id': pid,
                    'concept': feat,
                    'value': val,
                })
        
        result[db_name] = pd.DataFrame(rows)
    
    return result


def _generate_mock_cohort_dashboard_data(lang: str = 'en') -> pd.DataFrame:
    """生成模拟的队列仪表盘数据用于演示。
    
    Args:
        lang: 语言
        
    Returns:
        包含患者人口统计学和结局数据的DataFrame
    """
    np.random.seed(42)
    n_patients = 500
    
    # 基本人口统计学
    patient_ids = list(range(30000000, 30000000 + n_patients))
    ages = np.clip(np.random.normal(62, 16, n_patients), 18, 95).astype(int)
    genders = np.random.choice(['M', 'F'], n_patients, p=[0.56, 0.44])  # 使用M/F格式
    
    # 入住类型
    admission_types = np.random.choice(
        ['EMERGENCY', 'ELECTIVE', 'URGENT', 'OBSERVATION'],
        n_patients,
        p=[0.55, 0.25, 0.15, 0.05]
    )
    
    # 住院时长
    los_days = np.clip(np.random.lognormal(1.2, 0.9, n_patients), 0.5, 60)
    
    # 机械通气状态 - 约35%需要
    mech_vent = np.random.choice([True, False], n_patients, p=[0.35, 0.65])
    
    # 血管活性药物 - 约25%使用
    vasopressors = np.random.choice([True, False], n_patients, p=[0.25, 0.75])
    
    # SOFA分数 - 与病情严重度相关
    sofa_scores = np.clip(np.random.poisson(4, n_patients) + (mech_vent.astype(int) * 2), 0, 20)
    
    # 死亡结局 - 与SOFA、年龄、住院时长相关
    mortality_prob = 0.08 + (sofa_scores / 100) + (ages / 500) + (los_days / 200)
    mortality_prob = np.clip(mortality_prob, 0, 0.6)
    mortality = np.random.random(n_patients) < mortality_prob
    
    # 诊断类别
    diagnoses = np.random.choice(
        ['Sepsis', 'Respiratory Failure', 'Cardiac', 'Neurological', 'Post-surgical', 'Trauma', 'Other'],
        n_patients,
        p=[0.25, 0.20, 0.15, 0.12, 0.15, 0.08, 0.05]
    )
    
    df = pd.DataFrame({
        'stay_id': patient_ids,
        'age': ages,
        'gender': genders,
        'admission_type': admission_types,
        'los_days': los_days,
        'los_hours': los_days * 24,  # 添加los_hours列
        'mech_vent': mech_vent,
        'vasopressors': vasopressors,
        'sofa_max': sofa_scores,
        'mortality': mortality,
        'survived': [1 if not m else 0 for m in mortality],  # 添加survived列（1=存活，0=死亡）
        'first_icu_stay': np.random.choice([True, False], n_patients, p=[0.65, 0.35]),  # 添加first_icu_stay列
        'diagnosis_group': diagnoses,
    })
    
    return df


def render_cohort_comparison_page():
    """渲染队列对比可视化页面 - 包含多个子标签页"""
    lang = st.session_state.get('language', 'en')
    
    _coh_title = "Cohort Analysis" if lang == 'en' else "队列分析"
    _coh_sub = "Group comparison, multi-database distribution & dashboard" if lang == 'en' else "分组对比、多数据库分布与队列仪表板"
    st.markdown(f'''
    <div style="margin-bottom:16px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{_coh_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_coh_sub}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 子标签页
    if lang == 'en':
        sub_tabs = st.tabs([
            "👥 Group Comparison",
            "📈 Multi-DB Distribution", 
            "🎯 Cohort Dashboard"
        ])
    else:
        sub_tabs = st.tabs([
            "👥 分组对比",
            "📈 多数据库分布",
            "🎯 队列仪表板"
        ])
    
    with sub_tabs[0]:
        render_group_comparison_subtab(lang)
    
    with sub_tabs[1]:
        render_multidb_distribution_subtab(lang)
    
    with sub_tabs[2]:
        render_cohort_dashboard_subtab(lang)


def _render_demo_generation_card(icon: str, title: str, desc: str):
    """统一的 demo 生成空状态卡片。"""
    st.markdown(
        f'''
        <div style="text-align:center;padding:30px 28px;background:linear-gradient(135deg,#eff6ff 0%,#f0fdfa 55%,#f8fafc 100%);
                    border:1px solid #bfdbfe;border-radius:18px;margin:16px 0 18px;box-shadow:0 10px 28px rgba(37,99,235,0.08)">
            <div style="width:64px;height:64px;margin:0 auto 14px;border-radius:18px;background:linear-gradient(135deg,#2563eb 0%,#0891b2 100%);
                        display:flex;align-items:center;justify-content:center;font-size:1.9rem;box-shadow:0 10px 24px rgba(37,99,235,0.18)">{icon}</div>
            <div style="font-weight:800;color:#0f172a;font-size:1.4rem;letter-spacing:-0.02em">{title}</div>
            <div style="color:#475569;font-size:.95rem;margin-top:8px;line-height:1.65">{desc}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_group_comparison_subtab(lang: str):
    """分组对比子标签页 - 带独立数据加载配置"""
    
    st.markdown("### 👥 " + ("Group Comparison Analysis" if lang == 'en' else "分组对比分析"))
    
    # 获取当前入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')
    
    # ========== Demo模式：需要用户点击生成按钮 ==========
    if entry_mode == 'demo':
        # 检查是否已生成模拟数据
        has_demo_data = 'grp_demographics' in st.session_state and st.session_state.get('grp_is_demo') == True
        
        if not has_demo_data:
            # 尚未生成数据，显示生成界面
            st.markdown("---")
            
            # 居中的配置卡片
            _gen_title = "Generate Demo Cohort Data" if lang == 'en' else "生成演示队列数据"
            _gen_desc = "Configure patient count and generate simulated demographics" if lang == 'en' else "配置患者数量并生成模拟人口统计学数据"
            _render_demo_generation_card("👥", _gen_title, _gen_desc)
            
            # 配置区域
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                n_patients = st.slider(
                    "👥 " + ("Number of Patients" if lang == 'en' else "患者数量"),
                    min_value=50, max_value=500, value=100,
                    key="grp_demo_patients_init"
                )
                
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                
                if st.button(
                    "🚀 " + ("Generate Demo Data" if lang == 'en' else "生成演示数据"),
                    type="primary",
                    use_container_width=True,
                    key="grp_generate_demo_btn"
                ):
                    st.session_state.mock_params['n_patients'] = n_patients
                    demographics_df = _generate_mock_demographics(n_patients, lang)
                    st.session_state['grp_demographics'] = demographics_df
                    st.session_state['grp_loaded_db'] = 'demo'
                    st.session_state['grp_is_demo'] = True
                    st.rerun()
            
            # 显示提示信息
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.info("💡 " + ("Click the button above to generate demo data for cohort analysis" if lang == 'en' else "点击上方按钮生成队列分析演示数据"))
            return  # 未生成数据时不显示下方分析内容
        
        # 已生成数据，显示Demo模式提示
        demo_info = "🎭 Using simulated demographics data for demonstration" if lang == 'en' else "🎭 正在使用模拟人口统计学数据进行演示"
        st.info(demo_info)
        
        # 允许调整模拟数据参数
        with st.expander("⚙️ " + ("Demo Data Settings" if lang == 'en' else "模拟数据设置"), expanded=False):
            n_patients = st.slider(
                "Number of Patients" if lang == 'en' else "患者数量",
                min_value=50, max_value=500, value=st.session_state.mock_params.get('n_patients', 100),
                key="grp_demo_patients"
            )
            if st.button("🔄 " + ("Regenerate Data" if lang == 'en' else "重新生成数据"), key="grp_regen_btn"):
                st.session_state.mock_params['n_patients'] = n_patients
                st.session_state['grp_demographics'] = _generate_mock_demographics(n_patients, lang)
                st.rerun()
    
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
                    st.rerun()

            elif grp_src == "raw":
                # ===== 原始数据库模式 =====
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    data_root = _directory_input(
                        "📁 " + ("ICU Data Root" if lang == 'en' else "ICU数据根目录"),
                        value=os.environ.get('EASYICU_DATA_PATH', ''),
                        input_key="grp_data_root",
                        button_key="grp_data_root_browse",
                        placeholder="/path/to/icudb" if os.name != 'nt' else "D:\\data\\icudb",
                        help="Root directory containing database folders (mimiciv, eicu, aumc, hirid)" if lang == 'en' else "包含数据库文件夹的根目录"
                    )
                    render_directory_structure_guide(lang)
                
                with col2:
                    db_options = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
                    selected_db = st.selectbox(
                        "🏥 " + ("Database" if lang == 'en' else "数据库"),
                        options=list(db_options.keys()),
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
                
                full_data_path = find_database_path(data_root, selected_db)
                
                if os.path.exists(full_data_path):
                    st.success(f"✅ " + (f"Path valid: `{full_data_path}`" if lang == 'en' else f"路径有效: `{full_data_path}`"))
                else:
                    st.warning(f"⚠️ " + (f"Path not found: `{full_data_path}`" if lang == 'en' else f"路径不存在: `{full_data_path}`"))
                
                load_btn = st.button(
                    "🚀 " + ("Load Patient Demographics" if lang == 'en' else "加载患者人口统计学数据"),
                    type="primary",
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
    
    st.markdown("---")
    
    # ========== 分组对比区域 ==========
    if 'grp_demographics' not in st.session_state:
        st.info("👆 " + ("Configure data source and click 'Load' to start" if lang == 'en' else "配置数据源并点击'加载'开始"))
        return
    
    demographics_df = st.session_state['grp_demographics']
    database = st.session_state.get('grp_loaded_db', 'miiv')
    data_path = st.session_state.get('grp_loaded_path', '')
    
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
    
    st.markdown("---")
    
    # 对比模式选择
    st.markdown("#### " + ("🔀 Select Comparison Mode" if lang == 'en' else "🔀 选择对比模式"))
    
    compare_options = {
        'survival': ('💀 Survived vs Deceased', '💀 存活 vs 死亡'),
        'age': ('👴 Age Groups', '👴 年龄分组'),
        'gender': ('👫 Male vs Female', '👫 男性 vs 女性'),
        'los': ('🏥 Short vs Long Stay', '🏥 短住院 vs 长住院'),
    }
    
    compare_mode = st.radio(
        "Comparison Mode" if lang == 'en' else "对比模式",
        options=list(compare_options.keys()),
        format_func=lambda x: compare_options[x][0] if lang == 'en' else compare_options[x][1],
        horizontal=True,
        key="group_comp_mode"
    )
    
    # 根据模式显示额外配置
    if compare_mode == 'age':
        age_threshold = st.slider(
            "Age Threshold" if lang == 'en' else "年龄阈值",
            min_value=30, max_value=90, value=65, step=5,
            key="group_comp_age_threshold"
        )
    elif compare_mode == 'los' and 'los_hours' in demographics_df.columns:
        median_los = demographics_df['los_hours'].median()
        los_threshold = st.slider(
            "LOS Threshold (hours)" if lang == 'en' else "住院时长阈值（小时）",
            min_value=24,
            max_value=int(min(500, demographics_df['los_hours'].quantile(0.95))),
            value=int(median_los),
            step=12,
            key="group_comp_los_threshold"
        )
    
    st.markdown("---")
    
    # ========== 特征模块选择 ==========
    st.markdown("#### " + ("📊 Select Feature Modules" if lang == 'en' else "📊 选择特征模块"))
    
    # 定义所有可用的特征模块
    FEATURE_MODULES = {
        'demographic': {
            'name_en': '👤 Demographics',
            'name_zh': '👤 人口统计学',
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
            'name_en': '📈 Outcomes',
            'name_zh': '📈 结局指标',
            'features': [
                ('mortality', 'ICU Mortality', 'ICU死亡率', 'binary_survival'),
            ],
            'default': True
        },
        'vital': {
            'name_en': '💓 Vital Signs',
            'name_zh': '💓 生命体征',
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
            'name_en': '🧪 Laboratory',
            'name_zh': '🧪 实验室检查',
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
            'name_en': '🩸 Hematology',
            'name_zh': '🩸 血液学',
            'features': [
                ('hgb', 'Hemoglobin (g/dL)', '血红蛋白 (g/dL)', 'continuous'),
                ('plt', 'Platelets (K/uL)', '血小板 (K/uL)', 'continuous'),
                ('wbc', 'WBC (K/uL)', '白细胞 (K/uL)', 'continuous'),
                ('inr_pt', 'INR', 'INR', 'continuous'),
            ],
            'default': True
        },
        'blood_gas': {
            'name_en': '🩸 Blood Gas',
            'name_zh': '🩸 血气分析',
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
            'name_en': '🏥 SOFA Scores',
            'name_zh': '🏥 SOFA评分',
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
    
    # 模块多选
    default_modules = [k for k, v in FEATURE_MODULES.items() if v.get('default', False)]
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
        
        if concepts_to_load:
            with st.expander("🔬 " + (f"Features to load: {len(concepts_to_load)}" if lang == 'en' else f"待加载特征: {len(concepts_to_load)}个"), expanded=False):
                st.caption(", ".join(concepts_to_load))
    
    st.markdown("---")
    
    # 执行分组
    try:
        base_df = demographics_df
        group1_ids, group2_ids = [], []
        group1_name, group2_name = "", ""
        show_mortality = True
        
        # 检测ID列名（支持stay_id或patient_id）
        id_col = 'stay_id' if 'stay_id' in base_df.columns else 'patient_id'
        
        if compare_mode == 'survival':
            if 'survived' not in base_df.columns:
                st.warning("Survival data not available" if lang == 'en' else "无存活状态数据")
                return
            
            survived_df = base_df[base_df['survived'] == 1]
            deceased_df = base_df[base_df['survived'] == 0]
            group1_ids = survived_df[id_col].tolist()
            group2_ids = deceased_df[id_col].tolist()
            group1_name = 'Survived' if lang == 'en' else '存活'
            group2_name = 'Deceased' if lang == 'en' else '死亡'
            show_mortality = False
            
        elif compare_mode == 'age':
            threshold = st.session_state.get('group_comp_age_threshold', 65)
            young_df = base_df[base_df['age'] < threshold]
            old_df = base_df[base_df['age'] >= threshold]
            group1_ids = young_df[id_col].tolist()
            group2_ids = old_df[id_col].tolist()
            group1_name = f'Age < {threshold}' if lang == 'en' else f'年龄 < {threshold}'
            group2_name = f'Age ≥ {threshold}' if lang == 'en' else f'年龄 ≥ {threshold}'
            
        elif compare_mode == 'gender':
            if 'gender' not in base_df.columns:
                st.warning("Gender data not available" if lang == 'en' else "无性别数据")
                return
            male_df = base_df[base_df['gender'] == 'M']
            female_df = base_df[base_df['gender'] == 'F']
            group1_ids = male_df[id_col].tolist()
            group2_ids = female_df[id_col].tolist()
            group1_name = 'Male' if lang == 'en' else '男性'
            group2_name = 'Female' if lang == 'en' else '女性'
            
        elif compare_mode == 'los':
            if 'los_hours' not in base_df.columns:
                st.warning("Length of stay data not available" if lang == 'en' else "无住院时长数据")
                return
            threshold = st.session_state.get('group_comp_los_threshold', int(base_df['los_hours'].median()))
            short_df = base_df[base_df['los_hours'] < threshold]
            long_df = base_df[base_df['los_hours'] >= threshold]
            group1_ids = short_df[id_col].tolist()
            group2_ids = long_df[id_col].tolist()
            group1_name = f'LOS < {threshold}h' if lang == 'en' else f'住院 < {threshold}h'
            group2_name = f'LOS ≥ {threshold}h' if lang == 'en' else f'住院 ≥ {threshold}h'
        
        # 分组统计概览
        st.markdown("#### " + ("📊 Group Overview" if lang == 'en' else "📊 分组概览"))
        
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
        
        st.markdown("---")
        
        # ========== 基线特征对比表 (Table One) ==========
        st.markdown("#### " + ("📋 Baseline Characteristics Comparison" if lang == 'en' else "📋 基线特征对比表"))
        
        from scipy import stats
        
        # 获取两组数据 - 使用动态ID列
        group1_df = base_df[base_df[id_col].isin(group1_ids)].copy()
        group2_df = base_df[base_df[id_col].isin(group2_ids)].copy()
        
        # ========== 加载额外特征数据 ==========
        # 确定需要加载的概念
        concepts_to_load = []
        for mod in selected_modules:
            if mod not in ['demographic', 'outcome']:  # 这些从 demographics 表获取
                for feat in FEATURE_MODULES[mod]['features']:
                    concepts_to_load.append(feat[0])
        
        # 检查是否有需要加载的特征且尚未加载
        feature_data = st.session_state.get('grp_feature_data', {})
        
        # 合并两组患者ID
        all_patient_ids = list(set(group1_ids + group2_ids))
        
        if concepts_to_load:
            # 检查是否有新的概念需要加载
            missing_concepts = [c for c in concepts_to_load if c not in feature_data]
            
            if missing_concepts:
                # Demo模式：自动生成模拟数据，无需用户点击
                if entry_mode == 'demo' or database == 'demo':
                    auto_load_msg = "Auto-loading simulated features for demo mode..." if lang == 'en' else "演示模式自动加载模拟特征数据..."
                    with st.spinner(auto_load_msg):
                        # 特征的模拟参数 (均值, 标准差)
                        mock_params = {
                            'hr': (80, 15), 'sbp': (120, 20), 'dbp': (70, 12), 'map': (85, 15),
                            'resp': (18, 4), 'temp': (37.0, 0.6), 'o2sat': (96, 3),
                            'glu': (120, 40), 'na': (140, 4), 'k': (4.2, 0.5),
                            'crea': (1.2, 0.8), 'bili': (1.5, 2.0), 'lact': (1.5, 1.0),
                            'hgb': (11, 2), 'plt': (200, 80), 'wbc': (10, 4),
                            'alb': (3.5, 0.6), 'pco2': (40, 8), 'po2': (90, 20),
                            'ph': (7.38, 0.08), 'fio2': (40, 20),
                        }
                        
                        for concept in missing_concepts:
                            mean, std = mock_params.get(concept, (50, 15))
                            values = np.random.normal(mean, std, len(all_patient_ids))
                            feature_data[concept] = pd.DataFrame({
                                id_col: all_patient_ids,
                                concept: values
                            })
                        
                        st.session_state['grp_feature_data'] = feature_data
                else:
                    # 真实数据模式：显示加载提示和按钮
                    st.info(f"🔬 " + (f"{len(missing_concepts)} features need to be loaded: " if lang == 'en' else f"需要加载 {len(missing_concepts)} 个特征: ") + ", ".join(missing_concepts[:5]) + ("..." if len(missing_concepts) > 5 else ""))
                    
                    load_features_btn = st.button(
                        "🚀 " + (f"Load {len(missing_concepts)} Features" if lang == 'en' else f"加载 {len(missing_concepts)} 个特征"),
                        type="primary",
                        key="grp_load_features"
                    )
                    
                    if load_features_btn:
                        # Real Data模式：从数据库加载
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
                                            verbose=False
                                        )
                                        if df_concept is not None and len(df_concept) > 0:
                                            # 确定ID列
                                            feat_id_col = None
                                            for col in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'hadm_id']:
                                                if col in df_concept.columns:
                                                    feat_id_col = col
                                                    break
                                            if feat_id_col is None:
                                                feat_id_col = df_concept.columns[0]
                                            
                                            # 取每个患者的平均值
                                            if concept in df_concept.columns:
                                                agg_df = df_concept.groupby(feat_id_col)[concept].mean().reset_index()
                                                agg_df.columns = [id_col, concept]
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
        
        # 合并已加载的特征数据到分组 DataFrame
        # 确保 ID 类型一致
        group1_df[id_col] = group1_df[id_col].astype(int)
        group2_df[id_col] = group2_df[id_col].astype(int)
        
        for concept, feat_df in feature_data.items():
            if concept not in group1_df.columns and concept in concepts_to_load:
                try:
                    feat_df_copy = feat_df.copy()
                    # 检测特征数据中的ID列
                    feat_id_col = None
                    for col in ['stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
                        if col in feat_df_copy.columns:
                            feat_id_col = col
                            break
                    if feat_id_col is None:
                        continue
                    feat_df_copy[feat_id_col] = feat_df_copy[feat_id_col].astype(int)
                    # 重命名为统一的id_col
                    if feat_id_col != id_col:
                        feat_df_copy[id_col] = feat_df_copy[feat_id_col]
                    group1_df = group1_df.merge(feat_df_copy[[id_col, concept]], on=id_col, how='left')
                    group2_df = group2_df.merge(feat_df_copy[[id_col, concept]], on=id_col, how='left')
                except Exception:
                    pass
        
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
            except:
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
            except:
                return '-'
        
        # 构建表格数据 - 根据选中的模块动态生成
        table_data = []
        
        # 样本量 (总是显示)
        table_data.append({
            'Module': '',
            'Characteristic': 'N' if lang == 'en' else '样本量',
            group1_name: f"{len(group1_df):,}",
            group2_name: f"{len(group2_df):,}",
            'p-value': ''
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
                            'p-value': calc_pvalue_continuous(group1_df['age'], group2_df['age'])
                        })
                    elif feat_key == 'gender' and 'gender' in group1_df.columns:
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_categorical(group1_df['gender'], 'M', len(group1_df)),
                            group2_name: format_categorical(group2_df['gender'], 'M', len(group2_df)),
                            'p-value': calc_pvalue_categorical(group1_df['gender'], group2_df['gender'], ['M', 'F'])
                        })
                    elif feat_key == 'los_days' and 'los_hours' in group1_df.columns:
                        g1_los = group1_df['los_hours'] / 24
                        g2_los = group2_df['los_hours'] / 24
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_continuous(g1_los, 'los'),
                            group2_name: format_continuous(g2_los, 'los'),
                            'p-value': calc_pvalue_continuous(g1_los, g2_los)
                        })
                    elif feat_key == 'first_icu_stay' and 'first_icu_stay' in group1_df.columns:
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: format_categorical(group1_df['first_icu_stay'], True, len(group1_df)),
                            group2_name: format_categorical(group2_df['first_icu_stay'], True, len(group2_df)),
                            'p-value': calc_pvalue_categorical(group1_df['first_icu_stay'], group2_df['first_icu_stay'], [True, False])
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
                            'p-value': calc_pvalue_categorical(group1_df['survived'], group2_df['survived'], [0, 1])
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
                            'p-value': calc_pvalue_continuous(group1_df[feat_key], group2_df[feat_key])
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
                                'p-value': calc_pvalue_continuous(g1_vals, g2_vals) if len(g1_vals) > 0 and len(g2_vals) > 0 else '-'
                            })
                        else:
                            table_data.append({
                                'Module': module_display,
                                'Characteristic': feat_display,
                                group1_name: 'No data',
                                group2_name: 'No data',
                                'p-value': '-'
                            })
                    elif feat_key in concepts_to_load:
                        # 特征需要加载但尚未加载
                        table_data.append({
                            'Module': module_display,
                            'Characteristic': feat_display,
                            group1_name: '⏳ 待加载',
                            group2_name: '⏳ 待加载',
                            'p-value': '-'
                        })
        
        # 显示表格
        result_df = pd.DataFrame(table_data)
        
        # 使用 Streamlit 表格并应用样式
        st.dataframe(
            result_df,
            width='stretch',
            hide_index=True,
            column_config={
                'Module': st.column_config.TextColumn('Module' if lang == 'en' else '模块', width='small'),
                'Characteristic': st.column_config.TextColumn('Characteristic' if lang == 'en' else '特征', width='medium'),
                group1_name: st.column_config.TextColumn(group1_name, width='medium'),
                group2_name: st.column_config.TextColumn(group2_name, width='medium'),
                'p-value': st.column_config.TextColumn('p-value', width='small'),
            }
        )
        
        # 统计方法说明
        st.markdown("---")
        stats_note = """**Statistical Methods:**
- Continuous variables: Mean ± SD (Median [IQR]), Mann-Whitney U test
- Categorical variables: n (%), Chi-square test
- p < 0.05 considered statistically significant""" if lang == 'en' else """**统计方法说明：**
- 连续变量：Mean ± SD (Median [IQR])，Mann-Whitney U 检验
- 分类变量：n (%)，卡方检验
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


def render_multidb_distribution_subtab(lang: str):
    """多数据库特征分布对比子标签页"""
    import plotly.graph_objects as go
    
    # 🔧 FIX: 使用容器包装标题，确保与下方内容分隔，增加足够的间距
    st.markdown("""<div style="margin-bottom: 40px;">
        <h3 style="margin: 0 0 15px 0; padding: 0;">📈 """ + ("Multi-Database Feature Distribution Comparison" if lang == 'en' else "多数据库特征分布对比") + """</h3>
        <hr style="margin: 0 0 30px 0; border: none; border-top: 2px solid #e0e0e0;">
    </div>""", unsafe_allow_html=True)
    
    # 获取入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')
    
    # ========== Demo模式：需要用户点击生成按钮 ==========
    if entry_mode == 'demo':
        # 检查是否已生成数据
        has_demo_data = 'multidb_data' in st.session_state and st.session_state.get('multidb_is_demo') == True
        
        if not has_demo_data:
            # 尚未生成数据，显示生成界面
            st.markdown("---")
            
            # 居中的配置卡片
            _render_demo_generation_card(
                "📈",
                "Generate Multi-DB Distribution Data" if lang == 'en' else "生成多数据库分布数据",
                "Click below to generate simulated feature distribution across multiple databases" if lang == 'en' else "点击下方按钮生成多数据库特征分布模拟数据",
            )
            
            # 生成按钮
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                
                if st.button(
                    "🚀 " + ("Generate Demo Data" if lang == 'en' else "生成演示数据"),
                    type="primary",
                    use_container_width=True,
                    key="multidb_generate_demo_btn"
                ):
                    # 生成模拟的多数据库特征数据
                    mock_data = _generate_mock_multidb_data(lang)
                    st.session_state['multidb_data'] = mock_data
                    # 🔧 扩展默认显示的特征，包含更多临床指标
                    st.session_state['multidb_concepts'] = [
                        'hr', 'sbp', 'dbp', 'map', 'temp', 'resp', 'spo2',  # Vitals
                        'glu', 'na', 'k', 'crea', 'bili', 'lact',  # Labs
                        'hgb', 'plt', 'wbc',  # Hematology
                        'ph', 'po2', 'pco2', 'fio2',  # Blood Gas
                        'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',  # SOFA-2
                    ]
                    st.session_state['multidb_is_demo'] = True
                    st.rerun()
            
            # 显示提示信息
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.info("💡 " + ("Click the button above to generate demo data for multi-database distribution analysis" if lang == 'en' else "点击上方按钮生成多数据库分布分析演示数据"))
            return  # 未生成数据时不显示下方分析内容
        
        # 已生成数据，显示Demo模式提示
        st.info("🎭 " + ("Demo Mode: Showing simulated multi-database distribution" if lang == 'en' else "演示模式：显示模拟的多数据库分布"))
    
    # ========== Real Data模式 ==========
    if entry_mode != 'demo':
        # 配置区域
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            data_root = st.text_input(
                "🗂️ " + ("ICU Data Root" if lang == 'en' else "ICU数据根目录"),
                value=os.environ.get('EASYICU_DATA_PATH', ''),
                placeholder="/path/to/icudb" if os.name != 'nt' else "D:\\data\\icudb",
                key="multidb_data_root"
            )
            # 添加目录结构指南
            render_directory_structure_guide(lang)
        
        with col2:
            # 数据库选择
            db_options = ['miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic']
            db_labels = {'miiv': 'MIMIC-IV 🟢', 'eicu': 'eICU 🟠', 'aumc': 'Amsterdam 🔵', 'hirid': 'HiRID 🔴', 'mimic': 'MIMIC-III 🟣', 'sic': 'SICdb ⚫'}
            selected_dbs = st.multiselect(
                "🏥 " + ("Databases" if lang == 'en' else "数据库"),
                options=db_options,
                default=['miiv', 'eicu'],
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
        
        st.markdown("---")
        
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
            st.plotly_chart(fig, use_container_width=True)
            
            # 单特征详细对比
            st.markdown("---")
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
                    st.plotly_chart(fig_single, use_container_width=True)
                with col2:
                    st.markdown("**Statistics**" if lang == 'en' else "**统计信息**")
                    st.dataframe(
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


def render_cohort_dashboard_subtab(lang: str):
    """队列仪表板子标签页 - 使用Plotly实现交互式可视化"""
    
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.markdown("### 🎯 " + ("Cohort Dashboard" if lang == 'en' else "队列仪表板"))
    
    # 获取入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')
    
    # ========== Demo模式：需要用户点击生成按钮 ==========
    if entry_mode == 'demo':
        # 检查是否已生成数据
        has_demo_data = 'dash_demographics' in st.session_state and st.session_state.get('dash_is_demo') == True
        
        if not has_demo_data:
            # 尚未生成数据，显示生成界面
            st.markdown("---")
            
            # 居中的配置卡片
            _render_demo_generation_card(
                "🎯",
                "Generate Cohort Dashboard Data" if lang == 'en' else "生成队列仪表板数据",
                "Click below to generate simulated cohort dashboard with interactive visualizations" if lang == 'en' else "点击下方按钮生成带有交互式可视化的队列仪表板",
            )
            
            # 生成按钮
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                
                if st.button(
                    "🚀 " + ("Generate Demo Dashboard" if lang == 'en' else "生成演示仪表板"),
                    type="primary",
                    use_container_width=True,
                    key="dash_generate_demo_btn"
                ):
                    demo_df = _generate_mock_cohort_dashboard_data(lang)
                    st.session_state['dash_demographics'] = demo_df
                    st.session_state['dash_loaded_db'] = 'Demo'
                    st.session_state['dash_is_demo'] = True
                    st.rerun()
            
            # 显示提示信息
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.info("💡 " + ("Click the button above to generate demo data for cohort dashboard" if lang == 'en' else "点击上方按钮生成队列仪表板演示数据"))
            return  # 未生成数据时不显示下方分析内容
        
        # 已生成数据，显示Demo模式提示
        st.info("🎭 " + ("Demo Mode: Showing simulated cohort dashboard" if lang == 'en' else "演示模式：显示模拟的队列仪表板"))
    
    # ========== Real Data模式：显示数据配置 ==========
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
            dash_src = st.radio(
                _src_label, _src_keys,
                index=_src_keys.index(_default_src),
                format_func=lambda x: _src_labels[x],
                horizontal=True, key="dash_data_source"
            )

            if dash_src == "demo":
                # ===== 模拟数据模式 =====
                load_btn = st.button(
                    "🚀 " + ("Generate Demo Dashboard" if lang == 'en' else "生成演示仪表板"),
                    type="primary", key="dash_load_demo_btn"
                )
                if load_btn:
                    demo_df = _generate_mock_cohort_dashboard_data(lang)
                    st.session_state['dash_demographics'] = demo_df
                    st.session_state['dash_loaded_db'] = 'Demo'
                    st.session_state['dash_is_demo'] = True
                    st.rerun()

            elif dash_src == "raw":
                # ===== 原始数据库模式 =====
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    data_root = _directory_input(
                        "📁 " + ("ICU Data Root" if lang == 'en' else "ICU数据根目录"),
                        value=os.environ.get('EASYICU_DATA_PATH', ''),
                        input_key="dash_data_root",
                        button_key="dash_data_root_browse",
                        placeholder="/path/to/icudb" if os.name != 'nt' else "D:\\data\\icudb",
                        help="Root directory containing database folders" if lang == 'en' else "包含数据库文件夹的根目录"
                    )
                    render_directory_structure_guide(lang)
                
                with col2:
                    db_options = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
                    selected_db = st.selectbox(
                        "🏥 " + ("Database" if lang == 'en' else "数据库"),
                        options=list(db_options.keys()),
                        format_func=lambda x: db_options[x],
                        key="dash_db_select"
                    )
                
                with col3:
                    max_patients = st.number_input(
                        "👥 " + ("Max Patients" if lang == 'en' else "最大患者数"),
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                        key="dash_max_patients"
                    )
                
                full_data_path = find_database_path(data_root, selected_db)
                
                if os.path.exists(full_data_path):
                    st.success(f"✅ " + (f"Path valid: `{full_data_path}`" if lang == 'en' else f"路径有效: `{full_data_path}`"))
                else:
                    st.warning(f"⚠️ " + (f"Path not found: `{full_data_path}`" if lang == 'en' else f"路径不存在: `{full_data_path}`"))
                
                load_btn = st.button(
                    "🚀 " + ("Load Dashboard Data" if lang == 'en' else "加载仪表板数据"),
                    type="primary",
                    key="dash_load_btn"
                )
                
                if load_btn:
                    try:
                        from easyicu.patient_filter import PatientFilter
                        
                        with st.spinner("Loading demographics..." if lang == 'en' else "正在加载..."):
                            pf = PatientFilter(database=selected_db, data_path=full_data_path)
                            demographics_df = pf._load_demographics()
                            
                            if len(demographics_df) > max_patients:
                                demographics_df = demographics_df.head(max_patients)
                            
                            st.session_state['dash_demographics'] = demographics_df
                            st.session_state['dash_loaded_db'] = selected_db
                            st.session_state['dash_loaded_path'] = full_data_path
                            st.session_state['dash_is_demo'] = False
                            
                        st.success(f"✅ Loaded {len(demographics_df):,} patients" if lang == 'en' else f"✅ 已加载 {len(demographics_df):,} 名患者")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif dash_src == "exported":
                # ===== 导出文件模式 =====
                col1, col2 = st.columns([3, 2])
                with col1:
                    export_root = _directory_input(
                        "📦 " + ("Folder with Exported Data Files" if lang == 'en' else "存放导出结果文件的文件夹"),
                        value=st.session_state.get('export_path', ''),
                        input_key="dash_export_root",
                        button_key="dash_export_root_browse",
                        placeholder="/path/to/easyicu_export" if os.name != 'nt' else "D:\\easyicu_export",
                        help="Choose the folder that contains EasyICU exported result folders" if lang == 'en' else "选择包含 EasyICU 导出结果子文件夹的目录"
                    )
                
                _export_folders = _scan_export_folders(export_root if 'export_root' in dir() else st.session_state.get('dash_export_root', ''))
                
                with col2:
                    if _export_folders:
                        folder_options = {f[0]: f"📁 {f[0]} ({f[1]} files)" for f in _export_folders}
                        selected_folder = st.selectbox(
                            "📁 " + ("Select an Export Result Folder" if lang == 'en' else "选择一批导出结果"),
                            options=list(folder_options.keys()),
                            format_func=lambda x: folder_options[x],
                            key="dash_export_folder"
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
                        key="dash_load_export_btn"
                    )
                    
                    if load_btn:
                        try:
                            with st.spinner("Loading..." if lang == 'en' else "加载中..."):
                                demographics_df = _load_demographics_from_export(selected_path)
                                _detected_db = demographics_df.attrs.get('detected_db', 'unknown')
                                
                                st.session_state['dash_demographics'] = demographics_df
                                st.session_state['dash_loaded_db'] = _detected_db
                                st.session_state['dash_loaded_path'] = selected_path
                                st.session_state['dash_is_demo'] = False
                                
                            st.success(f"✅ Loaded {len(demographics_df):,} patients from exported result files" if lang == 'en' else f"✅ 已从这批导出结果文件中加载 {len(demographics_df):,} 名患者")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # ========== 仪表板内容 ==========
    if 'dash_demographics' not in st.session_state:
        st.info("👆 " + ("Configure data source and click 'Load' to view dashboard" if lang == 'en' else "配置数据源并点击'加载'查看仪表板"))
        return
    
    df = st.session_state['dash_demographics']
    
    try:
        # ========== 顶部指标卡片 ==========
        st.markdown("#### " + ("📊 Key Metrics" if lang == 'en' else "📊 关键指标"))
        
        metric_cols = st.columns(6)
        
        def metric_card(value, label, bg_gradient):
            st.markdown(f"""
            <div style="background: {bg_gradient}; 
                        padding: 15px 5px; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 1.8rem; font-weight: bold;">{value}</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[0]:
            metric_card(
                f"{len(df):,}", 
                "Total Patients" if lang == 'en' else "患者总数",
                "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
            )
        
        with metric_cols[1]:
            avg_age = df['age'].mean() if 'age' in df.columns else 0
            metric_card(
                f"{avg_age:.1f}", 
                "Mean Age" if lang == 'en' else "平均年龄",
                "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
            )
        
        with metric_cols[2]:
            male_pct = (df['gender'] == 'M').mean() * 100 if 'gender' in df.columns else 0
            metric_card(
                f"{male_pct:.1f}%", 
                "Male %" if lang == 'en' else "男性占比",
                "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
            )
        
        with metric_cols[3]:
            median_los = df['los_hours'].median() / 24 if 'los_hours' in df.columns else 0
            metric_card(
                f"{median_los:.1f}", 
                "Median LOS (days)" if lang == 'en' else "中位住院(天)",
                "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
            )
        
        with metric_cols[4]:
            mortality = (1 - df['survived'].mean()) * 100 if 'survived' in df.columns else 0
            metric_card(
                f"{mortality:.1f}%", 
                "Mortality" if lang == 'en' else "死亡率",
                "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            )
        
        with metric_cols[5]:
            first_icu_pct = df['first_icu_stay'].mean() * 100 if 'first_icu_stay' in df.columns else 0
            metric_card(
                f"{first_icu_pct:.1f}%", 
                "First ICU Stay" if lang == 'en' else "首次ICU",
                "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)"
            )
        
        st.markdown("---")
        
        # ========== 图表行1: 年龄分布和性别/生存 ==========
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("##### " + ("Age Distribution" if lang == 'en' else "年龄分布"))
            if 'age' in df.columns:
                fig = px.histogram(
                    df, 
                    x='age',
                    nbins=20,
                    color_discrete_sequence=['#667eea'],
                    labels={'age': "Age" if lang == 'en' else "年龄", 'count': "Count" if lang == 'en' else "人数"},
                    template="plotly_white"
                )
                fig.update_layout(bargap=0.1, margin=dict(l=20, r=20, t=20, b=20), height=320, font=dict(size=14, color='black'))
                fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                st.plotly_chart(fig, use_container_width=True, key="dash_age_dist")
            else:
                st.warning("No 'age' column found" if lang == 'en' else "未找到'age'列")
        
        with chart_col2:
            st.markdown("##### " + ("Gender & Survival Breakdown" if lang == 'en' else "性别与存活分布"))
            if 'gender' in df.columns and 'survived' in df.columns:
                # 预处理数据以进行可视化
                df_pie_gender = df['gender'].value_counts().reset_index()
                df_pie_gender.columns = ['label', 'value']
                
                df_pie_survival = df['survived'].value_counts().reset_index()
                df_pie_survival.columns = ['label', 'value']
                # 转换标签
                survived_label = "Survived" if lang == 'en' else "存活"
                deceased_label = "Deceased" if lang == 'en' else "死亡"
                df_pie_survival['label'] = df_pie_survival['label'].map({1: survived_label, 0: deceased_label})
                
                # 创建子图
                fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                                   subplot_titles=("Gender" if lang == 'en' else "性别", 
                                                   "Survival" if lang == 'en' else "存活"))
                
                fig.add_trace(go.Pie(labels=df_pie_gender['label'], values=df_pie_gender['value'], 
                                    name="Gender", marker_colors=['#4facfe', '#fa709a']), 1, 1)
                
                fig.add_trace(go.Pie(labels=df_pie_survival['label'], values=df_pie_survival['value'], 
                                    name="Survival", marker_colors=['#38ef7d', '#f5576c']), 1, 2)
                
                fig.update_traces(hole=.4, hoverinfo="label+percent+name")
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320, showlegend=True,
                                 legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                                 font=dict(size=14, color='black'))
                st.plotly_chart(fig, use_container_width=True, key="dash_pie_charts")
            else:
                st.warning("Data mismatch for pie charts" if lang == 'en' else "饼图数据缺失")
        
        # ========== 图表行2: 住院时长和死亡率趋势 ==========
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            st.markdown("##### " + ("Length of Stay Distribution" if lang == 'en' else "住院时长分布"))
            if 'los_hours' in df.columns:
                # 截断极值以便更好展示
                los_days = df['los_hours'] / 24
                p95 = los_days.quantile(0.95)
                df_filtered = df[los_days <= p95].copy()
                df_filtered['los_days'] = df_filtered['los_hours'] / 24
                
                median_los = los_days.median()
                
                fig = px.histogram(
                    df_filtered, 
                    x='los_days',
                    nbins=30,
                    color_discrete_sequence=['#11998e'],
                    labels={'los_days': "LOS (Days)" if lang == 'en' else "住院天数"},
                    template="plotly_white"
                )
                
                # 增加中位数线
                fig.add_vline(x=median_los, line_width=3, line_dash="dash", line_color="#f5576c",
                             annotation_text=f"Median: {median_los:.1f}d", 
                             annotation_position="top right")
                
                fig.update_layout(bargap=0.1, margin=dict(l=20, r=20, t=20, b=20), height=320, font=dict(size=14, color='black'))
                fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                st.plotly_chart(fig, use_container_width=True, key="dash_los_chart")
            else:
                st.warning("No 'los_hours' column" if lang == 'en' else "未找到'los_hours'列")
        
        with chart_col4:
            st.markdown("##### " + ("Mortality by Age Group" if lang == 'en' else "各年龄段死亡率趋势"))
            if 'age' in df.columns and 'survived' in df.columns:
                # 预处理数据
                df_age = df.copy()
                age_bins = [0, 30, 40, 50, 60, 70, 80, 90, 120]
                age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '≥90']
                df_age['age_group'] = pd.cut(df_age['age'], bins=age_bins, labels=age_labels, right=False)
                
                stats = df_age.groupby('age_group', observed=True).agg(
                    total=('survived', 'count'),
                    deaths=('survived', lambda x: (x == 0).sum())
                ).reset_index()
                stats['mortality'] = (stats['deaths'] / stats['total'] * 100).round(1)
                
                # 双轴图：柱状图（人数）+折线图（死亡率）
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 柱状图 - 患者数
                fig.add_trace(
                    go.Bar(x=stats['age_group'].astype(str), y=stats['total'], name="Patients" if lang == 'en' else "患者数",
                          marker_color='rgba(102, 126, 234, 0.6)'),
                    secondary_y=False,
                )
                
                # 折线图 - 死亡率
                fig.add_trace(
                    go.Scatter(x=stats['age_group'].astype(str), y=stats['mortality'], name="Mortality %" if lang == 'en' else "死亡率 %",
                              mode='lines+markers', marker_color='#f5576c', line=dict(width=3)),
                    secondary_y=True,
                )
                
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=20, b=40),
                    height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(size=14, color='black'),
                )
                
                fig.update_yaxes(title_text="Count" if lang == 'en' else "人数", secondary_y=False,
                                tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                fig.update_yaxes(title_text="Mortality %" if lang == 'en' else "死亡率 %", secondary_y=True, range=[0, 100],
                                tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                
                st.plotly_chart(fig, use_container_width=True, key="dash_mortality_chart")
            else:
                st.warning("Data not available" if lang == 'en' else "数据缺失")
                
    except Exception as e:
        st.error(f"Render error: {e}")
        import traceback
        st.code(traceback.format_exc())

def render_convert_dialog():
    """Render CSV to Parquet conversion dialog."""

    lang = st.session_state.get('language', 'en')
    source_path = st.session_state.get('convert_source_path', '')
    
    dialog_title = "## 🔄 CSV to Parquet Conversion" if lang == 'en' else "## 🔄 CSV 转换为 Parquet"
    st.markdown(dialog_title)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    source_info = f"📁 Source directory: `{source_path}`" if lang == 'en' else f"📁 源目录: `{source_path}`"
    st.info(source_info)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 目标目录（默认同目录）
        target_label = "Parquet Output Directory" if lang == 'en' else "Parquet输出目录"
        target_help = "Converted Parquet files will be saved to this directory" if lang == 'en' else "转换后的Parquet文件将保存到此目录"
        target_path = st.text_input(
            target_label,
            value=source_path,
            help=target_help
        )
    
    with col2:
        # 转换选项
        overwrite_label = "Overwrite existing Parquet files" if lang == 'en' else "覆盖已存在的Parquet文件"
        overwrite = st.checkbox(overwrite_label, value=False)
    
    # 扫描可转换文件
    if source_path and Path(source_path).exists():
        csv_files = list(Path(source_path).rglob('*.csv')) + list(Path(source_path).rglob('*.csv.gz'))
        found_msg = f"**Found {len(csv_files)} CSV files to convert**" if lang == 'en' else f"**发现 {len(csv_files)} 个CSV文件可转换**"
        st.markdown(found_msg)
        
        view_label = "View file list" if lang == 'en' else "查看文件列表"
        with st.expander(view_label, expanded=False):
            for f in csv_files[:20]:
                size_mb = f.stat().st_size / (1024 * 1024)
                st.caption(f"• {f.name} ({size_mb:.1f} MB)")
            if len(csv_files) > 20:
                more_msg = f"... and {len(csv_files) - 20} more files" if lang == 'en' else f"... 及其他 {len(csv_files) - 20} 个文件"
                st.caption(more_msg)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        start_label = "🚀 Start Conversion" if lang == 'en' else "🚀 开始转换"
        if st.button(start_label, type="primary", width="stretch"):
            if not target_path or not Path(target_path).exists():
                err_msg = "❌ Please set a valid output directory" if lang == 'en' else "❌ 请设置有效的输出目录"
                st.error(err_msg)
            else:
                spinner_msg = "Converting..." if lang == 'en' else "正在转换..."
                with st.spinner(spinner_msg):
                    success, failed = convert_csv_to_parquet(source_path, target_path, overwrite)
                
                # 转换完成后自动重新验证
                _database = st.session_state.get('database', 'miiv')
                _revalidation = validate_database_path(target_path, _database)
                
                if _revalidation['valid']:
                    # 全部就绪：转换成功 + 验证通过
                    st.session_state.path_validated = True
                    st.session_state.data_path = target_path
                    st.session_state.last_validation = _revalidation
                    st.session_state.last_validated_path = target_path
                    st.session_state.show_convert_dialog = False
                    _done_msg = (f"✅ Setup complete! Converted {success} items, all data validated."
                                 if lang == 'en' else
                                 f"✅ 设置完成！已转换 {success} 项，数据验证通过。")
                    st.success(_done_msg)
                    st.balloons()
                    import time as _t; _t.sleep(1.5)
                    st.rerun()
                elif success > 0:
                    # 部分完成：有转换但验证未完全通过
                    st.session_state.last_validation = _revalidation
                    st.session_state.last_validated_path = target_path
                    _partial_msg = (f"⚠️ Converted {success} items, but some data still needs attention."
                                    if lang == 'en' else
                                    f"⚠️ 已转换 {success} 项，但部分数据仍需处理。")
                    st.warning(_partial_msg)
                    st.error(_revalidation['message'])
                elif failed > 0:
                    fail_msg = (f"⚠️ {failed} files failed to convert. Please check error messages above."
                                if lang == 'en' else
                                f"⚠️ {failed} 个文件转换失败，请查看上方错误信息。")
                    st.warning(fail_msg)
                else:
                    no_files_msg = ("⚠️ No files were converted. Please check your data path."
                                    if lang == 'en' else
                                    "⚠️ 没有文件被转换，请检查数据路径。")
                    st.warning(no_files_msg)
    
    with col2:
        cancel_label = "❌ Cancel" if lang == 'en' else "❌ 取消"
        if st.button(cancel_label, width="stretch"):
            st.session_state.show_convert_dialog = False
            st.rerun()


def convert_csv_to_parquet(source_dir: str, target_dir: str, overwrite: bool = False) -> tuple:
    """将目录下的CSV文件转换为Parquet格式。
    
    大表自动使用分桶转换，普通表使用 DuckDB 直接转换。
    HiRID 特殊处理：已经是 parquet 格式，只需分桶转换。
    """
    import gc
    import time
    
    # 获取数据库类型
    database = st.session_state.get('database', 'miiv')
    
    # HiRID 特殊处理：数据已经是 parquet 格式，只需分桶
    if database == 'hirid':
        return _convert_hirid_data(source_dir, target_dir, overwrite)
    
    # 定义需要分桶转换的大表
    BUCKET_TABLES = {
        'miiv': {
            'chartevents': ('itemid', 100),
            'labevents': ('itemid', 100),
            'inputevents': ('itemid', 50),
        },
        'eicu': {
            'nursecharting': ('nursingchartcelltypevalname', 30),
            'lab': ('labname', 50),
        },
        'aumc': {
            'numericitems': ('itemid', 100),
            'listitems': ('itemid', 50),
        },
        'hirid': {
            'observations': ('variableid', 100),
            'pharma': ('pharmaid', 50),
        },
        'mimic': {
            'chartevents': ('itemid', 100),
            'labevents': ('itemid', 100),
        },
        'sic': {
            'data_float_h': ('dataid', 50),
            'laboratory': ('laboratoryid', 50),
        },
    }
    
    try:
        from easyicu.duckdb_converter import DuckDBConverter
        from easyicu.bucket_converter import convert_to_buckets, BucketConfig
        import time
    except ImportError as e:
        st.error(f"Converter not available: {e}")
        return 0, 0
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 收集 CSV 文件并去重：同一表名的 .csv 和 .csv.gz 只保留一个（优先 .csv.gz 更小更快）
    _all_csvs = list(source_path.rglob('*.csv')) + list(source_path.rglob('*.csv.gz'))
    _csv_by_stem = {}  # stem → file, 优先 .gz
    for f in _all_csvs:
        stem = f.stem.lower().replace('.csv', '')  # .csv.gz → .csv → stem
        key = (str(f.parent), stem)
        if key not in _csv_by_stem or f.suffix.lower() == '.gz':
            _csv_by_stem[key] = f
    csv_files = list(_csv_by_stem.values())
    
    # 分类文件：大表用分桶，小表用普通转换
    bucket_tables_config = BUCKET_TABLES.get(database, {})
    bucket_files = []
    normal_files = []
    
    # 计算总大小用于预估时间
    total_size_mb = 0
    for csv_file in csv_files:
        stem = csv_file.stem.lower().replace('.csv', '')
        file_size = csv_file.stat().st_size / (1024 * 1024)
        total_size_mb += file_size
        if stem in bucket_tables_config:
            bucket_files.append((csv_file, bucket_tables_config[stem]))
        else:
            normal_files.append(csv_file)
    
    success = 0
    failed = 0
    total = len(normal_files) + len(bucket_files)
    current = 0
    processed_size_mb = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty()
    details = st.container()
    
    # 转换速度跟踪
    start_time = time.time()
    
    def update_eta(processed_mb: float, elapsed_seconds: float):
        """更新预估剩余时间"""
        if elapsed_seconds > 0 and processed_mb > 0:
            speed_mb_per_sec = processed_mb / elapsed_seconds
            remaining_mb = total_size_mb - processed_mb
            if speed_mb_per_sec > 0:
                eta_seconds = remaining_mb / speed_mb_per_sec
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}min"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                eta_text.markdown(f"⏱️ **Speed**: {speed_mb_per_sec:.1f} MB/s | **ETA**: {eta_str} | **Total**: {total_size_mb:.0f} MB")
    
    # 创建 DuckDB 转换器（根据可用内存自动配置）
    try:
        from easyicu.memory_manager import get_available_memory_mb
        _avail_gb = get_available_memory_mb() / 1024
        # 使用可用内存的 60%，最低 2GB，最高 24GB
        _mem_limit_gb = max(2.0, min(_avail_gb * 0.6, 24.0))
    except Exception:
        _mem_limit_gb = 8.0  # 保守默认值
    converter = DuckDBConverter(
        data_path=str(source_path),
        memory_limit_gb=_mem_limit_gb,
        verbose=False
    )
    
    # 1. 转换普通表
    for csv_file in normal_files:
        current += 1
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        try:
            rel_path = csv_file.relative_to(source_path)
            # 小写化文件名以确保下游一致性（MIMIC-III 使用大写 ADMISSIONS.csv.gz）
            parquet_name = rel_path.stem.replace('.csv', '').lower() + '.parquet'
            parquet_file = target_path / rel_path.parent / parquet_name
            
            if parquet_file.exists() and not overwrite:
                with details:
                    st.caption(f"⏭️ {csv_file.name} (exists)")
                processed_size_mb += file_size_mb
                progress_bar.progress(current / total)
                update_eta(processed_size_mb, time.time() - start_time)
                continue
            
            parquet_file.parent.mkdir(parents=True, exist_ok=True)
            
            status_text.markdown(f"**Converting**: `{csv_file.name}` ({file_size_mb:.1f}MB) [{current}/{total}]")
            
            result = converter.convert_file(csv_file)
            
            # DuckDBConverter 使用原始文件名（可能大写），重命名为小写
            if result['status'] == 'success' and result.get('output_path'):
                _out = Path(result['output_path'])
                if _out.exists() and _out.name != parquet_name:
                    _out.rename(_out.parent / parquet_name)
            
            processed_size_mb += file_size_mb
            
            if result['status'] == 'success':
                success += 1
                with details:
                    st.caption(f"✅ {csv_file.name}: {result['row_count']:,} rows")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {csv_file.name}: {result.get('error', 'unknown')[:40]}")
            
            gc.collect()
            update_eta(processed_size_mb, time.time() - start_time)
            
        except Exception as e:
            failed += 1
            processed_size_mb += file_size_mb
            with details:
                st.caption(f"❌ {csv_file.name}: {str(e)[:40]}")
        
        progress_bar.progress(current / total)
    
    # 2. 分桶转换大表
    for csv_file, (partition_col, num_buckets) in bucket_files:
        current += 1
        stem = csv_file.stem.lower().replace('.csv', '')
        bucket_dir = target_path / f"{stem}_bucket"
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        
        try:
            # 检查分桶是否已完整完成（通过 _COMPLETE 标记文件）
            sentinel = bucket_dir / '_COMPLETE'
            if bucket_dir.exists() and sentinel.exists() and not overwrite:
                with details:
                    st.caption(f"⏭️ {csv_file.name} (bucket complete)")
                processed_size_mb += file_size_mb
                progress_bar.progress(current / total)
                update_eta(processed_size_mb, time.time() - start_time)
                continue
            # 如果目录存在但无 _COMPLETE 标记，说明上次转换不完整，清理后重新转换
            if bucket_dir.exists() and not sentinel.exists():
                import shutil
                with details:
                    st.caption(f"🔄 {csv_file.name} (incomplete, re-converting...)")
                shutil.rmtree(bucket_dir)
            
            status_text.markdown(f"**Bucketing**: `{csv_file.name}` ({file_size_mb:.1f}MB) → {num_buckets} buckets [{current}/{total}]")
            
            # 使用优化配置：跳过排序可加速2-3倍
            config = BucketConfig(
                num_buckets=num_buckets,
                partition_col=partition_col,
                memory_limit=f'{_mem_limit_gb:.0f}GB',
                threads=0,  # 自动检测CPU核心数
                row_group_size=1_000_000,
                compression='zstd',
                skip_sorting=True  # 跳过排序，大幅加速
            )
            result = convert_to_buckets(
                source_path=csv_file,
                output_dir=bucket_dir,
                config=config,
                overwrite=overwrite
            )
            
            processed_size_mb += file_size_mb
            
            if result.success:
                success += 1
                with details:
                    st.caption(f"✅ {csv_file.name} → {result.num_buckets} buckets, {result.total_rows:,} rows")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {csv_file.name}: {result.error[:40] if result.error else 'unknown'}")
            
            gc.collect()
            update_eta(processed_size_mb, time.time() - start_time)
            
        except Exception as e:
            failed += 1
            processed_size_mb += file_size_mb
            with details:
                st.caption(f"❌ {csv_file.name}: {str(e)[:40]}")
        
        progress_bar.progress(current / total)
    
    # 完成后显示总耗时
    total_time = time.time() - start_time
    if total_time < 60:
        time_str = f"{total_time:.1f}s"
    elif total_time < 3600:
        time_str = f"{total_time/60:.1f}min"
    else:
        time_str = f"{total_time/3600:.1f}h"
    
    progress_bar.progress(1.0)
    status_text.empty()
    eta_text.markdown(f"✅ **Completed** in {time_str} | **Avg Speed**: {total_size_mb/total_time:.1f} MB/s")
    
    return success, failed


def _convert_hirid_data(source_dir: str, target_dir: str, overwrite: bool = False) -> tuple:
    """HiRID 一站式转换：解压 → 重命名 parquet shards → CSV→parquet → 分桶。
    
    完整处理流程（用户只需点一次）：
    1. 解压 raw_stage/*.tar.gz + reference_data.tar.gz
    2. 重命名 observation_tables/parquet/part-N → observations/N.parquet
    3. general_table.csv → general.parquet（以及其他小 CSV 文件）
    4. observations/ → observations_bucket/ (variableid, 100桶)
    5. pharma/ → pharma_bucket/ (pharmaid, 50桶)
    """
    import time
    
    lang = st.session_state.get('language', 'en')
    
    try:
        from easyicu.bucket_converter import (
            convert_parquet_directory_to_buckets
        )
        from easyicu.duckdb_converter import DuckDBConverter
    except ImportError as e:
        st.error(f"Converter not available: {e}")
        return 0, 0
    
    source_path = Path(source_dir)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    details = st.container()
    
    success = 0
    failed = 0
    start_time = time.time()
    
    # ============================================================
    # 阶段 1: 解压 tar.gz 文件
    # ============================================================
    status_text.markdown("**Phase 1/4**: Extracting archives..." if lang == 'en' else "**阶段 1/4**: 解压归档文件...")
    
    # 解压 reference_data.tar.gz → general_table.csv
    reference_tar = source_path / 'reference_data.tar.gz'
    general_csv = source_path / 'general_table.csv'
    if reference_tar.exists() and not general_csv.exists():
        try:
            import tarfile
            with st.spinner("Extracting reference_data.tar.gz..." if lang == 'en' else "正在解压 reference_data.tar.gz..."):
                with tarfile.open(reference_tar, 'r:gz') as tar:
                    tar.extractall(path=source_path)
            with details:
                st.caption("✅ reference_data.tar.gz → general_table.csv")
        except Exception as e:
            with details:
                st.caption(f"❌ reference_data.tar.gz: {e}")
            failed += 1
    
    # 解压 observation_tables + pharma_records 
    raw_stage = source_path / 'raw_stage'
    archives = [
        ('observation_tables_parquet.tar.gz', 'observation_tables', 'observations'),
        ('pharma_records_parquet.tar.gz', 'pharma_records', 'pharma'),
    ]
    for archive_name, extract_dir, target_dir_name in archives:
        archive_path = raw_stage / archive_name if raw_stage.exists() else source_path / archive_name
        if not archive_path.exists():
            continue
        # 如果已有重命名后的 shard 目录（如 observations/），跳过
        shard_dir = source_path / target_dir_name
        if shard_dir.is_dir() and list(shard_dir.glob('[0-9]*.parquet')):
            with details:
                st.caption(f"⏭️ {archive_name} (already extracted → {target_dir_name}/)")
            continue
        # 如果解压目录已存在，跳过解压
        extracted_dir = source_path / extract_dir
        if not (extracted_dir.is_dir() and any(extracted_dir.rglob('*.parquet'))):
            try:
                import tarfile
                with st.spinner(f"Extracting {archive_name}..." if lang == 'en' else f"正在解压 {archive_name}..."):
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall(path=source_path)
                with details:
                    st.caption(f"✅ {archive_name}")
            except Exception as e:
                with details:
                    st.caption(f"❌ {archive_name}: {e}")
                failed += 1
                continue
        
        # 重命名 part-N.parquet → N.parquet (R ricu 格式)
        parquet_subdir = extracted_dir / 'parquet'
        if parquet_subdir.is_dir():
            src_files = sorted(parquet_subdir.glob('part-*.parquet'))
        elif extracted_dir.is_dir():
            src_files = sorted(extracted_dir.glob('part-*.parquet'))
        else:
            src_files = []
        
        if src_files and not (shard_dir.is_dir() and list(shard_dir.glob('[0-9]*.parquet'))):
            import shutil
            shard_dir.mkdir(parents=True, exist_ok=True)
            for f in src_files:
                try:
                    idx = int(f.stem.replace('part-', '')) + 1  # 1-based
                    dst = shard_dir / f'{idx}.parquet'
                    if not dst.exists():
                        shutil.copy2(f, dst)
                except (ValueError, OSError):
                    pass
            with details:
                st.caption(f"✅ Renamed {len(src_files)} shards → {target_dir_name}/")
    
    progress_bar.progress(0.2)
    
    # ============================================================
    # 阶段 2: CSV → Parquet (小表)
    # ============================================================
    status_text.markdown("**Phase 2/4**: Converting CSV → Parquet..." if lang == 'en' else "**阶段 2/4**: CSV 转换为 Parquet...")
    
    csv_files = list(source_path.glob('*.csv'))
    converter = DuckDBConverter(data_path=str(source_path), memory_limit_gb=8.0, verbose=False)
    
    for csv_file in csv_files:
        parquet_name = csv_file.stem + '.parquet'
        parquet_file = source_path / parquet_name
        if parquet_file.exists() and not overwrite:
            continue
        try:
            result = converter.convert_file(csv_file)
            if result['status'] == 'success':
                success += 1
                with details:
                    st.caption(f"✅ {csv_file.name} → {parquet_name} ({result['row_count']:,} rows)")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {csv_file.name}: {result.get('error', 'unknown')[:60]}")
        except Exception as e:
            failed += 1
            with details:
                st.caption(f"❌ {csv_file.name}: {str(e)[:60]}")
    
    progress_bar.progress(0.4)
    
    # ============================================================
    # 阶段 3+4: Parquet shard 目录 → 分桶
    # ============================================================
    obs_dir = None
    pharma_dir = None
    
    obs_candidates = [
        source_path / 'observations',
        source_path / 'observations' / 'parquet',
        source_path / 'observation_tables',
        source_path / 'observation_tables' / 'parquet',
    ]
    for cand in obs_candidates:
        if cand.exists() and cand.is_dir():
            if list(cand.glob('*.parquet')):
                obs_dir = cand
                break
    
    # 可能的 pharma 目录位置
    pharma_candidates = [
        source_path / 'pharma',
        source_path / 'pharma' / 'parquet',
        source_path / 'pharma_records',
        source_path / 'pharma_records' / 'parquet',
    ]
    for cand in pharma_candidates:
        if cand.exists() and cand.is_dir():
            if list(cand.glob('*.parquet')):
                pharma_dir = cand
                break
    
    # 检查是否有数据可分桶
    if not obs_dir and not pharma_dir:
        st.error("❌ No observation/pharma parquet directories found after extraction. "
                 "Please check your HiRID data." if lang == 'en' else
                 "❌ 解压后未找到 observation/pharma parquet 目录，请检查 HiRID 数据。")
        return success, max(1, failed)
    
    obs_bucket_dir = source_path / 'observations_bucket'
    pharma_bucket_dir = source_path / 'pharma_bucket'
    
    tasks = []
    if obs_dir:
        tasks.append(('observations', obs_dir, obs_bucket_dir, 'variableid', 100))
    if pharma_dir:
        tasks.append(('pharma', pharma_dir, pharma_bucket_dir, 'pharmaid', 50))
    
    total = len(tasks)
    
    for idx, (name, src_dir, bucket_dir, partition_col, num_buckets) in enumerate(tasks):
        _pct = 0.6 + 0.4 * idx / max(len(tasks), 1)
        progress_bar.progress(_pct)
        status_text.markdown(f"**Phase 3/4**: Bucketing `{name}` → {num_buckets} buckets..." if lang == 'en' else f"**阶段 3/4**: 分桶 `{name}` → {num_buckets} 个桶...")
        
        try:
            if bucket_dir.exists() and list(bucket_dir.rglob('*.parquet')) and not overwrite:
                with details:
                    st.caption(f"⏭️ {name} (bucket exists, skipped)" if lang == 'en' else f"⏭️ {name} (分桶已存在，跳过)")
                success += 1
                continue
            
            result = convert_parquet_directory_to_buckets(
                source_dir=src_dir,
                output_dir=bucket_dir,
                partition_col=partition_col,
                num_buckets=num_buckets,
                overwrite=overwrite
            )
            
            if result.success:
                success += 1
                with details:
                    st.caption(f"✅ {name} → {result.num_buckets} buckets, {result.total_rows:,} rows")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {name}: {result.error[:60] if result.error else 'unknown'}")
        except Exception as e:
            failed += 1
            with details:
                st.caption(f"❌ {name}: {str(e)[:60]}")
    
    total_time = time.time() - start_time
    progress_bar.progress(1.0)
    status_text.empty()
    
    if success > 0:
        st.success(f"✅ HiRID setup completed in {total_time:.1f}s ({success} steps)" if lang == 'en' else f"✅ HiRID 设置完成，耗时 {total_time:.1f}秒 ({success} 个步骤)")
    
    return success, failed


def _generate_cohort_prefix() -> str:
    """根据队列筛选条件生成文件名前缀。
    
    Returns:
        筛选条件前缀字符串，如 "age18-80_firstICU_los24h"，无筛选则返回空字符串
    """
    if not st.session_state.get('cohort_enabled', False):
        return ""
    
    cf = st.session_state.get('cohort_filter', {})
    parts = []
    
    # 年龄
    age_min = cf.get('age_min')
    age_max = cf.get('age_max')
    if age_min is not None or age_max is not None:
        age_str = f"age{int(age_min) if age_min else 0}-{int(age_max) if age_max else 'inf'}"
        parts.append(age_str)
    
    # 首次入ICU
    first_icu = cf.get('first_icu_stay')
    if first_icu is True:
        parts.append("firstICU")
    elif first_icu is False:
        parts.append("readmit")
    
    # 住院时长
    los_min = cf.get('los_min')
    if los_min is not None and los_min > 0:
        parts.append(f"los{int(los_min)}h")
    
    # 性别
    gender = cf.get('gender')
    if gender is not None:
        parts.append(f"sex{gender}")
    
    # 存活状态
    survived = cf.get('survived')
    if survived is True:
        parts.append("survived")
    elif survived is False:
        parts.append("deceased")
    
    # Sepsis
    has_sepsis = cf.get('has_sepsis')
    if has_sepsis is True:
        parts.append("sepsis")
    elif has_sepsis is False:
        parts.append("noSepsis")
    
    return "_".join(parts)


# 🔧 FIX Bug 62: Worker functions moved to subprocess_workers.py (separate from app.py)
# On Windows, multiprocessing.Process(start_method='spawn') creates a fresh Python process
# that imports the module containing the target function. If the target is in app.py,
# importing app.py triggers `import streamlit` + `st.set_page_config()` which crashes
# in non-Streamlit subprocesses. Moving workers to a streamlit-free module fixes this.
from easyicu.webapp.subprocess_workers import (
    _subprocess_load_module,
    _subprocess_load_and_export_module,
    _subprocess_load_special,
)


def execute_sidebar_export():
    """执行侧边栏触发的数据导出（直接导出到本地目录，带进度条）。
    
    🔧 进度显示在主内容区的专用容器中。
    🔧 支持三种模式：
        1. 模拟数据模式 (use_mock_data=True)
        2. 真实数据模式 (有有效的 data_path)
        3. 可视化导入模式 (loaded_data_origin='exported_files') - 直接导出已加载的数据
    """
    from datetime import datetime
    
    lang = st.session_state.get('language', 'en')
    export_path = st.session_state.get('export_path', '')
    export_format = st.session_state.get('export_format', 'Parquet').lower()
    selected_concepts = st.session_state.get('selected_concepts', [])
    use_mock = st.session_state.get('use_mock_data', False)
    
    # 🔧 FIX (2026-02-03): 检测是否是从可视化模式导入数据的场景
    loaded_concepts = st.session_state.get('loaded_concepts', {})
    has_loaded_data = len(loaded_concepts) > 0
    loaded_data_origin = st.session_state.get('loaded_data_origin', 'none')
    
    # 只有真正从“已导出文件”加载的数据，才视为 viz import mode。
    # Preview / quick_load / demo_viz 都不应该污染 Confirm Export 的真实提取流程。
    is_viz_import_mode = has_loaded_data and loaded_data_origin == 'exported_files'
    use_loaded_data_export = is_viz_import_mode
    
    # 🔧 FIX (2026-02-03): 在可视化导入模式下，如果 selected_concepts 为空，
    # 使用 loaded_concepts 的 keys 作为要导出的概念
    if is_viz_import_mode and not selected_concepts:
        selected_concepts = list(loaded_concepts.keys())
        st.session_state.selected_concepts = selected_concepts
        print(f"[DEBUG] Auto-set selected_concepts from loaded_concepts: {len(selected_concepts)} concepts")
    
    if not export_path or not Path(export_path).exists():
        err_msg = "❌ Please set a valid export path first" if lang == 'en' else "❌ 请先设置有效的导出路径"
        st.error(err_msg)
        return
    
    if not selected_concepts:
        err_msg = "❌ Please select features to export first" if lang == 'en' else "❌ 请先选择要导出的特征"
        st.error(err_msg)
        return
    
    try:
        export_title = "📤 Export Progress" if lang == 'en' else "📤 导出进度"
        st.markdown(f"### {export_title}")
        if loaded_data_origin == 'preview':
            preview_reextract_msg = (
                "ℹ️ Preview sample detected. Export will re-extract data from the source database instead of exporting the preview sample."
                if lang == 'en' else
                "ℹ️ 检测到当前是 Preview 样本。导出时会重新从源数据库提取数据，而不是直接导出这批 Preview 样本。"
            )
            st.markdown(f'<div class="compact-inline-notice info">{preview_reextract_msg}</div>', unsafe_allow_html=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 直接使用用户设置的导出路径（已包含数据库子目录）
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        module_times = {}
        all_exported_patient_ids = set()
        total_concepts = len(selected_concepts)
        
        # 创建进度条和状态显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 🔧 添加取消按钮
        import time as time_module
        cancel_placeholder = st.empty()
        cancel_key = f"cancel_export_{int(time_module.time() * 1000)}"
        
        # 初始化取消状态
        if '_export_cancelled' not in st.session_state:
            st.session_state._export_cancelled = False
        
        def check_cancelled():
            """检查是否已取消导出"""
            return st.session_state.get('_export_cancelled', False)
        
        # ============================================================
        # 🔧 步骤0：检测已存在的文件（适用于模拟数据和真实数据）
        # ============================================================
        # 构建 concept -> group_key 的映射
        concept_to_group = {}
        for group_key in CONCEPT_GROUPS_INTERNAL.keys():
            for c in CONCEPT_GROUPS_INTERNAL[group_key]:
                if c not in concept_to_group:
                    concept_to_group[c] = group_key
        
        # 找出用户选择的每个模块
        selected_modules = {}  # group_key -> [concepts]
        for c in selected_concepts:
            group_key = concept_to_group.get(c, 'other')
            if group_key not in selected_modules:
                selected_modules[group_key] = []
            selected_modules[group_key].append(c)
        
        # 检测哪些模块的文件已存在
        # 🔧 FIX (2026-02-05): 使用模块名开头匹配，cohort条件在后缀
        existing_modules = {}  # group_key -> file_path
        cohort_suffix = _generate_cohort_prefix()
        
        for group_key, group_concepts in selected_modules.items():
            # 🔧 按模块名开头查找已存在的文件
            search_prefix = f"{group_key}_"
            
            # 检查是否有匹配该模块的文件存在
            for ext in ['.parquet', '.csv', '.xlsx']:
                matching_files = list(export_dir.glob(f"{search_prefix}*{ext}"))
                if matching_files:
                    # 找到匹配的文件
                    existing_modules[group_key] = matching_files[0]
                    break
        
        # 如果有已存在的模块，显示让用户选择
        # 🔧 FIX (2026-02-03): 在 viz_import_mode 下自动覆盖，跳过对话框
        if existing_modules and not is_viz_import_mode:
            # 检查用户是否已做出所有决定
            skipped_modules = st.session_state.get('_skipped_modules', set())
            overwrite_modules = st.session_state.get('_overwrite_modules', set())
            
            # 🔧 DEBUG: 打印状态以便调试
            print(f"[DEBUG] existing_modules: {list(existing_modules.keys())}")
            print(f"[DEBUG] skipped_modules: {skipped_modules}")
            print(f"[DEBUG] overwrite_modules: {overwrite_modules}")
            
            # 找出尚未决定的模块
            pending_modules = [m for m in existing_modules.keys() 
                               if m not in skipped_modules and m not in overwrite_modules]
            
            print(f"[DEBUG] pending_modules: {pending_modules}")
            
            if pending_modules:
                # 🔧 FIX: 显示冲突时清除 _exporting_in_progress，避免显示 "Export in Progress"
                st.session_state['_exporting_in_progress'] = False
                
                # 显示所有冲突模块
                conflict_title = "⚠️ Existing Files Detected" if lang == 'en' else "⚠️ 检测到已存在的文件"
                st.warning(conflict_title)
                
                # 🔧 简化：只显示文件列表
                file_list_html = "<ul style='margin: 10px 0; padding-left: 20px;'>"
                for group_key in pending_modules:
                    file_path = existing_modules[group_key]
                    file_list_html += f"<li style='margin: 5px 0;'><b>{group_key}</b>: <code>{file_path.name}</code></li>"
                file_list_html += "</ul>"
                st.markdown(file_list_html, unsafe_allow_html=True)
                
                # 🔧 使用醒目的大按钮
                st.markdown("---")
                st.markdown("<p style='font-size: 1.1rem; font-weight: bold; margin-bottom: 15px;'>How do you want to handle these files?</p>" if lang == 'en' else "<p style='font-size: 1.1rem; font-weight: bold; margin-bottom: 15px;'>请选择如何处理这些文件：</p>", unsafe_allow_html=True)
                
                # 🔧 FIX: 使用 on_click 回调而不是 if st.button，避免页面跳转
                def on_overwrite_all():
                    """覆盖全部的回调函数"""
                    # 将所有 existing_modules 添加到 overwrite 列表
                    all_modules = set(st.session_state.get('_existing_modules_list', []))
                    st.session_state['_overwrite_modules'] = all_modules
                    st.session_state['_exporting_in_progress'] = True
                    # 🔧 FIX: 设置 trigger_export 并让它rerun来继续执行
                    st.session_state.trigger_export = True
                
                def on_skip_all():
                    """跳过全部的回调函数"""
                    all_modules = set(st.session_state.get('_existing_modules_list', []))
                    st.session_state['_skipped_modules'] = all_modules
                    st.session_state['_exporting_in_progress'] = True
                    # 🔧 FIX: 设置 trigger_export 并让它rerun来继续执行
                    st.session_state.trigger_export = True
                
                # 🔧 保存 pending_modules 到 session_state 让回调能访问
                st.session_state['_existing_modules_list'] = list(existing_modules.keys())
                
                col_all_overwrite, col_all_skip = st.columns(2)
                with col_all_overwrite:
                    all_overwrite_btn = "🔄 OVERWRITE ALL" if lang == 'en' else "🔄 全部覆盖"
                    st.markdown("<style>.stButton button[kind='primary'] { font-size: 1.2rem !important; padding: 15px !important; }</style>", unsafe_allow_html=True)
                    st.button(all_overwrite_btn, key="file_overwrite_all", type="primary", 
                             use_container_width=True, on_click=on_overwrite_all)
                with col_all_skip:
                    all_skip_btn = "⏭️ SKIP ALL" if lang == 'en' else "⏭️ 全部跳过"
                    st.button(all_skip_btn, key="file_skip_all", use_container_width=True,
                             on_click=on_skip_all)
                
                # 🔧 FIX: 重新检查用户是否已做出决定（回调可能已更新 session_state）
                overwrite_modules = st.session_state.get('_overwrite_modules', set())
                skipped_modules = st.session_state.get('_skipped_modules', set())
                pending_modules = [m for m in existing_modules.keys() 
                                   if m not in skipped_modules and m not in overwrite_modules]
                
                if pending_modules:
                    # 用户尚未做出决定，暂停导出
                    return
        
        # 根据用户选择，确定要跳过的模块
        skipped_modules = st.session_state.get('_skipped_modules', set())
        concepts_to_skip = set()
        for group_key in skipped_modules:
            if group_key in selected_modules:
                for c in selected_modules[group_key]:
                    concepts_to_skip.add(c)
        
        # 过滤掉将跳过的概念
        concepts_to_export = [c for c in selected_concepts if c not in concepts_to_skip]
        
        if not concepts_to_export:
            if concepts_to_skip:
                skip_msg = f"⏭️ All selected modules already exist, nothing to export" if lang == 'en' else "⏭️ 所有选中的模块都已存在，无需导出"
                st.info(skip_msg)
            # 清理状态
            if '_skipped_modules' in st.session_state:
                del st.session_state['_skipped_modules']
            if '_overwrite_modules' in st.session_state:
                del st.session_state['_overwrite_modules']
            return
        
        # 显示跳过信息
        if concepts_to_skip:
            skip_count = len(concepts_to_skip)
            load_count = len(concepts_to_export)
            skip_info = f"⏭️ Skipping {skip_count} concepts (files exist), exporting {load_count} concepts" if lang == 'en' else f"⏭️ 跳过 {skip_count} 个概念（文件已存在），导出 {load_count} 个概念"
            st.info(skip_info)
        
        # 🔧 FIX: 初始化变量，避免 demo 模式下引用未定义变量
        unsupported_concepts = []
        empty_concepts = []
        failed_concepts = []
        
        if use_mock or use_loaded_data_export:
            # 生成模拟数据并导出
            if use_mock:
                gen_msg = "**Generating mock data...**" if lang == 'en' else "**正在生成模拟数据...**"
                status_text.markdown(gen_msg)
                # 🔧 使用 get_mock_params_with_cohort 获取完整参数（包含最新的 cohort_filter）
                params = get_mock_params_with_cohort()
                all_mock_data, patient_ids = generate_mock_data(**params)
                
                # 保存患者ID列表（用于其他功能）
                st.session_state.patient_ids = patient_ids
                
                # 🔧 根据要导出的 concepts 过滤数据（排除跳过的）
                data = {}
                for concept in concepts_to_export:
                    if concept in all_mock_data:
                        data[concept] = all_mock_data[concept]
                
                # 显示加载情况
                loaded_count = len(data)
                if loaded_count < len(concepts_to_export):
                    missing = [c for c in concepts_to_export if c not in all_mock_data]
                    skip_msg = f"⚠️ {len(missing)} concepts not in mock data: {', '.join(missing[:5])}" if lang == 'en' else f"⚠️ 模拟数据中不存在 {len(missing)} 个概念: {', '.join(missing[:5])}"
                    st.warning(skip_msg)
            else:
                status_text.markdown("**Using loaded visualization data...**" if lang == 'en' else "**正在使用已加载的可视化数据...**")
                data = {concept: loaded_concepts[concept] for concept in concepts_to_export if concept in loaded_concepts}
                if not data:
                    err_msg = "❌ No loaded visualization data matches the selected features" if lang == 'en' else "❌ 当前已加载的可视化数据中没有匹配所选特征"
                    st.error(err_msg)
                    st.session_state['_exporting_in_progress'] = False
                    return
            
            progress_bar.progress(0.3)
        else:
            # 加载真实数据并导出（批量并行加载）
            from easyicu import load_concepts
            import os
            
            # 🔧 FIX (2026-02-15): 每次导出前清除 easyicu 缓存，防止上次导出的患者数据泄漏
            # 问题场景：用户先导出100患者到dir1，再导出"全部数据"到dir2
            # 如果不清缓存，_concept_data_cache 中的旧数据可能被混入新导出
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
            
            # 🔧 FIX: 检查 data_path 是否有效（可视化模式导入数据后可能无效）
            data_path_str = st.session_state.get('data_path', '')
            if not data_path_str or not Path(data_path_str).exists():
                err_msg = "❌ Data path is not set or invalid. Please go back to Tutorial tab and configure a valid database path first." if lang == 'en' else "❌ 数据路径未设置或无效。请返回Tutorial标签页先配置有效的数据库路径。"
                st.error(err_msg)
                st.session_state['_exporting_in_progress'] = False
                return
            
            # 批量并行加载所有特征
            patient_limit_display = st.session_state.get('patient_limit', 100)
            patient_info = f"({patient_limit_display} patients)" if patient_limit_display else "(all patients)"
            patient_info_cn = f"（{patient_limit_display}患者）" if patient_limit_display else "（全部患者）"
            batch_msg = f"**Loading concepts {patient_info}...**" if lang == 'en' else f"**正在加载概念 {patient_info_cn}...**"
            status_text.markdown(batch_msg)
            
            # 🚀 性能优化：参照 extract_baseline_features.py 的配置
            patient_limit = st.session_state.get('patient_limit', 1000)  # 🔧 FIX: 默认1000患者
            
            patient_ids_filter = None
            id_col = 'stay_id'
            data_path = Path(data_path_str)
            database = st.session_state.get('database', 'miiv')
            id_col_map = {'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid', 'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID'}
            id_col = id_col_map.get(database, 'stay_id')
            
            # 👥 先从数据库选patient_limit个患者，再对这些患者做人群筛选
            try:
                # Step 1: 先选patient_limit个患者作为候选集
                candidate_ids = None
                if patient_limit and patient_limit > 0:
                    try:
                        for f in _get_patient_id_table_files(database):
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_ids = icustays_df[id_col].unique().tolist()
                                    candidate_ids = _sample_patient_ids_random(all_ids, patient_limit)
                                    break
                    except Exception:
                        pass
                
                # Step 2: 在候选集上应用人群筛选
                database_for_cohort = st.session_state.get('database', 'miiv')
                cohort_result = apply_cohort_filter(data_path_str, database_for_cohort, candidate_ids=candidate_ids)
                if cohort_result is not None:
                    cohort_id_col = cohort_result['id_col']
                    filtered_ids = cohort_result['filtered_ids']
                    id_col = cohort_id_col
                    patient_ids_filter = {id_col: filtered_ids}
                    # Save cohort stats for completion message
                    n_before = len(candidate_ids) if candidate_ids else 0
                    n_after = len(filtered_ids)
                    st.session_state['_cohort_stats'] = {
                        'before': n_before, 'after': n_after, 'excluded': n_before - n_after,
                        'filter_details': cohort_result.get('filter_details', []),
                    }
                    # 🚫 Zero patients after cohort filter — abort export
                    if n_after == 0:
                        lang = st.session_state.get('language', 'en')
                        details_parts = []
                        for fd in cohort_result.get('filter_details', []):
                            label = fd[0] if lang == 'en' else fd[1]
                            details_parts.append(f"{label} (excluded {fd[2]})")
                        details_str = ", ".join(details_parts) if details_parts else ""
                        if lang == 'en':
                            msg = f"No patients meet the cohort criteria. {n_before} candidates were all excluded."
                            if details_str:
                                msg += f" Filters applied: {details_str}."
                            msg += " Please adjust your cohort filters in Step 2 and try again."
                        else:
                            msg = f"没有患者满足队列筛选条件。{n_before} 名候选患者全部被排除。"
                            if details_str:
                                msg += f" 已应用的筛选条件: {details_str}。"
                            msg += " 请在步骤2中调整筛选条件后重试。"
                        st.error(msg)
                        return
                elif candidate_ids is not None:
                    patient_ids_filter = {id_col: candidate_ids}
                    st.session_state['_cohort_stats'] = None
                else:
                    # 全量提取(patient_limit=0): 加载所有ID以支持分批
                    st.session_state['_cohort_stats'] = None
                    try:
                        for f in _get_patient_id_table_files(database):
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_ids = sorted(icustays_df[id_col].unique().tolist())
                                    patient_ids_filter = {id_col: all_ids}
                                    break
                    except Exception:
                        pass
            except Exception as _cohort_err:
                print(f"[COHORT] Error in execute_sidebar_export: {_cohort_err}")
                # Fallback: just apply patient_limit without cohort filter
                if patient_limit and patient_limit > 0:
                    try:
                        for f in _get_patient_id_table_files(database):
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_ids = icustays_df[id_col].unique().tolist()
                                    sample_ids = _sample_patient_ids_random(all_ids, patient_limit)
                                    patient_ids_filter = {id_col: sample_ids}
                                    break
                    except Exception:
                        pass
            
            num_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
            parallel_workers, parallel_backend = get_optimal_parallel_config(num_patients, task_type='export')
            
            # 显示系统资源信息（包含性能层级）
            resources = get_system_resources()
            perf_tier = resources.get('performance_tier', 'unknown')
            # 🔧 FIX: 显示实际使用的 concept_workers=1（串行），而非 recommended_workers
            # 实际加载时 concept_workers=1 避免死锁，不应显示 64 workers 误导用户
            actual_workers = 1  # 与 load_kwargs['concept_workers'] 一致
            tier_emoji = {
                'high-performance': '🚀',
                'server': '💻',
                'workstation': '🖥️',
                'standard': '💻',
                'limited': '⚠️'
            }.get(perf_tier, '💻')
            
            n_patients_display = num_patients or 'all'
            if lang == 'en':
                perf_msg = f"{tier_emoji} System: {resources['cpu_count']} cores, {resources['total_memory_gb']}GB RAM → Loading {n_patients_display} patients (sequential by module)"
            else:
                perf_msg = f"{tier_emoji} 系统: {resources['cpu_count']} 核心, {resources['total_memory_gb']}GB 内存 → 加载 {n_patients_display} 患者（按模块顺序加载）"
            st.info(perf_msg)
            
            try:
                # 📝 批量加载所有概念（触发宽表批量加载优化）
                data = {}
                failed_concepts = []
                empty_concepts = []  # 🆕 跟踪返回空结果的概念
                
                # 🚀 优化：先过滤掉当前数据库不支持的概念，避免批量加载失败
                from easyicu.concept import load_dictionary
                cd = load_dictionary(include_sofa2=True)  # 🔧 FIX: 包含 SOFA2 概念字典
                database = st.session_state.get('database', 'eicu')
                valid_concepts = []
                unsupported_concepts = []
                special_concepts_to_load = []  # 🆕 特殊概念（AKI, circ_failure等）
                
                # 🔧 FIX (2026-02-09): 递归检查概念是否有数据路径
                # 解决 has_callback=True 但无数据源的概念被误判为有效的问题
                def _has_any_source_recursive(concept_name, db, cd_dict, visited=None):
                    """递归检查概念或其子概念是否在目标数据库有数据源"""
                    if visited is None:
                        visited = set()
                    if concept_name in visited:
                        return False
                    visited.add(concept_name)
                    cdef = cd_dict.get(concept_name)
                    if not cdef:
                        return False
                    if cdef.sources.get(db):
                        return True
                    if cdef.sub_concepts:
                        return any(_has_any_source_recursive(sc, db, cd_dict, visited) for sc in cdef.sub_concepts)
                    return False
                
                # 🔧 使用 concepts_to_export 而不是 selected_concepts（跳过已存在模块的概念）
                for c in concepts_to_export:
                    # 🆕 先检查是否是特殊概念
                    if c in SPECIAL_CONCEPTS:
                        special_concepts_to_load.append(c)
                        continue
                    
                    concept_def = cd.get(c)
                    if concept_def:
                        # 🔧 FIX 2026-02-09: 使用递归检查替代简单的 has_callback 判断
                        # 旧逻辑: has_sources or has_sub_concepts or has_callback → 很多概念有 callback 但无数据源
                        # 新逻辑: 递归检查概念树中是否有至少一个数据源
                        if _has_any_source_recursive(c, database, cd):
                            valid_concepts.append(c)
                        else:
                            unsupported_concepts.append(c)
                    else:
                        unsupported_concepts.append(c)
                
                # 🔧 FIX: unsupported_concepts 警告移到 failed_concepts 处统一显示，避免重复
                # 这里只记录，不立即显示
                pass  # unsupported_concepts will be merged with failed_concepts later
                
                if not valid_concepts and not special_concepts_to_load:
                    st.error("❌ 所选概念在当前数据库中都不可用")
                    return
                
                # � FIX(2026-02-09): 强制 concept_workers=1，避免多线程加载引起的死锁
                # 原因：多线程同时加载概念会导致 DuckDB 连接竞争、缓存失效、
                #       以及 SIC/MIMIC-III 等数据库的复杂回调函数中的线程安全问题
                # 模块级顺序加载已足够高效（每模块 1-5s），无需概念级并行
                smart_concept_workers = 1
                
                # 🚀 FIX(2026-02-09): 按模块分组加载，实时显示进度
                # 解决 HiRID/SICdb 加载 100+ 概念时用户无法看到进度的问题
                # 按模块分组：保留同组概念的批量优化（宽表、共享子概念缓存）
                module_concept_map = {}  # {module_key: [concepts]}
                concept_to_module = {}
                for mod_key, mod_concepts in CONCEPT_GROUPS_INTERNAL.items():
                    for c in mod_concepts:
                        concept_to_module[c] = mod_key
                
                for c in valid_concepts:
                    mod = concept_to_module.get(c, '_other')
                    if mod not in module_concept_map:
                        module_concept_map[mod] = []
                    module_concept_map[mod].append(c)
                
                # 🔧 模块加载优先级：快的模块先加载（给用户更快的反馈）
                MODULE_PRIORITY = [
                    'vitals', 'demographics', 'outcome',        # 快
                    'chemistry', 'hematology', 'blood_gas',     # 中等
                    'medications', 'ventilator', 'respiratory',  # 中等
                    'vasopressors', 'renal', 'neurological',    # 较慢
                    'other_scores', 'circulatory',               # callback
                    'sofa1_score', 'sofa2_score',                # 慢（SOFA计算）
                    'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared',  # 慢（依赖SOFA）
                    '_other',
                ]
                ordered_modules = []
                for mod in MODULE_PRIORITY:
                    if mod in module_concept_map:
                        ordered_modules.append(mod)
                # 添加遗漏的module
                for mod in module_concept_map:
                    if mod not in ordered_modules:
                        ordered_modules.append(mod)
                
                import time as _time_mod
                _export_start = _time_mod.time()
                
                progress_bar.progress(0.15)
                
                # ⚡ PERF: 启用跨模块缓存复用 — 共享子概念（MAP/GCS/fio2等）无需重复加载
                # ⚠️ 大量患者时禁用缓存，防止内存膨胀
                #   实测: AUMC 23K患者，缓存200MB但Python内存碎片导致RSS增长56GB
                try:
                    from easyicu.api import _get_global_loader
                    _loader = _get_global_loader(database=database, data_path=st.session_state.data_path,
                                                 use_sofa2=True)
                    _actual_n_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
                    # 只对小规模提取启用缓存（<5000患者），大规模提取禁用以控制RSS
                    if _actual_n_patients is not None and _actual_n_patients <= 5000:
                        _loader.concept_resolver._keep_cache_between_calls = True
                    else:
                        _loader.concept_resolver._keep_cache_between_calls = False
                except Exception:
                    _loader = None
                    _actual_n_patients = None
                
                # 🧠 内存安全: 碎片感知的分批策略
                # 逐模块加载时，每模块的 pandas/numpy 操作在 pymalloc arena 中产生碎片，
                # gc.collect() + malloc_trim() 无法回收（arena 中只要有1个存活对象就不归还OS）。
                # 19个模块依次执行，碎片累积导致 RSS 远超实际数据量（实测: 3GB 数据 → 22GB RSS）。
                #
                # 分批加载可限制每批的碎片量。
                # 流式导出 (2026-04-13): 每批加载 → 合并 → 追加写入 → 释放，
                # 内存始终受限于单批，无需严格上限。
                #   16GB → 10000, 32GB → 12000, 64GB → 24000, 服务器(1.5TB) → 不分批
                _auto_batch_size = None
                _n_load_patients = _actual_n_patients if _actual_n_patients is not None else (
                    len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
                )
                if _n_load_patients is not None and _n_load_patients > 5000:
                    try:
                        from easyicu.memory_manager import get_available_memory_mb
                        _avail_mb = get_available_memory_mb()
                        # 流式导出: 每批峰值 ~0.05-0.2 MB/patient, 无需严格上限
                        _frag_safe_max = max(10000, int(_avail_mb * 0.5))
                        if _n_load_patients > _frag_safe_max:
                            _auto_batch_size = _frag_safe_max
                            _n_batches = (_n_load_patients + _auto_batch_size - 1) // _auto_batch_size
                            if lang == 'en':
                                st.info(f"🧠 Loading {_n_load_patients} patients (batch={_auto_batch_size}, "
                                        f"{_n_batches} batches/module), available memory {_avail_mb:.0f}MB")
                            else:
                                st.info(f"🧠 加载 {_n_load_patients} 患者 (分批={_auto_batch_size}, "
                                        f"每模块{_n_batches}批), 可用内存 {_avail_mb:.0f}MB")
                        else:
                            if lang == 'en':
                                st.info(f"🧠 Loading {_n_load_patients} patients, "
                                        f"available memory {_avail_mb:.0f}MB, no batching needed.")
                            else:
                                st.info(f"🧠 加载 {_n_load_patients} 患者, "
                                        f"可用内存 {_avail_mb:.0f}MB, 无需分批。")
                    except Exception:
                        # 无法检测内存时，使用保守默认值
                        if _n_load_patients > 5000:
                            _auto_batch_size = 5000
                
                # 🚀 FIX: 全模块批量加载 + 改善进度提示
                # 测试结果：逐概念加载比批量慢 3-10x（etco2: 873s 逐概念 vs <100s 批量）
                # 原因：每次 load_concepts() 调用后清除 _table_cache，重新从磁盘读取
                # 方案：保持批量加载（最快），但在每个模块前显示预估时间和概念列表
                
                total_modules = len(ordered_modules)
                # 加上特殊概念模块（如果有）
                _total_steps = total_modules + (1 if special_concepts_to_load else 0)
                
                # ──────────────────────────────────────────────
                # 🚀 流式导出: 预构建导出相关变量
                # 每个模块加载完后立即合并+写文件+释放，不在内存中累积
                # ──────────────────────────────────────────────
                import time as time_module
                module_times = {}
                _module_load_times = {}  # 记录每模块的加载耗时
                
                # 预构建 concept -> group 映射（不依赖加载结果）
                _concept_to_group_pre = {}
                _group_priority = list(CONCEPT_GROUPS_INTERNAL.keys())
                for _gk in _group_priority:
                    _gc = CONCEPT_GROUPS_INTERNAL[_gk]
                    for _c in _gc:
                        if _c not in _concept_to_group_pre:
                            _concept_to_group_pre[_c] = _gk
                
                # 收集导出文件列表和患者ID
                all_exported_patient_ids = set()
                skipped_modules = st.session_state.get('_skipped_modules', set())
                overwrite_modules = st.session_state.get('_overwrite_modules', set())
                
                # cohort filter: 在 demographics/outcome 模块加载后计算排除列表
                _cohort_exclude_ids = set()
                _cohort_filter_computed = False
                
                def _process_result(result, concept_names):
                    """处理 load_concepts 返回结果"""
                    if isinstance(result, dict):
                        for cname, df in result.items():
                            if hasattr(df, 'to_pandas'):
                                df = df.to_pandas()
                            elif hasattr(df, 'dataframe'):
                                df = df.dataframe()
                            elif hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                                df = df.data
                            
                            if isinstance(df, pd.DataFrame) and len(df) > 0:
                                data[cname] = df
                            elif isinstance(df, pd.Series):
                                data[cname] = df.to_frame().reset_index()
                            else:
                                if cname not in empty_concepts:
                                    empty_concepts.append(cname)
                    elif isinstance(result, pd.DataFrame):
                        if len(result) > 0:
                            for c in concept_names:
                                if c in result.columns:
                                    data[c] = result
                    # 检查空结果
                    for c in concept_names:
                        if c not in data and c not in empty_concepts:
                            empty_concepts.append(c)
                
                # ──────────────────────────────────────────────
                # 🚀 流式导出: 定义模块合并+导出辅助函数
                # ──────────────────────────────────────────────
                def _export_module_to_disk(group_name, concept_dfs_dict, step_idx, total_steps):
                    """将一个模块的概念合并为宽表并写入磁盘。
                    
                    Args:
                        group_name: 模块key (e.g. 'vitals')
                        concept_dfs_dict: {concept_name: DataFrame} 该模块的数据
                        step_idx: 当前步骤索引（用于进度条）
                        total_steps: 总步骤数
                    Returns:
                        True if exported, False if skipped/empty
                    """
                    nonlocal all_exported_patient_ids
                    
                    if not concept_dfs_dict:
                        return False
                    
                    # 🔧 检查是否已取消
                    if check_cancelled():
                        return False
                    
                    _mod_export_start = time_module.time()
                    
                    # 显示导出进度
                    concept_list = list(concept_dfs_dict.keys())
                    concepts_str = ', '.join(concept_list[:5]) + (f'... +{len(concept_list)-5}' if len(concept_list) > 5 else '')
                    if lang == 'en':
                        _emsg = f"**Exporting**: `{group_name}` ({step_idx+1}/{total_steps}) | {concepts_str}"
                    else:
                        _emsg = f"**正在导出**: `{group_name}` ({step_idx+1}/{total_steps}) | {concepts_str}"
                    cancel_placeholder.markdown(_emsg)
                    
                    # ── 合并为宽表（完整保留原有逻辑） ──
                    id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                    time_candidates = ['time', 'charttime', 'starttime', 'start', 'endtime', 'itemtime', 'datetime', 'Offset', 'measuredat_minutes', 'measuredat', 'givenat', 'enteredentryat', 'intakeoutputoffset', 'observationoffset', 'nursingchartoffset', 'labresultoffset', 'respchartoffset']
                    unified_time_col = 'charttime'
                    
                    # 统一时间列名称
                    normalized_concept_dfs = {}
                    for cname, cdf in concept_dfs_dict.items():
                        cdf = cdf.copy()
                        if unified_time_col in cdf.columns:
                            other_time_cols = [tc for tc in time_candidates if tc in cdf.columns and tc != unified_time_col]
                            if other_time_cols:
                                cdf = cdf.drop(columns=other_time_cols)
                        else:
                            for tc in time_candidates:
                                if tc in cdf.columns:
                                    cdf = cdf.rename(columns={tc: unified_time_col})
                                    other_time_cols = [t for t in time_candidates if t in cdf.columns and t != unified_time_col]
                                    if other_time_cols:
                                        cdf = cdf.drop(columns=other_time_cols)
                                    break
                        normalized_concept_dfs[cname] = cdf
                    concept_dfs_dict = normalized_concept_dfs
                    
                    # 确定主键列
                    merge_cols = []
                    _id_col = None
                    _time_col = None
                    potential_id_cols = set()
                    potential_time_cols = set()
                    for cname, cdf in concept_dfs_dict.items():
                        for col in id_candidates:
                            if col in cdf.columns:
                                potential_id_cols.add(col)
                                break
                        for col in time_candidates:
                            if col in cdf.columns:
                                potential_time_cols.add(col)
                                break
                    for col in id_candidates:
                        if col in potential_id_cols:
                            _id_col = col
                            merge_cols.append(col)
                            break
                    for col in time_candidates:
                        if col in potential_time_cols:
                            _time_col = col
                            merge_cols.append(col)
                            break
                    
                    if not merge_cols:
                        all_dfs = []
                        for cname, cdf in concept_dfs_dict.items():
                            cdf = cdf.copy()
                            cdf['_concept'] = cname
                            all_dfs.append(cdf)
                        merged_df = pd.concat(all_dfs, ignore_index=True)
                    else:
                        all_concept_dfs = []
                        for concept_name, df in concept_dfs_dict.items():
                            if _id_col and _id_col not in df.columns:
                                continue
                            metadata_cols = ['valueuom', 'unit', 'units', 'category', 'type',
                                            'dur_var', 'entertime',
                                            'intakeoutputentryoffset']  # dur_var/entertime: WinTbl; intakeoutputentryoffset: eICU extra
                            cols_to_drop = [c for c in df.columns if c in metadata_cols]
                            if cols_to_drop:
                                df = df.drop(columns=cols_to_drop)
                            value_cols = [c for c in df.columns if c not in merge_cols]
                            df_to_add = df.copy()
                            if len(value_cols) == 1:
                                df_to_add = df_to_add.rename(columns={value_cols[0]: concept_name})
                            elif len(value_cols) > 1:
                                if concept_name in value_cols:
                                    keep_val_cols = [concept_name]
                                else:
                                    keep_val_cols = value_cols
                                cols_to_keep = merge_cols + keep_val_cols
                                df_to_add = df_to_add[[c for c in cols_to_keep if c in df_to_add.columns]]
                                remaining_val_cols = [c for c in df_to_add.columns if c not in merge_cols]
                                if len(remaining_val_cols) == 1 and remaining_val_cols[0] != concept_name:
                                    df_to_add = df_to_add.rename(columns={remaining_val_cols[0]: concept_name})
                                elif len(remaining_val_cols) > 1:
                                    rename_map = {}
                                    for c in remaining_val_cols:
                                        if c != concept_name and not c.startswith(f"{concept_name}_"):
                                            rename_map[c] = f"{concept_name}_{c}"
                                    if rename_map:
                                        df_to_add = df_to_add.rename(columns=rename_map)
                            for mc in merge_cols:
                                if mc not in df_to_add.columns:
                                    if mc in {'charttime', 'time', 'starttime', 'endtime', 'itemtime'}:
                                        df_to_add[mc] = 0.0
                                    else:
                                        df_to_add[mc] = np.nan
                            keep_cols = merge_cols + [c for c in df_to_add.columns if c not in merge_cols]
                            all_concept_dfs.append(df_to_add[keep_cols])
                        
                        if len(all_concept_dfs) == 0:
                            merged_df = None
                        elif len(all_concept_dfs) == 1:
                            merged_df = all_concept_dfs[0]
                        else:
                            time_related_cols = {'charttime', 'time', 'starttime', 'endtime', 'itemtime'}
                            id_related_cols = {'stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID'}
                            for i, df in enumerate(all_concept_dfs):
                                for col in merge_cols:
                                    if col in df.columns:
                                        col_dtype = df[col].dtype
                                        if col in time_related_cols:
                                            if col_dtype == 'object' or not pd.api.types.is_numeric_dtype(col_dtype):
                                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce')
                                        elif col in id_related_cols:
                                            if col_dtype == 'object':
                                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                                            elif pd.api.types.is_numeric_dtype(col_dtype):
                                                all_concept_dfs[i][col] = df[col].astype('Int64')
                                        else:
                                            if col_dtype == 'object':
                                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce')
                            
                            row_counts = [len(df) for df in all_concept_dfs]
                            for i, df in enumerate(all_concept_dfs):
                                for col in merge_cols:
                                    if col in time_related_cols and pd.api.types.is_float_dtype(df[col]):
                                        all_concept_dfs[i][col] = df[col].round(2)
                            
                            total_rows_sum = sum(row_counts)
                            use_fast_path = (total_rows_sum < 2_000_000)
                            
                            if use_fast_path:
                                try:
                                    processed_dfs = []
                                    static_dfs = []
                                    _empty_concepts_local = []
                                    for df in all_concept_dfs:
                                        df_temp = df.copy()
                                        val_cols = [c for c in df_temp.columns if c not in merge_cols]
                                        if not val_cols:
                                            continue
                                        is_static = False
                                        if _time_col and _time_col not in df_temp.columns:
                                            is_static = True
                                        elif _time_col and _time_col in df_temp.columns:
                                            if df_temp[_time_col].isna().all():
                                                is_static = True
                                        if is_static:
                                            if _id_col and _id_col in df_temp.columns:
                                                static_cols = [_id_col] + val_cols
                                                static_df = df_temp[static_cols].drop_duplicates(subset=[_id_col], keep='last')
                                                static_dfs.append(static_df)
                                        else:
                                            df_temp = df_temp.drop_duplicates(subset=merge_cols, keep='last')
                                            for value_col in val_cols:
                                                if len(df_temp) == 0:
                                                    _empty_concepts_local.append(value_col)
                                                    continue
                                                single_val_df = df_temp[merge_cols + [value_col]].copy()
                                                single_val_df['_concept'] = str(value_col)
                                                single_val_df['_value'] = single_val_df[value_col]
                                                single_val_df.drop(columns=[value_col], inplace=True)
                                                processed_dfs.append(single_val_df)
                                    
                                    if not processed_dfs and not static_dfs:
                                        merged_df = None
                                    else:
                                        if processed_dfs:
                                            stacked = pd.concat(processed_dfs, ignore_index=True)
                                            merged_df = stacked.pivot_table(
                                                index=merge_cols, columns='_concept',
                                                values='_value', aggfunc='first'
                                            ).reset_index()
                                            for ec in _empty_concepts_local:
                                                if ec not in merged_df.columns:
                                                    merged_df[ec] = np.nan
                                        else:
                                            merged_df = None
                                        if static_dfs:
                                            from functools import reduce
                                            static_merged = reduce(
                                                lambda left, right: pd.merge(left, right, on=_id_col, how='outer'),
                                                static_dfs
                                            )
                                            if merged_df is not None and _id_col in merged_df.columns:
                                                merged_df = pd.merge(merged_df, static_merged, on=_id_col, how='left')
                                            else:
                                                merged_df = static_merged
                                except Exception:
                                    use_fast_path = False
                            
                            if not use_fast_path:
                                if len(all_concept_dfs) > 10:
                                    _batch_sz = 5
                                    batches = []
                                    for i in range(0, len(all_concept_dfs), _batch_sz):
                                        batch = all_concept_dfs[i:i+_batch_sz]
                                        from functools import reduce
                                        try:
                                            batch_merged = reduce(
                                                lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                                                batch
                                            )
                                            if len(batch_merged) > 0:
                                                batch_merged = batch_merged.drop_duplicates(subset=merge_cols)
                                            batches.append(batch_merged)
                                        except Exception:
                                            continue
                                    if not batches:
                                        merged_df = None
                                    else:
                                        merged_df = reduce(
                                            lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                                            batches
                                        )
                                else:
                                    from functools import reduce
                                    merged_df = reduce(
                                        lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                                        all_concept_dfs
                                    )
                                if merged_df is not None and len(merged_df) > 0:
                                    merged_df = merged_df.drop_duplicates(subset=merge_cols)
                    
                    if merged_df is None:
                        if merge_cols:
                            merged_df = pd.DataFrame(columns=merge_cols + list(concept_dfs_dict.keys()))
                        else:
                            return False
                    
                    # 生成文件名
                    concept_names_sorted = sorted(list(concept_dfs_dict.keys()))
                    if len(concept_names_sorted) <= 5:
                        concepts_suffix = '_'.join(concept_names_sorted)
                    else:
                        concepts_suffix = '_'.join(concept_names_sorted[:4]) + f'_etc{len(concept_names_sorted)}'
                    cohort_suffix = _generate_cohort_prefix()
                    if cohort_suffix:
                        safe_filename = f"{group_name}_{concepts_suffix}_{cohort_suffix}".replace('/', '_').replace('\\', '_')
                    else:
                        safe_filename = f"{group_name}_{concepts_suffix}".replace('/', '_').replace('\\', '_')
                    if len(safe_filename) > 150:
                        safe_filename = safe_filename[:150]
                    
                    if export_format == 'csv':
                        file_path = export_dir / f"{safe_filename}.csv"
                    elif export_format == 'parquet':
                        file_path = export_dir / f"{safe_filename}.parquet"
                    elif export_format == 'excel':
                        file_path = export_dir / f"{safe_filename}.xlsx"
                    else:
                        file_path = export_dir / f"{safe_filename}.parquet"
                    
                    # 覆盖模式
                    _ow_modules = st.session_state.get('_overwrite_modules', set())
                    if group_name in _ow_modules or is_viz_import_mode:
                        for ext in ['.parquet', '.csv', '.xlsx']:
                            for old_file in export_dir.glob(f"{group_name}_*{ext}"):
                                try:
                                    old_file.unlink()
                                except Exception:
                                    pass
                    
                    # 跳过检查
                    if not use_mock and not is_viz_import_mode and file_path.exists():
                        if group_name in skipped_modules:
                            return False
                    
                    # 收集患者ID
                    if merged_df is not None and len(merged_df) > 0:
                        for _idc in ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']:
                            if _idc in merged_df.columns:
                                all_exported_patient_ids.update(merged_df[_idc].dropna().unique())
                                break
                    
                    # 写入文件
                    if export_format == 'csv':
                        merged_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    elif export_format == 'parquet':
                        for col in merged_df.columns:
                            if merged_df[col].dtype == object:
                                numeric_vals = pd.to_numeric(merged_df[col], errors='coerce')
                                orig_valid = merged_df[col].notna().sum()
                                if orig_valid > 0 and numeric_vals.notna().sum() >= orig_valid * 0.5:
                                    merged_df[col] = numeric_vals
                                else:
                                    merged_df[col] = merged_df[col].astype(str)
                        merged_df.to_parquet(file_path, index=False)
                    elif export_format == 'excel':
                        merged_df.to_excel(file_path, index=False)
                    else:
                        for col in merged_df.columns:
                            if merged_df[col].dtype == object:
                                numeric_vals = pd.to_numeric(merged_df[col], errors='coerce')
                                orig_valid = merged_df[col].notna().sum()
                                if orig_valid > 0 and numeric_vals.notna().sum() >= orig_valid * 0.5:
                                    merged_df[col] = numeric_vals
                                else:
                                    merged_df[col] = merged_df[col].astype(str)
                        merged_df.to_parquet(file_path, index=False)
                    
                    exported_files.append(str(file_path))
                    _mod_exp_elapsed = time_module.time() - _mod_export_start
                    # 合并加载+导出耗时，反映该模块的真实总耗时
                    _load_time = _module_load_times.get(group_name, 0)
                    module_times[group_name] = _load_time + _mod_exp_elapsed
                    return True
                
                # ──────────────────────────────────────────────
                # 🚀 流式导出: cohort filter 计算辅助函数
                # ──────────────────────────────────────────────
                _cohort_filter_modules = {'demographics', 'outcome'}  # 包含 death/los_icu/age/sex 的模块
                
                def _compute_cohort_exclude_ids_inline():
                    """从 data 中计算要排除的患者ID集合。"""
                    cf = st.session_state.get('cohort_filter', {})
                    if not cf:
                        return set()
                    
                    id_col_map = {
                        'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid',
                        'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID',
                    }
                    _act_id_col = id_col_map.get(database, 'stay_id')
                    id_cands = [_act_id_col, 'stay_id', 'icustay_id', 'hadm_id',
                                'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                    actual_id = None
                    for df in data.values():
                        if isinstance(df, pd.DataFrame):
                            for c in id_cands:
                                if c in df.columns:
                                    actual_id = c
                                    break
                            if actual_id:
                                break
                    if not actual_id:
                        return set()
                    
                    all_pids = set()
                    for df in data.values():
                        if isinstance(df, pd.DataFrame) and actual_id in df.columns:
                            all_pids.update(df[actual_id].dropna().unique())
                    if not all_pids:
                        return set()
                    
                    excl = set()
                    if cf.get('survived') is not None and 'death' in data:
                        death_df = data['death']
                        if isinstance(death_df, pd.DataFrame) and actual_id in death_df.columns:
                            val_col = 'death' if 'death' in death_df.columns else death_df.columns[-1]
                            death_valid = death_df[death_df[val_col].notna()].copy()
                            death_vals = pd.to_numeric(death_valid[val_col], errors='coerce')
                            died_ids = set(death_valid.loc[death_vals == 1, actual_id].unique())
                            survived_ids = all_pids - died_ids
                            if not cf['survived']:
                                excl |= survived_ids
                            else:
                                excl |= died_ids
                    
                    if cf.get('los_min') is not None and 'los_icu' in data:
                        los_df = data['los_icu']
                        if isinstance(los_df, pd.DataFrame) and actual_id in los_df.columns:
                            val_col = 'los_icu' if 'los_icu' in los_df.columns else los_df.columns[-1]
                            los_valid = los_df[los_df[val_col].notna()].copy()
                            los_hours = pd.to_numeric(los_valid[val_col], errors='coerce') * 24
                            los_ok = set(los_valid.loc[los_hours >= cf['los_min'], actual_id].unique())
                            excl |= (all_pids - los_ok)
                    
                    if (cf.get('age_min') is not None or cf.get('age_max') is not None) and 'age' in data:
                        age_df = data['age']
                        if isinstance(age_df, pd.DataFrame) and actual_id in age_df.columns:
                            val_col = 'age' if 'age' in age_df.columns else age_df.columns[-1]
                            age_valid = age_df[age_df[val_col].notna()].copy()
                            age_vals = pd.to_numeric(age_valid[val_col], errors='coerce')
                            age_mask = pd.Series(True, index=age_valid.index)
                            if cf.get('age_min') is not None:
                                age_mask &= (age_vals >= cf['age_min'])
                            if cf.get('age_max') is not None:
                                age_mask &= (age_vals <= cf['age_max'])
                            age_ok = set(age_valid.loc[age_mask, actual_id].unique())
                            excl |= (all_pids - age_ok)
                    
                    if cf.get('gender') is not None and 'sex' in data:
                        sex_df = data['sex']
                        if isinstance(sex_df, pd.DataFrame) and actual_id in sex_df.columns:
                            val_col = 'sex' if 'sex' in sex_df.columns else sex_df.columns[-1]
                            sex_valid = sex_df[sex_df[val_col].notna()].copy()
                            sex_vals = sex_valid[val_col].astype(str).str.strip().str.upper()
                            target = cf['gender'].upper()
                            if target == 'M':
                                target_variants = {'M', 'MALE', 'MAN', 'MÄNNLICH'}
                            else:
                                target_variants = {'F', 'FEMALE', 'WOMAN', 'WEIBLICH', 'VROUW', 'W'}
                            sex_ok = set(sex_valid.loc[sex_vals.isin(target_variants), actual_id].unique())
                            excl |= (all_pids - sex_ok)
                    
                    if excl:
                        print(f"[COHORT POST-FILTER] Removing {len(excl)}/{len(all_pids)} patients")
                        cohort_stats = st.session_state.get('_cohort_stats')
                        if cohort_stats:
                            cohort_stats['after'] = len(all_pids) - len(excl)
                            cohort_stats['excluded'] = cohort_stats['before'] - (len(all_pids) - len(excl))
                            n_excl = len(excl)
                            detail_label_en = f"Data consistency check: -{n_excl}"
                            detail_label_cn = f"数据一致性检查: -{n_excl}"
                            cohort_stats.setdefault('filter_details', []).append(
                                (detail_label_en, detail_label_cn, n_excl)
                            )
                            st.session_state['_cohort_stats'] = cohort_stats
                    return excl
                
                def _apply_cohort_filter_to_dfs(concept_dfs_dict, exclude_ids):
                    """对一组概念数据应用 cohort filter，移除 exclude_ids 中的患者。"""
                    if not exclude_ids:
                        return concept_dfs_dict
                    id_cands = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                    filtered = {}
                    for cname, df in concept_dfs_dict.items():
                        if isinstance(df, pd.DataFrame):
                            for idc in id_cands:
                                if idc in df.columns:
                                    filtered[cname] = df[~df[idc].isin(exclude_ids)].copy()
                                    break
                            else:
                                filtered[cname] = df
                        else:
                            filtered[cname] = df
                    return filtered
                
                # 跟踪已处理的模块和等待 cohort filter 的缓冲模块
                _buffered_mod_keys = []  # 等待 cohort filter 计算后再导出的模块
                _loaded_mod_keys = set()  # 已加载的模块keys
                _export_step_counter = 0  # 导出计数器
                # 🔧 FIX: 检查是否有实际设置的 filter 值，而非仅检查 dict 是否非空
                # cohort_filter 初始化为 {'age_min': None, ...}（8 keys 全 None），
                # bool(non_empty_dict) = True 导致永远为 True，所有模块被缓冲
                _cf = st.session_state.get('cohort_filter', {})
                _has_cohort_filter = any(v is not None for v in _cf.values()) if _cf else False
                
                # 模块级别耗时累计（用于动态ETA计算）
                _module_elapsed_list = []
                
                # 🚀 优化: 缓存特殊概念(AKI/CircFailure)的依赖概念 parquet
                # 在主模块循环中保存，避免特殊概念子进程重新从数据库加载
                _special_dep_concepts = {'crea', 'urine', 'weight', 'rrt',
                                         'lact', 'map', 'norepi_rate', 'epi_rate',
                                         'dobu_rate', 'dopa_rate',
                                         'sofa', 'sofa2', 'susp_inf'}  # 🔧 FIX Bug 53: 包含 SOFA/susp_inf 供 Sep3 复用
                _deps_cache_dir = None
                if special_concepts_to_load:
                    import tempfile as _tmpf_deps
                    _deps_cache_dir = _tmpf_deps.mkdtemp(prefix='easyicu_deps_')
                
                for mod_idx, mod_key in enumerate(ordered_modules):
                    mod_concepts = module_concept_map[mod_key]
                    mod_display = CONCEPT_GROUP_NAMES.get(mod_key, (mod_key, mod_key))
                    mod_name = mod_display[1] if lang != 'en' else mod_display[0]
                    
                    # 计算已用时间和ETA
                    elapsed = _time_mod.time() - _export_start
                    if mod_idx > 0 and _module_elapsed_list:
                        avg_mod_time = sum(_module_elapsed_list) / len(_module_elapsed_list)
                        remaining_mods = total_modules - mod_idx
                        eta_seconds = int(avg_mod_time * remaining_mods)
                        if eta_seconds >= 60:
                            eta_str = f" | ETA ~{eta_seconds // 60}分{eta_seconds % 60:02d}秒" if lang != 'en' else f" | ETA ~{eta_seconds // 60}m{eta_seconds % 60:02d}s"
                        else:
                            eta_str = f" | ETA ~{eta_seconds}秒" if lang != 'en' else f" | ETA ~{eta_seconds}s"
                    else:
                        eta_str = ""
                    
                    # 概念列表（最多显示8个）
                    if len(mod_concepts) <= 8:
                        concept_list_str = ', '.join(mod_concepts)
                    else:
                        concept_list_str = ', '.join(mod_concepts[:6]) + f', ... +{len(mod_concepts)-6}'
                    
                    # 显示模块加载状态
                    elapsed_str = f"{elapsed:.0f}s"
                    if lang == 'en':
                        status_msg = (
                            f"**Loading {mod_name}** ({len(mod_concepts)} concepts, "
                            f"module {mod_idx+1}/{total_modules}, elapsed {elapsed_str}){eta_str}\n\n"
                            f"`{concept_list_str}`"
                        )
                    else:
                        status_msg = (
                            f"**正在加载 {mod_name}** ({len(mod_concepts)} 个概念, "
                            f"模块 {mod_idx+1}/{total_modules}, 已用 {elapsed_str}){eta_str}\n\n"
                            f"`{concept_list_str}`"
                        )
                    status_text.markdown(status_msg)
                    
                    # 批量加载整个模块
                    # 🔧 FIX (Bug 52): load + merge + export 全部在子进程内完成
                    # 主进程不接触任何 DataFrame，彻底消除 pymalloc arena 碎片累积
                    # 实测: 旧方案 MIIV 94K × 19模块 RSS 661GB; 新方案主进程 RSS 近零增长
                    mod_start = _time_mod.time()
                    
                    _loaded_mod_keys.add(mod_key)
                    
                    # 判断是否需要走旧路径（cohort filter 需要读回数据计算 exclude_ids）
                    _need_buffered_load = (
                        _has_cohort_filter and not _cohort_filter_computed
                        and mod_key in _cohort_filter_modules
                    )
                    
                    if _need_buffered_load:
                        # ── cohort filter 必需模块: 旧路径（读回小量数据） ──
                        try:
                            import multiprocessing as _mp_mod
                            import tempfile as _tmpf_mod
                            import json as _json_mod
                            
                            _tmp_dir = _tmpf_mod.mkdtemp(prefix='easyicu_mod_')
                            
                            _sub_proc = _mp_mod.Process(
                                target=_subprocess_load_module,
                                args=(mod_concepts, database,
                                      st.session_state.data_path,
                                      patient_ids_filter,
                                      _auto_batch_size, _tmp_dir),
                                daemon=True
                            )
                            _sub_proc.start()
                            
                            _keepalive_tick = 0
                            while _sub_proc.is_alive():
                                _sub_proc.join(timeout=2)
                                _keepalive_tick += 1
                                _mod_elapsed = _time_mod.time() - mod_start
                                _spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'][_keepalive_tick % 10]
                                _alive_msg = (
                                    f"{_spinner} **{'Loading' if lang == 'en' else '正在加载'} {mod_name}** "
                                    f"({len(mod_concepts)} {'concepts' if lang == 'en' else '个概念'}, "
                                    f"{'module' if lang == 'en' else '模块'} {mod_idx+1}/{total_modules}, "
                                    f"{'elapsed' if lang == 'en' else '已用'} {_mod_elapsed:.0f}s){eta_str}"
                                )
                                status_text.markdown(_alive_msg)
                            
                            if _sub_proc.exitcode != 0:
                                raise RuntimeError(f"Subprocess exited with code {_sub_proc.exitcode}")
                            
                            _manifest_path = os.path.join(_tmp_dir, '_manifest.json')
                            if os.path.exists(_manifest_path):
                                with open(_manifest_path) as _mf:
                                    _saved = _json_mod.load(_mf)
                                for _cname, _ppath in _saved.items():
                                    data[_cname] = pd.read_parquet(_ppath)
                            
                            for _cname in mod_concepts:
                                if _cname not in data and _cname not in empty_concepts:
                                    empty_concepts.append(_cname)
                            
                            import shutil as _shutil_mod
                            _shutil_mod.rmtree(_tmp_dir, ignore_errors=True)
                        except Exception:
                            try:
                                _shutil_mod.rmtree(_tmp_dir, ignore_errors=True)
                            except Exception:
                                pass
                            for concept in mod_concepts:
                                try:
                                    result = load_concepts(
                                        data_path=st.session_state.data_path,
                                        database=database, concepts=[concept],
                                        verbose=False, merge=False, concept_workers=1,
                                        **(dict(patient_ids=patient_ids_filter) if patient_ids_filter else {})
                                    )
                                    _process_result(result, [concept])
                                except Exception:
                                    failed_concepts.append(concept)
                        
                        _buffered_mod_keys.append(mod_key)
                        
                        # 检查是否可以计算 cohort filter 了
                        _filter_ready = (
                            _cohort_filter_modules.issubset(_loaded_mod_keys)
                            or mod_idx == len(ordered_modules) - 1
                        )
                        if _filter_ready:
                            _cohort_exclude_ids = _compute_cohort_exclude_ids_inline()
                            _cohort_filter_computed = True
                            
                            # 导出所有缓冲的模块（demographics/outcome 数据量小，主进程处理可接受）
                            for _buf_key in _buffered_mod_keys:
                                _buf_concepts = module_concept_map.get(_buf_key, [])
                                _buf_dfs = {c: data[c] for c in _buf_concepts if c in data}
                                if _buf_dfs:
                                    _buf_dfs = _apply_cohort_filter_to_dfs(_buf_dfs, _cohort_exclude_ids)
                                    try:
                                        _export_module_to_disk(_buf_key, _buf_dfs, _export_step_counter, _total_steps)
                                        _export_step_counter += 1
                                    except Exception as _exp_e:
                                        import traceback as _tb_mod
                                        _tb_mod.print_exc()
                                        st.warning(f"⚠️ Export failed for module '{_buf_key}': {_exp_e}")
                                    for c in _buf_concepts:
                                        data.pop(c, None)
                            _buffered_mod_keys.clear()
                    else:
                        # ── 正常路径: load + merge + export 全部在子进程内完成 ──
                        # 主进程不接触任何 DataFrame，子进程退出后 OS 完整回收所有内存
                        if not _cohort_filter_computed and not _has_cohort_filter:
                            _cohort_filter_computed = True
                        
                        try:
                            import multiprocessing as _mp_mod
                            import json as _json_mod
                            
                            _overwrite_this = (mod_key in overwrite_modules or is_viz_import_mode)
                            
                            _sub_proc = _mp_mod.Process(
                                target=_subprocess_load_and_export_module,
                                args=(mod_concepts, database,
                                      st.session_state.data_path,
                                      patient_ids_filter, _auto_batch_size,
                                      str(export_dir), export_format, mod_key,
                                      list(_cohort_exclude_ids) if _cohort_exclude_ids else None,
                                      _overwrite_this, cohort_suffix,
                                      _special_dep_concepts if _deps_cache_dir else None,
                                      _deps_cache_dir),
                                daemon=True
                            )
                            _sub_proc.start()
                            
                            _keepalive_tick = 0
                            while _sub_proc.is_alive():
                                _sub_proc.join(timeout=2)
                                _keepalive_tick += 1
                                _mod_elapsed = _time_mod.time() - mod_start
                                _spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'][_keepalive_tick % 10]
                                _alive_msg = (
                                    f"{_spinner} **{'Loading & exporting' if lang == 'en' else '正在加载并导出'} {mod_name}** "
                                    f"({len(mod_concepts)} {'concepts' if lang == 'en' else '个概念'}, "
                                    f"{'module' if lang == 'en' else '模块'} {mod_idx+1}/{total_modules}, "
                                    f"{'elapsed' if lang == 'en' else '已用'} {_mod_elapsed:.0f}s){eta_str}"
                                )
                                status_text.markdown(_alive_msg)
                            
                            if _sub_proc.exitcode != 0:
                                raise RuntimeError(f"Subprocess exited with code {_sub_proc.exitcode}")
                            
                            # 读取子进程元数据（不读 DataFrame！主进程零内存增长）
                            _manifest_path = os.path.join(str(export_dir), f'_manifest_{mod_key}.json')
                            if os.path.exists(_manifest_path):
                                with open(_manifest_path) as _mf:
                                    _meta = _json_mod.load(_mf)
                                if _meta.get('exported_file'):
                                    exported_files.append(_meta['exported_file'])
                                    module_times[mod_key] = _time_mod.time() - mod_start
                                if _meta.get('patient_ids'):
                                    all_exported_patient_ids.update(_meta['patient_ids'])
                                for _ec in _meta.get('empty_concepts', []):
                                    if _ec not in empty_concepts:
                                        empty_concepts.append(_ec)
                                _export_step_counter += 1
                                # 清理 manifest 文件
                                try:
                                    os.unlink(_manifest_path)
                                except Exception:
                                    pass
                            else:
                                # 子进程成功但无 manifest → 所有概念为空
                                for _cname in mod_concepts:
                                    if _cname not in empty_concepts:
                                        empty_concepts.append(_cname)
                        
                        except Exception as _sub_e:
                            # 子进程失败 → 逐概念回退（inprocess）
                            import traceback as _tb_mod
                            _tb_mod.print_exc()
                            for concept in mod_concepts:
                                try:
                                    result = load_concepts(
                                        data_path=st.session_state.data_path,
                                        database=database, concepts=[concept],
                                        verbose=False, merge=False, concept_workers=1,
                                        **(dict(patient_ids=patient_ids_filter) if patient_ids_filter else {})
                                    )
                                    _process_result(result, [concept])
                                except Exception:
                                    failed_concepts.append(concept)
                            # 回退模式: 用旧方式导出
                            _cur_dfs = {c: data[c] for c in mod_concepts if c in data}
                            if _cur_dfs:
                                _cur_dfs = _apply_cohort_filter_to_dfs(_cur_dfs, _cohort_exclude_ids)
                                try:
                                    _export_module_to_disk(mod_key, _cur_dfs, _export_step_counter, _total_steps)
                                    _export_step_counter += 1
                                except Exception:
                                    pass
                                for c in mod_concepts:
                                    data.pop(c, None)
                    
                    mod_time = _time_mod.time() - mod_start
                    _module_elapsed_list.append(mod_time)
                    if mod_key not in module_times:
                        _module_load_times[mod_key] = mod_time
                    
                    # 更新进度条
                    progress_bar.progress(0.15 + 0.65 * (mod_idx + 1) / _total_steps)
                
                progress_bar.progress(0.80)
                
                # 🆕 加载特殊概念（AKI, circ_failure等）— 子进程隔离
                if special_concepts_to_load:
                    special_msg = f"**Loading special concepts (AKI, CircFailure)...**" if lang == 'en' else f"**正在加载特殊概念 (AKI, 循环衰竭)...**"
                    status_text.markdown(special_msg)
                    
                    try:
                        import multiprocessing as _mp_mod
                        import tempfile as _tmpf_mod
                        import json as _json_mod
                        
                        _sp_tmp_dir = _tmpf_mod.mkdtemp(prefix='easyicu_special_')
                        _sp_start = _time_mod.time()
                        
                        # 构建 concept → group 映射，传给子进程做直接导出
                        _sp_concept_to_group = {c: _concept_to_group_pre.get(c, 'special')
                                                for c in special_concepts_to_load}
                        
                        _sp_proc = _mp_mod.Process(
                            target=_subprocess_load_special,
                            args=(special_concepts_to_load, database,
                                  st.session_state.data_path,
                                  patient_ids_filter,
                                  patient_limit if patient_limit and patient_limit > 0 else None,
                                  _sp_tmp_dir, _deps_cache_dir,
                                  str(export_dir), export_format,
                                  list(_cohort_exclude_ids) if _cohort_exclude_ids else None,
                                  _sp_concept_to_group, cohort_suffix),
                            daemon=True
                        )
                        _sp_proc.start()
                        
                        _sp_tick = 0
                        while _sp_proc.is_alive():
                            _sp_proc.join(timeout=2)
                            _sp_tick += 1
                            _sp_elapsed = _time_mod.time() - _sp_start
                            _sp_spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'][_sp_tick % 10]
                            if lang == 'en':
                                _sp_msg = f"{_sp_spinner} **Loading special concepts** (AKI, CircFailure, elapsed {_sp_elapsed:.0f}s)"
                            else:
                                _sp_msg = f"{_sp_spinner} **正在加载特殊概念** (AKI, 循环衰竭, 已用 {_sp_elapsed:.0f}s)"
                            status_text.markdown(_sp_msg)
                        
                        if _sp_proc.exitcode != 0:
                            raise RuntimeError(f"Special concepts subprocess exited with code {_sp_proc.exitcode}")
                        
                        # 读取子进程导出元数据（不读 DataFrame！主进程零内存增长）
                        _sp_export_manifest = os.path.join(_sp_tmp_dir, '_export_manifest.json')
                        _sp_loaded_concepts = []
                        if os.path.exists(_sp_export_manifest):
                            with open(_sp_export_manifest) as _mf:
                                _sp_exports = _json_mod.load(_mf)
                            for _sg_key, _sg_meta in _sp_exports.items():
                                if _sg_meta.get('exported_file'):
                                    exported_files.append(_sg_meta['exported_file'])
                                    _export_step_counter += 1
                                if _sg_meta.get('patient_ids'):
                                    all_exported_patient_ids.update(_sg_meta['patient_ids'])
                                if _sg_meta.get('concepts'):
                                    _sp_loaded_concepts.extend(_sg_meta['concepts'])
                        else:
                            # Fallback: 旧格式 manifest（个别 parquet）
                            _sp_manifest = os.path.join(_sp_tmp_dir, '_manifest.json')
                            if os.path.exists(_sp_manifest):
                                with open(_sp_manifest) as _mf:
                                    _sp_saved = _json_mod.load(_mf)
                                _sp_loaded_concepts = list(_sp_saved.keys())
                        
                        failed_special = [c for c in special_concepts_to_load if c not in _sp_loaded_concepts]
                        failed_concepts.extend(failed_special)
                        
                        _special_load_elapsed = _time_mod.time() - _sp_start
                        
                        import shutil as _shutil_mod
                        _shutil_mod.rmtree(_sp_tmp_dir, ignore_errors=True)
                        if _deps_cache_dir:
                            _shutil_mod.rmtree(_deps_cache_dir, ignore_errors=True)
                            _deps_cache_dir = None
                        
                    except Exception as special_e:
                        try:
                            _shutil_mod.rmtree(_sp_tmp_dir, ignore_errors=True)
                        except Exception:
                            pass
                        try:
                            if _deps_cache_dir:
                                _shutil_mod.rmtree(_deps_cache_dir, ignore_errors=True)
                                _deps_cache_dir = None
                        except Exception:
                            pass
                        st.warning(f"⚠️ Failed to load special concepts: {special_e}" if lang == 'en' else f"⚠️ 加载特殊概念失败: {special_e}")
                        failed_concepts.extend(special_concepts_to_load)
                    
                    progress_bar.progress(0.90)
                
                # 🚀 流式导出: 导出 data 中剩余的未导出概念（如果有）
                if data:
                    _remaining_by_group = {}
                    for _rc, _rdf in data.items():
                        if isinstance(_rdf, pd.DataFrame):
                            _rg = _concept_to_group_pre.get(_rc, 'other')
                            if _rg not in _remaining_by_group:
                                _remaining_by_group[_rg] = {}
                            _remaining_by_group[_rg][_rc] = _rdf
                    for _rg_key, _rg_dfs in _remaining_by_group.items():
                        _rg_dfs = _apply_cohort_filter_to_dfs(_rg_dfs, _cohort_exclude_ids)
                        try:
                            _export_module_to_disk(_rg_key, _rg_dfs, _export_step_counter, _total_steps)
                            _export_step_counter += 1
                        except Exception as _exp_e:
                            import traceback as _tb_mod
                            _tb_mod.print_exc()
                            st.warning(f"⚠️ Export failed for remaining module '{_rg_key}': {_exp_e}")
                    data.clear()
                
                # 🔧 FIX: 合并 unsupported 和 failed 概念，只显示一次警告
                all_skipped = list(set(unsupported_concepts + failed_concepts))
                if all_skipped:
                    skip_list = ', '.join(all_skipped[:5])
                    more_text = f'... +{len(all_skipped)-5}' if len(all_skipped) > 5 else ''
                    skip_msg = f"⚠️ Skipped {len(all_skipped)} unavailable: {skip_list}{more_text}" if lang == 'en' else f"⚠️ 跳过 {len(all_skipped)} 个不可用: {skip_list}{more_text}"
                    st.warning(skip_msg)
                
                # 🆕 显示空结果概念提示
                if empty_concepts:
                    empty_list = ', '.join(empty_concepts[:8])
                    more_text = f'... +{len(empty_concepts)-8}' if len(empty_concepts) > 8 else ''
                    empty_msg = f"ℹ️ {len(empty_concepts)} concepts returned empty (not configured or no data): {empty_list}{more_text}" if lang == 'en' else f"ℹ️ {len(empty_concepts)} 个概念返回空结果（未配置或无数据）: {empty_list}{more_text}"
                    st.info(empty_msg)
                
                # 概念计数：已导出 = exported_files 中的概念数（后面统计）
                _n_loaded_concepts = len(valid_concepts) - len(failed_concepts) - len(unsupported_concepts)
                loaded_msg = f"✅ Loaded & exported {_n_loaded_concepts} concepts" if lang == 'en' else f"✅ 已加载并导出 {_n_loaded_concepts} 个概念"
                status_text.markdown(loaded_msg)
                
            except Exception as e:
                import traceback as _tb_mod
                _tb_mod.print_exc()
                warn_msg = f"⚠️ Batch loading failed: {e}" if lang == 'en' else f"⚠️ 批量加载失败: {e}"
                st.warning(warn_msg)
                data = {}
        
        # 流式导出已在加载循环中完成，无需旧的 Phase 2 导出循环
        import time as time_module
        # 使用真正的开始时间计算总耗时
        export_start_time = _export_start if '_export_start' in dir() else time_module.time()

        if (use_mock or use_loaded_data_export) and data:
            from functools import reduce
            import warnings as _mock_warnings

            def _export_mock_module_to_disk(group_name, concept_dfs_dict):
                """Merge demo-mode concept frames and export them as one module file."""
                nonlocal all_exported_patient_ids

                if not concept_dfs_dict:
                    return False

                id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                time_candidates = ['time', 'charttime', 'starttime', 'start', 'endtime', 'itemtime', 'datetime', 'Offset', 'measuredat_minutes', 'measuredat', 'givenat', 'enteredentryat', 'intakeoutputoffset', 'observationoffset', 'nursingchartoffset', 'labresultoffset', 'respchartoffset']
                unified_time_col = 'charttime'

                normalized = {}
                for cname, cdf in concept_dfs_dict.items():
                    if not isinstance(cdf, pd.DataFrame) or cdf.empty:
                        continue
                    frame = cdf.copy()
                    if unified_time_col in frame.columns:
                        other_time_cols = [tc for tc in time_candidates if tc in frame.columns and tc != unified_time_col]
                        if other_time_cols:
                            frame = frame.drop(columns=other_time_cols)
                    else:
                        for tc in time_candidates:
                            if tc in frame.columns:
                                frame = frame.rename(columns={tc: unified_time_col})
                                break
                    normalized[cname] = frame

                if not normalized:
                    return False

                potential_id_cols = set()
                potential_time_cols = set()
                for frame in normalized.values():
                    for col in id_candidates:
                        if col in frame.columns:
                            potential_id_cols.add(col)
                            break
                    if unified_time_col in frame.columns:
                        potential_time_cols.add(unified_time_col)

                merge_cols = []
                id_col = next((col for col in id_candidates if col in potential_id_cols), None)
                if id_col:
                    merge_cols.append(id_col)
                time_col = unified_time_col if unified_time_col in potential_time_cols else None
                if time_col:
                    merge_cols.append(time_col)

                static_frames = []
                ts_frames = []
                metadata_cols = {'valueuom', 'unit', 'units', 'category', 'type',
                                 'dur_var', 'entertime',
                                 'intakeoutputentryoffset'}  # dur_var/entertime: WinTbl; intakeoutputentryoffset: eICU extra

                for concept_name, frame in normalized.items():
                    drop_cols = [c for c in frame.columns if c in metadata_cols]
                    if drop_cols:
                        frame = frame.drop(columns=drop_cols)

                    value_cols = [c for c in frame.columns if c not in merge_cols]
                    if not value_cols:
                        continue

                    if len(value_cols) == 1:
                        frame = frame.rename(columns={value_cols[0]: concept_name})
                        value_cols = [concept_name]

                    is_static = (not time_col) or (time_col not in frame.columns) or (time_col in frame.columns and frame[time_col].isna().all())
                    if is_static:
                        if id_col and id_col in frame.columns:
                            static_cols = [id_col] + [c for c in value_cols if c in frame.columns]
                            static_frames.append(frame[static_cols].drop_duplicates(subset=[id_col], keep='last'))
                    else:
                        keep_cols = merge_cols + [c for c in value_cols if c in frame.columns]
                        ts_frames.append(frame[keep_cols].drop_duplicates(subset=merge_cols, keep='last'))

                merged_df = None
                if ts_frames:
                    # 统一 id_col 为 Int64 避免 outer merge 中 int/float 混合警告
                    if id_col:
                        for i, f in enumerate(ts_frames):
                            if id_col in f.columns and f[id_col].dtype != 'Int64':
                                ts_frames[i] = f.copy()
                                ts_frames[i][id_col] = ts_frames[i][id_col].astype('Int64')
                    with _mock_warnings.catch_warnings():
                        _mock_warnings.simplefilter('ignore', UserWarning)
                        merged_df = ts_frames[0] if len(ts_frames) == 1 else reduce(
                            lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                            ts_frames
                        )
                if static_frames:
                    with _mock_warnings.catch_warnings():
                        _mock_warnings.simplefilter('ignore', UserWarning)
                        static_df = static_frames[0] if len(static_frames) == 1 else reduce(
                            lambda left, right: pd.merge(left, right, on=[id_col], how='outer'),
                            static_frames
                        )
                    if merged_df is not None and id_col and id_col in merged_df.columns:
                        merged_df = pd.merge(merged_df, static_df, on=[id_col], how='left')
                    else:
                        merged_df = static_df

                if merged_df is None or merged_df.empty:
                    return False

                if id_col and id_col in merged_df.columns:
                    all_exported_patient_ids.update(merged_df[id_col].dropna().unique())

                concept_names_sorted = sorted(list(concept_dfs_dict.keys()))
                if len(concept_names_sorted) <= 5:
                    concepts_suffix = '_'.join(concept_names_sorted)
                else:
                    concepts_suffix = '_'.join(concept_names_sorted[:4]) + f'_etc{len(concept_names_sorted)}'

                cohort_suffix = _generate_cohort_prefix()
                safe_filename = f"{group_name}_{concepts_suffix}"
                if cohort_suffix:
                    safe_filename = f"{safe_filename}_{cohort_suffix}"
                safe_filename = safe_filename.replace('/', '_').replace('\\', '_')
                if len(safe_filename) > 150:
                    safe_filename = safe_filename[:150]

                if export_format == 'csv':
                    file_path = export_dir / f"{safe_filename}.csv"
                    merged_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif export_format == 'excel':
                    file_path = export_dir / f"{safe_filename}.xlsx"
                    merged_df.to_excel(file_path, index=False)
                else:
                    file_path = export_dir / f"{safe_filename}.parquet"
                    merged_df.to_parquet(file_path, index=False)

                exported_files.append(str(file_path))
                return True

            mock_modules = []
            for group_key, group_concepts in selected_modules.items():
                group_dfs = {c: data[c] for c in group_concepts if c in data}
                if group_dfs:
                    mock_modules.append((group_key, group_dfs))

            total_mock_modules = len(mock_modules) or 1
            for mod_idx, (mod_key, mod_dfs) in enumerate(mock_modules):
                mod_display = CONCEPT_GROUP_NAMES.get(mod_key, (mod_key, mod_key))
                mod_name = mod_display[1] if lang != 'en' else mod_display[0]
                status_text.markdown(
                    f"**{'Exporting' if lang == 'en' else '正在导出'} {mod_name}** "
                    f"({mod_idx + 1}/{total_mock_modules})"
                )
                mod_start = time_module.time()
                _export_mock_module_to_disk(mod_key, mod_dfs)
                module_times[mod_key] = time_module.time() - mod_start
                progress_bar.progress(0.3 + 0.5 * (mod_idx + 1) / total_mock_modules)
        
        # 完成
        progress_bar.progress(1.0)
        status_text.empty()
        cancel_placeholder.empty()  # 🔧 清理取消按钮
        
        # ⚡ PERF: 清理跨模块缓存
        try:
            _loader.concept_resolver._keep_cache_between_calls = False
            _loader.concept_resolver._raw_concept_cache.clear()
            _loader.concept_resolver._table_cache.clear()
        except Exception:
            pass
        
        # 🔧 清理临时状态
        if '_skipped_modules' in st.session_state:
            del st.session_state['_skipped_modules']
        if '_overwrite_modules' in st.session_state:
            del st.session_state['_overwrite_modules']
        if '_export_cancelled' in st.session_state:
            del st.session_state['_export_cancelled']

        if exported_files:
            st.session_state.export_completed = True
            st.session_state.trigger_export = False  # 🔧 FIX (2026-02-03): 导出完成后重置触发状态
            st.session_state['_exporting_in_progress'] = False  # 清除导出进行中标记
            st.session_state.last_export_dir = str(export_dir)  # 保存实际导出目录
            st.session_state.last_export_full_dir = str(export_dir)  # 保存完整路径（含cohort子目录）
            st.session_state.viz_export_path = str(export_dir)  # 更新viz_export_path
            st.session_state.viz_data_source_mode = "exported"
            st.session_state._prefer_exported_viz = True
            # 🔧 FIX: 更新快速可视化的确认路径，这样切换到可视化页面时会自动填充
            st.session_state.viz_confirmed_path = str(export_dir)
            # 🔧 FIX: 强制重置 text_input 的版本号，确保显示新路径
            if '_viz_export_path_version' not in st.session_state:
                st.session_state._viz_export_path_version = 0
            st.session_state._viz_export_path_version += 1
            
            # 🆕 保存实际导出的患者数量（从数据中统计，是 cohort filter 后的真实数量）
            actual_patient_count = len(all_exported_patient_ids)
            st.session_state['_exported_patient_count'] = actual_patient_count
            
            # 🔧 FIX (2026-02-12): 统计实际导出的概念数量
            # 遍历导出的 parquet 文件，收集所有列名，然后规范化去重
            # 这与 load_from_exported() 的统计方式完全一致
            all_exported_columns = set()
            id_cols_set = {'stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid'}
            time_cols_set = {'time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'index'}
            meta_cols_set = {'_concept'}
            exclude_cols_set = id_cols_set | time_cols_set | meta_cols_set
            
            for file_path in exported_files:
                try:
                    if file_path.endswith('.parquet'):
                        # 🔧 FIX (2026-04-13): 仅读取 schema（列名），不读取数据
                        # 之前用 pd.read_parquet 读取完整数据仅为获取列名，
                        # 对 200K 患者的大 parquet 文件会在主进程产生 GB 级 pymalloc 碎片
                        import pyarrow.parquet as _pq_stat
                        _schema = _pq_stat.read_schema(file_path)
                        _cols = _schema.names
                    elif file_path.endswith('.csv'):
                        # 只读取列名，不读取全部数据
                        _cols = list(pd.read_csv(file_path, nrows=0).columns)
                    elif file_path.endswith('.xlsx'):
                        _cols = list(pd.read_excel(file_path, nrows=0).columns)
                    else:
                        continue
                    for col in _cols:
                        if col not in exclude_cols_set:
                            # 规范化列名
                            norm_col = normalize_column_name(col)
                            all_exported_columns.add(norm_col)
                except Exception:
                    pass  # 忽略读取错误的文件
            
            exported_concept_count = len(all_exported_columns)
            
            # 🔧 DEBUG: 打印实际收集到的患者数量和概念数量
            print(f"[DEBUG] Exported patient count: {actual_patient_count}, concept count: {exported_concept_count}")
            
            # 🆕 计算被选择但未能提取的概念列表
            # 这不是错误，只是一些概念在当前数据库中不可用
            selected_but_not_exported = []
            selected_concepts_set = set(selected_concepts) if selected_concepts else set()
            for c in selected_concepts_set:
                # 如果概念不在成功导出的列中，则添加到未提取列表
                norm_c = normalize_column_name(c)
                if norm_c not in all_exported_columns:
                    selected_but_not_exported.append(c)
            
            # 🆕 保存导出结果到 session state，rerun 后在 Guide: Complete 中显示
            total_elapsed = time_module.time() - export_start_time
            st.session_state['_export_success_result'] = {
                'files': exported_files,
                'export_dir': str(export_dir),
                'total_time': total_elapsed,
                'module_times': module_times.copy(),
                'patient_count': actual_patient_count,  # 🆕 保存实际患者数
                'concept_count': exported_concept_count,  # 🆕 保存实际概念数
                'unavailable_concepts': selected_but_not_exported,  # 🆕 被选择但未能提取的概念
                'unsupported_concepts': unsupported_concepts,  # 🆕 FIX(2026-02-09): 无数据源的概念
                'empty_data_concepts': empty_concepts,  # 🆕 FIX(2026-02-09): 有数据源但无数据的概念
            }
            st.rerun()  # 🆕 立即刷新页面，让 Step 4 变为 DONE
        else:
            st.session_state['_exporting_in_progress'] = False  # 🆕 清除导出进行中标记
            no_data_msg = "⚠️ No data was exported" if lang == 'en' else "⚠️ 没有数据被导出"
            st.warning(no_data_msg)
                
    except Exception as e:
        st.session_state['_exporting_in_progress'] = False  # 🆕 清除导出进行中标记
        fail_msg = f"❌ Export failed: {e}" if lang == 'en' else f"❌ 导出失败: {e}"
        st.error(fail_msg)


def render_export_page():
    """渲染数据导出页面。"""
    lang = st.session_state.get('language', 'en')
    _exp_title = "Data Export" if lang == 'en' else "数据导出"
    _exp_sub = "Download data in CSV, Parquet, or Excel" if lang == 'en' else "以CSV、Parquet或Excel格式下载数据"
    st.markdown(f'''
    <div style="margin-bottom:16px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{_exp_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_exp_sub}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        _msg = "Load data to enable export." if lang == 'en' else "请先加载数据以启用导出。"
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">💾</div>
            <div style="font-weight:600;color:#111827">{_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # 快速导出面板
    quick_title = "⚡ Quick Export" if lang == 'en' else "⚡ 快速导出"
    st.markdown(f"### {quick_title}")
    quick_cols = st.columns(4)
    
    import io
    from datetime import datetime
    
    with quick_cols[0]:
        # 一键导出所有CSV
        df_list = [df.assign(concept=name) for name, df in st.session_state.loaded_concepts.items() 
                   if isinstance(df, pd.DataFrame) and len(df) > 0]
        if df_list:
            all_data = pd.concat(df_list, ignore_index=True)
            csv_all = all_data.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: 添加 BOM 编码防止中文乱码
            all_csv_label = "📄 All CSV" if lang == 'en' else "📄 全部CSV"
            all_csv_help = "Export all data as CSV" if lang == 'en' else "一键导出所有数据为CSV"
            st.download_button(
                label=all_csv_label,
                data=csv_all,
                file_name=f"easyicu_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
                help=all_csv_help
            )
        else:
            no_data_label = "📄 No Data" if lang == 'en' else "📄 无数据"
            st.button(no_data_label, disabled=True, width="stretch")
    
    with quick_cols[1]:
        # 当前选中患者
        if st.session_state.get('selected_patient'):
            patient_id = st.session_state.selected_patient
            patient_data = {}
            for name, df in st.session_state.loaded_concepts.items():
                if isinstance(df, pd.DataFrame) and st.session_state.id_col in df.columns:
                    patient_df = df[df[st.session_state.id_col] == patient_id]
                    if len(patient_df) > 0:
                        patient_data[name] = patient_df
            
            if patient_data:
                patient_combined = pd.concat(
                    [df.assign(concept=name) for name, df in patient_data.items()],
                    ignore_index=True
                )
                patient_csv = patient_combined.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
                st.download_button(
                    label=f"👤 患者{patient_id}",
                    data=patient_csv,
                    file_name=f"patient_{patient_id}_{datetime.now().strftime('%H%M%S')}.csv",
                    mime="text/csv",
                    width="stretch",
                    help=f"Export all data for patient {patient_id}" if lang == 'en' else f"导出患者 {patient_id} 的所有数据"
                )
            else:
                no_pat = "👤 No Patient" if lang == 'en' else "👤 无患者"
                st.button(no_pat, disabled=True, width="stretch")
        else:
            no_sel = "👤 No Selection" if lang == 'en' else "👤 未选患者"
            no_sel_help = "Please select a patient in Patient View first" if lang == 'en' else "请先在患者视图中选择一位患者"
            st.button(no_sel, disabled=True, width="stretch", help=no_sel_help)
    
    with quick_cols[2]:
        # 生命体征快速导出
        vitals = ['hr', 'map', 'sbp', 'resp', 'spo2', 'temp']
        vitals_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                      if k in vitals and isinstance(v, pd.DataFrame) and len(v) > 0}
        if vitals_data:
            vitals_combined = pd.concat(
                [df.assign(concept=name) for name, df in vitals_data.items()],
                ignore_index=True
            )
            vitals_csv = vitals_combined.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
            vitals_label = "💓 Vitals" if lang == 'en' else "💓 生命体征"
            vitals_help = "Export all vital signs data" if lang == 'en' else "导出所有生命体征数据"
            st.download_button(
                label=vitals_label,
                data=vitals_csv,
                file_name=f"vitals_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
                help=vitals_help
            )
        else:
            no_vitals = "💓 No Vitals" if lang == 'en' else "💓 无体征数据"
            st.button(no_vitals, disabled=True, width="stretch")
    
    with quick_cols[3]:
        # 实验室数据快速导出
        labs = ['bili', 'crea', 'plt', 'lac', 'wbc', 'hgb']
        labs_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                    if k in labs and isinstance(v, pd.DataFrame) and len(v) > 0}
        if labs_data:
            labs_combined = pd.concat(
                [df.assign(concept=name) for name, df in labs_data.items()],
                ignore_index=True
            )
            labs_csv = labs_combined.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
            labs_label = "🧪 Labs" if lang == 'en' else "🧪 实验室"
            labs_help = "Export all laboratory data" if lang == 'en' else "导出所有实验室数据"
            st.download_button(
                label=labs_label,
                data=labs_csv,
                file_name=f"labs_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
                help=labs_help
            )
        else:
            no_labs = "🧪 No Labs Data" if lang == 'en' else "🧪 无实验室数据"
            st.button(no_labs, disabled=True, width="stretch")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 导出配置面板
    custom_title = "### 🎛️ Custom Export" if lang == 'en' else "### 🎛️ 自定义导出"
    st.markdown(custom_title)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        concepts_label = "📋 Select Concepts" if lang == 'en' else "📋 选择 Concepts"
        concepts_help = "Select data types to export" if lang == 'en' else "选择要导出的数据类型"
        concepts_to_export = st.multiselect(
            concepts_label,
            options=list(st.session_state.loaded_concepts.keys()),
            default=list(st.session_state.loaded_concepts.keys()),
            help=concepts_help
        )
    
    with col2:
        format_label = "📁 Export Format" if lang == 'en' else "📁 导出格式"
        format_help = "CSV: Universal format\nExcel: Multi-sheet support\nParquet: Efficient storage" if lang == 'en' else "CSV: 通用格式\nExcel: 支持多Sheet\nParquet: 高效存储"
        export_format = st.selectbox(
            format_label,
            options=['CSV', 'Excel', 'Parquet'],
            help=format_help
        )
        
        format_icons = {'CSV': '📄', 'Excel': '📊', 'Parquet': '⚡'}
        selected_text = "Selected" if lang == 'en' else "已选择"
        st.markdown(f"<small>{format_icons.get(export_format, '')} {selected_text} {export_format}</small>", unsafe_allow_html=True)
    
    with col3:
        merge_label = "📦 Merge Mode" if lang == 'en' else "📦 合并模式"
        merge_options = ['Separate Files', 'Merge Into One'] if lang == 'en' else ['分开保存', '合并为一个文件']
        merge_help = "Separate: One file per Concept\nMerge: All data in one file" if lang == 'en' else "分开: 每个Concept一个文件\n合并: 所有数据合并"
        merge_mode = st.selectbox(
            merge_label,
            options=merge_options,
            help=merge_help
        )
    
    # 高级选项
    adv_label = "⚙️ Advanced Options" if lang == 'en' else "⚙️ 高级选项"
    with st.expander(adv_label, expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            filter_label = "Filter by Patient" if lang == 'en' else "按患者过滤"
            filter_patient = st.checkbox(filter_label, value=False)
            if filter_patient and st.session_state.patient_ids:
                select_patients_label = "Select Patients" if lang == 'en' else "选择患者"
                selected_patients = st.multiselect(
                    select_patients_label,
                    options=st.session_state.patient_ids[:100],
                    default=st.session_state.patient_ids[:5]
                )
            else:
                selected_patients = None
        
        with col2:
            index_label = "Include Row Index" if lang == 'en' else "包含行索引"
            include_index = st.checkbox(index_label, value=False)
            timestamp_label = "Add Timestamp to Filename" if lang == 'en' else "文件名添加时间戳"
            add_timestamp = st.checkbox(timestamp_label, value=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 导出预览
    preview_title = "### 📋 Export Preview" if lang == 'en' else "### 📋 导出预览"
    st.markdown(preview_title)
    
    preview_data = {}
    total_rows = 0
    total_cols = 0
    
    for name in concepts_to_export:
        df = st.session_state.loaded_concepts[name]
        
        # 确保是 DataFrame
        if not isinstance(df, pd.DataFrame):
            continue
        
        if selected_patients and st.session_state.id_col in df.columns:
            df = df[df[st.session_state.id_col].isin(selected_patients)]
        
        preview_data[name] = df
        total_rows += len(df)
        total_cols = max(total_cols, len(df.columns))
    
    # 预览统计卡片
    total_records_label = "Total Records" if lang == 'en' else "总记录数"
    est_size_label = "Est. Size" if lang == 'en' else "预估大小"
    format_label_2 = "Format" if lang == 'en' else "格式"
    est_size = total_rows * total_cols * 10 / 1024
    size_str = f"{est_size:.0f} KB" if est_size < 1024 else f"{est_size/1024:.1f} MB"
    
    st.markdown(f'''
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:clamp(8px,.4rem + .4vw,16px);margin-bottom:16px">
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">Concepts</div>
            <div style="font-weight:700;color:#6366f1;font-size:1.4rem;margin-top:6px">{len(concepts_to_export)}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{total_records_label}</div>
            <div style="font-weight:700;color:#111827;font-size:1.4rem;margin-top:6px">{total_rows:,}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{est_size_label}</div>
            <div style="font-weight:700;color:#111827;font-size:1.2rem;margin-top:6px">{size_str}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{format_label_2}</div>
            <div style="font-weight:700;color:#111827;font-size:1.2rem;margin-top:6px">{export_format}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 数据预览表格
    if concepts_to_export:
        preview_exp_label = "👁️ Preview Data" if lang == 'en' else "👁️ 预览数据"
        with st.expander(preview_exp_label, expanded=False):
            select_preview_label = "Select Preview" if lang == 'en' else "选择预览"
            preview_concept = st.selectbox(select_preview_label, concepts_to_export)
            if preview_concept in preview_data:
                st.dataframe(preview_data[preview_concept].head(20), width="stretch", hide_index=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 导出按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        export_btn_label = "📥 Export Data" if lang == 'en' else "📥 导出数据"
        spinner_text = "Preparing export..." if lang == 'en' else "正在准备导出..."
        merge_single = "Merge Into One" if lang == 'en' else "合并为一个文件"
        
        if st.button(export_btn_label, type="primary", width="stretch"):
            with st.spinner(spinner_text):
                import io
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
                
                try:
                    filename_base = f"easyicu_export_{timestamp}" if timestamp else "easyicu_export"
                    
                    if export_format == 'CSV':
                        if merge_mode == merge_single:
                            combined = pd.concat(
                                [df.assign(concept=name) for name, df in preview_data.items()],
                                ignore_index=True
                            )
                            csv = combined.to_csv(index=include_index, encoding='utf-8-sig')  # 🔧 FIX: BOM编码防止中文乱码
                            dl_csv = "⬇️ Download CSV" if lang == 'en' else "⬇️ 下载 CSV"
                            st.download_button(
                                label=dl_csv,
                                data=csv,
                                file_name=f"{filename_base}.csv",
                                mime="text/csv",
                            )
                        else:
                            # 分开保存 - 创建 ZIP
                            import zipfile
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for name, df in preview_data.items():
                                    csv_data = df.to_csv(index=include_index, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
                                    zf.writestr(f"{name}.csv", csv_data)
                            
                            dl_zip = "⬇️ Download ZIP (Multiple CSVs)" if lang == 'en' else "⬇️ 下载 ZIP (多个CSV)"
                            st.download_button(
                                label=dl_zip,
                                data=zip_buffer.getvalue(),
                                file_name=f"{filename_base}.zip",
                                mime="application/zip",
                            )
                    
                    elif export_format == 'Excel':
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            if merge_mode == merge_single:
                                combined = pd.concat(
                                    [df.assign(concept=name) for name, df in preview_data.items()],
                                    ignore_index=True
                                )
                                combined.to_excel(writer, sheet_name='all_data', index=include_index)
                            else:
                                for name, df in preview_data.items():
                                    sheet_name = name[:31]  # Excel sheet name limit
                                    df.to_excel(writer, sheet_name=sheet_name, index=include_index)
                        
                        dl_excel = "⬇️ Download Excel" if lang == 'en' else "⬇️ 下载 Excel"
                        st.download_button(
                            label=dl_excel,
                            data=output.getvalue(),
                            file_name=f"{filename_base}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    
                    elif export_format == 'Parquet':
                        combined = pd.concat(
                            [df.assign(concept=name) for name, df in preview_data.items()],
                            ignore_index=True
                        )
                        output = io.BytesIO()
                        combined.to_parquet(output, index=include_index)
                        dl_parquet = "⬇️ Download Parquet" if lang == 'en' else "⬇️ 下载 Parquet"
                        st.download_button(
                            label=dl_parquet,
                            data=output.getvalue(),
                            file_name=f"{filename_base}.parquet",
                            mime="application/octet-stream",
                        )
                    
                    success_msg = "✅ Export ready! Click the button above to download" if lang == 'en' else "✅ 导出准备完成！点击上方按钮下载"
                    st.markdown(f'''
                    <div class="success-box">
                        {success_msg}
                    </div>
                    ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    err_msg = f"❌ Export failed: {e}" if lang == 'en' else f"❌ 导出失败: {e}"
                    st.error(err_msg)


def main():
    """主函数。"""
    init_session_state()
    
    # 获取入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')
    
    # ============ 入口页面：选择Demo或Real Data模式 ============
    if entry_mode == 'none':
        render_entry_page()
        return

    _apply_assistant_preset()
    _maybe_materialize_pending_preset()
    
    # ============ 进入具体模式后，显示完整应用 ============
    render_sidebar()
    
    # 处理CSV转换对话框
    if st.session_state.get('show_convert_dialog', False):
        render_convert_dialog()
    
    # 🔧 导出进度区域：优先使用 Guide: Complete 中创建的容器，否则创建备用容器
    # （实际导出在渲染 Home 页面后执行，确保 container 已创建）
    default_export_container = st.container()
    
    # ============ 顶部标题（精简现代风格） ============
    lang = st.session_state.get('language', 'en')

    # 用 badge 显示模式标识
    if entry_mode == 'demo':
        _mode_badge = '<span style="display:inline-block;background:var(--gradient-success);color:white;font-size:0.68rem;font-weight:700;padding:2px 10px;border-radius:100px;margin-left:8px;vertical-align:middle;letter-spacing:0.03em;">DEMO</span>'
    else:
        _mode_badge = ''

    if lang == 'en':
        st.markdown(f'<div class="main-header">🏥 EasyICU{_mode_badge}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">ICU Data Analytics Platform</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="main-header">🏥 EasyICU{_mode_badge}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">ICU 数据分析平台</div>', unsafe_allow_html=True)

    assistant_notice = st.session_state.pop('_assistant_notice', None)
    if assistant_notice:
        st.success(assistant_notice)
    
    # 主页面标签：Tutorial, Quick Visualization, Cohort Analysis
    tab1, tab2, tab3 = st.tabs([
        get_text('home'),
        get_text('quick_visualization'),
        get_text('cohort_compare'),
    ])
    
    with tab1:
        render_home()
    
    with tab2:
        # 处理侧边栏的 Preview 请求
        if st.session_state.get('_preview_requested', False):
            st.session_state['_preview_requested'] = False
            _preview_n = st.session_state.get('_preview_n', 10)
            _use_mock = st.session_state.get('use_mock_data', False)
            _sel_concepts = st.session_state.get('selected_concepts', [])
            
            if _use_mock:
                try:
                    with st.spinner(f"Generating preview with {_preview_n} mock patients..." if lang == 'en' else f"正在生成 {_preview_n} 位模拟患者的预览..."):
                        mock_data, preview_patient_ids = generate_mock_data(n_patients=_preview_n, hours=72)
                    if _sel_concepts:
                        mock_data = {k: v for k, v in mock_data.items() if k in _sel_concepts}
                    st.session_state.loaded_concepts = mock_data
                    st.session_state.loaded_data_origin = 'preview'
                    st.session_state.patient_ids = sorted(preview_patient_ids)
                    st.session_state.id_col = 'stay_id'
                    st.session_state.trigger_export = False
                    st.session_state['_exporting_in_progress'] = False
                    for _tmp_key in ['_skipped_modules', '_overwrite_modules', '_viz_import_export_auto_trigger']:
                        if _tmp_key in st.session_state:
                            del st.session_state[_tmp_key]
                    st.session_state['_viz_notices'] = [{
                        'level': 'success',
                        'message': f"✅ Preview loaded: {len(mock_data)} concepts, {len(st.session_state.patient_ids)} patients" if lang == 'en' else f"✅ 预览已加载: {len(mock_data)} 个概念, {len(st.session_state.patient_ids)} 位患者",
                    }]
                    st.session_state['_scroll_to_tab'] = 'viz'
                except Exception as e:
                    st.error(f"Preview error: {e}")
            else:
                _db = st.session_state.get('database', '')
                _data_path = st.session_state.get('data_path', '')
                if _db and _data_path:
                    try:
                        _preview_concepts = _sel_concepts
                        with st.spinner(f"Loading preview with {_preview_n} patients from {_db.upper()}..." if lang == 'en' else f"正在从 {_db.upper()} 加载 {_preview_n} 位患者的预览..."):
                            preview_result = load_preview_concepts(
                                concepts=_preview_concepts,
                                database=_db,
                                data_path=_data_path,
                                max_patients=_preview_n,
                                verbose=False,
                            )
                        st.session_state.loaded_concepts = preview_result.get('loaded_concepts', {})
                        st.session_state.loaded_data_origin = 'preview'
                        _unsupported_preview = preview_result.get('unsupported_concepts', [])
                        _id_map = {
                            'miiv': 'stay_id',
                            'mimic': 'icustay_id',
                            'eicu': 'patientunitstayid',
                            'aumc': 'admissionid',
                            'hirid': 'patientid',
                            'sic': 'CaseID',
                        }
                        st.session_state.id_col = _id_map.get(_db, 'stay_id')
                        _all_ids = set()
                        for _df in st.session_state.loaded_concepts.values():
                            if hasattr(_df, 'columns'):
                                for _ic in ['stay_id', 'patientunitstayid', 'hadm_id', 'admissionid', 'patientid', 'CaseID']:
                                    if _ic in _df.columns:
                                        _all_ids.update(_df[_ic].unique())
                                        break
                        st.session_state.patient_ids = sorted(_all_ids)
                        st.session_state.trigger_export = False
                        st.session_state['_exporting_in_progress'] = False
                        for _tmp_key in ['_skipped_modules', '_overwrite_modules', '_viz_import_export_auto_trigger']:
                            if _tmp_key in st.session_state:
                                del st.session_state[_tmp_key]
                        _viz_notices = []
                        if _unsupported_preview:
                            warn_prefix = "Skipped unsupported preview concepts" if lang == 'en' else "已跳过当前预览不支持的概念"
                            _viz_notices.append({
                                'level': 'warning',
                                'message': f"{warn_prefix}: {', '.join(_unsupported_preview[:8])}" + (" ..." if len(_unsupported_preview) > 8 else ""),
                            })
                        _viz_notices.append({
                            'level': 'success',
                            'message': f"✅ Preview: {len(st.session_state.loaded_concepts)} concepts, {len(st.session_state.patient_ids)} patients" if lang == 'en' else f"✅ 预览: {len(st.session_state.loaded_concepts)} 个概念, {len(st.session_state.patient_ids)} 位患者",
                        })
                        st.session_state['_viz_notices'] = _viz_notices
                        st.session_state['_scroll_to_tab'] = 'viz'
                    except Exception as e:
                        st.error(f"Preview error: {e}")
                else:
                    st.warning("Please configure data source first" if lang == 'en' else "请先配置数据源")
        
        render_quick_visualization_page()
    
    with tab3:
        render_cohort_comparison_page()
    
    # 🔧 处理侧边栏触发的导出（在标签页渲染后执行，确保 Guide: Complete 中的 container 已创建）
    if st.session_state.get('trigger_export', False):
        st.session_state.trigger_export = False
        # 🔧 FIX: 添加 try-except 防止白屏崩溃
        try:
            # 🔧 FIX: 使用 JavaScript 切换到 Tutorial 标签页（第1个标签）以显示导出进度
            js_switch_to_tutorial = '''
            <script>
                (function() {
                    // 滚动到页面顶部
                    var mainContainer = window.parent.document.querySelector('section.main');
                    if (mainContainer) mainContainer.scrollTop = 0;
                    window.parent.document.documentElement.scrollTop = 0;
                    window.parent.document.body.scrollTop = 0;
                    
                    // 点击第一个标签页 (Tutorial)
                    setTimeout(function() {
                        var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                        if (tabs && tabs.length >= 1) {
                            tabs[0].click();
                        }
                    }, 100);
                })();
            </script>
            '''
            st.components.v1.html(js_switch_to_tutorial, height=0)
            
            # 仅对真正的“已导出文件导入模式”自动回填 selected_concepts；Preview 不应触发该逻辑。
            if (
                not st.session_state.get('selected_concepts')
                and st.session_state.get('loaded_data_origin') == 'exported_files'
            ):
                loaded_concepts = st.session_state.get('loaded_concepts', {})
                if loaded_concepts:
                    st.session_state.selected_concepts = list(loaded_concepts.keys())
                    print(f"[DEBUG] main(): Auto-set selected_concepts from loaded_concepts: {len(st.session_state.selected_concepts)} concepts")
            
            # 🔧 只有在有选择的概念时才执行导出
            if st.session_state.get('selected_concepts'):
                # 优先使用 Guide: Complete 中创建的容器
                export_container = st.session_state.get('_export_progress_container', default_export_container)
                with export_container:
                    execute_sidebar_export()
            else:
                # 没有可导出的数据
                lang = st.session_state.get('language', 'en')
                st.warning("⚠️ No data to export. Please load data first." if lang == 'en' else "⚠️ 没有可导出的数据，请先加载数据。")
                st.session_state['_exporting_in_progress'] = False
        except Exception as e:
            import traceback
            lang = st.session_state.get('language', 'en')
            # 🔧 FIX: 打印详细错误堆栈便于调试
            error_detail = traceback.format_exc()
            print(f"[ERROR] Export failed with exception:\n{error_detail}")
            st.session_state['_exporting_in_progress'] = False
            if lang == 'en':
                st.error(f"❌ Export failed: {e}")
            else:
                st.error(f"❌ 导出失败: {e}")
            st.session_state['_exporting_in_progress'] = False
    
    # 🆕 处理页面跳转请求 - 在渲染完成后执行 JavaScript
    scroll_to_tab = st.session_state.pop('_scroll_to_tab', None)
    scroll_to_top = st.session_state.pop('_scroll_to_top', None)
    
    if scroll_to_tab == 'viz':
        # 跳转到 Quick Visualization 标签页（第2个标签，索引1）并滚动到顶部
        js_code = '''
        <script>
            (function() {
                // 滚动到页面顶部
                var mainContainer = window.parent.document.querySelector('section.main');
                if (mainContainer) mainContainer.scrollTop = 0;
                window.parent.document.documentElement.scrollTop = 0;
                window.parent.document.body.scrollTop = 0;
                
                // 点击第二个标签页
                setTimeout(function() {
                    var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    if (tabs && tabs.length >= 2) {
                        tabs[1].click();
                        // 再次滚动确保在顶部
                        setTimeout(function() {
                            var mainContainer = window.parent.document.querySelector('section.main');
                            if (mainContainer) mainContainer.scrollTop = 0;
                            window.parent.document.documentElement.scrollTop = 0;
                        }, 100);
                    }
                }, 200);
            })();
        </script>
        '''
        st.components.v1.html(js_code, height=0)
    elif scroll_to_tab == 'tutorial':
        # 跳转到 Tutorial 标签页（第1个标签，索引0）并滚动到顶部
        js_code = '''
        <script>
            (function() {
                var mainContainer = window.parent.document.querySelector('section.main');
                if (mainContainer) mainContainer.scrollTop = 0;
                window.parent.document.documentElement.scrollTop = 0;
                window.parent.document.body.scrollTop = 0;
                
                setTimeout(function() {
                    var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    if (tabs && tabs.length >= 1) {
                        tabs[0].click();
                        setTimeout(function() {
                            var mainContainer = window.parent.document.querySelector('section.main');
                            if (mainContainer) mainContainer.scrollTop = 0;
                            window.parent.document.documentElement.scrollTop = 0;
                        }, 120);
                    }
                }, 200);
            })();
        </script>
        '''
        st.components.v1.html(js_code, height=0)
    elif scroll_to_tab == 'home_dict':
        # 跳转到 Tutorial 标签页并滚动到数据字典锚点
        js_code = '''
        <script>
            (function() {
                function scrollToDictionary() {
                    var mainDoc = window.parent.document;
                    var mainContainer = mainDoc.querySelector('section.main');
                    var dictAnchor = mainDoc.getElementById('dictionary');
                    if (dictAnchor) {
                        dictAnchor.scrollIntoView({behavior: 'smooth', block: 'start'});
                        return true;
                    }

                    var headings = Array.from(mainDoc.querySelectorAll('h1, h2, h3, div, p, span'));
                    var dictHeading = headings.find(function(node) {
                        var text = (node.innerText || node.textContent || '').trim();
                        return text === '📖 Data Dictionary' ||
                               text === '📖 数据字典' ||
                               text === '📖 Complete Data Dictionary' ||
                               text === '📖 完整数据字典';
                    });
                    if (dictHeading) {
                        dictHeading.scrollIntoView({behavior: 'smooth', block: 'start'});
                        return true;
                    }

                    if (mainContainer) {
                        mainContainer.scrollTop = Math.max(mainContainer.scrollTop, 1800);
                    }
                    return false;
                }

                setTimeout(function() {
                    var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    if (tabs && tabs.length >= 1) {
                        tabs[0].click();
                        setTimeout(function() {
                            if (!scrollToDictionary()) {
                                setTimeout(scrollToDictionary, 300);
                                setTimeout(scrollToDictionary, 700);
                            }
                        }, 300);
                    }
                }, 200);
            })();
        </script>
        '''
        st.components.v1.html(js_code, height=0)
    elif scroll_to_tab == 'cohort':
        # 跳转到 Cohort Analysis 标签页（第3个标签，索引2）并滚动到顶部
        js_code = '''
        <script>
            (function() {
                var mainContainer = window.parent.document.querySelector('section.main');
                if (mainContainer) mainContainer.scrollTop = 0;
                window.parent.document.documentElement.scrollTop = 0;
                window.parent.document.body.scrollTop = 0;
                
                setTimeout(function() {
                    var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    if (tabs && tabs.length >= 3) {
                        tabs[2].click();
                        setTimeout(function() {
                            var mainContainer = window.parent.document.querySelector('section.main');
                            if (mainContainer) mainContainer.scrollTop = 0;
                            window.parent.document.documentElement.scrollTop = 0;
                        }, 100);
                    }
                }, 200);
            })();
        </script>
        '''
        st.components.v1.html(js_code, height=0)
    elif scroll_to_tab == 'ai_assistant':
        st.session_state['_floating_ai_open'] = True
        st.components.v1.html(
            '''
            <script>
                (function() {
                    var mainContainer = window.parent.document.querySelector('section.main');
                    if (mainContainer) mainContainer.scrollTop = 0;
                    window.parent.document.documentElement.scrollTop = 0;
                    window.parent.document.body.scrollTop = 0;
                })();
            </script>
            ''',
            height=0,
        )
    elif scroll_to_top:
        # 滚动到页面最顶部
        js_code = '''
        <script>
            (function() {
                // 尝试多种滚动方式确保生效
                var mainContainer = window.parent.document.querySelector('section.main');
                if (mainContainer) mainContainer.scrollTop = 0;
                window.parent.document.documentElement.scrollTop = 0;
                window.parent.document.body.scrollTop = 0;
                
                // 延迟再次滚动以确保页面完全加载后也在顶部
                setTimeout(function() {
                    var mainContainer = window.parent.document.querySelector('section.main');
                    if (mainContainer) mainContainer.scrollTop = 0;
                    window.parent.document.documentElement.scrollTop = 0;
                    window.parent.document.body.scrollTop = 0;
                }, 100);
            })();
        </script>
        '''
        st.components.v1.html(js_code, height=0)

    try:
        from easyicu.webapp.llm_chat import render_floating_chat_dock
        render_floating_chat_dock()
    except Exception:
        pass
    
    # 底部状态栏
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    footer_cols = st.columns([2, 2, 1])
    
    with footer_cols[0]:
        if st.session_state.language == 'en':
            data_status = "✅ Data Loaded" if len(st.session_state.loaded_concepts) > 0 else "⏳ No Data"
            patients_label = "Patients"
        else:
            data_status = "✅ 数据已加载" if len(st.session_state.loaded_concepts) > 0 else "⏳ 未加载数据"
            patients_label = "患者"
        # 🔧 FIX (2026-02-04): 统计唯一概念数
        n_concepts = count_unique_concepts(list(st.session_state.loaded_concepts.keys()))
        n_patients = len(st.session_state.patient_ids) if st.session_state.patient_ids else 0
        st.markdown(
            f"<small style='color:#888'>{data_status} | 📋 {n_concepts} Concepts | 👥 {n_patients} {patients_label}</small>",
            unsafe_allow_html=True
        )
    
    with footer_cols[1]:
        if st.session_state.get('selected_patient'):
            patient_label = "Current Patient" if st.session_state.language == 'en' else "当前患者"
            st.markdown(
                f"<small style='color:#888'>🎯 {patient_label}: {st.session_state.selected_patient}</small>",
                unsafe_allow_html=True
            )
    
    with footer_cols[2]:
        # 帮助按钮
        help_btn_text = "❓ Help" if st.session_state.language == 'en' else "❓ 帮助"
        with st.popover(help_btn_text):
            if st.session_state.language == 'en':
                st.markdown("""
                ### 🚀 Quick Start
                
                **📤 Data Extraction Mode**
                - **Step 1**: Select database & data path
                - **Step 2**: Filter cohort (age, LOS, etc.)
                - **Step 3**: Choose feature groups
                - **Step 4**: Export to CSV/Parquet/Excel
                
                **📊 Quick Visualization Mode**
                - Browse exported data folders
                - 📈 **Time Series**: Multi-patient trends
                - 🏥 **Patient View**: Single patient details
                - 📊 **Data Quality**: Completeness report
                
                **🔬 Cohort Analysis Mode**
                - Compare patient subgroups
                - Statistical analysis & hypothesis testing
                
                ---
                
                💡 **Tips**: 
                - Use sidebar tabs to extract features
                - Supports MIMIC-IV, eICU, AUMC, HiRID, MIMIC-III, SICdb
                - You can choose Demo Mode to explore EasyICU with simulated ICU data (no real data required)
                """)
            else:
                st.markdown("""
                ### 🚀 快速上手
                
                **📤 数据提取模式**
                - **步骤1**: 选择数据库和数据路径
                - **步骤2**: 筛选队列（年龄、住院时长等）
                - **步骤3**: 选择特征组
                - **步骤4**: 导出为 CSV/Parquet/Excel
                
                **📊 快速可视化模式**
                - 浏览已导出的数据文件夹
                - 📈 **时序分析**: 多患者趋势对比
                - 🏥 **患者视图**: 单患者详情
                - 📊 **数据质量**: 完整性报告
                
                **🔬 队列分析模式**
                - 比较患者亚组
                - 统计分析与假设检验
                
                ---
                
                💡 **提示**: 
                - 使用侧边栏标签提取特征
                - 支持 MIMIC-IV、eICU、AUMC、HiRID、MIMIC-III、SICdb
                - 可选择演示模式，使用模拟ICU数据快速体验EasyICU（无需真实数据）
                """)


if __name__ == "__main__":
    main()
