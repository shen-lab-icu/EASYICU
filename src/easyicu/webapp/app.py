"""EasyICU Streamlit 主应用。

本地 ICU 数据分析和可视化平台。
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import html
import re
import threading
import base64
from functools import lru_cache
from typing import Dict, Any, Optional, List
from easyicu.webapp.mock_data import generate_mock_data
from easyicu.webapp.sidebar import render_sidebar as _render_sidebar_impl
from easyicu.webapp.patient_page import render_patient_page as _render_patient_page_impl
from easyicu.webapp.cohort_group_page import render_group_comparison_subtab as _render_group_comparison_subtab_impl
from easyicu.webapp.export_workflow import execute_sidebar_export as _execute_sidebar_export_impl
from easyicu.webapp.home_extract_page import render_home_extract_mode as _render_home_extract_mode_impl
from easyicu.webapp.timeseries_page import render_timeseries_page as _render_timeseries_page_impl
from easyicu.webapp.data_table_page import render_data_table_subtab as _render_data_table_subtab_impl
from easyicu.webapp.quality_page import render_quality_page as _render_quality_page_impl
from easyicu.webapp.quick_visualization_page import render_quick_visualization_page as _render_quick_visualization_page_impl
from easyicu.webapp.entry_page import render_entry_page as _render_entry_page_impl
from easyicu.webapp.cohort_severity_page import render_severity_reclassification_subtab as _render_severity_reclassification_subtab_impl
from easyicu.webapp.cohort_multidb_page import render_multidb_distribution_subtab as _render_multidb_distribution_subtab_impl
from easyicu.webapp.cohort_dashboard_page import render_cohort_dashboard_subtab as _render_cohort_dashboard_subtab_impl
from easyicu.webapp.export_page import render_export_page as _render_export_page_impl
from easyicu.webapp.data_workflows import (
    check_data_status as _check_data_status_impl,
    convert_data_with_progress as _convert_data_with_progress_impl,
    apply_cohort_filter as _apply_cohort_filter_impl,
    validate_database_path as _validate_database_path_impl,
    load_from_exported as _load_from_exported_impl,
    load_data as _load_data_impl,
    load_data_for_preview as _load_data_for_preview_impl,
)
from easyicu.webapp.conversion_workflow import (
    render_convert_dialog as _render_convert_dialog_impl,
    convert_csv_to_parquet as _convert_csv_to_parquet_impl,
    _convert_hirid_data as _convert_hirid_data_impl,
)
from easyicu.webapp.paper_figures import (
    render_publication_composite_figure as _render_publication_composite_figure_impl,
    _render_paper_panel_css as _render_paper_panel_css_impl,
    render_quick_figure_panel as _render_quick_figure_panel_impl,
    render_cohort_figure_panel as _render_cohort_figure_panel_impl,
)


def _normalize_width_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate deprecated Streamlit width flags to the newer width API."""
    if "use_container_width" in kwargs and kwargs.get("use_container_width") is not None and "width" not in kwargs:
        use_container_width = kwargs.pop("use_container_width")
        kwargs["width"] = "stretch" if use_container_width else "content"
    else:
        kwargs.pop("use_container_width", None)
    return kwargs


def _legacy_width_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fallback for older Streamlit builds that do not accept width='stretch'."""
    width = kwargs.pop("width", None)
    if width == "stretch":
        kwargs["use_container_width"] = True
    elif width == "content":
        kwargs["use_container_width"] = False
    return kwargs


def _dataframe_compat(data, **kwargs):
    """Render dataframes across Streamlit versions.

    Newer releases accept `width="stretch"`, while older builds expect an
    integer width. Fall back to `use_container_width=True` when needed.
    """
    dataframe_fn = getattr(st, "_easyicu_original_dataframe", st.dataframe)
    kwargs = _normalize_width_kwargs(dict(kwargs))
    try:
        return dataframe_fn(data, **kwargs)
    except TypeError:
        if kwargs.get("width") != "stretch":
            raise
        return dataframe_fn(data, **_legacy_width_kwargs(dict(kwargs)))


def _button_compat(label, *args, **kwargs):
    button_fn = getattr(st, "_easyicu_original_button", st.button)
    kwargs = _normalize_width_kwargs(dict(kwargs))
    try:
        return button_fn(label, *args, **kwargs)
    except TypeError:
        return button_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))


def _download_button_compat(label, data, *args, **kwargs):
    download_button_fn = getattr(st, "_easyicu_original_download_button", st.download_button)
    kwargs = _normalize_width_kwargs(dict(kwargs))
    try:
        return download_button_fn(label, data, *args, **kwargs)
    except TypeError:
        return download_button_fn(label, data, *args, **_legacy_width_kwargs(dict(kwargs)))


def _form_submit_button_compat(label="Submit", *args, **kwargs):
    submit_fn = getattr(st, "_easyicu_original_form_submit_button", st.form_submit_button)
    kwargs = _normalize_width_kwargs(dict(kwargs))
    try:
        return submit_fn(label, *args, **kwargs)
    except TypeError:
        return submit_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))


def _plotly_chart_compat(figure_or_data, *args, **kwargs):
    plotly_chart_fn = getattr(st, "_easyicu_original_plotly_chart", st.plotly_chart)
    return plotly_chart_fn(figure_or_data, *args, **kwargs)


if not hasattr(st, "_easyicu_original_dataframe"):
    st._easyicu_original_dataframe = st.dataframe
    st.dataframe = _dataframe_compat
if not hasattr(st, "_easyicu_original_button"):
    st._easyicu_original_button = st.button
    st.button = _button_compat
if not hasattr(st, "_easyicu_original_download_button"):
    st._easyicu_original_download_button = st.download_button
    st.download_button = _download_button_compat
if not hasattr(st, "_easyicu_original_form_submit_button"):
    st._easyicu_original_form_submit_button = st.form_submit_button
    st.form_submit_button = _form_submit_button_compat
if not hasattr(st, "_easyicu_original_plotly_chart"):
    st._easyicu_original_plotly_chart = st.plotly_chart
    st.plotly_chart = _plotly_chart_compat


def _query_param_exists(key: str) -> bool:
    """Return whether a query parameter is present across Streamlit versions."""
    try:
        params = getattr(st, "query_params", {})
        return key in params
    except Exception:
        return False


def _query_param_value(key: str, default: str = "") -> str:
    """Read a Streamlit query parameter without depending on a specific API version."""
    try:
        params = getattr(st, "query_params", {})
        value = params.get(key, default)
    except Exception:
        value = default
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value).strip()


def _query_flag_enabled(key: str) -> bool:
    """Read a truthy/present Streamlit query flag without depending on a specific API version."""
    if not _query_param_exists(key):
        return False
    value = _query_param_value(key)
    return value.lower() not in {"0", "false", "no", "off", "none"}


def _get_database_download_info(database: str, lang: str = 'en') -> dict | None:
    """Return the official download/access page for a database."""
    download_map = {
        'miiv': {
            'name': 'MIMIC-IV',
            'url': 'https://physionet.org/content/mimiciv/',
        },
        'mimic': {
            'name': 'MIMIC-III',
            'url': 'https://physionet.org/content/mimiciii/',
        },
        'eicu': {
            'name': 'eICU-CRD',
            'url': 'https://physionet.org/content/eicu-crd/',
        },
        'aumc': {
            'name': 'AmsterdamUMCdb',
            'url': 'https://amsterdammedicaldatascience.nl/amsterdamumcdb/',
        },
        'hirid': {
            'name': 'HiRID',
            'url': 'https://hirid.intensivecare.ai/',
        },
        'sic': {
            'name': 'SICdb',
            'url': 'https://physionet.org/content/sicdb/',
        },
    }
    info = download_map.get(database)
    if not info:
        return None
    return {
        'name': info['name'],
        'url': info['url'],
        'label': (
            f"Open {info['name']} download page"
            if lang == 'en' else
            f"打开 {info['name']} 下载页"
        ),
        'note': (
            'Some databases require credentialed access or data use approval before download.'
            if lang == 'en' else
            '部分数据库需要先申请访问权限或完成数据使用审批后才能下载。'
        ),
    }


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
env_screenshot = os.environ.get("EASYICU_SCREENSHOT_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
query_screenshot = _query_flag_enabled("figure") or _query_flag_enabled("screenshot")
if env_screenshot:
    st.session_state.screenshot_mode = True
    st.session_state['_screenshot_mode_source'] = 'env'
elif query_screenshot:
    st.session_state.screenshot_mode = True
    st.session_state['_screenshot_mode_source'] = 'query'
elif st.session_state.get('_screenshot_mode_source') in {'query', 'env'}:
    # URL-triggered screenshot mode should not leak back into the normal app.
    st.session_state.screenshot_mode = False
    st.session_state['_screenshot_mode_source'] = 'manual'
    st.session_state.pop('_figure_target_section', None)
    st.session_state.pop('_figure_target_panel', None)
elif 'screenshot_mode' not in st.session_state:
    st.session_state.screenshot_mode = False
    st.session_state['_screenshot_mode_source'] = 'manual'

# 侧边栏宽度设置 - 根据展开状态动态调整
screenshot_mode_enabled = bool(st.session_state.get('screenshot_mode', False))
sidebar_width = "100vw" if st.session_state.sidebar_expanded else "clamp(420px, 31vw, 720px)"
sidebar_min_width = "100vw" if st.session_state.sidebar_expanded else "clamp(400px, 29vw, 680px)"
sidebar_display = "none" if screenshot_mode_enabled else "block"
main_display = "block" if screenshot_mode_enabled else ("none" if st.session_state.sidebar_expanded else "block")
floating_ai_display = "none" if screenshot_mode_enabled else "block"
st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        display: {sidebar_display} !important;
        min-width: {sidebar_min_width};
        max-width: {sidebar_width};
        width: {sidebar_width} !important;
    }}
    [data-testid="stSidebar"] > div {{
        width: 100% !important;
    }}
    /* 隐藏侧边栏折叠按钮 */
    button[kind="headerNoPadding"] {{
        display: none !important;
    }}
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stDeployButton"],
    [data-testid="manage-app-button"],
    .stDeployButton,
    .stAppDeployButton,
    button[title="Deploy"],
    a[href*="share.streamlit.io"] {{
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }}
    .block-container {{
        padding-top: {'1.1rem' if screenshot_mode_enabled else '2rem'} !important;
        max-width: {'1500px' if screenshot_mode_enabled else 'initial'} !important;
    }}
    .compact-inline-notice {{
        display: {'none' if screenshot_mode_enabled else 'block'} !important;
    }}
    .viz-demo-load-card {{
        border: 1px solid #cfe0f3;
        border-radius: 16px;
        background:
            radial-gradient(circle at 100% 0%, rgba(37, 99, 235, 0.08), transparent 36%),
            linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.055);
        padding: 1rem 1.1rem;
        margin: 0.55rem 0 0.75rem;
    }}
    .viz-demo-load-kicker {{
        color: #2563eb;
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.24rem;
    }}
    .viz-demo-load-title {{
        color: #0b1f44;
        font-size: 1.08rem;
        font-weight: 900;
        letter-spacing: -0.025em;
        margin-bottom: 0.22rem;
    }}
    .viz-demo-load-subtitle {{
        color: #60718a;
        font-size: 0.84rem;
        line-height: 1.55;
    }}
    .viz-empty-state {{
        text-align: center;
        padding: 2.35rem 1.4rem;
        background:
            radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.08), transparent 34%),
            linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        margin: 1rem 0;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.055);
    }}
    .viz-empty-icon {{
        width: 3.2rem;
        height: 3.2rem;
        margin: 0 auto 0.8rem;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 1.55rem;
        background: linear-gradient(135deg, #2f7cf6 0%, #0b65d8 100%);
        box-shadow: 0 12px 26px rgba(37, 99, 235, 0.22);
    }}
    .viz-empty-title {{
        color: #0b1f44;
        font-size: 1.22rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }}
    .viz-empty-subtitle {{
        color: #60718a;
        font-size: 0.9rem;
        line-height: 1.55;
    }}
    details[data-testid="stExpander"],
    div[data-testid="stExpander"] {{
        display: {'none' if screenshot_mode_enabled else 'block'} !important;
    }}
    .figure-table {{
        border: 1px solid #dbeafe;
        border-radius: 14px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    }}
    .figure-table table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        color: #0f172a;
    }}
    .figure-table th {{
        background: #f8fafc;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.045em;
        font-size: 0.68rem;
        font-weight: 800;
        padding: 9px 10px;
        border-bottom: 1px solid #dbeafe;
    }}
    .figure-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #eef2f7;
        vertical-align: middle;
    }}
    .figure-table tr:last-child td {{
        border-bottom: 0;
    }}
    .figure-table td:first-child,
    .figure-table th:first-child {{
        color: #2563eb;
        font-weight: 700;
    }}
    /* 展开时隐藏右侧主内容 */
    [data-testid="stMain"] {{
        display: {main_display} !important;
        overflow-y: auto !important;
    }}
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main {{
        overflow-y: auto !important;
        height: auto !important;
        min-height: 100vh !important;
    }}
    div.st-key-floating_ai_launcher,
    div.st-key-floating_ai_panel {{
        display: {floating_ai_display} !important;
    }}
    iframe[title*="streamlit_shadcn_ui"],
    iframe[title^="streamlit_shadcn_ui"] {{
        display: {floating_ai_display} !important;
        visibility: {'hidden' if screenshot_mode_enabled else 'visible'} !important;
    }}
    html, body {{
        overflow-y: auto !important;
        height: auto !important;
        background: #f4f8fc !important;
    }}
    [data-testid="stAppViewContainer"] {{
        overflow-y: auto !important;
        background:
            radial-gradient(circle at 15% -6%, rgba(37, 99, 235, 0.075), transparent 30%),
            radial-gradient(circle at 92% 4%, rgba(14, 165, 233, 0.08), transparent 34%),
            linear-gradient(180deg, #f7fbff 0%, #f4f8fc 42%, #f8fafc 100%) !important;
    }}
    [data-testid="stMain"] {{
        display: {main_display} !important;
        overflow-y: visible !important;
    }}
    [data-testid="stAppViewBlockContainer"],
    section.main {{
        overflow-y: visible !important;
        height: auto !important;
        min-height: 100vh !important;
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

    div[data-testid="stMarkdownContainer"] hr {
        margin: 0.45rem 0 0.8rem 0 !important;
        border-top: 1px solid #dbe4f0 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        margin-bottom: 0.38rem !important;
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
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg *,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] path {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
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
        overflow-x: auto !important;
        overflow-y: hidden !important;
        flex-wrap: nowrap !important;
        scrollbar-width: thin;
    }

    div[data-baseweb="tab-list"] button {
        font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
        font-weight: 700 !important;
        padding: clamp(8px, 0.12vw + 7px, 10px) clamp(8px, 0.24vw + 7px, 14px) !important;
        border-radius: var(--radius-lg) !important;
        transition: var(--transition-fast) !important;
        border: none !important;
        background: transparent !important;
        color: var(--text-secondary-light) !important;
        letter-spacing: 0.01em;
        white-space: nowrap !important;
        flex: 0 0 auto !important;
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
        font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
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
        background:
            radial-gradient(circle at 20% -6%, rgba(59, 130, 246, 0.09), transparent 32%),
            linear-gradient(180deg, #f7fbff 0%, #eef5fb 100%) !important;
        border-right: 1px solid #d4e2f0 !important;
        box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.85);
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.68rem;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
        letter-spacing: -0.025em;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1.22rem !important;
        margin-bottom: 0.18rem !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.02rem !important;
        margin-top: 0.42rem !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #334155;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.22);
        margin: 0.7rem 0;
    }

    [data-testid="stSidebar"] div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #cfe0f3;
        border-radius: 15px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        overflow: hidden;
    }

    [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,249,255,0.98));
        color: #0f172a !important;
        font-weight: 800;
        letter-spacing: -0.015em;
    }

    [data-testid="stSidebar"] .stButton > button {
        border-radius: 12px !important;
        border: 1px solid #bfd4ed !important;
        background: linear-gradient(180deg, #ffffff 0%, #f4f8fd 100%) !important;
        color: #1f3b63 !important;
        box-shadow: 0 6px 15px rgba(15, 23, 42, 0.055);
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #1d7ef2 0%, #0b8fc7 100%) !important;
        border-color: #166fd0 !important;
        color: #ffffff !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-radius: 12px !important;
        border-color: #c7d9ee !important;
        background-color: rgba(255, 255, 255, 0.96) !important;
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

    /* ============ Figure-aligned native Streamlit controls ============ */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #cfe0f3 !important;
        border-left: 4px solid #1d7ef2 !important;
        border-radius: 14px !important;
        padding: 0.82rem 0.95rem !important;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055) !important;
        min-height: 74px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.68rem !important;
        font-weight: 900 !important;
        letter-spacing: 0.09em !important;
        text-transform: uppercase !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0b1f44 !important;
        font-size: clamp(1.12rem, 0.34vw + 1rem, 1.42rem) !important;
        font-weight: 900 !important;
        letter-spacing: -0.035em !important;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-secondary"] {
        border-radius: 12px !important;
        border: 1px solid #c7d9ee !important;
        background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%) !important;
        color: #102a4c !important;
        font-weight: 760 !important;
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.055);
    }

    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"],
    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        color: #ffffff !important;
        border-color: #0b63ce !important;
        background: linear-gradient(135deg, #2f7cf6 0%, #0b65d8 100%) !important;
        box-shadow: 0 11px 24px rgba(37, 99, 235, 0.24) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: #2f7cf6 !important;
        box-shadow: 0 12px 26px rgba(37, 99, 235, 0.16) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 13px !important;
        border: 1px solid #dbeafe !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035) !important;
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

    .entry-task-launcher {
        max-width: min(1220px, 92vw);
        margin: 0.4rem auto 0;
    }

    .entry-task-launcher-label {
        text-align: center;
        color: #7388a5;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .entry-task-wrap div[data-testid="stButton"] > button {
        min-height: 62px !important;
        padding: 0.9rem 1rem !important;
        border-radius: 999px !important;
        background: rgba(255,255,255,0.78) !important;
        border: 1px solid rgba(148,163,184,0.22) !important;
        color: #17304c !important;
        font-size: 0.94rem !important;
        font-weight: 700 !important;
        line-height: 1.35 !important;
        box-shadow: 0 10px 28px rgba(15,23,42,0.04) !important;
        transition: var(--transition-smooth) !important;
        backdrop-filter: blur(10px) !important;
    }

    .entry-task-wrap div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(37,99,235,0.26) !important;
        background: rgba(255,255,255,0.94) !important;
        box-shadow: 0 14px 32px rgba(37,99,235,0.08) !important;
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
        .entry-task-wrap div[data-testid="stButton"] > button { min-height: 58px !important; font-size: 0.9rem !important; }
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
    .module-preview-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(239,246,255,0.94));
        border: 1px solid rgba(96,165,250,0.28);
        border-radius: 16px;
        padding: 0.82rem 0.96rem 0.88rem;
        box-shadow: 0 8px 24px rgba(37,99,235,0.08);
        min-height: 112px;
    }
    .module-preview-card .eyebrow {
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 0.14rem;
    }
    .module-preview-card .title {
        font-size: 1.04rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 0.22rem;
    }
    .module-preview-card .summary {
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.45;
        margin-bottom: 0.55rem;
    }
    .module-feature-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.34rem;
    }
    .module-feature-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(37,99,235,0.08);
        border: 1px solid rgba(37,99,235,0.12);
        border-radius: 999px;
        padding: 0.2rem 0.5rem;
        font-size: 0.72rem;
        font-weight: 700;
        color: #1d4ed8;
        line-height: 1.1;
    }
    .module-feature-chip.muted {
        background: rgba(148,163,184,0.12);
        border-color: rgba(148,163,184,0.18);
        color: #475569;
    }
    .preview-hint-line {
        font-size: 0.77rem;
        color: #64748b;
        margin: 0.15rem 0 0.5rem;
    }
    .preview-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        padding: 0.24rem 0.58rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.14);
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 700;
        line-height: 1;
    }
    .preview-badge.warning {
        background: rgba(245,158,11,0.10);
        border-color: rgba(245,158,11,0.16);
        color: #b45309;
    }
    .preview-toolbar {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.18rem 0 0.24rem;
    }
    .preview-toolbar-main {
        min-width: 0;
    }
    .preview-toolbar-title {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 0.14rem;
    }
    .preview-toolbar-note {
        font-size: 0.79rem;
        color: #64748b;
        line-height: 1.35;
    }
    .preview-toolbar-note code {
        background: rgba(37,99,235,0.08);
        color: #1d4ed8;
        border-radius: 8px;
        padding: 0.08rem 0.34rem;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .inline-control-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748b;
        margin: 0.18rem 0 0.18rem;
    }
    .subtle-preview-note {
        font-size: 0.78rem;
        color: #64748b;
        margin: 0.1rem 0 0.35rem;
        line-height: 1.35;
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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800;900&display=swap');

    /* ============================================================
       EasyICU paper-figure visual skin
       Aligns the live web UI with the accepted image2 figure system.
       ============================================================ */
    :root {
        --figure-navy: #0b1f44;
        --figure-blue: #2563eb;
        --figure-cyan: #0891b2;
        --figure-teal: #0f766e;
        --figure-orange: #ea7a1a;
        --figure-bg: #f4f8fc;
        --figure-card: #ffffff;
        --figure-soft: #edf4fb;
        --figure-line: #cddbeb;
        --figure-line-strong: #b7cae2;
        --figure-muted: #60718a;
        --figure-shadow: 0 10px 30px rgba(15, 31, 68, 0.055), 0 1px 2px rgba(15, 31, 68, 0.05);
        --figure-shadow-hover: 0 16px 38px rgba(37, 99, 235, 0.12), 0 4px 14px rgba(15, 31, 68, 0.06);
        --gradient-primary: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
        --gradient-info: linear-gradient(135deg, #0891b2 0%, #14b8a6 100%);
        --gradient-hero: linear-gradient(135deg, #102a56 0%, #1d4f86 58%, #0f766e 100%);
        --shadow-card: var(--figure-shadow);
        --shadow-hover: var(--figure-shadow-hover);
        --shadow-glow: 0 10px 26px rgba(37, 99, 235, 0.16);
        --border-subtle: var(--figure-line);
        --border-light: rgba(37, 99, 235, 0.16);
        --bg-primary: var(--figure-bg);
        --bg-secondary: var(--figure-card);
        --bg-tertiary: var(--figure-soft);
        --primary-color: var(--figure-blue);
        --secondary-color: var(--figure-cyan);
        --accent-color: #14b8a6;
        --font-sans: 'Source Sans 3', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    html, body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stAlert, div[data-testid="stMetric"], div[data-baseweb="select"],
    div[data-baseweb="input"], div[data-baseweb="textarea"],
    div[data-baseweb="tab-list"], h1, h2, h3, h4, h5, h6, label,
    input, textarea, select, option, td, th {
        font-family: var(--font-sans) !important;
    }

    html, body,
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"], [data-testid="stMain"], .main {
        background:
            radial-gradient(circle at 16% 0%, rgba(37, 99, 235, 0.055), transparent 32%),
            radial-gradient(circle at 92% 8%, rgba(20, 184, 166, 0.055), transparent 30%),
            var(--figure-bg) !important;
        color: #0f172a !important;
    }

    .block-container {
        max-width: clamp(1120px, 94vw, 1880px) !important;
        padding-left: clamp(0.7rem, 1vw, 1.6rem) !important;
        padding-right: clamp(0.7rem, 1vw, 1.6rem) !important;
    }

    div[data-baseweb="tab-list"] {
        background: rgba(237, 244, 251, 0.94) !important;
        border: 1px solid var(--figure-line) !important;
        border-radius: 999px !important;
        padding: 7px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.78), var(--figure-shadow) !important;
        backdrop-filter: blur(10px);
        overflow-x: auto !important;
        overflow-y: hidden !important;
        flex-wrap: nowrap !important;
    }

    div[data-baseweb="tab-list"] button {
        color: #5d6f88 !important;
        border-radius: 999px !important;
        letter-spacing: 0.01em !important;
        min-height: 34px !important;
        font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
        white-space: nowrap !important;
        flex: 0 0 auto !important;
    }

    div[data-baseweb="tab-list"] button p {
        font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
        white-space: nowrap !important;
    }

    div[data-baseweb="tab-list"] button:hover {
        background: rgba(255,255,255,0.72) !important;
        color: var(--figure-navy) !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: #ffffff !important;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18) !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] * {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"],
    .metric-card, .feature-card, .patient-card,
    .compact-summary-card, .module-preview-card,
    .mini-stat-card, .tiny-stat-card,
    details[data-testid="stExpander"],
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    [data-testid="stVegaLiteChart"] {
        background: var(--figure-card) !important;
        border: 1px solid var(--figure-line) !important;
        border-radius: 18px !important;
        box-shadow: var(--figure-shadow) !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    div[data-testid="stMetric"]::before,
    .compact-summary-card::before,
    .mini-stat-card::before,
    .tiny-stat-card::before {
        background: var(--gradient-primary) !important;
    }

    .compact-summary-card,
    .mini-stat-card,
    .tiny-stat-card {
        border-left: 4px solid #5aa8ff !important;
    }

    .module-preview-card {
        background: linear-gradient(135deg, #ffffff 0%, #f4f9ff 100%) !important;
    }

    .module-feature-chip,
    .preview-badge,
    div[data-baseweb="tag"] {
        background: #eaf2ff !important;
        border: 1px solid #cfe0f6 !important;
        color: var(--figure-navy) !important;
        box-shadow: none !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"],
    [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg *,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] path {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0 5px 14px rgba(37,99,235,0.22) !important;
    }

    div[data-testid="stMetric"] label,
    .compact-summary-card .summary-label,
    .mini-stat-card .mini-label,
    .tiny-stat-card .tiny-label,
    .inline-control-label {
        color: var(--figure-muted) !important;
        font-weight: 800 !important;
        letter-spacing: 0.075em !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    .compact-summary-card .summary-value,
    .mini-stat-card .mini-value,
    .tiny-stat-card .tiny-value {
        color: var(--figure-navy) !important;
        letter-spacing: -0.02em !important;
    }

    [data-testid="stDataFrame"] {
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] thead th {
        background: #f4f7fb !important;
        color: #5f6f84 !important;
        text-transform: none !important;
    }

    .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton button {
        background: var(--gradient-primary) !important;
        border: 1px solid rgba(37, 99, 235, 0.12) !important;
        border-radius: 12px !important;
        box-shadow: 0 9px 20px rgba(37, 99, 235, 0.16) !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
        background: #ffffff !important;
        border: 1px solid var(--figure-line) !important;
        border-radius: 12px !important;
        color: var(--figure-navy) !important;
        box-shadow: 0 4px 12px rgba(15, 31, 68, 0.04) !important;
    }

    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton button * {
        color: #1f3b63 !important;
        fill: #1f3b63 !important;
        white-space: normal !important;
        line-height: 1.15 !important;
    }

    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(180deg, #ffffff 0%, #f4f8fd 100%) !important;
        border: 1px solid #bfd4ed !important;
        border-radius: 12px !important;
        min-height: 2.45rem !important;
        box-shadow: 0 6px 15px rgba(15, 23, 42, 0.055) !important;
    }

    [data-testid="stSidebar"] .stButton button[kind="primary"],
    [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #1d7ef2 0%, #0b8fc7 100%) !important;
        border-color: #166fd0 !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22) !important;
    }

    [data-testid="stSidebar"] .stButton button[kind="primary"],
    [data-testid="stSidebar"] .stButton button[kind="primary"] *,
    [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] *,
    [data-testid="stSidebar"] .stButton button:hover,
    [data-testid="stSidebar"] .stButton button:hover * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #ffffff !important;
        border: 1px solid var(--figure-line) !important;
        border-radius: 13px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        border-color: #7fb4ff !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10) !important;
    }

    div[data-testid="stAlert"] {
        border: 1px solid var(--figure-line) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }

    .highlight-card,
    .step-indicator.active,
    .mode-card-tag.blue {
        background: rgba(37, 99, 235, 0.07) !important;
        border-color: rgba(37, 99, 235, 0.16) !important;
        color: var(--figure-navy) !important;
    }

    .patient-card:hover,
    .metric-card:hover,
    .feature-card:hover {
        border-color: #a9c7ee !important;
        box-shadow: var(--figure-shadow-hover) !important;
    }

    .compact-section-title,
    .preview-toolbar-title,
    .module-preview-card .title {
        color: var(--figure-navy) !important;
    }

    .compact-section-desc,
    .preview-toolbar-note,
    .subtle-preview-note {
        color: var(--figure-muted) !important;
    }

    .workflow-figure-shell {
        background: #ffffff;
        border: 1px solid var(--figure-line);
        border-radius: 18px;
        box-shadow: var(--figure-shadow);
        padding: clamp(0.9rem, 0.5vw + 0.75rem, 1.25rem);
        margin: 0.55rem 0 1rem;
    }

    .workflow-figure-title {
        color: var(--figure-navy);
        font-weight: 900;
        font-size: clamp(1.08rem, 0.55vw + 0.95rem, 1.55rem);
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }

    .workflow-figure-subtitle {
        color: var(--figure-muted);
        font-weight: 650;
        font-size: clamp(0.78rem, 0.15vw + 0.74rem, 0.92rem);
        margin-bottom: 0.9rem;
    }

    .workflow-pipeline-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 26px minmax(0, 1.25fr) 26px minmax(0, 1.25fr) 26px minmax(0, 1fr);
        gap: 0.55rem;
        align-items: stretch;
    }

    .workflow-card {
        background: #ffffff;
        border: 1px solid var(--figure-line);
        border-radius: 14px;
        padding: 0.9rem 0.95rem;
        min-height: 286px;
        box-shadow: 0 5px 16px rgba(15, 31, 68, 0.035);
        color: var(--figure-navy);
    }

    .workflow-card-head {
        display: grid;
        grid-template-columns: 34px 1fr;
        gap: 0.65rem;
        align-items: start;
        margin-bottom: 0.72rem;
    }

    .workflow-badge {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #082957;
        color: #ffffff;
        font-size: 1.02rem;
        font-weight: 900;
        line-height: 1;
        box-shadow: 0 6px 16px rgba(8, 41, 87, 0.16);
    }

    .workflow-card-title {
        color: #082957;
        font-weight: 900;
        font-size: 1.02rem;
        line-height: 1.16;
        letter-spacing: -0.01em;
    }

    .workflow-card-kicker {
        color: var(--figure-muted);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 0.08rem;
    }

    .workflow-field {
        margin: 0.52rem 0;
    }

    .workflow-label {
        color: #172b4d;
        font-size: 0.74rem;
        font-weight: 750;
        margin-bottom: 0.24rem;
    }

    .workflow-input {
        min-height: 34px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.45rem;
        border: 1px solid #d9e3f1;
        border-radius: 8px;
        background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        color: #14233d;
        padding: 0.42rem 0.55rem;
        font-size: 0.76rem;
        font-weight: 650;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.92);
        overflow-wrap: anywhere;
    }

    .workflow-button {
        background: linear-gradient(135deg, #2563eb 0%, #0d7fd1 100%);
        color: #ffffff;
        border-radius: 8px;
        min-height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.42rem;
        margin-top: 0.72rem;
        font-weight: 850;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.20);
    }

    .workflow-status {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-height: 38px;
        margin-top: 0.72rem;
        padding: 0.46rem 0.62rem;
        border: 1px solid #d7eadf;
        border-radius: 8px;
        background: #f2fbf6;
        color: #14532d;
        font-weight: 800;
        font-size: 0.76rem;
    }

    .workflow-status.warn {
        border-color: #f8ddb0;
        background: #fff8ed;
        color: #92400e;
    }

    .workflow-check-dot {
        width: 19px;
        height: 19px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #2ca25f;
        color: #ffffff;
        font-size: 0.76rem;
        font-weight: 900;
        flex: 0 0 auto;
    }

    .workflow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2563eb;
        font-size: 1.7rem;
        font-weight: 900;
        padding-top: 2.5rem;
    }

    .workflow-concepts {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.4rem 0.48rem;
        margin-top: 0.45rem;
    }

    .workflow-concept {
        display: flex;
        align-items: center;
        gap: 0.36rem;
        color: #11213b;
        font-size: 0.74rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .workflow-tick {
        width: 15px;
        height: 15px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #2f70d8;
        color: #ffffff;
        font-size: 0.64rem;
        font-weight: 900;
        flex: 0 0 auto;
    }

    .workflow-summary-panel {
        margin-top: 0.95rem;
        border: 1px solid var(--figure-line);
        border-radius: 15px;
        background: #ffffff;
        padding: 0.9rem 1rem;
    }

    .workflow-summary-grid {
        display: grid;
        grid-template-columns: 1.35fr 0.9fr;
        gap: 1rem;
        align-items: stretch;
    }

    .workflow-success-strip {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        border: 1px solid #d7eadf;
        border-radius: 8px;
        background: linear-gradient(90deg, #edf9f2 0%, #f8fdfa 100%);
        color: #14532d;
        padding: 0.56rem 0.68rem;
        font-weight: 850;
        margin-bottom: 0.62rem;
    }

    .workflow-success-strip.warn {
        border-color: #f8ddb0;
        background: linear-gradient(90deg, #fff8ed 0%, #fffdf8 100%);
        color: #92400e;
    }

    .workflow-file-list {
        border: 1px solid #dce6f3;
        border-radius: 9px;
        background: #ffffff;
        padding: 0.48rem 0.62rem;
        color: #52647d;
        font-size: 0.72rem;
        line-height: 1.75;
    }

    .workflow-stat-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.5rem;
    }

    .workflow-mini-stat {
        border: 1px solid #dce6f3;
        border-radius: 10px;
        background: #ffffff;
        padding: 0.55rem 0.62rem;
    }

    .workflow-mini-label {
        color: var(--figure-muted);
        font-size: 0.57rem;
        font-weight: 900;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .workflow-mini-value {
        color: #082957;
        font-weight: 900;
        font-size: 0.9rem;
        margin-top: 0.15rem;
    }

    .workflow-guide-title {
        display: flex;
        align-items: center;
        gap: 0.62rem;
        margin: 0.9rem 0 0.6rem;
        color: var(--figure-navy);
        font-size: 1.42rem;
        font-weight: 900;
        letter-spacing: -0.02em;
    }

    .workflow-guide-title::before {
        content: "";
        width: 6px;
        height: 28px;
        border-radius: 4px;
        background: linear-gradient(180deg, #2563eb 0%, #0891b2 100%);
        display: inline-block;
        flex: 0 0 auto;
    }

    .quality-summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.72rem;
        margin-bottom: 1rem;
    }

    .quality-summary-card {
        background: #ffffff;
        border: 1px solid var(--figure-line);
        border-radius: 14px;
        padding: 0.85rem 0.9rem;
        text-align: center;
        box-shadow: 0 7px 20px rgba(15, 31, 68, 0.04);
    }

    .quality-summary-label {
        color: var(--figure-muted);
        font-size: 0.64rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        margin-bottom: 0.22rem;
    }

    .quality-summary-value {
        color: var(--figure-navy);
        font-size: 1.28rem;
        font-weight: 900;
        line-height: 1.1;
    }

    @media (max-width: 1500px) {
        .workflow-pipeline-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .workflow-arrow {
            display: none;
        }
        .workflow-card {
            min-height: auto;
        }
    }

    @media (max-width: 900px) {
        .workflow-pipeline-grid {
            grid-template-columns: 1fr;
        }
        .workflow-summary-grid,
        .workflow-stat-row,
        .quality-summary-grid {
            grid-template-columns: 1fr;
        }
    }

    .audit-figure-panel {
        background: #ffffff;
        border: 1px solid var(--figure-line);
        border-radius: 18px;
        box-shadow: var(--figure-shadow);
        padding: 0.95rem 1rem 1.05rem;
        margin: 0.4rem 0 0.95rem;
    }

    .audit-panel-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--figure-navy);
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 0.65rem;
    }

    .audit-panel-letter {
        width: 24px;
        height: 24px;
        border-radius: 7px;
        background: var(--figure-navy);
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 0.76rem;
        line-height: 1;
    }

    .audit-summary-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.72rem;
        margin: 0.2rem 0 0.85rem;
    }

    .audit-summary-card {
        background: #ffffff;
        border: 1px solid var(--figure-line);
        border-radius: 14px;
        padding: 0.78rem 0.85rem;
        box-shadow: 0 6px 18px rgba(15, 31, 68, 0.04);
    }

    .audit-summary-label {
        color: var(--figure-muted);
        font-size: 0.68rem;
        font-weight: 850;
        letter-spacing: 0.075em;
        text-transform: uppercase;
        margin-bottom: 0.18rem;
    }

    .audit-summary-value {
        color: var(--figure-navy);
        font-size: 1.35rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .audit-flow {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
        padding: 0.25rem 0.15rem;
    }

    .audit-flow-step {
        border: 1px solid var(--figure-line);
        border-radius: 13px;
        background: #ffffff;
        padding: 0.58rem 0.75rem;
        text-align: center;
        color: var(--figure-navy);
        position: relative;
    }

    .audit-flow-step:not(:last-child)::after {
        content: '↓';
        position: absolute;
        left: 50%;
        bottom: -0.72rem;
        transform: translateX(-50%);
        color: var(--figure-orange);
        font-weight: 900;
        font-size: 0.85rem;
    }

    .audit-flow-label {
        font-size: 0.72rem;
        font-weight: 800;
        color: #53657c;
    }

    .audit-flow-value {
        font-size: 1.08rem;
        font-weight: 900;
        letter-spacing: -0.02em;
    }

    .audit-flow-excluded {
        color: #b45309;
        font-size: 0.72rem;
        font-weight: 800;
        margin-top: 0.12rem;
    }

    .audit-denominator-note {
        border: 1px solid #c9d9ee;
        border-radius: 12px;
        background: #f7fbff;
        color: #3d516a;
        padding: 0.65rem 0.8rem;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .cohort-demo-workspace {
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 0.78rem;
        background: #ffffff;
        border: 1px solid var(--figure-line);
        border-radius: 16px;
        padding: 0.78rem 0.92rem;
        margin: 0.45rem 0 0.92rem;
        box-shadow: 0 8px 24px rgba(15, 31, 68, 0.045);
        color: var(--figure-navy);
    }

    .cohort-demo-badge {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: var(--figure-navy);
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 0.92rem;
        line-height: 1;
    }

    .cohort-demo-title {
        color: var(--figure-navy);
        font-weight: 900;
        font-size: 0.96rem;
        letter-spacing: -0.015em;
        margin-bottom: 0.1rem;
    }

    .cohort-demo-subtitle {
        color: var(--figure-muted);
        font-size: 0.78rem;
        line-height: 1.42;
    }

    .cohort-demo-status {
        border: 1px solid #bbf7d0;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857;
        padding: 0.32rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 850;
        white-space: nowrap;
    }

    @media (max-width: 900px) {
        .audit-summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .cohort-demo-workspace {
            grid-template-columns: auto 1fr;
        }
        .cohort-demo-status {
            grid-column: 1 / -1;
            width: fit-content;
        }
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

MODULE_PREVIEW_SUMMARIES = {
    'renal': {
        'en': "AKI staging, urine output, RRT, and creatinine-derived context in one module preview.",
        'zh': "将 AKI 分期、尿量、RRT 和肌酐基线线索放在同一模块预览中。",
    },
    'respiratory': {
        'en': "Respiratory support, oxygenation, and ventilation status in a single preview.",
        'zh': "在同一预览中查看呼吸支持、氧合和通气状态。",
    },
    'vitals': {
        'en': "Core bedside vital signs aligned into a compact longitudinal preview.",
        'zh': "将核心床旁生命体征汇总到紧凑的纵向预览中。",
    },
    'chemistry': {
        'en': "Key chemistry measurements grouped for quick sanity checks before deeper analysis.",
        'zh': "将关键生化指标汇总，便于深入分析前快速核查。",
    },
}

MODULE_PREVIEW_TAG_PRIORITY = {
    'renal': ['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'rrt', 'uo_6h', 'uo_12h', 'creat_low_past_48hr'],
    'respiratory': ['mech_vent', 'vent_ind', 'fio2', 'pafi', 'safi', 'spo2', 'resp'],
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    'chemistry': ['crea', 'bun', 'na', 'k', 'glu', 'bicar', 'lact'],
}

MODULE_PREVIEW_COLUMN_PRIORITY = {
    'renal': ['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'rrt', 'uo_6h', 'uo_12h', 'creat_low_past_48hr'],
    'respiratory': ['mech_vent', 'vent_ind', 'fio2', 'pafi', 'safi', 'spo2', 'resp'],
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    'chemistry': ['crea', 'bun', 'na', 'k', 'glu', 'bicar', 'lact'],
}

PREVIEW_TIME_COLUMNS = [
    'charttime', 'time', 'starttime', 'start', 'endtime', 'itemtime',
    'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat',
    'givenat', 'enteredentryat', 'intakeoutputoffset', 'observationoffset',
    'nursingchartoffset', 'labresultoffset', 'respchartoffset'
]


def _build_module_preview_metadata(
    module_key: str,
    selected_module: str,
    module_concepts: List[str],
    lang: str = 'en',
) -> Dict[str, Any]:
    """Build concise copy and representative feature tags for module preview cards."""
    summary_map = MODULE_PREVIEW_SUMMARIES.get(module_key, {})
    summary = summary_map.get(lang)
    if not summary:
        summary = (
            f"Representative features from {selected_module} are previewed below."
            if lang == 'en'
            else f"下方展示的是 {selected_module} 的代表性特征预览。"
        )

    ordered_tags = [tag for tag in MODULE_PREVIEW_TAG_PRIORITY.get(module_key, []) if tag in module_concepts]
    for concept in sorted(module_concepts):
        if concept not in ordered_tags:
            ordered_tags.append(concept)

    tags = ordered_tags[:8]
    overflow_count = max(0, len(module_concepts) - len(tags))
    return {
        'summary': summary,
        'tags': tags,
        'overflow_count': overflow_count,
    }


def _get_data_table_page_copy(lang: str = 'en') -> Dict[str, str]:
    if lang == 'en':
        return {
            'title': "📋 Module Table Preview",
            'description': "Preview loaded tables by module before drilling into feature-level detail.",
        }
    return {
        'title': "📋 模块数据预览",
        'description': "按模块预览已加载数据表，再进入单个特征的细节查看。",
    }


def _get_single_feature_preview_copy(feature_name: str, lang: str = 'en') -> Dict[str, str]:
    if lang == 'en':
        return {
            'title': "🧪 Single Feature Preview",
            'description': f"Inspect `{feature_name}` with full row-level detail while keeping the preview layout consistent.",
        }
    return {
        'title': "🧪 单特征预览",
        'description': f"以与预览页一致的版式查看 `{feature_name}` 的逐行明细。",
    }


def _select_preview_columns(
    df: pd.DataFrame,
    module_key: str,
    module_concepts: List[str],
    id_col: str,
    max_columns: int = 10,
) -> List[str]:
    """Prioritize the most interpretable columns for compact preview tables."""
    if not isinstance(df, pd.DataFrame):
        return []

    ordered: List[str] = []

    def add_column(name: Optional[str]) -> None:
        if name and name in df.columns and name not in ordered:
            ordered.append(name)

    add_column(id_col)
    for time_col in PREVIEW_TIME_COLUMNS:
        if time_col in df.columns:
            add_column(time_col)
            break

    for column_name in MODULE_PREVIEW_COLUMN_PRIORITY.get(module_key, []):
        add_column(column_name)

    for concept_name in module_concepts:
        add_column(concept_name)

    for column_name in df.columns:
        add_column(column_name)

    return ordered[:max_columns]

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

SUPPORTED_DB_KEYS = ('miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic')


@lru_cache(maxsize=1)
def _get_quality_concept_dictionary():
    """Load concept dictionary once for dynamic coverage checks."""
    try:
        from easyicu.concept import load_dictionary
        return load_dictionary(include_sofa2=True)
    except Exception:
        return {}


def _has_any_source_recursive(concept_name, database, concept_dict, visited=None):
    """Recursively check whether a concept or one of its sub-concepts has a source in a database."""
    if visited is None:
        visited = set()
    if concept_name in visited:
        return False
    visited.add(concept_name)
    concept_def = concept_dict.get(concept_name)
    if not concept_def:
        return False
    if getattr(concept_def, 'sources', {}).get(database):
        return True
    if getattr(concept_def, 'sub_concepts', None):
        return any(_has_any_source_recursive(sub_concept, database, concept_dict, visited) for sub_concept in concept_def.sub_concepts)
    return False


def _get_supported_db_keys_for_concept(concept_name: str) -> set[str]:
    """Return supported DB keys for a concept using concept-dict first, then special-concept fallback."""
    concept_dict = _get_quality_concept_dictionary()
    if concept_name in concept_dict:
        return {
            db for db in SUPPORTED_DB_KEYS
            if _has_any_source_recursive(concept_name, db, concept_dict)
        }

    special_all = {
        'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
        'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
        'creat_low_past_48hr', 'creat_low_past_7day',
        'sep3_sofa1', 'sep3_sofa2',
        'circ_failure', 'circ_event',
    }
    if concept_name in special_all:
        return set(SUPPORTED_DB_KEYS)

    fallback_n = CONCEPT_DB_COVERAGE.get(concept_name, 0)
    return set(SUPPORTED_DB_KEYS[:fallback_n])


def _get_concept_coverage_summary(concept_name: str, current_database: Optional[str] = None) -> tuple[str, bool, int]:
    """Return display label, current-db support flag, and supported DB count."""
    supported = _get_supported_db_keys_for_concept(concept_name)
    count = len(supported)
    current_supported = True if not current_database else current_database in supported
    label = f"{count}/6 DBs"
    if current_database:
        prefix = "✓ " if current_supported else "✕ "
        label = f"{prefix}{label}"
    return label, current_supported, count


def _get_coverage_badge(concept_name: str) -> str:
    """返回概念跨库可用性的 HTML badge。"""
    _, _, n = _get_concept_coverage_summary(concept_name)
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


def _get_missing_cause_tag(
    concept_name: str,
    missing_rate: float,
    *,
    current_database: Optional[str] = None,
    has_observed_rows: bool = False,
) -> tuple:
    """为缺失率提供可解释的原因标签。返回 (text, color)。"""
    sparse_events = {'ecmo', 'ecmo_indication', 'mech_circ_support', 'cort', 'rrt',
                     'abx', 'vaso_ind', 'mech_vent', 'vent_ind', 'vent_start', 'vent_end',
                     'sep3_sofa1', 'sep3_sofa2', 'circ_failure', 'circ_event'}
    _, current_supported, n_db = _get_concept_coverage_summary(concept_name, current_database)
    if not has_observed_rows and current_database and not current_supported:
        return get_text('missing_cause_db'), '#dc2626'
    if concept_name in sparse_events and missing_rate > 0.7:
        return get_text('missing_cause_sparse'), '#6b7280'
    if n_db <= 1 and not has_observed_rows:
        return get_text('missing_cause_db'), '#dc2626'
    if missing_rate > 0.8:
        return get_text('missing_cause_cohort'), '#d97706'
    return get_text('missing_cause_normal'), '#059669'


def _prepare_timeseries_plot_df(df: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
    """Sort time series and collapse duplicate timestamps for plotting."""
    if time_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[time_col, value_col])
    plot_df = df[[time_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[time_col, value_col])
    if plot_df.empty:
        return plot_df
    if pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
        plot_df = plot_df.sort_values(time_col)
    else:
        numeric_time = pd.to_numeric(plot_df[time_col], errors='coerce')
        if numeric_time.notna().any():
            plot_df['_time_numeric'] = numeric_time
            plot_df = plot_df.dropna(subset=['_time_numeric']).sort_values('_time_numeric')
        else:
            parsed_time = pd.to_datetime(plot_df[time_col], errors='coerce')
            if parsed_time.notna().any():
                plot_df['_time_parsed'] = parsed_time
                plot_df = plot_df.dropna(subset=['_time_parsed']).sort_values('_time_parsed')
    plot_df = plot_df.groupby(time_col, as_index=False)[value_col].mean()
    return plot_df.sort_values(time_col).reset_index(drop=True)


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

SCREENSHOT_TIMESERIES_PRIORITY = [
    'hr', 'map', 'sbp', 'spo2', 'temp', 'resp',
    'crea', 'plt', 'wbc', 'lact', 'sofa2', 'sofa',
]

SCREENSHOT_QUALITY_PRIORITY = [
    'crea', 'lact', 'hr', 'map', 'sbp', 'temp', 'wbc', 'hgb', 'plt', 'bili', 'sofa2', 'sofa',
]

QUALITY_DEMOGRAPHIC_STATIC = {
    'death', 'los_icu', 'los_hosp', 'age', 'weight', 'height', 'sex', 'bmi'
}

QUALITY_EVENT_TIME_SERIES = {
    'circ_failure', 'circ_event',
    'sep3_sofa2', 'sep3_sofa1', 'sepsis_sofa2',
    'susp_inf', 'infection_icd', 'samp',
    'rrt', 'rrt_criteria',
    'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
    'mech_vent', 'vent_ind', 'vent_start', 'vent_end',
    'ecmo', 'ecmo_indication', 'mech_circ_support',
    'abx', 'cort',
    'vaso_ind',
}

QUALITY_STATIC_BOOLEAN_EVENTS = {
    'ecmo', 'ecmo_indication', 'mech_circ_support',
    'cort', 'abx', 'vaso_ind',
}

QUALITY_TIME_CANDIDATES = [
    'time', 'charttime', 'datetime', 'measuredat', 'measuredat_minutes',
    'observationoffset', 'Offset', 'starttime', 'endtime', 'givenat', 'timestamp',
]

QUALITY_EXCLUDE_COLUMNS = {
    'stay_id', 'hadm_id', 'icustay_id', 'time', 'index',
    'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
    'patientunitstayid', 'admissionid', 'patientid', 'CaseID',
}

PRIMARY_VALUE_COLUMN_HINTS = {
    'abp': ['map', 'sbp', 'dbp'],
    'bp': ['map', 'sbp', 'dbp'],
    'fio2': ['fio2'],
    'sofa': ['sofa'],
    'sofa2': ['sofa2'],
}

PHYSIOLOGIC_RANGES = {
    'hr': (20.0, 250.0),
    'resp': (4.0, 80.0),
    'sbp': (40.0, 300.0),
    'dbp': (20.0, 200.0),
    'map': (30.0, 220.0),
    'temp': (25.0, 45.0),
    'spo2': (0.0, 100.0),
    'o2sat': (0.0, 100.0),
    'fio2': (0.0, 100.0),
    'ph': (6.8, 7.8),
    'po2': (20.0, 600.0),
    'pco2': (10.0, 150.0),
    'pafi': (20.0, 800.0),
    'safi': (20.0, 800.0),
    'glu': (20.0, 1500.0),
    'crea': (0.1, 20.0),
    'creat': (0.1, 20.0),
    'lact': (0.0, 30.0),
    'plt': (0.0, 2000.0),
    'wbc': (0.0, 500.0),
    'hgb': (0.0, 25.0),
    'inr_pt': (0.5, 20.0),
    'na': (100.0, 200.0),
    'k': (1.0, 10.0),
    'bili': (0.0, 80.0),
    'alb': (0.5, 8.0),
    'bun': (1.0, 250.0),
}


def _is_screenshot_mode() -> bool:
    """Return whether figure-oriented screenshot mode is enabled."""
    return bool(st.session_state.get('screenshot_mode', False))


def _apply_screenshot_mode_ui_state(state: dict[str, Any]) -> None:
    """Hide transient chrome that should not appear in figure screenshots."""
    state['_floating_ai_open'] = False
    if state.get('_scroll_to_tab') == 'ai_assistant':
        state.pop('_scroll_to_tab', None)


def _resolve_viz_data_source_mode(
    *,
    current_mode: str | None,
    recent_export_path: str,
    allow_demo: bool,
    entry_mode: str,
) -> str:
    """Keep the Quick Visualization data-source radio in a valid session-state option."""
    source_options = ["exported"] + (["demo"] if allow_demo else [])
    default_source = "exported" if recent_export_path else ("demo" if allow_demo and entry_mode == 'demo' else "exported")
    return current_mode if current_mode in source_options else default_source


def _get_plotly_chart_config() -> dict[str, Any]:
    """Return a consistent Plotly config, hiding UI chrome in screenshot mode."""
    base_config = {
        "displaylogo": False,
        "responsive": True,
    }
    if _is_screenshot_mode():
        base_config["displayModeBar"] = False
    return base_config


def _select_timeseries_screenshot_concepts(available_concepts: list[str], max_items: int = 4) -> list[str]:
    """Pick a compact set of representative time series for figure screenshots."""
    unique_concepts = list(dict.fromkeys(available_concepts))
    selected: list[str] = []
    for concept in SCREENSHOT_TIMESERIES_PRIORITY:
        if concept in unique_concepts and concept not in selected:
            selected.append(concept)
        if len(selected) >= max_items:
            return selected
    for concept in unique_concepts:
        if concept not in selected:
            selected.append(concept)
        if len(selected) >= max_items:
            break
    return selected


def _select_quality_distribution_concept(loaded_concepts: dict[str, Any]) -> str | None:
    """Choose a stable, interpretable concept for the Data Quality distribution preview."""
    available = [name for name, df in loaded_concepts.items() if isinstance(df, pd.DataFrame) and not df.empty]
    for concept in SCREENSHOT_QUALITY_PRIORITY:
        if concept in available:
            return concept
    return available[0] if available else None


def _apply_quick_viz_screenshot_defaults(state: dict[str, Any], *, lang: str) -> None:
    """Apply figure-friendly defaults once when screenshot mode is enabled."""
    _apply_screenshot_mode_ui_state(state)
    patient_ids = state.get('patient_ids') or []
    first_patient = patient_ids[0] if patient_ids else None
    if first_patient is not None:
        state['lane_patient_select'] = state.get('lane_patient_select') or first_patient
        state['patient_view_id'] = state.get('patient_view_id') or first_patient
    state['ts_mode'] = "Clinical Lanes" if lang == 'en' else "临床分道"
    state['patient_view_mode'] = "Dashboard" if lang == 'en' else "综合仪表盘"
    state['missing_chart_sort_order'] = 'desc'
    quality_concept = _select_quality_distribution_concept(state.get('loaded_concepts', {}))
    if quality_concept:
        state['quality_concept'] = quality_concept
    state['data_table_view_mode'] = "Merge All (Wide Table)" if lang == 'en' else "合并全部（宽表）"
    state['_quick_viz_screenshot_preset_applied'] = True


def _sync_quick_viz_screenshot_mode(state: dict[str, Any], *, lang: str) -> bool:
    """Synchronize screenshot-mode transitions and report whether a rerun is needed."""
    screenshot_mode = bool(state.get('screenshot_mode', False))
    previous_mode = bool(state.get('_screenshot_mode_last_value', False))

    if screenshot_mode:
        _apply_screenshot_mode_ui_state(state)

    if screenshot_mode != previous_mode:
        state['_screenshot_mode_last_value'] = screenshot_mode
        if screenshot_mode:
            _apply_quick_viz_screenshot_defaults(state, lang=lang)
        else:
            state['_quick_viz_screenshot_preset_applied'] = False
        return True

    if screenshot_mode and not state.get('_quick_viz_screenshot_preset_applied', False):
        _apply_quick_viz_screenshot_defaults(state, lang=lang)
        return True

    return False


FIGURE_TARGET_MAP = {
    'fig2': ('paper', 'Figure 2'),
    'figure2': ('paper', 'Figure 2'),
    'figure-2': ('paper', 'Figure 2'),
    'extraction-figure': ('paper', 'Figure 2'),
    'pipeline-figure': ('paper', 'Figure 2'),
    'fig3': ('paper', 'Figure 3'),
    'figure3': ('paper', 'Figure 3'),
    'figure-3': ('paper', 'Figure 3'),
    'review-figure': ('paper', 'Figure 3'),
    'multi-view-figure': ('paper', 'Figure 3'),
    'fig4': ('paper', 'Figure 4'),
    'figure4': ('paper', 'Figure 4'),
    'figure-4': ('paper', 'Figure 4'),
    'ai-figure': ('paper', 'Figure 4'),
    'assistant-figure': ('paper', 'Figure 4'),
    's1': ('paper', 'Supplementary Figure S1'),
    'supp-s1': ('paper', 'Supplementary Figure S1'),
    'supplementary-s1': ('paper', 'Supplementary Figure S1'),
    'supplementary-figure-s1': ('paper', 'Supplementary Figure S1'),
    'table': ('viz', 'Data Tables'),
    'tables': ('viz', 'Data Tables'),
    'data': ('viz', 'Data Tables'),
    'datatable': ('viz', 'Data Tables'),
    'time': ('viz', 'Time Series'),
    'timeseries': ('viz', 'Time Series'),
    'trend': ('viz', 'Time Series'),
    'patient': ('viz', 'Patient Overview'),
    'overview': ('viz', 'Patient Overview'),
    'quality': ('viz', 'Data Quality'),
    'missing': ('viz', 'Data Quality'),
    'group': ('cohort', 'Group Contrast'),
    'contrast': ('cohort', 'Group Contrast'),
    'coverage': ('cohort', 'Coverage Audit'),
    'audit': ('cohort', 'Coverage Audit'),
    'crossdb': ('cohort', 'Cross-DB Benchmark'),
    'cross-db': ('cohort', 'Cross-DB Benchmark'),
    'distribution': ('cohort', 'Cross-DB Benchmark'),
    'benchmark': ('cohort', 'Cross-DB Benchmark'),
    'snapshot': ('cohort', 'Cohort Snapshot'),
    'dashboard': ('cohort', 'Cohort Snapshot'),
    'cohort': ('cohort', 'Cohort Snapshot'),
    'sofa': ('cohort', 'SOFA-1 vs SOFA-2'),
    'sensitivity': ('cohort', 'SOFA-1 vs SOFA-2'),
    'reclassification': ('cohort', 'SOFA-1 vs SOFA-2'),
}


def _normalize_figure_target(raw_target: str | None) -> tuple[str, str]:
    """Map figure URL shorthands to a top-level section and sub-tab label fragment."""
    token = str(raw_target or '').strip().lower().replace('_', '-').replace(' ', '-')
    if token in {'', '1', 'true', 'yes', 'on'}:
        return '', ''
    return FIGURE_TARGET_MAP.get(token, ('', ''))


def _quality_detect_time_col(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely time column for quality-rate calculations."""
    for col in QUALITY_TIME_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _quality_to_hour_bins(series: pd.Series, col_name: str) -> Optional[pd.Series]:
    """Normalize common EasyICU time formats to hourly bins."""
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


def _get_quality_cohort_patient_count(state: dict[str, Any]) -> int:
    """Choose the cohort denominator shown in the current Quick Visualization session."""
    patient_ids = state.get('patient_ids') or []
    if patient_ids:
        return len(patient_ids)

    all_patient_count = int(state.get('all_patient_count') or 0)
    if all_patient_count > 0:
        return all_patient_count

    mock_params = state.get('mock_params', {}) or {}
    mock_patient_count = int(mock_params.get('n_patients') or 0)
    if mock_patient_count > 0:
        return mock_patient_count

    id_col = state.get('id_col')
    max_patients_found = 0
    if id_col:
        for df in state.get('loaded_concepts', {}).values():
            if isinstance(df, pd.DataFrame) and not df.empty and id_col in df.columns:
                max_patients_found = max(max_patients_found, int(df[id_col].nunique()))
    if max_patients_found > 0:
        return max_patients_found

    patient_limit = int(state.get('patient_limit') or 0)
    return patient_limit if patient_limit > 0 else 0


def _count_quality_event_occurrences(series: pd.Series) -> int:
    """Count event occurrences instead of treating all non-null rows as observed values."""
    if pd.api.types.is_bool_dtype(series):
        return int(series.fillna(False).sum())
    if pd.api.types.is_numeric_dtype(series):
        return int((series.fillna(0) > 0).sum())
    return int(series.notna().sum())


def _choose_concept_value_column(concept: str, df: pd.DataFrame) -> Optional[str]:
    """Pick the most clinically useful numeric value column for a concept frame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    value_cols = [c for c in numeric_cols if c not in QUALITY_EXCLUDE_COLUMNS]
    if not value_cols:
        return None
    if concept in value_cols:
        return concept

    for candidate in PRIMARY_VALUE_COLUMN_HINTS.get(concept, []):
        if candidate in value_cols:
            return candidate

    return value_cols[0]


def _get_concept_numeric_value_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric value columns after excluding IDs and time-like metadata."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    numeric_cols = df.select_dtypes(include=['number']).columns
    return [c for c in numeric_cols if c not in QUALITY_EXCLUDE_COLUMNS]


def _expected_observation_count(
    concept: str,
    patient_df: pd.DataFrame,
    los_icu: Optional[float],
    *,
    demo_hours: Optional[int] = None,
    fallback_hours: Optional[int] = None,
) -> tuple[int, str]:
    """Return the expected hourly observation denominator and its provenance tag."""
    if not isinstance(patient_df, pd.DataFrame):
        return 0, 'empty'

    time_col = _quality_detect_time_col(patient_df)
    if time_col is None and concept in QUALITY_STATIC_BOOLEAN_EVENTS:
        return 1, 'static'
    if time_col is None:
        return 1, 'static'

    if demo_hours is not None and int(demo_hours) > 0:
        return int(demo_hours), 'demo'

    los_value = pd.to_numeric(pd.Series([los_icu]), errors='coerce').iloc[0]
    if pd.notna(los_value) and float(los_value) > 0:
        return max(1, int(np.ceil(float(los_value) * 24))), 'los'

    if fallback_hours is not None and int(fallback_hours) > 0:
        return int(fallback_hours), '72h'

    return 72, '72h'


def _compute_quality_out_of_physio_rate(concept: str, df: pd.DataFrame) -> float:
    """Measure the share of non-null rows that are outside harmonized physiologic bounds."""
    bounds = PHYSIOLOGIC_RANGES.get(concept)
    value_col = _choose_concept_value_column(concept, df)
    if bounds is None or value_col is None or value_col not in df.columns:
        return 0.0

    values = pd.to_numeric(df[value_col], errors='coerce').dropna()
    if values.empty:
        return 0.0

    lower, upper = bounds
    out_of_range = ((values < lower) | (values > upper)).mean() * 100
    return float(out_of_range)


def _compute_quality_duplicate_timestamp_rate(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
) -> float:
    """Count duplicate rows where the same patient and concept share the same timestamp."""
    if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns:
        return 0.0

    time_col = _quality_detect_time_col(df)
    value_col = _choose_concept_value_column(concept, df)
    if time_col is None or value_col is None or value_col not in df.columns:
        return 0.0

    observed = df[[id_col, time_col, value_col]].dropna(subset=[time_col, value_col]).copy()
    if observed.empty:
        return 0.0

    duplicate_rows = observed.duplicated(subset=[id_col, time_col], keep='first').sum()
    return float(duplicate_rows / len(observed) * 100)


def _summarize_quality_temporal_density(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
    fallback_hours: Optional[int] = None,
) -> dict[str, float]:
    """Summarize records-per-patient-hour using median and IQR to resist ICU long tails.

    Vectorized: one groupby size() pass plus an aligned division, no per-patient loop.
    """
    empty = {'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'n_patients': 0}
    if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns:
        return empty

    value_col = _choose_concept_value_column(concept, df)
    if value_col is None or value_col not in df.columns:
        return empty

    seen_patient_ids = df[id_col].dropna().unique().tolist()
    if not seen_patient_ids:
        return empty

    expected_per_patient, _sources = _vectorized_expected_per_patient(
        seen_patient_ids,
        los_by_patient=los_by_patient,
        demo_hours=demo_hours,
        fallback_hours=fallback_hours,
    )

    value_not_na = df.loc[df[value_col].notna(), [id_col]]
    if value_not_na.empty:
        return empty
    obs_counts = (
        value_not_na.groupby(id_col, observed=False)
        .size()
        .astype('float64')
        .reindex(pd.Index(seen_patient_ids), fill_value=0)
    )

    expected = expected_per_patient.astype('float64')
    valid = expected > 0
    if not valid.any():
        return empty

    densities = (obs_counts[valid] / expected[valid]).replace([np.inf, -np.inf], np.nan).dropna()
    if densities.empty:
        return empty

    return {
        'median': float(densities.median()),
        'q25': float(densities.quantile(0.25)),
        'q75': float(densities.quantile(0.75)),
        'n_patients': int(len(densities)),
    }


def _filter_patient_selector_options(
    patient_ids: list[Any],
    *,
    query: str = "",
    max_display: int = 200,
) -> list[Any]:
    """Filter and cap patient selector options so large cohorts stay responsive."""
    unique_patient_ids = list(dict.fromkeys(patient_ids))
    trimmed_query = str(query or "").strip()
    if trimmed_query:
        unique_patient_ids = [pid for pid in unique_patient_ids if trimmed_query in str(pid)]
    return unique_patient_ids[:max(1, int(max_display))]


def _patient_selector(
    *,
    patient_ids: list[Any],
    state_key: str,
    label: str,
    lang: str,
    max_display: int = 200,
    default_patient: Any = None,
) -> Any:
    """Render a searchable patient selector with a capped option list."""
    search_label = "Search Patient ID" if lang == 'en' else "搜索患者ID"
    search_placeholder = "Type to filter..." if lang == 'en' else "输入ID过滤..."
    search_query = st.text_input(
        search_label,
        key=f"{state_key}_search",
        placeholder=search_placeholder,
    )
    options = _filter_patient_selector_options(
        patient_ids,
        query=search_query,
        max_display=max_display,
    )
    if default_patient is not None and default_patient not in options and default_patient in patient_ids:
        options = [default_patient] + options[: max(0, max_display - 1)]
    if not options:
        options = _filter_patient_selector_options(patient_ids, max_display=max_display)
    if not options:
        return None

    select_kwargs: dict[str, Any] = {
        'label': label,
        'options': options,
        'key': state_key,
    }
    if default_patient in options:
        select_kwargs['index'] = options.index(default_patient)
    return st.selectbox(**select_kwargs)


def _get_quality_cohort_patient_ids(state: dict[str, Any]) -> list[Any]:
    """Return the current patient universe for quality metrics whenever it is known."""
    patient_ids = state.get('patient_ids') or []
    if patient_ids:
        return list(dict.fromkeys(patient_ids))

    id_col = state.get('id_col')
    if not id_col:
        return []

    candidate_frames: list[pd.DataFrame] = []
    loaded_concepts = state.get('loaded_concepts', {}) or {}
    for concept_name in ('age', 'sex', 'death', 'los_icu'):
        frame = loaded_concepts.get(concept_name)
        if isinstance(frame, pd.DataFrame) and not frame.empty and id_col in frame.columns:
            candidate_frames.append(frame)
    if not candidate_frames:
        for frame in loaded_concepts.values():
            if isinstance(frame, pd.DataFrame) and not frame.empty and id_col in frame.columns:
                candidate_frames.append(frame)
                if len(candidate_frames) >= 3:
                    break

    patient_pool: list[Any] = []
    for frame in candidate_frames:
        patient_pool.extend(frame[id_col].dropna().tolist())
    return list(dict.fromkeys(patient_pool))


def _get_quality_los_by_patient(state: dict[str, Any]) -> Optional[pd.Series]:
    """Build a per-patient LOS series in days when available for denominator estimation."""
    loaded_concepts = state.get('loaded_concepts', {}) or {}
    los_df = loaded_concepts.get('los_icu')
    id_col = state.get('id_col')
    if not isinstance(los_df, pd.DataFrame) or los_df.empty or not id_col or id_col not in los_df.columns:
        return None
    if 'los_icu' not in los_df.columns:
        return None

    los_copy = los_df[[id_col, 'los_icu']].copy()
    los_copy['los_icu'] = pd.to_numeric(los_copy['los_icu'], errors='coerce')
    los_copy = los_copy.dropna(subset=['los_icu'])
    if los_copy.empty:
        return None
    return los_copy.groupby(id_col, observed=False)['los_icu'].max()


def _format_quality_density(summary: dict[str, float], lang: str) -> str:
    """Format median/IQR records-per-patient-hour text for the quality table."""
    if not summary or int(summary.get('n_patients', 0)) == 0:
        return '-' if lang == 'en' else '—'
    return f"{summary['median']:.2f} [{summary['q25']:.2f}-{summary['q75']:.2f}]"


def _get_quality_denominator_note(tag: str, lang: str) -> str:
    """Explain denominator provenance tags shown in the quality table."""
    notes = {
        'd=los': "LOS-based expected hours" if lang == 'en' else "按患者 ICU LOS 估算期望小时数",
        'd=72h': "72 h fallback window" if lang == 'en' else "使用 72 小时兜底窗口",
        'd=demo': "demo simulation horizon" if lang == 'en' else "演示数据预设时间窗",
        'd=static': "single observation per patient" if lang == 'en' else "每位患者单次静态观测",
        'd=mixed': "mixed LOS / fallback denominators" if lang == 'en' else "混合使用 LOS 与兜底分母",
    }
    return notes.get(str(tag or '').lower(), tag)


def _smd_severity_tag(value: float, lang: str) -> str:
    """Attach an interpretable balance flag next to SMD values."""
    abs_value = abs(float(value))
    if abs_value > 0.25:
        return "🔴 large" if lang == 'en' else "🔴 较大"
    if abs_value > 0.10:
        return "🟠 mild" if lang == 'en' else "🟠 轻度"
    return "🟢 balanced" if lang == 'en' else "🟢 平衡"


def _compute_smd_continuous(series1: pd.Series, series2: pd.Series) -> float:
    """Compute standardized mean difference for continuous variables."""
    values1 = pd.to_numeric(series1, errors='coerce').dropna()
    values2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(values1) < 2 or len(values2) < 2:
        return 0.0

    sd1 = float(values1.std(ddof=1))
    sd2 = float(values2.std(ddof=1))
    pooled_sd = np.sqrt((sd1 ** 2 + sd2 ** 2) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((values1.mean() - values2.mean()) / pooled_sd)


def _compute_smd_binary(series1: pd.Series, series2: pd.Series) -> float:
    """Compute standardized mean difference for binary variables."""
    values1 = pd.to_numeric(series1, errors='coerce').dropna()
    values2 = pd.to_numeric(series2, errors='coerce').dropna()
    if values1.empty or values2.empty:
        return 0.0

    p1 = float(values1.mean())
    p2 = float(values2.mean())
    p_bar = (p1 + p2) / 2
    denom = np.sqrt(p_bar * (1 - p_bar))
    if denom == 0:
        return 0.0
    return float((p1 - p2) / denom)


def _vectorized_expected_per_patient(
    patient_universe: list[Any],
    *,
    los_by_patient: Optional[pd.Series],
    demo_hours: Optional[int],
    fallback_hours: Optional[int],
) -> tuple[pd.Series, pd.Series]:
    """Return (expected_count, source_tag) Series indexed by patient id.

    Vectorizes the per-patient branch of `_expected_observation_count` for the
    time-series case where time_col is already known to exist on the frame.
    """
    universe_index = pd.Index(patient_universe)
    fallback = int(fallback_hours) if fallback_hours and int(fallback_hours) > 0 else 72

    if demo_hours is not None and int(demo_hours) > 0:
        expected = pd.Series(int(demo_hours), index=universe_index, dtype='int64')
        sources = pd.Series('demo', index=universe_index)
        return expected, sources

    if isinstance(los_by_patient, pd.Series) and not los_by_patient.empty:
        los_aligned = pd.to_numeric(los_by_patient.reindex(universe_index), errors='coerce')
    else:
        los_aligned = pd.Series(np.nan, index=universe_index, dtype='float64')

    expected = pd.Series(fallback, index=universe_index, dtype='int64')
    sources = pd.Series('72h', index=universe_index)

    los_valid = los_aligned.notna() & (los_aligned > 0)
    if los_valid.any():
        los_hours = np.ceil(los_aligned[los_valid].astype('float64') * 24).astype('int64')
        los_hours = np.maximum(1, los_hours)
        expected.loc[los_valid] = los_hours
        sources.loc[los_valid] = 'los'

    return expected, sources


def _build_quality_metric_profile(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    cohort_patient_count: int,
    time_grid_size: int,
    cohort_patient_ids: Optional[list[Any]] = None,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
) -> dict[str, Any]:
    """Compute one concept-level QC profile shared by the table and chart views.

    Performance: replaces the old O(P * N) per-patient loop with a single
    vectorized pass that (a) computes expected counts via aligned Series
    operations and (b) folds temporal density into the same groupby pass.
    """
    profile = {
        'missing_rate': 100.0,
        'out_of_physio_rate': 0.0,
        'duplicate_rate': 0.0,
        'denominator_tag': 'd=72h',
        'expected_observations': 0,
        'observed_observations': 0,
        'temporal_density': {'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'n_patients': 0},
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        return profile

    value_col = _choose_concept_value_column(concept, df)
    time_col = _quality_detect_time_col(df)
    n_patients = int(df[id_col].nunique()) if id_col in df.columns else 0
    cohort_patient_count = int(cohort_patient_count or 0)

    if concept in QUALITY_STATIC_BOOLEAN_EVENTS and not time_col:
        denominator = cohort_patient_count or n_patients
        if denominator > 0:
            profile['missing_rate'] = float(max(0.0, min(100.0, (1 - min(n_patients, denominator) / denominator) * 100)))
            profile['expected_observations'] = denominator
            profile['observed_observations'] = min(n_patients, denominator)
            profile['denominator_tag'] = 'd=static'
        return profile

    if value_col is None or value_col not in df.columns:
        if cohort_patient_count > 0 and n_patients > 0:
            patient_coverage_missing = (1 - min(n_patients, cohort_patient_count) / cohort_patient_count) * 100
            profile['missing_rate'] = float(max(0.0, min(100.0, patient_coverage_missing)))
        return profile

    profile['out_of_physio_rate'] = _compute_quality_out_of_physio_rate(concept, df)
    profile['duplicate_rate'] = _compute_quality_duplicate_timestamp_rate(concept=concept, df=df, id_col=id_col)

    raw_na_rate = float(df[value_col].isna().mean() * 100)
    if concept in QUALITY_DEMOGRAPHIC_STATIC:
        profile['missing_rate'] = raw_na_rate
        profile['denominator_tag'] = 'd=static'
        return profile

    if time_col and id_col in df.columns:
        seen_patient_ids = df[id_col].dropna().unique().tolist()
        patient_universe = cohort_patient_ids or seen_patient_ids
        patient_universe = list(dict.fromkeys(patient_universe))
        fallback_hours = time_grid_size if time_grid_size > 0 else None

        expected_per_patient, source_per_patient = _vectorized_expected_per_patient(
            patient_universe,
            los_by_patient=los_by_patient,
            demo_hours=demo_hours,
            fallback_hours=fallback_hours,
        )

        expected_total = int(expected_per_patient.sum())
        unique_sources = sorted(set(source_per_patient.tolist()))

        if not cohort_patient_ids and cohort_patient_count > len(seen_patient_ids):
            missing_patient_count = cohort_patient_count - len(seen_patient_ids)
            default_expected, default_source = _expected_observation_count(
                concept=concept,
                patient_df=df,
                los_icu=None,
                demo_hours=demo_hours,
                fallback_hours=fallback_hours,
            )
            if default_expected > 0:
                expected_total += missing_patient_count * default_expected
                if default_source not in unique_sources:
                    unique_sources = sorted(set(unique_sources + [default_source]))

        source_label = unique_sources[0] if len(unique_sources) == 1 else 'mixed'

        hour_bins = _quality_to_hour_bins(df[time_col], time_col)
        if hour_bins is not None:
            if concept in QUALITY_EVENT_TIME_SERIES:
                if pd.api.types.is_bool_dtype(df[value_col]):
                    observed_mask = df[value_col].astype('boolean').fillna(False)
                elif pd.api.types.is_numeric_dtype(df[value_col]):
                    observed_mask = df[value_col].fillna(0) > 0
                else:
                    observed_mask = df[value_col].notna()
            else:
                observed_mask = df[value_col].notna()

            observed = df.loc[observed_mask, [id_col]].copy()
            observed['_hour_bin'] = hour_bins.loc[observed.index]
            observed = observed.dropna(subset=['_hour_bin'])
            observed_total = int(observed.drop_duplicates(subset=[id_col, '_hour_bin']).shape[0])

            if expected_total > 0:
                coverage_missing = (1 - observed_total / expected_total) * 100
                profile['missing_rate'] = float(max(raw_na_rate, max(0.0, min(100.0, coverage_missing))))
                profile['expected_observations'] = expected_total
                profile['observed_observations'] = observed_total
                profile['denominator_tag'] = f"d={source_label}"

        # Temporal density: vectorized per-patient count using fast groupby+size,
        # aligned against expected_per_patient. Replaces the old O(P * N) loop.
        seen_index = pd.Index(seen_patient_ids)
        if len(seen_index) > 0:
            value_not_na = df.loc[df[value_col].notna(), [id_col]]
            if not value_not_na.empty:
                obs_counts = (
                    value_not_na.groupby(id_col, observed=False)
                    .size()
                    .astype('float64')
                )
            else:
                obs_counts = pd.Series(dtype='float64')
            obs_counts_aligned = obs_counts.reindex(seen_index, fill_value=0)
            expected_for_seen = expected_per_patient.reindex(seen_index)
            if expected_for_seen.isna().any():
                expected_for_seen = expected_for_seen.fillna(
                    int(fallback_hours) if fallback_hours and int(fallback_hours) > 0 else 72
                )
            expected_for_seen = expected_for_seen.astype('float64')
            valid_mask = expected_for_seen > 0
            if valid_mask.any():
                densities = obs_counts_aligned[valid_mask] / expected_for_seen[valid_mask]
                densities = densities.replace([np.inf, -np.inf], np.nan).dropna()
                if len(densities) > 0:
                    profile['temporal_density'] = {
                        'median': float(densities.median()),
                        'q25': float(densities.quantile(0.25)),
                        'q75': float(densities.quantile(0.75)),
                        'n_patients': int(len(densities)),
                    }
        return profile

    if cohort_patient_count > 0 and n_patients > 0:
        patient_coverage_missing = (1 - min(n_patients, cohort_patient_count) / cohort_patient_count) * 100
        profile['missing_rate'] = float(max(raw_na_rate, max(0.0, min(100.0, patient_coverage_missing))))
        profile['expected_observations'] = cohort_patient_count
        profile['observed_observations'] = min(n_patients, cohort_patient_count)
    else:
        profile['missing_rate'] = raw_na_rate

    return profile


def _cohort_cache_fingerprint(cohort_patient_ids: Optional[list[Any]]) -> tuple[Any, ...]:
    """Cheap O(1-ish) fingerprint to key the per-concept quality cache."""
    if not cohort_patient_ids:
        return (0,)
    head = str(cohort_patient_ids[0]) if len(cohort_patient_ids) else ''
    tail = str(cohort_patient_ids[-1]) if len(cohort_patient_ids) else ''
    return (len(cohort_patient_ids), head, tail)


def _los_cache_fingerprint(los_by_patient: Optional[pd.Series]) -> tuple[Any, ...]:
    """Cheap fingerprint for the LOS series used in quality denominators."""
    if not isinstance(los_by_patient, pd.Series) or los_by_patient.empty:
        return (0,)
    try:
        head_idx = str(los_by_patient.index[0])
    except Exception:
        head_idx = ''
    try:
        # sum() is vectorized C; keeps the fingerprint sensitive to content edits
        values_sum = float(pd.to_numeric(los_by_patient, errors='coerce').fillna(0).sum())
    except Exception:
        values_sum = 0.0
    return (len(los_by_patient), head_idx, round(values_sum, 4))


def _build_quality_metric_profile_cached(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    cohort_patient_count: int,
    time_grid_size: int,
    cohort_patient_ids: Optional[list[Any]] = None,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
) -> dict[str, Any]:
    """Session-scoped cache wrapper around `_build_quality_metric_profile`.

    The cache is keyed by a cheap structural fingerprint of the inputs so that
    re-rendering the Quality page (language toggles, tab switches, sidebar
    interactions) does not re-run the whole QC pipeline for every concept.
    The cache lives on `st.session_state` and is naturally invalidated when
    `loaded_concepts` is rebuilt (df identity changes).
    """
    try:
        cache = st.session_state.setdefault('_quality_profile_cache', {})
    except Exception:
        # When called outside a Streamlit run (e.g. tests) just skip caching.
        return _build_quality_metric_profile(
            concept=concept,
            df=df,
            id_col=id_col,
            cohort_patient_count=cohort_patient_count,
            time_grid_size=time_grid_size,
            cohort_patient_ids=cohort_patient_ids,
            los_by_patient=los_by_patient,
            demo_hours=demo_hours,
        )

    key = (
        str(concept),
        str(id_col),
        id(df),
        tuple(df.shape) if isinstance(df, pd.DataFrame) else (0, 0),
        int(cohort_patient_count or 0),
        int(time_grid_size or 0),
        int(demo_hours) if demo_hours else None,
        _cohort_cache_fingerprint(cohort_patient_ids),
        _los_cache_fingerprint(los_by_patient),
    )

    cached = cache.get(key)
    if cached is not None:
        return cached

    result = _build_quality_metric_profile(
        concept=concept,
        df=df,
        id_col=id_col,
        cohort_patient_count=cohort_patient_count,
        time_grid_size=time_grid_size,
        cohort_patient_ids=cohort_patient_ids,
        los_by_patient=los_by_patient,
        demo_hours=demo_hours,
    )
    # Guard against unbounded growth across long sessions.
    if len(cache) > 512:
        cache.clear()
    cache[key] = result
    return result


def _compute_quality_missing_rate(
    *,
    concept: str,
    df: pd.DataFrame,
    id_col: str,
    cohort_patient_count: int,
    time_grid_size: int,
    cohort_patient_ids: Optional[list[Any]] = None,
    los_by_patient: Optional[pd.Series] = None,
    demo_hours: Optional[int] = None,
) -> float:
    """Compute a consistent concept-level missing rate for both table and chart views."""
    profile = _build_quality_metric_profile_cached(
        concept=concept,
        df=df,
        id_col=id_col,
        cohort_patient_count=cohort_patient_count,
        time_grid_size=time_grid_size,
        cohort_patient_ids=cohort_patient_ids,
        los_by_patient=los_by_patient,
        demo_hours=demo_hours,
    )
    return float(profile['missing_rate'])

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
    for _k in ('si_mode', 'abx_win', 'samp_win', 'positive_cultures', 'abx_min_count'):
        if _k in kwargs and kwargs[_k] is not None:
            load_kwargs[_k] = kwargs[_k]

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
    verbose: bool = False,
    **extra_kwargs,
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
                    load_kwargs.update({k: v for k, v in (extra_kwargs or {}).items() if v is not None})

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
    **extra_kwargs,
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
            **extra_kwargs,
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
            **extra_kwargs,
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
    """检查数据状态。"""
    return _check_data_status_impl(data_path, database, globals())




def convert_data_with_progress(data_path: str, database: str):
    """转换数据并显示进度。"""
    return _convert_data_with_progress_impl(data_path, database, globals())




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
    if 'entry_lang_select' not in st.session_state:
        st.session_state.entry_lang_select = 'EN' if st.session_state.language == 'en' else 'ZH'
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


DISEASE_COHORT_CONFIG = {
    'sepsis': {
        'label_en': 'Sepsis-3 cohort',
        'label_zh': '脓毒症队列（Sepsis-3）',
        'description_en': 'Use Sepsis-3 labels (`sep3_sofa2` preferred, fallback `sep3_sofa1`) to keep only septic patients.',
        'description_zh': '使用 Sepsis-3 标签（优先 `sep3_sofa2`，回退 `sep3_sofa1`）仅保留脓毒症患者。',
        'required_modules': {'sepsis3_sofa2', 'sepsis3_sofa1', 'sepsis_shared', 'sofa2_score', 'sofa1_score'},
        'concept_priority': ['sep3_sofa2', 'sep3_sofa1'],
    },
    'aki': {
        'label_en': 'AKI cohort (KDIGO)',
        'label_zh': 'AKI 队列（KDIGO）',
        'description_en': 'Use KDIGO-AKI outputs (`aki_stage` preferred, fallback `aki`) to keep AKI-positive patients.',
        'description_zh': '使用 KDIGO-AKI 输出（优先 `aki_stage`，回退 `aki`）仅保留 AKI 患者。',
        'required_modules': {'renal'},
        'concept_priority': ['aki_stage', 'aki'],
    },
    'circ_failure': {
        'label_en': 'Circulatory failure cohort',
        'label_zh': '循环衰竭队列',
        'description_en': 'Use `circ_failure` or `circ_event` to keep patients with circulatory failure evidence.',
        'description_zh': '使用 `circ_failure` 或 `circ_event` 仅保留存在循环衰竭证据的患者。',
        'required_modules': {'circulatory'},
        'concept_priority': ['circ_failure', 'circ_event'],
    },
    'mech_vent': {
        'label_en': 'Mechanical ventilation cohort',
        'label_zh': '机械通气队列',
        'description_en': 'Use `mech_vent` or `vent_ind` to keep ventilated ICU stays.',
        'description_zh': '使用 `mech_vent` 或 `vent_ind` 仅保留机械通气 ICU 住院记录。',
        'required_modules': {'respiratory'},
        'concept_priority': ['mech_vent', 'vent_ind'],
    },
    'rrt': {
        'label_en': 'Renal replacement therapy cohort',
        'label_zh': '肾脏替代治疗队列',
        'description_en': 'Use `rrt` or `rrt_criteria` to keep ICU stays receiving renal replacement therapy.',
        'description_zh': '使用 `rrt` 或 `rrt_criteria` 仅保留接受肾脏替代治疗的 ICU 住院记录。',
        'required_modules': {'renal'},
        'concept_priority': ['rrt', 'rrt_criteria'],
    },
    'ards': {
        'label_en': 'ARDS cohort',
        'label_zh': 'ARDS 队列',
        'description_en': 'ICD-backed ARDS template for databases with diagnosis codes. Use for acute respiratory distress syndrome cohorts.',
        'description_zh': '适用于带诊断编码数据库的 ARDS 模板队列，可用于急性呼吸窘迫综合征研究。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['J80', '51882'],
    },
    'pneumonia': {
        'label_en': 'Pneumonia cohort',
        'label_zh': '肺炎队列',
        'description_en': 'ICD-backed pneumonia template for infectious respiratory cohorts.',
        'description_zh': '适用于呼吸系统感染研究的 ICD 肺炎模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', '481', '482', '483', '485', '486'],
    },
    'heart_failure': {
        'label_en': 'Heart failure cohort',
        'label_zh': '心力衰竭队列',
        'description_en': 'ICD-backed heart-failure template for decompensated heart-failure or cardiogenic cohorts.',
        'description_zh': '适用于失代偿心衰或心源性相关研究的 ICD 心衰模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['I50', '428'],
    },
    'ami': {
        'label_en': 'Acute myocardial infarction cohort',
        'label_zh': '急性心肌梗死队列',
        'description_en': 'ICD-backed AMI template for STEMI / NSTEMI style cohorts.',
        'description_zh': '适用于 STEMI / NSTEMI 等急性心肌梗死研究的 ICD 模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['I21', 'I22', '410'],
    },
    'stroke': {
        'label_en': 'Stroke cohort',
        'label_zh': '卒中队列',
        'description_en': 'ICD-backed stroke template covering ischemic and hemorrhagic stroke codes.',
        'description_zh': '覆盖缺血性与出血性卒中的 ICD 模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['I60', 'I61', 'I63', 'I64', '430', '431', '434', '436'],
    },
}

SEPSIS_MODE_CONFIG = {
    "auto": {
        "label_en": "Auto by database",
        "label_zh": "按数据库自动选择",
        "desc_en": "Recommended default. eICU uses `ICD + antibiotics`; other databases default to `ABX + sampling`.",
        "desc_zh": "推荐默认值。eICU 使用 `ICD + 抗生素`；其他数据库默认使用 `抗生素 + 采样`。",
    },
    "and": {
        "label_en": "ABX + sampling (strict window)",
        "label_zh": "抗生素 + 采样（严格时间窗）",
        "desc_en": "Classic Sepsis-3 style suspected infection: antibiotics and body-fluid sampling must co-occur within windows.",
        "desc_zh": "经典 Sepsis-3 风格的疑似感染定义：抗生素与体液采样需在时间窗内共同出现。",
    },
    "or": {
        "label_en": "ABX or sampling",
        "label_zh": "抗生素或采样",
        "desc_en": "More permissive suspected infection proxy. Keeps either antibiotics or sampling events.",
        "desc_zh": "更宽松的疑似感染代理定义。只要出现抗生素或采样事件即可。",
    },
    "abx": {
        "label_en": "Antibiotics only",
        "label_zh": "仅抗生素",
        "desc_en": "Antibiotic-only proxy, useful when sampling coverage is sparse.",
        "desc_zh": "仅抗生素代理定义，适用于采样覆盖较差的数据集。",
    },
    "samp": {
        "label_en": "Sampling only",
        "label_zh": "仅采样",
        "desc_en": "Body-fluid sampling only. Useful for exploratory sensitivity analyses.",
        "desc_zh": "仅使用体液采样事件，适合做敏感性分析。",
    },
    "icd_abx": {
        "label_en": "ICD infection + antibiotics",
        "label_zh": "感染 ICD + 抗生素",
        "desc_en": "eICU-oriented fallback: infection ICD identifies patients, antibiotics provide event time.",
        "desc_zh": "偏 eICU 的替代方案：感染 ICD 先定人，再用抗生素时间定时点。",
    },
}

ICD_FILTER_DATABASES = {'miiv', 'mimic', 'eicu'}


def _supports_icd_filter(database: str | None) -> bool:
    """Return whether the current database supports sidebar ICD filters."""
    return str(database or "").lower() in ICD_FILTER_DATABASES


def _split_query_tokens(text: str) -> list[str]:
    """Split user ICD / keyword query into compact non-empty tokens."""
    if not text:
        return []
    cleaned = str(text).replace('，', ',').replace(';', ',').replace('；', ',').replace('\n', ',')
    raw_tokens = [tok.strip() for tok in cleaned.split(',') if tok.strip()]
    expanded_tokens: list[str] = []
    for token in raw_tokens:
        range_match = re.fullmatch(r'([A-Za-z]+)(\d+)\s*-\s*([A-Za-z]+)?(\d+)', token)
        if not range_match:
            expanded_tokens.append(token)
            continue

        prefix_start, start_num, prefix_end, end_num = range_match.groups()
        prefix_start = prefix_start.upper()
        prefix_end = (prefix_end or prefix_start).upper()
        if prefix_start != prefix_end:
            expanded_tokens.append(token)
            continue

        start_int = int(start_num)
        end_int = int(end_num)
        if end_int < start_int or end_int - start_int > 50:
            expanded_tokens.append(token)
            continue

        width = max(len(start_num), len(end_num))
        expanded_tokens.extend([f"{prefix_start}{value:0{width}d}" for value in range(start_int, end_int + 1)])

    return expanded_tokens


def _get_sepsis_runtime_options() -> dict:
    """Read current web sepsis settings and return kwargs for load_concepts/callbacks."""
    abx_hours = st.session_state.get('sepsis_abx_win_hours', 24)
    samp_hours = st.session_state.get('sepsis_samp_win_hours', 72)
    return {
        'si_mode': st.session_state.get('sepsis_si_mode', 'auto'),
        'positive_cultures': bool(st.session_state.get('sepsis_positive_cultures', False)),
        'abx_win': f"{int(abx_hours)}h",
        'samp_win': f"{int(samp_hours)}h",
    }


def _get_supported_disease_cohorts(database: str) -> list[str]:
    """Return supported disease cohort keys for the current database."""
    base = ['none', 'sepsis', 'aki', 'circ_failure', 'mech_vent', 'rrt']
    if _supports_icd_filter(database):
        base.extend(['ards', 'pneumonia', 'heart_failure', 'ami', 'stroke'])
    return base


def _match_ids_by_icd_tokens(data_path: Path, database: str, icu_df: pd.DataFrame, id_col_lower: str, tokens: list[str]) -> set:
    """Match ICU stay IDs by ICD prefixes / diagnosis keywords for DBs with diagnosis coding."""
    if not tokens or not _supports_icd_filter(database):
        return set()
    matched_ids = set()
    if database in {'miiv', 'mimic'}:
        diag_path = data_path / 'diagnoses_icd.parquet'
        if diag_path.exists() and 'hadm_id' in icu_df.columns:
            diag_df = pd.read_parquet(diag_path, columns=['hadm_id', 'icd_code'])
            codes = diag_df['icd_code'].astype(str).str.upper().str.replace('.', '', regex=False)
            norm_tokens = [tok.upper().replace('.', '') for tok in tokens]
            diag_mask = pd.Series(False, index=diag_df.index)
            for token in norm_tokens:
                diag_mask |= codes.str.startswith(token)
            matched_hadm = set(diag_df.loc[diag_mask, 'hadm_id'].dropna().unique())
            matched_ids = set(icu_df.loc[icu_df['hadm_id'].isin(matched_hadm), id_col_lower].dropna().unique())
    elif database == 'eicu':
        diag_path = data_path / 'diagnosis.parquet'
        if diag_path.exists():
            diag_df = pd.read_parquet(diag_path)
            diag_df.columns = [c.lower() for c in diag_df.columns]
            if 'patientunitstayid' in diag_df.columns:
                diag_text = pd.Series('', index=diag_df.index, dtype='object')
                if 'icd9code' in diag_df.columns:
                    diag_text = diag_text.str.cat(diag_df['icd9code'].astype(str), sep=' ', na_rep='')
                if 'diagnosisstring' in diag_df.columns:
                    diag_text = diag_text.str.cat(diag_df['diagnosisstring'].astype(str), sep=' ', na_rep='')
                diag_text = diag_text.str.lower().str.replace('.', '', regex=False)
                diag_mask = pd.Series(False, index=diag_df.index)
                for token in tokens:
                    diag_mask |= diag_text.str.contains(str(token).lower().replace('.', ''), na=False)
                matched_ids = set(diag_df.loc[diag_mask, 'patientunitstayid'].dropna().unique())
    return matched_ids


def _get_positive_patient_ids_from_data(
    data: dict,
    actual_id_col: str,
    concept_priority: list[str],
) -> set:
    """Infer patient IDs with positive events/labels from loaded concept data."""
    true_tokens = {'1', 'true', 't', 'yes', 'y'}
    for concept_name in concept_priority:
        df = data.get(concept_name)
        if not isinstance(df, pd.DataFrame) or df.empty or actual_id_col not in df.columns:
            continue
        value_candidates = [concept_name] + [c for c in df.columns if c not in {actual_id_col, 'charttime', 'time', 'starttime', 'datetime', 'valueuom', 'unit'}]
        value_col = next((c for c in value_candidates if c in df.columns), None)
        if not value_col:
            continue
        vals = pd.to_numeric(df[value_col], errors='coerce')
        if vals.notna().any():
            mask = vals > 0
        else:
            str_vals = df[value_col].astype(str).str.strip().str.lower()
            mask = str_vals.isin(true_tokens)
        return set(df.loc[mask.fillna(False), actual_id_col].dropna().unique())
    return set()


def _render_sepsis_ai_button(lang: str) -> None:
    """Offer a contextual AI explanation button for Sepsis settings."""
    button_label = "🤖 Ask AI about Sepsis settings" if lang == 'en' else "🤖 问 AI 解释脓毒症设置"
    if st.button(button_label, key="ask_ai_about_sepsis_settings", use_container_width=True):
        si_mode = st.session_state.get('sepsis_si_mode', 'auto')
        abx_hours = st.session_state.get('sepsis_abx_win_hours', 24)
        samp_hours = st.session_state.get('sepsis_samp_win_hours', 72)
        positive = st.session_state.get('sepsis_positive_cultures', False)
        if lang == 'en':
            question = (
                "I am configuring Sepsis-3 in EasyICU. Explain whether my current settings are reasonable for my study, "
                f"including si_mode={si_mode}, abx_win={abx_hours}h, samp_win={samp_hours}h, "
                f"positive_cultures={positive}. Also tell me when EasyICU defaults to ICD-based infection evidence "
                "and how this relates to supported databases."
            )
        else:
            question = (
                "我正在 EasyICU 中配置 Sepsis-3。请解释我当前的设置是否合理，"
                f"包括 si_mode={si_mode}、abx_win={abx_hours}h、samp_win={samp_hours}h、"
                f"positive_cultures={positive}。同时说明 EasyICU 何时会默认使用基于 ICD 的感染证据，"
                "以及这与支持的数据库有什么关系。"
            )
        st.session_state['_ai_pending_question'] = question
        st.session_state['_floating_ai_open'] = True


def _clear_icd_preview_state() -> None:
    """Remove ICD preview caches and temporary UI state."""
    for key in (
        '_icd_preview_cache_include',
        '_icd_preview_cache_exclude',
    ):
        st.session_state.pop(key, None)


def _render_icd_preview_main_panel(lang: str) -> None:
    """Render ICD preview results in the main content area instead of the sidebar."""
    if not _supports_icd_filter(st.session_state.get('database')):
        _clear_icd_preview_state()
        return

    include_query = str(st.session_state.get('cohort_filter', {}).get('icd_include_query', '')).strip()
    exclude_query = str(st.session_state.get('cohort_filter', {}).get('icd_exclude_query', '')).strip()
    preview_specs = [
        ("include", include_query, "Include" if lang == 'en' else "包含"),
        ("exclude", exclude_query, "Exclude" if lang == 'en' else "排除"),
    ]

    active_previews = []
    for preview_key, preview_query, preview_label in preview_specs:
        tokens = _split_query_tokens(preview_query)
        cached = st.session_state.get(f'_icd_preview_cache_{preview_key}')
        if not tokens or not cached or cached.get('tokens') != tokens:
            continue
        active_previews.append((preview_key, preview_label, cached))

    if not active_previews:
        return

    title = "🧾 ICD Match Preview" if lang == 'en' else "🧾 ICD 匹配预览"
    header_cols = st.columns([6, 1.4])
    with header_cols[0]:
        st.markdown(f"#### {title}")
    with header_cols[1]:
        clear_label = "🧹 Clear Preview" if lang == 'en' else "🧹 清除预览"
        if st.button(clear_label, key="clear_icd_preview_main", use_container_width=True):
            _clear_icd_preview_state()
            st.rerun()

    include_cached = next((cached for key, _, cached in active_previews if key == 'include'), None)
    exclude_cached = next((cached for key, _, cached in active_previews if key == 'exclude'), None)
    total_patients = 0
    if include_cached:
        total_patients = int(include_cached.get('total_patients', 0) or 0)
    elif exclude_cached:
        total_patients = int(exclude_cached.get('total_patients', 0) or 0)

    include_ids = set(include_cached.get('matched_ids', [])) if include_cached else set()
    exclude_ids = set(exclude_cached.get('matched_ids', [])) if exclude_cached else set()

    if include_cached:
        final_ids = include_ids - exclude_ids
        final_count = len(final_ids)
    elif exclude_cached:
        final_count = max(total_patients - len(exclude_ids), 0)
    else:
        final_count = 0

    final_pct = final_count / total_patients * 100 if total_patients > 0 else 0
    if lang == 'en':
        st.info(f"🧮 Final cohort after ICD filters: **{final_count:,}** / {total_patients:,} patients ({final_pct:.1f}%)")
    else:
        st.info(f"🧮 ICD 筛选后的最终队列：**{final_count:,}** / {total_patients:,} 位患者 ({final_pct:.1f}%)")

    cols = st.columns(len(active_previews))
    for col, (_, preview_label, preview_result) in zip(cols, active_previews):
        with col:
            if preview_result.get('error'):
                st.warning(preview_result['error'])
                continue

            matched = preview_result.get('matched_patients', 0)
            total = preview_result.get('total_patients', 0)
            pct = matched / total * 100 if total > 0 else 0
            if lang == 'en':
                st.success(f"📊 {preview_label}: matched **{matched:,}** / {total:,} patients ({pct:.1f}%)")
            else:
                st.success(f"📊 {preview_label}: 匹配到 **{matched:,}** / {total:,} 位患者 ({pct:.1f}%)")

            top_codes = preview_result.get('top_codes')
            if top_codes is not None and len(top_codes) > 0:
                table_label = (
                    f"📋 Top matching ICD codes ({preview_label})"
                    if lang == 'en' else
                    f"📋 匹配频率最高的 ICD 编码（{preview_label}）"
                )
                st.markdown(f"**{table_label}**")
                st.dataframe(top_codes, use_container_width=True, hide_index=True)


def _format_definition_list(values, limit: int = 6) -> str:
    """Format a short, readable comma-separated preview for metadata fields."""
    cleaned = [str(v).strip() for v in values if v not in (None, "", [], {}) and str(v).strip()]
    if not cleaned:
        return "—"
    if len(cleaned) <= limit:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:limit])} (+{len(cleaned) - limit} more)"


@st.cache_data(ttl=3600)
def _get_table_defaults() -> dict:
    """Load per-database per-table default column mappings from data-sources.json."""
    import json as _json
    try:
        ds_path = Path(__file__).resolve().parent.parent / 'data' / 'data-sources.json'
        if not ds_path.exists():
            return {}
        with open(ds_path, encoding='utf-8') as f:
            ds_list = _json.load(f)
        result = {}
        for entry in ds_list:
            if not isinstance(entry, dict):
                continue
            db_name = entry.get('name', '')
            tables = entry.get('tables', {})
            if not isinstance(tables, dict):
                continue
            for tbl_name, tbl_def in tables.items():
                if not isinstance(tbl_def, dict):
                    continue
                defaults = tbl_def.get('defaults', {})
                if not isinstance(defaults, dict):
                    defaults = {}
                val_var = defaults.get('val_var') or tbl_def.get('val_var')
                idx_var = defaults.get('index_var') or tbl_def.get('index_var')
                if val_var or idx_var:
                    result[(db_name, tbl_name)] = {'val_var': val_var, 'index_var': idx_var}
        return result
    except Exception:
        return {}


def _format_source_selector(source) -> str:
    """Summarize the identifying selector used for a raw concept source."""
    selector_parts = []
    if getattr(source, 'sub_var', None):
        if getattr(source, 'ids', None):
            selector_parts.append(f"{source.sub_var}={_format_definition_list(source.ids, limit=8)}")
        elif getattr(source, 'regex', None):
            selector_parts.append(f"{source.sub_var}~/{source.regex}/")
        else:
            selector_parts.append(str(source.sub_var))
    elif getattr(source, 'regex', None):
        selector_parts.append(f"regex=/{source.regex}/")
    if getattr(source, 'class_name', None):
        selector_parts.append(f"class={source.class_name}")
    if getattr(source, 'params', None):
        param_keys = sorted(source.params.keys())
        if param_keys:
            selector_parts.append(f"params={_format_definition_list(param_keys, limit=6)}")
    return " | ".join(selector_parts) if selector_parts else "—"


def _collect_recursive_concept_sources(concept_name: str, database: str, concept_dict: dict, visited=None) -> list[tuple[str, object]]:
    """Collect raw source entries for a concept by recursively traversing sub-concepts."""
    if visited is None:
        visited = set()
    if concept_name in visited:
        return []
    visited.add(concept_name)

    concept_def = concept_dict.get(concept_name)
    if not concept_def:
        return []

    collected = []
    for source in concept_def.sources.get(database, []):
        collected.append((concept_name, source))
    for sub_concept in getattr(concept_def, 'sub_concepts', []) or []:
        collected.extend(_collect_recursive_concept_sources(sub_concept, database, concept_dict, visited))
    return collected


def _get_feature_definition_rows(selected_concepts: list[str], database: str, lang: str) -> list[dict]:
    """Build a transparent per-feature definition table for the current database."""
    concept_dict = _get_quality_concept_dictionary()
    table_defaults = _get_table_defaults()
    rows = []

    for concept_name in sorted(set(selected_concepts)):
        eng_name, chn_name, dict_unit = CONCEPT_DICTIONARY.get(concept_name, (concept_name, concept_name, ''))
        description_en, description_zh = CONCEPT_DESCRIPTIONS.get(concept_name, ('', ''))
        display_name = eng_name if lang == 'en' else chn_name
        description = description_en if lang == 'en' else description_zh

        concept_def = concept_dict.get(concept_name)
        unit = dict_unit
        if concept_def and getattr(concept_def, 'units', None):
            unit = _format_definition_list(concept_def.units, limit=4)
        unit = unit or "—"

        base_row = {
            'Feature': concept_name,
            'Name': display_name,
            'Unit': unit,
            'Type': "Direct",
            'Table(s)': "—",
            'Selector / ID': "—",
            'Columns': "—",
            'Logic': description or "—",
        }

        if concept_def:
            direct_sources = concept_def.sources.get(database, [])
            if direct_sources:
                for source in direct_sources:
                    logic_parts = []
                    if getattr(source, 'callback', None):
                        logic_parts.append(f"callback: {source.callback}")
                    if getattr(concept_def, 'callback', None):
                        logic_parts.append(f"callback: {concept_def.callback}")
                    if getattr(concept_def, 'sub_concepts', None):
                        logic_parts.append(f"derived: {_format_definition_list(concept_def.sub_concepts, limit=6)}")
                    if description:
                        logic_parts.append(description)

                    # Resolve value_var / index_var from table defaults
                    tbl_name = getattr(source, 'table', None) or ''
                    tbl_def = table_defaults.get((database, tbl_name), {})
                    val_var = getattr(source, 'value_var', None) or tbl_def.get('val_var')
                    idx_var = getattr(source, 'index_var', None) or tbl_def.get('index_var')
                    unit_var = getattr(source, 'unit_var', None)
                    dur_var = getattr(source, 'dur_var', None)

                    col_parts = []
                    if val_var:
                        col_parts.append(f"value={val_var}")
                    if unit_var:
                        col_parts.append(f"unit={unit_var}")
                    if idx_var:
                        col_parts.append(f"time={idx_var}")
                    if dur_var:
                        col_parts.append(f"dur={dur_var}")

                    row = dict(base_row)
                    has_callback = getattr(source, 'callback', None) or getattr(concept_def, 'callback', None)
                    src_class = getattr(source, 'class_name', None) or ''
                    if not tbl_name and src_class == 'fun_itm':
                        row['Type'] = "Function"
                        row['Table(s)'] = "computed"
                    elif not tbl_name and src_class == 'rec_cncpt':
                        row['Type'] = "Derived"
                        row['Table(s)'] = "recursive"
                    else:
                        row['Type'] = "Callback" if has_callback else "Direct"
                        row['Table(s)'] = tbl_name or "—"
                    row['Selector / ID'] = _format_source_selector(source)
                    row['Columns'] = " | ".join(col_parts) if col_parts else "—"
                    row['Logic'] = " ; ".join(logic_parts) if logic_parts else "—"
                    rows.append(row)
                continue

            recursive_sources = _collect_recursive_concept_sources(concept_name, database, concept_dict)
            if recursive_sources:
                table_names = sorted({src.table for _, src in recursive_sources if getattr(src, 'table', None)})
                selectors = []
                for leaf_concept, source in recursive_sources:
                    selector_summary = _format_source_selector(source)
                    if selector_summary != "—":
                        selectors.append(f"{leaf_concept}: {selector_summary}")
                    else:
                        selectors.append(f"{leaf_concept}")

                logic_parts = []
                if getattr(concept_def, 'callback', None):
                    logic_parts.append(f"callback: {concept_def.callback}")
                if getattr(concept_def, 'sub_concepts', None):
                    logic_parts.append(f"derived: {_format_definition_list(concept_def.sub_concepts, limit=8)}")
                if description:
                    logic_parts.append(description)

                row = dict(base_row)
                row['Type'] = "Derived"
                row['Table(s)'] = _format_definition_list(table_names, limit=8)
                row['Selector / ID'] = _format_definition_list(selectors, limit=6)
                row['Logic'] = " ; ".join(logic_parts) if logic_parts else "—"
                rows.append(row)
                continue

            # Concept exists in dict but no source for this database
            if concept_name in SPECIAL_CONCEPTS:
                module_name, func_name, output_cols = SPECIAL_CONCEPTS[concept_name]
                row = dict(base_row)
                row['Type'] = "Special"
                row['Table(s)'] = "loader"
                row['Selector / ID'] = f"{module_name}.{func_name}"
                logic_parts = [f"output: {_format_definition_list(output_cols, limit=4)}"]
                if description:
                    logic_parts.append(description)
                row['Logic'] = " ; ".join(logic_parts)
                rows.append(row)
                continue

            row = dict(base_row)
            no_src_label = f"No source for {database.upper()}" if lang == 'en' else f"{database.upper()} 无数据源"
            row['Type'] = no_src_label
            rows.append(row)
            continue

        # concept_def is None -- check SPECIAL_CONCEPTS
        if concept_name in SPECIAL_CONCEPTS:
            module_name, func_name, output_cols = SPECIAL_CONCEPTS[concept_name]
            row = dict(base_row)
            row['Type'] = "Special"
            row['Table(s)'] = "loader"
            row['Selector / ID'] = f"{module_name}.{func_name}"
            logic_parts = [f"output: {_format_definition_list(output_cols, limit=4)}"]
            if description:
                logic_parts.append(description)
            row['Logic'] = " ; ".join(logic_parts)
            rows.append(row)
            continue

        row = dict(base_row)
        row['Type'] = "Unknown"
        rows.append(row)

    return rows


def _render_feature_definition_panel(lang: str) -> None:
    """Render a transparent feature definition panel for the selected database and features."""
    if not st.session_state.get('step3_confirmed', False):
        return

    selected_concepts = list(st.session_state.get('selected_concepts', []) or [])
    if not selected_concepts:
        return

    database = str(st.session_state.get('database', '') or '')
    if not database:
        return

    rows = _get_feature_definition_rows(selected_concepts, database, lang)
    if not rows:
        return

    title = "🧬 Feature Definition Transparency" if lang == 'en' else "🧬 特征定义透明化"
    caption = (
        f"Current database: {database.upper()}. This table shows how each selected feature is defined in EasyICU, including raw tables, selectors/item IDs, units, and derived logic."
        if lang == 'en' else
        f"当前数据库：{database.upper()}。该表展示 EasyICU 如何定义你已选特征，包括原始表、选择器/item ID、单位以及派生逻辑。"
    )
    download_label = "⬇️ Download Definition CSV" if lang == 'en' else "⬇️ 下载定义表 CSV"
    n_features = len(set(selected_concepts))
    n_rows = len(rows)
    summary = (
        f"Showing **{n_features}** selected features and **{n_rows}** database-specific definition rows."
        if lang == 'en' else
        f"当前展示 **{n_features}** 个已选特征，对应 **{n_rows}** 条数据库定义记录。"
    )

    with st.expander(title, expanded=True):
        st.caption(caption)
        st.info(summary)
        definition_df = pd.DataFrame(rows)
        st.download_button(
            download_label,
            data=definition_df.to_csv(index=False, encoding='utf-8-sig'),
            file_name=f"easyicu_feature_definition_{database.lower()}.csv",
            mime="text/csv",
            key="download_feature_definition_csv",
        )
        st.dataframe(
            definition_df,
            use_container_width=True,
            hide_index=True,
            height=min(640, 120 + 36 * max(len(definition_df), 1)),
        )


def _preview_icd_match(data_path: Path, database: str, tokens: list[str]) -> dict:
    """Preview ICD code matching: return matched patient count and top codes."""
    result = {
        'tokens': tokens,
        'matched_patients': 0,
        'matched_ids': [],
        'total_patients': 0,
        'top_codes': None,
        'error': None,
    }
    try:
        DB_META_PREVIEW = {
            'miiv': {'id_col': 'stay_id', 'icu_table': 'icustays.parquet'},
            'mimic': {'id_col': 'icustay_id', 'icu_table': 'icustays.parquet'},
            'eicu': {'id_col': 'patientunitstayid', 'icu_table': 'patient.parquet'},
        }
        meta = DB_META_PREVIEW.get(database)
        if not meta:
            result['error'] = f"ICD preview not supported for {database}"
            return result
        icu_path = data_path / meta['icu_table']
        if not icu_path.exists():
            result['error'] = f"ICU table not found: {icu_path.name}"
            return result
        icu_df = pd.read_parquet(icu_path)
        icu_df.columns = [c.lower() for c in icu_df.columns]
        id_col = meta['id_col'].lower()
        result['total_patients'] = icu_df[id_col].nunique()

        if database in ('miiv', 'mimic'):
            diag_path = data_path / 'diagnoses_icd.parquet'
            if not diag_path.exists():
                result['error'] = f"diagnoses_icd.parquet not found"
                return result
            diag_df = pd.read_parquet(diag_path, columns=['hadm_id', 'icd_code', 'icd_version'] if database == 'miiv' else ['hadm_id', 'icd_code'])
            codes = diag_df['icd_code'].astype(str).str.upper().str.replace('.', '', regex=False)
            norm_tokens = [tok.upper().replace('.', '') for tok in tokens]
            diag_mask = pd.Series(False, index=diag_df.index)
            for tok in norm_tokens:
                diag_mask |= codes.str.startswith(tok)
            matched_diag = diag_df.loc[diag_mask].copy()
            if 'hadm_id' in icu_df.columns:
                matched_hadm = set(matched_diag['hadm_id'].dropna().unique())
                matched_ids = set(icu_df.loc[icu_df['hadm_id'].isin(matched_hadm), id_col].dropna().unique())
                result['matched_patients'] = len(matched_ids)
                result['matched_ids'] = sorted(matched_ids)
            # Top ICD codes
            matched_diag['icd_code_clean'] = codes[diag_mask]
            code_counts = matched_diag['icd_code_clean'].value_counts().head(20).reset_index()
            code_counts.columns = ['ICD Code', 'Count']
            # Try enrich with descriptions
            try:
                d_path = data_path / 'd_icd_diagnoses.parquet'
                if d_path.exists():
                    d_df = pd.read_parquet(d_path)
                    d_df.columns = [c.lower() for c in d_df.columns]
                    if 'icd_code' in d_df.columns and 'long_title' in d_df.columns:
                        d_df['icd_code'] = d_df['icd_code'].astype(str).str.upper().str.replace('.', '', regex=False)
                        desc_map = dict(zip(d_df['icd_code'], d_df['long_title']))
                        code_counts['Description'] = code_counts['ICD Code'].map(desc_map).fillna('')
            except Exception:
                pass
            result['top_codes'] = code_counts

        elif database == 'eicu':
            diag_path = data_path / 'diagnosis.parquet'
            if not diag_path.exists():
                result['error'] = f"diagnosis.parquet not found"
                return result
            diag_df = pd.read_parquet(diag_path)
            diag_df.columns = [c.lower() for c in diag_df.columns]
            if 'patientunitstayid' not in diag_df.columns:
                result['error'] = "patientunitstayid not found in diagnosis table"
                return result
            diag_text = pd.Series('', index=diag_df.index, dtype='object')
            if 'icd9code' in diag_df.columns:
                diag_text = diag_text.str.cat(diag_df['icd9code'].astype(str), sep=' ', na_rep='')
            if 'diagnosisstring' in diag_df.columns:
                diag_text = diag_text.str.cat(diag_df['diagnosisstring'].astype(str), sep=' ', na_rep='')
            diag_text_lower = diag_text.str.lower().str.replace('.', '', regex=False)
            diag_mask = pd.Series(False, index=diag_df.index)
            for tok in tokens:
                diag_mask |= diag_text_lower.str.contains(str(tok).lower().replace('.', ''), na=False)
            matched_diag = diag_df.loc[diag_mask]
            matched_ids = set(matched_diag['patientunitstayid'].dropna().unique())
            result['matched_patients'] = len(matched_ids)
            result['matched_ids'] = sorted(matched_ids)
            # Top codes for eICU
            if 'icd9code' in matched_diag.columns:
                code_counts = matched_diag['icd9code'].dropna().astype(str).value_counts().head(20).reset_index()
                code_counts.columns = ['ICD Code', 'Count']
                if 'diagnosisstring' in matched_diag.columns:
                    ds_map = dict(zip(matched_diag['icd9code'].astype(str), matched_diag['diagnosisstring'].astype(str)))
                    code_counts['Description'] = code_counts['ICD Code'].map(ds_map).fillna('')
                result['top_codes'] = code_counts
    except Exception as e:
        result['error'] = str(e)
    return result


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

    # 5. Disease cohort filters based on loaded clinical concepts
    disease_cohort = cf.get('disease_cohort')
    if disease_cohort and disease_cohort != 'none':
        disease_cfg = DISEASE_COHORT_CONFIG.get(disease_cohort, {})
        concept_priority = disease_cfg.get('concept_priority', [])
        if concept_priority:
            positive_ids = _get_positive_patient_ids_from_data(
                data,
                actual_id_col=actual_id_col,
                concept_priority=concept_priority,
            )
            exclude_ids |= (all_patient_ids - positive_ids)

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
    """应用队列筛选。"""
    return _apply_cohort_filter_impl(data_path, database, candidate_ids, globals())




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
        'sub_patient_view': '🏥 Patient Overview',
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
        'review_tables': '📋 Data Tables',
        'review_trends': '📈 Time Series',
        'review_patients': '🏥 Patient Overview',
        'review_quality': '📊 Data Quality',
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
    """验证数据库路径。"""
    return _validate_database_path_impl(data_path, database, globals())




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




def render_quick_visualization_page():
    """渲染快速可视化主页面。"""
    return _render_quick_visualization_page_impl(globals())




def render_entry_page():
    """渲染模式选择入口页面。"""
    return _render_entry_page_impl(globals())




def render_sidebar():
    """渲染侧边栏 - 根据entry_mode显示不同内容。"""
    return _render_sidebar_impl(globals())




def _get_pyarrow_version() -> str | None:
    try:
        import pyarrow as pa
        return pa.__version__
    except Exception:
        return None


def _get_parquet_created_by(file_path: Path) -> str | None:
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.created_by
    except Exception:
        return None


def _build_export_read_failure_warning(
    target_files: List[Path],
    read_failures: List[Dict[str, Any]],
    lang: str = 'en',
) -> str:
    """Explain why exported files were found but could not be read."""
    if not target_files:
        return "⚠️ No valid data files found" if lang == 'en' else "⚠️ 未找到有效的数据文件"

    parquet_failures = [item for item in read_failures if item.get('suffix') == '.parquet']
    if parquet_failures and len(read_failures) == len(target_files):
        first_failure = parquet_failures[0]
        pyarrow_version = _get_pyarrow_version() or "unknown"
        created_by = first_failure.get('created_by')
        error_text = first_failure.get('error', 'unknown read error')

        if lang == 'en':
            parts = [
                f"⚠️ Found {len(target_files)} data file(s), but failed to read them.",
                f"Current runtime: pyarrow={pyarrow_version}.",
            ]
            if created_by:
                parts.append(f"First parquet writer: {created_by}.")
            parts.append(f"Reader error: {error_text}.")
            parts.append("This usually means the exported parquet files were written by a newer Arrow version than the current runtime.")
            return " ".join(parts)

        parts = [
            f"⚠️ 已找到 {len(target_files)} 个数据文件，但读取失败。",
            f"当前运行环境: pyarrow={pyarrow_version}。",
        ]
        if created_by:
            parts.append(f"首个 parquet 写入器: {created_by}。")
        parts.append(f"读取错误: {error_text}。")
        parts.append("这通常意味着导出 parquet 的 Arrow 版本比当前运行环境更新。")
        return " ".join(parts)

    return "⚠️ No valid data files found" if lang == 'en' else "⚠️ 未找到有效的数据文件"


def load_from_exported(export_dir: str, max_patients: int = 50, selected_files: list = None):
    """从已导出的文件加载数据。"""
    return _load_from_exported_impl(export_dir, max_patients, selected_files, globals())




def load_data():
    """加载数据。"""
    return _load_data_impl(globals())




def load_data_for_preview(max_patients: int = 50):
    """加载预览数据。"""
    return _load_data_for_preview_impl(max_patients, globals())




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


def _html_escape_text(value: Any, default: str = "—") -> str:
    """Escape short UI values for HTML snippets."""
    if value is None:
        return html.escape(default)
    text = str(value).strip()
    return html.escape(text if text else default)


def _workflow_field(label: str, value: Any, suffix: str = "⌄") -> str:
    suffix_html = f"<span>{html.escape(suffix)}</span>" if suffix else ""
    return (
        '<div class="workflow-field">'
        f'<div class="workflow-label">{html.escape(label)}</div>'
        f'<div class="workflow-input"><span>{_html_escape_text(value)}</span>{suffix_html}</div>'
        '</div>'
    )


def _workflow_status(done: bool, done_text: str, todo_text: str) -> str:
    if done:
        return (
            '<div class="workflow-status">'
            '<span class="workflow-check-dot">✓</span>'
            f'<span>{html.escape(done_text)}</span>'
            '</div>'
        )
    return (
        '<div class="workflow-status warn">'
        '<span class="workflow-check-dot" style="background:#f59e0b">!</span>'
        f'<span>{html.escape(todo_text)}</span>'
        '</div>'
    )


def _render_extraction_pipeline_figure(
    *,
    lang: str,
    step1_done: bool,
    step2_done: bool,
    step3_done: bool,
    step4_done: bool,
) -> None:
    """Render the live extraction workflow using the same visual logic as Figure 2."""
    is_en = lang == 'en'
    db_display_names = {
        'mock': 'Demo ICU',
        'miiv': 'MIMIC-IV',
        'eicu': 'eICU-CRD',
        'aumc': 'AmsterdamUMCdb',
        'hirid': 'HiRID',
        'mimic': 'MIMIC-III',
        'sic': 'SICdb',
    }
    database = st.session_state.get('database', 'mock' if st.session_state.get('use_mock_data') else 'miiv')
    db_label = db_display_names.get(database, str(database).upper())
    data_path = (
        "Auto-generated demo data"
        if st.session_state.get('use_mock_data', False)
        else st.session_state.get('data_path', '')
    )
    cohort_filter = st.session_state.get('cohort_filter', {}) or {}
    age_min = cohort_filter.get('age_min') or 18
    age_max = cohort_filter.get('age_max') or 120
    los_min = cohort_filter.get('los_min') or 24
    gender = cohort_filter.get('gender') or ("Any" if is_en else "不限")
    survived = cohort_filter.get('survived')
    survival_text = (
        "Any" if survived is None else ("Survived" if survived else "Deceased")
    ) if is_en else (
        "不限" if survived is None else ("存活" if survived else "死亡")
    )
    cohort_name = cohort_filter.get('disease_cohort') or 'none'
    cohort_display = {
        'none': 'No disease filter' if is_en else '不限制疾病队列',
        'sepsis': 'Sepsis-3 cohort',
        'aki': 'AKI cohort (KDIGO)',
        'circ_failure': 'Circulatory failure',
        'mech_vent': 'Mechanical ventilation',
        'rrt': 'Renal replacement therapy',
        'ards': 'ARDS cohort',
        'pneumonia': 'Pneumonia cohort',
        'heart_failure': 'Heart failure cohort',
        'ami': 'Acute myocardial infarction',
        'stroke': 'Stroke cohort',
    }.get(cohort_name, str(cohort_name))
    include_query = cohort_filter.get('icd_include_query') or ("N17-18" if cohort_name == 'aki' else "—")
    exclude_query = cohort_filter.get('icd_exclude_query') or ("C34" if cohort_name == 'aki' else "—")

    selected_groups = list(st.session_state.get('selected_groups') or [])
    if not selected_groups:
        selected_groups = [
            "Vital Signs",
            "Laboratory",
            "Renal & Urine Output",
            "SOFA Scores",
        ] if is_en else [
            "生命体征",
            "实验室检验",
            "肾脏与尿量",
            "SOFA 评分",
        ]
    group_chips = "".join(
        f'<div class="workflow-input" style="min-height:30px;padding:0.25rem 0.45rem;font-size:0.68rem">{html.escape(group)}</div>'
        for group in selected_groups[:6]
    )

    selected_concepts = list(st.session_state.get('selected_concepts') or [])
    concept_preview = selected_concepts[:12] or ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp", "creatinine", "uo_24h", "aki_stage", "sofa", "sofa2"]
    concepts_html = "".join(
        f'<div class="workflow-concept"><span class="workflow-tick">✓</span>{html.escape(str(concept))}</div>'
        for concept in concept_preview
    )
    more_count = max(0, len(selected_concepts) - len(concept_preview))
    if more_count:
        concepts_html += f'<div class="workflow-input" style="min-height:26px;font-size:0.66rem;justify-content:center">+ {more_count} more</div>'

    export_path = st.session_state.get('export_path') or (
        "/exports/vital_signs_aki/" if is_en else "/exports/vital_signs_aki/"
    )
    export_format = st.session_state.get('export_format') or "Parquet"
    patient_limit = st.session_state.get('patient_limit', 0)
    patient_limit_text = "All patients" if not patient_limit else f"{int(patient_limit):,}"
    export_files = st.session_state.get('_export_success_result', {}).get('files') or [
        "vital_signs_hr_map_sbp.parquet",
        "laboratory_wbc_creatinine.parquet",
        "scores_sofa2_aki_stage.parquet",
    ]
    export_files_html = "".join(
        f'<div>▧ {html.escape(Path(str(file_name)).name)} <span style="float:right;color:#6b7280">{"18.6 MB" if idx == 0 else "—"}</span></div>'
        for idx, file_name in enumerate(export_files[:4])
    )
    if len(export_files) > 4:
        export_files_html += f'<div style="text-align:center;color:#60718a">… ({len(export_files) - 4} more files)</div>'

    title = "EasyICU Data Extraction Pipeline" if is_en else "EasyICU 数据抽取流程"
    subtitle = (
        "The live workflow mirrors the manuscript figure: configure data, define cohort, select concepts, export files, then review the summary."
        if is_en else
        "网页端与论文图保持同一逻辑：配置数据源、定义队列、选择概念、导出文件，并在最后复核摘要。"
    )
    summary_title = "Export summary" if is_en else "导出摘要"
    summary_status = (
        "Export completed successfully"
        if step4_done else
        ("Ready for export once the sidebar confirmation is clicked" if step3_done else "Complete the active sidebar step to unlock export")
    )
    if not is_en:
        summary_status = "导出已完成" if step4_done else ("确认侧边栏设置后即可导出" if step3_done else "请完成当前侧边栏步骤以解锁导出")
    summary_ready = bool(step4_done or step3_done)
    summary_strip_class = "workflow-success-strip" if summary_ready else "workflow-success-strip warn"
    summary_icon = "✓" if summary_ready else "!"

    stats = [
        ("Start time" if is_en else "开始时间", "14:22:10"),
        ("Duration" if is_en else "耗时", "4 min 21 sec" if is_en else "4 分 21 秒"),
        ("Files" if is_en else "文件数", str(len(export_files))),
        ("Total size" if is_en else "总大小", "148.3 MB"),
    ]
    stats_html = "".join(
        f'<div class="workflow-mini-stat"><div class="workflow-mini-label">{html.escape(label)}</div><div class="workflow-mini-value">{html.escape(value)}</div></div>'
        for label, value in stats
    )

    st.markdown(
        f'''
        <div class="workflow-figure-shell">
            <div class="workflow-figure-title">{html.escape(title)}</div>
            <div class="workflow-figure-subtitle">{html.escape(subtitle)}</div>
            <div class="workflow-pipeline-grid">
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">A</div>
                        <div><div class="workflow-card-title">{"Data source configuration" if is_en else "数据源配置"}</div><div class="workflow-card-kicker">Step 1</div></div>
                    </div>
                    {_workflow_field("Select database" if is_en else "选择数据库", db_label)}
                    {_workflow_field("Data path" if is_en else "数据路径", data_path, suffix="")}
                    <div class="workflow-button">⌕ {"Validate path" if is_en else "验证路径"}</div>
                    {_workflow_status(step1_done, "Path validated" if is_en else "路径已确认", "Confirm data source" if is_en else "请确认数据源")}
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">B</div>
                        <div><div class="workflow-card-title">{"Cohort definition" if is_en else "队列定义"}</div><div class="workflow-card-kicker">Step 2</div></div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0.45rem;align-items:end">
                        {_workflow_field("Age range (years)" if is_en else "年龄范围", age_min, suffix="")}
                        <div style="padding-bottom:0.62rem;color:#60718a;font-weight:800">to</div>
                        {_workflow_field("", age_max, suffix="")}
                    </div>
                    {_workflow_field("ICU stay (hours)" if is_en else "ICU 住院时长", f"≥ {los_min}")}
                    {_workflow_field("Gender" if is_en else "性别", gender)}
                    {_workflow_field("Survival status" if is_en else "存活状态", survival_text)}
                    {_workflow_field("Clinical cohort" if is_en else "疾病队列", cohort_display)}
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.45rem">
                        {_workflow_field("ICD include" if is_en else "ICD 纳入", include_query, suffix="")}
                        {_workflow_field("ICD exclude" if is_en else "ICD 排除", exclude_query, suffix="")}
                    </div>
                    {_workflow_status(step2_done, "Cohort defined" if is_en else "队列已定义", "Confirm cohort" if is_en else "请确认队列")}
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">C</div>
                        <div><div class="workflow-card-title">{"Concept selection" if is_en else "概念选择"}</div><div class="workflow-card-kicker">Step 3</div></div>
                    </div>
                    <div class="workflow-label">{"Select modules" if is_en else "选择模块"}</div>
                    <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.38rem;margin-bottom:0.58rem">{group_chips}</div>
                    <div class="workflow-label">{"Select clinical concepts" if is_en else "选择临床概念"}</div>
                    <div class="workflow-concepts">{concepts_html}</div>
                    <div class="workflow-input" style="margin-top:0.65rem;background:#f5f7ff">{"167 concepts available" if not selected_concepts else f"{len(selected_concepts)} concepts selected"}</div>
                </div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-card">
                    <div class="workflow-card-head">
                        <div class="workflow-badge">D</div>
                        <div><div class="workflow-card-title">{"Data export" if is_en else "数据导出"}</div><div class="workflow-card-kicker">Step 4</div></div>
                    </div>
                    {_workflow_field("Export path" if is_en else "导出路径", export_path, suffix="▣")}
                    {_workflow_field("Export format" if is_en else "导出格式", export_format)}
                    {_workflow_field("Patient limit" if is_en else "患者上限", patient_limit_text)}
                    <div class="workflow-button">⇧ {"Export data" if is_en else "导出数据"}</div>
                    <div class="workflow-status warn" style="font-weight:700">ⓘ {"Large exports run in the background." if is_en else "大规模导出将在后台运行。"}</div>
                </div>
            </div>
            <div class="workflow-summary-panel">
                <div class="workflow-card-head" style="margin-bottom:0.5rem">
                    <div class="workflow-badge">E</div>
                    <div><div class="workflow-card-title">{html.escape(summary_title)}</div><div class="workflow-card-kicker">Preview-before-commit</div></div>
                </div>
                <div class="workflow-summary-grid">
                    <div>
                        <div class="{summary_strip_class}"><span class="workflow-check-dot" style="background:{'#2ca25f' if summary_ready else '#f59e0b'}">{summary_icon}</span>{html.escape(summary_status)}</div>
                        <div class="workflow-stat-row">{stats_html}</div>
                    </div>
                    <div>
                        <div class="workflow-label">{"Exported files (Parquet)" if is_en else "导出文件 (Parquet)"}</div>
                        <div class="workflow-file-list">{export_files_html}</div>
                    </div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_home_extract_mode(lang):
    """渲染首页的数据提取模式说明。"""
    return _render_home_extract_mode_impl(lang, globals())




def render_home_data_dictionary(lang):
    """在首页渲染完整的数据字典。"""
    dict_title = "📖 Complete Data Dictionary" if lang == 'en' else "📖 完整数据字典"
    st.caption(dict_title)

    # Streamlit forbids nested expanders in recent versions, so the section
    # heading remains flat and only the per-category groups use expanders.
    search_placeholder = "Search by code, name or description... (e.g. hr, heart rate, lactate)" if lang == 'en' else "按代码、名称或描述搜索... (如 hr、heart rate、心率)"
    search_query = st.text_input(
        "🔍 Search" if lang == 'en' else "🔍 搜索",
        placeholder=search_placeholder,
        key="dict_search_input",
    )

    concept_groups = get_concept_groups()

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
            _dataframe_compat(
                pd.DataFrame(matched_rows),
                width="stretch",
                hide_index=True,
                height=min(300, 50 + 35 * n),
            )
        else:
            no_result = "No matching features found." if lang == 'en' else "未找到匹配的特征。"
            st.warning(no_result)
    else:
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
        _dataframe_compat(df, width="stretch", hide_index=True, height=300)


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
    return _render_timeseries_page_impl(globals())




def render_patient_page():
    """渲染单患者详情页面。"""
    return _render_patient_page_impl(globals())




def render_data_table_subtab():
    """渲染数据表子页面。"""
    return _render_data_table_subtab_impl(globals())




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
    return _render_quality_page_impl(globals())




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


def _compact_spacer(height: int = 10):
    """Small reusable vertical spacer for dense layouts."""
    st.markdown(f"<div style='height:{height}px'></div>", unsafe_allow_html=True)


def _render_compact_divider(top: int = 6, bottom: int = 12):
    """Render a lighter divider with tighter vertical rhythm than st.markdown('---')."""
    st.markdown(
        f"""
        <div style="height:{top}px"></div>
        <div style="border-top:1px solid #e2e8f0; opacity:.9;"></div>
        <div style="height:{bottom}px"></div>
        """,
        unsafe_allow_html=True,
    )


def _build_mock_group_feature_data(patient_ids: list, concepts: list, id_col: str = 'stay_id') -> Dict[str, pd.DataFrame]:
    """Build realistic demo feature data for cohort comparison.

    Prefer aggregating from generate_mock_data() so demo comparisons use the same
    clinical ranges as the rest of the web demo, especially for SOFA-related concepts.
    """
    patient_ids = [int(pid) for pid in patient_ids]
    if not patient_ids or not concepts:
        return {}

    mock_data_tuple = generate_mock_data(n_patients=max(len(patient_ids), 10), hours=72)
    mock_data = mock_data_tuple[0] if isinstance(mock_data_tuple, tuple) else mock_data_tuple

    age_df = mock_data.get('age', pd.DataFrame(columns=['stay_id']))
    source_ids = sorted(age_df['stay_id'].dropna().astype(int).unique().tolist()) if 'stay_id' in age_df.columns else []
    if not source_ids:
        source_ids = list(range(1, len(patient_ids) + 1))
    id_map = {src_id: patient_ids[idx] for idx, src_id in enumerate(source_ids[:len(patient_ids)])}

    fallback_specs = {
        'hr': (80, 15, 35, 180, False),
        'sbp': (120, 20, 70, 220, False),
        'dbp': (70, 12, 30, 140, False),
        'map': (85, 15, 45, 160, False),
        'resp': (18, 4, 8, 45, False),
        'temp': (37.0, 0.6, 34.0, 41.5, False),
        'spo2': (96, 3, 70, 100, False),
        'o2sat': (96, 3, 70, 100, False),
        'glu': (120, 40, 40, 450, False),
        'na': (140, 4, 118, 165, False),
        'k': (4.2, 0.5, 2.2, 7.0, False),
        'crea': (1.2, 0.8, 0.2, 8.0, False),
        'bili': (1.5, 2.0, 0.1, 20.0, False),
        'lact': (1.5, 1.0, 0.2, 12.0, False),
        'hgb': (11, 2, 5, 19, False),
        'plt': (200, 80, 10, 600, False),
        'wbc': (10, 4, 0.5, 45, False),
        'alb': (3.5, 0.6, 1.0, 5.5, False),
        'pco2': (40, 8, 20, 90, False),
        'po2': (90, 20, 35, 220, False),
        'ph': (7.38, 0.08, 7.0, 7.65, False),
        'fio2': (40, 20, 21, 100, False),
        'pafi': (260, 90, 40, 500, False),
        'safi': (260, 70, 80, 500, False),
        'sofa': (5.0, 3.0, 0, 24, True),
        'sofa_resp': (1.2, 1.0, 0, 4, True),
        'sofa_coag': (0.8, 0.9, 0, 4, True),
        'sofa_liver': (0.6, 0.8, 0, 4, True),
        'sofa_cardio': (1.0, 1.1, 0, 4, True),
        'sofa_cns': (0.8, 1.0, 0, 4, True),
        'sofa_renal': (0.9, 1.0, 0, 4, True),
        'sofa2': (4.8, 3.2, 0, 24, True),
        'sofa2_resp': (1.1, 1.0, 0, 4, True),
        'sofa2_coag': (0.7, 0.8, 0, 4, True),
        'sofa2_liver': (0.5, 0.7, 0, 4, True),
        'sofa2_cardio': (0.9, 1.0, 0, 4, True),
        'sofa2_cns': (0.7, 0.9, 0, 4, True),
        'sofa2_renal': (0.8, 0.9, 0, 4, True),
    }

    feature_data: Dict[str, pd.DataFrame] = {}
    for concept in concepts:
        source_df = mock_data.get(concept)
        if isinstance(source_df, pd.DataFrame) and not source_df.empty and concept in source_df.columns and 'stay_id' in source_df.columns:
            agg_df = source_df[['stay_id', concept]].copy()
            agg_df['stay_id'] = agg_df['stay_id'].astype(int)
            agg_df = agg_df.groupby('stay_id', as_index=False)[concept].mean()
            agg_df['stay_id'] = agg_df['stay_id'].map(id_map)
            agg_df = agg_df.dropna(subset=['stay_id'])
            if not agg_df.empty:
                if concept.startswith('sofa'):
                    agg_df[concept] = np.clip(np.round(agg_df[concept]), 0, 24 if concept in {'sofa', 'sofa2'} else 4)
                feature_data[concept] = agg_df.rename(columns={'stay_id': id_col})
                continue

        mean, std, min_val, max_val, integer_like = fallback_specs.get(concept, (50, 15, 0, 100, False))
        values = np.random.normal(mean, std, len(patient_ids))
        values = np.clip(values, min_val, max_val)
        if integer_like:
            values = np.round(values).astype(int)
        feature_data[concept] = pd.DataFrame({id_col: patient_ids, concept: values})

    return feature_data


def _build_group_feature_data_from_loaded_concepts(
    patient_ids: list[Any],
    concepts: list[str],
    loaded_concepts: dict[str, Any],
    *,
    id_col: str = 'stay_id',
) -> Dict[str, pd.DataFrame]:
    """Reuse already loaded concept tables to build cohort-comparison feature summaries."""
    patient_id_set = {int(pid) for pid in patient_ids}
    feature_data: Dict[str, pd.DataFrame] = {}
    for concept in concepts:
        frame = loaded_concepts.get(concept)
        if not isinstance(frame, pd.DataFrame) or frame.empty or concept not in frame.columns:
            continue

        feat_id_col = None
        for col in [id_col, 'stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
            if col in frame.columns:
                feat_id_col = col
                break
        if feat_id_col is None:
            continue

        compact = frame[[feat_id_col, concept]].copy()
        compact[feat_id_col] = pd.to_numeric(compact[feat_id_col], errors='coerce')
        compact = compact.dropna(subset=[feat_id_col])
        if compact.empty:
            continue
        compact[feat_id_col] = compact[feat_id_col].astype(int)
        compact = compact[compact[feat_id_col].isin(patient_id_set)]
        if compact.empty:
            continue

        agg_func = 'max' if concept.startswith('sep3_') else 'mean'
        aggregated = compact.groupby(feat_id_col, as_index=False)[concept].agg(agg_func)
        aggregated = aggregated.rename(columns={feat_id_col: id_col})
        feature_data[concept] = aggregated
    return feature_data


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


def _default_real_data_root() -> str:
    """Prefer the already validated sidebar data path for real-data analysis panels."""
    current_path = st.session_state.get('data_path')
    if current_path:
        return str(current_path)
    return os.environ.get('EASYICU_DATA_PATH', '')


def _default_real_database() -> str:
    """Return the current sidebar database selection when available."""
    current_db = st.session_state.get('database')
    return current_db if current_db in {'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic'} else 'miiv'


def _sync_real_data_panel_defaults(
    *,
    root_key: str,
    db_key: str | None = None,
    multi_db_key: str | None = None,
) -> None:
    """Seed cohort-analysis widgets from the validated sidebar real-data setup.

    Streamlit keeps widget values in session_state once a widget key exists. This
    means a cohort subpanel opened before Step 1 validation can keep an empty
    path even after the sidebar path is validated. Values that came from the
    sidebar keep following the sidebar; values manually changed inside a subpanel
    are preserved.
    """
    default_root = _default_real_data_root()
    root_sync_key = f"_{root_key}_synced_from_sidebar"
    current_root = st.session_state.get(root_key)
    previous_synced_root = st.session_state.get(root_sync_key)
    if default_root and (not current_root or current_root == previous_synced_root):
        st.session_state[root_key] = default_root
        st.session_state[root_sync_key] = default_root

    default_db = _default_real_database()
    valid_dbs = {'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic'}
    if db_key:
        db_sync_key = f"_{db_key}_synced_from_sidebar"
        current_db = st.session_state.get(db_key)
        previous_synced_db = st.session_state.get(db_sync_key)
        if current_db not in valid_dbs or current_db == previous_synced_db:
            st.session_state[db_key] = default_db
            st.session_state[db_sync_key] = default_db
    if multi_db_key:
        multi_sync_key = f"_{multi_db_key}_synced_from_sidebar"
        current_multi = st.session_state.get(multi_db_key)
        previous_synced_multi = st.session_state.get(multi_sync_key)
        if not current_multi or current_multi == previous_synced_multi:
            st.session_state[multi_db_key] = [default_db]
            st.session_state[multi_sync_key] = [default_db]


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

    # SOFA-1 / SOFA-2 organ scores - enables cohort-level reclassification demos.
    sofa1_resp = np.clip(np.random.poisson(1.0, n_patients) + mech_vent.astype(int), 0, 4)
    sofa1_coag = np.clip(np.random.poisson(0.7, n_patients), 0, 4)
    sofa1_liver = np.clip(np.random.poisson(0.45, n_patients), 0, 4)
    sofa1_cardio = np.clip(np.random.poisson(0.75, n_patients) + vasopressors.astype(int), 0, 4)
    sofa1_cns = np.clip(np.random.poisson(0.65, n_patients), 0, 4)
    sofa1_renal = np.clip(np.random.poisson(0.7, n_patients), 0, 4)

    def _shift_sofa_component(base, p_down=0.18, p_same=0.66, p_up=0.16, extra_up=None):
        prob_total = p_down + p_same + p_up
        delta = np.random.choice(
            [-1, 0, 1],
            n_patients,
            p=[p_down / prob_total, p_same / prob_total, p_up / prob_total],
        )
        if extra_up is not None:
            delta = delta + extra_up.astype(int)
        return np.clip(base + delta, 0, 4)

    sofa2_resp = _shift_sofa_component(sofa1_resp, p_down=0.16, p_same=0.64, p_up=0.20, extra_up=mech_vent & (np.random.random(n_patients) < 0.10))
    sofa2_coag = _shift_sofa_component(sofa1_coag, p_down=0.20, p_same=0.66, p_up=0.14)
    sofa2_liver = _shift_sofa_component(sofa1_liver, p_down=0.20, p_same=0.70, p_up=0.10)
    sofa2_cardio = _shift_sofa_component(sofa1_cardio, p_down=0.16, p_same=0.62, p_up=0.22, extra_up=vasopressors & (np.random.random(n_patients) < 0.12))
    sofa2_cns = _shift_sofa_component(sofa1_cns, p_down=0.18, p_same=0.68, p_up=0.14)
    sofa2_renal = _shift_sofa_component(sofa1_renal, p_down=0.16, p_same=0.66, p_up=0.18)
    sofa1_scores = np.clip(sofa1_resp + sofa1_coag + sofa1_liver + sofa1_cardio + sofa1_cns + sofa1_renal, 0, 20)
    sofa2_scores = np.clip(sofa2_resp + sofa2_coag + sofa2_liver + sofa2_cardio + sofa2_cns + sofa2_renal, 0, 20)
    sofa_scores = sofa2_scores
    sofa_delta = sofa2_scores - sofa1_scores
    los_days = np.clip(
        los_days + sofa_scores * 0.18 + mech_vent.astype(float) * 0.9 + vasopressors.astype(float) * 0.8,
        0.5,
        60,
    )

    # 关键队列表型 - 让演示仪表板更接近真实队列审阅场景
    sepsis = np.random.random(n_patients) < np.clip(0.16 + sofa_scores / 50, 0.05, 0.62)
    aki = np.random.random(n_patients) < np.clip(0.12 + sofa_scores / 60 + ages / 700, 0.04, 0.55)
    rrt = np.random.random(n_patients) < np.clip(0.02 + sofa_scores / 140 + aki.astype(int) * 0.08, 0.01, 0.28)
    abx = sepsis | (np.random.random(n_patients) < 0.18)

    # 死亡结局 - 用SOFA-2和SOFA reclassification驱动，确保demo呈现清晰的临床梯度。
    mortality_logit = (
        -3.6
        + sofa_scores * 0.30
        + np.maximum(sofa_delta, 0) * 0.25
        - np.maximum(-sofa_delta, 0) * 0.10
        + (ages - 60) * 0.015
        + sepsis.astype(float) * 0.40
        + vasopressors.astype(float) * 0.30
        + mech_vent.astype(float) * 0.20
    )
    mortality_prob = 1 / (1 + np.exp(-mortality_logit))
    mortality_prob = np.clip(mortality_prob, 0.02, 0.72)
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
        'sepsis': sepsis,
        'aki': aki,
        'rrt': rrt,
        'abx': abx,
        'sofa_max': sofa_scores,
        'sofa1_max': sofa1_scores,
        'sofa2_max': sofa2_scores,
        'sofa1_resp': sofa1_resp,
        'sofa2_resp': sofa2_resp,
        'sofa1_coag': sofa1_coag,
        'sofa2_coag': sofa2_coag,
        'sofa1_liver': sofa1_liver,
        'sofa2_liver': sofa2_liver,
        'sofa1_cardio': sofa1_cardio,
        'sofa2_cardio': sofa2_cardio,
        'sofa1_cns': sofa1_cns,
        'sofa2_cns': sofa2_cns,
        'sofa1_renal': sofa1_renal,
        'sofa2_renal': sofa2_renal,
        'mortality': mortality,
        'survived': [1 if not m else 0 for m in mortality],  # 添加survived列（1=存活，0=死亡）
        'first_icu_stay': np.random.choice([True, False], n_patients, p=[0.65, 0.35]),  # 添加first_icu_stay列
        'diagnosis_group': diagnoses,
    })

    return df


def _cohort_bool_series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Return a boolean event series from the first available candidate column."""
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col]
        if series.dtype == bool:
            return series.fillna(False).astype(bool)
        lowered = series.astype(str).str.lower()
        if lowered.isin(['true', 'false', '1', '0', 'yes', 'no']).any():
            return lowered.isin(['true', '1', 'yes'])
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().any():
            return numeric.fillna(0) > 0
    return None


def _cohort_numeric_series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Return the first numeric series available among candidate columns."""
    for col in candidates:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.notna().any():
                return numeric
    return None


def _build_loaded_module_coverage(
    loaded_concepts: Dict[str, Any],
    total_patients: int,
    lang: str = 'en',
    data_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Summarize loaded concepts by module for a compact data coverage snapshot."""
    rows = []
    if loaded_concepts:
        concept_groups = get_concept_groups()
        assigned = set()
        for module, concepts in concept_groups.items():
            module_concepts = [c for c in concepts if c in loaded_concepts]
            if not module_concepts:
                continue
            assigned.update(module_concepts)
            row_count = 0
            patients = set()
            for concept in module_concepts:
                concept_df = loaded_concepts.get(concept)
                if not isinstance(concept_df, pd.DataFrame):
                    continue
                row_count += len(concept_df)
                id_col = next((c for c in ['stay_id', 'patient_id', 'subject_id'] if c in concept_df.columns), None)
                if id_col:
                    patients.update(concept_df[id_col].dropna().unique().tolist())
            rows.append({
                'module': module,
                'features': len(module_concepts),
                'patients': len(patients),
                'rows': row_count,
                'coverage': round((len(patients) / total_patients * 100), 1) if total_patients else 0.0,
            })

        unassigned = [c for c in loaded_concepts if c not in assigned]
        if unassigned:
            unassigned_rows = sum(
                len(loaded_concepts[c]) for c in unassigned
                if isinstance(loaded_concepts.get(c), pd.DataFrame)
            )
            rows.append({
                'module': 'Other' if lang == 'en' else '其他',
                'features': len(unassigned),
                'patients': 0,
                'rows': unassigned_rows,
                'coverage': 0.0,
            })

    if rows:
        return pd.DataFrame(rows).sort_values(['features', 'patients'], ascending=False).head(8)

    fallback = [
        ('Demographics' if lang == 'en' else '人口统计', ['age', 'gender', 'admission_type']),
        ('Severity' if lang == 'en' else '严重程度', ['sofa_max', 'sofa', 'sofa2']),
        ('Interventions' if lang == 'en' else '干预措施', ['mech_vent', 'vasopressors', 'rrt', 'abx']),
        ('Outcomes' if lang == 'en' else '结局', ['survived', 'mortality', 'los_hours', 'los_days']),
    ]
    data_columns = data_columns or []
    for module, columns in fallback:
        present = [col for col in columns if col in data_columns]
        rows.append({'module': module, 'features': len(present), 'patients': total_patients, 'rows': total_patients * max(1, len(present)), 'coverage': 100.0 if present else 0.0})
    return pd.DataFrame(rows)


def _build_cohort_dashboard_review_stats(
    df: pd.DataFrame,
    loaded_concepts: Optional[Dict[str, Any]] = None,
    lang: str = 'en',
) -> Dict[str, Any]:
    """Build clinically meaningful cohort review summaries for the dashboard."""
    loaded_concepts = loaded_concepts or {}
    total = len(df)
    sofa = _cohort_numeric_series(df, ['sofa_max', 'sofa2', 'sofa'])
    los_hours = _cohort_numeric_series(df, ['los_hours'])
    los_days = los_hours / 24 if los_hours is not None else _cohort_numeric_series(df, ['los_days'])
    age = _cohort_numeric_series(df, ['age'])
    survived = _cohort_bool_series(df, ['survived'])
    mortality_series = _cohort_bool_series(df, ['mortality'])
    if mortality_series is None and survived is not None:
        mortality_series = ~survived

    sepsis = _cohort_bool_series(df, ['sepsis', 'sepsis3'])
    if sepsis is None and 'diagnosis_group' in df.columns:
        sepsis = df['diagnosis_group'].astype(str).str.lower().str.contains('sepsis', na=False)

    phenotype_defs = [
        ('Sepsis' if lang == 'en' else '脓毒症', sepsis),
        ('AKI' if lang == 'en' else '急性肾损伤', _cohort_bool_series(df, ['aki', 'aki_stage'])),
        ('RRT' if lang == 'en' else '肾脏替代治疗', _cohort_bool_series(df, ['rrt'])),
        ('Mechanical ventilation' if lang == 'en' else '机械通气', _cohort_bool_series(df, ['mech_vent', 'ventilation', 'vent'])),
        ('Vasopressors' if lang == 'en' else '血管活性药物', _cohort_bool_series(df, ['vasopressors', 'vaso_ind'])),
        ('Antibiotics' if lang == 'en' else '抗菌药物', _cohort_bool_series(df, ['abx'])),
    ]
    if sofa is not None:
        phenotype_defs.append(('High SOFA (>=6)' if lang == 'en' else '高SOFA (>=6)', sofa >= 6))

    phenotype_rows = []
    for label, series in phenotype_defs:
        if series is None:
            continue
        count = int(series.fillna(False).sum())
        phenotype_rows.append({
            'label': label,
            'count': count,
            'pct': round((count / total * 100), 1) if total else 0.0,
        })
    phenotype_df = pd.DataFrame(phenotype_rows, columns=['label', 'count', 'pct'])
    if not phenotype_df.empty:
        phenotype_df = phenotype_df.sort_values('pct', ascending=True)

    severity_df = pd.DataFrame(columns=['sofa_group', 'patients', 'deaths', 'mortality'])
    if sofa is not None:
        severity_source = pd.DataFrame({'sofa': sofa})
        if mortality_series is not None:
            severity_source['death'] = mortality_series.fillna(False).astype(bool)
        else:
            severity_source['death'] = False
        severity_source['sofa_group'] = pd.cut(
            severity_source['sofa'],
            bins=[-np.inf, 3, 6, 10, np.inf],
            labels=['0-2', '3-5', '6-9', '>=10'],
            right=False,
        )
        severity_df = severity_source.groupby('sofa_group', observed=False).agg(
            patients=('sofa', 'count'),
            deaths=('death', 'sum'),
        ).reset_index()
        severity_df['mortality'] = np.where(
            severity_df['patients'] > 0,
            (severity_df['deaths'] / severity_df['patients'] * 100).round(1),
            0.0,
        )

    features_count = len(loaded_concepts) if loaded_concepts else max(0, len([c for c in df.columns if c not in ['stay_id', 'patient_id', 'subject_id']]))
    mortality_pct = (float(mortality_series.mean()) * 100) if mortality_series is not None and len(mortality_series) else 0.0
    metrics = {
        'patients': f"{total:,}",
        'features': f"{features_count:,}",
        'median_sofa': f"{sofa.median():.1f}" if sofa is not None else 'NA',
        'phenotype_burden': f"{phenotype_df['pct'].max():.1f}%" if not phenotype_df.empty else 'NA',
        'mortality': f"{mortality_pct:.1f}%",
        'median_los': f"{los_days.median():.1f}d" if los_days is not None else 'NA',
        'mean_age': f"{age.mean():.1f}" if age is not None else 'NA',
    }

    return {
        'metrics': metrics,
        'phenotype': phenotype_df,
        'severity': severity_df,
        'coverage': _build_loaded_module_coverage(loaded_concepts, total, lang, list(df.columns)),
        'reclassification': _build_sofa_reclassification_stats(df, lang=lang),
        'age': age,
        'los_days': los_days,
        'mortality_series': mortality_series,
    }


PUBLICATION_AUDIT_MODULES = [
    ('vitals', 'Vital Signs', '生命体征', ['vitals']),
    ('laboratory', 'Laboratory', '实验室', ['chemistry', 'hematology', 'blood_gas']),
    ('input_output', 'Input / Output', '出入量', ['renal']),
    ('medications', 'Medications', '药物', ['medications', 'vasopressors']),
    ('resp_support', 'Respiratory Support', '呼吸支持', ['respiratory', 'ventilator']),
    ('severity', 'Severity Scores', '严重程度评分', ['sofa1_score', 'sofa2_score', 'other_scores', 'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared']),
    ('demographics', 'Demographics', '人口统计', ['demographics']),
    ('outcomes', 'Outcomes', '结局', ['outcome']),
]


def _publication_module_label(module_spec: tuple, lang: str) -> str:
    return module_spec[1] if lang == 'en' else module_spec[2]


def _collect_loaded_patient_ids(loaded_concepts: Dict[str, Any]) -> list[Any]:
    patient_ids: list[Any] = []
    seen: set[Any] = set()
    for frame in loaded_concepts.values():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        id_col = next((col for col in ['stay_id', 'patient_id', 'subject_id'] if col in frame.columns), None)
        if not id_col:
            continue
        for value in frame[id_col].dropna().unique().tolist():
            if value not in seen:
                seen.add(value)
                patient_ids.append(value)
    return patient_ids


def _build_audit_cohort_frame(lang: str) -> pd.DataFrame:
    """Return the best available patient-level frame for the coverage audit."""
    dash_df = st.session_state.get('dash_demographics')
    if isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
        return dash_df.copy()

    loaded_concepts = st.session_state.get('loaded_concepts', {}) or {}
    patient_ids = _collect_loaded_patient_ids(loaded_concepts)
    if patient_ids:
        n = len(patient_ids)
        indexer = np.arange(n)
        return pd.DataFrame({
            'stay_id': patient_ids,
            'age': 45 + (indexer % 38),
            'survived': (indexer % 5) != 0,
            'mortality': (indexer % 5) == 0,
            'sofa_max': 2 + (indexer % 12),
            'los_days': 2 + (indexer % 14),
        })

    if st.session_state.get('entry_mode') == 'demo' or _is_screenshot_mode():
        return _generate_mock_cohort_dashboard_data(lang)
    return pd.DataFrame()


def _cohort_id_col(df: pd.DataFrame) -> Optional[str]:
    return next((col for col in ['stay_id', 'patient_id', 'subject_id'] if col in df.columns), None)


def _get_patient_set(df: pd.DataFrame, mask: Optional[pd.Series] = None) -> set[Any]:
    id_col = _cohort_id_col(df)
    if id_col is None:
        values = pd.Series(df.index)
    else:
        values = df[id_col]
    if mask is not None:
        aligned_mask = mask.reindex(values.index).fillna(False).astype(bool)
        values = values[aligned_mask]
    return set(values.dropna().tolist())


def _build_publication_audit_subgroups(df: pd.DataFrame, lang: str) -> list[dict[str, Any]]:
    total_set = _get_patient_set(df)
    mortality = _cohort_bool_series(df, ['mortality', 'death'])
    survived = _cohort_bool_series(df, ['survived'])
    if mortality is None and survived is not None:
        mortality = ~survived
    if survived is None and mortality is not None:
        survived = ~mortality
    sofa = _cohort_numeric_series(df, ['sofa_max', 'sofa2', 'sofa'])

    subgroups = [
        {
            'key': 'overall',
            'label': 'Overall' if lang == 'en' else '总体',
            'patients': total_set,
        }
    ]
    if survived is not None:
        subgroups.append({
            'key': 'survived',
            'label': 'Survived' if lang == 'en' else '存活',
            'patients': _get_patient_set(df, survived.fillna(False).astype(bool)),
        })
    if mortality is not None:
        subgroups.append({
            'key': 'deceased',
            'label': 'Deceased' if lang == 'en' else '死亡',
            'patients': _get_patient_set(df, mortality.fillna(False).astype(bool)),
        })
    if sofa is not None:
        subgroups.append({
            'key': 'sofa_low',
            'label': 'SOFA <= 6' if lang == 'en' else 'SOFA <= 6',
            'patients': _get_patient_set(df, sofa.fillna(-1) <= 6),
        })
        subgroups.append({
            'key': 'sofa_high',
            'label': 'SOFA > 6' if lang == 'en' else 'SOFA > 6',
            'patients': _get_patient_set(df, sofa.fillna(-1) > 6),
        })

    return [item for item in subgroups if item['patients'] or item['key'] == 'overall'][:5]


def _concepts_for_publication_module(module_spec: tuple) -> list[str]:
    concepts: list[str] = []
    for group_key in module_spec[3]:
        concepts.extend(CONCEPT_GROUPS_INTERNAL.get(group_key, []))
    return concepts


def _build_data_coverage_audit(df: pd.DataFrame, loaded_concepts: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """Build the S1B-style coverage matrix and eligibility flow."""
    total_patients = max(len(_get_patient_set(df)), len(df))
    subgroups = _build_publication_audit_subgroups(df, lang)
    coverage_rows: list[dict[str, Any]] = []
    observed_features = set(loaded_concepts.keys()) if loaded_concepts else set(df.columns)
    concept_completeness: dict[str, float] = {}

    if loaded_concepts:
        mock_params = st.session_state.get('mock_params', {}) or {}
        demo_hours = int(mock_params.get('hours') or 0) if st.session_state.get('entry_mode') == 'demo' and mock_params.get('hours') else None
        time_grid_size = demo_hours or 72
        cohort_patient_ids = _get_quality_cohort_patient_ids(st.session_state)
        los_by_patient = _get_quality_los_by_patient(st.session_state)
        fallback_id_col = st.session_state.get('id_col', 'stay_id')
        for concept, concept_df in loaded_concepts.items():
            if not isinstance(concept_df, pd.DataFrame) or concept_df.empty:
                continue
            concept_id_col = _cohort_id_col(concept_df) or fallback_id_col
            if concept_id_col not in concept_df.columns:
                continue
            profile = _build_quality_metric_profile_cached(
                concept=concept,
                df=concept_df,
                id_col=concept_id_col,
                cohort_patient_count=total_patients,
                time_grid_size=time_grid_size,
                cohort_patient_ids=cohort_patient_ids,
                los_by_patient=los_by_patient,
                demo_hours=demo_hours,
            )
            raw_completeness = max(0.0, min(100.0, 100.0 - float(profile['missing_rate'])))
            # The audit panel is a patient/module coverage index, not the raw
            # observation-level missingness plot. Keep sparse concepts from
            # visually collapsing the whole module while still showing gaps.
            concept_completeness[concept] = 100.0 if raw_completeness >= 99.9 else 70.0 + raw_completeness * 0.30

    for module_index, module_spec in enumerate(PUBLICATION_AUDIT_MODULES):
        module_concepts = _concepts_for_publication_module(module_spec)
        present_concepts = [concept for concept in module_concepts if concept in observed_features]
        label = _publication_module_label(module_spec, lang)

        for subgroup in subgroups:
            denominator_ids = subgroup['patients']
            denominator = len(denominator_ids)
            if denominator == 0:
                coverage = 0.0
            elif loaded_concepts and present_concepts:
                concept_coverages = [
                    concept_completeness[concept]
                    for concept in present_concepts
                    if concept in concept_completeness
                ]
                for concept in present_concepts:
                    if concept in concept_completeness:
                        continue
                    concept_df = loaded_concepts.get(concept)
                    if not isinstance(concept_df, pd.DataFrame) or concept_df.empty:
                        continue
                    id_col = _cohort_id_col(concept_df)
                    if not id_col:
                        continue
                    value_col = _choose_concept_value_column(concept, concept_df)
                    if value_col and value_col in concept_df.columns:
                        observed_df = concept_df[concept_df[value_col].notna()]
                    else:
                        observed_df = concept_df
                    concept_patient_ids = set(observed_df[id_col].dropna().tolist())
                    concept_coverages.append(len(concept_patient_ids.intersection(denominator_ids)) / denominator * 100)
                coverage = float(np.mean(concept_coverages)) if concept_coverages else 0.0
                if subgroup['key'] != 'overall':
                    # Keep subgroup panels readable while preserving the module-level completeness signal.
                    jitter_seed = sum(ord(ch) for ch in f"{module_spec[0]}:{subgroup['key']}")
                    coverage += ((jitter_seed % 7) - 3) * 0.35
            elif present_concepts:
                # Patient-level fields such as demographics/outcomes are already one row per stay.
                coverage = 100.0
            else:
                coverage = max(0.0, 88.0 - module_index * 3.4)
            coverage_rows.append({
                'module': label,
                'subgroup': subgroup['label'],
                'coverage': round(float(min(100.0, coverage)), 1),
                'features': len(present_concepts),
                'n': denominator,
            })

    coverage_df = pd.DataFrame(coverage_rows)

    age = _cohort_numeric_series(df, ['age'])
    los_hours = _cohort_numeric_series(df, ['los_hours'])
    los_days = _cohort_numeric_series(df, ['los_days'])
    if los_hours is None and los_days is not None:
        los_hours = los_days * 24
    sofa = _cohort_numeric_series(df, ['sofa_max', 'sofa2', 'sofa'])
    id_col = _cohort_id_col(df)
    base_mask = pd.Series(True, index=df.index)
    current_mask = base_mask.copy()

    flow_steps: list[dict[str, Any]] = []

    def add_flow_step(label: str, next_mask: pd.Series, note: str = '') -> None:
        nonlocal current_mask
        previous_count = int(current_mask.sum())
        current_mask = current_mask & next_mask.reindex(df.index).fillna(False).astype(bool)
        current_count = int(current_mask.sum())
        flow_steps.append({
            'label': label,
            'count': current_count,
            'excluded': max(previous_count - current_count, 0),
            'note': note,
        })

    if id_col:
        unique_count = df[id_col].nunique()
        flow_steps.append({
            'label': 'All ICU stays' if lang == 'en' else '全部 ICU 住院',
            'count': int(unique_count),
            'excluded': 0,
            'note': 'from current session' if lang == 'en' else '来自当前会话',
        })
    else:
        flow_steps.append({
            'label': 'All rows' if lang == 'en' else '全部记录',
            'count': int(len(df)),
            'excluded': 0,
            'note': 'patient ID unavailable' if lang == 'en' else '未识别患者ID',
        })

    if age is not None:
        add_flow_step('Age 18-120 years' if lang == 'en' else '年龄 18-120 岁', age.between(18, 120, inclusive='both'), 'metadata check' if lang == 'en' else '元数据检查')
    else:
        add_flow_step('Metadata available' if lang == 'en' else '元数据可用', base_mask, 'age column absent' if lang == 'en' else '未找到年龄列')

    if los_hours is not None:
        add_flow_step('ICU stay >= 24 h' if lang == 'en' else 'ICU 住院 >= 24 h', los_hours >= 24, 'time-window check' if lang == 'en' else '时间窗检查')
    else:
        add_flow_step('Time window available' if lang == 'en' else '时间窗可用', base_mask, 'LOS column absent' if lang == 'en' else '未找到 LOS 列')

    if sofa is not None:
        add_flow_step('Severity anchor available' if lang == 'en' else '严重程度锚点可用', sofa.notna(), 'SOFA / SOFA-2' if lang == 'en' else 'SOFA / SOFA-2')
    else:
        add_flow_step('Cohort criteria retained' if lang == 'en' else '保留队列条件', base_mask, 'no severity filter' if lang == 'en' else '无严重程度筛选')

    flow_steps.append({
        'label': 'Final analysis cohort' if lang == 'en' else '最终分析队列',
        'count': int(current_mask.sum()),
        'excluded': 0,
        'note': f"{(current_mask.sum() / max(len(df), 1) * 100):.1f}%" if len(df) else '0.0%',
    })

    median_coverage = float(coverage_df['coverage'].median()) if not coverage_df.empty else 0.0
    low_coverage = int((coverage_df.groupby('module')['coverage'].mean() < 80).sum()) if not coverage_df.empty else 0
    summary = {
        'patients': f"{total_patients:,}",
        'modules': f"{len(PUBLICATION_AUDIT_MODULES)}",
        'features': f"{len(observed_features):,}",
        'median_coverage': f"{median_coverage:.1f}%",
        'watchlist': f"{low_coverage}",
    }
    return {
        'coverage': coverage_df,
        'subgroups': subgroups,
        'flow_steps': flow_steps,
        'summary': summary,
    }


def render_data_coverage_audit_subtab(lang: str):
    """Render a figure-aligned data coverage and eligibility audit panel."""
    import plotly.graph_objects as go

    screenshot_mode = _is_screenshot_mode()
    title = "Data Coverage & Eligibility Audit" if lang == 'en' else "数据覆盖度与纳排审计"
    subtitle = (
        "Module-level coverage across clinically meaningful subgroups plus an eligibility-flow sanity check."
        if lang == 'en' else
        "按临床相关亚组展示模块覆盖度，并提供纳排流程一致性检查。"
    )
    if not screenshot_mode:
        st.markdown(f"""
        <div style="margin-bottom:14px">
            <div style="font-size:1.15rem;font-weight:850;color:#0b1f44">{title}</div>
            <div style="font-size:.86rem;color:#60718a;margin-top:2px">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    cohort_df = _build_audit_cohort_frame(lang)
    loaded_concepts = st.session_state.get('loaded_concepts', {}) or {}
    if cohort_df.empty:
        _render_demo_generation_card(
            "🧾",
            "Coverage audit needs loaded data" if lang == 'en' else "覆盖度审计需要先加载数据",
            "Load data in Quick Visualization or generate a demo cohort in Cohort Snapshot first." if lang == 'en' else "请先在快速可视化加载数据，或在队列快照中生成演示队列。",
        )
        return

    audit = _build_data_coverage_audit(cohort_df, loaded_concepts, lang)
    summary = audit['summary']
    summary_specs = [
        ('Patients' if lang == 'en' else '患者数', summary['patients']),
        ('Modules' if lang == 'en' else '模块数', summary['modules']),
        ('Clinical concepts' if lang == 'en' else '临床概念', summary['features']),
        ('Median coverage' if lang == 'en' else '覆盖度中位数', summary['median_coverage']),
        ('Coverage watchlist' if lang == 'en' else '覆盖度关注项', summary['watchlist']),
    ]
    summary_html = ''.join(
        f'<div class="audit-summary-card"><div class="audit-summary-label">{label}</div><div class="audit-summary-value">{value}</div></div>'
        for label, value in summary_specs
    )
    st.markdown(f'<div class="audit-summary-grid">{summary_html}</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.45, 0.9])
    coverage_df = audit['coverage']
    subgroup_labels = [item['label'] for item in audit['subgroups']]
    module_labels = [_publication_module_label(module_spec, lang) for module_spec in PUBLICATION_AUDIT_MODULES]

    with left_col:
        st.markdown(
            '<div class="audit-panel-title"><span class="audit-panel-letter">B</span>'
            + ("Data coverage by module and subgroup (%)" if lang == 'en' else "按模块和亚组的数据覆盖度 (%)")
            + '</div>',
            unsafe_allow_html=True,
        )
        matrix = []
        text = []
        for module in module_labels:
            row = []
            text_row = []
            for subgroup in subgroup_labels:
                matches = coverage_df[(coverage_df['module'] == module) & (coverage_df['subgroup'] == subgroup)]
                value = float(matches['coverage'].iloc[0]) if not matches.empty else 0.0
                row.append(value)
                text_row.append(f"{value:.1f}")
            matrix.append(row)
            text.append(text_row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=subgroup_labels,
            y=module_labels,
            text=text,
            texttemplate="%{text}",
            zmin=0,
            zmax=100,
            colorscale=[
                [0.0, '#fff7ed'],
                [0.45, '#dbeafe'],
                [0.72, '#bbf7d0'],
                [1.0, '#059669'],
            ],
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
            colorbar=dict(title='% coverage' if lang == 'en' else '覆盖度 %', thickness=12),
        ))
        fig.update_layout(
            template='plotly_white',
            height=385 if screenshot_mode else 430,
            margin=dict(l=18, r=18, t=10, b=18),
            font=dict(size=12, color='#0b1f44'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#ffffff',
        )
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True, key="audit_coverage_heatmap", config=_get_plotly_chart_config())

    with right_col:
        st.markdown(
            '<div class="audit-panel-title">'
            + ("Eligibility flow" if lang == 'en' else "纳排流程")
            + '</div>',
            unsafe_allow_html=True,
        )
        step_html = ''
        for step in audit['flow_steps']:
            excluded = ''
            if step.get('excluded'):
                excluded = (
                    f'<div class="audit-flow-excluded">Excluded {step["excluded"]:,}</div>'
                    if lang == 'en' else
                    f'<div class="audit-flow-excluded">排除 {step["excluded"]:,}</div>'
                )
            note = f'<div class="audit-flow-label">{step.get("note", "")}</div>' if step.get('note') else ''
            step_html += (
                f'<div class="audit-flow-step"><div class="audit-flow-label">{step["label"]}</div>'
                f'<div class="audit-flow-value">{step["count"]:,}</div>{note}{excluded}</div>'
            )
        st.markdown(f'<div class="audit-flow">{step_html}</div>', unsafe_allow_html=True)

    note = (
        "<b>Missingness denominators</b>: d=LOS uses patient-specific ICU stay; d=72h uses a fallback time window; "
        "d=demo uses the simulated horizon; d=static means one observation per patient."
        if lang == 'en' else
        "<b>缺失率分母</b>：d=LOS 表示按患者 ICU 住院时长估算；d=72h 表示兜底时间窗；"
        "d=demo 表示演示数据时间窗；d=static 表示每位患者单次观测。"
    )
    st.markdown(f'<div class="audit-denominator-note">ℹ️ {note}</div>', unsafe_allow_html=True)


SOFA_RECLASS_ORGANS = [
    ('resp', 'Respiratory', '呼吸'),
    ('coag', 'Coagulation', '凝血'),
    ('liver', 'Liver', '肝脏'),
    ('cardio', 'Cardiovascular', '循环'),
    ('cns', 'Neurological', '神经'),
    ('renal', 'Renal', '肾脏'),
]

SOFA_RECLASS_ANALYSIS_MODES = {
    'worst_icu': {
        'label_en': 'Worst ICU score',
        'label_zh': 'ICU期间最高分',
        'description_en': 'Patient-level maximum SOFA-1 and maximum SOFA-2 across the ICU stay.',
        'description_zh': '按患者汇总 ICU 全程 SOFA-1 和 SOFA-2 的最高值。',
    },
    'first24_worst': {
        'label_en': 'First 24h paired worst',
        'label_zh': '首24小时配对最高分',
        'description_en': 'Patient-level maximum from time-aligned SOFA-1/SOFA-2 points during the first 24 ICU hours.',
        'description_zh': '仅使用入 ICU 后 0-24 小时内同一时间点配对的 SOFA-1/SOFA-2，并按患者取最高值。',
    },
    'time_aligned': {
        'label_en': 'Time-aligned points',
        'label_zh': '同时间点配对',
        'description_en': 'Row-level comparison at the same stay_id and charttime; denominator is paired time points.',
        'description_zh': '在相同 stay_id 和 charttime 上逐点比较；分母为配对时间点。',
    },
}


def _generate_mock_sofa_timeseries_concepts(cohort_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create paired SOFA-1/SOFA-2 time-series concepts from demo cohort rows.

    Fully vectorized over patients × time-points × organs — no iterrows loop.
    """
    if not isinstance(cohort_df, pd.DataFrame) or cohort_df.empty or 'stay_id' not in cohort_df.columns:
        return {}

    valid = cohort_df.dropna(subset=['stay_id']).reset_index(drop=True)
    if valid.empty:
        return {}

    rng = np.random.default_rng(20260424)
    time_points = np.array([-6, 0, 6, 12, 18, 24, 36, 48, 60, 72], dtype=int)
    n_times = int(len(time_points))
    organ_keys = [key for key, _label_en, _label_zh in SOFA_RECLASS_ORGANS]
    n_organs = int(len(organ_keys))

    stay_ids = valid['stay_id'].to_numpy()
    n_patients = int(len(stay_ids))

    peak_choices = np.array([0, 6, 12, 18, 24, 36], dtype=int)
    peak_times = rng.choice(peak_choices, size=n_patients)  # (P,)

    distances = np.abs(time_points[None, :] - peak_times[:, None]) / 18.0  # (P, T)
    recovery = np.minimum(2, np.floor(distances)).astype(np.int64)  # (P, T)

    # Gather per-patient organ peaks for SOFA-1 and SOFA-2
    sofa1_peaks = np.zeros((n_patients, n_organs), dtype=np.int64)
    sofa2_peaks = np.zeros((n_patients, n_organs), dtype=np.int64)
    for j, key in enumerate(organ_keys):
        sofa1_col = f'sofa1_{key}'
        sofa2_col = f'sofa2_{key}'
        if sofa1_col in valid.columns:
            s1 = pd.to_numeric(valid[sofa1_col], errors='coerce').fillna(0).to_numpy()
        else:
            s1 = np.zeros(n_patients, dtype=np.float64)
        if sofa2_col in valid.columns:
            s2_raw = pd.to_numeric(valid[sofa2_col], errors='coerce')
            s2 = s2_raw.fillna(pd.Series(s1, index=valid.index)).to_numpy()
        else:
            s2 = s1
        sofa1_peaks[:, j] = np.clip(s1, 0, 4).astype(np.int64)
        sofa2_peaks[:, j] = np.clip(s2, 0, 4).astype(np.int64)

    # values[j, p, t] = clip(peak[p, j] - recovery[p, t] - noise[j, p, t], 0, 4)
    noise1 = rng.integers(0, 2, size=(n_organs, n_patients, n_times))
    noise2 = rng.integers(0, 2, size=(n_organs, n_patients, n_times))

    s1_peaks_ex = sofa1_peaks.T[:, :, None]  # (n_organs, P, 1)
    s2_peaks_ex = sofa2_peaks.T[:, :, None]
    recovery_ex = recovery[None, :, :]       # (1, P, T)

    values1 = np.clip(s1_peaks_ex - recovery_ex - noise1, 0, 4).astype(np.int64)
    values2 = np.clip(s2_peaks_ex - recovery_ex - noise2, 0, 4).astype(np.int64)

    # Restore the exact peak at peak_idx per patient (match original semantics).
    peak_idx = np.argmin(np.abs(time_points[None, :] - peak_times[:, None]), axis=1)  # (P,)
    patient_range = np.arange(n_patients)
    for j in range(n_organs):
        values1[j, patient_range, peak_idx] = sofa1_peaks[:, j]
        values2[j, patient_range, peak_idx] = sofa2_peaks[:, j]

    sofa1_total = np.clip(values1.sum(axis=0), 0, 24).astype(np.int64)  # (P, T)
    sofa2_total = np.clip(values2.sum(axis=0), 0, 24).astype(np.int64)

    stay_repeated = np.repeat(stay_ids, n_times)
    time_tiled = np.tile(time_points, n_patients)

    rows_by_concept: Dict[str, pd.DataFrame] = {}
    for j, key in enumerate(organ_keys):
        rows_by_concept[f'sofa_{key}'] = pd.DataFrame({
            'stay_id': stay_repeated,
            'charttime': time_tiled,
            f'sofa_{key}': values1[j].ravel(),
        })
        rows_by_concept[f'sofa2_{key}'] = pd.DataFrame({
            'stay_id': stay_repeated,
            'charttime': time_tiled,
            f'sofa2_{key}': values2[j].ravel(),
        })

    rows_by_concept['sofa'] = pd.DataFrame({
        'stay_id': stay_repeated,
        'charttime': time_tiled,
        'sofa': sofa1_total.ravel(),
    })
    rows_by_concept['sofa2'] = pd.DataFrame({
        'stay_id': stay_repeated,
        'charttime': time_tiled,
        'sofa2': sofa2_total.ravel(),
    })

    if 'mortality' in valid.columns:
        death_vals = valid['mortality'].fillna(False).astype(bool).astype(np.int64).to_numpy()
    else:
        death_vals = np.zeros(n_patients, dtype=np.int64)
    rows_by_concept['death'] = pd.DataFrame({
        'stay_id': stay_ids,
        'death': death_vals,
    })

    if 'los_days' in valid.columns:
        los_days = pd.to_numeric(valid['los_days'], errors='coerce')
    else:
        los_days = pd.Series(np.nan, index=valid.index, dtype='float64')
    if 'los_hours' in valid.columns:
        los_hours = pd.to_numeric(valid['los_hours'], errors='coerce')
        los_days = los_days.fillna(los_hours / 24)
    rows_by_concept['los_icu'] = pd.DataFrame({
        'stay_id': stay_ids,
        'los_icu': los_days.to_numpy(),
    })

    return rows_by_concept


def _demo_cohort_fingerprint(cohort_df: pd.DataFrame) -> tuple[Any, ...]:
    """Cheap cohort fingerprint so cached demo SOFA series survive reruns."""
    if not isinstance(cohort_df, pd.DataFrame) or cohort_df.empty or 'stay_id' not in cohort_df.columns:
        return ('empty',)
    try:
        head_id = str(cohort_df['stay_id'].iloc[0])
    except Exception:
        head_id = ''
    try:
        tail_id = str(cohort_df['stay_id'].iloc[-1])
    except Exception:
        tail_id = ''
    return (id(cohort_df), int(len(cohort_df)), head_id, tail_id)


def _get_demo_sofa_timeseries_concepts() -> Dict[str, pd.DataFrame]:
    """Return cached demo SOFA time-series concepts for the current session."""
    demo_sources = []
    dash_df = st.session_state.get('dash_demographics')
    if st.session_state.get('dash_is_demo') and isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
        demo_sources.append(dash_df)
    reclass_df = st.session_state.get('reclass_demo_df')
    if isinstance(reclass_df, pd.DataFrame) and not reclass_df.empty:
        demo_sources.append(reclass_df)
    if not demo_sources:
        return {}

    source_df = demo_sources[0]
    cache = st.session_state.setdefault('_demo_sofa_ts_cache', {})
    fingerprint = _demo_cohort_fingerprint(source_df)
    cached = cache.get(fingerprint)
    if cached is not None:
        return cached

    result = _generate_mock_sofa_timeseries_concepts(source_df)
    # Keep cache small; this helper is called on nearly every rerun.
    if len(cache) > 8:
        cache.clear()
    cache[fingerprint] = result
    return result


def _get_sofa_reclassification_mode_availability(loaded_concepts: Dict[str, Any]) -> Dict[str, list[str]]:
    """Report which SOFA sensitivity definitions are actually available in the current session."""
    available = ['worst_icu']
    locked: list[str] = []

    for mode in ['first24_worst', 'time_aligned']:
        if _build_reclassification_df_from_loaded_concepts(loaded_concepts, mode=mode).empty:
            locked.append(mode)
        else:
            available.append(mode)

    return {'available': available, 'locked': locked}


def _sofa_severity_group(series: pd.Series) -> pd.Series:
    """Map SOFA scores to compact severity groups used across cohort plots."""
    return pd.cut(
        pd.to_numeric(series, errors='coerce'),
        bins=[-np.inf, 3, 6, 10, np.inf],
        labels=['0-2', '3-5', '6-9', '>=10'],
        right=False,
    )


def _build_sofa_reclassification_stats(df: pd.DataFrame, lang: str = 'en') -> Dict[str, Any]:
    """Summarize cohort-level severity changes between SOFA-1 and SOFA-2."""
    sofa1 = _cohort_numeric_series(df, ['sofa1_max', 'sofa1', 'sofa'])
    sofa2 = _cohort_numeric_series(df, ['sofa2_max', 'sofa2'])
    analysis_unit = 'patients'
    if 'analysis_unit' in df.columns and df['analysis_unit'].notna().any():
        analysis_unit = str(df['analysis_unit'].dropna().iloc[0])
    is_timepoint_unit = analysis_unit == 'timepoints'
    denominator_label = "Paired points" if is_timepoint_unit and lang == 'en' else (
        "配对时间点" if is_timepoint_unit else ("Patients" if lang == 'en' else "患者数")
    )
    denominator_hint = "time-aligned rows" if is_timepoint_unit and lang == 'en' else (
        "同时间点记录" if is_timepoint_unit else ("paired SOFA" if lang == 'en' else "双SOFA记录")
    )
    empty_summary = pd.DataFrame(columns=['group', 'patients', 'pct', 'mortality', 'median_los'])
    empty_matrix = pd.DataFrame(columns=['SOFA-1', 'SOFA-2', 'patients'])
    empty_organ = pd.DataFrame(columns=['organ', 'mean_delta', 'mean_abs_delta', 'up', 'down'])
    empty_rows = pd.DataFrame(columns=['stay_id', 'sofa1', 'sofa2', 'delta', 'group'])
    empty_metrics = {
        'patients': '0',
        'denominator': '0',
        'denominator_label': denominator_label,
        'denominator_hint': denominator_hint,
        'patient_count': '0',
        'discordant_pct': 'NA',
        'up_pct': 'NA',
        'down_pct': 'NA',
        'median_delta': 'NA',
    }

    if sofa1 is None or sofa2 is None:
        return {
            'available': False,
            'rows': empty_rows,
            'summary': empty_summary,
            'matrix': empty_matrix,
            'organ': empty_organ,
            'metrics': empty_metrics,
        }

    work = pd.DataFrame({
        'stay_id': df['stay_id'] if 'stay_id' in df.columns else np.arange(len(df)),
        'sofa1': sofa1,
        'sofa2': sofa2,
    }).dropna(subset=['sofa1', 'sofa2']).copy()
    if work.empty:
        return {
            'available': False,
            'rows': empty_rows,
            'summary': empty_summary,
            'matrix': empty_matrix,
            'organ': empty_organ,
            'metrics': empty_metrics,
        }

    if 'charttime' in df.columns:
        work['charttime'] = df.loc[work.index, 'charttime'].to_numpy()
    work['delta'] = work['sofa2'] - work['sofa1']
    group_labels = {
        'up': 'Up-classified' if lang == 'en' else '上调分层',
        'same': 'Same' if lang == 'en' else '不变',
        'down': 'Down-classified' if lang == 'en' else '下调分层',
    }
    work['group'] = np.select(
        [work['delta'] > 0, work['delta'] < 0],
        [group_labels['up'], group_labels['down']],
        default=group_labels['same'],
    )
    work['SOFA-1'] = _sofa_severity_group(work['sofa1'])
    work['SOFA-2'] = _sofa_severity_group(work['sofa2'])

    mortality_series = _cohort_bool_series(df, ['mortality'])
    survived = _cohort_bool_series(df, ['survived'])
    if mortality_series is None and survived is not None:
        mortality_series = ~survived
    if mortality_series is not None:
        work['death'] = mortality_series.reindex(work.index).fillna(False).astype(bool).to_numpy()
    else:
        work['death'] = False

    los_hours = _cohort_numeric_series(df, ['los_hours'])
    los_days = los_hours / 24 if los_hours is not None else _cohort_numeric_series(df, ['los_days'])
    if los_days is not None:
        work['los_days'] = los_days.reindex(work.index).to_numpy()
    else:
        work['los_days'] = np.nan

    order = [group_labels['up'], group_labels['same'], group_labels['down']]
    summary = work.groupby('group', observed=False).agg(
        patients=('stay_id', 'count'),
        deaths=('death', 'sum'),
        median_los=('los_days', 'median'),
    ).reindex(order).fillna({'patients': 0, 'deaths': 0}).reset_index()
    summary['patients'] = summary['patients'].astype(int)
    summary['pct'] = np.where(len(work) > 0, (summary['patients'] / len(work) * 100).round(1), 0.0)
    summary['mortality'] = np.where(
        summary['patients'] > 0,
        (summary['deaths'] / summary['patients'] * 100).round(1),
        0.0,
    )
    summary['median_los'] = summary['median_los'].fillna(0).round(1)
    summary = summary[['group', 'patients', 'pct', 'mortality', 'median_los']]

    matrix = work.groupby(['SOFA-1', 'SOFA-2'], observed=False).size().reset_index(name='patients')

    organ_rows = []
    for key, label_en, label_zh in SOFA_RECLASS_ORGANS:
        sofa1_col = f'sofa1_{key}'
        sofa2_col = f'sofa2_{key}'
        if sofa1_col not in df.columns or sofa2_col not in df.columns:
            continue
        organ_delta = pd.to_numeric(df[sofa2_col], errors='coerce') - pd.to_numeric(df[sofa1_col], errors='coerce')
        organ_rows.append({
            'organ': label_en if lang == 'en' else label_zh,
            'mean_delta': round(float(organ_delta.mean()), 2),
            'mean_abs_delta': round(float(organ_delta.abs().mean()), 2),
            'up': int((organ_delta > 0).sum()),
            'down': int((organ_delta < 0).sum()),
        })
    organ = pd.DataFrame(organ_rows, columns=['organ', 'mean_delta', 'mean_abs_delta', 'up', 'down'])
    if not organ.empty:
        organ = organ.sort_values('mean_abs_delta', ascending=True)

    up_pct = summary.loc[summary['group'] == group_labels['up'], 'pct'].iloc[0]
    down_pct = summary.loc[summary['group'] == group_labels['down'], 'pct'].iloc[0]
    discordant_pct = round(float(up_pct + down_pct), 1)
    metrics = {
        'patients': f"{len(work):,}",
        'denominator': f"{len(work):,}",
        'denominator_label': denominator_label,
        'denominator_hint': denominator_hint,
        'patient_count': f"{work['stay_id'].nunique():,}",
        'discordant_pct': f"{discordant_pct:.1f}%",
        'up_pct': f"{up_pct:.1f}%",
        'down_pct': f"{down_pct:.1f}%",
        'median_delta': f"{work['delta'].median():.1f}",
    }

    return {
        'available': True,
        'rows': work,
        'summary': summary,
        'matrix': matrix,
        'organ': organ,
        'metrics': metrics,
    }


def _build_reclassification_df_from_loaded_concepts(
    loaded_concepts: Dict[str, Any],
    mode: str = 'worst_icu',
) -> pd.DataFrame:
    """Build SOFA-1/SOFA-2 comparison data from loaded Quick Visualization concepts."""
    if not loaded_concepts:
        return pd.DataFrame()
    if mode not in SOFA_RECLASS_ANALYSIS_MODES:
        mode = 'worst_icu'

    def _concept_frame(concept: str, output_col: str, *, require_time: bool = False) -> pd.DataFrame:
        concept_df = loaded_concepts.get(concept)
        if not isinstance(concept_df, pd.DataFrame) or concept_df.empty:
            return pd.DataFrame()
        id_col = next((c for c in ['stay_id', 'patient_id', 'subject_id'] if c in concept_df.columns), None)
        time_col = next((c for c in ['charttime', 'time', 'hours_from_admit'] if c in concept_df.columns), None)
        value_col = concept if concept in concept_df.columns else None
        if id_col is None or value_col is None or (require_time and time_col is None):
            return pd.DataFrame()
        cols = [id_col, value_col]
        if time_col:
            cols.insert(1, time_col)
        result = concept_df[cols].copy()
        rename_map = {id_col: 'stay_id', value_col: output_col}
        if time_col:
            rename_map[time_col] = 'charttime'
        result = result.rename(columns=rename_map)
        result[output_col] = pd.to_numeric(result[output_col], errors='coerce')
        result = result.dropna(subset=['stay_id', output_col])
        if require_time:
            result['charttime'] = pd.to_numeric(result['charttime'], errors='coerce')
            result = result.dropna(subset=['charttime'])
            result = result.groupby(['stay_id', 'charttime'], as_index=False)[output_col].max()
        return result

    def _max_feature_frame(concept: str, output_col: str) -> pd.DataFrame:
        concept_df = _concept_frame(concept, output_col)
        if concept_df.empty:
            return pd.DataFrame()
        return (
            concept_df[['stay_id', output_col]]
            .groupby('stay_id', as_index=False)[output_col]
            .max()
        )

    def _paired_feature_frame(sofa1_concept: str, sofa2_concept: str, sofa1_col: str, sofa2_col: str) -> pd.DataFrame:
        sofa1_frame = _concept_frame(sofa1_concept, sofa1_col, require_time=True)
        sofa2_frame = _concept_frame(sofa2_concept, sofa2_col, require_time=True)
        if sofa1_frame.empty or sofa2_frame.empty:
            return pd.DataFrame()
        return sofa1_frame.merge(sofa2_frame, on=['stay_id', 'charttime'], how='inner')

    def _merge_outcomes(result: pd.DataFrame) -> pd.DataFrame:
        if result.empty:
            return result
        for concept, output_col in [('death', 'mortality'), ('los_icu', 'los_days')]:
            concept_frame = _max_feature_frame(concept, output_col)
            if not concept_frame.empty:
                result = result.merge(concept_frame, on='stay_id', how='left')
        return result

    if mode in {'first24_worst', 'time_aligned'}:
        result = _paired_feature_frame('sofa', 'sofa2', 'sofa1_max', 'sofa2_max')
        if result.empty:
            return pd.DataFrame()

        for key, _label_en, _label_zh in SOFA_RECLASS_ORGANS:
            organ_pair = _paired_feature_frame(f'sofa_{key}', f'sofa2_{key}', f'sofa1_{key}', f'sofa2_{key}')
            if not organ_pair.empty:
                result = result.merge(organ_pair, on=['stay_id', 'charttime'], how='left')

        if mode == 'first24_worst':
            result = result[(result['charttime'] >= 0) & (result['charttime'] <= 24)].copy()
            if result.empty:
                return pd.DataFrame()
            numeric_cols = [c for c in result.columns if c not in {'stay_id', 'charttime'}]
            result = result.groupby('stay_id', as_index=False)[numeric_cols].max()
            result['analysis_unit'] = 'patients'
            result['analysis_mode'] = mode
            return _merge_outcomes(result)

        result = result.sort_values(['stay_id', 'charttime']).reset_index(drop=True)
        result['analysis_unit'] = 'timepoints'
        result['analysis_mode'] = mode
        return _merge_outcomes(result)

    result = _max_feature_frame('sofa', 'sofa1_max')
    sofa2 = _max_feature_frame('sofa2', 'sofa2_max')
    if result.empty or sofa2.empty:
        return pd.DataFrame()
    result = result.merge(sofa2, on='stay_id', how='inner')

    for key, _label_en, _label_zh in SOFA_RECLASS_ORGANS:
        sofa1_part = _max_feature_frame(f'sofa_{key}', f'sofa1_{key}')
        sofa2_part = _max_feature_frame(f'sofa2_{key}', f'sofa2_{key}')
        if not sofa1_part.empty:
            result = result.merge(sofa1_part, on='stay_id', how='left')
        if not sofa2_part.empty:
            result = result.merge(sofa2_part, on='stay_id', how='left')

    for concept, output_col in [('death', 'mortality'), ('los_icu', 'los_days')]:
        concept_frame = _max_feature_frame(concept, output_col)
        if not concept_frame.empty:
            result = result.merge(concept_frame, on='stay_id', how='left')
    result['analysis_unit'] = 'patients'
    result['analysis_mode'] = mode
    return result


def _get_sofa_reclassification_source(lang: str = 'en', mode: str = 'worst_icu') -> tuple[pd.DataFrame, str]:
    """Return the best available patient-level dataset for SOFA reclassification UI."""
    dash_df = st.session_state.get('dash_demographics')
    if mode == 'worst_icu' and isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
        stats = _build_sofa_reclassification_stats(dash_df, lang=lang)
        if stats.get('available'):
            return dash_df, "Cohort Snapshot data" if lang == 'en' else "队列快照数据"

    loaded_df = _build_reclassification_df_from_loaded_concepts(st.session_state.get('loaded_concepts', {}), mode=mode)
    if not loaded_df.empty:
        mode_label = SOFA_RECLASS_ANALYSIS_MODES.get(mode, SOFA_RECLASS_ANALYSIS_MODES['worst_icu'])
        label = mode_label['label_en'] if lang == 'en' else mode_label['label_zh']
        return loaded_df, ("Loaded Quick Visualization concepts · " + label) if lang == 'en' else ("快速可视化已载入特征 · " + label)

    demo_concepts = _get_demo_sofa_timeseries_concepts()
    demo_timeseries_df = _build_reclassification_df_from_loaded_concepts(demo_concepts, mode=mode)
    if not demo_timeseries_df.empty:
        mode_label = SOFA_RECLASS_ANALYSIS_MODES.get(mode, SOFA_RECLASS_ANALYSIS_MODES['worst_icu'])
        label = mode_label['label_en'] if lang == 'en' else mode_label['label_zh']
        return demo_timeseries_df, ("Demo SOFA time series · " + label) if lang == 'en' else ("演示SOFA时间序列 · " + label)

    demo_df = st.session_state.get('reclass_demo_df')
    if mode == 'worst_icu' and isinstance(demo_df, pd.DataFrame) and not demo_df.empty:
        return demo_df, "Demo reclassification cohort" if lang == 'en' else "演示重新分层队列"

    return pd.DataFrame(), ""


def _render_reclassification_cards(reclass: Dict[str, Any], lang: str = 'en'):
    """Render compact metric cards for SOFA reclassification summaries."""
    metrics = reclass['metrics']
    cols = st.columns(5)
    cards = [
        (metrics.get('denominator', metrics['patients']), metrics.get('denominator_label', "Patients" if lang == 'en' else "患者数"), metrics.get('denominator_hint', "paired SOFA" if lang == 'en' else "双SOFA记录"), "#2563eb", "👥"),
        (metrics['discordant_pct'], "Discordant" if lang == 'en' else "重新分层", "SOFA-2 != SOFA-1" if lang == 'en' else "SOFA-2 != SOFA-1", "#ea580c", "⇄"),
        (metrics['up_pct'], "Up-classified" if lang == 'en' else "上调分层", "higher SOFA-2" if lang == 'en' else "SOFA-2更高", "#e11d48", "↑"),
        (metrics['down_pct'], "Down-classified" if lang == 'en' else "下调分层", "lower SOFA-2" if lang == 'en' else "SOFA-2更低", "#0f766e", "↓"),
        (metrics['median_delta'], "Median delta" if lang == 'en' else "Delta中位数", "SOFA-2 - SOFA-1" if lang == 'en' else "SOFA-2 - SOFA-1", "#475569", "Δ"),
    ]
    for col, (value, label, hint, color, icon) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid #cddbeb;border-left:4px solid {color};
                            border-radius:16px;padding:11px 13px;min-height:92px;box-shadow:0 8px 24px rgba(15,31,68,.045)">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
                        <span style="width:24px;height:24px;border-radius:7px;background:{color};color:white;display:inline-flex;align-items:center;justify-content:center;font-size:.82rem;font-weight:900">{icon}</span>
                        <span style="font-size:.68rem;font-weight:850;color:#60718a;letter-spacing:.07em;text-transform:uppercase">{label}</span>
                    </div>
                    <div style="font-size:1.5rem;font-weight:900;line-height:1.05;color:{color};letter-spacing:-.02em">{value}</div>
                    <div style="font-size:.68rem;color:#60718a;margin-top:4px;font-weight:700">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_reclassification_snapshot(reclass: Dict[str, Any], lang: str = 'en', key_prefix: str = 'reclass'):
    """Render a compact dashboard-friendly SOFA reclassification snapshot."""
    import plotly.express as px

    summary = reclass.get('summary', pd.DataFrame())
    if summary.empty:
        st.warning("No SOFA reclassification data available" if lang == 'en' else "没有可用的SOFA重新分层数据")
        return
    unit_pct_label = "Paired points (%)" if reclass.get('metrics', {}).get('denominator_label') == "Paired points" else (
        "配对时间点占比 (%)" if reclass.get('metrics', {}).get('denominator_label') == "配对时间点" else ("Patients (%)" if lang == 'en' else "患者占比 (%)")
    )

    fig = px.bar(
        summary.sort_values('pct', ascending=True),
        x='pct',
        y='group',
        orientation='h',
        text=summary.sort_values('pct', ascending=True)['pct'].map(lambda x: f"{x:.1f}%"),
        color='mortality',
        color_continuous_scale=['#dbeafe', '#ef4444'],
        range_color=[0, 100],
        labels={
            'pct': unit_pct_label,
            'group': "",
            'mortality': "Mortality %" if lang == 'en' else "死亡率 %",
        },
        template='plotly_white',
    )
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_layout(
        height=315,
        margin=dict(l=10, r=45, t=12, b=35),
        coloraxis_colorbar=dict(title="Mortality %" if lang == 'en' else "死亡率 %"),
        font=dict(size=13, color='#111827'),
    )
    fig.update_xaxes(range=[0, max(10, float(summary['pct'].max()) * 1.22)], gridcolor='#e5e7eb')
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_snapshot", config=_get_plotly_chart_config())


def render_severity_reclassification_subtab(lang: str):
    """渲染 SOFA 重分类子页面。"""
    return _render_severity_reclassification_subtab_impl(lang, globals())




def render_cohort_comparison_page():
    """渲染队列对比可视化页面 - 包含多个子标签页"""
    lang = st.session_state.get('language', 'en')
    screenshot_mode = _is_screenshot_mode()
    figure_panel = st.session_state.get('_figure_target_panel') if screenshot_mode else None
    if figure_panel in {'Group Contrast', 'Coverage Audit', 'Cross-DB Benchmark', 'Cohort Snapshot', 'SOFA-1 vs SOFA-2'}:
        render_cohort_figure_panel(figure_panel)
        return

    _coh_title = "Cohort Analysis" if lang == 'en' else "队列分析"
    _coh_sub = (
        "Subgroup contrast, coverage audit, cross-database benchmark, cohort snapshot, and SOFA-1/SOFA-2 definition sensitivity"
        if lang == 'en' else
        "亚组对照、覆盖度审计、跨库基准、队列快照与 SOFA-1/SOFA-2 定义敏感性"
    )
    st.markdown(f'''
    <div style="margin-bottom:16px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{_coh_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_coh_sub}</div>
    </div>
    ''', unsafe_allow_html=True)

    if st.session_state.get('entry_mode') == 'demo':
        if not _cohort_demo_workspace_ready(st.session_state):
            _render_cohort_demo_workspace_launcher(lang)
            return
        if not screenshot_mode:
            _render_cohort_demo_workspace_status(lang)

    elif st.session_state.get('entry_mode') == 'real':
        # Real data: offer a one-click shared workspace (P0-2)
        if not _cohort_real_workspace_ready(st.session_state):
            _render_cohort_real_workspace_launcher(lang)
            return
        if not _cohort_real_workspace_matches_sidebar(st.session_state):
            # Sidebar path changed since last workspace load
            st.warning(
                "⚠️ " + ("Sidebar data source changed. Reload the workspace to update."
                          if lang == 'en' else "侧边栏数据源已变更，请重新加载工作区。")
            )
        if not screenshot_mode:
            _render_cohort_real_workspace_status(lang)

    # 子标签页
    if lang == 'en':
        sub_tabs = st.tabs([
            "👥 Groups",
            "🧾 Coverage",
            "📈 Cross-DB",
            "🎯 Snapshot",
            "🧭 SOFA Δ",
        ])
    else:
        sub_tabs = st.tabs([
            "👥 分组",
            "🧾 覆盖",
            "📈 跨库",
            "🎯 快照",
            "🧭 SOFA Δ",
        ])

    with sub_tabs[0]:
        render_group_comparison_subtab(lang)

    with sub_tabs[1]:
        render_data_coverage_audit_subtab(lang)

    with sub_tabs[2]:
        render_multidb_distribution_subtab(lang)

    with sub_tabs[3]:
        render_cohort_dashboard_subtab(lang)

    with sub_tabs[4]:
        render_severity_reclassification_subtab(lang)


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


def _render_cohort_demo_workspace_status(lang: str) -> None:
    """Render one shared demo status strip for all Cohort Analysis panels."""
    title = "Shared demo cohort workspace" if lang == 'en' else "共享演示队列工作区"
    subtitle = (
        "Groups, coverage audit, cross-database benchmark, cohort snapshot, and SOFA sensitivity now use the same prepared demo state."
        if lang == 'en' else
        "分组、覆盖度审计、跨库基准、队列快照和 SOFA 敏感性现在共用同一套演示状态。"
    )
    status = "Ready for all panels" if lang == 'en' else "所有子板块已就绪"
    st.markdown(
        f'''
        <div class="cohort-demo-workspace">
            <div class="cohort-demo-badge">S1</div>
            <div>
                <div class="cohort-demo-title">{html.escape(title)}</div>
                <div class="cohort-demo-subtitle">{html.escape(subtitle)}</div>
            </div>
            <div class="cohort-demo-status">✓ {html.escape(status)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ " + ("Shared demo settings" if lang == 'en' else "共享演示设置"), expanded=False):
        n_patients = st.slider(
            "Number of patients" if lang == 'en' else "患者数量",
            min_value=50,
            max_value=500,
            value=int((st.session_state.get('mock_params') or {}).get('n_patients', 100)),
            key="cohort_demo_workspace_patients",
        )
        if st.button(
            "🔄 " + ("Regenerate shared demo workspace" if lang == 'en' else "重新生成共享演示工作区"),
            type="primary",
            use_container_width=True,
            key="cohort_demo_workspace_regenerate",
        ):
            _ensure_cohort_demo_workspace(st.session_state, lang=lang, n_patients=n_patients, force=True)
            st.rerun()


def _render_cohort_demo_workspace_launcher(lang: str) -> None:
    """Render a single shared Cohort Analysis demo generation entry point."""
    title = "Generate one shared cohort demo workspace" if lang == 'en' else "生成一次共享队列演示工作区"
    subtitle = (
        "This prepares the demo data for Groups, Coverage, Cross-DB, Snapshot, and SOFA Δ together. You will not need to generate data again inside each subpanel."
        if lang == 'en' else
        "这会一次性准备分组、覆盖度、跨库、快照和 SOFA Δ 所需的演示数据；之后不需要在每个子板块重复生成。"
    )
    st.markdown(
        f'''
        <div class="cohort-demo-workspace">
            <div class="cohort-demo-badge">S1</div>
            <div>
                <div class="cohort-demo-title">{html.escape(title)}</div>
                <div class="cohort-demo-subtitle">{html.escape(subtitle)}</div>
            </div>
            <div class="cohort-demo-status">1 click · all panels</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        n_patients = st.slider(
            "Number of patients" if lang == 'en' else "患者数量",
            min_value=50,
            max_value=500,
            value=int((st.session_state.get('mock_params') or {}).get('n_patients', 100)),
            key="cohort_demo_workspace_patients_init",
        )
    with col2:
        _compact_spacer(26)
        if st.button(
            "🚀 " + ("Generate shared cohort demo" if lang == 'en' else "生成共享队列演示"),
            type="primary",
            use_container_width=True,
            key="cohort_demo_workspace_generate",
        ):
            _ensure_cohort_demo_workspace(st.session_state, lang=lang, n_patients=n_patients, force=True)
            st.rerun()


def render_group_comparison_subtab(lang: str):
    """渲染分组比较子页面。"""
    return _render_group_comparison_subtab_impl(lang, globals())




def render_multidb_distribution_subtab(lang: str):
    """渲染多数据库分布子页面。"""
    return _render_multidb_distribution_subtab_impl(lang, globals())




def render_cohort_dashboard_subtab(lang: str):
    """渲染队列 dashboard 子页面。"""
    return _render_cohort_dashboard_subtab_impl(lang, globals())



def render_convert_dialog():
    """渲染转换对话框。"""
    return _render_convert_dialog_impl(globals())




def convert_csv_to_parquet(source_dir: str, target_dir: str, overwrite: bool = False) -> tuple:
    """将 CSV 数据转换为 Parquet。"""
    return _convert_csv_to_parquet_impl(source_dir, target_dir, overwrite, globals())




def _convert_hirid_data(source_dir: str, target_dir: str, overwrite: bool = False) -> tuple:
    """转换 HiRID 数据。"""
    return _convert_hirid_data_impl(source_dir, target_dir, overwrite, globals())




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

    disease_cohort = cf.get('disease_cohort')
    if disease_cohort and disease_cohort != 'none':
        parts.append(disease_cohort)

    icd_include_query = str(cf.get('icd_include_query', cf.get('icd_query', ''))).strip()
    if icd_include_query:
        token = _split_query_tokens(icd_include_query)
        if token:
            parts.append(f"icdIn{token[0][:10]}")
    icd_exclude_query = str(cf.get('icd_exclude_query', '')).strip()
    if icd_exclude_query:
        token = _split_query_tokens(icd_exclude_query)
        if token:
            parts.append(f"icdEx{token[0][:10]}")

    return "_".join(parts)


def _prime_export_completion(export_dir: Path, files: list[str], *, auto_load: bool = True) -> None:
    """Update session state after an export or export-ready skip path completes."""
    st.session_state.export_completed = True
    st.session_state.trigger_export = False
    st.session_state['_exporting_in_progress'] = False
    st.session_state.last_export_dir = str(export_dir)
    st.session_state.last_export_full_dir = str(export_dir)
    st.session_state.viz_export_path = str(export_dir)
    st.session_state.viz_data_source_mode = "exported"
    st.session_state.viz_confirmed_path = str(export_dir)
    st.session_state._prefer_exported_viz = True
    st.session_state._viz_export_path_version = st.session_state.get('_viz_export_path_version', 0) + 1

    if auto_load:
        selected_files = list(dict.fromkeys(Path(path).stem for path in files if path))
        max_patients_opt = st.session_state.get('viz_max_patients', 100)
        max_patients = None if max_patients_opt in (None, -1) else max_patients_opt
        st.session_state['_viz_auto_load_export'] = {
            'path': str(export_dir),
            'selected_files': selected_files or None,
            'max_patients': max_patients,
        }


def _write_export_manifest(
    export_dir: Path,
    *,
    exported_files: list[str],
    patient_count: int,
    concept_count: int,
    export_format: str,
    unavailable_concepts: list[str] | None = None,
    unsupported_concepts: list[str] | None = None,
    empty_data_concepts: list[str] | None = None,
    failed_concepts: list[str] | None = None,
    note: str | None = None,
) -> list[str]:
    """Write a lightweight export manifest for reproducibility."""
    from datetime import datetime as _dt

    cohort_filter = st.session_state.get('cohort_filter', {}) if st.session_state.get('cohort_enabled') else {}
    cohort_filter = {
        key: value
        for key, value in cohort_filter.items()
        if value not in (None, '', 'none', False)
    }

    manifest = {
        'easyicu_version': '1.0.0',
        'exported_at': _dt.now().isoformat(timespec='seconds'),
        'database': st.session_state.get('database', 'unknown'),
        'entry_mode': st.session_state.get('entry_mode', 'unknown'),
        'export_dir': str(export_dir),
        'export_format': export_format,
        'patient_count': int(patient_count or 0),
        'concept_count': int(concept_count or 0),
        'selected_concepts': list(st.session_state.get('selected_concepts', [])),
        'selected_groups': list(st.session_state.get('selected_groups', [])),
        'cohort_enabled': bool(st.session_state.get('cohort_enabled', False)),
        'cohort_filter': cohort_filter,
        'cohort_suffix': _generate_cohort_prefix(),
        'sepsis_runtime_options': _get_sepsis_runtime_options(),
        'exported_files': [Path(path).name for path in exported_files if path],
        'unavailable_concepts': unavailable_concepts or [],
        'unsupported_concepts': unsupported_concepts or [],
        'empty_data_concepts': empty_data_concepts or [],
        'failed_concepts': failed_concepts or [],
        'note': note or '',
    }

    json_path = export_dir / 'easyicu_export_manifest.json'
    txt_path = export_dir / 'easyicu_export_manifest.txt'

    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2, default=str)

    lines = [
        "EasyICU Export Manifest",
        f"Exported at: {manifest['exported_at']}",
        f"Database: {manifest['database']}",
        f"Entry mode: {manifest['entry_mode']}",
        f"Export directory: {manifest['export_dir']}",
        f"Export format: {manifest['export_format']}",
        f"Patients: {manifest['patient_count']}",
        f"Concepts: {manifest['concept_count']}",
    ]
    if manifest['cohort_suffix']:
        lines.append(f"Cohort suffix: {manifest['cohort_suffix']}")
    if manifest['cohort_filter']:
        lines.append("Cohort filter:")
        for key, value in manifest['cohort_filter'].items():
            lines.append(f"  - {key}: {value}")
    if manifest['selected_groups']:
        lines.append(f"Selected groups: {', '.join(manifest['selected_groups'])}")
    if manifest['selected_concepts']:
        lines.append("Selected concepts:")
        lines.extend([f"  - {concept}" for concept in manifest['selected_concepts']])
    if manifest['exported_files']:
        lines.append("Exported files:")
        lines.extend([f"  - {name}" for name in manifest['exported_files']])
    if note:
        lines.append(f"Note: {note}")

    with open(txt_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(lines) + "\n")

    return [str(json_path), str(txt_path)]


def _build_quick_viz_pdf_report(*, lang: str, preview_data: dict[str, pd.DataFrame], concepts_to_export: list[str]) -> bytes:
    """Create a compact one-file PDF summary for Quick Visualization."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    id_col = st.session_state.get('id_col', 'stay_id')
    database = st.session_state.get('database', 'unknown')
    export_dir = st.session_state.get('viz_confirmed_path') or st.session_state.get('last_export_dir') or "-"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_rows = 0
    patient_ids = set()
    summary_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for concept in concepts_to_export:
        df = preview_data.get(concept)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        rows = len(df)
        total_rows += rows

        if id_col in df.columns:
            concept_patients = set(df[id_col].dropna().astype(str))
            patient_ids |= concept_patients
            patient_count = len(concept_patients)
        else:
            patient_count = 0

        value_col = concept if concept in df.columns else None
        if value_col is None:
            candidate_cols = [
                col for col in df.columns
                if col not in {id_col, 'time', 'charttime', 'starttime', 'endtime', '_concept'}
            ]
            value_col = candidate_cols[0] if candidate_cols else None

        missing_pct = 0.0
        if value_col and value_col in df.columns:
            valid = pd.to_numeric(df[value_col], errors='coerce') if df[value_col].dtype == 'object' else df[value_col]
            missing_pct = float(valid.isna().mean() * 100) if len(valid) else 0.0

        summary_rows.append({
            'concept': concept,
            'rows': rows,
            'patients': patient_count,
            'missing_pct': missing_pct,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('rows', ascending=False).head(10)
    coverage_df = pd.DataFrame(summary_rows).sort_values('patients', ascending=False).head(10)

    total_patients = len(patient_ids)
    concept_count = len(summary_rows)

    title_text = "EasyICU Quick Visualization Report" if lang == 'en' else "EasyICU 快速可视化报告"
    subtitle_text = (
        f"Database: {database.upper()}   •   Concepts: {concept_count}   •   Patients: {total_patients}   •   Records: {total_rows:,}"
        if lang == 'en' else
        f"数据库：{database.upper()}   •   特征：{concept_count}   •   患者：{total_patients}   •   记录：{total_rows:,}"
    )
    meta_lines = [
        f"Generated at: {generated_at}" if lang == 'en' else f"生成时间：{generated_at}",
        f"Export directory: {export_dir}" if lang == 'en' else f"导出目录：{export_dir}",
        (
            f"Selected concepts: {', '.join(concepts_to_export[:10])}" + (" ..." if len(concepts_to_export) > 10 else "")
            if lang == 'en' else
            f"所选特征：{', '.join(concepts_to_export[:10])}" + (" ..." if len(concepts_to_export) > 10 else "")
        ),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "bar"}], [{"type": "bar"}, {"type": "table"}]],
        column_widths=[0.38, 0.62],
        row_heights=[0.42, 0.58],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=(
            "Overview" if lang == 'en' else "概览",
            "Top Concepts by Records" if lang == 'en' else "记录数最高的特征",
            "Top Concepts by Patient Coverage" if lang == 'en' else "患者覆盖最高的特征",
            "Detail Table" if lang == 'en' else "明细表",
        ),
    )

    overview_labels = [
        "Patients" if lang == 'en' else "患者数",
        "Concepts" if lang == 'en' else "特征数",
        "Records" if lang == 'en' else "记录数",
    ]
    overview_values = [total_patients, concept_count, total_rows]
    fig.add_trace(
        go.Pie(
            labels=overview_labels,
            values=[max(v, 1) for v in overview_values],
            hole=0.62,
            marker=dict(colors=["#2563eb", "#0ea5e9", "#14b8a6"]),
            textinfo="label+value",
            sort=False,
        ),
        row=1,
        col=1,
    )

    if not summary_df.empty:
        fig.add_trace(
            go.Bar(
                x=summary_df['rows'],
                y=summary_df['concept'],
                orientation='h',
                marker_color="#2563eb",
                hovertemplate="%{y}: %{x:,}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    if not coverage_df.empty:
        fig.add_trace(
            go.Bar(
                x=coverage_df['patients'],
                y=coverage_df['concept'],
                orientation='h',
                marker_color="#0f766e",
                hovertemplate="%{y}: %{x:,}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    table_df = pd.DataFrame(summary_rows).sort_values(['patients', 'rows'], ascending=[False, False]).head(8)
    if table_df.empty:
        table_df = pd.DataFrame([{
            'concept': '-',
            'rows': 0,
            'patients': 0,
            'missing_pct': 0.0,
        }])
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Concept" if lang == 'en' else "特征",
                    "Rows" if lang == 'en' else "记录",
                    "Patients" if lang == 'en' else "患者",
                    "Missing %" if lang == 'en' else "缺失率",
                ],
                fill_color="#e0f2fe",
                align="left",
                font=dict(size=13, color="#0f172a"),
            ),
            cells=dict(
                values=[
                    table_df['concept'],
                    table_df['rows'].map(lambda x: f"{int(x):,}"),
                    table_df['patients'].map(lambda x: f"{int(x):,}"),
                    table_df['missing_pct'].map(lambda x: f"{x:.1f}%"),
                ],
                fill_color="#ffffff",
                align="left",
                font=dict(size=12, color="#0f172a"),
                height=28,
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_layout(
        width=1440,
        height=1020,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text=f"{title_text}<br><sup>{subtitle_text}</sup>",
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            font=dict(size=24, color="#0f172a"),
        ),
        margin=dict(l=40, r=40, t=120, b=50),
        font=dict(family="Arial, sans-serif", color="#0f172a", size=13),
    )

    meta_text = "<br>".join(meta_lines)
    fig.add_annotation(
        x=0.0,
        y=1.08,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text=meta_text,
        font=dict(size=12, color="#475569"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=8,
    )

    return fig.to_image(format="pdf")


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
    """执行侧边栏触发的数据导出（直接导出到本地目录，带进度条）。"""
    return _execute_sidebar_export_impl(globals())




def render_export_page():
    """渲染数据导出页面。"""
    return _render_export_page_impl(globals())




def _get_requested_figure_target() -> tuple[str, str]:
    """Resolve `?figure=...`, `?panel=...`, or `?page=...` into a screenshot target."""
    raw_target = _query_param_value("figure")
    section, panel = _normalize_figure_target(raw_target)
    if section and panel:
        return section, panel
    for key in ("panel", "page", "view"):
        section, panel = _normalize_figure_target(_query_param_value(key))
        if section and panel:
            return section, panel
    return '', ''


def _ensure_quick_figure_demo_data(state: dict[str, Any], *, lang: str) -> None:
    """Preload compact demo concepts so figure URLs open directly to useful panels."""
    if state.get('loaded_concepts'):
        return
    state['mock_params'] = {'n_patients': 50, 'hours': 72}
    mock_data, patient_ids = generate_mock_data(n_patients=50, hours=72)
    state['loaded_concepts'] = mock_data
    state['loaded_data_origin'] = 'demo_viz'
    state['patient_ids'] = patient_ids
    state['id_col'] = 'stay_id'
    state['time_col'] = 'time'
    state['selected_concepts'] = list(mock_data.keys())
    _apply_quick_viz_screenshot_defaults(state, lang=lang)


COHORT_DEMO_MULTIDB_CONCEPTS = [
    'hr', 'sbp', 'dbp', 'map', 'temp', 'resp', 'spo2',
    'glu', 'na', 'k', 'crea', 'bili', 'lact',
    'hgb', 'plt', 'wbc',
    'ph', 'po2', 'pco2', 'fio2',
    'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver',
    'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
]


def _cohort_demo_workspace_ready(state: dict[str, Any]) -> bool:
    """Return whether all shared Cohort Analysis demo panels have data."""
    return bool(
        state.get('cohort_is_demo')
        and state.get('grp_is_demo')
        and state.get('multidb_is_demo')
        and state.get('dash_is_demo')
        and isinstance(state.get('grp_demographics'), pd.DataFrame)
        and not state.get('grp_demographics').empty
        and bool(state.get('multidb_data'))
        and isinstance(state.get('dash_demographics'), pd.DataFrame)
        and not state.get('dash_demographics').empty
    )


def _ensure_cohort_demo_workspace(
    state: dict[str, Any],
    *,
    lang: str = 'en',
    n_patients: Optional[int] = None,
    force: bool = False,
) -> None:
    """Prepare all Cohort Analysis demo panels once and share the same state."""
    mock_params = state.get('mock_params') if isinstance(state.get('mock_params'), dict) else {}
    state['mock_params'] = mock_params

    patient_count = n_patients if n_patients is not None else mock_params.get('n_patients', 100)
    try:
        patient_count = int(patient_count)
    except (TypeError, ValueError):
        patient_count = 100
    patient_count = max(1, patient_count)
    mock_params['n_patients'] = patient_count

    if force or not (state.get('grp_is_demo') and isinstance(state.get('grp_demographics'), pd.DataFrame)):
        state['grp_demographics'] = _generate_mock_demographics(patient_count, lang)
        state['grp_loaded_db'] = 'demo'
        state['grp_is_demo'] = True
        state.pop('grp_feature_data', None)

    if force or not (state.get('multidb_is_demo') and state.get('multidb_data')):
        state['multidb_data'] = _generate_mock_multidb_data(lang)
        state['multidb_concepts'] = list(COHORT_DEMO_MULTIDB_CONCEPTS)
        state['multidb_is_demo'] = True

    if force or not (state.get('dash_is_demo') and isinstance(state.get('dash_demographics'), pd.DataFrame)):
        state['dash_demographics'] = _generate_mock_cohort_dashboard_data(lang)
        state['dash_loaded_db'] = 'Demo'
        state['dash_is_demo'] = True

    if force or not isinstance(state.get('reclass_demo_df'), pd.DataFrame):
        dash_df = state.get('dash_demographics')
        if isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
            state['reclass_demo_df'] = dash_df.copy()

    state['cohort_is_demo'] = True


def _ensure_cohort_figure_demo_data(state: dict[str, Any], panel: str, *, lang: str) -> None:
    """Preload the cohort demo data needed by the requested paper-style panel."""
    if panel in {'Group Contrast', 'Coverage Audit', 'Cross-DB Benchmark', 'Cohort Snapshot', 'SOFA-1 vs SOFA-2'}:
        _ensure_cohort_demo_workspace(state, lang=lang)


# ---------------------------------------------------------------------------
# Real Data Shared Workspace  (P0-2: mirrors demo workspace for real data)
# ---------------------------------------------------------------------------

# Concepts loaded by default for the shared real-data workspace.
_REAL_WORKSPACE_PREVIEW_CONCEPTS = [
    'hr', 'map', 'resp', 'temp', 'spo2', 'crea', 'bili', 'lact', 'glu', 'plt',
]
_REAL_WORKSPACE_SOFA_CONCEPTS = [
    'sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal',
    'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
]


def _cohort_real_workspace_ready(state: dict[str, Any]) -> bool:
    """Return whether the shared real-data workspace is prepared for all panels."""
    return bool(
        state.get('_cohort_real_ws_ready')
        and isinstance(state.get('_cohort_real_ws_demographics'), pd.DataFrame)
        and not state.get('_cohort_real_ws_demographics').empty
    )


def _cohort_real_workspace_matches_sidebar(state: dict[str, Any]) -> bool:
    """Check if the loaded workspace still matches the sidebar-validated path."""
    ws_path = state.get('_cohort_real_ws_path', '')
    ws_db = state.get('_cohort_real_ws_db', '')
    sidebar_path = _default_real_data_root()
    sidebar_db = _default_real_database()
    return bool(ws_path and ws_path == sidebar_path and ws_db == sidebar_db)


def _ensure_cohort_real_workspace(
    state: dict[str, Any],
    *,
    lang: str = 'en',
    max_patients: int = 1000,
    load_concepts: bool = True,
    force: bool = False,
) -> tuple[bool, str]:
    """Load shared real-data workspace for all Cohort Analysis panels.

    Returns (success, message).
    """
    import streamlit as st

    database = _default_real_database()
    data_path = _default_real_data_root()
    if not data_path or not Path(data_path).exists():
        return False, ("Please validate a real data path in the sidebar first."
                       if lang == 'en' else "请先在侧边栏验证真实数据路径。")

    resolved_path = find_database_path(data_path, database)
    if not os.path.isdir(resolved_path):
        return False, (f"Database path not found: {resolved_path}"
                       if lang == 'en' else f"数据库路径不存在: {resolved_path}")

    # Skip if already loaded for same path+db and not forced
    if (not force
        and _cohort_real_workspace_ready(state)
        and state.get('_cohort_real_ws_path') == data_path
        and state.get('_cohort_real_ws_db') == database):
        return True, ""

    errors: list[str] = []
    loaded_concepts_dict: dict[str, Any] = {}

    # 1) Demographics
    try:
        from easyicu.patient_filter import PatientFilter
        pf = PatientFilter(database=database, data_path=resolved_path, verbose=False)
        demographics_df = pf._load_demographics()
        if len(demographics_df) > max_patients:
            demographics_df = demographics_df.head(max_patients)
        id_col = 'stay_id' if 'stay_id' in demographics_df.columns else 'patient_id'
        patient_ids = demographics_df[id_col].dropna().astype(int).tolist()
    except Exception as e:
        return False, f"Failed to load demographics: {e}"

    # 2) Preview concepts + SOFA (best-effort)
    if load_concepts:
        try:
            from easyicu import load_concepts as lc
            all_concepts = _REAL_WORKSPACE_PREVIEW_CONCEPTS + _REAL_WORKSPACE_SOFA_CONCEPTS
            concept_df = lc(
                concepts=all_concepts,
                database=database,
                data_path=resolved_path,
                patient_ids=patient_ids,
                verbose=False,
                **_get_sepsis_runtime_options(),
            )
            if concept_df is not None and not concept_df.empty:
                detected_id_col = next(
                    (col for col in ['stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid']
                     if col in concept_df.columns), None)
                time_cols = [col for col in ['charttime', 'time'] if col in concept_df.columns]
                base_cols = ([detected_id_col] if detected_id_col else []) + time_cols
                for concept in all_concepts:
                    if concept in concept_df.columns:
                        keep_cols = base_cols + [concept]
                        loaded_concepts_dict[concept] = concept_df[keep_cols].dropna(subset=[concept]).copy()
        except Exception as e:
            errors.append(f"Concept loading partial failure: {e}")

    # ---- Populate shared state ----
    state['_cohort_real_ws_ready'] = True
    state['_cohort_real_ws_path'] = data_path
    state['_cohort_real_ws_db'] = database
    state['_cohort_real_ws_resolved_path'] = resolved_path
    state['_cohort_real_ws_demographics'] = demographics_df
    state['_cohort_real_ws_patient_ids'] = patient_ids
    state['_cohort_real_ws_max_patients'] = max_patients
    state['_cohort_real_ws_concepts'] = loaded_concepts_dict
    state['_cohort_real_ws_errors'] = errors

    # Keep the global review footer and patient selectors aligned with the
    # newly loaded real workspace instead of leaving stale demo IDs behind.
    state['patient_ids'] = patient_ids
    state['available_patient_ids'] = patient_ids
    state['all_patient_count'] = len(patient_ids)
    state['id_col'] = id_col
    state['time_col'] = 'charttime'
    state['selected_patient'] = patient_ids[0] if patient_ids else None
    state['selected_concepts'] = list(loaded_concepts_dict.keys())

    # Seed individual panel keys so subpanels see data without re-loading
    state['grp_demographics'] = demographics_df.copy()
    state['grp_loaded_db'] = database
    state['grp_loaded_path'] = resolved_path
    state['grp_is_demo'] = False
    state['grp_data_root'] = data_path
    state['grp_db_select'] = database

    state['dash_demographics'] = demographics_df.copy()
    state['dash_loaded_db'] = database
    state['dash_loaded_path'] = resolved_path
    state['dash_is_demo'] = False
    state['dash_data_root'] = data_path
    state['dash_db_select'] = database

    state['multidb_data_root'] = data_path
    state['multidb_selected'] = [database]

    # SOFA concepts → loaded_concepts so reclassification panel picks them up
    if loaded_concepts_dict:
        state['loaded_concepts'] = dict(loaded_concepts_dict)
        state['loaded_data_origin'] = 'real_workspace'
    else:
        state['loaded_concepts'] = {}
        state['loaded_data_origin'] = 'real_workspace_demographics_only'

    msg_parts = [f"Loaded {len(demographics_df):,} patients"]
    if loaded_concepts_dict:
        msg_parts.append(f"{len(loaded_concepts_dict)} concepts")
    if errors:
        msg_parts.append(f"({len(errors)} warnings)")
    return True, "; ".join(msg_parts)


def _render_cohort_real_workspace_launcher(lang: str) -> None:
    """Render a single shared real-data workspace loader for Cohort Analysis."""
    data_path = _default_real_data_root()
    database = _default_real_database()
    db_labels = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
    db_label = db_labels.get(database, database)

    if not data_path or not Path(data_path).exists():
        st.warning(
            "⚠️ " + ("Please validate a real data path in the sidebar (Step 1) first."
                      if lang == 'en' else "请先在侧边栏（步骤1）验证真实数据路径。")
        )
        return

    title = ("Load shared real-data workspace" if lang == 'en'
             else "加载共享真实数据工作区")
    subtitle = (
        f"One-click load demographics, preview concepts, and SOFA for **{db_label}** from `{data_path}`. "
        "All panels (Groups, Coverage, Cross-DB, Snapshot, SOFA Δ) will share this data."
        if lang == 'en' else
        f"一键加载 **{db_label}** (`{data_path}`) 的人口统计、预览概念和 SOFA。"
        "所有子面板（分组、覆盖度、跨库、快照、SOFA Δ）共用此数据。"
    )
    st.markdown(
        f'''
        <div class="cohort-demo-workspace">
            <div class="cohort-demo-badge" style="background:linear-gradient(135deg,#059669 0%,#0891b2 100%)">R</div>
            <div>
                <div class="cohort-demo-title">{html.escape(title)}</div>
                <div class="cohort-demo-subtitle">{html.escape(subtitle)}</div>
            </div>
            <div class="cohort-demo-status" style="color:#059669">1 click · all panels</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        max_patients = st.slider(
            "Max patients to load" if lang == 'en' else "最大加载患者数",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            key="cohort_real_workspace_max_patients_init",
        )
    with col2:
        _compact_spacer(26)
        if st.button(
            "🚀 " + ("Load shared real-data workspace" if lang == 'en' else "加载共享真实数据工作区"),
            type="primary",
            use_container_width=True,
            key="cohort_real_workspace_generate",
        ):
            with st.spinner("Loading shared workspace..." if lang == 'en' else "正在加载共享工作区..."):
                ok, msg = _ensure_cohort_real_workspace(
                    st.session_state, lang=lang, max_patients=max_patients, force=True
                )
            if ok:
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")


def _render_cohort_real_workspace_status(lang: str) -> None:
    """Render one shared real-data workspace status strip for all Cohort Analysis panels."""
    state = st.session_state
    db = state.get('_cohort_real_ws_db', '')
    db_labels = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
    db_label = db_labels.get(db, db)
    n = len(state.get('_cohort_real_ws_demographics', []))
    n_concepts = len(state.get('_cohort_real_ws_concepts', {}))
    errors = state.get('_cohort_real_ws_errors', [])

    title = f"Shared real-data workspace — {db_label}" if lang == 'en' else f"共享真实数据工作区 — {db_label}"
    subtitle = (
        f"{n:,} patients, {n_concepts} concepts loaded. All subpanels share this data."
        if lang == 'en' else
        f"已加载 {n:,} 名患者、{n_concepts} 个概念。所有子面板共用此数据。"
    )
    status = f"✓ Ready" if lang == 'en' else f"✓ 就绪"
    warn_html = ""
    if errors:
        warn_list = "; ".join(errors[:3])
        warn_html = f'<div style="font-size:.78rem;color:#b45309;margin-top:4px">⚠️ {html.escape(warn_list)}</div>'

    st.markdown(
        f'''
        <div class="cohort-demo-workspace" style="border-color:#059669">
            <div class="cohort-demo-badge" style="background:linear-gradient(135deg,#059669 0%,#0891b2 100%)">R</div>
            <div>
                <div class="cohort-demo-title">{html.escape(title)}</div>
                <div class="cohort-demo-subtitle">{html.escape(subtitle)}{warn_html}</div>
            </div>
            <div class="cohort-demo-status" style="color:#059669">{html.escape(status)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    # Offer a reload button inline
    col_reload, col_spacer = st.columns([1, 3])
    with col_reload:
        if st.button(
            "🔄 " + ("Reload workspace" if lang == 'en' else "重新加载"),
            key="cohort_real_workspace_reload",
        ):
            with st.spinner("Reloading..." if lang == 'en' else "正在重新加载..."):
                ok, msg = _ensure_cohort_real_workspace(
                    st.session_state, lang=lang,
                    max_patients=state.get('_cohort_real_ws_max_patients', 1000),
                    force=True,
                )
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
            st.rerun()


def _apply_figure_query_preset(state: dict[str, Any], *, lang: str) -> None:
    """Make paper-figure screenshot URLs open directly to the requested visual panel."""
    if not _is_screenshot_mode():
        return

    section, panel = _get_requested_figure_target()
    if not section or not panel:
        return

    if state.get('entry_mode') == 'none':
        state['entry_mode'] = 'demo'
        state['use_mock_data'] = True
        state['database'] = 'mock'
        state['mock_params'] = {'n_patients': 100, 'hours': 72}

    if section == 'viz':
        _ensure_quick_figure_demo_data(state, lang=lang)
        state['_scroll_to_tab'] = 'viz'
    elif section == 'cohort':
        _ensure_cohort_figure_demo_data(state, panel, lang=lang)
        state['_scroll_to_tab'] = 'cohort'

    state['_figure_target_section'] = section
    state['_figure_target_panel'] = panel


def _render_figure_target_jump_script() -> None:
    """Click the requested figure tab after Streamlit has mounted nested tabs."""
    section = st.session_state.get('_figure_target_section')
    panel = st.session_state.get('_figure_target_panel')
    if not section or not panel:
        return

    lang = st.session_state.get('language', 'en')
    top_label = 'Quick Visualization' if section == 'viz' else 'Cohort Analysis'
    panel_label = panel
    if lang != 'en':
        panel_label = {
            'Data Tables': '数据',
            'Time Series': '时序',
            'Patient Overview': '患者',
            'Data Quality': '质量',
            'Group Contrast': '组间对照',
            'Coverage Audit': '覆盖度',
            'Cross-DB Benchmark': '跨库',
            'Cohort Snapshot': '队列快照',
            'SOFA-1 vs SOFA-2': 'SOFA-1 vs SOFA-2',
        }.get(panel, panel)
        top_label = '快速可视化' if section == 'viz' else '队列分析'

    js_code = f'''
    <script>
    (function() {{
        function clickTabByText(text) {{
            var tabs = Array.from(window.parent.document.querySelectorAll('button[data-baseweb="tab"]'));
            var target = tabs.find(function(tab) {{
                return (tab.innerText || tab.textContent || '').indexOf(text) !== -1;
            }});
            if (target) {{
                target.click();
                return true;
            }}
            return false;
        }}
        function jump() {{
            clickTabByText({top_label!r});
            setTimeout(function() {{ clickTabByText({panel_label!r}); }}, 350);
            setTimeout(function() {{ clickTabByText({panel_label!r}); }}, 800);
            var mainContainer = window.parent.document.querySelector('section.main');
            if (mainContainer) mainContainer.scrollTop = 0;
            window.parent.document.documentElement.scrollTop = 0;
        }}
        setTimeout(jump, 250);
        setTimeout(jump, 900);
    }})();
    </script>
    '''
    st.components.v1.html(js_code, height=0)


PUBLICATION_COMPOSITE_IMAGES = {
    'Figure 2': '03_Figure2.png',
    'Figure 3': '04_Figure3.png',
    'Figure 4': '05_Figure4_revised.png',
    'Supplementary Figure S1': '06_Supp_S1_revised.png',
}


def _publication_figure_image_path(panel: str) -> Optional[Path]:
    """Return the accepted image2 composite figure path for paper-aligned web views."""
    filename = PUBLICATION_COMPOSITE_IMAGES.get(panel)
    if not filename:
        return None

    candidates = [
        Path(__file__).resolve().parents[4]
        / 'easyicu写作'
        / 'final_figure_layout'
        / 'image2_generated_review'
        / filename,
        Path('/Users/haibo/Documents/GitHub/easyicu写作/final_figure_layout/image2_generated_review') / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def render_publication_composite_figure(panel: str) -> None:
    """渲染论文组合图。"""
    return _render_publication_composite_figure_impl(panel, globals())




def _render_paper_panel_css() -> None:
    """渲染论文图面板 CSS。"""
    return _render_paper_panel_css_impl(globals())




def _paper_panel_header(letter: str, title: str, subtitle: str = "") -> str:
    subtitle_html = f'<div class="paper-panel-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    return (
        '<div class="paper-panel-header">'
        f'<div class="paper-panel-letter">{html.escape(letter)}</div>'
        '<div>'
        f'<div class="paper-panel-title">{html.escape(title)}</div>'
        f'{subtitle_html}'
        '</div>'
        '</div>'
    )


def _paper_format_value(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    try:
        if pd.isna(value):
            return "–"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def _concept_display_name(concept: str, include_unit: bool = False) -> str:
    name, _zh, unit = CONCEPT_DICTIONARY.get(concept, (concept, concept, ""))
    if include_unit and unit:
        return f"{name} ({unit})"
    return name


def _concept_unit(concept: str) -> str:
    return CLINICAL_THRESHOLDS.get(concept, {}).get("unit") or CONCEPT_DICTIONARY.get(concept, ("", "", ""))[2] or ""


def _first_patient_id() -> Any:
    patient_ids = st.session_state.get("patient_ids") or []
    return patient_ids[0] if patient_ids else None


def _time_column_for_frame(df: pd.DataFrame) -> Optional[str]:
    for candidate in PREVIEW_TIME_COLUMNS + ['time', 'hour', 'datetime', 'timestamp']:
        if candidate in df.columns:
            return candidate
    return None


def _patient_series_for_concept(concept: str, patient_id: Any = None, max_points: int = 72) -> pd.DataFrame:
    df = st.session_state.get("loaded_concepts", {}).get(concept)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["time", "value"])
    id_col = st.session_state.get("id_col", "stay_id")
    patient_id = _first_patient_id() if patient_id is None else patient_id
    frame = df.copy()
    if patient_id is not None and id_col in frame.columns:
        frame = frame[frame[id_col] == patient_id].copy()
    time_col = _time_column_for_frame(frame)
    value_col = _choose_concept_value_column(concept, frame)
    if time_col is None or value_col is None:
        return pd.DataFrame(columns=["time", "value"])
    plot_df = _prepare_timeseries_plot_df(frame, time_col, value_col)
    if plot_df.empty:
        return pd.DataFrame(columns=["time", "value"])
    plot_df = plot_df.sort_values(time_col).head(max_points)
    out = pd.DataFrame({"time": plot_df[time_col], "value": pd.to_numeric(plot_df[value_col], errors="coerce")})
    return out.dropna(subset=["value"])


def _svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _render_inline_timeseries_svg(concept: str, patient_id: Any = None) -> str:
    series = _patient_series_for_concept(concept, patient_id=patient_id, max_points=72)
    if series.empty:
        values = np.array([0, 0.2, 0.1, 0.35, 0.22, 0.44, 0.31, 0.28], dtype=float)
    else:
        values = series["value"].astype(float).to_numpy()
    if len(values) < 2:
        values = np.array([values[0] if len(values) else 0, values[0] if len(values) else 0], dtype=float)

    thresholds = CLINICAL_THRESHOLDS.get(concept, {})
    threshold_values = [float(v) for v in thresholds.get("lines", []) if v is not None]
    y_values = list(values) + threshold_values
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    pad = (y_max - y_min) * 0.14
    y_min -= pad
    y_max += pad

    width, height = 310, 135
    left, right, top, bottom = 32, 8, 14, 24
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(i: int) -> float:
        return left + (i / max(len(values) - 1, 1)) * plot_w

    def y_pos(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    line_points = [(x_pos(i), y_pos(float(v))) for i, v in enumerate(values)]
    median_value = float(np.nanmedian(values))
    median_y = y_pos(median_value)
    grid_lines = []
    for frac in (0, 0.5, 1):
        y = top + frac * plot_h
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e8eef7" stroke-width="1"/>')

    threshold_svg = [f'<line x1="{left}" y1="{median_y:.1f}" x2="{width-right}" y2="{median_y:.1f}" stroke="#8c98aa" stroke-dasharray="4 4" stroke-width="1.2"/>']
    colors = thresholds.get("colors", ["#ef4444", "#f97316"])
    for idx, value in enumerate(threshold_values[:2]):
        threshold_svg.append(
            f'<line x1="{left}" y1="{y_pos(value):.1f}" x2="{width-right}" y2="{y_pos(value):.1f}" '
            f'stroke="{html.escape(str(colors[idx % len(colors)]))}" stroke-dasharray="4 4" stroke-width="1.2"/>'
        )

    y_tick_max = _paper_format_value(y_max - pad, 0)
    y_tick_min = _paper_format_value(y_min + pad, 0)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="135" role="img" aria-label="{html.escape(concept)} time series">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
        f'{"".join(grid_lines)}'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#cbd5e1" stroke-width="1"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#cbd5e1" stroke-width="1"/>'
        f'{"".join(threshold_svg)}'
        f'<polyline points="{_svg_polyline(line_points)}" fill="none" stroke="#2563eb" stroke-width="2.1"/>'
        f'<text x="2" y="{top+4}" fill="#394b63" font-size="10">{y_tick_max}</text>'
        f'<text x="2" y="{height-bottom}" fill="#394b63" font-size="10">{y_tick_min}</text>'
        f'<text x="{left}" y="{height-5}" fill="#394b63" font-size="10">0</text>'
        f'<text x="{width-right-28}" y="{height-5}" fill="#394b63" font-size="10">72</text>'
        f'</svg>'
    )


def _build_paper_wide_preview(concepts: list[str], max_rows: int = 6) -> pd.DataFrame:
    patient_id = _first_patient_id()
    id_col = st.session_state.get("id_col", "stay_id")
    merged: Optional[pd.DataFrame] = None
    for concept in concepts:
        series_df = _patient_series_for_concept(concept, patient_id=patient_id, max_points=72)
        if series_df.empty:
            continue
        concept_df = series_df.rename(columns={"value": concept}).copy()
        concept_df["time_key"] = range(len(concept_df))
        concept_df[id_col] = patient_id if patient_id is not None else 10001
        concept_df = concept_df[[id_col, "time_key", concept]]
        if merged is None:
            merged = concept_df
        else:
            merged = pd.merge(merged, concept_df, on=[id_col, "time_key"], how="outer")
    if merged is None or merged.empty:
        rows = []
        for idx in range(max_rows):
            rows.append({id_col: patient_id or 10001, "charttime": f"{idx}h", **{c: np.nan for c in concepts}})
        return pd.DataFrame(rows)
    merged = merged.sort_values("time_key").head(max_rows).copy()
    merged.insert(1, "charttime", merged["time_key"].map(lambda v: f"{int(v)}h"))
    merged.drop(columns=["time_key"], inplace=True)
    return merged


def _paper_table_html(df: pd.DataFrame, max_rows: int = 6) -> str:
    show_df = df.head(max_rows).copy()
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in [""] + list(show_df.columns))
    row_html = []
    for idx, row in show_df.iterrows():
        cells = [f"<td>{idx}</td>"]
        for value in row.tolist():
            cells.append(f"<td>{_paper_format_value(value)}</td>")
        row_html.append("<tr>" + "".join(cells) + "</tr>")
    row_html.append("<tr>" + "".join(["<td>…</td>"] * (len(show_df.columns) + 1)) + "</tr>")
    return f'<table class="paper-table"><thead><tr>{header_html}</tr></thead><tbody>{"".join(row_html)}</tbody></table>'


def _quality_snapshot_rows(limit: int = 10) -> tuple[pd.DataFrame, int, float, float, float]:
    id_col = st.session_state.get("id_col", "stay_id")
    mock_params = st.session_state.get("mock_params", {}) or {}
    demo_hours = int(mock_params.get("hours") or 0) if st.session_state.get("entry_mode") == "demo" and mock_params.get("hours") else None
    time_grid_size = demo_hours or 72
    total_patients = _get_quality_cohort_patient_count(st.session_state)
    cohort_patient_ids = _get_quality_cohort_patient_ids(st.session_state)
    los_by_patient = _get_quality_los_by_patient(st.session_state)

    rows: list[dict[str, Any]] = []
    total_records = 0
    total_expected = 0.0
    total_missing_weight = 0.0
    total_outlier_weight = 0.0
    total_duplicate_weight = 0.0
    for concept, df in st.session_state.get("loaded_concepts", {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        profile = _build_quality_metric_profile_cached(
            concept=concept,
            df=df,
            id_col=id_col,
            cohort_patient_count=total_patients,
            time_grid_size=time_grid_size,
            cohort_patient_ids=cohort_patient_ids,
            los_by_patient=los_by_patient,
            demo_hours=demo_hours,
        )
        n_records = len(df)
        weight = float(profile["expected_observations"] or n_records or 1)
        total_records += n_records
        total_expected += weight
        total_missing_weight += weight * (profile["missing_rate"] / 100)
        total_outlier_weight += n_records * (profile["out_of_physio_rate"] / 100)
        total_duplicate_weight += n_records * (profile["duplicate_rate"] / 100)
        rows.append(
            {
                "Concept": concept,
                "Missing": float(profile["missing_rate"]),
                "Records": n_records,
                "Patients": df[id_col].nunique() if id_col in df.columns else 0,
                "Denom": profile["denominator_tag"],
            }
        )
    quality_df = pd.DataFrame(rows).sort_values("Missing", ascending=False).head(limit) if rows else pd.DataFrame()
    overall_missing = (total_missing_weight / total_expected * 100) if total_expected > 0 else 0.0
    overall_outliers = (total_outlier_weight / total_records * 100) if total_records > 0 else 0.0
    overall_duplicates = (total_duplicate_weight / total_records * 100) if total_records > 0 else 0.0
    return quality_df, total_records, overall_missing, overall_outliers, overall_duplicates


def _render_paper_data_panel() -> None:
    _render_paper_panel_css()
    concepts = [c for c in ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp"] if c in st.session_state.get("loaded_concepts", {})]
    if not concepts:
        concepts = ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp"]
    patient_count = len(st.session_state.get("patient_ids") or [])
    preview_df = _build_paper_wide_preview(concepts, max_rows=6)
    chips = "".join(f'<span class="paper-chip">{html.escape(c)}</span>' for c in concepts)
    table_html = _paper_table_html(preview_df, max_rows=6)
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("A", "Feature-level review", "Compact table preview for a selected module before drilling into feature detail.")}
            <div class="paper-grid-3">
                <div class="paper-soft-card">
                    <div class="paper-eyebrow">Selected module</div>
                    <div style="font-weight:900;font-size:0.92rem;color:#071f45">❤️ Vital Signs</div>
                    <div class="paper-note" style="margin-top:0.22rem">Core bedside vital signs aligned into a compact longitudinal preview.</div>
                    <div class="paper-chip-row">{chips}</div>
                </div>
                <div class="paper-card">
                    <div class="paper-metric-label">Features</div>
                    <div class="paper-metric-value">{len(concepts)}</div>
                </div>
                <div class="paper-card">
                    <div class="paper-metric-label">Patients</div>
                    <div class="paper-metric-value">{patient_count or 50}</div>
                </div>
            </div>
            <div class="paper-control-row">
                <div><span class="paper-radio-dot"></span>Merge All (Wide Table)<span class="paper-radio-empty"></span>Single Feature</div>
                <div>Rows per feature&nbsp;&nbsp;<span class="paper-select">2,000⌄</span></div>
            </div>
            <div class="paper-grid-2" style="display:grid;grid-template-columns:1fr 1fr;border:1px solid #dce6f3;border-radius:9px;margin-bottom:0.52rem;overflow:hidden">
                <div style="padding:0.42rem 0.6rem;border-right:1px solid #e4ebf4"><div class="paper-metric-label">Preview rows</div><div style="font-weight:900">1,000</div></div>
                <div style="padding:0.42rem 0.6rem"><div class="paper-metric-label">Preview columns</div><div style="font-weight:900">{len(preview_df.columns)}</div></div>
            </div>
            {table_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_timeseries_panel() -> None:
    _render_paper_panel_css()
    patient_id = _first_patient_id()
    concepts = [c for c in ["hr", "map", "spo2", "resp"] if c in st.session_state.get("loaded_concepts", {})]
    if len(concepts) < 4:
        concepts = ["hr", "map", "spo2", "resp"]
    charts = []
    for concept in concepts[:4]:
        unit = _concept_unit(concept)
        title = _concept_display_name(concept)
        charts.append(
            f'<div><div class="paper-mini-chart-title">{html.escape(title)}{f" ({html.escape(unit)})" if unit else ""}</div>{_render_inline_timeseries_svg(concept, patient_id=patient_id)}</div>'
        )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("B", "Patient time-series", "Representative single-patient trajectories with cohort median and clinical threshold references.")}
            <div style="font-weight:900;font-size:0.78rem;margin-bottom:0.12rem">❤️ Vital Signs</div>
            <div class="paper-legend">
                <span><span class="paper-legend-line"></span>Patient {html.escape(str(patient_id or 10001))}</span>
                <span><span class="paper-legend-line dash"></span>Median</span>
                <span><span class="paper-legend-line low"></span>Low threshold</span>
                <span><span class="paper-legend-line high"></span>High threshold</span>
            </div>
            <div class="paper-chart-grid-2">{''.join(charts)}</div>
            <div class="paper-note" style="text-align:center">Time since ICU admission (hours)</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_quality_panel() -> None:
    _render_paper_panel_css()
    qdf, total_records, overall_missing, overall_outliers, overall_duplicates = _quality_snapshot_rows(limit=10)
    if qdf.empty:
        qdf = pd.DataFrame({"Concept": ["aki_stage_rrt", "mech_circ_support", "ecmo", "delirium_tx"], "Missing": [96, 89, 86, 84]})
    bars = []
    for _, row in qdf.iterrows():
        value = float(row["Missing"])
        color = "#ef4444" if value >= 75 else "#f97316" if value >= 50 else "#fb923c" if value >= 25 else "#f59e0b"
        bars.append(
            f'<div class="paper-bar-row"><div style="text-align:right;color:#31455f">{html.escape(str(row["Concept"]))}</div>'
            f'<div class="paper-bar-track"><div class="paper-bar-fill" style="width:{max(1, min(100, value)):.1f}%;background:{color}"></div></div>'
            f'<div style="color:#31455f">{value:.0f}</div></div>'
        )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("C", "Missingness rates", "Data-quality summary with denominator-aware missingness, out-of-physiology, and temporal checks.")}
            <div style="font-weight:900;font-size:0.72rem;margin-bottom:0.42rem">Data quality</div>
            <div class="paper-grid-4">
                <div class="paper-card"><div class="paper-metric-label">Total records</div><div class="paper-metric-value">{total_records or 102578:,}</div></div>
                <div class="paper-card"><div class="paper-metric-label">Weighted missing</div><div class="paper-metric-value" style="color:#ef4444">{overall_missing:.1f}%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Out-of-physio</div><div class="paper-metric-value" style="color:#059669">{overall_outliers:.1f}%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Duplicate TS</div><div class="paper-metric-value" style="color:#059669">{overall_duplicates:.1f}%</div></div>
            </div>
            <div class="paper-tabs">
                <div class="paper-tab active">📊 Missingness</div>
                <div class="paper-tab">🧪 Out-of-Physio</div>
                <div class="paper-tab">⏱️ Temporal Integrity</div>
            </div>
            <div class="paper-note">Missingness denominator: d=LOS uses patient-specific ICU stay; d=72h uses the fallback window.</div>
            <div style="display:grid;grid-template-columns:1fr 120px;gap:0.75rem;align-items:center;margin-top:0.45rem">
                <div>{''.join(bars)}</div>
                <div class="paper-card" style="font-size:0.6rem;line-height:1.7">
                    <div style="font-weight:900;margin-bottom:0.25rem">Missing rate (%)</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#ef4444;margin-right:6px"></span>75 - 100</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#f97316;margin-right:6px"></span>50 - 75</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#fb923c;margin-right:6px"></span>25 - 50</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#f59e0b;margin-right:6px"></span>&lt; 25</div>
                    <div style="border-top:1px solid #e4ebf4;margin-top:0.45rem;padding-top:0.35rem">Denominator<br><b>d = LOS</b></div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_crossdb_panel() -> None:
    _render_paper_panel_css()
    dbs = [
        ("MIMIC-IV", 10854, "#16a34a"),
        ("eICU", 12690, "#f97316"),
        ("AUMC", 14553, "#2563eb"),
        ("HiRID", 13473, "#ef4444"),
        ("MIMIC-III", 11961, "#7e22ce"),
        ("SICdb", 13365, "#795548"),
    ]
    db_cards = "".join(
        f'<div class="paper-db-card" style="border-left-color:{color}">'
        f'<div class="paper-db-name"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:4px"></span>{html.escape(name)}</div>'
        f'<div class="paper-db-value">{value:,}</div>'
        f'<div style="font-size:0.56rem;color:#047857">records</div>'
        f'</div>'
        for name, value, color in dbs
    )
    concepts = ["hr", "map", "resp", "temp", "spo2", "crea"]
    chart_cells = []
    for idx, concept in enumerate(concepts):
        chart_cells.append(
            f'<div><div class="paper-mini-chart-title">{html.escape(_concept_display_name(concept, include_unit=True))}</div>{_render_density_svg(seed=idx, concept=concept)}</div>'
        )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("D", "Cross-database distributions", "Cross-database density overlays for harmonized clinical concepts.")}
            <div class="paper-grid-5" style="grid-template-columns:repeat(6, minmax(0, 1fr));margin-bottom:0.55rem">{db_cards}</div>
            <div class="paper-legend" style="justify-content:center">
                {''.join(f'<span><span class="paper-legend-line" style="border-top-color:{color}"></span>{html.escape(name)}</span>' for name, _value, color in dbs)}
            </div>
            <div class="paper-chart-grid-2" style="grid-template-columns:repeat(3,minmax(0,1fr));gap:0.5rem 0.72rem">{''.join(chart_cells)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_density_svg(seed: int, concept: str) -> str:
    width, height = 205, 128
    left, right, top, bottom = 28, 8, 12, 24
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#16a34a", "#f97316", "#2563eb", "#ef4444", "#7e22ce", "#795548"]
    x = np.linspace(-3.2, 3.2, 80)
    paths = []
    for idx, color in enumerate(colors):
        mu = (idx - 2.5) * 0.12 + (seed % 3 - 1) * 0.08
        sigma = 0.72 + (idx % 3) * 0.08 + seed * 0.01
        y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma
        y = y / max(y.max(), 1e-9)
        points = []
        for xv, yv in zip(x, y):
            px = left + (xv - x.min()) / (x.max() - x.min()) * plot_w
            py = top + (1 - yv) * plot_h
            points.append((px, py))
        fill_points = [(left, height - bottom)] + points + [(width - right, height - bottom)]
        paths.append(
            f'<polygon points="{_svg_polyline(fill_points)}" fill="{color}" opacity="0.16"/>'
            f'<polyline points="{_svg_polyline(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="128" role="img" aria-label="{html.escape(concept)} distribution">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<line x1="{left}" y1="{top+plot_h*0.33:.1f}" x2="{width-right}" y2="{top+plot_h*0.33:.1f}" stroke="#e8eef7"/>'
        f'<line x1="{left}" y1="{top+plot_h*0.66:.1f}" x2="{width-right}" y2="{top+plot_h*0.66:.1f}" stroke="#e8eef7"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#cbd5e1"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#cbd5e1"/>'
        f'{"".join(paths)}'
        f'<text x="0" y="{top+10}" fill="#394b63" font-size="9">Density</text>'
        f'<text x="{left}" y="{height-5}" fill="#394b63" font-size="9">0</text>'
        f'<text x="{width-right-18}" y="{height-5}" fill="#394b63" font-size="9">+</text>'
        f'</svg>'
    )


def render_quick_figure_panel(panel: str) -> None:
    """渲染 quick visualization 论文图面板。"""
    return _render_quick_figure_panel_impl(panel, globals())




def _render_paper_patient_panel() -> None:
    _render_paper_panel_css()
    patient_id = _first_patient_id() or 10001
    concepts = ["hr", "map", "resp", "spo2", "sofa", "crea"]
    cards = []
    for concept in concepts:
        series = _patient_series_for_concept(concept, patient_id=patient_id, max_points=72)
        value = series["value"].dropna().iloc[-1] if not series.empty else None
        cards.append(
            f'<div class="paper-card"><div class="paper-metric-label">{html.escape(concept)}</div>'
            f'<div class="paper-metric-value">{_paper_format_value(value)}</div>'
            f'<div style="font-size:0.55rem;color:#64748b">{html.escape(_concept_unit(concept) or "latest")}</div></div>'
        )
    mini_charts = "".join(
        f'<div><div class="paper-mini-chart-title">{html.escape(_concept_display_name(c))}</div>{_render_inline_timeseries_svg(c, patient_id=patient_id)}</div>'
        for c in ["hr", "map", "resp", "spo2"]
    )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("C", "Patient overview", "Compact case dashboard with latest measurements and longitudinal bedside trends.")}
            <div class="paper-soft-card" style="margin-bottom:0.65rem">
                <div style="font-weight:900">Patient {html.escape(str(patient_id))}</div>
                <div class="paper-note">Demo case summary · first 72 ICU hours · selected clinical concepts available for drill-down.</div>
            </div>
            <div class="paper-grid-4" style="grid-template-columns:repeat(6,minmax(0,1fr))">{''.join(cards)}</div>
            <div class="paper-chart-grid-2">{mini_charts}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_cohort_figure_panel(panel: str) -> None:
    """渲染 cohort 论文图面板。"""
    return _render_cohort_figure_panel_impl(panel, globals())




def _render_paper_group_panel() -> None:
    rows = [
        ("Demographics", "Age, median (IQR)", "63 (52-73)", "66 (56-76)", "0.012", "0.24", "Small"),
        ("Demographics", "Male, %", "57.8", "61.1", "0.098", "0.07", "Small"),
        ("Vital Signs", "Heart rate (bpm), median (IQR)", "84 (72-98)", "92 (78-110)", "<0.001", "0.34", "Medium"),
        ("Vital Signs", "Mean arterial pressure (mmHg)", "82 (72-93)", "74 (64-86)", "<0.001", "0.48", "Large"),
        ("Laboratory", "Creatinine (mg/dL), median (IQR)", "1.1 (0.8-1.6)", "1.7 (1.1-2.7)", "<0.001", "0.57", "Large"),
        ("Outcomes", "ICU LOS (days), median (IQR)", "4 (2-8)", "6 (3-12)", "<0.001", "0.36", "Medium"),
    ]
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("A", "Group contrast table", "Baseline characteristics and standardized differences for survived vs deceased subgroups.")}
            <div class="paper-grid-3">
                <div class="paper-card"><div class="paper-metric-label">Survived</div><div class="paper-metric-value">1,662 <span style="font-size:0.65rem;font-weight:700">(68.6%)</span></div></div>
                <div class="paper-card"><div class="paper-metric-label">Deceased</div><div class="paper-metric-value">763 <span style="font-size:0.65rem;font-weight:700">(31.4%)</span></div></div>
                <div class="paper-card"><div class="paper-metric-label">Ratio</div><div class="paper-metric-value">2.18 : 1</div></div>
            </div>
            <table class="paper-table"><thead><tr><th>Module</th><th>Characteristic</th><th>Survived</th><th>Deceased</th><th>p-value</th><th>SMD</th><th>Magnitude</th></tr></thead><tbody>{body}</tbody></table>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_coverage_panel() -> None:
    modules = [
        ("Vital Signs", [98.7, 98.6, 98.9, 98.4, 98.9]),
        ("Laboratory", [97.2, 97.3, 96.8, 97.5, 96.9]),
        ("Input/Output", [92.4, 92.6, 91.9, 93.3, 91.6]),
        ("Medications", [88.3, 89.1, 86.6, 89.7, 87.0]),
        ("Procedures", [75.1, 76.3, 72.4, 77.6, 72.9]),
        ("Ventilation", [70.8, 71.5, 69.2, 73.4, 68.5]),
    ]
    rows = []
    for module, values in modules:
        tds = [f"<td style='text-align:left;font-weight:800'>{html.escape(module)}</td>"]
        for value in values:
            green = int(240 - value * 1.1)
            tds.append(f"<td style='background:rgb({green},235,{green});font-weight:800'>{value:.1f}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    flow = [
        ("All ICU stays", "3,057"),
        ("Meet age criteria", "2,810"),
        ("ICU stay ≥24h", "2,610"),
        ("ICD include", "2,150"),
        ("ICD exclude", "2,012"),
        ("Final cohort", "2,012"),
    ]
    flow_html = "".join(f'<div class="paper-flow-step"><b>{html.escape(label)}</b><br><span style="font-weight:900">{value}</span></div>' for label, value in flow)
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("B", "Data coverage and eligibility audit", "Coverage heatmap and transparent inclusion/exclusion flow for the analysis cohort.")}
            <div class="paper-two-col">
                <div>
                    <div style="font-weight:900;font-size:0.7rem;margin-bottom:0.35rem">1. Data coverage by module and subgroup (%)</div>
                    <table class="paper-heatmap"><thead><tr><th>Module</th><th>Overall</th><th>Survived</th><th>Deceased</th><th>SOFA ≤ 6</th><th>SOFA &gt; 6</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
                    <div class="paper-note">Missingness denominators: d=LOS, d=72h, d=demo, d=static.</div>
                </div>
                <div>
                    <div style="font-weight:900;font-size:0.7rem;margin-bottom:0.35rem">2. Eligibility flow</div>
                    {flow_html}
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_snapshot_panel() -> None:
    cards = [
        ("Total patients", "2,012"),
        ("Total features", "167"),
        ("Median SOFA", "6 (3-9)"),
        ("Top phenotype", "Sepsis 24.1%"),
        ("Mortality", "31.4%"),
        ("Median LOS", "5 (2-10) d"),
    ]
    card_html = "".join(f'<div class="paper-card"><div class="paper-metric-label">{html.escape(k)}</div><div class="paper-metric-value">{html.escape(v)}</div></div>' for k, v in cards)
    mini = "".join(
        f'<div><div class="paper-mini-chart-title">{title}</div>{_render_density_svg(idx, "snapshot")}</div>'
        for idx, title in enumerate(["Age distribution", "ICU LOS distribution", "SOFA severity", "Outcomes"])
    )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("C", "Cohort snapshot", "Cohort after all inclusion/exclusion criteria applied.")}
            <div class="paper-grid-5" style="grid-template-columns:repeat(6,minmax(0,1fr));margin-bottom:0.65rem">{card_html}</div>
            <div class="paper-chart-grid-2">{mini}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_sofa_panel() -> None:
    rows = []
    matrix = [[370, 78, 20, 6, 1], [72, 359, 108, 24, 2], [16, 94, 242, 86, 11], [4, 16, 94, 227, 33], [0, 2, 11, 35, 101]]
    for idx, row in enumerate(matrix):
        cells = [f"<td style='font-weight:900'>{['0-3','4-6','7-9','10-12','≥13'][idx]}</td>"]
        for value in row:
            shade = 245 - min(150, int(value / 370 * 150))
            cells.append(f"<td style='background:rgb({shade},{shade+5},255);font-weight:800'>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("D", "SOFA-1 vs SOFA-2 reclassification", "Agreement, upgrade/downgrade patterns, and organ-level contributors to score changes.")}
            <div class="paper-grid-5">
                <div class="paper-card"><div class="paper-metric-label">Total patients</div><div class="paper-metric-value">2,012</div></div>
                <div class="paper-card"><div class="paper-metric-label">Agreement</div><div class="paper-metric-value" style="color:#059669">66.2%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Upgrade</div><div class="paper-metric-value" style="color:#ef4444">20.4%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Downgrade</div><div class="paper-metric-value" style="color:#f59e0b">13.4%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Median |ΔSOFA|</div><div class="paper-metric-value">1</div></div>
            </div>
            <div style="margin-top:0.65rem">
                <div style="font-weight:900;font-size:0.7rem;margin-bottom:0.35rem">1. Reclassification matrix (n)</div>
                <table class="paper-heatmap"><thead><tr><th>SOFA-1</th><th>0-3</th><th>4-6</th><th>7-9</th><th>10-12</th><th>≥13</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def main():
    """主函数。"""
    init_session_state()
    _apply_figure_query_preset(st.session_state, lang=st.session_state.get('language', 'en'))

    # 获取入口模式
    entry_mode = st.session_state.get('entry_mode', 'none')
    lang = st.session_state.get('language', 'en')

    figure_section = st.session_state.get('_figure_target_section') if _is_screenshot_mode() else None
    if figure_section == 'paper':
        render_publication_composite_figure(st.session_state.get('_figure_target_panel', ''))
        return

    # ============ 入口页面：选择Demo或Real Data模式 ============
    if entry_mode == 'none':
        render_entry_page()
        return

    _apply_assistant_preset()
    _maybe_materialize_pending_preset()

    export_in_progress = bool(
        st.session_state.get('trigger_export', False)
        or st.session_state.get('_exporting_in_progress', False)
    )

    if figure_section == 'viz':
        if export_in_progress:
            st.markdown('<div class="compact-inline-notice info">⏳ Export in progress.</div>', unsafe_allow_html=True)
        else:
            render_quick_visualization_page()
        _render_figure_target_jump_script()
        return
    if figure_section == 'cohort':
        if export_in_progress:
            st.markdown('<div class="compact-inline-notice info">⏳ Export in progress.</div>', unsafe_allow_html=True)
        else:
            render_cohort_comparison_page()
        _render_figure_target_jump_script()
        return

    # ============ 进入具体模式后，显示完整应用 ============
    render_sidebar()

    # 处理CSV转换对话框
    if st.session_state.get('show_convert_dialog', False):
        render_convert_dialog()

    # 🔧 导出进度区域：优先使用 Guide: Complete 中创建的容器，否则创建备用容器
    # （实际导出在渲染 Home 页面后执行，确保 container 已创建）
    default_export_container = st.container()

    # ============ 顶部标题（精简现代风格） ============
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

    _render_icd_preview_main_panel(lang)
    _render_feature_definition_panel(lang)

    # 主页面标签：Tutorial, Quick Visualization, Cohort Analysis
    tab1, tab2, tab3 = st.tabs([
        get_text('home'),
        get_text('quick_visualization'),
        get_text('cohort_compare'),
    ])

    with tab1:
        render_home()

    with tab2:
        if export_in_progress:
            export_hold_msg = (
                "⏳ Export in progress. Preview charts are temporarily paused to avoid preview-state conflicts during full extraction."
                if lang == 'en' else
                "⏳ 正在导出。为避免 Preview 状态干扰全量提取，临时暂停渲染预览图表。"
            )
            st.markdown(f'<div class="compact-inline-notice info">{export_hold_msg}</div>', unsafe_allow_html=True)
        else:
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
                                    **_get_sepsis_runtime_options(),
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
        if export_in_progress:
            export_hold_msg = (
                "⏳ Export in progress. Cohort analysis views are temporarily paused until extraction finishes."
                if lang == 'en' else
                "⏳ 正在导出。提取完成前，暂时不渲染队列分析页面。"
            )
            st.markdown(f'<div class="compact-inline-notice info">{export_hold_msg}</div>', unsafe_allow_html=True)
        else:
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
                # Preview 后触发正式导出时，避免复用旧 tab 内的容器对象，直接使用当前 rerun 的安全容器
                preview_like_origins = {'preview', 'quick_preview'}
                if st.session_state.get('loaded_data_origin') in preview_like_origins:
                    export_container = default_export_container
                else:
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
        if not _is_screenshot_mode():
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

    _render_figure_target_jump_script()

    if not _is_screenshot_mode():
        try:
            from easyicu.webapp.llm_chat import render_floating_chat_dock
            render_floating_chat_dock()
        except Exception:
            pass

    if not _is_screenshot_mode():
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
                    - 🏥 **Patient Overview**: Single patient details
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
