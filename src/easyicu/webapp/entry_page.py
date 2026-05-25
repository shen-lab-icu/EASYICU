"""Entry page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.components.constants import get_all_concepts
from easyicu.webapp.session_state import clear_run_state


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_entry_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_entry_page(app_context: dict[str, Any] | None = None):
    """渲染入口选择页面 - Premium Hero 设计"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')
    if 'entry_lang_select' not in st.session_state:
        st.session_state.entry_lang_select = 'EN' if lang == 'en' else 'ZH'

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
    _hero_subtitle = (
        "Local ICU Research Workflow · Extract · Review · Analyze · Draft"
        if lang == 'en' else
        "本地 ICU 科研工作流 · 提取 · 复核 · 分析 · 成文"
    )
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
    /* Demo card · accent-soft surface, recommended guided tour
       (matches easyicu design/page-entry.jsx ModeCardLarge tone="demo"). */
    .entry-btn-wrap.demo-col div[data-testid="stButton"] > button {
        background: var(--accent-soft) !important;
        color: var(--ink) !important;
        border: 1px solid var(--accent-border) !important;
        box-shadow: var(--sh-1) !important;
    }
    .entry-btn-wrap.demo-col div[data-testid="stButton"] > button:hover {
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
        box-shadow: var(--sh-2) !important;
    }
    /* Real card · neutral surface, prepared-data path
       (matches easyicu design/page-entry.jsx ModeCardLarge default tone). */
    .entry-btn-wrap.real-col div[data-testid="stButton"] > button {
        background: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--hair) !important;
        box-shadow: var(--sh-1) !important;
    }
    .entry-btn-wrap.real-col div[data-testid="stButton"] > button:hover {
        background: var(--surface-2) !important;
        border-color: var(--hair-3) !important;
        box-shadow: var(--sh-2) !important;
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
            demo_label = "🧪\n\nDemo Mode\n\nGuided tour with simulated ICU data, agent preview with no tokens\n\n✨ Quick Start"
        else:
            demo_label = "🧪\n\n演示模式\n\n使用模拟 ICU 数据完成导览，预览智能体不消耗 token\n\n✨ 快速开始"
        if st.button(demo_label, key="entry_demo_btn", use_container_width=True, type="secondary"):
            clear_run_state("all")
            st.session_state.entry_mode = 'demo'
            st.session_state.use_mock_data = True
            st.session_state.database = 'mock'
            st.session_state.loaded_concepts = {}
            st.session_state.loaded_data_origin = 'none'
            st.session_state.patient_ids = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="entry-btn-wrap real-col">', unsafe_allow_html=True)
        if lang == 'en':
            real_label = "📊\n\nReal Data Mode\n\nConnect local ICU data or module exports, then run analysis before drafting\n\n🔬 Research Ready"
        else:
            real_label = "📊\n\n真实数据模式\n\n连接本地 ICU 数据或模块导出，先分析复核再生成文章\n\n🔬 科研就绪"
        if st.button(real_label, key="entry_real_btn", use_container_width=True, type="primary"):
            clear_run_state("all")
            st.session_state.entry_mode = 'real'
            st.session_state.use_mock_data = False
            st.session_state.loaded_concepts = {}
            st.session_state.loaded_data_origin = 'none'
            st.session_state.patient_ids = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    try:
        import streamlit_shadcn_ui as ui
    except Exception:
        ui = None

    feature_count = len(get_all_concepts())

    if lang == 'en':
        lead_text = "Start from a clinical question, prepare local ICU data, review analysis tables and figures, then draft a manuscript only when the evidence is worth writing up."
        stat_pills = ["6 ICU databases", f"{feature_count} clinical concepts", "Module exports", "Analysis-first agent", "Optional manuscript"]
        tab_options = ["Research workflow", "Clinical layer", "Execution"]
        tab_default = "Research workflow"
        dbs_label = "Supported Databases"
        overview = {
            "Research workflow": (
                "End-to-end research workflow",
                "Move from study design to analysis-ready data, reviewable results, and optional manuscript drafting.",
                [
                    ("🧲", "Question → cohort", "Turn a research question into cohort filters, disease definitions, and study constraints."),
                    ("🧩", "Cohort → modules", "Use stay-level files, EasyICU module folders, or guided extraction when data is not prepared yet."),
                    ("👁️", "Analysis → review gate", "Inspect tables, figures, missingness, model metrics, and findings before drafting."),
                ],
            ),
            "Clinical layer": (
                "Clinical intelligence",
                "Standardized concepts and computable rules sit behind every extraction step.",
                [
                    ("📚", "Standardized concept library", "One interface spanning all 6 supported public ICU databases."),
                    ("🧠", "Computable rules & scores", "SOFA, SOFA-2, Sepsis-3, KDIGO-AKI, qSOFA, circulatory failure, and more."),
                    ("🤖", "Research Agent", "Plan methods, assemble evidence, and pause after analysis so users stay in control."),
                ],
            ),
            "Execution": (
                "Execution engine",
                "Fast, reproducible extraction with temporal alignment and scalable processing controls.",
                [
                    ("⚙️", "Structured extraction", "Export module-aware datasets to CSV, Parquet, or Excel."),
                    ("⏱️", "Temporal harmonization", "Align intervals and time axes across heterogeneous ICU tables."),
                    ("🚀", "Staged automation", "Build cohorts explicitly, stream progress, stop after analysis, then draft on request."),
                ],
            ),
        }
    else:
        lead_text = "从临床研究问题出发，准备本地 ICU 数据，先复核分析表格和图，再决定是否生成文章初稿。"
        stat_pills = ["支持 6 大 ICU 数据库", f"{feature_count} 个临床概念", "模块化导出", "先分析后写作", "文章按需生成"]
        tab_options = ["科研工作流", "临床智能层", "执行能力层"]
        tab_default = "科研工作流"
        dbs_label = "支持的数据库"
        overview = {
            "科研工作流": (
                "端到端科研工作流",
                "从研究设计到分析数据、可复核结果，再到可选文章初稿，尽量在同一界面完成。",
                [
                    ("🧲", "问题 → 队列", "把研究问题转成队列筛选、疾病定义和研究约束。"),
                    ("🧩", "队列 → 模块", "支持 stay-level 文件、EasyICU 模块文件夹，或在无数据时引导提取。"),
                    ("👁️", "分析 → 复核关口", "先查看表格、图、缺失、模型指标和发现，再决定是否写文章。"),
                ],
            ),
            "临床智能层": (
                "临床智能层",
                "标准化概念和可计算规则支撑提取结果的一致性与可解释性。",
                [
                    ("📚", "标准化概念库", "统一接口覆盖 6 个公开 ICU 数据库。"),
                    ("🧠", "可计算规则与评分", "内置 SOFA、SOFA-2、Sepsis-3、KDIGO-AKI、qSOFA、循环衰竭等定义。"),
                    ("🤖", "研究智能体", "规划方法、组织证据，并默认停在分析阶段，让用户保留决策权。"),
                ],
            ),
            "执行能力层": (
                "执行能力层",
                "提供结构化导出、时间统一和更适合大数据的执行机制。",
                [
                    ("⚙️", "结构化提取", "按模块导出为 CSV、Parquet 或 Excel。"),
                    ("⏱️", "时间维度统一", "处理多数据库时间轴、间隔和患者级对齐。"),
                    ("🚀", "分阶段自动化", "显式构建队列、显示运行进度、先停在分析阶段，再按需写作。"),
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
            "Workflow overview tabs",
            options=tab_options,
            index=tab_options.index(tab_default),
            horizontal=True,
            label_visibility="collapsed",
            key=f"entry_tabs_fallback_{lang}",
        )

    if selected_tab not in overview:
        selected_tab = tab_default
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
