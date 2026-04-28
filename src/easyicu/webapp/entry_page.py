"""Entry page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any


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
        lead_text = "Start from a clinical task—such as sepsis early warning, AKI cohort construction, or patient trajectory analysis—then map it into one local-first ICU workflow."
        stat_pills = ["6 ICU databases", "167 clinical concepts", "Local-first processing", "Demo + Real Data", "AI-assisted workflow"]
        tab_options = ["Workflow", "Clinical layer", "Execution"]
        tab_default = "Workflow"
        dbs_label = "Supported Databases"
        overview = {
            "Workflow": (
                "User-facing workflow",
                "Move from study design to analysis-ready data without leaving the same interface.",
                [
                    ("🧲", "Task → cohort design", "Turn a research question into cohort filters, disease cohorts, and study constraints."),
                    ("🧩", "Task → concept selection", "Map your goal to 167 ICU features across 19 clinical groups."),
                    ("👁️", "Task → review", "Inspect time series, patient view, quality reports, and cohort summaries before export."),
                ],
            ),
            "Clinical layer": (
                "Clinical intelligence",
                "Standardized concepts and computable rules sit behind every extraction step.",
                [
                    ("📚", "Standardized concept library", "One interface spanning all 6 supported public ICU databases."),
                    ("🧠", "Computable rules & scores", "SOFA, SOFA-2, Sepsis-3, KDIGO-AKI, qSOFA, circulatory failure, and more."),
                    ("🤖", "AI copilot", "Start from your task, then get guidance on cohort filters, concepts, settings, and evidence."),
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
        lead_text = "从研究任务出发，例如脓毒症预警、AKI 队列构建或患者轨迹分析，再映射到一个本地优先的 ICU 数据工作流。"
        stat_pills = ["支持 6 大 ICU 数据库", "167 个临床概念", "本地优先处理", "演示 + 真实数据模式", "AI 辅助工作流"]
        tab_options = ["用户工作流", "临床智能层", "执行能力层"]
        tab_default = "用户工作流"
        dbs_label = "支持的数据库"
        overview = {
            "用户工作流": (
                "用户工作流",
                "从研究设计到分析数据导出，尽量在同一界面内完成。",
                [
                    ("🧲", "任务 → 队列设计", "把研究问题转成队列筛选、疾病队列和研究约束。"),
                    ("🧩", "任务 → 特征选择", "把任务映射到 19 个分组、167 个 ICU 特征。"),
                    ("👁️", "任务 → 结果复核", "在导出前查看时序图、患者视图、质量报告和队列摘要。"),
                ],
            ),
            "临床智能层": (
                "临床智能层",
                "标准化概念和可计算规则支撑提取结果的一致性与可解释性。",
                [
                    ("📚", "标准化概念库", "统一接口覆盖 6 个公开 ICU 数据库。"),
                    ("🧠", "可计算规则与评分", "内置 SOFA、SOFA-2、Sepsis-3、KDIGO-AKI、qSOFA、循环衰竭等定义。"),
                    ("🤖", "AI 助手", "从任务出发，帮助规划队列、模块、设置和参考依据。"),
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
