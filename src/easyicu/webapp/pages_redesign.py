"""Shell-A redesign · Tutorial / Quick Visualization / Research Agent / Entry.

Pages here render the exact layouts from the design canvas:

* ``page-tutorial.jsx`` — hero + 4-step workflow strip + 3-up starting
  points + Resources card.
* ``page-quick-viz.jsx`` — 4 subtabs: Data Tables / Time Series /
  Patient Overview / Data Quality.
* ``page-research-agent.jsx`` — inbound cohort + question/plan/run
  controls + 5-up output gallery + findings + review gate.
* ``page-entry.jsx`` — minimal top bar + hero + 2 mode cards + no-data
  fallback row + footer strip.

Each renderer reads from whatever session_state the legacy pages
populate, and falls back to deterministic demo values so the layout
is never blank — mirroring the visual style of the design preview.

Pages here intentionally render the **visual surface only**. Real
data loading and side effects continue to live in the legacy
``*_page.py`` modules; this module only takes over the render layer.
"""

from __future__ import annotations

import html
from collections.abc import MutableMapping
from typing import Any, Sequence

import streamlit as st

from easyicu.webapp import cohort_charts as cc
from easyicu.webapp.concept_catalog import (
    CONCEPT_DESCRIPTIONS,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_INTERNAL,
)


def _T(lang: str, en: str, zh: str) -> str:
    return en if lang == "en" else zh


def _esc(value: object) -> str:
    return html.escape(str(value))


def _tutorial_start_card(
    *,
    title: str,
    subtitle: str,
    desc: str,
    bullets: Sequence[str],
    badge_html: str = "",
    tone: str = "neutral",
) -> str:
    tone_class = " primary" if tone == "primary" else ""
    bullets_html = "".join(
        '<li><span></span><p>' + _esc(item) + "</p></li>"
        for item in bullets
    )
    return (
        f'<div class="eu-start-card{tone_class}">'
        '<div class="eu-start-head">'
        '<div>'
        f'<div class="eu-start-kicker">{_esc(subtitle)}</div>'
        f'<h3>{_esc(title)}</h3>'
        '</div>'
        f'{badge_html}'
        '</div>'
        f'<p class="eu-start-desc">{_esc(desc)}</p>'
        f'<ul class="eu-start-list">{bullets_html}</ul>'
        '</div>'
    )


def _tutorial_flow_card(steps: Sequence[dict[str, str]], lang: str) -> str:
    rows = []
    for step in steps:
        rows.append(
            '<div class="eu-flow-step">'
            f'<div class="eu-flow-num">{_esc(step["number"])}</div>'
            '<div>'
            f'<div class="eu-flow-title">{_esc(step["label"])}</div>'
            f'<div class="eu-flow-desc">{_esc(step["desc"])}</div>'
            '</div>'
            f'<div class="eu-flow-tag">{_esc(step["sub"])}</div>'
            '</div>'
        )
    return (
        '<div class="eu-rail-card">'
        f'<div class="eu-rail-title">{_T(lang, "Workflow", "工作流")}</div>'
        f'<div class="eu-flow-list">{"".join(rows)}</div>'
        '</div>'
    )


def _tutorial_agent_card(lang: str) -> str:
    gates = [
        (_T(lang, "Input locked", "输入锁定"), _T(lang, "cohort + concept manifest", "队列 + 变量清单")),
        (_T(lang, "Evidence bound", "证据绑定"), _T(lang, "scripts, logs, tables", "脚本、日志、表格")),
        (_T(lang, "Draft gated", "草稿闸门"), _T(lang, "claim audit before report", "审计后再写报告")),
    ]
    rows = "".join(
        '<div class="eu-agent-mini-row">'
        f'<span></span><div><b>{_esc(title)}</b><small>{_esc(note)}</small></div>'
        '</div>'
        for title, note in gates
    )
    return (
        '<div class="eu-rail-card eu-agent-mini">'
        f'<div class="eu-rail-title">{_T(lang, "Agent handoff", "Agent 交接")}</div>'
        f'<p>{_T(lang, "The web flow now mirrors the agent review gates: inputs are fixed before execution, evidence is tracked while running, and drafts stay behind an audit gate.", "Web 端现在对齐 agent 的审阅闸门：先固定输入，再追踪证据，最后通过审计闸门再生成草稿。")}</p>'
        f'<div>{rows}</div>'
        '</div>'
    )


def _apply_demo_defaults_for_tutorial(state: MutableMapping[str, Any]) -> None:
    """Install the compact demo defaults used by tutorial jump actions."""
    state["entry_mode"] = "demo"
    state["use_mock_data"] = True
    state["database"] = "mock"
    state["mock_params"] = {
        "n_patients": 10,
        "hours": 24,
        "demo_profile": "lite",
    }
    state.setdefault("loaded_concepts", {})
    state.setdefault("loaded_data_origin", "none")
    state.setdefault("patient_ids", [])


def _route_to_extract_entry_mode(
    state: MutableMapping[str, Any],
    target: str,
) -> None:
    """Enter extraction from a mode CTA without carrying stale run state."""
    if target not in {"demo", "real"}:
        return

    previous_database = state.get("database")
    state["entry_mode"] = target
    state["use_mock_data"] = target == "demo"

    if target == "demo":
        state["database"] = "mock"
        state["mock_params"] = {
            "n_patients": 10,
            "hours": 24,
            "demo_profile": "lite",
        }
    else:
        valid_real_databases = {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}
        if previous_database not in valid_real_databases:
            state["database"] = "miiv"
            state["path_validated"] = False
            state.pop("last_validated_path", None)

    for key in ("step1_confirmed", "step2_confirmed", "step3_confirmed", "export_completed"):
        state[key] = False
    state["trigger_export"] = False
    state["_exporting_in_progress"] = False
    state["loaded_concepts"] = {}
    state["loaded_data_origin"] = "none"
    state["patient_ids"] = []
    state["all_patient_count"] = 0
    state["selected_patient"] = None
    state["selected_concepts"] = []
    state.pop("quick_viz_active_panel", None)
    state.pop("_preview_requested", None)
    state.pop("_viz_import_export_auto_trigger", None)
    state.pop("_scroll_to_tab", None)
    state["_active_main_page"] = "extract"


def _selected_tutorial_module_concepts(state: MutableMapping[str, Any]) -> list[str]:
    selected_module = state.get("_eu_tutorial_dict_module")
    if selected_module not in CONCEPT_GROUPS_INTERNAL:
        selected_module = next(iter(CONCEPT_GROUPS_INTERNAL), "")
    return [
        name
        for name in CONCEPT_GROUPS_INTERNAL.get(str(selected_module), [])
        if name in CONCEPT_DICTIONARY
    ]


def _apply_tutorial_resource_action(
    state: MutableMapping[str, Any],
    target: str,
) -> None:
    """Route tutorial resource buttons to real app destinations."""
    if target == "sample_cohorts":
        _apply_demo_defaults_for_tutorial(state)
        state["_active_main_page"] = "cohort"
        state["_eu_topbar_run_request"] = {
            "page": "cohort",
            "requested_at": "tutorial_resource_sample_cohorts",
        }
        return

    if target == "concept_catalog":
        if state.get("entry_mode") == "none":
            _apply_demo_defaults_for_tutorial(state)
        state["step1_confirmed"] = True
        state["step2_confirmed"] = True
        state["step3_confirmed"] = False
        module_concepts = _selected_tutorial_module_concepts(state)
        if module_concepts:
            state["selected_concepts"] = module_concepts
        state["_active_main_page"] = "extract"
        return

    if target == "citation_info":
        if state.get("entry_mode") == "none":
            _apply_demo_defaults_for_tutorial(state)
        state["_active_main_page"] = "research_agent"
        state["_ra_view"] = "setup"
        state["_eu_ra_resource_focus"] = "citation_info"
        return

    raise ValueError(f"Unknown tutorial resource target: {target}")


def _render_tutorial_resources_card(lang: str) -> None:
    resources = [
        ("sample_cohorts", _T(lang, "Sample cohorts", "样例队列")),
        ("concept_catalog", _T(lang, "Concept catalog", "概念目录")),
        ("citation_info", _T(lang, "Citation info", "引用信息")),
    ]
    with st.container(key="eu_tutorial_resources_card"):
        st.markdown(
            f'<div class="eu-rail-title">{_T(lang, "Resources", "资源")}</div>',
            unsafe_allow_html=True,
        )
        for target, label in resources:
            if st.button(
                f"{label}  ›",
                key=f"_eu_tutorial_resource_{target}",
                use_container_width=True,
            ):
                _apply_tutorial_resource_action(st.session_state, target)
                st.rerun()


def _dictionary_module_label(module_key: str, lang: str) -> str:
    names = CONCEPT_GROUP_NAMES.get(module_key, (module_key, module_key))
    return names[0] if lang == "en" else names[1]


def _tutorial_dictionary_modules(lang: str) -> list[dict[str, object]]:
    """Return all user-facing dictionary modules and their concepts."""
    modules: list[dict[str, object]] = []
    for module_key, concept_names in CONCEPT_GROUPS_INTERNAL.items():
        visible_concepts = [name for name in concept_names if name in CONCEPT_DICTIONARY]
        if not visible_concepts:
            continue
        modules.append({
            "key": module_key,
            "label": _dictionary_module_label(module_key, lang),
            "concepts": visible_concepts,
        })
    return modules


def _concept_display_name(concept: str, lang: str) -> str:
    name_en, name_zh, _unit = CONCEPT_DICTIONARY.get(concept, (concept, concept, ""))
    return name_en if lang == "en" else name_zh


def _concept_description(concept: str, lang: str) -> str:
    desc_en, desc_zh = CONCEPT_DESCRIPTIONS.get(concept, ("", ""))
    fallback = _concept_display_name(concept, lang)
    return (desc_en if lang == "en" else desc_zh) or fallback


def _tutorial_dictionary_module_html(
    lang: str,
    *,
    selected_module: str | None = None,
) -> str:
    """Render the selected dictionary module without an internal scroll area."""
    modules = _tutorial_dictionary_modules(lang)
    selected = next(
        (module for module in modules if module["key"] == selected_module),
        modules[0],
    )
    module_key = str(selected["key"])
    module_label = str(selected["label"])
    concept_names = list(selected["concepts"])
    rows: list[str] = []
    for concept in concept_names:
        _name_en, _name_zh, unit = CONCEPT_DICTIONARY.get(concept, (concept, concept, ""))
        rows.append(
            '<div class="eu-dict-list-row">'
            f'<code>{_esc(concept)}</code>'
            '<div>'
            f'<b>{_esc(_concept_display_name(concept, lang))}</b>'
            f'<p>{_esc(_concept_description(concept, lang))}</p>'
            '</div>'
            f'<em>{_esc(unit or "—")}</em>'
            '</div>'
        )
    return (
        '<section class="eu-dict-module-preview" data-active="true" '
        f'data-module="{_esc(module_key)}">'
        '<div class="eu-dict-module-heading">'
        '<div>'
        f'<span>{_T(lang, "Selected module", "当前模块")}</span>'
        f'<b>{_esc(module_label)}</b>'
        '</div>'
        f'<em>{len(concept_names)} {_T(lang, "features", "个特征")}</em>'
        '</div>'
        '<div class="eu-dict-table-head">'
        f'<span>{_T(lang, "Code", "代码")}</span>'
        f'<span>{_T(lang, "Meaning", "含义")}</span>'
        f'<span>{_T(lang, "Unit", "单位")}</span>'
        '</div>'
        f'{"".join(rows)}'
        '</section>'
    )


def _render_tutorial_dictionary(lang: str) -> None:
    """Render the full, selectable Tutorial data dictionary."""
    modules = _tutorial_dictionary_modules(lang)
    if not modules:
        return
    total_features = sum(len(module["concepts"]) for module in modules)
    module_keys = [str(module["key"]) for module in modules]
    labels_by_key = {str(module["key"]): str(module["label"]) for module in modules}
    concepts_by_key = {str(module["key"]): list(module["concepts"]) for module in modules}

    current_module = st.session_state.get("_eu_tutorial_dict_module")
    if current_module not in concepts_by_key:
        current_module = module_keys[0]
        st.session_state["_eu_tutorial_dict_module"] = current_module

    with st.container(key="eu_tutorial_dictionary_panel"):
        st.markdown(
            '<div class="eu-dict-head">'
            '<div>'
            f'<div class="eu-dict-kicker">{_T(lang, "Data dictionary", "数据字典")}</div>'
            f'<h3>{_T(lang, "Canonical ICU concept catalog", "标准 ICU 概念目录")}</h3>'
            f'<p>{_T(lang, "Browse the complete module-grouped EasyICU dictionary. Each row maps a compact code to its clinical meaning, description, and unit before extraction begins.", "浏览按模块组织的完整 EasyICU 数据字典。每一行都把简短代码映射到临床含义、解释和单位，供抽取前确认。")}</p>'
            '</div>'
            '<div class="eu-dict-source">'
            '<span>source</span>'
            '<code>src/easyicu/data/concept-dict.json</code>'
            f'<strong>{total_features} {_T(lang, "features", "个特征")} · {len(modules)} {_T(lang, "modules", "个模块")}</strong>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        selected_module = st.selectbox(
            _T(lang, "Module", "模块"),
            options=module_keys,
            format_func=lambda key: f"{labels_by_key[str(key)]} · {len(concepts_by_key[str(key)])}",
            key="_eu_tutorial_dict_module",
        )
        st.markdown(
            _tutorial_dictionary_module_html(
                lang,
                selected_module=selected_module,
            ),
            unsafe_allow_html=True,
        )


# =====================================================================
# Data extraction page
# =====================================================================


def _render_tutorial_redesign_page_legacy(lang: str) -> None:
    st.markdown(
        # Hero ------------------------------------------------------
        '<div style="padding:0 4px 8px">'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4);'
        f'letter-spacing:.06em;text-transform:uppercase">{_T(lang, "Data extraction", "数据提取")}</div>'
        f'<h1 style="margin:6px 0;font-size:28px;font-weight:500;letter-spacing:-0.02em;color:var(--ink)">'
        f'{_T(lang, "Extract, review, analyze, draft.", "数据抽取 → 审阅 → 分析 → 起草")}</h1>'
        f'<p style="margin:0;color:var(--ink-3);font-size:13.5px;max-width:760px;line-height:1.55">'
        f'{_T(lang, "EasyICU is a local-first ICU research workspace. The four steps below cover the core data-preparation flow — once complete you can move into the analysis modules or hand the cohort off to the Research Agent.", "EasyICU 是一套本地优先的 ICU 数据研究工作台。下面四步是核心数据准备流程，完成后即可进入分析模块或交给研究智能体。")}'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # 4-step workflow strip ---------------------------------------
    steps = [
        {
            "number": "1",
            "icon": "database",
            "label_en": _T(lang, "Data source", "Data source"),
            "label_zh": _T(lang, "数据源", "数据源"),
            "desc": _T(lang,
                "Demo · MIMIC-IV · eICU · AmsterdamUMCdb · HiRID · MIMIC-III · SICdb. Code-only mode without data is supported.",
                "Demo · MIMIC-IV · eICU · AmsterdamUMCdb · HiRID · MIMIC-III · SICdb。也支持仅生成代码不连数据。"),
            "sub": _T(lang, "3 modes", "3 种模式"),
        },
        {
            "number": "2",
            "icon": "users",
            "label_en": _T(lang, "Cohort", "Cohort"),
            "label_zh": _T(lang, "队列", "队列"),
            "desc": _T(lang,
                "Filter by age, sex, ICU LOS, outcome, clinical cohorts (Sepsis-3, AKI, ARDS), and ICD codes.",
                "按年龄、性别、ICU 时长、转归、Sepsis-3/AKI/ARDS 等临床队列、ICD 编码筛选。"),
            "sub": _T(lang, "9 filters", "9 项筛选"),
        },
        {
            "number": "3",
            "icon": "layers",
            "label_en": _T(lang, "Concepts", "Concepts"),
            "label_zh": _T(lang, "变量", "变量"),
            "desc": _T(lang,
                "Core clinical modules are ready for single-select or merge preview, with timestamps automatically aligned.",
                "核心临床模块可单选或合并预览，并自动对齐时间轴。"),
            "sub": _T(lang, "module catalog", "模块目录"),
        },
        {
            "number": "4",
            "icon": "bars",
            "label_en": _T(lang, "Analysis", "Analysis"),
            "label_zh": _T(lang, "分析", "分析"),
            "desc": _T(lang,
                "Quick Visualization, Cohort Statistics, Cross-DB Benchmark — or hand off to the Research Agent.",
                "患者审阅、队列统计、跨数据库对比，或交给研究智能体自动产出。"),
            "sub": _T(lang, "4 surfaces", "4 个面板"),
        },
    ]
    st.markdown(
        '<div style="margin-top:18px">' + cc.render_workflow_strip(steps) + '</div>',
        unsafe_allow_html=True,
    )

    # Starting points -------------------------------------------------
    st.markdown(
        f'<div class="eu-section-label" style="padding:0;margin:24px 0 10px">'
        f'<span>{_T(lang, "Choose a starting point · 选择起点", "选择起点")}</span></div>',
        unsafe_allow_html=True,
    )

    badge_recommended = (
        f'<span class="eu-pill" style="background:var(--surface);border-color:var(--hair-2)">{_T(lang, "try first", "新手先试")}</span>'
    )
    badge_localonly = (
        '<span class="eu-pill ok"><span class="dot"></span>local-only</span>'
    )
    demo_card = cc.render_tutorial_starting_card(
        tone="accent",
        icon="flask",
        title_en=_T(lang, "Demo Mode", "Demo Mode"),
        title_zh=_T(lang, "演示模式", "演示模式"),
        badge_html=badge_recommended,
        desc=_T(lang,
            "Automatically generates reproducible mock data. Full cohort-builder, Quick Viz, and Research Agent gallery experience. No tokens, no local data needed.",
            "自动生成可重复的模拟数据。完整体验队列构建、患者审阅和研究智能体静态画廊。无需令牌、无需本地数据。"),
        bullets=[
            _T(lang, "10-patient fast review set · 24h default",
                    "10 例快速审阅集 · 默认 24 小时"),
            _T(lang, "Lightweight review data is ready immediately",
                    "轻量审阅数据可立即打开"),
            _T(lang, "Research Agent static gallery available",
                    "研究智能体静态输出画廊可查看"),
            _T(lang, "Switching sessions never loses your real work",
                    "会话切换不会丢失你的真实工作"),
        ],
        cta_label=_T(lang, "Start demo", "开始演示"),
        cta_primary=True,
    )
    real_card = cc.render_tutorial_starting_card(
        tone="neutral",
        icon="database",
        title_en=_T(lang, "Real Data", "Real Data"),
        title_zh=_T(lang, "真实数据", "真实数据"),
        badge_html=badge_localonly,
        desc=_T(lang,
            "Connect to ICU database exports on your machine. Everything is processed locally — EasyICU never uploads or transmits data.",
            "连接你机器上的 ICU 数据库导出文件。所有处理在本地完成,EasyICU 不会上传或外发任何数据。"),
        bullets=[
            _T(lang, "MIMIC-IV · eICU · AUMC · HiRID · MIMIC-III · SICdb",
                    "MIMIC-IV · eICU · AUMC · HiRID · MIMIC-III · SICdb"),
            _T(lang, "Auto path detection + one-click CSV → parquet",
                    "路径自动检测 + 一键 CSV → parquet 转换"),
            _T(lang, "Module-folder mode reuses prior exports",
                    "模块文件夹模式支持复用之前的导出"),
            _T(lang, "Cross-DB Benchmark can connect ≥ 2 databases",
                    "跨数据库对比可同时连接 ≥ 2 个数据库"),
        ],
        cta_label=_T(lang, "Configure data path", "配置数据路径"),
    )
    nodata_card = cc.render_tutorial_starting_card(
        tone="neutral",
        icon="file",
        title_en=_T(lang, "No Data", "No Data"),
        title_zh=_T(lang, "仅代码", "仅代码"),
        badge_html="",
        desc=_T(lang,
            "No data yet? Let the Research Agent generate a reusable code skeleton first; plug in real data later.",
            "还没有数据？先让研究智能体生成可复用的代码骨架，稍后再接入真实数据。"),
        bullets=[
            _T(lang, "Generate cohort.py / analysis.py", "生成 cohort.py / analysis.py"),
            _T(lang, "Methods section draft", "Methods 段草稿"),
        ],
        cta_label=_T(lang, "Skip data for now", "跳过数据"),
        cta_dashed=True,
    )
    st.markdown(
        '<div style="display:grid;grid-template-columns:1fr 1fr 0.8fr;gap:12px;align-items:stretch">'
        f'{demo_card}{real_card}{nodata_card}</div>',
        unsafe_allow_html=True,
    )

    # Action buttons. Every CTA routes the user somewhere visible so
    # nothing is a dead click — the previous "Start demo" did nothing
    # when already in demo mode.
    cols = st.columns([1, 1, 0.8])
    with cols[0]:
        if st.button(_T(lang, "Start demo → Extract", "开始演示 → 提取"),
                     key="_eu_tutorial_demo", type="primary",
                     use_container_width=True):
            _route_to_extract_entry_mode(st.session_state, "demo")
            st.rerun()
    with cols[1]:
        if st.button(_T(lang, "Configure data path → Extract", "配置数据路径 → 提取"),
                     key="_eu_tutorial_real",
                     use_container_width=True):
            _route_to_extract_entry_mode(st.session_state, "real")
            st.rerun()
    with cols[2]:
        if st.button(_T(lang, "Skip data → Agent", "跳过数据 → Agent"),
                     key="_eu_tutorial_nodata",
                     use_container_width=True):
            st.session_state["_active_main_page"] = "research_agent"
            st.rerun()

    # Resources --------------------------------------------------
    st.markdown(
        '<div class="eu-card" style="padding:14px 18px;display:flex;align-items:center;gap:18px;margin-top:18px">'
        '<div style="display:flex;align-items:center;gap:10px">'
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">'
        '<path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H20v15H5.5A1.5 1.5 0 0 0 4 19.5v-15Z"/></svg>'
        '<div>'
        f'<div style="font-size:12.5px;font-weight:500">{_T(lang, "Resources", "资源")}</div>'
        f'<div class="eu-cn" style="font-size:11px;color:var(--ink-4)">'
        f'{_T(lang, "docs · samples · citation", "文档 / 样例 / 引用")}</div>'
        '</div></div>'
        '<div style="display:flex;gap:6px;margin-left:auto;flex-wrap:wrap">'
        f'<span class="eu-pill" style="background:var(--surface);height:26px">{_T(lang, "Sample cohorts", "样例队列")}</span>'
        f'<span class="eu-pill" style="background:var(--surface);height:26px">{_T(lang, "Concept catalog", "概念目录")}</span>'
        f'<span class="eu-pill" style="background:var(--surface);height:26px">{_T(lang, "Citation info", "引用信息")}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_tutorial_redesign_page(lang: str) -> None:
    st.markdown(
        '<div class="eu-tutorial-hero">'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4);'
        f'letter-spacing:0;text-transform:uppercase">{_T(lang, "Data extraction", "数据提取")}</div>'
        f'<h1>{_T(lang, "Extract, review, analyze, draft.", "数据抽取 → 审阅 → 分析 → 起草")}</h1>'
        f'<p>{_T(lang, "Start with demo data, a local database export, or code-only agent drafting. The page is arranged like a workbench: entry actions on the left, workflow and agent gates on the right.", "你可以从演示数据、本地数据库导出或仅代码 agent 起草开始。页面按工作台组织：左侧选择入口，右侧保留流程和 agent 审阅闸门。")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    steps = [
        {
            "number": "1",
            "label": _T(lang, "Data source", "数据源"),
            "desc": _T(lang, "Demo, real ICU exports, or code-only mode.", "演示、本地 ICU 导出或仅代码模式。"),
            "sub": _T(lang, "3 modes", "3 种"),
        },
        {
            "number": "2",
            "label": _T(lang, "Cohort", "队列"),
            "desc": _T(lang, "Filter age, outcome, LOS, ICD, Sepsis-3, AKI, ARDS.", "筛选年龄、转归、LOS、ICD、Sepsis-3、AKI、ARDS。"),
            "sub": _T(lang, "9 filters", "9 项"),
        },
        {
            "number": "3",
            "label": _T(lang, "Concepts", "变量"),
            "desc": _T(lang, "Select modules, merge previews, align timestamps.", "选择模块、合并预览并对齐时间轴。"),
            "sub": _T(lang, "module catalog", "模块目录"),
        },
        {
            "number": "4",
            "label": _T(lang, "Analysis", "分析"),
            "desc": _T(lang, "Review, compare databases, or hand off to the agent.", "审阅、比较多个数据库，或交给研究智能体。"),
            "sub": _T(lang, "4 views", "4 视图"),
        },
    ]

    badge_recommended = (
        f'<span class="eu-pill" style="background:var(--surface);border-color:var(--hair-2)">{_T(lang, "try first", "新手先试")}</span>'
    )
    badge_localonly = (
        f'<span class="eu-pill ok"><span class="dot"></span>{_T(lang, "local-only", "仅本地")}</span>'
    )
    demo_card = _tutorial_start_card(
        tone="primary",
        title=_T(lang, "Demo Mode", "演示模式"),
        subtitle=_T(lang, "Fastest complete path", "最快完整路径"),
        badge_html=badge_recommended,
        desc=_T(lang,
            "Generate reproducible mock ICU data and walk through extraction, review, visualization, and agent handoff without touching local files.",
            "生成可复现的模拟 ICU 数据，直接走完抽取、审阅、可视化和 agent 交接，不需要接触本地文件。"),
        bullets=[
            _T(lang, "10-patient fast review set with a 24h default window",
                    "10 例快速审阅集，默认 24 小时时间窗"),
            _T(lang, "Lightweight review data is ready immediately",
                    "轻量审阅数据可立即打开"),
            _T(lang, "Agent gallery and audit handoff are ready",
                    "Agent 画廊和审计交接可直接查看"),
        ],
    )
    real_card = _tutorial_start_card(
        title=_T(lang, "Real Data", "真实数据"),
        subtitle=_T(lang, "Use local ICU exports", "连接本地 ICU 导出"),
        badge_html=badge_localonly,
        desc=_T(lang,
            "Point EasyICU to prepared database folders and keep processing on your machine.",
            "将 EasyICU 指向已准备好的数据库目录，所有处理留在本机。"),
        bullets=[
            _T(lang, "MIMIC-IV, eICU, AUMC, HiRID, MIMIC-III, SICdb",
                    "MIMIC-IV、eICU、AUMC、HiRID、MIMIC-III、SICdb"),
            _T(lang, "Reuse prior module-folder exports",
                    "可复用之前的模块导出目录"),
        ],
    )
    nodata_card = _tutorial_start_card(
        title=_T(lang, "No Data", "仅代码"),
        subtitle=_T(lang, "Draft first", "先起草代码"),
        desc=_T(lang,
            "Let the Research Agent prepare a reusable analysis skeleton before data are available.",
            "还没有数据时，先让研究智能体准备可复用的分析代码骨架。"),
        bullets=[
            _T(lang, "Generate cohort.py / analysis.py", "生成 cohort.py / analysis.py"),
            _T(lang, "Create a methods-section draft", "生成 Methods 段草稿"),
        ],
    )

    main_col, rail_col = st.columns([1.42, 0.72], gap="large")
    with main_col:
        st.markdown(
            f'<div class="eu-section-label" style="padding:0;margin:4px 0 10px">'
            f'<span>{_T(lang, "Choose a starting point", "选择起点")}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(demo_card, unsafe_allow_html=True)
        if st.button(_T(lang, "Start demo -> Extract", "开始演示 -> 提取"),
                     key="_eu_tutorial_demo", type="primary",
                     use_container_width=True):
            _route_to_extract_entry_mode(st.session_state, "demo")
            st.rerun()

        with st.container(key="eu_tutorial_secondary_starts"):
            secondary_left, secondary_right = st.columns(2, gap="medium")
            with secondary_left:
                st.markdown(real_card, unsafe_allow_html=True)
                if st.button(_T(lang, "Configure data path -> Extract", "配置数据路径 -> 提取"),
                             key="_eu_tutorial_real",
                             use_container_width=True):
                    _route_to_extract_entry_mode(st.session_state, "real")
                    st.rerun()
            with secondary_right:
                st.markdown(nodata_card, unsafe_allow_html=True)
                if st.button(_T(lang, "Skip data -> Agent", "跳过数据 -> Agent"),
                             key="_eu_tutorial_nodata",
                             use_container_width=True):
                    st.session_state["_active_main_page"] = "research_agent"
                    st.rerun()

        _render_tutorial_dictionary(lang)
        if st.button(
            _T(lang, "Open selected module -> Concepts", "打开所选模块 -> 变量选择"),
            key="_eu_tutorial_dictionary",
            use_container_width=True,
        ):
            if st.session_state.get("entry_mode") == "none":
                st.session_state["entry_mode"] = "demo"
                st.session_state["use_mock_data"] = True
                st.session_state["database"] = "mock"
            st.session_state["step1_confirmed"] = True
            st.session_state["step2_confirmed"] = True
            st.session_state["step3_confirmed"] = False
            selected_module = st.session_state.get("_eu_tutorial_dict_module")
            module_concepts = [
                name
                for name in CONCEPT_GROUPS_INTERNAL.get(str(selected_module), [])
                if name in CONCEPT_DICTIONARY
            ]
            if module_concepts:
                st.session_state["selected_concepts"] = module_concepts
            st.session_state["_active_main_page"] = "extract"
            st.rerun()

    with rail_col:
        st.markdown(
            _tutorial_flow_card(steps, lang)
            + _tutorial_agent_card(lang),
            unsafe_allow_html=True,
        )
        _render_tutorial_resources_card(lang)


# =====================================================================
# Quick Visualization page
# =====================================================================


def _quick_viz_modules(lang: str) -> list[tuple[str, int, bool]]:
    label_map = {
        "Vital Signs":     ("Vital Signs", "生命体征"),
        "Chemistry":       ("Chemistry", "生化"),
        "CBC":             ("CBC", "血常规"),
        "Coagulation":     ("Coagulation", "凝血"),
        "Blood Gas":       ("Blood Gas", "血气"),
        "SOFA components": ("SOFA components", "SOFA 组分"),
        "Mech Vent":       ("Mech Vent", "机械通气"),
        "Fluid Balance":   ("Fluid Balance", "液体平衡"),
        "Demographics":    ("Demographics", "人口学"),
        "Outcomes":        ("Outcomes", "转归"),
        "Sepsis-3":        ("Sepsis-3", "Sepsis-3"),
        "AKI · KDIGO":     ("AKI · KDIGO", "AKI · KDIGO"),
    }
    counts = [7, 14, 9, 5, 10, 6, 9, 8, 4, 5, 4, 5]
    out = []
    for i, (key, (en, zh)) in enumerate(label_map.items()):
        out.append((en if lang == "en" else zh, counts[i], i == 0))
    return out


def _render_qv_data_tables(lang: str) -> None:
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Quick Visualization · 快速可视化", "快速可视化"),
            title_en=_T(lang, "Module table preview", "Module table preview"),
            title_zh=_T(lang, "模块表预览", "模块表预览"),
            desc=_T(lang,
                "Inspect exported data by module. Merge All shows the wide table; Single Feature shows the long table.",
                "按模块查看导出的数据。Merge All 显示宽表;Single Feature 显示单变量长表。"),
            right_html=(
                f'<span class="eu-pill mono">{_T(lang, "demo catalog · 10 patients", "演示目录 · 10 例")}</span>'
            ),
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 3.2], gap="medium")
    with left:
        st.markdown(
            cc.render_module_picker(_quick_viz_modules(lang)),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="eu-card" style="padding:14px">'
            '<div style="display:flex;align-items:center;justify-content:space-between">'
            '<div style="display:flex;align-items:center;gap:10px">'
            '<div style="width:32px;height:32px;border-radius:6px;background:var(--surface-2);'
            'display:flex;align-items:center;justify-content:center;color:var(--ink-3)">'
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="4"/></svg>'
            '</div>'
            '<div>'
            f'<div style="font-size:14px;font-weight:500">{_T(lang, "Vital Signs", "生命体征")}'
            f' <span class="eu-cn" style="color:var(--ink-3);font-weight:400;margin-left:6px">'
            f'{_T(lang, "Vital signs", "生命体征")}</span></div>'
            f'<div style="font-size:12px;color:var(--ink-3)">'
            f'{_T(lang, "Core bedside measurements aligned to a compact longitudinal preview.", "床旁核心测量,对齐成紧凑的纵向预览。")}'
            '</div></div></div>'
            '<div style="display:flex;gap:6px">'
            '<div style="padding:6px 10px;background:var(--surface-2);border-radius:6px">'
            '<div style="font-size:9.5px;color:var(--ink-4);letter-spacing:.06em;text-transform:uppercase">features</div>'
            '<div class="mono" style="font-size:14px;font-weight:500">7</div></div>'
            '<div style="padding:6px 10px;background:var(--surface-2);border-radius:6px">'
            '<div style="font-size:9.5px;color:var(--ink-4);letter-spacing:.06em;text-transform:uppercase">patients</div>'
            '<div class="mono" style="font-size:14px;font-weight:500">50</div></div>'
            '</div></div>'
            '<div style="margin-top:10px;display:flex;gap:4px;flex-wrap:wrap">'
            + "".join(f'<span class="eu-chip mono">{c}</span>'
                      for c in ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp"])
            + '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            cc.render_data_preview_table(
                title=_T(lang, "Merged preview · 1,000 of 7,200 rows", "合并预览 · 1,000 / 7,200 行"),
                meta=_T(lang, "2,000 rows/feature · 9 cols", "每特征 2,000 行 · 9 列"),
                columns=["stay_id", "charttime", "hr", "map", "sbp", "dbp", "temp", "spo2", "resp"],
                rows=[
                    [20001, "00:00", 92, 82, 132, 78, 36.8, 96, 18],
                    [20001, "01:00", 95, 78, 128, 76, 37.0, 95, 20],
                    [20001, "02:00", 101, 70, 119, 70, 37.4, 93, 24],
                    [20001, "03:00", 108, 64, 110, 64, 38.1, 91, 28],
                    [20001, "04:00", 110, 60, 105, 58, 38.5, 90, 30],
                    [20002, "00:00", 78, 88, 144, 84, 36.5, 98, 14],
                    [20002, "01:00", 80, 86, 141, 82, 36.6, 98, 15],
                    [20002, "02:00", 82, 85, 138, 80, 36.7, 97, 16],
                ],
            ),
            unsafe_allow_html=True,
        )


def _render_qv_time_series(lang: str) -> None:
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Quick Visualization · 快速可视化", "快速可视化"),
            title_en=_T(lang, "Time series", "Time series"),
            title_zh=_T(lang, "时间序列", "时间序列"),
            desc=_T(lang,
                "Interactive visualization, single & multi-patient comparison.",
                "交互式可视化,单患者 / 多患者对比。"),
            right_html='',
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eu-card" style="padding:14px 16px;display:flex;align-items:center;gap:14px;margin-top:6px">'
        '<div>'
        f'<div style="font-size:12px;font-weight:500">{_T(lang, "Analysis mode", "分析模式")}</div>'
        f'<div style="font-size:11px;color:var(--ink-4)">'
        f'{_T(lang, "Lanes group by clinical system; Single = drill-down; Multi = compare.", "Lanes 按系统分组;Single 钻取;Multi 对比。")}</div>'
        '</div>'
        '<div style="margin-left:auto;display:flex;gap:4px;align-items:center;background:var(--surface-2);'
        'border-radius:6px;padding:2px">'
        f'<span style="padding:4px 10px;background:var(--surface);border-radius:4px;font-size:12px;font-weight:500">'
        f'{_T(lang, "Clinical Lanes", "临床分组")}</span>'
        f'<span style="padding:4px 10px;font-size:12px;color:var(--ink-3)">{_T(lang, "Single Patient", "单患者")}</span>'
        f'<span style="padding:4px 10px;font-size:12px;color:var(--ink-3)">{_T(lang, "Multi-Patient", "多患者")}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eu-card" style="padding:12px 16px;display:flex;align-items:center;gap:12px;margin-top:10px">'
        '<div style="width:32px;height:32px;border-radius:6px;background:var(--surface-2);'
        'display:flex;align-items:center;justify-content:center">'
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">'
        '<circle cx="12" cy="8" r="3.5"/><path d="M4 21a8 8 0 0 1 16 0"/></svg></div>'
        '<div style="display:flex;gap:18px;align-items:baseline">'
        '<div>'
        '<div class="mono" style="font-size:14px;font-weight:500">stay_20001</div>'
        '<div style="font-size:10.5px;color:var(--ink-4)">72 y · M · sepsis-3 +</div>'
        '</div>'
        '<div class="mono" style="font-size:11.5px;color:var(--ink-3)">LOS 6.2d · SOFA max 9 · survived</div>'
        '</div>'
        f'<div style="margin-left:auto;display:flex;gap:6px">'
        f'<span class="eu-pill">{_T(lang, "Prev", "上一例")}</span>'
        f'<span class="eu-pill">{_T(lang, "Next", "下一例")}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    bedside_lanes = "".join([
        cc.render_lane(title_en="Heart rate", title_zh="心率", unit="bpm",
                       data=[18, 22, 16, 12, 15, 18, 14, 12, 10, 8, 14, 18, 16, 12]),
        cc.render_lane(title_en="MAP", title_zh="平均动脉压", unit="mmHg",
                       data=[12, 14, 16, 20, 24, 30, 28, 26, 22, 18, 14, 10, 8, 6], threshold=25),
        cc.render_lane(title_en="SpO₂", title_zh="血氧", unit="%",
                       data=[8, 10, 12, 16, 20, 18, 14, 10, 8, 6, 8, 10, 12, 8]),
        cc.render_lane(title_en="Temperature", title_zh="体温", unit="°C",
                       data=[20, 22, 26, 30, 32, 28, 24, 20, 18, 16, 18, 22, 26, 24], threshold=28),
    ])
    st.markdown(
        '<div style="margin-top:10px">'
        + cc.render_lane_group(
            _T(lang, "Bedside lane", "床旁面板"),
            "0h – 72h",
            bedside_lanes,
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    labs_lanes = "".join([
        cc.render_lane(title_en="Lactate", title_zh="乳酸", unit="mmol/L",
                       data=[14, 16, 20, 24, 28, 30, 26, 22, 18, 14], threshold=20),
        cc.render_lane(title_en="Creatinine", title_zh="肌酐", unit="mg/dL",
                       data=[10, 12, 14, 18, 22, 24, 22, 20, 18, 16]),
    ])
    st.markdown(
        '<div style="margin-top:10px">'
        + cc.render_lane_group(
            _T(lang, "Labs lane", "化验面板"),
            _T(lang, "q2h chemistry, q4h gas", "化验 q2h · 血气 q4h"),
            labs_lanes,
        )
        + '</div>',
        unsafe_allow_html=True,
    )


def _render_qv_patient_overview(lang: str) -> None:
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Quick Visualization · 快速可视化", "快速可视化"),
            title_en=_T(lang, "Patient overview", "Patient overview"),
            title_zh=_T(lang, "病人全景", "病人全景"),
            desc=_T(lang,
                "Single-patient summary — demographics, timeline, key features.",
                "单患者总览 · 基本信息、时间线、关键特征。"),
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eu-card" style="padding:18px;display:flex;gap:24px;margin-top:6px;align-items:center">'
        '<div style="width:56px;height:56px;border-radius:12px;background:var(--surface-2);'
        'display:flex;align-items:center;justify-content:center;flex:none">'
        '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">'
        '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg></div>'
        '<div style="flex:1">'
        '<div style="font-size:18px;font-weight:500">stay_20001 · M · 72 y</div>'
        f'<div style="font-size:12.5px;color:var(--ink-3);margin-top:2px">'
        f'{_T(lang, "Sepsis-3 positive · BMI 27.3 · admitted via ED · medical ICU", "Sepsis-3 阳性 · BMI 27.3 · 急诊入院 · 内科 ICU")}</div>'
        '</div>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;min-width:460px">'
        + "".join([
            '<div style="padding:10px;background:var(--surface-2);border-radius:6px">'
            f'<div style="font-size:9.5px;color:var(--ink-4);letter-spacing:.06em;text-transform:uppercase;font-weight:500">{l}</div>'
            f'<div class="mono" style="font-size:14px;font-weight:500;margin-top:1px;color:{tone}">{v}</div>'
            '</div>'
            for l, v, tone in [
                ("LOS · ICU", "6.2 d", "var(--ink)"),
                ("SOFA max", "9", "var(--ink)"),
                ("Lactate max", "4.8 mmol/L", "var(--bad)"),
                ("Outcome", _T(lang, "Survived", "存活"), "var(--ok)"),
            ]
        ])
        + '</div></div>',
        unsafe_allow_html=True,
    )

    timeline_svg = cc.render_timeline(
        events=[
            (40, _T(lang, "ICU admit", "入 ICU"), "var(--ink)"),
            (120, "Sepsis-3 +", "var(--bad)"),
            (200, _T(lang, "Vent start", "上机"), "var(--warn)"),
            (380, _T(lang, "Lactate peak 4.8", "乳酸峰 4.8"), "var(--bad)"),
            (560, _T(lang, "Wean trial", "脱机试验"), "var(--warn)"),
            (700, _T(lang, "Extubation", "拔管"), "var(--ok)"),
            (880, _T(lang, "Step-down", "转下"), "var(--ok)"),
            (960, _T(lang, "Discharge", "出院"), "var(--ok)"),
        ],
    )
    st.markdown(
        '<div class="eu-card" style="padding:14px 16px;margin-top:10px">'
        f'<div style="font-size:13px;font-weight:500;margin-bottom:10px">'
        f'{_T(lang, "Timeline · 0h → 6.2d", "时间线 · 0h → 6.2d")}</div>'
        f'{timeline_svg}</div>',
        unsafe_allow_html=True,
    )

    tiles = [
        ("HR",         "12 → 92",         "bpm",     [60, 65, 80, 95, 110, 90, 88, 92]),
        ("MAP",        "88 → 71",         "mmHg",    [88, 84, 75, 65, 58, 64, 70, 71]),
        ("Lactate",    "0.9 → 4.8 → 1.4", "mmol/L",  [0.9, 1.4, 2.1, 3.5, 4.8, 3.2, 2.0, 1.4]),
        ("SOFA",       "0 → 9 → 3",       "",        [0, 2, 4, 6, 9, 7, 5, 3]),
        ("Creatinine", "0.9 → 1.6",       "mg/dL",   [0.9, 1.0, 1.2, 1.4, 1.6, 1.5, 1.4, 1.3]),
        ("UO",         "1.2 → 0.4",       "ml/kg/h", [1.2, 0.9, 0.6, 0.4, 0.5, 0.7, 1.0, 1.1]),
        ("Vent",       "+24h → 96h",      "",        [0, 0, 1, 1, 1, 1, 1, 0]),
        ("FiO₂",       "21% → 60%",       "",        [21, 35, 50, 60, 55, 40, 30, 21]),
    ]
    tiles_html = "".join(
        cc.render_sparkline_tile(label=l, value=v, unit=u, data=d)
        for l, v, u, d in tiles
    )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px">'
        + tiles_html + '</div>',
        unsafe_allow_html=True,
    )


def _render_qv_data_quality(lang: str) -> None:
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Quick Visualization · 快速可视化", "快速可视化"),
            title_en=_T(lang, "Data quality", "Data quality"),
            title_zh=_T(lang, "数据质量", "数据质量"),
            desc=_T(lang,
                "Missing rate · out-of-physio · temporal integrity.",
                "缺失率 · 超生理范围 · 时序完整性。"),
        ),
        unsafe_allow_html=True,
    )

    cards: list[tuple[str, str, str, str]] = [
        (_T(lang, "Total records", "总记录数"), "102,578", "raw events", ""),
        (_T(lang, "Weighted missing", "加权缺失率"), "8.4%",
         _T(lang, "down from 82.5% (old denom)", "较旧分母 82.5% 显著下降"), "bad"),
        (_T(lang, "Out-of-physio", "超生理范围"), "0.12%",
         _T(lang, "124 of 102,578", "124 / 102,578"), ""),
        (_T(lang, "Duplicate TS", "重复时间戳"), "0.0%",
         _T(lang, "no duplicates detected", "未检出重复"), "ok"),
    ]
    st.markdown(
        cc.render_stat_grid(cards, columns=4),
        unsafe_allow_html=True,
    )

    bars = [
        ("aki_stage_rrt",  98.4, "d=LOS"),
        ("mech_circ_supp", 96.1, "d=LOS"),
        ("ecmo",           94.8, "d=LOS"),
        ("delirium_tx",    74.2, "d=demo"),
        ("rrt_started",    52.1, "d=72h"),
        ("lactate",        18.7, "d=72h"),
        ("vent_mode",      11.4, "d=72h"),
        ("sofa_renal",      4.2, "d=demo"),
        ("hr",              0.8, "d=LOS"),
        ("map",             0.6, "d=LOS"),
    ]
    st.markdown(
        '<div class="eu-card" style="padding:14px;margin-top:14px">'
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">'
        '<div>'
        f'<div style="font-size:12.5px;font-weight:500">{_T(lang, "Missingness by concept", "按概念的缺失率")}</div>'
        f'<div style="font-size:11px;color:var(--ink-4)">'
        f'{_T(lang, "denominator: d=LOS / d=72h / d=demo / d=static", "分母:d=LOS / d=72h / d=demo / d=static")}</div>'
        '</div></div>'
        f'{cc.render_missingness_bars(bars)}'
        '<div style="margin-top:12px;font-size:11.5px;color:var(--ink-4);font-family:var(--font-mono)">'
        f'{_T(lang, "Showing 10 demo concepts · sorted by missing rate desc", "显示 10 个演示概念 · 缺失率降序")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_quickviz_redesign_page(lang: str) -> None:
    tabs_labels = (
        ["Data Tables", "Time Series", "Patient Overview", "Data Quality"]
        if lang == "en" else
        ["数据表", "时间序列", "病人全景", "数据质量"]
    )
    tabs = st.tabs(tabs_labels)
    with tabs[0]:
        _render_qv_data_tables(lang)
    with tabs[1]:
        _render_qv_time_series(lang)
    with tabs[2]:
        _render_qv_patient_overview(lang)
    with tabs[3]:
        _render_qv_data_quality(lang)


# =====================================================================
# Research Agent page
# =====================================================================


def render_agent_redesign_page(lang: str) -> None:
    actions = (
        '<span class="eu-pill"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 5v5h5"/></svg>'
        f'{_T(lang, "Runs · 7", "运行 · 7")}</span>'
    )
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Research Agent · 研究智能体", "研究智能体"),
            title_en=_T(lang, "Sepsis mortality predictors", "Sepsis mortality predictors"),
            title_zh=_T(lang, "脓毒症死亡预测因子", "脓毒症死亡预测因子"),
            desc=_T(lang,
                "Analysis-first · manuscript stays behind a review gate.",
                "先做分析,稿件锁在审阅闸门后。"),
            right_html=actions,
        ),
        unsafe_allow_html=True,
    )

    # Inbound cohort + question/plan/run row
    col_l, col_r = st.columns([0.85, 2.15], gap="medium")
    with col_l:
        st.markdown(
            '<div class="eu-card" style="padding:16px;display:flex;flex-direction:column;gap:10px">'
            '<div class="eu-section-label" style="padding:0;display:flex;justify-content:space-between">'
            f'<span>{_T(lang, "Inbound cohort", "已交付队列")}</span>'
            f'<span class="mono" style="text-transform:none;letter-spacing:0;color:var(--ink-3)">'
            f'{_T(lang, "handed off", "已交付")}</span></div>'
            '<div>'
            f'<div style="font-size:14px;font-weight:500">{_T(lang, "Demo cohort", "演示队列")}</div>'
            f'<div class="mono" style="font-size:11px;color:var(--ink-4)">{_T(lang, "demo · 2,481 stays · review concept set", "演示 · 2,481 例 · 审阅概念集")}</div>'
            '</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
            + "".join([
                '<div style="padding:6px 8px;background:var(--surface-2);border-radius:6px">'
                f'<div style="font-size:10px;color:var(--ink-4);letter-spacing:.06em;text-transform:uppercase;font-weight:500">{l}</div>'
                f'<div class="mono" style="font-size:13px;font-weight:500;color:{c}">{v}</div>'
                '</div>'
                for l, v, c in [
                    (_T(lang, "Mean age", "平均年龄"), "63.2 y", "var(--ink)"),
                    (_T(lang, "Mortality", "死亡率"), "18.0%", "var(--bad)"),
                    ("Sepsis-3", "45.3%", "var(--ink)"),
                    (_T(lang, "Mech vent", "机械通气"), "52.1%", "var(--ink)"),
                ]
            ])
            + '</div>'
            f'<div class="eu-section-label" style="padding:0;margin-top:4px"><span>{_T(lang, "Concept tray · 8 selected", "概念抽屉 · 已选 8")}</span></div>'
            '<div style="display:flex;flex-wrap:wrap;gap:4px">'
            + "".join(f'<span class="eu-chip mono">{c}</span>'
                      for c in ["vitals", "labs", "sofa", "demographics", "outcomes", "fluids", "vent", "lactate"])
            + '</div></div>',
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            '<div class="eu-card" style="padding:16px;display:flex;flex-direction:column;gap:12px">'
            '<div style="display:flex;align-items:center;justify-content:space-between">'
            '<div>'
            f'<div style="font-size:13px;font-weight:500">{_T(lang, "Research question", "研究问题")}'
            f' <span class="eu-cn" style="color:var(--ink-3);font-weight:400;margin-left:6px">'
            f'{_T(lang, "Research question", "研究问题")}</span></div>'
            f'<div style="font-size:11.5px;color:var(--ink-4)">'
            f'{_T(lang, "One sentence. The agent drafts a plan first; you confirm before any LLM call.", "一句话。Agent 会先给出计划,你确认后才会进行 LLM 调用。")}</div>'
            '</div>'
            '<span class="eu-pill">'
            '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
            '<path d="M12 2v3"/><circle cx="12" cy="12" r="6"/></svg>'
            f'{_T(lang, "gpt-oss · via sidebar AI", "gpt-oss · 通过侧栏 AI")}</span>'
            '</div>'
            '<div style="border:1px solid var(--hair-2);border-radius:10px;padding:10px 12px;background:var(--surface)">'
            f'<div style="font-size:13.5px;color:var(--ink);line-height:1.5">'
            f'{_T(lang, "Which bedside features within the first 24 hours best predict in-hospital mortality among Sepsis-3 patients, and how does adding lactate change the model’s calibration?", "在前 24 小时内,哪些床旁特征对 Sepsis-3 患者的院内死亡率有最强预测?加入 lactate 后模型的 calibration 如何变化?")}'
            '</div>'
            '<div style="margin-top:8px;display:flex;align-items:center;gap:6px">'
            '<span class="eu-chip mono">@demo_cohort</span>'
            '<span class="eu-chip mono">@first_24h</span>'
            '<span class="eu-chip mono">@lactate</span>'
            '<span class="mono" style="margin-left:auto;font-size:10.5px;color:var(--ink-4)">42 / 600 words</span>'
            '</div></div>'
            '<div>'
            f'<div class="eu-section-label" style="padding:0;margin-bottom:6px"><span>{_T(lang, "Plan preview · 6 steps", "计划预览 · 6 步")}</span></div>'
            '<div style="display:flex;flex-wrap:wrap;gap:6px">'
            + "".join(
                '<span class="eu-pill ok"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M4 12 9 17 20 6"/></svg>'
                f'{label}</span>'
                for label in (
                    [_T(lang, "Cohort summary", "队列总结"),
                     _T(lang, "Table 1", "Table 1"),
                     _T(lang, "Missingness audit", "缺失审计"),
                     "LR + SOFA + lact",
                     _T(lang, "ROC · Calibration", "ROC · Calibration"),
                     _T(lang, "Feature effects", "特征效应")]
                )
            )
            + '<span class="eu-pill" style="border-style:dashed">'
            '<svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>'
            f'{_T(lang, "Manuscript draft · requires review", "稿件草稿 · 需审阅")}</span>'
            '</div></div>'
            '<div style="background:var(--surface-2);border-radius:8px;padding:10px;display:flex;align-items:center;gap:12px">'
            '<span class="eu-pill ok"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M4 12 9 17 20 6"/></svg>'
            f'{_T(lang, "Run complete", "运行完成")}</span>'
            f'<span class="mono" style="font-size:11.5px;color:var(--ink-3)">2m 14s · 6 of 6 steps · 12,408 tokens</span>'
            '<div style="flex:1">'
            '<div style="height:3px;background:var(--hair-2);border-radius:2px;overflow:hidden">'
            '<div style="width:100%;height:100%;background:var(--ink)"></div></div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # Output gallery (5-up + 3-up findings row)
    st.markdown(
        f'<div class="eu-section-label" style="padding:0;margin:18px 0 10px">'
        f'<span>{_T(lang, "Analysis outputs", "分析产出")}</span></div>',
        unsafe_allow_html=True,
    )

    tile_summary = cc.render_output_tile(
        kind="01 · summary",
        title=_T(lang, "Cohort summary", "队列总结"),
        sub=_T(lang, "n=2,481 · 18% mortality", "n=2,481 · 死亡率 18%"),
        preview_html=(
            '<div class="mono" style="font-size:32px;font-weight:500;'
            'color:var(--ink);font-family:var(--font-mono)">2,481</div>'
        ),
        badge_html='<span class="eu-pill">view</span>',
    )
    tile_t1 = cc.render_output_tile(
        kind="02 · table",
        title="Table 1",
        sub=_T(lang, "11 features · Sepsis vs Non", "11 特征 · Sepsis vs Non"),
        preview_html=cc.render_tile_table(),
    )
    tile_miss = cc.render_output_tile(
        kind="03 · audit",
        title=_T(lang, "Missingness", "缺失分析"),
        sub=_T(lang, "weighted 8.4%", "加权 8.4%"),
        preview_html=cc.render_tile_missing(),
    )
    tile_roc = cc.render_output_tile(
        kind="04 · roc",
        title="ROC · LR + lactate",
        sub="AUC 0.842 · 95% CI 0.81–0.87",
        preview_html=cc.render_tile_roc(),
    )
    tile_cal = cc.render_output_tile(
        kind="05 · calibration",
        title=_T(lang, "Calibration", "Calibration"),
        sub="Brier 0.108",
        preview_html=cc.render_tile_calibration(),
    )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px">'
        f'{tile_summary}{tile_t1}{tile_miss}{tile_roc}{tile_cal}</div>',
        unsafe_allow_html=True,
    )

    tile_eff = cc.render_output_tile(
        kind="06 · effects",
        title=_T(lang, "Feature effects (top 5)", "特征效应 · top 5"),
        sub="lactate · sofa · age · map · creatinine",
        preview_html=cc.render_tile_feature_effects(),
    )
    findings_card = (
        '<div class="eu-card" style="grid-column:span 2;padding:16px;display:flex;gap:14px;'
        'background:var(--warn-soft);border-color:oklch(86% 0.05 75)">'
        '<div style="flex:none;color:oklch(45% 0.10 75)">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">'
        '<path d="M9 3h6"/><path d="M10 3v6L4 20a1 1 0 0 0 .9 1.5h14.2A1 1 0 0 0 20 20l-6-11V3"/></svg></div>'
        '<div style="flex:1">'
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
        f'<span style="font-size:13px;font-weight:500;color:oklch(35% 0.12 75)">{_T(lang, "Findings", "主要发现")}'
        f' <span class="eu-cn" style="font-weight:400;margin-left:6px">{_T(lang, "Findings", "主要发现")}</span></span>'
        f'<span class="eu-pill" style="background:var(--surface)">{_T(lang, "auto-drafted · review needed", "自动起草 · 需审阅")}</span>'
        '</div>'
        f'<div style="font-size:12.5px;color:oklch(28% 0.10 75);line-height:1.55">'
        f'{_T(lang, "In the Sepsis-3 cohort the strongest 24h predictors are lactate, SOFA max and age. Adding lactate raises AUC from 0.815 → 0.842 and improves Brier by 0.014; calibration remains slightly high in the 0.4–0.7 range, suggesting overestimation for mid-risk patients.", "在 Sepsis-3 队列中,前 24h lactate、SOFA max、年龄为最强预测因子。加入 lactate 后 AUC 提升 0.027(0.815 → 0.842),Brier 改善 0.014。Calibration 曲线在 0.4–0.7 区间仍偏高,提示中危人群可能高估。")}'
        '</div>'
        '<div style="display:flex;gap:6px;margin-top:10px">'
        f'<span class="eu-pill" style="background:var(--surface);height:24px">{_T(lang, "See evidence", "查看证据")}</span>'
        f'<span class="eu-pill" style="background:var(--surface);height:24px">{_T(lang, "Mark as confirmed", "标记为已确认")}</span>'
        f'<span class="eu-pill" style="background:transparent;height:24px;border-color:transparent;color:var(--ink-3)">{_T(lang, "Request re-run", "请求重跑")}</span>'
        '</div></div></div>'
    )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px">'
        f'{tile_eff}{findings_card}</div>',
        unsafe_allow_html=True,
    )

    # Review gate
    st.markdown(
        '<div class="eu-card" style="padding:14px 18px;display:flex;align-items:center;gap:14px;margin-top:18px;'
        'border-color:var(--hair-2);background:repeating-linear-gradient(90deg,transparent 0,transparent 8px,var(--surface-2) 8px,var(--surface-2) 9px)">'
        '<span class="eu-pill" style="background:var(--surface)">'
        '<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>'
        f'{_T(lang, "Review gate", "审阅闸门")}</span>'
        '<div style="flex:1">'
        f'<div style="font-size:13px;font-weight:500">{_T(lang, "Analysis ready. Generate manuscript draft?", "分析就绪,是否生成稿件草稿?")} '
        f'<span class="eu-cn" style="color:var(--ink-3);font-weight:400">{_T(lang, "Analysis ready · draft manuscript?", "分析就绪，是否生成稿件草稿？")}</span></div>'
        f'<div style="font-size:11.5px;color:var(--ink-3)">'
        f'{_T(lang, "Manuscript drafting is intentionally a second-stage action. Confirm findings above before drafting.", "起稿是刻意设置为第二阶段动作,请先确认上面的发现再继续。")}</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([8, 1.5, 2.5])
    with cols[1]:
        st.button(_T(lang, "Decline", "拒绝"), key="_eu_review_decline", use_container_width=True)
    with cols[2]:
        st.button(_T(lang, "Draft methods + results", "起草 Methods + Results"),
                  type="primary", key="_eu_review_draft", use_container_width=True)


# =====================================================================
# Entry page
# =====================================================================


def render_entry_redesign_page(lang: str) -> None:
    """Render the design's Entry / Mode selection screen.

    Replaces the legacy ``render_entry_page`` visual surface. Buttons
    still drive the real ``entry_mode`` / ``use_mock_data`` session
    state so downstream pages continue to work.
    """
    with st.container(key="eu_entry_topbar_shell"):
        brand_col, lang_col, version_col = st.columns([1, 0.08, 0.14], gap="small")
        with brand_col:
            st.markdown(
                '<div class="eu-entry-brand">'
                '<div class="eu-entry-logo">E</div>'
                '<div style="line-height:1.1">'
                '<div class="eu-entry-brand-title">EasyICU</div>'
                f'<div class="eu-entry-brand-sub">{_T(lang, "ICU data research workspace", "ICU 数据研究台")}</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        with lang_col:
            next_lang = "zh" if lang == "en" else "en"
            if st.button(
                "中" if lang == "en" else "EN",
                key="_eu_entry_lang_toggle",
                help=_T(lang, "Switch to Chinese", "切换到 English"),
                use_container_width=True,
            ):
                st.session_state["language"] = next_lang
                st.session_state["entry_lang_select"] = "ZH" if next_lang == "zh" else "EN"
                st.rerun()
        with version_col:
            st.markdown(
                '<div class="eu-entry-version mono">v1.0 · py3.10+</div>',
                unsafe_allow_html=True,
            )

    # Hero
    st.markdown(
        '<div class="eu-entry-hero" style="padding:24px 0 20px;text-align:center">'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4);letter-spacing:.08em;'
        f'text-transform:uppercase;font-family:var(--font-mono)">'
        f'{_T(lang, "Local-first ICU research workflow", "本地优先 · ICU 数据研究工作流")}</div>'
        f'<h1 style="margin:12px 0 8px;font-size:38px;font-weight:500;letter-spacing:-0.025em;color:var(--ink)">'
        f'{_T(lang, "Extract. Review. Analyze. Draft.", "数据抽取 · 审阅 · 分析 · 起草")}</h1>'
        f'<div class="eu-cn" style="font-size:15px;color:var(--ink-3)">'
        f'{_T(lang, "Extract · Review · Analyze · Draft — all in one place", "数据抽取 · 审阅 · 分析 · 起草，一站完成")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Two mode cards rendered as a single column row, then a button
    # column row immediately below — same column ratios so CTAs line
    # up under each card.
    demo_card = (
        '<div style="padding:24px 24px 18px;background:var(--accent-soft);'
        'border:1px solid var(--accent-border);border-radius:14px;display:flex;'
        'flex-direction:column;gap:12px;min-height:300px">'
        '<div style="display:flex;align-items:center;gap:10px">'
        '<div style="width:36px;height:36px;border-radius:8px;background:var(--accent);color:#fff;'
        'display:flex;align-items:center;justify-content:center">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M9 3h6"/><path d="M10 3v6L4 20a1 1 0 0 0 .9 1.5h14.2A1 1 0 0 0 20 20l-6-11V3"/></svg></div>'
        '<div>'
        f'<div style="font-size:18px;font-weight:500;letter-spacing:-0.01em">{_T(lang, "Demo Mode", "演示模式")}</div>'
        f'<div class="eu-cn" style="font-size:12px;color:var(--ink-3)">{_T(lang, "Demo Mode", "演示模式")}</div>'
        '</div>'
        f'<span class="eu-pill" style="margin-left:auto;background:var(--surface)">'
        f'{_T(lang, "try first", "新手先试")}</span>'
        '</div>'
        f'<p style="margin:0;font-size:13.5px;color:var(--ink-2);line-height:1.55">'
        f'{_T(lang, "Best for a first run: experience the full workflow with reproducible mock ICU data. No tokens, no local data, no outbound calls.", "首次使用建议先体验演示模式：用可复现的模拟 ICU 数据走完整流程。无需令牌、无需本地数据、无外部连接。")}'
        '</p>'
        '<ul style="margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:4px">'
        + "".join([
            '<li style="display:flex;gap:8px;font-size:12.5px;color:var(--ink-2)">'
            '<span style="margin-top:6px;width:4px;height:4px;background:var(--ink-4);border-radius:999px;flex:none"></span>'
            f'<span>{b}</span></li>'
            for b in [
                _T(lang, "10-patient fast review set · 24h default",
                        "10 例快速审阅集 · 默认 24 小时数据"),
                _T(lang, "Start at data extraction, then open review panels when ready",
                        "先进入数据提取，再按需打开审阅面板"),
                _T(lang, "Research Agent static gallery viewable",
                        "研究智能体静态输出画廊可查看"),
                _T(lang, "Switch to real data anytime without losing work",
                        "随时可切换到真实数据,不丢工作"),
            ]
        ])
        + '</ul></div>'
    )
    real_card = (
        '<div style="padding:24px 24px 18px;background:var(--surface);'
        'border:1px solid var(--hair);border-radius:14px;display:flex;'
        'flex-direction:column;gap:12px;min-height:300px">'
        '<div style="display:flex;align-items:center;gap:10px">'
        '<div style="width:36px;height:36px;border-radius:8px;background:var(--ink);color:#fff;'
        'display:flex;align-items:center;justify-content:center">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v6c0 1.7 4 3 9 3s9-1.3 9-3V5"/>'
        '<path d="M3 11v6c0 1.7 4 3 9 3s9-1.3 9-3v-6"/></svg></div>'
        '<div>'
        f'<div style="font-size:18px;font-weight:500;letter-spacing:-0.01em">{_T(lang, "Real Data", "真实数据")}</div>'
        f'<div class="eu-cn" style="font-size:12px;color:var(--ink-3)">{_T(lang, "Real Data", "真实数据")}</div>'
        '</div>'
        '<span class="eu-pill ok" style="margin-left:auto"><span class="dot"></span>local-only</span>'
        '</div>'
        f'<p style="margin:0;font-size:13.5px;color:var(--ink-2);line-height:1.55">'
        f'{_T(lang, "Connect to local ICU exports. Everything is processed on your machine — EasyICU never uploads anything.", "连接本机的 ICU 数据库导出。所有处理都在你的机器上完成,EasyICU 不会上传或外发任何数据。")}'
        '</p>'
        '<ul style="margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:4px">'
        + "".join([
            '<li style="display:flex;gap:8px;font-size:12.5px;color:var(--ink-2)">'
            '<span style="margin-top:6px;width:4px;height:4px;background:var(--ink-4);border-radius:999px;flex:none"></span>'
            f'<span>{b}</span></li>'
            for b in [
                _T(lang, "6 databases · MIMIC-IV / eICU / AUMC / HiRID / MIMIC-III / SICdb",
                        "6 大数据库 · MIMIC-IV / eICU / AUMC / HiRID / MIMIC-III / SICdb"),
                _T(lang, "Choose a local export folder; EasyICU detects known layouts",
                        "选择本地导出目录；EasyICU 自动识别常见结构"),
                _T(lang, "Module-folder mode reuses prior exports",
                        "模块文件夹模式可复用之前的导出"),
                _T(lang, "Cross-DB Benchmark connects ≥2 databases",
                        "跨数据库对比可同时连接 ≥ 2 个数据库"),
            ]
        ])
        + '</ul></div>'
    )

    # Mode cards row (HTML) — two equal columns
    col_demo, col_real = st.columns(2, gap="medium")
    with col_demo:
        st.markdown(demo_card, unsafe_allow_html=True)
        if st.button(_T(lang, "Start demo →", "开始演示 →"),
                     key="_eu_entry_demo", type="primary", use_container_width=True):
            _route_to_extract_entry_mode(st.session_state, "demo")
            st.rerun()
    with col_real:
        st.markdown(real_card, unsafe_allow_html=True)
        if st.button(_T(lang, "Use local data folder →", "使用本地数据目录 →"),
                     key="_eu_entry_real", use_container_width=True):
            _route_to_extract_entry_mode(st.session_state, "real")
            st.rerun()

    # Code-only is a real third action, but visually stays secondary.
    with st.container(key="eu_entry_code_row"):
        code_copy_col, code_action_col = st.columns([5, 2], gap="medium")
        with code_copy_col:
            st.markdown(
                '<div class="eu-entry-nodata" style="display:flex;align-items:center;gap:14px;padding:12px 18px;'
                'background:var(--surface);border:1px dashed var(--hair-3);border-radius:12px">'
                '<div style="color:var(--ink-3)">'
                '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>'
                '<path d="M14 3v6h6"/></svg></div>'
                '<div style="flex:1">'
                f'<div style="font-size:13px;font-weight:500">{_T(lang, "No data yet?", "还没有数据?")}'
                f' <span class="eu-cn" style="color:var(--ink-3);font-weight:400;margin-left:6px">'
                f'{_T(lang, "No data yet?", "还没有数据？")}</span></div>'
                f'<div style="font-size:12px;color:var(--ink-3)">'
                f'{_T(lang, "Let the Research Agent generate a reusable code skeleton, then plug data in later.", "让研究智能体先生成可复用代码骨架，稍后再接入真实数据。")}</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        with code_action_col:
            if st.button(_T(lang, "Generate code only →", "仅生成代码 →"),
                         key="_eu_entry_nodata", use_container_width=True):
                st.session_state["entry_mode"] = "demo"
                st.session_state["_active_main_page"] = "research_agent"
                st.rerun()

    st.markdown(
        '<div class="eu-entry-next">'
        '<div class="eu-entry-next-head">'
        f'<span>{_T(lang, "After you choose a mode", "选择入口后")}</span>'
        f'<b>{_T(lang, "A quieter review path follows", "后续进入轻量审阅链路")}</b>'
        '</div>'
        '<div class="eu-entry-rail">'
        f'<div class="eu-entry-step"><small>{_T(lang, "01", "01")}</small><b>{_T(lang, "Data gate", "数据闸门")}</b><span>{_T(lang, "Normalize demo or local exports before analysis.", "先标准化演示数据或本地导出。")}</span></div>'
        f'<div class="eu-entry-step"><small>{_T(lang, "02", "02")}</small><b>{_T(lang, "Cohort review", "队列审阅")}</b><span>{_T(lang, "Confirm filters, time windows, and concept coverage.", "确认筛选、时间窗和概念覆盖。")}</span></div>'
        f'<div class="eu-entry-step"><small>{_T(lang, "03", "03")}</small><b>{_T(lang, "Quality checks", "质量检查")}</b><span>{_T(lang, "Inspect missingness, ranges, density, and audit notes.", "检查缺失、范围、密度和审计提示。")}</span></div>'
        f'<div class="eu-entry-step"><small>{_T(lang, "04", "04")}</small><b>{_T(lang, "Export handoff", "导出交付")}</b><span>{_T(lang, "Package code, tables, figures, and evidence ledger.", "打包代码、表格、图件和证据账本。")}</span></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
