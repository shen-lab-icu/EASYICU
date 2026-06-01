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
import json
import platform
import sys
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from easyicu.webapp import cohort_charts as cc
from easyicu.webapp.concept_catalog import (
    CONCEPT_DESCRIPTIONS,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_INTERNAL,
)
from easyicu.webapp.session_state import clear_agent_continuation_state


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
    state.pop("_export_failure_result", None)
    state.pop("_scroll_to_tab", None)
    state["_active_main_page"] = "extract"


def _route_to_research_agent_setup(
    state: MutableMapping[str, Any],
    *,
    force_real: bool = False,
    focus_module_folder: bool = False,
) -> None:
    """Open a fresh Agent setup without leaking an older resume/draft mode."""
    clear_agent_continuation_state(state)
    if force_real:
        state["entry_mode"] = "real"
        state["use_mock_data"] = False
        valid_real_databases = {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}
        if state.get("database") not in valid_real_databases:
            state["database"] = "miiv"
            state["path_validated"] = False
            state.pop("last_validated_path", None)
    if focus_module_folder:
        state["_eu_ra_focus_module_folder"] = True
        state.pop("research_agent_module_dir_pick", None)
    else:
        state.pop("_eu_ra_focus_module_folder", None)
    state.pop("_eu_ra_focus_no_data", None)
    state.pop("_eu_ra_no_data_entry", None)
    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "setup"
    state["_scroll_to_top"] = True


def _route_to_research_agent_no_data_setup(state: MutableMapping[str, Any]) -> None:
    """Open Agent setup on the real no-data/extraction path.

    The entry-page no-data CTA should not land on the demo guide: the backend
    path users need is the no-data cohort source, where they choose a database,
    modules, and an output folder before launching extraction.
    """
    _route_to_research_agent_setup(state, force_real=True)
    state["_eu_ra_focus_no_data"] = True
    state["_eu_ra_no_data_entry"] = True
    state.pop("research_agent_cohort_source", None)
    lang = str(state.get("language") or "en")
    from easyicu.webapp.research_agent import _seed_default_research_agent_question

    _seed_default_research_agent_question(state, is_en=lang == "en")


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
        _route_to_research_agent_setup(state)
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
            "Automatically generates reproducible mock data. Full cohort-builder, Quick Viz, and Research Agent setup/history preview. No tokens, no local data needed.",
            "自动生成可重复的模拟数据。完整体验队列构建、患者审阅和研究智能体设置/历史预览。无需令牌、无需本地数据。"),
        bullets=[
            _T(lang, "10-patient fast review set · 24h default",
                    "10 例快速审阅集 · 默认 24 小时"),
            _T(lang, "Lightweight review data is ready immediately",
                    "轻量审阅数据可立即打开"),
            _T(lang, "Research Agent setup and local-run handoff preview",
                    "研究智能体设置与本机 run 交接预览"),
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
        title_en=_T(lang, "No Data Yet", "No Data Yet"),
        title_zh=_T(lang, "先准备提取", "先准备提取"),
        badge_html="",
        desc=_T(lang,
            "Prepare extraction settings first; plug in real data when the folder is ready.",
            "先准备提取设置；等真实数据目录就绪后再接入。"),
        bullets=[
            _T(lang, "Choose database and modules", "选择数据库和模块"),
            _T(lang, "Hand settings to Data Extraction", "交给数据提取执行"),
        ],
        cta_label=_T(lang, "Prepare extraction", "准备提取"),
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
        if st.button(_T(lang, "Prepare extraction → Agent", "准备提取 → Agent"),
                     key="_eu_tutorial_nodata",
                     use_container_width=True):
            _route_to_research_agent_no_data_setup(st.session_state)
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


def _route_to_ai_assistant(state: MutableMapping[str, Any], question: str | None = None) -> None:
    """Route to the standalone assistant page without introducing a new backend path."""
    state["_active_main_page"] = "assistant"
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state["_sidebar_ai_open"] = False
    state["_scroll_to_top"] = True
    if question:
        state["_ai_pending_question"] = question


def _route_to_workspace_states(state: MutableMapping[str, Any]) -> None:
    state["_active_main_page"] = "states"
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state.pop("_ai_pending_question", None)
    state["_scroll_to_top"] = True


def _workspace_state_contexts(lang: str) -> list[dict[str, str]]:
    return [
        {"key": "patient", "label": _T(lang, "Patient Review", "患者审阅"), "crumb": _T(lang, "Data Visualization · Patient Review", "数据可视化 · 患者审阅"), "title": _T(lang, "Patient Review", "患者审阅")},
        {"key": "crossdb", "label": _T(lang, "Cross-DB Benchmark", "跨库基准"), "crumb": _T(lang, "Data Visualization · Cross-DB Benchmark", "数据可视化 · 跨库基准"), "title": _T(lang, "Cross-DB Benchmark", "跨库基准")},
        {"key": "agent", "label": _T(lang, "Research Agent", "研究智能体"), "crumb": _T(lang, "Research Agent · Run", "研究智能体 · 运行"), "title": _T(lang, "Research Agent", "研究智能体")},
    ]


def _workspace_state_modes(lang: str) -> list[dict[str, str]]:
    return [
        {"key": "demo", "label": _T(lang, "Demo", "演示")},
        {"key": "real", "label": _T(lang, "Real Data", "真实数据")},
    ]


def _workspace_state_options(lang: str) -> list[dict[str, str]]:
    return [
        {"key": "loading", "label": _T(lang, "Loading", "加载中")},
        {"key": "empty", "label": _T(lang, "Empty", "空状态")},
        {"key": "nodata", "label": _T(lang, "No data", "无数据")},
        {"key": "error", "label": _T(lang, "Error", "错误")},
        {"key": "blocked", "label": _T(lang, "Blocked", "阻断")},
        {"key": "success", "label": _T(lang, "Success", "成功")},
    ]


def _workspace_state_copy(context: str, mode: str, state_key: str, lang: str) -> dict[str, object]:
    is_demo = mode == "demo"
    if context == "crossdb":
        return {
            "loading": _T(lang, "Loading seeded frames for 6 databases", "正在加载 6 个数据库的种子数据") if is_demo else _T(lang, "Connecting to selected ICU databases", "正在连接所选 ICU 数据库"),
            "empty_title": _T(lang, "Select at least 2 databases", "至少选择 2 个数据库"),
            "empty_detail": _T(lang, "Compare one cohort definition across databases. Add a second source to begin.", "跨数据库比较同一个队列定义。至少添加第二个来源后开始。") if is_demo else _T(lang, "Connect two or more local database roots to compare standardized concepts side by side.", "连接两个或更多本地数据库根目录后并排比较标准化概念。"),
            "empty_chips": ["MIMIC-IV", "eICU-CRD", "AUMC", "HiRID", "SICdb"],
            "empty_action": _T(lang, "Load demo databases", "加载演示数据库") if is_demo else _T(lang, "Connect databases", "连接数据库"),
            "nodata_title": _T(lang, "No shared concepts across selection", "所选数据库没有共同概念"),
            "nodata_detail": _T(lang, "Adjust the database selection or concept set, then re-run the comparison.", "调整数据库选择或概念集后重新运行比较。"),
            "nodata_filters": ["MIMIC-IV", "SICdb", "concept = lactate", "window = 6h"],
            "error_title": _T(lang, "Couldn't assemble benchmark", "无法组装基准比较") if is_demo else _T(lang, "Database connection failed", "数据库连接失败"),
            "error_detail": _T(lang, "One selected source is unreadable or its concept map is missing.", "某个已选来源不可读，或缺少概念映射。"),
            "error_lines": ["$ benchmark --dbs selected", "SchemaError: concepts/ not found", "expected: <root>/concepts/*.parquet"],
            "blocked_title": _T(lang, "Export is locked until evidence checks pass", "证据检查通过前导出保持锁定"),
            "success_title": _T(lang, "Benchmark assembled", "基准比较已组装"),
            "success_stats": [(_T(lang, "Databases", "数据库"), "6"), (_T(lang, "Concepts", "概念"), "6"), (_T(lang, "Delta range", "差异范围"), "15.5"), (_T(lang, "Rows / DB", "每库行数"), "144")],
            "success_rows": [("hr median", "76.6", "80.3", "74.1"), ("sbp median", "125.4", "128.5", "119.9"), ("map median", "85.3", "89.6", "83.2")],
        }
    if context == "agent":
        return {
            "loading": _T(lang, "Running demo pipeline (no tokens)", "正在运行演示 pipeline（无 token）") if is_demo else _T(lang, "Executing plan · evidence-bound run", "正在执行计划 · 证据绑定运行"),
            "empty_title": _T(lang, "No run yet", "尚未运行"),
            "empty_detail": _T(lang, "Confirm a plan to preview the evidence-bound workflow.", "确认计划后预览证据绑定工作流。") if is_demo else _T(lang, "Define a research question and confirm the preflight gate before any model call.", "先定义研究问题并确认 preflight 关口，再进行任何模型调用。"),
            "empty_chips": ["plan", "build", "analyze", "gate", "review"],
            "empty_action": _T(lang, "Preview demo run", "预览演示运行") if is_demo else _T(lang, "Open plan setup", "打开计划配置"),
            "nodata_title": _T(lang, "Run produced no evidence artifacts", "运行没有产出证据产物"),
            "nodata_detail": _T(lang, "Every step completed but no artifact passed the evidence contract.", "步骤已结束，但没有产物通过证据契约。"),
            "nodata_filters": ["cohort = sepsis", "checks = coverage", "gate = strict"],
            "error_title": _T(lang, "Demo pipeline halted", "演示 pipeline 已停止") if is_demo else _T(lang, "Run failed at analysis step", "分析步骤运行失败"),
            "error_detail": _T(lang, "The draft stays locked; the run remains recoverable.", "草稿保持锁定；该运行仍可恢复。"),
            "error_lines": ["step 04 · LR + SOFA + lactate", "LinAlgError: singular matrix", "evidence ledger: partial"],
            "blocked_title": _T(lang, "Manuscript draft is locked until checks pass", "证据检查通过前锁定手稿草稿"),
            "success_title": _T(lang, "Run complete · awaiting review", "运行完成 · 等待复核"),
            "success_stats": [(_T(lang, "Steps", "步骤"), "6 / 6"), (_T(lang, "Figures", "图件"), "6"), (_T(lang, "Tables", "表格"), "3"), (_T(lang, "Duration", "时长"), "2m 14s")],
            "success_rows": [("Cohort summary", "n=10 · 20% mortality", "done"), ("Table 1", "11 features", "done"), ("ROC · LR + lactate", "AUC 0.84", "done")],
        }
    return {
        "loading": _T(lang, "Generating demo review data", "正在生成演示审阅数据") if is_demo else _T(lang, "Reading local export folder", "正在读取本地导出目录"),
        "empty_title": _T(lang, "No review workspace loaded yet", "尚未加载审阅工作区"),
        "empty_detail": _T(lang, "Generate a compact demo set to populate tables, time series, patient overview, and quality checks.", "生成一组紧凑演示数据来填充表格、时间序列、患者概览和质控。") if is_demo else _T(lang, "Point EasyICU at a local export folder. Files are parsed on your machine.", "将 EasyICU 指向本地导出目录；文件只在本机解析。"),
        "empty_chips": ["vitals", "labs", "sofa", "sepsis-3", "outcomes"],
        "empty_action": _T(lang, "Generate demo data", "生成演示数据") if is_demo else _T(lang, "Choose export folder", "选择导出目录"),
        "nodata_title": _T(lang, "Cohort matched 0 stays", "当前队列匹配 0 个 stay"),
        "nodata_detail": _T(lang, "Loosen a constraint or widen the time window, then re-run.", "放宽约束或扩大时间窗口后重新运行。"),
        "nodata_filters": ["age >= 80", "sepsis-3 = true", "LOS > 14d", "vasopressor = yes"],
        "error_title": _T(lang, "Demo generation failed", "演示数据生成失败") if is_demo else _T(lang, "Couldn't read the export folder", "无法读取导出目录"),
        "error_detail": _T(lang, "Retry rebuilds the review workspace from the selected source.", "重试会从所选来源重建审阅工作区。"),
        "error_lines": ["$ easyicu demo --seed 42 --patients 10", "ValueError: frame length 0", "hint: retry regenerates the seed"],
        "blocked_title": _T(lang, "Export is locked until evidence checks pass", "证据检查通过前导出保持锁定"),
        "success_title": _T(lang, "Review workspace loaded", "审阅工作区已加载"),
        "success_stats": [(_T(lang, "Stays", "Stay"), "10"), (_T(lang, "Time points", "时间点"), "240"), (_T(lang, "Modules", "模块"), "19"), (_T(lang, "Coverage", "覆盖率"), "94%")],
        "success_rows": [("Age, mean (SD)", "54.8 (16.2)", "-"), ("SOFA, median", "6", "0.08"), ("Lactate, mmol/L", "2.4", "0.12")],
    }


def _workspace_state_preview_html(context: str, mode: str, state_key: str, lang: str) -> str:
    ctx = next(item for item in _workspace_state_contexts(lang) if item["key"] == context)
    mode_label = next(item["label"] for item in _workspace_state_modes(lang) if item["key"] == mode)
    state_label = next(item["label"] for item in _workspace_state_options(lang) if item["key"] == state_key)
    copy = _workspace_state_copy(context, mode, state_key, lang)
    state_tone = "ok" if state_key == "success" else "warn" if state_key == "blocked" else "bad" if state_key == "error" else ""

    if state_key == "loading":
        body = (
            '<div class="eu-state-loading-row"><span class="eu-state-spinner"></span>'
            f'<div><b>{_esc(copy["loading"])}...</b><p>{_T(lang, "reproducible · no outbound calls", "可复现 · 无外部调用") if mode == "demo" else _T(lang, "local-only · nothing uploaded", "本地优先 · 不上传")}</p></div></div>'
            '<div class="eu-state-progress"><span></span></div>'
            '<div class="eu-state-skel-grid">'
            + ''.join('<div class="eu-state-skel-card"><span></span><b></b></div>' for _ in range(4))
            + '</div>'
        )
    elif state_key == "empty":
        chips = ''.join(f'<span class="eu-state-chip">{_esc(chip)}</span>' for chip in copy["empty_chips"])
        body = f'<div class="eu-state-body-pad"><div class="eu-state-hero empty-state"><div class="glyph"></div><div class="st-t">{_esc(copy["empty_title"])}</div><div class="st-d">{_esc(copy["empty_detail"])}</div><div class="filter-recap">{chips}</div></div></div>'
    elif state_key == "nodata":
        chips = ''.join(f'<span class="eu-state-chip solid">{_esc(chip)}</span>' for chip in copy["nodata_filters"])
        body = f'<div class="eu-state-body-pad"><div class="eu-state-hero nodata"><div class="glyph"></div><div class="st-t">{_esc(copy["nodata_title"])}</div><div class="st-d">{_esc(copy["nodata_detail"])}</div><div class="filter-recap">{chips}</div></div></div>'
    elif state_key == "error":
        lines = ''.join(
            f'<span class="{ "ln-bad" if idx == 1 else "ln-key" if idx == 0 else ""}">{_esc(line)}</span>\n'
            for idx, line in enumerate(copy["error_lines"])
        )
        body = f'<div class="eu-state-body-pad"><div class="eu-state-hero error solid"><div class="glyph"></div><div class="st-t">{_esc(copy["error_title"])}</div><div class="st-d">{_esc(copy["error_detail"])}</div><div class="detail-box">{lines}</div></div></div>'
    elif state_key == "blocked":
        checks = [
            (_T(lang, "Cohort denominators resolved", "队列分母已确认"), True),
            (_T(lang, "Evidence manifest attached", "证据 manifest 已绑定"), True),
            (_T(lang, "Tables and figures registered", "表格与图件已注册"), True),
            (_T(lang, "Reviewer sign-off", "审核者签字"), False),
        ]
        body = (
            '<div class="eu-state-body-pad"><div class="gate-block">'
            f'<div class="eu-state-callout warn">{_esc(copy["blocked_title"])}</div>'
            '<div class="checks">'
            + ''.join(
                f'<div class="check-row {"ok" if ok else "pending"}"><span class="check-mk"></span><span>{_esc(label)}</span><span class="grow"></span><em>{_T(lang, "passed", "通过") if ok else _T(lang, "pending", "待确认")}</em></div>'
                for label, ok in checks
            )
            + '</div></div></div>'
        )
    else:
        stats = ''.join(
            f'<div class="stat { "ok" if idx == 0 else "accent"}"><div class="label">{_esc(label)}</div><div class="val">{_esc(value)}</div></div>'
            for idx, (label, value) in enumerate(copy["success_stats"])
        )
        rows = ''.join('<tr>' + ''.join(f'<td>{_esc(cell)}</td>' for cell in row) + '</tr>' for row in copy["success_rows"])
        body = (
            '<div class="eu-state-body-pad">'
            f'<div class="ok-banner"><span class="mk"></span><div><strong>{_esc(copy["success_title"])}.</strong> <span>{_T(lang, "Seeded demo output - values are illustrative, not a real run.", "种子演示输出，仅用于界面说明，不是真实结果。") if mode == "demo" else _T(lang, "Local run - results stayed on your machine.", "本地运行；结果保留在本机。")}</span></div></div>'
            f'<div class="st-stats">{stats}</div><div class="table-wrap"><table class="eu-table"><tbody>{rows}</tbody></table></div></div>'
        )

    return (
        '<div class="eu-state-preview-card">'
        '<div class="eu-state-preview-head">'
        '<div class="eu-state-preview-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h10"/></svg></div>'
        f'<div><h2>{_esc(ctx["title"])}</h2><p>{_esc(ctx["crumb"])}</p></div>'
        '<div class="eu-state-pills">'
        f'<span class="eu-pill {mode}"><span class="dot"></span>{_esc(mode_label)}</span>'
        f'<span class="eu-pill {state_tone}"><span class="dot"></span>{_esc(state_label)}</span>'
        '</div></div>'
        f'{body}</div>'
    )


def _apply_workspace_state_action(state: MutableMapping[str, Any], context: str, mode: str) -> None:
    if mode == "demo":
        _apply_demo_defaults_for_tutorial(state)
    else:
        state["entry_mode"] = "real"
        state["use_mock_data"] = False
        if state.get("database") not in {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}:
            state["database"] = "miiv"
    if context == "patient":
        state["_active_main_page"] = "quick_viz"
        state["quick_viz_active_panel"] = "data_tables"
        if mode == "demo":
            state["_eu_topbar_run_request"] = {"page": "quick_viz", "requested_at": "workspace_states"}
    elif context == "crossdb":
        state["_active_main_page"] = "cross_db"
        if mode == "demo":
            state["_eu_topbar_run_request"] = {"page": "cross_db", "requested_at": "workspace_states"}
    else:
        state["_active_main_page"] = "research_agent"
        state["_ra_view"] = "setup"
    state["_scroll_to_top"] = True


def _workspace_state_action_label(context: str, mode: str, lang: str) -> str:
    if context == "crossdb":
        return (
            _T(lang, "Open Cross-DB demo", "打开跨库演示")
            if mode == "demo"
            else _T(lang, "Open Cross-DB setup", "打开跨库配置")
        )
    if context == "agent":
        return (
            _T(lang, "Preview Research Agent demo", "预览研究智能体演示")
            if mode == "demo"
            else _T(lang, "Open Research Agent setup", "打开研究智能体配置")
        )
    return (
        _T(lang, "Open Patient Review demo", "打开患者审阅演示")
        if mode == "demo"
        else _T(lang, "Open Patient Review", "打开患者审阅")
    )


def _workspace_status_mode(state: MutableMapping[str, Any], lang: str) -> tuple[str, str, bool]:
    entry_mode = str(state.get("entry_mode") or ("demo" if state.get("use_mock_data", True) else "real"))
    is_demo = entry_mode != "real" and bool(state.get("use_mock_data", True))
    db_labels = {
        "mock": _T(lang, "Demo mock ICU", "演示模拟 ICU"),
        "miiv": "MIMIC-IV",
        "eicu": "eICU-CRD",
        "aumc": "AmsterdamUMCdb",
        "hirid": "HiRID",
        "mimic": "MIMIC-III",
        "sic": "SICdb",
    }
    database = str(state.get("database") or ("mock" if is_demo else "miiv"))
    mode_label = _T(lang, "Demo", "演示") if is_demo else _T(lang, "Real data", "真实数据")
    return mode_label, db_labels.get(database, database.upper()), is_demo


def _workspace_status_extract_step(state: MutableMapping[str, Any], lang: str) -> dict[str, object]:
    done = [
        bool(state.get("step1_confirmed")),
        bool(state.get("step2_confirmed")),
        bool(state.get("step3_confirmed")),
        bool(state.get("export_completed")),
    ]
    labels = [
        _T(lang, "Data source", "数据源"),
        _T(lang, "Cohort", "队列"),
        _T(lang, "Concepts", "概念变量"),
        _T(lang, "Export", "导出"),
    ]
    hints = [
        _T(lang, "Choose demo or local data.", "选择演示或本地数据。"),
        _T(lang, "Confirm inclusion and filters.", "确认纳排与筛选。"),
        _T(lang, "Select modules and concepts.", "选择模块和变量。"),
        _T(lang, "Package tables and manifest.", "打包表格与 manifest。"),
    ]
    completed = sum(1 for item in done if item)
    active_index = min(completed, 3)
    return {
        "completed": completed,
        "labels": labels,
        "hints": hints,
        "done": done,
        "active": active_index,
        "current": labels[active_index],
    }


def _workspace_status_count(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))  # type: ignore[arg-type]
    except Exception:
        return 0


def _workspace_status_overview_html(state: MutableMapping[str, Any], lang: str) -> str:
    mode_label, database_label, is_demo = _workspace_status_mode(state, lang)
    extract = _workspace_status_extract_step(state, lang)
    loaded_concepts = state.get("loaded_concepts")
    selected_concepts = state.get("selected_concepts")
    patient_ids = state.get("patient_ids")
    cohort_stats = state.get("_cohort_stats") if isinstance(state.get("_cohort_stats"), dict) else {}
    exported = bool(state.get("export_completed"))
    loaded_count = _workspace_status_count(loaded_concepts) if isinstance(loaded_concepts, dict) else 0
    selected_count = _workspace_status_count(selected_concepts)
    patient_count = _workspace_status_count(patient_ids)
    if not patient_count and isinstance(cohort_stats, dict):
        patient_count = int(cohort_stats.get("after") or cohort_stats.get("before") or 0)
    if not patient_count and is_demo:
        patient_count = int(state.get("demo_mode_patients") or (state.get("mock_params") or {}).get("n_patients") or 10)

    extract_detail = (
        _T(lang, "Export complete", "导出完成")
        if exported else
        _T(lang, "Continue the four-step extraction flow", "继续四步数据提取流程")
    )
    review_value = (
        f"{patient_count:,} / {loaded_count:,}"
        if loaded_count else
        _T(lang, "Not loaded", "未加载")
    )
    review_detail = (
        _T(lang, "patients / loaded concepts", "患者 / 已加载变量")
        if loaded_count else
        _T(lang, "Open Patient Review or Cohort Statistics when data is ready.", "数据准备好后进入患者审阅或队列统计。")
    )
    agent_question = str(state.get("research_agent_question") or "").strip()
    agent_run = str(state.get("research_agent_last_run_id") or state.get("research_agent_resume_run_id") or "").strip()
    agent_value = (
        _T(lang, "Run ready", "已有运行")
        if agent_run else
        (_T(lang, "Question ready", "问题已准备") if agent_question else _T(lang, "Setup needed", "需要配置"))
    )
    agent_detail = (
        agent_run
        if agent_run else
        (_T(lang, "Research question is staged.", "研究问题已暂存。") if agent_question else _T(lang, "Define question, cohort, and launch gate.", "配置研究问题、队列和启动闸门。"))
    )

    tiles = [
        (
            _T(lang, "Data mode", "数据模式"),
            mode_label,
            database_label,
            "ok" if is_demo else "accent",
        ),
        (
            _T(lang, "Extraction", "数据提取"),
            f"{extract['completed']} / 4",
            extract_detail,
            "ok" if exported else "accent",
        ),
        (
            _T(lang, "Review workspace", "审阅工作区"),
            review_value,
            review_detail,
            "ok" if loaded_count else "neutral",
        ),
        (
            _T(lang, "Research Agent", "研究智能体"),
            agent_value,
            agent_detail,
            "ok" if agent_run else "neutral",
        ),
    ]
    tiles_html = "".join(
        '<div class="eu-workspace-status-tile {tone}">'
        '<span>{label}</span><b>{value}</b><p>{detail}</p></div>'.format(
            tone=_esc(tone),
            label=_esc(label),
            value=_esc(value),
            detail=_esc(detail),
        )
        for label, value, detail, tone in tiles
    )

    steps_html = ""
    done = extract["done"]
    labels = extract["labels"]
    hints = extract["hints"]
    active = int(extract["active"])
    for idx, label in enumerate(labels):
        cls = "done" if done[idx] else ("active" if idx == active else "pending")
        steps_html += (
            f'<div class="eu-workspace-flow-step {cls}">'
            f'<i>{idx + 1}</i><div><b>{_esc(label)}</b><span>{_esc(hints[idx])}</span></div></div>'
        )

    selected_detail = (
        f"{selected_count:,} " + (_T(lang, "selected concepts", "个已选变量"))
        if selected_count else
        _T(lang, "No concept selection yet", "尚未选择变量")
    )
    export_path = str(state.get("last_export_dir") or state.get("export_path") or "")
    handoff_rows = [
        (_T(lang, "Current source", "当前来源"), f"{mode_label} · {database_label}"),
        (_T(lang, "Selection", "变量选择"), selected_detail),
        (_T(lang, "Export folder", "导出目录"), export_path or _T(lang, "Not set", "未设置")),
        (_T(lang, "Safety", "安全边界"), _T(lang, "This page does not generate data or call models.", "本页不生成数据，也不会调用模型。")),
    ]
    handoff_html = "".join(
        f'<div><span>{_esc(label)}</span><b>{_esc(value)}</b></div>'
        for label, value in handoff_rows
    )

    next_copy = (
        _T(lang, "Finish export, then open Patient Review or hand the module folder to the Research Agent.", "先完成导出，再进入患者审阅，或把模块目录交给研究智能体。")
        if not exported else
        _T(lang, "Export is complete. You can review patients, inspect cohort statistics, or open the Research Agent with the exported module folder.", "导出已完成。现在可以审阅患者、查看队列统计，或用导出的模块目录打开研究智能体。")
    )

    return (
        '<div class="eu-workspace-status-grid">'
        f'{tiles_html}'
        '</div>'
        '<div class="eu-workspace-overview-card">'
        '<div class="eu-workspace-card-head">'
        f'<div><span>{_T(lang, "Workflow", "工作流")}</span><h2>{_T(lang, "Where the current session stands", "当前会话进展")}</h2></div>'
        f'<em>{_esc(next_copy)}</em>'
        '</div>'
        f'<div class="eu-workspace-flow">{steps_html}</div>'
        '</div>'
        '<div class="eu-workspace-overview-card compact">'
        '<div class="eu-workspace-card-head">'
        f'<div><span>{_T(lang, "Handoff", "交接")}</span><h2>{_T(lang, "What downstream pages will use", "下游页面会使用什么")}</h2></div>'
        '</div>'
        f'<div class="eu-workspace-handoff">{handoff_html}</div>'
        '</div>'
    )


def _route_to_extract_step(
    state: MutableMapping[str, Any],
    step: int,
) -> None:
    """Route tutorial actions to a concrete extraction step."""
    step = max(1, min(4, int(step)))
    state["_active_main_page"] = "extract"
    state["step1_confirmed"] = step > 1
    state["step2_confirmed"] = step > 2
    state["step3_confirmed"] = step > 3
    if step < 4:
        state["export_completed"] = False
        state["trigger_export"] = False
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state["_scroll_to_top"] = True


def _guide_step_html(number: str, title: str, desc: str) -> str:
    return (
        '<div class="eu-guide-step-copy">'
        f'<div class="eu-guide-step-num">{_esc(number)}</div>'
        '<div class="eu-guide-step-body">'
        f'<div class="eu-guide-step-title" role="heading" aria-level="3">{_esc(title)}</div>'
        f'<p>{_esc(desc)}</p>'
        '</div>'
        '</div>'
    )


def render_tutorial_redesign_page(lang: str) -> None:
    """Render the latest print-reference Get Started page."""
    is_en = lang == "en"
    st.markdown(
        '<div class="eu-getstarted-head">'
        f'<div class="eyebrow">{_T(lang, "Get started · 快速上手", "快速上手")}</div>'
        f'<h1>{_T(lang, "A quiet, reviewable path from data to draft", "从数据到草稿的可复核路径")}</h1>'
        f'<p>{_T(lang, "EasyICU runs entirely on your machine. Start with reproducible demo data to learn the flow, then point it at local ICU exports when you’re ready.", "EasyICU 完全在本机运行。先用可复现演示数据熟悉流程，准备好后再连接本地 ICU 导出。")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


    with st.container(key="eu_getstarted_demo_tour"):
        copy_col, action_col = st.columns([1, 0.22], gap="large")
        with copy_col:
            st.markdown(
                '<div class="eu-guide-hero-copy">'
                '<div class="eu-guide-mark" aria-hidden="true">'
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M9 3h6"/><path d="M10 3v6L4 20a1 1 0 0 0 .9 1.5h14.2A1 1 0 0 0 20 20l-6-11V3"/></svg>'
                '</div>'
                '<div>'
                f'<h2>{_T(lang, "New here? Take the 2-minute demo tour", "第一次使用？先走 2 分钟演示")}</h2>'
                f'<p>{_T(lang, "No tokens, no setup, no patient data. The demo generates 10 mock ICU stays so every screen, table, and review gate is fully explorable before you connect anything real.", "无需 token、无需配置、无需患者数据。演示会生成 10 个模拟 ICU stays，因此在连接真实数据前，每个页面、表格和复核关口都可以完整探索。")}</p>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with action_col:
            if st.button(
                _T(lang, "Start demo", "开始演示"),
                key="_eu_getstarted_start_demo",
                type="primary",
                icon=":material/play_arrow:",
                use_container_width=True,
            ):
                _route_to_extract_entry_mode(st.session_state, "demo")
                st.rerun()
            if st.button(
                _T(lang, "Browse states", "浏览状态"),
                key="_eu_getstarted_browse_states",
                icon=":material/grid_view:",
                use_container_width=True,
            ):
                _route_to_workspace_states(st.session_state)
                st.rerun()

    st.markdown(
        '<div class="eu-guide-section">'
        f'<div class="eyebrow">{_T(lang, "Five steps", "五步")}</div>'
        f'<h2>{_T(lang, "How a study moves through EasyICU", "一项研究如何经过 EasyICU")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(key="eu_getstarted_steps"):
        steps = [
            (
                "1",
                _T(lang, "Choose a data mode", "选择数据模式"),
                _T(lang, "Demo Mode generates reproducible mock data; Real Data connects a local export folder. EasyICU never uploads anything either way.", "演示模式生成可复现模拟数据；真实数据模式连接本地导出文件夹。两种模式都不会上传数据。"),
                [(_T(lang, "Configure source", "配置数据源"), "extract_source", "primary")],
            ),
            (
                "2",
                _T(lang, "Extract & gate data", "抽取并闸门检查数据"),
                _T(lang, "A four-step flow normalizes the source, confirms the cohort, resolves concept coverage, then packages an export — each step gated so nothing runs on incomplete data.", "四步流程会标准化数据源、确认队列、检查变量覆盖并打包导出；每一步都有闸门，避免在不完整数据上运行。"),
                [(_T(lang, "Open extraction", "打开数据抽取"), "extract", "primary")],
            ),
            (
                "3",
                _T(lang, "Review patients & cohorts", "审阅患者与队列"),
                _T(lang, "Inspect patient-level tables and time series, then step up to cohort contrasts, coverage audits, and SOFA reclassification.", "先查看患者级表格和时间序列，再进入队列对比、覆盖审计和 SOFA 重分类。"),
                [(_T(lang, "Patient Review", "患者审阅"), "quick_viz", "primary"), (_T(lang, "Cohort Statistics", "队列统计"), "cohort", "secondary")],
            ),
            (
                "4",
                _T(lang, "Benchmark across databases", "跨数据库基准对比"),
                _T(lang, "Compare one cohort definition across two or more ICU databases — availability matrix and distribution summaries, side by side.", "在两个或更多 ICU 数据库中比较同一队列定义，并排查看可用性矩阵和分布摘要。"),
                [(_T(lang, "Cross-DB Benchmark", "跨库基准"), "cross_db", "primary")],
            ),
            (
                "5",
                _T(lang, "Run the Research Agent", "运行研究智能体"),
                _T(lang, "Plan, run, and review an auditable pipeline. The manuscript draft stays locked until every evidence check passes and you confirm.", "规划、运行并复核一条可审计流水线。所有证据检查通过并由你确认前，手稿草稿保持锁定。"),
                [(_T(lang, "Open Research Agent", "打开研究智能体"), "research_agent", "primary"), (_T(lang, "Ask the assistant", "询问助手"), "assistant", "secondary")],
            ),
        ]
        for number, title, desc, actions in steps:
            with st.container(key=f"eu_getstarted_step_{number}"):
                st.markdown(_guide_step_html(number, title, desc), unsafe_allow_html=True)
                cols = st.columns([0.075, 0.925], gap="small")
                with cols[1]:
                    action_cols = st.columns(len(actions), gap="small")
                    for idx, (label, target, tone) in enumerate(actions):
                        with action_cols[idx]:
                            if st.button(
                                label + " ->",
                                key=f"_eu_getstarted_step_{number}_{idx}",
                                type="primary" if tone == "primary" else "secondary",
                                use_container_width=True,
                            ):
                                if target == "assistant":
                                    _route_to_ai_assistant(
                                        st.session_state,
                                        "Help me frame a researchable ICU analysis question." if is_en
                                        else "帮我把 ICU 分析问题整理成可研究问题。",
                                    )
                                elif target == "extract_source":
                                    _route_to_extract_step(st.session_state, 1)
                                elif target == "extract":
                                    st.session_state["_active_main_page"] = "extract"
                                    st.session_state["_scroll_to_top"] = True
                                elif target == "research_agent":
                                    _route_to_research_agent_setup(st.session_state)
                                else:
                                    st.session_state["_active_main_page"] = target
                                    st.session_state["_scroll_to_top"] = True
                                st.rerun()

    faq_items = [
        (
            _T(lang, "Is any patient data uploaded?", "会上传患者数据吗？"),
            _T(lang, "No. EasyICU is local-first and the guarantee is enforced — extraction, review, and analysis all run on your machine. The only thing that can ever leave (and only if you explicitly enable it) is the Research Agent’s plan text, never patient rows.", "不会。EasyICU 是本地优先，抽取、审阅和分析都在你的机器上运行。只有在你明确启用时，研究智能体的计划文本才可能发送给模型端，患者行数据不会离开本机。"),
            True,
        ),
        (
            _T(lang, "What exactly is Demo Mode?", "演示模式是什么？"),
            _T(lang, "A reproducible synthetic dataset — by default 10 ICU stays over 24 hours across 19 feature modules. It produces no scientific findings; every number is seeded so you can learn the interface safely.", "一套可复现的合成数据：默认 10 个 ICU stays、24 小时窗口、19 个特征模块。它不产生科学发现；所有数值都只是种子化示例，用来安全学习界面。"),
            False,
        ),
        (
            _T(lang, "Why is the manuscript draft locked?", "为什么手稿草稿会锁定？"),
            _T(lang, "Drafting is a deliberate second stage. The agent only writes after denominators, coverage, tables, figures, and evidence references pass review.", "写作是第二阶段。只有分母、覆盖率、表格、图件和证据引用通过复核后，智能体才会起草。"),
            False,
        ),
        (
            _T(lang, "Which databases are supported?", "支持哪些数据库？"),
            _T(lang, "MIMIC-IV, eICU-CRD, AmsterdamUMCdb, HiRID, MIMIC-III, and SICdb. EasyICU detects known export layouts when you select a local folder.", "支持 MIMIC-IV、eICU-CRD、AmsterdamUMCdb、HiRID、MIMIC-III 和 SICdb。选择本地目录后，EasyICU 会识别已知导出结构。"),
            False,
        ),
        (
            _T(lang, "Do I need API tokens?", "需要 API token 吗？"),
            _T(lang, "Not for Demo Mode or data extraction/review/benchmark work. Tokens only apply if you connect an external model endpoint for the Research Agent.", "演示模式、数据抽取、审阅和基准对比都不需要。只有为研究智能体连接外部模型端点时才需要 token。"),
            False,
        ),
    ]
    st.markdown(
        '<div class="eu-guide-section eu-guide-faq-head">'
        f'<div class="eyebrow">{_T(lang, "Good to know", "使用前须知")}</div>'
        f'<h2>{_T(lang, "Common questions", "常见问题")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    open_idx = int(st.session_state.get("_eu_getstarted_faq_open", 0))
    with st.container(key="eu_getstarted_faq_card"):
        for idx, (question, answer, _default_open) in enumerate(faq_items):
            is_open = open_idx == idx
            with st.container(key=f"eu_getstarted_faq_{idx}"):
                label = f"{question} {'⌄' if is_open else '›'}"
                if st.button(
                    label,
                    key=f"_eu_getstarted_faq_q_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["_eu_getstarted_faq_open"] = -1 if is_open else idx
                    st.rerun()
                if is_open:
                    st.markdown(
                        f'<div class="eu-faq-answer">{_esc(answer)}</div>',
                        unsafe_allow_html=True,
                    )


def render_workspace_states_reference_page(lang: str) -> None:
    """Render the operational workspace overview page."""
    state = st.session_state
    _mode_label, _database_label, is_demo = _workspace_status_mode(state, lang)
    st.markdown(
        '<div class="eu-states-head">'
        f'<div class="eyebrow">{_T(lang, "Workspace overview", "工作区总览")}</div>'
        f'<h1>{_T(lang, "Current session status", "当前会话状态")}</h1>'
        f'<p>{_T(lang, "This page is a control-room overview: it summarizes the current data mode, extraction progress, review workspace, and Research Agent readiness. It does not generate data, preview fake loading states, or run model calls.", "这是一个控制室式总览页：汇总当前数据模式、提取进度、审阅工作区和研究智能体准备状态。它不会生成数据、展示假的加载态，也不会调用模型。")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        _workspace_status_overview_html(state, lang),
        unsafe_allow_html=True,
    )

    with st.container(key="eu_workspace_status_actions"):
        action_cols = st.columns([0.25, 0.25, 0.25, 0.25], gap="small")
        with action_cols[0]:
            if st.button(
                _T(lang, "Continue extraction", "继续数据提取"),
                key="_eu_workspace_continue_extract",
                type="primary",
                icon=":material/tune:",
                use_container_width=True,
            ):
                state["_active_main_page"] = "extract"
                state["_scroll_to_top"] = True
                st.rerun()
        with action_cols[1]:
            if st.button(
                _T(lang, "Patient Review", "患者审阅"),
                key="_eu_workspace_open_review",
                icon=":material/table_view:",
                use_container_width=True,
            ):
                _apply_workspace_state_action(state, "patient", "demo" if is_demo else "real")
                st.rerun()
        with action_cols[2]:
            if st.button(
                _T(lang, "Cohort Statistics", "队列统计"),
                key="_eu_workspace_open_cohort",
                icon=":material/query_stats:",
                use_container_width=True,
            ):
                if is_demo:
                    _apply_demo_defaults_for_tutorial(state)
                    state["_eu_topbar_run_request"] = {"page": "cohort", "requested_at": "workspace_states"}
                state["_active_main_page"] = "cohort"
                state["_scroll_to_top"] = True
                st.rerun()
        with action_cols[3]:
            if st.button(
                _T(lang, "Research Agent", "研究智能体"),
                key="_eu_workspace_open_agent",
                icon=":material/auto_awesome:",
                use_container_width=True,
            ):
                _route_to_research_agent_setup(state, force_real=not is_demo)
                st.rerun()


def _settings_row_copy(title: str, desc: str) -> str:
    return (
        '<div class="eu-settings-row-copy">'
        f'<b>{_esc(title)}</b>'
        f'<p>{_esc(desc)}</p>'
        '</div>'
    )


def _settings_value_pill(value: object, *, icon: str = "folder") -> str:
    icon_path = (
        '<path d="M3 7.5h6l1.5 2H21v8.5A2 2 0 0 1 19 20H5a2 2 0 0 1-2-2V7.5Z"/>'
        if icon == "folder"
        else '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>'
    )
    return (
        '<div class="eu-settings-value mono">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
        f'stroke-linejoin="round">{icon_path}</svg>'
        f'<span>{_esc(value)}</span>'
        '</div>'
    )


def _settings_static_toggle(*, active: bool, label: str = "") -> str:
    cls = " on" if active else ""
    label_html = f'<span class="eu-settings-toggle-label">{_esc(label)}</span>' if label else ""
    return (
        '<div class="eu-settings-toggle-wrap">'
        f'{label_html}<span class="eu-settings-toggle{cls}"><span></span></span>'
        '</div>'
    )


def _settings_state_chip(label: str, *, tone: str = "neutral") -> str:
    return f'<span class="eu-settings-chip {tone}"><i></i>{_esc(label)}</span>'


def _settings_status_tile(
    *,
    label: str,
    value: str,
    detail: str,
    tone: str = "neutral",
) -> str:
    return (
        f'<div class="eu-settings-status-tile {tone}">'
        f'<span>{_esc(label)}</span>'
        f'<b>{_esc(value)}</b>'
        f'<p>{_esc(detail)}</p>'
        '</div>'
    )


def _settings_apply_demo_defaults(patients: int, hours: int) -> None:
    st.session_state["demo_mode_patients"] = int(patients)
    st.session_state["demo_mode_hours"] = int(hours)
    params = dict(st.session_state.get("mock_params") or {})
    params["n_patients"] = int(patients)
    params["hours"] = int(hours)
    params.setdefault("demo_profile", "lite")
    st.session_state["mock_params"] = params


def _settings_repo_file_text(relative_path: str, *, fallback: str) -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return (root / relative_path).read_text(encoding="utf-8")
    except Exception:
        return fallback


def _settings_release_notes_text(lang: str) -> str:
    title = "EasyICU release notes" if lang == "en" else "EasyICU 发布说明"
    return "\n".join([
        f"# {title}",
        "",
        "Version: EasyICU 1.0.0",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "- Local-first ICU data extraction, review, and research-agent workflow.",
        "- Demo and real-data modes share the same audited web shell.",
        "- Research Agent runs are gated by human preflight and evidence review.",
        "- Patient rows remain local; external model calls require explicit opt-in.",
        "",
    ])


def _settings_diagnostics_json(
    state: MutableMapping[str, Any],
    *,
    lang: str,
    workdir: str,
    export_hint: str,
    provider_label: str,
    model_label: str,
    base_url_label: str,
    provider_needs_key: bool,
    api_key_present: bool,
    agent_run_value: str,
) -> str:
    """Return a support bundle that intentionally omits secrets and patient rows."""
    agent_last_run_id = _settings_agent_run_id(state, workdir=workdir)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "easyicu_version": "1.0.0",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "language": lang,
        "workspace": {
            "entry_mode": str(state.get("entry_mode") or "demo"),
            "database": str(state.get("database") or "mock"),
            "use_mock_data": bool(state.get("use_mock_data", True)),
            "workdir": workdir,
            "default_export_folder": export_hint,
            "path_validated": bool(state.get("path_validated", False)),
        },
        "demo_defaults": {
            "patients": int(state.get("demo_mode_patients") or (state.get("mock_params") or {}).get("n_patients") or 10),
            "hours": int(state.get("demo_mode_hours") or (state.get("mock_params") or {}).get("hours") or 24),
        },
        "llm": {
            "outbound_enabled": bool(state.get("llm_enabled", False)),
            "provider": str(state.get("llm_provider") or "easyicu_hosted"),
            "provider_label": provider_label,
            "model": model_label,
            "base_url": base_url_label,
            "credential_state": "present" if (provider_needs_key and api_key_present) else ("missing" if provider_needs_key else "not_required"),
        },
        "research_agent": {
            "run_state": agent_run_value,
            "workdir": workdir,
            "module_folder_mode": bool(state.get("_eu_settings_module_folder_mode", False)),
            "current_view": str(state.get("_ra_view") or "setup"),
            "last_run_id": agent_last_run_id,
        },
        "privacy": {
            "local_only": True,
            "anonymous_telemetry": False,
            "secrets_included": False,
            "patient_rows_included": False,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _settings_agent_run_id(state: MutableMapping[str, Any], *, workdir: str) -> str:
    """Return the run ID diagnostics should report for the visible Agent state."""
    workdir_path = Path(workdir).expanduser()
    workbench = state.get("_agent_workbench")
    source_dir = str(state.get("_agent_workbench_source_run_dir") or "").strip()
    if bool(state.get("_agent_workbench_is_active_selection")) and isinstance(workbench, dict):
        candidate = str(workbench.get("run_id") or (Path(source_dir).name if source_dir else "")).strip()
        if candidate:
            if source_dir:
                if Path(source_dir).expanduser().exists():
                    return candidate
            elif (workdir_path / candidate).exists():
                return candidate

    for key in ("research_agent_resume_run_id", "research_agent_last_run_id"):
        candidate = str(state.get(key) or "").strip()
        if candidate and (workdir_path / candidate).exists():
            return candidate
    return ""


def render_settings_redesign_page(lang: str) -> None:
    """Render the print-reference Settings page using live session settings."""
    from easyicu.webapp.llm_config import (
        coerce_public_provider,
        is_configured as is_shared_llm_configured,
        needs_api_key as shared_llm_needs_api_key,
        public_provider_defaults,
    )

    is_en = lang == "en"
    state = st.session_state
    entry_mode = str(state.get("entry_mode") or "demo")
    demo_patients = int(state.get("demo_mode_patients") or (state.get("mock_params") or {}).get("n_patients") or 10)
    demo_hours = int(state.get("demo_mode_hours") or (state.get("mock_params") or {}).get("hours") or 24)
    last_export = str(state.get("export_path") or state.get("last_export_dir") or "")
    export_hint = last_export or str(Path.home() / "easyicu_export")
    workdir = str(state.get("research_agent_workdir") or (Path.cwd() / "research_output" / "webapp").resolve())
    module_folder = str(state.get("research_agent_module_dir_text") or "")
    outbound_enabled = bool(state.get("llm_enabled", False))
    provider_key = coerce_public_provider(str(state.get("llm_provider") or "easyicu_hosted"))
    provider_label, default_base_url, default_model, _provider_needs_key, _desc_en, _desc_zh = public_provider_defaults(provider_key)
    provider_needs_key = shared_llm_needs_api_key(provider_key)
    api_key_present = bool(str(state.get("llm_api_key") or "").strip())
    model_label = str(state.get("llm_model") or default_model or "").strip()
    base_url_label = str(state.get("llm_base_url") or default_base_url or "").strip()
    shared_provider_configured = is_shared_llm_configured()
    hosted_model_active = provider_key == "easyicu_hosted"
    density_pref = str(state.get("ui_density") or "comfortable")
    if density_pref not in {"comfortable", "compact"}:
        density_pref = "comfortable"
        state["ui_density"] = density_pref
    reduce_motion = bool(state.get("reduce_motion", False))
    state["_eu_settings_module_folder_mode"] = bool(module_folder)
    state["_eu_settings_allow_outbound_model_calls"] = outbound_enabled
    state["_eu_settings_reduce_motion"] = reduce_motion

    def _agent_run_status(current_outbound_enabled: bool) -> tuple[str, str, str]:
        if hosted_model_active:
            return (
                _T(lang, "Mock or per-run override", "离线或单次覆盖"),
                _T(
                    lang,
                    "Hosted relay is reserved for assistant/internal use; real Research Agent runs choose Mock or a user endpoint.",
                    "Hosted relay 仅用于助手/内部用途；真实 Research Agent 运行请用 Mock 或用户提供的端点。",
                ),
                "warn",
            )
        if shared_provider_configured and model_label and current_outbound_enabled:
            return _T(lang, "Shared external ready", "共享外部端点就绪"), f"{provider_label} · {model_label}", "ok"
        if shared_provider_configured and model_label:
            return (
                _T(lang, "Configured, calls off", "已配置，调用关闭"),
                _T(
                    lang,
                    "Turn on outbound model calls before reusing this endpoint in a real run.",
                    "真实运行复用该端点前，需要打开模型端调用。",
                ),
                "warn",
            )
        return (
            _T(lang, "Setup needed", "需要补齐设置"),
            _T(
                lang,
                "Add the API key/model, or choose MockLLMClient in Research Agent setup.",
                "请补齐 API key/模型，或在 Research Agent 配置中选择 MockLLMClient。",
            ),
            "bad",
        )

    agent_run_value, agent_run_detail, agent_run_tone = _agent_run_status(outbound_enabled)

    if provider_needs_key:
        credential_value = _T(lang, "Session key present", "会话 Key 已填写") if api_key_present else _T(lang, "API key missing", "缺少 API Key")
        credential_detail = _T(
            lang,
            "API keys stay in this browser session only.",
            "API Key 只保存在当前浏览器会话。",
        )
        credential_tone = "ok" if api_key_present else "bad"
    else:
        credential_value = _T(lang, "No user key required", "无需用户 Key")
        credential_detail = _T(
            lang,
            "The selected provider uses the configured EasyICU relay.",
            "当前服务商使用 EasyICU 已配置的代理。",
        )
        credential_tone = "neutral"

    def _settings_module_folder_mode_changed() -> None:
        enabled = bool(st.session_state.get("_eu_settings_module_folder_mode", False))
        if enabled and not str(st.session_state.get("research_agent_module_dir_text") or ""):
            st.session_state["research_agent_cohort_source"] = _T(
                lang,
                "Pick an EasyICU module export folder",
                "选择 EasyICU 模块导出文件夹",
            )
            _route_to_research_agent_setup(
                st.session_state,
                force_real=True,
                focus_module_folder=True,
            )
        elif not enabled:
            st.session_state["research_agent_module_dir_text"] = ""

    def _settings_outbound_model_calls_changed() -> None:
        enabled = bool(st.session_state.get("_eu_settings_allow_outbound_model_calls", False))
        st.session_state["llm_enabled"] = enabled
        st.session_state["_llm_toggle"] = enabled
        st.session_state["_llm_toggle_sync_pending"] = True
        if not enabled:
            st.session_state["_sidebar_ai_open"] = False
            st.session_state["_floating_ai_open"] = False
            st.session_state["_inline_ai_panel_open"] = False

    def _settings_reduce_motion_changed() -> None:
        st.session_state["reduce_motion"] = bool(
            st.session_state.get("_eu_settings_reduce_motion", False)
        )

    def _settings_start_path_edit(edit_key: str, input_key: str, current_value: str) -> None:
        state[edit_key] = True
        state[input_key] = str(current_value or "")
        state["_scroll_to_top"] = True

    def _settings_apply_path_edit(
        *,
        edit_key: str,
        input_key: str,
        state_key: str,
        fallback_value: str,
    ) -> None:
        raw_value = str(st.session_state.get(input_key) or fallback_value or "").strip()
        normalized = str(Path(raw_value or fallback_value).expanduser())
        state[state_key] = normalized
        state[edit_key] = False
        state["_scroll_to_top"] = True
        if state_key == "export_path":
            state["sidebar_export_path_input"] = normalized
            state["_sidebar_export_path_default"] = normalized

    def _settings_cancel_path_edit(edit_key: str) -> None:
        state[edit_key] = False
        state["_scroll_to_top"] = True

    st.markdown(
        '<div class="eu-settings-page-head">'
        f'<div class="eyebrow">{_T(lang, "Workspace · 设置", "工作区 · 设置")}</div>'
        f'<h1>{_T(lang, "Settings", "设置")}</h1>'
        f'<p>{_T(lang, "Configure how EasyICU reads data, runs the agent, and presents the workspace. Everything is local and reversible.", "配置 EasyICU 如何读取数据、运行 agent 并呈现工作区。所有设置都保留在本机，且可以随时改回。")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eu-settings-section-head">'
        f'<span>{_T(lang, "Workspace", "工作区")}</span>'
        f'<h2>{_T(lang, "Local paths", "本地路径")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="eu_settings_workspace_card"):
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Working directory", "工作目录"),
                    _T(lang, "Where agent runs, caches, and review bundles are written.", "Agent 运行、缓存与复核包写入的位置。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            value_col, action_col = st.columns([0.72, 0.28], gap="small")
            with value_col:
                if state.get("_eu_settings_edit_workdir"):
                    st.text_input(
                        _T(lang, "Working directory", "工作目录"),
                        key="_eu_settings_workdir_input",
                        label_visibility="collapsed",
                        placeholder=str((Path.cwd() / "research_output" / "webapp").resolve()),
                    )
                else:
                    st.markdown(_settings_value_pill(workdir, icon="folder"), unsafe_allow_html=True)
            with action_col:
                if state.get("_eu_settings_edit_workdir"):
                    if st.button(_T(lang, "Save", "保存"), key="_eu_settings_save_workdir", type="primary", use_container_width=True):
                        _settings_apply_path_edit(
                            edit_key="_eu_settings_edit_workdir",
                            input_key="_eu_settings_workdir_input",
                            state_key="research_agent_workdir",
                            fallback_value=workdir,
                        )
                        st.rerun()
                    if st.button(_T(lang, "Cancel", "取消"), key="_eu_settings_cancel_workdir", use_container_width=True):
                        _settings_cancel_path_edit("_eu_settings_edit_workdir")
                        st.rerun()
                elif st.button(
                    _T(lang, "Change", "修改"),
                    key="_eu_settings_change_workdir",
                    help=_T(lang, "Edit the working directory here", "在此处修改工作目录"),
                    use_container_width=True,
                ):
                    _settings_start_path_edit("_eu_settings_edit_workdir", "_eu_settings_workdir_input", workdir)
                    st.rerun()
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)

        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Default export folder", "默认导出目录"),
                    _T(lang, "Destination for code, tables, figures, and evidence-ledger bundles.", "代码、表格、图件与证据账本包的导出目的地。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            value_col, action_col = st.columns([0.72, 0.28], gap="small")
            with value_col:
                if state.get("_eu_settings_edit_export_path"):
                    st.text_input(
                        _T(lang, "Default export folder", "默认导出目录"),
                        key="_eu_settings_export_path_input",
                        label_visibility="collapsed",
                        placeholder=str(Path.home() / "easyicu_export"),
                    )
                else:
                    st.markdown(_settings_value_pill(export_hint, icon="download"), unsafe_allow_html=True)
            with action_col:
                if state.get("_eu_settings_edit_export_path"):
                    if st.button(_T(lang, "Save", "保存"), key="_eu_settings_save_export_path", type="primary", use_container_width=True):
                        _settings_apply_path_edit(
                            edit_key="_eu_settings_edit_export_path",
                            input_key="_eu_settings_export_path_input",
                            state_key="export_path",
                            fallback_value=export_hint,
                        )
                        st.rerun()
                    if st.button(_T(lang, "Cancel", "取消"), key="_eu_settings_cancel_export_path", use_container_width=True):
                        _settings_cancel_path_edit("_eu_settings_edit_export_path")
                        st.rerun()
                elif st.button(
                    _T(lang, "Change", "修改"),
                    key="_eu_settings_change_export",
                    help=_T(lang, "Edit the default export folder here", "在此处修改默认导出目录"),
                    use_container_width=True,
                ):
                    _settings_start_path_edit("_eu_settings_edit_export_path", "_eu_settings_export_path_input", export_hint)
                    st.rerun()
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)

        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Module-folder mode", "模块目录模式"),
                    _T(lang, "Reuse a previously exported module folder instead of re-extracting.", "复用已经导出的模块目录，而不是重新抽取。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            st.toggle(
                _T(lang, "Module-folder mode", "模块目录模式"),
                key="_eu_settings_module_folder_mode",
                label_visibility="collapsed",
                help=_T(
                    lang,
                    "Turn on to open Research Agent setup and choose a module export folder.",
                    "打开后进入 Research Agent 配置并选择模块导出目录。",
                ),
                on_change=_settings_module_folder_mode_changed,
            )

    st.markdown(
        '<div class="eu-settings-section-head">'
        f'<span>{_T(lang, "Data mode", "数据模式")}</span>'
        f'<h2>{_T(lang, "Defaults for new sessions", "新会话默认值")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="eu_settings_data_mode_card"):
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Start mode", "启动模式"),
                    _T(lang, "Which mode a new workspace opens in. You can always switch later.", "新工作区默认打开哪种模式，之后仍可切换。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            mode_cols = st.columns(2, gap="small")
            with mode_cols[0]:
                if st.button(
                    "Demo",
                    key="_eu_settings_mode_demo",
                    type="primary" if entry_mode == "demo" else "secondary",
                    use_container_width=True,
                ):
                    state["entry_mode"] = "demo"
                    state["use_mock_data"] = True
                    st.rerun()
            with mode_cols[1]:
                if st.button(
                    "Real Data",
                    key="_eu_settings_mode_real",
                    type="primary" if entry_mode == "real" else "secondary",
                    use_container_width=True,
                ):
                    state["entry_mode"] = "real"
                    state["use_mock_data"] = False
                    st.rerun()
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)

        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Demo patients", "演示患者数"),
                    _T(lang, "Default cohort size generated in Demo Mode.", "演示模式默认生成的队列大小。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            patient_cols = st.columns(3, gap="small")
            for idx, value in enumerate((10, 20, 50)):
                with patient_cols[idx]:
                    if st.button(
                        str(value),
                        key=f"_eu_settings_demo_patients_{value}",
                        type="primary" if demo_patients == value else "secondary",
                        use_container_width=True,
                    ):
                        _settings_apply_demo_defaults(value, demo_hours)
                        st.rerun()
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)

        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Demo duration", "演示时长"),
                    _T(lang, "Default hours of hourly time points per stay.", "每个 ICU stay 默认生成多少小时的小时级时间点。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            hour_cols = st.columns(3, gap="small")
            for idx, value in enumerate((24, 48, 168)):
                with hour_cols[idx]:
                    if st.button(
                        f"{value}h",
                        key=f"_eu_settings_demo_hours_{value}",
                        type="primary" if demo_hours == value else "secondary",
                        use_container_width=True,
                    ):
                        _settings_apply_demo_defaults(demo_patients, value)
                        st.rerun()

    st.markdown(
        '<div class="eu-settings-section-head">'
        f'<span>{_T(lang, "Privacy", "隐私")}</span>'
        f'<h2>{_T(lang, "Local-first guarantees", "本地优先保障")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="eu_settings_privacy_card"):
        static_privacy_rows = [
            (
                _T(lang, "Local-only mode", "本地模式"),
                _T(lang, "Patient data never leaves your machine. This guarantee is enforced and cannot be disabled.", "患者数据不会离开你的机器。该保障强制启用，不能关闭。"),
                True,
                _T(lang, "enforced", "强制"),
            ),
            (
                _T(lang, "Anonymous usage telemetry", "匿名使用遥测"),
                _T(lang, "EasyICU collects nothing unless you explicitly add your own instrumentation.", "EasyICU 不收集遥测，除非你自己显式添加。"),
                False,
                "",
            ),
            (
                _T(lang, "Cache cohort frames", "缓存队列数据帧"),
                _T(lang, "Keep extracted frames on disk to speed up repeat reviews.", "将已抽取数据帧保存在本地，加速重复复核。"),
                True,
                "",
            ),
        ]
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Allow outbound model calls", "允许模型端调用"),
                    _T(lang, "When enabled, only prompts, plans, and run logs may reach the configured endpoint; patient rows are never sent.", "启用后，只有提示词、计划和运行日志可能发送到配置的模型端点；患者行数据不会发送。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            outbound_widget_value = st.toggle(
                _T(lang, "Allow outbound model calls", "允许模型端调用"),
                key="_eu_settings_allow_outbound_model_calls",
                label_visibility="collapsed",
                help=_T(
                    lang,
                    "Shared switch used by AI Assistant and Research Agent. External Research Agent runs still show a per-run disclosure gate.",
                    "AI 助手和 Research Agent 共用该开关；外部模型运行仍会显示单次运行披露关口。",
                ),
                on_change=_settings_outbound_model_calls_changed,
            )
            if bool(outbound_widget_value) != bool(state.get("llm_enabled", False)):
                state["llm_enabled"] = bool(outbound_widget_value)
                state["_llm_toggle"] = bool(outbound_widget_value)
                state["_llm_toggle_sync_pending"] = True
                if not outbound_widget_value:
                    state["_sidebar_ai_open"] = False
                    state["_floating_ai_open"] = False
                    state["_inline_ai_panel_open"] = False
            outbound_enabled = bool(state.get("llm_enabled", False))
            agent_run_value, agent_run_detail, agent_run_tone = _agent_run_status(outbound_enabled)
            if bool(state.get("_eu_settings_allow_outbound_model_calls", False)) != outbound_enabled:
                _settings_outbound_model_calls_changed()
                st.rerun()
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)

        for title, desc, active, label in static_privacy_rows:
            st.markdown(
                '<div class="eu-settings-privacy-row">'
                f'{_settings_row_copy(title, desc)}'
                f'{_settings_static_toggle(active=active, label=label)}'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="eu-settings-section-head">'
        f'<span>{_T(lang, "Research Agent", "研究智能体")}</span>'
        f'<h2>{_T(lang, "Run behavior", "运行行为")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="eu_settings_agent_card"):
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                    _settings_row_copy(
                        _T(lang, "Model", "模型"),
                        _T(lang, "Shared provider used by the assistant. Research Agent can still choose Mock/offline for a local test run.", "助手使用的共享服务商；Research Agent 仍可为单次运行选择 Mock/离线模式。"),
                    ),
                    unsafe_allow_html=True,
                )
        with right:
            model_cols = st.columns(2, gap="small")
            with model_cols[0]:
                if st.button(
                    _T(lang, "Hosted (assistant)", "Hosted（助手）"),
                    key="_eu_settings_model_local",
                    type="primary" if hosted_model_active else "secondary",
                    use_container_width=True,
                ):
                    state["llm_provider"] = "easyicu_hosted"
                    state["_llm_provider_sel"] = "easyicu_hosted"
                    state["llm_api_key"] = ""
                    state["llm_base_url"] = ""
                    state["llm_model"] = ""
                    state["_llm_api_key_inp"] = ""
                    state["_llm_base_url_inp"] = ""
                    state["_llm_model_inp"] = ""
                    state["llm_configured"] = False
                    st.rerun()
            with model_cols[1]:
                if st.button(
                    "External endpoint",
                    key="_eu_settings_model_external",
                    type="primary" if not hosted_model_active else "secondary",
                    use_container_width=True,
                ):
                    external_provider = "openrouter" if str(state.get("llm_provider") or "") == "easyicu_hosted" else provider_key
                    _, external_base_url, external_model, _, _, _ = public_provider_defaults(external_provider)
                    _, hosted_base_url, hosted_model, _, _, _ = public_provider_defaults("easyicu_hosted")
                    state["llm_provider"] = external_provider
                    state["_llm_provider_sel"] = external_provider
                    if str(state.get("llm_base_url") or "") in {"", hosted_base_url}:
                        state["llm_base_url"] = external_base_url
                        state["_llm_base_url_inp"] = external_base_url
                    if str(state.get("llm_model") or "") in {"", hosted_model}:
                        state["llm_model"] = external_model
                        state["_llm_model_inp"] = external_model
                    if str(state.get("llm_provider") or "") != "easyicu_hosted":
                        state["llm_api_key"] = ""
                        state["_llm_api_key_inp"] = ""
                        state["llm_configured"] = False
                    st.rerun()
            st.markdown(
                '<div class="eu-settings-route-note">'
                f'{_settings_state_chip(agent_run_value, tone=agent_run_tone)}'
                f'<p>{_esc(agent_run_detail)}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)
        static_rows = [
            (
                _T(lang, "Token budget", "Token 预算"),
                _T(lang, "Soft cap per external run. Demo and Mock runs use zero tokens.", "外部模型运行的软上限。演示与 Mock 运行不使用 token。"),
                '<div class="eu-settings-value mono"><span>120,000</span></div>',
            ),
            (
                _T(lang, "Auto-repair steps", "自动修复步骤"),
                _T(lang, "Deterministically retry a failed analysis step before halting the run.", "分析步骤失败时，在停止前进行确定性重试。"),
                _settings_static_toggle(active=True),
            ),
            (
                _T(lang, "Evidence gate", "证据闸门"),
                _T(lang, "Strict requires every contract to pass before drafting unlocks.", "Strict 要求每个契约通过后才解锁草稿。"),
                '<div class="eu-settings-segment mono"><span class="active">Strict</span><span>Standard</span></div>',
            ),
        ]
        for title, desc, control_html in static_rows:
            left, right = st.columns([1.35, 1.0], gap="large")
            with left:
                st.markdown(_settings_row_copy(title, desc), unsafe_allow_html=True)
            with right:
                st.markdown(control_html, unsafe_allow_html=True)
            st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)

    with st.container(key="eu_settings_llm_live_card"):
        st.markdown(
            '<div class="eu-settings-live-head">'
            f'<span>{_T(lang, "AI / API connection", "AI / API 连接")}</span>'
            f'<p>{_T(lang, "These controls are the real shared settings used by the assistant and Research Agent.", "这里是真实共享设置，会被助手和 Research Agent 使用。")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="eu-settings-status-grid">'
            + _settings_status_tile(
                label=_T(lang, "Outbound calls", "模型端调用"),
                value=_T(lang, "Allowed", "已允许") if outbound_enabled else _T(lang, "Off by default", "默认关闭"),
                detail=_T(
                    lang,
                    "Patient rows stay local; only prompts/plans/logs may leave when enabled.",
                    "患者行数据仍保留本地；开启后仅提示词、计划和日志可能离开。",
                ),
                tone="ok" if outbound_enabled else "warn",
            )
            + _settings_status_tile(
                label=_T(lang, "Provider", "服务商"),
                value=provider_label,
                detail=base_url_label or _T(lang, "Endpoint will be requested in the per-run override.", "端点将在单次运行覆盖中填写。"),
                tone="neutral" if hosted_model_active else "ok",
            )
            + _settings_status_tile(
                label=_T(lang, "Credential", "凭证"),
                value=credential_value,
                detail=credential_detail,
                tone=credential_tone,
            )
            + _settings_status_tile(
                label=_T(lang, "Research Agent", "研究智能体"),
                value=agent_run_value,
                detail=agent_run_detail,
                tone=agent_run_tone,
            )
            + '</div>',
            unsafe_allow_html=True,
        )
        from easyicu.webapp.llm_chat import render_llm_settings
        render_llm_settings(
            show_status_card=False,
            controls_only=True,
            show_enable_toggle=False,
            open_sidebar_on_enable=False,
        )

    st.markdown(
        '<div class="eu-settings-section-head">'
        f'<span>{_T(lang, "Language", "语言")}</span>'
        f'<h2>{_T(lang, "Language & display", "语言与显示")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="eu_settings_language_card"):
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Interface language", "界面语言"),
                    _T(lang, "EasyICU is fully bilingual; labels fit both scripts.", "EasyICU 完整双语，标签适配中英文。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            lang_cols = st.columns(2, gap="small")
            with lang_cols[0]:
                if st.button("English", key="_eu_settings_page_lang_en", type="primary" if is_en else "secondary", use_container_width=True):
                    state["language"] = "en"
                    st.rerun()
            with lang_cols[1]:
                if st.button("中文", key="_eu_settings_page_lang_zh", type="primary" if not is_en else "secondary", use_container_width=True):
                    state["language"] = "zh"
                    st.rerun()
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Density", "密度"),
                    _T(lang, "Comfortable adds breathing room; compact maximises rows on screen.", "舒适模式增加留白；紧凑模式增加屏幕行数。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            density_cols = st.columns(2, gap="small")
            with density_cols[0]:
                if st.button(
                    _T(lang, "Comfortable", "舒适"),
                    key="_eu_settings_density_comfortable",
                    type="primary" if density_pref == "comfortable" else "secondary",
                    use_container_width=True,
                ):
                    state["ui_density"] = "comfortable"
                    st.rerun()
            with density_cols[1]:
                if st.button(
                    _T(lang, "Compact", "紧凑"),
                    key="_eu_settings_density_compact",
                    type="primary" if density_pref == "compact" else "secondary",
                    use_container_width=True,
                ):
                    state["ui_density"] = "compact"
                    st.rerun()
        st.markdown('<div class="eu-settings-divider"></div>', unsafe_allow_html=True)
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown(
                _settings_row_copy(
                    _T(lang, "Reduce motion", "减少动态效果"),
                    _T(lang, "Disable shimmer and progress animations.", "关闭 shimmer 和进度动画。"),
                ),
                unsafe_allow_html=True,
            )
        with right:
            st.toggle(
                _T(lang, "Reduce motion", "减少动态效果"),
                key="_eu_settings_reduce_motion",
                label_visibility="collapsed",
                on_change=_settings_reduce_motion_changed,
            )

    st.markdown(
        '<div class="eu-settings-section-head">'
        f'<span>{_T(lang, "About", "关于")}</span>'
        f'<h2>{_T(lang, "Environment", "环境")}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="eu-settings-card eu-settings-env">'
        '<div><span>Version</span><b class="mono">EasyICU 1.0.0</b></div>'
        '<div><span>Python</span><b class="mono">3.10+</b></div>'
        '<div><span>Databases detected</span><b class="mono">MIMIC-IV · eICU · AUMC · HiRID · MIMIC-III · SICdb</b></div>'
        f'<div><span>Workspace</span><b class="mono">{_esc(workdir)}</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    docs_text = _settings_repo_file_text(
        "README.md" if is_en else "README_zh.md",
        fallback=_T(lang, "EasyICU documentation is not available in this checkout.", "当前 checkout 中没有可用的 EasyICU 文档。"),
    )
    diagnostics_json = _settings_diagnostics_json(
        state,
        lang=lang,
        workdir=workdir,
        export_hint=export_hint,
        provider_label=provider_label,
        model_label=model_label,
        base_url_label=base_url_label,
        provider_needs_key=provider_needs_key,
        api_key_present=api_key_present,
        agent_run_value=agent_run_value,
    )
    with st.container(key="eu_settings_env_actions"):
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.download_button(
                _T(lang, "Release notes", "发布说明"),
                data=_settings_release_notes_text(lang),
                file_name="easyicu-release-notes.md",
                mime="text/markdown",
                use_container_width=True,
                key="_eu_settings_release_notes_download",
            )
        with c2:
            st.download_button(
                _T(lang, "Documentation", "文档"),
                data=docs_text,
                file_name="easyicu-readme.md",
                mime="text/markdown",
                use_container_width=True,
                key="_eu_settings_documentation_download",
            )
        with c3:
            st.download_button(
                _T(lang, "Export diagnostics", "导出诊断"),
                data=diagnostics_json,
                file_name="easyicu-settings-diagnostics.json",
                mime="application/json",
                use_container_width=True,
                key="_eu_settings_diagnostics_download",
            )


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
        brand_col, lang_col, version_col = st.columns([1, 0.12, 0.14], gap="small")
        with brand_col:
            st.markdown(
                '<div class="eu-entry-brand">'
                '<div class="eu-entry-logo">'
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
                'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M9 3h6"/><path d="M10 3v5.2L5.4 17a3 3 0 0 0 2.7 4.4h7.8'
                'a3 3 0 0 0 2.7-4.4L14 8.2V3"/><path d="M7.5 15h9"/></svg>'
                '</div>'
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
        '<div class="eu-entry-hero" style="padding:34px 0 32px;text-align:center">'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4);letter-spacing:.08em;'
        f'text-transform:uppercase;font-family:var(--font-mono)">'
        f'{_T(lang, "Local-first ICU research workflow", "本地优先 · ICU 数据研究工作流")}</div>'
        f'<h1 style="margin:12px 0 0;font-size:30px;font-weight:600;letter-spacing:-0.025em;color:var(--ink)">'
        f'{_T(lang, "Extract. Review. Analyze. Draft.", "数据抽取 · 审阅 · 分析 · 起草")}</h1>'
        f'<div class="eu-entry-hero-copy" style="font-size:13.5px;color:var(--ink-3);margin-top:10px">'
        f'{_T(lang, "Choose how data enters the workspace. Everything runs on your machine — nothing is uploaded.", "选择数据如何进入工作区。所有流程都在本机运行，不上传任何患者数据。")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    demo_card = (
        '<div class="eu-entry-mode-card demo" style="padding:24px 24px 18px;background:var(--accent-soft);'
        'border:1px solid var(--accent-border);border-radius:14px;display:flex;'
        'flex-direction:column;gap:12px">'
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
                        "可查看研究智能体静态示例库"),
                _T(lang, "Switch to real data anytime without losing work",
                        "随时可切换到真实数据,不丢工作"),
            ]
        ])
        + '</ul></div>'
    )
    real_card = (
        '<div class="eu-entry-mode-card real" style="padding:24px 24px 18px;background:var(--surface);'
        'border:1px solid var(--hair);border-radius:14px;display:flex;'
        'flex-direction:column;gap:12px">'
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

    with st.container(key="eu_entry_modes_grid"):
        demo_col, real_col = st.columns(2, gap="large")
        with demo_col:
            with st.container(key="eu_entry_demo_mode_card"):
                st.markdown(demo_card, unsafe_allow_html=True)
                if st.button(_T(lang, "Start demo →", "开始演示 →"),
                             key="_eu_entry_demo", type="primary", use_container_width=True):
                    _route_to_extract_entry_mode(st.session_state, "demo")
                    st.rerun()
        with real_col:
            with st.container(key="eu_entry_real_mode_card"):
                st.markdown(real_card, unsafe_allow_html=True)
                if st.button(_T(lang, "Use local data folder →", "使用本地数据目录 →"),
                             key="_eu_entry_real", use_container_width=True):
                    _route_to_extract_entry_mode(st.session_state, "real")
                    st.rerun()

    with st.container(key="eu_entry_code_row"):
        aux_col, code_col = st.columns([0.72, 0.28], gap="large")
        with aux_col:
            st.markdown(
                '<div class="eu-entry-nodata" style="display:flex;align-items:center;gap:14px">'
                '<div class="eu-entry-nodata-icon" style="color:var(--ink-3)">'
                '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>'
                '<path d="M14 3v6h6"/></svg></div>'
                '<div style="flex:1">'
                f'<div style="font-size:13px;font-weight:500">{_T(lang, "No data yet?", "还没有数据?")}</div>'
                f'<div style="font-size:12px;color:var(--ink-3)">'
                f'{_T(lang, "Let the Research Agent generate a reusable code skeleton, then plug data in later.", "让研究智能体先生成可复用代码骨架，稍后再接入数据。")}</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        with code_col:
            if st.button(_T(lang, "Generate code only →", "仅生成代码 →"),
                         key="_eu_entry_nodata", use_container_width=True):
                _route_to_research_agent_no_data_setup(st.session_state)
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
        f'<div class="eu-entry-step"><small>{_T(lang, "04", "04")}</small><b>{_T(lang, "Export & handoff", "导出与交付")}</b><span>{_T(lang, "Save concept data plus a reproducible manifest; hand findings to the agent.", "保存概念数据和可复现 manifest，并将发现交给智能体。")}</span></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
