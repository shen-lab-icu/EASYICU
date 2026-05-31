"""Sidebar rendering workflow for the EasyICU Streamlit app.

This is a transitional extraction: the sidebar still calls app-level helpers,
but the long Streamlit rendering block no longer lives in app.py.
"""

from __future__ import annotations

from typing import Any, MutableMapping
from pathlib import Path
import html
import os
import re

import pandas as pd
import streamlit as st
from easyicu.webapp.concept_catalog import (
    CONCEPT_DESCRIPTIONS,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_INTERNAL,
)
from easyicu.webapp.session_state import clear_agent_continuation_state, clear_run_state
from easyicu.webapp.ui_helpers import (
    PipelineStep,
    ShellNavItem,
    icon as _icon,
    render_brand_html,
    render_nav_item_html,
    render_pill_html,
)


_PROTECTED_CONTEXT_NAMES = {"render_sidebar", "render_extract_page", "_install_app_context"}

_STEP2_RESET_PENDING_KEY = "_step2_cohort_builder_reset_pending"
_STEP2_WIDGET_KEYS = (
    "cohort_enabled",
    "cohort_age_min_design",
    "cohort_age_max_design",
    "cohort_first_icu_design",
    "cohort_los_min_design",
    "cohort_gender_design",
    "cohort_survival_design",
    "cohort_icd_include_query_design",
    "cohort_icd_exclude_query_design",
)

_MODULE_LABEL_PREFIX_RE = re.compile(r"^[^\w\u4e00-\u9fff]+\s*")


def _clean_module_label(label: object) -> str:
    """Strip decorative icon prefixes while preserving the underlying module key."""
    text = str(label or "").strip()
    cleaned = _MODULE_LABEL_PREFIX_RE.sub("", text).strip()
    return cleaned or text


def _module_display_name(module_key_or_label: object, lang: str) -> str:
    """Return a clean display label for an internal module key or existing label."""
    raw = str(module_key_or_label or "").strip()
    if raw in CONCEPT_GROUP_NAMES:
        en_name, zh_name = CONCEPT_GROUP_NAMES[raw]
        return _clean_module_label(en_name if lang == "en" else zh_name)
    return _clean_module_label(raw.replace("_", " "))


def _concept_matches_search(concept: str, search: str) -> bool:
    """Return whether a query matches a concept id or user-visible metadata."""
    query = str(search or "").strip().lower()
    if not query:
        return True
    fields: list[str] = [str(concept)]
    fields.extend(str(item) for item in CONCEPT_DICTIONARY.get(concept, ()) if item)
    fields.extend(str(item) for item in CONCEPT_DESCRIPTIONS.get(concept, ()) if item)
    return any(query in field.lower() for field in fields)


def _concept_group_matches_search(group_name: str, concepts: list[str], search: str) -> bool:
    """Return whether the Step 3 module card should be shown for a query."""
    query = str(search or "").strip().lower()
    if not query:
        return True
    group_fields = [
        group_name,
        _module_display_name(group_name, "en"),
        _module_display_name(group_name, "zh"),
    ]
    if any(query in field.lower() for field in group_fields):
        return True
    return any(_concept_matches_search(concept, query) for concept in concepts)


def _activate_entry_mode(target: str) -> None:
    """Switch the workspace mode while keeping the shell on extraction."""
    if target not in {"demo", "real"}:
        return
    previous_database = st.session_state.get("database")
    st.session_state["entry_mode"] = target
    st.session_state["use_mock_data"] = target == "demo"
    if target == "demo":
        st.session_state["database"] = "mock"
        st.session_state["mock_params"] = {
            "n_patients": 10,
            "hours": 24,
            "demo_profile": "lite",
        }
        st.session_state["demo_mode_patients"] = 10
        st.session_state["demo_mode_hours"] = 24
    elif previous_database not in {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}:
        st.session_state["database"] = "miiv"
        st.session_state["path_validated"] = False
        st.session_state.pop("last_validated_path", None)
    for key in ("step1_confirmed", "step2_confirmed", "step3_confirmed", "export_completed"):
        st.session_state[key] = False
    st.session_state["trigger_export"] = False
    st.session_state["_exporting_in_progress"] = False
    st.session_state["loaded_concepts"] = {}
    st.session_state["loaded_data_origin"] = "none"
    st.session_state["patient_ids"] = []
    st.session_state["all_patient_count"] = 0
    st.session_state["selected_patient"] = None
    st.session_state["selected_concepts"] = []
    st.session_state["_active_main_page"] = "extract"


def _return_to_entry_home() -> None:
    """Return to the entry screen and clear run-specific workspace state."""
    clear_run_state("all")
    st.session_state.entry_mode = "none"
    st.session_state.use_mock_data = False
    st.session_state["_active_main_page"] = "tutorial"


def _render_source_mode_tabs(entry_mode: str) -> None:
    """Render the Data source segmented mode control from the design."""
    lang = st.session_state.get("language", "en")
    tabs = [
        ("demo", "Demo", "模拟数据", ":material/science:"),
        ("real", "Real Data", "真实数据", ":material/database:"),
    ]
    cols = st.columns(2, gap="small")
    for col, (target, label_en, label_zh, icon_name) in zip(cols, tabs):
        with col:
            label = label_en if lang == "en" else label_zh
            if st.button(
                label,
                key=f"_eu_source_mode_{target}",
                type="primary" if target == entry_mode else "secondary",
                use_container_width=True,
                icon=icon_name,
            ):
                _activate_entry_mode(target)
                st.rerun()


def _render_data_source_page_header(entry_mode: str, *, desc: str) -> None:
    """Top header used by the Step 1 data-source page."""
    lang = st.session_state.get("language", "en")
    title = "Configure data source" if lang == "en" else "配置数据源"
    left, right = st.columns([1.55, 1.25], gap="large")
    with left:
        st.markdown(
            '<div class="eu-source-header">'
            f'<h1>{html.escape(title)}</h1>'
            f'<p>{html.escape(desc)}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with right:
        _render_source_mode_tabs(entry_mode)


def _demo_module_catalog_html(lang: str) -> str:
    """Return a compact all-module catalog for the Step 1 demo preview."""
    title = "All demo feature modules" if lang == "en" else "全部演示特征模块"
    subtitle = (
        "These modules will be available in Step 3. Confirm the demo source first, then choose all or a smaller set."
        if lang == "en"
        else "这些模块会在第 3 步出现。请先确认演示数据源，然后选择全部或较小的一组。"
    )
    rows = []
    for key, concepts in CONCEPT_GROUPS_INTERNAL.items():
        en_name, zh_name = CONCEPT_GROUP_NAMES.get(key, (key, key))
        display_name = _clean_module_label(en_name if lang == "en" else zh_name)
        examples = ", ".join(concepts[:4])
        if len(concepts) > 4:
            examples += ", ..."
        rows.append(
            '<div class="eu-module-catalog-row">'
            f'<b>{html.escape(display_name)}</b>'
            f'<span>{len(concepts)} {html.escape("features" if lang == "en" else "个变量")}</span>'
            f'<em>{html.escape(examples)}</em>'
            '</div>'
        )
    return (
        '<div class="eu-module-catalog">'
        f'<div class="eu-module-catalog-title">{html.escape(title)}</div>'
        f'<div class="eu-module-catalog-sub">{html.escape(subtitle)}</div>'
        '<div class="eu-module-catalog-grid">'
        f'{"".join(rows)}'
        '</div>'
        '</div>'
    )


def _confirm_demo_data_source() -> None:
    """Confirm Step 1 for demo mode and move to the cohort setup step."""
    st.session_state.step1_confirmed = True


def _confirm_real_data_source() -> None:
    """Confirm Step 1 for a validated real-data source."""
    st.session_state.use_mock_data = False
    st.session_state.step1_confirmed = True
    st.session_state.step2_confirmed = False
    st.session_state.step3_confirmed = False
    st.session_state.export_completed = False


def _render_demo_preview_table(n_patients: int, hours: int) -> None:
    """Small deterministic vitals preview mirroring page-data-source.jsx."""
    lang = st.session_state.get("language", "en")
    sample_rows = [
        (20001, "00:00", 92, 82, 132, 96, 36.8, 18),
        (20001, "01:00", 95, 78, 128, 95, 37.0, 20),
        (20001, "02:00", 101, 70, 119, 93, 37.4, 24),
        (20002, "00:00", 78, 88, 144, 98, 36.5, 14),
        (20002, "01:00", 80, 86, 141, 98, 36.6, 15),
    ]
    row_html = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"muted" if idx == 0 else ""}">{html.escape(str(cell))}</td>'
            for idx, cell in enumerate(row)
        )
        + "</tr>"
        for row in sample_rows
    )
    headers = "".join(f"<th>{h}</th>" for h in ("stay_id", "time", "hr", "map", "sbp", "spo2", "temp", "resp"))
    with st.container(key="eu_demo_preview_card"):
        title_col, action_col = st.columns([1.0, 0.22], gap="small", vertical_alignment="center")
        with title_col:
            st.markdown(
                '<div class="eu-source-table-head-copy">'
                f'<div class="title">{html.escape("Sample preview · vitals module" if lang == "en" else "样本预览 · 生命体征模块")}</div>'
                f'<div class="sub">5 of {n_patients * hours:,} rows</div>'
                '</div>',
                unsafe_allow_html=True,
        )
        with action_col:
            modules_open = bool(st.session_state.get("_eu_demo_modules_open", False))
            if st.button(
                "Hide modules" if modules_open and lang == "en" else
                "收起模块" if modules_open else
                "View all modules" if lang == "en" else "查看全部模块",
                key="eu_demo_modules_toggle",
                use_container_width=True,
            ):
                st.session_state["_eu_demo_modules_open"] = not modules_open
                st.rerun()
        if st.session_state.get("_eu_demo_modules_open", False):
            with st.container(key="eu_demo_module_catalog_panel"):
                st.markdown(_demo_module_catalog_html(lang), unsafe_allow_html=True)
                if st.button(
                    "Continue to cohort setup" if lang == "en" else "继续到队列设置",
                    key="eu_demo_modules_continue",
                    type="primary",
                    use_container_width=True,
                    icon=":material/chevron_right:",
                ):
                    _confirm_demo_data_source()
                    st.rerun()
        st.markdown(
            '<div class="eu-source-table-scroll">'
            '<table>'
            f'<thead><tr>{headers}</tr></thead>'
            f'<tbody>{row_html}</tbody>'
            '</table>'
            '</div>',
            unsafe_allow_html=True,
        )


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to the extracted sidebar."""
    for name, value in app_context.items():
        if not name.startswith("__") and name not in _PROTECTED_CONTEXT_NAMES:
            globals()[name] = value


def _apply_sidebar_post_export_next_step(target: str, *, lang: str) -> None:
    """Route completed-export CTA buttons, using app helper when available."""
    app_helper = globals().get("_apply_post_export_next_step")
    if callable(app_helper):
        app_helper(st.session_state, target, lang=lang)
        return

    st.session_state["_post_export_guidance_dismissed"] = True
    st.session_state.pop("_post_export_navigation_pending", None)
    if target == "review":
        st.session_state["_active_main_page"] = "quick_viz"
        st.session_state["quick_viz_active_panel"] = "Data Tables"
        st.session_state["_scroll_to_top"] = True
    elif target == "cohort":
        st.session_state["_active_main_page"] = "cohort"
        st.session_state["_scroll_to_top"] = True
    elif target == "agent":
        clear_agent_continuation_state(st.session_state)
        st.session_state["_active_main_page"] = "research_agent"
        st.session_state["_ra_view"] = "setup"
        st.session_state["_scroll_to_top"] = True
        st.session_state["_eu_ra_focus_module_folder"] = True
        st.session_state["_eu_ra_module_pick_force_manual"] = True
        st.session_state["_eu_ra_apply_export_file_selection"] = True
        st.session_state.pop("research_agent_module_dir_pick", None)
        export_dir = str(st.session_state.get("last_export_dir") or st.session_state.get("export_path") or "")
        if export_dir:
            st.session_state["research_agent_module_dir_text"] = export_dir
        st.session_state["research_agent_cohort_source"] = (
            "选择 EasyICU 模块导出文件夹"
            if lang == "zh"
            else "Pick an EasyICU module export folder"
        )
    elif target == "dismiss":
        return
    else:
        raise ValueError(f"Unknown post-export next step: {target}")


def _ensure_default_directory_input_value(
    *,
    input_key: str,
    default_key: str,
    default_value: str,
) -> None:
    """Keep an empty/default directory input aligned without overwriting custom values."""
    previous_default = st.session_state.get(default_key)
    current_value = st.session_state.get(input_key)
    if not current_value:
        st.session_state[input_key] = default_value
    elif previous_default is not None and current_value == previous_default and previous_default != default_value:
        st.session_state[input_key] = default_value
    st.session_state[default_key] = default_value


def _hide_prefilled_directory_text(input_key: str, mirrored_value: str) -> None:
    pending_key = f"{input_key}__pending_value"
    current = str(st.session_state.get(input_key, "") or "")
    if pending_key in st.session_state:
        return
    if mirrored_value and current == str(mirrored_value):
        st.session_state[input_key] = ""


def _queue_sidebar_data_path_input(value: str) -> None:
    """Safely update the Step 1 data-path widget on the next rerun."""
    st.session_state["sidebar_data_path_input__pending_value"] = str(value or "")


def _real_data_source_ready() -> bool:
    """Real-data extraction can continue only after the current path validates."""
    data_path = str(st.session_state.get("data_path") or "").strip()
    if not data_path or not Path(data_path).exists():
        return False
    if not bool(st.session_state.get("path_validated")):
        return False
    last_validated = str(st.session_state.get("last_validated_path") or "").strip()
    if last_validated and Path(last_validated).expanduser() != Path(data_path).expanduser():
        return False
    return True


def _validation_resolved_path(validation_result: dict[str, Any], fallback: str) -> str:
    """Return the concrete database path resolved during validation."""
    for key in ("resolved_path", "csv_path"):
        value = str(validation_result.get(key) or "").strip()
        if value:
            return value
    return str(fallback or "").strip()


def _render_step1_data_source(entry_mode: str) -> None:
    """Render Step 1: data-source configuration.

    Two distinct UIs based on ``entry_mode``:

    - **demo**: a brief banner plus two sliders (patient count + duration);
      writes ``mock_params`` and toggles ``step1_confirmed`` via a confirm
      button.
    - **real**: database selector with auto-detection from the path,
      ``_directory_input`` for the data path, validate button (calls
      ``validate_database_path``), and conditional convert / download
      affordances based on the cached validation result.
    """
    if entry_mode == 'demo':
        # ===== DEMO 模式：按 easyicu design/page-data-source.jsx 渲染 =====
        lang = st.session_state.get("language", "en")
        _render_data_source_page_header(
            entry_mode,
            desc=(
                "Choose how patient data enters the workspace. You can switch later — sliders only affect demo runs."
                if lang == "en" else
                "选择患者数据如何进入工作区。稍后可以切换；滑块只影响演示运行。"
            ),
        )
        st.markdown(
            '<div class="eu-source-banner">'
            '<div class="banner-icon" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
            'stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M10 2v6.5L4.8 18.2A2.5 2.5 0 0 0 7 22h10a2.5 2.5 0 0 0 2.2-3.8L14 8.5V2"/>'
            '<path d="M8.5 2h7"/>'
            '<path d="M7.8 16h8.4"/>'
            '</svg>'
            '</div>'
            '<div class="banner-copy">'
            f'<div class="title">{html.escape("Demo mode" if lang == "en" else "演示模式")}</div>'
            f'<div class="sub">{html.escape("Automatically generates reproducible mock ICU data for tutorials and feature demos. No real database, token, or working directory is used." if lang == "en" else "自动生成可重复的模拟 ICU 数据，用于教程和功能演示。不连接任何真实数据库，token 与工作目录都不会被使用。")}</div>'
            '</div>'
            f'<span class="learn">{html.escape("Learn more" if lang == "en" else "了解更多")}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.session_state.database = 'mock'
        st.session_state.use_mock_data = True

        # 模拟数据参数
        n_patients_label = "Number of Patients" if st.session_state.language == 'en' else "患者数量"
        hours_label = "Data Duration (hours)" if st.session_state.language == 'en' else "数据时长(小时)"
        demo_patients_default = int(globals().get("LIGHTWEIGHT_DEMO_PATIENTS", 10))
        demo_hours_default = int(globals().get("LIGHTWEIGHT_DEMO_HOURS", 24))
        demo_patients_default = min(50, max(10, demo_patients_default))
        demo_hours_default = min(168, max(24, demo_hours_default))
        current_mock_params = st.session_state.get("mock_params") or {}

        def _clamped_int(value: Any, *, fallback: int, low: int, high: int) -> int:
            try:
                resolved = int(value)
            except (TypeError, ValueError):
                resolved = fallback
            return min(high, max(low, resolved))

        if 'demo_mode_patients' not in st.session_state:
            st.session_state.demo_mode_patients = _clamped_int(
                current_mock_params.get('n_patients'),
                fallback=demo_patients_default,
                low=10,
                high=50,
            )
        else:
            st.session_state.demo_mode_patients = _clamped_int(
                st.session_state.demo_mode_patients,
                fallback=demo_patients_default,
                low=10,
                high=50,
            )
        if 'demo_mode_hours' not in st.session_state:
            st.session_state.demo_mode_hours = _clamped_int(
                current_mock_params.get('hours'),
                fallback=demo_hours_default,
                low=24,
                high=168,
            )
        else:
            st.session_state.demo_mode_hours = _clamped_int(
                st.session_state.demo_mode_hours,
                fallback=demo_hours_default,
                low=24,
                high=168,
            )

        with st.container(key="eu_generation_card"):
            st.markdown(
                '<div class="eu-section-label" style="padding:0;margin:0 0 10px">'
                f'<span>{html.escape("Generation parameters" if lang == "en" else "生成参数")}</span></div>',
                unsafe_allow_html=True,
            )
            slider_cols = st.columns(2, gap="large")
            with slider_cols[0]:
                n_patients = st.slider(
                    n_patients_label,
                    10,
                    50,
                    key="demo_mode_patients",
                )
            with slider_cols[1]:
                hours = st.slider(
                    hours_label,
                    24,
                    168,
                    key="demo_mode_hours",
                )
            metric_html = "".join(
                '<div class="eu-source-metric">'
                f'<div class="label">{html.escape(label)}</div>'
                f'<div class="value">{html.escape(value)}</div>'
                f'<div class="sub">{html.escape(sub)}</div>'
                '</div>'
                for label, value, sub in [
                    (
                        "Expected stays" if lang == "en" else "预计 ICU stay",
                        f"{n_patients:,}",
                        "patients · single ICU stay" if lang == "en" else "患者 · 单次 ICU stay",
                    ),
                    (
                        "Time points" if lang == "en" else "时间点",
                        f"{n_patients * hours:,}",
                        f"hourly · {n_patients} × {hours}" if lang == "en" else f"每小时 · {n_patients} × {hours}",
                    ),
                    (
                        "Feature modules" if lang == "en" else "特征模块",
                        "19",
                        "vitals, labs, sofa, ..." if lang == "en" else "生命体征、化验、SOFA ...",
                    ),
                    (
                        "Estimated size" if lang == "en" else "预计大小",
                        f"~{max(0.8, n_patients * hours * 0.00044):.1f}",
                        "MB · in-memory" if lang == "en" else "MB · 内存中",
                    ),
                ]
            )
            st.markdown(f'<div class="eu-source-metrics">{metric_html}</div>', unsafe_allow_html=True)
        # 🔧 注意: mock_params 需要在 Step 2 (Cohort Selection) 之后更新
        # 这里只保存基本参数，cohort_filter 在 Step 2 之后的函数中动态获取
        st.session_state.mock_params = {'n_patients': n_patients, 'hours': hours}

        _render_demo_preview_table(n_patients, hours)

        # ✅ Step 1 确认按钮
        footer_l, reset_col, confirm_col = st.columns([5, 1.45, 2.25], gap="small")
        with footer_l:
            st.markdown(
                f'<div class="eu-source-footer-note">{html.escape("Step 1 of 4" if lang == "en" else "第 1 步 / 共 4 步")}</div>',
                unsafe_allow_html=True,
            )
        with reset_col:
            if st.button("Reset" if lang == "en" else "重置", use_container_width=True, key="step1_reset_demo"):
                st.session_state.demo_mode_patients = demo_patients_default
                st.session_state.demo_mode_hours = demo_hours_default
                st.session_state.mock_params = {"n_patients": demo_patients_default, "hours": demo_hours_default}
                st.rerun()
        step1_confirm_label = "Confirm data source" if st.session_state.language == 'en' else "确认数据源"
        with confirm_col:
            confirm_clicked = st.button(
                step1_confirm_label,
                type="primary",
                use_container_width=True,
                key="step1_confirm_demo",
                icon=":material/chevron_right:",
            )
        if confirm_clicked:
            _confirm_demo_data_source()
            st.rerun()

    elif entry_mode == 'real':
        # ===== REAL DATA 模式：只显示数据库选择，不显示Demo选项 =====
        _render_data_source_page_header(
            entry_mode,
            desc=(
                "Local-first. EasyICU never uploads data — paths and conversions happen on your machine."
                if st.session_state.language == "en" else
                "本地优先。EasyICU 不上传数据；路径验证和转换都在你的机器上完成。"
            ),
        )

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
        if st.session_state.get("data_path") and not st.session_state.get("sidebar_data_path_input"):
            st.session_state.sidebar_data_path_input = st.session_state.data_path
        data_path = _directory_input(
            path_label,
            value=st.session_state.data_path or "",
            input_key="sidebar_data_path_input",
            button_key="sidebar_data_path_browse",
            placeholder=_placeholder,
            help=_hint,
        )
        effective_data_path = data_path or st.session_state.get("data_path", "")

        # 🔧 当路径变化时自动检测并更新数据库
        if effective_data_path and effective_data_path != st.session_state.get('_last_data_path', ''):
            detected_db = detect_database_from_path(effective_data_path)
            if detected_db != database:
                st.session_state.database = detected_db
                st.session_state._last_data_path = effective_data_path
                st.rerun()
            st.session_state._last_data_path = effective_data_path

        # 验证按钮
        validate_btn = "Validate Data Path" if st.session_state.language == 'en' else "验证数据路径"
        validate_spacer, validate_action = st.columns([5, 1.8], gap="small")
        with validate_action:
            if st.button(validate_btn, use_container_width=True, key="validate_path", icon=":material/search:"):
                if not effective_data_path:
                    err_msg = "Please enter a data path" if st.session_state.language == 'en' else "请输入数据路径"
                    st.error(err_msg)
                elif not Path(effective_data_path).exists():
                    err_msg = "Path does not exist" if st.session_state.language == 'en' else "路径不存在"
                    st.error(err_msg)
                else:
                    # 检查数据库所需文件
                    validation_result = validate_database_path(effective_data_path, database)
                    resolved_data_path = _validation_resolved_path(validation_result, effective_data_path)
                    st.session_state.last_validation = validation_result

                    if validation_result['valid']:
                        st.session_state.data_path = resolved_data_path
                        _queue_sidebar_data_path_input(resolved_data_path)
                        st.session_state.last_validated_path = resolved_data_path
                        st.session_state.path_validated = True
                        st.session_state.step1_confirmed = False
                        st.session_state.step2_confirmed = False
                        st.session_state.step3_confirmed = False
                        st.session_state.export_completed = False
                        st.success(validation_result['message'])
                    else:
                        st.session_state.last_validated_path = effective_data_path
                        st.session_state.path_validated = False
                        st.session_state.step1_confirmed = False
                        st.error(validation_result['message'])
                    if validation_result.get('suggestion'):
                        st.info(validation_result['suggestion'])

        # 显示当前验证状态和转换按钮
        last_validation = st.session_state.get('last_validation', {})
        last_path = st.session_state.get('last_validated_path', '')

        if st.session_state.get('path_validated') and st.session_state.data_path == effective_data_path:
            validated_msg = "Path validated" if st.session_state.language == 'en' else "路径已验证"
            st.success(validated_msg)
        elif last_validation.get('can_convert') and last_path == effective_data_path:
            # 显示转换按钮
            convert_btn = "Convert & Setup" if st.session_state.language == 'en' else "转换并设置"
            if st.button(convert_btn, use_container_width=True, type="primary", key="convert_csv", icon=":material/sync:"):
                st.session_state.show_convert_dialog = True
                st.session_state.convert_source_path = _validation_resolved_path(last_validation, effective_data_path)
                st.rerun()
            convert_hint = "One-click: convert → validate → ready" if st.session_state.language == 'en' else "一键完成：转换 → 验证 → 就绪"
            st.caption(convert_hint)
            if last_validation.get('download_url'):
                st.link_button(
                    last_validation.get('download_label') or (
                        "Open database download page"
                        if st.session_state.language == 'en' else
                        "打开数据库下载页"
                    ),
                    last_validation['download_url'],
                    use_container_width=True,
                )
                if last_validation.get('download_note'):
                    st.caption(last_validation['download_note'])
        elif last_validation and (not last_validation.get('valid')) and last_path == effective_data_path:
            if last_validation.get('download_url'):
                st.link_button(
                    last_validation.get('download_label') or (
                        "Open database download page"
                        if st.session_state.language == 'en' else
                        "打开数据库下载页"
                    ),
                    last_validation['download_url'],
                    use_container_width=True,
                )
                if last_validation.get('download_note'):
                    st.caption(last_validation['download_note'])
        elif effective_data_path and Path(effective_data_path).exists():
            validate_hint = "Click the button above to validate data format" if st.session_state.language == 'en' else "点击上方按钮验证数据格式"
            st.caption(validate_hint)

        real_ready = _real_data_source_ready()
        footer_l, reset_col, confirm_col = st.columns([5, 1.45, 2.25], gap="small")
        with footer_l:
            st.markdown(
                f'<div class="eu-source-footer-note">{html.escape("Step 1 of 4" if st.session_state.language == "en" else "第 1 步 / 共 4 步")}</div>',
                unsafe_allow_html=True,
            )
        with reset_col:
            if st.button("Reset" if st.session_state.language == "en" else "重置", use_container_width=True, key="step1_reset_real"):
                st.session_state.data_path = ""
                _queue_sidebar_data_path_input("")
                st.session_state.path_validated = False
                st.session_state.last_validation = {}
                st.session_state.last_validated_path = ""
                st.session_state.step1_confirmed = False
                st.session_state.step2_confirmed = False
                st.session_state.step3_confirmed = False
                st.session_state.export_completed = False
                st.rerun()
        with confirm_col:
            confirm_clicked = st.button(
                "Confirm data source" if st.session_state.language == "en" else "确认数据源",
                type="primary",
                use_container_width=True,
                key="step1_confirm_real",
                icon=":material/chevron_right:",
                disabled=not real_ready,
            )
        if confirm_clicked and real_ready:
            _confirm_real_data_source()
            st.rerun()


def _shell_nav_items(entry_mode: str) -> list[ShellNavItem]:
    """Shell-A primary-nav items, matched to the page_registry."""
    lang = st.session_state.get("language", "en")
    if lang == "en":
        labels = {
            "tutorial": "Data Extraction",
            "quick_viz": "Patient Review",
            "cohort": "Cohort Statistics",
            "cross_db": "Cross-DB Benchmark",
            "research_agent": "Research Agent",
        }
    else:
        labels = {
            "tutorial": "数据提取",
            "quick_viz": "患者审阅",
            "cohort": "队列统计",
            "cross_db": "跨数据库对比",
            "research_agent": "研究智能体",
        }
    return [
        ShellNavItem(key="extract",        label=labels["tutorial"],       icon="extract", level="top"),
        ShellNavItem(key="quick_viz",      label=labels["quick_viz"],      icon="patient", level="child"),
        ShellNavItem(key="cohort",         label=labels["cohort"],         icon="layers", level="child"),
        ShellNavItem(key="cross_db",       label=labels["cross_db"],       icon="grid", level="child"),
        ShellNavItem(key="research_agent", label=labels["research_agent"], icon="agent", level="top"),
    ]


def _render_shell_brand(entry_mode: str) -> None:
    """Brand block at the top of the sidebar."""
    lang = st.session_state.get("language", "en")
    sub = "ICU Research Workspace" if lang == "en" else "ICU 数据研究台"
    with st.container(key="eu_brand_home"):
        st.markdown(
            render_brand_html(name="EasyICU", sub=sub, initials="E"),
            unsafe_allow_html=True,
        )
        if entry_mode != "none" and st.button(
            "Home",
            key="_eu_brand_home_button",
            help="Back to home" if lang == "en" else "返回首页",
            use_container_width=True,
        ):
            _return_to_entry_home()
            st.rerun()


def _context_summary_html(entry_mode: str, lang: str) -> str:
    """Return a plain-language summary of the data setup shown in the rail."""
    params = st.session_state.get("mock_params") or {}
    if entry_mode == "demo":
        data_value = (
            f"Demo · {params.get('n_patients', 100)} patients"
            if lang == "en" else
            f"演示 · {params.get('n_patients', 100)} 例"
        )
        mode_hint = "Demo" if lang == "en" else "演示"
    elif entry_mode == "real":
        db = st.session_state.get("database") or "real data"
        data_value = str(db).upper()
        mode_hint = "Real" if lang == "en" else "真实"
    else:
        data_value = "not selected" if lang == "en" else "未选择"
        mode_hint = "Not selected" if lang == "en" else "未选择"

    if st.session_state.get("step2_confirmed"):
        cohort_value = _format_step2_filter_meta(
            len(_active_step2_filter_chips(lang)),
            lang,
            empty_confirmed=True,
        )
    else:
        cohort_value = (
            "demo defaults" if entry_mode == "demo" and lang == "en" else
            "演示默认" if entry_mode == "demo" else
            "not configured" if lang == "en" else "未配置"
        )
    concepts = st.session_state.get("selected_concepts", []) or []
    concept_value = (
        f"{len(concepts)} selected" if concepts else
        ("demo defaults" if entry_mode == "demo" and lang == "en" else
         "演示默认" if entry_mode == "demo" else
         "auto defaults" if lang == "en" else "默认变量")
    )
    rows = [
        ("Dataset" if lang == "en" else "数据集", data_value),
        ("Cohort" if lang == "en" else "队列", cohort_value),
        ("Variables" if lang == "en" else "变量", concept_value),
    ]
    row_html = "".join(
        '<div class="eu-context-row">'
        f'<span>{html.escape(label)}</span>'
        f'<strong>{html.escape(str(value))}</strong>'
        '</div>'
        for label, value in rows
    )
    return (
        '<div class="eu-section-label eu-context-label">'
        f'<span>{html.escape("Current setup" if lang == "en" else "当前设置")}</span>'
        f'<span class="num">{html.escape(mode_hint)}</span>'
        '</div>'
        f'<div class="eu-context-card">{row_html}</div>'
    )


def _agent_sidebar_cohort_value(
    state: MutableMapping[str, Any],
    *,
    entry_mode: str,
    lang: str,
) -> str:
    """Summarize the cohort bound to Research Agent, not extraction setup."""
    cached_build = state.get("research_agent_module_built")
    if isinstance(cached_build, dict):
        built_df = cached_build.get("df")
        if isinstance(built_df, pd.DataFrame) and not built_df.empty:
            return (
                f"{len(built_df):,} rows"
                if lang == "en" else
                f"{len(built_df):,} 行"
            )

    inbound = state.get("research_agent_inbound_cohort")
    if isinstance(inbound, pd.DataFrame) and not inbound.empty:
        return (
            f"{len(inbound):,} rows"
            if lang == "en" else
            f"{len(inbound):,} 行"
        )

    loaded_concepts = state.get("loaded_concepts")
    if isinstance(loaded_concepts, dict) and loaded_concepts:
        patient_count = len(state.get("patient_ids") or [])
        if patient_count:
            return (
                f"{patient_count:,} loaded rows"
                if lang == "en" else
                f"已加载 {patient_count:,} 行"
            )
        return "loaded data" if lang == "en" else "已加载数据"

    params = state.get("mock_params") or {}
    if entry_mode == "demo":
        return (
            f"sepsis · {params.get('n_patients', 10)}"
            if lang == "en" else
            f"sepsis · {params.get('n_patients', 10)}"
        )
    return "not selected" if lang == "en" else "未选择"


def _agent_state_summary_html(entry_mode: str, lang: str) -> str:
    """Return the Research Agent-specific rail, matching the Claude reference."""
    state = st.session_state
    workbench = state.get("_agent_workbench")
    workbench = workbench if isinstance(workbench, dict) else {}
    audit = workbench.get("audit") if isinstance(workbench.get("audit"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    errors = int(counts.get("errors") or 0)
    warnings = int(counts.get("warnings") or 0)
    review = audit.get("review_decision") if isinstance(audit.get("review_decision"), dict) else {}
    review_status = str(review.get("decision") or review.get("status") or "").lower()
    reviewer_signed = review_status in {"approved", "accept", "accepted", "signed_off", "ready"}
    warning_review_pending = warnings > 0
    if warnings:
        try:
            from easyicu.webapp.agent_workbench import (
                _finding_review_id,
                _reviewable_findings,
            )

            reviewed_ids = set(state.get("_eu_wb_findings_acked") or [])
            warning_findings = [
                finding
                for finding in _reviewable_findings(audit)
                if str(finding.get("severity") or "").lower() == "warning"
            ]
            if warning_findings:
                warning_review_pending = any(
                    _finding_review_id(finding) not in reviewed_ids
                    for finding in warning_findings
                )
        except Exception:
            warning_review_pending = True
    blocked = any(
        isinstance(gate, dict) and gate.get("ok") is False
        for gate in (audit.get("gates") or [])
    )
    if errors:
        status_label = "Review" if lang == "en" else "复核"
        status_class = "warn"
    elif warning_review_pending:
        status_label = "Needs review" if lang == "en" else "需复核"
        status_class = "warn"
    elif warnings and not reviewer_signed:
        status_label = "Sign-off" if lang == "en" else "待签字"
        status_class = "warn"
    elif blocked:
        status_label = "Gate follow-up" if lang == "en" else "关口待跟进"
        status_class = "warn"
    else:
        status_label = "Ready" if lang == "en" else "就绪"
        status_class = "ready"

    run_id_raw = str(workbench.get("run_id") or "").strip()
    real_manifest_bound = bool(run_id_raw) and workbench.get("is_demo") is not True

    mode_value = (
        "Local run" if real_manifest_bound else
        "Demo" if entry_mode == "demo" else
        "Real" if entry_mode == "real" else
        "Not selected"
    )
    if lang != "en":
        mode_value = (
            "本机运行" if real_manifest_bound else
            "演示" if entry_mode == "demo" else
            "真实" if entry_mode == "real" else "未选择"
        )

    if real_manifest_bound:
        evidence_total = workbench.get("evidence_total")
        if evidence_total is None:
            evidence_items = workbench.get("evidence")
            evidence_total = len(evidence_items) if isinstance(evidence_items, list) else 0
        try:
            evidence_count = int(evidence_total or 0)
        except (TypeError, ValueError):
            evidence_count = 0
        cohort_value = (
            f"{evidence_count} evidence"
            if lang == "en" else
            f"{evidence_count} 条证据"
        )
    else:
        cohort_value = _agent_sidebar_cohort_value(
            state,
            entry_mode=entry_mode,
            lang=lang,
        )

    run_id = run_id_raw
    if not run_id and entry_mode == "demo":
        run_id = "preview" if lang == "en" else "预览"
    elif not run_id:
        run_id = "none" if lang == "en" else "无"
    if len(run_id) > 18:
        run_id = run_id[:7] + "…" + run_id[-6:]

    rows = [
        ("Mode" if lang == "en" else "模式", mode_value),
        (("Evidence" if lang == "en" else "证据") if real_manifest_bound else ("Cohort" if lang == "en" else "队列"), cohort_value),
        ("Last run" if lang == "en" else "最近运行", run_id),
    ]
    row_html = "".join(
        '<div class="eu-context-row">'
        f'<span>{html.escape(label)}</span>'
        f'<strong>{html.escape(str(value))}</strong>'
        '</div>'
        for label, value in rows
    )
    guarantees = [
        ("agent", "Local-first · no upload" if lang == "en" else "本地优先 · 不上传"),
        ("history", "Draft gated on evidence" if lang == "en" else "证据通过后才写作"),
        ("check", "Human confirms each run" if lang == "en" else "每次运行需人工确认"),
    ]
    guarantee_html = "".join(
        '<div class="eu-agent-guarantee-row">'
        f'<span>{_icon(icon_name)}</span>'
        f'<em>{html.escape(label)}</em>'
        '</div>'
        for icon_name, label in guarantees
    )
    return (
        '<div class="eu-section-label eu-context-label eu-agent-state-label">'
        f'<span>{html.escape("Agent state" if lang == "en" else "Agent 状态")}</span>'
        f'<span class="eu-agent-state-pill {status_class}"><span></span>{html.escape(status_label)}</span>'
        '</div>'
        f'<div class="eu-context-card eu-agent-state-card">{row_html}</div>'
        '<div class="eu-agent-guarantees">'
        f'<div class="eu-agent-guarantees-title">{html.escape("Guarantees" if lang == "en" else "保障")}</div>'
        f'{guarantee_html}'
        '</div>'
    )


def _render_shell_recent_cohorts() -> None:
    """Recent or demo cohort list (sidebar shell-A block)."""
    lang = st.session_state.get("language", "en")
    session_items = st.session_state.get("_eu_recent_cohorts")
    has_real_recent = bool(session_items)
    label = (
        ("Recent exports" if lang == "en" else "最近导出")
        if has_real_recent else
        ("Demo presets" if lang == "en" else "演示预设")
    )
    hint = (
        ("from this workspace" if lang == "en" else "来自当前工作区")
        if has_real_recent else
        ("samples" if lang == "en" else "示例")
    )
    st.markdown(
        '<div class="eu-section-label eu-recent-label" style="padding-top:0;margin-top:12px">'
        f'<span>{html.escape(label)}</span>'
        f'<span class="num">{html.escape(hint)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    items: list[tuple[str, str]] = session_items or [
        ("sepsis_mortality", "mock"),
        ("aki_kdigo_24h", "mock"),
        ("vent_weaning_72h", "mock"),
    ]
    for label_, count_ in items:
        st.markdown(
            '<div class="eu-recent-row">'
            '<span class="ico">'
            '<svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">'
            '<circle cx="12" cy="12" r="3"/></svg></span>'
            f'<span class="label">{html.escape(label_)}</span>'
            f'<span class="count mono">{html.escape(count_)}</span>'
            '</div>',
            unsafe_allow_html=True,
        )


def _render_shell_footer_icons() -> None:
    """Sidebar footer — 5 small icon controls (back / help / settings / lang / avatar).

    Uses Unicode glyphs instead of inline-SVG data URIs because the latter
    rendered as empty boxes in some Streamlit / Chromium combos.
    """
    lang = st.session_state.get("language", "en")
    with st.container(key="eu_sidebar_footer"):
        st.markdown(
            '<div class="eu-sidebar-footer-rule"></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 1, 1, 1, 1])
        with cols[0]:
            if st.button("", icon=":material/arrow_back:", key="_eu_footer_back", help=(
                    "Mode selection / 模式选择"
                    if lang == "en" else "返回模式选择 / Mode selection"),
                    use_container_width=True):
                _return_to_entry_home()
                st.rerun()
        with cols[1]:
            if st.button("", icon=":material/help:", key="_eu_footer_help", help=(
                    "Tutorial" if lang == "en" else "教程"),
                    use_container_width=True):
                st.session_state["_active_main_page"] = "tutorial"
                st.session_state["_main_nav_widget"] = "tutorial"
                st.session_state["_inline_ai_panel_open"] = False
                st.session_state["_floating_ai_open"] = False
                st.session_state.pop("_ai_pending_question", None)
                st.rerun()
        with cols[2]:
            if st.button("", icon=":material/settings:", key="_eu_footer_settings", help=(
                    "Settings" if lang == "en" else "设置"),
                    use_container_width=True):
                st.session_state["_active_main_page"] = "settings"
                st.session_state["_main_nav_widget"] = "settings"
                st.session_state["_scroll_to_top"] = True
                st.session_state["_inline_ai_panel_open"] = False
                st.session_state["_floating_ai_open"] = False
                st.session_state.pop("_ai_pending_question", None)
                st.rerun()
        with cols[3]:
            if st.button("中" if lang == "en" else "EN", icon=":material/language:", key="_eu_footer_lang", help=(
                    "Toggle 中 / EN" if lang == "en" else "切换 中 / EN"),
                    use_container_width=True):
                st.session_state["language"] = "zh" if lang == "en" else "en"
                st.rerun()
        with cols[4]:
            st.markdown(
                '<div style="height:28px;display:flex;align-items:center;justify-content:center">'
                '<div style="width:22px;height:22px;border-radius:999px;background:oklch(80% 0.05 70);'
                'display:flex;align-items:center;justify-content:center;font-size:12px;'
                'font-weight:500;color:var(--ink)">LK</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<style>'
        '.stApp [data-testid="stSidebar"] [class*="st-key-_eu_footer_"],'
        '.stApp [data-testid="stSidebar"] [class*="st-key-_eu_footer_"] .stButton{'
        'width:100% !important;'
        '}'
        '.stApp [data-testid="stSidebar"] [class*="st-key-_eu_footer_"] .stButton > button,'
        '.stApp [data-testid="stSidebar"] [class*="st-key-_eu_footer_"] button{'
        'width:100% !important; min-width:40px !important;'
        'min-height:30px !important; height:30px !important; padding:0 !important;'
        'border-radius:var(--r-2) !important;'
        'font-size:12px !important; font-weight:500 !important;'
        'color:var(--ink-3) !important; line-height:1;'
        'border:1px solid transparent !important;background:transparent !important;box-shadow:none !important;'
        '}'
        '.stApp [data-testid="stSidebar"] [class*="st-key-_eu_footer_"] .stButton > button:hover,'
        '.stApp [data-testid="stSidebar"] [class*="st-key-_eu_footer_"] button:hover{'
        'color:var(--ink) !important; background:var(--surface-2) !important;'
        '}'
        '</style>',
        unsafe_allow_html=True,
    )


def _render_sidebar_settings_panel() -> None:
    """Render the actual app settings exposed by the footer gear."""
    lang = st.session_state.get("language", "en")
    is_en = lang == "en"
    title = "Settings" if is_en else "设置"
    subtitle = (
        "Used by AI guidance and real research-agent runs."
        if is_en else
        "用于 AI 辅助和真实研究智能体运行。"
    )
    privacy = (
        "API keys are session-only. EasyICU does not write them to disk."
        if is_en else
        "API Key 只保存在当前会话，EasyICU 不会写入本地文件。"
    )
    st.markdown(
        '<div class="eu-settings-panel">'
        f'<div class="eu-settings-title">{html.escape(title)}</div>'
        f'<div class="eu-settings-subtitle">{html.escape(subtitle)}</div>'
        f'<div class="eu-settings-privacy">{html.escape(privacy)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eu-settings-section-label">'
        f'{html.escape("Language" if is_en else "语言")}'
        '</div>',
        unsafe_allow_html=True,
    )
    # Sidebar popovers cannot nest st.columns inside the columns row that
    # hosts the footer icons (Streamlit raises StreamlitAPIException). Stack
    # the two language buttons sequentially instead.
    if st.button(
        "English",
        key="_eu_settings_lang_en",
        type="primary" if lang == "en" else "secondary",
        use_container_width=True,
    ):
        st.session_state["language"] = "en"
        st.rerun()
    if st.button(
        "中文",
        key="_eu_settings_lang_zh",
        type="primary" if lang == "zh" else "secondary",
        use_container_width=True,
    ):
        st.session_state["language"] = "zh"
        st.rerun()

    st.markdown(
        '<div class="eu-settings-section-label">'
        f'{html.escape("AI / API connection" if is_en else "AI / API 连接")}'
        '</div>',
        unsafe_allow_html=True,
    )
    try:
        from easyicu.webapp.llm_chat import render_llm_settings
        render_llm_settings(show_status_card=False, controls_only=True)
    except Exception as exc:
        st.caption(
            ("AI settings are unavailable: " if is_en else "AI 设置暂不可用：")
            + str(exc)
        )


def _render_shell_primary_nav() -> None:
    """Render the three-level product flow: extraction, visualization, agent."""
    entry_mode = st.session_state.get("entry_mode", "none")
    items = {item.key: item for item in _shell_nav_items(entry_mode)}
    lang = st.session_state.get("language", "en")
    active = st.session_state.get("_active_main_page", "tutorial")
    visual_active = active
    visualization_keys = ["quick_viz", "cohort", "cross_db"]
    visualization_label = "Data Visualization" if lang == "en" else "数据可视化"

    def _render_nav_button(item: ShellNavItem) -> None:
        is_active = item.key == visual_active
        with st.container(key=f"eunavrow_{item.key}"):
            st.markdown(render_nav_item_html(item, active=is_active), unsafe_allow_html=True)
            if st.button(
                item.label,
                key=f"euonav_{item.key}",
                use_container_width=True,
            ):
                st.session_state["_active_main_page"] = item.key
                st.session_state["_main_nav_widget"] = item.key
                st.session_state["_inline_ai_panel_open"] = False
                st.session_state["_floating_ai_open"] = False
                st.session_state.pop("_ai_pending_question", None)
                st.rerun()

    def _render_visualization_menu() -> None:
        expanded_key = "_eu_visualization_nav_open"
        expanded = bool(st.session_state.get(expanded_key)) or visual_active in visualization_keys
        trigger = ShellNavItem(
            key="visualization",
            label=visualization_label,
            icon="bars",
            count="⌄" if expanded else "›",
            level="top",
        )
        with st.container(key="eunavrow_visualization"):
            st.markdown(
                render_nav_item_html(trigger, active=False),
                unsafe_allow_html=True,
            )
            if st.button(
                visualization_label,
                key="euonav_visualization",
                use_container_width=True,
            ):
                st.session_state[expanded_key] = not bool(st.session_state.get(expanded_key))
                st.rerun()
        if expanded:
            with st.container(key="eunavchildren_visualization"):
                for key in visualization_keys:
                    _render_nav_button(items[key])

    _render_nav_button(items["extract"])
    _render_visualization_menu()
    _render_nav_button(items["research_agent"])


def _render_shell_aux_nav() -> None:
    """Render the Tools / Reference groups from the latest design shell."""
    lang = st.session_state.get("language", "en")
    active = st.session_state.get("_active_main_page", "tutorial")

    labels = {
        "tools": "Tools" if lang == "en" else "工具",
        "reference": "Reference" if lang == "en" else "参考",
        "assistant": "AI Assistant" if lang == "en" else "AI 助手",
        "tutorial": "Get Started" if lang == "en" else "开始使用",
        "states": "Workspace States" if lang == "en" else "工作区状态",
    }

    def _render_group_label(text: str) -> None:
        st.markdown(
            f'<div class="eu-nav-group-label design-section">{html.escape(text)}</div>',
            unsafe_allow_html=True,
        )

    def _render_aux_button(item: ShellNavItem, *, active_state: bool) -> None:
        with st.container(key=f"eunavrow_{item.key}"):
            st.markdown(render_nav_item_html(item, active=active_state), unsafe_allow_html=True)
            if st.button(
                item.label,
                key=f"euonav_{item.key}",
                use_container_width=True,
            ):
                if item.key == "assistant":
                    st.session_state["_active_main_page"] = "assistant"
                    st.session_state["_main_nav_widget"] = "assistant"
                    st.session_state["_inline_ai_panel_open"] = False
                    st.session_state["_floating_ai_open"] = False
                    st.session_state["_scroll_to_top"] = True
                else:
                    st.session_state["_active_main_page"] = item.key
                    st.session_state["_main_nav_widget"] = item.key
                    st.session_state["_inline_ai_panel_open"] = False
                    st.session_state["_floating_ai_open"] = False
                    st.session_state.pop("_ai_pending_question", None)
                st.rerun()

    _render_group_label(labels["tools"])
    _render_aux_button(
        ShellNavItem(key="assistant", label=labels["assistant"], icon="sparkles", level="top"),
        active_state=active == "assistant",
    )
    _render_aux_button(
        ShellNavItem(key="tutorial", label=labels["tutorial"], icon="help", level="top"),
        active_state=active == "tutorial",
    )

    _render_group_label(labels["reference"])
    _render_aux_button(
        ShellNavItem(key="states", label=labels["states"], icon="grid", level="top"),
        active_state=active == "states",
    )


def _compute_pipeline_steps() -> list[PipelineStep]:
    """Compute the 4-step extraction pipeline status from session_state."""
    lang = st.session_state.get("language", "en")
    entry_mode = st.session_state.get("entry_mode", "none")
    s1 = bool(st.session_state.get("step1_confirmed"))
    s2 = bool(st.session_state.get("step2_confirmed"))
    s3 = bool(st.session_state.get("step3_confirmed"))
    s4 = bool(st.session_state.get("export_completed"))

    def _status(prev_done: bool, this_done: bool) -> str:
        if this_done:
            return "done"
        if prev_done:
            return "active"
        return "todo"

    if lang == "en":
        titles = {
            "data": "Data source",
            "cohort": "Cohort",
            "concepts": "Concepts",
            "export": "Export",
        }
    else:
        titles = {
            "data": "数据源",
            "cohort": "队列",
            "concepts": "概念变量",
            "export": "导出",
        }

    db = st.session_state.get("database", "")
    if entry_mode == "demo":
        params = st.session_state.get("mock_params") or {}
        n_pat = params.get("n_patients", 100)
        data_meta = f"Demo · {n_pat} patients" if lang == "en" else f"演示 · {n_pat} 例"
    elif db:
        data_meta = f"{db.upper()}"
    else:
        data_meta = "—"

    filter_n = len(_active_step2_filter_chips(lang))
    cohort_meta = _format_step2_filter_meta(filter_n, lang, empty_confirmed=s2)

    selected = st.session_state.get("selected_concepts", []) or []
    concept_meta = (
        f"{len(selected)} features" if lang == "en" else f"{len(selected)} 个特征"
    ) if selected else ("auto from cohort" if lang == "en" else "随队列自动")

    export_meta = ("completed" if lang == "en" else "已完成") if s4 else (
        "ready" if (s3 and not s4) else ("locked" if lang == "en" else "待解锁")
    )

    return [
        PipelineStep("data",     titles["data"],     data_meta,    _status(True, s1)),
        PipelineStep("cohort",   titles["cohort"],   cohort_meta,  _status(s1, s2)),
        PipelineStep("concepts", titles["concepts"], concept_meta, _status(s2, s3)),
        PipelineStep("export",   titles["export"],   export_meta,  _status(s3, s4)),
    ]


def _sidebar_extract_step_unlocked(state: dict[str, Any], step: int) -> bool:
    """Return whether a sidebar pipeline step can be opened."""
    if step <= 1:
        return True
    if step == 2:
        return bool(state.get("step1_confirmed"))
    if step == 3:
        return bool(state.get("step1_confirmed") and state.get("step2_confirmed"))
    return bool(
        state.get("step1_confirmed")
        and state.get("step2_confirmed")
        and state.get("step3_confirmed")
    )


def _sidebar_set_extract_step_state(state: dict[str, Any], step: int) -> None:
    """Move the extraction workflow back to an unlocked step."""
    step = max(1, min(4, int(step)))
    state["_active_main_page"] = "extract"
    state["step1_confirmed"] = step > 1
    state["step2_confirmed"] = step > 2
    state["step3_confirmed"] = step > 3
    if step < 4:
        state["export_completed"] = False


def _pipeline_step_html(step: PipelineStep, *, unlocked: bool) -> str:
    status = step.status
    dot_inner = _icon("check") if status == "done" else ""
    click_state = "unlocked" if unlocked else "locked"
    return (
        f'<div class="eu-pipe-step {status} {click_state}">'
        f'<div class="dot">{dot_inner}</div>'
        '<div class="body">'
        f'<div class="title">{html.escape(step.title)}</div>'
        f'<div class="meta">{html.escape(step.meta)}</div>'
        '</div>'
        '</div>'
    )


def _pipeline_step_button_label(step: PipelineStep) -> str:
    """Return a compact two-line label without Markdown code-chip styling."""
    return f"**{step.title}**  \n{step.meta}" if step.meta else f"**{step.title}**"


def _render_interactive_pipeline_block(steps: list[PipelineStep]) -> None:
    lang = st.session_state.get("language", "en")
    done_n = sum(1 for s in steps if s.status == "done")
    st.markdown(
        f'<div class="eu-section-label"><span>{html.escape("Data extraction" if lang == "en" else "数据提取")}</span>'
        f'<span class="num">{done_n} / {len(steps)}</span></div>',
        unsafe_allow_html=True,
    )

    for index, step in enumerate(steps, start=1):
        unlocked = _sidebar_extract_step_unlocked(st.session_state, index)
        state_key = "unlocked" if unlocked else "locked"
        step_icon = {
            "done": ":material/check_circle:",
            "active": ":material/radio_button_checked:",
        }.get(step.status, ":material/radio_button_unchecked:")
        help_text = (
            ("Open this completed step" if step.status == "done" else "Open this step")
            if lang == "en" else
            ("返回修改这一步" if step.status == "done" else "打开这一步")
        ) if unlocked else (
            "Complete the previous steps first" if lang == "en" else "请先完成前面的步骤"
        )
        with st.container(key=f"eu_pipeline_step_{step.status}_{state_key}_{step.key}"):
            if st.button(
                _pipeline_step_button_label(step),
                key=f"eu_pipeline_jump_{step.key}",
                disabled=not unlocked,
                icon=step_icon,
                help=help_text,
                use_container_width=True,
            ):
                _sidebar_set_extract_step_state(st.session_state, index)
                st.rerun()


def _render_shell_context_body() -> None:
    """Render extraction pipeline or compact data context inside the dock.

    The full 4-step extraction state is useful on the Extract page, but it
    reads like unrelated project state on Tutorial / analysis pages.
    """
    lang = st.session_state.get("language", "en")
    active = st.session_state.get("_active_main_page", "tutorial")
    if active == "extract":
        _render_interactive_pipeline_block(_compute_pipeline_steps())
        return

    entry_mode = st.session_state.get("entry_mode", "none")
    if active == "research_agent":
        st.markdown(_agent_state_summary_html(entry_mode, lang), unsafe_allow_html=True)
        return

    st.markdown(_context_summary_html(entry_mode, lang), unsafe_allow_html=True)
    if st.button(
        "Edit setup" if lang == "en" else "编辑配置",
        key="eu_context_edit_setup",
        use_container_width=True,
        icon=":material/tune:",
    ):
        st.session_state["_active_main_page"] = "extract"
        st.rerun()


def _render_shell_context_dock(entry_mode: str) -> None:
    """Lower sidebar dock: extraction progress or compact setup context."""
    with st.container(key="eu_sidebar_dock"):
        _render_shell_context_body()


def _render_sidebar_top(entry_mode: str) -> None:
    """Render the static top of the sidebar.

    Includes:
      - sidebar expand/collapse toggle
      - "Back to Mode Selection" button (only when an entry mode is active)
      - mode badge (Demo / Real Data pill)
      - app title
      - language selector
      - AI assistant settings panel (delegated to ``llm_chat.render_llm_settings``)
      - Data Extraction title heading (always last so the body knows where to
        start)
    """
    # 展开/收起按钮
    _, expand_col2 = st.columns([3, 1])
    with expand_col2:
        if st.session_state.sidebar_expanded:
            expand_label = "⬅️"
            expand_help = "Collapse sidebar" if st.session_state.language == 'en' else "收起侧边栏"
        else:
            expand_label = "⤢"
            expand_help = "Expand to full width" if st.session_state.language == 'en' else "展开到全屏"

        if st.button(expand_label, key="toggle_sidebar_expand", help=expand_help):
            st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
            st.rerun()

    # 返回入口页面按钮（始终显示，除非在入口页）
    if entry_mode != 'none':
        back_label = "🔙 Back to Mode Selection" if st.session_state.language == 'en' else "🔙 返回模式选择"
        if st.button(back_label, key="back_to_entry", use_container_width=True):
            _return_to_entry_home()
            st.rerun()

        # Workflow help — Tutorial is no longer a top tab (2026-05 Phase A
        # redesign); reach it from here instead. Toggles the active main
        # page to the still-routable Tutorial page; "Back" returns to the
        # previous page via the standard tab bar.
        help_label = (
            "📚 Workflow Help" if st.session_state.language == 'en'
            else "📚 工作流帮助"
        )
        if st.button(help_label, key="open_tutorial",
                     use_container_width=True,
                     help=("Open the data-preparation tutorial"
                           if st.session_state.language == 'en'
                           else "打开数据准备教程")):
            st.session_state['_active_main_page'] = 'tutorial'
            st.session_state['_main_nav_widget'] = 'tutorial'
            st.session_state['_inline_ai_panel_open'] = False
            st.session_state['_floating_ai_open'] = False
            st.session_state.pop('_ai_pending_question', None)
            st.rerun()
        st.markdown("---")

    # NOTE (shell-A redesign, 2026-05-21): the legacy mode pill and app
    # title are now rendered by ``_render_shell_brand`` at the top of
    # the sidebar; suppressed here to avoid duplication. The language
    # selector and AI assistant panel still live in the rail so they
    # remain reachable without expanding any subsection.

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

    # AI 助手设置（放在侧边栏最上方，方便用户看到）
    try:
        from easyicu.webapp.llm_chat import render_llm_settings
        render_llm_settings()
    except Exception:
        pass  # silently skip if module unavailable

    st.markdown("---")

    # 侧边栏仅用于数据提取导出模式
    sidebar_title = "📤 Data Extraction" if st.session_state.language == 'en' else "📤 数据提取导出"
    st.markdown(f"### {sidebar_title}")


def _render_export_completed_panel() -> bool:
    """Render the post-export success panel (status / restart / back-home).

    Shown in place of Step 1-4 when ``st.session_state.export_completed`` is
    True. Returns ``True`` if the panel was rendered (caller should stop the
    rest of sidebar render), ``False`` otherwise.
    """
    if not st.session_state.get('export_completed', False):
        return False

    lang = st.session_state.get("language", "en")
    export_dir = st.session_state.get('last_export_dir', '')
    result = st.session_state.get('_export_success_result', {})
    files = [str(path) for path in result.get('files', [])] if result else []
    n_files = len(files)
    n_patients = int(result.get('patient_count', 0) or 0) if result else 0
    n_concepts = int(result.get('concept_count', 0) or 0) if result else 0
    manifest_files = [str(path) for path in result.get('manifest_files', [])] if result else []
    total_time = float(result.get('total_time', 0) or 0) if result else 0.0
    duration = f"{total_time:.1f}s" if total_time else "local"
    file_summary = (
        f"{n_files} files" if lang == "en" else f"{n_files} 个文件"
    )
    patient_summary = (
        f"{n_patients:,} patients" if n_patients else ("patient count recorded in manifest" if lang == "en" else "患者数已写入清单")
    )
    concept_summary = (
        f"{n_concepts:,} concepts" if n_concepts else ("selected concepts" if lang == "en" else "已选概念")
    )
    success_title = "Export complete" if lang == "en" else "导出完成"
    success_desc = (
        f"{file_summary} + reproducibility manifest written to the local export folder. Everything stayed on your machine."
        if lang == "en"
        else f"{file_summary} 与可复现清单已写入本地导出文件夹。全程未离开本机。"
    )
    export_path_html = (
        f'<span class="mono">{html.escape(str(export_dir))}</span>' if export_dir else ""
    )
    st.markdown(
        f"""
        <div class="eu-export-complete-hero">
          <div class="glyph">✓</div>
          <div class="st-t">{html.escape(success_title)}</div>
          <div class="st-d">{html.escape(success_desc)} {export_path_html}</div>
          <div class="eu-export-complete-stats">
            <span><b>{html.escape(file_summary)}</b><small>{html.escape("tables" if lang == "en" else "表格")}</small></span>
            <span><b>{html.escape(patient_summary)}</b><small>{html.escape("cohort" if lang == "en" else "队列")}</small></span>
            <span><b>{html.escape(concept_summary)}</b><small>{html.escape("features" if lang == "en" else "特征")}</small></span>
            <span><b>{html.escape(duration)}</b><small>{html.escape("runtime" if lang == "en" else "耗时")}</small></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ledger_items: list[tuple[str, str, str]] = []
    for path in files[:4]:
        file_path = Path(path)
        ledger_items.append((file_path.name, "exported table" if lang == "en" else "导出表", "file"))
    if len(files) > 4:
        ledger_items.append((f"+ {len(files) - 4} more files", "" if lang == "en" else "", "more"))
    for path in manifest_files[:2]:
        ledger_items.append((Path(path).name, "reproducibility manifest" if lang == "en" else "可复现清单", "manifest"))
    if not ledger_items and export_dir:
        ledger_items.append((Path(str(export_dir)).name, "local export folder" if lang == "en" else "本地导出文件夹", "folder"))
    ledger_html = "".join(
        '<div class="eu-export-ledger-row">'
        f'<span class="{html.escape(kind)}"></span>'
        f'<div><b>{html.escape(title)}</b><small>{html.escape(desc)}</small></div>'
        "</div>"
        for title, desc, kind in ledger_items
    )
    if ledger_html:
        st.markdown(f'<div class="eu-export-ledger-grid">{ledger_html}</div>', unsafe_allow_html=True)

    # 显示导出统计
    if result:
        if result.get('note'):
            st.markdown(
                f'<div class="compact-inline-notice info">{html.escape(str(result["note"]))}</div>',
                unsafe_allow_html=True,
            )

    # 显示队列筛选统计
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
        st.markdown(
            f'<div class="compact-inline-notice info">{html.escape(cohort_info)}</div>',
            unsafe_allow_html=True,
        )

    next_msg = (
        "Next, review the exported tables, run cohort statistics, or use this folder as the Research Agent cohort source."
        if lang == "en"
        else "下一步可以审阅导出表、做队列统计，或把这个文件夹作为研究智能体的队列来源。"
    )
    st.markdown(
        f'<div class="compact-inline-notice info">{html.escape(next_msg)}</div>',
        unsafe_allow_html=True,
    )
    next_cols = st.columns(3, gap="small")
    with next_cols[0]:
        if st.button(
            "Review tables" if lang == "en" else "查看表格",
            key="post_export_completed_open_review",
            use_container_width=True,
            icon=":material/table_view:",
        ):
            _apply_sidebar_post_export_next_step("review", lang=lang)
            st.rerun()
    with next_cols[1]:
        if st.button(
            "Cohort stats" if lang == "en" else "队列统计",
            key="post_export_completed_open_cohort",
            use_container_width=True,
            icon=":material/query_stats:",
        ):
            _apply_sidebar_post_export_next_step("cohort", lang=lang)
            st.rerun()
    with next_cols[2]:
        if st.button(
            "Research Agent" if lang == "en" else "研究智能体",
            key="post_export_completed_open_agent",
            use_container_width=True,
            icon=":material/auto_awesome:",
        ):
            _apply_sidebar_post_export_next_step("agent", lang=lang)
            st.rerun()

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
        st.session_state.step3_confirmed = False
        st.session_state[_STEP2_RESET_PENDING_KEY] = True
        st.session_state.selected_concepts = []
        st.session_state.concept_checkboxes = {}
        st.session_state.selected_groups = []
        st.session_state.pop('_eu_concept_defaults_seeded', None)
        st.session_state.loaded_concepts = {}
        st.session_state.loaded_data_origin = 'none'
        # 🔧 FIX (2026-02-15): 重置采样参数，避免上次提取的 patient_limit/patient_ids 泄露到新提取
        st.session_state.patient_limit = 0  # 重置为默认值：全量患者
        st.session_state.patient_ids = []
        st.session_state.all_patient_count = 0
        st.session_state.pop('_viz_auto_load_export', None)
        st.session_state.pop('_post_export_guidance_dismissed', None)
        st.session_state.pop('_post_export_navigation_pending', None)
        st.session_state.pop('_post_export_target_panel', None)
        st.session_state.pop('_export_success_result', None)
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

    return True


def _render_step2_cohort_selection() -> bool:
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
        if use_mock:
            step_dep_msg = (
                "ℹ️ Confirm Step 1 to configure the extraction workflow. Cohort demo panels are already available in Cohort Analysis."
                if st.session_state.language == 'en' else
                "ℹ️ 如需配置提取流程，请先确认步骤1。队列分析的演示面板已可直接使用。"
            )
            st.info(step_dep_msg)
        else:
            # 提示用户先完成Step1
            step_dep_msg = "⚠️ Please complete Step 1 first" if st.session_state.language == 'en' else "⚠️ 请先完成步骤1"
            st.warning(step_dep_msg)
        return False  # 不渲染后续内容

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
            'disease_cohort': 'none',
            'icd_query': '',
            'icd_include_query': '',
            'icd_exclude_query': '',
            'icd_mode': 'include',
        }
    # Upgrade older sessions that only had a single ICD box.
    if 'icd_include_query' not in st.session_state.cohort_filter:
        legacy_query = str(st.session_state.cohort_filter.get('icd_query', '')).strip()
        legacy_mode = st.session_state.cohort_filter.get('icd_mode', 'include')
        st.session_state.cohort_filter['icd_include_query'] = legacy_query if legacy_mode != 'exclude' else ''
        st.session_state.cohort_filter['icd_exclude_query'] = legacy_query if legacy_mode == 'exclude' else ''
    else:
        st.session_state.cohort_filter.setdefault('icd_exclude_query', '')
    # Keep legacy keys for backward compatibility with existing export/session logic.
    st.session_state.cohort_filter.setdefault('icd_query', '')
    st.session_state.cohort_filter.setdefault('icd_mode', 'include')
    st.session_state.setdefault('sepsis_si_mode', 'auto')
    st.session_state.setdefault('sepsis_abx_win_hours', 24)
    st.session_state.setdefault('sepsis_samp_win_hours', 72)
    st.session_state.setdefault('sepsis_positive_cultures', False)
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
        # 紧凑年龄筛选：0 / 120 代表不限制，避免单独 expander 占高
        age_col1, age_col2 = st.columns(2)
        with age_col1:
            age_min = st.number_input(
                "🎂 Min Age" if st.session_state.language == 'en' else "🎂 最小年龄",
                min_value=0,
                max_value=120,
                value=0 if st.session_state.cohort_filter['age_min'] is None else int(st.session_state.cohort_filter['age_min']),
                key="cohort_age_min"
            )
            st.session_state.cohort_filter['age_min'] = age_min if age_min > 0 else None
        with age_col2:
            age_max = st.number_input(
                "🎂 Max Age" if st.session_state.language == 'en' else "🎂 最大年龄",
                min_value=0,
                max_value=120,
                value=120 if st.session_state.cohort_filter['age_max'] is None else int(st.session_state.cohort_filter['age_max']),
                key="cohort_age_max"
            )
            st.session_state.cohort_filter['age_max'] = age_max if age_max < 120 else None
        age_hint = "0 / 120 means no age limit" if st.session_state.language == 'en' else "0 / 120 表示不限制年龄"
        st.caption(age_hint)

        # 紧凑筛选布局：下拉框替代横向 radio，减少截图高度
        first_icu_label = "🏥 First ICU" if st.session_state.language == 'en' else "🏥 首次 ICU"
        first_icu_options = {
            'any': 'Any' if st.session_state.language == 'en' else '不限',
            'yes': 'Yes (First ICU only)' if st.session_state.language == 'en' else '是（仅首次）',
            'no': 'No (Readmissions only)' if st.session_state.language == 'en' else '否（仅再入院）',
        }
        current_first_icu = st.session_state.cohort_filter.get('first_icu_stay')
        current_first_icu_key = 'any'
        if current_first_icu is True:
            current_first_icu_key = 'yes'
        elif current_first_icu is False:
            current_first_icu_key = 'no'

        los_label = "⏱️ Min ICU Stay (hours)" if st.session_state.language == 'en' else "⏱️ 最短住院时长（小时）"
        los_help = "Minimum ICU stay duration to include patients (default 0h = no lower bound)" if st.session_state.language == 'en' else "纳入患者的最短ICU住院时长（默认0小时，表示不设下限）"

        compact_row1_col1, compact_row1_col2 = st.columns([1.3, 1.0])
        with compact_row1_col1:
            first_icu_val = st.selectbox(
                first_icu_label,
                options=list(first_icu_options.keys()),
                format_func=lambda x: first_icu_options[x],
                index=list(first_icu_options.keys()).index(current_first_icu_key),
                key="cohort_first_icu"
            )
        with compact_row1_col2:
            los_min = st.number_input(
                los_label, min_value=0, max_value=10000,
                value=0 if st.session_state.cohort_filter.get('los_min') is None else int(st.session_state.cohort_filter['los_min']),
                help=los_help,
                key="cohort_los_min"
            )

        if first_icu_val == 'yes':
            st.session_state.cohort_filter['first_icu_stay'] = True
        elif first_icu_val == 'no':
            st.session_state.cohort_filter['first_icu_stay'] = False
        else:
            st.session_state.cohort_filter['first_icu_stay'] = None

        st.session_state.cohort_filter['los_min'] = los_min if los_min > 0 else None
        st.session_state.cohort_filter['los_max'] = None  # 不再使用max

        gender_label = "👤 Gender" if st.session_state.language == 'en' else "👤 性别"
        gender_options = {
            'any': 'Any' if st.session_state.language == 'en' else '不限',
            'M': 'Male' if st.session_state.language == 'en' else '男性',
            'F': 'Female' if st.session_state.language == 'en' else '女性',
        }
        current_gender_key = st.session_state.cohort_filter.get('gender') or 'any'

        survival_label = "💚 Survival Status" if st.session_state.language == 'en' else "💚 存活状态"
        survival_options = {
            'any': 'Any' if st.session_state.language == 'en' else '不限',
            'survived': 'Survived' if st.session_state.language == 'en' else '存活',
            'deceased': 'Deceased' if st.session_state.language == 'en' else '死亡',
        }
        current_survival = st.session_state.cohort_filter.get('survived')
        current_survival_key = 'any'
        if current_survival is True:
            current_survival_key = 'survived'
        elif current_survival is False:
            current_survival_key = 'deceased'

        compact_row2_col1, compact_row2_col2 = st.columns(2)
        with compact_row2_col1:
            gender_val = st.selectbox(
                gender_label,
                options=list(gender_options.keys()),
                format_func=lambda x: gender_options[x],
                index=list(gender_options.keys()).index(current_gender_key),
                key="cohort_gender"
            )
        with compact_row2_col2:
            survival_val = st.selectbox(
                survival_label,
                options=list(survival_options.keys()),
                format_func=lambda x: survival_options[x],
                index=list(survival_options.keys()).index(current_survival_key),
                key="cohort_survival"
            )

        st.session_state.cohort_filter['gender'] = gender_val if gender_val != 'any' else None

        if survival_val == 'survived':
            st.session_state.cohort_filter['survived'] = True
        elif survival_val == 'deceased':
            st.session_state.cohort_filter['survived'] = False
        else:
            st.session_state.cohort_filter['survived'] = None

        # 疾病队列筛选（任务导向）
        disease_label = "🩺 Clinical Cohort" if st.session_state.language == 'en' else "🩺 疾病队列"
        supported_diseases = _get_supported_disease_cohorts(st.session_state.get('database', 'miiv'))
        disease_options_en = {
            'none': 'Any / No disease filter',
            'sepsis': 'Sepsis-3 cohort',
            'aki': 'AKI cohort (KDIGO)',
            'circ_failure': 'Circulatory failure cohort',
            'mech_vent': 'Mechanical ventilation cohort',
            'rrt': 'Renal replacement therapy cohort',
            'ards': 'ARDS cohort',
            'pneumonia': 'Pneumonia cohort',
            'heart_failure': 'Heart failure cohort',
            'ami': 'AMI cohort',
            'stroke': 'Stroke cohort',
        }
        disease_options_zh = {
            'none': '不限 / 不做疾病筛选',
            'sepsis': '脓毒症队列（Sepsis-3）',
            'aki': 'AKI 队列（KDIGO）',
            'circ_failure': '循环衰竭队列',
            'mech_vent': '机械通气队列',
            'rrt': '肾脏替代治疗队列',
            'ards': 'ARDS 队列',
            'pneumonia': '肺炎队列',
            'heart_failure': '心力衰竭队列',
            'ami': '急性心肌梗死队列',
            'stroke': '卒中队列',
        }
        disease_options = disease_options_en if st.session_state.language == 'en' else disease_options_zh
        current_disease = st.session_state.cohort_filter.get('disease_cohort', 'none')
        if current_disease not in supported_diseases:
            current_disease = 'none'
        disease_choice = st.selectbox(
            disease_label,
            options=supported_diseases,
            format_func=lambda x: disease_options.get(x, x),
            index=supported_diseases.index(current_disease),
            key="cohort_disease_cohort",
        )
        st.session_state.cohort_filter['disease_cohort'] = disease_choice
        st.session_state.cohort_filter['has_sepsis'] = True if disease_choice == 'sepsis' else None

        if disease_choice != 'none':
            disease_cfg = DISEASE_COHORT_CONFIG.get(disease_choice, {})
            disease_desc = disease_cfg.get('description_en') if st.session_state.language == 'en' else disease_cfg.get('description_zh')
            if disease_desc:
                st.caption(disease_desc)
            if disease_cfg.get('icd_tokens'):
                if _supports_icd_filter(st.session_state.get('database')):
                    icd_note = (
                        f"Template ICD prefixes: {', '.join(disease_cfg['icd_tokens'])}"
                        if st.session_state.language == 'en' else
                        f"模板 ICD 前缀：{', '.join(disease_cfg['icd_tokens'])}"
                    )
                    st.caption(icd_note)
                else:
                    _no_icd_warn = (
                        f"⚠️ This cohort requires ICD codes, but {st.session_state.get('database', '')} "
                        "does not have ICD tables. The disease filter will not take effect."
                        if st.session_state.language == 'en' else
                        f"⚠️ 此队列需要 ICD 编码，但 {st.session_state.get('database', '')} "
                        "没有 ICD 诊断表。疾病筛选将不会生效。"
                    )
                    st.warning(_no_icd_warn)

        if disease_choice == 'sepsis':
            sepsis_title = "🦠 Sepsis suspected-infection settings" if st.session_state.language == 'en' else "🦠 脓毒症疑似感染设置"
            with st.expander(sepsis_title, expanded=False):
                sepsis_mode_cfg = {
                    key: (value['label_en'] if st.session_state.language == 'en' else value['label_zh'])
                    for key, value in SEPSIS_MODE_CONFIG.items()
                }
                sepsis_mode = st.selectbox(
                    "Detection mode" if st.session_state.language == 'en' else "判定模式",
                    options=list(SEPSIS_MODE_CONFIG.keys()),
                    format_func=lambda x: sepsis_mode_cfg[x],
                    index=list(SEPSIS_MODE_CONFIG.keys()).index(st.session_state.get('sepsis_si_mode', 'auto')),
                    key="sepsis_si_mode",
                )
                mode_desc = SEPSIS_MODE_CONFIG[sepsis_mode]['desc_en'] if st.session_state.language == 'en' else SEPSIS_MODE_CONFIG[sepsis_mode]['desc_zh']
                st.caption(mode_desc)

                sepsis_col1, sepsis_col2 = st.columns(2)
                with sepsis_col1:
                    st.number_input(
                        "ABX → sampling window (hours)" if st.session_state.language == 'en' else "抗生素后采样时间窗（小时）",
                        min_value=1,
                        max_value=168,
                        value=int(st.session_state.get('sepsis_abx_win_hours', 24)),
                        key="sepsis_abx_win_hours",
                    )
                with sepsis_col2:
                    st.number_input(
                        "Sampling → ABX window (hours)" if st.session_state.language == 'en' else "采样后抗生素时间窗（小时）",
                        min_value=1,
                        max_value=168,
                        value=int(st.session_state.get('sepsis_samp_win_hours', 72)),
                        key="sepsis_samp_win_hours",
                    )

                st.checkbox(
                    "Require positive cultures only" if st.session_state.language == 'en' else "仅使用阳性培养",
                    value=bool(st.session_state.get('sepsis_positive_cultures', False)),
                    key="sepsis_positive_cultures",
                )
                sepsis_ref = (
                    "Reference note: EasyICU supports multiple suspected-infection definitions. "
                    "The strict ABX + sampling window follows common Sepsis-3 operational logic; "
                    "eICU defaults to ICD + antibiotics because microbiology coverage is sparse."
                    if st.session_state.language == 'en' else
                    "参考说明：EasyICU 支持多种疑似感染定义。严格的“抗生素 + 采样”时间窗对应常见的 Sepsis-3 操作化逻辑；"
                    "eICU 默认采用“ICD + 抗生素”，因为微生物采样覆盖较稀疏。"
                )
                st.info(sepsis_ref)
                _render_sepsis_ai_button(st.session_state.language)

        if _supports_icd_filter(st.session_state.get('database')):
            icd_include_label = "🧾 ICD include" if st.session_state.language == 'en' else "🧾 ICD 包含"
            icd_exclude_label = "🧾 ICD exclude" if st.session_state.language == 'en' else "🧾 ICD 排除"
            icd_help = (
                "Use commas for multiple ICD prefixes, e.g. A41,A42. Simple ranges like A41-42 are also supported. For eICU, keywords also work."
                if st.session_state.language == 'en' else
                "对于 MIMIC 数据库，可用逗号输入多个 ICD 前缀，如 A41,A42；也支持 A41-42 这种简单范围写法。对于 eICU，也可输入关键词。"
            )
            icd_col1, icd_col2 = st.columns(2)
            with icd_col1:
                icd_include_value = st.text_input(
                    icd_include_label,
                    value=st.session_state.cohort_filter.get('icd_include_query', ''),
                    help=icd_help,
                    key="cohort_icd_include_query",
                    placeholder="A41,A42 or A41-42" if st.session_state.language == 'en' else "A41,A42 或 A41-42",
                )
                st.session_state.cohort_filter['icd_include_query'] = icd_include_value.strip()
            with icd_col2:
                icd_exclude_value = st.text_input(
                    icd_exclude_label,
                    value=st.session_state.cohort_filter.get('icd_exclude_query', ''),
                    help=icd_help,
                    key="cohort_icd_exclude_query",
                    placeholder="I50,C34 or I50-51" if st.session_state.language == 'en' else "I50,C34 或 I50-51",
                )
                st.session_state.cohort_filter['icd_exclude_query'] = icd_exclude_value.strip()

            # 兼容旧逻辑：保留 legacy 单字段，但优先使用 include/exclude
            st.session_state.cohort_filter['icd_query'] = st.session_state.cohort_filter['icd_include_query']
            st.session_state.cohort_filter['icd_mode'] = 'include'

            icd_preview_label = "🔍 Preview ICD Match" if st.session_state.language == 'en' else "🔍 预览 ICD 匹配"
            preview_specs = [
                ("include", icd_include_value.strip(), "include"),
                ("exclude", icd_exclude_value.strip(), "exclude"),
            ]
            has_preview_input = any(
                _split_query_tokens(preview_value) for _, preview_value, _ in preview_specs
            )
            if has_preview_input and st.session_state.get('data_path'):
                if st.button(icd_preview_label, key="icd_preview_btn_combined"):
                    for preview_key, preview_value, _preview_mode in preview_specs:
                        _icd_tokens_preview = _split_query_tokens(preview_value)
                        if not _icd_tokens_preview:
                            st.session_state.pop(f'_icd_preview_cache_{preview_key}', None)
                            continue
                        _icd_preview_result = _preview_icd_match(
                            Path(st.session_state.data_path),
                            st.session_state.get('database', 'miiv'),
                            _icd_tokens_preview,
                        )
                        st.session_state[f'_icd_preview_cache_{preview_key}'] = _icd_preview_result
        else:
            st.session_state.cohort_filter['icd_query'] = ""
            st.session_state.cohort_filter['icd_include_query'] = ""
            st.session_state.cohort_filter['icd_exclude_query'] = ""
            st.session_state.cohort_filter['icd_mode'] = 'include'
            _clear_icd_preview_state()

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
        if cf.get('disease_cohort') and cf.get('disease_cohort') != 'none':
            disease_cfg = DISEASE_COHORT_CONFIG.get(cf['disease_cohort'], {})
            disease_label_summary = disease_cfg.get('label_en') if st.session_state.language == 'en' else disease_cfg.get('label_zh')
            if disease_label_summary:
                filter_summary.append(disease_label_summary)
        if _supports_icd_filter(st.session_state.get('database')) and cf.get('icd_include_query'):
            filter_summary.append(
                f"ICD include: {cf['icd_include_query']}"
                if st.session_state.language == 'en' else
                f"ICD 包含: {cf['icd_include_query']}"
            )
        if _supports_icd_filter(st.session_state.get('database')) and cf.get('icd_exclude_query'):
            filter_summary.append(
                f"ICD exclude: {cf['icd_exclude_query']}"
                if st.session_state.language == 'en' else
                f"ICD 排除: {cf['icd_exclude_query']}"
            )

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
            _clear_icd_preview_state()
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

    return True


def _render_step3_concept_selection(concept_groups: dict[str, list[str]]) -> list[str] | None:
    """Render Step 3 and return selected concepts, or None when blocked."""
    step3_title = "Step 3: Select Features" if st.session_state.language == 'en' else "步骤3: 选择特征"
    st.markdown(f"### 🔧 {step3_title}")

    # 🔧 FIX (2026-02-05): 检查步骤依赖 - Step2必须先确认，否则不显示特征选择
    step2_complete = st.session_state.get('step2_confirmed', False)
    if not step2_complete:
        # 提示用户先完成Step2，不显示后续内容
        step_dep_msg = "⚠️ Please complete Step 2 first (click Confirm Cohort Selection button)" if st.session_state.language == 'en' else "⚠️ 请先完成步骤2（点击确认队列筛选按钮）"
        st.warning(step_dep_msg)
        return None  # 不再显示Step 3的内容

    # 初始化 session state
    if 'concept_checkboxes' not in st.session_state:
        st.session_state.concept_checkboxes = {}
    if 'selected_groups' not in st.session_state:
        st.session_state.selected_groups = []
    if not st.session_state.selected_groups and not st.session_state.get("_eu_concept_defaults_seeded"):
        _reset_concepts_to_groups(concept_groups, _all_concept_groups(concept_groups))

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
        if st.button(all_label, key="select_all_groups", use_container_width=True):
            _reset_concepts_to_groups(concept_groups, _all_concept_groups(concept_groups))
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

        _detail_concept_count = sum(
            len(concept_groups.get(g, []))
            for g in st.session_state.selected_groups
            if g in concept_groups
        )
        detail_label = (
            f"🎯 Feature Detail Configuration ({_detail_concept_count})"
            if st.session_state.language == 'en'
            else f"🎯 特征详细配置（{_detail_concept_count}）"
        )
        # Collapsed by default: with all categories selected this expander
        # holds hundreds of checkboxes and would otherwise dominate the sidebar.
        # It is an advanced step (de-selecting individual concepts) — the
        # category chips above already cover the common case.
        with st.expander(detail_label, expanded=False):
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
            preview_desc = (
                "Load a small sample into Quick Visualization. This is optional and does not replace the final export."
                if st.session_state.language == 'en' else
                "将一小部分样本加载到快速可视化。此步骤可选，不会替代最终导出。"
            )
            st.caption(preview_desc)
            preview_patient_options = [10, 20, 50, 100]
            if st.session_state.get('preview_n_patients') not in preview_patient_options:
                st.session_state.pop('preview_n_patients', None)

            preview_slider_kwargs = {
                'options': preview_patient_options,
                'key': "preview_n_patients",
            }
            if 'preview_n_patients' not in st.session_state:
                preview_slider_kwargs['value'] = 10

            preview_n = st.select_slider(
                get_text('preview_patients'),
                **preview_slider_kwargs,
            )

            preview_btn_label = "👁️ Run Quick Preview" if st.session_state.language == 'en' else "👁️ 运行快速预览"
            if st.button(preview_btn_label, key="sidebar_preview_btn", use_container_width=True):
                st.session_state['_preview_requested'] = True
                st.session_state['_preview_n'] = preview_n
                st.session_state['_scroll_to_tab'] = 'viz'
                st.rerun()
            if st.session_state.get('_preview_requested', False):
                preview_pending_msg = (
                    "⏳ Quick Preview is loading. The Quick Visualization tab will open when the sample is ready."
                    if st.session_state.language == 'en' else
                    "⏳ 正在加载快速预览。样本准备好后会自动打开快速可视化标签页。"
                )
                st.info(preview_pending_msg)
    else:
        # 如果没有选中任何概念，重置确认状态
        st.session_state.step3_confirmed = False

    st.markdown("---")
    return selected_concepts


def _render_step4_export(selected_concepts: list[str]) -> bool:
    """Render Step 4 and return whether the sidebar should keep rendering."""
    lang = st.session_state.get("language", "en")
    step4_title = "Package & export" if lang == 'en' else "打包并导出"
    step4_desc = (
        "Export the extracted concept data and a reproducible manifest to a local folder. Code, figures, and the evidence ledger come from a Research Agent run."
        if lang == "en" else
        "将提取后的概念数据和可复现实验清单导出到本地文件夹；代码、图和证据账本由 Research Agent 运行生成。"
    )
    st.markdown(
        '<div class="eu-export-header">'
        f'<h1>{html.escape(step4_title)}</h1>'
        f'<p>{html.escape(step4_desc)}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    cancel_notice = st.session_state.pop("_export_cancel_notice", None)
    if cancel_notice:
        st.warning(str(cancel_notice))

    # 🔧 FIX (2026-02-05): 检查步骤依赖 - Step3必须先确认（点击确认选择按钮）
    step3_complete = st.session_state.get('step3_confirmed', False) and len(st.session_state.get('selected_concepts', [])) > 0
    if not step3_complete:
        # 提示用户先完成Step3并点击确认按钮
        step_dep_msg = "⚠️ Please complete Step 3 first (select features and click Confirm Selection)" if st.session_state.language == 'en' else "⚠️ 请先完成步骤3（选择特征并点击确认选择）"
        st.warning(step_dep_msg)
        # 不再继续显示Step4的内容
        return False

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
    cohort_suffix = _generate_cohort_prefix()
    default_dir_name = f"{db_name}_{timestamp_suffix}"
    if cohort_suffix:
        default_dir_name = f"{default_dir_name}_{cohort_suffix[:48]}"
    default_export_path = str(Path(base_export_path) / default_dir_name)
    export_input_key = "sidebar_export_path_input"
    export_default_key = "_sidebar_export_path_default"
    _ensure_default_directory_input_value(
        input_key=export_input_key,
        default_key=export_default_key,
        default_value=st.session_state.get("export_path", default_export_path),
    )

    st.session_state.setdefault("export_format", "Parquet")
    if st.session_state.export_format not in {"Parquet", "CSV", "Excel"}:
        st.session_state.export_format = "Parquet"
    patient_limit_options = [0, 100, 1000, 5000, 10000, 20000, 50000]
    if st.session_state.get('patient_limit', 0) not in patient_limit_options:
        st.session_state.patient_limit = 0

    settings_col, review_col = st.columns([1.08, 0.92], gap="large")
    with settings_col:
        st.markdown(
            '<div class="eu-export-contents-card">'
            f'<div class="eu-section-label"><span>{html.escape("Export contents" if lang == "en" else "导出内容")}</span></div>'
            '<div class="eu-export-content-row on">'
            f'<div><b>{html.escape("Stay-level tables" if lang == "en" else "住院级表格")}</b><span>{html.escape("One row per stay · selected concepts" if lang == "en" else "每次住院一行 · 已选概念")}</span></div><em>on</em>'
            '</div>'
            '<div class="eu-export-content-row on">'
            f'<div><b>{html.escape("Per-hour frames" if lang == "en" else "逐小时数据帧")}</b><span>{html.escape("Hourly time series · selected format" if lang == "en" else "小时级时间序列 · 当前格式")}</span></div><em>on</em>'
            '</div>'
            '<div class="eu-export-content-row on">'
            f'<div><b>{html.escape("Reproducibility manifest" if lang == "en" else "可复现清单")}</b><span>{html.escape("Cohort, concept, format, and local provenance metadata" if lang == "en" else "队列、概念、格式和本地来源元数据")}</span></div><em>on</em>'
            '</div>'
            '<div class="eu-export-content-row muted">'
            f'<div><b>{html.escape("Agent code, figures, evidence ledger" if lang == "en" else "Agent 代码、图和证据账本")}</b><span>{html.escape("Generated later in Research Agent from this exported cohort" if lang == "en" else "后续由 Research Agent 基于本队列生成")}</span></div><em>agent</em>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="eu_export_settings_card"):
            st.markdown(
                '<div class="eu-export-card-head">'
                f'<span>{html.escape("Destination" if lang == "en" else "导出位置")}</span>'
                f'<small>{html.escape(db_name.upper())}</small>'
                '</div>',
                unsafe_allow_html=True,
            )
            export_path = _directory_input(
                "Export path" if lang == 'en' else "导出路径",
                value=default_export_path,
                input_key=export_input_key,
                button_key="sidebar_export_path_browse",
                placeholder="Select export directory" if lang == 'en' else "选择导出目录",
                help=(
                    f"Data will be exported to this directory (Current database: {db_name.upper()})"
                    if lang == 'en' else
                    f"数据将导出到此目录（当前数据库: {db_name.upper()}）"
                ),
            )
            export_path = export_path or st.session_state.get("export_path", default_export_path)
            st.session_state.export_path = export_path
            path_exists = bool(export_path and Path(export_path).exists())
            if path_exists:
                st.markdown(
                    '<div class="eu-export-status ok">'
                    f'<span>✓</span>{html.escape("Path valid" if lang == "en" else "路径有效")}'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                missing_text = "Path does not exist" if lang == "en" else "路径不存在"
                st.markdown(
                    '<div class="eu-export-status warn">'
                    f'<span>!</span>{html.escape(missing_text)}'
                    '</div>',
                    unsafe_allow_html=True,
                )
                create_col, hint_col = st.columns([0.42, 0.58], gap="small")
                with create_col:
                    create_btn = "Create folder" if lang == 'en' else "创建目录"
                    if st.button(create_btn, key="create_export_dir", use_container_width=True):
                        try:
                            Path(export_path).mkdir(parents=True, exist_ok=True)
                            st.rerun()
                        except Exception as e:
                            err_msg = f"Creation failed: {e}" if lang == 'en' else f"创建失败: {e}"
                            st.error(err_msg)
                with hint_col:
                    st.caption(
                        "EasyICU can create the final folder before exporting."
                        if lang == "en" else
                        "EasyICU 可在导出前创建最终目录。"
                    )

            st.markdown(
                '<div class="eu-export-control-label">'
                f'{html.escape("Export format" if lang == "en" else "导出格式")}'
                '<small>Parquet recommended</small>'
                '</div>',
                unsafe_allow_html=True,
            )
            fmt_cols = st.columns(3, gap="small")
            for fmt_col, fmt in zip(fmt_cols, ["Parquet", "CSV", "Excel"]):
                with fmt_col:
                    if st.button(
                        fmt,
                        key=f"eu_export_format_{fmt.lower()}",
                        type="primary" if st.session_state.export_format == fmt else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.export_format = fmt
                        st.rerun()

            st.markdown(
                '<div class="eu-export-control-label">'
                f'{html.escape("Patient limit" if lang == "en" else "患者数量限制")}'
                f'<small>{html.escape("All patients for final runs" if lang == "en" else "正式运行建议全部患者")}</small>'
                '</div>',
                unsafe_allow_html=True,
            )
            limit_labels = {
                0: "All" if lang == "en" else "全部",
                100: "100",
                1000: "1k",
                5000: "5k",
                10000: "10k",
                20000: "20k",
                50000: "50k",
            }
            limit_cols = st.columns(len(patient_limit_options), gap="small")
            for limit_col, limit_value in zip(limit_cols, patient_limit_options):
                with limit_col:
                    if st.button(
                        limit_labels[limit_value],
                        key=f"eu_export_limit_{limit_value}",
                        type="primary" if st.session_state.patient_limit == limit_value else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.patient_limit = limit_value
                        st.rerun()

    export_format = st.session_state.export_format
    patient_limit = st.session_state.patient_limit

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

    _n_feats = len(selected_concepts)
    _n_mods = len(set(g for g, cs in CONCEPT_GROUPS_INTERNAL.items() if any(c in selected_concepts for c in cs)))
    _pat_str = str(patient_limit) if patient_limit and patient_limit > 0 else ("All" if lang == 'en' else "全部")
    _selected_groups = [
        g for g, cs in CONCEPT_GROUPS_INTERNAL.items() if any(c in selected_concepts for c in cs)
    ]
    if not _selected_groups and st.session_state.get("selected_groups"):
        _selected_groups = list(st.session_state.get("selected_groups", []))
    module_chip_html = "".join(
        f'<span>{html.escape(_module_display_name(group, lang))}</span>'
        for group in _selected_groups[:8]
    )
    if len(_selected_groups) > 8:
        module_chip_html += f'<em>+{len(_selected_groups) - 8}</em>'
    if not module_chip_html:
        module_chip_html = f'<em>{html.escape("No modules" if lang == "en" else "无模块")}</em>'
    path_exists = bool(export_path and Path(export_path).exists())
    cohort_filter_meta = _format_step2_filter_meta(
        len(_active_step2_filter_chips(lang)),
        lang,
        empty_confirmed=True,
    )
    check_items = [
        ("source", "Demo data" if use_mock else db_name.upper(), True),
        ("cohort", cohort_filter_meta, True),
        ("features", f"{_n_feats} selected", _n_feats > 0),
        ("path", "ready" if path_exists else "missing", path_exists),
    ]
    check_html = "".join(
        '<div class="eu-export-check">'
        f'<span class="{"ok" if ok else "warn"}">{"✓" if ok else "!"}</span>'
        f'<b>{html.escape(label)}</b>'
        f'<em>{html.escape(value)}</em>'
        '</div>'
        for label, value, ok in check_items
    )
    with review_col:
        st.markdown(
            '<div class="eu-export-review-card">'
            '<div class="eu-card-head">'
            f'<span>{html.escape("Bundle review" if lang == "en" else "导出包复核")}</span>'
            f'<small>{html.escape("Step 4 of 4" if lang == "en" else "第 4 / 4 步")}</small>'
            '</div>'
            '<div class="eu-export-summary-grid">'
            f'<div><small>{html.escape("Patients" if lang == "en" else "患者")}</small><strong>{html.escape(_pat_str)}</strong></div>'
            f'<div><small>{html.escape("Features" if lang == "en" else "特征")}</small><strong>{_n_feats}</strong></div>'
            f'<div><small>{html.escape("Modules" if lang == "en" else "模块")}</small><strong>{_n_mods}</strong></div>'
            f'<div><small>{html.escape("Format" if lang == "en" else "格式")}</small><strong>{html.escape(export_format)}</strong></div>'
            '</div>'
            '<div class="eu-export-path">'
            f'<small>{html.escape("Export path" if lang == "en" else "导出路径")}</small>'
            f'<code>{html.escape(str(export_path))}</code>'
            '</div>'
            f'<div class="eu-export-checklist">{check_html}</div>'
            '<div class="eu-export-module-strip">'
            f'<small>{html.escape("Selected modules" if lang == "en" else "已选模块")}</small>'
            f'<div>{module_chip_html}</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    export_btn = "Export data" if lang == 'en' else "导出数据"
    if st.session_state.get('_exporting_in_progress', False):
        export_pending_msg = (
            "⏳ Export has started. EasyICU is re-extracting data from the source database, not reusing the preview sample."
            if st.session_state.language == 'en' else
            "⏳ 导出已经开始。EasyICU 正在从源数据库重新提取数据，不会直接复用刚才的预览样本。"
        )
        st.info(export_pending_msg)
    with st.container(key="eu_export_footer_actions"):
        st.markdown(
            f'<div class="eu-source-footer-note">{html.escape("Step 4 of 4" if lang == "en" else "第 4 步 / 共 4 步")}</div>',
            unsafe_allow_html=True,
        )
        footer_left, footer_mid, footer_right = st.columns([1, 0.9, 1.05], gap="small")
        with footer_mid:
            if st.button(
                get_text('sanity_back'),
                use_container_width=True,
                key="sanity_back_btn",
                icon=":material/arrow_back:",
            ):
                st.session_state.step3_confirmed = False
                st.rerun()
        with footer_right:
            if st.button(
                get_text('sanity_confirm') if can_export else export_btn,
                type="primary",
                use_container_width=True,
                key="final_export_btn",
                disabled=not can_export,
                icon=":material/check:",
            ):
                st.session_state.trigger_export = True
                st.session_state.export_completed = False
                st.session_state['_exporting_in_progress'] = True
                st.session_state['_scroll_to_tab'] = 'export_progress'
                st.rerun()
    if not can_export:
        missing = []
        if not selected_concepts:
            missing.append("features" if lang == "en" else "特征")
        if not path_exists:
            missing.append("valid path" if lang == "en" else "有效路径")
        if not use_mock and not st.session_state.data_path:
            missing.append("data source" if lang == "en" else "数据源")
        warn_msg = (
            "Missing: " + ", ".join(missing)
            if lang == "en" else
            "缺少：" + "、".join(missing)
        )
        st.caption(warn_msg)

    return True


def _render_system_resource_panel() -> None:
    """Render the collapsible Performance / system-resource info block.

    Sits at the very bottom of the sidebar. Reads ``get_system_resources``
    from the installed app context and shows a brief CPU/RAM summary with
    auto-optimized worker recommendations.
    """
    resources = get_system_resources()
    lang = st.session_state.get("language", "en")
    title = "Performance" if lang == "en" else "性能配置"
    items = [
        ("CPU", f"{resources['cpu_count']} cores" if lang == "en" else f"{resources['cpu_count']} 核"),
        ("RAM", f"{resources['available_memory_gb']} / {resources['total_memory_gb']} GB"),
        ("Workers", str(resources['recommended_workers']) if lang == "en" else f"{resources['recommended_workers']} 并行"),
        ("Backend", str(resources['recommended_backend'])),
    ]
    item_html = "".join(
        '<div>'
        f'<small>{html.escape(label)}</small>'
        f'<strong>{html.escape(value)}</strong>'
        '</div>'
        for label, value in items
    )
    st.markdown(
        '<div class="eu-performance-strip">'
        '<div class="eu-card-head">'
        f'<span>{html.escape(title)}</span>'
        f'<small>{html.escape("auto-optimized" if lang == "en" else "自动优化")}</small>'
        '</div>'
        f'<div class="eu-performance-grid">{item_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _ensure_step2_state_defaults() -> None:
    """Initialize cohort-builder state without overwriting existing choices."""
    _apply_pending_step2_reset()

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
            'disease_cohort': 'none',
            'icd_query': '',
            'icd_include_query': '',
            'icd_exclude_query': '',
            'icd_mode': 'include',
        }
    cf = st.session_state.cohort_filter
    if 'icd_include_query' not in cf:
        legacy_query = str(cf.get('icd_query', '')).strip()
        legacy_mode = cf.get('icd_mode', 'include')
        cf['icd_include_query'] = legacy_query if legacy_mode != 'exclude' else ''
        cf['icd_exclude_query'] = legacy_query if legacy_mode == 'exclude' else ''
    cf.setdefault('icd_exclude_query', '')
    cf.setdefault('icd_query', '')
    cf.setdefault('icd_mode', 'include')
    st.session_state.setdefault('sepsis_si_mode', 'auto')
    st.session_state.setdefault('sepsis_abx_win_hours', 24)
    st.session_state.setdefault('sepsis_samp_win_hours', 72)
    st.session_state.setdefault('sepsis_positive_cultures', False)
    if 'cohort_enabled' not in st.session_state:
        st.session_state.cohort_enabled = st.session_state.get("entry_mode") == "demo"
    if 'filtered_patient_count' not in st.session_state:
        st.session_state.filtered_patient_count = None


def _apply_pending_step2_reset() -> None:
    """Apply a queued Step 2 reset before Step 2 widgets are rendered."""
    if st.session_state.pop(_STEP2_RESET_PENDING_KEY, False):
        st.session_state.pop('cohort_filter', None)
        for widget_key in _STEP2_WIDGET_KEYS:
            st.session_state.pop(widget_key, None)
        st.session_state.cohort_enabled = st.session_state.get("entry_mode") == "demo"
        st.session_state.filtered_patient_count = None
        clear_icd_preview = globals().get("_clear_icd_preview_state")
        if callable(clear_icd_preview):
            clear_icd_preview()


def _step2_filter_chips(lang: str) -> list[str]:
    """Return compact localized chips for the currently active filters."""
    cf = st.session_state.get('cohort_filter', {})
    chips: list[str] = []
    if cf.get('age_min') is not None or cf.get('age_max') is not None:
        chips.append(
            f"age {cf.get('age_min') or 0}-{cf.get('age_max') or 120}"
            if lang == "en" else
            f"年龄 {cf.get('age_min') or 0}-{cf.get('age_max') or 120}"
        )
    if cf.get('first_icu_stay') is True:
        chips.append("first stay only" if lang == "en" else "仅首次 ICU")
    elif cf.get('first_icu_stay') is False:
        chips.append("readmissions only" if lang == "en" else "仅再入院")
    if cf.get('los_min') is not None:
        chips.append(f">={cf['los_min']}h ICU" if lang == "en" else f"ICU >={cf['los_min']}小时")
    if cf.get('gender') is not None:
        chips.append(f"sex {cf['gender']}" if lang == "en" else f"性别 {cf['gender']}")
    if cf.get('survived') is True:
        chips.append("survived" if lang == "en" else "存活")
    elif cf.get('survived') is False:
        chips.append("deceased" if lang == "en" else "死亡")
    disease = cf.get('disease_cohort')
    if disease and disease != 'none':
        disease_cfg = DISEASE_COHORT_CONFIG.get(disease, {})
        disease_label = disease_cfg.get('label_en') if lang == "en" else disease_cfg.get('label_zh')
        chips.append(disease_label or disease)
    if cf.get('icd_include_query'):
        chips.append(f"ICD + {cf['icd_include_query']}")
    if cf.get('icd_exclude_query'):
        chips.append(f"ICD - {cf['icd_exclude_query']}")
    return chips


def _active_step2_filter_chips(lang: str) -> list[str]:
    """Return effective Step 2 chips, treating disabled filtering as all stays."""
    if st.session_state.get("cohort_enabled") is False:
        return []
    return _step2_filter_chips(lang)


def _format_step2_filter_meta(
    filter_n: int,
    lang: str,
    *,
    empty_confirmed: bool,
) -> str:
    """Format cohort filter counts without exposing internal sentinel defaults."""
    if filter_n:
        if lang == "en":
            return f"{filter_n} filter" if filter_n == 1 else f"{filter_n} filters"
        return f"{filter_n} 条筛选"
    if empty_confirmed:
        return "all stays" if lang == "en" else "所有 stay"
    return "not set" if lang == "en" else "未设置"


def _step2_database_display_name(database: str) -> str:
    """Return the compact clinical name shown in the Step 2 builder."""
    return {
        'miiv': 'MIMIC-IV',
        'eicu': 'eICU-CRD',
        'aumc': 'AmsterdamUMCdb',
        'hirid': 'HiRID',
        'mimic': 'MIMIC-III',
        'sic': 'SICdb',
    }.get(database, database.upper())


def _render_real_cohort_preview_pending(lang: str, chips: list[str]) -> None:
    """Render an honest pre-extraction preview for local-data workflows."""
    chips_html = "".join(
        f'<span class="eu-cohort-chip">{html.escape(chip)}<span>×</span></span>'
        for chip in chips
    ) or f'<span class="eu-cohort-empty">{html.escape("No active filters" if lang == "en" else "无启用筛选")}</span>'
    database = _step2_database_display_name(st.session_state.get('database', ''))
    pending_label = "pending extraction" if lang == "en" else "等待提取"
    pending_title = "Preview available after extraction" if lang == "en" else "完成提取后可查看预览"
    pending_body = (
        "Counts, distributions, and sample stays appear after EasyICU applies this cohort recipe to the local tables."
        if lang == "en" else
        "EasyICU 将队列配方应用到本地数据表后，才会显示人数、分布和 stay 样本。"
    )
    source_label = f"Local source · {database}" if lang == "en" else f"本地数据源 · {database}"
    st.markdown(
        '<div class="eu-cohort-preview-stack">'
        '<div class="eu-cohort-preview-card">'
        '<div class="preview-head"><div>'
        f'<div class="title">{html.escape("Live preview" if lang == "en" else "实时预览")}</div>'
        f'<div class="sub">{html.escape(source_label)}</div>'
        '</div>'
        f'<span class="eu-pill"><span class="dot"></span>{html.escape(pending_label)}</span></div>'
        '<div class="eu-cohort-real-pending">'
        '<span class="glyph">⌁</span>'
        f'<strong>{html.escape(pending_title)}</strong>'
        f'<p>{html.escape(pending_body)}</p>'
        '</div></div>'
        '<div class="eu-cohort-chip-card">'
        '<div class="chip-head">'
        f'<span>{html.escape("Active filters" if lang == "en" else "当前筛选")} · {len(chips)}</span>'
        f'<em>{html.escape("Clear all" if lang == "en" else "清空")}</em></div>'
        f'<div class="chip-wrap">{chips_html}</div></div></div>',
        unsafe_allow_html=True,
    )


def _render_cohort_live_preview(lang: str) -> None:
    """Render the stateful right-hand preview from the cohort-builder design."""
    cf = st.session_state.get('cohort_filter', {})
    chips = _step2_filter_chips(lang) if st.session_state.get('cohort_enabled') else []
    if st.session_state.get("entry_mode") != "demo":
        _render_real_cohort_preview_pending(lang, chips)
        return

    base_n = int(st.session_state.get('mock_params', {}).get('n_patients', 100) or 100)
    factor = 1.0
    if cf.get('age_min') is not None or cf.get('age_max') is not None:
        factor *= 0.86
    if cf.get('los_min') is not None:
        factor *= 0.82
    if cf.get('first_icu_stay') is not None:
        factor *= 0.74
    if cf.get('gender') is not None:
        factor *= 0.50
    if cf.get('survived') is not None:
        factor *= 0.58
    if cf.get('disease_cohort') and cf.get('disease_cohort') != 'none':
        factor *= 0.50
    if cf.get('icd_include_query'):
        factor *= 0.72
    filtered_n = max(1, int(round(base_n * factor))) if chips else base_n
    pct_drop = 0 if base_n == 0 else round((1 - filtered_n / base_n) * 100, 1)
    ready_label = f"{filtered_n:,} ready" if lang == "en" else f"{filtered_n:,} 可用"
    of_label = (
        f"of {base_n:,} stays · -{pct_drop}%"
        if lang == "en" else
        f"共 {base_n:,} 个 stay · -{pct_drop}%"
    )
    bar_w = max(6, min(300, int(300 * filtered_n / max(base_n, 1))))
    los_w = max(6, min(300, int(300 * 0.82)))
    chips_html = "".join(
        f'<span class="eu-cohort-chip">{html.escape(chip)}<span>×</span></span>'
        for chip in chips
    ) or f'<span class="eu-cohort-empty">{html.escape("No active filters" if lang == "en" else "无启用筛选")}</span>'
    sample_rows = [
        (20001, 71, "M", 6.2, 9, "—"),
        (20002, 58, "F", 3.8, 7, "—"),
        (20003, 83, "F", 8.1, 12, "✓"),
        (20004, 67, "M", 2.4, 5, "—"),
        (20005, 74, "M", 5.6, 8, "—"),
    ]
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in sample_rows
    )
    st.markdown(
        '<div class="eu-cohort-preview-stack">'
        '<div class="eu-cohort-preview-card">'
        '<div class="preview-head"><div>'
        f'<div class="title">{html.escape("Live preview" if lang == "en" else "实时预览")}</div>'
        f'<div class="sub">{html.escape("Sample of demo cohort" if lang == "en" else "演示队列样本")}</div>'
        '</div>'
        f'<span class="eu-pill ok"><span class="dot"></span>{html.escape(ready_label)}</span></div>'
        '<div class="preview-big mono">'
        f'<span>{filtered_n:,}</span><em>{html.escape(of_label)}</em></div>'
        '<svg class="eu-cohort-funnel" viewBox="0 0 300 34" preserveAspectRatio="none">'
        '<rect x="0" y="6" width="300" height="6" rx="3"></rect>'
        f'<rect class="ink" x="0" y="6" width="{bar_w}" height="6" rx="3"></rect>'
        '<rect x="0" y="20" width="300" height="6" rx="3"></rect>'
        f'<rect class="accent" x="0" y="20" width="{los_w}" height="6" rx="3"></rect>'
        '</svg>'
        '<div class="funnel-labels mono">'
        f'<span>{html.escape("after filters" if lang == "en" else "筛选后")}: {filtered_n:,}</span>'
        f'<span>{html.escape("after stay rule" if lang == "en" else "时长规则后")}: {int(base_n * 0.82):,}</span>'
        '</div>'
        '<div class="hist-head">'
        f'<span>{html.escape("Age distribution" if lang == "en" else "年龄分布")}</span>'
        '<span class="mono">mu 63.2 · sigma 14.8</span></div>'
        '<svg class="eu-cohort-hist" viewBox="0 0 252 60" preserveAspectRatio="none">'
        + "".join(
            f'<rect x="{i * 18 + 2}" y="{60 - h}" width="14" height="{h}" rx="1"></rect>'
            for i, h in enumerate([4, 7, 14, 22, 36, 48, 54, 51, 42, 31, 20, 11, 6, 3])
        )
        + '</svg>'
        '<div class="eu-cohort-metrics">'
        f'<div><b>{html.escape("MORTALITY" if lang == "en" else "死亡率")}</b><span>18.0%</span></div>'
        f'<div><b>{html.escape("MEAN LOS" if lang == "en" else "平均 LOS")}</b><span>4.8 d</span></div>'
        f'<div><b>{html.escape("MECH VENT" if lang == "en" else "机械通气")}</b><span>52.1%</span></div>'
        '</div></div>'
        '<div class="eu-cohort-chip-card">'
        '<div class="chip-head">'
        f'<span>{html.escape("Active filters" if lang == "en" else "当前筛选")} · {len(chips)}</span>'
        f'<em>{html.escape("Clear all" if lang == "en" else "清空")}</em></div>'
        f'<div class="chip-wrap">{chips_html}</div></div>'
        '<div class="eu-cohort-sample-card">'
        '<div class="table-head">'
        f'<span>{html.escape("Sample · first 5" if lang == "en" else "样本 · 前 5 行")}</span>'
        '<em class="mono">seed=42</em></div>'
        '<table class="mono"><thead><tr><th>stay</th><th>age</th><th>sex</th><th>los</th><th>sofa</th><th>died</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div></div>',
        unsafe_allow_html=True,
    )


def _render_step2_cohort_builder_design() -> bool:
    """Render Step 2 as the two-column cohort-builder design page."""
    lang = st.session_state.get("language", "en")
    use_mock = st.session_state.get('use_mock_data', False)
    step1_complete = (
        st.session_state.get('step1_confirmed', False)
        if use_mock else
        _real_data_source_ready()
    )
    if not step1_complete:
        st.warning("Confirm Step 1 before building the cohort." if lang == "en" else "请先确认第 1 步，再构建队列。")
        return False

    _ensure_step2_state_defaults()
    left, right = st.columns([1.42, 1.0], gap="large")
    with left:
        header_l, header_r = st.columns([1.0, 0.45], gap="small")
        with header_l:
            st.markdown(
                '<div class="eu-cohort-header">'
                f'<h1>{html.escape("Build cohort" if lang == "en" else "构建队列")}</h1>'
                f'<p>{html.escape("Filter the patient population. Preview updates live." if lang == "en" else "筛选患者人群，右侧预览实时更新。")}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with header_r:
            st.toggle(
                "Enable filtering" if lang == "en" else "启用筛选",
                key="cohort_enabled",
                help=("Filter patients by demographics and clinical criteria" if lang == "en" else "根据人口统计学和临床标准筛选患者"),
            )

        with st.container(key="eu_cohort_demographics_card"):
            st.markdown(
                f'<div class="eu-section-label"><span>{html.escape("Demographics & stay" if lang == "en" else "人口统计与 ICU stay")}</span></div>',
                unsafe_allow_html=True,
            )
            cf = st.session_state.cohort_filter
            row1 = st.columns([1, 1, 1.2], gap="small")
            with row1[0]:
                age_min = st.number_input(
                    "Min age" if lang == "en" else "最小年龄",
                    min_value=0, max_value=120,
                    value=0 if cf.get('age_min') is None else int(cf['age_min']),
                    key="cohort_age_min_design",
                    disabled=not st.session_state.cohort_enabled,
                )
                cf['age_min'] = age_min if st.session_state.cohort_enabled and age_min > 0 else None
            with row1[1]:
                age_max = st.number_input(
                    "Max age" if lang == "en" else "最大年龄",
                    min_value=0, max_value=120,
                    value=120 if cf.get('age_max') is None else int(cf['age_max']),
                    key="cohort_age_max_design",
                    disabled=not st.session_state.cohort_enabled,
                )
                cf['age_max'] = age_max if st.session_state.cohort_enabled and age_max < 120 else None
            with row1[2]:
                first_opts = {
                    'any': 'Any' if lang == "en" else '不限',
                    'yes': 'First stay only' if lang == "en" else '仅首次 ICU',
                    'no': 'Readmissions' if lang == "en" else '仅再入院',
                }
                current_first = 'yes' if cf.get('first_icu_stay') is True else 'no' if cf.get('first_icu_stay') is False else 'any'
                first_val = st.selectbox(
                    "First ICU stay" if lang == "en" else "首次 ICU",
                    options=list(first_opts.keys()),
                    format_func=lambda x: first_opts[x],
                    index=list(first_opts.keys()).index(current_first),
                    key="cohort_first_icu_design",
                    disabled=not st.session_state.cohort_enabled,
                )
                cf['first_icu_stay'] = True if first_val == 'yes' else False if first_val == 'no' else None

            row2 = st.columns([1, 1, 1], gap="small")
            with row2[0]:
                los_min = st.number_input(
                    "Min ICU stay (h)" if lang == "en" else "最短 ICU 时长(h)",
                    min_value=0, max_value=10000,
                    value=0 if cf.get('los_min') is None else int(cf['los_min']),
                    key="cohort_los_min_design",
                    disabled=not st.session_state.cohort_enabled,
                )
                cf['los_min'] = los_min if st.session_state.cohort_enabled and los_min > 0 else None
                cf['los_max'] = None
            with row2[1]:
                gender_opts = {'any': 'Any' if lang == "en" else '不限', 'M': 'Male' if lang == "en" else '男性', 'F': 'Female' if lang == "en" else '女性'}
                current_gender = cf.get('gender') or 'any'
                gender_val = st.selectbox(
                    "Gender" if lang == "en" else "性别",
                    options=list(gender_opts.keys()),
                    format_func=lambda x: gender_opts[x],
                    index=list(gender_opts.keys()).index(current_gender),
                    key="cohort_gender_design",
                    disabled=not st.session_state.cohort_enabled,
                )
                cf['gender'] = gender_val if st.session_state.cohort_enabled and gender_val != 'any' else None
            with row2[2]:
                survival_opts = {'any': 'Any' if lang == "en" else '不限', 'survived': 'Survived' if lang == "en" else '存活', 'deceased': 'Deceased' if lang == "en" else '死亡'}
                current_survival = 'survived' if cf.get('survived') is True else 'deceased' if cf.get('survived') is False else 'any'
                survival_val = st.selectbox(
                    "Outcome" if lang == "en" else "转归",
                    options=list(survival_opts.keys()),
                    format_func=lambda x: survival_opts[x],
                    index=list(survival_opts.keys()).index(current_survival),
                    key="cohort_survival_design",
                    disabled=not st.session_state.cohort_enabled,
                )
                cf['survived'] = True if survival_val == 'survived' else False if survival_val == 'deceased' else None

        with st.container(key="eu_cohort_clinical_card"):
            st.markdown(
                '<div class="eu-card-head">'
                f'<span>{html.escape("Clinical cohort" if lang == "en" else "临床队列")}</span>'
                '<em class="mono">disease_cohort</em></div>',
                unsafe_allow_html=True,
            )
            supported = _get_supported_disease_cohorts(st.session_state.get('database', 'miiv'))
            option_map_en = {
                'none': ('No filter', 'all stays'),
                'sepsis': ('Sepsis-3', 'SOFA delta >=2 + suspected infection'),
                'aki': ('AKI · KDIGO', 'stage >=1 within 48h'),
                'circ_failure': ('Circulatory failure', 'shock / vaso support'),
                'mech_vent': ('Mechanical ventilation', 'vent exposure'),
                'rrt': ('RRT', 'renal replacement therapy'),
                'ards': ('ARDS', 'ICD respiratory cohort'),
                'pneumonia': ('Pneumonia', 'ICD infection cohort'),
                'heart_failure': ('Heart failure', 'ICD cardiac cohort'),
            }
            option_map_zh = {
                'none': ('不限', '所有 stay'),
                'sepsis': ('脓毒症 Sepsis-3', 'SOFA delta >=2 + 疑似感染'),
                'aki': ('AKI · KDIGO', '48h 内 stage >=1'),
                'circ_failure': ('循环衰竭', '休克 / 血管活性支持'),
                'mech_vent': ('机械通气', '通气暴露'),
                'rrt': ('RRT', '肾脏替代治疗'),
                'ards': ('ARDS', 'ICD 呼吸队列'),
                'pneumonia': ('肺炎', 'ICD 感染队列'),
                'heart_failure': ('心力衰竭', 'ICD 心脏队列'),
            }
            option_map = option_map_en if lang == "en" else option_map_zh
            current = cf.get('disease_cohort', 'none')
            if current not in supported:
                current = 'none'
            disease_cols = st.columns(3, gap="small")
            for idx, disease_key in enumerate([key for key in supported if key in option_map][:9]):
                label, sub = option_map[disease_key]
                with disease_cols[idx % 3]:
                    if st.button(
                        label,
                        key=f"cohort_disease_card_{disease_key}",
                        type="primary" if current == disease_key else "secondary",
                        use_container_width=True,
                        disabled=not st.session_state.cohort_enabled,
                    ):
                        cf['disease_cohort'] = disease_key
                        cf['has_sepsis'] = True if disease_key == 'sepsis' else None
                        st.rerun()
                    st.caption(sub)
            if not st.session_state.cohort_enabled:
                cf['disease_cohort'] = 'none'
                cf['has_sepsis'] = None

        if _supports_icd_filter(st.session_state.get('database')) or st.session_state.get("entry_mode") == "demo":
            with st.container(key="eu_cohort_icd_card"):
                database_hint = _step2_database_display_name(st.session_state.get('database', ''))
                st.markdown(
                    '<div class="eu-card-head">'
                    f'<span>{html.escape("ICD codes" if lang == "en" else "ICD 编码")} <small>({html.escape(database_hint)})</small></span>'
                    '<em class="mono">comma / space separated</em></div>',
                    unsafe_allow_html=True,
                )
                icd_cols = st.columns(2, gap="small")
                with icd_cols[0]:
                    include_value = st.text_input(
                        "Include" if lang == "en" else "包含",
                        value=cf.get('icd_include_query', ''),
                        key="cohort_icd_include_query_design",
                        placeholder="A41,A42 or A41-42" if lang == "en" else "A41,A42 或 A41-42",
                        disabled=not st.session_state.cohort_enabled,
                    )
                    cf['icd_include_query'] = include_value.strip() if st.session_state.cohort_enabled else ""
                with icd_cols[1]:
                    exclude_value = st.text_input(
                        "Exclude" if lang == "en" else "排除",
                        value=cf.get('icd_exclude_query', ''),
                        key="cohort_icd_exclude_query_design",
                        placeholder="I50,C34 or I50-51" if lang == "en" else "I50,C34 或 I50-51",
                        disabled=not st.session_state.cohort_enabled,
                    )
                    cf['icd_exclude_query'] = exclude_value.strip() if st.session_state.cohort_enabled else ""
                cf['icd_query'] = cf.get('icd_include_query', '')
                cf['icd_mode'] = 'include'
        else:
            cf['icd_query'] = ""
            cf['icd_include_query'] = ""
            cf['icd_exclude_query'] = ""
            cf['icd_mode'] = 'include'
            _clear_icd_preview_state()

        footer_l, reset_col, preset_col, confirm_col = st.columns([2.2, 1.45, 1.45, 1.45], gap="small")
        with footer_l:
            st.markdown(
                f'<div class="eu-source-footer-note">{html.escape("Step 2 of 4" if lang == "en" else "第 2 步 / 共 4 步")}</div>',
                unsafe_allow_html=True,
            )
        with reset_col:
            if st.button("Reset" if lang == "en" else "重置", key="cohort_builder_reset", use_container_width=True):
                st.session_state[_STEP2_RESET_PENDING_KEY] = True
                st.rerun()
        with preset_col:
            if st.button("Save preset" if lang == "en" else "保存预设", key="cohort_builder_save_preset", use_container_width=True):
                st.toast("Preset saved for this session." if lang == "en" else "已保存为当前会话预设。")
        with confirm_col:
            if st.button(
                "Confirm cohort" if lang == "en" else "确认队列",
                key="step2_confirm_design",
                type="primary",
                use_container_width=True,
            ):
                _clear_icd_preview_state()
                concept_groups = get_concept_groups()
                st.session_state.pop("_eu_concept_defaults_seeded", None)
                _reset_concepts_to_groups(concept_groups, _default_concept_groups(concept_groups))
                st.session_state.step2_confirmed = True
                st.rerun()

    with right:
        _render_cohort_live_preview(lang)

    return True


def _default_concept_groups(concept_groups: dict[str, list[str]]) -> list[str]:
    """Pick stable sensible defaults by label substring across EN/ZH labels."""
    wanted = [
        "SOFA-2", "Vital", "生命体征", "Respiratory", "呼吸",
        "Chemistry", "生化", "Hematology", "血液",
        "Demographics", "人口", "Outcome", "结局",
    ]
    defaults: list[str] = []
    for group_name in concept_groups:
        if any(token in group_name for token in wanted):
            defaults.append(group_name)
        if len(defaults) >= 8:
            break
    return defaults or list(concept_groups.keys())[:4]


def _all_concept_groups(concept_groups: dict[str, list[str]]) -> list[str]:
    """Return every concept group in display order."""
    return list(concept_groups.keys())


def _selected_concepts_from_groups(
    concept_groups: dict[str, list[str]],
    groups: list[str],
) -> list[str]:
    """Collect selected concepts for groups while keeping output deterministic."""
    selected: list[str] = []
    for group_name in groups:
        for concept in concept_groups.get(group_name, []):
            selected.append(concept)
    return sorted(set(selected))


def _reset_concepts_to_groups(concept_groups: dict[str, list[str]], groups: list[str]) -> None:
    """Sync selected groups and concept checkbox state."""
    st.session_state.selected_groups = [g for g in groups if g in concept_groups]
    st.session_state.concept_checkboxes = {}
    for group_name in st.session_state.selected_groups:
        for concept in concept_groups.get(group_name, []):
            st.session_state.concept_checkboxes[concept] = True
    st.session_state.selected_concepts = _selected_concepts_from_groups(
        concept_groups,
        st.session_state.selected_groups,
    )
    st.session_state["_eu_concept_defaults_seeded"] = True


def _collect_selected_concepts(concept_groups: dict[str, list[str]]) -> list[str]:
    selected: list[str] = []
    for group_name in st.session_state.get("selected_groups", []):
        for concept in concept_groups.get(group_name, []):
            if st.session_state.concept_checkboxes.get(concept, True):
                selected.append(concept)
    return sorted(set(selected))


def _render_concept_summary(lang: str, concept_groups: dict[str, list[str]], selected_concepts: list[str]) -> None:
    selected_groups = [g for g in st.session_state.get("selected_groups", []) if g in concept_groups]
    total_features = sum(len(v) for v in concept_groups.values())
    module_tiles = "".join(
        '<div>'
        f'<b>{html.escape(_clean_module_label(group_name))}</b>'
        f'<span>{sum(1 for c in concept_groups.get(group_name, []) if c in selected_concepts)}/{len(concept_groups.get(group_name, []))}</span>'
        '</div>'
        for group_name in selected_groups[:6]
    )
    feature_list = "".join(
        f'<span class="eu-concept-chip">{html.escape(concept)}</span>'
        for concept in selected_concepts[:22]
    )
    if len(selected_concepts) > 22:
        feature_list += f'<span class="eu-concept-more">+{len(selected_concepts) - 22}</span>'
    if not feature_list:
        feature_list = f'<span class="eu-cohort-empty">{html.escape("No features selected" if lang == "en" else "尚未选择变量")}</span>'
    st.markdown(
        '<div class="eu-concept-summary-card">'
        f'<div class="label">{html.escape("Selection summary" if lang == "en" else "选择摘要")}</div>'
        '<div class="big mono">'
        f'<span>{len(selected_concepts)}</span>'
        f'<em>{html.escape(("of " + str(total_features) + " features") if lang == "en" else ("共 " + str(total_features) + " 个变量"))}</em>'
        '</div>'
        '<div class="eu-concept-module-grid">'
        f'{module_tiles}'
        '</div>'
        '</div>'
        '<div class="eu-concept-summary-card">'
        f'<div class="label">{html.escape("Selected features" if lang == "en" else "已选变量")}</div>'
        f'<div class="eu-concept-chip-wrap">{feature_list}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_step3_concept_selection_design(concept_groups: dict[str, list[str]]) -> list[str] | None:
    """Render Step 3 as the concept-selection artboard."""
    lang = st.session_state.get("language", "en")
    if not st.session_state.get('step2_confirmed', False):
        st.warning(
            "Please complete Step 2 first." if lang == "en" else "请先完成第 2 步。",
        )
        return None
    st.session_state.setdefault("concept_checkboxes", {})
    st.session_state.setdefault("selected_groups", [])
    if not st.session_state.selected_groups and not st.session_state.get("_eu_concept_defaults_seeded"):
        _reset_concepts_to_groups(concept_groups, _default_concept_groups(concept_groups))

    header_l, header_r = st.columns([1.4, 0.9], gap="large")
    with header_l:
        st.markdown(
            '<div class="eu-concept-header">'
            f'<div class="eu-step-kicker">{html.escape("Step 3 of 4" if lang == "en" else "第 3 步 / 共 4 步")}</div>'
            f'<h1>{html.escape("Select feature modules" if lang == "en" else "选择特征模块")}</h1>'
            f'<p>{html.escape("Concepts are pre-selected from the cohort. Add or remove modules — coverage is audited before analysis." if lang == "en" else "概念已根据队列预选。可增减模块；分析前会审计覆盖率。")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with header_r:
        st.write("")
        defaults_spacer, defaults_col = st.columns([1.0, 0.78], gap="small")
        with defaults_col:
            if st.button(
                "Reset to core" if lang == "en" else "重置核心模块",
                key="concept_defaults_design",
                use_container_width=True,
            ):
                _reset_concepts_to_groups(concept_groups, _default_concept_groups(concept_groups))
                st.rerun()
    selected_concepts = _collect_selected_concepts(concept_groups)
    st.session_state.selected_concepts = selected_concepts
    selected_groups_now = [g for g in st.session_state.get("selected_groups", []) if g in concept_groups]
    st.markdown(
        '<div class="eu-concept-status-strip">'
        '<span class="eu-mini-pill">auto</span>'
        f'<strong>{len(selected_groups_now)} modules · {len(selected_concepts)} concepts selected</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.48, 1.0], gap="large")
    with left:
        search = st.text_input(
            "Filter by name, code, or unit" if lang == "en" else "按名称、代码或单位筛选",
            key="concept_search_design",
            placeholder="sofa, lactate, renal..." if lang == "en" else "SOFA、乳酸、肾脏...",
        ).strip().lower()
        visible_groups = [
            group_name for group_name, concepts in concept_groups.items()
            if _concept_group_matches_search(group_name, concepts, search)
        ]
        st.markdown(
            f'<div class="eu-section-label"><span>{html.escape("Modules" if lang == "en" else "模块")}</span></div>',
            unsafe_allow_html=True,
        )
        if not visible_groups:
            empty_msg = (
                f'No modules match "{search}". Try a concept code, clinical name, or unit.'
                if lang == "en"
                else f'没有匹配“{search}”的模块。可尝试概念代码、临床名称或单位。'
            )
            st.markdown(
                '<div class="compact-inline-notice warn">'
                f'{html.escape(empty_msg)}'
                '</div>',
                unsafe_allow_html=True,
            )
        card_cols = st.columns(2, gap="small")
        for idx, group_name in enumerate(visible_groups):
            concepts = concept_groups.get(group_name, [])
            selected_count = sum(1 for c in concepts if st.session_state.concept_checkboxes.get(c, False))
            active = group_name in st.session_state.selected_groups
            display_name = _clean_module_label(group_name)
            module_key_prefix = "concept_module_active" if active else "concept_module_add"
            module_status = "on" if active else "add"
            with card_cols[idx % 2]:
                if st.button(
                    f"{display_name} · {module_status}",
                    key=f"{module_key_prefix}_{idx}",
                    type="secondary",
                    use_container_width=True,
                    icon=":material/layers:",
                ):
                    groups = list(st.session_state.selected_groups)
                    if active:
                        groups = [g for g in groups if g != group_name]
                        for concept in concepts:
                            st.session_state.concept_checkboxes.pop(concept, None)
                    else:
                        groups.append(group_name)
                        for concept in concepts:
                            st.session_state.concept_checkboxes[concept] = True
                    st.session_state.selected_groups = groups
                    st.rerun()
                caption = (
                    f"{len(concepts)} concepts" if lang == "en" else f"{len(concepts)} 个概念"
                )
                if active and selected_count != len(concepts):
                    caption = (
                        f"{selected_count}/{len(concepts)} concepts" if lang == "en" else f"{selected_count}/{len(concepts)} 个概念"
                    )
                st.caption(caption)

        selected_groups = [g for g in st.session_state.selected_groups if g in concept_groups]
        if selected_groups:
            with st.expander(
                "Feature detail configuration" if lang == "en" else "变量详细配置",
                expanded=False,
            ):
                import hashlib
                for group_name in selected_groups:
                    key_hash = hashlib.md5(group_name.encode()).hexdigest()[:8]
                    st.markdown(f"**{html.escape(_clean_module_label(group_name))}**")
                    cols = st.columns(3)
                    for cidx, concept in enumerate(concept_groups.get(group_name, [])):
                        with cols[cidx % 3]:
                            default_val = st.session_state.concept_checkboxes.get(concept, True)
                            checked = st.checkbox(concept, value=default_val, key=f"cb_design_{key_hash}_{concept}")
                            st.session_state.concept_checkboxes[concept] = checked

        selected_concepts = _collect_selected_concepts(concept_groups)
        st.session_state.selected_concepts = selected_concepts
        footer_l, reset_col, confirm_col = st.columns([4.2, 1.45, 1.45], gap="small")
        with footer_l:
            st.markdown(
                f'<div class="eu-source-footer-note">{html.escape("Step 3 of 4" if lang == "en" else "第 3 步 / 共 4 步")}</div>',
                unsafe_allow_html=True,
            )
        with reset_col:
            if st.button("Select all" if lang == "en" else "全选", key="concept_reset_design", use_container_width=True):
                _reset_concepts_to_groups(concept_groups, _all_concept_groups(concept_groups))
                st.rerun()
        with confirm_col:
            if st.button(
                "Confirm concepts" if lang == "en" else "确认变量",
                key="step3_confirm_design",
                type="primary",
                disabled=len(selected_concepts) == 0,
                use_container_width=True,
            ):
                st.session_state.step3_confirmed = True
                st.rerun()

    with right:
        _render_concept_summary(lang, concept_groups, st.session_state.get("selected_concepts", []) or [])

    return st.session_state.get("selected_concepts", []) or []


def render_sidebar(app_context: dict[str, Any] | None = None):
    """渲染侧边栏 - 根据entry_mode显示不同内容。"""
    if app_context is not None:
        _install_app_context(app_context)

    # 使用双语特征分组
    concept_groups = get_concept_groups()

    # 获取当前模式
    entry_mode = st.session_state.get('entry_mode', 'none')
    _apply_pending_step2_reset()

    with st.sidebar:
        # === Shell-A header: brand, workspace switcher, search, nav, pipeline ===
        _render_shell_brand(entry_mode)
        if entry_mode != 'none':
            with st.container(key="eu_sidebar_nav_area"):
                _render_shell_primary_nav()
                _render_shell_aux_nav()
            _render_shell_context_dock(entry_mode)

        # Footer icon row (back / help / settings / lang / avatar) — at
        # the very bottom of the sidebar per shell-a-frame.jsx.
        _render_shell_footer_icons()


def _render_sidebar_ai_and_lang() -> None:
    """Compatibility wrapper for older sessions that imported this helper.

    The footer gear now routes to the full print-reference Settings page.
    Keeping this tiny wrapper avoids breaking external tests or notebooks
    that referenced the old helper.
    """
    _render_sidebar_settings_panel()


def render_extract_page(lang: str, app_context: dict[str, Any] | None = None) -> None:
    """Main-area data-extraction workflow (steps 1-4).

    Faithful to the design canvas (page-data-source / page-cohort-builder
    / page-concepts) intent: the extraction pipeline is a *main page*
    reached from the sidebar pipeline, not a sidebar form. The actual
    per-step renderers are the real EU implementations
    (_render_step1_data_source etc.) so all extraction logic is
    preserved — only their host moved from sidebar to main pane.
    """
    if app_context is not None:
        _install_app_context(app_context)

    entry_mode = st.session_state.get("entry_mode", "none")
    concept_groups = get_concept_groups()

    if _render_export_completed_panel():
        return

    if not st.session_state.get("step1_confirmed"):
        _render_step1_data_source(entry_mode)
        return

    if not st.session_state.get("step2_confirmed"):
        if not _render_step2_cohort_builder_design():
            return
        return

    if not st.session_state.get("step3_confirmed"):
        selected_concepts = _render_step3_concept_selection_design(concept_groups)
        if selected_concepts is None:
            return
        return

    selected_concepts = st.session_state.get("selected_concepts", []) or []
    if not _render_step4_export(selected_concepts):
        return
    _render_system_resource_panel()
