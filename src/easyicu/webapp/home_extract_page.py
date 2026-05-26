"""Home extraction-mode rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from easyicu.webapp.components.constants import get_all_concepts
from easyicu.webapp.concept_catalog import CONCEPT_GROUPS_INTERNAL
from easyicu.webapp.data_dictionary_page import render_home_data_dictionary
from easyicu.webapp.ui_helpers import (
    GuidePanel,
    MiniCard,
    StatCard,
    render_anchor,
    render_file_list,
    render_footer_links,
    render_guide_card,
    render_inline_heading,
    render_note,
    render_option_card,
    render_stat_grid,
    render_status_banner,
)
from easyicu.webapp.workflow_figure import _render_extraction_pipeline_figure


def _pick(lang: str, english: str, chinese: str) -> str:
    return english if lang == "en" else chinese


def _step_done_flags() -> tuple[bool, bool, bool, bool]:
    if st.session_state.get("use_mock_data", False):
        step1_done = bool(st.session_state.get("step1_confirmed", False))
    else:
        data_path = st.session_state.get("data_path")
        step1_done = bool(data_path and Path(data_path).exists())

    return (
        step1_done,
        bool(st.session_state.get("step2_confirmed", False)),
        bool(st.session_state.get("step3_confirmed", False) and st.session_state.get("selected_concepts")),
        bool(st.session_state.get("export_completed", False)),
    )


def _guide_title(lang: str, *, step1_done: bool, step2_done: bool, step3_done: bool, step4_done: bool) -> str:
    if not step1_done:
        guide_step = _pick(lang, "Data Source", "数据源配置")
    elif not step2_done:
        guide_step = _pick(lang, "Cohort Selection", "队列筛选")
    elif not step3_done:
        guide_step = _pick(lang, "Select Features", "特征选择")
    elif not step4_done:
        guide_step = _pick(lang, "Export Data", "数据导出")
    else:
        guide_step = _pick(lang, "Export Summary", "导出摘要")

    return guide_step if step4_done else _pick(lang, f"Guide: {guide_step}", f"引导: {guide_step}")


def _render_step_guide(lang: str, *, step1_done: bool, step2_done: bool, step3_done: bool, step4_done: bool) -> None:
    if not step1_done:
        if lang == "en":
            render_guide_card(
                "Configure Data Source in the Sidebar",
                panels=[
                    GuidePanel(
                        "Demo Mode",
                        (
                            "No real data needed; EasyICU generates simulated ICU data.",
                            "Adjust patients (50-500) and duration (24-168h).",
                            'Click "Confirm Data Source" when ready.',
                        ),
                        "success",
                    ),
                    GuidePanel(
                        "Real Data Mode",
                        (
                            "MIMIC-IV, eICU, AUMC, HiRID, MIMIC-III, SICdb.",
                            "Enter your local database path.",
                            "All processing stays local.",
                        ),
                        "info",
                    ),
                ],
            )
        else:
            render_guide_card(
                "在侧边栏配置数据源",
                panels=[
                    GuidePanel(
                        "演示模式",
                        (
                            "无需真实数据，自动生成模拟 ICU 数据。",
                            "可调整患者数量（50-500）和时长（24-168h）。",
                            "设置后点击「确认数据源配置」。",
                        ),
                        "success",
                    ),
                    GuidePanel(
                        "真实数据模式",
                        (
                            "支持 MIMIC-IV、eICU、AUMC、HiRID、MIMIC-III、SICdb。",
                            "输入本地数据库路径。",
                            "所有处理均在本地完成。",
                        ),
                        "info",
                    ),
                ],
            )
        return

    if not step2_done:
        render_guide_card(
            _pick(lang, "Configure Cohort Selection", "配置队列筛选"),
            bullets=(
                _pick(lang, "Age range, for example 18-65 years.", "年龄范围，例如 18-65 岁。"),
                _pick(lang, "Gender: male, female, or any.", "性别：男、女或不限。"),
                _pick(lang, "Survival status: survivors, non-survivors, or all.", "存活状态：存活、死亡或全部。"),
                _pick(lang, "Minimum ICU length of stay.", "最短 ICU 住院时长。"),
                _pick(lang, "Clinical cohorts: Sepsis-3, AKI, circulatory failure, mechanical ventilation, RRT.", "疾病队列：Sepsis-3、AKI、循环衰竭、机械通气、RRT。"),
                _pick(lang, "ICD filter for MIMIC/eICU by prefixes or diagnosis keywords.", "MIMIC/eICU 可按 ICD 前缀或诊断关键词筛选。"),
            ),
            tip=_pick(
                lang,
                'Start from your target task, then define the cohort you need. You can still click "Confirm (No Filtering)" to skip this step.',
                "建议先从研究任务出发定义目标队列；若暂时不需要筛选，也可以点击「确认（不筛选）」跳过此步骤。",
            ),
        )
        return

    if not step3_done:
        feature_count = len(get_all_concepts())
        render_guide_card(
            _pick(
                lang,
                f"Select Features - {feature_count} ICU Clinical Features",
                f"选择特征 - {feature_count} 个 ICU 临床特征",
            ),
            mini_cards=[
                MiniCard(_pick(lang, "Vital Signs", "生命体征"), _pick(lang, "HR, BP, Temp, SpO2, Resp", "心率、血压、体温、SpO2、呼吸"), "primary"),
                MiniCard(_pick(lang, "Lab Tests", "实验室检验"), _pick(lang, "Chemistry, CBC, Coag, ABG", "生化、血常规、凝血、血气"), "success"),
                MiniCard(_pick(lang, "Medications", "药物治疗"), _pick(lang, "Vasopressors, sedatives, antibiotics", "血管活性药、镇静药、抗生素"), "warning"),
                MiniCard(_pick(lang, "Scores", "临床评分"), _pick(lang, "SOFA, GCS, AKI, Sepsis-3", "SOFA、GCS、AKI、Sepsis-3"), "purple"),
            ],
            tip=_pick(
                lang,
                "Select by category or pick individual features. Check the Data Dictionary below for details.",
                "按类别选择或选取单个特征，查看下方数据字典了解详情。",
            ),
        )
        return

    if not step4_done:
        if st.session_state.get("_export_conflict_pending", False):
            render_status_banner(
                _pick(lang, "Waiting for Your Decision", "等待你的选择"),
                _pick(
                    lang,
                    "Existing files were detected. Choose how to handle them in the panel below.",
                    "检测到已存在的文件，请在下方面板选择如何处理。",
                ),
                tone="warning",
                icon="!",
            )
            return
        if st.session_state.get("_exporting_in_progress", False):
            render_status_banner(
                _pick(lang, "Export in Progress...", "导出进行中..."),
                _pick(lang, "Please wait while your data is being exported.", "请稍候，数据正在导出中，进度详情将显示在下方。"),
                tone="warning",
                icon="!",
            )
            return

        selected = st.session_state.get("selected_concepts", [])
        source_info = (
            _pick(lang, "Demo Mode", "演示模式")
            if st.session_state.get("use_mock_data", False)
            else str(st.session_state.get("data_path", ""))
        )
        render_guide_card(
            _pick(lang, "How to Export Data", "如何导出数据"),
            ordered_steps=(
                _pick(lang, 'Go to the "Data Export" tab above.', "点击上方「数据导出」标签页。"),
                _pick(lang, "Select export format: CSV, Parquet, or Excel.", "选择导出格式：CSV、Parquet 或 Excel。"),
                _pick(lang, "Choose save location.", "选择保存位置。"),
                _pick(lang, 'Click the "Export Data" button.', "点击「导出数据」按钮。"),
            ),
            tip=_pick(
                lang,
                "Best for large datasets: saves directly to disk.",
                "适合大数据集：直接保存到磁盘，不占用内存。",
            ),
        )
        render_stat_grid(
            [
                StatCard(_pick(lang, "Data Source", "数据源"), source_info),
                StatCard(_pick(lang, "Selected Features", "已选特征"), str(len(selected)), "primary"),
            ],
            columns=2,
            compact=True,
        )


def _render_export_success(lang: str) -> None:
    export_result = st.session_state.get("_export_success_result")
    if not export_result:
        return

    exported_files = export_result["files"]
    export_dir = export_result["export_dir"]
    total_elapsed = export_result["total_time"]
    module_times = export_result.get("module_times", {})
    concept_count = export_result.get("concept_count", len(exported_files))

    success_msg = (
        f"Successfully exported {len(exported_files)} files to `{export_dir}`"
        if lang == "en"
        else f"成功导出 {concept_count} 个概念（{len(exported_files)} 个文件）到 `{export_dir}`"
    )
    st.success(success_msg)

    cohort_stats = st.session_state.get("_cohort_stats")
    if cohort_stats and cohort_stats.get("excluded", 0) > 0:
        n_before = cohort_stats["before"]
        n_excluded = cohort_stats["excluded"]
        n_after = cohort_stats["after"]
        details = cohort_stats.get("filter_details", [])
        if lang == "en":
            cohort_info = f"Cohort Selection: {n_before} candidates -> {n_after} patients exported ({n_excluded} excluded)"
            reasons = ", ".join(f"{label_en}: -{cnt}" for label_en, _, cnt in details if cnt > 0)
            if reasons:
                cohort_info += f"\n\nExclusion reasons: {reasons}"
        else:
            cohort_info = f"队列筛选：{n_before} 候选 -> 最终导出 {n_after} 位患者（排除 {n_excluded} 人）"
            reasons = "、".join(f"{label_cn}: -{cnt}人" for _, label_cn, cnt in details if cnt > 0)
            if reasons:
                cohort_info += f"\n\n排除原因：{reasons}"
        st.info(cohort_info)

    with st.expander(_pick(lang, "Export Time Statistics", "导出耗时统计"), expanded=False):
        for mod_name, mod_time in module_times.items():
            time_str = f"{mod_time / 60:.1f} min" if mod_time >= 60 else f"{mod_time:.1f} s"
            st.text(f"  • {mod_name}: {time_str}")
        total_str = f"{total_elapsed / 60:.1f} min" if total_elapsed >= 60 else f"{total_elapsed:.1f} s"
        st.markdown(_pick(lang, f"**Total: {total_str}**", f"**总计：{total_str}**"))

    with st.expander(_pick(lang, "View Exported Files", "查看导出文件"), expanded=False):
        files_to_show = [Path(path).name for path in exported_files[:6]]
        summary_msg = _pick(
            lang,
            f"Showing {len(files_to_show)} representative files out of {len(exported_files)} exported files.",
            f"当前展示 {len(files_to_show)} 个代表性文件，共导出 {len(exported_files)} 个文件。",
        )
        st.caption(summary_msg)
        more_text = ""
        if len(exported_files) > len(files_to_show):
            more_text = _pick(
                lang,
                f"... and {len(exported_files) - len(files_to_show)} more files",
                f"... 及其他 {len(exported_files) - len(files_to_show)} 个文件",
            )
        render_file_list(files_to_show, more_text=more_text)

    unavailable_concepts = export_result.get("unavailable_concepts", [])
    if unavailable_concepts:
        unsupported_list = set(export_result.get("unsupported_concepts", []))
        unsupported = sorted(concept for concept in unavailable_concepts if concept in unsupported_list)
        empty_or_other = sorted(concept for concept in unavailable_concepts if concept not in unsupported_list)
        total_unavailable = len(unsupported) + len(empty_or_other)
        st.warning(_pick(
            lang,
            f"{total_unavailable} selected concept(s) produced no exported data.",
            f"有 {total_unavailable} 个所选概念未能导出数据。",
        ))
        with st.expander(_pick(lang, "Show concepts with no data", "查看无数据的概念"), expanded=False):
            if unsupported:
                st.markdown(_pick(
                    lang,
                    f"**Not configured for this database ({len(unsupported)})**",
                    f"**该数据库未配置（{len(unsupported)}）**",
                ))
                st.caption(", ".join(unsupported))
            if empty_or_other:
                st.markdown(_pick(
                    lang,
                    f"**No data for selected patients ({len(empty_or_other)})**",
                    f"**所选患者无数据（{len(empty_or_other)}）**",
                ))
                st.caption(", ".join(empty_or_other))

    st.session_state["_last_export_concept_count"] = export_result.get("concept_count", len(exported_files))
    st.session_state["_last_export_patient_count"] = export_result.get("patient_count", 0)
    del st.session_state["_export_success_result"]


def _summary_counts() -> tuple[str, int, int]:
    db_display = "DEMO" if st.session_state.get("use_mock_data", False) else st.session_state.get("database", "N/A").upper()

    if "_last_export_concept_count" in st.session_state:
        n_concepts = st.session_state["_last_export_concept_count"]
    elif st.session_state.get("loaded_concepts"):
        n_concepts = len(st.session_state.loaded_concepts)
    elif st.session_state.get("selected_concepts"):
        n_concepts = len(st.session_state.selected_concepts)
    else:
        n_concepts = 0

    n_patients = int(st.session_state.get("_exported_patient_count") or 0)
    id_col = st.session_state.get("id_col", "stay_id")
    if n_patients == 0 and st.session_state.get("loaded_concepts"):
        all_ids = set()
        for df in st.session_state.loaded_concepts.values():
            if isinstance(df, pd.DataFrame) and id_col in df.columns:
                all_ids.update(df[id_col].unique())
        n_patients = len(all_ids)
    if n_patients == 0 and st.session_state.get("patient_ids"):
        n_patients = len(st.session_state.patient_ids)
    if n_patients == 0:
        n_patients = int((st.session_state.get("mock_params", {}) or {}).get("n_patients") or 0)

    return db_display, n_concepts, n_patients


def _render_complete_guide(lang: str) -> None:
    _render_export_success(lang)

    db_display, n_concepts, n_patients = _summary_counts()
    render_stat_grid(
        [
            StatCard(_pick(lang, "Database", "数据库"), db_display),
            StatCard(_pick(lang, "Loaded Concepts", "已加载概念"), str(n_concepts), "primary"),
            StatCard(_pick(lang, "Patients", "患者数量"), f"{n_patients:,}"),
            StatCard(_pick(lang, "Status", "数据状态"), _pick(lang, "Ready", "就绪"), "success"),
        ],
        compact=True,
    )

    render_inline_heading(_pick(lang, "What's Next?", "下一步？"))
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        render_option_card(
            _pick(lang, "Quick Visualization", "快速可视化"),
            (
                _pick(lang, "Data Tables Explorer: browse loaded data by module.", "数据表浏览器：按模块浏览已加载数据。"),
                _pick(lang, "Time Series Analysis: clinical trends over time.", "时序分析：临床指标随时间变化趋势。"),
                _pick(lang, "Patient Overview: single-patient dashboard.", "患者概览：单患者综合仪表盘。"),
                _pick(lang, "Data Quality: missing rates and completeness.", "数据质量：缺失率与完整性分析。"),
            ),
            tone="primary",
        )
        if st.button(_pick(lang, "Go to Visualization", "前往可视化"), use_container_width=True, key="goto_viz_home", type="primary"):
            if st.session_state.get("last_export_dir") or st.session_state.get("viz_confirmed_path"):
                st.session_state["viz_data_source_mode"] = "exported"
                st.session_state["_prefer_exported_viz"] = True
            st.session_state["_scroll_to_tab"] = "viz"
            st.rerun()

    with col_opt2:
        render_option_card(
            _pick(lang, "Cohort Analysis", "队列分析"),
            (
                _pick(lang, "Group Contrast Table: subgroup balance and tests.", "组间对照表：亚组平衡与统计检验。"),
                _pick(lang, "Coverage Audit: module coverage and eligibility flow.", "覆盖度审计：模块覆盖度与纳排流程。"),
                _pick(lang, "Cross-DB Benchmark: harmonized feature shifts.", "跨数据库对比：标准化特征的数据库差异。"),
                _pick(lang, "Cohort Snapshot: phenotype and outcome profile.", "队列快照：表型与结局画像。"),
                _pick(lang, "SOFA-1 vs SOFA-2: definition-driven reclassification.", "SOFA-1 vs SOFA-2：定义变化导致的重新分层。"),
            ),
            tone="purple",
        )
        if st.button(_pick(lang, "Go to Cohort Analysis", "前往队列分析"), use_container_width=True, key="goto_cohort_home", type="primary"):
            st.session_state["_scroll_to_tab"] = "cohort"
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def render_home_extract_mode(lang: str, app_context: dict[str, Any] | None = None) -> None:
    """Render the data extraction tutorial home page.

    ``app_context`` is accepted for backward compatibility with the current
    app wrapper, but this module now uses explicit imports instead of globals
    injection.
    """
    del app_context

    step1_done, step2_done, step3_done, step4_done = _step_done_flags()

    render_anchor("progress")
    _render_extraction_pipeline_figure(
        lang=lang,
        step1_done=step1_done,
        step2_done=step2_done,
        step3_done=step3_done,
        step4_done=step4_done,
    )

    render_anchor("guide")
    st.markdown(
        f'<div class="workflow-guide-title">{_guide_title(lang, step1_done=step1_done, step2_done=step2_done, step3_done=step3_done, step4_done=step4_done)}</div>',
        unsafe_allow_html=True,
    )

    if step4_done:
        _render_complete_guide(lang)
    else:
        _render_step_guide(
            lang,
            step1_done=step1_done,
            step2_done=step2_done,
            step3_done=step3_done,
            step4_done=step4_done,
        )

    render_anchor("export-progress", spacer=True)
    export_section = st.container()
    st.session_state["_export_progress_container"] = export_section

    render_anchor("dictionary", spacer=True)
    st.markdown(
        f'<h2 class="app-dictionary-heading">{_pick(lang, "Data Dictionary", "数据字典")}</h2>',
        unsafe_allow_html=True,
    )
    render_note(
        _pick(
            lang,
            f"Reference Guide: this dictionary contains all {len(get_all_concepts())} ICU clinical features available in EasyICU, organized into {len(CONCEPT_GROUPS_INTERNAL)} categories. Each feature includes its code name, full description, and measurement unit. Some features may not be available in all ICU databases.",
            f"参考指南：本字典包含 EasyICU 提供的全部 {len(get_all_concepts())} 个 ICU 临床特征，分为 {len(CONCEPT_GROUPS_INTERNAL)} 个类别。每个特征包括代码名称、完整描述和测量单位。部分特征可能并非所有 ICU 数据库都支持。",
        ),
        tone="info",
    )
    render_home_data_dictionary(lang)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    render_footer_links(
        "EasyICU - ICU Data Analysis Toolkit | GitHub: https://github.com/shen-lab-icu/EASYICU | Docs: README.md"
        if lang == "en"
        else "EasyICU - ICU 数据分析工具包 | GitHub: https://github.com/shen-lab-icu/EASYICU | Docs: README.md",
        _pick(
            lang,
            "All data processing is done locally; no data is uploaded to any server.",
            "所有数据处理均在本地完成，不会上传到任何服务器。",
        ),
    )
