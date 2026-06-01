"""EasyICU Streamlit 主应用。

本地 ICU 数据分析和可视化平台。
"""

from __future__ import annotations

import streamlit as st
from easyicu.webapp.bootstrap import (
    configure_page,
    configure_runtime_env,
    render_runtime_shell_styles,
    sync_screenshot_mode,
)
from easyicu.webapp.compat import (
    _button_compat as _button_compat_impl,
    _dataframe_compat as _dataframe_compat_impl,
    _normalize_width_kwargs,
    _plotly_chart_compat as _plotly_chart_compat_impl,
    apply_streamlit_compat,
    query_param_value,
)

# Streamlit 1.45+ enforces that page config is the first Streamlit command.
configure_page(st)

from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import html
import logging
import re
import threading
import base64
from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional
from easyicu.webapp.mock_data import generate_mock_data
from easyicu.webapp.sidebar import render_sidebar as _render_sidebar_impl
from easyicu.webapp.patient_page import render_patient_page as _render_patient_page_impl
from easyicu.webapp.cohort_group_page import render_group_comparison_subtab as _render_group_comparison_subtab_impl
from easyicu.webapp.export_workflow import execute_sidebar_export as _execute_sidebar_export_impl
from easyicu.webapp.home_extract_page import render_home_extract_mode as _render_home_extract_mode_impl
from easyicu.webapp.home_page import render_home as _render_home_impl
from easyicu.webapp.timeseries_page import render_timeseries_page as _render_timeseries_page_impl
from easyicu.webapp.data_table_page import render_data_table_subtab as _render_data_table_subtab_impl
from easyicu.webapp.quality_page import render_quality_page as _render_quality_page_impl
from easyicu.webapp import quality_metrics as _quality_metrics_impl
from easyicu.webapp.quick_visualization_page import render_quick_visualization_page as _render_quick_visualization_page_impl
from easyicu.webapp.entry_page import render_entry_page as _render_entry_page_impl
from easyicu.webapp.data_dictionary_page import (
    render_data_dictionary as _render_data_dictionary_impl,
    _get_feature_definition_rows as _get_feature_definition_rows_impl,
    _render_feature_definition_panel as _render_feature_definition_panel_impl,
    render_home_data_dictionary as _render_home_data_dictionary_impl,
    _render_home_dict_table as _render_home_dict_table_impl,
)
from easyicu.webapp.cohort_severity_page import render_severity_reclassification_subtab as _render_severity_reclassification_subtab_impl
from easyicu.webapp import sofa_reclassification as _sofa_reclassification_impl
from easyicu.webapp.cohort_multidb_page import render_multidb_distribution_subtab as _render_multidb_distribution_subtab_impl
from easyicu.webapp.cohort_dashboard_page import render_cohort_dashboard_subtab as _render_cohort_dashboard_subtab_impl
from easyicu.webapp.data_coverage_audit_page import (
    _build_data_coverage_audit as _build_data_coverage_audit_impl,
    render_data_coverage_audit_subtab as _render_data_coverage_audit_subtab_impl,
)
from easyicu.webapp.export_page import render_export_page as _render_export_page_impl
from easyicu.webapp.export_reports import (
    _generate_cohort_prefix as _generate_cohort_prefix_impl,
    _write_export_manifest as _write_export_manifest_impl,
    _build_quick_viz_pdf_report as _build_quick_viz_pdf_report_impl,
)
from easyicu.webapp.data_workflows import (
    check_data_status as _check_data_status_impl,
    convert_data_with_progress as _convert_data_with_progress_impl,
    apply_cohort_filter as _apply_cohort_filter_impl,
    validate_database_path as _validate_database_path_impl,
    load_from_exported as _load_from_exported_impl,
    load_data as _load_data_impl,
    load_data_for_preview as _load_data_for_preview_impl,
    _select_quick_preview_concepts,
)
from easyicu.webapp.conversion_workflow import (
    render_convert_dialog as _render_convert_dialog_impl,
    convert_csv_to_parquet as _convert_csv_to_parquet_impl,
)
from easyicu.webapp.paper_figures import (
    render_publication_composite_figure as _render_publication_composite_figure_impl,
    _render_paper_panel_css as _render_paper_panel_css_impl,
    render_quick_figure_panel as _render_quick_figure_panel_impl,
    render_cohort_figure_panel as _render_cohort_figure_panel_impl,
)
from easyicu.webapp.workflow_figure import _render_extraction_pipeline_figure as _render_extraction_pipeline_figure_impl
from easyicu.webapp import cohort_charts as cc
from easyicu.webapp.styles import render_global_styles
from easyicu.webapp.shell_styles import render_shell_styles
from easyicu.webapp.i18n import get_text, strip_emoji
from easyicu.webapp.page_registry import build_main_page_registry
from easyicu.webapp.page_header import render_page_header
from easyicu.webapp.session_state import (
    clear_agent_continuation_state,
    clear_run_state,
    get_state,
    init_session_state,
)
from easyicu.webapp.services import (
    COLUMN_NORMALIZATION_MAP,
    NORMALIZED_TO_ORIGINAL_MAP,
    count_unique_columns,
    cohort_feature_counts,
    count_unique_concepts,
    get_unique_concepts,
    map_column_to_concept,
    normalize_column_name,
)
from easyicu.webapp.demo_data import (
    LIGHTWEIGHT_DEMO_HOURS,
    LIGHTWEIGHT_DEMO_PATIENTS,
    _build_group_feature_data_from_loaded_concepts,
    _build_mock_group_feature_data,
    _generate_mock_cohort_dashboard_data,
    _generate_mock_demographics,
    _generate_mock_multidb_data,
    generate_lightweight_demo_data,
    get_mock_params_with_cohort,
)
from easyicu.webapp.data_paths import (
    _choose_directory_dialog,
    _closest_existing_dir,
    _default_real_data_root,
    _default_real_database,
    _directory_input,
    _get_database_download_info,
    _path_looks_like_database,
    _render_directory_browser_dialog,
    _sync_real_data_panel_defaults,
    find_database_path,
    render_directory_structure_guide,
)
from easyicu.webapp.cohort_config import (
    DISEASE_COHORT_CONFIG,
    ICD_FILTER_DATABASES,
    SEPSIS_MODE_CONFIG,
)
from easyicu.webapp.cohort_filters import (
    _get_age_series,
    _get_death_series,
    _get_first_icu_mask,
    _get_los_hours_series,
    _get_positive_patient_ids_from_data,
    _get_sex_series,
    _get_sepsis_runtime_options,
    _get_supported_disease_cohorts,
    _match_ids_by_icd_tokens,
    _pick_death_stay,
    _post_filter_cohort_data,
    _split_query_tokens,
    _supports_icd_filter,
)
from easyicu.webapp.icd_preview import (
    _clear_icd_preview_state,
    _preview_icd_match,
    _render_icd_preview_main_panel,
)
from easyicu.webapp.cohort_workspace import (
    _REAL_WORKSPACE_DEFAULT_MAX_PATIENTS,
    _REAL_WORKSPACE_MAX_PATIENTS,
    _cohort_demo_workspace_ready,
    _cohort_real_workspace_matches_sidebar,
    _cohort_real_workspace_ready,
    _ensure_cohort_demo_workspace,
    _ensure_cohort_figure_demo_data,
    _ensure_cohort_real_workspace,
    _ensure_cohort_real_workspace_from_loaded_concepts,
)
from easyicu.webapp.concept_catalog import (
    CLINICAL_LANES,
    CLINICAL_THRESHOLDS,
    CONCEPT_DB_COVERAGE,
    CONCEPT_DESCRIPTIONS,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_DISPLAY,
    CONCEPT_GROUPS_INTERNAL,
    MODULE_PREVIEW_COLUMN_PRIORITY,
    MODULE_PREVIEW_SUMMARIES,
    MODULE_PREVIEW_TAG_PRIORITY,
    PHYSIOLOGIC_RANGES,
    PREVIEW_TIME_COLUMNS,
    PRIMARY_VALUE_COLUMN_HINTS,
    QUALITY_DEMOGRAPHIC_STATIC,
    QUALITY_EVENT_TIME_SERIES,
    QUALITY_EXCLUDE_COLUMNS,
    QUALITY_STATIC_BOOLEAN_EVENTS,
    QUALITY_TIME_CANDIDATES,
    SCREENSHOT_QUALITY_PRIORITY,
    SCREENSHOT_TIMESERIES_PRIORITY,
    SUPPORTED_DB_KEYS,
    TIME_SERIES_COMPATIBLE_MODULES,
    _get_patient_id_table_files,
    _sample_patient_ids_random,
)


apply_streamlit_compat(st)
configure_runtime_env()

# 尝试导入美化组件
try:
    from streamlit_extras.metric_cards import style_metric_cards
    HAS_EXTRAS = True
except ImportError:
    HAS_EXTRAS = False

# Runtime UI shell styles are dynamic because screenshot mode and sidebar width
# live in session state. The static design system remains in styles.py.
sync_screenshot_mode(st)
render_runtime_shell_styles(st)
render_global_styles(st)
# Shell-A design layer — must come after render_global_styles so the new
# tokens (restrained-teal accent, IBM Plex stack, flat surfaces) override
# the legacy gradient palette from styles.py.
render_shell_styles(st)

_eu_density_pref = str(st.session_state.get("ui_density") or "comfortable").lower()
if _eu_density_pref not in {"comfortable", "compact"}:
    _eu_density_pref = "comfortable"
_eu_reduce_motion = "true" if bool(st.session_state.get("reduce_motion", False)) else "false"
st.markdown(
    f"""
    <div id="eu-display-preferences"
         data-density="{_eu_density_pref}"
         data-reduce-motion="{_eu_reduce_motion}"
         style="display:none"></div>
    <style>
    .stApp:has(#eu-display-preferences[data-density="compact"]) {{
      font-size: 13px;
    }}
    .stApp:has(#eu-display-preferences[data-density="compact"]) .main .block-container {{
      padding-top: 0.75rem !important;
    }}
    .stApp:has(#eu-display-preferences[data-density="compact"]) .eu-card,
    .stApp:has(#eu-display-preferences[data-density="compact"]) .eu-settings-card,
    .stApp:has(#eu-display-preferences[data-density="compact"]) .eu-agent-panel {{
      padding-block: 12px !important;
    }}
    .stApp:has(#eu-display-preferences[data-reduce-motion="true"]) *,
    .stApp:has(#eu-display-preferences[data-reduce-motion="true"]) *::before,
    .stApp:has(#eu-display-preferences[data-reduce-motion="true"]) *::after {{
      animation-duration: 0.001ms !important;
      animation-iteration-count: 1 !important;
      scroll-behavior: auto !important;
      transition-duration: 0.001ms !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _dataframe_compat(data, **kwargs):
    """Module-level wrapper so tests and extracted pages can share the shim."""
    return _dataframe_compat_impl(st, data, **kwargs)


def _button_compat(label, *args, **kwargs):
    """Module-level wrapper so tests and extracted pages can share the shim."""
    return _button_compat_impl(st, label, *args, **kwargs)


def _plotly_chart_compat(figure_or_data, *args, **kwargs):
    """Module-level wrapper so tests and extracted pages can share the shim."""
    return _plotly_chart_compat_impl(st, figure_or_data, *args, **kwargs)


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
            'title': "Module Table Preview",
            'description': "Preview loaded tables by module before drilling into feature-level detail.",
        }
    return {
        'title': "模块数据预览",
        'description': "按模块预览已加载数据表，再进入单个特征的细节查看。",
    }


def _get_single_feature_preview_copy(feature_name: str, lang: str = 'en') -> Dict[str, str]:
    if lang == 'en':
        return {
            'title': "Single Feature Preview",
            'description': f"Inspect `{feature_name}` with full row-level detail while keeping the preview layout consistent.",
        }
    return {
        'title': "单特征预览",
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
    if concept_name in SPECIAL_CONCEPTS:
        return True
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


def _is_screenshot_mode() -> bool:
    """Return whether figure-oriented screenshot mode is enabled."""
    return bool(st.session_state.get('screenshot_mode', False))


def _apply_screenshot_mode_ui_state(state: dict[str, Any]) -> None:
    """Hide transient chrome that should not appear in figure screenshots."""
    state['_floating_ai_open'] = False
    state['_sidebar_ai_open'] = False
    state['_eu_sidebar_settings_open'] = False
    state['_inline_ai_panel_open'] = False
    if state.get('_scroll_to_tab') == 'ai_assistant':
        state.pop('_scroll_to_tab', None)


def _clear_assistant_surfaces(
    state: MutableMapping[str, Any],
    *,
    clear_pending: bool = False,
) -> None:
    """Close assistant-only surfaces when navigation leaves the assistant page."""
    state['_inline_ai_panel_open'] = False
    state['_floating_ai_open'] = False
    state['_sidebar_ai_open'] = False
    if clear_pending:
        state.pop('_ai_pending_question', None)


def _open_embedded_ai_assistant(
    state: dict[str, Any],
    question: str | None = None,
) -> None:
    """Route to the standalone AI Assistant page and optionally queue a prompt."""
    if question:
        state['_ai_pending_question'] = question
    state['llm_enabled'] = True
    state['_llm_toggle'] = True
    state['_active_main_page'] = 'assistant'
    state['_scroll_to_top'] = True
    _clear_assistant_surfaces(state, clear_pending=False)


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


def _append_action_log(state: dict[str, Any], message: str, *, limit: int = 12) -> None:
    """Keep a compact human-readable action history for the topbar."""
    log = list(state.get('_eu_action_log') or [])
    log.append(message)
    state['_eu_action_log'] = log[-limit:]


def _real_data_source_ready_for_step1(state: dict[str, Any]) -> bool:
    """Return whether Real Data Step 1 has a validated local data path."""
    data_path = str(state.get('data_path') or '').strip()
    if not data_path or not Path(data_path).exists():
        return False
    if not bool(state.get('path_validated')):
        return False
    last_validated = str(state.get('last_validated_path') or '').strip()
    if last_validated and Path(last_validated).expanduser() != Path(data_path).expanduser():
        return False
    return True


def _set_extract_step_state(state: dict[str, Any], step: int) -> None:
    """Navigate the extraction workflow to an already unlocked step."""
    step = max(1, min(4, int(step)))
    state['_active_main_page'] = 'extract'
    state['step1_confirmed'] = step > 1
    state['step2_confirmed'] = step > 2
    state['step3_confirmed'] = step > 3
    state['_scroll_to_top'] = True
    if step < 4:
        state['export_completed'] = False


def _extract_step_unlocked(state: dict[str, Any], step: int) -> bool:
    """Return whether the requested extraction step can be reached from state."""
    if step <= 1:
        return True
    if step == 2:
        return bool(state.get('step1_confirmed'))
    if step == 3:
        return bool(state.get('step1_confirmed') and state.get('step2_confirmed'))
    return bool(
        state.get('step1_confirmed')
        and state.get('step2_confirmed')
        and state.get('step3_confirmed')
    )


def _switch_extract_entry_mode(state: dict[str, Any], target: str) -> None:
    """Switch demo/real source mode and reset the extraction workflow."""
    if target not in {'demo', 'real'}:
        return
    previous_database = state.get('database')
    state['entry_mode'] = target
    state['use_mock_data'] = target == 'demo'
    if target == 'demo':
        state['database'] = 'mock'
        state['mock_params'] = {
            'n_patients': LIGHTWEIGHT_DEMO_PATIENTS,
            'hours': LIGHTWEIGHT_DEMO_HOURS,
            'demo_profile': 'lite',
        }
        state['demo_mode_patients'] = LIGHTWEIGHT_DEMO_PATIENTS
        state['demo_mode_hours'] = LIGHTWEIGHT_DEMO_HOURS
    elif previous_database not in {'miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic'}:
        state['database'] = 'miiv'
        state['path_validated'] = False
        state.pop('last_validated_path', None)
    for key in ('step1_confirmed', 'step2_confirmed', 'step3_confirmed', 'export_completed'):
        state[key] = False
    state['trigger_export'] = False
    state['_exporting_in_progress'] = False
    state['loaded_concepts'] = {}
    state['loaded_data_origin'] = 'none'
    state['patient_ids'] = []
    state['all_patient_count'] = 0
    state['selected_patient'] = None
    state['selected_concepts'] = []
    state.pop('quick_viz_active_panel', None)
    state.pop('_preview_requested', None)
    state.pop('_viz_import_export_auto_trigger', None)
    state.pop('_scroll_to_tab', None)
    state['_active_main_page'] = 'extract'


def _apply_topbar_breadcrumb_target(state: dict[str, Any], target: str) -> None:
    """Navigate to a parent destination from a topbar breadcrumb segment."""
    if target == 'entry':
        clear_run_state("all")
        state['entry_mode'] = 'none'
        state['use_mock_data'] = False
        state['_active_main_page'] = 'tutorial'
    elif target == 'data_extraction':
        state['_active_main_page'] = 'extract'
    elif target == 'extract':
        state['_active_main_page'] = 'extract'
    elif target == 'data_visualization':
        state['_active_main_page'] = 'quick_viz'
    elif target in {'quick_viz', 'cohort', 'cross_db', 'research_agent'}:
        state['_active_main_page'] = target


def _route_completed_export_to_visualization(
    state: dict[str, Any],
    *,
    request_refresh: bool = False,
    sync_widget_keys: bool = True,
) -> None:
    """Send every completed export path to the first review panel."""
    state['_active_main_page'] = 'quick_viz'
    if sync_widget_keys:
        state['_main_nav_widget'] = 'quick_viz'
        state['quick_viz_active_panel'] = state.pop('_post_export_target_panel', 'Data Tables')
    else:
        state['_post_export_target_panel'] = 'Data Tables'
    state['_scroll_to_top'] = True
    if request_refresh:
        state['_post_export_navigation_pending'] = True


def _apply_post_export_next_step(
    state: dict[str, Any],
    target: str,
    *,
    lang: str = "en",
) -> None:
    """Route the post-export guidance buttons to real downstream pages."""
    state["_post_export_guidance_dismissed"] = True
    state.pop("_post_export_navigation_pending", None)

    if target == "review":
        state["_active_main_page"] = "quick_viz"
        state["quick_viz_active_panel"] = "Data Tables"
        state["_scroll_to_top"] = True
        return

    if target == "cohort":
        state["_active_main_page"] = "cohort"
        state["_scroll_to_top"] = True
        return

    if target == "agent":
        clear_agent_continuation_state(state)
        state["entry_mode"] = "real"
        state["use_mock_data"] = False
        valid_real_databases = {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}
        if state.get("database") not in valid_real_databases:
            state["database"] = "miiv"
            state["path_validated"] = False
            state.pop("last_validated_path", None)
        state["_active_main_page"] = "research_agent"
        state["_ra_view"] = "setup"
        state["_scroll_to_top"] = True
        state["_eu_ra_focus_module_folder"] = True
        state["_eu_ra_module_pick_force_manual"] = True
        state["_eu_ra_apply_export_file_selection"] = True
        state.pop("research_agent_module_dir_pick", None)
        export_dir = str(state.get("last_export_dir") or state.get("export_path") or "")
        if export_dir:
            state["research_agent_module_dir_text"] = export_dir
        state["research_agent_cohort_source"] = (
            "选择 EasyICU 模块导出文件夹"
            if lang == "zh"
            else "Pick an EasyICU module export folder"
        )
        return

    if target == "dismiss":
        return

    raise ValueError(f"Unknown post-export next step: {target}")


def _render_post_export_guidance(
    active_page: str,
    lang: str,
    *,
    export_in_progress: bool,
) -> None:
    """Show explicit downstream choices once a local export finishes."""
    if export_in_progress:
        return
    if not st.session_state.get("export_completed", False):
        return
    if st.session_state.get("_post_export_guidance_dismissed", False):
        return
    if active_page not in {"quick_viz", "cohort", "research_agent", "tutorial"}:
        return

    export_dir = str(st.session_state.get("last_export_dir", "") or "")
    result = st.session_state.get("_export_success_result", {})
    n_files = len(result.get("files", []) or []) if isinstance(result, dict) else 0
    n_patients = int(result.get("patient_count", 0) or 0) if isinstance(result, dict) else 0
    title = "Export complete" if lang == "en" else "导出完成"
    message = (
        "Review the loaded tables, run cohort statistics, or use this export as the Research Agent cohort source."
        if lang == "en"
        else "下一步可以审阅导出表、做队列统计，或把这个导出文件夹交给研究智能体。"
    )
    meta_bits = []
    if n_files:
        meta_bits.append(f"{n_files} files" if lang == "en" else f"{n_files} 个文件")
    if n_patients:
        meta_bits.append(f"{n_patients:,} patients" if lang == "en" else f"{n_patients:,} 位患者")
    if export_dir:
        meta_bits.append(export_dir)
    meta = " · ".join(meta_bits) if meta_bits else ("local export ready" if lang == "en" else "本地导出已就绪")
    st.markdown(
        f"""
        <div class="eu-post-export-hero">
          <span class="glyph">✓</span>
          <div>
            <b>{html.escape(title)}</b>
            <p>{html.escape(message)}</p>
            <code>{html.escape(meta)}</code>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    review_label = "Review tables" if lang == "en" else "查看导出表格"
    cohort_label = "Cohort stats" if lang == "en" else "队列统计"
    agent_label = "Use export in Agent" if lang == "en" else "用导出数据打开智能体"
    dismiss_label = "Dismiss" if lang == "en" else "收起"
    cols = st.columns([1.1, 1.1, 1.1, 0.8], gap="small")
    with cols[0]:
        if st.button(
            review_label,
            key="_post_export_open_review",
            use_container_width=True,
            icon=":material/table_view:",
        ):
            _apply_post_export_next_step(st.session_state, "review", lang=lang)
            st.rerun()
    with cols[1]:
        if st.button(
            cohort_label,
            key="_post_export_open_cohort",
            use_container_width=True,
            icon=":material/query_stats:",
        ):
            _apply_post_export_next_step(st.session_state, "cohort", lang=lang)
            st.rerun()
    with cols[2]:
        if st.button(
            agent_label,
            key="_post_export_open_agent",
            use_container_width=True,
            icon=":material/auto_awesome:",
        ):
            _apply_post_export_next_step(st.session_state, "agent", lang=lang)
            st.rerun()
    with cols[3]:
        if st.button(
            dismiss_label,
            key="_post_export_dismiss",
            use_container_width=True,
        ):
            _apply_post_export_next_step(st.session_state, "dismiss", lang=lang)
            st.rerun()


def _consume_completed_export_navigation(state: dict[str, Any]) -> bool:
    """Apply a queued post-export route once and report whether it fired."""
    if not state.pop('_post_export_navigation_pending', False):
        return False
    if not state.get('export_completed'):
        return False
    if state.get('_scroll_to_tab') == 'export_progress':
        state.pop('_scroll_to_tab', None)
    _route_completed_export_to_visualization(state)
    return True


def _prepare_quick_viz_demo_workspace(
    state: dict[str, Any],
    *,
    generate_data_func=generate_lightweight_demo_data,
) -> tuple[int, int]:
    """Load the compact demo review workspace from the topbar Render action."""
    params = state.get('mock_params') if isinstance(state.get('mock_params'), dict) else {}
    params = dict(params)
    try:
        params['n_patients'] = int(params.get('n_patients') or LIGHTWEIGHT_DEMO_PATIENTS)
    except (TypeError, ValueError):
        params['n_patients'] = LIGHTWEIGHT_DEMO_PATIENTS
    params['n_patients'] = min(max(1, params['n_patients']), LIGHTWEIGHT_DEMO_PATIENTS)
    try:
        params['hours'] = int(params.get('hours') or LIGHTWEIGHT_DEMO_HOURS)
    except (TypeError, ValueError):
        params['hours'] = LIGHTWEIGHT_DEMO_HOURS
    params['hours'] = min(max(24, params['hours']), LIGHTWEIGHT_DEMO_HOURS)
    params['demo_profile'] = 'lite'

    generated = generate_data_func(**params)
    if isinstance(generated, tuple):
        mock_data, patient_ids = generated
    else:
        mock_data = generated
        patient_ids = []

    state['mock_params'] = params
    state['loaded_concepts'] = mock_data
    state['loaded_data_origin'] = 'demo_viz'
    state['patient_ids'] = sorted(patient_ids) if patient_ids else []
    state['id_col'] = 'stay_id'
    state['time_col'] = 'time'
    state['selected_concepts'] = list(mock_data.keys())
    state['trigger_export'] = False
    state['_exporting_in_progress'] = False
    for tmp_key in ['_skipped_modules', '_overwrite_modules', '_viz_import_export_auto_trigger']:
        state.pop(tmp_key, None)
    return len(mock_data), len(state['patient_ids'])


def _reset_settings_defaults(state: dict[str, Any]) -> None:
    """Apply the visible Settings-page defaults without touching local paths."""
    try:
        from easyicu.webapp.llm_config import default_provider_key
        default_provider = default_provider_key()
    except Exception:
        default_provider = "easyicu_hosted"

    state['entry_mode'] = 'demo'
    state['use_mock_data'] = True
    state['database'] = 'mock'
    state['demo_mode_patients'] = LIGHTWEIGHT_DEMO_PATIENTS
    state['demo_mode_hours'] = LIGHTWEIGHT_DEMO_HOURS
    state['mock_params'] = {
        'n_patients': LIGHTWEIGHT_DEMO_PATIENTS,
        'hours': LIGHTWEIGHT_DEMO_HOURS,
        'demo_profile': 'lite',
    }
    state['data_path'] = None
    state['path_validated'] = False
    for key in (
        'last_validated_path',
        'sidebar_data_path_input',
        'research_agent_extract_data_path',
        'research_agent_extract_db',
        '_research_agent_extract_db_source',
        '_post_export_navigation_pending',
        '_post_export_target_panel',
        '_export_cancel_notice',
    ):
        state.pop(key, None)
    state['_post_export_guidance_dismissed'] = True
    clear_agent_continuation_state(state)
    for key in (
        '_agent_workbench',
        '_agent_workbench_source_run_dir',
        '_agent_workbench_is_active_selection',
        '_eu_ra_launch_requested',
        '_eu_ra_resource_focus',
        '_eu_ra_focus_module_folder',
        '_eu_ra_module_pick_force_manual',
        '_eu_ra_apply_export_file_selection',
        '_eu_wb_findings_acked',
        '_eu_wb_findings_acked_run_dir',
        '_eu_wb_review_details_expanded',
        '_eu_wb_action_panel',
    ):
        state.pop(key, None)
    for key in list(state):
        if str(key).startswith((
            '_eu_summary_review_note_',
            '_eu_wb_ev_sha_show_',
            '_eu_wb_ev_id_show_',
            '_eu_wb_evidence_pick_',
            '_eu_wb_timeline_jump_',
        )):
            state.pop(key, None)
    state['_ra_view'] = 'setup'

    state['llm_enabled'] = False
    state['llm_provider'] = default_provider
    state['llm_api_key'] = ''
    state['llm_model'] = ''
    state['llm_base_url'] = ''
    state['llm_configured'] = False
    state['_llm_toggle'] = False
    state['_llm_toggle_sync_pending'] = True
    state['_eu_settings_allow_outbound_model_calls'] = False
    state['_eu_settings_reduce_motion'] = False
    state['ui_density'] = 'comfortable'
    state['reduce_motion'] = False
    state['_llm_provider_sel'] = default_provider
    state['_llm_api_key_inp'] = ''
    state['_llm_base_url_inp'] = ''
    state['_llm_model_inp'] = ''
    state['_floating_ai_open'] = False


def _consume_topbar_run_request(
    state: dict[str, Any],
    active_page: str,
    lang: str,
    *,
    ensure_demo_workspace_fn=_ensure_cohort_demo_workspace,
    ensure_real_from_loaded_fn=_ensure_cohort_real_workspace_from_loaded_concepts,
    ensure_real_workspace_fn=_ensure_cohort_real_workspace,
    generate_data_func=generate_lightweight_demo_data,
) -> dict[str, str] | None:
    """Turn the shell topbar primary button into page-specific behavior."""
    request = state.get('_eu_topbar_run_request')
    if not isinstance(request, dict) or request.get('page') != active_page:
        return None

    state.pop('_eu_topbar_run_request', None)
    is_en = lang == 'en'
    entry_mode = state.get('entry_mode', 'none')

    if active_page == 'tutorial':
        state['_active_main_page'] = 'extract'
        message = 'Opened data extraction workflow.' if is_en else '已打开数据提取流程。'
        _append_action_log(state, message)
        return {'level': 'info', 'message': message}

    if active_page == 'quick_viz':
        if state.get('loaded_concepts'):
            counts = cohort_feature_counts(state)
            message = (
                f"Review workspace already loaded: {counts['features']} concepts, {counts['patients']} patients."
                if is_en else
                f"审阅工作区已加载：{counts['features']} 个概念，{counts['patients']} 名患者。"
            )
        elif entry_mode == 'demo':
            n_concepts, n_patients = _prepare_quick_viz_demo_workspace(
                state,
                generate_data_func=generate_data_func,
            )
            message = (
                f"Loaded lightweight demo review workspace: {n_concepts} concepts, {n_patients} patients."
                if is_en else
                f"已加载轻量演示审阅工作区：{n_concepts} 个概念，{n_patients} 名患者。"
            )
        else:
            state['viz_data_source_mode'] = 'exported'
            message = (
                'Choose an exported EasyICU folder, then load selected data.'
                if is_en else
                '请选择 EasyICU 导出文件夹，然后加载所选数据。'
            )
            state['_viz_notices'] = [{'level': 'warning', 'message': message}]
        _append_action_log(state, message)
        return {'level': 'success' if entry_mode == 'demo' else 'info', 'message': message}

    if active_page == 'cohort':
        if entry_mode == 'demo':
            ensure_demo_workspace_fn(state, lang=lang, force=True)
            message = 'Demo cohort workspace refreshed for all panels.' if is_en else '已刷新所有面板的演示队列工作区。'
            level = 'success'
        elif state.get('loaded_concepts'):
            ok, detail = ensure_real_from_loaded_fn(state, lang=lang)
            message = detail if detail else (
                'Built Cohort Statistics workspace from loaded exports.'
                if is_en else
                '已从加载的导出数据构建队列统计工作区。'
            )
            level = 'success' if ok else 'warning'
        else:
            ok, detail = ensure_real_workspace_fn(
                state,
                lang=lang,
                max_patients=state.get('_cohort_real_ws_max_patients', _REAL_WORKSPACE_DEFAULT_MAX_PATIENTS),
                force=True,
            )
            message = detail if detail else (
                'Shared real-data cohort workspace is ready.'
                if is_en else
                '共享真实数据队列工作区已就绪。'
            )
            level = 'success' if ok else 'warning'
        _append_action_log(state, message)
        return {'level': level, 'message': message}

    if active_page == 'cross_db':
        if entry_mode == 'demo':
            ensure_demo_workspace_fn(state, lang=lang, force=True)
            message = 'Demo Cross-DB benchmark data refreshed.' if is_en else '已刷新演示跨数据库对比数据。'
            level = 'success'
        elif state.get('multidb_data'):
            state['_eu_crossdb_distribution_open'] = True
            message = (
                'Detailed Cross-DB distributions are open below.'
                if is_en else
                '下方已打开跨数据库详细分布。'
            )
            level = 'success'
        else:
            state['_eu_crossdb_distribution_open'] = True
            message = (
                'Detailed Cross-DB distribution panel is open below; connect at least two database roots.'
                if is_en else
                '下方已打开跨数据库详细分布面板；请连接至少两个数据库根目录。'
            )
            level = 'warning'
        _append_action_log(state, message)
        return {'level': level, 'message': message}

    if active_page == 'settings':
        _reset_settings_defaults(state)
        message = 'Settings reset to workspace defaults.' if is_en else '设置已恢复为工作区默认值。'
        _append_action_log(state, message)
        return {'level': 'success', 'message': message}

    if active_page == 'assistant':
        from easyicu.webapp.llm_chat import _prepare_research_agent_handoff_from_ai

        seeded = _prepare_research_agent_handoff_from_ai(state)
        message = (
            'Opened Research Agent setup with the latest assistant question.'
            if seeded else
            'Opened Research Agent setup.'
        ) if is_en else (
            '已带入最近的助手问题并打开 Research Agent 配置。'
            if seeded else
            '已打开 Research Agent 配置。'
        )
        _append_action_log(state, message)
        return {'level': 'info', 'message': message}

    if active_page == 'states':
        state['_active_main_page'] = 'quick_viz'
        state['_scroll_to_top'] = True
        message = 'Opened Patient Review.' if is_en else '已打开患者审阅。'
        _append_action_log(state, message)
        return {'level': 'info', 'message': message}

    if active_page == 'research_agent':
        if entry_mode == 'demo':
            state['_ra_view'] = 'setup'
            message = 'Opened the demo Agent guide.' if is_en else '已打开演示 Agent 导览。'
            level = 'success'
        else:
            state['_ra_view'] = 'setup'
            state['_eu_ra_launch_requested'] = True
            message = (
                'Opened Research Agent run controls. Confirm the request, cohort, and LLM provider in Setup before launching.'
                if is_en else
                '已打开研究智能体运行控制。请在配置页确认研究问题、队列和模型服务后再启动。'
            )
            level = 'info'
        _append_action_log(state, message)
        return {'level': level, 'message': message}

    return None


_TOPBAR_STATE_RERUN_PAGES = {
    "assistant",
    "states",
    "settings",
    "quick_viz",
    "cohort",
    "cross_db",
    "research_agent",
}


def _handle_topbar_run_request(active_page: str, lang: str) -> dict[str, str] | None:
    """Consume a queued topbar request and surface a small user notice."""
    pending_notice = st.session_state.pop('_eu_topbar_notice_pending', None)
    if pending_notice:
        try:
            st.toast(pending_notice)
        except Exception:
            pass

    result = _consume_topbar_run_request(st.session_state, active_page, lang)
    if result and active_page in _TOPBAR_STATE_RERUN_PAGES:
        # The request is consumed after the sidebar/topbar have rendered.
        # Re-run once so those shell regions reflect the new page state too.
        st.session_state['_eu_topbar_notice_pending'] = result.get('message', '')
        st.rerun()

    if result and result.get('message'):
        try:
            st.toast(result['message'])
        except Exception:
            pass
    return result


def _topbar_primary_action_label(
    active_page: str,
    lang: str,
    *,
    entry_mode: str = "none",
) -> tuple[str, str]:
    """Return the global topbar action label for the current page."""
    if active_page == 'research_agent':
        return ('Agent guide', 'Agent 导览')
    label_map = {
        'assistant': ('Open Agent', '打开 Agent'),
        'states': ('Patient Review', '患者审阅'),
        'quick_viz': ('Render', '渲染'),
        'cohort': ('Re-run', '重新运行'),
        'cross_db': ('Run', '运行'),
        'settings': ('Reset to defaults', '恢复默认'),
    }
    return label_map.get(active_page, ('Run', '运行'))


def _topbar_primary_action_icon(active_page: str) -> str | None:
    """Return an optional material icon for the global topbar action."""
    if active_page == 'settings':
        return ':material/refresh:'
    if active_page == 'assistant':
        return ':material/smart_toy:'
    if active_page == 'states':
        return ':material/table_chart:'
    return None


def _render_narrow_view_notice(active_page: str, lang: str) -> None:
    """Show the dense-chart mobile notice only where it applies."""
    if active_page not in {'quick_viz', 'cohort', 'cross_db'}:
        return
    message = (
        "Narrow view: EasyICU keeps this page readable here. Use a ≥1024 px window for dense chart comparison."
        if lang == "en" else
        "窄屏视图：EasyICU 会保持当前页面可读；密集图表对比建议使用 ≥1024 px 窗口。"
    )
    st.markdown(
        f'<div class="eu-narrow-view-note">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


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
    # Cross-DB is its own top tab (2026-05 Phase B); keep the historical
    # 'cohort' section name so screenshot URLs that already exist still
    # land on the same chart.
    'crossdb': ('cross_db', 'Cross-DB Benchmark'),
    'cross-db': ('cross_db', 'Cross-DB Benchmark'),
    'distribution': ('cross_db', 'Cross-DB Benchmark'),
    'benchmark': ('cross_db', 'Cross-DB Benchmark'),
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

def _quality_metric_call(name: str, *args, **kwargs):
    _quality_metrics_impl._install_app_context(globals())
    return getattr(_quality_metrics_impl, name)(*args, **kwargs)

def _quality_detect_time_col(*args, **kwargs):
    return _quality_metric_call('_quality_detect_time_col', *args, **kwargs)

def _quality_to_hour_bins(*args, **kwargs):
    return _quality_metric_call('_quality_to_hour_bins', *args, **kwargs)

def _get_quality_cohort_patient_count(*args, **kwargs):
    return _quality_metric_call('_get_quality_cohort_patient_count', *args, **kwargs)

def _count_quality_event_occurrences(*args, **kwargs):
    return _quality_metric_call('_count_quality_event_occurrences', *args, **kwargs)

def _choose_concept_value_column(*args, **kwargs):
    return _quality_metric_call('_choose_concept_value_column', *args, **kwargs)

def _get_concept_numeric_value_columns(*args, **kwargs):
    return _quality_metric_call('_get_concept_numeric_value_columns', *args, **kwargs)

def _expected_observation_count(*args, **kwargs):
    return _quality_metric_call('_expected_observation_count', *args, **kwargs)

def _compute_quality_out_of_physio_rate(*args, **kwargs):
    return _quality_metric_call('_compute_quality_out_of_physio_rate', *args, **kwargs)

def _compute_quality_duplicate_timestamp_rate(*args, **kwargs):
    return _quality_metric_call('_compute_quality_duplicate_timestamp_rate', *args, **kwargs)

def _summarize_quality_temporal_density(*args, **kwargs):
    return _quality_metric_call('_summarize_quality_temporal_density', *args, **kwargs)

def _filter_patient_selector_options(*args, **kwargs):
    return _quality_metric_call('_filter_patient_selector_options', *args, **kwargs)

def _patient_selector(*args, **kwargs):
    return _quality_metric_call('_patient_selector', *args, **kwargs)

def _get_quality_cohort_patient_ids(*args, **kwargs):
    return _quality_metric_call('_get_quality_cohort_patient_ids', *args, **kwargs)

def _get_quality_los_by_patient(*args, **kwargs):
    return _quality_metric_call('_get_quality_los_by_patient', *args, **kwargs)

def _format_quality_density(*args, **kwargs):
    return _quality_metric_call('_format_quality_density', *args, **kwargs)

def _get_quality_denominator_note(*args, **kwargs):
    return _quality_metric_call('_get_quality_denominator_note', *args, **kwargs)

def _smd_severity_tag(*args, **kwargs):
    return _quality_metric_call('_smd_severity_tag', *args, **kwargs)

def _compute_smd_continuous(*args, **kwargs):
    return _quality_metric_call('_compute_smd_continuous', *args, **kwargs)

def _compute_smd_binary(*args, **kwargs):
    return _quality_metric_call('_compute_smd_binary', *args, **kwargs)

def _vectorized_expected_per_patient(*args, **kwargs):
    return _quality_metric_call('_vectorized_expected_per_patient', *args, **kwargs)

def _build_quality_metric_profile(*args, **kwargs):
    return _quality_metric_call('_build_quality_metric_profile', *args, **kwargs)

def _cohort_cache_fingerprint(*args, **kwargs):
    return _quality_metric_call('_cohort_cache_fingerprint', *args, **kwargs)

def _los_cache_fingerprint(*args, **kwargs):
    return _quality_metric_call('_los_cache_fingerprint', *args, **kwargs)

def _build_quality_metric_profile_cached(*args, **kwargs):
    return _quality_metric_call('_build_quality_metric_profile_cached', *args, **kwargs)

def _compute_quality_missing_rate(*args, **kwargs):
    return _quality_metric_call('_compute_quality_missing_rate', *args, **kwargs)



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


# Column normalization services are imported from easyicu.webapp.services.

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
    from easyicu.api import clear_global_loader
    from easyicu.concept import load_dictionary
    from easyicu.memory_manager import release_memory

    concept_dict = load_dictionary(include_sofa2=True)
    resolved_data_path = Path(find_database_path(str(data_path), database))
    if not resolved_data_path.exists():
        resolved_data_path = Path(data_path)
    preview_load_kwargs = dict(extra_kwargs or {})
    preview_load_kwargs.setdefault('memory_efficient', True)
    preview_load_kwargs.setdefault('concept_workers', 1)
    preview_load_kwargs.setdefault('parallel_workers', 1)
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

    try:
        if normal_concepts:
            normal_result = load_concepts(
                normal_concepts,
                database=database,
                data_path=str(resolved_data_path),
                max_patients=max_patients,
                merge=False,
                verbose=verbose,
                **preview_load_kwargs,
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
                data_path=str(resolved_data_path),
                max_patients=max_patients,
                verbose=verbose,
                **extra_kwargs,
            )
            for concept, df in (special_result or {}).items():
                if hasattr(df, 'data'):
                    df = df.data
                if isinstance(df, pd.DataFrame) and not df.empty:
                    loaded[concept] = df
    finally:
        clear_global_loader()
        release_memory(aggressive=True)

    return {
        'loaded_concepts': loaded,
        'unsupported_concepts': unsupported_concepts,
        'requested_normal': normal_concepts,
        'requested_special': special_concepts,
    }


def render_data_dictionary():
    """Render data dictionary (aligned with sidebar groups)."""
    return _render_data_dictionary_impl(globals())


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
        except Exception:
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


def _render_sepsis_ai_button(lang: str) -> None:
    """Offer a contextual AI explanation button for Sepsis settings."""
    button_label = "Ask AI about Sepsis settings" if lang == 'en' else "问 AI 解释脓毒症设置"
    if st.button(
        button_label,
        key="ask_ai_about_sepsis_settings",
        use_container_width=True,
        icon=":material/smart_toy:",
    ):
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
        _open_embedded_ai_assistant(st.session_state, question)
        st.rerun()






def _get_feature_definition_rows(selected_concepts: list[str], database: str, lang: str) -> list[dict]:
    """Build a transparent per-feature definition table for the current database."""
    return _get_feature_definition_rows_impl(selected_concepts, database, lang, globals())


def _render_feature_definition_panel(lang: str) -> None:
    """Render a transparent feature definition panel for the selected database and features."""
    return _render_feature_definition_panel_impl(lang, globals())




# ============ 辅助函数：加载后按队列条件过滤已提取数据中的 None 值患者 ============



# ============ 辅助函数：真正的 Cohort 筛选（读取 Parquet 元数据过滤患者） ============

def apply_cohort_filter(data_path, database, candidate_ids=None):
    """应用队列筛选。"""
    return _apply_cohort_filter_impl(data_path, database, candidate_ids, globals())


















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








def render_quick_visualization_page():
    """渲染快速可视化主页面。"""
    return _render_quick_visualization_page_impl(globals())




def render_entry_page():
    """渲染模式选择入口页面。"""
    return _render_entry_page_impl(globals())




def render_sidebar():
    """渲染侧边栏 - 根据entry_mode显示不同内容。"""
    return _render_sidebar_impl(globals())


def render_extract_page(lang: str):
    """渲染主区数据提取页（步骤 1-4，shell-A redesign）。"""
    from easyicu.webapp.sidebar import render_extract_page as _impl
    return _impl(lang, globals())


def _handle_sidebar_export_trigger(default_export_container) -> bool:
    """Run a queued sidebar export and keep the main page quiet while it runs."""
    if not (
        st.session_state.get('trigger_export', False)
        or st.session_state.get('_export_conflict_pending', False)
    ):
        return False

    st.session_state.trigger_export = False
    # 🔧 FIX: 添加 try-except 防止白屏崩溃
    try:
        # 导出中只定位进度区域，不再切回 Tutorial 正文，避免进度条下方继续渲染首页卡片。
        js_scroll_to_export = '''
        <script>
            (function() {
                function scrollToExportProgress() {
                    var doc = window.parent.document;
                    var anchor = doc.getElementById('export-progress');
                    if (anchor) {
                        anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
                        return true;
                    }
                    var headings = Array.from(doc.querySelectorAll('h1, h2, h3, div, p, span'));
                    var target = headings.find(function(node) {
                        var text = (node.innerText || node.textContent || '').trim();
                        return text === 'Packaging export bundle...' || text === '正在打包导出包...' ||
                            text === '📤 Export Progress' || text === '📤 导出进度';
                    });
                    if (target) {
                        target.scrollIntoView({behavior: 'smooth', block: 'start'});
                        return true;
                    }
                    return false;
                }

                setTimeout(scrollToExportProgress, 100);
                setTimeout(scrollToExportProgress, 600);
                setTimeout(scrollToExportProgress, 1400);
            })();
        </script>
        '''
        st.components.v1.html(js_scroll_to_export, height=0)

        # 仅对真正的“已导出文件导入模式”自动回填 selected_concepts；Preview 不应触发该逻辑。
        if (
            not st.session_state.get('selected_concepts')
            and st.session_state.get('loaded_data_origin') == 'exported_files'
        ):
            loaded_concepts = st.session_state.get('loaded_concepts', {})
            if loaded_concepts:
                st.session_state.selected_concepts = list(loaded_concepts.keys())

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
            warning_msg = (
                "⚠️ No features selected. Please select features in Step 3 before exporting."
                if lang == 'en'
                else "⚠️ 未选择特征。请先在步骤 3 选择要导出的特征。"
            )
            st.warning(warning_msg)
            st.session_state['_exporting_in_progress'] = False

        if st.session_state.get('_post_export_navigation_pending'):
            st.session_state['_active_main_page'] = 'quick_viz'
            st.rerun()
    except Exception as e:
        st.session_state['_exporting_in_progress'] = False
        err_msg = f"❌ Export failed: {str(e)}"
        st.error(err_msg)
        import traceback
        with st.expander("Error details" if st.session_state.get('language', 'en') == 'en' else "错误详情"):
            st.code(traceback.format_exc())
    return True




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


def render_home():
    """渲染首页 - 引导式教程，根据用户进度动态显示。"""
    return _render_home_impl(globals())


def _render_research_agent_handoff(label: str, lang: str, *, key_suffix: str) -> None:
    """Offer a one-click handoff from loaded concepts to Research Agent."""
    loaded_concepts = st.session_state.get("loaded_concepts") or {}
    if not loaded_concepts:
        return
    # 2026-05 unified counts: use the same dedup helper everywhere so the
    # handoff hint, gate guide, exports launcher, status strip, and
    # footer never disagree on "N concepts" for the same session.
    _counts = cohort_feature_counts(st.session_state)
    concept_count = _counts['features']
    patient_count = _counts['patients']
    signature = (
        tuple(sorted(str(k) for k in loaded_concepts.keys())),
        patient_count,
        st.session_state.get("id_col", "stay_id"),
    )
    if st.session_state.get("research_agent_inbound_signature") == signature:
        return

    left, right = st.columns([6.0, 1.6])
    with left:
        st.markdown(
            f'<div class="eu-handoff-note">{html.escape(get_text("ra_handoff_hint").format(concepts=concept_count, patients=patient_count))}</div>',
            unsafe_allow_html=True,
        )
    with right:
        if st.button(get_text("ra_handoff_button"), key=f"ra_handoff_{key_suffix}", use_container_width=True):
            try:
                from easyicu.webapp.research_agent import _stay_level_from_loaded_concepts

                df = _stay_level_from_loaded_concepts(
                    loaded_concepts,
                    id_col=st.session_state.get("id_col", "stay_id"),
                )
            except Exception as exc:
                st.error(f"{get_text('ra_handoff_error')} ({type(exc).__name__}: {exc})")
                return
            if df is None or df.empty:
                st.error(get_text("ra_handoff_error"))
                return
            st.session_state["research_agent_inbound_cohort"] = df
            st.session_state["research_agent_inbound_cohort_label"] = label
            st.session_state["research_agent_inbound_signature"] = signature
            st.session_state["research_agent_cohort_source"] = get_text("ra_source_handoff")
            st.session_state["_eu_ra_force_setup_from_handoff"] = True
            st.session_state["_active_main_page"] = "research_agent"
            st.session_state["_ra_view"] = "setup"
            st.session_state["research_agent_preflight_confirmed"] = False
            st.session_state.pop("research_agent_preflight_signature", None)
            message = get_text("ra_handoff_success").format(rows=len(df))
            st.session_state["_assistant_notice"] = message
            st.session_state["_eu_ra_handoff_success_message"] = message
            st.rerun()


def _research_agent_handoff_setup_ready(state: Dict[str, Any]) -> bool:
    """Return whether an in-session cohort should open the real setup surface."""
    inbound = state.get("research_agent_inbound_cohort")
    return (
        bool(state.get("_eu_ra_force_setup_from_handoff"))
        and isinstance(inbound, pd.DataFrame)
        and not inbound.empty
    )


def _research_agent_active_run_context(state: Dict[str, Any]) -> Dict[str, str]:
    """Return the active Workbench run that header-level actions can reuse."""
    workbench = state.get("_agent_workbench")
    if not isinstance(workbench, dict) or not workbench.get("steps"):
        return {}
    run_id = str(workbench.get("run_id") or "").strip()
    run_dir = str(workbench.get("run_dir") or state.get("_agent_workbench_source_run_dir") or "").strip()
    if not run_id and run_dir:
        run_id = Path(run_dir).name
    if not run_id:
        return {}
    question = str(workbench.get("research_question") or workbench.get("question") or "").strip()
    out = {"run_id": run_id, "question": question}
    if run_dir:
        out["run_dir"] = run_dir
    return out


def _prime_research_agent_header_rerun(state: Dict[str, Any], run_context: Dict[str, str]) -> None:
    """Route the header Re-run CTA to the existing checkpoint-resume setup path."""
    run_id = str(run_context.get("run_id") or "").strip()
    if not run_id:
        return
    state["research_agent_resume_run_id"] = run_id
    if run_context.get("run_dir"):
        state["research_agent_resume_run_dir"] = str(run_context["run_dir"])
    state["research_agent_force_manuscript"] = False
    state["research_agent_resume_mode"] = "continue"
    state["research_agent_resume_notes"] = ""
    state["research_agent_resume_relax_probe"] = False
    question = str(run_context.get("question") or "").strip()
    if question:
        state["research_agent_question"] = question
    state["research_agent_preflight_confirmed"] = False
    state.pop("research_agent_preflight_signature", None)
    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "setup"
    state["_research_agent_expand_history"] = False


def _render_research_agent_reference_header(lang: str, *, view: str = "setup") -> None:
    """Render the shared Agent identity before the view tabs."""
    entry_is_demo = st.session_state.get("entry_mode") == "demo"
    handoff_setup_ready = _research_agent_handoff_setup_ready(st.session_state)
    state = st.session_state.get("_agent_workbench")
    step_count = 0
    state_is_demo: bool | None = None
    if isinstance(state, dict):
        step_count = len(state.get("steps") or [])
        if step_count:
            state_is_demo = bool(state.get("is_demo"))
    is_static_guide = (entry_is_demo and not handoff_setup_ready) if state_is_demo is None else state_is_demo
    runs_value = "preview" if is_static_guide and not step_count else str(step_count)
    arm_label = "Static guide" if is_static_guide else "ICU-aware · default arm"
    arm_title = (
        "Demo mode does not call an LLM."
        if is_static_guide else
        "Web UI runs the ICU-aware arm only; the naive ablation is exposed via the CLI --arms flag."
    )
    if view == "history":
        actions = (
            f'<span class="eu-pill">{"Runs" if lang == "en" else "运行"} · '
            f'{"local history" if lang == "en" else "本地历史"}</span>'
            f'<span class="eu-pill" title="'
            f'{html.escape("Local manifests only; nothing leaves your machine." if lang == "en" else "仅读取本地 manifest；不会离开本机。")}">'
            f'<span class="dot" style="background:var(--accent)"></span>'
            f'{html.escape("Local manifests" if lang == "en" else "本地 manifest")}</span>'
        )
    else:
        actions = (
            f'<span class="eu-pill">{"Runs" if lang == "en" else "运行"} · {html.escape(runs_value)}</span>'
            f'<span class="eu-pill" title="{html.escape(arm_title)}">'
            f'<span class="dot" style="background:var(--accent)"></span>{html.escape(arm_label)}</span>'
        )
    header_html = cc.render_design_page_header(
        kicker="Research Agent · research workflow" if lang == "en" else "研究智能体 · 研究工作流",
        title_en="EasyICU Research Agent",
        title_zh="EasyICU Research Agent",
        desc=(
            "An auditable, evidence-bound workflow — plan, run, review, then draft."
            if lang == "en" else
            "可审计、证据绑定的工作流：计划、运行、复核，然后再写作。"
        ),
        right_html=actions,
        lang=lang,
    ).replace(
        'class="eu-design-page-header"',
        'class="eu-design-page-header eu-ra-reference-header"',
        1,
    )
    st.markdown(header_html, unsafe_allow_html=True)











def _render_extraction_pipeline_figure(
    *,
    lang: str,
    step1_done: bool,
    step2_done: bool,
    step3_done: bool,
    step4_done: bool,
) -> None:
    """Render the live extraction workflow using the same visual logic as Figure 2."""
    return _render_extraction_pipeline_figure_impl(
        lang=lang,
        step1_done=step1_done,
        step2_done=step2_done,
        step3_done=step3_done,
        step4_done=step4_done,
    )


def render_home_extract_mode(lang):
    """渲染首页的数据提取模式说明。"""
    return _render_home_extract_mode_impl(lang, globals())




def render_home_data_dictionary(lang):
    """在首页渲染完整的数据字典。"""
    return _render_home_data_dictionary_impl(lang, globals())


def _render_home_dict_table(concepts, lang, app_context=None):
    """为首页数据字典渲染表格。"""
    return _render_home_dict_table_impl(
        concepts,
        lang,
        app_context=app_context or globals(),
    )


def _add_clinical_thresholds(fig, concept_name: str, show: bool = True):
    """在 Plotly 时序图上添加临床阈值参考线。"""
    if not show:
        return fig
    thresholds = CLINICAL_THRESHOLDS.get(concept_name)
    if not thresholds:
        return fig
    # ``source`` (clinical guideline citation) is optional metadata added in
    # 2026-05 Phase D — surface it as the line hover so readers can see
    # provenance without cluttering the annotation text itself.
    source = thresholds.get('source', '')
    unit = thresholds.get('unit', '')
    for val, color, label in zip(thresholds['lines'], thresholds['colors'], thresholds['labels']):
        hover_text = f"{label}: {val}{(' ' + unit) if unit else ''}"
        if source:
            hover_text += f"<br><i>source: {source}</i>"
        fig.add_hline(
            y=val, line_dash="dot", line_color=color, line_width=1.5,
            opacity=0.7,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color=color,
            annotation_hovertext=hover_text,
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
    if st.button(label, key=btn_key, help=label):
        full_q = f"{label}"
        if concept:
            full_q += f" (concept: {concept})"
        if context:
            full_q += f" Context: {context}"
        _open_embedded_ai_assistant(st.session_state, full_q)
        st.toast("Question sent to AI Assistant" if st.session_state.get('language') == 'en' else "问题已发送到 AI 助手")
        st.rerun()


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




def _compact_spacer(height: int = 10):
    """Small reusable vertical spacer for dense layouts."""
    st.markdown(f"<div style='height:{height}px'></div>", unsafe_allow_html=True)


def _render_compact_divider(top: int = 6, bottom: int = 12):
    """Render a lighter divider with tighter vertical rhythm than st.markdown('---')."""
    st.markdown(
        f"""
        <div class="eu-compact-divider" style="--eu-divider-top:{top}px;--eu-divider-bottom:{bottom}px">
            <span aria-hidden="true"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )






















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


def _augment_cohort_dashboard_frame(
    df: pd.DataFrame,
    loaded_concepts: Dict[str, Any],
) -> pd.DataFrame:
    """Merge stay-level concept previews into the cohort dashboard frame."""
    if not loaded_concepts or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    id_candidates = [
        'stay_id', 'patient_id', 'subject_id', 'hadm_id', 'icustay_id',
        'patientunitstayid', 'admissionid', 'patientid', 'CaseID',
    ]
    base_id = next((col for col in id_candidates if col in df.columns), None)
    if base_id is None:
        return df

    out = df.copy()
    skip_cols = {
        'charttime', 'time', 'starttime', 'endtime', 'datetime',
        'valueuom', 'unit', 'label', 'source', 'database',
    }

    for concept, concept_df in (loaded_concepts or {}).items():
        if not isinstance(concept_df, pd.DataFrame) or concept_df.empty:
            continue
        concept_id = next((col for col in id_candidates if col in concept_df.columns), None)
        if concept_id is None:
            continue
        value_col = concept if concept in concept_df.columns else None
        if value_col is None:
            value_col = next(
                (
                    col for col in concept_df.columns
                    if col not in skip_cols and col != concept_id
                ),
                None,
            )
        if value_col is None:
            continue

        tmp = concept_df[[concept_id, value_col]].dropna(subset=[concept_id]).copy()
        if tmp.empty:
            continue
        try:
            tmp[concept_id] = tmp[concept_id].astype(out[base_id].dtype, copy=False)
        except Exception:
            out[base_id] = out[base_id].astype(str)
            tmp[concept_id] = tmp[concept_id].astype(str)
        numeric = pd.to_numeric(tmp[value_col], errors='coerce')
        if numeric.notna().any():
            tmp[value_col] = numeric
            aggregated = tmp.groupby(concept_id, as_index=False)[value_col].max()
        else:
            aggregated = tmp.groupby(concept_id, as_index=False)[value_col].first()
        target_col = str(concept)
        aggregated = aggregated.rename(columns={concept_id: base_id, value_col: target_col})
        if target_col in out.columns:
            continue
        out = out.merge(aggregated, on=base_id, how='left')

    if 'death' in out.columns and 'mortality' not in out.columns:
        out['mortality'] = out['death']
    if 'los_icu' in out.columns and 'los_hours' not in out.columns:
        out['los_hours'] = out['los_icu']
    if 'vaso' in out.columns and 'vasopressors' not in out.columns:
        out['vasopressors'] = out['vaso']
    return out


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
    df = _augment_cohort_dashboard_frame(df, loaded_concepts)
    total = len(df)
    sofa = _cohort_numeric_series(df, ['sofa_max', 'sofa2_max', 'sofa2', 'sofa'])
    los_hours = _cohort_numeric_series(df, ['los_hours', 'los_icu'])
    los_days = los_hours / 24 if los_hours is not None else _cohort_numeric_series(df, ['los_days'])
    age = _cohort_numeric_series(df, ['age'])
    survived = _cohort_bool_series(df, ['survived'])
    mortality_series = _cohort_bool_series(df, ['mortality', 'death'])
    if mortality_series is None and survived is not None:
        mortality_series = ~survived

    sepsis = _cohort_bool_series(df, ['sepsis', 'sepsis3'])
    if sepsis is None and 'diagnosis_group' in df.columns:
        sepsis = df['diagnosis_group'].astype(str).str.lower().str.contains('sepsis', na=False)

    phenotype_defs = [
        ('Sepsis' if lang == 'en' else '脓毒症', sepsis),
        ('AKI' if lang == 'en' else '急性肾损伤', _cohort_bool_series(df, ['aki', 'aki_stage'])),
        ('RRT' if lang == 'en' else '肾脏替代治疗', _cohort_bool_series(df, ['rrt'])),
        ('Mechanical ventilation' if lang == 'en' else '机械通气', _cohort_bool_series(df, ['mech_vent', 'ventilation', 'vent', 'vent_ind'])),
        ('Vasopressors' if lang == 'en' else '血管活性药物', _cohort_bool_series(df, ['vasopressors', 'vaso_ind', 'vaso'])),
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

def _build_data_coverage_audit(df: pd.DataFrame, loaded_concepts: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """Build the S1B-style coverage matrix and eligibility flow."""
    return _build_data_coverage_audit_impl(df, loaded_concepts, lang, globals())


def render_data_coverage_audit_subtab(lang: str):
    """Render a figure-aligned data coverage and eligibility audit panel."""
    return _render_data_coverage_audit_subtab_impl(lang, globals())


SOFA_RECLASS_ORGANS = _sofa_reclassification_impl.SOFA_RECLASS_ORGANS
SOFA_RECLASS_ANALYSIS_MODES = _sofa_reclassification_impl.SOFA_RECLASS_ANALYSIS_MODES

def _sofa_reclassification_call(name: str, *args, **kwargs):
    _sofa_reclassification_impl._install_app_context(globals())
    return getattr(_sofa_reclassification_impl, name)(*args, **kwargs)

def _generate_mock_sofa_timeseries_concepts(*args, **kwargs):
    return _sofa_reclassification_call('_generate_mock_sofa_timeseries_concepts', *args, **kwargs)

def _demo_cohort_fingerprint(*args, **kwargs):
    return _sofa_reclassification_call('_demo_cohort_fingerprint', *args, **kwargs)

def _get_demo_sofa_timeseries_concepts(*args, **kwargs):
    return _sofa_reclassification_call('_get_demo_sofa_timeseries_concepts', *args, **kwargs)

def _get_sofa_reclassification_mode_availability(*args, **kwargs):
    return _sofa_reclassification_call('_get_sofa_reclassification_mode_availability', *args, **kwargs)

def _sofa_severity_group(*args, **kwargs):
    return _sofa_reclassification_call('_sofa_severity_group', *args, **kwargs)

def _build_sofa_reclassification_stats(*args, **kwargs):
    return _sofa_reclassification_call('_build_sofa_reclassification_stats', *args, **kwargs)

def _build_reclassification_df_from_loaded_concepts(*args, **kwargs):
    return _sofa_reclassification_call('_build_reclassification_df_from_loaded_concepts', *args, **kwargs)

def _get_sofa_reclassification_source(*args, **kwargs):
    return _sofa_reclassification_call('_get_sofa_reclassification_source', *args, **kwargs)

def _render_reclassification_cards(*args, **kwargs):
    return _sofa_reclassification_call('_render_reclassification_cards', *args, **kwargs)

def _render_reclassification_snapshot(*args, **kwargs):
    return _sofa_reclassification_call('_render_reclassification_snapshot', *args, **kwargs)



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

    render_page_header(
        get_text('page_cohort_title'),
        get_text('page_cohort_subtitle'),
        icon="📊",
        kicker=get_text('page_cohort_kicker'),
    )

    if st.session_state.get('entry_mode') == 'demo':
        if not _cohort_demo_workspace_ready(st.session_state):
            _render_cohort_demo_workspace_launcher(lang)
            return
        if not screenshot_mode:
            _render_cohort_demo_workspace_status(lang)

    elif st.session_state.get('entry_mode') == 'real':
        # Two valid sources for the shared workspace:
        #   1. A sidebar-validated raw ICU data root (icustays/patients/...)
        #      — feeds all 5 panels including Cross-DB Benchmark.
        #   2. Module exports already loaded via Quick Visualization
        #      (state['loaded_concepts']) — feeds Group Contrast / Coverage Audit /
        #      Cohort Profile / SOFA Reclassification. Cross-DB still needs ≥2 DBs' raw schema.
        # Gate only when neither is available; otherwise let the launcher
        # offer the matching one-click prep.
        _real_data_path = _default_real_data_root()
        _has_raw_path = bool(_real_data_path) and Path(_real_data_path).exists()
        _has_loaded_exports = bool(st.session_state.get('loaded_concepts'))
        if not _has_raw_path and not _has_loaded_exports:
            if not screenshot_mode:
                _render_cohort_real_no_path_guide(lang)
                return

        # Real data: offer a one-click shared workspace without hiding the
        # original panel-level raw/exported-data import flows.
        if not _cohort_real_workspace_ready(st.session_state):
            if not screenshot_mode:
                if _has_raw_path:
                    _render_cohort_real_workspace_launcher(lang)
                else:
                    _render_cohort_real_loaded_exports_launcher(lang)
        else:
            if not _cohort_real_workspace_matches_sidebar(st.session_state):
                # Sidebar path changed since last workspace load
                st.warning(
                    "⚠️ " + ("Sidebar data source changed. Reload the workspace to update."
                              if lang == 'en' else "侧边栏数据源已变更，请重新加载工作区。")
                )
            if not screenshot_mode:
                _render_cohort_real_workspace_status(lang)

    # Streamlit tabs eagerly execute every tab body on every rerun. These
    # cohort panels build multiple charts and tables, so a lazy segmented
    # switcher keeps clicks responsive by rendering only the selected panel.
    panel_labels = (
        {
            "groups": "Group contrast",
            "coverage": "Coverage audit",
            "snapshot": "Cohort profile",
            "sofa": "SOFA reclassification",
        }
        if lang == "en"
        else {
            "groups": "组间对照",
            "coverage": "覆盖审计",
            "snapshot": "队列画像",
            "sofa": "SOFA 重分层",
        }
    )
    panel_keys = list(panel_labels)
    panel_state_key = "cohort_active_panel"
    if st.session_state.get(panel_state_key) not in panel_keys:
        st.session_state[panel_state_key] = panel_keys[0]

    st.markdown(
        f'<div class="inline-control-label">{html.escape("Cohort panel" if lang == "en" else "队列面板")}</div>',
        unsafe_allow_html=True,
    )
    active_panel = st.radio(
        "Cohort panel" if lang == "en" else "队列面板",
        options=panel_keys,
        format_func=lambda key: panel_labels.get(key, key),
        horizontal=True,
        key=panel_state_key,
        label_visibility="collapsed",
    )
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    if active_panel == "groups":
        render_group_comparison_subtab(lang)
    elif active_panel == "coverage":
        render_data_coverage_audit_subtab(lang)
    elif active_panel == "snapshot":
        render_cohort_dashboard_subtab(lang)
    elif active_panel == "sofa":
        render_severity_reclassification_subtab(lang)


def _render_demo_generation_card(icon: str, title: str, desc: str):
    """统一的 demo 生成空状态卡片。"""
    st.markdown(
        f'''
        <div style="text-align:center;padding:30px 28px;background:linear-gradient(135deg,#eff6ff 0%,#f0fdfa 55%,#f8fafc 100%);
                    border:1px solid #bfdbfe;border-radius:18px;margin:16px 0 18px;box-shadow:0 10px 28px rgba(37,99,235,0.08)">
            <div style="width:64px;height:64px;margin:0 auto 14px;border-radius:18px;background:linear-gradient(135deg,#2563eb 0%,#0891b2 100%);
                        display:flex;align-items:center;justify-content:center;font-size:1.9rem;box-shadow:0 10px 24px rgba(37,99,235,0.18)">{icon}</div>
            <div style="font-weight:800;color:#0f172a;font-size:1.4rem;letter-spacing:0">{title}</div>
            <div style="color:#475569;font-size:.95rem;margin-top:8px;line-height:1.65">{desc}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_cohort_real_no_path_guide(lang: str) -> None:
    """Single guide for Cohort Analysis (real mode) before a data path is validated.

    Replaces what used to be a contradictory mix of a "go to Step 1" banner
    plus fully-rendered sub-tabs and per-panel Data Configuration forms.

    When module exports have already been loaded via Quick Visualization, the
    guide explicitly acknowledges that state instead of contradicting the
    "Data Loaded" footer — Cohort Analysis still needs the raw DB schema
    (icustays/patients/admissions), which exports don't carry.
    """
    loaded_concepts = st.session_state.get('loaded_concepts') or {}
    _counts = cohort_feature_counts(st.session_state)
    n_concepts = _counts['features']
    n_patients = _counts['patients']
    has_exports = n_concepts > 0 and n_patients > 0

    if has_exports:
        if lang == 'en':
            title = "Cohort Analysis needs the raw database schema"
            desc = (
                f"You already have <b>{n_concepts} concepts × {n_patients} patients</b> "
                "loaded from module exports — those power <b>Quick Visualization</b> "
                "(tables, time series, patient overview, data quality) and "
                "<b>Research Agent</b>.<br><br>"
                "The Cohort Analysis panels (group contrast, coverage audit, "
                "cross-DB benchmark, cohort profile, and SOFA reclassification) "
                "read <code>icustays</code> / <code>patients</code> / "
                "<code>admissions</code> directly, so they require a converted raw "
                "ICU root. Use <b>🔄 Convert &amp; Setup</b> in the sidebar to "
                "convert raw CSV/CSV.GZ/tar.gz inputs into the expected layout, "
                "or point <b>Step 1 · Data Source</b> at an already-converted root."
            )
        else:
            title = "队列分析需要原始数据库 schema"
            desc = (
                f"你已经从导出模块加载了 <b>{n_concepts} 个概念 × {n_patients} 个患者</b>"
                "——这些数据可在 <b>快速可视化</b>（表格、时间序列、患者概览、"
                "数据质量）和 <b>研究智能体</b> 中使用。<br><br>"
                "队列分析的各个面板（组间对照、覆盖审计、跨数据库对比、队列画像、SOFA 重分层）"
                "直接读取 <code>icustays</code> / <code>patients</code> / "
                "<code>admissions</code> 表，因此需要一个已转换的原始 ICU 数据根。"
                "请在侧边栏使用 <b>🔄 Convert &amp; Setup</b> 把原始 "
                "CSV/CSV.GZ/tar.gz 转换成所需布局，或让 <b>步骤 1 · 数据源</b> "
                "指向一个已转换好的目录。"
            )
        _render_demo_generation_card("🔄", title, desc)
        return

    if lang == 'en':
        title = "Validate a data path first"
        desc = ("Cohort Analysis needs a real ICU database. Open "
                "<b>Step 1 · Data Source</b> in the sidebar, enter your ICU data "
                "root and validate it — all cohort panels unlock once the path "
                "is confirmed.")
    else:
        title = "请先验证数据路径"
        desc = ("队列分析需要真实 ICU 数据库。请在侧边栏打开 <b>步骤 1 · 数据源</b>，"
                "填入 ICU 数据根目录并完成验证——路径确认后所有队列面板会自动解锁。")
    _render_demo_generation_card("📁", title, desc)


def _render_cohort_demo_workspace_status(lang: str) -> None:
    """Render one shared demo status strip for all Cohort Analysis panels."""
    title = "Shared demo cohort workspace" if lang == 'en' else "共享演示队列工作区"
    subtitle = (
        "Group contrast, coverage audit, cross-database benchmark, cohort profile, and SOFA reclassification now use the same prepared demo state."
        if lang == 'en' else
        "组间对照、覆盖审计、跨数据库对比、队列画像和 SOFA 重分层现在共用同一套演示状态。"
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
            min_value=10,
            max_value=50,
            value=min(50, max(10, int((st.session_state.get('mock_params') or {}).get('n_patients', LIGHTWEIGHT_DEMO_PATIENTS)))),
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
        "This prepares demo data for group contrast, coverage audit, cross-DB benchmark, cohort profile, and SOFA reclassification together. You will not need to generate data again inside each subpanel."
        if lang == 'en' else
        "这会一次性准备组间对照、覆盖审计、跨数据库对比、队列画像和 SOFA 重分层所需的演示数据；之后不需要在每个子板块重复生成。"
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
            min_value=10,
            max_value=50,
            value=min(50, max(10, int((st.session_state.get('mock_params') or {}).get('n_patients', LIGHTWEIGHT_DEMO_PATIENTS)))),
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




def convert_csv_to_parquet(
    source_dir: str,
    overwrite: bool = False,
) -> tuple:
    """将 CSV 数据转换为 Parquet（含 HiRID 归档解压，统一走 DataConverter）。"""
    return _convert_csv_to_parquet_impl(
        source_dir,
        overwrite,
        globals(),
    )




def _generate_cohort_prefix() -> str:
    """根据队列筛选条件生成文件名前缀。"""
    return _generate_cohort_prefix_impl(globals())


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
    return _write_export_manifest_impl(
        export_dir,
        exported_files=exported_files,
        patient_count=patient_count,
        concept_count=concept_count,
        export_format=export_format,
        unavailable_concepts=unavailable_concepts,
        unsupported_concepts=unsupported_concepts,
        empty_data_concepts=empty_data_concepts,
        failed_concepts=failed_concepts,
        note=note,
        app_context=globals(),
    )


def _build_quick_viz_pdf_report(*, lang: str, preview_data: dict[str, pd.DataFrame], concepts_to_export: list[str]) -> bytes:
    """Create a compact one-file PDF summary for Quick Visualization."""
    return _build_quick_viz_pdf_report_impl(
        lang=lang,
        preview_data=preview_data,
        concepts_to_export=concepts_to_export,
        app_context=globals(),
    )


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
    st.session_state.pop('_post_export_guidance_dismissed', None)

    if auto_load:
        selected_files = list(dict.fromkeys(Path(path).stem for path in files if path))
        max_patients_opt = st.session_state.get('viz_max_patients', 100)
        max_patients = None if max_patients_opt in (None, -1) else max_patients_opt
        st.session_state['_viz_auto_load_export'] = {
            'path': str(export_dir),
            'selected_files': selected_files or None,
            'max_patients': max_patients,
        }
    _route_completed_export_to_visualization(
        st.session_state,
        request_refresh=True,
        sync_widget_keys=False,
    )


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
    raw_target = query_param_value(st, "figure")
    section, panel = _normalize_figure_target(raw_target)
    if section and panel:
        return section, panel
    for key in ("panel", "page", "view"):
        section, panel = _normalize_figure_target(query_param_value(st, key))
        if section and panel:
            return section, panel
    return '', ''


def _ensure_quick_figure_demo_data(state: dict[str, Any], *, lang: str) -> None:
    """Preload compact demo concepts so figure URLs open directly to useful panels."""
    if state.get('loaded_concepts'):
        return
    state['mock_params'] = {
        'n_patients': LIGHTWEIGHT_DEMO_PATIENTS,
        'hours': LIGHTWEIGHT_DEMO_HOURS,
        'demo_profile': 'lite',
    }
    mock_data, patient_ids = generate_lightweight_demo_data(
        n_patients=LIGHTWEIGHT_DEMO_PATIENTS,
        hours=LIGHTWEIGHT_DEMO_HOURS,
    )
    state['loaded_concepts'] = mock_data
    state['loaded_data_origin'] = 'demo_viz'
    state['patient_ids'] = patient_ids
    state['id_col'] = 'stay_id'
    state['time_col'] = 'time'
    state['selected_concepts'] = list(mock_data.keys())
    _apply_quick_viz_screenshot_defaults(state, lang=lang)










# ---------------------------------------------------------------------------
# Real Data Shared Workspace  (P0-2: mirrors demo workspace for real data)
# ---------------------------------------------------------------------------


# Concepts loaded by default for the shared real-data workspace.








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

    title = ("Optional: prepare all cohort panels from current real data" if lang == 'en'
             else "可选：从当前真实数据准备所有队列面板")
    subtitle = (
        f"Quick preview: load demographics, preview concepts, and SOFA for **{db_label}** from `{data_path}`. "
        "You can also use the group contrast or cohort profile panels below to load raw databases or exported results as before."
        if lang == 'en' else
        f"快速预览：加载 **{db_label}** (`{data_path}`) 的人口统计、预览概念和 SOFA。"
        "你也可以像以前一样，在下方组间对照或队列画像面板里加载原始数据库或已导出的结果。"
    )
    st.markdown(
        f'''
        <div class="cohort-demo-workspace">
            <div class="cohort-demo-badge" style="background:linear-gradient(135deg,#059669 0%,#0891b2 100%)">R</div>
            <div>
                <div class="cohort-demo-title">{html.escape(title)}</div>
                <div class="cohort-demo-subtitle">{html.escape(subtitle)}</div>
            </div>
            <div class="cohort-demo-status" style="color:#059669">optional shortcut</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        max_patients = st.slider(
            "Max patients to load" if lang == 'en' else "最大加载患者数",
            min_value=100,
            max_value=_REAL_WORKSPACE_MAX_PATIENTS,
            value=_REAL_WORKSPACE_DEFAULT_MAX_PATIENTS,
            step=100,
            help=(
                "Start with 100 for a fast import check; increase this after confirming the panels work."
                if lang == 'en' else
                "默认 100 用于快速检查导入；确认面板可用后可再调大样本量。"
            ),
            key="cohort_real_workspace_max_patients_init",
        )
    with col2:
        _compact_spacer(26)
        if st.button(
            "🚀 " + ("Load real workspace for all panels" if lang == 'en' else "加载真实数据工作区"),
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


def _render_cohort_real_loaded_exports_launcher(lang: str) -> None:
    """Offer a one-click bridge from already-loaded module exports to the
    shared Cohort Analysis workspace.

    Shown only when the sidebar Data Path is NOT validated but
    ``state['loaded_concepts']`` is non-empty (the user came in via Quick
    Visualization's "Previously Exported Data" path). Backs the bridge with
    :func:`_ensure_cohort_real_workspace_from_loaded_concepts`. Cross-DB
    Benchmark is intentionally left gated because it needs raw schema of
    multiple databases — exports of one DB can't fake the others.
    """
    _counts = cohort_feature_counts(st.session_state)
    n_concepts = _counts['features']
    n_patients = _counts['patients']
    db = st.session_state.get('database') or _default_real_database()
    db_labels = {'miiv': 'MIMIC-IV', 'eicu': 'eICU', 'aumc': 'AUMC', 'hirid': 'HiRID', 'mimic': 'MIMIC-III', 'sic': 'SICdb'}
    db_label = db_labels.get(db, db)

    if lang == 'en':
        title = "Use loaded module exports for Cohort Analysis"
        subtitle = (
            f"Bridge the <b>{n_concepts} concepts × {n_patients} patients</b> already "
            f"loaded for <b>{db_label}</b> into Group Contrast, Coverage Audit, Cohort Profile, "
            "and SOFA Reclassification. Cross-DB Benchmark stays gated — it needs raw schema for ≥2 databases."
        )
        button_label = "🚀 Build workspace from loaded exports"
        status_chip = "exports shortcut"
    else:
        title = "用已加载的模块导出运行队列分析"
        subtitle = (
            f"将已为 <b>{db_label}</b> 加载的 <b>{n_concepts} 个概念 × {n_patients} 个患者</b>"
            "桥接到组间对照、覆盖审计、队列画像、SOFA 重分层面板。"
            "跨数据库对比面板仍需要原始 schema（至少两个数据库），保持待解锁。"
        )
        button_label = "🚀 用已加载导出构建工作区"
        status_chip = "导出快捷"

    st.markdown(
        f'''
        <div class="cohort-demo-workspace">
            <div class="cohort-demo-badge" style="background:linear-gradient(135deg,#0891b2 0%,#7c3aed 100%)">E</div>
            <div>
                <div class="cohort-demo-title">{html.escape(title)}</div>
                <div class="cohort-demo-subtitle">{subtitle}</div>
            </div>
            <div class="cohort-demo-status" style="color:#0891b2">{html.escape(status_chip)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if st.button(
        button_label,
        type="primary",
        use_container_width=True,
        key="cohort_real_exports_bridge",
    ):
        with st.spinner("Bridging exports..." if lang == 'en' else "正在桥接导出数据..."):
            ok, msg = _ensure_cohort_real_workspace_from_loaded_concepts(
                st.session_state, lang=lang,
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
    # 2026-05 unified counts: dedup-counted features instead of raw len()
    # so this strip matches the gate / launcher / footer for the same
    # cohort. The workspace's concepts dict is built from loaded_concepts
    # so cohort_feature_counts is the correct source.
    n_concepts = count_unique_concepts(list(state.get('_cohort_real_ws_concepts', {}).keys()))
    errors = state.get('_cohort_real_ws_errors', [])
    is_exports = state.get('_cohort_real_ws_origin') == 'loaded_exports'

    if is_exports:
        title = (
            f"Shared workspace from module exports — {db_label}"
            if lang == 'en' else
            f"基于模块导出的共享工作区 — {db_label}"
        )
        subtitle = (
            f"{n:,} patients, {n_concepts} concepts bridged from loaded exports. "
            "Cross-DB Benchmark stays gated (needs raw schema for ≥2 DBs)."
            if lang == 'en' else
            f"已从导出文件桥接 {n:,} 名患者、{n_concepts} 个概念。"
            "跨数据库对比面板仍待解锁（需要至少两个数据库的原始 schema）。"
        )
    else:
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
                if is_exports:
                    ok, msg = _ensure_cohort_real_workspace_from_loaded_concepts(
                        st.session_state, lang=lang,
                    )
                else:
                    ok, msg = _ensure_cohort_real_workspace(
                        st.session_state, lang=lang,
                        max_patients=state.get('_cohort_real_ws_max_patients', _REAL_WORKSPACE_DEFAULT_MAX_PATIENTS),
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
        state['mock_params'] = {
            'n_patients': LIGHTWEIGHT_DEMO_PATIENTS,
            'hours': LIGHTWEIGHT_DEMO_HOURS,
            'demo_profile': 'lite',
        }

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
            'Coverage Audit': '覆盖审计',
            'Cross-DB Benchmark': '多库',
            'Cohort Snapshot': '队列画像',
            'SOFA-1 vs SOFA-2': 'SOFA 重分层',
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

    candidates = []
    env_dir = os.environ.get("EASYICU_PUBLICATION_FIGURE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / filename)

    candidates.extend([
        Path(__file__).resolve().parents[4]
        / 'easyicu写作'
        / 'final_figure_layout'
        / 'image2_generated_review'
        / filename,
    ])
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






































def render_quick_figure_panel(panel: str) -> None:
    """渲染 quick visualization 论文图面板。"""
    return _render_quick_figure_panel_impl(panel, globals())






def render_cohort_figure_panel(panel: str) -> None:
    """渲染 cohort 论文图面板。"""
    return _render_cohort_figure_panel_impl(panel, globals())


def _render_global_status_strip(lang: str, entry_mode: str) -> None:
    """Render a compact source-of-truth status strip for every work page."""
    db_display = {
        "mock": "Mock",
        "miiv": "MIMIC-IV",
        "eicu": "eICU-CRD",
        "aumc": "AmsterdamUMCdb",
        "hirid": "HiRID",
        "mimic": "MIMIC-III",
        "sic": "SICdb",
    }
    database = st.session_state.get("database", "")
    db_label = db_display.get(database, database.upper() if database else "—")

    if entry_mode == "demo":
        mode_value = "Demo" if lang == "en" else "演示"
        mode_tone = "demo"
        path_value = "mock data" if lang == "en" else "模拟数据"
        path_tone = "ok"
    else:
        mode_value = "Real data" if lang == "en" else "真实数据"
        mode_tone = "real"
        data_path = str(st.session_state.get("data_path") or "")
        if st.session_state.get("path_validated"):
            path_value = "valid" if lang == "en" else "已验证"
            path_tone = "ok"
        elif data_path:
            path_value = "needs validation" if lang == "en" else "待验证"
            path_tone = "warn"
        else:
            path_value = "not set" if lang == "en" else "未设置"
            path_tone = "warn"

    cohort_value = (
        "configured" if lang == "en" else "已配置"
    ) if st.session_state.get("step2_confirmed") else (
        "not set" if lang == "en" else "未设置"
    )
    cohort_tone = "ok" if st.session_state.get("step2_confirmed") else "warn"

    selected = st.session_state.get("selected_concepts", []) or []
    features_value = (
        f"{len(selected)} selected" if lang == "en" else f"已选 {len(selected)}"
    ) if selected else ("none" if lang == "en" else "未选择")
    features_tone = "ok" if selected else "warn"

    counts = cohort_feature_counts(st.session_state)
    patients = counts.get("patients", 0) or len(st.session_state.get("patient_ids", []) or [])
    patients_value = f"{patients:,}" if patients else ("not loaded" if lang == "en" else "未加载")
    patients_tone = "ok" if patients else ""

    if st.session_state.get("_exporting_in_progress"):
        export_value, export_tone = ("running" if lang == "en" else "进行中"), "info"
    elif st.session_state.get("export_completed"):
        export_value, export_tone = ("completed" if lang == "en" else "已完成"), "ok"
    elif st.session_state.get("step3_confirmed") and selected:
        export_value, export_tone = ("ready" if lang == "en" else "就绪"), "info"
    else:
        export_value, export_tone = ("not started" if lang == "en" else "未开始"), ""

    labels = {
        "mode": "Mode" if lang == "en" else "模式",
        "db": "Database" if lang == "en" else "数据库",
        "path": "Path" if lang == "en" else "路径",
        "cohort": "Cohort" if lang == "en" else "队列",
        "features": "Features" if lang == "en" else "特征",
        "patients": "Patients" if lang == "en" else "患者",
        "export": "Export" if lang == "en" else "导出",
        "privacy": "Privacy" if lang == "en" else "隐私",
    }
    items = [
        (labels["mode"], mode_value, mode_tone),
        (labels["db"], db_label, ""),
        (labels["path"], path_value, path_tone),
        (labels["cohort"], cohort_value, cohort_tone),
        (labels["features"], features_value, features_tone),
        (labels["patients"], patients_value, patients_tone),
        (labels["export"], export_value, export_tone),
        (labels["privacy"], "local only" if lang == "en" else "仅本地", "ok"),
    ]
    body = "".join(
        '<div class="eu-status-item">'
        f'<span class="label">{html.escape(label)}</span>'
        f'<span class="value {html.escape(tone)}">{html.escape(str(value))}</span>'
        '</div>'
        for label, value, tone in items
    )
    st.markdown(f'<div class="eu-status-strip">{body}</div>', unsafe_allow_html=True)












def main():
    """主函数。"""
    init_session_state()
    app_state = get_state()
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
        # Shell-A redesign: entry/mode-selection screen now uses the
        # design-canvas layout from page-entry.jsx.
        from easyicu.webapp.pages_redesign import render_entry_redesign_page
        render_entry_redesign_page(lang)
        return
        return

    _apply_assistant_preset()
    _maybe_materialize_pending_preset()

    export_in_progress = bool(
        st.session_state.get('trigger_export', False)
        or st.session_state.get('_exporting_in_progress', False)
        or st.session_state.get('_export_conflict_pending', False)
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

    if export_in_progress:
        if not _handle_sidebar_export_trigger(default_export_container):
            with default_export_container:
                st.markdown(
                    '<div class="compact-inline-notice info">'
                    + ("⏳ Export in progress. Keep this tab open." if lang == 'en' else "⏳ 正在导出，请保持此页面打开。")
                    + '</div>',
                    unsafe_allow_html=True,
                )
        return

    # ============ Shell-A top bar (breadcrumb + actions) ============
    _consume_completed_export_navigation(st.session_state)

    from easyicu.webapp.ui_helpers import (
        render_pill_html as _render_pill_html,
    )
    _active = st.session_state.get('_active_main_page', 'tutorial')
    _page_labels_for_topbar = {
        'extract':        'Data Extraction' if lang == 'en' else '数据提取',
        'tutorial':       'Get Started' if lang == 'en' else '开始使用',
        'assistant':      'AI Assistant' if lang == 'en' else 'AI 助手',
        'states':         'Workspace States' if lang == 'en' else '工作区状态',
        'settings':       'Settings' if lang == 'en' else '设置',
        'quick_viz':      'Patient Review' if lang == 'en' else '患者审阅',
        'cohort':         'Cohort Statistics' if lang == 'en' else '队列统计',
        'cross_db':       'Cross-DB Benchmark' if lang == 'en' else '跨数据库对比',
        'research_agent': 'Research Agent' if lang == 'en' else '研究智能体',
    }
    _topbar_home_label = 'Home' if lang == 'en' else '首页'
    _data_extraction_label = 'Data extraction' if lang == 'en' else '数据提取'
    _data_visualization_label = 'Data visualization' if lang == 'en' else '数据可视化'
    _tools_label = 'Tools' if lang == 'en' else '工具'
    _reference_label = 'Reference' if lang == 'en' else '参考'
    if _active == 'extract':
        if not st.session_state.get('step1_confirmed'):
            _step_crumb = 'Step 1 · Data source' if lang == 'en' else '第 1 步 · 数据源'
        elif not st.session_state.get('step2_confirmed'):
            _step_crumb = 'Step 2 · Cohort' if lang == 'en' else '第 2 步 · 队列'
        elif not st.session_state.get('step3_confirmed'):
            _step_crumb = 'Step 3 · Concepts' if lang == 'en' else '第 3 步 · 变量'
        else:
            _step_crumb = 'Step 4 · Export' if lang == 'en' else '第 4 步 · 导出'
        _mode_crumb = 'Demo' if entry_mode == 'demo' else ('Real Data' if lang == 'en' else '真实数据')
    def _render_topbar_path_nav(
        items: list[tuple[str, str | None]],
        *,
        key: str,
    ) -> None:
        visible_items = [(label, target) for label, target in items if label]
        if not visible_items:
            return
        widths: list[float] = []
        for idx, (label, _) in enumerate(visible_items):
            widths.append(max(0.24, min(1.05, len(label) / 16)))
            if idx < len(visible_items) - 1:
                widths.append(0.04)

        with st.container(key=key):
            bc_cols = st.columns(widths, gap="small", vertical_alignment="center")
            col_idx = 0
            for idx, (label, target) in enumerate(visible_items):
                with bc_cols[col_idx]:
                    if target:
                        if st.button(
                            label,
                            key=f"_{key}_{idx}_{target}",
                            use_container_width=False,
                            help=("Go to parent path" if lang == "en" else "返回上层路径"),
                        ):
                            _apply_topbar_breadcrumb_target(st.session_state, target)
                            st.rerun()
                    else:
                        st.markdown(
                            f'<span class="eu-bc-current">{html.escape(label)}</span>',
                            unsafe_allow_html=True,
                        )
                col_idx += 1
                if idx < len(visible_items) - 1:
                    with bc_cols[col_idx]:
                        st.markdown('<span class="eu-bc-sep">/</span>', unsafe_allow_html=True)
                    col_idx += 1

    _topbar_path_items_by_page: dict[str, list[tuple[str, str | None]]] = {
        'tutorial': [
            (_topbar_home_label, 'entry'),
            (_tools_label, None),
            (_page_labels_for_topbar['tutorial'], None),
        ],
        'assistant': [
            (_topbar_home_label, 'entry'),
            (_tools_label, None),
            (_page_labels_for_topbar['assistant'], None),
        ],
        'states': [
            (_topbar_home_label, 'entry'),
            (_reference_label, None),
            (_page_labels_for_topbar['states'], None),
        ],
        'settings': [
            (_topbar_home_label, 'entry'),
            (_page_labels_for_topbar['settings'], None),
        ],
        'quick_viz': [
            (_topbar_home_label, 'entry'),
            (_data_visualization_label, 'data_visualization'),
            (_page_labels_for_topbar['quick_viz'], None),
        ],
        'cohort': [
            (_topbar_home_label, 'entry'),
            (_data_visualization_label, 'data_visualization'),
            (_page_labels_for_topbar['cohort'], None),
        ],
        'cross_db': [
            (_topbar_home_label, 'entry'),
            (_data_visualization_label, 'data_visualization'),
            (_page_labels_for_topbar['cross_db'], None),
        ],
        'research_agent': [
            (_topbar_home_label, 'entry'),
            (_page_labels_for_topbar['research_agent'], None),
        ],
    }

    if _active == "extract":
        if not st.session_state.get('step1_confirmed'):
            _stage_label = '1/4  Data source' if lang == 'en' else '1/4  数据源'
        elif not st.session_state.get('step2_confirmed'):
            _stage_label = '2/4  Cohort' if lang == 'en' else '2/4  队列'
        elif not st.session_state.get('step3_confirmed'):
            _stage_label = '3/4  Concepts' if lang == 'en' else '3/4  变量'
        else:
            _stage_label = '4/4  Export' if lang == 'en' else '4/4  导出'

        _tb_left, _tb_stage = st.columns(
            [8.55, 1.45], gap="small"
        )
        with _tb_left:
            _render_topbar_path_nav(
                [
                    (_topbar_home_label, 'entry'),
                    (_data_extraction_label, 'data_extraction'),
                    (_step_crumb, None),
                    (_mode_crumb, None),
                ],
                key="eu_extract_breadcrumb_nav",
            )
        with _tb_stage:
            st.markdown(
                '<div class="eu-topbar-stage">'
                f'{_render_pill_html(_stage_label, tone="neutral")}'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        if _active == 'tutorial':
            # Tutorial already has explicit entry CTAs in the page body, so
            # the global action buttons are withheld here to avoid duplicate,
            # ambiguous calls to action.
            _tb_left = st.container()
            with _tb_left:
                _render_topbar_path_nav(
                    _topbar_path_items_by_page.get(
                        _active,
                        [
                            (_topbar_home_label, 'entry'),
                            (_page_labels_for_topbar.get(_active, _active), None),
                        ],
                    ),
                    key=f"eu_page_breadcrumb_nav_{_active}",
                )
        else:
            # Shell-A topbar: parent breadcrumb buttons plus one explicit action.
            _tb_left, _tb_run = st.columns(
                [8.6, 1.4], gap="small"
            )
            with _tb_left:
                _render_topbar_path_nav(
                    _topbar_path_items_by_page.get(
                        _active,
                        [
                            (_topbar_home_label, 'entry'),
                            (_page_labels_for_topbar.get(_active, _active), None),
                        ],
                    ),
                    key=f"eu_page_breadcrumb_nav_{_active}",
                )
            with _tb_run:
                _run_en, _run_zh = _topbar_primary_action_label(
                    _active,
                    lang,
                    entry_mode=entry_mode,
                )
                _run_button_kwargs = {
                    "key": "_eu_topbar_run",
                    "type": "primary",
                    "use_container_width": True,
                }
                _run_icon = _topbar_primary_action_icon(_active)
                if _run_icon:
                    _run_button_kwargs["icon"] = _run_icon
                if st.button(_run_en if lang == 'en' else _run_zh, **_run_button_kwargs):
                    st.session_state['_eu_topbar_run_request'] = {
                        'page': _active,
                        'requested_at': 'now',
                    }

    _render_global_status_strip(lang, entry_mode)
    _render_narrow_view_notice(_active, lang)

    if st.session_state.get('_eu_show_history'):
        with st.expander(
            "⟳ " + ("Activity" if lang == 'en' else "活动"),
            expanded=True,
        ):
            recent = st.session_state.get('_eu_action_log') or []
            if recent:
                for item in recent[-8:][::-1]:
                    st.markdown(f"- {item}")
            else:
                _empty_history = 'No recent actions yet.' if lang == 'en' else '暂无近期活动。'
                st.markdown(
                    f"<div style='color:var(--ink-3);font-size:12.5px'>{_empty_history}</div>",
                    unsafe_allow_html=True,
                )

    assistant_notice = st.session_state.pop('_assistant_notice', None)
    if assistant_notice:
        st.success(assistant_notice)

    _render_icd_preview_main_panel(lang)

    page_registry = build_main_page_registry(get_text)
    page_keys = [page["key"] for page in page_registry]
    page_labels = {page["key"]: page["label"] for page in page_registry}
    mobile_page_keys = ["extract"] + page_keys + ["assistant", "states", "settings"]
    page_labels["extract"] = "Data Extraction" if lang == "en" else "数据提取"
    page_labels["tutorial"] = "Get Started" if lang == "en" else "开始使用"
    page_labels["assistant"] = "AI Assistant" if lang == "en" else "AI 助手"
    page_labels["states"] = "Workspace States" if lang == "en" else "工作区状态"
    page_labels["settings"] = "Settings" if lang == "en" else "设置"

    # Resolve any pending navigation request (set by "Go to ..." buttons,
    # the sidebar, or the AI dock) into the active main page. This replaces
    # the previous JS-injection tab switching, which depended on Streamlit's
    # internal DOM and silently failed.
    _nav_request = st.session_state.pop('_scroll_to_tab', None)
    _nav_page_map = {
        'viz': 'quick_viz',
        'tutorial': 'tutorial',
        'cohort': 'cohort',
        'cross_db': 'cross_db',
        'crossdb': 'cross_db',
        'research_agent': 'research_agent',
        'assistant': 'assistant',
        'settings': 'settings',
        'export_progress': 'extract',
        'home_dict': 'tutorial',
    }
    if _nav_request == 'ai_assistant':
        if not _is_screenshot_mode():
            _open_embedded_ai_assistant(st.session_state)
    elif _nav_request in _nav_page_map:
        st.session_state['_active_main_page'] = _nav_page_map[_nav_request]

    # Tutorial is now the leftmost top tab again so it's discoverable from
    # the main pane (also still reachable via the sidebar "📚 Workflow Help"
    # button and ``_scroll_to_tab='tutorial'`` nav requests).
    # 'extract' is a special main page (the relocated data-extraction
    # workflow) reached via the sidebar pipeline; it is intentionally
    # NOT in the radio page_keys but is still a valid active page.
    _EXTRA_PAGES = {'extract', 'assistant', 'states', 'settings'}
    if st.session_state.get('_active_main_page') not in (set(mobile_page_keys) | _EXTRA_PAGES):
        st.session_state['_active_main_page'] = page_keys[0]

    # Do NOT bind ``st.radio`` directly to ``_active_main_page`` via
    # ``key=``. Streamlit serializes the widget state against the options
    # list, so any value that briefly falls outside ``mobile_page_keys`` (e.g.
    # an obsolete page name from a stale session) crashes the radio with
    # ``ValueError: <name> is not in iterable``. Using a separate widget
    # key + on_change-propagation keeps programmatic navigation safe.
    _current_active = st.session_state.get('_active_main_page', page_keys[0])
    _visible_active = _current_active if _current_active in mobile_page_keys else page_keys[0]
    # Force the widget to reflect the current (or fallback) visible page
    # each render — without this, Streamlit's widget-state persistence
    # would override programmatic navigation from sidebar buttons.
    st.session_state['_main_nav_widget'] = _visible_active

    def _propagate_main_nav() -> None:
        st.session_state['_active_main_page'] = st.session_state['_main_nav_widget']

    # The shell-A redesign moves primary navigation to the sidebar; the
    # radio below is kept (1) to preserve programmatic navigation paths
    # that wrote into ``_main_nav_widget``, and (2) as a fallback nav
    # when the sidebar is collapsed. It is visually hidden by the CSS
    # rule on ``[data-testid="stHorizontalBlock"][data-testid="main_nav_bar"]``
    # in shell_styles, but the markup below also hides the row when
    # the sidebar is on so it doesn't take vertical space.
    with st.container(key="main_nav_bar"):
        st.markdown(
            '<style>'
            '.stApp [class*="st-key-main_nav_bar"]{display:none !important;}'
            '@media (max-width: 900px){'
            '.stApp [class*="st-key-main_nav_bar"]{display:block !important;}'
            '}'
            '</style>',
            unsafe_allow_html=True,
        )
        st.radio(
            "Main navigation",
            options=mobile_page_keys,
            format_func=lambda key: page_labels.get(key, key),
            horizontal=True,
            label_visibility='collapsed',
            key='_main_nav_widget',
            on_change=_propagate_main_nav,
        )
    active_page = st.session_state.get('_active_main_page', page_keys[0])
    if active_page != "assistant":
        _clear_assistant_surfaces(st.session_state, clear_pending=True)

    _topbar_result = _handle_topbar_run_request(active_page, lang)
    if _topbar_result and st.session_state.get('_active_main_page') != active_page:
        active_page = st.session_state.get('_active_main_page', active_page)

    _render_post_export_guidance(
        active_page,
        lang,
        export_in_progress=export_in_progress,
    )

    if active_page == "extract":
        # Shell-A redesign: the data-extraction workflow (steps 1-4)
        # relocated from the sidebar into a main-area page, reached via
        # the sidebar pipeline "Open data extraction" button.
        if export_in_progress:
            st.markdown(
                f'<div class="compact-inline-notice info">'
                + ("⏳ Export in progress." if lang == 'en' else "⏳ 正在导出。")
                + '</div>',
                unsafe_allow_html=True,
            )
        else:
            render_extract_page(lang)

    elif active_page == "tutorial":
        # Shell-A redesign: Tutorial page now uses the hero + workflow
        # strip + starting-point cards layout from page-tutorial.jsx.
        from easyicu.webapp.pages_redesign import render_tutorial_redesign_page
        render_tutorial_redesign_page(lang)

    elif active_page == "states":
        from easyicu.webapp.pages_redesign import render_workspace_states_reference_page
        render_workspace_states_reference_page(lang)

    elif active_page == "assistant":
        from easyicu.webapp.llm_chat import render_ai_assistant_page
        render_ai_assistant_page(lang)

    elif active_page == "settings":
        from easyicu.webapp.pages_redesign import render_settings_redesign_page
        render_settings_redesign_page(lang)

    elif active_page == "quick_viz":
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
                            preview_n = min(int(_preview_n or LIGHTWEIGHT_DEMO_PATIENTS), LIGHTWEIGHT_DEMO_PATIENTS)
                            mock_data, preview_patient_ids = generate_lightweight_demo_data(
                                n_patients=preview_n,
                                hours=LIGHTWEIGHT_DEMO_HOURS,
                            )
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
                            _preview_concepts = _select_quick_preview_concepts(_sel_concepts, limit=5)
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

            # Shell-A redesign: render_quick_visualization_page already
            # ships its own ``render_page_header`` (now restyled to the
            # shell-A tokens via shell_styles.app-page-header rules), so
            # we don't add another header on top — that would just
            # duplicate the bilingual title. The topbar breadcrumb above
            # is the only chrome we contribute here.
            render_quick_visualization_page()

    elif active_page == "cohort":
        if export_in_progress:
            export_hold_msg = (
                "⏳ Export in progress. Cohort analysis views are temporarily paused until extraction finishes."
                if lang == 'en' else
                "⏳ 正在导出。提取完成前，暂时不渲染队列分析页面。"
            )
            st.markdown(f'<div class="compact-inline-notice info">{export_hold_msg}</div>', unsafe_allow_html=True)
        else:
            # Shell-A redesign: keep the new PageHeader + bilingual
            # breadcrumb chrome, but delegate each subtab body to the
            # original render functions so the *real* loaded cohort
            # data drives the charts. The synthetic SVG bodies remain
            # available behind st.session_state["_eu_shell_only"] for
            # design QA.
            from easyicu.webapp.cohort_redesign import render_cohort_redesign_page
            render_cohort_redesign_page(
                lang,
                group_fn=render_group_comparison_subtab,
                coverage_fn=render_data_coverage_audit_subtab,
                snapshot_fn=render_cohort_dashboard_subtab,
                sofa_fn=render_severity_reclassification_subtab,
            )

    elif active_page == "cross_db":
        # Promoted from a Cohort Statistics subtab to a top-level page
        # (2026-05 Phase B) because Cross-DB Benchmark structurally needs
        # ≥2 database roots — different from the single-DB inputs the
        # other cohort panels accept. Render with the standard cohort
        # page header so the chrome matches.
        if export_in_progress:
            xdb_hold_msg = (
                "⏳ Export in progress. Cross-DB benchmark is paused until extraction finishes."
                if lang == 'en' else
                "⏳ 正在导出。提取完成前，跨数据库对比面板暂停加载。"
            )
            st.markdown(f'<div class="compact-inline-notice info">{xdb_hold_msg}</div>', unsafe_allow_html=True)
        else:
            # Shell-A redesign: PageHeader chrome only; body delegates
            # to the real multi-DB feature distribution renderer so
            # actual ICU databases drive the comparison plots.
            from easyicu.webapp.cohort_redesign import render_cross_db_redesign_page
            render_cross_db_redesign_page(
                lang,
                multidb_fn=render_multidb_distribution_subtab,
            )

    elif active_page == "research_agent":
        # T1.7 — embed the ICU-aware research-agent page so reviewers can
        # run the full pipeline end-to-end (cohort → plan → code → run →
        # validators → bound manuscript) from the webapp without a
        # separate Streamlit process.
        if export_in_progress:
            ra_hold_msg = (
                "⏳ Export in progress. The research-agent page is paused until extraction finishes."
                if lang == 'en' else
                "⏳ 正在导出。提取完成前，研究智能体页面暂停加载。"
            )
            st.markdown(f'<div class="compact-inline-notice info">{ra_hold_msg}</div>', unsafe_allow_html=True)
        else:
            # Shell-A redesign: Research Agent has four stateful views.
            # Setup is the only page that configures and launches runs.
            # History is a local project picker. Workbench and Summary
            # render only a live/imported manifest, never a synthetic demo queue.
            st.session_state["_active_main_page"] = "research_agent"
            _default_ra_view = 'setup'
            _ra_view = st.session_state.get('_ra_view', _default_ra_view)
            _ra_run_context = _research_agent_active_run_context(st.session_state)
            _render_research_agent_reference_header(lang, view=_ra_view)

            with st.container(key="_eu_ra_tabs"):
                _seg_l, _seg_m, _seg_h, _seg_r, _seg_tail = st.columns([0.88, 1.18, 0.92, 0.98, 6.04])
                with _seg_l:
                    if st.button(
                        "Setup" if lang == 'en' else "配置",
                        icon=":material/tune:",
                        key="_eu_ra_view_setup", use_container_width=True,
                        type="primary" if _ra_view == 'setup' else "secondary",
                    ):
                        st.session_state['_active_main_page'] = 'research_agent'
                        st.session_state['_ra_view'] = 'setup'
                        st.rerun()
                with _seg_m:
                    if st.button(
                        "Workbench" if lang == 'en' else "工作台",
                        icon=":material/grid_view:",
                        key="_eu_ra_view_workbench", use_container_width=True,
                        type="primary" if _ra_view == 'workbench' else "secondary",
                    ):
                        st.session_state['_active_main_page'] = 'research_agent'
                        st.session_state['_ra_view'] = 'workbench'
                        st.rerun()
                with _seg_h:
                    if st.button(
                        "History" if lang == 'en' else "历史",
                        icon=":material/history:",
                        key="_eu_ra_view_history", use_container_width=True,
                        type="primary" if _ra_view == 'history' else "secondary",
                    ):
                        st.session_state['_active_main_page'] = 'research_agent'
                        st.session_state['_ra_view'] = 'history'
                        st.rerun()
                with _seg_r:
                    if st.button(
                        "Summary" if lang == 'en' else "总览",
                        icon=":material/shield:",
                        key="_eu_ra_view_summary", use_container_width=True,
                        type="primary" if _ra_view == 'summary' else "secondary",
                    ):
                        st.session_state['_active_main_page'] = 'research_agent'
                        st.session_state['_ra_view'] = 'summary'
                        st.rerun()
                with _seg_tail:
                    if _ra_run_context:
                        if st.button(
                            "Re-run" if lang == 'en' else "重新运行",
                            icon=":material/replay:",
                            key="_eu_ra_header_rerun",
                            type="primary",
                            use_container_width=False,
                            help=(
                                "Open Setup in checkpoint-resume mode for the active run."
                                if lang == 'en' else
                                "使用当前 run 打开配置页的 checkpoint 续跑模式。"
                            ),
                        ):
                            _prime_research_agent_header_rerun(st.session_state, _ra_run_context)
                            st.rerun()

            _ra_handoff_success = st.session_state.pop("_eu_ra_handoff_success_message", "")
            if _ra_handoff_success:
                st.success(str(_ra_handoff_success))

            if _ra_view == 'workbench':
                from easyicu.webapp.agent_workbench import render_agent_workbench
                render_agent_workbench(lang, show_header=False)
            elif _ra_view == 'history':
                from easyicu.webapp.research_agent import render_research_agent_history_page
                render_research_agent_history_page(lang, show_header=False)
            elif _ra_view == 'summary':
                from easyicu.webapp.agent_workbench import render_agent_output_summary
                render_agent_output_summary(lang, show_header=False)
            else:
                _render_research_agent_handoff(
                    "Loaded concepts" if lang == "en" else "已加载概念",
                    lang,
                    key_suffix="setup",
                )
                if st.session_state.get("_eu_ra_resource_focus") == "citation_info":
                    st.info(
                        "Citation info opens here because references and evidence links are handled by the Research Agent: "
                        "numeric claims are tied to evidence references before drafting."
                        if lang == "en" else
                        "引用信息在这里打开：Research Agent 负责把数值主张和证据引用绑定，审计通过后再进入起草。"
                    )
                try:
                    from easyicu.webapp.research_agent import (
                        render_research_agent_demo_page,
                        render_research_agent_page,
                    )
                    _draft_resume_pending = bool(
                        st.session_state.get("research_agent_resume_run_id")
                        or st.session_state.get("research_agent_force_manuscript")
                    )
                    _handoff_setup_pending = _research_agent_handoff_setup_ready(st.session_state)
                    if (
                        st.session_state.get('entry_mode') == 'demo'
                        and not _draft_resume_pending
                        and not _handoff_setup_pending
                    ):
                        render_research_agent_demo_page(show_header=False)
                    else:
                        render_research_agent_page(show_header=False)
                except Exception as _ra_exc:  # pragma: no cover - defensive
                    st.error(get_text("ra_page_load_failed").format(
                        error=f"{type(_ra_exc).__name__}: {_ra_exc}",
                    ))
                    st.caption(get_text("ra_optional_deps_hint"))

    _handle_sidebar_export_trigger(default_export_container)

    # Page navigation now happens via the segmented_control above. Only the
    # scroll-to-top request still needs a small script — and this one does not
    # depend on Streamlit's internal DOM structure.
    if st.session_state.pop('_scroll_to_top', False):
        js_scroll_to_top = """
        <script>
        (function() {
            function scrollEasyICUToTop() {
                var doc = window.parent.document;
                window.parent.scrollTo({top: 0, left: 0, behavior: 'auto'});
                doc.documentElement.scrollTop = 0;
                doc.body.scrollTop = 0;
                [
                    'section.main',
                    'section.stMain',
                    '[data-testid="stMain"]',
                    '[data-testid="stAppViewContainer"]'
                ].forEach(function(sel) {
                    var node = doc.querySelector(sel);
                    if (node) node.scrollTo({top: 0, left: 0, behavior: 'auto'});
                });
            }
            [0, 80, 240, 600, 1200].forEach(function(delay) {
                setTimeout(scrollEasyICUToTop, delay);
            });
        })();
        </script>
        """
        st.components.v1.html(
            js_scroll_to_top,
            height=0,
        )

    _render_figure_target_jump_script()

    # Shell-A carries status, navigation, and help affordances in the top bar
    # and sidebar. Suppress the legacy footer row so the design canvas starts
    # and ends cleanly.
    if False and not _is_screenshot_mode():
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
            # 2026-05 unified counts: route through cohort_feature_counts
            # so footer / handoff / gate guide / launcher always agree
            # on the loaded-feature number. Also show "loaded / dictionary
            # total" so users can see at a glance what fraction of the
            # canonical catalog is currently in memory.
            _ctr = cohort_feature_counts(st.session_state)
            n_concepts = _ctr['features']
            n_patients = _ctr['patients']
            n_total = _ctr['dictionary_total']
            ratio_suffix = f" / {n_total}" if n_total and n_concepts else ""
            st.markdown(
                f'<small class="app-footer-status">{data_status} | 📋 {n_concepts}{ratio_suffix} Concepts | 👥 {n_patients} {patients_label}</small>',
                unsafe_allow_html=True
            )

        with footer_cols[1]:
            if st.session_state.get('selected_patient'):
                patient_label = "Current Patient" if st.session_state.language == 'en' else "当前患者"
                st.markdown(
                    f'<small class="app-footer-status">🎯 {patient_label}: {st.session_state.selected_patient}</small>',
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
