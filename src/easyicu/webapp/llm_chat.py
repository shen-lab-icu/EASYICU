"""
EasyICU LLM Chat Assistant Module.

Provides an embedded conversational AI assistant that helps users
understand EasyICU features, interpret extraction results, and
answer ICU data analysis questions.

Supported providers:
        - HuggingFace (token required)
  - OpenAI, DeepSeek, Anthropic, OpenRouter, Together AI, Groq,
    SiliconFlow, and any custom OpenAI-compatible endpoint.

All API credentials are stored in session state only — never persisted.
"""

from __future__ import annotations

import contextlib
import html
import json
import os
import re
import threading
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import datetime
from pathlib import Path

import streamlit as st
from easyicu.webapp import copilot_engine as _copilot_engine
from easyicu.webapp.ai_optin import AIOptInError, enforce_external_llm_opt_in
# Phase-6 split: offline keyword routing + disease/sepsis option helpers now live
# in `copilot/routing.py`. Re-imported here so all existing references (and the
# `tests/webapp/test_copilot_keyword_route.py` `llm_chat.<fn>` accesses) keep
# working. routing.py imports the COPILOT_* constants lazily (inside the
# functions), so this top-level import does not create an import cycle.
from easyicu.webapp.copilot.routing import (  # noqa: F401  (re-exported)
    _copilot_branch_for_route_family,
    _copilot_create_route_completion,
    _copilot_extract_route_json,
    _copilot_full_disease_options,
    _copilot_keyword_route,
    _copilot_route_choice_actions,
    _copilot_route_family_label,
    _copilot_route_has_specific_goal,
    _copilot_route_uses_broad_question_type_choices,
    _copilot_sanitize_route_choice_id,
    _copilot_sepsis_mode_options,
    _copilot_specific_route_next_question,
)
# Phase-6 split: pure model-output sanitisers now live in `copilot/llm.py`.
# Re-imported here so all internal call sites and the `llm_chat.<fn>` test
# accesses keep working unchanged. The provider/client layer (`_get_client`,
# `_is_configured`, ...) intentionally stays in this module because the test
# suite swaps it wholesale via `monkeypatch.setattr(llm_chat, "st", ...)`; those
# functions must resolve `st` from this module's namespace, not llm.py's.
from easyicu.webapp.copilot.llm import (  # noqa: F401  (re-exported)
    _parse_verification_report,
    _strip_llm_reasoning,
)
# Phase-6 split: pure prompt-intent classifiers + lightweight text parsers now
# live in `copilot/intents.py`. Re-imported here so all call sites and
# `llm_chat.<fn>` test accesses keep working. intents.py lazy-imports the few
# `COPILOT_*` constants it needs from this module (inside the functions), so the
# top-level import below creates no cycle.
from easyicu.webapp.copilot.intents import (  # noqa: F401  (re-exported)
    _copilot_age_los_choice_requested,
    _copilot_api_setup_requested,
    _copilot_capability_overview_requested,
    _copilot_cohort_step_requested,
    _copilot_concepts_from_text,
    _copilot_confirm_suggested_concepts_requested,
    _copilot_current_reviewed_cohort_requested,
    _copilot_data_path_help_requested,
    _copilot_database_from_path,
    _copilot_disease_cohort_choice_requested,
    _copilot_endpoint_pinned,
    _copilot_explicit_local_command,
    _copilot_extract_data_path_from_text,
    _copilot_extract_patient_count,
    _copilot_extract_step_requested,
    _copilot_feature_pack_label,
    _copilot_feature_step_requested,
    _copilot_first_pass_goal_allowed,
    _copilot_free_study_goal_requested,
    _copilot_full_cohort_requested,
    _copilot_idea_topic,
    _copilot_is_loosen_filter_request,
    _copilot_is_strict_filter_request,
    _copilot_module_export_choice_requested,
    _copilot_module_pack_from_prompt,
    _copilot_modules_for_concepts,
    _copilot_next_step_help_requested,
    _copilot_no_disease_filter_requested,
    _copilot_normalize_database,
    _copilot_parse_idea_selection,
    _copilot_pick_branch,
    _copilot_prepared_data_choice_requested,
    _copilot_raw_files_choice_requested,
    _copilot_real_data_requested,
    _copilot_research_recommendation_requested,
    _copilot_review_step_requested,
    _copilot_route_next_question_asks_broad_type,
    _copilot_should_use_llm_route,
    _copilot_step_by_step_requested,
    _copilot_usage_help_requested,
    _infer_db_from_text,
)
# Phase-6 split (4th batch): study-state lifecycle + session-manifest persistence
# now live in `copilot/sessions.py`. Re-imported here so all call sites keep
# working. sessions.py lazy-imports the `COPILOT_*` constants and `_repo_root` it
# needs from this module (inside the functions), so no import cycle.
from easyicu.webapp.copilot.sessions import (  # noqa: F401  (re-exported)
    _copilot_is_legacy_default_question,
    _copilot_jsonable,
    _copilot_list_study_sessions,
    _copilot_normalize_legacy_study,
    _copilot_normalize_session_title,
    _copilot_sanitized_messages,
    _copilot_session_fallback_title,
    _copilot_session_now,
    _copilot_session_title_from_state,
    _copilot_study_manifest_path,
    _copilot_study_sessions_root,
    _default_copilot_study_state,
    _ensure_copilot_study_state,
    _invalidate_copilot_session_cache,
    _open_copilot_study_session,
    _read_copilot_study_session_manifest,
    _remember_copilot_guided_study_resume,
    _reset_copilot_study_state,
    _start_new_copilot_study_session,
    _touch_current_copilot_study_session,
    _write_copilot_study_session_manifest,
)
# Phase-6 split (5th batch): pure presentation helpers (labels, intros,
# stage/step rendering, workflow-snapshot HTML, relative-time) now live in
# `copilot/presentation.py`. Re-imported here so all call sites keep working. It
# lazy-imports the `COPILOT_*` constants + `_copilot_frame_question` from this
# module inside the functions, so no import cycle.
from easyicu.webapp.copilot.presentation import (  # noqa: F401  (re-exported)
    _copilot_cohort_is_empty,
    _copilot_cohort_label,
    _copilot_concept_label_list,
    _copilot_data_source_choice_label,
    _copilot_database_label,
    _copilot_disease_label,
    _copilot_rail_step_items,
    _copilot_relative_time,
    _copilot_stage_card_html,
    _copilot_stage_detail,
    _copilot_stage_status,
    _copilot_step_by_step_intro,
    _copilot_step_complete_for_navigation,
    _copilot_step_label,
    _copilot_step_unlocked_for_navigation,
    _copilot_uses_eligible_cohort,
    _copilot_visible_workflow_step_items,
    _copilot_workflow_button_label,
    _copilot_workflow_snapshot_html,
    _normalized_copilot_workflow_snapshot,
    _research_agent_source_label,
    _strip_module_label_icon,
    _workflow_status_step,
)
# Phase-6 split (7th batch): codebase/PubMed context + floating-copilot local
# answers now live in `copilot/context.py`. Re-imported here so all call sites
# keep working. It lazy-imports `_repo_root` (kept here for __file__-relative
# path stability) + the PROJECT_*_FILES constants, so no import cycle.
from easyicu.webapp.copilot.context import (  # noqa: F401  (re-exported)
    _append_floating_copilot_local_answer,
    _append_quick_links,
    _build_project_context,
    _build_pubmed_context,
    _collect_quick_links,
    _extract_identifiers,
    _extract_outline,
    _floating_copilot_answer_text,
    _get_instant_reply,
    _github_file_link,
    _inline_ai_context_payload,
    _is_code_question,
    _is_external_lookup_question,
    _load_project_index,
    _local_copilot_fallback_reply,
    _make_snippet,
    _normalize_prompt,
    _read_text,
    _repo_blob_base,
    _search_pubmed,
    _starter_prompts,
)
# Phase-6 split (8th batch): study-state -> research-agent handoff + workspace
# seeding now lives in `copilot/handoff.py`. Re-imported here so all call sites
# keep working. It lazy-imports COPILOT_BRANCH_CONFIG from this module and pulls
# leaf helpers from session_state/presentation/sessions, so no import cycle.
from easyicu.webapp.copilot.handoff import (  # noqa: F401  (re-exported)
    _apply_copilot_study_to_workspace,
    _clear_copilot_agent_cohort_handoff,
    _copilot_frame_question,
    _copilot_outcome_for_study,
    _copilot_patient_ids_for_handoff,
    _copilot_selected_concepts_for_study,
    _copilot_study_should_use_real_source,
    _copilot_template_for_study,
    _default_research_agent_idea_prompt,
    _latest_ai_handoff_question,
    _latest_module_export_dir_for_handoff,
    _prepare_research_agent_handoff_from_ai,
    _prepare_research_agent_idea_handoff_from_ai,
    _seed_copilot_crossdb_demo_workspace,
    _seed_crossdb_demo_workspace,
    _seed_demo_context_from_chat,
    _seed_research_agent_from_copilot_study,
    _seed_research_agent_real_source_from_copilot,
)
from easyicu.webapp.copilot.submit import (  # noqa: F401  (re-exported)
    _copilot_advance_step,
    _copilot_apply_entities,
    _copilot_confirm_classic_step2,
    _copilot_confirm_classic_step3,
    _copilot_default_cohort_filter,
    _copilot_invalidate_downstream_extraction,
    _copilot_parse_optional_int,
    _copilot_run_step,
    _copilot_set_module_export_path_in_chat,
    _copilot_set_raw_icu_path_in_chat,
    _copilot_set_real_data_path_in_chat,
    _copilot_submit_cohort_filter,
    _copilot_submit_data_source_path,
    _copilot_submit_extract,
    _copilot_submit_feature_modules,
    _copilot_submit_review,
    _copilot_validate_real_data_path,
)
from easyicu.webapp.copilot.idea import (  # noqa: F401  (re-exported)
    _copilot_handle_idea_selection,
    _copilot_idea_candidates_for_prompt,
    _copilot_idea_recommendation_reply,
    _copilot_reply,
    _copilot_store_idea_context,
    _is_idea_exploration_request,
    _suggest_ui_actions,
)
from easyicu.webapp.copilot.actions import (  # noqa: F401  (re-exported)
    _copilot_capability_overview_actions,
    _copilot_current_prompt_actions,
    _copilot_feature_module_action_keys,
    _copilot_feature_module_actions,
    _copilot_feature_module_prompt,
    _copilot_guided_choice_actions,
    _copilot_hint_prompts,
    _copilot_hint_prompts_for_state,
    _copilot_message_actions_for_current_step,
    _copilot_nonblocking_goal_fallback_actions,
    _copilot_primary_actions_for_state,
    _copilot_primary_prompts,
    _copilot_prompt_action,
    _copilot_started_in_chat,
    _copilot_study_actions,
    _normalized_copilot_message_actions,
)
from easyicu.webapp.concept_catalog import CONCEPT_GROUP_NAMES, CONCEPT_GROUPS_INTERNAL
from easyicu.webapp.components.constants import get_all_concepts
from easyicu.webapp.llm_config import (
    coerce_public_provider,
    ensure_llm_config_state,
    needs_api_key as _shared_needs_api_key,
    public_default_provider_key,
    public_provider_defaults,
    public_provider_keys,
)
from easyicu.webapp.session_state import clear_agent_continuation_state
from easyicu.webapp.ui_helpers import icon

_FEATURE_COUNT = len(get_all_concepts())

COPILOT_STUDY_STEPS: tuple[tuple[str, str], ...] = (
    ("question", "Research question"),
    ("data", "Data source"),
    ("cohort", "Cohort"),
    ("concepts", "Feature modules"),
    ("extract", "Extraction"),
    ("review", "Review"),
    ("analysis", "Analysis run"),
    ("draft", "Draft gate"),
)
COPILOT_STEP_INDEX = {step: idx for idx, (step, _label) in enumerate(COPILOT_STUDY_STEPS)}
# copilot_engine keeps an import-light copy of the step sequence; keep them in sync.
assert tuple(step for step, _ in COPILOT_STUDY_STEPS) == _copilot_engine.COPILOT_STEP_SEQUENCE, (
    "COPILOT_STUDY_STEPS drifted from copilot_engine.COPILOT_STEP_SEQUENCE"
)
COPILOT_DEFAULT_MODULES: list[str] = []
COPILOT_DATABASE_OPTIONS = ("miiv", "mimic", "eicu", "aumc", "hirid", "sic")
COPILOT_DATABASE_LABELS = {
    "miiv": ("MIMIC-IV", "MIMIC-IV"),
    "mimic": ("MIMIC-III", "MIMIC-III"),
    "eicu": ("eICU-CRD", "eICU-CRD"),
    "aumc": ("Amsterdam UMCdb", "Amsterdam UMCdb"),
    "hirid": ("HiRID", "HiRID"),
    "sic": ("SICdb", "SICdb"),
}
COPILOT_DISEASE_OPTIONS = {
    "none": ("Any / no disease filter", "不限 / 不做疾病筛选"),
    "sepsis": ("Sepsis-3 cohort", "脓毒症队列（Sepsis-3）"),
    "aki": ("AKI cohort", "AKI 队列"),
    "circ_failure": ("Circulatory failure", "循环衰竭"),
    "mech_vent": ("Mechanical ventilation", "机械通气"),
    "rrt": ("Renal replacement therapy", "肾脏替代治疗"),
}
COPILOT_FEATURE_MODULE_ACTION_KEYS = (
    "vitals",
    "chemistry",
    "blood_gas",
    "renal",
    "respiratory",
    "sofa2_score",
)


COPILOT_FEATURE_MODULE_PACKS = {
    key: {
        "label_en": _strip_module_label_icon(CONCEPT_GROUP_NAMES.get(key, (key, key))[0]),
        "label_zh": _strip_module_label_icon(CONCEPT_GROUP_NAMES.get(key, (key, key))[1]),
        "concepts": list(concepts),
    }
    for key, concepts in CONCEPT_GROUPS_INTERNAL.items()
}
COPILOT_CONCEPT_LABELS = {
    "age": "age",
    "death": "mortality",
    "hr": "heart rate",
    "lact": "lactate",
    "map": "MAP",
    "crea": "creatinine",
    "creat": "creatinine",
    "sofa2": "SOFA-2",
    "spo2": "SpO2",
    "temp": "temperature",
    "urine": "urine output",
    "vaso_ind": "vasopressor",
    "vaso": "vasopressor",
    "vent_ind": "ventilation",
    "rrt": "RRT",
}
COPILOT_STRICT_COHORT_FILTERS = ["sepsis-3", "age >= 80", "first 24h"]
# Free-text routing timeout. The default 1.5s is too tight for local/slow model
# endpoints (the route times out and the step stalls). Default raised to 6s and
# made configurable via EASYICU_COPILOT_ROUTE_TIMEOUT so self-hosted models work;
# fixed step buttons are deterministic and never hit this path.
try:
    COPILOT_ROUTE_TIMEOUT_SECONDS = max(1.0, float(os.environ.get("EASYICU_COPILOT_ROUTE_TIMEOUT", "6.0")))
except (TypeError, ValueError):
    COPILOT_ROUTE_TIMEOUT_SECONDS = 6.0
COPILOT_ROUTE_MAX_TOKENS = 360
COPILOT_SESSION_MESSAGE_SAVE_LIMIT = 80
COPILOT_RENDER_MESSAGE_LIMIT = 5
COPILOT_RECENT_SESSION_RENDER_LIMIT = 6
COPILOT_BRANCH_CONFIG = {
    "predict": {
        "chip": "Model ICU outcomes",
        "question_en": "Do first-24h bedside features predict a prespecified ICU outcome, and which added feature modules improve the model?",
        "question_zh": "前 24 小时床旁特征能否预测预设 ICU 结局，哪些新增特征模块能改善模型？",
        "review_target": "quick_viz",
        "selected_concepts": ["hr", "map", "temp", "spo2", "sofa2", "lact", "age", "death"],
        "why": {
            "question": "A vague aim becomes testable only after binding cohort, outcome, time window, and comparator.",
            "data": "Demo data lowers risk for the first pass; real data stays local and uses the same gates.",
            "cohort": "The cohort defines every downstream denominator, rate, and model row.",
            "concepts": "Only question-relevant modules are preselected, then coverage is audited before modelling.",
            "extract": "A frozen normalized frame keeps review panels and agent runs reproducible.",
            "review": "A human preview catches obvious data problems before spending an analysis run.",
            "analysis": "The run is evidence-bound; every table or figure must trace to an artifact.",
            "draft": "Drafting stays locked until checks pass and a human signs off.",
        },
    },
    "crossdb": {
        "chip": "Compare across ICU databases",
        "question_en": "Does a prespecified cohort and outcome signal replicate across ICU databases, and where do feature distributions diverge?",
        "question_zh": "预设队列和结局信号能否跨 ICU 数据库复现，哪些特征分布差异最大？",
        "review_target": "cross_db",
        "selected_concepts": ["hr", "map", "lact", "sofa2", "death", "age"],
        "why": {
            "question": "Replication needs one shared cohort definition applied consistently across databases.",
            "data": "Cross-database work needs at least two local sources; demo mode seeds all six safely.",
            "cohort": "Here the cohort includes both patients and the database set being compared.",
            "concepts": "Only concepts available across selected databases can support fair comparisons.",
            "extract": "Each source is normalized into the same concept names before comparison.",
            "review": "Availability and distribution review prevents over-reading database differences.",
            "analysis": "Per-database summaries and deltas are logged before any replication claim.",
            "draft": "Cross-database claims remain gated until each database artifact is traceable.",
        },
    },
    "quality": {
        "chip": "Audit data quality first",
        "question_en": "Before modelling, which ICU concepts are sparse, out-of-range, or trustworthy enough to analyze?",
        "question_zh": "建模前，哪些 ICU 概念稀疏、越界，哪些足够可信可以进入分析？",
        "review_target": "cohort",
        "selected_concepts": ["hr", "map", "temp", "spo2", "crea", "lact", "urine", "sofa2", "death"],
        "why": {
            "question": "A quality-first study starts with trustworthiness instead of effect estimates.",
            "data": "The same extraction gates apply; the first deliverable is a coverage and range audit.",
            "cohort": "The audit must use the same denominator as the later analysis would use.",
            "concepts": "Broad modules reveal sparse or unsafe variables before they bias a model.",
            "extract": "Consistent frames let coverage, ranges, and temporal density be measured uniformly.",
            "review": "The review table is the main deliverable for a QC-first branch.",
            "analysis": "Coverage, range, missingness, and density checks produce flags, not claims.",
            "draft": "Even a QC summary must trace every flag to logged evidence.",
        },
    },
}
COPILOT_ROUTE_ALLOWED_STEPS = {step for step, _label in COPILOT_STUDY_STEPS}
COPILOT_ROUTE_ALLOWED_FAMILIES = {
    "prediction",
    "association",
    "clustering",
    "trajectory",
    "quality_audit",
    "cross_database",
    "descriptive",
    "unknown",
}
COPILOT_ROUTE_FAMILY_LABELS = {
    "prediction": ("Outcome modelling", "结局建模"),
    "association": ("Association study", "关联分析"),
    "clustering": ("Patient clustering", "患者聚类"),
    "trajectory": ("Trajectory analysis", "轨迹分析"),
    "quality_audit": ("Data quality audit", "数据质量审计"),
    "cross_database": ("Cross-database comparison", "跨数据库比较"),
    "descriptive": ("Descriptive study", "描述性研究"),
    "unknown": ("Guided study", "引导式研究"),
}
COPILOT_IDEA_CANDIDATES: dict[str, list[dict[str, object]]] = {
    "sepsis": [
        {
            "id": "sepsis_early_outcome_model",
            "branch": "predict",
            "title_en": "Early deterioration model",
            "title_zh": "早期恶化风险建模",
            "question_en": "In adults with suspected or confirmed sepsis, do first-24h bedside features predict a prespecified ICU outcome, and which feature modules add value?",
            "question_zh": "在疑似或确认脓毒症成人 ICU 队列中，前 24 小时床旁特征能否预测预设 ICU 结局，哪些特征模块能带来增益？",
            "why_en": "Good first study because the endpoint, time window, cohort denominator, and model comparison can be made explicit.",
            "why_zh": "适合作为第一题，因为 endpoint、时间窗、队列分母和模型比较都可以明确绑定。",
            "concepts": ["age", "hr", "map", "lact", "sofa2", "death"],
        },
        {
            "id": "sepsis_vasopressor_renal_outcome",
            "branch": "predict",
            "title_en": "Hemodynamic support and kidney outcome",
            "title_zh": "血流动力学支持与肾脏结局",
            "question_en": "Among sepsis ICU stays, are early vasopressor patterns associated with subsequent RRT or mortality after accounting for baseline severity?",
            "question_zh": "在脓毒症 ICU stay 中，早期升压药模式在考虑基线严重程度后，是否与后续 RRT 或死亡相关？",
            "why_en": "Useful if you want a discovery-style analysis, but it needs careful confounding language and feasibility checks.",
            "why_zh": "适合探索发现型分析，但需要非常谨慎地区分相关性、混杂和可行性检查。",
            "concepts": ["age", "map", "vaso_ind", "crea", "urine", "rrt", "death"],
        },
        {
            "id": "sepsis_definition_quality",
            "branch": "quality",
            "title_en": "Sepsis definition and data quality audit",
            "title_zh": "脓毒症定义与数据质量审计",
            "question_en": "Before modelling sepsis outcomes, which Sepsis-3, SOFA-2, lactate, vital-sign, and outcome fields are complete and comparable enough to support analysis?",
            "question_zh": "在建模脓毒症结局前，Sepsis-3、SOFA-2、乳酸、生命体征和结局字段是否足够完整、可比并适合分析？",
            "why_en": "Best when the main risk is data reliability rather than model choice.",
            "why_zh": "当主要风险是数据可靠性而不是模型选择时，这个方向更稳。",
            "concepts": ["age", "hr", "map", "temp", "spo2", "lact", "sofa2", "death"],
        },
    ],
    "general": [
        {
            "id": "icu_outcome_model",
            "branch": "predict",
            "title_en": "Outcome model with explicit endpoint",
            "title_zh": "明确 endpoint 的 ICU 结局建模",
            "question_en": "Do first-24h bedside features predict a prespecified ICU outcome in the eligible cohort, and which added feature modules improve the model?",
            "question_zh": "前 24 小时床旁特征能否在合格 ICU 队列中预测预设 ICU 结局，哪些新增特征模块能改善模型？",
            "why_en": "A solid starting point because it forces endpoint, cohort, window, and feature modules to be explicit.",
            "why_zh": "这是稳妥入口，因为它会逼着我们明确 endpoint、队列、时间窗和特征模块。",
            "concepts": ["age", "hr", "map", "sofa2", "lact", "death"],
        },
        {
            "id": "icu_treatment_trajectory",
            "branch": "predict",
            "title_en": "Treatment trajectory and outcome",
            "title_zh": "治疗轨迹与结局",
            "question_en": "Are early ICU treatment trajectories associated with a prespecified outcome after severity and data availability checks?",
            "question_zh": "在完成严重程度和数据可用性检查后，早期 ICU 治疗轨迹是否与预设结局相关？",
            "why_en": "Useful for discovery, but it should stay evidence-bound and avoid causal overclaiming.",
            "why_zh": "适合探索发现，但必须保持证据绑定，避免直接做因果化表述。",
            "concepts": ["age", "map", "vaso_ind", "vent_ind", "crea", "death"],
        },
        {
            "id": "icu_quality_first",
            "branch": "quality",
            "title_en": "Quality-first feasibility audit",
            "title_zh": "质量优先的可行性审计",
            "question_en": "Which ICU concepts are complete, comparable, and safe enough to support a later modelling or association study?",
            "question_zh": "哪些 ICU 概念足够完整、可比且安全，能够支撑后续建模或关联分析？",
            "why_en": "Best when you need a trustworthy starting map before committing to a hypothesis.",
            "why_zh": "当你还没确定假设前，先画出可信变量地图最稳。",
            "concepts": ["age", "hr", "map", "temp", "spo2", "lact", "sofa2", "death"],
        },
    ],
}
# ---------------------------------------------------------------------------
# System prompt — enriched with EasyICU documentation & concept catalogue
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are an intelligent assistant embedded in **EasyICU**, an interactive platform \
for clinical data extraction and exploration across multiple public ICU databases.

## Platform Overview
EasyICU is a Python toolkit (v1.0) that provides:
- Unified access to **6 public ICU databases**: MIMIC-IV (miiv), MIMIC-III (mimic), \
eICU-CRD (eicu), AmsterdamUMCdb (aumc), HiRID (hirid), SICdb (sic)
- Automated extraction of **{_FEATURE_COUNT} standardized clinical concepts** across 19 feature modules
- A no-code Streamlit web interface for cohort construction, feature selection, \
quality review, and cohort comparison
- High-performance computing optimised for 16 GB RAM machines
- Export in Parquet, CSV, and XLSX formats; all parameters exportable as JSON for reproducibility

## Workflow (4 Steps)
1. **Data Source** — choose database & path (or Demo mode with simulated data)
2. **Cohort Selection** — filter by age, sex, ICU LOS, mortality, disease cohort (e.g. Sepsis-3, AKI, RRT, mechanical ventilation), and ICD keywords where supported
3. **Select export concepts** — pick from 19 modules ({_FEATURE_COUNT} concepts); supports SOFA-1, SOFA-2, \
Sepsis-3, KDIGO-AKI, circulatory failure, etc.
4. **Export Data** — batch export to disk; streaming architecture, subprocess memory isolation

## Main Web Areas
- **Tutorial** — workflow guide, usage examples, and the in-app data dictionary
- **Quick Visualization** — load extracted data and inspect trends, missingness, and distributions
- **Cohort Analysis** — compare groups and review downstream cohort summaries
- **Research Copilot** — chat-first help for navigation, feature planning, troubleshooting, evidence lookup, and handoff into the auditable Research Agent

## Feature Modules & Concepts (19 modules, {_FEATURE_COUNT} concepts)
- **Vital Signs** (8): hr, map, sbp, dbp, pulse_pressure, temp, spo2, resp
- **Respiratory** (14): pafi, safi, fio2, supp_o2, vent_ind, vent_start, vent_end, \
o2sat, sao2, mech_vent, ett_gcs, ecmo, ecmo_indication, adv_resp
- **Ventilator Parameters** (12): peep, tidal_vol, tidal_vol_set, pip, plateau_pres, \
mean_airway_pres, minute_vol, vent_rate, etco2, compliance, driving_pres, ps
- **Blood Gas** (9): be, cai, hbco, lact, methb, pco2, ph, po2, tco2
- **Chemistry** (22): alb, alp, alt, ast, anion_gap, bicar, bili, bili_dir, bun, ca, ck, ckmb, \
cl, crea, crp, glu, k, mg, na, phos, tnt, tri
- **Hematology** (20): bnd, basos, eos, esr, fgn, hba1c, hct, hgb, inr_pt, lymph, \
mch, mchc, mcv, neut, plt, pt, ptt, rbc, rdw, wbc
- **Vasopressors** (17): norepi_rate/dur/equiv/60, epi_rate/dur/60, \
dopa_rate/dur/60, dobu_rate/dur/60, adh_rate, phn_rate, vaso_ind, other_vaso
- **Medications** (15): abx, cort, dex, ins, amiodarone, dexmedetomidine, fentanyl, \
furosemide, heparin, mannitol, midazolam, milrinone, morphine, propofol, rocuronium
- **Renal / KDIGO** (17): urine, urine24, uo_6h/12h/24h, rrt, rrt_criteria, \
aki, aki_stage, aki_stage_creat/uo/rrt, creat_low_past_48hr/7day, uo_rt_6hr/12hr/24hr
- **Neurological** (11): avpu, egcs, gcs, mgcs, rass, tgcs, vgcs, sedated_gcs, \
motor_response, delirium_positive, delirium_tx
- **Circulatory** (3): mech_circ_support, circ_failure, circ_event
- **Demographics** (6): age, bmi, height, sex, weight, adm
- **Other Scores** (4): qsofa, sirs, mews, news
- **Outcome** (3): death, los_icu, los_hosp
- **SOFA-2 Score** (7): sofa2, sofa2_resp/coag/liver/cardio/cns/renal
- **SOFA-1 Score** (7): sofa, sofa_resp/coag/liver/cardio/cns/renal
- **Sepsis-3** (2): sep3_sofa1, sep3_sofa2
- **Sepsis Shared** (3): susp_inf, infection_icd, samp

## Python API Examples
```python
from easyicu import load_concepts, load_sofa, load_sofa2
# Load specific concepts
df = load_concepts(['hr','sbp','temp'], database='miiv', data_path='/data/miiv')
# Load SOFA scores
sofa = load_sofa(database='eicu', data_path='/data/eicu')
```

## Key Clinical Scoring
- **SOFA-2**: 2025 revised criteria implemented in EasyICU (6 sub-systems)
- **SOFA-1**: Original 1996 SOFA score
- **Sepsis-3**: Suspected infection + SOFA ≥ 2
- **KDIGO-AKI**: Stages 0-3 based on creatinine rise, urine output, and RRT
- **qSOFA**: Quick bedside screen (RR≥22, SBP≤100, altered mentation)

## Database Patient Counts (approx.)
| DB | Patients | ID column |
|----|----------|-----------|
| MIMIC-IV (miiv) | 94,458 | stay_id |
| MIMIC-III (mimic) | 61,532 | icustay_id |
| eICU-CRD (eicu) | 200,859 | patientunitstayid |
| AmsterdamUMCdb (aumc) | 23,106 | admissionid |
| HiRID (hirid) | 33,905 | patientid |
| SICdb (sic) | 27,386 | CaseID |

## Response Rules
- Respond in the same language the user is using.
- Be concise and practical. Start with the direct answer, then give short bullets only if useful.
- If the user describes a study goal or clinical task, answer in a task-first way:
  1. restate what EasyICU can support for that goal,
  2. suggest the recommended cohort definition,
  3. suggest which modules / concepts to extract,
  4. mention the key web steps to follow.
- Prioritize helping users use the EasyICU web interface and workflows. Code-level explanations are secondary unless the user explicitly asks for implementation details.
- When a user asks where something is in the web app, answer with the relevant page, step, or in-app action first. Do not default to repo files.
- When the user asks about EasyICU implementation, prefer the exact EasyICU concept names and outputs over generic medical summaries.
- Prefer suggesting concrete research workflows such as early warning, trajectory modelling, cohort construction, outcome analysis, sensitivity analysis, and cross-database feature planning.
- You will also receive a local EasyICU code snapshot at runtime. Use that code context when answering implementation or file-level questions.
- When pointing users to project files or docs, prefer clickable Markdown links.
- Do not invent version years, guideline dates, concept names, or unsupported claims. If uncertain, say what you can confirm from EasyICU and stop there.
- For Sepsis questions, anchor the answer to EasyICU outputs: `sep3_sofa1`, `sep3_sofa2`, `susp_inf`, `infection_icd`, `samp`, `sofa`, `sofa2`.
- Avoid padded closers like "有什么我可以帮你的吗？" unless the user asks for more detail."""


PROJECT_CONTEXT_FILES = [
    "src/easyicu/webapp/app.py",
    "src/easyicu/webapp/llm_chat.py",
    "src/easyicu/api.py",
    "src/easyicu/load_concepts.py",
    "src/easyicu/concept.py",
    "src/easyicu/concept_callbacks.py",
    "src/easyicu/sofa2.py",
    "src/easyicu/sepsis.py",
    "src/easyicu/scores.py",
    "src/easyicu/resources.py",
    "README.md",
]

PROJECT_LINK_FILES = [
    "README.md",
    "HOSTED_LLM.md",
    "src/easyicu/data/concept-dict.json",
    "src/easyicu/data/sofa2-dict.json",
    "src/easyicu/webapp/app.py",
    "src/easyicu/webapp/llm_chat.py",
    "src/easyicu/hosted_llm_server.py",
    "src/easyicu/api.py",
    "src/easyicu/load_concepts.py",
    "src/easyicu/concept.py",
    "src/easyicu/concept_callbacks.py",
    "src/easyicu/sofa2.py",
    "src/easyicu/sepsis.py",
]

ALL_PRESET_GROUP_KEYS = [
    "sofa2_score",
    "sofa1_score",
    "sepsis3_sofa2",
    "sepsis3_sofa1",
    "sepsis_shared",
    "vitals",
    "respiratory",
    "ventilator",
    "blood_gas",
    "chemistry",
    "hematology",
    "vasopressors",
    "medications",
    "renal",
    "neurological",
    "circulatory",
    "demographics",
    "other_scores",
    "outcome",
]

SEPSIS_PRESET_CONCEPTS = [
    "sep3_sofa2",
    "sep3_sofa1",
    "susp_inf",
    "infection_icd",
    "samp",
    "sofa2",
    "sofa2_resp",
    "sofa2_coag",
    "sofa2_liver",
    "sofa2_cardio",
    "sofa2_cns",
    "sofa2_renal",
    "sofa",
    "sofa_resp",
    "sofa_coag",
    "sofa_liver",
    "sofa_cardio",
    "sofa_cns",
    "sofa_renal",
    "abx",
    "cort",
    "lact",
    "hr",
    "map",
    "sbp",
    "temp",
    "resp",
    "spo2",
    "fio2",
    "pafi",
    "wbc",
    "crp",
    "crea",
    "bili",
    "urine",
    "urine24",
    "rrt",
    "death",
    "los_icu",
]


def _build_workflow_status_context(lang: str) -> str:
    """Summarize the current EasyICU web workflow state for the assistant."""
    entry_mode = st.session_state.get("entry_mode", "none")
    database = st.session_state.get("database", "miiv")
    data_path = (st.session_state.get("data_path") or "").strip()
    path_set = bool(data_path)
    path_validated = bool(st.session_state.get("path_validated", False))
    last_validation = st.session_state.get("last_validation", {}) or {}
    last_validated_path = st.session_state.get("last_validated_path", "")
    convert_needed = bool(
        path_set
        and data_path == last_validated_path
        and last_validation.get("can_convert")
        and not path_validated
    )
    step2_confirmed = bool(st.session_state.get("step2_confirmed", False))
    step3_confirmed = bool(st.session_state.get("step3_confirmed", False))
    selected_count = len(st.session_state.get("selected_concepts", []))
    loaded_count = len(st.session_state.get("loaded_concepts", {}))
    pending_preset = st.session_state.get("_assistant_pending_feature_preset")
    pending_db = pending_preset.get("database") if isinstance(pending_preset, dict) else None

    if lang == "en":
        lines = [
            "Current EasyICU workflow state:",
            f"- entry_mode: {entry_mode}",
            f"- database: {database}",
            f"- data_path_set: {'yes' if path_set else 'no'}",
            f"- data_path_validated: {'yes' if path_validated else 'no'}",
            f"- convert_or_setup_needed: {'yes' if convert_needed else 'no'}",
            f"- step2_confirmed: {'yes' if step2_confirmed else 'no'}",
            f"- step3_confirmed: {'yes' if step3_confirmed else 'no'}",
            f"- selected_concepts_count: {selected_count}",
            f"- loaded_concepts_count: {loaded_count}",
        ]
        if pending_db:
            lines.append(f"- pending_ai_preset_for_database: {pending_db}")
        lines.extend([
            "",
            "Guidance rules for this session:",
            "- For web workflow questions, first say what mode/page/step the user should be in.",
            "- If the user is not in Real Data mode and asks about extracting from public ICU databases, explicitly tell them to switch to Real Data mode first.",
            "- If data_path is not set, the next step is to fill the data path.",
            "- If data_path is set but not validated, the next step is Validate Data Path or Convert & Setup.",
            "- If conversion/setup is needed, warn that it may take time but normally only needs to be done once per dataset.",
            "- If step2 is not confirmed yet, tell the user to finish cohort selection before feature selection.",
            "- Only discuss code locations if the user explicitly asks about implementation.",
        ])
        return "\n".join(lines)

    lines = [
        "当前 EasyICU 工作流状态：",
        f"- entry_mode: {entry_mode}",
        f"- database: {database}",
        f"- 是否已填写数据路径: {'是' if path_set else '否'}",
        f"- 数据路径是否已验证: {'是' if path_validated else '否'}",
        f"- 是否需要转换/设置: {'是' if convert_needed else '否'}",
        f"- 步骤2是否已确认: {'是' if step2_confirmed else '否'}",
        f"- 步骤3是否已确认: {'是' if step3_confirmed else '否'}",
        f"- 当前已选特征数: {selected_count}",
        f"- 当前已加载概念数: {loaded_count}",
    ]
    if pending_db:
        lines.append(f"- 已挂起的 AI 预设数据库: {pending_db}")
    lines.extend([
        "",
        "本轮回答规则：",
        "- 对 Web 使用问题，先说明用户应该处于哪个模式、页面和步骤。",
        "- 如果用户问的是公共 ICU 数据库提取，而当前不是 Real Data 模式，要明确先切换到真实数据模式。",
        "- 如果还没填写数据路径，下一步就是先填写数据路径。",
        "- 如果数据路径已填写但还没验证，下一步就是点击验证数据路径或转换并设置。",
        "- 如果需要转换/设置，要提醒用户这一步可能较耗时，但通常同一份数据只需要做一次。",
        "- 如果步骤2还没确认，不要直接让用户去选特征；先让他完成队列筛选确认。",
        "- 只有当用户明确问实现细节时，才讨论代码位置。",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_chat_state():
    """Ensure all chat-related session keys exist."""
    ensure_llm_config_state()
    defaults = {
        "llm_enabled": False,
        "llm_provider": public_default_provider_key(),
        "llm_api_key": "",
        "llm_model": "",
        "llm_base_url": "",
        "llm_configured": False,
        "llm_messages": [],
        "llm_last_tool_events": [],
        "llm_last_verification": None,
        "_floating_ai_open": False,
        "_ai_pending_question": None,
        # Background response tracking
        "_ai_bg_responding": False,        # True while LLM is generating in background
        "_ai_bg_response_ready": False,    # True when background response finished
        "_ai_bg_unread_count": 0,          # Number of unread responses
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sync_llm_toggle_before_render() -> None:
    """Synchronize the sidebar toggle before its widget is instantiated."""
    if "_llm_toggle" not in st.session_state or st.session_state.pop("_llm_toggle_sync_pending", False):
        st.session_state["_llm_toggle"] = bool(st.session_state.get("llm_enabled", False))


def _close_floating_ai_panel(*, disable_assistant: bool = False) -> None:
    """Close the floating panel, optionally turning off the sidebar toggle too."""
    st.session_state["_floating_ai_open"] = False
    st.session_state["_ai_pending_question"] = None
    if disable_assistant:
        st.session_state.llm_enabled = False
        st.session_state["_llm_toggle_sync_pending"] = True


def _repo_root() -> Path:
    """Return the project root based on this module location."""
    return Path(__file__).resolve().parents[3]


_COPILOT_GENERIC_SESSION_TITLES = {
    "new study / workspace", "新建研究 / 工作目录",
    "new study", "新研究",
    "untitled study", "未命名研究",
    "guided study", "当前研究目标", "new study / workspace ",
}


def _copilot_cohort_step_intro(
    study: MutableMapping[str, object],
    lang: str,
    state: Mapping[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """Enter the cohort step directly, mirroring the classic extraction flow."""
    study["step"] = "cohort"
    study.setdefault("branch", "predict")
    study.setdefault("data_mode", "real")
    study["cohort_configured"] = False
    study.pop("cohort_substep", None)
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    if lang == "en":
        body = (
            "Cohort step opened. I will not choose an outcome or invent a research question here.\n\n"
            "This mirrors Classic Data Extraction: **Data source -> Cohort -> Feature modules -> Export/Review**. "
            "For the cohort step, choose one option below:\n\n"
            "- **Eligible cohort**: keep all analysis-eligible ICU stays for the current data source.\n"
            "- **Disease / diagnosis**: add a clinical filter such as Sepsis-3 or AKI/RRT.\n"
            "- **Age / ICU LOS**: constrain demographics or minimum ICU stay length.\n"
            "- **Current reviewed cohort**: reuse a cohort already loaded in Patient Review.\n\n"
            "If the real data source is not bound yet, I will keep the cohort as a pending definition and ask for the data path later in this same chat."
        )
    else:
        body = (
            "已进入**队列步骤**。我不会在这里替你选择研究结局，也不会编一个研究问题。\n\n"
            "这一步对应经典 Data Extraction 的顺序：**数据源 -> 队列 -> 特征模块 -> 导出/审阅**。"
            "现在只配置队列，请在下面选一种：\n\n"
            "- **全部合格队列**：保留当前数据源里满足分析条件的 ICU stay。\n"
            "- **按疾病/诊断**：加入 Sepsis-3、AKI/RRT 等临床过滤。\n"
            "- **按年龄/ICU LOS**：限制年龄或最短 ICU 住院时长。\n"
            "- **使用当前审阅队列**：复用 Patient Review 已加载的队列。\n\n"
            "如果真实数据源还没绑定，我会先把这个队列定义标记为待数据源确认，后面仍然在这个聊天里继续问路径，不跳到经典页面。"
        )
    return body, _copilot_guided_choice_actions(study, lang)


def _copilot_feature_step_intro(
    study: MutableMapping[str, object],
    lang: str,
    state: Mapping[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """Enter classic Step 3 module selection without making a default selection."""
    _ = state
    study["step"] = "concepts"
    study.setdefault("branch", "predict")
    study.setdefault("data_mode", "real")
    study["concepts_configured"] = False
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    if lang == "en":
        body = (
            "Feature-module step opened. This mirrors Classic Data Extraction Step 3.\n\n"
            "Pick real EasyICU modules below, or use the embedded checklist in this message. "
            "I will only write `selected_concepts` after you save the module selection."
        )
    else:
        body = (
            "已进入**特征模块步骤**。这里对应经典 Data Extraction Step 3。\n\n"
            "下面显示的是真实 EasyICU 模块；你可以点快捷按钮，也可以在这条消息里的清单中勾选。"
            "只有保存模块后，我才会写入 `selected_concepts`。"
        )
    return body, _copilot_guided_choice_actions(study, lang)


# `_copilot_full_disease_options` and `_copilot_sepsis_mode_options` moved to
# `copilot/routing.py` (Phase-6 split) and are re-imported at the top of this
# module, so references below resolve unchanged.


def _copilot_set_data_source_choice(
    state: MutableMapping[str, object],
    choice: str,
) -> MutableMapping[str, object]:
    """Open an in-page data-source form for the selected source type."""
    clean_choice = choice if choice in {"prepared_path", "module_export", "raw_files"} else "prepared_path"
    state["entry_mode"] = "real"
    state["use_mock_data"] = False
    if state.get("database") == "mock":
        state["database"] = "miiv"
    state["database"] = _copilot_normalize_database(state.get("database") or "miiv")
    state["_copilot_data_source_choice"] = clean_choice
    study = _ensure_copilot_study_state(state)
    study["data_mode"] = "real"
    study["step"] = "data"
    study["data_source_choice"] = clean_choice
    study["data_source_status"] = "awaiting_path"
    study["database"] = state["database"]
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return study


def _copilot_feature_selection_key(panel_key: str) -> str:
    return f"{panel_key}_feature_module_selection"


def _copilot_order_feature_modules(module_keys: Iterable[str]) -> list[str]:
    selected = {str(key) for key in module_keys if str(key) in COPILOT_FEATURE_MODULE_PACKS}
    return [key for key in COPILOT_FEATURE_MODULE_PACKS if key in selected]


def _copilot_feature_inline_selected_keys(
    state: MutableMapping[str, object],
    *,
    panel_key: str,
    default_keys: Iterable[str],
) -> list[str]:
    """Return the in-chat Step 3 module toggle state for this rendered message."""
    selection_key = _copilot_feature_selection_key(panel_key)
    existing = state.get(selection_key)
    if not isinstance(existing, list):
        selected = _copilot_order_feature_modules(default_keys)
        state[selection_key] = selected
        return selected
    selected = _copilot_order_feature_modules(str(item) for item in existing)
    state[selection_key] = selected
    return selected


def _copilot_toggle_feature_inline_module(
    state: MutableMapping[str, object],
    *,
    panel_key: str,
    module_key: str,
    default_keys: Iterable[str],
) -> list[str]:
    """Toggle one Step 3 module and persist visible button state before saving."""
    selected = _copilot_feature_inline_selected_keys(
        state,
        panel_key=panel_key,
        default_keys=default_keys,
    )
    if module_key in selected:
        next_selected = [key for key in selected if key != module_key]
    elif module_key in COPILOT_FEATURE_MODULE_PACKS:
        next_selected = selected + [module_key]
    else:
        next_selected = selected
    ordered = _copilot_order_feature_modules(next_selected)
    state[_copilot_feature_selection_key(panel_key)] = ordered
    return ordered


def _copilot_apply_strict_no_data_filter(study: MutableMapping[str, object]) -> None:
    study["step"] = "cohort"
    study["cohort_phase"] = "empty"
    study["cohort_filters"] = COPILOT_STRICT_COHORT_FILTERS[:]
    study["cohort_configured"] = True
    study["cohort_empty_reason"] = "Sepsis-3 + age >= 80 is empty in this demo set/export."
    study["draft_signed"] = False
    study["last_update"] = datetime.now().isoformat(timespec="seconds")


def _copilot_loosen_filters(study: MutableMapping[str, object]) -> None:
    study["step"] = "cohort"
    study["cohort_phase"] = "ready"
    study["cohort_filters"] = []
    study["cohort_configured"] = True
    study.pop("cohort_empty_reason", None)
    study["draft_signed"] = False
    study["last_update"] = datetime.now().isoformat(timespec="seconds")


def _copilot_api_connection_snapshot(state: Mapping[str, object], lang: str) -> dict[str, str | bool]:
    provider = coerce_public_provider(str(state.get("llm_provider") or public_default_provider_key()))
    provider_label, default_url, default_model, needs_key, _desc_en, _desc_zh = public_provider_defaults(provider)
    base_url = str(state.get("llm_base_url") or default_url or "").strip()
    model = str(state.get("llm_model") or default_model or "").strip()
    api_key_present = bool(str(state.get("llm_api_key") or "").strip())
    configured = bool(base_url and (api_key_present or not needs_key))
    enabled = bool(state.get("llm_enabled"))
    is_en = lang == "en"
    if configured and enabled:
        status = "connected"
        detail = (
            f"{provider_label} · {model or 'model selected'}"
            if is_en else
            f"{provider_label} · {model or '已选择模型'}"
        )
    elif configured:
        status = "configured"
        detail = (
            f"{provider_label} is configured; enable shared outbound calls before model use."
            if is_en else
            f"{provider_label} 已配置；模型调用前需要开启共享出站调用。"
        )
    else:
        status = "local"
        detail = (
            "Local workflow only; connect an OpenAI-compatible API for open-ended model replies."
            if is_en else
            "当前为本地工作流；接入 OpenAI 兼容 API 后可使用开放式模型回复。"
        )
    return {
        "provider": provider,
        "provider_label": provider_label,
        "model": model,
        "base_url": base_url,
        "configured": configured,
        "enabled": enabled,
        "status": status,
        "detail": detail,
    }


def _copilot_current_step_detail(study: Mapping[str, object], lang: str) -> tuple[str, str]:
    is_en = lang == "en"
    step = str(study.get("step") or "question")
    cohort_label = _copilot_cohort_label(study, lang)
    module_count = len(study.get("modules") or COPILOT_DEFAULT_MODULES)
    if step == "data":
        return (
            ("Data source", "Choose the source type below, then save the path in Copilot without leaving this page.")
            if is_en else
            ("数据源", "先在下方选择数据来源类型，再直接在 Copilot 中保存路径，不离开当前页。")
        )
    if step == "cohort":
        if not bool(study.get("cohort_configured")) and not _copilot_uses_eligible_cohort(study):
            return (
                ("Cohort", "Choose the cohort scope; I will keep it pending until you confirm an option.")
                if is_en else
                ("队列", "选择队列范围；确认选项前我会保持未选择状态。")
            )
        if _copilot_cohort_is_empty(dict(study)):
            return (
                ("Cohort", "No patients match the strict filters. Loosen one constraint to continue.")
                if is_en else
                ("队列", "严格过滤条件没有匹配患者。放宽一个条件后继续。")
            )
        if _copilot_uses_eligible_cohort(study):
            return (
                ("Cohort scope · eligible real-data cohort", "This is a scope, not a materialized row count. Bind a prepared data path or module export before Agent analysis.")
                if is_en else
                ("队列范围 · 真实数据合格队列", "这是队列范围，不是已经生成的行数；进入 Agent 分析前需要绑定 prepared 数据路径或模块导出。")
            )
        return (
            (f"Cohort · {cohort_label}", "This cohort definition drives review, analysis, and Agent setup.")
            if is_en else
            (f"队列 · {cohort_label}", "这个队列定义会同步到审阅、分析和 Agent 配置。")
        )
    if step == "concepts":
        if not bool(study.get("concepts_configured")) and not list(study.get("selected_concepts") or []):
            return (
                ("Feature modules", "Choose real EasyICU modules from the current chat controls.")
                if is_en else
                ("特征模块", "在当前聊天控件里选择真实 EasyICU 模块。")
            )
        return (
            (f"Feature modules · {module_count}", "Feature set is mapped; prepare extraction and Agent handoff from this Copilot flow.")
            if is_en else
            (f"特征模块 · {module_count}", "特征集已映射；继续在 Copilot 中准备提取和 Agent 交接。")
        )
    if step == "extract":
        return (
            ("Extraction", "Prepare the extraction plan in Copilot; Classic workspace is only for detailed manual review.")
            if is_en else
            ("提取", "在 Copilot 中准备提取计划；经典工作区仅作为手动细节审阅入口。")
        )
    if step == "review":
        return (
            ("Review", "Patient Review / Cross-DB review is ready from this chat context.")
            if is_en else
            ("审阅", "患者审阅 / 跨库审阅已由当前聊天上下文准备好。")
        )
    if step == "analysis":
        return (
            ("Research Agent run", "Question, cohort, concepts, and source context are ready for Agent setup.")
            if is_en else
            ("Research Agent 运行", "问题、队列、概念和数据源上下文已可交给 Agent 配置。")
        )
    if step == "draft":
        return (
            ("Evidence gate", "Drafting stays locked until checks pass and you sign off.")
            if is_en else
            ("证据闸门", "检查通过并人工确认前，草稿保持锁定。")
        )
    return (
        ("Research question", "Frame the question in one sentence; I will turn it into workflow state.")
        if is_en else
        ("研究问题", "用一句话描述问题；我会把它转成工作流状态。")
    )


def _copilot_real_source_ready(state: Mapping[str, object]) -> bool:
    loaded_concepts = state.get("loaded_concepts")
    if isinstance(loaded_concepts, Mapping) and bool(loaded_concepts):
        return True
    if str(state.get("last_export_dir") or state.get("export_path") or "").strip():
        return True
    data_path = str(state.get("data_path") or "").strip()
    return bool(data_path and state.get("path_validated"))


def _copilot_real_data_path_reply(
    state: Mapping[str, object],
    lang: str,
) -> str:
    data_path = str(state.get("data_path") or "").strip()
    export_dir = str(state.get("last_export_dir") or state.get("export_path") or "").strip()
    path_validated = bool(state.get("path_validated"))
    if lang == "en":
        current = ""
        if path_validated and data_path:
            current = f"\n\nCurrent validated data path: `{data_path}`."
        elif data_path:
            current = f"\n\nCurrent typed path: `{data_path}`. Validate it before analysis."
        if export_dir:
            current += f"\nCurrent module export folder: `{export_dir}`."
        return (
            "For real data, EasyICU uses a **prepared/converted data path**, not a raw download or a random temp folder. "
            "I opened the path field below this conversation; save the prepared path there. "
            "You can also use the chat shortcut `set data path /path/to/prepared_miiv`. "
            "If your ICU database has not been converted yet, use **Classic workspace** only when you explicitly want the classic Validate Data Path -> Convert & Setup screen. "
            "After conversion/export, Agent setup should use the prepared directory or the EasyICU module export folder."
            f"{current}"
        )
    current_zh = ""
    if path_validated and data_path:
        current_zh = f"\n\n当前已验证数据路径：`{data_path}`。"
    elif data_path:
        current_zh = f"\n\n当前已填写路径：`{data_path}`。正式分析前需要先验证。"
    if export_dir:
        current_zh += f"\n当前模块导出文件夹：`{export_dir}`。"
    return (
        "真实数据这里的“路径”指 **prepared/converted data path**，不是原始下载包，也不是随便一个 `/tmp` 文件夹。"
        "我已经在对话下方打开路径输入框；请把 prepared 路径保存到那里。"
        "你也可以用聊天快捷写法 `set data path /path/to/prepared_miiv`。"
        "如果数据库还没转换，只有你明确要走经典流程时，才打开 **经典工作区** 使用 Validate Data Path -> Convert & Setup。"
        "转换/导出完成后，Agent setup 使用 prepared 目录或 EasyICU 模块导出文件夹继续跑。"
        f"{current_zh}"
    )


def _copilot_workflow_snapshot(
    state: Mapping[str, object],
    lang: str,
) -> dict[str, object]:
    raw_study = state.get("_copilot_guided_study")
    study: Mapping[str, object] = raw_study if isinstance(raw_study, dict) else {}
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    active_step = str(study.get("step") or "question")
    active_idx = COPILOT_STEP_INDEX.get(active_step, 0)
    raw_question = str(study.get("question") or "").strip()
    analysis_label = str(study.get("analysis_label") or "").strip()
    if raw_question:
        question = raw_question
    elif active_step == "cohort":
        question = (
            "Research question not set yet. Cohort setup is open because you asked to configure the cohort first."
            if lang == "en" else
            "尚未设置研究问题；当前按你的要求先配置队列。"
        )
    elif active_step == "data":
        question = (
            "Research question not set yet. Data source setup is open first."
            if lang == "en" else
            "尚未设置研究问题；当前先确认数据源。"
        )
    else:
        question = (
            "Not framed yet. Choose the research question first."
            if lang == "en" else
            "尚未框定。先选择研究问题。"
        )
    step_title, step_detail = _copilot_current_step_detail(study, lang)
    api = _copilot_api_connection_snapshot(state, lang)
    is_en = lang == "en"
    data_mode = str(study.get("data_mode") or "real")
    data_status = str(study.get("data_source_status") or "")
    data_choice = str(study.get("data_source_choice") or "")
    if data_mode == "real":
        if data_status == "pending_validation":
            data_value = "path pending" if is_en else "路径待验证"
        elif data_status == "awaiting_path":
            data_value = "enter path" if is_en else "等待填写路径"
        elif data_status in {"module_export_recorded", "conversion_needed"}:
            data_value = "source recorded" if is_en else "数据源已记录"
        elif data_choice:
            data_value = "real source selected" if is_en else "已选择真实数据源"
        else:
            data_value = "not connected" if is_en else "未连接"
    else:
        data_value = "not connected" if is_en else "未连接"
    cohort_value = (
        _copilot_cohort_label(study, lang)
        if bool(study.get("cohort_configured")) or _copilot_uses_eligible_cohort(study) else
        "not chosen" if is_en else "未选择"
    )
    modules_value = (
        str(len(study.get("modules") or COPILOT_DEFAULT_MODULES))
        if bool(study.get("concepts_configured")) else
        "not chosen" if is_en else "未选择"
    )
    steps = []
    step_labels_zh = {
        "question": "问题",
        "data": "数据",
        "cohort": "队列",
        "concepts": "模块",
        "extract": "提取",
        "review": "审阅",
        "analysis": "Agent",
        "draft": "闸门",
    }
    for idx, (step, label_en) in enumerate(COPILOT_STUDY_STEPS):
        status = "done" if idx < active_idx else ("active" if idx == active_idx else "pending")
        if step == "draft" and study.get("draft_signed"):
            status = "done"
        steps.append({
            "id": step,
            "label": label_en if is_en else step_labels_zh.get(step, label_en),
            "status": status,
        })
    return {
        "title": "Workflow" if is_en else "工作流",
        "branch": analysis_label or (config["chip"] if is_en else {
            "predict": "建模 ICU 结局",
            "crossdb": "跨数据库比较",
            "quality": "数据质量审计",
        }.get(branch, str(config["chip"]))),
        "question": question,
        "active_step": active_step,
        "step_title": step_title,
        "step_detail": step_detail,
        "steps": steps,
        "facts": [
            {
                "id": "data",
                "label": "Data source" if is_en else "数据源",
                "value": data_value,
            },
            {
                "id": "cohort",
                "label": "Cohort" if is_en else "队列",
                "value": cohort_value,
            },
            {
                "id": "concepts",
                "label": "Modules" if is_en else "模块",
                "value": modules_value,
            },
            {
                "id": "api",
                "label": "API" if is_en else "API",
                "value": str(api["status"]),
            },
        ],
        "api": api,
        "gate": (
            "Evidence-bound · draft locked until checks pass"
            if is_en else
            "证据绑定 · 检查通过前草稿锁定"
        ) if not study.get("draft_signed") else (
            "Signed off locally · open Agent draft preview"
            if is_en else
            "已本地确认 · 可打开 Agent 草稿预览"
        ),
    }


def _normalized_copilot_message_content(content: str, lang: str) -> str:
    if lang == "en":
        return re.sub(
            r"Configured cohort scope \*\*(?:all eligible stays|eligible real-data cohort)\*\*\. "
            r"The chat state is now ready to open \*\*Classic workspace\*\* for extraction/review, "
            r"or \*\*Agent setup\*\* for an auditable run\.",
            "Cohort option noted: **eligible real-data cohort**. I paused here so you can confirm the data source, cohort scope, and feature modules before continuing.",
            content,
        )
    return re.sub(
        r"已配置队列范围 \*\*(?:全量合格 stay|真实数据合格队列)\*\*。现在可以打开 \*\*经典工作区\*\* 做提取/审阅，"
        r"或进入 \*\*Agent 配置\*\* 启动可审计分析。",
        "已记录队列选项：**真实数据合格队列**。我先停在这里，等你确认数据源、队列范围和特征模块后再继续。",
        content,
    )


def _request_copilot_scroll_to_latest(state: MutableMapping[str, object] | None = None) -> None:
    """Ask the Copilot chat shell to scroll to the newest turn on the next render."""
    if state is None:
        state = st.session_state
    state["_copilot_scroll_to_latest"] = True


def _copilot_user_message(prompt: str, display_prompt: str | None = None) -> dict[str, object]:
    """Build a visible user message while preserving hidden routing text when needed."""
    route_prompt = (prompt or "").strip()
    visible_prompt = (display_prompt or route_prompt).strip()
    message: dict[str, object] = {"role": "user", "content": visible_prompt or route_prompt}
    if route_prompt and visible_prompt and route_prompt != visible_prompt:
        message["route_prompt"] = route_prompt
    return message


def _append_copilot_workflow_step_action(step_id: str, lang: str) -> None:
    """Move the Copilot workflow card to a clicked step without leaving chat."""
    state = st.session_state
    valid_steps = {step for step, _label in COPILOT_STUDY_STEPS}
    if step_id not in valid_steps:
        return
    study = _ensure_copilot_study_state(state)
    study["step"] = step_id
    if step_id != "question":
        study.pop("question_substep", None)
    if step_id != "cohort":
        study.pop("cohort_substep", None)
    state["_copilot_guided_study"] = study
    label = _copilot_step_label(step_id, lang)
    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list):
        user_content = (
            f"Workflow step: {label}"
            if lang == "en" else
            f"工作流步骤：{label}"
        )
        assistant_content = (
            f"Switched the workflow card to **{label}**. Use the controls in this message; we stay in Copilot."
            if lang == "en" else
            f"已把 workflow 卡切到 **{label}**。直接在这条回复里选择或填写；我们仍然留在 Copilot。"
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "actions": _copilot_guided_choice_actions(study, lang),
            "workflow_snapshot": _copilot_workflow_snapshot(state, lang),
        })
    _request_copilot_scroll_to_latest(state)
    state["_active_main_page"] = "assistant"
    state["_assistant_notice"] = (
        f"Workflow card switched to {label}."
        if lang == "en" else
        f"Workflow 卡已切到{label}。"
    )


def _render_copilot_workflow_step_controls(
    snapshot: Mapping[str, object],
    lang: str,
    key_prefix: str,
) -> None:
    """Render a compact set of real buttons near the current workflow step.

    Disabled: the right-hand Study panel stepper is now the single, clickable
    step map. Re-rendering prev/current/next chips inside the thread duplicated
    that navigation, so we no longer draw them here.
    """
    return
    step_items = _copilot_visible_workflow_step_items(snapshot, lang)
    if not step_items:
        return
    with st.container(key=f"{key_prefix}_workflow_step_controls"):
        st.markdown(
            '<div class="eu-copilot-flow-controls-label">'
            f'{html.escape("Current workflow step" if lang == "en" else "当前工作流步骤")}'
            "</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(step_items))
        for col_idx, step in enumerate(step_items):
            step_id = str(step.get("id") or "")
            label = _copilot_workflow_button_label(
                {
                    "label": str(step.get("label") or _copilot_step_label(step_id, lang)),
                    "status": str(step.get("status") or ""),
                },
                lang,
            )
            with cols[col_idx]:
                if st.button(
                    label,
                    key=f"{key_prefix}_workflow_step_{col_idx}_{step_id}",
                    use_container_width=True,
                ):
                    _append_copilot_workflow_step_action(step_id, lang)
                    st.rerun()


def _append_copilot_inline_edit_message(
    state: MutableMapping[str, object],
    lang: str,
    *,
    user_content: str,
    assistant_content: str,
) -> None:
    """Record inline workflow edits as normal Copilot chat turns."""
    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list):
        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "actions": _copilot_guided_choice_actions(_ensure_copilot_study_state(state), lang),
            "workflow_snapshot": _copilot_workflow_snapshot(state, lang),
        })
    _request_copilot_scroll_to_latest(state)
    state["llm_last_tool_events"] = []
    state["llm_last_verification"] = {
        "status": "pass",
        "issues": [],
    }
    _touch_current_copilot_study_session(state, lang)


def _save_copilot_question_from_inline_editor(
    state: MutableMapping[str, object],
    lang: str,
    question: str,
) -> bool:
    """Persist a question edited from the workflow card."""
    question = (question or "").strip()
    if not question:
        state["_assistant_notice"] = (
            "Enter a research question before saving."
            if lang == "en" else
            "请先输入研究问题，再保存。"
        )
        return False
    study = _ensure_copilot_study_state(state)
    study["question"] = question
    study.pop("question_substep", None)
    if str(study.get("step") or "question") == "question":
        study["step"] = "data"
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    state["_copilot_guided_study"] = study
    state["_copilot_last_question"] = question
    state["_copilot_inline_question_editor_open"] = False
    state["_active_main_page"] = "assistant"
    _append_copilot_inline_edit_message(
        state,
        lang,
        user_content=(
            f"Set research question: {question}"
            if lang == "en" else
            f"设置研究问题：{question}"
        ),
        assistant_content=(
            "Research question saved in Copilot. Next, choose the data source in this same chat."
            if lang == "en" else
            "研究问题已保存在 Copilot。下一步继续在当前聊天里选择数据源。"
        ),
    )
    state["_assistant_notice"] = (
        "Research question saved in Copilot."
        if lang == "en" else
        "研究问题已保存在 Copilot。"
    )
    return True


def _save_copilot_api_from_inline_editor(
    state: MutableMapping[str, object],
    lang: str,
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    enabled: bool,
) -> bool:
    """Persist provider settings from the Copilot workflow card."""
    provider = coerce_public_provider(provider)
    provider_label, default_url, default_model, needs_key, _desc_en, _desc_zh = public_provider_defaults(provider)
    base_url = (base_url or default_url or "").strip()
    model = (model or default_model or "").strip()
    api_key = (api_key or "").strip()
    if needs_key and not api_key:
        state["_assistant_notice"] = (
            "Enter an API key before saving this provider."
            if lang == "en" else
            "请先填写 API Key，再保存该服务商。"
        )
        return False
    if not base_url:
        state["_assistant_notice"] = (
            "Enter an API base URL before saving."
            if lang == "en" else
            "请先填写 API Base URL，再保存。"
        )
        return False
    state["llm_provider"] = provider
    state["llm_api_key"] = api_key if needs_key else ""
    state["llm_base_url"] = base_url
    state["llm_model"] = model
    state["llm_enabled"] = bool(enabled)
    state["llm_configured"] = True
    state["_llm_provider_sel"] = provider
    state["_llm_api_key_inp"] = api_key if needs_key else ""
    state["_llm_base_url_inp"] = base_url
    state["_llm_model_inp"] = model
    state["_llm_toggle"] = bool(enabled)
    state["_eu_settings_allow_outbound_model_calls"] = bool(enabled)
    state["_copilot_inline_api_editor_open"] = False
    state["_active_main_page"] = "assistant"
    status_phrase = (
        "enabled for outbound model calls" if enabled else "configured but outbound calls are off"
    ) if lang == "en" else (
        "已允许模型调用" if enabled else "已配置，但模型调用未开启"
    )
    _append_copilot_inline_edit_message(
        state,
        lang,
        user_content=(
            f"Update API provider to {provider_label}."
            if lang == "en" else
            f"更新 API 服务商为 {provider_label}。"
        ),
        assistant_content=(
            f"API settings saved for this browser session: **{provider_label}** · `{model or 'model'}` · {status_phrase}. API keys are not written to disk."
            if lang == "en" else
            f"API 设置已保存到当前浏览器会话：**{provider_label}** · `{model or 'model'}` · {status_phrase}。API Key 不会写入本地文件。"
        ),
    )
    state["_assistant_notice"] = (
        "API settings saved in the current session."
        if lang == "en" else
        "API 设置已保存到当前会话。"
    )
    return True


def _render_copilot_question_inline_editor(lang: str, key_prefix: str) -> None:
    """Render the inline question editor opened from the workflow card."""
    if not st.session_state.get("_copilot_inline_question_editor_open"):
        return
    is_en = lang == "en"
    study = _ensure_copilot_study_state(st.session_state)
    current_question = str(study.get("question") or "").strip()
    with st.container(key=f"{key_prefix}_question_panel_editor"):
        st.markdown(
            '<div class="eu-copilot-inline-editor-title">'
            f'{html.escape("Edit research question" if is_en else "编辑研究问题")}'
            "</div>",
            unsafe_allow_html=True,
        )
        with st.form(f"{key_prefix}_question_editor_form", clear_on_submit=False):
            question = st.text_area(
                "Research question" if is_en else "研究问题",
                value=current_question,
                placeholder=(
                    "Describe the study question in one sentence..."
                    if is_en else
                    "用一句话描述你的研究问题..."
                ),
                key=f"{key_prefix}_question_editor_text",
                height=92,
            )
            cols = st.columns([1, 0.24], gap="small")
            with cols[0]:
                submitted = st.form_submit_button(
                    "Save question" if is_en else "保存研究问题",
                    type="primary",
                    use_container_width=True,
                )
            with cols[1]:
                cancelled = st.form_submit_button(
                    "Cancel" if is_en else "取消",
                    use_container_width=True,
                )
        if submitted:
            if _save_copilot_question_from_inline_editor(st.session_state, lang, question):
                st.rerun()
        if cancelled:
            st.session_state["_copilot_inline_question_editor_open"] = False
            st.rerun()


def _render_copilot_api_inline_editor(lang: str, key_prefix: str) -> None:
    """Render the inline API editor opened from the workflow card."""
    if not st.session_state.get("_copilot_inline_api_editor_open"):
        return
    is_en = lang == "en"
    provider_keys = public_provider_keys()
    current_provider = coerce_public_provider(
        str(st.session_state.get("llm_provider") or public_default_provider_key())
    )
    provider_index = provider_keys.index(current_provider) if current_provider in provider_keys else 0
    _provider_label, default_url, default_model, _needs_key, _desc_en, _desc_zh = public_provider_defaults(current_provider)
    current_base = str(st.session_state.get("llm_base_url") or default_url or "")
    current_model = str(st.session_state.get("llm_model") or default_model or "")
    current_key = str(st.session_state.get("llm_api_key") or "")
    with st.container(key=f"{key_prefix}_api_panel_editor"):
        st.markdown(
            '<div class="eu-copilot-inline-editor-title">'
            f'{html.escape("Configure API for this session" if is_en else "配置当前会话 API")}'
            "</div>",
            unsafe_allow_html=True,
        )
        with st.form(f"{key_prefix}_api_editor_form", clear_on_submit=False):
            provider = st.selectbox(
                "Provider" if is_en else "服务商",
                options=provider_keys,
                index=provider_index,
                format_func=lambda item: public_provider_defaults(item)[0],
                key=f"{key_prefix}_api_editor_provider",
            )
            _selected_label, selected_default_url, selected_default_model, selected_needs_key, desc_en, desc_zh = public_provider_defaults(provider)
            st.caption(desc_en if is_en else desc_zh)
            api_key = st.text_input(
                "API key" if is_en else "API Key",
                value=current_key if selected_needs_key else "",
                type="password",
                placeholder="sk-...",
                disabled=not selected_needs_key,
                key=f"{key_prefix}_api_editor_key",
            )
            base_url = st.text_input(
                "API Base URL",
                value=current_base or selected_default_url,
                placeholder=selected_default_url or "https://api.example.com/v1",
                key=f"{key_prefix}_api_editor_base",
            )
            model = st.text_input(
                "Model" if is_en else "模型",
                value=current_model or selected_default_model,
                placeholder=selected_default_model or "model-name",
                key=f"{key_prefix}_api_editor_model",
            )
            enabled = st.checkbox(
                "Allow outbound model calls for this session"
                if is_en else
                "允许当前会话进行模型调用",
                value=bool(st.session_state.get("llm_enabled")),
                key=f"{key_prefix}_api_editor_enabled",
            )
            st.caption(
                "The key stays only in Streamlit session state; EasyICU does not write it to local files."
                if is_en else
                "Key 只保存在 Streamlit 当前会话中，EasyICU 不会写入本地文件。"
            )
            cols = st.columns([1, 0.24], gap="small")
            with cols[0]:
                submitted = st.form_submit_button(
                    "Save API settings" if is_en else "保存 API 设置",
                    type="primary",
                    use_container_width=True,
                )
            with cols[1]:
                cancelled = st.form_submit_button(
                    "Cancel" if is_en else "取消",
                    use_container_width=True,
                )
        if submitted:
            if _save_copilot_api_from_inline_editor(
                st.session_state,
                lang,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                enabled=enabled,
            ):
                st.rerun()
        if cancelled:
            st.session_state["_copilot_inline_api_editor_open"] = False
            st.rerun()


def _render_copilot_workflow_inline_edit_controls(
    snapshot: Mapping[str, object],
    lang: str,
    key_prefix: str,
) -> None:
    """Render real buttons for editing the card's static-looking status rows."""
    is_en = lang == "en"
    with st.container(key=f"{key_prefix}_workflow_inline_edits"):
        st.markdown(
            '<div class="eu-copilot-flow-controls-label">'
            f'{html.escape("Edit in this chat" if is_en else "在当前聊天中编辑")}'
            "</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2, gap="small")
        with cols[0]:
            if st.button(
                "Edit research question" if is_en else "编辑研究问题",
                key=f"{key_prefix}_workflow_edit_question",
                use_container_width=True,
                help=(
                    "Open an in-chat field for the study question."
                    if is_en else
                    "在当前聊天中打开研究问题输入框。"
                ),
            ):
                st.session_state["_copilot_inline_question_editor_open"] = True
                st.session_state["_copilot_inline_api_editor_open"] = False
                study = _ensure_copilot_study_state(st.session_state)
                study["step"] = "question"
                study.pop("question_substep", None)
                st.session_state["_copilot_guided_study"] = study
                st.rerun()
        with cols[1]:
            api = snapshot.get("api") if isinstance(snapshot.get("api"), Mapping) else {}
            api_status = str(api.get("status") or "local")
            if st.button(
                "Configure API" if is_en else "配置 API",
                key=f"{key_prefix}_workflow_edit_api",
                use_container_width=True,
                help=(
                    f"Current API state: {api_status}. Settings stay in this browser session."
                    if is_en else
                    f"当前 API 状态：{api_status}。设置只保存在当前浏览器会话。"
                ),
            ):
                st.session_state["_copilot_inline_api_editor_open"] = True
                st.session_state["_copilot_inline_question_editor_open"] = False
                st.rerun()
    _render_copilot_question_inline_editor(lang, key_prefix)
    _render_copilot_api_inline_editor(lang, key_prefix)


def _render_copilot_workflow_snapshot(
    snapshot: object,
    lang: str,
    key_prefix: str,
) -> None:
    if not isinstance(snapshot, Mapping):
        return
    st.markdown(_copilot_workflow_snapshot_html(snapshot, lang), unsafe_allow_html=True)
    _render_copilot_workflow_inline_edit_controls(snapshot, lang, key_prefix)
    _render_copilot_workflow_step_controls(snapshot, lang, key_prefix)


def _copilot_workflow_control_context(lang: str) -> dict[str, object]:
    """Structured workflow/menu contract shown to the model controller."""
    is_en = lang == "en"
    return {
        "principle": (
            "The model leads the conversation by selecting the next EasyICU workflow step and returning UI choices. "
            "The app executes exact prompt commands and renders forms in the current Copilot page; it should not ask the user to navigate elsewhere."
        ),
        "steps": [
            {"id": step, "label": label if is_en else _copilot_step_label(step, "zh")}
            for step, label in COPILOT_STUDY_STEPS
        ],
        "choice_contract": {
            "button": {
                "kind": "copilot_prompt",
                "fields": ["id", "label", "prompt"],
                "note": "Use exact prompt commands below when you want a button to execute a known workflow control.",
            },
            "input_request": {
                "supported_kinds": ["prepared_data_path", "module_export_folder", "raw_icu_files"],
                "note": "Use this when the next best UI is a field, not another explanatory paragraph.",
            },
        },
        "known_controls": {
            "capability_overview": [
                {"label": "新建研究 / 工作目录", "prompt": "新研究"},
                {"label": "描述研究目标", "prompt": "我想自己描述研究问题。"},
                {
                    "label": "接入真实 ICU 数据",
                    "prompt": "逐步带我完成真实数据源步骤；先解释 prepared 数据路径，不要直接选择其他部分。",
                },
                {"label": "定义研究队列", "prompt": "逐步带我完成队列步骤；先解释选项，不要直接替我选择。"},
                {"label": "选择特征模块", "prompt": "选择特征模块。"},
                {"label": "探索研究 idea", "prompt": "在这个聊天里推荐 ICU 研究 idea，然后问我想继续看哪一个。"},
                {"label": "准备 Agent 分析", "prompt": "在聊天里准备提取计划。"},
            ],
            "question": [
                {"label": "建模 ICU 结局", "prompt": "研究类型：结局预测"},
                {"label": "治疗暴露研究", "prompt": "研究类型：治疗暴露"},
                {"label": "跨库比较", "prompt": "研究类型：跨库比较"},
                {"label": "数据质量审计", "prompt": "研究类型：数据质量审计"},
                {"label": "我自己描述问题", "prompt": "我想自己描述研究问题。"},
            ],
            "data": [
                {
                    "label": "已有 prepared 路径",
                    "prompt": "我有 prepared 数据路径。",
                    "input_request": "prepared_data_path",
                },
                {
                    "label": "已有模块导出",
                    "prompt": "我有 EasyICU 模块导出文件夹。",
                    "input_request": "module_export_folder",
                },
                {
                    "label": "只有 ICU 原始文件",
                    "prompt": "我只有 ICU 原始文件。",
                    "input_request": "raw_icu_files",
                },
            ],
            "cohort": [
                {"label": "全部合格队列", "prompt": "使用全部合格队列。"},
                {"label": "按疾病/诊断", "prompt": "配置疾病或诊断队列过滤。"},
                {"label": "按年龄/ICU LOS", "prompt": "配置年龄或 ICU LOS 限制。"},
                {"label": "使用当前审阅队列", "prompt": "使用当前审阅队列。"},
            ],
            "feature_modules": [
                {
                    "label": _copilot_feature_pack_label(key, "zh"),
                    "prompt": _copilot_feature_module_prompt(key, "zh"),
                }
                for key in _copilot_feature_module_action_keys()
            ] + [{"label": "使用模型推荐模块", "prompt": "用这些变量"}],
            "extract_review_agent": [
                {"label": "准备提取计划", "prompt": "在聊天里准备提取计划。"},
                {"label": "解释证据闸门", "prompt": "为什么这一步？"},
                {"label": "返回上一步", "prompt": "返回上一步"},
            ],
        },
        "classic_alignment": {
            "data_source": "Classic Data Extraction Step 1: database/source path/converted export.",
            "cohort": "Classic Data Extraction Step 2: disease, age, sex, ICU LOS, mortality/status filters.",
            "feature_modules": "Classic Data Extraction Step 3: concept/module selection and coverage expectations.",
            "extract": "Classic Data Extraction Step 4: export/freeze a reproducible frame.",
            "review": "Patient Review/Quick Visualization: inspect tables, time series, cohort statistics, quality flags.",
            "analysis": "Research Agent setup/run: audited plan, deterministic artifacts, evidence gate.",
        },
    }


def _copilot_current_state_context(
    state: Mapping[str, object],
    study: Mapping[str, object],
) -> dict[str, object]:
    return {
        "current_step": str(study.get("step") or "question"),
        "analysis_family": str(study.get("analysis_family") or ""),
        "analysis_label": str(study.get("analysis_label") or ""),
        "data_mode": str(study.get("data_mode") or "real"),
        "has_question": bool(str(study.get("question") or "").strip()),
        "question": str(study.get("question") or ""),
        "cohort_configured": bool(study.get("cohort_configured")),
        "concepts_configured": bool(study.get("concepts_configured")),
        "data_source_choice": str(study.get("data_source_choice") or state.get("_copilot_data_source_choice") or ""),
        "data_path_set": bool(str(state.get("real_data_path") or state.get("data_path") or "").strip()),
        "selected_concepts": [
            str(item)
            for item in list(study.get("selected_concepts") or [])
            if str(item).strip()
        ],
        "suggested_concepts": [
            str(item)
            for item in list(study.get("suggested_concepts") or [])
            if str(item).strip()
        ],
    }


def _copilot_route_transport_enabled(state: Mapping[str, object]) -> bool:
    if bool(state.get("_copilot_allow_route_model")):
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return state is st.session_state


def _copilot_route_with_llm(
    prompt: str,
    lang: str,
    state: Mapping[str, object],
    study: Mapping[str, object],
) -> dict[str, object] | None:
    """Ask the configured model to drive the next guided Copilot step."""
    if not _copilot_route_transport_enabled(state):
        return None
    try:
        if not _external_llm_ready(lang):
            return None
    except AIOptInError:
        return None

    client = _get_client()
    if client is None:
        return None

    is_en = lang == "en"
    status_context = _copilot_current_state_context(state, study)
    workflow_context = _copilot_workflow_control_context(lang)
    system_prompt = (
        "You are the EasyICU Research Copilot workflow controller. Return JSON only.\n"
        "You lead a guided chat workflow. The app provides EasyICU's process, allowed UI controls, "
        "and current state; you decide the next workflow step, the short assistant reply, and the exact "
        "buttons or inline input field the user should see now.\n"
        "Rules:\n"
        "- Do not tell the user to navigate to another page or Classic Workspace.\n"
        "- Do not output tutorial prose when the user needs a choice or a field.\n"
        "- Do not invent a different endpoint, cohort, task, or analysis family than the user asked for.\n"
        "- If the user gives a concrete study goal, do not return broad research-type/question-type choices. "
        "Classify the analysis_family and ask only the next missing parameter.\n"
        "- For real-time warning or prediction goals without a clear target event, set current_step=question and "
        "return target-event/endpoint choices, not research-type choices.\n"
        "- Use the provided known_controls prompt strings when a choice maps to an EasyICU control.\n"
        "- The broad known_controls.question menu is only for empty/generic help, not for concrete study messages.\n"
        "- If the next action is entering a path/folder, set input_request.kind instead of only describing it.\n"
        "- For a specific user goal, keep choices narrow and relevant; avoid broad unrelated endpoint lists.\n"
        "- If the user asks what they can do, use capability_overview choices: data, cohort, modules, idea exploration, Agent run, or describe a goal. Do not return research-type taxonomy.\n"
        "- If the user asks how to use Copilot, return the next useful workflow choices for the current state.\n"
        "- If the user asks to configure cohort/modules/data, return those exact controls in the current page.\n"
        "- Prefer real-data workflow unless the user explicitly asks for demo.\n"
        "- Do not request or expose patient rows. Paths and study text are allowed.\n"
        "- Use the user's language for `assistant_text`, `next_question`, and choice labels.\n"
        "- Allowed current_step values: question, data, cohort, concepts, extract, review, analysis, draft.\n"
        "- Allowed analysis_family values: prediction, association, clustering, trajectory, quality_audit, cross_database, descriptive, unknown.\n"
        "JSON schema:\n"
        "{"
        "\"analysis_family\":\"clustering\","
        "\"analysis_label\":\"short label\","
        "\"study_frame\":\"one sentence preserving the user's goal\","
        "\"current_step\":\"data\","
        "\"assistant_text\":\"short conversational reply\","
        "\"next_question\":\"what the user should choose or fill next\","
        "\"input_request\":{\"kind\":\"prepared_data_path\",\"label\":\"optional field label\"},"
        "\"cohort\":{\"label\":\"optional cohort hint\",\"filters\":[\"optional filters from the user only\"]},"
        "\"suggested_concepts\":[\"optional EasyICU concept keys if directly useful\"],"
        "\"choices\":[{\"id\":\"stable_id\",\"label\":\"button label\",\"prompt\":\"message to send if clicked\"}]"
        "}"
    )
    user_context = (
        "Language: {lang}\n"
        "Current Copilot state: {state_json}\n"
        "EasyICU workflow/control contract: {workflow_json}\n"
        "User message: {prompt}"
    ).format(
        lang="English" if is_en else "Chinese",
        state_json=json.dumps(status_context, ensure_ascii=False),
        workflow_json=json.dumps(workflow_context, ensure_ascii=False),
        prompt=prompt,
    )
    try:
        route_client = (
            client.with_options(timeout=COPILOT_ROUTE_TIMEOUT_SECONDS, max_retries=0)
            if hasattr(client, "with_options") else
            client
        )
        response = _copilot_create_route_completion(
            route_client,
            {
                "model": st.session_state.get("llm_model", "").strip()
                or public_provider_defaults(st.session_state.get("llm_provider", public_default_provider_key()))[2],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context},
                ],
                "temperature": 0,
                "max_tokens": COPILOT_ROUTE_MAX_TOKENS,
                "stream": False,
            },
        )
    except Exception:
        return None
    if response is None:
        return None
    text = response.choices[0].message.content if response.choices else ""
    return _copilot_extract_route_json(text)


def _copilot_apply_llm_route(
    route: Mapping[str, object],
    state: MutableMapping[str, object],
    study: MutableMapping[str, object],
    lang: str,
) -> tuple[str, list[dict[str, object]]]:
    family = str(route.get("analysis_family") or "unknown").strip().lower()
    if family not in COPILOT_ROUTE_ALLOWED_FAMILIES:
        family = "unknown"
    step = str(route.get("current_step") or "data").strip().lower()
    if step not in COPILOT_ROUTE_ALLOWED_STEPS:
        step = "data"
    frame = str(route.get("study_frame") or route.get("question") or "").strip()
    analysis_label = str(route.get("analysis_label") or "").strip() or _copilot_route_family_label(family, lang)

    study["analysis_family"] = family
    study["analysis_label"] = analysis_label
    study["branch"] = _copilot_branch_for_route_family(family)
    study["data_mode"] = "real"
    study["step"] = step
    study["cohort_configured"] = bool(study.get("cohort_configured"))
    study["concepts_configured"] = bool(study.get("concepts_configured"))
    if family == "prediction" and step == "question" and not str(study.get("question_substep") or ""):
        study["question_substep"] = "endpoint"
    if frame:
        study["question"] = frame

    input_request = route.get("input_request")
    if isinstance(input_request, Mapping):
        input_kind = str(input_request.get("kind") or "").strip()
        input_to_source_choice = {
            "prepared_data_path": "prepared_path",
            "module_export_folder": "module_export",
            "raw_icu_files": "raw_files",
        }
        source_choice = input_to_source_choice.get(input_kind)
        if source_choice:
            study = _copilot_set_data_source_choice(state, source_choice)
            study["analysis_family"] = family
            study["analysis_label"] = analysis_label
            study["branch"] = _copilot_branch_for_route_family(family)
            study["data_mode"] = "real"
            study["step"] = "data"
            if frame:
                study["question"] = frame

    cohort = route.get("cohort")
    if isinstance(cohort, Mapping):
        cohort_label = str(cohort.get("label") or "").strip()
        if cohort_label:
            study["cohort_hint"] = cohort_label
        filters = [
            str(item).strip()
            for item in list(cohort.get("filters") or [])
            if str(item).strip()
        ]
        if filters:
            study["route_cohort_filters"] = filters[:6]

    raw_concepts = route.get("suggested_concepts")
    if isinstance(raw_concepts, list):
        concepts = [
            str(item).strip()
            for item in raw_concepts
            if str(item).strip()
        ][:12]
        if concepts:
            study["suggested_concepts"] = concepts
            study["modules"] = _copilot_modules_for_concepts(concepts)

    actions = _copilot_route_choice_actions(route, lang)
    route_is_specific = _copilot_route_has_specific_goal(route)
    if route_is_specific and _copilot_route_uses_broad_question_type_choices(actions):
        actions = []
    if actions:
        study["_route_actions"] = actions
        study["_route_actions_step"] = step
    else:
        study.pop("_route_actions", None)
        study.pop("_route_actions_step", None)
        actions = _copilot_guided_choice_actions(study, lang)

    assistant_text = str(route.get("assistant_text") or "").strip()
    next_question = str(route.get("next_question") or "").strip()
    if not assistant_text:
        assistant_text = (
            f"I understand this as **{analysis_label}**."
            if lang == "en" else
            f"我理解这是 **{analysis_label}**。"
        )
    elif route_is_specific and _copilot_route_next_question_asks_broad_type(assistant_text):
        assistant_text = (
            f"I understand this as **{analysis_label}** and will keep the workflow specific."
            if lang == "en" else
            f"我理解这是 **{analysis_label}**，接下来只追问这个研究所缺的具体配置。"
        )
    if route_is_specific and _copilot_route_next_question_asks_broad_type(next_question):
        next_question = _copilot_specific_route_next_question(family, step, lang)
    body = assistant_text
    if next_question:
        body = f"{body}\n\n{next_question}"
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return _copilot_reply(study, body, lang, include_status=False), actions


# `_copilot_keyword_route` moved to `copilot/routing.py` (Phase-6 split) and is
# re-imported at the top of this module; callers below resolve unchanged.


def _copilot_llm_route_unavailable_reply(
    study: MutableMapping[str, object],
    lang: str,
    prompt: str = "",
) -> tuple[str, list[dict[str, object]]]:
    if _copilot_free_study_goal_requested(prompt):
        raw_prompt = str(prompt or "").strip()
        study["data_mode"] = "real"
        study["step"] = str(study.get("step") or "question")
        if not study.get("branch"):
            study["branch"] = "predict"
        if raw_prompt and not str(study.get("question") or "").strip():
            study["question"] = raw_prompt
            study["pending_user_goal"] = raw_prompt
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "The connected model did not route this quickly enough, so I recorded your goal and kept the page interactive. "
            "I will not send you back to a broad research-type menu. Continue with one concrete item below; the model can retry routing on your next free-text message."
            if lang == "en" else
            "接入的大模型没有在交互预算内返回路线，我先记录你的研究目标并保持页面可操作。"
            "我不会把你退回泛化的研究类型菜单。请先在下面选一个具体配置继续；你下一次自由输入时模型仍会重新判断路线。"
        )
        actions = _copilot_nonblocking_goal_fallback_actions(lang)
        study["_route_actions"] = actions
        study["_route_actions_step"] = str(study.get("step") or "question")
        return body, actions
    current_actions = _copilot_guided_choice_actions(study, lang)
    if current_actions:
        retry_goal = str(prompt or study.get("pending_user_goal") or study.get("question") or "").strip()
        if retry_goal:
            current_actions = [
                _copilot_prompt_action(
                    "choice_retry_routing",
                    "↻ Retry routing",
                    "↻ 重试路由",
                    retry_goal,
                    retry_goal,
                    lang,
                )
            ] + current_actions
        body = (
            "The connected model did not return a route quickly enough. I kept you in the current Copilot step; pick a control below, or tap **Retry routing** to ask the model again."
            if lang == "en" else
            "接入的大模型这次没有及时返回路线。我先把你留在当前 Copilot 步骤;可直接在下面选择,或点 **重试路由** 让模型再判断一次。"
        )
        return body, current_actions
    if lang == "en":
        body = (
            "This open-ended study goal needs the connected model to classify the route. "
            "I will not guess it with local keyword rules. Enable the OpenRouter/OpenAI-compatible provider, "
            "then send the same study goal again."
        )
        label = "Open API settings"
    else:
        body = (
            "这个开放式研究意图需要由接入的大模型来判断路线。"
            "我不会再用本地关键词规则替你猜。请先启用 OpenRouter/OpenAI-compatible provider，"
            "然后把同一句研究目标再发一次。"
        )
        label = "打开 API 设置"
    return body, [
        {
            "id": "workflow_api_settings",
            "kind": "workflow",
            "label": label,
            "workflow": "api_settings",
        }
    ]


def _copilot_capture_free_study_goal_first_pass(
    study: MutableMapping[str, object],
    lang: str,
    prompt: str,
) -> tuple[str, list[dict[str, object]]]:
    """Record an open study goal without blocking the UI on model routing.

    Even without a model, classify the goal deterministically (family / cohort /
    concept hints) so the first free-text goal is framed offline; a configured
    LLM still refines on the next message.
    """
    raw_prompt = str(prompt or "").strip()
    study["data_mode"] = "real"
    study["step"] = "data"
    kw = _copilot_keyword_route(raw_prompt, lang)
    classified_label = ""
    if isinstance(kw, Mapping):
        family = str(kw.get("analysis_family") or "unknown")
        study["analysis_family"] = family
        classified_label = str(kw.get("analysis_label") or "")
        study["analysis_label"] = classified_label or (
            "Guided study" if lang == "en" else "当前研究目标"
        )
        study["branch"] = _copilot_branch_for_route_family(family)
        cohort = kw.get("cohort")
        if isinstance(cohort, Mapping) and str(cohort.get("label") or "").strip():
            study["cohort_hint"] = str(cohort["label"]).strip()
        concepts = kw.get("suggested_concepts")
        if isinstance(concepts, list) and concepts:
            study["suggested_concepts"] = [str(c) for c in concepts]
            study["modules"] = _copilot_modules_for_concepts(study["suggested_concepts"])
    else:
        study["analysis_family"] = str(study.get("analysis_family") or "unknown")
        study["analysis_label"] = "Guided study" if lang == "en" else "当前研究目标"
        if not study.get("branch"):
            study["branch"] = "predict"
    study["question"] = raw_prompt
    study["pending_user_goal"] = raw_prompt
    study.pop("question_substep", None)
    study.pop("_route_actions", None)
    study.pop("_route_actions_step", None)
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    read_as = (
        (f" I read this as **{classified_label}**." if lang == "en"
         else f" 我把它理解为 **{classified_label}**。")
        if classified_label else ""
    )
    body = (
        f"I recorded your study goal: **{raw_prompt}**.{read_as}\n\n"
        "Next, choose how your real ICU data enters this Copilot workflow. I will keep every next choice in this chat."
        if lang == "en" else
        f"我已记录你的研究目标：**{raw_prompt}**。{read_as}\n\n"
        "下一步先选择真实 ICU 数据如何进入 Copilot。后续队列、特征模块和提取计划都会继续在当前聊天里完成。"
    )
    return body, _copilot_guided_choice_actions(study, lang)


def _copilot_guided_usage_reply(
    study: MutableMapping[str, object],
    lang: str,
    state: Mapping[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """Explain the current Copilot step and surface choices for it."""
    if (
        state.get("entry_mode") == "real"
        and str(study.get("step") or "question") == "question"
        and str(study.get("question") or "").strip()
    ):
        study["data_mode"] = "real"
        study["step"] = "data"
    step = str(study.get("step") or "question")
    step_label = {
        "question": ("Research question", "研究问题"),
        "data": ("Data source", "数据源"),
        "cohort": ("Cohort", "队列"),
        "concepts": ("Feature modules", "特征模块"),
        "extract": ("Extraction plan", "提取计划"),
        "review": ("Review", "审阅"),
        "analysis": ("Analysis", "分析"),
        "draft": ("Draft gate", "草稿闸门"),
    }.get(step, ("Research question", "研究问题"))
    if lang == "en":
        body = (
            "Use Copilot like a guided conversation, not like a page launcher.\n\n"
            f"Current step: **{step_label[0]}**. Pick one option below, then I will ask the next step in this same chat. "
            "Classic workspace stays available from the top bar, but Copilot choices do not jump there."
        )
    else:
        body = (
            "Copilot 的用法是**引导式对话**，不是告诉你去哪个页面。\n\n"
            f"当前步骤：**{step_label[1]}**。直接在下面选一个选项，我会在当前聊天里继续问下一步。"
            "顶部的经典工作区只是手动出口；Copilot 里的选择按钮不会跳转。"
        )
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return body, _copilot_guided_choice_actions(study, lang)


def _copilot_capability_overview_reply(
    study: MutableMapping[str, object],
    lang: str,
    state: Mapping[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """Surface broad capabilities without falling back to research-type taxonomy."""
    actions = _copilot_capability_overview_actions(study, state, lang)
    step = str(study.get("step") or "question")
    study["_route_actions"] = actions
    study["_route_actions_step"] = step
    study["data_mode"] = "real"
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    if lang == "en":
        body = (
            "In Copilot you can work at the main EasyICU levels: create a study workspace, connect real ICU data, "
            "define the cohort, select feature modules, explore research ideas, and prepare an auditable Agent run.\n\n"
            "Pick a high-level action below; I will keep the next form or choice in this chat."
        )
    else:
        body = (
            "在 Copilot 里，你可以从 EasyICU 的几个大层面开始：新建研究工作目录、接入真实 ICU 数据、"
            "定义研究队列、选择特征模块、探索研究 idea，以及准备可审计的 Agent 分析。\n\n"
            "直接选下面的大层级动作；下一步需要填写或选择的内容会继续出现在当前聊天里。"
        )
    return body, actions


def _copilot_infer_configured_step(study: Mapping[str, object]) -> str:
    """Infer the most useful visible workflow step from configured chat state."""
    current = str(study.get("step") or "question")
    if study.get("draft_signed") or current == "draft":
        return "draft"
    if _copilot_cohort_is_empty(dict(study)):
        return "cohort"
    if bool(study.get("concepts_configured")) or bool(study.get("selected_concepts")):
        return "concepts"
    if bool(study.get("cohort_configured")):
        return "cohort"
    if current in {"review", "analysis"}:
        return current
    if str(study.get("data_mode") or "demo") == "real":
        return "data"
    if str(study.get("question") or "").strip() or study.get("branch"):
        return "data"
    return "question"


def _copilot_sync_step_to_configuration(
    study: MutableMapping[str, object],
    *,
    allow_regress: bool = False,
) -> str:
    inferred = _copilot_infer_configured_step(study)
    current = str(study.get("step") or "question")
    current_idx = COPILOT_STEP_INDEX.get(current, 0)
    inferred_idx = COPILOT_STEP_INDEX.get(inferred, 0)
    if allow_regress or inferred_idx > current_idx:
        study["step"] = inferred
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return str(study.get("step") or inferred)


def _copilot_configured_reply(
    study: Mapping[str, object],
    lang: str,
    *,
    patient_count_requested: bool,
) -> str:
    patient_n = int(study.get("patient_n") or 10)
    cohort_label = _copilot_cohort_label(study, lang)
    concept_labels = _copilot_concept_label_list(study)
    concept_count = len(list(study.get("selected_concepts") or []))
    if lang == "en":
        pieces = []
        if _copilot_uses_eligible_cohort(study):
            pieces.append(f"cohort scope **{cohort_label}**")
        elif patient_count_requested or study.get("cohort_configured"):
            pieces.append(f"cohort denominator **{patient_n} stays**")
        if concept_count:
            label_text = ", ".join(concept_labels)
            if concept_count > len(concept_labels):
                label_text = f"{label_text}, +{concept_count - len(concept_labels)} more"
            pieces.append(f"feature set **{label_text}**")
        configured = " and ".join(pieces) or "the study frame"
        if _copilot_uses_eligible_cohort(study) and str(study.get("data_mode") or "demo") == "real":
            return (
                f"Configured {configured}. Paste the prepared data path/module export here in chat, "
                "or use **Classic workspace** only if you explicitly want the full validation screen. Agent setup will use this scope after the data source is ready."
            )
        return (
            f"Configured {configured}. The chat state is now ready to open **Classic workspace** "
            "for extraction/review, or **Agent setup** for an auditable run."
        )
    pieces = []
    if _copilot_uses_eligible_cohort(study):
        pieces.append(f"队列范围 **{cohort_label}**")
    elif patient_count_requested or study.get("cohort_configured"):
        pieces.append(f"队列分母 **{patient_n} 例 stay**")
    if concept_count:
        label_text = "、".join(concept_labels)
        if concept_count > len(concept_labels):
            label_text = f"{label_text}，另 {concept_count - len(concept_labels)} 个"
        pieces.append(f"特征集 **{label_text}**")
    configured = " 和 ".join(pieces) or "研究框架"
    if _copilot_uses_eligible_cohort(study) and str(study.get("data_mode") or "demo") == "real":
        return (
            f"已配置 {configured}。请直接在聊天框里粘贴 prepared 数据路径或模块导出；"
            "只有明确要走完整校验屏时再用 **经典工作区**。数据源就绪后 Agent 配置会使用这个队列范围。"
        )
    return (
        f"已配置 {configured}。现在可以打开 **经典工作区** 做提取/审阅，"
        "或进入 **Agent 配置** 启动可审计分析。"
    )


def _handle_copilot_guided_prompt(
    prompt: str,
    lang: str,
    state: MutableMapping[str, object] | None = None,
) -> tuple[str, list[dict[str, object]]] | None:
    """Handle local chat-first study control before falling back to an LLM."""
    state = state if state is not None else st.session_state
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    idea_selection = _copilot_handle_idea_selection(prompt, lang, state)
    if idea_selection is not None:
        return idea_selection
    if _copilot_research_recommendation_requested(prompt) or _is_idea_exploration_request(prompt):
        return _copilot_idea_recommendation_reply(prompt, lang, state)
    text_l = prompt.lower()
    study = _ensure_copilot_study_state(state)
    guided_active = bool(study.get("branch"))
    real_data_intent = _copilot_real_data_requested(prompt)
    full_cohort_intent = _copilot_full_cohort_requested(prompt)
    path_help_intent = _copilot_data_path_help_requested(prompt)
    api_intent = _copilot_api_setup_requested(prompt)
    step_by_step_intent = _copilot_step_by_step_requested(prompt)
    cohort_step_intent = _copilot_cohort_step_requested(prompt)
    feature_step_intent = _copilot_feature_step_requested(prompt)
    extract_step_intent = _copilot_extract_step_requested(prompt)
    review_step_intent = _copilot_review_step_requested(prompt)
    next_step_help_intent = _copilot_next_step_help_requested(prompt)
    usage_help_intent = _copilot_usage_help_requested(prompt)
    capability_overview_intent = _copilot_capability_overview_requested(prompt)
    patient_count_intent = _copilot_extract_patient_count(prompt) is not None
    guided_choice_intent = (
        _copilot_prepared_data_choice_requested(prompt)
        or _copilot_module_export_choice_requested(prompt)
        or _copilot_raw_files_choice_requested(prompt)
        or _copilot_disease_cohort_choice_requested(prompt)
        or _copilot_age_los_choice_requested(prompt)
        or _copilot_current_reviewed_cohort_requested(prompt)
        or _copilot_no_disease_filter_requested(prompt)
        or feature_step_intent
        or _copilot_module_pack_from_prompt(prompt) is not None
    )
    guided_intent = any(key in text_l for key in (
        "run the whole", "autopilot", "do it for me", "guided study", "whole demo",
        "walk me", "start a guided", "use demo", "use local", "go back", "why this step",
        "why?", "api", "openrouter", "model", "token", "研究", "帮我跑", "自动跑", "一键", "回退", "为什么",
    ))
    branch_intent = _copilot_pick_branch(prompt) != "predict" or any(key in text_l for key in (
        "sepsis", "mortality", "lactate", "aki", "trajectory", "cohort", "prediction",
        "脓毒症", "死亡", "乳酸", "队列", "预测",
    ))
    if full_cohort_intent or real_data_intent or path_help_intent:
        guided_intent = True
    if step_by_step_intent:
        guided_intent = True
    if cohort_step_intent:
        guided_intent = True
    if feature_step_intent:
        guided_intent = True
    if extract_step_intent:
        guided_intent = True
    if review_step_intent:
        guided_intent = True
    if next_step_help_intent:
        guided_intent = True
    if usage_help_intent:
        guided_intent = True
    if capability_overview_intent:
        guided_intent = True
    if patient_count_intent:
        guided_intent = True
    if guided_choice_intent:
        guided_intent = True
    if not (guided_active or guided_intent or branch_intent or api_intent or path_help_intent):
        return None

    if api_intent:
        api = _copilot_api_connection_snapshot(state, lang)
        provider_label = str(api.get("provider_label") or "OpenRouter")
        if lang == "en":
            if api.get("configured"):
                body = (
                    f"{provider_label} is already configured for this session. "
                    "Open Settings to enable shared outbound calls, pick a free model, or update the base URL. "
                    "Patient rows are never sent by the Copilot; model calls use plan text and page context only."
                )
            else:
                body = (
                    "API is optional for the local workflow. Open Settings → AI / API connection to paste an "
                    "OpenRouter or OpenAI-compatible key, choose a free model, and enable shared outbound calls. "
                    "Do not paste patient data into model prompts."
                )
        else:
            if api.get("configured"):
                body = (
                    f"{provider_label} 已为当前会话配置。打开设置可以开启共享出站调用、选择免费模型，"
                    "或更新 Base URL。Copilot 不会发送患者行；模型调用只使用计划文本和页面上下文。"
                )
            else:
                body = (
                    "API 对本地工作流是可选的。打开 设置 → AI / API 连接，可以粘贴 OpenRouter 或 OpenAI "
                    "兼容 Key、选择免费模型，并开启共享出站调用。不要把患者数据粘贴到模型 prompt。"
                )
        return _copilot_reply(study, body, lang, include_status=False), [
            {
                "id": "workflow_api_settings",
                "kind": "workflow",
                "label": "Open API settings" if lang == "en" else "打开 API 设置",
                "workflow": "api_settings",
            }
        ]

    if any(key in text_l for key in ("new study", "start over", "reset study", "重新开始", "新研究")):
        carried_messages = []
        messages = state.get("llm_messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if isinstance(last_message, Mapping) and str(last_message.get("role") or "").lower() == "user":
                carried_messages = [dict(last_message)]
        session = _start_new_copilot_study_session(state, lang, carry_messages=carried_messages)
        study = _ensure_copilot_study_state(state)
        workdir = Path(str(session.get("workdir") or "")).name or str(session.get("workdir") or "")
        body = (
            f"New study started with local workspace `{workdir}`. Describe the question or pick the first step."
            if lang == "en" else
            f"已新建研究，并创建本地工作目录 `{workdir}`。请描述研究问题，或从第一步开始选择。"
        )
        return _copilot_reply(study, body, lang), _copilot_guided_choice_actions(study, lang)

    if capability_overview_intent:
        return _copilot_capability_overview_reply(study, lang, state)

    if extract_step_intent:
        # Fixed "Prepare extraction plan" button → run the classic extract step
        # deterministically, never via the (timeout-prone) LLM router.
        return _copilot_submit_extract(state, lang)

    if review_step_intent:
        # Fixed "Run visual review" button → deterministic review step.
        return _copilot_submit_review(state, lang)

    waiting_for_custom_text = str(study.get("question_substep") or "") in {
        "custom",
        "custom_crossdb_signal",
        "custom_quality_target",
    }
    first_pass_goal = (
        not waiting_for_custom_text
        and not guided_choice_intent
        and not usage_help_intent
        and not step_by_step_intent
        and not cohort_step_intent
        and not feature_step_intent
        and not api_intent
        and not path_help_intent
        and str(study.get("step") or "question") == "question"
        and not str(study.get("question") or "").strip()
        and _copilot_first_pass_goal_allowed(prompt)
    )
    if first_pass_goal:
        return _copilot_capture_free_study_goal_first_pass(study, lang, prompt)
    if (
        not waiting_for_custom_text
        and _copilot_should_use_llm_route(
            prompt,
            usage_help_intent=usage_help_intent,
            step_by_step_intent=step_by_step_intent,
            cohort_step_intent=cohort_step_intent,
            api_intent=api_intent,
            path_help_intent=path_help_intent,
            guided_choice_intent=guided_choice_intent,
        )
    ):
        route = _copilot_route_with_llm(prompt, lang, state, study)
        if isinstance(route, Mapping):
            # A working model answered → it always wins.
            return _copilot_apply_llm_route(route, state, study, lang)
        # No route (timeout / no usable model — note the "hosted-default" provider
        # reports as configured even when unreachable, so we can't trust the
        # connection status). Deterministic keyword fallback keeps the natural-
        # language entry flowing offline; the LLM was already tried first above.
        kw_route = _copilot_keyword_route(prompt, lang)
        if kw_route is not None:
            return _copilot_apply_llm_route(kw_route, state, study, lang)
        # Only truly vague input reaches here → offer Retry + model guidance.
        if _copilot_route_transport_enabled(state):
            return _copilot_llm_route_unavailable_reply(study, lang, prompt=prompt)

    if not study.get("branch"):
        study["branch"] = _copilot_pick_branch(prompt)
    patient_count_requested = patient_count_intent
    if usage_help_intent:
        return _copilot_guided_usage_reply(study, lang, state)
    if cohort_step_intent:
        return _copilot_cohort_step_intro(study, lang, state)
    if feature_step_intent:
        return _copilot_feature_step_intro(study, lang, state)
    if step_by_step_intent:
        data_source_walkthrough_intent = (
            "data source step" in text_l
            or "prepared data path" in text_l
            or "prepared path" in text_l
            or "real data source" in text_l
            or "真实数据源步骤" in prompt
            or "prepared 数据路径" in prompt
            or "prepared 路径" in prompt
        )
        if data_source_walkthrough_intent:
            study["branch"] = _copilot_pick_branch(prompt)
            study["data_mode"] = "real"
            study["step"] = "data"
            study["question"] = ""
            study["cohort_configured"] = False
            study["concepts_configured"] = False
            study.pop("cohort_strategy", None)
            study.pop("selected_concepts", None)
            study.pop("cohort_substep", None)
            study.pop("data_source_choice", None)
            study.pop("data_source_status", None)
            state.pop("_copilot_data_source_choice", None)
            study["last_update"] = datetime.now().isoformat(timespec="seconds")
            body = (
                "Data source step opened. I will not choose a cohort, disease, endpoint, or modules yet.\n\n"
                "Pick what you have below: **Prepared data path**, **Module export folder**, or **Raw ICU files**. "
                "After this source choice, I will keep the rest of the workflow in this Copilot page."
                if lang == "en" else
                "已进入数据源步骤。我现在不会替你选择队列、疾病、endpoint 或模块。\n\n"
                "请在下面选择你手里的数据形式：**已有 prepared 路径**、**已有模块导出**，或 **只有 ICU 原始文件**。"
                "选择数据源后，后续步骤仍会留在当前 Copilot 页面完成。"
            )
            return body, _copilot_guided_choice_actions(study, lang)
        study["branch"] = _copilot_pick_branch(prompt)
        study["step"] = "question"
        study["question"] = ""
        study["cohort_configured"] = False
        study["concepts_configured"] = False
        study.pop("cohort_strategy", None)
        study.pop("selected_concepts", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        return (
            _copilot_step_by_step_intro(str(study.get("branch") or "predict"), lang),
            _copilot_guided_choice_actions(study, lang),
        )
    if next_step_help_intent:
        if real_data_intent or state.get("entry_mode") == "real" or str(study.get("data_mode") or "") == "real":
            study["data_mode"] = "real"
            study["step"] = "data"
            study.pop("data_source_choice", None)
            study.pop("data_source_status", None)
            state.pop("_copilot_data_source_choice", None)
            study["last_update"] = datetime.now().isoformat(timespec="seconds")
            state["_copilot_suppress_next_snapshot"] = True
            body = (
                "Stay in Copilot. I will not choose a disease, endpoint, cohort, or feature set for you yet.\n\n"
                "First, connect the data source in chat: tell me whether you already have a prepared/converted EasyICU data path, a module export folder, or raw ICU files that still need conversion. "
                "If you already have a prepared path, paste `set data path /path/to/prepared_data` here. After that I will ask for the research question, cohort scope, and feature modules one by one."
                if lang == "en" else
                "继续留在 Copilot 里。我现在不会替你选择疾病、endpoint、队列或特征集。\n\n"
                "第一步只确认数据源：告诉我你手里是已经 prepared/converted 的 EasyICU 数据路径、模块导出文件夹，还是还需要转换的 ICU 原始文件。"
                "如果已经有 prepared 路径，直接在这里发 `set data path /path/to/prepared_data`。之后我再逐个问研究问题、队列范围和特征模块。"
            )
            return body, _copilot_guided_choice_actions(study, lang)
        study["step"] = "question"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        state["_copilot_suppress_next_snapshot"] = True
        body = (
            "Stay in Copilot. I will not preselect the study for you. First tell me the broad clinical topic or endpoint you care about; then I will ask data source, cohort, modules, extraction, review, and Agent run choices one by one."
            if lang == "en" else
            "继续留在 Copilot 里。我不会先替你把研究选好。先告诉我你关心的大致临床主题或 endpoint；之后我再逐个问数据源、队列、模块、提取、审阅和 Agent 运行。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if patient_count_requested:
        concepts_before = set(str(item) for item in list(study.get("selected_concepts") or []))
        _copilot_apply_entities(study, prompt)
        study["data_mode"] = "real"
        concepts_after = set(str(item) for item in list(study.get("selected_concepts") or []))
        if concepts_after - concepts_before:
            study["concepts_configured"] = True
        _copilot_sync_step_to_configuration(study, allow_regress=True)
        body = _copilot_configured_reply(
            study,
            lang,
            patient_count_requested=True,
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if any(term in text_l for term in ("question type: outcome prediction", "研究类型：结局预测")):
        study["branch"] = "predict"
        study["question_kind"] = "outcome_model"
        study["question_substep"] = "endpoint"
        study["step"] = "question"
        study["question"] = ""
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Good. This will be an **ICU outcome model**. I still will not choose the endpoint for you. Pick the endpoint below, or type your own."
            if lang == "en" else
            "好，这会是一条 **ICU 结局建模** 流程。我仍然不会替你选择 endpoint。请在下面选择结局，或输入你自己的 endpoint。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("question type: treatment exposure", "研究类型：治疗暴露")):
        study["branch"] = "predict"
        study["question_kind"] = "treatment_exposure"
        study["question_substep"] = "exposure"
        study["step"] = "question"
        study["question"] = ""
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Good. This will be a **treatment exposure** study. First choose the exposure family; then I will ask for the endpoint and cohort."
            if lang == "en" else
            "好，这会是一条 **治疗暴露** 研究。先选择暴露类型；之后我再问 endpoint 和队列。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("question type: cross-database comparison", "研究类型：跨库比较")):
        study["branch"] = "crossdb"
        study["question_kind"] = "crossdb"
        study["question_substep"] = "crossdb_signal"
        study["step"] = "question"
        study["question"] = ""
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Good. This will be a **cross-database comparison**. Pick the signal to compare; then we will choose databases, cohort, modules, and review gates."
            if lang == "en" else
            "好，这会是一条 **跨库比较** 流程。先选择要比较的信号；之后再选择数据库、队列、模块和审阅闸门。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("question type: data quality audit", "研究类型：数据质量审计")):
        study["branch"] = "quality"
        study["question_kind"] = "quality_audit"
        study["question_substep"] = "quality_target"
        study["step"] = "question"
        study["question"] = ""
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Good. This will be a **data-quality audit**. Pick the audit target; then we will bind data source, cohort, modules, and review output."
            if lang == "en" else
            "好，这会是一条 **数据质量审计** 流程。先选择审计目标；之后再绑定数据源、队列、模块和审阅输出。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("describe my own research question", "自己描述研究问题")):
        study["question_substep"] = "custom"
        study["step"] = "question"
        study["question"] = ""
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Type the study question in one sentence. I will keep your wording, turn it into workflow state, and then ask for data source and cohort."
            if lang == "en" else
            "请用一句话输入你的研究问题。我会保留你的表达，把它转成工作流状态，然后继续问数据源和队列。"
        )
        return body, []

    if any(term in text_l for term in ("describe my own endpoint", "自己描述 endpoint")):
        study["question_substep"] = "custom_endpoint"
        study["step"] = "question"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Type the endpoint in one short phrase, for example `ICU length of stay` or `AKI within 48h`. I will not choose it for you."
            if lang == "en" else
            "请用一个短语输入 endpoint，例如 `ICU 住院时长` 或 `48 小时内 AKI`。我不会替你选择。"
        )
        return body, []

    if any(term in text_l for term in ("describe my own exposure", "自己描述暴露")):
        study["question_substep"] = "custom_exposure"
        study["step"] = "question"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Type the exposure or treatment pattern in one short phrase. After that I will ask for the endpoint."
            if lang == "en" else
            "请用一个短语输入暴露或治疗模式。之后我再问 endpoint。"
        )
        return body, []

    if any(term in text_l for term in ("describe my cross-database signal", "自己描述跨库信号")):
        study["question_substep"] = "custom_crossdb_signal"
        study["step"] = "question"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Type the cross-database signal you want to compare. I will keep it as the study frame and move to data source choices."
            if lang == "en" else
            "请描述你想跨库比较的信号。我会把它作为研究框架，然后进入数据源选择。"
        )
        return body, []

    if any(term in text_l for term in ("describe my audit target", "自己描述审计目标")):
        study["question_substep"] = "custom_quality_target"
        study["step"] = "question"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Type the quality-audit target. I will keep it as the audit frame and move to data source choices."
            if lang == "en" else
            "请描述你想审计的质量目标。我会把它作为审计框架，然后进入数据源选择。"
        )
        return body, []

    if any(term in text_l for term in ("exposure: vasopressor support", "暴露：升压药支持")):
        study["exposure"] = "vasopressor support"
        study["question_substep"] = "endpoint"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Exposure set to **vasopressor support**. Now choose the endpoint for this exposure study."
            if lang == "en" else
            "暴露已设为 **升压药支持**。现在选择这条暴露研究的 endpoint。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("exposure: mechanical ventilation", "暴露：机械通气")):
        study["exposure"] = "mechanical ventilation"
        study["question_substep"] = "endpoint"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Exposure set to **mechanical ventilation**. Now choose the endpoint for this exposure study."
            if lang == "en" else
            "暴露已设为 **机械通气**。现在选择这条暴露研究的 endpoint。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("exposure: renal replacement therapy", "暴露：肾脏替代治疗")):
        study["exposure"] = "renal replacement therapy"
        study["question_substep"] = "endpoint"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Exposure set to **renal replacement therapy**. Now choose the endpoint for this exposure study."
            if lang == "en" else
            "暴露已设为 **肾脏替代治疗**。现在选择这条暴露研究的 endpoint。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("cross-database signal: outcome model", "跨库信号：结局模型")):
        study["question"] = (
            "Does a prespecified ICU outcome signal replicate across the selected ICU databases?"
            if lang == "en" else
            "预设 ICU 结局信号能否在选择的 ICU 数据库之间复现？"
        )
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Study frame recorded: **{study['question']}**\n\nNext choose how the data enters this cross-database workflow."
            if lang == "en" else
            f"研究框架已记录：**{study['question']}**\n\n下一步选择这条跨库流程的数据入口。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("cross-database signal: treatment pattern", "跨库信号：治疗模式")):
        study["question"] = (
            "Do early ICU treatment patterns differ across databases after matching cohort and concept definitions?"
            if lang == "en" else
            "在对齐队列和概念定义后，早期 ICU 治疗模式是否在数据库之间存在差异？"
        )
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Study frame recorded: **{study['question']}**\n\nNext choose how the data enters this cross-database workflow."
            if lang == "en" else
            f"研究框架已记录：**{study['question']}**\n\n下一步选择这条跨库流程的数据入口。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("cross-database signal: concept availability", "跨库信号：概念可用性")):
        study["question"] = (
            "Which ICU concepts are available and comparable across the selected databases?"
            if lang == "en" else
            "哪些 ICU 概念在选择的数据库之间可用且可比？"
        )
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Study frame recorded: **{study['question']}**\n\nNext choose how the data enters this cross-database workflow."
            if lang == "en" else
            f"研究框架已记录：**{study['question']}**\n\n下一步选择这条跨库流程的数据入口。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("quality target: concept coverage", "质量目标：概念覆盖率")):
        study["question"] = (
            "Which selected ICU concepts have enough coverage and comparability to support downstream analysis?"
            if lang == "en" else
            "选择的 ICU 概念是否有足够覆盖率和可比性来支持后续分析？"
        )
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Audit frame recorded: **{study['question']}**\n\nNext choose data source."
            if lang == "en" else
            f"审计框架已记录：**{study['question']}**\n\n下一步选择数据源。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("quality target: mapping and units", "质量目标：映射和单位")):
        study["question"] = (
            "Are the selected concept mappings, units, and normalization rules safe enough for review?"
            if lang == "en" else
            "选择的概念映射、单位和归一化规则是否足够安全，能够进入审阅？"
        )
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Audit frame recorded: **{study['question']}**\n\nNext choose data source."
            if lang == "en" else
            f"审计框架已记录：**{study['question']}**\n\n下一步选择数据源。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if any(term in text_l for term in ("quality target: cohort attrition", "质量目标：队列流失")):
        study["question"] = (
            "Where does the cohort denominator change across source validation, cohort filters, feature availability, and export?"
            if lang == "en" else
            "队列分母在数据源校验、队列过滤、特征可用性和导出过程中在哪里发生变化？"
        )
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Audit frame recorded: **{study['question']}**\n\nNext choose data source."
            if lang == "en" else
            f"审计框架已记录：**{study['question']}**\n\n下一步选择数据源。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if _copilot_prepared_data_choice_requested(prompt):
        study = _copilot_set_data_source_choice(state, "prepared_path")
        body = (
            "Good. I opened a **Prepared data path** field below the conversation. Paste the prepared/converted folder there and save it; I will stay in Copilot and mark it pending validation."
            if lang == "en" else
            "好。我已经在对话下方打开 **prepared 数据路径** 输入框。把 prepared/converted 文件夹路径填进去并保存；我会继续留在 Copilot，并标记为待验证。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if _copilot_module_export_choice_requested(prompt):
        study = _copilot_set_data_source_choice(state, "module_export")
        body = (
            "Good. I opened a **Module export folder** field below the conversation. Save the EasyICU export folder here; then we will choose cohort scope and modules in this same Copilot flow."
            if lang == "en" else
            "好。我已经在对话下方打开 **模块导出文件夹** 输入框。把 EasyICU 导出目录保存到这里；之后仍在同一个 Copilot 流程里选择队列和模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if _copilot_raw_files_choice_requested(prompt):
        study = _copilot_set_data_source_choice(state, "raw_files")
        body = (
            "Raw files need a database type and root folder before conversion. I opened the raw-files field below; choose the database, paste the root folder, and I will keep the conversion plan in Copilot."
            if lang == "en" else
            "原始文件需要先确认数据库类型和根目录。我已经在下方打开原始文件输入框；选择数据库、粘贴根目录后，我会把转换计划继续留在 Copilot。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if _copilot_disease_cohort_choice_requested(prompt):
        study["step"] = "cohort"
        study["cohort_substep"] = "disease"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Choose the disease/diagnosis cohort filter in chat. This is only a cohort option; I will not lock features until the next step."
            if lang == "en" else
            "在聊天里选择疾病/诊断队列过滤。这里只配置队列，不会提前锁定特征。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if _copilot_age_los_choice_requested(prompt):
        study["step"] = "cohort"
        study["cohort_substep"] = "age_los"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Choose age / ICU length-of-stay constraints in chat. We can keep this broad for the first pass."
            if lang == "en" else
            "在聊天里选择年龄 / ICU LOS 限制。第一轮可以先保持宽松。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if _copilot_current_reviewed_cohort_requested(prompt):
        state["cohort_filter"] = _copilot_default_cohort_filter()
        state["cohort_enabled"] = False
        _copilot_confirm_classic_step2(state)
        study["cohort_strategy"] = "reviewed_current"
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Cohort source set to the current reviewed cohort. Next, choose feature modules in this chat."
            if lang == "en" else
            "队列来源已设为当前审阅队列。下一步在聊天里选择特征模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if _copilot_no_disease_filter_requested(prompt):
        cohort_filter = dict(state.get("cohort_filter") or _copilot_default_cohort_filter())
        cohort_filter["disease_cohort"] = "none"
        cohort_filter["has_sepsis"] = None
        cohort_filter["icd_query"] = ""
        cohort_filter["icd_include_query"] = ""
        cohort_filter["icd_exclude_query"] = ""
        state["cohort_filter"] = cohort_filter
        state["cohort_enabled"] = False
        _copilot_confirm_classic_step2(state)
        study["cohort_strategy"] = "eligible"
        study["cohort_filters"] = []
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Cohort kept broad: no disease filter. Next, choose feature modules."
            if lang == "en" else
            "队列保持宽松：不加疾病过滤。下一步选择特征模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if "filter cohort to sepsis-3" in text_l or "队列过滤为 sepsis-3" in text_l:
        cohort_filter = dict(state.get("cohort_filter") or _copilot_default_cohort_filter())
        cohort_filter["disease_cohort"] = "sepsis"
        cohort_filter["has_sepsis"] = True
        state["cohort_filter"] = cohort_filter
        state["cohort_enabled"] = True
        _copilot_confirm_classic_step2(state)
        study["cohort_filters"] = ["sepsis-3"]
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Cohort filter set to Sepsis-3. Next, choose feature modules."
            if lang == "en" else
            "队列过滤已设为 Sepsis-3。下一步选择特征模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if "filter cohort to aki" in text_l or "队列过滤为 aki" in text_l or "队列过滤为 rrt" in text_l:
        cohort_filter = dict(state.get("cohort_filter") or _copilot_default_cohort_filter())
        cohort_filter["disease_cohort"] = "aki"
        cohort_filter["has_sepsis"] = None
        state["cohort_filter"] = cohort_filter
        state["cohort_enabled"] = True
        _copilot_confirm_classic_step2(state)
        study["cohort_filters"] = ["aki/rrt"]
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Cohort filter set to AKI / RRT. Next, choose feature modules."
            if lang == "en" else
            "队列过滤已设为 AKI / RRT。下一步选择特征模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if "use adult icu stays" in text_l or "使用成人 icu stay" in text_l:
        cohort_filter = dict(state.get("cohort_filter") or _copilot_default_cohort_filter())
        cohort_filter["age_min"] = 18
        state["cohort_filter"] = cohort_filter
        state["cohort_enabled"] = True
        _copilot_confirm_classic_step2(state)
        filters = list(study.get("cohort_filters") or [])
        if "adult ICU stays" not in filters:
            filters.append("adult ICU stays")
        study["cohort_filters"] = filters
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Cohort constraint set to adult ICU stays. Next, choose feature modules."
            if lang == "en" else
            "队列限制已设为成人 ICU stay。下一步选择特征模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if "require icu los at least 24 hours" in text_l or "icu los 至少 24" in text_l:
        cohort_filter = dict(state.get("cohort_filter") or _copilot_default_cohort_filter())
        cohort_filter["los_min"] = 24
        cohort_filter["los_max"] = None
        state["cohort_filter"] = cohort_filter
        state["cohort_enabled"] = True
        _copilot_confirm_classic_step2(state)
        filters = list(study.get("cohort_filters") or [])
        if "ICU LOS >= 24h" not in filters:
            filters.append("ICU LOS >= 24h")
        study["cohort_filters"] = filters
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Cohort constraint set to ICU LOS >= 24h. Next, choose feature modules."
            if lang == "en" else
            "队列限制已设为 ICU LOS ≥ 24h。下一步选择特征模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    if "no age or los restriction" in text_l or "不加年龄或 icu los" in text_l or "不加年龄或 los" in text_l:
        cohort_filter = dict(state.get("cohort_filter") or _copilot_default_cohort_filter())
        cohort_filter["age_min"] = None
        cohort_filter["age_max"] = None
        cohort_filter["los_min"] = None
        cohort_filter["los_max"] = None
        state["cohort_filter"] = cohort_filter
        _copilot_confirm_classic_step2(state)
        filters = [
            str(item)
            for item in list(study.get("cohort_filters") or [])
            if str(item) not in {"adult ICU stays", "ICU LOS >= 24h"}
        ]
        study["cohort_filters"] = filters
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            "Age and ICU LOS constraints kept broad. Next, choose feature modules."
            if lang == "en" else
            "年龄和 ICU LOS 暂时保持宽松。下一步选择特征模块。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    module_pack = _copilot_module_pack_from_prompt(prompt)
    if module_pack is not None:
        existing = [
            str(item)
            for item in list(study.get("selected_concepts") or [])
            if str(item).strip()
        ]
        for concept in module_pack:
            if concept not in existing:
                existing.append(concept)
        state["selected_concepts"] = existing
        _copilot_confirm_classic_step3(state)
        study["selected_concepts"] = existing
        study["modules"] = _copilot_modules_for_concepts(existing)
        study["concepts_configured"] = True
        study["step"] = "extract"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        labels = _copilot_concept_label_list(study, limit=8)
        label_text = ", ".join(labels) if lang == "en" else "、".join(labels)
        body = (
            f"Feature modules selected in chat: **{label_text}**. Next I can prepare the extraction plan here, or explain the evidence gate."
            if lang == "en" else
            f"已在聊天里选择特征模块：**{label_text}**。下一步我可以在这里准备提取计划，或解释证据闸门。"
        )
        return body, _copilot_guided_choice_actions(study, lang)

    concepts_before = set(str(item) for item in list(study.get("selected_concepts") or []))
    _copilot_apply_entities(study, prompt)
    if patient_count_requested:
        study["data_mode"] = "real"
    concepts_after = set(str(item) for item in list(study.get("selected_concepts") or []))
    concepts_requested = bool(concepts_after - concepts_before)
    if _copilot_confirm_suggested_concepts_requested(prompt):
        suggested = [
            str(item)
            for item in list(study.get("suggested_concepts") or [])
            if str(item).strip()
        ]
        if suggested:
            existing = [
                str(item)
                for item in list(study.get("selected_concepts") or [])
                if str(item).strip()
            ]
            for concept in suggested:
                if concept not in existing:
                    existing.append(concept)
            state["selected_concepts"] = existing
            _copilot_confirm_classic_step3(state)
            study["selected_concepts"] = existing
            study["modules"] = _copilot_modules_for_concepts(existing)
            study["concepts_configured"] = True
            _copilot_sync_step_to_configuration(study, allow_regress=True)
            concept_labels = _copilot_concept_label_list(study)
            label_text = ", ".join(concept_labels) if lang == "en" else "、".join(concept_labels)
            body = (
                f"Modules confirmed in chat: **{label_text}**. Next, confirm the data source and cohort scope."
                if lang == "en" else
                f"已在聊天中确认模块：**{label_text}**。下一步确认数据源和队列范围。"
            )
            return _copilot_reply(study, body, lang, include_status=False), _copilot_study_actions(study, lang)
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    typed_data_path = _copilot_extract_data_path_from_text(prompt)

    if typed_data_path:
        _copilot_set_real_data_path_in_chat(state, typed_data_path)
        study = _ensure_copilot_study_state(state)
        study["step"] = "cohort"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        if lang == "en":
            body = (
                f"Saved the real-data path in chat: `{typed_data_path}`.\n\n"
                "I will keep you here in Copilot. This path is marked **pending validation**. Next, choose the cohort scope below; before analysis, we will validate the path or use an existing module export."
            )
        else:
            body = (
                f"已在聊天中记录真实数据路径：`{typed_data_path}`。\n\n"
                "我会继续留在 Copilot 里。这个路径目前标记为 **待验证**。下一步先在下面选择队列范围；分析前我们再验证路径，或使用已有模块导出。"
            )
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    custom_question_substep = str(study.get("question_substep") or "")
    if study.get("step") == "question" and custom_question_substep in {
        "custom",
        "custom_crossdb_signal",
        "custom_quality_target",
    }:
        study["question"] = prompt
        if custom_question_substep == "custom_crossdb_signal":
            study["branch"] = "crossdb"
        elif custom_question_substep == "custom_quality_target":
            study["branch"] = "quality"
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Recorded your study frame: **{study['question']}**\n\nNext choose the data source in this chat."
            if lang == "en" else
            f"已记录你的研究框架：**{study['question']}**\n\n下一步在当前聊天里选择数据源。"
        )
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    if study.get("step") == "question" and custom_question_substep == "custom_exposure":
        study["exposure"] = prompt
        study["question_kind"] = "treatment_exposure"
        study["question_substep"] = "endpoint"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Exposure recorded: **{study['exposure']}**. Now choose the endpoint for this exposure study."
            if lang == "en" else
            f"已记录暴露：**{study['exposure']}**。现在选择这条暴露研究的 endpoint。"
        )
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    if study.get("step") == "question" and custom_question_substep == "custom_endpoint":
        study["outcome"] = prompt
        study["question"] = _copilot_frame_question(study, lang)
        study["step"] = "data"
        study.pop("question_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = (
            f"Endpoint recorded. I framed the study as: **{study['question']}**\n\nNext choose how data enters this chat workflow."
            if lang == "en" else
            f"endpoint 已记录。我把研究问题框定为：**{study['question']}**\n\n下一步在当前聊天里选择数据入口。"
        )
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    if path_help_intent:
        study = _copilot_set_data_source_choice(state, "prepared_path")
        body = _copilot_real_data_path_reply(state, lang)
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    if full_cohort_intent:
        state["cohort_filter"] = _copilot_default_cohort_filter()
        state["cohort_enabled"] = False
        _copilot_confirm_classic_step2(state)
        study["data_mode"] = "real"
        study["cohort_strategy"] = "eligible"
        study["cohort_configured"] = True
        study["step"] = "concepts"
        study.pop("cohort_substep", None)
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        body = _copilot_configured_reply(
            study,
            lang,
            patient_count_requested=False,
        )
        if study.get("data_mode") == "real" and not _copilot_real_source_ready(state):
            body = (
                "Cohort scope selected: **eligible real-data cohort**. This is a scope, not a materialized row count. Next, choose feature modules in this chat; we will bind the prepared data path/module export before analysis."
                if lang == "en" else
                "队列范围已选择：**真实数据合格队列**。这是队列范围，不是已经生成的行数。下一步在聊天里选择特征模块；分析前再绑定 prepared 数据路径或模块导出。"
            )
            if concepts_requested:
                concept_labels = _copilot_concept_label_list(study)
                if concept_labels:
                    label_text = ", ".join(concept_labels) if lang == "en" else "、".join(concept_labels)
                    body += (
                        f" Feature set: **{label_text}**."
                        if lang == "en" else
                        f" 特征集：**{label_text}**。"
                    )
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    if _copilot_is_loosen_filter_request(prompt) and _copilot_cohort_is_empty(study):
        _copilot_loosen_filters(study)
        body = (
            f"Filters loosened. The demo cohort is back to **{int(study.get('patient_n') or 10)} stays**; I can now re-match and continue to concepts or review."
            if lang == "en" else
            f"已放宽过滤条件。演示队列恢复为 **{int(study.get('patient_n') or 10)} 例 stay**；现在可以重新匹配并继续到变量或审阅。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if _copilot_is_strict_filter_request(prompt):
        _copilot_apply_strict_no_data_filter(study)
        body = (
            "No patients match those filters. `Sepsis-3 + age >= 80` is empty in this demo set/export, so I stopped at the cohort card instead of opening review. Loosen one constraint and I will re-match."
            if lang == "en" else
            "没有患者匹配这些过滤条件。`Sepsis-3 + 年龄 ≥ 80` 在这个演示数据/导出中为空，所以我停在队列卡片，没有打开审阅。放宽一个条件后我会重新匹配。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if any(key in text_l for key in ("back", "go back", "undo", "change", "edit", "previous", "回退", "修改", "上一步")):
        current_idx = COPILOT_STEP_INDEX.get(str(study.get("step") or "question"), 0)
        editable = ["question", "data", "cohort", "concepts"]
        target = "question"
        for step in reversed(editable):
            if COPILOT_STEP_INDEX[step] < current_idx:
                target = step
                break
        study["step"] = target
        if target == "question":
            study["question"] = ""
            study.pop("selected_concepts", None)
            study["cohort_configured"] = False
            study["concepts_configured"] = False
            study.pop("cohort_strategy", None)
            study.pop("cohort_substep", None)
            study["cohort_filters"] = []
        elif target == "data":
            study.pop("selected_concepts", None)
            study["cohort_configured"] = False
            study["concepts_configured"] = False
            study.pop("cohort_strategy", None)
            study.pop("cohort_substep", None)
            study["cohort_filters"] = []
        elif target == "cohort":
            study.pop("selected_concepts", None)
            study["concepts_configured"] = False
        body = (
            f"Rewound to **{dict(COPILOT_STUDY_STEPS)[target]}**. Downstream choices will be refreshed from here."
            if lang == "en" else
            f"已回退到 **{target}**。后续选择会从这里重新刷新。"
        )
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    if study.get("step") == "extract" and any(
        key in text_l for key in (
            "prepare extraction",
            "extraction plan",
            "run extraction",
            "extract data",
            "准备提取",
            "提取计划",
            "提取数据",
        )
    ):
        study["extraction_configured"] = True
        study["step"] = "review"
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        db_label = _copilot_database_label(study.get("database") or state.get("database") or "miiv", lang)
        concepts = _copilot_concept_label_list(study, limit=8)
        concept_text = ", ".join(concepts) if lang == "en" else "、".join(concepts)
        cohort_text = _copilot_stage_detail(study, "cohort", lang)
        body = (
            f"Extraction plan assembled in Copilot: **{db_label}**, cohort **{cohort_text}**, modules `{concept_text}`.\n\n"
            "This mirrors classic Data Extraction Step 1-3. The actual heavy extraction/validation remains evidence-gated; next we review the prepared frame or hand it to Agent setup."
            if lang == "en" else
            f"已在 Copilot 中组装提取计划：**{db_label}**，队列 **{cohort_text}**，模块 `{concept_text}`。\n\n"
            "这对应经典 Data Extraction 的 Step 1-3。真正耗时的提取/验证仍然保持证据闸门；下一步审阅准备好的数据帧，或交给 Agent 配置。"
        )
        return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)

    if any(key in text_l for key in ("why", "explain", "reason", "为什么", "解释")):
        step = str(study.get("step") or "question")
        why = str(config["why"].get(step, config["why"]["question"]))
        body = (
            f"Why this step: {why}"
            if lang == "en" else
            f"为什么做这一步：{why}"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if patient_count_requested:
        _copilot_sync_step_to_configuration(study, allow_regress=True)
        body = _copilot_configured_reply(
            study,
            lang,
            patient_count_requested=True,
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if concepts_requested and study.get("step") != "question":
        _copilot_sync_step_to_configuration(study)
        body = _copilot_configured_reply(
            study,
            lang,
            patient_count_requested=False,
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if any(key in text_l for key in ("run the whole", "whole demo", "autopilot", "just do it", "do it for me", "帮我跑", "自动跑", "一键")):
        study["data_mode"] = "real"
        study["step"] = "data"
        study["draft_signed"] = False
        state.pop("_copilot_autopilot_ready", None)
        body = (
            "Copilot does not start a demo cohort here. I will keep this as a guided real-data workflow: first choose how data enters the chat, then we configure cohort, modules, extraction, review, and Agent run step by step."
            if lang == "en" else
            "Copilot 这里不启动演示队列。我会保持真实数据优先的引导流程：先选择数据如何进入聊天，再逐步配置队列、模块、提取、审阅和 Agent 运行。"
        )
        return _copilot_reply(study, body, lang), _copilot_guided_choice_actions(study, lang)

    if study.get("step") == "question" and branch == "predict" and not _copilot_endpoint_pinned(prompt):
        study["step"] = "question"
        body = (
            "Good direction. I will not choose the endpoint for you. First pick the outcome: **in-hospital mortality**, **ICU mortality**, **28-day mortality**, **AKI/RRT**, **length of stay**, or another endpoint."
            if lang == "en" else
            "方向可以。我不会替你选择 endpoint。先选结局：**院内死亡**、**ICU 死亡**、**28 天死亡**、**AKI/RRT**、**住院时长**，或你自己的 endpoint。"
        )
        return _copilot_reply(study, body, lang), []

    if _copilot_endpoint_pinned(prompt) and study.get("step") == "question":
        if text_l.startswith(("endpoint:", "endpoint：")):
            study["concepts_configured"] = False
            study.pop("selected_concepts", None)
            study["modules"] = COPILOT_DEFAULT_MODULES[:]
        study["question"] = _copilot_frame_question(study, lang)
        study["step"] = "data"
        study.pop("question_substep", None)
        body = (
            f"Got it. I framed the study as: **{study['question']}**\n\nNext, choose how real data enters this Copilot workflow."
            if lang == "en" else
            f"收到。我把研究问题框定为：**{study['question']}**\n\n下一步选择真实数据入口，仍然在当前 Copilot 页面完成。"
        )
        return _copilot_reply(study, body, lang), _copilot_guided_choice_actions(study, lang)

    if real_data_intent:
        study = _copilot_set_data_source_choice(state, "prepared_path")
        body = (
            "Real-data mode selected. Stay here: choose the source type below and save the path in the inline field. Use Classic workspace only if you explicitly want the full validation screen."
            if lang == "en" else
            "已选择真实数据模式。继续留在这里：在下面选择来源类型，并在内嵌输入框里保存路径。只有明确要走完整校验屏时再用经典工作区。"
        )
        return _copilot_reply(study, body, lang), _copilot_guided_choice_actions(study, lang)

    if any(key in text_l for key in ("demo", "use demo", "演示", "示例")) and study.get("step") in {"data", "question"}:
        study["data_mode"] = "real"
        study["step"] = "data"
        study.pop("data_source_choice", None)
        study.pop("data_source_status", None)
        body = (
            "Copilot mode is real-data first. Choose the local source type below; if you only want to inspect examples, Classic workspace still has its separate demo surfaces."
            if lang == "en" else
            "Copilot 模式是真实数据优先。请在下面选择本地数据来源类型；如果只是想看示例，经典工作区仍保留独立演示入口。"
        )
        return _copilot_reply(study, body, lang), _copilot_guided_choice_actions(study, lang)

    if study.get("step") == "question":
        study["question"] = _copilot_frame_question(study, lang)
        study["step"] = "data"
        body = (
            f"I framed your study as: **{study['question']}**\n\nNext, choose the real data source in this Copilot page."
            if lang == "en" else
            f"我把你的研究问题框定为：**{study['question']}**\n\n下一步在当前 Copilot 页面选择真实数据源。"
        )
        return _copilot_reply(study, body, lang), _copilot_guided_choice_actions(study, lang)

    next_step = _copilot_advance_step(study)
    if next_step == "cohort":
        cohort_label = _copilot_cohort_label(study, lang)
        body = (
            f"Cohort ready: **{cohort_label}** with the branch `{config['chip']}`. For real work, connect prepared data and use the eligible cohort; explicit small counts are only for smoke tests."
            if lang == "en" else
            f"队列已准备：**{cohort_label}**，研究方向为 `{config['chip']}`。正式分析请接入 prepared 真实数据并使用合格队列；小样本数只适合 smoke test。"
        )
    elif next_step == "concepts":
        body = (
            "I preselected the modules this question needs: demographics, vitals, labs, SOFA/SOFA-2, Sepsis-3, and outcomes. Coverage will be audited before analysis."
            if lang == "en" else
            "我已预选该问题需要的模块：人口学、生命体征、实验室、SOFA/SOFA-2、Sepsis-3 和结局。分析前会先做覆盖率审计。"
        )
    elif next_step == "extract":
        body = (
            "Extraction is ready. I can assemble the extraction plan in this chat, then move to review or Agent setup without leaving Copilot."
            if lang == "en" else
            "提取已准备好。我可以在当前聊天中组装提取计划，然后继续到审阅或 Agent 配置，不需要离开 Copilot。"
        )
    elif next_step == "review":
        body = (
            "Review is ready. Open Patient Review to inspect data tables, time series, patient overview, and quality flags."
            if lang == "en" else
            "审阅已准备好。可以打开患者审阅页查看表格、时间序列、患者概览和质量标记。"
        )
    elif next_step == "analysis":
        body = (
            "Analysis run assembled. It remains evidence-bound: deterministic steps must complete and produce traceable artifacts before any draft claim unlocks."
            if lang == "en" else
            "分析运行已组装。它仍然保持证据绑定：确定性步骤必须完成并产生可追溯产物，草稿论断才会解锁。"
        )
    else:
        study["draft_signed"] = False
        body = (
            "The draft gate is reached but locked. Review the workspace or hand the framed question to Research Agent; no claim is written until evidence checks pass and you sign off."
            if lang == "en" else
            "已到达草稿闸门，但仍处于锁定状态。请审阅工作区或把问题交给 Research Agent；证据检查通过并人工确认前不会写出论断。"
        )
    return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)


def _append_copilot_prompt_action(
    prompt: str,
    lang: str,
    *,
    display_prompt: str | None = None,
) -> None:
    """Append an in-chat choice and resolve it without navigating away."""
    state = st.session_state
    route_prompt = (prompt or "").strip()
    visible_prompt = (display_prompt or route_prompt).strip()
    if not route_prompt and visible_prompt:
        route_prompt = visible_prompt
    if not route_prompt:
        return
    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list):
        messages.append(_copilot_user_message(route_prompt, visible_prompt))
    guided_reply = _handle_copilot_guided_prompt(route_prompt, lang, state)
    if guided_reply is None:
        guided_reply = (
            _local_copilot_fallback_reply(route_prompt, lang),
            _copilot_guided_choice_actions(_ensure_copilot_study_state(state), lang),
        )
    reply_content, guided_actions = guided_reply
    suppress_snapshot = bool(state.pop("_copilot_suppress_next_snapshot", False))
    message: dict[str, object] = {
        "role": "assistant",
        "content": reply_content,
        "actions": guided_actions,
    }
    if not suppress_snapshot:
        message["workflow_snapshot"] = _copilot_workflow_snapshot(state, lang)
    if isinstance(messages, list):
        messages.append(message)
    _request_copilot_scroll_to_latest(state)
    state["_active_main_page"] = "assistant"
    state["_assistant_notice"] = (
        "Choice recorded in Copilot."
        if lang == "en" else
        "已在 Copilot 中记录选择。"
    )


_COPILOT_ACTION_ICON_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("agent", "代理"), "smart_toy"),  # before "age": "set up agent run" contains "age"
    (("eligible",), "groups"),
    (("disease", "diagnosis", "sepsis", "脓毒", "疾病"), "clinical_notes"),
    (("age", "los", "年龄", "住院时长"), "calendar_month"),
    (("reviewed", "current", "审阅队列"), "history"),
    (("extraction", "extract", "提取"), "database"),
    (("review", "审阅", "patient review"), "visibility"),
    (("gate", "why", "解释", "为什么"), "help"),
    (("classic", "经典"), "grid_view"),
    (("loosen", "放宽", "broad"), "tune"),
    (("data path", "数据路径", "set data", "设置路径"), "folder"),
    (("api", "设置"), "settings"),
    (("module", "模块"), "category"),
    (("draft", "草稿", "sign"), "description"),
)


def _render_nav_actions(actions: list[dict[str, object]], key_prefix: str) -> None:
    """Render in-app navigation and preset actions as buttons.

    Rendered as a vertical stack of full-width "option rows" (polish gd-opt
    look) inside a keyed container so the copilot CSS can card-style them.
    """
    if not actions:
        return
    with st.container(key=f"{key_prefix}_navactions"):
        for action_idx, action in enumerate(actions):
            # Optional gd-opt-style description: a second markdown paragraph in
            # the button label (Streamlit renders it as a separate <p>, styled
            # smaller/muted by the copilot CSS). The clean label is still what
            # the click handler shows as the user message.
            desc = str(action.get("desc") or "").strip()
            button_label = f"{action['label']}\n\n{desc}" if desc else action["label"]
            if st.button(
                button_label,
                key=f"{key_prefix}_{action_idx}_{action['id']}",
                use_container_width=True,
            ):
                if action.get("kind") == "preset":
                    st.session_state["_assistant_preset_request"] = dict(action.get("payload") or {})
                    st.session_state["_scroll_to_tab"] = str(action.get("scroll_to") or "tutorial")
                elif action.get("kind") == "copilot_prompt":
                    label = str(action.get("label") or "").strip()
                    _append_copilot_prompt_action(
                        str(action.get("prompt") or label),
                        str(st.session_state.get("language") or "en"),
                        display_prompt=label or None,
                    )
                elif action.get("kind") == "workflow":
                    _apply_chat_workflow_action(str(action.get("workflow") or ""))
                elif action.get("kind") == "agent_handoff":
                    _prepare_research_agent_handoff_from_ai(st.session_state)
                else:
                    st.session_state["_scroll_to_tab"] = str(action["id"])
                st.rerun()


def _start_guided_demo_from_prompt(state: MutableMapping[str, object], prompt: str, lang: str) -> None:
    """Start the full Copilot page from a concrete local guided-study prompt."""
    reply = _handle_copilot_guided_prompt(prompt, str(lang), state)
    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list) and reply is not None:
        content, actions = reply
        messages.append({"role": "user", "content": prompt})
        messages.append({
            "role": "assistant",
            "content": content,
            "actions": actions,
            "workflow_snapshot": _copilot_workflow_snapshot(state, lang),
        })
    state["_active_main_page"] = "assistant"
    state["_assistant_notice"] = (
        "Guided Research Copilot demo is ready in the chat."
        if lang == "en" else
        "引导式研究 Copilot 演示已在聊天中准备好。"
    )


def _sign_off_copilot_draft_gate(state: MutableMapping[str, object], lang: str) -> bool:
    """Record a local human sign-off for the guided Copilot draft gate."""
    study = _ensure_copilot_study_state(state)
    active_idx = COPILOT_STEP_INDEX.get(str(study.get("step") or "question"), 0)
    if active_idx < COPILOT_STEP_INDEX["draft"]:
        state["_assistant_notice"] = (
            "Review and analysis evidence must be assembled before the draft gate can be signed off."
            if lang == "en" else
            "需要先组装审阅与分析证据，才能确认草稿闸门。"
        )
        return False

    signed_at = datetime.now().isoformat(timespec="seconds")
    question = str(study.get("question") or _copilot_frame_question(study, lang)).strip()
    study["question"] = question
    study["step"] = "draft"
    study["draft_signed"] = True
    study["draft_signed_at"] = signed_at
    state["_copilot_guided_study"] = study
    _remember_copilot_guided_study_resume(state, study)
    state["_copilot_draft_unlocked"] = True
    state["_copilot_draft_signed_at"] = signed_at
    state["_copilot_last_question"] = question

    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list):
        user_text = "Review & sign off" if lang == "en" else "审阅并确认"
        assistant_text = (
            "Signed off. The guided draft gate is now unlocked for this local study; claims still need to be carried into Research Agent artifacts before manuscript export."
            if lang == "en" else
            "已确认。这个本地 guided study 的草稿闸门已解锁；正式导出手稿前，论断仍需要进入 Research Agent 产物链。"
        )
        messages.append({"role": "user", "content": user_text})
        messages.append({
            "role": "assistant",
            "content": assistant_text,
            "actions": _copilot_study_actions(study, lang),
            "workflow_snapshot": _copilot_workflow_snapshot(state, lang),
        })

    state["_active_main_page"] = "assistant"
    state["_assistant_notice"] = (
        "Guided draft gate signed off locally."
        if lang == "en" else
        "引导式草稿闸门已在本地确认。"
    )
    return True


def _open_copilot_guided_draft_preview(state: MutableMapping[str, object], lang: str) -> bool:
    """Route a signed guided study into the Research Agent Summary draft preview."""
    study = _ensure_copilot_study_state(state)
    if not study.get("draft_signed"):
        state["_active_main_page"] = "assistant"
        state["_assistant_notice"] = (
            "Review & sign off the guided draft gate before opening the draft preview."
            if lang == "en" else
            "请先审阅并确认引导式草稿闸门，再打开草稿预览。"
        )
        return False

    _apply_copilot_study_to_workspace(state)
    question = str(study.get("question") or _copilot_frame_question(study, lang)).strip()
    branch = str(study.get("branch") or "predict")
    branch_label = str(COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"]).get("chip") or branch)
    signed_at = str(study.get("draft_signed_at") or datetime.now().isoformat(timespec="seconds"))
    is_en = lang == "en"

    try:
        from easyicu.webapp.agent_workbench import _demo_state

        workbench = _demo_state(lang)
    except Exception:
        workbench = {
            "steps": [],
            "evidence": [],
            "summary_outputs": [],
            "audit": {"counts": {"errors": 0, "warnings": 0, "info": 0}, "gates": []},
            "is_demo": True,
        }

    workbench.update({
        "run_id": f"guided_copilot_{branch}_draft",
        "title": "Guided draft preview" if is_en else "引导式草稿预览",
        "research_question": question,
        "subtitle": (
            f"{branch_label} · local sign-off · no generated metrics"
            if is_en else
            f"{branch_label} · 本地签字 · 不生成指标"
        ),
        "subtitle_short": "local sign-off · preview slots only" if is_en else "本地签字 · 仅预览槽位",
        "status": "review",
        "status_step": "guided draft unlocked" if is_en else "引导式草稿已解锁",
        "source_label": "Guided Copilot draft preview" if is_en else "引导式 Copilot 草稿预览",
        "is_demo": True,
        "allow_summary_demo": True,
        "summary_demo_source": "guided_copilot_signed_draft",
        "demo_notice": (
            "Guided Copilot sign-off unlocked this draft preview. No cohort metrics, files, or manuscript export are fabricated; bind a real manifest before exporting manuscript text."
            if is_en else
            "引导式 Copilot 签字已解锁这个草稿预览。这里不会编造队列指标、文件或手稿导出；正式导出手稿前仍需绑定真实 manifest。"
        ),
        "audit": {
            "counts": {"errors": 0, "warnings": 0, "info": 0},
            "gates": [
                {"label": "guided_study_assembled", "ok": True},
                {"label": "local_signoff_recorded", "ok": True},
                {"label": "manifest_export_still_required", "ok": True},
            ],
            "review_decision": {
                "decision": "guided_signoff",
                "note": "Signed off from guided Research Copilot." if is_en else "已从引导式 Research Copilot 签字。",
                "updated_at": signed_at,
                "source": "easyicu_guided_copilot",
            },
        },
        "review_gate_actions": [
            {
                "label": "Guided draft unlocked" if is_en else "引导式草稿已解锁",
                "state": "ready",
                "detail": "Preview only; export still requires a real manifest." if is_en else "仅预览；导出仍需真实 manifest。",
            },
            {
                "label": "Open real Agent run" if is_en else "打开真实 Agent 运行",
                "state": "ready",
                "detail": "Use Setup to bind artifacts before manuscript export." if is_en else "用配置页绑定产物后再导出手稿。",
            },
        ],
        "review_decisions": [
            {
                "label": "Saved: guided_signoff" if is_en else "已保存: guided_signoff",
                "state": "selected",
                "detail": signed_at,
            },
        ],
    })
    workbench["steps"] = [
        {
            "label": "Question framed" if is_en else "研究问题已成型",
            "sub": branch_label,
            "status": "ok",
            "step_id": "guided_question",
        },
        {
            "label": "Cohort and concepts staged" if is_en else "队列与概念已暂存",
            "sub": f"{int(study.get('patient_n') or 10)} stays · {len(study.get('modules') or COPILOT_DEFAULT_MODULES)} modules",
            "status": "ok",
            "step_id": "guided_workspace",
        },
        {
            "label": "Review artifacts opened" if is_en else "审阅产物已打开",
            "sub": COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"]).get("review_target", "quick_viz"),
            "status": "ok",
            "step_id": "guided_review",
        },
        {
            "label": "Local sign-off recorded" if is_en else "本地签字已记录",
            "sub": signed_at,
            "status": "ok",
            "step_id": "guided_signoff",
        },
        {
            "label": "Manifest export gate" if is_en else "Manifest 导出关口",
            "sub": "real run required for manuscript export" if is_en else "正式手稿导出需要真实运行",
            "status": "pending",
            "step_id": "guided_manifest_gate",
        },
    ]
    workbench["summary_outputs"] = [
        {
            "kind": "plan" if is_en else "计划",
            "title": "Guided study plan" if is_en else "引导式研究计划",
            "sub": "chat-framed; no file written" if is_en else "聊天中成型；未写入文件",
            "badge": "signed" if is_en else "已签字",
        },
        {
            "kind": "review" if is_en else "复核",
            "title": "Evidence gate sign-off" if is_en else "证据闸门签字",
            "sub": "local guided state" if is_en else "本地引导式状态",
            "badge": "local" if is_en else "本地",
        },
        {
            "kind": "draft" if is_en else "草稿",
            "title": "Draft preview slot" if is_en else "草稿预览槽位",
            "sub": "manifest-bound export remains locked" if is_en else "manifest 绑定导出仍锁定",
            "badge": "preview" if is_en else "预览",
        },
        {
            "kind": "audit" if is_en else "审计",
            "title": "Real manifest slot" if is_en else "真实 manifest 槽位",
            "sub": "open Agent Setup to generate artifacts" if is_en else "打开 Agent 配置生成产物",
            "badge": "required" if is_en else "必需",
        },
    ]
    state["_agent_workbench"] = workbench
    clear_agent_continuation_state(state)
    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "summary"
    state["_scroll_to_top"] = True
    state["_assistant_notice"] = (
        "Guided draft preview opened in Research Agent Summary."
        if is_en else
        "引导式草稿预览已在 Research Agent Summary 打开。"
    )
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state["_sidebar_ai_open"] = False
    state.pop("_ai_pending_question", None)
    return True


def _completed_run_preview_state(lang: str) -> dict[str, object]:
    """Return a demo-safe Research Agent completed-run shell for the dock action."""
    is_en = lang == "en"
    try:
        from easyicu.webapp.agent_workbench import _demo_state

        workbench = _demo_state(lang)
    except Exception:
        workbench = {
            "steps": [],
            "evidence": [],
            "summary_outputs": [],
            "audit": {"counts": {"errors": 0, "warnings": 0, "info": 0}, "gates": []},
            "is_demo": True,
        }

    workbench.update({
        "run_id": "dock_completed_run_preview",
        "title": "Completed run preview" if is_en else "已完成运行预览",
        "research_question": (
            "Among Sepsis-3 patients, do first-24h bedside features predict in-hospital mortality?"
            if is_en else
            "在 Sepsis-3 患者中，前 24 小时床旁特征能否预测院内死亡？"
        ),
        "subtitle": (
            "Summary gate preview · no cohort metrics generated"
            if is_en else
            "Summary gate 预览 · 未生成队列指标"
        ),
        "subtitle_short": "completed-run shell · preview only" if is_en else "已完成运行壳 · 仅预览",
        "status": "preview",
        "status_step": "completed run shell" if is_en else "已完成运行壳",
        "source_label": "Dock completed-run preview" if is_en else "Dock 已完成运行预览",
        "is_demo": True,
        "allow_summary_demo": True,
        "summary_demo_source": "dock_completed_run_preview",
        "demo_notice": (
            "This completed-run preview mirrors the polish(2) dock action without fabricating cohort metrics, files, or manuscript export. Bind a real manifest to inspect real artifacts."
            if is_en else
            "这个已完成运行预览对齐 polish(2) 的 dock 动作，但不会编造队列指标、文件或手稿导出。请绑定真实 manifest 后查看真实产物。"
        ),
        "audit": {
            "counts": {"errors": 0, "warnings": 0, "info": 0},
            "gates": [
                {"label": "completed_run_shell_visible", "ok": True},
                {"label": "no_fabricated_metrics", "ok": True},
                {"label": "manifest_export_still_required", "ok": True},
            ],
        },
        "review_gate_actions": [
            {
                "label": "Summary gate visible" if is_en else "Summary gate 可见",
                "state": "ready",
                "detail": "Preview only; artifact export still requires a real manifest." if is_en else "仅预览；产物导出仍需真实 manifest。",
            },
            {
                "label": "Bind real manifest" if is_en else "绑定真实 manifest",
                "state": "ready",
                "detail": "Use Research Agent Setup or a saved run to replace this shell." if is_en else "用 Research Agent 配置或已保存运行替换这个壳。",
            },
        ],
    })
    workbench["steps"] = [
        {
            "label": "Manifest scan" if is_en else "Manifest 扫描",
            "sub": "no local completed manifest selected" if is_en else "尚未选择本地已完成 manifest",
            "status": "ok",
            "step_id": "dock_manifest_scan",
        },
        {
            "label": "Cohort summary slot" if is_en else "队列摘要槽位",
            "sub": "real run required" if is_en else "需要真实运行",
            "status": "ok",
            "step_id": "dock_cohort_slot",
        },
        {
            "label": "Evidence table slot" if is_en else "证据表槽位",
            "sub": "artifact placeholder only" if is_en else "仅产物占位",
            "status": "ok",
            "step_id": "dock_table_slot",
        },
        {
            "label": "Reviewer gate" if is_en else "审核关口",
            "sub": "previewed from dock" if is_en else "由 dock 预览",
            "status": "ok",
            "step_id": "dock_review_gate",
        },
        {
            "label": "Manifest-bound export" if is_en else "Manifest 绑定导出",
            "sub": "locked until real run" if is_en else "真实运行前锁定",
            "status": "pending",
            "step_id": "dock_export_gate",
        },
    ]
    workbench["summary_outputs"] = [
        {
            "kind": "summary" if is_en else "摘要",
            "title": "Summary gate" if is_en else "Summary gate",
            "sub": "visible without fabricating artifacts" if is_en else "可见，但不伪造产物",
            "badge": "preview" if is_en else "预览",
        },
        {
            "kind": "table" if is_en else "表格",
            "title": "Table artifact slot" if is_en else "表格产物槽位",
            "sub": "real manifest required" if is_en else "需要真实 manifest",
            "badge": "required" if is_en else "必需",
        },
        {
            "kind": "audit" if is_en else "审计",
            "title": "Audit ledger slot" if is_en else "审计账本槽位",
            "sub": "no file written in preview" if is_en else "预览不写文件",
            "badge": "locked" if is_en else "锁定",
        },
    ]
    return workbench


def _open_research_agent_completed_run_from_dock(state: MutableMapping[str, object], lang: str) -> bool:
    """Open the latest real Research Agent run, or a clearly-labelled preview shell."""
    is_en = lang == "en"
    opened_real = False
    latest_run_label = ""

    try:
        from easyicu.webapp.agent_workbench import build_workbench_state_from_manifest
        from easyicu.webapp.research_agent import (
            _default_research_agent_workdir,
            _load_run_manifest,
            _scan_research_agent_runs,
        )

        raw_workdir = str(state.get("research_agent_workdir") or "").strip()
        workdir = Path(raw_workdir).expanduser() if raw_workdir else Path(_default_research_agent_workdir()).expanduser()
        for row in _scan_research_agent_runs(workdir, limit=25):
            if str(row.get("status") or "").lower() in {"partial", "running"}:
                continue
            run_dir = row.get("run_dir")
            run_path = run_dir if isinstance(run_dir, Path) else Path(str(run_dir or ""))
            if not run_path.exists() or not run_path.is_dir():
                continue
            manifest, _manifest_path, partial = _load_run_manifest(run_path)
            if not manifest or partial:
                continue
            workbench = build_workbench_state_from_manifest(run_path, manifest, lang=lang, partial=partial)
            if not workbench.get("steps"):
                continue
            state["_agent_workbench"] = workbench
            state["_agent_workbench_source_run_dir"] = str(run_path)
            state["_agent_workbench_is_active_selection"] = True
            latest_run_label = str(manifest.get("run_id") or run_path.name)
            if latest_run_label:
                state["research_agent_last_run_id"] = latest_run_label
            opened_real = True
            break
    except Exception:
        opened_real = False

    if not opened_real:
        state["_agent_workbench"] = _completed_run_preview_state(lang)
        state["_agent_workbench_is_active_selection"] = False
        state.pop("_agent_workbench_source_run_dir", None)

    clear_agent_continuation_state(state)
    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "summary"
    state["_scroll_to_top"] = True
    state["_assistant_notice"] = (
        (f"Opened the latest completed Research Agent run in Summary: {latest_run_label}." if latest_run_label else "Opened the latest completed Research Agent run in Summary.")
        if opened_real and is_en else
        (f"已在 Summary 打开最新完成的 Research Agent 运行：{latest_run_label}。" if latest_run_label else "已在 Summary 打开最新完成的 Research Agent 运行。")
        if opened_real else
        "No local completed manifest was found, so a completed-run preview shell is open in Summary."
        if is_en else
        "未找到本地已完成 manifest，已在 Summary 打开已完成运行预览壳。"
    )
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state["_sidebar_ai_open"] = False
    state.pop("_ai_pending_question", None)
    return True


def _apply_chat_workflow_action(workflow: str) -> None:
    """Apply a chat-first command to the live Streamlit workspace."""
    state = st.session_state
    workflow = (workflow or "").strip()
    lang = state.get("language", "en")
    if workflow == "demo_extraction":
        _seed_demo_context_from_chat(state)
        state["_active_main_page"] = "extract"
        state["_assistant_notice"] = (
            "Demo extraction is ready. Confirm the data source, then continue through Cohort and Concepts."
            if lang == "en" else
            "演示提取已准备好。先确认数据源，再继续完成队列和变量步骤。"
        )
    elif workflow == "study_extract":
        study = _ensure_copilot_study_state(state)
        if _copilot_study_should_use_real_source(study, state):
            state["entry_mode"] = "real"
            state["use_mock_data"] = False
            if state.get("database") == "mock":
                state["database"] = "miiv"
            _apply_copilot_study_to_workspace(state)
            state["_active_main_page"] = "extract"
            state["_assistant_notice"] = (
                "Research Copilot opened Real Data extraction. Choose the local database path, then Validate Data Path or Convert & Setup before Agent analysis."
                if lang == "en" else
                "Research Copilot 已打开真实数据提取。请选择本地数据库路径，然后先验证路径或转换并设置，再交给 Agent 分析。"
            )
            return
        _seed_demo_context_from_chat(state)
        _apply_copilot_study_to_workspace(state)
        state["_active_main_page"] = "extract"
        state["_assistant_notice"] = (
            "Research Copilot prepared the classic extraction flow from your guided study."
            if lang == "en" else
            "研究 Copilot 已根据引导式研究准备好经典提取流程。"
        )
    elif workflow == "demo_review":
        _seed_demo_context_from_chat(state)
        state["selected_concepts"] = state.get("selected_concepts") or _copilot_selected_concepts_for_study(state)
        state["_preview_requested"] = True
        mock_params = state.get("mock_params") if isinstance(state.get("mock_params"), dict) else {}
        state["_preview_n"] = int(mock_params.get("n_patients") or state.get("demo_mode_patients") or 10)
        state["_active_main_page"] = "quick_viz"
        state["_assistant_notice"] = (
            "Demo review workspace is loading in Patient Review."
            if lang == "en" else
            "演示审阅工作区正在患者审阅页加载。"
        )
    elif workflow == "study_review":
        _seed_demo_context_from_chat(state)
        _apply_copilot_study_to_workspace(state)
        study = _ensure_copilot_study_state(state)
        if _copilot_cohort_is_empty(study):
            state["_preview_requested"] = False
            state["_active_main_page"] = "assistant"
            state["_assistant_notice"] = (
                "No patients match the strict cohort filters. Loosen filters before opening review."
                if lang == "en" else
                "严格队列过滤条件没有匹配患者。请先放宽过滤条件，再打开审阅。"
            )
        else:
            state["_preview_requested"] = True
            branch = str(study.get("branch") or "predict")
            target = str(COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"]).get("review_target") or "quick_viz")
            if branch == "crossdb":
                _seed_copilot_crossdb_demo_workspace(state)
            state["_active_main_page"] = target
            state["_assistant_notice"] = (
                "Research Copilot loaded the review workspace from your guided study."
                if lang == "en" else
                "研究 Copilot 已根据引导式研究加载审阅工作区。"
            )
    elif workflow == "study_strict_filters":
        study = _ensure_copilot_study_state(state)
        if not study.get("branch"):
            study["branch"] = "predict"
        study["question"] = str(study.get("question") or _copilot_frame_question(study, str(lang))).strip()
        _copilot_apply_strict_no_data_filter(study)
        state["_copilot_guided_study"] = study
        state["_active_main_page"] = "assistant"
        state["_assistant_notice"] = (
            "Strict cohort filters produced no matching demo patients. Loosen filters to continue."
            if lang == "en" else
            "严格队列过滤条件没有匹配演示患者。请放宽过滤条件后继续。"
        )
    elif workflow == "study_loosen_filters":
        study = _ensure_copilot_study_state(state)
        _copilot_loosen_filters(study)
        state["_copilot_guided_study"] = study
        state["_active_main_page"] = "assistant"
        state["_assistant_notice"] = (
            "Filters loosened. The guided cohort is ready again."
            if lang == "en" else
            "过滤条件已放宽，引导式队列已恢复可用。"
        )
    elif workflow == "crossdb_demo":
        _seed_crossdb_demo_workspace(state)
        state["_active_main_page"] = "cross_db"
        state["_assistant_notice"] = (
            "Demo Cross-DB benchmark is loaded with the summary and availability matrix first."
            if lang == "en" else
            "演示 Cross-DB benchmark 已加载，先显示摘要与可用性矩阵。"
        )
    elif workflow == "cohort_run":
        state["_active_main_page"] = "cohort"
        state["_eu_topbar_run_request"] = {
            "page": "cohort",
            "requested_at": "copilot_dock",
        }
        state["_assistant_notice"] = (
            "Cohort Statistics will re-run through the shared workspace refresh."
            if lang == "en" else
            "Cohort Statistics 会通过共享工作区刷新重新运行。"
        )
    elif workflow == "real_extraction":
        state["entry_mode"] = "real"
        state["use_mock_data"] = False
        if state.get("database") == "mock":
            state["database"] = "miiv"
        state["_active_main_page"] = "extract"
        state["_assistant_notice"] = (
            "Real Data extraction is open. Choose the database and local path. Raw folders must pass Validate Data Path -> Convert & Setup; Agent analysis uses the prepared path or module export."
            if lang == "en" else
            "真实数据提取已打开。请选择数据库和本地路径；原始目录需要先通过 Validate Data Path -> Convert & Setup，Agent 分析使用 prepared 路径或模块导出。"
        )
    elif workflow == "real_data_chat_setup":
        state["entry_mode"] = "real"
        state["use_mock_data"] = False
        if state.get("database") == "mock":
            state["database"] = "miiv"
        study = _copilot_set_data_source_choice(state, "prepared_path")
        state["_copilot_guided_study"] = study
        user_text = "Set data path in chat" if lang == "en" else "在聊天中设置路径"
        assistant_text = _copilot_real_data_path_reply(state, str(lang))
        messages = state.setdefault("llm_messages", [])
        if isinstance(messages, list):
            messages.append({"role": "user", "content": user_text})
            messages.append({
                "role": "assistant",
                "content": assistant_text,
                "actions": [],
                "workflow_snapshot": _copilot_workflow_snapshot(state, str(lang)),
            })
        state["_active_main_page"] = "assistant"
        state["_assistant_notice"] = (
            "Stay in Copilot: the prepared data path field is open below the conversation."
            if lang == "en" else
            "继续留在 Copilot：prepared 数据路径输入框已在对话下方打开。"
        )
    elif workflow == "api_settings":
        state["_active_main_page"] = "settings"
        state["_scroll_to_top"] = True
        state["_assistant_notice"] = (
            "Open AI / API connection settings. API keys stay in this browser session and are not written to the repository."
            if lang == "en" else
            "已打开 AI / API 连接设置。API Key 只保存在当前浏览器会话，不会写入仓库。"
        )
    elif workflow in {"study_depth_extract", "study_depth_review", "study_depth_full"}:
        study = _ensure_copilot_study_state(state)
        depth = workflow.rsplit("_", 1)[-1]
        study["depth"] = _copilot_engine.normalize_depth(depth)
        # Don't leave the active step past the new (possibly shallower) goal.
        study["step"] = _copilot_engine.clamp_step_to_goal(study["depth"], str(study.get("step") or "question"))
        study["last_update"] = datetime.now().isoformat(timespec="seconds")
        state["_copilot_guided_study"] = study
        goal = _copilot_engine.copilot_goal_step(study["depth"])
        state["_assistant_notice"] = (
            f"Study depth set — this run will finish at the {goal} step."
            if lang == "en" else
            f"已设置研究深度——本次将在 {goal} 步停下。"
        )
    elif workflow == "study_take_further":
        study = _ensure_copilot_study_state(state)
        study["depth"] = _copilot_engine.copilot_bump_depth(study.get("depth"))
        _copilot_advance_step(study)
        state["_copilot_guided_study"] = study
        # Reconcile the segmented control before its next instantiation (writing
        # the widget key directly after instantiation is forbidden by Streamlit).
        state["_copilot_depth_seg_pending"] = study["depth"]
        _apply_copilot_study_to_workspace(state)
        state["_assistant_notice"] = (
            f"Taking it further — depth is now {study['depth']}."
            if lang == "en" else
            f"继续深入——当前深度为 {study['depth']}。"
        )
    elif workflow == "study_agent":
        _apply_copilot_study_to_workspace(state)
        _prepare_research_agent_handoff_from_ai(state)
    elif workflow == "agent_idea_exploration":
        _prepare_research_agent_idea_handoff_from_ai(state)
    elif workflow == "guided_demo":
        _start_guided_demo_from_prompt(
            state,
            "Run the whole demo for me, then stop at the evidence gate.",
            str(lang),
        )
    elif workflow == "guided_crossdb_demo":
        _start_guided_demo_from_prompt(
            state,
            "Compare across ICU databases and run the whole demo for me, then stop at the evidence gate.",
            str(lang),
        )
    elif workflow == "study_signoff":
        _sign_off_copilot_draft_gate(state, str(lang))
    elif workflow == "study_draft":
        _open_copilot_guided_draft_preview(state, str(lang))
    elif workflow == "agent_completed_run":
        _open_research_agent_completed_run_from_dock(state, str(lang))
    else:
        state["_active_main_page"] = "assistant"
    state["_scroll_to_top"] = True
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state.pop("_ai_pending_question", None)


def _compose_agent_messages(prompt: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Assemble system/context messages and return tool activity log."""
    tool_events: list[dict[str, str]] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "Answer format requirements:\n"
                "- For web-usage questions, lead with the exact page, step, or action in EasyICU.\n"
                "- After the direct answer, give the next 1-3 concrete UI steps the user should take now.\n"
                "- Mention files/functions only when the user explicitly asks about implementation.\n"
                "- If local code context was used, mention the relevant file/function names.\n"
                "- If external medical sources were used, include a `Sources` section with direct URLs.\n"
                "- Do not reveal hidden chain-of-thought or private deliberation."
            ),
        },
        {"role": "system", "content": _build_workflow_status_context(st.session_state.get("language", "en"))},
    ]

    project_context = _build_project_context(prompt) if _is_code_question(prompt) else ""
    if project_context:
        tool_events.append({
            "tool": "local_code_search",
            "status": "ok",
            "detail": "Built local EasyICU code snapshot",
        })
        messages.append({"role": "system", "content": project_context})

    if _is_external_lookup_question(prompt):
        pubmed_context, pubmed_events = _build_pubmed_context(prompt)
        tool_events.extend(pubmed_events)
        if pubmed_context:
            messages.append({"role": "system", "content": pubmed_context})

    messages.extend(st.session_state.llm_messages)
    return messages, tool_events


def _stream_text(stream, placeholder):
    """Render streaming text manually."""
    chunks = []
    for token in _token_generator(stream):
        chunks.append(token)
        visible_text = _strip_llm_reasoning("".join(chunks))
        placeholder.markdown((visible_text + "▌") if visible_text else "…")
    text = _strip_llm_reasoning("".join(chunks))
    placeholder.markdown(text)
    return text


def _needs_api_key(provider: str) -> bool:
    return _shared_needs_api_key(provider)


def _is_configured() -> bool:
    """Return True when the provider is ready to use."""
    provider = st.session_state.get("llm_provider", public_default_provider_key())
    _, default_url, _default_model, _needs_key, _desc_en, _desc_zh = public_provider_defaults(provider)
    if _needs_api_key(provider) and not st.session_state.get("llm_api_key", "").strip():
        return False
    return bool((st.session_state.get("llm_base_url", "") or default_url).strip())


def _get_client():
    """Build and return an OpenAI-compatible client, or *None* on error."""
    try:
        from openai import OpenAI
        import httpx
    except ImportError:
        return None

    provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
    _display, default_url, _default_model, _needs_key, _desc_en, _desc_zh = public_provider_defaults(provider)
    api_key = st.session_state.get("llm_api_key", "").strip()
    base_url = st.session_state.get("llm_base_url", "").strip() or default_url or None

    if not api_key and not _needs_api_key(provider):
        api_key = os.getenv("EASYICU_HOSTED_CLIENT_TOKEN", "easyicu-hosted")

    if not api_key:
        return None

    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
    }
    default_headers = _provider_default_headers(provider)
    if default_headers:
        client_kwargs["default_headers"] = default_headers

    # Ignore system proxy env vars by default to avoid optional SOCKS dependency
    # failures inside embedded Streamlit sessions.
    http_client = httpx.Client(
        timeout=120.0,
        trust_env=False,
        follow_redirects=True,
    )
    return OpenAI(**client_kwargs, http_client=http_client)


def _provider_default_headers(provider: str) -> dict[str, str] | None:
    """Return provider-specific headers shared by foreground/background calls."""
    if provider != "openrouter":
        return None
    return {
        "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
        "X-Title": "EasyICU web copilot",
    }


def _current_provider_choice() -> str:
    return str(st.session_state.get("llm_provider", public_default_provider_key()))


def _external_llm_ready(lang: str) -> bool:
    provider = _current_provider_choice()
    enforce_external_llm_opt_in(provider, language=lang)
    return _is_configured()


# ---------------------------------------------------------------------------
# UI — Settings panel (rendered inside an expander in the sidebar)
# ---------------------------------------------------------------------------

def render_llm_settings(
    *,
    expanded: bool = False,
    show_status_card: bool = True,
    controls_only: bool = False,
    show_enable_toggle: bool = True,
    open_sidebar_on_enable: bool = True,
):
    """Render LLM configuration controls in the sidebar or settings popover."""
    _init_chat_state()
    _sync_llm_toggle_before_render()
    lang = st.session_state.get("language", "en")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .easyicu-ai-sidebar-card {
            display: grid;
            grid-template-columns: 2.2rem 1fr;
            gap: 0.62rem;
            align-items: center;
            padding: 0.72rem 0.82rem;
            border-radius: 14px;
            border: 1px solid #cfe0f3;
            background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
            box-shadow: 0 9px 22px rgba(15, 23, 42, 0.05);
            margin: 0.24rem 0 0.18rem;
        }
        [data-testid="stSidebar"] .easyicu-ai-sidebar-avatar {
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            background: linear-gradient(135deg, #f59e0b 0%, #f97316 68%, #2563eb 100%);
            box-shadow: 0 9px 18px rgba(249, 115, 22, 0.2);
            font-size: 1rem;
        }
        [data-testid="stSidebar"] .easyicu-ai-sidebar-title {
            color: #0b1f44;
            font-size: 0.9rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        [data-testid="stSidebar"] .easyicu-ai-sidebar-subtitle {
            color: #64748b;
            font-size: 0.72rem;
            line-height: 1.38;
            margin-top: 0.12rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if show_status_card:
        enabled = bool(st.session_state.llm_enabled)
        status_title = "Research Copilot" if lang == "en" else "研究 Copilot"
        status_subtitle = (
            "Embedded guidance for the current EasyICU workflow."
            if lang == "en" else
            "嵌入当前 EasyICU 工作流的页面感知式助手。"
        )
        if enabled and _is_configured():
            status_subtitle = (
                "Embedded chat is ready in this panel."
                if lang == "en" else
                "嵌入式聊天已在此面板中就绪。"
            )
        st.markdown(
            f"""
            <div class="easyicu-ai-sidebar-card">
                <div class="easyicu-ai-sidebar-avatar">AI</div>
                <div>
                    <div class="easyicu-ai-sidebar-title">{status_title}</div>
                    <div class="easyicu-ai-sidebar-subtitle">{status_subtitle}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    label = "AI settings" if lang == "en" else "AI 设置"
    if controls_only:
        _render_llm_settings_controls(
            lang,
            show_enable_toggle=show_enable_toggle,
            open_sidebar_on_enable=open_sidebar_on_enable,
        )
    else:
        with st.expander(label, expanded=expanded):
            _render_llm_settings_controls(
                lang,
                show_enable_toggle=show_enable_toggle,
                open_sidebar_on_enable=open_sidebar_on_enable,
            )


def _render_llm_settings_controls(
    lang: str,
    *,
    show_enable_toggle: bool = True,
    open_sidebar_on_enable: bool = True,
) -> None:
    previous_enabled = bool(st.session_state.llm_enabled)
    if show_enable_toggle:
        enabled = st.toggle(
            "Enable Research Copilot" if lang == "en" else "启用研究 Copilot",
            value=previous_enabled,
            key="_llm_toggle",
        )
        st.session_state.llm_enabled = bool(enabled)
        st.session_state["_floating_ai_open"] = False
        if enabled and open_sidebar_on_enable:
            st.session_state["_sidebar_ai_open"] = True
    else:
        enabled = previous_enabled
        status = (
            "Outbound model calls allowed" if enabled else "Outbound model calls disabled"
        ) if lang == "en" else (
            "已允许模型端调用" if enabled else "模型端调用已关闭"
        )
        if enabled:
            detail = (
                "Shared outbound calls are on; Research Agent still shows a per-run disclosure gate."
                if lang == "en" else
                "共享模型调用已开启；Research Agent 仍会显示单次运行披露关口。"
            )
        else:
            detail = (
                "Provider details can be prepared here; calls still require the shared outbound toggle."
                if lang == "en" else
                "可以先准备服务商配置；是否允许调用由上方共享开关控制。"
            )
        klass = "on" if enabled else "off"
        st.markdown(
            f"""
            <div class="eu-llm-settings-status {klass}">
              <span></span>
              <div><b>{status}</b><p>{detail}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if not enabled and show_enable_toggle:
        hint = ("Enable this to use the AI panel in the main workspace."
                if lang == "en"
                else "启用后可在主工作区使用 AI 面板。")
        st.caption(hint)
        return

    # Provider
    provider_keys = public_provider_keys()
    current_provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
    idx = provider_keys.index(current_provider) if current_provider in provider_keys else 0
    provider = st.selectbox(
        "Provider" if lang == "en" else "服务商",
        options=provider_keys,
        index=idx,
        format_func=lambda k: public_provider_defaults(k)[0],
        key="_llm_provider_sel",
    )
    provider_changed = provider != current_provider
    st.session_state.llm_provider = provider

    p_info = public_provider_defaults(provider)
    if provider_changed:
        _, default_url, default_model, _, _, _ = p_info
        st.session_state.llm_api_key = ""
        st.session_state.llm_base_url = default_url
        st.session_state.llm_model = default_model
        st.session_state["_llm_api_key_inp"] = ""
        st.session_state["_llm_base_url_inp"] = default_url
        st.session_state["_llm_model_inp"] = default_model
        st.session_state.llm_configured = False
        st.rerun()
    desc = p_info[4] if lang == "en" else p_info[5]
    st.caption(desc)

    needs_key = _needs_api_key(provider)

    if needs_key:
        api_key = st.text_input(
            "API Key",
            value=st.session_state.llm_api_key,
            type="password",
            key="_llm_api_key_inp",
            placeholder="sk-...",
        )
        st.session_state.llm_api_key = api_key

    _, default_url, default_model, _, _, _ = p_info
    base_url = st.text_input(
        "API Base URL",
        value=st.session_state.llm_base_url or default_url,
        key="_llm_base_url_inp",
        help="Leave default for standard providers" if lang == "en"
             else "标准服务商保持默认即可",
    )
    st.session_state.llm_base_url = base_url

    model = st.text_input(
        "Model" if lang == "en" else "模型名称",
        value=st.session_state.llm_model or default_model,
        key="_llm_model_inp",
        placeholder=default_model or "model-name",
    )
    st.session_state.llm_model = model

    configured = _is_configured()
    if configured and enabled:
        st.success("Ready" if lang == 'en' else "已就绪")
        st.session_state.llm_configured = True
    elif configured:
        st.info(
            "Provider configured. Turn on outbound model calls above when you want to use it."
            if lang == "en" else
            "服务商配置已就绪；需要使用时请打开上方模型端调用开关。"
        )
        st.session_state.llm_configured = True
    else:
        st.warning("Enter API Key to enable chat"
                   if lang == 'en' else "请输入 API Key")
        st.session_state.llm_configured = False

    st.caption(
        "API keys stay in this browser session only and are never saved by EasyICU."
        if lang == "en" else
        "API Key 只保存在当前浏览器会话中，EasyICU 不会写入本地文件。"
    )


def _build_chat_export_text() -> str:
    """Serialize chat history to markdown."""
    lines = ["# EasyICU AI Chat Export", ""]
    for msg in st.session_state.get("llm_messages", []):
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"## {role}")
        lines.append(msg.get("content", "").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# UI — Chat tab (rendered as a main-area tab)
# ---------------------------------------------------------------------------

def render_chat_tab():
    """Render the full chat interface inside a main tab."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")

    # ---- Guard: not enabled --------------------------------------------------
    if not st.session_state.llm_enabled:
        st.info(
            (
                "Research Copilot is disabled. Enable it in the sidebar Copilot settings."
                if lang == "en" else
                "研究 Copilot 当前已关闭。请在侧边栏 Copilot 设置中开启。"
            )
        )
        # Show a brief intro even when disabled
        _render_intro(lang)
        return

    # ---- Guard: not configured -----------------------------------------------
    if not _is_configured():
        st.warning(
            (
                "Please configure your API Key in the sidebar Copilot settings first."
                if lang == "en" else
                "请先在侧边栏 Copilot 设置中设置 API Key。"
            )
        )
        return

    # ---- Guard: openai package missing ---------------------------------------
    try:
        import openai as _openai_mod  # noqa: F401
    except ImportError:
        st.error(
            (
                "The `openai` Python package is required. "
                "Install it with: `pip install openai`"
                if lang == "en" else
                "需要安装 `openai` Python 包。请执行: `pip install openai`"
            )
        )
        return

    # ---- Header --------------------------------------------------------------
    provider_name = public_provider_defaults(st.session_state.get("llm_provider", public_default_provider_key()))[0]
    model_name = (st.session_state.get("llm_model", "").strip()
                  or public_provider_defaults(st.session_state.get("llm_provider", public_default_provider_key()))[2]
                  or "—")
    st.markdown(
        "##### " + ("Chat with Research Copilot" if lang == "en"
                     else "与研究 Copilot 对话")
    )
    st.caption(f"**{provider_name}** · `{model_name}`")
    st.caption(
        "Guided mode: web workflow first, then code and evidence when needed"
        if lang == "en" else
        "引导模式：优先教你如何使用 Web 工作流，再按需补充代码与证据"
    )

    with st.expander(("What can I ask?" if lang == "en" else "我可以问什么？"),
                      expanded=False):
        _render_tips(lang)

    if st.session_state.get("llm_last_tool_events"):
        with st.expander(("Last tool activity" if lang == "en" else "上次工具调用"), expanded=False):
            for event in st.session_state.llm_last_tool_events:
                status = "ok" if event.get("status") == "ok" else "needs attention"
                st.markdown(f"`{event.get('tool', 'tool')}` · {status} — {event.get('detail', '')}")

    # ---- Render message history -----------------------------------------------
    history_container = st.container(height=680, border=True)
    with history_container:
        for msg_idx, msg in enumerate(st.session_state.llm_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("actions"):
                    _render_nav_actions(msg["actions"], key_prefix=f"_llm_action_{msg_idx}")

        pending_prompt = st.session_state.pop("_ai_pending_question", None)
        if pending_prompt:
            _submit_prompt(pending_prompt, lang, history_container, key_prefix="_llm_tab")

    # ---- Chat input -----------------------------------------------------------
    placeholder = ("Ask a question about EasyICU …"
                    if lang == "en" else "输入关于 EasyICU 的问题 …")
    if prompt := st.chat_input(placeholder, key="_llm_chat_input"):
        _submit_prompt(prompt, lang, history_container, key_prefix="_llm_tab")


def _submit_prompt(
    prompt: str,
    lang: str,
    history_container,
    key_prefix: str = "_llm",
    *,
    display_prompt: str | None = None,
):
    """Append a prompt, render local instant replies, or stream the model response."""
    prompt = (prompt or "").strip()
    if not prompt:
        return
    if key_prefix == "_llm_floating" and _apply_floating_copilot_text_intent(prompt, lang):
        st.rerun()
        return
    if (
        key_prefix.startswith("_llm_ai_page_workspace")
        and not str(st.session_state.get("_copilot_current_session_id") or "").strip()
    ):
        _start_new_copilot_study_session(st.session_state, lang)

    visible_prompt = (display_prompt or prompt).strip()
    st.session_state.llm_messages.append(_copilot_user_message(prompt, visible_prompt))
    with history_container:
        with st.chat_message("user"):
            st.markdown(visible_prompt)

    guided_reply = _handle_copilot_guided_prompt(prompt, lang)
    if guided_reply is not None:
        reply_content, guided_actions = guided_reply
        suppress_snapshot = bool(st.session_state.pop("_copilot_suppress_next_snapshot", False))
        workflow_snapshot = None if suppress_snapshot else _copilot_workflow_snapshot(st.session_state, lang)
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        message = {
            "role": "assistant",
            "content": reply_content,
            "actions": guided_actions,
        }
        if workflow_snapshot is not None:
            message["workflow_snapshot"] = workflow_snapshot
        st.session_state.llm_messages.append(message)
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(reply_content)
                _render_nav_actions(guided_actions, key_prefix=f"{key_prefix}_guided")
                _render_copilot_inline_step_controls(lang, key_prefix)
                if workflow_snapshot is not None:
                    _render_copilot_workflow_snapshot(
                        workflow_snapshot,
                        lang,
                        key_prefix=f"{key_prefix}_guided",
                    )
        _request_copilot_scroll_to_latest(st.session_state)
        _touch_current_copilot_study_session(st.session_state, lang)
        return

    instant_reply = _get_instant_reply(prompt, lang)
    if instant_reply is not None:
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        instant_actions = _suggest_ui_actions(prompt, instant_reply, lang)
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": instant_reply,
                "actions": instant_actions,
            }
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(instant_reply)
                _render_nav_actions(instant_actions, key_prefix=f"{key_prefix}_instant")
        _request_copilot_scroll_to_latest(st.session_state)
        _touch_current_copilot_study_session(st.session_state, lang)
        return

    try:
        assistant_external_llm_ready = _external_llm_ready(lang)
    except AIOptInError:
        assistant_external_llm_ready = False
    if (
        st.session_state.get("_active_main_page") == "assistant"
        and not assistant_external_llm_ready
    ):
        fallback_reply = _local_copilot_fallback_reply(prompt, lang)
        fallback_actions = _suggest_ui_actions(prompt, fallback_reply, lang)
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        st.session_state.llm_messages.append(
            {"role": "assistant", "content": fallback_reply, "actions": fallback_actions}
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(fallback_reply)
                _render_nav_actions(fallback_actions, key_prefix=f"{key_prefix}_fallback")
        _request_copilot_scroll_to_latest(st.session_state)
        _touch_current_copilot_study_session(st.session_state, lang)
        return

    prep_placeholder = st.empty()
    prep_placeholder.info(
        "Preparing tools..." if lang == "en" else "正在准备工具..."
    )
    try:
        messages, tool_events = _compose_agent_messages(prompt)
    except Exception as exc:
        prep_placeholder.empty()
        error_message = _handle_api_error(exc, lang, render=False)
        st.session_state.llm_messages.append(
            {"role": "assistant", "content": error_message, "actions": []}
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(error_message)
        _request_copilot_scroll_to_latest(st.session_state)
        _touch_current_copilot_study_session(st.session_state, lang)
        return
    st.session_state.llm_last_tool_events = tool_events
    prep_placeholder.empty()

    with history_container:
        with st.chat_message("assistant"):
            _stream_response(messages, lang)
    _request_copilot_scroll_to_latest(st.session_state)
    _touch_current_copilot_study_session(st.session_state, lang)


def _submit_prompt_background(
    prompt: str,
    lang: str,
    history_container,
    key_prefix: str = "_llm",
    *,
    display_prompt: str | None = None,
) -> None:
    """Append a routed prompt and generate without blocking page navigation."""
    prompt = (prompt or "").strip()
    if not prompt:
        return
    if key_prefix == "_llm_floating" and _apply_floating_copilot_text_intent(prompt, lang):
        st.rerun()
        return
    if (
        key_prefix.startswith("_llm_ai_page_workspace")
        and not str(st.session_state.get("_copilot_current_session_id") or "").strip()
    ):
        _start_new_copilot_study_session(st.session_state, lang)

    visible_prompt = (display_prompt or prompt).strip()
    st.session_state.llm_messages.append(_copilot_user_message(prompt, visible_prompt))
    with history_container:
        with st.chat_message("user"):
            st.markdown(visible_prompt)

    guided_reply = _handle_copilot_guided_prompt(prompt, lang)
    if guided_reply is not None:
        reply_content, guided_actions = guided_reply
        suppress_snapshot = bool(st.session_state.pop("_copilot_suppress_next_snapshot", False))
        workflow_snapshot = None if suppress_snapshot else _copilot_workflow_snapshot(st.session_state, lang)
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        message = {
            "role": "assistant",
            "content": reply_content,
            "actions": guided_actions,
        }
        if workflow_snapshot is not None:
            message["workflow_snapshot"] = workflow_snapshot
        st.session_state.llm_messages.append(message)
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(reply_content)
                _render_nav_actions(guided_actions, key_prefix=f"{key_prefix}_guided")
                _render_copilot_inline_step_controls(lang, key_prefix)
                if workflow_snapshot is not None:
                    _render_copilot_workflow_snapshot(
                        workflow_snapshot,
                        lang,
                        key_prefix=f"{key_prefix}_guided",
                    )
        _request_copilot_scroll_to_latest(st.session_state)
        _touch_current_copilot_study_session(st.session_state, lang)
        return

    instant_reply = _get_instant_reply(prompt, lang)
    if instant_reply is not None:
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        instant_actions = _suggest_ui_actions(prompt, instant_reply, lang)
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": instant_reply,
                "actions": instant_actions,
            }
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(instant_reply)
                _render_nav_actions(instant_actions, key_prefix=f"{key_prefix}_instant")
        _request_copilot_scroll_to_latest(st.session_state)
        _touch_current_copilot_study_session(st.session_state, lang)
        return

    try:
        assistant_external_llm_ready = _external_llm_ready(lang)
    except AIOptInError:
        assistant_external_llm_ready = False
    if (
        st.session_state.get("_active_main_page") == "assistant"
        and not assistant_external_llm_ready
    ):
        fallback_reply = _local_copilot_fallback_reply(prompt, lang)
        fallback_actions = _suggest_ui_actions(prompt, fallback_reply, lang)
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        st.session_state.llm_messages.append(
            {"role": "assistant", "content": fallback_reply, "actions": fallback_actions}
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(fallback_reply)
                _render_nav_actions(fallback_actions, key_prefix=f"{key_prefix}_fallback")
        _request_copilot_scroll_to_latest(st.session_state)
        return

    session_id = _start_bg_response(prompt, lang)
    if session_id:
        st.session_state["_ai_bg_session_id"] = session_id
        st.session_state["_ai_bg_responding"] = True
        st.session_state["_ai_bg_response_ready"] = False
        st.session_state["_ai_bg_unread_count"] = 0
        status_text = (
            "Generating response in the background. You can switch pages while I work."
            if lang == "en" else
            "正在后台生成回答。你可以切换页面，助手不会悬浮残留。"
        )
        with history_container:
            st.markdown(
                f'<div class="inline-ai-status-strip">{html.escape(status_text)}</div>',
                unsafe_allow_html=True,
            )
        return

    error_message = (
        "I could not start the assistant response. Check the AI provider settings, then try again."
        if lang == "en" else
        "无法启动助手回答。请检查 AI 服务商设置后重试。"
    )
    st.session_state.llm_messages.append(
        {"role": "assistant", "content": error_message, "actions": []}
    )
    with history_container:
        with st.chat_message("assistant"):
            st.markdown(error_message)
    _request_copilot_scroll_to_latest(st.session_state)


def render_sidebar_chat_widget():
    """Render a compact embedded AI chat box in the sidebar."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")
    pending_prompt = st.session_state.get("_ai_pending_question")
    expanded = bool(st.session_state.get("_sidebar_ai_open") or pending_prompt)

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome {
            border: 1px solid #d8e2ee;
            border-left: 3px solid #0e7490;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.78rem 0.82rem;
            margin: 0.25rem 0 0.55rem;
            box-shadow: none;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome-title {
            color: #111827;
            font-size: 0.84rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 0.28rem;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome-subtitle,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-user-bubble,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-answer-card {
            color: #475569;
            font-size: 0.74rem;
            line-height: 1.45;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-sample {
            display: grid;
            gap: 0.38rem;
            margin: 0.52rem 0;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-user-bubble,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-answer-card,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-recommendation {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            padding: 0.52rem 0.58rem;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-recommendation span {
            display: block;
            color: #64748b;
            font-size: 0.62rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.16rem;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-recommendation strong,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome-hint {
            color: #0f172a;
            font-size: 0.72rem;
            line-height: 1.38;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel [data-testid="stChatMessageContent"] p,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel input,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel label {
            font-size: 0.75rem !important;
            line-height: 1.42;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .stButton > button,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel form button {
            min-height: 32px;
            border-radius: 7px;
            font-size: 0.74rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = "Embedded chat" if lang == "en" else "嵌入式对话"
    with st.expander(title, expanded=expanded):
        st.session_state["_sidebar_ai_open"] = expanded
        if not st.session_state.llm_enabled:
            st.caption(
                "Enable Research Copilot above to start chatting here."
                if lang == "en" else
                "请先在上方开启研究 Copilot，然后在这里对话。"
            )
            return

        if not _is_configured():
            st.caption(
                "Configure the provider/API key above, then chat here."
                if lang == "en" else
                "请先在上方完成服务商/API Key 配置，再在这里对话。"
            )
            return

        with st.container(key="embedded_ai_chat_panel"):
            _render_compact_chat_panel(
                lang=lang,
                panel_key="_llm_sidebar_embedded",
                history_height=300,
                show_starters=not bool(pending_prompt),
            )


def _inline_ai_context_html(lang: str, state=None) -> str:
    payload = _inline_ai_context_payload(st.session_state if state is None else state, lang)
    is_en = lang == "en"
    tags_html = "".join(f"<span>{html.escape(str(tag))}</span>" for tag in payload["tags"])
    return (
        '<div class="inline-ai-context-card">'
        f'<div class="inline-ai-section-label">{"In context" if is_en else "当前上下文"}</div>'
        f'<h3>{html.escape(str(payload["context_name"]))}</h3>'
        f'<p class="mono">{html.escape(str(payload["detail"]))}</p>'
        f'<div class="inline-ai-tag-row">{tags_html}</div>'
        '</div>'
    )


def _render_inline_ai_blocked_state(lang: str, *, enabled: bool) -> None:
    is_en = lang == "en"
    title = (
        "Assistant is off" if not enabled else "Provider key is missing"
    ) if is_en else (
        "助手已关闭" if not enabled else "缺少模型服务配置"
    )
    desc = (
        "Open Settings and enable the shared AI / API connection before chatting here."
        if not enabled else
        "Configure provider/API key in Settings first. API keys stay in this browser session only."
    ) if is_en else (
        "请先打开设置并启用共享 AI / API 连接，再在这里对话。"
        if not enabled else
        "请先在设置中配置服务商/API Key。API Key 只保存在当前浏览器会话。"
    )
    st.markdown(
        '<div class="inline-ai-blocked">'
        '<div class="inline-ai-blocked-icon">'
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
        '</svg></div>'
        f'<div><b>{html.escape(title)}</b><p>{html.escape(desc)}</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_inline_ai_context_and_handoff(lang: str) -> None:
    is_en = lang == "en"
    st.markdown(_inline_ai_context_html(lang), unsafe_allow_html=True)
    st.markdown(
        '<div class="inline-ai-evidence-note">'
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M20 6 9 17l-5-5"/></svg>'
        '<div>'
        f'<b>{html.escape("Evidence-bound" if is_en else "证据绑定")}</b>'
        f'<p>{html.escape("Suggestions only. Drafting stays locked until the agent checks pass." if is_en else "这里只给建议。Agent 检查通过前，草稿始终锁定。")}</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="inline-ai-handoff-card">'
        f'<div class="inline-ai-section-label">{html.escape("Hand off to" if is_en else "交接到")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        if st.button(
            "Research Agent setup" if is_en else "Research Agent 配置",
            key="_inline_ai_to_agent_setup",
            use_container_width=True,
        ):
            _prepare_research_agent_handoff_from_ai(st.session_state)
            st.rerun()
    with col_b:
        if st.button(
            "How the workflow works" if is_en else "查看工作流说明",
            key="_inline_ai_to_workflow",
            icon=":material/help:",
            use_container_width=True,
        ):
            st.session_state["_active_main_page"] = "tutorial"
            st.session_state["_scroll_to_top"] = True
            st.session_state["_inline_ai_panel_open"] = False
            st.session_state["_floating_ai_open"] = False
            st.session_state.pop("_ai_pending_question", None)
            st.rerun()


def _ai_panel_header_html(lang: str) -> str:
    provider_key = coerce_public_provider(
        st.session_state.get("llm_provider", public_default_provider_key())
    )
    provider_label = public_provider_defaults(provider_key)[0] or provider_key
    default_model = public_provider_defaults(provider_key)[2] or "model"
    model_label = st.session_state.get("llm_model") or default_model
    title = "Research Copilot" if lang == "en" else "研究 Copilot"
    if provider_key == "easyicu_hosted":
        subtitle = (
            "EasyICU hosted · evidence-bound"
            if lang == "en" else
            "EasyICU 托管 · 证据绑定"
        )
    else:
        subtitle = (
            f"{provider_label} · {model_label} · evidence-bound"
            if lang == "en" else
            f"{provider_label} · {model_label} · 证据绑定"
        )
    llm_configured = _is_configured()
    if not st.session_state.get("llm_enabled", False):
        status_label = "AI off" if lang == "en" else "AI 已关闭"
        status_color = "var(--ink-4)"
    elif not llm_configured:
        status_label = "key missing" if lang == "en" else "缺少 API key"
        status_color = "var(--warn)"
    else:
        status_label = "Ready" if lang == "en" else "已就绪"
        status_color = "var(--ok)"
    status_pill_html = (
        f'<span class="inline-ai-status-pill eu-pill mono" '
        f'style="font-size:10px;padding:2px 7px;height:18px;margin-left:8px;'
        f'background:var(--surface);color:{status_color};'
        f'border:1px solid var(--hair-2);vertical-align:middle">'
        f'<span class="dot" style="background:{status_color}"></span>'
        f'{html.escape(status_label)}</span>'
    )
    return (
        '<div class="inline-ai-header">'
        '<div class="inline-ai-avatar">AI</div>'
        '<div class="inline-ai-meta">'
        f'<div class="inline-ai-title">{html.escape(title)}</div>'
        f'<div class="inline-ai-subtitle">{html.escape(subtitle)}</div>'
        '</div>'
        f'{status_pill_html}'
        '</div>'
    )


def render_inline_ai_panel(*, force_open: bool = False, allow_close: bool = True):
    """Render the non-floating AI assistant as part of the main workspace."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")
    pending_prompt = bool(st.session_state.get("_ai_pending_question"))
    active_page = st.session_state.get("_active_main_page")
    if active_page != "assistant":
        st.session_state["_inline_ai_panel_open"] = False
        st.session_state["_floating_ai_open"] = False
        if force_open or pending_prompt:
            st.session_state["_active_main_page"] = "assistant"
            st.session_state["_scroll_to_top"] = True
            st.rerun()
        return
    if force_open:
        st.session_state["_inline_ai_panel_open"] = True
    if pending_prompt:
        st.session_state["_inline_ai_panel_open"] = True
        st.session_state["llm_enabled"] = True
        st.session_state["_llm_toggle"] = True
    if not st.session_state.get("_inline_ai_panel_open", False):
        return

    st.session_state["_floating_ai_open"] = False
    st.markdown(
        """
        <style>
        .stApp div.st-key-inline_ai_assistant_panel {
            position: relative;
            border: 1px solid var(--hair);
            border-radius: var(--r-3);
            background: color-mix(in oklab, white 74%, var(--surface));
            padding: 0;
            margin: 0.72rem 0 1.35rem;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-header {
            display: flex;
            align-items: center;
            gap: 0.78rem;
            min-width: 0;
            padding: 1.02rem 4.1rem 1rem 1.18rem;
            border-bottom: 1px solid var(--hair);
        }
        .stApp div.st-key-inline_ai_assistant_panel .st-key-_inline_ai_close {
            position: absolute !important;
            top: 0.66rem;
            right: 0.86rem;
            z-index: 5;
            width: 2.55rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel .st-key-_inline_ai_close button {
            width: 2.55rem !important;
            min-width: 2.55rem !important;
            min-height: 2.05rem !important;
            padding: 0 !important;
            border-radius: var(--r-2) !important;
            background: white !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-avatar {
            width: 2.25rem;
            height: 2.25rem;
            border-radius: var(--r-2);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid color-mix(in oklab, var(--accent) 30%, var(--hair));
            font-family: var(--font-mono);
            font-size: 0.76rem;
            font-weight: 760;
            flex: 0 0 auto;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 720;
            line-height: 1.2;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-subtitle {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            line-height: 1.35;
            margin-top: 0.1rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome {
            border: 0;
            border-radius: 0;
            background: transparent;
            padding: 0.2rem 0 0.25rem;
            margin: 0;
            box-shadow: none;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-title {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 650;
            letter-spacing: .12em;
            text-transform: uppercase;
            line-height: 1.25;
            margin: 0 0 0.5rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-subtitle,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card {
            color: var(--ink);
            font-size: 0.9rem;
            line-height: 1.58;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-sample {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 0.72rem;
            margin: 0.7rem 0 0.78rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation {
            border-radius: var(--r-3);
            border: 1px solid var(--hair);
            background: var(--surface);
            padding: 0.86rem 1rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble {
            max-width: 68%;
            margin-left: auto;
            background: var(--ink);
            color: white;
            border-color: var(--ink);
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card {
            max-width: 82%;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation {
            display: grid;
            gap: 0.32rem;
            max-width: 82%;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation span {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation strong,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-hint {
            color: var(--ink);
            font-size: 0.86rem;
            line-height: 1.5;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-handoff-card {
            margin: 0.78rem 1.18rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked {
            display: grid;
            grid-template-columns: 2.25rem 1fr;
            gap: 0.82rem;
            align-items: start;
            padding: 0.9rem 1rem;
            border: 1px solid color-mix(in oklab, var(--warn) 32%, var(--hair));
            border-radius: var(--r-3);
            background: color-mix(in oklab, var(--warn) 10%, white);
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked-icon {
            width: 2.1rem;
            height: 2.1rem;
            border-radius: var(--r-2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--warn);
            background: white;
            border: 1px solid color-mix(in oklab, var(--warn) 28%, var(--hair));
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked b,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card h3,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note b {
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.3;
            font-weight: 720;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked p,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card p,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note p {
            margin: 0.24rem 0 0;
            color: var(--ink-2);
            font-size: 0.84rem;
            line-height: 1.5;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card {
            padding: 1rem;
            border: 1px solid var(--hair);
            border-radius: var(--r-3);
            background: white;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-section-label {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 650;
            letter-spacing: .12em;
            line-height: 1.2;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card h3 {
            margin: 0;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-tag-row {
            display: flex;
            gap: 0.42rem;
            flex-wrap: wrap;
            margin-top: 0.82rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-tag-row span {
            display: inline-flex;
            align-items: center;
            min-height: 24px;
            padding: 0 0.72rem;
            border: 1px solid var(--hair);
            border-radius: 999px;
            background: var(--surface);
            color: var(--ink-2);
            font-family: var(--font-mono);
            font-size: 0.78rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note {
            display: grid;
            grid-template-columns: 1.3rem 1fr;
            gap: 0.72rem;
            align-items: start;
            padding: 0.98rem 1rem;
            border: 1px solid color-mix(in oklab, var(--ok) 30%, var(--hair));
            border-radius: var(--r-3);
            background: color-mix(in oklab, var(--ok) 13%, white);
            color: var(--ok);
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-handoff-card {
            margin-bottom: 0.2rem;
            padding-top: 0.2rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-status-strip {
            margin: 0.42rem 1.18rem 0.66rem;
            padding: 0.62rem 0.78rem;
            border: 1px solid var(--hair);
            border-radius: var(--r-2);
            background: color-mix(in oklab, var(--surface) 86%, white);
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.72rem;
            line-height: 1.35;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"] {
            display: flex !important;
            align-items: flex-start !important;
            gap: 0.72rem !important;
            padding: 0.36rem 1.18rem !important;
            background: transparent !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse !important;
            justify-content: flex-start !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"],
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"] {
            width: 2rem !important;
            height: 2rem !important;
            min-width: 2rem !important;
            border-radius: var(--r-2) !important;
            border: 1px solid var(--hair) !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"] {
            background: var(--accent-soft) !important;
            color: var(--accent) !important;
            border-color: color-mix(in oklab, var(--accent) 26%, var(--hair)) !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"] {
            background: color-mix(in oklab, var(--warn) 13%, white) !important;
            color: var(--ink) !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"] > *,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"] > * {
            display: none !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"]::after,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"]::after {
            display: grid !important;
            place-items: center !important;
            width: 100% !important;
            height: 100% !important;
            font-family: var(--font-mono) !important;
            font-size: 0.56rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.04em !important;
            line-height: 1 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"]::after {
            content: "YOU";
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"]::after {
            content: "AI";
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] {
            flex: 0 1 auto !important;
            width: fit-content !important;
            max-width: min(820px, 82%) !important;
            min-width: 0 !important;
            margin: 0 !important;
            padding: 0.84rem 0.98rem !important;
            border: 1px solid var(--hair) !important;
            border-radius: var(--r-3) !important;
            background: var(--surface) !important;
            color: var(--ink) !important;
            box-shadow: none !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"]::before {
            content: "Assistant";
            display: block;
            margin-bottom: 0.34rem;
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.68rem;
            font-weight: 680;
            letter-spacing: 0.12em;
            line-height: 1.1;
            text-transform: uppercase;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            max-width: min(650px, 68%) !important;
            background: var(--ink) !important;
            border-color: var(--ink) !important;
            color: white !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]::before {
            content: "You";
            color: rgba(255, 255, 255, 0.68);
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] li {
            color: white !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] p,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] li,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stMarkdownContainer"] p,
        .stApp div.st-key-inline_ai_assistant_panel input,
        .stApp div.st-key-inline_ai_assistant_panel label {
            font-size: 0.9rem !important;
            line-height: 1.56;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] p,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ul,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ol {
            margin-top: 0 !important;
            margin-bottom: 0.48rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] p:last-child,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ul:last-child,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ol:last-child {
            margin-bottom: 0 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel .stButton > button,
        .stApp div.st-key-inline_ai_assistant_panel form button {
            min-height: 38px;
            border-radius: var(--r-2);
            font-size: 0.83rem;
            font-weight: 650;
            box-shadow: none !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] {
            border: 0 !important;
            padding: 0 1.18rem 1.05rem !important;
            background: transparent !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: flex-end !important;
            gap: 0.5rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stColumn"] {
            min-width: 0 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stColumn"]:last-child {
            flex: 0 0 3.25rem !important;
            width: 3.25rem !important;
            min-width: 3.25rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stTextInput"] {
            margin-bottom: 0 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] input {
            min-height: 40px !important;
            border-radius: var(--r-2) !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stFormSubmitButton"] button {
            min-width: 42px !important;
            min-height: 40px !important;
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
            font-size: 1rem !important;
        }
        @media (max-width: 900px) {
            .stApp div.st-key-inline_ai_assistant_panel { margin-top: 0.55rem; }
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-header { padding: 0.86rem 3.85rem 0.86rem 0.86rem; }
            .stApp div.st-key-inline_ai_assistant_panel .st-key-_inline_ai_close {
                top: -2.65rem;
                right: 0.78rem;
            }
            .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
            .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card,
            .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation { max-width: 100%; }
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"] {
                padding-left: 0.86rem !important;
                padding-right: 0.86rem !important;
            }
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-status-strip {
                margin-left: 0.86rem;
                margin-right: 0.86rem;
            }
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"],
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
                max-width: calc(100% - 2.72rem) !important;
            }
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked,
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card,
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note,
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-handoff-card {
                margin-left: 0.86rem;
                margin-right: 0.86rem;
            }
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] {
                padding-left: 0.86rem !important;
                padding-right: 0.86rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="inline_ai_assistant_panel"):
        if allow_close:
            left, close_col = st.columns([10, 1], gap="small")
        else:
            left = st.container()
            close_col = None
        with left:
            st.markdown(_ai_panel_header_html(lang), unsafe_allow_html=True)
        if close_col is not None:
            with close_col:
                if st.button(
                    "",
                    icon=":material/close:",
                    key="_inline_ai_close",
                    help="Close AI panel" if lang == "en" else "关闭 AI 面板",
                    use_container_width=True,
                ):
                    st.session_state["_inline_ai_panel_open"] = False
                    st.session_state["_ai_pending_question"] = None
                    st.rerun()

        enabled = bool(st.session_state.get("llm_enabled", False))
        configured = _is_configured()
        if not enabled:
            _render_inline_ai_blocked_state(lang, enabled=False)
            _render_inline_ai_context_and_handoff(lang)
            return

        if not configured:
            _render_inline_ai_blocked_state(lang, enabled=True)
            _render_inline_ai_context_and_handoff(lang)
            return

        _render_compact_chat_panel(
            lang=lang,
            panel_key="_llm_inline_workspace",
            history_height=440,
            show_starters=not pending_prompt,
        )
        _render_inline_ai_context_and_handoff(lang)


def _render_ai_assistant_workspace_page(lang: str, *, pending_prompt: bool) -> None:
    _consume_standalone_guided_pending_prompt(lang, st.session_state)
    with st.container(key="ai_assistant_page_panel"):
        st.markdown('<div class="eu-copilot-page-marker"></div>', unsafe_allow_html=True)
        is_en = lang == "en"

        with st.container(key="eu_copilot_guided_top"):
            brand_col, exit_col, classic_col = st.columns([1, 0.16, 0.28], gap="small")
            with brand_col:
                brand_icon = icon("flask")
                st.markdown(
                    '<div class="eu-copilot-topbrand">'
                    f'<span class="brand-mark">{brand_icon}</span>'
                    '<span><b>Research Copilot</b>'
                    f'<em>{html.escape("EasyICU · real-data first" if is_en else "EasyICU · 真实数据优先")}</em></span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with exit_col:
                if st.button(
                    "Exit" if is_en else "退出",
                    key="_copilot_top_exit",
                    use_container_width=True,
                ):
                    st.session_state["entry_mode"] = "none"
                    st.session_state["_active_main_page"] = "entry"
                    st.session_state["_scroll_to_top"] = True
                    st.session_state["_inline_ai_panel_open"] = False
                    st.session_state["_floating_ai_open"] = False
                    st.session_state.pop("_ai_pending_question", None)
                    st.rerun()
            with classic_col:
                if st.button(
                    "Classic workspace" if is_en else "经典工作区",
                    key="_copilot_top_classic_workspace",
                    use_container_width=True,
                ):
                    _apply_chat_workflow_action("study_extract")
                    st.rerun()

        with st.container(key="eu_copilot_guided_shell"):
            rail_col, chat_col, study_col = st.columns([232, 1040, 322], gap="small")
            with rail_col:
                with st.container(key="eu_copilot_left_rail"):
                    _render_copilot_session_rail(lang)
            with chat_col:
                with st.container(key="eu_copilot_conversation_shell"):
                    _render_compact_chat_panel(
                        lang=lang,
                        panel_key="_llm_ai_page_workspace",
                        history_height=520,
                        show_starters=False,
                        show_hint_chips=True,
                        welcome_variant="codex",
                        background_pending_prompts=False,
                    )
            with study_col:
                with st.container(key="eu_copilot_right_rail"):
                    _render_copilot_stage_workspace(lang)


def _consume_standalone_guided_pending_prompt(
    lang: str,
    state: MutableMapping[str, object],
) -> bool:
    """Apply homepage-seeded guided prompts before rendering the three columns."""
    prompt = str(state.get("_ai_pending_question") or "").strip()
    if not prompt:
        return False
    guided_reply = _handle_copilot_guided_prompt(prompt, lang, state)
    if guided_reply is None:
        return False
    state.pop("_ai_pending_question", None)
    display_prompt = state.pop("_ai_pending_question_display", None)
    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list):
        messages.append(_copilot_user_message(prompt, display_prompt))
        reply_content, guided_actions = guided_reply
        suppress_snapshot = bool(state.pop("_copilot_suppress_next_snapshot", False))
        message = {
            "role": "assistant",
            "content": reply_content,
            "actions": guided_actions,
        }
        if not suppress_snapshot:
            message["workflow_snapshot"] = _copilot_workflow_snapshot(state, lang)
        messages.append(message)
    _request_copilot_scroll_to_latest(state)
    state["llm_last_tool_events"] = []
    state["llm_last_verification"] = {
        "status": "pass",
        "issues": [],
    }
    return True


def _render_copilot_quick_actions(lang: str) -> None:
    """Render chat-first commands that also drive the classic workspace."""
    is_en = lang == "en"
    st.markdown(
        '<div class="inline-ai-section-label">'
        f'{html.escape("Start here" if is_en else "从这里开始")}'
        '</div>',
        unsafe_allow_html=True,
    )
    actions = [
        (
            "Guided demo" if is_en else "引导演示",
            "guided_demo",
            ":material/auto_awesome:",
            "_copilot_guided_demo",
            "Run the chat-first demo to the evidence gate" if is_en else "用聊天方式跑到证据闸门",
        ),
        (
            "Demo extraction" if is_en else "演示提取",
            "demo_extraction",
            ":material/science:",
            "_copilot_demo_extraction",
            "Start the classic 4-step flow with demo data" if is_en else "用演示数据进入经典四步流程",
        ),
        (
            "Demo review" if is_en else "演示审阅",
            "demo_review",
            ":material/table_chart:",
            "_copilot_demo_review",
            "Load a lightweight Patient Review workspace" if is_en else "加载轻量患者审阅工作区",
        ),
        (
            "Real data" if is_en else "真实数据",
            "real_extraction",
            ":material/database:",
            "_copilot_real_extraction",
            "Open local data-source setup" if is_en else "打开本地真实数据源配置",
        ),
        (
            "Agent setup" if is_en else "Agent 配置",
            "agent_handoff",
            ":material/smart_toy:",
            "_copilot_agent_setup",
            "Hand the latest question to Research Agent" if is_en else "把最近的问题交给 Research Agent",
        ),
    ]
    with st.container(key="eu_copilot_quick_actions"):
        cols = st.columns(5, gap="small")
        for col, (label, workflow, icon_name, key, help_text) in zip(cols, actions):
            with col:
                if st.button(
                    label,
                    key=key,
                    icon=icon_name,
                    use_container_width=True,
                    help=help_text,
                ):
                    if workflow == "agent_handoff":
                        _prepare_research_agent_handoff_from_ai(st.session_state)
                    else:
                        _apply_chat_workflow_action(workflow)
                    st.rerun()


def _append_copilot_rail_step_action(step_id: str, lang: str) -> bool:
    """Gated jump handler for right-rail Study workspace steps."""
    study = _ensure_copilot_study_state(st.session_state)
    if not _copilot_step_unlocked_for_navigation(study, step_id):
        label = _copilot_step_label(step_id, lang)
        st.session_state["_assistant_notice"] = (
            f"Complete the previous step before opening {label}."
            if lang == "en" else
            f"请先完成上一步，再打开{label}。"
        )
        return False
    _append_copilot_workflow_step_action(step_id, lang)
    return True


def _render_copilot_session_rail(lang: str) -> None:
    """Render the polish-design session rail for the standalone Copilot page."""
    is_en = lang == "en"
    state = st.session_state
    _ensure_copilot_study_state(state)

    if st.button(
        "New study" if is_en else "新研究",
        key="_copilot_new_study",
        use_container_width=True,
    ):
        _start_new_copilot_study_session(state, lang)
        st.rerun()

    sessions = _copilot_list_study_sessions(state)
    current_session_id = str(state.get("_copilot_current_session_id") or "")
    st.markdown(
        '<div class="eu-copilot-rail-body">'
        f'<div class="eu-copilot-rail-eyebrow">{html.escape("Recent" if is_en else "最近")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if not sessions:
        st.markdown(
            '<div class="eu-copilot-session-empty">'
            f'<b>{html.escape("No studies yet" if is_en else "还没有研究")}</b>'
            f'<p>{html.escape("Create a study to make a local workspace and start the first conversation." if is_en else "点击新研究，创建本地研究工作目录后再开始第一段对话。")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for idx, session in enumerate(sessions[:COPILOT_RECENT_SESSION_RENDER_LIMIT]):
            session_id = str(session.get("id") or "")
            title = str(session.get("title") or _copilot_session_fallback_title(lang)).strip()
            title = " ".join(title.split())
            if len(title) > 40:
                title = title[:39].rstrip() + "…"
            active = session_id == current_session_id
            rel = ("active now" if is_en else "进行中") if active else _copilot_relative_time(
                str(session.get("updated_at") or ""), lang
            )
            # Clean session row: small dot + title + inline relative time (design parity).
            with st.container(key=f"eu_copilot_session_{'active' if active else 'idle'}_{idx}"):
                if st.button(
                    title,
                    key=f"_copilot_session_open_{session_id}_{idx}",
                    use_container_width=True,
                    help=rel or None,
                ):
                    if session_id:
                        _open_copilot_study_session(state, session_id, lang)
                    st.rerun()
                if rel:
                    st.markdown(
                        f'<div class="eu-copilot-session-meta{" active" if active else ""}">'
                        f'{html.escape(rel)}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown('<div class="eu-copilot-left-spacer"></div>', unsafe_allow_html=True)

    if st.button(
        "Classic workspace" if is_en else "经典工作区",
        key="_copilot_rail_classic_workspace",
        use_container_width=True,
        help="Open the classic workspace only when you explicitly want to leave Copilot" if is_en else "只有明确想离开 Copilot 时才打开经典工作区",
    ):
        _apply_chat_workflow_action("study_extract")
        st.rerun()


# Active step -> (classic page key, workflow action, EN label, ZH label).
# The "advanced drawer": one click drops from the conversation into the SAME
# classic component for the current step, so users are never trapped in chat.
_COPILOT_STEP_ESCAPE = {
    "data": ("extract", "real_extraction", "Open classic Data Extraction", "打开经典数据提取"),
    "cohort": ("extract", "study_extract", "Open classic Cohort filters", "打开经典队列筛选"),
    "concepts": ("extract", "study_extract", "Open classic Feature modules", "打开经典特征模块"),
    "extract": ("extract", "study_extract", "Open classic Extraction", "打开经典提取"),
    "review": ("quick_viz", "study_review", "Open classic Patient Review", "打开经典患者审阅"),
    "analysis": ("assistant", "study_agent", "Open Research Agent", "打开研究 Agent"),
}


def _render_copilot_step_escape_hatch(study: Mapping[str, object], lang: str) -> None:
    """Advanced drawer / escape hatch: jump to the classic view for the active step."""
    is_en = lang == "en"
    step = str(study.get("step") or "question")
    entry = _COPILOT_STEP_ESCAPE.get(step)
    if not entry:
        return
    _page, workflow, label_en, label_zh = entry
    with st.expander("Advanced · classic view" if is_en else "高级 · 经典视图", expanded=False):
        st.caption(
            "Drop into the full classic component for this step. Your study state stays in sync."
            if is_en else
            "下钻到该步骤的完整经典组件，研究状态保持同步。"
        )
        if st.button(
            label_en if is_en else label_zh,
            key=f"_copilot_escape_{step}",
            use_container_width=True,
            icon=":material/open_in_new:",
        ):
            _apply_chat_workflow_action(workflow)
            st.rerun()


def _render_copilot_depth_control(study: Mapping[str, object], lang: str) -> None:
    """Render the depth selector (how far the study runs) above the step list.

    extract = filter + extract · review = + visual review · full = full agent run.
    Orthogonal to branch (what to study); drives where the study stops.
    """
    is_en = lang == "en"
    current = _copilot_engine.normalize_depth(study.get("depth"))
    labels = {
        "extract": "Extract" if is_en else "提取",
        "review": "Review" if is_en else "审阅",
        "full": "Full agent" if is_en else "全流程",
    }
    key = "_copilot_depth_seg"
    # Reconcile from a pending value (e.g. set by "take it further") BEFORE the
    # widget is instantiated — the only point where writing the key is allowed.
    pending = st.session_state.pop("_copilot_depth_seg_pending", None)
    if pending in labels:
        st.session_state[key] = pending
    elif st.session_state.get(key) not in labels:
        st.session_state[key] = current
    st.markdown(
        '<div class="eu-copilot-rail-eyebrow">'
        f'{html.escape("How far to run" if is_en else "本次跑多远")}</div>',
        unsafe_allow_html=True,
    )
    choice = st.segmented_control(
        "How far to run" if is_en else "本次跑多远",
        options=["extract", "review", "full"],
        format_func=lambda d: labels[d],
        key=key,
        label_visibility="collapsed",
    )
    if choice and choice != current:
        _apply_chat_workflow_action(f"study_depth_{choice}")
        st.rerun()


def _render_copilot_stage_workspace(lang: str) -> None:
    """Render the lightweight study progress rail from the polish design."""
    is_en = lang == "en"
    study = _ensure_copilot_study_state(st.session_state)
    step_items = _copilot_rail_step_items(study, lang)
    note_title = "Evidence-bound" if is_en else "证据绑定"
    note_body = (
        "Draft stays gated until checks pass."
        if is_en else
        "检查通过前，草稿保持锁定。"
    )
    with st.container(key="eu_copilot_study_rail"):
        st.markdown(
            '<div class="eu-copilot-study-rail">'
            '<div class="eu-copilot-study-rail-head">'
            f'<div class="eu-copilot-rail-eyebrow">{html.escape("Building your study" if is_en else "构建你的研究")}</div>'
            f'<h3>{html.escape("Study workspace" if is_en else "研究工作区")}</h3>'
            f'<p>{html.escape("Assembles as we talk · edit any step" if is_en else "边聊边组装 · 可随时改步骤")}</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _render_copilot_depth_control(study, lang)
        with st.container(key="eu_study_step_list"):
            for idx, item in enumerate(step_items):
                step_id = str(item["id"])
                status = str(item["status"])
                unlocked = bool(item["unlocked"])
                detail = str(item.get("detail") or "")
                display_detail = detail if status != "locked" else ""
                button_label = str(item["label"])
                with st.container(key=f"eu_study_step_row_{idx}_{status}_{step_id}"):
                    st.markdown(
                        '<span class="si-dot">{}</span>'.format(
                            icon(str(item.get("icon") or "grid"))
                        ),
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        button_label,
                        key=f"_copilot_study_rail_step_{step_id}",
                        disabled=not unlocked,
                        use_container_width=True,
                        help=detail or (
                            "Complete the previous step first"
                            if is_en else
                            "请先完成上一步"
                        ),
                    ):
                        if _append_copilot_rail_step_action(step_id, lang):
                            st.rerun()
                    if display_detail:
                        st.markdown(
                            f'<p class="eu-study-step-detail">{html.escape(display_detail)}</p>',
                            unsafe_allow_html=True,
                        )
                # Finish-line after the depth goal step (prototype .study-finishline).
                if bool(item.get("is_goal")):
                    depth_label = _copilot_engine.normalize_depth(study.get("depth"))
                    st.markdown(
                        '<div class="study-finishline">'
                        f'<span class="fl-flag">{icon("check")}</span>'
                        f'<span class="fl-t">{html.escape(("Finish line" if is_en else "终点线") + " · " + depth_label)}</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    if str(study.get("depth") or "full") != "full":
                        if st.button(
                            "Take it further →" if is_en else "继续深入 →",
                            key="_copilot_take_further",
                            use_container_width=True,
                            help=("Extend the study one stage deeper"
                                  if is_en else "把研究再向后推进一档"),
                        ):
                            _apply_chat_workflow_action("study_take_further")
                            st.rerun()
        _render_copilot_step_escape_hatch(study, lang)
        st.markdown(
            '<div class="eu-copilot-evidence-note">'
            f'<span>{icon("check")}</span>'
            '<div>'
            f'<b>{html.escape(note_title)}</b>'
            f'<p>{html.escape(note_body)}</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def _render_copilot_state_panel(lang: str) -> None:
    """Show the shared GUI/chat state so users can see both architectures linked."""
    is_en = lang == "en"
    state = st.session_state
    entry_mode = str(state.get("entry_mode") or "none")
    database = str(state.get("database") or "miiv")
    selected_count = len(state.get("selected_concepts") or [])
    loaded_count = len(state.get("loaded_concepts") or {})
    patient_count = len(state.get("patient_ids") or [])
    data_path = str(state.get("data_path") or "").strip()
    path_validated = bool(state.get("path_validated"))
    study = _ensure_copilot_study_state(state)
    study_question = str(study.get("question") or "").strip()
    study_step = str(study.get("step") or "question")
    study_idx = COPILOT_STEP_INDEX.get(study_step, 0)
    if entry_mode == "none" and study.get("data_mode") in {"demo", "real"}:
        entry_mode = str(study.get("data_mode"))

    mode_label = {
        "demo": "Demo" if is_en else "演示",
        "real": "Real Data" if is_en else "真实数据",
        "none": "Not selected" if is_en else "未选择",
    }.get(entry_mode, entry_mode)
    db_label = "mock" if database == "mock" else database.upper()

    step1_done = entry_mode == "demo" or bool(data_path and path_validated) or study_idx >= COPILOT_STEP_INDEX["data"]
    step2_done = bool(state.get("step2_confirmed")) or study_idx >= COPILOT_STEP_INDEX["cohort"]
    step3_done = bool(state.get("step3_confirmed")) or selected_count > 0 or study_idx >= COPILOT_STEP_INDEX["concepts"]
    review_done = loaded_count > 0 or patient_count > 0 or study_idx >= COPILOT_STEP_INDEX["review"]
    agent_ready = bool(state.get("research_agent_question")) or review_done or study_idx >= COPILOT_STEP_INDEX["analysis"]

    def status(done: bool, active: bool = False) -> str:
        if done:
            return "done"
        return "active" if active else "todo"

    rows = [
        _workflow_status_step(
            "Question" if is_en else "研究问题",
            (str(state.get("research_agent_question") or study_question).strip()[:82] or (
                "Ask in chat or choose a starter prompt" if is_en else "可在聊天中提出，也可选择起始问题"
            )),
            status(bool(state.get("research_agent_question") or study_question), active=study_step == "question"),
        ),
        _workflow_status_step(
            "Data source" if is_en else "数据源",
            f"{mode_label} · {db_label}" + ("" if not data_path else f" · {Path(data_path).name}"),
            status(step1_done, active=study_step == "data" or entry_mode == "none"),
        ),
        _workflow_status_step(
            "Cohort" if is_en else "队列",
            ("confirmed" if step2_done else "waiting for Step 2") if is_en else ("已确认" if step2_done else "等待第 2 步"),
            status(step2_done, active=study_step == "cohort" or (step1_done and not step2_done)),
        ),
        _workflow_status_step(
            "Concepts" if is_en else "变量",
            (f"{selected_count or len(study.get('modules') or [])} selected" if is_en else f"已选 {selected_count or len(study.get('modules') or [])} 个"),
            status(step3_done, active=study_step == "concepts" or (step2_done and not step3_done)),
        ),
        _workflow_status_step(
            "Review" if is_en else "审阅",
            f"{loaded_count} concepts · {patient_count} patients" if is_en else f"{loaded_count} 个概念 · {patient_count} 位患者",
            status(review_done, active=study_step == "review" or (step3_done and not review_done)),
        ),
        _workflow_status_step(
            "Agent" if is_en else "Agent",
            ("draft gated" if study_step == "draft" else "ready for handoff") if is_en else ("草稿已闸门锁定" if study_step == "draft" else "可交接"),
            status(agent_ready, active=study_step in {"analysis", "draft"} or review_done),
        ),
    ]

    st.markdown(
        '<div class="eu-copilot-state-card">'
        f'<div class="inline-ai-section-label">{html.escape("Workspace state" if is_en else "工作区状态")}</div>'
        f'<h3>{html.escape("Chat and GUI stay linked" if is_en else "聊天和图形界面保持联动")}</h3>'
        f'<p>{html.escape("Every chat command writes into the same session state as the classic EasyICU panels." if is_en else "每个聊天命令都会写入经典 EasyICU 面板共用的会话状态。")}</p>'
        '<div class="eu-copilot-stepper">'
        + "".join(rows)
        + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    nav_cols = st.columns(2, gap="small")
    with nav_cols[0]:
        if st.button(
            "Classic flow" if is_en else "经典流程",
            key="_copilot_open_classic_flow",
            icon=":material/account_tree:",
            use_container_width=True,
        ):
            state["_active_main_page"] = "extract"
            state["_scroll_to_top"] = True
            st.rerun()
    with nav_cols[1]:
        if st.button(
            "Patient Review" if is_en else "患者审阅",
            key="_copilot_open_patient_review",
            icon=":material/table_chart:",
            use_container_width=True,
        ):
            state["_active_main_page"] = "quick_viz"
            state["_scroll_to_top"] = True
            st.rerun()


def render_ai_assistant_page(lang: str | None = None, app_context: dict | None = None) -> None:
    """Render the latest Tools · Research Copilot page using the live chat backend."""
    _init_chat_state()
    if app_context is not None:
        # Stash the host app's context so Copilot steps can reach the SAME
        # classic engine functions (which need `st`/`pd`/helpers injected).
        st.session_state["_app_context"] = app_context
    lang = lang or st.session_state.get("language", "en")
    pending_prompt = bool(st.session_state.get("_ai_pending_question"))
    st.session_state["_floating_ai_open"] = False
    st.session_state["_inline_ai_panel_open"] = False
    _render_ai_assistant_workspace_page(lang, pending_prompt=pending_prompt)


def _render_chat_welcome(
    *,
    lang: str,
    panel_key: str,
    history_container,
    show_starters: bool = True,
    welcome_variant: str = "rich",
) -> None:
    if welcome_variant == "codex":
        intro = (
            "Hi - I'm the EasyICU <strong>Research Copilot</strong>. I'll walk a whole study through by chat: "
            "framing the question, pulling data, running the analysis, and preparing a gated draft. Everything runs locally."
            if lang == "en" else
            "你好，我是 EasyICU <strong>Research Copilot</strong>。我会用聊天带你走完整研究：框定问题、拉取数据、运行分析，并准备带证据闸门的草稿。所有流程都在本机运行。"
        )
        ask = (
            "What would you like to study? Pick a starting point or just describe it."
            if lang == "en" else
            "你想研究什么？可以点一个起点，也可以直接描述。"
        )
        st.markdown(
            '<div class="eu-copilot-welcome-thread">'
            '<div class="eu-copilot-msg bot">'
            '<span class="m-ava">AI</span>'
            f'<div class="m-bubble">{intro}</div>'
            '</div>'
            '<div class="eu-copilot-msg bot">'
            '<span class="m-ava">AI</span>'
            f'<div class="m-bubble compact">{html.escape(ask)}</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    title = "Research Copilot" if lang == "en" else "研究 Copilot"
    subtitle = (
        "I can walk a whole study through by chat: frame the question, pick data, assemble the cohort, load review, and hand a gated question to Research Agent."
        if lang == "en" else
        "我可以用聊天带你走完整研究：框定问题、选择数据、组装队列、加载审阅，并把带闸门的问题交给 Research Agent。"
    )
    prompt_hint = (
        "Try one of these prompts:" if lang == "en" else "可以从这些提示开始："
    ) if show_starters else (
        "Ask about your current workflow, cohort, or export settings."
        if lang == "en" else
        "可以直接询问当前流程、队列筛选或导出设置。"
    )
    sample_q = (
        'Help me turn an ICU outcome idea into a researchable question.'
        if lang == "en" else
        "帮我把一个 ICU 结局研究想法整理成可研究问题。"
    )
    sample_a = (
        "I will bind it to a cohort, outcome, first-24h window, feature modules, review gate, and Research Agent handoff."
        if lang == "en" else
        "我会把它绑定到队列、结局、前 24 小时窗口、特征模块、审阅闸门和 Research Agent 交接。"
    )
    rec_label = "Evidence-bound note" if lang == "en" else "证据绑定提示"
    rec_text = (
        "I will not assert effect sizes here; those come from the run, and only after the evidence gate opens."
        if lang == "en" else
        "我不会在这里断言效应量；这些必须来自实际运行，并且只有证据闸门通过后才能写入。"
    )

    st.markdown(
        f'''
        <div class="floating-ai-welcome">
            <div class="floating-ai-welcome-title">{title}</div>
            <div class="floating-ai-welcome-subtitle">{subtitle}</div>
            <div class="floating-ai-sample">
                <div class="floating-ai-user-bubble">{sample_q}</div>
                <div class="floating-ai-answer-card">{sample_a}</div>
                <div class="floating-ai-recommendation">
                    <span>{rec_label}</span>
                    <strong>{rec_text}</strong>
                </div>
            </div>
            <div class="floating-ai-welcome-hint">{prompt_hint}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if show_starters:
        prompts = _starter_prompts(lang)
        for idx, starter in enumerate(prompts):
            if st.button(
                starter,
                key=f"{panel_key}_starter_{idx}",
                use_container_width=True,
            ):
                _submit_prompt(starter, lang, history_container, key_prefix=f"{panel_key}_starter")
                st.rerun()


def _render_copilot_primary_chips(lang: str, panel_key: str) -> None:
    """Render the main intent chips above the guided composer.

    Only shown at the welcome state as study entry points. Once the study has
    started, the current step's choices are already injected inline on the
    latest assistant reply (see `_copilot_message_actions_for_current_step`),
    so repeating them as a bottom chip row was pure duplication — each step now
    has a single place to act. The contextual "Try:" hint row still renders.
    """
    if st.session_state.get("_active_main_page") != "assistant":
        return
    raw_study = st.session_state.get("_copilot_guided_study")
    study: Mapping[str, object] = raw_study if isinstance(raw_study, Mapping) else {}
    if _copilot_started_in_chat(st.session_state, study):
        return
    actions = _copilot_primary_actions_for_state(st.session_state, lang)
    if not actions:
        return
    with st.container(key=f"{panel_key}_guided_intents"):
        cols = st.columns(len(actions), gap="small")
        for idx, action in enumerate(actions):
            with cols[idx]:
                if st.button(
                    str(action["label"]),
                    key=f"{panel_key}_guided_intent_{idx}_{action['id']}",
                    type="secondary",
                ):
                    label = str(action.get("label") or "").strip()
                    st.session_state["_ai_pending_question"] = str(action.get("prompt") or label)
                    st.session_state["_ai_pending_question_display"] = label
                    st.rerun()


def _render_copilot_hint_chips(lang: str, panel_key: str) -> None:
    """Render guided composer hint chips from the polish design."""
    is_en = lang == "en"
    if st.session_state.get("_active_main_page") != "assistant":
        return
    label = "Try:" if is_en else "试试："
    prompts = _copilot_hint_prompts_for_state(st.session_state, lang)
    with st.container(key=f"{panel_key}_guided_hints"):
        cols = st.columns([0.18, 0.34, 0.34, 0.24, 0.38], gap="small")
        with cols[0]:
            st.markdown(
                '<div class="eu-copilot-hint-row">'
                f'<span>{html.escape(label)}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        for idx, (button_label, prompt) in enumerate(prompts[:4]):
            with cols[idx + 1]:
                if st.button(
                    button_label,
                    key=f"{panel_key}_guided_hint_{idx}",
                ):
                    st.session_state["_ai_pending_question"] = prompt
                    st.session_state["_ai_pending_question_display"] = button_label
                    st.rerun()


def _copilot_active_data_source_choice(state: Mapping[str, object]) -> str:
    """Return the current inline data-source form kind, if one should be shown."""
    raw_study = state.get("_copilot_guided_study")
    if not isinstance(raw_study, Mapping):
        return ""
    if str(raw_study.get("step") or "") != "data":
        return ""
    choice = str(state.get("_copilot_data_source_choice") or raw_study.get("data_source_choice") or "")
    status = str(raw_study.get("data_source_status") or "")
    if status not in {"awaiting_path", ""}:
        return ""
    return choice if choice in {"prepared_path", "module_export", "raw_files"} else ""


def _render_copilot_data_source_inline_form(lang: str, panel_key: str) -> None:
    """Render the in-page data-source controls for Copilot's guided flow."""
    if st.session_state.get("_active_main_page") != "assistant":
        return
    choice = _copilot_active_data_source_choice(st.session_state)
    if not choice:
        return
    is_en = lang == "en"
    title = _copilot_data_source_choice_label(choice, lang)
    if choice == "module_export":
        detail = (
            "Save the EasyICU module export folder. This can be handed to Research Agent after cohort/module choices."
            if is_en else
            "保存 EasyICU 模块导出文件夹。后续选择队列/模块后，可直接交给 Research Agent。"
        )
        placeholder = "/path/to/easyicu_module_export"
        submit_label = "Save export" if is_en else "保存导出目录"
    elif choice == "raw_files":
        detail = (
            "Record the raw ICU database root. It will stay pending conversion before any analysis run."
            if is_en else
            "记录 ICU 原始数据库根目录。正式分析前它会保持待转换状态。"
        )
        placeholder = "/path/to/raw_mimiciv"
        submit_label = "Save raw path" if is_en else "保存原始目录"
    else:
        detail = (
            "Paste a prepared/converted EasyICU folder. Saving it records the path here and marks it pending validation."
            if is_en else
            "粘贴 prepared/converted EasyICU 文件夹。保存后会在当前 Copilot 记录路径，并标记为待验证。"
        )
        placeholder = "/path/to/prepared_miiv"
        submit_label = "Save path" if is_en else "保存路径"

    current_path = ""
    if choice == "module_export":
        current_path = str(st.session_state.get("last_export_dir") or st.session_state.get("export_path") or "")
    elif choice == "raw_files":
        current_path = str(st.session_state.get("raw_data_path") or "")
    else:
        current_path = str(st.session_state.get("data_path") or "")

    with st.container(key=f"{panel_key}_data_source_inline"):
        st.markdown(
            '<div class="eu-copilot-datasource-card">'
            f'<div class="ds-eyebrow">{html.escape("Data source" if is_en else "数据源")}</div>'
            f'<div class="ds-title">{html.escape(title)}</div>'
            f'<p>{html.escape(detail)}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.form(f"{panel_key}_data_source_form", clear_on_submit=False):
            selected_database = str(st.session_state.get("database") or "miiv")
            selected_database = _copilot_normalize_database(selected_database)
            db_col, path_col, save_col = st.columns([0.28, 1, 0.22], gap="small")
            with db_col:
                selected_database = st.selectbox(
                    "Database" if is_en else "数据库",
                    options=list(COPILOT_DATABASE_OPTIONS),
                    format_func=lambda value: _copilot_database_label(value, lang),
                    index=list(COPILOT_DATABASE_OPTIONS).index(selected_database),
                    key=f"{panel_key}_data_source_database_select",
                )
            with path_col:
                path_value = st.text_input(
                    title,
                    value=current_path,
                    placeholder=placeholder,
                    key=f"{panel_key}_data_source_path_input",
                    help=(
                        "EasyICU stores the path in this browser session and does not upload patient rows."
                        if is_en else
                        "EasyICU 只把路径保存在当前浏览器会话中，不上传患者行数据。"
                    ),
                )
            with save_col:
                submit_clicked = st.form_submit_button(submit_label, type="primary", use_container_width=True)
        if submit_clicked:
            result = _copilot_submit_data_source_path(
                st.session_state,
                path=path_value,
                kind=choice,
                lang=lang,
                database=selected_database,
            )
            if result is None:
                st.session_state["_assistant_notice"] = (
                    "Paste a folder path first."
                    if is_en else
                    "请先填写文件夹路径。"
                )
            else:
                content, actions = result
                user_content = (
                    f"Set {title}: `{path_value.strip()}`"
                    if is_en else
                    f"设置{title}：`{path_value.strip()}`"
                )
                messages = st.session_state.setdefault("llm_messages", [])
                if isinstance(messages, list):
                    messages.append({"role": "user", "content": user_content})
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "actions": actions,
                        "workflow_snapshot": _copilot_workflow_snapshot(st.session_state, lang),
                    })
                    _request_copilot_scroll_to_latest(st.session_state)
                st.session_state["_assistant_notice"] = (
                    "Data source saved in Copilot."
                    if is_en else
                    "数据源已在 Copilot 中保存。"
                )
            st.rerun()


def _render_copilot_cohort_inline_form(lang: str, panel_key: str) -> None:
    """Render classic Step 2 cohort controls inside the latest Copilot reply."""
    if st.session_state.get("_active_main_page") != "assistant":
        return
    raw_study = st.session_state.get("_copilot_guided_study")
    if not isinstance(raw_study, Mapping) or str(raw_study.get("step") or "") != "cohort":
        return
    is_en = lang == "en"
    current_filter = st.session_state.get("cohort_filter")
    if not isinstance(current_filter, Mapping):
        current_filter = _copilot_default_cohort_filter()
    disease_options = _copilot_full_disease_options(lang)
    current_disease = str(current_filter.get("disease_cohort") or "none")
    if current_disease not in disease_options:
        current_disease = "none"
    sepsis_mode_options = _copilot_sepsis_mode_options(lang)
    current_si_mode = str(st.session_state.get("sepsis_si_mode") or "auto")
    if current_si_mode not in sepsis_mode_options:
        current_si_mode = next(iter(sepsis_mode_options), "auto")
    current_first = current_filter.get("first_icu_stay")
    first_key = "yes" if current_first is True else "no" if current_first is False else "any"
    gender_key = str(current_filter.get("gender") or "any")
    if gender_key not in {"any", "M", "F"}:
        gender_key = "any"
    survived = current_filter.get("survived")
    survival_key = "survived" if survived is True else "deceased" if survived is False else "any"
    with st.container(key=f"{panel_key}_cohort_inline"):
        st.markdown(
            '<div class="eu-copilot-datasource-card eu-copilot-cohort-card">'
            f'<div class="ds-eyebrow">{html.escape("Cohort selection" if is_en else "队列筛选")}</div>'
            f'<div class="ds-title">{html.escape("Filter the analysis denominator" if is_en else "筛选分析队列分母")}</div>'
            f'<p>{html.escape("These controls mirror classic Step 2: demographics, first ICU stay, LOS, survival, and disease cohort." if is_en else "这些控件对应经典 Step 2：人口学、首次 ICU、ICU LOS、存活状态和疾病队列。")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.form(f"{panel_key}_cohort_filter_form", clear_on_submit=False):
            disease = st.selectbox(
                "Clinical cohort" if is_en else "疾病队列",
                options=list(disease_options),
                format_func=lambda value: disease_options.get(value, value),
                index=list(disease_options).index(current_disease),
                key=f"{panel_key}_cohort_disease_select",
            )
            row1 = st.columns(2, gap="small")
            with row1[0]:
                age_min = st.text_input(
                    "Min age" if is_en else "最小年龄",
                    value="" if current_filter.get("age_min") in {None, ""} else str(current_filter.get("age_min")),
                    placeholder="18",
                    key=f"{panel_key}_cohort_age_min",
                )
            with row1[1]:
                los_min = st.text_input(
                    "Min ICU LOS (h)" if is_en else "最短 ICU LOS (小时)",
                    value="" if current_filter.get("los_min") in {None, ""} else str(current_filter.get("los_min")),
                    placeholder="24",
                    key=f"{panel_key}_cohort_los_min",
                )
            row2 = st.columns(2, gap="small")
            with row2[0]:
                first_icu = st.selectbox(
                    "ICU stay" if is_en else "ICU stay",
                    options=["yes", "any", "no"],
                    format_func=lambda value: {
                        "yes": "First only" if is_en else "仅首次",
                        "any": "Any" if is_en else "不限",
                        "no": "Readmit" if is_en else "再入院",
                    }[value],
                    index=["yes", "any", "no"].index(first_key),
                    key=f"{panel_key}_cohort_first_icu_select",
                )
            with row2[1]:
                gender = st.selectbox(
                    "Sex" if is_en else "性别",
                    options=["any", "M", "F"],
                    format_func=lambda value: {
                        "any": "Any" if is_en else "不限",
                        "M": "Male" if is_en else "男性",
                        "F": "Female" if is_en else "女性",
                    }[value],
                    index=["any", "M", "F"].index(gender_key),
                    key=f"{panel_key}_cohort_gender_select",
                )
            survival = st.selectbox(
                "Outcome status" if is_en else "结局状态",
                options=["any", "survived", "deceased"],
                format_func=lambda value: {
                    "any": "Any" if is_en else "不限",
                    "survived": "Survived" if is_en else "存活",
                    "deceased": "Deceased" if is_en else "死亡",
                }[value],
                index=["any", "survived", "deceased"].index(survival_key),
                key=f"{panel_key}_cohort_survival_select",
            )
            # Tier-3: in-chat advanced disclosure of the remaining classic
            # Step 2 controls (max age, sepsis suspected-infection mode). These
            # write the same cohort_filter / session keys the classic view uses.
            with st.expander(
                "Advanced options" if is_en else "高级选项",
                expanded=False,
            ):
                adv1 = st.columns(2, gap="small")
                with adv1[0]:
                    age_max = st.text_input(
                        "Max age" if is_en else "最大年龄",
                        value="" if current_filter.get("age_max") in {None, ""} else str(current_filter.get("age_max")),
                        placeholder="89",
                        key=f"{panel_key}_cohort_age_max",
                    )
                with adv1[1]:
                    si_mode = st.selectbox(
                        "Sepsis definition" if is_en else "脓毒症判定",
                        options=list(sepsis_mode_options),
                        format_func=lambda value: sepsis_mode_options.get(value, value),
                        index=list(sepsis_mode_options).index(current_si_mode),
                        key=f"{panel_key}_cohort_si_mode",
                        help=(
                            "Suspected-infection rule used to build the Sepsis-3 cohort; only applies when the clinical cohort is Sepsis-3."
                            if is_en else
                            "构建 Sepsis-3 队列时的疑似感染判定规则；仅当疾病队列为脓毒症时生效。"
                        ),
                    )
                st.caption(
                    "Need the full classic cohort builder (ICD code filters, etc.)? Use Advanced · classic view below."
                    if is_en else
                    "需要完整经典队列构建器（ICD 代码过滤等）？用下方的「高级 · 经典视图」。"
                )
            submit_clicked = st.form_submit_button(
                "Save cohort" if is_en else "保存队列",
                type="primary",
                use_container_width=True,
            )
    if submit_clicked:
        content, actions = _copilot_submit_cohort_filter(
            st.session_state,
            disease=disease,
            age_min=age_min,
            age_max=age_max,
            los_min=los_min,
            first_icu=first_icu,
            gender=gender,
            survival=survival,
            si_mode=si_mode,
            lang=lang,
        )
        messages = st.session_state.setdefault("llm_messages", [])
        if isinstance(messages, list):
            messages.append({
                "role": "user",
                "content": (
                    "Configured cohort filters in Copilot."
                    if is_en else
                    "已在 Copilot 中配置队列筛选。"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": content,
                "actions": actions,
                "workflow_snapshot": _copilot_workflow_snapshot(st.session_state, lang),
            })
            _request_copilot_scroll_to_latest(st.session_state)
        st.session_state["_assistant_notice"] = (
            "Cohort filter saved in Copilot."
            if is_en else
            "队列筛选已在 Copilot 中保存。"
        )
        st.rerun()


def _render_copilot_feature_inline_form(lang: str, panel_key: str) -> None:
    """Render classic Step 3 feature-module controls inside the latest Copilot reply."""
    if st.session_state.get("_active_main_page") != "assistant":
        return
    raw_study = st.session_state.get("_copilot_guided_study")
    if not isinstance(raw_study, Mapping) or str(raw_study.get("step") or "") != "concepts":
        return
    is_en = lang == "en"
    selected_concepts = set(str(item) for item in list(raw_study.get("selected_concepts") or st.session_state.get("selected_concepts") or []))
    default_keys = [
        key for key, pack in COPILOT_FEATURE_MODULE_PACKS.items()
        if any(str(concept) in selected_concepts for concept in pack["concepts"])
    ]
    module_keys = _copilot_feature_inline_selected_keys(
        st.session_state,
        panel_key=panel_key,
        default_keys=default_keys,
    )
    selected_module_set = set(module_keys)
    with st.container(key=f"{panel_key}_feature_inline"):
        st.markdown(
            '<div class="eu-copilot-datasource-card eu-copilot-feature-card">'
            f'<div class="ds-eyebrow">{html.escape("Feature modules" if is_en else "特征模块")}</div>'
            f'<div class="ds-title">{html.escape("Select classic Step 3 modules" if is_en else "选择经典 Step 3 模块")}</div>'
            f'<p>{html.escape("Pick module groups here; Copilot writes the resulting concepts to selected_concepts for extraction and Agent handoff." if is_en else "在这里选择模块组；Copilot 会把对应概念写入 selected_concepts，用于提取和 Agent 交接。")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container(key=f"{panel_key}_feature_module_picker"):
            st.markdown(
                '<div class="eu-copilot-feature-toggle-grid-marker"></div>',
                unsafe_allow_html=True,
            )
            columns = st.columns(2, gap="small")
            for idx, key in enumerate(COPILOT_FEATURE_MODULE_PACKS):
                label = _copilot_feature_pack_label(key, lang)
                selected = key in selected_module_set
                concept_n = len(COPILOT_FEATURE_MODULE_PACKS[key].get("concepts") or [])
                button_label = f"{label}　{concept_n}" if concept_n else label
                row_state = "on" if selected else "off"
                with columns[idx % 2]:
                    with st.container(key=f"{panel_key}_feature_module_toggle_{row_state}_{key}"):
                        if st.button(
                            button_label,
                            key=f"{panel_key}_feature_module_button_{key}",
                            use_container_width=True,
                        ):
                            _copilot_toggle_feature_inline_module(
                                st.session_state,
                                panel_key=panel_key,
                                module_key=key,
                                default_keys=default_keys,
                            )
                            st.rerun()
            save_clicked = st.button(
                "Save modules" if is_en else "保存模块",
                key=f"{panel_key}_feature_module_save",
                type="secondary",
                use_container_width=True,
            )
    if save_clicked:
        result = _copilot_submit_feature_modules(st.session_state, module_keys=list(module_keys), lang=lang)
        if result is None:
            st.session_state["_assistant_notice"] = (
                "Choose at least one feature module."
                if is_en else
                "请至少选择一个特征模块。"
            )
        else:
            content, actions = result
            messages = st.session_state.setdefault("llm_messages", [])
            if isinstance(messages, list):
                messages.append({
                    "role": "user",
                    "content": (
                        "Configured feature modules in Copilot."
                        if is_en else
                        "已在 Copilot 中配置特征模块。"
                    ),
                })
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "actions": actions,
                    "workflow_snapshot": _copilot_workflow_snapshot(st.session_state, lang),
                })
                _request_copilot_scroll_to_latest(st.session_state)
            st.session_state["_assistant_notice"] = (
                "Feature modules saved in Copilot."
                if is_en else
                "特征模块已在 Copilot 中保存。"
            )
            st.rerun()


def _render_copilot_review_inline(lang: str, panel_key: str) -> None:
    """C3: in-chat visual review — summary + table picker + sample, no page jump.

    Embeds a compact review of the extracted cohort right in the conversation
    (instead of only routing to the classic Patient Review page). Reuses the
    SAME loaded frame + `_quick_viz_workspace_summary` the classic view uses; the
    full classic page stays available via the escape-hatch button.
    """
    if st.session_state.get("_active_main_page") != "assistant":
        return
    raw_study = st.session_state.get("_copilot_guided_study")
    if not isinstance(raw_study, Mapping) or str(raw_study.get("step") or "") != "review":
        return
    loaded = st.session_state.get("loaded_concepts")
    if not isinstance(loaded, Mapping) or not loaded:
        return  # nothing extracted yet; _copilot_submit_review guides to extract
    is_en = lang == "en"
    summary: Mapping[str, object] = {}
    try:
        from easyicu.webapp.quick_visualization_page import _quick_viz_workspace_summary
        summary = _quick_viz_workspace_summary(dict(st.session_state), lang) or {}
    except Exception:  # pragma: no cover - defensive UI guard
        summary = {}
    concept_count = int(summary.get("concept_count") or len(loaded))
    module_count = int(summary.get("module_count") or 0)
    patients = summary.get("loaded_patient_count") or summary.get("all_patient_count") or 0
    source_label = str(summary.get("source_label") or "")
    meta = (
        f"{concept_count} concept tables · {module_count} modules · {patients} patients · {source_label}"
        if is_en else
        f"{concept_count} 张概念表 · {module_count} 模块 · {patients} 位患者 · {source_label}"
    )
    with st.container(key=f"{panel_key}_review_inline"):
        st.markdown(
            '<div class="eu-copilot-datasource-card eu-copilot-review-card">'
            f'<div class="ds-eyebrow">{html.escape("Visual review" if is_en else "可视化审阅")}</div>'
            f'<div class="ds-title">{html.escape("Extracted cohort preview" if is_en else "提取队列预览")}</div>'
            f'<p>{html.escape(meta)}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        names = sorted(str(name) for name in loaded)
        selected = st.selectbox(
            "Preview table" if is_en else "预览表",
            options=names,
            index=0,
            key=f"{panel_key}_review_table_select",
        )
        frame = loaded.get(selected)
        if hasattr(frame, "head"):
            try:
                st.dataframe(frame.head(8), use_container_width=True, height=240)
            except Exception:  # pragma: no cover - defensive
                st.caption(
                    "Preview unavailable for this table." if is_en else "该表暂无法预览。"
                )
        if st.button(
            "Open full Patient Review" if is_en else "打开完整患者审阅",
            key=f"{panel_key}_review_open_full",
            use_container_width=True,
            icon=":material/open_in_new:",
        ):
            _apply_chat_workflow_action("study_review")
            st.rerun()


def _render_copilot_inline_step_controls(lang: str, panel_key: str) -> None:
    """Render the current guided form in the assistant message area."""
    _render_copilot_data_source_inline_form(lang, panel_key)
    _render_copilot_cohort_inline_form(lang, panel_key)
    _render_copilot_feature_inline_form(lang, panel_key)
    _render_copilot_review_inline(lang, panel_key)


def _render_copilot_scroll_to_latest_once(panel_key: str) -> None:
    """Scroll the Copilot chat viewport to the newest visible turn after rerun."""
    if not st.session_state.pop("_copilot_scroll_to_latest", False):
        return
    panel_key_json = json.dumps(panel_key)
    st.components.v1.html(
        f"""
        <script>
        (function() {{
          const panelKey = {panel_key_json};
          const doc = window.parent && window.parent.document;
          if (!doc) return;

          function scrollLatest() {{
            const history = doc.querySelector('div[class*="st-key-' + panelKey + '_history"]');
            const messages = Array.from(doc.querySelectorAll('[data-testid="stChatMessage"]'));
            const latest = messages.length ? messages[messages.length - 1] : null;
            const candidates = [];

            if (history) {{
              candidates.push(history);
              history.querySelectorAll('*').forEach((el) => {{
                const style = window.parent.getComputedStyle(el);
                if (
                  (style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                  el.scrollHeight > el.clientHeight + 4
                ) {{
                  candidates.push(el);
                }}
              }});
            }}

            const scroller = candidates.find((el) => el.scrollHeight > el.clientHeight + 4) || history;
            if (scroller) {{
              scroller.scrollTo({{ top: scroller.scrollHeight, behavior: 'smooth' }});
            }}
            if (latest) {{
              latest.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
            }}
          }}

          [80, 220, 520, 900].forEach((delay) => window.setTimeout(scrollLatest, delay));
        }})();
        </script>
        """,
        height=0,
    )


def _render_compact_chat_panel(
    *,
    lang: str,
    panel_key: str,
    history_height: int = 320,
    show_starters: bool = True,
    show_hint_chips: bool = True,
    welcome_variant: str = "rich",
    background_pending_prompts: bool = False,
) -> None:
    """Render a compact chat history + input form panel."""
    is_codex_workspace = welcome_variant == "codex"
    # Check for background response that completed while panel was minimized
    bg_result = _check_bg_response()
    if bg_result:
        st.session_state["_ai_bg_responding"] = False
        st.session_state["_ai_bg_session_id"] = None
        if bg_result.get("status") == "done":
            final_answer = bg_result["answer"]
            # Apply post-processing
            if st.session_state.llm_messages:
                last_user_msg = ""
                for m in reversed(st.session_state.llm_messages):
                    if m["role"] == "user":
                        last_user_msg = m["content"]
                        break
                final_answer = _append_quick_links(last_user_msg, final_answer, lang)
                response_actions = _suggest_ui_actions(last_user_msg, final_answer, lang)
            else:
                response_actions = []
            st.session_state.llm_messages.append({
                "role": "assistant",
                "content": final_answer,
                "actions": response_actions,
            })
            _request_copilot_scroll_to_latest(st.session_state)
        elif bg_result.get("status") == "error":
            error_msg = _handle_api_error(Exception(bg_result["answer"]), lang, render=False)
            st.session_state.llm_messages.append({
                "role": "assistant",
                "content": error_msg,
                "actions": [],
            })
            _request_copilot_scroll_to_latest(st.session_state)
        # Clear unread since panel is now open
        st.session_state["_ai_bg_unread_count"] = 0
        st.session_state["_ai_bg_response_ready"] = False

    if is_codex_workspace:
        st.markdown('<div class="eu-copilot-gd-conv-marker"></div>', unsafe_allow_html=True)

    history_container = st.container(
        height=history_height,
        border=not is_codex_workspace,
        key=f"{panel_key}_history" if is_codex_workspace else None,
    )
    with history_container:
        recent_messages = st.session_state.llm_messages[-COPILOT_RENDER_MESSAGE_LIMIT:]
        latest_assistant_idx = -1
        if is_codex_workspace:
            for idx, item in enumerate(recent_messages):
                if str(item.get("role") or "") == "assistant":
                    latest_assistant_idx = idx
        queued_prompt = st.session_state.pop("_ai_pending_question", None)

        if not recent_messages and not queued_prompt:
            _render_chat_welcome(
                lang=lang,
                panel_key=panel_key,
                history_container=history_container,
                show_starters=show_starters,
                welcome_variant=welcome_variant,
            )
        else:
            for msg_idx, msg in enumerate(recent_messages):
                role = str(msg.get("role") or "assistant")
                avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
                with st.chat_message(role, avatar=avatar):
                    content = str(msg.get("content") or "")
                    if role == "user" and not content.strip() and is_codex_workspace:
                        content = str(msg.get("display_content") or msg.get("label") or "").strip()
                        if not content:
                            content = "Selected option" if lang == "en" else "已选择一个选项"
                    if role == "assistant" and is_codex_workspace:
                        content = _normalized_copilot_message_content(content, lang)
                    st.markdown(content)
                    rendered_actions = False
                    if role == "assistant" and is_codex_workspace:
                        actions = _copilot_message_actions_for_current_step(
                            msg.get("actions"),
                            lang,
                            st.session_state,
                            is_latest=msg_idx == latest_assistant_idx,
                        )
                        _render_nav_actions(actions, key_prefix=f"{panel_key}_{msg_idx}")
                        rendered_actions = True
                    elif role == "assistant" and msg.get("actions"):
                        actions = msg["actions"]
                    if role == "assistant" and is_codex_workspace:
                        if msg_idx == latest_assistant_idx:
                            _render_copilot_inline_step_controls(lang, panel_key)
                            _render_copilot_workflow_snapshot(
                                msg.get("workflow_snapshot"),
                                lang,
                                key_prefix=f"{panel_key}_{msg_idx}",
                            )
                    if role == "assistant" and msg.get("actions") and not rendered_actions:
                        _render_nav_actions(actions, key_prefix=f"{panel_key}_{msg_idx}")
            if queued_prompt:
                status_text = (
                    "Using page context to ask the assistant..."
                    if lang == "en" else
                    "正在带着当前页面上下文向 AI 提问..."
                )
                st.markdown(
                    f'<div class="inline-ai-status-strip">{html.escape(status_text)}</div>',
                    unsafe_allow_html=True,
                )
                if background_pending_prompts:
                    _submit_prompt_background(
                        queued_prompt,
                        lang,
                        history_container,
                        key_prefix=panel_key,
                        display_prompt=st.session_state.pop("_ai_pending_question_display", None),
                    )
                else:
                    _submit_prompt(
                        queued_prompt,
                        lang,
                        history_container,
                        key_prefix=panel_key,
                        display_prompt=st.session_state.pop("_ai_pending_question_display", None),
                    )
            elif st.session_state.get("_ai_bg_responding", False):
                status_text = (
                    "Generating response in the background. You can continue elsewhere in EasyICU."
                    if lang == "en" else
                    "正在后台生成回答。你可以继续使用 EasyICU 的其他页面。"
                )
                st.markdown(
                    f'<div class="inline-ai-status-strip">{html.escape(status_text)}</div>',
                    unsafe_allow_html=True,
                )

    composer_host = st.container(key=f"{panel_key}_composer_wrap") if is_codex_workspace else contextlib.nullcontext()
    with composer_host:
        if show_hint_chips and is_codex_workspace:
            _render_copilot_primary_chips(lang, panel_key)
        if show_hint_chips:
            _render_copilot_hint_chips(lang, panel_key)

        with st.form(f"{panel_key}_form", clear_on_submit=True):
            input_col, send_col = st.columns([1, 0.065], gap="small")
            with input_col:
                placeholder = (
                    'Reply, ask “why?”, choose cohort, or ask about the data path...'
                    if is_codex_workspace and lang == "en" else
                    "Ask about the current workflow..."
                    if lang == "en" else
                    "回复、问“为什么”、选择队列，或询问真实数据路径..."
                    if is_codex_workspace else
                    "询问当前流程、概念或报错..."
                )
                prompt = st.text_input(
                    "Ask EasyICU AI" if lang == "en" else "向 EasyICU AI 提问",
                    placeholder=placeholder,
                    label_visibility="collapsed",
                )
            with send_col:
                send_clicked = st.form_submit_button(
                    "Send" if lang == "en" else "发送",
                    type="primary",
                    use_container_width=True,
                )
        if is_codex_workspace:
            foot_note = (
                "Real-data first · reproducible · nothing leaves your machine"
                if lang == "en" else
                "真实数据优先 · 可复现 · 数据不离开本机"
            )
            st.markdown(
                f'<div class="eu-copilot-composer-foot">{html.escape(foot_note)}</div>',
                unsafe_allow_html=True,
            )
    if send_clicked and prompt.strip():
        st.session_state["_ai_pending_question"] = prompt.strip()
        st.session_state.pop("_ai_pending_question_display", None)
        st.rerun()

    if is_codex_workspace:
        _render_copilot_scroll_to_latest_once(panel_key)

    if st.session_state.llm_messages and not is_codex_workspace:
        action_cols = st.columns(2)
        with action_cols[0]:
            st.download_button(
                "Export Chat" if lang == "en" else "导出对话",
                data=_build_chat_export_text(),
                file_name=f"easyicu_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"{panel_key}_export_chat",
            )
        with action_cols[1]:
            if st.button(
                "Clear Chat" if lang == "en" else "清空对话",
                key=f"{panel_key}_clear_chat",
                use_container_width=True,
                icon=":material/delete:",
            ):
                st.session_state.llm_messages = []
                st.session_state["_ai_pending_question"] = None
                st.rerun()


def _floating_copilot_route(state: Mapping[str, object] | None = None) -> str:
    """Return the active workspace route used for dock context chips."""
    if state is None:
        state = st.session_state
    route = str(state.get("_active_main_page") or state.get("_scroll_to_tab") or "entry")
    return route if route else "entry"


def _floating_copilot_context_intro(
    lang: str,
    state: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Return the polish dock label and page-specific greeting for the active route."""
    is_en = lang == "en"
    route = _floating_copilot_route(state)
    by_route = {
        "entry": {
            "label_en": "Home",
            "label_zh": "首页",
            "hi_en": (
                "I am your EasyICU companion. I can explain any screen, drive it for you, "
                "or run a whole study by chat."
            ),
            "hi_zh": "我是你的 EasyICU 伴随助手，可以解释当前屏幕、帮你驱动工作区，或用聊天跑完整研究。",
        },
        "extract": {
            "label_en": "Data Extraction",
            "label_zh": "数据提取",
            "hi_en": (
                "You are in Data Extraction, the four-step gate that turns a source into "
                "analysis-ready frames."
            ),
            "hi_zh": "你正在 Data Extraction，这是把数据源转换为 analysis-ready frames 的四步闸门。",
        },
        "quick_viz": {
            "label_en": "Patient Review",
            "label_zh": "患者审阅",
            "hi_en": (
                "This is Patient Review: tables, time series, patient overview, and data-quality flags. "
                "I can load a demo workspace so it is populated."
            ),
            "hi_zh": "这里是 Patient Review：表格、时间序列、患者概览和数据质量标记。我可以加载演示工作区。",
        },
        "cohort": {
            "label_en": "Cohort Statistics",
            "label_zh": "队列统计",
            "hi_en": (
                "Cohort Statistics compares groups, audits coverage, and reclassifies SOFA. "
                "I can re-run it or explain what you are seeing."
            ),
            "hi_zh": "Cohort Statistics 用于组间比较、覆盖率审计和 SOFA 重分类。我可以重跑统计或解释当前面板。",
        },
        "cross_db": {
            "label_en": "Cross-DB Benchmark",
            "label_zh": "跨库 Benchmark",
            "hi_en": (
                "Cross-DB Benchmark applies one cohort definition across ICU databases. "
                "I can load it or explain where databases diverge."
            ),
            "hi_zh": "Cross-DB Benchmark 把同一个队列定义应用到多个 ICU 数据库。我可以加载 benchmark 或解释数据库差异。",
        },
        "research_agent": {
            "label_en": "Research Agent",
            "label_zh": "Research Agent",
            "hi_en": (
                "The Research Agent runs an auditable pipeline and drafts findings, while the draft "
                "stays gated until checks pass."
            ),
            "hi_zh": "Research Agent 运行可审计 pipeline 并起草发现；检查通过前，草稿会保持闸门锁定。",
        },
        "states": {
            "label_en": "Workspace States",
            "label_zh": "工作区状态",
            "hi_en": (
                "This is the states reference: loading, empty, no-data, error, blocked, and success. "
                "Ask me when each one shows."
            ),
            "hi_zh": "这里是工作区状态库：loading、empty、no-data、error、blocked、success。你可以问每种状态何时出现。",
        },
        "settings": {
            "label_en": "Settings",
            "label_zh": "设置",
            "hi_en": "Settings are local-first and reversible. Ask me what any option does.",
            "hi_zh": "Settings 遵循本地优先，且设置可逆。你可以问任一选项的作用。",
        },
        "tutorial": {
            "label_en": "Get Started",
            "label_zh": "开始使用",
            "hi_en": "Get Started orients you to the workflow. I can run the whole thing by chat instead.",
            "hi_zh": "Get Started 用来理解工作流；我也可以直接用聊天替你跑完整流程。",
        },
        "assistant": {
            "label_en": "Research Copilot",
            "label_zh": "Research Copilot",
            "hi_en": "The full guided Copilot is open. Use it to move from question to evidence-gated draft.",
            "hi_zh": "完整引导式 Copilot 已打开，可从研究问题推进到 evidence-gated draft。",
        },
    }
    current = by_route.get(route, by_route["entry"])
    return {
        "route": route,
        "label": current["label_en"] if is_en else current["label_zh"],
        "hi": current["hi_en"] if is_en else current["hi_zh"],
    }


def _floating_copilot_context_chips(
    lang: str,
    state: Mapping[str, object] | None = None,
) -> list[dict[str, str]]:
    """Return contextual dock chips adapted from the polish(2) dock design."""
    is_en = lang == "en"
    route = _floating_copilot_route(state)

    def answer(action_id: str, label_en: str, label_zh: str, answer_id: str) -> dict[str, str]:
        return {
            "id": action_id,
            "kind": "answer",
            "label": label_en if is_en else label_zh,
            "answer_id": answer_id,
        }

    def workflow(action_id: str, label_en: str, label_zh: str, target: str) -> dict[str, str]:
        return {
            "id": action_id,
            "kind": "workflow",
            "label": label_en if is_en else label_zh,
            "workflow": target,
        }

    by_route: dict[str, list[dict[str, str]]] = {
        "extract": [
            answer("explain_extract", "Explain the 4 steps", "解释 4 个步骤", "extract"),
            answer("explain_export", "What gets exported?", "导出包含什么？", "export"),
            workflow("guided_study", "Run a guided study", "运行引导式研究", "guided_demo"),
        ],
        "quick_viz": [
            workflow("demo_review", "Load demo workspace", "加载演示工作区", "demo_review"),
            answer("explain_tabs", "What's in each tab?", "每个标签页有什么？", "tabs"),
            answer("explain_quality", "Explain data-quality flags", "解释数据质量标记", "quality"),
        ],
        "cohort": [
            workflow("cohort_run", "Re-run statistics", "重新运行统计", "cohort_run"),
            answer("explain_sofa", "Explain SOFA reclassification", "解释 SOFA 重分类", "sofa"),
            answer("explain_contrast", "What's the comparison?", "当前比较是什么？", "contrast"),
        ],
        "cross_db": [
            workflow("load_crossdb", "Load the benchmark", "加载 benchmark", "crossdb_demo"),
            answer("explain_overlap", "Which databases overlap?", "哪些数据库重叠？", "overlap"),
            workflow("guided_crossdb", "Run a guided comparison", "运行引导式跨库比较", "guided_crossdb_demo"),
        ],
        "research_agent": [
            answer("explain_gate", "Why is the draft locked?", "为什么草稿锁定？", "gate"),
            answer("explore_idea", "Explore ideas in chat", "在聊天里探索 idea", "idea"),
            workflow("show_completed_run", "Show a completed run", "显示已完成运行", "agent_completed_run"),
        ],
        "states": [
            answer("explain_states", "When do states show?", "状态何时出现？", "states"),
            workflow("guided_study", "Start a guided study", "开始引导式研究", "guided_demo"),
        ],
        "tutorial": [
            workflow("guided_study", "Run a guided study", "运行引导式研究", "guided_demo"),
            answer("explain_how", "How does EasyICU work?", "EasyICU 如何工作？", "how"),
        ],
        "settings": [
            answer("explain_privacy", "Is my data uploaded?", "我的数据会上传吗？", "privacy"),
            answer("explain_gate", "Explain the evidence gate", "解释证据闸门", "gate"),
        ],
    }
    return by_route.get(route, [
        workflow("guided_study", "Start a guided study", "开始引导式研究", "guided_demo"),
        answer("explain_how", "How does EasyICU work?", "EasyICU 如何工作？", "how"),
        workflow("demo_review", "Load a demo workspace", "加载演示工作区", "demo_review"),
    ])[:3]


def _floating_copilot_text_intent(
    prompt: str,
    lang: str,
    state: Mapping[str, object] | None = None,
) -> dict[str, str] | None:
    """Map short dock free-text commands to the polish(2) local dock behavior."""
    text = (prompt or "").strip()
    if not text:
        return None
    text_l = text.lower()
    route = _floating_copilot_route(state)
    is_en = lang == "en"

    def workflow(target: str, content_en: str, content_zh: str) -> dict[str, str]:
        return {
            "kind": "workflow",
            "workflow": target,
            "content": content_en if is_en else content_zh,
        }

    def answer(answer_id: str) -> dict[str, str]:
        return {
            "kind": "answer",
            "answer_id": answer_id,
            "content": _floating_copilot_answer_text(answer_id, lang),
        }

    if _copilot_research_recommendation_requested(text) or _is_idea_exploration_request(text):
        idea_state = state if isinstance(state, MutableMapping) else {}
        content, _actions = _copilot_idea_recommendation_reply(text, lang, idea_state)
        return {
            "kind": "answer",
            "answer_id": "idea",
            "content": content,
        }

    if re.search(r"\b(guided|whole|run it|do it for me|walk me)\b", text_l) or any(
        term in text for term in ("引导", "完整演示", "跑完整", "帮我跑", "带我")
    ):
        return workflow(
            "guided_demo",
            "Opening the guided Copilot — it will run the full study by chat.",
            "正在打开引导式 Copilot，它会用聊天方式跑完整研究流程。",
        )

    if route == "research_agent" and (
        re.search(r"\b(completed run|show.*run|agent run|summary gate|open.*run)\b", text_l)
        or any(term in text for term in ("已完成运行", "完成的运行", "打开运行", "显示运行", "Summary"))
    ):
        return workflow(
            "agent_completed_run",
            "Opening a completed Research Agent run in Summary.",
            "正在 Summary 中打开一个已完成的 Research Agent 运行。",
        )

    if route == "cross_db" and (
        re.search(r"\b(load|demo|populate|benchmark|cross[- ]?db|cross database)\b", text_l)
        or any(term in text for term in ("加载", "演示", "benchmark", "跨库"))
    ):
        return workflow(
            "crossdb_demo",
            "Loaded the benchmark — Cross-DB is open with the availability matrix.",
            "已加载 benchmark，Cross-DB 页面会显示可用性矩阵。",
        )

    if route == "cohort" and (
        re.search(r"\b(re[- ]?run|rerun|run|refresh|statistics|stats|recompute|cohort run)\b", text_l)
        or any(term in text for term in ("重新运行", "重跑", "刷新", "统计", "队列统计"))
    ):
        return workflow(
            "cohort_run",
            "Re-running cohort statistics — the Cohort page will refresh all panels.",
            "正在重新运行队列统计，Cohort 页面会刷新所有面板。",
        )

    if route == "cohort" and re.search(r"\bsofa|reclassification\b", text_l):
        return answer("sofa")

    if route == "cohort" and (
        re.search(r"\b(comparison|compare|contrast|group|survived|deceased)\b", text_l)
        or any(term in text for term in ("比较", "对比", "分组", "存活", "死亡"))
    ):
        return answer("contrast")

    if route == "extract" and (
        re.search(r"\b(4 steps?|four steps?|extract(?:ion)? steps?|what gets exported|export bundle|manifest)\b", text_l)
        or any(term in text for term in ("四步", "4 步", "4步", "导出包含", "导出什么", "manifest"))
    ):
        if re.search(r"\b(export|exported|bundle|manifest)\b", text_l) or any(
            term in text for term in ("导出", "manifest")
        ):
            return answer("export")
        return answer("extract")

    if route == "cross_db" and (
        re.search(r"\b(overlap|availability|matrix|which databases|shared concepts?|missing concepts?)\b", text_l)
        or any(term in text for term in ("重叠", "可用性", "矩阵", "共享概念", "缺失概念"))
    ):
        return answer("overlap")

    if route == "states" and (
        re.search(r"\b(state|states|loading|empty|no[- ]?data|blocked|success|error)\b", text_l)
        or any(term in text for term in ("状态", "加载", "空", "无数据", "阻塞", "成功", "错误"))
    ):
        return answer("states")

    if route == "quick_viz" and (
        re.search(r"\b(tab|tabs|each tab|time series|patient overview|data tables?)\b", text_l)
        or any(term in text for term in ("标签", "标签页", "时间序列", "患者概览", "数据表"))
    ):
        return answer("tabs")

    if re.search(r"\b(load|demo|populate|show me data)\b", text_l) or any(
        term in text for term in ("加载", "演示", "填充", "看看数据", "显示数据")
    ):
        return workflow(
            "demo_review",
            "Loaded a demo workspace — Patient Review is populated.",
            "已加载演示工作区，Patient Review 会填充演示数据。",
        )

    if re.search(r"\b(privacy|upload|local|phi|leave(?:s)? my machine)\b", text_l) or any(
        term in text for term in ("隐私", "上传", "本机", "离开", "患者行")
    ):
        return answer("privacy")

    if re.search(r"\b(gate|lock|locked|draft|sign|sign-off|signoff)\b", text_l) or any(
        term in text for term in ("闸门", "锁", "草稿", "签字", "确认")
    ):
        return answer("gate")

    if re.search(r"\b(quality|missing|coverage|flag|sparse|audit)\b", text_l) or any(
        term in text for term in ("质量", "缺失", "覆盖", "稀疏", "审计", "旗标")
    ):
        return answer("quality")

    if re.search(r"\b(how|what is|explain|work)\b", text_l) or any(
        term in text for term in ("怎么", "如何", "解释", "是什么")
    ):
        return answer("how")

    return None


def _apply_floating_copilot_text_intent(prompt: str, lang: str) -> bool:
    """Apply a local dock text intent before falling through to LLM chat."""
    state = st.session_state
    intent = _floating_copilot_text_intent(prompt, lang, state)
    if intent is None:
        return False

    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list):
        messages.append({"role": "user", "content": prompt.strip()})
        messages.append({
            "role": "assistant",
            "content": str(intent.get("content") or ""),
            "actions": [],
        })
    state["llm_last_tool_events"] = []
    state["llm_last_verification"] = {"status": "pass", "issues": []}
    if intent.get("kind") == "workflow":
        _apply_chat_workflow_action(str(intent.get("workflow") or ""))
    else:
        state["_floating_ai_open"] = True
        state.pop("_ai_pending_question", None)
    return True


def _render_floating_copilot_context_actions(lang: str) -> None:
    """Render compact context and workspace-driving actions for the global dock."""
    is_en = lang == "en"
    st.markdown(_inline_ai_context_html(lang), unsafe_allow_html=True)
    intro = _floating_copilot_context_intro(lang)
    st.markdown(
        '<div class="inline-ai-evidence-note">'
        '<b>' + html.escape(("on " if is_en else "当前页面: ") + intro["label"]) + '</b>'
        '<p>' + html.escape(intro["hi"]) + '</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    chips = _floating_copilot_context_chips(lang)
    if chips:
        st.markdown(
            '<div class="floating-ai-welcome-hint">'
            + html.escape("Context shortcuts" if is_en else "当前页面快捷提问")
            + '</div>',
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(len(chips), gap="small")
        for chip_idx, chip in enumerate(chips):
            with chip_cols[chip_idx]:
                if st.button(
                    str(chip["label"]),
                    key=f"_floating_ai_chip_{chip['id']}",
                    use_container_width=True,
                ):
                    if chip.get("kind") == "workflow":
                        _apply_chat_workflow_action(str(chip.get("workflow") or ""))
                    else:
                        _append_floating_copilot_local_answer(
                            st.session_state,
                            label=str(chip["label"]),
                            answer_id=str(chip.get("answer_id") or "how"),
                            lang=lang,
                        )
                    st.rerun()
    action_cols = st.columns(3, gap="small")
    with action_cols[0]:
        if st.button(
            "Full Copilot" if is_en else "完整 Copilot",
            key="_floating_ai_full_copilot",
            use_container_width=True,
        ):
            st.session_state["llm_enabled"] = True
            st.session_state["_llm_toggle"] = True
            st.session_state["_active_main_page"] = "assistant"
            st.session_state["_floating_ai_open"] = False
            st.session_state["_scroll_to_top"] = True
            st.rerun()
    with action_cols[1]:
        if st.button(
            "Demo review" if is_en else "演示审阅",
            key="_floating_ai_demo_review",
            use_container_width=True,
            icon=":material/preview:",
        ):
            _apply_chat_workflow_action("demo_review")
            st.rerun()
    with action_cols[2]:
        if st.button(
            "Agent setup" if is_en else "Agent 设置",
            key="_floating_ai_agent_setup",
            use_container_width=True,
            icon=":material/assignment:",
        ):
            _prepare_research_agent_handoff_from_ai(st.session_state)
            st.rerun()


def render_floating_chat_dock(app_context: dict | None = None):
    """Render a fixed bottom-right floating AI chat dock."""
    _init_chat_state()
    if app_context is not None:
        st.session_state["_app_context"] = app_context
    lang = st.session_state.get("language", "en")
    if "_floating_ai_open" not in st.session_state:
        st.session_state["_floating_ai_open"] = False
    if "_floating_ai_size" not in st.session_state:
        st.session_state["_floating_ai_size"] = "m"
    if st.session_state.get("_ai_pending_question"):
        st.session_state["_floating_ai_open"] = True

    # Defaults match easyicu design/page-ai-chat.jsx: 420 × 780 panel.
    size_presets = {
        "s": {
            "panel_width": "clamp(320px, 28vw, 380px)",
            "panel_max_height": "min(72vh, 640px)",
            "history_height": 280,
        },
        "m": {
            "panel_width": "420px",
            "panel_max_height": "min(86vh, 780px)",
            "history_height": 420,
        },
        "l": {
            "panel_width": "clamp(460px, 38vw, 640px)",
            "panel_max_height": "min(92vh, 960px)",
            "history_height": 560,
        },
    }
    size_key = st.session_state.get("_floating_ai_size", "m")
    if size_key not in size_presets:
        size_key = "m"
        st.session_state["_floating_ai_size"] = size_key
    size_cfg = size_presets[size_key]
    export_locked = bool(st.session_state.get("_exporting_in_progress", False))

    st.markdown(
        f"""
        <style>
        :root {{
            --easyicu-ai-launcher-size: clamp(56px, 4.2vw, 78px);
            --easyicu-ai-panel-width: {size_cfg["panel_width"]};
            --easyicu-ai-panel-max-height: {size_cfg["panel_max_height"]};
            --easyicu-ai-title-size: clamp(0.92rem, 0.35vw + 0.84rem, 1.06rem);
            --easyicu-ai-subtitle-size: clamp(0.72rem, 0.18vw + 0.68rem, 0.82rem);
            --easyicu-ai-body-size: clamp(0.82rem, 0.16vw + 0.78rem, 0.93rem);
            --easyicu-ai-button-size: clamp(2rem, 2vw, 2.45rem);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* ---- Animation keyframes ---- */
        @keyframes easyicuPanelSlideIn {
            from { opacity: 0; transform: translateY(24px) scale(0.96); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes easyicuPanelSlideOut {
            from { opacity: 1; transform: translateY(0) scale(1); }
            to   { opacity: 0; transform: translateY(24px) scale(0.96); }
        }
        @keyframes easyicuLauncherPop {
            0%   { transform: scale(0.5); opacity: 0; }
            70%  { transform: scale(1.1); }
            100% { transform: scale(1); opacity: 1; }
        }
        @keyframes easyicuBadgePulse {
            0%, 100% { transform: scale(1); }
            50%      { transform: scale(1.2); }
        }
        @keyframes easyicuBadgeBounce {
            0%   { transform: scale(0); }
            50%  { transform: scale(1.3); }
            70%  { transform: scale(0.9); }
            100% { transform: scale(1); }
        }

        div.st-key-floating_ai_launcher,
        div.st-key-floating_ai_panel {
            position: fixed !important;
            z-index: 99990 !important;
            margin: 0 !important;
        }

        div.st-key-floating_ai_launcher {
            right: clamp(12px, 1.25vw, 20px);
            bottom: clamp(12px, 1.25vw, 20px);
            width: calc(var(--easyicu-ai-launcher-size) + 12px);
            animation: easyicuLauncherPop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        }

        /* Notification badge on the launcher button — design token bad. */
        div.st-key-floating_ai_launcher .ai-notif-badge {
            position: absolute;
            top: -4px; right: 2px;
            min-width: 18px; height: 18px;
            border-radius: 9px;
            background: var(--bad, #b54f3e);
            color: #fff;
            font-size: 10.5px;
            font-weight: 500;
            display: flex; align-items: center; justify-content: center;
            padding: 0 5px;
            box-shadow: var(--sh-1, 0 1px 3px rgba(14,17,22,0.15));
            animation: easyicuBadgeBounce 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) both;
            z-index: 99999;
            pointer-events: none;
        }
        div.st-key-floating_ai_launcher .ai-notif-badge.pulse {
            animation: easyicuBadgePulse 1.5s ease-in-out infinite;
        }

        div.st-key-floating_ai_launcher .stButton > button {
            width: var(--easyicu-ai-launcher-size);
            min-width: var(--easyicu-ai-launcher-size);
            height: var(--easyicu-ai-launcher-size);
            border-radius: 999px;
            border: 1px solid var(--ink);
            background: var(--ink) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-size: clamp(1.05rem, 0.8vw + 0.9rem, 1.35rem);
            box-shadow: var(--sh-3, 0 18px 44px rgba(14,17,22,0.18));
        }


        /* Floating panel — page-ai-chat.jsx 420×780 surface card. */
        div.st-key-floating_ai_panel {
            right: clamp(12px, 1.25vw, 20px);
            bottom: clamp(12px, 1.25vw, 20px);
            width: min(var(--easyicu-ai-panel-width), calc(100vw - 24px));
            max-width: calc(100vw - 24px);
            max-height: var(--easyicu-ai-panel-max-height);
            border-radius: 14px;
            border: 1px solid var(--hair-2);
            background: var(--surface);
            box-shadow: var(--sh-3, 0 24px 58px rgba(14, 17, 22, 0.16));
            backdrop-filter: none;
            padding: 0;
            overflow-x: hidden;
            overflow-y: auto;
            animation: easyicuPanelSlideIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both;
            color: var(--ink);
        }

        div.st-key-floating_ai_panel .floating-ai-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 14px 8px;
            border-bottom: 1px solid var(--hair);
        }

        div.st-key-floating_ai_panel .floating-ai-title-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        div.st-key-floating_ai_panel .floating-ai-avatar {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-ink);
            background: var(--accent-soft);
            border: 1px solid var(--accent-border);
            box-shadow: none;
            font-size: 14px;
            flex: 0 0 auto;
        }

        div.st-key-floating_ai_panel .floating-ai-title {
            font-size: 13.5px;
            font-weight: 500;
            color: var(--ink);
            letter-spacing: 0;
        }

        div.st-key-floating_ai_panel .floating-ai-subtitle {
            font-size: 10.5px;
            color: var(--ink-4);
            margin-top: 2px;
            line-height: 1.35;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome {
            background: var(--accent-soft);
            border: 1px solid var(--accent-border);
            border-left: 3px solid var(--accent);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
            box-shadow: none;
            color: var(--accent-ink);
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-title {
            font-size: 12.5px;
            font-weight: 500;
            color: var(--ink);
            margin-bottom: 4px;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-subtitle {
            font-size: 11.5px;
            line-height: 1.5;
            color: var(--ink-2);
            margin-bottom: 8px;
        }

        div.st-key-floating_ai_panel .floating-ai-sample {
            display: grid;
            gap: 6px;
            margin: 6px 0 8px;
        }

        /* page-ai-chat user bubble: ink fill, white text */
        div.st-key-floating_ai_panel .floating-ai-user-bubble {
            justify-self: end;
            max-width: 85%;
            color: #fff;
            background: var(--ink);
            border: 1px solid var(--ink);
            border-radius: 10px;
            padding: 8px 12px;
            font-size: 12.5px;
            line-height: 1.55;
            box-shadow: none;
        }

        /* page-ai-chat agent bubble: surface, hair border */
        div.st-key-floating_ai_panel .floating-ai-answer-card {
            color: var(--ink);
            background: var(--surface);
            border: 1px solid var(--hair);
            border-radius: 10px;
            padding: 8px 12px;
            font-size: 12.5px;
            line-height: 1.55;
            box-shadow: none;
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation {
            display: grid;
            gap: 2px;
            color: var(--ink-2);
            background: var(--surface-2);
            border: 1px solid var(--hair);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 11px;
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation span {
            color: var(--ink-4);
            font-size: 9.5px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation strong {
            color: var(--ink);
            font-weight: 500;
            line-height: 1.42;
        }

        div.st-key-floating_ai_panel .inline-ai-context-card,
        div.st-key-floating_ai_panel .inline-ai-evidence-note {
            border-radius: 10px;
            border: 1px solid var(--hair);
            background: var(--surface-2);
            padding: 10px 12px;
            margin: 8px 8px 10px;
        }

        div.st-key-floating_ai_panel .inline-ai-context-card h3,
        div.st-key-floating_ai_panel .inline-ai-evidence-note b {
            display: block;
            margin: 0;
            color: var(--ink);
            font-size: 12.5px;
            font-weight: 600;
            line-height: 1.35;
        }

        div.st-key-floating_ai_panel .inline-ai-context-card p,
        div.st-key-floating_ai_panel .inline-ai-evidence-note p {
            margin: 4px 0 0;
            color: var(--ink-3);
            font-size: 11.5px;
            line-height: 1.45;
        }

        div.st-key-floating_ai_panel .inline-ai-section-label {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 9.5px;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        div.st-key-floating_ai_panel .inline-ai-tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 8px;
        }

        div.st-key-floating_ai_panel .inline-ai-tag-row span {
            border-radius: 999px;
            border: 1px solid var(--hair);
            background: var(--surface);
            color: var(--ink-3);
            font-size: 10.5px;
            padding: 2px 7px;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-hint {
            font-size: 10px;
            color: var(--accent-ink);
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        div.st-key-floating_ai_panel div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.2rem 0.3rem 0.4rem 0.3rem;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessage"] {
            margin-bottom: 0.45rem;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] {
            border-radius: 10px;
            padding: 8px 12px;
            background: var(--surface);
            border: 1px solid var(--hair);
            box-shadow: none;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] p,
        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] li {
            font-size: 12.5px;
            line-height: 1.55;
            color: var(--ink);
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] ul,
        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] ol {
            padding-left: 1.15rem;
            margin-top: 0.45rem;
            margin-bottom: 0.2rem;
        }

        div.st-key-floating_ai_panel .stButton > button {
            border-radius: 6px;
            min-height: 28px;
            font-size: 11.5px;
            padding: 4px 10px;
            border: 1px solid var(--hair-2) !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            background: var(--surface) !important;
            box-shadow: none !important;
        }
        div.st-key-floating_ai_panel .stButton > button:hover {
            background: var(--surface-2) !important;
            border-color: var(--hair-3) !important;
        }

        div.st-key-floating_ai_panel .stButton > button[kind="primary"],
        div.st-key-floating_ai_panel .stButton > button[data-testid="stBaseButton-primary"],
        div.st-key-floating_ai_panel form button,
        div.st-key-floating_ai_panel div[data-testid="stFormSubmitButton"] button {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border: 1px solid var(--ink) !important;
            background: var(--ink) !important;
            box-shadow: none !important;
        }

        div.st-key-floating_ai_panel .stButton > button[kind="primary"] *,
        div.st-key-floating_ai_panel .stButton > button[data-testid="stBaseButton-primary"] *,
        div.st-key-floating_ai_panel form button *,
        div.st-key-floating_ai_panel div[data-testid="stFormSubmitButton"] button *,
        div.st-key-floating_ai_panel .stButton > button:hover * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        div.st-key-floating_ai_launcher .stButton > button:hover {
            filter: brightness(1.02);
        }

        div.st-key-floating_ai_panel .stChatInput {
            margin-top: 0.3rem;
        }

        div.st-key-floating_ai_panel input,
        div.st-key-floating_ai_panel textarea,
        div.st-key-floating_ai_panel label,
        div.st-key-floating_ai_panel [data-testid="stMarkdownContainer"] p,
        div.st-key-floating_ai_panel [data-testid="stCaptionContainer"] {
            font-size: var(--easyicu-ai-body-size) !important;
        }

        @media (max-width: 1512px) {
            :root {
                --easyicu-ai-panel-width: clamp(330px, 31vw, 540px);
                --easyicu-ai-panel-max-height: min(80vh, 740px);
                --easyicu-ai-title-size: clamp(0.88rem, 0.26vw + 0.82rem, 0.98rem);
                --easyicu-ai-subtitle-size: clamp(0.68rem, 0.14vw + 0.64rem, 0.76rem);
                --easyicu-ai-body-size: clamp(0.78rem, 0.12vw + 0.75rem, 0.86rem);
                --easyicu-ai-button-size: clamp(1.86rem, 1.7vw, 2.2rem);
            }

            div.st-key-floating_ai_panel {
                border-radius: 16px;
                padding: 0.28rem 0.28rem 0.34rem 0.28rem;
            }

            div.st-key-floating_ai_panel .floating-ai-welcome {
                padding: 0.8rem 0.86rem;
                margin-bottom: 0.6rem;
            }
        }

        @media (max-width: 1280px) {
            :root {
                --easyicu-ai-panel-width: clamp(320px, 29vw, 500px);
                --easyicu-ai-panel-max-height: min(78vh, 680px);
                --easyicu-ai-body-size: 0.8rem;
            }
        }

        @media (max-width: 768px) {
            div.st-key-floating_ai_launcher {
                right: 12px;
                bottom: 12px;
            }

            div.st-key-floating_ai_panel {
                left: 12px;
                right: 12px;
                bottom: 82px;
                width: auto;
                max-height: min(74vh, 640px);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Check for unread background responses
    _unread = st.session_state.get("_ai_bg_unread_count", 0)
    _responding = st.session_state.get("_ai_bg_responding", False)
    if not st.session_state.get("_floating_ai_open", False):
        with st.container(key="floating_ai_launcher"):
            if _unread > 0:
                _badge_class = "ai-notif-badge pulse" if _responding else "ai-notif-badge"
                st.markdown(
                    f'<div class="{_badge_class}">{_unread}</div>',
                    unsafe_allow_html=True,
                )
            elif _responding:
                st.markdown(
                    '<div class="ai-notif-badge pulse">⋯</div>',
                    unsafe_allow_html=True,
                )
            if st.button(
                "AI",
                key="_floating_ai_open_btn",
                help="Open Research Copilot dock" if lang == "en" else "打开研究 Copilot dock",
            ):
                st.session_state["_floating_ai_open"] = True
                st.rerun()

    if st.session_state.get("_floating_ai_open", False):
        with st.container(key="floating_ai_panel"):
            dock_title = "Research Copilot" if lang == "en" else "研究 Copilot"
            dock_subtitle = (
                "Grounded in your current page and selections."
                if lang == "en" else
                "基于当前页面和已选配置给出建议。"
            )
            provider_key = coerce_public_provider(
                st.session_state.get("llm_provider", public_default_provider_key())
            )
            provider_label = public_provider_defaults(provider_key)[0] or provider_key
            llm_configured = _is_configured()
            if llm_configured:
                status_label = f"{provider_label} · online"
                status_color = "var(--ok)"
            else:
                status_label = "local Copilot" if lang == "en" else "本地 Copilot"
                status_color = "var(--accent)"
            status_pill = (
                f'<span class="eu-pill mono" '
                f'style="font-size:10px;padding:2px 7px;height:18px;'
                f'background:var(--surface);color:{status_color};'
                f'border:1px solid var(--hair-2)">'
                f'<span class="dot" style="background:{status_color}"></span>'
                f'{html.escape(status_label)}</span>'
            )
            header_cols = st.columns([4.8, 0.8, 0.8, 0.8, 1, 1])
            with header_cols[0]:
                st.markdown(
                    f'''
                    <div class="floating-ai-header">
                        <div class="floating-ai-title-row">
                            <span class="floating-ai-avatar" aria-label="AI assistant">✦</span>
                            <div>
                                <div class="floating-ai-title">{dock_title} {status_pill}</div>
                                <div class="floating-ai-subtitle">{dock_subtitle}</div>
                            </div>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            with header_cols[1]:
                if st.button("S", key="_floating_ai_size_s_btn", use_container_width=True, help="Compact size" if lang == "en" else "紧凑尺寸", disabled=export_locked):
                    st.session_state["_floating_ai_size"] = "s"
                    st.rerun()
            with header_cols[2]:
                if st.button("M", key="_floating_ai_size_m_btn", use_container_width=True, help="Medium size" if lang == "en" else "中等尺寸", disabled=export_locked):
                    st.session_state["_floating_ai_size"] = "m"
                    st.rerun()
            with header_cols[3]:
                if st.button("L", key="_floating_ai_size_l_btn", use_container_width=True, help="Large size" if lang == "en" else "大尺寸", disabled=export_locked):
                    st.session_state["_floating_ai_size"] = "l"
                    st.rerun()
            with header_cols[4]:
                if st.button("—", key="_floating_ai_minimize_btn", use_container_width=True, help="Minimize" if lang == "en" else "最小化", disabled=export_locked):
                    _close_floating_ai_panel(disable_assistant=False)
                    st.rerun()
            with header_cols[5]:
                if st.button("✕", key="_floating_ai_close_btn", use_container_width=True, help="Close" if lang == "en" else "关闭", disabled=export_locked):
                    _close_floating_ai_panel(disable_assistant=True)
                    st.rerun()

            _render_floating_copilot_context_actions(lang)
            if not _is_configured():
                st.caption(
                    "Local Copilot is available. Configure a provider only for open-ended external model calls."
                    if lang == "en" else
                    "本地 Copilot 可直接使用。只有开放式外部模型调用才需要配置服务商/API Key。"
                )
            _render_compact_chat_panel(
                lang=lang,
                panel_key="_llm_floating",
                history_height=size_cfg["history_height"],
                show_starters=False,
            )


# ---------------------------------------------------------------------------
# Intro / tips helpers
# ---------------------------------------------------------------------------

def _render_intro(lang: str):
    """Show a brief feature overview when the assistant is disabled."""
    if lang == "en":
        st.markdown("""\
#### What is Research Copilot?
A built-in conversational helper that knows EasyICU inside-out. It can:
- Start from your study goal, then map it to the right EasyICU workflow
- Explain which cohort filters, feature modules, and scores fit your task
- List supported databases, concepts, and scoring systems
- Help interpret extraction results (SOFA, Sepsis-3, missingness, etc.)

**Getting started:**
1. Toggle **Enable Research Copilot** in the sidebar
2. Choose a provider and enter your API key or token
3. Start by describing your task, e.g. "I want to build a sepsis early-warning cohort."
4. If you already know the disease cohort you want, ask directly for AKI / Sepsis / ventilation / ICD-based filtering.
""")
    else:
        st.markdown("""\
#### 研究 Copilot 是什么？
内置的对话助手，熟知 EasyICU 的所有功能。它可以：
- 从你的研究目标出发，反推最合适的 EasyICU 工作流
- 解释适合该任务的队列筛选、特征模块和临床评分
- 列出支持的数据库、概念和评分系统
- 帮助解读提取结果（SOFA、Sepsis-3、缺失率等）

**快速开始：**
1. 在侧边栏开启 **启用研究 Copilot**
2. 选择服务商并填写对应的 API Key / Token
3. 先描述你的任务，例如“我想做脓毒症实时预警队列”。
4. 如果你已经知道要筛选的疾病队列，也可以直接问 AKI / Sepsis / 机械通气 / ICD 队列怎么设置。
""")


def _render_tips(lang: str):
    if lang == "en":
        st.markdown("""\
- **Onboarding**: "I want to extract SOFA-2 from MIMIC-IV. What exact steps should I follow in the web UI?"
- **Task-first planning**: "I want to build a sepsis early-warning model. How can EasyICU support this, and which modules should I extract?"
- **Disease cohort setup**: "I want to build an AKI cohort. Which cohort filters should I enable, and which concepts should I export?"
- **Trajectory analysis**: "I want to cluster sepsis trajectories over time. Which time-series concepts and scores should I export?"
- **Feature planning**: "For septic shock research, which EasyICU concepts should I export besides SOFA and vasopressors?"
- **ICD filtering**: "I want an ICD-defined pneumonia or heart-failure cohort in MIMIC-IV. How should I use the ICD filter in Step 2?"
- **Cross-database mapping**: "Which respiratory concepts are available across miiv, mimic, eicu, aumc, hirid, and sic?"
- **Troubleshooting**: "My exported data shows high missingness for fio2. What are the most likely causes and checks?"
- **Interpretation**: "How should I interpret `sep3_sofa2`, `susp_inf`, and `sofa2` together?"
- **Definition help**: "Explain the Sepsis suspected-infection settings in the web app and when I should use `auto`, `and`, or `icd_abx`."
- **Code-aware help**: "Where is export implemented in app.py?" / "How does `load_concepts` work?"
- **Evidence-backed answers**: "Explain Sepsis-3 with PubMed sources and relate it to EasyICU outputs."
- **Python workflow**: "Show me a minimal Python example to load pafi, sofa2, and sep3_sofa2."
""")
    else:
        st.markdown("""\
- **新手引导**: "我想从 MIMIC-IV 提取 SOFA-2，网页端具体点哪里、按什么顺序做？"
- **任务规划**: "我想做脓毒症实时预警，EasyICU 能怎么支持，建议提取哪些模块？"
- **疾病队列设置**: "我想构建 AKI 队列，Step 2 应该怎么筛，Step 3 建议提取哪些特征？"
- **轨迹分析**: "我想做脓毒症患者轨迹聚类，应该优先导出哪些时间序列特征和评分？"
- **选特征建议**: "如果我要做脓毒症休克研究，除了 SOFA 和升压药，还建议导出哪些概念？"
- **ICD 队列**: "我想在 MIMIC-IV 里按 ICD 筛肺炎或心衰患者，Step 2 应该怎么设置？"
- **跨库对照**: "miiv、mimic、eicu、aumc、hirid、sic 里哪些呼吸相关概念都能取到？"
- **排错诊断**: "我导出的 fio2 缺失率很高，最可能是什么原因，应该检查哪几步？"
- **结果解读**: "`sep3_sofa2`、`susp_inf` 和 `sofa2` 应该怎么一起解释？"
- **定义说明**: "请解释网页端的 Sepsis 疑似感染设置，什么时候该用 `auto`、`and` 或 `icd_abx`？"
- **代码问题**: "app.py 里 export 在哪实现？" / "`load_concepts` 是怎么工作的？"
- **带证据医学回答**: "结合 PubMed 解释 Sepsis-3，并对应到 EasyICU 的输出概念。"
- **Python 工作流**: "给我一个最小 Python 例子，同时加载 pafi、sofa2 和 sep3_sofa2。"
""")


# ---------------------------------------------------------------------------
# Fast local replies for trivial prompts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Background LLM response (when panel is minimized)
# ---------------------------------------------------------------------------

_bg_lock = threading.Lock()


def _bg_llm_call(messages: list, lang: str, provider: str, model: str,
                 base_url: str, api_key: str, session_id: str):
    """Run an LLM call in a background thread and store the result."""
    try:
        import openai
        client_kwargs = {"base_url": base_url}
        default_headers = _provider_default_headers(coerce_public_provider(provider))
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            client_kwargs["api_key"] = "unused"
        # Strip proxy env vars
        env_backup = {}
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                    "ALL_PROXY", "all_proxy"):
            if var in os.environ:
                env_backup[var] = os.environ.pop(var)
        try:
            client = openai.OpenAI(**client_kwargs)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            draft = _strip_llm_reasoning(resp.choices[0].message.content or "")
            if not draft:
                raise RuntimeError("Model returned an empty response.")
        finally:
            os.environ.update(env_backup)

        # Store result in a module-level dict (accessible cross-rerun)
        with _bg_lock:
            _bg_results[session_id] = {
                "status": "done",
                "answer": draft,
                "lang": lang,
            }
    except Exception as exc:
        with _bg_lock:
            _bg_results[session_id] = {
                "status": "error",
                "answer": str(exc),
                "lang": lang,
            }


# Module-level dict to store background results (survives Streamlit reruns)
_bg_results: dict[str, dict] = {}


def _start_bg_response(prompt: str, lang: str) -> str | None:
    """Prepare and start a background LLM call. Returns a session_id or None."""
    try:
        enforce_external_llm_opt_in(_current_provider_choice(), language=lang)
    except AIOptInError as exc:
        st.error(str(exc))
        return None
    if not _is_configured():
        return None
    try:
        messages, tool_events = _compose_agent_messages(prompt)
    except Exception:
        return None
    st.session_state.llm_last_tool_events = tool_events
    provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
    p_info = public_provider_defaults(provider)
    model = (st.session_state.get("llm_model", "").strip() or p_info[2])
    base_url = st.session_state.get("llm_base_url", "").strip() or p_info[1]
    api_key = st.session_state.get("llm_api_key", "")
    session_id = f"bg_{id(st.session_state)}_{len(st.session_state.llm_messages)}"
    t = threading.Thread(
        target=_bg_llm_call,
        args=(messages, lang, provider, model, base_url, api_key, session_id),
        daemon=True,
    )
    t.start()
    return session_id


def _check_bg_response() -> dict | None:
    """Check if any background response is ready. Returns result dict or None."""
    session_id = st.session_state.get("_ai_bg_session_id")
    if not session_id:
        return None
    with _bg_lock:
        result = _bg_results.pop(session_id, None)
    return result


# ---------------------------------------------------------------------------
# Streaming & error handling
# ---------------------------------------------------------------------------

def _stream_response(messages: list, lang: str):
    """Call the LLM API with streaming and render tokens incrementally."""
    try:
        enforce_external_llm_opt_in(_current_provider_choice(), language=lang)
    except AIOptInError as exc:
        st.error(str(exc))
        return
    client = _get_client()
    if client is None:
        err = ("Failed to create API client. Check your configuration."
               if lang == "en" else "无法创建 API 客户端，请检查配置。")
        st.error(err)
        return

    provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
    model = (st.session_state.get("llm_model", "").strip()
             or public_provider_defaults(provider)[2])
    if not model:
        st.error("No model specified." if lang == "en" else "未指定模型名称。")
        return

    status_placeholder = st.empty()
    answer_placeholder = st.empty()

    try:
        status_placeholder.info(
            "Thinking..." if lang == "en" else "正在思考..."
        )
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        status_placeholder.info(
            "Generating response..." if lang == "en" else "正在生成回答..."
        )
        final_response = _stream_text(stream, answer_placeholder)
        if not final_response:
            raise RuntimeError("Model returned an empty response.")
        final_response = _append_quick_links(
            prompt=st.session_state.llm_messages[-1]["content"],
            answer=final_response,
            lang=lang,
        )
        answer_placeholder.markdown(final_response)
        st.session_state.llm_last_verification = None
        response_actions = _suggest_ui_actions(
            st.session_state.llm_messages[-1]["content"],
            final_response,
            lang,
        )
        _render_nav_actions(response_actions, key_prefix="_llm_action_live")
        status_placeholder.empty()
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": final_response,
                "actions": response_actions,
            }
        )
    except Exception as exc:
        status_placeholder.empty()
        error_message = _handle_api_error(exc, lang, render=False)
        answer_placeholder.markdown(error_message)
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "actions": [],
            }
        )


def _token_generator(stream):
    """Yield content tokens from an OpenAI-style streaming response."""
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                continue
            token = getattr(delta, "content", None)
            if token:
                yield token


def _handle_api_error(exc: Exception, lang: str, render: bool = True) -> str:
    """Build and optionally display a user-friendly error for common API failures."""
    err_str = str(exc)
    err_lower = err_str.lower()
    if "authentication" in err_str.lower() or "401" in err_str:
        provider = st.session_state.get("llm_provider", "custom")
        if provider == "huggingface_free":
            msg = (
                "Hugging Face requires your own token. "
                "Please create an HF token and paste it here."
                if lang == "en" else
                "Hugging Face 需要你自己提供 token。"
                "请创建 HF token 后再填写。"
            )
        else:
            msg = ("Authentication failed — please check your API Key."
                   if lang == "en" else "认证失败 — 请检查 API Key 是否正确。")
    elif (
        "429" in err_str
        or "rate" in err_lower
        or "temporarily rate-limited" in err_lower
        or "provider returned error" in err_lower
        or "retry shortly" in err_lower
    ):
        msg = (
            "The current hosted model is being rate-limited upstream. Please retry shortly, or switch the hosted default model to a more stable free model."
            if lang == "en" else
            "当前托管模型被上游限流了。请稍后重试，或把 hosted 默认模型切换到更稳定的免费模型。"
        )
    elif "socksio" in err_lower or "using socks proxy" in err_lower:
        msg = (
            "Proxy configuration error — the app detected a SOCKS proxy from the environment. "
            "EasyICU now ignores system proxy variables by default. Please retry. "
            "If you explicitly need a SOCKS proxy, install `httpx[socks]`."
            if lang == "en" else
            "代理配置异常 — 应用检测到了环境变量中的 SOCKS 代理。"
            "EasyICU 现在默认忽略系统代理变量，请重试。"
            "如果你确实需要 SOCKS 代理，请安装 `httpx[socks]`。"
        )
    elif "empty response" in err_lower or "no content" in err_lower:
        msg = (
            "The model returned an empty response. Retry once, or switch to another OpenRouter free model."
            if lang == "en" else
            "模型返回了空回复。请重试一次，或切换到另一个 OpenRouter 免费模型。"
        )
    elif "model" in err_lower or "404" in err_str:
        msg = ("Model not found — please verify the model name."
               if lang == "en" else "模型未找到 — 请确认模型名称是否正确。")
    elif "connect" in err_lower or "timeout" in err_lower:
        msg = ("Connection error — check the API Base URL and your network."
               if lang == "en" else "连接失败 — 请检查 API Base URL 和网络连接。")
    else:
        msg = ("API error: " if lang == "en" else "API 调用出错: ") + err_str
    if render:
        st.error(msg)
    return msg
