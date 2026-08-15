"""Compatibility facade for the split agent implementations.

Historical import surface of the former ``agents/core.py`` monolith.
Implementations now live in sibling owner modules (``_support``,
``planner``, ``replanner``, ``roles``, ``coder``, ``reporting``);
this module re-exports every previous top-level name so
``from ..agents.core import X`` and the ``agents.X is core.X``
identity contract keep holding. New code should import from the
owner modules, not from here.
"""


from __future__ import annotations

from ..planning.analysis_types import infer_analysis_type  # noqa: F401
from ..planning.primary_result_contract import (  # noqa: F401
    validate_required_primary_result as _validate_required_primary_result,
)
from ..providers.prompts import PROMPT_PACK_VERSION  # noqa: F401 — __all__ export
from ..research_context.prompt_variables import format_observed_domain

from . import plan_payload as _payload  # noqa: F401 — historical alias
from .coder_generation import generate_initial_coder_candidate  # noqa: F401
from .plan_payload import (  # noqa: F401
    _canonicalise_figure_output_alias,
    _canonicalise_planned_analysis_role,
    _declared_field_names,
    _is_untyped_figure_alias_output,
    _normalise_plan_payload,
)


from ._support import (  # noqa: F401 — re-export facade
    LLM_PARSE_DEBUG_CHARS,
    PLANNER_MAX_RETRIES,
    _CODER_AUTHORITY_PRECEDENCE,
    _CODER_GUIDE,
    _NATURE_WRITING_GUIDE,
    _PROMPT_PACK,
    _REPLANNER_GUIDE,
    _SYSTEM_GUIDE,
    _WRITER_GUIDE,
    _bounded_utf8_excerpt,
    _closed_cohort_product_sentence,
    _coder_relevant_notes,
    _coder_system_messages,
    _coerce_primary_estimate,
    _dump_raw,
    _empty_df_placeholder,
    _first_json_block,
    _format_context,
    _host_executed_cohort_step_sentence,
    _initial_reflection_memory,
    _looks_like_python_script,
    _outbound_repair_diagnosis,
    _repair_diagnosis_excerpt,
    _robustness_axis_vocabulary,
    _sentences_missing_evidence_tokens,
    _strip_code_fence,
    _suggest_repairs_for,
)

from .planner import (  # noqa: F401 — re-export facade
    PlannerAgent,
    PlannerArticleContractError,
    PlannerPromptBudgetError,
    _COHORT_PREDICATE_AGGREGATIONS,
    _PLANNER_MAX_TOKENS,
    _PLANNER_PROMPT_BYTE_LIMIT,
    _PLANNER_RETRY_PROJECTION_BYTE_LIMIT,
    _PRINCIPLES_GUIDE,
    _bindable_concept_ids,
    _build_planner_user_prompt,
    _format_concept_id_allowlist,
    _format_ctas_schema_constraints,
    _planner_prompt_within_budget,
    _planner_retry_response_projection,
    _validate_table_one_observed_levels,
    describe_article_contract_family_switch,
)

from .replanner import (  # noqa: F401 — re-export facade
    ReplannerAgent,
    _REPLANNER_FINDING_KEYS,
    _REPLANNER_FINDING_MESSAGE_CHARS,
    _REPLANNER_MAX_FINDINGS_PER_LIST,
    _REPLANNER_PROBE_CHAR_BUDGET,
    _REPLANNER_RECORD_FINDING_KEYS,
    _REPLANNER_RECORD_KEEP_KEYS,
    _REPLANNER_STEP_SUMMARY_CHAR_BUDGET,
    _REPLANNER_TOTAL_RECORDS_CHAR_BUDGET,
    _clip_json,
    _compact_findings,
    _slim_completed_records_for_prompt,
    _slim_record_for_replanner,
)

from .roles import (  # noqa: F401 — re-export facade
    ClinicalSemanticsAgent,
    CriticAgent,
    DataExtractionAgent,
    ManuscriptAgent,
    RuntimeSupervisor,
    StatisticalAnalysisAgent,
    VisualizationAgent,
)

from .coder import (  # noqa: F401 — re-export facade
    CoderAgent,
    CoderPromptBudgetError,
    _CODER_INITIAL_PROMPT_TARGET_BYTES,
    _CODER_MAX_TOKENS,
    _CODER_PATCH_MAX_EXCERPT_CHARS,
    _CODER_PATCH_PROMPT_BYTE_LIMIT,
    _CODER_TYPED_PATCH_DIAGNOSTIC_BYTE_LIMIT,
    _MAX_PRE_EXEC_COMPATIBILITY_REPAIRS,
    _NO_PROVENANCE_PAIR_RULE,
    _coder_prompt_payload_bytes,
    _cohort_predicate_partition_safety_contract,
    _compact_repair_scope_contract,
    _declared_output_scope_contract,
    _declared_statistic_products,
    _declares_no_measurement_provenance_pair,
    _enforce_coder_prompt_budget,
    _primary_analysis_cohort_output_contract,
    _repair_specialization,
    _statistic_payload_shape_directive,
    _typed_input_scope_contract,
)

from .reporting import (  # noqa: F401 — re-export facade
    AnalyzerAgent,
    ReportingPromptBudgetError,
    WriterAgent,
    _ANALYZER_PROMPT_BYTE_LIMIT,
    _WRITER_PROMPT_BYTE_LIMIT,
    _enforce_reporting_prompt_budget,
    _writer_language_instruction,
)

# Compatibility alias for callers/tests that imported the former local helper.
_format_observed_domain = format_observed_domain

__all__ = [
    "PlannerAgent",
    "ReplannerAgent",
    "ClinicalSemanticsAgent",
    "DataExtractionAgent",
    "StatisticalAnalysisAgent",
    "VisualizationAgent",
    "ManuscriptAgent",
    "CriticAgent",
    "RuntimeSupervisor",
    "CoderAgent",
    "AnalyzerAgent",
    "WriterAgent",
    "PROMPT_PACK_VERSION",
]
