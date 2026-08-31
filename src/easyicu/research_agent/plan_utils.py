"""Deprecated compatibility exports for historical plan helper imports.

Production modules import the responsibility owner directly. This module remains
temporarily for external callers and older focused tests; deleting it does not
break the production dependency graph.
"""

# ruff: noqa: F401 - compatibility surface intentionally re-exports owners

from __future__ import annotations

from .contracts.model_covariates import (
    _covariate_names_from_code,
    _name_intends_covariates,
    _primary_exposure_overadjustment_findings,
    _primary_model_leakage_findings,
    read_adjustment_covariates,
    read_model_covariate_names,
)
from .contracts.product_identity import (
    normalised_expected_output_names as _normalised_expected_output_names,
    normalised_structured_output_names as _normalised_structured_output_names,
)
from .contracts.step_families import (
    _EFFECT_CONTRACT_METHODS,
    _ROBUSTNESS_EFFECT_CONTRACT_METHODS,
    _article_display_roles,
    _clustering_contract_applies,
    _cohort_change_contract_applies,
    _effect_contract_applies,
    _prediction_contract_applies,
    _step_expects_figure,
    _cohort_definition_contract_findings,
    clustering_contract_applies,
    cohort_change_contract_applies,
    effect_output_authorized,
    prediction_contract_applies,
)
from .gates.step_contract import _step_contract_findings
from .gates.step_repair import (
    _cohort_predicate_partition_safety_rules,
    _primary_analysis_cohort_canonical_schema_rules,
    _step_contract_repair_guidance,
)
from .gates.step_result_evidence import (
    _exposure_names_match,
    _finite_float,
    _primary_effect_from_summary,
    _primary_exposure_contract_findings,
    _primary_exposure_measurement_filter_findings,
)
from .planning.advanced_plan_contract import _enforce_advanced_plan_contract
from .planning.cohort_contract import (
    cohort_definition_is_empty as _cohort_definition_is_empty,
    cohort_definition_prose as _cohort_definition_prose,
    plan_expects_analysis_cohort as _plan_expects_analysis_cohort,
)
from .planning.endpoint_contract import endpoint_contract_findings
from .planning.figure_plan_mutation import (
    _effect_figure_semantics_supported_by_inputs,
    _effect_figure_semantics_supported_by_model_roster,
    _effect_figure_source_authorized,
    _ensure_publication_figure_step_in_plan,
    _render_only_figure_step_intent,
    _research_question_implies_figure,
    _split_table_and_figure_outputs_in_plan,
)
from .planning.figure_plan_shaping import (
    augment_report_typed_product_inputs as _augment_report_typed_product_inputs,
)
from .planning.figure_step_contract import (
    _output_declares_figure,
    _parent_step_id_for_figure_step,
    _preserve_figure_steps_after_replan,
    _step_produces_figure,
)
from .planning.plan_graph import (
    _cap_plan_preserving_figure_steps,
    _step_is_primary_estimand_model,
    _typed_plan_dag_findings,
)
