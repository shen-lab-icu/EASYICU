"""Select trusted executors through one typed, duplicate-refusing registry."""

from __future__ import annotations

from typing import Any, Mapping

from ...authority.current_case_scientific_runtime import (
    AssociationModelGridRuntimeAuthority,
    LandmarkSplineRuntimeAuthority,
    LandmarkSurvivalRuntimeAuthority,
    SourceFeasibilityRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...authority.time_varying_runtime import TimeVaryingRuntimeAuthority
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.time_varying_exposure import TIME_VARYING_ANALYSIS_KIND
from ...schema import AnalysisPlan, AnalysisStep
from ..step_executor_registry import (
    AmbiguousExecutorOwnership,
    StandardExecutorCandidate,
    StandardExecutorSelection,
    StepExecutor,
    StepExecutorContext,
    StepExecutorDecision,
    StepExecutorRegistry,
)
from .adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
    adjusted_association_executor_code,
    adjusted_association_executor_verdict,
)
from .adjusted_association_figure_executor import (
    ADJUSTED_ASSOCIATION_FIGURE_INPUT,
    ASSOCIATION_OVERVIEW_FIGURE_INPUTS,
    association_figure_design_verdict,
    adjusted_association_figure_executor_code,
    adjusted_association_figure_executor_owns_step,
    association_overview_figure_executor_code,
    association_overview_figure_executor_owns_step,
)
from .association_model_grid_executor import (
    ASSOCIATION_MODEL_GRID_ANALYSIS_KIND,
    association_model_grid_executor_code,
    association_model_grid_executor_owns_step,
)
from .audit_panel_executor import (
    audit_panel_executor_code,
    audit_panel_executor_owns_step,
)
from .cohort_flow_figure_executor import (
    COHORT_FLOW_INPUT,
    cohort_flow_figure_executor_code,
    cohort_flow_figure_executor_owns_step,
)
from .cohort_summary_executor import (
    cohort_summary_executor_code,
    cohort_summary_executor_owns_step,
)
from .composite_descriptive_figure_executor import (
    composite_descriptive_figure_consumed_input_keys,
    composite_descriptive_figure_executor_code,
    composite_descriptive_figure_executor_owns_step,
)
from .cross_sectional_phenotyping_executor import (
    PHENOTYPING_ANALYSIS_KIND,
    cross_sectional_phenotyping_consumed_input_keys,
    cross_sectional_phenotyping_executor_code,
    cross_sectional_phenotyping_executor_owns_step,
)
from .cross_sectional_phenotyping_figure_executor import (
    PHENOTYPING_FIGURE_ANALYSIS_KIND,
    PHENOTYPING_FIGURE_INPUTS,
    cross_sectional_phenotyping_figure_executor_code,
    cross_sectional_phenotyping_figure_executor_owns_step,
)
from .descriptive_association_executor import (
    DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND,
    descriptive_association_executor_code,
    descriptive_association_executor_owns_step,
)
from .descriptive_distribution_executor import (
    DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND,
    descriptive_distribution_executor_code,
    descriptive_distribution_executor_owns_step,
)
from .descriptive_result_figure_executor import (
    descriptive_result_figure_executor_code,
    descriptive_result_figure_executor_owns_step,
)
from .deterministic_missingness import (
    is_compact_missingness_measurement_contract,
    is_measurement_bias_audit_contract,
    is_missingness_complete_case_contract,
    missingness_audit_cohort_input_key,
    missingness_audit_executor_owns_step,
    missingness_contract_details,
    missingness_measurement_audit_code,
    source_availability_audit_executor_owns_step,
)
from .deterministic_robustness import (
    ROBUSTNESS_REPLAY_ANALYSIS_KIND,
    robustness_replay_declaration_verdict,
)
from .exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_declaration_verdict,
    exposure_outcome_distribution_executor_code,
    exposure_outcome_distribution_executor_owns_step,
)
from .exposure_outcome_distribution_render import (
    EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
    exposure_outcome_distribution_figure_code,
    exposure_outcome_distribution_figure_declaration_verdict,
    exposure_outcome_distribution_figure_owns_step,
)
from .feasibility_protocol_executor import (
    FEASIBILITY_PROTOCOL_ANALYSIS_KIND,
    feasibility_protocol_consumed_input_keys,
    feasibility_protocol_executor_code,
    feasibility_protocol_executor_owns_step,
)
from .host_bound_cohort_executor import (
    HOST_BOUND_COHORT_ANALYSIS_KIND,
    host_bound_cohort_executor_code,
    host_bound_cohort_executor_owns_step,
)
from .landmark_association_figure_executor import (
    landmark_association_figure_executor_code,
    landmark_association_figure_executor_owns_step,
)
from .landmark_spline_executor import (
    LANDMARK_SPLINE_ANALYSIS_KIND,
    landmark_spline_executor_code,
    landmark_spline_executor_owns_step,
)
from .landmark_spline_functional_form_executor import (
    LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND,
    landmark_spline_functional_form_executor_code,
    landmark_spline_functional_form_executor_owns_step,
)
from .landmark_spline_robustness_executor import (
    LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND,
    landmark_spline_robustness_executor_code,
    landmark_spline_robustness_executor_owns_step,
)
from .landmark_survival_executor import (
    LANDMARK_SURVIVAL_ANALYSIS_KIND,
    LANDMARK_SURVIVAL_FIGURE_ANALYSIS_KIND,
    landmark_survival_executor_code,
    landmark_survival_executor_owns_step,
    landmark_survival_figure_executor_code,
    landmark_survival_figure_executor_owns_step,
)
from .missingness_measurement_figure_executor import (
    measurement_missingness_figure_executor_code,
    measurement_missingness_figure_executor_owns_step,
    missingness_measurement_figure_declaration_verdict,
    missingness_measurement_figure_executor_code,
    missingness_measurement_figure_executor_owns_step,
)
from .ordered_stratified_executor import (
    ORDERED_STRATIFIED_ANALYSIS_KIND,
    ordered_stratified_consumed_input_keys,
    ordered_stratified_executor_code,
    ordered_stratified_executor_owns_step,
)
from .prediction_figure_executor import (
    PREDICTION_COMPOSITE_FIGURE_INPUTS,
    PREDICTION_FIGURE_ANALYSIS_KIND,
    prediction_figure_executor_code,
    prediction_figure_executor_owns_step,
)
from .prediction_model_executor import (
    PREDICTION_MODEL_ANALYSIS_KIND,
    prediction_model_consumed_input_keys,
    prediction_model_executor_code,
    prediction_model_executor_owns_step,
)
from .prevalence_mortality_figure_executor import (
    PREVALENCE_MORTALITY_FIGURE_INPUTS,
    prevalence_mortality_figure_executor_code,
    prevalence_mortality_figure_executor_owns_step,
)
from .prevalence_outcome_figure_executor import (
    PREVALENCE_OUTCOME_FIGURE_INPUT,
    prevalence_outcome_figure_executor_code,
    prevalence_outcome_figure_executor_owns_step,
)
from .robustness_figure_executor import (
    robustness_figure_consumed_input_keys,
    robustness_figure_executor_code,
    robustness_figure_executor_owns_step,
)
from .scientific_reporting_executor import (
    SCIENTIFIC_REPORTING_ANALYSIS_KIND,
    scientific_reporting_consumed_input_keys,
    scientific_reporting_executor_code,
    scientific_reporting_executor_owns_step,
)
from .source_feasibility_executor import (
    SOURCE_FEASIBILITY_ANALYSIS_KIND,
    source_feasibility_executor_code,
    source_feasibility_executor_owns_step,
)
from .survival_primary_executor import (
    SURVIVAL_PRIMARY_ANALYSIS_KIND,
    survival_primary_executor_code,
    survival_primary_executor_verdict,
)
from .table_one_executor import table_one_executor_code, table_one_executor_owns_step
from .time_varying_executor import time_varying_executor_code
from .trajectory_scientific_candidate_executor import (
    SCIENTIFIC_CANDIDATE_INPUTS,
    trajectory_scientific_candidate_executor_code,
    trajectory_scientific_candidate_executor_owns_step,
)
from .trajectory_scientific_representation_executor import (
    trajectory_scientific_representation_executor_code,
    trajectory_scientific_representation_executor_owns_step,
)
from .trajectory_selection_figure_executor import (
    TRAJECTORY_SELECTION_FIGURE_INPUTS,
    trajectory_selection_figure_executor_code,
    trajectory_selection_figure_executor_owns_step,
)
from .trajectory_stability_executor import (
    STABILITY_EXECUTOR_INPUTS,
    trajectory_stability_executor_code,
    trajectory_stability_executor_owns_step,
)
from .typed_input_binding import sole_typed_cohort_input

__all__ = [
    "STANDARD_EXECUTORS",
    "StandardExecutorCandidate",
    "StandardExecutorSelection",
    "select_standard_executor",
    "resolve_standard_executor",
]


def _consumed_typed_cohort_inputs(step: AnalysisStep) -> tuple[str, ...]:
    value = sole_typed_cohort_input(step)
    return (value,) if value else ()


def _missingness_kind(context: StepExecutorContext) -> str:
    step = context.step
    if source_availability_audit_executor_owns_step(step):
        return "missingness_source_availability_audit"
    if is_measurement_bias_audit_contract(step.method, step.expected_outputs):
        return "measurement_bias_audit"
    if is_compact_missingness_measurement_contract(step.method, step.expected_outputs):
        return "missingness_measurement_audit"
    if is_missingness_complete_case_contract(step.method, step.expected_outputs):
        return "missingness_complete_case_audit"
    return "declared_missingness_audit_products"


def _missingness_reason(context: StepExecutorContext) -> str:
    return {
        "missingness_source_availability_audit": "missingness_source_availability_contract_preflight",
        "measurement_bias_audit": "measurement_bias_contract_preflight",
        "missingness_measurement_audit": "missingness_measurement_contract_preflight",
        "missingness_complete_case_audit": "missingness_complete_case_contract_preflight",
        "declared_missingness_audit_products": "missingness_audit_product_capability_preflight",
    }[_missingness_kind(context)]


def _build_registry() -> StepExecutorRegistry:
    registry = StepExecutorRegistry()
    routes = (
        StepExecutor(
            key=HOST_BOUND_COHORT_ANALYSIS_KIND,
            owns=lambda c: host_bound_cohort_executor_owns_step(c.step),
            render=lambda c: host_bound_cohort_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind=HOST_BOUND_COHORT_ANALYSIS_KIND,
            selection_reason="sealed_run_cohort_root_contract_preflight",
            progress_message="Publishing the sealed run cohort as a typed root",
            consumed_input_keys=lambda _c: (),
        ),
        StepExecutor(
            key=ASSOCIATION_MODEL_GRID_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                AssociationModelGridRuntimeAuthority,
            ),
            owns=lambda c: association_model_grid_executor_owns_step(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
            ),
            render=lambda c: association_model_grid_executor_code(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
                plausibility_scope=c.plausibility_scope,
            ),
            analysis_kind=ASSOCIATION_MODEL_GRID_ANALYSIS_KIND,
            selection_reason="signed_association_model_grid_contract_preflight",
            progress_message="Using verified adjusted-association model-grid adapter",
            consumed_input_keys=lambda c: (
                c.current_case_scientific_runtime_authority.cohort_product,
                c.current_case_scientific_runtime_authority.parent_product,
            ),
        ),
        StepExecutor(
            key=LANDMARK_SURVIVAL_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                LandmarkSurvivalRuntimeAuthority,
            ),
            owns=lambda c: landmark_survival_executor_owns_step(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
            ),
            render=lambda c: landmark_survival_executor_code(
                c.step,
                authority=c.current_case_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
                plausibility_scope=c.plausibility_scope,
            ),
            analysis_kind=LANDMARK_SURVIVAL_ANALYSIS_KIND,
            selection_reason="signed_landmark_survival_suite_contract_preflight",
            progress_message="Using signed deterministic landmark survival suite",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key=LANDMARK_SURVIVAL_FIGURE_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                LandmarkSurvivalRuntimeAuthority,
            ),
            owns=lambda c: landmark_survival_figure_executor_owns_step(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
            ),
            render=lambda c: landmark_survival_figure_executor_code(
                c.step, authority=c.current_case_scientific_runtime_authority
            ),
            analysis_kind=LANDMARK_SURVIVAL_FIGURE_ANALYSIS_KIND,
            selection_reason="signed_landmark_survival_figure_contract_preflight",
            progress_message="Using source-bound landmark survival renderer",
            consumed_input_keys=lambda c: (
                c.current_case_scientific_runtime_authority.figure_input_products
            ),
            host_sealed_renderer=True,
        ),
        StepExecutor(
            key=TIME_VARYING_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                TimeVaryingRuntimeAuthority,
            ),
            owns=lambda c: (
                c.current_case_scientific_runtime_authority.governed_step(c.plan)
                == c.step
            ),
            render=lambda c: time_varying_executor_code(
                c.step,
                authority=c.current_case_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
                plausibility_scope=c.plausibility_scope,
            ),
            analysis_kind=TIME_VARYING_ANALYSIS_KIND,
            selection_reason="signed_time_varying_contract_preflight",
            progress_message="Using source-bound time-varying Cox executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key=LANDMARK_SPLINE_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                LandmarkSplineRuntimeAuthority,
            ),
            owns=lambda c: landmark_spline_executor_owns_step(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
            ),
            render=lambda c: landmark_spline_executor_code(
                c.step,
                authority=c.current_case_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
                plausibility_scope=c.plausibility_scope,
            ),
            analysis_kind=LANDMARK_SPLINE_ANALYSIS_KIND,
            selection_reason="signed_landmark_spline_contract_preflight",
            progress_message="Using signed landmark spline executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key=LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                LandmarkSplineRuntimeAuthority,
            ),
            owns=lambda c: landmark_spline_robustness_executor_owns_step(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
            ),
            render=lambda c: landmark_spline_robustness_executor_code(
                c.step,
                authority=c.current_case_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
            ),
            analysis_kind=LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND,
            selection_reason="signed_landmark_spline_robustness_projection_preflight",
            progress_message="Using signed landmark spline robustness projection",
            consumed_input_keys=lambda c: tuple(
                value for value in c.step.inputs if ":" in value
            ),
        ),
        StepExecutor(
            key=LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                LandmarkSplineRuntimeAuthority,
            ),
            owns=lambda c: landmark_spline_functional_form_executor_owns_step(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
            ),
            render=lambda c: landmark_spline_functional_form_executor_code(
                c.step,
                authority=c.current_case_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
            ),
            analysis_kind=LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND,
            selection_reason="signed_landmark_spline_functional_form_preflight",
            progress_message="Using signed landmark spline functional-form projection",
            consumed_input_keys=lambda c: (
                c.current_case_scientific_runtime_authority.downstream_parent_product,
                c.current_case_scientific_runtime_authority.linear_sensitivity_product,
            ),
        ),
        StepExecutor(
            key=SOURCE_FEASIBILITY_ANALYSIS_KIND,
            applicable=lambda c: isinstance(
                c.current_case_scientific_runtime_authority,
                SourceFeasibilityRuntimeAuthority,
            ),
            owns=lambda c: source_feasibility_executor_owns_step(
                c.step,
                plan=c.plan,
                authority=c.current_case_scientific_runtime_authority,
            ),
            render=lambda c: source_feasibility_executor_code(
                authority=c.current_case_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
            ),
            analysis_kind=SOURCE_FEASIBILITY_ANALYSIS_KIND,
            selection_reason="signed_source_feasibility_contract_preflight",
            progress_message="Using signed source-feasibility executor",
            consumed_input_keys=lambda _c: (),
        ),
        StepExecutor(
            key=SCIENTIFIC_REPORTING_ANALYSIS_KIND,
            owns=lambda c: scientific_reporting_executor_owns_step(c.step),
            render=lambda c: scientific_reporting_executor_code(c.step),
            analysis_kind=SCIENTIFIC_REPORTING_ANALYSIS_KIND,
            selection_reason="typed_evidence_bound_scientific_report",
            progress_message="Indexing registered scientific results",
            consumed_input_keys=lambda c: scientific_reporting_consumed_input_keys(
                c.step
            ),
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key=FEASIBILITY_PROTOCOL_ANALYSIS_KIND,
            owns=lambda c: feasibility_protocol_executor_owns_step(c.step),
            render=lambda c: feasibility_protocol_executor_code(c.step),
            analysis_kind=FEASIBILITY_PROTOCOL_ANALYSIS_KIND,
            selection_reason="planner_declared_feasibility_protocol",
            progress_message="Recording the Planner-declared non-executable protocol",
            consumed_input_keys=lambda c: feasibility_protocol_consumed_input_keys(
                c.step
            ),
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key=PREDICTION_MODEL_ANALYSIS_KIND,
            owns=lambda c: prediction_model_executor_owns_step(c.step),
            render=lambda c: prediction_model_executor_code(c.step),
            analysis_kind=PREDICTION_MODEL_ANALYSIS_KIND,
            selection_reason="typed_static_prediction_contract_preflight",
            progress_message="Using deterministic static prediction adapter",
            consumed_input_keys=lambda c: prediction_model_consumed_input_keys(c.step),
        ),
        StepExecutor(
            key=PHENOTYPING_ANALYSIS_KIND,
            owns=lambda c: cross_sectional_phenotyping_executor_owns_step(c.step),
            render=lambda c: cross_sectional_phenotyping_executor_code(c.step),
            analysis_kind=PHENOTYPING_ANALYSIS_KIND,
            selection_reason="typed_cross_sectional_phenotyping_contract_preflight",
            progress_message="Using deterministic cross-sectional phenotyping adapter",
            consumed_input_keys=lambda c: (
                cross_sectional_phenotyping_consumed_input_keys(c.step)
            ),
        ),
        StepExecutor(
            key="descriptive_cohort_summary",
            owns=lambda c: cohort_summary_executor_owns_step(c.step),
            render=lambda c: cohort_summary_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind="descriptive_cohort_summary",
            selection_reason="cohort_summary_contract_preflight",
            progress_message="Using planner-scoped cohort summary executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key=ORDERED_STRATIFIED_ANALYSIS_KIND,
            owns=lambda c: ordered_stratified_executor_owns_step(c.step, plan=c.plan),
            render=lambda c: ordered_stratified_executor_code(c.step, plan=c.plan),
            analysis_kind=ORDERED_STRATIFIED_ANALYSIS_KIND,
            selection_reason="typed_ordered_stratified_contract_preflight",
            progress_message="Using deterministic ordered-trend adapter",
            consumed_input_keys=lambda c: ordered_stratified_consumed_input_keys(
                c.step
            ),
        ),
        StepExecutor(
            key="exposure_outcome_distribution",
            owns=lambda c: exposure_outcome_distribution_executor_owns_step(c.step),
            declaration_verdict=lambda c: (
                exposure_outcome_distribution_declaration_verdict(c.step)
            ),
            render=lambda c: exposure_outcome_distribution_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind="exposure_outcome_distribution",
            selection_reason="exposure_outcome_distribution_contract_preflight",
            progress_message="Using planner-declared exposure/outcome distribution executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key="exposure_outcome_distribution_figure",
            owns=lambda c: exposure_outcome_distribution_figure_owns_step(c.step),
            declaration_verdict=lambda c: (
                exposure_outcome_distribution_figure_declaration_verdict(c.step)
            ),
            render=lambda c: exposure_outcome_distribution_figure_code(
                c.step, display_labels=c.plan.display_labels
            ),
            analysis_kind="exposure_outcome_distribution_figure",
            selection_reason="exposure_outcome_distribution_figure_contract_preflight",
            progress_message="Using planner-scoped exposure/outcome distribution renderer",
            consumed_input_keys=lambda _c: (
                EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
            ),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="prevalence_outcome_figure",
            owns=lambda c: prevalence_outcome_figure_executor_owns_step(c.step),
            render=lambda c: prevalence_outcome_figure_executor_code(c.step),
            analysis_kind="prevalence_outcome_figure",
            selection_reason="prevalence_outcome_figure_contract_preflight",
            progress_message="Using planner-scoped prevalence/outcome figure executor",
            consumed_input_keys=lambda _c: (PREVALENCE_OUTCOME_FIGURE_INPUT,),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="robustness_figure",
            owns=lambda c: robustness_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: robustness_figure_executor_code(c.step),
            analysis_kind="robustness_figure",
            selection_reason="robustness_figure_contract_preflight",
            progress_message="Using planner-scoped robustness figure executor",
            consumed_input_keys=lambda c: robustness_figure_consumed_input_keys(c.step),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="association_overview_figure",
            accepts_figure_presentation=True,
            declaration_verdict=lambda c: association_figure_design_verdict(
                c.step, overview=True
            ),
            owns=lambda c: association_overview_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: association_overview_figure_executor_code(
                c.step, display_labels=c.plan.display_labels
            ),
            analysis_kind="association_overview_figure",
            selection_reason="association_overview_figure_contract_preflight",
            progress_message="Using source-bound association overview renderer",
            consumed_input_keys=lambda _c: ASSOCIATION_OVERVIEW_FIGURE_INPUTS,
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="adjusted_association_figure",
            accepts_figure_presentation=True,
            declaration_verdict=lambda c: association_figure_design_verdict(c.step),
            owns=lambda c: adjusted_association_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: adjusted_association_figure_executor_code(c.step),
            analysis_kind="adjusted_association_figure",
            selection_reason="adjusted_association_figure_contract_preflight",
            progress_message="Using planner-scoped adjusted association figure executor",
            consumed_input_keys=lambda _c: (ADJUSTED_ASSOCIATION_FIGURE_INPUT,),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="cohort_flow_figure",
            owns=lambda c: cohort_flow_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: cohort_flow_figure_executor_code(c.step),
            analysis_kind="cohort_flow_figure",
            selection_reason="cohort_flow_figure_contract_preflight",
            progress_message="Using digest-bound cohort-flow renderer",
            consumed_input_keys=lambda _c: (COHORT_FLOW_INPUT,),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="landmark_association_composite_figure",
            owns=lambda c: landmark_association_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: landmark_association_figure_executor_code(c.step),
            analysis_kind="landmark_association_composite_figure",
            selection_reason="landmark_association_composite_figure_contract_preflight",
            progress_message="Using digest-bound landmark association composite renderer",
            consumed_input_keys=lambda c: tuple(str(value) for value in c.step.inputs),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key=PREDICTION_FIGURE_ANALYSIS_KIND,
            owns=lambda c: prediction_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: prediction_figure_executor_code(c.step),
            analysis_kind=PREDICTION_FIGURE_ANALYSIS_KIND,
            selection_reason="static_prediction_figure_contract_preflight",
            progress_message="Using source-bound static prediction renderer",
            consumed_input_keys=lambda _c: PREDICTION_COMPOSITE_FIGURE_INPUTS,
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key=PHENOTYPING_FIGURE_ANALYSIS_KIND,
            owns=lambda c: cross_sectional_phenotyping_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: cross_sectional_phenotyping_figure_executor_code(c.step),
            analysis_kind=PHENOTYPING_FIGURE_ANALYSIS_KIND,
            selection_reason="cross_sectional_phenotyping_figure_contract_preflight",
            progress_message="Using source-bound cross-sectional phenotyping renderer",
            consumed_input_keys=lambda _c: PHENOTYPING_FIGURE_INPUTS,
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="composite_descriptive_figure",
            owns=lambda c: composite_descriptive_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: composite_descriptive_figure_executor_code(
                c.step, display_labels=c.plan.display_labels
            ),
            analysis_kind="composite_descriptive_figure",
            selection_reason="composite_descriptive_figure_contract_preflight",
            progress_message="Using digest-bound composite descriptive renderer",
            consumed_input_keys=lambda c: (
                composite_descriptive_figure_consumed_input_keys(c.step)
            ),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="descriptive_result_figure",
            owns=lambda c: descriptive_result_figure_executor_owns_step(
                c.step, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: descriptive_result_figure_executor_code(c.step),
            analysis_kind="descriptive_result_figure",
            selection_reason="descriptive_result_figure_contract_preflight",
            progress_message="Using digest-bound descriptive result renderer",
            consumed_input_keys=lambda c: (c.step.inputs[0],),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="prevalence_mortality_figure",
            owns=lambda c: prevalence_mortality_figure_executor_owns_step(c.step),
            render=lambda c: prevalence_mortality_figure_executor_code(
                c.step, display_labels=c.plan.display_labels
            ),
            analysis_kind="prevalence_mortality_figure",
            selection_reason="prevalence_mortality_figure_contract_preflight",
            progress_message="Using planner-scoped prevalence/mortality figure executor",
            consumed_input_keys=lambda _c: PREVALENCE_MORTALITY_FIGURE_INPUTS,
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="measurement_missingness_figure",
            owns=lambda c: measurement_missingness_figure_executor_owns_step(
                c.step, plan=c.plan, resolved_bindings=c.resolved_bindings
            ),
            render=lambda c: measurement_missingness_figure_executor_code(
                c.step, plan=c.plan
            ),
            analysis_kind="measurement_missingness_figure",
            selection_reason="measurement_missingness_figure_contract_preflight",
            progress_message="Using digest-bound measurement-missingness figure renderer",
            consumed_input_keys=lambda c: (str(c.step.inputs[0]),),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="missingness_measurement_figure",
            owns=lambda c: missingness_measurement_figure_executor_owns_step(
                c.step, plan=c.plan, resolved_bindings=c.resolved_bindings
            ),
            declaration_verdict=lambda c: (
                missingness_measurement_figure_declaration_verdict(c.step, plan=c.plan)
            ),
            render=lambda c: missingness_measurement_figure_executor_code(
                c.step, plan=c.plan
            ),
            analysis_kind="missingness_measurement_figure",
            selection_reason="missingness_measurement_figure_contract_preflight",
            progress_message="Using planner-scoped missingness/measurement figure executor",
            consumed_input_keys=lambda c: tuple(str(value) for value in c.step.inputs),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="audit_panel",
            owns=lambda c: audit_panel_executor_owns_step(c.step),
            render=lambda c: audit_panel_executor_code(c.step),
            analysis_kind="audit_panel",
            selection_reason="framework_audit_panel_contract_preflight",
            progress_message="Using deterministic audit-panel renderer",
            consumed_input_keys=lambda _c: (),
            host_sealed_renderer=True,
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key=DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND,
            owns=lambda c: descriptive_distribution_executor_owns_step(c.step),
            render=lambda c: descriptive_distribution_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind=DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND,
            selection_reason="grouped_descriptive_distribution_contract_preflight",
            progress_message="Using planner-scoped grouped descriptive distribution executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key=DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND,
            owns=lambda c: descriptive_association_executor_owns_step(c.step),
            render=lambda c: descriptive_association_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind=DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND,
            selection_reason="descriptive_association_contract_preflight",
            progress_message="Using planner-scoped descriptive association executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key="grouped_table_one",
            owns=lambda c: table_one_executor_owns_step(c.step),
            render=lambda c: table_one_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind="grouped_table_one",
            selection_reason="table_one_spec_preflight",
            progress_message="Using planner-specified grouped Table 1 executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key="missingness_audit",
            contract_details=lambda c: missingness_contract_details(c.step),
            owns=lambda c: missingness_audit_executor_owns_step(c.step),
            render=lambda c: missingness_measurement_audit_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind=_missingness_kind,
            selection_reason=_missingness_reason,
            progress_message="Using planner-specified missingness audit executor",
            consumed_input_keys=lambda c: (
                (missingness_audit_cohort_input_key(c.step),)
                if missingness_audit_cohort_input_key(c.step) is not None
                else ()
            ),
        ),
        StepExecutor(
            key="trajectory_signed_representation",
            owns=lambda c: trajectory_scientific_representation_executor_owns_step(
                c.step, plan=c.plan, authority=c.trajectory_scientific_runtime_authority
            ),
            render=lambda c: trajectory_scientific_representation_executor_code(
                authority=c.trajectory_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
            ),
            analysis_kind="trajectory_signed_representation",
            selection_reason="signed_trajectory_representation_authority",
            progress_message="Using signed trajectory representation executor",
            consumed_input_keys=lambda _c: (),
        ),
        StepExecutor(
            key="trajectory_signed_candidate_selection",
            owns=lambda c: trajectory_scientific_candidate_executor_owns_step(
                c.step, plan=c.plan, authority=c.trajectory_scientific_runtime_authority
            ),
            render=lambda c: trajectory_scientific_candidate_executor_code(
                authority=c.trajectory_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
            ),
            analysis_kind="trajectory_signed_candidate_selection",
            selection_reason="signed_trajectory_candidate_authority",
            progress_message="Using signed trajectory candidate selector",
            consumed_input_keys=lambda _c: tuple(sorted(SCIENTIFIC_CANDIDATE_INPUTS)),
        ),
        StepExecutor(
            key="trajectory_cluster_stability",
            owns=lambda c: trajectory_stability_executor_owns_step(c.step, plan=c.plan),
            render=lambda c: trajectory_stability_executor_code(
                c.step,
                plan=c.plan,
                scientific_runtime_authority=c.trajectory_scientific_runtime_authority,
                runtime_projection_sha256=c.scientific_runtime_projection_sha256,
            ),
            analysis_kind="trajectory_cluster_stability",
            selection_reason="trajectory_stability_spec_preflight",
            progress_message="Using planner-specified trajectory stability executor",
            consumed_input_keys=lambda _c: tuple(sorted(STABILITY_EXECUTOR_INPUTS)),
            blocks_on_plausibility_receipt=True,
        ),
        StepExecutor(
            key="trajectory_selection_diagnostic_figure",
            owns=lambda c: trajectory_selection_figure_executor_owns_step(c.step),
            render=lambda c: trajectory_selection_figure_executor_code(c.step),
            analysis_kind="trajectory_selection_diagnostic_figure",
            selection_reason="signed_trajectory_selection_figure_contract",
            progress_message="Rendering signed trajectory selection diagnostics",
            consumed_input_keys=lambda _c: TRAJECTORY_SELECTION_FIGURE_INPUTS,
        ),
        StepExecutor(
            key=SURVIVAL_PRIMARY_ANALYSIS_KIND,
            owns=lambda c: survival_primary_executor_verdict(c.step),
            render=lambda c: survival_primary_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind=SURVIVAL_PRIMARY_ANALYSIS_KIND,
            selection_reason="survival_primary_contract_preflight",
            progress_message="Using planner-declared primary Cox executor",
            consumed_input_keys=lambda c: (
                str(c.step.family_primary_result_requirement.input_product),
            ),
        ),
        StepExecutor(
            key=ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            owns=lambda c: adjusted_association_executor_verdict(c.step),
            render=lambda c: adjusted_association_executor_code(
                c.step, plausibility_scope=c.plausibility_scope
            ),
            analysis_kind=ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            selection_reason="adjusted_association_model_contract_preflight",
            progress_message="Using planner-declared adjusted-association executor",
            consumed_input_keys=lambda c: c.typed_cohort_inputs(),
        ),
        StepExecutor(
            key=ROBUSTNESS_REPLAY_ANALYSIS_KIND,
            owns=lambda _c: False,
            declaration_verdict=lambda c: robustness_replay_declaration_verdict(c.step),
            render=lambda _c: "",
            analysis_kind=ROBUSTNESS_REPLAY_ANALYSIS_KIND,
            selection_reason="unreachable_declaration_only_route",
            progress_message="",
            consumed_input_keys=lambda _c: (),
        ),
    )
    for route in routes:
        registry.declare(route)
    return registry


STANDARD_EXECUTORS = _build_registry()


def resolve_standard_executor(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
    resolved_bindings: Mapping[str, Any] | None = None,
    trajectory_scientific_runtime_authority: Mapping[str, Any] | None = None,
    current_case_scientific_runtime_authority: Mapping[str, Any] | None = None,
    scientific_runtime_projection_sha256: str | None = None,
) -> StepExecutorDecision:
    """Resolve ownership by exact typed contract without generating code."""

    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    sealed_current = (
        load_current_case_scientific_runtime_authority(
            current_case_scientific_runtime_authority
        )
        if current_case_scientific_runtime_authority is not None
        else None
    )
    context = StepExecutorContext(
        step=step,
        plan=plan,
        plausibility_scope=plausibility_scope,
        resolved_bindings=resolved_bindings,
        trajectory_scientific_runtime_authority=trajectory_scientific_runtime_authority,
        current_case_scientific_runtime_authority=sealed_current,
        scientific_runtime_projection_sha256=str(
            scientific_runtime_projection_sha256 or ""
        ),
    )
    return STANDARD_EXECUTORS.resolve(context)


def select_standard_executor(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
    resolved_bindings: Mapping[str, Any] | None = None,
    trajectory_scientific_runtime_authority: Mapping[str, Any] | None = None,
    current_case_scientific_runtime_authority: Mapping[str, Any] | None = None,
    scientific_runtime_projection_sha256: str | None = None,
    trace: list[StandardExecutorCandidate] | None = None,
) -> StandardExecutorSelection | None:
    """Select by exact typed contract, never prose or benchmark identity."""

    try:
        decision = resolve_standard_executor(
            step, plan=plan, plausibility_scope=plausibility_scope,
            resolved_bindings=resolved_bindings,
            trajectory_scientific_runtime_authority=trajectory_scientific_runtime_authority,
            current_case_scientific_runtime_authority=current_case_scientific_runtime_authority,
            scientific_runtime_projection_sha256=scientific_runtime_projection_sha256,
        )
    except AmbiguousExecutorOwnership as exc:
        if trace is not None:
            trace.extend(exc.candidates)
        raise
    if trace is not None:
        trace.extend(decision.candidates)
    return decision.render_selection()
