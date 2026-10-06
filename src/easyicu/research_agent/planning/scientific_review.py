"""Deterministic pre-execution scientific review for a proposed plan.

This module owns one boundary: deciding whether a plan may be offered for
human approval.  It does not choose a better estimand, add covariates, or
rewrite a user's question.  Instead it turns the sealed ResearchContext,
Planner plan, pre-plan literature authority, and article figure strategy into
a digest-bound review packet with stable findings and dimension scores.

The distinction from :mod:`easyicu.research_agent.reporting.scientific_maturity`
is intentional.  Scientific maturity audits an *executed* run and manuscript;
this owner runs before execution so an attractive but scientifically incomplete
plan cannot be approved first and downgraded only after provider work has run.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..canonical_json import canonical_sha256
from ..authority.declared_levels import closed_planning_levels_for
from ..authority.current_case_scientific_runtime import (
    CurrentCaseScientificRuntimeAuthority,
    LandmarkCategoricalAssociationRuntimeAuthority,
    LandmarkSplineRuntimeAuthority,
)
from ..concept_availability import normalize_database_name
from ..gates.plan_declared_inputs import declared_raw_input_plan_findings
from ..contracts.host_action_robustness import host_action_prespecified_axes
from ..contracts.sealed_suite_robustness import sealed_suite_prespecified_axes
from ..contracts.cohort_product_keys import (
    is_closed_cohort_product_key,
    sole_typed_cohort_input,
)
from ..contracts.association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
)
from ..contracts.descriptive_execution import (
    DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID,
    exposure_outcome_distribution_execution_verdict,
)
from ..contracts.ordered_stratified import is_ordered_stratified_analysis_step
from ..contracts.primary_cohort import step_cohort_population
from ..contracts.functional_form import functional_form_products
from ..contracts.phenotyping_features import PHENOTYPING_PRIMARY_ACTION, require_phenotyping_features
from ..contracts.prediction_execution import (
    PREDICTION_PRIMARY_ACTION,
    static_prediction_execution_verdict,
    static_prediction_executes_robustness_spec,
    static_prediction_features,
    static_prediction_model_columns,
    static_prediction_owns_step,
)
from ..contracts.phenotype_comparison import (
    ASSIGNMENTS_PRODUCT, COMPARISON_ACTION, comparison_cohort_input, comparison_label_source,
    validate_comparison_step,
)
from ..contracts.scientific_runtime_ownership import declared_runtime_outcomes
from ..contracts.trajectory_design import (
    TRAJECTORY_PRIMARY_ACTION,
    trajectory_coordinate_proposal,
    trajectory_population_design,
    trajectory_window_design,
)
from ..contracts.source_feasibility_validation import (
    context_declares_source_feasibility_scope,
)
from ..literature import LiteratureBundle, manuscript_citable_records
from ..research_context.concept_population import (
    ConceptCohortWindowError,
    concept_cohort_window,
)
from ..research_context.minimum_stay import minimum_icu_stay_hours
from ..research_context.temporal_semantics import (
    normalise_time_anchor,
    primary_exposure_time_anchor_alignment,
    study_time_origin_alignment,
    trajectory_window_statements,
    window_extends_after_anchor,
)
from ..research_context.typed import declared_domain_for_variable
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext
from ..trajectory.contract import trajectory_phenotyping_contract_applies
from ..trajectory.plan_contract import trajectory_context_is_bound
from ..trajectory.runtime_validation import (
    signed_trajectory_plan_claimed,
    signed_trajectory_plan_contract_errors,
)
from .cohort_eligibility import (
    PredicateAfterTimeZero,
    cohort_predicates_after_time_zero,
    eligibility_after_time_zero,
)
from .figure_strategy import ArticleFigureStrategy
from easyicu.outcome_availability import fixed_horizon_mortality_endpoint

from .adjustment_authority import (
    AdjustmentSetAuthority,
    host_outer_feature_window_end_hours,
    owner_declared_baseline_static,
    primary_landmark_hours,
)
from .analysis_types import (
    canonical_analysis_family,
    longitudinal_trajectory_requested,
    requested_dose_response_cues,
    requested_exposure_occurrence_cues,
)
from .population_requirements import context_population_requirements
from .baseline_requirements import (
    baseline_requirement_coverage,
    baseline_requirement_projection,
    context_baseline_requirements,
)
from .dependence_authority import (
    context_patient_group_authority,
    descriptive_counts_only_required,
    dependence_matches_context,
    repeat_units_possible,
)
from .distribution_authority import (
    DISTRIBUTION_MISSINGNESS_GUIDANCE,
    distribution_policy_issues,
)
from .method_literature import method_binding_support
from .novelty_contract import NOVELTY_REVIEW_DIMENSIONS
from .publication_readiness import build_publication_readiness_facts
from .sensitivity_authority import (
    EXECUTABLE_METHODS_BY_STRATEGY,
    FUNCTIONAL_FORM_EXECUTABLE_METHODS,
)
from .capability_registry import assess_scientific_capability
from .robustness_contract import complete_case_variables
from ..contracts.model_retention import (
    RETENTION_BLOCKER_BELOW,
    RETENTION_MAJOR_BELOW,
    PrimaryModelRetentionEvidence,
)


ScientificReviewSeverity = Literal["blocker", "major", "minor"]
ScientificRemediationRoute = Literal[
    "agent_plan_revision",
    "runtime_capability",
    "study_authority_change",
    "external_evidence",
    "independent_review",
    "unclassified",
]

class PlanScientificFinding(BaseModel):
    """One stable, reviewable defect in the proposed scientific plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: ScientificReviewSeverity
    dimension: str
    message: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    remediation: str
    remediation_route: ScientificRemediationRoute = "unclassified"
    requires_user_authorization: bool = False
    authorization_question: Optional[str] = None


CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION = "easyicu.plan_scientific_review/14"


class PlanScientificReview(BaseModel):
    """Digest-bound pre-approval review of an exact context/plan/literature set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Archived reviews remain readable, but cannot substitute for a /14
    # execution review (the resume gate also binds the review version).  /14
    # measures the primary model's retention on the cohort execution reads.
    schema_version: Literal[
        "easyicu.plan_scientific_review/10", "easyicu.plan_scientific_review/11",
        "easyicu.plan_scientific_review/12",
        "easyicu.plan_scientific_review/13",
        "easyicu.plan_scientific_review/14",
    ] = (
        CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION
    )
    status: Literal["changes_required", "analysis_only", "ready_for_approval"]
    review_scope: Literal["pre_execution_plan"] = "pre_execution_plan"
    rendered_outputs_assessed: Literal[False] = False
    approval_allowed: bool
    top_journal_candidate: bool
    score: int = Field(ge=0, le=100)
    dimension_scores: dict[str, int]
    findings: list[PlanScientificFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    literature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    figure_strategy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: str


def scientific_steps(plan: Optional[AnalysisPlan]) -> tuple[AnalysisStep, ...]:
    if plan is None:
        return ()
    return tuple(
        step
        for step in plan.steps
        if step.planned_analysis_role in {"primary", "secondary", "sensitivity"}
    )


def association_study(plan: Optional[AnalysisPlan]) -> bool:
    if plan is None:
        return False
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(plan.analysis_type or "").strip().lower()
    ).strip("_")
    if any(token in normalized for token in ("association", "regression", "effect")):
        return True
    return any(
        step.planned_analysis_role == "primary"
        and any(
            token in " ".join((step.method or "", step.intent or "")).lower()
            for token in ("association", "odds ratio", "risk ratio", "hazard ratio")
        )
        for step in plan.steps
    )


def model_covariates(plan: Optional[AnalysisPlan]) -> tuple[str, ...]:
    if plan is None:
        return ()
    values: list[str] = []
    for step in plan.steps:
        if step.planned_analysis_role != "primary":
            continue
        for requirement in step.model_requirements or ():
            for value in requirement.covariates or ():
                text = str(value).strip()
                if text and text not in values:
                    values.append(text)
    if not values and plan.adjustment_proposal is not None:
        # A signed runtime owner replaced the primary's model requirement and
        # retained the reviewed roster plan-wide.
        values = [str(value).strip() for value in plan.adjustment_proposal.covariates]
    return tuple(values)


def planned_model_outcomes(
    plan: Optional[AnalysisPlan],
    context: Optional[ResearchContext] = None,
) -> tuple[str, ...]:
    """Return outcomes covered by exact executable analysis contracts.

    Most association owners declare their outcome in ``model_requirements``.
    The controlled ordered-stratified owner instead carries one fixed typed
    three-column contract and estimates both its binary and continuous outcome.
    Count those context-declared outcome inputs as coverage rather than forcing
    the Planner to invent a duplicate conventional model solely to satisfy the
    review projection.
    Native owners publish their endpoint contract without conventional model
    requirements. Review consumes that projection, never a method-name guess.
    It is not signature verification or execution authority.
    """

    if plan is None:
        return ()
    values: list[str] = []
    for step in scientific_steps(plan):
        for requirement in step.model_requirements or ():
            outcome = str(requirement.outcome or "").strip()
            if outcome and outcome not in values:
                values.append(outcome)
        # A descriptive question requires its declared summary, not a new
        # regression. Conversely a summary cannot substitute for a requested
        # model merely because it reads the same outcome column.
        if (
            canonical_analysis_family(plan.analysis_type) == "descriptive_epidemiology"
            and exposure_outcome_distribution_execution_verdict(step).claimed
        ):
            spec = step.exposure_outcome_distribution_spec
            if (
                spec is not None
                and {spec.exposure, spec.outcome}.issubset(step.inputs)
                and spec.outcome not in values
            ):
                values.append(spec.outcome)
        if context is not None and is_ordered_stratified_analysis_step(step):
            for input_key in step.inputs:
                descriptor = context.variable(str(input_key or "").strip())
                if descriptor is None or str(descriptor.role.value) != "outcome":
                    continue
                if descriptor.name not in values:
                    values.append(descriptor.name)
        if context is not None and static_prediction_execution_verdict(step).claimed:
            # The host static prediction owner fits exactly the context target
            # outcome from the declared model-column prefix and, by contract,
            # carries no conventional model requirement. Count that declared
            # endpoint like the ordered-stratified owner's typed inputs.
            outcome = str(context.target_outcome or "").strip()
            if (
                outcome
                and outcome in static_prediction_model_columns(step)
                and outcome not in values
            ):
                values.append(outcome)
        if context is not None:
            for outcome in declared_runtime_outcomes(step):
                descriptor = context.variable(outcome)
                if descriptor is not None and descriptor.role.value == "outcome" and outcome not in values:
                    values.append(outcome)
    return tuple(values)


def requested_outcomes(context: ResearchContext) -> tuple[str, ...]:
    """Project explicitly requested endpoints, never infer intent from availability."""

    values = [
        str(value).strip()
        for value in context.cohort.requested_outcome_columns or ()
        if str(value).strip()
    ]
    primary = str(context.target_outcome or "").strip()
    if primary and primary not in values:
        values.append(primary)
    return tuple(dict.fromkeys(values))


def requested_exposure_occurrence(
    context: ResearchContext, plan: Optional[AnalysisPlan]
) -> tuple[str, ...]:
    """Words by which the question asks how often its exposure occurs.

    Empty unless a step could answer it: an association or descriptive plan,
    an exposure with at least two closed levels, and a two-level target.
    """

    if plan is None or not (
        association_study(plan)
        or canonical_analysis_family(plan.analysis_type) == "descriptive_epidemiology"
    ):
        return ()
    exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    variables = {item.name: item for item in context.variables}
    if (
        not exposure
        or not outcome
        or len(closed_planning_levels_for(name=exposure, variables=variables)) < 2
        or len(closed_planning_levels_for(name=outcome, variables=variables)) != 2
    ):
        return ()
    return requested_exposure_occurrence_cues(context)


def requested_dose_response_on_two_levels(
    context: ResearchContext,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The dose-response cues of a question whose exposure has two levels.

    A gradient needs at least three ordered levels or a continuous exposure;
    across two levels there is one contrast, which answers another question.
    Returns the cues and the exposure's levels, or two empty tuples when the
    question does not ask for a gradient or its exposure can carry one.
    """

    cues = requested_dose_response_cues(context)
    exposure = str(context.primary_exposure or "").strip()
    if not cues or not exposure:
        return (), ()
    variables = {item.name: item for item in context.variables}
    levels = closed_planning_levels_for(name=exposure, variables=variables)
    if len(levels) != 2:
        return (), ()
    return cues, tuple(str(level) for level in levels)


def exposure_occurrence_steps(
    plan: Optional[AnalysisPlan], context: ResearchContext
) -> tuple[str, ...]:
    """Steps that report how often the primary exposure occurs in the study cohort.

    A claimed exposure/outcome distribution of the primary exposure counts, as
    does a two-column prevalence table naming it.  Either must read the
    population the study selected: a landmark analysis cohort keeps only the
    stays alive and observed at the landmark, so its level counts answer a
    different question.
    """

    if plan is None:
        return ()
    exposure = str(context.primary_exposure or "").strip()
    covering: list[str] = []
    for step in plan.steps:
        if exposure_outcome_distribution_execution_verdict(step).claimed:
            spec = step.exposure_outcome_distribution_spec
            reports = spec is not None and spec.exposure == exposure
        else:
            declared = [
                str(value).strip()
                for value in step.inputs
                if str(value).strip() and ":" not in str(value)
            ]
            reports = (
                (_method_head(step), tuple(step.expected_outputs)) in _PREVALENCE_STEP_SHAPES
                and exposure in declared
                and len(set(declared)) == 2
            )
        if reports and step_cohort_population(step=step, plan=plan) == "study_cohort":
            covering.append(step.step_id)
    return tuple(covering)


def model_covariate_plan_authority(
    plan: Optional[AnalysisPlan],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return Planner-owned adjustment rationale and timing from the primary model.

    This is distinct from ``UserPreferences`` authority.  When the researcher
    leaves covariate selection to EasyICU, these maps are part of the Agent's
    complete reviewable proposal rather than extra form fields for the user.
    """

    rationales: dict[str, str] = {}
    temporal_roles: dict[str, str] = {}
    if plan is None:
        return rationales, temporal_roles
    for step in plan.steps:
        if step.planned_analysis_role != "primary":
            continue
        for requirement in step.model_requirements or ():
            rationales.update(requirement.covariate_rationales)
            temporal_roles.update(requirement.covariate_temporal_roles)
    if not rationales and not temporal_roles and plan.adjustment_proposal is not None:
        rationales.update(plan.adjustment_proposal.covariate_rationales)
        temporal_roles.update(plan.adjustment_proposal.covariate_temporal_roles)
    return rationales, temporal_roles


def post_baseline_exposure(context: ResearchContext) -> tuple[bool, Optional[str]]:
    """Classify post-zero exposure opportunity from owner-issued coordinates.

    A concept's clinical definition window remains the first authority.  The
    Web host may also bind an exact physical feature-materialization window in
    ``data_constraints`` before a concrete wide-column descriptor exists (for
    example while candidate planning is metadata-only).  That outer window is
    *not* promoted to a clinical-definition anchor; it only establishes that
    a primary feature can be observed after ICU admission, so association
    plans must close early-event/exposure opportunity.
    """

    exposure = str(context.primary_exposure or "").strip()
    if not exposure:
        return False, None
    descriptor = context.variable(exposure)
    window = str(getattr(descriptor, "analysis_window", "") or "").strip()
    if window:
        return window_extends_after_anchor(window), window
    if descriptor is not None and owner_declared_baseline_static(descriptor):
        # The outer window bounds *measurement* opportunity inside the stay.
        # It cannot move an owner-declared baseline attribute (age, sex,
        # admission type) after time zero, and the host already relies on that
        # same declaration for baseline covariate timing.  A concept whose own
        # window is post-baseline was returned above, so this never overrides
        # physical window evidence.
        return False, None

    preferences = context.user_preferences
    raw_constraints = getattr(preferences, "data_constraints", None)
    if not isinstance(raw_constraints, str) or not raw_constraints.strip():
        return False, None
    try:
        constraints = json.loads(raw_constraints)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(constraints, Mapping):
        return False, None
    materialization = constraints.get("materialization_window")
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("role") != "outer_observation_window"
        or normalise_time_anchor(str(materialization.get("anchor") or ""))
        != "icu_admission"
    ):
        return False, None
    # This detects a risk in the host-declared physical window; it grants no
    # execution authority. An absent confirmation cannot hide that risk.
    if isinstance(materialization.get("hours"), bool):
        return False, None
    try:
        hours = float(materialization["hours"])
    except (KeyError, TypeError, ValueError):
        return False, None
    if not math.isfinite(hours) or hours <= 0:
        return False, None
    # This label deliberately names the physical coordinate, rather than
    # implying a phenotype definition or a follow-up horizon.
    return True, f"outer_materialization:icu_admission[0,{hours:g}]h"


def patient_identity_available(context: ResearchContext) -> bool:
    return context_patient_group_authority(context) is not None


#: The host's landmark survival suite.  It closes a temporal design only once
#: its signed runtime authority binds the step; a draft that names it is a
#: proposal the host has not sealed.
_LANDMARK_SURVIVAL_SUITE_METHOD = "signed_landmark_survival_suite"
_EXECUTABLE_TEMPORAL_METHODS = frozenset(
    {
        "signed_landmark_restricted_cubic_spline",
        "signed_landmark_categorical_association",
        "signed_landmark_survival_suite",
        "time_varying_exposure_model",
        "landmark_analysis",
    }
)
_EXECUTABLE_DEPENDENCE_METHODS = frozenset(
    {
        "cluster_robust_association",
        "mixed_effects_association",
        "mixed_effects_regression",
        "one_stay_per_patient_association",
        "first_stay_association",
    }
)
_DESCRIPTIVE_ONLY_STEP_SHAPES = frozenset(
    {
        ("descriptive_distribution", ("table:distribution_prevalence",)),
        ("descriptive_distribution_summary", ("table:distribution_prevalence",)),
        ("descriptive", ("table:exposure_outcome_distribution",)),
    }
)
_POST_BASELINE_OPPORTUNITY_LIMITATION = (
    "post_baseline_exposure_opportunity_unresolved"
)
#: Two-column prevalence tables that name the exposure among their inputs.
_PREVALENCE_STEP_SHAPES = frozenset(
    shape
    for shape in _DESCRIPTIVE_ONLY_STEP_SHAPES
    if shape[1] == ("table:distribution_prevalence",)
)


def _method_head(step: AnalysisStep) -> str:
    return str(step.method or "").strip().casefold().split("(", 1)[0].strip()


def _has_scientific_output(step: AnalysisStep) -> bool:
    return any(
        str(value).partition(":")[0] in {"table", "statistic", "model", "dataset", "artifact"}
        for value in step.expected_outputs
    )


def executable_scientific_step(step: AnalysisStep) -> bool:
    method = _method_head(step)
    return bool(
        step.planned_analysis_role in {"primary", "secondary", "sensitivity"}
        and method not in {"", "feasibility_protocol", "protocol", "visualization"}
        and _has_scientific_output(step)
    )


def descriptive_only_step(step: AnalysisStep) -> bool:
    """Whether one step has a closed, typed descriptive claim ceiling.

    Exact method/product pairs are used because arbitrary table names or prose
    such as "descriptive association" cannot prove that a model/effect estimate
    is absent.  A model or non-descriptive capability wins over the ceiling and
    keeps the plan on the inferential path.
    """

    contract = step.descriptive_claim
    declared_columns = [
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value).strip()
    ]
    shape = (_method_head(step), tuple(step.expected_outputs))
    closed_shape = bool(
        shape in _DESCRIPTIVE_ONLY_STEP_SHAPES
        and (
            step.exposure_outcome_distribution_spec is not None
            if shape == ("descriptive", ("table:exposure_outcome_distribution",))
            else len(declared_columns) == 2 and len(set(declared_columns)) == 2
        )
    )
    return bool(
        contract is not None
        and contract.claim_ceiling == "descriptive_only"
        and _POST_BASELINE_OPPORTUNITY_LIMITATION
        in contract.unresolved_limitations
        and closed_shape
        and not step.model_requirements
        and step.family_primary_result_requirement is None
        and step.scientific_capability
        in {None, DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID}
        and sole_typed_cohort_input(step) not in {None, ""}
    )


def _step_requires_temporal_inference(step: AnalysisStep) -> bool:
    if descriptive_only_step(step):
        return False
    if (
        step.exposure_outcome_distribution_spec is not None
        or step.model_requirements
        or step.family_primary_result_requirement is not None
        or step.scientific_capability is not None
        or step.functional_form_spec is not None
    ):
        return True
    return _method_head(step) in (
        _EXECUTABLE_TEMPORAL_METHODS | _EXECUTABLE_DEPENDENCE_METHODS
    )


def _signed_temporal_result_projection(
    step: AnalysisStep, plan: AnalysisPlan
) -> bool:
    """Recognize a sensitivity table projected from one signed estimator.

    The landmark runtime can mechanically expose separate named sensitivity
    tables after its single signed fit.  Those child nodes consume only the
    signed primary's products; they do not refit patient rows and therefore do
    not need a second temporal estimator.  The graph and sensitivity ids must
    both close exactly so Planner prose cannot obtain this exemption.
    """

    if (
        step.planned_analysis_role != "sensitivity"
        or step.model_requirements
        or step.family_primary_result_requirement is not None
        or not step.sensitivity_spec_ids
    ):
        return False
    step_cohort_inputs = {
        str(value)
        for value in step.inputs
        if is_closed_cohort_product_key(str(value))
    }
    signed_landmark_primaries = [
        primary
        for primary in plan.steps
        if primary.planned_analysis_role == "primary"
        and _method_head(primary) == "signed_landmark_categorical_association"
        and any(
            str(ref).startswith("scientific_runtime_contract:")
            for ref in primary.icu_rule_refs
        )
        and sole_typed_cohort_input(primary) not in {None, ""}
        and {str(sole_typed_cohort_input(primary))} == step_cohort_inputs
        and bool(set(primary.expected_outputs) & set(step.inputs))
    ]
    if len(signed_landmark_primaries) == 1:
        # These are refits or projections over the same already-filtered,
        # digest-bound landmark cohort. This closes only their temporal
        # eligibility; their own method/execution contracts remain subject to
        # the separate sensitivity-capability gates.
        return True
    if step.functional_form_spec is not None and step.scientific_capability is None:
        # The signed spline owner refits a covariate on its own exact landmark
        # rows, or projects an exact exposure comparison. This recognizes only
        # temporal closure; runtime authority still validates method and target.
        parents = [primary for primary in plan.steps
                   if primary.planned_analysis_role == "primary"
                   and _method_head(primary) == "signed_landmark_restricted_cubic_spline"
                   and any(str(ref).startswith("scientific_runtime_contract:")
                           for ref in set(primary.icu_rule_refs) & set(step.icu_rule_refs))
                   and set(step.inputs) <= set(primary.inputs) | set(primary.expected_outputs)
                   and set(step.inputs) & set(primary.expected_outputs)
                   and step.functional_form_spec.target_column in primary.inputs]
        if len(parents) == 1:
            return True
    if step.scientific_capability is not None:
        return False
    candidates = [
        primary
        for primary in plan.steps
        if primary.planned_analysis_role == "primary"
        and _method_head(primary) in _EXECUTABLE_TEMPORAL_METHODS
        and set(step.sensitivity_spec_ids).issubset(primary.sensitivity_spec_ids)
        and set(step.inputs)
        and set(step.inputs).issubset(primary.expected_outputs)
    ]
    return len(candidates) == 1


def timing_design_closed(plan: Optional[AnalysisPlan]) -> bool:
    """Require every applicable estimator to close its own temporal design."""

    if plan is None:
        return False
    applicable = [
        step
        for step in scientific_steps(plan)
        if executable_scientific_step(step) and _step_requires_temporal_inference(step)
        and not _signed_temporal_result_projection(step, plan)
    ]
    return bool(applicable) and all(
        _method_head(step) in _EXECUTABLE_TEMPORAL_METHODS
        and (_method_head(step) != "time_varying_exposure_model" or (
            step.scientific_capability == "association_time_varying_exposure_v1"
            and _bound_to_runtime_contract(step)
        ))
        and (_method_head(step) != _LANDMARK_SURVIVAL_SUITE_METHOD or _bound_to_runtime_contract(step))
        for step in applicable
    )


def _signed_survival_suite_step(
    plan: AnalysisPlan,
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None,
) -> Optional[AnalysisStep]:
    """The one plan step the given signed landmark survival suite binds, if any."""

    if getattr(runtime_authority, "plan_method", None) != _LANDMARK_SURVIVAL_SUITE_METHOD:
        return None
    steps = [
        step for step in plan.steps
        if _method_head(step) == _LANDMARK_SURVIVAL_SUITE_METHOD
        and runtime_authority.plan_rule_ref in step.icu_rule_refs
        and runtime_authority.table_one_product in step.expected_outputs
    ]
    return steps[0] if len(steps) == 1 else None


def _bound_to_runtime_contract(step: AnalysisStep) -> bool:
    return any(str(ref).startswith("scientific_runtime_contract:") for ref in step.icu_rule_refs)


def temporal_inference_required(plan: Optional[AnalysisPlan]) -> bool:
    """Whether an exact executable step estimates exposure inference.

    Table 1, missingness and other supporting scientific tables can be marked
    secondary without becoming exposure estimators.  Treating every
    primary/secondary/sensitivity table as inference made a purely descriptive
    plan fail merely because it also reported its denominator.  This predicate
    therefore follows typed estimator ownership, not role labels or prose.
    """

    return any(_step_requires_temporal_inference(step) for step in scientific_steps(plan))


def _sensitivity_specs(context: ResearchContext) -> tuple[Any, ...]:
    """Return the optional user-owned sensitivity contract as an empty tuple.

    ``ResearchContext.user_preferences`` is nullable for legacy and generic
    programmatic callers.  Absence means that the user requested no additional
    sensitivity axis; it must not crash or silently bypass the remaining
    scientific review.
    """

    preferences = context.user_preferences
    return tuple(getattr(preferences, "sensitivity_specs", ()) or ())


_FIRST_STAY_RULE_METHODS = frozenset(
    {
        "non_readmission_restriction",
        *EXECUTABLE_METHODS_BY_STRATEGY["first_stay"],
    }
)


def _repeated_stay_step_declared(step: Any, context: Any) -> bool:
    """Whether one step explicitly declares its repeat-stay rule."""

    if _method_head(step) in _FIRST_STAY_RULE_METHODS:
        return True
    for requirement in step.model_requirements or ():
        if getattr(requirement, "dependence", None) is not None:
            return True
    distribution = getattr(step, "exposure_outcome_distribution_spec", None)
    if (
        distribution is not None
        and getattr(distribution, "schema_version", "")
        == "easyicu.exposure_outcome_distribution/3"
    ):
        # Counts-only descriptive projection: counts are counts, and no
        # independence-assuming estimate is produced here. (Independent-row
        # Table 1 tests stay blocked elsewhere.)
        return True
    if context is None:
        return False
    patient_group = context_patient_group_authority(context)
    if patient_group is not None and static_prediction_execution_verdict(step).claimed:
        # The host static prediction owner splits development and validation
        # rows by the verified patient group, reports the repeated-subject
        # structure of the evaluation partition, and fails closed without
        # that authority; its repeat-stay rule is therefore executable, not
        # prose.
        return True
    return bool(
        _method_head(step)
        in {"signed_landmark_restricted_cubic_spline", "time_varying_exposure_model"}
        and patient_group is not None
        and patient_group.group_source in (step.inputs or ())
    )


def _repeated_stay_rule_declared(
    plan: Optional[AnalysisPlan],
    context: Any = None,
    sensitivity_executable_axes: Sequence[str] = (),
) -> bool:
    """Whether the repeat-stay rule is explicitly declared somewhere.

    Checkable exits only, no prose inference (review finding: the
    remediation promises that confirming clustered/mixed handling clears
    the finding, so those confirmations must count here too): a first-stay
    method, a bound dependence contract, an executed repeated-stays
    sensitivity axis, a counts-only descriptive projection, or a reviewed
    signed runtime bound to patient grouping.
    """

    if plan is None:
        return False
    if any(
        _repeated_stay_step_declared(step, context) for step in plan.steps
    ):
        return True
    return "readmission" in set(sensitivity_executable_axes or ())


def _signed_landmark_dependence(step: AnalysisStep, context: ResearchContext) -> bool:
    patient_group = context_patient_group_authority(context)
    return bool(
        _method_head(step)
        in {"signed_landmark_restricted_cubic_spline", "time_varying_exposure_model"}
        and patient_group is not None
        and patient_group.group_source in step.inputs
        and any(
            str(value).startswith("scientific_runtime_contract:")
            for value in step.icu_rule_refs
        )
    )


def _repeated_unit_consumer(step: AnalysisStep, context: ResearchContext) -> bool:
    """Whether one step's estimator can carry a dependence contract."""

    method = _method_head(step)
    return bool(
        step.model_requirements
        or step.exposure_outcome_distribution_spec is not None
        or method in _EXECUTABLE_DEPENDENCE_METHODS
        or method == "signed_landmark_restricted_cubic_spline"
        or _signed_landmark_dependence(step, context)
        or method == "non_readmission_restriction"
    )


def repeated_unit_estimator_present(
    context: ResearchContext, plan: Optional[AnalysisPlan]
) -> bool:
    """Whether a plan revision has an estimator to bind patient dependence to.

    A plan whose executable steps all fit one row per ICU stay through host
    actions without a model, interval, or baseline-table contract (a
    cross-sectional clustering suite, for example) gives a revision nothing to
    bind: only a population that keeps one stay per patient closes its
    repeated stays.
    """

    if plan is None:
        return False
    if any(step.table_one_spec is not None for step in plan.steps):
        return True
    return any(
        executable_scientific_step(step) and _repeated_unit_consumer(step, context)
        for step in scientific_steps(plan)
    )


def repeated_unit_design_closed(
    context: ResearchContext, plan: Optional[AnalysisPlan]
) -> bool:
    table_one_steps = [
        step for step in (plan.steps if plan is not None else ())
        if step.table_one_spec is not None
    ]
    if any(step.table_one_spec.p_values_required for step in table_one_steps):
        return False
    applicable = bool(table_one_steps)
    for step in scientific_steps(plan):
        if not executable_scientific_step(step):
            continue
        if not _repeated_unit_consumer(step, context):
            continue
        method = _method_head(step)
        model_requirements = tuple(step.model_requirements)
        distribution = step.exposure_outcome_distribution_spec
        signed_landmark_dependence = _signed_landmark_dependence(step, context)
        counts_only_distribution = bool(
            distribution is not None
            and distribution.schema_version
            == "easyicu.exposure_outcome_distribution/3"
        )
        applicable = True
        has_patient_authority = patient_identity_available(context)
        if model_requirements and not (
            has_patient_authority
            and all(
                dependence_matches_context(
                    context=context,
                    dependence=requirement.dependence,
                )
                for requirement in model_requirements
            )
        ):
            return False
        if distribution is not None and not counts_only_distribution and not (
            has_patient_authority
            and dependence_matches_context(
                context=context,
                dependence=distribution.dependence,
            )
        ):
            return False
        # A mixed product step must close every covariance consumer above;
        # neither its model nor its marginal distribution may borrow the
        # other's authority. Once both present contracts are closed, the step
        # is complete regardless of the human-readable method label.
        if model_requirements or distribution is not None:
            continue
        if signed_landmark_dependence:
            continue
        if has_patient_authority and method in _EXECUTABLE_DEPENDENCE_METHODS:
            continue
        if method == "non_readmission_restriction" and any(
            variable == "icu_readmission"
            for spec in _sensitivity_specs(context)
            if spec.spec_id in step.sensitivity_spec_ids
            and spec.axis == "repeated_stays"
            and spec.strategy == "non_readmission_restriction"
            for variable in spec.execution_variables
        ):
            continue
        return False
    return applicable


def required_method_layers_for_context(
    context: ResearchContext,
) -> tuple[str, ...]:
    """Return method layers already fixed before plan generation.

    These decisions come from sealed study authority, so publishing them in the
    initial Planner contract avoids spending a retry merely to discover that a
    required method card was applicable. Plan-dependent decisions remain in
    :func:`required_method_layers_for_plan`.
    """

    required = {"reporting_standard"}
    if post_baseline_exposure(context)[0]:
        required.add("time_alignment")
    if repeat_units_possible(context):
        required.add("dependence")
    return tuple(sorted(required))


def required_method_layers_for_plan(
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[str, ...]:
    """Return the case-neutral method decisions this exact plan must source."""

    steps = scientific_steps(plan)
    if not steps:
        return ()
    required = set(required_method_layers_for_context(context))
    if any(step.model_requirements for step in steps):
        required.add("interpretation")
        if any(
            term.coding == "continuous"
            for step in steps
            for requirement in step.model_requirements
            for term in (requirement.model_terms or ())
        ):
            required.add("functional_form")
    if any(spec.axis == "missing" for spec in plan.robustness_specs) or any(
        "missing" in " ".join([step.intent or "", step.method or ""]).casefold()
        for step in steps
    ):
        required.add("missing_data")
    plan_tokens = " ".join(
        [
            str(plan.analysis_type or ""),
            *(
                token
                for step in steps
                for token in (
                    str(step.method or ""),
                    str(step.intent or ""),
                    *(str(output or "") for output in step.expected_outputs),
                )
            ),
        ]
    ).casefold()
    if str(plan.analysis_type or "").casefold() == "survival":
        if any(
            marker in plan_tokens
            for marker in (
                "cox",
                "proportional hazard",
                "ph_diagnostic",
                "ph diagnostic",
            )
        ):
            required.add("survival_assumption")
        if any(
            marker in plan_tokens
            for marker in (
                "rmst",
                "restricted mean survival",
                "restricted_mean_survival",
            )
        ):
            required.add("survival_estimand")
    return tuple(sorted(required))


def method_source_facts(
    plan: AnalysisPlan,
    context: ResearchContext,
) -> dict[str, Any]:
    """Project exact, card-supported method bindings for both review phases."""

    layers_by_step: dict[str, list[str]] = {}
    method_source_gaps: list[str] = []
    unsupported_bindings: list[dict[str, Any]] = []
    scientific_step_ids = {str(step.step_id) for step in scientific_steps(plan)}
    for step in plan.steps:
        layers: set[str] = set()
        for binding in step.literature_design_bindings:
            support = method_binding_support(
                binding.citation_key,
                binding.design_elements,
            )
            layers.update(support["matched_layers"])
            if support["method_source"] and support["unsupported_design_elements"]:
                unsupported_bindings.append(
                    {
                        "step_id": str(step.step_id),
                        "citation_key": binding.citation_key,
                        "unsupported_design_elements": support[
                            "unsupported_design_elements"
                        ],
                        "matched_card_ids": support["matched_card_ids"],
                    }
                )
        sorted_layers = sorted(layers)
        layers_by_step[str(step.step_id)] = sorted_layers
        if not sorted_layers and str(step.step_id) in scientific_step_ids:
            method_source_gaps.append(str(step.step_id))
    cited_layers = sorted({layer for values in layers_by_step.values() for layer in values})
    required_layers = list(required_method_layers_for_plan(plan, context))
    return {
        "method_source_gaps": method_source_gaps,
        "method_layers_by_step": layers_by_step,
        "required_method_layers": required_layers,
        "cited_method_layers": cited_layers,
        "missing_method_layers": sorted(set(required_layers) - set(cited_layers)),
        "unsupported_method_bindings": unsupported_bindings,
    }


def _literature_facts(
    literature: Optional[LiteratureBundle],
    context: ResearchContext,
) -> dict[str, Any]:
    if literature is None:
        return {
            "search_conducted": False,
            "sources_returning": [],
            "queries": {},
            "direct_comparator_keys": [],
            "design_analogue_keys": [],
            "comparison_source_keys": [],
            "direct_comparator_years": [],
            "comparison_source_years": [],
            "newest_direct_comparator_year": None,
            "newest_comparison_source_year": None,
            "search_year": datetime.now(timezone.utc).year,
        }
    provenance = literature.search_provenance
    citations = {item.key: item for item in literature.citations}
    direct_keys = sorted(
        {
            item.citation_key
            for item in literature.screening_decisions
            if item.disposition == "include"
            and item.evidence_role == "direct_comparator"
            and item.population_match
            and item.exposure_match
            and item.outcome_match
            and item.design_excerpt_available
            and item.publication_type_eligible
        }
    )
    analogue_keys = sorted(
        {
            item.citation_key
            for item in literature.screening_decisions
            if item.disposition == "include"
            and item.evidence_role == "design_analogue"
            and item.population_match
            and item.design_excerpt_available
            and item.publication_type_eligible
        }
    )
    comparison_keys = sorted(
        set(direct_keys)
        | (set(analogue_keys) if not context.primary_exposure else set())
    )
    years = sorted(
        {
            int(citations[key].year)
            for key in direct_keys
            if key in citations and str(citations[key].year).isdigit()
        }
    )
    comparison_years = sorted(
        {
            int(citations[key].year)
            for key in comparison_keys
            if key in citations and str(citations[key].year).isdigit()
        }
    )
    search_year = datetime.now(timezone.utc).year
    if provenance is not None:
        match = re.search(r"\b(20\d{2})\b", str(provenance.searched_at or ""))
        if match:
            search_year = int(match.group(1))
    return {
        "search_conducted": bool(provenance and provenance.search_conducted),
        "sources_returning": list(provenance.sources_returning if provenance else ()),
        "queries": dict(provenance.search_queries if provenance else {}),
        "direct_comparator_keys": direct_keys,
        "design_analogue_keys": analogue_keys,
        "comparison_source_keys": comparison_keys,
        "direct_comparator_years": years,
        "comparison_source_years": comparison_years,
        "newest_direct_comparator_year": years[-1] if years else None,
        "newest_comparison_source_year": (
            comparison_years[-1] if comparison_years else None
        ),
        "search_year": search_year,
    }


def _literature_design_bindings(
    plan: AnalysisPlan,
    literature: Optional[LiteratureBundle],
) -> dict[str, Any]:
    """Join typed Planner adoption claims to sealed source evidence."""

    citations = {
        item.key: item for item in manuscript_citable_records(literature)
    }
    screening = (
        {item.citation_key: item for item in literature.screening_decisions}
        if literature is not None
        else {}
    )
    bindings: list[dict[str, Any]] = []
    unresolved_steps: list[str] = []
    unexplained_citations_by_step: dict[str, list[str]] = {}
    for step in scientific_steps(plan):
        step_bindings: list[dict[str, Any]] = []
        for binding in step.literature_design_bindings:
            key = binding.citation_key
            record = citations.get(key)
            if record is None:
                continue
            decision = screening.get(key)
            step_bindings.append(
                {
                    "citation_key": key,
                    "title": record.title,
                    "year": record.year,
                    "source_excerpt": str(record.relevance or "")[:900] or None,
                    "evidence_role": (
                        decision.evidence_role if decision is not None else "method_or_context"
                    ),
                    "design_elements": list(binding.design_elements),
                    "application": binding.application,
                    "divergence": binding.divergence,
                    "binding_status": "typed_source_joined",
                    "method_card_support": method_binding_support(
                        key,
                        binding.design_elements,
                    ),
                }
            )
        explained_keys = {
            str(row.get("citation_key") or "") for row in step_bindings
        }
        unexplained = sorted(
            set(step.literature_citation_keys) - explained_keys
        )
        if unexplained:
            unexplained_citations_by_step[step.step_id] = unexplained
        if not step_bindings or unexplained:
            unresolved_steps.append(step.step_id)
        bindings.append(
            {
                "step_id": step.step_id,
                "planned_analysis_role": step.planned_analysis_role,
                "citations": step_bindings,
                "design_binding_status": (
                    "typed_source_joined"
                    if step_bindings and not unexplained
                    else "citation_only_or_unresolved"
                ),
                "unexplained_citation_keys": unexplained,
            }
        )
    return {
        "steps": bindings,
        "unresolved_steps": unresolved_steps,
        "unexplained_citations_by_step": unexplained_citations_by_step,
        "all_scientific_steps_have_design_binding": not unresolved_steps,
        "boundary": (
            "Typed adoption is inspectable but is not proof of applicability. "
            "Human review must still compare the sealed source excerpt with the "
            "Planner's exact application and any declared divergence."
        ),
    }


def _requested_sensitivity_axes(context: ResearchContext) -> set[str]:
    def review_axis(axis: str) -> str:
        return {
            "repeated_stays": "readmission",
            "missing_data": "missing",
        }.get(axis, axis)

    # Only the typed StudyContext roster can create a required sensitivity.
    # Free text remains useful planning context but cannot survive negation,
    # so token scans here produced obligations users had explicitly declined.
    return {review_axis(spec.axis) for spec in _sensitivity_specs(context)}


def _signed_grid_spec_ids(
    context: ResearchContext,
    plan: AnalysisPlan,
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None,
) -> set[str]:
    """Credit only variants present in the validated categorical grid seal."""

    if not isinstance(runtime_authority, LandmarkCategoricalAssociationRuntimeAuthority):
        return set()
    grid = runtime_authority.association_model_grid
    if grid is None:
        return set()
    try:
        runtime_authority.validate_plan(plan)
    except ValueError:
        return set()
    operationalizations = dict(
        AdjustmentSetAuthority.from_context(context).operationalizations
    )
    return grid.covered_prespecified_spec_ids(
        _sensitivity_specs(context),
        operationalizations=operationalizations,
    )


#: Strategies a signed primary executes as coordinates of its own design: its
#: landmark, its cluster-robust variance, and the spline form of its exposure.
#: A linear per-unit refit is an alternative form and stays a sensitivity.
_PRIMARY_DESIGN_STRATEGIES = frozenset({"landmark", "cluster_robust", "restricted_cubic_spline"})


def _sensitivity_facts(
    context: ResearchContext,
    plan: AnalysisPlan,
    *,
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None = None,
) -> dict[str, Any]:
    requested = _requested_sensitivity_axes(context)
    typed_specs = {spec.spec_id: spec for spec in _sensitivity_specs(context)}
    operationalizations = dict(AdjustmentSetAuthority.from_context(context).operationalizations)
    unsupported_spec_ids = {
        spec_id
        for spec_id, spec in typed_specs.items()
        if (not EXECUTABLE_METHODS_BY_STRATEGY[spec.strategy]
            or (spec.strategy == "time_varying" and spec.time_varying_execution is None))
    }

    def review_axis(spec: Any) -> str:
        return {
            "repeated_stays": "readmission",
            "missing_data": "missing",
        }.get(spec.axis, spec.axis)

    supported_axes = {
        review_axis(spec)
        for spec_id, spec in typed_specs.items()
        if spec_id not in unsupported_spec_ids
    }
    unsupported_only_axes = {
        review_axis(spec)
        for spec_id, spec in typed_specs.items()
        if spec_id in unsupported_spec_ids
    } - supported_axes
    executed_spec_ids: set[str] = set()
    # Specs the signed primary executes as its own design: executed, but they
    # restate the primary analysis rather than vary it.
    primary_design_spec_ids: set[str] = set()
    survival_suite = _signed_survival_suite_step(plan, runtime_authority)
    executable: set[str] = set()
    typed_executable: set[str] = set()
    protocol_only: set[str] = set()
    for step in scientific_steps(plan):
        text = " ".join(
            [
                step.step_id,
                step.intent,
                step.method or "",
                *step.expected_outputs,
                *step.icu_rule_refs,
            ]
        ).casefold()
        axes: set[str] = set()
        if any(token in text for token in ("timing", "landmark", "time-varying", "time varying")):
            axes.add("timing")
        if any(token in text for token in ("readmission", "re-admission", "first stay")):
            axes.add("readmission")
        if any(token in text for token in ("missing", "complete case")):
            axes.add("missing")
        if executable_scientific_step(step):
            executable.update(axes)
            method = _method_head(step)
            if (
                step.planned_analysis_role == "sensitivity"
                and method in FUNCTIONAL_FORM_EXECUTABLE_METHODS
                and (
                    step.scientific_capability == ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID
                    or _signed_temporal_result_projection(step, plan)
                )
                and step.sensitivity_spec_ids
                and step.functional_form_spec is not None
                and bool(step.expected_outputs)
                and str(step.expected_outputs[0]).startswith("table:")
                and tuple(step.expected_outputs) == functional_form_products(
                    step.expected_outputs[0], include_effects=len(step.expected_outputs) != 1,
                )
                and (len(step.expected_outputs) == 1 or step.scientific_capability is None)
            ):
                # The progressive compiler signs this exact custom-sensitivity
                # shape against the primary adjusted-association product. It is
                # plan-owned typed authority awaiting whole-plan approval, not
                # a prose mention or an unregistered robustness-axis alias.
                executable.add("functional_form")
                typed_executable.add("functional_form")
            for spec_id in step.sensitivity_spec_ids:
                spec = typed_specs.get(spec_id)
                if (
                    spec is not None
                    and method != "verified_association_model_grid"
                    and method in EXECUTABLE_METHODS_BY_STRATEGY[spec.strategy]
                    and (
                        spec.axis != "functional_form" or (
                            step.functional_form_spec is not None
                            and tuple(operationalizations.get(name, name) for name in spec.execution_variables)
                            == (step.functional_form_spec.target_column,)
                        )
                    )
                ):
                    executed_spec_ids.add(spec_id)
            # A signed runtime method is the host-bound implementation of the
            # exact StudyContext digest. Its primary step may execute typed
            # landmark, spline, and linear contracts without mislabelling the
            # primary estimator as a separate sensitivity step. Credit only
            # strategies explicitly supported by that signed method and only
            # when their source coordinates are present in the governed step.
            # Ordinary Planner prose and generic method names never reach this
            # branch.
            if method == "signed_landmark_restricted_cubic_spline":
                step_inputs = set(step.inputs)
                for spec_id, spec in typed_specs.items():
                    if method not in EXECUTABLE_METHODS_BY_STRATEGY[spec.strategy]:
                        continue
                    if spec.axis == "functional_form" and spec.execution_variables != (context.primary_exposure,):
                        continue
                    required_inputs = set(spec.execution_variables)
                    if spec.strategy == "landmark":
                        required_inputs.update(
                            value
                            for value in (
                                spec.event_time_variable,
                                spec.observation_duration_variable,
                            )
                            if value
                        )
                    if required_inputs.issubset(step_inputs):
                        executed_spec_ids.add(spec_id)
                        if spec.strategy in _PRIMARY_DESIGN_STRATEGIES:
                            primary_design_spec_ids.add(spec_id)
            if survival_suite is not None and step.step_id == survival_suite.step_id:
                # The signed survival suite executes the declared landmark
                # design itself: the same landmark and follow-up coordinate.
                for spec_id, spec in typed_specs.items():
                    timing = {spec.event_time_variable, spec.observation_duration_variable} - {None}
                    if (
                        spec.strategy == "landmark"
                        and spec.landmark_hours == runtime_authority.landmark_hours
                        and timing
                        and timing <= {runtime_authority.followup_time_column}
                    ):
                        executed_spec_ids.add(spec_id)
                        primary_design_spec_ids.add(spec_id)
            if method == "signed_landmark_categorical_association":
                signed_refs = {
                    str(ref)
                    for ref in step.icu_rule_refs
                    if str(ref).startswith("scientific_runtime_contract:")
                }
                cohort_owners = [
                    candidate
                    for candidate in plan.steps
                    if _method_head(candidate) == "signed_landmark_analysis_cohort"
                    and signed_refs
                    & {
                        str(ref)
                        for ref in candidate.icu_rule_refs
                        if str(ref).startswith("scientific_runtime_contract:")
                    }
                ]
                if len(cohort_owners) == 1:
                    cohort_inputs = set(cohort_owners[0].inputs)
                    for spec_id, spec in typed_specs.items():
                        if spec.strategy != "landmark":
                            continue
                        required_inputs = {
                            value
                            for value in (
                                spec.event_time_variable,
                                spec.observation_duration_variable,
                            )
                            if value
                        }
                        if required_inputs and required_inputs.issubset(cohort_inputs):
                            executed_spec_ids.add(spec_id)
                            primary_design_spec_ids.add(spec_id)
        else:
            protocol_only.update(axes)
    executed_spec_ids.update(_signed_grid_spec_ids(context, plan, runtime_authority))
    replay_steps = [
        step
        for step in plan.steps
        if _method_head(step) == "robustness_sensitivity"
        and step.robustness_replay_spec is not None
        and _has_scientific_output(step)
    ]
    plan_specs_by_id = {spec.spec_id: spec for spec in plan.robustness_specs}
    for step in replay_steps:
        for spec_id in step.sensitivity_spec_ids:
            typed_spec = typed_specs.get(spec_id)
            plan_spec = plan_specs_by_id.get(spec_id)
            if typed_spec is None or plan_spec is None:
                continue
            missing_override = dict(plan_spec.missing_override or {})
            if (
                typed_spec.axis == "missing_data"
                and typed_spec.strategy == "complete_case"
                and plan_spec.axis == "missing"
                and str(missing_override.get("strategy") or "")
                .strip()
                .casefold()
                == "complete_case"
            ):
                # ``robustness_sensitivity`` is the deterministic replay owner
                # for a locked plan-level complete-case spec.  The context and
                # plan ids must agree exactly; prose or a method label alone is
                # never enough to credit execution.
                executed_spec_ids.add(spec_id)
    plan_axis_names = {
        "missing": "missing",
        "cohort": "cohort",
        "outcome": "outcome_definition",
    }
    # A plan-locked spec that a host owner executes inside its own step is not
    # waiting for a replay step: the static prediction owner refits the exact
    # model roster on complete cases beside its primary imputation.
    owner_executed_spec_ids = _owner_executed_plan_spec_ids(context, plan)
    owner_executed_axes = {
        plan_axis_names[spec.axis]
        for spec in plan.robustness_specs
        if spec.spec_id in owner_executed_spec_ids
    }
    # A refit that restates the primary is documented, never evidence.
    restating = set(_complete_case_specs_restating_primary(plan))
    plan_spec_axes = {
        plan_axis_names[spec.axis]
        for spec in plan.robustness_specs
        if spec.spec_id not in owner_executed_spec_ids
        and spec.spec_id not in restating
    }
    if len(replay_steps) == 1:
        executable.update(plan_spec_axes)
        typed_executable.update(plan_spec_axes)
    else:
        protocol_only.update(plan_spec_axes)
    executable.update(owner_executed_axes)
    typed_executable.update(owner_executed_axes)
    # A temporal phrase in a generic generated-code step is not proof that the
    # estimator closes immortal-time/exposure-opportunity bias.
    if "timing" in executable and not timing_design_closed(plan):
        executable.discard("timing")
        protocol_only.add("timing")
    if "readmission" in executable and not repeated_unit_design_closed(context, plan):
        executable.discard("readmission")
        protocol_only.add("readmission")
    missing_spec_ids = sorted(
        (set(typed_specs) - executed_spec_ids) - unsupported_spec_ids
    )
    typed_axes = {
        {
            "repeated_stays": "readmission",
            "missing_data": "missing",
        }.get(spec.axis, spec.axis)
        for spec_id, spec in typed_specs.items()
        if spec_id in executed_spec_ids
        and spec_id not in restating
        and spec_id not in primary_design_spec_ids
    }
    executable.update(typed_axes)
    typed_executable.update(typed_axes)
    # A sealed suite prespecifies its own sensitivity design under a contract
    # digest its executors require. That is typed, executable authority the
    # reviewer can read in the plan, so it counts -- but only for a step the
    # host actually sealed, never for a draft that spells the same method.
    # Every step of the plan, not only the Planner-role ones: a sealed suite
    # assigns its own step roles, and its stability owner is auxiliary by the
    # authority's design rather than by a Planner choice.
    sealed_axes = {
        axis
        for step in plan.steps
        for axis in sealed_suite_prespecified_axes(
            method=str(step.method or ""),
            rule_refs=step.icu_rule_refs,
            expected_outputs=step.expected_outputs,
        )
    }
    executable.update(sealed_axes)
    typed_executable.update(sealed_axes)
    # A host action whose owner runs a published robustness design -- the
    # phenotyping candidate-k grid and its resampling stability -- is
    # prespecified the same way: the plan may include the step but cannot
    # choose its settings. Only a step the owning executor would claim counts.
    host_action_axes = set(host_action_prespecified_axes(plan.steps))
    executable.update(host_action_axes)
    typed_executable.update(host_action_axes)
    return {
        "requested": sorted(requested),
        "sealed_suite_axes": sorted(sealed_axes),
        "host_action_axes": sorted(host_action_axes),
        "executable": sorted(executable),
        "typed_executable": sorted(typed_executable),
        "protocol_only": sorted(protocol_only - executable),
        "missing_required": sorted(
            requested - executable - unsupported_only_axes
        ),
        "typed_spec_ids": sorted(typed_specs),
        "executed_spec_ids": sorted(executed_spec_ids),
        "missing_spec_ids": missing_spec_ids,
        "unsupported_spec_ids": sorted(unsupported_spec_ids),
        "unsupported_strategies": sorted(
            {
                typed_specs[spec_id].strategy
                for spec_id in unsupported_spec_ids
            }
        ),
        "plan_robustness_spec_ids": sorted(
            spec.spec_id for spec in plan.robustness_specs
        ),
        "plan_robustness_replay_step_ids": sorted(
            step.step_id for step in replay_steps
        ),
        "owner_executed_plan_spec_ids": sorted(owner_executed_spec_ids),
        # Plan-locked specs with neither a single replay step nor an owner
        # that executes them in its own step; the run-level panel would fail
        # them closed after execution.
        "protocol_only_plan_spec_ids": (
            sorted(
                spec.spec_id
                for spec in plan.robustness_specs
                if spec.spec_id not in owner_executed_spec_ids
            )
            if len(replay_steps) != 1
            else []
        ),
    }


def _owner_executed_plan_spec_ids(
    context: ResearchContext, plan: AnalysisPlan
) -> set[str]:
    """Plan-locked specs the host static prediction owner executes itself.

    Asked through the owner's own contract: exactly one host-owned primary,
    its declared roster minus the outcome and patient group, and only the
    complete-case variant of that exact roster.
    """

    primaries = [
        step
        for step in plan.steps
        if step.scientific_action_id == PREDICTION_PRIMARY_ACTION
        and static_prediction_owns_step(step)
    ]
    outcome = str(context.target_outcome or "").strip()
    if len(primaries) != 1 or not outcome:
        return set()
    group = context_patient_group_authority(context)
    features = static_prediction_features(
        static_prediction_model_columns(primaries[0]),
        outcome=outcome,
        group_source=group.group_source if group is not None else None,
    )
    return {
        spec.spec_id
        for spec in plan.robustness_specs
        if static_prediction_executes_robustness_spec(
            spec, features=features, outcome=outcome
        )
    }


def _continuous_linearity_facts(plan: AnalysisPlan) -> dict[str, Any]:
    identity_terms: list[str] = []
    for step in scientific_steps(plan):
        for requirement in step.model_requirements:
            for term in requirement.model_terms or ():
                if term.role == "covariate" and term.coding == "continuous" and str(term.transform or "").casefold() in {"", "identity"}:
                    identity_terms.append(term.name)
    checked_targets = {
        step.functional_form_spec.target_column
        for step in scientific_steps(plan)
        if step.functional_form_spec is not None
        and step.planned_analysis_role == "sensitivity"
        and executable_scientific_step(step)
        and _method_head(step) in FUNCTIONAL_FORM_EXECUTABLE_METHODS
    }
    unchecked = set(identity_terms) - checked_targets
    return {
        "linear_identity_terms": sorted(set(identity_terms)),
        "checked_functional_form_targets": sorted(checked_targets),
        "unchecked_linear_identity_terms": sorted(unchecked),
        "functional_form_sensitivity_executable": bool(checked_targets) and not unchecked,
    }


def _model_term_domain_conflicts(
    context: ResearchContext,
    plan: AnalysisPlan,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for step in scientific_steps(plan):
        for requirement in step.model_requirements:
            for term in requirement.model_terms or ():
                variable = context.variable(term.name)
                if variable is None or term.coding != "continuous":
                    continue
                declared_levels, declared_basis = declared_domain_for_variable(variable)
                if not declared_levels:
                    continue
                conflicts.append(
                    {
                        "step_id": step.step_id,
                        "requirement_id": requirement.requirement_id,
                        "variable": term.name,
                        "coding": term.coding,
                        "declared_basis": declared_basis,
                        "declared_level_count": len(declared_levels),
                    }
                )
    return conflicts


def _endpoint_conflict_recorded(context: ResearchContext) -> bool:
    """Whether the context builder recorded that the question asks for another endpoint."""

    target = str(context.target_outcome or "").strip()
    descriptor = context.variable(target) if target else None
    return descriptor is not None and any(
        "endpoint-definition conflict" in str(value).casefold()
        for value in descriptor.clinical_caveats
    )


#: Families whose plans report a result on the study endpoint.
_ENDPOINT_RESULT_FAMILIES = frozenset(
    {
        "descriptive_epidemiology", "prediction_model", "dynamic_prediction",
        "ordinal_dose_response", "survival",
    }
)


def plan_reports_endpoint_result(plan: Optional[AnalysisPlan]) -> bool:
    """Whether the plan reports a result on the study endpoint."""

    return association_study(plan) or (
        plan is not None
        and canonical_analysis_family(plan.analysis_type) in _ENDPOINT_RESULT_FAMILIES
    )


def study_endpoint_required(
    context: ResearchContext, plan: Optional[AnalysisPlan]
) -> bool:
    """Whether the study needs a resolved endpoint definition.

    A declared target outcome needs one, and so does a plan that reports a
    result on the endpoint.  A discovery or audit plan whose study declares no
    outcome, such as trajectory clustering, has no endpoint to resolve.
    """

    return bool(str(context.target_outcome or "").strip()) or plan_reports_endpoint_result(
        plan
    )


def _endpoint_resolved(context: ResearchContext) -> bool:
    target = str(context.target_outcome or "").strip()
    descriptor = context.variable(target) if target else None
    if context.endpoint is None or descriptor is None:
        return False
    description = str(descriptor.description or "").strip()
    description_text = description.casefold()
    return bool(
        description
        and "mortality_unspecified" not in description_text
        and "declared_primary_outcome" not in description_text
        and not _endpoint_conflict_recorded(context)
    )


def _clinical_definition_facts(context: ResearchContext) -> dict[str, Any]:
    """Project clinical-definition provenance without inventing sign-off.

    Automated golden-vector conformance is valuable implementation evidence,
    but it is not an independent ICU-clinician review.  The typed descriptor
    already carries the owner registry's validation status; this owner makes
    the distinction visible before a plan can be described as a top-journal
    candidate.
    """

    names = list(
        dict.fromkeys(
            value
            for value in (
                str(context.primary_exposure or "").strip(),
                str(context.target_outcome or "").strip(),
            )
            if value
        )
    )
    rows: list[dict[str, Any]] = []
    pending: list[str] = []
    conformance_gaps: list[dict[str, str]] = []
    database = normalize_database_name(context.cohort.database)
    for name in names:
        descriptor = context.variable(name)
        definition = getattr(descriptor, "clinical_definition", None)
        if definition is None:
            continue
        status = str(definition.validation_status or "").casefold()
        independently_reviewed = bool(
            "independent_clinical_review_complete" in status
            or "independent_clinical_review_passed" in status
        ) and "pending" not in status
        database_conformance = str(
            definition.database_conformance.get(database, "not_assessed")
        )
        rows.append(
            {
                "variable": name,
                "contract_id": definition.contract_id,
                "definition": definition.definition,
                "version": definition.version,
                "source_id": definition.source_id,
                "definition_time_anchor": definition.definition_time_anchor,
                "status": definition.status,
                "validation_status": definition.validation_status,
                "canonical_definition": definition.canonical_definition,
                "ascertainment_limitations": list(
                    definition.ascertainment_limitations
                ),
                "database": database,
                "database_conformance": database_conformance,
                "independent_clinical_review_complete": independently_reviewed,
            }
        )
        if not independently_reviewed:
            pending.append(definition.contract_id)
        if database_conformance != "algorithm_golden":
            conformance_gaps.append(
                {
                    "variable": name,
                    "contract_id": definition.contract_id,
                    "database": database,
                    "conformance": database_conformance,
                }
            )
    return {
        "definitions": rows,
        "independent_clinical_review_pending_contracts": sorted(set(pending)),
        "database_conformance_gaps": conformance_gaps,
    }


def render_plan_scientific_guardrails(context: ResearchContext) -> str:
    """Render case-neutral, context-derived guardrails before Planner generation."""

    lines = ["PRE-APPROVAL SCIENTIFIC PLAN GUARDRAILS (host-derived):"]
    baseline = baseline_requirement_projection(context)
    if baseline["tables"]:
        lines.append(
            "- ACCEPTED BASELINE CONTENT: retain every required variable in an "
            "actual table_one_spec with the required grouping. Choose and explain "
            "any still-open value aggregation from the available host-declared "
            "columns; counts/timestamps, step-input mentions, and prose do not "
            "satisfy a clinical-value requirement. Unavailable items remain gaps, "
            "not permission to omit or substitute them. "
            + json.dumps(baseline, ensure_ascii=False, sort_keys=True)
        )
    lines.append("- " + DISTRIBUTION_MISSINGNESS_GUIDANCE)
    alignment = primary_exposure_time_anchor_alignment(context)
    if alignment.status in {"mismatch", "declared_only"}:
        lines.append(
            "- The sealed study time anchor and the owner-issued clinical "
            "definition of the primary exposure are not proven identical. The "
            "physical observation window is a separate coordinate and cannot "
            "repair that identity. Do not relabel or reinterpret either in a "
            "Plan; StudyContext/concept authority must issue a matching version "
            "before scientific execution."
        )
    post_baseline, window = post_baseline_exposure(context)
    if post_baseline:
        lines.append(
            "- The primary exposure is observed after the declared anchor "
            f"({window}). A feasibility/protocol report does not close this: "
            "plan an executable landmark, time-varying, or otherwise typed "
            "temporal estimator, or leave the plan non-approvable."
        )
    if repeat_units_possible(context):
        if patient_identity_available(context):
            lines.append(
                "- Repeated ICU stays are possible. Plan an executable one-stay-per-"
                "patient or clustered/mixed estimator; prose alone is insufficient."
            )
        else:
            lines.append(
                "- Repeated ICU stays are possible but patient identity is absent. "
                "Do not assume stay-level independence or claim clustered/first-stay "
                "analysis. State the materialization requirement and expect the "
                "pre-approval gate to stop article-grade association execution."
            )
    preferences = context.user_preferences
    if preferences is not None and preferences.covariate_selection == "planner_selectable":
        lines.append(
            "- Candidate covariates are not a user-approved adjustment set. Any exact "
            "roster emitted by the Planner is a proposal for explicit review, not a "
            "pre-specified fact; explain clinical rationale and time-zero availability."
        )
    requested = _requested_sensitivity_axes(context)
    if requested:
        lines.append(
            "- User-required sensitivity axes must be executable and re-estimate the "
            "relevant quantity. A feasibility_protocol/report does not satisfy them: "
            + ", ".join(sorted(requested))
            + "."
        )
    question = str(context.research_question or "").casefold()
    if any(
        token in question
        for token in (
            "association",
            "associated",
            "predict",
            "risk factor",
            "关联",
            "相关",
            "预测",
        )
    ):
        lines.extend(
            [
                "- If a continuous adjustment variable is selected, include an "
                "executable, source-bound functional-form check. Citing spline "
                "guidance without executing that check does not close the rule.",
                "- Article-grade association plans need at least two distinct, "
                "data-supported executable robustness axes. Protocol prose and "
                "duplicate replays of one axis do not count.",
                "- A literature source governs only the exact design elements "
                "supported by its displayed method card. Do not use citation "
                "presence to claim unrelated timing, dependence, interpretation, "
                "missing-data, or reporting coverage.",
            ]
        )
    return "\n".join(lines)


_EXTERNAL_EVIDENCE_FINDINGS = frozenset(
    {
        "TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED",
        "LITERATURE_SEARCH_PROVENANCE_INCOMPLETE",
        "DIRECT_COMPARATOR_NOT_ESTABLISHED",
        "DESIGN_ANALOGUE_NOT_ESTABLISHED",
        "RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED",
        "RECENT_DESIGN_ANALOGUE_NOT_ESTABLISHED",
        "NOVELTY_NOT_ESTABLISHED",
    }
)
_INDEPENDENT_REVIEW_FINDINGS = frozenset(
    {
        "NOVELTY_POSITIONING_REVIEW_REQUIRED",
        "CLINICAL_DEFINITION_INDEPENDENT_REVIEW_PENDING",
        "CLINICAL_DEFINITION_DATABASE_CONFORMANCE_NOT_ESTABLISHED",
    }
)
_RUNTIME_CAPABILITY_FINDINGS = frozenset(
    {
        "TIME_VARYING_RUNTIME_UNAVAILABLE",
        "REPEATED_STAY_IDENTITY_UNAVAILABLE",
    }
)


def remediation_route_for_finding(
    finding: PlanScientificFinding,
) -> ScientificRemediationRoute:
    """Assign one owner lane without letting the Agent revise the estimand."""

    if finding.requires_user_authorization:
        return "study_authority_change"
    if finding.remediation_route != "unclassified":
        return finding.remediation_route
    if finding.code in _RUNTIME_CAPABILITY_FINDINGS:
        return "runtime_capability"
    if finding.code in _EXTERNAL_EVIDENCE_FINDINGS:
        return "external_evidence"
    if finding.code in _INDEPENDENT_REVIEW_FINDINGS:
        return "independent_review"
    return "agent_plan_revision"


def plan_revision_blocker_codes(findings: list[PlanScientificFinding]) -> tuple[str, ...]:
    """Block futile plan retries until the responsible non-Planner owner acts.

    Major/minor maturity limitations do not prevent bounded plan repair.
    Blocking runtime, authority, evidence and independent-review gaps do.
    """
    return tuple(sorted({
        finding.code for finding in findings
        if finding.severity == "blocker"
        and remediation_route_for_finding(finding) != "agent_plan_revision"
    }))


def render_agent_plan_revision_contract(review: PlanScientificReview) -> str:
    """Render only plan-fixable findings from an exact prior review.

    Interactive hosts may bind this projection to a *fresh* Planner run when
    the StudyContext scientific digest is unchanged.  It never authorizes the
    Planner to answer questions that belong to the user, an external search,
    or an independent novelty reviewer.
    """

    automatic = [
        item
        for item in review.findings
        if remediation_route_for_finding(item) == "agent_plan_revision"
    ]
    if not automatic:
        return ""
    lines = [
        "DIGEST-BOUND PLAN REVISION CONTRACT (host-derived):",
        f"- source_plan_sha256: {review.plan_sha256}",
        f"- source_context_sha256: {review.context_sha256}",
        "- scope: generate a fresh plan; never mutate or resume the reviewed plan.",
        "- preserve the exact research question, cohort, exposure, all typed "
        "outcomes, time window, and user-authorized covariate authority.",
        "- do not claim that this revision closes study-authority, external-"
        "evidence, or independent-review findings.",
        "- fix these plan-owned findings with executable typed steps:",
    ]
    lines.extend(
        f"  - {item.code}: {item.remediation}" for item in automatic
    )
    return "\n".join(lines)


def primary_model_retention_findings(
    evidence: Optional[PrimaryModelRetentionEvidence],
) -> list[PlanScientificFinding]:
    """Judge the primary model by the rows it would fit, not by its labels.

    Nothing is said without a measurement (``None``), and candidate planning
    that reads no rows records the fact only.  Every finding routes to the
    runtime owner: a Planner must not "repair" a lost cohort by dropping the
    confounders whose missingness caused it.
    """

    if evidence is None or evidence.status == "not_applicable":
        return []
    refs = ["scientific_plan_review.json.facts.primary_model_retention"]
    if evidence.status == "rows_unavailable":
        if evidence.reason_code == "metadata_only_planning":
            return []
        return [
            PlanScientificFinding(
                code="PRIMARY_MODEL_RETENTION_UNVERIFIED",
                severity="minor",
                dimension="statistical_design",
                message=(
                    "The rows the primary model would fit could not be counted before "
                    f"approval ({evidence.reason_code})."
                ),
                evidence_refs=refs,
                remediation=(
                    "Count the primary model's rows on the cohort execution binds; the "
                    "fitted denominator is still audited after execution."
                ),
                remediation_route="runtime_capability",
            )
        ]
    if evidence.status == "not_evaluable":
        return [
            PlanScientificFinding(
                code="PRIMARY_MODEL_NOT_EVALUABLE_ON_SEALED_COHORT",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The primary model cannot be evaluated on the cohort execution will "
                    f"read ({evidence.reason_code}); the approved run would fail there."
                ),
                evidence_refs=refs,
                remediation=(
                    "Bind the primary model to columns and levels the sealed cohort "
                    "actually carries before approval."
                ),
                remediation_route="runtime_capability",
            )
        ]
    if evidence.status == "probe_failed":
        return [
            PlanScientificFinding(
                code="PRIMARY_MODEL_RETENTION_PROBE_FAILED",
                severity="major",
                dimension="statistical_design",
                message=(
                    "Counting the primary model's rows failed unexpectedly "
                    f"({evidence.reason_code}); its retention is unknown."
                ),
                evidence_refs=refs,
                remediation="Repair the retention measurement before relying on the plan.",
                remediation_route="runtime_capability",
            )
        ]
    findings: list[PlanScientificFinding] = []
    for item in evidence.requirements:
        if item.retention is None or item.retention >= RETENTION_MAJOR_BELOW:
            continue
        drivers = sorted(
            (
                entry
                for entry in item.covariates
                if entry.handling != "unmeasured_category" and entry.n_missing
            ),
            key=lambda entry: (-entry.n_missing, entry.name),
        )[:3]
        driver_text = (
            "; unmeasured "
            + ", ".join(f"{entry.name} ({entry.n_missing} rows)" for entry in drivers)
            if drivers
            else ""
        )
        rate_text = (
            f"; outcome rate {item.outcome_rate_retained:.1%} in fitted rows vs "
            f"{item.outcome_rate_dropped:.1%} in dropped rows"
            if item.outcome_rate_retained is not None
            and item.outcome_rate_dropped is not None
            else ""
        )
        insufficient = item.retention < RETENTION_BLOCKER_BELOW
        findings.append(
            PlanScientificFinding(
                code=(
                    "PRIMARY_MODEL_RETENTION_INSUFFICIENT"
                    if insufficient
                    else "PRIMARY_MODEL_RETENTION_REDUCED"
                ),
                severity="blocker" if insufficient else "major",
                dimension="statistical_design",
                message=(
                    f"The primary model {item.requirement_id} would fit {item.model_n} of "
                    f"{item.evaluable_n} evaluable rows ({item.retention:.0%})"
                    f"{driver_text}{rate_text}."
                ),
                evidence_refs=refs,
                remediation=(
                    "Keep frequently unmeasured confounders' rows as an explicit "
                    "unmeasured state where the primary owner supports it, or change the "
                    "design so the fitted rows answer the reviewed question; the "
                    "complete-case fit stays a sensitivity analysis."
                ),
                remediation_route="runtime_capability",
            )
        )
    return findings


def _missing_category_complete_case_findings(plan: AnalysisPlan) -> list[PlanScientificFinding]:
    """A kept unmeasured state needs the complete-case fit beside it."""

    covered = [
        set(variables)
        for spec in plan.robustness_specs
        for variables in [complete_case_variables(spec)]
        if variables
    ]
    uncovered = [
        requirement.requirement_id
        for step in plan.steps
        for requirement in (step.model_requirements or ())
        if requirement.analysis_role == "primary"
        and requirement.missing_category_covariates()
        and not any(
            set(requirement.missing_category_covariates()) <= variables
            for variables in covered
        )
    ]
    if not uncovered:
        return []
    return [
        PlanScientificFinding(
            code="MISSING_CATEGORY_WITHOUT_COMPLETE_CASE_SENSITIVITY",
            severity="major",
            dimension="robustness",
            message=(
                "The primary model keeps unmeasured covariate rows as their own state "
                "but no prespecified complete-case refit covers those covariates: "
                + ", ".join(uncovered)
            ),
            evidence_refs=[
                "analysis_plan.json.model_requirements",
                "analysis_plan.json.robustness_specs",
            ],
            remediation=(
                "Prespecify a complete-case refit whose locked variables include every "
                "covariate kept as unmeasured."
            ),
            remediation_route="agent_plan_revision",
        )
    ]


def _primary_model_columns(requirement: Any) -> set[str]:
    """Every column whose missingness removes a row from this model's fit.

    Model terms need no separate reading: the requirement's validator holds
    them to the exposure and the covariates.
    """

    columns = {requirement.exposure_source, requirement.outcome}
    columns.update(requirement.covariates or ())
    if requirement.dependence is not None:
        columns.add(requirement.dependence.group_source)
    return columns


def _complete_case_specs_restating_primary(plan: AnalysisPlan) -> tuple[str, ...]:
    """Locked complete-case refits over exactly the rows the primary fits.

    Restricting to rows complete in columns the primary model already
    requires, none of them kept as an unmeasured state, drops no row the
    primary fits, so the refit reproduces the primary estimate.  Judged
    against every primary model, since the plan does not say which one a
    replay binds.
    """

    primaries = [
        requirement
        for step in plan.steps
        for requirement in (step.model_requirements or ())
        if requirement.analysis_role == "primary"
    ]
    if not primaries:
        return ()
    return tuple(
        spec.spec_id
        for spec in plan.robustness_specs
        for variables in [complete_case_variables(spec)]
        if variables
        and spec.cohort_override is None
        and not spec.outcome_override
        and all(
            set(variables) <= _primary_model_columns(requirement)
            and not set(variables) & set(requirement.missing_category_covariates())
            for requirement in primaries
        )
    )


def _complete_case_repeats_primary_findings(plan: AnalysisPlan) -> list[PlanScientificFinding]:
    """Say which complete-case refits restate the primary analysis.

    Which covariates keep unmeasured rows is the host's decision from measured
    missingness, so the Planner cannot see this coming; the review records it
    and credits no robustness axis to such a refit.
    """

    repeated = _complete_case_specs_restating_primary(plan)
    if not repeated:
        return []
    return [
        PlanScientificFinding(
            code="COMPLETE_CASE_SENSITIVITY_REPEATS_PRIMARY",
            severity="minor",
            dimension="robustness",
            message=(
                "These complete-case refits keep exactly the rows the primary model "
                "fits, so each restates the primary estimate; it is documented and "
                "not counted as robustness evidence: " + ", ".join(repeated)
            ),
            evidence_refs=[
                "analysis_plan.json.model_requirements",
                "analysis_plan.json.robustness_specs",
            ],
            remediation=(
                "Rely on sensitivity analyses that change an executable coordinate of "
                "the primary analysis; a complete-case refit differs from the primary "
                "only when the primary keeps unmeasured covariate rows."
            ),
            remediation_route="runtime_capability",
        )
    ]


def trajectory_representation_facts(
    context: ResearchContext, plan: AnalysisPlan
) -> Optional[dict[str, Any]]:
    """What a plan claiming trajectory classes actually clusters.

    ``None`` unless the plan claims the signed trajectory owners, its primary
    step declares the longitudinal trajectory action, or the question asks for
    trajectories (``analysis_types.longitudinal_trajectory_requested``, the
    cue the family router reads).  Cross-sectional phenotype discovery shares
    this analysis family and is not a trajectory claim by itself; answering a
    trajectory question with it is one.  A plan whose per-timepoint
    representation has an owner -- the signed fixed-window suite, the
    fixed-window contract over ordered windows of one concept, or the
    run-level contract of a bound long trajectory -- is left to that owner's
    gates.  Otherwise the primary clusters one value per ICU stay, and the
    facts name what the signed owner could model instead, under the design
    owner's rule (``contracts.trajectory_design.trajectory_coordinate_proposal``):
    the Host compiles exactly these coordinates, over the window the question
    states for its trajectories (``trajectory_window_design``), in the
    population the plan states (``trajectory_population_design``).  A stated
    window the owner cannot count is not replaced by its default, and a
    stated population it cannot apply is not dropped.
    """

    if canonical_analysis_family(plan.analysis_type) != "trajectory_clustering":
        return None
    primaries = [step for step in plan.steps if step.planned_analysis_role == "primary"]
    signed = signed_trajectory_plan_claimed(plan)
    if (
        not signed
        and not any(
            step.scientific_action_id == TRAJECTORY_PRIMARY_ACTION for step in primaries
        )
        and not longitudinal_trajectory_requested(context)
    ):
        return None
    longitudinal_owner = (
        "signed_fixed_window_suite"
        if signed
        else "fixed_window_columns"
        if any(
            trajectory_phenotyping_contract_applies(context=context, step=step)
            for step in primaries
        )
        else "bound_long_trajectory"
        if trajectory_context_is_bound(context)
        else None
    )
    outcomes = (
        *context.cohort.outcome_columns,
        *([context.target_outcome] if context.target_outcome else []),
    )
    proposal = trajectory_coordinate_proposal(
        plan,
        variables=context.variables,
        outcomes=outcomes,
        excluded=context.cohort.id_columns,
    )
    window = trajectory_window_design(
        trajectory_window_statements(str(context.research_question or ""))
    )
    # Demographics are fixed at admission; any other concept can change
    # inside the trajectory window.
    static_concepts = {
        name
        for variable in context.variables
        if str(getattr(variable.role, "value", variable.role)) == "demographic"
        for name in (variable.name, getattr(variable, "source_concept", None))
        if name
    }
    population = trajectory_population_design(
        plan.cohort, window, static_concepts=static_concepts
    )
    return {
        "longitudinal_owner": longitudinal_owner,
        **proposal,
        "trajectory_window": window,
        "trajectory_population": population,
        "executable": bool(
            proposal["executable"] and window["executable"] and population["executable"]
        ),
    }


def landmark_survival_suite_facts(
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
) -> Optional[dict[str, Any]]:
    """Publish the coordinates of a survival plan that names the landmark suite.

    ``None`` unless the plan's one primary step names the suite.  A step its
    signed runtime authority binds is sealed.  Otherwise the plan is the host's
    proposal: the exposure, the fixed-horizon endpoint with its paired
    follow-up, the landmark at the end of the host-bound feature window, and
    the reviewed roster the plan keeps as its ``adjustment_proposal``.  It is
    executable only when every coordinate closes from those owners.
    """

    if plan is None or canonical_analysis_family(plan.analysis_type) != "survival":
        return None
    primaries = [
        step for step in plan.steps
        if step.planned_analysis_role == "primary"
        and _method_head(step) == _LANDMARK_SURVIVAL_SUITE_METHOD
    ]
    if len(primaries) != 1:
        return None
    primary = primaries[0]
    if _bound_to_runtime_contract(primary):
        return {"sealed": True}
    exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    endpoint = fixed_horizon_mortality_endpoint(outcome)
    landmark = host_outer_feature_window_end_hours(context)
    proposal = plan.adjustment_proposal
    covariates = [str(name) for name in proposal.covariates] if proposal is not None else []
    rationales = dict(proposal.covariate_rationales) if proposal is not None else {}
    roles = dict(proposal.covariate_temporal_roles) if proposal is not None else {}
    executable = bool(
        exposure
        and endpoint is not None
        and landmark is not None
        and 0 < landmark < endpoint.horizon_days * 24.0
        and {exposure, endpoint.event_concept, endpoint.followup_concept}.issubset(primary.inputs)
        and proposal is not None
        and set(covariates).issubset(primary.inputs)
        and set(rationales) == set(covariates)
        and set(roles) == set(covariates)
    )
    return {
        "sealed": False,
        "executable": executable,
        "exposure": exposure or None,
        "event_column": endpoint.event_concept if endpoint is not None else (outcome or None),
        "followup_column": endpoint.followup_concept if endpoint is not None else None,
        "endpoint_horizon_days": float(endpoint.horizon_days) if endpoint is not None else None,
        "landmark_hours": float(landmark) if landmark is not None else None,
        "covariates": covariates,
        "covariate_rationales": rationales,
        "covariate_temporal_roles": roles,
    }


def landmark_survival_suite_findings(
    facts: Optional[Mapping[str, Any]],
) -> list[PlanScientificFinding]:
    """Block a plan that names the landmark survival suite the host has not sealed.

    When its coordinates close, the host can seal the suite from them, so the
    remedy is a runtime capability, not a question for the researcher.
    Otherwise the plan names an owner nothing can execute and must be revised.
    """

    if facts is None or facts.get("sealed"):
        return []
    refs = [
        "analysis_plan.json.steps",
        "analysis_plan.json.adjustment_proposal",
        "research_context.json.variables",
    ]
    if facts.get("executable"):
        roster = ", ".join(facts["covariates"]) or "no covariates"
        return [
            PlanScientificFinding(
                code="SURVIVAL_LANDMARK_OWNER_NOT_SEALED",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The survival plan names the landmark survival suite, but the study "
                    "declares no survival design, so the host has not sealed it. Its "
                    f"coordinates close: exposure {facts['exposure']}, endpoint "
                    f"{facts['event_column']} with follow-up {facts['followup_column']}, "
                    f"landmark {facts['landmark_hours']:g} h, adjustment for {roster}."
                ),
                evidence_refs=refs,
                remediation=(
                    "Compile these coordinates into the study's survival design and "
                    "replan on the signed landmark survival suite. Keep the question, "
                    "the exposure, the endpoint and the reviewed roster; the researcher "
                    "does not choose the method."
                ),
                remediation_route="runtime_capability",
            )
        ]
    return [
        PlanScientificFinding(
            code="SURVIVAL_LANDMARK_OWNER_NOT_SEALED",
            severity="blocker",
            dimension="statistical_design",
            message=(
                "The survival plan names the landmark survival suite, but the host has "
                "not sealed it and its coordinates do not close: it needs a fixed-horizon "
                "mortality endpoint with its follow-up, a host-bound landmark inside the "
                "horizon, and a reviewed roster with a rationale and timing for every "
                "covariate."
            ),
            evidence_refs=refs,
            remediation=(
                "Revise the plan so its primary step is an estimator this study can "
                "execute, or declare the survival design the suite needs."
            ),
            remediation_route="agent_plan_revision",
        )
    ]


def _trajectory_window_clause(window: Mapping[str, Any]) -> str:
    """Where the windows of the design the Host would compile lie."""

    if window.get("executable") is not True:
        return ""
    span = (
        f"{window['window_start_hours']}–{window['window_end_hours']} h after ICU "
        f"admission on a {window['grid_width_hours']} h grid"
    )
    if window.get("source") == "question":
        return f" over {span}, the window the question states"
    return f" over its default {span}; the question states no trajectory window it reads"


def _trajectory_population_clause(population: Mapping[str, Any]) -> str:
    """Which stays the design the Host would compile clusters."""

    if population.get("source") == "none":
        # The compile takes a population only from the plan's cohort, so a
        # question's population that the cohort does not state is not applied.
        return ", in every input row: the plan's cohort states no population predicate"
    if population.get("source") != "plan" or population.get("executable") is not True:
        return ""
    counts = [
        f"{len(population[kind])} {kind} predicate"
        + ("" if len(population[kind]) == 1 else "s")
        for kind in ("inclusion", "exclusion")
        if population.get(kind)
    ]
    clause = (
        ", in the population the plan states ("
        + " and ".join(counts)
        + ", each settled by the end of the trajectory window)"
    )
    within = list(population.get("within_trajectory_window") or ())
    if within:
        clause += (
            ". Membership is decided inside the trajectory window by "
            + "; ".join(within)
            + ", so it depends on the hours the classes describe"
        )
    return clause


def _sealed_trajectory_population_findings(
    facts: Mapping[str, Any],
) -> list[PlanScientificFinding]:
    """Say so when a sealed trajectory design clusters every input row.

    The plan that hands a trajectory question to the signed suite says which
    stays the design would cluster; the plan on that suite is reviewed again,
    and is the one a researcher approves, so it says the same.
    """

    # Only the signed suite replaces the plan's cohort with its sealed design's.
    if facts["longitudinal_owner"] != "signed_fixed_window_suite":
        return []
    population = facts.get("trajectory_population") or {}
    if population.get("source") != "none":
        return []
    return [
        PlanScientificFinding(
            code="TRAJECTORY_DESIGN_STATES_NO_POPULATION",
            severity="minor",
            dimension="icu_clinical_design",
            message=(
                "The sealed trajectory design clusters every input row of the "
                "source cohort: it states no population predicate, so a "
                "population that the question names is not applied to these "
                "classes."
            ),
            evidence_refs=[
                "analysis_plan.json.cohort",
                "research_context.json.research_question",
            ],
            remediation=(
                "When the question names a population, compile the study's "
                "trajectory design again from a plan whose cohort states that "
                "population as predicates; a plan on this design cannot add it."
            ),
            # The study's sealed design owns the population, not this plan.
            remediation_route="study_authority_change",
        )
    ]


def trajectory_representation_findings(
    facts: Optional[Mapping[str, Any]],
) -> list[PlanScientificFinding]:
    """Say when claimed trajectory classes are built from one value per stay.

    The Host can seal the signed owner over the plan's own coordinates and
    the window the question states, so that case blocks until it does.  When
    the primary's time-varying coordinates alone would be such a design and
    only the outcomes or one-per-stay variables it also clusters on stand in
    the way, one Planner revision reaches the owner, and an outcome in the
    clustering is leakage: that case blocks until the plan is revised.
    Otherwise the finding does not push the plan toward other coordinates or
    another window: a question about variables, or a window, the owner
    cannot model keeps them, with the limitation stated.
    """

    if facts is None:
        return []
    if facts["longitudinal_owner"] is not None:
        return _sealed_trajectory_population_findings(facts)
    coordinates = list(facts["proposed_coordinates"])
    prefix = str(facts["eligibility_coordinate_prefix"])
    window = facts.get("trajectory_window") or {}
    window_executable = window.get("executable", True) is not False
    population = facts.get("trajectory_population") or {}
    population_executable = population.get("executable", True) is not False
    refs = ["analysis_plan.json.steps", "research_context.json.variables"]
    if window.get("source") == "question":
        refs.append("research_context.json.research_question")
    if population.get("source") == "plan":
        refs.append("analysis_plan.json.cohort")
    if facts["executable"]:
        return [
            PlanScientificFinding(
                code="TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The trajectory plan clusters one value per ICU stay of "
                    + ", ".join(coordinates)
                    + "; no step builds a per-timepoint representation, so its "
                    "classes would not describe trajectories. The signed "
                    "fixed-window trajectory owner can model these coordinates"
                    + _trajectory_window_clause(window)
                    + _trajectory_population_clause(population)
                    + "."
                ),
                evidence_refs=refs,
                remediation=(
                    "Compile these coordinates"
                    + (
                        ", this window and this population"
                        if population.get("source") == "plan"
                        else " and this window"
                    )
                    + " into the study's fixed-window trajectory design and replan "
                    "on the signed trajectory suite. Keep the question, the "
                    "coordinates"
                    + (
                        ", the window and the population"
                        if population.get("source") == "plan"
                        else " and the window"
                    )
                    + "; the researcher does not choose the method."
                ),
                remediation_route="runtime_capability",
            )
        ]
    reasons = []
    if facts["outcome_inputs"]:
        reasons.append(
            "it clusters on outcomes (" + ", ".join(facts["outcome_inputs"]) + ")"
        )
    if facts["one_per_stay_inputs"]:
        reasons.append(
            "it clusters on one-per-stay variables ("
            + ", ".join(facts["one_per_stay_inputs"])
            + ")"
        )
    if len(coordinates) < 2:
        reasons.append("it names fewer than two time-varying coordinates")
    if not any(name.startswith(prefix) for name in coordinates):
        available = list(facts["study_eligibility_coordinates"])
        reasons.append(
            "none of its coordinates is a SOFA-2 component ("
            + (
                "this study provides " + ", ".join(available)
                if available
                else "this study's variables include none"
            )
            + ")"
        )
    if not window_executable:
        reasons.append(str(window.get("reason") or "its window is not a fixed-window design"))
    if not population_executable:
        reasons.append(
            str(population.get("reason") or "its population is not one the owner applies")
        )
    if not reasons:
        reasons.append("its coordinates are not a valid trajectory design")
    # A revision of the plan's inputs reaches the owner only when the
    # question's own window is one the owner counts, and the plan's own
    # population one it applies.
    if facts.get("coordinates_executable") and window_executable and population_executable:
        return [
            PlanScientificFinding(
                code="TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The trajectory plan's primary step clusters one value per "
                    "ICU stay and the signed fixed-window owner cannot take it "
                    "over: "
                    + "; ".join(reasons)
                    + ". Its time-varying coordinates "
                    + ", ".join(coordinates)
                    + " alone are a design the owner can model."
                ),
                evidence_refs=refs,
                remediation=(
                    "Revise the primary step so its inputs are only "
                    + ", ".join(coordinates)
                    + ". The host then replaces it with the signed fixed-window "
                    "suite, which describes the requested outcomes on its frozen "
                    "classes. Keep the question and these coordinates."
                ),
                remediation_route="agent_plan_revision",
            )
        ]
    return [
        PlanScientificFinding(
            code="TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL",
            severity="major",
            dimension="statistical_design",
            message=(
                "The trajectory plan clusters one value per ICU stay, so its "
                "classes summarize stays rather than describe trajectories, and "
                "the signed fixed-window owner cannot take it over: "
                + "; ".join(reasons)
                + "."
            ),
            evidence_refs=refs,
            remediation=(
                "Where the question's own coordinates allow it, revise the "
                "primary step to cluster at least two time-varying variables of "
                "this study, including a SOFA-2 component (a variable whose name "
                f"starts with {prefix!r}) on which the signed owner counts each "
                "stay's eligible windows, and no outcome or one-per-stay "
                "variable. Do not substitute another variable or score version "
                "for the one the question names; otherwise state that the "
                "classes summarize per-stay values."
                + (
                    ""
                    if window_executable
                    else " Keep the window the question states; do not count it "
                    "from ICU admission or change its length to fit the owner."
                )
                + (
                    ""
                    if population_executable
                    else " Keep the population the plan states; do not drop a "
                    "predicate or move its window to fit the owner."
                )
            ),
            remediation_route="agent_plan_revision",
        )
    ]


def _trajectory_window_end_hours(
    trajectory_representation: Optional[Mapping[str, Any]],
) -> Optional[float]:
    """The end of a trajectory plan's executable window, its time zero."""

    window = (trajectory_representation or {}).get("trajectory_window") or {}
    if window.get("executable") and window.get("window_end_hours") is not None:
        return float(window["window_end_hours"])
    return None


def plan_time_zero_hours(
    context: ResearchContext,
    trajectory_representation: Optional[Mapping[str, Any]],
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None,
) -> Optional[float]:
    """Hours after ICU admission at which the plan's analysis starts, if typed.

    A trajectory plan's time zero is its window's end: the trajectory
    population design decides the plan's own predicates by then too.
    Otherwise the family-spec request's order (``cohort_time_zero_hours``)
    holds: the study's declared landmark, else the signed runtime authority's,
    else the end of the host-bound feature window.
    """

    trajectory_end = _trajectory_window_end_hours(trajectory_representation)
    if trajectory_end is not None:
        return trajectory_end
    signed = getattr(runtime_authority, "landmark_hours", None)
    signed_landmark = (
        float(signed) if isinstance(signed, (int, float)) and not isinstance(signed, bool) else None
    )
    return (
        primary_landmark_hours(context)
        or signed_landmark
        or host_outer_feature_window_end_hours(context)
    )


def cohort_eligibility_findings(
    context: ResearchContext,
    trajectory_representation: Optional[Mapping[str, Any]],
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None,
) -> list[PlanScientificFinding]:
    """Refuse a plan whose source export decides membership after its time zero.

    The export's concept population and typed minimum ICU stay must be decided
    by the plan's time zero (``planning.cohort_eligibility``, the rule the
    family-spec request applies before planning).  A concept-population record
    the review cannot read is refused on its own: its remedy is the export's
    record, not the window.
    """

    try:
        concept = concept_cohort_window(context)
    except ConceptCohortWindowError as exc:
        return [
            PlanScientificFinding(
                code="CONCEPT_POPULATION_RECORD_UNREADABLE",
                severity="blocker",
                dimension="icu_clinical_design",
                message=(
                    "The source export's concept-population record cannot be read "
                    f"({exc}), so the review cannot confirm when the export decided "
                    "who is in the cohort."
                ),
                evidence_refs=["research_context.json.user_preferences.data_constraints"],
                remediation=(
                    "Prepare the study's export again so the host restates its "
                    "concept-population record, then review a fresh plan."
                ),
                remediation_route="study_authority_change",
            )
        ]
    return [
        PlanScientificFinding(
            code="COHORT_ELIGIBILITY_AFTER_TIME_ZERO",
            severity="blocker",
            dimension="icu_clinical_design",
            message=(
                f"Cohort eligibility is decided after the plan's time zero: {found.message()}. "
                f"Membership decided by {found.time_zero_hours:g} h, such as a positive "
                "record within the window, is allowed; only a criterion decided later is not."
            ),
            evidence_refs=[
                "research_context.json.user_preferences.data_constraints",
                "analysis_plan.json",
            ],
            remediation=(
                "Change the study so the export decides membership by "
                f"{found.time_zero_hours:g} h after ICU admission (end its concept window or "
                f"minimum ICU stay there), or move the analysis time zero to "
                f"{found.decided_by_hours:g} h or later; then prepare the export again and "
                "review a fresh plan."
            ),
            remediation_route="study_authority_change",
        )
        for found in eligibility_after_time_zero(
            time_zero_hours=plan_time_zero_hours(
                context, trajectory_representation, runtime_authority
            ),
            minimum_icu_hours=minimum_icu_stay_hours(context),
            concept_population=concept,
        )
    ]


#: Who repairs a plan predicate the host cannot show decided by time zero.
#: The Planner restates a predicate's window over a column decided by then.  A
#: selection decided later is the study's population or time zero, which the
#: Agent may not revise; a column window nothing records is the host's to record.
_COHORT_PREDICATE_FINDINGS: dict[str, tuple[str, ScientificRemediationRoute]] = {
    "anchor": ("COHORT_PREDICATE_WINDOW_AFTER_TIME_ZERO", "agent_plan_revision"),
    "window": ("COHORT_PREDICATE_WINDOW_AFTER_TIME_ZERO", "agent_plan_revision"),
    "column_window": ("COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO", "study_authority_change"),
    "icu_stay_length": ("COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO", "study_authority_change"),
    "stay_outcome": ("COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO", "study_authority_change"),
    "stay_level": ("COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO", "study_authority_change"),
    "event_time": ("COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO", "study_authority_change"),
    "unrecorded": ("COHORT_PREDICATE_COLUMN_WINDOW_UNRECORDED", "runtime_capability"),
    "event_time_unrecorded": ("COHORT_PREDICATE_COLUMN_WINDOW_UNRECORDED", "runtime_capability"),
}


def _cohort_predicate_finding(item: PredicateAfterTimeZero, source: str) -> PlanScientificFinding:
    code, route = _COHORT_PREDICATE_FINDINGS[item.reason]
    zero = f"{item.time_zero_hours:g} h after ICU admission"
    if code == "COHORT_PREDICATE_WINDOW_AFTER_TIME_ZERO":
        message = f"A cohort predicate's window does not end by the plan's time zero: {item.message()}."
        remediation = (
            f"Restate the predicate's window from ICU admission to {zero}. The column "
            "it filters is decided by then, so the selection stays the same and the "
            "plan states what the host executes."
        )
    elif code == "COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO":
        message = f"The plan's cohort decides membership after its time zero: {item.message()}."
        later = (
            f" ({item.decided_by_hours:g} h or later)" if item.decided_by_hours is not None else ""
        )
        remediation = (
            "Change the study, not only the plan: move its time zero to when this "
            f"selection is known{later}, or select by what is known at {zero}: a "
            "column the host summarizes by then, an ICU length of stay tested only up "
            "to that hour, and no outcome or undated stay-level score."
        )
    else:
        message = f"The host cannot date a column the plan's cohort filters: {item.message()}."
        record = (
            "this column's time origin and unit, as its typed relative-time resolution,"
            if item.reason == "event_time_unrecorded"
            else "the window it summarized this column over, as the column's analysis "
            "window or the study's materialization window,"
        )
        remediation = (
            f"The host must record {record} before a plan can select on it; a plan "
            "revision cannot supply that record."
        )
    return PlanScientificFinding(
        code=code,
        severity="blocker",
        dimension="icu_clinical_design",
        message=message + source,
        evidence_refs=[
            "analysis_plan.json.cohort",
            "research_context.json.variables",
            "research_context.json.user_preferences.data_constraints",
        ],
        remediation=remediation,
        remediation_route=route,
    )


def cohort_predicate_findings(
    context: ResearchContext,
    plan: AnalysisPlan,
    trajectory_representation: Optional[Mapping[str, Any]],
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None,
) -> list[PlanScientificFinding]:
    """Refuse a plan whose own cohort predicates decide membership after its time zero.

    ``planning.cohort_eligibility`` judges each predicate as data against the
    run's own variables and the window the host records it materialized the
    column over, so a column only the run's roster knows is judged too.  The
    finding names who can repair it (``_COHORT_PREDICATE_FINDINGS``).
    """

    cohort = plan.cohort
    if cohort is None:
        return []
    found = cohort_predicates_after_time_zero(
        context,
        inclusion=[predicate.to_dict() for predicate in cohort.inclusion],
        exclusion=[predicate.to_dict() for predicate in cohort.exclusion],
        time_zero_hours=plan_time_zero_hours(context, trajectory_representation, runtime_authority),
    )
    # The trajectory population design reads the same predicates against the
    # same hour; saying where it comes from keeps the two findings consistent.
    source = (
        " A trajectory plan's time zero is the end of its trajectory window."
        if _trajectory_window_end_hours(trajectory_representation) is not None
        else ""
    )
    return [_cohort_predicate_finding(item, source) for item in found]


def build_plan_scientific_review(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    literature: Optional[LiteratureBundle] = None,
    figure_strategy: Optional[ArticleFigureStrategy] = None,
    require_reportable_capability: bool = False,
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None = None,
    model_retention: Optional[PrimaryModelRetentionEvidence] = None,
) -> PlanScientificReview:
    """Score and adjudicate the exact proposed plan before human approval."""

    findings: list[PlanScientificFinding] = []
    diagnostic_products = {
        product for source in plan.steps if source.functional_form_spec is not None
        for product in source.expected_outputs[:1]
    }
    for step in plan.steps:
        if ("table:robustness_matrix" in step.inputs
                and diagnostic_products.intersection(step.inputs)
                and any(product.startswith("figure:") for product in step.expected_outputs)):
            findings.append(PlanScientificFinding(
                code="ROBUSTNESS_DIAGNOSTIC_DISPLAY_MISMATCH",
                severity="blocker", dimension="figures",
                message="The robustness figure binds functional-form diagnostics that are not effect estimates.",
                evidence_refs=["analysis_plan.json"],
                remediation="Keep the functional-form comparison as a report diagnostic table; bind only supported robustness results to the specification-grid figure. Preserve the requested sensitivity analysis.",
                remediation_route="agent_plan_revision", requires_user_authorization=False,
            ))
        if step.method == "absolute_risk_context" and step.population_scope != "analysis_cohort":
            findings.append(PlanScientificFinding(
                code="DESCRIPTIVE_POPULATION_SCOPE_UNRESOLVED",
                severity="blocker",
                dimension="icu_clinical_design",
                message="A descriptive risk step has no executable choice of population; its prose cannot establish the denominator.",
                evidence_refs=["analysis_plan.json"],
                remediation="Declare analysis_cohort or primary_model in the planning contract and bind the matching execution owner before approval.",
                remediation_route="agent_plan_revision",
                requires_user_authorization=False,
            ))
        if step.method == "primary_population_absolute_risk_context" and step.runtime_outcome_contract is None:
            findings.append(PlanScientificFinding(
                code="PRIMARY_POPULATION_EXECUTION_OWNER_MISSING",
                severity="blocker",
                dimension="icu_clinical_design",
                message="A descriptive risk step requests the primary model population, but no typed runtime owner binds that population.",
                evidence_refs=["analysis_plan.json"],
                remediation="Bind the declared primary population through its supported execution adapter; do not fall back to the broader cohort.",
                remediation_route="runtime_capability",
                requires_user_authorization=False,
            ))
    population_requirements = context_population_requirements(context)
    population_changes = []
    population_labels = {
        "primary_model": "the primary model's eligible complete-case population",
        "analysis_cohort": "the broader analysis cohort",
    }
    if population_requirements is not None:
        for required in population_requirements.populations:
            matches = [step for step in plan.steps if required.output_product in step.expected_outputs]
            if len(matches) == 1 and matches[0].population_scope == required.population_scope:
                continue
            step = matches[0] if len(matches) == 1 else None
            reason = step.population_scope_change_reason if step is not None else None
            declared = bool(step is not None and step.population_scope is not None and reason)
            population_changes.append({
                "product": required.output_product, "previous_scope": required.population_scope,
                "proposed_scope": step.population_scope if step is not None else None,
                "reason": reason, "explicit_amendment": declared,
            })
            findings.append(PlanScientificFinding(
                code="POPULATION_SCOPE_AMENDMENT_DECLARED" if declared else "PLAN_POPULATION_REQUIREMENT_DRIFT",
                severity="major" if declared else "blocker", dimension="icu_clinical_design",
                message=(f"The descriptive result changes its population from {population_labels[required.population_scope]} "
                         f"to {population_labels.get(step.population_scope if step else None, 'a missing or ambiguous population')}. "
                         f"Declared amendment: {reason or 'none'}."),
                evidence_refs=["research_context", "analysis_plan"],
                remediation=("Review this explicit scientific scope change in the complete new plan; the reason is not execution approval."
                             if declared else "Restore the source-bound population or declare an intentional scientific amendment for complete-plan review."),
                remediation_route="study_authority_change" if declared else "agent_plan_revision",
                requires_user_authorization=declared,
            ))
    # The signed survival suite describes its own Table 1 roster by exposure.
    survival_suite = _signed_survival_suite_step(plan, runtime_authority)
    baseline_coverage = baseline_requirement_coverage(
        context, plan,
        signed_rosters=(
            [(
                survival_suite.step_id,
                {runtime_authority.exposure_status_column},
                set(runtime_authority.table_one_columns),
            )]
            if survival_suite is not None
            else []
        ),
    )
    accepted_baseline = context_baseline_requirements(context)
    for table in baseline_coverage["tables"]:
        if table["complete"]:
            continue
        unavailable = table["unavailable_coordinates"]
        findings.append(PlanScientificFinding(
            code=("ACCEPTED_BASELINE_MATERIALIZATION_MISSING" if unavailable
                  else "ACCEPTED_BASELINE_CONTENT_MISSING"),
            severity="blocker",
            dimension="content_completeness",
            message=(
                f"Accepted baseline {table['source_step_id']!r} "
                f"(grouping: {table['group_by']['required'] or 'not required'}), is not preserved. "
                f"Missing variables: {table['missing_variables']}; "
                f"unavailable coordinates: {unavailable}; "
                f"matched table step: {table['matched_step_id']!r}."
            ),
            evidence_refs=["research_context", "analysis_plan"],
            remediation=(
                "Restore the missing source-bound data coordinates before replanning; "
                "a change to the accepted requirement needs a newly reviewed scope."
                if unavailable else
                "Restore every missing variable in a typed baseline table with the "
                "accepted grouping; keep any aggregation choice explicit for review."
            ),
            remediation_route=("runtime_capability" if unavailable else "agent_plan_revision"),
        ))
    variables = {variable.name: variable for variable in context.variables}
    primary_clusters = [step for step in plan.steps if step.scientific_action_id == PHENOTYPING_PRIMARY_ACTION]
    # The signed suite freezes its classes in the stability owner, and the
    # signed contract checks how a description of them is wired.
    signed_suite = signed_trajectory_plan_claimed(plan)
    compared_outcomes: set[str] = set()
    for step in plan.steps:
        if step.scientific_action_id != COMPARISON_ACTION:
            continue
        try:
            validate_comparison_step(step, context)
            if comparison_label_source(step) == ASSIGNMENTS_PRODUCT:
                if len(primary_clusters) != 1 or comparison_cohort_input(step) != sole_typed_cohort_input(primary_clusters[0]):
                    raise ValueError("phenotype_comparison_primary_source_invalid")
            elif not signed_suite or signed_trajectory_plan_contract_errors(plan):
                raise ValueError("phenotype_comparison_trajectory_source_invalid")
            compared_outcomes.update(step.phenotype_comparison_spec.outcome_columns)
        except ValueError as exc:
            findings.append(PlanScientificFinding(
                code="PHENOTYPING_COMPARISON_CONTRACT_INVALID", severity="blocker", dimension="statistical_design",
                message=f"Step {step.step_id!r}: {exc}", evidence_refs=[f"analysis_plan.json.steps.{step.step_id}.phenotype_comparison_spec"],
                remediation=("Bind a separate descriptive comparison to the exact primary cluster cohort, frozen assignments and explicitly selected clinical/outcome summaries; "
                             "frozen trajectory classes are described only through the signed suite's stability owner."),
                remediation_route="agent_plan_revision",
            ))
    missing_cluster_outcomes = set(requested_outcomes(context)) - compared_outcomes
    if (primary_clusters or signed_suite) and missing_cluster_outcomes:
        findings.append(PlanScientificFinding(
            code="PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE", severity="blocker", dimension="statistical_design",
            message="No executable post-clustering descriptive comparison covers the requested outcomes: " + ", ".join(sorted(missing_cluster_outcomes)),
            evidence_refs=["research_context.json.cohort.requested_outcome_columns", "analysis_plan.json.steps"],
            remediation="Add a secondary phenotyping.outcome_by_cluster step with an explicit summary roster for every requested outcome. Readable inputs, feature profiles and figures alone do not execute this comparison.",
            remediation_route="agent_plan_revision",
        ))
    for step in plan.steps:
        if step.scientific_action_id != PHENOTYPING_PRIMARY_ACTION:
            continue
        try:
            require_phenotyping_features(
                step.phenotyping_feature_columns, inputs=step.inputs, descriptors=context.variables,
                outcome_columns=(*context.cohort.outcome_columns, *([context.target_outcome] if context.target_outcome else [])),
            )
        except ValueError as exc:
            findings.append(PlanScientificFinding(
                code="PHENOTYPING_FIT_ROSTER_INVALID", severity="blocker", dimension="statistical_design",
                message=f"Step {step.step_id!r}: {exc}",
                evidence_refs=[f"analysis_plan.json.steps.{step.step_id}.phenotyping_feature_columns"],
                remediation="Declare the exact non-outcome fitting roster separately from readable profile inputs, then review a fresh plan.",
                remediation_route="agent_plan_revision",
            ))
    trajectory_representation = trajectory_representation_facts(context, plan)
    findings.extend(trajectory_representation_findings(trajectory_representation))
    survival_suite = landmark_survival_suite_facts(context, plan)
    findings.extend(landmark_survival_suite_findings(survival_suite))
    findings.extend(
        cohort_eligibility_findings(context, trajectory_representation, runtime_authority)
    )
    findings.extend(
        cohort_predicate_findings(context, plan, trajectory_representation, runtime_authority)
    )
    required_source_columns = {
        context.primary_exposure, context.target_outcome,
        *context.cohort.outcome_columns,
        *AdjustmentSetAuthority.from_context(context).operational_covariates,
    }
    for issue in declared_raw_input_plan_findings(plan=plan, context=context):
        if issue.detail.get("reason") != "declared_raw_input_structurally_unavailable":
            continue
        required_source = bool(
            required_source_columns.intersection(issue.detail["unavailable_inputs"])
        )
        findings.append(PlanScientificFinding(
            code="PLAN_INPUT_STRUCTURALLY_UNAVAILABLE",
            severity="blocker", dimension="statistical_design",
            message=issue.message,
            evidence_refs=[
                f"analysis_plan.json.steps.{issue.detail['step_id']}.inputs",
                "research_context.json.variables.source_concept",
                "easyicu.outcome_availability.OUTCOME_CONCEPT_SUPPORTED_DATABASES",
            ],
            remediation=(
                "Establish source-owner support for the required scientific "
                "variable; preserve the reviewed question and do not substitute "
                "an endpoint, exposure, or required adjustment."
                if required_source else
                "Omit optional structurally unavailable inputs from a fresh "
                "plan, retaining the source limitation and original data. "
                "If the variable is needed to answer the question, report the "
                "source-owner capability gap rather than substituting a result."
            ),
            remediation_route="runtime_capability" if required_source else "agent_plan_revision",
        ))
    for step in plan.steps:
        if step.exposure_outcome_distribution_spec is None:
            continue
        for issue in distribution_policy_issues(
            step.exposure_outcome_distribution_spec, variables=variables,
        ):
            findings.append(PlanScientificFinding(
                code="DISTRIBUTION_MISSINGNESS_AUTHORITY_INVALID",
                severity="blocker",
                dimension="statistical_design",
                message=f"Step {step.step_id!r}: {issue.message}",
                evidence_refs=[
                    f"analysis_plan.json.steps.{step.step_id}.exposure_outcome_distribution_spec.{issue.field}",
                    "research_context.json.variables",
                ],
                remediation=DISTRIBUTION_MISSINGNESS_GUIDANCE,
                remediation_route="agent_plan_revision",
            ))
    literature_facts = _literature_facts(literature, context)
    method_facts = method_source_facts(plan, context)
    design_bindings = _literature_design_bindings(plan, literature)
    sensitivity = _sensitivity_facts(
        context, plan, runtime_authority=runtime_authority
    )
    publication_readiness = build_publication_readiness_facts(
        context=context,
        plan=plan,
        figure_strategy=figure_strategy,
        sensitivity=sensitivity,
    )
    robustness_readiness = publication_readiness["robustness"]
    figure_roles = publication_readiness["figure_roles"]
    content_roles = publication_readiness["content_roles"]
    linearity = _continuous_linearity_facts(plan)
    model_term_domain_conflicts = _model_term_domain_conflicts(context, plan)
    clinical_definitions = _clinical_definition_facts(context)
    time_anchor_alignment = primary_exposure_time_anchor_alignment(context)
    post_baseline, exposure_window = post_baseline_exposure(context)
    repeats = repeat_units_possible(context)
    patient_identity = patient_identity_available(context)
    covariates = model_covariates(plan)
    preferences = context.user_preferences
    adjustment_authority = AdjustmentSetAuthority.from_context(context)
    if not covariates and any(
        step.planned_analysis_role == "primary"
        and _method_head(step) in {"signed_landmark_restricted_cubic_spline", "time_varying_exposure_model"}
        for step in plan.steps
    ):
        # This method can appear only after the digest-bound runtime owner has
        # replaced the generic primary step. Its exact adjustment columns are
        # therefore the operational projection of the sealed StudyContext,
        # not an inference from plan inputs or available data. A plan-bound
        # roster lives in the sealed runtime contract instead, sealed from the
        # reviewed primary model before the owner replaced it.
        covariates = adjustment_authority.operational_covariates
        if (
            not covariates
            and isinstance(runtime_authority, LandmarkSplineRuntimeAuthority)
            and runtime_authority.plan_bound_adjustment_roster is not None
            and runtime_authority.adjustment_roster_sealed
        ):
            covariates = tuple(runtime_authority.required_adjustment_columns)
    covariate_selection = (
        preferences.covariate_selection if preferences is not None else "planner_selectable"
    )
    if covariate_selection == "exact":
        covariate_rationales = adjustment_authority.operational_rationales
        covariate_temporal_roles = adjustment_authority.operational_temporal_roles
    else:
        covariate_rationales, covariate_temporal_roles = (
            model_covariate_plan_authority(plan)
        )
    capability_assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type,
        context=context,
        plan=plan,
    )
    expected_outcomes = requested_outcomes(context)
    covered_outcomes = planned_model_outcomes(plan, context)
    if survival_suite is not None and survival_suite.get("executable"):
        # The proposed suite's event column has its owner; compiling the
        # design seals the suite that produces it.
        covered_outcomes = (*covered_outcomes, str(survival_suite["event_column"]))
    missing_model_outcomes = tuple(
        outcome for outcome in expected_outcomes if outcome not in covered_outcomes
    )
    occurrence_cues = requested_exposure_occurrence(context, plan)
    occurrence_step_ids = exposure_occurrence_steps(plan, context)
    selected_design = (
        plan.design_selection.selected if plan.design_selection is not None else None
    )
    if selected_design is not None and selected_design.reviewable_plan is None:
        findings.append(
            PlanScientificFinding(
                code="REVIEWABLE_PLAN_SPECIFICATION_MISSING",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The selected design does not contain a complete Planner-owned "
                    "recommendation for researcher review."
                ),
                evidence_refs=["analysis_plan.json.design_selection"],
                remediation=(
                    "Generate a fresh candidate plan that recommends the cohort "
                    "and analysis unit, exposure timing and aggregation, outcome "
                    "follow-up, adjustment/model, missing-data strategy, and "
                    "sensitivity plus feasibility checks before requesting approval."
                ),
            )
        )
    if (
        require_reportable_capability
        and not capability_assessment.claim_ceiling_allows_reportable
    ):
        findings.append(
            PlanScientificFinding(
                code="SCIENTIFIC_CAPABILITY_NOT_REPORTABLE",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "This formal run requires a reportable scientific capability, "
                    f"but {capability_assessment.capability_id or plan.analysis_type!r} "
                    f"has claim ceiling {capability_assessment.claim_ceiling!r}."
                ),
                evidence_refs=["analysis_plan.json"],
                remediation=(
                    "Revise the plan to use a registered typed capability with a "
                    "deterministic scientific validator, or start a separately "
                    "labelled diagnostic run that permits analysis-only output."
                ),
            )
        )

    if not literature_facts["search_conducted"] or not literature_facts["sources_returning"]:
        findings.append(
            PlanScientificFinding(
                code="TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED",
                severity="major",
                dimension="literature",
                message="No dated retrieval source returned current literature for this plan.",
                evidence_refs=["preplan_literature_bundle.json"],
                remediation="Run and retain a reproducible database search before claiming article-level prior-art coverage.",
            )
        )
    elif not any(literature_facts["queries"].values()):
        findings.append(
            PlanScientificFinding(
                code="LITERATURE_SEARCH_QUERY_NOT_RECORDED",
                severity="major",
                dimension="literature",
                message="The retrieval receipt omits its exact source queries.",
                evidence_refs=["preplan_literature_bundle.json.search_provenance"],
                remediation="Persist normalized source queries and record-to-query bindings.",
            )
        )
    direct_required = bool(context.primary_exposure)
    if not literature_facts["comparison_source_keys"]:
        findings.append(
            PlanScientificFinding(
                code=(
                    "DIRECT_COMPARATOR_NOT_ESTABLISHED"
                    if direct_required
                    else "DESIGN_ANALOGUE_NOT_ESTABLISHED"
                ),
                severity="major",
                dimension="literature",
                message=(
                    "No retrieved study passed all population, exposure, outcome, "
                    "and design-excerpt checks as a direct comparator."
                    if direct_required
                    else (
                        "No retrieved study passed the ICU population, clinical "
                        "topic, analysis-intent, and design-excerpt checks as a "
                        "design analogue."
                    )
                ),
                evidence_refs=["preplan_literature_bundle.json.screening_decisions"],
                remediation=(
                    "Run a direct observational-comparator search stratum and "
                    "record source-backed inclusion/exclusion decisions."
                    if direct_required
                    else (
                        "Run a design-analogue search and retain source-backed "
                        "topic/design inclusion decisions without inventing a P/E/O "
                        "contrast."
                    )
                ),
            )
        )
    newest = literature_facts["newest_comparison_source_year"]
    if newest is not None and literature_facts["search_year"] - newest > 5:
        findings.append(
            PlanScientificFinding(
                code=(
                    "RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED"
                    if direct_required
                    else "RECENT_DESIGN_ANALOGUE_NOT_ESTABLISHED"
                ),
                severity="major",
                dimension="literature",
                message=(
                    "The newest screened comparison source is more than five years "
                    "older than the search year."
                ),
                evidence_refs=["preplan_literature_bundle.json.citations"],
                remediation="Document whether current similar work is truly absent or retrieval/screening missed it.",
            )
        )
    primary_keys = {
        key
        for step in scientific_steps(plan)
        if step.planned_analysis_role == "primary"
        for key in step.literature_citation_keys
    }
    if literature_facts["comparison_source_keys"] and set(
        literature_facts["comparison_source_keys"]
    ).isdisjoint(primary_keys):
        findings.append(
            PlanScientificFinding(
                code=(
                    "DIRECT_COMPARATOR_NOT_BOUND_TO_PRIMARY_PLAN"
                    if direct_required
                    else "DESIGN_ANALOGUE_NOT_BOUND_TO_PRIMARY_PLAN"
                ),
                severity="major",
                dimension="literature_to_plan",
                message=(
                    "A screened comparison source exists but does not govern any "
                    "primary analysis step."
                ),
                evidence_refs=["analysis_plan.json", "preplan_literature_bundle.json"],
                remediation=(
                    "Bind the exact comparator or design-analogue key to the primary "
                    "step and record what design element was borrowed or deliberately "
                    "differed."
                ),
            )
        )
    if method_facts["method_source_gaps"]:
        findings.append(
            PlanScientificFinding(
                code="SCIENTIFIC_STEP_METHOD_SOURCE_NOT_BOUND",
                severity="major",
                dimension="literature_to_plan",
                message="Scientific steps cite no source that governs their method: " + ", ".join(method_facts["method_source_gaps"]),
                evidence_refs=["analysis_plan.json", "method_literature_pack"],
                remediation="Bind each scientific step to an applicable method card, not only a disease definition or database paper.",
            )
        )
    if method_facts["unsupported_method_bindings"]:
        finding_rows = [
            f"{item['step_id']}:{item['citation_key']}="
            + ",".join(item["unsupported_design_elements"])
            for item in method_facts["unsupported_method_bindings"]
        ]
        findings.append(
            PlanScientificFinding(
                code="METHOD_SOURCE_DESIGN_ELEMENT_UNSUPPORTED",
                severity="major",
                dimension="literature_to_plan",
                message=(
                    "Method citations are bound to design elements that their "
                    "curated decision cards do not support: "
                    + "; ".join(finding_rows)
                ),
                evidence_refs=["analysis_plan.json", "method_literature_pack"],
                remediation=(
                    "Bind the exact method card through a supported design "
                    "element, or cite a different sealed source; do not credit "
                    "all decisions merely because the paper appears in the step."
                ),
            )
        )
    if method_facts["missing_method_layers"]:
        findings.append(
            PlanScientificFinding(
                code="APPLICABLE_METHOD_LAYERS_NOT_BOUND",
                severity="major",
                dimension="literature_to_plan",
                message="No plan citation covers applicable method layers: " + ", ".join(method_facts["missing_method_layers"]),
                evidence_refs=["analysis_plan.json", "method_literature_pack"],
                remediation="Bind timing, dependence, missing-data, functional-form, interpretation, and reporting sources where applicable.",
            )
        )
    if design_bindings["unresolved_steps"]:
        findings.append(
            PlanScientificFinding(
                code="LITERATURE_DESIGN_ROUTE_NOT_EXPLICIT",
                severity="major",
                dimension="literature_to_plan",
                message=(
                    "Citations are attached, but source-backed design elements "
                    "are not explicitly reflected in scientific steps: "
                    + ", ".join(design_bindings["unresolved_steps"])
                ),
                evidence_refs=["analysis_plan.json", "preplan_literature_bundle.json"],
                remediation=(
                    "Record which population, timing, estimand, adjustment, "
                    "missing-data, robustness, or reporting element every exact "
                    "cited source informs; citation presence alone is insufficient."
                ),
            )
        )
    # A proposed survival suite that closes binds the requested time-to-event
    # endpoint from the fixed-horizon vocabulary (event, paired follow-up,
    # origin, censoring) when its design is compiled; that is not a question
    # for the researcher.  A recorded endpoint-definition conflict is: the
    # question asked for another endpoint (survival to day 28 of a 90-day
    # endpoint), and compiling the suite would not answer it.
    survival_endpoint_proposed = bool(
        survival_suite is not None
        and survival_suite.get("executable")
        and survival_suite.get("event_column") == str(context.target_outcome or "").strip()
        and not _endpoint_conflict_recorded(context)
    )
    if (
        not _endpoint_resolved(context)
        and study_endpoint_required(context, plan)
        and not survival_endpoint_proposed
        and not context_declares_source_feasibility_scope(context)
    ):
        # A fail-closed feasibility scope analyses no outcome: the reviewed
        # protocol declared the contrast non-identifiable before any endpoint.
        findings.append(
            PlanScientificFinding(
                code="OUTCOME_DEFINITION_UNRESOLVED",
                severity="blocker",
                dimension="icu_clinical_design",
                message="The primary outcome lacks a complete owner-issued endpoint definition.",
                evidence_refs=["research_context.json"],
                remediation="Bind the physical outcome to its clinical meaning and horizon before execution.",
                requires_user_authorization=True,
                authorization_question="Please confirm the intended clinical endpoint and time horizon in a new study version.",
            )
        )
    endpoint_result_required = plan_reports_endpoint_result(plan)
    if endpoint_result_required and missing_model_outcomes:
        findings.append(
            PlanScientificFinding(
                code="REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The plan does not provide an outcome-appropriate executable "
                    "contract for every outcome identified from the research "
                    "question: "
                    + ", ".join(missing_model_outcomes)
                    + "."
                ),
                evidence_refs=[
                    "research_context.json.cohort.requested_outcome_columns",
                    "analysis_plan.json.steps",
                ],
                remediation=(
                    "Add an executable analysis contract for every missing typed "
                    "outcome, preserving the reviewed analysis family. Descriptive "
                    "questions need typed summaries, not an added regression or "
                    "uncertainty; model questions need their declared model result. "
                    "Readable inputs, baseline tables and figure labels alone do "
                    "not answer an endpoint. Do not reduce a multi-outcome question "
                    "to its primary endpoint."
                ),
                remediation_route="agent_plan_revision",
            )
        )
    if occurrence_cues and not occurrence_step_ids:
        findings.append(
            PlanScientificFinding(
                code="REQUESTED_OCCURRENCE_COVERAGE_INCOMPLETE",
                severity="blocker",
                dimension="content_completeness",
                message=(
                    "The research question asks how often the primary exposure "
                    f"occurs ({', '.join(occurrence_cues)}), but no step reports the "
                    "exposure's level distribution in the study cohort."
                ),
                evidence_refs=[
                    "research_context.json.research_question",
                    "analysis_plan.json.steps",
                ],
                remediation=(
                    "Add a step that reports each exposure level's count and "
                    "proportion, with denominators, among the stays the study "
                    "selected. When the primary analysis runs on a narrower "
                    "cohort (a landmark cohort, for example), that step reads the "
                    "study cohort itself; counts on the narrower cohort answer a "
                    "different question."
                ),
                remediation_route="agent_plan_revision",
            )
        )
    dose_response_cues, two_levels = requested_dose_response_on_two_levels(context)
    if dose_response_cues:
        findings.append(
            PlanScientificFinding(
                code="REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The research question asks for a dose-response relationship "
                    f"({', '.join(dose_response_cues)}), but the primary exposure "
                    f"has two levels ({', '.join(two_levels)}); a gradient needs at "
                    "least three ordered levels or a continuous exposure."
                ),
                evidence_refs=[
                    "research_context.json.research_question",
                    "research_context.json.primary_exposure",
                ],
                remediation=(
                    "A two-level contrast answers another question, so it does not "
                    "stand in for the requested gradient. A new study version can "
                    "name a graded or continuous exposure, or ask for the two-level "
                    "contrast itself."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "The question asks for a dose-response gradient, but its exposure "
                    "has two levels. Should a new study version use a graded or "
                    "continuous exposure, or ask for the two-level contrast instead?"
                ),
            )
        )
    if clinical_definitions["independent_clinical_review_pending_contracts"]:
        findings.append(
            PlanScientificFinding(
                code="CLINICAL_DEFINITION_INDEPENDENT_REVIEW_PENDING",
                severity="major",
                dimension="icu_clinical_design",
                message=(
                    "Automated conformance is available, but independent ICU-"
                    "clinician review remains pending for clinical definition "
                    "contracts: "
                    + ", ".join(
                        clinical_definitions[
                            "independent_clinical_review_pending_contracts"
                        ]
                    )
                    + "."
                ),
                evidence_refs=[
                    "research_context.json.variables.clinical_definition"
                ],
                remediation=(
                    "Obtain and bind an independent clinical review of the exact "
                    "definition/version/digest. Do not relabel automated golden-"
                    "vector validation as clinician sign-off."
                ),
            )
        )
    if clinical_definitions["database_conformance_gaps"]:
        gap_labels = [
            (
                f"{item['variable']}:{item['contract_id']}@{item['database']}="
                f"{item['conformance']}"
            )
            for item in clinical_definitions["database_conformance_gaps"]
        ]
        findings.append(
            PlanScientificFinding(
                code="CLINICAL_DEFINITION_DATABASE_CONFORMANCE_NOT_ESTABLISHED",
                severity="major",
                dimension="icu_clinical_design",
                message=(
                    "The owner registry has not established algorithm-level "
                    "clinical-definition conformance in the analysis database: "
                    + ", ".join(gap_labels)
                    + ". A mapping-only receipt is not phenotype validation."
                ),
                evidence_refs=[
                    "research_context.json.variables.clinical_definition.database_conformance"
                ],
                remediation=(
                    "Independently review the exact database implementation and "
                    "bind algorithm-level conformance evidence, or preserve this "
                    "limitation and withhold top-journal readiness."
                ),
            )
        )
    if time_anchor_alignment.status in {"mismatch", "declared_only"}:
        mismatch = time_anchor_alignment.status == "mismatch"
        findings.append(
            PlanScientificFinding(
                code=(
                    "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH"
                    if mismatch
                    else "PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED"
                ),
                severity="blocker",
                dimension="icu_clinical_design",
                message=(
                    "The primary exposure's owner-issued clinical-definition "
                    "anchor does not match the user-declared clinical time zero."
                    if mismatch
                    else (
                        "The user declared a clinical time zero, but the primary "
                        "exposure carries no verifiable clinical-definition "
                        "anchor."
                    )
                ),
                evidence_refs=[
                    "research_context.json.user_preferences.timing_and_design",
                    "research_context.json.variables.clinical_definition",
                    "research_context.json.variables.analysis_window",
                    "research_context.json.variables.analysis_window_role",
                ],
                remediation=(
                    "Create a new StudyContext/concept-authority revision whose "
                    "typed clinical-definition anchor matches the declared study "
                    "anchor; the Planner may not infer it from, or relabel, the "
                    "outer observation window."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Should the study adopt the owner-issued clinical-definition "
                    "anchor, or should a new concept/study version be issued for "
                    "the intended clinical anchor?"
                ),
            )
        )
    if study_time_origin_alignment(context).status == "mismatch":
        findings.append(
            PlanScientificFinding(
                code="STUDY_TIME_ZERO_MISMATCH",
                severity="blocker",
                dimension="icu_clinical_design",
                message=(
                    "The study's declared time zero is not the event its "
                    "materialized windows count from, and the study has no "
                    "primary exposure whose definition could carry it."
                ),
                evidence_refs=[
                    "research_context.json.research_question",
                    "research_context.json.temporal_constraints",
                    "research_context.json.user_preferences.timing_and_design",
                    "research_context.json.variables.analysis_window",
                ],
                remediation=(
                    "Create a new StudyContext revision whose time zero is the "
                    "windows' origin, or a materialization whose windows count "
                    "from the declared time zero; the Planner may not relabel a "
                    "window's origin."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Should the study take the windows' origin as its time zero, "
                    "or be revised so its windows count from the declared time "
                    "zero?"
                ),
            )
        )
    needs_temporal_inference = temporal_inference_required(plan)
    unsupported_timing_spec_ids = [
        spec.spec_id
        for spec in _sensitivity_specs(context)
        if spec.axis == "timing"
        and (not EXECUTABLE_METHODS_BY_STRATEGY[spec.strategy]
             or (spec.strategy == "time_varying" and spec.time_varying_execution is None))
    ]
    if post_baseline and needs_temporal_inference and not timing_design_closed(plan):
        if unsupported_timing_spec_ids:
            findings.append(
                PlanScientificFinding(
                    code="TIME_VARYING_RUNTIME_UNAVAILABLE",
                    severity="blocker",
                    dimension="icu_clinical_design",
                    message=(
                        "The requested time-varying sensitivity has no registered "
                        "deterministic runtime: "
                        + ", ".join(sorted(unsupported_timing_spec_ids))
                        + "."
                    ),
                    evidence_refs=[
                        "research_context.json.user_preferences",
                        "analysis_plan.json",
                    ],
                    remediation=(
                        "Build and verify a source-bound longitudinal outcome "
                        "follow-up contract and deterministic time-varying runtime. "
                        "Preserve the current StudyContext; do not rerun Planner or "
                        "ask the researcher to choose the same design again."
                    ),
                )
            )
        else:
            selected_temporal_design_declared = selected_design is not None
            findings.append(
                PlanScientificFinding(
                    code="POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
                    severity="blocker",
                    dimension="icu_clinical_design",
                    message="Exposure is classified after ICU time zero, but no executable temporal estimator closes exposure opportunity and early events.",
                    evidence_refs=["research_context.json", "analysis_plan.json"],
                    remediation=(
                        "Compile the Agent-selected temporal design into an "
                        "executable typed landmark or time-varying runtime without "
                        "asking the researcher to choose the method again."
                        if selected_temporal_design_declared
                        else (
                            "Revise the candidate plan so it selects an executable "
                            "typed landmark or time-varying estimator. Method "
                            "selection belongs to the Agent plan and remains "
                            "subject to the later whole-plan review."
                        )
                    ),
                    remediation_route=(
                        "runtime_capability"
                        if selected_temporal_design_declared
                        else "agent_plan_revision"
                    ),
                )
            )
    if descriptive_counts_only_required(context, analysis_type=plan.analysis_type):
        unresolved_intervals = [
            step.step_id for step in plan.steps
            if step.exposure_outcome_distribution_spec is not None
            and step.exposure_outcome_distribution_spec.schema_version
            != "easyicu.exposure_outcome_distribution/3"
        ]
        if unresolved_intervals:
            findings.append(PlanScientificFinding(
                code="DESCRIPTIVE_INTERVAL_DEPENDENCE_UNRESOLVED",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "Descriptive intervals lack independent-unit or patient-grouping "
                    "authority: " + ", ".join(unresolved_intervals)
                ),
                evidence_refs=["research_context.json.cohort.provenance", "analysis_plan.json"],
                remediation=(
                    "Regenerate the descriptive plan through the shared source-bound "
                    "counts-only compiler. Retain all stays and report counts and "
                    "proportions; do not promise intervals or inferential contrasts. "
                    "Patient-level inference requires verified grouping authority."
                ),
                remediation_route="agent_plan_revision",
            ))
    if repeats and not patient_identity and not repeated_unit_design_closed(context, plan):
        findings.append(
            PlanScientificFinding(
                code="REPEATED_STAY_IDENTITY_UNAVAILABLE",
                severity="major",
                dimension="icu_clinical_design",
                message=(
                    "The stay-level source does not expose patient identity, so "
                    "repeated ICU stays cannot be ruled out. Development analysis "
                    "may continue with this explicit limitation, but patient-level "
                    "independence and paper authority remain unavailable."
                ),
                evidence_refs=["research_context.json.cohort.provenance"],
                remediation=(
                    "Have EasyICU materialize a verified patient-grouping coordinate "
                    "when the source can provide one. Until then, retain all stays, "
                    "state the dependence limitation, and keep paper authority off; "
                    "an ICU-readmission flag must not be mislabeled as patient identity."
                ),
                remediation_route="runtime_capability",
            )
        )
    elif repeats and not repeated_unit_design_closed(context, plan):
        if repeated_unit_estimator_present(context, plan):
            findings.append(
                PlanScientificFinding(
                    code="REPEATED_STAY_METHOD_NOT_DECLARED",
                    severity="blocker",
                    dimension="icu_clinical_design",
                    message="Patient identity exists, but no executable estimator addresses repeated ICU stays.",
                    evidence_refs=["research_context.json", "analysis_plan.json"],
                    remediation=(
                        "Revise the Agent plan so it selects and binds one executable "
                        "one-stay, clustered, or mixed estimator. The researcher reviews "
                        "the resulting plan as a whole and is not asked to choose the "
                        "statistical implementation."
                    ),
                    remediation_route="agent_plan_revision",
                )
            )
        else:
            # A revision of these steps cannot bind a dependence contract,
            # so another Planner turn would only repeat this finding.
            findings.append(
                PlanScientificFinding(
                    code="REPEATED_STAY_METHOD_NOT_DECLARED",
                    severity="blocker",
                    dimension="icu_clinical_design",
                    message=(
                        "Patient identity exists, but every analysis step fits one row per "
                        "ICU stay and none can carry patient-level dependence."
                    ),
                    evidence_refs=["research_context.json", "analysis_plan.json"],
                    remediation=(
                        "Keep each patient's first ICU stay, identified by the host from "
                        "the bound stay table, and plan again on that population."
                    ),
                    remediation_route="runtime_capability",
                )
            )
    elif repeats and not _repeated_stay_rule_declared(
        plan, context, sensitivity.get("executable", ())
    ):
        findings.append(
            PlanScientificFinding(
                code="REPEATED_STAY_DEDUP_UNDECLARED",
                severity="major",
                dimension="icu_clinical_design",
                message=(
                    "Repeated stays are possible and the design closes them, but "
                    "no step declares the repeat-stay rule: first-stay "
                    "restriction or confirmed clustered/mixed handling with a "
                    "reported repeat structure. Silence here is how repeated "
                    "stays leak into independence-assuming estimates."
                ),
                evidence_refs=["research_context.json", "analysis_plan.json"],
                remediation=(
                    "Declare the rule explicitly: restrict to one stay per "
                    "patient with an audit step, or confirm the clustered/mixed "
                    "handling and report the repeat-stay structure (patients "
                    "with >1 stay, distribution of stay counts)."
                ),
                remediation_route="agent_plan_revision",
            )
        )
    model_requirement_sets = [
        requirement.analysis_set
        for step in plan.steps
        for requirement in (step.model_requirements or ())
    ]
    if (
        model_requirement_sets
        and all(value == "complete_case" for value in model_requirement_sets)
        and "missing" not in set(sensitivity.get("executable", ()))
        and not any(
            getattr(spec, "missing_override", None)
            for spec in plan.robustness_specs
        )
    ):
        findings.append(
            PlanScientificFinding(
                code="MISSINGNESS_UNEXAMINED_COMPLETE_CASE",
                severity="major",
                dimension="statistical_design",
                message=(
                    "Every model requirement runs complete-case with no "
                    "missing-data sensitivity axis and no missing override: "
                    "the plan never examines what complete-case deletion or "
                    "any single-value handling assumes away."
                ),
                evidence_refs=[
                    "analysis_plan.json.model_requirements",
                    "analysis_plan.json.robustness_specs",
                ],
                remediation=(
                    "Add a missing-data sensitivity axis (multiple imputation "
                    "or a declared complete-case examination) or a locked "
                    "missing override with variables; single-value handling "
                    "in generated code must additionally pass the "
                    "single-imputation preflight rule."
                ),
                remediation_route="agent_plan_revision",
            )
        )
    repeated_unit_table_one_tests = [
        step.step_id
        for step in plan.steps
        if step.table_one_spec is not None
        and step.table_one_spec.p_values_required
    ]
    if repeats and repeated_unit_table_one_tests:
        findings.append(
            PlanScientificFinding(
                code="TABLE_ONE_INDEPENDENT_TESTS_IGNORE_REPEATED_UNITS",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "Table 1 requests independent-row tests although the cohort "
                    "retains repeated units: "
                    + ", ".join(repeated_unit_table_one_tests)
                    + "."
                ),
                evidence_refs=[
                    "research_context.json.cohort",
                    "analysis_plan.json.steps.table_one_spec",
                ],
                remediation=(
                    "Use the host-bound descriptive/SMD-only Table 1 projection, "
                    "or issue a new typed clustered-test contract; do not relabel "
                    "Mann-Whitney, Welch, chi-square, or Fisher tests as clustered."
                ),
                remediation_route="agent_plan_revision",
            )
        )
    if sensitivity["missing_required"] or sensitivity["missing_spec_ids"]:
        findings.append(
            PlanScientificFinding(
                code="REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY",
                severity="blocker",
                dimension="robustness",
                message=(
                    "User-required sensitivity analyses are absent or protocol-only: "
                    + ", ".join(
                        sensitivity["missing_spec_ids"]
                        or sensitivity["missing_required"]
                    )
                ),
                evidence_refs=["research_context.json.user_preferences", "analysis_plan.json"],
                remediation="Add executable, evidence-producing re-estimation steps or explicitly revise the requested outputs in a new user-authorized version.",
                requires_user_authorization=True,
                authorization_question="Do you want a new study version that executes these sensitivity analyses, or should the requested outputs be reduced?",
            )
        )
    if association_study(plan) and not covariates:
        planner_owned = covariate_selection != "exact"
        findings.append(
            PlanScientificFinding(
                code="UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE",
                severity="major",
                dimension="statistical_design",
                message="The primary association is unadjusted and therefore supports descriptive, not independent-association, interpretation.",
                evidence_refs=["analysis_plan.json"],
                remediation=(
                    "Generate a clinically justified pre-time-zero adjustment "
                    "proposal, or retain an explicitly descriptive claim ceiling."
                    if planner_owned
                    else "Retain the user's exact unadjusted choice and its descriptive claim ceiling, or create a new user-authorized study version."
                ),
                remediation_route=(
                    "agent_plan_revision" if planner_owned else "study_authority_change"
                ),
                requires_user_authorization=not planner_owned,
                authorization_question=(
                    None
                    if planner_owned
                    else "Keep the analysis descriptive, or authorize a new clinically timed adjustment strategy?"
                ),
            )
        )
    if covariates and covariate_selection != "exact" and (
        set(covariate_rationales) != set(covariates)
        or set(covariate_temporal_roles) != set(covariates)
    ):
        findings.append(
            PlanScientificFinding(
                code="PLANNER_ADJUSTMENT_PROPOSAL_INCOMPLETE",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "The Agent proposed an adjustment roster without supplying a "
                    "complete confounding rationale and pre-time-zero role for "
                    "every covariate."
                ),
                evidence_refs=["analysis_plan.json.model_requirements"],
                remediation=(
                    "Revise the candidate plan so the Agent justifies every "
                    "selected covariate and proves its baseline timing; do not ask "
                    "the researcher to fill these internal planning fields."
                ),
                remediation_route="agent_plan_revision",
            )
        )
    elif covariates and covariate_selection == "exact" and (
        set(covariate_rationales) != set(covariates)
        or set(covariate_temporal_roles) != set(covariates)
    ):
        findings.append(
            PlanScientificFinding(
                code="ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND",
                severity="major",
                dimension="statistical_design",
                message=(
                    "The exact adjustment roster lacks a complete user-reviewed "
                    "clinical rationale or pre-time-zero temporal role."
                ),
                evidence_refs=["research_context.json.user_preferences"],
                remediation=(
                    "Record one confounding rationale and one baseline temporal "
                    "role for every exact covariate in a new StudyContext revision."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Do you approve the clinical rationale and baseline timing for "
                    "every exact adjustment covariate in a new study version?"
                ),
            )
        )
    if model_term_domain_conflicts:
        conflict_variables = sorted(
            {str(item["variable"]) for item in model_term_domain_conflicts}
        )
        findings.append(
            PlanScientificFinding(
                code="MODEL_TERM_CODING_CONFLICTS_WITH_DECLARED_DOMAIN",
                severity="blocker",
                dimension="statistical_design",
                message=(
                    "Continuous model coding conflicts with an owner-declared "
                    "closed variable domain: " + ", ".join(conflict_variables)
                ),
                evidence_refs=[
                    "research_context.json.variables",
                    "analysis_plan.json.model_requirements",
                ],
                remediation=(
                    "Regenerate the plan using categorical, binary, or ordinal "
                    "coding that matches the concept owner's declared domain; do "
                    "not reinterpret factor codes as interval measurements."
                ),
            )
        )
    if linearity["linear_identity_terms"] and not linearity["functional_form_sensitivity_executable"]:
        findings.append(
            PlanScientificFinding(
                code="CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED",
                severity="major",
                dimension="statistical_design",
                message="Continuous covariates enter linearly without an executable functional-form check: " + ", ".join(linearity["unchecked_linear_identity_terms"]),
                evidence_refs=["analysis_plan.json.model_requirements"],
                remediation="Add a prespecified spline/nonlinearity sensitivity with source binding, without changing the headline estimand after results are seen.",
            )
        )
    if (
        robustness_readiness["status"] == "blocked"
        and robustness_readiness["reason"] == "no_typed_sensitivity_authority"
    ):
        planner_can_repair_robustness = robustness_readiness["planner_revision_supported"]
        findings.append(
            PlanScientificFinding(
                code="ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
                severity="major",
                dimension="robustness",
                message=(
                    "The study-family playbook calls for robustness evidence, "
                    "but no typed, executable sensitivity authority was "
                    "prespecified for this study version."
                ),
                evidence_refs=[
                    "study_design_brief.json.sensitivity_requirements",
                    "research_context.json.user_preferences.sensitivity_specs",
                    "analysis_plan.json.robustness_specs",
                ],
                remediation=(
                    "Have the Agent plan prespecify task-supported executable "
                    "denominator, missingness/measurement, outcome-definition, "
                    "timing, or model sensitivities. The researcher reviews the "
                    "complete plan rather than selecting internal sensitivity "
                    "implementations. Descriptive studies must not invent an "
                    "effect-estimate replay grid."
                    if planner_can_repair_robustness
                    else (
                        "The selected family does not expose a sensitivity replay "
                        "or custom-analysis owner. Implement a typed family-appropriate "
                        "sensitivity capability before requesting plan revision. "
                        "Denominator and measurement audits remain required but do "
                        "not prove sensitivity robustness. Preserve this limitation "
                        "and withhold publication readiness; do not widen the question "
                        "to an adjusted model or retry the same unavailable contract."
                    )
                ),
                remediation_route=(
                    "agent_plan_revision"
                    if planner_can_repair_robustness
                    else "runtime_capability"
                ),
            )
        )
    elif robustness_readiness["status"] == "too_narrow":
        findings.append(
            PlanScientificFinding(
                code="ROBUSTNESS_AXES_TOO_NARROW",
                severity="major",
                dimension="robustness",
                message=(
                    "The proposed plan has fewer typed executable robustness axes "
                    "than its study-family playbook requires."
                ),
                evidence_refs=[
                    "study_design_brief.json.sensitivity_requirements",
                    "analysis_plan.json.robustness_specs",
                ],
                remediation=(
                    "Prespecify only task-supported sensitivity alternatives "
                    "appropriate to this study family before execution."
                ),
            )
        )
    elif (
        robustness_readiness["status"] == "blocked"
        and robustness_readiness["reason"] == "typed_sensitivity_authority_not_executable"
        # User-required specs that are not executed already raise the
        # blocker above; this is the plan's own declaration.
        and not sensitivity["missing_spec_ids"]
        and not sensitivity["missing_required"]
    ):
        unexecuted = sensitivity["protocol_only_plan_spec_ids"]
        findings.append(
            PlanScientificFinding(
                code="ROBUSTNESS_SPECS_NOT_EXECUTABLE",
                severity="major",
                dimension="robustness",
                message=(
                    "The plan declares robustness that no host owner executes"
                    + (f" (specifications: {', '.join(unexecuted)})" if unexecuted else "")
                    + (
                        f"; protocol-only axes: {', '.join(sensitivity['protocol_only'])}"
                        if sensitivity["protocol_only"]
                        else ""
                    )
                    + ". Fewer typed executable axes remain than the study-family "
                    "playbook requires, and an unexecuted locked specification "
                    "fails the run closed at the robustness panel."
                ),
                evidence_refs=[
                    "analysis_plan.json.robustness_specs",
                    "study_design_brief.json.sensitivity_requirements",
                ],
                remediation=(
                    "Declare each robustness specification in the exact shape "
                    "an executing owner claims -- one robustness replay step, or "
                    "the specification the primary owner executes in its own "
                    "step -- or remove it and prespecify a task-supported "
                    "executable alternative. Do not change the headline estimand."
                ),
            )
        )
    if figure_roles["missing_roles"]:
        findings.append(
            PlanScientificFinding(
                code="FIGURE_ROLE_COVERAGE_INCOMPLETE",
                severity="major",
                dimension="figures",
                message="Explicit figure steps do not cover required article roles: " + ", ".join(figure_roles["missing_roles"]),
                evidence_refs=["article_figure_strategy.json", "analysis_plan.json"],
                remediation="Plan source-data-bound figures for each missing role; table prose elsewhere does not count as a figure.",
            )
        )
    if content_roles["missing_roles"]:
        findings.append(
            PlanScientificFinding(
                code="ARTICLE_CONTENT_ROLES_INCOMPLETE",
                severity="major",
                dimension="content_completeness",
                message="The plan lacks article content roles: " + ", ".join(content_roles["missing_roles"]),
                evidence_refs=["analysis_plan.json"],
                remediation="Add evidence-producing cohort, baseline, quality, descriptive, primary, or robustness modules as applicable.",
            )
        )
    if not literature_facts["comparison_source_keys"]:
        findings.append(
            PlanScientificFinding(
                code="NOVELTY_NOT_ESTABLISHED",
                severity="major",
                dimension="novelty",
                message=(
                    "Without a screened direct comparator or eligible design "
                    "analogue, the system cannot distinguish a genuinely novel "
                    "design from a new database instantiation."
                ),
                evidence_refs=["preplan_literature_bundle.json"],
                remediation=(
                    "Complete source-backed comparison-source screening and a "
                    "separate prespecified novelty review before making novelty "
                    "claims."
                ),
            )
        )
    else:
        findings.append(
            PlanScientificFinding(
                code="NOVELTY_POSITIONING_REVIEW_REQUIRED",
                severity="major",
                dimension="novelty",
                message=(
                    "A screened comparison-source candidate exists, but retrieval "
                    "and deterministic screening do not establish novelty. "
                    "Population, exposure, time zero, estimand, analysis route, and "
                    "clinical contribution still require an independent appraisal."
                ),
                evidence_refs=[
                    "preplan_literature_bundle.json",
                    "scientific_plan_review.json.facts.novelty_review",
                ],
                remediation=(
                    "Review the exact comparator against the six prespecified novelty "
                    "dimensions. Record what is already known, what is reused, and what "
                    "the proposed study adds before making a top-journal novelty claim."
                ),
            )
        )

    findings.extend(primary_model_retention_findings(model_retention))
    findings.extend(_missing_category_complete_case_findings(plan))
    findings.extend(_complete_case_repeats_primary_findings(plan))
    routed_findings = [
        finding.model_copy(
            update={"remediation_route": remediation_route_for_finding(finding)}
        )
        for finding in findings
    ]
    findings = routed_findings
    remediation_buckets = {
        route: [
            item.code for item in findings if item.remediation_route == route
        ]
        for route in (
            "agent_plan_revision",
            "runtime_capability",
            "study_authority_change",
            "external_evidence",
            "independent_review",
        )
    }

    dimensions = {
        "literature": 100,
        "novelty": 100,
        "literature_to_plan": 100,
        "icu_clinical_design": 100,
        "statistical_design": 100,
        "robustness": 100,
        "figures": 100,
        "content_completeness": 100,
    }
    penalty = {"blocker": 55, "major": 30, "minor": 10}
    for finding in findings:
        dimensions[finding.dimension] = max(
            0,
            dimensions.get(finding.dimension, 100) - penalty[finding.severity],
        )
    score = round(sum(dimensions.values()) / max(1, len(dimensions)))
    blockers = [item for item in findings if item.severity == "blocker"]
    majors = [item for item in findings if item.severity == "major"]
    status: Literal["changes_required", "analysis_only", "ready_for_approval"] = (
        "changes_required"
        if blockers
        else ("analysis_only" if majors else "ready_for_approval")
    )
    context_payload = context.model_dump(mode="json")
    plan_payload = plan.model_dump(mode="json")
    literature_payload = literature.model_dump(mode="json") if literature is not None else None
    figure_payload = figure_strategy.model_dump(mode="json") if figure_strategy is not None else None
    return PlanScientificReview(
        status=status,
        approval_allowed=not blockers,
        top_journal_candidate=not blockers and not majors,
        score=score,
        dimension_scores=dimensions,
        findings=findings,
        facts={
            "primary_model_retention": (
                model_retention.facts() if model_retention is not None else None
            ),
            "plan_population_requirements": population_requirements.model_dump(mode="json") if population_requirements else None,
            "population_scope_changes": population_changes,
            "accepted_baseline_coverage": baseline_coverage,
            # Carry the exact host contract into a subsequent plan-revision
            # request; a failed first replan must not erase its own requirements.
            "accepted_baseline_requirements": (
                accepted_baseline.model_dump(mode="json") if accepted_baseline else None
            ),
            "scientific_capability": capability_assessment.to_dict(),
            "reportable_capability_required": bool(require_reportable_capability),
            "score_interpretation": {
                "scope": "pre_execution_plan",
                "figures": (
                    "Figure score covers typed planned roles only. Rendered visual "
                    "quality, labels, export integrity, and source-data fidelity are "
                    "not assessed until execution."
                ),
                "content_completeness": (
                    "Content score covers planned article roles only. Result richness "
                    "and manuscript quality remain unassessed until evidence exists."
                ),
            },
            "literature": literature_facts,
            "primary_plan_citation_keys": sorted(primary_keys),
            "method_sources": method_facts,
            "literature_design_bindings": design_bindings,
            "primary_exposure_time_anchor_alignment": (
                time_anchor_alignment.to_dict()
            ),
            "post_baseline_exposure": post_baseline,
            "exposure_window": exposure_window,
            "repeat_units_possible": repeats,
            "patient_identity_available": patient_identity,
            "timing_design_executable": timing_design_closed(plan),
            "temporal_inference_required": needs_temporal_inference,
            "descriptive_only_step_ids": [
                step.step_id
                for step in scientific_steps(plan)
                if descriptive_only_step(step)
            ],
            "repeated_unit_design_executable": repeated_unit_design_closed(context, plan),
            "trajectory_representation": trajectory_representation,
            **(
                {"landmark_survival_suite": survival_suite}
                if survival_suite is not None
                else {}
            ),
            "primary_covariates": list(covariates),
            "requested_outcomes": list(expected_outcomes),
            "requested_estimate_coverage": {
                "schema_version": "easyicu.requested_estimate_coverage/1",
                "exposure_occurrence": {
                    "requested_cues": list(occurrence_cues),
                    "covering_step_ids": list(occurrence_step_ids),
                },
            },
            "model_covered_outcomes": list(covered_outcomes),
            "missing_model_outcomes": list(missing_model_outcomes),
            "covariate_selection": covariate_selection,
            "covariate_rationales": covariate_rationales,
            "covariate_temporal_roles": covariate_temporal_roles,
            "sensitivity": sensitivity,
            "robustness_readiness": robustness_readiness,
            "linearity": linearity,
            "model_term_domain_conflicts": model_term_domain_conflicts,
            "clinical_definitions": clinical_definitions,
            "figure_roles": figure_roles,
            "content_roles": content_roles,
            "novelty_status": (
                "independent_review_required"
                if literature_facts["comparison_source_keys"]
                else "not_established"
            ),
            "novelty_review": {
                "status": (
                    "independent_review_required"
                    if literature_facts["comparison_source_keys"]
                    else "comparison_source_required"
                ),
                "direct_comparator_keys": list(
                    literature_facts["direct_comparator_keys"]
                ),
                "design_analogue_keys": list(literature_facts["design_analogue_keys"]),
                "comparison_source_keys": list(
                    literature_facts["comparison_source_keys"]
                ),
                "prespecified_dimensions": list(NOVELTY_REVIEW_DIMENSIONS),
                "claim_boundary": (
                    "Candidate novelty only. The Agent may execute an approved analysis, "
                    "but neither the plan nor manuscript may claim top-journal novelty "
                    "until an independent appraisal is bound."
                ),
            },
            "remediation_buckets": remediation_buckets,
            "automatic_revision_blockers": list(plan_revision_blocker_codes(findings)),
            "remediation_boundary": (
                "Only agent_plan_revision findings may be fed to a fresh Planner "
                "without changing StudyContext authority. Runtime-capability "
                "findings require a host implementation receipt; the other lanes "
                "require user authorization, new external evidence, or independent "
                "review."
            ),
        },
        context_sha256=canonical_sha256(context_payload),
        plan_sha256=canonical_sha256(plan_payload),
        literature_sha256=canonical_sha256(literature_payload),
        figure_strategy_sha256=canonical_sha256(figure_payload),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = [
    "PlanScientificFinding",
    "PlanScientificReview",
    "association_study",
    "build_plan_scientific_review",
    "executable_scientific_step",
    "model_covariates",
    "plan_reports_endpoint_result",
    "planned_model_outcomes",
    "requested_outcomes",
    "requested_exposure_occurrence",
    "exposure_occurrence_steps",
    "method_source_facts",
    "patient_identity_available",
    "post_baseline_exposure",
    "repeat_units_possible",
    "repeated_unit_design_closed",
    "repeated_unit_estimator_present",
    "render_plan_scientific_guardrails",
    "render_agent_plan_revision_contract",
    "plan_revision_blocker_codes",
    "primary_model_retention_findings",
    "remediation_route_for_finding",
    "required_method_layers_for_context",
    "required_method_layers_for_plan",
    "scientific_steps",
    "study_endpoint_required",
    "timing_design_closed",
    "landmark_survival_suite_facts",
    "landmark_survival_suite_findings",
    "trajectory_representation_facts",
    "trajectory_representation_findings",
]
