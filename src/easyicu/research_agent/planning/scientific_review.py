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

import re
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..canonical_json import canonical_sha256
from ..concept_availability import normalize_database_name
from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..contracts.association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
)
from ..contracts.descriptive_execution import (
    DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID,
)
from ..literature import LiteratureBundle, manuscript_citable_records
from ..research_context.temporal_semantics import (
    primary_exposure_time_anchor_alignment,
    window_extends_after_anchor,
)
from ..research_context.typed import declared_domain_for_variable
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext
from .figure_strategy import ArticleFigureStrategy
from .adjustment_authority import AdjustmentSetAuthority
from .dependence_authority import (
    context_patient_group_authority,
    dependence_matches_context,
)
from .method_literature import method_binding_support
from .novelty_contract import NOVELTY_REVIEW_DIMENSIONS
from .publication_readiness import build_publication_readiness_facts
from .sensitivity_authority import (
    EXECUTABLE_METHODS_BY_STRATEGY,
    FUNCTIONAL_FORM_EXECUTABLE_METHODS,
)
from .capability_registry import assess_scientific_capability


ScientificReviewSeverity = Literal["blocker", "major", "minor"]
ScientificRemediationRoute = Literal[
    "agent_plan_revision",
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


class PlanScientificReview(BaseModel):
    """Digest-bound pre-approval review of an exact context/plan/literature set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.plan_scientific_review/4"] = (
        "easyicu.plan_scientific_review/4"
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
    return tuple(values)


def post_baseline_exposure(context: ResearchContext) -> tuple[bool, Optional[str]]:
    exposure = str(context.primary_exposure or "").strip()
    if not exposure:
        return False, None
    descriptor = context.variable(exposure)
    window = str(getattr(descriptor, "analysis_window", "") or "").strip()
    if not window:
        return False, None
    return window_extends_after_anchor(window), window


def repeat_units_possible(context: ResearchContext) -> bool:
    provenance = context.cohort.provenance or {}
    preferences = context.user_preferences
    if hasattr(preferences, "model_dump"):
        preferences = preferences.model_dump(mode="json")
    preferences = preferences if isinstance(preferences, Mapping) else {}
    n_patients = context.cohort.n_patients
    n_stays = context.cohort.n_stays
    if n_patients is not None and n_stays is not None and n_stays > n_patients:
        return True
    # Both counts being known and equal is itself owner-issued proof that every
    # stay belongs to a different person, so dependence is already ruled out.
    counts_establish_one_stay_per_patient = (
        n_patients is not None and n_stays is not None and n_stays <= n_patients
    )
    # Otherwise a stay-level cohort without patient identity cannot establish
    # that every row belongs to a different person. Treat dependence as
    # possible rather than silently upgrading "unknown" to "independent". The
    # review below can then offer the governed remedies: materialize patient
    # grouping or use an owner-issued one-stay/readmission restriction.
    analysis_unit = str(provenance.get("analysis_unit") or "").strip().casefold()
    if (
        provenance.get("evidence_stage") == "metadata_only_planning"
        and analysis_unit == "icu_stay"
        and context_patient_group_authority(context) is None
    ):
        return True
    if (
        n_stays is not None
        and n_stays > 1
        and analysis_unit == "icu_stay"
        and not counts_establish_one_stay_per_patient
        and context_patient_group_authority(context) is None
    ):
        return True
    text = " ".join(
        [
            *[str(value) for value in context.cohort.inclusion_criteria],
            *[str(value) for value in context.cohort.exclusion_criteria],
            *[str(value) for value in provenance.get("inclusion_criteria") or ()],
            *[str(value) for value in provenance.get("exclusion_criteria") or ()],
            str(preferences.get("data_constraints") or ""),
            str(preferences.get("extra_notes") or ""),
        ]
    ).casefold()
    return any(
        token in text
        for token in (
            "repeat",
            "readmission",
            "re-admission",
            "repeated stay",
            "multiple icu",
            "retain icu readmissions",
            "重复",
            "再次 icu",
            "再次icu",
        )
    )


def patient_identity_available(context: ResearchContext) -> bool:
    return context_patient_group_authority(context) is not None


_EXECUTABLE_TEMPORAL_METHODS = frozenset(
    {
        "signed_landmark_restricted_cubic_spline",
        "signed_landmark_survival_suite",
        "landmark_analysis",
        "time_varying_exposure_model",
        "clone_censor_weight",
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
        or step.scientific_capability is not None
        or step.model_requirements
        or step.family_primary_result_requirement is not None
        or not step.sensitivity_spec_ids
    ):
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
        _method_head(step) in _EXECUTABLE_TEMPORAL_METHODS for step in applicable
    )


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
        method = _method_head(step)
        model_requirements = tuple(step.model_requirements)
        distribution = step.exposure_outcome_distribution_spec
        patient_group = context_patient_group_authority(context)
        signed_landmark_dependence = bool(
            method == "signed_landmark_restricted_cubic_spline"
            and patient_group is not None
            and patient_group.group_source in step.inputs
            and any(
                str(value).startswith("scientific_runtime_contract:")
                for value in step.icu_rule_refs
            )
        )
        counts_only_distribution = bool(
            distribution is not None
            and distribution.schema_version
            == "easyicu.exposure_outcome_distribution/3"
        )
        if not (
            model_requirements
            or distribution is not None
            or method in _EXECUTABLE_DEPENDENCE_METHODS
            or signed_landmark_dependence
            or method == "non_readmission_restriction"
        ):
            continue
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
    for step in scientific_steps(plan):
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
        if not sorted_layers:
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


def _sensitivity_facts(
    context: ResearchContext, plan: AnalysisPlan
) -> dict[str, Any]:
    requested = _requested_sensitivity_axes(context)
    typed_specs = {spec.spec_id: spec for spec in _sensitivity_specs(context)}
    executed_spec_ids: set[str] = set()
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
                and step.scientific_capability
                == ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID
                and step.sensitivity_spec_ids
                and len(step.expected_outputs) == 1
                and str(step.expected_outputs[0]).startswith("table:")
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
                    and method in EXECUTABLE_METHODS_BY_STRATEGY[spec.strategy]
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
        else:
            protocol_only.update(axes)
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
    plan_spec_axes = {
        {"missing": "missing", "cohort": "cohort", "outcome": "outcome_definition"}[
            spec.axis
        ]
        for spec in plan.robustness_specs
    }
    if len(replay_steps) == 1:
        executable.update(plan_spec_axes)
        typed_executable.update(plan_spec_axes)
    else:
        protocol_only.update(plan_spec_axes)
    # A temporal phrase in a generic generated-code step is not proof that the
    # estimator closes immortal-time/exposure-opportunity bias.
    if "timing" in executable and not timing_design_closed(plan):
        executable.discard("timing")
        protocol_only.add("timing")
    if "readmission" in executable and not repeated_unit_design_closed(context, plan):
        executable.discard("readmission")
        protocol_only.add("readmission")
    missing_spec_ids = sorted(set(typed_specs) - executed_spec_ids)
    typed_axes = {
        {
            "repeated_stays": "readmission",
            "missing_data": "missing",
        }.get(spec.axis, spec.axis)
        for spec_id, spec in typed_specs.items()
        if spec_id in executed_spec_ids
    }
    executable.update(typed_axes)
    typed_executable.update(typed_axes)
    return {
        "requested": sorted(requested),
        "executable": sorted(executable),
        "typed_executable": sorted(typed_executable),
        "protocol_only": sorted(protocol_only - executable),
        "missing_required": sorted(requested - executable),
        "typed_spec_ids": sorted(typed_specs),
        "executed_spec_ids": sorted(executed_spec_ids),
        "missing_spec_ids": missing_spec_ids,
        "plan_robustness_spec_ids": sorted(
            spec.spec_id for spec in plan.robustness_specs
        ),
        "plan_robustness_replay_step_ids": sorted(
            step.step_id for step in replay_steps
        ),
    }


def _continuous_linearity_facts(plan: AnalysisPlan) -> dict[str, Any]:
    identity_terms: list[str] = []
    for step in scientific_steps(plan):
        for requirement in step.model_requirements:
            for term in requirement.model_terms or ():
                if term.role == "covariate" and term.coding == "continuous" and str(term.transform or "").casefold() in {"", "identity"}:
                    identity_terms.append(term.name)
    has_functional_form_sensitivity = any(
        executable_scientific_step(step)
        and (
            _method_head(step) in FUNCTIONAL_FORM_EXECUTABLE_METHODS
            or any(
                token
                in " ".join(
                    [step.step_id, step.intent, step.method or "", *step.expected_outputs]
                ).casefold()
                for token in (
                    "spline",
                    "nonlinear",
                    "non-linear",
                    "functional form",
                    "functional_form",
                    "fractional polynomial",
                )
            )
        )
        for step in scientific_steps(plan)
    )
    return {
        "linear_identity_terms": sorted(set(identity_terms)),
        "functional_form_sensitivity_executable": has_functional_form_sensitivity,
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
        and not any(
            "endpoint-definition conflict" in str(value).casefold()
            for value in descriptor.clinical_caveats
        )
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


def remediation_route_for_finding(
    finding: PlanScientificFinding,
) -> ScientificRemediationRoute:
    """Assign one owner lane without letting the Agent revise the estimand."""

    if finding.requires_user_authorization:
        return "study_authority_change"
    if finding.code in _EXTERNAL_EVIDENCE_FINDINGS:
        return "external_evidence"
    if finding.code in _INDEPENDENT_REVIEW_FINDINGS:
        return "independent_review"
    return "agent_plan_revision"


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
        "- preserve the exact research question, endpoint, cohort, exposure, "
        "outcome, time window, and user-authorized covariate authority.",
        "- do not claim that this revision closes study-authority, external-"
        "evidence, or independent-review findings.",
        "- fix these plan-owned findings with executable typed steps:",
    ]
    lines.extend(
        f"  - {item.code}: {item.remediation}" for item in automatic
    )
    return "\n".join(lines)


def build_plan_scientific_review(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    literature: Optional[LiteratureBundle] = None,
    figure_strategy: Optional[ArticleFigureStrategy] = None,
    require_reportable_capability: bool = False,
) -> PlanScientificReview:
    """Score and adjudicate the exact proposed plan before human approval."""

    findings: list[PlanScientificFinding] = []
    literature_facts = _literature_facts(literature, context)
    method_facts = method_source_facts(plan, context)
    design_bindings = _literature_design_bindings(plan, literature)
    sensitivity = _sensitivity_facts(context, plan)
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
        and _method_head(step) == "signed_landmark_restricted_cubic_spline"
        for step in plan.steps
    ):
        # This method can appear only after the digest-bound runtime owner has
        # replaced the generic primary step. Its exact adjustment columns are
        # therefore the operational projection of the sealed StudyContext,
        # not an inference from plan inputs or available data.
        covariates = adjustment_authority.operational_covariates
    covariate_selection = (
        preferences.covariate_selection if preferences is not None else "planner_selectable"
    )
    covariate_rationales = adjustment_authority.operational_rationales
    covariate_temporal_roles = adjustment_authority.operational_temporal_roles
    capability_assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type,
        context=context,
        plan=plan,
    )
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
    if not _endpoint_resolved(context):
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
    needs_temporal_inference = temporal_inference_required(plan)
    if post_baseline and needs_temporal_inference and not timing_design_closed(plan):
        findings.append(
            PlanScientificFinding(
                code="POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
                severity="blocker",
                dimension="icu_clinical_design",
                message="Exposure is classified after ICU time zero, but no executable temporal estimator closes exposure opportunity and early events.",
                evidence_refs=["research_context.json", "analysis_plan.json"],
                remediation="Create a new, user-authorized landmark/time-varying study version or keep this version descriptive; a protocol-only step cannot close the bias.",
                requires_user_authorization=True,
                authorization_question="Should a new study version use a prespecified landmark/time-varying design, or should the current question remain descriptive?",
            )
        )
    if repeats and not patient_identity and not repeated_unit_design_closed(context, plan):
        findings.append(
            PlanScientificFinding(
                code="REPEATED_STAY_IDENTITY_UNAVAILABLE",
                severity="blocker",
                dimension="icu_clinical_design",
                message="The stay-level cohort does not expose patient identity, so repeated ICU stays cannot be ruled out or handled.",
                evidence_refs=["research_context.json.cohort.provenance"],
                remediation="Materialize an authorized patient identifier for clustered/first-stay inference, or prespecify an executable non-readmission restriction using an owner-issued readmission indicator in a new study version.",
                requires_user_authorization=True,
                authorization_question="Should a new study version materialize patient identity for first-stay/clustered estimation, or use a prespecified non-readmission restriction when an owner-issued indicator is available?",
            )
        )
    elif repeats and not repeated_unit_design_closed(context, plan):
        findings.append(
            PlanScientificFinding(
                code="REPEATED_STAY_METHOD_NOT_DECLARED",
                severity="blocker",
                dimension="icu_clinical_design",
                message="Patient identity exists, but no executable estimator addresses repeated ICU stays.",
                evidence_refs=["research_context.json", "analysis_plan.json"],
                remediation="Prespecify an executable one-stay, clustered, or mixed model in a new study version.",
                requires_user_authorization=True,
                authorization_question="Should the new study use one stay per patient or retain stays with clustered/mixed estimation?",
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
        findings.append(
            PlanScientificFinding(
                code="UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE",
                severity="major",
                dimension="statistical_design",
                message="The primary association is unadjusted and therefore supports descriptive, not independent-association, interpretation.",
                evidence_refs=["analysis_plan.json"],
                remediation="Ask the user to authorize a clinically justified, time-zero-available adjustment strategy or retain an explicitly descriptive claim ceiling.",
                requires_user_authorization=True,
                authorization_question="Keep the analysis descriptive, or authorize a new clinically timed adjustment strategy?",
            )
        )
    if covariates and covariate_selection != "exact":
        findings.append(
            PlanScientificFinding(
                code="ADJUSTMENT_SET_NOT_USER_CONFIRMED",
                severity="blocker",
                dimension="statistical_design",
                message="The Planner selected an exact covariate roster from candidates the user had not approved as a prespecified adjustment set.",
                evidence_refs=["research_context.json.user_preferences", "analysis_plan.json.model_requirements"],
                remediation="Show the proposed roster, clinical rationale, and pre-time-zero availability for explicit user approval in a new StudyContext revision.",
                requires_user_authorization=True,
                authorization_question="Do you approve this exact adjustment set and its clinical/time-zero rationale for a new study version?",
            )
        )
    elif covariates and (
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
                message="Continuous covariates enter linearly without an executable functional-form check: " + ", ".join(linearity["linear_identity_terms"]),
                evidence_refs=["analysis_plan.json.model_requirements"],
                remediation="Add a prespecified spline/nonlinearity sensitivity with source binding, without changing the headline estimand after results are seen.",
            )
        )
    if (
        robustness_readiness["status"] == "blocked"
        and robustness_readiness["reason"] == "no_typed_sensitivity_authority"
    ):
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
                    "Issue a new user-reviewed sensitivity authority for a "
                    "task-supported denominator, missingness/measurement, "
                    "outcome-definition, timing, or model axis. Descriptive "
                    "studies must not invent an effect-estimate replay grid."
                ),
                requires_user_authorization=True,
                authorization_question=(
                    "Do you want to prespecify executable, study-family-appropriate "
                    "sensitivity analyses in a new study version?"
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
    if not figure_roles["distinct_chart_types_complete"]:
        findings.append(
            PlanScientificFinding(
                code="FIGURE_CHART_TYPES_TOO_NARROW",
                severity="major",
                dimension="figures",
                message=(
                    "The typed figure plan declares fewer distinct valid chart "
                    "families than the article figure strategy requires."
                ),
                evidence_refs=[
                    "article_figure_strategy.json.minimum_distinct_chart_types",
                    "analysis_plan.json.steps.figure_panels",
                ],
                remediation=(
                    "Declare complementary typed panels with exact article roles, "
                    "accepted chart types, figure outputs, and source products; "
                    "renaming a table or generic overview does not count."
                ),
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
            "primary_covariates": list(covariates),
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
            "remediation_boundary": (
                "Only agent_plan_revision findings may be fed to a fresh Planner "
                "without changing StudyContext authority. The other lanes require "
                "user authorization, new external evidence, or independent review."
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
    "method_source_facts",
    "patient_identity_available",
    "post_baseline_exposure",
    "repeat_units_possible",
    "repeated_unit_design_closed",
    "render_plan_scientific_guardrails",
    "render_agent_plan_revision_contract",
    "remediation_route_for_finding",
    "required_method_layers_for_context",
    "required_method_layers_for_plan",
    "scientific_steps",
    "timing_design_closed",
]
