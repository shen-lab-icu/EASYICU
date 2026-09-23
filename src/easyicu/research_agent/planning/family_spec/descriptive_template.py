"""Host template for the descriptive exposure–outcome family.

A descriptive epidemiology question with a closed-level primary exposure and a
0/1 outcome (for example: the proportion of ICU stays meeting a phenotype and
the in-hospital mortality of each group). The Planner's whole contribution is
the baseline-table variable set, the reader labels (including the two level
labels a binary exposure needs), and the comparator applications; everything
executable is projected from typed StudyContext facts and judged by the
unchanged Progressive validators and compiler.

Step layout (mirrors the family's reference workflow):

1. ``cohort_accounting``             denominators and cohort flow for every input row
2. ``baseline_context``              baseline table by exposure level, SMD only
3. ``exposure_outcome_distribution`` counts and proportions by exposure level (primary)
4. ``measurement_audit``             measurement coverage and missingness

The host later adds the descriptive-context, cohort-accounting, and
data-quality figures; the family declares no report step, matching the
reference plans it was projected from.
"""

from __future__ import annotations

from typing import Callable

from ...canonical_json import canonical_sha256
from ..design_selection import ResearchDesignCandidate, ResearchDesignSelection
from ..progressive_contract import (
    ProgressiveDisplayLabel,
    ProgressiveFoundationMaterialization,
    ProgressiveLiteratureBinding,
    ProgressiveOutlineStep,
    ProgressiveOutputIntent,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressiveProductRef,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
    ProgressiveTableOneVariable,
)
from .contract import DESCRIPTIVE_FAMILY_ID, FamilyPlanSpec, FamilySpecError, FamilySpecRequest
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _cohort_intent,
    _label,
    _method_card_elements,
    _method_card_ids,
)

TABLE_ONE_ACTION = "descriptive.table_one"
DISTRIBUTION_ACTION = "descriptive.descriptive_summary"
MISSINGNESS_ACTION = "descriptive.missingness_audit"
_DISTRIBUTION_DESIGN_ELEMENTS = (
    "dependence",
    "estimand",
    "exposure",
    "outcome",
    "reporting",
    "time_zero",
)
_AUDIT_OUTPUTS = (
    ("table:measurement_missingness", "measurement_missingness"),
    ("table:measurement_process_audit", "measurement_process"),
)


def _bindings(
    outline_step: ProgressiveOutlineStep,
    *,
    comparator_applications: dict[str, str],
) -> list[ProgressiveLiteratureBinding]:
    desired = set(_DISTRIBUTION_DESIGN_ELEMENTS)
    bindings: list[ProgressiveLiteratureBinding] = []
    for key in outline_step.literature_citation_keys:
        if key in comparator_applications:
            bindings.append(
                ProgressiveLiteratureBinding(
                    citation_key=key,
                    design_elements=["population", "exposure", "time_zero", "outcome", "estimand"],
                    application=comparator_applications[key],
                    divergence=None,
                )
            )
            continue
        elements = sorted(desired & _method_card_elements(key))
        if not elements:
            continue
        bindings.append(
            ProgressiveLiteratureBinding(
                citation_key=key,
                design_elements=elements,
                application=(
                    f"Apply the host-curated method card(s) {_method_card_ids(key, desired)} to the "
                    "descriptive exposure–outcome distribution; the family template binds only "
                    "the curated design elements and retains the analysis-only claim ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _table_one_summary(coding: str) -> str:
    return "count_percent" if coding in {"binary", "categorical"} else "both"


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    required_variables: list[str],
    method_keys: list[str],
) -> ResearchDesignSelection:
    exposure = _label(spec, request.primary_exposure)
    outcome = _label(spec, request.outcome)
    levels = ", ".join(request.exposure_levels)
    window = (
        f"ICU admission; the exposure uses the 0–{request.observation_window_hours:g} h observation window"
        if request.observation_window_hours is not None
        else "ICU admission; the exposure uses the host-materialized observation window"
    )
    unit_text = (
        "each analysis row is one ICU stay; patient-level dependence is declared and repeated "
        "stays are reported, not assumed independent"
        if request.cluster_unit == "patient"
        else "each analysis row is one ICU stay and rows are not assumed to be distinct patients"
    )
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    selected = ResearchDesignCandidate(
        design_id="stay_level_descriptive",
        analysis_type="descriptive_epidemiology",
        estimand=(
            f"The number and proportion of analysis rows in each {exposure} level ({levels}) and "
            f"the number and proportion with {outcome} within each level; {unit_text}."
        ),
        time_zero=window,
        observation_window=(
            f"Exposure measured in the host-bound observation window; {outcome} taken from the "
            "available hospital outcome record for every analysis row."
        ),
        primary_method=(
            "Counts and proportions by exposure level, plus a baseline table by level; no adjusted "
            "model, no inferential contrast, and no confidence interval, standard error, or p-value "
            "(typed counts-only design: repeated units cannot be resolved)."
            if request.counts_only
            else "Counts and proportions by exposure level with confidence intervals, plus a baseline "
            "table by level; no adjusted model and no inferential contrast is proposed."
        ),
        required_variables=required_variables,
        assumptions=[
            "Each input row is one analysis unit; rows are not assumed to be distinct patients.",
            "The exposure classification uses only the sealed observation window.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=[],
        novelty_positioning=(
            "No novelty is claimed before completion; the description is positioned against each "
            "screened comparator on population, exposure definition, time zero, and estimand."
        ),
        figure_role=(
            "The exposure distribution and level-specific outcome proportions as the main display, "
            "with cohort denominators and data-quality context."
        ),
        supports=(
            f"Row-level counts and proportions of {exposure} levels and of {outcome} within each level."
        ),
        cannot_prove=(
            "No causal effect, no patient-level incidence, no adjusted or independent association, and "
            "no transportability beyond the source population."
        ),
        reviewable_plan=[
            (
                f"Population and unit: all input rows of cohort {request.cohort_name}"
                + (
                    " that meet the typed eligibility bound"
                    if request.cohort_selection_mode == "predicate_filtered"
                    else ""
                )
                + f"; row identity {request.identity_column}; {unit_text}."
            ),
            f"Exposure: {exposure} with closed levels {levels}; rows without an evaluable level are counted and reported, never recoded.",
            f"Outcome: {outcome}; counts and proportions are reported overall and by exposure level.",
            "No adjustment model: the design is descriptive and proposes no adjusted association.",
            "Missing data and coverage: measurement availability and missingness of the exposure, "
            "outcome, and baseline variables are audited before any proportion is interpreted.",
            "Feasibility: closed exposure levels, denominator consistency, repeated-row possibility, "
            "and window availability are checked before materialization.",
        ],
        disposition="selected",
        decision_reason=(
            "The question asks for occurrence proportions and outcome proportions by group; the sealed "
            "data support a closed row-level description without an adjusted estimand, chosen before "
            "any data are read."
        ),
    )
    patient_level_supported = request.cluster_unit == "patient"
    rejected = ResearchDesignCandidate(
        design_id="patient_level_first_stay",
        analysis_type="descriptive_epidemiology",
        estimand=(
            f"The proportion of distinct patients whose first ICU stay is in each {exposure} level and "
            f"the proportion with {outcome} in each level."
        ),
        time_zero="Each patient's first ICU admission; the exposure uses the same observation window.",
        observation_window="The first ICU stay's observation window and its hospital outcome record.",
        primary_method="Counts and proportions after keeping one first stay per patient.",
        required_variables=required_variables,
        assumptions=[
            "A verified cross-stay patient identity and a first-stay rule exist for every row."
        ],
        literature_citation_keys=[
            key for key in ("strobe_2007", "record_2015") if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning=(
            "Recorded as the patient-level alternative for audit; no superiority or novelty is claimed."
        ),
        figure_role="Patient-level exposure distribution and level-specific outcome proportions.",
        supports=(
            "Patient-level descriptive proportions when a verified first-stay rule is authorized."
            if patient_level_supported
            else "Nothing on the sealed data: no verified patient identity is available."
        ),
        cannot_prove=(
            "It changes the analysis unit the user authorized and needs a separate first-stay "
            "authority; it is not the primary description."
            if patient_level_supported
            else "It cannot verify de-duplication or first stays without a patient identity, so it "
            "would introduce an unauthorized de-duplication assumption."
        ),
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because the authorized analysis unit is the ICU stay; a first-stay restriction "
            "is a separate reviewable sensitivity, not the primary description."
            if patient_level_supported
            else "Rejected because the sealed data provide no verified patient identity; de-duplicating "
            "to first stays would rest on an unauthorized assumption."
        ),
    )
    return ResearchDesignSelection(candidates=[selected, rejected])


def _outline_step(
    *,
    step_id: str,
    role: str,
    module_id: str,
    objective: str,
    depends_on: list[str],
    variable_names: list[str],
    citations: list[str],
    action: str | None = None,
) -> ProgressiveOutlineStep:
    return ProgressiveOutlineStep(
        step_id=step_id,
        planned_analysis_role=role,
        module_id=module_id,
        objective=objective,
        depends_on=depends_on,
        variable_names=list(dict.fromkeys(variable_names)),
        literature_citation_keys=list(dict.fromkeys(citations))[:12],
        scientific_action_id=action,
    )


def _ref(producer: str, product: str) -> ProgressiveProductRef:
    return ProgressiveProductRef(producer_step_id=producer, product_id=product)


def build_descriptive_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from typed facts."""

    if request.family_id != DESCRIPTIVE_FAMILY_ID:
        raise FamilySpecError(
            "family_spec_template_mismatch",
            "the descriptive template received a request for another family",
            path="family_id",
        )
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "the spec does not bind this request",
            path="request_sha256",
        )
    exposure = request.primary_exposure
    outcome = request.outcome
    identity = request.identity_column
    baseline = [str(value).strip() for value in spec.baseline_variables]
    candidates = {item.name: item for item in request.adjustment_candidates}
    method_keys = [key for key in request.allowed_literature_citation_keys if _method_card_elements(key)]
    distribution_keys = list(
        dict.fromkeys(
            [
                *(k for k in method_keys if _method_card_elements(k) & set(_DISTRIBUTION_DESIGN_ELEMENTS)),
                *request.direct_comparator_literature_keys,
            ]
        )
    )[:12]
    required_variables = list(dict.fromkeys([identity, exposure, outcome]))
    design = _design_selection(
        request,
        spec,
        required_variables=required_variables,
        method_keys=[k for k in method_keys if k in set(distribution_keys)][:6],
    )
    exposure_label = _label(spec, exposure)
    outcome_label = _label(spec, outcome)
    objectives = {
        "cohort_accounting": (
            "Account for every input analysis row: the source denominator, row count, and cohort "
            "flow that later proportions are read against."
        ),
        "baseline_context": (
            f"Describe baseline characteristics of the analysis rows by {exposure_label} level with "
            "standardized differences only; repeated units carry no independent tests."
        ),
        "exposure_outcome_distribution": (
            f"Describe the distribution of {exposure_label} levels and the proportion with "
            f"{outcome_label} within each level, with denominators"
            + (
                "; no confidence interval or p-value under the typed counts-only design."
                if request.counts_only
                else " and confidence intervals."
            )
        ),
        "measurement_audit": (
            f"Audit measurement coverage, missingness, and process for {exposure_label}, "
            f"{outcome_label}, and the baseline variables before any proportion is read."
        ),
    }
    outline_steps = [
        _outline_step(
            step_id="cohort_accounting", role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            variable_names=[identity, exposure, outcome], citations=[],
        ),
        _outline_step(
            step_id="baseline_context", role="auxiliary", module_id="table_one",
            objective=objectives["baseline_context"], depends_on=["cohort_accounting"],
            variable_names=[exposure, *baseline], citations=[], action=TABLE_ONE_ACTION,
        ),
        _outline_step(
            step_id="exposure_outcome_distribution", role="primary",
            module_id="exposure_outcome_distribution",
            objective=objectives["exposure_outcome_distribution"], depends_on=["cohort_accounting"],
            variable_names=[exposure, outcome], citations=distribution_keys,
            action=DISTRIBUTION_ACTION,
        ),
        _outline_step(
            step_id="measurement_audit", role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
            variable_names=[exposure, outcome, *baseline, *request.measurement_audit_columns],
            citations=[], action=MISSINGNESS_ACTION,
        ),
    ]
    outline = ProgressivePlanOutline(
        analysis_type="descriptive_epidemiology",
        cohort_objective=(
            f"Describe the distribution of {exposure_label} and the proportion with {outcome_label} "
            f"by level among the analysis rows of cohort {request.cohort_name}, with denominators, "
            "baseline context, and measurement limits visible for review."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: the host projected every executable coordinate "
            "from typed StudyContext facts; the Planner supplied the baseline-table variables, reader "
            "labels, and comparator applications. Results are row-level counts and proportions under "
            "the analysis-only claim ceiling."
        ),
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    bound = {step.step_id: step for step in outline.steps}
    steps = [
        ProgressiveSkeletonStep(
            step_id="cohort_accounting", planned_analysis_role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            raw_inputs=[identity, exposure, outcome], literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="baseline_context", planned_analysis_role="auxiliary", module_id="table_one",
            objective=objectives["baseline_context"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([exposure, *baseline])),
            scientific_action_id=TABLE_ONE_ACTION,
            table_one_group_by=exposure, table_one_mode="descriptive_smd_only",
            table_one_variables=[
                ProgressiveTableOneVariable(
                    name=name,
                    summary=_table_one_summary(
                        candidates[name].allowed_codings[0] if name in candidates else "continuous"
                    ),
                )
                for name in baseline
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="exposure_outcome_distribution", planned_analysis_role="primary",
            module_id="exposure_outcome_distribution",
            objective=objectives["exposure_outcome_distribution"], depends_on=["cohort_accounting"],
            raw_inputs=[exposure, outcome],
            product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
            scientific_action_id=DISTRIBUTION_ACTION,
            primary_exposure=exposure, outcome=outcome, outcome_type="binary",
            event_level_index=request.event_level_index,
            reference_exposure_level_index=request.reference_level_index,
            comparison_exposure_level_index=request.primary_contrast_level_index,
            denominator_policy="all_declared_rows",
            missing_exposure_policy="fail_closed",
            missing_outcome_policy="fail_closed",
            confidence_level=0.95,
            literature_bindings=_bindings(
                bound["exposure_outcome_distribution"], comparator_applications=spec.applications
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="measurement_audit", planned_analysis_role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([exposure, outcome, *baseline, *request.measurement_audit_columns])),
            product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role=role)
                for product, role in _AUDIT_OUTPUTS
            ],
            scientific_action_id=MISSINGNESS_ACTION,
            literature_bindings=[],
        ),
    ]
    if [step.step_id for step in steps] != [step.step_id for step in outline.steps]:
        raise FamilySpecError(
            "family_spec_template_step_mismatch",
            "template outline and materialization rosters diverged",
            path="steps",
        )
    outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
    allowed_label_keys = set(request.variable_roster) | set(request.level_label_keys)
    foundation = ProgressiveFoundationMaterialization(
        outline_sha256=outline_sha256,
        foundation=ProgressivePlanFoundation(
            cohort=_cohort_intent(request),
            display_labels=[
                ProgressiveDisplayLabel(key=key, value=value)
                for key, value in spec.labels.items()
                if key in allowed_label_keys
            ],
            robustness_intents=[],
            know_how_decisions=[],
        ),
    )
    materializations = tuple(
        ProgressiveStepMaterialization(
            outline_step_sha256=canonical_sha256(bound[step.step_id].model_dump(mode="json")),
            foundation=None,
            step=step,
        )
        for step in steps
    )
    return FamilySkeletonDraft(outline=outline, foundation=foundation, materializations=materializations)


__all__ = [
    "DISTRIBUTION_ACTION",
    "MISSINGNESS_ACTION",
    "TABLE_ONE_ACTION",
    "build_descriptive_skeleton",
]
