"""Host template for the sealed fixed-landmark survival suite family.

A time-to-event question whose runtime is already sealed by a
``LandmarkSurvivalRuntimeAuthority`` (risk-set rule, Table 1, Kaplan-Meier,
adjusted Cox, PH audit, PH-free alternatives and the composite figure).  The
Planner decides nothing scientific here: it supplies reader labels for the
sealed columns and the comparator application sentences.  The template
projects a cohort-accounting step and one primary step that names the sealed
owner and copies its source columns and products verbatim; the host's
``bind_plan`` later replaces that primary with the signed suite and its
renderer, so every analysis coordinate is the authority's.

Step layout:

1. ``cohort_accounting``        denominators and cohort flow
2. ``baseline_context``         Table 1 by exposure status, SMD only
3. ``primary_survival_suite``   the sealed suite owner (primary)
4. ``report``                   zero-patient-row plan report

The Table 1 and cohort-flow drafts satisfy the outline's article-role owners;
after ``bind_plan`` the sealed suite carries both inside its own step.
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
from .contract import (
    LANDMARK_SURVIVAL_FAMILY_ID,
    FamilyPlanSpec,
    FamilySpecError,
    FamilySpecRequest,
)
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _cohort_intent,
    _label,
    _method_card_elements,
    _method_card_ids,
)
from .plan_language import listing, plan_language, sentence

_PRIMARY_DESIGN_ELEMENTS = (
    "dependence",
    "estimand",
    "exposure",
    "outcome",
    "reporting",
    "time_zero",
    "missing_data",
    "robustness",
)


def _bindings(
    outline_step: ProgressiveOutlineStep,
    *,
    comparator_applications: dict[str, str],
) -> list[ProgressiveLiteratureBinding]:
    desired = set(_PRIMARY_DESIGN_ELEMENTS)
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
                    "sealed landmark survival suite; the family template binds only the curated "
                    "design elements and retains the analysis-only claim ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    method_keys: list[str],
) -> ResearchDesignSelection:
    sealed = request.sealed_suite
    assert sealed is not None
    exposure = _label(spec, request.primary_exposure)
    outcome = _label(spec, request.outcome)
    adjustment_text = ", ".join(_label(spec, name) for name in sealed.adjustment_columns)
    landmark = f"{sealed.landmark_hours:g} h after ICU admission"
    horizon = f"{sealed.endpoint_horizon_days:g} days"
    unit_text = (
        "each analysis row is one ICU stay; patient-level dependence is declared"
        if request.cluster_unit == "patient"
        else "each analysis row is one ICU stay and rows are not assumed to be distinct patients"
    )
    language = plan_language(request.research_question)
    unit_text_zh = (
        "每行为一次 ICU 入住；已声明患者层面的相关性"
        if request.cluster_unit == "patient"
        else "每行为一次 ICU 入住，不假定各行来自不同患者"
    )
    landmark_zh = f"ICU 入院后 {sealed.landmark_hours:g} h"
    horizon_zh = f"{sealed.endpoint_horizon_days:g} 天"
    adjustment_zh = listing([_label(spec, name) for name in sealed.adjustment_columns], language)
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    selected = ResearchDesignCandidate(
        design_id="fixed_landmark_survival_suite",
        analysis_type="survival",
        estimand=(
            f"The adjusted hazard ratio for {outcome} through {horizon} comparing stays with "
            f"incident {exposure} by {landmark} against stays without it, among stays alive and "
            f"event-free at the landmark, adjusted for {adjustment_text}; reported as a descriptive "
            "prognostic association, not a causal effect."
        ),
        time_zero=f"ICU admission; follow-up starts at the {landmark} landmark.",
        observation_window=(
            f"Exposure status is fixed at the {landmark} landmark from exposure onset times; the "
            f"endpoint is followed to {horizon} with administrative censoring."
        ),
        primary_method=(
            "Sealed fixed-landmark survival suite: risk-set accounting, Table 1, Kaplan-Meier, "
            "adjusted Cox with a Schoenfeld audit, and a signed non-PH policy (interval-specific "
            "Cox or an unadjusted RMST contrast), rendered as one composite figure."
        ),
        required_variables=[request.identity_column, *sealed.source_columns],
        assumptions=[
            "Exposure onset times are recorded so prevalent exposure at time zero can be excluded.",
            f"{unit_text[0].upper()}{unit_text[1:]}.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=[],
        novelty_positioning=(
            "No novelty is claimed before completion; the design is positioned against each screened "
            "comparator on population, exposure timing, landmark, and endpoint."
        ),
        figure_role=(
            "Kaplan-Meier survival as the main visual with the adjusted contrast, risk-set "
            "accounting, and proportional-hazards diagnostics in one composite figure."
        ),
        supports=(
            f"A prespecified landmark survival association between {exposure} and {outcome} with "
            "its risk-set accounting and assumption audit."
        ),
        cannot_prove=(
            "No causal ventilation effect, no immortal-time-free estimate outside the landmark rule, "
            "and no independence of repeated ICU stays beyond the declared grouping."
        ),
        reviewable_plan=(
            [
                f"研究队列；{unit_text_zh}；纳入在 {landmark_zh} landmark 时存活且终点有效的入住。",
                f"截至 {landmark_zh} 新发的 {exposure}；排除时间零点时已存在的暴露。",
                f"自 ICU 入院起 {horizon_zh} 内的 {outcome}，按行政截尾处理。",
                f"调整 {adjustment_zh} 的 Cox 比例风险模型；Wald 95% CI；按封印的处理政策做 "
                "Schoenfeld 残差审计。",
                "对封印的列做完整病例分析，并审计分母。",
                "比例风险假设被拒绝时，以预先设定的分段 Cox 模型和未调整的限制平均生存时间对比，"
                "替代恒定的风险比。",
            ]
            if language == "zh"
            else [
                f"The study cohort; {unit_text}; stays alive at the {landmark} landmark with a valid "
                "endpoint.",
                f"Incident {exposure} by {landmark}; prevalent exposure at time zero is excluded.",
                sentence(f"{outcome} through {horizon} from ICU admission, censored administratively."),
                f"Adjusted Cox proportional-hazards model for {adjustment_text}; Wald 95% CI; Schoenfeld "
                "residual audit with the sealed handling policy.",
                "Complete-case on the sealed columns with an audited denominator.",
                "Prespecified interval-specific Cox and an unadjusted restricted-mean survival-time "
                "contrast replace a constant hazard ratio when proportional hazards is rejected.",
            ]
        ),
        disposition="selected",
        decision_reason=(
            "The question asks for a time-respecting survival association; the sealed landmark "
            "suite fixes exposure status before follow-up starts and audits its own assumptions, "
            "chosen before any data are read."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="binary_logistic_at_horizon",
        analysis_type="survival",
        estimand=f"An adjusted odds ratio for {outcome} by {horizon} ignoring follow-up time.",
        time_zero="ICU admission.",
        observation_window=f"Status at {horizon} only.",
        primary_method="Logistic regression on the horizon status.",
        required_variables=[request.identity_column, *sealed.source_columns],
        assumptions=["Censoring before the horizon is negligible."],
        literature_citation_keys=[
            key for key in ("strobe_2007", "record_2015") if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning="Recorded as the rejected alternative for audit; no novelty is claimed.",
        figure_role="A single forest plot.",
        supports="A horizon-status contrast when censoring is negligible.",
        cannot_prove=(
            "It discards event timing and censoring and cannot respect the landmark exposure rule."
        ),
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because the question is time-to-event with censoring and the sealed suite "
            "already fixes the landmark design."
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
) -> ProgressiveOutlineStep:
    return ProgressiveOutlineStep(
        step_id=step_id,
        planned_analysis_role=role,
        module_id=module_id,
        objective=objective,
        depends_on=depends_on,
        variable_names=list(dict.fromkeys(variable_names)),
        literature_citation_keys=list(dict.fromkeys(citations))[:12],
        scientific_action_id=None,
    )


def _ref(producer: str, product: str) -> ProgressiveProductRef:
    return ProgressiveProductRef(producer_step_id=producer, product_id=product)


def build_landmark_survival_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from the sealed suite."""

    if request.family_id != LANDMARK_SURVIVAL_FAMILY_ID or request.sealed_suite is None:
        raise FamilySpecError(
            "family_spec_template_mismatch",
            "the landmark survival template received a request for another family",
            path="family_id",
        )
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "the spec does not bind this request",
            path="request_sha256",
        )
    sealed = request.sealed_suite
    identity = request.identity_column
    method_keys = [key for key in request.allowed_literature_citation_keys if _method_card_elements(key)]
    primary_keys = list(
        dict.fromkeys(
            [
                *(k for k in method_keys if _method_card_elements(k) & set(_PRIMARY_DESIGN_ELEMENTS)),
                *request.direct_comparator_literature_keys,
            ]
        )
    )[:12]
    design = _design_selection(request, spec, method_keys=[k for k in method_keys if k in set(primary_keys)][:6])
    exposure_label = _label(spec, request.primary_exposure)
    outcome_label = _label(spec, request.outcome)
    def _summary_for(name: str) -> str:
        descriptor = next((item for item in request.adjustment_candidates if item.name == name), None)
        coding = descriptor.allowed_codings[0] if descriptor is not None else "continuous"
        return "count_percent" if coding in {"binary", "categorical"} else "both"

    objectives = {
        "cohort_accounting": (
            "Account for every input analysis row and record the denominator before the sealed "
            "landmark risk-set gates are applied."
        ),
        "baseline_context": (
            f"Describe the sealed adjustment columns by {exposure_label} with standardized "
            "differences only; the sealed suite recomputes Table 1 on the landmark risk set."
        ),
        "primary_survival_suite": (
            f"Execute the sealed {sealed.landmark_hours:g} h landmark survival suite for "
            f"{exposure_label} and {outcome_label}: risk-set accounting, Table 1, Kaplan-Meier, "
            "adjusted Cox with the Schoenfeld audit and its signed non-proportional-hazards policy."
        ),
        "report": (
            "Produce the zero-patient-row plan report for human review: sources, denominators, "
            "the landmark rule, the sealed model and policy, limitations, and the analysis-only boundary."
        ),
    }
    outline_steps = [
        _outline_step(
            step_id="cohort_accounting", role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            variable_names=[identity, request.outcome], citations=[],
        ),
        _outline_step(
            step_id="baseline_context", role="auxiliary", module_id="table_one",
            objective=objectives["baseline_context"], depends_on=["cohort_accounting"],
            variable_names=[request.primary_exposure, *sealed.adjustment_columns], citations=[],
        ),
        _outline_step(
            step_id="primary_survival_suite", role="primary", module_id="custom_analysis",
            objective=objectives["primary_survival_suite"], depends_on=["cohort_accounting"],
            variable_names=list(sealed.source_columns), citations=primary_keys,
        ),
    ]
    outline_steps.append(
        _outline_step(
            step_id="report", role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=[step.step_id for step in outline_steps],
            variable_names=[identity, *sealed.source_columns], citations=[],
        )
    )
    outline = ProgressivePlanOutline(
        analysis_type="survival",
        cohort_objective=(
            f"Estimate the sealed landmark survival association between {exposure_label} and "
            f"{outcome_label} on the study cohort, keeping the risk-set rule, "
            "missingness, and repeated stays visible for review."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: every executable coordinate is the sealed runtime "
            "authority's; the Planner supplied reader labels and comparator applications only. All "
            "results stay at plan level under the analysis-only claim ceiling."
        ),
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    bound = {step.step_id: step for step in outline.steps}
    steps = [
        ProgressiveSkeletonStep(
            step_id="cohort_accounting", planned_analysis_role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            raw_inputs=[identity, request.outcome], literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="baseline_context", planned_analysis_role="auxiliary", module_id="table_one",
            objective=objectives["baseline_context"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([request.primary_exposure, *sealed.adjustment_columns])),
            table_one_group_by=request.primary_exposure, table_one_mode="descriptive_smd_only",
            table_one_variables=[
                ProgressiveTableOneVariable(name=name, summary=_summary_for(name))
                for name in sealed.adjustment_columns
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="primary_survival_suite", planned_analysis_role="primary", module_id="custom_analysis",
            objective=objectives["primary_survival_suite"], depends_on=["cohort_accounting"],
            # Exactly the sealed source columns and owned products: the host's
            # bind_plan replaces this step with the signed suite and refuses any
            # drift from these coordinates.
            raw_inputs=list(sealed.source_columns),
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role="custom")
                for product in sealed.analysis_outputs
            ],
            custom_method=sealed.primary_owner,
            literature_bindings=_bindings(
                bound["primary_survival_suite"], comparator_applications=spec.applications,
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="report", planned_analysis_role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=bound["report"].depends_on, raw_inputs=[],
            product_inputs=[
                _ref("cohort_accounting", "artifact:analysis_cohort"),
                _ref("cohort_accounting", "table:cohort_flow"),
                _ref("baseline_context", "table:table_one"),
                *(
                    _ref("primary_survival_suite", product)
                    for product in sealed.analysis_outputs
                    if product.startswith("table:")
                ),
            ],
            outputs=[ProgressiveOutputIntent(product_id="report:report", semantic_role="report")],
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
    foundation = ProgressiveFoundationMaterialization(
        outline_sha256=outline_sha256,
        foundation=ProgressivePlanFoundation(
            cohort=_cohort_intent(request),
            display_labels=[
                ProgressiveDisplayLabel(key=key, value=value)
                for key, value in spec.labels.items()
                if key in set(request.variable_roster)
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


__all__ = ["build_landmark_survival_skeleton"]
