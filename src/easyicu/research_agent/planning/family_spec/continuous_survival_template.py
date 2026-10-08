"""Host template for the continuous-exposure landmark survival suite family.

A time-to-event question whose exposure is continuous -- a laboratory value or
a vital sign summarised over the window from ICU admission to the landmark --
is planned on the continuous survival suite: the risk set alive at the
landmark, Table 1 and Kaplan-Meier curves by exposure tertile, the Cox model
per readable step of the exposure with its Schoenfeld audit, a prespecified
interval model and a restricted cubic spline check of the linear term, whose
percentile contrasts replace the per-step estimate when it rejects linearity
while the PH test holds.  The suite is sealed by a
``LandmarkContinuousSurvivalRuntimeAuthority``; before
that, the host proposes it (``request.proposed_continuous_suite``) and the
Planner selects the adjustment roster, which the host keeps with its
rationales and timing as the plan's ``adjustment_proposal``.  Review compiles
the design into the study configuration; nothing runs until the host seals
the suite and replans.

Step layout (the binary suite's, see ``survival_template``):

1. ``cohort_accounting``        denominators and cohort flow
2. ``baseline_context``         Table 1 by the outcome, describing the exposure
3. ``measurement_audit``        availability of the suite's source columns
4. ``primary_survival_suite``   the sealed suite owner (primary)
5. ``report``                   zero-patient-row plan report

A continuous exposure has no levels to group the draft Table 1 by, so it is
grouped by the binary outcome, as in the spline family; after ``bind_plan``
the sealed suite describes its landmark risk set by exposure tertile inside
its own step.
"""

from __future__ import annotations

from typing import Callable

from ...canonical_json import canonical_sha256
from ..design_selection import ResearchDesignCandidate, ResearchDesignSelection
from ..progressive_contract import (
    ProgressiveDisplayLabel,
    ProgressiveFoundationMaterialization,
    ProgressiveOutputIntent,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
    ProgressiveTableOneVariable,
)
from .contract import (
    LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID,
    FamilyPlanSpec,
    FamilySpecError,
    FamilySpecRequest,
    SealedContinuousSuiteCoordinates,
    design_field_max_length,
)
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _cohort_intent,
    _label,
    _method_card_elements,
    stated_population_sentence,
)
from .plan_language import adjusted_roster_sentence, listing, plan_language, sentence
from .survival_template import (
    _AUDIT_OUTPUTS,
    _PRIMARY_DESIGN_ELEMENTS,
    _adjustment_proposal,
    _bindings,
    _outline_step,
    _ref,
)

#: The suite's own data-quality product, which binding keeps in its step.
SUITE_MEASUREMENT_AUDIT = "table:landmark_continuous_measurement_audit"
_SUMMARY_WORDS = {
    "max": ("highest value", "最高值"),
    "min": ("lowest value", "最低值"),
    "mean": ("mean value", "均值"),
    "first": ("first recorded value", "首次记录值"),
}


def _suite(request: FamilySpecRequest) -> SealedContinuousSuiteCoordinates:
    suite = request.sealed_continuous_suite or request.proposed_continuous_suite
    assert suite is not None
    return suite


def _roster(request: FamilySpecRequest, spec: FamilyPlanSpec) -> list[str]:
    """The sealed roster, the user's exact roster, or the Planner's selection."""

    if request.sealed_continuous_suite is not None:
        return list(request.sealed_continuous_suite.adjustment_columns)
    if request.adjustment_selection == "exact":
        return list(request.exact_roster)
    return [item.name for item in spec.adjustment_set]


def _bound_columns(request: FamilySpecRequest, roster: list[str]) -> list[str]:
    """Columns the primary step binds: the exposure, endpoint, follow-up and roster."""

    suite = _suite(request)
    if request.sealed_continuous_suite is not None:
        return list(suite.source_columns)
    return list(
        dict.fromkeys([suite.exposure_column, suite.event_column, suite.followup_time_column, *roster])
    )


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    method_keys: list[str],
    roster: list[str],
) -> ResearchDesignSelection:
    suite = _suite(request)
    proposed = request.proposed_continuous_suite is not None
    exposure = _label(spec, request.primary_exposure)
    outcome = _label(spec, request.outcome)
    summary_en, summary_zh = _SUMMARY_WORDS[suite.exposure_window_summary]
    unit = suite.exposure_unit or "units"
    increment = f"per step of {exposure} (1, 2 or 5 x 10^n {unit}, within its IQR)"
    increment_zh = (
        f"{exposure}每增加一步（模型人群中暴露四分位距以内最大的 1、2 或 5×10^n "
        f"{suite.exposure_unit or '个单位'}）"
    )
    adjusted = bool(roster)
    labels = [_label(spec, name) for name in roster]
    adjustment_text = ", ".join(labels) or "no covariates"
    landmark = f"{suite.landmark_hours:g} h after ICU admission"
    window_end = f"the {suite.landmark_hours:g} h landmark"
    horizon = f"{suite.endpoint_horizon_days:g} days"
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
    landmark_zh = f"ICU 入院后 {suite.landmark_hours:g} h"
    horizon_zh = f"{suite.endpoint_horizon_days:g} 天"
    adjustment_zh = listing(labels, language) or "无协变量"
    columns = _bound_columns(request, roster)
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    selected = ResearchDesignCandidate(
        design_id="fixed_landmark_continuous_survival_suite",
        analysis_type="survival",
        # The design bounds its estimand: the roster is named there while it
        # fits, and the reviewable plan below names it in full.
        estimand=adjusted_roster_sentence(
            f"The {'adjusted ' if adjusted else ''}hazard ratio for {outcome} through "
            f"{horizon} {increment} (its {summary_en} from ICU admission to "
            f"{window_end}), among stays alive and event-free at the landmark, ",
            labels,
            "; the spline's 10th- and 90th-percentile contrasts replace it if linearity "
            "is rejected; a descriptive prognostic association, not a causal effect.",
            bound=design_field_max_length("estimand"),
            unadjusted="without covariate adjustment",
        ),
        time_zero=f"ICU admission; follow-up starts at the {landmark} landmark.",
        observation_window=(
            f"The exposure is its {summary_en} recorded from ICU admission to {window_end}; "
            f"the endpoint is followed to {horizon} with administrative censoring."
        ),
        primary_method=(
            (
                "Proposed continuous-exposure landmark survival suite, sealed by the host after review"
                if proposed
                else "Sealed continuous-exposure landmark survival suite"
            )
            + f": tertile Table 1 and Kaplan-Meier, {'adjusted ' if adjusted else ''}Cox per "
            "exposure step with a Schoenfeld audit, a prespecified interval model and a spline "
            "check of the linear term, in one composite figure."
        ),
        required_variables=[request.identity_column, *columns],
        assumptions=[
            "Stays without a recorded exposure value by the landmark leave the risk set, so the "
            "estimate describes stays in which the exposure was measured.",
            "A restricted cubic spline checks whether the log hazard is linear in the exposure; "
            "when it rejects linearity at alpha 0.05 while proportional hazards hold, the "
            "spline's hazard ratios at the 10th and 90th percentiles relative to the median "
            "are the estimate.",
            f"{unit_text[0].upper()}{unit_text[1:]}.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=list(spec.literature_design_decisions),
        novelty_positioning=(
            "No novelty is claimed before completion; the design is positioned against each screened "
            "comparator on population, exposure measurement, landmark, and endpoint."
        ),
        figure_role=(
            f"Kaplan-Meier survival by exposure tertile with the {'adjusted ' if adjusted else ''}"
            "hazard-ratio curve across the exposure, risk-set accounting, and "
            "proportional-hazards diagnostics in one composite figure."
        ),
        supports=(
            f"A prespecified landmark survival association between {exposure} and {outcome} per "
            "step of the exposure, or at two of its percentiles when the spline check rejects a "
            "linear term, with its risk-set accounting and assumption audit."
        ),
        cannot_prove=(
            f"No causal effect of {exposure}, no threshold or dose-response shape beyond the "
            "reported spline check, and no independence of repeated ICU stays beyond the declared "
            "grouping."
        ),
        reviewable_plan=(
            [
                f"研究队列；{unit_text_zh}；纳入在 {landmark_zh} landmark 时存活、终点有效且有暴露记录的入住。"
                + stated_population_sentence(spec.population, language),
                f"暴露为 {exposure} 自 ICU 入院至 {landmark_zh} landmark 的{summary_zh}，按连续变量建模。",
                f"自 ICU 入院起 {horizon_zh} 内的 {outcome}，按行政截尾处理。",
                (f"调整 {adjustment_zh} 的" if adjusted else "不调整协变量的")
                + f" Cox 比例风险模型，报告{increment_zh}的风险比；Wald 95% CI；"
                "按封印的处理政策做 Schoenfeld 残差审计。",
                "按暴露三分位描述基线表与 Kaplan-Meier 曲线；Cox 模型对封印的列做完整病例分析，并审计分母。",
                "比例风险假设被拒绝时，以预先设定的分段 Cox 模型给出各随访区间的风险比，替代恒定的风险比；"
                "以限制性立方样条检验线性项：线性在 α=0.05 被拒且比例风险成立时，以样条在第 10、90 "
                "百分位相对中位数的风险比替代每步风险比。",
            ]
            if language == "zh"
            else [
                f"The study cohort; {unit_text}; stays alive at the {landmark} landmark with a valid "
                "endpoint and a recorded exposure value."
                + stated_population_sentence(spec.population, language),
                f"{exposure}: its {summary_en} from ICU admission to {window_end}, "
                "modelled as a continuous variable.",
                sentence(f"{outcome} through {horizon} from ICU admission, censored administratively."),
                (
                    f"Adjusted Cox proportional-hazards model for {adjustment_text}"
                    if adjusted
                    else "Cox proportional-hazards model without covariates"
                )
                + f", reported {increment}; "
                "Wald 95% CI; Schoenfeld residual audit with the sealed handling policy.",
                "Table 1 and Kaplan-Meier curves by exposure tertile; complete-case models on "
                "the sealed columns with an audited denominator.",
                "A prespecified interval-specific Cox model replaces a constant hazard ratio when "
                "proportional hazards is rejected; a restricted cubic spline checks the linear term "
                "and its hazard ratios at the 10th and 90th percentiles relative to the median "
                "replace the per-step estimate when it rejects linearity at alpha 0.05 while "
                "proportional hazards hold.",
            ]
        ),
        disposition="selected",
        decision_reason=(
            "The question asks for a time-respecting survival association with a continuous "
            "exposure; the host's continuous landmark suite fixes the exposure window before "
            "follow-up starts, keeps the exposure continuous and audits its own assumptions; it "
            "runs only after review compiles this design and the host seals it."
            if proposed
            else "The question asks for a time-respecting survival association with a continuous "
            "exposure; the sealed continuous landmark suite fixes the exposure window before "
            "follow-up starts and audits its own assumptions, chosen before any data are read."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="tertile_cox_contrast",
        analysis_type="survival",
        estimand=(
            f"An adjusted hazard ratio for {outcome} through {horizon} comparing the highest "
            f"with the lowest tertile of {exposure}."
        ),
        time_zero=f"ICU admission; follow-up starts at the {landmark} landmark.",
        observation_window=(
            f"The exposure tertile is fixed at the {landmark} landmark; the endpoint is "
            f"followed to {horizon}."
        ),
        primary_method="Adjusted Cox model of the exposure's sample tertiles.",
        required_variables=[request.identity_column, *columns],
        assumptions=["The hazard is constant within each tertile of the exposure."],
        literature_citation_keys=[
            key for key in ("strobe_2007", "record_2015") if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning="Recorded as the rejected alternative for audit; no novelty is claimed.",
        figure_role="A forest plot of the tertile hazard ratios.",
        supports="A contrast between the extreme thirds of the sample's exposure distribution.",
        cannot_prove=(
            "Its cutpoints come from the sample, and it discards the variation within each tertile."
        ),
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because categorising a continuous exposure at sample tertiles discards "
            "information within each group and ties the contrast to this sample's cutpoints; "
            "the suite keeps tertiles for description only."
        ),
    )
    return ResearchDesignSelection(candidates=[selected, rejected])


def build_landmark_continuous_survival_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from the suite."""

    if request.family_id != LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID or (
        request.sealed_continuous_suite is None and request.proposed_continuous_suite is None
    ):
        raise FamilySpecError(
            "family_spec_template_mismatch",
            "the continuous survival template received a request for another family",
            path="family_id",
        )
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "the spec does not bind this request",
            path="request_sha256",
        )
    suite = _suite(request)
    proposed = request.proposed_continuous_suite is not None
    state = "proposed" if proposed else "sealed"
    roster = _roster(request, spec)
    columns = _bound_columns(request, roster)
    identity = request.identity_column
    method_keys = [
        key
        for key in request.allowed_literature_citation_keys
        if _method_card_elements(key, time_to_event=True)
    ]
    primary_keys = list(
        dict.fromkeys(
            [
                *(
                    k
                    for k in method_keys
                    if _method_card_elements(k, time_to_event=True) & set(_PRIMARY_DESIGN_ELEMENTS)
                ),
                *request.direct_comparator_literature_keys,
            ]
        )
    )[:12]
    design = _design_selection(
        request, spec, method_keys=[k for k in method_keys if k in set(primary_keys)][:6], roster=roster,
    )
    exposure_label = _label(spec, request.primary_exposure)
    outcome_label = _label(spec, request.outcome)

    def _summary_for(name: str) -> str:
        descriptor = next((item for item in request.adjustment_candidates if item.name == name), None)
        coding = descriptor.allowed_codings[0] if descriptor is not None else "continuous"
        return "count_percent" if coding in {"binary", "categorical"} else "both"

    table_one_variables = [request.primary_exposure, *roster, request.outcome]
    objectives = {
        "cohort_accounting": (
            "Account for every input analysis row and record the denominator before the sealed "
            "landmark risk-set gates are applied."
        ),
        "baseline_context": (
            f"Describe {exposure_label} and the {'selected' if proposed else 'sealed'} adjustment "
            f"columns by {outcome_label} with standardized differences only; the suite recomputes "
            "Table 1 by exposure tertile on the landmark risk set."
        ),
        "measurement_audit": (
            f"Audit the availability of every source column of the {state} suite before the "
            f"landmark risk set; the signed suite publishes this audit as {SUITE_MEASUREMENT_AUDIT}."
        ),
        "primary_survival_suite": (
            f"Execute the {state} {suite.landmark_hours:g} h landmark survival suite for "
            f"{exposure_label} per exposure step and {outcome_label}: risk-set accounting, Table 1 "
            "and Kaplan-Meier curves by exposure tertile, the Cox model with the Schoenfeld "
            "audit, its interval model and the spline check"
            + ("; it runs once review compiles the design and the host seals it." if proposed else ".")
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
            variable_names=table_one_variables, citations=[],
        ),
        _outline_step(
            step_id="measurement_audit", role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
            variable_names=[identity, *columns], citations=[],
        ),
        _outline_step(
            step_id="primary_survival_suite", role="primary", module_id="custom_analysis",
            objective=objectives["primary_survival_suite"], depends_on=["cohort_accounting"],
            variable_names=columns, citations=primary_keys,
        ),
    ]
    outline_steps.append(
        _outline_step(
            step_id="report", role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=[step.step_id for step in outline_steps],
            variable_names=[identity, *columns], citations=[],
        )
    )
    outline = ProgressivePlanOutline(
        analysis_type="survival",
        cohort_objective=(
            f"Estimate the {state} landmark survival association between {exposure_label} per "
            f"unit and {outcome_label} on the study cohort, keeping the risk-set rule, "
            "missingness, and repeated stays visible for review."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: every coordinate except the adjustment roster "
            "is the host's proposed suite, compiled into the study configuration at review before "
            "the host seals it; the Planner selected the roster and supplied reader labels and "
            "comparator applications. All results stay at plan level under the analysis-only "
            "claim ceiling."
            if proposed
            else f"Family template {request.family_id}: every executable coordinate is the sealed "
            "runtime authority's; the Planner supplied reader labels and comparator applications "
            "only. All results stay at plan level under the analysis-only claim ceiling."
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
            raw_inputs=list(dict.fromkeys(table_one_variables)),
            table_one_group_by=request.outcome, table_one_mode="descriptive_smd_only",
            table_one_variables=[
                ProgressiveTableOneVariable(name=request.primary_exposure, summary="median_iqr"),
                *(ProgressiveTableOneVariable(name=name, summary=_summary_for(name)) for name in roster),
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="measurement_audit", planned_analysis_role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([identity, *columns])),
            product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role=role)
                for product, role in _AUDIT_OUTPUTS
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="primary_survival_suite", planned_analysis_role="primary", module_id="custom_analysis",
            objective=objectives["primary_survival_suite"], depends_on=["cohort_accounting"],
            # Exactly the suite's source columns and owned products: the host's
            # bind_plan replaces this step with the signed suite and refuses any
            # drift from these coordinates.
            raw_inputs=columns,
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role="custom")
                for product in suite.analysis_outputs
            ],
            custom_method=suite.primary_owner,
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
                    for product in suite.analysis_outputs
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
            cohort=_cohort_intent(request, spec.population),
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
    return FamilySkeletonDraft(
        outline=outline,
        foundation=foundation,
        materializations=materializations,
        adjustment_proposal=(
            _adjustment_proposal(
                request, spec, roster,
                requirement_id="proposed_landmark_continuous_survival_suite",
            )
            if proposed
            else None
        ),
    )


__all__ = ["build_landmark_continuous_survival_skeleton"]
