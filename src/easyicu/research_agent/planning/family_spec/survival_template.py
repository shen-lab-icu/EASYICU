"""Host template for the fixed-landmark survival suite family.

A time-to-event question whose runtime is already sealed by a
``LandmarkSurvivalRuntimeAuthority`` (risk-set rule, Table 1, Kaplan-Meier,
adjusted Cox, PH audit, PH-free alternatives and the composite figure).  The
Planner decides nothing scientific here: it supplies reader labels for the
sealed columns and the comparator application sentences.  The template
projects a cohort-accounting step and one primary step that names the sealed
owner and copies its source columns and products verbatim; the host's
``bind_plan`` later replaces that primary with the signed suite and its
renderer, so every analysis coordinate is the authority's.

A study with no survival design yet gets the suite the host *could* seal as a
proposal (``request.proposed_suite``): the same layout, but the Planner selects
the adjustment roster, the not-yet-materialized onset column stays out of the
step inputs, and the host keeps the roster with its rationales and timing as
the plan's ``adjustment_proposal``.  Review compiles that design into the study
configuration; nothing runs until the host seals the suite and replans.

Step layout:

1. ``cohort_accounting``        denominators and cohort flow
2. ``baseline_context``         Table 1 by exposure status, SMD only
3. ``measurement_audit``        availability of the suite's source columns
                                (only when the suite publishes its own audit)
4. ``primary_survival_suite``   the sealed suite owner (primary)
5. ``report``                   zero-patient-row plan report

The Table 1, measurement-audit and cohort-flow drafts satisfy the outline's
article-role owners; after ``bind_plan`` the sealed suite carries all three
inside its own step.
"""

from __future__ import annotations

from typing import Callable

from ...canonical_json import canonical_sha256
from ...contracts.model_terms import AdjustmentProposal
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
    SealedSuiteCoordinates,
)
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _cohort_intent,
    _label,
    _method_card_elements,
    _method_card_ids,
    stated_population_sentence,
)
from .plan_language import listing, plan_language, sentence

#: The suite's own data-quality product.  A suite that publishes it audits
#: its source columns itself, so the outline names a measurement audit that
#: binding replaces with this product.
SUITE_MEASUREMENT_AUDIT = "table:landmark_measurement_audit"
_AUDIT_OUTPUTS = (
    ("table:measurement_missingness", "measurement_missingness"),
    ("table:measurement_process_audit", "measurement_process"),
)
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
        elements = sorted(desired & _method_card_elements(key, time_to_event=True))
        if not elements:
            continue
        bindings.append(
            ProgressiveLiteratureBinding(
                citation_key=key,
                design_elements=elements,
                application=(
                    f"Apply the host-curated method card(s) {_method_card_ids(key, desired, time_to_event=True)} to the "
                    "sealed landmark survival suite; the family template binds only the curated "
                    "design elements and retains the analysis-only claim ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _suite(request: FamilySpecRequest) -> SealedSuiteCoordinates:
    suite = request.sealed_suite or request.proposed_suite
    assert suite is not None
    return suite


def _roster(request: FamilySpecRequest, spec: FamilyPlanSpec) -> list[str]:
    """The sealed roster, the user's exact roster, or the Planner's selection."""

    if request.sealed_suite is not None:
        return list(request.sealed_suite.adjustment_columns)
    if request.adjustment_selection == "exact":
        return list(request.exact_roster)
    return [item.name for item in spec.adjustment_set]


def _bound_columns(request: FamilySpecRequest, roster: list[str]) -> list[str]:
    """Columns the primary step binds; a proposal's onset is not materialized yet."""

    suite = _suite(request)
    if request.sealed_suite is not None:
        return list(suite.source_columns)
    return list(
        dict.fromkeys(
            [suite.exposure_status_column, suite.event_column, suite.followup_time_column, *roster]
        )
    )


def _adjustment_proposal(
    request: FamilySpecRequest, spec: FamilyPlanSpec, roster: list[str]
) -> AdjustmentProposal:
    """Keep the proposal's roster with its rationale and host-proven timing."""

    if request.adjustment_selection == "exact":
        rationales = dict(request.exact_rationales)
        roles = dict(request.exact_temporal_roles)
    else:
        rationales = {item.name: item.clinical_rationale for item in spec.adjustment_set}
        roles = {}
        for name in roster:
            candidate = request.candidate(name)
            role = candidate.host_temporal_role if candidate is not None else None
            if role is not None:
                roles[name] = role
    return AdjustmentProposal(
        source_step_id="primary_survival_suite",
        source_requirement_id="proposed_landmark_survival_suite",
        covariates=roster,
        covariate_rationales={name: rationales[name] for name in roster if name in rationales},
        covariate_temporal_roles={name: roles[name] for name in roster if name in roles},
    )


def _prevalence_sensitivity_sentences(hours: list[float], *, present: bool) -> tuple[str, str]:
    """The reviewable words for the suite's prevalence-definition sensitivity analysis."""

    if not hours:
        return "", ""
    named = [f"hour {hour:g}" for hour in hours]
    listed = named[0] if len(named) == 1 else ", ".join(named[:-1]) + f" or, separately, {named[-1]}"
    analyses = (
        "A prespecified sensitivity analysis of the prevalence definition also excludes"
        if len(hours) == 1
        else "Prespecified sensitivity analyses of the prevalence definition also exclude"
    )
    repeats = "repeats" if len(hours) == 1 else "repeat"
    english = (
        f" {analyses} exposure first recorded{' as present' if present else ''} at or before "
        f"{listed} and {repeats} the reported estimate; a descriptive table shows the exposed "
        "group's first-record hours."
    )
    named_zh = [f"第 {hour:g} 小时" for hour in hours]
    listed_zh = named_zh[0] if len(named_zh) == 1 else (
        "、".join(named_zh[:-1]) + f"或（另一项分析中）{named_zh[-1]}"
    )
    chinese = (
        f"预先设定的现患定义敏感性分析另外排除首次{'阳性' if present else ''}记录在{listed_zh}"
        "及以前的暴露，并重复报告的估计；描述性表格给出暴露组首次记录的小时分布。"
    )
    return english, chinese


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    method_keys: list[str],
    roster: list[str] | None = None,
) -> ResearchDesignSelection:
    roster = _roster(request, spec) if roster is None else roster
    sealed = _suite(request)
    proposed = request.proposed_suite is not None
    # A suite signed before the onset representation timed the exposure by its
    # first record of any value, and its plan keeps those words.
    present = sealed.exposure_onset_representation == "first_truthy_event_time"
    sensitivity_en, sensitivity_zh = _prevalence_sensitivity_sentences(
        sealed.prevalence_sensitivity_cutoffs_hours or [], present=present
    )
    exposure = _label(spec, request.primary_exposure)
    outcome = _label(spec, request.outcome)
    adjustment_text = ", ".join(_label(spec, name) for name in roster) or "no covariates"
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
    adjustment_zh = listing([_label(spec, name) for name in roster], language) or "无协变量"
    columns = _bound_columns(request, roster)
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
            f"Exposure status is fixed at the {landmark} landmark from the exposure's "
            + ("first records as present" if present else "first recorded times")
            + f"; the endpoint is followed to {horizon} with administrative censoring."
        ),
        primary_method=(
            (
                "Proposed fixed-landmark survival suite, sealed by the host once review "
                "compiles this design"
                if proposed
                else "Sealed fixed-landmark survival suite"
            )
            + ": risk-set accounting, Table 1, Kaplan-Meier, adjusted Cox with a Schoenfeld "
            "audit, and a signed non-PH policy (interval-specific Cox or an unadjusted RMST "
            "contrast), rendered as one composite figure."
        ),
        required_variables=[request.identity_column, *columns],
        assumptions=[
            # The onset column is the first record (as present), not a verified onset.
            (
                "Exposure timing is the first time the exposure source recorded the exposure as "
                "present; exposure that began before that record, such as before ICU admission, is "
                "not observed, so some prevalent exposure may count as incident."
                if present
                else "Exposure timing is the first recorded time of the exposure source; exposure "
                "that began before its first record, such as before ICU admission, is not observed, "
                "so some prevalent exposure may count as incident."
            ),
            f"{unit_text[0].upper()}{unit_text[1:]}.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=list(spec.literature_design_decisions),
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
            f"No causal effect of {exposure}, no immortal-time-free estimate outside the landmark "
            "rule, and no independence of repeated ICU stays beyond the declared grouping."
        ),
        reviewable_plan=(
            [
                f"研究队列；{unit_text_zh}；纳入在 {landmark_zh} landmark 时存活且终点有效的入住。"
                + stated_population_sentence(spec.population, language),
                (
                    f"截至 {landmark_zh} 新发的 {exposure}，以暴露首次记录为阳性的时间计；首次阳性记录在"
                    "时间零点及以前的暴露按已存在暴露排除。"
                    if present
                    else f"截至 {landmark_zh} 新发的 {exposure}，以暴露首次记录时间计；首次记录在时间零点"
                    "及以前的暴露按已存在暴露排除。"
                ),
                f"自 ICU 入院起 {horizon_zh} 内的 {outcome}，按行政截尾处理。",
                f"调整 {adjustment_zh} 的 Cox 比例风险模型；Wald 95% CI；按封印的处理政策做 "
                "Schoenfeld 残差审计。",
                "校正模型对封印的列做完整病例分析，并审计分母；Kaplan-Meier 曲线和限制平均生存时间使用"
                "完整风险集。",
                "比例风险假设被拒绝时，以预先设定的分段 Cox 模型和未调整的限制平均生存时间对比，"
                "替代恒定的风险比。" + sensitivity_zh,
            ]
            if language == "zh"
            else [
                f"The study cohort; {unit_text}; stays alive at the {landmark} landmark with a valid "
                "endpoint."
                + stated_population_sentence(spec.population, language),
                f"Incident {exposure} by {landmark}, timed by its first record"
                + (" as present; exposure first recorded as present " if present
                   else "; exposure first recorded ")
                + "at or before time zero is excluded as prevalent.",
                sentence(f"{outcome} through {horizon} from ICU admission, censored administratively."),
                f"Adjusted Cox proportional-hazards model for {adjustment_text}; Wald 95% CI; Schoenfeld "
                "residual audit with the sealed handling policy.",
                "Complete-case adjusted models on the sealed columns with an audited denominator; "
                "Kaplan-Meier and the restricted mean use the whole risk set.",
                "Prespecified interval-specific Cox and an unadjusted restricted-mean survival-time "
                "contrast replace a constant hazard ratio when proportional hazards is rejected."
                + sensitivity_en,
            ]
        ),
        disposition="selected",
        decision_reason=(
            "The question asks for a time-respecting survival association; the host's landmark "
            "survival suite fixes exposure status before follow-up starts and audits its own "
            "assumptions; it runs only after review compiles this design and the host seals it."
            if proposed
            else "The question asks for a time-respecting survival association; the sealed "
            "landmark suite fixes exposure status before follow-up starts and audits its own "
            "assumptions, chosen before any data are read."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="binary_logistic_at_horizon",
        analysis_type="survival",
        estimand=f"An adjusted odds ratio for {outcome} by {horizon} ignoring follow-up time.",
        time_zero="ICU admission.",
        observation_window=f"Status at {horizon} only.",
        primary_method="Logistic regression on the horizon status.",
        required_variables=[request.identity_column, *columns],
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

    if request.family_id != LANDMARK_SURVIVAL_FAMILY_ID or (
        request.sealed_suite is None and request.proposed_suite is None
    ):
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
    sealed = _suite(request)
    proposed = request.proposed_suite is not None
    roster = _roster(request, spec)
    columns = _bound_columns(request, roster)
    identity = request.identity_column
    audited = SUITE_MEASUREMENT_AUDIT in sealed.analysis_outputs
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

    objectives = {
        "cohort_accounting": (
            "Account for every input analysis row and record the denominator before the sealed "
            "landmark risk-set gates are applied."
        ),
        "baseline_context": (
            f"Describe the {'selected' if proposed else 'sealed'} adjustment columns by "
            f"{exposure_label} with standardized differences only; the suite recomputes "
            "Table 1 on the landmark risk set."
        ),
        "measurement_audit": (
            "Audit the availability of every source column of the "
            f"{'proposed' if proposed else 'sealed'} suite before the landmark risk set; "
            f"the signed suite publishes this audit as {SUITE_MEASUREMENT_AUDIT}."
        ),
        "primary_survival_suite": (
            f"Execute the {'proposed' if proposed else 'sealed'} {sealed.landmark_hours:g} h "
            f"landmark survival suite for {exposure_label} and {outcome_label}: risk-set "
            "accounting, Table 1, Kaplan-Meier, adjusted Cox with the Schoenfeld audit and its "
            "signed non-proportional-hazards policy"
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
            variable_names=[request.primary_exposure, *roster], citations=[],
        ),
        *(
            [_outline_step(
                step_id="measurement_audit", role="auxiliary", module_id="measurement_audit",
                objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
                variable_names=[identity, *columns], citations=[],
            )]
            if audited
            else []
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
            f"Estimate the sealed landmark survival association between {exposure_label} and "
            f"{outcome_label} on the study cohort, keeping the risk-set rule, "
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
            raw_inputs=list(dict.fromkeys([request.primary_exposure, *roster])),
            table_one_group_by=request.primary_exposure, table_one_mode="descriptive_smd_only",
            table_one_variables=[
                ProgressiveTableOneVariable(name=name, summary=_summary_for(name))
                for name in roster
            ],
            literature_bindings=[],
        ),
        *(
            [ProgressiveSkeletonStep(
                step_id="measurement_audit", planned_analysis_role="auxiliary", module_id="measurement_audit",
                objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
                raw_inputs=list(dict.fromkeys([identity, *columns])),
                product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
                outputs=[
                    ProgressiveOutputIntent(product_id=product, semantic_role=role)
                    for product, role in _AUDIT_OUTPUTS
                ],
                literature_bindings=[],
            )]
            if audited
            else []
        ),
        ProgressiveSkeletonStep(
            step_id="primary_survival_suite", planned_analysis_role="primary", module_id="custom_analysis",
            objective=objectives["primary_survival_suite"], depends_on=["cohort_accounting"],
            # Exactly the sealed source columns and owned products: the host's
            # bind_plan replaces this step with the signed suite and refuses any
            # drift from these coordinates.
            raw_inputs=columns,
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
        adjustment_proposal=_adjustment_proposal(request, spec, roster) if proposed else None,
    )


__all__ = ["build_landmark_survival_skeleton"]
