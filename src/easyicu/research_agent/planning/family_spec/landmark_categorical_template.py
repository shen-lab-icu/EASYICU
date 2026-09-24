"""Host template for the fixed-landmark association families.

Given a sealed :class:`FamilySpecRequest` and a validated
:class:`FamilyPlanSpec`, project the complete Progressive v2 skeleton — outline
(with the design selection), foundation, and every step materialization — from
typed facts.  The projection is deterministic and case-neutral: the only
question-specific text it carries is copied from the spec (reader labels,
covariate rationales, comparator applications) or from the StudyContext.  The
ordinary Progressive validators and compiler judge the result; this module
never relaxes them.

Two exposure kinds share the layout. A **categorical** exposure (closed
levels) gets treatment contrasts, an ordered trend when a secondary continuous
outcome exists, and alternate-definition replays. A **continuous** exposure
gets a continuous primary term that the signed landmark spline runtime later
binds to its restricted-cubic-spline estimator, the exposure's companion
aggregates in the measurement audit, and a baseline table grouped by outcome.
Everything else — cohort, audit, absolute risk, robustness replay, covariate
functional forms, figure, report — is identical.

Step layout (mirrors the family's reference workflow):

1. ``cohort_definition``      landmark analysis cohort + cohort flow
2. ``table_one``              baseline by exposure level, SMD only
3. ``measurement_audit``      source / completeness / missingness / process / denominators
4. ``adjusted_association``   primary model, declared contrast vs reference
5. ``absolute_risk_context``  absolute risk by level in the primary population
6. ``robustness_replay``      prespecified alternate exposures, first stay, complete case
7. ``<covariate>_functional_form`` one RCS-vs-linear check per continuous covariate
8. ``ordinal_trend``          typed ordered trend for the binary + continuous outcomes
9. ``visualization`` / ``report``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ...canonical_json import canonical_sha256
from ..design_selection import ResearchDesignCandidate, ResearchDesignSelection
from ..method_literature import METHOD_CARDS
from ..progressive_contract import (
    ProgressiveCohortIntent,
    ProgressiveCohortPredicate,
    ProgressiveDisplayLabel,
    ProgressiveFoundationMaterialization,
    ProgressiveLiteratureBinding,
    ProgressiveModelTermIntent,
    ProgressiveOutlineStep,
    ProgressiveOutputIntent,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressivePredicateValue,
    ProgressiveProductRef,
    ProgressiveRobustnessIntent,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
    ProgressiveTableOneVariable,
)
from .contract import (
    FamilyPlanSpec,
    FamilySpecError,
    FamilySpecRequest,
    SpecCovariateDecision,
    design_field_max_length,
    landmark_design_roster,
)
from .plan_language import bounded_roster, listing, plan_language, sentence

FUNCTIONAL_FORM_METHOD = "restricted_cubic_spline_sensitivity"
FUNCTIONAL_FORM_KNOT_QUANTILES = (0.1, 0.5, 0.9)
ORDINAL_TREND_METHOD = "prespecified_ordinal_trend"
ORDINAL_TREND_ACTION = "association.ordinal_trend"
PRIMARY_ACTION = "association.adjusted_association"
_MEASUREMENT_AUDIT_OUTPUTS = (
    ("table:measurement_audit_source", "measurement_source"),
    ("table:measurement_audit_completeness", "component_completeness"),
    ("table:measurement_audit_missingness", "measurement_missingness"),
    ("table:measurement_audit_process", "measurement_process"),
    ("table:measurement_audit_denominators", "analytic_denominators"),
)
# Design elements each module may bind from a curated method card.  This is the
# same authority the host materializer uses for low-entropy steps.
_MODULE_DESIGN_ELEMENTS: dict[str, tuple[str, ...]] = {
    "adjusted_association": (
        "adjustment",
        "dependence",
        "estimand",
        "exposure",
        "missing_data",
        "outcome",
        "reporting",
        "robustness",
        "time_zero",
    ),
    "absolute_risk_context": ("estimand", "outcome", "reporting"),
    "robustness_replay": ("robustness", "missing_data", "time_zero", "exposure", "estimand"),
    "functional_form": ("adjustment", "robustness", "estimand"),
    "ordinal_trend": ("estimand", "exposure", "outcome", "reporting", "time_zero"),
}


@dataclass(frozen=True)
class FamilySkeletonDraft:
    """Outline, foundation, and step materializations projected by a template."""

    outline: ProgressivePlanOutline
    foundation: ProgressiveFoundationMaterialization
    materializations: tuple[ProgressiveStepMaterialization, ...]


def _label(spec: FamilyPlanSpec, key: str) -> str:
    return spec.labels.get(key) or key


#: Method-card layers that govern a time-to-event estimator.  Only a template
#: that fits one (the sealed survival suite) binds them: attaching the
#: proportional-hazards diagnostics or a restricted-mean contrast to a
#: logistic, descriptive or prediction plan cites a method the plan never
#: applies, however well the card's generic design elements match.
_TIME_TO_EVENT_LAYER_PREFIX = "survival_"


def _applicable_method_cards(citation_key: str, *, time_to_event: bool) -> list:
    return [
        card
        for card in METHOD_CARDS
        if card.source_key == citation_key
        and (time_to_event or not card.layer.startswith(_TIME_TO_EVENT_LAYER_PREFIX))
    ]


def _method_card_elements(citation_key: str, *, time_to_event: bool = False) -> set[str]:
    return {
        element
        for card in _applicable_method_cards(citation_key, time_to_event=time_to_event)
        for element in card.design_elements
    }


def _method_card_ids(
    citation_key: str, elements: set[str], *, time_to_event: bool = False
) -> str:
    return ", ".join(
        card.id
        for card in _applicable_method_cards(citation_key, time_to_event=time_to_event)
        if elements & set(card.design_elements)
    )


def _step_bindings(
    outline_step: ProgressiveOutlineStep,
    *,
    module: str,
    coordinate: str,
    comparator_applications: dict[str, str],
) -> list[ProgressiveLiteratureBinding]:
    """Bind exactly the outline step's sealed citation keys.

    Curated method cards bind the design elements the module may use; a
    screened comparator binds the Planner-authored application.  A key that
    fits neither is left unbound so the ordinary compiler reports it instead
    of the template inventing an application.
    """

    desired = set(_MODULE_DESIGN_ELEMENTS[module])
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
                    f"Apply the host-curated method card(s) {_method_card_ids(key, desired)} "
                    f"to the {coordinate}; the family template binds only the curated "
                    "design elements and retains the run's analysis-only claim ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _covariate_terms(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
) -> list[ProgressiveModelTermIntent]:
    """Project the adjustment roster into typed model-term intents.

    Under an exact user roster the user's rationales travel with the terms; the
    spec cannot add or remove a covariate.  Under Planner selection the spec's
    decisions are used exactly as validated against the sealed candidates.
    """

    decisions: list[SpecCovariateDecision]
    if request.adjustment_selection == "exact":
        by_name = {item.name: item for item in spec.adjustment_set}
        decisions = []
        for name in request.exact_roster:
            candidate = request.candidate(name)
            chosen = by_name.get(name)
            coding = (
                chosen.coding
                if chosen is not None
                else (candidate.allowed_codings[0] if candidate else "continuous")
            )
            reference = (
                chosen.reference_level_index
                if chosen is not None
                else (0 if coding in {"binary", "categorical"} else None)
            )
            rationale = request.exact_rationales.get(name) or (
                chosen.clinical_rationale if chosen is not None else ""
            )
            if len(rationale) < 16:
                rationale = (
                    f"User-reviewed exact adjustment covariate {name} with a "
                    "declared pre-time-zero role."
                )
            decisions.append(
                SpecCovariateDecision(
                    name=name,
                    coding=coding,
                    reference_level_index=reference,
                    clinical_rationale=rationale[:500],
                )
            )
    else:
        decisions = list(spec.adjustment_set)
    return [
        ProgressiveModelTermIntent(
            name=item.name,
            role="covariate",
            coding=item.coding,
            reference_level_index=item.reference_level_index,
            clinical_rationale=item.clinical_rationale,
        )
        for item in decisions
    ]


def _continuous_covariates(terms: list[ProgressiveModelTermIntent]) -> list[str]:
    return [term.name for term in terms if term.coding == "continuous"]


def _table_one_summary(coding: str) -> str:
    return "count_percent" if coding in {"binary", "categorical"} else "both"


def _estimand(head: str, labels: list[str], tail: str) -> str:
    """Name the adjustment roster while the estimand holds it; the plan lists it in full."""

    if not labels:
        return f"{head}unadjusted (no covariate was authorized){tail}"
    prefix = "adjusted for "
    roster = bounded_roster(
        labels, budget=design_field_max_length("estimand") - len(head) - len(prefix) - len(tail)
    )
    if roster:
        return f"{head}{prefix}{roster}{tail}"
    return f"{head}{prefix}{len(labels)} prespecified covariates named in the plan{tail}"


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    covariates: list[str],
    spline_covariates: list[str],
    required_variables: list[str],
    method_keys: list[str],
) -> ResearchDesignSelection:
    language = plan_language(request.research_question)
    exposure = _label(spec, request.primary_exposure)
    outcome = _label(spec, request.outcome)
    continuous_exposure = request.exposure_kind == "continuous"
    reference = request.exposure_levels[request.reference_level_index] if not continuous_exposure else ""
    contrast = request.exposure_levels[request.primary_contrast_level_index] if not continuous_exposure else ""
    hours = f"{request.landmark_hours:g}"
    adjustment_text = (
        "adjusted for " + ", ".join(_label(spec, name) for name in covariates)
        if covariates
        else "unadjusted (no covariate was authorized)"
    )
    cluster_text = (
        "with patient-level cluster-robust inference for repeated ICU stays"
        if request.cluster_unit == "patient"
        else "treating each analysis row as independent"
    )
    cluster_text_zh = (
        "对重复 ICU 入住按患者做聚类稳健推断"
        if request.cluster_unit == "patient"
        else "各分析行视为相互独立"
    )
    secondary_text = (
        f"; {_label(spec, request.secondary_continuous_outcome)} is described by level as a secondary outcome"
        if request.secondary_continuous_outcome
        else ""
    )
    secondary_text_zh = (
        f"；{_label(spec, request.secondary_continuous_outcome)} 按暴露水平描述，作为次要结局"
        if request.secondary_continuous_outcome
        else ""
    )
    duration_label = _label(spec, request.observation_duration_column)
    duration_unit = str(request.observation_duration_unit or "").strip()
    # A reader label usually carries its own unit ("... duration (hours)");
    # repeating it reads "(hours) (hours)".
    duration_text = (
        f"{duration_label} ({duration_unit})"
        if duration_unit and duration_unit.casefold() not in duration_label.casefold()
        else duration_label
    )
    adjustment_text_zh = (
        "调整 " + listing([_label(spec, name) for name in covariates], language)
        if covariates
        else "不调整（未授权任何协变量）"
    )
    # Only the covariates that really get a functional-form step: a spline
    # promised for a binary or categorical covariate would be a check the
    # plan never runs (and could not run).
    sensitivity_bits = [
        *(f"alternate exposure definition {_label(spec, item.execution_variables[0])}" for item in request.alternate_exposures),
        *(["first-ICU-stay restriction"] if request.first_stay else []),
        "complete-case reanalysis",
        *(f"restricted cubic spline for {_label(spec, name)}" for name in spline_covariates),
    ]
    sensitivity_bits_zh = [
        *(f"替代暴露定义 {_label(spec, item.execution_variables[0])}" for item in request.alternate_exposures),
        *(["仅限首次 ICU 入住"] if request.first_stay else []),
        "完整病例重分析",
        *(f"{_label(spec, name)} 的限制性立方样条" for name in spline_covariates),
    ]
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    exposure_clause = (
        f"{exposure} (continuous, modelled as a restricted cubic spline with a per-unit linear sensitivity) measured"
        if continuous_exposure
        else f"{exposure} (levels {', '.join(request.exposure_levels)}; reference {reference}, primary contrast {contrast}) classified"
    )
    selected = ResearchDesignCandidate(
        design_id="landmark_adjusted_association",
        analysis_type="association_study",
        estimand=_estimand(
            f"Observational association between {exposure_clause} from information available "
            f"before a {hours} h landmark and {outcome} after the landmark, among analysis rows alive and "
            "under observation at the landmark, ",
            [_label(spec, name) for name in covariates],
            f", {cluster_text}.",
        ),
        time_zero=(
            f"{hours} h after ICU admission (fixed landmark); the exposure uses only information "
            f"available in the 0–{hours} h window."
        ),
        observation_window=(
            f"From the {hours} h landmark until the end of {duration_text}; rows with "
            f"{_label(spec, request.event_time_column)} at or "
            f"before the landmark or negative event times are counted in the cohort flow, not analysed"
            f"{secondary_text}."
        ),
        primary_method=(
            "Prespecified continuous-exposure logistic association model after the landmark with a "
            "restricted cubic spline for the exposure, a per-unit linear sensitivity, adjusted absolute "
            "risks along the exposure range, and prespecified sensitivity refits."
            if continuous_exposure
            else "Prespecified categorical-exposure logistic association model after the landmark, "
            "with absolute risks by level, a typed ordered trend, and prespecified sensitivity refits."
        ),
        required_variables=required_variables,
        assumptions=[
            "The exposure level is fixed at the landmark and uses no later information.",
            "Repeated ICU stays within a patient are handled by the declared dependence contract.",
            "Estimates are observational associations under the analysis-only claim ceiling.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=[],
        novelty_positioning=(
            "No novelty is claimed before completion; the study is compared with each screened "
            "direct comparator on population, exposure, time zero, estimand, analysis path, and "
            "clinical contribution."
        ),
        figure_role=(
            "The adjusted exposure–risk curve and absolute risk along the exposure range as the reader "
            "entry point, followed by the linear sensitivity, robustness grid, and measurement audit."
            if continuous_exposure
            else "Exposure ascertainment and absolute risk by level as the reader entry point, followed by "
            "the adjusted association, ordered trend, robustness grid, and measurement audit."
        ),
        supports=(
            f"A prespecified adjusted observational association between {exposure} and {outcome} after "
            f"the {hours} h landmark, with its functional form and adjusted absolute risks along the range."
            if continuous_exposure
            else f"A prespecified adjusted observational association and ordered gradient between {exposure} "
            f"and {outcome} after the {hours} h landmark, with level-specific absolute risks."
        ),
        cannot_prove=(
            "No causal effect, no transportability beyond the source population, no recoding of an "
            "unevaluable exposure as the reference level, and no re-derivation of phenotypes from raw data."
        ),
        reviewable_plan=(
            [
                (
                    f"研究队列中满足类型化纳入界限、且在 {hours} h landmark 时仍在观察中的分析行；"
                    f"每行为一次 ICU 入住，{cluster_text_zh}。"
                ),
                (
                    f"{exposure} 取 0–{hours} h 窗口内的测量值；无测量的行作为单独的未测量状态在审计中"
                    "描述，不做数值插补。"
                    if continuous_exposure
                    else f"{exposure} 按 0–{hours} h 窗口分级，水平为 "
                    f"{listing(request.exposure_levels, language)}；无法评估的行保留为单独的未知状态，"
                    "不重编码为参照水平。"
                ),
                (
                    f"{outcome}，自 landmark 起至 {_label(spec, request.observation_duration_column)} "
                    f"结束{secondary_text_zh}。"
                ),
                (
                    f"logistic 模型，{exposure} 以限制性立方样条表示（节点位于第 10/50/90 百分位，以中位数"
                    f"为参照），另做每单位线性敏感性分析；{adjustment_text_zh}。"
                    if continuous_exposure
                    else f"logistic 模型，采用处理对比（{contrast} vs {reference}），{adjustment_text_zh}；"
                    "另按每级增量报告有序线性趋势项。"
                ),
                (
                    "暴露未知的行只做描述、不进入模型；协变量缺失的行从主模型中排除并报告；预先设定完整"
                    "病例重拟合。"
                ),
                (
                    "；".join(dict.fromkeys(sensitivity_bits_zh))
                    + "；每次重拟合都复用主分析内核，解读前先审计分母。"
                ),
            ]
            if language == "zh"
            else [
                (
                    "Analysis rows of the study cohort that meet the typed eligibility bound and survive "
                    f"under observation to the {hours} h landmark; each analysis row is one ICU stay, "
                    f"{cluster_text}."
                ),
                sentence(
                    f"{exposure} measured in the 0–{hours} h window; rows without a measurement stay a "
                    "separate unmeasured state described in the audit and are never imputed as a value."
                    if continuous_exposure
                    else f"{exposure} classified from the 0–{hours} h window with levels "
                    f"{', '.join(request.exposure_levels)}; unevaluable rows stay a separate unknown state "
                    "and are never recoded to the reference level."
                ),
                sentence(
                    f"{outcome} from the landmark to the end of "
                    f"{_label(spec, request.observation_duration_column)}{secondary_text}."
                ),
                (
                    f"Logistic model with {exposure} as a restricted cubic spline (10/50/90 knots, median "
                    f"reference) and a per-unit linear sensitivity, {adjustment_text}."
                    if continuous_exposure
                    else f"Logistic model with treatment contrasts ({contrast} vs {reference}), "
                    f"{adjustment_text}; an ordinal-linear trend term is reported per level increment."
                ),
                (
                    "Unknown exposure rows are described but not modelled; covariate-missing rows are "
                    "excluded from the primary model and reported; a complete-case refit is prespecified."
                ),
                sentence(
                    "; ".join(dict.fromkeys(sensitivity_bits))
                    + "; every refit reuses the primary kernel and denominators are audited before "
                    "interpretation."
                ),
            ]
        ),
        disposition="selected",
        decision_reason=(
            "The landmark aligns exposure ascertainment with the start of follow-up so the measurement "
            "window is not counted as exposed time; it answers the prespecified adjusted dose–response and "
            "lets dependence, missingness, and functional-form sensitivity be planned together."
            if continuous_exposure
            else "The landmark aligns exposure ascertainment with the start of follow-up so the classification "
            "window is not counted as exposed time; it answers the prespecified level contrast and lets "
            "dependence, missingness, functional form, and exposure-definition sensitivity be planned together."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="whole_stay_exposure_association",
        analysis_type="association_study",
        estimand=(
            f"Observational association between the final whole-stay {exposure} "
            f"{'summary' if continuous_exposure else 'classification'} and {outcome} counted from ICU admission."
        ),
        time_zero="ICU admission, before the exposure is fully ascertained.",
        observation_window="From ICU admission to the end of the hospital observation period.",
        primary_method="Adjusted association model with the exposure classified over the whole stay.",
        required_variables=[
            request.identity_column,
            request.primary_exposure,
            request.outcome,
            request.event_time_column,
            request.observation_duration_column,
        ],
        assumptions=[
            "The final exposure classification would have to be knowable without bias at admission.",
        ],
        literature_citation_keys=[
            key for key in ("suissa_immortal_time_2008", "strobe_2007", "record_2015")
            if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning=(
            "Recorded as the alternative time-zero path for audit; no superiority or novelty is claimed."
        ),
        figure_role="Not the primary display; retained as an audit comparison of an unselected design.",
        supports=(
            f"A crude description of the whole-stay {exposure} "
            f"{'summary' if continuous_exposure else 'classification'} against {outcome}."
        ),
        cannot_prove=(
            "It cannot estimate the association fairly before the exposure is determined and is exposed to "
            "immortal-time and time-alignment bias; it does not replace the landmark design."
        ),
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because the exposure forms during the ascertainment window after ICU admission; "
            "starting follow-up at admission mixes the classification window with follow-up and creates "
            "immortal-time risk."
        ),
    )
    return ResearchDesignSelection(candidates=[selected, rejected])


def _predicate(concept_id: str, op: str, value: float, *, end_hours: float) -> ProgressiveCohortPredicate:
    return ProgressiveCohortPredicate(
        concept_id=concept_id,
        anchor="icu_admission",
        start_offset_hours=0.0,
        end_offset_hours=end_hours,
        aggregation="first",
        op=op,
        value=ProgressivePredicateValue(mode="number", number_value=float(value)),
    )


def _cohort_intent(request: FamilySpecRequest) -> ProgressiveCohortIntent:
    if request.cohort_selection_mode == "all_input_rows":
        return ProgressiveCohortIntent(
            name=request.cohort_name, selection_mode="all_input_rows", inclusion=[], exclusion=[]
        )
    end_hours = request.landmark_hours or request.observation_window_hours
    if end_hours is None:
        raise FamilySpecError(
            "family_spec_cohort_window_unavailable",
            "a predicate-filtered cohort needs a typed landmark or observation window",
            path="cohort",
        )
    inclusion: list[ProgressiveCohortPredicate] = []
    if request.age_min is not None:
        inclusion.append(_predicate("age", ">=", request.age_min, end_hours=end_hours))
    if request.age_max is not None:
        inclusion.append(_predicate("age", "<=", request.age_max, end_hours=end_hours))
    if not inclusion:
        raise FamilySpecError(
            "family_spec_cohort_predicate_unavailable",
            "predicate-filtered cohort intent needs a typed age bound",
            path="cohort",
        )
    return ProgressiveCohortIntent(
        name=request.cohort_name,
        selection_mode="predicate_filtered",
        inclusion=inclusion,
        exclusion=[],
    )


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
    population_scope: str | None = None,
) -> ProgressiveOutlineStep:
    kwargs = dict(
        step_id=step_id,
        planned_analysis_role=role,
        module_id=module_id,
        objective=objective,
        depends_on=depends_on,
        variable_names=list(dict.fromkeys(variable_names)),
        literature_citation_keys=list(dict.fromkeys(citations))[:12],
        scientific_action_id=action,
    )
    if population_scope is not None:
        kwargs["population_scope"] = population_scope
    return ProgressiveOutlineStep(**kwargs)


def _materialization(
    outline_step: ProgressiveOutlineStep,
    step: ProgressiveSkeletonStep,
) -> ProgressiveStepMaterialization:
    return ProgressiveStepMaterialization(
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        foundation=None,
        step=step,
    )


def _ref(producer: str, product: str) -> ProgressiveProductRef:
    return ProgressiveProductRef(producer_step_id=producer, product_id=product)


def build_landmark_categorical_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from typed facts.

    ``bind_outline`` lets the caller apply the host's ordinary outline binders
    (runtime action edges, required method sources, direct comparator) before
    the step materializations seal each outline step's digest.
    """

    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "the spec does not bind this request",
            path="request_sha256",
        )
    exposure = request.primary_exposure
    outcome = request.outcome
    identity = request.identity_column
    continuous_exposure = request.exposure_kind == "continuous"
    terms = _covariate_terms(request, spec)
    covariates = [term.name for term in terms]
    continuous = _continuous_covariates(terms)
    alternates = [item.execution_variables[0] for item in request.alternate_exposures]
    companions = list(request.exposure_companion_columns)
    first_stay = request.first_stay.execution_variables[0] if request.first_stay else None
    secondary = request.secondary_continuous_outcome
    landmark_columns = [request.event_time_column, request.observation_duration_column]
    method_keys = [
        key
        for key in request.allowed_literature_citation_keys
        if _method_card_elements(key)
    ]
    primary_keys = list(
        dict.fromkeys(
            [
                *(k for k in method_keys if _method_card_elements(k) & set(_MODULE_DESIGN_ELEMENTS["adjusted_association"])),
                *request.direct_comparator_literature_keys,
            ]
        )
    )[:12]
    absolute_keys = [k for k in method_keys if _method_card_elements(k) & set(_MODULE_DESIGN_ELEMENTS["absolute_risk_context"])][:12]
    robustness_keys = [k for k in method_keys if _method_card_elements(k) & set(_MODULE_DESIGN_ELEMENTS["robustness_replay"])][:12]
    functional_keys = [k for k in method_keys if _method_card_elements(k) & set(_MODULE_DESIGN_ELEMENTS["functional_form"])][:12]
    ordinal_keys = list(
        dict.fromkeys(
            [
                *(k for k in method_keys if _method_card_elements(k) & set(_MODULE_DESIGN_ELEMENTS["ordinal_trend"])),
                *request.direct_comparator_literature_keys,
            ]
        )
    )[:12]

    cohort_variables = [
        identity,
        exposure,
        outcome,
        *landmark_columns,
        *covariates,
        *([secondary] if secondary else []),
        *([first_stay] if first_stay else []),
        *request.measurement_audit_columns,
    ]
    # A continuous exposure has no levels to group by: the baseline table is
    # grouped by the binary outcome and describes the exposure as a variable.
    table_one_group = outcome if continuous_exposure else exposure
    table_one_variables = (
        [exposure, *covariates, outcome] if continuous_exposure else [exposure, *covariates]
    )
    audit_variables = [
        exposure,
        *alternates,
        *companions,
        *request.measurement_audit_columns,
        *([first_stay] if first_stay else []),
    ]
    primary_variables = [identity, exposure, outcome, *landmark_columns, *covariates]
    absolute_variables = [exposure, outcome, *([secondary] if secondary else []), *landmark_columns]
    robustness_variables = [exposure, *alternates, *([first_stay] if first_stay else []), outcome, *covariates, identity]
    visualization_variables = [exposure, outcome, *([secondary] if secondary else []), *covariates, *request.measurement_audit_columns]
    report_variables = [identity, exposure, outcome, *([secondary] if secondary else []), *covariates]

    required_variables = landmark_design_roster(request, covariates)
    design_limit = design_field_max_length("required_variables")
    if len(required_variables) > design_limit:
        # The spec contract refuses this roster before a template is built;
        # an exact roster reaching here must fail, never lose variables.
        raise FamilySpecError(
            "family_spec_roster_exceeds_design",
            f"the design names at most {design_limit} variables; this roster needs "
            f"{len(required_variables)}",
            path="adjustment_set",
        )
    design = _design_selection(
        request,
        spec,
        covariates=covariates,
        spline_covariates=continuous,
        required_variables=required_variables,
        method_keys=[k for k in method_keys if k in set(primary_keys)][:6],
    )
    hours = f"{request.landmark_hours:g}"
    exposure_label = _label(spec, exposure)
    outcome_label = _label(spec, outcome)

    applications = spec.applications
    reference = request.exposure_levels[request.reference_level_index] if not continuous_exposure else ""
    contrast = request.exposure_levels[request.primary_contrast_level_index] if not continuous_exposure else ""
    ordinal_present = bool(
        not continuous_exposure
        and secondary
        and request.exposure_is_ordered
        and len(request.exposure_levels) >= 3
    )
    functional_step_ids = [f"{name}_functional_form" for name in continuous]

    # Pass 1: the outline (scientific intent only).  Objectives are shared with
    # the materializations below so the compiler sees one statement per step.
    objectives = {
        "cohort_definition": (
            f"Build the {hours} h landmark analysis cohort with the typed eligibility bound and record "
            "every selection stage (source rows, non-negative event time, alive and observed at the "
            "landmark, exposure evaluable) with counts."
        ),
        "table_one": (
            f"Describe baseline characteristics of the analysis cohort overall and by {outcome_label} "
            "with standardized differences only; repeated units carry no independent tests."
            if continuous_exposure
            else f"Describe baseline characteristics of the analysis cohort overall and by {exposure_label} "
            "level with standardized differences only; repeated units carry no independent tests."
        ),
        "measurement_audit": (
            f"Audit the source, component completeness, missingness, measurement process, and analytic "
            f"denominators of {exposure_label} and its companion measurement fields before any model is read."
            if continuous_exposure
            else f"Audit the source, component completeness, missingness, measurement process, and analytic "
            f"denominators of {exposure_label} and its alternate definitions before any model is read."
        ),
        "adjusted_association": (
            f"Estimate the prespecified adjusted association between {exposure_label} and {outcome_label} "
            f"after the {hours} h landmark with the exposure as a restricted cubic spline, a per-unit "
            "linear sensitivity, and the declared dependence contract."
            if continuous_exposure
            else f"Estimate the prespecified adjusted association between {exposure_label} and {outcome_label} "
            f"after the {hours} h landmark with treatment contrasts (primary contrast {contrast} vs "
            f"{reference}) and the declared dependence contract."
        ),
        "absolute_risk_context": (
            f"Report the absolute {outcome_label} risk across the {exposure_label} range in the primary "
            "model's comparable landmark population as context for the relative estimates."
            if continuous_exposure
            else f"Report the absolute {outcome_label} risk by {exposure_label} level in the primary model's "
            "comparable landmark population as context for the relative estimates."
        ),
        "robustness_replay": (
            "Replay the prespecified sensitivity axes (first-ICU-stay restriction, complete-case "
            "reanalysis, landmark timing, covariate functional form) against the primary association and "
            "report comparability of estimand, denominators, and direction."
            if continuous_exposure
            else "Replay the prespecified sensitivity axes (alternate exposure definitions, first-ICU-stay "
            "restriction, complete-case reanalysis, landmark timing) against the primary association and "
            "report comparability of estimand, denominators, and direction."
        ),
        "ordinal_trend": (
            f"Estimate the ordered trend of {outcome_label} and describe "
            f"{_label(spec, secondary) if secondary else ''} across {exposure_label} levels with per-level "
            "total and evaluable denominators, median and IQR, prespecified ordered trend tests, and "
            "explicit handling of deaths before and after the landmark."
        ),
        "visualization": (
            "Assemble the complementary main and supplementary displays: the adjusted exposure–risk curve "
            "and absolute risk, the linear sensitivity, the robustness summary, and the measurement audit."
            if continuous_exposure
            else "Assemble the complementary main and supplementary displays: exposure ascertainment and absolute "
            "risk, the primary adjusted association, the robustness summary, and the measurement audit."
        ),
        "report": (
            "Produce the zero-patient-row plan report for human review: sources, selection stages, "
            "denominators, missingness, sensitivity coverage, limitations, and the analysis-only "
            "interpretation boundary."
        ),
    }
    for name in continuous:
        objectives[f"{name}_functional_form"] = (
            f"Refit the primary model with {_label(spec, name)} as a restricted cubic spline instead of a "
            "linear term and compare the primary contrast with the linear form; the primary specification "
            "is retained regardless of the result."
        )

    outline_steps: list[ProgressiveOutlineStep] = [
        _outline_step(
            step_id="cohort_definition", role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_definition"], depends_on=[],
            variable_names=cohort_variables, citations=[],
        ),
        _outline_step(
            step_id="table_one", role="auxiliary", module_id="table_one",
            objective=objectives["table_one"], depends_on=["cohort_definition"],
            variable_names=table_one_variables, citations=[],
        ),
        _outline_step(
            step_id="measurement_audit", role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_definition"],
            variable_names=audit_variables, citations=[],
        ),
        _outline_step(
            step_id="adjusted_association", role="primary", module_id="adjusted_association",
            objective=objectives["adjusted_association"],
            depends_on=["cohort_definition", "measurement_audit"],
            variable_names=primary_variables, citations=primary_keys, action=PRIMARY_ACTION,
        ),
        _outline_step(
            step_id="absolute_risk_context", role="secondary", module_id="absolute_risk_context",
            objective=objectives["absolute_risk_context"], depends_on=["adjusted_association"],
            variable_names=absolute_variables, citations=absolute_keys,
            population_scope="primary_model",
        ),
        _outline_step(
            step_id="robustness_replay", role="sensitivity", module_id="robustness_replay",
            objective=objectives["robustness_replay"], depends_on=["adjusted_association"],
            variable_names=robustness_variables, citations=robustness_keys,
        ),
        *(
            _outline_step(
                step_id=f"{name}_functional_form", role="sensitivity", module_id="custom_analysis",
                objective=objectives[f"{name}_functional_form"], depends_on=["adjusted_association"],
                variable_names=[name], citations=functional_keys,
            )
            for name in continuous
        ),
        *(
            [
                _outline_step(
                    step_id="ordinal_trend", role="secondary", module_id="custom_analysis",
                    objective=objectives["ordinal_trend"], depends_on=["adjusted_association"],
                    variable_names=[exposure, outcome, secondary], citations=ordinal_keys,
                    action=ORDINAL_TREND_ACTION,
                )
            ]
            if ordinal_present
            else []
        ),
    ]
    visualization_deps = [
        "measurement_audit", "adjusted_association", "absolute_risk_context", "robustness_replay",
    ]
    outline_steps.append(
        _outline_step(
            step_id="visualization", role="auxiliary", module_id="visualization",
            objective=objectives["visualization"], depends_on=visualization_deps,
            variable_names=visualization_variables, citations=[],
        )
    )
    outline_steps.append(
        _outline_step(
            step_id="report", role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=[step.step_id for step in outline_steps],
            variable_names=report_variables, citations=[],
        )
    )
    outline = ProgressivePlanOutline(
        analysis_type="association_study",
        cohort_objective=(
            f"Estimate the adjusted observational dose–response between {exposure_label} "
            f"(0–{hours} h) and {outcome_label} after the {hours} h landmark in the study cohort, "
            "keeping unmeasured exposure, repeated stays, missingness, and measurement limits visible for review."
            if continuous_exposure
            else f"Estimate the observational association and ordered gradient between {exposure_label} "
            f"(0–{hours} h) and {outcome_label} after the {hours} h landmark in the study cohort, "
            "keeping unknown exposure, repeated stays, missingness, and measurement limits visible for review."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: the host projected every executable "
            "coordinate from typed StudyContext facts; the Planner supplied the adjustment roster with "
            "rationales, reader labels, and comparator applications. All results stay at plan level "
            "with the analysis-only claim ceiling."
        ),
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    bound = {step.step_id: step for step in outline.steps}

    # Pass 2: executable detail for each sealed outline step.
    def bindings(step_id: str, module: str, coordinate: str) -> list[ProgressiveLiteratureBinding]:
        return _step_bindings(
            bound[step_id], module=module, coordinate=coordinate,
            comparator_applications=applications,
        )

    replay_ids = [
        *(item.spec_id for item in request.alternate_exposures),
        *([request.first_stay.spec_id] if request.first_stay else []),
        request.complete_case_spec_id or "complete_case_primary_covariates",
        request.landmark_spec_id,
    ]
    steps: list[ProgressiveSkeletonStep] = [
        ProgressiveSkeletonStep(
            step_id="cohort_definition", planned_analysis_role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_definition"], depends_on=[],
            raw_inputs=list(dict.fromkeys(cohort_variables)), literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="table_one", planned_analysis_role="auxiliary", module_id="table_one",
            objective=objectives["table_one"], depends_on=["cohort_definition"],
            raw_inputs=list(dict.fromkeys(table_one_variables)),
            table_one_group_by=table_one_group, table_one_mode="descriptive_smd_only",
            table_one_variables=[
                *(
                    [ProgressiveTableOneVariable(name=exposure, summary="median_iqr")]
                    if continuous_exposure
                    else []
                ),
                *(
                    ProgressiveTableOneVariable(name=term.name, summary=_table_one_summary(term.coding))
                    for term in terms
                ),
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="measurement_audit", planned_analysis_role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_definition"],
            raw_inputs=list(dict.fromkeys(audit_variables)),
            product_inputs=[_ref("cohort_definition", "artifact:analysis_cohort")],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role=role)
                for product, role in _MEASUREMENT_AUDIT_OUTPUTS
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="adjusted_association", planned_analysis_role="primary", module_id="adjusted_association",
            objective=objectives["adjusted_association"],
            depends_on=["cohort_definition", "measurement_audit"],
            raw_inputs=list(dict.fromkeys(primary_variables)),
            product_inputs=[
                _ref("cohort_definition", "artifact:analysis_cohort"),
                _ref("measurement_audit", "table:measurement_audit_denominators"),
            ],
            scientific_action_id=PRIMARY_ACTION,
            primary_exposure=exposure, outcome=outcome, outcome_type="binary",
            model_terms=[
                ProgressiveModelTermIntent(
                    name=exposure, role="exposure",
                    coding=(
                        "continuous"
                        if continuous_exposure
                        else "categorical" if len(request.exposure_levels) > 2 else "binary"
                    ),
                    reference_level_index=None if continuous_exposure else request.reference_level_index,
                ),
                *terms,
            ],
            event_level_index=request.event_level_index,
            reference_exposure_level_index=None if continuous_exposure else request.reference_level_index,
            comparison_exposure_level_index=(
                None if continuous_exposure else request.primary_contrast_level_index
            ),
            primary_contrast_level_index=(
                None if continuous_exposure else request.primary_contrast_level_index
            ),
            confidence_level=0.95,
            # The signed spline runtime owns the landmark obligation through the
            # primary fit itself; declaring the spec here keeps plan shaping from
            # projecting a duplicate analysis-only landmark step.
            sensitivity_spec_ids=[request.landmark_spec_id] if continuous_exposure else [],
            literature_bindings=bindings(
                "adjusted_association", "adjusted_association", "primary adjusted association"
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="absolute_risk_context", planned_analysis_role="secondary",
            module_id="absolute_risk_context", objective=objectives["absolute_risk_context"],
            depends_on=["adjusted_association"],
            raw_inputs=list(dict.fromkeys([exposure, outcome, *([secondary] if secondary else [])])),
            product_inputs=[_ref("adjusted_association", "table:adjusted_association_estimates")],
            primary_exposure=exposure, outcome=outcome, population_scope="primary_model",
            literature_bindings=bindings(
                "absolute_risk_context", "absolute_risk_context", "absolute-risk context"
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="robustness_replay", planned_analysis_role="sensitivity", module_id="robustness_replay",
            objective=objectives["robustness_replay"], depends_on=["adjusted_association"],
            raw_inputs=list(dict.fromkeys(robustness_variables)),
            product_inputs=[_ref("adjusted_association", "table:adjusted_association_estimates")],
            sensitivity_spec_ids=list(dict.fromkeys(replay_ids)),
            literature_bindings=bindings(
                "robustness_replay", "robustness_replay", "prespecified robustness replay"
            ),
        ),
    ]
    for name in continuous:
        step_id = f"{name}_functional_form"
        steps.append(
            ProgressiveSkeletonStep(
                step_id=step_id, planned_analysis_role="sensitivity", module_id="custom_analysis",
                objective=objectives[step_id], depends_on=["adjusted_association"],
                raw_inputs=[name],
                product_inputs=[_ref("adjusted_association", "table:adjusted_association_estimates")],
                outputs=[
                    ProgressiveOutputIntent(
                        product_id=f"table:{name}_functional_form_sensitivity",
                        semantic_role="scientific_sensitivity",
                    )
                ],
                custom_method=FUNCTIONAL_FORM_METHOD,
                sensitivity_spec_ids=[
                    request.functional_form_spec_ids.get(name, f"{name}_restricted_cubic_spline")
                ],
                functional_form_spec={
                    "target_column": name,
                    "comparison": "restricted_cubic_spline_vs_linear",
                    "knot_quantiles": list(FUNCTIONAL_FORM_KNOT_QUANTILES),
                },
                literature_bindings=bindings(
                    step_id, "functional_form", f"{name} functional-form sensitivity"
                ),
            )
        )
    if ordinal_present:
        steps.append(
            ProgressiveSkeletonStep(
                step_id="ordinal_trend", planned_analysis_role="secondary", module_id="custom_analysis",
                objective=objectives["ordinal_trend"], depends_on=["adjusted_association"],
                raw_inputs=[exposure, outcome, secondary],
                product_inputs=[_ref("adjusted_association", "table:adjusted_association_estimates")],
                outputs=[ProgressiveOutputIntent(product_id="table:ordinal_trend", semantic_role="custom")],
                scientific_action_id=ORDINAL_TREND_ACTION, custom_method=ORDINAL_TREND_METHOD,
                literature_bindings=bindings("ordinal_trend", "ordinal_trend", "typed ordered trend"),
            )
        )
    steps.append(
        ProgressiveSkeletonStep(
            step_id="visualization", planned_analysis_role="auxiliary", module_id="visualization",
            objective=objectives["visualization"], depends_on=visualization_deps, raw_inputs=[],
            product_inputs=[
                _ref("absolute_risk_context", "table:absolute_risk_context"),
                _ref("adjusted_association", "table:adjusted_association_estimates"),
                _ref("robustness_replay", "table:robustness_matrix"),
                _ref("robustness_replay", "table:robustness_summary"),
            ],
            outputs=[ProgressiveOutputIntent(product_id="figure:visualization", semantic_role="figure")],
            literature_bindings=[],
        )
    )
    steps.append(
        ProgressiveSkeletonStep(
            step_id="report", planned_analysis_role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=bound["report"].depends_on, raw_inputs=[],
            product_inputs=[
                _ref("cohort_definition", "artifact:analysis_cohort"),
                _ref("cohort_definition", "table:cohort_flow"),
                _ref("table_one", "table:table_one"),
                *(_ref("measurement_audit", product) for product, _role in _MEASUREMENT_AUDIT_OUTPUTS),
                _ref("adjusted_association", "table:adjusted_association_estimates"),
                _ref("absolute_risk_context", "table:absolute_risk_context"),
                _ref("robustness_replay", "table:robustness_matrix"),
                _ref("robustness_replay", "table:robustness_summary"),
                *(
                    _ref(step_id, f"table:{step_id}_sensitivity")
                    for step_id in functional_step_ids
                ),
                *([_ref("ordinal_trend", "table:ordinal_trend")] if ordinal_present else []),
                _ref("visualization", "figure:visualization"),
            ],
            outputs=[ProgressiveOutputIntent(product_id="report:report", semantic_role="report")],
            literature_bindings=[],
        )
    )
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
            robustness_intents=[
                ProgressiveRobustnessIntent(
                    spec_id=request.complete_case_spec_id or "complete_case_primary_covariates",
                    axis="missing",
                    description=(
                        "Re-estimate the primary association on rows with the exposure, every adjustment "
                        "covariate, and the outcome all available, and compare with the primary analysis; "
                        "event times are not used to define completeness."
                    ),
                    missing_strategy="complete_case",
                    complete_case_variables=list(dict.fromkeys([exposure, *covariates, outcome])),
                )
            ],
            know_how_decisions=[],
        ),
    )
    materializations = tuple(
        _materialization(bound[step.step_id], step) for step in steps
    )
    return FamilySkeletonDraft(
        outline=outline, foundation=foundation, materializations=materializations
    )


#: The one projection entry point for both landmark association families; the
#: categorical name stays as the original public alias.
build_landmark_association_skeleton = build_landmark_categorical_skeleton


__all__ = [
    "FUNCTIONAL_FORM_KNOT_QUANTILES",
    "FUNCTIONAL_FORM_METHOD",
    "FamilySkeletonDraft",
    "ORDINAL_TREND_ACTION",
    "ORDINAL_TREND_METHOD",
    "PRIMARY_ACTION",
    "build_landmark_association_skeleton",
    "build_landmark_categorical_skeleton",
]
