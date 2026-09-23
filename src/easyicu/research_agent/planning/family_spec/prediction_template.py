"""Host template for the static binary prediction family.

A prognostic model question over predictors measured inside the sealed
observation window and a 0/1 outcome (for example: in-hospital death from
first-24-hour vitals and labs). The Planner decides the predictor roster,
reader labels, and comparator applications; the host projects the
discrimination/calibration primary, the calibration, internal validation, and
decision-curve secondaries, the composite figure, and the report, and the
unchanged Progressive validators and compiler judge the result. Every
analysis step is owned by the host prediction adapter; the figure by the host
prediction composite renderer.

Step layout (mirrors the family's reference workflow):

1. ``cohort_accounting``     denominators and cohort flow
2. ``baseline_context``      predictors by outcome, SMD only
3. ``measurement_audit``     predictor availability and missingness
4. ``primary_performance``   scores, discrimination, calibration (primary)
5. ``calibration_metrics``   calibration slope/intercept/Brier (secondary)
6. ``internal_validation``   optimism-corrected internal validation (secondary)
7. ``clinical_utility``      decision curve (secondary)
8. ``visualization`` / ``report``
"""

from __future__ import annotations

from typing import Callable

from ...canonical_json import canonical_sha256
from ...contracts.figure_plan import STATIC_PREDICTION_VALIDATION_FIGURE_SUFFIX
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
    ProgressiveRobustnessIntent,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
    ProgressiveTableOneVariable,
)
from .contract import PREDICTION_FAMILY_ID, FamilyPlanSpec, FamilySpecError, FamilySpecRequest
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _cohort_intent,
    _label,
    _method_card_elements,
    _method_card_ids,
)
from .plan_language import listing, plan_language, sentence

PRIMARY_ACTION = "prediction.discrimination_calibration"
CALIBRATION_ACTION = "prediction.calibration_metrics"
VALIDATION_ACTION = "prediction.internal_validation"
UTILITY_ACTION = "prediction.decision_curve"
PRIMARY_METHOD = "prespecified_prediction_model_discrimination_calibration"
CALIBRATION_METHOD = "prespecified_calibration_metrics"
VALIDATION_METHOD = "prespecified_internal_validation"
UTILITY_METHOD = "prespecified_decision_curve_analysis"
#: The one plan-locked robustness variant the static prediction owner executes.
COMPLETE_CASE_SPEC_ID = "complete_case_model_roster"
_AUDIT_OUTPUTS = (
    ("table:measurement_missingness", "measurement_missingness"),
    ("table:measurement_process_audit", "measurement_process"),
)
#: The four core sources a rendering-only figure may bind directly; the host
#: figure shaping adds the registered clinical-utility table afterwards, which
#: completes the composite renderer's exact input set.
_FIGURE_INPUTS = (
    ("primary_performance", "table:prediction_scores"),
    ("primary_performance", "table:model_performance"),
    ("internal_validation", "table:validation"),
    ("calibration_metrics", "table:calibration"),
)
_REPORT_RESULT_INPUTS = (*_FIGURE_INPUTS, ("clinical_utility", "table:clinical_utility"))
_MODULE_DESIGN_ELEMENTS: dict[str, tuple[str, ...]] = {
    "primary": ("dependence", "estimand", "outcome", "reporting", "time_zero", "missing_data"),
    "secondary": ("reporting", "robustness", "estimand"),
}


def _bindings(
    outline_step: ProgressiveOutlineStep,
    *,
    module: str,
    comparator_applications: dict[str, str],
) -> list[ProgressiveLiteratureBinding]:
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
                    f"Apply the host-curated method card(s) {_method_card_ids(key, desired)} to the "
                    f"prediction {module} step; the family template binds only the curated design "
                    "elements and retains the analysis-only claim ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _summary_for(request: FamilySpecRequest, name: str) -> str:
    candidate = next((item for item in request.feature_candidates if item.name == name), None)
    coding = candidate.allowed_codings[0] if candidate is not None else "continuous"
    return "count_percent" if coding in {"binary", "categorical"} else "median_iqr"


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    predictors: list[str],
    required_variables: list[str],
    method_keys: list[str],
) -> ResearchDesignSelection:
    outcome = _label(spec, request.outcome)
    hours = (
        f"0–{request.observation_window_hours:g} h after ICU admission"
        if request.observation_window_hours is not None
        else "the host-materialized observation window"
    )
    predictor_text = ", ".join(_label(spec, name) for name in predictors)
    unit_text = (
        "each analysis row is one ICU stay; patient-level dependence is declared and the split "
        "keeps a patient's stays together"
        if request.cluster_unit == "patient"
        else "each analysis row is one ICU stay and rows are not assumed to be distinct patients"
    )
    language = plan_language(request.research_question)
    unit_text_zh = (
        "每行为一次 ICU 入住；已声明患者层面的相关性，数据切分时同一患者的入住保持在同一侧"
        if request.cluster_unit == "patient"
        else "每行为一次 ICU 入住，不假定各行来自不同患者"
    )
    hours_zh = (
        f"ICU 入院后 0–{request.observation_window_hours:g} h"
        if request.observation_window_hours is not None
        else "宿主物化的观察窗口"
    )
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    selected = ResearchDesignCandidate(
        design_id="static_window_prediction",
        analysis_type="prediction_model",
        estimand=(
            f"The predicted probability of {outcome} for each analysis row from {len(predictors)} "
            f"prespecified predictors measured in {hours} ({predictor_text}), evaluated by "
            "discrimination, calibration, internal validation, and decision-curve utility."
        ),
        time_zero=f"ICU admission; predictors use only information available in {hours}.",
        observation_window=(
            f"Predictors measured in {hours}; {outcome} taken from the hospital outcome record "
            "after the window."
        ),
        primary_method=(
            "Prespecified static prediction model with host-owned preprocessing, a patient-grouped "
            "split, discrimination and calibration on held-out rows, optimism-corrected internal "
            "validation, calibration metrics, and a decision curve; no causal claim."
        ),
        required_variables=required_variables,
        assumptions=[
            "Predictor timestamps separate the observation window from information after the outcome.",
            f"{unit_text[0].upper()}{unit_text[1:]}.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=[],
        novelty_positioning=(
            "No novelty is claimed before completion; the model is positioned against each screened "
            "comparator on population, predictors, time zero, and outcome."
        ),
        figure_role=(
            "Calibration as the main visual, with discrimination, internal validation, and decision-"
            "curve utility, and data-quality context in the supplement."
        ),
        supports=(
            f"Discrimination, calibration, internal-validation, and threshold-utility evidence for a "
            f"{outcome} model on the sealed predictor set."
        ),
        cannot_prove=(
            "No causal effect, no external transportability, no clinical benefit of acting on the "
            "score, and no independence of repeated ICU stays beyond the declared grouping."
        ),
        reviewable_plan=(
            [
                f"研究队列的全部输入行；{unit_text_zh}。",
                f"预测变量为 {listing([_label(spec, name) for name in predictors], language)}，"
                f"每项均在 {hours_zh} 内测量，并按行汇总。",
                f"{outcome}，取自观察窗口之后。",
                "预先设定的静态模型，由宿主负责的预处理只在训练集上拟合；在留出行上评估区分度与校准。",
                "审计预测变量的可得性与缺失情况；插补只在训练集上拟合并报告。",
                "预先设定乐观校正的内部验证、校准指标和决策曲线；拟合前检查信息泄漏、重复入住结构和结局编码。",
            ]
            if language == "zh"
            else [
                f"All input rows of the study cohort; {unit_text}.",
                f"Predictors {predictor_text}, each measured inside {hours} and aggregated per row.",
                sentence(f"{outcome}, taken after the observation window."),
                "Prespecified static model with host-owned preprocessing fitted on the training split "
                "only; discrimination and calibration on held-out rows.",
                "Predictor availability and missingness are audited; imputation is fitted on the "
                "training split only and reported.",
                "Optimism-corrected internal validation, calibration metrics, and a decision curve are "
                "prespecified; leakage, repeated-stay structure, and outcome coding are checked before "
                "fitting.",
            ]
        ),
        disposition="selected",
        decision_reason=(
            "The question asks for a prognostic model from window-bound information; a static "
            "prediction design with prespecified performance, validation, and utility reporting "
            "answers it and needs no longitudinal landmark structure, chosen before any data are read."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="dynamic_landmark_prediction",
        analysis_type="prediction_model",
        estimand=(
            f"Dynamically updated predictions of {outcome} at successive landmarks from repeated "
            "measurements."
        ),
        time_zero="Each landmark after ICU admission.",
        observation_window="Repeated measurement windows up to each landmark.",
        primary_method="Landmark-updated prediction with time-varying predictors.",
        required_variables=required_variables,
        assumptions=["A verified longitudinal time structure and landmark schedule exist."],
        literature_citation_keys=[
            key for key in ("strobe_2007", "record_2015") if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning="Recorded as the dynamic alternative for audit; no novelty is claimed.",
        figure_role="Landmark-specific performance curves.",
        supports="Time-updated predictions when a longitudinal structure is materialized.",
        cannot_prove=(
            "It cannot run on a single window-aggregated feature vector and would need a materialized "
            "longitudinal structure the sealed context does not carry."
        ),
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because the sealed context carries one aggregated feature vector per row and no "
            "longitudinal landmark structure."
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


def build_prediction_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from typed facts."""

    if request.family_id != PREDICTION_FAMILY_ID:
        raise FamilySpecError(
            "family_spec_template_mismatch",
            "the prediction template received a request for another family",
            path="family_id",
        )
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "the spec does not bind this request",
            path="request_sha256",
        )
    identity = request.identity_column
    outcome = request.outcome
    predictors = [str(value).strip() for value in spec.feature_variables]
    method_keys = [key for key in request.allowed_literature_citation_keys if _method_card_elements(key)]

    def keys_for(module: str, *, comparators: bool = False) -> list[str]:
        desired = set(_MODULE_DESIGN_ELEMENTS[module])
        return list(
            dict.fromkeys(
                [
                    *(k for k in method_keys if _method_card_elements(k) & desired),
                    *(request.direct_comparator_literature_keys if comparators else []),
                ]
            )
        )[:12]

    primary_keys = keys_for("primary", comparators=True)
    secondary_keys = keys_for("secondary")
    required_variables = list(dict.fromkeys([identity, outcome, *predictors]))[:64]
    design = _design_selection(
        request,
        spec,
        predictors=predictors,
        required_variables=required_variables,
        method_keys=[k for k in method_keys if k in set(primary_keys)][:6],
    )
    outcome_label = _label(spec, outcome)
    objectives = {
        "cohort_accounting": (
            "Account for every input analysis row and record the modelling denominator and cohort flow."
        ),
        "baseline_context": (
            f"Describe the prespecified predictors by {outcome_label} with standardized differences "
            "only; repeated units carry no independent tests."
        ),
        "measurement_audit": (
            "Audit predictor availability, missingness, and measurement process so the training-split "
            "imputation is applied to known gaps."
        ),
        "primary_performance": (
            f"Fit the prespecified static {outcome_label} model on the training split with host-owned "
            "preprocessing and report per-row scores plus held-out discrimination and calibration."
        ),
        "calibration_metrics": (
            "Quantify calibration slope, intercept, and Brier score of the held-out scores."
        ),
        "internal_validation": (
            "Report optimism-corrected internal validation of the discrimination and calibration."
        ),
        "clinical_utility": (
            "Report the decision curve of the held-out scores across prespecified thresholds."
        ),
        "visualization": (
            "Assemble the composite prediction display: calibration, discrimination, internal "
            "validation, and decision-curve utility from the host-owned tables."
        ),
        "report": (
            "Produce the zero-patient-row plan report for human review: sources, denominators, "
            "missingness, performance, validation, utility, limitations, and the analysis-only boundary."
        ),
    }
    outline_steps = [
        _outline_step(
            step_id="cohort_accounting", role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            variable_names=[identity, outcome], citations=[],
        ),
        _outline_step(
            step_id="baseline_context", role="auxiliary", module_id="table_one",
            objective=objectives["baseline_context"], depends_on=["cohort_accounting"],
            variable_names=[outcome, *predictors], citations=[],
        ),
        _outline_step(
            step_id="measurement_audit", role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
            variable_names=[*predictors, *request.measurement_audit_columns], citations=[],
        ),
        _outline_step(
            step_id="primary_performance", role="primary", module_id="custom_analysis",
            objective=objectives["primary_performance"],
            depends_on=["cohort_accounting", "measurement_audit"],
            variable_names=[outcome, *predictors], citations=primary_keys,
            action=PRIMARY_ACTION,
        ),
        *(
            _outline_step(
                step_id=step_id, role="secondary", module_id="custom_analysis",
                objective=objectives[step_id], depends_on=["primary_performance"],
                variable_names=[identity, outcome], citations=secondary_keys, action=action,
            )
            for step_id, action in (
                ("calibration_metrics", CALIBRATION_ACTION),
                ("internal_validation", VALIDATION_ACTION),
                ("clinical_utility", UTILITY_ACTION),
            )
        ),
        _outline_step(
            step_id="visualization", role="auxiliary", module_id="visualization",
            objective=objectives["visualization"],
            depends_on=["primary_performance", "calibration_metrics", "internal_validation", "clinical_utility"],
            variable_names=[outcome], citations=[],
        ),
    ]
    outline_steps.append(
        _outline_step(
            step_id="report", role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=[step.step_id for step in outline_steps],
            variable_names=[identity, outcome, *predictors], citations=[],
        )
    )
    outline = ProgressivePlanOutline(
        analysis_type="prediction_model",
        cohort_objective=(
            f"Develop and internally validate a prespecified static {outcome_label} prediction model "
            "on the study cohort from window-bound predictors, keeping missingness, "
            "leakage checks, and repeated stays visible for review."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: the host projected every executable coordinate "
            "from typed StudyContext facts; the Planner supplied the predictor roster, reader labels, "
            "and comparator applications. All results stay at plan level under the analysis-only "
            "claim ceiling."
        ),
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    bound = {step.step_id: step for step in outline.steps}
    applications = spec.applications
    steps = [
        ProgressiveSkeletonStep(
            step_id="cohort_accounting", planned_analysis_role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            raw_inputs=[identity, outcome], literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="baseline_context", planned_analysis_role="auxiliary", module_id="table_one",
            objective=objectives["baseline_context"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([outcome, *predictors])),
            table_one_group_by=outcome, table_one_mode="descriptive_smd_only",
            table_one_variables=[
                ProgressiveTableOneVariable(name=name, summary=_summary_for(request, name))
                for name in predictors
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="measurement_audit", planned_analysis_role="auxiliary", module_id="measurement_audit",
            objective=objectives["measurement_audit"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([*predictors, *request.measurement_audit_columns])),
            product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role=role)
                for product, role in _AUDIT_OUTPUTS
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="primary_performance", planned_analysis_role="primary", module_id="custom_analysis",
            objective=objectives["primary_performance"],
            depends_on=["cohort_accounting", "measurement_audit"],
            # Model-column prefix only: the host owner treats every raw column
            # before its typed cohort input as outcome or predictor, so the
            # row identity must not appear here.
            raw_inputs=list(dict.fromkeys([outcome, *predictors])),
            outputs=[
                ProgressiveOutputIntent(product_id="table:prediction_scores", semantic_role="custom"),
                ProgressiveOutputIntent(product_id="table:model_performance", semantic_role="custom"),
            ],
            scientific_action_id=PRIMARY_ACTION, custom_method=PRIMARY_METHOD,
            literature_bindings=_bindings(
                bound["primary_performance"], module="primary", comparator_applications=applications,
            ),
        ),
        *(
            ProgressiveSkeletonStep(
                step_id=step_id, planned_analysis_role="secondary", module_id="custom_analysis",
                objective=objectives[step_id], depends_on=["primary_performance"],
                product_inputs=[_ref("primary_performance", "table:prediction_scores")],
                outputs=[ProgressiveOutputIntent(product_id=product, semantic_role="custom")],
                scientific_action_id=action, custom_method=method,
                literature_bindings=_bindings(
                    bound[step_id], module="secondary", comparator_applications=applications,
                ),
            )
            for step_id, action, method, product in (
                ("calibration_metrics", CALIBRATION_ACTION, CALIBRATION_METHOD, "table:calibration"),
                ("internal_validation", VALIDATION_ACTION, VALIDATION_METHOD, "table:validation"),
                ("clinical_utility", UTILITY_ACTION, UTILITY_METHOD, "table:clinical_utility"),
            )
        ),
        ProgressiveSkeletonStep(
            step_id="visualization", planned_analysis_role="auxiliary", module_id="visualization",
            objective=objectives["visualization"],
            depends_on=["primary_performance", "calibration_metrics", "internal_validation", "clinical_utility"],
            raw_inputs=[],
            product_inputs=[_ref(producer, product) for producer, product in _FIGURE_INPUTS],
            outputs=[
                ProgressiveOutputIntent(product_id="figure:visualization", semantic_role="figure"),
                # The host renderer exports repeated patient-level split
                # variability on its own surface. Declaring it makes the plan
                # promise the validation roles a reviewer is owed instead of
                # leaving a rendered figure outside the reviewed plan.
                ProgressiveOutputIntent(
                    product_id=f"figure:visualization{STATIC_PREDICTION_VALIDATION_FIGURE_SUFFIX}",
                    semantic_role="figure",
                ),
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="report", planned_analysis_role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=bound["report"].depends_on, raw_inputs=[],
            product_inputs=[
                _ref("cohort_accounting", "artifact:analysis_cohort"),
                _ref("cohort_accounting", "table:cohort_flow"),
                _ref("baseline_context", "table:table_one"),
                *(_ref("measurement_audit", product) for product, _role in _AUDIT_OUTPUTS),
                *(_ref(producer, product) for producer, product in _REPORT_RESULT_INPUTS),
                _ref("visualization", "figure:visualization"),
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
            robustness_intents=[
                # The playbook's "complete-case versus imputed workflow": the
                # static prediction owner refits the unchanged model, outcome,
                # predictor roster, and patient split on complete cases and
                # reports it beside the training-split imputation.  The set is
                # the model roster itself -- a narrower or wider set would be
                # another analysis, and the owner would not execute it.
                ProgressiveRobustnessIntent(
                    spec_id=COMPLETE_CASE_SPEC_ID,
                    axis="missing",
                    description=(
                        "Refit the prespecified model on rows with the outcome and every predictor "
                        "observed, keeping the model, predictor roster, and patient split unchanged, "
                        "and compare held-out discrimination and calibration with the primary's "
                        "training-split imputation."
                    ),
                    missing_strategy="complete_case",
                    complete_case_variables=list(dict.fromkeys([outcome, *predictors])),
                )
            ],
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
    "CALIBRATION_ACTION",
    "PRIMARY_ACTION",
    "UTILITY_ACTION",
    "VALIDATION_ACTION",
    "build_prediction_skeleton",
]
