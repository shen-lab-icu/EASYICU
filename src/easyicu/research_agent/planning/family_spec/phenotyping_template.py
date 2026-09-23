"""Host template for the cross-sectional phenotyping family.

An unsupervised clustering question over one feature vector per analysis row
measured inside the sealed observation window (for example: candidate
subphenotypes of a sepsis cohort from first-24-hour vitals and labs, described
by their clinical profile and outcome distribution). The Planner decides the
fit-feature roster, the optional cohort-membership flag, the characterization
variables, the reader labels, and the comparator applications; the host
projects every executable step and the unchanged Progressive validators and
compiler judge the result. Clusters are described, never called established
subtypes or causal groups.

Step layout (mirrors the family's reference workflow):

1. ``cohort_accounting``           denominators and cohort flow
2. ``feature_quality_audit``       feature availability, missingness, measurement process
3. ``primary_cluster_solution``    the host-owned cluster solution (primary)
4. ``cluster_number_selection``    prespecified candidate-k comparison
5. ``cluster_stability``           prespecified resampling stability
6. ``cluster_characterization``    outcome and clinical profile by cluster (secondary)
"""

from __future__ import annotations

from typing import Callable

from ...canonical_json import canonical_sha256
from ..design_selection import ResearchDesignCandidate, ResearchDesignSelection
from ..progressive_contract import (
    ProgressiveCohortIntent,
    ProgressiveCohortPredicate,
    ProgressiveDisplayLabel,
    ProgressiveFoundationMaterialization,
    ProgressiveLiteratureBinding,
    ProgressiveOutlineStep,
    ProgressiveOutputIntent,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressivePredicateValue,
    ProgressiveProductRef,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
    ProgressiveTableOneVariable,
)
from .contract import (
    PHENOTYPING_FAMILY_ID,
    FamilyPlanSpec,
    FamilySpecError,
    FamilySpecRequest,
    design_field_max_length,
)
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _label,
    _method_card_elements,
    _method_card_ids,
)
from .plan_language import bounded_roster, listing, plan_language, sentence

CLUSTER_SOLUTION_ACTION = "phenotyping.cluster_solution"
K_SELECTION_ACTION = "phenotyping.k_selection"
STABILITY_ACTION = "phenotyping.cluster_stability"
OUTCOME_BY_CLUSTER_ACTION = "phenotyping.outcome_by_cluster"
CLUSTER_SOLUTION_METHOD = "cross_sectional_phenotyping"
K_SELECTION_METHOD = "prespecified_cluster_number_selection"
STABILITY_METHOD = "prespecified_bootstrap_cluster_stability"
OUTCOME_BY_CLUSTER_METHOD = "descriptive_outcome_by_cluster"
_AUDIT_OUTPUTS = (
    ("table:feature_availability_audit", "measurement_missingness"),
    ("table:measurement_process_audit", "measurement_process"),
)
_MODULE_DESIGN_ELEMENTS: dict[str, tuple[str, ...]] = {
    "cluster_solution": ("dependence", "estimand", "exposure", "reporting", "time_zero"),
    "k_selection": ("reporting", "robustness"),
    "stability": ("reporting", "robustness"),
    "characterization": ("outcome", "reporting", "estimand"),
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
                    f"{module.replace('_', ' ')} step; the family template binds only the curated "
                    "design elements and retains the analysis-only claim ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _summary_for(request: FamilySpecRequest, name: str) -> str:
    candidate = request.candidate(name)
    coding = candidate.allowed_codings[0] if candidate is not None else "continuous"
    if name == request.outcome:
        return "count_percent"
    return "count_percent" if coding in {"binary", "categorical"} else "median_iqr"


def _cohort_intent(request: FamilySpecRequest, spec: FamilyPlanSpec) -> ProgressiveCohortIntent:
    end_hours = request.observation_window_hours
    inclusion: list[ProgressiveCohortPredicate] = []
    if spec.cohort_membership_column is not None:
        if end_hours is None:
            raise FamilySpecError(
                "family_spec_cohort_window_unavailable",
                "a membership restriction needs a typed observation window",
                path="cohort",
            )
        inclusion.append(
            ProgressiveCohortPredicate(
                concept_id=spec.cohort_membership_column,
                anchor="icu_admission",
                start_offset_hours=0.0,
                end_offset_hours=float(end_hours),
                aggregation="any",
                op="==",
                value=ProgressivePredicateValue(mode="number", number_value=1.0),
            )
        )
    for bound, op in ((request.age_min, ">="), (request.age_max, "<=")):
        if bound is None:
            continue
        if end_hours is None:
            raise FamilySpecError(
                "family_spec_cohort_window_unavailable",
                "a predicate-filtered cohort needs a typed observation window",
                path="cohort",
            )
        inclusion.append(
            ProgressiveCohortPredicate(
                concept_id="age",
                anchor="icu_admission",
                start_offset_hours=0.0,
                end_offset_hours=float(end_hours),
                aggregation="first",
                op=op,
                value=ProgressivePredicateValue(mode="number", number_value=float(bound)),
            )
        )
    if not inclusion:
        return ProgressiveCohortIntent(
            name=request.cohort_name, selection_mode="all_input_rows", inclusion=[], exclusion=[]
        )
    return ProgressiveCohortIntent(
        name=request.cohort_name, selection_mode="predicate_filtered", inclusion=inclusion, exclusion=[]
    )


def _estimand(head: str, labels: list[str], tail: str) -> str:
    """Name the features in the estimand while they fit; the plan lists them in full."""

    roster = bounded_roster(
        labels, budget=design_field_max_length("estimand") - len(head) - len(tail) - len(" ()")
    )
    return f"{head} ({roster}){tail}" if roster else f"{head}, named in the plan{tail}"


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    features: list[str],
    required_variables: list[str],
    method_keys: list[str],
) -> ResearchDesignSelection:
    outcome = _label(spec, request.outcome)
    membership = (
        _label(spec, spec.cohort_membership_column) if spec.cohort_membership_column else None
    )
    population = (
        f"analysis rows of the study cohort with {membership} present in the window"
        if membership
        else "all analysis rows of the study cohort"
    )
    language = plan_language(request.research_question)
    population_zh = (
        f"研究队列中窗口内存在 {membership} 的分析行" if membership else "研究队列的全部分析行"
    )
    hours_zh = (
        f"ICU 入院后 0–{request.observation_window_hours:g} h"
        if request.observation_window_hours is not None
        else "宿主物化的观察窗口"
    )
    hours = (
        f"0–{request.observation_window_hours:g} h after ICU admission"
        if request.observation_window_hours is not None
        else "the host-materialized observation window"
    )
    feature_text = ", ".join(_label(spec, name) for name in features)
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    selected = ResearchDesignCandidate(
        design_id="earlyfeature_cluster_phenotyping",
        analysis_type="trajectory_clustering",
        estimand=_estimand(
            f"The descriptive structure of candidate phenotypes among {population}, formed from "
            f"{len(features)} prespecified features measured in {hours}",
            [_label(spec, name) for name in features],
            f", with each cluster's size, feature profile, and {outcome} distribution.",
        ),
        time_zero="ICU admission for every analysis row; features use only the sealed window.",
        observation_window=(
            f"Features measured in {hours}; {outcome} taken from the available hospital outcome record."
        ),
        primary_method=(
            "Prespecified unsupervised clustering of standardized, median-imputed features with "
            "candidate-k comparison and resampling stability; clusters are described, not tested, "
            "and no causal or predictive claim is made."
        ),
        required_variables=required_variables,
        assumptions=[
            "Feature definitions, the observation window, and missing-data handling are fixed before "
            "clustering; clusters are descriptive candidate phenotypes, not established subtypes.",
            "Each analysis row is one ICU stay; rows are not assumed to be distinct patients.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=[],
        novelty_positioning=(
            "No novelty is claimed before completion; the candidate phenotypes are positioned "
            "against each screened comparator on population, features, time zero, and estimand."
        ),
        figure_role=(
            "Cluster structure as the main display, with the feature profile by cluster, the "
            "stability diagnostic, and the outcome distribution as downstream descriptive context."
        ),
        supports=(
            "Cluster sizes, standardized feature profiles, candidate-k and stability diagnostics, "
            f"and the {outcome} distribution by cluster."
        ),
        cannot_prove=(
            "No causal role, prognostic advantage, or treatment effect of any cluster; no established "
            "clinical subtype; no independence of repeated ICU stays; no transportability."
        ),
        reviewable_plan=(
            [
                f"{population_zh}；每行为一次 ICU 入住。",
                f"聚类特征为 {listing([_label(spec, name) for name in features], language)}，"
                f"每项均为 {hours_zh} 内预先设定的数值测量。",
                f"{outcome}，按聚类描述，作为下游的非因果分布。",
                "预先设定的无监督聚类，比较候选 k 值，描述特征谱，并评估重抽样稳定性；不设因果调整集。",
                "审计特征的可得性与缺失情况；插补规则在聚类前固定并报告。",
                "聚类前检查特征的可得性、量纲与极端值；以不同 k 值和重抽样稳定性检验稳健性。",
            ]
            if language == "zh"
            else [
                sentence(f"{population}; each analysis row is one ICU stay."),
                f"Clustering features {feature_text}, each a prespecified numeric measurement inside {hours}.",
                sentence(f"{outcome}, described by cluster as a downstream non-causal distribution."),
                "Prespecified unsupervised clustering with candidate-k comparison, feature-profile "
                "description, and resampling stability; no causal adjustment set.",
                "Feature availability and missingness are audited; the imputation rule is fixed before "
                "clustering and reported.",
                "Feature availability, scale, and extremes are checked before clustering; alternate k and "
                "resampling stability address robustness.",
            ]
        ),
        disposition="selected",
        decision_reason=(
            "The question asks for candidate phenotypes from early multivariate measurements and "
            "their clinical characteristics; unsupervised clustering with prespecified diagnostics "
            "answers it without a causal interpretation, chosen before any data are read."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="descriptive_group_summary",
        # Kept inside the authorized family: the alternative is the same
        # clustering study collapsed to a single partition, not a new family.
        analysis_type="trajectory_clustering",
        estimand=(
            f"Univariate summaries of each feature and {outcome} without any grouping structure."
        ),
        time_zero="ICU admission for every analysis row.",
        observation_window=f"Features measured in {hours}.",
        primary_method="Descriptive summaries per feature; no phenotype structure.",
        required_variables=required_variables,
        assumptions=["Each analysis row is one ICU stay."],
        literature_citation_keys=[
            key for key in ("strobe_2007", "record_2015") if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning="Recorded as the non-clustering alternative for audit; no novelty is claimed.",
        figure_role="Feature distributions only.",
        supports="Marginal feature and outcome descriptions.",
        cannot_prove=(
            "It cannot answer the question of candidate phenotypes because it forms no groups."
        ),
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because the question asks for candidate phenotypes; univariate summaries "
            "form no groups to characterize."
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


def build_phenotyping_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from typed facts."""

    if request.family_id != PHENOTYPING_FAMILY_ID:
        raise FamilySpecError(
            "family_spec_template_mismatch",
            "the phenotyping template received a request for another family",
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
    features = [str(value).strip() for value in spec.feature_variables]
    baseline = [str(value).strip() for value in spec.baseline_variables]
    membership = spec.cohort_membership_column
    method_keys = [key for key in request.allowed_literature_citation_keys if _method_card_elements(key)]

    def keys_for(module: str) -> list[str]:
        desired = set(_MODULE_DESIGN_ELEMENTS[module])
        return list(
            dict.fromkeys(
                [
                    *(k for k in method_keys if _method_card_elements(k) & desired),
                    *(request.direct_comparator_literature_keys if module == "cluster_solution" else []),
                ]
            )
        )[:12]

    cohort_variables = [identity, *([membership] if membership else []), outcome]
    characterization_variables = list(dict.fromkeys([identity, outcome, *baseline, *features]))
    required_variables = list(
        dict.fromkeys([identity, *([membership] if membership else []), outcome, *baseline, *features])
    )
    design = _design_selection(
        request,
        spec,
        features=features,
        required_variables=required_variables,
        method_keys=[k for k in method_keys if k in set(keys_for("cluster_solution"))][:6],
    )
    outcome_label = _label(spec, outcome)
    objectives = {
        "cohort_accounting": (
            "Account for every analysis row and record the cohort flow and denominators the "
            "phenotype description is read against."
        ),
        "feature_quality_audit": (
            "Audit the availability, missingness, and measurement process of every prespecified "
            "fit feature before clustering so the imputation rule is applied to known gaps."
        ),
        "primary_cluster_solution": (
            f"Form candidate phenotypes from the {len(features)} prespecified window-bound features "
            "with the host-owned standardization, imputation, and clustering mechanics, and report "
            "cluster sizes and standardized feature profiles."
        ),
        "cluster_number_selection": (
            "Compare the prespecified candidate numbers of clusters with the host's fixed criteria "
            "and record the selection basis for review."
        ),
        "cluster_stability": (
            "Assess phenotype reproducibility with the host's prespecified resampling stability "
            "diagnostic; unstable solutions are reported, not hidden."
        ),
        "cluster_characterization": (
            f"Describe each cluster's {outcome_label} distribution and clinical profile with "
            "standardized differences only; no causal or predictive claim."
        ),
    }
    outline_steps = [
        _outline_step(
            step_id="cohort_accounting", role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            variable_names=cohort_variables, citations=[],
        ),
        _outline_step(
            step_id="feature_quality_audit", role="auxiliary", module_id="measurement_audit",
            objective=objectives["feature_quality_audit"], depends_on=["cohort_accounting"],
            variable_names=[*features, *request.measurement_audit_columns], citations=[],
        ),
        _outline_step(
            step_id="primary_cluster_solution", role="primary", module_id="custom_analysis",
            objective=objectives["primary_cluster_solution"],
            depends_on=["cohort_accounting", "feature_quality_audit"],
            variable_names=features, citations=keys_for("cluster_solution"),
            action=CLUSTER_SOLUTION_ACTION,
        ),
        # The outline names the scientific variables a step reasons about even
        # when its executable inputs are typed products; the k-selection and
        # stability diagnostics re-read the fitted feature space.
        _outline_step(
            step_id="cluster_number_selection", role="secondary", module_id="custom_analysis",
            objective=objectives["cluster_number_selection"], depends_on=["primary_cluster_solution"],
            variable_names=[identity, *features], citations=keys_for("k_selection"),
            action=K_SELECTION_ACTION,
        ),
        _outline_step(
            step_id="cluster_stability", role="sensitivity", module_id="custom_analysis",
            objective=objectives["cluster_stability"], depends_on=["primary_cluster_solution"],
            variable_names=[identity, *features], citations=keys_for("stability"),
            action=STABILITY_ACTION,
        ),
        _outline_step(
            step_id="cluster_characterization", role="secondary", module_id="custom_analysis",
            objective=objectives["cluster_characterization"],
            depends_on=["cohort_accounting", "primary_cluster_solution"],
            variable_names=characterization_variables, citations=keys_for("characterization"),
            action=OUTCOME_BY_CLUSTER_ACTION,
        ),
    ]
    outline = ProgressivePlanOutline(
        analysis_type="trajectory_clustering",
        cohort_objective=(
            "Describe candidate phenotypes of the study cohort from prespecified "
            f"window-bound features and their {outcome_label} distribution, keeping feature "
            "availability, stability, and repeated stays visible for review."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: the host projected every executable coordinate "
            "from typed StudyContext facts; the Planner supplied the fit-feature roster, the "
            "characterization variables, reader labels, and comparator applications. Clusters are "
            "descriptive under the analysis-only claim ceiling."
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
            raw_inputs=list(dict.fromkeys(cohort_variables)), literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="feature_quality_audit", planned_analysis_role="auxiliary", module_id="measurement_audit",
            objective=objectives["feature_quality_audit"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([*features, *request.measurement_audit_columns])),
            product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role=role)
                for product, role in _AUDIT_OUTPUTS
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="primary_cluster_solution", planned_analysis_role="primary", module_id="custom_analysis",
            objective=objectives["primary_cluster_solution"],
            depends_on=["cohort_accounting", "feature_quality_audit"],
            raw_inputs=list(features),
            # The action contract declares no product inputs; the compiler's
            # typed-cohort closure binds the analysis cohort to the fit itself.
            outputs=[
                ProgressiveOutputIntent(product_id="table:phenotype_profiles", semantic_role="custom"),
                ProgressiveOutputIntent(product_id="table:phenotype_assignments", semantic_role="custom"),
            ],
            scientific_action_id=CLUSTER_SOLUTION_ACTION,
            custom_method=CLUSTER_SOLUTION_METHOD,
            phenotyping_feature_columns=list(features),
            literature_bindings=_bindings(
                bound["primary_cluster_solution"], module="cluster_solution",
                comparator_applications=applications,
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="cluster_number_selection", planned_analysis_role="secondary", module_id="custom_analysis",
            objective=objectives["cluster_number_selection"], depends_on=["primary_cluster_solution"],
            product_inputs=[_ref("primary_cluster_solution", "table:phenotype_assignments")],
            outputs=[ProgressiveOutputIntent(product_id="table:cluster_selection", semantic_role="custom")],
            scientific_action_id=K_SELECTION_ACTION, custom_method=K_SELECTION_METHOD,
            literature_bindings=_bindings(
                bound["cluster_number_selection"], module="k_selection",
                comparator_applications=applications,
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="cluster_stability", planned_analysis_role="sensitivity", module_id="custom_analysis",
            objective=objectives["cluster_stability"], depends_on=["primary_cluster_solution"],
            product_inputs=[_ref("primary_cluster_solution", "table:phenotype_assignments")],
            outputs=[ProgressiveOutputIntent(product_id="table:cluster_stability", semantic_role="custom")],
            scientific_action_id=STABILITY_ACTION, custom_method=STABILITY_METHOD,
            literature_bindings=_bindings(
                bound["cluster_stability"], module="stability", comparator_applications=applications,
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="cluster_characterization", planned_analysis_role="secondary", module_id="custom_analysis",
            objective=objectives["cluster_characterization"],
            depends_on=["cohort_accounting", "primary_cluster_solution"],
            raw_inputs=characterization_variables,
            product_inputs=[
                _ref("cohort_accounting", "artifact:analysis_cohort"),
                _ref("primary_cluster_solution", "table:phenotype_assignments"),
            ],
            outputs=[ProgressiveOutputIntent(product_id="table:outcome_by_cluster", semantic_role="custom")],
            scientific_action_id=OUTCOME_BY_CLUSTER_ACTION, custom_method=OUTCOME_BY_CLUSTER_METHOD,
            phenotyping_comparison_variables=[
                ProgressiveTableOneVariable(name=name, summary=_summary_for(request, name))
                for name in dict.fromkeys([outcome, *baseline, *features])
            ],
            literature_bindings=_bindings(
                bound["cluster_characterization"], module="characterization",
                comparator_applications=applications,
            ),
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
            cohort=_cohort_intent(request, spec),
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


__all__ = [
    "CLUSTER_SOLUTION_ACTION",
    "K_SELECTION_ACTION",
    "OUTCOME_BY_CLUSTER_ACTION",
    "STABILITY_ACTION",
    "build_phenotyping_skeleton",
]
