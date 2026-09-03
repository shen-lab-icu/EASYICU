"""Analysis-family ownership and typed product authorization rules."""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .association_execution import association_binary_sensitivity_contract
from .declared_product import effect_bearing_name, effect_bearing_product, typed_product
from .primary_cohort import _is_primary_analysis_cohort_method
from .product_identity import (
    normalised_expected_output_names as _normalised_expected_output_names,
    normalised_method_head as _normalised_method_head,
    normalised_structured_output_names as _normalised_structured_output_names,
)
from ..planning.cohort_contract import cohort_definition_contract_issue
from ..planning.figure_step_contract import _output_declares_figure, _step_produces_figure
from ..schema import AnalysisPlan, AnalysisStep, ValidationFinding


def _output_declares_auxiliary_log(output: str) -> bool:
    token = str(output or "").strip().lower()
    if ":" not in token:
        return False
    kind, _ = token.split(":", 1)
    return kind.strip() in {"log", "receipt", "manifest"}


_FIGURE_METHODS = frozenset(
    {
        "figure",
        "visualization",
        "visualisation",
        "plotting",
        "publication_figure",
        "publication_figure_generation",
        "render_figure",
        "figure_generation",
        "chart_generation",
    }
)


def _step_expects_figure(step: AnalysisStep) -> bool:
    method = re.sub(
        r"[^a-z0-9]+", "_", str(step.method or "").strip().lower()
    ).strip("_")
    return method in _FIGURE_METHODS or _step_produces_figure(step)


def _step_is_figure_only(step: AnalysisStep) -> bool:
    """A pure figure/render step: it declares a figure and owns NO non-figure
    structured product.

    A render *child* (``expected_outputs=['figure:forest_plot']``) is figure-only
    and is never the primary estimand. But a *combined* model+figure step that
    the replanner can emit before the figure/table splitter runs
    (``['statistic:primary_estimate', 'figure:forest_plot']``) still owns the
    result product and must remain eligible as the primary model.
    """

    if not _step_expects_figure(step):
        return False
    return not any(
        not _output_declares_figure(output)
        and not _output_declares_auxiliary_log(output)
        for output in step.expected_outputs or []
    )


def _cohort_definition_contract_findings(
    plan: AnalysisPlan,
) -> List[ValidationFinding]:
    """Adapt the dependency-neutral cohort owner issue to runtime findings."""

    issue = cohort_definition_contract_issue(plan)
    return [ValidationFinding(**issue)] if issue is not None else []


# Contract "family" buckets this enforcer knows how to normalise. These are the
# figure/metric-contract groupings, NOT the analysis_types registry keys — the
# two vocabularies diverged historically. _normalise_contract_family bridges
# them so the authoritative stamped ``plan.analysis_type`` (registry key) drives
# the contract instead of only the keyword heuristic.
_CONTRACT_FAMILIES = {
    "prediction_model",
    "clustering",
    "robustness",
    "bias_audit",
    "survival",
    "dynamic_prediction",
    "causal_inference",
    "treatment_response",
    "validation",
}
# Registry analysis_type keys whose bucket name DIFFERS from the key. Keys that
# already equal a contract bucket (survival, dynamic_prediction, ...) pass
# through via the ``in _CONTRACT_FAMILIES`` check below, so only true renames go
# here. Families with no figure/metric contract (descriptive_epidemiology,
# association_study, multimodal, reinforcement_learning, the *_audit and
# *_sensitivity shapes) are intentionally absent → they fall back to the keyword
# heuristic / robustness forest rather than getting a forced filler figure.
_ANALYSIS_TYPE_TO_CONTRACT_FAMILY = {
    "trajectory_clustering": "clustering",
}
# Buckets the keyword heuristic below can already produce on its own. The
# authoritative stamped ``plan.analysis_type`` is only allowed to INTRODUCE
# buckets OUTSIDE this set, because infer_analysis_type is deliberately looser
# than these gates — e.g. a bare "model" maps to prediction_model — and must not
# widen the conservative prediction/clustering/bias enforcement onto plans the
# heuristic leaves alone. The newer result-bearing buckets (survival,
# dynamic_prediction, causal_inference, treatment_response, validation) are NOT
# heuristic-reachable, so the stamped type is allowed to introduce them, and
# their markers are specific enough to avoid the bare-"model" false-match.
_HEURISTIC_REACHABLE_FAMILIES = {
    "prediction_model",
    "bias_audit",
    "robustness",
}


def _normalise_contract_family(raw: Optional[str]) -> str:
    """Map a registry analysis_type (or a legacy bucket) to a contract family.

    Returns ``""`` for families this enforcer has no figure/metric contract for
    (e.g. ``association_study``, ``descriptive_epidemiology``), so the caller
    falls back to the keyword heuristic exactly as before. Values that are
    already contract buckets pass through unchanged for backward compatibility.
    """
    value = (raw or "").strip().lower()
    if value in _CONTRACT_FAMILIES:
        return value
    return _ANALYSIS_TYPE_TO_CONTRACT_FAMILY.get(value, "")


def _article_display_roles(steps: Sequence[AnalysisStep]) -> set[str]:
    """Infer article-display roles from method owners and declared products.

    Free-text ids and intents are deliberately excluded: this helper decides
    whether contract augmentation may target an existing article module.
    """

    roles: set[str] = set()
    for step in steps or []:
        method = _normalised_method_head(str(step.method or ""))
        outputs = _normalised_structured_output_names(step.expected_outputs or [])
        if _cohort_change_contract_applies(step):
            roles.add("cohort_accounting")
        if method in {"descriptive", "table_one", "baseline_characteristics"} and (
            outputs & {"table_one", "baseline_table", "baseline_characteristics"}
        ):
            roles.add("baseline_context")
        if method in {
            "data_quality_audit",
            "missingness",
            "missingness_audit",
            "missingness_measurement_audit",
        } and (
            outputs
            & {
                "data_quality",
                "measurement_availability",
                "missingness",
                "missingness_measurement_audit",
                "missingness_profile",
            }
        ):
            roles.add("data_quality")
        if step.planned_analysis_role == "primary" and (
            _effect_contract_applies(step)
            or _prediction_contract_applies(step)
            or _clustering_contract_applies(
                method=str(step.method or ""),
                step_id=step.step_id,
                intent=step.intent or "",
                expected_outputs=step.expected_outputs or [],
            )
        ):
            roles.add("primary_estimand")
        if method in _PLAN_FAMILY_METHODS["robustness"] and (
            outputs
            & {
                "complete_case_n",
                "robustness_matrix",
                "robustness_summary",
                "sensitivity_grid",
            }
        ):
            roles.add("robustness")
    return roles


def _best_contract_step_for_outputs(steps: Sequence[AnalysisStep]) -> AnalysisStep:
    """Use the final agent-declared owner; callers already filtered by family."""

    return list(steps)[-1]


_CLUSTERING_ANALYSIS_METHODS = frozenset(
    {
        "trajectory_clustering",
        "trajectory_clustering_analysis",
        "trajectory_feature_clustering",
        "clustering",
        "kmeans",
        "k_means",
        "kmeans_clustering",
        "k_means_clustering",
        "phenotyping",
        "phenotype_clustering",
        "phenotype_clustering_and_structure",
        "unsupervised_clustering",
        "latent_class",
        "latent_class_analysis",
        "latent_class_model",
        "cluster_analysis",
        "gmm",
        "gaussian_mixture",
        "gaussian_mixture_model",
    }
)
_KMEANS_AUXILIARY_METHODS = frozenset(
    {
        "kmeans",
        "k_means",
        "kmeans_clustering",
        "k_means_clustering",
        "kmeans_trajectory_clustering",
        "k_means_trajectory_clustering",
        "trajectory_kmeans_clustering",
        "trajectory_k_means_clustering",
    }
)
_CLUSTERING_CONTRACT_OUTPUTS = frozenset(
    {
        "cluster_assignments",
        "cluster_characteristics",
        "cluster_mortality",
        "outcome_by_cluster",
        "clustering_metrics",
        "silhouette_score",
        "silhouette_metrics",
        "cluster_count",
        "cluster_sizes",
        "cluster_stability",
        "cluster_stability_assignments",
        "cluster_selection",
        "cluster_selection_criterion",
        "trajectory_features",
        "trajectory_profiles",
        "trajectory_membership",
        "trajectory_missingness_policy",
        "cohort_flow",
        "clustering_methodology",
        "clustering_visualization",
    }
)


def _clustering_contract_applies(
    *,
    method: str,
    step_id: str = "",
    intent: str = "",
    expected_outputs: Sequence[str] | str = (),
    auxiliary_kmeans_only: bool = False,
    minimum_output_signals: int = 1,
) -> bool:
    """Return whether a step owns a closed clustering-analysis contract.

    Scientific ownership comes from an exact method head plus declared standard
    products, never prose or output names alone.  The stricter auxiliary mode
    additionally prevents a KMeans fallback from replacing latent-class/GMM or
    otherwise agent-selected phenotyping methods.
    """

    head = _normalised_method_head(method)
    outputs = _normalised_structured_output_names(expected_outputs)
    output_signals = outputs & _CLUSTERING_CONTRACT_OUTPUTS
    allowed_methods = (
        _KMEANS_AUXILIARY_METHODS
        if auxiliary_kmeans_only
        else _CLUSTERING_ANALYSIS_METHODS
    )
    return head in allowed_methods and len(output_signals) >= minimum_output_signals


def clustering_contract_applies(step: AnalysisStep) -> bool:
    """Return whether ``step`` owns a closed clustering-analysis contract.

    This public, case-neutral predicate is shared by prompt projection and
    execution routing so method-family guidance cannot drift into a second
    private allowlist.  Ownership still requires both an exact normalized
    method family and declared structured clustering products.
    """

    return _clustering_contract_applies(
        method=str(step.method or ""),
        step_id=str(step.step_id or ""),
        intent=str(step.intent or ""),
        expected_outputs=step.expected_outputs or [],
    )


_PLAN_FAMILY_METHODS: dict[str, frozenset[str]] = {
    "prediction_model": frozenset(
        {
            "prediction_model",
            "prediction_model_analysis",
            "model_training",
            "risk_prediction",
            "classification_model",
        }
    ),
    "survival": frozenset(
        {
            "survival_analysis",
            "time_to_event",
            "cox",
            "cox_ph",
            "cox_proportional_hazards",
            "kaplan_meier",
        }
    ),
    "dynamic_prediction": frozenset(
        {"dynamic_prediction", "landmark_model", "time_updated_prediction"}
    ),
    "causal_inference": frozenset(
        {
            "causal_inference",
            "causal_emulation",
            "iptw",
            "ipw",
            "psm",
            "propensity_score",
            "target_trial",
            "target_trial_emulation",
            "g_computation",
        }
    ),
    "treatment_response": frozenset(
        {"treatment_response", "interaction_model", "effect_modification"}
    ),
    "validation": frozenset(
        {"validation", "external_validation", "transportability_validation"}
    ),
    "bias_audit": frozenset(
        {
            "bias_audit_association",
            "selection_bias_audit",
            "confounding_bias_audit",
        }
    ),
    "robustness": frozenset(
        {
            "association_robustness",
            "prespecified_robustness",
            "robustness_sensitivity",
            "sensitivity_comparison",
        }
    ),
}
_PLAN_FAMILY_OUTPUTS: dict[str, frozenset[str]] = {
    "prediction_model": frozenset(
        {"auroc", "brier_score", "model_performance", "roc_curve", "calibration_curve"}
    ),
    "survival": frozenset(
        {"hazard_ratio", "hr", "cox_summary", "km_curve", "survival_curves"}
    ),
    "dynamic_prediction": frozenset(
        {"time_varying_auroc", "prediction_horizon", "horizon_performance"}
    ),
    "causal_inference": frozenset(
        {"adjusted_effect", "covariate_balance", "causal_effect", "balance_pre_post"}
    ),
    "treatment_response": frozenset(
        {"overall_effect", "interaction_pvalue", "subgroup_effects"}
    ),
    "validation": frozenset(
        {"validation_auroc", "calibration_slope", "validation_performance"}
    ),
    "bias_audit": frozenset(
        {"selection_bias_warning", "association_summary", "clinical_constraint_warning"}
    ),
    "robustness": frozenset(
        {"robustness_matrix", "robustness_summary", "complete_case_n"}
    ),
}


_EFFECT_CONTRACT_METHODS = frozenset(
    {
        "association",
        "association_analysis",
        "association_study",
        "multivariable_association",
        "logistic_regression_association",
        "regression",
        "logit",
        "logistic_regression",
        "adjusted_logistic_regression",
        "regularized_logistic_regression",
        "mixed_effects_regression",
        "generalized_linear_model",
        "glm",
        "gee",
        "ordinal_logistic_regression",
        "ordinal_dose_response",
        "cox",
        "cox_ph",
        "cox_regression",
        "cox_proportional_hazards",
        "survival_primary_cox",
        "iptw",
        "ipw",
        "causal_primary_iptw",
        "g_computation",
        "treatment_effect",
        "effect_modification",
        "interaction_model",
        # A prespecified cohort-definition sensitivity may legitimately refit
        # the already agent-planned estimand in each declared cohort. The
        # effect product remains required as structural evidence; this method
        # name alone never creates or routes an effect analysis.
        "cohort_definition_sensitivity",
    }
)
_EFFECT_RESULT_OUTPUT_KINDS = frozenset(
    {"artifact", "dataset", "manifest", "model", "statistic", "table"}
)

# A robustness owner may refit an already planned estimand across locked
# specifications.  Its closed product is a typed grid/summary of those refits,
# rather than one primary-effect product.  Keep this separate from the ordinary
# effect vocabulary so a free-standing ``table:robustness_grid`` cannot grant
# effect authority to an unrelated method.
_ROBUSTNESS_EFFECT_CONTRACT_METHODS = frozenset(
    {
        "prespecified_robustness_analysis",
        "association_robustness",
        "prespecified_robustness",
        "robustness_sensitivity",
    }
)
_ROBUSTNESS_EFFECT_CONTRACT_PRODUCTS = frozenset(
    {
        "robustness_grid",
        "robustness_summary",
    }
)


def _typed_effect_result_identities(
    outputs: Sequence[str],
) -> set[Tuple[str, str]]:
    """Return typed result products governed by the shared effect vocabulary."""

    return {
        parsed
        for raw in (outputs or [])
        if (parsed := typed_product(raw)) is not None
        and parsed[0] in _EFFECT_RESULT_OUTPUT_KINDS
        and effect_bearing_product(raw)
    }


def _has_closed_effect_contract_product(outputs: Sequence[str] | str) -> bool:
    """Accept typed effect results plus legacy bare structured product names."""

    values = [outputs] if isinstance(outputs, str) else list(outputs or [])
    return bool(_typed_effect_result_identities(values)) or any(
        effect_bearing_name(name)
        for name in _normalised_structured_output_names(outputs)
    )


_PREDICTION_CONTRACT_METHODS = frozenset(
    {
        *_PLAN_FAMILY_METHODS["prediction_model"],
        *_PLAN_FAMILY_METHODS["dynamic_prediction"],
        "model_evaluation",
        "prediction_model_evaluation",
        "cross_validated_prediction_model",
        "logistic_regression",
        "regularized_logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "elastic_net",
    }
)
_PREDICTION_CONTRACT_PRODUCTS = frozenset(
    {
        "auroc",
        "auc",
        "held_out_auroc",
        "cv_auroc",
        "cv_auroc_mean",
        "mean_auroc",
        "auroc_mean",
        "auroc_median",
        "brier_score",
        "cv_brier_mean",
        "brier_mean",
        "held_out_brier",
        "brier_median",
        "calibration_slope",
        "calibration_slope_median",
        "calibration_intercept",
        "calibration_intercept_median",
        "model_performance",
        "model_performance_train_test",
        "prediction_performance",
        "validation_performance",
        "horizon_performance",
        "time_varying_auroc",
        "prediction_horizon",
    }
)
_PREDICTION_CONTRACT_PRODUCT_PREFIXES = (
    "auroc_",
    "auc_",
    "brier_score_",
    "calibration_slope_",
    "calibration_intercept_",
)

_COHORT_CHANGE_OWNER_METHODS = frozenset(
    {
        "cohort_definition",
        "cohort_definition_and_attrition",
        "primary_cohort_definition",
        "eligibility_definition",
        "cohort_definition_sensitivity",
        "cohort_sensitivity",
        "definition_sensitivity",
        "alternative_cohort_definition",
        "alternative_eligibility_comparison",
        "cohort_overlap",
    }
)
_COHORT_CHANGE_PRODUCTS = frozenset(
    {
        "cohort_flow",
        "cohort_attrition",
        "attrition",
        "attrition_by_rule",
        "locked_cohort",
        "analysis_cohort",
        "alternative_cohort_attrition",
        "cohort_overlap",
        "cohort_overlap_matrix",
        "cohort_overlap_and_attrition",
        "overlap_and_movement_across_cohorts",
    }
)


def _has_closed_contract_product(
    expected_outputs: Sequence[str] | str,
    *,
    products: frozenset[str],
    product_prefixes: Sequence[str] = (),
) -> bool:
    names = _normalised_structured_output_names(expected_outputs)
    return bool(names & products) or any(
        name.startswith(tuple(product_prefixes)) for name in names if product_prefixes
    )


def _signed_standard_effect_output_authorized(
    step: AnalysisStep,
    step_record: Optional[Mapping[str, Any]],
) -> bool:
    """Recognize a host-selected effect owner without widening Coder scope."""

    if not isinstance(step_record, Mapping):
        return False
    candidates = step_record.get("standard_executor_candidates")
    return bool(
        step.method == "verified_association_model_grid"
        and step.planned_analysis_role == "sensitivity"
        and len(step.expected_outputs or []) == 1
        and typed_product(step.expected_outputs[0]) is not None
        and typed_product(step.expected_outputs[0])[0] == "table"
        and any(
            re.fullmatch(r"scientific_runtime_contract:[0-9a-f]{64}", str(ref))
            for ref in (step.icu_rule_refs or [])
        )
        and step_record.get("deterministic_standard_analysis")
        == "association_model_grid"
        and step_record.get("deterministic_standard_selection_reason")
        == "signed_association_model_grid_contract_preflight"
        and isinstance(candidates, Mapping)
        and candidates.get("claimed_by") == "association_model_grid"
    )


def _signed_landmark_survival_effect_output_authorized(
    step: AnalysisStep,
    step_record: Optional[Mapping[str, Any]],
) -> bool:
    """Recognize the digest-bound deterministic survival suite as effect owner."""

    if not isinstance(step_record, Mapping):
        return False
    candidates = step_record.get("standard_executor_candidates")
    typed_outputs = [typed_product(value) for value in step.expected_outputs or ()]
    return bool(
        step.method == "signed_landmark_survival_suite"
        and step.planned_analysis_role == "primary"
        and any(
            product is not None and product[0] == "table" for product in typed_outputs
        )
        and any(
            re.fullmatch(r"scientific_runtime_contract:[0-9a-f]{64}", str(ref))
            for ref in (step.icu_rule_refs or [])
        )
        and step_record.get("deterministic_standard_analysis")
        == "signed_landmark_survival_suite"
        and step_record.get("deterministic_standard_selection_reason")
        == "signed_landmark_survival_suite_contract_preflight"
        and isinstance(candidates, Mapping)
        and candidates.get("claimed_by") == "signed_landmark_survival_suite"
    )


def _signed_landmark_spline_effect_output_authorized(
    step: AnalysisStep,
    step_record: Optional[Mapping[str, Any]],
) -> bool:
    """Recognize the digest-bound deterministic landmark-spline effect owner."""

    if not isinstance(step_record, Mapping):
        return False
    candidates = step_record.get("standard_executor_candidates")
    typed_outputs = [typed_product(value) for value in step.expected_outputs or ()]
    return bool(
        step.method == "signed_landmark_restricted_cubic_spline"
        and step.planned_analysis_role == "primary"
        and any(
            product is not None and product[0] == "table" for product in typed_outputs
        )
        and any(
            re.fullmatch(r"scientific_runtime_contract:[0-9a-f]{64}", str(ref))
            for ref in (step.icu_rule_refs or [])
        )
        and step_record.get("deterministic_standard_analysis")
        == "signed_landmark_spline_association"
        and step_record.get("deterministic_standard_selection_reason")
        == "signed_landmark_spline_contract_preflight"
        and isinstance(candidates, Mapping)
        and candidates.get("claimed_by") == "signed_landmark_spline_association"
    )


def effect_output_authorized(
    step: AnalysisStep,
    *,
    step_record: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return whether the plan authorizes this step to own effect outputs.

    This is the single authorization predicate shared by pre-execution coder
    prompts and post-execution product validation.  Authority requires an exact
    effect-method head *and* a closed, typed effect product, or a non-empty
    planner-owned model-requirement roster (whose schema already fixes both the
    method and product). Free-text intent and inferred analysis-family labels
    never grant authority.
    """

    return (
        _effect_contract_applies(step)
        or bool(getattr(step, "model_requirements", None))
        or association_binary_sensitivity_contract(step) is not None
        or _signed_standard_effect_output_authorized(step, step_record)
        or _signed_landmark_spline_effect_output_authorized(step, step_record)
        or _signed_landmark_survival_effect_output_authorized(step, step_record)
    )


def _effect_contract_applies(step: AnalysisStep) -> bool:
    """Whether this exact method owner declares a result-bearing effect product."""

    method_head = _normalised_method_head(str(step.method or ""))
    outputs = step.expected_outputs or []
    ordinary_effect_contract = (
        method_head in _EFFECT_CONTRACT_METHODS
        and _has_closed_effect_contract_product(outputs)
    )
    robustness_refit_contract = (
        method_head in _ROBUSTNESS_EFFECT_CONTRACT_METHODS
        and _has_closed_contract_product(
            outputs,
            products=_ROBUSTNESS_EFFECT_CONTRACT_PRODUCTS,
        )
    )
    return ordinary_effect_contract or robustness_refit_contract


def _prediction_contract_applies(step: AnalysisStep) -> bool:
    """Whether this exact method owner declares a prediction-performance product."""

    return _normalised_method_head(
        str(step.method or "")
    ) in _PREDICTION_CONTRACT_METHODS and _has_closed_contract_product(
        step.expected_outputs or [],
        products=_PREDICTION_CONTRACT_PRODUCTS,
        product_prefixes=_PREDICTION_CONTRACT_PRODUCT_PREFIXES,
    )


def prediction_contract_applies(step: AnalysisStep) -> bool:
    """Public single-source predicate for a typed prediction owner."""

    return _prediction_contract_applies(step)


def _cohort_change_contract_applies(step: AnalysisStep) -> bool:
    """Whether a cohort owner declares a closed attrition/overlap product."""

    method = str(step.method or "")
    method_matches = _normalised_method_head(
        method
    ) in _COHORT_CHANGE_OWNER_METHODS or _is_primary_analysis_cohort_method(method)
    return method_matches and _has_closed_contract_product(
        step.expected_outputs or [],
        products=_COHORT_CHANGE_PRODUCTS,
    )


def cohort_change_contract_applies(step: AnalysisStep) -> bool:
    """Public single-source predicate for a structured cohort-change owner."""

    return _cohort_change_contract_applies(step)


def _plan_step_owns_contract_family(family: str, step: AnalysisStep) -> bool:
    """Match a scientific plan step by method head plus structured products.

    Free-text mentions are deliberately ignored.  A cohort step that says
    "survival cohort" is not a Cox/KM owner, and an association step that says
    "cluster-robust" is not a phenotype-discovery owner.
    """

    if family == "clustering":
        return _clustering_contract_applies(
            method=str(step.method or ""),
            step_id=str(step.step_id or ""),
            intent=str(step.intent or ""),
            expected_outputs=step.expected_outputs or [],
        )
    if family == "prediction_model":
        return _prediction_contract_applies(step)
    head = _normalised_method_head(str(step.method or ""))
    methods = _PLAN_FAMILY_METHODS.get(family, frozenset())
    products = _PLAN_FAMILY_OUTPUTS.get(family, frozenset())
    outputs = _normalised_expected_output_names(step.expected_outputs or [])
    return head in methods and bool(outputs & products)

