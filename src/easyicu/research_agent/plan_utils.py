"""Plan / step manipulation helpers.

Pure functions that read or transform an :class:`AnalysisPlan` or an
:class:`AnalysisStep` without touching pipeline runtime state. They are
used at three points in the loop:

* before execution — :func:`_enforce_advanced_plan_contract`,
  :func:`_split_table_and_figure_outputs_in_plan`,
  :func:`_ensure_publication_figure_step_in_plan` shape the planner
  output into the contract the runner expects.
* during execution — :func:`_step_produces_figure`,
  :func:`_step_expects_figure`, :func:`_parent_step_id_for_figure_step`,
  :func:`_step_contract_findings` decide what a given step is supposed
  to emit and whether it complied.
* after a replan — :func:`_preserve_figure_steps_after_replan` keeps the
  manuscript / figure scaffolding stable across plan revisions.

Plus a small cluster of predictor inference utilities that read the
research question and turn it into a primary-predictor guess.

Lifted out of :mod:`pipeline` (which still exposes them under the same
underscore-prefixed names by re-import) so the file stays readable.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import ValidationError

from .contracts.declared_product import (
    PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS,
    RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
    _is_primary_analysis_cohort_method,
    _primary_analysis_cohort_attrition_candidate,
    declared_product_contract_findings,
    effect_adjustment_family,
    effect_bearing_name,
    effect_bearing_product,
    effect_estimand_tier,
    effect_measure_family,
    effect_role_family,
    is_failed_step_status,
    typed_product,
)
from .icu_rules import (
    detect_outcome_as_predictor,
    detect_overadjustment,
    outcome_leakage_caution,
    overadjustment_caution,
    treatment_mediator_caution,
)
from .contracts.ordered_stratified import (
    is_ordered_stratified_analysis_step,
    ordered_stratified_structure_findings,
)
from .contracts.table_one import bind_table_one_execution_spec, table_one_output_findings
from .scalar_utils import (
    _first_numeric_scalar_with_key_fragment,
    _first_present_scalar,
    _flatten_scalar_dict,
)
from .schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    ClusterSelectionManifest,
    ResearchContext,
    ValidationFinding,
    VariableRole,
)
from .trajectory.contract import (
    TRAJECTORY_PHENOTYPING_REQUIRED_OUTPUTS,
    trajectory_phenotyping_contract_applies,
)
from .trajectory.plan_contract import trajectory_plan_contract_applies

_WIDE_MEASUREMENT_VALUE_SUFFIXES = (
    "_median",
    "_first",
    "_last",
    "_mean",
    "_max",
    "_min",
    "_sum",
)


def _migrate_render_step_contract(
    child: AnalysisStep,
    source_tokens: Sequence[str],
    *,
    intent: Optional[str] = None,
    method: Optional[str] = None,
) -> AnalysisStep:
    """Rebind one render step and its cardinality contracts atomically."""

    existing = {
        str(contract.input_key): contract
        for contract in child.input_consumption_contracts
    }
    contracts = [
        existing.get(token)
        or ArtifactConsumptionContract(input_key=token, mode="all_rows")
        for token in source_tokens
        if (parsed := typed_product(token)) is not None
        and parsed[0] in {"table", "statistic"}
    ]
    update: Dict[str, Any] = {
        "inputs": list(source_tokens),
        "input_consumption_contracts": contracts,
    }
    if intent is not None:
        update["intent"] = intent
    if method is not None:
        update["method"] = method
    return child.model_copy(update=update)


def _augment_measurement_companion_inputs(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Close structural provenance inputs for selected wide summaries.

    The planner remains the owner of which clinical values a step analyzes.
    Exact count/measured companions are provenance inputs, not new scientific
    choices; add only registered companions and never fuzzy-match concepts.
    """

    available = {str(variable.name) for variable in context.variables}
    revised_steps: List[AnalysisStep] = []
    additions_by_step: Dict[str, List[str]] = {}
    for step in plan.steps or []:
        inputs = [str(value) for value in (step.inputs or [])]
        seen = set(inputs)
        additions: List[str] = []
        for input_name in list(inputs):
            if ":" in input_name:
                continue
            suffix = next(
                (
                    candidate
                    for candidate in _WIDE_MEASUREMENT_VALUE_SUFFIXES
                    if input_name.endswith(candidate)
                ),
                None,
            )
            if suffix is None:
                continue
            base = input_name[: -len(suffix)]
            if not base:
                continue
            for companion in (f"{base}_measured", f"{base}_n"):
                if companion in available and companion not in seen:
                    inputs.append(companion)
                    additions.append(companion)
                    seen.add(companion)
        if additions:
            additions_by_step[str(step.step_id)] = additions
            revised_steps.append(step.model_copy(update={"inputs": inputs}))
        else:
            revised_steps.append(step)
        bind_table_one_execution_spec(revised_steps[-1], context)

    if not additions_by_step:
        return plan, []
    revised = plan.model_copy(update={"steps": revised_steps})
    finding = ValidationFinding(
        validator="planner_input_closure",
        severity="info",
        message=(
            "Added registered count/measured provenance companions for "
            "planner-selected per-stay measurement summaries."
        ),
        detail={
            "reason": "measurement_companion_input_closure",
            "added_inputs_by_step": additions_by_step,
        },
    )
    return revised, [finding]


_REPORT_INPUT_PRODUCT_KINDS = frozenset({"manifest", "statistic", "table"})


def _augment_report_typed_product_inputs(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Bind report steps to prior Planner-declared result products.

    This is structural dependency closure only: the Planner still chooses every
    analysis and product.  A report consumer should not silently recompute those
    results from raw cohort columns when their typed producers fail.
    """

    producer_counts: Dict[Tuple[str, str], int] = {}
    for step in plan.steps or []:
        for output in step.expected_outputs or []:
            product = typed_product(output)
            if product is not None:
                producer_counts[product] = producer_counts.get(product, 0) + 1

    prior_outputs: List[str] = []
    revised_steps: List[AnalysisStep] = []
    additions_by_step: Dict[str, List[str]] = {}
    for step in plan.steps or []:
        is_report = any(
            (product := typed_product(output)) is not None and product[0] == "report"
            for output in step.expected_outputs or []
        )
        inputs = list(step.inputs or [])
        seen = set(inputs)
        additions: List[str] = []
        if is_report:
            for output in prior_outputs:
                product = typed_product(output)
                if (
                    product is None
                    or product[0] not in _REPORT_INPUT_PRODUCT_KINDS
                    or producer_counts.get(product) != 1
                    or output in seen
                ):
                    continue
                inputs.append(output)
                additions.append(output)
                seen.add(output)
        if additions:
            additions_by_step[str(step.step_id)] = additions
            revised_steps.append(step.model_copy(update={"inputs": inputs}))
        else:
            revised_steps.append(step)
        prior_outputs.extend(str(output) for output in step.expected_outputs or [])

    if not additions_by_step:
        return plan, []
    return plan.model_copy(update={"steps": revised_steps}), [
        ValidationFinding(
            validator="planner_input_closure",
            severity="info",
            message=(
                "Bound report consumers to unique prior typed result products "
                "so failed producers cannot be silently recomputed from raw data."
            ),
            detail={
                "reason": "report_typed_product_input_closure",
                "added_inputs_by_step": additions_by_step,
            },
        )
    ]


def _problematic_metric_keys(
    payload: Any,
    fragments: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return metric-like keys that were present but null/non-finite."""

    lowered_fragments = tuple(fragment.lower() for fragment in fragments if fragment)
    if not lowered_fragments:
        return []
    problems: List[Dict[str, Any]] = []

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
            return
        lowered_path = path.lower()
        if not any(fragment in lowered_path for fragment in lowered_fragments):
            return
        bad = False
        if value is None:
            bad = True
        elif isinstance(value, bool):
            bad = False
        elif isinstance(value, (int, float)):
            bad = not math.isfinite(float(value))
        elif isinstance(value, str):
            text = value.strip().lower()
            bad = (
                text in {"", "nan", "none", "null", "model not fitted"}
                or "not fitted" in text
            )
        if bad:
            problems.append({"key": path, "value": value})

    walk(payload)
    return problems


def _parent_step_id_for_figure_step(step: AnalysisStep) -> Optional[str]:
    step_id = str(step.step_id or "")
    if step_id.endswith("_figure") and len(step_id) > len("_figure"):
        return step_id[: -len("_figure")]
    match = re.search(
        r"declared by step ['`]([^'`]+)['`]",
        str(step.intent or ""),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


def _step_expects_figure(step: AnalysisStep) -> bool:
    method = re.sub(r"[^a-z0-9]+", "_", str(step.method or "").strip().lower()).strip(
        "_"
    )
    if method in _FIGURE_METHODS:
        return True
    return any(
        _output_declares_figure(output) for output in step.expected_outputs or []
    )


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


_PRIMARY_COHORT_OWNER_METHODS = frozenset(
    {
        "cohort_construction",
        "cohort_definition",
        "cohort_definition_and_attrition",
        "eligibility_definition",
        "primary_cohort_definition",
    }
)
_PRIMARY_COHORT_OWNER_PRODUCTS = frozenset(
    {
        "analysis_cohort",
        "attrition",
        "attrition_by_rule",
        "cohort_attrition",
        "cohort_denominator",
        "cohort_denominators",
        "cohort_flow",
        "eligibility_flow",
        "locked_cohort",
    }
)


def _step_owns_primary_cohort_contract(step: AnalysisStep) -> bool:
    """Require an exact cohort-owner method and a closed population product."""

    head = _normalised_method_head(str(step.method or ""))
    products = _normalised_structured_output_names(step.expected_outputs or [])
    return head in _PRIMARY_COHORT_OWNER_METHODS and bool(
        products & _PRIMARY_COHORT_OWNER_PRODUCTS
    )


def _plan_expects_analysis_cohort(plan: AnalysisPlan) -> bool:
    """True when the plan clearly intends to *define* an analysis population.

    Scientific ownership comes from the structured method/output contract.  A
    prose mention such as "treatment eligibility bias" is not a cohort owner and
    must never send an effect step through automatic cohort materialisation.
    """
    return any(_step_owns_primary_cohort_contract(step) for step in plan.steps or [])


def _cohort_definition_prose(plan: AnalysisPlan) -> str:
    """Concatenated ``intent`` prose of the plan's cohort-defining step(s).

    This is the free-text 纳排 the agent wrote in lieu of a structured
    ``plan.cohort``; ``cohort_repair`` translates it into typed predicates.
    Uses the same structured owner predicate as
    :func:`_plan_expects_analysis_cohort` so unrelated scientific prose is never
    translated into inclusion/exclusion predicates.
    """
    prose: List[str] = []
    for step in plan.steps or []:
        if _step_owns_primary_cohort_contract(step) and step.intent:
            prose.append(step.intent)
    return "\n".join(prose)


def _cohort_definition_is_empty(plan: AnalysisPlan) -> bool:
    cohort = getattr(plan, "cohort", None)
    if cohort is None:
        return True
    return not (getattr(cohort, "inclusion", ()) or getattr(cohort, "exclusion", ()))


def _cohort_definition_contract_findings(
    plan: AnalysisPlan,
) -> List[ValidationFinding]:
    """Reject a 纳排 that lives only in free-text step intents.

    The planner must express the analysis cohort's inclusion/exclusion as
    structured predicates so the framework can materialise and enforce it
    (``materialize_locked_analysis_cohort``). When the plan implies a cohort
    but ``plan.cohort`` has no structured predicates, the 纳排 is unenforceable
    and unauditable and downstream steps silently run on the full universe.
    Surface that as an error instead of passing silently.
    """
    if not _cohort_definition_is_empty(plan):
        return []
    if not _plan_expects_analysis_cohort(plan):
        return []
    return [
        ValidationFinding(
            validator="cohort_contract",
            severity="error",
            message=(
                "The plan defines an analysis cohort in prose (a cohort / "
                "eligibility / attrition step) but plan.cohort carries no "
                "structured inclusion/exclusion predicates. The 纳排 cannot be "
                "materialised, enforced, or audited, and downstream steps will "
                "run on the full universe. Express the inclusion/exclusion as "
                "typed cohort predicates (concept_id, time_window, aggregation, "
                "op, value)."
            ),
            detail={"cohort": "empty", "expects_cohort": True},
        )
    ]


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


def _normalised_expected_output_names(
    expected_outputs: Sequence[str] | str,
) -> set[str]:
    if isinstance(expected_outputs, str):
        values = re.split(r"[\s,]+", expected_outputs)
    else:
        values = [str(value or "") for value in (expected_outputs or [])]
    names: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        name = value.split(":", 1)[-1].rsplit("/", 1)[-1]
        name = re.sub(r"\.(?:csv|json|parquet|png|svg|pdf|tiff?)$", "", name)
        names.add(name)
    return names


_STRUCTURED_CONTRACT_OUTPUT_KINDS = frozenset(
    {"", "statistic", "table", "model", "manifest", "dataset", "artifact"}
)


def _normalised_structured_output_names(
    expected_outputs: Sequence[str] | str,
) -> set[str]:
    if isinstance(expected_outputs, str):
        values = re.split(r"[\s,]+", expected_outputs)
    else:
        values = [str(value or "") for value in (expected_outputs or [])]
    names: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        parsed = typed_product(value)
        if parsed is not None and parsed[0] in _STRUCTURED_CONTRACT_OUTPUT_KINDS:
            names.add(parsed[1])
            continue
        kind, separator, product = value.partition(":")
        if separator and kind not in _STRUCTURED_CONTRACT_OUTPUT_KINDS:
            continue
        name = (product if separator else kind).rsplit("/", 1)[-1]
        name = re.sub(r"\.(?:csv|json|parquet)$", "", name)
        names.add(name)
    return names


def _normalised_method_head(method: str) -> str:
    """Return the exact normalized method head before an optional ``with`` rider."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(method or "").strip().lower()).strip(
        "_"
    )
    return normalized.split("_with_", 1)[0]


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


def effect_output_authorized(step: AnalysisStep) -> bool:
    """Return whether the plan authorizes this step to own effect outputs.

    This is the single authorization predicate shared by pre-execution coder
    prompts and post-execution product validation.  Authority requires an exact
    effect-method head *and* a closed, typed effect product, or a non-empty
    planner-owned model-requirement roster (whose schema already fixes both the
    method and product). Free-text intent and inferred analysis-family labels
    never grant authority.
    """

    return _effect_contract_applies(step) or bool(
        getattr(step, "model_requirements", None)
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


def _enforce_advanced_plan_contract(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Constrain advanced plan shape while leaving analysis code to the agent."""

    # Fixed-window trajectory plans have a role/DAG contract that supports
    # legitimate agent decomposition. The generic clustering normalizer assumes
    # one method owner and would push all products into that step, recreating a
    # mega-pipeline. Leave this family to the dedicated role normalizer.
    if trajectory_plan_contract_applies(
        plan=plan,
        context=context,
    ) and not any(
        trajectory_phenotyping_contract_applies(context=context, step=step)
        for step in (plan.steps or [])
    ):
        return plan, []

    # Priority: explicit user-declared family > authoritative stamped
    # plan.analysis_type (specific, non-heuristic-reachable families only) >
    # keyword heuristic. The stamped type is consulted before the heuristic so a
    # specific family (survival, dynamic_prediction, causal_inference,
    # treatment_response, validation) wins over the heuristic's looser keyword
    # match (e.g. the bare word "prediction" -> prediction_model). It is gated to
    # families the heuristic cannot reach because infer_analysis_type is itself
    # too loose for the reachable ones (bare "model" -> prediction_model).
    family = _normalise_contract_family(
        context.user_preferences.inferred_analysis_family
        if context.user_preferences
        else None
    )
    if not family:
        stamped = _normalise_contract_family(getattr(plan, "analysis_type", None))
        if stamped and stamped not in _HEURISTIC_REACHABLE_FAMILIES:
            family = stamped
    if not family:
        # An unstamped plan may opt into a contract only through an explicit
        # method family plus a structured product.  Free-text benchmark prose is
        # not an execution routing surface.
        for candidate in (
            "robustness",
            "clustering",
            "bias_audit",
            "prediction_model",
        ):
            if any(
                _plan_step_owns_contract_family(candidate, step)
                for step in (plan.steps or [])
            ):
                family = candidate
                break
    if family not in _CONTRACT_FAMILIES:
        return plan, []

    if family == "prediction_model":
        required_outputs = [
            "statistic:auroc",
            "statistic:brier_score",
            "statistic:baseline_prevalence",
            "statistic:split_strategy",
            "table:model_performance",
            "table:roc_curve",
            "table:calibration_curve",
            "figure:discrimination_calibration",
        ]
    elif family == "clustering":
        required_outputs = [
            "statistic:cluster_count",
            "manifest:cluster_selection",
            "table:cluster_characteristics",
            "figure:clustering_visualization",
            "log:clustering_algorithm_details",
            "manifest:clustering_methodology",
        ]
        if any(
            trajectory_phenotyping_contract_applies(context=context, step=step)
            for step in (plan.steps or [])
        ):
            required_outputs.extend(
                item
                for item in TRAJECTORY_PHENOTYPING_REQUIRED_OUTPUTS
                if item not in required_outputs
            )
    elif family == "survival":
        required_outputs = [
            "statistic:hazard_ratio",
            "statistic:n_events",
            "statistic:median_followup",
            "table:cox_summary",
            "table:km_curve",
            "figure:survival_curves",
            "log:survival_time_definition",
        ]
    elif family == "dynamic_prediction":
        required_outputs = [
            "statistic:time_varying_auroc",
            "statistic:prediction_horizon",
            "table:horizon_performance",
            "figure:time_varying_discrimination",
            "log:anti_leakage_audit",
        ]
    elif family == "causal_inference":
        required_outputs = [
            "statistic:adjusted_effect",
            "statistic:max_smd_after_weighting",
            "table:covariate_balance",
            "table:causal_effect",
            "figure:covariate_balance",
            "log:identification_assumptions",
        ]
    elif family == "treatment_response":
        required_outputs = [
            "statistic:overall_effect",
            "statistic:interaction_pvalue",
            "table:subgroup_effects",
            "figure:subgroup_forest",
            "log:multiplicity_note",
        ]
    elif family == "validation":
        required_outputs = [
            "statistic:validation_auroc",
            "statistic:calibration_slope",
            "table:validation_performance",
            "figure:external_validation",
            "log:validation_cohort_definition",
        ]
    elif family == "bias_audit":
        required_outputs = [
            "statistic:primary_or",
            "statistic:selection_bias_warning",
            "statistic:mortality_rate",
            "table:association_summary",
            "table:missingness_profile",
            "log:clinical_constraint_warning",
        ]
    else:
        required_outputs = [
            "statistic:primary_or",
            "statistic:complete_case_n",
            "table:robustness_summary",
            "figure:robustness_plot",
            "log:missingness_strategy_notes",
        ]

    def _is_relevant(step: AnalysisStep) -> bool:
        return _plan_step_owns_contract_family(family, step)

    relevant_indexes = [
        idx for idx, step in enumerate(plan.steps) if _is_relevant(step)
    ]
    if not relevant_indexes:
        # Family inference is advisory evidence for the planner/critic, not
        # authority to rewrite scientific choices.  If no method-head +
        # structured-product owner exists, preserve every agent step verbatim
        # and surface the mismatch for replanning.  In particular, never turn a
        # cohort step or a mixed-effects association into Cox/IPTW/KMeans merely
        # because free text scored highly for another family.
        return plan, [
            ValidationFinding(
                validator="plan_contract",
                severity="warning",
                message=(
                    f"The inferred {family} family has no agent-declared method "
                    "owner with a compatible structured product contract. The "
                    "plan was preserved unchanged for critic/replanner review; "
                    "the framework did not choose or replace the scientific method."
                ),
                detail={
                    "family": family,
                    "preserved_step_ids": [step.step_id for step in plan.steps],
                    "missing_structured_owner": True,
                    "required_outputs": required_outputs,
                },
            )
        ]

    first_index = relevant_indexes[0]
    relevant_steps = [plan.steps[idx] for idx in relevant_indexes]
    combined_outputs = list(required_outputs)
    # The agent already selected a compatible method. Preserve every additional
    # requested product and add only missing standard machine-readable outputs.
    for step in relevant_steps:
        for item in step.expected_outputs or []:
            if item not in combined_outputs:
                combined_outputs.append(item)

    article_roles = _article_display_roles(plan.steps)
    if family == "robustness" and len(plan.steps or []) > 2 and len(article_roles) >= 4:
        contract_step = _best_contract_step_for_outputs(relevant_steps)
        missing_outputs = [
            item
            for item in required_outputs
            if item not in (contract_step.expected_outputs or [])
        ]
        if not missing_outputs:
            return plan, []
        new_steps: List[AnalysisStep] = []
        for step in plan.steps:
            if step.step_id == contract_step.step_id:
                new_steps.append(
                    step.model_copy(
                        update={
                            "expected_outputs": [
                                *(step.expected_outputs or []),
                                *missing_outputs,
                            ],
                        }
                    )
                )
            else:
                new_steps.append(step)
        revised = plan.model_copy(
            update={"steps": new_steps, "revision": max(1, plan.revision) + 1}
        )
        finding = ValidationFinding(
            validator="plan_contract",
            severity="info",
            message=(
                "Planner output already contains an article-level display suite; "
                "preserved step structure and augmented the robustness contract "
                "instead of collapsing steps."
            ),
            detail={
                "family": family,
                "preserved_step_ids": [step.step_id for step in plan.steps],
                "contract_step_id": contract_step.step_id,
                "article_display_roles": sorted(article_roles),
                "added_outputs": missing_outputs,
            },
        )
        return revised, [finding]

    if len(relevant_indexes) > 1:
        declared_outputs = {
            item for step in relevant_steps for item in (step.expected_outputs or [])
        }
        missing_across_family = [
            item for item in required_outputs if item not in declared_outputs
        ]
        if not missing_across_family:
            return plan, []
        owner_index = relevant_indexes[0]
        new_steps = list(plan.steps)
        owner = new_steps[owner_index]
        new_steps[owner_index] = owner.model_copy(
            update={
                "expected_outputs": [
                    *(owner.expected_outputs or []),
                    *missing_across_family,
                ]
            }
        )
        revised = plan.model_copy(
            update={"steps": new_steps, "revision": max(1, plan.revision) + 1}
        )
        return revised, [
            ValidationFinding(
                validator="plan_contract",
                severity="info",
                message=(
                    f"Preserved the planner's {family} step boundaries and added "
                    "missing machine-readable products to the existing method owner."
                ),
                detail={
                    "family": family,
                    "preserved_step_ids": [step.step_id for step in plan.steps],
                    "contract_step_id": owner.step_id,
                    "added_outputs": missing_across_family,
                },
            )
        ]

    current = relevant_steps[0]
    missing_outputs = [
        item for item in required_outputs if item not in current.expected_outputs
    ]
    needs_normalisation = len(relevant_indexes) != 1 or bool(missing_outputs)
    if not needs_normalisation:
        return plan, []

    # Add products only. Never relabel KMeans as generic clustering, PSM as
    # generic causal inference, or a specific Cox/mixed-effects method as a broad
    # family recipe.
    contract_step = current.model_copy(update={"expected_outputs": combined_outputs})
    new_steps: List[AnalysisStep] = []
    inserted = False
    relevant_set = set(relevant_indexes)
    for idx, step in enumerate(plan.steps):
        if idx in relevant_set:
            if not inserted:
                new_steps.append(contract_step)
                inserted = True
            continue
        new_steps.append(step)

    revised = plan.model_copy(
        update={"steps": new_steps, "revision": max(1, plan.revision) + 1}
    )
    message = (
        f"Planner output for {family} kept its agent-selected method and "
        "step identity while receiving the missing machine-readable metric "
        "and artefact contracts."
    )
    finding = ValidationFinding(
        validator="plan_contract",
        severity="warning",
        message=message,
        detail={
            "family": family,
            "original_step_ids": [step.step_id for step in relevant_steps],
            "contract_step_id": contract_step.step_id,
            "contract_step_index": first_index,
            "required_outputs": required_outputs,
            "converted_from_association": False,
        },
    )
    return revised, [finding]


def _infer_primary_predictor_from_context(
    context: ResearchContext,
) -> Optional[str]:
    """Infer the named exposure/predictor from the question and variables.

    This intentionally stays task-family generic. It scores variable-name
    tokens that appear before adjustment language higher than tokens that
    appear only in an "adjusted for ..." covariate clause.
    """

    question = (context.research_question or "").lower()
    if not question:
        return None
    primary_span = re.split(
        r"\b(?:after adjustment for|adjusted for|controlling for|with adjustment for|including covariates|covariates?)\b",
        question,
        maxsplit=1,
    )[0]
    best_name: Optional[str] = None
    best_score = 0
    for variable in context.variables:
        if variable.role in {VariableRole.ID, VariableRole.TIME, VariableRole.OUTCOME}:
            continue
        tokens = _predictor_tokens(variable.name)
        if not tokens:
            continue
        score = 0
        for token in tokens:
            if token in primary_span:
                score += 20
            elif token in question:
                score += 5
        if score > best_score:
            best_score = score
            best_name = variable.name
    return best_name if best_score > 0 else None


def _predictor_tokens(name: Optional[str]) -> set[str]:
    if not name:
        return set()
    raw_tokens = re.split(r"[^a-zA-Z0-9]+", str(name).lower())
    stop = {
        "",
        "24h",
        "48h",
        "72h",
        "max",
        "min",
        "mean",
        "median",
        "first",
        "last",
        "any",
        "flag",
        "binary",
        "value",
        "score",
    }
    tokens = {token for token in raw_tokens if token not in stop}
    if "vaso" in tokens:
        tokens.add("vasopressor")
    if "norepi" in tokens:
        tokens.add("norepinephrine")
    return tokens


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
_FIGURE_OUTPUT_KINDS = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
_FIGURE_FILE_SUFFIXES = (".png", ".svg", ".pdf", ".tif", ".tiff")


def _output_declares_figure(output: str) -> bool:
    token = str(output or "").strip().lower()
    if not token:
        return False
    kind, separator, name = token.partition(":")
    if separator:
        # A typed declaration's artifact kind is authoritative.  Do not let a
        # table/model product such as ``table:figure_summary`` become a figure
        # merely because its product name contains a presentation word.
        return kind.strip() in _FIGURE_OUTPUT_KINDS and bool(name.strip())
    if token.endswith(_FIGURE_FILE_SUFFIXES):
        return True
    words = set(filter(None, re.split(r"[^a-z0-9]+", token)))
    return bool(words & {"figure", "plot", "chart", "heatmap"})


def _output_declares_auxiliary_log(output: str) -> bool:
    """Return whether an output is an explicitly typed, non-scientific log."""

    parsed = typed_product(output)
    return parsed is not None and parsed[0] == "log"


_RENDER_SOURCE_OUTPUT_KINDS = frozenset({"statistic", "table"})


def _typed_render_source_outputs(outputs: Sequence[str]) -> List[str]:
    """Return exact finalized parent result products a render child may consume.

    Raw artifacts, datasets, and models stay on the scientific parent.  A
    rendering-only child receives only finalized table/statistic products so it
    cannot silently reopen cohort, exposure, outcome, or model decisions.
    """

    render_inputs: List[str] = []
    for raw in outputs or []:
        value = str(raw or "").strip()
        parsed = typed_product(value)
        if parsed is not None and parsed[0] in _RENDER_SOURCE_OUTPUT_KINDS:
            render_inputs.append(value)
    return render_inputs


def _typed_render_source_identities(outputs: Sequence[str]) -> set[Tuple[str, str]]:
    """Return canonical typed identities eligible as scientific render inputs."""

    return {
        parsed
        for raw in (outputs or [])
        if (parsed := typed_product(raw)) is not None
        and parsed[0] in _RENDER_SOURCE_OUTPUT_KINDS
    }


def _effect_figure_semantics_supported_by_inputs(
    *,
    figure_outputs: Sequence[str],
    effect_input_products: set[Tuple[str, str]],
) -> bool:
    """Return whether each effect figure is supported by one bound table.

    Scientific authority is conjunctive and input-local.  A renderer may not
    borrow an OR scale from one sibling product, an ``adjusted`` qualifier from
    another, and a subgroup role from a third.  Every planned effect figure
    must have at least one *actually bound* effect table whose explicit scale,
    role, and adjustment qualifiers support that figure.  A generic figure may
    use a generic effect table, but may not relabel a subgroup/interaction-only
    table as an overall effect.
    """

    effect_figures = [
        raw
        for raw in figure_outputs
        if (parsed := typed_product(raw)) is not None
        and parsed[0] == "figure"
        and effect_bearing_product(raw)
    ]
    if not effect_figures:
        return True

    table_inputs = {
        product for product in effect_input_products if product[0] == "table"
    }
    if not table_inputs:
        return False

    declarations = [f"{kind}:{name}" for kind, name in table_inputs]
    for figure_output in effect_figures:
        output_measure = effect_measure_family(figure_output)
        output_role = effect_role_family(figure_output)
        output_tier = effect_estimand_tier(figure_output)
        output_adjustment = effect_adjustment_family(figure_output)
        supported = False
        for declaration in declarations:
            input_measure = effect_measure_family(declaration)
            input_role = effect_role_family(declaration)
            input_tier = effect_estimand_tier(declaration)
            input_adjustment = effect_adjustment_family(declaration)
            if output_measure is not None and input_measure != output_measure:
                continue
            if output_role is not None:
                if input_role != output_role:
                    continue
            elif input_role is not None:
                # A specialized-only source cannot silently become an overall
                # or otherwise generic effect display.
                continue
            if output_tier is not None:
                if input_tier != output_tier:
                    continue
            elif input_tier in {"secondary", "sensitivity", "corroborative"}:
                # Primary is the default estimand tier for an otherwise generic
                # effect figure. Supporting-only estimates may not silently be
                # promoted into that default role.
                continue
            if output_adjustment is not None and input_adjustment != output_adjustment:
                continue
            supported = True
            break
        if not supported:
            return False
    return True


def _effect_figure_semantics_supported_by_model_roster(
    *,
    step: AnalysisStep,
    figure_outputs: Sequence[str],
    effect_input_products: set[Tuple[str, str]],
) -> bool:
    """Authorize a primary adjusted-effect render from a typed model roster.

    The legacy adjusted-association product name is intentionally generic, but
    a non-empty ``model_requirements`` roster is Planner-owned and fixes the
    single primary model.  It can therefore support only a generic/primary
    adjusted-effect figure (or an explicit OR for a binary logistic primary),
    never a subgroup, interaction, secondary, sensitivity, HR, RR, or RD claim.
    """

    if ("table", "adjusted_association_estimates") not in effect_input_products:
        return False
    primary_requirements = [
        requirement
        for requirement in step.model_requirements or []
        if requirement.analysis_role == "primary"
    ]
    if len(primary_requirements) != 1:
        return False
    primary = primary_requirements[0]
    primary_method = re.sub(
        r"[^a-z0-9]+", "_", str(primary.method_family or "").lower()
    ).strip("_")
    for output in figure_outputs:
        if not effect_bearing_product(output):
            continue
        if effect_role_family(output) is not None:
            return False
        if effect_estimand_tier(output) not in {None, "primary"}:
            return False
        if effect_adjustment_family(output) not in {None, "adjusted"}:
            return False
        measure = effect_measure_family(output)
        if measure is None:
            continue
        if not (
            measure == "odds_ratio"
            and primary.outcome_type == "binary"
            and primary_method in ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES
        ):
            return False
    return True


def _render_only_figure_step_intent(
    *,
    source_step_id: str,
    figure_outputs: Sequence[str],
) -> str:
    """Return the exact framework-authored intent for a split render child."""

    return (
        f"Render the publication figure(s) declared by step "
        f"'{source_step_id}' ({', '.join(figure_outputs)}). Treat this as "
        "a rendering-only step: first read the table/statistic outputs "
        f"produced by '{source_step_id}' from the registered evidence files "
        "or from that step's outputs directory. Do not redefine the "
        "cohort, exposure, outcome, missing-data policy, or model inside "
        "this figure step; if the upstream table cannot support the "
        "requested figure, write a step_summary.json explaining the "
        "missing source-data contract instead of re-analysing "
        "``os.environ['COHORT_PARQUET']``. Save PNG and SVG copies of "
        "every figure with matching stems into "
        "``os.environ['STEP_OUT_DIR']``. Always write a valid "
        "step_summary.json into ``STEP_OUT_DIR`` listing each produced "
        "file under ``figure_files`` even if rendering fails — use a "
        "try/except so the step never aborts before writing the summary."
    )


def _effect_figure_source_authorized(
    *,
    step: AnalysisStep,
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    resolved_input_bindings: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> bool:
    """Authorize only a figure name rendered from a successful effect parent.

    A render child never becomes an effect-method owner. This narrow host-side
    authority permits its planned/registered *figure* name only when the child
    is structurally linked to the latest successful direct parent through an
    exact typed effect-result input. Numeric summaries and non-figure effect
    products remain governed by the ordinary effect-method authorization.
    """

    step_id = str(step.step_id or "")
    output_products = [typed_product(raw) for raw in (step.expected_outputs or [])]
    if (
        _normalised_method_head(str(step.method or "")) not in _FIGURE_METHODS
        or not output_products
        or any(product is None for product in output_products)
        or not any(product[0] == "figure" for product in output_products if product)
        or any(
            product[0] not in {"figure", "log"}
            for product in output_products
            if product
        )
        or not completed_step_records
        or not resolved_input_bindings
    ):
        return False

    child_inputs: List[Tuple[Tuple[str, str], str]] = []
    producer_by_product: Dict[Tuple[str, str], str] = {}
    effect_parent_steps: Dict[str, AnalysisStep] = {}
    for raw in step.inputs or []:
        raw_input = str(raw or "")
        parsed = typed_product(raw_input)
        if parsed is None or parsed[0] not in _RENDER_SOURCE_OUTPUT_KINDS:
            return False
        binding = resolved_input_bindings.get(raw_input)
        if not isinstance(binding, Mapping):
            return False
        binding_product = typed_product(
            f"{binding.get('declared_kind', '')}:{binding.get('product', '')}"
        )
        if (
            binding_product != parsed
            or not str(binding.get("evidence_id") or "").strip()
            or re.fullmatch(
                r"[0-9a-fA-F]{64}", str(binding.get("sha256") or "").strip()
            )
            is None
        ):
            return False
        producer_id = str(binding.get("produced_by_step") or "").strip()
        if not producer_id:
            return False
        prior_producer = producer_by_product.get(parsed)
        if prior_producer is not None and prior_producer != producer_id:
            return False
        producer_by_product[parsed] = producer_id
        child_inputs.append((parsed, producer_id))

    if not child_inputs:
        return False
    latest_records: Dict[str, Mapping[str, Any]] = {}
    for record in completed_step_records:
        if isinstance(record, Mapping):
            record_step_id = str(record.get("step_id") or "").strip()
            if record_step_id:
                latest_records[record_step_id] = record

    effect_input_products: set[Tuple[str, str]] = set()
    for child_product, parent_step_id in child_inputs:
        if step_id == parent_step_id:
            return False
        parent_record = latest_records.get(parent_step_id)
        if (
            parent_record is None
            or str(parent_record.get("status") or "").strip().lower() != "ok"
        ):
            return False
        analysis_request = parent_record.get("analysis_request")
        raw_parent_step = (
            analysis_request.get("step")
            if isinstance(analysis_request, Mapping)
            else None
        )
        if not isinstance(raw_parent_step, Mapping):
            return False
        try:
            parent_step = AnalysisStep.model_validate(raw_parent_step)
        except (TypeError, ValueError, ValidationError):
            return False
        parent_render_products = _typed_render_source_identities(
            parent_step.expected_outputs or []
        )
        if (
            str(parent_step.step_id) != parent_step_id
            or child_product not in parent_render_products
        ):
            return False
        parent_effect_products = _typed_effect_result_identities(
            parent_step.expected_outputs or []
        )
        if child_product in parent_effect_products:
            if not effect_output_authorized(parent_step):
                return False
            effect_input_products.add(child_product)
            effect_parent_steps[parent_step_id] = parent_step

    return bool(
        any(kind == "table" for kind, _product in effect_input_products)
        and (
            _effect_figure_semantics_supported_by_inputs(
                figure_outputs=step.expected_outputs or [],
                effect_input_products=effect_input_products,
            )
            or (
                len(effect_parent_steps) == 1
                and _effect_figure_semantics_supported_by_model_roster(
                    step=next(iter(effect_parent_steps.values())),
                    figure_outputs=step.expected_outputs or [],
                    effect_input_products=effect_input_products,
                )
            )
        )
    )


_PUBLICATION_FIGURE_TRIGGER_TOKENS = (
    "publication-ready figure",
    "publication ready figure",
    "publication figure",
    "produce a heatmap",
    "produce a figure",
    "publication-ready",
    "and a figure",
    "and a heatmap",
    "and a publication",
    "publication-quality figure",
)


def _split_table_and_figure_outputs_in_plan(
    plan: AnalysisPlan,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Split steps that declare both table and figure outputs into two steps.

    A step like ``expected_outputs=['table:table_one', 'figure:table_one_visual']``
    asks the coder agent to produce *both* a CSV table and a publication
    figure inside a single executable script. Naive arms frequently
    deliver only the table and ignore the figure, then exhaust the
    LLM-repair budget without recovering. Splitting the step into a
    table-only step plus a downstream figure-only step gives the agent
    a focused target for each artefact while keeping the analytic
    intent intact.

    The split is conservative: it only fires when a single step
    declares at least one ``table:`` (or ``statistic:``) output *and*
    at least one ``figure:`` output. Non-figure outputs stay on the
    original step; figure outputs migrate to a new appended step
    inserted directly after the original. Other steps in the plan are
    left untouched.
    """
    if not plan.steps:
        return plan, []

    new_steps: List[AnalysisStep] = []
    findings: List[ValidationFinding] = []
    existing_step_ids = {str(step.step_id) for step in plan.steps}
    outputs_by_step = {
        str(step.step_id): list(step.expected_outputs or []) for step in plan.steps
    }
    rehomed_figure_dependencies: Dict[str, Dict[str, str]] = {}

    # A planner may attach a figure to the wrong mixed-output step even though
    # another step declares the figure's exact typed table/statistic product.
    # Repair only that structural, case-neutral identity: ``figure:x`` can be
    # rehomed to the sole ``table:x``/``statistic:x`` producer.  Figures whose
    # names intentionally differ from their source (for example ``love_plot``)
    # remain Planner-owned and are not guessed from keywords.
    render_sources_by_name: Dict[str, List[Tuple[str, str]]] = {}
    for candidate in plan.steps:
        candidate_id = str(candidate.step_id)
        for output in candidate.expected_outputs or []:
            parsed = typed_product(output)
            if parsed is not None and parsed[0] in _RENDER_SOURCE_OUTPUT_KINDS:
                render_sources_by_name.setdefault(parsed[1], []).append(
                    (candidate_id, str(output))
                )

    for step in plan.steps:
        step_id = str(step.step_id)
        for output in list(outputs_by_step[step_id]):
            parsed = typed_product(output)
            if parsed is None or parsed[0] != "figure":
                continue
            exact_sources = render_sources_by_name.get(parsed[1], [])
            if len(exact_sources) != 1:
                continue
            source_step_id, source_output = exact_sources[0]
            if source_step_id == step_id:
                continue
            source_step = next(
                item for item in plan.steps if str(item.step_id) == source_step_id
            )
            source_already_owns_figure = any(
                (candidate_product := typed_product(candidate_output)) is not None
                and candidate_product[0] == "figure"
                for candidate_output in outputs_by_step[source_step_id]
            )
            if (
                f"{source_step_id}_figure" in existing_step_ids
                or source_already_owns_figure
                or _normalised_method_head(str(source_step.method or ""))
                in {"association_robustness", "bias_audit_association", "clustering"}
                or typed_product(source_output)[0] != "table"
            ):
                continue
            outputs_by_step[step_id].remove(output)
            outputs_by_step[source_step_id].append(output)
            rehomed_figure_dependencies.setdefault(source_step_id, {})[
                str(output)
            ] = source_output
            findings.append(
                ValidationFinding(
                    validator="plan_contract",
                    severity="warning",
                    message=(
                        f"Rehomed '{output}' from step '{step_id}' to the sole "
                        f"exact typed source producer '{source_step_id}'."
                    ),
                    detail={
                        "reason": "figure_exact_typed_source_rehome",
                        "figure_output": str(output),
                        "original_step_id": step_id,
                        "source_step_id": source_step_id,
                        "source_output": source_output,
                    },
                )
            )

    typed_product_producers: Dict[Tuple[str, str], Set[str]] = {}
    for candidate in plan.steps:
        for output in outputs_by_step[str(candidate.step_id)]:
            parsed = typed_product(output)
            if parsed is not None:
                typed_product_producers.setdefault(parsed, set()).add(
                    str(candidate.step_id)
                )

    for step in plan.steps:
        outputs = outputs_by_step[str(step.step_id)]
        working_step = (
            step
            if outputs == list(step.expected_outputs or [])
            else step.model_copy(update={"expected_outputs": outputs})
        )
        method = _normalised_method_head(str(working_step.method or ""))
        typed_table_inputs = [
            str(raw_input)
            for raw_input in working_step.inputs
            if (parsed_input := typed_product(raw_input)) is not None
            and parsed_input[0] == "table"
        ]
        if (
            method == "visualization"
            and typed_table_inputs
            and not working_step.input_consumption_contracts
        ):
            working_step = working_step.model_copy(
                update={
                    "input_consumption_contracts": [
                        ArtifactConsumptionContract(
                            input_key=input_key,
                            mode="all_rows",
                        )
                        for input_key in typed_table_inputs
                    ]
                }
            )
            findings.append(
                ValidationFinding(
                    validator="plan_contract",
                    severity="warning",
                    message=(
                        f"Bound visualization step '{working_step.step_id}' to "
                        "preserve all rows from each exact typed table input; "
                        "role-specific row selection requires an explicit Planner "
                        "consumption contract."
                    ),
                    detail={
                        "reason": "visualization_all_rows_consumption_default",
                        "step_id": working_step.step_id,
                        "inputs": typed_table_inputs,
                    },
                )
            )
        if method in {
            "association_robustness",
            "bias_audit_association",
            "clustering",
        }:
            # ``prediction_model`` is intentionally NOT in this skip-list:
            # the canonical ``01_model_training`` step bundles both a
            # ``table:model_performance`` analytic payload and a
            # ``figure:discrimination_calibration`` figure, and the agent
            # frequently forgets to render the figure when both are demanded
            # in a single script. Splitting yields a sibling
            # ``01_model_training_figure`` whose contract is purely visual,
            # which is what
            # ``test_mock_planner_emits_prediction_analysis_and_publication_for_prediction_question``
            # pins.
            new_steps.append(working_step)
            continue
        figure_outputs = [out for out in outputs if _output_declares_figure(out)]
        non_figure_outputs = [out for out in outputs if out not in figure_outputs]
        # Split only when the figure has a typed parent data/model product to
        # consume. A log is a sidecar, not render source data; splitting a
        # ``figure + log`` step would create an empty-input child that can only
        # guess or scan unrelated evidence.
        render_source_outputs = _typed_render_source_outputs(non_figure_outputs)
        explicit_rehomed_dependencies = rehomed_figure_dependencies.get(
            str(step.step_id), {}
        )
        if explicit_rehomed_dependencies:
            # Rehoming is authorized by the exact typed product role that caused
            # the move. Do not widen that closed dependency to every table owned
            # by the producer. Figures requiring a multi-product renderer
            # contract remain Planner-owned and are not inferred here.
            render_source_outputs = list(
                dict.fromkeys(
                    explicit_rehomed_dependencies[figure_output]
                    for figure_output in figure_outputs
                    if figure_output in explicit_rehomed_dependencies
                )
            )
        render_source_identities = {
            parsed
            for output in render_source_outputs
            if (parsed := typed_product(output)) is not None
        }
        sources_have_unique_parent = all(
            typed_product_producers.get(identity) == {str(step.step_id)}
            for identity in render_source_identities
        )
        has_render_source_table = any(
            (parsed := typed_product(output)) is not None and parsed[0] == "table"
            for output in render_source_outputs
        )
        effect_figure_requested = any(
            effect_bearing_product(output) for output in figure_outputs
        )
        effect_source_products = _typed_effect_result_identities(render_source_outputs)
        effect_figure_supported = _effect_figure_semantics_supported_by_inputs(
            figure_outputs=figure_outputs,
            effect_input_products=effect_source_products,
        ) or _effect_figure_semantics_supported_by_model_roster(
            step=step,
            figure_outputs=figure_outputs,
            effect_input_products=effect_source_products,
        )
        if (
            not figure_outputs
            or not has_render_source_table
            or not sources_have_unique_parent
            or (effect_figure_requested and not effect_figure_supported)
        ):
            new_steps.append(working_step)
            continue
        # Keep the original step with the non-figure outputs.
        non_figure_step = working_step.model_copy(
            update={"expected_outputs": non_figure_outputs}
        )
        new_steps.append(non_figure_step)
        # Synthesise a follow-up figure-only step.
        figure_step_id = f"{step.step_id}_figure"
        if figure_step_id in existing_step_ids:
            new_steps[-1] = step
            continue
        figure_intent = _render_only_figure_step_intent(
            source_step_id=str(step.step_id),
            figure_outputs=figure_outputs,
        )
        figure_step = AnalysisStep(
            step_id=figure_step_id,
            planned_analysis_role="auxiliary",
            intent=figure_intent,
            inputs=render_source_outputs,
            expected_outputs=figure_outputs,
            method="visualization",
            icu_rule_refs=list(working_step.icu_rule_refs or [])
            + ["visualization_rule"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=str(input_key),
                    mode="all_rows",
                )
                for input_key in render_source_outputs
                if (parsed_input := typed_product(input_key)) is not None
                and parsed_input[0] == "table"
            ],
        )
        new_steps.append(figure_step)
        findings.append(
            ValidationFinding(
                validator="plan_contract",
                severity="warning",
                message=(
                    f"Split step '{step.step_id}' into a table/statistic "
                    f"step and a follow-up figure step "
                    f"'{figure_step_id}' so the coder can target each "
                    "artefact independently."
                ),
                detail={
                    "source_step_id": step.step_id,
                    "non_figure_outputs": non_figure_outputs,
                    "figure_outputs": figure_outputs,
                    "appended_step_id": figure_step_id,
                },
            )
        )

    if not findings:
        return plan, []
    return plan.model_copy(update={"steps": new_steps}), findings


def _research_question_implies_figure(question: str) -> bool:
    """Heuristic: does the research question call for a figure deliverable?"""
    text = (question or "").lower()
    if not text:
        return False
    if any(token in text for token in _PUBLICATION_FIGURE_TRIGGER_TOKENS):
        return True
    return re.search(r"\bfigure\s+or\b", text) is not None


def _ensure_publication_figure_step_in_plan(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    force: bool = False,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Append a fallback figure step when the planner forgot one.

    Naive arms (no ICU narrative context) sometimes emit a single-step
    plan that omits the publication figure even when the research
    question explicitly asks for one. Task contracts in the EasyICU
    experiment runner still require a ``figure`` artefact in those
    cases. Detect the gap and append a generic figure step so the
    coder agent has a concrete target. The step's ``intent`` is broad
    enough that the coder can still tailor the chart shape (bar, box,
    forest, heatmap…) based on the upstream analytics.

    ``force=True`` bypasses the research-question heuristic: the caller
    already knows a figure will be produced (e.g. the publication-figure
    skill is enabled), so the plan should *declare* the figure even when
    the question text never says "figure". Used by the execute phase,
    where the plan that actually runs is the replanner's — which the
    plan-phase, question-gated guard never sees.
    """
    if any(_step_produces_figure(step) for step in plan.steps or []):
        return plan, []
    if not force and not _research_question_implies_figure(
        context.research_question or ""
    ):
        return plan, []

    # A host guard may request a missing display deliverable, but it must not
    # choose a scientific result by scanning arbitrary run files.  Bind the
    # renderer only to planner-declared table products with a unique producer.
    producer_ids: Dict[Tuple[str, str], Set[str]] = {}
    ordered_table_outputs: List[Tuple[Tuple[str, str], str]] = []
    for candidate in plan.steps or []:
        for raw_output in candidate.expected_outputs or []:
            parsed = typed_product(raw_output)
            if parsed is None or parsed[0] != "table":
                continue
            producer_ids.setdefault(parsed, set()).add(str(candidate.step_id))
            ordered_table_outputs.append((parsed, str(raw_output)))
    render_inputs: List[str] = []
    seen_inputs: Set[Tuple[str, str]] = set()
    for identity, raw_output in ordered_table_outputs:
        if identity in seen_inputs or len(producer_ids.get(identity, set())) != 1:
            continue
        seen_inputs.add(identity)
        render_inputs.append(raw_output)
    if not render_inputs:
        return plan, [
            ValidationFinding(
                validator="plan_contract",
                severity="error",
                message=(
                    "The plan requires a publication figure but declares no "
                    "uniquely produced typed table that a rendering-only step "
                    "can consume. The planner must declare the intended figure "
                    "and its exact typed source instead of asking the runtime "
                    "to scan prior outputs and choose a scientific result."
                ),
                detail={"reason": "missing_typed_figure_source"},
            )
        ]

    next_index = len(plan.steps or []) + 1
    fallback_step = AnalysisStep(
        step_id=f"{next_index:02d}_publication_figure_fallback",
        planned_analysis_role="auxiliary",
        intent=(
            "Render a publication-ready overview using only the exact typed "
            "table inputs bound by the host. Do not scan the run directory, "
            "choose a different result, fit a model, or calculate a new "
            "estimand. Copy every plotted value into a matching source-data "
            "CSV and declare that CSV in the figure contract, then save "
            "the figure as both PNG and SVG with the same stem into "
            "``os.environ['STEP_OUT_DIR']`` (set by the runner). Record "
            "every produced path in step_summary.json under "
            "``figure_files``."
        ),
        method="visualization",
        inputs=render_inputs,
        expected_outputs=["figure:overview"],
        icu_rule_refs=["visualization_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
            for input_key in render_inputs
        ],
    )
    new_steps = list(plan.steps or []) + [fallback_step]
    preserved = plan.model_copy(update={"steps": new_steps})
    findings = [
        ValidationFinding(
            validator="plan_contract",
            severity="warning",
            message=(
                "Plan did not declare a figure step even though the "
                "research question asked for a publication-ready "
                "figure; appended a fallback figure step "
                f"'{fallback_step.step_id}' to preserve the task contract."
            ),
            detail={"appended_step_id": fallback_step.step_id},
        )
    ]
    return preserved, findings


# Mirror of ``evaluation_scorecard._AUDIT_OUTPUT_HINTS`` (kept in sync by hand to
# avoid a plan_utils -> evaluation_scorecard import cycle). A plan "declares an
# audit panel" when a step's intent or expected_outputs contain one of these as
# a complete word or snake-case segment.
_AUDIT_PANEL_TOKENS = ("audit", "completeness", "sensitivity", "leakage", "calibration")


def _step_declares_audit_panel(step: AnalysisStep) -> bool:
    """True if the step declares an audit/sensitivity/robustness display item."""
    for text in [step.intent or "", *(step.expected_outputs or [])]:
        lowered = (text or "").lower()
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered)
            for token in _AUDIT_PANEL_TOKENS
        ):
            return True
    return False


def _ensure_audit_panel_step_in_plan(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Append an audit-panel step when the plan declares none.

    The framework already produces audit/robustness evidence (the locked
    robustness specs, the missingness/data-quality summaries, the causal-audit
    report), but the plan that the replanner grows often never *declares* an
    audit display item, so the manuscript ships without a panel tying those
    checks together. This appends a step that renders one from the existing
    step_summary.json files — no new analysis, just a display of what was
    already computed. Skipped when an audit/sensitivity display is already
    declared.
    """
    if any(_step_declares_audit_panel(step) for step in plan.steps or []):
        return plan, []

    next_index = len(plan.steps or []) + 1
    audit_step = AnalysisStep(
        step_id=f"{next_index:02d}_audit_panel",
        planned_analysis_role="auxiliary",
        intent=(
            "Render an audit panel that summarises the analysis's robustness: "
            "data completeness / missingness, the pre-specified sensitivity / "
            "robustness specifications, and any leakage or calibration checks. "
            "Read the prior step_summary.json files under the run directory, do "
            "not re-run the primary analysis, and save the panel as both PNG and "
            "SVG with the same stem into ``os.environ['STEP_OUT_DIR']`` (set by "
            "the runner). Record every produced path in step_summary.json under "
            "``figure_files``."
        ),
        method="visualization",
        inputs=[],
        expected_outputs=["figure:audit_panel"],
        icu_rule_refs=["visualization_rule"],
    )
    new_steps = list(plan.steps or []) + [audit_step]
    preserved = plan.model_copy(update={"steps": new_steps})
    findings = [
        ValidationFinding(
            validator="plan_contract",
            severity="warning",
            message=(
                "Plan declared no audit / sensitivity display item; appended a "
                f"fallback audit-panel step '{audit_step.step_id}' so the produced "
                "robustness and data-quality evidence is presented."
            ),
            detail={"appended_step_id": audit_step.step_id},
        )
    ]
    return preserved, findings


def _step_produces_figure(step: AnalysisStep) -> bool:
    """True if the step's expected_outputs declare a figure/plot artefact."""
    return any(
        _output_declares_figure(output) for output in step.expected_outputs or []
    )


def _preserve_figure_steps_after_replan(
    *,
    current: AnalysisPlan,
    revised: AnalysisPlan,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Re-add figure-producing steps that the replanner silently dropped.

    The replanner is an LLM call and can rationalise away figure steps when
    upstream context (e.g. ICU narrative) is missing. Task contracts in the
    EasyICU experiment runner still require figure artefacts regardless of
    the planner's framing, so we treat any step whose ``expected_outputs``
    declare a figure/plot as load-bearing: if such a step is present in the
    *current* plan but absent from the *revised* plan, append it back to
    the revised plan and emit a warning so the manifest preserves the audit
    trail.
    """
    revised_ids = {step.step_id for step in revised.steps}
    dropped_figure_steps = [
        step
        for step in current.steps
        if step.step_id not in revised_ids and _step_produces_figure(step)
    ]
    new_steps = list(revised.steps) + list(dropped_figure_steps)

    # A host-split render child carries exact typed inputs from its direct
    # parent.  A replanner may echo the original, pre-normalised parent while
    # dropping the split child.  Re-attaching only the child would then create
    # an impossible DAG: the child asks for products that the echoed parent no
    # longer declares.  Restore only products that were already declared by
    # that same direct parent in ``current``.  This is structural contract
    # preservation, not authority to choose a new table, model, or estimand.
    current_output_owners: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for step in current.steps:
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is not None:
                current_output_owners.setdefault(product, []).append(
                    (str(step.step_id), str(raw_output))
                )

    resulting_producers: Dict[Tuple[str, str], Set[str]] = {}
    for step in new_steps:
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is not None:
                resulting_producers.setdefault(product, set()).add(str(step.step_id))

    result_ids = {str(step.step_id) for step in new_steps}
    current_figure_ids = {
        str(step.step_id) for step in current.steps if _step_produces_figure(step)
    }
    restored_by_parent: Dict[str, List[str]] = {}
    for figure_step in new_steps:
        if str(figure_step.step_id) not in current_figure_ids:
            continue
        parent_id = _parent_step_id_for_figure_step(figure_step)
        if not parent_id or parent_id not in result_ids:
            continue
        for raw_input in figure_step.inputs or []:
            product = typed_product(raw_input)
            if product is None or resulting_producers.get(product):
                continue
            prior_owners = current_output_owners.get(product, [])
            if len(prior_owners) != 1 or prior_owners[0][0] != parent_id:
                continue
            restored_output = prior_owners[0][1]
            restored_by_parent.setdefault(parent_id, []).append(restored_output)
            resulting_producers.setdefault(product, set()).add(parent_id)

    if restored_by_parent:
        repaired_steps: List[AnalysisStep] = []
        for step in new_steps:
            additions = restored_by_parent.get(str(step.step_id), [])
            if not additions:
                repaired_steps.append(step)
                continue
            repaired_steps.append(
                step.model_copy(
                    update={
                        "expected_outputs": list(
                            dict.fromkeys([*(step.expected_outputs or []), *additions])
                        )
                    }
                )
            )
        new_steps = repaired_steps

    if not dropped_figure_steps and not restored_by_parent:
        return revised, []

    preserved = revised.model_copy(update={"steps": new_steps})
    findings: List[ValidationFinding] = []
    if dropped_figure_steps:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="warning",
                message=(
                    "Replanner attempted to drop "
                    f"{len(dropped_figure_steps)} figure-producing step(s); "
                    "they were re-attached to preserve task contract."
                ),
                detail={
                    "preserved_step_ids": [s.step_id for s in dropped_figure_steps],
                },
            )
        )
    if restored_by_parent:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="warning",
                message=(
                    "Restored exact typed outputs on existing direct parent "
                    "steps so preserved figure children retain a valid product DAG."
                ),
                detail={
                    "reason": "preserved_figure_parent_output_contract",
                    "restored_outputs_by_parent": restored_by_parent,
                },
            )
        )
    return preserved, findings


def _step_is_primary_estimand_model(step: AnalysisStep) -> bool:
    """True when ``step`` is a result-bearing PRIMARY model (the estimand).

    Requires the Planner's typed ``primary`` role, a compatible method family,
    and a structured result product. Free-text id/intent tokens and
    preparation-only outputs do not establish ownership of the primary
    estimand.
    """

    if step.planned_analysis_role != "primary":
        return False

    # Exclude only a PURE figure/render child, not a combined model+figure step
    # (which the replanner can emit before the figure/table splitter runs). Both
    # contract helpers below already require a closed result-bearing product, so
    # a combined step that owns the estimand stays primary.
    if _step_is_figure_only(step):
        return False
    # Both helpers normalize only the ``<head>`` of a ``<head>_with_<rider>``
    # method and require a closed result-bearing product.  Thus a legitimate
    # mixed-effects model with a cohort-robust rider remains primary, while a
    # propensity-preparation or audit step cannot qualify through prose.
    return _effect_contract_applies(step) or _prediction_contract_applies(step)


def _step_is_baseline_context_table(step: AnalysisStep) -> bool:
    """True for a structured Table 1 / baseline-context analysis step.

    Match only the step id and declared outputs. Replan repair prose often
    mentions missing baseline context without owning a baseline artifact, so
    intent and free-form method text are deliberately excluded.
    """

    if _step_produces_figure(step):
        return False
    structured = " ".join(
        [step.step_id or "", " ".join(step.expected_outputs or [])]
    ).lower()
    return any(
        token in structured
        for token in (
            "table_one",
            "table one",
            "baseline_context",
            "baseline context",
            "baseline_table",
            "baseline table",
            "baseline_characteristics",
            "baseline characteristics",
        )
    )


def _typed_plan_dependency_graph(
    steps: Sequence[AnalysisStep],
) -> Tuple[Dict[str, Set[str]], List[ValidationFinding]]:
    """Build the unique producer graph for every typed ``kind:product`` input.

    The graph is deliberately method-agnostic.  Scientific methods remain
    planner-owned; this helper only enforces the execution fact that a typed
    input must have one declared producer in the same plan.  Missing and
    ambiguous producers are reported rather than guessed.
    """

    producers: Dict[Tuple[str, str], List[str]] = {}
    findings: List[ValidationFinding] = []
    for step in steps:
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is None:
                continue
            if product[0] not in PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan output uses a product kind that the "
                            "runtime cannot materialise; the plan must be revised "
                            "before execution."
                        ),
                        detail={
                            "reason": "typed_output_kind_not_materializable",
                            "producer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                            "supported_kinds": sorted(
                                PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS
                            ),
                        },
                    )
                )
                continue
            producers.setdefault(product, []).append(step.step_id)

    dependencies: Dict[str, Set[str]] = {step.step_id: set() for step in steps}
    for step in steps:
        for raw_input in step.inputs or []:
            product = typed_product(raw_input)
            if product is None:
                continue
            if product[0] not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan input uses a product kind that the "
                            "runtime cannot bind to current evidence; the plan "
                            "must be revised before execution."
                        ),
                        detail={
                            "reason": "typed_input_kind_not_runtime_bindable",
                            "consumer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                            "supported_kinds": sorted(
                                RUNTIME_BINDABLE_TYPED_INPUT_KINDS
                            ),
                        },
                    )
                )
                continue
            owner_ids = sorted(set(producers.get(product, [])))
            if not owner_ids:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan input has no declared producer; the "
                            "plan must be revised before execution."
                        ),
                        detail={
                            "reason": "typed_input_producer_missing",
                            "consumer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                        },
                    )
                )
                continue
            if len(owner_ids) != 1:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan input has multiple declared producers; "
                            "the framework cannot choose one on the agent's behalf."
                        ),
                        detail={
                            "reason": "typed_input_producer_ambiguous",
                            "consumer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                            "producer_step_ids": owner_ids,
                        },
                    )
                )
                continue
            producer_id = owner_ids[0]
            if producer_id != step.step_id:
                dependencies[step.step_id].add(producer_id)

    # Figure children created by the plan splitter remain paired with their
    # direct parent even when a legacy child omitted its typed table input.
    step_ids = set(dependencies)
    for step in steps:
        if not _step_produces_figure(step):
            continue
        parent_id = _parent_step_id_for_figure_step(step)
        if parent_id in step_ids and parent_id != step.step_id:
            dependencies[step.step_id].add(parent_id)
    return dependencies, findings


def _stable_topological_plan_steps(
    steps: Sequence[AnalysisStep],
    dependencies: Mapping[str, Set[str]],
) -> Tuple[List[AnalysisStep], List[str]]:
    """Return a stable producer-before-consumer order and any cycle members."""

    step_by_id = {step.step_id: step for step in steps}
    original_index = {step.step_id: idx for idx, step in enumerate(steps)}
    active_ids = set(step_by_id)
    remaining = {
        step_id: set(dependencies.get(step_id, set())) & active_ids
        for step_id in active_ids
    }
    dependents: Dict[str, Set[str]] = {step_id: set() for step_id in active_ids}
    for consumer_id, producer_ids in remaining.items():
        for producer_id in producer_ids:
            dependents[producer_id].add(consumer_id)

    ready = sorted(
        (step_id for step_id, producer_ids in remaining.items() if not producer_ids),
        key=lambda step_id: original_index[step_id],
    )
    ordered_ids: List[str] = []
    while ready:
        step_id = ready.pop(0)
        ordered_ids.append(step_id)
        for consumer_id in sorted(
            dependents[step_id], key=lambda value: original_index[value]
        ):
            remaining[consumer_id].discard(step_id)
            if not remaining[consumer_id] and consumer_id not in ordered_ids:
                ready.append(consumer_id)
        ready.sort(key=lambda value: original_index[value])

    cycle_ids = sorted(
        active_ids - set(ordered_ids), key=lambda value: original_index[value]
    )
    if cycle_ids:
        return list(steps), cycle_ids
    return [step_by_id[step_id] for step_id in ordered_ids], []


def _typed_plan_dag_findings(plan: AnalysisPlan) -> List[ValidationFinding]:
    """Validate the generic typed product DAG without choosing any science."""

    steps = list(plan.steps or [])
    dependencies, findings = _typed_plan_dependency_graph(steps)
    original_index = {step.step_id: idx for idx, step in enumerate(steps)}
    for consumer_id, producer_ids in dependencies.items():
        for producer_id in sorted(producer_ids):
            if original_index.get(producer_id, -1) >= original_index.get(
                consumer_id, len(steps)
            ):
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan producer must precede its consumer; the "
                            "plan requires topological repair before execution."
                        ),
                        detail={
                            "reason": "typed_input_producer_not_preceding_consumer",
                            "producer_step_id": producer_id,
                            "consumer_step_id": consumer_id,
                        },
                    )
                )
    _ordered, cycle_ids = _stable_topological_plan_steps(steps, dependencies)
    if cycle_ids:
        findings.append(
            ValidationFinding(
                validator="plan_typed_dag",
                severity="error",
                message=(
                    "The typed plan dependency graph contains a cycle and cannot "
                    "be executed without planner revision."
                ),
                detail={
                    "reason": "typed_dependency_cycle",
                    "cycle_step_ids": cycle_ids,
                },
            )
        )
    return findings


def _cap_plan_preserving_figure_steps(
    *,
    plan: AnalysisPlan,
    cap: int,
    protected_step_ids: Optional[Sequence[str]] = None,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Truncate a plan without orphaning required figure steps.

    Figure-only child steps produced by the splitter are load-bearing only when
    their upstream source step remains in the plan. Treat the parent and child
    as a small dependency unit: a cap may displace another non-figure step to
    keep both, but it must not preserve a figure child by replacing its parent.

    The first genuine primary-estimand model and first structured baseline /
    Table 1 step are article-contract anchors as well. Replan repair steps can
    push these anchors past ``steps[:cap]``; silently dropping either makes a
    busy plan incapable of answering the research question.
    """

    steps = list(plan.steps or [])
    if cap <= 0:
        return plan, []

    # Even a plan already under the numerical cap still needs a stable typed
    # dependency order.  Reordering unique producer edges is structural only;
    # missing, ambiguous, or cyclic edges remain fail-closed findings.
    if len(steps) <= cap:
        dependencies, findings = _typed_plan_dependency_graph(steps)
        ordered, cycle_ids = _stable_topological_plan_steps(steps, dependencies)
        if cycle_ids:
            findings.append(
                ValidationFinding(
                    validator="planner",
                    severity="error",
                    message=(
                        "The plan has a typed dependency cycle; planner revision "
                        "is required before execution."
                    ),
                    detail={
                        "reason": "typed_dependency_cycle",
                        "cycle_step_ids": cycle_ids,
                    },
                )
            )
        elif [step.step_id for step in ordered] != [step.step_id for step in steps]:
            findings.append(
                ValidationFinding(
                    validator="planner",
                    severity="warning",
                    message=(
                        "Reordered plan steps into stable typed producer-before-"
                        "consumer order."
                    ),
                    detail={
                        "reason": "typed_dependency_topological_reorder",
                        "original_step_ids": [step.step_id for step in steps],
                        "reordered_step_ids": [step.step_id for step in ordered],
                    },
                )
            )
        return plan.model_copy(update={"steps": ordered}), findings

    step_by_id = {step.step_id: step for step in steps}
    original_index = {step.step_id: idx for idx, step in enumerate(steps)}
    kept_ids = {step.step_id for step in steps[:cap]}
    protected_ids = {
        str(step_id) for step_id in (protected_step_ids or []) if step_id in step_by_id
    }
    # Role authority is method-family agnostic.  A survival, phenotyping, or
    # causal primary result is just as load-bearing as an association model and
    # must not disappear merely because it sits beyond the numerical cap.
    primary_owner = next(
        (step for step in steps if step.planned_analysis_role == "primary"),
        None,
    )
    if primary_owner is not None:
        protected_ids.add(primary_owner.step_id)
    for predicate in (_step_is_baseline_context_table,):
        owner = next((step for step in steps if predicate(step)), None)
        if owner is not None:
            protected_ids.add(owner.step_id)
    kept_ids.update(protected_ids)
    original_kept_ids = set(kept_ids)
    preserved_step_ids: List[str] = []
    displaced_step_ids: List[str] = []

    def _protected_parent_ids(ids: set[str]) -> set[str]:
        protected: set[str] = set()
        for step_id in ids:
            step = step_by_id.get(step_id)
            if step is None or not _step_produces_figure(step):
                continue
            parent_id = _parent_step_id_for_figure_step(step)
            if parent_id in ids:
                protected.add(parent_id)
        return protected

    def _remove_displaceable(required_ids: set[str]) -> bool:
        protected = set(required_ids) | protected_ids | _protected_parent_ids(kept_ids)
        candidates = [
            step_id
            for step_id in kept_ids
            if step_id not in protected
            and not _step_produces_figure(step_by_id[step_id])
        ]
        if not candidates:
            candidates = [step_id for step_id in kept_ids if step_id not in protected]
        if not candidates:
            return False
        displaced_id = max(candidates, key=lambda sid: original_index.get(sid, -1))
        kept_ids.remove(displaced_id)
        displaced_step_ids.append(displaced_id)
        return True

    # A protected article-contract anchor may sit beyond the initial
    # first-``cap`` slice. Make room immediately rather than relying on a later
    # figure step to happen to trigger eviction.
    while len(kept_ids) > cap:
        if not _remove_displaceable(set()):
            break

    for step in steps[cap:]:
        if not _step_produces_figure(step):
            continue
        parent_id = _parent_step_id_for_figure_step(step)
        required_ids = {step.step_id}
        if parent_id in step_by_id:
            required_ids.add(parent_id)
        if required_ids <= kept_ids:
            continue
        added_ids: List[str] = []
        removed_before = list(displaced_step_ids)
        for required_id in sorted(
            required_ids - kept_ids,
            key=lambda sid: original_index.get(sid, len(steps)),
        ):
            kept_ids.add(required_id)
            added_ids.append(required_id)
        while len(kept_ids) > cap:
            if not _remove_displaceable(required_ids):
                for added_id in added_ids:
                    kept_ids.discard(added_id)
                displaced_step_ids = removed_before
                break
        if step.step_id in kept_ids and step.step_id not in original_kept_ids:
            preserved_step_ids.append(step.step_id)

    # Dependency closure outranks display preservation.  A retained consumer is
    # never allowed to lose its unique typed producer merely to fit one more
    # figure under the cap.
    dependencies, _full_plan_dependency_findings = _typed_plan_dependency_graph(steps)

    def _expand_dependency_closure(ids: Set[str]) -> Set[str]:
        closed = set(ids)
        pending = list(ids)
        while pending:
            consumer_id = pending.pop()
            for producer_id in dependencies.get(consumer_id, set()):
                if producer_id not in closed:
                    closed.add(producer_id)
                    pending.append(producer_id)
        return closed

    kept_ids = _expand_dependency_closure(kept_ids)
    hard_protected_ids = _expand_dependency_closure(set(protected_ids))

    # Remove dependency leaves: first non-protected rendering leaves, then
    # other non-protected leaves.  Removing a consumer can make its producers
    # removable on the next pass, while no retained consumer is orphaned.
    while len(kept_ids) > cap:
        required_as_producer = {
            producer_id
            for consumer_id in kept_ids
            for producer_id in dependencies.get(consumer_id, set())
            if producer_id in kept_ids
        }
        leaf_candidates = [
            step_id
            for step_id in kept_ids
            if step_id not in hard_protected_ids and step_id not in required_as_producer
        ]
        if not leaf_candidates:
            break
        figure_leaves = [
            step_id
            for step_id in leaf_candidates
            if _step_produces_figure(step_by_id[step_id])
        ]
        candidates = figure_leaves or leaf_candidates
        displaced_id = max(candidates, key=lambda sid: original_index.get(sid, -1))
        kept_ids.remove(displaced_id)
        displaced_step_ids.append(displaced_id)

    kept = [step for step in steps if step.step_id in kept_ids]
    _retained_dependencies, dependency_findings = _typed_plan_dependency_graph(kept)
    kept_dependencies = {
        step_id: set(dependencies.get(step_id, set())) & kept_ids
        for step_id in kept_ids
    }
    kept, cycle_ids = _stable_topological_plan_steps(kept, kept_dependencies)
    dropped_ids = [step.step_id for step in steps if step.step_id not in kept_ids]
    dependency_displaced_figure_step_ids = [
        step_id for step_id in preserved_step_ids if step_id not in kept_ids
    ]
    preserved_step_ids = [
        step_id for step_id in preserved_step_ids if step_id in kept_ids
    ]
    capped = plan.model_copy(update={"steps": kept})
    findings = [
        ValidationFinding(
            validator="planner",
            severity="warning",
            message=(
                f"Initial plan had {len(steps)} steps; truncated to "
                f"max_total_steps={cap}. Dropped: "
                f"{', '.join(dropped_ids[:6])}"
                + (" ..." if len(dropped_ids) > 6 else "")
            ),
            detail={
                "dropped_step_ids": dropped_ids,
                "cap": cap,
                "protected_step_ids": sorted(protected_ids),
                "preserved_figure_step_ids": preserved_step_ids,
                "dependency_displaced_figure_step_ids": (
                    dependency_displaced_figure_step_ids
                ),
                "displaced_step_ids": displaced_step_ids,
            },
        )
    ]
    findings.extend(dependency_findings)
    if len(kept_ids) > cap:
        findings.append(
            ValidationFinding(
                validator="planner",
                severity="error",
                message=(
                    "The plan cap cannot be satisfied without dropping a protected "
                    "step or one of its typed producers; planner revision is required."
                ),
                detail={
                    "reason": "typed_dependency_closure_exceeds_cap",
                    "cap": cap,
                    "retained_step_ids": [step.step_id for step in kept],
                    "protected_step_ids": sorted(hard_protected_ids),
                },
            )
        )
    if cycle_ids:
        findings.append(
            ValidationFinding(
                validator="planner",
                severity="error",
                message=(
                    "The retained plan has a typed dependency cycle; planner "
                    "revision is required before execution."
                ),
                detail={
                    "reason": "typed_dependency_cycle",
                    "cycle_step_ids": cycle_ids,
                },
            )
        )
    return capped, findings


_PRIMARY_EFFECT_DIRECT_KEYS = (
    "estimate",
    "statistic:estimate",
    "primary_or",
    "statistic:primary_or",
    "odds_ratio",
    "statistic:odds_ratio",
    "adjusted_or",
    "statistic:adjusted_or",
    "adjusted_odds_ratio",
    "statistic:adjusted_odds_ratio",
    "primary_association_estimate",
    "statistic:primary_association_estimate",
    "association_estimate",
    "statistic:association_estimate",
    "or",
)

_PRIMARY_EFFECT_VALUE_KEYS = (
    "primary_or",
    "odds_ratio",
    "adjusted_odds_ratio",
    "adjusted_or",
    "or",
    "estimate",
    "value",
)

_PRIMARY_EFFECT_CI_LOW_KEYS = (
    "ci_low",
    "ci_lower",
    "lower_ci",
    "ci_lower_95",
    "confidence_interval_low",
)

_PRIMARY_EFFECT_CI_HIGH_KEYS = (
    "ci_high",
    "ci_upper",
    "upper_ci",
    "ci_upper_95",
    "confidence_interval_high",
)


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _first_finite_present_scalar(
    payload: Dict[str, Any], keys: Sequence[str]
) -> Optional[float]:
    value = _first_present_scalar(payload, keys)
    return _finite_float(value)


def _lookup_first_finite(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[float]:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key.lower() not in lowered:
            continue
        numeric = _finite_float(lowered[key.lower()])
        if numeric is not None:
            return numeric
    return None


def _primary_effect_name_matches(source_path: str) -> bool:
    lowered = source_path.lower()
    return bool(
        "primary" in lowered
        or "odds_ratio" in lowered
        or re.search(r"(?:^|[.:\-\[\]])or(?:$|[.:\-\[\]])", lowered)
    )


def _flattened_primary_effect_key_matches(source_path: str) -> bool:
    """True for flattened scalar paths that represent an effect value.

    Generated step summaries sometimes report the primary association as a
    dictionary of per-level effects, for example
    ``primary.adjusted_odds_ratio_sofa.sofa2_5.0``.  The contract only needs
    to know that a finite primary effect was recorded, while avoiding CI
    bounds and p-values from sibling paths such as
    ``primary.adjusted_odds_ratio_sofa_ci95.sofa2_5.0.low``.
    """

    lowered = source_path.lower()
    if any(
        marker in lowered
        for marker in (
            "ci95",
            "_ci",
            ".ci",
            "confidence",
            "p_value",
            "pvalue",
        )
    ):
        return False
    if re.search(
        r"(?:^|[._:\-\[\]])(?:low|high|lower|upper|p|se|stderr)(?:$|[._:\-\[\]])",
        lowered,
    ):
        return False
    return bool(
        "odds_ratio" in lowered
        or "primary_or" in lowered
        or "adjusted_or" in lowered
        or re.search(r"(?:^|[.:\-\[\]])or(?:$|[.:\-\[\]])", lowered)
        or lowered.endswith("_estimate")
        or lowered.endswith(".estimate")
    )


def _primary_effect_from_mapping(
    payload: Mapping[str, Any],
    *,
    require_ci: bool,
) -> Optional[float]:
    effect = _lookup_first_finite(payload, _PRIMARY_EFFECT_VALUE_KEYS)
    if effect is None:
        return None
    if not require_ci:
        return effect
    ci_low = _lookup_first_finite(payload, _PRIMARY_EFFECT_CI_LOW_KEYS)
    ci_high = _lookup_first_finite(payload, _PRIMARY_EFFECT_CI_HIGH_KEYS)
    if ci_low is None or ci_high is None:
        return None
    return effect


def _primary_effect_from_estimates_list(payload: Mapping[str, Any]) -> Optional[float]:
    estimates = payload.get("primary_estimates")
    if not isinstance(estimates, list):
        return None
    for idx, item in enumerate(estimates):
        if not isinstance(item, Mapping):
            continue
        effect = _primary_effect_from_mapping(
            item,
            require_ci=False,
        )
        if effect is not None:
            return effect
    return None


def _primary_effect_from_statistic_dicts(payload: Mapping[str, Any]) -> Optional[float]:
    for key, value in payload.items():
        source_path = str(key)
        if isinstance(value, Mapping):
            if source_path.lower().startswith(
                "statistic:"
            ) and _primary_effect_name_matches(source_path):
                effect = _primary_effect_from_mapping(
                    value,
                    require_ci=True,
                )
                if effect is not None:
                    return effect
            nested = _primary_effect_from_statistic_dicts(value)
            if nested is not None:
                return nested
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if not isinstance(item, Mapping):
                    continue
                nested = _primary_effect_from_statistic_dicts(
                    {f"{source_path}[{idx}]": item}
                )
                if nested is not None:
                    return nested
    return None


def _primary_effect_from_summary(step_summary: Dict[str, Any]) -> Optional[float]:
    effect = _first_finite_present_scalar(step_summary, _PRIMARY_EFFECT_DIRECT_KEYS)
    if effect is not None:
        return effect
    effect = _primary_effect_from_estimates_list(step_summary)
    if effect is not None:
        return effect
    effect = _primary_effect_from_statistic_dicts(step_summary)
    if effect is not None:
        return effect
    for key, value in _flatten_scalar_dict(step_summary).items():
        lowered = key.lower()
        if (
            lowered.endswith("_or")
            or lowered.endswith("_odds_ratio")
            or lowered.endswith("_estimate")
            or _flattened_primary_effect_key_matches(lowered)
        ):
            effect = _finite_float(value)
            if effect is not None:
                return effect
    # Canonical effects must come from structured numeric fields or tables.
    # Free prose is not an evidence contract and can contain ordinary language
    # such as "or 1.5-1.9 times baseline" that resembles an OR label.
    return None


_AUROC_SCALAR_KEYS = (
    "auroc",
    "statistic:auroc",
    "auroc_test",
    "statistic:auroc_test",
    "auc",
    "statistic:auc",
    "held_out_auroc",
    "statistic:held_out_auroc",
    "cv_auroc",
    "statistic:cv_auroc",
    "cv_auroc_mean",
    "statistic:cv_auroc_mean",
    "mean_auroc",
    "auroc_mean",
    "auroc_median",
)


def _prediction_auroc_from_completed_records(
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    *,
    current_step_id: str,
) -> Optional[Tuple[str, float]]:
    """Find an auditable AUROC in a *sibling* completed step's summary.

    This fallback is limited to the prediction requirement: a figure/rendering
    step (e.g. ``*_model_training_figure``)
    often does not re-register the metric under a key its own renderer
    recognises, but the discrimination estimate is genuinely produced and bound
    (``statistic:auroc``) by the upstream training step it renders. When that is
    so, the requirement is satisfied by the sibling step, not missing.
    """
    if not completed_step_records:
        return None
    for record in completed_step_records:
        if not isinstance(record, dict):
            continue
        source_step_id = str(record.get("step_id") or "")
        if not source_step_id or source_step_id == current_step_id:
            continue
        if record.get("status") != "ok":
            continue
        step_summary = record.get("step_summary")
        if not isinstance(step_summary, dict):
            continue
        value = _first_present_scalar(step_summary, _AUROC_SCALAR_KEYS)
        if value is None:
            value = _first_numeric_scalar_with_key_fragment(
                step_summary, ("auroc", "auc")
            )
        if value is not None:
            return source_step_id, value
    return None


_CALIBRATION_SCALAR_KEYS = (
    "brier_score",
    "statistic:brier_score",
    "brier_test",
    "statistic:brier_test",
    "cv_brier_mean",
    "statistic:cv_brier_mean",
    "brier_mean",
    "held_out_brier",
    "statistic:held_out_brier",
    "brier_median",
    "calibration_slope",
    "statistic:calibration_slope",
    "calibration_slope_median",
    "calibration_intercept",
    "statistic:calibration_intercept",
    "calibration_intercept_median",
)


def _prediction_calibration_from_completed_records(
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    *,
    current_step_id: str,
) -> Optional[Tuple[str, float]]:
    """Calibration/Brier analogue of :func:`_prediction_auroc_from_completed_records`."""
    if not completed_step_records:
        return None
    for record in completed_step_records:
        if not isinstance(record, dict):
            continue
        source_step_id = str(record.get("step_id") or "")
        if not source_step_id or source_step_id == current_step_id:
            continue
        if record.get("status") != "ok":
            continue
        step_summary = record.get("step_summary")
        if not isinstance(step_summary, dict):
            continue
        value = _first_present_scalar(step_summary, _CALIBRATION_SCALAR_KEYS)
        if value is None:
            value = _first_numeric_scalar_with_key_fragment(
                step_summary, ("brier", "calibration_slope", "calibration_intercept")
            )
        if value is not None:
            return source_step_id, value
    return None


_CLUSTER_COUNT_SCALAR_KEYS = (
    "n_clusters",
    "statistic:n_clusters",
    "cluster_count",
    "statistic:cluster_count",
)


def _cluster_count_from_summary(payload: Mapping[str, Any]) -> Optional[float]:
    value = _first_present_scalar(dict(payload), _CLUSTER_COUNT_SCALAR_KEYS)
    numeric = _finite_float(value)
    if numeric is None or numeric < 1 or not numeric.is_integer():
        return None
    return numeric


def _cluster_selection_evidence_key(
    payload: Mapping[str, Any],
    *,
    cluster_count: Optional[float] = None,
) -> Tuple[Optional[str], bool]:
    """Return a typed selection manifest or substantive stability mapping.

    Bare strings and paths are declarations, not evidence, and intentionally do
    not satisfy the scientific step contract.  The boolean return value marks
    an explicitly declared but invalid/contradictory selection manifest; callers
    must fail closed instead of laundering it through stability or sibling
    fallback evidence.
    """

    def valid_stability(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        if cluster_count is None:
            return False
        selected_n_clusters = value.get("selected_n_clusters")
        try:
            selected_valid = (
                int(selected_n_clusters) >= 1
                and float(selected_n_clusters).is_integer()
                and int(selected_n_clusters) == int(cluster_count)
            )
        except (TypeError, ValueError):
            selected_valid = False
        if not selected_valid:
            return False
        n_resamples = value.get("n_resamples")
        try:
            n_valid = int(n_resamples) >= 2 and float(n_resamples).is_integer()
        except (TypeError, ValueError):
            n_valid = False
        if not n_valid:
            resamples = value.get("resamples")
            n_valid = isinstance(resamples, list) and len(resamples) >= 2
        metric_keys = {
            "adjusted_rand_index",
            "mean_adjusted_rand_index",
            "stability_score",
            "mean_jaccard",
        }
        has_metric = any(
            str(key).strip().lower().rsplit(".", 1)[-1] in metric_keys
            and _finite_float(child) is not None
            for key, child in _flatten_scalar_dict(value).items()
        )
        return n_valid and has_metric

    def valid_selection(value: Any) -> bool:
        try:
            manifest = ClusterSelectionManifest.model_validate(value)
        except ValidationError:
            return False
        if cluster_count is not None and manifest.selected_n_clusters != int(
            cluster_count
        ):
            return False
        selected_value = next(
            item.criterion_value
            for item in manifest.candidates
            if item.n_clusters == manifest.selected_n_clusters
        )
        candidate_values = [item.criterion_value for item in manifest.candidates]
        if manifest.selection_rule == "minimum":
            return math.isclose(
                selected_value,
                min(candidate_values),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        if manifest.selection_rule == "maximum":
            return math.isclose(
                selected_value,
                max(candidate_values),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        return True

    explicit_manifests: List[Tuple[str, Any]] = []
    stability_alternatives: List[Tuple[str, Any]] = []

    def collect(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).strip().lower()
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in {"cluster_selection", "cluster_selection_manifest"}:
                    explicit_manifests.append((child_path, child))
                if key_text in {"cluster_stability", "stability_evidence"}:
                    stability_alternatives.append((child_path, child))
                collect(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, f"{path}[{index}]")

    collect(payload)
    if explicit_manifests:
        # An explicit manifest is authoritative.  If any declared copy is
        # malformed or contradicts cluster_count, neither a stability mapping in
        # the same summary nor a completed sibling may rescue it.
        if any(not valid_selection(value) for _, value in explicit_manifests):
            return None, True
        return explicit_manifests[0][0], False
    for path, value in stability_alternatives:
        if valid_stability(value):
            return path, False
    return None, False


def _clustering_evidence_from_completed_records(
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    *,
    current_step_id: str,
) -> Tuple[Optional[Tuple[str, float, str]], bool]:
    """Find count plus native selection evidence in a completed sibling step.

    Clustering analog of :func:`_prediction_auroc_from_completed_records`. A
    feature-freeze / figure / rendering step often does not fit clusters
    itself. The genuine clustering step may satisfy the contract with its
    agent-selected native criterion (for example BIC, ICL, gap statistic,
    silhouette, or resampling stability); no one metric family is privileged.
    """
    if not completed_step_records:
        return None, False
    for record in completed_step_records:
        if not isinstance(record, dict):
            continue
        source_step_id = str(record.get("step_id") or "")
        if not source_step_id or source_step_id == current_step_id:
            continue
        if record.get("status") != "ok":
            continue
        step_summary = record.get("step_summary")
        if not isinstance(step_summary, dict):
            continue
        count = _cluster_count_from_summary(step_summary)
        selection_key, explicit_manifest_invalid = _cluster_selection_evidence_key(
            step_summary,
            cluster_count=count,
        )
        if explicit_manifest_invalid:
            return None, True
        if count is not None and selection_key is not None:
            return (source_step_id, count, selection_key), False
    return None, False


_EXPOSURE_PREDICTOR_KEYS = (
    "primary_predictor",
    "predictor",
    "exposure",
    "primary_association_term",
    "primary_term",
)
_ASSOCIATION_EFFECT_KEYS = (
    "primary_or",
    "odds_ratio",
    "adjusted_or",
    "primary_odds_ratio",
    "primary_odds_ratio_per_point",
    "primary_association_estimate",
    "hazard_ratio",
)


def _summary_primary_predictor(step_summary: Mapping[str, Any]) -> str:
    for key in _EXPOSURE_PREDICTOR_KEYS:
        value = step_summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _summary_has_association_effect(step_summary: Mapping[str, Any]) -> bool:
    return any(step_summary.get(key) is not None for key in _ASSOCIATION_EFFECT_KEYS)


def _exposure_names_match(required: str, actual: str) -> bool:
    """Lenient name match: only a *clearly unrelated* predictor counts as wrong.

    Normalises to alphabetic characters, then treats the names as matching on
    a substring or any shared 4-gram. Being lenient means a false *non*-match
    (which would trigger an unnecessary repair) is rare; a genuine swap like
    ``sepsis3`` -> ``sofa_max_int`` shares nothing and is flagged.
    """
    r = re.sub(r"[^a-z]", "", required.lower())
    a = re.sub(r"[^a-z]", "", actual.lower())
    if not r or not a:
        return True
    if r in a or a in r:
        return True
    n = 4
    if len(r) < n or len(a) < n:
        return False
    grams = {r[i : i + n] for i in range(len(r) - n + 1)}
    return any(a[i : i + n] in grams for i in range(len(a) - n + 1))


def _primary_exposure_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: ResearchContext,
) -> List[ValidationFinding]:
    """Flag when the primary association model estimated the wrong exposure.

    When the question names a required primary exposure
    (``context.primary_exposure``) and this step fitted an association model
    whose declared predictor is clearly a different variable, emit an error
    finding. The exposure named in the question is *what the analysis must
    estimate* — modelling a different one answers a different question — so
    this is an objective contract error, not an analytical-preference call
    (it never dictates the model form, covariates, or estimator). Routed
    through the existing contract-repair loop so the agent re-fits in-run
    without restarting the whole pipeline.
    """
    if not isinstance(step_summary, Mapping):
        return []
    required = (getattr(context, "primary_exposure", None) or "").strip()
    if not required:
        return []
    actual = _summary_primary_predictor(step_summary)
    # Only judge the primary model step: it declares a predictor *and* an
    # association-effect estimate. An effect with no declared predictor is the
    # separate "omitted predictor" case handled by the deterministic repairs.
    if not actual or not _summary_has_association_effect(step_summary):
        return []
    if _exposure_names_match(required, actual):
        return []
    return [
        ValidationFinding(
            validator="exposure_contract_auditor",
            severity="error",
            message=(
                f"The question's primary exposure is `{required}`, but this "
                f"primary model estimated `{actual}`. Re-fit the association "
                f"with `{required}` as the primary exposure using the "
                "prespecified representation and measurement semantics. Label "
                "other exposure representations secondary/corroborative and fit "
                "them separately unless the study contract explicitly justifies "
                "including one in the other's adjustment set."
            ),
            detail={
                "kind": "exposure_contract",
                "step_id": step.step_id,
                "required_exposure": required,
                "actual_predictor": actual,
            },
        )
    ]


def _iter_nested_mappings(payload: Any) -> List[Mapping[str, Any]]:
    mappings: List[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        mappings.append(payload)
        for value in payload.values():
            mappings.extend(_iter_nested_mappings(value))
    elif isinstance(payload, list):
        for value in payload:
            mappings.extend(_iter_nested_mappings(value))
    return mappings


def _numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _mapping_number_for_any_key(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[float]:
    lowered_keys = {key.lower() for key in keys}
    for key, value in mapping.items():
        if str(key).lower() not in lowered_keys:
            continue
        number = _numeric_value(value)
        if number is not None:
            return number
    return None


def _summary_has_single_level_exposure(step_summary: Mapping[str, Any]) -> bool:
    text = json.dumps(step_summary, ensure_ascii=False, default=str).lower()
    if any(
        marker in text
        for marker in (
            "no variation",
            "zero variance",
            "single level",
            "single-level",
            "constant exposure",
            "exposure has no variation",
            "singular design",
        )
    ):
        return True
    for mapping in _iter_nested_mappings(step_summary):
        exposed = _mapping_number_for_any_key(
            mapping,
            (
                "exposed_n",
                "exposure_positive_n",
                "positive_n",
                "event_positive_n",
            ),
        )
        unexposed = _mapping_number_for_any_key(
            mapping,
            (
                "unexposed_n",
                "exposure_negative_n",
                "negative_n",
                "event_negative_n",
            ),
        )
        if exposed is None or unexposed is None:
            continue
        total = exposed + unexposed
        if total >= 10 and (
            (exposed == 0 and unexposed > 0) or (unexposed == 0 and exposed > 0)
        ):
            return True
    return False


def _summary_has_measurement_filter_signal(step_summary: Mapping[str, Any]) -> bool:
    for mapping in _iter_nested_mappings(step_summary):
        for key, value in mapping.items():
            lowered = str(key).lower()
            if not any(
                marker in lowered
                for marker in (
                    "unmeasured",
                    "unascertain",
                    "no_source",
                    "no-source",
                    "no_positive_evidence",
                )
            ):
                continue
            number = _numeric_value(value)
            if number is not None and number > 0:
                return True
    return False


def _payload_mentions_required_exposure(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    required: str,
) -> bool:
    actual = _summary_primary_predictor(step_summary)
    if actual and _exposure_names_match(required, actual):
        return True
    blob = " ".join(
        [
            getattr(step, "step_id", None) or "",
            getattr(step, "intent", None) or "",
            json.dumps(step_summary, ensure_ascii=False, default=str),
        ]
    ).lower()
    required_norm = re.sub(r"[^a-z0-9]", "", required.lower())
    blob_norm = re.sub(r"[^a-z0-9]", "", blob)
    return bool(required_norm and required_norm in blob_norm)


def _primary_exposure_measurement_filter_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: ResearchContext,
) -> List[ValidationFinding]:
    """Catch sparse-event exposures collapsed by filtering on measurement flags.

    Some generated scripts treat ``<concept>_measured == 0`` or ``<concept>_n == 0``
    as exposure-missing and remove those stays before modelling. For sparse binary
    event indicators, that often drops event-negative/untriggered patients and
    turns the primary exposure into a constant. This is an objective contract
    failure only when the summary both (1) refers to the question's primary
    exposure and (2) shows a single-level exposure plus a positive unmeasured /
    unascertainable exclusion signal.
    """
    if not isinstance(step_summary, Mapping):
        return []
    required = (getattr(context, "primary_exposure", None) or "").strip()
    if not required:
        return []
    if not _payload_mentions_required_exposure(
        step=step, step_summary=step_summary, required=required
    ):
        return []
    if not _summary_has_single_level_exposure(step_summary):
        return []
    if not _summary_has_measurement_filter_signal(step_summary):
        return []
    return [
        ValidationFinding(
            validator="exposure_contract_auditor",
            severity="error",
            message=(
                f"The primary exposure `{required}` collapsed to a single level "
                "after the step filtered records as unmeasured/unascertainable. "
                "Do not exclude event-negative or untriggered rows solely because "
                "`<concept>_measured == 0` or `<concept>_n == 0`. Rebuild the "
                "binary exposure denominator from the source value columns so "
                "event-absent records remain 0/False unless concept metadata "
                "explicitly says the state is unassessed and uninterpretable. "
                "If the exposure is truly single-level after that audit, report "
                "the model as infeasible with source-data evidence."
            ),
            detail={
                "kind": "exposure_measurement_filter",
                "step_id": step.step_id,
                "required_exposure": required,
            },
        )
    ]


# Coefficient-table detection rides the stable column contract, not a filename:
# runs emit primary_association.csv / model_coefficients.csv / regression_results.csv
# interchangeably, but a model coefficient table always carries a ``variable``
# column plus at least one coefficient-like column. Requiring both excludes
# variable-listing tables that are NOT models (missingness.csv, table_one.csv),
# so they cannot inject phantom covariates into the overadjustment check.
_COEF_TABLE_VALUE_COLUMNS = frozenset(
    {"coef", "beta", "estimate", "log_or", "odds_ratio", "or", "hazard_ratio", "hr"}
)
# A coefficient table's identifier column is named differently across ecosystems:
# statsmodels summary frames use ``variable``; R's broom::tidy and many hand-rolled
# tables use ``term``; others use ``predictor`` / ``covariate`` / ``parameter`` /
# ``feature``. Recognise any of these, but only paired with a coefficient-value
# column (above) — that pairing is what distinguishes a model coefficient table
# from a missingness / table-one CSV, so broadening the id column stays safe.
_COEF_TABLE_ID_COLUMNS = frozenset(
    {"variable", "term", "predictor", "covariate", "parameter", "feature"}
)
_NON_COVARIATE_TERMS = frozenset({"const", "intercept", "(intercept)"})


def read_model_covariate_names(
    directory: Path,
    *,
    files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Variable names from every model coefficient table under ``directory``.

    De-duplicated, intercept rows dropped, first-seen order preserved. Returns
    ``[]`` when the directory is absent or holds no coefficient table — the
    overadjustment check then stays silent rather than guessing. Filename-agnostic:
    a CSV counts as a coefficient table only when its header has a ``variable``
    column and a coefficient-like column, so non-model tables are ignored.
    """
    names: List[str] = []
    base = Path(directory)
    if files is None and not base.exists():
        return names
    candidates = (
        sorted(base.rglob("*.csv"))
        if files is None
        else sorted(
            Path(path)
            for path in files
            if Path(path).is_file() and Path(path).suffix.lower() == ".csv"
        )
    )
    for path in candidates:
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                raw_fields = reader.fieldnames or []
                header = {(h or "").strip().lower() for h in raw_fields}
                if header.isdisjoint(_COEF_TABLE_ID_COLUMNS) or header.isdisjoint(
                    _COEF_TABLE_VALUE_COLUMNS
                ):
                    continue
                # The identifier column actually present (first in file order);
                # read variable names from it rather than assuming ``variable``.
                id_field = next(
                    (
                        h
                        for h in raw_fields
                        if (h or "").strip().lower() in _COEF_TABLE_ID_COLUMNS
                    ),
                    None,
                )
                if id_field is None:
                    continue
                for row in reader:
                    value = (row.get(id_field) or "").strip()
                    if (
                        value
                        and value.lower() not in _NON_COVARIATE_TERMS
                        and value not in names
                        and row.get("term_role") not in ("exposure", "outcome")
                    ):
                        names.append(value)
        except (OSError, ValueError):
            continue
    return names


# A coefficient table is the ground truth of what entered a model, but a run
# that reports only a model-level OR summary (rows = models, cols = OR/CI) never
# writes one — the per-covariate adjustment set then lives only in the analysis
# code. These recover it from the code as a fallback, generally: the patterns
# below are how any statsmodels/patsy analysis declares its adjustment set, and
# every extracted token is routed through the dictionary-driven detectors, so no
# case (exposure / covariate / score) is hard-coded here.
#
# A variable whose name *intends* the adjustment set (covariates, confounders,
# adjustment_cols, predictors, ...) assigned a list/tuple of string column names.
# Names that denote the predictor / adjustment side of a model. Deliberately
# NOT "all model variables" names (``model_vars`` / ``vars`` / ``cols``): those
# bundle the outcome in with the predictors, which would let a study endpoint
# leak into the recovered adjustment set and trip a spurious outcome-leakage
# error. X / design / regressors / rhs exclude the outcome by convention.
_COVARIATE_INTENT_SUBSTRINGS = ("covariate", "covar", "confound", "adjust", "predictor")
_COVARIATE_INTENT_EXACT = frozenset(
    {"x_cols", "design_cols", "regressors", "rhs", "rhs_cols"}
)
# Exclusion/negation markers. A list named for what is deliberately kept OUT of
# the model (``renal_source_not_adjusted``, ``excluded_covariates``,
# ``dropped_for_overadjustment``, the columns of the ``unadjusted`` model) is the
# inverse of the adjustment set. Reading it as the adjustment set inverts its
# meaning and manufactures a phantom overadjustment/leakage finding. These
# markers are unambiguous:
# each *means* "not in the model", so suppressing them cannot hide a genuine
# adjustment set (which is never named this way) — no false-negative risk. Only
# clear negations are listed; transformation words ("drop"/"remove"/"omit") are
# excluded because ``covariates_after_dropping_missing`` can name the final set.
_COVARIATE_EXCLUSION_MARKERS = (
    "not_adjust",
    "notadjust",
    "non_adjust",
    "nonadjust",
    "unadjust",
    "overadjust",
    "exclud",
    "not_covariat",
    "not_confound",
)


def _name_intends_covariates(name: str) -> bool:
    low = name.lower()
    if any(marker in low for marker in _COVARIATE_EXCLUSION_MARKERS):
        return False
    if low in _COVARIATE_INTENT_EXACT:
        return True
    return any(sub in low for sub in _COVARIATE_INTENT_SUBSTRINGS)


def _string_list_elements(node: ast.AST) -> List[str]:
    """String constants in a list/tuple literal, or ``[]`` if not one."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    out: List[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            tok = elt.value.strip()
            if tok:
                out.append(tok)
    return out


def _formula_rhs_terms(formula: str) -> List[str]:
    """Right-hand-side column tokens of a patsy/statsmodels formula string.

    ``"death ~ sepsis3 + age + C(sex) + map_max"`` -> ``[sepsis3, age, sex,
    map_max]``. Interaction (``:`` / ``*``) is split to its main terms; the
    ``C(...)`` categorical wrapper is unwrapped; intercept tokens are dropped.
    The exposure may appear on the RHS — that is fine, the detectors exclude the
    exposure itself.

    Conservative: a term is kept only if it is a clean Python identifier, so
    prose strings that merely contain ``~`` (e.g. a note "adjusted OR ~1.11") do
    not masquerade as a formula — their "terms" are not identifiers and the
    string yields nothing.
    """
    if "~" not in formula:
        return []
    # Require an identifier-ish left-hand side so "OR ~1.11" still parses to a
    # RHS, but the identifier check below is what actually rejects the prose.
    rhs = formula.split("~", 1)[1]
    terms: List[str] = []
    for raw in re.split(r"[+*:]", rhs):
        tok = raw.strip()
        # unwrap C(col), C(col, Treatment(...)) -> col
        m = re.match(r"^[A-Za-z_]\w*\(\s*([A-Za-z_]\w*)", tok)
        if m:
            tok = m.group(1)
        if re.fullmatch(r"[A-Za-z_]\w*", tok) and tok not in ("C", "I"):
            terms.append(tok)
    return terms


def _covariate_names_from_code(
    directory: Path,
    *,
    files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Adjustment-set column names recovered from a run's analysis code.

    General + conservative: parses the analysis ``*.py`` near ``directory`` and
    collects column names from (1) a list/tuple literal assigned to a variable
    whose name intends the adjustment set (``covariates`` / ``confounders`` /
    ``adjustment_cols`` / ``x_cols`` ...) and (2) statsmodels/patsy formula
    strings (the RHS of ``y ~ ...``). Anything it cannot read with confidence is
    skipped (unparseable file, ambiguous slice) so it never invents covariates.
    First-seen order, de-duplicated. Returns ``[]`` when nothing recognisable.
    """
    base = Path(directory)
    seen: List[str] = []

    def _add(tok: str) -> None:
        value = tok.strip()
        if value and value.lower() not in _NON_COVARIATE_TERMS and value not in seen:
            seen.append(value)

    # Search the directory, its parent (a step's outputs/ sits beside analysis.py),
    # and any steps/*/analysis.py beneath it (the post-hoc run-root case). Bounded.
    candidates: List[Path]
    if files is None:
        candidates = []
        for src in (base, base.parent):
            if src.exists():
                candidates.extend(sorted(src.glob("*.py")))
        if base.exists():
            candidates.extend(sorted(base.rglob("analysis.py")))
    else:
        candidates = [
            Path(path)
            for path in files
            if Path(path).is_file() and Path(path).suffix.lower() == ".py"
        ]

    visited: set = set()
    for path in candidates:
        rp = path.resolve()
        if rp in visited or not path.is_file():
            continue
        visited.add(rp)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError):
            continue  # a file we cannot read with confidence is skipped
        for node in ast.walk(tree):
            # (1) covariate-intent list/tuple assignment
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
                if any(_name_intends_covariates(t.id) for t in targets):
                    for tok in _string_list_elements(node.value):
                        _add(tok)
            # (2) formula strings anywhere (y ~ rhs)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "~" in node.value and len(node.value) <= 4000:
                    for tok in _formula_rhs_terms(node.value):
                        _add(tok)
    return seen


def read_adjustment_covariates(
    directory: Path,
    *,
    files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """The model's adjustment set, preferring the coefficient table.

    A per-covariate coefficient table is the ground truth of what entered the
    model, so it wins when present. ``files`` restricts discovery to an explicit
    authority list (for example, current manifest evidence); when omitted the
    historical directory scan is preserved. When a run reports only a
    model-level OR summary (no coefficient table), the adjustment set is
    recovered from the analysis code instead, so the overadjustment / leakage
    auditors are not blind to summary-only outputs. Returns ``[]`` when neither
    source yields anything.
    """
    coef_names = read_model_covariate_names(directory, files=files)
    if coef_names:
        return coef_names
    return _covariate_names_from_code(directory, files=files)


def _primary_exposure_overadjustment_findings(
    *,
    step: AnalysisStep,
    context: ResearchContext,
    out_dir: Path,
) -> List[ValidationFinding]:
    """Hard-block overadjustment: adjusting for a constituent of the exposure.

    When the question names a primary exposure that is a known composite/derived
    score and this step's fitted model conditioned on one of that exposure's
    definitional constituents, emit an error finding routed through the same
    in-run contract-repair loop the exposure-contract auditor uses (re-fit in
    run, no full restart). This is an objective design error — conditioning on a
    component of the exposure nulls the very signal under study — never an
    analytical-preference call: it dictates only the removal of the offending
    constituent from the adjustment set, not the model form, covariates beyond
    the offenders, or estimator. The exposure must be declared
    (``context.primary_exposure``); it is never inferred, so the check stays
    silent rather than guessing.
    """
    exposure = (getattr(context, "primary_exposure", None) or "").strip()
    if not exposure:
        return []
    covariates = read_adjustment_covariates(out_dir)
    covariates = step.without_required_primary_exposure_terms(covariates)
    offenders = detect_overadjustment(exposure, covariates)
    if not offenders:
        # No resolvable constituent matched. If the exposure is nonetheless a
        # derived/composite concept whose inputs could not be resolved (a
        # callback score with an empty dependency closure, e.g. mews/news/sirs),
        # the deterministic check is blind — surface a caution so the risk is
        # not silently passed. A caution is a warning, never a gating error: it
        # prompts the analyst to verify, it does not re-fit or block.
        caution = overadjustment_caution(exposure, covariates)
        if not caution:
            return []
        return [
            ValidationFinding(
                validator="overadjustment_auditor",
                severity="warning",
                message="Overadjustment could not be auto-checked: " + caution,
                detail={
                    "kind": "overadjustment_caution",
                    "step_id": step.step_id,
                    "exposure": exposure,
                    "adjustment_covariates": list(covariates),
                },
            )
        ]
    joined = ", ".join(f"`{o}`" for o in offenders)
    return [
        ValidationFinding(
            validator="overadjustment_auditor",
            severity="error",
            message=(
                f"The primary exposure `{exposure}` is a composite/derived score, "
                f"and this model adjusted for {joined}, which constitute(s) or "
                f"derive(s) it. Conditioning on a component of the exposure removes "
                f"the signal under study (overadjustment). Re-fit dropping {joined} "
                f"from the adjustment set; keep only confounders that are neither "
                f"constituents nor downstream mediators of the exposure."
            ),
            detail={
                "kind": "overadjustment",
                "step_id": step.step_id,
                "exposure": exposure,
                "offending_covariates": list(offenders),
            },
        )
    ]


def _primary_model_leakage_findings(
    *,
    step: AnalysisStep,
    context: ResearchContext,
    out_dir: Path,
) -> List[ValidationFinding]:
    """Outcome-leakage (error) + endpoint/treatment-as-mediator (caution).

    Complements the overadjustment hard-block with two more model-methodology
    checks on this step's fitted covariates, keeping the same impartiality split:

    - ERROR: the declared primary outcome appears among the model's predictors.
      Conditioning a model on its own dependent variable is target leakage by
      construction — an objective design error routed through the same in-run
      re-fit loop (no full restart), like overadjustment.
    - CAUTION (warning, never gates): a *different* endpoint concept used as a
      predictor (timing-dependent leakage), or a treatment/intervention covariate
      that may be a mediator on the exposure→outcome path. Both are defensible
      depending on timing/DAG the auditor cannot see, so they prompt the analyst
      to verify rather than re-fitting or blocking.

    The outcome / exposure must be declared (``context.target_outcome`` /
    ``context.primary_exposure``); they are never inferred, so each check stays
    silent rather than guessing.
    """
    covariates = read_adjustment_covariates(out_dir)
    if not covariates:
        return []
    outcome = (getattr(context, "target_outcome", None) or "").strip()
    exposure = (getattr(context, "primary_exposure", None) or "").strip()
    findings: List[ValidationFinding] = []

    leak = detect_outcome_as_predictor(covariates, study_outcome=outcome)
    if leak:
        joined = ", ".join(f"`{o}`" for o in leak)
        findings.append(
            ValidationFinding(
                validator="outcome_leakage_auditor",
                severity="error",
                message=(
                    f"The declared primary outcome `{outcome}` appears among this "
                    f"model's predictors ({joined}). Conditioning a model on its own "
                    f"dependent variable is target leakage. Re-fit dropping {joined} "
                    f"from the predictors; the outcome must appear only as the "
                    f"dependent variable."
                ),
                detail={
                    "kind": "outcome_leakage",
                    "step_id": step.step_id,
                    "outcome": outcome,
                    "offending_predictors": list(leak),
                },
            )
        )

    endpoint_caution = outcome_leakage_caution(covariates, study_outcome=outcome)
    if endpoint_caution:
        findings.append(
            ValidationFinding(
                validator="outcome_leakage_auditor",
                severity="warning",
                message="Possible outcome leakage: " + endpoint_caution,
                detail={
                    "kind": "outcome_leakage_caution",
                    "step_id": step.step_id,
                    "outcome": outcome,
                    "adjustment_covariates": list(covariates),
                },
            )
        )

    if exposure:
        mediator_caution = treatment_mediator_caution(exposure, covariates)
        if mediator_caution:
            findings.append(
                ValidationFinding(
                    validator="overadjustment_auditor",
                    severity="warning",
                    message="Possible mediator adjustment: " + mediator_caution,
                    detail={
                        "kind": "treatment_mediator_caution",
                        "step_id": step.step_id,
                        "exposure": exposure,
                        "adjustment_covariates": list(covariates),
                    },
                )
            )
    return findings


def _step_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    context: Optional[ResearchContext] = None,
    completed_step_records: Optional[Sequence[Dict[str, Any]]] = None,
    resolved_input_bindings: Optional[Mapping[str, Mapping[str, Any]]] = None,
    out_dir: Optional[Path] = None,
) -> List[ValidationFinding]:
    if not isinstance(step_summary, dict) or not step_summary:
        return [
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} did not produce a readable step_summary.json, "
                    "so required outputs cannot be verified."
                ),
                detail={"step_id": step.step_id},
            )
        ]

    findings: List[ValidationFinding] = []
    reported_status = str(step_summary.get("status") or "").strip().lower()
    if is_failed_step_status(reported_status):
        findings.append(
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} reported status={reported_status!r} "
                    "inside step_summary.json and cannot be recorded as a "
                    "successful completed step."
                ),
                detail={
                    "step_id": step.step_id,
                    "reported_status": reported_status,
                    "blocking_reason": step_summary.get("blocking_reason"),
                    "error": step_summary.get("error"),
                },
            )
        )
    expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
    intent = (step.intent or "").lower()

    # Closed, method-specific declaration for an agent-authored ordered-group
    # descriptive step.  The contract records the agent's variable/order/
    # denominator decisions; a separate cohort replay verifies the numbers.
    findings.extend(
        ordered_stratified_structure_findings(
            step=step,
            step_summary=step_summary,
        )
    )
    findings.extend(table_one_output_findings(step=step, out_dir=out_dir))

    # Figure-only follow-up steps (created by ``_split_table_and_figure_outputs_in_plan``)
    # inherit the parent's step_id with a ``_figure`` suffix, e.g.
    # ``04_primary_association_figure`` / ``01_model_training_figure``. Their
    # expected_outputs contain *only* figure items — the analytic payload
    # (table/statistic/etc.) lives in the sibling parent step. Without this guard
    # the substring matches ``primary_association``/``model_training``/``cluster``
    # below would falsely demand effect/prediction/clustering metrics from a
    # render-only step that legitimately has no such fields in its summary.
    figure_only_step = (
        bool(step.expected_outputs)
        and any(_output_declares_figure(out) for out in step.expected_outputs)
        and all(
            _output_declares_figure(out) or _output_declares_auxiliary_log(out)
            for out in step.expected_outputs
        )
    )
    findings.extend(
        declared_product_contract_findings(
            step=step,
            step_summary=step_summary,
            effect_method_authorized=effect_output_authorized(step),
            effect_figure_source_authorized=_effect_figure_source_authorized(
                step=step,
                completed_step_records=completed_step_records,
                resolved_input_bindings=resolved_input_bindings,
            ),
            out_dir=out_dir,
        )
    )
    from .figures.distribution_availability import (
        distribution_availability_parent_contract_issue,
    )

    distribution_parent_issue = distribution_availability_parent_contract_issue(
        planned_method=step.method,
        parent_out=out_dir,
        parent_summary=step_summary,
        expected_outputs=step.expected_outputs or [],
        planned_inputs=step.inputs or [],
        host_context=context,
    )
    if distribution_parent_issue is not None:
        findings.append(
            ValidationFinding(
                validator="distribution_availability_parent_contract",
                severity="error",
                message=(
                    "The controlled distribution/availability audit did not "
                    "produce the closed parent schema required by its declared "
                    "renderer. Preserve the Planner-selected exposure and write "
                    "the two declared table roles plus their matching summary "
                    "contracts before this step can be successful."
                ),
                detail={
                    "step_id": step.step_id,
                    **distribution_parent_issue,
                },
            )
        )

    # The input parquet is already the locked analysis cohort. A generated
    # downstream QC/model/descriptive script must not relabel itself as a cohort
    # definition/sensitivity step and silently re-run eligibility. Check the
    # plan's own method/id/intent/output contract rather than trusting the
    # generated summary's family (the latter is exactly what can drift).
    cohort_change_authorized = _cohort_change_contract_applies(step)
    summary_family = str(step_summary.get("analysis_family") or "").lower()
    summary_cohort = step_summary.get("cohort_definition")
    summary_claims_cohort_change = summary_family in {
        "cohort_definition",
        "cohort_definition_sensitivity",
        "cohort_sensitivity",
        "definition_sensitivity",
    } or bool(
        isinstance(summary_cohort, dict)
        and summary_cohort.get("current_step_is_cohort_definition_sensitivity")
    )
    if (
        not figure_only_step
        and summary_claims_cohort_change
        and not cohort_change_authorized
    ):
        findings.append(
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} is not a cohort-definition or "
                    "alternative-cohort step, but its summary relabels it as "
                    "cohort-definition sensitivity. Treat COHORT_PARQUET as the "
                    "already locked analysis cohort; remove age, length-of-stay, "
                    "identifier, outcome-availability, and other eligibility "
                    "filters from this step."
                ),
                detail={
                    "kind": "unauthorized_cohort_redefinition",
                    "step_id": step.step_id,
                    "planned_method": step.method,
                    "reported_analysis_family": summary_family or None,
                    "reported_current_step_is_cohort_definition_sensitivity": (
                        summary_cohort.get(
                            "current_step_is_cohort_definition_sensitivity"
                        )
                        if isinstance(summary_cohort, dict)
                        else None
                    ),
                },
            )
        )

    def _append_missing(message: str, keys: Sequence[str]) -> None:
        findings.append(
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=message,
                detail={
                    "step_id": step.step_id,
                    "expected_outputs": list(step.expected_outputs or []),
                    "summary_keys": sorted(step_summary.keys()),
                    "skipped": step_summary.get("skipped"),
                    "error": step_summary.get("error"),
                    "required_keys": list(keys),
                },
            )
        )

    effect_required = not figure_only_step and _effect_contract_applies(step)
    if effect_required:
        effect_value = _primary_effect_from_summary(step_summary)
        if effect_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a primary association "
                    "estimate, but no numeric effect size was recorded."
                ),
                ("estimate", "primary_or", "odds_ratio", "adjusted_or"),
            )

    prediction_required = not figure_only_step and _prediction_contract_applies(step)
    if prediction_required:
        auroc_value = _first_present_scalar(
            step_summary,
            (
                "auroc",
                "statistic:auroc",
                "auc",
                "statistic:auc",
                "held_out_auroc",
                "statistic:held_out_auroc",
                "cv_auroc",
                "statistic:cv_auroc",
                "cv_auroc_mean",
                "statistic:cv_auroc_mean",
                "mean_auroc",
                "auroc_mean",
                "auroc_median",
            ),
        )
        if auroc_value is None:
            auroc_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("auroc", "auc"),
            )
        if auroc_value is None:
            # The discrimination estimate may have been produced and bound by an
            # upstream training step that this (figure/rendering) step renders;
            # mirror the primary-association cross-step fallback so a key-naming
            # mismatch between two steps does not fail the run when the metric is
            # genuinely auditable elsewhere.
            auroc_fallback = _prediction_auroc_from_completed_records(
                completed_step_records,
                current_step_id=str(step.step_id or ""),
            )
            if auroc_fallback is not None:
                source_step_id, _source_auroc = auroc_fallback
                auroc_value = _source_auroc
                findings.append(
                    ValidationFinding(
                        validator="step_contract",
                        severity="warning",
                        message=(
                            f"Step {step.step_id} did not record its own AUROC-style "
                            f"discrimination metric, but the requirement was satisfied "
                            f"by successful step {source_step_id}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "fallback_step_id": source_step_id,
                            "expected_outputs": list(step.expected_outputs or []),
                            "summary_keys": sorted(step_summary.keys()),
                        },
                    )
                )
        if auroc_value is None:
            problematic = _problematic_metric_keys(step_summary, ("auroc", "auc"))
            if problematic:
                keys = ", ".join(str(item["key"]) for item in problematic)
                message = (
                    f"Step {step.step_id} was expected to report AUROC-style "
                    "discrimination. AUROC-like metric keys were present but "
                    f"null/non-finite ({keys}), so the validation model did not "
                    "produce an auditable discrimination estimate."
                )
            else:
                message = (
                    f"Step {step.step_id} was expected to report AUROC-style "
                    "discrimination, but no AUROC metric was recorded."
                )
            _append_missing(
                message,
                ("auroc", "cv_auroc", "mean_auroc", "auroc_median"),
            )
        calibration_value = _first_present_scalar(
            step_summary,
            (
                "brier_score",
                "statistic:brier_score",
                "cv_brier_mean",
                "statistic:cv_brier_mean",
                "brier_mean",
                "held_out_brier",
                "statistic:held_out_brier",
                "brier_median",
                "calibration_slope",
                "statistic:calibration_slope",
                "calibration_slope_median",
                "calibration_intercept",
                "statistic:calibration_intercept",
                "calibration_intercept_median",
            ),
        )
        if calibration_value is None:
            calibration_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("brier", "calibration_slope", "calibration_intercept"),
            )
        if calibration_value is None:
            calibration_fallback = _prediction_calibration_from_completed_records(
                completed_step_records,
                current_step_id=str(step.step_id or ""),
            )
            if calibration_fallback is not None:
                source_step_id, _source_cal = calibration_fallback
                calibration_value = _source_cal
                findings.append(
                    ValidationFinding(
                        validator="step_contract",
                        severity="warning",
                        message=(
                            f"Step {step.step_id} did not record its own calibration/"
                            f"Brier-style metric, but the requirement was satisfied by "
                            f"successful step {source_step_id}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "fallback_step_id": source_step_id,
                            "expected_outputs": list(step.expected_outputs or []),
                            "summary_keys": sorted(step_summary.keys()),
                        },
                    )
                )
        if calibration_value is None:
            problematic = _problematic_metric_keys(
                step_summary,
                ("brier", "calibration_slope", "calibration_intercept"),
            )
            if problematic:
                keys = ", ".join(str(item["key"]) for item in problematic)
                message = (
                    f"Step {step.step_id} was expected to report calibration or "
                    "Brier-style evaluation metrics. Calibration/Brier-like keys "
                    f"were present but null/non-finite ({keys}), so the validation "
                    "model did not produce an auditable calibration estimate."
                )
            else:
                message = (
                    f"Step {step.step_id} was expected to report calibration or "
                    "Brier-style evaluation metrics, but none were recorded."
                )
            _append_missing(
                message,
                (
                    "brier_score",
                    "cv_brier_mean",
                    "held_out_brier",
                    "calibration_slope",
                    "calibration_intercept",
                ),
            )

    # Apply the cluster metric contract only to a method-owned clustering step
    # with declared standard products.  Existing class membership, hospital-level
    # clustering and cluster-robust standard errors are association details, not
    # phenotype-discovery ownership.
    clustering_required = (not figure_only_step) and _clustering_contract_applies(
        method=str(step.method or ""),
        step_id=str(step.step_id or ""),
        intent=str(step.intent or ""),
        expected_outputs=step.expected_outputs or [],
    )
    if clustering_required:
        cluster_count = _cluster_count_from_summary(step_summary)
        selection_key, explicit_manifest_invalid = _cluster_selection_evidence_key(
            step_summary,
            cluster_count=cluster_count,
        )
        if not explicit_manifest_invalid and (
            cluster_count is None or selection_key is None
        ):
            # The clustering estimate may have been produced and bound by a
            # dedicated sibling clustering step that this (figure/rendering or
            # feature-prep) step does not re-register under a recognised key;
            # require both selected cluster count and the agent's native
            # selection/stability evidence from the same successful owner.
            cluster_fallback, sibling_manifest_invalid = (
                _clustering_evidence_from_completed_records(
                    completed_step_records,
                    current_step_id=str(step.step_id or ""),
                )
            )
            if sibling_manifest_invalid:
                explicit_manifest_invalid = True
            elif cluster_fallback is not None:
                source_step_id, source_count, source_selection_key = cluster_fallback
                cluster_count = source_count
                selection_key = source_selection_key
                findings.append(
                    ValidationFinding(
                        validator="step_contract",
                        severity="warning",
                        message=(
                            f"Step {step.step_id} did not record its own "
                            f"cluster count and native selection/stability evidence, "
                            f"but the requirement "
                            f"was satisfied by successful step {source_step_id}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "fallback_step_id": source_step_id,
                            "expected_outputs": list(step.expected_outputs or []),
                            "summary_keys": sorted(step_summary.keys()),
                        },
                    )
                )
        if cluster_count is None or selection_key is None:
            missing = []
            if cluster_count is None:
                missing.extend(("n_clusters", "cluster_count"))
            if selection_key is None:
                missing.extend(
                    (
                        "cluster_selection",
                        "cluster_stability",
                    )
                )
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a clustering summary, "
                    "but it did not record both the selected cluster count and an "
                    "agent-declared native selection/stability criterion."
                ),
                tuple(missing),
            )

    # Enforce figure_required when:
    # (a) the intent *explicitly* demands a publication-ready figure, OR
    # (b) the step is figure-only (its expected_outputs are exclusively
    #     figure tokens — usually the child produced by
    #     ``_split_table_and_figure_outputs_in_plan``).
    # For unsplit mixed steps (figure declared alongside table/statistic
    # outputs without an explicit "publication-ready figure" intent), the
    # splitter handles decomposition in production, so we treat the figure
    # output as an optional companion here. This mirrors how downstream
    # contracts evaluate the parent and the figure-only child separately.
    figure_required = ("publication-ready figure" in intent) or (
        figure_only_step and "figure:" in expected
    )
    if figure_required:
        # When the step itself declares it skipped because the underlying data
        # are unavailable, do not fail the figure contract. The
        # skipped reason is the documented absence; the manuscript binder
        # already treats `skipped` as a first-class signal. Otherwise figure-
        # only steps would block the entire run whenever a sensitivity branch
        # has no eligible cohort.
        _skipped = (
            step_summary.get("skipped") if isinstance(step_summary, dict) else None
        )
        if _skipped:
            return findings
        figure_value = None
        for key, value in _flatten_scalar_dict(step_summary).items():
            lowered_key = key.lower()
            lowered_value = str(value).lower()
            if (
                "figure" in lowered_key
                or "plot" in lowered_key
                or lowered_value.endswith((".png", ".svg", ".pdf", ".tiff", ".tif"))
                or ".png" in lowered_value
                or ".svg" in lowered_value
                or ".pdf" in lowered_value
                or ".tiff" in lowered_value
                or ".tif" in lowered_value
            ):
                figure_value = value
                break
        if figure_value is None:
            # ``_flatten_scalar_dict`` drops lists, but the coder prompt itself
            # recommends recording multiple figure paths in list-valued keys
            # such as ``figure_files`` / ``figure_file`` / ``figure_paths``.
            # Accept those when they contain at least one figure-shaped path.
            for list_key in (
                "figure_files",
                "figure_file",
                "figure_paths",
                "plot_files",
            ):
                candidate = (step_summary or {}).get(list_key)
                if isinstance(candidate, (list, tuple)):
                    candidate_values = []
                    for item in candidate:
                        if isinstance(item, dict):
                            candidate_values.extend(
                                str(value) for value in item.values()
                            )
                        else:
                            candidate_values.append(str(item))
                    if any(
                        value.lower().endswith(
                            (".png", ".svg", ".pdf", ".tiff", ".tif")
                        )
                        for value in candidate_values
                    ):
                        figure_value = candidate
                        break
        if figure_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to produce a figure artifact, "
                    "but the step summary did not record any figure path or figure output."
                ),
                (
                    "figure_path",
                    "figure_files",
                    "figure_file",
                    "plot_path",
                    "png",
                    "svg",
                ),
            )

    return findings


def _primary_analysis_cohort_canonical_schema_rules(
    step: AnalysisStep,
) -> tuple[str, ...]:
    """Return schema rules only for the closed primary-cohort product family.

    These rules describe how to render Planner-owned eligibility decisions; they
    do not choose or modify any cohort predicate.  Exact typed products are the
    routing authority so benchmark prose, step ids, and clinical variable names
    cannot activate the contract.
    """

    if not _primary_analysis_cohort_attrition_candidate(step):
        return ()
    return (
        "Write the declared primary analysis-cohort product with every physical "
        "column from the host-authoritative locked cohort, preserving its exact "
        "ordered row identity and values; additional derived columns are allowed, "
        "but authoritative columns may not be dropped or changed.",
        "Write exact top-level integer fields `n_universe` and "
        "`n_final_analysis_cohort` in step_summary.json; do not hide either "
        "denominator in a nested mapping or under an approximate alias.",
        "For every declared cohort-flow or cohort-attrition table, write exactly "
        "one first `universe` row followed by exactly one row for every "
        "Planner-owned inclusion predicate and then every Planner-owned exclusion "
        "predicate, preserving their declared order. Do not split a predicate "
        "into an additional missingness, unknown-status, or complete-case row.",
        "Each such table must contain the canonical columns `criterion_id`, "
        "`n_at_start_rows`, `n_remaining_rows`, and `n_excluded_rows`. The universe "
        "row starts and remains at `n_universe` with zero excluded; every later "
        "row starts at the previous row's remaining count and satisfies "
        "n_excluded_rows = n_at_start_rows - n_remaining_rows.",
        "Set `criterion_id` to exactly `universe` for the first row. For predicates, "
        "use `{include|exclude}_{order:02d}_{normalized_concept_id}`, with one "
        "1-based order across the Planner inclusion list followed by the exclusion "
        "list; normalize concept_id to lowercase ASCII tokens separated by single "
        "underscores. Use the identical ordered ids and counts in every declared "
        "flow/attrition table.",
    )


def _cohort_predicate_partition_safety_rules(
    step: AnalysisStep,
) -> tuple[str, ...]:
    """Render mechanical safety rules for a declared cohort-flow owner.

    The rules make Planner-owned predicates executable without choosing their
    scientific meaning.  Routing relies on the same closed method/product
    contract used by the host cohort-change gate; prose, benchmark names, and
    variable names cannot activate it.
    """

    if not _cohort_change_contract_applies(step):
        return ()
    return (
        "Before evaluating each Planner-owned numeric eligibility predicate, "
        "coerce its declared value explicitly and build a finite-value mask; "
        "a non-null check alone is insufficient because positive or negative "
        "infinity can otherwise satisfy a threshold.",
        "Never allow a missing, unparseable, or non-finite value to satisfy a "
        "numeric eligibility predicate. Apply only the missing/invalid policy "
        "already declared by the Planner or host contract. If such values are "
        "observed and no policy authorizes their retained/excluded placement, "
        "fail the cohort step closed instead of inventing a scientific rule.",
        "At every predicate stage, construct retained and excluded masks that "
        "are mutually exclusive and exhaustive over the rows at that stage, "
        "and assert n_at_start_rows = n_remaining_rows + n_excluded_rows before "
        "writing outputs. Any optional missing/invalid diagnostic categories "
        "must also be mutually exclusive and exhaustive and must not become "
        "additional Planner predicates.",
    )


def _step_contract_repair_guidance(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    code: str,
    input_bindings: Optional[Mapping[str, Any]] = None,
) -> str:
    guidance: List[str] = []
    if not isinstance(step_summary, dict):
        # Hosted models sometimes emit a bare string as step_summary
        # when the generated code prints JSON as stdout. Treat non-dict
        # summaries as empty for repair guidance so we never crash in
        # the middle of the loop.
        step_summary = {}
    predictor = str(
        step_summary.get("primary_predictor") or step_summary.get("predictor") or ""
    ).strip()
    summary_text = json.dumps(step_summary or {}, ensure_ascii=False, default=str)
    assignment_binding = (
        input_bindings.get("artifact:assignment_model")
        if isinstance(input_bindings, Mapping)
        else None
    )
    assignment_contract = (
        assignment_binding.get("product_contract")
        if isinstance(assignment_binding, Mapping)
        else None
    )
    assignment_models = (
        assignment_contract.get("models")
        if isinstance(assignment_contract, Mapping)
        else None
    )
    if isinstance(assignment_models, list) and len(assignment_models) > 1:
        roster = [
            {
                key: model.get(key)
                for key in (
                    "model_id",
                    "analysis_set",
                    "fit_status",
                    "propensity_score_column",
                    "weight_column",
                )
                if model.get(key) is not None
            }
            for model in assignment_models
            if isinstance(model, Mapping)
        ]
        guidance.append(
            "The digest-bound assignment product is a Planner-owned model roster, "
            "not an ambiguous list from which the engine may choose a primary model. "
            "If its contract declares `diagnostic_model_id` or `selected_model_id`, "
            "use that exact entry. Otherwise compute and report the planned diagnostic "
            "separately for every fitted roster entry, keyed by its `model_id` and "
            "`analysis_set`; do not choose the first row, collapse variants, refit, or "
            "imply that one is primary. Preserve each entry's exact declared propensity "
            "and weight columns and its own analysis-set denominator. "
            "Current typed roster facts: "
            + json.dumps(roster, ensure_ascii=False, sort_keys=True)
        )
    guidance.extend(_primary_analysis_cohort_canonical_schema_rules(step))
    guidance.extend(_cohort_predicate_partition_safety_rules(step))
    if is_ordered_stratified_analysis_step(step):
        guidance.append(
            "Keep this as an agent-authored ordered-stratified analysis, but call "
            "the documented wilson_interval, cochran_armitage_trend, and "
            "jonckheere_terpstra_trend primitives. Use explicit CA scores, "
            "individual-level values for JT, nonzero bounded p-values with log-p "
            "metadata, one two-test Holm family, the canonical flat CSV columns, "
            "and a complete ordered_stratified_contract declaration. Spearman "
            "must not be substituted or relabelled as JT."
        )
    if predictor and predictor in summary_text:
        guidance.append(
            f"The machine summary identifies `{predictor}` as the primary predictor. "
            f"The repaired script must include `{predictor}` in the fitted design matrix."
        )
        lookup_patterns = (
            f"result.params['{predictor}'",
            f'result.params["{predictor}"',
            f"result.conf_int().loc['{predictor}'",
            f'result.conf_int().loc["{predictor}"',
            f"result.pvalues['{predictor}'",
            f'result.pvalues["{predictor}"',
            f"coef_table.loc['{predictor}'",
            f'coef_table.loc["{predictor}"',
        )
        if any(pattern in code for pattern in lookup_patterns):
            guidance.append(
                f"The previous script read model results for `{predictor}`. "
                f"Before fitting, build `x_cols` so `{predictor}` is present in `X.columns`; "
                "otherwise statsmodels will fit a model that cannot report the requested coefficient."
            )
    if "pd.get_dummies" in code and "drop_first" in code:
        guidance.append(
            "The previous script used dummy encoding. Rebuild the predictor list after "
            "dummy encoding: primary predictor + numeric covariates + generated dummy columns."
        )
    if (
        (
            (step_summary or {}).get("n_total") == 0
            or "zero-size array" in summary_text.lower()
            or "empty" in summary_text.lower()
        )
        and "pd.to_numeric" in code
        and "sex" in code
    ):
        guidance.append(
            "The previous script appears to have dropped the entire cohort by applying "
            "`pd.to_numeric(..., errors='coerce')` to `sex` before dummy encoding. "
            "Repair preprocessing by dummy-encoding `sex` first, rebuilding `x_cols`, "
            "then numeric-coercing only `[outcome] + x_cols` and dropping missing rows "
            "with that rebuilt list."
        )
        guidance.append(
            "Do not keep a null estimate summary for this contract failure. The repair "
            "should produce a numeric odds ratio when enough non-missing rows/events exist."
        )
    if (
        "pandas data cast to numpy dtype of object" in summary_text.lower()
        or "dtype of object" in summary_text.lower()
    ) and ("sm.logit(" in code.lower() or "pd.get_dummies" in code):
        guidance.append(
            "The prior script passed an object-dtype design matrix into statsmodels. "
            "After `pd.get_dummies(...)`, rebuild the predictor frame and convert every "
            "column in `X` with `pd.to_numeric(..., errors='coerce')`, cast boolean "
            "dummy columns to int when needed, and fit `sm.Logit(y, X.astype(float))`."
        )
        guidance.append(
            "Check the final design matrix dtypes before fitting and keep only rows with "
            "non-missing numeric predictors/outcome so the repaired script writes a "
            "non-null odds ratio."
        )
    expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
    effect_required = _effect_contract_applies(step)
    if effect_required:
        guidance.append(
            "This association step must write a non-null numeric primary effect "
            "estimate in step_summary.json, such as `adjusted_or`, `primary_or`, "
            "`odds_ratio`, or `primary_association_estimate`. Do not satisfy the "
            "contract by leaving association fields null."
        )
    prediction_required = _prediction_contract_applies(step)
    if prediction_required:
        guidance.append(
            "This prediction step must produce numeric AUROC and Brier/calibration metrics "
            "in step_summary.json (for example `cv_auroc_mean` and `brier_score`). "
            "Do not return only null metrics unless validation is truly impossible."
        )
        if "could not convert string to float" in summary_text.lower() or (
            "passthrough" in code and "onehot" in code.lower()
        ):
            guidance.append(
                "The failure indicates a categorical variable reached a numeric estimator. "
                "Use a scikit-learn ColumnTransformer with numeric features in a median-impute/"
                "scale branch and categorical features in a most-frequent-impute + "
                "OneHotEncoder(handle_unknown='ignore', sparse_output=False) branch. "
                "Never use `('onehot', 'passthrough')` for the categorical branch."
            )
        if "pd.to_numeric" in code and "categorical" in code.lower():
            guidance.append(
                "Do not numeric-coerce the full mixed feature frame. Keep categorical "
                "columns such as sex as object/string until the categorical transformer "
                "encodes them."
            )
    if "simpleimputer does not support data with dtype bool" in summary_text.lower():
        guidance.append(
            "A boolean dummy column reached SimpleImputer. Cast boolean dummy columns "
            "to int before fitting scikit-learn pipelines, or route them through a "
            "numeric branch with median imputation after conversion."
        )
    clustering_required = _clustering_contract_applies(
        method=str(step.method or ""),
        step_id=str(step.step_id or ""),
        intent=str(step.intent or ""),
        expected_outputs=step.expected_outputs or [],
    )
    if clustering_required:
        guidance.append(
            "This clustering step must write the selected `cluster_count` (or "
            "`n_clusters`) and its agent-declared native selection/stability "
            "evidence in step_summary.json. Record a full `cluster_selection` "
            "mapping (criterion, rule/direction, selected k, and at least two "
            "finite candidate values), or a substantive `cluster_stability` "
            "mapping with at least two resamples and a finite stability metric. "
            "A bare criterion string or artifact path does not satisfy this gate. "
            "Use the method-appropriate evidence (for example BIC/AIC/ICL, gap "
            "statistic, resampling stability, or silhouette when appropriate)."
        )
        guidance.append(
            "Keep clustering self-contained: create labels, cluster characteristics, "
            "method/selection metadata, and the clustering figure inside this "
            "script. Add descriptive outcomes only when the plan declares them; "
            "do not rely on labels saved by another step."
        )
        guidance.append(
            "Also save a table artefact named `cluster_characteristics.csv` and "
            "the declared cluster-selection manifest so manuscript evidence aliases bind."
        )
    if "figure:" in expected:
        guidance.append(
            "This step declares a figure output. Save a real figure file such as PNG/SVG/"
            "PDF/TIFF and record its path in step_summary.json using a key such as "
            "`figure_path`, `figure_file`, or `figure_files`."
        )
        guidance.append(
            "In every top-level FigureContract, `source_data` must be one local CSV "
            "basename string or a flat list of local CSV basename strings from the "
            "current step output directory. Never write a dict, list of dicts, "
            "evidence object, absolute path, or path metadata there; put evidence ids "
            "in panel `evidence_ids` and other provenance in step_summary metadata."
        )
    if not guidance:
        guidance.append(
            "Repair the script so each expected output is written as machine-readable "
            "numbers in step_summary.json, or write a precise skipped/error reason."
        )
    return "\n".join(f"- {item}" for item in guidance)
