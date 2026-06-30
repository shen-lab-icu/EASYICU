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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .icu_rules import (
    detect_outcome_as_predictor,
    detect_overadjustment,
    outcome_leakage_caution,
    overadjustment_caution,
    treatment_mediator_caution,
)
from .scalar_utils import (
    _first_numeric_effect_from_text,
    _first_numeric_scalar_with_key_fragment,
    _first_present_scalar,
    _flatten_scalar_dict,
)
from .schema import (
    AnalysisPlan,
    AnalysisStep,
    ResearchContext,
    ValidationFinding,
    VariableRole,
)


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
    if "figure" in str(step.method or "").lower():
        return True
    return any(
        "figure" in str(output or "").lower() or "plot" in str(output or "").lower()
        for output in (step.expected_outputs or [])
    )


# Markers that a step is *defining* the analysis population (not merely
# describing "the cohort"). The bare word "cohort" is deliberately excluded —
# it appears in descriptive intents ("cohort table", "describe the cohort")
# that do not apply inclusion/exclusion, and would false-positive the contract.
_COHORT_DEFINITION_MARKERS = (
    "cohort_def",  # step_id: 01_cohort_definition
    "cohort definition",
    "attrition",
    "eligib",  # eligibility / eligible
    "inclusion criteria",
    "exclusion criteria",
    "inclusion/exclusion",
    "纳排",
    "纳入",
    "排除",
)


def _plan_expects_analysis_cohort(plan: AnalysisPlan) -> bool:
    """True when the plan clearly intends to *define* an analysis population.

    A cohort-definition / eligibility / attrition step means the agent is
    applying inclusion/exclusion, so leaving ``plan.cohort`` empty is a contract
    violation — not a legitimate whole-universe analysis. Mere descriptive
    mentions of "the cohort" do not count.
    """
    for step in plan.steps or []:
        blob = " ".join([step.step_id or "", step.intent or ""]).lower()
        if any(marker in blob for marker in _COHORT_DEFINITION_MARKERS):
            return True
    return False


def _cohort_definition_prose(plan: AnalysisPlan) -> str:
    """Concatenated ``intent`` prose of the plan's cohort-defining step(s).

    This is the free-text 纳排 the agent wrote in lieu of a structured
    ``plan.cohort``; ``cohort_repair`` translates it into typed predicates.
    Uses the same markers as :func:`_plan_expects_analysis_cohort` so the
    definition of "a cohort step" stays in one place.
    """
    prose: List[str] = []
    for step in plan.steps or []:
        blob = " ".join([step.step_id or "", step.intent or ""]).lower()
        if any(marker in blob for marker in _COHORT_DEFINITION_MARKERS):
            if step.intent:
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
    "clustering",
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


def _enforce_advanced_plan_contract(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Constrain advanced plan shape while leaving analysis code to the agent."""

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
        plan_blob = " ".join(
            [
                context.research_question or "",
                " ".join(
                    " ".join(
                        [
                            step.step_id or "",
                            step.intent or "",
                            step.method or "",
                            " ".join(step.expected_outputs or []),
                        ]
                    )
                    for step in (plan.steps or [])
                ),
            ]
        ).lower()
        if any(
            marker in plan_blob
            for marker in (
                "complete-case",
                "complete case",
                "missing-indicator",
                "missing indicator",
                "reduced-variable",
                "reduced variable",
                "robustness",
            )
        ):
            family = "robustness"
        elif any(
            marker in plan_blob
            for marker in (
                "cluster",
                "clustering",
                "phenotype",
                "trajectory",
                "silhouette",
            )
        ):
            family = "clustering"
        elif (
            "sofa" not in plan_blob
            and _question_primary_predictor_is_vasopressor_or_unknown(context)
            and (
                any(
                    marker in plan_blob
                    for marker in (
                        "selection bias",
                        "confounding by indication",
                        "confounded by indication",
                    )
                )
                or (
                    ("vasopressor" in plan_blob or "vaso" in plan_blob)
                    and "association" in plan_blob
                    and "mortality" in plan_blob
                )
            )
        ):
            family = "bias_audit"
        elif any(
            marker in plan_blob
            for marker in (
                "prediction",
                "auroc",
                "auc",
                "brier",
                "calibration",
                "cross-validation",
                "cross validation",
                "held-out",
                "held out",
            )
        ):
            family = "prediction_model"
    if family not in _CONTRACT_FAMILIES:
        return plan, []

    if family == "prediction_model":
        markers = (
            "prediction",
            "model",
            "training",
            "performance",
            "auroc",
            "auc",
            "brier",
            "calibration",
            "discrimination",
        )
        canonical_step_id = "01_model_training"
        canonical_method = "prediction_model"
        canonical_intent = (
            "Train and validate the mortality prediction model in one "
            "self-contained executable step."
        )
        required_outputs = [
            "statistic:auroc",
            "statistic:brier_score",
            "statistic:baseline_prevalence",
            "statistic:split_strategy",
            "table:model_performance",
            "figure:discrimination_calibration",
        ]
    elif family == "clustering":
        markers = (
            "cluster",
            "clustering",
            "phenotype",
            "trajectory",
            "silhouette",
            "mortality_by_cluster",
        )
        canonical_step_id = "01_phenotype_trajectory_clustering"
        canonical_method = "clustering"
        canonical_intent = (
            "Generate trajectory clusters over the predictor variables named "
            "in the research context, with cluster summaries, post-hoc "
            "outcome rate by cluster, validation metrics and a figure in one "
            "self-contained executable step."
        )
        required_outputs = [
            "statistic:silhouette_score",
            "statistic:cluster_count",
            "table:cluster_characteristics",
            "table:cluster_mortality",
            "figure:clustering_visualization",
            "log:clustering_algorithm_details",
            "manifest:clustering_methodology",
        ]
    elif family == "survival":
        markers = (
            "survival",
            "time-to-event",
            "time to event",
            "kaplan",
            "kaplan-meier",
            "kaplan meier",
            "cox",
            "hazard",
            "proportional hazards",
            "log-rank",
            "log rank",
            "person-time",
        )
        canonical_step_id = "01_survival_analysis"
        canonical_method = "survival_analysis"
        canonical_intent = (
            "Run the time-to-event survival analysis in one self-contained "
            "executable step: define time-zero and the follow-up window, "
            "estimate Kaplan-Meier curves by the primary stratum, fit a Cox "
            "proportional-hazards model, and write the hazard-ratio table plus "
            "the survival-curve figure. Respect censoring and do not collapse "
            "time-to-event into a static binary outcome."
        )
        required_outputs = [
            "statistic:hazard_ratio",
            "statistic:n_events",
            "statistic:median_followup",
            "table:cox_summary",
            "figure:survival_curves",
            "log:survival_time_definition",
        ]
    elif family == "dynamic_prediction":
        markers = (
            "dynamic prediction",
            "time-updated",
            "time updated",
            "time-varying",
            "time varying",
            "time-dependent",
            "rolling",
            "landmark",
            "early warning",
            "deterioration",
            "prediction horizon",
            "update cadence",
        )
        canonical_step_id = "01_dynamic_prediction"
        canonical_method = "dynamic_prediction"
        canonical_intent = (
            "Build the time-updated prediction analysis in one self-contained "
            "executable step: keep prediction time, observation window, and "
            "target horizon distinct, evaluate discrimination at each horizon "
            "with a strict anti-leakage split, and write the time-varying "
            "discrimination figure. Do not collapse longitudinal forecasting "
            "into a single static prediction."
        )
        required_outputs = [
            "statistic:time_varying_auroc",
            "statistic:prediction_horizon",
            "table:horizon_performance",
            "figure:time_varying_discrimination",
            "log:anti_leakage_audit",
        ]
    elif family == "causal_inference":
        markers = (
            "causal",
            "propensity",
            "ipw",
            "iptw",
            "inverse probability",
            "g-formula",
            "g-computation",
            "doubly robust",
            "target trial",
            "instrumental variable",
            "marginal structural",
            "counterfactual",
            "standardized mean difference",
            "covariate balance",
        )
        canonical_step_id = "01_causal_effect_estimation"
        canonical_method = "causal_inference"
        canonical_intent = (
            "Estimate the adjusted treatment effect in one self-contained "
            "executable step: state the identification assumptions, check "
            "covariate balance / positivity before and after weighting, report "
            "the adjusted effect with its uncertainty, and write the "
            "covariate-balance (love) figure. Keep causal language conditional "
            "on the stated assumptions; do not over-claim causality."
        )
        required_outputs = [
            "statistic:adjusted_effect",
            "statistic:max_smd_after_weighting",
            "table:covariate_balance",
            "figure:covariate_balance",
            "log:identification_assumptions",
        ]
    elif family == "treatment_response":
        markers = (
            "treatment response",
            "responder",
            "nonresponder",
            "non-responder",
            "heterogeneous treatment effect",
            "effect modification",
            "drug response",
            "therapy response",
            "cate",
        )
        canonical_step_id = "01_treatment_response_heterogeneity"
        canonical_method = "treatment_response"
        canonical_intent = (
            "Characterize treatment-response heterogeneity in one self-contained "
            "executable step: estimate the overall effect, test for effect "
            "modification with an explicit interaction term, summarize "
            "pre-specified subgroup effects, and write the subgroup forest "
            "figure. Treat subgroup effects as exploratory and report the "
            "interaction test plus multiplicity, not subgroup p-values alone."
        )
        required_outputs = [
            "statistic:overall_effect",
            "statistic:interaction_pvalue",
            "table:subgroup_effects",
            "figure:subgroup_forest",
            "log:multiplicity_note",
        ]
    elif family == "validation":
        markers = (
            "external validation",
            "externally validate",
            "transportability",
            "reclassification",
            "net reclassification",
            "score comparison",
            "compare score",
            "validate score",
            "calibration-in-the-large",
        )
        canonical_step_id = "01_external_validation"
        canonical_method = "validation"
        canonical_intent = (
            "Validate the score/model in one self-contained executable step: "
            "evaluate discrimination and calibration on the validation cohort, "
            "report calibration-in-the-large and slope alongside AUROC, and "
            "write the external discrimination/calibration figure. Keep the "
            "development and validation cohorts and time windows distinct."
        )
        required_outputs = [
            "statistic:validation_auroc",
            "statistic:calibration_slope",
            "table:validation_performance",
            "figure:external_validation",
            "log:validation_cohort_definition",
        ]
    elif family == "bias_audit":
        markers = (
            "association",
            "vasopressor",
            "vaso",
            "mortality",
            "selection",
            "bias",
            "confounding",
            "missingness",
            "clinical-constraint",
            "clinical constraint",
            "logistic",
            "regression",
            "model",
            "strategy",
            "adjusted",
            "odds ratio",
            "or",
        )
        canonical_step_id = "02_treatment_exposure_bias_association"
        canonical_method = "bias_audit_association"
        canonical_intent = (
            "Fit an outcome association model for the treatment exposure named "
            "in the research context with severity / missingness covariates; "
            "report the primary odds ratio, selection-bias or "
            "confounding-by-indication warning, missingness profile, and avoid "
            "causal treatment-effect language."
        )
        required_outputs = [
            "statistic:primary_or",
            "statistic:selection_bias_warning",
            "statistic:mortality_rate",
            "table:association_summary",
            "table:missingness_profile",
            "log:clinical_constraint_warning",
        ]
    else:
        markers = (
            "complete-case",
            "complete case",
            "missing-indicator",
            "missing indicator",
            "reduced-variable",
            "reduced variable",
            "robustness",
            "odds ratio",
            "logistic",
            "model",
            "figure",
            "performance",
        )
        canonical_step_id = "03_complete_case_robustness"
        canonical_method = "association_robustness"
        canonical_intent = (
            "Fit complete-case, missing-indicator, and reduced-variable association "
            "models; extract the primary effect estimate and complete-case sample "
            "size; write the summary table and robustness figure in one "
            "self-contained executable step."
        )
        required_outputs = [
            "statistic:primary_or",
            "statistic:complete_case_n",
            "table:robustness_summary",
            "figure:robustness_plot",
            "log:missingness_strategy_notes",
        ]

    def _is_relevant(step: AnalysisStep) -> bool:
        text = " ".join(
            [
                step.step_id or "",
                step.intent or "",
                step.method or "",
                " ".join(step.expected_outputs or []),
            ]
        ).lower()
        return any(marker in text for marker in markers)

    relevant_indexes = [
        idx for idx, step in enumerate(plan.steps) if _is_relevant(step)
    ]
    if not relevant_indexes:
        return plan, []

    first_index = relevant_indexes[0]
    relevant_steps = [plan.steps[idx] for idx in relevant_indexes]
    combined_inputs: List[str] = []
    for step in relevant_steps:
        for item in step.inputs or []:
            if item not in combined_inputs:
                combined_inputs.append(item)
    combined_outputs = list(required_outputs)
    for step in relevant_steps:
        for item in step.expected_outputs or []:
            if item not in combined_outputs:
                combined_outputs.append(item)

    current = relevant_steps[0]
    missing_outputs = [
        item for item in required_outputs if item not in current.expected_outputs
    ]
    needs_normalisation = (
        len(relevant_indexes) != 1
        or bool(missing_outputs)
        or current.step_id != canonical_step_id
    )
    if not needs_normalisation:
        return plan, []

    canonical_step = current.model_copy(
        update={
            "step_id": canonical_step_id,
            "intent": canonical_intent,
            "inputs": combined_inputs or current.inputs,
            "expected_outputs": combined_outputs,
            "method": canonical_method,
        }
    )
    new_steps: List[AnalysisStep] = []
    inserted = False
    relevant_set = set(relevant_indexes)
    for idx, step in enumerate(plan.steps):
        if idx in relevant_set:
            if not inserted:
                new_steps.append(canonical_step)
                inserted = True
            continue
        new_steps.append(step)

    revised = plan.model_copy(
        update={"steps": new_steps, "revision": max(1, plan.revision) + 1}
    )
    finding = ValidationFinding(
        validator="plan_contract",
        severity="warning",
        message=(
            f"Planner output for {family} was normalized to a single "
            "self-contained advanced-analysis step with explicit v14 metric "
            "and artefact contracts."
        ),
        detail={
            "family": family,
            "original_step_ids": [step.step_id for step in relevant_steps],
            "canonical_step_id": canonical_step_id,
            "canonical_insert_index": first_index,
            "required_outputs": required_outputs,
        },
    )
    return revised, [finding]


def _question_primary_predictor_is_vasopressor_or_unknown(
    context: ResearchContext,
) -> bool:
    predictor = _infer_primary_predictor_from_context(context)
    if not predictor:
        return True
    tokens = _predictor_tokens(predictor)
    return bool(tokens & {"vaso", "vasopressor", "vasopressors", "norepinephrine"})


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


_FIGURE_STEP_TOKENS = ("figure", "plot", "chart", "fig:", "figure:", "plot:")


_PUBLICATION_FIGURE_TRIGGER_TOKENS = (
    "publication-ready figure",
    "publication ready figure",
    "publication figure",
    "produce a heatmap",
    "produce a figure",
    "publication-ready",
    "figure or",
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

    for step in plan.steps:
        outputs = list(step.expected_outputs or [])
        method = (step.method or "").lower()
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
            new_steps.append(step)
            continue
        figure_outputs = [
            out
            for out in outputs
            if any(needle in (out or "").lower() for needle in _FIGURE_STEP_TOKENS)
        ]
        non_figure_outputs = [out for out in outputs if out not in figure_outputs]
        # Only split when the step is genuinely mixed — at least one
        # non-figure analytic payload (table, statistic, log, model, ...)
        # alongside a figure. Splitting a figure-only step would create
        # an empty stub that the coder cannot anchor to a parent
        # artefact, so we require the non-figure outputs to look like
        # real deliverables. ``model:*`` is included because regression
        # / prediction / association steps frequently bundle a model
        # object with a companion figure, and the agent forgets to draw
        # the figure when both are demanded in the same script.
        has_non_figure_payload = any(
            (out or "").lower().startswith(("table:", "statistic:", "log:", "model:"))
            or (out or "").lower() in {"table", "statistic", "log", "model"}
            for out in non_figure_outputs
        )
        if not figure_outputs or not has_non_figure_payload:
            new_steps.append(step)
            continue
        # Keep the original step with the non-figure outputs.
        non_figure_step = step.model_copy(
            update={"expected_outputs": non_figure_outputs}
        )
        new_steps.append(non_figure_step)
        # Synthesise a follow-up figure-only step.
        figure_step_id = f"{step.step_id}_figure"
        figure_intent = (
            f"Render the publication figure(s) declared by step "
            f"'{step.step_id}' ({', '.join(figure_outputs)}). Load the "
            "cohort from ``os.environ['COHORT_PARQUET']`` (full path is "
            "provided by the runner) and, if needed, read tables produced "
            f"by '{step.step_id}' from any of the registered evidence "
            "files. Save PNG and SVG copies of every figure with matching "
            "stems into ``os.environ['STEP_OUT_DIR']``. Always write a "
            "valid step_summary.json into ``STEP_OUT_DIR`` listing each "
            "produced file under ``figure_files`` even if rendering fails — "
            "use a try/except so the step never aborts before writing the "
            "summary."
        )
        figure_step = AnalysisStep(
            step_id=figure_step_id,
            intent=figure_intent,
            inputs=list(step.inputs or []),
            expected_outputs=figure_outputs,
            method=(step.method or "visualization"),
            icu_rule_refs=list(step.icu_rule_refs or []) + ["visualization_rule"],
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
    return any(token in text for token in _PUBLICATION_FIGURE_TRIGGER_TOKENS)


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

    next_index = len(plan.steps or []) + 1
    fallback_step = AnalysisStep(
        step_id=f"{next_index:02d}_publication_figure_fallback",
        intent=(
            "Render a publication-ready figure that summarises the "
            "analytics produced by the previous steps. Read the latest "
            "step_summary.json files under the run directory, pick the "
            "most informative numeric structure (e.g. mortality by "
            "stratum, correlation values, model performance), and save "
            "the figure as both PNG and SVG with the same stem into "
            "``os.environ['STEP_OUT_DIR']`` (set by the runner). Record "
            "every produced path in step_summary.json under "
            "``figure_files``."
        ),
        method="visualization",
        inputs=[],
        expected_outputs=["figure:overview"],
        icu_rule_refs=["visualization_rule"],
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
# audit panel" when a step's intent or expected_outputs mention any of these.
_AUDIT_PANEL_TOKENS = ("audit", "completeness", "sensitivity", "leakage", "calibration")


def _step_declares_audit_panel(step: AnalysisStep) -> bool:
    """True if the step declares an audit/sensitivity/robustness display item."""
    for text in [step.intent or "", *(step.expected_outputs or [])]:
        lowered = (text or "").lower()
        if any(token in lowered for token in _AUDIT_PANEL_TOKENS):
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
    for output in step.expected_outputs or []:
        token = (output or "").lower()
        for needle in _FIGURE_STEP_TOKENS:
            if needle in token:
                return True
    return False


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
    if not dropped_figure_steps:
        return revised, []
    new_steps = list(revised.steps) + list(dropped_figure_steps)
    preserved = revised.model_copy(update={"steps": new_steps})
    findings = [
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
    ]
    return preserved, findings


def _cap_plan_preserving_figure_steps(
    *,
    plan: AnalysisPlan,
    cap: int,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Truncate an initial plan without dropping required figure steps."""

    steps = list(plan.steps or [])
    if cap <= 0 or len(steps) <= cap:
        return plan, []

    kept = list(steps[:cap])
    dropped = list(steps[cap:])
    preserved_step_ids: List[str] = []
    displaced_step_ids: List[str] = []

    for step in dropped:
        if not _step_produces_figure(step):
            continue
        replace_idx: Optional[int] = None
        for idx in range(len(kept) - 1, -1, -1):
            if not _step_produces_figure(kept[idx]):
                replace_idx = idx
                break
        if replace_idx is None:
            continue
        displaced = kept[replace_idx]
        kept[replace_idx] = step
        preserved_step_ids.append(step.step_id)
        displaced_step_ids.append(displaced.step_id)

    kept_ids = {step.step_id for step in kept}
    dropped_ids = [step.step_id for step in steps if step.step_id not in kept_ids]
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
                "preserved_figure_step_ids": preserved_step_ids,
                "displaced_step_ids": displaced_step_ids,
            },
        )
    ]
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
        or re.search(r"(?:^|[._:\-])or(?:$|[._:\-])", lowered)
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
        or re.search(r"(?:^|[._:\-])or(?:$|[._:\-])", lowered)
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
    effect = _first_numeric_effect_from_text(step_summary)
    return _finite_float(effect)


def _primary_effect_from_completed_records(
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    *,
    current_step_id: str,
) -> Optional[Tuple[str, float]]:
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
        effect = _primary_effect_from_summary(step_summary)
        if effect is not None:
            return source_step_id, effect
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

    Mirrors :func:`_primary_effect_from_completed_records` for the prediction
    requirement: a figure/rendering step (e.g. ``*_model_training_figure``)
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
                f"with `{required}` as the primary exposure — derive its binary "
                f"indicator from the source columns if needed (an absent event "
                f"is 0/False, not missing data). Keep illness-severity scores "
                f"as adjustment covariates only, never as the exposure."
            ),
            detail={
                "kind": "exposure_contract",
                "step_id": step.step_id,
                "required_exposure": required,
                "actual_predictor": actual,
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


def read_model_covariate_names(directory: Path) -> List[str]:
    """Variable names from every model coefficient table under ``directory``.

    De-duplicated, intercept rows dropped, first-seen order preserved. Returns
    ``[]`` when the directory is absent or holds no coefficient table — the
    overadjustment check then stays silent rather than guessing. Filename-agnostic:
    a CSV counts as a coefficient table only when its header has a ``variable``
    column and a coefficient-like column, so non-model tables are ignored.
    """
    names: List[str] = []
    base = Path(directory)
    if not base.exists():
        return names
    for path in sorted(base.rglob("*.csv")):
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


def _name_intends_covariates(name: str) -> bool:
    low = name.lower()
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


def _covariate_names_from_code(directory: Path) -> List[str]:
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
    candidates: List[Path] = []
    for src in (base, base.parent):
        if src.exists():
            candidates.extend(sorted(src.glob("*.py")))
    if base.exists():
        candidates.extend(sorted(base.rglob("analysis.py")))

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


def read_adjustment_covariates(directory: Path) -> List[str]:
    """The model's adjustment set, preferring the coefficient table.

    A per-covariate coefficient table is the ground truth of what entered the
    model, so it wins when present. When a run reports only a model-level OR
    summary (no coefficient table), the adjustment set is recovered from the
    analysis code instead, so the overadjustment / leakage auditors are not blind
    to summary-only outputs. Returns ``[]`` when neither source yields anything.
    """
    coef_names = read_model_covariate_names(directory)
    if coef_names:
        return coef_names
    return _covariate_names_from_code(directory)


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
    if not covariates:
        return []
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
    completed_step_records: Optional[Sequence[Dict[str, Any]]] = None,
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
    expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
    step_id = (step.step_id or "").lower()
    intent = (step.intent or "").lower()

    # Figure-only follow-up steps (created by ``_split_table_and_figure_outputs_in_plan``)
    # inherit the parent's step_id with a ``_figure`` suffix, e.g.
    # ``04_primary_association_figure`` / ``01_model_training_figure``. Their
    # expected_outputs contain *only* figure items — the analytic payload
    # (table/statistic/etc.) lives in the sibling parent step. Without this guard
    # the substring matches ``primary_association``/``model_training``/``cluster``
    # below would falsely demand effect/prediction/clustering metrics from a
    # render-only step that legitimately has no such fields in its summary.
    figure_only_step = bool(step.expected_outputs) and all(
        any(needle in (out or "").lower() for needle in _FIGURE_STEP_TOKENS)
        for out in step.expected_outputs
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

    effect_required = not figure_only_step and (
        any(
            token in expected
            for token in (
                "adjusted_or_ci",
                "primary_association",
                "odds_ratio",
                "primary_or",
                "adjusted_or",
            )
        )
        or "primary_association" in step_id
    )
    if not effect_required and not figure_only_step and "association" in expected:
        effect_required = (
            "model" in step_id
            or "regression" in intent
            or "estimate" in intent
            or "odds" in expected
        )
    if (
        not effect_required
        and not figure_only_step
        and (
            ("logistic" in expected or "logistic" in intent or "odds" in intent)
            and ("model" in step_id or "model" in expected or "regression" in intent)
        )
    ):
        effect_required = True
    if effect_required:
        effect_value = _primary_effect_from_summary(step_summary)
        fallback_effect = None
        if effect_value is None:
            fallback_effect = _primary_effect_from_completed_records(
                completed_step_records,
                current_step_id=str(step.step_id or ""),
            )
        if effect_value is None and fallback_effect is not None:
            source_step_id, _source_effect = fallback_effect
            findings.append(
                ValidationFinding(
                    validator="step_contract",
                    severity="warning",
                    message=(
                        f"Step {step.step_id} did not record its own primary association "
                        f"estimate, but the requirement was satisfied by successful step "
                        f"{source_step_id}."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "fallback_step_id": source_step_id,
                        "expected_outputs": list(step.expected_outputs or []),
                        "summary_keys": sorted(step_summary.keys()),
                    },
                )
            )
        elif effect_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a primary association "
                    "estimate, but no numeric effect size was recorded."
                ),
                ("estimate", "primary_or", "odds_ratio", "adjusted_or"),
            )

    prediction_step = (
        "training_and_evaluation" in step_id
        or "model_training" in step_id
        or "prediction" in step_id
        or "prediction" in intent
    )
    prediction_required = (not figure_only_step) and (
        any(token in expected for token in ("auroc", "auc", "brier", "discrimination"))
        or ("calibration" in expected and prediction_step)
        or prediction_step
    )
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

    clustering_required = (not figure_only_step) and (
        any(token in expected for token in ("cluster", "silhouette"))
        or "cluster" in step_id
        or "clustering" in intent
    )
    if clustering_required:
        cluster_value = _first_present_scalar(
            step_summary,
            (
                "silhouette_score",
                "statistic:silhouette_score",
                "silhouette",
                "statistic:silhouette",
                "n_clusters",
                "statistic:n_clusters",
                "cluster_count",
                "statistic:cluster_count",
            ),
        )
        if cluster_value is None:
            cluster_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("silhouette", "cluster_count", "n_clusters"),
            )
        if cluster_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a clustering summary, "
                    "but no cluster metric or cluster count was recorded."
                ),
                ("silhouette_score", "silhouette", "n_clusters", "cluster_count"),
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
        # 🔧 2026-05-16: when the step itself declares it skipped (typically
        # because the underlying data wasn't present — e.g.
        # `"skipped": ["No SOFA-2 components available in the dataset"]` from
        # 11_sofa2_component_figure), don't fail the figure contract. The
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


def _step_contract_repair_guidance(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    code: str,
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
    step_id = (step.step_id or "").lower()
    intent = (step.intent or "").lower()
    effect_required = (
        any(
            token in expected
            for token in (
                "adjusted_or_ci",
                "primary_association",
                "odds_ratio",
                "primary_or",
                "adjusted_or",
            )
        )
        or "primary_association" in step_id
    )
    if not effect_required and "association" in expected:
        effect_required = (
            "model" in step_id
            or "regression" in intent
            or "estimate" in intent
            or "odds" in expected
        )
    if not effect_required and (
        ("logistic" in expected or "logistic" in intent or "odds" in intent)
        and ("model" in step_id or "model" in expected or "regression" in intent)
    ):
        effect_required = True
    if effect_required:
        guidance.append(
            "This association step must write a non-null numeric primary effect "
            "estimate in step_summary.json, such as `adjusted_or`, `primary_or`, "
            "`odds_ratio`, or `primary_association_estimate`. Do not satisfy the "
            "contract by leaving association fields null."
        )
    prediction_step = (
        any(
            token in step_id
            for token in (
                "prediction",
                "model_training",
                "training_and_evaluation",
                "performance",
            )
        )
        or "prediction" in intent
    )
    prediction_required = (
        any(token in expected for token in ("auroc", "auc", "brier", "discrimination"))
        or ("calibration" in expected and prediction_step)
        or prediction_step
    )
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
    clustering_required = (
        any(token in expected for token in ("cluster", "silhouette"))
        or any(token in step_id for token in ("cluster", "trajectory"))
        or "clustering" in intent
    )
    if clustering_required:
        guidance.append(
            "This clustering step must write machine-readable clustering metrics "
            "in step_summary.json. Use keys such as `silhouette_score` or "
            "`statistic:silhouette_score`, plus `cluster_count` or "
            "`statistic:cluster_count`."
        )
        guidance.append(
            "Keep clustering self-contained: create labels, cluster characteristics, "
            "post-hoc mortality by cluster, method metadata, and the clustering "
            "figure inside this script. Do not rely on labels saved by another step."
        )
        guidance.append(
            "Also save table artefacts named `cluster_characteristics.csv` and "
            "`cluster_mortality.csv` when feasible so manuscript evidence aliases bind."
        )
    if "figure:" in expected:
        guidance.append(
            "This step declares a figure output. Save a real figure file such as PNG/SVG/"
            "PDF/TIFF and record its path in step_summary.json using a key such as "
            "`figure_path`, `figure_file`, or `figure_files`."
        )
    if not guidance:
        guidance.append(
            "Repair the script so each expected output is written as machine-readable "
            "numbers in step_summary.json, or write a precise skipped/error reason."
        )
    return "\n".join(f"- {item}" for item in guidance)
