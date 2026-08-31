"""Compile planner output into the advanced analysis-family contract."""

from __future__ import annotations

from typing import List

from ..contracts.step_families import (
    _CONTRACT_FAMILIES,
    _HEURISTIC_REACHABLE_FAMILIES,
    _article_display_roles,
    _best_contract_step_for_outputs,
    _normalise_contract_family,
    _plan_step_owns_contract_family,
)
from .figure_plan_shaping import dedicated_renderer_consumes_typed_source as _dedicated_renderer_consumes_typed_source
from .robustness_plan_mutation import ROBUSTNESS_ARTICLE_OUTPUT_BINDINGS, with_family_contract_outputs
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext, ValidationFinding
from ..trajectory.contract import TRAJECTORY_PHENOTYPING_REQUIRED_OUTPUTS, trajectory_phenotyping_contract_applies
from ..trajectory.plan_contract import trajectory_plan_contract_applies

def _enforce_advanced_plan_contract(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    long_trajectory_bound: bool = False,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Constrain advanced plan shape while leaving analysis code to the agent."""

    # Fixed-window trajectory plans have a role/DAG contract that supports
    # legitimate agent decomposition. The generic clustering normalizer assumes
    # one method owner and would push all products into that step, recreating a
    # mega-pipeline. Leave this family to the dedicated role normalizer.
    if trajectory_plan_contract_applies(
        plan=plan,
        context=context,
        long_trajectory_bound=long_trajectory_bound,
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
            *(
                product
                for product, _output in ROBUSTNESS_ARTICLE_OUTPUT_BINDINGS[:3]
            ),
            "figure:robustness_plot",
            ROBUSTNESS_ARTICLE_OUTPUT_BINDINGS[3][0],
        ]

    if family == "robustness" and _dedicated_renderer_consumes_typed_source(
        plan.steps,
        source="table:robustness_matrix",
    ):
        # A Planner-owned renderer already presents the deterministic replay
        # result.  Do not add a differently named conventional figure to the
        # scientific producer: the subsequent mixed-output split would create
        # two visual owners for the same verified matrix.
        required_outputs = [
            output for output in required_outputs if output != "figure:robustness_plot"
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
                expected_outputs = [
                    *(step.expected_outputs or []),
                    *missing_outputs,
                ]
                new_steps.append(
                    with_family_contract_outputs(
                        step,
                        family=family,
                        expected_outputs=expected_outputs,
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
        new_steps[owner_index] = with_family_contract_outputs(
            owner,
            family=family,
            expected_outputs=[
                *(owner.expected_outputs or []),
                *missing_across_family,
            ],
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
    contract_step = with_family_contract_outputs(
        current,
        family=family,
        expected_outputs=combined_outputs,
    )
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

