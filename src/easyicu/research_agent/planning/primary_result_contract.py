"""Planner contract for headline results and missing-data boundaries."""

from __future__ import annotations

import re

from ..schema import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    AnalysisPlan,
    AnalysisStep,
    ResearchContext,
)
from .analysis_types import infer_analysis_type


def primary_result_contract_guide() -> str:
    """Return case-neutral Planner guidance for primary-result ownership."""

    return (
        "Every step MUST explicitly declare `planned_analysis_role`; at most one "
        "step may be primary. "
        "An exposure-outcome association needs one primary adjusted model, not "
        "secondary feasibility work; a protocol/data audit may have none. "
        "For causal or survival work, the primary step MUST also declare one "
        "`family_primary_result_requirement`: its exact result table, exposure, "
        "outcome, estimator, effect scale, population and uncertainty method. "
        "Causal contracts additionally name estimand, treatment/comparator, "
        "adjustment and overlap diagnostic; survival contracts name time origin, "
        "event, censoring, competing-risk strategy, horizon, effect measure, "
        "and a PH diagnostic when using Cox.\n\n"
        "Missingness is scientific design: Do not impute the primary exposure or "
        "outcome. For prediction, split first and fit every imputer/scaler only on "
        "the training split/fold. For longitudinal data, never use future "
        "observations to fill an earlier window. Do not impute away missingness "
        "being studied; report complete-case attrition."
    )


def validate_required_primary_result(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> None:
    """Reject a result-bearing plan that omits its family-owned headline."""

    inferred_family = infer_analysis_type(context).key
    declared_family = str(plan.analysis_type or inferred_family).strip()
    question = str(context.research_question or "").casefold()
    question_requires_association = bool(
        re.search(
            r"\b(?:association|associated|relationship|risk factor|prognostic|"
            r"odds ratio|odds ratios)\b|关联|相关性",
            question,
        )
    )
    association_required = declared_family == "association_study" or (
        inferred_family == "association_study" and question_requires_association
    )
    family_result_required = declared_family in {"causal_inference", "survival"}
    if not association_required and not family_result_required:
        return
    exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    if not exposure or not outcome:
        return

    primary_steps = [
        step for step in plan.steps if step.planned_analysis_role == "primary"
    ]
    if len(primary_steps) != 1:
        raise ValueError(
            "Result-bearing analysis family "
            f"{inferred_family!r} requires exactly one Planner-owned primary "
            "result step; secondary, sensitivity, feasibility, protocol, and "
            "audit steps cannot replace the headline result"
        )

    primary = primary_steps[0]
    if family_result_required:
        requirement = primary.family_primary_result_requirement
        if requirement is None:
            raise ValueError(
                f"A {declared_family} context with a declared exposure and outcome "
                "requires one primary family_primary_result_requirement"
            )
        if requirement.analysis_family != declared_family:
            raise ValueError(
                "family_primary_result_requirement analysis_family must match the "
                f"plan family {declared_family!r}"
            )
        if requirement.exposure_source != exposure or requirement.outcome != outcome:
            raise ValueError(
                "The primary family-result contract must use the exact "
                "ResearchContext operational exposure and outcome columns; expected "
                f"exposure_source={exposure!r}, outcome={outcome!r}"
            )
        if requirement.expected_result_product not in primary.expected_outputs:
            raise ValueError(
                "The primary family-result contract must name a result product "
                "declared by its primary step"
            )
        if declared_family == "survival":
            endpoint = context.endpoint
            if endpoint is None or endpoint.kind != "time_to_event":
                raise ValueError(
                    "A survival primary-result contract requires a declared "
                    "time_to_event EndpointSpec; time/event columns are never "
                    "inferred from names or dtypes"
                )
            if (
                requirement.time_column != endpoint.time_column
                or requirement.event_column != endpoint.event_column
                or requirement.time_origin != endpoint.time_origin
            ):
                raise ValueError(
                    "The survival primary-result contract must use the exact "
                    "EndpointSpec time_origin, time_column, and event_column"
                )
            if SURVIVAL_ANALYSIS_RECEIPT_PRODUCT not in primary.expected_outputs:
                raise ValueError(
                    "A survival primary step must declare "
                    f"{SURVIVAL_ANALYSIS_RECEIPT_PRODUCT!r} so execution can "
                    "record the applied time/event/censoring design"
                )
        return

    method = re.sub(r"[^a-z0-9]+", "_", str(primary.method or "").lower()).strip("_")
    products: set[tuple[str, str]] = set()
    for raw in primary.expected_outputs:
        kind, separator, name = str(raw).partition(":")
        if separator:
            products.add(
                (
                    re.sub(r"[^a-z0-9]+", "_", kind.lower()).strip("_"),
                    re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"),
                )
            )
    primary_requirements = [
        item
        for item in primary.model_requirements
        if item.analysis_role == "primary" and item.required_for_step_success
    ]
    if (
        method != "adjusted_association_models"
        or ("table", "adjusted_association_estimates") not in products
        or len(primary_requirements) != 1
    ):
        raise ValueError(
            "An association context with a declared exposure and outcome requires "
            "one primary step using method='adjusted_association_models', output "
            "'table:adjusted_association_estimates', and exactly one required "
            "primary model_requirements entry"
        )
    requirement = primary_requirements[0]
    if requirement.exposure_source != exposure or requirement.outcome != outcome:
        raise ValueError(
            "The required primary association model must use the exact "
            "ResearchContext operational exposure and outcome columns; expected "
            f"exposure_source={exposure!r}, outcome={outcome!r}"
        )


def family_primary_result_execution_guide(step: AnalysisStep) -> str:
    """Return the Coder's source-data obligation for one family headline."""

    requirement = step.family_primary_result_requirement
    if requirement is None:
        return ""
    return (
        "\nFAMILY PRIMARY-RESULT EXECUTION CONTRACT:\n"
        f"- Materialise `{requirement.expected_result_product}` as a CSV inside "
        "OUTPUT_DIR and register that exact product/path in "
        "`summary['output_files']`.\n"
        "- The result CSV must include `exposure_source`, `outcome`, "
        "`effect_scale`, one finite effect column (`effect_estimate`, "
        "`estimate`, `point_estimate`, `adjusted_effect`, `risk_difference`, "
        "`risk_ratio`, `odds_ratio`, or `hazard_ratio`), and either `ci_low` + "
        "`ci_high` or `standard_error`.\n"
        f"- Its primary row must state exposure_source={requirement.exposure_source!r}, "
        f"outcome={requirement.outcome!r}, and effect_scale={requirement.effect_scale!r}. "
        "Do not substitute chart geometry, prose, or an unregistered side file "
        "for this evidence table.\n"
        + (
            "- Also materialise `log:survival_analysis_receipt` as JSON inside "
            "OUTPUT_DIR and register it in `summary['output_files']`. It must "
            "validate as `SurvivalAnalysisReceipt`, repeat the exact declared "
            "time origin, time/event columns, event definition, censoring, "
            "competing-risk strategy, horizon, estimator, effect measure, "
            "population and result product, and record n_analysis_rows/n_events. "
            "For Cox, it MUST record the executed PH diagnostic and its p value.\n"
            if requirement.analysis_family == "survival"
            else ""
        )
    )


__all__ = [
    "family_primary_result_execution_guide",
    "primary_result_contract_guide",
    "validate_required_primary_result",
]
