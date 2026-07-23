"""Planner contract for headline results and missing-data boundaries."""

from __future__ import annotations

import re

from ..schema import AnalysisPlan, ResearchContext
from .analysis_types import infer_analysis_type


def primary_result_contract_guide() -> str:
    """Return case-neutral Planner guidance for primary-result ownership."""

    return (
        "Every step MUST explicitly declare `planned_analysis_role`; at most one "
        "step may be primary. "
        "An exposure-outcome association needs one primary adjusted model, not "
        "secondary feasibility work; a protocol/data audit may have none.\n\n"
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
    """Reject an association plan that omits its typed headline model."""

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
    if declared_family != "association_study" and not (
        inferred_family == "association_study" and question_requires_association
    ):
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


__all__ = [
    "primary_result_contract_guide",
    "validate_required_primary_result",
]
