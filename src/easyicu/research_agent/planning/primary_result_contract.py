"""Planner contract for headline results and missing-data boundaries."""

from __future__ import annotations

import re

from ..schema import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    AnalysisPlan,
    AnalysisStep,
    ResearchContext,
)
from ..contracts.survival import SURVIVAL_PH_DIAGNOSTIC_PRODUCT
from ..contracts.survival_execution import survival_execution_verdict
from ..contracts.time_units import canonical_time_unit
from .analysis_types import infer_analysis_type
from .capability_registry import resolve_primary_capability


#: Capability verdicts a plan cannot repair by declaring a missing field: the
#: plan is asking for one capability and structurally describing another.
#: ``primary_owner_declaration_incomplete`` is deliberately absent -- the
#: plan-time owner-declaration gate already turns that into one focused
#: replan directive, and raising here would convert a repairable gap into a
#: parse failure.
_STRUCTURAL_CAPABILITY_REFUSALS = frozenset(
    {
        "primary_capability_owner_mismatch",
        "freeform_step_claims_host_product",
        "scientific_capability_declaration_required",
        "scientific_capability_unknown",
        "scientific_capability_family_mismatch",
        "scientific_capability_step_incompatible",
    }
)


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
        "a numeric horizon/unit, event value, exact adjustment set, input product, "
        "typed coding for every model term, and a PH diagnostic plus declared "
        "handling policy/alpha when using Cox. Primary Cox execution and its "
        "receipt are host-owned, not Coder-authored.\n\n"
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
            time_descriptor = context.variable(str(requirement.time_column))
            if time_descriptor is None or not str(time_descriptor.unit or "").strip():
                raise ValueError(
                    "The survival time column requires an authoritative unit in "
                    "ResearchContext.variables; units are never inferred from a "
                    "column name or value magnitude"
                )
            authoritative_time_unit = canonical_time_unit(time_descriptor.unit)
            if authoritative_time_unit is None:
                raise ValueError(
                    "The survival time column unit in ResearchContext is not in "
                    "the supported closed vocabulary (minutes, hours, days)"
                )
            if authoritative_time_unit != requirement.time_unit:
                raise ValueError(
                    "family_primary_result_requirement.time_unit must match the "
                    "host-owned ConceptDescriptor.unit for the exact time column; "
                    f"expected {authoritative_time_unit!r}, declared "
                    f"{requirement.time_unit!r}"
                )
            if SURVIVAL_ANALYSIS_RECEIPT_PRODUCT not in primary.expected_outputs:
                raise ValueError(
                    "A survival primary step must declare "
                    f"{SURVIVAL_ANALYSIS_RECEIPT_PRODUCT!r} so execution can "
                    "record the applied time/event/censoring design"
                )
            if SURVIVAL_PH_DIAGNOSTIC_PRODUCT not in primary.expected_outputs:
                raise ValueError(
                    "A survival primary step must declare "
                    f"{SURVIVAL_PH_DIAGNOSTIC_PRODUCT!r} so the executed PH "
                    "diagnostic is materialised as evidence"
                )
            levels = list(endpoint.levels or [])
            try:
                numeric_levels = [float(value) for value in levels]
                integral_levels = {
                    int(value) for value in numeric_levels if value.is_integer()
                }
            except (TypeError, ValueError):
                integral_levels = set()
            if (
                len(levels) != 2
                or len(integral_levels) != 2
                or integral_levels != {0, int(requirement.event_value)}
            ):
                raise ValueError(
                    "The host Cox executor requires a binary EndpointSpec whose "
                    "closed levels are exactly censor code 0 and "
                    "family_primary_result_requirement.event_value"
                )
            verdict = survival_execution_verdict(
                requirement=requirement,
                planned_analysis_role=primary.planned_analysis_role,
                expected_outputs=primary.expected_outputs,
                inputs=primary.inputs,
            )
            if not verdict.claimed:
                raise ValueError(
                    "The primary survival contract is not executable by the "
                    f"host-owned survival runner: {verdict.reason}"
                )
        return

    # Which association contract this plan declared is the capability
    # resolver's answer, not a second copy of the routing rule here. The
    # registry advertises two association capabilities; validating every one of
    # them against the exact single-model contract made the registered
    # free-form capability unreachable through Planner parse, so a plan could
    # only ever be labelled with a capability it was forbidden to declare.
    verdict = resolve_primary_capability(analysis_type=declared_family, plan=plan)
    if verdict.failure_reason in _STRUCTURAL_CAPABILITY_REFUSALS:
        raise ValueError(verdict.detail)
    if verdict.capability_id == "association_freeform_v1":
        # The agent-coded kernel carries no typed model contract to validate:
        # ``AnalysisStep`` already refuses ``model_requirements`` on any step
        # that is not the exact host method/product pair, and ``AnalysisPlan``
        # already requires every primary step to declare a typed non-rendering
        # result product. The one obligation that is specific to this
        # capability -- not borrowing the sealed executor's product key -- is
        # the resolver's ``freeform_step_claims_host_product`` above. A
        # separate free-form validator here would only restate checks that
        # already ran, which reads as coverage it does not provide.
        return
    _validate_exact_adjusted_association(
        primary=primary, exposure=exposure, outcome=outcome
    )


def _validate_exact_adjusted_association(
    *,
    primary: AnalysisStep,
    exposure: str,
    outcome: str,
) -> None:
    """The host-owned single-model contract: one model, declared coding."""

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
    if requirement.model_terms is None:
        raise ValueError(
            "The required primary association model must explicitly declare "
            "model_terms; the host cannot infer continuous, binary, categorical "
            "or ordinal coding from names or dtypes"
        )


def family_primary_result_execution_guide(step: AnalysisStep) -> str:
    """Return the Coder's source-data obligation for one family headline."""

    requirement = step.family_primary_result_requirement
    if requirement is None:
        return ""
    if requirement.analysis_family == "survival":
        return (
            "\nHOST-OWNED SURVIVAL PRIMARY CONTRACT:\n"
            "- Do not implement or rewrite this step and do not create a survival "
            "receipt. The sealed EasyICU survival executor fits the declared Cox "
            "model, runs the PH diagnostic, and issues the digest-bound receipt.\n"
        )
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
    )


__all__ = [
    "family_primary_result_execution_guide",
    "primary_result_contract_guide",
    "validate_required_primary_result",
]
