"""Canonical fingerprints for Planner-owned scientific scope.

The functions here serialize existing Planner decisions for resume and typed
authority checks.  They do not choose or revise a cohort, exposure, outcome,
method, covariate set, or estimand.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from ..schema import AnalysisPlan, AnalysisStep

__all__ = [
    "_normalise_scientific_text",
    "_plan_signature",
    "_plan_scientific_scope_signature",
    "_serializable_plan_scientific_scope_signature",
    "_step_scientific_signature",
]


def _step_scientific_signature(step: AnalysisStep) -> Tuple[Any, ...]:
    """Fingerprint every Planner-owned field that can change execution.

    The current schema still carries exposure/outcome definitions, time windows,
    covariates, and missingness policy in ``intent``.  Until those coordinates
    are fully structured, ordinary semantic paraphrases cannot safely be
    distinguished from a changed estimand.  Only case/whitespace normalization
    is ignored.
    """

    return (
        step.step_id,
        step.method,
        tuple(step.inputs),
        tuple(step.expected_outputs),
        " ".join(str(step.intent or "").split()).casefold(),
        tuple(step.icu_rule_refs),
        step.planned_analysis_role,
        tuple(
            (
                requirement.requirement_id,
                requirement.outcome,
                requirement.outcome_type,
                requirement.method_family,
                requirement.exposure_source,
                requirement.analysis_role,
                requirement.analysis_set,
                requirement.required_for_step_success,
            )
            for requirement in step.model_requirements
        ),
        tuple(
            json.dumps(
                contract.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for contract in step.input_consumption_contracts
        ),
        (
            json.dumps(
                step.trajectory_stability_spec.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            if step.trajectory_stability_spec is not None
            else None
        ),
    )


def _normalise_scientific_text(value: Any) -> Optional[str]:
    """Normalize cosmetic prose differences without erasing scientific edits."""

    if value is None:
        return None
    return " ".join(str(value).split()).casefold()


def _plan_scientific_scope_signature(plan: AnalysisPlan) -> Tuple[Optional[str], ...]:
    """Fingerprint Planner-owned science that applies to every plan step.

    ``revision`` is deliberately absent: it records plan history, not a change
    in the research question, analysis family, cohort, robustness contract, or
    rationale. Structured values use canonical JSON so the signature remains
    stable when it is serialized into a step record and loaded on resume.
    """

    plan_payload = plan.model_dump(
        mode="json",
        include={"cohort", "robustness_specs", "display_labels"},
    )
    return (
        _normalise_scientific_text(plan.research_question),
        _normalise_scientific_text(plan.analysis_type),
        json.dumps(
            plan_payload.get("cohort"),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            plan_payload.get("robustness_specs", []),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            plan_payload.get("display_labels", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        _normalise_scientific_text(plan.rationale),
    )


def _serializable_plan_scientific_scope_signature(
    plan: AnalysisPlan,
) -> List[Optional[str]]:
    """Return the plan-level signature in manifest-safe form."""

    return list(_plan_scientific_scope_signature(plan))


def _plan_signature(
    plan: AnalysisPlan,
) -> Tuple[Any, ...]:
    """Substantive fingerprint of a plan's step DAG and scientific requests.

    Intent remains authoritative because several estimand coordinates are not
    yet structured in :class:`AnalysisStep`; only case and whitespace changes
    are cosmetic. Structured model requirements, ICU rules, trajectory specs,
    typed DAG edges, and result roles are also included.
    """
    return (
        _plan_scientific_scope_signature(plan),
        tuple(_step_scientific_signature(step) for step in plan.steps),
    )
