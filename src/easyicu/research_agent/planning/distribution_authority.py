"""Bind distribution missing-value decisions to exact source semantics.

The compiler and pre-approval review share this owner. It diagnoses incompatible
policies; it never changes a denominator or treats unknown status as an event
absence on the Planner's behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..contracts.exposure_outcome_distribution import ExposureOutcomeDistributionSpec
from ..schema import ConceptDescriptor


DISTRIBUTION_MISSINGNESS_GUIDANCE = (
    "Distribution missing-value policies must match each exact source variable. "
    "structural_absence_is_non_event requires verified positive_only_event "
    "observation semantics for that outcome, not another variable's semantics "
    "or a sample with zero missing values. Unknown status remains unknown. "
    "When fail_closed conflicts with known missing observations, propose and "
    "justify an explicit missing-value and denominator policy; the host will "
    "not silently replace it with complete-case analysis."
)


@dataclass(frozen=True)
class DistributionPolicyIssue:
    field: str
    message: str


def distribution_policy_issues(
    spec: ExposureOutcomeDistributionSpec,
    *,
    variables: Mapping[str, ConceptDescriptor],
) -> tuple[DistributionPolicyIssue, ...]:
    issues: list[DistributionPolicyIssue] = []
    if spec.missing_outcome_policy == "structural_absence_is_non_event":
        descriptor = variables.get(spec.outcome)
        semantics = descriptor.observation_semantics if descriptor is not None else None
        if (
            semantics is None
            or semantics.kind != "positive_only_event"
            or semantics.representative_column != spec.outcome
        ):
            issues.append(DistributionPolicyIssue(
                "missing_outcome_policy",
                f"Outcome {spec.outcome!r} has no verified positive-only event "
                "representation authorizing structural_absence_is_non_event; "
                "unknown outcomes cannot be counted as non-events.",
            ))
    for role, name, policy in (
        ("exposure", spec.exposure, spec.missing_exposure_policy),
        ("outcome", spec.outcome, spec.missing_outcome_policy),
    ):
        descriptor = variables.get(name)
        missingness = descriptor.missingness if descriptor is not None else None
        if policy == "fail_closed" and missingness is not None and missingness.n_missing:
            issues.append(DistributionPolicyIssue(
                f"missing_{role}_policy",
                f"{role.capitalize()} {name!r} has {missingness.n_missing} known "
                "missing observations but the plan declares fail_closed. "
                "Revise the policy explicitly; the host cannot drop these rows "
                "or change the denominator automatically.",
            ))
    return tuple(issues)
