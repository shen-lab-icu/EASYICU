"""Bind distribution missing-value decisions to exact source semantics.

The request, compiler and pre-approval review share these exact-column facts.
The family template binds complete-case defaults before review; validation
diagnoses incompatible policies without changing an executable specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ..contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY, read_grouping_rules
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


class DistributionMissingnessFacts(BaseModel):
    """Measured missingness and the exact grouping's declared exclusion.

    A metadata placeholder with no denominator is unknown, not a measured
    zero. No source column's missing count is imputed to a derived column.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_missing: int | None = Field(default=None, ge=0)
    exclusion_declared: bool = False

    @property
    def requires_exclusion(self) -> bool:
        return bool(self.n_missing) or self.exclusion_declared


def distribution_missingness_facts(
    variables: Mapping[str, ConceptDescriptor],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, DistributionMissingnessFacts]:
    """Read the same facts for host defaults and compile-time refusals.

    The context builder projects grouping parameters from verified staged
    receipts or strictly read planned columns. Only that exact grouping's
    ``exclude`` declaration authorizes exclusion without measured missingness.
    """

    exclusions: set[str] = set()
    for item in (provenance or {}).get(EXPOSURE_GROUP_LABELS_KEY) or ():
        if item.get("variable") in variables and item.get("parameters") is not None:
            if read_grouping_rules(item["parameters"]).unmeasured == "exclude":
                exclusions.add(item["variable"])
    facts = {}
    for name, descriptor in variables.items():
        missingness = descriptor.missingness
        facts[name] = DistributionMissingnessFacts(
            n_missing=(
                missingness.n_missing
                if missingness is not None and (missingness.n_total or missingness.n_missing)
                else None
            ),
            exclusion_declared=name in exclusions,
        )
    return facts


def distribution_policy_issues(
    spec: ExposureOutcomeDistributionSpec,
    *,
    variables: Mapping[str, ConceptDescriptor],
    provenance: Mapping[str, Any] | None = None,
) -> tuple[DistributionPolicyIssue, ...]:
    issues: list[DistributionPolicyIssue] = []
    facts = distribution_missingness_facts(variables, provenance=provenance)
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
        missingness = facts.get(name, DistributionMissingnessFacts())
        if policy == "fail_closed" and missingness.n_missing:
            issues.append(DistributionPolicyIssue(
                f"missing_{role}_policy",
                f"{role.capitalize()} {name!r} has {missingness.n_missing} known "
                "missing observations but the plan declares fail_closed. "
                "Revise the policy explicitly; the host cannot drop these rows "
                "or change the denominator automatically.",
            ))
        elif policy == "fail_closed" and missingness.exclusion_declared:
            issues.append(DistributionPolicyIssue(
                f"missing_{role}_policy",
                f"{role.capitalize()} {name!r} declares unmeasured handling 'exclude' "
                "but the plan declares fail_closed. Bind the declared exclusion "
                "to exclude_from_denominator before review.",
            ))
    return tuple(issues)
