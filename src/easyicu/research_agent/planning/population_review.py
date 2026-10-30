"""Read sealed population facts and validate explicit acceptance of omissions.

An acceptance never changes which rows the plan selects. It answers only the
named, unapplied inclusions in one exact review request. Extraction stops and
all other scientific stops retain their own rules.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..authority.plan_review import PlanReviewAuthority
from ..canonical_json import canonical_sha256
from ..contracts.frozen_payload import thaw_payload
from .cohort_contract import CohortDefinition
from .population_compile import POPULATION_COMPILE_SCHEMA_VERSION
from .population_spec import POPULATION_SPEC_SCHEMA_VERSION, PopulationSpec

NOT_APPLIED_STOP = "population_inclusion_not_applied"


def sealed_population(request: Any, *, reason: str) -> dict | None:
    """Return typed facts only when the compilation still describes the plan."""
    try:
        payload = thaw_payload(request.payload)
        record = payload.get("population_compile")
        if (
            payload.get("reason") != reason
            or payload.get("approval_allowed") is not False
            or not isinstance(record, dict)
            or record.get("schema_version") != POPULATION_COMPILE_SCHEMA_VERSION
            or record.get("spec_schema_version") != POPULATION_SPEC_SCHEMA_VERSION
            or canonical_sha256(record) != payload.get("population_compile_sha256")
            or canonical_sha256(payload) != request.authority_sha256
        ):
            return None
        authority = PlanReviewAuthority.model_validate(
            payload.get("plan_review_authority")
        )
        compiled = record.get("cohort")
        offered = thaw_payload(authority.plan_payload).get("cohort")
        if not isinstance(compiled, Mapping) or not isinstance(offered, Mapping):
            return None
        # Use the cohort owner's normalization for omitted legacy defaults.
        left = CohortDefinition.from_dict(dict(compiled)).plan_dict()
        right = CohortDefinition.from_dict(dict(offered)).plan_dict()
        if {k: v for k, v in left.items() if k != "name"} != {
            k: v for k, v in right.items() if k != "name"
        }:
            return None
        PopulationSpec.model_validate(
            {"criteria": [row["criterion"] for row in record["criteria"]]}
        )
        if any(
            row["disposition"]
            not in {
                "applied_by_plan",
                "applied_by_source",
                "requires_extraction",
                "not_applied",
            }
            for row in record["criteria"]
        ):
            return None
        return record
    except (ValueError, KeyError, TypeError):
        return None


def unapplied_inclusions(request: Any) -> tuple[dict, ...]:
    record = sealed_population(request, reason=NOT_APPLIED_STOP)
    if record is None:
        return ()
    return tuple(
        row
        for row in record["criteria"]
        if (
            row["disposition"] == "not_applied"
            and row["criterion"]["role"] == "include"
        )
    )


def request_allows_approval(request: Any, accepted_ids: Sequence[str] = ()) -> bool:
    """Require every omitted inclusion, with no extras or duplicate answers.

    Ordinary approval alone still refuses this stop. The final decision binds
    the complete plan and evidence through the unchanged request authority.
    """
    if request.payload.get("approval_allowed") is not False:
        return not accepted_ids
    record = sealed_population(request, reason=NOT_APPLIED_STOP)
    if record is None or any(
        row["disposition"] == "not_applied" and row["criterion"]["role"] == "exclude"
        for row in record["criteria"]
    ):
        # The existing acceptance receipt authorizes only omitted inclusions;
        # accepting those cannot waive an unrelated exclusion. Revise the plan
        # rather than rewriting an old signed acceptance into broader authority.
        return False
    required = unapplied_inclusions(request)
    ids = [row["criterion"]["id"] for row in required]
    return (
        bool(ids)
        and len(accepted_ids) == len(set(accepted_ids))
        and set(accepted_ids) == set(ids)
    )


def acceptance_record(request: Any, accepted_ids: Sequence[str]) -> dict:
    """Keep the exact accepted facts beside the eventual full-plan decision."""
    if not accepted_ids or not request_allows_approval(request, accepted_ids):
        raise ValueError("population acceptance does not cover the sealed review")
    return {
        "schema_version": "easyicu.population_inclusion_acceptance/1",
        "population_compile_sha256": request.payload["population_compile_sha256"],
        "plan_sha256": request.payload["plan_review_authority"]["plan_sha256"],
        "criteria": list(unapplied_inclusions(request)),
        "accepted_criterion_ids": sorted(accepted_ids),
        "effect": "accept_unapplied_inclusions_without_changing_plan_rows",
    }
