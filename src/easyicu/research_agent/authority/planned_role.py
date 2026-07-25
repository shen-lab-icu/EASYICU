"""Strict Planner-owned analysis-role authority for step records.

The Planner declares the scientific role once on :class:`AnalysisStep`.  The
host persists that literal both on the outer step record and in the immutable
``analysis_request.step`` snapshot.  Consumers must verify both bindings; they
must never infer a primary result from prose, method names, filenames, or model
self-claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, get_args

from ..schema import PlannedAnalysisRole

__all__ = [
    "PLANNED_ANALYSIS_ROLES",
    "unique_verified_primary_record",
    "verified_planned_analysis_role",
]

PLANNED_ANALYSIS_ROLES = frozenset(get_args(PlannedAnalysisRole))

# A probe is created before a Planner step exists, so it cannot carry an
# analysis-request snapshot.  It is the sole exception.  Planned deterministic
# steps (including cohort materialisation) must carry the same two bindings as
# ordinary Coder steps.
_HOST_UNPLANNED_AUXILIARY_AUTHORITIES = {
    "host_deterministic_probe": "00_probe",
}


def verified_planned_analysis_role(record: Mapping[str, Any]) -> Optional[str]:
    """Return an exact, doubly bound Planner role or fail closed with ``None``."""

    outer_role = record.get("planned_analysis_role")
    if not isinstance(outer_role, str) or outer_role not in PLANNED_ANALYSIS_ROLES:
        return None

    analysis_request = record.get("analysis_request")
    request_step = (
        analysis_request.get("step") if isinstance(analysis_request, Mapping) else None
    )
    if not isinstance(request_step, Mapping):
        authority_kind = record.get("step_authority_kind")
        outer_step_id = record.get("step_id")
        if (
            outer_role == "auxiliary"
            and isinstance(authority_kind, str)
            and isinstance(outer_step_id, str)
            and _HOST_UNPLANNED_AUXILIARY_AUTHORITIES.get(authority_kind)
            == outer_step_id
        ):
            return "auxiliary"
        return None

    embedded_role = request_step.get("planned_analysis_role")
    if (
        not isinstance(embedded_role, str)
        or embedded_role not in PLANNED_ANALYSIS_ROLES
        or embedded_role != outer_role
    ):
        return None

    outer_step_id = record.get("step_id")
    embedded_step_id = request_step.get("step_id")
    if (
        not isinstance(outer_step_id, str)
        or not outer_step_id
        or not isinstance(embedded_step_id, str)
        or embedded_step_id != outer_step_id
    ):
        return None
    return outer_role


def unique_verified_primary_record(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Return the sole verified primary record, rejecting any invalid ledger."""

    primary_records: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            return None
        role = verified_planned_analysis_role(record)
        if role is None:
            return None
        if role == "primary":
            primary_records.append(record)
    return primary_records[0] if len(primary_records) == 1 else None
