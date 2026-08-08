"""Unique endpoint-authority validation at the plan/context boundary."""

from __future__ import annotations

from typing import Literal, Optional

from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from .analysis_types import required_endpoint_kind_for_family


def endpoint_contract_findings(
    plan: AnalysisPlan,
    *,
    context: Optional[ResearchContext] = None,
    severity: Literal["info", "warning", "error"] = "warning",
) -> list[ValidationFinding]:
    """Validate an optional plan projection against sealed context authority."""

    required_kind = required_endpoint_kind_for_family(plan.analysis_type)
    projection = plan.endpoint
    if context is None:
        if required_kind is None and projection is None:
            return []
        return [
            ValidationFinding(
                validator="endpoint_contract",
                severity=severity,
                message=(
                    "Endpoint validation requires the sealed ResearchContext; "
                    "AnalysisPlan.endpoint is only a backward-compatible projection "
                    "and cannot become study authority on its own."
                ),
                detail={"reason": "endpoint_context_authority_required"},
            )
        ]

    endpoint = context.endpoint
    if projection is not None and projection != endpoint:
        return [
            ValidationFinding(
                validator="endpoint_contract",
                severity=severity,
                message=(
                    "AnalysisPlan.endpoint does not equal the sealed "
                    "ResearchContext.endpoint. The context is the unique study "
                    "authority; repair or remove the stale plan projection."
                ),
                detail={
                    "reason": "endpoint_projection_mismatch",
                    "context_endpoint": (
                        endpoint.model_dump(mode="json")
                        if endpoint is not None
                        else None
                    ),
                    "plan_endpoint": projection.model_dump(mode="json"),
                },
            )
        ]
    if required_kind is None:
        return []
    declared_kind = getattr(endpoint, "kind", None)
    if declared_kind == required_kind:
        return []
    return [
        ValidationFinding(
            validator="endpoint_contract",
            severity=severity,
            message=(
                f"The plan declares analysis_type={plan.analysis_type!r}, whose "
                f"registry entry requires a typed endpoint of kind {required_kind!r}, "
                "but ResearchContext.endpoint "
                + (
                    "is absent"
                    if endpoint is None
                    else f"declares kind {declared_kind!r}"
                )
                + ". Follow-up time, origin, censoring, and event levels cannot be "
                "recovered from names, dtypes, or prose. Seal the context endpoint "
                "with kind, levels, event_column, time_column, time_origin and "
                "censoring_rule."
            ),
            detail={
                "analysis_type": plan.analysis_type,
                "required_endpoint_kind": required_kind,
                "declared_endpoint_kind": declared_kind,
                "reason": "endpoint_context_missing_or_wrong_kind",
            },
        )
    ]


__all__ = ["endpoint_contract_findings"]
