"""Exact projection shared by planned and reported model contracts."""

from __future__ import annotations

from typing import Any, Mapping


MODEL_REQUIREMENT_REPORT_FIELDS = (
    "outcome",
    "outcome_type",
    "method_family",
    "exposure_source",
    "analysis_role",
    "analysis_set",
    "dependence",
)


def reported_model_requirement_fields(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one emitted model contract onto its planned comparison shape."""

    reported = {
        field: contract.get(field) for field in MODEL_REQUIREMENT_REPORT_FIELDS
    }
    reported["method_family"] = contract.get("method_family") or contract.get(
        "model_family"
    )
    return reported


__all__ = ["MODEL_REQUIREMENT_REPORT_FIELDS", "reported_model_requirement_fields"]
