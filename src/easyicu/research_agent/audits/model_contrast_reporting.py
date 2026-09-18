"""Require reportable comparisons from the native aggregate projection.

This is an output-completeness gate, not a new model or a scientific plan
revision. On resume the existing gate replay retires the incomplete projection
and its dependent products while keeping the sealed parent fit reusable.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..authority.model_contrast_scientific_claims import derive_model_contrast_claim_payloads
from ..schema import ValidationFinding


def model_contrast_reporting_findings(
    *, step_record: Mapping[str, Any], step_summary: Mapping[str, Any]
) -> list[ValidationFinding]:
    """Validate only the explicit native aggregate-reporting owner."""

    if step_record.get("deterministic_standard_analysis") != "signed_landmark_spline_robustness":
        return []
    if step_summary.get("status") != "ok":
        # Execution failure already has its own diagnostic. Missing report
        # fields must not distract a repair from that original failure.
        return []
    try:
        if "reportable_model_contrasts" not in step_summary:
            raise ValueError("completed aggregate projection lacks the model contrast reporting contract")
        derive_model_contrast_claim_payloads(dict(step_summary))
    except ValueError as exc:
        return [ValidationFinding(
            validator="model_contrast_reporting",
            severity="error",
            message="The native model-result projection is not complete enough for reporting.",
            detail={
                "kind": "native_model_contrast_reporting_incomplete",
                "scope": "aggregate_reporting_projection",
                "reason": str(exc),
                "repair": "Regenerate the aggregate projection from its sealed parent result tables; retain the parent model fit.",
            },
        )]
    return []
