"""ValidationFinding adapter for association sensitivity result contracts."""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import AnalysisStep, ValidationFinding
from .association_execution import association_binary_sensitivity_result_issues


def association_binary_sensitivity_findings(
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
) -> list[ValidationFinding]:
    """Translate pure contract issues into the shared validation envelope."""

    return [
        ValidationFinding(
            validator="association_binary_sensitivity_contract",
            severity="error",
            message=issue.message,
            detail={
                "kind": issue.reason_code,
                "step_id": step.step_id,
                **dict(issue.detail),
            },
        )
        for issue in association_binary_sensitivity_result_issues(step, step_summary)
    ]


__all__ = ["association_binary_sensitivity_findings"]
