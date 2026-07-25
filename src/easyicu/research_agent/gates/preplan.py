"""Host-owned data gates that run before Planner/provider execution."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..audits.validators import CohortAuditor
from ..schema import ResearchContext, ValidationFinding
from .data_answerability import analysis_answerability_findings


def preplan_data_findings(
    *,
    context: ResearchContext,
    cohort_path: Path,
) -> list[ValidationFinding]:
    """Return integrity and proven scientific-infeasibility findings."""

    return [
        *CohortAuditor().audit(context=context, cohort_path=cohort_path),
        *analysis_answerability_findings(context),
    ]


def preplan_data_failure_reason(findings: Sequence[ValidationFinding]) -> str:
    """Classify a blocking pre-plan result without hiding its typed findings."""

    if any(
        finding.severity == "error" and finding.validator == "data_answerability_gate"
        for finding in findings
    ):
        return "data_answerability_failed"
    return "cohort_audit_failed"


__all__ = ["preplan_data_failure_reason", "preplan_data_findings"]
