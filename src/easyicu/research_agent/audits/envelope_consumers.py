"""Fail-closed Validator consumers for the StepResultEnvelope migration.

The bounded fraction/percentage adapter is wired only at the sealed final and
resume gate; early repair validation remains legacy.  The registered-output
adapter is still opt-in.  Both run the current Validator, compare its source
view with a canonical envelope, and retain the legacy decision only when both
views agree exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from easyicu.research_agent.execution.result_envelope import StepResultEnvelope
from easyicu.research_agent.schema import AnalysisStep, ValidationFinding

from .envelope_shadow import (
    compare_fraction_scale_shadow,
    compare_registered_output_shadow,
    fraction_scale_shadow_blocking_finding,
    registered_output_shadow_blocking_finding,
)
from .validators import (
    CrossStepRegisteredOutputValidator,
    StepSummaryFractionValidator,
)


class StepSummaryFractionEnvelopeDualReader(StepSummaryFractionValidator):
    """Retain the legacy bounded-metric decision only after exact comparison."""

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        envelope: StepResultEnvelope | None,
        current_status: str | None = None,
        legacy_findings: Sequence[ValidationFinding] | None = None,
    ) -> List[ValidationFinding]:
        retained_legacy = (
            list(legacy_findings)
            if legacy_findings is not None
            else super().audit(step=step, step_summary=step_summary)
        )
        comparison = compare_fraction_scale_shadow(
            step=step,
            step_summary=step_summary,
            current_status=current_status,
            envelope=envelope,
            legacy_findings=retained_legacy,
        )
        if comparison.exact_match:
            return retained_legacy
        return [
            fraction_scale_shadow_blocking_finding(
                validator_name=self.name,
                step_id=step.step_id,
                comparison=comparison,
            )
        ]


class CrossStepRegisteredOutputEnvelopeDualReader(CrossStepRegisteredOutputValidator):
    """Observe registered-output envelopes without changing live wiring."""

    @classmethod
    def _successful_upstream_record(
        cls,
        upstream_step: str,
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        for record in reversed(completed_step_records):
            if str(record.get("step_id") or "") != upstream_step:
                continue
            record_status = str(record.get("status") or "").strip().lower()
            if record_status and record_status not in cls._SUCCESSFUL_STATUSES:
                continue
            summary = record.get("step_summary")
            if isinstance(summary, dict):
                summary_status = str(summary.get("status") or "").strip().lower()
                if summary_status and summary_status not in cls._SUCCESSFUL_STATUSES:
                    continue
            return record
        return None

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
        completed_step_envelopes: Mapping[str, StepResultEnvelope],
    ) -> List[ValidationFinding]:
        legacy_findings = super().audit(
            step=step,
            step_summary=step_summary,
            completed_step_records=completed_step_records,
        )
        blockers: list[ValidationFinding] = []
        blocked_upstream_steps: set[str] = set()
        for block in self._availability_blocks(step_summary):
            if block["available"]:
                continue
            upstream_step = block["upstream_step"]
            record = self._successful_upstream_record(
                upstream_step,
                completed_step_records,
            )
            if record is None:
                continue
            comparison = compare_registered_output_shadow(
                step_id=upstream_step,
                step_summary=record.get("step_summary"),
                current_status=(
                    str(record.get("status")).strip()
                    if record.get("status") is not None
                    else None
                ),
                legacy_table_artifacts=self._table_artifacts(record),
                envelope=completed_step_envelopes.get(upstream_step),
            )
            if comparison.exact_match:
                continue
            blocked_upstream_steps.add(upstream_step)
            blockers.append(
                registered_output_shadow_blocking_finding(
                    validator_name=self.name,
                    consumer_step_id=step.step_id,
                    upstream_step=upstream_step,
                    comparison=comparison,
                )
            )
        retained_legacy = [
            finding
            for finding in legacy_findings
            if finding.detail.get("upstream_step") not in blocked_upstream_steps
        ]
        return retained_legacy + blockers


__all__ = [
    "CrossStepRegisteredOutputEnvelopeDualReader",
    "StepSummaryFractionEnvelopeDualReader",
]
