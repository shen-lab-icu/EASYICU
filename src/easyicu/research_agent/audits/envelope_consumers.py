"""Fail-closed Validator consumers for the StepResultEnvelope migration.

The bounded fraction/percentage adapter is wired only at the sealed final and
resume gate; early repair validation remains legacy.  The registered-output
adapter is still opt-in.  Both run the current Validator, compare its source
view with a canonical envelope, and retain the legacy decision only when both
views agree exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from easyicu.research_agent.authority.result_envelope_sidecar import (
    SUCCESSFUL_TERMINAL_STATUS,
    LoadedStepResultEnvelopeSidecar,
    StepResultEnvelopeSidecarLoad,
    StepResultEnvelopeSidecarQuery,
    StepResultEnvelopeSidecarUnavailable,
    load_current_step_result_envelope_sidecar,
    step_record_declares_sidecar,
)
from easyicu.research_agent.contracts.result_envelope import StepResultEnvelope
from easyicu.research_agent.schema import AnalysisStep, ValidationFinding

from .envelope_shadow import (
    canonical_registered_output_table_artifacts,
    compare_fraction_scale_shadow,
    fraction_scale_shadow_blocking_finding,
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


class RegisteredOutputEnvelopeConsumer(CrossStepRegisteredOutputValidator):
    """Envelope-authoritative registered-output consumer (M8).

    An upstream step's table presence is read ONLY from the canonical
    :class:`StepResultEnvelope` recovered through the M7 sidecar loader -- never
    from a raw ``evidence_ids`` / ``output_files`` glob, and never as an
    envelope-or-legacy choice.

    Two lanes, decided per upstream step by the step's own record (resume-safe):

    * The successful upstream record declares a sidecar (a modern run).  The
      canonical envelope must load and self-verify.  A missing / stale /
      coordinate-drifted / tampered / unreadable sidecar is a typed fail-close
      (``registered_output_sidecar_unrecoverable``), never a silent pass.
    * The record never declared a sidecar (a genuinely legacy archived run).
      The legacy raw table parse runs in an explicit diagnostic lane; its
      findings are ``diagnostic_only`` and carry ``paper_authority=False``.
    """

    name = CrossStepRegisteredOutputValidator.name

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

    @staticmethod
    def _sidecar_query(
        upstream_step: str, record: Dict[str, Any]
    ) -> StepResultEnvelopeSidecarQuery | None:
        attempt_id = str(record.get("attempt_id") or "").strip()
        checkpoint_id = str(
            record.get("review_checkpoint_id") or record.get("checkpoint_id") or ""
        ).strip()
        script_evidence_id = str(record.get("script_evidence_id") or "").strip()
        if not (attempt_id and checkpoint_id and script_evidence_id):
            return None
        try:
            return StepResultEnvelopeSidecarQuery(
                step_id=upstream_step,
                terminal_status=SUCCESSFUL_TERMINAL_STATUS,
                script_evidence_id=script_evidence_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
        except ValueError:
            return None

    def _load_upstream_envelope(
        self,
        upstream_step: str,
        record: Dict[str, Any],
        evidence_store: Any,
    ) -> StepResultEnvelopeSidecarLoad:
        query = self._sidecar_query(upstream_step, record)
        if query is None:
            return StepResultEnvelopeSidecarUnavailable(reason="incomplete_query")
        return load_current_step_result_envelope_sidecar(
            evidence_store=evidence_store, query=query
        )

    def _falsely_unavailable_finding(
        self,
        *,
        consumer_step_id: str,
        upstream_step: str,
        block: Dict[str, Any],
        table_artifacts: Sequence[str],
        source: str,
        diagnostic_only: bool,
    ) -> ValidationFinding:
        return ValidationFinding(
            validator=self.name,
            severity="error",
            message=(
                f"Registered upstream table was falsely reported unavailable in "
                f"step {consumer_step_id}: completed step {upstream_step} "
                f"registered table evidence {list(table_artifacts)}. Filter "
                "manifest records by the exact produced_by_step and table kind, "
                "resolve relative_path from the run directory, and use the sole "
                "compatible table even when its filename does not repeat the "
                "current step's semantic label."
            ),
            detail={
                "step_id": consumer_step_id,
                "summary_path": block["path"],
                "availability_key": block["availability_key"],
                "reported_path": block["reported_path"],
                "upstream_step": upstream_step,
                "registered_table_artifacts": list(table_artifacts),
                "table_presence_source": source,
                "diagnostic_only": diagnostic_only,
                "paper_authority": False,
            },
        )

    def _sidecar_unrecoverable_finding(
        self,
        *,
        consumer_step_id: str,
        upstream_step: str,
        reason: str,
    ) -> ValidationFinding:
        return ValidationFinding(
            validator=self.name,
            severity="error",
            message=(
                "Registered-output authority could not be recovered for "
                f"completed step {upstream_step} while validating step "
                f"{consumer_step_id}: its declared step-result envelope sidecar "
                f"was unrecoverable ({reason}). Refusing to read table presence "
                "from a raw output glob."
            ),
            detail={
                "step_id": consumer_step_id,
                "upstream_step": upstream_step,
                "registered_output_sidecar_unrecoverable": True,
                "sidecar_unavailable_reason": reason,
                "paper_authority": False,
            },
        )

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
        evidence_store: Any,
    ) -> List[ValidationFinding]:
        findings: list[ValidationFinding] = []
        for block in self._availability_blocks(step_summary):
            if block["available"]:
                continue
            upstream_step = block["upstream_step"]
            record = self._successful_upstream_record(
                upstream_step, completed_step_records
            )
            if record is None:
                # No successful upstream table producer: a genuine gap is allowed.
                continue
            if step_record_declares_sidecar(record.get("evidence_ids") or []):
                loaded = self._load_upstream_envelope(
                    upstream_step, record, evidence_store
                )
                if isinstance(loaded, LoadedStepResultEnvelopeSidecar):
                    canonical_artifacts = canonical_registered_output_table_artifacts(
                        loaded.envelope
                    )
                    if canonical_artifacts:
                        findings.append(
                            self._falsely_unavailable_finding(
                                consumer_step_id=step.step_id,
                                upstream_step=upstream_step,
                                block=block,
                                table_artifacts=canonical_artifacts,
                                source="canonical_envelope",
                                diagnostic_only=False,
                            )
                        )
                else:
                    findings.append(
                        self._sidecar_unrecoverable_finding(
                            consumer_step_id=step.step_id,
                            upstream_step=upstream_step,
                            reason=loaded.reason,
                        )
                    )
                continue
            # Legacy archived run: diagnostic lane over the legacy raw parse.
            legacy_artifacts = self._table_artifacts(record)
            if legacy_artifacts:
                findings.append(
                    self._falsely_unavailable_finding(
                        consumer_step_id=step.step_id,
                        upstream_step=upstream_step,
                        block=block,
                        table_artifacts=legacy_artifacts,
                        source="legacy_diagnostic",
                        diagnostic_only=True,
                    )
                )
        return findings


__all__ = [
    "RegisteredOutputEnvelopeConsumer",
    "StepSummaryFractionEnvelopeDualReader",
]
