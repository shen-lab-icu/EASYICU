"""Fail-closed Validator consumers for the StepResultEnvelope migration.

The bounded fraction/percentage and registered-output adapters are wired at the
sealed final and resume gates; early repair validation remains legacy. Both run
the current Validator, compare its source view with a canonical envelope, and
retain a decision only when both views agree exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from easyicu.research_agent.authority.result_envelope_sidecar import (
    SUCCESSFUL_TERMINAL_STATUS,
    LoadedStepResultEnvelopeSidecar,
    StepResultEnvelopeSidecarLoad,
    StepResultEnvelopeSidecarQuery,
    StepResultEnvelopeSidecarUnavailable,
    load_current_step_result_envelope_sidecar,
    step_record_declares_sidecar,
)
from easyicu.research_agent.contracts.result_envelope import (
    StepResultEnvelope,
    rebuild_observed_scalar_tree,
)
from easyicu.research_agent.schema import AnalysisStep, ValidationFinding

from .envelope_shadow import (
    canonical_registered_output_table_artifacts,
    compare_fraction_scale_shadow,
    compare_validator_shadow_inputs,
    fraction_scale_shadow_blocking_finding,
    fraction_scale_shadow_observed_findings,
)
from .validators import (
    CrossStepRegisteredOutputValidator,
    StepSummaryFractionValidator,
)


class RegisteredOutputAuthorityError(RuntimeError):
    """A live writer could not recover one verified result-envelope authority."""


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
            # The two DECISIONS agree. A canonical normalizer that complained
            # about the input it was handed is a migration observation, not a
            # disagreement, and it is recorded rather than allowed to replace a
            # verdict both views reached.
            #
            # MEASURED over every recorded run: 11 distinct steps across 5 of
            # the 9 tasks died ``contract_failed`` here, and in ALL ELEVEN
            # ``legacy_findings_sha256 == canonical_findings_sha256`` with zero
            # findings on both sides. 8 of the 11 never attempted a repair --
            # correctly, since nothing in the agent's script can reconcile two
            # host implementations. One of them (m2's feature audit, today) I
            # verified by hand: both of its count/fraction pairs reconcile to
            # six decimals, and the canonical normalizer had paired a
            # stratum-level count with the whole-cohort denominator because it
            # selects the denominator from a fixed list of column names.
            return [
                *retained_legacy,
                *fraction_scale_shadow_observed_findings(
                    validator_name=self.name,
                    step_id=step.step_id,
                    comparison=comparison,
                ),
            ]
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

    def _falsely_available_finding(
        self,
        *,
        consumer_step_id: str,
        upstream_step: str,
        block: Dict[str, Any],
    ) -> ValidationFinding:
        return ValidationFinding(
            validator=self.name,
            severity="error",
            message=(
                f"Step {consumer_step_id} reported a registered upstream table "
                f"from {upstream_step} as available, but the verified result "
                "envelope registers no table artifact."
            ),
            detail={
                "step_id": consumer_step_id,
                "summary_path": block["path"],
                "availability_key": block["availability_key"],
                "reported_path": block["reported_path"],
                "upstream_step": upstream_step,
                "registered_table_artifacts": [],
                "table_presence_source": "canonical_envelope",
                "registered_output_availability_disagreement": True,
                "diagnostic_only": False,
                "paper_authority": False,
            },
        )

    def _legacy_envelope_disagreement_finding(
        self,
        *,
        consumer_step_id: str,
        upstream_step: str,
        mismatch_codes: Sequence[str],
    ) -> ValidationFinding:
        return ValidationFinding(
            validator=self.name,
            severity="error",
            message=(
                "Registered-output envelope authority disagreed with the sealed "
                f"legacy step_summary for completed step {upstream_step} while "
                f"validating step {consumer_step_id}; refusing either view."
            ),
            detail={
                "step_id": consumer_step_id,
                "upstream_step": upstream_step,
                "registered_output_authority_disagreement": True,
                "mismatch_codes": sorted(set(mismatch_codes)),
                "paper_authority": False,
            },
        )

    def authoritative_writer_records(
        self,
        completed_step_records: Sequence[Mapping[str, Any]],
        *,
        evidence_store: Any,
    ) -> List[Dict[str, Any]]:
        """Project current records through their verified envelope sidecars.

        Unlike the validator's archived diagnostic lane, a live manuscript has
        no authority fallback: every successful step must declare and recover a
        current sidecar compiled from the exact legacy summary in the ledger.
        """

        authoritative: list[Dict[str, Any]] = []
        for raw_record in completed_step_records:
            record = dict(raw_record)
            status = str(record.get("status") or "").strip().lower()
            if status not in self._SUCCESSFUL_STATUSES:
                authoritative.append(record)
                continue
            step_id = str(record.get("step_id") or "").strip()
            if not step_id or not step_record_declares_sidecar(
                record.get("evidence_ids") or []
            ):
                raise RegisteredOutputAuthorityError(
                    f"successful step {step_id or '<missing>'} has no live "
                    "step-result envelope sidecar authority"
                )
            loaded = self._load_upstream_envelope(step_id, record, evidence_store)
            if not isinstance(loaded, LoadedStepResultEnvelopeSidecar):
                raise RegisteredOutputAuthorityError(
                    f"step-result envelope authority for {step_id} is "
                    f"unrecoverable ({loaded.reason})"
                )
            comparison = compare_validator_shadow_inputs(
                step_summary=record.get("step_summary"),
                envelope=loaded.envelope,
                current_status=str(record.get("status") or "") or None,
            )
            if not comparison.exact_match:
                codes = sorted(
                    {item.code for item in comparison.decisive_mismatches}
                )
                raise RegisteredOutputAuthorityError(
                    f"step-result envelope authority for {step_id} disagrees "
                    f"with step_summary ({', '.join(codes)})"
                )
            canonical_summary = rebuild_observed_scalar_tree(
                loaded.envelope.observed_scalars
            )
            if canonical_summary is None:
                raise RegisteredOutputAuthorityError(
                    f"step-result envelope authority for {step_id} has an invalid "
                    "canonical scalar tree"
                )
            record["step_summary"] = canonical_summary
            record["writer_result_envelope_evidence_id"] = loaded.evidence_id
            authoritative.append(record)
        return authoritative

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
                    comparison = compare_validator_shadow_inputs(
                        step_summary=record.get("step_summary"),
                        envelope=loaded.envelope,
                        current_status=str(record.get("status") or "") or None,
                    )
                    if not comparison.exact_match:
                        findings.append(
                            self._legacy_envelope_disagreement_finding(
                                consumer_step_id=step.step_id,
                                upstream_step=upstream_step,
                                mismatch_codes=[
                                    item.code
                                    for item in comparison.decisive_mismatches
                                ],
                            )
                        )
                        continue
                    canonical_artifacts = canonical_registered_output_table_artifacts(
                        loaded.envelope
                    )
                    if canonical_artifacts and not block["available"]:
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
                    elif block["available"] and not canonical_artifacts:
                        findings.append(
                            self._falsely_available_finding(
                                consumer_step_id=step.step_id,
                                upstream_step=upstream_step,
                                block=block,
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
            if block["available"]:
                continue
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
    "RegisteredOutputAuthorityError",
    "RegisteredOutputEnvelopeConsumer",
    "StepSummaryFractionEnvelopeDualReader",
]
