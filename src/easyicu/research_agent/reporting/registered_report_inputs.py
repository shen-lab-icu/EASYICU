"""Read-only admission of registered aggregate inputs for Web report repair.

No EvidenceStore constructor, cohort parser, runtime probe, or execution entry
point is used here. Old execution identity remains old execution identity.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ..audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
from ..authority.evidence_snapshot import load_current_evidence_snapshot
from ..authority.evidence_store import EvidenceStore, EvidenceEnforcementMode
from ..authority.numeric_claim_identity import NumericClaim
from ..authority.manuscript_claim_policy import expand_scientific_claim_tokens
from ..authority.scientific_claim_registry import load_registered_scientific_claims
from ..authority.runtime_artifacts import (
    verified_run_evidence_path,
    current_step_records,
)
from ..schema import EvidenceRecord
from .writer_evidence import _executed_method_boundary_rows
from .writer_only_migration import (
    PreparedWriterOnlyMigration,
    WriterOnlyMigrationError,
    prepare_writer_only_migration,
)


class ReadOnlyReportEvidence:
    """The small read-only interface needed by the aggregate envelope reader."""

    def __init__(self, root: Path) -> None:
        self.root = root
        snapshot = load_current_evidence_snapshot(root)
        self._records = tuple(
            EvidenceRecord.model_validate(row) for row in snapshot.records
        )
        self._aliases = dict(snapshot.aliases)
        self._numeric_claims = tuple(
            NumericClaim.from_dict(row) for row in snapshot.numeric_claims
        )

    def numeric_claims(self):
        return list(self._numeric_claims)

    def current_verified_records(self, records):
        return EvidenceStore.current_verified_records(self, records)

    def authoritative_numeric_claims(self, records):
        return EvidenceStore.authoritative_numeric_claims(self, records)

    def records(self):
        return list(self._records)

    def aliases(self):
        return dict(self._aliases)

    def get(self, key: str):
        canonical = self._aliases.get(key, key)
        return next(
            (row for row in self._records if row.evidence_id == canonical), None
        )

    def verify_input(self, name: str, evidence_id: str) -> bytes:
        record = self.get(evidence_id)
        sealed = verified_run_evidence_path(self.root, record) if record else None
        source = self.root / name
        if (
            sealed is None
            or source.is_symlink()
            or not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != record.sha256
        ):
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_REGISTERED_INPUT_CHANGED",
                detail=name,
            )
        return sealed.read_bytes()


def prepare_registered_report_repair(run_dir: Path) -> PreparedWriterOnlyMigration:
    """Require sealed inputs and completed analysis before any Writer call."""

    evidence = ReadOnlyReportEvidence(Path(run_dir).resolve(strict=True))
    for name, evidence_id in (
        ("research_context.json", "research_context"),
        ("analysis_plan.json", "analysis_plan"),
        ("preplan_literature_bundle.json", "preplan_literature_bundle"),
        ("writer_evidence_digest.md", "writer_evidence_digest"),
        ("manuscript_scaffold.md", "manuscript_scaffold_raw"),
    ):
        evidence.verify_input(name, evidence_id)
    status = json.loads(evidence.verify_input("run_status.json", "run_status"))
    gates = status.get("gates", {})
    if (
        any(
            gates.get(key) is not True
            for key in (
                "execution_complete",
                "analysis_validated",
                "numeric_verified",
            )
        )
        or gates.get("failed_steps")
        or gates.get("missing_steps")
    ):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_COMPLETED_ANALYSIS_REQUIRED",
            detail="Only a completed, validated analysis can enter report-only repair.",
        )
    prepared = prepare_writer_only_migration(evidence.root)
    if prepared.plan is None:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_CURRENT_PLAN_REQUIRED",
            detail="The sealed plan must parse.",
        )
    # Resolve every method from its sealed envelope and exact source summary.
    # Only refresh this host-generated block; preserve the sealed numeric and
    # scientific-claim digest byte-for-byte outside it.
    manifest = json.loads((evidence.root / "manifest.json").read_text())
    records = manifest.get("per_step_records", [])
    projected = RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
        records,
        evidence_store=evidence,
    )
    planned_ids = {step.step_id for step in prepared.plan.steps}
    # The consumer also verifies dedicated host cohort/probe authorities, then
    # deliberately omits these non-result records from the Writer digest.
    completed_ids = {
        row.get("step_id")
        for row in current_step_records(records)
        if row.get("status") == "ok"
    }
    if not planned_ids or not planned_ids.issubset(completed_ids):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_PLAN_RESULTS_INCOMPLETE",
            detail="Registered plan steps are missing.",
        )
    rows = _executed_method_boundary_rows(projected, evidence=evidence)
    marker = "\n## EXECUTED METHOD BOUNDARY"
    digest = prepared.evidence_digest
    if marker in digest:
        before, rest = digest.split(marker, 1)
        next_heading = rest.find("\n## ")
        after = rest[next_heading:] if next_heading >= 0 else ""
        digest = (
            before + marker + "\n" + json.dumps(rows, ensure_ascii=False) + "\n" + after
        )
    else:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_METHOD_DIGEST_UNAVAILABLE",
            detail="No registered method block.",
        )
    return replace(prepared, evidence_digest=digest)


def bind_registered_report_numbers(run_dir: Path, manuscript: str) -> tuple[str, int]:
    """Apply the same strict per-value/owner gate as the full report pipeline."""

    from .manuscript_post import bind_numeric_values
    from ..audits.manuscript_claims import audit_manuscript_numeric_claims

    evidence = ReadOnlyReportEvidence(run_dir)
    records = json.loads((run_dir / "manifest.json").read_text())["per_step_records"]
    RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
        records,
        evidence_store=evidence,
    )
    claims = load_registered_scientific_claims(root=run_dir, records=evidence.records())
    expanded = expand_scientific_claim_tokens(
        manuscript,
        resolve_claim={claim.claim_ref: claim for claim in claims}.get,
        current_evidence_ids={
            row.evidence_id for row in evidence.current_verified_records(records)
        },
    )
    if expanded.missing_claim_refs or expanded.malformed_sentences:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_CLAIM_BINDING_FAILED",
            detail="Unresolved scientific claims.",
        )
    bound, bindings, untraced = bind_numeric_values(
        expanded.scaffold,
        evidence=evidence,
        enforcement_mode=EvidenceEnforcementMode.STRICT,
        per_step_records=records,
    )
    findings = audit_manuscript_numeric_claims(bound, per_step_records=records)
    if untraced or any(row.severity == "error" for row in findings):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_NUMERIC_AUDIT_FAILED",
            detail="Numeric claims did not match registered results.",
        )
    return bound, len(bindings)


__all__ = ["prepare_registered_report_repair", "bind_registered_report_numbers"]
