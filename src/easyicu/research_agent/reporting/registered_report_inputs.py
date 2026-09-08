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
from ..schema import AnalysisPlan
from ..literature import LiteratureBundle
from .manuscript_reader import build_manuscript_reader
from .writer_evidence import _render_writer_evidence_digest_v2
from ..research_context.typed import parse_research_context_json
from .descriptive_report_facts import (
    compile_counts_only_report_facts, render_descriptive_report_claims,
    verified_descriptive_source_records,
)
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

    def scientific_claims(self):
        return list(load_registered_scientific_claims(root=self.root, records=self._records))

    def authoritative_scientific_claims(self, records):
        return EvidenceStore.authoritative_scientific_claims(self, records)

    def records(self):
        return list(self._records)

    def aliases(self):
        return dict(self._aliases)

    def get(self, key: str):
        canonical = self._aliases.get(key, key)
        return next(
            (row for row in self._records if row.evidence_id == canonical), None
        )

    def latest_sealed_record(self, evidence_id: str):
        record = self.get(evidence_id)
        # Stable citation ids deliberately retain their first version. A live
        # input file must instead match the newest explicitly sealed revision,
        # never an arbitrary same-byte record or an older rollback candidate.
        revisions = [row for row in self._records if (
            row.evidence_id == evidence_id
            or (row.metadata or {}).get("resume_supersedes") == evidence_id
        )]
        if revisions:
            record = revisions[-1]
        return record

    def read_sealed(self, evidence_id: str) -> bytes:
        record = self.latest_sealed_record(evidence_id)
        sealed = verified_run_evidence_path(self.root, record) if record else None
        if sealed is None:
            raise WriterOnlyMigrationError(code="WRITER_ONLY_REGISTERED_INPUT_CHANGED", detail=evidence_id)
        return sealed.read_bytes()

    def verify_input(self, name: str, evidence_id: str) -> bytes:
        record = self.latest_sealed_record(evidence_id)
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
        ("manuscript_scaffold.md", "manuscript_scaffold_raw"),
    ):
        evidence.verify_input(name, evidence_id)
    # Reporting caches can change after a failed Writer without a corresponding
    # new seal in older runs. Never trust those mutable caches as authority.
    # Require an immutable completed-analysis receipt AND revalidate every
    # current planned output below. Manuscript numeric success is an OUTPUT
    # gate of repair, not a prerequisite that makes failed writing unrepairable.
    status = json.loads(evidence.read_sealed("run_status"))
    gates = status.get("gates", {})
    if (
        any(
            gates.get(key) is not True
            for key in (
                "execution_complete",
                "analysis_validated",
            )
        )
        or gates.get("failed_steps")
        or gates.get("missing_steps")
    ):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_COMPLETED_ANALYSIS_REQUIRED",
            detail="Only a completed, validated analysis can enter report-only repair.",
        )
    plan = AnalysisPlan.model_validate_json(evidence.verify_input("analysis_plan.json", "analysis_plan"))
    context = parse_research_context_json(evidence.verify_input("research_context.json", "research_context"))
    manifest = json.loads((evidence.root / "manifest.json").read_text())
    records = current_step_records(manifest.get("per_step_records", []))
    _require_completed_plan_records(plan, records)
    projected = RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
        records,
        evidence_store=evidence,
    )
    digest = _render_writer_evidence_digest_v2(
        projected, context=context, run_dir=evidence.root, evidence=evidence,
    )
    prepared = prepare_writer_only_migration(
        evidence.root, host_verified_evidence_digest=digest,
    )
    return replace(
        prepared, evidence_digest=digest,
        host_result_facts=compile_counts_only_report_facts(
            verified_descriptive_source_records(projected, evidence),
            evidence=evidence, reader_display_labels=prepared.plan.display_labels,
            scientific_claims=load_registered_scientific_claims(root=evidence.root, records=evidence.records()),
        ),
    )


def _require_completed_plan_records(plan, records):
    """An earlier success cannot conceal a newer failure or a missing step."""
    current = current_step_records(records)
    planned_ids = {step.step_id for step in plan.steps}
    completed_ids = {row.get("step_id") for row in current if row.get("status") == "ok"}
    if (not planned_ids or not planned_ids.issubset(completed_ids)
            or any(row.get("status") != "ok" for row in current)):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_PLAN_RESULTS_INCOMPLETE",
            detail="Every current planned step must be complete before report-only repair.",
        )


def bind_registered_report_numbers(run_dir: Path, manuscript: str) -> tuple[str, int]:
    """Apply the same strict per-value/owner gate as the full report pipeline."""

    from .manuscript_post import bind_numeric_values
    from ..audits.manuscript_claims import audit_manuscript_numeric_claims

    evidence = ReadOnlyReportEvidence(run_dir)
    records = json.loads((run_dir / "manifest.json").read_text())["per_step_records"]
    projected = RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
        records,
        evidence_store=evidence,
    )
    claims = load_registered_scientific_claims(root=run_dir, records=evidence.records())
    plan = AnalysisPlan.model_validate_json(evidence.verify_input("analysis_plan.json", "analysis_plan"))
    facts = compile_counts_only_report_facts(
        verified_descriptive_source_records(projected, evidence), evidence=evidence,
        reader_display_labels=plan.display_labels, scientific_claims=claims,
    )
    manuscript = render_descriptive_report_claims(manuscript, facts)
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
        literature=LiteratureBundle.model_validate_json(evidence.verify_input(
            "preplan_literature_bundle.json", "preplan_literature_bundle",
        )),
    )
    findings = audit_manuscript_numeric_claims(bound, per_step_records=records)
    if untraced or any(row.severity == "error" for row in findings):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_NUMERIC_AUDIT_FAILED",
            detail="Numeric claims did not match registered results.",
        )
    return bound, len(bindings)


def build_registered_report_reader(run_dir: Path, manuscript: str) -> dict:
    """Build the current report reader from unchanged registered source inputs."""

    evidence = ReadOnlyReportEvidence(run_dir)
    plan = AnalysisPlan.model_validate_json(evidence.verify_input("analysis_plan.json", "analysis_plan"))
    literature = LiteratureBundle.model_validate_json(evidence.verify_input(
        "preplan_literature_bundle.json", "preplan_literature_bundle",
    ))
    records = json.loads((run_dir / "manifest.json").read_text())["per_step_records"]
    RegisteredOutputEnvelopeConsumer().authoritative_writer_records(records, evidence_store=evidence)
    return build_manuscript_reader(
        manuscript=manuscript, evidence=evidence, plan=plan, literature=literature,
        evidence_records=evidence.current_verified_records(records),
    )


__all__ = ["prepare_registered_report_repair", "bind_registered_report_numbers", "build_registered_report_reader"]
