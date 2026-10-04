"""A Writer that failed before drafting is the manuscript's whole cause.

When the Writer raised before producing a draft, the binding stage still
audited the empty manuscript for the host's source definitions and reported
``writer_source_method_facts_missing`` beside the Writer's own failure, and the
manuscript state named the generic ``writer_produced_no_bindable_prose``.  The
missing definitions were a consequence, not a second defect.  The audit now
runs only on a Writer draft, and the state names the Writer failure.  A draft
that leaves nothing bindable keeps the generic code and the audit.

Synthetic manuscripts (an empty one, and one with headings but no prose) and
an empty STRICT evidence store; the definitions audit is replaced by one that
always reports missing definitions, so the test shows whether the stage
consults it.
"""

from __future__ import annotations

from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.manuscript_state import read_manuscript_state
from easyicu.research_agent.schema import ValidationFinding

MISSING = "writer_source_method_facts_missing"


def _bind(tmp_path, monkeypatch, *, scaffold: str, writer_error_message: str | None):
    audited = []

    def audit(bound, **_kwargs):
        audited.append(bound)
        return ValidationFinding(
            validator="evidence_bound_writer", severity="error", message="missing",
            detail={"reason_code": MISSING, "source_fields": ["design"]},
        )

    monkeypatch.setattr(write_phase, "audit_bound_source_method_facts", audit)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    findings = []
    binding = write_phase._bind_and_review_manuscript(
        SimpleNamespace(_evidence_enforcement_mode=EvidenceEnforcementMode.STRICT),
        critic=None, evidence=EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT),
        findings=findings, literature=None, per_step_records=[], current_evidence_names=[],
        scaffold=scaffold, writer_error_message=writer_error_message, writer_probe_mode=False,
        writer_probe_failed_steps=(), run_dir=run_dir, reader_display_labels={}, manuscript_language="en",
    )
    reason_codes = {str((item.detail or {}).get("reason_code")) for item in findings if item.detail}
    return read_manuscript_state(binding.bound), bool(audited), reason_codes


def test_a_writer_that_failed_before_drafting_is_the_cause(tmp_path, monkeypatch):
    state, audited, reason_codes = _bind(
        tmp_path, monkeypatch, scaffold="", writer_error_message="TimeoutError: provider unreachable",
    )

    assert state.reason_code == "writer_failed_before_draft"
    assert not audited
    assert MISSING not in reason_codes


def test_a_draft_with_nothing_bindable_keeps_its_code_and_the_audit(tmp_path, monkeypatch):
    state, audited, reason_codes = _bind(
        tmp_path, monkeypatch, scaffold="## Results\n\n### Primary association\n", writer_error_message=None,
    )

    assert state.reason_code == "writer_produced_no_bindable_prose"
    assert audited
    assert MISSING in reason_codes
