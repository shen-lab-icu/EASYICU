from __future__ import annotations

import json

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceStore,
    EvidenceEnforcementError,
)
from easyicu.research_agent.reporting.registered_report_inputs import (
    ReadOnlyReportEvidence,
)
from easyicu.research_agent.reporting.writer_only_migration import (
    WriterOnlyMigrationError,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode


def _bind_numbers(root, text):
    # Exercise the unchanged value binder through the new read-only facade;
    # envelope admission has its own owner/real-run replay tests.
    bound, bindings, _ = bind_numeric_values(
        text,
        evidence=ReadOnlyReportEvidence(root),
        enforcement_mode=EvidenceEnforcementMode.STRICT,
        per_step_records=json.loads((root / "manifest.json").read_text())[
            "per_step_records"
        ],
    )
    return bound, len(bindings)


def _source(tmp_path):
    root = tmp_path / "run"
    store = EvidenceStore(root)
    rows = []
    for step_id, number in (("cohort", 120), ("other", 240)):
        path = tmp_path / f"{step_id}.json"
        summary = {"n_total": number}
        path.write_text(json.dumps(summary))
        evidence = store.register_file(
            kind="statistic",
            description="Aggregate counts",
            source_path=path,
            evidence_id=step_id + "_summary",
            produced_by_step=step_id,
        )
        store.register_step_summary_numerics(
            step_id=step_id, evidence_id=evidence.evidence_id, summary=summary
        )
        rows.append(
            {
                "step_id": step_id,
                "status": "ok",
                "step_summary": summary,
                "step_summary_evidence_id": evidence.evidence_id,
                "evidence_ids": [evidence.evidence_id],
            }
        )
    (root / "manifest.json").write_text(json.dumps({"per_step_records": rows}))
    return root, store


def test_reader_uses_snapshot_without_constructing_or_repairing_store(
    tmp_path, monkeypatch
):
    root, store = _source(tmp_path)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        EvidenceStore, "__init__", lambda *a, **kw: pytest.fail("mutable store opened")
    )
    reader = ReadOnlyReportEvidence(root)
    assert (
        len(
            reader.authoritative_numeric_claims(
                json.loads((root / "manifest.json").read_text())["per_step_records"]
            )
        )
        == 2
    )
    assert reader.get("cohort_summary").sha256 == store.get("cohort_summary").sha256
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_named_report_input_must_equal_its_sealed_registered_copy(tmp_path):
    root, store = _source(tmp_path)
    path = root / "input.json"
    path.write_text('{"n_total":120}')
    store.register_file(
        kind="log", description="Input", source_path=path, evidence_id="input"
    )
    reader = ReadOnlyReportEvidence(root)
    assert reader.verify_input("input.json", "input") == path.read_bytes()
    path.write_text('{"n_total":121}')
    with pytest.raises(WriterOnlyMigrationError, match="REGISTERED_INPUT_CHANGED"):
        reader.verify_input("input.json", "input")


def test_report_repair_binds_values_to_exact_step_not_only_valid_citations(tmp_path):
    root, _ = _source(tmp_path)
    text = "The cohort included 120 ICU stays {evidence:cohort_summary}."
    bound, count = _bind_numbers(root, text)
    assert count == 1 and "[^claim_1]" in bound
    assert "step=cohort" in bound
    with pytest.raises(EvidenceEnforcementError):
        _bind_numbers(root, text.replace("120", "9999"))
    with pytest.raises(EvidenceEnforcementError):
        _bind_numbers(root, text.replace("120", "240"))


def test_report_numbers_cannot_use_tampered_evidence(tmp_path):
    root, store = _source(tmp_path)
    (root / store.get("cohort_summary").relative_path).write_text('{"n_total":9999}')
    with pytest.raises(EvidenceEnforcementError):
        _bind_numbers(root, "The cohort included 120 stays {evidence:cohort_summary}.")
