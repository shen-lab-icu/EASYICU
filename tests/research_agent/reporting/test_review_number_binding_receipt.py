"""The review retains the existing binder's result, including unknown numbers."""

import hashlib


def test_reader_retains_binder_receipt_and_marks_legacy_coverage_unknown(ra, tmp_path):
    from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    evidence = ra.EvidenceStore(root=tmp_path, enforcement_mode="soft")
    manuscript, bindings, untraced = bind_numeric_values("The synthetic value was 987.65.", evidence=evidence)
    assert untraced == ["987.65"]
    reader = build_manuscript_reader(manuscript=manuscript, evidence=evidence,
                                     binding_map=bindings, untraced_numerics=untraced)
    assert reader["number_binding"] == {"status": "recorded", "untraced": ["987.65"]}
    assert reader["manuscript_sha256"] == hashlib.sha256(manuscript.encode()).hexdigest()
    legacy = build_manuscript_reader(manuscript=manuscript, evidence=evidence)
    assert legacy["number_binding"] == {"status": "unavailable", "untraced": []}


def test_write_phase_persists_the_same_untraced_receipt(ra, tmp_path):
    import json

    from easyicu.research_agent.reporting.write_phase import _persist_manuscript_provenance_artifact

    evidence = ra.EvidenceStore(root=tmp_path, enforcement_mode="soft")
    findings = []
    _persist_manuscript_provenance_artifact(bound="Synthetic draft.", numeric_binding_map={}, run_dir=tmp_path,
                                          evidence=evidence, findings=findings, untraced_numerics=["987.65"])
    assert findings == []
    assert json.loads((tmp_path / "manuscript_provenance.json").read_text())["number_binding"]["untraced"] == ["987.65"]


def test_binding_stage_carries_unbound_values_through_both_persistence_owners(ra, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from easyicu.research_agent.reporting import write_phase

    evidence = ra.EvidenceStore(root=tmp_path, enforcement_mode="soft")
    # Inject after the separate sentence-level filter; the real value binder
    # and every receipt/persistence owner still run.
    monkeypatch.setattr(evidence, "enforce_evidence_bound_scaffold", lambda text, **kw: (text, []))
    monkeypatch.setattr(write_phase, "_review_manuscript_with_fail_safe", lambda *a, **kw: (
        ra.CritiqueReport(status="pass", reviewer="synthetic"), None))
    text = "\n\n".join(f"## {section}\nA synthetic measurement was described in this draft for review. The observations remain descriptive."
                        for section in ("Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"))
    text = text.replace("## Results", "## Results\nThe synthetic value was 987.65.")
    write_phase._bind_and_review_manuscript(
        SimpleNamespace(_evidence_enforcement_mode=ra.EvidenceEnforcementMode.SOFT), critic=object(), evidence=evidence,
        findings=[], literature=None, per_step_records=[], current_evidence_names=[], scaffold=text,
        writer_error_message=None, writer_probe_mode=False, writer_probe_failed_steps=[], run_dir=tmp_path,
        reader_display_labels={}, manuscript_language="en",
    )
    assert json.loads((tmp_path / "manuscript_provenance.json").read_text())["number_binding"] == {
        "status": "recorded", "untraced": ["987.65"]}
