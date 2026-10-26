"""Readiness producers cite only records the reader can resolve uniquely."""

import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.webserver.research_evidence_preview import project_review_evidence_refs
from easyicu.webserver.scientific_readiness_projection import build_scientific_readiness_projection


@pytest.mark.parametrize("state", ["current", "unregistered", "ambiguous", "stale", "missing"])
def test_every_finding_reference_resolves_to_one_registered_record(tmp_path, state):
    store = EvidenceStore(tmp_path)
    report = tmp_path / "reviewer_report.json"
    report.write_text(json.dumps({"summary": {"aggregated_recommendation": "major_revision", "counts": {"major": 1}}}))
    if state != "unregistered":
        record = store.register_file(source_path=report, evidence_id="review", kind="log", description="Synthetic review", producer="pipeline", generation_mode="system")
        if state == "ambiguous":
            store.register_file(source_path=report, evidence_id="second_review", kind="log", description="Other review", producer="pipeline", generation_mode="system")
        elif state == "stale":
            (tmp_path / record.relative_path).write_text("{}")
        elif state == "missing":
            (tmp_path / record.relative_path).unlink()
    projection = build_scientific_readiness_projection(
        run_id="synthetic_run", run_dir=tmp_path,
        axes={"analysis_validated": True}, literature_evidence={}, study={},
    )
    refs = [ref for finding in projection.findings for ref in finding.evidence_refs]
    resolved = project_review_evidence_refs(tmp_path, refs)
    assert all(sum(row["reference"] == ref for row in resolved) == 1 for ref in refs)
    idea = next(item for item in projection.findings if item.code == "IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED")
    assert idea.evidence_refs == []
    review = next(item for item in projection.findings if item.code == "SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN")
    if state == "current":
        assert review.evidence_refs == ["reviewer_report.json"]
        assert resolved[0]["evidence_id"] == "review"
        assert resolved[0]["sha256"] == record.sha256
    else:
        assert review.evidence_refs == []
    assert projection.status == "analysis_only"


def test_maturity_references_remain_resolvable_after_reader_budgets(tmp_path):
    from easyicu.webserver.pi_copilot.projections import project_run_outcome
    from easyicu.webserver.pi_copilot.workflow import _scientific_review_references

    store = EvidenceStore(tmp_path)
    findings = []
    for index in range(30):
        name = f"synthetic_audit_{index}.json"
        path = tmp_path / name
        path.write_text('{"checks":{"value":true}}')
        store.register_file(source_path=path, evidence_id=f"audit_{index}", kind="log", description="Synthetic checks", producer="pipeline", generation_mode="system")
        findings.append({
            "code": "ROBUSTNESS_AXES_TOO_NARROW", "severity": "major", "dimension": "analysis",
            "message": "Synthetic check", "remediation": "Inspect the receipt",
            "evidence_refs": [name, name + ".checks.value", "reporting/owner.py", "StudyContext.idea_handoff"],
        })
    (tmp_path / "scientific_maturity_audit.json").write_text(json.dumps({"findings": findings}))
    projection = build_scientific_readiness_projection(
        run_id="synthetic_run", run_dir=tmp_path,
        axes={"analysis_validated": True}, literature_evidence={}, study={},
    )
    review = {"ok": True, "run_id": "synthetic_run", "artifact_payloads": {
        "scientific_readiness.json": projection.model_dump(mode="json"),
    }}
    resolved = project_review_evidence_refs(tmp_path, _scientific_review_references(review))
    outcome = project_run_outcome(review, review_evidence_refs=resolved)
    refs = [ref for finding in projection.findings for ref in finding.evidence_refs]
    assert refs and len(outcome["review_evidence_refs"]) <= 40
    assert all(sum(row["reference"] == ref for row in outcome["review_evidence_refs"]) == 1 for ref in refs)
    assert any(row.get("pointer") == "/checks/value" for row in outcome["review_evidence_refs"])


def test_a_projection_without_a_run_never_cites_working_files():
    projection = build_scientific_readiness_projection(
        run_id="synthetic_run", run_dir=None, axes={}, literature_evidence={}, study={},
    )
    assert projection.findings
    assert all(not row.evidence_refs for row in [*projection.domains, *projection.findings])
