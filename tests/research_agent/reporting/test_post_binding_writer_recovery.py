"""Final binding failures feed only affected sections back through the gates."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore, sha256_of_file
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CritiqueReport, ValidationFinding

from .test_manuscript_quality import _valid_manuscript


@pytest.mark.parametrize("still_broken", [False, True])
def test_final_quality_failure_repairs_once_and_revalidates_without_analysis(
    tmp_path, monkeypatch, still_broken,
):
    good = _valid_manuscript()
    bad = good.replace("## Results", "## Missing results", 1)
    drafts, bindings, publications = [], [], []
    evidence = EvidenceStore(tmp_path)
    plan = AnalysisPlan(research_question="ICU cohort description", steps=[])
    for name in ("_activate_publication_figure", "_activate_publication_inputs", "_write_reproducibility_artifacts"):
        monkeypatch.setattr(write_phase, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(write_phase, "execution_gate_status", lambda **kwargs: {"execution_complete": True})

    def draft(*args, **kwargs):
        drafts.append(kwargs.get("section_repair"))
        return write_phase._DraftStageResult((), (), None, None, good)

    def bind(*args, **kwargs):
        bindings.append(kwargs["scaffold"])
        text = bad if len(bindings) == 1 or still_broken else good
        return write_phase._BindingStageResult(
            text, tmp_path / "bound.md",
            CritiqueReport(status="blocked" if text == bad else "pass", reviewer="test"),
        )

    monkeypatch.setattr(write_phase, "_draft_manuscript", draft)
    monkeypatch.setattr(write_phase, "_bind_and_review_manuscript", bind)
    monkeypatch.setattr(write_phase, "_publish_and_audit_manuscript", lambda *args, **kwargs: publications.append(kwargs["bound"]))
    result = write_phase.run_write_phase(
        SimpleNamespace(_user_extension_activation=SimpleNamespace(receipt={"activation_sha256": "a" * 64, "skills": [], "mcp_servers": []})),
        plan_result=SimpleNamespace(
            context=None, agent_context=None, evidence=evidence, findings=[],
            role_resolver=lambda _: object(), prompt_version="test", resume_state=None,
            repro_envelope=None,
        ),
        execute_result=SimpleNamespace(plan=plan, runtime_state=object(), per_step_records=[]),
        run_dir=tmp_path, run_id="run", stop_after_analysis=False,
        manuscript_title=None, manuscript_authors=None, run_language="en",
        emit_progress=lambda *args, **kwargs: None,
    )
    assert len(drafts) == len(bindings) == 2
    assert drafts[0] is None
    assert drafts[1][0] == good  # Repair source remains raw, never rendered citations.
    assert "results" in drafts[1][1]
    assert publications == [bad if still_broken else good]
    assert result.manuscript_critique.status == ("blocked" if still_broken else "pass")


def test_repaired_quality_artifacts_preserve_old_versions_and_bind_new_source(tmp_path):
    store = EvidenceStore(tmp_path)
    first = _valid_manuscript().replace("## Results", "## Missing results", 1)
    findings = []
    write_phase._persist_manuscript_quality_artifacts(
        bound=first, bound_evidence_id="bound_old", run_dir=tmp_path,
        evidence=store, findings=findings,
    )
    old = {r.evidence_id: (r.relative_path, r.sha256) for r in store.records()}
    errors = write_phase._persist_manuscript_quality_artifacts(
        bound=_valid_manuscript(), bound_evidence_id="bound_new", run_dir=tmp_path,
        evidence=store, findings=findings,
    )
    assert not errors
    for path, digest in old.values():
        assert sha256_of_file(Path(tmp_path) / path) == digest
    latest = [r for r in store.records() if r.metadata.get("source_evidence_id") == "bound_new"]
    assert len(latest) == 1
    assert latest[0].evidence_id not in old


def test_drafting_reviewer_sees_current_errors_and_preserves_recovered_history(tmp_path):
    import json

    evidence = EvidenceStore(tmp_path)
    plan = AnalysisPlan(research_question="Describe the cohort", steps=[
        AnalysisStep(step_id="model", intent="Summarise the cohort", expected_outputs=["table:result"]),
    ])
    findings = [
        ValidationFinding(validator="recovered_runtime", severity="error", message="Recovered old runtime error",
                          detail={"step_id": "model", "attempt_id": "old"}),
        ValidationFinding(validator="current_runtime", severity="error", message="Unresolved current runtime error",
                          detail={"step_id": "model", "attempt_id": "current"}),
    ]
    records = [
        {"step_id": "model", "attempt_id": "old", "status": "error"},
        {"step_id": "model", "attempt_id": "current", "status": "ok"},
    ]
    write_phase._run_drafting_reviewer_round(
        SimpleNamespace(), plan=plan, per_step_records=records,
        evidence=evidence, findings=findings, bound=_valid_manuscript(),
        repro_envelope=None, run_dir=tmp_path,
    )
    review = json.dumps(json.loads((tmp_path / "reviewer_report.json").read_text()))
    assert "current_runtime" in review
    assert "recovered_runtime" not in review
    assert len([f for f in findings if f.validator.endswith("_runtime")]) == 2
