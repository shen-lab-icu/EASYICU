"""Diagnostic writer-probe flag tests.

The writer probe is an explicit engineering bypass for Phase-1 pilot
triage. It must never change the default fail-closed manuscript gate.
"""

from __future__ import annotations

import json
from pathlib import Path


def _patch_failed_execute(monkeypatch, ra) -> None:
    from easyicu.research_agent.contracts.runtime import _ExecutePhaseResult
    from easyicu.research_agent.schema import AgentRuntimeState

    def _fake_execute(
        self,
        *,
        plan_result,
        cohort_path,
        run_dir,
        run_id,
        skill_obj,
        notes,
        emit_progress,
        resume_from_step_id=None,
        stop_after_step_id=None,
    ):
        del self, cohort_path, run_dir, skill_obj, notes, emit_progress
        del resume_from_step_id, stop_after_step_id
        first_step = plan_result.plan.steps[0].step_id
        return _ExecutePhaseResult(
            plan=plan_result.plan,
            per_step_records=[
                {"step_id": first_step, "status": "execution_failed"}
            ],
            probe_summary={},
            runtime_state=AgentRuntimeState(run_id=run_id),
            flush_partial_manifest=lambda extra=None: None,
        )

    monkeypatch.setattr(ra.ResearchAgentPipeline, "_run_execute_phase", _fake_execute)


def test_writer_probe_false_default_fail_closed(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    _patch_failed_execute(monkeypatch, ra)
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())

    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="writer_probe_default",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)
    normal = run_dir / "manuscript_scaffold_bound.md"
    probe = run_dir / "manuscript_scaffold_writer_probe.md"
    assert normal.exists()
    assert not probe.exists()
    assert "Manuscript scaffold not generated" in normal.read_text(encoding="utf-8")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["writer_probe_mode"] is False
    assert manifest["manuscript_path"] == "manuscript_scaffold_bound.md"
    assert any(
        f["validator"] == "manuscript_gate" and f["severity"] == "error"
        for f in manifest["findings"]
    )


def test_writer_probe_true_forces_writer_phase(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    _patch_failed_execute(monkeypatch, ra)
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())

    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="writer_probe_forced",
        database="synthetic",
        target_outcome="death",
        force_writer_probe=True,
    )

    run_dir = Path(result.workdir)
    normal = run_dir / "manuscript_scaffold_bound.md"
    probe = run_dir / "manuscript_scaffold_writer_probe.md"
    assert not normal.exists()
    assert probe.exists()
    probe_text = probe.read_text(encoding="utf-8")
    assert "DIAGNOSTIC PROBE ONLY" in probe_text
    assert "MUST NOT be cited" in probe_text
    assert "execution_failed" in probe_text

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["writer_probe_mode"] is True
    assert manifest["manuscript_path"] == "manuscript_scaffold_writer_probe.md"
    assert manifest["writer_probe_failed_steps"]
    assert any(
        f["validator"] == "manuscript_gate" and f["severity"] == "warning"
        for f in manifest["findings"]
    )

    run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["writer_probe_mode"] is True
    assert run_status["gates"]["writer_probe_mode"] is True
    assert run_status["gates"]["manuscript_generated"] is False
