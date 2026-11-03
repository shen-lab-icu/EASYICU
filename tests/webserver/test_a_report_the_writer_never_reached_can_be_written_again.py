"""A report the Writer never reached can be written again without rerunning.

11-03 eICU walk: the outbound proxy returned 503 while the Writer ran, so a
validated analysis finished with ``writer_provider_transport_unavailable``
and no draft.  The page said "try again later" but offered nothing: the
report-only rewrite accepted only failed-closed executions, and a chat
request cannot grant a provider run.  The workflow now offers the rewrite for
exactly that stop, and the resume owner accepts that run for a report-only
rewrite alone; an analysis retry still refuses it.  Synthetic rows only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.reporting import registered_report_inputs
from easyicu.webserver import agent_pipeline_runs, agent_runs
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.agent_pipeline_runs import ResearchPipelineRunError, writer_never_reached
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

STUDY = {
    "id": "study-writer", "revision": 3,
    "question": "Is lactate associated with in-hospital mortality?",
    "data_source": {"path": "/private/prepared/source", "database": "eicu_demo"},
    "cohort": {"label": "ICU patients"}, "modules": [], "outcome": "", "analysis_goal": "",
    "time_window": {}, "export_format": "", "confirmations": {},
}
UNREACHED = {"status": "not_passed", "reason_code": "writer_provider_transport_unavailable"}


def _run(**changes):
    run = {
        "run_id": "run_writer", "run_type": "full", "engine": "easyicu.research_agent.pipeline",
        "gate_status": "analysis_only",
        "gate_reason": "research_agent_pipeline_complete_human_interpretation_required",
        "gate_checks": {"execution_complete": True, "analysis_validated": True,
                        "evidence_complete": True, "numeric_verified": False, "manuscript_ready": False},
        "run_status": "completed",
        "analysis_outcome": {"status": "completed", "reason_code": None},
        "manuscript_outcome": UNREACHED,
        "scientific_configuration_sha256": study_context_owner.scientific_configuration_sha256(STUDY),
        "artifact_names": ["agent_plan.json", "evidence_ledger.json", "result_tables.json",
                           "figure_gallery.json", "source_run_manifest.json"],
    }
    return {**run, **changes}


def _snapshot(run):
    return build_research_workflow_snapshot(
        study=STUDY, active_export_present=False, active_job=None, latest_run=run,
    )


def test_the_workflow_offers_a_rewrite_only_when_the_writer_never_reached_the_service():
    assert _snapshot(_run()).report_rewrite_available is True
    assert _snapshot(_run()).analysis_validation_retry_available is False
    other = {"status": "not_passed", "reason_code": "manuscript_checks_not_passed"}
    assert _snapshot(_run(manuscript_outcome=other)).report_rewrite_available is False
    # A changed study setup has no sealed analysis to write up.
    assert _snapshot(_run(scientific_configuration_sha256="0" * 64)).report_rewrite_available is False
    assert writer_never_reached(_run()) is True
    assert writer_never_reached(_run(run_status="blocked")) is False


@pytest.fixture
def history(tmp_path, monkeypatch):
    wrapper = tmp_path / "study-writer" / "run_wrapper"
    (wrapper / "pipeline").mkdir(parents=True)
    rows = [{**_run(), "project_dir": str(wrapper)}]
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_kwargs: {"runs": rows})
    return tmp_path, rows


def _resolve(root: Path, *, report_only: bool):
    return agent_pipeline_runs._resolve_execution_resume_wrapper(
        study=STUDY, project_root=str(root), source_run_id="run_writer", report_only=report_only,
    )


def test_only_a_report_only_rewrite_may_resume_that_run(history):
    root, rows = history
    with pytest.raises(ResearchPipelineRunError) as analysis_retry:
        _resolve(root, report_only=False)
    assert analysis_retry.value.code == "research_pipeline_execution_retry_source_not_failed_execution"
    # The report-only rewrite passes the source gate and then needs the same
    # complete approved checkpoint as every other resume (none here).
    with pytest.raises(ResearchPipelineRunError) as rewrite:
        _resolve(root, report_only=True)
    assert rewrite.value.code == "research_pipeline_execution_retry_checkpoint_missing"
    rows[0]["manuscript_outcome"] = {"status": "not_passed", "reason_code": "manuscript_checks_not_passed"}
    with pytest.raises(ResearchPipelineRunError) as other:
        _resolve(root, report_only=True)
    assert other.value.code == "research_pipeline_execution_retry_source_not_failed_execution"


@pytest.mark.parametrize("reason,kept", [
    ("writer_provider_transport_unavailable", True),
    ("manuscript_checks_not_passed", False),
])
def test_a_finish_the_writer_never_reached_keeps_its_approved_configuration(tmp_path, monkeypatch, reason, kept):
    from easyicu.webserver import research_append_source

    removed, retained = [], []
    monkeypatch.setattr(agent_pipeline_runs, "_remove_local_recovery", removed.append)
    monkeypatch.setattr(research_append_source, "retain_completed_append_source",
                        lambda **kwargs: retained.append(kwargs["wrapper_dir"]))
    gate = {"reason": "research_agent_pipeline_complete_human_interpretation_required",
            "manuscript_outcome": {"status": "not_passed", "reason_code": reason}}
    agent_pipeline_runs._cleanup_recovery_after_projection(wrapper_dir=tmp_path, projection={"gate": gate})
    # The completed results stay appendable either way.
    assert retained == [tmp_path]
    assert removed == ([] if kept else [tmp_path])


def test_no_rewrite_is_offered_without_the_kept_configuration(history, monkeypatch):
    root, _ = history
    ready = lambda: agent_pipeline_runs.report_rewrite_ready(  # noqa: E731
        study=STUDY, project_root=str(root), source_run_id="run_writer")
    # A run that finished before this stop kept its configuration has none.
    assert ready() is False
    monkeypatch.setattr(agent_pipeline_runs, "_resolve_execution_resume_wrapper", lambda **_kwargs: type(
        "Target", (), {"wrapper_dir": root, "pipeline_run_id": "run_pipeline"})())
    monkeypatch.setattr(agent_pipeline_runs, "load_recovery_seed", lambda _wrapper: object())
    checked = []
    monkeypatch.setattr(registered_report_inputs, "registered_report_inputs_intact",
                        lambda run_dir: checked.append(run_dir) or True)
    assert ready() is True
    assert checked == [root / "pipeline" / "run_pipeline"]
    # Sealed inputs that no longer match would refuse the rewrite at once.
    monkeypatch.setattr(registered_report_inputs, "registered_report_inputs_intact", lambda _run_dir: False)
    assert ready() is False


def _sealed_draft(run_dir: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.write_phase import _store_raw_scaffold

    evidence = EvidenceStore(run_dir)
    store = lambda text: _store_raw_scaffold(  # noqa: E731
        run_dir, text, evidence=evidence, prompt_version="synthetic")
    return evidence, store


def test_a_writer_that_never_reached_its_service_keeps_the_sealed_draft(tmp_path):
    evidence, store = _sealed_draft(tmp_path)
    store("## Results\n\nSynthetic draft {evidence:result}.\n")
    sealed = (tmp_path / "manuscript_scaffold.md").read_bytes()

    store("")

    # The live copy still equals its seal, so the rewrite can start.
    assert (tmp_path / "manuscript_scaffold.md").read_bytes() == sealed
    reader = registered_report_inputs.ReadOnlyReportEvidence(tmp_path)
    assert reader.verify_input("manuscript_scaffold.md", "manuscript_scaffold_raw") == sealed
    assert registered_report_inputs.registered_report_inputs_intact(tmp_path) is False  # other inputs absent here


def test_a_new_draft_still_replaces_and_reseals_the_old_one(tmp_path):
    evidence, store = _sealed_draft(tmp_path)
    store("first draft\n")
    store("second draft\n")
    reader = registered_report_inputs.ReadOnlyReportEvidence(tmp_path)
    assert reader.verify_input("manuscript_scaffold.md", "manuscript_scaffold_raw") == b"second draft\n"


def test_a_first_empty_draft_is_still_recorded(tmp_path):
    evidence, store = _sealed_draft(tmp_path)
    store("")
    assert (tmp_path / "manuscript_scaffold.md").read_text() == ""
    assert evidence.get("manuscript_scaffold_raw") is not None
