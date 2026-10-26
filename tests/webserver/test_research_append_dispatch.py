"""Deterministic additions use governed durable jobs and recorded approvals."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support import pytest_state  # noqa: F401
from tests.support.result_append import completed_source
from easyicu.webserver import research_append_run as owner
from easyicu.webserver.research_append_source import read_json, seal_completed_source


class Dispatcher:
    def __init__(self, path, callback, **kwargs):
        self.submitted = []

    def submit(self, identity):
        self.submitted.append(identity)

    def status(self, identity):
        return None

    def launch(self):
        pass

    def close(self, timeout=4):
        return True


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from easyicu.webserver import durable_research, jobs, study_contexts, settings
    from easyicu.webserver.routes import agent

    wrapper, run, _, _, baseline = completed_source(tmp_path)
    study = {"id": "study", "revision": 0}
    source = seal_completed_source(wrapper=wrapper, run_dir=run, study=study,
        baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True})
    before = {str(p.relative_to(wrapper)): p.read_bytes() for p in wrapper.rglob("*") if p.is_file()}
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(study_contexts, "get_context", lambda _: study)
    monkeypatch.setattr(agent, "research_pipeline_workspace", lambda: SimpleNamespace(project_root=lambda _: tmp_path))
    monkeypatch.setattr(agent, "submit_job", lambda *a, **k: pytest.fail("durable dispatch bypassed"))
    def handoff(*a, **kw):
        study.update(active_job_id=kw["active_job_id"], revision=study["revision"] + 1)
        return study
    def cleanup(*a, **kw):
        study.update(active_job_id=None, revision=study["revision"] + 1)
        return {"context": study}
    monkeypatch.setattr(study_contexts, "handoff_context", handoff)
    monkeypatch.setattr(study_contexts, "clear_active_job_if", cleanup)
    manager = jobs.JobManager()
    manager.enable_persistence(tmp_path / "jobs.sqlite3")
    runtime = durable_research.DurableResearchService(tmp_path / "dispatch", manager=manager, dispatcher_factory=Dispatcher)
    monkeypatch.setattr(durable_research, "current_service", lambda: runtime)
    yield SimpleNamespace(source=source, study=study, runtime=runtime, manager=manager, before=before, root=tmp_path)
    runtime.dispatcher.close()


@pytest.mark.parametrize("kind", ["sensitivity", "exposure", "subgroup", "article"])
def test_governed_approval_dispatches_only_the_addition(prepared, kind):
    from easyicu.webserver.routes.agent import submit_agent_run_review

    p = prepared
    child = owner.propose_append_run(source=p.source, kind=kind, wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    answer = submit_agent_run_review(dict(run_id=child.checkpoint.run_id, study_context_id="study",
                                          decision="approved", external_llm_opt_in=True))
    assert p.runtime.dispatcher.submitted == [answer["job_id"]]
    assert owner.load_append_run(child.wrapper).checkpoint.state == "approved_pending_execution"
    assert not (child.run_dir / "steps").exists()
    assert p.runtime.execute(answer["job_id"])["status"] == "done"
    finished = owner.load_append_run(child.wrapper)
    assert finished.checkpoint.state == "completed", p.manager.get(answer["job_id"]).snapshot()
    assert {path.name for path in (child.run_dir / "steps").iterdir()} == {s.step_id for s in child.plan.steps}
    assert {str(path.relative_to(p.source.wrapper)): path.read_bytes()
            for path in p.source.wrapper.rglob("*") if path.is_file()} == p.before
    assert p.source.plan.design_selection.execution_tier == "first_result"
    assert child.plan.design_selection.execution_tier == ("article" if kind == "article" else "first_result")
    result = read_json(child.wrapper / "result_tables.json")
    assert "adjusted_association" in {t["produced_by_step"] for t in result["tables"]}
    assert p.runtime.execute(answer["job_id"])["status"] == "blocked"


@pytest.mark.parametrize("recorded", [False, True])
def test_restart_never_invents_an_append_approval(prepared, monkeypatch, recorded):
    from easyicu.webserver import research_auto_resume

    p = prepared
    child = owner.propose_append_run(source=p.source, kind="subgroup", wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    descriptor = research_auto_resume.prepare_descriptor(study=p.study, project_root=p.root,
        budget_mode="full_reviewed", credential_source="scientific_provider", external_llm_opt_in=True,
        execution_resume_source_run_id=child.checkpoint.run_id)
    if recorded:
        write = owner.write_checkpoint
        def crash(path, checkpoint):
            if checkpoint.state == "executing":
                raise OSError("synthetic restart")
            write(path, checkpoint)
        monkeypatch.setattr(owner, "write_checkpoint", crash)
        with pytest.raises(OSError, match="synthetic restart"):
            owner.approve_append_run(child.wrapper, decision="approved", reviewer="server-researcher")
        monkeypatch.setattr(owner, "write_checkpoint", write)
    before = owner.load_append_run(child.wrapper).checkpoint.approved_decisions
    job = SimpleNamespace(id="restarted", cancel_requested=False, emit=lambda _: None)
    if recorded:
        result = research_auto_resume.recover_runner(descriptor, job)(job)
        assert result["gate"]["status"] == "analysis_only"
        assert owner.load_append_run(child.wrapper).checkpoint.approved_decisions == before
    else:
        with pytest.raises(ValueError, match="recorded human approval"):
            research_auto_resume.recover_runner(descriptor, job)(job)
        assert not (child.run_dir / "steps").exists()


def test_recovery_flag_cannot_approve_pending_append(prepared):
    from easyicu.webserver.agent_pipeline_runs import resume_research_pipeline

    p = prepared
    child = owner.propose_append_run(source=p.source, kind="subgroup", wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    with pytest.raises(ValueError, match="recorded human approval"):
        resume_research_pipeline(run_id=child.checkpoint.run_id, study_context_id="study", decision="approved",
            reviewer="server-researcher", note="", job=SimpleNamespace(id="restart", cancel_requested=False, emit=lambda _: None),
            current_study_context=p.study, require_recorded_decision=True)
    assert owner.load_append_run(Path(child.wrapper)).checkpoint.state == "pending"


@pytest.mark.parametrize("kind", ["sensitivity", "exposure", "subgroup", "article"])
def test_accepted_append_approval_survives_restart_before_dispatch(prepared, kind):
    from easyicu.webserver.durable_research import DurableResearchService
    from easyicu.webserver.routes.agent import submit_agent_run_review

    p = prepared
    child = owner.propose_append_run(source=p.source, kind=kind, wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    answer = submit_agent_run_review(dict(run_id=child.checkpoint.run_id, study_context_id="study",
                                          decision="approved", external_llm_opt_in=True, note="Run this addition."))
    before = owner.load_append_run(child.wrapper).checkpoint
    assert not (child.run_dir / "steps").exists()
    # Reconstruct the dispatcher from its durable store without the live runner
    # closure. An accepted human approval must survive this queue boundary.
    restarted = DurableResearchService(p.root / "dispatch", manager=p.manager, dispatcher_factory=Dispatcher)
    try:
        receipt = restarted.execute(answer["job_id"])
    finally:
        restarted.dispatcher.close()
    assert receipt["status"] == "done", p.manager.get(answer["job_id"]).snapshot()
    after = owner.load_append_run(child.wrapper).checkpoint
    assert after.state == "completed"
    assert before.state == "approved_pending_execution"
    assert after.approved_decisions == before.approved_decisions
    assert after.approved_decision_records == before.approved_decision_records
    assert after.approved_decisions[0]["note"] == "Run this addition."
    assert {path.name for path in (child.run_dir / "steps").iterdir()} == {s.step_id for s in child.plan.steps}
    assert {str(path.relative_to(p.source.wrapper)): path.read_bytes()
            for path in p.source.wrapper.rglob("*") if path.is_file()} == p.before


def test_live_dispatch_never_invents_a_missing_queued_approval(prepared):
    from easyicu.webserver.routes.agent import submit_agent_run_review

    p = prepared
    child = owner.propose_append_run(source=p.source, kind="subgroup", wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    checkpoint_path = child.run_dir / "human_review_checkpoint.json"
    pending_bytes = checkpoint_path.read_bytes()
    answer = submit_agent_run_review(dict(run_id=child.checkpoint.run_id, study_context_id="study",
                                          decision="approved", external_llm_opt_in=True))
    checkpoint_path.write_bytes(pending_bytes)
    assert p.runtime.execute(answer["job_id"])["status"] == "failed"
    assert not (child.run_dir / "steps").exists()
    assert not owner.load_append_run(child.wrapper).checkpoint.approved_decisions


def test_failed_study_handoff_does_not_record_an_append_approval(prepared, monkeypatch):
    from fastapi import HTTPException
    from easyicu.webserver import study_contexts
    from easyicu.webserver.routes.agent import submit_agent_run_review

    p = prepared
    child = owner.propose_append_run(source=p.source, kind="subgroup", wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    def refuse(*args, **kwargs):
        raise RuntimeError("synthetic study handoff refusal")
    monkeypatch.setattr(study_contexts, "handoff_context", refuse)
    with pytest.raises(HTTPException):
        submit_agent_run_review(dict(run_id=child.checkpoint.run_id, study_context_id="study",
                                     decision="approved", external_llm_opt_in=True))
    assert p.runtime.dispatcher.submitted == []
    assert not owner.load_append_run(child.wrapper).checkpoint.approved_decisions
    assert not (child.run_dir / "steps").exists()


def test_governed_rejection_does_not_require_a_recorded_approval(prepared, monkeypatch):
    from easyicu.webserver.routes import agent

    p = prepared
    child = owner.propose_append_run(source=p.source, kind="subgroup", wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    queued = []
    job = SimpleNamespace(id="rejected-append", kind="agent-run", status="running", cancel_requested=False, emit=lambda _: None)
    monkeypatch.setattr(agent, "submit_job", lambda kind, runner: queued.append(runner) or job)
    agent.submit_agent_run_review(dict(run_id=child.checkpoint.run_id, study_context_id="study",
                                      decision="rejected", external_llm_opt_in=True))
    assert queued[0](job)["gate"]["reason"] == "human_plan_review_rejected"
    assert owner.load_append_run(child.wrapper).checkpoint.state == "rejected"
    assert not (child.run_dir / "steps").exists()


def test_staged_approval_must_belong_to_the_submitting_study(prepared):
    from easyicu.webserver.research_append_dispatch import stage_append_approval

    p = prepared
    child = owner.propose_append_run(source=p.source, kind="subgroup", wrapper=p.source.wrapper.parent / "run_child",
                                     study=p.study, registered_source_changed=False)
    with pytest.raises(ValueError, match="another study"):
        stage_append_approval(run_id=child.checkpoint.run_id, study={"id": "different-study"},
                              reviewer="server-researcher", note="")
    assert not owner.load_append_run(child.wrapper).checkpoint.approved_decisions
