"""Recovery uses launch authority, not a reconstructed plan or approval."""

import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs, research_job_recovery, research_launch_resume
from easyicu.webserver import provider_job_recovery as policy
from easyicu.webserver import study_contexts
from easyicu.webserver.jobs import Job
from tests.support.grouped_resume import source as source_fixture

source = source_fixture


@pytest.mark.parametrize("approved_preparation", [False, True])
def test_planner_resume_uses_the_offer_owner_and_normal_launch_without_approval(source, monkeypatch, approved_preparation):
    wrapper = source.run.parents[1]
    if approved_preparation:
        from easyicu.webserver.agent_review_recovery import load_recovery_seed, put_recovery_seed, WebReviewRecoverySeed
        seed = load_recovery_seed(wrapper)
        payload = seed.model_dump(exclude={"seed_sha256", "schema_version"})
        payload.update(budget_mode="full_reviewed", prepared_package_binding={"sha256": "b" * 64})
        put_recovery_seed(WebReviewRecoverySeed.create(**payload))
    monkeypatch.setattr(study_contexts, "get_context", lambda _: source.study)
    offers = []
    original = research_launch_resume.development_resume_available
    def offer(**kwargs):
        offers.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(research_launch_resume, "development_resume_available", offer)
    launches = []
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: launches.append(kw) or (lambda job: {"pending_review": True}))
    monkeypatch.setattr(agent_pipeline_runs, "resume_research_pipeline", lambda **kw: pytest.fail("automatic approval"))
    state = policy.ProviderJobRecovery(Job("current", "agent-run"))
    state.wrapper_dir = wrapper
    launch = {"study_context": source.study, "project_root": str(wrapper.parent.parent),
              "budget_mode": "full_reviewed" if approved_preparation else "planner_canary",
              "plan_revision_source_run_id": "accepted-candidate", "provider_environment": {"synthetic": "frozen"}}
    resumed = research_job_recovery._continuation(state, launch)
    assert resumed.run(None) == {"pending_review": True}
    assert offers and resumed.checkpoint["kind"] == "planner"
    assert resumed.checkpoint["path"].endswith("checkpoint_002.json")
    assert launches[0]["development_resume_source_job_id"] == "prior"
    assert launches[0]["plan_revision_source_run_id"] == ""
    assert not launches[0].get("execution_resume_source_run_id")
    assert launches[0]["provider_environment"] == launch["provider_environment"]
    assert launch["plan_revision_source_run_id"] == "accepted-candidate"


def test_changed_study_or_owner_refusal_never_launches(source, monkeypatch):
    state = policy.ProviderJobRecovery(Job("current", "agent-run"))
    state.wrapper_dir = source.run.parents[1]
    launch = {"study_context": source.study}
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: pytest.fail("no checkpoint"))
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {**source.study, "question": "Changed question"})
    assert research_job_recovery._continuation(state, launch) is None
    monkeypatch.setattr(study_contexts, "get_context", lambda _: source.study)
    monkeypatch.setattr(research_launch_resume, "development_resume_available", lambda **kw: False)
    assert research_job_recovery._continuation(state, launch) is None
    assert state.continuation_refusal is not None
    assert "prepare the same reviewed candidate again" in state.continuation_refusal["message"]


def test_exhausted_outage_preserves_the_existing_planner_page_action(source, monkeypatch):
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(policy, "RECOVERY_WINDOW_SECONDS", 1)
    monkeypatch.setattr(policy.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(policy.time, "sleep", lambda s: setattr(clock, "now", clock.now + s))
    monkeypatch.setattr(study_contexts, "get_context", lambda _: source.study)
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw:
                        lambda job: pytest.fail("the outage has not recovered"))
    wrapper = source.run.parents[1]
    state = policy.ProviderJobRecovery(Job("bounded", "agent-run"))
    task = SimpleNamespace(assert_active=lambda: 3600, ledger=SimpleNamespace(snapshot=lambda: {"tasks": []}))
    state.bind(wrapper_dir=wrapper, task=task, client=object())
    launch = {"study_context": source.study}
    error = ConnectionError("synthetic outage")
    with pytest.raises(policy.ProviderRecoveryExhausted):
        state.run(lambda job: (_ for _ in ()).throw(error), lambda exc: research_job_recovery._continuation(state, launch))
    assert research_launch_resume.development_resume_available(
        wrapper_dir=wrapper, study_id=source.study["id"],
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study),
    )
    assert research_job_recovery._continuation(state, launch) is not None


def test_approved_execution_uses_only_exact_execution_launch(source, monkeypatch):
    state = policy.ProviderJobRecovery(Job("current", "agent-review"))
    state.wrapper_dir = source.run.parents[1]
    state.execution_resume_run_id = "run-approved"
    monkeypatch.setattr(study_contexts, "get_context", lambda _: source.study)
    monkeypatch.setattr(research_launch_resume, "development_resume_available", lambda **kw: pytest.fail("replan"))
    checkpoint = {"kind": "execution", "sha256": "c" * 64}
    monkeypatch.setattr(agent_pipeline_runs, "execution_resume_checkpoint", lambda **kw: checkpoint)
    calls = []
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: calls.append(kw) or (lambda job: "complete"))
    continuation = research_job_recovery._continuation(state, {"study_context": source.study, "project_root": "synthetic"})
    assert continuation.run(None) == "complete"
    assert calls[0]["execution_resume_source_run_id"] == "run-approved"
    assert not calls[0]["development_resume_source_job_id"] and calls[0]["plan_change_request"] is None


def test_wrapped_continuation_reuses_the_same_ledger_and_job(source):
    from dataclasses import dataclass
    @dataclass
    class Request:
        study_context: dict
    original_job = Job("original", "agent-run")
    state = policy.ProviderJobRecovery(original_job)
    state.task = object()
    job = policy.RecoveryJob(state, "original-auto-1")
    wrapped = research_job_recovery.wrap_pipeline_runner(lambda bound: bound, Request(source.study))
    assert wrapped(job) is job
    assert research_job_recovery.existing_provider_task(job) is state.task
    job.emit({"type": "progress", "step": "planning"})
    assert original_job.events[-1]["step"] == "planning"


@pytest.mark.parametrize("transient", [True, False])
def test_execution_exception_seals_transport_identity_without_response_text(tmp_path, transient):
    from easyicu.research_agent.execution.phase import _step_record_step_exception
    error = RuntimeError("private response")
    if transient:
        error.status_code = 503
    records = []
    _step_record_step_exception(
        SimpleNamespace(step_id="synthetic-model"), error,
        shared_lock=threading.RLock(), findings=[], _flush_partial_manifest=lambda _: None,
        per_step_records=records, _append_terminal_step_record=lambda rows, row: rows.append(row),
    )
    diagnostic = records[0].get("transient_provider_failure")
    assert bool(diagnostic) is transient
    if transient:
        assert diagnostic == {"reason_code": "provider_http_error", "status_code": 503}


@pytest.mark.parametrize("failure", [
    {"reason_code": "provider_http_error", "status_code": 503},
    {"reason_code": "provider_timeout"}, {"reason_code": "provider_connection_failure"},
])
def test_sealed_execution_failure_reenters_recovery_without_finishing_ledger(tmp_path, monkeypatch, failure):
    from easyicu.research_agent.authority import runtime_artifacts
    state = policy.ProviderJobRecovery(Job("original", "agent-run"))
    rows = [{"step_id": "model", "status": "execution_raised", "transient_provider_failure": failure}]
    monkeypatch.setattr(runtime_artifacts, "load_run_artifact_authority", lambda _: {"per_step_records": rows})
    with pytest.raises(research_job_recovery.RecordedProviderStop) as stopped:
        research_job_recovery.recoverable_execution_stop(policy.RecoveryJob(state, "original"), tmp_path / "run-exec", {"gate": {"status": "blocked"}})
    assert policy.transient_provider_cause(stopped.value) == failure["reason_code"]
    assert state.execution_resume_run_id == "run-exec"
    rows.append({"step_id": "scientific", "status": "contract_failed"})
    research_job_recovery.recoverable_execution_stop(policy.RecoveryJob(state, "original"), tmp_path, {"gate": {"status": "blocked"}})


def test_execution_checkpoint_checks_capsule_and_returns_exact_digest(tmp_path, monkeypatch):
    run = tmp_path / "pipeline/run-exec"
    run.mkdir(parents=True)
    path = run / "human_review_checkpoint.json"
    path.write_text("sealed approval")
    target = SimpleNamespace(wrapper_dir=tmp_path, pipeline_run_id="run-exec")
    monkeypatch.setattr(agent_pipeline_runs, "_resolve_execution_resume_wrapper", lambda **kw: target)
    verified = []
    monkeypatch.setattr(agent_pipeline_runs, "_verified_execution_resume_inputs", lambda t: verified.append(t))
    checkpoint = agent_pipeline_runs.execution_resume_checkpoint(study={}, project_root="synthetic", source_run_id="run-exec")
    assert verified == [target]
    assert checkpoint["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    def refuse(t):
        raise agent_pipeline_runs.ResearchPipelineRunError("capsule_changed", "changed")
    monkeypatch.setattr(agent_pipeline_runs, "_verified_execution_resume_inputs", refuse)
    assert agent_pipeline_runs.execution_resume_checkpoint(study={}, project_root="synthetic", source_run_id="run-exec") is None


@pytest.mark.parametrize("error", [TimeoutError("timeout"), ConnectionError("offline")])
def test_writer_returns_a_typed_stop_and_recovers_only_its_approved_run(tmp_path, monkeypatch, error):
    from easyicu.research_agent.reporting.writer_stop import writer_failure_detail, writer_stop
    from easyicu.research_agent.authority import runtime_artifacts
    detail = writer_failure_detail(error)
    assert writer_stop(detail) == detail
    assert writer_stop({**detail, "provider_http_status": 401}) is None
    state = policy.ProviderJobRecovery(Job("writer", "agent-review"))
    monkeypatch.setattr(runtime_artifacts, "load_run_artifact_authority", lambda _: {"per_step_records": [{"status": "ok", "step_id": "analysis"}]})
    with pytest.raises(research_job_recovery.RecordedProviderStop):
        research_job_recovery.recoverable_execution_stop(
            policy.RecoveryJob(state, "writer"), tmp_path / "run-approved",
            {"gate": {"status": "blocked", "detail": detail}},
        )
    assert state.execution_resume_run_id == "run-approved"


def test_system_resume_survives_conversation_progress_tail():
    from easyicu.webserver.pi_copilot.projections import project_job
    events = [{"type": "progress", "step": "provider_auto_resume", "current": 1}]
    events += [{"type": "progress", "step": "planning"} for _ in range(30)]
    progress = project_job({"id": "test", "events": events})["progress"]
    assert progress[0]["step"] == "provider_auto_resume" and len(progress) == 21


def test_system_recovery_labels_in_both_languages():
    from pathlib import Path
    from tests.support.node import run_node
    root = Path(__file__).resolve().parents[2]
    script = """
global.window = {EU_HTML: {esc: x => x}, EasyICU: {guidedPi: {declare: (n, a) => {global.api=a;}}}};
require(process.argv[1]);
const labels = ['en', 'zh'].map(lang => {
  const owner = api.create({tr: (en,zh) => lang === 'zh' ? zh : en});
  return ['provider_auto_resume', 'provider_recovery_wait'].map(step => owner.pipelineEventLabel({step}));
});
require(process.argv[2]);
const stops = ['en', 'zh'].map(lang => api.create({tr:(en,zh)=>lang==='zh'?zh:en})
  .runFailureText('research_pipeline_provider_recovery_exhausted'));
process.stdout.write(JSON.stringify({labels,stops}));
"""
    result = run_node("node", script,
                      str(root / "src/easyicu/webserver/static/js/screens-guided-pi-activity.js"),
                      str(root / "src/easyicu/webserver/static/js/screens-guided-pi-error-text.js"), check=True)
    output = json.loads(result.stdout)
    labels = output["labels"]
    assert labels[0][0].startswith("System action:") and "verified checkpoint" in labels[0][0]
    assert labels[1][0].startswith("系统操作：") and "自动继续" in labels[1][0]
    assert "job budget" in labels[0][1] and "任务预算" in labels[1][1]
    assert "checkpoint is preserved" in output["stops"][0] and "已保留" in output["stops"][1]


def test_continuation_refusal_uses_the_error_text_owner_in_both_languages():
    from pathlib import Path
    from tests.support.node import run_node

    root = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js"
    script = r"""
const modules = {};
global.window = {EU_HTML: {esc:x=>x}, EasyICU: {guidedPi: {declare: (name, api) => {modules[name] = api;}}}};
require(process.argv[1]); require(process.argv[2]);
const shell = require('node:fs').readFileSync(process.argv[3], 'utf8');
const activityInit = shell.match(/const ACTIVITY = MODULES\.require\('activity'\)\.create\(\{[\s\S]*?\}\);/)[0];
const output = ['en','zh'].map(lang => {
  const tr = (en,zh) => lang === 'zh' ? zh : en;
  const errors = modules.errorText.create({tr});
  // Execute the production host wiring, including its error-text dependency.
  const activity = require('node:vm').runInNewContext(activityInit + '\nACTIVITY', {
    MODULES:{require:name=>modules[name]}, tr, runFailureText:errors.runFailureText,
    esc:x=>x, iconHtml:()=>'', resourceName:x=>x, resourceKey:x=>x,
    resourceButton:()=>'', publicAssistantText:x=>x,
  });
  const code = 'research_pipeline_development_resume_phase_not_replayed';
  const live = activity.pipelineEventLabel({step:'provider_continuation_refused',reason:code});
  const saved = activity.pipelineEventLabel({step:'provider_continuation_refused',reason_code:code});
  return {live,saved,expected:errors.runFailureText(code)};
});
process.stdout.write(JSON.stringify(output));
"""
    output = json.loads(run_node("node", script, str(root / "screens-guided-pi-error-text.js"),
                                 str(root / "screens-guided-pi-activity.js"),
                                 str(root / "screens-guided-pi.js"), check=True).stdout)
    for row in output:
        assert row["live"] == row["saved"] == row["expected"]
        assert "research_pipeline" not in row["live"]
    assert "same reviewed candidate again" in output[0]["live"]
    assert output[0]["live"].count(".") == 1
    assert "原已审阅的候选方案重新准备数据" in output[1]["live"]
    assert output[1]["live"].count("。") == 1


def test_real_launch_failure_defers_ledger_finish_until_checkpoint_continuation(tmp_path, monkeypatch):
    from easyicu import research_agent
    from easyicu.research_agent.acquisition import foundation
    from easyicu.webserver import provider_adapter, research_pipeline_run_preparation
    from easyicu.research_agent.execution import runner as runner_module
    from tests.support.typed_export import typed_export
    from tests.webserver.copilot.research_workflow_fixtures import (
        _acquisition_receipt, _foundation_profile, complete_study,
    )

    monkeypatch.setattr(runner_module, "probe_runner_availability", lambda kind, **kw:
                        runner_module.RunnerAvailability(kind=kind, available=True, image="synthetic"))
    acquisition = _acquisition_receipt()
    acquisition.blocked = False
    acquisition.universe_path = tmp_path / "universe.parquet"
    acquisition.universe_path.write_bytes(b"synthetic metadata placeholder")
    for key in ("cohort_authority_path", "cohort_authority_ref", "trajectory_path", "trajectory_authority_path", "trajectory_authority_ref"):
        setattr(acquisition, key, None)
    monkeypatch.setattr(foundation, "acquire_universe_for_question", lambda **kw: acquisition)
    monkeypatch.setattr(research_pipeline_run_preparation, "_data_foundation_profile", lambda **kw: _foundation_profile())
    client = SimpleNamespace(complete_with_usage=lambda *a, **kw: ("OK", {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}))
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (client, {"provider": "openai", "model": "synthetic"}))
    original_task = []
    class Pipeline:
        def run(self, **kwargs):
            error = RuntimeError("synthetic outage")
            error.status_code = 503
            raise error
    def build(config, *, services):
        original_task.append(services.provider_hard_stop)
        return Pipeline()
    monkeypatch.setattr(research_agent.ResearchAgentPipeline, "from_config", build)
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(policy.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(policy.time, "sleep", lambda s: setattr(clock, "now", clock.now + s))
    checks = []
    def select(state, launch):
        assert state.task is original_task[0]
        assert state.task.assert_active() > 0
        checks.append(launch)
        def continue_checkpoint(job):
            assert research_job_recovery.existing_provider_task(job) is original_task[0]
            original_task[0].finish()
            return {"continued": True}
        return policy.CheckpointContinuation(checkpoint={"kind": "planner", "sha256": "a" * 64}, run=continue_checkpoint)
    monkeypatch.setattr(research_job_recovery, "_continuation", select)
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(typed_export(tmp_path / "export")),
        study_context=complete_study(), project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment={"EASYICU_DISABLE_PROVIDER_ENV_FILE": "1"}, budget_mode="full_reviewed",
    )
    job = Job("synthetic-outage", "agent-run")
    assert runner(job) == {"continued": True}
    assert len(checks) == 2 and len(original_task) == 1
    assert sum(row.get("step") == "provider_auto_resume" for row in job.events) == 1


def test_conversation_keeps_two_distinct_system_resumes():
    from pathlib import Path
    from tests.support.node import run_node
    root = Path(__file__).resolve().parents[2]
    script = """
global.window = {EasyICU: {guidedPi: {declare: (n,a) => {global.api=a;}, require: () => null}}};
require(process.argv[1]);
const messages = [];
const owner = api.create({
  tr: (en,zh) => en, messages: () => messages, session: () => ({}),
  activity: {pipelineEventLabel: e => 'System action: ' + e.current},
  upsertActivityStep: (a,s) => {const i=a.steps.findIndex(x=>x.id===s.id); if(i<0) a.steps.push(s); else a.steps[i]=s;}
});
owner.hydrateProjectedJob({present:true, job_id:'synthetic', status:'done', progress:
  [1,2].map(current => ({type:'progress',step:'provider_auto_resume',current,total:2}))});
process.stdout.write(JSON.stringify(messages));
"""
    result = run_node("node", script, str(root / "src/easyicu/webserver/static/js/screens-guided-pi-childjob.js"), check=True)
    rows = json.loads(result.stdout)
    assert len(rows) == 1 and rows[0]["role"] == "activity"
    resumes = [row for row in rows[0]["steps"] if row["step"] == "provider_auto_resume"]
    assert len(resumes) == 2 and len({row["id"] for row in resumes}) == 2


@pytest.mark.parametrize("transient", [True, False])
def test_terminal_projection_settles_usage_unless_continuation_is_needed(tmp_path, monkeypatch, transient):
    events = []
    monkeypatch.setattr(agent_pipeline_runs, "_readiness_axes", lambda path: {})
    monkeypatch.setattr(agent_pipeline_runs, "_finish_web_provider_hard_stop", lambda task: events.append("settled"))
    monkeypatch.setattr(agent_pipeline_runs, "_write_projection", lambda **kw: events.append("projected") or {"done": True})
    error = research_job_recovery.RecordedProviderStop("research_pipeline_execution_failed", "outage")
    def check(job, path, projection):
        if transient:
            raise error
    monkeypatch.setattr(agent_pipeline_runs, "recoverable_execution_stop", check)
    if transient:
        with pytest.raises(research_job_recovery.RecordedProviderStop):
            agent_pipeline_runs._write_terminal_projection(job=object(), provider_task=object(), run_dir=tmp_path)
        assert events == ["projected"]
    else:
        assert agent_pipeline_runs._write_terminal_projection(job=object(), provider_task=object(), run_dir=tmp_path) == {"done": True}
        assert events == ["settled", "projected"]


def test_review_recovery_binds_the_existing_provider_and_approved_coordinates(source):
    state = policy.ProviderJobRecovery(Job("approval", "agent-review"))
    client, task = object(), object()
    entry = SimpleNamespace(
        provider_client=client, provider_hard_stop=task, wrapper_dir=source.run.parents[1],
        study=source.study, credential_source="pi_verified", budget_mode="full_reviewed",
        pending=SimpleNamespace(run_id="approved"),
    )
    research_job_recovery.bind_review_recovery(policy.RecoveryJob(state, "approval"), entry, {"synthetic": "frozen"})
    assert state.client is client and state.task is task
    assert state.review_launch["execution_resume_source_run_id"] == "approved"
    assert state.review_launch["study_context"] is source.study
    assert state.review_launch["provider_environment"] == {"synthetic": "frozen"}


@pytest.mark.parametrize("changed", [None, "study", "source", "revision", "checkpoint"])
def test_report_recovery_requires_the_same_source_revision_and_saved_request(tmp_path, monkeypatch, changed):
    from easyicu.webserver import manuscript_repair as writer
    study = {"id": "writer-study", "question": "Synthetic registered report"}
    target = SimpleNamespace(wrapper_dir=tmp_path / "run_source", pipeline_run_id="approved")
    seed = SimpleNamespace(prepared_package_binding={}, scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study))
    monkeypatch.setattr(writer.pipeline_owner, "_resolve_execution_resume_wrapper", lambda **kw: target)
    monkeypatch.setattr(writer.pipeline_owner, "load_recovery_seed", lambda path: seed)
    monkeypatch.setattr(writer.pipeline_owner, "_validated_execution_retry_config", lambda **kw: object())
    monkeypatch.setattr(writer.dataio, "validate_research_pipeline_source", lambda *a, **kw: None)
    monkeypatch.setattr(writer, "_current_revision_input", lambda t: (None, None))
    monkeypatch.setattr(writer, "prepare_registered_report_repair", lambda *a, **kw: object())
    monkeypatch.setattr(writer, "_source_fingerprint", lambda path: "sealed")
    monkeypatch.setattr(study_contexts, "get_context", lambda _: study)
    monkeypatch.setattr(writer.provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: pytest.fail("selection cannot call provider"))
    class SelectOnly:
        def __init__(self, job):
            pass
        def run(self, runner, select):
            return select(ConnectionError("synthetic"))
    monkeypatch.setattr(writer, "ProviderJobRecovery", SelectOnly)
    requests = []
    def replay(*args, revision_request):
        requests.append(revision_request)
        return None if changed == "checkpoint" else SimpleNamespace(revision_id="saved", rows=[(tmp_path / "section.json", "a" * 64, {})])
    monkeypatch.setattr(writer, "load_failed_writer_replay", replay)
    runner = writer.make_report_only_run_runner(
        study_context=study, project_root=str(tmp_path), provider={}, provider_environment={},
        credential_source="pi_verified", execution_resume_source_run_id="approved",
        budget_mode="full_reviewed", export_path="synthetic", report_revision_request="Clarify wording.",
    )
    if changed == "study":
        monkeypatch.setattr(study_contexts, "get_context", lambda _: {**study, "question": "changed"})
    elif changed == "source":
        monkeypatch.setattr(writer, "_source_fingerprint", lambda path: "changed")
    elif changed == "revision":
        monkeypatch.setattr(writer, "_current_revision_input", lambda t: (tmp_path, "changed"))
    selected = runner(Job("writer", "agent-run"))
    if changed:
        assert selected is None
    else:
        assert selected.checkpoint["kind"] == "writer"
        assert selected.checkpoint["revision_id"] == "saved" and callable(selected.run)
        assert requests == ["Clarify wording."]
