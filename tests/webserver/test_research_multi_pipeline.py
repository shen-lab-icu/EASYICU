"""Direct-owner integration; no ASGI, provider, export or research process runs."""
from types import SimpleNamespace

import pytest

from easyicu.webserver.research_multi_contracts import MultiDatabaseError, MultiDatabaseRequest, DatabaseTarget
from easyicu.webserver.research_multi_pipeline import PipelineDatabaseOwner
from easyicu.webserver.agent_review_recovery import _write


def request():
    return MultiDatabaseRequest(project_id="p", session_id="s", study_context_id="source",
                                targets=(DatabaseTarget(database="miiv"),))


def test_preparation_uses_single_run_factory_and_normal_study_handoff(tmp_path, monkeypatch):
    from easyicu.webserver import research_multi_pipeline as module, research_multi_extraction

    source = {"path": str(tmp_path / "synthetic"), "database": "miiv"}
    study = {"id": "child", "revision": 2, "data_source": source}
    trace = []
    owner = PipelineDatabaseOwner(None, request(), tmp_path)
    monkeypatch.setattr(owner, "_provider", lambda source: ({}, {}, "pi_verified", "a" * 64, "on_device"))
    monkeypatch.setattr(research_multi_extraction, "prepare_source", lambda study, source, job: (study, source))
    monkeypatch.setattr(module.dataio, "validate_research_pipeline_source", lambda *a, **k: trace.append("input"))
    monkeypatch.setattr(module, "research_pipeline_workspace", lambda: SimpleNamespace(project_root=lambda _: tmp_path))
    def make_runner(**kwargs):
        assert kwargs["budget_mode"] == "full_reviewed"
        assert kwargs["study_context"] == study
        def runner(job):
            trace.append("plan")
            return {"run_id": "target_run", "project_dir": str(tmp_path / "run_target")}
        return runner
    monkeypatch.setattr(module.agent_pipeline_runs, "make_research_pipeline_run_runner", make_runner)
    monkeypatch.setattr(module.agent_pipeline_runs, "pending_review", lambda _: {"budget_mode": "full_reviewed"})
    monkeypatch.setattr(module, "plan_approval_allowed", lambda _: True)
    monkeypatch.setattr(module.study_contexts, "handoff_context", lambda *a, **k: trace.append("handoff"))
    monkeypatch.setattr(module.study_contexts, "clear_active_job_if", lambda *a, **k: trace.append("clear"))
    receipt = tmp_path / "miiv.run.json"
    result = owner._plan(study, source, job=SimpleNamespace(id="job_miiv"), receipt=receipt)
    assert result["run_id"] == "target_run"
    assert trace == ["input", "handoff", "plan", "clear"]
    assert owner._plan(study, source, job=None, receipt=receipt) == result
    assert trace.count("plan") == 1


def test_interrupted_preparation_does_not_spend_a_fresh_planning_budget(tmp_path):
    receipt = tmp_path / "miiv.run.json"
    _write(receipt.with_suffix(".started.json"), {"job_id": "interrupted"})
    with pytest.raises(MultiDatabaseError, match="preparation_interrupted"):
        PipelineDatabaseOwner(None, request(), tmp_path)._plan({}, {}, job=None, receipt=receipt)


def test_prepared_scientific_configuration_is_used_for_the_study_handoff(tmp_path, monkeypatch):
    from easyicu.webserver import research_multi_pipeline as module, research_multi_extraction

    source = {"path": str(tmp_path / "synthetic"), "database": "miiv"}
    study = {"id": "child", "revision": 2, "data_source": source}
    prepared = {**study, "revision": 3, "analysis_timing": {"mode": "baseline"}}
    owner = PipelineDatabaseOwner(None, request(), tmp_path)
    monkeypatch.setattr(owner, "_provider", lambda source: ({}, {}, "pi_verified", "a" * 64, "on_device"))
    monkeypatch.setattr(research_multi_extraction, "prepare_source", lambda study, source, job: (study, source))
    monkeypatch.setattr(module.dataio, "validate_research_pipeline_source", lambda *a, **k: None)
    monkeypatch.setattr(module, "research_pipeline_workspace", lambda: SimpleNamespace(project_root=lambda _: tmp_path))
    def factory(**kwargs):
        callback = kwargs.get("on_prepared_study")
        if callback:
            callback(prepared)
        return lambda job: {"run_id": "target", "project_dir": str(tmp_path)}
    monkeypatch.setattr(module.agent_pipeline_runs, "make_research_pipeline_run_runner", factory)
    monkeypatch.setattr(module.agent_pipeline_runs, "pending_review", lambda _: {"budget_mode": "full_reviewed"})
    monkeypatch.setattr(module, "plan_approval_allowed", lambda _: True)
    handoffs = []
    monkeypatch.setattr(module.study_contexts, "handoff_context", lambda *a, **k: handoffs.append(k))
    monkeypatch.setattr(module.study_contexts, "clear_active_job_if", lambda *a, **k: None)
    owner._plan(study, source, job=SimpleNamespace(id="job"), receipt=tmp_path / "target.json")
    assert handoffs[0]["expected_revision"] == 3


@pytest.mark.parametrize("opt_in,mode", [(False, "research"), (True, "conversation")])
def test_revoked_session_permission_stops_before_provider_resolution(tmp_path, monkeypatch, opt_in, mode):
    from easyicu.webserver import research_multi_pipeline as module

    selected = SimpleNamespace(credential_source="pi_verified", provider="synthetic", model_dump=lambda **kw: {})
    service = SimpleNamespace(host_action_session=lambda *a, **k: None,
        _scoped_record=lambda *a, **k: SimpleNamespace(research_provider=selected, external_llm_opt_in=opt_in, agent_mode=mode))
    monkeypatch.setattr(module, "provider_environment_for_agent_run", lambda **kw: pytest.fail("revoked session resolved a provider"))
    with pytest.raises(MultiDatabaseError, match="multi_database_opt_in_required"):
        PipelineDatabaseOwner(service, request(), tmp_path)._provider({"path": "synthetic", "database": "miiv"})


def test_cancelled_target_never_opens_its_source_or_provider(tmp_path, monkeypatch):
    from easyicu.webserver import research_multi_pipeline as module
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import StudyRequirements, assess_study

    monkeypatch.setattr(module, "source_result", lambda _: None)
    decision = assess_study(research_capability_profile("miiv"), StudyRequirements(family="association_study"))
    monkeypatch.setattr(module, "target_decision", lambda *a: decision)
    opened = []
    def select(target):
        opened.append(target)
        raise MultiDatabaseError("synthetic_source_opened")
    monkeypatch.setattr(module, "select_source", select)
    result = PipelineDatabaseOwner(None, request(), tmp_path).prepare({}, DatabaseTarget(database="miiv"),
        batch_id="a" * 64, job=SimpleNamespace(is_cancel_requested=lambda: True))
    assert result.stop_code == "multi_database_cancelled"
    assert opened == []


def test_resume_enters_only_the_targets_verified_synthetic_scope(tmp_path, monkeypatch):
    from easyicu.synthetic_sources import generate_source
    from easyicu.research_agent.providers.client_trust import source_requires_clearance
    from easyicu.webserver import research_multi_pipeline as module, research_followup_source

    path = tmp_path / "synthetic"
    generate_source(path, database="hirid", stays=4)
    source = {"path": str(path), "database": "hirid"}
    owner = PipelineDatabaseOwner(None, request(), tmp_path)
    monkeypatch.setattr(module, "select_source", lambda _: (source, "a" * 64))
    monkeypatch.setattr(owner, "_provider", lambda _: ({}, {}, "pi_verified", "b" * 64, "unspecified"))
    target = SimpleNamespace(target=DatabaseTarget(database="hirid"),
        review=SimpleNamespace(study_context_id="child", run_id="target"))
    _write(tmp_path / "hirid.run.json", {"wrapper": str(tmp_path)})
    monkeypatch.setattr(module, "read_review", lambda *a: (None, None, None, SimpleNamespace(state="pending"), None))
    monkeypatch.setattr(module.study_contexts, "get_context", lambda _: {"id": "child", "revision": 1, "data_source": source})
    monkeypatch.setattr(module.study_contexts, "handoff_context", lambda *a, **k: None)
    monkeypatch.setattr(module.study_contexts, "clear_active_job_if", lambda *a, **k: None)
    monkeypatch.setattr(module.agent_pipeline_runs, "pending_review", lambda _: {"budget_mode": "full_reviewed"})
    monkeypatch.setattr(module, "plan_approval_allowed", lambda _: True)
    def resume(**kwargs):
        assert not source_requires_clearance("hirid")
        assert source_requires_clearance("miiv")
        return {"run_id": "target"}
    monkeypatch.setattr(module.agent_pipeline_runs, "resume_research_pipeline", resume)
    monkeypatch.setattr(research_followup_source, "load_completed_result", lambda *a, **k: None)
    assert owner.execute(target, reviewer="reviewer", job=SimpleNamespace(id="job"), recovering=False) == {"run_id": "target"}
    assert source_requires_clearance("hirid")


@pytest.mark.parametrize("changed", ["source", "provider", "handling", "review"])
def test_pipeline_revalidates_the_target_grant_and_review(tmp_path, monkeypatch, changed):
    from easyicu.webserver import research_multi_pipeline as module
    from easyicu.webserver.research_multi_contracts import ReviewedRun

    source = {"id": "fixture_miiv", "path": str(tmp_path / "credentialed"), "database": "miiv"}
    review = ReviewedRun(run_id="run", study_context_id="study", plan_sha256="a" * 64,
        context_sha256="a" * 64, scientific_review_sha256="a" * 64, requests_sha256="a" * 64,
        input_sha256="a" * 64, configuration_sha256="a" * 64)
    target = SimpleNamespace(target=DatabaseTarget(database="miiv", source_id=source["id"]),
        source_sha256="a" * 64, review=review, decision=object(),
        egress=module.egress_grant(source, "a" * 64, provider_sha256="f" * 64, data_handling="on_device"))
    digest = "d" * 64 if changed == "source" else target.source_sha256
    provider = "b" * 64 if changed == "provider" else target.egress.provider_sha256
    handling = "zero_retention_declared" if changed == "handling" else "on_device"
    binding = target.review.model_copy(update={"plan_sha256": "c" * 64}) if changed == "review" else target.review
    owner = PipelineDatabaseOwner(None, request(), tmp_path)
    monkeypatch.setattr(module, "select_source", lambda _: (source, digest))
    monkeypatch.setattr(owner, "_provider", lambda _: ({}, {}, "pi_verified", provider, handling))
    _write(tmp_path / "miiv.run.json", {"wrapper": str(tmp_path)})
    monkeypatch.setattr(module.study_contexts, "get_context", lambda _: {"id": target.review.study_context_id})
    monkeypatch.setattr(module, "read_review", lambda *a: (binding, None, None, SimpleNamespace(state="pending"), None))
    monkeypatch.setattr(module, "source_result", lambda _: None)
    monkeypatch.setattr(module, "target_decision", lambda *a: target.decision)
    with pytest.raises(MultiDatabaseError, match="multi_database_(egress|review)_changed"):
        owner.verify({}, target)


def test_inline_extraction_goes_through_existing_transaction_and_submission(tmp_path, monkeypatch):
    from easyicu.webserver import research_multi_extraction as owner
    from easyicu.webserver.routes import jobs as route

    trace = []
    study = {"id": "child", "revision": 1}
    source = {"path": str(tmp_path / "raw"), "database": "miiv"}
    prepared = {"path": str(tmp_path / "prepared"), "database": "miiv"}
    job = SimpleNamespace(id="child_job", is_cancel_requested=lambda: False)
    def transaction(**kwargs):
        assert kwargs["study"] is study
        assert kwargs["registered_source"] is None
        kwargs["submit"]({"study_context_id": "child"})
        trace.append("handoff_complete")
        return SimpleNamespace(reused=False)
    def submission(body, *, submit):
        assert body == {"study_context_id": "child"}
        def run(bound_job):
            assert bound_job is job
            assert trace == ["handoff_complete"]
            trace.append("extract")
            return {"study_context_rebound": True}
        assert submit("extract", run) is job
        return {"job_id": job.id}
    monkeypatch.setattr(owner, "submit_study_extraction", transaction)
    monkeypatch.setattr(route, "submit_extraction", submission)
    monkeypatch.setattr(owner.dataio, "prepared_export_manifest_path", lambda _: None)
    monkeypatch.setattr(owner.dataio, "validate_research_pipeline_source", lambda *a, **k: trace.append("validate"))
    monkeypatch.setattr(owner.study_contexts, "get_context", lambda _: {**study, "data_source": prepared})
    result = owner.prepare_source(study, source, job)
    assert result[1] == prepared
    assert trace == ["handoff_complete", "extract", "validate"]


def test_other_database_followup_dispatch_revalidates_the_suggestion(tmp_path, monkeypatch):
    from easyicu.webserver import research_multi_actions as actions, research_followups

    offered = dict(source_run_sha256="a" * 64, suggestions=[dict(suggestion_id="b" * 64,
        target_action="other_database", delta={"capability_decision": {"database": "aumc"}})])
    monkeypatch.setattr(research_followups, "result_followups", lambda *a, **k: offered)
    service = SimpleNamespace(_research_run_row=lambda *a: {"study_id": "study"})
    requests = []
    monkeypatch.setattr(actions, "propose", lambda service, request: requests.append(request) or {"batch": "same_owner"})
    result = actions.dispatch_followup(service, project_id="p", session_id="s", run_id="r",
                                       suggestion_id="b" * 64, expected_source_sha256="a" * 64)
    assert result == {"batch": "same_owner"}
    assert requests[0].targets == (DatabaseTarget(database="aumc"),)
    actions.dispatch_followup(service, project_id="p", session_id="s", run_id="r",
        suggestion_id="b" * 64, expected_source_sha256="a" * 64,
        targets=[{"database": "aumc"}, {"database": "sic"}])
    assert [row.database for row in requests[1].targets] == ["aumc", "sic"]
    with pytest.raises(MultiDatabaseError, match="followup_changed"):
        actions.dispatch_followup(service, project_id="p", session_id="s", run_id="r",
                                  suggestion_id="c" * 64, expected_source_sha256="a" * 64)
    with pytest.raises(MultiDatabaseError, match="followup_changed"):
        actions.dispatch_followup(service, project_id="p", session_id="s", run_id="r",
                                  suggestion_id="b" * 64, expected_source_sha256="c" * 64)
    assert len(requests) == 2


def test_durable_dispatch_recovers_the_same_owner(monkeypatch):
    from easyicu.webserver import research_auto_resume, research_multi_actions

    marker = object()
    monkeypatch.setattr(research_multi_actions, "recover_multi_runner", lambda descriptor, job: marker)
    assert research_auto_resume.recover_runner({"schema_version": research_multi_actions.SCHEMA}, None) is marker


def test_direct_routes_pass_only_scoped_bounded_host_inputs(monkeypatch):
    from easyicu.webserver.routes import research_multi as routes
    from fastapi import HTTPException

    service = object()
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    calls = []
    monkeypatch.setattr(routes.actions, "propose", lambda current, **kwargs: calls.append((current, kwargs)) or {"job_id": "j"})
    assert routes.propose(request()) == {"job_id": "j"}
    assert calls == [(service, {"request": request()})]
    with pytest.raises(ValueError):
        routes.Approval(project_id="p", session_id="s", batch_id="a" * 64, card_sha256="b" * 64,
                        plan={"steps": []})
    def fail(*a, **k):
        raise MultiDatabaseError("multi_database_approval_changed")
    monkeypatch.setattr(routes.actions, "approve", fail)
    with pytest.raises(HTTPException) as error:
        routes.approve(routes.Approval(project_id="p", session_id="s", batch_id="a" * 64, card_sha256="b" * 64))
    assert error.value.detail == {"error": "multi_database_approval_changed"}


def test_status_rejects_an_invalid_batch_id_with_a_stable_reason():
    from easyicu.webserver.routes import research_multi as routes
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        routes.status("../synthetic", project_id="p", session_id="s")
    assert error.value.status_code == 422
    assert error.value.detail == {"error": "multi_database_request_invalid"}


def test_account_provider_uses_only_the_conversation_bound_account(tmp_path, monkeypatch):
    from easyicu.webserver import research_multi_pipeline as module, codex_account_sessions

    selected = SimpleNamespace(credential_source="codex_user_auth", provider="codex", account_session_sha256="a" * 64,
        model="synthetic-model", model_dump=lambda **kw: {"account_session_sha256": "a" * 64})
    service = SimpleNamespace(host_action_session=lambda *a, **k: None,
        _scoped_record=lambda *a, **k: SimpleNamespace(research_provider=selected, external_llm_opt_in=True, agent_mode="research"))
    calls = []
    def account(binding, *, model):
        calls.append((binding, model))
        return {"synthetic_account_coordinate": "private"}
    monkeypatch.setattr(codex_account_sessions, "environment_for_binding", account)
    monkeypatch.setattr(module, "provider_environment_for_agent_run", lambda **kw: kw["account_environment"])
    monkeypatch.setattr(module.settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(module.agent_runs, "resolve_agent_provider_config", lambda **kw: {"name": "synthetic-provider"})
    # The real source-scope owner is tested separately; never inspect host data here.
    from contextlib import nullcontext
    monkeypatch.setattr(module, "public_demo_egress_scope", lambda _: nullcontext())
    monkeypatch.setattr(module, "egress_source_database", lambda path, database: database)
    value = PipelineDatabaseOwner(service, request(), tmp_path)._provider({"path": "synthetic", "database": "miiv"})
    assert calls == [("a" * 64, "synthetic-model")]
    assert value[2] == "codex_user_auth" and len(value[3]) == 64


def test_review_snapshot_binds_the_real_input_capsule_and_review_evidence(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore, sha256_of_file
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.webserver.research_multi_review import read_review
    from tests.support.result_append import completed_source, approvable_review

    wrapper, run, plan, context, _ = completed_source(tmp_path)
    review = approvable_review(plan).model_copy(update={"context_sha256": canonical_sha256(context.model_dump(mode="json"))})
    evidence = EvidenceStore(run)
    item = evidence.register_json(kind="log", description="Synthetic scientific review", evidence_id="scientific_plan_review",
                                  filename="scientific_plan_review.json", payload=review.model_dump(mode="json"))
    requests = human_review_requests_for_plan(findings=[], plan=plan, evidence=evidence, require_plan_review=True)
    checkpoint = HumanReviewCheckpoint.create(run_id="source", pipeline_config_sha256="d" * 64,
        environment_identity={}, llm_signature_sha256="e" * 64, run_input_capsule_sha256=sha256_of_file(run / "run_input_capsule.json"),
        capability_activation_sha256="f" * 64, runtime_capabilities=(), runtime_bundle=None,
        requests=requests, plan_handoff={"plan": plan.model_dump(mode="json")}, execution_coordinates={})
    write_checkpoint(run / "human_review_checkpoint.json", checkpoint)
    binding, restored, expected, _, _ = read_review(wrapper, "source", {"id": "study"})
    assert restored == context and expected == plan
    assert binding.scientific_review_sha256 == item.sha256
    (run / item.relative_path).write_text("{}")
    with pytest.raises(MultiDatabaseError, match="review_changed"):
        read_review(wrapper, "source", {"id": "study"})
