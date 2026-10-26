"""Public-demo acceptance keeps approvals, evidence checks and honest stops."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE

from tools import windows_laptop_acceptance as acceptance


def test_cassette_reaches_the_acceptance_owner(tmp_path, monkeypatch):
    from easyicu.research_agent import facade as owner
    from easyicu.webserver import dataio, research_launch_runtime
    from easyicu.research_agent.execution import runner

    calls = []
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.setattr(dataio, "validate_research_pipeline_source", lambda *a, **k: None)
    monkeypatch.setattr(research_launch_runtime, "_require_execution_runtime", lambda **k: "host")
    monkeypatch.setattr(runner, "probe_runner_availability", lambda *a, **k: SimpleNamespace(reason_code="docker_executable_missing"))
    def replay(**kwargs):
        calls.append(kwargs)
        return {"status": "passed", "reason_code": "public_demo_replay_result_verified",
                "evidence_bound_result": True, "provider_kind": "replay", "cassette_sha256": "a" * 64}
    monkeypatch.setattr(owner, "replay_public_demo", replay)
    from easyicu.webserver.demo_source_contracts import SOURCES

    cassette = tmp_path / "recorded.json"
    # One prepared export per registered demo source, in registry order.
    exports = [tmp_path / source.database for source in SOURCES]
    result = acceptance.probe_desktop_first_result(exports, cassette_path=cassette)
    assert calls[0]["cassette_path"] == cassette
    assert calls[0]["export_path"] == tmp_path / "eicu_demo"
    assert result["first_result"]["evidence_bound_result"] is True
    assert result["provider_kind"] == "replay"
    assert result["cassette_sha256"] == "a" * 64
    assert result["analysis_executed"] is True


@pytest.mark.parametrize("elapsed,expected", [(5, "passed"), (901, "failed")])
def test_only_a_verified_result_can_meet_the_time_budget(tmp_path, monkeypatch, elapsed, expected):
    work = tmp_path / "EasyICU 数据"
    cassette = tmp_path / "recorded.json"
    cassette.write_text("{}")
    monkeypatch.setattr(acceptance, "prepare_dataset", lambda source, paths, *a, **kw: paths.export)
    monkeypatch.setattr(acceptance, "probe_web", lambda *a: None)
    def result(exports, **kwargs):
        assert kwargs["cassette_path"] == cassette
        return {"first_result": {"status": "passed", "reason_code": "public_demo_replay_result_verified",
                                "evidence_bound_result": True}}
    monkeypatch.setattr(acceptance, "probe_desktop_first_result", result)
    ticks = iter([0, elapsed])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(ticks))
    report = acceptance.run_acceptance(work, work / "cache", tmp_path / "receipts",
        memory_pressure_gb=0, desktop_first_result=True, cassette_path=cassette)
    assert report["status"] == expected
    assert report["research"]["first_result"]["seconds_to_first_result"] == elapsed
    assert report["research"]["first_result"]["budget_met"] is (elapsed <= 900)
    assert "No evidence-bound result" not in (tmp_path / "receipts/summary.md").read_text()


def test_unrecorded_review_cannot_be_autoapproved():
    from easyicu.research_agent.facade import replay_review_decisions
    from easyicu.research_agent.providers.protocol import ReplayError

    request = SimpleNamespace(review_id="review-" + "a" * 16, authority_sha256="a" * 64)
    with pytest.raises(ReplayError, match="model_replay_review_missing"):
        replay_review_decisions(SimpleNamespace(requests=[request]), ())


def test_recorded_approval_must_match_the_full_authority():
    from easyicu.research_agent.facade import replay_review_decisions
    from easyicu.research_agent.providers.protocol import RecordedApproval, ReplayError

    recorded = RecordedApproval(review_id="review-" + "a" * 16, authority_sha256="a" * 64, decision="approved")
    request = SimpleNamespace(review_id=recorded.review_id, authority_sha256=recorded.authority_sha256)
    pending = SimpleNamespace(requests=[request])
    decisions = replay_review_decisions(pending, (recorded,))
    assert decisions[0].authority_sha256 == request.authority_sha256
    assert decisions[0].decision == "approved"
    request.authority_sha256 = "b" * 64
    with pytest.raises(ReplayError, match="model_replay_review_missing"):
        replay_review_decisions(pending, (recorded,))


def test_a_success_flag_without_evidence_is_not_an_m1_result(tmp_path):
    from easyicu.research_agent.facade import verify_first_result
    from easyicu.research_agent.providers.protocol import ReplayError

    (tmp_path / "manifest.json").write_text(json.dumps({"readiness": {"execution_complete": True, "analysis_validated": True}}))
    with pytest.raises(ReplayError, match="model_replay_result_unverified"):
        verify_first_result(tmp_path)


@pytest.mark.parametrize("changed", [None, "plan", "cohort", "reason", "grant"])
def test_acceptance_reuse_grant_binds_the_entire_plan_and_input(tmp_path, monkeypatch, changed):
    from easyicu.research_agent import facade as owner
    from easyicu.research_agent.providers.protocol import RecordedApproval, ReplayError

    grant = RecordedApproval(review_id="review-" + "a" * 16, authority_sha256="a" * 64,
        decision="approved", acceptance_reuse_authorized=changed != "grant",
        plan_sha256="b" * 64, cohort_sha256="c" * 64)
    request = SimpleNamespace(review_id="review-" + "d" * 16, authority_sha256="d" * 64,
        payload={"reason": "operator_plan_approval_required" if changed != "reason" else "scientific_stop"})
    monkeypatch.setattr(owner, "verified_review_coordinates", lambda pending, req: (
        "e" * 64 if changed == "plan" else "b" * 64,
        "f" * 64 if changed == "cohort" else "c" * 64,
    ))
    pending = SimpleNamespace(requests=[request], run_dir=str(tmp_path))
    if changed is None:
        decision = owner.replay_review_decisions(pending, (grant,))[0]
        assert decision.authority_sha256 == request.authority_sha256
        assert "acceptance" in decision.note
    else:
        with pytest.raises(ReplayError, match="model_replay_review_missing"):
            owner.replay_review_decisions(pending, (grant,))


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/outside.json", "missing.json"])
def test_workflow_cassette_is_confined_to_an_existing_repository_file(tmp_path, relative):
    from pathlib import Path

    with pytest.raises(acceptance.AcceptanceFailure, match="model_replay_cassette_path_forbidden"):
        acceptance.repository_cassette_path(Path(relative), root=tmp_path)


def test_repository_cassette_resolves_before_the_driver_changes_directory(tmp_path):
    from pathlib import Path

    cassette = tmp_path / "cassette.json"
    cassette.write_text("{}")
    assert acceptance.repository_cassette_path(Path("cassette.json"), root=tmp_path) == cassette


def demo_cassette(tmp_path, monkeypatch):
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope
    from easyicu.research_agent.providers.protocol import ModelIdentity, ReplayCassette
    from easyicu.webserver import demo_source_storage as storage

    monkeypatch.setattr(storage, "egress_source_database", lambda path, database: database)
    with public_demo_replay_scope(export_path=tmp_path, database="eicu_demo",
                                  profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref) as scope:
        return ReplayCassette.create(source=scope.source, model=ModelIdentity(provider="synthetic", model="synthetic-model"))


@pytest.mark.parametrize("failure", [RuntimeError("private details"), None])
def test_replay_receipts_mark_both_success_and_downstream_failure(tmp_path, monkeypatch, failure):
    from easyicu.research_agent import facade

    cassette = demo_cassette(tmp_path, monkeypatch)
    path = tmp_path / "cassette.json"
    cassette.write(path)
    def run(**kwargs):
        from easyicu.research_agent.providers.clients import ReplayLLMClient
        assert type(kwargs["client"]) is ReplayLLMClient
        if failure is not None:
            raise failure
        return {"status": "passed", "evidence_bound_result": True}
    monkeypatch.setattr(facade, "run_public_demo", run)
    result = facade.replay_public_demo(cassette_path=path, export_path=tmp_path, database="eicu_demo",
        work_dir=tmp_path / "work", question="Describe an invented shape.", primary_exposure="a",
        target_outcome="b", covariates=())
    assert result["provider_kind"] == "replay"
    assert result["cassette_sha256"] == cassette.cassette_sha256
    assert result["status"] == ("blocked" if failure else "passed")
    assert result["reportable"] is False and result["publishable"] is False
    assert "private details" not in json.dumps(result)
    assert json.loads((tmp_path / "work/model_replay_receipt.json").read_text())["cassette_sha256"] == cassette.cassette_sha256


def test_recording_command_uses_the_live_factory_and_writes_only_a_cassette(tmp_path, monkeypatch):
    from tools import record_public_demo_model as tool
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.client_trust import authorized_complete
    from easyicu.research_agent.providers.protocol import LLMMessage, ReplayCassette

    cassette = demo_cassette(tmp_path, monkeypatch)
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return ScriptedMockLLMClient(["an invented circle"])
    monkeypatch.setattr(tool, "build_provider_client", build)
    monkeypatch.setattr(tool, "recording_model_identity", lambda client: cassette.model)
    def run(**kwargs):
        assert authorized_complete(kwargs["client"], [LLMMessage("user", "Describe an invented circle.")]) == "an invented circle"
        return {"evidence_bound_result": True}
    monkeypatch.setattr(tool, "run_public_demo", run)
    output = tmp_path / "recorded.json"
    assert tool.main(["--export-path", str(tmp_path), "--database", "eicu_demo", "--provider", "openai",
        "--model", "chosen-model", "--output", str(output), "--work-dir", str(tmp_path / "work"),
        "--question", "Describe an invented circle.", "--primary-exposure", "a", "--target-outcome", "b"]) == 0
    assert calls[0]["provider"] == "openai"
    assert calls[0]["model"] == "chosen-model"
    assert "environment" not in calls[0]
    loaded = ReplayCassette.read(output)
    assert loaded.entries[0].response == "an invented circle"


def test_acceptance_entry_refuses_a_different_export_before_materialization(tmp_path, monkeypatch):
    from easyicu.research_agent import facade
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope
    from easyicu.research_agent.providers.protocol import ReplayError

    cassette = demo_cassette(tmp_path, monkeypatch)
    with public_demo_replay_scope(export_path=tmp_path, database="eicu_demo", profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref):
        client = ReplayLLMClient(cassette)
        with pytest.raises(ReplayError, match="model_replay_source_mismatch"):
            facade.run_public_demo(export_path=tmp_path / "credentialed", database="eicu_demo", work_dir=tmp_path / "work",
                client=client, question="Describe an invented shape.", primary_exposure="a", target_outcome="b",
                covariates=(), review=lambda pending: [])


def tiny_plan():
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    return AnalysisPlan(research_question="Describe an invented shape.", steps=[
        AnalysisStep(step_id="primary", planned_analysis_role="primary", intent="Count invented sides.",
                     expected_outputs=["statistic:synthetic_stat"])])


@pytest.mark.parametrize("defect", [None, "execution_complete", "analysis_validated", "plan_authority_verified",
                                   "paper_authorized", "missing_seal", "wrong_output", "stale_evidence", "no_primary"])
def test_m1_requires_current_verified_primary_execution_evidence(tmp_path, monkeypatch, defect):
    from easyicu.research_agent import facade
    from easyicu.research_agent.authority import plan_input_closure, step_runtime
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
    from easyicu.research_agent.providers.protocol import ReplayError

    plan = tiny_plan()
    if defect == "no_primary":
        plan.steps[0].planned_analysis_role = "auxiliary"
    (tmp_path / "plan.json").write_text(plan.model_dump_json())
    output = tmp_path / "invented.json"
    output.write_text('{"sides": 3}')
    store = EvidenceStore(root=tmp_path)
    record = store.register_file(kind="statistic", description="Invented fixture", source_path=output,
                                 evidence_id="synthetic_stat", produced_by_step="primary")
    gates = {"execution_complete": True, "analysis_validated": True, "plan_authority_verified": True, "paper_authorized": False}
    if defect in gates:
        gates[defect] = not gates[defect]
    row = {"step_id": "primary", "step_authority_capsule_ref": {"sha256": "a" * 64}}
    if defect == "missing_seal":
        row.pop("step_authority_capsule_ref")
    (tmp_path / "manifest.json").write_text(json.dumps({"plan_path": "plan.json", "readiness": gates, "per_step_records": [row]}))
    verified_plan = []
    monkeypatch.setattr(plan_input_closure, "resolve_registered_plan_authority", lambda **kwargs: verified_plan.append(kwargs["plan"]))
    digest = "b" * 64 if defect == "wrong_output" else record.sha256
    monkeypatch.setattr(step_runtime, "load_explicit_executed_success_step_capsule", lambda *a, **k:
        SimpleNamespace(capsule=SimpleNamespace(execution=SimpleNamespace(outputs=[SimpleNamespace(content=SimpleNamespace(sha256=digest))]))))
    if defect == "stale_evidence":
        verified_run_evidence_path(tmp_path, record).write_text("changed")
    if defect is None:
        assert facade.verify_first_result(tmp_path) == 1
        assert verified_plan == [plan]
    else:
        with pytest.raises(ReplayError, match="model_replay_result_unverified"):
            facade.verify_first_result(tmp_path)


@pytest.mark.parametrize("defect", [None, "evidence", "capsule", "cohort", "empty_evidence"])
def test_review_reuse_verifies_the_fresh_evidence_packet(tmp_path, defect):
    from easyicu.research_agent.facade import verified_review_coordinates
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority, ReviewExecutionAuthority
    from easyicu.research_agent.canonical_json import sha256_file
    from easyicu.research_agent.providers.protocol import ReplayError

    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"tiny synthetic input")
    capsule = tmp_path / "run_input_capsule.json"
    capsule.write_text(json.dumps({"cohort_sha256": sha256_file(cohort)}))
    store = EvidenceStore(root=tmp_path)
    evidence = store.register_file(kind="log", description="Synthetic capsule", source_path=capsule, evidence_id="run_input_capsule")
    execution = ReviewExecutionAuthority(pipeline_config_sha256="a" * 64, capability_activation_sha256="b" * 64,
        run_input_capsule_sha256="c" * 64 if defect == "capsule" else sha256_file(capsule))
    evidence_digests = {evidence.evidence_id: "d" * 64 if defect == "evidence" else evidence.sha256}
    if defect == "empty_evidence":
        evidence_digests = {}
    authority = PlanReviewAuthority.create(plan=tiny_plan(), evidence_sha256=evidence_digests, execution=execution)
    request = SimpleNamespace(payload={"plan_review_authority": authority.model_dump(mode="json")})
    pending = SimpleNamespace(run_dir=str(tmp_path))
    if defect == "cohort":
        cohort.write_bytes(b"different synthetic input")
    if defect is None:
        assert verified_review_coordinates(pending, request) == (authority.plan_sha256, sha256_file(cohort))
    else:
        with pytest.raises(ReplayError, match="model_replay_review_evidence_mismatch"):
            verified_review_coordinates(pending, request)


@pytest.mark.parametrize("answer", ["approve", "reject"])
def test_recorder_requires_an_explicit_acceptance_reuse_decision(monkeypatch, answer):
    from easyicu.research_agent import facade
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
    from easyicu.research_agent.providers.protocol import ReplayError

    request = HumanReviewRequest.create(kind="scientific_stop", summary="Review this invented plan.",
        authority_sha256="a" * 64, payload={"reason": "operator_plan_approval_required"})
    pending = SimpleNamespace(requests=[request])
    recorder = SimpleNamespace(approvals={})
    monkeypatch.setattr(facade, "verified_review_coordinates", lambda *a: ("b" * 64, "c" * 64))
    if answer == "approve":
        decisions = facade.record_review(pending, recorder, ask=lambda prompt: answer)
        assert decisions[0].authority_sha256 == request.authority_sha256
        grant = recorder.approvals[request.review_id]
        assert grant.acceptance_reuse_authorized is True
        assert grant.plan_sha256 == "b" * 64 and grant.cohort_sha256 == "c" * 64
    else:
        with pytest.raises(ReplayError, match="model_replay_recording_not_approved"):
            facade.record_review(pending, recorder, ask=lambda prompt: answer)
        assert recorder.approvals == {}


def test_recorder_cannot_turn_a_scientific_stop_into_a_reusable_grant():
    from easyicu.research_agent.facade import record_review
    from easyicu.research_agent.providers.protocol import ReplayError

    pending = SimpleNamespace(requests=[SimpleNamespace(payload={"reason": "scientific_stop"})])
    with pytest.raises(ReplayError, match="model_replay_scientific_review_required"):
        record_review(pending, SimpleNamespace(approvals={}), ask=lambda prompt: pytest.fail("must not offer approval"))


def test_acceptance_uses_the_normal_pipeline_review_and_evidence_owners(tmp_path, monkeypatch):
    from easyicu.research_agent import facade, pipeline as pipeline_module
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.orchestration.workflow import HumanReviewPending
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    cassette = demo_cassette(tmp_path, monkeypatch)
    materialized, events = [], []
    work = tmp_path / "work"
    def materialize(output, **kwargs):
        materialized.append((output, kwargs))
        return {"parquet": work / "invented.parquet"}
    monkeypatch.setattr(materializer, "materialize_to_parquet", materialize)
    pending = HumanReviewPending(run_id="synthetic", thread_id="synthetic", run_dir=str(work / "runs/synthetic"), requests=[])
    class Pipeline:
        def __init__(self, *, config, services):
            assert config.submission_profile_name == "public_demo_acceptance_dev"
            assert config.runner_kind == "host" and config.host_runner_authorized is True
            assert config.development_diagnostic is True
            assert config.require_human_plan_review is True and config.generate_manuscript is False
            assert services.llm is client
        def run(self, **kwargs):
            events.append("run")
            assert kwargs["cohort"] == work / "invented.parquet"
            return pending
        def resume_human_review(self, decisions, *, run_id):
            assert decisions == [] and run_id == pending.run_id
            events.append("resume")
            return SimpleNamespace(manifest_path=work / "runs/synthetic/manifest.json")
    monkeypatch.setattr(pipeline_module, "ResearchAgentPipeline", Pipeline)
    def review(outcome):
        assert outcome is pending
        events.append("review")
        return []
    def verify(run_dir):
        assert run_dir == work / "runs/synthetic"
        events.append("verify")
        return 1
    monkeypatch.setattr(facade, "verify_first_result", verify)
    with public_demo_replay_scope(export_path=tmp_path, database="eicu_demo", profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref):
        client = ReplayLLMClient(cassette)
        result = facade.run_public_demo(export_path=tmp_path, database="eicu_demo", work_dir=work, client=client,
            question="Describe an invented shape.", primary_exposure="a", target_outcome="b", covariates=("c",), review=review)
    assert materialized[0][1]["static_concepts"] == ("a", "c")
    assert events == ["run", "review", "resume", "verify"]
    assert result["evidence_bound_result"] is True and result["verified_primary_artifacts"] == 1


def test_cassette_is_refused_outside_desktop_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "prepare_dataset", lambda *a, **k: pytest.fail("source access must not run"))
    report = acceptance.run_acceptance(tmp_path / "work", tmp_path / "cache", tmp_path / "receipts",
        memory_pressure_gb=0, cassette_path=tmp_path / "cassette.json")
    assert report["status"] == "failed"
    assert report["failures"][0]["reason_code"] == "model_replay_requires_desktop_acceptance"


def test_acceptance_requires_a_fresh_run_directory(tmp_path, monkeypatch):
    from easyicu.research_agent import facade
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import ReplayError
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    cassette = demo_cassette(tmp_path, monkeypatch)
    (tmp_path / "work/runs").mkdir(parents=True)
    with public_demo_replay_scope(export_path=tmp_path, database="eicu_demo", profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref):
        with pytest.raises(ReplayError, match="model_replay_workdir_not_empty"):
            facade.run_public_demo(export_path=tmp_path, database="eicu_demo", work_dir=tmp_path / "work",
                client=ReplayLLMClient(cassette), question="Describe an invented shape.", primary_exposure="a",
                target_outcome="b", covariates=(), review=lambda pending: [])
