"""Project-scoped additions require one durable approval, never a Planner turn."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.result_append import completed_source
from easyicu.webserver.research_append_source import seal_completed_source


@pytest.fixture
def source(tmp_path):
    wrapper, run, _, _, baseline = completed_source(tmp_path)
    return seal_completed_source(wrapper=wrapper, run_dir=run, study={"id": "study"},
                                 baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True})


def test_result_actions_are_bound_to_source_and_only_propose(source, monkeypatch):
    from easyicu.webserver.research_append_actions import append_options, propose_for_conversation
    from easyicu.webserver import study_contexts
    from easyicu.webserver.research_append_run import append_run_for_id, pending_append_review

    calls = []
    service = SimpleNamespace(
        _research_run_row=lambda project, run: {"project_dir": str(source.wrapper), "study_id": "study"},
        rebind_session=lambda *args, **kwargs: None,
        host_action_session=lambda session, **kw: calls.append((session, kw)),
    )
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    options = append_options(service, project_id="project", run_id=source.seal["run_id"])
    assert {row["kind"] for row in options["options"] if row["available"]} == {"sensitivity", "exposure", "subgroup", "article"}
    assert options["source_run_sha256"] == source.digest
    assert "earlier export" in options["source_notice"]
    result = propose_for_conversation(service, session_id="session", project_id="project",
                                     run_id=source.seal["run_id"], kind="subgroup", expected_source_sha256=source.digest)
    child = append_run_for_id(result["run_id"])
    assert child.checkpoint.state == "pending"
    assert not (child.run_dir / "steps").exists()
    assert pending_append_review(result["run_id"])["is_append_run"] is True
    assert calls[0][1]["study_context_id"] == "study"
    again = propose_for_conversation(service, session_id="session", project_id="project",
                                    run_id=source.seal["run_id"], kind="subgroup", expected_source_sha256=source.digest)
    assert again["run_id"] == result["run_id"]
    with pytest.raises(ValueError, match="changed"):
        propose_for_conversation(service, session_id="session", project_id="project",
                                 run_id=source.seal["run_id"], kind="subgroup", expected_source_sha256="a" * 64)


def test_missing_source_seal_shows_status_and_owner_entry(tmp_path):
    from easyicu.webserver.research_append_actions import append_options

    service = SimpleNamespace(_research_run_row=lambda *args: {"project_dir": str(tmp_path), "study_id": "study"})
    result = append_options(service, project_id="project", run_id="old_run")
    assert len(result["options"]) == 4
    assert all(not row["available"] and row["owner"] and row["status_zh"] for row in result["options"])


@pytest.mark.parametrize("case", ["unfinished", "rejected"])
def test_source_requires_completed_human_approval(tmp_path, case):
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
    from easyicu.webserver.research_append_source import read_json

    wrapper, run, _, _, baseline = completed_source(tmp_path)
    path = run / "human_review_checkpoint.json"
    body = read_json(path)
    if case == "unfinished":
        body["state"] = "approved_pending_execution"
    else:
        body["approved_decisions"][0]["decision"] = "rejected"
        body["approved_decisions_sha256"] = canonical_sha256({
            "decisions": body["approved_decisions"], "records": body["approved_decision_records"],
        })
        body["consumed_decision_sha256"] = canonical_sha256(body["approved_decisions"])
    body["checkpoint_sha256"] = canonical_sha256({key: value for key, value in body.items() if key != "checkpoint_sha256"})
    write_checkpoint(path, HumanReviewCheckpoint.model_validate(body))
    with pytest.raises(ValueError, match="approved source run|Source approval differs"):
        seal_completed_source(wrapper=wrapper, run_dir=run, study={"id": "study"},
                              baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True})


def test_rejected_append_never_executes_or_changes_its_source(source):
    from easyicu.webserver import research_append_run as owner

    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_child",
                                    study={"id": "study"}, registered_source_changed=False)
    result = owner.approve_append_run(child.wrapper, decision="rejected", reviewer="researcher")
    assert result == {"execution_complete": False, "rejected": True}
    assert not (child.run_dir / "steps").exists()
    assert owner.load_append_run(child.wrapper).checkpoint.state == "rejected"
    assert source.verify().digest == source.digest
    with pytest.raises(ValueError, match="consumed"):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="researcher")


@pytest.mark.parametrize("stop_state", ["approved_pending_execution", "executing"])
def test_restart_reuses_recorded_approval_once(source, monkeypatch, stop_state):
    from easyicu.webserver import research_append_run as owner

    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_child",
                                    study={"id": "study"}, registered_source_changed=False)
    original = owner.write_checkpoint

    def power_loss(path, checkpoint):
        if checkpoint.state == stop_state:
            raise OSError("synthetic process stop")
        original(path, checkpoint)

    monkeypatch.setattr(owner, "write_checkpoint", power_loss)
    with pytest.raises(OSError, match="synthetic process stop"):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="researcher")
    staged = owner.load_append_run(child.wrapper).checkpoint
    assert staged.state == ("pending" if stop_state == "approved_pending_execution" else "approved_pending_execution")
    assert staged.approved_decisions
    monkeypatch.setattr(owner, "write_checkpoint", original)
    result = owner.approve_append_run(child.wrapper, decision="approved", reviewer="researcher")
    assert result["execution_complete"]
    assert owner.load_append_run(child.wrapper).checkpoint.approved_decisions == staged.approved_decisions
    with pytest.raises(ValueError, match="consumed"):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="researcher")
    assert len(list((child.run_dir / "steps").iterdir())) == 1


@pytest.mark.parametrize("target", ["cohort", "manifest"])
def test_tampered_source_is_rejected_before_execution(source, target):
    from easyicu.webserver import research_append_run as owner

    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_child",
                                    study={"id": "study"}, registered_source_changed=False)
    cohort = source.seal["products"]["artifact:analysis_cohort"]
    path = source.run_dir / (cohort["relative_path"] if target == "cohort" else "manifest.json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="researcher", source=source)
    assert not (child.run_dir / "steps").exists()


def test_changed_source_is_disclosed_at_approval_and_keeps_adopted_bytes(source, monkeypatch):
    from easyicu.webserver import research_append_run as owner, research_append_source

    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_child",
                                    study={"id": "study"}, registered_source_changed=False)
    monkeypatch.setattr(research_append_source, "registered_source_changed", lambda *args: True)
    result = owner.resume_append_review(run_id=child.checkpoint.run_id, study_context_id="study", decision="approved",
                                       reviewer="researcher", note="", current_study={"id": "study"})
    manifest = research_append_source.read_json(Path(result["project_dir"]) / "source_run_manifest.json")
    assert "registered source has changed" in manifest["append"]["source_notice"]
    assert "earlier export" in manifest["append"]["source_notice"]


def test_adopted_input_tamper_is_rejected_before_execution(source):
    from easyicu.webserver import research_append_run as owner

    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_child",
                                    study={"id": "study"}, registered_source_changed=False)
    binding = child.record["inputs"]["artifact:analysis_cohort"]
    (child.run_dir / binding["relative_path"]).write_bytes(b"synthetic changed input")
    with pytest.raises(ValueError, match="evidence changed"):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="researcher")
    assert not (child.run_dir / "steps").exists()


def test_runtime_rechecks_adopted_inputs_after_approval(source, monkeypatch):
    from easyicu.webserver import research_append_run as owner

    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_child",
                                    study={"id": "study"}, registered_source_changed=False)
    execute = owner.execute_result_append

    def changed_after_review(**kwargs):
        binding = child.record["inputs"]["artifact:analysis_cohort"]
        path = child.run_dir / binding["relative_path"]
        path.write_bytes(path.read_bytes() + b" ")
        return execute(**kwargs)

    monkeypatch.setattr(owner, "execute_result_append", changed_after_review)
    with pytest.raises(ValueError, match="Adopted append input changed"):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="researcher")
    assert owner.load_append_run(child.wrapper).checkpoint.state == "failed"


def test_append_card_survives_a_source_registry_change(source):
    from easyicu.webserver import agent_runs, research_append_run as owner
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot, _enrich_plan_review

    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_child",
                                    study={"id": "study"}, registered_source_changed=False)
    record = agent_runs.read_run_record(str(child.wrapper))
    row = agent_runs._history_row(record, child.wrapper)
    changed = {"id": "study", "question": source.plan.research_question,
               "data_source": {"database": "miiv", "path": str(source.wrapper.parent / "new_export")}}
    snapshot = build_research_workflow_snapshot(study=changed, active_export_present=False, active_job=None,
                                                latest_run=row, plan_review_authority=owner.pending_append_review(child.checkpoint.run_id))
    assert snapshot.plan_execution_ready
    assert snapshot.next_action_code == "operator_plan_approval_required"
    enriched = _enrich_plan_review(snapshot, study=changed, review=record.to_dict())
    preview = enriched.plan_conversation_preview
    assert len(preview["clinical_summary"]["lines"]) == 5
    assert preview["append"]["source_run_id"] == "source"
    assert len(preview["append"]["steps"]) == 1


@pytest.mark.parametrize("mismatch", ["none", "registry", "binding", "path"])
def test_source_identity_is_checked_against_the_registered_export(monkeypatch, mismatch):
    from easyicu.webserver import sources, dataio
    from easyicu.webserver.research_append_source import registered_source_changed

    old = {"path": "/synthetic/export", "database": "miiv", "source_id": "export-one"}
    binding = {"sha256": "a" * 64}
    source = SimpleNamespace(seal={"data_source": old, "prepared_package_binding": binding})
    calls = []

    def registry(path):
        calls.append("registry")
        if mismatch == "registry":
            raise ValueError("synthetic registry changed")

    def validate(path, *, database, expected_binding):
        calls.append("binding")
        assert expected_binding == binding and database == "miiv"
        if mismatch == "binding":
            raise ValueError("synthetic export changed in place")

    monkeypatch.setattr(sources, "resolve_registered_export", registry)
    monkeypatch.setattr(dataio, "validate_research_pipeline_source", validate)
    current = {**old, "path": "/synthetic/new"} if mismatch == "path" else old
    assert registered_source_changed(source, {"data_source": current}) is (mismatch != "none")
    if mismatch == "none":
        assert calls == ["registry", "binding"]


def test_completion_retains_the_source_seal_before_recovery_cleanup(tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, agent_review_recovery
    from easyicu.webserver.research_append_source import load_append_source, SEAL_NAME

    wrapper, run, _, _, baseline = completed_source(tmp_path)
    seed = SimpleNamespace(study={"id": "study"}, prepared_package_binding={"synthetic": True},
                           pipeline_config={"current_case_scientific_runtime_authority": baseline.model_dump(mode="json")})
    monkeypatch.setattr(agent_review_recovery, "load_recovery_seed", lambda _: seed)
    removed = []

    def cleanup(path):
        assert (path / SEAL_NAME).exists()
        removed.append(path)

    monkeypatch.setattr(agent_pipeline_runs, "_remove_local_recovery", cleanup)
    agent_pipeline_runs._cleanup_recovery_after_projection(wrapper_dir=wrapper, projection={"run_id": run.name})
    assert removed == [wrapper]
    assert load_append_source(wrapper).seal["run_id"] == run.name


def test_unsupported_subgroup_and_exposure_have_no_executable_owner(source):
    from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
    from easyicu.research_agent.planning.family_spec.baseline_association_template import ExposureAppend, SubgroupAppend

    context = source.context.model_copy(deep=True, update={"variables": [
        v.model_copy(update={"dtype": "string"}) if v.name == "marker" else v for v in source.context.variables]})
    with pytest.raises(ValueError, match="numeric source"):
        ExposureAppend().compile(source.plan, context, source.baseline)
    plan = source.plan.model_copy(deep=True)
    primary = plan.steps[3]
    requirement = primary.model_requirements[0]
    terms = [term.model_copy(update={"coding": "continuous", "transform": "identity", "levels": None,
                                                       "reference_level": None}) if term.name == "stratum" else term
             for term in requirement.model_terms]
    primary.model_requirements = [requirement.model_copy(update={"model_terms": terms})]
    primary.scientific_capability = "association_adjusted_v1"
    authority = build_baseline_authority(**{**source.baseline.model_dump(mode="python", exclude={"execution_contract_sha256"}),
                                            "model_requirement_sha256": None})
    authority = authority.seal_adjustment_roster(plan)
    plan = authority.bind_plan(plan)
    with pytest.raises(ValueError, match="No eligible stratifier"):
        SubgroupAppend().compile(plan, source.context, authority)
