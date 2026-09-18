"""Checkpoint continuation restores its sealed launch mode, not UI guesses."""

import json
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_review_recovery as recovery
from easyicu.webserver import research_launch_resume as owner
from easyicu.webserver import study_contexts
from easyicu.webserver import research_run_submission as submission


@pytest.fixture
def sealed_source(tmp_path, monkeypatch):
    study = {
        "id": "study-scope", "question": "Describe the study cohort",
        "data_source": {"path": str(tmp_path / "source"), "database": "miiv"},
    }
    wrapper = tmp_path / study["id"] / "run_prior"
    run = wrapper / "pipeline/run_pipeline"
    run.mkdir(parents=True)
    checkpoint = run / "progressive_planner_checkpoint_003.json"
    checkpoint.write_text("fixture: chain validation is the separate existing owner")
    monkeypatch.setattr(owner, "_development_progressive_resume_binding", lambda **k: (checkpoint, "a" * 64))

    def save(mode, contract=None):
        seed = recovery.WebReviewRecoverySeed.create(
            wrapper_dir=str(wrapper.resolve()), study=study,
            scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
            provider_meta={}, provider_public={}, credential_source="pi_verified",
            budget_mode=mode,
            prepared_package_binding={"binding_sha256": "b" * 64} if mode == "full_reviewed" else None,
            pipeline_config={"bound_plan_revision_contract": contract},
            pipeline_config_sha256="c" * 64, acquisition_projection={},
            hard_stop_ledger_path="", hard_stop_task_id="web-prior",
            hard_stop_declaration_sha256="d" * 64, created_at=1.0,
        )
        path = wrapper / ".runtime/web_review_recovery_seed.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(seed.model_dump_json())
        return path

    def read(current=None):
        return owner._development_resume_launch_scope(
            project_root=str(tmp_path), study=study if current is None else current,
            source_job_id="prior",
        ).budget_mode
    return study, save, read


@pytest.mark.parametrize("mode", ["planner_canary", "full_reviewed"])
def test_restores_exact_digest_bound_launch_mode_without_changing_seed(sealed_source, mode):
    _, save, read = sealed_source
    path = save(mode)
    before = path.read_bytes()
    assert read() == mode
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "mode_tampered", "symlink"])
def test_missing_or_tampered_launch_scope_never_defaults_to_execution(sealed_source, mutation):
    _, save, read = sealed_source
    path = save("planner_canary")
    if mutation == "missing":
        path.unlink()
    elif mutation == "corrupt":
        path.write_text("{}")
    elif mutation == "symlink":
        target = path.with_name("other.json")
        path.rename(target)
        path.symlink_to(target)
    else:
        raw = json.loads(path.read_text())
        raw["budget_mode"] = "full_reviewed"
        path.write_text(json.dumps(raw))
    with pytest.raises(owner.ResearchPipelineRunError, match="launch scope"):
        read()


@pytest.mark.parametrize("change", [{"question": "Another question"}, {"id": "another-study"}])
def test_another_study_or_configuration_cannot_inherit_launch_scope(sealed_source, change):
    study, save, read = sealed_source
    save("full_reviewed")
    with pytest.raises(owner.ResearchPipelineRunError, match="study configuration"):
        read({**study, **change})


@pytest.mark.parametrize("version,mode,allowed", [
    (1, "planner_canary", False), (1, "full_reviewed", False),
    (2, "full_reviewed", False), (2, "planner_canary", True),
])
def test_legacy_seeds_require_digest_bound_mode_and_package_authority(
    sealed_source, version, mode, allowed,
):
    _, save, read = sealed_source
    path = save(mode)
    raw = json.loads(path.read_text())
    raw["schema_version"] = f"easyicu.web-review-recovery-seed/{version}"
    raw.pop("prepared_package_binding")
    payload = {k: v for k, v in raw.items() if k != "seed_sha256"}
    if version == 1:
        payload.pop("budget_mode")
    raw["seed_sha256"] = recovery.canonical_sha256(payload)
    path.write_text(json.dumps(raw))
    if allowed:
        assert read() == mode
    else:
        with pytest.raises(owner.ResearchPipelineRunError, match="launch scope"):
            read()


def test_seed_study_payload_must_match_its_declared_scientific_digest(sealed_source):
    _, save, read = sealed_source
    path = save("planner_canary")
    raw = json.loads(path.read_text())
    raw["study"]["question"] = "Another question"
    raw["seed_sha256"] = recovery.canonical_sha256(
        {k: v for k, v in raw.items() if k != "seed_sha256"},
    )
    path.write_text(json.dumps(raw))
    with pytest.raises(owner.ResearchPipelineRunError, match="study configuration"):
        read()


def test_legacy_unsigned_package_binding_cannot_restore_full_scope(sealed_source):
    _, save, read = sealed_source
    path = save("full_reviewed")
    raw = json.loads(path.read_text())
    raw["schema_version"] = "easyicu.web-review-recovery-seed/2"
    raw["seed_sha256"] = recovery.canonical_sha256({
        k: v for k, v in raw.items() if k not in {"seed_sha256", "prepared_package_binding"}
    })
    path.write_text(json.dumps(raw))
    with pytest.raises(owner.ResearchPipelineRunError, match="launch scope"):
        read()


@pytest.mark.parametrize("contract", [None, "", "  Exact sealed plan constraint\n保持原方案。\n"])
def test_launch_scope_preserves_original_contract_bytes(sealed_source, tmp_path, contract):
    study, save, _ = sealed_source
    path = save("full_reviewed", contract)
    before = path.read_bytes()
    scope = owner._development_resume_launch_scope(
        project_root=str(tmp_path), study=study, source_job_id="prior",
    )
    assert scope.plan_contract == contract
    assert path.read_bytes() == before


@pytest.mark.parametrize("contract", [[], {}, 7, "x" * (2 * 1024 * 1024 + 1)])
def test_untyped_or_oversize_contract_is_not_prompt_authority(sealed_source, contract):
    _, save, read = sealed_source
    save("full_reviewed", contract)
    with pytest.raises(owner.ResearchPipelineRunError, match="plan contract"):
        read()


@pytest.mark.parametrize("sealed,current,expected", [
    ("original", "", "original"), ("original", None, "original"),
    ("original", "original", "original"), (None, None, None),
])
def test_resume_reuses_only_the_sealed_plan_constraint(sealed, current, expected):
    scope = owner._DevelopmentResumeLaunchScope("full_reviewed", sealed)
    assert owner._development_resume_plan_contract(scope=scope, current_contract=current) == expected


@pytest.mark.parametrize("sealed,current", [(None, "new"), ("original", "changed")])
def test_resume_cannot_replace_the_original_plan_constraint(sealed, current):
    scope = owner._DevelopmentResumeLaunchScope("full_reviewed", sealed)
    with pytest.raises(owner.ResearchPipelineRunError, match="sealed plan contract"):
        owner._development_resume_plan_contract(scope=scope, current_contract=current)


@pytest.fixture
def submit_source(sealed_source, tmp_path, monkeypatch):
    study, save, _ = sealed_source
    events = []
    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    monkeypatch.setattr(submission.context_store, "get_context", lambda _: study)
    monkeypatch.setattr(submission.dataio, "describe_export_source", lambda _: {"ok": True})
    monkeypatch.setattr(submission.dataio, "prepared_export_manifest_path", lambda _: None)
    monkeypatch.setattr(submission.dataio, "validate_research_pipeline_source", lambda *a, **k: events.append("source"))
    monkeypatch.setattr(submission, "build_research_workflow_snapshot", lambda **k: SimpleNamespace(planning_prerequisites_missing=[]))
    monkeypatch.setattr(submission, "provider_environment_for_agent_run", lambda **k: {})
    monkeypatch.setattr(submission.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(submission.capabilities, "validate_compute_target", lambda _: {"ok": True})
    monkeypatch.setattr(submission.agent_runs, "resolve_agent_provider_config", lambda **k: {})
    monkeypatch.setattr(submission.context_store, "build_agent_context_binding", lambda *a, **k: {})
    monkeypatch.setattr(submission, "research_pipeline_workspace", lambda: SimpleNamespace(project_root=lambda _: tmp_path))
    monkeypatch.setattr(submission, "list_bound_run_history", lambda **k: [])
    monkeypatch.setattr(submission, "resumable_planner_checkpoint_job_id", lambda **k: "prior")
    monkeypatch.setattr(submission.context_store, "handoff_context", lambda *a, **k: {"revision": 1})
    monkeypatch.setattr(submission.capabilities, "record_tool_event", lambda *a, **k: None)

    def make_runner(**kwargs):
        events.append(("runner", kwargs))
        return lambda _: {}

    def enqueue(*args):
        events.append("job")
        return SimpleNamespace(id="next-job", kind="agent-run", status="queued")

    monkeypatch.setattr(submission.agent_pipeline_runs, "make_research_pipeline_run_runner", make_runner)
    monkeypatch.setattr(submission, "_submit_job", enqueue)

    def submit(intent="candidate_plan", start="resume_checkpoint"):
        return submission.submit_research_run(
            submission.ResearchRunSubmissionRequest(
                study_context_id=study["id"], provider="mock", credential_source="pi_verified",
                external_llm_opt_in=True, intent=intent, planner_start_mode=start,
            ),
            authorize=lambda: events.append("authorize"),
        )
    return study, save, submit, events


@pytest.mark.parametrize("intent", ["candidate_plan", "reviewed_analysis"])
@pytest.mark.parametrize("mode", ["planner_canary", "full_reviewed"])
def test_shared_submission_restores_scope_before_runner_and_authorization(submit_source, intent, mode):
    _, save, submit, events = submit_source
    save(mode)
    receipt = submit(intent)
    runner = next(event[1] for event in events if isinstance(event, tuple))
    assert receipt.budget_mode == runner["budget_mode"] == mode
    assert receipt.resume_source_job_id == runner["development_resume_source_job_id"] == "prior"
    assert events[-2:] == ["authorize", "job"]
    assert events.count("source") == (1 if mode == "full_reviewed" else 0)
    if mode == "full_reviewed":
        assert events[0] == "source"


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "stale"])
def test_invalid_scope_consumes_no_grant_or_job(submit_source, mutation):
    study, save, submit, events = submit_source
    path = save("full_reviewed")
    if mutation == "missing":
        path.unlink()
    elif mutation == "corrupt":
        path.write_text("{}")
    else:
        study["question"] = "Another question"
    with pytest.raises(submission.ResearchRunSubmissionError, match="resume_scope_"):
        submit()
    assert events == []


def test_fresh_candidate_ignores_old_prepared_checkpoint(submit_source):
    _, save, submit, events = submit_source
    save("full_reviewed")
    receipt = submit(start="fresh")
    runner = next(event[1] for event in events if isinstance(event, tuple))
    assert receipt.budget_mode == runner["budget_mode"] == "planner_canary"
    assert receipt.resume_source_job_id is None
    assert "development_resume_source_job_id" not in runner
    assert "source" not in events


def test_full_scope_does_not_bypass_prepared_source_validation(submit_source, monkeypatch):
    _, save, submit, events = submit_source
    save("full_reviewed")

    def reject(*args, **kwargs):
        raise submission.dataio.ExportCohortError("research_pipeline_manifest_required")

    monkeypatch.setattr(submission.dataio, "validate_research_pipeline_source", reject)
    with pytest.raises(submission.ResearchRunSubmissionError, match="manifest_required"):
        submit()
    assert events == []
