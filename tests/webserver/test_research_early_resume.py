"""Pre-call restart never replays an unknown Provider reservation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.webserver import agent_pipeline_runs, provider_adapter, research_auto_resume, research_early_resume, settings, study_contexts
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


@pytest.mark.parametrize("scope,acquisition,literature,expected", [
    (None, None, None, True),
    (object(), None, None, False),
    (None, object(), None, False),
    (None, None, {}, False),
])
def test_fresh_seed_never_drops_continuation_authority(scope, acquisition, literature, expected):
    assert research_early_resume.fresh_seed_scope(scope=scope, acquisition=acquisition, literature=literature) is expected


@pytest.fixture
def early(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    study = {"id": "study-a", "revision": 1,
             "question": "How common is Sepsis-3 in adult ICU stays, and is it associated with ICU mortality?",
             "data_source": {"path": str(source), "database": "miiv"}}
    digest = study_contexts.scientific_configuration_sha256(study)
    wrapper = tmp_path / "study-a" / "run_original"
    wrapper.mkdir(parents=True)
    task_id = "web-original"
    ledger = ProviderHardStopLedger(
        path=wrapper / ".runtime" / "provider_hard_stop_ledger.json", task_ids=(task_id,),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id=task_id, declaration_sha256=digest,
    )
    ledger.start_task(task_id)
    source_binding = {"database": "miiv", "export_path": str(source), "source_identity": {"database": "miiv", "registration_id": "r"}, "catalog_sha256": "a" * 64, "without_values": []}
    monkeypatch.setattr(research_early_resume, "metadata_binding", lambda database, path: dict(source_binding))
    public = {"provider": "openai", "model": "synthetic", "credential_fingerprint": "b" * 64}
    research_early_resume.create_seed(
        wrapper=wrapper, study_id=study["id"], scientific_sha256=digest,
        credential_source="pi_verified", budget_mode="planner_canary",
        provider_public=public, database="miiv", export_path=str(source), runner_image=None,
        original_source_identity=source_binding["source_identity"],
        hard_stop_ledger_path=str(ledger.path), hard_stop_task_id=task_id,
        hard_stop_declaration_sha256=digest,
    )
    descriptor = research_auto_resume.prepare_descriptor(
        study=study, project_root=tmp_path, credential_source="pi_verified",
        budget_mode="planner_canary", external_llm_opt_in=True,
    )
    descriptor["provider_binding"] = {
        "wrapper_dir": str(wrapper), "hard_stop_ledger_path": str(ledger.path),
        "hard_stop_task_id": task_id, "hard_stop_declaration_sha256": digest,
    }
    descriptor["_resume_attempt"] = 2
    monkeypatch.setattr(study_contexts, "get_context", lambda _id: study)
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(research_auto_resume, "_provider_environment", lambda _seed: {})
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (object(), dict(public)))
    job = SimpleNamespace(id="original", is_cancel_requested=lambda: False, emit=lambda _event: None)
    return SimpleNamespace(study=study, wrapper=wrapper, ledger=ledger, task_id=task_id, descriptor=descriptor, job=job, source_binding=source_binding, public=public)


def test_unstarted_recovery_uses_same_ledger_and_new_wrapper(early, monkeypatch):
    from easyicu.webserver.research_job_recovery import existing_provider_task

    captured = {}
    def make(**kwargs):
        captured.update(kwargs)
        def run(job):
            task = existing_provider_task(job)
            assert task.task_id == early.task_id
            assert task.ledger.path == early.ledger.path
            assert task.ledger.snapshot()["tasks"][0]["calls"] == []
            assert job.id == "original-restart-2"
            return {"continued": True}
        return run
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", make)
    assert research_auto_resume.recover_runner(early.descriptor, early.job)(early.job) == {"continued": True}
    assert captured["allow_environment_resume"] is False
    assert captured["export_path"] == early.source_binding["export_path"]
    assert "development_resume_source_job_id" not in captured


@pytest.mark.parametrize("frozen", [False, True])
def test_recovered_real_runner_reserves_original_ledger_in_new_wrapper(early, monkeypatch, frozen):
    """Exercise the real launch runner up to a fake Provider, never patient analysis."""
    import pandas as pd

    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver import durable_research, sources
    from easyicu.webserver import metadata_provider_client
    import sys

    source = early.study["data_source"]["path"]
    from pathlib import Path
    root = Path(source)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(root / "demographics.parquet", index=False)
    (root / "_manifest.json").write_text(json.dumps({
        "database": "miiv", "format": "parquet",
        "concept_selection": {"mode": "explicit", "modules": {"demographics": ["age"]}},
        "feature_definitions": {"included": False},
        "files": [{"file": "demographics.parquet", "module": "demographics", "concepts": 1,
                   "concept_ids": ["age"], "rows": 1}],
    }))
    monkeypatch.setattr(sources, "registered_source_identity", lambda _path: SourceIdentity(database="miiv", registration_id="r"))
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client",
                        lambda *a, **kw: (ScriptedMockLLMClient(["{}"]), dict(early.public)))
    bindings = []
    service = SimpleNamespace(store=SimpleNamespace(bind_provider=lambda job_id, value: bindings.append((job_id, value))))
    monkeypatch.setattr(durable_research, "_SERVICE", service)
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    if frozen:
        monkeypatch.setattr(metadata_provider_client, "MetadataProviderClient", lambda **kwargs: pytest.fail("frozen worker is unsupported"))
    else:
        monkeypatch.setattr(metadata_provider_client, "MetadataProviderClient", lambda **kwargs: kwargs["inner"])

    def acquisition(**kwargs):
        kwargs["llm"].complete([{"role": "user", "content": "synthetic metadata only"}], max_tokens=8)
        raise ResearchPipelineRunError("test_after_fake_provider", "stop before scientific execution")
    monkeypatch.setattr(agent_pipeline_runs, "_metadata_only_planning_acquisition", acquisition)
    early.job.cancel_requested = False
    with pytest.raises(ResearchPipelineRunError) as error:
        research_auto_resume.recover_runner(early.descriptor, early.job)(early.job)
    assert error.value.code == "test_after_fake_provider"
    new_wrapper = early.wrapper.parent / "run_original-restart-2"
    assert (new_wrapper / research_early_resume.FILENAME).is_file()
    assert (early.wrapper / research_early_resume.FILENAME).is_file()
    assert bindings[-1][0] == "original"
    assert bindings[-1][1]["wrapper_dir"] == str(new_wrapper)
    assert bindings[-1][1]["hard_stop_ledger_path"] == str(early.ledger.path)
    calls = early.ledger.snapshot()["tasks"][0]["calls"]
    assert len(calls) == 1 and calls[0]["role"] == "acquisition"


def test_unknown_inflight_reservation_never_replays(early, monkeypatch):
    early.ledger.reserve_transport_attempt(task_id=early.task_id, role="acquisition", model="synthetic", messages=[], max_tokens=4, prior_attempt_id=None)
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: pytest.fail("unsafe replay"))
    with pytest.raises(ResearchPipelineRunError) as error:
        research_auto_resume.recover_runner(early.descriptor, early.job)
    assert error.value.code == "research_auto_resume_early_provider_attempt_already_reserved"
    assert early.ledger.snapshot()["tasks"][0]["calls"][0]["state"] == "in_progress"


def test_second_pre_call_crash_keeps_the_first_ledger(early, monkeypatch):
    next_wrapper = early.wrapper.parent / "run_original-restart-2"
    next_wrapper.mkdir()
    research_early_resume.create_seed(
        wrapper=next_wrapper, study_id=early.study["id"],
        scientific_sha256=early.descriptor["scientific_configuration_sha256"],
        credential_source="pi_verified", budget_mode="planner_canary",
        provider_public=early.public, database="miiv", export_path=early.source_binding["export_path"],
        runner_image=None, original_source_identity=early.source_binding["source_identity"],
        hard_stop_ledger_path=str(early.ledger.path), hard_stop_task_id=early.task_id,
        hard_stop_declaration_sha256=early.descriptor["scientific_configuration_sha256"],
    )
    early.descriptor["provider_binding"]["wrapper_dir"] = str(next_wrapper)
    early.descriptor["_resume_attempt"] = 3
    def make(**kwargs):
        def run(job):
            assert job.id == "original-restart-3"
            assert job.provider_recovery.task.ledger.path == early.ledger.path
            return {"continued": True}
        return run
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", make)
    assert research_auto_resume.recover_runner(early.descriptor, early.job)(early.job) == {"continued": True}


@pytest.mark.parametrize("escape", ["outside", "runtime_symlink", "copied_ledger"])
def test_early_resume_refuses_switched_or_escaping_ledger(early, monkeypatch, tmp_path, escape):
    import shutil

    fake_wrapper = early.wrapper.parent / "run_fake"
    fake_wrapper.mkdir()
    runtime = fake_wrapper / ".runtime"
    if escape == "outside":
        foreign = tmp_path / "outside" / ".runtime"
        foreign.mkdir(parents=True)
        target = foreign / "provider_hard_stop_ledger.json"
        shutil.copyfile(early.ledger.path, target)
    elif escape == "runtime_symlink":
        runtime.symlink_to(early.ledger.path.parent, target_is_directory=True)
        target = runtime / "provider_hard_stop_ledger.json"
    else:
        runtime.mkdir()
        target = runtime / "provider_hard_stop_ledger.json"
        shutil.copyfile(early.ledger.path, target)
    early.descriptor["provider_binding"]["hard_stop_ledger_path"] = str(target)
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: pytest.fail("switched ledger"))
    with pytest.raises(ResearchPipelineRunError):
        research_auto_resume.recover_runner(early.descriptor, early.job)


@pytest.mark.parametrize("change", ["scope", "source", "provider", "opt_out", "cancel", "tamper"])
def test_changed_authority_refuses_early_resume(early, monkeypatch, change):
    if change == "scope":
        early.study["question"] = "different"
    elif change == "source":
        monkeypatch.setattr(research_early_resume, "metadata_binding", lambda *a: {**early.source_binding, "catalog_sha256": "c" * 64})
    elif change == "provider":
        monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (object(), {**early.public, "model": "changed"}))
    elif change == "opt_out":
        monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": False})
    elif change == "cancel":
        early.job.is_cancel_requested = lambda: True
    else:
        path = early.wrapper / research_early_resume.FILENAME
        payload = json.loads(path.read_text())
        payload["runner_image"] = "different"
        path.write_text(json.dumps(payload))
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: pytest.fail("unsafe replay"))
    with pytest.raises(ResearchPipelineRunError):
        research_auto_resume.recover_runner(early.descriptor, early.job)
