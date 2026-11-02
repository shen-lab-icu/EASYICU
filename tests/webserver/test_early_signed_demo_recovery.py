"""A signed source choice survives an early metadata-only restart."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.webserver import (
    agent_pipeline_runs, provider_adapter, research_auto_resume,
    research_early_resume as early, settings, sources, study_contexts,
)
from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw


@pytest.fixture
def signed_restart(choice, tmp_path, monkeypatch):
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    identity = resolve_choice(study)
    digest = study_contexts.scientific_configuration_sha256(study)
    wrapper = tmp_path / "runs" / study["id"] / "run_original"
    wrapper.mkdir(parents=True)
    task_id = "original-task"
    ledger = ProviderHardStopLedger(
        path=wrapper / ".runtime" / "provider_hard_stop_ledger.json", task_ids=(task_id,),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id=task_id, declaration_sha256=digest,
    )
    ledger.start_task(task_id)
    public = {"provider": "openai", "model": "synthetic", "credential_fingerprint": "b" * 64}
    early.create_seed(
        wrapper=wrapper, study_id=study["id"], scientific_sha256=digest,
        credential_source="pi_verified", budget_mode="planner_canary", provider_public=public,
        database=study["data_source"]["database"], export_path=study["data_source"]["path"],
        runner_image=None, original_source_identity=identity.to_dict(),
        hard_stop_ledger_path=str(ledger.path), hard_stop_task_id=task_id,
        hard_stop_declaration_sha256=digest, study=study,
    )
    descriptor = research_auto_resume.prepare_descriptor(
        study=study, project_root=tmp_path / "runs", credential_source="pi_verified",
        budget_mode="planner_canary", external_llm_opt_in=True,
    )
    descriptor["provider_binding"] = {"wrapper_dir": str(wrapper), "hard_stop_ledger_path": str(ledger.path),
                                      "hard_stop_task_id": task_id, "hard_stop_declaration_sha256": digest}
    descriptor["_resume_attempt"] = 2
    monkeypatch.setattr(study_contexts, "get_context", lambda _id: study)
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(research_auto_resume, "_provider_environment", lambda _seed: {})
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client",
                        lambda *a, **kw: (object(), dict(public)))
    job = SimpleNamespace(id="original", is_cancel_requested=lambda: False, emit=lambda _event: None)
    return SimpleNamespace(choice=choice, study=study, wrapper=wrapper, ledger=ledger,
                           descriptor=descriptor, job=job, identity=identity)


def test_signed_raw_choice_creates_and_recovers_exact_seed(signed_restart, monkeypatch):
    state = signed_restart
    seed = early.load_seed(state.wrapper)
    assert seed["source"]["source_identity"] == state.identity.to_dict()
    assert seed["source"]["source_authority_ref"] == state.study["demo_planning_authority_ref"]
    assert seed["source"]["source_kind"] == "signed_demo_planning_raw"
    captured = {}
    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner",
                        lambda **kwargs: captured.update(kwargs) or (lambda job: {"continued": True}))
    assert research_auto_resume.recover_runner(state.descriptor, state.job)(state.job) == {"continued": True}
    assert captured["export_path"] == state.study["data_source"]["path"]
    assert state.ledger.snapshot()["tasks"][0]["calls"] == []


@pytest.mark.parametrize("change", ["choice_missing", "raw_path", "database", "study_id", "signature",
                                   "export_manifest", "conversion_manifest", "registry", "prepared_marker"])
def test_changed_raw_choice_never_reaches_provider(signed_restart, monkeypatch, tmp_path, change):
    from easyicu.webserver.raw_preparation_decisions import _private_directory

    state = signed_restart
    current = deepcopy(state.study)
    if change == "choice_missing":
        current["demo_planning_authority_ref"] = ""
    elif change == "raw_path":
        path = tmp_path / "other-raw"
        path.mkdir()
        current["data_source"]["path"] = str(path)
    elif change == "database":
        current["data_source"]["database"] = "mimic_iv"
    elif change == "study_id":
        current["id"] = "another-study"
    else:
        path = (_private_directory() / f"demo-planning-{current['demo_planning_authority_ref']}.json"
                if change == "signature" else
                state.choice.paths.export / "_manifest.json" if change == "export_manifest" else
                state.choice.paths.raw / "conversion_manifest.json" if change == "conversion_manifest" else
                sources._CONFIG_PATH if change == "registry" else state.choice.paths.prepared_marker)
        path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(study_contexts, "get_context", lambda _id: current)
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client",
                        lambda *a, **kw: pytest.fail("provider must not start"))
    with pytest.raises(ResearchPipelineRunError):
        research_auto_resume.recover_runner(state.descriptor, state.job)
    assert state.ledger.snapshot()["tasks"][0]["calls"] == []


def test_metadata_binding_rejects_path_database_and_study_identity_before_seed(choice, tmp_path):
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    with pytest.raises(ValueError, match="early_source_changed"):
        early.metadata_binding("mimic_iv", study["data_source"]["path"], study)
    with pytest.raises(ValueError, match="early_study_changed"):
        early.create_seed(wrapper=tmp_path, study_id="other", scientific_sha256="a" * 64,
                          credential_source="pi_verified", budget_mode="planner_canary", provider_public={},
                          database=study["data_source"]["database"], export_path=study["data_source"]["path"],
                          runner_image=None, original_source_identity={}, hard_stop_ledger_path="",
                          hard_stop_task_id="", hard_stop_declaration_sha256="", study=study)
    assert not (tmp_path / early.FILENAME).exists()


def test_source_choice_is_rechecked_after_catalog_read(choice, monkeypatch):
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    original = agent_pipeline_runs._metadata_only_planning_catalog

    def changed_catalog(**kwargs):
        catalog = original(**kwargs)
        marker = choice.paths.prepared_marker
        marker.write_bytes(marker.read_bytes() + b" ")
        return catalog

    monkeypatch.setattr(agent_pipeline_runs, "_metadata_only_planning_catalog", changed_catalog)
    with pytest.raises(ValueError, match="early_source_choice_changed"):
        early.metadata_binding(study["data_source"]["database"], study["data_source"]["path"], study)


def test_registered_export_seed_binding_keeps_legacy_shape(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    @dataclass
    class Catalog:
        concepts: tuple = ()

    identity = SourceIdentity(database="miiv", registration_id="registered")
    path = tmp_path / "export"
    path.mkdir()
    monkeypatch.setattr(sources, "registered_source_identity", lambda value: identity if value == str(path) else None)
    monkeypatch.setattr(agent_pipeline_runs, "_metadata_only_planning_catalog", lambda **kwargs: Catalog())
    monkeypatch.setattr(agent_pipeline_runs, "_source_concepts_without_values", lambda _path: ())
    original = early.metadata_binding("miiv", str(path))
    assert early.metadata_binding("miiv", str(path), {"data_source": {"database": "miiv", "path": str(path)}}) == original
    assert set(original) == {"export_path", "database", "source_identity", "catalog_sha256", "without_values"}
