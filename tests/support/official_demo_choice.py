"""Synthetic official demo source-selection fixture."""
import json
from types import SimpleNamespace
import pytest
from easyicu.webserver import dataio, demo_source_prepare, demo_source_storage as storage, sources, study_contexts
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
from easyicu.webserver.pi_copilot.demo_planning_source import ACTION, resolve_demo_planning_source
from easyicu.webserver.pi_copilot.service import PiCopilotService


@pytest.fixture
def choice(official_raw, tmp_path, monkeypatch):
    paths, source, _ = official_raw
    paths.export.mkdir(parents=True)
    cohort = dataio.normalize_export_cohort_contract({"preset": "all_icu"})
    manifest = {
        "schema_version": "easyicu_native_export_v2", "database": source.database,
        "data_path": str(paths.raw), "format": "parquet",
        "cohort_contract": cohort, "cohort_execution": dataio.export_cohort_execution(cohort),
        "files": [{"file": "demographics.parquet", "module": "demographics"}],
    }
    (paths.export / "_manifest.json").write_text(json.dumps(manifest))
    storage.write_export_marker(paths, source, archive_sha256=source.archive_sha256, export={})
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "contexts.json")
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(sources.dataio, "describe_export_source", lambda path: {"ok": True, "database": source.database, "files": []})
    demo_source_prepare.register_export(source, paths, SimpleNamespace(emit=lambda event: None))
    study = study_contexts.upsert_context({
        "question": "描述成年患者每次住院首次 ICU 入住的院内死亡比例。",
        "data_source": {"database": source.database, "path": str(paths.export),
                        "stays": 42},
        "cohort": {"age_min": 18, "first_icu_stay_per_hospitalization": True},
        "modules": ["demographics"], "confirmations": {"extraction_completed": True},
    })
    service = PiCopilotService.__new__(PiCopilotService)
    record = PiSessionRecord(session_id="pi-demo-choice", project_id="project",
                            binding=service._binding_for_context(study))
    service.project_store = SimpleNamespace(assert_matches=lambda _project, context_id: context_id)
    monkeypatch.setattr(service, "_get_record", lambda _id: record)
    monkeypatch.setattr(service, "_stale_details", lambda _record: {})
    saved = []
    monkeypatch.setattr(service, "_save_record", lambda value: saved.append(value.model_dump(mode="json")))
    monkeypatch.setattr(service, "_public_session", lambda value: value.model_dump(mode="json"))
    resolved = resolve_demo_planning_source(study)
    assert resolved is not None
    body = {"project_id": "project", "action": ACTION,
            "expected_revision": study["revision"], "offer_sha256": resolved.offer["offer_sha256"]}
    return SimpleNamespace(paths=paths, study=study, service=service, record=record,
                           saved=saved, body=body, resolved=resolved)
