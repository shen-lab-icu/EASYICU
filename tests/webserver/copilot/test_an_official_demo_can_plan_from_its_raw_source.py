"""A source-choice button resolves official raw data without granting execution."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from easyicu.webserver import dataio, demo_source_storage as storage, study_contexts
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord
from easyicu.webserver.pi_copilot.demo_planning_source import ACTION, resolve_demo_planning_source
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.research_launch_scientific import bound_export_cohort_refusal
from tests.support.official_demo_raw import official_raw as official_raw
from tests.support.node import run_node


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


def snapshot(study):
    return build_research_workflow_snapshot(
        study=study, active_export_present=False, active_job=None, latest_run=None,
        export_cohort_refusal=bound_export_cohort_refusal(study, study["data_source"]["path"]),
    )


def test_offered_choice_moves_only_the_same_source_into_candidate_planning(choice):
    before = snapshot(choice.study)
    assert before.next_action_code == "bound_export_cohort_mismatch"
    assert before.bound_export_cohort_refusal["planning_source"] == choice.resolved.offer
    assert str(choice.paths.root) not in before.model_dump_json()

    result = choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    updated = study_contexts.get_context(choice.study["id"])
    assert updated["data_source"] == {
        "database": "eicu_demo", "path": str(choice.paths.raw), "label": choice.resolved.label,
    }
    assert updated["question"] == choice.study["question"]
    assert updated["cohort"] == choice.study["cohort"]
    assert updated["revision"] == choice.study["revision"] + 1
    assert updated["confirmations"]["extraction_completed"] is False
    assert updated.get("active_job_id") is None
    assert choice.record.binding.study_revision == updated["revision"]
    assert choice.record.data_source_authorization.extraction_scope == "study_required"
    assert result["resource"] is None
    assert str(choice.paths.root) not in json.dumps(result)
    assert snapshot(updated).next_action_code == "provider_ready_to_generate_plan"
    assert len(choice.saved) == 1
    # Raw identity is still not a prepared package or an executable-plan grant.
    with pytest.raises(dataio.ExportCohortError):
        dataio.validate_research_pipeline_source(str(choice.paths.raw), database="eicu_demo")
    with pytest.raises(PiCopilotError):
        choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    assert study_contexts.get_context(choice.study["id"])["revision"] == updated["revision"]


def render_choice(workflow, session):
    """Render the production confirmation HTML, including its actual actions."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    owner = Path(storage.__file__).parent / "static/js/screens-guided-pi-confirmation.js"
    result = run_node(node, r"""
let module;
global.window = { EasyICU: { guidedPi: {
  declare: (_name, api) => { module = api; }, optional: () => null,
} } };
require(process.argv[1]);
const session = JSON.parse(process.argv[3]);
const renderer = module.create({
  workflow: () => JSON.parse(process.argv[2]), session: () => session,
  sessionIsStale: () => Boolean((session.stale || {}).stale), busy: () => false,
  tr: (en, zh) => zh || en, esc: value => String(value),
  iconHtml: () => '', resourceButton: () => '',
});
process.stdout.write(renderer.workflowConfirmationHtml());
""", str(owner), json.dumps(workflow), json.dumps(session), check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_an_upgrade_offers_raw_in_the_real_renderer_without_reusing_the_old_export(choice, monkeypatch):
    source = next(item for item in storage.SOURCES if item.database == "eicu_demo")
    monkeypatch.setattr(storage, "native_export_producer_identity", lambda: "sha256:" + "b" * 64)
    assert choice.paths.export != storage.source_paths(source).export
    assert not storage.export_ready(choice.paths, source)
    assert storage._verified_public_demo(str(choice.paths.export)) is None
    assert bound_export_cohort_refusal(choice.study, str(choice.paths.export)) is not None

    workflow = snapshot(choice.study).model_dump(mode="json")
    offer = workflow["bound_export_cohort_refusal"]["planning_source"]
    assert workflow["next_action_code"] == "bound_export_cohort_mismatch"
    assert offer["expected_revision"] == choice.study["revision"]
    session = choice.record.model_dump(mode="json")
    html = render_choice(workflow, session)
    assert "data-gpi-confirm-action" in html
    assert "使用同源原始数据规划</button>" in html
    assert "已找到并核验" in html
    workflow["bound_export_cohort_refusal"].pop("planning_source")
    assert "data-gpi-confirm-action" not in render_choice(workflow, session)

    # The path-free action still binds only the same verified raw after upgrade.
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    assert study_contexts.get_context(choice.study["id"])["data_source"]["path"] == str(choice.paths.raw)


@pytest.mark.parametrize("defect", ["producer_missing", "producer_malformed", "producer_folder_mismatch",
                                  "release_mismatch", "manifest_symlink"])
def test_historical_exports_need_their_own_exact_official_receipts(choice, monkeypatch, defect):
    monkeypatch.setattr(storage, "native_export_producer_identity", lambda: "sha256:" + "b" * 64)
    marker = json.loads(choice.paths.prepared_marker.read_text())
    if defect == "producer_missing":
        marker.pop("producer_identity")
    elif defect == "producer_malformed":
        marker["producer_identity"] = "sha256:invalid"
    elif defect == "producer_folder_mismatch":
        marker["producer_identity"] = "sha256:" + "c" * 64
    elif defect == "release_mismatch":
        marker["version"] = "another-release"
    else:
        manifest = choice.paths.export / "_manifest.json"
        moved = choice.paths.root / "moved-manifest.json"
        manifest.rename(moved)
        manifest.symlink_to(moved)
    choice.paths.prepared_marker.write_text(json.dumps(marker))
    assert resolve_demo_planning_source(choice.study) is None


@pytest.mark.parametrize("change", [
    "study_revision", "session_revision", "project", "active_job", "active_turn", "offer_digest",
    "export_manifest", "raw_manifest", "raw_table", "raw_path", "prepared_marker",
])
def test_stale_or_changed_source_choices_write_nothing(choice, change):
    body = dict(choice.body)
    if change == "study_revision":
        study_contexts.upsert_context({"id": choice.study["id"], "purpose": "A newer purpose"})
    elif change == "session_revision":
        choice.record.binding = choice.record.binding.model_copy(update={
            "study_revision": choice.record.binding.study_revision + 1,
        })
    elif change == "project":
        body["project_id"] = "another-project"
    elif change == "active_job":
        study_contexts.handoff_context(choice.study["id"], active_job_id="job-running")
    elif change == "active_turn":
        choice.record.active_message_job_id = "message-running"
    elif change == "offer_digest":
        body["offer_sha256"] = "0" * 64
    elif change == "raw_table":
        with (choice.paths.raw / "patient.parquet").open("ab") as handle:
            handle.write(b"changed")
    else:
        path = (choice.paths.raw / "conversion_manifest.json" if change == "raw_manifest"
                else choice.paths.prepared_marker if change == "prepared_marker"
                else choice.paths.export / "_manifest.json")
        value = json.loads(path.read_text())
        value["data_path" if change == "raw_path" else "archive_sha256" if change == "prepared_marker"
              else "generation"] = str(choice.paths.root / "unrelated") if change == "raw_path" else "changed"
        path.write_text(json.dumps(value))
    current = study_contexts.get_context(choice.study["id"])
    with pytest.raises(PiCopilotError):
        choice.service.authorize_data_source(choice.record.session_id, **body)
    assert study_contexts.get_context(choice.study["id"]) == current
    assert choice.saved == []


def test_a_write_racing_with_source_choice_is_rejected_by_the_owner(choice, monkeypatch):
    original = study_contexts.upsert_context

    def racing_write(*args, **kwargs):
        original({"id": choice.study["id"], "purpose": "A concurrent study edit"})
        return original(*args, **kwargs)

    monkeypatch.setattr(study_contexts, "upsert_context", racing_write)
    with pytest.raises(PiCopilotError, match="study changed"):
        choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    current = study_contexts.get_context(choice.study["id"])
    assert current["purpose"] == "A concurrent study edit"
    assert current["data_source"] == choice.study["data_source"]
    assert current["revision"] == choice.study["revision"] + 1
    assert choice.saved == []


@pytest.mark.parametrize("change", ["full_database", "other_path", "matching_population", "busy"])
def test_only_an_official_mismatched_idle_source_is_offered(choice, change):
    study = deepcopy(choice.study)
    if change == "full_database":
        study["data_source"]["database"] = "eicu"
    elif change == "other_path":
        study["data_source"]["path"] = str(choice.paths.root / "other-export")
    elif change == "matching_population":
        study["cohort"] = {"preset": "all_icu"}
    else:
        study["active_job_id"] = "running"
    assert resolve_demo_planning_source(study) is None


def test_http_action_accepts_only_the_displayed_offer_and_no_paths(choice, monkeypatch):
    from fastapi.testclient import TestClient
    from easyicu.webserver.app import app
    from easyicu.webserver.routes import pi_copilot

    monkeypatch.setattr(pi_copilot, "get_pi_copilot_service", lambda: choice.service)
    client = TestClient(app)
    endpoint = f"/api/copilot/pi/sessions/{choice.record.session_id}/data-source-authorization"
    for field, value in (("path", "/arbitrary"), ("raw_path", "/arbitrary"),
                         ("database", "eicu_demo"), ("source_id", "another"),
                         ("grants", ["extract", "provider_run"]), ("expected_revision", True)):
        assert client.post(endpoint, json={**choice.body, field: value}).status_code == 422
    incomplete = {key: value for key, value in choice.body.items() if key != "offer_sha256"}
    assert client.post(endpoint, json=incomplete).status_code == 422
    assert choice.saved == []
    response = client.post(endpoint, json=choice.body)
    assert response.status_code == 200, response.text
    assert "job_id" not in response.json()
