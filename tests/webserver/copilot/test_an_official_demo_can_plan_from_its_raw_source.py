"""A source-choice button resolves official raw data without granting execution."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from easyicu.webserver import dataio, demo_source_prepare, demo_source_storage as storage, sources, study_contexts
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord
from easyicu.webserver.pi_copilot.demo_planning_source import ACTION, resolve_demo_planning_source
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.research_launch_scientific import bound_export_cohort_refusal
from tests.support.official_demo_raw import official_raw as official_raw
from tests.support.official_demo_choice import choice as choice
from tests.support.node import run_node




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
global.window = {};
const path = require('node:path');
const ownerDir = path.dirname(process.argv[1]);
require(path.join(ownerDir, 'screens-guided-pi-modules.js'));
require(path.join(ownerDir, 'screens-guided-pi-confirmation-data.js'));
require(process.argv[1]);
const session = JSON.parse(process.argv[3]);
const renderer = window.EasyICU.guidedPi.require('confirmation').create({
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


def test_confirmed_choice_recheck_does_not_scan_export_registry(choice, monkeypatch):
    from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice

    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])

    def forbidden(*args, **kwargs):
        pytest.fail("rechecking a bound source must not discover or inspect exports")

    monkeypatch.setattr(sources, "load_registry", forbidden)
    monkeypatch.setattr(dataio, "describe_export_source", forbidden)
    assert resolve_choice(study) is not None


def test_confirmed_raw_choice_binds_launch_and_survives_only_lifecycle_changes(choice, monkeypatch):
    from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    identity = resolve_choice(study)
    assert identity == sources.registered_source_identity(str(choice.paths.export))
    with pytest.raises(study_contexts.StudyContextError, match="server_owned"):
        study_contexts.upsert_context({"id": study["id"], "demo_planning_authority_ref": study["demo_planning_authority_ref"]})

    running = study_contexts.handoff_context(study["id"], active_job_id="job-test", expected_revision=study["revision"])
    finished = study_contexts.clear_active_job_if(study["id"], "job-test", current_stage="agent")["context"]
    assert finished["revision"] > study["revision"]
    assert resolve_choice(finished) == identity
    monkeypatch.setattr(preparation, "_prepare_scientific_launch", lambda request: None)
    monkeypatch.setattr(preparation, "_authorize_launch_provider", lambda request: None)
    monkeypatch.setattr(preparation, "_prepare_launch_execution", lambda *args: (None, None))
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=finished["data_source"]["path"], study_context=finished, project_root=None,
        provider={}, provider_environment={}, credential_source="pi_verified", literature_search_authorized=False,
        plan_revision_source_run_id="", execution_resume_source_run_id="", development_resume_source_job_id="",
        budget_mode="planner_canary", runner_image=None,
    )
    assert preparation.prepare_research_pipeline_run(request).source_identity == identity
    from dataclasses import replace
    with pytest.raises(preparation.ResearchPipelineRunError, match="confirmed Demo planning source changed"):
        preparation.prepare_research_pipeline_run(replace(request, export_path=str(choice.paths.export)))
    other = {**finished, "id": "different-study"}
    with pytest.raises(ValueError, match="stale"):
        resolve_choice(other)
    study_contexts.upsert_context({"id": study["id"], "data_source": choice.study["data_source"]})
    changed = study_contexts.get_context(study["id"])
    assert not changed.get("demo_planning_authority_ref")
    study_contexts.upsert_context({"id": study["id"], "data_source": finished["data_source"]})
    assert resolve_choice(study_contexts.get_context(study["id"])) is None


@pytest.mark.parametrize("change", ["registry", "reregister", "export_manifest", "conversion_manifest", "prepared_marker", "converted_marker", "signature", "science"])
def test_confirmed_raw_choice_fails_closed_after_drift(choice, change):
    from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
    from easyicu.webserver.raw_preparation_decisions import _private_directory

    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    assert resolve_choice(study) is not None
    if change == "registry":
        sources.remove_source(str(choice.paths.export))
    elif change == "reregister":
        sources.remove_source(str(choice.paths.export))
        source = storage.verified_public_demo_raw(choice.paths.raw)
        demo_source_prepare.register_export(source, choice.paths, SimpleNamespace(emit=lambda event: None))
    elif change == "export_manifest":
        path = choice.paths.export / "_manifest.json"
        path.write_text(path.read_text() + " ")
    elif change == "conversion_manifest":
        path = choice.paths.raw / "conversion_manifest.json"
        path.write_text(path.read_text() + " ")
    elif change in {"prepared_marker", "converted_marker"}:
        path = getattr(choice.paths, change)
        path.write_text(path.read_text() + " ")
    elif change == "signature":
        path = _private_directory() / f"demo-planning-{study['demo_planning_authority_ref']}.json"
        path.write_text(path.read_text() + " ")
    else:
        study_contexts.upsert_context({"id": study["id"], "question": "A different question"})
        assert not study_contexts.get_context(study["id"]).get("demo_planning_authority_ref")
        return
    with pytest.raises((ValueError, OSError, dataio.ExportCohortError)):
        resolve_choice(study)


def test_source_change_during_confirmation_cannot_issue_binding(choice, monkeypatch):
    from easyicu.webserver.pi_copilot import demo_planning_authority as authority

    original = authority._source_facts
    def change_after_first_read(*args):
        facts = original(*args)
        path = choice.paths.prepared_marker
        path.write_text(path.read_text() + " ")
        return facts
    monkeypatch.setattr(authority, "_source_facts", change_after_first_read)
    with pytest.raises(PiCopilotError):
        choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    after = study_contexts.get_context(choice.study["id"])
    assert after["revision"] == choice.study["revision"]
    assert not after.get("demo_planning_authority_ref")
    assert after["data_source"] == choice.study["data_source"]


def test_restoring_a_turn_never_replays_demo_source_authority(choice):
    import hashlib
    from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice

    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    assert resolve_choice(study) is not None
    source = study["data_source"]
    snapshot = {
        "schema_version": "easyicu.pi-turn-study-snapshot/1", "study_context_id": study["id"],
        "source_revision": study["revision"], "configuration": {
            "question": study["question"], "data_source": {
                "label": source["label"], "database": source["database"],
                "path_hash": hashlib.sha256(source["path"].encode()).hexdigest()[:16],
            }, "demo_planning_authority_ref": study["demo_planning_authority_ref"],
        },
    }
    restored = study_contexts.restore_turn_configuration_snapshot(
        study["id"], snapshot, expected_revision=study["revision"],
    )
    assert not restored.get("demo_planning_authority_ref")
    assert resolve_choice(restored) is None
