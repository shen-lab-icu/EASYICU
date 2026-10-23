"""A named database can use its registered export through one explicit choice.

Only synthetic registry metadata and disposable StudyContexts are used here;
no export directory or patient rows are needed for source authorization.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from easyicu.databases.profiles import iter_database_profiles
from easyicu.webserver import research_run_submission, sources, study_contexts
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import source_recommendation, study_context_update
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    PiSessionDataSourceAuthorization,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.project_authority import ProjectAuthorityStore
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as routes
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

PROJECT = "project-registered-offer"
SESSION = "pi_registered_offer"


def _row(source_id: str, database: str = "eicu", *, stays: int = 40, ok: bool = True) -> dict:
    return {
        "id": source_id, "database": database, "ok": ok,
        "path": f"/synthetic/private/{source_id}", "label": "internal-run-label",
        "path_hash": "private-path-hash", "generated": "2026-10-19T08:00:00Z",
        "modules": ["demographics", "outcome"], "summary": {"stays": stays},
    }


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[PiCopilotService, dict]:
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "contexts.json")
    registry = {"active_path": None, "sources": [
        _row("src_best"), _row("src_small", stays=10),
        _row("src_other", "miiv", stays=100), _row("src_invalid", stays=200, ok=False),
    ]}
    monkeypatch.setattr(sources, "load_registry", lambda: copy.deepcopy(registry))
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json", gateway=FakeGateway(),
        project_store=ProjectAuthorityStore(tmp_path / "projects.json"),
    )
    monkeypatch.setattr(service, "_latest_run_id", lambda *args, **kwargs: None)
    study = study_contexts.upsert_context({
        "id": "study-offer", "question": "How does an early measurement relate to ICU death?",
    }, active=True)
    service.project_store.bind(PROJECT, study["id"])
    service._save_record(PiSessionRecord(
        session_id=SESSION, project_id=PROJECT,
        binding=service._binding_for_context(study),
        data_source_authorization=service._new_session_data_authorization(study, agent_mode="research"),
    ))
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    return service, registry


def _offer(service: PiCopilotService, database: str = "eicu") -> dict:
    return service.get_registered_source_offer(SESSION, project_id=PROJECT, database=database)


def _use(service: PiCopilotService, source_id: str = "src_best", database: str = "eicu") -> dict:
    return service.authorize_data_source(
        SESSION, project_id=PROJECT, action="use_registered_source",
        database=database, source_id=source_id,
    )


def _bytes(service: PiCopilotService) -> tuple[bytes, bytes]:
    return study_contexts._CONFIG_PATH.read_bytes(), service.store_path.read_bytes()


def test_offer_is_family_scoped_path_free_and_read_only(host) -> None:
    service, _ = host
    before = _bytes(service)
    offer = _offer(service)
    assert offer == {
        "database": "eicu",
        "recommended": {"source_id": "src_best", "label": "eICU v2.0", "stays": 40,
                        "module_count": 2, "generated_date": "2026-10-19"},
        "choices": [], "reason": "most_complete_local_dataset",
    }
    assert _offer(service, "miiv")["recommended"]["source_id"] == "src_other"
    assert _offer(service, "hirid") == {
        "database": "hirid", "recommended": None, "choices": [], "reason": None,
    }
    encoded = json.dumps(offer)
    for private in ("/synthetic/", "path", "private-path-hash", "internal-run-label", "src_invalid"):
        assert private not in encoded
    assert _bytes(service) == before


def test_ties_are_numbered_choices_and_each_can_be_selected(host) -> None:
    service, registry = host
    registry["sources"].append(_row("src_tied"))
    offer = _offer(service)
    assert offer["recommended"] is None
    assert offer["reason"] is None
    assert [row["source_id"] for row in offer["choices"]] == ["src_best", "src_tied"]
    assert [row["choice"] for row in offer["choices"]] == [1, 2]
    assert all(set(row) == {"choice", "source_id", "label", "stays", "module_count", "generated_date"}
               for row in offer["choices"])
    assert _use(service, "src_tied")["session"]["data_source_authorization"]["status"] == "confirmed"


def test_tool_and_offer_use_the_same_recommendation_owner(host, monkeypatch) -> None:
    service, _ = host
    calls = []
    original = source_recommendation.recommend_registered_export

    def recommend(rows, *, allow_official_demo=False):
        calls.append([row["source_id"] for row in rows])
        # Changing the shared policy must affect both consumers.
        return original([row for row in rows if row["source_id"] == "src_small"],
                        allow_official_demo=allow_official_demo)

    monkeypatch.setattr(source_recommendation, "recommend_registered_export", recommend)
    offer = _offer(service)
    tool = tool_module.execute_tool("easyicu_list_data_sources", {"database": "eicu"},
        ToolExecutionContext(session=service._get_record(SESSION), user_message="Use eICU for this study"))
    assert offer["recommended"]["source_id"] == "src_small"
    assert tool["details"]["recommended_source"]["source_id"] == "src_small"
    assert calls == [["src_best", "src_small"], ["src_best", "src_small"]]


def test_tied_tool_and_offer_choices_match(host) -> None:
    service, registry = host
    registry["sources"].append(_row("src_tied"))
    tool = tool_module.execute_tool("easyicu_list_data_sources", {"database": "eicu"},
        ToolExecutionContext(session=service._get_record(SESSION)))
    assert _offer(service)["choices"] == tool["details"]["registered_source_choices"]


def test_binding_uses_the_registry_snapshot_that_was_offered(host, monkeypatch) -> None:
    service, registry = host
    calls = []

    def load_registry():
        calls.append(True)
        snapshot = copy.deepcopy(registry)
        if len(calls) > 1:
            snapshot["sources"][0]["database"] = "miiv"
            snapshot["sources"][0]["path"] = "/synthetic/replaced-export"
        return snapshot

    monkeypatch.setattr(sources, "load_registry", load_registry)
    _use(service)
    study = study_contexts.get_context("study-offer")
    assert study["data_source"]["database"] == "eicu"
    assert study["data_source"]["path"] == registry["sources"][0]["path"]


def test_binding_revision_conflict_leaves_both_stores_byte_identical(host, monkeypatch) -> None:
    service, _ = host
    before = _bytes(service)

    def conflict(*args, **kwargs):
        assert kwargs["require_revision"] is True
        assert kwargs["expected_revision"] == service._get_record(SESSION).binding.study_revision
        raise study_contexts.StudyContextError({"error": "study_context_revision_conflict"})

    monkeypatch.setattr(study_contexts, "upsert_context", conflict)
    with pytest.raises(PiCopilotError) as error:
        _use(service)
    assert error.value.code == "study_context_revision_conflict"
    assert _bytes(service) == before


def test_use_binds_through_the_tool_owner_and_confirms_once(host, monkeypatch) -> None:
    service, _ = host
    owner_calls = []
    original = study_context_update.update_study_context

    def update(context, params, **kwargs):
        owner_calls.append(params)
        return original(context, params, **kwargs)

    monkeypatch.setattr(study_context_update, "update_study_context", update)
    result = _use(service)
    authorization = result["session"]["data_source_authorization"]
    study = study_contexts.get_context("study-offer")
    assert owner_calls == [{"bind_source_id": "src_best"}]
    assert authorization["status"] == "confirmed"
    assert authorization["confirmation_mode"] == "select_registered_source"
    assert authorization["extraction_scope"] == "study_required"
    assert authorization["confirmed_at"]
    assert authorization["source"] == service._session_source_reference(study).model_dump(mode="json")
    assert study["data_source"]["path"] == "/synthetic/private/src_best"
    assert study["confirmations"]["extraction_completed"] is True
    assert result["session"]["binding"]["study_revision"] == study["revision"]
    assert result["resource"] is None
    assert service._stale_details(service._get_record(SESSION))["stale"] is False
    # Rebinding the conversation preserves this explicit choice rather than
    # generating a second pending consent card.
    rebound = service.rebind_session(SESSION, project_id=PROJECT)["session"]
    assert rebound["data_source_authorization"] == authorization


@pytest.mark.parametrize(("case", "code"), [
    ("unknown", "pi_data_source_not_registered"),
    ("invalid", "pi_data_source_not_registered"),
    ("other_family", "pi_data_source_database_mismatch"),
    ("not_offered", "pi_data_source_not_offered"),
    ("missing_id", "pi_source_id_required"),
    ("missing_database", "pi_database_not_supported"),
    ("unknown_database", "pi_database_not_supported"),
    ("demo_database", "pi_data_source_database_mismatch"),
    ("workspace", "pi_session_data_source_authorization_not_required"),
    ("confirmed", "pi_session_data_source_authorization_not_pending"),
    ("selecting", "pi_session_data_source_authorization_not_pending"),
    ("active_job", "study_context_active_job_conflict"),
    ("stale", "pi_session_authority_stale"),
    ("missing_path", "no_active_export"),
    ("owner_refusal", "synthetic_binding_refused"),
])
def test_each_refusal_leaves_both_stores_byte_identical(host, monkeypatch, case, code) -> None:
    service, registry = host
    source_id = {"unknown": "src_missing", "invalid": "src_invalid", "other_family": "src_other",
                 "not_offered": "src_small", "missing_id": ""}.get(case, "src_best")
    database = {"missing_database": "", "unknown_database": "unknown_database",
                "demo_database": "eicu_demo"}.get(case, "eicu")
    record = service._get_record(SESSION)
    if case == "workspace":
        record.agent_mode = "workspace"
    elif case in {"confirmed", "selecting"}:
        record.data_source_authorization = PiSessionDataSourceAuthorization(
            status="confirmed" if case == "confirmed" else "selection_in_progress")
    elif case in {"active_job", "stale"}:
        study = study_contexts.get_context("study-offer")
        study_contexts.upsert_context({"id": study["id"],
            **({"active_job_id": "job-synthetic"} if case == "active_job" else {"title": "Changed"})},
            expected_revision=study["revision"], lifecycle_write=True)
    elif case == "missing_path":
        registry["sources"][0]["path"] = ""
    elif case == "owner_refusal":
        monkeypatch.setattr(study_context_update, "update_study_context", lambda *args, **kwargs: {
            "status": "blocked", "code": code, "summary": "Synthetic refusal", "details": {}})
    service._save_record(record)
    before = _bytes(service)
    with pytest.raises(PiCopilotError) as error:
        _use(service, source_id, database)
    assert error.value.code == code
    assert _bytes(service) == before


@pytest.mark.parametrize("status", ["confirmed", "selection_in_progress", "not_required", "legacy_confirmed"])
def test_offer_requires_pending_research_authorization(host, status) -> None:
    service, _ = host
    record = service._get_record(SESSION)
    record.data_source_authorization.status = status
    service._save_record(record)
    before = _bytes(service)
    with pytest.raises(PiCopilotError) as error:
        _offer(service)
    assert error.value.code == "pi_session_data_source_authorization_not_pending"
    assert _bytes(service) == before


@pytest.mark.parametrize("database", ["", "unknown"])
def test_offer_rejects_missing_or_unknown_database(host, database) -> None:
    service, _ = host
    before = _bytes(service)
    with pytest.raises(PiCopilotError) as error:
        _offer(service, database)
    assert error.value.code == "pi_database_not_supported"
    assert _bytes(service) == before


def test_full_export_offer_excludes_a_more_complete_demo_and_exact_demo_can_be_selected(host) -> None:
    service, registry = host
    registry["sources"].append(_row("src_demo", "eicu_demo", stays=80))
    assert _offer(service)["recommended"]["source_id"] == "src_best"
    demo = _offer(service, "eicu_demo")["recommended"]
    assert demo["source_id"] == "src_demo"
    assert demo["label"] == "eICU Demo v2.0.1"
    assert demo["stays"] == 80
    selected = _use(service, "src_demo", "eicu_demo")["session"]["data_source_authorization"]["source"]
    assert selected["database"] == "eicu_demo"
    assert selected["stays"] == 80


@pytest.mark.parametrize("database", [profile.key for profile in iter_database_profiles()])
def test_authorization_request_accepts_every_registered_profile_key(database) -> None:
    request = routes.PiDataSourceAuthorizationRequest(
        project_id=PROJECT, action="use_registered_source", database=database, source_id="src_demo",
    )
    assert request.database == database


@pytest.mark.parametrize("database", ["unknown_database", "eICU", " eicu"])
def test_authorization_request_rejects_non_profile_keys_without_binding(host, database) -> None:
    service, _ = host
    before = _bytes(service)
    with pytest.raises(ValidationError) as caught:
        routes.PiDataSourceAuthorizationRequest(
            project_id=PROJECT, action="use_registered_source", database=database, source_id="src_best",
        )
    assert caught.value.errors()[0]["loc"] == ("database",)
    assert _bytes(service) == before


@pytest.mark.parametrize("hosted", [False, True])
def test_demo_authorization_request_binds_through_the_route_owner(host, monkeypatch, hosted) -> None:
    service, registry = host
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    registry["sources"].append(_row("src_demo", "eicu_demo", stays=12))
    request = routes.PiDataSourceAuthorizationRequest(
        project_id=PROJECT, action="use_registered_source", database="eicu_demo", source_id="src_demo",
    )
    result = routes.post_pi_copilot_data_source_authorization(SESSION, request)
    authorization = result["session"]["data_source_authorization"]
    assert authorization["status"] == "confirmed"
    assert authorization["source"]["database"] == "eicu_demo"
    assert authorization["source"]["stays"] == 12
    saved = study_contexts.get_context("study-offer")["data_source"]
    assert saved["database"] == "eicu_demo"
    assert saved["path"] == "/synthetic/private/src_demo"
    assert service._get_record(SESSION).data_source_authorization.source.database == "eicu_demo"


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize(("database", "source_id"), [("eicu", "src_demo"), ("eicu_demo", "src_best")])
def test_request_database_must_match_the_registered_source_exactly(
    host, monkeypatch, hosted, database, source_id,
) -> None:
    from fastapi import HTTPException

    service, registry = host
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    registry["sources"].append(_row("src_demo", "eicu_demo"))
    request = routes.PiDataSourceAuthorizationRequest(
        project_id=PROJECT, action="use_registered_source", database=database, source_id=source_id,
    )
    before = _bytes(service)
    with pytest.raises(HTTPException) as caught:
        routes.post_pi_copilot_data_source_authorization(SESSION, request)
    assert caught.value.status_code == 409
    assert caught.value.detail["error"] == "pi_data_source_database_mismatch"
    assert _bytes(service) == before


@pytest.mark.parametrize("alias_owner", ["request", "registry"])
def test_registered_binding_never_substitutes_a_database_alias_for_the_exact_id(host, alias_owner) -> None:
    service, registry = host
    if alias_owner == "registry":
        registry["sources"][0]["database"] = "eICU"
    before = _bytes(service)
    with pytest.raises(PiCopilotError) as caught:
        _use(service, database="eICU" if alias_owner == "request" else "eicu")
    assert caught.value.code == "pi_data_source_database_mismatch"
    assert _bytes(service) == before


def test_workspace_session_cannot_read_a_research_offer(host) -> None:
    service, _ = host
    record = service._get_record(SESSION)
    record.agent_mode = "workspace"
    service._save_record(record)
    before = _bytes(service)
    with pytest.raises(PiCopilotError) as error:
        _offer(service)
    assert error.value.code == "pi_session_data_source_authorization_not_required"
    assert _bytes(service) == before


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_offer_and_authorization_routes_keep_typed_errors_and_project_scope(host, monkeypatch, hosted, database) -> None:
    service, registry = host
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    registry["sources"][0]["database"] = database
    client = TestClient(app)
    url = f"/api/copilot/pi/sessions/{SESSION}/registered-source-offer"
    response = client.get(url, params={"project_id": PROJECT, "database": database})
    assert response.status_code == 200
    assert response.json() == _offer(service, database)
    before = _bytes(service)
    mismatch = client.get(url, params={"project_id": "another-project", "database": database})
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["error"] == "pi_session_project_mismatch"
    invalid = client.post(f"/api/copilot/pi/sessions/{SESSION}/data-source-authorization", json={
        "project_id": PROJECT, "action": "use_registered_source", "database": database, "source_id": "missing"})
    assert invalid.status_code == 409
    assert invalid.json()["detail"]["error"] == "pi_data_source_not_registered"
    unknown = client.post(f"/api/copilot/pi/sessions/{SESSION}/data-source-authorization", json={
        "project_id": PROJECT, "action": "use_registered_source", "database": "unknown_database", "source_id": "src_best"})
    assert unknown.status_code == 422
    assert unknown.json()["detail"][0]["loc"] == ["body", "database"]
    wrong_database = client.post(f"/api/copilot/pi/sessions/{SESSION}/data-source-authorization", json={
        "project_id": PROJECT, "action": "use_registered_source", "database": database, "source_id": "src_other"})
    assert wrong_database.status_code == 409
    assert wrong_database.json()["detail"]["error"] == "pi_data_source_database_mismatch"
    assert _bytes(service) == before
    used = client.post(f"/api/copilot/pi/sessions/{SESSION}/data-source-authorization", json={
        "project_id": PROJECT, "action": "use_registered_source", "database": database, "source_id": "src_best"})
    assert used.status_code == 200
    assert used.json()["session"]["data_source_authorization"]["confirmation_mode"] == "select_registered_source"
    assert used.json()["session"]["data_source_authorization"]["source"]["database"] == database


def test_one_step_choice_and_study_required_consent_have_the_same_downstream_state(host, monkeypatch) -> None:
    service, _ = host
    registered = _use(service)["session"]
    study = study_contexts.get_context("study-offer")
    service._save_record(PiSessionRecord(
        session_id="pi_reuse", project_id=PROJECT, binding=service._binding_for_context(study),
        data_source_authorization=service._new_session_data_authorization(study, agent_mode="research"),
    ))
    reused = service.authorize_data_source(
        "pi_reuse", project_id=PROJECT, action="use_study_required_data")["session"]
    assert registered["binding"] == reused["binding"]
    a, b = registered["data_source_authorization"], reused["data_source_authorization"]
    assert a.pop("confirmation_mode") == "select_registered_source"
    assert b.pop("confirmation_mode") == "reuse_project_source"
    assert a.pop("confirmed_at") and b.pop("confirmed_at")
    assert a == b
    assert a["status"] == "confirmed" and a["extraction_scope"] == "study_required"

    # No patient reads or execution: capture the governed submission boundary.
    from easyicu.webserver.routes import agent

    submissions = []
    monkeypatch.setattr(agent, "submit_agent_run", lambda request: submissions.append(request) or {"job_id": "job-test"})
    for session_id in (SESSION, "pi_reuse"):
        result = tool_module.execute_tool("easyicu_run", {"run_type": "preflight"}, ToolExecutionContext(
            session=service._get_record(session_id), allowed_actions={"run"}))
        assert result["status"] == "ok"
    assert submissions[0] == submissions[1]
    assert submissions[0]["path"] == study["data_source"]["path"]
    assert submissions[0]["study_context_id"] == study["id"]


def test_choice_permits_candidate_planning_without_approving_execution(host, monkeypatch) -> None:
    service, _ = host
    _use(service)
    record = service._get_record(SESSION)
    record.external_llm_opt_in = True
    monkeypatch.setattr(tool_module, "_account_environment_for_research_provider", lambda context: ({}, None))
    monkeypatch.setattr(tool_module, "_workflow_snapshot", lambda *args, **kwargs: {
        "next_action_code": "provider_ready_to_generate_plan"})
    submitted = []

    def submit(request, *, authorize, **kwargs):
        authorize()
        submitted.append(request)
        return research_run_submission.ResearchRunSubmissionReceipt(
            job_id="job-plan", kind="agent-run", status="queued", study_context_id="study-offer",
            study_context_revision=record.binding.study_revision, budget_mode="planner_canary", planner_start_mode="auto")

    monkeypatch.setattr(research_run_submission, "submit_research_run", submit)
    result = tool_module.execute_tool("easyicu_run", {"run_type": "full"}, ToolExecutionContext(
        session=record, user_message="Generate the analysis plan", allowed_actions={"provider_run"}))
    assert result["status"] == "ok", result
    assert len(submitted) == 1
    assert submitted[0].intent == "candidate_plan"
    assert submitted[0].study_context_id == "study-offer"
