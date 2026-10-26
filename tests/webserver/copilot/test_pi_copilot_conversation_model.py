"""Conversation-only model changes, with synthetic transports and ASGI requests."""

from copy import deepcopy
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import provider_config
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, ResearchProviderBinding
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as routes
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway, study_state,  # noqa: F401
)


class ModelGateway(FakeGateway):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.streaming = False
        self.fail_switch = False

    def apply_provider_config(self, config):
        self.environ = config.as_environment()

    def request(self, method, params, **kwargs):
        if method == "session.set_model" and self.fail_switch:
            raise PiCopilotError("pi_model_unavailable", "Unavailable", status_code=409)
        result = super().request(method, params, **kwargs)
        sid = params.get("session_id")
        if method in {"session.create", "session.set_model"}:
            self.models[sid] = params.get("model") or self.environ["EASYICU_PI_MODEL"]
        result["model"] = {"provider": "fixture", "id": self.models.get(sid, "model-a")}
        result["streaming"] = self.streaming
        return result


@pytest.fixture
def model_session(tmp_path, monkeypatch, request):
    current_study = request.getfixturevalue("study_state")
    for key in provider_config._CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    current_study["data_source"] = {}
    store = provider_config.PiProviderConfigStore(
        config_path=tmp_path / "connection", receipt_path=tmp_path / "verified.json",
    )
    calls = []
    def transport(method, *args):
        calls.append((method, *args))
        return 200, ({"data": [{"id": "model-a"}, {"id": "model-b"}]}
                     if method == "GET" else {"choices": [{"message": {"content": "READY"}}]})
    monkeypatch.setattr(provider_config, "_default_verification_transport", transport)
    monkeypatch.setattr(provider_config, "validate_credential_endpoint", lambda url: None)
    connection = {"provider": "fixture", "base_url": "http://localhost:8317/v1",
                  **getattr(request, "param", {})}
    config, _ = store.verify_and_save(
        **connection, model="model-a",
        api_key="synthetic-key", api_transport="openai-completions", verifier=transport,
    )
    gateway = ModelGateway()
    gateway.environ = config.as_environment()
    gateway.verification_calls = calls
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway, provider_store=store)
    session = service.create_session(
        project_id="project-model", external_llm_opt_in=True,
        research_provider=service.verified_api_research_provider_binding(),
    )["session"]
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    return TestClient(app, base_url="http://127.0.0.1:8765"), service, gateway, store, session, current_study


def switch(fixture, **changes):
    client, _, _, _, session, _ = fixture
    return client.post(f"/api/copilot/pi/sessions/{session['session_id']}/model", json={
        "project_id": "project-model", "model": "model-b", **changes,
    })


def test_switch_changes_future_conversation_only_and_records_system_event(model_session):
    client, service, gateway, store, session, study = model_session
    sid = session["session_id"]
    before = deepcopy(study)
    provider = service._get_record(sid).research_provider
    settings_before = store.config_path.read_bytes(), store.receipt_path.read_bytes()
    response = switch(model_session)
    assert response.status_code == 200, response.text
    updated = response.json()["session"]
    assert updated["model_connection"]["model"] == updated["model"]["id"] == "model-b"
    assert updated["research_provider"] == provider.public_projection()
    assert service._get_record(sid).research_provider == provider
    assert study == before
    assert (store.config_path.read_bytes(), store.receipt_path.read_bytes()) == settings_before
    assert gateway.models[sid] == "model-b"
    event = updated["conversation_replay"]["turns"][-1]
    assert event["kind"] == "host_action"
    assert event["action_code"] == "conversation_model_changed"
    assert event["allowed_actions"] == []
    assert "user_message" not in event
    assert event["events"][0] == {
        "type": "model_changed", "role": "system", "previous_model": "model-a", "model": "model-b",
    }
    memory = service.list_sessions(project_id="project-model")["sessions"]
    assert memory[0]["model_connection"]["model"] == "model-b"
    workflow = client.get("/api/copilot/pi/projects/project-model/workflow")
    assert workflow.status_code == 200
    assert workflow.json()["connection_state"][sid]["model"] == "model-b"
    reopened = client.get(f"/api/copilot/pi/sessions/{sid}", params={"project_id": "project-model"})
    assert reopened.status_code == 200
    assert reopened.json()["session"]["model_connection"]["model"] == "model-b"
    assert "synthetic-key" not in response.text


@pytest.mark.parametrize("case,code,status", [
    ("busy", "pi_session_busy", 409),
    ("streaming", "pi_session_busy", 409),
    ("queued", "pi_session_busy", 409),
    ("admitted", "pi_session_busy", 409),
    ("unknown", "pi_session_model_unavailable", 422),
    ("project", "pi_session_project_mismatch", 409),
    ("gateway", "pi_model_unavailable", 409),
])
def test_refusal_has_no_binding_replay_or_global_mutation(model_session, case, code, status):
    _, service, gateway, store, session, _ = model_session
    sid = session["session_id"]
    before = service._get_record(sid).model_dump(), store.config_path.read_bytes()
    if case == "busy":
        service._busy_sessions.add(sid)
    if case == "streaming":
        gateway.streaming = True
    if case == "queued":
        service._dispatching_followups[sid] = object()
    if case == "admitted":
        record = service._get_record(sid)
        record.active_message_job_id = "synthetic-admitted-turn"
        service._save_record(record)
        before = record.model_dump(), store.config_path.read_bytes()
    gateway.fail_switch = case == "gateway"
    changes = {"model": "not-discovered"} if case == "unknown" else {"project_id": "other"} if case == "project" else {}
    response = switch(model_session, **changes)
    assert response.status_code == status, response.text
    assert response.json()["detail"]["error"] == code
    assert (service._get_record(sid).model_dump(), store.config_path.read_bytes()) == before
    assert not service.replay_store.snapshot(session_id=sid, project_id="project-model")["turns"]


@pytest.mark.parametrize("database", ["miiv", "eicu"])
@pytest.mark.parametrize("handling,accepted", [("unspecified", False), ("zero_retention_declared", True), ("on_device", True)])
def test_credentialed_source_uses_existing_clearance_owner(model_session, monkeypatch, database, handling, accepted):
    _, _, _, _, _, study = model_session
    study["data_source"] = {"database": database, "path": "/synthetic/credentialed-source"}
    monkeypatch.setenv("EASYICU_PI_DATA_HANDLING", handling)
    response = switch(model_session)
    assert response.status_code == (200 if accepted else 403), response.text
    if not accepted:
        assert response.json()["detail"]["error"] == "provider_not_cleared_for_credentialed_data"


def test_source_removal_does_not_clear_credentialed_history(model_session):
    _, service, _, _, session, _ = model_session
    record = service._get_record(session["session_id"])
    record.credentialed_data_in_history = True
    service._save_record(record)
    response = switch(model_session)
    assert response.status_code == 403


def test_original_model_declaration_is_not_transferred(model_session):
    _, _, _, store, _, study = model_session
    store.declare_data_handling(
        value="zero_retention_declared", connection={"provider": "fixture", "base_url": "http://localhost:8317/v1", "model": "model-a"},
        notice_key="physionet_llm_data_handling_declaration", notice_version="1", channel="settings_ui",
    )
    before = store.declaration_path.read_bytes()
    study["data_source"] = {"database": "miiv", "path": "/synthetic/source"}
    assert switch(model_session).status_code == 403
    assert store.declaration_path.read_bytes() == before


def test_catalog_is_scoped_to_the_saved_connection(model_session, monkeypatch):
    client, _, _, _, session, _ = model_session
    url = f"/api/copilot/pi/sessions/{session['session_id']}/models"
    response = client.get(url, params={"project_id": "project-model"})
    assert response.status_code == 200
    assert response.json()["models"] == ["model-a", "model-b"]
    monkeypatch.setenv("EASYICU_PI_BASE_URL", "http://localhost:8318/v1")
    assert client.get(url, params={"project_id": "project-model"}).json()["detail"]["error"] == "pi_session_connection_changed"
    assert switch(model_session).status_code == 409


def test_reopen_after_sidecar_restart_uses_saved_model(model_session):
    _, service, gateway, store, session, _ = model_session
    sid = session["session_id"]
    assert switch(model_session).status_code == 200
    original = gateway.request

    def forgetful(method, params, **kwargs):
        if method == "session.state":
            raise PiCopilotError("pi_session_not_open", "Synthetic restart", status_code=409)
        return original(method, params, **kwargs)

    gateway.request = forgetful
    restarted = PiCopilotService(store_path=service.store_path, gateway=gateway, provider_store=store)
    opened = restarted._ensure_open(restarted._get_record(sid))
    assert opened["model"]["id"] == "model-b"
    assert [p for m, p, _ in gateway.calls if m == "session.create"][-1]["model"] == "model-b"


def test_stale_session_writes_cannot_erase_the_model_choice(model_session):
    _, service, _, _, session, _ = model_session
    stale = service._get_record(session["session_id"])
    assert switch(model_session).status_code == 200
    stale.title = "Synthetic title"
    service._save_record(stale)
    assert service._get_record(stale.session_id).conversation_provider.model == "model-b"


def test_noop_has_no_system_event_and_another_conversation_keeps_its_model(model_session):
    _, service, gateway, _, session, _ = model_session
    other = service.create_session(project_id="project-model", external_llm_opt_in=True,
                                   research_provider=service.verified_api_research_provider_binding())["session"]
    assert switch(model_session, model="model-a").status_code == 200
    assert not service.replay_store.snapshot(session_id=session["session_id"], project_id="project-model")["turns"]
    assert switch(model_session).status_code == 200
    assert gateway.models[other["session_id"]] == "model-a"


def test_account_catalogue_is_resolved_from_the_saved_account(model_session, monkeypatch):
    from easyicu.webserver import codex_account_sessions

    client, service, gateway, _, session, _ = model_session
    binding = ResearchProviderBinding(provider="codex", credential_source="codex_user_auth",
                                      authentication_mode="chatgpt_account", model="model-a",
                                      account_session_sha256="a" * 64)
    record = service._get_record(session["session_id"]).model_copy(update={"research_provider": binding})
    service._save_record(record)
    calls = []

    def models(account):
        calls.append(account)
        return ["model-a", "model-b"]

    monkeypatch.setattr(codex_account_sessions, "discovered_models_for_binding", models)
    monkeypatch.setattr(service.codex_gateway_pool, "gateway_for", lambda *a, **kw: gateway)
    response = client.get(f"/api/copilot/pi/sessions/{record.session_id}/models", params={"project_id": "project-model"})
    assert response.status_code == 200
    assert switch(model_session).status_code == 200
    assert calls == ["a" * 64, "a" * 64]
    assert service._get_record(record.session_id).research_provider == binding


@pytest.mark.parametrize("legacy", ["missing_pin", "v1"])
def test_unpinned_legacy_connection_is_not_guessed(model_session, legacy):
    client, service, _, _, session, _ = model_session
    record = service._get_record(session["session_id"])
    if legacy == "missing_pin":
        record.conversation_connection_sha256 = None
    else:
        record.schema_version = "easyicu.pi-copilot-session/1"
    service._save_record(record, model_change=True)
    for response in (switch(model_session), client.get(
        f"/api/copilot/pi/sessions/{record.session_id}/models", params={"project_id": "project-model"},
    )):
        assert response.status_code == 409
        assert response.json()["detail"]["error"] == "pi_session_connection_unpinned"


def test_admitted_research_run_keeps_its_provider_while_conversation_changes(model_session):
    _, service, _, _, session, study = model_session
    # Synthetic reviewed plan and admitted run. Switching is not a study update.
    study["active_job_id"] = "synthetic-admitted-research-job"
    study["reviewed_plan"] = {"provider": "openai", "model": "model-a", "approved": True}
    before = deepcopy(study)
    assert switch(model_session).status_code == 200
    assert study == before
    assert service._get_record(session["session_id"]).research_provider.model == "model-a"


def test_demo_clearance_is_delegated_to_the_source_owner(model_session, monkeypatch):
    from easyicu.research_agent.providers.client_trust import public_demo_source_scope
    from easyicu.webserver import demo_source_storage

    _, _, _, _, _, study = model_session
    study["data_source"] = {"database": "miiv", "path": "/synthetic/public-demo"}
    # The source owner supplies release verification; no real dataset is read.
    monkeypatch.setattr(demo_source_storage, "public_demo_egress_scope", lambda _: public_demo_source_scope("miiv"))
    assert switch(model_session).status_code == 200


def test_a_turn_prepared_before_the_model_switch_is_refused(model_session, monkeypatch):
    _, service, gateway, _, session, _ = model_session
    original = gateway.request
    switched = False

    def interleaved(method, params, **kwargs):
        nonlocal switched
        if method == "session.state" and not switched:
            switched = True
            assert switch(model_session).status_code == 200
        return original(method, params, **kwargs)

    monkeypatch.setattr(gateway, "request", interleaved)
    with pytest.raises(PiCopilotError) as caught:
        service.send_message(session["session_id"], project_id="project-model", message="Summarize the study setup.")
    assert caught.value.code == "pi_session_model_changed"
    assert not [m for m, _, _ in gateway.calls if m == "session.prompt"]


@pytest.mark.parametrize("key,value", [
    ("EASYICU_PI_API_KEY", "synthetic-other-key"),
    ("EASYICU_PI_PROVIDER", "different-provider"),
    ("EASYICU_PI_API", "openai-responses"),
])
def test_saved_credential_provider_and_transport_drift_are_refused(model_session, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    response = switch(model_session)
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "pi_session_connection_changed"


def test_running_gateway_drift_is_refused_even_when_saved_settings_match(model_session):
    _, _, gateway, _, _, _ = model_session
    gateway.environ["EASYICU_PI_BASE_URL"] = "http://localhost:8319/v1"
    assert switch(model_session).status_code == 409


def test_account_identity_lookup_does_not_lapse_the_api_declaration(model_session):
    _, service, _, store, _, _ = model_session
    store.declare_data_handling(
        value="zero_retention_declared", connection={"provider": "fixture", "base_url": "http://localhost:8317/v1", "model": "model-a"},
        notice_key="physionet_llm_data_handling_declaration", notice_version="1", channel="settings_ui",
    )
    before = store.declaration_path.read_bytes()
    assert service._conversation_connection_sha256(object()) is None
    assert store.declaration_path.read_bytes() == before


def test_switching_back_uses_the_original_models_current_declaration(model_session):
    _, _, _, store, _, study = model_session
    assert switch(model_session).status_code == 200
    store.declare_data_handling(
        value="zero_retention_declared", connection={"provider": "fixture", "base_url": "http://localhost:8317/v1", "model": "model-a"},
        notice_key="physionet_llm_data_handling_declaration", notice_version="1", channel="settings_ui",
    )
    study["data_source"] = {"database": "miiv", "path": "/synthetic/source"}
    assert switch(model_session, model="model-a").status_code == 200


# Real preset coordinates with synthetic keys, catalogs and transports only.
# Endpoint security has its own tests; the fixture never resolves these hosts.
_PRESET_CONNECTIONS = [
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("deepseek", "https://api.deepseek.com/v1"),
    ("dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("zhipu", "https://open.bigmodel.cn/api/paas/v4"),
    ("moonshot", "https://api.moonshot.cn/v1"),
    ("volcengine-ark", "https://ark.cn-beijing.volces.com/api/v3"),
    ("siliconflow", "https://api.siliconflow.cn/v1"),
    ("minimax", "https://api.minimaxi.com/v1"),
    ("azure-openai", "https://synthetic.openai.azure.com/openai/v1"),
    ("amazon-bedrock", "https://bedrock-mantle.synthetic.api.aws/v1"),
    ("ollama", "http://127.0.0.1:11434/v1"),
    ("lmstudio", "http://127.0.0.1:1234/v1"),
]


@pytest.mark.parametrize("model_session", [
    {"provider": provider, "base_url": endpoint} for provider, endpoint in _PRESET_CONNECTIONS
], indirect=True, ids=[provider for provider, _ in _PRESET_CONNECTIONS])
@pytest.mark.parametrize("drift", [None, "PROVIDER", "BASE_URL", "API", "API_KEY"])
def test_presets_switch_only_within_the_pinned_connection(model_session, monkeypatch, drift):
    client, service, gateway, store, session, _ = model_session
    sid = session["session_id"]
    before = service._get_record(sid).model_dump(), store.config_path.read_bytes()
    gateway.verification_calls.clear()
    if drift:
        changed = {"PROVIDER": "another-provider", "BASE_URL": "http://localhost:9999/v1",
                   "API": "openai-responses", "API_KEY": "synthetic-replacement-key"}
        monkeypatch.setenv(f"EASYICU_PI_{drift}", changed[drift])
    listed = client.get(f"/api/copilot/pi/sessions/{sid}/models", params={"project_id": "project-model"})
    switched = switch(model_session)
    if drift:
        for response in (listed, switched):
            assert response.status_code == 409
            assert response.json()["detail"]["error"] == "pi_session_connection_changed"
        assert gateway.verification_calls == []
        assert (service._get_record(sid).model_dump(), store.config_path.read_bytes()) == before
    else:
        assert listed.status_code == switched.status_code == 200
        assert switched.json()["session"]["model_connection"]["model"] == "model-b"
        assert service._get_record(sid).research_provider.model == "model-a"
        assert store.config_path.read_bytes() == before[1]
        assert len(gateway.verification_calls) == 2
        assert all(call[:2] == ("GET", gateway.environ["EASYICU_PI_BASE_URL"] + "/models")
                   and call[2]["Authorization"] == "Bearer synthetic-key"
                   for call in gateway.verification_calls)
        for response in (listed, switched):
            assert "synthetic-key" not in response.text
            assert before[0]["conversation_connection_sha256"] not in response.text


def test_connection_identity_excludes_model_and_policy_but_keeps_each_wire_component(model_session):
    _, _, _, store, _, _ = model_session
    config = store.resolved_config()
    fingerprint = config.conversation_connection_sha256()
    assert replace(config, model="model-b", data_handling="zero_retention_declared").conversation_connection_sha256() == fingerprint
    assert replace(config, inference_location="on_device").conversation_connection_sha256() == fingerprint
    for changes in ({"provider": "other"}, {"base_url": "http://localhost:9999/v1"},
                    {"api_transport": "openai-responses"}, {"api_key": "synthetic-other"}):
        assert replace(config, **changes).conversation_connection_sha256() != fingerprint


@pytest.mark.parametrize("operation", ["catalog", "switch"])
def test_both_model_routes_check_project_ownership_before_discovery(model_session, operation):
    client, _, gateway, _, session, _ = model_session
    gateway.verification_calls.clear()
    response = (switch(model_session, project_id="another-project") if operation == "switch" else
                client.get(f"/api/copilot/pi/sessions/{session['session_id']}/models",
                           params={"project_id": "another-project"}))
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "pi_session_project_mismatch"
    assert gateway.verification_calls == []


def test_openrouter_login_pins_the_exchanged_key_after_the_login_is_consumed(model_session, monkeypatch):
    from easyicu.webserver.pi_copilot import openrouter_login

    client, service, gateway, store, _, _ = model_session
    monkeypatch.setattr(openrouter_login, "_STORE", openrouter_login.OpenRouterLoginStore())
    monkeypatch.setattr(openrouter_login, "_default_exchange", lambda *_: (200, {"key": "synthetic-openrouter-key"}))
    monkeypatch.setattr(service, "runtime_status", lambda: {"runtime": {"status": "ready"}})
    assert client.post("/api/copilot/pi/provider-config/openrouter/login", json={"enable_ai": True}).status_code == 200
    landed = client.get(openrouter_login.CALLBACK_PATH + "?code=synthetic-code", follow_redirects=False)
    assert landed.status_code == 303
    body = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
            "model": "model-a", "api_transport": "openai-completions", "api_key": "",
            "enable_ai": True, "credential_source": "openrouter_login"}
    saved = client.post("/api/copilot/pi/provider-config", json=body)
    assert saved.status_code == 200, saved.text
    assert not client.cookies.get(openrouter_login.COOKIE_NAME)
    created = client.post("/api/copilot/pi/sessions", json={"project_id": "project-model",
                          "external_llm_opt_in": True, "research_provider": "api"})
    assert created.status_code == 200, created.text
    session = created.json()["session"]
    pin = service._get_record(session["session_id"]).conversation_connection_sha256
    assert pin == store.resolved_config().conversation_connection_sha256()
    fixture = (client, service, gateway, store, session, {})
    assert switch(fixture).status_code == 200
    # An unrelated/new browser sign-in does not repin the saved conversation.
    client.cookies.set(openrouter_login.COOKIE_NAME, "synthetic-unrelated-login")
    assert switch(fixture, model="model-a").status_code == 200
    assert service._get_record(session["session_id"]).conversation_connection_sha256 == pin
    gateway.environ["EASYICU_PI_API_KEY"] = "synthetic-rotated-key"
    refused = switch(fixture)
    assert refused.status_code == 409
    assert refused.json()["detail"]["error"] == "pi_session_connection_changed"


@pytest.mark.parametrize("status", [404, 405])
def test_an_unlisted_model_still_connects_and_opens_a_pinned_conversation(model_session, monkeypatch, status):
    client, service, gateway, store, _, _ = model_session
    probes = []
    def transport(method, url, headers, body, timeout):
        probes.append((method, body))
        return (status, {}) if method == "GET" else (200, {"choices": [{"message": {"content": "READY"}}]})
    monkeypatch.setattr(provider_config, "_default_verification_transport", transport)
    monkeypatch.setattr(service, "runtime_status", lambda: {"runtime": {"status": "ready"}})
    body = {"provider": "zhipu", "base_url": "http://localhost:9100/v1", "model": "unlisted-model",
            "api_transport": "openai-completions", "api_key": "synthetic-unlisted-key", "enable_ai": True}
    saved = client.post("/api/copilot/pi/provider-config", json=body)
    assert saved.status_code == 200, saved.text
    assert probes[-1][1]["model"] == "unlisted-model"
    created = client.post("/api/copilot/pi/sessions", json={"project_id": "project-model",
                          "external_llm_opt_in": True, "research_provider": "api"})
    assert created.status_code == 200, created.text
    sid = created.json()["session"]["session_id"]
    assert created.json()["session"]["model_connection"]["model"] == "unlisted-model"
    assert gateway.models[sid] == "unlisted-model"
    assert service._get_record(sid).conversation_connection_sha256 == store.resolved_config().conversation_connection_sha256()
    before = service._get_record(sid).model_dump()
    # Discovery does not guess alternatives or probe inference during selection.
    probes.clear()
    listed = client.get(f"/api/copilot/pi/sessions/{sid}/models", params={"project_id": "project-model"})
    assert listed.json()["detail"]["error"] == "pi_provider_model_list_unavailable"
    assert probes == [("GET", None)]
    reopened = client.get(f"/api/copilot/pi/sessions/{sid}", params={"project_id": "project-model"})
    assert reopened.status_code == 200
    assert reopened.json()["session"]["model"]["id"] == "unlisted-model"
    assert service._get_record(sid).model_dump() == before
