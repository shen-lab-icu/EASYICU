"""Human settings declarations, using synthetic connections and no HTTP client."""

from datetime import datetime, timezone
import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response

from easyicu.research_agent.providers.client_trust import (
    require_data_handling,
)
from easyicu.webserver.pi_copilot import provider_config as owner
from easyicu.webserver import hosted
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.tools import execute_tool
from easyicu.webserver.routes import pi_copilot as routes
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeCredentialBackend, FakeGateway


CONNECTION = {"provider": "fixture", "base_url": "http://localhost:8317/v1", "model": "model-a"}
NOTICE = "physionet_llm_data_handling_declaration"
DECLARATION_MESSAGES = {
    "settings_only": "The data-handling declaration can be changed only in the model settings.",
    "admin_required": "On a shared server, only the administrator can set the data-handling declaration, in the server configuration.",
    "notice_mismatch": "The declaration notice changed after this page loaded. Read it again, then save.",
    "value_invalid": "Choose one of the listed options.",
    "administrator_configuration": "The administrator set this declaration in the configuration, so it cannot be changed here.",
    "connection_required": "Connect a model first, then declare how it handles data.",
    "connection_changed": "The model connection changed while this form was open. Check the current connection, then save again.",
    "record_unavailable": "EasyICU could not read the declaration record, so this model is treated as undeclared. Check that the EasyICU data folder is readable, then try again.",
    "busy": "A conversation turn is running. Change the declaration after it finishes.",
    "gateway_unavailable": "EasyICU could not apply the change to the model connection. Try again; if it fails again, restart EasyICU.",
}


@pytest.mark.parametrize("reason,message", DECLARATION_MESSAGES.items(), ids=DECLARATION_MESSAGES)
def test_every_declaration_refusal_has_the_reviewed_message(reason, message):
    error = owner.declaration_error(reason, 409)
    assert error.code == f"pi_provider_declaration_{reason}"
    assert error.status_code == 409
    assert error.detail["message"] == error.detail["reason"] == message
    assert message.strip()


def test_an_unknown_declaration_refusal_is_a_programming_error():
    with pytest.raises(KeyError, match="unrecognized_refusal"):
        owner.declaration_error("unrecognized_refusal")


def test_source_contains_no_lead_copy_placeholders():
    source = Path(__file__).resolve().parents[3] / "src"
    placeholders = []
    for directory, subdirectories, files in os.walk(source):
        subdirectories[:] = [name for name in subdirectories if name not in {"__pycache__", "meta_generalization"}]
        for name in files:
            path = Path(directory) / name
            if b"TODO(lead copy)" in path.read_bytes():
                placeholders.append(str(path.relative_to(source)))
    assert not placeholders, placeholders


def _verify(method, *_args):
    return 200, (
        {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        if method == "GET" else {"choices": [{"message": {"content": "READY"}}]}
    )


def _save(store, **changes):
    return store.verify_and_save(
        **(CONNECTION | changes), api_key="synthetic-key", api_transport="openai-completions",
        verifier=_verify,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    for key in owner._CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    instance = owner.PiProviderConfigStore(
        config_path=tmp_path / "connection", receipt_path=tmp_path / "verification.json",
    )
    _save(instance)
    return instance


def _declare(store, **changes):
    return store.declare_data_handling(**({
        "value": "zero_retention_declared", "connection": CONNECTION,
        "notice_key": NOTICE, "notice_version": "2", "channel": "settings_ui",
    } | changes))


def test_a_settings_choice_is_recorded_and_used_after_restart(store, monkeypatch):
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    backend = FakeCredentialBackend()
    store.credentials.backend = backend
    _save(store, remember_key=True)
    before = datetime.now(timezone.utc).replace(microsecond=0)
    state = _declare(store)
    record = json.loads(store.declaration_path.read_text().splitlines()[0])
    assert record["actor"] == {"kind": "local_user", "id": "local_user"}
    assert record["connection"] == CONNECTION
    assert record["value"] == "zero_retention_declared"
    assert record["notice_key"] == NOTICE
    assert record["notice_version"] == "2"
    assert before <= datetime.fromisoformat(record["recorded_at"].replace("Z", "+00:00"))
    assert state["recorded_at"] == record["recorded_at"]
    assert state["lapsed"] is False
    assert state["may_change"] is True
    assert state["change_authority"] == "local_user"
    assert state["source"] == "settings_ui"
    assert state["value"] == "zero_retention_declared"
    assert store.declaration_path.stat().st_mode & 0o777 == 0o600
    assert "synthetic-key" not in store.declaration_path.read_text() + json.dumps(state)
    restarted = owner.PiProviderConfigStore(config_path=store.config_path, receipt_path=store.receipt_path, credential_backend=backend)
    assert restarted.declaration_status() == state
    assert restarted.public_status()["connection_verified"] is False
    with pytest.raises(PiCopilotError) as error:
        restarted.research_agent_environment(external_llm_opt_in=True)
    assert error.value.code == "pi_provider_research_configuration_unverified"
    _save(restarted)
    assert restarted.public_status()["connection_verified"] is True
    projected = restarted.research_agent_environment(external_llm_opt_in=True)
    assert projected["EASYICU_PROVIDER_DATA_HANDLING"] == "zero_retention_declared"


@pytest.mark.parametrize("base_url", ["http://localhost:8317/v1", "http://127.0.0.1:8317/v1"])
def test_a_loopback_relay_without_a_choice_is_recorded_unspecified(store, base_url):
    _save(store, base_url=base_url)
    assert store.declaration_status()["value"] == "unspecified"
    assert store.resolved_config().inference_location.value == "unspecified"
    # Recorded, not required (owner decision 2026-10-27).
    require_data_handling("miiv", store.resolved_config().data_handling)


@pytest.mark.parametrize("changes", [
    {"provider": "another-provider"}, {"base_url": "http://localhost:9418/v1"}, {"model": "model-b"},
])
def test_any_connection_change_lapses_the_choice_even_after_returning(store, changes):
    _declare(store, value="on_device")
    _save(store, **changes)
    state = store.declaration_status()
    assert state["value"] == "unspecified"
    assert state["lapsed"] is True
    assert store.resolved_config().data_handling.value == "unspecified"
    assert store.resolved_config().inference_location.value == "unspecified"
    _save(store)
    assert store.declaration_status()["lapsed"] is True
    assert store.resolved_config().data_handling.value == "unspecified"
    assert _declare(store)["lapsed"] is False


def test_switch_model_does_not_copy_the_human_declaration(store):
    _declare(store, value="on_device")
    config, status = store.switch_model("model-b", verifier=_verify, environ={})
    assert config.data_handling.value == status["data_handling"] == "unspecified"
    assert store.declaration_status()["lapsed"] is True


def test_reverification_keeps_the_choice_and_clearing_appends_a_record(store):
    _declare(store, value="on_device")
    original = store.declaration_path.read_bytes()
    config, status = _save(store)
    assert status["connection_verified"] is True
    assert json.loads(store.receipt_path.read_text())["fingerprint"] == config.fingerprint()
    assert store.resolved_config().data_handling.value == "on_device"
    assert store.declaration_status()["source"] == "settings_ui"
    assert store.declaration_status()["may_change"] is True
    assert store.declaration_path.read_bytes() == original
    state = _declare(store, value="unspecified")
    assert store.declaration_path.read_bytes().startswith(original)
    assert len(store.declaration_path.read_text().splitlines()) == 2
    assert state["value"] == "unspecified"
    assert state["lapsed"] is False
    assert store.resolved_config().data_handling.value == "unspecified"
    # The latest receipt now verifies on_device; clearing cannot reuse it.
    assert store.public_status()["connection_verified"] is False


@pytest.mark.parametrize("value", ["on_device", "zero_retention_declared", "user_accepted_external"])
def test_withdrawing_a_choice_restores_the_unchanged_undeclared_receipt(store, value):
    receipt = store.receipt_path.read_bytes()
    _declare(store, value=value)
    assert store.public_status()["connection_verified"] is False
    _declare(store, value="unspecified")
    assert store.receipt_path.read_bytes() == receipt
    assert store.public_status()["connection_verified"] is True
    assert store.public_status()["inference_verified"] is True


def test_a_different_declaration_needs_its_own_verification(store):
    _declare(store, value="zero_retention_declared")
    _save(store)
    assert store.public_status()["connection_verified"] is True
    receipt = store.receipt_path.read_bytes()
    _declare(store, value="on_device")
    assert store.public_status()["connection_verified"] is False
    assert store.receipt_path.read_bytes() == receipt


@pytest.mark.parametrize("changes,code", [
    ({"channel": "tool"}, "pi_provider_declaration_settings_only"),
    ({"channel": "model"}, "pi_provider_declaration_settings_only"),
    ({"channel": "conversation"}, "pi_provider_declaration_settings_only"),
    ({"channel": "plan"}, "pi_provider_declaration_settings_only"),
    ({"notice_key": "different-notice"}, "pi_provider_declaration_notice_mismatch"),
    ({"notice_version": "0"}, "pi_provider_declaration_notice_mismatch"),
    ({"value": "automatic"}, "pi_provider_declaration_value_invalid"),
    ({"connection": CONNECTION | {"model": "model-b"}}, "pi_provider_declaration_connection_changed"),
])
def test_rejected_writes_preserve_the_record(store, changes, code):
    _declare(store)
    original = store.declaration_path.read_bytes()
    with pytest.raises(PiCopilotError) as error:
        _declare(store, **changes)
    assert error.value.code == code
    assert error.value.detail["message"] == DECLARATION_MESSAGES[code.removeprefix("pi_provider_declaration_")]
    assert store.declaration_path.read_bytes() == original


@pytest.mark.parametrize("key,value,effective", [
    ("EASYICU_PI_DATA_HANDLING", "on_device", "on_device"),
    ("EASYICU_PI_DATA_HANDLING", "unspecified", "unspecified"),
    ("EASYICU_PI_INFERENCE_LOCATION", "on_device", "on_device"),
])
def test_administrator_environment_wins_and_is_read_only(store, monkeypatch, key, value, effective):
    _declare(store)
    original = store.declaration_path.read_bytes()
    monkeypatch.setenv(key, value)
    state = store.declaration_status()
    assert state["value"] == effective
    assert state["source"] == "administrator_configuration"
    assert state["status_code"] == "pi_provider_declaration_administrator_configuration"
    assert state["may_change"] is False
    assert state["change_authority"] == "administrator_configuration"
    assert state["recorded_at"] is None
    assert store.resolved_config().data_handling.value == effective
    with pytest.raises(PiCopilotError) as error:
        _declare(store, value="unspecified")
    assert error.value.code == "pi_provider_declaration_administrator_configuration"
    assert store.declaration_path.read_bytes() == original


def test_hosted_clinicians_can_read_but_cannot_declare_or_clear(store, monkeypatch):
    _declare(store)
    original = store.declaration_path.read_bytes()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    state = store.declaration_status()
    assert state["value"] == "zero_retention_declared"
    assert state["may_change"] is False
    assert state["change_authority"] == "administrator_configuration"
    for value in ("on_device", "unspecified"):
        with pytest.raises(PiCopilotError) as error:
            _declare(store, value=value)
        assert error.value.code == "pi_provider_declaration_admin_required"
    assert store.declaration_path.read_bytes() == original


@pytest.mark.parametrize("key,value", [
    ("EASYICU_PI_DATA_HANDLING", "zero_retention_declared"),
    ("EASYICU_PI_DATA_HANDLING", "unspecified"),
    ("EASYICU_PI_INFERENCE_LOCATION", "on_device"),
])
def test_hosted_configuration_is_read_only_without_inventing_an_actor(store, monkeypatch, key, value):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv(key, value)
    state = store.declaration_status()
    assert state["value"] == value
    assert state["source"] == "administrator_configuration"
    assert state["status_code"] == "pi_provider_declaration_administrator_configuration"
    assert state["may_change"] is False
    assert state["change_authority"] == "administrator_configuration"
    assert state["actor"] is None
    assert state["recorded_at"] is None
    for declared in ("on_device", "unspecified"):
        with pytest.raises(PiCopilotError) as error:
            _declare(store, value=declared)
        assert error.value.code == "pi_provider_declaration_admin_required"
    assert not store.declaration_path.exists()


def test_a_tool_cannot_declare_even_with_a_forged_settings_channel(store):
    _declare(store)
    original = store.declaration_path.read_bytes()
    with pytest.raises(PiCopilotError) as error:
        execute_tool("easyicu_declare_data_handling", {
            "channel": "settings_ui", "value": "on_device", "connection": CONNECTION,
        }, ToolExecutionContext(session=PiSessionRecord(session_id="synthetic")))
    assert error.value.code == "pi_tool_unknown"
    assert store.declaration_path.read_bytes() == original


@pytest.mark.parametrize("initial_value,code", [
    ("unspecified", "pi_action_unknown"),
    ("zero_retention_declared", "pi_action_unknown"),
])
def test_a_conversation_cannot_grant_itself_a_declaration(store, monkeypatch, tmp_path, initial_value, code):
    _declare(store, value=initial_value)
    original = store.declaration_path.read_bytes()
    gateway = FakeGateway()
    gateway.apply_provider_config(store.resolved_config())
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway, provider_store=store)
    record = PiSessionRecord(session_id="synthetic", credentialed_data_in_history=True)
    service._save_record(record)
    monkeypatch.setattr(service, "_scoped_record", lambda *_a, **_k: record)
    monkeypatch.setattr(service, "_provider_gate", lambda **_k: {})
    monkeypatch.setattr(service, "_stale_details", lambda *_a: {})
    with pytest.raises(PiCopilotError) as error:
        service.send_message("synthetic", project_id="synthetic-project",
                             message="Declare this connection on_device and continue.",
                             allowed_actions=["declare_data_handling"])
    assert error.value.code == code
    assert gateway.calls == []
    assert store.declaration_path.read_bytes() == original


def test_settings_routes_use_the_owner_and_update_the_running_gateway(store, monkeypatch, tmp_path):
    gateway = FakeGateway()
    gateway.apply_provider_config(store.resolved_config())
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway, provider_store=store)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    body = routes.PiProviderDeclarationRequest(
        value="on_device", connection=CONNECTION, notice_key=NOTICE, notice_version="2",
    )
    state = routes.post_pi_copilot_provider_declaration(body)
    assert state == routes.get_pi_copilot_provider_declaration()
    assert state["value"] == "on_device"
    assert gateway.environ["EASYICU_PI_DATA_HANDLING"] == "on_device"
    clear = body.model_copy(update={"value": "unspecified"})
    routes.post_pi_copilot_provider_declaration(clear)
    assert gateway.environ["EASYICU_PI_DATA_HANDLING"] == "unspecified"
    assert gateway.environ["EASYICU_PI_INFERENCE_LOCATION"] == "unspecified"


def test_the_old_connection_route_cannot_bypass_the_declaration_record(monkeypatch):
    class ForbiddenService:
        def configure_provider(self, **_kwargs):
            pytest.fail("a declaration reached the connection writer")

    monkeypatch.setattr(routes, "get_pi_copilot_service", ForbiddenService)
    body = routes.PiProviderConfigRequest(api_key="synthetic", data_handling="on_device")
    with pytest.raises(HTTPException) as error:
        routes.post_pi_copilot_provider_config(
            body, Request({"type": "http", "headers": []}), Response(),
        )
    assert error.value.detail["error"] == "pi_provider_declaration_settings_only"


def test_a_registered_tool_cannot_smuggle_a_declaration_argument(store):
    _declare(store)
    original = store.declaration_path.read_bytes()
    with pytest.raises(PiCopilotError) as error:
        execute_tool("easyicu_workspace_status", {"data_handling": "on_device"},
                     ToolExecutionContext(session=PiSessionRecord(session_id="synthetic")))
    assert error.value.code == "pi_tool_unknown_arguments"
    assert store.declaration_path.read_bytes() == original


def test_the_hosted_guard_refuses_a_forged_admin_before_dispatch(store, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    for method in ("GET", "POST"):
        request = Request({"type": "http", "method": method, "scheme": "https",
                           "path": "/api/copilot/pi/provider-config/declaration",
                           "query_string": b"", "headers": [(b"x-role", b"admin")]})
        request._body = b'{"actor":{"id":"administrator"}}'
        if method == "GET":
            asyncio.run(hosted.guard_request(request))
        else:
            with pytest.raises(HTTPException) as error:
                asyncio.run(hosted.guard_request(request))
            assert error.value.detail["error"] == "pi_provider_declaration_admin_required"
            assert error.value.detail["reason"] == DECLARATION_MESSAGES["admin_required"]


def test_a_new_gateway_reads_the_human_policy_without_starting_a_process(store, tmp_path):
    from easyicu.webserver.pi_copilot.gateway import PiGatewayClient

    _declare(store, value="on_device")
    gateway = PiGatewayClient(app_dir=tmp_path, session_dir=tmp_path / "pi-sessions",
                              cwd=tmp_path, provider_store=store)
    try:
        assert gateway.environ["EASYICU_PI_DATA_HANDLING"] == "on_device"
        assert gateway.environ["EASYICU_PI_INFERENCE_LOCATION"] == "on_device"
    finally:
        gateway.close()


def test_an_environment_connection_change_lapses_the_record(store, monkeypatch):
    _declare(store)
    monkeypatch.setenv("EASYICU_PI_MODEL", "model-b")
    assert store.resolved_config().data_handling.value == "unspecified"
    monkeypatch.delenv("EASYICU_PI_MODEL")
    assert store.declaration_status()["lapsed"] is True
    assert store.resolved_config().data_handling.value == "unspecified"


def test_legacy_verification_receipts_require_reverification_for_a_human_choice(store):
    receipt = json.loads(store.receipt_path.read_text())
    receipt.pop("connection_fingerprint", None)
    store.receipt_path.write_text(json.dumps(receipt))
    assert store.public_status()["inference_verified"] is True
    _declare(store, value="on_device")
    assert store.public_status()["inference_verified"] is False
    _save(store)
    assert store.public_status()["inference_verified"] is True
    assert store.research_agent_environment(external_llm_opt_in=True)["EASYICU_PROVIDER_DATA_HANDLING"] == "on_device"


def test_an_unconfigured_connection_cannot_be_declared(store):
    store.config_path.unlink()
    state = store.declaration_status()
    assert state["may_change"] is False
    with pytest.raises(PiCopilotError) as error:
        _declare(store)
    assert error.value.code == "pi_provider_declaration_connection_required"
    assert not store.declaration_path.exists()


@pytest.mark.parametrize("damage", ["malformed", "permissions"])
def test_a_damaged_record_fails_closed_without_overwriting_it(store, damage):
    _declare(store)
    if damage == "malformed":
        with store.declaration_path.open("a") as stream:
            stream.write("{broken-record\n")
    else:
        store.declaration_path.chmod(0o644)
    original = store.declaration_path.read_bytes()
    for operation in (store.resolved_config, lambda: _declare(store, value="unspecified")):
        with pytest.raises(PiCopilotError) as error:
            operation()
        assert error.value.code == "pi_provider_declaration_record_unavailable"
    assert store.declaration_path.read_bytes() == original


@pytest.mark.parametrize("blocked", ["busy", "gateway_unavailable"])
def test_the_service_cannot_change_policy_under_an_active_turn(store, tmp_path, blocked):
    _declare(store)
    original = store.declaration_path.read_bytes()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", provider_store=store,
                              gateway=FakeGateway() if blocked == "busy" else object())
    if blocked == "busy":
        service._busy_sessions.add("synthetic")
    with pytest.raises(PiCopilotError) as error:
        service.declare_provider_data_handling(
            value="on_device", connection=CONNECTION, channel="settings_ui",
            notice_key=NOTICE, notice_version="2",
        )
    assert error.value.code == f"pi_provider_declaration_{blocked}"
    assert store.declaration_path.read_bytes() == original


def test_saving_a_connection_keeps_administrator_environment_policy(store, monkeypatch):
    monkeypatch.setenv("EASYICU_PI_DATA_HANDLING", "on_device")
    config, status = _save(store)
    assert config.data_handling.value == status["data_handling"] == "on_device"
    assert status["connection_verified"] is True
    assert json.loads(store.receipt_path.read_text())["fingerprint"] == config.fingerprint()


def test_an_explicit_env_clear_overrides_the_entire_saved_policy(store, monkeypatch):
    _save(store, data_handling="on_device")
    monkeypatch.setenv("EASYICU_PI_DATA_HANDLING", "unspecified")
    assert store.declaration_status()["value"] == "unspecified"
    assert store.resolved_config().inference_location.value == "unspecified"


@pytest.mark.parametrize("change", ["clear", "connection"])
def test_a_running_conversation_cannot_keep_a_cleared_or_lapsed_policy(store, tmp_path, change):
    _declare(store, value="on_device")
    gateway = FakeGateway()
    gateway.apply_provider_config(store.resolved_config())
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway, provider_store=store)
    record = PiSessionRecord(session_id="synthetic")
    assert service._data_handling_for_session(record) == "on_device"
    if change == "clear":
        _declare(store, value="unspecified")
    else:
        _save(store, model="model-b")
    assert service._data_handling_for_session(record) == "unspecified"


@pytest.mark.parametrize("field", ["data_handling", "inference_location"])
def test_the_connection_service_also_refuses_unrecorded_policy_writes(store, tmp_path, field):
    _declare(store)
    original = store.declaration_path.read_bytes()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", provider_store=store, gateway=object())
    with pytest.raises(PiCopilotError) as error:
        service.configure_provider(**CONNECTION, api_key="synthetic-key", api_transport="openai-completions",
                                   enable_ai=True, **{field: "on_device"})
    assert error.value.code == "pi_provider_declaration_settings_only"
    assert store.declaration_path.read_bytes() == original


@pytest.mark.parametrize("extra", [{"actor": "administrator"}, {"is_admin": True}])
def test_the_settings_contract_does_not_accept_client_identity(extra):
    with pytest.raises(ValidationError):
        routes.PiProviderDeclarationRequest(
            connection=CONNECTION, value="on_device", notice_key=NOTICE, notice_version="2", **extra,
        )


def test_saved_administrator_policy_has_the_same_read_only_precedence(store):
    _save(store, data_handling="zero_retention_declared")
    state = store.declaration_status()
    assert state["source"] == "administrator_configuration"
    assert state["value"] == "zero_retention_declared"
    assert state["may_change"] is False
    with pytest.raises(PiCopilotError) as error:
        _declare(store, value="unspecified")
    assert error.value.code == "pi_provider_declaration_administrator_configuration"


def test_invalid_administrator_policy_returns_a_typed_failure(store, monkeypatch):
    monkeypatch.setenv("EASYICU_PI_DATA_HANDLING", "automatic")
    with pytest.raises(PiCopilotError) as error:
        _save(store)
    assert error.value.code == "pi_provider_connection_not_configured"


def test_an_accepted_external_service_is_recorded_as_that_and_clears_the_gate(store):
    from easyicu.research_agent.providers.client_trust import ProviderAuthorization

    state = _declare(store, value="user_accepted_external")
    record = json.loads(store.declaration_path.read_text().splitlines()[-1])
    # The record says what the user decided, never zero retention or on-device.
    assert record["value"] == state["value"] == "user_accepted_external"
    _save(store)
    projected = store.research_agent_environment(external_llm_opt_in=True)
    assert projected["EASYICU_PROVIDER_DATA_HANDLING"] == "user_accepted_external"
    require_data_handling("miiv", projected["EASYICU_PROVIDER_DATA_HANDLING"])
    authorization = ProviderAuthorization.create(
        provider="fixture", model="model-a", base_url=CONNECTION["base_url"], destination="external",
        authorization_mode="authorization", data_handling="user_accepted_external",
    )
    # It stays an external destination, so every egress restriction still applies.
    assert authorization.destination == "external"
    assert authorization.inference_location.value == "unspecified"


def test_a_declaration_made_under_the_first_notice_still_applies(store):
    _declare(store, value="on_device", notice_version="2")
    lines = store.declaration_path.read_text().splitlines()
    legacy = json.loads(lines[-1]) | {"notice_version": "1"}
    store.declaration_path.write_text("\n".join([*lines[:-1], json.dumps(legacy)]) + "\n")
    assert store.declaration_status()["value"] == "on_device"
    with pytest.raises(PiCopilotError) as error:
        _declare(store, value="unspecified", notice_version="1")
    assert error.value.code == "pi_provider_declaration_notice_mismatch"
