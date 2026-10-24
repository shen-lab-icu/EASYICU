"""The connection form may reuse a private key only for its saved endpoint."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import settings
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import provider_config
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as routes


ENDPOINT = "/api/copilot/pi/provider-config"
KEY = "synthetic-saved-connection-private-key"
BASE_URL = "http://127.0.0.1:8317/v1"


@pytest.fixture
def connection(tmp_path, monkeypatch):
    for name in ("PROVIDER", "API_KEY", "BASE_URL", "MODEL", "API", "DATA_HANDLING", "INFERENCE_LOCATION"):
        monkeypatch.delenv(f"EASYICU_PI_{name}", raising=False)
    monkeypatch.setattr(settings, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "settings.json")
    store = provider_config.PiProviderConfigStore(
        config_path=tmp_path / "provider.env", receipt_path=tmp_path / "verification.json",
    )
    applied, calls = [], []
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json", provider_store=store,
        gateway=SimpleNamespace(apply_provider_config=applied.append),
    )
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    monkeypatch.setattr(service, "runtime_status", lambda: {"runtime": {"status": "ready"}})

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, dict(headers)))
        return 200, ({"data": [{"id": "old-model"}, {"id": "new-model"}]}
                     if method == "GET" else {"choices": [{"message": {"content": "READY"}}]})

    monkeypatch.setattr(provider_config, "_default_verification_transport", transport)
    store.verify_and_save(
        provider="easyicu-local", api_key=KEY, base_url=BASE_URL, model="old-model",
        api_transport="openai-completions", data_handling="zero_retention_declared",
    )
    calls.clear()
    return TestClient(app), store, calls, applied


def _body(**changes):
    return {"provider": "easyicu-local", "api_key": "", "base_url": BASE_URL,
            "model": "new-model", "api_transport": "openai-completions", "enable_ai": True,
            **changes}


@pytest.mark.parametrize("blank", ["", " \t ", None])
def test_list_then_verify_and_save_reuses_key_without_returning_it(connection, blank):
    client, store, calls, applied = connection
    body = _body(api_key=blank)
    if blank is None:
        body.pop("api_key")
    body.update(provider=" easyicu-local ", base_url=BASE_URL + "///")
    discovery_body = {key: value for key, value in body.items() if key != "model"}
    catalog = client.post(ENDPOINT + "/models", json=discovery_body)
    assert catalog.status_code == 200
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 200, response.text
    assert [call[0] for call in calls] == ["GET", "GET", "POST"]
    assert all(call[2]["x-api-key"] == KEY for call in calls)
    config = store.resolved_config()
    assert config.model == "new-model" and config.api_key == KEY
    assert config.data_handling.value == "zero_retention_declared"
    assert applied == [config]
    assert KEY not in response.text + catalog.text


@pytest.mark.parametrize("changes", [
    {"provider": "custom-openai"}, {"base_url": "http://127.0.0.2:8317/v1"},
    {"base_url": "http://127.0.0.1:8318/v1"}, {"base_url": BASE_URL + "/other"},
    {"api_transport": "openai-responses"},
])
def test_save_cannot_forward_saved_key_to_another_connection(connection, changes):
    client, store, calls, applied = connection
    before = store.config_path.read_bytes(), store.receipt_path.read_bytes()
    response = client.post(ENDPOINT, json=_body(**changes))
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "pi_provider_credential_required"
    assert calls == applied == []
    assert before == (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    assert KEY not in response.text


def test_blank_key_without_a_saved_connection_is_actionable(connection):
    client, store, calls, applied = connection
    store.config_path.unlink()
    response = client.post(ENDPOINT, json=_body())
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "pi_provider_credential_required"
    assert calls == applied == []


def test_saved_key_still_requires_explicit_opt_in(connection):
    client, _, calls, applied = connection
    response = client.post(ENDPOINT, json=_body(enable_ai=False))
    assert response.status_code == 403
    assert calls == applied == []


def test_legacy_default_declaration_fields_do_not_clear_saved_policy(connection):
    client, store, _, _ = connection
    response = client.post(ENDPOINT, json=_body(data_handling="unspecified"))
    assert response.status_code == 200
    assert store.resolved_config().data_handling.value == "zero_retention_declared"


@pytest.mark.parametrize("changes", [
    {"data_handling": "on_device"}, {"inference_location": "on_device"},
    {"data_handling": "zero_retention_declared"},
])
def test_connection_save_cannot_change_the_settings_only_declaration(connection, changes):
    client, store, calls, applied = connection
    before = store.config_path.read_bytes(), store.receipt_path.read_bytes()
    response = client.post(ENDPOINT, json=_body(**changes))
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "pi_provider_declaration_settings_only"
    assert before == (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    assert calls == applied == []


@pytest.mark.parametrize("model,expected", [("old-model", "zero_retention_declared"),
                                           ("new-model", "unspecified")])
def test_saved_settings_declaration_keeps_its_journal_authority(connection, model, expected):
    client, store, _, _ = connection
    config, _ = store.verify_and_save(
        provider="easyicu-local", api_key=KEY, base_url=BASE_URL, model="old-model",
        api_transport="openai-completions",
    )
    store.declare_data_handling(
        value="zero_retention_declared", connection=config.connection().model_dump(),
        notice_key="physionet_llm_data_handling_declaration", notice_version="1",
        channel="settings_ui",
    )
    before = store.declaration_path.read_bytes()
    response = client.post(ENDPOINT, json=_body(model=model))
    assert response.status_code == 200
    state = store.declaration_status()
    assert state["source"] == "settings_ui"
    assert state["actor"] == {"kind": "local_user", "id": "local_user"}
    assert state["value"] == expected
    assert state["lapsed"] is (model != "old-model")
    assert store._configured_connection().data_handling.value == "unspecified"
    assert store.declaration_path.read_bytes().startswith(before)
    if model == "old-model":
        assert store.declaration_path.read_bytes() == before


def test_omitted_declarations_preserve_the_saved_on_device_location(connection):
    client, store, _, _ = connection
    store.verify_and_save(
        provider="easyicu-local", api_key=KEY, base_url=BASE_URL, model="old-model",
        api_transport="openai-completions", data_handling="on_device",
        inference_location="on_device",
    )
    response = client.post(ENDPOINT, json=_body())
    assert response.status_code == 200
    config = store.resolved_config()
    assert config.data_handling.value == "on_device"
    assert config.inference_location.value == "on_device"


def test_a_new_credential_does_not_inherit_the_previous_service_declaration(connection):
    client, store, calls, _ = connection
    response = client.post(ENDPOINT, json=_body(
        api_key="synthetic-new-service-key", base_url="http://127.0.0.1:8318/v1",
    ))
    assert response.status_code == 200
    assert all(call[2]["x-api-key"] == "synthetic-new-service-key" for call in calls)
    assert store.resolved_config().data_handling.value == "unspecified"


def test_failed_verification_preserves_the_saved_key_and_receipt(connection, monkeypatch):
    client, store, _, applied = connection
    before = store.config_path.read_bytes(), store.receipt_path.read_bytes()
    monkeypatch.setattr(provider_config, "_default_verification_transport", lambda *_: (401, {}))
    response = client.post(ENDPOINT, json=_body())
    assert response.status_code == 400
    assert before == (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    assert applied == []


def test_invalid_save_body_never_echoes_a_private_key(connection):
    client, _, calls, applied = connection
    response = client.post(ENDPOINT, json=_body(api_key={"invalid": KEY}))
    assert response.status_code == 422
    assert KEY not in response.text
    assert calls == applied == []
