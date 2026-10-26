"""HTTP integration for the lead's Mac lane (TestClient is blocked on server)."""

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import provider_config as owner
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as routes
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway


ENDPOINT = "/api/copilot/pi/provider-config/declaration"
CONNECTION = {"provider": "fixture", "base_url": "http://localhost:8317/v1", "model": "fixture-model"}
BODY = {
    "connection": CONNECTION, "value": "on_device",
    "notice_key": "physionet_llm_data_handling_declaration", "notice_version": "2",
}


@pytest.fixture
def declaration_api(tmp_path, monkeypatch):
    for key in owner._CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    store = owner.PiProviderConfigStore(config_path=tmp_path / "connection",
                                       receipt_path=tmp_path / "verification.json")
    config, _ = store.verify_and_save(
        **CONNECTION, api_key="synthetic-key", api_transport="openai-completions",
        verifier=lambda method, *_: (200, {"data": [{"id": "fixture-model"}]} if method == "GET"
                                     else {"choices": [{"message": {"content": "READY"}}]}),
    )
    gateway = FakeGateway()
    gateway.apply_provider_config(config)
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway, provider_store=store)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    with TestClient(app) as client:
        yield client, store, gateway


def test_settings_can_read_declare_and_clear_without_exposing_credentials(declaration_api):
    client, store, gateway = declaration_api
    state = client.get(ENDPOINT).json()
    assert state["value"] == "unspecified" and state["may_change"] is True
    declared = client.post(ENDPOINT, json=BODY)
    assert declared.status_code == 200
    assert declared.json()["value"] == "on_device"
    assert client.get(ENDPOINT).json() == declared.json()
    assert "synthetic-key" not in declared.text
    assert gateway.environ["EASYICU_PI_DATA_HANDLING"] == "on_device"
    original = store.declaration_path.read_bytes()
    cleared = client.post(ENDPOINT, json=BODY | {"value": "unspecified"})
    assert cleared.status_code == 200 and cleared.json()["value"] == "unspecified"
    assert store.declaration_path.read_bytes().startswith(original)


@pytest.mark.parametrize("channel", ["tool", "model", "conversation", "plan"])
def test_non_settings_channels_are_refused_without_a_record(declaration_api, channel):
    client, store, _ = declaration_api
    response = client.post(ENDPOINT, json=BODY | {"channel": channel})
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "pi_provider_declaration_settings_only"
    assert not store.declaration_path.exists()


@pytest.mark.parametrize("extra", [{"actor": {"id": "administrator"}}, {"is_admin": True}])
def test_the_client_cannot_supply_identity_or_admin_authority(declaration_api, extra):
    client, store, _ = declaration_api
    assert client.post(ENDPOINT, json=BODY | extra).status_code == 422
    assert not store.declaration_path.exists()


def test_hosted_readers_cannot_set_or_clear_even_with_forged_headers(declaration_api, monkeypatch):
    client, store, _ = declaration_api
    assert client.post(ENDPOINT, json=BODY).status_code == 200
    original = store.declaration_path.read_bytes()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    read = client.get(ENDPOINT)
    assert read.status_code == 200
    assert read.json()["change_authority"] == "administrator_configuration"
    assert read.json()["may_change"] is False
    for value in ("zero_retention_declared", "unspecified"):
        response = client.post(ENDPOINT, json=BODY | {"value": value}, headers={"X-Role": "admin"})
        assert response.status_code == 403
        assert response.json()["detail"]["error"] == "pi_provider_declaration_admin_required"
        assert response.json()["detail"]["reason"] == owner.declaration_error("admin_required").message
    assert store.declaration_path.read_bytes() == original


def test_hosted_administrator_configuration_is_read_only(declaration_api, monkeypatch):
    client, store, _ = declaration_api
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_PI_DATA_HANDLING", "zero_retention_declared")
    response = client.get(ENDPOINT)
    assert response.status_code == 200
    state = response.json()
    assert state["value"] == "zero_retention_declared"
    assert state["source"] == "administrator_configuration"
    assert state["change_authority"] == "administrator_configuration"
    assert state["may_change"] is False
    assert state["actor"] is None and state["recorded_at"] is None
    for value in ("on_device", "unspecified"):
        refused = client.post(ENDPOINT, json=BODY | {"value": value})
        assert refused.status_code == 403
        assert refused.json()["detail"]["error"] == "pi_provider_declaration_admin_required"
        assert refused.json()["detail"]["reason"] == owner.declaration_error("admin_required").message
    assert not store.declaration_path.exists()


@pytest.mark.parametrize("changes,code", [
    ({"notice_version": "old"}, "notice_mismatch"),
    ({"value": "automatic"}, "value_invalid"),
    ({"connection": CONNECTION | {"model": "changed"}}, "connection_changed"),
])
def test_write_refusals_have_stable_codes(declaration_api, changes, code):
    client, store, _ = declaration_api
    response = client.post(ENDPOINT, json=BODY | changes)
    assert response.status_code in {409, 422}
    assert response.json()["detail"]["error"] == f"pi_provider_declaration_{code}"
    assert response.json()["detail"]["message"] == owner.declaration_error(code).message
    assert not store.declaration_path.exists()
