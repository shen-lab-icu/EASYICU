"""Unsaved provider catalog discovery with synthetic, offline transports."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import settings
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import provider_config
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as routes
from tests.support.node import run_node

ENDPOINT = "/api/copilot/pi/provider-config/models"
KEY = "synthetic-discovery-private-key"
BASE_URL = "http://127.0.0.1:8317/v1"


@pytest.fixture
def discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("PROVIDER", "API_KEY", "BASE_URL", "MODEL", "API"):
        monkeypatch.delenv(f"EASYICU_PI_{name}", raising=False)
    monkeypatch.setattr(settings, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "settings.json")
    store = provider_config.PiProviderConfigStore(
        config_path=tmp_path / "provider.env",
        receipt_path=tmp_path / "verification.json",
    )
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=object(),  # type: ignore[arg-type]
        provider_store=store,
    )
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    calls: list[tuple[Any, ...]] = []

    def transport(method, url, headers, body, timeout):
        assert settings.load_settings()["ai_enabled"] is True
        calls.append((method, url, dict(headers), body, timeout))
        return 200, {"data": [{"id": "zeta-model"}, {"id": "alpha-model"}]}

    monkeypatch.setattr(provider_config, "_default_verification_transport", transport)
    return TestClient(app), store, calls


def _body(**changes: Any) -> dict[str, Any]:
    return {
        "provider": "easyicu-local",
        "api_key": KEY,
        "base_url": BASE_URL,
        "api_transport": "openai-completions",
        "enable_ai": True,
        **changes,
    }


def _save(store: provider_config.PiProviderConfigStore, model="stored-model") -> None:
    store.verify_and_save(
        provider="easyicu-local",
        api_key=KEY,
        base_url=BASE_URL,
        model=model,
        api_transport="openai-completions",
        verifier=lambda method, *_: (
            200,
            {"data": [{"id": model}]}
            if method == "GET"
            else {"choices": [{"message": {"content": "READY"}}]},
        ),
    )


def test_discovery_lists_sorted_models_without_saving_or_selecting(discovery, caplog):
    client, store, calls = discovery

    response = client.post(ENDPOINT, json=_body())

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "schema_version": "easyicu.provider-model-catalog/1",
        "catalog_status": "endpoint_reported",
        "models": ["alpha-model", "zeta-model"],
        "models_reported": 2,
        "recommended": None,
        "secrets_returned": False,
    }
    assert len(calls) == 1
    assert calls[0][:2] == ("GET", BASE_URL + "/models")
    assert calls[0][2]["x-api-key"] == KEY
    assert calls[0][3] is None
    assert not store.config_path.exists()
    assert not store.receipt_path.exists()
    assert KEY not in response.text + caplog.text


def test_discovery_filters_secret_reflections(discovery, monkeypatch):
    client, _, _ = discovery
    monkeypatch.setattr(
        provider_config,
        "_default_verification_transport",
        lambda *_: (
            200,
            {"data": [{"id": KEY}, {"id": "safe-model"}, {"id": f"reflected-{KEY}"}]},
        ),
    )
    response = client.post(ENDPOINT, json=_body())
    assert response.status_code == 200
    assert response.json()["models"] == ["safe-model"]
    assert KEY not in response.text


@pytest.mark.parametrize(
    ("status", "payload", "code"),
    [
        (401, {"error": KEY}, "pi_provider_auth_failed"),
        (403, {"error": KEY}, "pi_provider_auth_failed"),
        (503, {"error": KEY}, "pi_provider_verification_failed"),
        (200, {"data": KEY}, "pi_provider_models_response_invalid"),
        (200, [KEY], "pi_provider_models_response_invalid"),
    ],
)
def test_discovery_preserves_sanitized_typed_errors(
    discovery, monkeypatch, caplog, status, payload, code
):
    client, store, _ = discovery
    monkeypatch.setattr(
        provider_config, "_default_verification_transport", lambda *_: (status, payload)
    )
    response = client.post(ENDPOINT, json=_body())
    assert response.status_code == 400  # Same mapping as provider verification.
    assert response.json()["detail"]["error"] == code
    assert KEY not in response.text + caplog.text
    assert not store.config_path.exists()
    assert not store.receipt_path.exists()


@pytest.mark.parametrize(
    ("changes", "code", "status"),
    [
        ({"base_url": "file:///tmp/catalog"}, "pi_provider_base_url_rejected", 400),
        (
            {"api_transport": "unsupported"},
            "pi_provider_api_transport_unsupported",
            400,
        ),
        ({"enable_ai": False}, "external_llm_opt_in_required", 403),
        ({"api_key": ""}, "pi_provider_credential_required", 422),
        ({"api_key": KEY + "\ninvalid"}, "pi_provider_api_key_invalid", 400),
        ({"api_key": KEY * 400}, "pi_provider_api_key_invalid", 400),
    ],
)
def test_discovery_rejects_before_transport(discovery, changes, code, status):
    client, store, calls = discovery
    response = client.post(ENDPOINT, json=_body(**changes))
    assert response.status_code == status
    assert response.json()["detail"]["error"] == code
    assert calls == []
    assert not store.config_path.exists()
    assert not store.receipt_path.exists()
    assert KEY not in response.text


@pytest.mark.parametrize("enabled_before", [False, True])
def test_discovery_requires_explicit_opt_in_even_with_saved_key(
    discovery, enabled_before
):
    client, store, calls = discovery
    _save(store)
    settings.update_settings({"ai_enabled": enabled_before})
    response = client.post(ENDPOINT, json=_body(api_key="", enable_ai=False))
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "external_llm_opt_in_required"
    assert calls == []
    assert settings.load_settings()["ai_enabled"] is enabled_before


@pytest.mark.parametrize("blank", ["", "  \t"])
def test_blank_key_reuses_only_identical_normalized_connection(discovery, blank):
    client, store, calls = discovery
    _save(store)
    before = (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    response = client.post(
        ENDPOINT,
        json=_body(
            api_key=blank,
            provider=" easyicu-local ",
            base_url=" " + BASE_URL + "/// ",
            api_transport="OPENAI-COMPLETIONS",
        ),
    )
    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][1] == BASE_URL + "/models"
    assert calls[0][2]["x-api-key"] == KEY
    assert before == (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    assert KEY not in response.text


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "custom-openai"},
        {"base_url": "http://127.0.0.2:8317/v1"},  # Host only.
        {"base_url": "http://127.0.0.1:8318/v1"},  # Port only.
        {"base_url": "http://127.0.0.1:8317/other"},
        {"api_transport": "openai-responses"},
    ],
)
def test_blank_key_refuses_other_connection_before_transport(discovery, changes):
    client, store, calls = discovery
    _save(store)
    before = (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    response = client.post(ENDPOINT, json=_body(api_key="", **changes))
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "pi_provider_credential_required"
    assert calls == []
    assert before == (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    assert KEY not in response.text


def test_new_key_for_other_connection_does_not_replace_stored_credential(discovery):
    client, store, calls = discovery
    _save(store)
    before = (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    response = client.post(
        ENDPOINT,
        json=_body(
            api_key="another-synthetic-key", base_url="http://127.0.0.1:8318/v1"
        ),
    )
    assert response.status_code == 200
    assert calls[0][2]["x-api-key"] == "another-synthetic-key"
    assert before == (store.config_path.read_bytes(), store.receipt_path.read_bytes())
    assert "another-synthetic-key" not in response.text


@pytest.mark.parametrize(
    ("provider", "default_model"),
    [
        ("easyicu-local", provider_config.DEFAULT_MODEL),
        ("openai", provider_config.DEFAULT_MODEL),
        ("deepseek", "deepseek-chat"),
        ("anthropic", "claude-sonnet-4-6"),
        ("google", "gemini-3.5-flash"),
        ("openrouter", None),
        ("custom-openai", None),
    ],
)
def test_recommendation_uses_only_reported_preset_default(
    discovery, monkeypatch, provider, default_model
):
    client, _, _ = discovery
    models = ["zeta-model", "alpha-model"] + ([default_model] if default_model else [])
    monkeypatch.setattr(
        provider_config,
        "_default_verification_transport",
        lambda *_: (200, {"data": [{"id": row} for row in reversed(models)]}),
    )
    response = client.post(ENDPOINT, json=_body(provider=provider))
    assert response.status_code == 200
    assert response.json()["models"] == sorted(models)
    assert response.json()["recommended"] == default_model


@pytest.mark.parametrize("stored_available", [False, True])
def test_stored_model_precedes_preset_default_only_when_reported(
    discovery, monkeypatch, stored_available
):
    client, store, _ = discovery
    _save(store)
    default = provider_config.DEFAULT_MODEL
    models = [default] + (["stored-model"] if stored_available else [])
    monkeypatch.setattr(
        provider_config,
        "_default_verification_transport",
        lambda *_: (200, {"data": [{"id": row} for row in models]}),
    )
    response = client.post(ENDPOINT, json=_body(api_key=""))
    assert response.status_code == 200
    assert response.json()["recommended"] == (
        "stored-model" if stored_available else default
    )


@pytest.mark.parametrize(
    "changes", [{"enable_ai": "true"}, {"model": "not-a-selection"}]
)
def test_discovery_body_rejects_coerced_opt_in_and_model_selection(discovery, changes):
    client, _, calls = discovery
    response = client.post(ENDPOINT, json=_body(**changes))
    assert response.status_code == 422
    assert calls == []
    assert KEY not in response.text


@pytest.mark.parametrize("payload", [_body(api_key={"value": KEY}), [_body()]])
def test_request_validation_never_echoes_private_inputs(discovery, payload):
    client, _, calls = discovery
    response = client.post(ENDPOINT, json=payload)
    assert response.status_code == 422
    assert KEY not in response.text
    assert calls == []


def test_invalid_json_never_echoes_private_inputs(discovery):
    client, _, calls = discovery
    response = client.post(
        ENDPOINT,
        content='{"api_key":"' + KEY + '",',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "pi_provider_request_invalid"
    assert KEY not in response.text
    assert calls == []


@pytest.mark.parametrize(
    ("provider", "api_transport", "payload", "header", "model"),
    [
        (
            "anthropic",
            "anthropic-messages",
            {"data": [{"id": "served-model"}]},
            "x-api-key",
            "served-model",
        ),
        (
            "google",
            "google-generative-ai",
            {"models": [{"name": "models/served-model"}]},
            "x-goog-api-key",
            "served-model",
        ),
        (
            "openai",
            "openai-responses",
            {"data": [{"id": "served-model"}]},
            "Authorization",
            "served-model",
        ),
    ],
)
def test_unsaved_discovery_uses_requested_wire_protocol(
    discovery, monkeypatch, provider, api_transport, payload, header, model
):
    client, store, _ = discovery

    def transport(method, url, headers, body, timeout):
        assert method == "GET"
        assert body is None
        assert headers[header] == (
            f"Bearer {KEY}" if header == "Authorization" else KEY
        )
        return 200, payload

    monkeypatch.setattr(provider_config, "_default_verification_transport", transport)
    response = client.post(
        ENDPOINT, json=_body(provider=provider, api_transport=api_transport)
    )
    assert response.status_code == 200
    assert response.json()["models"] == [model]
    assert not store.config_path.exists()
    assert not store.receipt_path.exists()
    assert KEY not in response.text


def test_api_client_discovery_posts_body_and_preserves_typed_error():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the API-client harness")
    source = Path("src/easyicu/webserver/static/js/api.js").read_text(encoding="utf-8")
    program = (
        """
const assert = require('node:assert/strict');
global.window = {};
global.document = { readyState: 'loading', addEventListener() {} };
const calls = [];
global.fetch = async (url, options) => {
  calls.push([url, options]);
  return { ok: false, status: 422, json: async () => ({ detail: {
    error: 'pi_provider_credential_required', reason: 'Enter a credential.',
    details: { retryable: false }
  } }) };
};
"""
        + source
        + """
(async () => {
  const body = { provider: 'custom-openai', enable_ai: true };
  await assert.rejects(window.EU_API.discoverPiCopilotProviderModels(body), err => {
    assert.equal(err.code, 'pi_provider_credential_required');
    assert.equal(err.status, 422);
    assert.deepEqual(err.details, { retryable: false });
    return true;
  });
  assert.equal(calls[0][0], '/api/copilot/pi/provider-config/models');
  assert.equal(calls[0][1].method, 'POST');
  assert.deepEqual(JSON.parse(calls[0][1].body), body);
})().catch(err => { console.error(err); process.exitCode = 1; });
"""
    )
    result = run_node(node, program, timeout=15)
    assert result.returncode == 0, result.stderr
