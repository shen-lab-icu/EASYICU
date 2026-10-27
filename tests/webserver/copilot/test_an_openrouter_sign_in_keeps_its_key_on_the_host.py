"""An OpenRouter account sign-in connects OpenRouter without the key reaching the page.

OpenRouter's OAuth PKCE flow returns a fresh API key to the app. The host
keeps it behind an HttpOnly cookie, uses it for OpenRouter's own API only, and
forgets it once the ordinary verify-and-save path has stored the connection.
Synthetic codes, keys and transports only; nothing leaves the test process.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import settings
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import openrouter_login, provider_config
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as routes

KEY = "sk-or-v1-synthetic-signin-key-0001"
OPENROUTER = {
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "api_transport": "openai-completions",
    "enable_ai": True,
}


class _Gateway:
    def __init__(self) -> None:
        self.applied: list[Any] = []

    def apply_provider_config(self, config) -> None:
        self.applied.append(config)


@pytest.fixture
def signin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("PROVIDER", "API_KEY", "BASE_URL", "MODEL", "API"):
        monkeypatch.delenv(f"EASYICU_PI_{name}", raising=False)
    monkeypatch.setattr(settings, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "settings.json")
    store = provider_config.PiProviderConfigStore(
        config_path=tmp_path / "provider.env",
        receipt_path=tmp_path / "verification.json",
    )
    gateway = _Gateway()
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,  # type: ignore[arg-type]
        provider_store=store,
    )
    monkeypatch.setattr(service, "runtime_status", lambda: {"runtime": {"status": "ready"}})
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    logins = openrouter_login.OpenRouterLoginStore()
    monkeypatch.setattr(openrouter_login, "_STORE", logins)
    exchanges: list[tuple[str, str]] = []

    def exchange(code: str, verifier: str):
        exchanges.append((code, verifier))
        return 200, {"key": KEY}

    monkeypatch.setattr(openrouter_login, "_default_exchange", exchange)
    calls: list[tuple[str, str, dict[str, str]]] = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, dict(headers)))
        if method == "GET":
            return 200, {"data": [{"id": "vendor/model-a"}, {"id": "vendor/model-b:free"}]}
        return 200, {"choices": [{"message": {"content": "READY"}}]}

    monkeypatch.setattr(provider_config, "_default_verification_transport", transport)
    # Offline: the public hostname is not resolved; the address rule under
    # test here is the sign-in owner's exact OpenRouter match.
    monkeypatch.setattr(provider_config, "validate_credential_endpoint", lambda url: None)
    return TestClient(app, base_url="http://127.0.0.1:8765"), store, exchanges, calls, gateway


def _sign_in(client: TestClient) -> str:
    started = client.post("/api/copilot/pi/provider-config/openrouter/login", json={"enable_ai": True})
    assert started.status_code == 200, started.text
    query = parse_qs(urlsplit(started.json()["auth_url"]).query)
    landed = client.get(openrouter_login.CALLBACK_PATH + "?code=synthetic-code-1", follow_redirects=False)
    assert landed.status_code == 303
    assert landed.headers["location"] == "/?openrouter_login=signed_in#guided"
    return query["code_challenge"][0]


def test_start_offers_a_pkce_authorization_and_an_httponly_login_cookie(signin):
    client, _store, exchanges, _calls, _gateway = signin
    started = client.post("/api/copilot/pi/provider-config/openrouter/login", json={"enable_ai": True})
    payload = started.json()
    parts = urlsplit(payload["auth_url"])
    query = parse_qs(parts.query)

    assert (parts.scheme, parts.netloc, parts.path) == ("https", "openrouter.ai", "/auth")
    assert query["callback_url"] == ["http://127.0.0.1:8765" + openrouter_login.CALLBACK_PATH]
    assert query["code_challenge_method"] == ["S256"]
    assert payload["secrets_returned"] is False
    cookie = started.headers["set-cookie"]
    assert openrouter_login.COOKIE_NAME in cookie and "HttpOnly" in cookie
    assert f"Path={openrouter_login.COOKIE_PATH}" in cookie and "samesite=lax" in cookie.lower()

    client.get(openrouter_login.CALLBACK_PATH + "?code=synthetic-code-1", follow_redirects=False)
    (_code, verifier), = exchanges
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == [expected]


def test_start_needs_the_external_ai_confirmation(signin):
    client, *_ = signin
    refused = client.post("/api/copilot/pi/provider-config/openrouter/login", json={"enable_ai": False})
    assert refused.status_code == 403
    assert refused.json()["detail"]["error"] == "external_llm_opt_in_required"


def test_signed_in_browser_lists_and_saves_openrouter_without_holding_the_key(signin):
    client, store, _exchanges, calls, gateway = signin
    _sign_in(client)

    listed = client.post(
        "/api/copilot/pi/provider-config/models",
        json={**OPENROUTER, "api_key": "", "credential_source": "openrouter_login"},
    )
    assert listed.status_code == 200, listed.text
    assert listed.json()["models"] == ["vendor/model-a", "vendor/model-b:free"]
    assert KEY not in listed.text
    assert calls[-1][1] == "https://openrouter.ai/api/v1/models"
    assert KEY in calls[-1][2].get("Authorization", "")

    token = client.cookies.get(openrouter_login.COOKIE_NAME)
    saved = client.post(
        "/api/copilot/pi/provider-config",
        json={**OPENROUTER, "api_key": "", "model": "vendor/model-b:free", "credential_source": "openrouter_login"},
    )
    assert saved.status_code == 200, saved.text
    assert KEY not in saved.text
    assert store.resolved_config().api_key == KEY
    assert gateway.applied[-1].model == "vendor/model-b:free"
    assert openrouter_login.COOKIE_NAME in saved.headers.get("set-cookie", "")
    # Replaying the old cookie finds nothing: the host forgot the key.
    with pytest.raises(PiCopilotError):
        openrouter_login.login_store().resolve_api_key(
            token, credential_source="openrouter_login", provider="openrouter",
            base_url=OPENROUTER["base_url"], api_key="",
        )

    # The sign-in is spent: the stored connection is now the only holder.
    again = client.post(
        "/api/copilot/pi/provider-config/models",
        json={**OPENROUTER, "api_key": "", "credential_source": "openrouter_login"},
    )
    assert again.status_code == 409
    assert again.json()["detail"]["error"] == "openrouter_login_not_signed_in"


@pytest.mark.parametrize(
    "changes",
    [
        {"base_url": "https://openrouter.ai.evil.test/api/v1"},
        {"base_url": "http://127.0.0.1:8317/v1"},
        {"provider": "custom-openai"},
        {"api_key": "sk-typed-alongside"},
    ],
)
def test_the_signed_in_key_goes_to_openrouter_only(signin, changes):
    client, _store, _exchanges, calls, _gateway = signin
    _sign_in(client)
    before = len(calls)
    refused = client.post(
        "/api/copilot/pi/provider-config/models",
        json={**OPENROUTER, "api_key": "", "credential_source": "openrouter_login", **changes},
    )
    assert refused.status_code == 422
    assert refused.json()["detail"]["error"] == "openrouter_login_wrong_service"
    assert len(calls) == before


def test_a_callback_without_a_started_login_or_with_a_spent_code_is_refused(signin):
    client, _store, exchanges, _calls, _gateway = signin
    stray = client.get(openrouter_login.CALLBACK_PATH + "?code=synthetic-code-1", follow_redirects=False)
    assert stray.headers["location"] == "/?openrouter_login=failed&reason=openrouter_login_not_started#guided"
    assert exchanges == []

    _sign_in(client)
    replay = client.get(openrouter_login.CALLBACK_PATH + "?code=synthetic-code-2", follow_redirects=False)
    assert replay.headers["location"] == "/?openrouter_login=failed&reason=openrouter_login_not_started#guided"
    assert len(exchanges) == 1


def test_a_denied_or_failed_exchange_returns_to_the_form_with_its_reason(signin, monkeypatch):
    client, *_ = signin
    client.post("/api/copilot/pi/provider-config/openrouter/login", json={"enable_ai": True})
    denied = client.get(openrouter_login.CALLBACK_PATH, follow_redirects=False)
    assert denied.headers["location"] == "/?openrouter_login=failed&reason=openrouter_login_denied#guided"

    monkeypatch.setattr(openrouter_login, "_default_exchange", lambda code, verifier: (400, {"error": "bad"}))
    client.post("/api/copilot/pi/provider-config/openrouter/login", json={"enable_ai": True})
    failed = client.get(openrouter_login.CALLBACK_PATH + "?code=synthetic-code-3", follow_redirects=False)
    assert failed.headers["location"] == "/?openrouter_login=failed&reason=openrouter_login_exchange_failed#guided"
    listed = client.post(
        "/api/copilot/pi/provider-config/models",
        json={**OPENROUTER, "api_key": "", "credential_source": "openrouter_login"},
    )
    assert listed.json()["detail"]["error"] == "openrouter_login_not_signed_in"


def test_a_sign_in_expires():
    now = [1000.0]
    logins = openrouter_login.OpenRouterLoginStore(clock=lambda: now[0])
    token, _url = logins.start("http://127.0.0.1:8765" + openrouter_login.CALLBACK_PATH)
    logins.exchange_authorization_code(token, "synthetic-code", exchange=lambda code, verifier: (200, {"key": KEY}))
    assert logins.resolve_api_key(token, credential_source="openrouter_login", **{k: OPENROUTER[k] for k in ("provider", "base_url")}, api_key="") == KEY
    now[0] += openrouter_login.LOGIN_TTL_SECONDS + 1
    with pytest.raises(PiCopilotError) as expired:
        logins.resolve_api_key(token, credential_source="openrouter_login", provider="openrouter", base_url=OPENROUTER["base_url"], api_key="")
    assert expired.value.code == "openrouter_login_not_signed_in"


def test_a_typed_credential_is_unchanged_by_the_sign_in_owner():
    logins = openrouter_login.OpenRouterLoginStore()
    assert logins.resolve_api_key("", credential_source="typed", provider="deepseek", base_url="https://api.deepseek.com/v1", api_key="sk-typed") == "sk-typed"


def _verifier(models_status: int):
    def verifier(method, url, headers, body, timeout):
        if method == "GET":
            return models_status, {}
        return 200, {"choices": [{"message": {"content": "READY"}}]}
    return verifier


def test_a_service_without_a_model_list_is_verified_by_its_inference_probe():
    config = provider_config.PiProviderConfigStore.make_config(
        provider="zhipu", api_key="sk-synthetic", base_url="http://127.0.0.1:9100/v1",
        model="named-model", api_transport="openai-completions",
    )
    with pytest.raises(PiCopilotError) as listed:
        provider_config.discover_provider_models(config, transport=_verifier(404))
    assert listed.value.code == "pi_provider_model_list_unavailable"

    verified = provider_config.verify_provider_connection(config, transport=_verifier(404))
    assert verified["inference_verified"] is True
    assert verified["catalog_status"] == "endpoint_unavailable"


def test_without_a_model_list_a_failed_probe_or_another_wire_still_fails():
    config = provider_config.PiProviderConfigStore.make_config(
        provider="zhipu", api_key="sk-synthetic", base_url="http://127.0.0.1:9100/v1",
        model="named-model", api_transport="openai-completions",
    )

    def no_model(method, url, headers, body, timeout):
        return (404, {}) if method == "GET" else (404, {"error": "model not found"})

    with pytest.raises(PiCopilotError) as failed:
        provider_config.verify_provider_connection(config, transport=no_model)
    assert failed.value.code == "pi_provider_inference_verification_failed"

    anthropic = provider_config.PiProviderConfigStore.make_config(
        provider="anthropic", api_key="sk-synthetic", base_url="http://127.0.0.1:9100/v1",
        model="named-model", api_transport="anthropic-messages",
    )
    with pytest.raises(PiCopilotError) as unlisted:
        provider_config.verify_provider_connection(anthropic, transport=_verifier(404))
    assert unlisted.value.code == "pi_provider_model_list_unavailable"
