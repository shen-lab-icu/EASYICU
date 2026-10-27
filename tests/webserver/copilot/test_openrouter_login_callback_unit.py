"""The login callback exchanges a synthetic authorization code without ASGI."""

from starlette.requests import Request

from easyicu.webserver.pi_copilot import openrouter_login
from easyicu.webserver.routes.pi_copilot import get_pi_copilot_openrouter_callback


def _callback_request(token, code="synthetic-code"):
    return Request({
        "type": "http",
        "method": "GET",
        "path": openrouter_login.CALLBACK_PATH,
        "query_string": f"code={code}".encode(),
        "headers": [(b"cookie", f"{openrouter_login.COOKIE_NAME}={token}".encode())],
    })


def test_callback_exchanges_authorization_code_once_and_keeps_key_on_host(monkeypatch):
    logins = openrouter_login.OpenRouterLoginStore()
    monkeypatch.setattr(openrouter_login, "_STORE", logins)
    token, _ = logins.start("http://127.0.0.1" + openrouter_login.CALLBACK_PATH)
    exchanges = []
    key = "sk-or-synthetic-callback-0001"

    def exchange(code, verifier):
        exchanges.append((code, verifier))
        return 200, {"key": key}

    monkeypatch.setattr(openrouter_login, "_default_exchange", exchange)
    response = get_pi_copilot_openrouter_callback(_callback_request(token))

    assert response.status_code == 303
    assert response.headers["location"] == "/?openrouter_login=signed_in#guided"
    assert len(exchanges) == 1
    assert exchanges[0][0] == "synthetic-code"
    assert exchanges[0][1]
    assert key not in str(dict(response.headers))
    assert key.encode() not in response.body
    assert logins.resolve_api_key(
        token,
        credential_source=openrouter_login.CREDENTIAL_SOURCE,
        provider=openrouter_login.OPENROUTER_PROVIDER,
        base_url=openrouter_login.OPENROUTER_BASE_URL,
        api_key="",
    ) == key

    replay = get_pi_copilot_openrouter_callback(_callback_request(token))
    assert replay.headers["location"] == (
        "/?openrouter_login=failed&reason=openrouter_login_not_started#guided"
    )
    assert len(exchanges) == 1


def test_callback_without_a_started_login_cannot_exchange_code(monkeypatch):
    monkeypatch.setattr(openrouter_login, "_STORE", openrouter_login.OpenRouterLoginStore())
    exchanges = []
    monkeypatch.setattr(openrouter_login, "_default_exchange", lambda *args: exchanges.append(args))

    response = get_pi_copilot_openrouter_callback(_callback_request("unknown-login"))

    assert response.status_code == 303
    assert response.headers["location"] == (
        "/?openrouter_login=failed&reason=openrouter_login_not_started#guided"
    )
    assert exchanges == []
