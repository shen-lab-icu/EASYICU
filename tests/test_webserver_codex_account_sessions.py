from __future__ import annotations

from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from easyicu.webserver import codex_account_sessions
from easyicu.webserver.app import app


def _request(cookie: str = "", *, scheme: str = "http") -> Request:
    headers = []
    if cookie:
        headers.append((b"cookie", cookie.encode("ascii")))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": scheme,
            "path": "/api/agent-runs/codex-auth/status",
            "raw_path": b"/api/agent-runs/codex-auth/status",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8765),
        }
    )


def _cookie_from_response(response: Response) -> str:
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    morsel = parsed[codex_account_sessions.COOKIE_NAME]
    return f"{codex_account_sessions.COOKIE_NAME}={morsel.value}"


class _FakeRuntime:
    instances: list["_FakeRuntime"] = []
    logged_in = False
    auth_url = "https://auth.openai.com/oauth/authorize?state=opaque"
    verification_url = "https://auth.openai.com/codex/device"
    model_rows: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def request(
        self,
        method: str,
        params: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "account/read":
            return {
                "account": (
                    {
                        "type": "chatgpt",
                        "email": "researcher@example.org",
                        "planType": "plus",
                    }
                    if self.__class__.logged_in
                    else None
                ),
                "requiresOpenaiAuth": not self.__class__.logged_in,
            }
        if method == "account/login/start":
            if params.get("type") == "chatgpt":
                return {
                    "type": "chatgpt",
                    "loginId": "login-private",
                    "authUrl": self.__class__.auth_url,
                }
            return {
                "type": "chatgptDeviceCode",
                "loginId": "login-private",
                "verificationUrl": self.__class__.verification_url,
                "userCode": "ABCD-EFGH",
            }
        if method == "account/login/cancel":
            return {}
        if method == "account/logout":
            self.__class__.logged_in = False
            return {}
        if method == "model/list":
            return {"data": [dict(row) for row in self.__class__.model_rows]}
        raise AssertionError(method)

    def __enter__(self) -> "_FakeRuntime":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def notifications_since(self, _index: int = 0) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolated_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from easyicu.research_agent.providers import codex_app_server

    codex_account_sessions.shutdown_all()
    _FakeRuntime.instances.clear()
    _FakeRuntime.logged_in = False
    _FakeRuntime.auth_url = "https://auth.openai.com/oauth/authorize?state=opaque"
    _FakeRuntime.verification_url = "https://auth.openai.com/codex/device"
    _FakeRuntime.model_rows = [
        {
            "model": "gpt-5.6-luna",
            "displayName": "GPT-5.6 Luna",
            "description": "Fast Codex model",
            "hidden": False,
            "isDefault": True,
        },
        {
            "model": "gpt-5.6-sol",
            "displayName": "GPT-5.6 Sol",
            "hidden": False,
            "isDefault": False,
        },
    ]
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", "/private/operator/.codex")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-account-boundary")
    monkeypatch.setattr(
        codex_account_sessions,
        "CodexAppServerRuntime",
        _FakeRuntime,
    )
    monkeypatch.setattr(
        codex_app_server,
        "CodexAppServerRuntime",
        _FakeRuntime,
    )
    yield
    codex_account_sessions.shutdown_all()


def test_browser_login_creates_http_only_isolated_user_session() -> None:
    response = Response()

    result = codex_account_sessions.start_login(_request(), response)

    assert result["login_started"] is True
    assert result["flow"] == "browser"
    assert result["auth_url"].startswith("https://auth.openai.com/")
    assert "user_code" not in result
    header = response.headers["set-cookie"]
    assert "HttpOnly" in header
    assert "SameSite=strict" in header
    runtime_environment = _FakeRuntime.instances[0].kwargs["environment"]
    assert runtime_environment["CODEX_HOME"] != "/private/operator/.codex"
    assert "codex-user-sessions" in runtime_environment["CODEX_HOME"]
    assert "OPENAI_API_KEY" not in runtime_environment
    assert (
        "account/login/start",
        {"type": "chatgpt", "codexStreamlinedLogin": True},
    ) in _FakeRuntime.instances[0].calls


def test_pi_conversation_projection_returns_only_private_auth_coordinate() -> None:
    response = Response()
    codex_account_sessions.start_login(_request(), response)
    cookie = _cookie_from_response(response)
    _FakeRuntime.logged_in = True
    request = _request(cookie)
    binding = codex_account_sessions.binding_for_request(request)
    codex_home = Path(_FakeRuntime.instances[0].kwargs["environment"]["CODEX_HOME"])
    auth_file = codex_home / "auth.json"
    auth_file.write_text('{"tokens":{"access_token":"not-returned"}}', encoding="utf-8")
    auth_file.chmod(0o600)

    environment = codex_account_sessions.pi_conversation_environment_for_binding(
        binding,
        model="gpt-5.6-luna",
    )

    assert environment["EASYICU_PI_PROVIDER"] == "openai-codex"
    assert environment["EASYICU_PI_API"] == "openai-codex-responses"
    assert environment["EASYICU_PI_MODEL"] == "gpt-5.6-luna"
    assert environment["EASYICU_PI_CODEX_AUTH_FILE"] == str(auth_file)
    assert environment["EASYICU_PI_CODEX_SESSION_SHA256"] == binding
    assert "not-returned" not in str(environment)
    assert (
        "account/read",
        {"refreshToken": True},
    ) in _FakeRuntime.instances[0].calls


def test_two_browser_sessions_never_share_codex_home() -> None:
    first_response = Response()
    second_response = Response()
    codex_account_sessions.start_login(_request(), first_response)
    codex_account_sessions.start_login(_request(), second_response)

    first_cookie = _cookie_from_response(first_response)
    second_cookie = _cookie_from_response(second_response)
    assert first_cookie != second_cookie
    _FakeRuntime.logged_in = True
    first_environment = codex_account_sessions.environment_for_request(
        _request(first_cookie)
    )
    second_environment = codex_account_sessions.environment_for_request(
        _request(second_cookie)
    )
    assert first_environment["CODEX_HOME"] != second_environment["CODEX_HOME"]
    assert (
        first_environment["EASYICU_CODEX_SESSION_SHA256"]
        != second_environment["EASYICU_CODEX_SESSION_SHA256"]
    )


def test_repeated_login_request_reuses_one_browser_ceremony() -> None:
    first_response = Response()
    codex_account_sessions.start_login(_request(), first_response)
    cookie = _cookie_from_response(first_response)

    repeated = codex_account_sessions.start_login(
        _request(cookie),
        Response(),
    )

    methods = [method for method, _params in _FakeRuntime.instances[0].calls]
    assert methods.count("account/login/start") == 1
    assert repeated["login_started"] is True
    assert repeated["auth_url"].startswith("https://auth.openai.com/")


def test_missing_or_forged_cookie_never_probes_operator_login() -> None:
    missing = codex_account_sessions.status(_request())
    forged = codex_account_sessions.status(
        _request(f"{codex_account_sessions.COOKIE_NAME}=forged")
    )

    assert missing["account_session_status"] == "codex_auth_login_required"
    assert forged["account_session_status"] == "codex_auth_login_required"
    assert missing["authentication_verified"] is False
    assert _FakeRuntime.instances == []


def test_session_codex_home_symlink_never_reaches_operator_login(
    tmp_path: Path,
) -> None:
    response = Response()
    codex_account_sessions.start_login(_request(), response)
    cookie = _cookie_from_response(response)
    _FakeRuntime.logged_in = True
    environment = codex_account_sessions.environment_for_request(_request(cookie))
    codex_home = Path(environment["CODEX_HOME"])
    operator_home = tmp_path / "operator-codex"
    operator_home.mkdir()
    codex_account_sessions.shutdown_all()
    codex_home.rmdir()
    codex_home.symlink_to(operator_home, target_is_directory=True)

    with pytest.raises(
        codex_account_sessions.CodexAccountSessionError,
        match="codex_auth_session_invalid",
    ):
        codex_account_sessions.environment_for_request(_request(cookie))

    assert str(operator_home) not in str(_FakeRuntime.instances[-1].kwargs)


def test_status_returns_only_masked_account_metadata_after_login() -> None:
    response = Response()
    codex_account_sessions.start_login(_request(), response)
    cookie = _cookie_from_response(response)
    _FakeRuntime.logged_in = True

    result = codex_account_sessions.status(_request(cookie))

    assert result["authentication_verified"] is True
    assert result["account_session_status"] == "codex_auth_ready"
    assert result["account_label"] == "r***@example.org"
    assert result["plan_type"] == "plus"
    assert "researcher@example.org" not in str(result)


def test_cancel_and_logout_are_bound_to_the_same_browser_session() -> None:
    response = Response()
    codex_account_sessions.start_login(_request(), response)
    cookie = _cookie_from_response(response)

    cancelled = codex_account_sessions.cancel_login(_request(cookie))
    assert _FakeRuntime.instances[0].closed is True
    _FakeRuntime.logged_in = True
    logged_out = codex_account_sessions.logout(_request(cookie))

    methods = [
        method
        for runtime in _FakeRuntime.instances
        for method, _params in runtime.calls
    ]
    assert "account/login/cancel" in methods
    assert "account/logout" in methods
    assert len(_FakeRuntime.instances) == 2
    assert cancelled["ok"] is True
    assert logged_out["auth"]["authentication_verified"] is False


def test_device_login_rejects_non_openai_verification_url() -> None:
    _FakeRuntime.verification_url = "https://attacker.example/codex/device"

    with pytest.raises(
        codex_account_sessions.CodexAccountSessionError,
        match="codex_auth_url_invalid",
    ):
        codex_account_sessions.start_login(
            _request(),
            Response(),
            flow="device_code",
        )


def test_device_code_remains_an_explicit_fallback() -> None:
    result = codex_account_sessions.start_login(
        _request(),
        Response(),
        flow="device_code",
    )

    assert result["flow"] == "device_code"
    assert result["verification_url"] == "https://auth.openai.com/codex/device"
    assert result["user_code"] == "ABCD-EFGH"
    assert (
        "account/login/start",
        {"type": "chatgptDeviceCode"},
    ) in _FakeRuntime.instances[0].calls


def test_model_catalog_and_immutable_binding_use_the_same_account_home() -> None:
    response = Response()
    codex_account_sessions.start_login(_request(), response)
    cookie = _cookie_from_response(response)
    request = _request(cookie)
    _FakeRuntime.logged_in = True

    catalog = codex_account_sessions.models(request)
    binding = codex_account_sessions.binding_for_request(request)
    environment = codex_account_sessions.environment_for_binding(
        binding,
        model="gpt-5.6-luna",
    )

    assert [row["id"] for row in catalog["models"]] == [
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    ]
    assert catalog["catalog_authority"] == (
        "easyicu.codex-subscription-model-catalog/1"
    )
    assert catalog["models"][0]["catalog_source"] == "codex_app_server"
    assert catalog["models"][2]["catalog_source"] == ("openai_documented_subscription")
    assert environment["EASYICU_CODEX_MODEL"] == "gpt-5.6-luna"
    assert environment["EASYICU_CODEX_SESSION_SHA256"] == binding
    assert "OPENAI_API_KEY" not in environment


def test_documented_models_fill_an_older_app_server_catalog() -> None:
    _FakeRuntime.model_rows = [
        {
            "model": "gpt-5.5",
            "displayName": "GPT-5.5",
            "hidden": False,
            "isDefault": True,
        }
    ]
    response = Response()
    codex_account_sessions.start_login(_request(), response)
    request = _request(_cookie_from_response(response))
    _FakeRuntime.logged_in = True

    catalog = codex_account_sessions.models(request)

    assert [row["id"] for row in catalog["models"]] == [
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    ]
    assert catalog["models"][0]["is_default"] is True
    assert catalog["models"][1]["catalog_source"] == ("openai_documented_subscription")
    assert (
        codex_account_sessions.validated_model_for_request(
            request,
            "gpt-5.6-luna",
        )
        == "gpt-5.6-luna"
    )


def test_http_routes_keep_auth_bound_to_the_browser_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.setattr(
        agent_route.settings_store,
        "load_settings",
        lambda: {"ai_enabled": True},
    )
    client = TestClient(app)

    missing = client.get("/api/agent-runs/codex-auth/status")
    assert missing.status_code == 200
    assert missing.json()["auth"]["authentication_verified"] is False
    assert _FakeRuntime.instances == []

    started = client.post("/api/agent-runs/codex-auth/login", json={})
    assert started.status_code == 200
    assert "HttpOnly" in started.headers["set-cookie"]
    assert started.json()["flow"] == "browser"
    assert started.json()["auth_url"].startswith("https://auth.openai.com/")

    _FakeRuntime.logged_in = True
    ready = client.get("/api/agent-runs/provider-status?provider=codex")
    assert ready.status_code == 200
    status = ready.json()["provider_status"]
    assert status["ready"] is True
    assert status["authentication_mode"] == "chatgpt_account"
    assert status["credential_source"] == "codex_user_auth"
    assert status["account_label"] == "r***@example.org"
    assert "researcher@example.org" not in ready.text

    copilot_status = client.get("/api/copilot/pi/research-provider/codex/status")
    models = client.get("/api/copilot/pi/research-provider/codex/models")
    assert copilot_status.status_code == 200
    assert copilot_status.json()["auth"]["authentication_verified"] is True
    assert models.status_code == 200
    assert models.json()["models"][0]["id"] == "gpt-5.6-luna"
    assert models.json()["catalog_authority"] == (
        "easyicu.codex-subscription-model-catalog/1"
    )
    assert models.json()["secrets_returned"] is False
