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
    verification_url = "https://auth.openai.com/codex/device"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
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
        raise AssertionError(method)

    def __enter__(self) -> "_FakeRuntime":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def notifications_since(self, _index: int = 0) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolated_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from easyicu.research_agent.providers import codex_app_server

    codex_account_sessions.shutdown_all()
    _FakeRuntime.instances.clear()
    _FakeRuntime.logged_in = False
    _FakeRuntime.verification_url = "https://auth.openai.com/codex/device"
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


def test_login_creates_http_only_isolated_user_session() -> None:
    response = Response()

    result = codex_account_sessions.start_login(_request(), response)

    assert result["login_started"] is True
    assert result["verification_url"] == "https://auth.openai.com/codex/device"
    assert result["user_code"] == "ABCD-EFGH"
    header = response.headers["set-cookie"]
    assert "HttpOnly" in header
    assert "SameSite=strict" in header
    runtime_environment = _FakeRuntime.instances[0].kwargs["environment"]
    assert runtime_environment["CODEX_HOME"] != "/private/operator/.codex"
    assert "codex-user-sessions" in runtime_environment["CODEX_HOME"]
    assert "OPENAI_API_KEY" not in runtime_environment


def test_two_browser_sessions_never_share_codex_home() -> None:
    first_response = Response()
    second_response = Response()
    codex_account_sessions.start_login(_request(), first_response)
    codex_account_sessions.start_login(_request(), second_response)

    first_cookie = _cookie_from_response(first_response)
    second_cookie = _cookie_from_response(second_response)
    assert first_cookie != second_cookie
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


def test_repeated_login_request_reuses_one_device_ceremony() -> None:
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
    assert repeated["user_code"] == "ABCD-EFGH"


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
    _FakeRuntime.logged_in = True
    logged_out = codex_account_sessions.logout(_request(cookie))

    methods = [method for method, _params in _FakeRuntime.instances[0].calls]
    assert "account/login/cancel" in methods
    assert "account/logout" in methods
    assert cancelled["ok"] is True
    assert logged_out["auth"]["authentication_verified"] is False


def test_device_login_rejects_non_openai_verification_url() -> None:
    _FakeRuntime.verification_url = "https://attacker.example/codex/device"

    with pytest.raises(
        codex_account_sessions.CodexAccountSessionError,
        match="codex_auth_verification_url_invalid",
    ):
        codex_account_sessions.start_login(_request(), Response())


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
    assert started.json()["user_code"] == "ABCD-EFGH"

    _FakeRuntime.logged_in = True
    ready = client.get("/api/agent-runs/provider-status?provider=codex")
    assert ready.status_code == 200
    status = ready.json()["provider_status"]
    assert status["ready"] is True
    assert status["authentication_mode"] == "chatgpt_account"
    assert status["credential_source"] == "codex_user_auth"
    assert status["account_label"] == "r***@example.org"
    assert "researcher@example.org" not in ready.text
