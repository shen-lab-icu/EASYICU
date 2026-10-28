"""Hosted defaults, administrator boundaries, and gateway origin contract."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import __main__ as web_cli
from easyicu.webserver import agent_pipeline_runs, capabilities, settings, sources
from easyicu.webserver.app import app
from tests.support.hosted_http import stub_route


ADMIN_ROUTES = [
    "/api/settings/reset", "/api/extensions/mcp/install",
    "/api/extensions/state", "/api/extensions/remove", "/api/extensions/mcp/test",
    "/api/copilot/pi/provider-config", "/api/copilot/pi/provider-config/models",
    "/api/agent-runs/provider-config", "/api/agent-runs/codex-auth/login",
    "/api/agent-runs/codex-auth/cancel", "/api/agent-runs/codex-auth/logout",
    "/api/copilot/pi/research-provider/model", "/api/copilot/pi/research-provider/codex/login",
    "/api/copilot/pi/research-provider/codex/cancel", "/api/copilot/pi/research-provider/codex/logout",
]


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "settings.json")
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": []})
    monkeypatch.delenv("EASYICU_PUBLIC_ORIGIN", raising=False)
    monkeypatch.delenv("EASYICU_WEB_TRUST_PROXY", raising=False)


@pytest.mark.parametrize("hosted", ["1", "0", "true", ""])
def test_hosted_defaults_only_apply_to_explicit_flag(monkeypatch, hosted):
    monkeypatch.setenv("EASYICU_HOSTED", hosted)
    result = TestClient(app).get("/api/settings").json()
    assert result["language"] == ("zh" if hosted == "1" else "en")
    assert result["data_mode"] == ("real" if hosted == "1" else "demo")
    assert result["ai_enabled"] is False


@pytest.mark.parametrize("hosted", [True, False])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_settings_about_keeps_host_details_on_desktop_only(monkeypatch, hosted, method):
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    export_dir = str(settings._CONFIG_DIR / "exports")
    settings.update_settings({"export_dir": export_dir})
    response = TestClient(app).request(
        method, "/api/settings", **({"json": {"language": "zh"}} if method == "POST" else {}),
    )
    assert response.status_code == 200
    about = response.json()["about"]
    assert about["version"] and about["python"]
    assert settings.load_settings()["export_dir"] == export_dir
    if hosted:
        assert set(about) == {"version", "python"}
        assert "export_dir" not in response.json()
    else:
        assert response.json()["export_dir"] == export_dir
        assert about == {
            "version": settings._easyicu_version(),
            "python": settings.about()["python"],
            "config_path": str(settings._CONFIG_PATH),
            "local_access": settings.local_access_policy(),
        }


@pytest.mark.parametrize("key", sorted(set(settings.DEFAULTS) - {"language", "density", "reduce_motion", "support_contact"}))
def test_hosted_admin_settings_patch_is_atomic(monkeypatch, key):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    before = settings.load_settings()
    response = TestClient(app).post("/api/settings", json={key: settings.DEFAULTS[key], "language": "en"})
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "unavailable_when_hosted"
    assert settings.load_settings() == before


def test_hosted_display_settings_remain_writable_and_persist(monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    response = TestClient(app).post("/api/settings", json={"language": "en", "density": "compact", "reduce_motion": True})
    assert response.status_code == 200
    assert settings.load_settings()["language"] == "en"
    assert settings.load_settings()["density"] == "compact"
    assert settings.load_settings()["reduce_motion"] is True


@pytest.mark.parametrize("route", ADMIN_ROUTES)
def test_hosted_admin_routes_refuse_before_owner(monkeypatch, route):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    calls = stub_route(monkeypatch, route)
    response = TestClient(app).post(route, json={})
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"] == "unavailable_when_hosted"
    assert not calls


def test_hosted_open_output_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    response = TestClient(app).post("/api/jobs/synthetic/open-output", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "unavailable_when_hosted"


@pytest.mark.parametrize("route", ADMIN_ROUTES)
def test_desktop_admin_routes_keep_their_existing_validation(monkeypatch, route):
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    stub_route(monkeypatch, route)
    assert TestClient(app).post(route, json={}).status_code in {200, 422}


@pytest.mark.parametrize("hosted,expected", [("1", "王医生"), ("0", "easyicu_local_web_operator")])
def test_reviewer_comes_from_host_identity_only(monkeypatch, hosted, expected):
    from easyicu.webserver import agent_runs

    monkeypatch.setenv("EASYICU_HOSTED", hosted)
    monkeypatch.setenv("EASYICU_USER_DISPLAY_NAME", "王医生")
    captured = []
    monkeypatch.setattr(agent_runs, "create_human_signoff", lambda *a, **kw: captured.append(kw) or {"ok": True})
    assert isinstance(agent_pipeline_runs.server_reviewer_identity(), str)
    assert agent_pipeline_runs.server_reviewer_identity() == expected
    response = TestClient(app).post("/api/agent-runs/signoff", json={"reviewer": "Forged browser identity"})
    assert response.status_code == 200
    assert captured[0]["reviewer"] == expected


def test_unconfigured_hosted_reviewer_keeps_existing_typed_fallback(monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.delenv("EASYICU_USER_DISPLAY_NAME", raising=False)
    assert agent_pipeline_runs.server_reviewer_identity() == "easyicu_local_web_operator"


@pytest.mark.parametrize("origin,status", [("https://clinic.example", 200), ("https://foreign.example", 403), ("http://clinic.example", 403), ("https://clinic.example:444", 403), ("https://clinic.example/foreign", 403)])
def test_gateway_public_origin_is_exact(monkeypatch, origin, status):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_WEB_TRUST_PROXY", "1")
    monkeypatch.setenv("EASYICU_PUBLIC_ORIGIN", "https://clinic.example")
    client = TestClient(app, base_url="http://clinic.example", client=("127.0.0.1", 1234))
    response = client.post("/api/settings", json={"language": "zh"}, headers={"Origin": origin, "X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.10"})
    assert response.status_code == status, response.text
    assert client.get("/api/health", headers={"Host": "foreign.example"}).status_code == 400


def test_public_origin_never_grants_proxy_trust(monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_PUBLIC_ORIGIN", "https://clinic.example")
    response = TestClient(app).post("/api/settings", json={"language": "zh"}, headers={"Origin": "https://clinic.example", "X-Forwarded-Proto": "https"})
    assert response.status_code == 403
    assert TestClient(app).post("/api/settings", json={"language": "zh"}, headers={"X-Forwarded-Proto": "https"}).status_code == 403


def test_desktop_ignores_public_origin_and_keeps_admin_settings(monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    monkeypatch.setenv("EASYICU_PUBLIC_ORIGIN", "https://clinic.example")
    response = TestClient(app).post("/api/settings", json={"ai_enabled": True, "data_mode": "real"}, headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    assert response.json()["ai_enabled"] is True
    assert TestClient(app).post("/api/settings/reset").status_code == 200


def test_hosted_cli_requires_private_unix_socket(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    calls = []
    monkeypatch.setattr(web_cli.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    assert web_cli.run_app() == 2
    assert calls == []
    tmp_path.chmod(0o700)
    assert web_cli.run_app(uds=str(tmp_path / "web.sock")) == 0
    assert "--uds" in calls[0][0] and "--host" not in calls[0][0]


def test_trusted_hosted_uvicorn_limits_forwarding_to_its_socket(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_WEB_TRUST_PROXY", "1")
    socket = str(tmp_path / "web.sock")
    command = web_cli._uvicorn_cmd("127.0.0.1", 8765, uds=socket)
    assert "--proxy-headers" in command
    assert command[command.index("--forwarded-allow-ips") + 1] == "*"
    assert "--uds" in command and "--host" not in command


def test_proxy_headers_over_private_uds_keep_authenticated_gateway_access(monkeypatch, tmp_path):
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    tmp_path.chmod(0o700)
    socket = str(tmp_path / "web.sock")
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_WEB_TRUST_PROXY", "1")
    monkeypatch.setenv("EASYICU_PUBLIC_ORIGIN", "https://clinic.example")
    monkeypatch.setenv("EASYICU_WEB_UNIX_SOCKET", socket)
    proxy = ProxyHeadersMiddleware(app, trusted_hosts="*")

    async def socket_listener(scope, receive, send):
        scope.update(server=(socket, None), client=None)
        await proxy(scope, receive, send)

    client = TestClient(socket_listener, base_url="http://clinic.example")
    for origin, status in [("https://clinic.example", 200), ("https://foreign.example", 403)]:
        result = client.post("/api/settings", json={"language": "zh"}, headers={
            "Origin": origin, "X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.10",
        })
        assert result.status_code == status, result.text


def test_hosted_startup_refuses_a_tcp_only_configuration(monkeypatch):
    from easyicu.webserver.app import _refuse_a_shared_unix_socket

    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.delenv("EASYICU_WEB_UNIX_SOCKET", raising=False)
    with pytest.raises(RuntimeError, match="private Unix socket"):
        _refuse_a_shared_unix_socket()


def test_capabilities_expose_only_a_provisioned_boolean(monkeypatch, tmp_path):
    from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore

    monkeypatch.setenv("EASYICU_HOSTED", "1")
    store = PiProviderConfigStore(config_path=tmp_path / "provider", receipt_path=tmp_path / "verified")
    monkeypatch.setattr(PiProviderConfigStore, "public_status", lambda self: {"connection_verified": False})
    assert TestClient(app).get("/api/capabilities").json()["deployment"] == {"hosted": True, "provider_provisioned": False}
    monkeypatch.setattr(PiProviderConfigStore, "public_status", lambda self: {"connection_verified": True, "model_available": True, "provider": "must-not-leak"})
    deployment = TestClient(app).get("/api/capabilities").json()["deployment"]
    assert deployment == {"hosted": True, "provider_provisioned": True}
    assert store is not None and capabilities is not None
