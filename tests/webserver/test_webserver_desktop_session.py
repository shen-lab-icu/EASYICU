from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyicu.webserver.desktop_session import (
    DESKTOP_COOKIE,
    DESKTOP_HEADER,
    DESKTOP_SESSION_ENV,
    install_desktop_session,
)


def _desktop_app(monkeypatch, token="desktop-secret") -> FastAPI:
    monkeypatch.setenv(DESKTOP_SESSION_ENV, token)
    app = FastAPI()

    @app.get("/")
    def root():
        return {"ready": True}

    @app.get("/api/catalog")
    def catalog():
        return {"ready": True}

    assert install_desktop_session(app) is True
    return app


def test_desktop_session_is_disabled_without_launch_token(monkeypatch):
    monkeypatch.delenv(DESKTOP_SESSION_ENV, raising=False)
    app = FastAPI()

    @app.get("/")
    def root():
        return {"ready": True}

    assert install_desktop_session(app) is False
    assert TestClient(app).get("/").status_code == 200


def test_desktop_health_probe_requires_exact_private_header(monkeypatch):
    client = TestClient(_desktop_app(monkeypatch))

    assert client.get("/api/catalog").status_code == 403
    assert client.get("/api/catalog", headers={DESKTOP_HEADER: "wrong"}).status_code == 403
    response = client.get(
        "/api/catalog", headers={DESKTOP_HEADER: "desktop-secret"}
    )
    assert response.status_code == 200


def test_bootstrap_token_is_exchanged_for_httponly_cookie(monkeypatch):
    client = TestClient(_desktop_app(monkeypatch))
    response = client.get(
        "/?desktop_token=desktop-secret&lang=zh",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/?lang=zh"
    cookie = response.headers["set-cookie"]
    assert f"{DESKTOP_COOKIE}=desktop-secret" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie

    client.cookies.set(DESKTOP_COOKIE, "desktop-secret")
    assert client.get("/").status_code == 200


def test_bootstrap_rejects_token_on_non_root_route(monkeypatch):
    client = TestClient(_desktop_app(monkeypatch))
    response = client.get("/api/catalog?desktop_token=desktop-secret")
    assert response.status_code == 403
