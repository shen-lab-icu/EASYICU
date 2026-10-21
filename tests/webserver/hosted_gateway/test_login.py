import json
import re

import pytest
from starlette.testclient import TestClient


def test_login_template_and_cookie_contract(client, login):
    from easyicu.hosted_gateway.auth import CSRF_COOKIE, SESSION_COOKIE

    page = client.get("/login")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert page.headers["content-security-policy"] == (
        "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
        "form-action 'self'; frame-ancestors 'none'")
    assert '<p class="error" role="alert"></p>' in page.text
    assert "{{csrf}}" not in page.text and client.cookies.get(CSRF_COOKIE)
    response = login(client)
    assert response.status_code == 303 and response.headers["location"] == "/"
    cookie = next(h for h in response.headers.get_list("set-cookie") if SESSION_COOKIE in h)
    for attr in ("HttpOnly", "Secure", "SameSite=lax", "Path=/", "Max-Age=43200"):
        assert attr in cookie
    assert "Domain=" not in cookie


def test_unknown_and_wrong_password_are_identical(client):
    page = client.get("/login")
    csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
    replies = [client.post("/login", data={"username": name, "password": "wrong", "csrf": csrf},
                           headers={"origin": "https://icu.example:8443"})
               for name in ("alice", "missing")]
    assert [r.status_code for r in replies] == [401, 401]
    assert replies[0].content == replies[1].content
    assert "用户名或密码错误，请重试。" in replies[0].text
    assert "wrong" not in replies[0].text and "missing" not in replies[0].text


@pytest.mark.parametrize("path", ["/api/jobs/demo/events", "/api/data", "/api"])
@pytest.mark.parametrize("kind", ["missing", "expired", "tampered"])
def test_invalid_session_is_exact_api_error(client, login, gateway, path, kind):
    from easyicu.hosted_gateway.auth import SESSION_COOKIE

    if kind != "missing":
        login(client)
        value = client.cookies.get(SESSION_COOKIE)
        if kind == "expired":
            gateway[2][0] += 43200
        else:
            client.cookies.clear()
            client.cookies.set(SESSION_COOKIE, value[:-1] + ("a" if value[-1] != "a" else "b"))
    reply = client.get(path)
    assert reply.status_code == 401
    assert reply.json() == {"detail": {"error": "login_required"}}
    assert reply.headers["cache-control"] == "no-store"


def test_missing_session_html_redirects(client):
    reply = client.get("/projects?private=ignored", headers={"accept": "text/html"})
    assert reply.status_code == 302 and reply.headers["location"] == "/login"


@pytest.mark.parametrize("origin", [None, "https://attacker.example", "null"])
@pytest.mark.parametrize("path", ["/login", "/logout"])
def test_auth_posts_require_public_origin(client, origin, path):
    headers = {} if origin is None else {"origin": origin}
    assert client.post(path, headers=headers).status_code == 403


@pytest.mark.parametrize("kind", ["missing", "mismatch", "expired"])
def test_login_csrf_is_bound_to_browser_and_expiry(client, gateway, kind):
    page = client.get("/login")
    token = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
    if kind == "missing":
        client.cookies.clear()
    elif kind == "mismatch":
        token += "tamper"
    else:
        gateway[2][0] += 900
    response = client.post("/login", data={"username": "alice", "password": "synthetic-pass",
                                          "csrf": token},
                           headers={"origin": "https://icu.example:8443"})
    assert response.status_code == 403


def test_logout_clears_secure_cookie(client, login):
    from easyicu.hosted_gateway.auth import SESSION_COOKIE

    login(client)
    reply = client.post("/logout", headers={"origin": "https://icu.example:8443"})
    assert reply.status_code == 303 and reply.headers["location"] == "/login"
    assert client.cookies.get(SESSION_COOKIE) is None
    assert "Secure" in reply.headers["set-cookie"]
    assert client.get("/api/data").status_code == 401


def test_five_failures_lock_user_across_ips_for_fifteen_minutes(gateway, login):
    app, _, clock, _ = gateway

    def from_ip(ip):
        async def remote_app(scope, receive, send):
            if scope["type"] == "http":
                scope["client"] = (ip, 1)
            await app(scope, receive, send)
        return TestClient(remote_app, base_url="https://icu.example:8443", follow_redirects=False)

    for i in range(5):
        with from_ip(f"192.0.2.{i}") as client:
            assert login(client, password="wrong").status_code == 401
    with from_ip("192.0.2.99") as client:
        clock[0] += 899
        assert login(client).status_code == 429
        clock[0] += 1
        assert login(client).status_code == 303


def test_five_failures_lock_ip_across_usernames(client, login, gateway):
    for i in range(5):
        assert login(client, username=f"unknown{i}", password="wrong").status_code == 401
    assert login(client, username="bob").status_code == 429
    gateway[2][0] += 900
    assert login(client, username="bob").status_code == 303


def test_failures_outside_window_do_not_lock(client, login, gateway):
    for _ in range(4):
        assert login(client, password="wrong").status_code == 401
    gateway[2][0] += 900
    assert login(client, password="wrong").status_code == 401
    assert login(client).status_code == 303


def test_password_reset_and_removal_revoke_live_sessions(client, login, gateway):
    _, config, _, users = gateway
    login(client)
    users["alice"]["version"] = "changed"
    config.users_file.write_text(json.dumps(users))
    assert client.get("/api/data").status_code == 401
    login(client)
    del users["alice"]
    config.users_file.write_text(json.dumps(users))
    assert client.get("/api/data").status_code == 401


def test_untrusted_host_and_http_are_rejected(client):
    assert client.get("/login", headers={"host": "attacker.example"}).status_code == 400
    assert client.get("http://icu.example:8443/login").status_code == 400


def test_login_body_is_bounded(client):
    reply = client.post("/login", content=b"x" * 17000,
                        headers={"origin": "https://icu.example:8443",
                                 "content-type": "application/x-www-form-urlencoded"})
    assert reply.status_code == 413


def test_unknown_user_still_performs_scrypt(client, login, monkeypatch):
    from easyicu.hosted_gateway import auth

    calls = []
    derive = auth._derive

    def observe(password, salt):
        calls.append(salt)
        return derive(password, salt)

    monkeypatch.setattr(auth, "_derive", observe)
    assert login(client, username="unknown").status_code == 401
    assert len(calls) == 1


def test_csrf_signature_cannot_be_used_as_session(client, gateway):
    from easyicu.hosted_gateway.auth import SESSION_COOKIE, Signer

    token = Signer(gateway[1].secret_file.read_bytes(), lambda: gateway[2][0]).issue(
        "csrf", {"user": "alice", "version": "alice-v1"}, 43200)
    client.cookies.set(SESSION_COOKIE, token)
    assert client.get("/api/data").status_code == 401


def test_corrupt_live_users_file_fails_closed(client, login, gateway):
    login(client)
    gateway[1].users_file.write_text("broken")
    response = client.get("/api/data")
    assert response.status_code == 503
    assert "broken" not in response.text


def test_csrf_and_error_slots_are_escaped(client, monkeypatch):
    from easyicu.hosted_gateway import app, auth

    monkeypatch.setattr(auth.Signer, "issue", lambda *a, **kw: '<unsafe"&>')
    response = client.get("/login")
    assert 'value="&lt;unsafe&quot;&amp;&gt;"' in response.text
    monkeypatch.undo()
    page = client.get("/login")
    csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
    monkeypatch.setattr(app, "LOGIN_ERROR", '<unsafe"&>')
    response = client.post("/login", data={"username": "alice", "password": "wrong", "csrf": csrf},
                           headers={"origin": "https://icu.example:8443"})
    assert '&lt;unsafe&quot;&amp;&gt;</p>' in response.text


def test_concurrent_failures_cannot_bypass_five_attempt_limit(client):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    page = client.get("/login")
    csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
    barrier = threading.Barrier(6)

    def attempt(_):
        barrier.wait(timeout=5)
        return client.post("/login", data={"username": "alice", "password": "wrong", "csrf": csrf},
                           headers={"origin": "https://icu.example:8443"}).status_code

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(attempt, range(6)))
    assert sorted(results) == [401] * 5 + [429]
