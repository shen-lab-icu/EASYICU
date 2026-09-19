from __future__ import annotations

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.routes import extensions as extension_routes


def _skill() -> str:
    return (
        "---\n"
        "name: clear-writing\n"
        "description: Keep scientific writing concise.\n"
        "category: Clinical Research\n"
        "---\n"
        "Use short paragraphs.\n"
    )


def _revision(client) -> str:
    """Read the current extension activation revision (D-P2-7 confirmation)."""

    response = client.get("/api/extensions")
    assert response.status_code == 200
    revision = response.json()["activation_sha256"]
    assert len(revision) == 64
    return revision


def test_extension_api_installs_toggles_and_removes_skill(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    client = TestClient(app)

    installed = client.post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill(),
            "stages": ["conversation", "writing"],
            "enabled": True,
            "expected_sha256": _revision(client),
        },
    )
    assert installed.status_code == 200
    row = installed.json()["skill"]
    assert row["name"] == "clear-writing"
    assert row["category"] == "Clinical Research"
    assert len(row["digest"]) == 64
    assert "path" not in str(installed.json()).casefold()

    detail = client.get("/api/extensions/skills/clear-writing")
    assert detail.status_code == 200
    assert detail.json()["instructions"] == "Use short paragraphs."
    assert detail.json()["skill_md"] == _skill()
    assert detail.json()["digest"] == row["digest"]
    assert detail.json()["category"] == "Clinical Research"
    assert "path" not in str(detail.json()).casefold()

    invalid_category = client.post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill().replace("Clinical Research", "../private"),
            "stages": ["conversation"],
            "enabled": False,
            "expected_sha256": _revision(client),
        },
    )
    assert invalid_category.status_code == 400

    missing = client.get("/api/extensions/skills/not-installed")
    assert missing.status_code == 400

    disabled = client.post(
        "/api/extensions/state",
        json={
            "kind": "skill",
            "name": "clear-writing",
            "enabled": False,
            "expected_sha256": _revision(client),
        },
    )
    assert disabled.status_code == 200
    assert disabled.json()["extension"]["enabled"] is False

    removed = client.post(
        "/api/extensions/remove",
        json={
            "kind": "skill",
            "name": "clear-writing",
            "expected_sha256": _revision(client),
        },
    )
    assert removed.status_code == 200
    assert removed.json()["extensions"]["skills"] == []


def test_extension_api_tests_and_installs_allowlisted_mcp(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    monkeypatch.setattr(
        extension_routes,
        "list_mcp_tools",
        lambda url: {
            "ok": True,
            "transport": "streamable-http",
            "tools": [{"name": "search", "description": "Search metadata"}],
            "tool_count": 1,
        },
    )
    client = TestClient(app)

    tested = client.post(
        "/api/extensions/mcp/test",
        json={"url": "http://127.0.0.1:9876/mcp"},
    )
    assert tested.status_code == 200
    assert tested.json()["tools"][0]["name"] == "search"

    installed = client.post(
        "/api/extensions/mcp/install",
        json={
            "name": "metadata-tools",
            "url": "http://127.0.0.1:9876/mcp",
            "allowed_tools": ["search"],
            "enabled": True,
            "expected_sha256": _revision(client),
        },
    )
    assert installed.status_code == 200
    row = installed.json()["mcp_server"]
    assert row["transport"] == "streamable-http"
    assert row["allowed_tools"] == ["search"]
    assert row["authentication"] == "none"


def test_extension_api_rejects_unknown_install_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    response = TestClient(app).post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill(),
            "stages": ["conversation"],
            "enabled": True,
            "filesystem_path": "/tmp/skill",
        },
    )
    assert response.status_code == 422


def test_extension_mutations_reject_a_missing_revision(tmp_path, monkeypatch) -> None:
    """D-P2-7: install/overwrite/remove without expected_sha256 must fail."""

    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    client = TestClient(app)

    assert (
        client.post(
            "/api/extensions/skills/install",
            json={"skill_md": _skill(), "stages": ["conversation"]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/extensions/mcp/install",
            json={
                "name": "metadata-tools",
                "url": "http://127.0.0.1:9876/mcp",
                "allowed_tools": ["search"],
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/extensions/remove",
            json={"kind": "skill", "name": "clear-writing"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/extensions/state",
            json={"kind": "skill", "name": "clear-writing", "enabled": True},
        ).status_code
        == 422
    )


def test_extension_mutations_reject_a_stale_revision(tmp_path, monkeypatch) -> None:
    """D-P2-7: a revision the caller never saw fails closed with 409."""

    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    client = TestClient(app)
    stale = "0" * 64

    skill = client.post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill(),
            "stages": ["conversation"],
            "expected_sha256": stale,
        },
    )
    assert skill.status_code == 409
    assert skill.json()["detail"]["error"] == "extension_revision_mismatch"

    current = _revision(client)
    assert current != stale
    installed = client.post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill(),
            "stages": ["conversation"],
            "expected_sha256": current,
        },
    )
    assert installed.status_code == 200

    # The install above moved the revision: replaying the now-stale value
    # must fail instead of overwriting or removing unseen state.
    overwrite = client.post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill(),
            "stages": ["conversation", "writing"],
            "expected_sha256": current,
        },
    )
    assert overwrite.status_code == 409

    removed = client.post(
        "/api/extensions/remove",
        json={"kind": "skill", "name": "clear-writing", "expected_sha256": current},
    )
    assert removed.status_code == 409
    assert removed.json()["detail"]["error"] == "extension_revision_mismatch"

    fresh_removed = client.post(
        "/api/extensions/remove",
        json={
            "kind": "skill",
            "name": "clear-writing",
            "expected_sha256": _revision(client),
        },
    )
    assert fresh_removed.status_code == 200

    malformed = client.post(
        "/api/extensions/remove",
        json={"kind": "skill", "name": "clear-writing", "expected_sha256": "zz"},
    )
    assert malformed.status_code == 422


def test_extension_state_toggle_requires_a_current_revision(
    tmp_path, monkeypatch
) -> None:
    """State toggles can activate an installed external MCP: same CAS gate."""

    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    client = TestClient(app)
    installed = client.post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill(),
            "stages": ["conversation"],
            "expected_sha256": _revision(client),
        },
    )
    assert installed.status_code == 200

    stale_toggle = client.post(
        "/api/extensions/state",
        json={
            "kind": "skill",
            "name": "clear-writing",
            "enabled": False,
            "expected_sha256": "0" * 64,
        },
    )
    assert stale_toggle.status_code == 409
    assert stale_toggle.json()["detail"]["error"] == "extension_revision_mismatch"


def test_concurrent_overwrites_with_one_stale_digest_land_once(
    tmp_path, monkeypatch
) -> None:
    """CAS: two concurrent overwrites on the same stale digest -> 1x200+1x409."""

    import threading

    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    client = TestClient(app)
    installed = client.post(
        "/api/extensions/skills/install",
        json={
            "skill_md": _skill(),
            "stages": ["conversation"],
            "expected_sha256": _revision(client),
        },
    )
    assert installed.status_code == 200
    stale = _revision(client)

    barrier = threading.Barrier(2)
    statuses: list[int] = []

    def overwrite() -> None:
        barrier.wait()
        response = client.post(
            "/api/extensions/skills/install",
            json={
                "skill_md": _skill(),
                "stages": ["conversation", "writing"],
                "expected_sha256": stale,
            },
        )
        statuses.append(response.status_code)

    threads = [threading.Thread(target=overwrite) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(statuses) == [200, 409]


def test_pinned_transport_rewrites_loopback_dns_to_validated_ip() -> None:
    """TOCTOU close: loopback-HTTP DNS names connect to the validated IP."""

    import httpx

    from easyicu.extensions.mcp_client import _validated_endpoint

    connect_url, transport = _validated_endpoint("http://localhost:8317/mcp")
    assert "localhost" not in connect_url
    assert transport is not None

    plain_url, plain_transport = _validated_endpoint("http://127.0.0.1:8317/mcp")
    assert plain_url == "http://127.0.0.1:8317/mcp"
    assert plain_transport is None

    https_url, https_transport = _validated_endpoint("https://localhost:8443/mcp")
    assert https_url == "https://localhost:8443/mcp"
    assert https_transport is None


def test_pinned_transport_preserves_host_header_end_to_end() -> None:
    """The pinned connection reaches 127.0.0.1 with the original Host."""

    import anyio
    import httpx
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            seen["host"] = self.headers.get("Host")
            seen["peer"] = self.client_address[0]
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from easyicu.extensions.mcp_client import _validated_endpoint

        connect_url, transport = _validated_endpoint(f"http://localhost:{port}/mcp")
        assert transport is not None

        async def post() -> int:
            async with httpx.AsyncClient(transport=transport) as client:
                response = await client.post(connect_url, json={"hello": "world"})
                return response.status_code

        assert anyio.run(post) == 200
    finally:
        server.shutdown()
    assert seen["host"] == f"localhost:{port}"
    assert seen["peer"] == "127.0.0.1"
