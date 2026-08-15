from __future__ import annotations

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.routes import extensions as extension_routes


def _skill() -> str:
    return (
        "---\n"
        "name: clear-writing\n"
        "description: Keep scientific writing concise.\n"
        "---\n"
        "Use short paragraphs.\n"
    )


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
        },
    )
    assert installed.status_code == 200
    row = installed.json()["skill"]
    assert row["name"] == "clear-writing"
    assert len(row["digest"]) == 64
    assert "path" not in str(installed.json()).casefold()

    disabled = client.post(
        "/api/extensions/state",
        json={"kind": "skill", "name": "clear-writing", "enabled": False},
    )
    assert disabled.status_code == 200
    assert disabled.json()["extension"]["enabled"] is False

    removed = client.post(
        "/api/extensions/remove",
        json={"kind": "skill", "name": "clear-writing"},
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
