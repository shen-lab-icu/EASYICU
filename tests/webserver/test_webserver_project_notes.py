"""Project notes live in the project's local folder, not in the browser."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import guided_sessions


@pytest.fixture
def isolated_guided(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "guided-cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "guided-cfg" / "drafts.json"
    )
    projects = tmp_path / "guided-projects"
    projects.mkdir()
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", projects)
    return projects


def test_project_notes_round_trip_through_the_project_folder(isolated_guided: Path) -> None:
    created = guided_sessions.create_guided_draft(
        {"title": "Lactate notes", "parent_dir": str(isolated_guided)}
    )
    assert created["ok"] is True
    project_id = created["draft"]["id"]
    project_dir = Path(created["draft"]["project_dir"])

    empty = guided_sessions.read_project_notes(project_id)
    assert empty == {
        "ok": True, "project_id": project_id, "available": True, "present": False,
        "text": "", "updated_at": None,
    }
    # Writing nothing to a project without notes creates no file.
    assert guided_sessions.write_project_notes(project_id, "   ")["present"] is False
    assert not (project_dir / "project_notes.md").exists()

    saved = guided_sessions.write_project_notes(project_id, "第一版想法\r\n- 先看乳酸分布")
    assert saved["ok"] is True and saved["present"] is True
    assert saved["text"] == "第一版想法\n- 先看乳酸分布"
    assert saved["updated_at"].endswith("Z")
    assert (project_dir / "project_notes.md").read_text(encoding="utf-8") == "第一版想法\n- 先看乳酸分布"
    assert not (project_dir / ".project_notes.md.tmp").exists()
    assert guided_sessions.read_project_notes(project_id)["text"] == "第一版想法\n- 先看乳酸分布"

    too_long = guided_sessions.write_project_notes(project_id, "x" * 20001)
    assert too_long["ok"] is False and too_long["error"] == "project_notes_too_long"
    not_text = guided_sessions.write_project_notes(project_id, {"text": "no"})
    assert not_text["ok"] is False and not_text["error"] == "project_notes_invalid"

    # A project without a registered folder keeps the browser-local fallback.
    missing = guided_sessions.read_project_notes("draft_missing")
    assert missing["available"] is False and missing["text"] == ""
    blocked = guided_sessions.write_project_notes("draft_missing", "text")
    assert blocked["ok"] is False and blocked["error"] == "project_notes_unavailable"


def test_project_notes_routes(isolated_guided: Path) -> None:
    from easyicu.webserver.routes.guided import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    created = guided_sessions.create_guided_draft(
        {"title": "Route notes", "parent_dir": str(isolated_guided)}
    )
    project_id = created["draft"]["id"]

    assert client.get(f"/api/guided/projects/{project_id}/notes").json()["present"] is False
    saved = client.post(f"/api/guided/projects/{project_id}/notes", json={"text": "备忘"})
    assert saved.status_code == 200 and saved.json()["text"] == "备忘"
    assert client.get(f"/api/guided/projects/{project_id}/notes").json()["text"] == "备忘"
    rejected = client.post(f"/api/guided/projects/{project_id}/notes", json={"text": "y" * 20001})
    assert rejected.status_code == 400
    assert rejected.json()["detail"]["error"] == "project_notes_too_long"
    assert client.post("/api/guided/projects/draft_none/notes", json={"text": "t"}).status_code == 400
