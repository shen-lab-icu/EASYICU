from __future__ import annotations

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.workspace import (
    WORKSPACE_ARTIFACT_AUTHORITY,
    ProjectWorkspace,
    project_workspace_id,
)


def test_project_workspace_writes_reads_edits_checks_and_previews(tmp_path: Path) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")

    written = workspace.write_file(
        "project-a",
        "demo/index.html",
        "<!doctype html><title>Demo</title><h1>Draft</h1>",
    )
    assert written["file"] == "demo/index.html"
    assert written["created"] is True
    assert {
        key: written[key] for key in WORKSPACE_ARTIFACT_AUTHORITY
    } == dict(WORKSPACE_ARTIFACT_AUTHORITY)

    read = workspace.read_file("project-a", "demo/index.html")
    assert read["text"].endswith("<h1>Draft</h1>")
    assert read["total_lines"] == 1

    edited = workspace.edit_file(
        "project-a",
        "demo/index.html",
        old_text="<h1>Draft</h1>",
        new_text="<h1>Ready</h1>",
        expected_sha256=written["sha256"],
    )
    assert edited["replacements"] == 1
    assert workspace.check_file("project-a", "demo/index.html") == {
        "file": "demo/index.html",
        "media_type": "text/html",
        "checker": "html.parser",
        "valid": True,
        "check_scope": "bounded_static_syntax",
        **dict(WORKSPACE_ARTIFACT_AUTHORITY),
    }
    assert "<h1>Ready</h1>" in workspace.preview_file(
        "project-a", "demo/index.html"
    )["text"]
    assert workspace.list_files("project-a")[0]["file"] == "demo/index.html"


@pytest.mark.parametrize(
    "relative_file",
    ["../outside.html", "/tmp/outside.html", "nested/../../outside.html"],
)
def test_project_workspace_rejects_path_escape(
    tmp_path: Path, relative_file: str
) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")

    with pytest.raises(PiCopilotError) as raised:
        workspace.write_file("project-a", relative_file, "blocked")

    assert raised.value.code == "pi_workspace_path_escape"


def test_project_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    root = workspace.project_root("project-a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PiCopilotError) as raised:
        workspace.write_file("project-a", "linked/nested/outside.html", "blocked")

    assert raised.value.code == "pi_workspace_path_escape"
    assert not (outside / "nested").exists()
    assert not (outside / "outside.html").exists()


def test_project_workspace_rejects_project_root_symlink(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    projects_root = workspace_root / "projects"
    projects_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "index.html").write_text("private", encoding="utf-8")
    (projects_root / project_workspace_id("project-a")).symlink_to(
        outside, target_is_directory=True
    )
    workspace = ProjectWorkspace(workspace_root)

    with pytest.raises(PiCopilotError) as raised:
        workspace.write_file("project-a", "index.html", "blocked")

    assert raised.value.code == "pi_workspace_project_root_symlink_blocked"
    with pytest.raises(PiCopilotError) as read_blocked:
        workspace.read_file("project-a", "index.html")
    assert read_blocked.value.code == "pi_workspace_project_root_symlink_blocked"
    assert (outside / "index.html").read_text(encoding="utf-8") == "private"


def test_project_workspace_rejects_projects_directory_symlink(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace_root / "projects").symlink_to(outside, target_is_directory=True)
    workspace = ProjectWorkspace(workspace_root)

    with pytest.raises(PiCopilotError) as raised:
        workspace.write_file("project-a", "index.html", "blocked")

    assert raised.value.code == "pi_workspace_projects_root_symlink_blocked"
    assert list(outside.iterdir()) == []


def test_project_workspace_rejects_workspace_root_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.md").write_text("private", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    workspace_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(PiCopilotError) as raised:
        ProjectWorkspace(workspace_root)

    assert raised.value.code == "pi_workspace_base_root_symlink_blocked"
    assert (outside / "private.md").read_text(encoding="utf-8") == "private"
    assert not (outside / "projects").exists()


def test_project_workspace_allows_stable_declared_ancestor_symlink(
    tmp_path: Path,
) -> None:
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    alias = tmp_path / "easyicu-home"
    alias.symlink_to(relocated, target_is_directory=True)

    workspace = ProjectWorkspace(alias / "workspace")

    assert workspace.project_root("project-a").is_dir()
    assert workspace.base_dir == (relocated / "workspace").resolve()


def test_project_workspace_rejects_retargeted_ancestor_symlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "easyicu-home"
    alias.symlink_to(first, target_is_directory=True)
    workspace = ProjectWorkspace(alias / "workspace")
    workspace.project_root("project-a")

    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)

    with pytest.raises(PiCopilotError) as caught:
        workspace.project_root("project-a")

    assert caught.value.code == "pi_workspace_base_root_changed"


def test_project_workspace_write_is_create_only_and_edit_requires_current_digest(
    tmp_path: Path,
) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    created = workspace.write_file("project-a", "notes.md", "first")

    with pytest.raises(PiCopilotError) as replace_blocked:
        workspace.write_file("project-a", "notes.md", "second")
    assert replace_blocked.value.code == "pi_workspace_write_create_only"
    assert workspace.read_file("project-a", "notes.md")["text"] == "first"

    with pytest.raises(PiCopilotError) as missing_edit:
        workspace.edit_file(
            "project-a", "notes.md", old_text="first", new_text="second"
        )
    assert missing_edit.value.code == "pi_workspace_expected_sha256_required"

    edited = workspace.edit_file(
        "project-a",
        "notes.md",
        old_text="first",
        new_text="second",
        expected_sha256=created["sha256"],
    )
    assert edited["sha256"] != created["sha256"]


def test_truncated_read_digest_cannot_authorize_whole_file_replacement(
    tmp_path: Path,
) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    original = "A" * 100_000
    created = workspace.write_file("project-a", "bundle.js", original)
    read = workspace.read_file("project-a", "bundle.js")

    assert read["truncated"] is True
    assert len(read["text"]) == 24_000
    assert read["sha256"] == created["sha256"]
    with pytest.raises(PiCopilotError) as blocked:
        workspace.write_file("project-a", "bundle.js", read["text"])

    assert blocked.value.code == "pi_workspace_write_create_only"
    assert (workspace.project_root("project-a") / "bundle.js").read_text(
        encoding="utf-8"
    ) == original


def test_project_workspace_compare_and_swap_allows_only_one_concurrent_editor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    created = ProjectWorkspace(root).write_file("project-a", "notes.md", "base")
    barrier = threading.Barrier(2)

    def replace(content: str) -> str:
        workspace = ProjectWorkspace(root)
        barrier.wait(timeout=5)
        try:
            workspace.edit_file(
                "project-a",
                "notes.md",
                old_text="base",
                new_text=content,
                expected_sha256=created["sha256"],
            )
        except PiCopilotError as exc:
            return exc.code
        return "written"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(replace, ("writer-a", "writer-b")))

    assert outcomes == ["pi_workspace_file_changed", "written"]


def test_javascript_checker_uses_a_minimal_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    workspace.write_file("project-a", "app.js", "const ready = true;\n")
    observed: dict[str, object] = {}
    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("NODE_OPTIONS", "--require=/private/inject.js")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-node")
    monkeypatch.setattr(
        "easyicu.webserver.pi_copilot.workspace.shutil.which",
        lambda name: "/safe/bin/node" if name == "node" else None,
    )

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(
        "easyicu.webserver.pi_copilot.workspace.subprocess.run", fake_run
    )

    result = workspace.check_file("project-a", "app.js")

    assert result["valid"] is True
    expected = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "TMPDIR", "LANG"} or key.startswith("LC_")
    }
    assert observed["env"] == expected
    assert "NODE_OPTIONS" not in observed["env"]
    assert "OPENAI_API_KEY" not in observed["env"]


def test_project_workspace_requires_unique_exact_edit_target(tmp_path: Path) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    written = workspace.write_file("project-a", "notes.md", "same\nsame\n")

    with pytest.raises(PiCopilotError) as raised:
        workspace.edit_file(
            "project-a",
            "notes.md",
            old_text="same",
            new_text="changed",
            expected_sha256=written["sha256"],
        )

    assert raised.value.code == "pi_workspace_edit_target_not_unique"
