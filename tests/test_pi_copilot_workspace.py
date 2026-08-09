from __future__ import annotations

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


def test_project_workspace_requires_current_digest_for_replacement_and_edit(
    tmp_path: Path,
) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    created = workspace.write_file("project-a", "notes.md", "first")

    with pytest.raises(PiCopilotError) as missing_write:
        workspace.write_file("project-a", "notes.md", "second")
    assert missing_write.value.code == "pi_workspace_expected_sha256_required"

    replaced = workspace.write_file(
        "project-a",
        "notes.md",
        "second",
        expected_sha256=created["sha256"],
    )
    with pytest.raises(PiCopilotError) as stale_write:
        workspace.write_file(
            "project-a",
            "notes.md",
            "third",
            expected_sha256=created["sha256"],
        )
    assert stale_write.value.code == "pi_workspace_file_changed"

    with pytest.raises(PiCopilotError) as missing_edit:
        workspace.edit_file(
            "project-a", "notes.md", old_text="second", new_text="third"
        )
    assert missing_edit.value.code == "pi_workspace_expected_sha256_required"

    edited = workspace.edit_file(
        "project-a",
        "notes.md",
        old_text="second",
        new_text="third",
        expected_sha256=replaced["sha256"],
    )
    assert edited["sha256"] != replaced["sha256"]


def test_project_workspace_compare_and_swap_allows_only_one_concurrent_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    created = ProjectWorkspace(root).write_file("project-a", "notes.md", "base")
    barrier = threading.Barrier(2)

    def replace(content: str) -> str:
        workspace = ProjectWorkspace(root)
        barrier.wait(timeout=5)
        try:
            workspace.write_file(
                "project-a",
                "notes.md",
                content,
                expected_sha256=created["sha256"],
            )
        except PiCopilotError as exc:
            return exc.code
        return "written"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(replace, ("writer-a", "writer-b")))

    assert outcomes == ["pi_workspace_file_changed", "written"]


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
