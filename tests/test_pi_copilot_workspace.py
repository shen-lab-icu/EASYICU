from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace


def test_project_workspace_writes_reads_edits_checks_and_previews(tmp_path: Path) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")

    written = workspace.write_file(
        "project-a",
        "demo/index.html",
        "<!doctype html><title>Demo</title><h1>Draft</h1>",
    )
    assert written["file"] == "demo/index.html"
    assert written["created"] is True

    read = workspace.read_file("project-a", "demo/index.html")
    assert read["text"].endswith("<h1>Draft</h1>")
    assert read["total_lines"] == 1

    edited = workspace.edit_file(
        "project-a",
        "demo/index.html",
        old_text="<h1>Draft</h1>",
        new_text="<h1>Ready</h1>",
    )
    assert edited["replacements"] == 1
    assert workspace.check_file("project-a", "demo/index.html") == {
        "file": "demo/index.html",
        "media_type": "text/html",
        "checker": "html.parser",
        "valid": True,
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


def test_project_workspace_requires_unique_exact_edit_target(tmp_path: Path) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    workspace.write_file("project-a", "notes.md", "same\nsame\n")

    with pytest.raises(PiCopilotError) as raised:
        workspace.edit_file(
            "project-a",
            "notes.md",
            old_text="same",
            new_text="changed",
        )

    assert raised.value.code == "pi_workspace_edit_target_not_unique"
