from __future__ import annotations

import os
import hashlib
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
    checked = workspace.check_file("project-a", "demo/index.html")
    assert checked == {
        "file": "demo/index.html",
        "media_type": "text/html",
        "checker": "html.parser",
        "valid": True,
        "checked_sha256": edited["sha256"],
        "check_scope": "bounded_static_syntax",
        **dict(WORKSPACE_ARTIFACT_AUTHORITY),
    }
    assert "<h1>Ready</h1>" in workspace.preview_file(
        "project-a",
        "demo/index.html",
        checked_sha256=checked["checked_sha256"],
    )["text"]
    assert workspace.list_files("project-a")[0]["file"] == "demo/index.html"


def test_existing_project_root_is_read_only_and_reuses_owned_project(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace = ProjectWorkspace(workspace_root)

    assert workspace.existing_project_root("project-a") is None
    assert not workspace_root.exists()

    created = workspace.project_root("project-a")
    assert workspace.existing_project_root("project-a") == created
    assert workspace.existing_project_root("project-b") is None
    assert not (workspace_root / "projects" / project_workspace_id("project-b")).exists()


def test_preview_refuses_bytes_changed_after_static_check(tmp_path: Path) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    written = workspace.write_file("project-a", "index.html", "<h1>A</h1>")
    checked = workspace.check_file("project-a", "index.html")
    workspace.edit_file(
        "project-a",
        "index.html",
        old_text="<h1>A</h1>",
        new_text="<h1>B</h1>",
        expected_sha256=written["sha256"],
    )

    with pytest.raises(PiCopilotError) as raised:
        workspace.preview_file(
            "project-a",
            "index.html",
            checked_sha256=checked["checked_sha256"],
        )
    assert raised.value.code == "pi_workspace_preview_check_stale"


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


@pytest.mark.parametrize(
    ("occupied_level", "expected_code"),
    [
        ("base", "pi_workspace_base_root_not_directory"),
        ("projects", "pi_workspace_projects_root_not_directory"),
        ("project", "pi_workspace_project_root_not_directory"),
    ],
)
def test_project_workspace_normalizes_directory_occupied_by_file(
    tmp_path: Path,
    occupied_level: str,
    expected_code: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    if occupied_level == "base":
        workspace_root.write_text("occupied", encoding="utf-8")
    else:
        workspace_root.mkdir()
        projects_root = workspace_root / "projects"
        if occupied_level == "projects":
            projects_root.write_text("occupied", encoding="utf-8")
        else:
            projects_root.mkdir()
            (projects_root / project_workspace_id("project-a")).write_text(
                "occupied", encoding="utf-8"
            )

    workspace = ProjectWorkspace(workspace_root)
    with pytest.raises(PiCopilotError) as raised:
        workspace.project_root("project-a")

    assert raised.value.code == expected_code


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
    assert not (second / "workspace").exists()


def test_project_workspace_read_returns_one_text_and_digest_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = ProjectWorkspace(tmp_path / "workspace")
    workspace.write_file("project-a", "notes.md", "first")
    target = workspace.project_root("project-a") / "notes.md"
    original_read_text = Path.read_text

    def mutate_after_text_read(path: Path, *args, **kwargs) -> str:
        text = original_read_text(path, *args, **kwargs)
        if path == target:
            path.write_text("second", encoding="utf-8")
        return text

    monkeypatch.setattr(Path, "read_text", mutate_after_text_read)

    snapshot = workspace.read_file("project-a", "notes.md")

    assert snapshot["sha256"] == hashlib.sha256(
        snapshot["text"].encode("utf-8")
    ).hexdigest()


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


_HOSTED_DATA_FILES = [
    "table.csv", "table.tsv", "table.csv.gz", "table.parquet", "table.feather",
    "table.xlsx", "table.xls", "table.jsonl", "table.ndjson", "table.json",
    "table.yaml", "table.yml", "table.CSV", "table.csv.zip", "table.arrow",
    "exports/note.md", "export/nested/note.txt", "export_synthetic/note.md",
    "runs/run_synthetic/data/note.md", "run_synthetic/note.md", "run/note.md",
    "nested/RUNS/note.md", "data/note.md", "output/note.md", "outputs/note.md",
    "nested\\exports\\note.md", "run-synthetic/note.md",
]


def _workspace_read_tool(tmp_path, relative, text):
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
    from easyicu.webserver.pi_copilot.tools import execute_tool

    workspace = ProjectWorkspace(tmp_path / "workspace")
    candidate = workspace.project_root("synthetic-project") / relative.replace("\\", "/")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(text)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="synthetic-session", project_id="synthetic-project", agent_mode="workspace",
        ),
        workspace=workspace,
    )
    return workspace, lambda: execute_tool("easyicu_read_project_file", {"file": relative}, context)


@pytest.mark.parametrize("relative", _HOSTED_DATA_FILES)
def test_hosted_workspace_data_is_refused_before_read(tmp_path, monkeypatch, relative):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    workspace, read = _workspace_read_tool(tmp_path, relative, "x,y\n1,2")
    with pytest.raises(PiCopilotError) as raised:
        read()
    assert raised.value.code == "tabular_read_unavailable_when_hosted"
    assert raised.value.details["cause_code"] in {"workspace_data_file", "workspace_data_directory"}

    def forbidden(*args, **kwargs):
        raise AssertionError("hosted data bytes were read")

    monkeypatch.setattr(workspace, "_read_complete_file", forbidden)
    with pytest.raises(PiCopilotError) as raised:
        workspace.read_file("synthetic-project", relative)
    assert raised.value.code == "tabular_read_unavailable_when_hosted"


@pytest.mark.parametrize("suffix", [".md", ".txt", ".html", ".htm", ".css", ".js", ".mjs", ".py", ".svg"])
def test_hosted_workspace_permitted_text_is_projected(tmp_path, monkeypatch, suffix):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    _, read = _workspace_read_tool(tmp_path, "notes/plan" + suffix, "Synthetic study plan.")
    result = read()
    assert result["code"] == "pi_workspace_file_read"
    assert result["details"]["text"] == "Synthetic study plan."


@pytest.mark.parametrize("text,code", [
    ("stay_id=123456789", "pi_projection_blocked"),
    ("x" * 2001, "pi_projection_too_large"),
])
def test_hosted_workspace_text_keeps_projection_refusal_cause(tmp_path, monkeypatch, text, code):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    _, read = _workspace_read_tool(tmp_path, "notes/plan.md", text)
    with pytest.raises(PiCopilotError) as raised:
        read()
    assert raised.value.code == code


@pytest.mark.parametrize("relative", _HOSTED_DATA_FILES)
def test_non_hosted_workspace_read_contract_is_unchanged(tmp_path, monkeypatch, relative):
    from easyicu.webserver.pi_copilot.workspace import TEXT_SUFFIXES

    monkeypatch.setenv("EASYICU_HOSTED", "0")
    _, read = _workspace_read_tool(tmp_path, relative, "stay_id=123456789")
    if Path(relative).suffix.lower() in TEXT_SUFFIXES:
        assert read()["details"]["text"] == "stay_id=123456789"
    else:
        with pytest.raises(PiCopilotError) as raised:
            read()
        assert raised.value.code == "pi_workspace_file_type_unsupported"


@pytest.mark.parametrize("hosted", [False, True])
def test_workspace_directory_alias_cannot_bypass_hosted_data_policy(tmp_path, monkeypatch, hosted):
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    workspace, _ = _workspace_read_tool(tmp_path, "exports/table.md", "x,y\n1,2")
    root = workspace.project_root("synthetic-project")
    (root / "notes").symlink_to(root / "exports", target_is_directory=True)
    if not hosted:
        assert workspace.read_file("synthetic-project", "notes/table.md")["text"] == "x,y\n1,2"
        return

    def forbidden(*args, **kwargs):
        raise AssertionError("aliased hosted data bytes were read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(PiCopilotError) as raised:
        workspace.read_file("synthetic-project", "notes/table.md")
    assert raised.value.code == "tabular_read_unavailable_when_hosted"
    assert raised.value.details["cause_code"] == "workspace_data_directory"
