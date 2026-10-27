"""New local projects go to a visible Documents/EasyICU folder, named after them."""

from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import guided_sessions, state_paths
from easyicu.webserver.app import app


@pytest.fixture
def isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "real-home"))
    monkeypatch.delenv("EASYICU_DEPLOYMENT", raising=False)
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", state_paths.projects_root())
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "registry")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "registry" / "guided.json"
    )
    # Without a Documents folder the default is EasyICU in the home itself.
    return state_paths.user_home() / "EasyICU"


def _create(title, **extra):
    result = guided_sessions.create_guided_draft({"title": title, **extra})
    assert result["ok"] is True, result
    return Path(result["draft"]["project_dir"])


def test_guided_list_exposes_default_parent_without_creating_it(isolated_projects):
    result = TestClient(app).post("/api/guided/drafts/list", json={}).json()

    assert result["ok"] is True
    assert result["default_parent_dir"] == str(isolated_projects.resolve())
    assert not isolated_projects.exists()


def test_default_parent_is_documents_easyicu_when_documents_exists(isolated_projects):
    documents = state_paths.user_home() / "Documents"
    documents.mkdir(parents=True)

    project = _create("Synthetic first study")

    assert project == documents / "EasyICU" / "Synthetic first study"
    assert (project / "guided_draft.json").is_file()
    assert not isolated_projects.exists()


def test_desktop_uses_the_os_users_documents_not_the_app_state_home(
    isolated_projects, monkeypatch
):
    # The desktop app points EASYICU_HOME at its private state folder.
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    documents = Path.home() / "Documents"
    documents.mkdir(parents=True)

    project = _create("Synthetic desktop study")

    assert project.parent == documents / "EasyICU"
    assert not (state_paths.user_home() / "EasyICU").exists()


@pytest.mark.parametrize("explicit_parent", [True, False])
def test_guided_create_accepts_absent_default_parent(isolated_projects, explicit_parent):
    extra = {"parent_dir": str(isolated_projects) if explicit_parent else ""}

    project = _create("Synthetic first study", **extra)

    assert project == isolated_projects / "Synthetic first study"


def test_the_old_projects_root_is_still_accepted(isolated_projects):
    old_root = state_paths.projects_root()

    project = _create("Synthetic old-root study", parent_dir=str(old_root))

    assert project == old_root / "Synthetic old-root study"
    listed = guided_sessions.list_guided_drafts()["drafts"]
    assert [row["project_dir"] for row in listed] == [str(project)]


@pytest.mark.parametrize(
    "title,folder",
    [
        ("脓毒症 乳酸 与死亡", "脓毒症 乳酸 与死亡"),
        ("AKI / CKD: 30-day <risk>?", "AKI CKD 30-day risk"),
        ('  ..Trailing dots and "quotes"..  ', 'Trailing dots and quotes'),
        ("tab\tand\nnewline", "tab and newline"),
        ("CON", "CON_"),
        ("lpt1.study", "lpt1.study_"),
        ("x" * 120, "x" * 80),
        ("///", "New study"),
    ],
)
def test_folder_is_named_after_the_title_and_safe_on_every_os(
    isolated_projects, title, folder
):
    project = _create(title)

    assert project == isolated_projects / folder


def test_a_name_collision_gets_a_number(isolated_projects):
    first = _create("脓毒症研究")
    second = _create("脓毒症研究")
    (isolated_projects / "脓毒症研究 3").mkdir()
    fourth = _create("脓毒症研究")

    assert [first.name, second.name, fourth.name] == ["脓毒症研究", "脓毒症研究 2", "脓毒症研究 4"]
    # An existing folder with that name is never adopted.
    assert not (isolated_projects / "脓毒症研究 3" / "guided_draft.json").exists()


def test_a_registered_project_whose_folder_is_gone_still_reserves_its_name(isolated_projects):
    first = _create("Synthetic study")
    shutil.rmtree(first)

    second = _create("Synthetic study")

    assert second.name == "Synthetic study 2"


def test_the_default_parent_itself_is_never_moved_to_the_trash(isolated_projects):
    project = _create("Synthetic study")
    marker = project / "guided_draft.json"
    shutil.copy(marker, isolated_projects / "guided_draft.json")

    with pytest.raises(guided_sessions.GuidedProjectTrashError) as refused:
        guided_sessions._trash_guided_project_folder(str(isolated_projects), "any")

    assert refused.value.code == "project_folder_scope_not_allowed"


def test_hosted_projects_keep_the_instance_projects_root(isolated_projects, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    (state_paths.user_home() / "Documents").mkdir(parents=True)

    project = _create("Synthetic hosted study")

    assert project.parent == state_paths.projects_root()


def test_guided_create_refuses_absent_real_home_parent(isolated_projects):
    real_home_parent = Path.home() / "easyicu" / "projects"
    assert not real_home_parent.exists()

    result = guided_sessions.create_guided_draft(
        {"title": "Synthetic invalid parent", "parent_dir": "~/easyicu/projects"}
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_guided_parent_dir"
    assert not real_home_parent.exists()
    assert not isolated_projects.exists()


def test_folder_browse_does_not_create_absent_default(isolated_projects):
    result = TestClient(app).get(
        "/api/fs/list", params={"path": str(isolated_projects)}
    ).json()

    assert result["ok"] is False
    assert result["error"] == "not_a_directory"
    assert result["path"] == str(isolated_projects)
    assert result["entries"] == []
    assert not isolated_projects.exists()


def test_guided_folder_dialog_node_contract():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, str(root / "tests/js/guided_folder_dialog.test.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
