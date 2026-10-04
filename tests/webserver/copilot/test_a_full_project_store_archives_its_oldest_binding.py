"""A full project authority store archives its oldest binding.

The store mapping each research project to its StudyContext held at most 200
bindings and never released one. Projects leave the project list without
telling it, so after 200 projects every new project was refused with
``pi_project_authority_capacity_reached``; in use, 129 of the 200 bindings
belonged to projects no longer listed. A new binding now moves the oldest
binding to an archive file beside the store. Lookups and both refusals read
the two files, and reopening an archived project moves its binding back.

Synthetic identifiers only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.webserver.pi_copilot import project_authority
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.project_authority import ProjectAuthorityStore

ACTIVE_SCHEMA = "easyicu.pi-project-authority/1"
ARCHIVE_SCHEMA = "easyicu.pi-project-authority-archive/1"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(project_authority, "_MAX_PROJECTS", 3)
    return ProjectAuthorityStore(tmp_path / "pi_project_authority.json")


def _bind_projects(store, *numbers):
    for number in numbers:
        store.bind(f"project_{number}", f"study_{number}")


def _ids(path):
    return [row["project_id"] for row in json.loads(path.read_text())["bindings"]]


def _write(path, schema, rows):
    path.write_text(
        json.dumps({"schema_version": schema, "updated_at": "", "bindings": rows})
    )


def test_a_full_store_archives_its_oldest_binding_and_admits_a_new_project(store):
    _bind_projects(store, 1, 2, 3)

    assert store.bind("project_4", "study_4") == "study_4"

    assert _ids(store.path) == ["project_4", "project_3", "project_2"]
    assert _ids(store.archive_path) == ["project_1"]
    # The archived project still owns its StudyContext.
    assert store.resolve("project_1") == "study_1"
    assert sorted(row.project_id for row in store.bindings()) == [
        "project_1",
        "project_2",
        "project_3",
        "project_4",
    ]


def test_reopening_an_archived_project_moves_its_binding_back(store):
    _bind_projects(store, 1, 2, 3, 4)

    # Opening a project binds it again with the context it already owns.
    assert store.bind("project_1", "study_1") == "study_1"

    assert _ids(store.path) == ["project_1", "project_4", "project_3"]
    assert _ids(store.archive_path) == ["project_2"]
    assert store.resolve("project_2") == "study_2"


@pytest.mark.parametrize(
    "project, study, code",
    [
        ("project_1", "study_9", "pi_project_study_context_mismatch"),
        ("project_9", "study_1", "pi_study_context_project_mismatch"),
    ],
)
def test_an_archived_binding_still_refuses_a_second_owner(store, project, study, code):
    _bind_projects(store, 1, 2, 3, 4)  # project_1 is archived

    with pytest.raises(PiCopilotError) as caught:
        store.bind(project, study)

    assert caught.value.code == code
    assert store.resolve("project_1") == "study_1"


def test_a_store_filled_in_use_admits_a_new_project(tmp_path):
    # The shape the old code left: the full count, newest first, no archive.
    path = tmp_path / "pi_project_authority.json"
    _write(
        path,
        ACTIVE_SCHEMA,
        [
            {
                "project_id": f"draft_{number:03d}",
                "study_context_id": f"ctx_{number:03d}",
                "created_at": "2026-08-01T00:00:00+00:00",
            }
            for number in range(200, 0, -1)
        ],
    )
    store = ProjectAuthorityStore(path)

    assert store.bind("draft_new", "ctx_new") == "ctx_new"

    active = _ids(path)
    assert len(active) == 200 and active[0] == "draft_new"
    assert _ids(store.archive_path) == ["draft_001"]
    assert store.resolve("draft_001") == "ctx_001"


def test_a_binding_left_in_both_files_by_an_interrupted_move_counts_once(store):
    _bind_projects(store, 1, 2, 3, 4)
    rows = {row.project_id: row.model_dump(mode="json") for row in store.bindings()}
    # A restoration of project_1 that stopped after its active write.
    _write(store.path, ACTIVE_SCHEMA, [rows[f"project_{n}"] for n in (1, 4, 3)])
    _write(store.archive_path, ARCHIVE_SCHEMA, [rows[f"project_{n}"] for n in (2, 1)])

    assert sorted(row.project_id for row in store.bindings()) == [
        "project_1",
        "project_2",
        "project_3",
        "project_4",
    ]
    store.bind("project_5", "study_5")
    # The next archive write drops the leftover copy.
    assert _ids(store.archive_path) == ["project_3", "project_2"]


def test_two_different_bindings_for_one_project_make_the_store_invalid(store):
    _bind_projects(store, 1, 2, 3)
    _write(
        store.archive_path,
        ARCHIVE_SCHEMA,
        [{"project_id": "project_1", "study_context_id": "study_9", "created_at": ""}],
    )

    with pytest.raises(PiCopilotError) as caught:
        store.bindings()

    assert caught.value.code == "pi_project_authority_store_invalid"


def test_a_lookup_racing_a_restoration_still_finds_the_binding(store, monkeypatch):
    _bind_projects(store, 1, 2, 3, 4)  # project_1 is archived
    writer = ProjectAuthorityStore(store.path)
    read_archive = ProjectAuthorityStore._read_archive
    raced = []

    def restore_before_the_archive_read(self):
        if self is store and not raced:
            # Between the reader's active read and its archive read, another
            # process reopens project_1: active gains it, the archive loses it.
            raced.append(True)
            writer.bind("project_1", "study_1")
        return read_archive(self)

    monkeypatch.setattr(
        ProjectAuthorityStore, "_read_archive", restore_before_the_archive_read
    )

    assert store.resolve("project_1") == "study_1"
    assert raced


def test_a_full_archive_still_refuses_with_the_capacity_code(store, monkeypatch):
    monkeypatch.setattr(project_authority, "_MAX_ARCHIVED_PROJECTS", 1)
    _bind_projects(store, 1, 2, 3, 4)  # the archive now holds its one binding

    with pytest.raises(PiCopilotError) as caught:
        store.bind("project_5", "study_5")

    assert caught.value.code == "pi_project_authority_capacity_reached"
    # Nothing moved.
    assert _ids(store.path) == ["project_4", "project_3", "project_2"]
    assert _ids(store.archive_path) == ["project_1"]


@pytest.mark.parametrize(
    "reopened, moved",
    [
        (None, "project_1"),  # a new project evicts project_1 to the archive
        ("project_1", "project_1"),  # reopening project_1 brings it back
    ],
)
def test_a_move_whose_second_write_fails_keeps_the_binding(
    store, monkeypatch, reopened, moved
):
    _bind_projects(store, 1, 2, 3)
    if reopened:
        store.bind("project_4", "study_4")  # project_1 is archived
    write_rows = ProjectAuthorityStore._write_rows
    writes = []

    def fail_the_second_write(path, *args):
        writes.append(path)
        if len(writes) == 2:
            raise OSError("interrupted")
        return write_rows(path, *args)

    monkeypatch.setattr(
        ProjectAuthorityStore, "_write_rows", staticmethod(fail_the_second_write)
    )
    with pytest.raises(OSError):
        if reopened:
            store.bind(reopened, "study_1")
        else:
            store.bind("project_4", "study_4")

    # Whichever file the move reached first already holds the binding.
    assert store.resolve(moved) == "study_1"


def test_a_restoration_with_room_whose_archive_write_fails_keeps_the_binding(
    store, monkeypatch
):
    rows = [
        {
            "project_id": f"project_{n}",
            "study_context_id": f"study_{n}",
            "created_at": "",
        }
        for n in (1, 2, 3)
    ]
    _write(store.path, ACTIVE_SCHEMA, rows[1:])
    _write(store.archive_path, ARCHIVE_SCHEMA, rows[:1])
    write_rows = ProjectAuthorityStore._write_rows

    def fail_the_archive_write(path, *args):
        if path == store.archive_path:
            raise OSError("interrupted")
        return write_rows(path, *args)

    monkeypatch.setattr(
        ProjectAuthorityStore, "_write_rows", staticmethod(fail_the_archive_write)
    )
    with pytest.raises(OSError):
        store.bind("project_1", "study_1")

    assert _ids(store.path)[0] == "project_1"
    assert store.resolve("project_1") == "study_1"
