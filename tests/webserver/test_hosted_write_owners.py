"""Indirect metadata writers must not follow coordinates into shared data."""

import json
from functools import partial
from pathlib import Path

import pytest
from fastapi import HTTPException

from easyicu.webserver import copilot_sessions, guided_sessions, hosted
from easyicu.webserver.pi_copilot.session_storage import SessionStorageMaintenance
from tests.support.hosted_http import assert_hosted_denied as denied
from tests.support.hosted_http import hosted_boundary as _hosted_boundary  # noqa: F401


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("phase", ["create", "message"])
def test_legacy_copilot_project_writes_are_instance_only(boundary, monkeypatch, enabled, phase):
    root, home = boundary
    monkeypatch.setattr(copilot_sessions, "_CONFIG_DIR", home)
    monkeypatch.setattr(copilot_sessions, "_CONFIG_PATH", home / "copilot.json")
    monkeypatch.setattr(copilot_sessions, "_PROJECTS_ROOT", root)
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    if phase == "message":
        session = copilot_sessions.create_session({})["session"]
        project = Path(session["project_dir"])
        before = (project / "page_guide_session.json").read_bytes()
        action = partial(copilot_sessions.post_message, {"session_id": session["id"], "message": "synthetic"})
    else:
        action = partial(copilot_sessions.create_session, {})
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            action()
        denied(error.value)
        if phase == "message":
            assert (project / "page_guide_session.json").read_bytes() == before
        else:
            assert list(root.iterdir()) == []
    else:
        assert action()["ok"]


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("child", ["project", "guided_draft.json"])
def test_guided_creation_rechecks_the_final_child(boundary, monkeypatch, enabled, child):
    root, home = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    monkeypatch.setattr(guided_sessions, "_draft_id", lambda value: "draft_synthetic")
    project = guided_sessions._project_dir_for("draft_synthetic", "Synthetic")
    project.parent.mkdir(parents=True)
    if child == "project":
        project.symlink_to(root, target_is_directory=True)
    else:
        project.mkdir()
        (project / child).symlink_to(root / "new.json")
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            guided_sessions.create_guided_draft({"title": "Synthetic"})
        denied(error.value)
        assert list(root.iterdir()) == []
    else:
        assert guided_sessions.create_guided_draft({"title": "Synthetic"})["ok"]
        assert list(root.iterdir())


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("child", ["project_notes.md", ".project_notes.md.tmp"])
def test_notes_recheck_both_write_targets_before_writing(boundary, monkeypatch, enabled, child):
    root, home = boundary
    project = home / "projects" / "synthetic"
    project.mkdir(parents=True)
    guided_sessions._CONFIG_PATH.write_text(json.dumps({"drafts": [{"id": "synthetic", "project_dir": str(project)}]}))
    (project / child).symlink_to(root / "notes.md")
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            guided_sessions.write_project_notes("synthetic", "Synthetic notes")
        denied(error.value)
        assert list(root.iterdir()) == []
        assert not (project / ".project_notes.md.tmp").is_file()
    else:
        assert guided_sessions.write_project_notes("synthetic", "Synthetic notes")["ok"]


@pytest.mark.parametrize("enabled", [True, False])
def test_hosted_project_removal_does_not_use_the_unix_users_trash(boundary, monkeypatch, enabled):
    _, home = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    draft = guided_sessions.create_guided_draft({"title": "Synthetic"})["draft"]
    calls = []
    monkeypatch.setattr(guided_sessions, "_send2trash", calls.append)
    if enabled:
        with pytest.raises(HTTPException) as error:
            guided_sessions._trash_guided_project_folder(draft["project_dir"], draft["id"])
        denied(error.value, "unavailable_when_hosted")
        assert calls == []
    else:
        guided_sessions._trash_guided_project_folder(draft["project_dir"], draft["id"])
        assert calls == [draft["project_dir"]]
    assert Path(draft["project_dir"]).is_relative_to(home)


@pytest.mark.parametrize("enabled", [True, False])
def test_quarantine_destination_cannot_redirect_to_shared_data(boundary, monkeypatch, enabled):
    root, home = boundary
    owner = SessionStorageMaintenance(home / "sessions", grace_seconds=60)
    owner.session_dir.mkdir()
    transcript = owner.session_dir / "synthetic.jsonl"
    transcript.write_text("{}\n")
    owner.quarantine_root.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            owner.quarantine([], confirm=True, now=transcript.stat().st_mtime + 100)
        denied(error.value)
        assert transcript.is_file() and list(root.iterdir()) == []
    else:
        assert owner.quarantine([], confirm=True, now=transcript.stat().st_mtime + 100)["moved_files"] == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_scratch_parent_stays_in_instance_home_only_when_hosted(boundary, monkeypatch, enabled):
    _, home = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    parent = hosted.instance_scratch_dir()
    if enabled:
        assert parent.is_relative_to(home) and parent.is_dir()
    else:
        assert parent is None


@pytest.mark.parametrize("enabled", [True, False])
def test_scratch_parent_cannot_link_into_shared_data(boundary, monkeypatch, enabled):
    root, home = boundary
    scratch = home / ".easyicu" / "tmp"
    scratch.parent.mkdir()
    scratch.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            hosted.instance_scratch_dir()
        denied(error.value)
    else:
        assert hosted.instance_scratch_dir() is None
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("enabled", [True, False])
def test_export_download_destination_is_instance_only(boundary, monkeypatch, enabled):
    from easyicu.webserver.export_download import prepare_registered_export_bundle

    root, _ = boundary
    (root / "_manifest.json").write_text(json.dumps({"files": [{"file": "synthetic.csv"}]}))
    (root / "synthetic.csv").write_text("synthetic\n1\n")
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    registry = {"sources": [{"id": "src_0123456789ab", "path": str(root), "ok": True}]}
    kwargs = dict(source_id="src_0123456789ab", registry=registry, temp_dir=root / "downloads")
    if enabled:
        with pytest.raises(HTTPException) as error:
            prepare_registered_export_bundle(**kwargs)
        denied(error.value)
        assert not (root / "downloads").exists()
    else:
        assert prepare_registered_export_bundle(**kwargs).path.is_file()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("stage", ["build", "execute"])
def test_demo_pipeline_refuses_before_any_preparation_operation(boundary, monkeypatch, enabled, stage):
    from types import SimpleNamespace
    from easyicu.webserver import demo_source_prepare as prepare
    from easyicu.webserver.jobs import Job

    source = SimpleNamespace(id="synthetic")
    calls = []
    monkeypatch.setenv("EASYICU_HOSTED", "0" if stage == "execute" else ("1" if enabled else "0"))
    job = Job("synthetic", "demo-source-prepare")

    def reached(payload):
        calls.append(payload)
        raise RuntimeError("synthetic runner reached")

    monkeypatch.setattr(job, "emit", reached)
    # Accessing the title happens before emit, so supply the small public catalog.
    source.title, source.version, source.scope_summary, source.size_bytes = "Synthetic", "1", "test", 0
    if stage == "execute":
        runner = prepare.build_prepare_runner(source, None, None)
        monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
        action = partial(runner, job)
    else:
        action = partial(prepare.build_prepare_runner, source, None, None)
    if enabled:
        with pytest.raises(HTTPException) as error:
            action()
        denied(error.value, "unavailable_when_hosted")
        assert calls == []
    elif stage == "execute":
        with pytest.raises(RuntimeError, match="synthetic runner"):
            action()
        assert calls
    else:
        assert callable(action())


@pytest.mark.parametrize("owner", ["resume", "trial"])
@pytest.mark.parametrize("enabled", [True, False])
def test_request_scratch_owners_choose_the_instance_parent(boundary, monkeypatch, owner, enabled):
    import tempfile
    from types import SimpleNamespace
    from easyicu.webserver import research_prepared_resume, target_trial_setup
    from easyicu.webserver import research_pipeline_run_preparation

    _, home = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    calls = []

    def temporary(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic scratch reached")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", temporary)
    monkeypatch.setattr(research_pipeline_run_preparation, "prepare_target_trial_launch", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="synthetic scratch"):
        if owner == "resume":
            research_prepared_resume._rehearse_grouped_continuation(None, None, None)
        else:
            target_trial_setup._compile(None, study={"id": "synthetic"},
                statement=SimpleNamespace(spec=None, population_spec=None),
                export_path="synthetic", database="miiv", revision=1)
    assert len(calls) == 1
    assert calls[0].get("dir") == (home / ".easyicu" / "tmp" if enabled else None)


@pytest.mark.parametrize("enabled", [True, False])
def test_idea_project_seed_cannot_be_written_to_shared_data(boundary, monkeypatch, enabled):
    from easyicu.webserver.ideas import mining

    root, home = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    monkeypatch.setattr(mining, "_AGENT_PROJECTS_ROOT", root)
    monkeypatch.setattr(mining, "_AGENT_PROJECTS_PATH", home / "idea-projects.json")
    monkeypatch.setattr(mining, "_load_handoff", lambda _: {"run_id": "synthetic", "idea_id": "idea"})
    monkeypatch.setattr(mining, "is_legacy_handoff_envelope", lambda _: False)
    monkeypatch.setattr(mining, "load_validated_canonical_handoff", lambda *a, **k: object())
    monkeypatch.setattr(mining, "_handoff_needs_refresh", lambda *a: False)
    monkeypatch.setattr(mining, "_handoff_plan_is_stale", lambda *a: False)
    monkeypatch.setattr(mining, "_agent_project_seed", lambda *a: {"study_id": "synthetic"})
    if enabled:
        with pytest.raises(HTTPException) as error:
            mining.create_agent_project({"run_id": "synthetic", "idea_id": "idea"})
        denied(error.value)
        assert list(root.iterdir()) == []
        assert not mining._AGENT_PROJECTS_PATH.exists()
    else:
        assert mining.create_agent_project({"run_id": "synthetic", "idea_id": "idea"})["ok"]
        assert (root / "synthetic" / "project_seed.json").is_file()


@pytest.mark.parametrize("owner", ["preflight", "pipeline"])
@pytest.mark.parametrize("symlink", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_run_receipt_writers_check_the_file_destination(boundary, monkeypatch, owner, symlink, enabled):
    from easyicu.webserver import agent_runs, agent_pipeline_runs

    root, home = boundary
    target = root / "receipt.json"
    target.write_text("{}")
    path = target
    if symlink:
        path = home / "receipt.json"
        path.symlink_to(target)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    writer = agent_runs if owner == "preflight" else agent_pipeline_runs
    if enabled:
        with pytest.raises(HTTPException) as error:
            writer._write_json(path, {"synthetic": True})
        denied(error.value)
        assert target.read_text() == "{}"
    else:
        writer._write_json(path, {"synthetic": True})
        assert json.loads(target.read_text()) == {"synthetic": True}


@pytest.mark.parametrize("enabled", [True, False])
def test_new_guided_session_checks_its_generated_directory_before_mkdir(boundary, monkeypatch, enabled):
    root, home = boundary
    link = home / "linked-projects"
    link.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", link)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            guided_sessions._persist_session({"id": "synthetic"})
        denied(error.value)
        assert list(root.iterdir()) == []
    else:
        guided_sessions._persist_session({"id": "synthetic"})
        assert list(root.glob("*/guided_copilot_session.json"))


@pytest.mark.parametrize("enabled", [True, False])
def test_restore_checks_destination_even_when_the_manifest_is_inside_home(boundary, monkeypatch, enabled):
    root, home = boundary
    owner = SessionStorageMaintenance(root)
    batch = "20261021T000000Z-01234567"
    selected = home / "quarantine" / batch
    selected.mkdir(parents=True)
    transcript = selected / "synthetic.jsonl"
    transcript.write_text("{}\n")
    (selected / "manifest.json").write_text(json.dumps({
        "schema_version": "easyicu.pi-session-quarantine/1", "quarantine_id": batch,
        "files": [{"file": "synthetic.jsonl"}],
    }))
    owner.quarantine_root.symlink_to(selected.parent, target_is_directory=True)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            owner.restore(batch, confirm=True)
        denied(error.value)
        assert transcript.is_file() and not (root / transcript.name).exists()
    else:
        assert owner.restore(batch, confirm=True)["restored_files"] == 1
        assert (root / transcript.name).is_file()
