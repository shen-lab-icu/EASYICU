"""Shared prepared data is input only; all fixtures below are synthetic."""

import json
from functools import partial
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from easyicu.webserver import (
    agent_runs, dataio, demo_sources, guided_sessions, hosted, settings, sources,
)
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining
from easyicu.webserver.jobs import Job
from easyicu.webserver.pi_copilot.session_storage import SessionStorageMaintenance
from easyicu.webserver.routes import agent as agent_routes
from easyicu.webserver.routes import jobs as job_routes
from easyicu.webserver.run_record import RunDirectory
from tests.support.hosted_http import assert_hosted_denied as denied, stub_route
from tests.support.hosted_http import hosted_boundary as _hosted_boundary  # noqa: F401


# Keep the feature handler inert: refusal must precede even its first call.
WRITE_ROUTES = [
    ("/api/jobs/extract", "out_dir", "hosted_data_read_only"),
    ("/api/jobs/convert", "path", "unavailable_when_hosted"),
    ("/api/fs/mkdir", "path", "unavailable_when_hosted"),
    ("/api/jobs/demo-source-prepare", None, "unavailable_when_hosted"),
    ("/api/guided/drafts", "parent_dir", "hosted_data_read_only"),
    ("/api/guided/project/open", "project_dir", "hosted_data_read_only"),
    ("/api/guided/session", "project_dir", "hosted_data_read_only"),
    ("/api/jobs/agent-run", "project_root", "hosted_data_read_only"),
    ("/api/jobs/agent-run", "project_seed_dir", "hosted_data_read_only"),
    ("/api/jobs/agent-run", "project_seed_path", "hosted_data_read_only"),
    ("/api/agent-runs/signoff", "project_dir", "hosted_data_read_only"),
    ("/api/agent-runs/review", "project_dir", "hosted_data_read_only"),
    ("/api/agent-runs/history", "project_root", "hosted_data_read_only"),
]


@pytest.mark.parametrize("route,field,code", WRITE_ROUTES)
@pytest.mark.parametrize("enabled", [True, False])
def test_shared_write_coordinates_stop_before_dispatch(boundary, monkeypatch, route, field, code, enabled):
    root, _ = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    calls = stub_route(monkeypatch, route)
    body = {field: str(root / "target")} if field else {"source_id": "mimic-iv-demo"}
    response = TestClient(app).post(route, json=body)
    if enabled:
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["error"] == code
        assert calls == []
    else:
        assert response.status_code == 200
        assert len(calls) == 1
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("kind", ["shared", "symlink", "dotdot", "prefix", "relative"])
@pytest.mark.parametrize("enabled", [True, False])
def test_instance_policy_resolves_the_actual_destination(boundary, monkeypatch, kind, enabled):
    root, home = boundary
    (home / "link").symlink_to(root, target_is_directory=True)
    candidate = {
        "shared": root / "target", "symlink": home / "link" / "target",
        "dotdot": home / ".." / root.name / "target",
        "prefix": home.with_name(home.name + "-other") / "target", "relative": Path("relative"),
    }[kind]
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            hosted.require_instance_path(candidate)
        denied(error.value)
    else:
        hosted.require_instance_path(candidate)
    assert not candidate.exists()


@pytest.mark.parametrize("home_value", [None, "", "relative", "/"])
def test_instance_writes_need_an_explicit_bounded_home(boundary, monkeypatch, home_value):
    root, _ = boundary
    if home_value is None:
        monkeypatch.delenv("EASYICU_HOME")
    else:
        monkeypatch.setenv("EASYICU_HOME", home_value)
    with pytest.raises(HTTPException) as error:
        hosted.require_instance_path(root / "new")
    denied(error.value)


def test_instance_home_needs_no_data_roots_and_does_not_open_the_picker(boundary, monkeypatch):
    _, home = boundary
    monkeypatch.delenv("EASYICU_DATA_ROOTS")
    hosted.require_instance_path(home / "new")
    assert TestClient(app).get("/api/fs/list", params={"path": str(home)}).status_code == 403
    for route, field in [("/api/jobs/extract", "out_dir"), ("/api/guided/drafts", "parent_dir")]:
        calls = stub_route(monkeypatch, route)
        assert TestClient(app).post(route, json={field: str(home / "new")}).status_code == 200
        assert len(calls) == 1


@pytest.mark.parametrize("stage", ["build", "execute", "destination"])
@pytest.mark.parametrize("enabled", [True, False])
def test_export_owner_checks_before_loading_or_creating(boundary, monkeypatch, stage, enabled):
    root, _ = boundary
    target = root / "export"
    reached = []

    def quality(path):
        reached.append(path)
        raise RuntimeError("synthetic input owner reached")

    monkeypatch.setattr(dataio, "_conversion_quality_for_path", quality)
    monkeypatch.setenv("EASYICU_HOSTED", "0" if stage == "execute" else ("1" if enabled else "0"))
    kwargs = dict(data_path=str(root), database="miiv", out_dir=str(target))
    if stage == "execute":
        runner = dataio.make_export_runner(**kwargs)
        monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
        action = partial(runner, Job("synthetic", "extract"))
    elif stage == "destination":
        action = partial(dataio._resolve_export_out_dir,
            out_dir=str(target), database="miiv", export_format="csv", create_run_subdir=True,
        )
    else:
        action = partial(dataio.make_export_runner, **kwargs)
    if enabled:
        with pytest.raises(HTTPException) as error:
            action()
        denied(error.value)
        assert reached == []
    elif stage == "execute":
        with pytest.raises(RuntimeError, match="synthetic input owner"):
            action()
        assert reached == [root]
    else:
        action()
    assert not target.exists()


@pytest.mark.parametrize("owner", ["convert", "mkdir", "demo"])
@pytest.mark.parametrize("enabled", [True, False])
def test_preparation_owners_are_administrator_only(boundary, monkeypatch, owner, enabled):
    root, _ = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    calls = []
    monkeypatch.setattr(demo_sources, "get_source", lambda value: calls.append(value))
    if owner == "demo":
        # A sentinel at the very first catalog lookup also prevents downloads.
        def source(value):
            calls.append(value)
            raise RuntimeError("synthetic catalog reached")
        monkeypatch.setattr(demo_sources, "get_source", source)
        action = partial(demo_sources.make_prepare_runner, "synthetic")
    elif owner == "convert":
        action = partial(dataio.make_convert_runner, str(root), "miiv")
    else:
        action = partial(dataio.create_dir, str(root / "new"))
    if enabled:
        with pytest.raises(HTTPException) as error:
            action()
        denied(error.value, "unavailable_when_hosted")
        assert calls == []
        assert list(root.iterdir()) == []
    elif owner == "demo":
        with pytest.raises(RuntimeError, match="synthetic catalog"):
            action()
        assert calls == ["synthetic"]
    elif owner == "mkdir":
        assert action()["ok"] and (root / "new").is_dir()
    else:
        assert callable(action())


def test_convert_rechecks_at_execution_before_constructing_converter(boundary, monkeypatch):
    from easyicu.io.data_converter import DataConverter

    root, _ = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    runner = dataio.make_convert_runner(str(root), "miiv")
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    calls = []
    monkeypatch.setattr(DataConverter, "__init__", lambda *a, **k: calls.append(k))
    with pytest.raises(HTTPException) as error:
        runner(Job("synthetic", "convert"))
    denied(error.value, "unavailable_when_hosted")
    assert calls == [] and list(root.iterdir()) == []


@pytest.mark.parametrize("explicit", [True, False])
def test_queued_export_rechecks_a_retargeted_destination(boundary, monkeypatch, explicit):
    root, home = boundary
    target = home / "results" if explicit else home / ".easyicu" / "exports"
    runner = dataio.make_export_runner(
        data_path=str(root), database="miiv", out_dir=str(target) if explicit else None,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(root, target_is_directory=True)
    reads = []
    monkeypatch.setattr(dataio, "_conversion_quality_for_path", lambda path: reads.append(path))
    with pytest.raises(HTTPException) as error:
        runner(Job("synthetic", "extract"))
    denied(error.value)
    assert reads == [] and list(root.iterdir()) == []


@pytest.mark.parametrize("owner", ["run-resolve", "run-create", "history", "seed", "project", "parent", "session", "idea"])
@pytest.mark.parametrize("enabled", [True, False])
def test_project_and_receipt_owners_never_use_shared_roots(boundary, monkeypatch, owner, enabled):
    root, _ = boundary
    (root / "project_seed.json").write_text("{}")
    (root / "guided_draft.json").write_text("{}")
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    monkeypatch.setattr(mining, "_RUN_ROOT", root)
    actions = {
        "run-resolve": lambda: RunDirectory.resolve(root),
        "run-create": lambda: RunDirectory.create(root, "synthetic", "job"),
        "history": lambda: agent_runs.list_run_history(project_root=str(root)),
        "seed": lambda: agent_routes._validate_agent_project_seed_for_run(str(root), "synthetic"),
        "project": lambda: guided_sessions._safe_project_dir(root, allow_marked_external=True),
        "parent": lambda: guided_sessions._safe_project_parent_dir(root),
        "session": lambda: guided_sessions._session_file(root),
        "idea": lambda: mining._run_dir("synthetic"),
    }
    reads = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.is_relative_to(root):
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    if enabled:
        with pytest.raises(HTTPException) as error:
            actions[owner]()
        denied(error.value)
        assert reads == []
    else:
        assert actions[owner]() is not None
    assert sorted(p.name for p in root.iterdir()) == ["guided_draft.json", "project_seed.json"]


@pytest.mark.parametrize("enabled", [True, False])
def test_guided_draft_shared_parent_does_not_create_files(boundary, monkeypatch, enabled):
    root, _ = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    if enabled:
        with pytest.raises(HTTPException) as error:
            guided_sessions.create_guided_draft({"title": "Synthetic", "parent_dir": str(root)})
        denied(error.value)
        assert list(root.iterdir()) == []
    else:
        result = guided_sessions.create_guided_draft({"title": "Synthetic", "parent_dir": str(root)})
        assert result["ok"]
        assert (Path(result["draft"]["project_dir"]) / "guided_draft.json").is_file()


@pytest.mark.parametrize("operation", ["inventory", "quarantine", "restore"])
@pytest.mark.parametrize("enabled", [True, False])
def test_session_storage_is_instance_only(boundary, monkeypatch, operation, enabled):
    root, _ = boundary
    owner = SessionStorageMaintenance(root)
    batch = "20261021T000000Z-01234567"
    selected = owner.quarantine_root / batch
    selected.mkdir(parents=True)
    (selected / "manifest.json").write_text(json.dumps({
        "schema_version": "easyicu.pi-session-quarantine/1", "quarantine_id": batch, "files": [],
    }))
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    action = {
        "inventory": lambda: owner.inventory([]),
        "quarantine": lambda: owner.quarantine([], confirm=True),
        "restore": lambda: owner.restore(batch, confirm=True),
    }[operation]
    if enabled:
        with pytest.raises(HTTPException) as error:
            action()
        denied(error.value)
    else:
        action()
    assert list(root.iterdir()) == [owner.quarantine_root]


@pytest.mark.parametrize("enabled", [True, False])
def test_default_extraction_writes_and_registers_in_instance_home(boundary, monkeypatch, enabled):
    import easyicu.api as api
    from contextlib import nullcontext
    import pandas as pd

    root, home = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if enabled else "0")
    # A stale desktop preference must not send a hosted omitted out_dir to shared data.
    settings.update_settings({"export_dir": str(root / "desktop-results")})
    loaded = []
    def load_concepts(concepts, **kwargs):
        loaded.append(concepts)
        ids = kwargs["patient_ids"]["stay_id"]
        return pd.DataFrame({"stay_id": ids, "age": [65.0] * len(ids)})

    monkeypatch.setattr(api, "load_concepts", load_concepts)
    monkeypatch.setattr(api, "keep_cache", lambda **kwargs: nullcontext())
    monkeypatch.setattr(api, "get_all_patient_ids", lambda *a, **k: ([1, 2], "stay_id"))
    queued = []
    job = Job("synthetic-export", "extract")
    monkeypatch.setattr(job_routes, "submit_job", lambda kind, runner: queued.append(runner) or job)
    response = TestClient(app).post("/api/jobs/extract", json={
        "path": str(root), "database": "miiv", "modules": ["demographics"],
        "concepts": {"demographics": ["age"]}, "include_feature_definitions": False,
    })
    assert response.status_code == 200, response.text
    assert len(queued) == 1
    result = queued[0](job)
    out = Path(result["out_dir"])
    assert out.is_relative_to(home if enabled else root / "desktop-results")
    assert (out / "_manifest.json").is_file()
    assert result["registered_source"]["ok"]
    assert sources.load_registry()["active_path"] == str(out)
    assert loaded
    if enabled:
        assert list(root.iterdir()) == []
