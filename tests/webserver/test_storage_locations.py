"""Synthetic owner checks for remembered, visible data and project folders."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import demo_source_storage, demo_sources, guided_sessions, settings
from easyicu.webserver.demo_source_contracts import DemoSourceError, get_source


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    monkeypatch.delenv("EASYICU_DEPLOYMENT", raising=False)
    monkeypatch.delenv("EASYICU_DEMO_CACHE_DIR", raising=False)
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", tmp_path / "projects.json")
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "old-projects")
    monkeypatch.setattr(demo_source_storage, "DEFAULT_CACHE_ROOT", tmp_path / "old-cache")
    return tmp_path


def test_defaults_are_visible_and_do_not_create_folders(isolated):
    loaded = settings.load_settings()
    assert Path(loaded["data_dir"]) == isolated / "home" / "EasyICU" / "Data"
    assert Path(loaded["project_dir"]) == isolated / "home" / "EasyICU"
    assert loaded["data_dir_confirmed"] is False
    assert not Path(loaded["data_dir"]).exists()


def test_missing_folder_is_created_confirmed_and_reused(isolated):
    folder = isolated / "新数据目录"
    settings.update_settings({"data_dir": str(folder), "data_dir_confirmed": True})
    assert folder.is_dir()
    assert settings.load_settings()["data_dir_confirmed"] is True
    for source in demo_source_storage.SOURCES:
        assert demo_source_storage.source_paths(source).root.parent == folder


def test_changing_data_folder_requires_confirmation_again(isolated):
    first = isolated / "first"
    settings.update_settings({"data_dir": str(first), "data_dir_confirmed": True})
    after = settings.update_settings({"data_dir": str(isolated / "second")})
    assert after["data_dir_confirmed"] is False


@pytest.mark.parametrize("case,code", [
    ("relative", "storage_path_absolute_required"),
    ("file", "storage_path_not_directory"),
    ("unwritable", "storage_path_not_writable"),
    ("missing", "storage_path_not_creatable"),
])
def test_invalid_folder_rejects_entire_patch(isolated, monkeypatch, case, code):
    from easyicu.webserver import storage_locations

    folder = isolated / case
    if case == "relative":
        value = "relative/folder"
    else:
        value = str(folder)
    if case == "file":
        folder.write_text("synthetic")
    if case == "unwritable":
        monkeypatch.setattr(storage_locations.os, "access", lambda *_: False)
    if case == "missing":
        mkdir = Path.mkdir
        def refuse(path, *args, **kwargs):
            if path == folder:
                raise PermissionError("synthetic refusal")
            return mkdir(path, *args, **kwargs)
        monkeypatch.setattr(Path, "mkdir", refuse)
    with pytest.raises(settings.SettingsValidationError) as error:
        settings.update_settings({"data_dir": value, "density": "compact"})
    assert error.value.rejected[0]["code"] == code
    assert settings.load_settings()["density"] == "comfortable"
    assert not settings._CONFIG_PATH.exists()


def test_first_download_requires_confirmation_before_job_creation(isolated, monkeypatch):
    from easyicu.webserver.routes import demo_sources as routes
    from fastapi import HTTPException

    submitted = []
    monkeypatch.setattr(routes, "submit_job", lambda *args: submitted.append(args) or SimpleNamespace(id="synthetic", kind="demo-source-prepare", status="queued"))
    with pytest.raises(HTTPException) as error:
        routes.jobs_demo_source_prepare({"source_id": "mimic_iv_demo_v2_2"})
    assert error.value.status_code == 409
    assert error.value.detail["error"] == "data_directory_confirmation_required"
    assert submitted == []
    assert not (isolated / "old-cache").exists()


def test_insufficient_space_is_rejected_before_download(isolated, monkeypatch):
    from easyicu.webserver import storage_locations

    settings.update_settings({"data_dir": str(isolated / "data"), "data_dir_confirmed": True})
    monkeypatch.setattr(storage_locations.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(DemoSourceError) as error:
        demo_sources.make_prepare_runner("mimic_iv_demo_v2_2")
    assert error.value.code == "storage_insufficient_space"
    assert error.value.detail["required_bytes"] >= get_source("mimic_iv_demo_v2_2").size_bytes


def test_prepared_legacy_cache_survives_new_default(isolated):
    source = get_source("mimic_iv_demo_v2_2")
    legacy = isolated / "old-cache" / source.id
    raw = legacy / "raw"
    demo_source_storage.write_marker(raw / ".easyicu-demo-extracted.json", source)
    export = demo_source_storage.current_export_dir(legacy)
    export.mkdir(parents=True)
    (export / "_manifest.json").write_text("{}")
    paths = demo_source_storage.source_paths(source)
    demo_source_storage.write_export_marker(paths, source, archive_sha256="synthetic", export={})
    settings.update_settings({"data_dir": str(isolated / "new-data"), "data_dir_confirmed": True})
    # Incomplete preparation in the new folder must not hide a finished one.
    demo_source_storage.write_marker(isolated / "new-data" / source.id / "raw" / ".easyicu-demo-extracted.json", source)
    assert demo_source_storage.source_paths(source).root == legacy
    assert demo_source_storage.export_ready(demo_source_storage.source_paths(source), source)
    assert (export / "_manifest.json").read_text() == "{}"


def test_project_default_preserves_chinese_collision_and_existing_paths(isolated, monkeypatch):
    monkeypatch.setattr(guided_sessions, "_now", lambda: "2026-10-28T00:00:00Z")
    first_parent = isolated / "项目"
    settings.update_settings({"project_dir": str(first_parent)})
    first = guided_sessions.create_guided_draft({"title": "合成研究"})["draft"]
    second = guided_sessions.create_guided_draft({"title": "合成研究"})["draft"]
    assert Path(first["project_dir"]) == first_parent / "合成研究"
    assert Path(second["project_dir"]) == first_parent / "合成研究 2"
    settings.update_settings({"project_dir": str(isolated / "新项目")})
    listed = guided_sessions.list_guided_drafts()["drafts"]
    assert first["project_dir"] in [row["project_dir"] for row in listed]
    third = guided_sessions.create_guided_draft({"title": "合成研究"})["draft"]
    assert Path(third["project_dir"]).parent == isolated / "新项目"


def test_catalog_names_destination_and_confirmation_at_action(isolated):
    source = get_source("mimic_iv_demo_v2_2")
    status = demo_source_storage.status_payload(source)
    assert status["destination"] == str(isolated / "home" / "EasyICU" / "Data")
    assert status["data_dir_confirmed"] is False


def test_previous_data_location_keeps_valid_preparation_and_resume(isolated):
    source = get_source("mimic_iv_demo_v2_2")
    first = isolated / "first"
    settings.update_settings({"data_dir": str(first), "data_dir_confirmed": True})
    paths = demo_source_storage.source_paths(source)
    demo_source_storage.write_marker(paths.extracted_marker, source)
    settings.update_settings({"data_dir": str(isolated / "second"), "data_dir_confirmed": True})
    assert demo_source_storage.source_paths(source).root == paths.root


def test_hosted_cannot_set_machine_storage_preferences(isolated, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    with pytest.raises(settings.SettingsValidationError) as error:
        settings.update_settings({"data_dir": str(isolated / "outside")})
    assert error.value.rejected[0]["code"] == "storage_managed_by_administrator"


def test_reset_keeps_previous_data_readable(isolated):
    source = get_source("mimic_iv_demo_v2_2")
    settings.update_settings({"data_dir": str(isolated / "custom"), "data_dir_confirmed": True})
    paths = demo_source_storage.source_paths(source)
    demo_source_storage.write_marker(paths.extracted_marker, source)
    settings.reset_settings()
    assert demo_source_storage.source_paths(source).root == paths.root


def test_hosted_catalog_and_settings_hide_machine_paths(isolated, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    status = demo_source_storage.status_payload(get_source("mimic_iv_demo_v2_2"))
    assert "destination" not in status
    response = settings.settings_response(settings.load_settings())
    assert "data_dir" not in response and "project_dir" not in response


def test_data_location_history_is_not_browser_writable(isolated):
    with pytest.raises(settings.SettingsValidationError) as error:
        settings.update_settings({"data_dir_history": [str(isolated / "arbitrary")]})
    assert error.value.rejected[0]["code"] == "storage_history_read_only"
