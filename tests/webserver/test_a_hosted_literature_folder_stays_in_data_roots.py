"""A hosted clinician scans literature only inside the administrator's data roots.

The hosted request guard matches path-bearing field names. The literature
scan also accepted ``folder``, which the guard did not list, so a request
could scan PDFs anywhere on the server. The guard now lists ``folder``, and
the scan owner checks the folder itself, so a field name the guard does not
know cannot reach the filesystem. A local build scans any folder as before.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from easyicu.webserver import settings
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining


@pytest.fixture
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_CONFIG_DIR", tmp_path / "state")
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "state" / "settings.json")
    root = tmp_path / "roots"
    inside = root / "papers"
    outside = tmp_path / "elsewhere"
    inside.mkdir(parents=True)
    outside.mkdir()
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(root))
    return inside, outside


@pytest.mark.parametrize("field", ["path", "folder"])
def test_a_hosted_request_cannot_name_a_folder_outside_the_roots(monkeypatch, folders, field) -> None:
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    calls = []
    monkeypatch.setattr(mining, "scan_literature_folder", lambda body: calls.append(body) or {})
    _, outside = folders

    response = TestClient(app).post("/api/ideas/literature-folder", json={field: str(outside)})

    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "path_outside_data_roots"
    assert calls == []


@pytest.mark.parametrize("field", ["path", "folder"])
def test_the_scan_owner_refuses_a_folder_outside_the_roots_on_its_own(monkeypatch, folders, field) -> None:
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    inside, outside = folders

    with pytest.raises(HTTPException) as refused:
        mining.scan_literature_folder({field: str(outside)})
    assert refused.value.status_code == 403
    assert refused.value.detail["error"] == "path_outside_data_roots"
    # Inside the roots the scan proceeds (an empty folder has no papers).
    assert mining.scan_literature_folder({field: str(inside)})


@pytest.mark.parametrize("flag", ["0", ""])
def test_a_local_build_scans_any_folder(monkeypatch, folders, flag) -> None:
    monkeypatch.setenv("EASYICU_HOSTED", flag)
    _, outside = folders

    assert mining.scan_literature_folder({"folder": str(outside)})
