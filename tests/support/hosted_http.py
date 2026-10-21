"""Replace HTTP feature handlers while retaining routing and dependencies."""

import pytest

from easyicu.webserver import guided_sessions, settings, sources
from easyicu.webserver.ideas import mining


def stub_route(monkeypatch, path):
    from easyicu.webserver.app import app

    calls = []

    def reached(**kwargs):
        calls.append(kwargs)
        return {"reached_owner": True}

    found = False
    for branch in app.routes:
        candidates = branch.effective_route_contexts() if hasattr(branch, "effective_route_contexts") else [branch]
        for route in candidates:
            if getattr(route, "path", None) == path and hasattr(route, "dependant"):
                monkeypatch.setattr(route.dependant, "call", reached)
                found = True
    assert found, path
    return calls


@pytest.fixture(name="boundary")
def hosted_boundary(tmp_path, monkeypatch):
    root, home = tmp_path / "prepared", tmp_path / "instance"
    root.mkdir()
    home.mkdir()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(root))
    monkeypatch.setenv("EASYICU_DISABLE_PROVIDER_ENV_FILE", "1")
    for owner in (settings, sources, guided_sessions):
        monkeypatch.setattr(owner, "_CONFIG_DIR", home)
        monkeypatch.setattr(owner, "_CONFIG_PATH", home / f"{owner.__name__.split('.')[-1]}.json")
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", home / "projects")
    monkeypatch.setattr(mining, "_RUN_ROOT", home / "ideas")
    return root, home


def assert_hosted_denied(error, code="hosted_data_read_only"):
    assert error.status_code == 403
    assert error.detail["error"] == code
