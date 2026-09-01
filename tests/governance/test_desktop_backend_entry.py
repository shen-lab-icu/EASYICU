import importlib.util
import os
from pathlib import Path

import pytest


def _backend_entry_module():
    path = Path(__file__).parents[1] / "desktop" / "backend_entry.py"
    spec = importlib.util.spec_from_file_location("easyicu_desktop_backend_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_desktop_backend_environment_isolated_from_real_home(tmp_path, monkeypatch):
    module = _backend_entry_module()
    original_home = os.environ.get("HOME")
    # _configure_environment is an entry-point helper and intentionally writes
    # os.environ directly. Register every key it owns with monkeypatch first so
    # this test cannot leak desktop state into later tests in the same process.
    for name in (
        "EASYICU_HOME",
        "EASYICU_RUNTIME_DIR",
        "EASYICU_DESKTOP_SESSION_TOKEN",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "EASYICU_VERBOSE",
    ):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    node = tmp_path / "bin" / "node"
    node.parent.mkdir()
    node.write_text("node", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin")

    module._configure_environment(
        state_dir=str(tmp_path / "state"),
        runtime_dir=str(tmp_path / "runtime"),
        session_token="a" * 32,
        node_bin=str(node),
    )

    assert os.environ.get("HOME") == original_home
    assert os.environ["EASYICU_HOME"] == str((tmp_path / "state").resolve())
    assert os.environ["EASYICU_RUNTIME_DIR"] == str((tmp_path / "runtime").resolve())
    assert os.environ["PATH"].split(os.pathsep)[0] == str(node.parent)


@pytest.mark.parametrize("token", ["", "short"])
def test_desktop_backend_rejects_weak_session_token(tmp_path, token):
    module = _backend_entry_module()
    with pytest.raises(ValueError, match="at least 32"):
        module._configure_environment(
            state_dir=str(tmp_path / "state"),
            runtime_dir=str(tmp_path / "runtime"),
            session_token=token,
            node_bin=None,
        )


def test_desktop_backend_rejects_relative_state_path(tmp_path):
    module = _backend_entry_module()
    with pytest.raises(ValueError, match="absolute"):
        module._configure_environment(
            state_dir="relative-state",
            runtime_dir=str(tmp_path / "runtime"),
            session_token="b" * 32,
            node_bin=None,
        )


@pytest.mark.parametrize("parent_pid_spec", [0, 1, "current_process"])
def test_desktop_parent_watch_rejects_invalid_owner(parent_pid_spec):
    module = _backend_entry_module()
    parent_pid = os.getpid() if parent_pid_spec == "current_process" else parent_pid_spec
    with pytest.raises(ValueError, match="desktop shell"):
        module._watch_parent_process(parent_pid, interval=0.01)
