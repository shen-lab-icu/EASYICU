import importlib.util
import os
import threading
from types import SimpleNamespace
from pathlib import Path
import warnings

import pytest


def _backend_entry_module():
    path = Path(__file__).parents[2] / "desktop" / "backend_entry.py"
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
        "EASYICU_DEPLOYMENT",
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
    assert os.environ["EASYICU_DEPLOYMENT"] == "desktop"
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


def test_parent_loss_requests_graceful_shutdown_not_hard_exit(monkeypatch):
    import psutil
    module = _backend_entry_module()
    shutdown = threading.Event()
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(is_running=lambda: False))
    monkeypatch.setattr(module.os, "_exit", lambda code: pytest.fail("cleanup bypassed"))
    stop = module._watch_parent_process(987654, interval=0.01, request_shutdown=shutdown.set)
    assert shutdown.wait(1)
    stop.set()


@pytest.mark.parametrize("path,expected", [
    ("C:\\Users\\临床 user\\" + "nested\\" * 40 + "state", "\\\\?\\C:\\Users\\临床 user\\" + "nested\\" * 40 + "state"),
    (r"\\server\share\临床 user\state", r"\\?\UNC\server\share\临床 user\state"),
    (r"\\?\C:\Users\state", r"\\?\C:\Users\state"),
])
def test_windows_state_paths_preserve_unicode_spaces_unc_and_long_paths(path, expected):
    assert _backend_entry_module()._windows_extended_path(path) == expected


@pytest.mark.parametrize("path", ["relative", r"C:relative", r"\rooted-without-drive"])
def test_windows_state_path_requires_an_absolute_drive_or_share(path):
    with pytest.raises(ValueError, match="absolute drive or UNC"):
        _backend_entry_module()._windows_extended_path(path)


def test_windows_directory_setup_applies_the_extended_path_before_creation(tmp_path, monkeypatch):
    module = _backend_entry_module()
    calls = []

    def windows_path(raw):
        calls.append(raw)
        return raw  # Use the host filesystem for this orchestration check.

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module, "_windows_extended_path", windows_path)
    directory = tmp_path / "临床 state space"
    assert module._absolute_directory(str(directory), name="state-dir") == directory
    assert calls == [str(directory)]
    assert directory.is_dir()


def test_spawn_freeze_support_precedes_application_argument_parsing(monkeypatch):
    import multiprocessing
    import runpy

    class SpawnChildDispatched(Exception):
        pass

    def dispatch_child():
        raise SpawnChildDispatched

    monkeypatch.setattr(multiprocessing, "freeze_support", dispatch_child)
    monkeypatch.setattr("sys.argv", ["easyicu-backend", "--multiprocessing-fork", "pipe_handle=42"])
    path = Path(__file__).parents[2] / "desktop/backend_entry.py"
    with pytest.raises(SpawnChildDispatched):
        runpy.run_path(str(path), run_name="__main__")


@pytest.mark.parametrize("frozen", [False, True])
def test_desktop_startup_proves_a_running_dispatcher_before_logging(monkeypatch, caplog, frozen):
    from easyicu.webserver import durable_research

    module = _backend_entry_module()
    monkeypatch.setattr(module.sys, "frozen", frozen, raising=False)
    calls = []

    def status(identity):
        assert "Durable dispatcher started" not in caplog.text
        calls.append(identity)
        return None

    monkeypatch.setattr(durable_research, "current_service", lambda: SimpleNamespace(
        dispatcher=SimpleNamespace(status=status),
    ))
    with caplog.at_level("INFO", logger="uvicorn.error"):
        module._verify_durable_startup()
    assert calls == ["desktop-startup-probe"]
    assert f"Durable dispatcher started (frozen={frozen})" in caplog.text


@pytest.mark.parametrize("reason", ["missing_service", "durable_dependency_unavailable", "durable_launch_failed"])
def test_desktop_startup_retains_typed_durable_failure(monkeypatch, caplog, reason):
    from easyicu.webserver import durable_research
    from easyicu.webserver.durable_workflows import DurableWorkflowError

    def fail(identity):
        raise DurableWorkflowError(reason)

    service = None if reason == "missing_service" else SimpleNamespace(dispatcher=SimpleNamespace(status=fail))
    monkeypatch.setattr(durable_research, "current_service", lambda: service)
    expected = "durable_executor_unavailable" if service is None else reason
    with pytest.raises(DurableWorkflowError, match=expected):
        _backend_entry_module()._verify_durable_startup()
    assert "Durable dispatcher started" not in caplog.text


def test_desktop_registers_durable_probe_after_the_web_startup_handlers(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    module = _backend_entry_module()
    order = []
    monkeypatch.setattr(module, "_verify_durable_startup", lambda: order.append("probe"))
    app = FastAPI()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # The Web app registers its own startup handlers this way.
        @app.on_event("startup")
        def web_startup() -> None:
            order.append("web")

    # A registration the installed FastAPI lacks fails here, as the frozen app did.
    module._register_startup_probe(app)
    with TestClient(app):
        assert order == ["web", "probe"]

    source = (Path(__file__).parents[2] / "desktop/backend_entry.py").read_text()
    imported = source.index("from easyicu.webserver.app import app")
    probe = source.index("_register_startup_probe(app)\n")
    assert imported < probe < source.index("server.run()")
