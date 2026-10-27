"""The macOS DMG smoke launches the backend as the shell does and names the real failure."""

from pathlib import Path
import re

import pytest

from tests.support.desktop_bundle import DESKTOP, load_script

STARTED = "INFO:     Application startup complete.\nDurable dispatcher started (frozen=True)\n"
CRASHED = ('Traceback (most recent call last):\n  File "backend_entry.py", line 149, in main\n'
           "AttributeError: 'FastAPI' object has no attribute 'add_event_handler'\n"
           "[PYI-9024:ERROR] Failed to execute script 'backend_entry' due to unhandled exception!\n")


@pytest.mark.parametrize(("log", "launch", "expected"), [
    (STARTED, None, None),
    (STARTED, "backend readiness timed out", "backend readiness timed out"),
    ("easyicu.webserver.durable_workflows.DurableWorkflowError: durable_launch_failed\n",
     "backend exited before readiness: 1", "durable_launch_failed"),
    (CRASHED, "backend exited before readiness: 1",
     "desktop_backend_startup_incomplete: AttributeError: 'FastAPI' object has no attribute 'add_event_handler'"),
    ("", "backend exited before readiness: -9", "desktop_backend_startup_incomplete: backend exited before readiness: -9"),
    ("", None, "desktop_backend_startup_incomplete"),
    ("INFO:     Application startup complete.\n", None, "durable_executor_unavailable"),
])
def test_the_most_specific_startup_failure_wins(log, launch, expected):
    assert load_script("smoke_macos").startup_failure(log, launch) == expected


def test_macos_and_windows_smokes_read_the_backend_log_alike():
    module = load_script("smoke_macos")
    windows = (DESKTOP / "scripts/smoke_windows.ps1").read_text()
    assert re.search(r"\[regex\]::Match\(\$log, '([^']+)'\)", windows).group(1) == module.TYPED_FAILURE.pattern
    assert re.search(r"\[regex\]::Matches\(\$log, '([^']+)'\)", windows).group(1) == module.RAISED.pattern


def test_the_backend_is_launched_with_the_shells_arguments():
    shell = (DESKTOP / "src-tauri/src/lib.rs").read_text()
    start = shell.split("fn start_backend", 1)[1].split(".env(DESKTOP_TOKEN_ENV", 1)[0]
    shell_flags = re.findall(r'"(--[a-z-]+)"\.to_string\(\)', start)
    app = Path("/Applications/EasyICU.app")
    command = load_script("smoke_macos").backend_command(
        app, port=8123, state=Path("/s"), runtime=Path("/r"), parent_pid=7)
    assert command[0] == str(app / "Contents/Resources/resources/backend/easyicu-backend")
    assert [part for part in command if part.startswith("--")] == shell_flags
    assert command[command.index("--node-bin") + 1] == str(app / "Contents/Resources/resources/node")
    assert "X-EasyICU-Desktop-Token: {token}" in shell
    smoke = (DESKTOP / "scripts/smoke_macos.py").read_text()
    assert '"X-EasyICU-Desktop-Token": token' in smoke
    assert 'GET /api/catalog HTTP/1.1' in shell and '"/api/catalog"' in smoke
