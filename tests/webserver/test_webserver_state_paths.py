"""Contracts for the WebApp's state-root owner.

Twenty-four webserver modules used to call ``Path.home()`` and rebuild
``~/.easyicu`` or ``~/easyicu/projects`` inline. The effect was that the
server's state location could not be moved: running a second instance meant
overriding the process ``HOME``, which also relocates the user's real home.
``state_paths`` is the single owner, and ``EASYICU_HOME`` is the one knob.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

from easyicu.webserver import state_paths, study_contexts


WEBSERVER = Path(state_paths.__file__).resolve().parent


def test_pytest_collection_isolates_import_time_state_paths() -> None:
    pytest_home = Path(os.environ["EASYICU_HOME"])

    assert study_contexts._CONFIG_PATH == (
        pytest_home / ".easyicu" / "webserver_study_contexts.json"
    )
    assert study_contexts._CONFIG_PATH != (
        Path.home() / ".easyicu" / "webserver_study_contexts.json"
    )


def test_defaults_resolve_under_the_real_home(monkeypatch) -> None:
    monkeypatch.delenv("EASYICU_HOME", raising=False)

    assert state_paths.user_home() == Path.home()
    assert state_paths.state_root() == Path.home() / ".easyicu"
    assert state_paths.projects_root() == Path.home() / "easyicu" / "projects"


def test_easyicu_home_relocates_every_state_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))

    assert state_paths.user_home() == tmp_path
    assert state_paths.state_root() == tmp_path / ".easyicu"
    assert state_paths.projects_root() == tmp_path / "easyicu" / "projects"
    assert state_paths.exports_root() == tmp_path / "easyicu" / "exports"


def test_blank_override_is_treated_as_unset(monkeypatch) -> None:
    """An exported-but-empty variable must not relocate a user's studies."""

    monkeypatch.setenv("EASYICU_HOME", "   ")

    assert state_paths.user_home() == Path.home()


def test_override_is_resolved_per_call_not_at_import(monkeypatch, tmp_path) -> None:
    """So a test can point one process at tmp_path without reloading modules."""

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "first"))
    assert state_paths.state_root() == tmp_path / "first" / ".easyicu"

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "second"))
    assert state_paths.state_root() == tmp_path / "second" / ".easyicu"


# The folder picker starts from — and lists OS shortcuts under — the user's
# real home. Redirecting that with EASYICU_HOME would strand a user looking
# for their own exported data, so dataio keeps calling Path.home() directly.
# gateway._node_binary searches for an nvm-installed node, which likewise
# lives in the real home.
ALLOWED_REAL_HOME = {"dataio.py", "gateway.py", "state_paths.py"}


def test_no_module_rebuilds_a_state_root_from_path_home() -> None:
    offenders: list[str] = []
    for path in sorted(WEBSERVER.rglob("*.py")):
        if path.name in ALLOWED_REAL_HOME:
            continue
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"Path\.home\(\)", source):
            line = source[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(WEBSERVER)}:{line}")
    assert offenders == [], (
        "these modules bypass the state_paths owner; use state_root() / "
        f"projects_root() / user_home() instead: {offenders}"
    )
