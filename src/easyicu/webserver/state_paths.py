"""Owner for the roots under which the WebApp keeps its own state.

Before this module, twenty-four webserver modules each called ``Path.home()``
and rebuilt one of two roots inline (``~/.easyicu`` for config/state, and
``~/easyicu/projects`` for study project folders). Nothing was wrong with any
single call site, but together they made the server's state location
impossible to move: running a second instance — a review instance, an
end-to-end test instance, two studies side by side — meant overriding the
process ``HOME``, which also moves the user's real home directory out from
under everything else.

``EASYICU_HOME`` replaces the home directory that EasyICU derives *its own*
state from, and nothing else. Set it, and one process keeps its settings,
sessions, project folders, caches and receipts entirely separate from another.

What it deliberately does not move: the home directory the local folder
picker starts from. A user browsing for their exported data still wants their
real home, so ``dataio.list_dir`` keeps calling ``Path.home()`` directly.
That distinction is the reason this module exposes a named ``user_home()``
rather than letting call sites reach for ``Path.home()`` and hope.

Resolution happens per call, not at import, so a test can point one test at a
tmp_path with ``monkeypatch.setenv`` without reloading modules. The existing
module-level ``_CONFIG_DIR`` / ``_PROJECTS_ROOT`` constants still resolve at
import time and remain monkeypatchable exactly as before; this module only
changes where their default value comes from.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["user_home", "state_root", "projects_root", "exports_root"]

_ENV_VAR = "EASYICU_HOME"


def user_home() -> Path:
    """The home directory EasyICU derives its own state roots from.

    Returns ``$EASYICU_HOME`` when set to a non-empty value, otherwise the
    real home. A blank or whitespace-only value is treated as unset rather
    than as the current directory, so an exported-but-empty variable in a
    shell profile cannot silently relocate a user's studies.
    """

    override = str(os.environ.get(_ENV_VAR) or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home()


def state_root() -> Path:
    """``~/.easyicu`` — settings, sessions, caches, receipts, lock files."""

    return user_home() / ".easyicu"


def projects_root() -> Path:
    """``~/easyicu/projects`` — user-visible study project folders."""

    return user_home() / "easyicu" / "projects"


def exports_root() -> Path:
    """``~/easyicu/exports`` — the conventional export destination."""

    return user_home() / "easyicu" / "exports"
