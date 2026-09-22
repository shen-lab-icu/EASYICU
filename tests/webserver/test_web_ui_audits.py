"""The native web UI fit audits: route coverage always, live runs on request.

``tools/audit_web_popovers.py`` presses every menu and disclosure and
``tools/audit_web_fit.py`` measures each route at rest. They caught what
contract tests could not see: a model menu opening off-screen below the
composer, a menu clipped by a 520 px panel, menus that never closed on an
outside press. The coverage check keeps every product route in their shared
route table; the live check runs both against a server named by
``EASYICU_WEB_AUDIT_BASE`` (optionally ``EASYICU_WEB_AUDIT_SESSION`` =
``pi_project=…&pi_session=…`` for the Guided composers) and is skipped, and
counted as skipped, without one.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools import audit_web_fit, audit_web_popovers, web_ui_audit_routes

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_JS = REPO_ROOT / "src" / "easyicu" / "webserver" / "static" / "js"


def _registered_screens() -> set[str]:
    screens: set[str] = set()
    for path in STATIC_JS.glob("screens-*.js"):
        screens.update(re.findall(r"^\s*S\.([a-zA-Z]+) = \{", path.read_text(encoding="utf-8"), re.M))
    return screens


def test_audits_cover_every_registered_product_route() -> None:
    """A screen added to window.SCREENS is audited unless it is not a product route."""

    screens = _registered_screens()
    assert {"guided", "skills", "settings", "extraction"} <= screens
    covered = set(web_ui_audit_routes.ROUTES_WITHOUT_SESSION) | {web_ui_audit_routes.SESSION_ROUTE}
    excluded = set(web_ui_audit_routes.EXCLUDED_ROUTES)
    assert screens - covered - excluded == set(), "audit the new route or record why it is excluded"
    assert excluded <= screens, "an exclusion names a screen that no longer exists"
    assert covered & excluded == set()
    for name, url in web_ui_audit_routes.ROUTES_WITHOUT_SESSION.items():
        assert url == f"/#{name}"
    # Both tools read the one table rather than keeping their own copies.
    for tool in (audit_web_popovers, audit_web_fit):
        assert tool.ROUTES_WITHOUT_SESSION is web_ui_audit_routes.ROUTES_WITHOUT_SESSION
        assert tool.settle is web_ui_audit_routes.settle


@pytest.mark.requires_web_server
@pytest.mark.parametrize("tool", ["audit_web_popovers.py", "audit_web_fit.py"])
def test_live_ui_audit_finds_nothing(tool: str, tmp_path: Path) -> None:
    args = [
        sys.executable, str(REPO_ROOT / "tools" / tool),
        "--base", os.environ["EASYICU_WEB_AUDIT_BASE"],
        "--out", str(tmp_path / "report.json"),
    ]
    session = os.environ.get("EASYICU_WEB_AUDIT_SESSION", "")
    if session:
        args += ["--session", session]
    completed = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    assert completed.returncode == 0, completed.stdout[-6000:] + completed.stderr[-2000:]
