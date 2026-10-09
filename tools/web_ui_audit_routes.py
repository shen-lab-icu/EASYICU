"""Routes and page settling shared by the native web UI audits.

``audit_web_popovers.py`` and ``audit_web_fit.py`` press and measure the same
product routes. The list lives here so a new route is audited by both, and
``tests/webserver/test_web_ui_audits.py`` checks it against the screens the
shell registers on ``window.SCREENS``.
"""

from __future__ import annotations

import json
import urllib.request

# Routes that render without a Guided conversation. The Guided route itself is
# audited through ``--session`` (its conversation and entry composers).
ROUTES_WITHOUT_SESSION = {
    "skills": "/#skills",
    "settings": "/#settings",
    "ideas": "/#ideas",
    "extraction": "/#extraction",
    "patient": "/#patient",
    "cohort": "/#cohort",
    "crossdb": "/#crossdb",
    "dictionary": "/#dictionary",
    "tutorial": "/#tutorial",
}
SESSION_ROUTE = "guided"
# Registered screens that are not product routes of their own.
EXCLUDED_ROUTES = {
    "entry": "redirects to #guided",
    "states": "design reference catalogue of global states, not a product route",
}
# Desktop and laptop sizes, then the tablet and phone widths the shell supports.
DEFAULT_VIEWPORTS = ("1542x1000", "1280x760", "1180x680", "768x1024", "390x844")


def new_audit_page(browser, base: str, width: int, height: int):
    """A page that starts in the server's saved language and data mode.

    A fresh browser profile renders English demo mode first and rebuilds the
    whole screen once ``/api/settings`` says otherwise; that late rebuild
    replaces the very menus an audit is pressing. Seeding the browser choice
    with the server's own settings keeps the first render final, and nothing
    is written back because the two already agree.
    """
    try:
        with urllib.request.urlopen(base + "/api/settings", timeout=10) as response:
            settings = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        settings = {}
    seed = {
        key: settings[field]
        for key, field in (("easyicu_lang", "language"), ("easyicu_home_data", "data_mode"))
        if isinstance(settings.get(field), str) and settings[field]
    }
    page = browser.new_page(viewport={"width": width, "height": height})
    if seed:
        page.add_init_script(
            "(() => { try { for (const [k, v] of Object.entries(%s)) localStorage.setItem(k, v); } catch (e) {} })();"
            % json.dumps(seed)
        )
    return page


def settle(page) -> None:
    """Give the route its data: the Guided startup shield covers the rail until then."""
    page.wait_for_timeout(1500)
    try:
        page.wait_for_selector("[data-guided-startup-shield]", state="detached", timeout=20000)
    except Exception:
        pass
    page.wait_for_timeout(2000)
