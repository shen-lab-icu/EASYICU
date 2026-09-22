"""Routes and page settling shared by the native web UI audits.

``audit_web_popovers.py`` and ``audit_web_fit.py`` press and measure the same
product routes. The list lives here so a new route is audited by both, and
``tests/webserver/test_web_ui_audits.py`` checks it against the screens the
shell registers on ``window.SCREENS``.
"""

from __future__ import annotations

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
DEFAULT_VIEWPORTS = ("1542x1000", "1280x760", "1180x680")


def settle(page) -> None:
    """Give the route its data: the Guided startup shield covers the rail until then."""
    page.wait_for_timeout(1500)
    try:
        page.wait_for_selector("[data-guided-startup-shield]", state="detached", timeout=20000)
    except Exception:
        pass
    page.wait_for_timeout(2000)
