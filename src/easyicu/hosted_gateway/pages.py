"""The gateway's own pages: packaged templates and the policy they are served with."""

from __future__ import annotations

from pathlib import Path

from starlette.responses import HTMLResponse

TEMPLATES = Path(__file__).parent / "templates"
CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'self'; frame-ancestors 'none'"
_UNAVAILABLE = (TEMPLATES / "unavailable.html").read_text(encoding="utf-8")


def unavailable() -> HTMLResponse:
    """A down instance or unreadable gateway state: no socket, path or cause."""
    return HTMLResponse(_UNAVAILABLE, status_code=503,
                        headers={"cache-control": "no-store", "content-security-policy": CSP})
