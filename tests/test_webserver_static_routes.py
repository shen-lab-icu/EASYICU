from __future__ import annotations

import re
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "static"


def _static_js(name: str) -> str:
    return (STATIC_DIR / "js" / name).read_text(encoding="utf-8")


def test_native_static_route_registry_contains_fallback_only_routes() -> None:
    screen_ids: set[str] = set()
    for path in (STATIC_DIR / "js").glob("screens-*.js"):
        screen_ids.update(re.findall(r"\bS\.([a-zA-Z0-9_]+)\s*=", path.read_text(encoding="utf-8")))

    assert {
        "entry",
        "extraction",
        "patient",
        "cohort",
        "crossdb",
        "agent",
        "settings",
        "dictionary",
        "states",
        "tutorial",
        "guided",
    } <= screen_ids


def test_native_hash_router_has_help_alias_and_unknown_hash_fails_safe() -> None:
    app_js = _static_js("app.js")

    assert "const FALLBACK_ROUTE = 'entry';" in app_js
    assert "if (r === 'help') return 'tutorial';" in app_js
    assert "history.replaceState(null, '', next)" in app_js
    assert "resolveRoute(rawRouteFromHash(), { rewrite: true })" in app_js
    assert "resolved.fallback" in app_js


def test_native_extraction_advanced_filters_are_backend_wired() -> None:
    api_js = _static_js("api.js")
    extraction_js = _static_js("screens-extraction.js")

    assert "/api/extraction/filter-options" in api_js
    assert "/api/extraction/filter-preview" in api_js
    assert "loadExtractionFilterOptions" in extraction_js
    assert "previewExtractionFilters" in extraction_js
    assert "Real-source filter audit" in extraction_js
    assert "Unsupported filters stay blocked" in extraction_js


def test_native_webapp_foreground_interrupt_returns_shell_status(monkeypatch) -> None:
    from easyicu.webserver import __main__ as webmain

    def fake_run(cmd, env):  # noqa: ANN001
        raise KeyboardInterrupt

    monkeypatch.setattr(webmain.subprocess, "run", fake_run)

    assert webmain.run_app(port=9876) == 130
