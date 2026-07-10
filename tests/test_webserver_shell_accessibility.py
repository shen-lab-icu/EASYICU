"""Focused accessibility contracts for the native WebApp shell."""

from pathlib import Path


STATIC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
)


def _asset(*parts: str) -> str:
    return STATIC.joinpath(*parts).read_text(encoding="utf-8")


def test_sidebar_navigation_uses_native_controls_and_named_landmarks() -> None:
    app_js = _asset("js", "app.js")

    assert '<aside class="sidebar" aria-label=' in app_js
    assert '<nav class="shell-nav" aria-label=' in app_js
    assert '<header class="topbar" aria-label=' in app_js
    assert '<header class="mtopbar" aria-label=' in app_js
    assert '<main class="main" aria-label=' in app_js
    assert '<nav class="mbottomnav" aria-label=' in app_js

    assert '<button type="button" class="brand" data-nav="entry">' in app_js
    assert '<button type="button" class="nav-item ${route ===' in app_js
    assert '<button type="button" class="icobtn" title=' in app_js
    assert '<button type="button" class="mark" data-nav="entry" aria-label=' in app_js
    assert '<div class="brand" data-nav=' not in app_js
    assert '<div class="nav-item ${route ===' not in app_js
    assert '<div class="icobtn' not in app_js
    assert '<div class="mark" data-nav=' not in app_js
    assert '<button type="button" class="crumb-link" data-nav="entry">' in app_js
    assert '<a data-nav="entry">' not in app_js


def test_shell_exposes_current_route_and_disclosure_state() -> None:
    app_js = _asset("js", "app.js")

    assert "root.querySelectorAll('.sidebar [data-nav], .mbottomnav [data-nav]')" in app_js
    assert "control.setAttribute('aria-current', 'page')" in app_js
    assert "control.removeAttribute('aria-current')" in app_js
    assert '<span class="cur" aria-current="page">' in app_js
    assert 'data-ws-toggle aria-expanded="${wsOpen}"' in app_js
    assert 'aria-controls="data-workspace-links"' in app_js
    assert 'id="data-workspace-links"' in app_js


def test_data_and_language_segments_publish_pressed_state() -> None:
    app_js = _asset("js", "app.js")

    assert "root.querySelectorAll('[data-datamode], [data-hd]')" in app_js
    assert "control.dataset.datamode || control.dataset.hd" in app_js
    assert "root.querySelectorAll('[data-lang]')" in app_js
    assert "control.setAttribute('aria-pressed'" in app_js
    assert 'data-datamode="demo" aria-pressed=' in app_js
    assert 'data-datamode="real" aria-pressed=' in app_js
    assert 'data-lang="en" aria-pressed=' in app_js
    assert 'data-lang="zh" aria-pressed=' in app_js


def test_button_conversion_keeps_shell_visual_resets_in_owner_css() -> None:
    app_css = _asset("css", "app.css")

    assert "border: 0; background: transparent; color: inherit; font: inherit; text-align: left;" in app_css
    assert "width: 100%; border: 0; background: transparent; font: inherit; text-align: left;" in app_css
    assert "padding: 0; background: transparent; font: inherit;" in app_css
