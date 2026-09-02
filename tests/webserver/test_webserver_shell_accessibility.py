"""Focused accessibility contracts for the native WebApp shell."""

from pathlib import Path


STATIC = (
    Path(__file__).resolve().parents[2]
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

    # aria-label is required here, not incidental: the brand button wraps a
    # name span and a tagline span, so its computed accessible name was the
    # two run together ("EasyICUICU Research Workspace").
    assert '<button type="button" class="brand" data-nav="entry" aria-label=' in app_js
    assert '<button type="button" class="nav-item ${route ===' in app_js
    assert '<button type="button" class="icobtn" title=' in app_js
    assert '<button type="button" class="mark" data-nav="entry" aria-label=' in app_js
    assert '<div class="brand" data-nav=' not in app_js
    assert '<div class="nav-item ${route ===' not in app_js
    assert '<div class="icobtn' not in app_js
    assert '<div class="mark" data-nav=' not in app_js
    assert '<button type="button" class="crumb-link" data-nav="entry">' in app_js
    assert '<a data-nav="entry">' not in app_js


def test_nav_entries_separate_their_title_from_their_sublabel() -> None:
    """Title + sublabel spans concatenate into one run-together name.

    "Patient Reviewtables · trends · patients" is what a screen reader
    announced. navLabel rejoins them with a separator instead of hiding the
    sublabel, and keeps the visible title first so voice control still matches
    on it (WCAG 2.5.3 Label in Name).
    """

    app_js = _asset("js", "app.js")

    assert "const navLabel = (label, sub) =>" in app_js
    # Every sidebar entry that renders a sublabel must supply the joined name.
    assert 'data-nav="${c.id}" aria-label="${navLabel(c.label, c.sub)}"' in app_js
    assert 'data-nav="guided" aria-label="${navLabel(' in app_js
    assert 'data-nav="agent" aria-label="${navLabel(' in app_js


def test_shell_exposes_current_route_and_disclosure_state() -> None:
    app_js = _asset("js", "app.js")

    assert "root.querySelectorAll('.sidebar [data-nav], .mbottomnav [data-nav]')" in app_js
    assert "control.setAttribute('aria-current', 'page')" in app_js
    assert "control.removeAttribute('aria-current')" in app_js
    assert '<span class="cur" aria-current="page">' in app_js
    assert 'data-ws-toggle aria-expanded="${wsOpen}"' in app_js
    assert 'aria-controls="data-workspace-links"' in app_js
    assert 'id="data-workspace-links"' in app_js


def test_spa_navigation_updates_title_announces_route_and_moves_focus() -> None:
    app_js = _asset("js", "app.js")
    shell_css = _asset("css", "shell.css")

    assert "const routeAnnouncer = document.createElement('div')" in app_js
    assert "routeAnnouncer.setAttribute('aria-live', 'polite')" in app_js
    assert "document.title = routeDocumentTitle(title)" in app_js
    assert "focusRouteContent()" in app_js
    assert "main.setAttribute('tabindex', '-1')" in app_js
    assert "if (preserveRouteFocus) focusRouteContent()" in app_js
    assert ".shell-sr-only" in shell_css


def test_guided_fullscreen_route_has_one_focusable_page_heading() -> None:
    guided = _asset("js", "screens-guided.js")

    assert '<h1 class="shell-sr-only" tabindex="-1">${t(\'Guided Copilot\', \'研究引导\')}</h1>' in guided


# Every route's own h1, keyed by the module that owns it. This used to cover
# only the guided route, which is exactly why Patient Review could regress to
# zero headings of any level while the suite stayed green: screens-viz.js owns
# three routes and still contained an <h1> for the other two, so any
# file-level check would have passed. app.js resolves both the document title
# and the post-navigation focus target from the route's h1, so a missing one
# is a functional regression, not only a WCAG 1.3.1 heading-order defect.
ROUTE_HEADINGS = {
    "screens-extraction.js": [
        "<h1>${t('Data Extraction', '数据抽取')}</h1>",
    ],
    "screens-viz-patient.js": [
        # Dense workspace: the visible title lives in the card eyebrow and the
        # loaded bar, so the route heading is screen-reader-only.
        '<h1 class="shell-sr-only" tabindex="-1">${t(\'Patient Review\', \'患者审阅\')}</h1>',
    ],
    "screens-viz-cohort.js": [
        "<h1 style=\"margin-top:0;\">${t('Cohort Statistics', '队列统计')}</h1>",
    ],
    "screens-viz.js": [
        "<h1 style=\"margin-top:0;\">${t('Cross-database comparison', '跨库对比')}</h1>",
    ],
    "screens-ideas.js": ["<h1"],
    "screens-agent.js": ["<h1"],
    "screens-settings.js": ["<h1"],
    "screens-dict.js": ["<h1"],
    "screens-states.js": ["<h1"],
    "screens-help.js": ["<h1"],
}


def test_every_route_owner_renders_its_own_page_heading() -> None:
    missing: list[str] = []
    for module, headings in ROUTE_HEADINGS.items():
        source = _asset("js", module)
        for heading in headings:
            if heading not in source:
                missing.append(f"{module}: {heading}")
    assert missing == [], "route owners missing their page heading:\n" + "\n".join(missing)


def test_patient_review_sections_sit_under_the_route_heading() -> None:
    """Loaded and idle Patient Review both had zero headings at any level."""

    viz = _asset("js", "screens-viz-patient.js")

    assert '<h2 class="patient-flow-title">' in viz
    assert '<div class="patient-flow-title">' not in viz
    assert '<h2 class="panel-title"' in viz
    # No global heading reset exists, so a promoted heading must clear the UA
    # margins or it silently changes the layout it was added to.
    assert "margin:0;" in _asset("css", "patient.css")


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
