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


def test_pages_sit_in_the_module_shell_with_named_landmarks() -> None:
    """app.js draws no page chrome of its own: the old sidebar, top bar and
    mobile bars are retired, and every non-full page uses the module shell."""
    app_js = _asset("js", "app.js")
    shell = _asset("js", "desktop-module-shell.js")

    assert '<nav class="euh-rail" aria-label=' in shell
    assert '<main class="${opts.mainClass || \'euh-main\'}" aria-label=' in shell
    assert 'aria-current="page"' in shell
    assert "moduleShell.render({ active: route, content: scr.render() })" in app_js
    for retired in (
        'class="sidebar"', 'class="topbar"', 'class="mtopbar"', 'class="mbottomnav"',
        "function sidebar()", "function topbar()", "function mobileChrome()", "navLabel",
        "data-ws-toggle", "data-cpopen",
    ):
        assert retired not in app_js, retired


def test_spa_navigation_updates_title_announces_route_and_moves_focus() -> None:
    app_js = _asset("js", "app.js")
    shell_css = _asset("css", "shell.css")

    assert "const routeAnnouncer = document.createElement('div')" in app_js
    assert "routeAnnouncer.setAttribute('aria-live', 'polite')" in app_js
    assert "document.title = routeDocumentTitle(title)" in app_js
    assert "focusRouteContent()" in app_js
    assert "main.setAttribute('tabindex', '-1')" in app_js
    assert "target.dataset.euRouteFocus = 'true'" in app_js
    assert "if (preserveRouteFocus) focusRouteContent()" in app_js
    assert ".shell-sr-only" in shell_css
    assert '[data-eu-route-focus="true"]:focus' in _asset("css", "workspace-workbench.css")


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
    "screens-settings.js": ["<h1"],
    "screens-dict.js": ["<h1"],
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


def test_button_conversion_keeps_shell_visual_resets_in_owner_css() -> None:
    app_css = _asset("css", "app.css")

    # .nav-item is a <button> (Settings section jumps); its UA chrome is reset here.
    assert "width: 100%; border: 0; background: transparent; font: inherit; text-align: left;" in app_css
