"""Contracts for top-level desktop modules sharing one navigation shell."""

from pathlib import Path


STATIC = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
)


def _read(*parts: str) -> str:
    return STATIC.joinpath(*parts).read_text(encoding="utf-8")


def test_data_and_skills_share_the_same_persistent_desktop_shell() -> None:
    index = _read("index.html")
    shell = _read("js", "desktop-module-shell.js")
    extraction = _read("js", "screens-extraction.js")
    skills = _read("js", "screens-skills.js")

    assert "js/desktop-module-shell.js?v=20260919-data-viz-shell1" in index
    assert index.index("js/desktop-module-shell.js?") < index.index(
        "js/screens-extraction.js?"
    )
    assert index.index("js/desktop-module-shell.js?") < index.index(
        "js/screens-skills.js?"
    )
    assert "window.EU_DESKTOP_MODULE_SHELL" in shell
    assert "function contextRail(active, options)" in shell
    assert "function rememberContext(value)" in shell
    assert "const moduleActive = isDataRoute(active) ? 'extraction' : active" in shell
    assert "navItem(moduleActive, 'projects'" in shell
    assert "navItem(moduleActive, 'skills'" in shell
    assert "navItem(moduleActive, 'extraction'" in shell
    assert "full: true" in extraction
    assert "active: 'extraction'" in extraction
    assert "active: 'skills'" in skills


def test_skills_use_the_research_workspace_palette() -> None:
    index = _read("index.html")
    css = _read("css", "skills-hub.css")

    assert "css/skills-hub.css?v=20260920-packages1" in index
    for token in (
        "--sk-canvas:#f7f6f2",
        "--sk-surface:#fbfaf7",
        "--sk-wash:#eeede8",
        "--sk-line-strong:#8f8b82",
        "--sk-accent:#246366",
    ):
        assert token in css
    assert ".euh-rail button.active svg,.eusk-rail button.active svg" in css
    assert "background:var(--sk-accent);color:#fff" in css


def test_desktop_modules_share_rail_geometry_and_route_motion() -> None:
    index = _read("index.html")
    skills = _read("css", "skills-hub.css")
    canvas = _read("css", "workspace-canvas.css")
    guided = "\n".join((
        _read("css", "guided-pi-workspace.css"),
        _read("css", "guided-pi-desktop.css"),
    ))
    app = _read("js", "app.js")

    assert "css/workspace-canvas.css?v=20260921-evidence1" in index
    assert "js/app.js?v=20260921-evidence1" in index
    assert "--workspace-global-rail-width: 59px" in canvas
    assert "--workspace-project-rail-width: clamp(330px, 15vw, 380px)" in canvas
    assert "var(--workspace-global-rail-width,59px)" in skills
    assert "var(--workspace-project-rail-width,clamp(330px,15vw,380px))" in skills
    assert "--gd-project-rail-width:var(--workspace-project-rail-width" in guided
    assert "function renderRoute(opts = {})" in app
    assert "document.startViewTransition" in app
    assert "'.gd-main.gpi-workspace .gd-conv, .eusk-main, .eudata-main" in app
    assert "(prefers-reduced-motion: reduce)" in app
    assert "view-transition-name: easyicu-global-navigation" in canvas
    assert "@media (prefers-reduced-motion: reduce)" in canvas


def test_visualization_routes_stay_inside_the_data_workspace_shell() -> None:
    shell = _read("js", "desktop-module-shell.js")
    app = _read("js", "app.js")

    for route in ("extraction", "patient", "cohort", "crossdb"):
        assert f"['{route}'," in shell
    assert "function dataNav(active, actions)" in shell
    assert "function renderData(options)" in shell
    assert "active: isDataRoute(opts.active) ? opts.active : 'extraction'" in shell
    assert "moduleShell.isDataRoute(route)" in app
    assert "content: scr.render()" in app
    assert "actions: actionHtmlOf(scr)" in app
    assert "app.querySelector('.eudata-main')" in app
    assert "sharedSettingsRoute" in app
    assert "active: 'settings'" in app


def test_data_shell_keeps_source_controls_and_extraction_actions_visible() -> None:
    extraction = _read("js", "screens-extraction.js")
    css = _read("css", "extraction.css")

    assert 'data-datamode="demo"' in extraction
    assert 'data-datamode="real"' in extraction
    assert 'data-ex-run="recommended"' in extraction
    assert "data-ex-advc" in extraction
    assert "data-ex-adve" in extraction
    assert "ex-action-dock" in extraction
    assert "data-ex-sync-guided" in extraction
    assert ".eudata-main-inner" in css
    assert ".ex-modules-primary" in css
