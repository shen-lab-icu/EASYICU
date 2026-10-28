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

    assert "js/desktop-module-shell.js?v=20261028-flow1" in index
    assert index.index("js/desktop-module-shell.js?") < index.index(
        "js/screens-extraction.js?"
    )
    assert index.index("js/desktop-module-shell.js?") < index.index(
        "js/screens-skills.js?"
    )
    assert "window.EU_DESKTOP_MODULE_SHELL" in shell
    assert "function contextRail(active, options)" in shell
    assert "function rememberContext(value)" in shell
    assert "const moduleActive = isDataRoute(active) ? 'extraction' : workflow ? 'workflows' : active" in shell
    assert 'data-nav="skills" data-sk-mode="workflows"' in shell
    assert "navItem(moduleActive, 'projects'" in shell
    assert "navItem(moduleActive, 'skills'" in shell
    assert "navItem(moduleActive, 'extraction'" in shell
    assert "full: true" in extraction
    assert "active: 'extraction'" in extraction
    assert "active: 'skills'" in skills


def test_module_shell_stacks_its_context_above_the_main_column_on_phones() -> None:
    """At phone width the three-child shell must not squeeze the main pane.

    Skills, settings and the four data routes share one shell: rail, context
    aside, main. The phone rule gave it two columns, so the main pane wrapped
    into a new row inside the 50px rail column. The rail now spans both rows,
    the context sits above the main column (bounded and scrollable), and the
    main fills the second column.
    """

    css = (STATIC / "css" / "skills-hub.css").read_text(encoding="utf-8")
    phone = css[css.index("@media(max-width:700px){"):]
    phone = phone[: phone.index("\n")]
    assert ".euh-shell,.eusk-shell{grid-template-columns:50px minmax(0,1fr);grid-template-rows:auto minmax(0,1fr)}" in phone
    assert ".euh-shell>.euh-rail,.eusk-shell>.eusk-rail,.euh-shell>.eusk-rail{grid-column:1;grid-row:1/span 2}" in phone
    assert ".euh-shell>.euh-context{grid-column:2;grid-row:1;max-height:38dvh;overflow:auto;" in phone
    assert ".euh-shell>main,.eusk-shell>main{grid-column:2;grid-row:2;min-width:0;min-height:0}" in phone
    # The create menu stays right-aligned to its button but fits the column.
    assert ".eusk-create-menu>div{width:min(290px,calc(100vw - 82px))}" in phone


def test_skills_use_the_research_workspace_palette() -> None:
    index = _read("index.html")
    css = _read("css", "skills-hub.css")

    assert "css/skills-hub.css?v=20261028-flow3" in index
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

    assert "css/workspace-canvas.css?v=20261022-winfont1" in index
    assert "js/app.js?v=20261027-dmap2" in index
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

    for route in ("extraction", "patient", "cohort", "crossdb", "dictionary"):
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


def test_get_started_sits_in_the_module_shell_under_help() -> None:
    shell = _read("js", "desktop-module-shell.js")
    app = _read("js", "app.js")
    css = _read("css", "skills-hub.css")

    # Every deployment renders Get Started in this shell with Help in its
    # rail; the retired sidebar shell no longer hosts a full-page tutorial.
    assert "${navItem(moduleActive, 'tutorial', 'help', 'Help', '帮助')}" in shell
    assert "hostedInstance() ? navItem(moduleActive, 'tutorial'" not in shell
    assert "const sharedHelpRoute = moduleShell && route === 'tutorial';" in app
    assert "active: 'tutorial'" in app
    assert "aside: scr.contextAside ? scr.contextAside() : ''" in app
    assert "active === 'tutorial' ? tr('Help', '帮助')" in shell
    assert "</nav>${options.aside || ''}`" in shell
    assert ".euh-context-contact{" in css


def test_page_actions_wrap_inside_the_context_column() -> None:
    # Cross-database results put three actions in the 260 px context column;
    # the third ran past its edge before they wrapped.
    css = _read("css", "extraction.css")
    assert ".euh-context-nav .eudata-module-actions{flex-wrap:wrap;margin-top:10px}" in css


def test_data_shell_keeps_source_controls_and_extraction_actions_visible() -> None:
    extraction = _read("js", "screens-extraction.js")
    css = _read("css", "extraction.css")

    assert "data-datamode" not in extraction
    assert 'data-ex-run="recommended"' in extraction
    assert "data-ex-advc" in extraction
    assert "data-ex-adve" in extraction
    assert "ex-action-dock" in extraction
    assert "data-ex-sync-guided" in extraction
    assert ".eudata-main-inner" in css
    assert ".ex-modules-primary" in css
