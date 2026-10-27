"""The research conversation is the home; retired pages land there."""

import json
import re
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node


STATIC = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
)

RETIRED_FILES = (
    "js/screens-entry.js",
    "js/screens-ideas.js",
    "js/screens-ideas-zotero.js",
    "js/screens-states.js",
    "js/screens-agent-science.js",
    "js/screens-agent-run-history.js",
    "js/gate-remedy.js",
    "js/study-progress.js",
    "css/ideas.css",
    "css/ideas-review.css",
    "css/ideas-connectors.css",
    "css/agent-science.css",
    "css/agent-science-detail.css",
    "css/gate-remedy.css",
)


def _asset(*parts: str) -> str:
    return STATIC.joinpath(*parts).read_text(encoding="utf-8")


def test_browser_title_uses_the_product_name_not_an_internal_polish_label() -> None:
    index = _asset("index.html")

    assert "<title>EasyICU — ICU Research Workspace</title>" in index
    assert "Polished Workspace" not in index


def test_index_loads_no_retired_page_and_every_asset_it_names_exists() -> None:
    index = _asset("index.html")
    named = re.findall(r'(?:src|href)="((?:js|css)/[^"?]+)\?v=', index)

    assert named
    for path in RETIRED_FILES:
        assert path not in named, path
        assert not (STATIC / path).exists(), path
    assert [path for path in named if not (STATIC / path).is_file()] == []


def test_no_live_screen_registers_or_links_a_retired_route() -> None:
    sources = {path.name: path.read_text(encoding="utf-8") for path in (STATIC / "js").glob("*.js")}
    for name, source in sources.items():
        for route in ("entry", "ideas", "states"):
            assert f"S.{route} = {{" not in source, (name, route)
        assert 'data-nav="ideas"' not in source, name
        assert "location.hash = '#ideas'" not in source, name
        assert 'data-settings-open="ideas"' not in source, name


_ROUTER = r"""
const source = require('fs').readFileSync(process.argv[1], 'utf8');
const start = source.indexOf('  const FALLBACK_ROUTE');
const end = source.indexOf('  function rawRouteFromHash()');
global.window = {};
const normRoute = new Function(`${source.slice(start, end)}; return normRoute;`)();
const out = {};
for (const route of ['entry', 'ideas', 'states', 'agent', 'help', 'guided', 'patient']) {
  window.__euAlias = false;
  out[route] = [normRoute(route), window.__euAlias];
}
process.stdout.write(JSON.stringify(out));
"""


def test_retired_hashes_resolve_to_the_conversation() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(node, _ROUTER, str(STATIC / "js" / "app.js"), check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    routes = json.loads(result.stdout)

    for retired in ("entry", "ideas", "states"):
        assert routes[retired] == ["guided", True], retired
    assert routes["agent"] == ["guided", True]
    assert routes["help"][0] == "tutorial"
    assert routes["guided"] == ["guided", False]
    assert routes["patient"] == ["patient", False]


def test_get_started_sits_in_the_module_shell_with_help_in_its_rail() -> None:
    app_js = _asset("js", "app.js")
    shell = _asset("js", "desktop-module-shell.js")

    assert "const sharedHelpRoute = moduleShell && route === 'tutorial';" in app_js
    assert "${navItem(moduleActive, 'tutorial', 'help', 'Help', '帮助')}" in shell
    assert "hostedInstance() ? navItem(moduleActive, 'tutorial'" not in shell


def test_single_key_shortcuts_reach_live_sections_only() -> None:
    app_js = _asset("js", "app.js")

    assert "const SHORTCUT_SECTIONS = ['guided', 'extraction', 'patient', 'crossdb'];" in app_js
    assert "if (e.key >= '1' && e.key <= '4')" in app_js
