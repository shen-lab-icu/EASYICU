"""Ownership and size guards for the native WebApp frontend.

Three things are locked here.

**One escaping owner.** HTML escaping used to be re-implemented in nineteen
separate IIFEs, and the copies had drifted into four incompatible contracts.
The weakest of them (`[&<>]`, no quote escaping) was interpolated into HTML
attribute positions in screens-agent.js, where a single `"` in a run label or
artifact path truncates the attribute. `js/html-escape.js` is now the sole
implementation; this file fails the build if a second one reappears.

**One categorical chart palette.** Six palettes lived in five route files
across three colour systems (hex, oklch, and a fourth ad-hoc list in the cohort
route), and two of them were byte-identical. Fixing an adjacency in one left
the other five shipping it — including both SVG fallback renderers, which run
precisely when the ECharts shell failed to load. `js/chart-palette.js` is now
the sole definition.

The guard below *scans* for palettes rather than checking a list of known
copies. The previous version enumerated the five files it knew about and
grepped them for two specific colour values; it could not see the sixth copy
in screens-viz.js, which shipped the same converging pair under a different
hex. A guard written from the copies you already found only defends against
the copies you already found.

**Owner-file size ratchets.** CLAUDE.md sets soft budgets (~1500 lines for a
route/screen JS module, ~600 for a route CSS file) but nothing enforced them,
and six JS files and six CSS files had grown past. The project already has the
enforcement pattern — `test_api_module_ownership.py::test_public_api_facade_stays_thin`
asserts `api.py` stays under 600 lines — it just had not been applied here.

The ceilings below are *ratchets pinned at the values measured on 2026-08-16*,
not endorsements: files already over budget may not grow further, and files
under budget may not cross it. Lowering a ceiling after a split is the point.
Raising one is not a fix — split the file along its internal seams instead
(see the JS-split guidance in CLAUDE.md: promote the shared closure state to
an explicit namespace, do not `git mv` function bodies).
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"

# Files already over the CLAUDE.md budget. Ratchet: may shrink, never grow.
# A split should lower these numbers and add the new sibling owner files.
OVER_BUDGET_JS = {
    # 6000 before the Idea Mining sub-flow moved to screens-guided-idea.js,
    # 4961 before its session-slot serialisation followed it.
    "screens-guided.js": 4906,
    "screens-viz.js": 4947,
    "screens-agent.js": 2485,
    "screens-extraction.js": 2060,
    "screens-guided-pi.js": 1822,
    "screens-ideas.js": 1504,
}
OVER_BUDGET_CSS = {
    "guided.css": 2191,
    "agent.css": 1010,
    "patient.css": 702,
    "ideas.css": 695,
    "extraction.css": 639,
    "app.css": 634,
}

JS_BUDGET = 1500
CSS_BUDGET = 600

# Slack so an ordinary bug fix inside an already-oversized file does not fail
# the build; it only catches a file resuming real growth.
RATCHET_SLACK = 40


def _lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_html_escaping_has_exactly_one_implementation() -> None:
    owner = STATIC / "js" / "html-escape.js"
    assert owner.is_file(), "js/html-escape.js is the escaping owner and must exist"

    source = owner.read_text(encoding="utf-8")
    # Quotes must be covered: the defect that motivated this owner was an
    # attribute-position escape that only handled & < >.
    assert "/[&<>\"']/g" in source
    assert "window.EU_HTML" in source

    offenders = [
        path.name
        for path in sorted((STATIC / "js").glob("*.js"))
        if path.name != "html-escape.js"
        and re.search(r"^[ \t]*function esc\(", path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == [], (
        "these modules define a local esc() instead of destructuring the shared "
        f"owner (`const {{ esc }} = window.EU_HTML;`): {offenders}"
    )


def test_escaping_owner_loads_before_every_consumer() -> None:
    """The consumers destructure at IIFE top, so load order is load-bearing."""

    index = (STATIC / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="js/([^"?]+)', index)
    assert scripts, "index.html should load its scripts with js/ paths"
    assert scripts[0] == "html-escape.js", (
        f"html-escape.js must be the first js/ script; found {scripts[0]}"
    )


# A categorical palette is an array literal holding three or more colours.
# Near-white ramps are excluded structurally rather than by filename: the
# appearance tokens in tweaks.js are paper tints, not series colours, and the
# distinction should hold if that code ever moves.
_COLOUR = re.compile(
    r"#[0-9a-fA-F]{3,8}\b|oklch\([^)]*\)|hsl\([^)]*\)|rgba?\([^)]*\)|var\(--accent[^)]*\)"
)
_NEAR_WHITE = 0xE0


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", source)


def _is_tint_ramp(colours: list[str]) -> bool:
    """A ramp of near-white paper tints, not a categorical series palette."""

    for colour in colours:
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
            return False
        channels = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
        if min(channels) < _NEAR_WHITE:
            return False
    return True


def _palette_literals() -> list[tuple[str, int, list[str]]]:
    found = []
    for path in sorted((STATIC / "js").glob("*.js")):
        code = _strip_comments(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\[[^\[\]]*\]", code):
            colours = _COLOUR.findall(match.group(0))
            if len(colours) >= 3 and not _is_tint_ramp(colours):
                line = code[: match.start()].count("\n") + 1
                found.append((path.name, line, colours))
    return found


def test_categorical_chart_palette_has_exactly_one_definition() -> None:
    owner = STATIC / "js" / "chart-palette.js"
    assert owner.is_file(), "js/chart-palette.js is the palette owner and must exist"

    offenders = [
        f"{name}:{line} defines its own palette {colours}"
        for name, line, colours in _palette_literals()
        if name != "chart-palette.js"
    ]
    assert offenders == [], "\n".join(
        ["read window.EU_PALETTE instead of re-declaring the colours:"] + offenders
    )


def test_palette_owner_loads_before_every_consumer() -> None:
    """Consumers call window.EU_PALETTE during render, so order matters."""

    index = (STATIC / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="js/([^"?]+)', index)
    assert "chart-palette.js" in scripts
    consumers = [
        name
        for name in scripts
        if name != "chart-palette.js"
        and "window.EU_PALETTE" in (STATIC / "js" / name).read_text(encoding="utf-8")
    ]
    assert consumers, "the palette owner should have consumers"
    owner_at = scripts.index("chart-palette.js")
    late = [name for name in consumers if scripts.index(name) < owner_at]
    assert late == [], f"these load before the palette owner: {late}"


def test_palette_separates_hues_that_converge_under_deuteranopia() -> None:
    """The specific defect: a blue and a violet ~35-40 degrees apart."""

    source = _strip_comments(
        (STATIC / "js" / "chart-palette.js").read_text(encoding="utf-8")
    )

    # Both violets that shipped next to #2563eb, in two different files.
    assert "#7c3aed" not in source
    assert "#8b5cf6" not in source
    assert "oklch(58% 0.10 300)" not in source

    series = re.search(r"const SERIES = \[(.*?)\];", source, re.S)
    assert series, "the owner should still expose a SERIES list"
    colours = _COLOUR.findall(series.group(1))
    # SERIES[0] is null — resolved from --accent at read time — so the first
    # literal is index 1. A two-group contrast is the common case, so it must
    # not be another cool hue sitting next to the teal accent.
    assert colours[0] == "#b45309"
    assert colours.index("#b45309") < colours.index("#2563eb")

    # One stroke per colour, so a full cycle never repeats a (colour, stroke)
    # pair — the property the palette comment claims.
    dashes = re.search(r"const DASHES = \[(.*?)\];", source, re.S)
    assert dashes, "the owner should still expose a DASHES list"
    # One inner list per non-accent colour; the accent slot pairs with solid.
    assert dashes.group(1).count("[") == len(colours)


def test_every_multi_series_chart_encodes_more_than_colour() -> None:
    """Including the SVG fallbacks, which render when ECharts is missing."""

    echarts_paths = {
        "screens-viz-cohort-charts.js": "lineStyle(index",      # KM curves
        "screens-viz-crossdb-charts.js": "lineStyle(index",     # density
        "screens-viz-patient-charts.js": "lineStyle(index",     # comparison
    }
    for name, needle in echarts_paths.items():
        source = (STATIC / "js" / name).read_text(encoding="utf-8")
        assert needle in source, f"{name} distinguishes series by colour alone"
        # The old rule left series 0 and 1 — the common pair — both solid.
        assert "index % 3 === 2 ? 'dashed'" not in source, f"{name} kept the old rule"

    # The fallback renderers draw their own strokes and were colour-only.
    for name in ("screens-viz-crossdb-charts.js", "screens-viz-patient-series.js"):
        source = (STATIC / "js" / name).read_text(encoding="utf-8")
        assert "stroke-dasharray" in source, (
            f"{name} draws multi-series SVG lines without a per-series stroke"
        )


def test_every_js_contract_test_has_a_recorded_invocation() -> None:
    """A harness with no recorded argument list cannot be run at all.

    These tests take the owner files they exercise as positional arguments and
    stub `window` before loading them. CI wires up exactly one; the rest were
    invocable only by whoever remembered the argument list, so a change to an
    owner could not be checked against its own executable contract. The
    mapping lives in tools/run_js_contracts.py — this keeps it in sync without
    paying for a node run in the Python suite.
    """

    from tools.run_js_contracts import CONTRACTS

    harnesses = {path.name for path in (ROOT / "tests" / "js").glob("*.test.js")}
    assert harnesses, "tests/js should hold executable contract tests"
    assert harnesses == set(CONTRACTS), (
        "tools/run_js_contracts.py is out of sync with tests/js/: "
        f"unlisted={sorted(harnesses - set(CONTRACTS))} "
        f"stale={sorted(set(CONTRACTS) - harnesses)}"
    )

    missing = [
        f"{name} -> {owner}"
        for name, owners in CONTRACTS.items()
        for owner in owners
        if not (STATIC / "js" / owner).is_file()
    ]
    assert missing == [], f"recorded owner files that no longer exist: {missing}"


def test_strict_mode_directives_stay_in_the_prologue() -> None:
    """`'use strict'` only counts as a directive before any statement.

    Injecting `const { esc } = window.EU_HTML;` above an existing
    `'use strict';` demotes it to a plain string expression and silently turns
    strict mode off for the whole module — no syntax error, no console warning.
    Anything inserted at the top of an IIFE has to go after the directive.
    """

    offenders: list[str] = []
    for path in sorted((STATIC / "js").glob("*.js")):
        source = path.read_text(encoding="utf-8")
        if "'use strict'" not in source:
            continue
        # Strip comments first: these modules open with multi-line banner
        # comments whose continuation lines are ordinary prose.
        stripped_source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
        stripped_source = re.sub(r"^\s*//.*$", "", stripped_source, flags=re.M)
        statements = [
            line.strip() for line in stripped_source.splitlines() if line.strip()
        ]
        # Drop the IIFE opener(s); the directive belongs immediately inside.
        while statements and re.fullmatch(r"\(?\s*function\s*\(\s*\)\s*\{", statements[0]):
            statements.pop(0)
        if not statements:
            continue
        if statements[0] != "'use strict';":
            offenders.append(
                f"{path.name}: {statements[0][:60]!r} runs before 'use strict'"
            )
    assert offenders == [], "\n".join(offenders)


def test_the_idea_sub_flow_does_not_reach_back_into_the_guided_closure() -> None:
    """The split is only real if the boundary is one-way.

    screens-guided.js was 6000 lines holding a conversation engine, four
    sub-flows and an 835-line DOM event binder in a single closure. Idea Mining
    came out first because it was the least entangled seam: measured against
    the rest of the file it read 15 of the parent functions and 7 of its ~40
    mutable closure variables, where the sibling project-folder block of the
    same size touched 39.

    What makes it a split rather than a move is that the sub-flow now receives
    those 15 through `init()` and owns its own state. If a bare parent name
    reappears here, the file is back to depending on a closure it no longer
    lives in — and it would only fail at runtime, on the one branch that calls
    it.
    """

    owner = STATIC / "js" / "screens-guided-idea.js"
    assert owner.is_file()
    source = owner.read_text(encoding="utf-8")

    assert "window.EU_GUIDED_IDEA" in source
    assert "let host = {" in source, "the shell's callbacks must arrive through init()"

    host_bound = [
        "activeExportSource", "bi", "compactHash", "compactPath", "fmtInt",
        "fmtNum", "fmtPct", "guidedMetricCard", "markThrough", "pushUser",
        "renderAside", "renderChips", "renderThread", "scheduleGuidedSlotSave",
        "setVal",
    ]
    body = source[source.index("let guidedIdea = null;") :]
    offenders = [
        name
        for name in host_bound
        if re.search(rf"(?<![.\w]){name}\s*\(", body)
    ]
    assert offenders == [], (
        f"these call the guided closure directly instead of host.<name>: {offenders}"
    )

    # A getter cannot be assigned to; that mistake parses and only fails when
    # the branch runs. `chips = []` became `host.clearChips()` for this reason.
    assert not re.search(r"host\.[A-Za-z0-9_$]+\(\)\s*=[^=]", source), (
        "a host accessor is being used as an assignment target"
    )

    # The state has one owner: the shell must not declare it again.
    shell = (STATIC / "js" / "screens-guided.js").read_text(encoding="utf-8")
    for name in ("guidedIdea", "guidedIdeaProvider", "guidedLiteratureBrowser"):
        assert not re.search(rf"^  let .*\b{name}\b", shell, re.M), (
            f"{name} is declared in the shell as well as the owner"
        )
    assert "const IDEA = window.EU_GUIDED_IDEA;" in shell


def test_the_guided_module_table_carries_no_hand_written_concept_counts() -> None:
    """A number promised before extraction has to come from the catalog.

    `GUIDED_EXTRACT_MODULES` used to carry a per-module concept count as a
    fallback for `guidedModuleConceptCount()`. Two of the nineteen had gone
    stale — renal listed 22 against the catalog's 35, neurological 12 against
    14 — so the fallback could only ever under-promise what the module would
    actually extract. The catalog itself is held to the backend concept by
    concept (test_concept_catalog_consistency.py), so it is the one source.
    """

    guided = (STATIC / "js" / "screens-guided.js").read_text(encoding="utf-8")
    table = guided[guided.index("const GUIDED_EXTRACT_MODULES = [") :]
    table = table[: table.index("];")]

    rows = re.findall(r"\['([^']+)', '[^']*', '[^']*', (\w+)\]", table)
    assert len(rows) == 19, f"expected 19 modules, parsed {len(rows)}"
    for module, flag in rows:
        assert flag in ("true", "false"), (
            f"{module}: the fourth column is the is-core flag, not a count"
        )

    # The helper must not accept a fallback again, in either owner.
    assert "function guidedModuleConceptCount(key)" in guided
    assert "moduleConceptCount(key, " not in guided
    extract_owner = (STATIC / "js" / "screens-guided-extract.js").read_text(encoding="utf-8")
    assert "moduleConceptCount(key, " not in extract_owner
    # Both renderers read the same column layout.
    assert "ctx.modules.filter(m => m[3])" in extract_owner


def test_owner_js_files_do_not_grow_past_their_ratchet() -> None:
    too_big: list[str] = []
    for path in sorted((STATIC / "js").glob("*.js")):
        count = _lines(path)
        ceiling = OVER_BUDGET_JS.get(path.name)
        if ceiling is None:
            if count > JS_BUDGET:
                too_big.append(
                    f"{path.name}: {count} lines crosses the {JS_BUDGET}-line budget; "
                    "split it into sibling owner files rather than adding it here"
                )
        elif count > ceiling + RATCHET_SLACK:
            too_big.append(
                f"{path.name}: {count} lines exceeds its {ceiling}-line ratchet; "
                "this file is already over budget and must not grow"
            )
    assert too_big == [], "\n".join(too_big)


def test_owner_css_files_do_not_grow_past_their_ratchet() -> None:
    too_big: list[str] = []
    for path in sorted((STATIC / "css").glob("*.css")):
        count = _lines(path)
        ceiling = OVER_BUDGET_CSS.get(path.name)
        if ceiling is None:
            if count > CSS_BUDGET:
                too_big.append(
                    f"{path.name}: {count} lines crosses the {CSS_BUDGET}-line budget; "
                    "give the new rules their own owner file"
                )
        elif count > ceiling + RATCHET_SLACK:
            too_big.append(
                f"{path.name}: {count} lines exceeds its {ceiling}-line ratchet; "
                "this file is already over budget and must not grow"
            )
    assert too_big == [], "\n".join(too_big)


def test_ratchets_track_files_that_still_exist() -> None:
    """A split renames files; a stale ratchet entry would silently stop guarding."""

    missing_js = [n for n in OVER_BUDGET_JS if not (STATIC / "js" / n).is_file()]
    missing_css = [n for n in OVER_BUDGET_CSS if not (STATIC / "css" / n).is_file()]
    assert missing_js == [], f"stale JS ratchet entries: {missing_js}"
    assert missing_css == [], f"stale CSS ratchet entries: {missing_css}"
