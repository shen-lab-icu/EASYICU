"""Theme colours draw in Chrome/Edge before 111 (Windows 7 stops at 109).

Such a browser draws neither oklch() nor color-mix(). A declaration without
var() is dropped while the stylesheet is parsed, so the sources write those
colours as hex/rgba. A declaration with var() keeps its raw text in the CSSOM
and resolves to nothing; js/legacy-colors.js computes it from the tokens and
writes rgb()/rgba() back. These tests pin both halves.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node

STATIC = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"
_DECLARATION = re.compile(r"(?:^|[{;])\s*(?P<prop>[-\w]+)\s*:\s*(?P<value>[^;{}]*)(?=[;}])")
_STYLE_ATTRIBUTE = re.compile(r'style="(?P<body>[^"]*)"')
_MODERN = re.compile(r"\b(oklch|color-mix)\(")


def _static_colour_declarations(text: str) -> list[str]:
    found = []
    for match in _DECLARATION.finditer(text):
        prop, value = match["prop"], match["value"]
        if prop.startswith("--") or "var(" in value or not _MODERN.search(value):
            continue
        found.append(f"{prop}: {value.strip()}")
    return found


def test_no_stylesheet_writes_a_static_colour_an_old_browser_drops() -> None:
    offenders = {}
    for path in sorted((STATIC / "css").glob("*.css")):
        text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
        found = _static_colour_declarations(text)
        if found:
            offenders[path.name] = found
    assert offenders == {}


def test_no_rendered_inline_style_writes_a_static_colour_an_old_browser_drops() -> None:
    offenders = {}
    for path in sorted((STATIC / "js").glob("*.js")):
        if path.name == "legacy-colors.js":
            continue
        found = [
            item
            for attribute in _STYLE_ATTRIBUTE.finditer(path.read_text(encoding="utf-8"))
            for item in _static_colour_declarations(";" + attribute["body"] + ";")
        ]
        if found:
            offenders[path.name] = found
    assert offenders == {}


def test_the_polyfill_runs_after_every_stylesheet_and_before_any_page_renders() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="js/([^"?]+)', index)
    # html-escape.js stays first (test_static_frontend_ownership); it renders nothing.
    assert scripts[:2] == ["html-escape.js", "legacy-colors.js"]
    assert index.rindex('rel="stylesheet"') < index.index('<script src="js/legacy-colors.js?v=')


_FAKE_DOM = r"""
global.window = global;
class Style {
  constructor(entries) { this.map = new Map(entries || []); this.calls = []; }
  get cssText() {
    return [...this.map].map(([k, v]) => `${k}: ${v.value}${v.important ? ' !important' : ''};`).join(' ');
  }
  getPropertyValue(name) { const v = this.map.get(name); return v ? v.value : ''; }
  setProperty(name, value, priority) {
    this.calls.push([name, value, priority || '']);
    this.map.set(name, { value, important: priority === 'important' });
  }
}
const decl = (k, v, important) => [k, { value: v, important: !!important }];
const rootStyle = new Style();
const rootEl = { style: rootStyle, getAttribute: () => rootStyle.cssText, nodeType: 1, isConnected: true };
global.document = { documentElement: rootEl, head: null, styleSheets: [] };
global.getComputedStyle = el => ({ getPropertyValue: name => el.style.getPropertyValue(name) });
let observerCallback = null;
global.MutationObserver = class { constructor(cb) { observerCallback = cb; } observe() {} };
global.addEventListener = () => {};
"""


def _node(program: str) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    source = (STATIC / "js" / "legacy-colors.js").read_text(encoding="utf-8")
    return json.loads(run_node(node, _FAKE_DOM + program.replace("__POLYFILL__", source), check=True).stdout)


def _oklch_channels(lightness: float, chroma: float, hue: float) -> tuple[float, float, float]:
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
    l_ = (lightness / 100 + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness / 100 - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness / 100 - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )

    def encode(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return 255 * (12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055)
    return tuple(encode(x) for x in linear)


def _oklch_srgb(lightness: float, chroma: float, hue: float) -> tuple[int, int, int]:
    return tuple(round(x) for x in _oklch_channels(lightness, chroma, hue))


def test_an_old_browser_gets_every_theme_value_computed_into_its_declaration() -> None:
    shown = _node(r"""
      const tokens = new Style([
        decl('--accent', 'oklch(52% 0.07 205)'),
        decl('--accent-soft', 'oklch(96% 0.02 205)'),
        decl('--surface', '#ffffff'),
        decl('--tint', 'color-mix(in srgb, var(--accent) 10%, white)'),
      ]);
      const scoped = new Style([decl('--provider-ink', '#20211f'), decl('color', 'var(--provider-ink)')]);
      const overlay = new Style([decl('background', 'color-mix(in srgb,var(--provider-ink) 34%,transparent)')]);
      const card = new Style([
        decl('border', '1px solid color-mix(in srgb,var(--accent) 12%,transparent)'),
        decl('background', 'linear-gradient(180deg, color-mix(in srgb, var(--accent-soft) 40%, var(--surface)) 0%, var(--surface) 100%)'),
        decl('box-shadow', '0 0 0 3px color-mix(in srgb, var(--missing, #2563eb) 12%, transparent)', true),
        decl('color', 'var(--accent)'),
      ]);
      document.styleSheets = [{ cssRules: [
        { selectorText: ':root', style: tokens },
        { cssRules: [{ selectorText: '.gpi-provider-shell', style: scoped }] },
        { selectorText: '.gpi-connect-backdrop', style: overlay },
        { selectorText: '.card', style: card },
      ] }];
      global.CSS = { supports: () => false };
      __POLYFILL__
      const before = card.calls.length;
      rootStyle.setProperty('--accent', 'oklch(52% 0.07 245)');
      observerCallback([{ type: 'attributes', target: rootEl }]);
      process.stdout.write(JSON.stringify({
        active: window.EU_LEGACY_COLORS.active,
        tokens: tokens.calls, overlay: overlay.calls, card: card.calls.slice(0, before),
        afterTweak: card.calls.slice(before), rootAfterTweak: rootStyle.calls.slice(1),
      }));
    """)

    teal = _oklch_srgb(52, 0.07, 205)
    assert shown["active"] is True
    assert ["--accent", "rgb({}, {}, {})".format(*teal), ""] in shown["tokens"]
    tint = tuple(round(0.1 * t + 0.9 * 255) for t in _oklch_channels(52, 0.07, 205))
    assert ["--tint", "rgb({}, {}, {})".format(*tint), ""] in shown["tokens"]
    assert shown["overlay"] == [["background", "rgba(32, 33, 31, 0.34)", ""]]
    mixed = tuple(round(0.4 * s + 0.6 * 255) for s in _oklch_channels(96, 0.02, 205))
    assert shown["card"] == [
        ["border", "1px solid rgba({}, {}, {}, 0.12)".format(*teal), ""],
        ["background", "linear-gradient(180deg, rgb({}, {}, {}) 0%, #ffffff 100%)".format(*mixed), ""],
        ["box-shadow", "0 0 0 3px rgba(37, 99, 235, 0.12)", "important"],
    ]
    # The appearance editor rotates the accent hue on the root: the root
    # token and everything mixed from it follow.
    blue = _oklch_srgb(52, 0.07, 245)
    assert shown["rootAfterTweak"] == [["--accent", "rgb({}, {}, {})".format(*blue), ""]]
    assert shown["afterTweak"] == [["border", "1px solid rgba({}, {}, {}, 0.12)".format(*blue), ""]]


def test_a_current_browser_is_left_alone() -> None:
    shown = _node(r"""
      const card = new Style([decl('border', '1px solid color-mix(in srgb,var(--accent) 12%,transparent)')]);
      document.styleSheets = [{ cssRules: [{ selectorText: '.card', style: card }] }];
      global.CSS = { supports: () => true };
      __POLYFILL__
      process.stdout.write(JSON.stringify({ active: window.EU_LEGACY_COLORS.active, calls: card.calls }));
    """)
    assert shown == {"active": False, "calls": []}


def test_the_runtime_conversion_matches_the_static_hex_for_every_token() -> None:
    tokens = (STATIC / "css" / "tokens.css").read_text(encoding="utf-8")
    values = re.findall(r"oklch\(([\d.]+)%\s+([\d.]+)\s+([\d.]+)\)", tokens)
    assert len(values) >= 8
    shown = _node(r"""
      global.CSS = { supports: () => true };
      __POLYFILL__
      const values = JSON.parse(process.argv[1] || '[]');
      process.stdout.write(JSON.stringify(%s.map(v => window.EU_LEGACY_COLORS.parseColor(`oklch(${v[0]}%% ${v[1]} ${v[2]})`).slice(0, 3).map(Math.round))));
    """ % json.dumps(values))
    assert shown == [list(_oklch_srgb(float(lightness), float(chroma), float(hue))) for lightness, chroma, hue in values]
