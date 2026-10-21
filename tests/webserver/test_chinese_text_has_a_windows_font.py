"""Chinese text names a Windows font in every font stack that ends in a generic.

The hospital's clinicians use Chrome on Chinese Windows, and the page loads no
webfonts. For a Chinese glyph that no named font covers, Chrome falls back to
its per-script default: NSimSun (新宋体) for monospace. Each stack therefore
names PingFang SC (macOS) and then Microsoft YaHei (Windows) before the generic
family, so macOS keeps its glyphs and Windows gets a sans face.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"

_CSS_DECLARATION = re.compile(r"(?P<prop>--[\w-]+|font-family|font)\s*:\s*(?P<value>[^;{}]+)")
_JS_FONT_FAMILY = re.compile(r"fontFamily:\s*'(?P<value>[^']+)'")


def _stacks() -> list[tuple[str, str]]:
    stacks = []
    for path in sorted((STATIC / "css").glob("*.css")):
        for match in _CSS_DECLARATION.finditer(path.read_text(encoding="utf-8")):
            value = match["value"].strip()
            prop = match["prop"]
            if prop.startswith("--") and not re.search(r"font|mono", prop):
                continue
            if value.startswith("var(") or not re.search(r",\s*(monospace|sans-serif)$", value):
                continue
            stacks.append((f"{path.name} {prop}", value))
    for path in sorted((STATIC / "js").glob("*.js")):
        for match in _JS_FONT_FAMILY.finditer(path.read_text(encoding="utf-8")):
            stacks.append((f"{path.name} fontFamily", match["value"]))
    return stacks


def _requires_chinese_font(value: str) -> bool:
    # Every monospace stack does; a sans stack does when it is a shared token
    # or names system fonts that cover no Chinese.
    return value.endswith("monospace") or "system-ui" in value


def test_every_monospace_and_system_stack_names_pingfang_then_yahei() -> None:
    stacks = [(where, value) for where, value in _stacks() if _requires_chinese_font(value)]
    assert len(stacks) >= 10, stacks
    missing = []
    for where, value in stacks:
        pingfang = value.find("PingFang SC")
        yahei = value.find("Microsoft YaHei")
        generic = value.rfind(",")
        if not (0 <= pingfang < yahei < generic):
            missing.append((where, value))
    assert missing == []


def test_the_shared_tokens_are_among_the_checked_stacks() -> None:
    names = {where for where, value in _stacks() if _requires_chinese_font(value)}
    assert {
        "tokens.css --font-sans",
        "tokens.css --font-mono",
        "guided-pi-skin.css --skin-font",
        "guided-pi-skin.css --skin-mono",
        "workspace-canvas.css --canvas-mono",
    } <= names
