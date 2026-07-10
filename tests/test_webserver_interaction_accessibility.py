"""Keyboard and naming contracts for critical native WebApp interactions."""

from pathlib import Path
import re


STATIC_JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
    / "js"
)
STATIC_CSS = STATIC_JS.parent / "css"


def _js(name: str) -> str:
    return (STATIC_JS / name).read_text(encoding="utf-8")


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    high, low = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def test_confirmation_dialog_is_named_trapped_and_restores_focus() -> None:
    source = _js("i18n.js")

    assert 'role="dialog" aria-modal="true" aria-labelledby="' in source
    assert 'aria-describedby="' in source
    assert "const priorFocus = document.activeElement" in source
    assert "if (e.key === 'Escape')" in source
    assert "if (e.key !== 'Tab'" in source
    assert "document.removeEventListener('keydown', onKeydown, true)" in source
    assert "priorFocus.focus()" in source


def test_guided_composer_has_accessible_input_and_send_names() -> None:
    source = _js("screens-guided.js")

    assert 'id="gdInput"' in source
    assert "aria-label=\"${t('Message Guided Copilot'" in source
    assert 'type="button" class="gd-send"' in source
    assert "aria-label=\"${t('Send message'" in source


def test_muted_text_tokens_meet_aa_on_sunken_surface() -> None:
    source = (STATIC_CSS / "tokens.css").read_text(encoding="utf-8")

    def token(name: str) -> str:
        match = re.search(rf"--{re.escape(name)}:\s*(#[0-9A-Fa-f]{{6}})", source)
        assert match, name
        return match.group(1)

    background = token("surface-2")
    assert _contrast(token("ink-3"), background) >= 4.5
    assert _contrast(token("ink-4"), background) >= 4.5
