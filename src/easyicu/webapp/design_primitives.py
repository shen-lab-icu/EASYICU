"""Shell-A design primitives shared across redesigned pages.

These helpers emit HTML chrome that matches ``easyicu design/page-misc.jsx``
and ``page-ai-chat.jsx``. Streamlit cannot place real widgets inside
arbitrary HTML cards, so the convention is: emit the card chrome with
``render_card_open`` / ``render_card_close`` around real Streamlit
widgets (``st.selectbox``, ``st.checkbox``, ``st.text_input``), and use
the visual-only helpers (``render_seg_static``, ``render_toggle_static``)
only when the row is for display, not user input.

The page header helper lives in ``cohort_charts.render_design_page_header``;
import it directly from there to keep one source of truth.
"""

from __future__ import annotations

import html as _html
import json as _json
from typing import Sequence


def _esc(value: object) -> str:
    return _html.escape(str(value), quote=True)


# ---------------------------------------------------------------------
# FilterRow — label (en + zh) | control
# ---------------------------------------------------------------------

def render_filter_row(
    *,
    label_en: str,
    label_zh: str,
    control_html: str,
    lang: str = "en",
) -> str:
    """Match page-misc.jsx ``FilterRow``: 130px label column + control."""
    label = label_zh if lang == "zh" else label_en
    sub = label_en if lang == "zh" else label_zh
    return (
        '<div style="display:flex;align-items:center;gap:14px;'
        'padding:8px 0;border-top:1px solid var(--hair)">'
        '<div style="min-width:130px;display:flex;flex-direction:column;gap:2px">'
        f'<span style="font-size:11.5px;color:var(--ink-2);font-weight:500">'
        f'{_esc(label)}</span>'
        f'<span class="mono" style="font-size:10.5px;color:var(--ink-4)">'
        f'{_esc(sub)}</span>'
        '</div>'
        f'<div style="flex:1;min-width:0">{control_html}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------
# Seg — segmented pill (visual only)
# ---------------------------------------------------------------------

def render_seg_static(value: str, options: Sequence[tuple[str, str]]) -> str:
    """Static segmented control. ``options`` is ``[(value, label), ...]``.

    For interactive use, render ``st.radio`` / ``st.selectbox`` below
    instead — this is purely a visual indicator.
    """
    parts = []
    for v, l in options:
        active = v == value
        bg = "var(--ink)" if active else "var(--surface)"
        fg = "#fff" if active else "var(--ink-3)"
        border = "var(--ink)" if active else "var(--hair-2)"
        parts.append(
            f'<span style="display:inline-flex;align-items:center;height:24px;'
            f'padding:0 10px;border-radius:var(--r-2);border:1px solid {border};'
            f'background:{bg};color:{fg};font-size:11.5px;font-weight:500;'
            f'letter-spacing:0">{_esc(l)}</span>'
        )
    return (
        '<div style="display:inline-flex;gap:4px;flex-wrap:wrap;align-items:center">'
        + "".join(parts) +
        '</div>'
    )


# ---------------------------------------------------------------------
# Toggle indicator (visual only)
# ---------------------------------------------------------------------

def render_toggle_static(*, on: bool, label_en: str, label_zh: str = "", lang: str = "en") -> str:
    color = "var(--ink-2)" if on else "var(--ink-4)"
    knob_left = "16px" if on else "2px"
    track_bg = "var(--ink)" if on else "var(--hair-2)"
    label = label_zh if lang == "zh" and label_zh else label_en
    return (
        '<label style="display:inline-flex;align-items:center;gap:6px;'
        f'font-size:12px;color:{color}">'
        f'<span style="position:relative;width:30px;height:16px;border-radius:999px;'
        f'background:{track_bg};display:inline-block">'
        f'<span style="position:absolute;top:1px;left:{knob_left};width:14px;'
        f'height:14px;border-radius:50%;background:#fff;'
        f'box-shadow:0 1px 2px rgba(0,0,0,.18);transition:left .12s ease"></span>'
        '</span>'
        f'{_esc(label)}'
        '</label>'
    )


# ---------------------------------------------------------------------
# Card chrome (matches page-misc.jsx .card style)
# ---------------------------------------------------------------------

def render_card_open(
    *,
    padding: str = "14px 16px",
    flush: bool = False,
    sunken: bool = False,
    extra_style: str = "",
) -> str:
    """Open a ``.eu-card``. Always pair with ``render_card_close()``."""
    pad = "0" if flush else padding
    bg = "var(--surface-2)" if sunken else "var(--surface)"
    # Match the canonical .eu-card radius (var(--r-3) = 10px) so design
    # primitives and shell_overrides cards land on the same value.
    return (
        '<div style="background:' + bg + ';border:1px solid var(--hair);'
        'border-radius:var(--r-3);padding:' + pad + ';' + extra_style + '">'
    )


def render_card_close() -> str:
    return '</div>'


def render_card_header(
    *,
    title_en: str,
    title_zh: str = "",
    right_html: str = "",
    lang: str = "en",
    sub_mono: str = "",
) -> str:
    title = title_zh if lang == "zh" and title_zh else title_en
    sub_html = (
        f'<span class="mono" style="margin-left:8px;color:var(--ink-4);font-weight:400">'
        f'{_esc(sub_mono)}</span>'
        if sub_mono else ""
    )
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'padding:10px 14px;border-bottom:1px solid var(--hair)">'
        f'<div style="font-size:12.5px;font-weight:500;color:var(--ink)">'
        f'{_esc(title)}{sub_html}</div>'
        f'<div style="display:flex;align-items:center;gap:6px">{right_html}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------
# Mono code-style preview (manifest, json, command)
# ---------------------------------------------------------------------

def render_mono_preview(
    *,
    text: str,
    dark: bool = True,
    height_px: int | None = None,
) -> str:
    """Code-block style preview matching page-misc Manifest preview."""
    if isinstance(text, (dict, list)):
        text = _json.dumps(text, indent=2, ensure_ascii=False)
    bg = "var(--ink)" if dark else "var(--surface-2)"
    fg = "#E8E6DD" if dark else "var(--ink)"
    height_css = f"max-height:{height_px}px;overflow-y:auto;" if height_px else ""
    return (
        f'<pre class="mono" style="margin:0;padding:10px 12px;background:{bg};'
        f'color:{fg};border-radius:var(--r-2);font-size:11px;line-height:1.55;'
        f'overflow-x:auto;white-space:pre;{height_css}">'
        f'{_esc(text)}</pre>'
    )


# ---------------------------------------------------------------------
# Bilingual bullet list (matches the "What this reproduces" card)
# ---------------------------------------------------------------------

def render_bullet_list(items: Sequence[str]) -> str:
    lis = "".join(
        '<li style="font-size:12px;color:var(--ink-2);padding:2px 0">'
        f'{_esc(item)}</li>'
        for item in items
    )
    return f'<ul style="margin:0;padding:0 0 0 16px">{lis}</ul>'


# ---------------------------------------------------------------------
# Status list (used by Convert dialog progress + log lines)
# ---------------------------------------------------------------------

_STATUS_COLORS = {
    "ok": "var(--ok)",
    "active": "var(--ink-2)",
    "queued": "var(--ink-4)",
    "fail": "var(--bad)",
}


def render_status_list(items: Sequence[tuple[str, str]]) -> str:
    """``items`` is ``[(status, text), ...]`` where status is ok/active/queued/fail."""
    lis = []
    for status, text in items:
        color = _STATUS_COLORS.get(status, "var(--ink-3)")
        lis.append(
            f'<li style="font-size:12px;color:{color};padding:2px 0;list-style:none;'
            f'display:flex;align-items:center;gap:8px">'
            f'<span class="mono">{_esc(text)}</span></li>'
        )
    return f'<ul style="margin:0;padding:0">{"".join(lis)}</ul>'


# ---------------------------------------------------------------------
# Progress bar (matches Convert dialog .bar)
# ---------------------------------------------------------------------

def render_design_progress(
    *,
    pct: float,
    label_en: str,
    label_zh: str = "",
    right_text: str = "",
    sub_left: str = "",
    sub_right: str = "",
    lang: str = "en",
) -> str:
    pct = max(0.0, min(1.0, float(pct)))
    label = label_zh if lang == "zh" and label_zh else label_en
    fg = "var(--ink)"
    return (
        '<div>'
        '<div style="display:flex;justify-content:space-between;align-items:baseline;'
        'margin-bottom:6px">'
        f'<span style="font-size:11.5px;font-weight:500;color:var(--ink-2)">'
        f'{_esc(label)}</span>'
        f'<span class="mono" style="font-size:11px;color:var(--ink-3)">'
        f'{_esc(right_text)}</span>'
        '</div>'
        f'<div style="height:6px;background:var(--hair-2);border-radius:var(--r-1);overflow:hidden">'
        f'<div style="width:{pct*100:.1f}%;height:100%;background:{fg}"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;margin-top:6px;'
        'font-size:10.5px;color:var(--ink-4)" class="mono">'
        f'<span>{_esc(sub_left)}</span><span>{_esc(sub_right)}</span>'
        '</div>'
        '</div>'
    )
