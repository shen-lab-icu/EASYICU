"""Inline SVG mini-charts for the EasyICU shell-A redesign.

These primitives mirror the design-canvas / page-cohort-subtabs charts:
they are pure SVG strings driven by simple Python data (lists / dicts)
and styled with the design tokens defined in ``tokens.css``
(``--ink``, ``--accent``, ``--warn``, ``--bad``, ``--hair-2``,
``--ink-3``, ``--ink-4``).

Cohort / Cross-DB pages call these helpers in place of matplotlib /
plotly so the on-screen charts visually match the design while still
being driven by the real session-state data.

Each helper returns an HTML/SVG string (never renders directly) so the
page module can compose multiple charts inside a single
``st.markdown(..., unsafe_allow_html=True)`` call.
"""

from __future__ import annotations

import hashlib
import html
import math
from collections.abc import Sequence
from typing import Iterable


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def _stable_hash(label: str, mod: int) -> int:
    """Deterministic 0..mod-1 pseudo-random index keyed by label."""
    h = hashlib.sha1(label.encode("utf-8")).digest()
    return h[0] % mod


# ---------------------------------------------------------------------
# Stat card grids (cohort hero rows)
# ---------------------------------------------------------------------

def render_stat_grid(
    cards: Sequence[tuple[str, str, str, str]],
    *,
    columns: int = 4,
) -> str:
    """Render an N-column stat-card grid.

    Each card is ``(label, value, sub, tone)`` where tone is one of
    ``""`` / ``"ok"`` / ``"warn"`` / ``"bad"`` / ``"info"``.
    """
    if not cards:
        return ""
    items = []
    for label, value, sub, tone in cards:
        tone_class = f" {tone}" if tone else ""
        items.append(
            '<div class="eu-card eu-stat">'
            f'<div class="label">{_esc(label)}</div>'
            f'<div class="val{tone_class}">{_esc(value)}</div>'
            f'<div class="delta">{_esc(sub)}</div>'
            '</div>'
        )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({columns},1fr);gap:12px">'
        + "".join(items)
        + "</div>"
    )


def render_compact_kpi_grid(
    cards: Sequence[tuple[str, str, str]],
    *,
    columns: int = 6,
) -> str:
    """Inline-style mini KPI cells used in the Snapshot 6-up grid.

    Each tuple is ``(label, value, tone)``.
    """
    if not cards:
        return ""
    cells = []
    for label, value, tone in cards:
        color = {
            "bad": "var(--bad)",
            "warn": "var(--warn)",
            "ok": "var(--ok)",
            "info": "var(--info)",
        }.get(tone, "var(--ink)")
        cells.append(
            '<div style="padding:10px;background:var(--surface-2);border-radius:6px">'
            f'<div style="font-size:9.5px;color:var(--ink-4);letter-spacing:.06em;'
            f'text-transform:uppercase;font-weight:500">{_esc(label)}</div>'
            f'<div class="mono" style="font-size:16px;font-weight:500;margin-top:2px;'
            f'color:{color};font-family:var(--font-mono);letter-spacing:-0.01em">{_esc(value)}</div>'
            '</div>'
        )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({columns},1fr);gap:10px">'
        + "".join(cells)
        + "</div>"
    )


# ---------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------

def render_bar_chart(
    values: Sequence[float],
    *,
    width: int = 240,
    height: int = 80,
    color: str = "var(--ink)",
    opacity: float = 1.0,
    bar_gap: int = 3,
) -> str:
    """Solid bar histogram from a flat value list."""
    if not values:
        return ""
    vmax = max(values) or 1.0
    n = len(values)
    bar_w = max(1.0, (width - bar_gap * (n + 1)) / n)
    bars = []
    for i, v in enumerate(values):
        h = (v / vmax) * (height - 10)
        x = bar_gap + i * (bar_w + bar_gap)
        bars.append(
            f'<rect x="{x:.2f}" y="{(height - 5 - h):.2f}" '
            f'width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" '
            f'opacity="{opacity}" rx="1"/>'
        )
    return (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none">{"".join(bars)}</svg>'
    )


def render_quartile_bars(
    values: Sequence[float],
    *,
    width: int = 240,
    height: int = 80,
    labels: Sequence[str] = ("Q1", "Q2", "Q3", "Q4"),
) -> str:
    """Quartile bars with mono labels under each bar (Mortality by Q)."""
    if not values:
        return ""
    vmax = max(values) or 1.0
    n = len(values)
    bar_w = 34
    slot = (width - 40) / max(n, 1)
    bars = []
    for i, v in enumerate(values):
        h = (v / vmax) * (height - 15)
        x = 20 + i * slot + (slot - bar_w) / 2
        bars.append(
            f'<rect x="{x:.2f}" y="{(height - 5 - h):.2f}" '
            f'width="{bar_w}" height="{h:.2f}" fill="var(--ink)" rx="2"/>'
        )
        bars.append(
            f'<text x="{(x + bar_w / 2):.2f}" y="{(height - 5):.2f}" '
            f'font-size="9" fill="var(--ink-4)" text-anchor="middle" '
            f'font-family="var(--font-mono)" dy="9">{_esc(labels[i] if i < len(labels) else "")}</text>'
        )
    return (
        f'<svg width="100%" height="{height + 12}" '
        f'viewBox="0 0 {width} {height + 12}" preserveAspectRatio="none">'
        f'{"".join(bars)}</svg>'
    )


def render_grouped_bars(
    groups: Sequence[tuple[str, float, float]],
    *,
    width: int = 540,
    height: int = 210,
    a_label: str = "Sepsis",
    b_label: str = "Non-sepsis",
    a_color: str = "var(--ink)",
    b_color: str = "var(--accent)",
    y_max: float | None = None,
    y_unit: str = "%",
    legend: bool = True,
) -> str:
    """Two-series grouped bars (e.g. Mortality by SOFA quartile).

    ``groups`` is a list of ``(label, a_value, b_value)``.
    """
    if not groups:
        return ""
    vmax = y_max or max(max(a, b) for _, a, b in groups) or 1.0
    rounded = math.ceil(vmax / 15) * 15 if y_unit == "%" else vmax
    inner_h = height - 50
    base_y = height - 40
    gridlines = []
    ticks = 4
    for i in range(ticks + 1):
        y = base_y - (i * inner_h / ticks)
        val = (i / ticks) * rounded
        if i > 0:
            gridlines.append(
                f'<line x1="36" y1="{y:.2f}" x2="{width - 12}" y2="{y:.2f}" '
                f'stroke="var(--hair)" stroke-dasharray="2 4"/>'
            )
        gridlines.append(
            f'<text x="30" y="{(y + 4):.2f}" text-anchor="end" font-size="10" '
            f'fill="var(--ink-4)" font-family="var(--font-mono)">'
            f'{int(val)}{y_unit}</text>'
        )
    gridlines.append(
        f'<line x1="36" y1="{base_y}" x2="{width - 12}" y2="{base_y}" '
        f'stroke="var(--hair-2)" stroke-width="1"/>'
    )
    gridlines.append(
        f'<line x1="36" y1="20" x2="36" y2="{base_y}" '
        f'stroke="var(--hair-2)" stroke-width="1"/>'
    )

    n = len(groups)
    avail_w = width - 60
    slot = avail_w / n
    bar_w = min(34, slot * 0.3)
    bars = []
    for i, (label, a, b) in enumerate(groups):
        h_a = max(0, (a / rounded) * inner_h)
        h_b = max(0, (b / rounded) * inner_h)
        x = 50 + i * slot + (slot - 2 * bar_w - 4) / 2
        bars.append(
            f'<rect x="{x:.2f}" y="{(base_y - h_a):.2f}" '
            f'width="{bar_w:.2f}" height="{h_a:.2f}" fill="{a_color}" rx="2"/>'
        )
        bars.append(
            f'<rect x="{(x + bar_w + 4):.2f}" y="{(base_y - h_b):.2f}" '
            f'width="{bar_w:.2f}" height="{h_b:.2f}" fill="{b_color}" rx="2" opacity="0.7"/>'
        )
        bars.append(
            f'<text x="{(x + bar_w + 2):.2f}" y="{(base_y + 16):.2f}" '
            f'text-anchor="middle" font-size="10.5" fill="var(--ink-3)" '
            f'font-family="var(--font-mono)">{_esc(label)}</text>'
        )

    legend_html = ""
    if legend:
        lx = width - 200
        legend_html = (
            f'<g transform="translate({lx}, 6)">'
            f'<rect width="10" height="10" fill="{a_color}" rx="2"/>'
            f'<text x="14" y="9" font-size="10.5" fill="var(--ink-3)">{_esc(a_label)}</text>'
            f'<rect x="78" width="10" height="10" fill="{b_color}" opacity="0.7" rx="2"/>'
            f'<text x="92" y="9" font-size="10.5" fill="var(--ink-3)">{_esc(b_label)}</text>'
            '</g>'
        )

    return (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="margin-top:8px">{"".join(gridlines)}{"".join(bars)}{legend_html}</svg>'
    )


# ---------------------------------------------------------------------
# Coverage / availability matrices
# ---------------------------------------------------------------------

def render_coverage_matrix(
    rows: Sequence[tuple[str, Sequence[float]]],
    *,
    cell_w: int = 22,
    cell_gap: int = 4,
    row_h: int = 21,
    label_w: int = 66,
) -> str:
    """Concepts × patients heat grid (Coverage matrix).

    ``rows`` is a list of ``(concept_label, [coverage_score 0..1] per patient)``.
    Cells colored ``var(--ink)`` (present), ``var(--warn)``
    (partial), ``var(--bad)`` (absent).
    """
    if not rows:
        return ""
    n_cols = max(len(r[1]) for r in rows)
    total_w = label_w + n_cols * cell_w + (n_cols - 1) * (cell_gap - 18)
    total_h = len(rows) * row_h + 10
    parts: list[str] = []
    for r, (label, values) in enumerate(rows):
        parts.append(
            f'<text x="0" y="{(20 + r * row_h):.0f}" font-size="10" '
            f'fill="var(--ink-3)" font-family="var(--font-mono)">{_esc(label)}</text>'
        )
        for c, v in enumerate(values):
            x = label_w + c * (cell_w + 4)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                fill, opacity = "var(--bad)", 0.85
            elif v >= 0.92:
                fill, opacity = "var(--ink)", 0.85
            elif v >= 0.5:
                fill, opacity = "var(--ink)", 0.22 + v * 0.5
            elif v > 0.15:
                fill, opacity = "var(--warn)", 0.7
            else:
                fill, opacity = "var(--bad)", 0.85
            parts.append(
                f'<rect x="{x}" y="{(10 + r * row_h):.0f}" width="{cell_w}" '
                f'height="16" fill="{fill}" opacity="{opacity:.2f}" rx="2"/>'
            )
    legend = (
        '<div style="display:flex;gap:14px;margin-top:8px;font-size:11px;color:var(--ink-3)">'
        '<span style="display:flex;align-items:center;gap:4px">'
        '<span style="width:10px;height:10px;background:var(--ink);opacity:.85;border-radius:2px"></span>present</span>'
        '<span style="display:flex;align-items:center;gap:4px">'
        '<span style="width:10px;height:10px;background:var(--ink);opacity:.5;border-radius:2px"></span>sparse</span>'
        '<span style="display:flex;align-items:center;gap:4px">'
        '<span style="width:10px;height:10px;background:var(--warn);border-radius:2px"></span>partial</span>'
        '<span style="display:flex;align-items:center;gap:4px">'
        '<span style="width:10px;height:10px;background:var(--bad);border-radius:2px"></span>absent</span>'
        '</div>'
    )
    return (
        f'<svg width="100%" height="{total_h}" viewBox="0 0 {total_w} {total_h}" '
        f'preserveAspectRatio="none">{"".join(parts)}</svg>{legend}'
    )


def render_availability_matrix(
    rows: Sequence[tuple[str, Sequence[float]]],
    *,
    columns: Sequence[str],
    cell_h: int = 22,
) -> str:
    """Concept × database availability heat grid (Cross-DB).

    Values in [0, 1]; 1.0 renders a ✓, partial values render a percentage.
    """
    if not rows:
        return ""
    n_cols = len(columns)
    grid_template = f'160px repeat({n_cols}, 1fr)'
    header_cells = [
        f'<div class="mono" style="font-size:10.5px;color:var(--ink-4);'
        f'letter-spacing:.06em;text-transform:uppercase;text-align:center;'
        f'font-family:var(--font-mono)">{_esc(h)}</div>'
        for h in columns
    ]
    body_cells: list[str] = []
    for label, values in rows:
        body_cells.append(
            f'<span style="font-size:12px;color:var(--ink-2)">{_esc(label)}</span>'
        )
        for v in values:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                fill, op, text = "var(--bad)", 0.7, "—"
            elif v >= 0.95:
                fill, op, text = "var(--ink)", 0.85, "✓"
            elif v >= 0.5:
                fill, op, text = "var(--warn)", 0.7, f"{int(round(v * 100))}%"
            else:
                fill, op, text = "var(--bad)", 0.7, f"{int(round(v * 100))}%"
            body_cells.append(
                f'<div style="height:{cell_h}px;background:{fill};opacity:{op};'
                f'border-radius:3px;display:flex;align-items:center;justify-content:center;'
                f'font-size:10.5px;color:#fff;font-family:var(--font-mono)">'
                f'{_esc(text)}</div>'
            )
    return (
        f'<div style="display:grid;grid-template-columns:{grid_template};'
        f'gap:4px;align-items:center">'
        '<div></div>' + "".join(header_cells) + "".join(body_cells) +
        '</div>'
    )


# ---------------------------------------------------------------------
# Mono tables (group contrast, mortality benchmark, reclassification)
# ---------------------------------------------------------------------

def render_mono_table(
    *,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_meta: str = "",
    align: Sequence[str] | None = None,
) -> str:
    """Compact mono-font table inside an .eu-card surface."""
    if align is None:
        align = ["left"] + ["right"] * (len(columns) - 1)
    head = "".join(
        f'<th style="text-align:{align[i]};padding:8px 14px;font-weight:500;'
        f'font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;'
        f'color:var(--ink-4)">{_esc(c)}</th>'
        for i, c in enumerate(columns)
    )
    body_rows = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            color = "var(--ink)" if i == 0 else (
                "var(--ink)" if i == 1 else "var(--ink-3)"
            )
            font_family = (
                "var(--font-sans)" if i == 0 else "var(--font-mono)"
            )
            cells.append(
                f'<td style="padding:8px 14px;text-align:{align[i]};'
                f'color:{color};font-family:{font_family}">{_esc(cell)}</td>'
            )
        body_rows.append(
            f'<tr style="border-top:1px solid var(--hair)">{"".join(cells)}</tr>'
        )
    meta = (
        f'<span class="mono" style="font-size:11px;color:var(--ink-4)">'
        f'{_esc(right_meta)}</span>' if right_meta else ""
    )
    return (
        '<div class="eu-card" style="padding:0;overflow:hidden">'
        '<div style="padding:10px 14px;border-bottom:1px solid var(--hair);'
        'display:flex;justify-content:space-between;align-items:center">'
        f'<div style="font-size:12.5px;font-weight:500">{_esc(title)}</div>'
        f'{meta}</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:12px">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
        '</div>'
    )


# ---------------------------------------------------------------------
# Snapshot composition
# ---------------------------------------------------------------------

def render_snapshot_card(
    *,
    name: str,
    description: str,
    chips: Sequence[str],
    meta: str,
    kpis: Sequence[tuple[str, str, str]],
    inline_charts: Sequence[tuple[str, str]],
) -> str:
    """Snapshot composite card matching ``Cohort_Snapshot`` in the design.

    ``inline_charts`` is a list of ``(title, svg_html)`` rendered as a
    3-up row beneath the KPI grid.
    """
    chip_html = "".join(
        f'<span class="eu-chip mono" style="font-size:10.5px">{_esc(c)}</span>'
        for c in chips
    )
    kpi_html = render_compact_kpi_grid(list(kpis), columns=6)
    charts_html = "".join(
        '<div>'
        f'<div style="font-size:11.5px;color:var(--ink-3)">{_esc(t)}</div>'
        f'{svg}'
        '</div>'
        for t, svg in inline_charts
    )
    return (
        '<div class="eu-card" style="padding:22px 26px">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px">'
        '<div>'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4);'
        f'letter-spacing:.06em;text-transform:uppercase">{_esc(meta)}</div>'
        f'<h2 style="margin:6px 0 4px;font-size:22px;font-weight:500;'
        f'letter-spacing:-0.015em">{_esc(name)}</h2>'
        f'<div style="font-size:12.5px;color:var(--ink-3)">{_esc(description)}</div>'
        '</div>'
        f'<div style="display:flex;gap:4px;flex-wrap:wrap;justify-content:flex-end;'
        f'max-width:40%">{chip_html}</div>'
        '</div>'
        f'<div style="margin-top:22px">{kpi_html}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);'
        f'gap:14px;margin-top:18px">{charts_html}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------
# Reclassification (SOFA Δ)
# ---------------------------------------------------------------------

def render_definition_pair(
    *,
    title: str,
    left: tuple[str, str],
    right: tuple[str, str],
) -> str:
    """Two side-by-side definition tiles (SOFA-1 / SOFA-2)."""
    return (
        '<div class="eu-card" style="padding:14px">'
        f'<div style="font-size:12.5px;font-weight:500;margin-bottom:8px">'
        f'{_esc(title)}</div>'
        '<div style="display:flex;gap:12px">'
        '<div style="flex:1;padding:12px;background:var(--surface-2);border-radius:6px">'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4)">{_esc(left[0])}</div>'
        f'<div style="font-size:13px;margin-top:4px">{_esc(left[1])}</div>'
        '</div>'
        '<div style="flex:1;padding:12px;background:var(--surface-2);border-radius:6px">'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4)">{_esc(right[0])}</div>'
        f'<div style="font-size:13px;margin-top:4px">{_esc(right[1])}</div>'
        '</div>'
        '</div></div>'
    )


def render_effect_summary(
    *,
    title: str,
    cells: Sequence[tuple[str, str, str]],
    footnote: str = "",
) -> str:
    """Cohort effect tile (Sepsis · SOFA-1 / SOFA-2 / Δ)."""
    grid = render_compact_kpi_grid(list(cells), columns=len(cells) or 1)
    foot = (
        f'<div style="margin-top:10px;font-size:11.5px;color:var(--ink-3)">'
        f'{_esc(footnote)}</div>' if footnote else ""
    )
    return (
        '<div class="eu-card" style="padding:14px">'
        f'<div style="font-size:12.5px;font-weight:500;margin-bottom:8px">'
        f'{_esc(title)}</div>{grid}{foot}</div>'
    )


def render_reclassification_table(
    *,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    n_total: int | None = None,
) -> str:
    right_meta = f"n={n_total:,}" if n_total is not None else ""
    return render_mono_table(
        title=title,
        columns=columns,
        rows=rows,
        right_meta=right_meta,
    )


# ---------------------------------------------------------------------
# Active databases row (Cross-DB)
# ---------------------------------------------------------------------

def render_active_databases(
    databases: Sequence[tuple[str, str, bool, bool]],
) -> str:
    """Active-database cards for the Cross-DB benchmark.

    Each tuple is ``(name, sub_meta, ready_ok, is_primary)``.
    """
    if not databases:
        return ""
    cards = []
    for name, sub, ok, is_primary in databases:
        border = ("1.5px solid var(--ink)" if is_primary
                  else "1px solid var(--hair)")
        pill_class = "eu-pill ok" if ok else "eu-pill warn"
        pill_label = "ready" if ok else "missing"
        cards.append(
            f'<div style="flex:1;padding:10px 12px;border:{border};'
            f'border-radius:8px;background:var(--surface);display:flex;'
            f'align-items:center;gap:10px">'
            '<div style="width:26px;height:26px;border-radius:6px;'
            'background:var(--surface-2);color:var(--ink-3);display:flex;'
            'align-items:center;justify-content:center">'
            '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            'stroke-linejoin="round">'
            '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
            '<path d="M3 5v6c0 1.7 4 3 9 3s9-1.3 9-3V5"/>'
            '<path d="M3 11v6c0 1.7 4 3 9 3s9-1.3 9-3v-6"/>'
            '</svg></div>'
            '<div>'
            f'<div style="font-size:12.5px;font-weight:500">{_esc(name)}</div>'
            f'<div class="mono" style="font-size:10.5px;color:var(--ink-4)">'
            f'{_esc(sub)}</div></div>'
            f'<span class="{pill_class}" style="margin-left:auto">'
            f'<span class="dot"></span>{_esc(pill_label)}</span>'
            '</div>'
        )
    return (
        '<div class="eu-card" style="padding:12px 14px">'
        f'<div class="eu-section-label" style="padding:0;margin-bottom:8px">'
        f'<span>Active databases · {len(databases)} active</span></div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap">'
        + "".join(cards)
        + '</div></div>'
    )


# ---------------------------------------------------------------------
# Sub-tabs (purely visual; routing handled by caller)
# ---------------------------------------------------------------------

def render_subtabs(items: Sequence[str], *, active: str) -> str:
    """Render a flat sub-tab strip (visual only)."""
    bars = []
    for it in items:
        is_active = it == active
        bars.append(
            f'<div style="padding:8px 14px;font-size:12.5px;'
            f'color:{"var(--ink)" if is_active else "var(--ink-3)"};'
            f'border-bottom:{"2px solid var(--ink)" if is_active else "2px solid transparent"};'
            f'margin-bottom:-1px;font-weight:{500 if is_active else 400}">'
            f'{_esc(it)}</div>'
        )
    return (
        '<div style="border-bottom:1px solid var(--hair);display:flex;gap:0">'
        + "".join(bars)
        + '</div>'
    )


# ---------------------------------------------------------------------
# Pre-baked deterministic demo coverage matrix
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Design-system PageHeader (kicker + bilingual title + desc + right)
# ---------------------------------------------------------------------

def render_design_page_header(
    *,
    kicker: str,
    title_en: str,
    title_zh: str,
    desc: str,
    right_html: str = "",
) -> str:
    return (
        '<div style="margin-bottom:6px">'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4);'
        f'letter-spacing:0.06em;text-transform:uppercase">{_esc(kicker)}</div>'
        '<div style="display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-top:6px">'
        '<div>'
        f'<h1 style="margin:0;font-size:22px;font-weight:500;letter-spacing:-0.015em;color:var(--ink)">'
        f'{_esc(title_en)} <span class="eu-cn" style="color:var(--ink-3);font-weight:400">{_esc(title_zh)}</span>'
        '</h1>'
        f'<div style="margin-top:4px;color:var(--ink-3);font-size:12.5px">{_esc(desc)}</div>'
        '</div>'
        f'<div style="display:flex;gap:6px;flex-shrink:0">{right_html}</div>'
        '</div></div>'
    )


# ---------------------------------------------------------------------
# Tutorial — 4-step workflow strip
# ---------------------------------------------------------------------

_STEP_ICONS = {
    "database": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v6c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 11v6c0 1.7 4 3 9 3s9-1.3 9-3v-6"/></svg>',
    "users":    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="8" r="3.5"/><circle cx="17" cy="9" r="2.5"/><path d="M3 19a6 6 0 0 1 12 0"/><path d="M14 19c0-1.6.8-3 2-3.8"/></svg>',
    "layers":   '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/><path d="m3 18 9 5 9-5"/></svg>',
    "bars":     '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-8"/><path d="M22 20v-5"/></svg>',
    "flask":    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6"/><path d="M10 3v6L4 20a1 1 0 0 0 .9 1.5h14.2A1 1 0 0 0 20 20l-6-11V3"/><path d="M8 14h8"/></svg>',
    "file":     '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/></svg>',
    "book":     '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H20v15H5.5A1.5 1.5 0 0 0 4 19.5v-15Z"/></svg>',
    "chevron":  '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>',
    "chevronR": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>',
}


def render_step_card(
    *,
    number: str,
    icon: str,
    label_en: str,
    label_zh: str,
    desc: str,
    sub: str,
) -> str:
    return (
        '<div style="flex:1;border:1px solid var(--hair);background:var(--surface);'
        'border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;gap:10px;'
        'position:relative;min-width:0">'
        '<div style="display:flex;align-items:center;gap:10px">'
        f'<div class="mono" style="width:26px;height:26px;border-radius:6px;background:var(--ink);'
        f'color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;'
        f'font-weight:500;font-family:var(--font-mono)">{_esc(number)}</div>'
        f'<div style="color:var(--ink-3)">{_STEP_ICONS.get(icon, "")}</div>'
        f'<div class="mono" style="margin-left:auto;font-size:10.5px;color:var(--ink-4);'
        f'font-family:var(--font-mono)">{_esc(sub)}</div>'
        '</div>'
        '<div>'
        f'<div style="font-size:14px;font-weight:500">{_esc(label_en)}</div>'
        f'<div class="eu-cn" style="font-size:11px;color:var(--ink-4)">{_esc(label_zh)}</div>'
        '</div>'
        f'<div style="font-size:12px;color:var(--ink-3);line-height:1.5">{_esc(desc)}</div>'
        '</div>'
    )


def render_workflow_strip(steps: Sequence[dict]) -> str:
    chevron = (
        '<div style="display:flex;align-items:center;color:var(--ink-4)">'
        f'{_STEP_ICONS["chevronR"]}</div>'
    )
    parts: list[str] = []
    for i, s in enumerate(steps):
        if i > 0:
            parts.append(chevron)
        parts.append(render_step_card(**s))
    return (
        '<div style="display:flex;gap:12px;align-items:stretch">'
        + "".join(parts)
        + '</div>'
    )


def render_tutorial_starting_card(
    *,
    tone: str,
    icon: str,
    title_en: str,
    title_zh: str,
    badge_html: str,
    desc: str,
    bullets: Sequence[str],
    cta_label: str = "",  # accepted for backwards compatibility but unused
    cta_primary: bool = False,  # noqa: ARG001 — see CTA buttons rendered as real st.button by caller
    cta_dashed: bool = False,  # noqa: ARG001
) -> str:
    """Tutorial starting-point card body — *no embedded CTA button*.

    The CTA itself is rendered as a real ``st.button`` immediately
    below the card so the click actually wires through to the
    session_state mode switch. Embedding a decorative HTML button
    inside the card would just duplicate the visual without any
    behaviour.
    """
    border = (
        "border:1px solid var(--accent-border);background:linear-gradient(180deg,var(--accent-soft),var(--surface))"
        if tone == "accent" else
        "border:1px solid var(--hair);background:var(--surface)"
    )
    bullets_html = "".join(
        '<li style="display:flex;gap:8px;padding:4px 0;font-size:12.5px;color:var(--ink-2)">'
        f'<span style="color:var(--ink-4);margin-top:3px">·</span><span>{_esc(b)}</span></li>'
        for b in bullets
    )
    title_color = "var(--accent-ink)" if tone == "accent" else "var(--ink)"
    return (
        f'<div class="eu-card" style="padding:18px;display:flex;flex-direction:column;gap:10px;'
        f'min-height:240px;{border}">'
        '<div style="display:flex;align-items:center;gap:10px">'
        f'<span style="color:{title_color}">{_STEP_ICONS.get(icon, "")}</span>'
        f'<div style="font-size:14px;font-weight:500;color:{title_color}">{_esc(title_en)}'
        f' <span class="eu-cn" style="font-weight:400;margin-left:6px">{_esc(title_zh)}</span></div>'
        f'<span style="margin-left:auto">{badge_html}</span>'
        '</div>'
        f'<div style="font-size:12.5px;color:var(--ink-2);line-height:1.55">{_esc(desc)}</div>'
        f'<ul style="margin:0;padding:0;list-style:none">{bullets_html}</ul>'
        '</div>'
    )


# ---------------------------------------------------------------------
# Quick Viz · time-series lane
# ---------------------------------------------------------------------

def render_lane(
    *,
    title_en: str,
    title_zh: str,
    unit: str,
    data: Sequence[float],
    threshold: float | None = None,
) -> str:
    if not data:
        return ""
    last = data[-1]
    mu = sum(data) / len(data)
    # Map data to 0-40 plot area (invert so larger = upper)
    vmin, vmax = min(data), max(data)
    span = (vmax - vmin) or 1
    pts = []
    for i, v in enumerate(data):
        x = (i / (len(data) - 1)) * 400 if len(data) > 1 else 0
        y = 35 - ((v - vmin) / span) * 30
        pts.append(f"{x:.2f},{y:.2f}")
    threshold_html = (
        f'<line x1="0" y1="{threshold:.2f}" x2="400" y2="{threshold:.2f}" '
        f'stroke="var(--warn)" stroke-dasharray="3 3" opacity="0.5"/>'
        if threshold is not None else ""
    )
    return (
        '<div style="border-top:1px solid var(--hair);padding:10px 14px;display:grid;'
        'grid-template-columns:160px 1fr 84px;align-items:center;gap:12px">'
        '<div>'
        f'<div style="font-size:12.5px;font-weight:500">{_esc(title_en)}</div>'
        f'<div class="mono" style="font-size:10.5px;color:var(--ink-4)">{_esc(title_zh)} · {_esc(unit)}</div>'
        '</div>'
        '<svg width="100%" height="40" viewBox="0 0 400 40" preserveAspectRatio="none">'
        f'{threshold_html}'
        f'<polyline fill="none" stroke="var(--ink)" stroke-width="1.5" points="{" ".join(pts)}"/>'
        '</svg>'
        f'<div class="mono" style="font-size:11.5px;color:var(--ink);text-align:right">'
        f'<div>{last}<span style="color:var(--ink-4);margin-left:3px">{_esc(unit)}</span></div>'
        f'<div style="color:var(--ink-4);font-size:10px">μ {mu:.1f}</div>'
        '</div></div>'
    )


def render_lane_group(title_en: str, meta: str, lanes_html: str) -> str:
    return (
        '<div class="eu-card" style="padding:0;overflow:hidden">'
        '<div style="padding:10px 14px;display:flex;align-items:center;justify-content:space-between">'
        f'<div style="font-size:12px;font-weight:500">{_esc(title_en)}</div>'
        f'<span class="mono" style="font-size:10.5px;color:var(--ink-4)">{_esc(meta)}</span>'
        '</div>'
        f'{lanes_html}</div>'
    )


# ---------------------------------------------------------------------
# Quick Viz · patient timeline + sparkline tile
# ---------------------------------------------------------------------

def render_timeline(
    events: Sequence[tuple[float, str, str]],
    *,
    total_days: float = 6.2,
) -> str:
    """Events: [(x_pos_0_to_1000, label, color)]"""
    parts: list[str] = []
    parts.append('<line x1="0" y1="45" x2="1000" y2="45" stroke="var(--hair-2)"/>')
    for x, label, color in events:
        parts.append(
            f'<line x1="{x}" y1="45" x2="{x}" y2="20" stroke="{color}" stroke-width="1.5"/>'
            f'<circle cx="{x}" cy="20" r="3.5" fill="{color}"/>'
            f'<text x="{x}" y="14" font-size="9.5" fill="var(--ink-2)" text-anchor="middle">{_esc(label)}</text>'
        )
    for d in range(int(total_days) + 1):
        xd = d * (1000.0 / total_days)
        parts.append(
            f'<text x="{xd:.1f}" y="58" font-size="9" fill="var(--ink-4)" '
            f'font-family="var(--font-mono)">{d}d</text>'
        )
    return (
        '<svg width="100%" height="60" viewBox="0 0 1000 60">'
        + "".join(parts) + '</svg>'
    )


def render_sparkline_tile(
    *,
    label: str,
    value: str,
    unit: str,
    data: Sequence[float],
) -> str:
    if not data:
        return ""
    vmax = max(data) or 1.0
    pts = " ".join(
        f"{(i / (len(data) - 1)) * 120 if len(data) > 1 else 0:.1f},"
        f"{30 - (v / vmax) * 26:.1f}"
        for i, v in enumerate(data)
    )
    unit_html = (
        f'<span style="color:var(--ink-4);font-weight:400;margin-left:4px">{_esc(unit)}</span>'
        if unit else ""
    )
    return (
        '<div class="eu-card" style="padding:12px">'
        f'<div style="font-size:10px;color:var(--ink-4);letter-spacing:0.06em;'
        f'text-transform:uppercase;font-weight:500">{_esc(label)}</div>'
        f'<div class="mono" style="font-size:12.5px;margin-top:2px;color:var(--ink);'
        f'font-family:var(--font-mono)">{_esc(value)}{unit_html}</div>'
        '<svg width="100%" height="34" viewBox="0 0 120 34" style="margin-top:4px">'
        f'<polyline fill="none" stroke="var(--ink)" stroke-width="1.4" points="{pts}"/>'
        '</svg></div>'
    )


# ---------------------------------------------------------------------
# Quick Viz · missingness horizontal bars
# ---------------------------------------------------------------------

def render_missingness_bars(
    rows: Sequence[tuple[str, float, str]],
) -> str:
    """Each row: (concept_name, percent_missing 0..100, denominator_label)."""
    cells: list[str] = []
    for n, p, denom in rows:
        if p > 80:
            color = "var(--bad)"
        elif p > 40:
            color = "var(--warn)"
        else:
            color = "var(--ink)"
        cells.append(
            f'<span class="mono" style="font-size:11.5px;color:var(--ink-2)">{_esc(n)}</span>'
            f'<div style="height:14px;background:var(--surface-2);border-radius:3px;overflow:hidden">'
            f'<div style="height:100%;width:{p:.1f}%;background:{color};opacity:0.85"></div>'
            '</div>'
            f'<span class="mono" style="font-size:11px;color:var(--ink-3);text-align:right">{p:.1f}%</span>'
            f'<span class="eu-chip mono" style="font-size:9.5px;padding:0 5px">{_esc(denom)}</span>'
        )
    return (
        '<div style="display:grid;grid-template-columns:180px 1fr 64px 36px;'
        'gap:10px;align-items:center">' + "".join(cells) + '</div>'
    )


# ---------------------------------------------------------------------
# Research Agent · output tile previews
# ---------------------------------------------------------------------

def render_tile_table() -> str:
    rows = []
    for r in range(5):
        head_color = "var(--ink-3)" if r == 0 else "var(--hair-3)"
        rows.append(
            f'<rect x="10" y="{8 + r * 13}" width="40" height="6" fill="{head_color}" rx="1"/>'
            f'<rect x="56" y="{8 + r * 13}" width="22" height="6" fill="var(--hair-2)" rx="1"/>'
            f'<rect x="84" y="{8 + r * 13}" width="22" height="6" fill="var(--hair-2)" rx="1"/>'
        )
    return f'<svg width="120" height="78" viewBox="0 0 120 78">{"".join(rows)}</svg>'


def render_tile_roc() -> str:
    return (
        '<svg width="120" height="78" viewBox="0 0 120 78">'
        '<line x1="14" y1="64" x2="106" y2="64" stroke="var(--hair-3)"/>'
        '<line x1="14" y1="64" x2="14" y2="10" stroke="var(--hair-3)"/>'
        '<line x1="14" y1="64" x2="106" y2="10" stroke="var(--hair-2)" stroke-dasharray="2 3"/>'
        '<path d="M14 64 Q 30 30 60 22 Q 90 16 106 12" stroke="var(--ink)" stroke-width="1.5" fill="none"/>'
        '</svg>'
    )


def render_tile_calibration() -> str:
    points = [(20, 56), (35, 44), (55, 32), (70, 22), (92, 16)]
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="2.5" fill="var(--ink)"/>' for x, y in points)
    return (
        '<svg width="120" height="78" viewBox="0 0 120 78">'
        '<line x1="14" y1="64" x2="106" y2="64" stroke="var(--hair-3)"/>'
        '<line x1="14" y1="64" x2="14" y2="10" stroke="var(--hair-3)"/>'
        '<line x1="14" y1="64" x2="106" y2="10" stroke="var(--hair-2)" stroke-dasharray="2 3"/>'
        f'{dots}</svg>'
    )


def render_tile_feature_effects() -> str:
    widths = [62, 48, 35, 24, 18]
    bars = "".join(
        f'<rect x="34" y="{10 + i * 12}" width="{w}" height="8" fill="var(--ink)" '
        f'opacity="{1 - i * 0.15:.2f}" rx="1"/>'
        for i, w in enumerate(widths)
    )
    return f'<svg width="120" height="78" viewBox="0 0 120 78">{bars}</svg>'


def render_tile_missing() -> str:
    cells = []
    for r in range(9):
        for c in range(12):
            seed = _stable_hash(f"miss-{r}-{c}", 100) / 100.0
            if seed > 0.7:
                fill, op = "var(--bad)", 0.7
            else:
                fill, op = "var(--hair-3)", 1.0
            cells.append(
                f'<rect x="{8 + c * 8.5:.1f}" y="{8 + r * 7.2:.1f}" '
                f'width="6" height="5" fill="{fill}" opacity="{op}" rx="0.5"/>'
            )
    return f'<svg width="120" height="78" viewBox="0 0 120 78">{"".join(cells)}</svg>'


def render_output_tile(
    *,
    kind: str,
    title: str,
    sub: str,
    preview_html: str,
    badge_html: str = "",
) -> str:
    badge = (
        f'<span style="position:absolute;top:8px;right:8px">{badge_html}</span>'
        if badge_html else ""
    )
    return (
        '<div class="eu-card" style="padding:0;overflow:hidden;display:flex;flex-direction:column">'
        '<div style="height:110px;background:var(--surface-2);border-bottom:1px solid var(--hair);'
        f'display:flex;align-items:center;justify-content:center;position:relative">{preview_html}{badge}</div>'
        '<div style="padding:8px 12px">'
        f'<div class="mono" style="font-size:10.5px;color:var(--ink-4)">{_esc(kind)}</div>'
        f'<div style="font-size:12.5px;font-weight:500;margin-top:1px">{_esc(title)}</div>'
        f'<div style="font-size:11px;color:var(--ink-4)">{_esc(sub)}</div>'
        '</div></div>'
    )


# ---------------------------------------------------------------------
# Quick Viz · module picker (Data Tables subtab)
# ---------------------------------------------------------------------

def render_module_picker(
    modules: Sequence[tuple[str, int, bool]],
) -> str:
    """Each module: (name, feature_count, is_active)."""
    items = []
    for name, count, active in modules:
        cls = "eu-nav-item active" if active else "eu-nav-item"
        items.append(
            f'<div class="{cls}" style="height:28px;padding:4px 10px">'
            '<span class="ico" style="width:10px;display:inline-flex">'
            '<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="3"/></svg>'
            '</span>'
            f'<span class="label" style="font-size:12px">{_esc(name)}</span>'
            f'<span class="count mono">{count}</span>'
            '</div>'
        )
    return (
        '<div class="eu-card" style="padding:12px;display:flex;flex-direction:column;gap:1px">'
        f'<div class="eu-section-label" style="padding:0;margin:0 0 6px">'
        f'<span>Modules · {len(modules)}</span></div>'
        + "".join(items)
        + '</div>'
    )


# ---------------------------------------------------------------------
# Data preview table (Quick Viz · Data Tables)
# ---------------------------------------------------------------------

def render_data_preview_table(
    *,
    title: str,
    meta: str,
    columns: Sequence[str],
    rows: Sequence[Sequence],
) -> str:
    head = "".join(
        f'<th style="text-align:left;padding:6px 12px;font-weight:500">{_esc(c)}</th>'
        for c in columns
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td style="padding:5px 12px;color:{"var(--ink-3)" if i == 0 else "var(--ink)"}">{_esc(c)}</td>'
            for i, c in enumerate(row)
        )
        body_rows.append(f'<tr style="border-top:1px solid var(--hair)">{cells}</tr>')
    return (
        '<div class="eu-card" style="padding:0;overflow:hidden">'
        '<div style="padding:10px 14px;border-bottom:1px solid var(--hair);'
        'display:flex;justify-content:space-between;align-items:center">'
        f'<div style="font-size:12px;font-weight:500">{_esc(title)}</div>'
        f'<span class="mono" style="font-size:11px;color:var(--ink-4)">{_esc(meta)}</span>'
        '</div>'
        '<table class="mono" style="width:100%;border-collapse:collapse;font-size:11.5px;font-family:var(--font-mono)">'
        '<thead><tr style="background:var(--surface-2);color:var(--ink-4);font-size:10px;'
        'letter-spacing:0.06em;text-transform:uppercase">'
        f'{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
        '</div>'
    )


def synth_coverage_matrix(
    concepts: Sequence[str],
    n_patients: int = 30,
) -> list[tuple[str, list[float]]]:
    """Generate a stable demo coverage matrix when no real data is loaded.

    Deterministic per concept label so re-renders are stable.
    """
    out: list[tuple[str, list[float]]] = []
    for concept in concepts:
        seed = int(hashlib.sha1(concept.encode("utf-8")).digest()[0])
        values: list[float] = []
        for j in range(n_patients):
            mix = ((seed * 7 + j * 11) % 100) / 100.0
            if mix < 0.08:
                values.append(0.0)  # absent
            elif mix < 0.18:
                values.append(0.35)  # partial
            elif mix < 0.5:
                values.append(0.6)  # sparse
            else:
                values.append(0.95)  # present
        out.append((concept, values))
    return out
