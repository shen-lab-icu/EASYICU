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
