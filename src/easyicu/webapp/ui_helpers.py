"""Small semantic HTML render helpers for Streamlit pages."""

from __future__ import annotations

import html
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

import streamlit as st


# =====================================================================
# Shell-A primitives (EasyICU redesign)
# ---------------------------------------------------------------------
# Lightweight HTML factories used by the sidebar, top bar, and per-page
# headers. All output is class-only — the actual styling lives in
# ``shell_styles.py`` so token edits flow through immediately.
# =====================================================================


_ICON_SVG: dict[str, str] = {
    "book":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H20v15H5.5A1.5 1.5 0 0 0 4 19.5v-15Z"/><path d="M4 19.5A1.5 1.5 0 0 0 5.5 21H20"/></svg>',
    "bars":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-8"/><path d="M22 20v-5"/></svg>',
    "layers":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/><path d="m3 18 9 5 9-5"/></svg>',
    "grid":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    "sparkles": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v4"/><path d="M12 17v4"/><path d="M3 12h4"/><path d="M17 12h4"/><path d="m5.6 5.6 2.8 2.8"/><path d="m15.6 15.6 2.8 2.8"/><path d="m18.4 5.6-2.8 2.8"/><path d="m8.4 15.6-2.8 2.8"/></svg>',
    "search":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    "chevron":  '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>',
    "check":    '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12 9 17 20 6"/></svg>',
    "play":     '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M7 5v14l12-7L7 5Z"/></svg>',
    "history":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 5v5h5"/><path d="M12 8v5l3 2"/></svg>',
    "agent":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v3"/><circle cx="12" cy="12" r="6"/><path d="m5 19 2-2"/><path d="m17 17 2 2"/></svg>',
    "clock":    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    "user":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>',
    "settings": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1.2l2-1.6-2-3.5-2.4.8a7 7 0 0 0-2-1.1L14 3h-4l-.5 2.4a7 7 0 0 0-2 1.1l-2.4-.8-2 3.5 2 1.6A7 7 0 0 0 5 12c0 .4 0 .8.1 1.2l-2 1.6 2 3.5 2.4-.8a7 7 0 0 0 2 1.1L10 21h4l.5-2.4a7 7 0 0 0 2-1.1l2.4.8 2-3.5-2-1.6c.1-.4.1-.8.1-1.2Z"/></svg>',
}


def icon(name: str) -> str:
    """Return the inline SVG string for a registered shell-A icon, or empty."""
    return _ICON_SVG.get(name, "")


PillTone = Literal["neutral", "demo", "real", "ok", "warn", "bad", "info"]


def render_pill_html(label: str, *, tone: PillTone = "neutral", dot: bool = True) -> str:
    """Return the HTML string for a small status pill."""
    tone_class = "" if tone == "neutral" else f" {tone}"
    dot_html = '<span class="dot"></span>' if dot else ""
    return f'<span class="eu-pill{tone_class}">{dot_html}{html.escape(label)}</span>'


def render_pill(label: str, *, tone: PillTone = "neutral", dot: bool = True, container=None) -> None:
    """Render a pill directly into Streamlit."""
    _render_html(render_pill_html(label, tone=tone, dot=dot), container=container)


@dataclass(frozen=True)
class ShellStat:
    label: str
    value: str
    sub: str = ""
    tone: Literal["", "ok", "warn", "bad", "info"] = ""


def render_shell_stat_html(stat: ShellStat) -> str:
    val_tone = f" {stat.tone}" if stat.tone else ""
    return (
        f'<div class="eu-card eu-stat">'
        f'<div class="label">{html.escape(stat.label)}</div>'
        f'<div class="val{val_tone}">{html.escape(stat.value)}</div>'
        f'<div class="delta">{html.escape(stat.sub)}</div>'
        f'</div>'
    )


def render_shell_stats(stats: Sequence[ShellStat], *, columns: int = 4) -> None:
    """Render a row of shell-A stat cards using Streamlit columns."""
    if not stats:
        return
    cols = st.columns(min(columns, len(stats)))
    for col, stat in zip(cols, stats):
        with col:
            _render_html(render_shell_stat_html(stat))


def render_chip_html(text: str, *, removable: bool = False) -> str:
    x = '<span class="x">&times;</span>' if removable else ""
    return f'<span class="eu-chip">{html.escape(text)}{x}</span>'


def render_chip_row(chips: Iterable[str], *, removable: bool = False, container=None) -> None:
    body = " ".join(render_chip_html(c, removable=removable) for c in chips)
    _render_html(f'<div style="display:flex;flex-wrap:wrap;gap:6px">{body}</div>', container=container)


def render_section_label(text: str, *, hint: str = "", container=None) -> None:
    hint_html = f'<span class="num">{html.escape(hint)}</span>' if hint else ""
    _render_html(
        f'<div class="eu-section-label"><span>{html.escape(text)}</span>{hint_html}</div>',
        container=container,
    )


def render_brand_html(name: str = "EasyICU", sub: str = "ICU 数据分析平台", initials: str = "E") -> str:
    """Sidebar brand block — logo tile + name + bilingual subtitle."""
    return (
        '<div class="eu-brand">'
        f'<div class="logo">{html.escape(initials)}</div>'
        '<div class="text">'
        f'<span class="name">{html.escape(name)}</span>'
        f'<span class="sub eu-cn">{html.escape(sub)}</span>'
        '</div>'
        '</div>'
    )


@dataclass(frozen=True)
class ShellNavItem:
    key: str
    label: str
    icon: str = "bars"
    count: str = ""


def render_nav_item_html(item: ShellNavItem, *, active: bool) -> str:
    cls = "eu-nav-item active" if active else "eu-nav-item"
    ico = icon(item.icon) or ""
    count_html = f'<span class="count">{html.escape(item.count)}</span>' if item.count else ""
    return (
        f'<div class="{cls}">'
        f'<span class="ico">{ico}</span>'
        f'<span class="label">{html.escape(item.label)}</span>'
        f'{count_html}'
        f'</div>'
    )


@dataclass(frozen=True)
class PipelineStep:
    key: str
    title: str
    meta: str
    status: Literal["done", "active", "todo"]


def render_pipeline_step_html(step: PipelineStep) -> str:
    status = step.status
    dot_inner = icon("check") if status == "done" else ""
    return (
        f'<div class="eu-pipe-step {status}">'
        f'<div class="dot">{dot_inner}</div>'
        '<div class="body">'
        f'<div class="title">{html.escape(step.title)}</div>'
        f'<div class="meta">{html.escape(step.meta)}</div>'
        '</div>'
        '</div>'
    )


def render_pipeline_block(steps: Sequence[PipelineStep], *, container=None) -> None:
    done_n = sum(1 for s in steps if s.status == "done")
    parts = [
        f'<div class="eu-section-label"><span>Data extraction</span>'
        f'<span class="num">{done_n} / {len(steps)}</span></div>'
    ]
    parts.extend(render_pipeline_step_html(s) for s in steps)
    _render_html("".join(parts), container=container)


def render_topbar(
    breadcrumb: Sequence[str],
    *,
    pills: Sequence[tuple[str, PillTone]] = (),
    container=None,
) -> None:
    """Render the shell-A top bar (breadcrumb + status pills).

    Action buttons (history / ask agent / run) are rendered separately
    by the caller via real ``st.button`` widgets so callbacks survive.
    """
    crumbs: list[str] = []
    for i, crumb in enumerate(breadcrumb):
        is_last = i == len(breadcrumb) - 1
        cls = "crumb current" if is_last else "crumb"
        crumbs.append(f'<span class="{cls}">{html.escape(crumb)}</span>')
        if not is_last:
            crumbs.append('<span class="sep">/</span>')
    pills_html = " ".join(render_pill_html(label, tone=tone) for label, tone in pills)
    body = (
        '<div class="eu-topbar">'
        f'<div class="bc">{"".join(crumbs)}</div>'
        f'<div class="right">{pills_html}</div>'
        '</div>'
    )
    _render_html(body, container=container)

Tone = Literal["success", "info", "warning", "danger", "neutral", "primary", "purple"]


@dataclass(frozen=True)
class StatCard:
    label: str
    value: str
    tone: Tone = "neutral"


@dataclass(frozen=True)
class FeatureCard:
    icon: str
    title: str
    description: str


@dataclass(frozen=True)
class GuidePanel:
    title: str
    items: tuple[str, ...]
    tone: Tone = "info"


@dataclass(frozen=True)
class MiniCard:
    title: str
    description: str
    tone: Tone = "info"


def _target(container):
    return container if container is not None else st


def _render_html(body: str, *, container=None) -> None:
    """Render trusted in-app HTML without passing it through Markdown parsing."""
    target = _target(container)
    html_renderer = getattr(target, "html", None)
    if callable(html_renderer):
        html_renderer(body)
    else:
        target.markdown(body, unsafe_allow_html=True)


def _tone_class(prefix: str, tone: Tone) -> str:
    if tone not in get_args(Tone):
        raise ValueError(f"Unknown UI tone: {tone}")
    return f"{prefix}--{tone}"


def render_status_banner(
    title: str,
    subtitle: str,
    *,
    tone: Tone = "success",
    icon: str = "✓",
    container=None,
) -> None:
    """Render a reusable status banner without inline style attributes."""
    _render_html(
        f"""
        <div class="app-status-banner {_tone_class("app-status-banner", tone)}">
            <div class="app-status-banner__icon" aria-hidden="true">{html.escape(icon)}</div>
            <div>
                <div class="app-status-banner__title">{html.escape(title)}</div>
                <div class="app-status-banner__subtitle">{html.escape(subtitle)}</div>
            </div>
        </div>
        """,
        container=container,
    )


def render_stat_grid(cards: Iterable[StatCard], *, columns: int = 4, compact: bool = False, container=None) -> None:
    """Render a grid of compact statistic cards."""
    modifier = " app-stat-grid--compact" if compact else ""
    card_html = "\n".join(
        f"""
        <div class="app-stat-card {_tone_class("app-stat-card", card.tone)}">
            <div class="app-stat-card__label">{html.escape(card.label)}</div>
            <div class="app-stat-card__value">{html.escape(card.value)}</div>
        </div>
        """
        for card in cards
    )
    _render_html(
        f'<div class="app-stat-grid app-stat-grid--{columns}{modifier}">{card_html}</div>',
        container=container,
    )


def render_feature_grid(cards: Iterable[FeatureCard], *, columns: int = 3, muted: bool = False, container=None) -> None:
    """Render a responsive semantic feature-card grid."""
    modifier = " app-feature-grid--muted" if muted else ""
    card_html = "\n".join(
        f"""
        <div class="app-feature-card">
            <div class="app-feature-card__icon" aria-hidden="true">{html.escape(card.icon)}</div>
            <div class="app-feature-card__title">{html.escape(card.title)}</div>
            <div class="app-feature-card__description">{html.escape(card.description)}</div>
        </div>
        """
        for card in cards
    )
    _render_html(
        f'<div class="app-feature-grid app-feature-grid--{columns}{modifier}">{card_html}</div>',
        container=container,
    )


def render_inline_heading(title: str, subtitle: str = "", *, container=None) -> None:
    """Render a compact section heading used inside Streamlit tabs."""
    subtitle_html = f'<span class="app-inline-heading__subtitle">{html.escape(subtitle)}</span>' if subtitle else ""
    _render_html(
        f"""
        <div class="app-inline-heading">
            <span class="app-inline-heading__title">{html.escape(title)}</span>
            {subtitle_html}
        </div>
        """,
        container=container,
    )


def render_steps(steps: Iterable[tuple[str, str]], *, current_index: int, container=None) -> None:
    """Render a compact horizontal step indicator."""
    step_html = []
    for idx, (title, description) in enumerate(steps):
        if idx < current_index:
            dot_class = "step-dot done"
            dot_text = "✓"
        elif idx == current_index:
            dot_class = "step-dot active"
            dot_text = str(idx + 1)
        else:
            dot_class = "step-dot"
            dot_text = str(idx + 1)
        step_html.append(
            f"""
            <div class="step-indicator">
                <div class="app-step-row">
                    <div class="{dot_class}">{html.escape(dot_text)}</div>
                    <div class="step-text">
                        <div>{html.escape(title)}</div>
                        <small>{html.escape(description)}</small>
                    </div>
                </div>
            </div>
            """
        )
    _render_html(f'<div class="app-step-grid">{"".join(step_html)}</div>', container=container)


def render_kicker(text: str, *, container=None) -> None:
    """Render a small uppercase section kicker."""
    _render_html(f'<div class="app-kicker">{html.escape(text)}</div>', container=container)


def render_anchor(anchor_id: str, *, spacer: bool = False, container=None) -> None:
    """Render a stable in-page anchor."""
    class_name = "app-anchor app-anchor--spaced" if spacer else "app-anchor"
    _render_html(
        f'<div id="{html.escape(anchor_id)}" class="{class_name}"></div>',
        container=container,
    )


def render_guide_card(
    title: str,
    *,
    panels: Iterable[GuidePanel] = (),
    bullets: Iterable[str] = (),
    ordered_steps: Iterable[str] = (),
    mini_cards: Iterable[MiniCard] = (),
    tip: str = "",
    tone: Tone = "neutral",
    container=None,
) -> None:
    """Render a reusable workflow guide card."""
    panel_html = "".join(
        f"""
        <div class="app-guide-panel {_tone_class("app-guide-panel", panel.tone)}">
            <div class="app-guide-panel__title">{html.escape(panel.title)}</div>
            <ul class="app-guide-list">
                {"".join(f"<li>{html.escape(item)}</li>" for item in panel.items)}
            </ul>
        </div>
        """
        for panel in panels
    )
    bullet_html = ""
    bullet_items = list(bullets)
    if bullet_items:
        bullet_html = '<ul class="app-guide-list">' + "".join(
            f"<li>{html.escape(item)}</li>" for item in bullet_items
        ) + "</ul>"
    step_items = list(ordered_steps)
    ordered_html = ""
    if step_items:
        ordered_html = '<ol class="app-guide-list app-guide-list--ordered">' + "".join(
            f"<li>{html.escape(item)}</li>" for item in step_items
        ) + "</ol>"
    mini_html = "".join(
        f"""
        <div class="app-mini-card {_tone_class("app-mini-card", card.tone)}">
            <div class="app-mini-card__title">{html.escape(card.title)}</div>
            <div class="app-mini-card__description">{html.escape(card.description)}</div>
        </div>
        """
        for card in mini_cards
    )
    tip_html = f'<div class="app-guide-tip">{html.escape(tip)}</div>' if tip else ""
    mini_grid = f'<div class="app-mini-grid">{mini_html}</div>' if mini_html else ""
    panel_grid = f'<div class="app-guide-panel-grid">{panel_html}</div>' if panel_html else ""

    _render_html(
        textwrap.dedent(
            f"""
        <div class="app-guide-card {_tone_class("app-guide-card", tone)}">
            <div class="app-guide-card__title">{html.escape(title)}</div>
            {panel_grid}
            {mini_grid}
            {bullet_html}
            {ordered_html}
            {tip_html}
        </div>
        """
        ).strip(),
        container=container,
    )


def render_option_card(title: str, items: Iterable[str], *, tone: Tone = "primary", container=None) -> None:
    """Render a compact next-step option card."""
    _target(container).markdown(
        f"""
        <div class="app-option-card {_tone_class("app-option-card", tone)}">
            <div class="app-option-card__title">{html.escape(title)}</div>
            <ul class="app-option-card__list">
                {"".join(f"<li>{html.escape(item)}</li>" for item in items)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note(text: str, *, tone: Tone = "info", container=None) -> None:
    """Render a semantic note block."""
    _target(container).markdown(
        f'<div class="app-note {_tone_class("app-note", tone)}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_file_list(paths: Iterable[str], *, more_text: str = "", container=None) -> None:
    """Render a small exported-file list without inline styles."""
    file_html = "".join(f"<li>{html.escape(path)}</li>" for path in paths)
    more_html = f'<div class="app-file-list__more">{html.escape(more_text)}</div>' if more_text else ""
    _target(container).markdown(
        f'<ul class="app-file-list">{file_html}</ul>{more_html}',
        unsafe_allow_html=True,
    )


def render_footer_links(line_one: str, line_two: str, *, container=None) -> None:
    """Render the app footer with stable classes."""
    _target(container).markdown(
        f"""
        <div class="app-footer">
            <p>{html.escape(line_one)}</p>
            <p>{html.escape(line_two)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
