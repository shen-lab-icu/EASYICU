"""Small semantic HTML render helpers for Streamlit pages."""

from __future__ import annotations

import html
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

import streamlit as st

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
    target = _target(container)
    target.markdown(
        f"""
        <div class="app-status-banner {_tone_class("app-status-banner", tone)}">
            <div class="app-status-banner__icon" aria-hidden="true">{html.escape(icon)}</div>
            <div>
                <div class="app-status-banner__title">{html.escape(title)}</div>
                <div class="app-status-banner__subtitle">{html.escape(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
    _target(container).markdown(
        f'<div class="app-stat-grid app-stat-grid--{columns}{modifier}">{card_html}</div>',
        unsafe_allow_html=True,
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
    _target(container).markdown(
        f'<div class="app-feature-grid app-feature-grid--{columns}{modifier}">{card_html}</div>',
        unsafe_allow_html=True,
    )


def render_inline_heading(title: str, subtitle: str = "", *, container=None) -> None:
    """Render a compact section heading used inside Streamlit tabs."""
    subtitle_html = f'<span class="app-inline-heading__subtitle">{html.escape(subtitle)}</span>' if subtitle else ""
    _target(container).markdown(
        f"""
        <div class="app-inline-heading">
            <span class="app-inline-heading__title">{html.escape(title)}</span>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
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
    _target(container).markdown(f'<div class="app-step-grid">{"".join(step_html)}</div>', unsafe_allow_html=True)


def render_kicker(text: str, *, container=None) -> None:
    """Render a small uppercase section kicker."""
    _target(container).markdown(f'<div class="app-kicker">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_anchor(anchor_id: str, *, spacer: bool = False, container=None) -> None:
    """Render a stable in-page anchor."""
    class_name = "app-anchor app-anchor--spaced" if spacer else "app-anchor"
    _target(container).markdown(
        f'<div id="{html.escape(anchor_id)}" class="{class_name}"></div>',
        unsafe_allow_html=True,
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

    _target(container).markdown(
        f"""
        <div class="app-guide-card {_tone_class("app-guide-card", tone)}">
            <div class="app-guide-card__title">{html.escape(title)}</div>
            {panel_grid}
            {mini_grid}
            {bullet_html}
            {ordered_html}
            {tip_html}
        </div>
        """,
        unsafe_allow_html=True,
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
