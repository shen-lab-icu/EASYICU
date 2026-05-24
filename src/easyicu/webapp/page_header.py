"""Shared compact page header used by top-level EasyICU tabs."""

from __future__ import annotations

import html

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    *,
    icon: str = "",
    kicker: str = "",
) -> None:
    """Render a consistent, compact page header inside a main tab."""
    icon_html = f'<span class="app-page-icon">{html.escape(icon)}</span>' if icon else ""
    kicker_html = f'<div class="app-page-kicker">{html.escape(kicker)}</div>' if kicker else ""
    header_html = (
        '<div class="app-page-header">'
        f'{kicker_html}'
        '<div class="app-page-title-row">'
        f'{icon_html}'
        f'<div class="app-page-title">{html.escape(title)}</div>'
        '</div>'
        f'<div class="app-page-subtitle">{html.escape(subtitle)}</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
