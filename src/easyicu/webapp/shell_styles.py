"""Shell A · EasyICU design-system layer.

This module injects the EasyICU shell-A design tokens (``tokens.css``)
plus all the Streamlit-specific overrides needed to land the redesign on
top of the existing app:

* IBM Plex font stack
* Override the older :root tokens from ``styles.py`` so accent / surface
  references resolve to the new restrained-teal palette
* Hide Streamlit's native chrome (deploy header, toolbar, footer, hamburger)
* Reskin native widgets (button, selectbox, text_input, slider, radio,
  tabs) so they blend into the EasyICU shell
* Restyle the legacy ``.main-header`` / ``.sub-header`` / ``.main-nav``
  blocks so the redesign reaches the existing page chrome without
  invasive edits

The module exposes a single entry point, :func:`render_shell_styles`,
which should be called **after** :func:`easyicu.webapp.styles.render_global_styles`
so it wins the CSS cascade.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

_TOKENS_PATH = Path(__file__).with_name("tokens.css")


@lru_cache(maxsize=1)
def _load_tokens_css() -> str:
    """Read tokens.css once per process (cached across reruns).

    Streamlit re-runs the whole script on every interaction; reading
    this file from disk each time was a needless per-rerun cost.
    """
    try:
        return _TOKENS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


_OVERRIDES_PATH = Path(__file__).with_name("shell_overrides.css")


@lru_cache(maxsize=1)
def _load_shell_overrides_css() -> str:
    """Read shell_overrides.css once per process (cached across reruns).

    The Streamlit-specific re-skin lives in a sibling .css file so the
    Python module stays small. Streamlit re-runs the whole script on
    every interaction; the cache keeps this to a single read per
    process.
    """
    try:
        return _OVERRIDES_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""




_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Sans:wght@300;400;500;600&'
    'family=IBM+Plex+Sans+SC:wght@300;400;500;600&'
    'family=IBM+Plex+Mono:wght@400;500&display=swap">'
)


def render_shell_styles(st: Any) -> None:
    """Inject the shell-A token layer + Streamlit re-skin.

    Must be called after :func:`easyicu.webapp.styles.render_global_styles`
    so the cascade resolves to the new tokens.

    Kept as separate ``st.markdown`` calls (font <link> tags, the
    tokens <style>, and the overrides <style>) — combining them into a
    single markdown string made Streamlit's markdown/directive parser
    throw "Cannot set properties of undefined (directiveAttributes)"
    and drop ALL the styles. The per-rerun cost is just the cached
    token read + three small emits.

    NOTE: do **not** gate this with a session_state flag. Streamlit
    only keeps DOM elements that were re-emitted on the current rerun;
    skipping the markdown calls dropped the entire shell CSS from the
    page on the second rerun (everything went unstyled). The framework
    already dedupes identical content on the wire, so the bandwidth
    cost is small and *not* worth re-introducing the regression.
    """
    st.markdown(_FONTS_LINK, unsafe_allow_html=True)
    tokens = _load_tokens_css()
    if tokens:
        st.markdown(f"<style>{tokens}</style>", unsafe_allow_html=True)
    overrides = _load_shell_overrides_css()
    if overrides:
        st.markdown(f"<style>{overrides}</style>", unsafe_allow_html=True)
