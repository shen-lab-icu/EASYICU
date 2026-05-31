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


def _css_mtime_ns(path: Path) -> int:
    """Return an mtime cache key for hot-reloaded local CSS files."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


@lru_cache(maxsize=8)
def _load_css_file(path_text: str, mtime_ns: int) -> str:
    """Read a CSS file, invalidating the cache when the file changes.

    Streamlit re-runs the app on every interaction, so the mtime-aware
    cache avoids repeated reads while still making local design edits
    visible without restarting the server.
    """
    try:
        return Path(path_text).read_text(encoding="utf-8")
    except OSError:
        return ""


def _load_tokens_css() -> str:
    """Read tokens.css with mtime-aware caching."""
    return _load_css_file(str(_TOKENS_PATH), _css_mtime_ns(_TOKENS_PATH))


_OVERRIDES_PATH = Path(__file__).with_name("shell_overrides.css")


def _load_shell_overrides_css() -> str:
    """Read shell_overrides.css with mtime-aware caching."""
    return _load_css_file(str(_OVERRIDES_PATH), _css_mtime_ns(_OVERRIDES_PATH))




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
