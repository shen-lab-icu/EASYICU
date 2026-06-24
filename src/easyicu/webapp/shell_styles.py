"""Shell A · EasyICU design-system layer.

This module injects the EasyICU shell-A design tokens (``tokens.css``)
and the minimal Streamlit reset still kept for the deprecated legacy app.
The historical route split CSS is no longer part of the default runtime path;
set ``EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1`` to re-enable it temporarily:

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

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

LEGACY_STREAMLIT_CSS_ENV = "EASYICU_ENABLE_LEGACY_STREAMLIT_CSS"
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


_SHELL_NAVIGATION_OVERRIDES_PATH = Path(__file__).with_name("shell_navigation_overrides.css")


def _load_shell_navigation_overrides_css() -> str:
    """Read shared shell navigation/chrome CSS with mtime-aware caching."""
    return _load_css_file(
        str(_SHELL_NAVIGATION_OVERRIDES_PATH),
        _css_mtime_ns(_SHELL_NAVIGATION_OVERRIDES_PATH),
    )


_ENTRY_OVERRIDES_PATH = Path(__file__).with_name("entry_overrides.css")


def _load_entry_overrides_css() -> str:
    """Read Entry/Home route CSS with mtime-aware caching."""
    return _load_css_file(str(_ENTRY_OVERRIDES_PATH), _css_mtime_ns(_ENTRY_OVERRIDES_PATH))


_TUTORIAL_OVERRIDES_PATH = Path(__file__).with_name("tutorial_overrides.css")


def _load_tutorial_overrides_css() -> str:
    """Read Tutorial/Get Started route CSS with mtime-aware caching."""
    return _load_css_file(
        str(_TUTORIAL_OVERRIDES_PATH),
        _css_mtime_ns(_TUTORIAL_OVERRIDES_PATH),
    )


_DICTIONARY_OVERRIDES_PATH = Path(__file__).with_name("dictionary_overrides.css")


def _load_dictionary_overrides_css() -> str:
    """Read Data Dictionary route CSS with mtime-aware caching."""
    return _load_css_file(
        str(_DICTIONARY_OVERRIDES_PATH),
        _css_mtime_ns(_DICTIONARY_OVERRIDES_PATH),
    )


_STATES_OVERRIDES_PATH = Path(__file__).with_name("states_overrides.css")


def _load_states_overrides_css() -> str:
    """Read Workspace States route CSS with mtime-aware caching."""
    return _load_css_file(
        str(_STATES_OVERRIDES_PATH),
        _css_mtime_ns(_STATES_OVERRIDES_PATH),
    )


_GUIDED_OVERRIDES_PATH = Path(__file__).with_name("guided_overrides.css")


def _load_guided_overrides_css() -> str:
    """Read assistant/guided-route CSS with mtime-aware caching."""
    return _load_css_file(str(_GUIDED_OVERRIDES_PATH), _css_mtime_ns(_GUIDED_OVERRIDES_PATH))


_AGENT_OVERRIDES_PATH = Path(__file__).with_name("agent_overrides.css")


def _load_agent_overrides_css() -> str:
    """Read Agent Projects route CSS with mtime-aware caching."""
    return _load_css_file(str(_AGENT_OVERRIDES_PATH), _css_mtime_ns(_AGENT_OVERRIDES_PATH))


_PATIENT_OVERRIDES_PATH = Path(__file__).with_name("patient_overrides.css")


def _load_patient_overrides_css() -> str:
    """Read Patient Review route CSS with mtime-aware caching."""
    return _load_css_file(str(_PATIENT_OVERRIDES_PATH), _css_mtime_ns(_PATIENT_OVERRIDES_PATH))


_VISUALIZATION_SHELL_OVERRIDES_PATH = Path(__file__).with_name("visualization_shell_overrides.css")


def _load_visualization_shell_overrides_css() -> str:
    """Read shared Patient/Cohort/Cross-DB shell CSS with mtime-aware caching."""
    return _load_css_file(
        str(_VISUALIZATION_SHELL_OVERRIDES_PATH),
        _css_mtime_ns(_VISUALIZATION_SHELL_OVERRIDES_PATH),
    )


_COHORT_OVERRIDES_PATH = Path(__file__).with_name("cohort_overrides.css")


def _load_cohort_overrides_css() -> str:
    """Read Cohort Statistics route CSS with mtime-aware caching."""
    return _load_css_file(str(_COHORT_OVERRIDES_PATH), _css_mtime_ns(_COHORT_OVERRIDES_PATH))


_CROSSDB_OVERRIDES_PATH = Path(__file__).with_name("crossdb_overrides.css")


def _load_crossdb_overrides_css() -> str:
    """Read Cross-DB route CSS with mtime-aware caching."""
    return _load_css_file(str(_CROSSDB_OVERRIDES_PATH), _css_mtime_ns(_CROSSDB_OVERRIDES_PATH))


_SETTINGS_OVERRIDES_PATH = Path(__file__).with_name("settings_overrides.css")


def _load_settings_overrides_css() -> str:
    """Read Settings route CSS with mtime-aware caching."""
    return _load_css_file(str(_SETTINGS_OVERRIDES_PATH), _css_mtime_ns(_SETTINGS_OVERRIDES_PATH))


_EXTRACT_OVERRIDES_PATH = Path(__file__).with_name("extract_overrides.css")


def _load_extract_overrides_css() -> str:
    """Read Data Extraction route CSS with mtime-aware caching."""
    return _load_css_file(str(_EXTRACT_OVERRIDES_PATH), _css_mtime_ns(_EXTRACT_OVERRIDES_PATH))


_ALIGNMENT_PATH = Path(__file__).with_name("alignment.css")


def _load_alignment_css() -> str:
    """Read the minimal alignment compatibility marker."""
    return _load_css_file(str(_ALIGNMENT_PATH), _css_mtime_ns(_ALIGNMENT_PATH))


def _legacy_streamlit_css_enabled() -> bool:
    """Return True only for the explicit temporary legacy CSS opt-in."""
    return os.environ.get(LEGACY_STREAMLIT_CSS_ENV) == "1"


_GUIDED_ROUTE_TOKENS = {"assistant", "copilot", "guided"}
_ENTRY_ROUTE_TOKENS = {"entry", "home", "landing"}
_TUTORIAL_ROUTE_TOKENS = {"tutorial", "get-started", "getstarted", "help", "guide"}
_DICTIONARY_ROUTE_TOKENS = {"dictionary", "dict", "data-dictionary"}
_STATES_ROUTE_TOKENS = {"states", "workspace-states", "workspace-status"}
_AGENT_ROUTE_TOKENS = {"agent", "agent-projects", "research-agent"}
_SETTINGS_ROUTE_TOKENS = {"settings", "preferences"}
_EXTRACT_ROUTE_TOKENS = {
    "data-extraction",
    "extract",
    "extraction",
    "real-extraction",
    "source",
}
_PATIENT_ROUTE_TOKENS = {"patient", "quick-viz", "review", "viz"}
_COHORT_ROUTE_TOKENS = {
    "audit",
    "cohort",
    "cohort-statistics",
    "coverage",
    "coverage-audit",
    "group-contrast",
    "sofa",
    "sofa-reclassification",
}
_CROSSDB_ROUTE_TOKENS = {"cross-db", "crossdb", "cross-db-benchmark", "multidb"}
_VISUALIZATION_SHELL_ROUTE_TOKENS = _PATIENT_ROUTE_TOKENS | _COHORT_ROUTE_TOKENS | _CROSSDB_ROUTE_TOKENS
_PRIMARY_ROUTE_QUERY_KEYS = ("page", "route")
_LEGACY_ROUTE_QUERY_KEYS = ("mode", "data_mode", "entry_mode")


def _query_param_value(st: Any, key: str) -> str | None:
    params = getattr(st, "query_params", None)
    if not params:
        return None
    try:
        value = params.get(key)
    except Exception:
        try:
            value = params[key]
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value)


def _route_token(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def _explicit_query_route_match(st: Any, route_tokens: set[str]) -> bool | None:
    """Return explicit page/route match, letting URL routes beat stale state."""
    for key in _PRIMARY_ROUTE_QUERY_KEYS:
        token = _route_token(_query_param_value(st, key))
        if token:
            return token in route_tokens
    return None


def _legacy_query_route_match(st: Any, route_tokens: set[str]) -> bool:
    for key in _LEGACY_ROUTE_QUERY_KEYS:
        if _route_token(_query_param_value(st, key)) in route_tokens:
            return True
    return False


def _should_load_entry_overrides(st: Any) -> bool:
    """Return True when the current rerun can render the Entry/Home route."""
    explicit_query_match = _explicit_query_route_match(st, _ENTRY_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        active_page = _route_token(state.get("_active_main_page"))
        if active_page:
            return active_page == "entry"
    except Exception:
        pass

    if _legacy_query_route_match(st, _ENTRY_ROUTE_TOKENS):
        return True

    # The app opens on Entry/Home before route state is always present.
    return True


def _should_load_tutorial_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Tutorial/Get Started."""
    explicit_query_match = _explicit_query_route_match(st, _TUTORIAL_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "tutorial":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _TUTORIAL_ROUTE_TOKENS)


def _should_load_dictionary_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Data Dictionary."""
    explicit_query_match = _explicit_query_route_match(st, _DICTIONARY_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "dictionary":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _DICTIONARY_ROUTE_TOKENS)


def _should_load_states_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Workspace States."""
    explicit_query_match = _explicit_query_route_match(st, _STATES_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "states":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _STATES_ROUTE_TOKENS)


def _should_load_guided_overrides(st: Any) -> bool:
    """Return True when the current rerun can render the guided Copilot shell."""
    explicit_query_match = _explicit_query_route_match(st, _GUIDED_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "assistant":
            return True
        if bool(state.get("_eu_guided_fullscreen")):
            return True
    except Exception:
        pass

    # Query routes are consumed later in app.py, after global styles are emitted.
    # Check the raw query string here so first-load ?page=guided does not flash
    # without the guided shell CSS.
    return _legacy_query_route_match(st, _GUIDED_ROUTE_TOKENS)


def _should_load_agent_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Agent Projects."""
    explicit_query_match = _explicit_query_route_match(st, _AGENT_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "research-agent":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _AGENT_ROUTE_TOKENS)


def _should_load_patient_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Patient Review."""
    explicit_query_match = _explicit_query_route_match(st, _PATIENT_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "quick-viz":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _PATIENT_ROUTE_TOKENS)


def _should_load_cohort_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Cohort Statistics."""
    explicit_query_match = _explicit_query_route_match(st, _COHORT_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "cohort":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _COHORT_ROUTE_TOKENS)


def _should_load_crossdb_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Cross-DB."""
    explicit_query_match = _explicit_query_route_match(st, _CROSSDB_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "cross-db":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _CROSSDB_ROUTE_TOKENS)


def _should_load_visualization_shell_overrides(st: Any) -> bool:
    """Return True for shared Patient Review, Cohort Statistics, or Cross-DB chrome."""
    explicit_query_match = _explicit_query_route_match(st, _VISUALIZATION_SHELL_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        active_page = _route_token(state.get("_active_main_page"))
        if active_page in {"quick-viz", "cohort", "cross-db"}:
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _VISUALIZATION_SHELL_ROUTE_TOKENS)


def _should_load_settings_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Settings."""
    explicit_query_match = _explicit_query_route_match(st, _SETTINGS_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "settings":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _SETTINGS_ROUTE_TOKENS)


def _should_load_extract_overrides(st: Any) -> bool:
    """Return True when the current rerun can render Data Extraction."""
    explicit_query_match = _explicit_query_route_match(st, _EXTRACT_ROUTE_TOKENS)
    if explicit_query_match is not None:
        return explicit_query_match

    state = getattr(st, "session_state", {})
    try:
        if _route_token(state.get("_active_main_page")) == "extract":
            return True
    except Exception:
        pass

    return _legacy_query_route_match(st, _EXTRACT_ROUTE_TOKENS)


_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Mono:wght@400;500;600&'
    'family=IBM+Plex+Sans:wght@400;500;600;700&'
    'family=IBM+Plex+Sans+SC:wght@400;500;600&display=swap">'
    '<style>@import url("https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Mono:wght@400;500;600&'
    'family=IBM+Plex+Sans:wght@400;500;600;700&'
    'family=IBM+Plex+Sans+SC:wght@400;500;600&display=swap");</style>'
)

_FONTS_PRELOAD_SCRIPT = """
<script>
(function() {
  function loadFonts() {
    try {
      var doc = window.parent && window.parent.document ? window.parent.document : document;
      if (!doc.fonts || !doc.fonts.load) return;
      [
        '400 13.5px "IBM Plex Sans"',
        '500 13.5px "IBM Plex Sans"',
        '600 13.5px "IBM Plex Sans"',
        '700 13.5px "IBM Plex Sans"',
        '400 11px "IBM Plex Mono"',
        '500 11px "IBM Plex Mono"',
        '600 11px "IBM Plex Mono"',
        '400 13.5px "IBM Plex Sans SC"',
        '500 13.5px "IBM Plex Sans SC"',
        '600 13.5px "IBM Plex Sans SC"'
      ].forEach(function(spec) { doc.fonts.load(spec).catch(function() {}); });
    } catch (err) {}
  }
  loadFonts();
  setTimeout(loadFonts, 250);
  setTimeout(loadFonts, 1000);
  setTimeout(loadFonts, 3000);
  setTimeout(loadFonts, 6000);
})();
</script>
"""


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
    try:
        st.components.v1.html(_FONTS_PRELOAD_SCRIPT, height=0)
    except Exception:
        pass
    tokens = _load_tokens_css()
    if tokens:
        st.markdown(f"<style>{tokens}</style>", unsafe_allow_html=True)
    overrides = _load_shell_overrides_css()
    if overrides:
        st.markdown(f"<style>{overrides}</style>", unsafe_allow_html=True)
    if not _legacy_streamlit_css_enabled():
        return
    # alignment.css is now a tiny compatibility marker. Shared shell chrome
    # follows it so migrated sidebar/topbar/mobile locks keep their former
    # cascade strength; route-specific CSS then wins over both shared layers.
    alignment = _load_alignment_css()
    if alignment:
        st.markdown(f"<style>{alignment}</style>", unsafe_allow_html=True)
    shell_navigation_overrides = _load_shell_navigation_overrides_css()
    if shell_navigation_overrides:
        st.markdown(f"<style>{shell_navigation_overrides}</style>", unsafe_allow_html=True)
    entry_overrides = _load_entry_overrides_css() if _should_load_entry_overrides(st) else ""
    if entry_overrides:
        st.markdown(f"<style>{entry_overrides}</style>", unsafe_allow_html=True)
    tutorial_overrides = (
        _load_tutorial_overrides_css()
        if _should_load_tutorial_overrides(st)
        else ""
    )
    if tutorial_overrides:
        st.markdown(f"<style>{tutorial_overrides}</style>", unsafe_allow_html=True)
    dictionary_overrides = (
        _load_dictionary_overrides_css()
        if _should_load_dictionary_overrides(st)
        else ""
    )
    if dictionary_overrides:
        st.markdown(f"<style>{dictionary_overrides}</style>", unsafe_allow_html=True)
    states_overrides = _load_states_overrides_css() if _should_load_states_overrides(st) else ""
    if states_overrides:
        st.markdown(f"<style>{states_overrides}</style>", unsafe_allow_html=True)
    settings_overrides = (
        _load_settings_overrides_css()
        if _should_load_settings_overrides(st)
        else ""
    )
    if settings_overrides:
        st.markdown(f"<style>{settings_overrides}</style>", unsafe_allow_html=True)
    extract_overrides = _load_extract_overrides_css() if _should_load_extract_overrides(st) else ""
    if extract_overrides:
        st.markdown(f"<style>{extract_overrides}</style>", unsafe_allow_html=True)
    visualization_shell_overrides = (
        _load_visualization_shell_overrides_css()
        if _should_load_visualization_shell_overrides(st)
        else ""
    )
    if visualization_shell_overrides:
        st.markdown(f"<style>{visualization_shell_overrides}</style>", unsafe_allow_html=True)
    patient_overrides = _load_patient_overrides_css() if _should_load_patient_overrides(st) else ""
    if patient_overrides:
        st.markdown(f"<style>{patient_overrides}</style>", unsafe_allow_html=True)
    cohort_overrides = _load_cohort_overrides_css() if _should_load_cohort_overrides(st) else ""
    if cohort_overrides:
        st.markdown(f"<style>{cohort_overrides}</style>", unsafe_allow_html=True)
    crossdb_overrides = _load_crossdb_overrides_css() if _should_load_crossdb_overrides(st) else ""
    if crossdb_overrides:
        st.markdown(f"<style>{crossdb_overrides}</style>", unsafe_allow_html=True)
    agent_overrides = _load_agent_overrides_css() if _should_load_agent_overrides(st) else ""
    if agent_overrides:
        st.markdown(f"<style>{agent_overrides}</style>", unsafe_allow_html=True)
    guided_overrides = _load_guided_overrides_css() if _should_load_guided_overrides(st) else ""
    if guided_overrides:
        st.markdown(f"<style>{guided_overrides}</style>", unsafe_allow_html=True)
