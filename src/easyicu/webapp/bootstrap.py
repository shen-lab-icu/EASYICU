"""Startup and runtime shell helpers for the EasyICU Streamlit app."""

from __future__ import annotations

import os
from typing import Any

from easyicu.webapp.compat import query_flag_enabled


def configure_page(st: Any) -> None:
    """Configure Streamlit before any other Streamlit command runs."""
    st.set_page_config(
        page_title="EasyICU Data Explorer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def configure_runtime_env() -> None:
    """Apply process-level runtime defaults for the web app."""
    os.environ["EASYICU_AUTO_CLEAR_CACHE"] = "False"


def sync_screenshot_mode(st: Any) -> bool:
    """Synchronize screenshot mode from env/query params into session state."""
    if "sidebar_expanded" not in st.session_state:
        st.session_state.sidebar_expanded = False

    env_screenshot = os.environ.get("EASYICU_SCREENSHOT_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    query_screenshot = query_flag_enabled(st, "figure") or query_flag_enabled(st, "screenshot")
    if env_screenshot:
        st.session_state.screenshot_mode = True
        st.session_state["_screenshot_mode_source"] = "env"
    elif query_screenshot:
        st.session_state.screenshot_mode = True
        st.session_state["_screenshot_mode_source"] = "query"
    elif st.session_state.get("_screenshot_mode_source") in {"query", "env"}:
        st.session_state.screenshot_mode = False
        st.session_state["_screenshot_mode_source"] = "manual"
        st.session_state.pop("_figure_target_section", None)
        st.session_state.pop("_figure_target_panel", None)
    elif "screenshot_mode" not in st.session_state:
        st.session_state.screenshot_mode = False
        st.session_state["_screenshot_mode_source"] = "manual"

    return bool(st.session_state.get("screenshot_mode", False))


def render_runtime_shell_styles(st: Any) -> None:
    """Inject the small dynamic CSS block that depends on session state."""
    screenshot_mode_enabled = bool(st.session_state.get("screenshot_mode", False))
    sidebar_width = "250px" if not st.session_state.sidebar_expanded else "100vw"
    sidebar_min_width = "250px" if not st.session_state.sidebar_expanded else "100vw"
    sidebar_display = "none" if screenshot_mode_enabled else "block"
    main_display = "block" if screenshot_mode_enabled else ("none" if st.session_state.sidebar_expanded else "block")
    floating_ai_display = "none" if screenshot_mode_enabled else "block"
    screenshot_visibility = "hidden" if screenshot_mode_enabled else "visible"
    expander_display = "none" if screenshot_mode_enabled else "block"
    compact_notice_display = "none" if screenshot_mode_enabled else "block"
    block_padding_top = "0.75rem" if screenshot_mode_enabled else "0"
    block_max_width = "1500px" if screenshot_mode_enabled else "initial"

    st.markdown(
        f"""
<style>
    [data-testid="stSidebar"] {{
        display: {sidebar_display} !important;
        min-width: {sidebar_min_width};
        max-width: {sidebar_width};
        width: {sidebar_width} !important;
    }}
    [data-testid="stSidebar"] > div {{
        width: 100% !important;
    }}
    header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stDeployButton"],
    [data-testid="manage-app-button"],
    .stDeployButton,
    .stAppDeployButton,
    button[title="Deploy"],
    a[href*="share.streamlit.io"] {{
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }}
    .block-container {{
        padding-top: {block_padding_top} !important;
        max-width: {block_max_width} !important;
    }}
    .compact-inline-notice {{
        display: {compact_notice_display} !important;
    }}
    details[data-testid="stExpander"],
    div[data-testid="stExpander"] {{
        display: {expander_display} !important;
    }}
    [data-testid="stMain"] {{
        display: {main_display} !important;
        overflow-y: visible !important;
    }}
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main {{
        overflow-y: visible !important;
        height: auto !important;
        min-height: 100vh !important;
    }}
    div.st-key-floating_ai_launcher,
    div.st-key-floating_ai_panel {{
        display: {floating_ai_display} !important;
    }}
    iframe[title*="streamlit_shadcn_ui"],
    iframe[title^="streamlit_shadcn_ui"] {{
        display: {floating_ai_display} !important;
        visibility: {screenshot_visibility} !important;
    }}
    html, body {{
        overflow-y: auto !important;
        height: auto !important;
        background: var(--bg, #FAFAF7) !important;
    }}
    [data-testid="stAppViewContainer"] {{
        overflow-y: auto !important;
        background: var(--bg, #FAFAF7) !important;
    }}
    @media (max-width: 1280px) {{
        [data-testid="stSidebar"] {{
            min-width: 250px;
            max-width: 250px;
            width: 250px !important;
        }}
    }}
    @media (max-width: 900px) {{
        [data-testid="stSidebar"] {{
            min-width: 100vw;
            max-width: 100vw;
            width: 100vw !important;
        }}
    }}
</style>
""",
        unsafe_allow_html=True,
    )
