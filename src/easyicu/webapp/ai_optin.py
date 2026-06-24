"""Central opt-in gate for AI / LLM features in the web app.

This module is the **single chokepoint** for the invariant documented in
both workspace and EASYICU CLAUDE.md:

    AI assistant features are opt-in and start disabled until the user
    enables them in the sidebar.

Any code path in `easyicu.webapp.*` that may issue an *external* LLM call
(real OpenAI / OpenRouter / Anthropic / etc. — anything that reaches the
network) MUST go through `enforce_external_llm_opt_in()` before
instantiating its client.

Offline / deterministic clients (MockLLMClient) are intentionally NOT
gated: they are safe to run without user consent because no data leaves
the machine.

If you add a new page or workflow that talks to a hosted LLM, call
`enforce_external_llm_opt_in()` at the top of the resolution path.
"""

from __future__ import annotations

from typing import Optional

from easyicu.ai_optin import (
    AIOptInError,
    check_external_llm_opt_in,
    is_offline_llm_choice,
)


def is_ai_enabled() -> bool:
    """Return the canonical sidebar opt-in flag."""
    import streamlit as st

    return bool(st.session_state.get("llm_enabled", False))


def enforce_external_llm_opt_in(
    llm_choice: Optional[str] = None,
    *,
    language: str = "zh",
) -> None:
    """Block external LLM calls when the sidebar AI toggle is off.

    Parameters
    ----------
    llm_choice:
        Identifier of the LLM client about to be instantiated. If it
        names an offline / mock client, the gate is bypassed.
    language:
        ``"zh"`` or ``"en"`` — controls the error message. Pages that
        already know the current language should pass it explicitly;
        otherwise we fall back to whatever is stored in
        ``st.session_state['language']``.

    Raises
    ------
    AIOptInError
        When ``llm_choice`` requires an external call and the sidebar
        ``llm_enabled`` flag is False.
    """
    if language:
        resolved_language = language
    else:
        import streamlit as st

        resolved_language = st.session_state.get("language", "zh")
    check_external_llm_opt_in(
        llm_choice,
        ai_enabled=is_ai_enabled(),
        language=resolved_language,
    )


__all__ = [
    "AIOptInError",
    "check_external_llm_opt_in",
    "enforce_external_llm_opt_in",
    "is_ai_enabled",
    "is_offline_llm_choice",
]
